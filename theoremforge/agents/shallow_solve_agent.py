from loguru import logger
from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from openai import AsyncOpenAI
from theoremforge.utils import extract_lean_code
from theoremforge.prompt_manager import prompt_manager


class ShallowSolveAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
        max_rounds: int = 5,
    ):
        super().__init__(
            agent_name="shallow_solve_agent",
            context=context,
        )
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.sampling_params = sampling_params
        self.max_rounds = max_rounds

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(f"Shallow Solve Agent: Start to process state {state.id}")

                # Check black_list with lock
                async with self.context.black_list_lock:
                    is_blacklisted = state.id in self.context.black_list or state.parent_id in self.context.black_list

                if is_blacklisted:
                    logger.debug(f"Shallow Solve Agent: State {state.id} is blacklisted")
                    await self.add_state_request("finish_agent", state)
                    continue

                if not state.formal_statement:
                    logger.error(f"Shallow Solve Agent: No formal statement found for state {state.id}")
                    await self.add_state_request("finish_agent", state)
                    continue

                # Initialize chat history if not present
                if "shallow_solve_history" not in state.metadata:
                    state.metadata["shallow_solve_history"] = []
                    current_round = 0
                else:
                    current_round = len(state.metadata["shallow_solve_history"])

                if current_round >= self.max_rounds:
                    logger.info(
                        f"Shallow Solve Agent: Max rounds ({self.max_rounds}) reached for state {state.id}"
                    )
                    await self.add_state_request("prover_agent", state)
                    continue

                # Build messages for chat history
                messages = []

                if current_round == 0:
                    # First round - generate initial proof
                    useful_theorems = state.metadata.get("useful_theorems", "")
                    initial_prompt = prompt_manager.shallow_solve_initial(
                        state.formal_statement,
                        state.informal_proof,
                        useful_theorems,
                    )
                    messages.append({"role": "user", "content": initial_prompt})
                    logger.info(f"Shallow Solve Agent: Round 0 (initial) for state {state.id}")
                else:
                    # Subsequent rounds - refine based on errors
                    # Add all previous conversation history
                    for hist in state.metadata["shallow_solve_history"]:
                        messages.append({"role": "user", "content": hist["user_message"]})
                        messages.append({"role": "assistant", "content": hist["assistant_message"]})

                    # Add refinement prompt with latest error
                    refinement_prompt = prompt_manager.shallow_solve_refinement(
                        failed_code=state.metadata["shallow_solve_history"][-1]["code"],
                        error_message=state.metadata["shallow_solve_history"][-1]["error"],
                        useful_theorems=state.metadata.get("useful_theorems", ""),
                    )
                    messages.append({"role": "user", "content": refinement_prompt})
                    logger.info(f"Shallow Solve Agent: Round {current_round} (refinement) for state {state.id}")

                # Call LLM with chat history
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **self.sampling_params,
                )

                output = response.choices[0].message.content
                code = extract_lean_code(output)

                if not code:
                    logger.error(
                        f"Shallow Solve Agent: Failed to extract code for state {state.id} at round {current_round}"
                    )
                    # Record failure in history
                    state.metadata["shallow_solve_history"].append({
                        "round": current_round,
                        "user_message": messages[-1]["content"],
                        "assistant_message": output,
                        "code": None,
                        "error": "Failed to extract code",
                        "valid": False,
                    })
                    # Try again in next round
                    await self.task_queue.put(state)
                    continue

                # Verify the proof
                valid, lean_messages, error_str = await self.context.verifier.verify(
                    code, False
                )

                # Record this round in history
                history_entry = {
                    "round": current_round,
                    "user_message": messages[-1]["content"] if current_round == 0 else refinement_prompt,
                    "assistant_message": output,
                    "code": code,
                    "error": error_str,
                    "valid": valid,
                }
                state.metadata["shallow_solve_history"].append(history_entry)

                # Save to database
                await self.db.shallowsolvetrace.create(
                    data={
                        "round": current_round,
                        "prompt": messages[-1]["content"],
                        "output": output,
                        "code": code,
                        "valid": valid,
                        "errorMessage": error_str,
                        "stateId": state.id,
                        "totalRounds": len(state.metadata["shallow_solve_history"]),
                    }
                )

                if valid:
                    logger.info(
                        f"Shallow Solve Agent: Successfully proved state {state.id} at round {current_round}"
                    )
                    state.formal_proof = code
                    state.success = True
                    await self.add_state_request("finish_agent", state)
                else:
                    logger.info(
                        f"Shallow Solve Agent: Round {current_round} failed for state {state.id}, will retry"
                    )
                    # Put back in queue for next round
                    await self.task_queue.put(state)

            except Exception as e:
                logger.error(f"Shallow Solve Agent: Error processing state: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if 'state' in locals():
                        await self.add_state_request("finish_agent", state)
                except Exception:
                    pass

