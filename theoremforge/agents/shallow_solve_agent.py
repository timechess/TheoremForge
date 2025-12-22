from loguru import logger
from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext, TheoremForgeState
from openai import AsyncOpenAI
from google.genai import Client
from theoremforge.utils import extract_lean_code, call_llm_interruptible
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
        if model_name.startswith("gemini-"):
            self.client = Client(api_key=api_key, http_options={"base_url": base_url}, vertexai=True)
        else:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1500)
        self.model_name = model_name
        self.sampling_params = sampling_params
        self.max_rounds = max_rounds

    async def _run(self, state: TheoremForgeState):
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
            await self.add_state_request("finish_agent", state)
            return

        if current_round == 0:
            # First round - generate initial proof
            useful_theorems = state.metadata.get("useful_theorems", "")
            initial_prompt = prompt_manager.shallow_solve_initial(
                state.formal_statement,
                state.informal_proof,
                useful_theorems,
            )
            prompt = initial_prompt
            logger.info(f"Shallow Solve Agent: Round 0 (initial) for state {state.id}")
        else:
            # Add refinement prompt with latest error
            refinement_prompt = prompt_manager.shallow_solve_refinement(
                formal_statement=state.formal_statement,
                failed_code=state.metadata["shallow_solve_history"][-1]["code"],
                error_message=state.metadata["shallow_solve_history"][-1]["error"],
                useful_theorems=state.metadata.get("useful_theorems", ""),
            )
            prompt = refinement_prompt
            logger.info(f"Shallow Solve Agent: Round {current_round} (refinement) for state {state.id}")

        # Call LLM with chat history (interruptible)
        response = await call_llm_interruptible(
            state,
            self.context,
            self.client,
            self.model_name,
            prompt,
            self.sampling_params,
            "shallow_solve_agent",
        )

        output = response[0]
        code = extract_lean_code(output)

        if not code:
            logger.error(
                f"Shallow Solve Agent: Failed to extract code for state {state.id} at round {current_round}"
            )
            # Record failure in history
            state.metadata["shallow_solve_history"].append({
                "round": current_round,
                "user_message": prompt,
                "assistant_message": output,
                "code": None,
                "error": "Failed to extract code",
                "valid": False,
            })
            # Try again in next round
            await self.task_queue.put(state)
            return

        # Check for cancellation before verification
        if await self.is_cancelled(state):
            logger.info(
                f"Shallow Solve Agent: State {state.id} cancelled before verification"
            )
            await self.add_state_request("finish_agent", state)
            return

        # Verify the proof
        valid, lean_messages, error_str = await self.context.verifier.verify(
            code, False
        )

        # Record this round in history
        history_entry = {
            "round": current_round,
            "user_message": prompt,
            "assistant_message": output,
            "code": code,
            "error": error_str,
            "valid": valid,
        }
        state.metadata["shallow_solve_history"].append(history_entry)


        if valid:
            logger.info(
                f"Shallow Solve Agent: Successfully proved state {state.id} at round {current_round}"
            )
            state.formal_proof = code
            state.success = True
            await self.add_state_request("finish_agent", state)
            return
        else:
            logger.info(
                f"Shallow Solve Agent: Round {current_round} failed for state {state.id}, will retry"
            )
            await self.task_queue.put(state)

