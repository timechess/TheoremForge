from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from theoremforge.prompt_manager import prompt_manager
from openai import AsyncOpenAI
import re
from loguru import logger
from theoremforge.utils import call_llm_interruptible, CancellationError


class InformalProofAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(
            agent_name="informal_proof_agent",
            context=context,
        )
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.sampling_params = sampling_params

    def _extract_informal_proof(self, text: str) -> str | None:
        pattern = r"<informal_proof>(.*?)</informal_proof>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(f"Informal Proof Agent: Start to process state {state.id}")

                # Register cancellation event for this state
                await self.register_cancellation_event(state)

                # Check if state should be skipped (blacklisted or cancelled)
                if await self.should_skip_state(state):
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue

                prompt = prompt_manager.informal_proof_generation(
                    state.formal_statement,
                    state.metadata["useful_theorems"],
                )
                response = await call_llm_interruptible(
                    state,
                    self.context,
                    self.client,
                    self.model_name,
                    prompt,
                    self.sampling_params,
                    "informal_proof_agent",
                )
                informal_proofs = [
                    self._extract_informal_proof(proof)
                    for proof in response
                ]

                for i, informal_proof in enumerate(informal_proofs):
                    if not informal_proof:
                        await self.db.informalprooftrace.create(
                            data={
                                "prompt": prompt,
                                "output": response[i],
                                "formalStatement": state.formal_statement,
                                "informalProof": None,
                                "usefulTheorems": state.metadata["useful_theorems"],
                                "stateId": state.id,
                            }
                        )
                        continue
                    await self.db.informalprooftrace.create(
                        data={
                            "prompt": prompt,
                            "output": response[i],
                            "formalStatement": state.formal_statement,
                            "informalProof": informal_proof,
                            "usefulTheorems": state.metadata["useful_theorems"],
                            "stateId": state.id,
                        }
                    )
                    state.informal_proof = informal_proof
                    logger.debug(
                        f"Informal Proof Agent: Routing state {state.id} to prover_agent"
                    )
                    await self.add_state_request("prover_agent", state)
                    break
                if not any(informal_proofs):
                    state.informal_proof = "No informal proof"
                    logger.info(
                        f"Informal Proof Agent: Failed to generate informal proof for state {state.id}"
                    )
                    await self.add_state_request("prover_agent", state)

                # Cleanup cancellation event after routing
                await self.cleanup_cancellation_event(state)

            except CancellationError as e:
                # State was cancelled during processing
                logger.info(f"Informal Proof Agent: {e}")
                if "state" in locals():
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
            except Exception as e:
                logger.error(f"Informal Proof Agent: Error processing state: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                        await self.cleanup_cancellation_event(state)
                except Exception:
                    pass
