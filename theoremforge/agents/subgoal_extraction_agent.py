from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext, TheoremForgeState
from loguru import logger
from uuid import uuid4
from theoremforge.prompt_manager import prompt_manager
import re
from openai import AsyncOpenAI
from theoremforge.utils import call_llm_interruptible, CancellationError

class SubgoalExtractionAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(agent_name="subgoal_extraction_agent", context=context)
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.sampling_params = sampling_params

    def _extract_subgoals(self, proof_sketch: str) -> list[str]:
        pattern = r"<subgoal>(.*?)</subgoal>"
        matches = re.findall(pattern, proof_sketch, re.DOTALL)
        return [match.strip() for match in matches]

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(
                    f"Subgoal Extraction Agent: Start to process state {state.id}"
                )

                # Register cancellation event for this state
                await self.register_cancellation_event(state)

                # Check if state should be skipped (blacklisted or cancelled)
                if await self.should_skip_state(state):
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue
                prompt = prompt_manager.subgoal_extraction(state.proof_sketch)
                response = await call_llm_interruptible(
                    state,
                    self.context,
                    self.client,
                    self.model_name,
                    prompt,
                    self.sampling_params,
                    "subgoal_extraction_agent",
                )
                output = response[0]
                subgoals = self._extract_subgoals(output)

                subgoal_ids = [str(uuid4()) for _ in subgoals]

                # Save trace to database
                await self.db.subgoalextractiontrace.create(
                    data={
                        "prompt": prompt,
                        "output": output,
                        "subgoals": subgoals,
                        "subgoalIds": subgoal_ids,
                        "numSubgoals": len(subgoals),
                        "stateId": state.id,
                    }
                )

                if not subgoals:
                    logger.info(
                        f"Subgoal Extraction Agent: No subgoals found for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue
                logger.info(
                    f"Subgoal Extraction Agent: Found {len(subgoals)} subgoals for state {state.id}"
                )
                for subgoal_id, subgoal in zip(subgoal_ids, subgoals):
                    new_state = TheoremForgeState(
                        id=subgoal_id,
                        formal_statement=subgoal,
                        parent_id=state.id,
                        siblings=subgoal_ids,
                        depth=state.depth + 1,
                        metadata={
                            "useful_theorems": state.metadata["useful_theorems"],
                        },
                    )
                    await self.add_state_request("informal_proof_agent", new_state)
                state.subgoals = subgoal_ids
                await self.add_state_request("proof_assembly_agent", state)
                
                # Cleanup cancellation event after routing
                await self.cleanup_cancellation_event(state)
            except CancellationError as e:
                # State was cancelled during processing
                logger.info(f"Subgoal Extraction Agent: {e}")
                if "state" in locals():
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
            except Exception as e:
                logger.error(f"Subgoal Extraction Agent: Error processing state: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                        await self.cleanup_cancellation_event(state)
                except Exception:
                    pass
