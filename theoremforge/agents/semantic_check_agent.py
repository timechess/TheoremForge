from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from loguru import logger
from openai import AsyncOpenAI
import re
import asyncio
from theoremforge.prompt_manager import prompt_manager
from theoremforge.utils import call_llm_interruptible, CancellationError

class SemanticCheckAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(agent_name="semantic_check_agent", context=context)
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.sampling_params = sampling_params

    def _extract_semantic_check_analysis(self, response: str) -> str:
        pattern = r"<analysis>(.*?)</analysis>"
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_semantic_check_conclusion(self, response: str) -> str:
        pattern = r"<verdict>(.*?)</verdict>"
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1).strip() if match else None

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(f"Semantic Check Agent: Start to process state {state.id}")

                # Register cancellation event for this state
                await self.register_cancellation_event(state)

                # Check if state should be skipped (blacklisted or cancelled)
                if await self.should_skip_state(state):
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue

                if not state.informal_statement:
                    logger.error(
                        f"Semantic Check Agent: Missing informal statement for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue

                # Get valid formalizations from metadata (from autoformalization or statement correction)
                valid_formalizations = state.metadata.get("valid_formalizations", [])

                # If no valid formalizations in metadata but we have a formal_statement,
                # use that (for backward compatibility or direct submissions)
                if not valid_formalizations and state.formal_statement:
                    valid_formalizations = [state.formal_statement]

                if not valid_formalizations:
                    logger.error(
                        f"Semantic Check Agent: No valid formalizations to check for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue

                logger.info(
                    f"Semantic Check Agent: Checking {len(valid_formalizations)} valid formalizations concurrently for state {state.id}"
                )

                # Check all valid formalizations for semantic alignment concurrently
                normalized_statement = state.normalized_statement or ""
                useful_definitions = state.metadata.get("useful_definitions", "") or ""

                # Create tasks for concurrent processing
                check_prompts = [
                    prompt_manager.semantic_check(
                        informal_statement=state.informal_statement,
                        formal_statement=formal_statement,
                        normalized_statement=normalized_statement,
                        useful_definitions=useful_definitions,
                    )
                    for formal_statement in valid_formalizations
                ]
                check_tasks = [
                    call_llm_interruptible(
                        state,
                        self.context,
                        self.client,
                        self.model_name,
                        prompt,
                        self.sampling_params,
                        "semantic_check_agent",
                    )
                    for prompt in check_prompts
                ]
                responses = await asyncio.gather(*check_tasks)

                # Process results and save to database
                aligned_formalizations = []
                for i, response in enumerate(responses):
                    semantic_check_output = response[0]
                    semantic_check_analysis = self._extract_semantic_check_analysis(semantic_check_output)
                    semantic_check_conclusion = self._extract_semantic_check_conclusion(semantic_check_output)
                    if not semantic_check_analysis or not semantic_check_conclusion:
                        logger.warning(
                            f"Semantic Check Agent: Failed to extract analysis or conclusion for one formalization of state {state.id}"
                        )
                        continue

                    # Save to database
                    await self.db.semanticchecktrace.create(
                        data={
                            "prompt": check_prompts[i],
                            "output": semantic_check_output,
                            "analysis": semantic_check_analysis,
                            "conclusion": semantic_check_conclusion,
                            "stateId": state.id,
                        }
                    )

                    if semantic_check_conclusion == "ALIGNED":
                        logger.info(
                            f"Semantic Check Agent: Found semantically aligned formalization for state {state.id}"
                        )
                        aligned_formalizations.append(valid_formalizations[i])

                if aligned_formalizations:
                    logger.info(
                        f"Semantic Check Agent: Found {len(aligned_formalizations)} semantically aligned formalizations for state {state.id}, routing to formalization_selection_agent"
                    )
                    # Store all aligned formalizations in metadata
                    state.metadata["aligned_formalizations"] = aligned_formalizations
                    await self.add_state_request("formalization_selection_agent", state)
                else:
                    logger.info(
                        f"Semantic Check Agent: No semantically aligned formalization found for state {state.id}, routing to finish_agent"
                    )
                    await self.add_state_request("finish_agent", state)
                
                # Cleanup cancellation event after routing
                await self.cleanup_cancellation_event(state)

            except CancellationError as e:
                # State was cancelled during processing
                logger.info(f"Semantic Check Agent: {e}")
                if "state" in locals():
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
            except Exception as e:
                logger.error(f"Error in Semantic Check Agent: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                        await self.cleanup_cancellation_event(state)
                except Exception:
                    pass
