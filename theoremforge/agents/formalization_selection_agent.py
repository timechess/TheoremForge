from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from loguru import logger
from openai import AsyncOpenAI
import re
from theoremforge.prompt_manager import prompt_manager
from theoremforge.utils import call_llm_interruptible, CancellationError


class FormalizationSelectionAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(agent_name="formalization_selection_agent", context=context)
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.sampling_params = sampling_params

    def _extract_selection_analysis(self, response: str) -> str:
        """Extract the analysis from the response."""
        pattern = r"<analysis>(.*?)</analysis>"
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_selected_index(self, response: str) -> int:
        """Extract the selected formalization index from the response."""
        pattern = r"<selected>(\d+)</selected>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return int(match.group(1).strip())
        return None

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(
                    f"Formalization Selection Agent: Start to process state {state.id}"
                )

                # Register cancellation event for this state
                await self.register_cancellation_event(state)

                # Check if state should be skipped (blacklisted or cancelled)
                if await self.should_skip_state(state):
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue

                if not state.informal_statement:
                    logger.error(
                        f"Formalization Selection Agent: Missing informal statement for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue

                # Get aligned formalizations from metadata
                aligned_formalizations = state.metadata.get("aligned_formalizations", [])

                if not aligned_formalizations:
                    logger.error(
                        f"Formalization Selection Agent: No aligned formalizations found for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue

                # If only one formalization, no need to select
                if len(aligned_formalizations) == 1:
                    logger.info(
                        f"Formalization Selection Agent: Only one formalization available for state {state.id}, skipping selection"
                    )
                    state.formal_statement = aligned_formalizations[0]
                    await self.add_state_request("theorem_retrieval_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue

                logger.info(
                    f"Formalization Selection Agent: Selecting from {len(aligned_formalizations)} aligned formalizations for state {state.id}"
                )

                # Generate prompt for formalization selection
                selection_prompt = prompt_manager.formalization_selection(
                    informal_statement=state.informal_statement,
                    formalizations=aligned_formalizations,
                )

                # Get LLM response
                response = await call_llm_interruptible(
                    state,
                    self.context,
                    self.client,
                    self.model_name,
                    selection_prompt,
                    self.sampling_params,
                    "formalization_selection_agent",
                )
                selection_output = response[0]

                # Extract analysis and selected index
                analysis = self._extract_selection_analysis(selection_output)
                selected_index = self._extract_selected_index(selection_output)

                if not analysis or selected_index is None:
                    logger.warning(
                        f"Formalization Selection Agent: Failed to extract selection for state {state.id}, defaulting to first formalization"
                    )
                    selected_index = 1

                # Validate index (LLM returns 1-based index, convert to 0-based)
                selected_index_zero_based = selected_index - 1
                if selected_index_zero_based < 0 or selected_index_zero_based >= len(
                    aligned_formalizations
                ):
                    logger.warning(
                        f"Formalization Selection Agent: Invalid selection index {selected_index} for state {state.id}, defaulting to first formalization"
                    )
                    selected_index_zero_based = 0

                selected_formalization = aligned_formalizations[selected_index_zero_based]

                logger.info(
                    f"Formalization Selection Agent: Selected formalization {selected_index} (out of {len(aligned_formalizations)}) for state {state.id}"
                )

                # Save trace to database
                await self.db.formalizationselectiontrace.create(
                    data={
                        "prompt": selection_prompt,
                        "output": selection_output,
                        "analysis": analysis,
                        "selectedIndex": selected_index,
                        "selectedFormalization": selected_formalization,
                        "stateId": state.id,
                    }
                )

                # Set the selected formalization as the formal statement
                state.formal_statement = selected_formalization
                state.metadata["selected_formalization"] = selected_formalization
                state.metadata["selected_formalization_index"] = selected_index_zero_based

                # Route to statement refinement agent
                await self.add_state_request("statement_refinement_agent", state)
                
                # Cleanup cancellation event after routing
                await self.cleanup_cancellation_event(state)

            except CancellationError as e:
                # State was cancelled during processing
                logger.info(f"Formalization Selection Agent: {e}")
                if "state" in locals():
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
            except Exception as e:
                logger.error(f"Error in Formalization Selection Agent: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                        await self.cleanup_cancellation_event(state)
                except Exception:
                    pass

