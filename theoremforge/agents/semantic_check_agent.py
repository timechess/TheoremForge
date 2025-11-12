from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from loguru import logger
from openai import AsyncOpenAI
import re
import asyncio
from theoremforge.prompt_manager import prompt_manager


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

    async def _check_single_formalization(
        self, state_id: str, informal_statement: str, formal_statement: str, 
        normalized_statement: str, useful_definitions: str
    ) -> tuple[str, str, str, str, str]:
        """
        Check a single formalization for semantic alignment.
        
        Returns:
            Tuple of (formal_statement, prompt, output, analysis, conclusion)
        """
        try:
            semantic_check_prompt = prompt_manager.semantic_check(
                informal_statement=informal_statement,
                formal_statement=formal_statement,
                normalized_statement=normalized_statement,
                useful_definitions=useful_definitions,
            )
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": semantic_check_prompt}],
                **self.sampling_params,
            )
            
            semantic_check_output = response.choices[0].message.content
            semantic_check_analysis = self._extract_semantic_check_analysis(
                semantic_check_output
            )
            semantic_check_conclusion = self._extract_semantic_check_conclusion(
                semantic_check_output
            )
            
            return (
                formal_statement, 
                semantic_check_prompt, 
                semantic_check_output, 
                semantic_check_analysis, 
                semantic_check_conclusion
            )
            
        except Exception as e:
            logger.error(
                f"Semantic Check Agent: Error checking formalization for state {state_id}: {e}"
            )
            return (formal_statement, "", "", None, None)

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(f"Semantic Check Agent: Start to process state {state.id}")

                # Check black_list with lock
                async with self.context.black_list_lock:
                    is_blacklisted = state.id in self.context.black_list or state.parent_id in self.context.black_list

                if is_blacklisted:
                    logger.debug(
                        f"Semantic Check Agent: State {state.id} is blacklisted, routing to finish"
                    )
                    await self.add_state_request("finish_agent", state)
                    continue

                if not state.informal_statement:
                    logger.error(
                        f"Semantic Check Agent: Missing informal statement for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
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
                    continue

                logger.info(
                    f"Semantic Check Agent: Checking {len(valid_formalizations)} valid formalizations concurrently for state {state.id}"
                )

                # Check all valid formalizations for semantic alignment concurrently
                normalized_statement = state.normalized_statement or ""
                useful_definitions = state.metadata.get("useful_definitions", "") or ""
                
                # Create tasks for concurrent processing
                check_tasks = [
                    self._check_single_formalization(
                        state.id,
                        state.informal_statement,
                        formal_statement,
                        normalized_statement,
                        useful_definitions
                    )
                    for formal_statement in valid_formalizations
                ]
                
                # Run all checks concurrently
                check_results = await asyncio.gather(*check_tasks)
                
                # Process results and save to database
                aligned_formalizations = []
                for (
                    formal_statement, 
                    semantic_check_prompt, 
                    semantic_check_output, 
                    semantic_check_analysis, 
                    semantic_check_conclusion
                ) in check_results:
                    
                    if not semantic_check_analysis or not semantic_check_conclusion:
                        logger.warning(
                            f"Semantic Check Agent: Failed to extract analysis or conclusion for one formalization of state {state.id}"
                        )
                        continue

                    # Save to database
                    await self.db.semanticchecktrace.create(
                        data={
                            "prompt": semantic_check_prompt,
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
                        aligned_formalizations.append(formal_statement)

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

            except Exception as e:
                logger.error(f"Error in Semantic Check Agent: {e}")
                import traceback

                traceback.print_exc()
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                except Exception:
                    pass
