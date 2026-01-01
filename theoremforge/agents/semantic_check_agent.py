from theoremforge.agents.base_agent import BaseAgent
from theoremforge.lean_server.server import erase_header
from theoremforge.state import TheoremForgeContext, TheoremForgeState
from loguru import logger
from openai import AsyncOpenAI
from google.genai import Client
import re
import asyncio
from theoremforge.prompt_manager import prompt_manager
from theoremforge.utils import call_llm_interruptible, extract_lean_code
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
        if model_name.startswith("gemini-"):
            self.client = Client(api_key=api_key, http_options={"base_url": base_url}, vertexai=True)
        else:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1500)
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

    def _extract_fixed_formal_statement(self, response: str) -> str:
        pattern = r"<fixed_formal_statement>(.*?)</fixed_formal_statement>"
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1).strip() if match else None

    async def _run(self, state: TheoremForgeState):
        valid_formalizations = state.metadata.get("valid_formalizations", [])
        if not valid_formalizations:
            logger.error(
                f"Semantic Check Agent: No valid formalizations to check for state {state.id}"
            )
            await self.add_state_request("finish_agent", state)
            return

        # Get valid formalizations from metadata (from autoformalization or statement correction)
        valid_formalizations = state.metadata.get("valid_formalizations", [])

        logger.debug(
            f"Semantic Check Agent: Checking {len(valid_formalizations)} valid formalizations concurrently for state {state.id}"
        )

        # Check all valid formalizations for semantic alignment concurrently
        normalized_statement = state.normalized_statement
        useful_definitions = state.metadata.get("useful_definitions", "")

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
            semantic_check_analysis = self._extract_semantic_check_analysis(
                semantic_check_output
            )
            semantic_check_conclusion = self._extract_semantic_check_conclusion(
                semantic_check_output
            )
            fixed_formal_statement = self._extract_fixed_formal_statement(
                semantic_check_output
            )
            if fixed_formal_statement:
                fixed_formal_statement = erase_header(extract_lean_code(fixed_formal_statement))
            if not semantic_check_analysis or not semantic_check_conclusion:
                logger.warning(
                    f"Semantic Check Agent: Failed to extract analysis or conclusion for one formalization of state {state.id}"
                )
                continue

            if semantic_check_conclusion == "ALIGNED":
                logger.debug(
                    f"Semantic Check Agent: Found semantically aligned formalization for state {state.id}"
                )
                aligned_formalizations.append(valid_formalizations[i])

            if semantic_check_conclusion == "NOT_ALIGNED":
                if not fixed_formal_statement:
                    logger.warning(
                        f"Semantic Check Agent: Failed to extract fixed formalization for state {state.id}"
                    )
                    continue    
                valid, messages, error_str = await self.context.verifier.verify(
                    fixed_formal_statement, True
                )
                if valid:
                    aligned_formalizations.append(fixed_formal_statement)
                else:
                    logger.warning(
                        f"Semantic Check Agent: Failed to verify fixed formalization for state {state.id}"
                    )
                    continue

        if aligned_formalizations:
            logger.debug(
                f"Semantic Check Agent: Found {len(aligned_formalizations)} semantically aligned formalizations for state {state.id}, routing to formalization_selection_agent"
            )
            # Store all aligned formalizations in metadata
            state.metadata["aligned_formalizations"] = aligned_formalizations
            await self.add_state_request("formalization_selection_agent", state)
        else:
            logger.debug(
                f"Semantic Check Agent: No semantically aligned formalization found for state {state.id}, routing to finish_agent"
            )
            await self.add_state_request("finish_agent", state)
