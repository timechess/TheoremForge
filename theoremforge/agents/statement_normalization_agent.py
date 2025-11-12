from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from openai import AsyncOpenAI
from loguru import logger
from theoremforge.prompt_manager import prompt_manager
import re


class StatementNormalizationAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(agent_name="statement_normalization_agent", context=context)
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.sampling_params = sampling_params

    def _extract_statement_normalization(self, response: str) -> str:
        pattern = r"<normalized>(.*?)</normalized>"
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1).strip() if match else None

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(
                    f"Statement Normalization Agent: Start to process state {state.id}"
                )

                # Check black_list with lock
                async with self.context.black_list_lock:
                    is_blacklisted = state.id in self.context.black_list or state.parent_id in self.context.black_list

                if is_blacklisted:
                    logger.debug(
                        f"Statement Normalization Agent: State {state.id} is blacklisted, routing to finish"
                    )
                    await self.add_state_request("finish_agent", state)
                    continue

                if not state.informal_statement:
                    logger.error(
                        f"Statement Normalization Agent: No informal statement for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
                    continue

                statement_normalization_prompt = prompt_manager.statement_normalization(
                    state.informal_statement
                )
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": statement_normalization_prompt}
                    ],
                    **self.sampling_params,
                )
                statement_normalization_output = response.choices[0].message.content
                normalized_statement = self._extract_statement_normalization(
                    statement_normalization_output
                )
                if not normalized_statement:
                    logger.error(
                        f"Statement Normalization Agent: Failed to normalize statement {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
                    continue
                logger.info(
                    f"Statement Normalization Agent: Normalized statement {state.id}"
                )
                state.normalized_statement = normalized_statement
                await self.db.statementnormalizationtrace.create(
                    data={
                        "prompt": statement_normalization_prompt,
                        "output": statement_normalization_output,
                        "normalizedStatement": normalized_statement,
                        "stateId": state.id,
                    }
                )
                logger.debug(f"Statement Normalization Agent: Routing state {state.id} to definition_retrieval_agent")
                await self.add_state_request("definition_retrieval_agent", state)

            except Exception as e:
                logger.error(f"Error in StatementNormalizationAgent: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                except Exception:
                    pass