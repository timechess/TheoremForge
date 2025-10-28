from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from theoremforge.prompt_manager import prompt_manager
from openai import AsyncOpenAI
import re
from loguru import logger


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

                # Check black_list with lock
                async with self.context.black_list_lock:
                    is_blacklisted = state.id in self.context.black_list

                if is_blacklisted:
                    logger.debug(f"Informal Proof Agent: State {state.id} is blacklisted")
                    await self.add_state_request("finish_agent", state)
                    continue

                prompt = prompt_manager.informal_proof_generation(
                    state.formal_statement,
                    state.metadata["useful_theorems"],
                )
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    **self.sampling_params,
                )
                informal_proofs = [
                    self._extract_informal_proof(choice.message.content)
                    for choice in response.choices
                ]

                for i, informal_proof in enumerate(informal_proofs):
                    if not informal_proof:
                        continue
                    await self.db.informalprooftrace.create(
                        data={
                            "prompt": prompt,
                            "output": response.choices[i].message.content,
                            "formalStatement": state.formal_statement,
                            "informalProof": informal_proof,
                            "usefulTheorems": state.metadata["useful_theorems"],
                            "stateId": state.id,
                        }
                    )
                    state.informal_proof = informal_proof
                    logger.debug(f"Informal Proof Agent: Routing state {state.id} to proof_sketch_agent")
                    await self.add_state_request("proof_sketch_agent", state)
                    break
                if not any(informal_proofs):
                    logger.info(
                        f"Informal Proof Agent: Failed to generate informal proof for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)

            except Exception as e:
                logger.error(f"Informal Proof Agent: Error processing state: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if 'state' in locals():
                        await self.add_state_request("finish_agent", state)
                except Exception:
                    pass
