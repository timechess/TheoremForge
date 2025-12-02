from openai import AsyncOpenAI
from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from theoremforge.utils import extract_lean_code, call_llm_interruptible, CancellationError
from theoremforge.prompt_manager import prompt_manager
from loguru import logger


class ProofSketchAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(
            agent_name="proof_sketch_agent",
            context=context,
        )
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.sampling_params = sampling_params

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(f"Proof Sketch Agent: Processing state {state.id}")


                # Register cancellation event for this state

                await self.register_cancellation_event(state)


                # Check if state should be skipped (blacklisted or cancelled)

                if await self.should_skip_state(state):

                    await self.add_state_request("finish_agent", state)

                    await self.cleanup_cancellation_event(state)

                    continue

                prompt = prompt_manager.proof_sketch_generation(
                    state.formal_statement,
                    state.informal_proof,
                    state.metadata["useful_theorems"],
                )
                response = await call_llm_interruptible(
                    state,
                    self.context,
                    self.client,
                    self.model_name,
                    prompt,
                    self.sampling_params,
                    "proof_sketch_agent",
                )
                proof_sketches = [
                    extract_lean_code(code) for code in response
                ]
                if not any(proof_sketches):
                    logger.info(
                        "Proof Sketch Agent: Proof sketch generation failed. Moving to finished."
                    )
                    await self.add_state_request("finish_agent", state)
                    await self.db.proofsketchtrace.create(
                        data={
                            "prompt": prompt,
                            "output": response[0],
                            "formalStatement": state.formal_statement,
                            "informalProof": state.informal_proof,
                            "usefulTheorems": state.metadata["useful_theorems"],
                            "proofSketch": None,
                            "valid": False,
                            "errorMessage": "No proof sketch generated",
                            "stateId": state.id,
                        }
                    )
                    continue
                routed = False
                last_i = 0
                last_valid = False
                last_error_str = "No valid proof sketch"

                for i, proof_sketch in enumerate(proof_sketches):
                    last_i = i
                    if not proof_sketch:
                        continue
                    valid, messages, error_str = await self.context.verifier.verify(
                        proof_sketch, True
                    )
                    last_valid = valid
                    last_error_str = error_str

                    if valid:
                        logger.info(
                            f"Proof Sketch Agent: Successfully generated proof sketch for state {state.id}"
                        )
                        state.proof_sketch = proof_sketch
                        routed = True
                        await self.add_state_request("subgoal_extraction_agent", state)
                        break
                    else:
                        logger.info(
                            f"Proof Sketch Agent: Failed to generate proof sketch for state {state.id}, routing to sketch_correction"
                        )
                        state.metadata["failed_attempt"] = {
                            "code": proof_sketch,
                            "error_str": error_str,
                        }
                        routed = True
                        await self.add_state_request("sketch_correction_agent", state)
                        break

                # Save trace only if we processed at least one sketch
                if proof_sketches and any(proof_sketches):
                    await self.db.proofsketchtrace.create(
                        data={
                            "prompt": prompt,
                            "output": response[last_i],
                            "formalStatement": state.formal_statement,
                            "informalProof": state.informal_proof,
                            "usefulTheorems": state.metadata.get("useful_theorems", ""),
                            "proofSketch": state.proof_sketch,
                            "valid": last_valid,
                            "errorMessage": last_error_str,
                            "stateId": state.id,
                        }
                    )

                # If no valid sketch was generated and not routed, send to finish
                if not routed:
                    logger.warning(
                        f"Proof Sketch Agent: All sketches were empty for state {state.id}, sending to finish"
                    )
                    await self.add_state_request("finish_agent", state)
            except CancellationError as e:

                # State was cancelled during processing

                logger.info(f"Proof Sketch Agent: {e}")

                if "state" in locals():

                    await self.add_state_request("finish_agent", state)

                    await self.cleanup_cancellation_event(state)

            except Exception as e:
                logger.error(f"Proof Sketch Agent: Error processing state: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                except Exception:
                    pass