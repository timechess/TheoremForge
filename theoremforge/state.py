from pydantic import BaseModel
from typing import Optional, Literal, List, Union


class BaseTrace(BaseModel):
    step: str
    agent_name: str


class ProverTrace(BaseTrace):
    prompt: str
    output: List[str]
    formal_statement: str
    output_code: List[Optional[str]]


class VerifierTrace(BaseTrace):
    code: str
    messages: List[dict]
    valid: bool
    error_str: str


class SelfCorrectionTrace(BaseTrace):
    prompt: str
    output: str
    output_code: Optional[str] = None


class TheoremRetrieverTrace(BaseTrace):
    query_generation_prompt: str
    query_generation_output: str
    query_results: List[dict]
    theorem_selection_prompt: str
    theorem_selection_output: str
    theorem_selection_results: List[dict]


class InformalProofTrace(BaseTrace):
    prompt: str
    output: List[str]
    formal_statement: str
    informal_proof: List[str]


class ProblemDecompositionTrace(BaseTrace):
    prompt: List[str]
    output: List[List[str]]
    formal_statement: str
    informal_proof: List[Optional[str]]
    proof_sketch: List[Optional[str]]


class SubgoalExtractionTrace(BaseTrace):
    proof_sketch: List[str]
    subgoals: List[List[str]]


class SubgoalSolvingTrace(BaseTrace):
    formal_statements: List[str]
    formal_proofs: Optional[List[str]] = None


class ProofAssemblyTrace(BaseTrace):
    prompt: str
    output: str
    formal_statement: str
    proof_sketch: str
    subgoal_proofs: List[str]
    final_proof: Optional[str] = None


class TheoremForgeState(BaseModel):
    id: str
    informal_statement: Optional[str] = None
    header: Optional[str] = "import Mathlib\n"
    formal_statement: Optional[str] = None
    formal_proof: Optional[str] = None
    proof_sketch: Optional[List[str]] = None
    subgoals: Optional[List[List[str]]] = None
    subgoal_proofs: Optional[List[str]] = None
    successful_id: Optional[int] = None
    stage: Literal[
        "first_attempt",
        "problem_decoposition",
        "subgoal_solving",
        "proof_assembly",
        "finished",
    ] = "first_attempt"
    result: Literal["success", "failure", "not_finished"] = "not_finished"
    trace: List[
        Union[
            ProverTrace,
            VerifierTrace,
            SelfCorrectionTrace,
            TheoremRetrieverTrace,
            InformalProofTrace,
            ProblemDecompositionTrace,
            SubgoalExtractionTrace,
            SubgoalSolvingTrace,
            ProofAssemblyTrace,
        ]
    ] = []
