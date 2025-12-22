from jinja2 import FileSystemLoader, Environment
from pathlib import Path


class PromptManager:
    def __init__(self, prompt_dir: Path):
        self.env = Environment(loader=FileSystemLoader(prompt_dir))

    def get_prompt(self, prompt_name: str) -> str:
        return self.env.get_template(prompt_name).render()

    def proof_attempt(self, formal_statement: str) -> str:
        return self.env.get_template("proof_attempt.j2").render(
            formal_statement=formal_statement,
        )

    def proof_correction(self, lean_code: str, error_message: str) -> str:
        return self.env.get_template("proof_correction.j2").render(
            lean_code=lean_code,
            error_message=error_message,
        )

    def sketch_correction(
        self, formal_statement: str, proof_sketch: str, error_message: str
    ) -> str:
        return self.env.get_template("sketch_correction.j2").render(
            formal_statement=formal_statement,
            proof_sketch=proof_sketch,
            error_message=error_message,
        )

    def theorem_query_generation(self, formal_statement: str) -> str:
        return self.env.get_template("theorem_query_generation.j2").render(
            formal_statement=formal_statement,
        )

    def definition_query_generation(self, informal_statement: str) -> str:
        return self.env.get_template("definition_query_generation.j2").render(
            informal_statement=informal_statement,
        )

    def theorem_selection(self, formal_statement: str, theorems: str) -> str:
        return self.env.get_template("theorem_selection.j2").render(
            formal_statement=formal_statement,
            theorems=theorems,
        )

    def definition_selection(self, informal_statement: str, definitions: str) -> str:
        return self.env.get_template("definition_selection.j2").render(
            informal_statement=informal_statement,
            definitions=definitions,
        )

    def statement_normalization(self, informal_statement: str) -> str:
        return self.env.get_template("statement_normalization.j2").render(
            informal_statement=informal_statement,
        )

    def informal_proof_generation(
        self, formal_statement: str, useful_theorems: str
    ) -> str:
        return self.env.get_template("informal_proof_generation.j2").render(
            formal_statement=formal_statement,
            useful_theorems=useful_theorems,
        )

    def proof_sketch_generation(
        self, formal_statement: str, informal_proof: str, useful_theorems: str
    ) -> str:
        return self.env.get_template("proof_sketch_generation.j2").render(
            formal_statement=formal_statement,
            informal_proof=informal_proof,
            useful_theorems=useful_theorems,
        )

    def shallow_solve_initial(
        self, formal_statement: str, informal_proof: str, useful_theorems: str
    ) -> str:
        return self.env.get_template("shallow_solve_initial.j2").render(
            formal_statement=formal_statement,
            informal_proof=informal_proof,
            useful_theorems=useful_theorems,
        )

    def shallow_solve_refinement(
        self,
        formal_statement: str,
        failed_code: str,
        error_message: str,
        useful_theorems: str,
    ) -> str:
        return self.env.get_template("shallow_solve_refinement.j2").render(
            formal_statement=formal_statement,
            failed_code=failed_code,
            error_message=error_message,
            useful_theorems=useful_theorems,
        )

    def correctness_check(self, formal_statement: str) -> str:
        return self.env.get_template("correctness_check.j2").render(
            formal_statement=formal_statement,
        )

    def proof_assembly(
        self, formal_statement: str, proof_sketch: str, subgoal_proofs: str
    ) -> str:
        return self.env.get_template("proof_assembly.j2").render(
            formal_statement=formal_statement,
            proof_sketch=proof_sketch,
            subgoal_proofs=subgoal_proofs,
        )

    def autoformalization(
        self, informal_statement: str, retrieved_definitions: str = ""
    ) -> str:
        return self.env.get_template("autoformalization.j2").render(
            informal_statement=informal_statement,
            retrieved_definitions=retrieved_definitions,
        )

    def semantic_check(
        self,
        informal_statement: str,
        formal_statement: str,
        normalized_statement: str = "",
        useful_definitions: str = "",
    ) -> str:
        return self.env.get_template("semantic_check.j2").render(
            informal_statement=informal_statement,
            formal_statement=formal_statement,
            normalized_statement=normalized_statement,
            useful_definitions=useful_definitions,
        )

    def statement_correction(
        self,
        informal_statement: str,
        failed_code: str,
        error_message: str,
        normalized_statement: str = "",
        useful_definitions: str = "",
    ) -> str:
        return self.env.get_template("statement_correction.j2").render(
            informal_statement=informal_statement,
            failed_code=failed_code,
            error_message=error_message,
            normalized_statement=normalized_statement,
            useful_definitions=useful_definitions,
        )

    def statement_refinement(
        self,
        informal_statement: str,
        failed_code: str,
        normalized_statement: str = "",
        useful_definitions: str = "",
    ) -> str:
        return self.env.get_template("statement_refinement.j2").render(
            informal_statement=informal_statement,
            failed_code=failed_code,
            normalized_statement=normalized_statement,
            useful_definitions=useful_definitions,
        )

    def formalization_selection(
        self, informal_statement: str, formalizations: list[str]
    ) -> str:
        return self.env.get_template("formalization_selection.j2").render(
            informal_statement=informal_statement,
            formalizations=enumerate(formalizations),
        )

    def subgoal_extraction(self, proof_sketch: str) -> str:
        return self.env.get_template("subgoal_extraction.j2").render(
            proof_sketch=proof_sketch,
        )


prompt_manager = PromptManager(Path(__file__).parent / "prompts")
