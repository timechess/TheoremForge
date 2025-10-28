from jinja2 import FileSystemLoader, Environment
from pathlib import Path


class PromptManager:
    def __init__(self, prompt_dir: Path):
        self.env = Environment(loader=FileSystemLoader(prompt_dir))

    def get_prompt(self, prompt_name: str) -> str:
        return self.env.get_template(prompt_name).render()

    def proof_attempt(self, formal_statement: str) -> str:
        return self.env.get_template("proof_attempt.j2").render(
            formal_statement=formal_statement
        )

    def self_correction(self, initial_problem_statement: str, incorrect_code: str, error_message: str) -> str:
        return self.env.get_template("self_correction.j2").render(
            initial_problem_statement=initial_problem_statement,
            incorrect_code=incorrect_code,
            error_message=error_message,
        )

    def search_query_generation(self, formal_statement: str) -> str:
        return self.env.get_template("search_query_generation.j2").render(
            formal_statement=formal_statement,
        )

    def theorem_selection(self, formal_statement: str, theorems: str) -> str:
        return self.env.get_template("theorem_selection.j2").render(
            formal_statement=formal_statement,
            theorems=theorems,
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

    def proof_assembly(
        self, formal_statement: str, proof_sketch: str, subgoal_proofs: str
    ) -> str:
        return self.env.get_template("proof_assembly.j2").render(
            formal_statement=formal_statement,
            proof_sketch=proof_sketch,
            subgoal_proofs=subgoal_proofs,
        )


prompt_manager = PromptManager(Path(__file__).parent / "prompts")
