import json
import argparse
from google.genai import Client
from google.genai import types
from google.genai.errors import APIError
import re
import os
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError
from httpx import RemoteProtocolError
import time
from theoremforge.utils import remove_comments

load_dotenv()

prompt_template = """Role: Lean & Formal Verification Expert

Input:

- Mathematical_Text: A math problem and its answer (no proof).
- Lean4Code: A Lean 4 theorem statement formalizing the problem. Proof is intentionally omitted (e.g., sorry).

Goal:
Determine if the Lean theorem statement is an exact and faithful formalization of the mathematical problem.  
**Do not evaluate or consider the answer or the proof. Your sole task is to verify the correctness of the formalization.**

Evaluation Stages (All required):

1. Math Assertion Analysis  
   Identify all structurally and semantically relevant components of the mathematical problem, including variables, types, quantifiers, constraints, logic structure, conclusion, and so on. The analysis should be based on the actual content of the text.

2. Lean Statement Analysis (ignore proof part)  
   Extract all structurally and semantically relevant components from the Lean statement, including variables, types, conditions, quantifiers, constraints, the final claim, and so on. The analysis should reflect the actual content present in the Lean code.

3. Comparative Verification  
   Check for exact correspondence between the math and Lean statements; you may refer to aspects like:
   - Semantic alignment, logic structure, and quantifier correctness.
   - Preservation of constraints and boundary assumptions.
   - Accurate typing and use of variables.
   - Syntactic validity and proper Lean usage (free from errors).
   - Use of symbols and constructs without semantic drift.
   - No missing elements, no unjustified additions, and no automatic corrections or completions.

4. Final Judgement  
   Based solely on the above analysis, judge whether the Lean statement is a correct and exact formalization of the mathematical problem.

5. Accuracy Confirmation  
   If correct: clearly confirm why all elements match.  
   If incorrect: list all mismatches and explain how each one affects correctness.

Note: While the analysis may be broad and open to interpreting all relevant features, the final judgment must be based only on what is explicitly and formally expressed in the Lean statement.  
**Do not consider or assess any part of the proof. Your judgment should be entirely about the accuracy of the statement formalization.**

Output Format:
Return exactly one JSON object:

```json
{{
  "reasons": "Your detailed CoT analysis:
1. Math Assertion Analysis: [...]
2. Lean Statement Analysis (Proof Ignored): [...]
3. Comparative Verification: [...]
4. Conclusion: [...]
5. Accuracy Confirmation: [match confirmation or list of discrepancies...]",
  "is_assistant_correct": "[Correct/Incorrect]"
}}
```

Input Data:
— Start of Mathematical_Text —
{mathematical_statement}
— End of Mathematical_Text —

— Start of Lean4Code —
{autoformalization_placeholder}
— End of Lean4Code —
"""


class VerifiedData(BaseModel):
    reasons: str = Field(description="The reasons for the verification")
    is_assistant_correct: str = Field(
        description="Whether the assistant is correct [Correct/Incorrect]"
    )


def axiom_check(lean_code: str) -> bool:
    """
    Check if the given Lean 4 code defines new axioms.
    """
    if "axiom " in lean_code:
        return True
    return False


def statement_check(statement: str, proof: str) -> bool:
    """
    Check if the given statement is in the proof.

    Example:
    statement: "def xxx ... \n theorem ex (h: ...) ... := by sorry"
    Check if "theorem ex (h: ...) ..." in proof (do not include := by sorry)
    """
    # Find theorem/lemma declarations followed by := by sorry or := sorry
    # Pattern: capture (theorem|lemma name ...) before := by sorry or := sorry
    statement = remove_comments(statement)
    pattern = r"((?:theorem|lemma)\s+\w+.*?)\s*:=\s*(?:by\s+)?sorry"

    matches = re.findall(pattern, statement, re.DOTALL)

    if not matches:
        return True  # No theorem/lemma declarations to check

    for match in matches:
        # Normalize whitespace in both declaration and proof for comparison
        decl_normalized = " ".join(match.strip().split())
        proof_normalized = " ".join(proof.split())

        if decl_normalized not in proof_normalized:
            return False

    return True


def extract_json(text: str) -> dict:
    """
    Extract the JSON object from the given text.
    """
    pattern = r"```json(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text
    return json.loads(json_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = [json.loads(line) for line in f]

    success_data = [data for data in data if data["success"]]
    print(f"Success data: {len(success_data)}")
    after_axiom_check_data = [
        data for data in success_data if not axiom_check(data["formal_proof"])
    ]
    print(f"After axiom check: {len(after_axiom_check_data)}")
    after_statement_check_data = [
        data
        for data in after_axiom_check_data
        if statement_check(data["formal_statement"], data["formal_proof"])
    ]
    print(f"After statement check: {len(after_statement_check_data)}")

    prompts = [
        prompt_template.format(
            mathematical_statement=data["informal_statement"],
            autoformalization_placeholder=data["formal_statement"],
        )
        for data in after_statement_check_data
    ]
    client = Client(
        api_key=os.getenv("CLOSEAI_API_KEY"),
        vertexai=True,
        http_options={"base_url": "https://api.openai-proxy.org/google"},
    )

    verified_data = []
    max_retries = 3
    for data, prompt in tqdm(
        zip(after_statement_check_data, prompts), total=len(after_statement_check_data)
    ):
        for _ in range(max_retries):
            response = client.models.generate_content(
                model="gemini-3-pro-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="high"),
                    temperature=1.0,
                    response_schema=VerifiedData.model_json_schema(),
                ),
            )
            try:
                judge = VerifiedData.model_validate_json(response.text)
                data["analysis"] = judge.reasons
                data["is_assistant_correct"] = judge.is_assistant_correct
                verified_data.append(data)
                break
            except (ValidationError, APIError, RemoteProtocolError) as e:
                print(f"Error: {e}")
                print("Retrying...")
                time.sleep(1)
    valid_data = [
        data for data in verified_data if data["is_assistant_correct"] == "Correct"
    ]
    print(f"Valid data: {len(valid_data)}")
    with open(args.output_file, "w") as f:
        for data in verified_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
