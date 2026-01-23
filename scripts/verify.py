import json
import argparse
from google.genai import Client
from google.genai import types
from google.genai.errors import APIError
from openai import AsyncOpenAI
import re
import os
from dotenv import load_dotenv
from tqdm import tqdm
from httpx import RemoteProtocolError
import asyncio
from typing import Optional, Dict, List, Tuple
from theoremforge.utils import statement_check

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
Please provide your analysis and conclusion. At the end of your response, clearly state your final judgment using one of these exact phrases:
- "Final Judgment: Correct" (if the formalization is correct)
- "Final Judgment: Incorrect" (if the formalization is incorrect)

You may format your response in any way you prefer (plain text, markdown, etc.), but make sure to include the "Final Judgment: Correct" or "Final Judgment: Incorrect" statement at the end.

Input Data:
— Start of Mathematical_Text —
{mathematical_statement}
— End of Mathematical_Text —

— Start of Lean4Code —
{autoformalization_placeholder}
— End of Lean4Code —
"""

MODEL_CONFIGS = [
    ("gemini-3-pro-preview", "genai", "https://api.openai-proxy.org/google", "CLOSEAI_API_KEY"),
    ("gemini-3-flash-preview", "genai", "https://api.openai-proxy.org/google", "CLOSEAI_API_KEY"),
    ("gpt-5.2", "openai", "https://api.openai-proxy.org/v1", "CLOSEAI_API_KEY"),
    ("deepseek-chat", "openai", "https://api.openai-proxy.org/v1", "CLOSEAI_API_KEY"),
    ("deepseek-reasoner", "openai", "https://api.openai-proxy.org/v1", "CLOSEAI_API_KEY"),
    ("claude-sonnet-4-5", "openai", "https://api.openai-proxy.org/v1", "CLOSEAI_API_KEY"),
    ("qwen-max", "openai", "https://dashscope.aliyuncs.com/compatible-mode/v1", "QWEN_API_KEY")
]


def extract_judgment(text: str) -> Optional[str]:
    if not text:
        return None
    
    text_lower = text.lower()
    
    patterns = [
        r"final\s+judgment\s*:\s*(correct|incorrect)",
        r"judgment\s*:\s*(correct|incorrect)",
        r"conclusion\s*:\s*(correct|incorrect)",
        r"verdict\s*:\s*(correct|incorrect)",
        r"result\s*:\s*(correct|incorrect)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            result = match.group(1).strip().lower()
            if result == "correct":
                return "Correct"
            elif result == "incorrect":
                return "Incorrect"
    
    lines = text.split('\n')
    last_lines = '\n'.join(lines[-10:]).lower()

    incorrect_match = re.search(r'\bincorrect\b', last_lines)
    correct_match = re.search(r'\bcorrect\b', last_lines)
    
    if incorrect_match:
        return "Incorrect"
    elif correct_match:
        return "Correct"
    
    return None


def axiom_check(lean_code: str) -> bool:
    """
    Check if the given Lean 4 code defines new axioms.
    """
    if "axiom " in lean_code:
        return True
    return False


async def call_model_async(
    client: AsyncOpenAI | Client,
    client_type: str,
    model_name: str,
    prompt: str,
    max_retries: int = 3,
) -> Tuple[Optional[str], Optional[str]]:
    for attempt in range(max_retries):
        try:
            if client_type == "openai":
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.choices[0].message.content
            elif client_type == "genai":
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=1.0,
                        thinking_config=types.ThinkingConfig(thinking_level="low"),
                    ),
                )
                response_text = response.text
            else:
                raise ValueError(f"Unknown client_type: {client_type}")
            
            judgment = extract_judgment(response_text)
            return response_text, judgment
            
        except (APIError, RemoteProtocolError, Exception) as e:
            if attempt < max_retries - 1:
                print(f"Error (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(1)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return None, None
    
    return None, None


def create_clients(
    model_configs: List[Tuple[str, str, str, str]],
) -> Dict[str, Tuple[AsyncOpenAI | Client, str]]:
    clients = {}
    for model_name, client_type, base_url, api_key_env_var in model_configs:
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            print(f"Warning: API key {api_key_env_var} not found for model {model_name}")
            continue
        
        if client_type == "openai":
            client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1500)
        elif client_type == "genai":
            client = Client(
                api_key=api_key,
                vertexai=True,
                http_options={"base_url": base_url},
            )
        else:
            print(f"Warning: Unknown client_type {client_type} for model {model_name}")
            continue
        
        clients[model_name] = (client, client_type)
    
    return clients


async def verify_with_all_models(
    prompt: str,
    clients: Dict[str, Tuple[AsyncOpenAI | Client, str]],
) -> Dict[str, Dict[str, Optional[str]]]:
    tasks = []
    model_names = []
    
    for model_name, (client, client_type) in clients.items():
        model_names.append(model_name)
        tasks.append(call_model_async(client, client_type, model_name, prompt))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    model_results = {}
    for model_name, result in zip(model_names, results):
        if isinstance(result, Exception):
            print(f"Error for model {model_name}: {result}")
            model_results[model_name] = {"response": None, "judgment": None}
        else:
            response_text, judgment = result
            model_results[model_name] = {
                "response": response_text,
                "judgment": judgment,
            }
    
    return model_results


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    if not MODEL_CONFIGS:
        print("Warning: MODEL_CONFIGS is empty. Please configure models in the script.")
        return

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

    clients = create_clients(MODEL_CONFIGS)
    if not clients:
        print("Error: No valid clients created. Please check MODEL_CONFIGS and API keys.")
        return

    verified_data = []
    
    for data, prompt in tqdm(
        zip(after_statement_check_data, prompts), total=len(after_statement_check_data)
    ):
        model_results = await verify_with_all_models(prompt, clients)
        
        data["model_results"] = {}
        for model_name, result in model_results.items():
            data["model_results"][model_name] = {
                "response": result["response"],
                "judgment": result["judgment"],
            }
        
        verified_data.append(data)
    
    model_names = list(clients.keys())
    for model_name in model_names:
        valid_count = sum(
            1
            for data in verified_data
            if data.get("model_results", {}).get(model_name, {}).get("judgment") == "Correct"
        )
        print(f"Model {model_name} - Valid (Correct): {valid_count}/{len(verified_data)}")
    
    with open(args.output_file, "w") as f:
        for data in verified_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
