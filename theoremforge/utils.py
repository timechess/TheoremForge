import re
import asyncio
from typing import Optional, List

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from theoremforge.state import TheoremForgeState, TheoremForgeContext


def extract_lean_code(text: str) -> str:
    # Match ```lean4 followed by content and closing ```
    # The (?:\r?\n) handles different line endings
    # Strip whitespace from captured content
    pattern = r"```lean4(?:\r?\n)(.*?)(?:\r?\n)?```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Fallback: try with ```lean tag
    pattern = r"```lean(?:\r?\n)(.*?)(?:\r?\n)?```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    return None


def remove_comments(text):  # remove comments
    # First remove all /- ... -/ blocks
    text = re.sub(r"/-.*?-/", "", text, flags=re.DOTALL)
    # text = re.sub(r'/- (?!special open -/).*?-/', '', text, flags=re.DOTALL)
    # text = re.sub(r'/-{1,2} (?!special open -/).*?-{1,2}/', '', text, flags=re.DOTALL)
    # Then remove -- comments from each line
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Split on -- and keep only the first part
        cleaned_line = line.split("--", 1)[0]
        if "--" in line:
            cleaned_lines.append(cleaned_line.rstrip())
        else:
            cleaned_lines.append(cleaned_line)
    # Join back together and remove excessive empty lines
    cleaned_text = "\n".join(cleaned_lines)
    # Remove multiple consecutive empty lines
    # cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    return cleaned_text.strip()


def payload_to_string(payload: dict) -> str:
    return f"""full_name: {payload["name"]}
type: {payload["type"]}
informal_name: {payload["informal_name"]}
informal_description: {payload["informal_description"]}
"""


def get_error_str(
    code: str, errors: List[dict], error_thres: Optional[int] = None
) -> str:
    err_str = ""
    code_lines = code.split("\n")
    # token_lengths = [len(line) + 1 for line in code_lines]

    # error_thres = False

    # error_num_thres = 8 if error_thres else error_num_thres
    error_num_thres = 8 if error_thres else len(errors)

    for i, error in enumerate(errors[:error_num_thres]):
        start_line = error["start_pos"]["line"] - 1
        start_col = error["start_pos"]["column"]

        if error["end_pos"] is None:
            end_line = start_line
            end_col = len(code_lines[start_line])
        else:
            end_line = error["end_pos"]["line"] - 1
            end_col = error["end_pos"]["column"]

        # start_char_pos = sum(token_lengths[:start_line]) + start_col
        # end_char_pos = sum(token_lengths[:end_line]) + end_col

        err_str += f"\nError {i + 1}:\n"
        err_str += "\nCorresponding Code:\n```lean4\n"

        error_code = ""
        for ii in range(-4, 0):
            if start_line + ii >= 0:
                error_code += f"{code_lines[start_line + ii]}\n"
        if start_line != end_line:
            error_code += (
                code_lines[start_line][:start_col]
                + "<error>"
                + code_lines[start_line][start_col:]
                + "\n"
            )

            if not error_thres:
                for j in range(start_line + 1, end_line):
                    error_code += f"{code_lines[j]}\n"
            else:
                show_line = 6
                for j in range(start_line + 1, min(end_line, start_line + show_line)):
                    error_code += f"{code_lines[j]}\n"
                if end_line > start_line + show_line:
                    leading_spaces = len(code_lines[j]) - len(code_lines[j].lstrip(" "))
                    error_code += (
                        "\n" + " " * leading_spaces + "... --[Truncated]-- ...\n"
                    )

            error_code += (
                code_lines[end_line][:end_col]
                + "</error>"
                + code_lines[end_line][end_col:]
                + "\n"
            )
        else:
            error_code += (
                code_lines[start_line][:start_col]
                + "<error>"
                + code_lines[start_line][start_col:end_col]
                + "</error>"
                + code_lines[start_line][end_col:]
                + "\n"
            )
        if end_line + 1 < len(code_lines):
            error_code += f"{code_lines[end_line + 1]}\n"

        err_str += error_code
        err_str += "\n```\n"
        err_str += f"\nError Message: {error['data']}\n"

    if len(errors) > error_num_thres:
        err_str += f"\n... [Omitted {len(errors) - error_num_thres} more errors] ...\n"

    return err_str


class CancellationError(Exception):
    """Raised when an operation is cancelled via cancellation event."""
    pass


async def call_llm(
    state: TheoremForgeState,
    client: AsyncOpenAI,
    model_name: str,
    prompt: str | List[dict],
    sampling_params: dict,
    agent_name: str,
) -> List[str]:
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    response: ChatCompletion = await client.chat.completions.create(
        model=model_name,
        messages=prompt,
        **sampling_params,
    )

    if agent_name in state.token_trace:
        state.token_trace[agent_name] = {
            "prompt_tokens": response.usage.prompt_tokens
            + state.token_trace[agent_name]["prompt_tokens"],
            "completion_tokens": response.usage.completion_tokens
            + state.token_trace[agent_name]["completion_tokens"],
            "total_tokens": response.usage.total_tokens
            + state.token_trace[agent_name]["total_tokens"],
        }
    else:
        state.token_trace[agent_name] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return [choice.message.content for choice in response.choices]


async def call_llm_interruptible(
    state: TheoremForgeState,
    context: TheoremForgeContext,
    client: AsyncOpenAI,
    model_name: str,
    prompt: str | List[dict],
    sampling_params: dict,
    agent_name: str,
) -> List[str]:
    """
    Call LLM with support for cancellation via context.cancellation_events.
    
    Args:
        state: The state being processed
        context: TheoremForgeContext with cancellation events
        client: OpenAI client
        model_name: Model to use
        prompt: Prompt or message list
        sampling_params: Sampling parameters
        agent_name: Name of the calling agent
        
    Returns:
        List of generated responses
        
    Raises:
        CancellationError: If the state is cancelled during execution
    """
    # Check if already cancelled before starting
    async with context.cancellation_lock:
        cancellation_event = context.cancellation_events.get(state.id)
        if cancellation_event and cancellation_event.is_set():
            raise CancellationError(f"State {state.id} was cancelled before LLM call")
    
    # Create the LLM call task
    llm_task = asyncio.create_task(
        call_llm(state, client, model_name, prompt, sampling_params, agent_name)
    )
    
    # If there's a cancellation event, wait for either completion or cancellation
    async with context.cancellation_lock:
        cancellation_event = context.cancellation_events.get(state.id)
    
    if cancellation_event:
        # Create cancellation wait task
        cancel_wait_task = asyncio.create_task(cancellation_event.wait())
        
        # Wait for either the LLM call or cancellation
        done, pending = await asyncio.wait(
            [llm_task, cancel_wait_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel all pending tasks to avoid "task destroyed but pending" warnings
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # If cancellation happened first, raise cancellation error
        if cancellation_event.is_set():
            raise CancellationError(f"State {state.id} was cancelled during LLM call")
        
        # Otherwise, return the LLM result (from done tasks)
        for task in done:
            if task == llm_task:
                return task.result()
        
        # Fallback - should not reach here
        return await llm_task
    else:
        # No cancellation event registered, just wait normally
        return await llm_task
