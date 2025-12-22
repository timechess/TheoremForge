import re
import asyncio
from typing import Optional, List
import json
import logging
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError
from openai.types.chat.chat_completion import ChatCompletion
from google.genai import Client, types
from google.genai.errors import APIError
from theoremforge.state import TheoremForgeState, TheoremForgeContext
from lean_explore.local.service import APISearchResultItem

logger = logging.getLogger(__name__)


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


def format_search_results(result: APISearchResultItem) -> str:
    return f"name: {result.primary_declaration.lean_name}\nlean_code: {result.display_statement_text}\ninformal_description: {result.informal_description}\ndocstring: {result.docstring}"


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


def convert_openai_messages_to_genai(messages):
    genai_messages = []
    system_instruction = None

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            # 系统提示放在 system_instruction，不放在 contents 里
            system_instruction = content
        else:
            genai_messages.append(
                {
                    "role": "user" if role == "user" else "model",
                    "parts": [{"text": content}],
                }
            )

    return system_instruction, genai_messages


async def call_llm(
    state: TheoremForgeState,
    context: TheoremForgeContext,
    client: AsyncOpenAI | Client,
    model_name: str,
    prompt: str | List[dict],
    sampling_params: dict,
    agent_name: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> List[str]:
    """
    Call LLM with automatic retry on network errors.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay in seconds between retries (default: 60.0)
    """
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            if isinstance(client, AsyncOpenAI):
                response: ChatCompletion = await client.chat.completions.create(
                    model=model_name,
                    messages=prompt,
                    **sampling_params,
                )
                await context.db.create_trace(
                    trace={
                        "state_id": state.id,
                        "agent_name": agent_name,
                        "prompt": json.dumps(prompt, ensure_ascii=False),
                        "response": [
                            choice.message.content for choice in response.choices
                        ],
                        "input_token": response.usage.prompt_tokens,
                        "output_token": response.usage.completion_tokens,
                    }
                )
                return [choice.message.content for choice in response.choices]
            elif isinstance(client, Client):
                _, genai_prompt = convert_openai_messages_to_genai(prompt)
                response: types.GenerateContentResponse = (
                    await client.aio.models.generate_content(
                        model=model_name,
                        contents=genai_prompt,
                        config=types.GenerateContentConfig(
                            temperature=1.0,
                            thinking_config=types.ThinkingConfig(
                                thinking_level=sampling_params.get(
                                    "thinking_level", "low"
                                ),
                            ),
                        ),
                    )
                )
                input_token = response.usage_metadata.prompt_token_count or 0
                output_token = (response.usage_metadata.candidates_token_count or 0) + (
                    response.usage_metadata.thoughts_token_count or 0
                )
                await context.db.create_trace(
                    trace={
                        "state_id": state.id,
                        "agent_name": agent_name,
                        "prompt": json.dumps(prompt, ensure_ascii=False),
                        "response": [response.text],
                        "input_token": input_token,
                        "output_token": output_token,
                    }
                )
                return [response.text]
        except (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            APIError,
            asyncio.TimeoutError,
            OSError,
        ) as e:
            last_exception = e
            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = min(base_delay * (2**attempt), max_delay)
                logger.warning(
                    f"[{agent_name}] LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {type(e).__name__}: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"[{agent_name}] LLM call failed after {max_retries + 1} attempts: {type(e).__name__}: {e}"
                )
                raise last_exception


async def call_llm_interruptible(
    state: TheoremForgeState,
    context: TheoremForgeContext,
    client: AsyncOpenAI,
    model_name: str,
    prompt: str | List[dict],
    sampling_params: dict,
    agent_name: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
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
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay in seconds between retries (default: 60.0)

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
        call_llm(
            state,
            context,
            client,
            model_name,
            prompt,
            sampling_params,
            agent_name,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )
    )

    # If there's a cancellation event, wait for either completion or cancellation
    async with context.cancellation_lock:
        cancellation_event = context.cancellation_events.get(state.id)

    if cancellation_event:
        # Create cancellation wait task
        cancel_wait_task = asyncio.create_task(cancellation_event.wait())

        # Wait for either the LLM call or cancellation
        done, pending = await asyncio.wait(
            [llm_task, cancel_wait_task], return_when=asyncio.FIRST_COMPLETED
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
