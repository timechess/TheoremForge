import re
from typing import Optional, List


def extract_lean_code(text: str) -> str:
    pattern = r"```lean4\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    pattern = r"```lean4\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    pattern = r"```lean\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    return None


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
