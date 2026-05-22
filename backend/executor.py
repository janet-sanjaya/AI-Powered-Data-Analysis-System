import ast
import traceback
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


ALLOWED_BUILTINS = {
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "sorted": sorted,
    "abs": abs,
    "round": round,
    "range": range,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "float": float,
    "int": int,
    "str": str,
    "print": print,
}

BLOCKED_KEYWORDS = [
    "import ",
    "os.",
    "sys.",
    "subprocess",
    "open(",
    "eval(",
    "exec(",
    "__",
    "requests",
    "socket",
    "pathlib",
    "shutil",
]


class UnsafeCodeError(Exception):
    pass


def _basic_safety_check(code: str) -> None:
    lowered = code.lower()
    for keyword in BLOCKED_KEYWORDS:
        if keyword in lowered:
            raise UnsafeCodeError(f"Blocked unsafe code pattern: {keyword}")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCodeError(f"Syntax error in generated code: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("Import statements are not allowed.")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"open", "eval", "exec", "__import__"}:
                raise UnsafeCodeError(f"Blocked function call: {node.func.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCodeError("Dunder attributes are not allowed.")


def execute_pandas_code(code: str, df: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    """
    Execute LLM-generated Pandas code in a restricted environment.
    The generated code must assign the final output to a variable named `result`.
    """
    _basic_safety_check(code)

    local_env = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "__builtins__": ALLOWED_BUILTINS,
    }

    metadata = {
        "success": False,
        "error": None,
        "traceback": None,
        "result_type": None,
    }

    try:
        exec(code, local_env, local_env)

        if "result" not in local_env:
            raise UnsafeCodeError("Generated code must assign output to a variable named `result`.")

        result = local_env["result"]
        metadata["success"] = True
        metadata["result_type"] = type(result).__name__

        return result, metadata

    except Exception as e:
        metadata["error"] = str(e)
        metadata["traceback"] = traceback.format_exc()
        return None, metadata


def result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Normalize common result types into a DataFrame for display.
    """
    if result is None:
        return pd.DataFrame()

    if isinstance(result, pd.DataFrame):
        return result.reset_index(drop=True)

    if isinstance(result, pd.Series):
        name = result.name if result.name is not None else "result"
        return result.reset_index(name=name)

    if isinstance(result, pd.Index):
        return pd.DataFrame({"result": result.tolist()})

    if isinstance(result, dict):
        return pd.DataFrame([result])

    if isinstance(result, (list, tuple)):
        if len(result) == 0:
            return pd.DataFrame()
        if all(isinstance(x, dict) for x in result):
            return pd.DataFrame(result)
        return pd.DataFrame({"result": list(result)})

    if isinstance(result, (np.generic,)):
        return pd.DataFrame([{"result": result.item()}])

    if isinstance(result, (pd.Timestamp, pd.Timedelta)):
        return pd.DataFrame([{"result": str(result)}])

    if isinstance(result, (int, float, bool, str)):
        return pd.DataFrame([{"result": result}])

    return pd.DataFrame([{"result": str(result)}])


if __name__ == "__main__":
    sample_df = pd.DataFrame({
        "branch": ["A", "A", "B"],
        "sales": [100, 200, 300]
    })

    code = """
result = df.groupby('branch')['sales'].sum().reset_index()
"""

    result, meta = execute_pandas_code(code, sample_df)
    print(meta)
    print(result_to_dataframe(result))