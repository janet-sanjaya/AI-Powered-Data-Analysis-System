import ast
import json
import traceback
from pathlib import Path

import pandas as pd
import numpy as np


WORKDIR = Path("/workspace")
CODE_PATH = WORKDIR / "code.py"
INPUT_CSV_PATH = WORKDIR / "input.csv"
OUTPUT_JSON_PATH = WORKDIR / "output.json"


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


class UnsafeCodeError(Exception):
    pass


def basic_safety_check(code: str) -> None:
    lowered = code.lower()

    blocked_patterns = [
        "import ",
        "os.",
        "sys.",
        "subprocess",
        "socket",
        "requests",
        "http://",
        "https://",
        "open(",
        "eval(",
        "exec(",
        "__",
        "pathlib",
        "shutil",
        "pickle",
    ]

    for pattern in blocked_patterns:
        if pattern in lowered:
            raise UnsafeCodeError(f"Blocked unsafe pattern: {pattern}")

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


def _normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "branch" in df.columns:
        s = df["branch"].astype("string").str.strip()
        s = s.fillna("Unknown")
        s = s.replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
        s = s.str.upper()
        s = s.replace({"UNKNOWN": "Unknown"})
        df["branch"] = s

    if "region" in df.columns:
        s = df["region"].astype("string").str.strip()
        s = s.fillna("Unknown")
        s = s.replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
        s = s.str.title()
        df["region"] = s

    if "category" in df.columns:
        s = df["category"].astype("string").str.strip()
        s = s.fillna("Unknown")
        s = s.replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
        s = s.str.title()
        df["category"] = s

    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the dataframe usable for generated code.
    """
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["sales", "profit", "quantity", "orders"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _normalize_text_columns(df)

    return df


def result_to_jsonable(result):
    if result is None:
        return None

    if isinstance(result, pd.DataFrame):
        return result.to_dict(orient="records")

    if isinstance(result, pd.Series):
        return result.reset_index().to_dict(orient="records")

    if isinstance(result, (list, tuple)):
        return list(result)

    if isinstance(result, dict):
        return result

    if isinstance(result, (np.integer, np.floating)):
        return result.item()

    if isinstance(result, (int, float, str, bool)):
        return result

    return str(result)


def main():
    output = {
        "success": False,
        "error": None,
        "traceback": None,
        "result": None,
        "result_type": None,
    }

    try:
        if not CODE_PATH.exists():
            raise FileNotFoundError("code.py not found in /workspace")
        if not INPUT_CSV_PATH.exists():
            raise FileNotFoundError("input.csv not found in /workspace")

        code = CODE_PATH.read_text(encoding="utf-8")
        basic_safety_check(code)

        df = pd.read_csv(INPUT_CSV_PATH)
        df = prepare_dataframe(df)

        local_env = {
            "df": df,
            "pd": pd,
            "np": np,
            "__builtins__": ALLOWED_BUILTINS,
        }

        exec(code, local_env, local_env)

        if "result" not in local_env:
            raise UnsafeCodeError("Generated code must assign final output to variable named result.")

        result = local_env["result"]

        output["success"] = True
        output["result_type"] = type(result).__name__
        output["result"] = result_to_jsonable(result)

    except Exception as e:
        output["error"] = str(e)
        output["traceback"] = traceback.format_exc()

    OUTPUT_JSON_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()