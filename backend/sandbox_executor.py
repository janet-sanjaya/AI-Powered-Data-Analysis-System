import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


SANDBOX_IMAGE = "ainsight-sandbox:latest"


class SandboxExecutionError(Exception):
    pass


def ensure_image_built() -> None:
    """
    Build the Docker image if it does not exist locally.
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", SANDBOX_IMAGE],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return

        sandbox_dir = Path(__file__).resolve().parent.parent / "sandbox"
        if not sandbox_dir.exists():
            raise FileNotFoundError(f"Sandbox directory not found: {sandbox_dir}")

        build = subprocess.run(
            ["docker", "build", "-t", SANDBOX_IMAGE, str(sandbox_dir)],
            capture_output=True,
            text=True,
            check=False,
        )

        if build.returncode != 0:
            raise SandboxExecutionError(
                f"Failed to build sandbox image.\nSTDOUT:\n{build.stdout}\nSTDERR:\n{build.stderr}"
            )

    except FileNotFoundError as e:
        raise SandboxExecutionError(
            "Docker is not installed or not available in PATH."
        ) from e


def execute_pandas_code_in_sandbox(code: str, df: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    """
    Execute generated Pandas code in a Docker sandbox.

    Returns:
        result, metadata
    """
    ensure_image_built()

    metadata = {
        "success": False,
        "error": None,
        "traceback": None,
        "result_type": None,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        code_path = tmp_path / "code.py"
        csv_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.json"

        code_path.write_text(code, encoding="utf-8")
        df.to_csv(csv_path, index=False)

        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cpus",
            "1",
            "--memory",
            "512m",
            "--pids-limit",
            "128",
            "-v",
            f"{tmp_path}:/workspace",
            SANDBOX_IMAGE,
        ]

        run = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if run.returncode != 0:
            metadata["error"] = f"Docker execution failed: {run.stderr.strip() or run.stdout.strip()}"
            return None, metadata

        if not output_path.exists():
            metadata["error"] = "Sandbox did not produce output.json."
            return None, metadata

        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            metadata["error"] = f"Failed to parse sandbox output: {e}"
            return None, metadata

        metadata["result_type"] = payload.get("result_type")

        if not payload.get("success"):
            metadata["error"] = payload.get("error")
            metadata["traceback"] = payload.get("traceback")
            return None, metadata

        metadata["success"] = True
        return payload.get("result"), metadata