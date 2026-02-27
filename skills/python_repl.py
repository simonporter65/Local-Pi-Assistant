"""
skills/python_repl.py
Execute Python code snippets and return output.
For data analysis, computation, and quick scripting.
"""

DESCRIPTION = (
    "Execute Python code and return stdout output. "
    "Args: code (str), timeout (int, default 30)"
)

import subprocess
import sys
import os
import tempfile


def run(code: str, timeout: int = 30) -> str:
    # Write code to temp file (avoids shell escaping nightmares)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.environ.get("AGENT_WORKSPACE", "/tmp"),
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n[stderr]\n" if output else "[stderr]\n") + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"

        if not output.strip():
            return "(code ran with no output)"

        # Trim huge outputs
        if len(output) > 5000:
            output = output[:4900] + f"\n...[truncated, {len(output)} total chars]"

        return output.strip()

    except subprocess.TimeoutExpired:
        return f"TIMEOUT: Code exceeded {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
