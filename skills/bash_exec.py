"""
skills/bash_exec.py
Execute bash commands on the Pi. Has safety guards against destructive commands.
"""

DESCRIPTION = "Execute a bash command and return stdout+stderr. Args: command (str), timeout (int, default 30), workdir (str, optional)"

import subprocess
import os
import shlex

# Hard-blocked patterns — never execute these
BLOCKED_PATTERNS = [
    "rm -rf /",
    "rm -rf ~",
    "mkfs",
    ":(){:|:&};:",    # fork bomb
    "dd if=/dev/zero of=/dev/sd",
    "dd if=/dev/zero of=/dev/nvme",
    "> /dev/sda",
    "chmod -R 777 /",
    "chown -R root /",
    "shutdown",
    "reboot",
    "halt",
    "init 0",
    "init 6",
]

WORKSPACE = os.environ.get("AGENT_WORKSPACE", "/mnt/nvme/agent/workspace")


def run(command: str, timeout: int = 60, workdir: str = "") -> str:
    # Safety check
    for pattern in BLOCKED_PATTERNS:
        if pattern in command:
            return f"BLOCKED: Command contains dangerous pattern '{pattern}'"

    # Default working directory
    cwd = workdir if workdir and os.path.isdir(workdir) else WORKSPACE
    os.makedirs(cwd, exist_ok=True)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env={**os.environ, "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + f"[stderr]\n{result.stderr}"

        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"

        return output.strip() or "(command produced no output)"

    except subprocess.TimeoutExpired:
        return f"TIMEOUT: Command exceeded {timeout}s: {command}"
    except Exception as e:
        return f"ERROR: {e}"
