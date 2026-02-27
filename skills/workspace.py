"""
skills/workspace.py
Read/write/list files in the agent's persistent workspace on NVMe.
"""

DESCRIPTION = (
    "Manage files in agent workspace. "
    "Args: action (read|write|append|list|delete|exists|mkdir), "
    "path (str), content (str, for write/append)"
)

import os
import shutil

WORKSPACE = os.environ.get("AGENT_WORKSPACE", "/mnt/nvme/agent/workspace")


def _safe_path(path: str) -> str:
    """Resolve path safely within workspace."""
    # Strip leading slashes so path.join works
    clean = path.lstrip("/")
    full = os.path.normpath(os.path.join(WORKSPACE, clean))
    # Ensure we stay inside workspace
    if not full.startswith(WORKSPACE):
        raise ValueError(f"Path escape attempt: {path}")
    return full


def run(action: str, path: str = "", content: str = "") -> str:
    os.makedirs(WORKSPACE, exist_ok=True)

    try:
        if action == "list":
            target = _safe_path(path) if path else WORKSPACE
            if not os.path.exists(target):
                return f"Path not found: {target}"
            entries = []
            for entry in sorted(os.listdir(target)):
                full = os.path.join(target, entry)
                size = os.path.getsize(full) if os.path.isfile(full) else 0
                kind = "dir" if os.path.isdir(full) else "file"
                entries.append(f"{kind:4s}  {size:>10,}  {entry}")
            return "\n".join(entries) if entries else "(empty)"

        elif action == "read":
            if not path:
                return "ERROR: path required for read"
            full = _safe_path(path)
            if not os.path.exists(full):
                return f"File not found: {path}"
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
            return data if len(data) <= 8000 else data[:8000] + f"\n...[truncated, {len(data)} total chars]"

        elif action == "write":
            if not path:
                return "ERROR: path required for write"
            full = _safe_path(path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Written {len(content)} chars to {path}"

        elif action == "append":
            if not path:
                return "ERROR: path required for append"
            full = _safe_path(path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "a", encoding="utf-8") as f:
                f.write(content)
            return f"Appended {len(content)} chars to {path}"

        elif action == "delete":
            if not path:
                return "ERROR: path required for delete"
            full = _safe_path(path)
            if not os.path.exists(full):
                return f"Not found: {path}"
            if os.path.isdir(full):
                shutil.rmtree(full)
                return f"Deleted directory: {path}"
            else:
                os.remove(full)
                return f"Deleted file: {path}"

        elif action == "exists":
            full = _safe_path(path)
            return str(os.path.exists(full))

        elif action == "mkdir":
            full = _safe_path(path)
            os.makedirs(full, exist_ok=True)
            return f"Created directory: {path}"

        else:
            return f"Unknown action: {action}. Use: read, write, append, list, delete, exists, mkdir"

    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"
