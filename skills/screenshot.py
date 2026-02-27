"""
skills/screenshot.py
Capture a screenshot on the Pi 5 (headless or headed).
Tries multiple methods in order of preference.
"""

DESCRIPTION = (
    "Take a screenshot. Returns the file path. "
    "Args: save_path (str, optional), display (str, default ':99')"
)

import subprocess
import os
from datetime import datetime

SCREENSHOT_DIR = os.environ.get("AGENT_SCREENSHOTS", "/mnt/nvme/agent/screenshots")


def run(save_path: str = "", display: str = ":99") -> str:
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    if not save_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(SCREENSHOT_DIR, f"shot_{ts}.png")

    env = {**os.environ, "DISPLAY": display}

    methods = [
        # scrot — lightweight, works great headless with Xvfb
        ["scrot", save_path],
        # gnome-screenshot
        ["gnome-screenshot", "-f", save_path],
        # ImageMagick
        ["import", "-window", "root", save_path],
        # grim (Wayland)
        ["grim", save_path],
    ]

    errors = []
    for cmd in methods:
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=15, env=env
            )
            if result.returncode == 0 and os.path.exists(save_path):
                size = os.path.getsize(save_path)
                return f"Screenshot saved: {save_path} ({size:,} bytes)"
            errors.append(f"{cmd[0]}: exit {result.returncode} — {result.stderr.decode()[:100]}")
        except FileNotFoundError:
            errors.append(f"{cmd[0]}: not installed")
        except subprocess.TimeoutExpired:
            errors.append(f"{cmd[0]}: timeout")
        except Exception as e:
            errors.append(f"{cmd[0]}: {e}")

    return (
        "Screenshot failed — no working tool found.\n"
        "Errors:\n" + "\n".join(errors) +
        "\nFix: sudo apt install scrot  and  Xvfb :99 -screen 0 1920x1080x24 &"
    )
