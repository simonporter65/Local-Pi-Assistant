"""
skills/vision.py
Send an image to qwen3.5:0.8b (multimodal) and get a description or answer.

qwen3.5 is natively multimodal — no separate vision model needed.
Works with screenshots from browser.py or screenshot.py.
"""

DESCRIPTION = (
    "Interpret an image using the local multimodal model. "
    "Args: image_path (str), question (str, optional — default: 'Describe what you see')"
)

import base64
import os


def run(image_path: str, question: str = "Describe what you see in detail.") -> str:
    if not image_path or not os.path.exists(image_path):
        return f"ERROR: image not found at {image_path!r}"

    try:
        import ollama
    except ImportError:
        return "ERROR: ollama package not installed"

    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        resp = ollama.chat(
            model="qwen3.5:0.8b",
            messages=[{
                "role": "user",
                "content": question,
                "images": [image_b64],
            }],
            options={"temperature": 0.3, "num_predict": 800, "num_ctx": 4096},
            think=False,
        )
        return resp["message"]["content"].strip()

    except Exception as e:
        return f"Vision error: {e}"
