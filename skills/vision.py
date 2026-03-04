"""
skills/vision.py
Send an image to llama3.2-vision:11b and get a description or answer.
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
            model="llama3.2-vision:11b",
            messages=[{
                "role": "user",
                "content": question,
                "images": [image_b64],
            }],
            options={"temperature": 0.3, "num_predict": 800, "num_ctx": 4096},
        )
        return resp["message"]["content"].strip()

    except Exception as e:
        return f"Vision error: {e}"
