"""Shared test helpers."""

import json
import shutil
import urllib.request

import pytest

needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="Tesseract is not installed"
)


def ollama_has_model(model: str, url: str = "http://localhost:11434") -> bool:
    """True if an Ollama server is reachable and has `model` pulled."""
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=1) as response:
            data = json.load(response)
    except OSError:
        return False
    return any(m.get("name", "").split(":")[0] == model for m in data.get("models", []))


needs_ollama_embeddings = pytest.mark.skipif(
    not ollama_has_model("nomic-embed-text"),
    reason="Ollama with nomic-embed-text is not running",
)


needs_ollama_llm = pytest.mark.skipif(
    not ollama_has_model("llama3.2"),
    reason="Ollama with llama3.2 is not running",
)
