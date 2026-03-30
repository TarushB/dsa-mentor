"""
LLM Utilities — Ollama only.

invoke_llm()         — send a plain text prompt, get text back
invoke_llm_messages()— send LangChain message list, get text back
stream_llm()         — generator that streams tokens (plain prompt)
stream_llm_messages()— generator that streams tokens (message list)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


#  Ollama constructor 

def _make_ollama(temperature: float = 0.3, model: str = None, streaming: bool = False):
    from langchain_ollama import ChatOllama
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=model or OLLAMA_MODEL,
        temperature=temperature,
    )


#  Public API 

def get_llm(temperature: float = 0.3, streaming: bool = False,
            provider: str = None, model: str = None):
    """Return a LangChain Ollama chat model."""
    m = model or OLLAMA_MODEL
    print(f"[LLM] Provider: Ollama | Model: {m} | temp={temperature}")
    return _make_ollama(temperature, model=m, streaming=streaming)


def get_llm_name(provider: str = None, model: str = None) -> str:
    return f"Ollama ({model or OLLAMA_MODEL})"


def list_available_providers() -> list[dict]:
    return [
        {
            "id":     "ollama",
            "label":  "Ollama (local)",
            "model":  OLLAMA_MODEL,
            "active": True,
            "models": ["llama3.2", "llama3.1", "mistral", "codellama", "deepseek-coder"],
        },
    ]


def invoke_llm(prompt: str, temperature: float = 0.3,
               provider: str = None, model: str = None) -> str:
    """Send a plain text prompt and return the response."""
    try:
        print(f"[LLM] >> invoke [ollama] prompt_len={len(prompt)} chars", flush=True)
        llm      = _make_ollama(temperature, model=model)
        response = llm.invoke(prompt)
        text     = response.content.strip()
        print(f"[LLM] ok response [ollama] {len(text)} chars", flush=True)
        return text
    except Exception as e:
        raise RuntimeError(
            f"Ollama request failed: {e}\n"
            "Make sure Ollama is running: ollama serve"
        ) from e


def invoke_llm_messages(messages: list, temperature: float = 0.3,
                        provider: str = None, model: str = None) -> str:
    """Send a LangChain message list and return the response."""
    try:
        print(f"[LLM] >> invoke_messages [ollama] msgs={len(messages)}", flush=True)
        llm      = _make_ollama(temperature, model=model)
        response = llm.invoke(messages)
        text     = response.content.strip()
        print(f"[LLM] ok response [ollama] {len(text)} chars", flush=True)
        return text
    except Exception as e:
        raise RuntimeError(
            f"Ollama request failed: {e}\n"
            "Make sure Ollama is running: ollama serve"
        ) from e


def stream_llm(prompt: str, temperature: float = 0.3,
               provider: str = None, model: str = None):
    """Generator that streams response tokens (plain text prompt)."""
    try:
        print(f"[LLM] >> stream [ollama]", flush=True)
        llm = _make_ollama(temperature, model=model, streaming=True)
        for chunk in llm.stream(prompt):
            content = getattr(chunk, "content", chunk) or ""
            if content:
                yield content
    except Exception as e:
        raise RuntimeError(
            f"Ollama stream failed: {e}\n"
            "Make sure Ollama is running: ollama serve"
        ) from e


def stream_llm_messages(messages: list, temperature: float = 0.3,
                        provider: str = None, model: str = None):
    """Generator that streams response tokens (LangChain message list)."""
    try:
        print(f"[LLM] >> stream_messages [ollama] msgs={len(messages)}", flush=True)
        llm = _make_ollama(temperature, model=model, streaming=True)
        for chunk in llm.stream(messages):
            content = getattr(chunk, "content", chunk) or ""
            if content:
                yield content
    except Exception as e:
        raise RuntimeError(
            f"Ollama stream failed: {e}\n"
            "Make sure Ollama is running: ollama serve"
        ) from e
