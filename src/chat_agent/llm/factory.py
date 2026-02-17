from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .settings import LLMSettings

@dataclass(frozen=True)
class LLMClient:
    """Dünner Wrapper: hält das konkrete LLM-Objekt + Metadaten."""
    provider: str
    model: str
    temperature: float
    impl: Any  # konkretes Objekt (z.B. LangChain ChatOpenAI/ChatOllama)

def get_chat_llm(cfg: LLMSettings) -> LLMClient:
    provider = (cfg.llm_provider or "").strip().lower()

    if provider == "openai":
        # LangChain Wrapper wie in deinem check_env/debug_key
        from langchain_openai import ChatOpenAI

        if not cfg.openai_api_key.strip():
            raise RuntimeError("OPENAI_API_KEY fehlt oder ist leer.")

        llm = ChatOpenAI(model=cfg.openai_model, temperature=cfg.llm_temperature)
        return LLMClient(provider="openai", model=cfg.openai_model, temperature=cfg.llm_temperature, impl=llm)

    if provider == "ollama":
        # Empfehlung: langchain-ollama installieren
        # pip install langchain-ollama
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=cfg.llm_temperature)
        return LLMClient(provider="ollama", model=cfg.ollama_model, temperature=cfg.llm_temperature, impl=llm)

    raise RuntimeError(f"Unbekannter LLM_PROVIDER: {provider!r} (erwartet: openai|ollama)")
