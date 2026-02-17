from __future__ import annotations

import os
import time
from typing import Iterable, List, Dict, Optional

from .factory import get_chat_llm
from .settings import load_settings


def _env_flag(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in ("1", "true", "yes", "on")


cfg = load_settings()


def _dbg(msg: str) -> None:
    if cfg.llm_debug:
        print(msg)


def _to_langchain_messages(msgs: Iterable[Dict[str, str]]):
    """Konvertiert {'role': 'system'|'user'|'assistant', 'content': '...'} in LangChain Messages."""
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    out = []
    for m in msgs:
        role = (m.get("role") or "").strip().lower()
        content = m.get("content") or ""
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role in ("user", "human"):
            out.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            out.append(AIMessage(content=content))
        else:
            raise ValueError(f"Unknown role: {role!r}")
    return out


def ask_messages(messages: List[Dict[str, str]], *, fallback: bool = True) -> str:
    """
    Einzige öffentliche Funktion für Chat-Text.
    messages: Liste aus dicts mit role/content.
    """

    def _call(provider_override: Optional[str] = None) -> tuple[str, str, str]:
        """
        Returns: (text, provider, model)
        """
        if provider_override:
            cfg2 = cfg.model_copy(update={"llm_provider": provider_override})
            client = get_chat_llm(cfg2)
        else:
            client = get_chat_llm(cfg)

        lc_messages = _to_langchain_messages(messages)
        resp = client.impl.invoke(lc_messages)
        text = getattr(resp, "content", str(resp))
        return text, client.provider, client.model

    t0 = time.perf_counter()
    used_fallback = False

    try:
        text, provider, model = _call()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _dbg(f"[LLM] provider={provider} model={model} ms={dt_ms:.0f} fallback={used_fallback}")
        return text

    except Exception as e:
        fb = (cfg.llm_fallback_provider or "").strip().lower()
        cur = (cfg.llm_provider or "").strip().lower()

        if fallback and fb and fb != cur:
            used_fallback = True
            _dbg(f"[LLM] primary failed ({cur}): {type(e).__name__}: {e}")
            t1 = time.perf_counter()

            text, provider, model = _call(provider_override=fb)
            dt_ms = (time.perf_counter() - t1) * 1000.0
            _dbg(f"[LLM] provider={provider} model={model} ms={dt_ms:.0f} fallback={used_fallback}")
            return text

        # kein Fallback → Fehler hoch
        _dbg(f"[LLM] failed (no fallback): {type(e).__name__}: {e}")
        raise


def ask_text(prompt: str, system: Optional[str] = None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return ask_messages(msgs)
