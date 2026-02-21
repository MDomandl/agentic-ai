from __future__ import annotations

import json
from typing import Type, TypeVar, Optional

from pydantic import BaseModel, ValidationError

from .ask import ask_messages

T = TypeVar("T", bound=BaseModel)

_JSON_SYSTEM = (
    "Du gibst IMMER NUR gültiges JSON zurück, ohne Einleitung, ohne Markdown, ohne Codeblock.\n"
    "Kein Zusatztext. Nur JSON.\n"
)

def _repair_prompt(schema_name: str, schema_json: str, bad_output: str, error: str) -> str:
    return (
        f"Die folgende Ausgabe ist ungültig oder passt nicht zum Schema.\n\n"
        f"SCHEMA-NAME: {schema_name}\n"
        f"SCHEMA (JSON Schema): {schema_json}\n\n"
        f"UNGÜLTIGE AUSGABE:\n{bad_output}\n\n"
        f"VALIDIERUNGSFEHLER:\n{error}\n\n"
        "Gib jetzt NUR das korrigierte JSON zurück, das exakt zum Schema passt."
    )

def parse_structured(
    model: Type[T],
    user_text: str,
    *,
    system: Optional[str] = None,
    max_retries: int = 2,
) -> T:
    """
    Erzwingt strukturierte Ausgabe via JSON + Pydantic Validierung + Repair-Loop.
    Funktioniert provider-unabhängig (OpenAI/Ollama), weil wir uns nicht auf tool-calling verlassen.
    """
    # Pydantic v2: JSON Schema als dict
    schema_dict = model.model_json_schema()
    schema_json = json.dumps(schema_dict, ensure_ascii=False)

    base_system = _JSON_SYSTEM + (
        f"Deine Ausgabe muss dem folgenden JSON Schema entsprechen:\n{schema_json}\n"
    )
    if system:
        base_system = system + "\n\n" + base_system

    messages = [
        {"role": "system", "content": base_system},
        {"role": "user", "content": user_text},
    ]

    last_error = None
    raw = None

    for _ in range(max_retries + 1):
        raw = ask_messages(messages)
        try:
            data = json.loads(raw)
            return model.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            # Repair-Schleife: wir geben Fehler + Schema zurück
            messages = [
                {"role": "system", "content": _JSON_SYSTEM},
                {"role": "user", "content": _repair_prompt(model.__name__, schema_json, raw, str(e))},
            ]

    # Wenn wir hier landen: endgültig gescheitert
    raise RuntimeError(f"Structured parse failed after retries. Last error: {last_error}. Raw: {raw!r}")
