from __future__ import annotations
from typing import Any, Dict
from pydantic import BaseModel

def tool_echo(args: Dict[str, Any]) -> str:
    return str(args.get("text", ""))

def _get_number(args: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in args:
            return float(args[k])
    return default

def tool_add(args: Dict[str, Any]) -> str:
    # akzeptiert a/b sowie num1/num2 sowie x/y
    a = _get_number(args, "a", "num1", "x")
    b = _get_number(args, "b", "num2", "y")
    return str(a + b)

class AddArgs(BaseModel):
    a: float
    b: float

class EchoArgs(BaseModel):
    text: str