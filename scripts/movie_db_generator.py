from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from chat_agent.llm.structured import parse_structured


class Movie(BaseModel):
    title: str = Field(description="Filmtitel")
    year: int = Field(description="Erscheinungsjahr (vierstellig)", ge=1888, le=2100)
    director: str = Field(description="Regisseur")
    rotten_tomatoes_score_estimate: Optional[int] = Field(
        default=None,
        description=(
            "GESCHÄTZTER Rotten-Tomatoes-Score (0..100). "
            "Wenn du dir unsicher bist oder es reine Vermutung wäre: null."
        ),
        ge=0,
        le=100,
    )
    confidence_score: float = Field(
        description="0..1: Wie sicher bist du, dass Titel/Jahr/Regisseur plausibel sind (Extraktion/Generierung).",
        ge=0.0,
        le=1.0,
    )


class MovieList(BaseModel):
    movies: List[Movie]


SYSTEM = (
    "Du bist eine Daten-API. Du gibst ausschließlich JSON gemäß Schema zurück.\n"
    "Aufgabe:\n"
    "- Gib eine Liste passender Filme zurück.\n"
    "- Fokus: Science-Fiction, 1990er Jahre, Roboter oder Außerirdische.\n"
    "- Keine Erklärungen, kein Markdown.\n"
    "- rotten_tomatoes_score_estimate ist NUR eine Schätzung. Wenn du nicht sicher bist: null.\n"
)


def main() -> int:
    user_query = (
        "Ich möchte mir ein paar Science-Fiction-Filme aus den 90ern ansehen, "
        "vielleicht etwas mit Robotern oder Außerirdischen."
    )

    result = parse_structured(MovieList, user_query, system=SYSTEM, max_retries=2)

    for m in result.movies:
        print(f"Title: {m.title} (Dir: {m.director}) (Erscheinungsjahr:  {m.year})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
