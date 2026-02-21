from __future__ import annotations

import argparse
import csv
from datetime import date
from typing import List

from pydantic import BaseModel, Field

from chat_agent.llm.structured import parse_structured
from chat_agent.policy.risk import RiskLevel, decide


def _weekday_de(d: date) -> str:
    names = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    return names[d.weekday()]


class Task(BaseModel):
    description: str = Field(description="Kurze, umsetzbare Zusammenfassung der Aufgabe.")
    assignee: str = Field(
        description="Name der verantwortlichen Person. Falls nicht genannt: 'Nicht zugewiesen'."
    )
    due_date: str = Field(
        description="Fälligkeitsdatum im Format YYYY-MM-DD. Relative Angaben werden anhand des heutigen Datums aufgelöst.",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    confidence_score: float = Field(
        description=(
            "0..1: Wie sicher bist du, dass die Extraktion korrekt ist. "
            "Hoch, wenn eindeutig; niedriger, wenn unklar/mehrdeutig/fehlende Infos."
        ),
        ge=0.0,
        le=1.0,
    )


class ProjectUpdate(BaseModel):
    project_name: str = Field(description="Abgeleiteter Projekt-/Kontextname (kurz).")
    tasks: List[Task]


def build_system_prompt(today: date) -> str:
    return (
        "Du bist ein präziser Projektmanagement-Assistent.\n"
        f"Heutiges Datum ist {today.isoformat()} ({_weekday_de(today)}).\n\n"
        "Aufgabe:\n"
        "- Extrahiere aus der E-Mail ALLE konkreten Aufgaben.\n"
        "- Jede Aufgabe bekommt: description, assignee, due_date (YYYY-MM-DD) und confidence_score (0..1).\n"
        "- Relative Datumsangaben (z.B. 'nächsten Dienstag', 'kommenden Freitag') musst du in ein absolutes Datum umrechnen.\n"
        "- Wenn kein Bearbeiter genannt ist: assignee = 'Nicht zugewiesen'.\n"
        "- Wenn kein Datum genannt ist: setze due_date auf das heutige Datum.\n"
        "- Gib nur Daten gemäß Schema zurück (kein Zusatztext)."
    )


def read_email_text(path: str | None) -> str:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    print("E-Mail-Inhalt eingeben (Ende mit Strg+Z dann Enter in Windows):\n")
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse E-Mail → Tasks → CSV (structured, provider-unabhängig)")
    ap.add_argument("--in", dest="in_path", default=None, help="Pfad zur Eingabe-Textdatei (UTF-8). Optional.")
    ap.add_argument("--out", dest="out_path", default="tasks.csv", help="Ziel-CSV (default: tasks.csv)")
    ap.add_argument(
        "--risk",
        dest="risk",
        default="medium",
        choices=["low", "medium", "high"],
        help="Risk-Level: low|medium|high (default: medium)",
    )
    ap.add_argument("--retries", dest="retries", type=int, default=2, help="Anzahl Repair-Retries (default: 2)")
    args = ap.parse_args()

    email_text = read_email_text(args.in_path)
    if not email_text:
        print("Keine Eingabe erhalten.")
        return 2

    today = date.today()
    system_prompt = build_system_prompt(today)

    result = parse_structured(
        ProjectUpdate,
        email_text,
        system=system_prompt,
        max_retries=args.retries,
    )

    level = RiskLevel(args.risk)
    rows = []
    for t in result.tasks:
        decision = decide(t.confidence_score, level)
        rows.append((t.description, t.assignee, t.due_date, f"{t.confidence_score:.2f}", decision))

    # UTF-8-SIG hilft Excel unter Windows häufig beim Erkennen der Kodierung
    with open(args.out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["description", "assignee", "due_date", "confidence", "decision"])
        w.writerows(rows)

    print(f"OK: {len(rows)} Tasks nach {args.out_path} geschrieben. (project_name={result.project_name!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
