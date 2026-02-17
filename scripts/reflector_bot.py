import sys
import time

from chat_agent.llm.ask import ask_messages

# SYSTEM_REFLECTOR = (
#     "Du bist der 'Reflektor'. Deine Aufgabe ist NICHT, die Frage zu beantworten. "
#     "Stattdessen analysierst du die Eingabe des Benutzers auf Klarheit, Absicht und Ton.\n\n"
#     "Wenn der Benutzer eine Nachricht sendet:\n"
#     "1) Ermittle die Kernaussage.\n"
#     "2) Prüfe die Formulierung: mehrdeutig? aggressiv? zu vage?\n"
#     "3) Schlage eine bessere Formulierung vor, um ein besseres Ergebnis von einer KI zu erhalten.\n\n"
#     "Sei konstruktiv, aber direkt."
# )

SYSTEM_REFLECTOR = (
    "Du bist ein sokratischer Lehrer. Antworte NIE direkt. "
    "Stattdessen stelle genau EINE klärende Frage, die dem Benutzer hilft, "
    "die richtige Frage zu stellen oder die Antwort selbst herzuleiten. "
    "Sei freundlich und präzise."
)

def think_dots(n: int = 3, delay: float = 0.5) -> None:
    print("Reflektor denkt", end="")
    for _ in range(n):
        time.sleep(delay)
        print(".", end="")
        sys.stdout.flush()
    print("\n")

def main() -> int:
    print("--- DER REFLEKTOR BOT ONLINE ---")
    print("Geben Sie 'exit' ein, um zu beenden.\n")

    while True:
        try:
            user_input = input("Du > ")
            if user_input.strip().lower() in ("exit", "quit"):
                print("Reflektor schaltet sich ab...")
                return 0

            think_dots()

            messages = [
                {"role": "system", "content": SYSTEM_REFLECTOR},
                {"role": "user", "content": user_input},
            ]
            response = ask_messages(messages)
            print(f"Reflektor > {response}\n")

        except KeyboardInterrupt:
            print("\n\nErzwungenes Beenden erkannt. Auf Wiedersehen.")
            return 0
        except Exception as e:
            print(f"\nEs ist ein Fehler aufgetreten: {e}")
            return 1

if __name__ == "__main__":
    raise SystemExit(main())
