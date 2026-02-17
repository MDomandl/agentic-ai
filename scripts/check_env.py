import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def check_env() -> int:
    load_dotenv("../.env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY fehlt")
        return 2

    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))

    print(f"Modell: {model}")
    print(f"Temperatur: {temperature}")

    try:
        llm = ChatOpenAI(
            model=model,
            temperature=temperature
        )
        response = llm.invoke("Say 'System Operational'")
        print("Antwort:", response.content)
        return 0

    except Exception as e:
        print("Fehler:", e)
        return 2


if __name__ == "__main__":
    exit(check_env())
