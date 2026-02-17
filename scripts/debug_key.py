import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI





def debug_key() -> int:
    print("cwd:", os.getcwd())
    print(".env abs:", os.path.abspath("../.env"))

    geladen = load_dotenv("../.env")
    print("dotenv geladen:", geladen)

    raw = os.getenv("OPENAI_API_KEY")
    print("raw is None:", raw is None)

    if raw:
        print("raw len:", len(raw), "prefix:", raw[:6], "suffix:", raw[-4:])

    api_key = (raw or "").strip().strip('"').strip("'")
    print("clean len:", len(api_key), "prefix:", api_key[:6], "suffix:", api_key[-4:])

    print("starts sk-:", api_key.startswith("sk-"))

    print("Verbindung zu OpenAI wird getestet...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    response = llm.invoke("Say 'System Operational'")
    print(f"ANTWORT DER KI: {response.content}")
    return 0

if __name__ == "__main__":
    debug_key()