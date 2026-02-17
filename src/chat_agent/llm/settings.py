from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_fallback_provider: str = Field(default="", alias="LLM_FALLBACK_PROVIDER")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")

    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    ollama_model: str = Field(default="llama3", alias="OLLAMA_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    llm_debug: bool = Field(default=False, alias="LLM_DEBUG")

def load_settings() -> LLMSettings:
    return LLMSettings()
