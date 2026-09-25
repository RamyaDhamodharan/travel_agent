import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    google_maps_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    cerebras_api_key: str = ""
    tavily_api_key: str = ""

    groq_model: str = "openai/gpt-oss-120b"
    openai_model: str = "gpt-5.4-mini"

    openrouter_model: str = "deepseek/deepseek-chat"
    cerebras_model: str = "llama-3.3-70b"

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/trip_planner"
    sync_database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/trip_planner"

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "travel-planning-agent"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

if settings.langsmith_tracing:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project