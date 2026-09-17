from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    google_maps_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/trip_planner"
    sync_database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/trip_planner"

    class Config:
        env_file = ".env"


settings = Settings()