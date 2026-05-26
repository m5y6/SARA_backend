"""
Configuration module for SARA API.
Loads environment variables and provides configuration settings.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration from environment variables."""

    db_host: str
    db_port: int = 5432
    db_user: str
    db_pass: str
    db_name: str
    gemini_api_key: str
    api_title: str = "SARA - Sistema de Asistencia de Reglamentos Académicos"
    api_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    frontend_url: str = "http://localhost:3000"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    llm_model: str = "gemini-1.5-flash"
    max_tokens: int = 2048
    top_k_fragments: int = 3
    similarity_threshold: float = 0.3

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_name}"


settings = Settings()
