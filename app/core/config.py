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
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_dim: int = 768
    llm_model: str = "gemini-2.5-flash"
    max_tokens: int = 2048
    top_k_fragments: int = 3
    similarity_threshold: float = 0.65
    aws_region: str
    s3_bucket_name: str
    s3_prefix: str = "documents/"
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
    }

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_name}"


settings = Settings()
