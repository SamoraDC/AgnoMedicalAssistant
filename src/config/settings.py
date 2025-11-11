"""Application settings and configuration management."""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    app_name: str = "AgnoMedicalAssistant"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # Security
    secret_key: str = Field(default="your-secret-key-change-in-production")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Groq Configuration
    groq_api_key: str = Field(default="", description="Groq API key for LLM access")
    groq_model: str = Field(default="llama-3.3-70b-versatile", description="Default Groq model")
    groq_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    groq_max_tokens: int = Field(default=8000, ge=1)

    # Langfuse Configuration
    langfuse_public_key: str = Field(default="", description="Langfuse public key")
    langfuse_secret_key: str = Field(default="", description="Langfuse secret key")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")
    langfuse_enabled: bool = Field(default=False)

    # RAG Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_db_path: str = ".data/vector_store.db"
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    top_k_results: int = Field(default=5, ge=1, le=20)

    # OCR Configuration
    ocr_engine: str = Field(default="pymupdf", description="Primary OCR engine: pymupdf or tesseract")
    tesseract_cmd: Optional[str] = Field(default=None, description="Path to tesseract executable")
    ocr_languages: str = Field(default="eng+por", description="Tesseract language codes")

    # File Upload
    max_file_size_mb: int = Field(default=10, ge=1, le=100)
    allowed_file_types: list[str] = Field(
        default=[".pdf", ".png", ".jpg", ".jpeg", ".txt"]
    )
    upload_dir: str = ".data/uploads"

    # Agent Configuration
    agent_max_retries: int = Field(default=3, ge=1, le=10)
    agent_timeout_seconds: int = Field(default=60, ge=10, le=300)

    # Database
    duckdb_path: str = ".data/medical_assistant.db"

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    @field_validator("groq_api_key")
    @classmethod
    def validate_groq_key(cls, v: str) -> str:
        """Validate Groq API key is provided in production."""
        if not v:
            import warnings
            warnings.warn("GROQ_API_KEY not set. LLM features will not work.")
        return v

    @property
    def upload_path(self) -> Path:
        """Get upload directory as Path object."""
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def vector_db_full_path(self) -> Path:
        """Get vector database path as Path object."""
        path = Path(self.vector_db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
