"""Custom exception classes for the medical assistant system."""


class AgnoMedicalError(Exception):
    """Base exception for all medical assistant errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(AgnoMedicalError):
    """Raised when configuration is invalid or missing."""
    pass


class LLMError(AgnoMedicalError):
    """Raised when LLM operations fail."""
    pass


class OCRError(AgnoMedicalError):
    """Raised when OCR processing fails."""
    pass


class RAGError(AgnoMedicalError):
    """Raised when RAG pipeline operations fail."""
    pass


class ValidationError(AgnoMedicalError):
    """Raised when input validation fails."""
    pass
