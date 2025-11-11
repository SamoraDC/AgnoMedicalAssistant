"""FastAPI application setup with middleware and routes."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.config import get_settings
from src.core.exceptions import AgnoMedicalError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Initialize services
    # TODO: Add service initialization (DB connections, etc.)

    yield

    # Shutdown
    logger.info("Shutting down application")
    # TODO: Add cleanup logic


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI Medical Assistant System with Triage, RAG, and OCR",
        docs_url="/api/docs" if settings.debug else None,
        redoc_url="/api/redoc" if settings.debug else None,
        lifespan=lifespan
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Security: Trusted Host Middleware
    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
        )

    # Exception Handlers
    @app.exception_handler(AgnoMedicalError)
    async def medical_error_handler(request: Request, exc: AgnoMedicalError):
        """Handle custom medical assistant errors."""
        logger.error(f"Medical error: {exc.message} | Details: {exc.details}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": exc.__class__.__name__,
                "message": exc.message,
                "details": exc.details
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "ValidationError",
                "message": "Invalid request data",
                "details": exc.errors()
            }
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        """Handle unexpected errors."""
        logger.exception(f"Unexpected error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred"
            }
        )

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "app_name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment
        }

    # TODO: Add route modules
    # from src.api.routes import triage, rag, ocr
    # app.include_router(triage.router, prefix=f"{settings.api_prefix}/triage")
    # app.include_router(rag.router, prefix=f"{settings.api_prefix}/rag")
    # app.include_router(ocr.router, prefix=f"{settings.api_prefix}/ocr")

    logger.info("FastAPI application configured successfully")
    return app


# Create application instance
app = create_app()
