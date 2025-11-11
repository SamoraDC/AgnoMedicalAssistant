"""Base agent class following Agno framework patterns."""
from abc import ABC, abstractmethod
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config import get_settings
from src.core.exceptions import LLMError, AgnoMedicalError


class AgentConfig(BaseModel):
    """Configuration for agent instances."""

    name: str = Field(description="Agent name")
    model: str = Field(description="LLM model to use")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4000, ge=1)
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout: int = Field(default=60, ge=10, le=300)
    system_prompt: str = Field(default="", description="Agent system prompt")
    enable_observability: bool = Field(default=False)


class BaseAgent(ABC):
    """Base class for all AI agents in the medical assistant system.

    Follows Agno framework patterns for agent implementation with:
    - Structured configuration
    - Error handling with retries
    - Observability integration
    - Context management
    """

    def __init__(self, config: AgentConfig):
        """Initialize base agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.settings = get_settings()
        self._setup_logging()

        # Initialize Langfuse if enabled
        self.langfuse = None
        if config.enable_observability and self.settings.langfuse_enabled:
            self._init_langfuse()

    def _setup_logging(self) -> None:
        """Configure logging for the agent."""
        logger.add(
            self.settings.log_file,
            rotation="500 MB",
            retention="10 days",
            level=self.settings.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
        )

    def _init_langfuse(self) -> None:
        """Initialize Langfuse observability client."""
        try:
            from langfuse import Langfuse
            self.langfuse = Langfuse(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                host=self.settings.langfuse_host
            )
            logger.info(f"Langfuse observability enabled for {self.config.name}")
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse: {e}")
            self.langfuse = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(LLMError),
        reraise=True
    )
    async def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Call LLM with retry logic and observability.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt override
            **kwargs: Additional LLM parameters

        Returns:
            LLM response text

        Raises:
            LLMError: If LLM call fails after retries
        """
        try:
            from groq import Groq

            client = Groq(api_key=self.settings.groq_api_key)

            messages = []
            if system_prompt or self.config.system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt or self.config.system_prompt
                })

            messages.append({"role": "user", "content": prompt})

            # Start Langfuse trace if enabled
            trace_id = None
            if self.langfuse:
                trace = self.langfuse.trace(name=f"{self.config.name}_call")
                trace_id = trace.id

            response = client.chat.completions.create(
                model=kwargs.get("model", self.config.model),
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            )

            result = response.choices[0].message.content

            # Log to Langfuse
            if self.langfuse and trace_id:
                self.langfuse.generation(
                    trace_id=trace_id,
                    name=f"{self.config.name}_generation",
                    model=self.config.model,
                    input=messages,
                    output=result,
                    usage={
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                )

            logger.info(f"{self.config.name} LLM call successful")
            return result

        except Exception as e:
            logger.error(f"{self.config.name} LLM call failed: {e}")
            raise LLMError(
                f"LLM call failed for {self.config.name}",
                details={"error": str(e)}
            )

    @abstractmethod
    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Process input and return structured output.

        Args:
            input_data: Input data for processing

        Returns:
            Structured output dictionary

        Raises:
            AgnoMedicalError: If processing fails
        """
        pass

    async def __aenter__(self) -> "BaseAgent":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self.langfuse:
            self.langfuse.flush()
