"""Triador (Triage) Agent for medical symptom assessment."""
import json
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

from src.core import BaseAgent, AgentConfig, ValidationError


class TriageResult(BaseModel):
    """Structured triage assessment result."""

    urgency_level: Literal["emergency", "urgent", "non_urgent", "routine"] = Field(
        description="Assessment of case urgency"
    )
    reasoning: str = Field(description="Step-by-step reasoning for the assessment")
    symptoms_identified: list[str] = Field(description="Key symptoms identified")
    red_flags: list[str] = Field(default_factory=list, description="Critical warning signs")
    recommended_actions: list[str] = Field(description="Immediate actions recommended")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in assessment")
    requires_specialist: bool = Field(default=False, description="Whether specialist is needed")
    specialist_type: str | None = Field(default=None, description="Type of specialist if needed")


class TriadorAgent(BaseAgent):
    """AI agent for medical triage using Chain-of-Thought reasoning.

    Implements the Triador (Triage) functionality with:
    - Symptom analysis
    - Urgency classification
    - Red flag detection
    - Structured reasoning output
    """

    URGENCY_CRITERIA = {
        "emergency": [
            "chest pain", "difficulty breathing", "severe bleeding", "loss of consciousness",
            "sudden severe headache", "stroke symptoms", "allergic reaction", "seizure",
            "severe abdominal pain", "suicidal thoughts"
        ],
        "urgent": [
            "high fever", "severe pain", "persistent vomiting", "deep wound",
            "severe dehydration", "head injury", "suspected fracture", "severe infection"
        ],
        "non_urgent": [
            "mild fever", "minor cut", "cold symptoms", "muscle ache", "mild headache"
        ],
        "routine": [
            "health checkup", "prescription refill", "minor skin issue", "mild discomfort"
        ]
    }

    SYSTEM_PROMPT = """You are an expert medical triage AI assistant. Your role is to assess patient symptoms and provide structured triage recommendations.

CRITICAL RULES:
1. Always use Chain-of-Thought reasoning - explain your thought process step by step
2. Identify red flags that indicate emergency situations
3. Be conservative - when in doubt, escalate urgency level
4. Never provide specific medical diagnoses - only triage assessments
5. Always recommend professional medical evaluation for serious symptoms
6. Consider patient age, medical history, and symptom duration

URGENCY LEVELS:
- emergency: Life-threatening conditions requiring immediate care (911/ER)
- urgent: Serious conditions needing prompt attention within hours
- non_urgent: Conditions requiring medical attention within 24-48 hours
- routine: Non-urgent issues that can wait for regular appointment

You must respond with valid JSON following this exact structure:
{
    "urgency_level": "emergency|urgent|non_urgent|routine",
    "reasoning": "Step-by-step explanation of your assessment",
    "symptoms_identified": ["symptom1", "symptom2"],
    "red_flags": ["flag1", "flag2"],
    "recommended_actions": ["action1", "action2"],
    "confidence_score": 0.0-1.0,
    "requires_specialist": true|false,
    "specialist_type": "type of specialist or null"
}"""

    def __init__(self, config: AgentConfig | None = None):
        """Initialize Triador agent.

        Args:
            config: Optional agent configuration. Uses defaults if not provided.
        """
        if config is None:
            config = AgentConfig(
                name="Triador",
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=2000,
                system_prompt=self.SYSTEM_PROMPT,
                enable_observability=True
            )
        super().__init__(config)

    def _build_triage_prompt(self, input_data: dict[str, Any]) -> str:
        """Build structured prompt for triage assessment.

        Args:
            input_data: Patient information and symptoms

        Returns:
            Formatted prompt string
        """
        symptoms = input_data.get("symptoms", "")
        age = input_data.get("age")
        gender = input_data.get("gender")
        medical_history = input_data.get("medical_history", "")
        duration = input_data.get("symptom_duration", "")
        severity = input_data.get("severity_rating")

        prompt_parts = [
            "PATIENT INFORMATION:",
            f"Symptoms: {symptoms}",
        ]

        if age:
            prompt_parts.append(f"Age: {age}")
        if gender:
            prompt_parts.append(f"Gender: {gender}")
        if medical_history:
            prompt_parts.append(f"Medical History: {medical_history}")
        if duration:
            prompt_parts.append(f"Symptom Duration: {duration}")
        if severity:
            prompt_parts.append(f"Self-Reported Severity (1-10): {severity}")

        prompt_parts.extend([
            "",
            "TASK: Perform a thorough triage assessment using Chain-of-Thought reasoning.",
            "Consider:",
            "1. What are the key symptoms presented?",
            "2. Are there any red flags or emergency indicators?",
            "3. What is the appropriate urgency level?",
            "4. What immediate actions should be taken?",
            "5. Is specialist consultation needed?",
            "",
            "Provide your assessment as valid JSON following the specified structure."
        ])

        return "\n".join(prompt_parts)

    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Process patient symptoms and perform triage assessment.

        Args:
            input_data: Dictionary containing:
                - symptoms (str, required): Patient symptoms description
                - age (int, optional): Patient age
                - gender (str, optional): Patient gender
                - medical_history (str, optional): Relevant medical history
                - symptom_duration (str, optional): How long symptoms have lasted
                - severity_rating (int, optional): Self-reported severity 1-10

        Returns:
            Dictionary containing TriageResult and metadata

        Raises:
            ValidationError: If input validation fails
        """
        logger.info(f"Starting triage assessment for case")

        # Validate input
        if "symptoms" not in input_data or not input_data["symptoms"]:
            raise ValidationError("Missing required field: symptoms")

        try:
            # Build prompt
            prompt = self._build_triage_prompt(input_data)

            # Call LLM
            response = await self._call_llm(prompt)

            # Parse JSON response
            # Remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            result_dict = json.loads(response)

            # Validate and structure result
            triage_result = TriageResult(**result_dict)

            logger.info(
                f"Triage completed: {triage_result.urgency_level} "
                f"(confidence: {triage_result.confidence_score:.2f})"
            )

            return {
                "success": True,
                "triage_result": triage_result.model_dump(),
                "input_summary": {
                    "symptoms": input_data.get("symptoms"),
                    "age": input_data.get("age"),
                    "duration": input_data.get("symptom_duration")
                }
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {response}")
            raise ValidationError(
                "Invalid response format from triage assessment",
                details={"error": str(e), "response": response}
            )
        except Exception as e:
            logger.error(f"Triage assessment failed: {e}")
            raise
