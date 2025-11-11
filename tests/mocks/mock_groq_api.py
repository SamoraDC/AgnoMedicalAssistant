"""
Mock Groq API client for testing without actual API calls

Provides deterministic responses based on input patterns
Simulates realistic latency for performance testing
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class MockUsage:
    """Mock token usage statistics"""
    prompt_tokens: int = 150
    completion_tokens: int = 200
    total_tokens: int = 350


@dataclass
class MockMessage:
    """Mock message response"""
    content: str
    role: str = "assistant"


@dataclass
class MockChoice:
    """Mock choice in response"""
    message: MockMessage
    finish_reason: str = "stop"
    index: int = 0


@dataclass
class MockChatCompletion:
    """Mock chat completion response"""
    id: str
    choices: List[MockChoice]
    usage: MockUsage
    model: str
    created: int = field(default_factory=lambda: int(time.time()))


class MockGroqClient:
    """
    Mock Groq API client for testing

    Features:
    - Deterministic responses based on message content
    - Configurable latency simulation
    - Token usage tracking
    - Call count tracking
    - Error simulation
    """

    def __init__(
        self,
        latency_ms: int = 50,
        simulate_errors: bool = False,
        error_rate: float = 0.0
    ):
        """
        Initialize mock Groq client

        Args:
            latency_ms: Simulated API latency in milliseconds
            simulate_errors: Whether to simulate API errors
            error_rate: Probability of error (0.0 to 1.0)
        """
        self.latency_ms = latency_ms
        self.simulate_errors = simulate_errors
        self.error_rate = error_rate
        self.call_count = 0
        self.total_tokens = 0
        self.call_history: List[Dict[str, Any]] = []

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "mixtral-8x7b-32768",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> MockChatCompletion:
        """
        Mock chat completion endpoint

        Args:
            messages: List of message dictionaries
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            MockChatCompletion with appropriate response
        """
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000)

        # Track call
        self.call_count += 1
        self.call_history.append({
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timestamp": time.time()
        })

        # Simulate errors if configured
        if self.simulate_errors and self._should_error():
            raise Exception("Simulated Groq API error")

        # Get last user message
        last_message = messages[-1]["content"].lower() if messages else ""

        # Generate deterministic response based on content
        response_content = self._generate_response(last_message)

        # Create mock response
        usage = MockUsage(
            prompt_tokens=self._count_tokens(messages),
            completion_tokens=self._count_tokens([{"content": response_content}]),
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

        self.total_tokens += usage.total_tokens

        return MockChatCompletion(
            id=f"chatcmpl-mock-{self.call_count}",
            choices=[
                MockChoice(
                    message=MockMessage(content=response_content)
                )
            ],
            usage=usage,
            model=model
        )

    def _generate_response(self, content: str) -> str:
        """Generate deterministic response based on message content"""
        # Cardiology responses
        if "cardiology" in content or "chest pain" in content or "troponin" in content:
            return """Based on the elevated troponin (0.5 ng/mL) and BNP (450 pg/mL) levels, combined with the patient's presenting symptoms of chest pain and shortness of breath, this is highly suggestive of Acute Coronary Syndrome (ACS).

Key findings:
1. Elevated cardiac biomarkers indicate myocardial injury
2. Vital signs show tachycardia (HR 92) and borderline hypertension
3. Medical history of hypertension and hyperlipidemia are risk factors

Recommendation:
- Immediate cardiology consultation
- ECG and cardiac catheterization
- Initiate ACS protocol
- Consider antiplatelet therapy if not contraindicated

Confidence: 0.85"""

        # Neurology responses
        elif "neurology" in content or "headache" in content or "neurological" in content:
            return """Neurological assessment indicates potential migraine with atypical features.

Key findings:
1. Pattern of headache consistent with migraine
2. No focal neurological deficits
3. Normal vital signs

Recommendation:
- Consider prophylactic treatment
- Lifestyle modifications
- Rule out secondary causes with imaging if red flags present

Confidence: 0.75"""

        # Endocrinology responses
        elif "endocrinology" in content or "thyroid" in content or "diabetes" in content:
            return """Laboratory findings indicate:
1. Hypothyroidism: TSH 12.5, Free T4 0.6 (below normal)
2. Uncontrolled diabetes: HbA1c 8.2%, glucose 180 mg/dL
3. Vitamin D deficiency

Current levothyroxine dose (50mcg) appears insufficient. Symptoms of fatigue and weight gain are consistent with undertreated hypothyroidism.

Recommendations:
- Increase levothyroxine to 75mcg daily
- Optimize diabetes management (consider medication adjustment)
- Vitamin D supplementation
- Recheck labs in 6 weeks

Confidence: 0.80"""

        # Validation/conflict resolution
        elif "conflict" in content or "contraindication" in content:
            return """Conflict resolution analysis:

After reviewing the evidence presented by both specialties, I identify the following:

1. Medication interaction: The proposed beta-blocker from Cardiology may worsen hypothyroid symptoms noted by Endocrinology
2. Alternative: Consider calcium channel blocker instead
3. Both specialists' concerns are valid and need addressing

Synthesized recommendation:
- Address cardiac issues with CCB rather than beta-blocker
- Optimize thyroid treatment first
- Monitor closely for 2 weeks
- Reassess cardiac medications after thyroid stabilization

Confidence: 0.78"""

        # Triaging/classification
        elif "classify" in content or "triage" in content or "specialty" in content:
            return """Case classification analysis:

Primary specialties required:
1. Cardiology (confidence: 0.90) - Elevated cardiac markers
2. Endocrinology (confidence: 0.75) - Thyroid dysfunction
3. Internal Medicine (confidence: 0.65) - General oversight

Urgency level: HIGH
Recommended routing: Multi-specialty approach with Cardiology lead

This case requires collaborative management due to overlapping symptoms and potential medication interactions."""

        # Patient communication
        elif "patient" in content or "explain" in content or "summary" in content:
            return """Patient-Friendly Summary:

Your test results show that your heart is under some stress, which is causing your chest pain. The blood tests detected proteins that indicate your heart muscle needs attention.

What this means:
- Your heart isn't getting enough blood flow
- This is treatable, but we need to act quickly
- You'll need some additional tests (like a heart scan)

Next steps:
- Our cardiology team will see you today
- We may need to do a procedure to look at your heart vessels
- In the meantime, please rest and avoid strenuous activity

Don't worry - this condition is very treatable when caught early. We're taking good care of you."""

        # Default response
        else:
            return """Based on the information provided, I recommend:

1. Comprehensive evaluation of presenting symptoms
2. Correlation with laboratory findings
3. Consideration of relevant medical history
4. Multi-disciplinary approach if needed

Further information may be required for definitive assessment.

Confidence: 0.70"""

    def _should_error(self) -> bool:
        """Determine if this call should simulate an error"""
        import random
        return random.random() < self.error_rate

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Rough token count estimation (4 chars ≈ 1 token)"""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return total_chars // 4

    def reset_stats(self):
        """Reset call statistics"""
        self.call_count = 0
        self.total_tokens = 0
        self.call_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "average_tokens_per_call": self.total_tokens / max(self.call_count, 1),
            "configured_latency_ms": self.latency_ms,
            "call_history_size": len(self.call_history)
        }
