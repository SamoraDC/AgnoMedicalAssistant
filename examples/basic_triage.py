"""Basic triage agent usage example."""
import asyncio
from src.agents import TriadorAgent
from src.core import AgentConfig


async def main():
    """Run basic triage example."""
    # Initialize agent
    config = AgentConfig(
        name="Triador",
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        enable_observability=False  # Set to True if you have Langfuse configured
    )

    agent = TriadorAgent(config)

    # Example case 1: Emergency
    print("=" * 80)
    print("CASE 1: Emergency Symptoms")
    print("=" * 80)

    async with agent:
        result = await agent.process({
            "symptoms": "severe chest pain radiating to left arm, shortness of breath, sweating",
            "age": 55,
            "gender": "male",
            "symptom_duration": "30 minutes",
            "severity_rating": 9
        })

        triage = result['triage_result']
        print(f"\nUrgency Level: {triage['urgency_level'].upper()}")
        print(f"Confidence: {triage['confidence_score']:.2f}")
        print(f"\nReasoning:\n{triage['reasoning']}")
        print(f"\nRecommended Actions:")
        for action in triage['recommended_actions']:
            print(f"  - {action}")

        if triage['red_flags']:
            print(f"\nRED FLAGS:")
            for flag in triage['red_flags']:
                print(f"  ⚠️  {flag}")

    # Example case 2: Non-urgent
    print("\n" + "=" * 80)
    print("CASE 2: Non-Urgent Symptoms")
    print("=" * 80)

    async with agent:
        result = await agent.process({
            "symptoms": "mild headache and stuffy nose for 2 days",
            "age": 28,
            "gender": "female",
            "symptom_duration": "2 days",
            "severity_rating": 3
        })

        triage = result['triage_result']
        print(f"\nUrgency Level: {triage['urgency_level'].upper()}")
        print(f"Confidence: {triage['confidence_score']:.2f}")
        print(f"\nRecommended Actions:")
        for action in triage['recommended_actions']:
            print(f"  - {action}")


if __name__ == "__main__":
    asyncio.run(main())
