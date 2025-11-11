# Developer Guide

Complete guide for extending and customizing the Agno Medical Assistant system.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Adding New Specialist Agents](#adding-new-specialist-agents)
4. [Extending RAG Knowledge Bases](#extending-rag-knowledge-bases)
5. [Customizing ACP Debates](#customizing-acp-debates)
6. [Testing Procedures](#testing-procedures)
7. [Performance Optimization](#performance-optimization)
8. [Security Best Practices](#security-best-practices)

## Development Setup

### Environment Setup with UV

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/yourusername/AgnoMedicalAssistant.git
cd AgnoMedicalAssistant

# Create virtual environment
uv venv --python 3.13

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
uv add fastapi uvicorn agno groq
uv add python-multipart pymupdf pytesseract
uv add duckdb duckdb-engine sentence-transformers
uv add guardrails-ai edge-tts
uv add python-jose[cryptography] passlib[bcrypt]
uv add websockets aiortc

# Development dependencies
uv add --dev pytest pytest-asyncio pytest-cov
uv add --dev black isort mypy ruff
uv add --dev httpx  # For API testing
uv add --dev faker  # For generating test data
```

### Environment Variables

Create `.env` file:

```bash
# Copy template
cp .env.example .env

# Edit with your values
nano .env
```

Required variables:
```env
GROQ_API_KEY=your_groq_api_key
SECRET_KEY=$(openssl rand -hex 32)
DATABASE_URL=duckdb:///data/medical.db
TESSERACT_PATH=/usr/bin/tesseract  # Adjust for your system
```

### Pre-commit Hooks

```bash
# Install pre-commit
uv add --dev pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        language_version: python3.13

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.291
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

## Project Structure

```
AgnoMedicalAssistant/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                 # Base agent class
│   │   ├── triager.py              # Triager agent
│   │   ├── specialists/
│   │   │   ├── __init__.py
│   │   │   ├── cardiology.py       # Cardiology specialist
│   │   │   ├── neurology.py        # Neurology specialist
│   │   │   └── endocrinology.py    # Endocrinology specialist
│   │   ├── validator.py            # Validator agent
│   │   └── communicator.py         # Communicator agent
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application
│   │   ├── routes/
│   │   │   ├── auth.py             # Authentication endpoints
│   │   │   ├── cases.py            # Case management endpoints
│   │   │   ├── agents.py           # Agent management endpoints
│   │   │   └── reports.py          # Report generation endpoints
│   │   └── dependencies.py         # FastAPI dependencies
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Configuration management
│   │   ├── security.py             # Security utilities
│   │   └── orchestration.py        # Agent orchestration logic
│   │
│   ├── acp/
│   │   ├── __init__.py
│   │   ├── protocol.py             # ACP implementation
│   │   ├── debate.py               # Debate orchestration
│   │   └── schemas.py              # ACP message schemas
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py           # Embedding generation
│   │   ├── retrieval.py            # Vector search
│   │   └── knowledge_base.py       # Knowledge base management
│   │
│   ├── ocr/
│   │   ├── __init__.py
│   │   ├── processor.py            # OCR orchestration
│   │   ├── pymupdf_handler.py      # PyMuPDF implementation
│   │   └── tesseract_handler.py    # Tesseract fallback
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── case_memory.py          # Case-based reasoning
│   │   └── embeddings_store.py     # DuckDB vector operations
│   │
│   └── utils/
│       ├── __init__.py
│       ├── groq_client.py          # Groq API wrapper
│       ├── phi_detection.py        # GuardRails integration
│       └── audio_generator.py      # Edge-TTS wrapper
│
├── tests/
│   ├── unit/
│   │   ├── test_agents.py
│   │   ├── test_acp.py
│   │   └── test_rag.py
│   ├── integration/
│   │   ├── test_api.py
│   │   └── test_workflows.py
│   └── e2e/
│       └── test_full_case.py
│
├── data/
│   ├── medical.db                  # DuckDB database
│   └── knowledge/                  # Knowledge base documents
│
├── docs/
│   ├── README.md
│   ├── architecture.md
│   ├── api-reference.md
│   ├── developer-guide.md          # This file
│   └── deployment-guide.md
│
├── scripts/
│   ├── init_db.py                  # Initialize database
│   ├── load_knowledge.py           # Load knowledge base
│   └── benchmark_agents.py         # Performance benchmarking
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── main.py
```

## Adding New Specialist Agents

### Step 1: Define Agent Class

Create `src/agents/specialists/dermatology.py`:

```python
from typing import Dict, List, Any
from src.agents.base import BaseSpecialistAgent
from src.rag.retrieval import RAGRetriever
from src.utils.groq_client import GroqClient

class DermatologyAgent(BaseSpecialistAgent):
    """
    Specialist agent for dermatological analysis.

    Capabilities:
    - Skin lesion classification
    - Treatment recommendation
    - Drug interaction checking
    """

    def __init__(
        self,
        agent_id: str,
        groq_client: GroqClient,
        rag_retriever: RAGRetriever,
        specialty: str = "dermatology"
    ):
        super().__init__(
            agent_id=agent_id,
            specialty=specialty,
            groq_client=groq_client,
            rag_retriever=rag_retriever
        )

        # Define agent-specific tools
        self.tools = [
            self.classify_lesion,
            self.check_drug_interactions,
            self.recommend_treatment
        ]

    async def analyze_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform dermatological analysis on case data.

        Args:
            case_data: Dictionary containing patient data, images, notes

        Returns:
            Analysis results with hypothesis, confidence, and recommendations
        """
        # Extract relevant data
        symptoms = case_data.get("symptoms", [])
        images = case_data.get("images", [])
        medications = case_data.get("medications", [])

        # Retrieve relevant knowledge from RAG
        knowledge_context = await self.rag_retriever.search(
            query=f"dermatology {' '.join(symptoms)}",
            specialty="dermatology",
            top_k=5
        )

        # Build prompt for Groq inference
        prompt = self._build_analysis_prompt(
            symptoms=symptoms,
            images=images,
            knowledge=knowledge_context
        )

        # Run inference
        response = await self.groq_client.chat_completion(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert dermatologist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2048
        )

        # Parse response
        analysis = self._parse_response(response)

        # Check drug interactions
        interactions = await self.check_drug_interactions(medications)
        if interactions:
            analysis["warnings"] = interactions

        # Calculate confidence score
        confidence = self._calculate_confidence(analysis)

        return {
            "agent_id": self.agent_id,
            "agent_type": self.specialty,
            "hypothesis": analysis["diagnosis"],
            "recommendations": analysis["treatment_plan"],
            "confidence": confidence,
            "evidence_sources": [k["source"] for k in knowledge_context],
            "warnings": analysis.get("warnings", [])
        }

    def _build_analysis_prompt(
        self,
        symptoms: List[str],
        images: List[str],
        knowledge: List[Dict]
    ) -> str:
        """Build structured prompt for Groq inference."""
        prompt = "## Dermatological Case Analysis\n\n"
        prompt += "**Symptoms:**\n"
        for symptom in symptoms:
            prompt += f"- {symptom}\n"

        if images:
            prompt += f"\n**Images:** {len(images)} skin lesion images provided\n"

        prompt += "\n**Relevant Medical Knowledge:**\n"
        for kb in knowledge:
            prompt += f"- {kb['excerpt']} (Source: {kb['source']})\n"

        prompt += "\n**Task:** Provide a differential diagnosis, confidence score, and treatment recommendations."

        return prompt

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        # Implement parsing logic
        # In production, use structured output or JSON mode
        return {
            "diagnosis": "Extracted diagnosis",
            "treatment_plan": ["Treatment 1", "Treatment 2"],
            "differential": ["Alternative 1", "Alternative 2"]
        }

    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score based on analysis."""
        # Implement confidence scoring logic
        return 0.85

    async def classify_lesion(self, image_path: str) -> Dict[str, Any]:
        """Classify skin lesion type."""
        # Implement lesion classification
        pass

    async def check_drug_interactions(self, medications: List[str]) -> List[Dict]:
        """Check for drug interactions."""
        # Query drug interaction database
        pass

    async def recommend_treatment(self, diagnosis: str) -> List[str]:
        """Generate treatment recommendations."""
        # Query treatment guidelines
        pass
```

### Step 2: Register Agent in Triager

Edit `src/agents/triager.py`:

```python
SPECIALTY_MAPPING = {
    "cardiology": ["chest pain", "heart", "cardiac", "ecg"],
    "neurology": ["headache", "seizure", "stroke", "neurological"],
    "endocrinology": ["diabetes", "thyroid", "hormone"],
    "dermatology": ["rash", "skin lesion", "itching", "melanoma"],  # Add this
    # ... other specialties
}

async def route_to_specialists(self, symptoms: List[str]) -> List[str]:
    """Route case to appropriate specialists based on symptoms."""
    relevant_specialties = set()

    for symptom in symptoms:
        for specialty, keywords in SPECIALTY_MAPPING.items():
            if any(keyword in symptom.lower() for keyword in keywords):
                relevant_specialties.add(specialty)

    return list(relevant_specialties)
```

### Step 3: Add to Orchestration

Edit `src/core/orchestration.py`:

```python
from src.agents.specialists.dermatology import DermatologyAgent

AGENT_REGISTRY = {
    "cardiology": CardiologyAgent,
    "neurology": NeurologyAgent,
    "endocrinology": EndocrinologyAgent,
    "dermatology": DermatologyAgent,  # Add this
    # ... other specialists
}
```

### Step 4: Create Tests

Create `tests/unit/test_dermatology_agent.py`:

```python
import pytest
from src.agents.specialists.dermatology import DermatologyAgent
from tests.fixtures import mock_groq_client, mock_rag_retriever

@pytest.mark.asyncio
async def test_dermatology_agent_analysis():
    """Test dermatology agent case analysis."""
    agent = DermatologyAgent(
        agent_id="derma_001",
        groq_client=mock_groq_client(),
        rag_retriever=mock_rag_retriever()
    )

    case_data = {
        "symptoms": ["red itchy rash on arms", "scaling skin"],
        "images": ["rash_image_1.jpg"],
        "medications": ["lisinopril", "metformin"]
    }

    result = await agent.analyze_case(case_data)

    assert result["agent_type"] == "dermatology"
    assert "hypothesis" in result
    assert result["confidence"] > 0.5
    assert len(result["recommendations"]) > 0
```

### Step 5: Load Specialty Knowledge

Create script `scripts/load_dermatology_knowledge.py`:

```python
from src.rag.knowledge_base import KnowledgeBaseManager

def load_dermatology_knowledge():
    """Load dermatology knowledge into RAG system."""
    kb_manager = KnowledgeBaseManager()

    # Load guidelines
    kb_manager.add_document(
        specialty="dermatology",
        source="American Academy of Dermatology",
        file_path="data/knowledge/aad_guidelines.pdf"
    )

    # Load research papers
    kb_manager.bulk_load(
        directory="data/knowledge/dermatology_papers/",
        specialty="dermatology"
    )

    print("Dermatology knowledge loaded successfully")

if __name__ == "__main__":
    load_dermatology_knowledge()
```

## Extending RAG Knowledge Bases

### Adding Documents to Existing Specialty

```python
from src.rag.knowledge_base import KnowledgeBaseManager

kb_manager = KnowledgeBaseManager()

# Single document
kb_manager.add_document(
    specialty="cardiology",
    source="Journal of American College of Cardiology",
    file_path="data/knowledge/jacc_2024.pdf",
    metadata={
        "publication_date": "2024-01-15",
        "doi": "10.1016/j.jacc.2024.01.001"
    }
)

# Bulk load
kb_manager.bulk_load(
    directory="data/knowledge/cardiology_updates/",
    specialty="cardiology",
    file_pattern="*.pdf"
)
```

### Custom Embedding Models

By default, the system uses `sentence-transformers/all-MiniLM-L6-v2`. To use a different model:

```python
# src/rag/embeddings.py

from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
```

Update configuration:
```env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIMENSION=768
```

### Reindexing Knowledge Base

```bash
# Rebuild vector indices after bulk updates
python scripts/reindex_knowledge_base.py --specialty cardiology

# Full reindex
python scripts/reindex_knowledge_base.py --all
```

### Knowledge Base Quality Metrics

Monitor RAG performance:

```python
from src.rag.metrics import RAGMetrics

metrics = RAGMetrics()

# Evaluate retrieval quality
results = metrics.evaluate_retrieval(
    test_queries="data/test_queries.json",
    ground_truth="data/ground_truth.json"
)

print(f"Precision@5: {results['precision_at_5']}")
print(f"Recall@10: {results['recall_at_10']}")
print(f"MRR: {results['mean_reciprocal_rank']}")
```

## Customizing ACP Debates

### Debate Configuration

Edit `src/acp/debate.py`:

```python
DEBATE_CONFIG = {
    "max_turns": 5,                    # Maximum debate rounds
    "timeout_seconds": 30,             # Total debate timeout
    "consensus_threshold": 0.8,        # Agreement threshold
    "evidence_required": True,         # Require evidence citations
    "escalate_on_timeout": True        # Escalate to human if timeout
}
```

### Custom Debate Strategies

```python
from src.acp.debate import DebateOrchestrator

class CustomDebateStrategy(DebateOrchestrator):
    """Custom debate resolution strategy."""

    async def resolve_conflict(
        self,
        agents: List[BaseAgent],
        conflict: Conflict
    ) -> DebateResult:
        """
        Implement custom conflict resolution logic.

        Example: Weighted voting based on agent confidence scores.
        """
        votes = []

        for agent in agents:
            # Get agent's position with evidence
            position = await agent.state_position(conflict)
            votes.append({
                "agent_id": agent.id,
                "position": position["stance"],
                "confidence": position["confidence"],
                "evidence": position["evidence"]
            })

        # Weighted voting
        total_weight = sum(v["confidence"] for v in votes)
        weighted_votes = {
            v["position"]: v["confidence"] / total_weight
            for v in votes
        }

        # Determine consensus
        winner = max(weighted_votes, key=weighted_votes.get)

        if weighted_votes[winner] >= self.consensus_threshold:
            return DebateResult(
                consensus_reached=True,
                resolution=winner,
                confidence=weighted_votes[winner],
                supporting_evidence=[v["evidence"] for v in votes if v["position"] == winner]
            )
        else:
            return DebateResult(
                consensus_reached=False,
                escalate_to_human=True,
                reason="No clear consensus after weighted voting"
            )
```

### Logging Debate Transcripts

All debates are automatically logged for auditing:

```python
# src/acp/protocol.py

async def send_acp_message(
    self,
    sender: str,
    recipient: str,
    message_type: str,
    content: Dict[str, Any]
) -> None:
    """Send ACP message and log for audit trail."""

    message = ACPMessage(
        sender=sender,
        recipient=recipient,
        message_type=message_type,
        content=content,
        timestamp=datetime.utcnow()
    )

    # Log to database
    await self.db.audit_log.insert({
        "type": "acp_message",
        "sender": sender,
        "recipient": recipient,
        "content": message.dict(),
        "timestamp": message.timestamp
    })

    # Send message
    await self.transport.send(message)
```

## Testing Procedures

### Unit Tests

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific agent tests
pytest tests/unit/test_agents.py::TestCardiologyAgent -v

# With coverage
pytest tests/unit/ --cov=src --cov-report=html
```

### Integration Tests

```bash
# Full integration test suite
pytest tests/integration/ -v

# Test API endpoints
pytest tests/integration/test_api.py -v
```

### End-to-End Tests

```bash
# Complete case workflow
pytest tests/e2e/test_full_case.py -v -s
```

### Test Fixtures

Create reusable test fixtures in `tests/fixtures.py`:

```python
import pytest
from faker import Faker

fake = Faker()

@pytest.fixture
def mock_case_data():
    """Generate mock case data for testing."""
    return {
        "patient_id": f"pat_{fake.uuid4()}",
        "symptoms": ["chest pain", "shortness of breath"],
        "lab_results": {
            "troponin": 0.15,  # Elevated
            "bnp": 450
        },
        "medications": ["aspirin", "metformin"],
        "priority": "high"
    }

@pytest.fixture
def mock_groq_client():
    """Mock Groq client for testing."""
    class MockGroqClient:
        async def chat_completion(self, **kwargs):
            return "Mock LLM response"

    return MockGroqClient()
```

### Performance Benchmarks

```bash
# Run performance benchmarks
python scripts/benchmark_agents.py

# Expected results:
# - Triager agent: < 1.5s
# - Specialist agents: < 1.0s each
# - Validator: < 2.0s
# - Communicator: < 1.0s
# - Total pipeline: < 60s
```

## Performance Optimization

### Caching Strategies

```python
from functools import lru_cache
from cachetools import TTLCache

# Cache RAG results (5 minute TTL)
rag_cache = TTLCache(maxsize=1000, ttl=300)

async def cached_rag_search(query: str, specialty: str):
    cache_key = f"{specialty}:{hash(query)}"

    if cache_key in rag_cache:
        return rag_cache[cache_key]

    results = await rag_retriever.search(query, specialty)
    rag_cache[cache_key] = results

    return results
```

### Parallel Agent Execution

```python
import asyncio

async def execute_specialists_parallel(
    specialists: List[BaseSpecialistAgent],
    case_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Execute multiple specialists in parallel."""

    tasks = [
        specialist.analyze_case(case_data)
        for specialist in specialists
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions
    valid_results = [
        r for r in results if not isinstance(r, Exception)
    ]

    return valid_results
```

### Database Optimization

```sql
-- Optimize vector search with HNSW index
CREATE INDEX idx_knowledge_embedding
ON medical_knowledge
USING HNSW (embedding)
WITH (m = 16, ef_construction = 200);

-- Partition case_memory by date
CREATE TABLE case_memory_2024 PARTITION OF case_memory
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

## Security Best Practices

### Input Validation

```python
from pydantic import BaseModel, validator, Field

class CaseCreateRequest(BaseModel):
    patient_id: str = Field(..., regex=r'^pat_[a-zA-Z0-9]+$')
    priority: str = Field(..., regex=r'^(low|medium|high|critical)$')
    clinical_notes: str = Field(..., max_length=10000)

    @validator('clinical_notes')
    def check_phi(cls, v):
        """Validate that obvious PHI is not present."""
        phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{10}\b',              # Phone
        ]

        for pattern in phi_patterns:
            if re.search(pattern, v):
                raise ValueError("Potential PHI detected in notes")

        return v
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/cases")
@limiter.limit("10/minute")
async def create_case(request: CaseCreateRequest):
    # ...
```

### Secrets Management

Never hardcode secrets. Use environment variables or secret managers:

```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Azure Key Vault example
credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://myvault.vault.azure.net", credential=credential)

GROQ_API_KEY = client.get_secret("groq-api-key").value
```

---

**Last Updated**: 2025-11-11
**Maintained By**: Development Team
