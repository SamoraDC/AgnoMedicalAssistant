# Agno Medical Assistant

AI-powered medical assistant system with triage, RAG (Retrieval-Augmented Generation), and OCR capabilities.

## Features

- **Triador (Triage) Agent**: AI-powered symptom assessment with Chain-of-Thought reasoning
- **RAG System**: Medical knowledge retrieval using DuckDB vector storage
- **OCR Processing**: Document text extraction with PyMuPDF and Tesseract fallback
- **FastAPI Backend**: Production-ready API with security middleware
- **Langfuse Integration**: Observability and monitoring for AI operations

## Architecture

```
src/
├── agents/          # AI agents (Triador, etc.)
├── api/             # FastAPI application and routes
├── core/            # Base classes and exceptions
├── rag/             # RAG pipeline and vector store
├── ocr/             # OCR processing
├── config/          # Configuration management
└── utils/           # Utility functions

tests/
├── unit/            # Unit tests
└── integration/     # Integration tests

config/              # Configuration files (.env)
docs/                # Documentation
examples/            # Example usage scripts
```

## Quick Start

### Prerequisites

- Python 3.11+
- UV package manager
- Groq API key (for LLM access)
- Optional: Tesseract OCR for scanned documents

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd AgnoMedicalAssistant
```

2. **Install UV** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Create virtual environment and install dependencies**:
```bash
# UV automatically creates a virtual environment
uv sync
```

4. **Install additional dependencies as needed**:
```bash
# Core dependencies (already in pyproject.toml)
uv add agno groq langfuse fastapi uvicorn duckdb sentence-transformers

# OCR dependencies
uv add pymupdf pytesseract pillow pdf2image

# Development dependencies
uv add --dev pytest pytest-asyncio pytest-cov black ruff mypy
```

5. **Configure environment**:
```bash
cp config/.env.example .env
# Edit .env and add your API keys
```

6. **Set up Groq API key**:
- Get your API key from [Groq Console](https://console.groq.com/)
- Add to `.env`: `GROQ_API_KEY=your-key-here`

### Running the Application

**Start the FastAPI server**:
```bash
uv run uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

**Access the API**:
- API Docs: http://localhost:8000/api/docs
- Health Check: http://localhost:8000/health

## Usage Examples

### Triage Agent

```python
from src.agents import TriadorAgent
from src.core import AgentConfig

# Initialize agent
config = AgentConfig(
    name="Triador",
    model="llama-3.3-70b-versatile",
    enable_observability=True
)
agent = TriadorAgent(config)

# Perform triage
async with agent:
    result = await agent.process({
        "symptoms": "severe chest pain radiating to left arm",
        "age": 55,
        "gender": "male",
        "symptom_duration": "30 minutes",
        "severity_rating": 9
    })

    print(f"Urgency: {result['triage_result']['urgency_level']}")
    print(f"Reasoning: {result['triage_result']['reasoning']}")
    print(f"Actions: {result['triage_result']['recommended_actions']}")
```

### RAG System

```python
from src.rag import RAGPipeline

# Initialize RAG pipeline
rag = RAGPipeline()

# Ingest medical document
result = rag.ingest_document(
    text="Medical knowledge content...",
    metadata={"source": "medical_textbook", "chapter": "cardiology"},
    doc_id="cardio_001"
)

# Query the knowledge base
query_result = rag.query(
    query="What are the symptoms of myocardial infarction?",
    k=5
)

print(f"Found {query_result['num_results']} relevant documents")
for doc in query_result['results']:
    print(f"- {doc['content'][:100]}... (score: {doc['similarity_score']:.3f})")
```

### OCR Processing

```python
from src.ocr import OCRProcessor

# Initialize OCR processor
ocr = OCRProcessor()

# Process document
result = ocr.process("medical_report.pdf")

print(f"Extracted {result.page_count} pages using {result.method}")
print(f"Text: {result.text[:200]}...")
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src tests/

# Run specific test file
uv run pytest tests/unit/test_triador.py
```

### Code Quality

```bash
# Format code
uv run black src/ tests/

# Lint
uv run ruff check src/ tests/

# Type checking
uv run mypy src/
```

### Adding New Dependencies

```bash
# Add production dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Update dependencies
uv sync
```

## Configuration

All configuration is managed through environment variables (see `config/.env.example`):

- **Application**: Basic app settings
- **Security**: API keys and tokens
- **Groq**: LLM model configuration
- **Langfuse**: Observability settings (optional)
- **RAG**: Embedding and vector store settings
- **OCR**: Document processing configuration

## Project Structure Details

### Agents (`src/agents/`)
AI agents following Agno framework patterns:
- `triador.py`: Medical triage agent with CoT reasoning

### Core (`src/core/`)
- `base_agent.py`: Base class for all agents with LLM integration
- `exceptions.py`: Custom exception hierarchy

### RAG (`src/rag/`)
- `vectorstore.py`: DuckDB-based vector storage
- `pipeline.py`: Complete RAG pipeline with chunking and retrieval

### OCR (`src/ocr/`)
- `processor.py`: Document text extraction with fallback strategies

### API (`src/api/`)
- `app.py`: FastAPI application with middleware and error handling

### Configuration (`src/config/`)
- `settings.py`: Pydantic-based configuration management

## Observability

When Langfuse is enabled, the system tracks:
- LLM calls and token usage
- Agent performance metrics
- Error rates and patterns
- Response times

Access your Langfuse dashboard to monitor AI operations.

## Security

- Environment-based configuration (no hardcoded secrets)
- CORS middleware with configurable origins
- Trusted host middleware for production
- Input validation with Pydantic
- Secure error handling (no sensitive data in responses)

## License

MIT

## Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Documentation: See `docs/` directory

---

Built with ❤️ using [Agno Framework](https://github.com/agno-agi/agno)