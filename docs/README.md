# Agno Medical Assistant

A sophisticated multi-agent medical analysis system built with Agno framework, featuring HIPAA/LGPD compliance, real-time multimodal processing, and collaborative AI agents for clinical decision support.

## Quick Start

### Prerequisites

- Python 3.13 or higher
- UV package manager
- Groq API access (for LPU inference)
- Node.js 18+ (for frontend)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AgnoMedicalAssistant.git
cd AgnoMedicalAssistant

# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project dependencies
uv add fastapi uvicorn agno groq
uv add python-multipart pymupdf pytesseract
uv add duckdb duckdb-engine
uv add guardrails-ai edge-tts
uv add python-jose[cryptography] passlib[bcrypt]
uv add websockets aiortc  # For WebRTC support
```

### Configuration

Create a `.env` file in the project root:

```env
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# Security
SECRET_KEY=your-secret-key-for-jwt-signing
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
DATABASE_URL=duckdb:///data/medical.db

# HIPAA/LGPD Compliance
ENCRYPT_AT_REST=true
AUDIT_LOG_ENABLED=true
PHI_ANONYMIZATION=true

# Agent Configuration
MAX_CONCURRENT_AGENTS=10
DEBATE_TIMEOUT_SECONDS=30
CONFIDENCE_THRESHOLD=0.85
```

### Running the Application

```bash
# Start the FastAPI backend
uvicorn src.main:app --reload --port 8000

# In another terminal, start the frontend (if available)
cd frontend
npm install
npm run dev
```

## System Overview

The Agno Medical Assistant implements a hierarchical multi-agent architecture designed for:

- **Low-latency medical analysis** using Groq LPU inference
- **Multimodal data processing** (PDFs, lab results, images, clinical text)
- **Collaborative reasoning** through Agent Communication Protocol (ACP)
- **HIPAA/LGPD compliance** with encryption and audit trails
- **Real-time teleconsultation** via WebRTC integration

### Key Features

- **Mixture of Experts (MoE)** agent architecture
- **OCR processing** with PyMuPDF and Tesseract
- **Retrieval-Augmented Generation (RAG)** with DuckDB vector store
- **Structured agent debates** for conflict resolution
- **Text-to-speech** medical report generation
- **Case-based reasoning** with semantic memory

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React/TypeScript)              │
│              Patient Dashboard & Teleconsultation            │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTPS/WebSocket
┌─────────────────────▼───────────────────────────────────────┐
│                  FastAPI Backend (OAuth2/JWT)                │
│                    GuardRails AI (PHI Filter)                │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   Agno Agent Orchestration                   │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐         │
│  │  Triager   │→ │ Specialists │→ │  Validator   │         │
│  │   Agent    │  │   (MoE)     │  │    Agent     │         │
│  └────────────┘  └─────────────┘  └──────────────┘         │
│                          │                 │                 │
│                          ▼                 ▼                 │
│                  ┌───────────────────────────┐               │
│                  │  Communicator Agent       │               │
│                  │  (Report + Audio)         │               │
│                  └───────────────────────────┘               │
└───────────────────────────────────┬─────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────┐
│  Groq LPU Inference (Sub-second latency)                     │
│  DuckDB Vector Store (RAG + Case Memory)                     │
│  OCR Pipeline (PyMuPDF + Tesseract)                          │
└───────────────────────────────────────────────────────────────┘
```

## Core Components

### Agent Hierarchy

1. **Triager Agent**: Initial case analysis, OCR processing, specialty routing
2. **Specialist Agents**: Domain experts (Cardiology, Neurology, Endocrinology, etc.)
3. **Validator Agent**: Conflict detection and resolution orchestration
4. **Communicator Agent**: Patient-friendly report generation and audio synthesis

### Technology Stack

- **Backend**: FastAPI, Python 3.13, Agno framework
- **Inference**: Groq LPU (Language Processing Units)
- **Database**: DuckDB with VSS extension for vector storage
- **OCR**: PyMuPDF with Tesseract fallback
- **Security**: OAuth2, JWT, GuardRails AI
- **Audio**: Edge-TTS for text-to-speech
- **Real-time Communication**: WebRTC for teleconsultation

## Documentation Index

- [Architecture Documentation](./architecture.md) - System design and agent patterns
- [API Reference](./api-reference.md) - FastAPI endpoint specifications
- [Developer Guide](./developer-guide.md) - Extending the system
- [Deployment Guide](./deployment-guide.md) - Production deployment procedures

## Security & Compliance

This system is designed with HIPAA and LGPD compliance as core requirements:

- All PHI/PII is encrypted at rest and in transit
- GuardRails AI automatically detects and anonymizes sensitive data
- Complete audit trails for all data access
- OAuth2 authentication with JWT tokens
- Secure WebRTC connections for teleconsultation

## Performance Characteristics

- **Inference Latency**: <1 second per specialist agent (Groq LPU)
- **Agent Debate Resolution**: ~5-10 seconds for multi-turn debates
- **OCR Processing**: 2-5 seconds per page (PyMuPDF)
- **Case Similarity Search**: <100ms (DuckDB vector queries)
- **End-to-End Analysis**: ~30-60 seconds for complex cases

## Support & Contributing

- **Issues**: Report bugs on GitHub Issues
- **Documentation**: Full docs in `/docs` directory
- **Contributing**: See CONTRIBUTING.md for guidelines

## License

[Your License Here]

---

**⚠️ MEDICAL DISCLAIMER**: This system is designed as a clinical decision support tool. All outputs must be reviewed by qualified medical professionals. This software is not a substitute for professional medical judgment.
