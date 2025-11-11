# Setup Guide

## Prerequisites

### Required
- **Python 3.11+**: The project requires Python 3.11 or higher
- **UV Package Manager**: Modern Python package and project manager
- **Groq API Key**: For LLM access (get from [console.groq.com](https://console.groq.com/))

### Optional
- **Tesseract OCR**: For scanned document processing
- **Langfuse Account**: For observability (get from [langfuse.com](https://langfuse.com))

## Step-by-Step Installation

### 1. Install UV

**Linux/macOS**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows**:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Setup Project

```bash
# Clone repository
git clone <repository-url>
cd AgnoMedicalAssistant

# Create virtual environment and install dependencies
uv sync

# Verify installation
uv run python --version
```

### 3. Configure Environment

```bash
# Copy example environment file
cp config/.env.example .env

# Edit .env with your settings
nano .env  # or use your preferred editor
```

**Required Environment Variables**:
```bash
# Groq API (Required)
GROQ_API_KEY=your-groq-api-key-here

# Optional: Langfuse Observability
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_ENABLED=true
```

### 4. Install Optional Dependencies

**Tesseract OCR** (for scanned documents):

**Ubuntu/Debian**:
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-por
```

**macOS**:
```bash
brew install tesseract tesseract-lang
```

**Windows**:
- Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- Add to PATH or set `TESSERACT_CMD` in `.env`

### 5. Verify Installation

```bash
# Run tests
uv run pytest

# Start development server
uv run uvicorn src.api.app:app --reload

# Try example scripts
uv run python examples/basic_triage.py
```

## Directory Setup

The project will create these directories automatically:
- `.data/` - Database and vector store files
- `.data/uploads/` - Uploaded files
- `logs/` - Application logs
- `.swarm/` - Coordination memory

## Common Issues

### UV Not Found
```bash
# Add to PATH (Linux/macOS)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Groq API Errors
- Verify API key is correct
- Check [Groq Status](https://status.groq.com/)
- Ensure sufficient API credits

### Tesseract Not Found
```bash
# Set path in .env
TESSERACT_CMD=/usr/local/bin/tesseract
```

### Import Errors
```bash
# Reinstall dependencies
uv sync --reinstall
```

## Development Setup

```bash
# Install development dependencies
uv sync --extra dev

# Setup pre-commit hooks (optional)
uv run pre-commit install

# Run code quality checks
uv run black src/ tests/
uv run ruff check src/ tests/
uv run mypy src/
```

## Next Steps

1. Read [Usage Examples](../README.md#usage-examples)
2. Try the [example scripts](../examples/)
3. Check [API Documentation](http://localhost:8000/api/docs) after starting server
4. Review [Architecture Documentation](ARCHITECTURE.md)

## Support

For issues:
- Check [Common Issues](#common-issues)
- Review [GitHub Issues](repository-url/issues)
- Check UV documentation: [docs.astral.sh/uv](https://docs.astral.sh/uv/)
