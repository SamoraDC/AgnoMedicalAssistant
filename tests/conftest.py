"""
Pytest configuration and shared fixtures for medical assistant system tests

This module provides:
- Test environment setup
- Database fixtures
- Mock API clients
- Common test utilities
- Async test support
"""

import os
import sys
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ==================== Pytest Configuration ====================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (require external services)")
    config.addinivalue_line("markers", "security: Security and compliance tests")
    config.addinivalue_line("markers", "performance: Performance and load tests")
    config.addinivalue_line("markers", "e2e: End-to-end workflow tests")
    config.addinivalue_line("markers", "slow: Slow-running tests (>1 second)")
    config.addinivalue_line("markers", "requires_api: Tests requiring real API access")


# ==================== Async Support ====================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ==================== Test Environment ====================

@pytest.fixture(scope="session")
def test_env():
    """Setup test environment variables"""
    original_env = os.environ.copy()

    # Set test environment variables
    os.environ["ENV"] = "test"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY_TEST", "test-key")
    os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY_TEST", "test-key")
    os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY_TEST", "test-secret")
    os.environ["DATABASE_PATH"] = ":memory:"  # In-memory database for tests

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# ==================== Database Fixtures ====================

@pytest.fixture
def duckdb_connection():
    """Create in-memory DuckDB connection for testing"""
    import duckdb

    conn = duckdb.connect(":memory:")

    # Install and load VSS extension
    conn.execute("INSTALL vss")
    conn.execute("LOAD vss")

    # Create test tables
    conn.execute("""
        CREATE TABLE embeddings (
            id VARCHAR PRIMARY KEY,
            vector FLOAT[768],
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE cases (
            case_id VARCHAR PRIMARY KEY,
            patient_id VARCHAR,
            case_data JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    yield conn

    conn.close()


# ==================== Mock API Clients ====================

@pytest.fixture
def mock_groq_client():
    """Mock Groq API client"""
    from tests.mocks.mock_groq_api import MockGroqClient
    return MockGroqClient(latency_ms=50)


@pytest.fixture
def mock_langfuse_client():
    """Mock Langfuse tracking client"""
    from tests.mocks.mock_langfuse import MockLangfuseClient
    return MockLangfuseClient()


@pytest.fixture
def mock_acp_server():
    """Mock ACP server for agent communication"""
    from tests.mocks.mock_acp_server import MockACPServer

    server = MockACPServer(port=8001)
    server.start()

    yield server

    server.stop()


# ==================== Agent Fixtures ====================

@pytest.fixture
def triador_agent(mock_groq_client, duckdb_connection):
    """Create Triador agent instance"""
    # This will be implemented once agent code exists
    # from src.agents.triador import TriadorAgent
    # return TriadorAgent(groq_client=mock_groq_client, db=duckdb_connection)
    pass


@pytest.fixture
def cardiology_agent(mock_groq_client, duckdb_connection):
    """Create Cardiology specialist agent"""
    # from src.agents.specialists.cardiology import CardiologyAgent
    # return CardiologyAgent(groq_client=mock_groq_client, db=duckdb_connection)
    pass


@pytest.fixture
def validator_agent(mock_groq_client):
    """Create Validator agent instance"""
    # from src.agents.validator import ValidatorAgent
    # return ValidatorAgent(groq_client=mock_groq_client)
    pass


# ==================== Test Data Fixtures ====================

@pytest.fixture
def cardiology_case_data():
    """Sample cardiology case data"""
    return {
        "patient_id": "TEST_PATIENT_001",
        "age": 65,
        "gender": "M",
        "chief_complaint": "Chest pain and shortness of breath",
        "vital_signs": {
            "bp_systolic": 145,
            "bp_diastolic": 95,
            "heart_rate": 92,
            "respiratory_rate": 20,
            "temperature": 37.1,
            "oxygen_saturation": 94
        },
        "lab_results": {
            "troponin": 0.5,  # Elevated (normal <0.04)
            "bnp": 450,       # Elevated (normal <100)
            "cholesterol": 220,
            "hdl": 38,        # Low
            "ldl": 155,       # High
            "triglycerides": 180
        },
        "medications": [
            "Aspirin 81mg daily",
            "Lisinopril 10mg daily"
        ],
        "medical_history": [
            "Hypertension (5 years)",
            "Hyperlipidemia (3 years)"
        ]
    }


@pytest.fixture
def multi_specialty_case_data():
    """Complex case requiring multiple specialists"""
    return {
        "patient_id": "TEST_PATIENT_002",
        "age": 58,
        "gender": "F",
        "chief_complaint": "Fatigue, weight gain, depression",
        "vital_signs": {
            "bp_systolic": 130,
            "bp_diastolic": 85,
            "heart_rate": 68,
            "weight": 85,  # kg
            "bmi": 32.5
        },
        "lab_results": {
            "tsh": 12.5,      # Elevated (normal 0.5-5.0)
            "t4_free": 0.6,   # Low (normal 0.8-1.8)
            "glucose": 180,   # Elevated
            "hba1c": 8.2,     # Elevated (diabetic range)
            "vitamin_d": 15   # Low
        },
        "medications": [
            "Metformin 1000mg twice daily",
            "Levothyroxine 50mcg daily"
        ],
        "medical_history": [
            "Type 2 Diabetes (2 years)",
            "Hypothyroidism (newly diagnosed)",
            "Depression (6 months)"
        ]
    }


@pytest.fixture
def sample_medical_document(tmp_path):
    """Create sample medical document PDF"""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "lab_report.pdf"

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(100, 750, "Laboratory Test Results")
    c.drawString(100, 730, "Patient ID: TEST_PATIENT_001")
    c.drawString(100, 710, "Date: 2024-01-15")
    c.drawString(100, 680, "Troponin I: 0.5 ng/mL (High)")
    c.drawString(100, 660, "BNP: 450 pg/mL (High)")
    c.drawString(100, 640, "Total Cholesterol: 220 mg/dL")
    c.save()

    return str(pdf_path)


# ==================== Security Testing Fixtures ====================

@pytest.fixture
def phi_test_data():
    """Sample PHI data for anonymization testing"""
    return {
        "patient_name": "John Doe",
        "ssn": "123-45-6789",
        "date_of_birth": "1958-03-15",
        "phone": "(555) 123-4567",
        "email": "john.doe@example.com",
        "address": "123 Main St, Springfield, IL 62701",
        "medical_record_number": "MRN-123456"
    }


@pytest.fixture
def guardrails_config():
    """GuardRails configuration for testing"""
    return {
        "input_guardrails": {
            "detect_phi": True,
            "detect_prompt_injection": True,
            "max_input_length": 10000
        },
        "output_guardrails": {
            "prevent_phi_leak": True,
            "content_policy": True,
            "hallucination_detection": True
        },
        "phi_patterns": {
            "ssn": r"\d{3}-\d{2}-\d{4}",
            "phone": r"\(\d{3}\) \d{3}-\d{4}",
            "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}",
            "mrn": r"MRN-\d{6}"
        }
    }


# ==================== Performance Testing Fixtures ====================

@pytest.fixture
def performance_thresholds():
    """Performance benchmark thresholds"""
    return {
        "groq_inference_p95": 100,  # ms
        "triador_analysis": 500,    # ms
        "specialist_analysis": 2000, # ms
        "validator_synthesis": 1000, # ms
        "end_to_end": 5000,         # ms
        "vector_search": 100,        # ms
        "throughput_cases_per_hour": 100
    }


@pytest.fixture
def load_test_config():
    """Configuration for load testing"""
    return {
        "concurrent_users": 10,
        "cases_per_user": 5,
        "ramp_up_time": 30,  # seconds
        "test_duration": 300, # seconds
        "think_time": 5       # seconds between actions
    }


# ==================== Cleanup ====================

@pytest.fixture(autouse=True)
def cleanup_test_artifacts(tmp_path):
    """Automatically cleanup test artifacts after each test"""
    yield
    # Cleanup code here if needed


# ==================== Test Utilities ====================

@pytest.fixture
def assert_latency():
    """Utility to assert operation latency"""
    def _assert_latency(duration_ms: float, threshold_ms: float, operation: str):
        assert duration_ms < threshold_ms, \
            f"{operation} took {duration_ms}ms, expected <{threshold_ms}ms"
    return _assert_latency


@pytest.fixture
def assert_no_phi():
    """Utility to assert no PHI in text"""
    import re

    def _assert_no_phi(text: str):
        phi_patterns = {
            "ssn": r"\d{3}-\d{2}-\d{4}",
            "phone": r"\(\d{3}\) \d{3}-\d{4}",
            "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}",
            "mrn": r"MRN-\d{6}"
        }

        for pattern_name, pattern in phi_patterns.items():
            matches = re.findall(pattern, text)
            assert not matches, f"PHI detected ({pattern_name}): {matches}"

    return _assert_no_phi
