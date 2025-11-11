# Comprehensive Testing Strategy - Medical Assistant System

## Executive Summary

This document outlines a comprehensive testing strategy for the multi-agent medical assistant system built with Agno, focusing on HIPAA/LGPD compliance, low-latency performance, and multi-agent coordination reliability.

**Coverage Target**: 90%+ across all test categories
**CI/CD Integration**: Automated testing on every commit with quality gates
**Test Philosophy**: Test-Driven Development (TDD) with security-first approach

---

## 1. Unit Testing Strategy

### 1.1 Agent Behavior Tests

**Objective**: Validate individual agent logic and decision-making

#### 1.1.1 Triador Agent Tests
```python
# tests/unit/agents/test_triador_agent.py
- test_case_classification_accuracy()
  - Validates specialty routing (Cardiology, Neurology, Endocrinology)
  - Tests confidence score calculation
  - Validates multi-specialty detection

- test_ocr_quality_assessment()
  - Tests OCR quality scoring
  - Validates fallback mechanism triggering
  - Tests preprocessing pipeline selection

- test_case_prioritization()
  - Tests critical case flagging
  - Validates urgency scoring
  - Tests escalation triggers
```

#### 1.1.2 Specialist Agent Tests
```python
# tests/unit/agents/test_specialist_agents.py
- test_cardiology_agent_analysis()
  - Tests cardiac pattern recognition
  - Validates medication contraindication detection
  - Tests confidence scoring for diagnoses

- test_neurology_agent_analysis()
  - Tests neurological symptom analysis
  - Validates interaction with other specialties
  - Tests hypothesis generation

- test_endocrinology_agent_analysis()
  - Tests hormonal pattern recognition
  - Validates lab result interpretation
  - Tests differential diagnosis generation
```

#### 1.1.3 Validator Agent Tests
```python
# tests/unit/agents/test_validator_agent.py
- test_conflict_detection()
  - Tests medication contradiction identification
  - Validates diagnostic disagreement detection
  - Tests confidence threshold validation

- test_acp_debate_initiation()
  - Tests ACP message formatting
  - Validates debate participant selection
  - Tests debate timeout handling

- test_consensus_synthesis()
  - Tests final recommendation generation
  - Validates evidence aggregation
  - Tests uncertainty quantification
```

#### 1.1.4 Communicator Agent Tests
```python
# tests/unit/agents/test_communicator_agent.py
- test_medical_jargon_simplification()
  - Tests technical term translation
  - Validates readability scoring
  - Tests patient-friendly language generation

- test_report_structure_generation()
  - Tests section organization
  - Validates completeness checks
  - Tests formatting consistency

- test_audio_script_generation()
  - Tests Edge-TTS script formatting
  - Validates pronunciation guides
  - Tests pacing markers
```

### 1.2 OCR Accuracy Tests

```python
# tests/unit/ocr/test_ocr_pipeline.py
- test_pymupdf_extraction_quality()
  - Tests text extraction from clean PDFs
  - Validates layout preservation
  - Tests multi-column document handling

- test_tesseract_fallback_mechanism()
  - Tests quality threshold triggering
  - Validates preprocessing strategies
  - Tests accuracy improvement verification

- test_ocr_error_handling()
  - Tests corrupted document handling
  - Validates unsupported format detection
  - Tests graceful degradation

- test_handwriting_recognition()
  - Tests handwritten note extraction
  - Validates confidence scoring
  - Tests hybrid digital/handwritten documents
```

### 1.3 RAG Retrieval Precision Tests

```python
# tests/unit/rag/test_rag_pipeline.py
- test_embedding_generation()
  - Tests vector quality for medical texts
  - Validates dimensionality consistency
  - Tests batch processing efficiency

- test_similarity_search_accuracy()
  - Tests relevant document retrieval (precision@k)
  - Validates ranking quality (NDCG)
  - Tests query reformulation

- test_context_window_optimization()
  - Tests chunk size impact on relevance
  - Validates overlap strategy
  - Tests token budget management

- test_medical_terminology_handling()
  - Tests synonym recognition
  - Validates abbreviation expansion
  - Tests drug name variations
```

### 1.4 PII/PHI Anonymization Tests

```python
# tests/unit/security/test_guardrails.py
- test_phi_detection_accuracy()
  - Tests patient name detection (100% recall)
  - Validates date masking (DOB, appointment dates)
  - Tests ID number redaction (MRN, SSN)

- test_phi_anonymization_quality()
  - Tests consistent pseudonymization
  - Validates de-identification completeness
  - Tests re-identification resistance

- test_false_positive_minimization()
  - Tests medical term preservation
  - Validates context-aware detection
  - Tests common false positive scenarios

- test_output_guardrails()
  - Tests response sanitization
  - Validates PHI leak prevention
  - Tests audit log completeness
```

**Coverage Target**: 95%+ for security-critical components

---

## 2. Integration Testing Strategy

### 2.1 Agent Communication via ACP Tests

```python
# tests/integration/acp/test_agent_communication.py
- test_acp_message_exchange()
  - Tests REST API message delivery
  - Validates message format compliance
  - Tests asynchronous message handling

- test_debate_protocol_flow()
  - Tests multi-turn debate sequences
  - Validates evidence exchange format
  - Tests debate resolution logic

- test_agent_coordination_timing()
  - Tests parallel agent execution
  - Validates synchronization points
  - Tests timeout and retry mechanisms

- test_acp_error_recovery()
  - Tests connection failure handling
  - Validates message queue integrity
  - Tests graceful degradation
```

### 2.2 Groq API Integration Tests

```python
# tests/integration/groq/test_groq_integration.py
- test_low_latency_inference()
  - Tests <100ms response times (p95)
  - Validates streaming response handling
  - Tests batch inference optimization

- test_model_selection_strategy()
  - Tests model routing by task type
  - Validates fallback model selection
  - Tests cost optimization

- test_rate_limit_handling()
  - Tests rate limit detection
  - Validates exponential backoff
  - Tests request queuing

- test_groq_error_handling()
  - Tests API timeout handling
  - Validates malformed response recovery
  - Tests service unavailability fallback
```

### 2.3 Langfuse Tracking Integration Tests

```python
# tests/integration/langfuse/test_langfuse_tracking.py
- test_trace_generation()
  - Tests complete workflow tracing
  - Validates span hierarchy integrity
  - Tests metadata attachment

- test_token_usage_tracking()
  - Tests accurate token counting
  - Validates cost calculation
  - Tests aggregation by agent/user

- test_performance_metrics_collection()
  - Tests latency measurement accuracy
  - Validates throughput tracking
  - Tests error rate calculation

- test_audit_trail_completeness()
  - Tests HIPAA-compliant logging
  - Validates PHI exclusion from logs
  - Tests log retention policies
```

### 2.4 DuckDB Vector Operations Tests

```python
# tests/integration/duckdb/test_vector_operations.py
- test_embedding_storage_retrieval()
  - Tests bulk embedding insertion
  - Validates indexing performance
  - Tests concurrent read/write operations

- test_vss_extension_functionality()
  - Tests similarity search performance
  - Validates distance metric accuracy
  - Tests filtering combined with similarity

- test_case_similarity_search()
  - Tests historical case retrieval
  - Validates relevance ranking
  - Tests multi-criteria search

- test_database_performance_optimization()
  - Tests query plan efficiency
  - Validates index usage
  - Tests memory footprint
```

**Coverage Target**: 85%+ for integration paths

---

## 3. Security and Compliance Testing

### 3.1 HIPAA Compliance Tests

```python
# tests/security/compliance/test_hipaa_compliance.py
- test_authentication_requirements()
  - Tests OAuth2 flow implementation
  - Validates JWT token expiration (15 min)
  - Tests refresh token rotation

- test_authorization_controls()
  - Tests role-based access control (RBAC)
  - Validates patient data isolation
  - Tests privilege escalation prevention

- test_audit_trail_requirements()
  - Tests access logging completeness
  - Validates tamper-proof log storage
  - Tests audit log retention (6 years)

- test_data_encryption()
  - Tests TLS 1.3 enforcement
  - Validates AES-256 at-rest encryption
  - Tests key rotation procedures
```

### 3.2 LGPD Compliance Tests

```python
# tests/security/compliance/test_lgpd_compliance.py
- test_data_subject_rights()
  - Tests right to access (data export)
  - Validates right to deletion
  - Tests data portability

- test_consent_management()
  - Tests explicit consent capture
  - Validates consent withdrawal
  - Tests purpose limitation

- test_cross_border_data_transfer()
  - Tests data localization requirements
  - Validates transfer mechanisms
  - Tests data residency compliance
```

### 3.3 Authentication Security Tests

```python
# tests/security/auth/test_authentication_security.py
- test_password_security()
  - Tests minimum complexity requirements
  - Validates bcrypt hashing (cost factor 12)
  - Tests password breach detection

- test_session_management()
  - Tests session timeout enforcement
  - Validates concurrent session handling
  - Tests session fixation prevention

- test_multi_factor_authentication()
  - Tests TOTP implementation
  - Validates backup code generation
  - Tests MFA bypass prevention

- test_brute_force_protection()
  - Tests account lockout (5 failed attempts)
  - Validates CAPTCHA integration
  - Tests rate limiting effectiveness
```

### 3.4 Data Encryption Tests

```python
# tests/security/encryption/test_data_encryption.py
- test_at_rest_encryption()
  - Tests file-level encryption
  - Validates database column encryption
  - Tests key management (AWS KMS/Azure Key Vault)

- test_in_transit_encryption()
  - Tests TLS 1.3 configuration
  - Validates certificate validation
  - Tests cipher suite security

- test_key_rotation()
  - Tests automated key rotation
  - Validates backward compatibility
  - Tests emergency key revocation
```

### 3.5 GuardRails Effectiveness Tests

```python
# tests/security/guardrails/test_guardrails_effectiveness.py
- test_input_guardrails()
  - Tests prompt injection detection
  - Validates malicious payload blocking
  - Tests PHI detection rate (100% recall)

- test_output_guardrails()
  - Tests PHI leak prevention
  - Validates content policy enforcement
  - Tests hallucination detection

- test_guardrail_performance_impact()
  - Tests latency overhead (<50ms)
  - Validates throughput impact
  - Tests resource utilization

- test_bypass_attempts()
  - Tests encoding-based bypasses
  - Validates obfuscation resistance
  - Tests multi-step attack prevention
```

**Coverage Target**: 100% for security-critical paths

---

## 4. Performance Testing Strategy

### 4.1 Low-Latency Inference Tests

```python
# tests/performance/inference/test_inference_latency.py
- test_groq_response_time()
  - Tests p50 latency (<50ms)
  - Validates p95 latency (<100ms)
  - Tests p99 latency (<200ms)

- test_agent_response_time()
  - Tests Triador response (<500ms)
  - Validates Specialist analysis (<2s)
  - Tests Validator synthesis (<1s)

- test_cold_start_performance()
  - Tests first request latency
  - Validates model loading time
  - Tests connection establishment
```

### 4.2 Multi-Agent Parallel Processing Tests

```python
# tests/performance/concurrency/test_parallel_processing.py
- test_concurrent_agent_execution()
  - Tests 5 specialist agents in parallel
  - Validates resource contention handling
  - Tests speedup ratio (target: 4x)

- test_acp_debate_parallelism()
  - Tests multiple simultaneous debates
  - Validates message queue performance
  - Tests deadlock prevention

- test_system_throughput()
  - Tests cases processed per minute
  - Validates system saturation point
  - Tests auto-scaling triggers
```

### 4.3 Database Query Optimization Tests

```python
# tests/performance/database/test_query_performance.py
- test_vector_search_performance()
  - Tests search latency (<100ms for 1M vectors)
  - Validates index efficiency
  - Tests approximate vs exact search

- test_rag_retrieval_speed()
  - Tests top-k retrieval latency
  - Validates batch retrieval optimization
  - Tests cache hit rates

- test_write_performance()
  - Tests embedding insertion throughput
  - Validates transaction batching
  - Tests concurrent write handling
```

### 4.4 Memory and Token Efficiency Tests

```python
# tests/performance/efficiency/test_resource_efficiency.py
- test_memory_usage()
  - Tests agent memory footprint (<500MB/agent)
  - Validates memory leak detection
  - Tests garbage collection efficiency

- test_token_optimization()
  - Tests prompt compression effectiveness
  - Validates context window utilization
  - Tests token budget enforcement

- test_cost_optimization()
  - Tests cost per case analysis
  - Validates model selection efficiency
  - Tests caching effectiveness
```

**Performance Targets**:
- P95 end-to-end latency: <5 seconds
- System throughput: >100 cases/hour
- Memory per agent: <500MB
- Token efficiency: >80% relevant context

---

## 5. End-to-End Testing Strategy

### 5.1 Complete Case Processing Workflow

```python
# tests/e2e/workflows/test_case_processing.py
- test_cardiology_case_workflow()
  - Tests full workflow: upload → OCR → analysis → validation → report
  - Validates correct specialist routing
  - Tests report accuracy and completeness

- test_multi_specialty_case_workflow()
  - Tests complex case with 3+ specialties
  - Validates specialist coordination
  - Tests synthesis quality

- test_critical_case_workflow()
  - Tests urgent case prioritization
  - Validates escalation to human specialist
  - Tests WebRTC teleconsult initialization

- test_workflow_error_recovery()
  - Tests OCR failure recovery
  - Validates specialist unavailability handling
  - Tests partial result generation
```

### 5.2 Specialist Debate Scenarios

```python
# tests/e2e/debates/test_specialist_debates.py
- test_medication_conflict_resolution()
  - Tests contraindication detection
  - Validates evidence-based resolution
  - Tests consensus generation

- test_diagnostic_disagreement_debate()
  - Tests multi-turn debate protocol
  - Validates evidence presentation
  - Tests resolution within time budget

- test_unresolvable_conflict_handling()
  - Tests debate timeout
  - Validates human escalation
  - Tests partial recommendation generation

- test_debate_audit_trail()
  - Tests complete debate logging
  - Validates reasoning transparency
  - Tests HIPAA-compliant storage
```

### 5.3 Report Generation Validation

```python
# tests/e2e/reports/test_report_generation.py
- test_comprehensive_report_structure()
  - Tests all required sections present
  - Validates clinical accuracy
  - Tests citation completeness

- test_patient_friendly_summary()
  - Tests readability score (Flesch-Kincaid >60)
  - Validates jargon elimination
  - Tests actionable recommendations

- test_audio_report_generation()
  - Tests Edge-TTS integration
  - Validates audio quality
  - Tests synchronization with text

- test_multi_language_support()
  - Tests Portuguese translation accuracy
  - Validates medical term consistency
  - Tests cultural appropriateness
```

### 5.4 User Journey Tests

```python
# tests/e2e/user_journeys/test_user_journeys.py
- test_physician_complete_journey()
  - Tests login → upload → analysis → review → download
  - Validates UI responsiveness
  - Tests notification delivery

- test_patient_portal_journey()
  - Tests report access
  - Validates consent management
  - Tests data export functionality

- test_admin_management_journey()
  - Tests user management
  - Validates audit log access
  - Tests system configuration
```

**Coverage Target**: 80%+ of user workflows

---

## 6. Test Infrastructure and Automation

### 6.1 Test Directory Structure

```
tests/
├── unit/                      # Unit tests (fast, isolated)
│   ├── agents/               # Agent behavior tests
│   │   ├── __init__.py
│   │   ├── test_triador_agent.py
│   │   ├── test_specialist_agents.py
│   │   ├── test_validator_agent.py
│   │   └── test_communicator_agent.py
│   ├── ocr/                  # OCR pipeline tests
│   │   ├── __init__.py
│   │   ├── test_pymupdf_extraction.py
│   │   └── test_tesseract_fallback.py
│   ├── rag/                  # RAG system tests
│   │   ├── __init__.py
│   │   ├── test_embedding_generation.py
│   │   └── test_similarity_search.py
│   └── security/             # Security unit tests
│       ├── __init__.py
│       └── test_guardrails.py
│
├── integration/               # Integration tests
│   ├── acp/                  # ACP communication tests
│   │   ├── __init__.py
│   │   └── test_agent_communication.py
│   ├── groq/                 # Groq API integration
│   │   ├── __init__.py
│   │   └── test_groq_integration.py
│   ├── langfuse/             # Langfuse tracking
│   │   ├── __init__.py
│   │   └── test_langfuse_tracking.py
│   └── duckdb/               # Database integration
│       ├── __init__.py
│       └── test_vector_operations.py
│
├── security/                  # Security and compliance tests
│   ├── compliance/
│   │   ├── __init__.py
│   │   ├── test_hipaa_compliance.py
│   │   └── test_lgpd_compliance.py
│   ├── auth/
│   │   ├── __init__.py
│   │   └── test_authentication_security.py
│   ├── encryption/
│   │   ├── __init__.py
│   │   └── test_data_encryption.py
│   └── guardrails/
│       ├── __init__.py
│       └── test_guardrails_effectiveness.py
│
├── performance/               # Performance and load tests
│   ├── inference/
│   │   ├── __init__.py
│   │   └── test_inference_latency.py
│   ├── concurrency/
│   │   ├── __init__.py
│   │   └── test_parallel_processing.py
│   ├── database/
│   │   ├── __init__.py
│   │   └── test_query_performance.py
│   └── efficiency/
│       ├── __init__.py
│       └── test_resource_efficiency.py
│
├── e2e/                       # End-to-end tests
│   ├── workflows/
│   │   ├── __init__.py
│   │   └── test_case_processing.py
│   ├── debates/
│   │   ├── __init__.py
│   │   └── test_specialist_debates.py
│   ├── reports/
│   │   ├── __init__.py
│   │   └── test_report_generation.py
│   └── user_journeys/
│       ├── __init__.py
│       └── test_user_journeys.py
│
├── fixtures/                  # Test data and fixtures
│   ├── __init__.py
│   ├── case_data/            # Sample medical cases
│   │   ├── cardiology_case_001.json
│   │   ├── neurology_case_001.json
│   │   └── multi_specialty_case_001.json
│   ├── documents/            # Sample PDFs and images
│   │   ├── lab_report_001.pdf
│   │   ├── ecg_report_001.pdf
│   │   └── handwritten_note_001.jpg
│   ├── embeddings/           # Pre-generated embeddings
│   │   └── medical_knowledge_base.pkl
│   └── expected_outputs/     # Expected test results
│       ├── report_001.json
│       └── audio_script_001.txt
│
├── mocks/                     # Mock objects and stubs
│   ├── __init__.py
│   ├── mock_groq_api.py
│   ├── mock_duckdb.py
│   ├── mock_agents.py
│   └── mock_acp_server.py
│
├── conftest.py                # Pytest configuration and fixtures
├── pytest.ini                 # Pytest settings
└── requirements-test.txt      # Test dependencies
```

### 6.2 Mock Data and Fixtures

```python
# tests/fixtures/__init__.py
"""
Test fixtures for medical assistant system

Provides:
- Sample medical cases (anonymized)
- Mock documents (lab reports, ECGs, prescriptions)
- Pre-generated embeddings
- Expected test outputs
"""

# tests/fixtures/case_data.py
def cardiology_case_fixture():
    """Sample cardiology case with known diagnosis"""
    return {
        "patient_id": "TEST_PATIENT_001",
        "age": 65,
        "gender": "M",
        "chief_complaint": "Chest pain and shortness of breath",
        "lab_results": {
            "troponin": 0.5,  # Elevated
            "bnp": 450,       # Elevated
            "cholesterol": 220
        },
        "documents": [
            "tests/fixtures/documents/ecg_report_001.pdf"
        ],
        "expected_specialties": ["cardiology"],
        "expected_diagnosis": "Acute Coronary Syndrome",
        "expected_confidence": 0.85
    }

def multi_specialty_case_fixture():
    """Complex case requiring multiple specialists"""
    return {
        "patient_id": "TEST_PATIENT_002",
        "age": 58,
        "gender": "F",
        "chief_complaint": "Fatigue, weight gain, depression",
        "lab_results": {
            "tsh": 12.5,      # Elevated (hypothyroidism)
            "glucose": 180,   # Elevated (diabetes)
            "hba1c": 8.2      # Elevated
        },
        "medications": [
            "Metformin 1000mg",
            "Levothyroxine 50mcg"
        ],
        "expected_specialties": ["endocrinology", "psychiatry"],
        "expected_conflicts": ["medication_interaction"],
        "expected_debate": True
    }
```

```python
# tests/mocks/mock_groq_api.py
"""Mock Groq API for testing without API calls"""

class MockGroqClient:
    def __init__(self, latency_ms=50):
        self.latency_ms = latency_ms
        self.call_count = 0

    async def chat_completion(self, messages, model="mixtral-8x7b"):
        self.call_count += 1
        await asyncio.sleep(self.latency_ms / 1000)

        # Return deterministic responses based on message content
        if "cardiology" in messages[-1]["content"].lower():
            return {
                "choices": [{
                    "message": {
                        "content": "Likely Acute Coronary Syndrome based on elevated troponin..."
                    }
                }],
                "usage": {
                    "prompt_tokens": 150,
                    "completion_tokens": 200,
                    "total_tokens": 350
                }
            }
        return {"choices": [{"message": {"content": "General response"}}]}
```

### 6.3 CI/CD Pipeline Configuration

```yaml
# .github/workflows/test-pipeline.yml
name: Comprehensive Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements-test.txt

      - name: Run unit tests
        run: |
          pytest tests/unit/ \
            --cov=src \
            --cov-report=xml \
            --cov-report=html \
            --junit-xml=test-results/unit-tests.xml \
            -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          flags: unittests

  integration-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    services:
      duckdb:
        image: duckdb/duckdb:latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements-test.txt

      - name: Run integration tests
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY_TEST }}
          LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY_TEST }}
        run: |
          pytest tests/integration/ \
            --junit-xml=test-results/integration-tests.xml \
            -v

  security-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements-test.txt

      - name: Run security tests
        run: |
          pytest tests/security/ \
            --junit-xml=test-results/security-tests.xml \
            -v

      - name: Security scan with Bandit
        run: bandit -r src/ -f json -o security-report.json

      - name: Dependency vulnerability check
        run: safety check --json > safety-report.json

  performance-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements-test.txt

      - name: Run performance tests
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY_TEST }}
        run: |
          pytest tests/performance/ \
            --benchmark-only \
            --benchmark-json=performance-report.json \
            -v

  e2e-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements-test.txt

      - name: Start test environment
        run: docker-compose -f docker-compose.test.yml up -d

      - name: Wait for services
        run: |
          sleep 30
          curl --retry 10 --retry-delay 5 http://localhost:8000/health

      - name: Run E2E tests
        env:
          TEST_ENV: "ci"
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY_TEST }}
        run: |
          pytest tests/e2e/ \
            --junit-xml=test-results/e2e-tests.xml \
            -v

      - name: Teardown test environment
        if: always()
        run: docker-compose -f docker-compose.test.yml down

  quality-gates:
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests, security-tests]
    steps:
      - name: Check coverage threshold
        run: |
          if [ ${{ coverage }} -lt 90 ]; then
            echo "Coverage below 90% threshold"
            exit 1
          fi

      - name: Check security scan results
        run: |
          # Fail if high severity vulnerabilities found
          if [ $(jq '.results | length' security-report.json) -gt 0 ]; then
            echo "Security vulnerabilities detected"
            exit 1
          fi
```

### 6.4 Test Execution Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/unit/ -v                  # Unit tests only
pytest tests/integration/ -v           # Integration tests only
pytest tests/security/ -v              # Security tests only
pytest tests/performance/ -v           # Performance tests only
pytest tests/e2e/ -v                   # E2E tests only

# Run with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test file
pytest tests/unit/agents/test_triador_agent.py -v

# Run tests matching pattern
pytest tests/ -k "cardiology" -v

# Run tests with markers
pytest tests/ -m "security" -v         # Security tests
pytest tests/ -m "slow" -v             # Slow tests only
pytest tests/ -m "not slow" -v         # Skip slow tests

# Run performance benchmarks
pytest tests/performance/ --benchmark-only

# Run with parallel execution (8 workers)
pytest tests/ -n 8

# Run with detailed output
pytest tests/ -vv --tb=short

# Run with PDB on failure
pytest tests/ --pdb

# Generate HTML test report
pytest tests/ --html=test-report.html --self-contained-html
```

---

## 7. Test Coverage Targets and Quality Gates

### 7.1 Coverage Requirements by Component

| Component | Target Coverage | Critical Path Coverage |
|-----------|----------------|------------------------|
| Agent Logic | 95% | 100% |
| Security/Guardrails | 100% | 100% |
| OCR Pipeline | 90% | 95% |
| RAG System | 90% | 95% |
| ACP Communication | 85% | 95% |
| API Endpoints | 90% | 100% |
| Database Operations | 85% | 90% |
| Frontend Components | 80% | 85% |
| **Overall System** | **90%** | **95%** |

### 7.2 Quality Gates (CI/CD)

**Mandatory Checks (Block PR merge if failed)**:
1. ✅ All unit tests pass (100%)
2. ✅ All security tests pass (100%)
3. ✅ Code coverage ≥ 90%
4. ✅ Security scan: 0 high/critical vulnerabilities
5. ✅ No PHI in test data or logs
6. ✅ Performance benchmarks within thresholds

**Advisory Checks (Warning, don't block)**:
1. ⚠️ Integration test failures (manual review)
2. ⚠️ E2E test flakiness (manual review)
3. ⚠️ Performance degradation <10%

### 7.3 Performance Benchmarks

**Latency Targets**:
- Groq inference (p95): <100ms
- Triador analysis: <500ms
- Specialist analysis: <2s
- Validator synthesis: <1s
- End-to-end case: <5s

**Throughput Targets**:
- Cases per hour: >100
- Concurrent agents: >20
- Vector search QPS: >1000

**Resource Targets**:
- Memory per agent: <500MB
- Database size: <10GB per 100K cases
- Token efficiency: >80% relevant context

---

## 8. Test Data Management

### 8.1 Synthetic Medical Data Generation

```python
# tests/fixtures/data_generator.py
"""Generate HIPAA-compliant synthetic medical data for testing"""

from faker import Faker
import random

fake = Faker()

def generate_synthetic_patient():
    """Generate synthetic patient data"""
    return {
        "patient_id": f"TEST_{fake.uuid4()}",
        "age": random.randint(18, 90),
        "gender": random.choice(["M", "F"]),
        "chief_complaint": random.choice([
            "Chest pain",
            "Shortness of breath",
            "Abdominal pain",
            "Headache",
            "Fatigue"
        ]),
        "medical_history": generate_medical_history(),
        "medications": generate_medications(),
        "lab_results": generate_lab_results()
    }

def generate_lab_results():
    """Generate realistic lab values"""
    return {
        "glucose": random.randint(70, 200),
        "cholesterol": random.randint(150, 300),
        "hdl": random.randint(30, 80),
        "ldl": random.randint(50, 190),
        "triglycerides": random.randint(50, 250),
        "tsh": round(random.uniform(0.5, 10.0), 2),
        "hba1c": round(random.uniform(4.0, 12.0), 1)
    }
```

### 8.2 Anonymization Verification

```python
# tests/security/test_data_anonymization.py
def test_no_phi_in_test_data():
    """Verify no real PHI in test fixtures"""
    fixtures_path = "tests/fixtures/case_data/"

    for file in os.listdir(fixtures_path):
        with open(os.path.join(fixtures_path, file)) as f:
            content = f.read()

            # Check for common PHI patterns
            assert not re.search(r'\d{3}-\d{2}-\d{4}', content), "SSN detected"
            assert not re.search(r'\(\d{3}\) \d{3}-\d{4}', content), "Phone detected"
            assert not re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}', content), "Email detected"
```

---

## 9. Continuous Improvement

### 9.1 Test Metrics Dashboard

**Key Metrics to Track**:
- Test execution time trends
- Coverage trends by component
- Flaky test rate
- Test failure patterns
- Security scan results over time
- Performance benchmark trends

### 9.2 Test Maintenance Strategy

**Weekly**:
- Review flaky tests and fix/remove
- Update mock data for new scenarios
- Review and merge test improvements

**Monthly**:
- Audit test coverage gaps
- Performance benchmark review
- Security test update for new threats
- Test execution time optimization

**Quarterly**:
- Major test refactoring
- Test infrastructure upgrades
- Load testing with production-scale data
- Penetration testing coordination

---

## 10. Conclusion

This comprehensive testing strategy ensures the medical assistant system meets the highest standards of reliability, security, and performance required for healthcare applications. By maintaining 90%+ test coverage with a focus on security-critical paths, implementing automated CI/CD quality gates, and continuously improving test effectiveness, we can deliver a production-ready system that physicians and patients can trust.

**Key Success Factors**:
1. 🔒 Security-first testing approach (100% coverage of security paths)
2. ⚡ Performance validation at every layer (<5s end-to-end)
3. 🤝 Multi-agent coordination testing (ACP reliability)
4. 📊 Comprehensive monitoring and metrics
5. 🔄 Continuous improvement and maintenance

**Next Steps**:
1. Implement core test infrastructure
2. Create test fixtures and mocks
3. Configure CI/CD pipeline
4. Train team on testing practices
5. Establish test metrics dashboard
