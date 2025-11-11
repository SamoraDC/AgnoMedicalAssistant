# System Architecture Documentation

## Overview

The Agno Medical Assistant implements a hierarchical multi-agent architecture based on the Mixture of Experts (MoE) pattern. This design prioritizes low latency, multimodal processing, and collaborative reasoning while maintaining HIPAA/LGPD compliance.

## Architectural Principles

### 1. Mixture of Experts (MoE) Agent Pattern

The system avoids using a single generalist medical agent, which would be prone to hallucinations and lack domain depth. Instead, it implements a hierarchical structure:

```
Triager (Router)
    │
    ├─→ Cardiology Specialist
    ├─→ Neurology Specialist
    ├─→ Endocrinology Specialist
    ├─→ Radiology Specialist
    └─→ [Other Specialists]
         │
         └─→ Validator (Aggregator)
              │
              └─→ Communicator (Output Generator)
```

**Benefits of MoE Pattern**:
- Reduced hallucination risk through domain specialization
- Parallel processing of different aspects of a case
- Expert-level depth in each medical domain
- Scalable to additional specialties

### 2. Agent Communication Protocol (ACP)

When conflicts arise between specialist agents, the Validator initiates a structured debate using ACP:

```
┌──────────────────────────────────────────────────────────────┐
│                    Conflict Detection                         │
│  Cardiology Agent: "Recommend beta-blocker increase"         │
│  Neurology Agent: "Beta-blockers contraindicated for         │
│                     patient's neurological condition"         │
└───────────────────┬──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│              Validator Initiates ACP Debate                   │
│                                                               │
│  1. Send conflict details to both agents                     │
│  2. Request evidence from knowledge bases                    │
│  3. Exchange multiple reasoning turns                        │
│  4. Reach consensus or escalate to human                     │
└──────────────────────────────────────────────────────────────┘
```

**ACP Characteristics**:
- REST-based, framework-agnostic protocol
- Asynchronous, long-running conversations
- Auditable message logs for compliance
- Structured evidence exchange format

**Critical Dependency**: ACP debates are only feasible due to Groq's sub-second inference latency. Traditional GPU-based inference would make multi-turn debates impractically slow.

## System Layers

### Layer 1: Data Ingestion & Security

```
Patient Data Upload
    │
    ├─→ HTTPS/TLS Encryption (Transport)
    │
    ├─→ OAuth2 Authentication
    │
    ├─→ GuardRails AI (PHI Detection)
    │       │
    │       ├─→ Input Guardrail: Anonymize PHI
    │       └─→ PII Detection: Flag sensitive data
    │
    └─→ Encrypted Storage (AES-256)
```

**Security Components**:
- **Authentication**: OAuth2 with JWT tokens (30-minute expiry)
- **Encryption**: AES-256 at rest, TLS 1.3 in transit
- **GuardRails AI**: Automatic PHI/PII detection and anonymization
- **Audit Logging**: Every data access logged with timestamp, user, action

### Layer 2: Multimodal Processing Pipeline

```
Raw Input (PDF, Images, Text)
    │
    ├─→ Document Type Detection
    │
    ├─→ OCR Processing
    │   ├─→ PyMuPDF (Primary)
    │   │   ├─→ Quality Score Check
    │   │   └─→ If score < threshold → Tesseract fallback
    │   │
    │   └─→ Tesseract OCR (Fallback)
    │       ├─→ Preprocessing: Binarization, Denoising
    │       └─→ Multiple PSM modes for optimal results
    │
    ├─→ Text Extraction & Normalization
    │
    └─→ Metadata Extraction (Patient ID, Date, Lab values)
```

**OCR Strategy**:
- **Primary**: PyMuPDF (fast, integrated Tesseract)
- **Fallback**: Direct Tesseract with preprocessing when quality is low
- **Quality Metrics**: Character confidence scores, word accuracy
- **Preprocessing**: Adaptive thresholding, morphological operations

### Layer 3: Agent Orchestration (Agno Framework)

#### Triager Agent

**Role**: Initial analysis and routing

**Capabilities**:
- OCR coordination and quality validation
- Medical specialty classification
- Case priority assessment
- Specialist agent selection and spawning

**Tools**:
- `classify_specialty(symptoms, lab_results)` → List[Specialty]
- `extract_medical_entities(text)` → Dict[entity_type, values]
- `search_similar_cases(embeddings)` → List[CaseMatch]
- `spawn_specialists(specialties)` → List[AgentID]

**Knowledge Base**: General medical triage guidelines, ICD-10 codes

#### Specialist Agents

**Role**: Domain-specific analysis

**Example: Cardiology Agent**

**Capabilities**:
- ECG interpretation
- Cardiac biomarker analysis
- Drug interaction checking
- Treatment recommendation generation

**Tools**:
- `analyze_cardiac_markers(lab_results)` → Analysis
- `check_drug_interactions(medications)` → List[Interaction]
- `rag_search(query, domain="cardiology")` → List[Evidence]
- `calculate_risk_score(patient_data)` → RiskScore

**Knowledge Base**:
- Cardiology journals and guidelines
- Drug interaction databases
- Cardiac condition protocols
- Historical case embeddings

**Each specialist follows the same pattern but with domain-specific:**
- Knowledge bases (RAG)
- Risk scoring algorithms
- Treatment protocols
- Diagnostic criteria

#### Validator Agent

**Role**: Aggregation and conflict resolution

**Capabilities**:
- Hypothesis aggregation from specialists
- Contradiction detection
- ACP debate orchestration
- Consensus building

**Conflict Resolution Flow**:
```python
def validate_hypotheses(specialist_outputs: List[Hypothesis]) -> Validation:
    # 1. Collect all hypotheses
    hypotheses = [output.hypothesis for output in specialist_outputs]

    # 2. Detect conflicts
    conflicts = detect_contradictions(hypotheses)

    if not conflicts:
        return synthesize_report(hypotheses)

    # 3. Initiate ACP debate
    for conflict in conflicts:
        agents_in_conflict = conflict.agents

        # Send ACP message to initiate debate
        debate_result = await acp_debate(
            agents=agents_in_conflict,
            issue=conflict.description,
            evidence_sources=[agent.knowledge_base for agent in agents_in_conflict],
            max_turns=5,
            timeout=30  # seconds
        )

        # 4. Update hypotheses with debate resolution
        if debate_result.consensus_reached:
            update_hypothesis(debate_result.resolution)
        else:
            flag_for_human_review(conflict)

    return synthesize_report(hypotheses)
```

**Tools**:
- `detect_contradictions(hypotheses)` → List[Conflict]
- `initiate_acp_debate(conflict)` → DebateSession
- `synthesize_consensus(debate_results)` → Synthesis
- `flag_critical_cases()` → EscalationAlert

#### Communicator Agent

**Role**: Patient-facing output generation

**Capabilities**:
- Medical jargon simplification
- Multi-language support
- Audio report generation (Edge-TTS)
- Summary formatting for different audiences

**Output Formats**:
1. **Physician Report**: Technical, detailed, with evidence citations
2. **Patient Summary**: Clear language, key findings, next steps
3. **Audio Report**: Text-to-speech for patient accessibility
4. **Structured Data**: JSON/FHIR format for EHR integration

**Tools**:
- `simplify_medical_terms(text)` → SimplifiedText
- `generate_audio_report(text, language)` → AudioFile
- `format_for_audience(content, audience_type)` → FormattedReport
- `translate_report(text, target_language)` → TranslatedText

### Layer 4: Knowledge & Memory Systems

#### RAG (Retrieval-Augmented Generation)

Each specialist agent has a dedicated RAG pipeline:

```
Query from Agent
    │
    ├─→ Query Embedding (Sentence Transformers)
    │
    ├─→ Vector Similarity Search (DuckDB VSS)
    │   └─→ Top-K relevant documents (k=5-10)
    │
    ├─→ Reranking (Cross-encoder)
    │
    ├─→ Context Injection into Agent Prompt
    │
    └─→ LLM Generation with Evidence
```

**Knowledge Sources**:
- Medical journals (PubMed, NEJM, Lancet)
- Clinical guidelines (WHO, CDC, specialty societies)
- Drug databases (RxNorm, DrugBank)
- Interaction databases

**DuckDB Schema**:
```sql
CREATE TABLE medical_knowledge (
    id INTEGER PRIMARY KEY,
    document_id VARCHAR,
    specialty VARCHAR,
    content TEXT,
    embedding FLOAT[768],  -- Using VSS extension
    source VARCHAR,
    publication_date DATE,
    confidence_score FLOAT
);

CREATE INDEX idx_specialty ON medical_knowledge(specialty);
CREATE INDEX idx_embedding ON medical_knowledge USING HNSW (embedding);
```

#### Case-Based Reasoning (CBR)

Semantic memory of anonymized historical cases:

```
New Case Arrives
    │
    ├─→ Generate Case Embedding
    │
    ├─→ Similarity Search in DuckDB
    │   └─→ Find top 3-5 similar cases
    │
    ├─→ Extract Patterns
    │   ├─→ Common diagnoses
    │   ├─→ Successful treatments
    │   └─→ Complications encountered
    │
    └─→ Provide Context to Triager Agent
        └─→ "85% similar to Case #1234 (Diagnosis: X)"
```

**DuckDB Schema**:
```sql
CREATE TABLE case_memory (
    case_id VARCHAR PRIMARY KEY,
    anonymized_data TEXT,  -- All PHI removed
    symptoms_embedding FLOAT[768],
    diagnosis VARCHAR,
    treatment_plan TEXT,
    outcome VARCHAR,
    specialist_involved VARCHAR[],
    case_embedding FLOAT[768]
);

CREATE INDEX idx_case_embedding ON case_memory USING HNSW (case_embedding);
```

**Anonymization Process**:
1. Remove all direct identifiers (names, dates, IDs)
2. Generalize quasi-identifiers (age → age range, location → region)
3. Apply differential privacy for aggregate queries
4. Store only clinical patterns, not raw data

### Layer 5: Inference & Performance

#### Groq LPU Integration

**Why Groq LPUs?**
- **Latency**: <1 second per inference (vs. 3-10s on GPU)
- **Throughput**: 500+ tokens/second
- **Critical for ACP**: Makes multi-turn agent debates feasible
- **Cost Efficiency**: Lower cost per token than cloud GPU inference

**Inference Configuration**:
```python
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Specialist agent inference
def specialist_inference(prompt: str, specialist_type: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",  # or mixtral-8x7b-32768
        messages=[
            {"role": "system", "content": f"You are an expert {specialist_type}."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Low temperature for medical accuracy
        max_tokens=2048,
        top_p=0.95
    )
    return response.choices[0].message.content
```

**Performance Monitoring**:
- Track latency per agent type
- Monitor token usage
- Alert on timeout/errors
- Load balancing across multiple API keys

## Data Flow Example

### Complete Case Processing Flow

```
1. UPLOAD
   └─→ Doctor uploads: Lab results PDF + Clinical notes + ECG image

2. SECURITY LAYER
   ├─→ OAuth2 validates doctor identity
   ├─→ GuardRails detects patient name, DOB → anonymizes
   └─→ Encrypted storage in DuckDB

3. TRIAGER AGENT
   ├─→ OCR extracts text from PDF
   ├─→ Identifies: Elevated troponin, chest pain symptoms
   ├─→ Case similarity search → 3 similar cases found
   ├─→ Classifies specialties needed: Cardiology + Emergency Medicine
   └─→ Spawns 2 specialist agents

4. PARALLEL SPECIALIST ANALYSIS (5-10 seconds each)

   Cardiology Agent:
   ├─→ RAG search: "elevated troponin + chest pain"
   ├─→ Retrieves: MI guidelines, drug protocols
   ├─→ Hypothesis: "Likely acute myocardial infarction"
   ├─→ Recommendation: "Emergency catheterization + aspirin"
   └─→ Confidence: 0.89

   Emergency Medicine Agent:
   ├─→ RAG search: "chest pain triage"
   ├─→ Retrieves: Emergency protocols
   ├─→ Hypothesis: "Possible MI, rule out pulmonary embolism"
   ├─→ Recommendation: "D-dimer test + CT angiography"
   └─→ Confidence: 0.75

5. VALIDATOR AGENT
   ├─→ Receives both hypotheses
   ├─→ Detects conflict: Different diagnostic priorities
   ├─→ Initiates ACP debate (3 turns, 15 seconds)
   │   Turn 1: Cardiology presents troponin evidence
   │   Turn 2: EM presents PE risk factors from notes
   │   Turn 3: Consensus → Prioritize MI, but order D-dimer
   └─→ Synthesized plan: "Primary focus MI with PE ruled out"

6. COMMUNICATOR AGENT
   ├─→ Generates physician report (technical)
   ├─→ Generates patient summary (simplified)
   ├─→ Edge-TTS creates audio version
   └─→ Delivers all formats to dashboard

7. OUTPUT DELIVERED (Total time: ~45 seconds)
   ├─→ Physician sees detailed analysis + evidence
   ├─→ Patient receives understandable summary + audio
   └─→ Critical flag triggers teleconsultation option
```

## Scalability Considerations

### Horizontal Scaling

```
Load Balancer
    │
    ├─→ FastAPI Instance 1 (Agents 1-3)
    ├─→ FastAPI Instance 2 (Agents 4-6)
    └─→ FastAPI Instance 3 (Agents 7-9)
         │
         └─→ Shared DuckDB (or distributed vector DB)
```

**Strategies**:
- Stateless FastAPI instances
- Agent orchestration via message queue (Redis/RabbitMQ)
- Read replicas for DuckDB knowledge base
- CDN for audio report delivery

### Performance Optimization

1. **Agent Caching**: Cache specialist analyses for similar case patterns
2. **Embedding Precomputation**: Precompute embeddings for knowledge base updates
3. **Lazy Specialist Spawning**: Only spawn specialists with high relevance scores
4. **Parallel RAG Queries**: Run all specialist RAG queries in parallel
5. **Result Memoization**: Cache debate resolutions for common conflicts

## Monitoring & Observability

### Key Metrics

```python
# Prometheus metrics
agent_inference_latency = Histogram('agent_inference_seconds', 'Agent inference time', ['agent_type'])
debate_resolution_time = Histogram('debate_resolution_seconds', 'ACP debate duration')
case_processing_time = Histogram('case_total_seconds', 'End-to-end case processing')
specialist_confidence = Histogram('specialist_confidence', 'Confidence scores', ['specialty'])
phi_detection_rate = Counter('phi_detections_total', 'PHI items detected and anonymized')
```

### Logging Strategy

```python
import structlog

logger = structlog.get_logger()

# Every agent action logged
logger.info(
    "specialist_analysis_complete",
    agent_id=agent.id,
    agent_type="cardiology",
    case_id=case.id,
    confidence=0.89,
    hypothesis="acute MI",
    processing_time_ms=850,
    evidence_sources=["pubmed:12345", "guideline:aha2023"]
)
```

### Audit Trail

All actions are logged for HIPAA compliance:
- Who accessed the data (user_id)
- What action was performed (view, analyze, modify)
- When it occurred (timestamp)
- What data was accessed (case_id, patient_id)
- System response (success, failure, reason)

## Disaster Recovery

### Backup Strategy

1. **Database Backups**: Daily DuckDB snapshots to encrypted cloud storage
2. **Knowledge Base Versioning**: Track RAG database versions
3. **Case Memory Exports**: Weekly anonymized case exports
4. **Configuration Backups**: Infrastructure as Code (Terraform)

### Recovery Procedures

```bash
# Restore DuckDB from backup
aws s3 cp s3://medical-backups/2024-01-15-medical.db ./data/medical.db

# Reinitialize vector indices
python scripts/rebuild_vector_indices.py

# Verify agent functionality
python scripts/test_agents.py --full-suite
```

## Security Architecture

### Defense in Depth

```
┌─────────────────────────────────────────────────────┐
│ Layer 1: Network Security                           │
│   - TLS 1.3 encryption                              │
│   - Firewall rules (Allow HTTPS/WSS only)           │
│   - DDoS protection                                 │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│ Layer 2: Application Security                       │
│   - OAuth2 authentication                           │
│   - JWT with short expiry                           │
│   - Rate limiting (10 req/min per user)             │
│   - Input validation                                │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│ Layer 3: Data Security                              │
│   - GuardRails AI (PHI detection)                   │
│   - Field-level encryption                          │
│   - Anonymization pipeline                          │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│ Layer 4: Storage Security                           │
│   - AES-256 encryption at rest                      │
│   - Access control lists                            │
│   - Audit logging                                   │
└─────────────────────────────────────────────────────┘
```

## Future Architecture Enhancements

### Planned Improvements

1. **Federated Learning**: Train models on distributed hospital data without centralizing PHI
2. **Explainable AI**: SHAP/LIME integration for agent decision transparency
3. **Continuous Learning**: Agents update from validated case outcomes
4. **Multi-hospital Deployment**: Federated architecture across healthcare systems
5. **Genomic Data Integration**: Specialist agent for pharmacogenomics

---

**Last Updated**: 2025-11-11
**Maintained By**: Architecture Team
