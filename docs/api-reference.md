# API Reference

Complete OpenAPI 3.0 specification for the Agno Medical Assistant REST API.

## Base Information

- **Base URL**: `https://api.agnomedical.example.com/v1`
- **Protocol**: HTTPS only
- **Authentication**: OAuth2 with JWT Bearer tokens
- **Content-Type**: `application/json` (default), `multipart/form-data` (file uploads)

## Authentication

### POST /auth/token

Obtain a JWT access token.

**Request**:
```http
POST /auth/token
Content-Type: application/x-www-form-urlencoded

username=doctor@hospital.com&password=securepassword
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user_id": "usr_12345",
  "role": "physician"
}
```

**Status Codes**:
- `200 OK`: Authentication successful
- `401 Unauthorized`: Invalid credentials
- `429 Too Many Requests`: Rate limit exceeded

### POST /auth/refresh

Refresh an expired JWT token.

**Request**:
```http
POST /auth/refresh
Authorization: Bearer <expired_token>
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

## Case Management

### POST /cases

Create a new medical case for analysis.

**Request**:
```http
POST /cases
Authorization: Bearer <token>
Content-Type: multipart/form-data

patient_id=pat_67890
priority=high
files[]=lab_results.pdf
files[]=ecg_image.png
clinical_notes="Patient presents with chest pain..."
```

**Request Body**:
- `patient_id` (string, required): Anonymized patient identifier
- `priority` (enum, optional): `low`, `medium`, `high`, `critical` (default: `medium`)
- `files[]` (array[file], optional): Medical documents (PDF, images)
- `clinical_notes` (string, optional): Free-text clinical notes
- `specialties_hint` (array[string], optional): Suggested specialties for routing

**Response**:
```json
{
  "case_id": "case_abc123",
  "status": "processing",
  "created_at": "2025-11-11T10:30:00Z",
  "estimated_completion": "2025-11-11T10:31:30Z",
  "agents_assigned": [
    {
      "agent_id": "agent_triager_001",
      "agent_type": "triager",
      "status": "active"
    }
  ]
}
```

**Status Codes**:
- `201 Created`: Case created successfully
- `400 Bad Request`: Invalid input data
- `401 Unauthorized`: Missing or invalid token
- `413 Payload Too Large`: Files exceed size limit (50MB)
- `422 Unprocessable Entity`: PHI detection failed, cannot anonymize

### GET /cases/{case_id}

Retrieve case details and current processing status.

**Request**:
```http
GET /cases/case_abc123
Authorization: Bearer <token>
```

**Response**:
```json
{
  "case_id": "case_abc123",
  "status": "completed",
  "created_at": "2025-11-11T10:30:00Z",
  "completed_at": "2025-11-11T10:31:45Z",
  "processing_time_seconds": 105,
  "patient_id": "pat_67890",
  "priority": "high",
  "agents_involved": [
    {
      "agent_type": "triager",
      "confidence": 0.95,
      "specialties_identified": ["cardiology", "emergency_medicine"]
    },
    {
      "agent_type": "cardiology_specialist",
      "confidence": 0.89,
      "hypothesis": "Acute myocardial infarction",
      "evidence_sources": ["pubmed:12345", "guideline:aha2023"]
    },
    {
      "agent_type": "emergency_medicine_specialist",
      "confidence": 0.75,
      "hypothesis": "Possible MI, rule out pulmonary embolism"
    }
  ],
  "debates": [
    {
      "debate_id": "debate_001",
      "agents": ["cardiology_specialist", "emergency_medicine_specialist"],
      "issue": "Diagnostic priority conflict",
      "resolution": "Primary focus MI with PE ruled out via D-dimer",
      "consensus_reached": true,
      "turns": 3,
      "duration_seconds": 15
    }
  ],
  "final_report": {
    "synthesis": "Patient likely experiencing acute myocardial infarction...",
    "recommendations": [
      "Emergency cardiac catheterization",
      "Administer aspirin 325mg",
      "Order D-dimer to rule out PE"
    ],
    "confidence_score": 0.87,
    "critical_flag": true
  },
  "outputs": {
    "physician_report_url": "/reports/case_abc123_physician.pdf",
    "patient_summary_url": "/reports/case_abc123_patient.pdf",
    "audio_report_url": "/reports/case_abc123_audio.mp3"
  }
}
```

**Status Codes**:
- `200 OK`: Case found
- `401 Unauthorized`: Invalid token
- `403 Forbidden`: User doesn't have access to this case
- `404 Not Found`: Case doesn't exist

### GET /cases

List all cases for the authenticated user.

**Query Parameters**:
- `status` (enum, optional): Filter by status (`processing`, `completed`, `escalated`, `error`)
- `priority` (enum, optional): Filter by priority
- `patient_id` (string, optional): Filter by patient
- `from_date` (ISO8601, optional): Cases created after this date
- `to_date` (ISO8601, optional): Cases created before this date
- `limit` (integer, optional): Max results (default: 50, max: 200)
- `offset` (integer, optional): Pagination offset (default: 0)

**Request**:
```http
GET /cases?status=completed&priority=high&limit=20
Authorization: Bearer <token>
```

**Response**:
```json
{
  "cases": [
    {
      "case_id": "case_abc123",
      "status": "completed",
      "priority": "high",
      "created_at": "2025-11-11T10:30:00Z",
      "patient_id": "pat_67890",
      "confidence_score": 0.87
    }
  ],
  "total": 156,
  "limit": 20,
  "offset": 0,
  "has_more": true
}
```

**Status Codes**:
- `200 OK`: Success
- `401 Unauthorized`: Invalid token

### DELETE /cases/{case_id}

Delete a case and all associated data (HIPAA right to deletion).

**Request**:
```http
DELETE /cases/case_abc123
Authorization: Bearer <token>
```

**Response**:
```json
{
  "message": "Case deleted successfully",
  "deleted_at": "2025-11-11T11:00:00Z",
  "audit_trail_retained": true
}
```

**Status Codes**:
- `200 OK`: Case deleted
- `401 Unauthorized`: Invalid token
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Case doesn't exist

## Agent Management

### GET /agents/status

Get current status of all agent types.

**Request**:
```http
GET /agents/status
Authorization: Bearer <token>
```

**Response**:
```json
{
  "agents": [
    {
      "agent_type": "triager",
      "instances_active": 5,
      "avg_processing_time_ms": 1200,
      "success_rate": 0.98,
      "current_load": 0.65
    },
    {
      "agent_type": "cardiology_specialist",
      "instances_active": 3,
      "avg_processing_time_ms": 850,
      "success_rate": 0.95,
      "current_load": 0.45
    }
  ],
  "system_health": "healthy",
  "total_active_agents": 18,
  "groq_api_status": "operational"
}
```

**Status Codes**:
- `200 OK`: Success
- `401 Unauthorized`: Invalid token

### POST /agents/spawn

Manually spawn a specialist agent (admin only).

**Request**:
```http
POST /agents/spawn
Authorization: Bearer <admin_token>
Content-Type: application/json

{
  "agent_type": "cardiology_specialist",
  "case_id": "case_abc123",
  "priority": "high"
}
```

**Response**:
```json
{
  "agent_id": "agent_cardio_007",
  "agent_type": "cardiology_specialist",
  "status": "active",
  "assigned_case": "case_abc123",
  "spawned_at": "2025-11-11T10:30:05Z"
}
```

**Status Codes**:
- `201 Created`: Agent spawned
- `401 Unauthorized`: Invalid token
- `403 Forbidden`: Requires admin role
- `429 Too Many Requests`: Max agents reached

## Reports

### GET /reports/{case_id}/physician

Download detailed physician report (PDF).

**Request**:
```http
GET /reports/case_abc123/physician
Authorization: Bearer <token>
```

**Response**:
```
Content-Type: application/pdf
Content-Disposition: attachment; filename="case_abc123_physician.pdf"

[PDF binary data]
```

**Status Codes**:
- `200 OK`: Report downloaded
- `401 Unauthorized`: Invalid token
- `403 Forbidden`: No access to this case
- `404 Not Found`: Report not ready or case doesn't exist

### GET /reports/{case_id}/patient

Download patient-friendly summary (PDF).

**Request**:
```http
GET /reports/case_abc123/patient
Authorization: Bearer <token>
```

**Response**:
```
Content-Type: application/pdf
Content-Disposition: attachment; filename="case_abc123_patient.pdf"

[PDF binary data]
```

### GET /reports/{case_id}/audio

Download audio report (MP3).

**Request**:
```http
GET /reports/case_abc123/audio
Authorization: Bearer <token>
```

**Response**:
```
Content-Type: audio/mpeg
Content-Disposition: attachment; filename="case_abc123_audio.mp3"

[MP3 binary data]
```

### POST /reports/{case_id}/regenerate

Regenerate reports with updated formatting or language.

**Request**:
```http
POST /reports/case_abc123/regenerate
Authorization: Bearer <token>
Content-Type: application/json

{
  "language": "pt-BR",
  "audio_voice": "female",
  "include_evidence": true
}
```

**Response**:
```json
{
  "message": "Reports regenerated successfully",
  "physician_report_url": "/reports/case_abc123_physician.pdf",
  "patient_summary_url": "/reports/case_abc123_patient.pdf",
  "audio_report_url": "/reports/case_abc123_audio.mp3"
}
```

**Status Codes**:
- `200 OK`: Reports regenerated
- `401 Unauthorized`: Invalid token
- `404 Not Found`: Case doesn't exist

## Knowledge Base

### GET /knowledge/search

Search the RAG knowledge base.

**Query Parameters**:
- `query` (string, required): Search query
- `specialty` (string, optional): Filter by specialty
- `top_k` (integer, optional): Number of results (default: 5, max: 20)

**Request**:
```http
GET /knowledge/search?query=acute%20myocardial%20infarction&specialty=cardiology&top_k=5
Authorization: Bearer <token>
```

**Response**:
```json
{
  "query": "acute myocardial infarction",
  "results": [
    {
      "document_id": "pubmed:12345",
      "title": "Management of Acute Myocardial Infarction",
      "source": "American Heart Association Guidelines 2023",
      "relevance_score": 0.94,
      "excerpt": "Immediate administration of aspirin and urgent cardiac catheterization...",
      "specialty": "cardiology",
      "publication_date": "2023-08-15"
    }
  ],
  "total_results": 5
}
```

**Status Codes**:
- `200 OK`: Search completed
- `400 Bad Request`: Invalid query
- `401 Unauthorized`: Invalid token

### POST /knowledge/add

Add a document to the knowledge base (admin only).

**Request**:
```http
POST /knowledge/add
Authorization: Bearer <admin_token>
Content-Type: multipart/form-data

specialty=cardiology
source=American Heart Association
file=aha_guidelines_2023.pdf
```

**Response**:
```json
{
  "document_id": "doc_xyz789",
  "status": "processing",
  "estimated_completion": "2025-11-11T10:35:00Z"
}
```

**Status Codes**:
- `201 Created`: Document added
- `401 Unauthorized`: Invalid token
- `403 Forbidden`: Requires admin role

## Case Memory (Case-Based Reasoning)

### GET /memory/similar-cases

Find similar historical cases.

**Query Parameters**:
- `case_id` (string, required): Reference case ID
- `top_k` (integer, optional): Number of results (default: 3, max: 10)

**Request**:
```http
GET /memory/similar-cases?case_id=case_abc123&top_k=3
Authorization: Bearer <token>
```

**Response**:
```json
{
  "reference_case": "case_abc123",
  "similar_cases": [
    {
      "case_id": "case_historical_001",
      "similarity_score": 0.87,
      "diagnosis": "Acute myocardial infarction",
      "treatment": "Emergency PCI",
      "outcome": "Successful revascularization",
      "specialists_involved": ["cardiology", "emergency_medicine"]
    }
  ]
}
```

**Status Codes**:
- `200 OK`: Similar cases found
- `401 Unauthorized`: Invalid token
- `404 Not Found`: Reference case doesn't exist

## Teleconsultation

### POST /teleconsult/initiate

Initiate a WebRTC teleconsultation session.

**Request**:
```http
POST /teleconsult/initiate
Authorization: Bearer <token>
Content-Type: application/json

{
  "case_id": "case_abc123",
  "specialist_type": "cardiology",
  "reason": "Urgent case requiring expert review"
}
```

**Response**:
```json
{
  "session_id": "rtc_session_456",
  "signaling_server": "wss://signal.agnomedical.example.com",
  "ice_servers": [
    {
      "urls": "stun:stun.agnomedical.example.com:3478"
    },
    {
      "urls": "turn:turn.agnomedical.example.com:3478",
      "username": "temporary_user",
      "credential": "temporary_password"
    }
  ],
  "expires_at": "2025-11-11T11:00:00Z"
}
```

**Status Codes**:
- `201 Created`: Session initiated
- `401 Unauthorized`: Invalid token
- `404 Not Found`: Case doesn't exist
- `503 Service Unavailable`: No specialists available

### DELETE /teleconsult/{session_id}

End a teleconsultation session.

**Request**:
```http
DELETE /teleconsult/rtc_session_456
Authorization: Bearer <token>
```

**Response**:
```json
{
  "message": "Session ended",
  "duration_seconds": 1234,
  "recorded": false
}
```

**Status Codes**:
- `200 OK`: Session ended
- `401 Unauthorized`: Invalid token
- `404 Not Found`: Session doesn't exist

## Audit & Compliance

### GET /audit/log

Retrieve audit log entries (admin only).

**Query Parameters**:
- `user_id` (string, optional): Filter by user
- `action` (string, optional): Filter by action type
- `from_date` (ISO8601, optional): Start date
- `to_date` (ISO8601, optional): End date
- `limit` (integer, optional): Max results (default: 100, max: 1000)

**Request**:
```http
GET /audit/log?user_id=usr_12345&from_date=2025-11-01T00:00:00Z&limit=50
Authorization: Bearer <admin_token>
```

**Response**:
```json
{
  "entries": [
    {
      "entry_id": "audit_001",
      "timestamp": "2025-11-11T10:30:00Z",
      "user_id": "usr_12345",
      "action": "case_created",
      "case_id": "case_abc123",
      "ip_address": "192.168.1.100",
      "user_agent": "Mozilla/5.0...",
      "result": "success"
    }
  ],
  "total": 50,
  "limit": 50
}
```

**Status Codes**:
- `200 OK`: Audit log retrieved
- `401 Unauthorized`: Invalid token
- `403 Forbidden`: Requires admin role

## System Status

### GET /health

Health check endpoint (no authentication required).

**Request**:
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "api": "operational",
    "database": "operational",
    "groq_inference": "operational",
    "agent_orchestration": "operational"
  },
  "timestamp": "2025-11-11T10:30:00Z"
}
```

**Status Codes**:
- `200 OK`: System healthy
- `503 Service Unavailable`: System degraded or down

### GET /metrics

Prometheus metrics endpoint (admin only).

**Request**:
```http
GET /metrics
Authorization: Bearer <admin_token>
```

**Response**:
```
# HELP agent_inference_seconds Agent inference time
# TYPE agent_inference_seconds histogram
agent_inference_seconds_bucket{agent_type="cardiology",le="0.5"} 10
agent_inference_seconds_bucket{agent_type="cardiology",le="1.0"} 45
...
```

**Status Codes**:
- `200 OK`: Metrics retrieved
- `401 Unauthorized`: Invalid token
- `403 Forbidden`: Requires admin role

## Error Responses

All error responses follow this format:

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "The 'priority' field must be one of: low, medium, high, critical",
    "details": {
      "field": "priority",
      "value": "super_urgent",
      "allowed_values": ["low", "medium", "high", "critical"]
    },
    "timestamp": "2025-11-11T10:30:00Z",
    "request_id": "req_abc123"
  }
}
```

**Common Error Codes**:
- `INVALID_INPUT`: Malformed request data
- `AUTHENTICATION_FAILED`: Invalid credentials
- `TOKEN_EXPIRED`: JWT token expired
- `INSUFFICIENT_PERMISSIONS`: User lacks required permissions
- `RESOURCE_NOT_FOUND`: Requested resource doesn't exist
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INTERNAL_ERROR`: Server error
- `PHI_DETECTION_FAILED`: Cannot safely process data
- `AGENT_TIMEOUT`: Agent processing exceeded timeout

## Rate Limits

- **Authentication**: 5 requests/minute per IP
- **Case Creation**: 10 requests/minute per user
- **Case Retrieval**: 60 requests/minute per user
- **Knowledge Search**: 30 requests/minute per user
- **Report Download**: 20 requests/minute per user

Rate limit headers:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 1699704600
```

## OpenAPI Specification

Full OpenAPI 3.0 YAML specification available at:
```
GET /openapi.yaml
```

Import this into Swagger UI or Postman for interactive API exploration.

---

**API Version**: 1.0.0
**Last Updated**: 2025-11-11
**Support**: api-support@agnomedical.example.com
