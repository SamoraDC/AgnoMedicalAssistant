# Deployment Guide

Production deployment procedures for the Agno Medical Assistant system with HIPAA/LGPD compliance.

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Security Hardening](#security-hardening)
6. [Monitoring & Observability](#monitoring--observability)
7. [Backup & Recovery](#backup--recovery)
8. [Scaling Strategies](#scaling-strategies)
9. [Incident Response](#incident-response)

## Pre-Deployment Checklist

### Security Compliance

- [ ] **HIPAA Business Associate Agreement** (BAA) signed with all cloud providers
- [ ] **Data encryption** verified (AES-256 at rest, TLS 1.3 in transit)
- [ ] **Access controls** configured (RBAC, principle of least privilege)
- [ ] **Audit logging** enabled for all data access
- [ ] **GuardRails AI** configured for PHI/PII detection
- [ ] **Vulnerability scan** passed (no high/critical vulnerabilities)
- [ ] **Penetration testing** completed
- [ ] **Security incident response plan** documented

### Technical Readiness

- [ ] **Load testing** completed (target: 100 concurrent cases)
- [ ] **Failover testing** validated
- [ ] **Backup restoration** tested
- [ ] **Monitoring dashboards** configured
- [ ] **Alert thresholds** set
- [ ] **Runbook** documentation complete
- [ ] **On-call rotation** established

### Legal & Compliance

- [ ] **Privacy policy** reviewed by legal
- [ ] **Terms of service** finalized
- [ ] **HIPAA compliance** verified
- [ ] **LGPD compliance** verified (if applicable)
- [ ] **Data retention policy** implemented
- [ ] **Data deletion procedures** tested

## Infrastructure Requirements

### Minimum Production Specs

#### Application Servers (3+ instances)

```yaml
CPU: 4 vCPUs
RAM: 16 GB
Storage: 100 GB SSD
Network: 10 Gbps
OS: Ubuntu 22.04 LTS
```

#### Database Server (DuckDB + Backups)

```yaml
CPU: 8 vCPUs
RAM: 32 GB
Storage: 500 GB NVMe SSD (data) + 2 TB HDD (backups)
Network: 10 Gbps
```

#### Load Balancer

```yaml
Type: Application Load Balancer (Layer 7)
SSL/TLS: Certificate from trusted CA
Health checks: Enabled on /health endpoint
Sticky sessions: Enabled (for WebRTC)
```

### Cloud Provider Recommendations

#### AWS Configuration

```yaml
Application: EC2 t3.xlarge (or ECS Fargate)
Database: EBS gp3 volumes with encryption
Load Balancer: Application Load Balancer
Secrets: AWS Secrets Manager
Monitoring: CloudWatch + CloudTrail
Backup: S3 with versioning + lifecycle policies
```

#### Azure Configuration

```yaml
Application: Azure Container Instances or App Service
Database: Azure Managed Disks (Premium SSD)
Load Balancer: Azure Application Gateway
Secrets: Azure Key Vault
Monitoring: Azure Monitor + Log Analytics
Backup: Azure Blob Storage with immutable storage
```

#### GCP Configuration

```yaml
Application: Google Kubernetes Engine (GKE) or Cloud Run
Database: Persistent Disk SSD
Load Balancer: Google Cloud Load Balancing
Secrets: Google Secret Manager
Monitoring: Google Cloud Monitoring
Backup: Google Cloud Storage with object versioning
```

## Docker Deployment

### Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install UV
RUN pip install uv

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv add fastapi uvicorn agno groq \
    python-multipart pymupdf pytesseract \
    duckdb duckdb-engine sentence-transformers \
    guardrails-ai edge-tts \
    python-jose[cryptography] passlib[bcrypt] \
    websockets aiortc

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create data directories
RUN mkdir -p /app/data /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD [".venv/bin/uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    image: agnomedical:latest
    container_name: agno_api
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - SECRET_KEY=${SECRET_KEY}
      - DATABASE_URL=duckdb:///data/medical.db
      - ENCRYPT_AT_REST=true
      - AUDIT_LOG_ENABLED=true
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    networks:
      - agno_network

  nginx:
    image: nginx:alpine
    container_name: agno_nginx
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    restart: unless-stopped
    networks:
      - agno_network

networks:
  agno_network:
    driver: bridge
```

### NGINX Configuration

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        least_conn;
        server api:8000 max_fails=3 fail_timeout=30s;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/m;
    limit_req_zone $binary_remote_addr zone=auth_limit:10m rate=5r/m;

    server {
        listen 80;
        server_name agnomedical.example.com;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name agnomedical.example.com;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/fullchain.pem;
        ssl_certificate_key /etc/nginx/ssl/privkey.pem;
        ssl_protocols TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;

        # Security headers
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        add_header X-Frame-Options "DENY" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Content-Security-Policy "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';" always;

        # File upload size
        client_max_body_size 50M;

        # Authentication endpoints (stricter rate limit)
        location /auth/ {
            limit_req zone=auth_limit burst=5;
            proxy_pass http://api_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # API endpoints
        location /v1/ {
            limit_req zone=api_limit burst=20;
            proxy_pass http://api_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Timeouts for long-running requests
            proxy_connect_timeout 60s;
            proxy_send_timeout 120s;
            proxy_read_timeout 120s;
        }

        # WebSocket for teleconsultation
        location /teleconsult/ {
            proxy_pass http://api_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_read_timeout 3600s;
        }

        # Health check (no rate limit)
        location /health {
            proxy_pass http://api_backend;
            access_log off;
        }
    }
}
```

### Build and Deploy

```bash
# Build image
docker build -t agnomedical:latest .

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## Kubernetes Deployment

### Kubernetes Manifests

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agno-api
  namespace: agnomedical
  labels:
    app: agno-api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: agno-api
  template:
    metadata:
      labels:
        app: agno-api
        version: v1.0.0
    spec:
      containers:
      - name: api
        image: agnomedical:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: GROQ_API_KEY
          valueFrom:
            secretKeyRef:
              name: agno-secrets
              key: groq-api-key
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: agno-secrets
              key: jwt-secret-key
        - name: DATABASE_URL
          value: "duckdb:///data/medical.db"
        resources:
          requests:
            cpu: "1000m"
            memory: "4Gi"
          limits:
            cpu: "2000m"
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: agno-data-pvc
      - name: logs-volume
        persistentVolumeClaim:
          claimName: agno-logs-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: agno-api-service
  namespace: agnomedical
spec:
  type: ClusterIP
  selector:
    app: agno-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: agno-data-pvc
  namespace: agnomedical
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 500Gi
  storageClassName: fast-ssd
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: agno-logs-pvc
  namespace: agnomedical
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: standard
```

Create `k8s/ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agno-ingress
  namespace: agnomedical
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/limit-rps: "10"
spec:
  tls:
  - hosts:
    - agnomedical.example.com
    secretName: agno-tls-secret
  rules:
  - host: agnomedical.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: agno-api-service
            port:
              number: 80
```

Create `k8s/secrets.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: agno-secrets
  namespace: agnomedical
type: Opaque
data:
  groq-api-key: <base64-encoded-groq-key>
  jwt-secret-key: <base64-encoded-jwt-secret>
```

### Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace agnomedical

# Create secrets
kubectl apply -f k8s/secrets.yaml

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/ingress.yaml

# Check status
kubectl get pods -n agnomedical
kubectl get svc -n agnomedical
kubectl get ingress -n agnomedical

# View logs
kubectl logs -f deployment/agno-api -n agnomedical

# Scale deployment
kubectl scale deployment/agno-api --replicas=5 -n agnomedical
```

## Security Hardening

### 1. Network Security

```bash
# AWS Security Group (example)
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxx \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0

# Block all other ports
aws ec2 revoke-security-group-ingress \
  --group-id sg-xxxx \
  --protocol tcp \
  --port 0-65535 \
  --cidr 0.0.0.0/0
```

### 2. Application Security

Enable security features in `src/core/config.py`:

```python
class SecurityConfig:
    # JWT settings
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRY_MINUTES = 30
    JWT_REFRESH_DAYS = 7

    # Rate limiting
    RATE_LIMIT_PER_MINUTE = 10
    RATE_LIMIT_PER_HOUR = 100

    # Password policy
    MIN_PASSWORD_LENGTH = 12
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGITS = True
    REQUIRE_SPECIAL = True

    # Session management
    SESSION_TIMEOUT_MINUTES = 30
    MAX_CONCURRENT_SESSIONS = 3

    # PHI protection
    GUARDRAILS_ENABLED = True
    PHI_ANONYMIZATION = True
    AUDIT_ALL_ACCESS = True
```

### 3. Database Encryption

```python
# Enable encryption at rest for DuckDB
import duckdb

con = duckdb.connect('data/medical.db')

# Set encryption key
con.execute("PRAGMA set_encryption_key='your-encryption-key'")

# Enable encryption
con.execute("PRAGMA enable_encryption=true")
```

### 4. Secrets Rotation

```bash
# Rotate JWT secret (example script)
python scripts/rotate_jwt_secret.py

# Rotate API keys
python scripts/rotate_api_keys.py

# Update all active sessions
python scripts/invalidate_sessions.py
```

## Monitoring & Observability

### Prometheus Metrics

Create `src/utils/metrics.py`:

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# Agent metrics
agent_inference_duration = Histogram(
    'agent_inference_seconds',
    'Agent inference time',
    ['agent_type']
)

agent_confidence_score = Histogram(
    'agent_confidence_score',
    'Agent confidence scores',
    ['agent_type']
)

# Case processing metrics
case_processing_duration = Histogram(
    'case_processing_seconds',
    'Total case processing time'
)

cases_total = Counter(
    'cases_total',
    'Total cases processed',
    ['status', 'priority']
)

# System health
active_agents = Gauge(
    'active_agents',
    'Number of active agents',
    ['agent_type']
)

database_connections = Gauge(
    'database_connections',
    'Number of database connections'
)
```

### Grafana Dashboard

Import dashboard from `monitoring/grafana-dashboard.json`:

```json
{
  "dashboard": {
    "title": "Agno Medical Assistant",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Agent Performance",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, agent_inference_seconds)"
          }
        ]
      },
      {
        "title": "Case Processing Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, case_processing_seconds)"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules

Create `monitoring/alerts.yaml`:

```yaml
groups:
  - name: agno_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} requests/second"

      # Slow agent inference
      - alert: SlowAgentInference
        expr: histogram_quantile(0.95, agent_inference_seconds) > 2.0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Agent inference is slow"
          description: "95th percentile is {{ $value }} seconds"

      # Database connection issues
      - alert: DatabaseConnectionHigh
        expr: database_connections > 50
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High database connection count"
          description: "{{ $value }} active connections"
```

## Backup & Recovery

### Automated Backup Script

Create `scripts/backup.sh`:

```bash
#!/bin/bash
set -e

# Configuration
BACKUP_DIR="/backups"
DATABASE_PATH="/app/data/medical.db"
S3_BUCKET="s3://agno-backups"
RETENTION_DAYS=30

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="medical_${TIMESTAMP}.db"

# Create backup
echo "Creating backup: ${BACKUP_NAME}"
cp "${DATABASE_PATH}" "${BACKUP_DIR}/${BACKUP_NAME}"

# Compress backup
echo "Compressing backup..."
gzip "${BACKUP_DIR}/${BACKUP_NAME}"

# Encrypt backup
echo "Encrypting backup..."
openssl enc -aes-256-cbc -salt \
  -in "${BACKUP_DIR}/${BACKUP_NAME}.gz" \
  -out "${BACKUP_DIR}/${BACKUP_NAME}.gz.enc" \
  -pass pass:${BACKUP_ENCRYPTION_KEY}

# Upload to S3
echo "Uploading to S3..."
aws s3 cp "${BACKUP_DIR}/${BACKUP_NAME}.gz.enc" "${S3_BUCKET}/"

# Clean up local backup
rm "${BACKUP_DIR}/${BACKUP_NAME}.gz.enc"
rm "${BACKUP_DIR}/${BACKUP_NAME}.gz"

# Delete old backups from S3
echo "Cleaning up old backups..."
aws s3 ls "${S3_BUCKET}/" | \
  awk '{print $4}' | \
  head -n -${RETENTION_DAYS} | \
  xargs -I {} aws s3 rm "${S3_BUCKET}/{}"

echo "Backup complete: ${BACKUP_NAME}.gz.enc"
```

### Recovery Procedure

```bash
# Download backup from S3
aws s3 cp s3://agno-backups/medical_20240115_120000.db.gz.enc /tmp/

# Decrypt backup
openssl enc -aes-256-cbc -d \
  -in /tmp/medical_20240115_120000.db.gz.enc \
  -out /tmp/medical_20240115_120000.db.gz \
  -pass pass:${BACKUP_ENCRYPTION_KEY}

# Decompress
gunzip /tmp/medical_20240115_120000.db.gz

# Stop application
docker-compose down

# Restore database
cp /tmp/medical_20240115_120000.db /app/data/medical.db

# Start application
docker-compose up -d

# Verify restoration
curl -f http://localhost:8000/health
```

### Disaster Recovery Plan

1. **RPO (Recovery Point Objective)**: 1 hour (hourly backups)
2. **RTO (Recovery Time Objective)**: 4 hours (time to restore service)

**Recovery Steps**:
1. Notify stakeholders of outage
2. Assess extent of data loss
3. Provision new infrastructure (if needed)
4. Restore latest backup
5. Verify data integrity
6. Resume service
7. Post-mortem analysis

## Scaling Strategies

### Horizontal Scaling

```bash
# Kubernetes auto-scaling
kubectl autoscale deployment agno-api \
  --cpu-percent=70 \
  --min=3 \
  --max=10 \
  -n agnomedical
```

### Load Testing

```bash
# Install k6
brew install k6

# Run load test
k6 run scripts/load_test.js

# Expected results:
# - p95 latency < 2s
# - Error rate < 1%
# - 100 concurrent users
```

Load test script `scripts/load_test.js`:

```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '5m', target: 50 },  // Ramp up
    { duration: '10m', target: 100 }, // Stay at 100 users
    { duration: '5m', target: 0 },   // Ramp down
  ],
  thresholds: {
    'http_req_duration': ['p(95)<2000'], // 95% under 2s
    'http_req_failed': ['rate<0.01'],    // Error rate < 1%
  },
};

export default function () {
  // Simulate case creation
  let payload = JSON.stringify({
    patient_id: 'pat_test_' + __VU,
    priority: 'medium',
    clinical_notes: 'Test case for load testing',
  });

  let params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer test_token',
    },
  };

  let res = http.post('https://api.agnomedical.example.com/v1/cases', payload, params);

  check(res, {
    'status is 201': (r) => r.status === 201,
    'response time < 2s': (r) => r.timings.duration < 2000,
  });

  sleep(1);
}
```

## Incident Response

### On-Call Procedures

1. **Alert Received**:
   - Acknowledge alert within 5 minutes
   - Assess severity (P1-P4)
   - Notify team if P1/P2

2. **Investigation**:
   - Check monitoring dashboards
   - Review recent deployments
   - Check error logs
   - Identify root cause

3. **Mitigation**:
   - Implement fix or rollback
   - Verify resolution
   - Update status page

4. **Post-Mortem** (within 24 hours):
   - Document timeline
   - Identify root cause
   - Action items to prevent recurrence

### Rollback Procedure

```bash
# Kubernetes rollback
kubectl rollout undo deployment/agno-api -n agnomedical

# Docker rollback
docker-compose down
docker tag agnomedical:previous agnomedical:latest
docker-compose up -d

# Verify rollback
curl -f https://api.agnomedical.example.com/health
```

---

**Last Updated**: 2025-11-11
**Maintained By**: DevOps Team
**Emergency Contact**: oncall@agnomedical.example.com
