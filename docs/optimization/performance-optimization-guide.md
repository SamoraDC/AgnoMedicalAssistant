# Performance Optimization Guide - Medical Assistant System

## Executive Summary

This guide provides comprehensive optimization strategies for achieving ultra-low latency (<500ms) in a medical assistant system using Groq LPUs, Agno framework, and multi-agent orchestration.

**Target Performance Metrics:**
- Agent spawn time: <50ms per agent
- Inference latency: <100ms per request (Groq LPU)
- Total system response: <500ms for simple queries
- Multi-agent debate: <2s for complex conflict resolution
- Database query: <10ms for RAG retrieval

---

## 1. Groq LPU Inference Optimization

### 1.1 Model Selection Strategy

```python
# Optimal model selection by use case
MODEL_STRATEGY = {
    "triager": "llama3-70b-8192",      # High accuracy for routing
    "specialists": "llama3-8b-8192",    # Fast specialist analysis
    "validator": "mixtral-8x7b-32768",  # Long context for conflict resolution
    "communicator": "llama3-8b-8192",   # Fast patient summaries
}

# Latency characteristics (Groq benchmarks)
GROQ_LATENCY = {
    "llama3-8b-8192": 40,    # ~40ms average
    "llama3-70b-8192": 80,   # ~80ms average
    "mixtral-8x7b-32768": 60 # ~60ms average
}
```

**Optimization Strategies:**

1. **Request Batching for Parallel Agents**
```python
import asyncio
from groq import AsyncGroq

async def parallel_specialist_inference(case_data, specialists):
    """Execute all specialist inferences in parallel"""
    client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])

    tasks = [
        client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": spec.build_prompt(case_data)}],
            temperature=0.3,  # Lower for medical consistency
            max_tokens=512,   # Limit for faster response
        )
        for spec in specialists
    ]

    results = await asyncio.gather(*tasks)
    return results

# Expected total time: ~40-60ms (not 40ms x N specialists)
```

2. **Streaming Responses for Progressive UI**
```python
async def stream_diagnosis(case_data):
    """Stream tokens as they're generated"""
    client = AsyncGroq()

    stream = await client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": case_data}],
        stream=True,
        max_tokens=1024,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta  # Send to frontend immediately
```

3. **Prompt Engineering for Speed**
```python
# ❌ SLOW: Verbose, unfocused prompt
prompt_slow = """
As a cardiologist with 20 years of experience, please carefully
review the following patient case and provide a detailed analysis...
[5000 characters of context]
"""

# ✅ FAST: Concise, structured prompt
prompt_fast = """
ROLE: Cardiologist
TASK: Analyze cardiac markers
DATA: {troponin: 0.5, BNP: 450, ECG: normal_sinus}
OUTPUT: JSON {diagnosis: str, confidence: float, reasoning: str}
LIMIT: 200 words
"""
```

**Expected Improvement:** 30-50% reduction in inference time

### 1.2 Context Window Management

```python
class ContextOptimizer:
    """Manage context efficiently for Groq models"""

    MAX_CONTEXT = {
        "llama3-8b-8192": 8192,
        "llama3-70b-8192": 8192,
        "mixtral-8x7b-32768": 32768,
    }

    def optimize_context(self, case_data, model):
        """Prioritize most relevant context"""
        max_tokens = self.MAX_CONTEXT[model]

        # Priority ranking
        priority = [
            case_data["chief_complaint"],      # Critical
            case_data["vital_signs"],          # Critical
            case_data["lab_results"],          # High
            case_data["medications"],          # High
            case_data["medical_history"],      # Medium
            case_data["social_history"],       # Low
        ]

        context = []
        token_count = 0

        for item in priority:
            item_tokens = len(item.split()) * 1.3  # Rough estimate
            if token_count + item_tokens < max_tokens * 0.8:  # 80% buffer
                context.append(item)
                token_count += item_tokens
            else:
                break

        return "\n".join(context)
```

**Expected Improvement:** Consistent <100ms inference times

---

## 2. Agent Orchestration Optimization

### 2.1 Parallel Agent Execution Pattern

```python
import agno
from concurrent.futures import ThreadPoolExecutor, as_completed

class ParallelAgentOrchestrator:
    """Optimized agent execution with true parallelism"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.triager = agno.Agent(name="Triager")
        self.specialists = {
            "cardiology": agno.Agent(name="Cardiologist"),
            "endocrinology": agno.Agent(name="Endocrinologist"),
            "neurology": agno.Agent(name="Neurologist"),
        }
        self.validator = agno.Agent(name="Validator")
        self.communicator = agno.Agent(name="Communicator")

    async def process_case(self, case_data):
        """Process medical case with optimal parallelism"""

        # Stage 1: Triager (must run first)
        start = time.time()
        triage_result = await self.triager.run(case_data)
        print(f"Triage: {time.time() - start:.3f}s")

        # Stage 2: Parallel specialist analysis
        start = time.time()
        active_specialists = triage_result["required_specialties"]

        futures = {
            self.executor.submit(
                self.specialists[spec].run,
                case_data,
                triage_result
            ): spec
            for spec in active_specialists
        }

        specialist_results = {}
        for future in as_completed(futures):
            specialty = futures[future]
            specialist_results[specialty] = future.result()

        print(f"Specialists ({len(active_specialists)}): {time.time() - start:.3f}s")

        # Stage 3: Validation (sequential, but fast)
        start = time.time()
        validated = await self.validator.run(specialist_results)
        print(f"Validation: {time.time() - start:.3f}s")

        # Stage 4: Communication (can run in parallel with validation)
        start = time.time()
        report = await self.communicator.run(validated)
        print(f"Communication: {time.time() - start:.3f}s")

        return report
```

**Expected Performance:**
- Triager: 80ms
- 3 Specialists (parallel): 120ms (not 3x120ms)
- Validator: 100ms
- Communicator: 80ms
- **Total: ~380ms**

### 2.2 Asynchronous Communication Patterns

```python
import asyncio
from agno import Agent, Message

class AsyncAgentCommunicator:
    """Non-blocking agent communication with ACP"""

    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.response_futures = {}

    async def send_message(self, from_agent, to_agent, content):
        """Send message without blocking"""
        message_id = str(uuid.uuid4())
        message = Message(
            id=message_id,
            from_agent=from_agent,
            to_agent=to_agent,
            content=content,
            protocol="ACP/1.0"
        )

        future = asyncio.Future()
        self.response_futures[message_id] = future
        await self.message_queue.put(message)

        return await future  # Awaitable response

    async def process_messages(self):
        """Background task for message processing"""
        while True:
            message = await self.message_queue.get()

            # Process in background
            asyncio.create_task(self._handle_message(message))

    async def _handle_message(self, message):
        """Handle individual message"""
        agent = self.get_agent(message.to_agent)
        response = await agent.process_message(message)

        # Complete the future
        if message.id in self.response_futures:
            self.response_futures[message.id].set_result(response)
            del self.response_futures[message.id]
```

**Expected Improvement:** 40-60% reduction in multi-agent debate time

### 2.3 Resource Allocation Strategy

```python
class ResourceAllocator:
    """Intelligent resource allocation for agents"""

    AGENT_PRIORITY = {
        "triager": 10,      # Always run immediately
        "cardiology": 8,    # Critical specialties
        "neurology": 8,
        "endocrinology": 6,
        "communicator": 5,
        "validator": 9,
    }

    def __init__(self, max_concurrent=10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_agents = {}

    async def allocate(self, agent_name, task):
        """Allocate resources with priority queue"""
        priority = self.AGENT_PRIORITY.get(agent_name, 5)

        async with self.semaphore:
            start = time.time()
            result = await task()
            duration = time.time() - start

            # Track resource usage
            self.active_agents[agent_name] = {
                "duration": duration,
                "priority": priority,
            }

            return result
```

---

## 3. Database Performance Optimization

### 3.1 DuckDB Query Optimization

```python
import duckdb
import numpy as np

class OptimizedMedicalKnowledgeBase:
    """High-performance DuckDB with vector search"""

    def __init__(self, db_path="medical_kb.duckdb"):
        self.conn = duckdb.connect(db_path)
        self._initialize_schema()
        self._create_indexes()

    def _initialize_schema(self):
        """Optimized schema for medical knowledge"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS medical_cases (
                case_id INTEGER PRIMARY KEY,
                patient_age INTEGER,
                chief_complaint VARCHAR,
                diagnosis VARCHAR,
                specialty VARCHAR,
                embedding FLOAT[768],  -- Vector embedding
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS medical_guidelines (
                guideline_id INTEGER PRIMARY KEY,
                specialty VARCHAR,
                title VARCHAR,
                content TEXT,
                embedding FLOAT[768],
                last_updated TIMESTAMP
            )
        """)

    def _create_indexes(self):
        """Create indexes for fast retrieval"""
        # Standard B-tree indexes
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_specialty
            ON medical_cases(specialty)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_diagnosis
            ON medical_cases(diagnosis)
        """)

        # Vector similarity index using vss extension
        self.conn.execute("INSTALL vss")
        self.conn.execute("LOAD vss")

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_case_embedding
            ON medical_cases
            USING HNSW (embedding)
            WITH (metric = 'cosine')
        """)

    async def similarity_search(self, query_embedding, specialty=None, top_k=5):
        """Ultra-fast vector similarity search"""

        query = """
            SELECT
                case_id,
                chief_complaint,
                diagnosis,
                1 - array_cosine_similarity(embedding, $1::FLOAT[768]) as similarity
            FROM medical_cases
        """

        if specialty:
            query += " WHERE specialty = $2"
            params = [query_embedding, specialty]
        else:
            params = [query_embedding]

        query += f"""
            ORDER BY similarity DESC
            LIMIT {top_k}
        """

        results = self.conn.execute(query, params).fetchall()
        return results

    async def batch_retrieve(self, case_ids):
        """Batch retrieval for multiple cases"""
        query = """
            SELECT * FROM medical_cases
            WHERE case_id IN (SELECT unnest($1::INTEGER[]))
        """

        results = self.conn.execute(query, [case_ids]).fetchall()
        return results
```

**Expected Performance:**
- Similarity search: <10ms for 1M records
- Batch retrieval: <5ms for 100 cases
- Index creation: One-time cost, ~30s for 1M records

### 3.2 Caching Strategy

```python
from functools import lru_cache
import redis
import pickle

class MultiTierCache:
    """Multi-tier caching for medical knowledge"""

    def __init__(self):
        # Tier 1: In-memory LRU cache (Python)
        self.memory_cache = {}
        self.max_memory_entries = 1000

        # Tier 2: Redis cache (fast, shared)
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=False
        )

        # Tier 3: DuckDB (persistent)
        self.db = OptimizedMedicalKnowledgeBase()

    @lru_cache(maxsize=1000)
    def get_guideline(self, guideline_id):
        """Get guideline with automatic caching"""
        # Tier 1: Memory (< 1ms)
        if guideline_id in self.memory_cache:
            return self.memory_cache[guideline_id]

        # Tier 2: Redis (~2ms)
        cached = self.redis_client.get(f"guideline:{guideline_id}")
        if cached:
            data = pickle.loads(cached)
            self.memory_cache[guideline_id] = data
            return data

        # Tier 3: Database (~10ms)
        data = self.db.get_guideline(guideline_id)

        # Populate caches
        self.redis_client.setex(
            f"guideline:{guideline_id}",
            3600,  # 1 hour TTL
            pickle.dumps(data)
        )
        self.memory_cache[guideline_id] = data

        return data

    async def warmup_cache(self, common_specialties):
        """Pre-warm cache with common guidelines"""
        for specialty in common_specialties:
            guidelines = await self.db.get_guidelines_by_specialty(specialty)

            for guideline in guidelines:
                self.redis_client.setex(
                    f"guideline:{guideline['id']}",
                    3600,
                    pickle.dumps(guideline)
                )
```

**Expected Improvement:**
- Cache hit rate: >80% for common cases
- Latency reduction: 10ms → 1ms for cached queries

---

## 4. Context Efficiency Optimization

### 4.1 Token Usage Minimization

```python
class TokenOptimizer:
    """Minimize token usage while preserving quality"""

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def compress_medical_record(self, record):
        """Compress medical record for efficient processing"""

        # Extract only critical information
        compressed = {
            "cc": record["chief_complaint"][:200],  # Chief complaint
            "hpi": self._summarize_hpi(record["history_present_illness"]),
            "pmh": self._extract_key_history(record["past_medical_history"]),
            "meds": self._abbreviate_medications(record["medications"]),
            "vs": record["vital_signs"],  # Always include full vitals
            "labs": self._select_abnormal_labs(record["lab_results"]),
        }

        return compressed

    def _summarize_hpi(self, hpi_text):
        """Summarize history of present illness"""
        # Extract key symptoms, duration, severity
        sentences = hpi_text.split(". ")
        key_sentences = [
            s for s in sentences
            if any(keyword in s.lower() for keyword in
                   ["pain", "started", "worse", "associated", "denies"])
        ]
        return ". ".join(key_sentences[:5])

    def _select_abnormal_labs(self, labs):
        """Include only abnormal lab values"""
        abnormal = {}

        for test, value in labs.items():
            if self._is_abnormal(test, value):
                abnormal[test] = value

        return abnormal

    def estimate_tokens(self, text):
        """Estimate token count"""
        return len(self.tokenizer.encode(text))
```

**Expected Improvement:** 40-60% reduction in token usage

### 4.2 Shared Memory Optimization

```python
import mmap
import json

class SharedMemoryManager:
    """Share context between agents without duplication"""

    def __init__(self, memory_size=1024*1024*100):  # 100MB
        self.memory_size = memory_size
        self.shared_mem = mmap.mmap(-1, memory_size)
        self.offsets = {}

    def store_case(self, case_id, case_data):
        """Store case in shared memory"""
        serialized = json.dumps(case_data).encode('utf-8')

        # Write to shared memory
        offset = len(self.offsets) * 1024  # 1KB per case offset
        self.shared_mem.seek(offset)
        self.shared_mem.write(serialized)

        self.offsets[case_id] = {
            "offset": offset,
            "length": len(serialized)
        }

    def get_case(self, case_id):
        """Retrieve case from shared memory"""
        if case_id not in self.offsets:
            return None

        metadata = self.offsets[case_id]
        self.shared_mem.seek(metadata["offset"])
        serialized = self.shared_mem.read(metadata["length"])

        return json.loads(serialized.decode('utf-8'))

    def cleanup(self):
        """Clean up shared memory"""
        self.shared_mem.close()
```

**Expected Improvement:** Eliminate redundant case data passing between agents

### 4.3 Prompt Compression

```python
class PromptCompressor:
    """Compress prompts while maintaining quality"""

    MEDICAL_ABBREVIATIONS = {
        "history of present illness": "HPI",
        "past medical history": "PMH",
        "chief complaint": "CC",
        "review of systems": "ROS",
        "vital signs": "VS",
        "blood pressure": "BP",
        "heart rate": "HR",
        "respiratory rate": "RR",
        "temperature": "Temp",
    }

    def compress(self, prompt):
        """Compress prompt using medical abbreviations"""
        compressed = prompt

        for full, abbr in self.MEDICAL_ABBREVIATIONS.items():
            compressed = compressed.replace(full, abbr)

        # Remove unnecessary whitespace
        compressed = " ".join(compressed.split())

        return compressed

    def compress_with_templates(self, case_type, data):
        """Use templates for common case types"""

        templates = {
            "chest_pain": """
                CC: {chief_complaint}
                VS: BP {bp}, HR {hr}
                Cardiac: {cardiac_findings}
                ECG: {ecg}
                Trop: {troponin}
            """,
            "shortness_of_breath": """
                CC: {chief_complaint}
                VS: RR {rr}, O2 {oxygen_sat}
                Lungs: {lung_findings}
                CXR: {chest_xray}
            """
        }

        template = templates.get(case_type, "{raw_data}")
        return template.format(**data).strip()
```

**Expected Improvement:** 30-40% reduction in prompt tokens

---

## 5. End-to-End Performance Benchmarking

### 5.1 Comprehensive Benchmark Suite

```python
import time
import statistics
from typing import List, Dict

class PerformanceBenchmark:
    """Comprehensive performance testing"""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.results = []

    async def run_benchmark(self, test_cases: List[Dict], iterations=10):
        """Run full benchmark suite"""

        print("=" * 60)
        print("MEDICAL ASSISTANT PERFORMANCE BENCHMARK")
        print("=" * 60)

        for test_case in test_cases:
            print(f"\nTest: {test_case['name']}")
            print("-" * 60)

            latencies = []

            for i in range(iterations):
                start = time.time()
                result = await self.orchestrator.process_case(test_case['data'])
                latency = (time.time() - start) * 1000  # Convert to ms

                latencies.append(latency)
                print(f"  Iteration {i+1}: {latency:.2f}ms")

            # Calculate statistics
            stats = {
                "test": test_case['name'],
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "p95": self._percentile(latencies, 95),
                "p99": self._percentile(latencies, 99),
                "min": min(latencies),
                "max": max(latencies),
            }

            self.results.append(stats)

            print(f"\n  Statistics:")
            print(f"    Mean:   {stats['mean']:.2f}ms")
            print(f"    Median: {stats['median']:.2f}ms")
            print(f"    P95:    {stats['p95']:.2f}ms")
            print(f"    P99:    {stats['p99']:.2f}ms")

        self._print_summary()

    def _percentile(self, data, percentile):
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[index]

    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        total_mean = statistics.mean([r['mean'] for r in self.results])
        total_p95 = statistics.mean([r['p95'] for r in self.results])

        print(f"Average Mean Latency: {total_mean:.2f}ms")
        print(f"Average P95 Latency:  {total_p95:.2f}ms")

        if total_mean < 500:
            print("\n✅ PERFORMANCE TARGET MET (<500ms)")
        else:
            print(f"\n❌ PERFORMANCE TARGET MISSED (target: 500ms, actual: {total_mean:.2f}ms)")

# Example usage
async def main():
    orchestrator = ParallelAgentOrchestrator()
    benchmark = PerformanceBenchmark(orchestrator)

    test_cases = [
        {
            "name": "Simple Chest Pain Case",
            "data": {
                "chief_complaint": "Chest pain",
                "vital_signs": {"bp": "140/90", "hr": 88},
                "troponin": 0.03,
            }
        },
        {
            "name": "Complex Multi-System Case",
            "data": {
                "chief_complaint": "Weakness and confusion",
                "vital_signs": {"bp": "100/60", "hr": 110, "temp": 101.2},
                "lab_results": {"glucose": 450, "sodium": 128},
            }
        }
    ]

    await benchmark.run_benchmark(test_cases, iterations=10)
```

### 5.2 Component-Level Profiling

```python
import cProfile
import pstats
from functools import wraps

def profile_function(func):
    """Decorator for profiling individual functions"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        result = await func(*args, **kwargs)

        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')

        print(f"\nProfile for {func.__name__}:")
        stats.print_stats(10)  # Top 10 functions

        return result

    return wrapper

# Usage
@profile_function
async def process_medical_case(case_data):
    # Function implementation
    pass
```

---

## 6. Monitoring and Continuous Optimization

### 6.1 Real-Time Performance Monitoring

```python
from prometheus_client import Counter, Histogram, Gauge
import time

class PerformanceMonitor:
    """Real-time performance monitoring with Prometheus"""

    def __init__(self):
        # Metrics
        self.request_count = Counter(
            'medical_assistant_requests_total',
            'Total number of case processing requests'
        )

        self.request_latency = Histogram(
            'medical_assistant_latency_seconds',
            'Request latency in seconds',
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )

        self.active_agents = Gauge(
            'medical_assistant_active_agents',
            'Number of active agents'
        )

        self.groq_api_latency = Histogram(
            'groq_api_latency_seconds',
            'Groq API latency',
            buckets=[0.05, 0.1, 0.15, 0.2, 0.3]
        )

    def track_request(self, func):
        """Decorator to track request metrics"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            self.request_count.inc()

            start = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start

            self.request_latency.observe(duration)

            return result

        return wrapper
```

### 6.2 Alerting Rules

```yaml
# Prometheus alerting rules (alerts.yml)
groups:
  - name: medical_assistant_performance
    interval: 30s
    rules:
      - alert: HighLatency
        expr: medical_assistant_latency_seconds{quantile="0.95"} > 0.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "P95 latency is {{ $value }}s (threshold: 0.5s)"

      - alert: CriticalLatency
        expr: medical_assistant_latency_seconds{quantile="0.95"} > 1.0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Critical latency detected"
          description: "P95 latency is {{ $value }}s (threshold: 1.0s)"

      - alert: GroqAPISlowdown
        expr: groq_api_latency_seconds{quantile="0.95"} > 0.2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Groq API experiencing slowdown"
          description: "P95 Groq latency is {{ $value }}s (threshold: 0.2s)"
```

---

## 7. Performance Optimization Checklist

### Pre-Production Checklist

- [ ] **Groq API optimization**
  - [ ] Model selection optimized per agent type
  - [ ] Parallel inference implemented
  - [ ] Streaming responses enabled
  - [ ] Prompt compression applied
  - [ ] Context window management implemented

- [ ] **Agent orchestration**
  - [ ] Parallel agent execution verified
  - [ ] Asynchronous communication patterns implemented
  - [ ] Resource allocation strategy defined
  - [ ] Load balancing configured

- [ ] **Database optimization**
  - [ ] DuckDB indexes created
  - [ ] Vector search benchmarked (<10ms)
  - [ ] Batch retrieval implemented
  - [ ] Multi-tier caching deployed

- [ ] **Memory management**
  - [ ] Shared memory implemented
  - [ ] Token usage minimized
  - [ ] Cache hit rate >80%

- [ ] **Monitoring**
  - [ ] Prometheus metrics enabled
  - [ ] Alerting rules configured
  - [ ] Performance dashboard deployed
  - [ ] Benchmark suite automated

---

## 8. Expected Performance Summary

| Component | Target | Optimized |
|-----------|--------|-----------|
| Triager Agent | <100ms | ~80ms |
| Specialist Agent (single) | <100ms | ~40ms |
| 3 Specialists (parallel) | <300ms | ~120ms |
| Validator Agent | <150ms | ~100ms |
| Communicator Agent | <100ms | ~80ms |
| Database Query (RAG) | <20ms | ~8ms |
| **Total Simple Case** | **<500ms** | **~380ms** |
| **Total Complex Case** | **<2000ms** | **~1200ms** |

---

## 9. Next Steps

1. **Implement core optimizations** (Week 1)
   - Groq parallel inference
   - DuckDB with vector indexes
   - Basic caching layer

2. **Deploy monitoring** (Week 2)
   - Prometheus metrics
   - Performance dashboard
   - Alerting rules

3. **Benchmark and tune** (Week 3)
   - Run comprehensive benchmarks
   - Identify bottlenecks
   - Fine-tune parameters

4. **Production optimization** (Week 4)
   - Load testing
   - Cache warming
   - Continuous monitoring

---

## Contact

For questions about this optimization guide, contact the performance engineering team.

**Last Updated:** 2025-11-11
**Version:** 1.0.0
