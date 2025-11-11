# Latency Reduction Techniques - Medical Assistant System

## Overview

This document provides actionable techniques for achieving and maintaining ultra-low latency (<500ms) in the medical assistant system.

---

## 1. Critical Path Analysis

### 1.1 Identify Bottlenecks

```python
# src/utils/latency_profiler.py

import time
import functools
from collections import defaultdict

class LatencyProfiler:
    """Profile latency across the critical path"""

    def __init__(self):
        self.measurements = defaultdict(list)

    def measure(self, component_name):
        """Decorator to measure component latency"""

        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.perf_counter()

                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    latency = (time.perf_counter() - start) * 1000
                    self.measurements[component_name].append(latency)

                    # Log slow operations
                    if latency > 100:
                        print(f"⚠️  Slow operation: {component_name} took {latency:.2f}ms")

            return wrapper
        return decorator

    def get_critical_path(self):
        """Identify components on critical path"""

        summary = {}

        for component, latencies in self.measurements.items():
            summary[component] = {
                "count": len(latencies),
                "mean": sum(latencies) / len(latencies),
                "max": max(latencies),
                "total": sum(latencies),
            }

        # Sort by total time
        sorted_components = sorted(
            summary.items(),
            key=lambda x: x[1]["total"],
            reverse=True
        )

        return sorted_components

# Usage
profiler = LatencyProfiler()

@profiler.measure("triager")
async def run_triager(case_data):
    # Implementation
    pass

@profiler.measure("specialist_cardiology")
async def run_cardiologist(case_data):
    # Implementation
    pass
```

---

## 2. Groq LPU Optimization

### 2.1 Minimize Token Count

```python
class TokenMinimizer:
    """Minimize token count in prompts"""

    ABBREVIATIONS = {
        "chief complaint": "CC",
        "history of present illness": "HPI",
        "past medical history": "PMH",
        "blood pressure": "BP",
        "heart rate": "HR",
        "respiratory rate": "RR",
        "temperature": "T",
        "years old": "y/o",
    }

    def minimize_prompt(self, prompt):
        """Reduce token count"""

        # Apply abbreviations
        for full, abbr in self.ABBREVIATIONS.items():
            prompt = prompt.replace(full, abbr)

        # Remove unnecessary whitespace
        prompt = " ".join(prompt.split())

        # Remove filler words
        fillers = ["please", "kindly", "very", "really", "just"]
        for filler in fillers:
            prompt = prompt.replace(f" {filler} ", " ")

        return prompt

    def extract_essentials(self, case_data):
        """Extract only essential information"""

        essentials = {
            "age": case_data.get("age"),
            "sex": case_data.get("sex"),
            "cc": case_data.get("chief_complaint", "")[:100],  # Limit to 100 chars
            "vs": case_data.get("vital_signs", {}),
            "labs": {
                k: v for k, v in case_data.get("lab_results", {}).items()
                if self._is_abnormal(k, v)
            },
        }

        return essentials
```

### 2.2 Parallel Inference

```python
class ParallelInferenceOptimizer:
    """Optimize parallel Groq inference"""

    async def parallel_specialists(self, case_data, specialists):
        """Run all specialists in true parallel"""

        # Prepare all prompts first (no I/O)
        prompts = [
            self._prepare_specialist_prompt(specialist, case_data)
            for specialist in specialists
        ]

        # Execute all at once
        client = AsyncGroq()

        tasks = [
            client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.3,
            )
            for prompt in prompts
        ]

        # All requests happen in parallel
        results = await asyncio.gather(*tasks)

        return results

    async def pipeline_execution(self, case_data):
        """Pipeline execution for sequential dependencies"""

        # Stage 1: Triager (must complete first)
        triager_task = self._run_triager(case_data)

        # Stage 2: Prepare specialists while triager runs
        specialist_prep_task = self._prepare_specialists()

        # Wait for triager
        triage_result = await triager_task
        specialists = await specialist_prep_task

        # Stage 3: Run relevant specialists in parallel
        active_specialists = [
            s for s in specialists
            if s.specialty in triage_result["required_specialties"]
        ]

        specialist_results = await self.parallel_specialists(
            case_data,
            active_specialists
        )

        return specialist_results
```

### 2.3 Response Streaming

```python
class StreamingResponseHandler:
    """Handle streaming responses for progressive UI"""

    async def stream_diagnosis(self, case_data, callback):
        """Stream diagnosis as tokens arrive"""

        client = AsyncGroq()

        stream = await client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": self._build_prompt(case_data)}],
            stream=True,
            max_tokens=1024,
        )

        buffer = ""

        async for chunk in stream:
            delta = chunk.choices[0].delta.content

            if delta:
                buffer += delta

                # Send complete sentences to frontend
                if any(delimiter in buffer for delimiter in [". ", "! ", "? ", "\n"]):
                    await callback(buffer)
                    buffer = ""

        # Send remaining content
        if buffer:
            await callback(buffer)
```

---

## 3. Database Query Optimization

### 3.1 Query Preparation and Caching

```python
class QueryOptimizer:
    """Optimize database queries"""

    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.prepared_statements = {}

    def prepare_query(self, query_name, query):
        """Prepare query for repeated execution"""

        # DuckDB doesn't support prepared statements directly,
        # but we can optimize by pre-compiling
        self.prepared_statements[query_name] = query

    async def execute_prepared(self, query_name, params):
        """Execute prepared query"""

        if query_name not in self.prepared_statements:
            raise ValueError(f"Query '{query_name}' not prepared")

        query = self.prepared_statements[query_name]

        async with self.db_pool.acquire() as conn:
            return conn.execute(query, params).fetchall()

    async def parallel_queries(self, queries):
        """Execute multiple queries in parallel"""

        tasks = [
            self.execute_prepared(name, params)
            for name, params in queries
        ]

        results = await asyncio.gather(*tasks)
        return results
```

### 3.2 Index Optimization

```sql
-- Create optimal indexes for fast retrieval

-- Specialty-based retrieval
CREATE INDEX IF NOT EXISTS idx_specialty_btree
ON medical_cases(specialty);

-- Diagnosis lookup
CREATE INDEX IF NOT EXISTS idx_diagnosis_btree
ON medical_cases(diagnosis);

-- Vector similarity search (HNSW for speed)
CREATE INDEX IF NOT EXISTS idx_case_embedding_hnsw
ON medical_cases
USING HNSW (embedding)
WITH (
    metric = 'cosine',
    ef_construction = 200,  -- Higher = better quality, slower build
    M = 16                   -- Higher = better recall, more memory
);

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS idx_specialty_diagnosis
ON medical_cases(specialty, diagnosis);
```

### 3.3 Batch Retrieval

```python
class BatchRetriever:
    """Batch database retrieval"""

    async def batch_get_cases(self, case_ids):
        """Get multiple cases in one query"""

        if not case_ids:
            return []

        query = """
            SELECT * FROM medical_cases
            WHERE case_id IN (SELECT unnest($1::INTEGER[]))
        """

        async with self.db_pool.acquire() as conn:
            results = conn.execute(query, [case_ids]).fetchall()

        return results

    async def batch_similarity_search(self, embeddings, top_k=5):
        """Batch vector similarity search"""

        tasks = [
            self._similarity_search_single(emb, top_k)
            for emb in embeddings
        ]

        results = await asyncio.gather(*tasks)
        return results
```

---

## 4. Agent Communication Optimization

### 4.1 Message Queue with Priority

```python
from queue import PriorityQueue
import asyncio

class PriorityMessageQueue:
    """Priority-based message queue"""

    def __init__(self):
        self.queue = asyncio.PriorityQueue()

    async def send(self, from_agent, to_agent, content, priority="normal"):
        """Send message with priority"""

        priority_map = {
            "critical": 0,
            "high": 1,
            "normal": 2,
            "low": 3,
        }

        message = {
            "from": from_agent,
            "to": to_agent,
            "content": content,
            "timestamp": time.time(),
        }

        await self.queue.put((priority_map[priority], message))

    async def receive(self):
        """Receive highest priority message"""

        priority, message = await self.queue.get()
        return message

    async def process_messages(self, handler):
        """Process messages with priority"""

        while True:
            message = await self.receive()

            # Process in background
            asyncio.create_task(handler(message))
```

### 4.2 Asynchronous Agent Communication

```python
class AsyncAgentCommunicator:
    """Non-blocking agent communication"""

    def __init__(self):
        self.channels = defaultdict(asyncio.Queue)
        self.subscribers = defaultdict(list)

    async def publish(self, channel, message):
        """Publish message to channel"""

        if channel not in self.channels:
            return

        # Non-blocking send
        await self.channels[channel].put(message)

    async def subscribe(self, channel, callback):
        """Subscribe to channel"""

        async def listener():
            while True:
                message = await self.channels[channel].get()
                await callback(message)

        task = asyncio.create_task(listener())
        self.subscribers[channel].append(task)

    async def request_response(self, to_agent, request, timeout=5.0):
        """Request-response with timeout"""

        response_channel = f"response_{id(request)}"

        # Subscribe to response channel
        response_future = asyncio.Future()

        async def response_handler(message):
            if not response_future.done():
                response_future.set_result(message)

        await self.subscribe(response_channel, response_handler)

        # Send request
        await self.publish(to_agent, {
            "request": request,
            "response_channel": response_channel,
        })

        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            return None
```

---

## 5. Caching Strategy

### 5.1 Multi-Level Cache

```python
class MultiLevelCache:
    """L1 (memory) -> L2 (Redis) -> L3 (DuckDB)"""

    def __init__(self, redis_client, db_pool):
        self.l1_cache = {}  # In-memory
        self.l1_max_size = 1000
        self.l2_client = redis_client  # Redis
        self.l3_db = db_pool  # DuckDB

    async def get(self, key):
        """Get value from cache hierarchy"""

        # L1: Memory (< 1ms)
        if key in self.l1_cache:
            return self.l1_cache[key]

        # L2: Redis (~2ms)
        value = await self._get_from_redis(key)
        if value is not None:
            self._set_l1(key, value)
            return value

        # L3: Database (~10ms)
        value = await self._get_from_db(key)
        if value is not None:
            await self._set_l2(key, value)
            self._set_l1(key, value)

        return value

    async def set(self, key, value, ttl=3600):
        """Set value in all cache levels"""

        # Set in all levels
        self._set_l1(key, value)
        await self._set_l2(key, value, ttl)
        await self._set_l3(key, value)

    def _set_l1(self, key, value):
        """Set in L1 cache"""

        # Evict if cache full (LRU)
        if len(self.l1_cache) >= self.l1_max_size:
            # Remove random key (simple eviction)
            self.l1_cache.pop(next(iter(self.l1_cache)))

        self.l1_cache[key] = value
```

### 5.2 Predictive Caching

```python
class PredictiveCache:
    """Predict and pre-fetch likely needed data"""

    def __init__(self, cache):
        self.cache = cache
        self.access_patterns = defaultdict(list)

    def record_access(self, key):
        """Record access pattern"""

        timestamp = time.time()
        self.access_patterns[key].append(timestamp)

        # Keep last 100 accesses
        self.access_patterns[key] = self.access_patterns[key][-100:]

    async def prefetch_likely_items(self):
        """Pre-fetch likely to be accessed items"""

        # Analyze patterns
        likely_keys = self._predict_next_accesses()

        # Pre-fetch in background
        for key in likely_keys:
            if key not in self.cache.l1_cache:
                asyncio.create_task(self.cache.get(key))

    def _predict_next_accesses(self):
        """Predict next accesses based on patterns"""

        predictions = []

        for key, timestamps in self.access_patterns.items():
            if len(timestamps) < 2:
                continue

            # Calculate access frequency
            time_deltas = [
                timestamps[i] - timestamps[i-1]
                for i in range(1, len(timestamps))
            ]

            avg_delta = sum(time_deltas) / len(time_deltas)

            # Predict if likely to be accessed soon
            time_since_last = time.time() - timestamps[-1]

            if time_since_last >= avg_delta * 0.8:
                predictions.append(key)

        return predictions[:10]  # Top 10 predictions
```

---

## 6. Network Optimization

### 6.1 HTTP/2 with Connection Reuse

```python
import httpx

class OptimizedHTTPClient:
    """HTTP client optimized for latency"""

    def __init__(self):
        # HTTP/2 with connection pooling
        self.client = httpx.AsyncClient(
            http2=True,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
            timeout=httpx.Timeout(5.0, connect=2.0),
        )

    async def post_with_retry(self, url, data, max_retries=3):
        """POST with exponential backoff"""

        for attempt in range(max_retries):
            try:
                response = await self.client.post(url, json=data)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                if attempt == max_retries - 1:
                    raise

                # Exponential backoff
                wait_time = 0.1 * (2 ** attempt)
                await asyncio.sleep(wait_time)
```

### 6.2 Request Compression

```python
import gzip
import json

class RequestCompressor:
    """Compress request payloads"""

    @staticmethod
    def compress_json(data):
        """Compress JSON data"""

        json_str = json.dumps(data)
        compressed = gzip.compress(json_str.encode('utf-8'))

        # Only use compression if it saves space
        if len(compressed) < len(json_str):
            return compressed, True
        else:
            return json_str.encode('utf-8'), False

    @staticmethod
    async def send_compressed(client, url, data):
        """Send compressed request"""

        payload, is_compressed = RequestCompressor.compress_json(data)

        headers = {}
        if is_compressed:
            headers["Content-Encoding"] = "gzip"
            headers["Content-Type"] = "application/json"

        response = await client.post(url, content=payload, headers=headers)
        return response
```

---

## 7. OCR Processing Optimization

### 7.1 Parallel Page Processing

```python
class ParallelOCRProcessor:
    """Process PDF pages in parallel"""

    def __init__(self, num_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    async def process_pdf(self, pdf_path):
        """Process PDF with parallel page OCR"""

        import fitz

        # Load PDF
        doc = fitz.open(pdf_path)

        # Process pages in parallel
        loop = asyncio.get_event_loop()

        tasks = [
            loop.run_in_executor(self.executor, self._process_page, doc, page_num)
            for page_num in range(len(doc))
        ]

        results = await asyncio.gather(*tasks)

        return "\n".join(results)

    @staticmethod
    def _process_page(doc, page_num):
        """Process single page (CPU-bound)"""

        import pytesseract
        from PIL import Image

        page = doc[page_num]

        # Try PyMuPDF first (faster)
        text = page.get_text()

        # If quality is low, use Tesseract
        if len(text.strip()) < 50:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)

        return text
```

### 7.2 Adaptive Quality OCR

```python
class AdaptiveOCR:
    """Adapt OCR quality based on image characteristics"""

    @staticmethod
    def analyze_image_quality(image):
        """Analyze image to determine OCR approach"""

        import cv2
        import numpy as np

        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Calculate metrics
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = gray.std()

        if sharpness > 100 and contrast > 50:
            return "fast"  # Good quality, use fast OCR
        else:
            return "accurate"  # Poor quality, use careful OCR

    async def process_adaptive(self, image):
        """Process with adaptive quality"""

        quality = self.analyze_image_quality(image)

        if quality == "fast":
            # Fast OCR
            text = pytesseract.image_to_string(image, config='--psm 1')
        else:
            # Accurate OCR with preprocessing
            preprocessed = self._preprocess_image(image)
            text = pytesseract.image_to_string(
                preprocessed,
                config='--psm 1 --oem 3'
            )

        return text
```

---

## 8. Latency Monitoring

### 8.1 Real-Time Latency Tracking

```python
from prometheus_client import Histogram

class LatencyMonitor:
    """Monitor latency in real-time"""

    def __init__(self):
        self.latency_histogram = Histogram(
            'component_latency_seconds',
            'Component latency',
            ['component'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )

    def track(self, component_name):
        """Track component latency"""

        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                with self.latency_histogram.labels(component=component_name).time():
                    return await func(*args, **kwargs)
            return wrapper
        return decorator
```

---

## 9. Latency Budget

```python
LATENCY_BUDGET = {
    "triager": 80,           # 80ms
    "specialist": 40,        # 40ms per specialist
    "specialists_parallel": 120,  # 120ms for 3 specialists in parallel
    "validator": 100,        # 100ms
    "communicator": 80,      # 80ms
    "database_rag": 10,      # 10ms per query
    "ocr": 200,              # 200ms for document processing
    "network_overhead": 20,  # 20ms
    "total_target": 500,     # 500ms total
}
```

---

## 10. Quick Wins Summary

1. **Enable parallel specialist execution** → 60-70% reduction
2. **Implement multi-level caching** → 80% hit rate, 90% faster
3. **Use streaming responses** → Progressive UI, perceived 2x faster
4. **Optimize prompts** → 30-40% token reduction
5. **Batch database queries** → 50% reduction in query time
6. **Pre-warm caches** → Eliminate cold start latency
7. **Use HTTP/2** → 20-30% network speedup
8. **Parallel OCR** → 4x faster document processing

---

## Contact

For questions about latency optimization, contact the performance team.

**Last Updated:** 2025-11-11
**Version:** 1.0.0
