# Resource Utilization and Optimization - Medical Assistant System

## Overview

This document provides detailed strategies for optimizing resource utilization across all components of the medical assistant system, ensuring efficient operation while maintaining ultra-low latency.

---

## 1. Groq LPU Resource Optimization

### 1.1 Request Quota Management

**Groq API Limits:**
- Free tier: 30 requests/minute, 7,000 requests/day
- Paid tier: Higher limits based on plan

```python
# src/services/groq_resource_manager.py

import asyncio
from collections import deque
from datetime import datetime, timedelta

class GroqResourceManager:
    """Manage Groq API quota and rate limiting"""

    def __init__(self, rpm_limit=30, rpd_limit=7000):
        self.rpm_limit = rpm_limit  # Requests per minute
        self.rpd_limit = rpd_limit  # Requests per day

        self.minute_window = deque(maxlen=rpm_limit)
        self.day_window = deque(maxlen=rpd_limit)

        self.semaphore = asyncio.Semaphore(rpm_limit // 2)  # Conservative limit

    async def acquire(self):
        """Acquire permission to make request"""

        async with self.semaphore:
            now = datetime.now()

            # Clean up old entries
            self._cleanup_windows(now)

            # Check if within limits
            if len(self.minute_window) >= self.rpm_limit:
                # Wait for next minute window
                sleep_time = 60 - (now - self.minute_window[0]).total_seconds()
                await asyncio.sleep(sleep_time)

            if len(self.day_window) >= self.rpd_limit:
                # Out of daily quota
                raise Exception("Daily Groq quota exceeded")

            # Record request
            self.minute_window.append(now)
            self.day_window.append(now)

    def _cleanup_windows(self, now):
        """Remove expired entries"""

        # Clean minute window (keep last 60 seconds)
        minute_ago = now - timedelta(seconds=60)
        while self.minute_window and self.minute_window[0] < minute_ago:
            self.minute_window.popleft()

        # Clean day window (keep last 24 hours)
        day_ago = now - timedelta(hours=24)
        while self.day_window and self.day_window[0] < day_ago:
            self.day_window.popleft()

    def get_usage(self):
        """Get current usage statistics"""
        return {
            "requests_per_minute": len(self.minute_window),
            "rpm_limit": self.rpm_limit,
            "requests_per_day": len(self.day_window),
            "rpd_limit": self.rpd_limit,
            "remaining_minute": self.rpm_limit - len(self.minute_window),
            "remaining_day": self.rpd_limit - len(self.day_window),
        }
```

### 1.2 Intelligent Request Batching

```python
class GroqBatchOptimizer:
    """Optimize Groq requests through intelligent batching"""

    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self.pending_requests = []
        self.batch_interval = 0.1  # 100ms batching window

    async def submit_request(self, agent_type, prompt, priority="normal"):
        """Submit request for batched execution"""

        future = asyncio.Future()

        request = {
            "agent_type": agent_type,
            "prompt": prompt,
            "priority": priority,
            "future": future,
            "submitted_at": datetime.now(),
        }

        self.pending_requests.append(request)

        # Trigger batch processing
        asyncio.create_task(self._process_batch())

        return await future

    async def _process_batch(self):
        """Process batched requests"""

        await asyncio.sleep(self.batch_interval)

        if not self.pending_requests:
            return

        # Sort by priority
        batch = sorted(
            self.pending_requests,
            key=lambda x: (
                0 if x["priority"] == "critical" else
                1 if x["priority"] == "high" else 2
            )
        )

        self.pending_requests = []

        # Group by agent type for efficient processing
        by_agent_type = {}
        for req in batch:
            agent_type = req["agent_type"]
            if agent_type not in by_agent_type:
                by_agent_type[agent_type] = []
            by_agent_type[agent_type].append(req)

        # Process each group in parallel
        tasks = []
        for agent_type, requests in by_agent_type.items():
            tasks.append(self._process_agent_batch(agent_type, requests))

        await asyncio.gather(*tasks)

    async def _process_agent_batch(self, agent_type, requests):
        """Process batch for specific agent type"""

        for request in requests:
            await self.resource_manager.acquire()

            # Execute request
            result = await self._execute_groq_request(
                request["agent_type"],
                request["prompt"]
            )

            # Complete future
            request["future"].set_result(result)
```

---

## 2. Memory Management

### 2.1 Shared Memory for Case Data

```python
# src/services/memory_manager.py

import mmap
import struct
import json
from typing import Dict, Any

class SharedCaseMemory:
    """Efficient shared memory for case data"""

    CASE_SIZE = 1024 * 10  # 10KB per case
    MAX_CASES = 1000

    def __init__(self):
        self.memory_size = self.CASE_SIZE * self.MAX_CASES
        self.shared_mem = mmap.mmap(-1, self.memory_size)
        self.case_index = {}  # case_id -> offset
        self.free_slots = list(range(self.MAX_CASES))

    def store_case(self, case_id: str, case_data: Dict[Any, Any]):
        """Store case in shared memory"""

        if not self.free_slots:
            raise MemoryError("No free slots available")

        # Get free slot
        slot = self.free_slots.pop(0)
        offset = slot * self.CASE_SIZE

        # Serialize case data
        serialized = json.dumps(case_data).encode('utf-8')

        if len(serialized) > self.CASE_SIZE - 8:
            raise ValueError(f"Case data too large: {len(serialized)} bytes")

        # Write to shared memory
        self.shared_mem.seek(offset)
        self.shared_mem.write(struct.pack('I', len(serialized)))  # Length prefix
        self.shared_mem.write(serialized)

        # Update index
        self.case_index[case_id] = slot

    def get_case(self, case_id: str) -> Dict[Any, Any]:
        """Retrieve case from shared memory"""

        if case_id not in self.case_index:
            return None

        slot = self.case_index[case_id]
        offset = slot * self.CASE_SIZE

        # Read from shared memory
        self.shared_mem.seek(offset)
        length = struct.unpack('I', self.shared_mem.read(4))[0]
        serialized = self.shared_mem.read(length)

        return json.loads(serialized.decode('utf-8'))

    def release_case(self, case_id: str):
        """Free case memory slot"""

        if case_id in self.case_index:
            slot = self.case_index[case_id]
            del self.case_index[case_id]
            self.free_slots.append(slot)

    def get_memory_stats(self):
        """Get memory usage statistics"""
        return {
            "total_slots": self.MAX_CASES,
            "used_slots": len(self.case_index),
            "free_slots": len(self.free_slots),
            "memory_used_mb": (len(self.case_index) * self.CASE_SIZE) / (1024 * 1024),
            "memory_total_mb": self.memory_size / (1024 * 1024),
        }
```

### 2.2 Memory Pool for Agent Contexts

```python
class AgentContextPool:
    """Memory pool for agent contexts"""

    def __init__(self, pool_size=100):
        self.pool_size = pool_size
        self.contexts = []
        self.active_contexts = {}

        # Pre-allocate contexts
        for _ in range(pool_size):
            self.contexts.append(self._create_context())

    def _create_context(self):
        """Create new agent context"""
        return {
            "messages": [],
            "state": {},
            "metadata": {},
        }

    def acquire(self, agent_id: str):
        """Acquire context from pool"""

        if agent_id in self.active_contexts:
            return self.active_contexts[agent_id]

        if not self.contexts:
            # Pool exhausted, create new context
            context = self._create_context()
        else:
            context = self.contexts.pop()

        self.active_contexts[agent_id] = context
        return context

    def release(self, agent_id: str):
        """Release context back to pool"""

        if agent_id not in self.active_contexts:
            return

        context = self.active_contexts[agent_id]
        del self.active_contexts[agent_id]

        # Clear context and return to pool
        context["messages"].clear()
        context["state"].clear()
        context["metadata"].clear()

        if len(self.contexts) < self.pool_size:
            self.contexts.append(context)
```

---

## 3. Database Resource Optimization

### 3.1 Connection Pooling

```python
# src/services/database_pool.py

import duckdb
from contextlib import asynccontextmanager

class DuckDBConnectionPool:
    """Connection pool for DuckDB"""

    def __init__(self, db_path, pool_size=10):
        self.db_path = db_path
        self.pool_size = pool_size
        self.connections = []
        self.available = asyncio.Queue()

        # Initialize pool
        for _ in range(pool_size):
            conn = duckdb.connect(db_path, read_only=True)
            self.connections.append(conn)
            self.available.put_nowait(conn)

    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool"""

        conn = await self.available.get()

        try:
            yield conn
        finally:
            await self.available.put(conn)

    async def execute(self, query, params=None):
        """Execute query using pooled connection"""

        async with self.acquire() as conn:
            return conn.execute(query, params).fetchall()

    def close_all(self):
        """Close all connections"""
        for conn in self.connections:
            conn.close()
```

### 3.2 Query Result Caching

```python
from functools import lru_cache
import hashlib

class QueryCache:
    """Cache query results"""

    def __init__(self, max_size=1000, ttl=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.timestamps = {}

    def _hash_query(self, query, params):
        """Generate cache key"""
        key = f"{query}:{str(params)}"
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, query, params):
        """Get cached result"""

        key = self._hash_query(query, params)

        if key not in self.cache:
            return None

        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            return None

        return self.cache[key]

    def set(self, query, params, result):
        """Cache result"""

        key = self._hash_query(query, params)

        # Enforce max size
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]

        self.cache[key] = result
        self.timestamps[key] = time.time()
```

---

## 4. Network and I/O Optimization

### 4.1 Async File Operations

```python
import aiofiles
from typing import AsyncIterator

class AsyncFileProcessor:
    """Asynchronous file processing"""

    @staticmethod
    async def read_chunks(file_path: str, chunk_size=8192) -> AsyncIterator[bytes]:
        """Read file in chunks asynchronously"""

        async with aiofiles.open(file_path, 'rb') as f:
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    @staticmethod
    async def write_async(file_path: str, content: str):
        """Write file asynchronously"""

        async with aiofiles.open(file_path, 'w') as f:
            await f.write(content)

    @staticmethod
    async def process_pdf_async(pdf_path: str):
        """Process PDF asynchronously"""

        import fitz  # PyMuPDF

        # Read file asynchronously
        async with aiofiles.open(pdf_path, 'rb') as f:
            pdf_data = await f.read()

        # Process in executor to avoid blocking
        loop = asyncio.get_event_loop()

        def extract_text():
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text

        text = await loop.run_in_executor(None, extract_text)
        return text
```

### 4.2 Network Connection Pooling

```python
import aiohttp
from aiohttp import TCPConnector

class HTTPClientPool:
    """HTTP client with connection pooling"""

    def __init__(self, max_connections=100, max_per_host=10):
        self.connector = TCPConnector(
            limit=max_connections,
            limit_per_host=max_per_host,
            ttl_dns_cache=300,
        )

        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(connector=self.connector)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    async def post(self, url, **kwargs):
        """POST request with connection reuse"""
        async with self.session.post(url, **kwargs) as response:
            return await response.json()
```

---

## 5. CPU and Threading Optimization

### 5.1 Thread Pool Configuration

```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class OptimizedThreadPool:
    """Optimized thread pool for CPU-bound tasks"""

    def __init__(self):
        # CPU-bound tasks
        cpu_count = multiprocessing.cpu_count()
        self.cpu_executor = ThreadPoolExecutor(
            max_workers=cpu_count,
            thread_name_prefix="cpu-"
        )

        # I/O-bound tasks
        self.io_executor = ThreadPoolExecutor(
            max_workers=cpu_count * 4,
            thread_name_prefix="io-"
        )

    async def run_cpu_bound(self, func, *args):
        """Run CPU-bound task"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.cpu_executor, func, *args)

    async def run_io_bound(self, func, *args):
        """Run I/O-bound task"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.io_executor, func, *args)

    def shutdown(self):
        """Shutdown executors"""
        self.cpu_executor.shutdown(wait=True)
        self.io_executor.shutdown(wait=True)
```

### 5.2 Process Pool for Heavy Computation

```python
from multiprocessing import Pool

class HeavyComputationPool:
    """Process pool for heavy computations"""

    def __init__(self, num_processes=None):
        self.num_processes = num_processes or multiprocessing.cpu_count()
        self.pool = Pool(processes=self.num_processes)

    async def process_ocr_batch(self, image_paths):
        """Process OCR for multiple images in parallel"""

        loop = asyncio.get_event_loop()

        # Process in parallel across multiple processes
        results = await loop.run_in_executor(
            None,
            self.pool.map,
            self._ocr_single_image,
            image_paths
        )

        return results

    @staticmethod
    def _ocr_single_image(image_path):
        """OCR single image (CPU-intensive)"""
        import pytesseract
        from PIL import Image

        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text

    def close(self):
        """Close pool"""
        self.pool.close()
        self.pool.join()
```

---

## 6. Resource Monitoring

### 6.1 Real-Time Resource Tracking

```python
import psutil
import time

class ResourceMonitor:
    """Monitor system resource usage"""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()

    def get_current_usage(self):
        """Get current resource usage"""

        cpu_percent = self.process.cpu_percent(interval=0.1)
        memory_info = self.process.memory_info()

        return {
            "cpu_percent": cpu_percent,
            "memory_mb": memory_info.rss / (1024 * 1024),
            "memory_percent": self.process.memory_percent(),
            "num_threads": self.process.num_threads(),
            "num_fds": self.process.num_fds() if hasattr(self.process, 'num_fds') else 0,
            "uptime_seconds": time.time() - self.start_time,
        }

    def get_system_usage(self):
        """Get system-wide resource usage"""

        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=0.1, percpu=True),
            "memory": psutil.virtual_memory()._asdict(),
            "disk": psutil.disk_usage('/')._asdict(),
        }

    async def monitor_continuous(self, interval=5.0, callback=None):
        """Continuously monitor resources"""

        while True:
            usage = self.get_current_usage()

            if callback:
                await callback(usage)

            # Alert on high usage
            if usage["cpu_percent"] > 90:
                print(f"⚠️  High CPU usage: {usage['cpu_percent']:.1f}%")

            if usage["memory_percent"] > 80:
                print(f"⚠️  High memory usage: {usage['memory_percent']:.1f}%")

            await asyncio.sleep(interval)
```

---

## 7. Resource Optimization Recommendations

### 7.1 Configuration Guidelines

```python
# config/resource_limits.py

RESOURCE_LIMITS = {
    # Groq API
    "groq_rpm": 30,  # Requests per minute
    "groq_rpd": 7000,  # Requests per day
    "groq_concurrent": 10,  # Max concurrent requests

    # Memory
    "max_cases_in_memory": 1000,
    "case_memory_mb": 10,
    "agent_context_pool_size": 100,

    # Database
    "db_connection_pool_size": 10,
    "query_cache_size": 1000,
    "query_cache_ttl": 3600,

    # Threading
    "cpu_threads": multiprocessing.cpu_count(),
    "io_threads": multiprocessing.cpu_count() * 4,
    "process_pool_size": multiprocessing.cpu_count(),

    # Network
    "http_max_connections": 100,
    "http_max_per_host": 10,

    # File I/O
    "file_chunk_size": 8192,
    "max_file_size_mb": 10,
}
```

### 7.2 Auto-Tuning

```python
class ResourceAutoTuner:
    """Automatically tune resource limits based on load"""

    def __init__(self, monitor):
        self.monitor = monitor
        self.config = RESOURCE_LIMITS.copy()

    async def auto_tune(self):
        """Adjust limits based on current usage"""

        usage = self.monitor.get_current_usage()
        system = self.monitor.get_system_usage()

        # Tune thread pool sizes
        if usage["cpu_percent"] < 50 and system["memory"]["percent"] < 60:
            # System has capacity, increase parallelism
            self.config["io_threads"] = min(
                self.config["io_threads"] + 4,
                multiprocessing.cpu_count() * 8
            )
        elif usage["cpu_percent"] > 80:
            # System overloaded, reduce parallelism
            self.config["io_threads"] = max(
                self.config["io_threads"] - 4,
                multiprocessing.cpu_count()
            )

        # Tune cache sizes based on memory
        available_memory_mb = system["memory"]["available"] / (1024 * 1024)

        if available_memory_mb > 1000:
            self.config["max_cases_in_memory"] = 2000
            self.config["query_cache_size"] = 2000
        elif available_memory_mb < 500:
            self.config["max_cases_in_memory"] = 500
            self.config["query_cache_size"] = 500

        return self.config
```

---

## 8. Best Practices Summary

### DO:
- Use connection/thread pools
- Implement multi-tier caching
- Monitor resource usage continuously
- Batch similar operations
- Use async I/O for files and network
- Pre-allocate memory pools
- Auto-tune based on load

### DON'T:
- Create new connections for each request
- Load entire files into memory
- Use blocking I/O operations
- Ignore resource limits
- Over-provision resources
- Under-utilize available CPUs

---

## Contact

For questions about resource optimization, contact the infrastructure team.

**Last Updated:** 2025-11-11
**Version:** 1.0.0
