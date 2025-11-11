# Benchmarking Strategy - Medical Assistant System

## Overview

This document outlines a comprehensive benchmarking strategy for the medical assistant system to ensure consistent ultra-low latency performance in production.

---

## 1. Benchmark Categories

### 1.1 Microbenchmarks (Component-Level)

**Purpose:** Measure individual component performance in isolation

```python
# examples/benchmarks/micro/test_groq_inference.py

import asyncio
import time
from groq import AsyncGroq
from statistics import mean, stdev

async def benchmark_groq_model(model_name, num_iterations=100):
    """Benchmark single Groq model inference"""

    client = AsyncGroq()
    latencies = []

    test_prompt = """
    ROLE: Cardiologist
    TASK: Analyze cardiac markers
    DATA: {troponin: 0.5, BNP: 450, ECG: normal_sinus}
    OUTPUT: JSON diagnosis
    """

    # Warm-up
    for _ in range(5):
        await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": test_prompt}],
            max_tokens=256,
        )

    # Benchmark
    for i in range(num_iterations):
        start = time.perf_counter()

        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": test_prompt}],
            max_tokens=256,
        )

        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        if i % 10 == 0:
            print(f"Iteration {i}: {latency:.2f}ms")

    return {
        "model": model_name,
        "iterations": num_iterations,
        "mean": mean(latencies),
        "stdev": stdev(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "p50": sorted(latencies)[int(num_iterations * 0.50)],
        "p95": sorted(latencies)[int(num_iterations * 0.95)],
        "p99": sorted(latencies)[int(num_iterations * 0.99)],
    }

async def main():
    models = [
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
    ]

    results = []
    for model in models:
        print(f"\nBenchmarking {model}...")
        result = await benchmark_groq_model(model)
        results.append(result)

    # Print comparison
    print("\n" + "=" * 80)
    print("GROQ MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Model':<25} {'Mean':<10} {'P95':<10} {'P99':<10}")
    print("-" * 80)

    for result in results:
        print(f"{result['model']:<25} {result['mean']:>7.2f}ms {result['p95']:>7.2f}ms {result['p99']:>7.2f}ms")

if __name__ == "__main__":
    asyncio.run(main())
```

**Target Metrics:**
- llama3-8b-8192: Mean <50ms, P95 <80ms
- llama3-70b-8192: Mean <100ms, P95 <150ms
- mixtral-8x7b-32768: Mean <80ms, P95 <120ms

### 1.2 Integration Benchmarks (Agent-Level)

**Purpose:** Measure agent performance with all dependencies

```python
# examples/benchmarks/integration/test_agent_performance.py

import asyncio
import agno
from typing import List, Dict

class AgentBenchmarkSuite:
    """Benchmark suite for individual agents"""

    def __init__(self):
        self.results = {}

    async def benchmark_triager(self, test_cases: List[Dict]):
        """Benchmark triager agent"""

        triager = agno.Agent(
            name="Triager",
            model="groq:llama3-70b-8192",
            instructions="Analyze case and route to specialists"
        )

        latencies = []

        for case in test_cases:
            start = time.perf_counter()
            result = await triager.run(case)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

        self.results["triager"] = {
            "mean": mean(latencies),
            "p95": sorted(latencies)[int(len(latencies) * 0.95)],
        }

    async def benchmark_specialist_parallel(self, test_case: Dict, num_specialists=3):
        """Benchmark parallel specialist execution"""

        specialists = [
            agno.Agent(name=f"Specialist-{i}", model="groq:llama3-8b-8192")
            for i in range(num_specialists)
        ]

        start = time.perf_counter()

        results = await asyncio.gather(*[
            specialist.run(test_case)
            for specialist in specialists
        ])

        total_time = (time.perf_counter() - start) * 1000

        self.results["specialists_parallel"] = {
            "num_specialists": num_specialists,
            "total_time": total_time,
            "time_per_agent": total_time / num_specialists,
        }

    def print_results(self):
        """Print benchmark results"""
        print("\n" + "=" * 80)
        print("AGENT PERFORMANCE BENCHMARK")
        print("=" * 80)

        for agent_type, metrics in self.results.items():
            print(f"\n{agent_type.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.2f}" + ("ms" if "time" in metric else ""))
```

### 1.3 End-to-End Benchmarks (System-Level)

**Purpose:** Measure complete workflow performance

```python
# examples/benchmarks/e2e/test_full_pipeline.py

class E2EBenchmark:
    """End-to-end system benchmarking"""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.case_categories = self._load_test_cases()

    def _load_test_cases(self):
        """Load categorized test cases"""
        return {
            "simple": [
                # Cases requiring 1-2 specialists
                {"chief_complaint": "Headache", "duration": "2 days"},
                {"chief_complaint": "Chest pain", "troponin": 0.03},
            ],
            "moderate": [
                # Cases requiring 2-3 specialists
                {"chief_complaint": "Shortness of breath", "history": "diabetes"},
            ],
            "complex": [
                # Cases requiring 3+ specialists, debate resolution
                {"chief_complaint": "Weakness", "labs": {"sodium": 125, "glucose": 450}},
            ]
        }

    async def run_full_benchmark(self):
        """Run complete benchmark suite"""

        results = {}

        for category, cases in self.case_categories.items():
            print(f"\nBenchmarking {category.upper()} cases...")

            latencies = []

            for case in cases:
                start = time.perf_counter()

                result = await self.orchestrator.process_case(case)

                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)

                print(f"  Case: {latency:.2f}ms")

            results[category] = {
                "mean": mean(latencies),
                "p95": sorted(latencies)[int(len(latencies) * 0.95)],
                "cases_tested": len(cases),
            }

        self._evaluate_results(results)
        return results

    def _evaluate_results(self, results):
        """Evaluate against performance targets"""

        targets = {
            "simple": 500,    # <500ms
            "moderate": 1000, # <1s
            "complex": 2000,  # <2s
        }

        print("\n" + "=" * 80)
        print("PERFORMANCE EVALUATION")
        print("=" * 80)

        for category, metrics in results.items():
            target = targets[category]
            actual = metrics["mean"]

            status = "✅ PASS" if actual < target else "❌ FAIL"
            print(f"\n{category.upper()}: {status}")
            print(f"  Target: <{target}ms")
            print(f"  Actual: {actual:.2f}ms")
            print(f"  P95: {metrics['p95']:.2f}ms")
```

---

## 2. Load Testing Strategy

### 2.1 Concurrency Testing

```python
# examples/benchmarks/load/test_concurrent_load.py

import asyncio
import aiohttp
from datetime import datetime

class LoadTester:
    """Load testing for concurrent requests"""

    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    async def generate_load(self, num_requests, concurrency_level):
        """Generate concurrent load"""

        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(concurrency_level)

            async def make_request(request_id):
                async with semaphore:
                    test_case = {
                        "case_id": f"load-test-{request_id}",
                        "chief_complaint": "Chest pain",
                        "vital_signs": {"bp": "140/90", "hr": 88},
                    }

                    start = time.perf_counter()

                    async with session.post(
                        f"{self.base_url}/api/cases",
                        json=test_case
                    ) as response:
                        result = await response.json()
                        latency = (time.perf_counter() - start) * 1000

                        return {
                            "request_id": request_id,
                            "latency": latency,
                            "status": response.status,
                        }

            # Generate concurrent requests
            tasks = [make_request(i) for i in range(num_requests)]
            results = await asyncio.gather(*tasks)

            return self._analyze_results(results, concurrency_level)

    def _analyze_results(self, results, concurrency):
        """Analyze load test results"""

        latencies = [r["latency"] for r in results]
        successful = [r for r in results if r["status"] == 200]

        return {
            "total_requests": len(results),
            "successful_requests": len(successful),
            "success_rate": len(successful) / len(results) * 100,
            "concurrency_level": concurrency,
            "mean_latency": mean(latencies),
            "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99_latency": sorted(latencies)[int(len(latencies) * 0.99)],
            "throughput": len(results) / (max(latencies) / 1000),  # requests/second
        }

async def main():
    """Run load testing scenarios"""

    tester = LoadTester()

    scenarios = [
        {"requests": 100, "concurrency": 10},
        {"requests": 200, "concurrency": 20},
        {"requests": 500, "concurrency": 50},
    ]

    print("=" * 80)
    print("LOAD TESTING")
    print("=" * 80)

    for scenario in scenarios:
        print(f"\nScenario: {scenario['requests']} requests, {scenario['concurrency']} concurrent")

        result = await tester.generate_load(
            scenario['requests'],
            scenario['concurrency']
        )

        print(f"  Success Rate: {result['success_rate']:.1f}%")
        print(f"  Mean Latency: {result['mean_latency']:.2f}ms")
        print(f"  P95 Latency: {result['p95_latency']:.2f}ms")
        print(f"  Throughput: {result['throughput']:.1f} req/s")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2.2 Stress Testing

```python
# examples/benchmarks/load/test_stress.py

class StressTester:
    """Stress testing to find breaking points"""

    async def ramp_up_test(self, max_concurrency=200, step=10, duration_per_step=30):
        """Gradually increase load to find limits"""

        results = []

        for concurrency in range(step, max_concurrency + 1, step):
            print(f"\nTesting concurrency level: {concurrency}")

            # Test for duration_per_step seconds
            start_time = time.time()
            request_count = 0
            errors = 0
            latencies = []

            while time.time() - start_time < duration_per_step:
                batch_start = time.time()

                # Send batch of concurrent requests
                tasks = [self._send_request() for _ in range(concurrency)]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in batch_results:
                    request_count += 1
                    if isinstance(result, Exception):
                        errors += 1
                    else:
                        latencies.append(result)

                # Small delay between batches
                batch_duration = time.time() - batch_start
                if batch_duration < 1.0:
                    await asyncio.sleep(1.0 - batch_duration)

            error_rate = (errors / request_count) * 100 if request_count > 0 else 0

            result = {
                "concurrency": concurrency,
                "total_requests": request_count,
                "errors": errors,
                "error_rate": error_rate,
                "mean_latency": mean(latencies) if latencies else 0,
                "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            }

            results.append(result)

            print(f"  Requests: {request_count}, Errors: {error_rate:.1f}%, P95: {result['p95_latency']:.2f}ms")

            # Stop if error rate exceeds threshold
            if error_rate > 5.0:
                print(f"\n⚠️  Error rate exceeded 5% at concurrency level {concurrency}")
                break

        return results
```

---

## 3. Regression Testing

### 3.1 Performance Regression Suite

```python
# examples/benchmarks/regression/test_performance_regression.py

import json
from pathlib import Path

class PerformanceRegressionTester:
    """Detect performance regressions"""

    def __init__(self, baseline_file="benchmarks/baseline.json"):
        self.baseline_file = Path(baseline_file)
        self.baseline = self._load_baseline()

    def _load_baseline(self):
        """Load baseline performance metrics"""
        if self.baseline_file.exists():
            with open(self.baseline_file) as f:
                return json.load(f)
        return None

    def save_baseline(self, results):
        """Save new baseline"""
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_file, 'w') as f:
            json.dump(results, f, indent=2)

    async def run_regression_test(self):
        """Run regression test against baseline"""

        # Run current benchmarks
        current = await self._run_current_benchmarks()

        if not self.baseline:
            print("No baseline found. Saving current results as baseline.")
            self.save_baseline(current)
            return

        # Compare against baseline
        regressions = []
        improvements = []

        for metric, current_value in current.items():
            if metric not in self.baseline:
                continue

            baseline_value = self.baseline[metric]
            change_percent = ((current_value - baseline_value) / baseline_value) * 100

            if change_percent > 10:  # >10% slower is regression
                regressions.append({
                    "metric": metric,
                    "baseline": baseline_value,
                    "current": current_value,
                    "change": change_percent,
                })
            elif change_percent < -10:  # >10% faster is improvement
                improvements.append({
                    "metric": metric,
                    "baseline": baseline_value,
                    "current": current_value,
                    "change": change_percent,
                })

        self._print_regression_report(regressions, improvements)

        return len(regressions) == 0  # Return True if no regressions

    def _print_regression_report(self, regressions, improvements):
        """Print regression test report"""

        print("\n" + "=" * 80)
        print("PERFORMANCE REGRESSION TEST")
        print("=" * 80)

        if regressions:
            print("\n❌ REGRESSIONS DETECTED:")
            for r in regressions:
                print(f"  {r['metric']}: {r['baseline']:.2f}ms → {r['current']:.2f}ms ({r['change']:+.1f}%)")
        else:
            print("\n✅ NO REGRESSIONS DETECTED")

        if improvements:
            print("\n✨ IMPROVEMENTS:")
            for i in improvements:
                print(f"  {i['metric']}: {i['baseline']:.2f}ms → {i['current']:.2f}ms ({i['change']:+.1f}%)")
```

---

## 4. Continuous Benchmarking

### 4.1 Automated Benchmark Pipeline

```yaml
# .github/workflows/performance-benchmark.yml

name: Performance Benchmark

on:
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.13'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest-benchmark

      - name: Run microbenchmarks
        run: |
          python examples/benchmarks/micro/test_groq_inference.py

      - name: Run integration benchmarks
        run: |
          python examples/benchmarks/integration/test_agent_performance.py

      - name: Run e2e benchmarks
        run: |
          python examples/benchmarks/e2e/test_full_pipeline.py

      - name: Run regression test
        run: |
          python examples/benchmarks/regression/test_performance_regression.py

      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmarks/results/

      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('benchmarks/results/summary.json'));

            const comment = `
            ## Performance Benchmark Results

            | Metric | Target | Current | Status |
            |--------|--------|---------|--------|
            | Simple Case | <500ms | ${results.simple_case}ms | ${results.simple_case < 500 ? '✅' : '❌'} |
            | Complex Case | <2000ms | ${results.complex_case}ms | ${results.complex_case < 2000 ? '✅' : '❌'} |
            | Throughput | >50 req/s | ${results.throughput} req/s | ${results.throughput > 50 ? '✅' : '❌'} |
            `;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
```

---

## 5. Benchmark Reporting

### 5.1 Performance Dashboard

```python
# examples/benchmarks/reporting/generate_dashboard.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BenchmarkDashboard:
    """Generate performance dashboard"""

    def __init__(self, results):
        self.results = results

    def generate(self, output_file="dashboard.html"):
        """Generate interactive dashboard"""

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Latency Distribution",
                "Component Breakdown",
                "Load Test Results",
                "Regression Trend"
            ]
        )

        # Latency distribution
        fig.add_trace(
            go.Histogram(
                x=self.results["latencies"],
                name="Latency",
                nbinsx=50
            ),
            row=1, col=1
        )

        # Component breakdown
        fig.add_trace(
            go.Bar(
                x=list(self.results["components"].keys()),
                y=list(self.results["components"].values()),
                name="Components"
            ),
            row=1, col=2
        )

        # Load test
        fig.add_trace(
            go.Scatter(
                x=self.results["concurrency_levels"],
                y=self.results["throughput"],
                mode='lines+markers',
                name="Throughput"
            ),
            row=2, col=1
        )

        # Regression trend
        fig.add_trace(
            go.Scatter(
                x=self.results["dates"],
                y=self.results["p95_over_time"],
                mode='lines',
                name="P95 Latency"
            ),
            row=2, col=2
        )

        fig.update_layout(height=800, showlegend=True, title_text="Performance Dashboard")
        fig.write_html(output_file)

        print(f"Dashboard generated: {output_file}")
```

---

## 6. Performance Testing Schedule

### Daily
- Quick smoke tests (<5 min)
- Basic regression tests
- Monitor production metrics

### Weekly
- Full benchmark suite
- Load testing
- Performance regression analysis

### Monthly
- Comprehensive stress testing
- Capacity planning
- Optimization review

### Pre-Release
- Complete benchmark suite
- Extended load testing
- Sign-off on performance metrics

---

## 7. Success Criteria

### Performance Targets

| Test Type | Metric | Target | Critical |
|-----------|--------|--------|----------|
| Simple Case | P95 Latency | <500ms | <1000ms |
| Complex Case | P95 Latency | <2000ms | <3000ms |
| Throughput | Requests/sec | >50 | >20 |
| Error Rate | % | <1% | <5% |
| Success Rate | % | >99% | >95% |

### Regression Thresholds

- Performance degradation >10%: **Block deployment**
- Performance degradation 5-10%: **Investigate before deployment**
- Performance degradation <5%: **Acceptable variance**

---

## 8. Tools and Infrastructure

### Required Tools
- Python 3.13+
- pytest-benchmark
- Locust (load testing)
- Prometheus (monitoring)
- Grafana (visualization)

### CI/CD Integration
- GitHub Actions for automated testing
- Performance benchmarks on every PR
- Nightly comprehensive tests
- Weekly stress tests

---

## Contact

For questions about benchmarking strategy, contact the performance engineering team.

**Last Updated:** 2025-11-11
**Version:** 1.0.0
