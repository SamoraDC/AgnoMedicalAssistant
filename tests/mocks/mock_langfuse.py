"""
Mock Langfuse client for testing without actual tracking

Provides in-memory tracking of traces, spans, and metrics
"""

import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MockSpan:
    """Mock span for tracing"""
    id: str
    trace_id: str
    parent_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    input: Optional[Any] = None
    output: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class MockTrace:
    """Mock trace for workflow tracking"""
    id: str
    name: str
    user_id: Optional[str]
    session_id: Optional[str]
    start_time: float
    end_time: Optional[float] = None
    spans: List[MockSpan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)


class MockLangfuseClient:
    """
    Mock Langfuse client for testing

    Features:
    - In-memory trace storage
    - Span hierarchy tracking
    - Token usage aggregation
    - Performance metrics
    """

    def __init__(self):
        """Initialize mock Langfuse client"""
        self.traces: Dict[str, MockTrace] = {}
        self.spans: Dict[str, MockSpan] = {}
        self.current_trace: Optional[MockTrace] = None
        self.current_span: Optional[MockSpan] = None

    def trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "TraceContext":
        """
        Start a new trace

        Args:
            name: Trace name
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata

        Returns:
            TraceContext for span creation
        """
        trace_id = str(uuid.uuid4())
        trace = MockTrace(
            id=trace_id,
            name=name,
            user_id=user_id,
            session_id=session_id,
            start_time=time.time(),
            metadata=metadata or {}
        )
        self.traces[trace_id] = trace
        self.current_trace = trace

        return TraceContext(self, trace)

    def span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "SpanContext":
        """
        Create a span within a trace

        Args:
            name: Span name
            trace_id: Parent trace ID
            parent_id: Parent span ID
            metadata: Additional metadata

        Returns:
            SpanContext for recording span data
        """
        if trace_id is None and self.current_trace:
            trace_id = self.current_trace.id

        if parent_id is None and self.current_span:
            parent_id = self.current_span.id

        span_id = str(uuid.uuid4())
        span = MockSpan(
            id=span_id,
            trace_id=trace_id,
            parent_id=parent_id,
            name=name,
            start_time=time.time(),
            metadata=metadata or {}
        )
        self.spans[span_id] = span

        if trace_id and trace_id in self.traces:
            self.traces[trace_id].spans.append(span)

        self.current_span = span
        return SpanContext(self, span)

    def record_token_usage(
        self,
        trace_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int
    ):
        """Record token usage for a trace"""
        if trace_id in self.traces:
            trace = self.traces[trace_id]
            trace.token_usage["prompt_tokens"] = trace.token_usage.get("prompt_tokens", 0) + prompt_tokens
            trace.token_usage["completion_tokens"] = trace.token_usage.get("completion_tokens", 0) + completion_tokens
            trace.token_usage["total_tokens"] = trace.token_usage.get("total_tokens", 0) + total_tokens

    def get_trace(self, trace_id: str) -> Optional[MockTrace]:
        """Get trace by ID"""
        return self.traces.get(trace_id)

    def get_span(self, span_id: str) -> Optional[MockSpan]:
        """Get span by ID"""
        return self.spans.get(span_id)

    def get_traces_by_user(self, user_id: str) -> List[MockTrace]:
        """Get all traces for a user"""
        return [t for t in self.traces.values() if t.user_id == user_id]

    def get_traces_by_session(self, session_id: str) -> List[MockTrace]:
        """Get all traces for a session"""
        return [t for t in self.traces.values() if t.session_id == session_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        total_tokens = sum(
            t.token_usage.get("total_tokens", 0) for t in self.traces.values()
        )

        completed_traces = [t for t in self.traces.values() if t.end_time]
        avg_duration = (
            sum(t.end_time - t.start_time for t in completed_traces) / len(completed_traces)
            if completed_traces else 0
        )

        return {
            "total_traces": len(self.traces),
            "total_spans": len(self.spans),
            "total_tokens": total_tokens,
            "completed_traces": len(completed_traces),
            "average_trace_duration": avg_duration
        }

    def reset(self):
        """Reset all tracking data"""
        self.traces.clear()
        self.spans.clear()
        self.current_trace = None
        self.current_span = None


class TraceContext:
    """Context manager for traces"""

    def __init__(self, client: MockLangfuseClient, trace: MockTrace):
        self.client = client
        self.trace = trace

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.trace.end_time = time.time()
        if exc_val:
            self.trace.metadata["error"] = str(exc_val)

    def span(self, name: str, **kwargs) -> "SpanContext":
        """Create a span within this trace"""
        return self.client.span(name, trace_id=self.trace.id, **kwargs)


class SpanContext:
    """Context manager for spans"""

    def __init__(self, client: MockLangfuseClient, span: MockSpan):
        self.client = client
        self.span = span

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.span.end_time = time.time()
        if exc_val:
            self.span.error = str(exc_val)

    def update(
        self,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update span with data"""
        if input is not None:
            self.span.input = input
        if output is not None:
            self.span.output = output
        if metadata:
            self.span.metadata.update(metadata)
