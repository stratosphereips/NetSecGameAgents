from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Optional

class _NoopSpan:
    def update(self, **kwargs: Any) -> None:
        pass

class ITracer:
    """Interface for tracing LLM interactions."""
    @contextmanager
    def start_span(self, name: str, **kwargs: Any):  # type: ignore
        yield _NoopSpan()

    @contextmanager
    def start_generation(self, name: str, **kwargs: Any):  # type: ignore
        yield _NoopSpan()

class NoopTracer(ITracer):
    """Tracer that performs no operations."""
    pass

class LangfuseTracer(ITracer):
    """Tracer backed by Langfuse if available."""
    def __init__(self) -> None:
        from langfuse import Langfuse
        self.client = Langfuse()

    @contextmanager
    def start_span(self, name: str, **kwargs: Any):  # type: ignore
        with self.client.start_as_current_span(name=name, **kwargs) as span:
            yield span

    @contextmanager
    def start_generation(self, name: str, **kwargs: Any):  # type: ignore
        with self.client.start_as_current_generation(name=name, **kwargs) as generation:
            yield generation

def get_tracer(enabled: bool = False) -> ITracer:
    """Return a tracer instance. Uses Langfuse if enabled and available."""
    if not enabled:
        return NoopTracer()
    try:
        return LangfuseTracer()
    except Exception:
        return NoopTracer()
