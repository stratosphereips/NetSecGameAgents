from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Optional
from uuid import uuid4

class _NoopSpan:
    def update(self, **kwargs: Any) -> None:
        pass

class ITracer:
    """Interface for tracing LLM interactions."""
    @contextmanager
    def start_trace(self, name: str, **kwargs: Any):  # type: ignore
        # Fallback: treat a trace as a top-level span
        with self.start_span(name=name, **kwargs) as span:  # type: ignore
            yield span
    @contextmanager
    def start_span(self, name: str, **kwargs: Any):  # type: ignore
        yield _NoopSpan()

    @contextmanager
    def start_generation(self, name: str, **kwargs: Any):  # type: ignore
        yield _NoopSpan()

    def flush(self) -> None:
        pass

class NoopTracer(ITracer):
    """Tracer that performs no operations."""
    pass

class LangfuseTracer(ITracer):
    """Tracer backed by Langfuse if available."""
    def __init__(self) -> None:
        from langfuse import Langfuse
        self.client = Langfuse()

    @contextmanager
    def start_trace(self, name: str, **kwargs: Any):  # type: ignore
        # Prefer a real trace to segment episodes. Fallback to a span if the
        # method is unavailable (older SDKs)
        safe_kwargs = dict(kwargs)
        safe_kwargs.pop("parent_span", None)
        # Reset any lingering context if the SDK exposes it
        try:
            import langfuse as lf  # type: ignore
            if hasattr(lf, "reset_context"):
                lf.reset_context()  # type: ignore[attr-defined]
        except Exception:
            try:
                if hasattr(self.client, "reset_context"):
                    self.client.reset_context()  # type: ignore[attr-defined]
            except Exception:
                pass

        trace_id = f"episode-{uuid4()}"
        # Try modern API first
        if hasattr(self.client, "trace"):
            for call in (
                lambda: self.client.trace(name=name, id=trace_id, **safe_kwargs),
                lambda: self.client.trace(operation_name=name, id=trace_id, **safe_kwargs),
                lambda: self.client.trace(name=name, **safe_kwargs),
            ):
                try:
                    with call() as trace:  # type: ignore[misc]
                        yield trace
                        return
                except TypeError:
                    continue
        # Older API variants
        if hasattr(self.client, "start_as_current_trace"):
            for call in (
                lambda: self.client.start_as_current_trace(name=name, id=trace_id, **safe_kwargs),
                lambda: self.client.start_as_current_trace(operation_name=name, id=trace_id, **safe_kwargs),
                lambda: self.client.start_as_current_trace(name=name, **safe_kwargs),
            ):
                try:
                    with call() as trace:  # type: ignore[misc]
                        yield trace
                        return
                except TypeError:
                    continue
        # Fallback: create a top-level span (not ideal, but prevents crashes)
        with self.client.start_as_current_span(name=name, **safe_kwargs) as span:
            yield span

    @contextmanager
    def start_span(self, name: str, **kwargs: Any):  # type: ignore
        # Prefer attaching to an explicit parent when provided to avoid leaking
        # context across traces. Fallback to setting current span in context.
        safe_kwargs = dict(kwargs)
        parent = safe_kwargs.pop("parent_span", None)
        if parent is not None:
            # Newer SDKs expose child creation from the parent span/trace
            for method_name in ("span", "start_span", "start_as_current_span"):
                if hasattr(parent, method_name):
                    method = getattr(parent, method_name)
                    try:
                        with method(name=name, **safe_kwargs) as span:  # type: ignore[misc]
                            yield span
                            return
                    except TypeError:
                        # Signature mismatch, try next
                        pass
        # Fallback: create a top-level span in current context
        with self.client.start_as_current_span(name=name, **safe_kwargs) as span:
            yield span

    @contextmanager
    def start_generation(self, name: str, **kwargs: Any):  # type: ignore
        safe_kwargs = dict(kwargs)
        parent = safe_kwargs.pop("parent_span", None)
        if parent is not None:
            for method_name in ("generation", "start_generation", "start_as_current_generation"):
                if hasattr(parent, method_name):
                    method = getattr(parent, method_name)
                    try:
                        with method(name=name, **safe_kwargs) as gen:  # type: ignore[misc]
                            yield gen
                            return
                    except TypeError:
                        pass
        with self.client.start_as_current_generation(name=name, **safe_kwargs) as generation:
            yield generation

    def flush(self) -> None:
        # Ensure buffered events are delivered
        try:
            if hasattr(self.client, "flush"):
                self.client.flush()  # type: ignore[attr-defined]
            elif hasattr(self.client, "shutdown"):
                self.client.shutdown()  # type: ignore[attr-defined]
        except Exception:
            pass

def get_tracer(enabled: bool = False) -> ITracer:
    """Return a tracer instance. Uses Langfuse if enabled and available."""
    if not enabled:
        return NoopTracer()
    try:
        return LangfuseTracer()
    except Exception:
        return NoopTracer()
