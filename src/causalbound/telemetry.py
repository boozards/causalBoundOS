"""OpenTelemetry gen_ai.* instrumentation & in-memory waterfall collector.

Captures semantic LLM attributes (gen_ai.system, gen_ai.request.model,
gen_ai.usage.input_tokens, gen_ai.usage.output_tokens, TTFT, tokens/sec,
cost USD) alongside deterministic solver spans (solver.engine, solver.ms,
solver.states_visited) so the War Room UI can render exact LLM-ms vs. Solver-ms
waterfalls in real time.
"""

from __future__ import annotations

import math
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

# Approximate per-1M-token pricing USD (externalized for benchmark cost accounting)
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gemini-3.8-flash": (0.15, 0.60),
    "gemini-3.1-pro": (1.25, 5.00),
}


def utc_now_iso() -> str:
    """Strict UTC ISO-8601 timestamp with rounded 3-digit milliseconds."""
    dt = datetime.now(timezone.utc)
    ms = round(dt.microsecond / 1000.0)
    if ms == 1000:
        dt = dt.replace(microsecond=0)
        ms = 0
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms:03d}Z"


def estimate_llm_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = MODEL_PRICING_PER_1M.get(model, (0.25, 1.00))
    cost = (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000.0
    return round(cost, 8) if math.isfinite(cost) else 0.0


@dataclass(frozen=True, slots=True)
class RecordedSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    category: str  # "llm" | "solver" | "mcp" | "hitl" | "workflow"
    start_ms: float
    duration_ms: float
    status: str
    attributes: dict[str, Any] = field(default_factory=dict)


class InMemoryWaterfallExporter(SpanExporter):
    """Ring-buffer OTel exporter powering the Risk War Room telemetry waterfall."""

    def __init__(self, max_spans: int = 500) -> None:
        self._lock = Lock()
        self._spans: deque[RecordedSpan] = deque(maxlen=max_spans)

    def export(self, spans: Any) -> SpanExportResult:
        with self._lock:
            for sp in spans:
                self._spans.append(self._convert(sp))
        return SpanExportResult.SUCCESS

    def _convert(self, sp: ReadableSpan) -> RecordedSpan:
        ctx = sp.context
        parent = sp.parent
        start_ns = sp.start_time or 0
        end_ns = sp.end_time or start_ns
        attrs = dict(sp.attributes or {})
        category = str(attrs.get("causalbound.span_category", "workflow"))
        return RecordedSpan(
            trace_id=f"{ctx.trace_id:032x}",
            span_id=f"{ctx.span_id:016x}",
            parent_span_id=f"{parent.span_id:016x}" if parent else None,
            name=sp.name,
            category=category,
            start_ms=round(start_ns / 1_000_000.0, 3),
            duration_ms=round(max(0, end_ns - start_ns) / 1_000_000.0, 3),
            status=sp.status.status_code.name,
            attributes=attrs,
        )

    def recent_spans(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._spans)[-limit:]
        return [asdict(item) for item in items]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            items = list(self._spans)
        llm_ms = [s.duration_ms for s in items if s.category == "llm"]
        solver_ms = [s.duration_ms for s in items if s.category == "solver"]
        ttft_vals = [
            float(s.attributes["gen_ai.client.ttft_ms"])
            for s in items
            if "gen_ai.client.ttft_ms" in s.attributes
        ]
        tps_vals = [
            float(s.attributes["gen_ai.client.tokens_per_sec"])
            for s in items
            if "gen_ai.client.tokens_per_sec" in s.attributes
        ]
        total_cost = sum(float(s.attributes.get("gen_ai.usage.cost_usd", 0.0)) for s in items)
        return {
            "total_spans": len(items),
            "llm_avg_ms": round(sum(llm_ms) / len(llm_ms), 2) if llm_ms else 0.0,
            "solver_avg_ms": round(sum(solver_ms) / len(solver_ms), 2) if solver_ms else 0.0,
            "avg_ttft_ms": round(sum(ttft_vals) / len(ttft_vals), 2) if ttft_vals else 0.0,
            "avg_tokens_per_sec": round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else 0.0,
            "cumulative_llm_cost_usd": round(total_cost, 6),
        }

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()


_WATERFALL_EXPORTER = InMemoryWaterfallExporter(max_spans=600)
_PROVIDER_INITIALIZED = False


def get_waterfall_exporter() -> InMemoryWaterfallExporter:
    return _WATERFALL_EXPORTER


def init_telemetry(service_name: str = "causalbound-control-plane") -> trace.Tracer:
    global _PROVIDER_INITIALIZED
    if not _PROVIDER_INITIALIZED:
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(SimpleSpanProcessor(_WATERFALL_EXPORTER))
        trace.set_tracer_provider(provider)
        _PROVIDER_INITIALIZED = True
    return trace.get_tracer(service_name)


@contextmanager
def trace_solver_span(
    engine_name: str,
    operation: str,
    extra_attrs: dict[str, Any] | None = None,
) -> Iterator[trace.Span]:
    tracer = init_telemetry()
    t0 = time.perf_counter()
    with tracer.start_as_current_span(f"solver.{engine_name}.{operation}") as span:
        span.set_attribute("causalbound.span_category", "solver")
        span.set_attribute("solver.engine", engine_name)
        span.set_attribute("solver.operation", operation)
        if extra_attrs:
            for k, v in extra_attrs.items():
                span.set_attribute(k, v)
        try:
            yield span
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            span.set_attribute("solver.duration_ms", round(elapsed_ms, 3))


@contextmanager
def trace_genai_span(
    agent_name: str,
    model_name: str,
    operation: str = "generate_content",
) -> Iterator[dict[str, Any]]:
    """Record gen_ai.* semantic conventions including TTFT, tokens/sec, and cost."""
    tracer = init_telemetry()
    metrics: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "ttft_ms": 0.0,
        "escalated": False,
    }
    t0 = time.perf_counter()
    with tracer.start_as_current_span(f"gen_ai.{agent_name}.{operation}") as span:
        span.set_attribute("causalbound.span_category", "llm")
        span.set_attribute("gen_ai.system", "google_gemini")
        span.set_attribute("gen_ai.operation.name", operation)
        span.set_attribute("gen_ai.request.model", model_name)
        span.set_attribute("gen_ai.agent.name", agent_name)
        try:
            yield metrics
        finally:
            elapsed_sec = max(1e-6, time.perf_counter() - t0)
            in_tok = int(metrics.get("input_tokens", 0))
            out_tok = int(metrics.get("output_tokens", 0))
            ttft_ms = float(metrics.get("ttft_ms", elapsed_sec * 350.0))
            tps = out_tok / elapsed_sec if out_tok > 0 else 0.0
            cost_usd = estimate_llm_cost_usd(model_name, in_tok, out_tok)

            span.set_attribute("gen_ai.usage.input_tokens", in_tok)
            span.set_attribute("gen_ai.usage.output_tokens", out_tok)
            span.set_attribute("gen_ai.client.ttft_ms", round(ttft_ms, 2))
            span.set_attribute("gen_ai.client.tokens_per_sec", round(tps, 2))
            span.set_attribute("gen_ai.usage.cost_usd", cost_usd)
            span.set_attribute("causalbound.escalated_to_pro", bool(metrics.get("escalated", False)))
