"""Persist query-execution state before the final Reader runs.

Reader cache preflight intentionally aborts before inference when any prompt is
missing.  Without this export, a single missing cache key also hides the
already-completed relational execution for every query in the batch.  The
export is diagnostic only: it neither changes prompts nor execution state.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence


def _query_field(query: Any, name: str, default: Any = None) -> Any:
    if isinstance(query, Mapping):
        return query.get(name, default)
    return getattr(query, name, default)


def pre_reader_trace_path(cache_coverage_report_path: str) -> str:
    """Return the trace path paired with one cache-coverage report."""
    report_path = os.path.abspath(str(cache_coverage_report_path or ""))
    stem, _ = os.path.splitext(report_path)
    return f"{stem}_pre_reader_trace.jsonl"


def write_pre_reader_trace(
    path: str,
    queries: Sequence[Any],
    coverage_records: Sequence[Mapping[str, Any]],
    *,
    original_query_count: int,
    reader_messages: Sequence[Any] | None = None,
) -> str:
    """Atomically write execution state aligned with Reader cache records."""
    output_path = os.path.abspath(path)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    coverage_by_index = {
        int(record.get("index", -1)): dict(record)
        for record in coverage_records
        if int(record.get("index", -1)) >= 0
    }
    temp_path = f"{output_path}.tmp.{os.getpid()}"
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            for index, query in enumerate(queries):
                cache_record = coverage_by_index.get(index, {})
                record = {
                    "version": "s2rag_pre_reader_trace_v1",
                    "processing_index": index,
                    "original_query": index < int(original_query_count),
                    "sample_key": str(_query_field(query, "sample_key", "") or ""),
                    "question": str(_query_field(query, "question", "") or ""),
                    "qa_evidence_signature": str(
                        _query_field(query, "qa_evidence_signature", "") or ""
                    ),
                    "qa_doc_titles": list(
                        _query_field(query, "qa_doc_titles", []) or []
                    ),
                    "logical_query_plan": _query_field(
                        query, "logical_query_plan", None
                    ),
                    "relation_fact_join": _query_field(
                        query, "relation_fact_join", None
                    ),
                    "paper_evidence_execution": _query_field(
                        query, "paper_evidence_execution", None
                    ),
                    "reader_cache": {
                        key: cache_record.get(key)
                        for key in (
                            "key",
                            "present",
                            "message_sha256",
                            "message_chars",
                            "counterfactual_reader_prompt",
                        )
                        if key in cache_record
                    },
                }
                if reader_messages is not None:
                    if index >= len(reader_messages):
                        raise ValueError(
                            "reader_messages must align with every pre-reader query"
                        )
                    record["reader_messages"] = reader_messages[index]
                handle.write(json.dumps(record, ensure_ascii=False, default=str))
                handle.write("\n")
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return output_path
