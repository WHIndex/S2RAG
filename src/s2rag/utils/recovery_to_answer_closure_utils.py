"""Frozen evidence rendering helpers for the recovery-to-answer diagnostic.

This module is deliberately independent from the online query executor.  It
only consumes immutable executor traces and turns already certified source
documents into paired Reader inputs.  It must not repair, verify, or otherwise
reinterpret tuples while rendering evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence


RENDERER_VERSION = "recovery_to_answer_reader_renderer_v1"
REQUEST_VERSION = "recovery_to_answer_exact_request_v1"
ARMS = ("DENSE_BASE", "INDEX_EVIDENCE", "RECOVERY_EVIDENCE")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceDocument:
    source_id: int
    title: str
    content: str
    origin: str
    rank: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "content": self.content,
            "content_sha256": sha256_text(self.content),
            "origin": self.origin,
            "rank": self.rank,
        }


def document_from_content(
    source_id: int,
    content: str,
    *,
    origin: str,
    rank: int,
) -> SourceDocument:
    normalized = str(content or "").strip()
    title = normalized.splitlines()[0].strip() if normalized else ""
    return SourceDocument(
        source_id=int(source_id),
        title=title,
        content=normalized,
        origin=str(origin),
        rank=int(rank),
    )


def _fact_usage_id(fact: Mapping[str, Any]) -> str:
    return str(fact.get("fact_usage_id") or fact.get("fact_id") or "")


def fact_map(trace: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index both physical facts and per-obligation usages without conflating them."""
    result: dict[str, dict[str, Any]] = {}
    for key in ("facts", "fact_usages"):
        for raw in trace.get(key) or []:
            fact = dict(raw)
            usage_id = _fact_usage_id(fact)
            physical_id = str(fact.get("physical_fact_id") or "")
            fact_id = str(fact.get("fact_id") or "")
            for identifier in (usage_id, fact_id, physical_id):
                if identifier and identifier not in result:
                    result[identifier] = fact
            if usage_id:


                result[usage_id] = fact
    return result


def requested_evidence_fact_ids(trace: Mapping[str, Any]) -> list[str]:
    """Return only evidence already exposed by the frozen execution contract."""
    explicit = trace.get("reader_evidence_fact_ids")
    if isinstance(explicit, list):
        return [str(value) for value in explicit if str(value)]
    if bool(trace.get("question_complete_supported_witness")):
        selected = dict(trace.get("selected_witness") or {})
        return [
            str(value)
            for value in selected.get("path_fact_usage_ids") or []
            if str(value)
        ]
    return []


def select_certified_evidence(
    trace: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select frozen Reader evidence and explain every rejected requested row.

    No verifier is run here.  A tuple is renderable only when the canonical
    trace already records source support, obligation admission, and an exact
    source certificate.
    """
    facts = fact_map(trace)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for requested_id in requested_evidence_fact_ids(trace):
        if requested_id in seen:
            continue
        seen.add(requested_id)
        fact = facts.get(requested_id)
        reasons: list[str] = []
        if not fact:
            reasons.append("fact_usage_not_found_in_frozen_trace")
        else:
            if str(fact.get("support_status") or "") != "SUPPORTED":
                reasons.append("not_source_supported")
            if str(fact.get("obligation_admission_status") or "") != "ADMITTED":
                reasons.append("not_obligation_admitted")
            certificate = dict(fact.get("certified_assertion") or {})
            if not certificate:
                reasons.append("missing_certified_assertion")
            if fact.get("source_id") is None and certificate.get("source_id") is None:
                reasons.append("missing_source_id")
        if reasons:
            rejected.append({"fact_usage_id": requested_id, "reasons": reasons})
            continue
        row = dict(fact or {})
        certificate = dict(row.get("certified_assertion") or {})
        row["resolved_source_id"] = int(
            row.get("source_id")
            if row.get("source_id") is not None
            else certificate["source_id"]
        )
        accepted.append(row)
    return accepted, rejected


def stable_unique_documents(
    base_documents: Sequence[SourceDocument],
    evidence_source_ids: Sequence[int],
    get_content: Callable[[int], str],
) -> tuple[list[SourceDocument], list[dict[str, Any]]]:
    """Append full source passages in stable order and report de-duplication."""
    final = list(base_documents)
    seen = {document.source_id for document in base_documents}
    decisions: list[dict[str, Any]] = []
    for evidence_rank, raw_source_id in enumerate(evidence_source_ids):
        source_id = int(raw_source_id)
        if source_id in seen:
            decisions.append(
                {
                    "source_id": source_id,
                    "decision": "DEDUPLICATED_ALREADY_IN_BASE",
                    "evidence_rank": evidence_rank,
                }
            )
            continue
        content = str(get_content(source_id) or "").strip()
        if not content:
            decisions.append(
                {
                    "source_id": source_id,
                    "decision": "REJECTED_MISSING_SOURCE_CONTENT",
                    "evidence_rank": evidence_rank,
                }
            )
            continue
        seen.add(source_id)
        final.append(
            document_from_content(
                source_id,
                content,
                origin="certified_execution_evidence",
                rank=len(final),
            )
        )
        decisions.append(
            {
                "source_id": source_id,
                "decision": "APPENDED",
                "evidence_rank": evidence_rank,
            }
        )
    return final, decisions


def truncate_documents_to_token_budget(
    *,
    question: str,
    documents: Sequence[SourceDocument],
    render_messages: Callable[[str, Sequence[str]], Sequence[Mapping[str, str]]],
    count_message_tokens: Callable[[Sequence[Mapping[str, str]]], int],
    max_input_tokens: int,
    protected_base_count: int,
) -> tuple[list[SourceDocument], list[dict[str, Any]], int]:
    """Apply one shared tail-drop policy without shortening evidence to answer spans."""
    kept = list(documents)
    dropped: list[dict[str, Any]] = []
    while kept:
        messages = list(
            render_messages(question, [document.content for document in kept])
        )
        token_count = int(count_message_tokens(messages))
        if token_count <= int(max_input_tokens):
            return kept, dropped, token_count
        if len(kept) <= int(protected_base_count):
            raise ValueError(
                "base_dense_context_exceeds_fixed_reader_token_budget:"
                f"tokens={token_count}:limit={max_input_tokens}"
            )
        removed = kept.pop()
        dropped.append(
            {
                "source_id": removed.source_id,
                "content_sha256": sha256_text(removed.content),
                "reason": "fixed_input_token_budget_tail_drop",
            }
        )
    raise ValueError("reader_context_became_empty")


def build_exact_request(
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    seed: int,
    temperature: float,
    max_completion_tokens: int,
    n: int = 1,
) -> dict[str, Any]:
    request = {
        "version": REQUEST_VERSION,
        "renderer_version": RENDERER_VERSION,
        "model": str(model),
        "messages": [dict(message) for message in messages],
        "seed": int(seed),
        "temperature": float(temperature),
        "max_completion_tokens": int(max_completion_tokens),
        "n": int(n),
    }
    request["request_hash"] = sha256_value(request)
    return request


def conservative_message_token_upper_bound(
    messages: Sequence[Mapping[str, str]],
) -> int:
    """Bound BPE tokens by UTF-8 bytes plus conservative chat framing.

    This is intentionally looser than the tokenizer estimate used for context
    truncation.  It is used only to enforce the experiment-wide API budget
    when the exact provider tokenizer is unavailable.
    """
    return 3 + sum(
        8
        + len(str(message.get("role") or "").encode("utf-8"))
        + len(str(message.get("content") or "").encode("utf-8"))
        for message in messages
    )


def assert_gold_free_reader_record(record: Mapping[str, Any]) -> None:
    forbidden_keys = {
        "answer",
        "answers",
        "gold",
        "gold_answer",
        "gold_answers",
        "supporting_facts",
        "best_gold",
    }

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).strip().lower()
                if normalized in forbidden_keys:
                    raise ValueError(f"gold_key_in_reader_input:{path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(record, "reader_record")


def retrieval_hash(ids: Sequence[int], scores: Sequence[float]) -> str:
    return sha256_value(
        {"ids": [int(value) for value in ids], "scores": [float(value) for value in scores]}
    )


def evidence_source_ids(facts: Sequence[Mapping[str, Any]]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for fact in facts:
        source_id = int(fact["resolved_source_id"])
        if source_id not in seen:
            result.append(source_id)
            seen.add(source_id)
    return result


def request_groups_in_frozen_order(
    records: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    by_query: dict[str, list[Mapping[str, Any]]] = {}
    order: list[str] = []
    for record in records:
        query_id = str(record.get("query_id") or "")
        if query_id not in by_query:
            order.append(query_id)
            by_query[query_id] = []
        by_query[query_id].append(record)
    return [by_query[query_id] for query_id in order]


def choose_budgeted_query_prefix(
    records: Sequence[Mapping[str, Any]],
    *,
    max_requests: int,
    max_token_upper_bound: int,
) -> tuple[set[str], dict[str, Any]]:
    """Admit complete query triplets in frozen order under hard Reader limits."""
    selected_queries: set[str] = set()
    seen_requests: set[str] = set()
    input_tokens = 0
    output_tokens = 0
    for group in request_groups_in_frozen_order(records):
        new_records: dict[str, Mapping[str, Any]] = {}
        for record in group:
            request_hash = str(record["request_hash"])
            if request_hash not in seen_requests:
                new_records[request_hash] = record
        proposed_requests = len(seen_requests) + len(new_records)
        proposed_input = input_tokens + sum(
            int(
                record.get("input_token_budget_upper_bound")
                or record["input_tokens_estimate"]
            )
            for record in new_records.values()
        )
        proposed_output = output_tokens + sum(
            int(record["max_completion_tokens"])
            for record in new_records.values()
        )
        if (
            proposed_requests > int(max_requests)
            or proposed_input + proposed_output > int(max_token_upper_bound)
        ):
            break
        query_id = str(group[0]["query_id"])
        selected_queries.add(query_id)
        seen_requests.update(new_records)
        input_tokens = proposed_input
        output_tokens = proposed_output
    return selected_queries, {
        "selected_query_count": len(selected_queries),
        "unique_request_count": len(seen_requests),
        "input_token_budget_upper_bound": input_tokens,
        "max_output_tokens": output_tokens,
        "total_token_upper_bound": input_tokens + output_tokens,
        "max_requests": int(max_requests),
        "max_token_upper_bound": int(max_token_upper_bound),
    }


def iter_complete_actions(trace: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for row in trace.get("operator_trace") or []:
        if str(row.get("operator") or "") == "COMPLETE":
            yield dict(row)
