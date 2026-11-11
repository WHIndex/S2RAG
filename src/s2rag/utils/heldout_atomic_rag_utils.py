"""Gold-free helpers for the held-out recovery versus atomic-RAG study."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence


ATOMIC_QUERY_VERSION = "plan_atomic_retrieval_query_v1"


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def operator_relation_label(operator: Mapping[str, Any]) -> str:
    """Return a deterministic, question-derived label for one plan operator."""
    tokens = [_clean(value) for value in operator.get("relation_tokens") or []]
    tokens = [value for value in tokens if value]
    if tokens:
        return " / ".join(dict.fromkeys(tokens[:4]))
    families = [_clean(value) for value in operator.get("predicate_families") or []]
    families = [value for value in families if value]
    if families:
        return " / ".join(dict.fromkeys(families[:3]))
    return _clean(operator.get("canonical_relation")) or "unspecified relation"


def build_plan_atomic_retrieval_queries(
    question: str,
    plan: Mapping[str, Any],
    *,
    max_queries: int = 2,
) -> list[dict[str, Any]]:
    """Render one retrieval request per physical relation without using answers.

    The baseline deliberately does not consume runtime bindings or S2RAG
    recovery decisions.  Every request is fixed by the original question and
    the already-frozen logical plan before retrieval starts.
    """
    normalized_question = _clean(question)
    rows: list[dict[str, Any]] = []
    for position, raw_operator in enumerate(plan.get("operators") or []):
        if len(rows) >= max(0, int(max_queries)):
            break
        operator = dict(raw_operator or {})
        if _clean(operator.get("operator_type")) not in {
            "indexed_fact_lookup",
            "lookup_join",
            "project",
        }:
            continue
        relation = operator_relation_label(operator)
        input_role = _clean(operator.get("semantic_input_role") or operator.get("input_role"))
        output_role = _clean(operator.get("semantic_output_role") or operator.get("output_role"))
        query = (
            f"Original question: {normalized_question}\n"
            f"Retrieve a passage that states evidence for relation step {position + 1}: "
            f"{input_role or 'input entity'} --[{relation}]--> "
            f"{output_role or _clean(operator.get('output_type')) or 'output entity'}."
        )
        rows.append(
            {
                "version": ATOMIC_QUERY_VERSION,
                "operator_id": _clean(operator.get("operator_id")) or f"op{position}",
                "stage_index": int(operator.get("stage_index", position) or position),
                "relation_label": relation,
                "input_role": input_role,
                "output_role": output_role,
                "query": query,
            }
        )
    return rows


def budgeted_unique_source_ids(
    ranked_source_groups: Sequence[Iterable[int]],
    *,
    excluded_source_ids: Iterable[int] = (),
    max_sources: int = 4,
) -> list[int]:
    """Round-robin ranked sources under one shared additional-source budget."""
    groups = [[int(value) for value in group] for group in ranked_source_groups]
    seen = {int(value) for value in excluded_source_ids}
    selected: list[int] = []
    depth = 0
    while len(selected) < max(0, int(max_sources)):
        progressed = False
        for group in groups:
            if depth >= len(group):
                continue
            progressed = True
            source_id = group[depth]
            if source_id in seen:
                continue
            seen.add(source_id)
            selected.append(source_id)
            if len(selected) >= int(max_sources):
                break
        if not progressed:
            break
        depth += 1
    return selected
