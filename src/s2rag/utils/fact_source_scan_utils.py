from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .relation_fact_join_utils import (
    canonical_fact_tuple_key,
    canonical_schema_predicate_families,
)


_FALLBACK_STOP_REASONS = {
    "budget_insufficient",
    "no_executable_action",
    "no_progress_limit",
    "root_binding_unavailable",
    "unsupported_query_shape",
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return result


def _fact_score(
    fact_rank: int,
    top_k_fact_indices: Sequence[int],
    query_fact_scores: Any,
) -> float:
    if fact_rank >= len(top_k_fact_indices):
        return 0.0
    try:
        fact_index = int(top_k_fact_indices[fact_rank])
        return _as_float(query_fact_scores[fact_index])
    except (IndexError, TypeError, ValueError):
        return 0.0


def _logical_obligations(logical_plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    obligations: List[Dict[str, Any]] = []
    for index, raw_operator in enumerate((logical_plan or {}).get("operators") or []):
        operator = dict(raw_operator or {})
        obligation_id = str(operator.get("operator_id") or f"op{index}")
        predicate_families = {
            str(value).strip()
            for value in operator.get("predicate_families") or []
            if str(value).strip()
        }
        relation_tokens = {
            str(value).strip().casefold()
            for value in operator.get("relation_tokens") or []
            if str(value).strip()
        }
        obligations.append({
            "obligation_id": obligation_id,
            "stage_index": int(operator.get("stage_index", index) or index),
            "slot_plan_stage": str(operator.get("slot_plan_stage") or ""),
            "predicate_families": predicate_families,
            "relation_tokens": relation_tokens,
        })
    return obligations


def _fact_coverage_keys(
    fact: Sequence[Any],
    stats: Mapping[str, Any],
    obligations: Sequence[Mapping[str, Any]],
    tuple_key: str,
) -> Tuple[List[str], List[str], List[str]]:
    relation = fact[1] if len(fact) > 1 else ""
    predicate_families = {
        str(value).strip()
        for value in canonical_schema_predicate_families(relation)
        if str(value).strip()
    }
    relation_tokens = {
        str(value).strip().casefold()
        for value in stats.get("relation_tokens") or []
        if str(value).strip()
    }
    matched_obligations = []
    for obligation in obligations:
        if (
            predicate_families
            & set(obligation.get("predicate_families") or set())
            or relation_tokens & set(obligation.get("relation_tokens") or set())
        ):
            matched_obligations.append(str(obligation["obligation_id"]))
    if matched_obligations:
        coverage_keys = [f"obligation:{value}" for value in matched_obligations]
    elif obligations:
        coverage_keys = []
    elif predicate_families:
        coverage_keys = [f"family:{value}" for value in sorted(predicate_families)]
    else:
        coverage_keys = [f"tuple:{tuple_key}"]
    return (
        sorted(set(coverage_keys)),
        sorted(predicate_families),
        sorted(set(matched_obligations)),
    )
def promote_fact_source_docs(
    sorted_doc_ids: Sequence[int],
    sorted_doc_scores: Sequence[float],
    *,
    top_k_facts: Sequence[Sequence[Any]],
    top_k_fact_indices: Sequence[int],
    query_fact_scores: Any,
    relation_index: Mapping[str, Any],
    executor_stop_reason: str = "",
    enabled: bool = False,
    fallback_only: bool = True,
    fact_top_k: int = 8,
    target_top_k: int = 5,
    max_promotions: int = 1,
    min_fact_score: float = 0.60,
    max_source_docs_per_fact: int = 2,
    logical_plan: Mapping[str, Any] | None = None,
    selection_mode: str = "legacy",
    protect_dense_top_k: int = 0,
) -> Tuple[List[int], List[float], Dict[str, Any]]:
    """Promote provenance documents for semantically retrieved facts.

    This is a physical source-access operator, not an answer materializer.  It
    consumes facts already selected by the dense fact retriever, resolves each
    exact tuple through the fact index, and replaces only the tail of the
    reader-visible top-k prefix.  It never fabricates a tuple or exposes a fact
    value without its source passage.
    """
    doc_ids = [
        int(value)
        for value in list(sorted_doc_ids if sorted_doc_ids is not None else [])
    ]
    scores = [
        _as_float(value)
        for value in list(
            sorted_doc_scores if sorted_doc_scores is not None else []
        )
    ]
    if len(scores) < len(doc_ids):
        scores.extend([0.0] * (len(doc_ids) - len(scores)))

    stop_reason = str(executor_stop_reason or "").strip()
    fact_top_k = max(0, int(fact_top_k))
    target_top_k = max(1, int(target_top_k))
    max_promotions = max(0, int(max_promotions))
    max_source_docs_per_fact = max(1, int(max_source_docs_per_fact))
    selection_mode = str(selection_mode or "legacy").strip().lower()
    if selection_mode not in {"legacy", "obligation_aware"}:
        selection_mode = "legacy"
    protect_dense_top_k = max(0, int(protect_dense_top_k))
    obligations = _logical_obligations(logical_plan or {})
    required_obligations = {
        f"obligation:{row['obligation_id']}" for row in obligations
    }
    info: Dict[str, Any] = {
        "version": (
            "obligation_aware_fact_source_scan_v2"
            if selection_mode == "obligation_aware"
            else "fact_source_scan_v1"
        ),
        "enabled": bool(enabled),
        "applied": False,
        "fallback_only": bool(fallback_only),
        "selection_mode": selection_mode,
        "protect_dense_top_k": protect_dense_top_k,
        "executor_stop_reason": stop_reason,
        "fact_top_k": fact_top_k,
        "target_top_k": target_top_k,
        "max_promotions": max_promotions,
        "min_fact_score": float(min_fact_score),
        "candidate_fact_count": 0,
        "matched_fact_count": 0,
        "candidate_doc_count": 0,
        "promoted_doc_ids": [],
        "displaced_doc_ids": [],
        "required_obligations": sorted(required_obligations),
        "covered_before": [],
        "covered_after": [],
        "uncovered_after": sorted(required_obligations),
        "selection_trace": [],
        "candidate_rows": [],
        "reason": "disabled" if not enabled else "no_candidates",
    }
    if not enabled:
        return doc_ids, scores[: len(doc_ids)], info
    if not doc_ids or max_promotions == 0 or fact_top_k == 0:
        info["reason"] = "empty_budget_or_ranking"
        return doc_ids, scores[: len(doc_ids)], info
    if fallback_only and stop_reason not in _FALLBACK_STOP_REASONS:
        info["reason"] = "verified_or_conflicted_plan_preferred"
        return doc_ids, scores[: len(doc_ids)], info

    tuple_stats = dict((relation_index or {}).get("tuple_stats") or {})
    if not tuple_stats:
        tuple_stats = dict(
            ((relation_index or {}).get("fact_reliability") or {}).get(
                "tuple_stats"
            )
            or {}
        )
    if not tuple_stats:
        info["reason"] = "tuple_index_unavailable"
        return doc_ids, scores[: len(doc_ids)], info

    original_rank = {doc_id: rank for rank, doc_id in enumerate(doc_ids)}
    valid_doc_ids = set(doc_ids)
    candidates: Dict[int, Dict[str, Any]] = {}
    facts = list(top_k_facts or [])[:fact_top_k]
    info["candidate_fact_count"] = len(facts)
    for fact_rank, fact in enumerate(facts):
        tuple_key = canonical_fact_tuple_key(fact)
        if not tuple_key:
            continue
        stats = dict(tuple_stats.get(tuple_key) or {})
        if not stats:
            continue
        fact_score = _fact_score(
            fact_rank,
            top_k_fact_indices,
            query_fact_scores,
        )
        if fact_score < float(min_fact_score):
            continue
        info["matched_fact_count"] += 1
        triple = [str(part) for part in list(fact)[:3]]
        coverage_keys, predicate_families, matched_obligations = (
            _fact_coverage_keys(triple, stats, obligations, tuple_key)
        )
        if selection_mode == "obligation_aware" and obligations and not coverage_keys:
            continue
        source_doc_ids = []
        for raw_doc_id in stats.get("doc_ids") or []:
            try:
                doc_id = int(raw_doc_id)
            except (TypeError, ValueError):
                continue
            if doc_id in valid_doc_ids:
                source_doc_ids.append(doc_id)
        source_doc_ids = sorted(
            set(source_doc_ids),
            key=lambda value: (original_rank[value], value),
        )[:max_source_docs_per_fact]
        for doc_id in source_doc_ids:
            row = candidates.setdefault(doc_id, {
                "doc_id": doc_id,
                "dense_rank_before": original_rank[doc_id] + 1,
                "fact_rank": fact_rank + 1,
                "fact_score": round(fact_score, 6),
                "tuple_keys": [],
                "triples": [],
                "predicate_families": [],
                "matched_obligations": [],
                "coverage_keys": [],
                "tuple_doc_count": int(stats.get("doc_count", 0) or 0),
            })
            row["fact_rank"] = min(int(row["fact_rank"]), fact_rank + 1)
            row["fact_score"] = max(float(row["fact_score"]), round(fact_score, 6))
            row["tuple_doc_count"] = min(
                int(row["tuple_doc_count"]),
                int(stats.get("doc_count", 0) or 0),
            )
            if tuple_key not in row["tuple_keys"]:
                row["tuple_keys"].append(tuple_key)
            if triple not in row["triples"]:
                row["triples"].append(triple)
            row["predicate_families"] = sorted(set(
                list(row["predicate_families"]) + predicate_families
            ))
            row["matched_obligations"] = sorted(set(
                list(row["matched_obligations"]) + matched_obligations
            ))
            row["coverage_keys"] = sorted(set(
                list(row["coverage_keys"]) + coverage_keys
            ))

    target_count = min(len(doc_ids), target_top_k)
    ranked_candidates = sorted(
        (
            row
            for row in candidates.values()
            if row["dense_rank_before"] > target_count
        ),
        key=lambda row: (
            -row["fact_score"],
            row["fact_rank"],
            row["dense_rank_before"],
            row["doc_id"],
        ),
    )
    info["candidate_doc_count"] = len(ranked_candidates)
    info["candidate_rows"] = ranked_candidates[:16]
    if selection_mode == "obligation_aware":
        covered = set()
        for row in candidates.values():
            if int(row["dense_rank_before"]) <= target_count:
                covered.update(row.get("coverage_keys") or [])
        info["covered_before"] = sorted(covered)
        selected = []
        remaining = list(ranked_candidates)
        promotion_budget = min(
            max_promotions,
            max(0, target_count - min(protect_dense_top_k, target_count)),
        )
        while remaining and len(selected) < promotion_budget:
            ranked = []
            for row in remaining:
                new_coverage = set(row.get("coverage_keys") or []) - covered
                if not new_coverage:
                    continue
                new_required = new_coverage & required_obligations
                ranked.append((
                    -len(new_required),
                    -len(new_coverage),
                    -float(row.get("fact_score", 0.0)),
                    int(row.get("fact_rank", 10**9)),
                    int(row.get("tuple_doc_count", 10**9)),
                    int(row.get("dense_rank_before", 10**9)),
                    int(row["doc_id"]),
                    row,
                    sorted(new_coverage),
                    sorted(new_required),
                ))
            if not ranked:
                break
            *_, row, new_coverage, new_required = min(ranked)
            selected.append(row)
            covered.update(new_coverage)
            info["selection_trace"].append({
                "doc_id": int(row["doc_id"]),
                "new_coverage": new_coverage,
                "new_required_obligations": new_required,
                "fact_score": float(row["fact_score"]),
                "dense_rank_before": int(row["dense_rank_before"]),
            })
            remaining = [
                candidate
                for candidate in remaining
                if int(candidate["doc_id"]) != int(row["doc_id"])
            ]
        info["covered_after"] = sorted(covered)
        info["uncovered_after"] = sorted(required_obligations - covered)
    else:
        selected = ranked_candidates[:max_promotions]
    if not selected:
        if not candidates:
            info["reason"] = "no_exact_high_score_fact_sources"
        elif not ranked_candidates:
            info["reason"] = "all_source_docs_already_in_target_topk"
        elif selection_mode == "obligation_aware":
            info["reason"] = "no_new_obligation_coverage"
        else:
            info["reason"] = "no_candidates"
        return doc_ids, scores[: len(doc_ids)], info

    promoted_doc_ids = [int(row["doc_id"]) for row in selected]
    protected_count = max(
        min(protect_dense_top_k, target_count),
        target_count - len(promoted_doc_ids),
    )
    prefix = doc_ids[:protected_count]
    selected_set = set(prefix) | set(promoted_doc_ids)
    reordered = (
        prefix
        + promoted_doc_ids
        + [doc_id for doc_id in doc_ids if doc_id not in selected_set]
    )
    score_by_doc = dict(zip(doc_ids, scores))
    high = scores[0] if scores else 1.0
    epsilon = max(abs(high) * 1e-7, 1e-7)
    reordered_scores = [high - epsilon * rank for rank in range(len(reordered))]
    for rank, doc_id in enumerate(reordered[:protected_count]):
        reordered_scores[rank] = score_by_doc.get(doc_id, reordered_scores[rank])

    new_topk = set(reordered[:target_count])
    info.update(
        {
            "applied": reordered != doc_ids,
            "promoted_doc_ids": promoted_doc_ids,
            "displaced_doc_ids": [
                doc_id for doc_id in doc_ids[:target_count] if doc_id not in new_topk
            ],
            "reason": (
                "obligation_fact_sources_promoted"
                if selection_mode == "obligation_aware"
                else "provenance_fact_source_promoted"
            ),
        }
    )
    return reordered, reordered_scores, info
