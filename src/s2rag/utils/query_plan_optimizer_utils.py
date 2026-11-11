"""Shared features and scoring for the plan-level RAG query optimizer."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_PLAN_ORDER = (
    "base_retrieval",
    "schema_memory",
    "fact_join",
    "full_execution",
)
PLAN_ORDER = DEFAULT_PLAN_ORDER

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_ENTITY_RE = re.compile(r"(?:\b[A-Z][\w'-]*\b(?:\s+|$)){1,5}")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sample_key(row: Dict[str, Any]) -> Tuple[str, str]:
    dataset = str(row.get("dataset") or "unknown").strip().lower()
    sample_id = row.get("sample_id", row.get("index", row.get("question", "")))
    return dataset, str(sample_id)


def answer_quality(row: Dict[str, Any]) -> float:
    return 0.85 * as_float(row.get("f1")) + 0.15 * as_float(
        row.get("exact_match")
    )


def evidence_signature(row: Dict[str, Any]) -> str:
    """Hash only reader-visible evidence, never the sampled answer."""
    payload = {
        "qa_doc_titles": row.get("qa_doc_titles") or [],
        "qa_docs_preview": row.get("qa_docs_preview") or [],
        "qa_source_doc_titles": row.get("qa_source_doc_titles") or [],
        "qa_source_docs_preview": row.get("qa_source_docs_preview") or [],
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def query_state_features(row: Dict[str, Any]) -> Dict[str, float]:
    """Features available after base retrieval and before optional operators.

    Dataset names and gold-support coverage are intentionally excluded so a
    leave-one-dataset-out evaluation cannot memorize benchmarks or use labels.
    """
    question = str(row.get("question") or "")
    lower = question.lower()
    tokens = _TOKEN_RE.findall(lower)
    features: Dict[str, float] = {
        "bias": 1.0,
        "log_question_tokens": math.log1p(len(tokens)),
        "question_has_comparison": float(
            any(term in lower for term in ("both", "either", "older", "younger", "before", "after", "difference between"))
        ),
        "question_has_temporal_relation": float(
            any(term in lower for term in ("when", "year", "date", "born", "died", "before", "after"))
        ),
        "question_has_location_relation": float(
            any(term in lower for term in ("where", "country", "city", "location", "place"))
        ),
        "question_has_creator_relation": float(
            any(
                term in lower
                for term in (
                    "director",
                    "directed",
                    "author",
                    "wrote",
                    "written",
                    "created",
                    "composer",
                    "composed",
                    "producer",
                    "produced",
                )
            )
        ),
        "question_entity_count": float(min(8, len(_ENTITY_RE.findall(question)))),
    }
    wh_type = "other"
    for candidate in ("who", "where", "when", "which", "what", "how", "why"):
        if re.search(rf"\b{candidate}\b", lower):
            wh_type = candidate
            break
    features[f"wh={wh_type}"] = 1.0

    scores = [
        as_float(value)
        for value in (row.get("retrieved_doc_scores_top5") or [])
        if value is not None
    ][:5]
    if scores:
        top = max(0.0, scores[0])
        second = max(0.0, scores[1] if len(scores) > 1 else 0.0)
        total = sum(max(0.0, value) for value in scores)
        features.update(
            {
                "retrieval_log_top_score": math.log1p(1000.0 * top),
                "retrieval_top_margin": (top - second) / max(top, 1e-9),
                "retrieval_top_share": top / max(total, 1e-9),
                "retrieval_score_decay": scores[-1] / max(top, 1e-9),
            }
        )
    titles = [str(value or "") for value in row.get("retrieved_doc_titles_top5") or []]
    question_tokens = set(tokens)
    title_overlaps = []
    for title in titles[:5]:
        title_tokens = set(_TOKEN_RE.findall(title.lower()))
        title_overlaps.append(
            len(question_tokens & title_tokens) / max(1.0, float(len(title_tokens)))
        )
    features["retrieval_max_title_overlap"] = max(title_overlaps, default=0.0)
    features["retrieval_mean_title_overlap"] = (
        sum(title_overlaps) / max(1.0, float(len(title_overlaps)))
    )
    return features


def _dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def plan_cost_record(row: Dict[str, Any], runtime_per_query: float = 0.0) -> Dict[str, float]:
    """Return query-local optimizer cost plus observed wall-clock diagnostics.

    Wall-clock time is deliberately excluded from ``cost_score`` because whole
    runs are affected by API caching, retries, and server load. It remains in
    the record for latency reporting, while plan selection uses only costs that
    can be attributed to the current query.
    """
    metadata = _dict(row.get("metadata"))
    prompt_tokens = as_float(metadata.get("prompt_tokens"))
    completion_tokens = as_float(metadata.get("completion_tokens"))
    qsm = _dict(row.get("query_schema_memory"))
    fact_join = _dict(row.get("relation_fact_join"))
    planned = _dict(row.get("planned_retrieval"))
    support_join = _dict(row.get("selective_support_join"))
    admission = _dict(row.get("evidence_admission"))

    qsm_candidates = as_float(
        qsm.get("candidate_count", qsm.get("raw_candidate_count"))
    )
    fact_candidates = as_float(
        fact_join.get("candidate_count", fact_join.get("raw_candidate_count"))
    )
    planned_calls = (
        as_float(planned.get("retriever_calls"))
        + as_float(planned.get("planner_calls"))
        + as_float(planned.get("verifier_calls"))
    )
    support_calls = float(
        bool(support_join.get("applied"))
        or as_float(support_join.get("accepted_count")) > 0.0
    )
    operator_work = (
        0.02 * min(50.0, qsm_candidates)
        + 0.015 * min(80.0, fact_candidates)
        + 0.12 * planned_calls
        + 0.08 * support_calls
    )
    llm_tokens = prompt_tokens + completion_tokens


    runtime_per_query = max(0.0, float(runtime_per_query))
    cost_score = llm_tokens / 1000.0 + operator_work
    observed_wall_cost_score = cost_score + runtime_per_query / 10.0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_tokens": llm_tokens,
        "admitted_tokens": as_float(admission.get("admitted_candidate_tokens")),
        "qsm_candidates": qsm_candidates,
        "fact_join_candidates": fact_candidates,
        "planned_calls": planned_calls,
        "support_calls": support_calls,
        "operator_work": operator_work,
        "runtime_per_query": runtime_per_query,
        "cost_score": cost_score,
        "observed_wall_cost_score": observed_wall_cost_score,
    }


def pareto_nondominated(
    outcomes: Dict[str, Dict[str, float]],
    *,
    plan_order: Optional[Iterable[str]] = None,
    quality_epsilon: float = 1e-9,
    cost_epsilon: float = 1e-9,
) -> List[str]:
    ordered_plans = tuple(plan_order or PLAN_ORDER)
    plan_rank = {plan: index for index, plan in enumerate(ordered_plans)}
    nondominated = []
    for plan, current in outcomes.items():
        dominated = False
        for other_plan, other in outcomes.items():
            if other_plan == plan:
                continue
            quality_no_worse = other["quality"] >= current["quality"] - quality_epsilon
            cost_no_worse = other["cost_score"] <= current["cost_score"] + cost_epsilon
            strictly_better = (
                other["quality"] > current["quality"] + quality_epsilon
                or other["cost_score"] < current["cost_score"] - cost_epsilon
            )
            if quality_no_worse and cost_no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            nondominated.append(plan)
    return sorted(nondominated, key=lambda value: plan_rank.get(value, len(plan_rank)))


def predict_query_plan(
    model: Optional[Dict[str, Any]],
    features: Dict[str, float],
) -> Dict[str, Any]:
    if not model or not bool(model.get("enabled")):
        fallback = str((model or {}).get("fallback_plan") or "full_execution")
        return {"plan": fallback, "confidence": 0.0, "reason": "model_unavailable"}
    classes = [str(value) for value in model.get("classes") or []]
    coefficients = model.get("coefficients") or {}
    intercepts = model.get("intercepts") or {}
    if not classes:
        return {"plan": str(model.get("fallback_plan") or "full_execution"), "confidence": 0.0, "reason": "no_classes"}
    logits = {}
    for plan in classes:
        value = as_float(intercepts.get(plan))
        plan_weights = coefficients.get(plan) or {}
        for feature, feature_value in features.items():
            value += as_float(plan_weights.get(feature)) * as_float(feature_value)
        logits[plan] = value
    max_logit = max(logits.values())
    exp_values = {plan: math.exp(min(60.0, value - max_logit)) for plan, value in logits.items()}
    denominator = max(1e-12, sum(exp_values.values()))
    probabilities = {plan: value / denominator for plan, value in exp_values.items()}
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    selected, confidence = ordered[0]
    margin = confidence - (ordered[1][1] if len(ordered) > 1 else 0.0)
    min_margin = as_float(model.get("min_probability_margin"), 0.05)
    if margin < min_margin:
        selected = str(model.get("fallback_plan") or selected)
        reason = "low_margin_fallback"
    else:
        reason = "learned_plan"
    return {
        "plan": selected,
        "confidence": round(confidence, 6),
        "margin": round(margin, 6),
        "reason": reason,
        "probabilities": {plan: round(value, 6) for plan, value in probabilities.items()},
    }
