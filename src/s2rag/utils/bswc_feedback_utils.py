"""Feedback-calibrated source values for Budgeted Witness Completion.

The global model is a frozen, interpretable table of smoothed Bernoulli
outcomes.  Query-local updates are kept in the executor state and never leak
between queries.  Neither path consumes QA answers or unread passage text.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple


CALIBRATED_POLICIES = {
    "bswc_empirical_source_type",
    "bswc_feedback_calibrated",
    "bswc_feedback_replan",
    "bswc_feedback_stopping",
    "bswc_witness_calibrated",
    "bswc_witness_stopping",
    "bswc_feedback_full",
    "bswc_risk_aware_search",
    "bswc_adaptive_risk_cover",
    "bswc_feasibility_preserving_cover",
    "bswc_continuation_safe_cover",
    "bswc_binding_qualified_continuation_cover",
    "bswc_rank_consistent_binding_qualified_cover",
    "bswc_precedence_aware_cover",
    "bswc_quality_first_precedence_cover",
    "bswc_global_witness_reservation_cover",
}

FEEDBACK_POLICIES = {
    "bswc_feedback_replan",
    "bswc_feedback_stopping",
    "bswc_feedback_full",
    "bswc_risk_aware_search",
    "bswc_adaptive_risk_cover",
    "bswc_feasibility_preserving_cover",
    "bswc_continuation_safe_cover",
    "bswc_binding_qualified_continuation_cover",
    "bswc_rank_consistent_binding_qualified_cover",
    "bswc_precedence_aware_cover",
    "bswc_quality_first_precedence_cover",
    "bswc_global_witness_reservation_cover",
}

STOPPING_POLICIES = {
    "bswc_feedback_stopping",
    "bswc_witness_stopping",
    "bswc_feedback_full",
}




WITNESS_GAIN_POLICIES = {
    "bswc_witness_calibrated",
    "bswc_witness_stopping",
    "bswc_feedback_full",
}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _bounded(value: Any, default: float = 0.5) -> float:
    return min(1.0, max(0.0, _float(value, default)))


def stable_fold(sample_key: str, folds: int = 5) -> int:
    digest = hashlib.sha256(str(sample_key).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % max(2, int(folds))


def relevance_bucket(value: Any) -> str:
    score = _bounded(value, 0.0)
    if score >= 0.85:
        return "very_high"
    if score >= 0.60:
        return "high"
    if score >= 0.30:
        return "medium"
    return "low"


def event_type_for_operator(operator: str, *, witness: bool = False) -> str:
    name = str(operator or "")
    if witness and name == "SOURCE_INSPECT":
        return "inspect_witness_progress"
    return {
        "BOUND_DENSE_PROBE": "probe_candidate_gain",
        "SOURCE_INSPECT": "inspect_binding_gain",
        "VERIFY": "verify_source_support",
    }.get(name, "")


def action_features(
    action: Mapping[str, Any], state: Mapping[str, Any], *, witness: bool = False
) -> Dict[str, str]:
    """Extract only information available before the action executes."""
    plan = dict(state.get("logical_plan") or {})
    operators = list(plan.get("operators") or [])
    operator_index = _int(action.get("operator_index"), -1)
    operator = (
        dict(operators[operator_index])
        if 0 <= operator_index < len(operators)
        else {}
    )
    candidate = dict(action.get("source_candidate") or {})
    predicate = "|".join(sorted(
        str(value) for value in operator.get("predicate_families") or [] if str(value)
    )) or "unknown"
    subtype = "|".join(sorted(
        str(value)
        for value in operator.get("required_predicate_subtypes") or []
        if str(value)
    )) or "unknown"
    source_type = str(
        candidate.get("source_type") or action.get("source_type") or "none"
    )
    title_match = bool(
        candidate.get("candidate_entity_key")
        and candidate.get("candidate_entity_key") == candidate.get("bound_key")
    ) or source_type == "title_entity_catalog"
    binding = next(
        (
            row for row in state.get("bindings") or []
            if str(row.get("binding_id") or "") == str(action.get("binding_id") or "")
        ),
        {},
    )
    path_fact_ids = list(binding.get("path_fact_ids") or [])
    facts = state.get("facts") or {}
    binding_supported = bool(path_fact_ids) and all(
        str((facts.get(fact_id) or {}).get("support_status") or "") == "SUPPORTED"
        for fact_id in path_fact_ids
    )
    return {
        "event": event_type_for_operator(str(action.get("operator") or ""), witness=witness),
        "source_type": source_type,
        "query_shape": str(plan.get("plan_kind") or "unknown"),
        "predicate": predicate,
        "subtype": subtype,
        "stage": str(operator.get("slot_plan_stage") or "unknown"),
        "title_match": "yes" if title_match else "no",
        "relevance": relevance_bucket(
            action.get("source_relevance", candidate.get("score", 0.0))
        ),
        "reuse": "yes" if action.get("source_reused") else "no",
        "binding_supported": "yes" if binding_supported else "no",
        "upstream_satisfied": "yes" if path_fact_ids or operator_index == 0 else "no",
    }


def feature_keys(features: Mapping[str, str], *, coarse_only: bool = False) -> Sequence[Tuple[str, str]]:
    event = str(features.get("event") or "unknown")
    source = str(features.get("source_type") or "none")
    if coarse_only:
        return (
            ("source_type", "|".join((event, source))),
            ("global", event),
        )
    exact_names = (
        "event", "source_type", "query_shape", "predicate", "subtype",
        "stage", "title_match", "relevance", "reuse", "binding_supported",
        "upstream_satisfied",
    )
    exact = "|".join(str(features.get(name) or "unknown") for name in exact_names)
    return (
        ("exact", exact),
        ("source_predicate_stage", "|".join((
            event, source, str(features.get("predicate") or "unknown"),
            str(features.get("stage") or "unknown"),
        ))),
        ("source_predicate", "|".join((
            event, source, str(features.get("predicate") or "unknown"),
        ))),
        ("source_type", "|".join((event, source))),
        ("global", event),
    )


class HierarchicalSourceValueModel:
    """Frozen held-out calibrated probability tables with explicit backoff."""

    version = "bswc_hierarchical_beta_binomial_v1"

    def __init__(self, payload: Mapping[str, Any] | None = None, *, min_samples: int = 8):
        self.payload = dict(payload or {})
        self.min_samples = max(1, int(min_samples))
        self.loaded = bool(self.payload.get("models"))
        self.source_hash = str(self.payload.get("source_hash") or "")
        self.folds = max(2, _int(self.payload.get("folds"), 5))

    @classmethod
    def from_path(cls, path: str, *, min_samples: int = 8) -> "HierarchicalSourceValueModel":
        model_path = Path(str(path or ""))
        if not path or not model_path.is_file():
            return cls(min_samples=min_samples)
        try:
            return cls(json.loads(model_path.read_text(encoding="utf-8")), min_samples=min_samples)
        except (OSError, json.JSONDecodeError, TypeError):
            return cls(min_samples=min_samples)

    def _table(self, sample_key: str) -> Tuple[Mapping[str, Any], str]:
        models = self.payload.get("models") or {}
        dataset = str(sample_key).split(":", 1)[0]
        if dataset == "2wikimultihopqa":
            name = f"exclude_fold_{stable_fold(sample_key, self.folds)}"
            if name in models:
                return dict(models[name] or {}), name
        return dict(models.get("full") or {}), "full"

    def predict(
        self,
        features: Mapping[str, str],
        *,
        sample_key: str,
        default: float,
        coarse_only: bool = False,
    ) -> Dict[str, Any]:
        table, split_name = self._table(sample_key)
        levels = table.get("levels") or {}
        fallback_row = None
        for level, key in feature_keys(features, coarse_only=coarse_only):
            row = dict((levels.get(level) or {}).get(key) or {})
            if not row:
                continue
            fallback_row = (level, row)
            if _int(row.get("count")) >= self.min_samples or level == "global":
                probability = _bounded(row.get("probability"), default)
                return {
                    "probability": probability,
                    "sample_count": _int(row.get("count")),
                    "success_count": _int(row.get("success")),
                    "backoff_level": level,
                    "uncertainty": _float(row.get("uncertainty"), 0.5),
                    "covered": True,
                    "model_split": split_name,
                    "features": dict(features),
                }
        if fallback_row is not None:
            level, row = fallback_row
            return {
                "probability": _bounded(row.get("probability"), default),
                "sample_count": _int(row.get("count")),
                "success_count": _int(row.get("success")),
                "backoff_level": f"sparse_{level}",
                "uncertainty": _float(row.get("uncertainty"), 0.5),
                "covered": True,
                "model_split": split_name,
                "features": dict(features),
            }
        return {
            "probability": _bounded(default),
            "sample_count": 0,
            "success_count": 0,
            "backoff_level": "static_prior",
            "uncertainty": 1.0,
            "covered": False,
            "model_split": split_name,
            "features": dict(features),
        }


def feedback_key(features: Mapping[str, str]) -> str:
    return "|".join((
        str(features.get("event") or "unknown"),
        str(features.get("source_type") or "none"),
        str(features.get("predicate") or "unknown"),
    ))


def feedback_adjusted_probability(
    prior: float,
    features: Mapping[str, str],
    state: Mapping[str, Any],
    *,
    strength: float = 3.0,
) -> Dict[str, Any]:
    stats = dict((state.get("bswc_feedback_stats") or {}).get(feedback_key(features)) or {})
    successes = _int(stats.get("success"))
    failures = _int(stats.get("failure"))
    count = successes + failures
    posterior = (max(0.1, strength) * _bounded(prior) + successes) / (
        max(0.1, strength) + count
    )
    return {
        "probability": _bounded(posterior),
        "prior_probability": _bounded(prior),
        "query_feedback_count": count,
        "query_feedback_success": successes,
        "query_feedback_failure": failures,
        "query_feedback_key": feedback_key(features),
    }


def update_query_feedback(
    state: MutableMapping[str, Any],
    *,
    features: Mapping[str, str],
    success: bool,
    action_key: str,
    probability_before: float,
) -> Dict[str, Any]:
    key = feedback_key(features)
    table = state.setdefault("bswc_feedback_stats", {})
    row = table.setdefault(key, {"success": 0, "failure": 0})
    row["success" if success else "failure"] = _int(
        row.get("success" if success else "failure")
    ) + 1
    updated = feedback_adjusted_probability(probability_before, features, state)
    event = {
        "action_key": str(action_key),
        "feedback_key": key,
        "success": bool(success),
        "probability_before": round(_bounded(probability_before), 8),
        "probability_after": round(_bounded(updated["probability"]), 8),
        "success_count": _int(row.get("success")),
        "failure_count": _int(row.get("failure")),
    }
    state.setdefault("bswc_feedback_events", []).append(event)
    return event


def source_action_success(operator: str, transition: Mapping[str, Any]) -> bool:
    if str(operator) == "BOUND_DENSE_PROBE":
        return _int(transition.get("source_candidate_count_added")) > 0
    if str(operator) == "SOURCE_INSPECT":
        return bool(transition.get("created_binding_ids"))
    if str(operator) == "VERIFY":
        return str(transition.get("support_status") or "") == "SUPPORTED"
    return bool(transition.get("logical_progress"))


def should_stop_low_value(
    actions: Sequence[Mapping[str, Any]],
    estimates: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    *,
    threshold: float,
    min_exploration: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Stop only recovery actions after preserving a minimum exploration path."""
    recovery = [
        action for action in actions
        if str(action.get("operator") or "") in {"SOURCE_INSPECT", "BOUND_DENSE_PROBE"}
    ]
    if not recovery or len(recovery) != len(actions):
        return False, {"reason": "non_recovery_action_available"}
    inspections = _int(state.get("bswc_source_inspection_count"))
    probes = _int(state.get("bswc_dense_probe_count"))
    if inspections + probes < max(0, int(min_exploration)):
        return False, {"reason": "exploration_floor", "exploration_count": inspections + probes}
    rows = []
    for action in recovery:
        estimate = estimates.get(str(action.get("action_key") or ""), {})
        witness = _bounded(
            estimate.get("expected_witness_progress", estimate.get("support_probability")),
            0.0,
        )
        cost = max(
            0.05,
            _float(estimate.get("expected_scan_rows"))
            + 8.0 * _float(estimate.get("expected_retrieval_calls"))
            + 12.0 * _float(estimate.get("expected_model_calls")),
        )
        rows.append({
            "action_key": str(action.get("action_key") or ""),
            "expected_witness_progress": witness,
            "expected_cost": cost,
            "expected_gain_per_cost": witness / cost,
        })
    best = max(rows, key=lambda row: row["expected_gain_per_cost"])
    stop = best["expected_gain_per_cost"] < max(0.0, float(threshold))
    return stop, {
        "reason": "max_expected_witness_gain_below_threshold" if stop else "value_above_threshold",
        "threshold": float(threshold),
        "best": best,
        "candidate_count": len(rows),
    }


def beta_uncertainty(probability: float, count: int) -> float:
    probability = _bounded(probability)
    return math.sqrt(probability * (1.0 - probability) / max(1, count + 2))
