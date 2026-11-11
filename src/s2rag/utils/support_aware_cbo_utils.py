"""Calibrated action estimates for support-aware query execution.

This module has no QA-specific features.  It consumes aggregate execution
statistics and exposes estimates for an already-legal executor action.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def scan_bucket(value: Any) -> str:
    rows = max(0.0, _number(value))
    if rows <= 0:
        return "0"
    if rows <= 4:
        return "1_4"
    if rows <= 16:
        return "5_16"
    if rows <= 64:
        return "17_64"
    return "65_plus"


def depth_bucket(value: Any) -> str:
    depth = max(0, int(_number(value)))
    if depth <= 0:
        return "0"
    if depth == 1:
        return "1"
    if depth == 2:
        return "2"
    return "3_plus"


def action_features(
    action: Mapping[str, Any],
    state: Mapping[str, Any],
    base_estimate: Mapping[str, Any],
) -> Dict[str, str]:
    operator = str(action.get("operator") or "UNKNOWN")
    plan = dict(state.get("logical_plan") or {})
    operators = list(plan.get("operators") or [])
    index = int(_number(action.get("operator_index"), -1))
    origin = "na"
    if operator == "VERIFY":
        fact = dict((state.get("facts") or {}).get(action.get("fact_id")) or {})
        origin = str(fact.get("origin") or "unknown")
        if index < 0:
            index = int(_number(fact.get("producer_operator_index"), -1))
    remaining = max(0, len(operators) - max(0, index) - 1) if operators else 0
    return {
        "operator": operator,
        "query_shape": str(plan.get("plan_kind") or "unknown"),
        "origin": origin,
        "depth_bucket": depth_bucket(remaining),
        "scan_bucket": scan_bucket(base_estimate.get("expected_scan_rows")),
    }


def feature_keys(features: Mapping[str, str]) -> Tuple[str, str, str]:
    operator = features["operator"]
    shape = features["query_shape"]
    exact = "|".join((
        operator,
        shape,
        features["origin"],
        features["depth_bucket"],
        features["scan_bucket"],
    ))
    return operator, f"{operator}|{shape}", exact


class SampledActionCostModel:
    """Hierarchical sample-based estimates with conservative fallback."""

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        min_samples: int = 8,
    ) -> None:
        self.payload = dict(payload or {})
        self.min_samples = max(1, int(min_samples))
        self.levels = dict(self.payload.get("levels") or {})
        self.version = str(self.payload.get("version") or "unknown")
        self.source_hash = str(self.payload.get("source_trace_hash") or "")

    @classmethod
    def from_path(
        cls, path: str | Path | None, *, min_samples: int = 8
    ) -> Optional["SampledActionCostModel"]:
        if not path:
            return None
        model_path = Path(path)
        if not model_path.exists():
            return None
        with model_path.open(encoding="utf-8") as handle:
            return cls(json.load(handle), min_samples=min_samples)

    def _lookup(self, features: Mapping[str, str]) -> tuple[dict[str, Any], str]:
        operator, operator_shape, exact = feature_keys(features)
        choices = (
            ("exact", exact),
            ("operator_shape", operator_shape),
            ("operator", operator),
        )
        for level, key in choices:
            row = dict((self.levels.get(level) or {}).get(key) or {})
            if int(row.get("sample_count") or 0) >= self.min_samples:
                return row, level


        row = dict((self.levels.get("operator") or {}).get(operator) or {})
        return row, "operator_low_support" if row else "no_calibration"

    def estimate(
        self,
        action: Mapping[str, Any],
        state: Mapping[str, Any],
        base_estimate: Mapping[str, Any],
    ) -> Dict[str, Any]:
        base = dict(base_estimate or {})
        features = action_features(action, state, base)
        row, level = self._lookup(features)
        if not row:
            return {
                **base,
                "cbo_calibrated": False,
                "cbo_fallback_reason": "no_matching_statistics",
                "cbo_features": features,
            }
        sample_count = int(row.get("sample_count") or 0)
        confidence = min(1.0, sample_count / 32.0)

        def blend(name: str, base_name: str, default: float = 0.0) -> float:
            observed = _number(row.get(name), default)
            prior = _number(base.get(base_name), default)
            return confidence * observed + (1.0 - confidence) * prior

        output = dict(base)
        output.update({
            "expected_progress": blend(
                "progress_probability", "expected_progress", 0.5
            ),
            "support_probability": blend(
                "support_probability", "support_probability", 0.5
            ),
            "expected_scan_rows": blend(
                "mean_scan_rows", "expected_scan_rows", 0.0
            ),
            "expected_retrieval_calls": blend(
                "mean_retrieval_calls", "expected_retrieval_calls", 0.0
            ),
            "expected_model_calls": blend(
                "mean_model_calls", "expected_model_calls", 0.0
            ),
            "expected_tokens": blend(
                "mean_tokens", "expected_tokens", 0.0
            ),
            "expected_wall_ms": _number(row.get("mean_wall_ms")),
            "expected_downstream_scan_rows": _number(
                row.get("mean_downstream_scan_rows")
            ),
            "expected_downstream_calls": _number(
                row.get("mean_downstream_calls")
            ),
            "estimate_source": "sampled_support_aware_cost_model",
            "cbo_calibrated": True,
            "cbo_model_version": self.version,
            "cbo_model_source_hash": self.source_hash,
            "cbo_backoff_level": level,
            "cbo_sample_count": sample_count,
            "cbo_confidence": round(confidence, 6),
            "cbo_features": features,
        })
        return output


def support_aware_action_score(
    action: Mapping[str, Any],
    estimate: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    risk_weight: float = 0.8,
    cost_weight: float = 0.25,
) -> tuple[float, Dict[str, float]]:
    """Score one legal action without weakening witness safety semantics."""
    operator = str(action.get("operator") or "")
    progress = min(1.0, max(0.0, _number(estimate.get("expected_progress"), 0.0)))
    support = min(1.0, max(0.0, _number(estimate.get("support_probability"), 0.5)))
    unlock = max(0.0, _number(estimate.get("unlock_value"), progress))
    pruning = min(1.0, max(0.0, _number(estimate.get("expected_pruning"), 0.0)))
    scan = max(0.0, _number(estimate.get("expected_scan_rows")))
    retrieval = max(0.0, _number(estimate.get("expected_retrieval_calls")))
    model = max(0.0, _number(estimate.get("expected_model_calls")))
    tokens = max(0.0, _number(estimate.get("expected_tokens")))
    wall_ms = max(0.0, _number(estimate.get("expected_wall_ms")))
    downstream_scan = max(
        0.0,
        _number(estimate.get("expected_downstream_scan_rows")),
    )
    downstream_calls = max(
        0.0,
        _number(estimate.get("expected_downstream_calls")),
    )
    cost_units = (
        scan / 32.0
        + 3.0 * retrieval
        + 6.0 * model
        + tokens / 1500.0
        + wall_ms / 100.0
    )
    downstream_units = downstream_scan / 32.0 + 3.0 * downstream_calls
    expected_gain = progress * max(0.25, unlock)
    risk_penalty = (1.0 - support) * risk_weight
    avoided_waste = 0.0
    if operator == "VERIFY":

        avoided_waste = (1.0 - support) * downstream_units
        expected_gain = 0.5 * progress + avoided_waste
        risk_penalty = 0.0
    elif operator == "LOOKUP_JOIN":
        expected_gain += 0.35 * pruning
    elif operator == "COMPLETE":


        expected_gain *= support
    utility = expected_gain - risk_penalty - cost_weight * cost_units
    components = {
        "expected_gain": round(expected_gain, 8),
        "support_probability": round(support, 8),
        "risk_penalty": round(risk_penalty, 8),
        "cost_units": round(cost_units, 8),
        "avoided_downstream_waste": round(avoided_waste, 8),
        "risk_weight": round(float(risk_weight), 8),
        "cost_weight": round(float(cost_weight), 8),
    }
    return round(utility, 8), components
