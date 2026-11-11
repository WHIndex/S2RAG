"""Calibrated risk estimates for stopping a lazy evidence query plan.

The adaptive executor already estimates the risk of running an operator.  This
module models the complementary decision: whether stopping now, and therefore
skipping the remaining eager incumbent, may change reader-visible evidence or
degrade answer quality.  Features are restricted to state available before an
operator is invoked and deliberately exclude dataset identifiers.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


MODEL_VERSION = 1
FEATURE_SCHEMA = "adaptive_reader_skip_risk_v1"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, _as_float(value, low)))


def _bucket(value: Any, cuts: Sequence[float]) -> str:
    number = _as_float(value, 0.0)
    for index, cut in enumerate(cuts):
        if number < float(cut):
            return f"b{index}"
    return f"b{len(cuts)}"


def _operator_names(operators: Iterable[Mapping[str, Any]]) -> List[str]:
    return sorted(
        str(operator.get("operator") or "")
        for operator in operators
        if str(operator.get("operator") or "")
    )


def skip_risk_feature_names(
    context: Mapping[str, Any],
    state: Mapping[str, Any],
    remaining_operators: Sequence[Mapping[str, Any]],
    executed_operators: Sequence[str],
) -> List[str]:
    """Return stable, pre-execution categorical features for a stop decision."""
    remaining = "+".join(_operator_names(remaining_operators)) or "none"
    executed = "+".join(sorted(str(value) for value in executed_operators)) or "none"
    coverage = _as_float(
        state.get("query_coverage"),
        _as_float(context.get("query_coverage"), 0.0),
    )
    schema_scan = _as_int(context.get("schema_estimated_scan_rows"))
    relation_scan = _as_int(context.get("relation_estimated_scan_rows"))
    schema_capacity = max(1, _as_int(context.get("schema_capacity"), 1))
    relation_capacity = max(1, _as_int(context.get("relation_capacity"), 1))
    schema_selectivity = schema_scan / float(schema_capacity)
    relation_selectivity = relation_scan / float(relation_capacity)
    residual = bool(state.get("residual_deficit"))
    bridge = bool(context.get("bridge_deficit"))
    features = [
        f"remaining={remaining}",
        f"executed={executed}",
        f"executed_count={min(2, len(executed_operators))}",
        f"residual_deficit={int(residual)}",
        f"bridge_deficit={int(bridge)}",
        f"coverage={_bucket(coverage, (0.35, 0.50, 0.65, 0.75, 0.85, 0.95))}",
        f"question_tokens={_bucket(context.get('question_token_count'), (6, 10, 15, 22, 32))}",
        f"base_doc_count={_bucket(context.get('base_doc_count'), (2, 4, 6, 9))}",
        f"avg_doc_chars={_bucket(context.get('avg_base_doc_chars'), (400, 800, 1200, 1800, 2600))}",
        f"bridge_strength={_bucket(context.get('bridge_strength'), (1, 2, 3, 5, 8))}",
        f"active_schema_count={_bucket(context.get('active_schema_count'), (1, 2, 4, 8, 16))}",
        f"schema_scan={_bucket(schema_scan, (1, 8, 24, 64, 160, 400))}",
        f"schema_selectivity={_bucket(schema_selectivity, (0.05, 0.15, 0.35, 0.60, 0.85))}",
        f"relation_probe_count={_bucket(context.get('relation_probe_count'), (1, 3, 6, 12, 24))}",
        f"relation_scan={_bucket(relation_scan, (1, 16, 64, 160, 360, 800))}",
        f"relation_selectivity={_bucket(relation_selectivity, (0.05, 0.15, 0.35, 0.60, 0.85))}",
        f"remaining_residual={remaining}|{int(residual)}",
        f"remaining_coverage={remaining}|{_bucket(coverage, (0.50, 0.65, 0.75, 0.85))}",
        f"scan_pair={_bucket(schema_scan, (1, 24, 96, 320))}|{_bucket(relation_scan, (1, 64, 240, 700))}",
    ]
    return features


def empty_skip_risk_model(reason: str) -> Dict[str, Any]:
    return {
        "version": MODEL_VERSION,
        "feature_schema": FEATURE_SCHEMA,
        "enabled": False,
        "reason": reason,
        "global": {"n": 0, "reader_changed": 0, "harm": 0, "delta_f1_sum": 0.0},
        "features": {},
    }


@lru_cache(maxsize=8)
def load_adaptive_reader_risk_model(model_path: str) -> Dict[str, Any]:
    if not model_path:
        return empty_skip_risk_model("missing_model_path")
    expanded = os.path.abspath(os.path.expanduser(str(model_path)))
    if not os.path.exists(expanded):
        return empty_skip_risk_model("model_file_not_found")
    try:
        with open(expanded, "r", encoding="utf-8") as handle:
            model = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        result = empty_skip_risk_model("model_load_failed")
        result["error"] = str(exc)[:200]
        return result
    if not isinstance(model, dict):
        return empty_skip_risk_model("model_not_dict")
    if str(model.get("feature_schema") or "") != FEATURE_SCHEMA:
        return empty_skip_risk_model("feature_schema_mismatch")
    model["enabled"] = bool(model.get("features")) and _as_int(
        (model.get("global") or {}).get("n")
    ) > 0
    model["model_path"] = expanded
    return model


def _posterior_rate(
    positives: float,
    total: float,
    global_rate: float,
    *,
    prior_strength: float = 20.0,
) -> float:
    return _clamp(
        (max(0.0, positives) + prior_strength * _clamp(global_rate))
        / max(1.0, max(0.0, total) + prior_strength)
    )


def _wilson_upper(rate: float, total: float, z: float = 1.96) -> float:
    n = max(1.0, float(total))
    p = _clamp(rate)
    z2 = z * z
    center = p + z2 / (2.0 * n)
    spread = z * math.sqrt(max(0.0, p * (1.0 - p) / n + z2 / (4.0 * n * n)))
    return _clamp((center + spread) / (1.0 + z2 / n))


def assess_adaptive_stop_risk(
    *,
    context: Mapping[str, Any],
    state: Mapping[str, Any],
    remaining_operators: Sequence[Mapping[str, Any]],
    executed_operators: Sequence[str],
    model: Optional[Mapping[str, Any]],
    min_support: int = 25,
    min_confidence: float = 0.25,
) -> Dict[str, Any]:
    """Estimate risk of stopping instead of executing the eager incumbent."""
    features = skip_risk_feature_names(
        context,
        state,
        remaining_operators,
        executed_operators,
    )
    if not model or not bool(model.get("enabled")):
        return {
            "enabled": False,
            "supported": False,
            "reason": str((model or {}).get("reason") or "model_unavailable"),
            "features": features,
            "p_reader_change": 0.0,
            "p_harm": 0.0,
            "p_harm_upper": 0.0,
            "confidence": 0.0,
        }

    global_stat = dict(model.get("global") or {})
    global_n = max(1.0, _as_float(global_stat.get("n"), 1.0))
    global_change = _as_float(global_stat.get("reader_changed")) / global_n
    global_harm = _as_float(global_stat.get("harm")) / global_n
    feature_stats = model.get("features") or {}
    matched = []
    for feature in features:
        stat = feature_stats.get(feature)
        if not isinstance(stat, dict):
            continue
        support = _as_float(stat.get("n"))
        if support < max(1, int(min_support)):
            continue
        change_rate = _posterior_rate(
            _as_float(stat.get("reader_changed")),
            support,
            global_change,
        )
        harm_rate = _posterior_rate(
            _as_float(stat.get("harm")),
            support,
            global_harm,
        )
        matched.append({
            "feature": feature,
            "support": support,
            "p_reader_change": change_rate,
            "p_harm": harm_rate,
            "p_harm_upper": _wilson_upper(harm_rate, support),
        })

    if not matched:
        return {
            "enabled": True,
            "supported": False,
            "reason": "no_supported_features",
            "features": features,
            "p_reader_change": round(global_change, 6),
            "p_harm": round(global_harm, 6),
            "p_harm_upper": round(_wilson_upper(global_harm, global_n), 6),
            "confidence": 0.0,
            "matched_features": [],
        }



    change_row = max(matched, key=lambda row: (row["p_reader_change"], row["support"]))
    harm_row = max(matched, key=lambda row: (row["p_harm_upper"], row["support"]))
    max_support = max(row["support"] for row in matched)
    confidence = _clamp(max_support / (max_support + max(1.0, float(min_support))))
    supported = confidence >= float(min_confidence)
    ranked = sorted(
        matched,
        key=lambda row: (row["p_harm_upper"], row["p_reader_change"], row["support"]),
        reverse=True,
    )
    return {
        "enabled": True,
        "supported": supported,
        "reason": "calibrated" if supported else "low_confidence",
        "p_reader_change": round(float(change_row["p_reader_change"]), 6),
        "p_harm": round(float(harm_row["p_harm"]), 6),
        "p_harm_upper": round(float(harm_row["p_harm_upper"]), 6),
        "confidence": round(confidence, 6),
        "matched_feature_count": len(matched),
        "matched_features": [
            {
                "feature": row["feature"],
                "support": int(row["support"]),
                "p_reader_change": round(float(row["p_reader_change"]), 6),
                "p_harm": round(float(row["p_harm"]), 6),
                "p_harm_upper": round(float(row["p_harm_upper"]), 6),
            }
            for row in ranked[:6]
        ],
    }
