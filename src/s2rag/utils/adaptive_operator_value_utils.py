"""Output-aware value estimates for adaptive evidence operators.

The model intentionally uses only state that is available before an operator
is invoked.  It predicts whether an index access will return candidates and
whether those candidates will resolve the current evidence deficit.  Version
2 uses the terminal-resolution estimate as a one-sided admission guard.
Version 3 calibrates that guard on the optimizer's actual decision frontier
and applies it only to the physical operator that would otherwise execute.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence


MODEL_VERSION = 3
FEATURE_SCHEMA = "adaptive_operator_terminal_value_v3"
LEGACY_FEATURE_SCHEMAS = {
    "adaptive_operator_terminal_value_v2",
    "adaptive_operator_output_value_v1",
}
TERMINAL_POLICY_MODE = "terminal_net_benefit_v2"
DECISION_CONDITIONAL_POLICY_MODE = (
    "decision_conditional_terminal_net_benefit_v3"
)
TERMINAL_POLICY_MODES = {
    TERMINAL_POLICY_MODE,
    DECISION_CONDITIONAL_POLICY_MODE,
}


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


def operator_value_feature_names(
    context: Mapping[str, Any],
    state: Mapping[str, Any],
    operator: Mapping[str, Any],
) -> List[str]:
    """Build stable categorical features without dataset identifiers."""
    name = str(operator.get("operator") or "UNKNOWN")
    coverage = _as_float(
        state.get("query_coverage"), context.get("query_coverage", 0.0)
    )
    residual = int(bool(state.get("residual_deficit")))
    scan_rows = _as_int(operator.get("estimated_scan_rows"))
    if name == "SCHEMA_PROBE":
        capacity = max(1, _as_int(context.get("schema_capacity"), 1))
        active_count = _as_int(context.get("active_schema_count"))
    else:
        capacity = max(1, _as_int(context.get("relation_capacity"), 1))
        active_count = _as_int(context.get("relation_probe_count"))
    selectivity = scan_rows / float(capacity)
    coverage_bucket = _bucket(coverage, (0.35, 0.50, 0.65, 0.75, 0.85, 0.95))
    scan_bucket = _bucket(scan_rows, (1, 8, 24, 64, 160, 400, 800))
    selectivity_bucket = _bucket(selectivity, (0.05, 0.15, 0.35, 0.60, 0.85))
    active_bucket = _bucket(active_count, (1, 2, 4, 8, 16, 32))
    bridge_bucket = _bucket(context.get("bridge_strength"), (1, 2, 3, 5, 8))
    question_bucket = _bucket(
        context.get("question_token_count"), (6, 10, 15, 22, 32)
    )
    return [
        f"operator={name}|residual={residual}",
        f"operator={name}|coverage={coverage_bucket}",
        f"operator={name}|scan={scan_bucket}",
        f"operator={name}|selectivity={selectivity_bucket}",
        f"operator={name}|active={active_bucket}",
        f"operator={name}|bridge={bridge_bucket}",
        f"operator={name}|question_tokens={question_bucket}",
        (
            f"operator={name}|residual={residual}|coverage={coverage_bucket}"
            f"|scan={scan_bucket}"
        ),
    ]


def empty_operator_value_model(reason: str) -> Dict[str, Any]:
    return {
        "version": MODEL_VERSION,
        "feature_schema": FEATURE_SCHEMA,
        "enabled": False,
        "reason": reason,
        "global": {"n": 0, "nonempty": 0, "resolved": 0},
        "operators": {},
        "features": {},
    }


@lru_cache(maxsize=8)
def load_adaptive_operator_value_model(model_path: str) -> Dict[str, Any]:
    if not model_path:
        return empty_operator_value_model("missing_model_path")
    expanded = os.path.abspath(os.path.expanduser(str(model_path)))
    if not os.path.exists(expanded):
        return empty_operator_value_model("model_file_not_found")
    try:
        with open(expanded, "r", encoding="utf-8") as handle:
            model = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        result = empty_operator_value_model("model_load_failed")
        result["error"] = str(exc)[:200]
        return result
    if not isinstance(model, dict):
        return empty_operator_value_model("model_not_dict")
    if str(model.get("feature_schema") or "") not in (
        {FEATURE_SCHEMA} | LEGACY_FEATURE_SCHEMAS
    ):
        return empty_operator_value_model("feature_schema_mismatch")
    model["enabled"] = bool(model.get("operators")) and _as_int(
        (model.get("global") or {}).get("n")
    ) > 0
    model["model_path"] = expanded
    return model


def _posterior_rate(
    positives: float,
    total: float,
    prior_rate: float,
    *,
    prior_strength: float,
) -> float:
    return _clamp(
        (max(0.0, positives) + max(0.0, prior_strength) * _clamp(prior_rate))
        / max(1.0, max(0.0, total) + max(0.0, prior_strength))
    )


def _productive_probability(
    p_nonempty: float,
    p_resolve: float,
    resolution_floor: float,
) -> float:


    floor = _clamp(resolution_floor)
    return _clamp(p_nonempty) * (floor + (1.0 - floor) * _clamp(p_resolve))


def _wilson_upper(positives: float, total: float, z: float = 1.96) -> float:
    if total <= 0:
        return 1.0
    rate = _clamp(positives / total)
    denominator = 1.0 + z * z / total
    center = rate + z * z / (2.0 * total)
    radius = z * math.sqrt(
        max(0.0, rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    )
    return _clamp((center + radius) / denominator)


def assess_counterfactual_operator_harm(
    *,
    context: Mapping[str, Any],
    state: Mapping[str, Any],
    operator: Mapping[str, Any],
    model: Optional[Mapping[str, Any]],
    min_operator_support: int = 25,
    min_feature_support: int = 20,
    min_confidence: float = 0.25,
) -> Dict[str, Any]:
    """Estimate answer-level benefit and harm from paired executions."""
    name = str(operator.get("operator") or "UNKNOWN")
    features = operator_value_feature_names(context, state, operator)
    unavailable = {
        "counterfactual_supported": False,
        "counterfactual_reason": "counterfactual_model_unavailable",
        "counterfactual_support": 0,
        "p_answer_improve": 0.0,
        "p_answer_harm": 0.0,
        "p_answer_harm_upper": 1.0,
        "expected_quality_delta": 0.0,
        "counterfactual_utility_multiplier": 1.0,
        "counterfactual_matched_features": [],
    }
    counterfactual = dict((model or {}).get("counterfactual") or {})
    if not bool(counterfactual.get("enabled")):
        unavailable["counterfactual_reason"] = str(
            counterfactual.get("reason") or "counterfactual_model_unavailable"
        )
        return unavailable

    global_stat = dict(counterfactual.get("global") or {})
    global_n = max(1.0, _as_float(global_stat.get("n"), 1.0))
    global_harm = _as_float(global_stat.get("harm")) / global_n
    global_improve = _as_float(global_stat.get("improve")) / global_n
    global_delta = _as_float(global_stat.get("quality_delta_sum")) / global_n
    operator_stat = dict(
        (counterfactual.get("operators") or {}).get(name) or {}
    )
    operator_n = _as_float(operator_stat.get("n"))
    if operator_n <= 0:
        unavailable.update({
            "counterfactual_reason": "counterfactual_operator_unseen",
            "p_answer_improve": round(global_improve, 6),
            "p_answer_harm": round(global_harm, 6),
        })
        return unavailable

    operator_harm = _posterior_rate(
        _as_float(operator_stat.get("harm")),
        operator_n,
        global_harm,
        prior_strength=12.0,
    )
    operator_improve = _posterior_rate(
        _as_float(operator_stat.get("improve")),
        operator_n,
        global_improve,
        prior_strength=12.0,
    )
    operator_delta = (
        _as_float(operator_stat.get("quality_delta_sum"))
        + 12.0 * global_delta
    ) / max(1.0, operator_n + 12.0)
    matched: List[Dict[str, Any]] = []
    for feature in features:
        stat = (counterfactual.get("features") or {}).get(feature)
        if not isinstance(stat, dict):
            continue
        support = _as_float(stat.get("n"))
        if support < max(1, int(min_feature_support)):
            continue
        matched.append({
            "feature": feature,
            "support": support,
            "p_harm": _posterior_rate(
                _as_float(stat.get("harm")),
                support,
                operator_harm,
                prior_strength=8.0,
            ),
            "p_improve": _posterior_rate(
                _as_float(stat.get("improve")),
                support,
                operator_improve,
                prior_strength=8.0,
            ),
            "quality_delta": (
                _as_float(stat.get("quality_delta_sum"))
                + 8.0 * operator_delta
            ) / max(1.0, support + 8.0),
            "harm_upper": _wilson_upper(
                _as_float(stat.get("harm")), support
            ),
        })

    base_weight = math.sqrt(max(1.0, operator_n))
    total_weight = base_weight
    harm_sum = base_weight * operator_harm
    improve_sum = base_weight * operator_improve
    delta_sum = base_weight * operator_delta
    for row in matched:
        weight = math.sqrt(max(1.0, row["support"]))
        total_weight += weight
        harm_sum += weight * row["p_harm"]
        improve_sum += weight * row["p_improve"]
        delta_sum += weight * row["quality_delta"]
    p_harm = _clamp(harm_sum / max(1e-9, total_weight))
    p_improve = _clamp(improve_sum / max(1e-9, total_weight))
    expected_delta = max(-1.0, min(1.0, delta_sum / max(1e-9, total_weight)))
    p_harm_upper = max(
        _wilson_upper(_as_float(operator_stat.get("harm")), operator_n),
        max((float(row["harm_upper"]) for row in matched), default=0.0),
    )
    confidence = _clamp(
        operator_n / (operator_n + max(1.0, float(min_operator_support)))
    )
    supported = bool(
        operator_n >= max(1, int(min_operator_support))
        and confidence >= float(min_confidence)
    )
    multiplier = max(0.25, min(1.50, 1.0 + 4.0 * expected_delta))
    return {
        "counterfactual_supported": supported,
        "counterfactual_reason": (
            "paired_answer_calibrated" if supported else "low_counterfactual_support"
        ),
        "counterfactual_support": int(operator_n),
        "p_answer_improve": round(p_improve, 6),
        "p_answer_harm": round(p_harm, 6),
        "p_answer_harm_upper": round(p_harm_upper, 6),
        "expected_quality_delta": round(expected_delta, 6),
        "counterfactual_utility_multiplier": round(multiplier, 6),
        "counterfactual_confidence": round(confidence, 6),
        "counterfactual_matched_features": [
            {
                "feature": row["feature"],
                "support": int(row["support"]),
                "p_harm": round(float(row["p_harm"]), 6),
                "p_improve": round(float(row["p_improve"]), 6),
                "harm_upper": round(float(row["harm_upper"]), 6),
            }
            for row in sorted(
                matched, key=lambda item: item["support"], reverse=True
            )[:8]
        ],
    }


def terminal_net_benefit(
    *,
    context: Mapping[str, Any],
    operator: Mapping[str, Any],
    p_terminal: float,
    policy: Mapping[str, Any],
) -> Dict[str, float]:
    """Compute the deployable terminal value of an index access.

    All inputs are available before execution.  The calibrated threshold is
    learned separately, so these weights define a stable ordering rather than
    an uncalibrated zero boundary.
    """
    name = str(operator.get("operator") or "UNKNOWN")
    if name == "SCHEMA_PROBE":
        capacity = max(1, _as_int(context.get("schema_capacity"), 1))
    else:
        capacity = max(1, _as_int(context.get("relation_capacity"), 1))
    scan_rows = max(0, _as_int(operator.get("estimated_scan_rows")))
    scan_ratio = _clamp(
        math.log1p(scan_rows) / max(1e-9, math.log1p(capacity))
    )
    estimated_utility = _clamp(operator.get("estimated_utility"), 0.0, 1.0)
    compute_cost = _clamp(operator.get("estimated_compute_cost"), 0.0, 1.0)
    predicted_risk = _clamp(operator.get("predicted_risk"), 0.0, 1.0)
    reward = max(0.0, _as_float(policy.get("terminal_reward"), 1.0))
    compute_weight = max(0.0, _as_float(policy.get("compute_weight"), 0.015))
    scan_weight = max(0.0, _as_float(policy.get("scan_weight"), 0.010))
    risk_weight = max(0.0, _as_float(policy.get("risk_weight"), 0.010))
    expected_benefit = reward * _clamp(p_terminal) * estimated_utility
    expected_cost = (
        compute_weight * compute_cost
        + scan_weight * scan_ratio
        + risk_weight * predicted_risk
    )
    return {
        "expected_terminal_benefit": expected_benefit,
        "expected_access_cost": expected_cost,
        "terminal_net_benefit": expected_benefit - expected_cost,
        "normalized_scan_cost": scan_ratio,
    }


def assess_adaptive_operator_value(
    *,
    context: Mapping[str, Any],
    state: Mapping[str, Any],
    operator: Mapping[str, Any],
    model: Optional[Mapping[str, Any]],
    min_operator_support: int = 25,
    min_feature_support: int = 20,
    min_confidence: float = 0.25,
    resolution_floor: float = 0.50,
    min_utility_multiplier: float = 0.35,
    max_utility_multiplier: float = 1.50,
) -> Dict[str, Any]:
    """Estimate output-aware utility for one pre-execution alternative."""
    features = operator_value_feature_names(context, state, operator)
    name = str(operator.get("operator") or "UNKNOWN")
    unavailable = {
        "enabled": False,
        "supported": False,
        "reason": str((model or {}).get("reason") or "model_unavailable"),
        "operator": name,
        "features": features,
        "p_nonempty": 0.0,
        "p_resolve": 0.0,
        "expected_productivity": 0.0,
        "reference_productivity": 0.0,
        "utility_multiplier": 1.0,
        "confidence": 0.0,
        "matched_features": [],
    }
    if not model or not bool(model.get("enabled")):
        return unavailable

    global_stat = dict(model.get("global") or {})
    global_n = max(1.0, _as_float(global_stat.get("n"), 1.0))
    global_nonempty = _as_float(global_stat.get("nonempty")) / global_n
    global_resolve = _as_float(
        global_stat.get("terminal", global_stat.get("resolved"))
    ) / global_n
    operator_stat = dict((model.get("operators") or {}).get(name) or {})
    operator_n = _as_float(operator_stat.get("n"))
    if operator_n <= 0:
        unavailable.update({
            "enabled": True,
            "reason": "operator_unseen",
            "p_nonempty": round(global_nonempty, 6),
            "p_resolve": round(global_resolve, 6),
        })
        return unavailable

    operator_nonempty = _posterior_rate(
        _as_float(operator_stat.get("nonempty")),
        operator_n,
        global_nonempty,
        prior_strength=20.0,
    )
    operator_resolve = _posterior_rate(
        _as_float(operator_stat.get("terminal", operator_stat.get("resolved"))),
        operator_n,
        global_resolve,
        prior_strength=20.0,
    )
    matched: List[Dict[str, Any]] = []
    for feature in features:
        stat = (model.get("features") or {}).get(feature)
        if not isinstance(stat, dict):
            continue
        support = _as_float(stat.get("n"))
        if support < max(1, int(min_feature_support)):
            continue
        matched.append({
            "feature": feature,
            "support": support,
            "p_nonempty": _posterior_rate(
                _as_float(stat.get("nonempty")),
                support,
                operator_nonempty,
                prior_strength=12.0,
            ),
            "p_resolve": _posterior_rate(
                _as_float(stat.get("terminal", stat.get("resolved"))),
                support,
                operator_resolve,
                prior_strength=12.0,
            ),
        })

    base_weight = math.sqrt(max(1.0, operator_n))
    total_weight = base_weight
    nonempty_sum = base_weight * operator_nonempty
    resolve_sum = base_weight * operator_resolve
    for row in matched:
        weight = math.sqrt(max(1.0, row["support"]))
        total_weight += weight
        nonempty_sum += weight * row["p_nonempty"]
        resolve_sum += weight * row["p_resolve"]
    p_nonempty = _clamp(nonempty_sum / max(1e-9, total_weight))
    p_resolve = _clamp(resolve_sum / max(1e-9, total_weight))
    productivity = _productive_probability(
        p_nonempty, p_resolve, resolution_floor
    )
    reference = _productive_probability(
        global_nonempty, global_resolve, resolution_floor
    )
    multiplier = productivity / max(0.05, reference)
    multiplier = max(
        float(min_utility_multiplier),
        min(float(max_utility_multiplier), multiplier),
    )
    confidence = _clamp(
        operator_n / (operator_n + max(1.0, float(min_operator_support)))
    )
    supported = bool(
        operator_n >= max(1, int(min_operator_support))
        and confidence >= float(min_confidence)
    )
    policy = dict(model.get("policy") or {})
    policy_mode = str(policy.get("mode") or "")
    if policy_mode in TERMINAL_POLICY_MODES:
        value = terminal_net_benefit(
            context=context,
            operator=operator,
            p_terminal=p_resolve,
            policy=policy,
        )
        threshold_row = dict((policy.get("thresholds") or {}).get(name) or {})
        threshold_enabled = bool(threshold_row.get("enabled"))
        threshold = _as_float(
            threshold_row.get(
                "min_terminal_value_score",
                threshold_row.get("min_terminal_net_benefit"),
            )
        )
        terminal_value_score = value["terminal_net_benefit"]
        terminal_net_benefit_value = (
            terminal_value_score - threshold
            if threshold_enabled
            else terminal_value_score
        )
        terminal_veto = bool(
            supported
            and threshold_enabled
            and terminal_net_benefit_value < 0.0
        )
        result = {
            "enabled": True,
            "supported": supported,
            "reason": (
                "terminal_value_veto"
                if terminal_veto
                else "terminal_value_admitted"
                if supported and threshold_enabled
                else "terminal_policy_uncalibrated"
                if supported
                else "low_operator_support"
            ),
            "policy_mode": policy_mode,
            "monotone_veto_only": True,
            "decision_conditional_top_only": bool(
                policy_mode == DECISION_CONDITIONAL_POLICY_MODE
                or policy.get("decision_conditional_top_only")
            ),
            "operator": name,
            "features": features,
            "operator_support": int(operator_n),
            "matched_feature_count": len(matched),
            "p_nonempty": round(p_nonempty, 6),
            "p_resolve": round(p_resolve, 6),
            "p_terminal": round(p_resolve, 6),
            "expected_productivity": round(p_resolve, 6),
            "reference_productivity": round(global_resolve, 6),
            "utility_multiplier": 1.0,
            "terminal_veto": terminal_veto,
            "terminal_policy_calibrated": threshold_enabled,
            "terminal_net_benefit_threshold": round(threshold, 6),
            "expected_terminal_benefit": round(
                value["expected_terminal_benefit"], 6
            ),
            "expected_access_cost": round(value["expected_access_cost"], 6),
            "terminal_value_score": round(terminal_value_score, 6),
            "terminal_net_benefit": round(terminal_net_benefit_value, 6),
            "normalized_scan_cost": round(value["normalized_scan_cost"], 6),
            "confidence": round(confidence, 6),
            "matched_features": [
                {
                    "feature": row["feature"],
                    "support": int(row["support"]),
                    "p_nonempty": round(float(row["p_nonempty"]), 6),
                    "p_resolve": round(float(row["p_resolve"]), 6),
                }
                for row in sorted(
                    matched, key=lambda item: item["support"], reverse=True
                )[:8]
            ],
        }
        result.update(assess_counterfactual_operator_harm(
            context=context,
            state=state,
            operator=operator,
            model=model,
            min_operator_support=min_operator_support,
            min_feature_support=min_feature_support,
            min_confidence=min_confidence,
        ))
        return result
    result = {
        "enabled": True,
        "supported": supported,
        "reason": "calibrated" if supported else "low_operator_support",
        "operator": name,
        "features": features,
        "operator_support": int(operator_n),
        "matched_feature_count": len(matched),
        "p_nonempty": round(p_nonempty, 6),
        "p_resolve": round(p_resolve, 6),
        "expected_productivity": round(productivity, 6),
        "reference_productivity": round(reference, 6),
        "utility_multiplier": round(multiplier if supported else 1.0, 6),
        "confidence": round(confidence, 6),
        "matched_features": [
            {
                "feature": row["feature"],
                "support": int(row["support"]),
                "p_nonempty": round(float(row["p_nonempty"]), 6),
                "p_resolve": round(float(row["p_resolve"]), 6),
            }
            for row in sorted(
                matched, key=lambda item: item["support"], reverse=True
            )[:8]
        ],
    }
    result.update(assess_counterfactual_operator_harm(
        context=context,
        state=state,
        operator=operator,
        model=model,
        min_operator_support=min_operator_support,
        min_feature_support=min_feature_support,
        min_confidence=min_confidence,
    ))
    return result
