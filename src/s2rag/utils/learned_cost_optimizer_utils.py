import json
import math
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple


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


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _bucket(value: Any, cuts: Tuple[float, ...]) -> str:
    value = _as_float(value, 0.0)
    for idx, cut in enumerate(cuts):
        if value < cut:
            return f"b{idx}"
    return f"b{len(cuts)}"


def _rank_bucket(rank: Any) -> str:
    rank = _as_int(rank, 0)
    if rank <= 0:
        return "missing"
    if rank <= 2:
        return "r1_2"
    if rank <= 5:
        return "r3_5"
    if rank <= 12:
        return "r6_12"
    if rank <= 24:
        return "r13_24"
    if rank <= 50:
        return "r25_50"
    return "r51_plus"


def candidate_feature_names(candidate: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return sparse feature names shared by training and online admission.

    The model deliberately uses coarse bins. These features are stable across
    datasets and avoid turning run-specific thresholds into memorized rules.
    """
    state = state or {}
    source = str(candidate.get("source") or "unknown")
    plan_type = str(candidate.get("plan_type") or "none")
    question_type = str(state.get("question_type") or candidate.get("question_type") or "unknown")
    planned_reason = str(state.get("reason") or candidate.get("planned_reason") or "unknown")
    answer_type = str(candidate.get("answer_type") or "unknown")
    answer_slot = str(candidate.get("answer_slot") or "unknown")
    compact = "compact" if bool(candidate.get("compact_table")) else "passage"
    bridge = "bridge" if bool(state.get("bridge_cues") or candidate.get("bridge_cues")) else "no_bridge"
    deficit = (
        "deficit"
        if bool(state.get("evidence_insufficient") or candidate.get("evidence_insufficient"))
        else "no_deficit"
    )
    long_context = (
        "long_context"
        if bool(state.get("long_context") or candidate.get("long_context"))
        else "short_context"
    )
    query_overlap_bucket = f"query_overlap={_bucket(candidate.get('query_overlap'), (0.03, 0.08, 0.16, 0.28, 0.45, 0.70))}"
    missing_gain_bucket = f"missing_gain={_bucket(candidate.get('missing_gain'), (0.01, 0.05, 0.10, 0.18, 0.30, 0.50))}"
    novelty_bucket = f"novelty={_bucket(candidate.get('novelty'), (0.10, 0.25, 0.45, 0.65, 0.82, 0.93))}"
    utility_bucket = f"utility={_bucket(candidate.get('utility'), (0.32, 0.42, 0.50, 0.58, 0.66, 0.76))}"
    risk_bucket = f"risk={_bucket(candidate.get('risk'), (0.12, 0.22, 0.34, 0.46, 0.58, 0.72))}"
    net_bucket = f"net_utility={_bucket(candidate.get('net_utility'), (-0.05, 0.02, 0.08, 0.16, 0.28, 0.42))}"
    plan_utility_bucket = f"plan_utility={_bucket(candidate.get('plan_utility'), (0.38, 0.48, 0.56, 0.64, 0.74, 0.84))}"
    plan_net_bucket = f"plan_net={_bucket(candidate.get('plan_net_utility'), (0.00, 0.08, 0.16, 0.26, 0.38, 0.52))}"
    plan_risk_bucket = f"plan_risk={_bucket(candidate.get('plan_risk'), (0.12, 0.22, 0.34, 0.46, 0.58, 0.72))}"
    endpoint_bucket = f"endpoint={_bucket(candidate.get('endpoint_overlap'), (0.12, 0.24, 0.40, 0.58, 0.78, 0.94))}"
    relation_bucket = f"relation={_bucket(candidate.get('relation_overlap'), (0.02, 0.08, 0.16, 0.28, 0.45, 0.65))}"
    schema_bucket = f"schema={_bucket(candidate.get('schema_selectivity'), (0.30, 0.45, 0.58, 0.70, 0.84, 0.95))}"
    provenance_bucket = f"provenance={_bucket(candidate.get('provenance_score'), (0.05, 0.12, 0.22, 0.34, 0.45, 0.60))}"
    best_rank_bucket = f"best_rank={_rank_bucket(candidate.get('best_rank'))}"
    worst_rank_bucket = f"worst_rank={_rank_bucket(candidate.get('worst_rank'))}"
    coverage_bucket = f"coverage_gain={_bucket(candidate.get('coverage_gain'), (0.01, 0.25, 0.50, 0.75, 1.0))}"
    lineage_bucket = f"lineage_confidence={_bucket(candidate.get('lineage_confidence'), (0.10, 0.25, 0.40, 0.60, 0.80))}"
    slate_size_bucket = f"slate_size={_bucket(candidate.get('slate_size'), (1.5, 2.5, 3.5, 5.5))}"
    slate_rank_bucket = f"slate_rank={_bucket(candidate.get('slate_rank'), (0.5, 1.5, 2.5, 4.5))}"
    slate_token_share_bucket = f"slate_token_share={_bucket(candidate.get('slate_token_share'), (0.12, 0.24, 0.40, 0.60, 0.80))}"
    slate_source_count_bucket = f"slate_source_count={_bucket(candidate.get('slate_source_count'), (1.5, 2.5, 3.5))}"
    covers_slot = "covers_slot" if candidate.get("uncovered_slot_ids") else "no_new_slot"
    comparison = "comparison" if candidate.get("comparison_query") else "non_comparison"
    predicate_tokens = set(candidate.get("predicate_tokens") or [])
    if predicate_tokens & {"birth", "born", "death", "died", "date", "year"}:
        predicate_family = "temporal"
    elif predicate_tokens & {"country", "location", "place", "burial", "buried", "capital"}:
        predicate_family = "location"
    elif predicate_tokens & {"director", "directed", "creator", "producer"}:
        predicate_family = "creator"
    elif predicate_tokens & {"father", "mother", "spouse"}:
        predicate_family = "kinship"
    else:
        predicate_family = "other"
    features = [
        f"source={source}",
        f"plan_type={plan_type}",
        f"source_plan={source}|{plan_type}",
        f"question_type={question_type}",
        f"planned_reason={planned_reason}",
        f"answer_type={answer_type}",
        f"answer_slot={answer_slot}",
        f"surface={compact}",
        f"bridge={bridge}",
        f"deficit={deficit}",
        f"context={long_context}",
        query_overlap_bucket,
        missing_gain_bucket,
        novelty_bucket,
        utility_bucket,
        risk_bucket,
        net_bucket,
        plan_utility_bucket,
        plan_net_bucket,
        plan_risk_bucket,
        endpoint_bucket,
        relation_bucket,
        schema_bucket,
        provenance_bucket,
        best_rank_bucket,
        worst_rank_bucket,
        coverage_bucket,
        lineage_bucket,
        slate_size_bucket,
        slate_rank_bucket,
        slate_token_share_bucket,
        slate_source_count_bucket,
        f"slot_coverage={covers_slot}",
        f"comparison={comparison}",
        f"predicate_family={predicate_family}",
    ]
    features.extend(
        [
            f"source_qtype={source}|{question_type}",
            f"plan_qtype={plan_type}|{question_type}",
            f"source_deficit={source}|{deficit}",
            f"plan_deficit={plan_type}|{deficit}",
            f"schema_relation={schema_bucket}|{relation_bucket}",
            f"rank_provenance={best_rank_bucket}|{provenance_bucket}",
            f"source_slot={source}|{covers_slot}",
            f"slot_qtype={answer_slot}|{question_type}",
            f"coverage_predicate={covers_slot}|{predicate_family}",
        ]
    )
    return features


_CONDITIONAL_NUMERIC_FEATURES = (
    "token_count",
    "query_overlap",
    "missing_gain",
    "novelty",
    "utility",
    "risk",
    "token_cost",
    "compute_cost",
    "net_utility",
    "plan_utility",
    "plan_net_utility",
    "plan_risk",
    "endpoint_overlap",
    "relation_overlap",
    "schema_selectivity",
    "provenance_score",
    "best_rank",
    "worst_rank",
    "coverage_gain",
    "lineage_confidence",
    "slate_size",
    "slate_rank",
    "slate_token_share",
    "slate_source_count",
    "slate_trace_size",
    "slate_source_diversity",
    "slate_same_lineage_count",
    "slate_coverage_redundancy",
    "slate_net_utility_minus_mean",
    "slate_net_utility_minus_max",
    "slate_net_utility_percentile",
    "slate_utility_minus_mean",
    "slate_utility_minus_max",
    "slate_utility_percentile",
    "slate_risk_minus_mean",
    "slate_risk_minus_max",
    "slate_risk_percentile",
    "slate_relation_overlap_minus_mean",
    "slate_relation_overlap_minus_max",
    "slate_relation_overlap_percentile",
    "slate_schema_selectivity_minus_mean",
    "slate_schema_selectivity_minus_max",
    "slate_schema_selectivity_percentile",
    "slate_provenance_score_minus_mean",
    "slate_provenance_score_minus_max",
    "slate_provenance_score_percentile",
)


_LOG_NUMERIC_FEATURES = {
    "token_count",
    "best_rank",
    "worst_rank",
    "slate_size",
    "slate_rank",
    "slate_source_count",
    "slate_trace_size",
    "slate_source_diversity",
    "slate_same_lineage_count",
    "slate_coverage_redundancy",
}


_SLATE_RELATIVE_FEATURES = (
    "net_utility",
    "utility",
    "risk",
    "relation_overlap",
    "schema_selectivity",
    "provenance_score",
)


def annotate_slate_relative_features(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach candidate-within-slate features shared by training and serving."""
    rows = [row for row in candidates if isinstance(row, dict)]
    if not rows:
        return candidates
    source_diversity = len({str(row.get("source") or "unknown") for row in rows})
    lineage_counts: Dict[str, int] = {}
    for row in rows:
        lineage = str(row.get("lineage_group_id") or "")
        if lineage:
            lineage_counts[lineage] = lineage_counts.get(lineage, 0) + 1
    coverage_sets = [set(row.get("covered_slot_ids") or []) for row in rows]
    for idx, row in enumerate(rows):
        row["slate_trace_size"] = len(rows)
        row["slate_source_diversity"] = source_diversity
        lineage = str(row.get("lineage_group_id") or "")
        row["slate_same_lineage_count"] = max(0, lineage_counts.get(lineage, 0) - 1)
        covered = coverage_sets[idx]
        row["slate_coverage_redundancy"] = sum(
            bool(covered & other)
            for other_idx, other in enumerate(coverage_sets)
            if other_idx != idx
        )
    for feature in _SLATE_RELATIVE_FEATURES:
        values = [_as_float(row.get(feature), 0.0) for row in rows]
        mean_value = sum(values) / max(1, len(values))
        max_value = max(values)
        for row, value in zip(rows, values):
            row[f"slate_{feature}_minus_mean"] = value - mean_value
            row[f"slate_{feature}_minus_max"] = value - max_value
            row[f"slate_{feature}_percentile"] = sum(
                other <= value for other in values
            ) / max(1.0, float(len(values)))
    return candidates


def candidate_conditional_feature_values(
    candidate: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Build the exported conditional model feature vector without dataset IDs."""
    values = {f"cat:{feature}": 1.0 for feature in candidate_feature_names(candidate, state)}
    for feature in _CONDITIONAL_NUMERIC_FEATURES:
        value = _as_float(candidate.get(feature), 0.0)
        if feature in _LOG_NUMERIC_FEATURES:
            value = math.log1p(max(0.0, value))
        values[f"num:{feature}"] = value
    return values


_SLATE_HARM_FAMILY_BY_PREFIX = {
    "source": "source",
    "source_qtype": "source",
    "source_slot": "source",
    "source_deficit": "source",
    "plan_type": "plan",
    "plan_qtype": "plan",
    "plan_deficit": "plan",
    "answer_slot": "answer_contract",
    "slot_qtype": "answer_contract",
    "predicate_family": "predicate",
    "coverage_predicate": "predicate",
    "slot_coverage": "coverage",
    "coverage_gain": "coverage",
    "lineage_confidence": "lineage",
    "relation": "relation",
    "schema": "schema",
    "schema_relation": "schema",
    "provenance": "provenance",
    "rank_provenance": "provenance",
    "best_rank": "rank",
    "worst_rank": "rank",
    "risk": "utility_risk",
    "net_utility": "utility_risk",
    "missing_gain": "utility_risk",
    "slate_size": "slate",
    "slate_rank": "slate",
    "slate_token_share": "slate",
    "slate_source_count": "slate",
}


def _feature_family(feature: str) -> str:
    return _SLATE_HARM_FAMILY_BY_PREFIX.get(str(feature).split("=", 1)[0], "")


def _counterfactual_feature_evidence(
    feature: str,
    stat: Dict[str, Any],
    *,
    min_support: int,
    min_stable_datasets: int,
    min_effect: float,
) -> Optional[Dict[str, Any]]:
    family = _feature_family(feature)
    n = max(0.0, _as_float(stat.get("n"), 0.0))
    if not family or n < max(1, int(min_support)):
        return None
    delta_sum = _as_float(stat.get("delta_sum"), 0.0)
    delta_sq_sum = _as_float(stat.get("delta_sq_sum"), 0.0)
    mean_delta = delta_sum / max(1.0, n)
    variance = max(0.0, delta_sq_sum / max(1.0, n) - mean_delta * mean_delta)
    standard_error = max(math.sqrt(variance / max(1.0, n)), 0.01 / math.sqrt(max(1.0, n)))
    upper_ci = mean_delta + 1.96 * standard_error
    if mean_delta > -float(min_effect) or upper_ci >= 0.0:
        return None

    harmful_datasets = []
    beneficial_datasets = []
    for dataset, dataset_stat in sorted((stat.get("dataset_stats") or {}).items()):
        if not isinstance(dataset_stat, dict):
            continue
        dataset_n = max(0.0, _as_float(dataset_stat.get("n"), 0.0))
        if dataset_n <= 0.0:
            continue
        dataset_delta = _as_float(dataset_stat.get("delta_sum"), 0.0) / dataset_n
        negative = _as_float(dataset_stat.get("negative"), 0.0)
        positive = _as_float(dataset_stat.get("positive"), 0.0)
        if dataset_delta < 0.0 and negative > positive:
            harmful_datasets.append(str(dataset))
        elif dataset_delta > 0.0 and positive > negative:
            beneficial_datasets.append(str(dataset))
    if (
        len(harmful_datasets) < max(1, int(min_stable_datasets))
        or len(harmful_datasets) <= len(beneficial_datasets)
    ):
        return None
    return {
        "feature": feature,
        "family": family,
        "support": n,
        "mean_delta": mean_delta,
        "upper_ci": upper_ci,
        "p_harm": _as_float(stat.get("negative"), 0.0) / max(1.0, n),
        "p_benefit": _as_float(stat.get("positive"), 0.0) / max(1.0, n),
        "harmful_datasets": harmful_datasets,
        "beneficial_datasets": beneficial_datasets,
    }


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_value = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(max(value, -60.0))
    return exp_value / (1.0 + exp_value)


def _conditional_effect_bin(
    probability: float,
    bins: List[Dict[str, Any]],
) -> Dict[str, Any]:
    for row in bins:
        if probability <= _as_float(row.get("max_probability"), 1.0):
            return row
    return bins[-1] if bins else {}


def _score_candidate_with_conditional_harm_model(
    candidate: Dict[str, Any],
    conditional_model: Dict[str, Any],
    *,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    feature_values = candidate_conditional_feature_values(candidate, state)
    coefficients = conditional_model.get("coefficients") or {}
    logit = _as_float(conditional_model.get("intercept"), 0.0)
    matched = []
    for feature, value in feature_values.items():
        coefficient = _as_float(coefficients.get(feature), 0.0)
        if coefficient == 0.0 or value == 0.0:
            continue
        contribution = coefficient * value
        logit += contribution
        matched.append((feature, contribution))
    p_harm = _sigmoid(logit)
    effect_bin = _conditional_effect_bin(
        p_harm,
        list(conditional_model.get("effect_calibration_bins") or []),
    )
    expected_candidate_value = _as_float(effect_bin.get("mean_delta"), 0.0)
    quality_upper = _as_float(
        effect_bin.get("upper_ci"),
        expected_candidate_value,
    )
    token_credit = _as_float(conditional_model.get("token_cost_weight"), 0.02) * min(
        1.0,
        max(0.0, _as_float(candidate.get("token_count"), 0.0))
        / max(1.0, _as_float(conditional_model.get("token_budget"), 256.0)),
    )
    compute_credit = _as_float(conditional_model.get("compute_cost_weight"), 0.01) * max(
        0.0,
        _as_float(candidate.get("compute_cost"), 0.0),
    )
    net_drop_utility = -expected_candidate_value + token_credit + compute_credit
    coverage_protected = bool(candidate.get("uncovered_slot_ids"))
    threshold = _as_float(conditional_model.get("p_harm_threshold"), 0.80)
    max_candidate_value = _as_float(
        conditional_model.get("max_predicted_candidate_value"),
        0.0,
    )
    max_quality_upper_ci = _as_float(
        conditional_model.get("max_quality_upper_ci"),
        0.0,
    )
    min_net_drop_utility = _as_float(
        conditional_model.get("min_net_drop_utility"),
        0.005,
    )
    veto_recommended = bool(
        p_harm >= threshold
        and expected_candidate_value <= max_candidate_value
        and quality_upper <= max_quality_upper_ci
        and net_drop_utility >= min_net_drop_utility
        and not coverage_protected
    )
    top_features = sorted(matched, key=lambda row: abs(row[1]), reverse=True)[:8]
    return {
        "enabled": True,
        "reason": "conditional_harm_veto" if veto_recommended else "conditional_keep",
        "decision_type": "conditional_cost_effect",
        "matched_feature_count": len(matched),
        "harm_signal_count": len(top_features),
        "harm_family_count": 0,
        "stable_dataset_count": int(
            conditional_model.get("training_dataset_count", 0) or 0
        ),
        "support": _as_float(effect_bin.get("n"), 0.0),
        "confidence": round(p_harm, 6),
        "expected_gain": round(expected_candidate_value, 6),
        "quality_upper_ci": round(quality_upper, 6),
        "p_degrade": round(p_harm, 6),
        "coverage_protected": coverage_protected,
        "veto_recommended": veto_recommended,
        "net_drop_utility": round(net_drop_utility, 6),
        "token_cost_credit": round(token_credit, 6),
        "compute_cost_credit": round(compute_credit, 6),
        "policy_threshold": round(threshold, 6),
        "max_quality_upper_ci": round(max_quality_upper_ci, 6),
        "top_features": [
            {
                "feature": feature,
                "contribution": round(float(contribution), 6),
            }
            for feature, contribution in top_features
        ],
    }


def score_candidate_with_counterfactual_slate_model(
    candidate: Dict[str, Any],
    model: Optional[Dict[str, Any]],
    *,
    state: Optional[Dict[str, Any]] = None,
    min_support: int = 4,
    min_harm_signals: int = 2,
    min_stable_datasets: int = 3,
    min_effect: float = 0.012,
) -> Dict[str, Any]:
    """Score a candidate using direct leave-one-candidate-out interventions.

    A hard veto requires independent feature families, cross-dataset sign
    stability, and a negative upper confidence bound. Candidates that cover an
    outstanding required slot are protected regardless of the learned score.
    """
    if not model or not bool(model.get("enabled")):
        return {"enabled": False, "reason": (model or {}).get("reason", "model_unavailable")}
    if str(model.get("model_type") or "") != "counterfactual_slate_cost_optimizer":
        return {"enabled": False, "reason": "incompatible_model_type"}
    conditional_model = model.get("conditional_harm_model") or {}
    if bool(conditional_model.get("enabled")):
        return _score_candidate_with_conditional_harm_model(
            candidate,
            conditional_model,
            state=state,
        )

    evidence_by_family: Dict[str, Dict[str, Any]] = {}
    feature_stats = model.get("features") or {}
    for feature in candidate_feature_names(candidate, state):
        stat = feature_stats.get(feature)
        if not isinstance(stat, dict):
            continue
        evidence = _counterfactual_feature_evidence(
            feature,
            stat,
            min_support=int(min_support),
            min_stable_datasets=int(min_stable_datasets),
            min_effect=float(min_effect),
        )
        if evidence is None:
            continue
        current = evidence_by_family.get(evidence["family"])
        if current is None or (
            evidence["upper_ci"], evidence["mean_delta"], -evidence["support"]
        ) < (
            current["upper_ci"], current["mean_delta"], -current["support"]
        ):
            evidence_by_family[evidence["family"]] = evidence

    signals = sorted(
        evidence_by_family.values(),
        key=lambda row: (row["upper_ci"], row["mean_delta"], -row["support"]),
    )
    if signals:
        weights = [math.log1p(float(row["support"])) for row in signals]
        denominator = max(1e-9, sum(weights))
        expected_delta = sum(weight * float(row["mean_delta"]) for weight, row in zip(weights, signals)) / denominator
        upper_ci = sum(weight * float(row["upper_ci"]) for weight, row in zip(weights, signals)) / denominator
        p_harm = sum(weight * float(row["p_harm"]) for weight, row in zip(weights, signals)) / denominator
        support = sum(float(row["support"]) for row in signals)
    else:
        expected_delta = 0.0
        upper_ci = 0.0
        p_harm = 0.0
        support = 0.0

    coverage_protected = bool(candidate.get("uncovered_slot_ids"))
    stable_dataset_count = max(
        (len(row["harmful_datasets"]) for row in signals),
        default=0,
    )
    veto_recommended = bool(
        len(signals) >= max(2, int(min_harm_signals))
        and upper_ci <= -float(min_effect)
        and not coverage_protected
    )
    confidence = _clamp(
        min(
            len(signals) / float(max(2, int(min_harm_signals))),
            stable_dataset_count / float(max(1, int(min_stable_datasets))),
            support / float(max(1, int(min_support) * max(2, int(min_harm_signals)))),
        )
    )
    return {
        "enabled": True,
        "reason": "stable_harm" if veto_recommended else "insufficient_stable_harm",
        "matched_feature_count": len(signals),
        "harm_signal_count": len(signals),
        "harm_family_count": len(evidence_by_family),
        "stable_dataset_count": stable_dataset_count,
        "support": round(support, 4),
        "confidence": round(confidence, 6),
        "expected_gain": round(expected_delta, 6),
        "quality_upper_ci": round(upper_ci, 6),
        "p_degrade": round(p_harm, 6),
        "coverage_protected": coverage_protected,
        "veto_recommended": veto_recommended,
        "top_features": [
            {
                "feature": row["feature"],
                "family": row["family"],
                "support": round(float(row["support"]), 4),
                "mean_delta": round(float(row["mean_delta"]), 6),
                "upper_ci": round(float(row["upper_ci"]), 6),
                "harmful_datasets": list(row["harmful_datasets"]),
                "beneficial_datasets": list(row["beneficial_datasets"]),
            }
            for row in signals[:6]
        ],
    }


def _empty_model(reason: str) -> Dict[str, Any]:
    return {
        "version": 1,
        "enabled": False,
        "reason": reason,
        "global": {"n": 0.0, "positive": 0.0, "negative": 0.0, "delta_sum": 0.0},
        "features": {},
    }


@lru_cache(maxsize=8)
def load_learned_cost_optimizer_model(model_path: str) -> Dict[str, Any]:
    if not model_path:
        return _empty_model("missing_model_path")
    expanded = os.path.abspath(os.path.expanduser(str(model_path)))
    if not os.path.exists(expanded):
        return _empty_model("model_file_not_found")
    try:
        with open(expanded, "r", encoding="utf-8") as handle:
            model = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        bad = _empty_model("model_load_failed")
        bad["error"] = str(exc)[:200]
        return bad
    if not isinstance(model, dict):
        return _empty_model("model_not_dict")
    model["enabled"] = bool(model.get("features")) or bool(
        (model.get("conditional_harm_model") or {}).get("enabled")
    )
    model["model_path"] = expanded
    return model


def _stat_rates(stat: Dict[str, Any], global_stat: Dict[str, Any], prior_strength: float = 3.0) -> Dict[str, float]:
    n = max(0.0, _as_float(stat.get("n"), 0.0))
    positive = max(0.0, _as_float(stat.get("positive"), 0.0))
    negative = max(0.0, _as_float(stat.get("negative"), 0.0))
    global_n = max(1.0, _as_float(global_stat.get("n"), 1.0))
    global_pos = _as_float(global_stat.get("positive"), 0.0) / global_n
    global_neg = _as_float(global_stat.get("negative"), 0.0) / global_n
    global_delta = _as_float(global_stat.get("delta_sum"), 0.0) / global_n
    denom = n + float(prior_strength)
    return {
        "support": n,
        "positive": positive,
        "negative": negative,
        "p_improve": (positive + prior_strength * global_pos) / denom,
        "p_degrade": (negative + prior_strength * global_neg) / denom,
        "avg_delta": (_as_float(stat.get("delta_sum"), 0.0) + prior_strength * global_delta) / denom,
    }


_HARM_FEATURE_PREFIXES = (
    "source=",
    "source_qtype=",
    "source_slot=",
    "source_deficit=",
    "plan_type=",
    "plan_qtype=",
    "plan_deficit=",
    "answer_slot=",
    "predicate_family=",
    "coverage_predicate=",
    "slot_qtype=",
    "slot_coverage=",
    "coverage_gain=",
    "lineage_confidence=",
    "missing_gain=",
    "risk=",
    "net_utility=",
    "relation=",
    "schema=",
    "provenance=",
    "schema_relation=",
    "rank_provenance=",
)


def _harm_feature_rates(
    matched: List[Tuple[str, Dict[str, float]]],
    *,
    min_support: int,
) -> List[Tuple[str, Dict[str, float]]]:
    harmful = []
    for feature, rates in matched:
        if not feature.startswith(_HARM_FEATURE_PREFIXES):
            continue
        if rates["support"] < max(1, int(min_support)):
            continue
        margin = rates["p_degrade"] - rates["p_improve"]
        if rates["avg_delta"] >= 0.0 or margin < 0.0 or rates["negative"] <= 0.0:
            continue
        enriched = dict(rates)
        enriched["harm_margin"] = margin
        harmful.append((feature, enriched))
    return sorted(
        harmful,
        key=lambda item: (
            item[1]["harm_margin"],
            -item[1]["avg_delta"],
            math.log1p(item[1]["support"]),
        ),
        reverse=True,
    )


def score_candidate_with_learned_cost_model(
    candidate: Dict[str, Any],
    model: Optional[Dict[str, Any]],
    *,
    state: Optional[Dict[str, Any]] = None,
    min_support: int = 4,
) -> Dict[str, Any]:
    if not model or not bool(model.get("enabled")):
        return {"enabled": False, "reason": (model or {}).get("reason", "model_unavailable")}
    features = candidate_feature_names(candidate, state)
    feature_stats = model.get("features") or {}
    global_stat = model.get("global") or {}
    matched: List[Tuple[str, Dict[str, float]]] = []
    for feature in features:
        stat = feature_stats.get(feature)
        if not isinstance(stat, dict):
            continue
        rates = _stat_rates(stat, global_stat)
        if rates["support"] <= 0:
            continue
        matched.append((feature, rates))

    if not matched:
        global_rates = _stat_rates({}, global_stat)
        return {
            "enabled": True,
            "reason": "global_only",
            "matched_feature_count": 0,
            "support": 0.0,
            "confidence": 0.0,
            "p_improve": round(global_rates["p_improve"], 6),
            "p_degrade": round(global_rates["p_degrade"], 6),
            "expected_gain": round(global_rates["avg_delta"], 6),
            "top_features": [],
        }

    weighted_support = 0.0
    p_improve = 0.0
    p_degrade = 0.0
    expected_gain = 0.0
    top_features = []
    for feature, rates in matched:
        support = rates["support"]
        weight = math.log1p(support)
        weighted_support += weight
        p_improve += weight * rates["p_improve"]
        p_degrade += weight * rates["p_degrade"]
        expected_gain += weight * rates["avg_delta"]
        top_features.append((feature, support, rates["avg_delta"]))

    denom = max(weighted_support, 1e-6)
    raw_support = sum(rates["support"] for _, rates in matched)
    confidence = _clamp(raw_support / max(float(min_support), raw_support + float(min_support)))
    expected_gain = expected_gain / denom


    probability_margin = (p_improve - p_degrade) / denom
    expected_utility = 0.72 * expected_gain + 0.28 * probability_margin
    top_features = sorted(top_features, key=lambda row: abs(row[2]) * math.log1p(row[1]), reverse=True)[:6]
    harmful = _harm_feature_rates(matched, min_support=int(min_support))
    if harmful:
        harm_feature, harm_rates = harmful[0]
        harm_support = harm_rates["support"]
        harm_confidence = _clamp(
            harm_support / max(float(min_support), harm_support + float(min_support))
        )
        harm_payload = {
            "harm_feature_count": len(harmful),
            "harm_support": round(harm_support, 4),
            "harm_confidence": round(harm_confidence, 6),
            "harm_p_improve": round(harm_rates["p_improve"], 6),
            "harm_p_degrade": round(harm_rates["p_degrade"], 6),
            "harm_margin": round(harm_rates["harm_margin"], 6),
            "harm_expected_gain": round(harm_rates["avg_delta"], 6),
            "harm_top_features": [
                {
                    "feature": feature,
                    "support": round(rates["support"], 4),
                    "p_degrade": round(rates["p_degrade"], 6),
                    "p_improve": round(rates["p_improve"], 6),
                    "avg_delta": round(rates["avg_delta"], 6),
                }
                for feature, rates in harmful[:4]
            ],
            "harm_primary_feature": harm_feature,
        }
    else:
        harm_payload = {
            "harm_feature_count": 0,
            "harm_support": 0.0,
            "harm_confidence": 0.0,
            "harm_p_improve": 0.0,
            "harm_p_degrade": 0.0,
            "harm_margin": 0.0,
            "harm_expected_gain": 0.0,
            "harm_top_features": [],
            "harm_primary_feature": "",
        }
    return {
        "enabled": True,
        "reason": "matched_features",
        "matched_feature_count": len(matched),
        "support": round(raw_support, 4),
        "confidence": round(confidence, 6),
        "p_improve": round(p_improve / denom, 6),
        "p_degrade": round(p_degrade / denom, 6),
        "expected_gain": round(expected_gain, 6),
        "expected_utility": round(expected_utility, 6),
        "top_features": [
            {"feature": feature, "support": round(support, 4), "avg_delta": round(delta, 6)}
            for feature, support, delta in top_features
        ],
        **harm_payload,
    }
