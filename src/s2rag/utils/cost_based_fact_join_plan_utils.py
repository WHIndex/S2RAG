import itertools
import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Set, Tuple

from .adaptive_physical_plan_utils import compile_physical_evidence_plan


_JOIN_SOURCES = {
    "query_schema_memory",
    "entity_predicate_fact_join",
    "typed_fact_join_shadow",
    "relation_fact_join",
    "relation_fact_table",
    "selective_support_join",
}

_OPERATOR_STARTUP_COST = {
    "query_schema_memory": 0.035,
    "entity_predicate_fact_join": 0.055,
    "typed_fact_join_shadow": 0.080,
    "relation_fact_join": 0.060,
    "relation_fact_table": 0.045,
    "selective_support_join": 0.050,
    "planned_retrieval": 0.120,
    "dual_channel_projection": 0.030,
    "support_set_completion": 0.045,
    "path_evidence": 0.040,
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


def _round(value: Any) -> float:
    return round(_as_float(value, 0.0), 6)


def _candidate_cost(
    candidate: Dict[str, Any],
    *,
    token_budget: int,
) -> Dict[str, float]:
    source = str(candidate.get("source") or "extra_evidence")
    token_count = max(1, _as_int(candidate.get("token_count"), 1))
    scan_rows = max(
        1,
        _as_int(
            candidate.get("estimated_endpoint_scan_rows"),
            _as_int(candidate.get("schema_frequency"), 1),
        ),
    )
    scan_cost = _clamp(math.log1p(scan_rows) / math.log1p(1000.0))
    measured_latency_ms = max(
        0.0, _as_float(candidate.get("operator_latency_ms"), 0.0)
    )
    latency_cost = _clamp(math.log1p(measured_latency_ms) / math.log1p(5000.0))
    startup_cost = _OPERATOR_STARTUP_COST.get(source, 0.065)
    compute_cost = _clamp(
        startup_cost
        + 0.52 * scan_cost
        + 0.20 * _clamp(candidate.get("compute_cost"))
        + 0.18 * latency_cost
    )
    return {
        "tokens": float(token_count),
        "token_cost": _clamp(token_count / float(max(1, int(token_budget)))),
        "scan_rows": float(scan_rows),
        "scan_cost": scan_cost,
        "latency_ms": measured_latency_ms,
        "compute_cost": compute_cost,
    }


def _candidate_benefit(candidate: Dict[str, Any]) -> float:
    learned_confidence = _clamp(candidate.get("learned_confidence"))
    learned_gain = _as_float(candidate.get("learned_expected_utility"), 0.0)
    learned_signal = max(-0.25, min(0.25, learned_gain)) * learned_confidence
    structural_signal = (
        0.18 * _clamp(candidate.get("coverage_gain"))
        + 0.08 * _clamp(candidate.get("relation_overlap"))
        + 0.07 * _clamp(candidate.get("provenance_score"))
        + 0.05 * _clamp(candidate.get("schema_selectivity"))
        + 0.04 * _clamp(candidate.get("join_key_overlap"))
        + 0.04 * float(bool(candidate.get("predicate_contract_match")))
        + 0.03 * float(bool(candidate.get("answer_type_match")))
    )
    return (
        0.51 * _clamp(candidate.get("utility"))
        + 0.16 * learned_signal
        + structural_signal
    )


def _covered_slots(rows: Iterable[Dict[str, Any]]) -> Set[str]:
    covered: Set[str] = set()
    for row in rows:
        covered.update(str(value) for value in row.get("covered_slot_ids") or [])
        covered.update(str(value) for value in row.get("uncovered_slot_ids") or [])
    return covered


def _score_plan(
    rows: Tuple[Dict[str, Any], ...],
    *,
    required_slot_ids: Set[str],
    token_budget: int,
    token_weight: float,
    compute_weight: float,
    risk_weight: float,
    redundancy_weight: float,
) -> Dict[str, Any]:
    if not rows:
        physical_plan = compile_physical_evidence_plan(
            (),
            required_slot_ids=required_slot_ids,
        )
        return {
            "candidate_ids": [],
            "operators": [],
            "objective": 0.0,
            "benefit": 0.0,
            "risk": 0.0,
            "token_count": 0,
            "token_cost": 0.0,
            "compute_cost": 0.0,
            "estimated_scan_rows": 0,
            "covered_slot_ids": [],
            "required_slot_coverage": 0.0,
            "join_candidate_count": 0,
            "physical_plan": physical_plan,
        }

    costs = [_candidate_cost(row, token_budget=token_budget) for row in rows]
    benefits = sorted((_candidate_benefit(row) for row in rows), reverse=True)
    benefit = sum(value / math.sqrt(rank + 1.0) for rank, value in enumerate(benefits))
    covered_slots = _covered_slots(rows)
    required_covered = required_slot_ids & covered_slots
    slot_coverage = len(required_covered) / float(max(1, len(required_slot_ids)))
    if required_slot_ids:
        benefit += 0.14 * slot_coverage

    sources = [str(row.get("source") or "extra_evidence") for row in rows]
    source_counts = Counter(sources)
    lineage_counts = Counter(
        str(row.get("lineage_group_id") or "")
        for row in rows
        if str(row.get("lineage_group_id") or "")
    )
    diversity_bonus = 0.018 * max(0, len(source_counts) - 1)
    duplicate_penalty = 0.45 * float(redundancy_weight) * sum(
        max(0, count - 1) for count in source_counts.values()
    )
    duplicate_penalty += 0.70 * float(redundancy_weight) * sum(
        max(0, count - 1) for count in lineage_counts.values()
    )
    aggregate_risk = 0.62 * max(_clamp(row.get("risk")) for row in rows)
    aggregate_risk += 0.38 * sum(_clamp(row.get("risk")) for row in rows) / len(rows)
    token_count = int(sum(cost["tokens"] for cost in costs))
    token_cost = token_count / float(max(1, int(token_budget)))
    compute_cost = sum(cost["compute_cost"] for cost in costs)
    step_metrics = {
        str(row.get("candidate_id") or ""): cost
        for row, cost in zip(rows, costs)
    }
    physical_plan = compile_physical_evidence_plan(
        rows,
        required_slot_ids=required_slot_ids,
        step_metrics=step_metrics,
    )
    objective = (
        benefit
        + diversity_bonus
        - float(risk_weight) * aggregate_risk
        - float(token_weight) * token_cost
        - float(compute_weight) * compute_cost
        - duplicate_penalty
    )
    return {
        "candidate_ids": [str(row.get("candidate_id") or "") for row in rows],
        "operators": sorted(set(sources)),
        "objective": _round(objective),
        "benefit": _round(benefit),
        "risk": _round(aggregate_risk),
        "token_count": token_count,
        "token_cost": _round(token_cost),
        "compute_cost": _round(compute_cost),
        "estimated_scan_rows": int(sum(cost["scan_rows"] for cost in costs)),
        "covered_slot_ids": sorted(covered_slots),
        "required_slot_coverage": _round(slot_coverage),
        "join_candidate_count": sum(source in _JOIN_SOURCES for source in sources),
        "source_counts": dict(source_counts),
        "duplicate_penalty": _round(duplicate_penalty),
        "diversity_bonus": _round(diversity_bonus),
        "physical_plan": physical_plan,
    }


def optimize_cost_based_fact_join_plan(
    candidates: List[Dict[str, Any]],
    *,
    required_slot_ids: Iterable[str],
    max_candidates: int,
    token_budget: int,
    compute_budget: float,
    token_weight: float,
    compute_weight: float,
    risk_weight: float,
    redundancy_weight: float = 0.08,
    min_objective: float = 0.0,
    max_enumerated_candidates: int = 12,
) -> Dict[str, Any]:
    """Choose a reader-visible operator slate under token and access budgets.

    Candidate generation and plan selection are intentionally separate. The
    no-op plan is always present, which makes an unnecessary fact join a plan
    alternative instead of an unconditional side effect.
    """
    max_candidates = max(0, int(max_candidates))
    token_budget = max(1, int(token_budget))
    compute_budget = max(0.0, float(compute_budget))
    required = {str(value) for value in required_slot_ids if str(value)}
    rows = [dict(row) for row in candidates if str(row.get("candidate_id") or "")]

    def pre_score(row: Dict[str, Any]) -> Tuple[float, float, int]:
        cost = _candidate_cost(row, token_budget=token_budget)
        value = _candidate_benefit(row) - float(risk_weight) * _clamp(row.get("risk"))
        value -= float(token_weight) * cost["token_cost"]
        value -= float(compute_weight) * cost["compute_cost"]
        return (
            value,
            _clamp(row.get("coverage_gain")),
            -_as_int(row.get("input_extra_rank"), 0),
        )

    rows.sort(key=pre_score, reverse=True)
    enumerated_rows = rows[: max(1, int(max_enumerated_candidates))]
    plans = [
        _score_plan(
            (),
            required_slot_ids=required,
            token_budget=token_budget,
            token_weight=token_weight,
            compute_weight=compute_weight,
            risk_weight=risk_weight,
            redundancy_weight=redundancy_weight,
        )
    ]
    for size in range(1, min(max_candidates, len(enumerated_rows)) + 1):
        for combination in itertools.combinations(enumerated_rows, size):
            plan = _score_plan(
                combination,
                required_slot_ids=required,
                token_budget=token_budget,
                token_weight=token_weight,
                compute_weight=compute_weight,
                risk_weight=risk_weight,
                redundancy_weight=redundancy_weight,
            )
            if plan["token_count"] > token_budget:
                continue
            if plan["compute_cost"] > compute_budget:
                continue
            plans.append(plan)

    plans.sort(
        key=lambda plan: (
            float(plan.get("objective", 0.0)),
            float(plan.get("required_slot_coverage", 0.0)),
            -int(plan.get("token_count", 0)),
            -len(plan.get("candidate_ids") or []),
        ),
        reverse=True,
    )
    best = plans[0] if plans else _score_plan(
        (),
        required_slot_ids=required,
        token_budget=token_budget,
        token_weight=token_weight,
        compute_weight=compute_weight,
        risk_weight=risk_weight,
        redundancy_weight=redundancy_weight,
    )
    if float(best.get("objective", 0.0)) < float(min_objective):
        best = next(plan for plan in plans if not plan.get("candidate_ids"))
    selected_ids = list(best.get("candidate_ids") or [])
    alternatives = [
        plan
        for plan in plans
        if list(plan.get("candidate_ids") or []) != selected_ids
    ]
    runner_up = alternatives[0] if alternatives else best
    return {
        "enabled": True,
        "reason": "join_plan_selected" if selected_ids else "no_op_plan_selected",
        "input_candidate_count": len(rows),
        "enumerated_candidate_count": len(enumerated_rows),
        "plan_count": len(plans),
        "max_candidates": max_candidates,
        "token_budget": token_budget,
        "compute_budget": _round(compute_budget),
        "min_objective": _round(min_objective),
        "selected_candidate_ids": selected_ids,
        "selected_operators": list(best.get("operators") or []),
        "selected_join_candidate_count": int(best.get("join_candidate_count", 0)),
        "selected_objective": _round(best.get("objective", 0.0)),
        "selected_benefit": _round(best.get("benefit", 0.0)),
        "selected_risk": _round(best.get("risk", 0.0)),
        "selected_token_count": int(best.get("token_count", 0)),
        "selected_compute_cost": _round(best.get("compute_cost", 0.0)),
        "selected_estimated_scan_rows": int(best.get("estimated_scan_rows", 0)),
        "selected_required_slot_coverage": _round(
            best.get("required_slot_coverage", 0.0)
        ),
        "plan_grammar_version": int(
            (best.get("physical_plan") or {}).get("grammar_version", 0)
        ),
        "planning_scope": str(
            (best.get("physical_plan") or {}).get("planning_scope") or ""
        ),
        "selected_plan_expression": str(
            (best.get("physical_plan") or {}).get("expression") or "NO_OP"
        ),
        "selected_plan_fingerprint": str(
            (best.get("physical_plan") or {}).get("plan_fingerprint") or ""
        ),
        "selected_physical_plan": dict(best.get("physical_plan") or {}),
        "runtime_reoptimization_ready": bool(
            (best.get("physical_plan") or {}).get(
                "runtime_reoptimization_ready", False
            )
        ),
        "no_op_margin": _round(best.get("objective", 0.0)),
        "runner_up_margin": _round(
            float(best.get("objective", 0.0))
            - float(runner_up.get("objective", 0.0))
        ),
        "top_plans": plans[:8],
        "pruned_candidate_count": max(0, len(rows) - len(enumerated_rows)),
    }
