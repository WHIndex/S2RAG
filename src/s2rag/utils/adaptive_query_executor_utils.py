"""Lazy, state-aware execution for retrieval-time evidence operators.

Unlike the candidate-slate optimizer, this executor invokes an operator only
after choosing it.  It then feeds observed cardinality and materialization
signals back into the residual evidence state before choosing the next step.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple


EXECUTOR_VERSION = 10
PLANNING_SCOPE = "logical_physical_decoupled_lazy_evidence_execution"

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do",
    "does", "for", "from", "how", "in", "is", "it", "of", "on", "or",
    "that", "the", "to", "was", "were", "what", "when", "where", "which",
    "who", "whom", "whose", "with",
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


def _tokens(text: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def lexical_query_coverage(question: str, docs: Sequence[str]) -> float:
    question_tokens = _tokens(question)
    if not question_tokens:
        return 1.0
    evidence_tokens: set[str] = set()
    for doc in docs:
        evidence_tokens.update(_tokens(str(doc)[:2400]))
    return len(question_tokens & evidence_tokens) / float(len(question_tokens))


def estimate_index_scan_rows(
    active_keys: Iterable[Any],
    index: Mapping[Any, Any],
    *,
    per_key_cap: int,
) -> int:
    cap = max(1, int(per_key_cap))
    total = 0
    for key in active_keys:
        values = index.get(key)
        if values is None:
            continue
        try:
            cardinality = len(values)
        except TypeError:
            cardinality = 1
        total += min(cap, max(0, int(cardinality)))
    return total


def estimate_inverted_index_access(
    access_paths: Sequence[Tuple[str, Iterable[Any], Mapping[Any, Any]]],
    *,
    per_key_cap: int,
    max_key_fanout: int,
) -> Dict[str, Any]:
    """Estimate probes and touched rows across multiple inverted indexes."""
    row_cap = max(1, int(per_key_cap))
    fanout_cap = max(0, int(max_key_fanout))
    probe_rows: Dict[str, int] = {}
    probe_counts: Dict[str, int] = {}
    lookup_counts: Dict[str, int] = {}
    fanout_rejected_keys: Dict[str, int] = {}
    fanout_rejected_rows: Dict[str, int] = {}
    truncated_rows: Dict[str, int] = {}
    for name, keys, index in access_paths:
        rows = 0
        probes = 0
        lookups = 0
        rejected_keys = 0
        rejected_rows = 0
        truncated = 0
        for key in keys:
            lookups += 1
            values = index.get(key)
            if not values:
                continue
            try:
                cardinality = len(values)
            except TypeError:
                cardinality = 1
            if fanout_cap > 0 and cardinality > fanout_cap:
                rejected_keys += 1
                rejected_rows += cardinality
                continue
            probes += 1
            rows += min(row_cap, cardinality)
            truncated += max(0, cardinality - row_cap)
        probe_rows[str(name)] = rows
        probe_counts[str(name)] = probes
        lookup_counts[str(name)] = lookups
        fanout_rejected_keys[str(name)] = rejected_keys
        fanout_rejected_rows[str(name)] = rejected_rows
        truncated_rows[str(name)] = truncated
    return {
        "estimated_scan_rows": sum(probe_rows.values()),
        "probe_rows": probe_rows,
        "probe_counts": probe_counts,
        "lookup_counts": lookup_counts,
        "fanout_rejected_keys": fanout_rejected_keys,
        "fanout_rejected_rows": fanout_rejected_rows,
        "truncated_rows": truncated_rows,
    }


def build_logical_plan_execution_policy(
    *,
    logical_plan: Mapping[str, Any] | None,
    schema_probe_enabled: bool,
    relation_fact_join_enabled: bool,
) -> Dict[str, Any]:
    """Map a logical fact-join plan to a conservative physical seed.

    A compiled logical plan describes the binding that must eventually be
    produced; it must not force an expensive relation join to run first.
    Schema probing is the cheapest operator that can establish instance-bound
    tuples, so it seeds execution when available.  Relation join remains a
    conditional fallback when the seed leaves the terminal binding unresolved.
    """
    plan = dict(logical_plan or {})
    compiled = bool(
        plan.get("compile_status") == "compiled"
        and _as_int(plan.get("operator_count")) > 0
    )
    seed_operator = ""
    fallback_operator = ""
    reason = "logical_plan_unavailable"
    if compiled and schema_probe_enabled:
        seed_operator = "SCHEMA_PROBE"
        fallback_operator = (
            "RELATION_FACT_JOIN" if relation_fact_join_enabled else ""
        )
        reason = "schema_probe_seeds_instance_binding"
    elif compiled and relation_fact_join_enabled:
        seed_operator = "RELATION_FACT_JOIN"
        reason = "schema_probe_unavailable_relation_fallback"
    elif compiled:
        reason = "no_physical_operator_available"

    return {
        "policy_version": 1,
        "logical_plan_compiled": compiled,
        "semantic_obligations": ["terminal_binding"] if compiled else [],
        "physical_seed_operator": seed_operator,
        "required_seed_operators": [seed_operator] if seed_operator else [],
        "conditional_fallback_operator": fallback_operator,
        "relation_join_unconditionally_required": bool(
            seed_operator == "RELATION_FACT_JOIN"
        ),
        "reason": reason,
    }


def build_lazy_operator_spec(
    *,
    operator: str,
    source: str,
    access_path: str,
    enabled: bool,
    estimated_utility: float,
    predicted_risk: float,
    risk_uncertainty: float = 0.0,
    estimated_scan_rows: int,
    estimated_tokens: int,
    startup_cost: float,
    execute: Callable[[], Mapping[str, Any]],
    terminal_candidate_fields: Sequence[str] = (),
    terminal_requires_novel_candidate: bool = False,
    terminal_candidates_resolve_residual: bool = True,
    plan_obligation_assessor: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ] | None = None,
    requires_executed_operators: Sequence[str] = (),
    incumbent_priority: int = 100,
) -> Dict[str, Any]:
    scan_rows = max(0, int(estimated_scan_rows))
    scan_cost = min(1.0, math.log1p(scan_rows) / math.log1p(1000.0))
    return {
        "operator": str(operator),
        "source": str(source),
        "access_path": str(access_path),
        "enabled": bool(enabled),
        "estimated_utility": _clamp(estimated_utility),
        "predicted_risk": _clamp(predicted_risk),
        "risk_uncertainty": _clamp(risk_uncertainty, 0.0, 0.5),
        "estimated_scan_rows": scan_rows,
        "estimated_tokens": max(0, int(estimated_tokens)),
        "estimated_compute_cost": _clamp(float(startup_cost) + 0.55 * scan_cost),
        "terminal_candidate_fields": [str(field) for field in terminal_candidate_fields],
        "terminal_requires_novel_candidate": bool(terminal_requires_novel_candidate),
        "terminal_candidates_resolve_residual": bool(
            terminal_candidates_resolve_residual
        ),
        "plan_obligation_assessor": plan_obligation_assessor,
        "requires_executed_operators": [
            str(name) for name in requires_executed_operators if str(name)
        ],
        "incumbent_priority": int(incumbent_priority),
        "execute": execute,
    }


def _public_operator(spec: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in spec.items()
        if key not in {"execute", "plan_obligation_assessor"}
    }


def _operator_names(operators: Sequence[Mapping[str, Any]]) -> List[str]:
    return sorted(
        str(operator.get("operator") or "")
        for operator in operators
        if str(operator.get("operator") or "")
    )


def _objective(
    spec: Mapping[str, Any],
    *,
    residual_deficit: bool,
    token_budget: int,
    token_weight: float,
    compute_weight: float,
    risk_weight: float,
    utility_multiplier: float = 1.0,
) -> float:
    deficit_multiplier = 1.0 if residual_deficit else 0.20
    token_cost = _as_int(spec.get("estimated_tokens")) / float(max(1, token_budget))
    return (
        deficit_multiplier
        * _clamp(spec.get("estimated_utility"))
        * max(0.0, _as_float(utility_multiplier, 1.0))
        - float(risk_weight) * _clamp(spec.get("predicted_risk"))
        - float(token_weight) * token_cost
        - float(compute_weight) * _clamp(spec.get("estimated_compute_cost"))
    )


def _observation(info: Mapping[str, Any], spec: Mapping[str, Any]) -> Dict[str, Any]:
    candidate_count = max(
        _as_int(info.get("candidate_count")),
        _as_int(info.get("raw_candidate_count")),
    )
    accepted_count = max(
        _as_int(info.get("accepted_count")),
        _as_int(info.get("promoted_to_top5")),
    )
    terminal_count = accepted_count
    terminal_fields = list(spec.get("terminal_candidate_fields") or [])
    for field in terminal_fields:
        terminal_count = max(terminal_count, _as_int(info.get(field)))
    if bool(spec.get("terminal_requires_novel_candidate")):
        novel_count = sum(
            bool(item.get("schema_fact_tuple"))
            and not bool(item.get("already_top5"))
            and bool(item.get("entity_bound", item.get("join_key_matched", False)))
            for item in info.get("candidate_items") or []
        )
        terminal_count = min(terminal_count, novel_count)
    index_access = dict(info.get("index_access") or {})
    observed_scan_rows = max(
        _as_int(index_access.get("input_scan_rows")),
        _as_int(index_access.get("rows_after_cap")),
        candidate_count,
        _as_int(info.get("raw_candidate_count")),
    )
    obligation_assessment: Dict[str, Any] = {}
    assessor = spec.get("plan_obligation_assessor")
    if callable(assessor):
        obligation_assessment = dict(assessor(info) or {})
    obligation_satisfied = None
    if obligation_assessment:
        obligation_satisfied = bool(
            obligation_assessment.get(
                "completed",
                obligation_assessment.get("satisfied", False),
            )
        )
    return {
        "reason": str(info.get("reason") or "unknown"),
        "candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "terminal_candidate_count": terminal_count,
        "observed_scan_rows": observed_scan_rows,
        "deduplicated_rows": _as_int(index_access.get("deduplicated_rows")),
        "duplicate_rows": _as_int(index_access.get("duplicate_rows")),
        "fanout_rejected_keys": _as_int(index_access.get("fanout_rejected_keys")),
        "fanout_rejected_rows": _as_int(index_access.get("fanout_rejected_rows")),
        "truncated_rows": _as_int(index_access.get("truncated_rows")),
        "loaded_doc_rows": _as_int(index_access.get("loaded_doc_rows")),
        "upstream_bind_input_count": _as_int(
            info.get("upstream_bind_input_count")
        ),
        "upstream_bind_fact_count": _as_int(
            info.get("upstream_bind_fact_count")
        ),
        "bind_output_count": _as_int(info.get("bind_output_count")),
        "applied": bool(info.get("applied")),
        "plan_obligation_assessment": obligation_assessment,
        "plan_obligation_satisfied": obligation_satisfied,
        "resolved_residual": bool(info.get("applied")) or bool(
            spec.get("terminal_candidates_resolve_residual", True)
            and terminal_count > 0
        ),
    }


def _runtime_adjusted_operator(
    row: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
) -> Dict[str, Any]:
    """Adjust downstream estimates using already observed dependency output."""
    adjusted = dict(row)
    utility = _clamp(row.get("estimated_utility"))
    risk = _clamp(row.get("predicted_risk"))
    observations = dict(state.get("operator_observations") or {})
    dependencies = [
        str(value)
        for value in row.get("requires_executed_operators") or []
        if str(value)
    ]
    dependency_yield = 0
    dependency_scan_rows = 0
    for dependency in dependencies:
        observed = dict(observations.get(dependency) or {})
        dependency_yield += max(
            _as_int(observed.get("bind_output_count")),
            _as_int(observed.get("terminal_candidate_count")),
            _as_int(observed.get("candidate_count")),
        )
        dependency_scan_rows += _as_int(observed.get("observed_scan_rows"))
    adjustment_reason = "catalog_estimate"
    if dependencies and all(dependency in observations for dependency in dependencies):
        if dependency_yield <= 0:
            utility *= 0.15
            risk = min(1.0, risk + 0.08)
            adjustment_reason = "empty_dependency_output"
        else:
            productivity = dependency_yield / float(max(1, dependency_scan_rows))
            utility = min(
                1.0,
                utility
                * (
                    1.0
                    + min(0.30, 0.08 * math.log1p(dependency_yield))
                    + min(0.12, productivity)
                ),
            )
            risk = max(0.0, risk - min(0.08, 0.02 * dependency_yield))
            adjustment_reason = "observed_dependency_cardinality"
    adjusted["estimated_utility"] = _round(utility)
    adjusted["predicted_risk"] = _round(risk)
    adjusted["runtime_adjustment_reason"] = adjustment_reason
    adjusted["observed_dependency_yield"] = dependency_yield
    adjusted["observed_dependency_scan_rows"] = dependency_scan_rows
    return adjusted


def enumerate_risk_bounded_plans(
    alternatives: Sequence[Mapping[str, Any]],
    *,
    state: Mapping[str, Any],
    executed_operators: Iterable[str] = (),
    horizon: int = 2,
    beam_width: int = 8,
    token_budget: int = 900,
    compute_budget: float = 1.0,
    max_plan_risk: float = 0.55,
    default_risk_uncertainty: float = 0.0,
    token_weight: float = 0.12,
    compute_weight: float = 0.10,
) -> List[Dict[str, Any]]:
    """Enumerate dependency-valid plans under hard risk and resource bounds.

    Risk is composed as a union probability upper bound.  Benefit uses
    diminishing returns, so two redundant operators do not receive twice the
    credit.  The function is deterministic and executes no callbacks.
    """
    horizon = max(1, int(horizon))
    beam_width = max(1, int(beam_width))
    token_budget = max(1, int(token_budget))
    compute_budget = max(0.0, float(compute_budget))
    max_plan_risk = _clamp(max_plan_risk)
    executed = {str(value) for value in executed_operators if str(value)}
    cumulative_tokens = _as_int(state.get("cumulative_tokens"))
    cumulative_compute = _as_float(state.get("cumulative_compute_cost"))
    residual_multiplier = 1.0 if bool(state.get("residual_deficit")) else 0.20

    rows = {
        str(row.get("operator") or ""): _runtime_adjusted_operator(
            row,
            state=state,
        )
        for row in alternatives
        if str(row.get("operator") or "")
        and not bool(row.get("terminal_value_veto"))
    }
    frontier: List[Dict[str, Any]] = [{
        "operators": [],
        "expected_gain": 0.0,
        "risk_upper": 0.0,
        "token_count": 0,
        "compute_cost": 0.0,
        "scan_rows": 0,
        "objective": 0.0,
        "steps": [],
    }]
    completed: List[Dict[str, Any]] = []

    for _depth in range(horizon):
        expanded: List[Dict[str, Any]] = []
        for plan in frontier:
            selected = set(plan["operators"])
            available = executed | selected
            for name, row in rows.items():
                if name in selected or name in executed:
                    continue
                dependencies = {
                    str(value)
                    for value in row.get("requires_executed_operators") or []
                    if str(value)
                }
                if not dependencies.issubset(available):
                    continue
                step_tokens = _as_int(row.get("estimated_tokens"))
                step_compute = _as_float(row.get("estimated_compute_cost"))
                token_count = _as_int(plan["token_count"]) + step_tokens
                compute_cost = _as_float(plan["compute_cost"]) + step_compute
                if cumulative_tokens + token_count > token_budget:
                    continue
                if cumulative_compute + compute_cost > compute_budget:
                    continue
                value_info = dict(row.get("operator_value") or {})
                counterfactual_supported = bool(
                    value_info.get("counterfactual_supported")
                )
                if counterfactual_supported:
                    risk_uncertainty = 0.0
                    step_risk_upper = _clamp(
                        value_info.get("p_answer_harm_upper"), 0.0, 1.0
                    )
                else:
                    risk_uncertainty = max(
                        _clamp(default_risk_uncertainty, 0.0, 0.5),
                        _clamp(row.get("risk_uncertainty"), 0.0, 0.5),
                    )
                    step_risk_upper = min(
                        1.0,
                        _clamp(row.get("predicted_risk")) + risk_uncertainty,
                    )
                risk_upper = 1.0 - (
                    (1.0 - _as_float(plan["risk_upper"]))
                    * (1.0 - step_risk_upper)
                )
                if risk_upper > max_plan_risk:
                    continue
                utility_multiplier = 1.0
                if counterfactual_supported:
                    step_gain = _clamp(
                        residual_multiplier
                        * (
                            max(
                                0.0,
                                _as_float(value_info.get("expected_quality_delta")),
                            )
                            + _clamp(value_info.get("p_answer_improve"))
                            * _clamp(row.get("estimated_utility"))
                        )
                    )
                elif bool(value_info.get("supported")) and not bool(
                    value_info.get("monotone_veto_only")
                ):
                    utility_multiplier = max(
                        0.0,
                        _as_float(value_info.get("utility_multiplier"), 1.0),
                    )
                    step_gain = _clamp(
                        residual_multiplier
                        * _clamp(row.get("estimated_utility"))
                        * utility_multiplier
                    )
                else:
                    step_gain = _clamp(
                        residual_multiplier
                        * _clamp(row.get("estimated_utility"))
                    )
                expected_gain = 1.0 - (
                    (1.0 - _as_float(plan["expected_gain"]))
                    * (1.0 - step_gain)
                )
                objective = (
                    expected_gain
                    - float(token_weight) * token_count / float(token_budget)
                    - float(compute_weight) * compute_cost
                )
                next_plan = {
                    "operators": list(plan["operators"]) + [name],
                    "expected_gain": _round(expected_gain),
                    "risk_upper": _round(risk_upper),
                    "token_count": token_count,
                    "compute_cost": _round(compute_cost),
                    "scan_rows": _as_int(plan["scan_rows"])
                    + _as_int(row.get("estimated_scan_rows")),
                    "objective": _round(objective),
                    "steps": list(plan["steps"]) + [{
                        "operator": name,
                        "estimated_utility": _round(row.get("estimated_utility")),
                        "predicted_risk": _round(row.get("predicted_risk")),
                        "risk_upper": _round(step_risk_upper),
                        "runtime_adjustment_reason": str(
                            row.get("runtime_adjustment_reason") or ""
                        ),
                    }],
                }
                next_plan["plan_expression"] = " -> ".join(
                    next_plan["operators"]
                )
                expanded.append(next_plan)
                completed.append(next_plan)
        expanded.sort(
            key=lambda plan: (
                _as_float(plan.get("objective")),
                _as_float(plan.get("expected_gain")),
                -_as_float(plan.get("risk_upper")),
                -_as_int(plan.get("scan_rows")),
                str(plan.get("plan_expression") or ""),
            ),
            reverse=True,
        )
        frontier = expanded[:beam_width]
        if not frontier:
            break

    completed.sort(
        key=lambda plan: (
            _as_float(plan.get("objective")),
            _as_float(plan.get("expected_gain")),
            -_as_float(plan.get("risk_upper")),
            -_as_int(plan.get("scan_rows")),
            str(plan.get("plan_expression") or ""),
        ),
        reverse=True,
    )
    return completed[:beam_width]


def execute_adaptive_query_plan(
    *,
    question: str,
    base_docs: Sequence[str],
    operators: Sequence[Mapping[str, Any]],
    bridge_deficit: bool,
    max_steps: int = 2,
    token_budget: int = 900,
    compute_budget: float = 1.0,
    min_objective: float = 0.0,
    max_predicted_risk: float = 0.55,
    token_weight: float = 0.12,
    compute_weight: float = 0.10,
    risk_weight: float = 0.42,
    risk_context: Mapping[str, Any] | None = None,
    stop_risk_assessor: Callable[
        [Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[str]],
        Mapping[str, Any],
    ] | None = None,
    incumbent_preserving_admission: bool = False,
    incumbent_admission_shadow_only: bool = True,
    max_skip_harm_probability: float = 0.025,
    max_reader_change_probability: float = 0.12,
    operator_value_assessor: Callable[
        [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
    ] | None = None,
    min_fallback_productivity: float = 0.12,
    enable_risk_bounded_optimizer: bool = False,
    risk_bounded_plan_horizon: int = 2,
    risk_bounded_beam_width: int = 8,
    risk_bounded_uncertainty: float = 0.0,
    risk_bounded_max_plan_risk: float | None = None,
    logical_plan: Mapping[str, Any] | None = None,
    required_plan_operators: Sequence[str] = (),
    plan_execution_policy: Mapping[str, Any] | None = None,
) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, Any]]:
    """Choose and execute one physical operator at a time.

    Each operator owns a callback, so unselected alternatives are never
    invoked.  A terminal candidate is not assumed in advance: it is learned
    from the operator's observed output before the next optimization round.
    """
    max_steps = max(0, int(max_steps))
    token_budget = max(1, int(token_budget))
    compute_budget = max(0.0, float(compute_budget))
    public_specs = [_public_operator(spec) for spec in operators]
    enabled_names = {
        str(spec.get("operator"))
        for spec in operators
        if bool(spec.get("enabled"))
    }
    logical_plan_summary = dict(logical_plan or {})
    physical_policy = dict(plan_execution_policy or {})
    initial_plan_obligations = list(dict.fromkeys(
        str(name)
        for name in required_plan_operators
        if str(name)
    ))
    pending_plan_obligations = list(initial_plan_obligations)
    state = {
        "query_coverage": _round(lexical_query_coverage(question, base_docs)),
        "evidence_residual_deficit": bool(bridge_deficit),
        "logical_plan_compiled": bool(
            logical_plan_summary.get("compile_status") == "compiled"
        ),
        "logical_plan_kind": str(logical_plan_summary.get("plan_kind") or ""),
        "pending_plan_operators": list(pending_plan_obligations),
        "completed_plan_operators": [],
        "attempted_plan_operators": [],
        "incomplete_plan_operators": {},
        "residual_deficit": bool(
            bridge_deficit or pending_plan_obligations
        ),
        "candidate_available": False,
        "materialized": False,
        "cumulative_tokens": 0,
        "cumulative_compute_cost": 0.0,
        "cumulative_scan_rows": 0,
        "operator_observations": {},
        "schema_bind_count": 0,
        "conditional_plan_fallback_required": False,
    }
    outputs: Dict[str, Mapping[str, Any]] = {}
    executed: set[str] = set()
    rounds: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []
    stop_admission_decisions: List[Dict[str, Any]] = []
    forced_next_operator = ""
    stop_reason = "max_steps_reached" if max_steps == 0 else "not_started"
    max_plan_risk = (
        float(max_predicted_risk)
        if risk_bounded_max_plan_risk is None
        else float(risk_bounded_max_plan_risk)
    )

    def assess_stop(candidate_reason: str) -> Dict[str, Any]:
        remaining = [
            spec
            for spec in operators
            if bool(spec.get("enabled"))
            and str(spec.get("operator") or "") not in executed
        ]
        public_remaining = [_public_operator(spec) for spec in remaining]
        if callable(stop_risk_assessor):
            assessment = dict(
                stop_risk_assessor(
                    dict(state),
                    public_remaining,
                    [step["operator"] for step in steps],
                )
                or {}
            )
        else:
            assessment = {
                "enabled": False,
                "supported": False,
                "reason": "risk_assessor_unavailable",
                "p_reader_change": 0.0,
                "p_harm": 0.0,
                "p_harm_upper": 0.0,
                "confidence": 0.0,
            }
        harm_breach = bool(
            assessment.get("supported")
            and _as_float(assessment.get("p_harm_upper"))
            > float(max_skip_harm_probability)
        )
        change_breach = bool(
            assessment.get("supported")
            and _as_float(assessment.get("p_reader_change"))
            > float(max_reader_change_probability)
        )
        force_incumbent = bool(
            incumbent_preserving_admission
            and remaining
            and (harm_breach or change_breach)
            and not incumbent_admission_shadow_only
        )
        selected_fallback = ""
        fallback_value_assessment: Dict[str, Any] = {}
        fallback_productivity_vetoed = False
        if force_incumbent:
            fallback_spec = min(
                remaining,
                key=lambda spec: (
                    _as_int(spec.get("incumbent_priority"), 100),
                    str(spec.get("operator") or ""),
                ),
            )
            if callable(operator_value_assessor):
                fallback_value_assessment = dict(
                    operator_value_assessor(dict(state), _public_operator(fallback_spec))
                    or {}
                )
                fallback_productivity_vetoed = bool(
                    fallback_value_assessment.get("supported")
                    and (
                        bool(fallback_value_assessment.get("terminal_veto"))
                        or _as_float(
                            fallback_value_assessment.get("expected_productivity")
                        ) < float(min_fallback_productivity)
                    )
                )
            if not fallback_productivity_vetoed:
                selected_fallback = str(fallback_spec.get("operator") or "")
            else:
                force_incumbent = False
        decision = {
            "candidate_stop_reason": str(candidate_reason),
            "state_at_stop": dict(state),
            "executed_operators": [step["operator"] for step in steps],
            "remaining_operators": _operator_names(public_remaining),
            "assessment": assessment,
            "max_skip_harm_probability": _round(max_skip_harm_probability),
            "max_reader_change_probability": _round(max_reader_change_probability),
            "harm_bound_breached": harm_breach,
            "reader_change_bound_breached": change_breach,
            "shadow_only": bool(incumbent_admission_shadow_only),
            "force_incumbent": force_incumbent,
            "selected_fallback_operator": selected_fallback,
            "fallback_value_assessment": fallback_value_assessment,
            "fallback_productivity_vetoed": fallback_productivity_vetoed,
        }
        stop_admission_decisions.append(decision)
        return decision

    for round_index in range(max_steps):
        alternatives: List[Dict[str, Any]] = []
        for spec in operators:
            name = str(spec.get("operator") or "")
            if not name or name in executed or not bool(spec.get("enabled")):
                continue
            required_operators = {
                str(value)
                for value in spec.get("requires_executed_operators") or []
                if str(value)
            }
            dependencies_satisfied = required_operators.issubset(executed)
            projected_compute = _as_float(state["cumulative_compute_cost"]) + _as_float(
                spec.get("estimated_compute_cost")
            )
            projected_tokens = _as_int(state["cumulative_tokens"]) + _as_int(
                spec.get("estimated_tokens")
            )
            if callable(operator_value_assessor):
                value_assessment = dict(
                    operator_value_assessor(dict(state), _public_operator(spec)) or {}
                )
            else:
                value_assessment = {
                    "enabled": False,
                    "supported": False,
                    "reason": "operator_value_assessor_unavailable",
                    "utility_multiplier": 1.0,
                }
            value_supported = bool(value_assessment.get("supported"))
            counterfactual_supported = bool(
                value_assessment.get("counterfactual_supported")
            )
            base_predicted_risk = _clamp(spec.get("predicted_risk"))
            effective_predicted_risk = (
                _clamp(value_assessment.get("p_answer_harm_upper"))
                if counterfactual_supported
                else base_predicted_risk
            )
            effective_spec = {
                **spec,
                "predicted_risk": effective_predicted_risk,
                "risk_uncertainty": (
                    0.0
                    if counterfactual_supported
                    else spec.get("risk_uncertainty", 0.0)
                ),
            }
            resource_feasible = (
                dependencies_satisfied
                and effective_predicted_risk <= float(max_predicted_risk)
                and projected_compute <= compute_budget
                and projected_tokens <= token_budget
            )
            monotone_veto_only = bool(
                value_supported
                and value_assessment.get("monotone_veto_only")
            )
            terminal_value_veto = bool(
                monotone_veto_only and value_assessment.get("terminal_veto")
            )
            decision_conditional_top_only = bool(
                terminal_value_veto
                and value_assessment.get("decision_conditional_top_only")
            )
            utility_multiplier = (
                _as_float(
                    value_assessment.get("counterfactual_utility_multiplier"),
                    1.0,
                )
                if counterfactual_supported
                else
                _as_float(value_assessment.get("utility_multiplier"), 1.0)
                if value_supported and not monotone_veto_only
                else 1.0
            )
            base_objective = _objective(
                effective_spec,
                residual_deficit=bool(state["residual_deficit"]),
                token_budget=token_budget,
                token_weight=token_weight,
                compute_weight=compute_weight,
                risk_weight=risk_weight,
            )
            objective = base_objective if monotone_veto_only else _objective(
                    effective_spec,
                    residual_deficit=bool(state["residual_deficit"]),
                    token_budget=token_budget,
                    token_weight=token_weight,
                    compute_weight=compute_weight,
                    risk_weight=risk_weight,
                    utility_multiplier=utility_multiplier,
                )




            feasible = bool(
                resource_feasible
                and (not terminal_value_veto or decision_conditional_top_only)
            )
            alternatives.append({
                **_public_operator(effective_spec),
                "base_predicted_risk": _round(base_predicted_risk),
                "counterfactual_risk_calibrated": counterfactual_supported,
                "base_objective": _round(base_objective),
                "objective": _round(objective),
                "feasible": feasible,
                "resource_feasible": resource_feasible,
                "dependencies_satisfied": dependencies_satisfied,
                "missing_required_operators": sorted(
                    required_operators - executed
                ),
                "operator_value": value_assessment,
                "value_calibrated": value_supported,
                "terminal_value_veto": terminal_value_veto,
                "decision_conditional_top_only": (
                    decision_conditional_top_only
                ),
                "decision_frontier_veto": False,
                "value_induced_veto": bool(
                    resource_feasible
                    and base_objective >= float(min_objective)
                    and not decision_conditional_top_only
                    and (
                        terminal_value_veto
                        or objective < float(min_objective)
                    )
                ),
            })

        feasible = [
            row
            for row in alternatives
            if row["feasible"] and float(row["objective"]) >= float(min_objective)
        ]
        feasible.sort(
            key=lambda row: (
                float(row["objective"]),
                -float(row.get("predicted_risk", 0.0)),
                -int(row.get("estimated_scan_rows", 0)),
                str(row.get("operator") or ""),
            ),
            reverse=True,
        )
        risk_bounded_plans: List[Dict[str, Any]] = []
        selected_risk_bounded_plan: Dict[str, Any] = {}
        if enable_risk_bounded_optimizer and not forced_next_operator:
            risk_bounded_plans = enumerate_risk_bounded_plans(
                alternatives,
                state=state,
                executed_operators=executed,
                horizon=min(
                    max(1, int(risk_bounded_plan_horizon)),
                    max(1, max_steps - round_index),
                ),
                beam_width=risk_bounded_beam_width,
                token_budget=token_budget,
                compute_budget=compute_budget,
                max_plan_risk=max_plan_risk,
                default_risk_uncertainty=risk_bounded_uncertainty,
                token_weight=token_weight,
                compute_weight=compute_weight,
            )
            selected_risk_bounded_plan = (
                dict(risk_bounded_plans[0]) if risk_bounded_plans else {}
            )
        decision_frontier_veto = False
        if (
            not forced_next_operator
            and feasible
            and bool(feasible[0].get("decision_conditional_top_only"))
            and bool(feasible[0].get("terminal_value_veto"))
        ):
            frontier = feasible[0]
            frontier["feasible"] = False
            frontier["value_induced_veto"] = True
            frontier["decision_frontier_veto"] = True
            decision_frontier_veto = True
            feasible = []
        if forced_next_operator:
            selected_name = forced_next_operator
            forced_next_operator = ""
            selection_reason = "incumbent_preserving_fallback"
        elif enable_risk_bounded_optimizer:
            if (
                selected_risk_bounded_plan
                and float(selected_risk_bounded_plan.get("objective", 0.0))
                >= float(min_objective)
            ):
                selected_name = str(
                    (selected_risk_bounded_plan.get("operators") or ["NO_OP"])[0]
                )
                selection_reason = "risk_bounded_plan_frontier"
            else:
                selected_name = "NO_OP"
                selection_reason = "no_feasible_risk_bounded_plan"
        else:
            selected_name = str(feasible[0]["operator"]) if feasible else "NO_OP"
            selection_reason = (
                "max_objective"
                if feasible
                else "decision_frontier_value_veto"
                if decision_frontier_veto
                else "no_feasible_operator"
            )
        round_trace = {
            "round": round_index,
            "state_before": dict(state),
            "alternatives": alternatives,
            "selected_operator": selected_name,
            "selection_reason": selection_reason,
            "risk_bounded_plan_candidates": risk_bounded_plans,
            "selected_risk_bounded_plan": selected_risk_bounded_plan,
        }
        if selected_name == "NO_OP":
            decision = assess_stop("no_feasible_positive_operator")
            fallback = str(decision.get("selected_fallback_operator") or "")
            if fallback:
                selected_name = fallback
                round_trace["selected_operator"] = selected_name
                round_trace["selection_reason"] = "incumbent_preserving_fallback"
                round_trace["stop_admission"] = decision
            else:
                stop_reason = "no_feasible_positive_operator"
                round_trace["stop_admission"] = decision
                round_trace["observation"] = {}
                round_trace["state_after"] = dict(state)
                rounds.append(round_trace)
                break

        selected = next(
            spec for spec in operators if str(spec.get("operator")) == selected_name
        )
        selected_alternative = next(
            (
                row
                for row in alternatives
                if str(row.get("operator") or "") == selected_name
            ),
            {},
        )
        started = time.perf_counter()
        result = selected["execute"]()
        latency_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        info = dict(result.get("info") or {})
        observed = _observation(info, selected)
        observed["latency_ms"] = _round(latency_ms)
        outputs[str(selected.get("source") or selected_name)] = result
        executed.add(selected_name)

        state["candidate_available"] = bool(state["candidate_available"]) or (
            observed["candidate_count"] > 0
        )
        state["materialized"] = bool(state["materialized"]) or bool(
            observed["applied"]
        )
        if observed["resolved_residual"]:
            state["evidence_residual_deficit"] = False
        if selected_name in pending_plan_obligations:
            state["attempted_plan_operators"] = list(dict.fromkeys(
                list(state.get("attempted_plan_operators") or [])
                + [selected_name]
            ))
            obligation_satisfied = observed.get("plan_obligation_satisfied")



            if obligation_satisfied is not False:
                pending_plan_obligations.remove(selected_name)
                state["completed_plan_operators"] = list(dict.fromkeys(
                    list(state.get("completed_plan_operators") or [])
                    + [selected_name]
                ))
                state["incomplete_plan_operators"] = {
                    key: value
                    for key, value in dict(
                        state.get("incomplete_plan_operators") or {}
                    ).items()
                    if key != selected_name
                }
            else:
                state["incomplete_plan_operators"] = {
                    **dict(state.get("incomplete_plan_operators") or {}),
                    selected_name: dict(
                        observed.get("plan_obligation_assessment") or {}
                    ),
                }
        state["pending_plan_operators"] = list(pending_plan_obligations)
        state["residual_deficit"] = bool(
            state["evidence_residual_deficit"]
            or pending_plan_obligations
        )
        state["cumulative_tokens"] = _as_int(state["cumulative_tokens"]) + _as_int(
            selected.get("estimated_tokens")
        )
        state["cumulative_compute_cost"] = _round(
            _as_float(state["cumulative_compute_cost"])
            + _as_float(selected.get("estimated_compute_cost"))
        )
        state["cumulative_scan_rows"] = _as_int(state["cumulative_scan_rows"]) + _as_int(
            observed.get("observed_scan_rows")
        )
        state["operator_observations"] = {
            **dict(state.get("operator_observations") or {}),
            selected_name: dict(observed),
        }
        if selected_name == "SCHEMA_PROBE":
            state["schema_bind_count"] = max(
                _as_int(observed.get("bind_output_count")),
                _as_int(observed.get("terminal_candidate_count")),
            )
        seed_operator = str(physical_policy.get("physical_seed_operator") or "")
        fallback_operator = str(
            physical_policy.get("conditional_fallback_operator") or ""
        )
        if (
            selected_name == seed_operator
            and fallback_operator
            and fallback_operator not in executed
            and not bool(observed.get("applied"))
            and _as_int(observed.get("terminal_candidate_count")) <= 0
        ):
            state["evidence_residual_deficit"] = True
            state["conditional_plan_fallback_required"] = True
        elif selected_name == fallback_operator:
            state["conditional_plan_fallback_required"] = False
        state["residual_deficit"] = bool(
            state["evidence_residual_deficit"]
            or pending_plan_obligations
        )
        round_trace["observation"] = observed
        round_trace["state_after"] = dict(state)
        rounds.append(round_trace)
        steps.append({
            "step_id": f"runtime_step_{len(steps)}",
            "operator": selected_name,
            "source": str(selected.get("source") or ""),
            "access_path": str(selected.get("access_path") or ""),
            "predicted_risk": _round(selected.get("predicted_risk")),
            "estimated_utility": _round(selected.get("estimated_utility")),
            "base_objective": _round(selected_alternative.get("base_objective")),
            "objective": _round(selected_alternative.get("objective")),
            "operator_value": dict(
                selected_alternative.get("operator_value") or {}
            ),
            "estimated_tokens": _as_int(selected.get("estimated_tokens")),
            "estimated_compute_cost": _round(
                selected.get("estimated_compute_cost")
            ),
            "estimated_scan_rows": _as_int(selected.get("estimated_scan_rows")),
            "observed_scan_rows": _as_int(observed.get("observed_scan_rows")),
            "deduplicated_rows": _as_int(observed.get("deduplicated_rows")),
            "duplicate_rows": _as_int(observed.get("duplicate_rows")),
            "fanout_rejected_keys": _as_int(observed.get("fanout_rejected_keys")),
            "fanout_rejected_rows": _as_int(observed.get("fanout_rejected_rows")),
            "truncated_rows": _as_int(observed.get("truncated_rows")),
            "loaded_doc_rows": _as_int(observed.get("loaded_doc_rows")),
            "upstream_bind_input_count": _as_int(
                observed.get("upstream_bind_input_count")
            ),
            "upstream_bind_fact_count": _as_int(
                observed.get("upstream_bind_fact_count")
            ),
            "latency_ms": _round(latency_ms),
            "reoptimized_after": True,
            "forced_by_incumbent_admission": (
                round_trace.get("selection_reason")
                == "incumbent_preserving_fallback"
            ),
        })
        if not state["residual_deficit"]:
            decision = assess_stop("residual_resolved")
            fallback = str(decision.get("selected_fallback_operator") or "")
            if fallback and round_index + 1 < max_steps:
                forced_next_operator = fallback
                stop_reason = "incumbent_fallback_pending"
                continue
            stop_reason = (
                "incumbent_fallback_budget_exhausted"
                if fallback
                else "residual_resolved"
            )
            break
        stop_reason = "max_steps_reached"

    expression = "NO_OP"
    if steps:
        expression = "ADAPTIVE(" + " -> REOPT -> ".join(
            str(step["operator"]) for step in steps
        ) + ")"
    trace = {
        "enabled": True,
        "executor_version": EXECUTOR_VERSION,
        "planning_scope": PLANNING_SCOPE,
        "runtime_reoptimization_ready": True,
        "runtime_reoptimization_performed": len(rounds) > 1,
        "materialization_deferred_to_admission": True,
        "selected_plan_expression": expression,
        "stop_reason": stop_reason,
        "max_steps": max_steps,
        "token_budget": token_budget,
        "compute_budget": _round(compute_budget),
        "min_objective": _round(min_objective),
        "max_predicted_risk": _round(max_predicted_risk),
        "risk_bounded_optimizer_enabled": bool(enable_risk_bounded_optimizer),
        "risk_bounded_plan_horizon": max(1, int(risk_bounded_plan_horizon)),
        "risk_bounded_beam_width": max(1, int(risk_bounded_beam_width)),
        "risk_bounded_uncertainty": _round(risk_bounded_uncertainty),
        "risk_bounded_max_plan_risk": _round(max_plan_risk),
        "risk_bounded_plan_decision_count": sum(
            bool(round_trace.get("risk_bounded_plan_candidates"))
            for round_trace in rounds
        ),
        "risk_context": dict(risk_context or {}),
        "logical_plan": logical_plan_summary,
        "logical_plan_execution_policy": physical_policy,
        "logical_plan_obligations": {
            "initial": list(initial_plan_obligations),
            "attempted": list(state.get("attempted_plan_operators") or []),
            "completed": list(state.get("completed_plan_operators") or []),
            "incomplete": dict(state.get("incomplete_plan_operators") or {}),
            "pending": list(pending_plan_obligations),
            "all_satisfied": not bool(pending_plan_obligations),
        },
        "incumbent_preserving_admission": bool(incumbent_preserving_admission),
        "incumbent_admission_shadow_only": bool(incumbent_admission_shadow_only),
        "max_skip_harm_probability": _round(max_skip_harm_probability),
        "max_reader_change_probability": _round(max_reader_change_probability),
        "operator_value_model_enabled": bool(callable(operator_value_assessor)),
        "min_fallback_productivity": _round(min_fallback_productivity),
        "operator_value_decision_count": sum(
            len(round_trace.get("alternatives") or []) for round_trace in rounds
        ),
        "operator_value_supported_count": sum(
            bool(alternative.get("value_calibrated"))
            for round_trace in rounds
            for alternative in round_trace.get("alternatives") or []
        ),
        "operator_value_veto_count": sum(
            bool(alternative.get("value_induced_veto"))
            for round_trace in rounds
            for alternative in round_trace.get("alternatives") or []
        ),
        "counterfactual_risk_supported_count": sum(
            bool((alternative.get("operator_value") or {}).get(
                "counterfactual_supported"
            ))
            for round_trace in rounds
            for alternative in round_trace.get("alternatives") or []
        ),
        "terminal_value_veto_count": sum(
            bool(alternative.get("terminal_value_veto"))
            for round_trace in rounds
            for alternative in round_trace.get("alternatives") or []
        ),
        "decision_frontier_veto_count": sum(
            bool(alternative.get("decision_frontier_veto"))
            for round_trace in rounds
            for alternative in round_trace.get("alternatives") or []
        ),
        "terminal_value_admission_count": sum(
            bool(
                alternative.get("value_calibrated")
                and (alternative.get("operator_value") or {}).get(
                    "monotone_veto_only"
                )
                and not alternative.get("terminal_value_veto")
            )
            for round_trace in rounds
            for alternative in round_trace.get("alternatives") or []
        ),
        "fallback_productivity_veto_count": sum(
            bool(decision.get("fallback_productivity_vetoed"))
            for decision in stop_admission_decisions
        ),
        "stop_admission_decisions": stop_admission_decisions,
        "incumbent_fallback_count": sum(
            bool(step.get("forced_by_incumbent_admission")) for step in steps
        ),
        "initial_operators": public_specs,
        "enabled_operator_count": len(enabled_names),
        "executed_operators": [step["operator"] for step in steps],
        "skipped_operators": sorted(enabled_names - executed),
        "rounds": rounds,
        "steps": steps,
        "final_state": dict(state),
    }
    return outputs, trace
