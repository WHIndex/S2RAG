"""Physical-plan IR for adaptive evidence query execution.

The current executor plans over an eagerly generated candidate pool. This
module makes that scope explicit while providing the operator, dependency, and
state-transition representation needed for later lazy execution and runtime
re-optimization.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


PLAN_GRAMMAR_VERSION = 1
PLANNING_SCOPE = "candidate_pool_materialization"


_OPERATOR_SPECS: Dict[str, Dict[str, Any]] = {
    "planned_retrieval": {
        "operator": "RETRIEVE",
        "access_path": "ranked_passage_access",
        "phase": 10,
    },
    "path_evidence": {
        "operator": "PATH_EXPAND",
        "access_path": "graph_path_access",
        "phase": 20,
    },
    "query_schema_memory": {
        "operator": "SCHEMA_PROBE",
        "access_path": "schema_signature_index",
        "phase": 30,
    },
    "relation_fact_table": {
        "operator": "FACT_SCAN",
        "access_path": "relation_fact_index",
        "phase": 35,
    },
    "entity_predicate_fact_join": {
        "operator": "BOUND_FACT_JOIN",
        "access_path": "entity_predicate_index",
        "phase": 40,
    },
    "relation_fact_join": {
        "operator": "RELATION_FACT_JOIN",
        "access_path": "relation_fact_index",
        "phase": 40,
    },
    "typed_fact_join_shadow": {
        "operator": "TYPED_FACT_JOIN",
        "access_path": "typed_fact_index",
        "phase": 40,
    },
    "selective_support_join": {
        "operator": "SUPPORT_FACT_JOIN",
        "access_path": "support_fact_index",
        "phase": 45,
    },
    "support_set_completion": {
        "operator": "SUPPORT_COMPLETE",
        "access_path": "support_set_view",
        "phase": 50,
    },
    "dual_channel_projection": {
        "operator": "PROJECT",
        "access_path": "answer_slot_projection",
        "phase": 60,
    },
}


_JOIN_OPERATORS = {
    "BOUND_FACT_JOIN",
    "RELATION_FACT_JOIN",
    "TYPED_FACT_JOIN",
    "SUPPORT_FACT_JOIN",
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


def _round(value: Any) -> float:
    return round(_as_float(value, 0.0), 6)


def _candidate_id(row: Mapping[str, Any]) -> str:
    return str(row.get("candidate_id") or "")


def _slot_ids(row: Mapping[str, Any]) -> Set[str]:
    slots = {
        str(value)
        for value in row.get("covered_slot_ids") or []
        if str(value)
    }
    slots.update(
        str(value)
        for value in row.get("uncovered_slot_ids") or []
        if str(value)
    )
    return slots


def _operator_spec(row: Mapping[str, Any]) -> Dict[str, Any]:
    source = str(row.get("source") or "extra_evidence")
    spec = dict(
        _OPERATOR_SPECS.get(
            source,
            {
                "operator": "EVIDENCE_SCAN",
                "access_path": "materialized_candidate_pool",
                "phase": 25,
            },
        )
    )
    spec["source"] = source
    return spec


def _explicit_dependencies(row: Mapping[str, Any]) -> Set[str]:
    dependencies = row.get("depends_on_candidate_ids") or []
    if isinstance(dependencies, str):
        dependencies = [dependencies]
    return {str(value) for value in dependencies if str(value)}


def _order_ready_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Topologically order rows, using access selectivity as join order."""
    pending = {_candidate_id(row): row for row in rows if _candidate_id(row)}
    available_ids = set(pending)
    emitted: Set[str] = set()
    ordered: List[Mapping[str, Any]] = []
    while pending:
        ready = [
            row
            for row in pending.values()
            if not (_explicit_dependencies(row) & available_ids) - emitted
        ]
        if not ready:


            ready = list(pending.values())
        ready.sort(
            key=lambda row: (
                _as_int(_operator_spec(row).get("phase"), 100),
                _as_int(row.get("lineage_depth"), 0),
                _as_int(
                    row.get("estimated_endpoint_scan_rows"),
                    _as_int(row.get("schema_frequency"), 1),
                ),
                -_as_float(row.get("schema_selectivity"), 0.0),
                -_as_float(row.get("coverage_gain"), 0.0),
                _candidate_id(row),
            )
        )
        selected = ready[0]
        selected_id = _candidate_id(selected)
        ordered.append(selected)
        emitted.add(selected_id)
        pending.pop(selected_id, None)
    return ordered


def _plan_expression(steps: Sequence[Mapping[str, Any]]) -> str:
    if not steps or str(steps[0].get("operator") or "") == "NO_OP":
        return "NO_OP"
    operators = [
        f"{step['operator']}[{step['candidate_id']}]"
        for step in steps
        if step.get("candidate_id")
    ]
    inner = operators[0] if len(operators) == 1 else f"SEQ({', '.join(operators)})"
    return f"SAFE_MATERIALIZE({inner})"


def _plan_fingerprint(plan: Mapping[str, Any]) -> str:
    payload = {
        "grammar_version": plan.get("grammar_version"),
        "planning_scope": plan.get("planning_scope"),
        "expression": plan.get("expression"),
        "steps": [
            {
                "operator": step.get("operator"),
                "candidate_id": step.get("candidate_id"),
                "dependencies": step.get("dependencies"),
            }
            for step in plan.get("steps") or []
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()[:16]


def compile_physical_evidence_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_slot_ids: Iterable[str] = (),
    step_metrics: Mapping[str, Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Compile selected candidates into a dependency-aware physical-plan IR.

    The output is observational in the current executor: candidates have
    already been generated. ``runtime_reoptimization_ready`` remains false
    until candidate generation itself is made lazy and driven by this IR.
    """
    required = {str(value) for value in required_slot_ids if str(value)}
    step_metrics = step_metrics or {}
    selected_rows = [dict(row) for row in rows if _candidate_id(row)]
    if not selected_rows:
        plan = {
            "grammar_version": PLAN_GRAMMAR_VERSION,
            "planning_scope": PLANNING_SCOPE,
            "runtime_reoptimization_ready": False,
            "expression": "NO_OP",
            "steps": [
                {
                    "step_id": "step_0",
                    "operator": "NO_OP",
                    "candidate_id": "",
                    "source": "",
                    "access_path": "none",
                    "dependencies": [],
                    "reoptimize_after": False,
                }
            ],
            "state_transitions": [],
            "join_order_candidate_ids": [],
            "required_slot_ids": sorted(required),
            "covered_slot_ids": [],
            "remaining_slot_ids": sorted(required),
            "validation": {"valid": True, "warnings": []},
        }
        plan["plan_fingerprint"] = _plan_fingerprint(plan)
        return plan

    known_ids = {_candidate_id(row) for row in selected_rows}
    ordered = _order_ready_rows(selected_rows)
    covered: Set[str] = set()
    cumulative_tokens = 0
    cumulative_scan_rows = 0
    cumulative_compute_cost = 0.0
    steps: List[Dict[str, Any]] = []
    transitions: List[Dict[str, Any]] = []
    warnings: List[str] = []
    emitted_ids: Set[str] = set()
    join_order: List[str] = []

    for index, row in enumerate(ordered):
        candidate_id = _candidate_id(row)
        spec = _operator_spec(row)
        operator = str(spec["operator"])
        dependencies = _explicit_dependencies(row)
        missing_dependencies = sorted(dependencies - known_ids)
        not_ready_dependencies = sorted((dependencies & known_ids) - emitted_ids)
        if missing_dependencies:
            warnings.append(
                f"{candidate_id}:missing_dependencies={','.join(missing_dependencies)}"
            )
        if not_ready_dependencies:
            warnings.append(
                f"{candidate_id}:dependency_order_violation="
                f"{','.join(not_ready_dependencies)}"
            )
        binding_signals = sum(
            [
                _as_float(row.get("join_key_overlap"), 0.0) > 0.0,
                bool(row.get("lineage_bound")),
                bool(str(row.get("active_entity") or "")),
                bool(str(row.get("root_entity") or "")),
                bool(row.get("matched_join_tokens")),
            ]
        )
        if operator in _JOIN_OPERATORS and binding_signals == 0:
            warnings.append(f"{candidate_id}:join_without_observed_binding")

        metrics = dict(step_metrics.get(candidate_id) or {})
        tokens = max(0, _as_int(metrics.get("tokens"), _as_int(row.get("token_count"), 0)))
        scan_rows = max(
            0,
            _as_int(
                metrics.get("scan_rows"),
                _as_int(row.get("estimated_endpoint_scan_rows"), 0),
            ),
        )
        compute_cost = max(
            0.0,
            _as_float(metrics.get("compute_cost"), _as_float(row.get("compute_cost"), 0.0)),
        )
        candidate_slots = _slot_ids(row)
        before_covered = set(covered)
        before_remaining = required - before_covered
        newly_covered = candidate_slots - before_covered
        covered.update(candidate_slots)
        cumulative_tokens += tokens
        cumulative_scan_rows += scan_rows
        cumulative_compute_cost += compute_cost

        steps.append(
            {
                "step_id": f"step_{index}",
                "operator": operator,
                "candidate_id": candidate_id,
                "source": str(spec["source"]),
                "access_path": str(spec["access_path"]),
                "dependencies": sorted(dependencies),
                "estimated_tokens": tokens,
                "estimated_scan_rows": scan_rows,
                "estimated_compute_cost": _round(compute_cost),
                "predicted_risk": _round(row.get("predicted_harm_probability", row.get("risk", 0.0))),
                "predicate": str(row.get("predicate") or ""),
                "join_key": str(row.get("active_entity") or row.get("root_entity") or ""),
                "reoptimize_after": True,
            }
        )
        transitions.append(
            {
                "after_step_id": f"step_{index}",
                "covered_slot_ids_before": sorted(before_covered),
                "newly_covered_slot_ids": sorted(newly_covered),
                "covered_slot_ids_after": sorted(covered),
                "remaining_slot_ids_before": sorted(before_remaining),
                "remaining_slot_ids_after": sorted(required - covered),
                "cumulative_tokens": cumulative_tokens,
                "cumulative_scan_rows": cumulative_scan_rows,
                "cumulative_compute_cost": _round(cumulative_compute_cost),
            }
        )
        if operator in _JOIN_OPERATORS:
            join_order.append(candidate_id)
        emitted_ids.add(candidate_id)

    materialize_step_id = f"step_{len(steps)}"
    steps.append(
        {
            "step_id": materialize_step_id,
            "operator": "SAFE_MATERIALIZE",
            "candidate_id": "",
            "source": "",
            "access_path": "reader_visible_evidence_view",
            "dependencies": [_candidate_id(row) for row in ordered],
            "estimated_tokens": 0,
            "estimated_scan_rows": 0,
            "estimated_compute_cost": 0.0,
            "predicted_risk": 0.0,
            "predicate": "",
            "join_key": "",
            "reoptimize_after": False,
        }
    )
    plan = {
        "grammar_version": PLAN_GRAMMAR_VERSION,
        "planning_scope": PLANNING_SCOPE,
        "runtime_reoptimization_ready": False,
        "expression": _plan_expression(steps),
        "steps": steps,
        "state_transitions": transitions,
        "reoptimization_points": [
            transition["after_step_id"] for transition in transitions
        ],
        "join_order_candidate_ids": join_order,
        "required_slot_ids": sorted(required),
        "covered_slot_ids": sorted(covered),
        "remaining_slot_ids": sorted(required - covered),
        "estimated_tokens": cumulative_tokens,
        "estimated_scan_rows": cumulative_scan_rows,
        "estimated_compute_cost": _round(cumulative_compute_cost),
        "validation": {"valid": not warnings, "warnings": warnings},
    }
    plan["plan_fingerprint"] = _plan_fingerprint(plan)
    return plan
