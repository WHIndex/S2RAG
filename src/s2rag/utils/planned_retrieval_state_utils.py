"""Shared terminal-state semantics for planned retrieval and materialization."""

import re
from typing import Any, Dict, Iterable


PLANNER_SUFFICIENT_REASONS = frozenset({
    "initially_sufficient",
    "verified_sufficient",
})

TERMINAL_EVIDENCE_GAP_REASONS = frozenset({
    "cost_controller_fixed_point_stop",
    "max_rounds_reached",
    "no_new_planned_queries",
    "residual_deficit_query_appended",
    "verifier_insufficient_no_retrieval",
})

OPTIMIZER_STOP_REASONS = frozenset({
    "cost_controller_fixed_point_stop",
})


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _last_verifier(planned_retrieval: Dict[str, Any]) -> Dict[str, Any]:
    direct = _as_dict(
        planned_retrieval.get("verifier")
        or planned_retrieval.get("verification")
    )
    if direct:
        return direct
    rounds = planned_retrieval.get("rounds") or []
    for round_info in reversed(rounds if isinstance(rounds, list) else []):
        verifier = _as_dict(_as_dict(round_info).get("verifier"))
        if verifier:
            return verifier
    return {}


def planned_retrieval_terminal_state(
    planned_retrieval: Any,
) -> Dict[str, Any]:
    """Return reason-independent execution semantics for downstream operators.

    A cost-controller STOP ends evidence expansion, not the query. Downstream
    operators must still run deterministic finalization against the unresolved
    verifier state. Keeping this contract in one place prevents a new stop
    reason from silently bypassing reader-visible materialization.
    """

    planned = _as_dict(planned_retrieval)
    router = _as_dict(planned.get("router"))
    verifier = _last_verifier(planned)
    reason = str(
        planned.get("reason")
        or planned.get("stop_reason")
        or planned.get("final_status")
        or ""
    ).strip()
    router_reason = str(router.get("reason") or "").strip()

    verifier_sufficient = verifier.get("sufficient") is True
    verifier_insufficient = bool(
        verifier.get("sufficient") is False
        or verifier.get("answerable") is False
        or verifier.get("evidence_insufficient")
        or verifier.get("insufficient")
    )
    explicitly_sufficient = bool(
        planned.get("evidence_sufficient") is True
        or reason in PLANNER_SUFFICIENT_REASONS
        or verifier_sufficient
    )
    explicit_gap = planned.get("terminal_evidence_gap")
    if explicit_gap is None:
        terminal_evidence_gap = bool(
            not explicitly_sufficient
            and (
                verifier_insufficient
                or reason in TERMINAL_EVIDENCE_GAP_REASONS
                or router_reason in TERMINAL_EVIDENCE_GAP_REASONS
            )
        )
    else:
        terminal_evidence_gap = bool(explicit_gap and not explicitly_sufficient)

    expansion_stopped = bool(
        planned.get("expansion_stopped_by_optimizer")
        or reason in OPTIMIZER_STOP_REASONS
    )
    materialization_required = bool(
        planned.get("terminal_materialization_required")
        or terminal_evidence_gap
    )
    return {
        "source_reason": reason,
        "router_reason": router_reason,
        "verifier_sufficient": bool(verifier_sufficient),
        "verifier_insufficient": bool(verifier_insufficient),
        "evidence_sufficient": bool(explicitly_sufficient),
        "terminal_evidence_gap": bool(terminal_evidence_gap),
        "expansion_stopped_by_optimizer": bool(expansion_stopped),
        "terminal_materialization_required": bool(materialization_required),
        "next_phase": "FINALIZE" if materialization_required else "ANSWER",
    }


def annotate_planned_retrieval_terminal_state(
    planned_retrieval: Any,
) -> Dict[str, Any]:
    """Annotate a planned-retrieval trace without replacing its source reason."""

    planned = _as_dict(planned_retrieval)
    state = planned_retrieval_terminal_state(planned)
    planned["terminal_state"] = dict(state)
    planned["terminal_evidence_gap"] = state["terminal_evidence_gap"]
    planned["expansion_stopped_by_optimizer"] = state[
        "expansion_stopped_by_optimizer"
    ]
    planned["terminal_materialization_required"] = state[
        "terminal_materialization_required"
    ]
    planned["next_execution_phase"] = state["next_phase"]
    return planned


def _normalized_docs(docs: Iterable[Any]) -> tuple:
    return tuple(
        re.sub(r"\s+", " ", str(doc or "")).strip()
        for doc in docs or []
        if str(doc or "").strip()
    )


def build_terminal_materialization_trace(
    planned_retrieval: Any,
    *,
    source_docs: Iterable[Any],
    reader_docs: Iterable[Any],
    components: Dict[str, Any],
) -> Dict[str, Any]:
    """Record that deterministic answer materialization ran after expansion."""

    state = planned_retrieval_terminal_state(planned_retrieval)
    attempted = []
    applied = []
    component_reasons: Dict[str, str] = {}
    applied_count_fields = (
        "accepted_count",
        "appended_count",
        "inserted_count",
        "selected_count",
        "terminal_tuple_accepted_count",
    )
    for name, raw_info in (components or {}).items():
        info = _as_dict(raw_info)
        if not info:
            continue
        if info.get("enabled", True):
            attempted.append(str(name))
        component_reasons[str(name)] = str(info.get("reason") or "")
        materialized = bool(info.get("applied")) or any(
            int(info.get(field, 0) or 0) > 0 for field in applied_count_fields
        )
        if materialized:
            applied.append(str(name))

    source_signature = _normalized_docs(source_docs)
    reader_signature = _normalized_docs(reader_docs)
    return {
        "required": state["terminal_materialization_required"],
        "completed": True,
        "execution_transition": (
            "STOP_EXPANSION->FINALIZE"
            if state["expansion_stopped_by_optimizer"]
            else f"{state['next_phase']}"
        ),
        "source_reason": state["source_reason"],
        "terminal_evidence_gap": state["terminal_evidence_gap"],
        "expansion_stopped_by_optimizer": state[
            "expansion_stopped_by_optimizer"
        ],
        "attempted_components": attempted,
        "attempted_component_count": len(attempted),
        "applied_components": applied,
        "applied_component_count": len(applied),
        "component_reasons": component_reasons,
        "source_doc_count": len(source_signature),
        "reader_doc_count": len(reader_signature),
        "reader_evidence_changed": source_signature != reader_signature,
    }
