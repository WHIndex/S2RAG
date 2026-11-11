"""Minimal stateful lazy executor used as a competitive control.

The executor intentionally shares physical access and correctness contracts
with SAQE, but not SAQE's action frontier, invalidation, reopen, or scheduling
state machine.  It verifies a candidate before committing its output binding,
keeps a query-local recovery overlay, and explores candidates in stable source
order.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from .evidence_query_execution_utils import (
    COMPLETE,
    INSUFFICIENT,
    LOOKUP_JOIN,
    SUPPORTED,
    CostLedger,
    EvidenceBackend,
    ExecutionBudget,
    _as_int,
    _binding_to_obligation_contract,
    _clean_root_answer_surface,
    _json_safe,
    _key,
    _normalise_cost,
    _normalise_fact,
    _reduce_witnesses,
    _register_fact_usage,
    _structured_witnesses,
)


SIMPLE_STATEFUL_LAZY_VERSION = "simple_stateful_lazy_v1"


def _budget_allows(
    ledger: CostLedger,
    budget: ExecutionBudget,
    *,
    may_retrieve: bool,
) -> bool:
    if ledger.actions >= budget.max_actions:
        return False
    if ledger.scan_rows >= budget.max_scan_rows:
        return False
    if may_retrieve and ledger.retrieval_calls >= budget.max_retrieval_calls:
        return False
    return True


def _charge(
    ledger: CostLedger,
    budget: ExecutionBudget,
    cost: Mapping[str, Any],
    elapsed_ms: float,
) -> Dict[str, Any]:
    row = _normalise_cost(cost, elapsed_ms)
    ledger.actions += 1
    ledger.scan_rows += row["scan_rows"]
    ledger.retrieval_calls += row["retrieval_calls"]
    ledger.model_calls += row["model_calls"]
    if row["input_tokens"] is None or row["output_tokens"] is None:
        ledger.unknown_token_events += 1
    else:
        ledger.input_tokens += _as_int(row["input_tokens"])
        ledger.output_tokens += _as_int(row["output_tokens"])
    ledger.operator_wall_ms += row["wall_ms"]
    if row.get("cache_hit"):
        ledger.cache_hits += 1
    if ledger.actions > budget.max_actions:
        ledger.budget_violations.append("action_budget_exceeded")
    if ledger.scan_rows > budget.max_scan_rows:
        ledger.budget_violations.append("scan_budget_exceeded")
    if ledger.retrieval_calls > budget.max_retrieval_calls:
        ledger.budget_violations.append("retrieval_budget_exceeded")
    if ledger.model_calls > budget.max_model_calls:
        ledger.budget_violations.append("model_budget_exceeded")
    return row


def _root_bindings(
    question: str,
    plan: Mapping[str, Any],
    backend: EvidenceBackend,
) -> Tuple[List[Dict[str, Any]], str]:
    operators = list(plan.get("operators") or [])
    if not operators:
        return [], "missing_operators"
    root_slot = str(operators[0].get("input_slot") or "$root")
    branch = str(plan.get("plan_kind") or "").startswith("branch_")
    rows = []
    for index, raw in enumerate(backend.initial_bindings(question, plan) or []):
        surface = str(raw.get("surface") or "").strip()
        key = str(raw.get("key") or _key(surface))
        if not surface or not key:
            continue
        binding_id = f"b{index}"
        rows.append({
            "binding_id": binding_id,
            "parent_binding_id": "",
            "root_binding_id": binding_id,
            "root_surface": _clean_root_answer_surface(
                raw.get("mention_surface") or surface
            ),
            "root_key": key,
            "active": True,
            "invalidated": False,
            "slot_bindings": {
                root_slot: {
                    "surface": surface,
                    "key": key,
                    "canonical_identity_surface": (
                        surface if raw.get("title_catalog_exact") else ""
                    ),
                    "canonical_identity_key": (
                        key if raw.get("title_catalog_exact") else ""
                    ),
                    "canonical_identity_validated": bool(
                        raw.get("title_catalog_exact")
                    ),
                    "canonical_identity_source": (
                        "query_title_catalog" if raw.get("title_catalog_exact") else ""
                    ),
                    "canonical_title_family_key": str(
                        raw.get("canonical_title_family_key") or ""
                    ),
                    "canonical_title_doc_ids": list(
                        raw.get("canonical_title_doc_ids") or []
                    ),
                    "canonical_title_family_doc_ids": list(
                        raw.get("canonical_title_family_doc_ids") or []
                    ),
                    "canonical_title_family_unambiguous": bool(
                        raw.get("canonical_title_family_unambiguous")
                    ),
                }
            },
            "path_fact_ids": [],
            "path_fact_usage_ids": [],
            "next_operator_index": 0,
            "root_metadata": dict(raw),
        })
    expected = 2 if branch else 1
    return (rows, "bound") if len(rows) == expected else ([], "root_binding_failed")


def _verified_fact(
    state: Dict[str, Any],
    *,
    raw_fact: Mapping[str, Any],
    binding: Mapping[str, Any],
    operator: Mapping[str, Any],
    origin: str,
    backend: EvidenceBackend,
    ledger: CostLedger,
    budget: ExecutionBudget,
    trace: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    fact = _normalise_fact(raw_fact, origin)
    physical_fact_id = str(fact["fact_id"])
    fact.setdefault("operator_id", str(operator.get("operator_id") or ""))
    usage = _register_fact_usage(
        state,
        fact,
        binding_id=str(binding.get("binding_id") or ""),
        obligation_id=str(operator.get("operator_id") or ""),
    )
    contract = _binding_to_obligation_contract(usage)
    usage["binding_obligation_contract_status"] = contract["status"]
    usage["binding_obligation_contract_reason"] = contract["reason"]
    if contract["status"] in {"REJECTED", "DEFERRED"}:
        return None
    if not _budget_allows(ledger, budget, may_retrieve=False):
        return None
    verification_fact = dict(usage)
    verification_fact["fact_usage_id"] = str(usage["fact_id"])
    verification_fact["fact_id"] = physical_fact_id
    started = time.perf_counter()
    result = dict(backend.verify(verification_fact) or {})
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    charged = _charge(ledger, budget, result.get("cost") or {}, elapsed_ms)
    status = str(result.get("support_status") or INSUFFICIENT)
    source_status = str(
        result.get("source_support_gate_status") or status
    )
    admission = str(result.get("obligation_admission_status") or "")
    if not admission:
        admission = "ADMITTED" if status == SUPPORTED else "REJECTED"
    usage.update({
        "support_status": status,
        "support_reason": str(result.get("support_reason") or "unspecified"),
        "source_support_status": source_status,
        "source_support_reason": str(
            result.get("source_support_gate_reason")
            or result.get("support_reason")
            or "unspecified"
        ),
        "obligation_admission_status": admission,
        "obligation_admission_reason": str(
            result.get("obligation_admission_reason") or ""
        ),
        "source_sentence_id": _as_int(result.get("source_sentence_id"), -1),
        "source_sentence_span": str(
            result.get("source_sentence_span") or ""
        )[:480],
        "source_title": str(result.get("source_title") or ""),
        "verifier_version": str(
            result.get("verifier_version") or backend.verifier_version
        ),
        "guarantee_scope": str(
            result.get("guarantee_scope") or "source_support_only"
        ),
        "provenance_scope": str(result.get("provenance_scope") or ""),
        "certified_assertion": dict(result.get("certified_assertion") or {}),
        "source_subject_surface": str(
            result.get("source_subject_surface") or ""
        ),
        "source_object_surface": str(
            result.get("source_object_surface") or ""
        ),
        "source_identity_qualifier_tokens": list(
            result.get("source_identity_qualifier_tokens") or []
        ),
        "tuple_state": "SUPPORTED" if status == SUPPORTED else "REJECTED",
    })
    trace.append({
        "operator": "VERIFY",
        "binding_id": str(binding.get("binding_id") or ""),
        "obligation_id": str(operator.get("operator_id") or ""),
        "physical_fact_id": physical_fact_id,
        "fact_usage_id": str(usage.get("fact_id") or ""),
        "support_status": status,
        "source_support_status": source_status,
        "obligation_admission_status": admission,
        "cost": charged,
    })
    return usage if (
        status == SUPPORTED
        and source_status == SUPPORTED
        and admission in {"ADMITTED", "OFF"}
        and usage.get("source_id") is not None
        and str(usage.get("source_sentence_span") or "").strip()
    ) else None


def _expand_result(
    state: Dict[str, Any],
    *,
    binding: Dict[str, Any],
    operator: Mapping[str, Any],
    result: Mapping[str, Any],
    origin: str,
    backend: EvidenceBackend,
    ledger: CostLedger,
    budget: ExecutionBudget,
    trace: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    facts = {
        str(row.get("fact_id") or row.get("completion_fact_id") or ""): dict(row)
        for row in result.get("facts") or []
    }
    children = []
    for match in result.get("matches") or []:
        raw_fact = facts.get(str(match.get("fact_id") or ""))
        if raw_fact is None:
            continue
        usage = _verified_fact(
            state,
            raw_fact=raw_fact,
            binding=binding,
            operator=operator,
            origin=origin,
            backend=backend,
            ledger=ledger,
            budget=budget,
            trace=trace,
        )
        if usage is None:
            continue
        output_slot = str(operator.get("output_slot") or "")
        output_surface = str(match.get("output_surface") or "").strip()
        output_key = str(match.get("output_key") or _key(output_surface))
        output_role = str(usage.get("bound_output_role") or "")








        verified_key = str(usage.get("bound_output_key") or "").strip()
        verified_surface = str(
            result.get(f"source_{output_role}_surface")
            or usage.get(f"source_{output_role}_surface")
            or ""
        ).strip()
        if verified_surface:
            output_surface = verified_surface
        if verified_key:
            output_key = verified_key
        elif verified_surface:
            output_key = _key(verified_surface)
        if not output_slot or not output_surface or not output_key:
            continue
        physical_fact_id = str(usage.get("physical_fact_id") or "")
        usage_id = str(usage.get("fact_id") or "")
        child_id = f"b{state['next_binding_id']}"
        state["next_binding_id"] += 1
        child = {
            **binding,
            "binding_id": child_id,
            "parent_binding_id": str(binding.get("binding_id") or ""),
            "active": True,
            "invalidated": False,
            "slot_bindings": {
                **dict(binding.get("slot_bindings") or {}),
                output_slot: {
                    "surface": output_surface,
                    "key": output_key,
                    "fact_id": usage_id,
                    "physical_fact_id": physical_fact_id,
                    "fact_usage_id": usage_id,
                    "source_id": usage.get("source_id"),
                    "origin": usage.get("origin"),
                    "source_identity_qualifier_tokens": list(
                        usage.get("source_identity_qualifier_tokens") or []
                    ),
                },
            },
            "path_fact_ids": list(binding.get("path_fact_ids") or [])
            + [physical_fact_id],
            "path_fact_usage_ids": list(
                binding.get("path_fact_usage_ids") or []
            ) + [usage_id],
            "next_operator_index": _as_int(
                binding.get("next_operator_index")
            ) + 1,
            "last_action": origin,
        }
        state["bindings"].append(child)
        children.append(child)
    return children


def execute_simple_stateful_lazy(
    *,
    query_id: str,
    question: str,
    logical_plan: Mapping[str, Any],
    backend: EvidenceBackend,
    budget: Optional[ExecutionBudget] = None,
    completion_enabled: bool = True,
    allow_external_plan_review: bool = False,
) -> Dict[str, Any]:
    """Execute one frozen plan with deterministic verify-before-commit search."""
    started = time.perf_counter()
    budget = budget or ExecutionBudget()
    plan = dict(logical_plan or {})
    if (
        plan.get("ai_plan_provenance")
        and not allow_external_plan_review
    ):
        return {
            "version": SIMPLE_STATEFUL_LAZY_VERSION,
            "query_id": str(query_id),
            "question": question,
            "logical_plan": plan,
            "question_complete_supported_witness": False,
            "structured_admission": False,
            "stop_reason": "question_semantic_review_required",
        }
    bindings, bind_reason = _root_bindings(question, plan, backend)
    if not bindings:
        return {
            "version": SIMPLE_STATEFUL_LAZY_VERSION,
            "query_id": str(query_id),
            "question": question,
            "logical_plan": plan,
            "question_complete_supported_witness": False,
            "structured_admission": False,
            "stop_reason": bind_reason,
            "bindings": [],
            "facts": [],
        }

    operators = list(plan.get("operators") or [])
    state: Dict[str, Any] = {
        "logical_plan": plan,
        "bindings": bindings,
        "root_binding_ids": [str(row["binding_id"]) for row in bindings],
        "facts": {},
        "fact_usage_lookup": {},
        "next_binding_id": len(bindings),
    }
    ledger = CostLedger()
    trace: List[Dict[str, Any]] = []
    queue: Deque[Dict[str, Any]] = deque(bindings)
    overlay: Dict[Tuple[str, str], Dict[str, Any]] = {}
    expanded = set()
    stop_reason = "work_queue_exhausted"

    while queue:
        binding = queue.popleft()
        operator_index = _as_int(binding.get("next_operator_index"))
        if operator_index >= len(operators):
            continue
        operator = dict(operators[operator_index])
        input_slot = str(operator.get("input_slot") or "")
        bound = dict((binding.get("slot_bindings") or {}).get(input_slot) or {})
        expansion_key = (
            str(binding.get("binding_id") or ""),
            str(operator.get("operator_id") or ""),
        )
        if expansion_key in expanded:
            continue
        expanded.add(expansion_key)
        if not _budget_allows(ledger, budget, may_retrieve=False):
            stop_reason = "hard_budget_exhausted"
            break
        remaining_scan = max(0, budget.max_scan_rows - ledger.scan_rows)
        access_operator = {
            **operator,
            "execution_scan_limit": remaining_scan,
            "execution_retrieval_limit": 0,
            "execution_model_limit": 0,
        }
        access_started = time.perf_counter()
        lookup = dict(backend.lookup_join(binding, access_operator) or {})
        access_elapsed = (time.perf_counter() - access_started) * 1000.0
        lookup_cost = _charge(
            ledger, budget, lookup.get("cost") or {}, access_elapsed
        )
        trace.append({
            "operator": LOOKUP_JOIN,
            "binding_id": str(binding.get("binding_id") or ""),
            "obligation_id": str(operator.get("operator_id") or ""),
            "bound_key": str(bound.get("key") or ""),
            "candidate_count": len(lookup.get("matches") or []),
            "cost": lookup_cost,
        })
        children = _expand_result(
            state,
            binding=binding,
            operator=operator,
            result=lookup,
            origin="fact_index",
            backend=backend,
            ledger=ledger,
            budget=budget,
            trace=trace,
        )

        overlay_key = (
            str(bound.get("key") or ""),
            str(operator.get("operator_id") or ""),
        )
        if not children and completion_enabled:
            recovered = overlay.get(overlay_key)
            if recovered is not None:
                recovered = {
                    **recovered,
                    "cost": {"cache_hit": True},
                }
                ledger.cache_hits += 1
            elif _budget_allows(ledger, budget, may_retrieve=True):
                remaining_scan = max(0, budget.max_scan_rows - ledger.scan_rows)
                completion_operator = {
                    **operator,
                    "execution_scan_limit": remaining_scan,
                    "execution_retrieval_limit": max(
                        0, budget.max_retrieval_calls - ledger.retrieval_calls
                    ),
                    "execution_model_limit": max(
                        0, budget.max_model_calls - ledger.model_calls
                    ),
                }
                completion_started = time.perf_counter()
                recovered = dict(
                    backend.complete(binding, completion_operator) or {}
                )
                completion_elapsed = (
                    time.perf_counter() - completion_started
                ) * 1000.0
                completion_cost = _charge(
                    ledger, budget, recovered.get("cost") or {}, completion_elapsed
                )
                overlay[overlay_key] = {
                    **recovered,
                    "facts": [dict(row) for row in recovered.get("facts") or []],
                    "matches": [dict(row) for row in recovered.get("matches") or []],
                }
                trace.append({
                    "operator": COMPLETE,
                    "binding_id": str(binding.get("binding_id") or ""),
                    "obligation_id": str(operator.get("operator_id") or ""),
                    "bound_key": str(bound.get("key") or ""),
                    "candidate_count": len(recovered.get("matches") or []),
                    "cost": completion_cost,
                })
            else:
                recovered = None
            if recovered:
                children = _expand_result(
                    state,
                    binding=binding,
                    operator=operator,
                    result=recovered,
                    origin="query_local_overlay",
                    backend=backend,
                    ledger=ledger,
                    budget=budget,
                    trace=trace,
                )

        binding["active"] = False
        queue.extend(children)

    witnesses = _structured_witnesses(state)
    selected, reduction_reason, reduction = _reduce_witnesses(
        witnesses,
        plan,
        state["root_binding_ids"],
        facts=state["facts"],
        conflict_safe_terminal=True,
        provenance_grouped_terminal=True,
    )
    if selected:
        stop_reason = "complete_supported_witness"
    answer = str(
        (((selected or {}).get("slot_bindings") or {}).get("$answer") or {}).get(
            "surface"
        )
        or ""
    ).strip()
    return _json_safe({
        "version": SIMPLE_STATEFUL_LAZY_VERSION,
        "query_id": str(query_id),
        "question": question,
        "policy": "stable_verify_before_commit",
        "logical_plan": plan,
        "question_complete_supported_witness": bool(selected),
        "structured_admission": bool(selected and answer),
        "structured_answer": answer if selected else "",
        "selected_witness": selected or {},
        "witness_reduction_reason": reduction_reason,
        "witness_reduction": reduction,
        "supported_witnesses": witnesses,
        "stop_reason": stop_reason,
        "bindings": list(state["bindings"]),
        "facts": list(state["facts"].values()),
        "operator_trace": trace,
        "cost_ledger": ledger.public(),
        "execution_summary": {
            "lookup_join_count": sum(
                row.get("operator") == LOOKUP_JOIN for row in trace
            ),
            "complete_count": sum(row.get("operator") == COMPLETE for row in trace),
            "verify_count": sum(row.get("operator") == "VERIFY" for row in trace),
            "query_local_overlay_fact_count": sum(
                row.get("origin") == "query_local_overlay"
                for row in state["facts"].values()
            ),
            "overlay_entry_count": len(overlay),
        },
        "experiment_controls": {
            "completion_enabled": bool(completion_enabled),
            "orchestration": SIMPLE_STATEFUL_LAZY_VERSION,
            "bswc_enabled": False,
            "query_time_llm_materialization": False,
        },
        "end_to_end_wall_ms": round(
            (time.perf_counter() - started) * 1000.0, 6
        ),
    })
