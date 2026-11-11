"""Canonical Rule-first Semantic V3.12 runtime compiler.

This module closes the gap between the compiler-only V3.12 experiments and
the production query-executor entry point.  It deliberately keeps four
states separate: candidate generation, static validity, semantic admission,
and automatic execution eligibility.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .evidence_query_compiler_utils import compile_evidence_query_plan
from .evidence_query_compiler_semantic_v3_1_utils import (
    MODEL_BASED_SEMANTIC_ADMISSION_DEV,
    _request_record,
    _response_parts,
)
from .evidence_query_compiler_semantic_v3_6_factorized_gate_utils import (
    combine_v3_6_gates,
    validate_v3_6_grounding_gate,
)
from .evidence_query_compiler_semantic_v3_7_stepwise_gate_utils import (
    build_v3_7_relation_messages,
    validate_v3_7_relation_gate,
)
from .evidence_query_compiler_semantic_v3_9_grounding_gate_utils import (
    build_v3_9_grounding_messages,
)
from .evidence_query_compiler_semantic_v3_12_skeleton_utils import (
    SEMANTIC_V3_12_CACHE_VERSION,
    SEMANTIC_V3_12_PIPELINE_VERSION,
    SEMANTIC_V3_12_PROMPT_VERSION,
    build_semantic_v3_12_skeleton_messages,
    deterministic_v3_12_grounding_certificate,
    lower_semantic_v3_12_skeleton,
)
from .late_bound_textual_relation_utils import parse_json_object
from .relation_fact_join_utils import (
    PLAN_COMPLETE,
    attach_logical_query_plan_identity,
)


SEMANTIC_V3_12_RUNTIME_VERSION = "rule_first_semantic_v3_12_runtime_v1"
SEMANTIC_V3_12_RUNTIME_CACHE_VERSION = "semantic_v3_12_runtime_cache_v1"


def _static_valid(plan: Mapping[str, Any]) -> bool:
    return bool(
        (plan.get("compile_phase_status") or {}).get(
            "plan_static_valid",
            plan.get("compile_status") == "compiled",
        )
    )


def _plan_coverage_status(plan: Mapping[str, Any]) -> str:
    return str(
        plan.get("plan_coverage_status")
        or (plan.get("plan_coverage") or {}).get("status")
        or ""
    )


def _call_json(
    infer: Callable[..., Any],
    messages: Sequence[Mapping[str, str]],
    *,
    purpose: str,
    completion: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    response = infer(
        list(messages),
        max_completion_tokens=max(64, int(completion)),
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    content, metadata, cache_hit = _response_parts(response)
    payload = parse_json_object(content)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{purpose}_response_not_json")
    request = _request_record(
        attempt_id=purpose,
        purpose=purpose,
        messages=messages,
        content=content,
        metadata=metadata,
        cache_hit=cache_hit,
        payload=payload,
    )
    return dict(payload), request


def _question_cue(
    question: str,
    obligation: Mapping[str, Any],
    operator: Mapping[str, Any],
) -> str:
    source_span = dict(obligation.get("source_span") or {})
    cue = str(source_span.get("text") or "").strip()
    if cue and cue.casefold() in question.casefold():
        return cue
    for token in list(obligation.get("relation_tokens") or []) + list(
        operator.get("relation_tokens") or []
    ):
        token = str(token or "").strip()
        if token and token.casefold() in question.casefold():
            return token
    return str(obligation.get("semantic_label") or "").strip()


def _rule_gate_view(question: str, plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Render a rule plan for the existing relation-fidelity gate.

    Rule plans predate Semantic IR and therefore cannot be passed directly to
    the V3 relation gate.  This adapter copies only already-compiled operator
    and question-contract fields; it does not infer new relations or values.
    """
    wrapped = copy.deepcopy(dict(plan))
    operators = [dict(value) for value in plan.get("operators") or []]
    obligations = [
        dict(value)
        for value in plan.get("question_obligations") or []
        if str(value.get("kind") or "relation") == "relation"
        and bool(value.get("required", True))
    ]
    relations = []
    requirements = []
    for index, operator in enumerate(operators):
        obligation = obligations[index] if index < len(obligations) else {}
        cue = _question_cue(question, obligation, operator)
        input_variable = str(
            obligation.get("input_variable")
            or operator.get("input_slot")
            or ("$root" if index == 0 else f"$v{index}")
        )
        output_variable = str(
            obligation.get("output_variable")
            or operator.get("output_slot")
            or ("$answer" if index == len(operators) - 1 else f"$v{index + 1}")
        )
        families = list(operator.get("predicate_families") or [])
        subtypes = list(operator.get("required_predicate_subtypes") or [])
        relations.append({
            "relation_cues": [cue] if cue else [],
            "predicate_family": str(families[0] if families else ""),
            "predicate_subtype": str(subtypes[0] if subtypes else ""),
            "canonical_relation": str(operator.get("canonical_relation") or ""),
            "input_variable": input_variable,
            "output_variable": output_variable,
            "input_semantic_role": str(
                operator.get("semantic_input_role")
                or operator.get("input_role")
                or ""
            ),
            "output_semantic_role": str(
                operator.get("semantic_output_role")
                or operator.get("output_role")
                or ""
            ),
            "semantic_direction": str(
                operator.get("semantic_direction")
                or obligation.get("direction")
                or ""
            ),
            "output_entity_type": str(
                obligation.get("expected_output_type")
                or operator.get("output_type")
                or "entity"
            ),
        })
        requirements.append({
            "requirement_id": str(
                obligation.get("obligation_id") or f"q{index}"
            ),
            "covered_by_relation_index": index,
            "relation_cues": [cue] if cue else [],
            "expected_output_type": relations[-1]["output_entity_type"],
            "answer_category": "",
        })
    wrapped["semantic_query_ir"] = {
        "start_entities": [],
        "relations": relations,
        "question_requirements": requirements,
        "answer_operation": str(plan.get("answer_operator") or "project"),
        "final_answer_type": str(plan.get("final_answer_type") or ""),
        "final_answer_category": "",
        "final_answer_variable": "$answer",
    }
    return wrapped


def _rule_contract_status(plan: Mapping[str, Any]) -> tuple[bool, str]:
    if not _static_valid(plan):
        return False, "rule_plan_static_invalid"
    if _plan_coverage_status(plan) != PLAN_COMPLETE:
        return False, "rule_question_contract_incomplete"
    kind = str(plan.get("plan_kind") or "")
    answer_operator = str(plan.get("answer_operator") or "")
    if kind in {"branch_compare", "branch_temporal_compare"} and answer_operator in {
        "compare_eq",
        "argmin_root",
        "argmax_root",
    }:
        return True, "deterministic_branch_template_contract"
    operators = list(plan.get("operators") or [])
    relation_obligations = [
        value
        for value in plan.get("question_obligations") or []
        if str(value.get("kind") or "relation") == "relation"
        and bool(value.get("required", True))
    ]
    if len(operators) != len(relation_obligations):
        return False, "rule_relation_operator_count_mismatch"
    return False, "model_relation_admission_required"


def _apply_semantic_admission(
    plan: Mapping[str, Any],
    *,
    decision: str,
    origin: str,
    gate: Mapping[str, Any],
    deterministic: bool = False,
) -> Dict[str, Any]:
    wrapped = copy.deepcopy(dict(plan))
    coverage = _plan_coverage_status(wrapped)
    eligible = bool(decision == "PASS" and coverage == PLAN_COMPLETE)
    if deterministic:
        wrapped["question_semantic_validation"] = {
            "version": SEMANTIC_V3_12_RUNTIME_VERSION,
            "status": (
                "DETERMINISTIC_RULE_CONTRACT_VALIDATED"
                if eligible
                else "DETERMINISTIC_RULE_CONTRACT_REJECTED"
            ),
            "admission_source": origin,
            "review_decision": decision,
            "auto_execution_eligible": eligible,
            "experimental_execution_eligible": eligible,
            "human_reviewed": False,
            "is_human_gold": False,
            "gate": dict(gate),
        }
    else:
        wrapped["question_semantic_validation"] = {
            "version": SEMANTIC_V3_12_RUNTIME_VERSION,
            "status": MODEL_BASED_SEMANTIC_ADMISSION_DEV,
            "admission_source": MODEL_BASED_SEMANTIC_ADMISSION_DEV,
            "admission_origin": origin,
            "review_decision": decision,
            "auto_execution_eligible": False,
            "experimental_execution_eligible": eligible,
            "human_reviewed": False,
            "is_human_gold": False,
            "gate": dict(gate),
        }
        wrapped["model_based_semantic_admission"] = {
            "version": SEMANTIC_V3_12_RUNTIME_VERSION,
            "admission_source": MODEL_BASED_SEMANTIC_ADMISSION_DEV,
            "admission_origin": origin,
            "review_decision": decision,
            "experimental_execution_eligible": eligible,
            "human_reviewed": False,
            "is_human_gold": False,
            "uses_gold_answer": False,
            "uses_supporting_facts": False,
            "uses_retrieval_or_execution": False,
        }
    validation = dict(wrapped.get("question_semantic_validation") or {})
    auto_eligible = bool(validation.get("auto_execution_eligible"))
    experimental_eligible = bool(
        validation.get("experimental_execution_eligible")
    )
    wrapped["compiler_stage_status"] = {
        "candidate_generated": bool(wrapped),
        "static_valid": _static_valid(wrapped),
        "semantic_admission": decision,
        "question_coverage": coverage,



        "auto_execution_eligible": auto_eligible,
        "experimental_execution_eligible": experimental_eligible,
        "runtime_execution_eligible_with_model_flag": bool(
            auto_eligible or experimental_eligible
        ),
        "execution_requires_model_admission_flag": not deterministic,
    }
    return attach_logical_query_plan_identity(
        wrapped,
        compiler_source=origin,
        compiler_mode="rule_first_semantic_v3_12",
        attempt_id="canonical",
    )


def _relation_gate(
    question: str,
    plan: Mapping[str, Any],
    infer: Callable[..., Any],
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    payload, request = _call_json(
        infer,
        build_v3_7_relation_messages(question, plan),
        purpose="canonical_rule_relation_gate_v3_12",
        completion=180,
    )
    relation_gate, reason = validate_v3_7_relation_gate(plan, payload)
    return {
        "decision": (
            str(relation_gate.get("decision") or "PROTOCOL_INVALID")
            if reason == "validated"
            else "PROTOCOL_INVALID"
        ),
        "validation_reason": reason,
        "relation_gate": relation_gate,
        "root_grounding": "DEFERRED_TO_DETERMINISTIC_EXECUTION_BINDING",
    }, [request]


def _factorized_gate(
    question: str,
    plan: Mapping[str, Any],
    infer: Callable[..., Any],
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    relation_payload, relation_request = _call_json(
        infer,
        build_v3_7_relation_messages(question, plan),
        purpose="canonical_fallback_relation_gate_v3_12",
        completion=180,
    )
    relation_gate, relation_reason = validate_v3_7_relation_gate(
        plan, relation_payload
    )
    grounding_payload, grounding_request = _call_json(
        infer,
        build_v3_9_grounding_messages(question, plan),
        purpose="canonical_fallback_grounding_gate_v3_12",
        completion=100,
    )
    grounding_gate, grounding_reason = validate_v3_6_grounding_gate(
        grounding_payload
    )
    deterministic_grounding = deterministic_v3_12_grounding_certificate(
        question, plan
    )
    effective_grounding = deterministic_grounding or grounding_gate
    combined: Dict[str, Any] = {}
    if relation_reason == grounding_reason == "validated":
        combined = combine_v3_6_gates(relation_gate, effective_grounding)
    return {
        "decision": str(combined.get("decision") or "PROTOCOL_INVALID"),
        "relation_validation_reason": relation_reason,
        "grounding_validation_reason": grounding_reason,
        "relation_gate": relation_gate,
        "grounding_gate": grounding_gate,
        "deterministic_grounding_certificate": deterministic_grounding,
        "combined": combined,
    }, [relation_request, grounding_request]


def compile_rule_first_semantic_v3_12_query_plan(
    question: str,
    *,
    llm_infer: Optional[Callable[..., Any]],
    model_name: str,
    rule_options: Optional[Mapping[str, Any]] = None,
    generation_max_completion_tokens: int = 650,
    semantic_skeleton_lowerer: Callable[
        [str, Mapping[str, Any]], Dict[str, Any]
    ] = lower_semantic_v3_12_skeleton,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Compile one question through the frozen canonical admission boundary."""
    started = time.perf_counter()
    rule_plan, rule_trace = compile_evidence_query_plan(
        question,
        mode="rule",
        model_name=model_name,
        rule_options=rule_options,
    )
    trace: Dict[str, Any] = {
        "version": SEMANTIC_V3_12_RUNTIME_VERSION,
        "mode": "rule_first_semantic_v3_12",
        "model": str(model_name or ""),
        "selected_source": "none",
        "rule_trace": rule_trace,
        "rule_static_valid": _static_valid(rule_plan),
        "requests": [],
        "llm_requests": 0,
        "uses_gold_answer": False,
        "uses_supporting_facts": False,
        "uses_retrieval_reader_or_witness": False,
        "versions": {
            "runtime": SEMANTIC_V3_12_RUNTIME_VERSION,
            "runtime_cache": SEMANTIC_V3_12_RUNTIME_CACHE_VERSION,
            "skeleton_prompt": SEMANTIC_V3_12_PROMPT_VERSION,
            "skeleton_cache": SEMANTIC_V3_12_CACHE_VERSION,
            "skeleton_lowering": SEMANTIC_V3_12_PIPELINE_VERSION,
        },
    }
    deterministic_rule, rule_reason = _rule_contract_status(rule_plan)
    trace["rule_admission_precheck"] = rule_reason
    rejected_rule_plan: Dict[str, Any] = {}

    if deterministic_rule:
        selected = _apply_semantic_admission(
            rule_plan,
            decision="PASS",
            origin="rule_deterministic_template_contract_v1",
            gate={"decision": "PASS", "reason": rule_reason},
            deterministic=True,
        )
        trace.update({
            "selected_source": "rule_deterministic_template_contract_v1",
            "reason": rule_reason,
        })
        selected["query_compiler_trace"] = trace
        trace["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 6)
        return selected, trace

    if _static_valid(rule_plan):
        if rule_reason == "model_relation_admission_required" and llm_infer is not None:
            try:
                gate, requests = _relation_gate(
                    question, _rule_gate_view(question, rule_plan), llm_infer
                )
                trace["requests"].extend(requests)
                trace["rule_semantic_gate"] = gate
                rejected_rule_plan = _apply_semantic_admission(
                    rule_plan,
                    decision=str(gate.get("decision") or "PROTOCOL_INVALID"),
                    origin="rule_model_relation_admission_v3_12",
                    gate=gate,
                )
                if gate.get("decision") == "PASS":
                    trace.update({
                        "selected_source": "rule_model_relation_admission_v3_12",
                        "reason": "rule_relation_gate_pass",
                    })
                    trace["llm_requests"] = len(trace["requests"])
                    trace["wall_ms"] = round(
                        (time.perf_counter() - started) * 1000.0, 6
                    )
                    rejected_rule_plan["query_compiler_trace"] = trace
                    return rejected_rule_plan, trace
            except Exception as exc:
                trace["rule_gate_error_type"] = type(exc).__name__
                trace["rule_gate_error_preview"] = str(exc)[:240]
                rejected_rule_plan = _apply_semantic_admission(
                    rule_plan,
                    decision="UNKNOWN",
                    origin="rule_model_relation_admission_v3_12",
                    gate={
                        "decision": "UNKNOWN",
                        "reason": "rule_relation_gate_failed",
                    },
                )
        elif llm_infer is None:
            rejected_rule_plan = _apply_semantic_admission(
                rule_plan,
                decision="UNKNOWN",
                origin="rule_model_relation_admission_v3_12",
                gate={"decision": "UNKNOWN", "reason": "llm_not_available"},
            )
        else:
            rejected_rule_plan = _apply_semantic_admission(
                rule_plan,
                decision="REJECT",
                origin="rule_deterministic_contract_rejection_v1",
                gate={"decision": "REJECT", "reason": rule_reason},
                deterministic=True,
            )

    if llm_infer is None:
        selected = rejected_rule_plan or rule_plan
        trace.update({
            "reason": "semantic_admission_requires_llm",
            "llm_requests": len(trace["requests"]),
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        selected["query_compiler_trace"] = trace
        return selected, trace

    fallback_candidate: Dict[str, Any] = {}
    try:
        payload, request = _call_json(
            llm_infer,
            build_semantic_v3_12_skeleton_messages(
                question,
                max_completion_tokens=generation_max_completion_tokens,
            ),
            purpose="canonical_semantic_skeleton_generation_v3_12",
            completion=generation_max_completion_tokens,
        )
        trace["requests"].append(request)
        trace["semantic_skeleton_payload"] = payload
        lowering = semantic_skeleton_lowerer(question, payload)
        trace["semantic_skeleton_lowering"] = {
            key: value for key, value in lowering.items() if key != "plan"
        }
        if bool(lowering.get("static_valid")) and lowering.get("plan"):
            fallback_candidate = dict(lowering["plan"])
            gate, requests = _factorized_gate(
                question, fallback_candidate, llm_infer
            )
            trace["requests"].extend(requests)
            trace["fallback_semantic_gate"] = gate
            fallback_candidate = _apply_semantic_admission(
                fallback_candidate,
                decision=str(gate.get("decision") or "PROTOCOL_INVALID"),
                origin="llm_semantic_skeleton_v3_12",
                gate=gate,
            )
            eligible = bool(
                (fallback_candidate.get("compiler_stage_status") or {}).get(
                    "experimental_execution_eligible"
                )
            )
            if eligible:
                trace.update({
                    "selected_source": "llm_semantic_skeleton_v3_12",
                    "reason": "fallback_factorized_gate_pass",
                })
                trace["llm_requests"] = len(trace["requests"])
                trace["wall_ms"] = round(
                    (time.perf_counter() - started) * 1000.0, 6
                )
                fallback_candidate["query_compiler_trace"] = trace
                return fallback_candidate, trace
    except Exception as exc:
        trace["fallback_error_type"] = type(exc).__name__
        trace["fallback_error_preview"] = str(exc)[:240]

    selected = fallback_candidate or rejected_rule_plan or rule_plan
    trace.update({
        "reason": "no_semantically_admitted_plan",
        "llm_requests": len(trace["requests"]),
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
    })
    selected["query_compiler_trace"] = trace
    return selected, trace
