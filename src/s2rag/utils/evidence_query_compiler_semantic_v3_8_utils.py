"""Integrated rule-first semantic compiler with factorized safety gates.

V3.8 connects the previously isolated V3.2 semantic generator, frozen V2.3
physical lowerer, V3.7 stepwise relation gate, and V3.6 start-grounding gate.
It remains a compiler-only development path: it does not inspect retrieval,
execution, QA answers, supporting facts, witnesses, or Reader outputs.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Mapping, Optional

from .evidence_query_compiler_semantic_v3_1_utils import (
    _request_record,
    _response_parts,
)
from .evidence_query_compiler_semantic_v3_2_utils import (
    build_semantic_v3_2_generation_messages,
    validate_semantic_v3_2_generation_payload,
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
from .evidence_query_compiler_semantic_v3_10_dependency_utils import (
    normalize_semantic_dependency_order,
)
from .evidence_query_compiler_utils import compile_evidence_query_plan
from .evidence_query_compiler_v2_utils import (
    compile_semantic_payload_v2_3,
    validate_v2_physical_plan,
)
from .late_bound_textual_relation_utils import parse_json_object
from .relation_fact_join_utils import PLAN_COMPLETE, attach_logical_query_plan_identity


SEMANTIC_V3_8_PIPELINE_VERSION = "rule_first_semantic_v3_8_2_20261029"
MODEL_BASED_SEMANTIC_ADMISSION_V3_8 = "MODEL_BASED_SEMANTIC_ADMISSION_DEV_V3_8"


def _call_json(
    llm_infer: Callable[..., Any],
    messages: list[dict[str, str]],
    *,
    attempt_id: str,
    purpose: str,
    max_completion_tokens: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    response = llm_infer(
        messages,
        max_completion_tokens=max(32, int(max_completion_tokens)),
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    content, metadata, cache_hit = _response_parts(response)
    payload = parse_json_object(content)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{purpose}_response_not_json_object")
    record = _request_record(
        attempt_id=attempt_id,
        purpose=purpose,
        messages=messages,
        content=content,
        metadata=metadata,
        cache_hit=cache_hit,
        payload=payload,
    )
    return dict(payload), record


def _admit(
    plan: Mapping[str, Any],
    combined_gate: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> Dict[str, Any]:
    admitted = dict(plan)
    admitted["plan_coverage"] = {
        "status": PLAN_COMPLETE,
        "reason": "factorized_semantic_gate_pass",
        "evidence": MODEL_BASED_SEMANTIC_ADMISSION_V3_8,
        "prior_structural_coverage": dict(admitted.get("plan_coverage") or {}),
    }
    admitted["plan_coverage_status"] = PLAN_COMPLETE
    admitted["question_semantic_validation"] = {
        "version": "question_semantic_validation_v3_8",
        "status": MODEL_BASED_SEMANTIC_ADMISSION_V3_8,
        "basis": "stepwise_relation_and_start_grounding_gates",
        "review_origin": "MODEL_BASED_DEVELOPMENT_ONLY",
        "auto_execution_eligible": False,
        "experimental_execution_eligible": True,
        "human_reviewed": False,
        "is_human_gold": False,
        "uses_gold_answer": False,
        "uses_supporting_facts": False,
        "uses_retrieval_or_execution": False,
        "gate": dict(combined_gate),
    }
    admitted["model_based_semantic_admission"] = dict(
        admitted["question_semantic_validation"]
    )
    admitted = attach_logical_query_plan_identity(
        admitted,
        compiler_source="llm_semantic_v3_8_factorized_gate",
        compiler_mode="rule_first_semantic_v3_8",
        attempt_id="attempt1",
    )
    admitted["query_compiler_trace"] = dict(trace)
    return admitted


def compile_rule_first_semantic_v3_8_query_plan(
    question: str,
    *,
    llm_infer: Optional[Callable[..., Any]],
    model_name: str,
    rule_options: Optional[Mapping[str, Any]] = None,
    generation_max_completion_tokens: int = 850,
    relation_gate_max_completion_tokens: int = 160,
    grounding_gate_max_completion_tokens: int = 80,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Compile with Rule first, then one semantic generation and two gates."""
    started = time.perf_counter()
    rule_plan, rule_trace = compile_evidence_query_plan(
        question, mode="rule", model_name=model_name, rule_options=rule_options
    )
    if bool((rule_plan.get("compile_phase_status") or {}).get("plan_static_valid")):
        return rule_plan, rule_trace

    trace: Dict[str, Any] = {
        "version": SEMANTIC_V3_8_PIPELINE_VERSION,
        "mode": "rule_first_semantic_v3_8",
        "model": str(model_name or ""),
        "rule_trace": rule_trace,
        "selected_source": "rule_fallback",
        "reason": "rule_pattern_unmatched",
        "requests": [],
        "request_count": 0,
        "selection_observed_execution": False,
        "uses_gold_answer": False,
        "uses_supporting_facts": False,
        "uses_retrieval_reader_or_witness": False,
    }
    if llm_infer is None:
        trace["reason"] = "llm_not_available"
        trace["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 6)
        rule_plan["query_compiler_trace"] = trace
        return rule_plan, trace

    try:
        generation_messages = build_semantic_v3_2_generation_messages(
            question, max_completion_tokens=generation_max_completion_tokens
        )
        raw_payload, request = _call_json(
            llm_infer,
            generation_messages,
            attempt_id="attempt1",
            purpose="semantic_generation_v3_2",
            max_completion_tokens=generation_max_completion_tokens,
        )
        trace["requests"].append(request)
        payload, protocol_reason = validate_semantic_v3_2_generation_payload(
            question, raw_payload
        )
        if protocol_reason == "generation_contract_final_variable_not_terminal":
            normalized_payload, normalization_reason, normalization_trace = (
                normalize_semantic_dependency_order(raw_payload)
            )
            trace.update({
                "dependency_normalization_reason": normalization_reason,
                "dependency_normalization": normalization_trace,
            })
            if normalized_payload:
                payload, protocol_reason = validate_semantic_v3_2_generation_payload(
                    question, normalized_payload
                )
                trace["normalized_generation_payload"] = normalized_payload
        trace["generation_protocol_reason"] = protocol_reason
        if protocol_reason.startswith("abstain_"):
            trace.update({
                "reason": protocol_reason,
                "abstain_reason_code": payload.get("abstain_reason_code"),
                "unsupported_question_spans": payload.get("unsupported_question_spans"),
            })
            return rule_plan, _finish_trace(rule_plan, trace, started)
        if protocol_reason != "validated":
            trace["reason"] = protocol_reason
            return rule_plan, _finish_trace(rule_plan, trace, started)

        semantic_ir, candidate_plan, lowering_reason = compile_semantic_payload_v2_3(
            question, payload, allow_external_plan_review=True
        )
        trace.update({
            "semantic_ir": semantic_ir or None,
            "lowering_reason": str(lowering_reason),
            "structural_candidate_valid": bool(candidate_plan)
            and lowering_reason == "validated",
        })
        if not candidate_plan or lowering_reason != "validated":
            trace["reason"] = str(lowering_reason)
            return rule_plan, _finish_trace(rule_plan, trace, started)

        relation_payload, relation_request = _call_json(
            llm_infer,
            build_v3_7_relation_messages(question, candidate_plan),
            attempt_id="attempt2",
            purpose="stepwise_relation_coverage_gate_v3_7",
            max_completion_tokens=relation_gate_max_completion_tokens,
        )
        trace["requests"].append(relation_request)
        relation_gate, relation_reason = validate_v3_7_relation_gate(
            candidate_plan, relation_payload
        )
        grounding_payload, grounding_request = _call_json(
            llm_infer,
            build_v3_9_grounding_messages(question, candidate_plan),
            attempt_id="attempt3",
            purpose="start_entity_grounding_gate_v3_9",
            max_completion_tokens=grounding_gate_max_completion_tokens,
        )
        trace["requests"].append(grounding_request)
        grounding_gate, grounding_reason = validate_v3_6_grounding_gate(
            grounding_payload
        )
        trace.update({
            "request_count": len(trace["requests"]),
            "relation_gate_validation_reason": relation_reason,
            "relation_gate": relation_gate or None,
            "grounding_gate_validation_reason": grounding_reason,
            "grounding_gate": grounding_gate or None,
        })
        if relation_reason != grounding_reason or relation_reason != "validated":
            trace["reason"] = "factorized_gate_protocol_invalid"
            return rule_plan, _finish_trace(rule_plan, trace, started)
        combined = combine_v3_6_gates(relation_gate, grounding_gate)
        trace["combined_gate"] = combined
        if combined.get("decision") != "PASS":
            trace["reason"] = f"factorized_gate_{str(combined.get('decision')).casefold()}"
            return rule_plan, _finish_trace(rule_plan, trace, started)

        trace.update({
            "selected_source": "model_based_semantic_admission_dev_v3_8",
            "admission_source": MODEL_BASED_SEMANTIC_ADMISSION_V3_8,
            "reason": "factorized_semantic_gate_pass",
        })
        selected = _admit(candidate_plan, combined, trace)
        static_validation = validate_v2_physical_plan(selected)
        if not static_validation["valid"]:
            trace["reason"] = f"post_gate_{static_validation['reason']}"
            return rule_plan, _finish_trace(rule_plan, trace, started)
        trace["selected_plan_id"] = str(selected.get("plan_id") or "")
        trace["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 6)
        selected["query_compiler_trace"] = trace
        return selected, trace
    except Exception as exc:
        trace.update({
            "request_count": len(trace["requests"]),
            "reason": "semantic_v3_8_call_or_parse_failed",
            "error_type": type(exc).__name__,
            "error_preview": str(exc)[:240],
        })
        return rule_plan, _finish_trace(rule_plan, trace, started)


def _finish_trace(
    rule_plan: Dict[str, Any], trace: Dict[str, Any], started: float
) -> Dict[str, Any]:
    trace["request_count"] = len(trace.get("requests") or [])
    trace["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 6)
    rule_plan["query_compiler_trace"] = trace
    return trace
