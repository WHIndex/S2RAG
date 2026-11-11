"""Rule-first Semantic V3.13 runtime compiler.

The runtime boundary is intentionally parallel to V3.12 so historical runs
remain reproducible.  Only the semantic skeleton prompt/lowering is upgraded;
the existing static validator and factorized semantic gates remain unchanged.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, Mapping, Optional

from .evidence_query_compiler_semantic_v3_1_utils import (
    MODEL_BASED_SEMANTIC_ADMISSION_DEV,
)
from .evidence_query_compiler_semantic_v3_12_runtime_utils import (
    _call_json,
    _factorized_gate,
    _plan_coverage_status,
    _relation_gate,
    _rule_contract_status,
    _rule_gate_view,
    _static_valid,
)
from .evidence_query_compiler_semantic_v3_13_skeleton_utils import (
    SEMANTIC_V3_13_CACHE_VERSION,
    SEMANTIC_V3_13_PIPELINE_VERSION,
    SEMANTIC_V3_13_PROMPT_VERSION,
    build_semantic_v3_13_skeleton_messages,
    lower_semantic_v3_13_skeleton,
)
from .evidence_query_compiler_utils import compile_evidence_query_plan
from .relation_fact_join_utils import (
    PLAN_COMPLETE,
    attach_logical_query_plan_identity,
)


SEMANTIC_V3_13_RUNTIME_VERSION = "rule_first_semantic_v3_13_runtime_v1"
SEMANTIC_V3_13_RUNTIME_CACHE_VERSION = "semantic_v3_13_runtime_cache_v1"


def _model_gate_certifies_complete(
    plan: Mapping[str, Any], gate: Mapping[str, Any]
) -> bool:
    """Check the explicit V3.13 model-admission contract for full coverage."""
    if not _static_valid(plan) or str(gate.get("decision") or "") != "PASS":
        return False
    relation_gate = dict(gate.get("relation_gate") or {})
    grounding_gate = dict(
        gate.get("deterministic_grounding_certificate")
        or gate.get("grounding_gate")
        or {}
    )
    if (
        str(relation_gate.get("decision") or "") != "PASS"
        or str(relation_gate.get("uncovered_question_relation") or "") != "NONE"
        or str(grounding_gate.get("decision") or "") != "PASS"
    ):
        return False
    semantic_ir = dict(plan.get("semantic_query_ir") or {})
    relations = list(semantic_ir.get("relations") or [])
    requirements = list(semantic_ir.get("question_requirements") or [])
    operators = list(plan.get("operators") or [])
    if not relations or len(relations) != len(requirements) or len(relations) != len(operators):
        return False
    covered = set()
    for requirement in requirements:
        if not isinstance(requirement, Mapping):
            continue
        value = requirement.get("covered_by_relation_index")
        if isinstance(value, int) or str(value).isdigit():
            covered.add(int(value))
    return covered == set(range(len(relations)))


def _promote_model_validated_coverage(
    plan: Mapping[str, Any], gate: Mapping[str, Any]
) -> Dict[str, Any]:
    """Promote only a fully certified model plan past the legacy rule parser."""
    wrapped = copy.deepcopy(dict(plan))
    if _plan_coverage_status(wrapped) == PLAN_COMPLETE:
        return wrapped
    if not _model_gate_certifies_complete(wrapped, gate):
        return wrapped
    previous = copy.deepcopy(dict(wrapped.get("plan_coverage") or {}))
    wrapped["plan_coverage"] = {
        "status": PLAN_COMPLETE,
        "reason": "factorized_model_gate_certified_all_semantic_requirements",
        "evidence": "model_based_semantic_admission_dev_not_human_gold",
        "previous_rule_parser_coverage": previous,
    }
    wrapped["plan_coverage_status"] = PLAN_COMPLETE
    wrapped["model_question_contract_promotion"] = {
        "version": SEMANTIC_V3_13_RUNTIME_VERSION,
        "promoted": True,
        "basis": "static_lowering_plus_relation_and_grounding_gate",
        "human_reviewed": False,
        "is_human_gold": False,
        "uses_gold_answer": False,
        "uses_supporting_facts": False,
    }
    return wrapped


def _apply_semantic_admission_v3_13(
    plan: Mapping[str, Any],
    *,
    decision: str,
    origin: str,
    gate: Mapping[str, Any],
    deterministic: bool = False,
    promote_model_contract: bool = False,
) -> Dict[str, Any]:
    wrapped = copy.deepcopy(dict(plan))
    if not deterministic and promote_model_contract:
        wrapped = _promote_model_validated_coverage(wrapped, gate)
    coverage = _plan_coverage_status(wrapped)
    eligible = bool(decision == "PASS" and coverage == PLAN_COMPLETE)
    if deterministic:
        wrapped["question_semantic_validation"] = {
            "version": SEMANTIC_V3_13_RUNTIME_VERSION,
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
            "version": SEMANTIC_V3_13_RUNTIME_VERSION,
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
            "version": SEMANTIC_V3_13_RUNTIME_VERSION,
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
        compiler_mode="rule_first_semantic_v3_13",
        attempt_id="canonical",
    )


def compile_rule_first_semantic_v3_13_query_plan(
    question: str,
    *,
    llm_infer: Optional[Callable[..., Any]],
    model_name: str,
    rule_options: Optional[Mapping[str, Any]] = None,
    generation_max_completion_tokens: int = 650,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Compile one question through the V3.13 canonical admission boundary."""
    started = time.perf_counter()
    rule_plan, rule_trace = compile_evidence_query_plan(
        question,
        mode="rule",
        model_name=model_name,
        rule_options=rule_options,
    )
    trace: Dict[str, Any] = {
        "version": SEMANTIC_V3_13_RUNTIME_VERSION,
        "mode": "rule_first_semantic_v3_13",
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
            "runtime": SEMANTIC_V3_13_RUNTIME_VERSION,
            "runtime_cache": SEMANTIC_V3_13_RUNTIME_CACHE_VERSION,
            "skeleton_prompt": SEMANTIC_V3_13_PROMPT_VERSION,
            "skeleton_cache": SEMANTIC_V3_13_CACHE_VERSION,
            "skeleton_lowering": SEMANTIC_V3_13_PIPELINE_VERSION,
        },
    }
    deterministic_rule, rule_reason = _rule_contract_status(rule_plan)
    trace["rule_admission_precheck"] = rule_reason
    rejected_rule_plan: Dict[str, Any] = {}

    if deterministic_rule:
        selected = _apply_semantic_admission_v3_13(
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
                rejected_rule_plan = _apply_semantic_admission_v3_13(
                    rule_plan,
                    decision=str(gate.get("decision") or "PROTOCOL_INVALID"),
                    origin="rule_model_relation_admission_v3_13",
                    gate=gate,
                )
                if gate.get("decision") == "PASS":
                    trace.update({
                        "selected_source": "rule_model_relation_admission_v3_13",
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
                rejected_rule_plan = _apply_semantic_admission_v3_13(
                    rule_plan,
                    decision="UNKNOWN",
                    origin="rule_model_relation_admission_v3_13",
                    gate={
                        "decision": "UNKNOWN",
                        "reason": "rule_relation_gate_failed",
                    },
                )
        elif llm_infer is None:
            rejected_rule_plan = _apply_semantic_admission_v3_13(
                rule_plan,
                decision="UNKNOWN",
                origin="rule_model_relation_admission_v3_13",
                gate={"decision": "UNKNOWN", "reason": "llm_not_available"},
            )
        else:
            rejected_rule_plan = _apply_semantic_admission_v3_13(
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
            build_semantic_v3_13_skeleton_messages(
                question,
                max_completion_tokens=generation_max_completion_tokens,
            ),
            purpose="canonical_semantic_skeleton_generation_v3_13",
            completion=generation_max_completion_tokens,
        )
        trace["requests"].append(request)
        trace["semantic_skeleton_payload"] = payload
        lowering = lower_semantic_v3_13_skeleton(question, payload)
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
            fallback_candidate = _apply_semantic_admission_v3_13(
                fallback_candidate,
                decision=str(gate.get("decision") or "PROTOCOL_INVALID"),
                origin="llm_semantic_skeleton_v3_13",
                gate=gate,
                promote_model_contract=True,
            )
            eligible = bool(
                (fallback_candidate.get("compiler_stage_status") or {}).get(
                    "experimental_execution_eligible"
                )
            )
            if eligible:
                trace.update({
                    "selected_source": "llm_semantic_skeleton_v3_13",
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
