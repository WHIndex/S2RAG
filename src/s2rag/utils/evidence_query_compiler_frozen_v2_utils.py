"""Single-entry frozen V2 compiler used by the new-sample convergence run.

The pipeline is deliberately fixed before execution: rule compilation first,
one V2.2 semantic request lowered by the V2.3 deterministic lowerer, and at
most one validation-guided V2.4 repair for a pre-registered failure reason.
No retrieval, witness, reader, or answer signal can affect plan selection.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Dict, Mapping, Optional

from .evidence_query_compiler_utils import compile_evidence_query_plan
from .evidence_query_compiler_v2_utils import (
    HYBRID_V2_2_CACHE_KEY_VERSION,
    HYBRID_V2_2_PROMPT_VERSION,
    HYBRID_V2_4_REPAIR_CACHE_KEY_VERSION,
    HYBRID_V2_4_REPAIR_PROMPT_VERSION,
    PHYSICAL_LOWERING_V2_3_VERSION,
    PREDICATE_ENDPOINT_SCHEMA_VERSION,
    SEMANTIC_QUERY_IR_V2_3_VERSION,
    build_hybrid_v2_2_messages,
    build_hybrid_v2_4_repair_messages,
    compile_semantic_payload_v2_3,
    is_hybrid_v2_4_repair_eligible,
)
from .late_bound_textual_relation_utils import parse_json_object


FROZEN_V2_PIPELINE_VERSION = "rule_first_frozen_semantic_v2_20261028"
FROZEN_V2_CACHE_CONTRACT_VERSION = "frozen_v2_pipeline_cache_v1"


def _valid(plan: Mapping[str, Any]) -> bool:
    phase = dict(plan.get("compile_phase_status") or {})
    return bool(phase.get("plan_static_valid"))


def _auto_execution_eligible(plan: Mapping[str, Any]) -> bool:
    validation = plan.get("question_semantic_validation")
    if not isinstance(validation, Mapping):


        return True
    return bool(validation.get("auto_execution_eligible"))


def _record_structural_candidate(
    trace: Dict[str, Any],
    *,
    plan: Mapping[str, Any],
    source: str,
    attempt_id: str,
) -> None:
    validation = dict(plan.get("question_semantic_validation") or {})
    trace.update({
        "selected_source": "none",
        "selected_attempt_id": "",
        "structural_candidate_source": source,
        "structural_candidate_attempt_id": attempt_id,
        "structural_candidate_valid": True,
        "question_semantic_validation": validation,
        "auto_execution_eligible": False,
        "reason": "question_semantic_review_required",
    })


def _response_parts(response: Any) -> tuple[str, Dict[str, Any], bool]:
    if not isinstance(response, tuple):
        return str(response or ""), {}, False
    content = str(response[0] or "")
    metadata = dict(response[1] or {}) if len(response) > 1 else {}
    cache_hit = bool(response[2]) if len(response) > 2 else bool(
        metadata.get("cache_hit")
    )
    return content, metadata, cache_hit


def _request_record(
    *,
    attempt_id: str,
    purpose: str,
    messages: list[dict[str, str]],
    content: str,
    metadata: Mapping[str, Any],
    cache_hit: bool,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return {
        "attempt_id": attempt_id,
        "purpose": purpose,
        "messages": messages,
        "request_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "response_payload": dict(payload),
        "cache_hit": bool(cache_hit),
        "input_tokens": metadata.get("prompt_tokens"),
        "output_tokens": metadata.get("completion_tokens"),
    }


def compile_frozen_v2_query_plan(
    question: str,
    *,
    llm_infer: Optional[Callable[..., Any]],
    model_name: str,
    rule_options: Optional[Mapping[str, Any]] = None,
    initial_max_completion_tokens: int = 400,
    repair_max_completion_tokens: int = 500,
    allow_repair: bool = True,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Compile one question using the frozen rule-first V2 policy."""
    started = time.perf_counter()
    rule_plan, rule_trace = compile_evidence_query_plan(
        question,
        mode="rule",
        model_name=model_name,
        rule_options=rule_options,
    )
    trace: Dict[str, Any] = {
        "version": FROZEN_V2_PIPELINE_VERSION,
        "cache_contract_version": FROZEN_V2_CACHE_CONTRACT_VERSION,
        "model": str(model_name or ""),
        "rule_trace": rule_trace,
        "requests": [],
        "request_count": 0,
        "selected_source": "none",
        "selected_attempt_id": "",
        "reason": "",
        "versions": {
            "initial_prompt": HYBRID_V2_2_PROMPT_VERSION,
            "initial_cache": HYBRID_V2_2_CACHE_KEY_VERSION,
            "repair_prompt": HYBRID_V2_4_REPAIR_PROMPT_VERSION,
            "repair_cache": HYBRID_V2_4_REPAIR_CACHE_KEY_VERSION,
            "semantic_ir": SEMANTIC_QUERY_IR_V2_3_VERSION,
            "physical_lowering": PHYSICAL_LOWERING_V2_3_VERSION,
            "predicate_endpoint_schema": PREDICATE_ENDPOINT_SCHEMA_VERSION,
        },
        "selection_observed_execution": False,
        "gold_or_reader_input": False,
    }
    if _valid(rule_plan):
        trace.update({
            "selected_source": "rule",
            "selected_attempt_id": "attempt0",
            "reason": "rule_plan_static_valid",
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        selected = dict(rule_plan)
        selected["query_compiler_trace"] = trace
        return selected, trace
    if llm_infer is None:
        trace.update({
            "selected_source": "none",
            "reason": "llm_not_available",
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        return {}, trace

    initial_messages = build_hybrid_v2_2_messages(
        question,
        max_completion_tokens=int(initial_max_completion_tokens),
    )
    try:
        response = llm_infer(
            initial_messages,
            max_completion_tokens=max(64, int(initial_max_completion_tokens)),
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content, metadata, cache_hit = _response_parts(response)
        payload = parse_json_object(content)
        if not isinstance(payload, Mapping):
            raise ValueError("invalid_initial_semantic_payload")
        trace["requests"].append(_request_record(
            attempt_id="attempt1",
            purpose="semantic_generation",
            messages=initial_messages,
            content=content,
            metadata=metadata,
            cache_hit=cache_hit,
            payload=payload,
        ))
        trace["request_count"] = 1
        semantic_ir, plan, reason = compile_semantic_payload_v2_3(
            question, payload
        )
        trace.update({
            "initial_semantic_ir": semantic_ir,
            "initial_reason": reason,
            "initial_static_valid": bool(plan) and reason == "validated",
        })
    except Exception as exc:
        trace.update({
            "reason": (
                "compiler_budget_exhausted"
                if type(exc).__name__ == "CompilerBudgetExhausted"
                else "compiler_cache_miss"
                if type(exc).__name__ == "CacheOnlyMissError"
                else "initial_compiler_call_failed"
            ),
            "error_type": type(exc).__name__,
            "error_preview": str(exc)[:240],
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        return {}, trace

    if plan and reason == "validated" and _auto_execution_eligible(plan):
        trace.update({
            "selected_source": "semantic_v2_initial",
            "selected_attempt_id": "attempt1",
            "reason": "validated",
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        selected = dict(plan)
        selected["query_compiler_trace"] = trace
        return selected, trace
    if plan and reason == "validated":
        _record_structural_candidate(
            trace,
            plan=plan,
            source="semantic_v2_initial",
            attempt_id="attempt1",
        )
        trace["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 6)
        selected = dict(plan)
        selected["query_compiler_trace"] = trace
        return selected, trace

    repair_eligible = bool(
        allow_repair and is_hybrid_v2_4_repair_eligible(reason)
    )
    trace["repair_eligible"] = repair_eligible
    if not repair_eligible:
        trace.update({
            "reason": reason,
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        return {}, trace

    repair_messages = build_hybrid_v2_4_repair_messages(
        question,
        previous_payload=payload,
        validator_reason=reason,
        max_completion_tokens=int(repair_max_completion_tokens),
    )
    try:
        response = llm_infer(
            repair_messages,
            max_completion_tokens=max(64, int(repair_max_completion_tokens)),
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content, metadata, cache_hit = _response_parts(response)
        repaired_payload = parse_json_object(content)
        if not isinstance(repaired_payload, Mapping):
            raise ValueError("invalid_repair_semantic_payload")
        trace["requests"].append(_request_record(
            attempt_id="attempt2",
            purpose="validation_guided_repair",
            messages=repair_messages,
            content=content,
            metadata=metadata,
            cache_hit=cache_hit,
            payload=repaired_payload,
        ))
        trace["request_count"] = 2
        semantic_ir, repaired_plan, repair_reason = compile_semantic_payload_v2_3(
            question, repaired_payload
        )
        trace.update({
            "repair_semantic_ir": semantic_ir,
            "repair_reason": repair_reason,
            "repair_static_valid": bool(repaired_plan)
            and repair_reason == "validated",
        })
    except Exception as exc:
        trace.update({
            "reason": (
                "compiler_budget_exhausted"
                if type(exc).__name__ == "CompilerBudgetExhausted"
                else "compiler_cache_miss"
                if type(exc).__name__ == "CacheOnlyMissError"
                else "repair_compiler_call_failed"
            ),
            "error_type": type(exc).__name__,
            "error_preview": str(exc)[:240],
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        return {}, trace

    if repaired_plan and repair_reason == "validated" and _auto_execution_eligible(
        repaired_plan
    ):
        trace.update({
            "selected_source": "semantic_v2_repair",
            "selected_attempt_id": "attempt2",
            "reason": "validated",
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        selected = dict(repaired_plan)
        selected["query_compiler_trace"] = trace
        return selected, trace
    if repaired_plan and repair_reason == "validated":
        _record_structural_candidate(
            trace,
            plan=repaired_plan,
            source="semantic_v2_repair",
            attempt_id="attempt2",
        )
        trace["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 6)
        selected = dict(repaired_plan)
        selected["query_compiler_trace"] = trace
        return selected, trace
    trace.update({
        "reason": repair_reason,
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
    })
    return {}, trace
