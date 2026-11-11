"""Rule-first, capability-constrained query compilation for S2RAG.

The LLM fallback emits the same logical IR as the deterministic compiler.  It
does not see retrieval results, reader answers, supporting-fact labels, or QA
gold, and its output cannot execute until it passes the shared static
validator in ``relation_fact_join_utils``.
"""

from __future__ import annotations

import copy
import json
import hashlib
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .late_bound_textual_relation_utils import parse_json_object
from .evidence_query_compiler_v2_utils import (
    HYBRID_V2_1_COMPILER_VERSION,
    HYBRID_V2_1_PROMPT_VERSION,
    HYBRID_V2_2_COMPILER_VERSION,
    HYBRID_V2_2_PROMPT_VERSION,
    HYBRID_V2_COMPILER_VERSION,
    HYBRID_V2_PROMPT_VERSION,
    PHYSICAL_LOWERING_V2_1_VERSION,
    PHYSICAL_LOWERING_V2_2_VERSION,
    PHYSICAL_LOWERING_VERSION,
    PREDICATE_ENDPOINT_SCHEMA_VERSION,
    SEMANTIC_QUERY_IR_V2_1_VERSION,
    SEMANTIC_QUERY_IR_V2_2_VERSION,
    SEMANTIC_QUERY_IR_VERSION,
    build_hybrid_v2_1_messages,
    build_hybrid_v2_2_messages,
    build_hybrid_v2_messages,
    compile_semantic_payload_v2_1,
    compile_semantic_payload_v2_2,
    compile_semantic_payload_v2,
)
from .relation_fact_join_utils import (
    PLAN_COMPLETE,
    PLAN_COVERAGE_CONTRACT_VERSION,
    PREDICATE_FAMILY_TERMS,
    SCHEMA_PREDICATE_FAMILY_TERMS,
    assess_question_plan_coverage,
    attach_logical_query_plan_identity,
    compile_question_obligations,
    compile_stage_aware_fact_join_plan,
    validate_logical_query_plan_static,
)


HYBRID_QUERY_COMPILER_VERSION = "hybrid_query_compiler_v1"
HYBRID_QUERY_COMPILER_PROMPT_VERSION = "hybrid_query_compiler_prompt_v1"
EXECUTOR_CAPABILITY_VERSION = "linear_relation_projection_capability_v1"
RULE_CONTRACT_COMPILER_VERSION = "rule_contract_query_compiler_v1"
RULE_CONTRACT_VALIDATION_VERSION = "deterministic_rule_question_contract_v1"

SUPPORTED_PREDICATE_FAMILIES = tuple(sorted(
    set(PREDICATE_FAMILY_TERMS) | set(SCHEMA_PREDICATE_FAMILY_TERMS)
))
SUPPORTED_OUTPUT_TYPES = {
    "entity", "person", "place", "date", "year", "number", "work",
    "organization", "nationality", "occupation",
}
SUPPORTED_DIRECTIONS = {"forward", "reverse", "either"}
SUPPORTED_SUBTYPES = {
    "birth_name", "birth_date", "birth_place",
    "death_cause", "death_date", "death_place",
    "occupation_role", "employment_organization", "employment_location",
    "employment_generic", "distribution_actual_release",
    "distribution_intended_release", "distribution_premiere",
    *SUPPORTED_PREDICATE_FAMILIES,
}


def compiler_capability_manifest() -> Dict[str, Any]:
    return {
        "version": EXECUTOR_CAPABILITY_VERSION,
        "plan_kinds": ["relation_projection", "path_projection"],
        "answer_operators": ["project"],
        "physical_operators": ["indexed_fact_lookup"],
        "predicate_families": list(SUPPORTED_PREDICATE_FAMILIES),
        "predicate_subtypes": sorted(SUPPORTED_SUBTYPES),
        "output_types": sorted(SUPPORTED_OUTPUT_TYPES),
        "directions": sorted(SUPPORTED_DIRECTIONS),
        "max_relation_hops": 4,
    }


def build_hybrid_query_compiler_messages(
    question: str,
    *,
    max_completion_tokens: int = 700,
) -> list[dict[str, str]]:
    capability = compiler_capability_manifest()
    system = (
        "Compile one natural-language question into the supplied relational "
        "query-plan capability. Use only relations explicitly requested by the "
        "question. Do not answer the question and do not invent intermediate "
        "entities. Intermediate values are variables produced at execution. "
        "Return JSON only. If the question cannot be represented safely, return "
        '{"abstain": true, "reason": "..."}. Otherwise return an object with '
        "final_answer_type and operators. operators is an ordered list; each "
        "item has predicate_families (one or more values from the capability), "
        "relation_tokens (literal relation cues from the question), output_type, "
        "direction (forward, reverse, or either), input_role, output_role, and "
        "optional required_predicate_subtypes. Do not emit entity values, "
        "answers, SQL, filters, free-form code, or unsupported operators."
    )
    user = "\n".join([
        f"Prompt version: {HYBRID_QUERY_COMPILER_PROMPT_VERSION}",
        (
            "Compiler request parameters: "
            f"max_completion_tokens={int(max_completion_tokens)};"
            "temperature=0;response_format=json_object"
        ),
        "Executor capability:",
        json.dumps(capability, ensure_ascii=True, sort_keys=True),
        "Question:",
        str(question or "").strip(),
    ])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _decode_llm_response(value: Any) -> Mapping[str, Any]:
    payload = parse_json_object(value)
    if not isinstance(payload, Mapping):
        raise ValueError("invalid_plan_format")
    return payload


def _normalize_llm_plan(
    question: str,
    payload: Mapping[str, Any],
) -> tuple[Dict[str, Any], str]:
    if bool(payload.get("abstain")):
        return {}, str(payload.get("reason") or "llm_abstained")
    raw_operators = payload.get("operators")
    if not isinstance(raw_operators, Sequence) or isinstance(
        raw_operators, (str, bytes)
    ):
        return {}, "invalid_plan_format"
    if not 1 <= len(raw_operators) <= 4:
        return {}, "unsupported_operator_count"
    final_answer_type = str(
        payload.get("final_answer_type") or "entity"
    ).strip().lower()
    if final_answer_type not in SUPPORTED_OUTPUT_TYPES:
        return {}, "unsupported_answer_type"

    operators = []
    normalized_question = " ".join(str(question or "").casefold().split())
    for index, raw in enumerate(raw_operators):
        if not isinstance(raw, Mapping):
            return {}, f"invalid_plan_format_op{index}"
        forbidden_value_fields = {
            "entity", "entity_value", "bound_entity", "answer",
            "answer_value", "constant", "intermediate_value",
        }
        if any(str(raw.get(field) or "").strip() for field in forbidden_value_fields):
            return {}, f"invented_runtime_value_op{index}"
        families = {
            str(value or "").strip().lower()
            for value in raw.get("predicate_families") or []
            if str(value or "").strip()
        }
        if not families or not families.issubset(SUPPORTED_PREDICATE_FAMILIES):
            return {}, f"unsupported_predicate_or_subtype_op{index}"
        relation_tokens = sorted({
            str(value or "").strip().lower()
            for value in raw.get("relation_tokens") or []
            if str(value or "").strip()
        })
        if not relation_tokens:
            return {}, f"missing_relation_contract_op{index}"
        if any(
            " ".join(token.casefold().split()) not in normalized_question
            for token in relation_tokens
        ):
            return {}, f"relation_token_not_grounded_in_question_op{index}"
        subtypes = {
            str(value or "").strip().lower()
            for value in raw.get("required_predicate_subtypes") or []
            if str(value or "").strip()
        }
        if not subtypes.issubset(SUPPORTED_SUBTYPES):
            return {}, f"unsupported_predicate_or_subtype_op{index}"
        for subtype in subtypes:
            subtype_family = subtype.split("_", 1)[0]
            if subtype_family in {"birth", "death", "employment", "distribution"}:
                if subtype_family not in families:
                    return {}, f"unsupported_predicate_or_subtype_op{index}"
        output_type = str(
            raw.get("output_type")
            or (final_answer_type if index == len(raw_operators) - 1 else "entity")
        ).strip().lower()
        if output_type not in SUPPORTED_OUTPUT_TYPES:
            return {}, f"unsupported_output_type_op{index}"
        direction = str(raw.get("direction") or "forward").strip().lower()
        if direction not in SUPPORTED_DIRECTIONS:
            return {}, f"invalid_direction_or_role_op{index}"
        input_slot = "$root" if index == 0 else f"$v{index}"
        output_slot = (
            "$answer" if index == len(raw_operators) - 1 else f"$v{index + 1}"
        )
        operator = {
            "operator_id": f"op{index}",
            "operator_type": "indexed_fact_lookup",
            "stage_index": index,
            "slot_plan_stage": (
                "final_answer"
                if index == len(raw_operators) - 1
                else "intermediate_binding"
            ),
            "input_slot": input_slot,
            "output_slot": output_slot,
            "predicate_families": sorted(families),
            "relation_tokens": relation_tokens,
            "required_predicate_subtypes": sorted(subtypes),
            "strict_predicate_subtype": bool(subtypes),
            "output_type": output_type,
            "direction": direction,
            "input_role": str(raw.get("input_role") or "subject").strip().lower(),
            "output_role": str(raw.get("output_role") or "object").strip().lower(),
            "join_key_contract": (
                "entity_identity+predicate_subtype+output_type+title_family"
                if subtypes else "exact_normalized_entity_key"
            ),
        }
        if operator["input_role"] not in {"subject", "object", "either"}:
            return {}, f"invalid_direction_or_role_op{index}"
        if operator["output_role"] not in {"subject", "object", "either"}:
            return {}, f"invalid_direction_or_role_op{index}"
        if direction == "forward" and (
            operator["input_role"] != "subject"
            or operator["output_role"] != "object"
        ):
            return {}, f"invalid_direction_or_role_op{index}"
        if direction == "reverse" and (
            operator["input_role"] != "object"
            or operator["output_role"] != "subject"
        ):
            return {}, f"invalid_direction_or_role_op{index}"
        operators.append(operator)

    if str(operators[-1].get("output_type") or "") != final_answer_type:
        return {}, "invalid_answer_projection"

    plan_kind = "relation_projection" if len(operators) == 1 else "path_projection"
    plan = {
        "version": "slot_dependency_query_plan_v3",
        "logical_plan_revision": HYBRID_QUERY_COMPILER_VERSION,
        "enabled": True,
        "question": str(question or ""),
        "plan_kind": plan_kind,
        "answer_operator": "project",
        "final_answer_type": final_answer_type,
        "requires_bridge": len(operators) > 1,
        "slots": ["$root"] + [
            str(row["output_slot"]) for row in operators
        ],
        "operators": operators,
        "steps": operators,
        "operator_count": len(operators),
        "physical_operator_count": len(operators),
        "logical_operators": list(operators),
        "logical_operator_count": len(operators),
        "terminal_operator": None,
        "compile_status": "compiled",
        "compile_reason": "compiled_by_constrained_llm_fallback",
        "llm_declared_plan_contract": True,
    }
    question_contract = compile_question_obligations(question)
    parsed_coverage = assess_question_plan_coverage(question_contract, plan)
    if (
        str(question_contract.get("parse_status") or "") == "parsed"
        and str(parsed_coverage.get("status") or "") != PLAN_COMPLETE
    ):
        return {}, "llm_plan_failed_question_contract"
    coverage = (
        parsed_coverage
        if str(question_contract.get("parse_status") or "") == "parsed"
        else {
            "status": PLAN_COMPLETE,
            "reason": "compiler_declared_obligations_covered",
            "evidence": "llm_plan_not_independent_semantic_gold",
            "required_obligation_count": len(operators),
            "covered_obligation_count": len(operators),
        }
    )
    plan.update({
        "question_obligation_contract": question_contract,
        "question_obligations": list(question_contract.get("obligations") or []),
        "plan_coverage_contract_version": PLAN_COVERAGE_CONTRACT_VERSION,
        "plan_coverage": coverage,
        "plan_coverage_status": str(coverage.get("status") or ""),
    })
    plan = attach_logical_query_plan_identity(
        plan,
        compiler_source="llm_fallback",
        compiler_mode="hybrid_v1",
        attempt_id="attempt1",
    )
    validation = validate_logical_query_plan_static(plan)
    if not validation["valid"]:
        return {}, str(validation["reason"])
    return plan, "validated"


def _attach_deterministic_rule_contract(
    question: str,
    plan: Mapping[str, Any],
) -> tuple[Dict[str, Any], str]:
    """Attach an execution certificate only for independently covered rules.

    Static IR validity is necessary but not sufficient for execution.  This
    boundary reparses the question into obligations and checks the concrete
    rule plan against that independent contract.  It uses no retrieval,
    reader, supporting-fact, or answer labels.
    """
    wrapped = copy.deepcopy(dict(plan or {}))
    static_validation = validate_logical_query_plan_static(wrapped)
    question_contract = compile_question_obligations(question)
    coverage = assess_question_plan_coverage(question_contract, wrapped)
    operators = [dict(row) for row in wrapped.get("operators") or []]
    required_relations = [
        dict(row)
        for row in question_contract.get("obligations") or []
        if bool(row.get("required", True)) and row.get("kind") == "relation"
    ]
    operator_matches = list(coverage.get("operator_matches") or [])

    reason = "deterministic_rule_question_contract_validated"
    eligible = True
    if not bool(static_validation.get("valid")):
        eligible = False
        reason = str(static_validation.get("reason") or "rule_static_invalid")
    elif str(question_contract.get("parse_status") or "") != "parsed":
        eligible = False
        reason = "question_obligation_parse_not_high_confidence"
    elif str(question_contract.get("parse_confidence") or "") != "high":
        eligible = False
        reason = "question_obligation_parse_not_high_confidence"
    elif str(coverage.get("status") or "") != PLAN_COMPLETE:
        eligible = False
        reason = str(coverage.get("reason") or "rule_question_contract_incomplete")
    elif len(operators) != len(required_relations):
        eligible = False
        reason = "rule_relation_operator_count_mismatch"
    elif len(operator_matches) != len(required_relations) or not all(
        bool(row.get("matched")) for row in operator_matches
    ):
        eligible = False
        reason = "rule_operator_obligation_alignment_failed"

    wrapped.update({
        "question_obligation_contract": question_contract,
        "question_obligations": list(question_contract.get("obligations") or []),
        "plan_coverage_contract_version": PLAN_COVERAGE_CONTRACT_VERSION,
        "plan_coverage": coverage,
        "plan_coverage_status": str(coverage.get("status") or ""),
        "question_semantic_validation": {
            "version": RULE_CONTRACT_VALIDATION_VERSION,
            "status": (
                "DETERMINISTIC_RULE_CONTRACT_VALIDATED"
                if eligible
                else "DETERMINISTIC_RULE_CONTRACT_REJECTED"
            ),
            "basis": "independent_rule_question_obligation_contract",
            "independent_question_contract": True,
            "review_decision": "PASS" if eligible else "REJECT",
            "auto_execution_eligible": eligible,
            "experimental_execution_eligible": False,
            "human_reviewed": False,
            "is_human_gold": False,
            "uses_gold_answer": False,
            "uses_supporting_facts": False,
            "uses_retrieval_or_reader": False,
            "reason": reason,
        },
        "compiler_stage_status": {
            "candidate_generated": bool(wrapped),
            "static_valid": bool(static_validation.get("valid")),
            "semantic_admission": "PASS" if eligible else "REJECT",
            "question_coverage": str(coverage.get("status") or ""),
            "auto_execution_eligible": eligible,
            "experimental_execution_eligible": False,
        },
    })
    wrapped = attach_logical_query_plan_identity(
        wrapped,
        compiler_source="rule_deterministic_question_contract_v1",
        compiler_mode="rule_contract_v1",
        attempt_id="attempt0",
    )
    return wrapped, reason


def compile_evidence_query_plan(
    question: str,
    *,
    mode: str = "rule",
    llm_infer: Optional[Callable[..., Any]] = None,
    model_name: str = "",
    max_completion_tokens: int = 700,
    rule_options: Optional[Mapping[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Compile with rules first and at most one constrained LLM fallback."""
    if mode not in {
        "rule", "rule_contract_v1", "hybrid_v1", "hybrid_v2",
        "hybrid_v2_1", "hybrid_v2_2",
    }:
        raise ValueError(f"Unsupported query compiler mode: {mode}")
    started = time.perf_counter()
    rule_plan = compile_stage_aware_fact_join_plan(
        question,
        **dict(rule_options or {}),
    )
    rule_validation = validate_logical_query_plan_static(rule_plan)
    attempts = [{
        "attempt_id": "attempt0",
        "compiler_source": "rule",
        "plan_id": str(rule_plan.get("plan_id") or ""),
        "compile_status": str(rule_plan.get("compile_status") or ""),
        "static_valid": bool(rule_validation["valid"]),
        "reason": str(rule_validation["reason"]),
    }]
    is_v2 = mode == "hybrid_v2"
    is_v2_1 = mode == "hybrid_v2_1"
    is_v2_2 = mode == "hybrid_v2_2"
    is_semantic_v2 = is_v2 or is_v2_1 or is_v2_2
    compiler_version = (
        RULE_CONTRACT_COMPILER_VERSION if mode == "rule_contract_v1"
        else
        HYBRID_V2_2_COMPILER_VERSION if is_v2_2
        else HYBRID_V2_1_COMPILER_VERSION if is_v2_1
        else HYBRID_V2_COMPILER_VERSION if is_v2
        else HYBRID_QUERY_COMPILER_VERSION
    )
    prompt_version = (
        HYBRID_V2_2_PROMPT_VERSION if is_v2_2
        else HYBRID_V2_1_PROMPT_VERSION if is_v2_1
        else HYBRID_V2_PROMPT_VERSION if is_v2
        else HYBRID_QUERY_COMPILER_PROMPT_VERSION
    )
    trace: Dict[str, Any] = {
        "version": compiler_version,
        "mode": mode,
        "model": str(model_name or ""),
        "prompt_version": prompt_version,
        "capability_version": EXECUTOR_CAPABILITY_VERSION,
        "attempts": attempts,
        "llm_attempted": False,
        "llm_requests": 0,
        "cache_hit": None,
        "input_tokens": None,
        "output_tokens": None,
    }
    if is_semantic_v2:
        trace.update({
            "semantic_ir_version": (
                SEMANTIC_QUERY_IR_V2_2_VERSION if is_v2_2
                else SEMANTIC_QUERY_IR_V2_1_VERSION if is_v2_1
                else SEMANTIC_QUERY_IR_VERSION
            ),
            "physical_lowering_version": (
                PHYSICAL_LOWERING_V2_2_VERSION if is_v2_2
                else PHYSICAL_LOWERING_V2_1_VERSION if is_v2_1
                else PHYSICAL_LOWERING_VERSION
            ),
            "predicate_endpoint_schema_version": PREDICATE_ENDPOINT_SCHEMA_VERSION,
        })
    if rule_validation["valid"]:
        if mode == "rule_contract_v1":
            rule_plan, admission_reason = _attach_deterministic_rule_contract(
                question, rule_plan
            )
            semantic_validation = dict(
                rule_plan.get("question_semantic_validation") or {}
            )
            trace.update({
                "selected_plan_id": str(rule_plan.get("plan_id") or ""),
                "selected_source": "rule_deterministic_question_contract_v1",
                "reason": admission_reason,
                "semantic_admitted": bool(
                    semantic_validation.get("auto_execution_eligible")
                ),
                "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
            })
            rule_plan["query_compiler_trace"] = trace
            return rule_plan, trace
        trace.update({
            "selected_plan_id": str(rule_plan.get("plan_id") or ""),
            "selected_source": "rule",
            "reason": "rule_plan_static_valid",
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        rule_plan["query_compiler_trace"] = trace
        return rule_plan, trace
    if mode in {"rule", "rule_contract_v1"}:
        trace.update({
            "selected_plan_id": str(rule_plan.get("plan_id") or ""),
            "selected_source": "rule",
            "reason": "rule_pattern_unmatched",
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        rule_plan["query_compiler_trace"] = trace
        return rule_plan, trace
    if llm_infer is None:
        trace.update({
            "selected_plan_id": "",
            "selected_source": "none",
            "reason": "llm_not_attempted",
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        rule_plan["query_compiler_trace"] = trace
        return rule_plan, trace

    trace["llm_attempted"] = True
    trace["llm_requests"] = 1
    try:
        messages = (
            build_hybrid_v2_2_messages(
                question,
                max_completion_tokens=max_completion_tokens,
            )
            if is_v2_2 else
            build_hybrid_v2_1_messages(
                question,
                max_completion_tokens=max_completion_tokens,
            )
            if is_v2_1 else
            build_hybrid_v2_messages(
                question,
                max_completion_tokens=max_completion_tokens,
            )
            if is_v2 else
            build_hybrid_query_compiler_messages(
                question,
                max_completion_tokens=max_completion_tokens,
            )
        )
        response = llm_infer(
            messages,
            max_completion_tokens=max(64, int(max_completion_tokens)),
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        if isinstance(response, tuple):
            content = response[0]
            metadata = dict(response[1] or {}) if len(response) > 1 else {}
            cache_hit = (
                bool(response[2])
                if len(response) > 2
                else bool(metadata.get("cache_hit"))
            )
        else:
            content = response
            metadata = {}
            cache_hit = False
        trace.update({
            "cache_hit": cache_hit,
            "input_tokens": metadata.get("prompt_tokens"),
            "output_tokens": metadata.get("completion_tokens"),
        })
        payload = _decode_llm_response(content)
        trace["response_sha256"] = hashlib.sha256(
            str(content or "").encode("utf-8")
        ).hexdigest()
        trace["llm_response_payload"] = dict(payload)
        if is_semantic_v2:
            semantic_ir, llm_plan, reason = (
                compile_semantic_payload_v2_2(question, payload)
                if is_v2_2 else
                compile_semantic_payload_v2_1(question, payload)
                if is_v2_1 else
                compile_semantic_payload_v2(question, payload)
            )
            trace["semantic_ir"] = semantic_ir or None
        else:
            llm_plan, reason = _normalize_llm_plan(question, payload)
    except Exception as exc:
        llm_plan = {}
        if type(exc).__name__ == "CompilerBudgetExhausted":
            reason = "llm_budget_exhausted"
        elif type(exc).__name__ == "CacheOnlyMissError":
            reason = "llm_cache_miss_not_executed"
        else:
            reason = "llm_call_failed" if not isinstance(exc, ValueError) else str(exc)
        trace["error_type"] = type(exc).__name__
        trace["error_preview"] = str(exc)[:240]

    attempts.append({
        "attempt_id": "attempt1",
        "compiler_source": "llm_semantic_fallback" if is_semantic_v2 else "llm_fallback",
        "plan_id": str(llm_plan.get("plan_id") or ""),
        "compile_status": str(llm_plan.get("compile_status") or "unsupported"),
        "static_valid": bool(llm_plan),
        "reason": str(reason),
    })
    selected = llm_plan if llm_plan else rule_plan
    trace.update({
        "selected_plan_id": str(selected.get("plan_id") or ""),
        "selected_source": (
            "llm_semantic_fallback" if is_semantic_v2 and llm_plan
            else "llm_fallback" if llm_plan
            else "none"
        ),
        "reason": str(reason),
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
    })
    selected["query_compiler_trace"] = trace
    return selected, trace
