"""Targeted V3.2 semantic frontend with a concrete, non-echoable protocol.

V3.2 changes only the question-to-semantic-IR generation protocol. It reuses
the frozen V2.3 deterministic lowering and the strict V3.1 independent
relation-coverage review. Retrieval, execution, COMPLETE, VERIFY, BSWC, and
Reader behavior remain outside this module.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .evidence_query_compiler_semantic_v3_1_utils import (
    MODEL_BASED_SEMANTIC_ADMISSION_DEV,
    _request_record,
    _response_parts,
    _schema_catalog,
    _valid,
    build_semantic_v3_1_review_messages,
    validate_semantic_v3_1_review,
)
from .evidence_query_compiler_utils import compile_evidence_query_plan
from .evidence_query_compiler_v2_utils import (
    PHYSICAL_LOWERING_V2_3_VERSION,
    PREDICATE_ENDPOINT_SCHEMA_VERSION,
    SEMANTIC_QUERY_IR_V2_3_VERSION,
    SUPPORTED_ANSWER_OPERATIONS,
    SUPPORTED_ENTITY_TYPES,
    SUPPORTED_SUBTYPES,
    compile_semantic_payload_v2_3,
    validate_v2_physical_plan,
)
from .late_bound_textual_relation_utils import parse_json_object
from .relation_fact_join_utils import PLAN_COMPLETE, attach_logical_query_plan_identity


SEMANTIC_V3_2_PIPELINE_VERSION = "rule_first_semantic_v3_2_20261029"
SEMANTIC_V3_2_GENERATION_PROMPT_VERSION = "semantic_query_generation_v3_2"
SEMANTIC_V3_2_GENERATION_CACHE_VERSION = "semantic_v3_2_generation_cache_v1"
SEMANTIC_V3_2_REVIEW_PROMPT_VERSION = "independent_relation_coverage_review_v3_2"
SEMANTIC_V3_2_REVIEW_CACHE_VERSION = "semantic_v3_2_review_cache_v1"

ABSTAIN_REASON_CODES = frozenset({
    "AMBIGUOUS_START_ENTITY",
    "SCHEMA_ALIGNMENT_UNCERTAIN",
    "UNSUPPORTED_AGGREGATION",
    "UNSUPPORTED_COMPARISON",
    "UNSUPPORTED_ENTITY_TYPE",
    "UNSUPPORTED_PLAN_SHAPE",
    "UNSUPPORTED_RELATION",
})

_PLACEHOLDER_FRAGMENTS = (
    "$semantic_name",
    "$terminal_semantic_variable",
    "allowed family",
    "allowed operation",
    "allowed type",
    "catalog_family",
    "exact endpoint role",
    "exact named question span",
    "exact question span",
    "optional allowed subtype",
    "same exact cue as relation",
    "same type as relation output",
    "terminal only, else empty",
)

_REQUIRED_PLAN_FIELDS = frozenset({
    "start_entities",
    "relations",
    "question_requirements",
    "answer_operation",
    "final_answer_type",
    "final_answer_category",
    "final_answer_variable",
})


def _valid_examples() -> list[dict[str, Any]]:
    """Return concrete protocol examples that pass the frozen V2.3 lowerer."""
    return [
        {
            "question": "Where did the director of Film Z study?",
            "output": {
                "abstain": False,
                "start_entities": [{
                    "mention": "Film Z", "variable": "$film", "semantic_type": "work",
                }],
                "relations": [
                    {
                        "relation_cues": ["director"],
                        "predicate_family": "direction",
                        "predicate_subtype": "",
                        "canonical_relation": "",
                        "input_variable": "$film",
                        "output_variable": "$director",
                        "input_semantic_role": "work",
                        "output_semantic_role": "director",
                        "semantic_direction": "canonical",
                        "output_entity_type": "person",
                    },
                    {
                        "relation_cues": ["study"],
                        "predicate_family": "education",
                        "predicate_subtype": "",
                        "canonical_relation": "",
                        "input_variable": "$director",
                        "output_variable": "$institution",
                        "input_semantic_role": "person",
                        "output_semantic_role": "education_institution",
                        "semantic_direction": "canonical",
                        "output_entity_type": "organization",
                    },
                ],
                "question_requirements": [
                    {
                        "requirement_id": "q0", "covered_by_relation_index": 0,
                        "relation_cues": ["director"],
                        "expected_output_type": "person", "answer_category": "",
                    },
                    {
                        "requirement_id": "q1", "covered_by_relation_index": 1,
                        "relation_cues": ["study"],
                        "expected_output_type": "organization", "answer_category": "",
                    },
                ],
                "answer_operation": "project",
                "final_answer_type": "organization",
                "final_answer_category": "",
                "final_answer_variable": "$institution",
            },
        },
        {
            "question": "What is the birthday of Alice Example?",
            "output": {
                "abstain": False,
                "start_entities": [{
                    "mention": "Alice Example", "variable": "$person", "semantic_type": "person",
                }],
                "relations": [{
                    "relation_cues": ["birthday"],
                    "predicate_family": "birth",
                    "predicate_subtype": "birth_date",
                    "canonical_relation": "",
                    "input_variable": "$person",
                    "output_variable": "$date",
                    "input_semantic_role": "person",
                    "output_semantic_role": "birth_value",
                    "semantic_direction": "canonical",
                    "output_entity_type": "date",
                }],
                "question_requirements": [{
                    "requirement_id": "q0", "covered_by_relation_index": 0,
                    "relation_cues": ["birthday"],
                    "expected_output_type": "date", "answer_category": "",
                }],
                "answer_operation": "project",
                "final_answer_type": "date",
                "final_answer_category": "",
                "final_answer_variable": "$date",
            },
        },
        {
            "question": "Who is the spouse of the producer of Film Z?",
            "output": {
                "abstain": False,
                "start_entities": [{
                    "mention": "Film Z", "variable": "$film", "semantic_type": "work",
                }],
                "relations": [
                    {
                        "relation_cues": ["producer"],
                        "predicate_family": "production",
                        "predicate_subtype": "",
                        "canonical_relation": "",
                        "input_variable": "$film",
                        "output_variable": "$producer",
                        "input_semantic_role": "work",
                        "output_semantic_role": "producer",
                        "semantic_direction": "canonical",
                        "output_entity_type": "person",
                    },
                    {
                        "relation_cues": ["spouse"],
                        "predicate_family": "spouse_of",
                        "predicate_subtype": "",
                        "canonical_relation": "spouse_of",
                        "input_variable": "$producer",
                        "output_variable": "$spouse",
                        "input_semantic_role": "spouse",
                        "output_semantic_role": "spouse",
                        "semantic_direction": "canonical",
                        "output_entity_type": "person",
                    },
                ],
                "question_requirements": [
                    {
                        "requirement_id": "q0", "covered_by_relation_index": 0,
                        "relation_cues": ["producer"],
                        "expected_output_type": "person", "answer_category": "",
                    },
                    {
                        "requirement_id": "q1", "covered_by_relation_index": 1,
                        "relation_cues": ["spouse"],
                        "expected_output_type": "person", "answer_category": "",
                    },
                ],
                "answer_operation": "project",
                "final_answer_type": "person",
                "final_answer_category": "",
                "final_answer_variable": "$spouse",
            },
        },
        {
            "question": "Who is Alice Example's mother?",
            "output": {
                "abstain": False,
                "start_entities": [{
                    "mention": "Alice Example", "variable": "$child", "semantic_type": "person",
                }],
                "relations": [{
                    "relation_cues": ["mother"],
                    "predicate_family": "mother_of",
                    "predicate_subtype": "",
                    "canonical_relation": "mother_of",
                    "input_variable": "$child",
                    "output_variable": "$mother",
                    "input_semantic_role": "child",
                    "output_semantic_role": "mother",
                    "semantic_direction": "inverse",
                    "output_entity_type": "person",
                }],
                "question_requirements": [{
                    "requirement_id": "q0", "covered_by_relation_index": 0,
                    "relation_cues": ["mother"],
                    "expected_output_type": "person", "answer_category": "",
                }],
                "answer_operation": "project",
                "final_answer_type": "person",
                "final_answer_category": "",
                "final_answer_variable": "$mother",
            },
        },
    ]


def _abstain_example() -> dict[str, Any]:
    return {
        "question": "How many people live in City Z?",
        "output": {
            "abstain": True,
            "abstain_reason_code": "UNSUPPORTED_AGGREGATION",
            "unsupported_question_spans": ["How many"],
            "reason": "Counting residents requires an aggregation operator that is not supported.",
        },
    }


def build_semantic_v3_2_generation_messages(
    question: str,
    *,
    max_completion_tokens: int = 850,
) -> list[dict[str, str]]:
    """Build a semantic-only request without an executable placeholder template."""
    system = (
        "Compile the question into one semantic query JSON object; never answer it. "
        "Copy named start instances and relation cues exactly from the question. A role "
        "description, WH phrase, unknown intermediate, or requested answer is a variable, "
        "never a start entity. Emit every relation-bearing clause needed for the answer in "
        "dependency order. Each edge consumes an already bound variable and produces one new "
        "variable. For project, final_answer_variable must equal the last edge output_variable. "
        "Use exact predicate families, endpoint roles, entity types, and optional subtypes from "
        "the supplied catalog. Never emit instructional placeholders or generic family names. "
        "Use canonical direction for left_role to right_role and inverse for right_role to "
        "left_role. Every relation must have exactly one question_requirement with identical "
        "relation_cues and output type. Do not omit, substitute, or invent a relation to make a "
        "plan executable. If the complete question is not safely representable, emit only the "
        "documented abstain object with a fixed reason code and grounded unsupported span. "
        "Return JSON only."
    )
    user_payload = {
        "protocol": SEMANTIC_V3_2_GENERATION_PROMPT_VERSION,
        "cache_contract": SEMANTIC_V3_2_GENERATION_CACHE_VERSION,
        "semantic_ir": SEMANTIC_QUERY_IR_V2_3_VERSION,
        "physical_lowering": PHYSICAL_LOWERING_V2_3_VERSION,
        "schema_version": PREDICATE_ENDPOINT_SCHEMA_VERSION,
        "max_completion_tokens": int(max_completion_tokens),
        "required_plan_fields": sorted(_REQUIRED_PLAN_FIELDS | {"abstain"}),
        "allowed_abstain_reason_codes": sorted(ABSTAIN_REASON_CODES),
        "allowed_answer_operations": sorted(SUPPORTED_ANSWER_OPERATIONS),
        "allowed_entity_types": sorted(SUPPORTED_ENTITY_TYPES),
        "allowed_predicate_subtypes": sorted(SUPPORTED_SUBTYPES),
        "endpoint_schema": _schema_catalog(),
        "worked_examples": _valid_examples() + [_abstain_example()],
        "question": str(question or "").strip(),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True)},
    ]


def build_semantic_v3_2_review_messages(
    question: str,
    plan: Mapping[str, Any],
) -> list[dict[str, str]]:
    messages = build_semantic_v3_1_review_messages(question, plan)
    payload = json.loads(messages[1]["content"])
    payload["protocol"] = SEMANTIC_V3_2_REVIEW_PROMPT_VERSION
    payload["cache_contract"] = SEMANTIC_V3_2_REVIEW_CACHE_VERSION
    return [messages[0], {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}]


def _all_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        values = []
        for key, item in value.items():
            values.extend(_all_strings(key))
            values.extend(_all_strings(item))
        return values
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = []
        for item in value:
            values.extend(_all_strings(item))
        return values
    return []


def validate_semantic_v3_2_generation_payload(
    question: str,
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """Validate protocol compliance without repairing semantic content."""
    if not isinstance(payload, Mapping):
        return {}, "invalid_generation_payload"
    folded_strings = [value.strip().casefold() for value in _all_strings(payload)]
    if any(
        fragment in value
        for value in folded_strings
        for fragment in _PLACEHOLDER_FRAGMENTS
    ):
        return {}, "schema_placeholder_echo"

    if payload.get("abstain") is True:
        code = str(payload.get("abstain_reason_code") or "").strip().upper()
        if code not in ABSTAIN_REASON_CODES:
            return {}, "invalid_abstain_reason_code"
        spans = [
            str(value).strip()
            for value in payload.get("unsupported_question_spans") or []
            if str(value).strip()
        ]
        folded_question = str(question or "").casefold()
        if not spans or any(span.casefold() not in folded_question for span in spans):
            return {}, "ungrounded_abstain_span"
        if not str(payload.get("reason") or "").strip():
            return {}, "missing_abstain_reason"
        if payload.get("relations") or payload.get("start_entities"):
            return {}, "abstain_contains_partial_plan"
        normalized = {
            "abstain": True,
            "abstain_reason_code": code,
            "unsupported_question_spans": spans,
            "reason": str(payload.get("reason") or "").strip(),
        }
        return normalized, f"abstain_{code.casefold()}"

    if payload.get("abstain") is not False:
        return {}, "missing_boolean_abstain_flag"
    missing = sorted(field for field in _REQUIRED_PLAN_FIELDS if field not in payload)
    if missing:
        return {}, "generation_contract_missing_fields:" + ",".join(missing)
    relations = payload.get("relations")
    requirements = payload.get("question_requirements")
    if not isinstance(relations, list) or not relations:
        return {}, "generation_contract_missing_relations"
    if not isinstance(requirements, list) or len(requirements) != len(relations):
        return {}, "generation_contract_requirement_count_mismatch"
    terminal = relations[-1] if isinstance(relations[-1], Mapping) else {}
    if str(payload.get("final_answer_variable") or "").strip() != str(
        terminal.get("output_variable") or ""
    ).strip():
        return {}, "generation_contract_final_variable_not_terminal"
    return dict(payload), "validated"


def _admit_v3_2_plan(plan: Mapping[str, Any], review: Mapping[str, Any]) -> Dict[str, Any]:
    admitted = dict(plan)
    admitted["plan_coverage"] = {
        "status": PLAN_COMPLETE,
        "reason": "independent_relation_coverage_review_pass",
        "evidence": MODEL_BASED_SEMANTIC_ADMISSION_DEV,
        "prior_structural_coverage": dict(admitted.get("plan_coverage") or {}),
    }
    admitted["plan_coverage_status"] = PLAN_COMPLETE
    admitted["question_semantic_validation"] = {
        "version": "question_semantic_validation_v3_2",
        "status": MODEL_BASED_SEMANTIC_ADMISSION_DEV,
        "basis": "independent_relation_coverage_model_review",
        "admission_source": MODEL_BASED_SEMANTIC_ADMISSION_DEV,
        "review_decision": "PASS",
        "review_origin": "MODEL_BASED_DEVELOPMENT_ONLY",
        "human_review_status": "NOT_PERFORMED",
        "human_reviewed": False,
        "is_human_gold": False,
        "auto_execution_eligible": False,
        "experimental_execution_eligible": True,
        "review": dict(review),
    }
    admitted["model_based_semantic_admission"] = {
        "version": SEMANTIC_V3_2_REVIEW_PROMPT_VERSION,
        "admission_source": MODEL_BASED_SEMANTIC_ADMISSION_DEV,
        "experimental_execution_eligible": True,
        "human_reviewed": False,
        "is_human_gold": False,
        "review": dict(review),
    }
    return attach_logical_query_plan_identity(
        admitted,
        compiler_source="llm_semantic_v3_2_model_reviewed",
        compiler_mode="rule_first_semantic_v3_2",
        attempt_id="attempt1",
    )


def compile_rule_first_semantic_v3_2_query_plan(
    question: str,
    *,
    llm_infer: Optional[Callable[..., Any]],
    model_name: str,
    rule_options: Optional[Mapping[str, Any]] = None,
    generation_max_completion_tokens: int = 850,
    review_max_completion_tokens: int = 900,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Compile Rule first, then one V3.2 generation and one strict review."""
    started = time.perf_counter()
    rule_plan, rule_trace = compile_evidence_query_plan(
        question, mode="rule", model_name=model_name, rule_options=rule_options
    )
    if _valid(rule_plan):
        return rule_plan, rule_trace

    trace: Dict[str, Any] = {
        "version": SEMANTIC_V3_2_PIPELINE_VERSION,
        "mode": "rule_first_semantic_v3_2",
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
        "versions": {
            "generation_prompt": SEMANTIC_V3_2_GENERATION_PROMPT_VERSION,
            "generation_cache": SEMANTIC_V3_2_GENERATION_CACHE_VERSION,
            "semantic_ir": SEMANTIC_QUERY_IR_V2_3_VERSION,
            "physical_lowering": PHYSICAL_LOWERING_V2_3_VERSION,
            "predicate_endpoint_schema": PREDICATE_ENDPOINT_SCHEMA_VERSION,
            "review_prompt": SEMANTIC_V3_2_REVIEW_PROMPT_VERSION,
            "review_cache": SEMANTIC_V3_2_REVIEW_CACHE_VERSION,
        },
    }
    if llm_infer is None:
        trace["reason"] = "llm_not_available"
        trace["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 6)
        rule_plan["query_compiler_trace"] = trace
        return rule_plan, trace

    generation_messages = build_semantic_v3_2_generation_messages(
        question, max_completion_tokens=generation_max_completion_tokens
    )
    try:
        response = llm_infer(
            generation_messages,
            max_completion_tokens=max(64, int(generation_max_completion_tokens)),
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content, metadata, cache_hit = _response_parts(response)
        raw_payload = parse_json_object(content)
        if not isinstance(raw_payload, Mapping):
            raise ValueError("invalid_semantic_generation_payload")
        trace["requests"].append(_request_record(
            attempt_id="attempt1",
            purpose="semantic_generation_v3_2",
            messages=generation_messages,
            content=content,
            metadata=metadata,
            cache_hit=cache_hit,
            payload=raw_payload,
        ))
        trace["request_count"] = 1
        payload, protocol_reason = validate_semantic_v3_2_generation_payload(
            question, raw_payload
        )
        trace["generation_protocol_reason"] = protocol_reason
        if protocol_reason.startswith("abstain_"):
            trace.update({
                "reason": protocol_reason,
                "abstain_reason_code": payload.get("abstain_reason_code"),
                "unsupported_question_spans": payload.get("unsupported_question_spans"),
                "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
            })
            rule_plan["query_compiler_trace"] = trace
            return rule_plan, trace
        if protocol_reason != "validated":
            trace.update({
                "reason": protocol_reason,
                "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
            })
            rule_plan["query_compiler_trace"] = trace
            return rule_plan, trace
        semantic_ir, candidate_plan, lowering_reason = compile_semantic_payload_v2_3(
            question, payload, allow_external_plan_review=True
        )
        trace.update({
            "semantic_ir": semantic_ir or None,
            "lowering_reason": str(lowering_reason),
            "structural_candidate_valid": bool(candidate_plan) and lowering_reason == "validated",
        })
    except Exception as exc:
        trace.update({
            "reason": "semantic_generation_failed",
            "error_type": type(exc).__name__,
            "error_preview": str(exc)[:240],
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        rule_plan["query_compiler_trace"] = trace
        return rule_plan, trace

    if not candidate_plan or lowering_reason != "validated":
        trace.update({
            "reason": str(lowering_reason),
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        rule_plan["query_compiler_trace"] = trace
        return rule_plan, trace

    review_messages = build_semantic_v3_2_review_messages(question, candidate_plan)
    try:
        response = llm_infer(
            review_messages,
            max_completion_tokens=max(64, int(review_max_completion_tokens)),
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content, metadata, cache_hit = _response_parts(response)
        review_payload = parse_json_object(content)
        if not isinstance(review_payload, Mapping):
            raise ValueError("invalid_semantic_review_payload")
        trace["requests"].append(_request_record(
            attempt_id="attempt2",
            purpose="independent_relation_coverage_review_v3_2",
            messages=review_messages,
            content=content,
            metadata=metadata,
            cache_hit=cache_hit,
            payload=review_payload,
        ))
        trace["request_count"] = 2
        review, review_reason = validate_semantic_v3_1_review(
            question, candidate_plan, review_payload
        )
        trace.update({
            "semantic_review": review or None,
            "semantic_review_validation_reason": review_reason,
            "semantic_review_decision": str(review.get("decision") or "UNKNOWN"),
        })
    except Exception as exc:
        trace.update({
            "reason": "semantic_review_failed",
            "semantic_review_decision": "UNKNOWN",
            "review_error_type": type(exc).__name__,
            "review_error_preview": str(exc)[:240],
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        rule_plan["query_compiler_trace"] = trace
        return rule_plan, trace

    if not review or str(review.get("decision") or "") != "PASS":
        trace.update({
            "reason": (
                "semantic_review_rejected"
                if str(review.get("decision") or "") == "REJECT"
                else "semantic_review_unknown"
            ),
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        rule_plan["query_compiler_trace"] = trace
        return rule_plan, trace

    selected = _admit_v3_2_plan(candidate_plan, review)
    static_validation = validate_v2_physical_plan(selected)
    if not static_validation["valid"]:
        trace.update({
            "reason": f"post_review_{static_validation['reason']}",
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        })
        rule_plan["query_compiler_trace"] = trace
        return rule_plan, trace
    trace.update({
        "selected_source": "model_based_semantic_admission_dev_v3_2",
        "selected_plan_id": str(selected.get("plan_id") or ""),
        "admission_source": MODEL_BASED_SEMANTIC_ADMISSION_DEV,
        "reason": "independent_relation_coverage_review_pass",
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
    })
    selected["query_compiler_trace"] = trace
    return selected, trace
