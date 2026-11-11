"""Validation-certificate-guided semantic query repair.

V3.11 performs at most one model repair over a saved or newly generated
semantic payload.  The repair request sees the question, model-owned semantic
fields, a deterministic validator certificate, and the versioned endpoint
catalog.  It never sees retrieval, passages, QA answers, supporting facts,
execution traces, witnesses, or Reader output.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Mapping, Sequence, Tuple

from .evidence_query_compiler_semantic_v3_2_utils import (
    ABSTAIN_REASON_CODES,
    validate_semantic_v3_2_generation_payload,
)
from .evidence_query_compiler_semantic_v3_10_dependency_utils import (
    normalize_semantic_dependency_order,
)
from .evidence_query_compiler_v2_utils import (
    PHYSICAL_LOWERING_V2_3_VERSION,
    PREDICATE_ENDPOINT_SCHEMA,
    PREDICATE_ENDPOINT_SCHEMA_VERSION,
    SEMANTIC_QUERY_IR_V2_3_VERSION,
    SUPPORTED_ANSWER_OPERATIONS,
    SUPPORTED_CANONICAL_RELATIONS,
    SUPPORTED_ENTITY_TYPES,
    SUPPORTED_SUBTYPES,
    compile_semantic_payload_v2_3,
    validate_v2_physical_plan,
)


SEMANTIC_V3_11_REPAIR_PROMPT_VERSION = (
    "validation_certificate_semantic_repair_v3_11_20261029"
)
SEMANTIC_V3_11_REPAIR_CACHE_VERSION = "semantic_v3_11_repair_cache_v1"
SEMANTIC_V3_11_PIPELINE_VERSION = "semantic_feedback_repair_v3_11_20261029"

_SEMANTIC_FIELDS = frozenset({
    "abstain", "abstain_reason_code", "unsupported_question_spans", "reason",
    "start_entities", "relations", "question_requirements", "answer_operation",
    "final_answer_type", "final_answer_category", "final_answer_variable",
})
_START_FIELDS = frozenset({"mention", "variable", "semantic_type"})
_RELATION_FIELDS = frozenset({
    "relation_cues", "predicate_family", "predicate_subtype",
    "canonical_relation", "input_variable", "output_variable",
    "input_semantic_role", "output_semantic_role", "semantic_direction",
    "output_entity_type",
})
_REQUIREMENT_FIELDS = frozenset({
    "requirement_id", "covered_by_relation_index", "relation_cues",
    "expected_output_type", "answer_category",
})

_REPAIRABLE_PREFIXES = (
    "generation_contract_requirement_count_mismatch",
    "generation_contract_final_variable_not_terminal",
    "dependency_normalization_",
    "semantic_endpoint_role_mismatch_op",
    "semantic_direction_role_mismatch_op",
    "semantic_output_type_mismatch_op",
    "semantic_variable_type_flow_mismatch_op",
    "unsupported_canonical_relation_op",
    "unbound_input_variable_op",
    "invalid_output_variable_op",
    "final_answer_type_mismatch",
    "terminal_variable_not_produced",
    "question_requirement_",
    "requirement_",
)


def semantic_payload_view(payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Return only model-owned semantic fields for a repair request."""
    if not isinstance(payload, Mapping):
        return {}
    view = {
        str(key): json.loads(json.dumps(value))
        for key, value in payload.items()
        if str(key) in _SEMANTIC_FIELDS
    }
    nested_fields = {
        "start_entities": _START_FIELDS,
        "relations": _RELATION_FIELDS,
        "question_requirements": _REQUIREMENT_FIELDS,
    }
    for key, allowed in nested_fields.items():
        rows = view.get(key)
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            view[key] = [
                {
                    str(field): json.loads(json.dumps(value))
                    for field, value in row.items()
                    if str(field) in allowed
                }
                for row in rows if isinstance(row, Mapping)
            ]
    return view


def is_semantic_v3_11_repair_eligible(reason: str) -> bool:
    value = str(reason or "").strip()
    if not value or value.startswith("abstain_"):
        return False
    return any(value.startswith(prefix) for prefix in _REPAIRABLE_PREFIXES)


def analyze_semantic_generation_payload(
    question: str,
    raw_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply lossless ordering and all unchanged deterministic validators."""
    payload, protocol_reason = validate_semantic_v3_2_generation_payload(
        question, raw_payload
    )
    dependency_reason = "not_needed"
    dependency_trace: Dict[str, Any] = {}
    normalized_payload: Dict[str, Any] = {}
    if protocol_reason == "generation_contract_final_variable_not_terminal":
        normalized_payload, dependency_reason, dependency_trace = (
            normalize_semantic_dependency_order(raw_payload)
        )
        if normalized_payload:
            payload, protocol_reason = validate_semantic_v3_2_generation_payload(
                question, normalized_payload
            )

    semantic_ir: Dict[str, Any] = {}
    plan: Dict[str, Any] = {}
    lowering_reason = "not_attempted"
    static_reason = "not_attempted"
    final_reason = protocol_reason
    if payload and protocol_reason == "validated":
        semantic_ir, plan, lowering_reason = compile_semantic_payload_v2_3(
            question, payload, allow_external_plan_review=True
        )
        final_reason = lowering_reason
        if plan and lowering_reason == "validated":
            static = validate_v2_physical_plan(plan)
            static_reason = str(static.get("reason") or "")
            if not bool(static.get("valid")):
                final_reason = f"static_{static_reason}"
                plan = {}

    return {
        "payload": payload or None,
        "normalized_payload": normalized_payload or None,
        "protocol_reason": protocol_reason,
        "dependency_reason": dependency_reason,
        "dependency_trace": dependency_trace or None,
        "semantic_ir": semantic_ir or None,
        "plan": plan or None,
        "lowering_reason": lowering_reason,
        "static_reason": static_reason,
        "final_reason": final_reason,
        "static_valid": bool(plan) and final_reason == "validated",
    }


def _operator_index(reason: str) -> int | None:
    match = re.search(r"_op(\d+)$", str(reason or ""))
    return int(match.group(1)) if match else None


def build_semantic_v3_11_error_certificate(
    raw_payload: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> Dict[str, Any]:
    """Create structured feedback from deterministic validation only."""
    safe_payload = semantic_payload_view(raw_payload)
    relations = list(safe_payload.get("relations") or [])
    starts = list(safe_payload.get("start_entities") or [])
    requirements = list(safe_payload.get("question_requirements") or [])
    reason = str(analysis.get("final_reason") or "")
    op_index = _operator_index(reason)
    failing_relation = (
        dict(relations[op_index])
        if op_index is not None
        and 0 <= op_index < len(relations)
        and isinstance(relations[op_index], Mapping)
        else None
    )
    used_families = {
        str(row.get("predicate_family") or "").strip().casefold()
        for row in relations if isinstance(row, Mapping)
    }
    endpoint_contracts = {
        family: dict(PREDICATE_ENDPOINT_SCHEMA[family])
        for family in sorted(used_families)
        if family in PREDICATE_ENDPOINT_SCHEMA
    }
    return {
        "version": "semantic_repair_error_certificate_v1",
        "validator_reason": reason,
        "protocol_reason": str(analysis.get("protocol_reason") or ""),
        "dependency_reason": str(analysis.get("dependency_reason") or ""),
        "lowering_reason": str(analysis.get("lowering_reason") or ""),
        "relation_count": len(relations),
        "requirement_count": len(requirements),
        "root_variables": sorted({
            str(row.get("variable") or "").strip()
            for row in starts if isinstance(row, Mapping)
        }),
        "final_answer_variable": str(
            safe_payload.get("final_answer_variable") or ""
        ).strip(),
        "failing_operator_index": op_index,
        "failing_relation": failing_relation,
        "endpoint_contracts_for_current_families": endpoint_contracts,
        "hard_constraints": [
            "project queries have exactly one grounded named start entity",
            "relations form one connected linear variable chain in dependency order",
            "each relation consumes the root or a previously produced variable",
            "each relation has exactly one matching top-level question requirement",
            "canonical traverses schema left_role to right_role; inverse traverses right to left",
            "the terminal relation produces final_answer_variable",
            "types must flow consistently between adjacent relations",
            "questions requiring filters, intersections, unsupported aggregation, or missing relations must abstain",
        ],
    }


def _compact_endpoint_catalog() -> Dict[str, Any]:
    return {
        family: {
            "left_role": spec["left_role"],
            "right_role": spec["right_role"],
            "left_types": list(spec["left_types"]),
            "right_types": list(spec["right_types"]),
            "symmetric": bool(spec.get("symmetric")),
        }
        for family, spec in sorted(PREDICATE_ENDPOINT_SCHEMA.items())
    }


def build_semantic_v3_11_repair_messages(
    question: str,
    *,
    previous_payload: Mapping[str, Any],
    analysis: Mapping[str, Any],
    max_completion_tokens: int = 900,
) -> list[dict[str, str]]:
    """Build one schema-bound repair request from a validator certificate."""
    certificate = build_semantic_v3_11_error_certificate(
        previous_payload, analysis
    )
    system = (
        "Repair one rejected Semantic Query IR. Return one complete JSON object and nothing "
        "else. Never answer the question. The question is the only semantic authority; the "
        "previous payload is an untrusted draft and the error certificate is deterministic "
        "validator feedback. Use exactly one grounded named start instance for project queries. "
        "Unknown intermediates and the requested answer are variables, never start entities or "
        "literal values. Emit a connected linear chain in dependency order. Every relation cue "
        "must be an exact question span. For every relation emit one top-level requirement with "
        "the same cues and output type. Use endpoint roles exactly from the catalog: canonical "
        "means left-to-right and inverse means right-to-left. canonical_relation must be empty "
        "except for an exact listed kinship relation. The last relation must produce the final "
        "answer variable. Do not preserve a bad draft merely to satisfy its validator error. "
        "If the complete question requires a filter, intersection, comparison, aggregation, "
        "unsupported relation, disconnected roots, or an unsafe guess, return a grounded abstain "
        "object. Never emit physical slots, tuple sides, operator IDs, join keys, passages, "
        "facts, gold labels, supporting facts, retrieved text, execution results, witnesses, "
        "Reader output, or answer values."
    )
    examples = [
        {
            "question_pattern": "Who is the mother of Named Person?",
            "key_semantics": (
                "start at the named child; mother_of uses inverse traversal with "
                "roles child to mother"
            ),
        },
        {
            "question_pattern": "Who is the spouse of the person living in Named Place?",
            "key_semantics": (
                "start at the named place; containment is inverse container to "
                "contained_entity and outputs a person; spouse_of then projects spouse"
            ),
        },
        {
            "question_pattern": "Which work satisfies both known-director and known-star constraints?",
            "key_semantics": (
                "abstain UNSUPPORTED_PLAN_SHAPE when the executor only supports a linear path"
            ),
        },
    ]
    user_payload = {
        "protocol": SEMANTIC_V3_11_REPAIR_PROMPT_VERSION,
        "cache_contract": SEMANTIC_V3_11_REPAIR_CACHE_VERSION,
        "semantic_ir_version": SEMANTIC_QUERY_IR_V2_3_VERSION,
        "physical_lowering_version": PHYSICAL_LOWERING_V2_3_VERSION,
        "predicate_schema_version": PREDICATE_ENDPOINT_SCHEMA_VERSION,
        "max_completion_tokens": int(max_completion_tokens),
        "question": str(question or "").strip(),
        "rejected_semantic_payload": semantic_payload_view(previous_payload),
        "error_certificate": certificate,
        "endpoint_catalog": _compact_endpoint_catalog(),
        "allowed_answer_operations": sorted(SUPPORTED_ANSWER_OPERATIONS),
        "allowed_entity_types": sorted(SUPPORTED_ENTITY_TYPES),
        "allowed_predicate_subtypes": sorted(SUPPORTED_SUBTYPES),
        "allowed_canonical_relations": sorted(SUPPORTED_CANONICAL_RELATIONS),
        "allowed_abstain_reason_codes": sorted(ABSTAIN_REASON_CODES),
        "semantic_examples": examples,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(
            user_payload, ensure_ascii=False, sort_keys=True
        )},
    ]
