"""Minimal semantic skeleton generation with deterministic IR construction.

The model emits only a grounded start mention and an ordered linear sequence
of schema edges.  Variables, endpoint roles, requirements, canonical
relations, and the physical plan are constructed deterministically.  This
keeps model-owned semantics separate from executable query-plan fields.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Mapping, Sequence, Tuple

from .evidence_query_compiler_semantic_v3_11_feedback_utils import (
    analyze_semantic_generation_payload,
)
from .evidence_query_compiler_semantic_v3_2_utils import ABSTAIN_REASON_CODES
from .evidence_query_compiler_v2_utils import (
    PREDICATE_ENDPOINT_SCHEMA,
    PREDICATE_ENDPOINT_SCHEMA_VERSION,
    SUPPORTED_CANONICAL_RELATIONS,
    SUPPORTED_ENTITY_TYPES,
    SUPPORTED_SUBTYPES,
    _question_terminal_category,
)


SEMANTIC_V3_12_PROMPT_VERSION = "semantic_skeleton_generation_v3_12_20261029"
SEMANTIC_V3_12_CACHE_VERSION = "semantic_v3_12_skeleton_cache_v1"
SEMANTIC_V3_12_PIPELINE_VERSION = "deterministic_skeleton_lowering_v3_12_20261029"

_TRAVERSALS = {"LEFT_TO_RIGHT", "RIGHT_TO_LEFT"}

_EXPLICIT_KINSHIP_FAMILIES = (
    (re.compile(r"\b(?:maternal|mother)\b", re.IGNORECASE), "mother_of"),
    (re.compile(r"\b(?:paternal|father)\b", re.IGNORECASE), "father_of"),
    (re.compile(r"\bparent\b", re.IGNORECASE), "parent_of"),
    (re.compile(r"\bdaughter\b", re.IGNORECASE), "daughter_of"),
    (re.compile(r"\bson\b", re.IGNORECASE), "son_of"),
    (re.compile(r"\bchild\b", re.IGNORECASE), "child_of"),
    (re.compile(r"\b(?:spouse|wife|husband|married|partner|consort)\b", re.IGNORECASE), "spouse_of"),
)

_NAMED_WORK_PREFIX_RE = re.compile(
    r"\b(?:film|movie|song|album|book|novel|series|painting|play)\s+$",
    re.IGNORECASE,
)


def _compact_catalog() -> Dict[str, Any]:
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


def build_semantic_v3_12_skeleton_messages(
    question: str,
    *,
    max_completion_tokens: int = 650,
) -> list[dict[str, str]]:
    """Build the deliberately small model-owned semantic interface."""
    system = (
        "Translate one question into a linear relation skeleton; never answer it. "
        "Return one JSON object only. Use decision PLAN only when the complete answer "
        "can be reached from exactly one named instance by following one connected "
        "linear sequence of catalog relations. The start mention and every relation "
        "cue must be copied exactly from the question. Unknown intermediates and the "
        "answer are not literal values and must not appear as starts. For each step, "
        "choose one catalog family and LEFT_TO_RIGHT or RIGHT_TO_LEFT according to the "
        "catalog endpoint meaning. Do not emit variables, endpoint roles, requirements, "
        "operator IDs, tuple sides, slots, join keys, facts, passages, or answer values. "
        "Use ABSTAIN for filters/intersections, comparisons, aggregation, disconnected "
        "roots, descriptive starts that still require resolution, unsupported relations, "
        "or uncertain schema alignment. Do not omit a question relation merely to make "
        "a linear plan."
    )
    examples = [
        {
            "question": "Where did the director of Film Z study?",
            "output": {
                "decision": "PLAN",
                "start_entity": {"mention": "Film Z", "semantic_type": "work"},
                "steps": [
                    {
                        "relation_cue": "director",
                        "predicate_family": "direction",
                        "predicate_subtype": "",
                        "traversal": "LEFT_TO_RIGHT",
                        "output_entity_type": "person",
                    },
                    {
                        "relation_cue": "study",
                        "predicate_family": "education",
                        "predicate_subtype": "",
                        "traversal": "LEFT_TO_RIGHT",
                        "output_entity_type": "organization",
                    },
                ],
                "final_answer_type": "organization",
            },
        },
        {
            "question": "Who is Alice Example's mother?",
            "output": {
                "decision": "PLAN",
                "start_entity": {"mention": "Alice Example", "semantic_type": "person"},
                "steps": [{
                    "relation_cue": "mother",
                    "predicate_family": "mother_of",
                    "predicate_subtype": "",
                    "traversal": "RIGHT_TO_LEFT",
                    "output_entity_type": "person",
                }],
                "final_answer_type": "person",
            },
        },
        {
            "question": "Which of Film A and Film B was released first?",
            "output": {
                "decision": "ABSTAIN",
                "abstain_reason_code": "UNSUPPORTED_COMPARISON",
                "unsupported_question_span": "released first",
                "reason": "The question compares two roots rather than following one linear path.",
            },
        },
    ]
    payload = {
        "protocol": SEMANTIC_V3_12_PROMPT_VERSION,
        "cache_contract": SEMANTIC_V3_12_CACHE_VERSION,
        "schema_version": PREDICATE_ENDPOINT_SCHEMA_VERSION,
        "max_completion_tokens": int(max_completion_tokens),
        "plan_schema": {
            "decision": "PLAN",
            "start_entity": {"mention": "exact question span", "semantic_type": "allowed type"},
            "steps": [{
                "relation_cue": "exact question span",
                "predicate_family": "catalog family",
                "predicate_subtype": "optional allowed subtype or empty",
                "traversal": "LEFT_TO_RIGHT or RIGHT_TO_LEFT",
                "output_entity_type": "allowed type",
            }],
            "final_answer_type": "same as final step output type",
        },
        "abstain_schema": {
            "decision": "ABSTAIN",
            "abstain_reason_code": "allowed fixed code",
            "unsupported_question_span": "exact question span",
            "reason": "short reason",
        },
        "allowed_abstain_reason_codes": sorted(ABSTAIN_REASON_CODES),
        "allowed_entity_types": sorted(SUPPORTED_ENTITY_TYPES),
        "allowed_predicate_subtypes": sorted(SUPPORTED_SUBTYPES),
        "endpoint_catalog": _compact_catalog(),
        "worked_examples": examples,
        "question": str(question or "").strip(),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def _exact_question_span(question: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.casefold() not in str(question or "").casefold():
        return ""
    return text


def _type_compatible(actual: str, allowed: Sequence[str]) -> bool:
    actual = str(actual or "").strip().casefold()
    values = {str(item).strip().casefold() for item in allowed}



    return bool(actual and (actual in values or actual == "entity"))


def _safe_subtype(family: str, subtype: str, output_type: str) -> Tuple[str, str]:
    subtype = str(subtype or "").strip().casefold()
    if subtype and subtype not in SUPPORTED_SUBTYPES:
        return "", "unsupported_predicate_subtype"
    inferred = {
        ("birth", "date"): "birth_date",
        ("birth", "year"): "birth_date",
        ("birth", "place"): "birth_place",
        ("death", "date"): "death_date",
        ("death", "year"): "death_date",
        ("death", "place"): "death_place",
        ("employment", "organization"): "employment_organization",
        ("employment", "place"): "employment_location",
        ("employment", "occupation"): "occupation_role",
    }.get((family, output_type), "")
    if subtype and inferred and subtype != inferred:
        return "", "predicate_subtype_output_type_mismatch"
    return subtype or inferred, "validated"


def _schema_link_family(cue: str, proposed_family: str) -> Tuple[str, str]:
    """Resolve explicit role-specific kinship cues deterministically."""
    proposed = str(proposed_family or "").strip().casefold()
    for pattern, family in _EXPLICIT_KINSHIP_FAMILIES:
        if pattern.search(str(cue or "")):
            return family, "unchanged" if family == proposed else f"explicit_kinship:{proposed}->{family}"
    return proposed, "unchanged"


def deterministic_v3_12_grounding_certificate(
    question: str,
    plan: Mapping[str, Any],
) -> Dict[str, Any] | None:
    """Certify only explicit work-title constructions in the question text.

    This is intentionally narrow.  It does not attempt general NER and cannot
    turn a relational description such as ``the director of X`` into a start.
    """
    semantic_ir = plan.get("semantic_query_ir") if isinstance(plan, Mapping) else None
    starts = list((semantic_ir or {}).get("start_entities") or [])
    if len(starts) != 1 or not isinstance(starts[0], Mapping):
        return None
    mention = str(starts[0].get("mention") or "").strip()
    semantic_type = str(starts[0].get("semantic_type") or "").strip().casefold()
    if semantic_type != "work" or not mention:
        return None
    folded_question = str(question or "").casefold()
    folded_mention = mention.casefold()
    start = folded_question.find(folded_mention)
    if start < 0:
        return None
    prefix = str(question or "")[:start]
    if not _NAMED_WORK_PREFIX_RE.search(prefix):
        return None
    return {
        "start_entity_grounding": "NAMED_INSTANCE",
        "decision": "PASS",
        "basis": "explicit_work_type_prefix_in_question",
        "mention": mention,
    }


def lower_semantic_v3_12_skeleton(
    question: str,
    raw_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a skeleton, construct full semantic IR, and lower it strictly."""
    payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
    decision = str(payload.get("decision") or "").strip().upper()
    if decision == "ABSTAIN":
        code = str(
            payload.get("abstain_reason_code")
            or payload.get("abstain_reason")
            or ""
        ).strip().upper()
        span = _exact_question_span(
            question,
            payload.get("unsupported_question_span")
            or next(iter(payload.get("unsupported_question_spans") or []), ""),
        )
        reason = str(payload.get("reason") or "").strip()
        valid = bool(code in ABSTAIN_REASON_CODES and span and reason)
        return {
            "decision": "ABSTAIN",
            "reason": f"abstain_{code.casefold()}" if valid else "invalid_abstain_contract",
            "protocol_valid": valid,
            "static_valid": False,
            "normalized_abstain": {
                "abstain": True,
                "abstain_reason_code": code,
                "unsupported_question_spans": [span] if span else [],
                "reason": reason,
            } if valid else None,
            "semantic_payload": None,
            "analysis": None,
            "plan": None,
        }
    if decision != "PLAN":
        return {"decision": decision, "reason": "invalid_decision", "protocol_valid": False, "static_valid": False, "semantic_payload": None, "analysis": None, "plan": None}

    start = payload.get("start_entity")
    steps = payload.get("steps")
    if not isinstance(start, Mapping):
        reason = "missing_start_entity"
    elif not isinstance(steps, list) or not 1 <= len(steps) <= 4:
        reason = "invalid_step_count"
    else:
        reason = "validated"
    if reason != "validated":
        return {"decision": decision, "reason": reason, "protocol_valid": False, "static_valid": False, "semantic_payload": None, "analysis": None, "plan": None}

    mention = _exact_question_span(question, start.get("mention"))
    start_type = str(start.get("semantic_type") or "").strip().casefold()
    if not mention:
        reason = "ungrounded_start_mention"
    elif start_type not in SUPPORTED_ENTITY_TYPES:
        reason = "unsupported_start_type"
    if reason != "validated":
        return {"decision": decision, "reason": reason, "protocol_valid": False, "static_valid": False, "semantic_payload": None, "analysis": None, "plan": None}

    relations = []
    requirements = []
    schema_link_adaptations = []
    current_type = start_type


    input_variable = "$start_entity"
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            reason = f"invalid_step_{index}"
            break
        cue = _exact_question_span(question, step.get("relation_cue"))
        proposed_family = str(step.get("predicate_family") or "").strip().casefold()
        family, family_adaptation = _schema_link_family(cue, proposed_family)
        if family_adaptation != "unchanged":
            schema_link_adaptations.append({"step": index, "adaptation": family_adaptation})
        traversal = str(step.get("traversal") or "").strip().upper()
        output_type = str(step.get("output_entity_type") or "").strip().casefold()
        spec = PREDICATE_ENDPOINT_SCHEMA.get(family)
        if not cue:
            reason = f"ungrounded_relation_cue_{index}"
            break
        if not spec:
            reason = f"unsupported_predicate_family_{index}"
            break
        if traversal not in _TRAVERSALS:
            reason = f"invalid_traversal_{index}"
            break
        if output_type not in SUPPORTED_ENTITY_TYPES:
            reason = f"unsupported_output_type_{index}"
            break
        left_to_right = traversal == "LEFT_TO_RIGHT"
        input_role = str(spec["left_role"] if left_to_right else spec["right_role"])
        output_role = str(spec["right_role"] if left_to_right else spec["left_role"])
        input_types = spec["left_types"] if left_to_right else spec["right_types"]
        output_types = spec["right_types"] if left_to_right else spec["left_types"]
        if not _type_compatible(current_type, input_types):
            reason = f"input_type_endpoint_mismatch_{index}"
            break
        if not _type_compatible(output_type, output_types):
            reason = f"output_type_endpoint_mismatch_{index}"
            break
        subtype, subtype_reason = _safe_subtype(
            family, str(step.get("predicate_subtype") or ""), output_type
        )
        if subtype_reason != "validated":
            reason = f"{subtype_reason}_{index}"
            break
        output_variable = "$final_value" if index == len(steps) - 1 else f"$intermediate{index + 1}"
        relations.append({
            "relation_cues": [cue],
            "predicate_family": family,
            "predicate_subtype": subtype,
            "canonical_relation": family if family in SUPPORTED_CANONICAL_RELATIONS else "",
            "input_variable": input_variable,
            "output_variable": output_variable,
            "input_semantic_role": input_role,
            "output_semantic_role": output_role,
            "semantic_direction": "canonical" if left_to_right else "inverse",
            "output_entity_type": output_type,
        })
        requirements.append({
            "requirement_id": f"q{index}",
            "covered_by_relation_index": index,
            "relation_cues": [cue],
            "expected_output_type": output_type,
            "answer_category": "",
        })
        current_type = output_type
        input_variable = output_variable
    if reason != "validated":
        return {"decision": decision, "reason": reason, "protocol_valid": False, "static_valid": False, "semantic_payload": None, "analysis": None, "plan": None}

    final_type = str(payload.get("final_answer_type") or "").strip().casefold()
    if final_type != current_type:
        return {"decision": decision, "reason": "final_answer_type_mismatch", "protocol_valid": False, "static_valid": False, "semantic_payload": None, "analysis": None, "plan": None}
    final_category = _question_terminal_category(question)
    semantic_payload = {
        "abstain": False,
        "start_entities": [{"mention": mention, "variable": "$start_entity", "semantic_type": start_type}],
        "relations": relations,
        "question_requirements": requirements,
        "answer_operation": "project",
        "final_answer_type": final_type,
        "final_answer_category": final_category,
        "final_answer_variable": "$final_value",
    }
    if requirements:
        requirements[-1]["answer_category"] = final_category
    analysis = analyze_semantic_generation_payload(question, semantic_payload)
    return {
        "decision": decision,
        "reason": str(analysis.get("final_reason") or ""),
        "protocol_valid": True,
        "static_valid": bool(analysis.get("static_valid")),
        "semantic_payload": semantic_payload,
        "schema_link_adaptations": schema_link_adaptations,
        "analysis": analysis,
        "plan": analysis.get("plan"),
    }
