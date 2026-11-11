"""Semantic query IR and deterministic physical lowering for compiler V2.

The semantic compiler is deliberately separated from the executable plan.  An
LLM may name grounded entities, relation cues, semantic endpoint roles, and
variable dependencies.  It may not choose tuple sides, physical slots, join
keys, or operator identifiers.  Those fields are derived here from a versioned
endpoint catalog and are rejected when the catalog cannot determine a safe
contract.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .relation_fact_join_utils import (
    PLAN_COMPLETE,
    PLAN_COVERAGE_CONTRACT_VERSION,
    SCHEMA_PREDICATE_FAMILY_TERMS,
    TERMINAL_ENTITY_CATEGORY_TERMS,
    assess_question_plan_coverage,
    attach_logical_query_plan_identity,
    compile_question_obligations,
    validate_logical_query_plan_static,
)


SEMANTIC_QUERY_IR_VERSION = "semantic_query_ir_v2"
PHYSICAL_LOWERING_VERSION = "deterministic_physical_lowering_v2"
PREDICATE_ENDPOINT_SCHEMA_VERSION = "predicate_endpoint_schema_v2"
HYBRID_V2_PROMPT_VERSION = "hybrid_query_compiler_prompt_v2"
HYBRID_V2_COMPILER_VERSION = "hybrid_query_compiler_v2"
HYBRID_V2_CACHE_KEY_VERSION = "hybrid_v2_cache_contract_v1"

SEMANTIC_QUERY_IR_V2_1_VERSION = "semantic_query_ir_v2_1"
PHYSICAL_LOWERING_V2_1_VERSION = "deterministic_physical_lowering_v2_1"
HYBRID_V2_1_PROMPT_VERSION = "hybrid_query_compiler_prompt_v2_1"
HYBRID_V2_1_COMPILER_VERSION = "hybrid_query_compiler_v2_1"
HYBRID_V2_1_CACHE_KEY_VERSION = "hybrid_v2_1_cache_contract_v1"

SEMANTIC_QUERY_IR_V2_2_VERSION = "semantic_query_ir_v2_2"
PHYSICAL_LOWERING_V2_2_VERSION = "deterministic_physical_lowering_v2_2"
HYBRID_V2_2_PROMPT_VERSION = "hybrid_query_compiler_prompt_v2_2"
HYBRID_V2_2_COMPILER_VERSION = "hybrid_query_compiler_v2_2"
HYBRID_V2_2_CACHE_KEY_VERSION = "hybrid_v2_2_cache_contract_v1"

SEMANTIC_QUERY_IR_V2_3_VERSION = "semantic_query_ir_v2_3"
PHYSICAL_LOWERING_V2_3_VERSION = "deterministic_physical_lowering_v2_3"

HYBRID_V2_4_REPAIR_PROMPT_VERSION = "hybrid_query_compiler_repair_prompt_v2_4"
HYBRID_V2_4_REPAIR_CACHE_KEY_VERSION = "hybrid_v2_4_repair_cache_contract_v1"
HYBRID_V2_4_1_REPAIR_PROMPT_VERSION = "hybrid_query_compiler_repair_prompt_v2_4_1"
HYBRID_V2_4_1_REPAIR_CACHE_KEY_VERSION = "hybrid_v2_4_1_repair_cache_contract_v1"

QUESTION_SEMANTIC_VALIDATION_VERSION = "question_semantic_validation_v1"
QUESTION_SEMANTIC_NEEDS_REVIEW = "NEEDS_REVIEW"
QUESTION_SEMANTIC_CONTRACT_VALIDATED = "INDEPENDENT_CONTRACT_VALIDATED"





V2_4_REPAIRABLE_REASONS = frozenset({
    "ambiguous_predicate_subtype_op1",
    "missing_output_entity_type_op0",
    "requirement_relation_cue_mismatch_1",
    "semantic_direction_role_mismatch_op0",
    "semantic_endpoint_role_mismatch_op0",
    "unbound_input_variable_op0",
    "missing_or_truncated_saved_payload",
})

SUPPORTED_ENTITY_TYPES = {
    "entity", "person", "place", "date", "year", "number", "work",
    "organization", "nationality", "occupation", "event", "water_body",
}
SUPPORTED_ANSWER_OPERATIONS = {"project", "compare_eq", "argmin_root", "argmax_root"}
SUPPORTED_CANONICAL_RELATIONS = {
    "mother_of", "father_of", "parent_of", "daughter_of", "son_of",
    "child_of", "wife_of", "husband_of", "spouse_of",
}
SUPPORTED_SUBTYPES = {
    "birth_name", "birth_date", "birth_place",
    "death_cause", "death_date", "death_place",
    "occupation_role", "employment_organization", "employment_location",
    "employment_generic", "distribution_actual_release",
    "distribution_intended_release", "distribution_premiere",
}

_VARIABLE_RE = re.compile(r"^\$[A-Za-z][A-Za-z0-9_]*$")


def _endpoint(
    left_role: str,
    right_role: str,
    *,
    left_types: Iterable[str] = ("entity",),
    right_types: Iterable[str] = ("entity",),
    layout: str = "either",
    symmetric: bool = False,
) -> Dict[str, Any]:
    return {
        "left_role": left_role,
        "right_role": right_role,
        "left_types": tuple(left_types),
        "right_types": tuple(right_types),
        "physical_layout": layout,
        "symmetric": bool(symmetric),
    }






PREDICATE_ENDPOINT_SCHEMA: Dict[str, Dict[str, Any]] = {
    "birth": _endpoint("person", "birth_value", left_types=("person", "entity"), right_types=("place", "date", "year", "entity"), layout="subject_object"),
    "death": _endpoint("person", "death_value", left_types=("person", "entity"), right_types=("place", "date", "year", "entity"), layout="subject_object"),
    "capital": _endpoint("region", "capital", left_types=("place", "entity"), right_types=("place", "entity")),
    "containment": _endpoint("contained_entity", "container", left_types=("entity", "place", "organization", "work"), right_types=("place", "organization", "entity")),
    "geopolitical_location": _endpoint("located_entity", "location", left_types=("entity", "place", "organization"), right_types=("place", "entity")),
    "founding": _endpoint("organization", "founding_value", left_types=("organization", "entity"), right_types=("person", "date", "year", "entity")),
    "leadership": _endpoint("organization", "leader", left_types=("organization", "entity"), right_types=("person", "entity")),
    "authorship": _endpoint("work", "author", left_types=("work", "entity"), right_types=("person", "entity")),
    "direction": _endpoint("work", "director", left_types=("work", "entity"), right_types=("person", "entity")),
    "composition": _endpoint("work", "composer", left_types=("work", "entity"), right_types=("person", "entity")),
    "performance": _endpoint("work", "performer", left_types=("work", "entity"), right_types=("person", "entity")),
    "production": _endpoint("work", "producer", left_types=("work", "entity"), right_types=("person", "organization", "entity")),
    "visual_art": _endpoint("work", "artist", left_types=("work", "entity"), right_types=("person", "entity")),
    "design": _endpoint("work", "designer", left_types=("work", "entity"), right_types=("person", "organization", "entity")),
    "creation": _endpoint("work", "creator", left_types=("work", "entity"), right_types=("person", "organization", "entity")),
    "distribution": _endpoint("work", "distribution_value", left_types=("work", "entity"), right_types=("date", "year", "place", "organization", "entity")),
    "education": _endpoint("person", "education_institution", left_types=("person", "entity"), right_types=("organization", "place", "entity")),
    "employment": _endpoint("person", "employment_value", left_types=("person", "entity"), right_types=("organization", "place", "occupation", "entity")),
    "nationality": _endpoint("person", "nationality", left_types=("person", "entity"), right_types=("nationality", "place", "entity")),
    "membership": _endpoint("member", "organization", left_types=("person", "organization", "entity"), right_types=("organization", "entity")),
    "draft_selection": _endpoint("person", "draft_destination", left_types=("person", "entity"), right_types=("organization", "place", "entity")),
    "portrayal": _endpoint("work", "performer", left_types=("work", "entity"), right_types=("person", "entity")),
    "border": _endpoint("place", "place", left_types=("place", "entity"), right_types=("place", "entity"), symmetric=True),
    "hydrology_outflow": _endpoint("water_body", "outflow", left_types=("water_body", "place", "entity"), right_types=("water_body", "place", "entity")),
    "start_time": _endpoint("event", "start_time", left_types=("event", "entity"), right_types=("date", "year", "entity")),
    "end_time": _endpoint("event", "end_time", left_types=("event", "entity"), right_types=("date", "year", "entity")),
    "identity": _endpoint("entity", "alias", left_types=("entity",), right_types=("entity",), symmetric=True),
    "mother_of": _endpoint("mother", "child", left_types=("person",), right_types=("person",), layout="subject_object"),
    "father_of": _endpoint("father", "child", left_types=("person",), right_types=("person",), layout="subject_object"),
    "parent_of": _endpoint("parent", "child", left_types=("person",), right_types=("person",), layout="subject_object"),
    "daughter_of": _endpoint("daughter", "parent", left_types=("person",), right_types=("person",), layout="subject_object"),
    "son_of": _endpoint("son", "parent", left_types=("person",), right_types=("person",), layout="subject_object"),
    "child_of": _endpoint("child", "parent", left_types=("person",), right_types=("person",), layout="subject_object"),
    "spouse_of": _endpoint("spouse", "spouse", left_types=("person",), right_types=("person",), symmetric=True),
}

_ROLE_ALIASES = {
    "birth_value": {"birth_value", "birth_date", "birth_place", "birth_name"},
    "death_value": {"death_value", "death_date", "death_place", "death_cause"},
    "founding_value": {"founding_value", "founder", "founding_date"},
    "distribution_value": {"distribution_value", "release_date", "premiere_date", "publisher", "distributor"},
    "employment_value": {"employment_value", "employer", "organization", "occupation", "employment_location"},
    "contained_entity": {"contained_entity", "entity", "place", "organization", "work"},
    "located_entity": {"located_entity", "entity", "place", "organization"},
    "region": {"region", "place", "country", "state", "province"},
    "capital": {"capital", "city", "place"},
    "location": {"location", "place", "country", "state", "province", "city", "county"},
    "container": {"container", "location", "place", "country", "state", "province", "city", "county", "organization"},
    "work": {"work", "film", "movie", "song", "album", "book", "series", "painting", "publication"},
    "organization": {"organization", "company", "institution", "team", "station"},
    "education_institution": {"education_institution", "school", "university", "institution", "place"},
    "draft_destination": {"draft_destination", "team", "organization", "place"},
    "outflow": {"outflow", "water_body", "sea", "river", "lake", "place"},
    "start_time": {"start_time", "date", "year", "month"},
    "end_time": {"end_time", "date", "year", "month"},
}


def semantic_compiler_capability_manifest() -> Dict[str, Any]:
    return {
        "semantic_ir_version": SEMANTIC_QUERY_IR_VERSION,
        "physical_lowering_version": PHYSICAL_LOWERING_VERSION,
        "predicate_endpoint_schema_version": PREDICATE_ENDPOINT_SCHEMA_VERSION,
        "cache_key_contract_version": HYBRID_V2_CACHE_KEY_VERSION,
        "answer_operations": sorted(SUPPORTED_ANSWER_OPERATIONS),
        "predicate_families": sorted(PREDICATE_ENDPOINT_SCHEMA),
        "canonical_relations": sorted(SUPPORTED_CANONICAL_RELATIONS),
        "predicate_subtypes": sorted(SUPPORTED_SUBTYPES),
        "entity_types": sorted(SUPPORTED_ENTITY_TYPES),
        "endpoint_contracts": PREDICATE_ENDPOINT_SCHEMA,
        "max_relation_hops": 4,
        "physical_fields_forbidden_in_llm_output": [
            "operator_id", "operator_type", "input_slot", "output_slot",
            "join_key", "join_key_contract", "input_role", "output_role",
            "physical_input_side", "physical_output_side",
        ],
    }


def build_hybrid_v2_messages(
    question: str,
    *,
    max_completion_tokens: int = 900,
) -> list[dict[str, str]]:
    compact_relations = ";".join(
        f"{family}({spec['left_role']}->{spec['right_role']})"
        for family, spec in sorted(PREDICATE_ENDPOINT_SCHEMA.items())
    )
    system = (
        "Compile the question into SEMANTIC query IR only. Never answer the "
        "question. Return JSON only. Use start_entities (literal mentions copied "
        "from the question), relations (ordered variable dependencies), "
        "answer_operation, final_answer_type, and final_answer_variable. Each "
        "relation must contain relation_cues copied literally from the question, "
        "exactly one predicate_family from the catalog, optional predicate_subtype "
        "or canonical_relation when justified, input_variable, output_variable, "
        "input_semantic_role, output_semantic_role, semantic_direction "
        "(canonical or inverse), and output_entity_type. Intermediate entities "
        "must be variables, never values. Do not emit tuple subject/object/either, "
        "physical slots, operator IDs, join keys, SQL, answers, or unknown entity "
        "values. For project plans use one grounded start variable $root and end "
        "at $answer. For a supported two-root comparison use $root_each and "
        "$comparison_value. If the catalog cannot express every required relation "
        "without guessing, return {\"abstain\":true,\"reason\":\"...\"}."
    )
    user = "\n".join([
        f"Prompt version: {HYBRID_V2_PROMPT_VERSION}",
        f"Semantic IR version: {SEMANTIC_QUERY_IR_VERSION}",
        f"Predicate schema version: {PREDICATE_ENDPOINT_SCHEMA_VERSION}",
        f"Physical lowering version: {PHYSICAL_LOWERING_VERSION}",
        f"Cache key contract version: {HYBRID_V2_CACHE_KEY_VERSION}",
        (
            "Compiler request parameters: "
            f"max_completion_tokens={int(max_completion_tokens)};"
            "temperature=0;response_format=json_object"
        ),
        "Answer operations: " + ",".join(sorted(SUPPORTED_ANSWER_OPERATIONS)),
        "Entity types: " + ",".join(sorted(SUPPORTED_ENTITY_TYPES)),
        "Predicate subtypes: " + ",".join(sorted(SUPPORTED_SUBTYPES)),
        "Semantic relation catalog (canonical left role -> right role):",
        compact_relations,
        "Question:",
        str(question or "").strip(),
    ])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_hybrid_v2_1_messages(
    question: str,
    *,
    max_completion_tokens: int = 900,
) -> list[dict[str, str]]:
    """Build the V2.1 semantic-only compiler prompt.

    V2.1 deliberately removes physical slot names from the model contract. The
    examples are compact so the frozen 42-request smoke remains under its
    registered 50k-token budget.
    """
    compact_relations = ";".join(
        f"{family}({spec['left_role']}->{spec['right_role']})"
        for family, spec in sorted(PREDICATE_ENDPOINT_SCHEMA.items())
    )
    system = (
        "Return semantic JSON, never an answer. Starts are literal named entities copied from "
        "the question, not unknown answers or roles. Use semantic variables; never $root, "
        "$root_each, $v1, $comparison_value, physical slots, operators, or join fields. Cues "
        "are literal question spans. Relations form one connected chain using catalog endpoint "
        "roles. Abstain on unsupported or ungrounded queries.\n"
        "Project example: {\"start_entities\":[{\"mention\":\"Film Z\",\"variable\":\"$film\","
        "\"semantic_type\":\"work\"}],\"relations\":[{\"relation_cues\":[\"director\"],"
        "\"predicate_family\":\"direction\",\"input_variable\":\"$film\","
        "\"output_variable\":\"$director\",\"input_semantic_role\":\"work\","
        "\"output_semantic_role\":\"director\",\"semantic_direction\":\"canonical\","
        "\"output_entity_type\":\"person\"},{\"relation_cues\":[\"born\"],"
        "\"predicate_family\":\"birth\",\"predicate_subtype\":\"birth_place\","
        "\"input_variable\":\"$director\",\"output_variable\":\"$birth_place\","
        "\"input_semantic_role\":\"person\",\"output_semantic_role\":\"birth_place\","
        "\"semantic_direction\":\"canonical\",\"output_entity_type\":\"place\"}],"
        "\"answer_operation\":\"project\",\"final_answer_type\":\"place\","
        "\"final_answer_variable\":\"$birth_place\"}.\n"
        "Comparison example: {\"start_entities\":[{\"mention\":\"Alice\","
        "\"variable\":\"$candidate\",\"semantic_type\":\"person\"},{\"mention\":\"Bob\","
        "\"variable\":\"$candidate\",\"semantic_type\":\"person\"}],\"relations\":[{"
        "\"relation_cues\":[\"born\"],\"predicate_family\":\"birth\","
        "\"predicate_subtype\":\"birth_date\",\"input_variable\":\"$candidate\","
        "\"output_variable\":\"$birth_date\",\"input_semantic_role\":\"person\","
        "\"output_semantic_role\":\"birth_date\",\"semantic_direction\":\"canonical\","
        "\"output_entity_type\":\"date\"}],\"answer_operation\":\"argmin_root\","
        "\"final_answer_type\":\"person\",\"final_answer_variable\":\"$earlier_person\"}.\n"
        "Abstain example: {\"abstain\":true,\"reason\":\"unsupported_intersection\"}."
    )
    user = "\n".join([
        f"Protocol={HYBRID_V2_1_PROMPT_VERSION};Schema={PREDICATE_ENDPOINT_SCHEMA_VERSION}",
        "Subtypes=" + ",".join(sorted(SUPPORTED_SUBTYPES)),
        "Catalog=" + compact_relations,
        "Q=" + str(question or "").strip(),
    ])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_hybrid_v2_2_messages(
    question: str,
    *,
    max_completion_tokens: int = 900,
) -> list[dict[str, str]]:
    """Build the V2.2 semantic-requirement compiler prompt.

    V2.2 keeps model output semantic. In addition to the relation chain, the
    model must expose a separately checkable requirement for every relation and
    the fine-grained terminal category requested by the question.
    """
    compact_relations = ";".join(
        f"{family}({spec['left_role']}->{spec['right_role']})"
        for family, spec in sorted(PREDICATE_ENDPOINT_SCHEMA.items())
    )
    system = (
        "Return semantic JSON only, never an answer. Starts and cues must be question spans. "
        "Use semantic variables, never physical slots or join fields. Emit start_entities, "
        "relations, question_requirements, answer_operation, final_answer_type, "
        "final_answer_category, and final_answer_variable. Relations are one connected chain. "
        "Each requirement has requirement_id, covered_by_relation_index, "
        "relation_cues, expected_output_type, and answer_category; there must be exactly one "
        "requirement per relation. Only the terminal requirement may carry answer_category. "
        "Unknown intermediates are variables. Use unsupported_intersection only when one answer "
        "must satisfy independent roots, never for a linear or kinship chain. Abstain if the "
        "catalog cannot express every required relation safely.\n"
        "Example for 'In what state is Painting X displayed?': {\"start_entities\":[{"
        "\"mention\":\"Painting X\",\"variable\":\"$painting\",\"semantic_type\":\"work\"}],"
        "\"relations\":[{\"relation_cues\":[\"displayed\"],\"predicate_family\":\"containment\","
        "\"input_variable\":\"$painting\",\"output_variable\":\"$venue\","
        "\"input_semantic_role\":\"work\",\"output_semantic_role\":\"container\","
        "\"semantic_direction\":\"canonical\",\"output_entity_type\":\"organization\"},{"
        "\"relation_cues\":[\"state\"],\"predicate_family\":\"geopolitical_location\","
        "\"input_variable\":\"$venue\",\"output_variable\":\"$state\","
        "\"input_semantic_role\":\"located_entity\",\"output_semantic_role\":\"location\","
        "\"semantic_direction\":\"canonical\",\"output_entity_type\":\"place\"}],"
        "\"question_requirements\":[{\"requirement_id\":\"q0\","
        "\"covered_by_relation_index\":0,\"relation_cues\":[\"displayed\"],"
        "\"expected_output_type\":\"organization\",\"answer_category\":\"\"},{"
        "\"requirement_id\":\"q1\",\"covered_by_relation_index\":1,"
        "\"relation_cues\":[\"state\"],\"expected_output_type\":\"place\","
        "\"answer_category\":\"state\"}],\"answer_operation\":\"project\","
        "\"final_answer_type\":\"place\",\"final_answer_category\":\"state\","
        "\"final_answer_variable\":\"$state\"}."
    )
    user = "\n".join([
        f"Protocol={HYBRID_V2_2_PROMPT_VERSION};Schema={PREDICATE_ENDPOINT_SCHEMA_VERSION}",
        "Catalog=" + compact_relations,
        "Q=" + str(question or "").strip(),
    ])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def is_hybrid_v2_4_repair_eligible(reason: str) -> bool:
    """Return whether one failed semantic payload may receive one repair call."""
    return str(reason or "").strip() in V2_4_REPAIRABLE_REASONS


def _semantic_repair_payload_view(payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Keep only model-owned semantic fields when constructing repair input."""
    if not isinstance(payload, Mapping):
        return {}
    top_level = {
        "start_entities", "relations", "question_requirements",
        "answer_operation", "final_answer_type", "final_answer_category",
        "final_answer_variable", "abstain", "reason",
    }
    return {
        str(key): json.loads(json.dumps(value))
        for key, value in payload.items()
        if str(key) in top_level
    }


def build_hybrid_v2_4_repair_messages(
    question: str,
    *,
    previous_payload: Mapping[str, Any] | None,
    validator_reason: str,
    max_completion_tokens: int = 500,
) -> list[dict[str, str]]:
    """Build one validation-guided semantic repair request.

    The request contains the question, the rejected semantic payload, the
    deterministic rejection reason, and the versioned endpoint catalog.  It
    never contains passages, retrieval output, supporting-fact labels, reader
    predictions, or QA answers.
    """
    compact_relations = ";".join(
        (
            f"{family}(left={spec['left_role']}:{'/'.join(spec['left_types'])},"
            f"right={spec['right_role']}:{'/'.join(spec['right_types'])})"
        )
        for family, spec in sorted(PREDICATE_ENDPOINT_SCHEMA.items())
    )
    system = (
        "Repair one rejected semantic query IR and return JSON only. The question is the "
        "only semantic authority; the previous payload is untrusted. Do not answer the "
        "question and do not invent an intermediate entity. Emit exactly one grounded root "
        "for a project query and a connected linear relation chain of one to four edges. "
        "Every start mention and relation cue must be an exact question span. Every relation "
        "must contain predicate_family, optional predicate_subtype/canonical_relation, "
        "input_variable, output_variable, input_semantic_role, output_semantic_role, "
        "semantic_direction, and output_entity_type. canonical means traversal from the "
        "catalog left endpoint to right; inverse means right to left. Emit one "
        "question_requirement per relation whose cues exactly equal that relation's cues. "
        "The final variable must be produced by the terminal relation. Never emit physical "
        "slots, tuple sides, operator identifiers, join keys, source facts, passages, gold "
        "labels, or answer values. Abstain when the question needs aggregation, comparison, "
        "intersection/filtering, an unsupported relation, or cannot be represented safely. "
        "A validator error is diagnostic feedback, not permission to weaken semantics."
    )
    user = "\n".join([
        (
            f"Protocol={HYBRID_V2_4_REPAIR_PROMPT_VERSION};"
            f"Schema={PREDICATE_ENDPOINT_SCHEMA_VERSION};"
            f"max_completion_tokens={int(max_completion_tokens)}"
        ),
        "EndpointCatalog=" + compact_relations,
        "ValidatorReason=" + str(validator_reason or "").strip(),
        "RejectedSemanticPayload=" + json.dumps(
            _semantic_repair_payload_view(previous_payload),
            ensure_ascii=True,
            sort_keys=True,
        ),
        "Question=" + str(question or "").strip(),
    ])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_hybrid_v2_4_repair_envelope(
    payload: Mapping[str, Any] | None,
) -> Tuple[Dict[str, Any], list[str], str]:
    """Losslessly restore top-level V2 fields from a nested repair response.

    The V2.4 model often emitted each requirement inside its relation and
    omitted projection fields that are functionally determined by the terminal
    relation.  This adapter moves only those fields.  It never changes a
    family, subtype, direction, role, variable dependency, type, or entity.
    """
    if not isinstance(payload, Mapping):
        return {}, [], "invalid_repair_payload"
    adapted = json.loads(json.dumps(dict(payload)))
    starts = adapted.get("start_entities")
    relations = adapted.get("relations")
    if not isinstance(starts, list) or not starts:
        return {}, [], "unsupported_repair_envelope"
    if not isinstance(relations, list) or not relations:
        return {}, [], "unsupported_repair_envelope"
    if any(not isinstance(row, dict) for row in relations):
        return {}, [], "unsupported_repair_envelope"

    adaptations: list[str] = []
    nested_requirements = []
    for index, relation_row in enumerate(relations):
        requirement = relation_row.pop("question_requirement", None)
        if requirement is not None:
            if not isinstance(requirement, Mapping):
                return {}, [], f"invalid_nested_question_requirement_{index}"
            nested_requirements.append(dict(requirement))
            adaptations.append(f"lift_nested_question_requirement_op{index}")
            if not relation_row.get("relation_cues") and requirement.get("relation_cues"):
                relation_row["relation_cues"] = list(requirement["relation_cues"])
                adaptations.append(f"copy_requirement_cues_op{index}")
            if (
                not str(relation_row.get("output_entity_type") or "").strip()
                and str(requirement.get("expected_output_type") or "").strip()
            ):
                relation_row["output_entity_type"] = str(
                    requirement["expected_output_type"]
                )
                adaptations.append(f"copy_requirement_output_type_op{index}")

    existing_requirements = adapted.get("question_requirements")
    if existing_requirements is None and len(nested_requirements) == len(relations):
        adapted["question_requirements"] = nested_requirements
        adaptations.append("lift_question_requirements_to_top_level")
    elif existing_requirements is None:
        return {}, [], "incomplete_nested_question_requirements"

    terminal = relations[-1]
    terminal_requirements = adapted.get("question_requirements") or []
    terminal_requirement = (
        terminal_requirements[-1]
        if isinstance(terminal_requirements, list)
        and len(terminal_requirements) == len(relations)
        and isinstance(terminal_requirements[-1], Mapping)
        else {}
    )
    determined_fields = {
        "answer_operation": "project",
        "final_answer_type": str(terminal.get("output_entity_type") or "").strip(),
        "final_answer_category": str(
            terminal_requirement.get("answer_category") or ""
        ).strip(),
        "final_answer_variable": str(terminal.get("output_variable") or "").strip(),
    }
    for key, value in determined_fields.items():
        if key not in adapted and value:
            adapted[key] = value
            adaptations.append(f"restore_determined_{key}")
        elif key == "final_answer_category" and key not in adapted:
            adapted[key] = ""
            adaptations.append(f"restore_determined_{key}")
    return adapted, adaptations, "adapted"


def build_hybrid_v2_4_1_repair_messages(
    question: str,
    *,
    previous_payload: Mapping[str, Any] | None,
    validator_reason: str,
    max_completion_tokens: int = 650,
) -> list[dict[str, str]]:
    """Build a strict second repair request with an explicit output envelope."""
    compact_relations = ";".join(
        (
            f"{family}(left={spec['left_role']}:{'/'.join(spec['left_types'])},"
            f"right={spec['right_role']}:{'/'.join(spec['right_types'])})"
        )
        for family, spec in sorted(PREDICATE_ENDPOINT_SCHEMA.items())
    )
    system = (
        "Return exactly one complete semantic-query JSON object or an abstain object. "
        "The question is the only semantic authority. Required top-level keys are "
        "start_entities, relations, question_requirements, answer_operation, "
        "final_answer_type, final_answer_category, and final_answer_variable. "
        "question_requirements must be a TOP-LEVEL array, never nested inside relations. "
        "answer_operation must be project. final_answer_variable must equal the terminal "
        "relation output_variable and must be a semantic variable such as $grandfather, "
        "never $answer. Every relation must contain relation_cues, predicate_family, "
        "input_variable, output_variable, input_semantic_role, output_semantic_role, "
        "semantic_direction, and output_entity_type. Omit predicate_subtype unless it is "
        "one of birth_name,birth_date,birth_place,death_cause,death_date,death_place,"
        "occupation_role,employment_organization,employment_location,employment_generic,"
        "distribution_actual_release,distribution_intended_release,distribution_premiere. "
        "canonical traverses catalog left-to-right; inverse traverses right-to-left. "
        "For example, child to mother with mother_of is inverse with roles child->mother; "
        "son to parent with son_of is canonical with roles son->parent. Use exactly one "
        "grounded start mention and a connected linear chain. Cues must be exact question "
        "spans and each top-level requirement's cues must exactly equal its relation cues. "
        "Do not output physical fields, passages, facts, gold labels, intermediate entity "
        "values, or an answer. Abstain if the catalog cannot express the question safely."
    )
    skeleton = {
        "start_entities": [{
            "mention": "QUESTION_SPAN",
            "variable": "$root_entity",
            "semantic_type": "entity",
        }],
        "relations": [{
            "relation_cues": ["QUESTION_SPAN"],
            "predicate_family": "CATALOG_FAMILY",
            "input_variable": "$root_entity",
            "output_variable": "$terminal_value",
            "input_semantic_role": "CATALOG_ENDPOINT_ROLE",
            "output_semantic_role": "CATALOG_ENDPOINT_ROLE",
            "semantic_direction": "canonical_or_inverse",
            "output_entity_type": "entity",
        }],
        "question_requirements": [{
            "requirement_id": "q0",
            "covered_by_relation_index": 0,
            "relation_cues": ["SAME_QUESTION_SPAN"],
            "expected_output_type": "entity",
            "answer_category": "",
        }],
        "answer_operation": "project",
        "final_answer_type": "entity",
        "final_answer_category": "",
        "final_answer_variable": "$terminal_value",
    }
    user = "\n".join([
        (
            f"Protocol={HYBRID_V2_4_1_REPAIR_PROMPT_VERSION};"
            f"Schema={PREDICATE_ENDPOINT_SCHEMA_VERSION};"
            f"max_completion_tokens={int(max_completion_tokens)}"
        ),
        "OutputSkeleton=" + json.dumps(skeleton, ensure_ascii=True, sort_keys=True),
        "EndpointCatalog=" + compact_relations,
        "ValidatorReason=" + str(validator_reason or "").strip(),
        "RejectedSemanticPayload=" + json.dumps(
            _semantic_repair_payload_view(previous_payload),
            ensure_ascii=True,
            sort_keys=True,
        ),
        "Question=" + str(question or "").strip(),
    ])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _grounded(question: str, value: Any) -> bool:
    needle = _normalized_text(value)
    return bool(needle and needle in _normalized_text(question))


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    return {str(item or "").strip().casefold() for item in value if str(item or "").strip()}


def _role_matches(actual: str, expected: str) -> bool:
    actual = str(actual or "").strip().casefold()
    expected = str(expected or "").strip().casefold()
    return actual == expected or actual in _ROLE_ALIASES.get(expected, set())


def _resolve_subtype(
    family: str,
    explicit: str,
    output_type: str,
    relation_cues: set[str],
) -> Tuple[str, str]:
    if explicit and explicit not in SUPPORTED_SUBTYPES:
        return "", "unsupported_predicate_subtype"
    if family == "birth":
        inferred = "birth_date" if output_type in {"date", "year"} else "birth_place" if output_type == "place" else ""
        if explicit and inferred and explicit != inferred:
            return "", "predicate_subtype_output_type_mismatch"
        return explicit or inferred, "validated" if (explicit or inferred) else "ambiguous_predicate_subtype"
    if family == "death":
        inferred = "death_date" if output_type in {"date", "year"} else "death_place" if output_type == "place" else ""
        if explicit and inferred and explicit != inferred:
            return "", "predicate_subtype_output_type_mismatch"
        return explicit or inferred, "validated" if (explicit or inferred) else "ambiguous_predicate_subtype"
    if family == "employment":
        inferred = "employment_organization" if output_type == "organization" else "employment_location" if output_type == "place" else "occupation_role" if output_type == "occupation" else ""
        if explicit and inferred and explicit != inferred:
            return "", "predicate_subtype_output_type_mismatch"
        return explicit or inferred, "validated" if (explicit or inferred) else "ambiguous_predicate_subtype"
    if family == "distribution":
        if not explicit:
            return "", "ambiguous_predicate_subtype"
        cues = " ".join(sorted(relation_cues))
        if explicit == "distribution_premiere" and not any(term in cues for term in ("premiere", "first performed", "debut")):
            return "", "predicate_subtype_not_grounded"
        if explicit == "distribution_intended_release" and not any(term in cues for term in ("intend", "scheduled", "planned")):
            return "", "predicate_subtype_not_grounded"
        return explicit, "validated"
    return explicit, "validated"


def normalize_semantic_ir_v2(
    question: str,
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """Validate model-owned semantic fields without creating physical fields."""
    if bool(payload.get("abstain")):
        return {}, str(payload.get("reason") or "llm_abstained")
    forbidden = {
        "operators", "operator_id", "operator_type", "input_slot", "output_slot",
        "join_key", "join_key_contract", "input_role", "output_role", "answer",
        "answer_value", "gold_answer", "supporting_facts",
    }
    if forbidden & set(payload):
        return {}, "physical_or_answer_field_in_semantic_ir"
    answer_operation = str(payload.get("answer_operation") or "project").strip().casefold()
    if answer_operation not in SUPPORTED_ANSWER_OPERATIONS:
        return {}, "unsupported_answer_operation"
    final_type = str(payload.get("final_answer_type") or "").strip().casefold()
    if final_type not in SUPPORTED_ENTITY_TYPES:
        return {}, "unsupported_answer_type"
    final_variable = str(payload.get("final_answer_variable") or "").strip()
    if final_variable != "$answer":
        return {}, "invalid_final_answer_variable"

    raw_starts = payload.get("start_entities")
    if not isinstance(raw_starts, Sequence) or isinstance(raw_starts, (str, bytes)) or not raw_starts:
        return {}, "missing_grounded_start_entity"
    starts = []
    for index, raw in enumerate(raw_starts):
        if not isinstance(raw, Mapping):
            return {}, f"invalid_start_entity_{index}"
        if any(str(raw.get(key) or "").strip() for key in ("answer", "value", "gold_answer")):
            return {}, f"invented_start_value_{index}"
        mention = str(raw.get("mention") or "").strip()
        variable = str(raw.get("variable") or "").strip()
        entity_type = str(raw.get("semantic_type") or "entity").strip().casefold()
        if not _grounded(question, mention):
            return {}, f"start_entity_not_grounded_{index}"
        if not _VARIABLE_RE.fullmatch(variable):
            return {}, f"invalid_start_variable_{index}"
        if entity_type not in SUPPORTED_ENTITY_TYPES:
            return {}, f"unsupported_start_type_{index}"
        starts.append({"mention": mention, "variable": variable, "semantic_type": entity_type})

    is_branch = answer_operation != "project"
    expected_start = "$root_each" if is_branch else "$root"
    if any(row["variable"] != expected_start for row in starts):
        return {}, "invalid_start_variable_contract"
    if (is_branch and len(starts) < 2) or (not is_branch and len(starts) != 1):
        return {}, "invalid_start_entity_cardinality"

    raw_relations = payload.get("relations")
    if not isinstance(raw_relations, Sequence) or isinstance(raw_relations, (str, bytes)):
        return {}, "invalid_relation_list"
    if not 1 <= len(raw_relations) <= 4:
        return {}, "unsupported_relation_count"

    produced = {expected_start}
    relations = []
    for index, raw in enumerate(raw_relations):
        if not isinstance(raw, Mapping):
            return {}, f"invalid_relation_{index}"
        raw_forbidden = {
            "operator_id", "operator_type", "input_slot", "output_slot",
            "join_key", "join_key_contract", "input_role", "output_role",
            "physical_input_side", "physical_output_side", "answer", "value",
            "entity_value", "intermediate_value",
        }
        if raw_forbidden & set(raw):
            return {}, f"physical_or_value_field_in_relation_{index}"
        family = str(raw.get("predicate_family") or "").strip().casefold()
        if family not in PREDICATE_ENDPOINT_SCHEMA:
            return {}, f"unsupported_predicate_family_op{index}"
        if family == "kinship":
            return {}, f"ambiguous_kinship_family_op{index}"
        cues = _string_set(raw.get("relation_cues"))
        if not cues:
            return {}, f"missing_relation_cues_op{index}"
        if any(not _grounded(question, cue) for cue in cues):
            return {}, f"relation_cue_not_grounded_op{index}"
        input_variable = str(raw.get("input_variable") or "").strip()
        output_variable = str(raw.get("output_variable") or "").strip()
        if input_variable not in produced:
            return {}, f"unbound_input_variable_op{index}"
        if not _VARIABLE_RE.fullmatch(output_variable) or output_variable in produced:
            return {}, f"invalid_output_variable_op{index}"
        if index < len(raw_relations) - 1 and output_variable == "$answer":
            return {}, f"premature_answer_slot_op{index}"
        if index == len(raw_relations) - 1:
            expected_output = "$comparison_value" if is_branch else "$answer"
            if output_variable != expected_output:
                return {}, "terminal_variable_not_produced"
        output_type = str(raw.get("output_entity_type") or "entity").strip().casefold()
        if output_type not in SUPPORTED_ENTITY_TYPES:
            return {}, f"unsupported_output_type_op{index}"
        semantic_direction = str(raw.get("semantic_direction") or "").strip().casefold()
        if semantic_direction not in {"canonical", "inverse"}:
            return {}, f"invalid_semantic_direction_op{index}"
        canonical_relation = str(raw.get("canonical_relation") or "").strip().casefold()
        if canonical_relation and canonical_relation not in SUPPORTED_CANONICAL_RELATIONS:
            return {}, f"unsupported_canonical_relation_op{index}"
        if canonical_relation and canonical_relation != family:
            return {}, f"canonical_relation_family_mismatch_op{index}"
        explicit_subtype = str(raw.get("predicate_subtype") or "").strip().casefold()
        subtype, subtype_reason = _resolve_subtype(family, explicit_subtype, output_type, cues)
        if subtype_reason != "validated":
            return {}, f"{subtype_reason}_op{index}"
        spec = PREDICATE_ENDPOINT_SCHEMA[family]
        input_semantic_role = str(raw.get("input_semantic_role") or "").strip().casefold()
        output_semantic_role = str(raw.get("output_semantic_role") or "").strip().casefold()
        canonical_pair = (
            _role_matches(input_semantic_role, spec["left_role"])
            and _role_matches(output_semantic_role, spec["right_role"])
        )
        inverse_pair = (
            _role_matches(input_semantic_role, spec["right_role"])
            and _role_matches(output_semantic_role, spec["left_role"])
        )
        if bool(spec.get("symmetric")):
            canonical_pair = canonical_pair or inverse_pair
            inverse_pair = canonical_pair
        if not canonical_pair and not inverse_pair:
            return {}, f"semantic_endpoint_role_mismatch_op{index}"
        resolved_direction = "canonical" if canonical_pair else "inverse"
        if semantic_direction != resolved_direction and not bool(spec.get("symmetric")):
            return {}, f"semantic_direction_role_mismatch_op{index}"
        expected_types = spec["right_types"] if resolved_direction == "canonical" else spec["left_types"]
        if output_type not in expected_types and "entity" not in expected_types:
            return {}, f"semantic_output_type_mismatch_op{index}"
        relations.append({
            "relation_id": f"r{index}",
            "predicate_family": family,
            "predicate_subtype": subtype,
            "canonical_relation": canonical_relation,
            "relation_cues": sorted(cues),
            "input_variable": input_variable,
            "output_variable": output_variable,
            "input_semantic_role": input_semantic_role,
            "output_semantic_role": output_semantic_role,
            "semantic_direction": resolved_direction,
            "output_entity_type": output_type,
        })
        produced.add(output_variable)

    if answer_operation == "project" and relations[-1]["output_entity_type"] != final_type:
        return {}, "final_answer_type_mismatch"
    if answer_operation in {"argmin_root", "argmax_root"} and relations[-1]["output_entity_type"] not in {"date", "year", "number"}:
        return {}, "comparison_value_type_mismatch"
    semantic_ir = {
        "version": SEMANTIC_QUERY_IR_VERSION,
        "question": str(question or ""),
        "start_entities": starts,
        "relations": relations,
        "answer_operation": answer_operation,
        "final_answer_type": final_type,
        "final_answer_variable": "$answer",
        "contains_runtime_entity_values": False,
    }
    return semantic_ir, "validated"


_V2_1_FORBIDDEN_PHYSICAL_VARIABLES = {
    "$root", "$root_each", "$answer", "$comparison_value",
}


def normalize_semantic_ir_v2_1(
    question: str,
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """Validate V2.1 semantic variables, then reuse the strict V2 contracts.

    The adapter only performs lossless structural normalization. In particular,
    it may wrap one relation-cue string in a list, but it never invents a start
    entity, entity type, relation, endpoint role, direction, or answer value.
    """
    if bool(payload.get("abstain")):
        return {}, str(payload.get("reason") or "llm_abstained")

    answer_operation = str(payload.get("answer_operation") or "project").strip().casefold()
    if answer_operation not in SUPPORTED_ANSWER_OPERATIONS:
        return {}, "unsupported_answer_operation"
    is_branch = answer_operation != "project"
    final_variable = str(payload.get("final_answer_variable") or "").strip()
    if not _VARIABLE_RE.fullmatch(final_variable):
        return {}, "invalid_final_answer_variable"
    if final_variable in _V2_1_FORBIDDEN_PHYSICAL_VARIABLES:
        return {}, "physical_slot_used_as_semantic_variable"

    raw_starts = payload.get("start_entities")
    if not isinstance(raw_starts, Sequence) or isinstance(raw_starts, (str, bytes)) or not raw_starts:
        return {}, "missing_grounded_start_entity"
    starts = []
    for index, raw in enumerate(raw_starts):
        if not isinstance(raw, Mapping):
            return {}, f"invalid_start_entity_{index}"
        if any(str(raw.get(key) or "").strip() for key in ("answer", "value", "gold_answer")):
            return {}, f"invented_start_value_{index}"
        mention = str(raw.get("mention") or "").strip()
        variable = str(raw.get("variable") or "").strip()
        entity_type = str(raw.get("semantic_type") or "").strip().casefold()
        if not _grounded(question, mention):
            return {}, f"start_entity_not_grounded_{index}"
        if not _VARIABLE_RE.fullmatch(variable):
            return {}, f"invalid_start_variable_{index}"
        if variable in _V2_1_FORBIDDEN_PHYSICAL_VARIABLES:
            return {}, f"physical_start_variable_{index}"
        if entity_type not in SUPPORTED_ENTITY_TYPES:
            return {}, f"unsupported_start_type_{index}"
        starts.append({"mention": mention, "variable": variable, "semantic_type": entity_type})

    if (is_branch and len(starts) < 2) or (not is_branch and len(starts) != 1):
        return {}, "invalid_start_entity_cardinality"
    start_variables = {row["variable"] for row in starts}
    if len(start_variables) != 1:
        return {}, "multiple_semantic_root_variables"
    semantic_root = next(iter(start_variables))

    raw_relations = payload.get("relations")
    if not isinstance(raw_relations, Sequence) or isinstance(raw_relations, (str, bytes)):
        return {}, "invalid_relation_list"
    if not 1 <= len(raw_relations) <= 4:
        return {}, "unsupported_relation_count"

    produced = {semantic_root}
    adapted_relations = []
    lossless_adaptations = []
    for index, raw in enumerate(raw_relations):
        if not isinstance(raw, Mapping):
            return {}, f"invalid_relation_{index}"
        input_variable = str(raw.get("input_variable") or "").strip()
        output_variable = str(raw.get("output_variable") or "").strip()
        if input_variable not in produced:
            return {}, f"unbound_input_variable_op{index}"
        if not _VARIABLE_RE.fullmatch(output_variable) or output_variable in produced:
            return {}, f"invalid_output_variable_op{index}"
        if output_variable in _V2_1_FORBIDDEN_PHYSICAL_VARIABLES:
            return {}, f"physical_output_variable_op{index}"
        if not str(raw.get("output_entity_type") or "").strip():
            return {}, f"missing_output_entity_type_op{index}"
        adapted = dict(raw)
        raw_cues = raw.get("relation_cues")
        if isinstance(raw_cues, str):
            if not raw_cues.strip():
                return {}, f"missing_relation_cues_op{index}"
            adapted["relation_cues"] = [raw_cues.strip()]
            lossless_adaptations.append(f"relation_cues_scalar_to_list_op{index}")
        adapted_relations.append(adapted)
        produced.add(output_variable)

    terminal_relation_variable = str(adapted_relations[-1].get("output_variable") or "")
    if not is_branch and final_variable != terminal_relation_variable:
        return {}, "final_variable_not_terminal_relation_output"
    if is_branch and final_variable in produced:
        return {}, "branch_answer_variable_conflicts_with_relation_variable"



    physical_root = "$root_each" if is_branch else "$root"
    variable_map = {semantic_root: physical_root}
    legacy_relations = []
    for index, relation_row in enumerate(adapted_relations):
        legacy = dict(relation_row)
        legacy["input_variable"] = variable_map[str(relation_row["input_variable"])]
        if is_branch:
            physical_output = (
                "$comparison_value"
                if index == len(adapted_relations) - 1
                else f"$branch_v{index + 1}"
            )
        elif index == len(adapted_relations) - 1:
            physical_output = "$answer"
        else:
            physical_output = f"$v{index + 1}"
        variable_map[str(relation_row["output_variable"])] = physical_output
        legacy["output_variable"] = physical_output
        legacy_relations.append(legacy)

    legacy_payload = dict(payload)
    legacy_payload["start_entities"] = [
        {**row, "variable": physical_root} for row in starts
    ]
    legacy_payload["relations"] = legacy_relations
    legacy_payload["final_answer_variable"] = "$answer"
    legacy_semantic_ir, reason = normalize_semantic_ir_v2(question, legacy_payload)
    if not legacy_semantic_ir:
        return {}, reason

    restored_relations = []
    for original, validated in zip(adapted_relations, legacy_semantic_ir["relations"]):
        restored = dict(validated)
        restored["input_variable"] = str(original["input_variable"])
        restored["output_variable"] = str(original["output_variable"])
        restored_relations.append(restored)
    semantic_ir = dict(legacy_semantic_ir)
    semantic_ir.update({
        "version": SEMANTIC_QUERY_IR_V2_1_VERSION,
        "start_entities": starts,
        "relations": restored_relations,
        "final_answer_variable": final_variable,
        "lossless_protocol_adaptations": lossless_adaptations,
    })
    return semantic_ir, "validated"


def _question_terminal_category(question: str) -> str:
    """Return a fine-grained answer category only when explicitly requested."""
    text = _normalized_text(question)
    if not text:
        return ""
    category_aliases = {
        category: sorted(terms, key=len, reverse=True)
        for category, terms in TERMINAL_ENTITY_CATEGORY_TERMS.items()
    }
    for category, aliases in category_aliases.items():
        alternatives = "|".join(re.escape(alias) for alias in aliases)
        patterns = (
            rf"\b(?:what|which)\s+(?:[a-z-]+\s+){{0,3}}(?:{alternatives})\b",
            rf"\b(?:in|at|from|within)\s+(?:what|which)\s+(?:{alternatives})\b",
        )
        if any(re.search(pattern, text) for pattern in patterns):
            return category
    return ""


def normalize_semantic_ir_v2_2(
    question: str,
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """Validate a V2.2 relation chain and its semantic coverage certificate."""
    semantic_ir, reason = normalize_semantic_ir_v2_1(question, payload)
    if not semantic_ir:
        return {}, reason

    relations = list(semantic_ir.get("relations") or [])
    raw_requirements = payload.get("question_requirements")
    if not isinstance(raw_requirements, Sequence) or isinstance(
        raw_requirements, (str, bytes)
    ):
        return {}, "missing_question_requirements"
    if len(raw_requirements) != len(relations):
        return {}, "question_requirement_relation_count_mismatch"

    requested_category = _question_terminal_category(question)
    declared_category = str(
        payload.get("final_answer_category") or ""
    ).strip().casefold()
    if declared_category and declared_category not in TERMINAL_ENTITY_CATEGORY_TERMS:
        return {}, "unsupported_final_answer_category"
    if requested_category and declared_category != requested_category:
        return {}, "final_answer_category_question_mismatch"
    if not requested_category and declared_category:
        return {}, "ungrounded_final_answer_category"

    requirements = []
    seen_ids = set()
    seen_relation_indices = set()
    for requirement_index, raw in enumerate(raw_requirements):
        if not isinstance(raw, Mapping):
            return {}, f"invalid_question_requirement_{requirement_index}"
        requirement_id = str(raw.get("requirement_id") or "").strip()
        if not requirement_id or requirement_id in seen_ids:
            return {}, f"invalid_requirement_id_{requirement_index}"
        covered_index = raw.get("covered_by_relation_index")
        if isinstance(covered_index, bool) or not isinstance(covered_index, int):
            return {}, f"invalid_requirement_relation_index_{requirement_index}"
        if covered_index < 0 or covered_index >= len(relations):
            return {}, f"invalid_requirement_relation_index_{requirement_index}"
        if covered_index in seen_relation_indices:
            return {}, f"duplicate_requirement_relation_coverage_{covered_index}"
        relation_row = relations[covered_index]
        raw_cues = raw.get("relation_cues")
        if isinstance(raw_cues, str):
            raw_cues = [raw_cues]
        cues = _string_set(raw_cues)
        relation_cues = _string_set(relation_row.get("relation_cues"))
        if not cues or cues != relation_cues:
            return {}, f"requirement_relation_cue_mismatch_{requirement_index}"
        if any(not _grounded(question, cue) for cue in cues):
            return {}, f"requirement_cue_not_grounded_{requirement_index}"
        expected_type = str(
            raw.get("expected_output_type") or ""
        ).strip().casefold()
        if expected_type != str(relation_row.get("output_entity_type") or ""):
            return {}, f"requirement_output_type_mismatch_{requirement_index}"
        answer_category = str(raw.get("answer_category") or "").strip().casefold()
        if answer_category and answer_category not in TERMINAL_ENTITY_CATEGORY_TERMS:
            return {}, f"unsupported_requirement_answer_category_{requirement_index}"
        is_terminal = covered_index == len(relations) - 1
        if is_terminal:
            if answer_category != declared_category:
                return {}, "terminal_requirement_answer_category_mismatch"
        elif answer_category:
            return {}, f"nonterminal_requirement_has_answer_category_{requirement_index}"
        requirements.append({
            "requirement_id": requirement_id,
            "covered_by_relation_id": str(relation_row.get("relation_id") or ""),
            "covered_by_relation_index": covered_index,
            "relation_cues": sorted(cues),
            "expected_output_type": expected_type,
            "answer_category": answer_category,
        })
        seen_ids.add(requirement_id)
        seen_relation_indices.add(covered_index)

    expected_indices = set(range(len(relations)))
    if seen_relation_indices != expected_indices:
        return {}, "question_requirement_coverage_incomplete"
    certificate = {
        "version": "semantic_requirement_coverage_v2_2",
        "status": PLAN_COMPLETE,
        "reason": "all_declared_requirements_cross_checked",
        "independent_semantic_gold": False,
        "required_requirement_count": len(requirements),
        "covered_requirement_count": len(requirements),
        "covered_relation_indices": sorted(seen_relation_indices),
        "terminal_answer_category": declared_category,
        "question_surface_answer_category": requested_category,
    }
    semantic_ir.update({
        "version": SEMANTIC_QUERY_IR_V2_2_VERSION,
        "question_requirements": requirements,
        "final_answer_category": declared_category,
        "semantic_requirement_coverage": certificate,
    })
    return semantic_ir, "validated"


def _canonicalize_v2_3_terminal_category(
    question: str,
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], list[str], str]:
    """Normalize only categories justified by the question or terminal type.

    V2.2 required the model to emit the catalog key (for example ``state``),
    even when it emitted a catalog alias such as ``province``. V2.3 accepts
    such aliases but does not infer a fine-grained category that is absent from
    the question. Broad semantic roles remain represented by
    ``final_answer_type`` and therefore lower to an empty terminal category.
    """
    adapted = json.loads(json.dumps(dict(payload)))
    relations = adapted.get("relations")
    requirements = adapted.get("question_requirements")
    if not isinstance(relations, list) or not relations:
        return {}, [], "missing_relations"
    if not isinstance(requirements, list) or not requirements:
        return {}, [], "missing_question_requirements"

    declared = str(adapted.get("final_answer_category") or "").strip().casefold()
    requested = _question_terminal_category(question)
    terminal = relations[-1] if isinstance(relations[-1], Mapping) else {}
    terminal_role = str(terminal.get("output_semantic_role") or "").strip().casefold()
    terminal_type = str(terminal.get("output_entity_type") or "").strip().casefold()
    final_type = str(adapted.get("final_answer_type") or "").strip().casefold()
    adaptations: list[str] = []

    if requested:
        accepted_aliases = {
            str(value).strip().casefold()
            for value in TERMINAL_ENTITY_CATEGORY_TERMS.get(requested, ())
        }
        accepted_aliases.add(requested)
        if declared not in accepted_aliases:
            return {}, [], "final_answer_category_question_mismatch"
        canonical = requested
        if declared != canonical:
            adaptations.append(f"terminal_category_alias:{declared}->{canonical}")
    else:
        broad_categories = set(SUPPORTED_ENTITY_TYPES) | {terminal_role, terminal_type, final_type}
        if declared and declared not in broad_categories:
            return {}, [], "ungrounded_final_answer_category"
        canonical = ""
        if declared:
            adaptations.append(f"broad_terminal_category_removed:{declared}")

    adapted["final_answer_category"] = canonical
    terminal_requirement_index = len(relations) - 1
    for row in requirements:
        if not isinstance(row, dict):
            continue
        if row.get("covered_by_relation_index") == terminal_requirement_index:
            row["answer_category"] = canonical
    return adapted, adaptations, "validated"


def _v2_3_type_is_compatible(actual_type: str, expected_types: Iterable[str]) -> bool:
    """Check semantic variable flow without treating unrelated types as aliases."""
    actual = str(actual_type or "").strip().casefold()
    expected = {
        str(value or "").strip().casefold()
        for value in expected_types
        if str(value or "").strip()
    }
    if not actual or actual == "entity":
        return True
    if actual in expected:
        return True


    return actual == "year" and "date" in expected


def _validate_v2_3_variable_type_flow(
    semantic_ir: Mapping[str, Any],
) -> Tuple[bool, str, Dict[str, str]]:
    variable_types: Dict[str, str] = {}
    for row in semantic_ir.get("start_entities") or []:
        if isinstance(row, Mapping):
            variable_types[str(row.get("variable") or "")] = str(
                row.get("semantic_type") or "entity"
            ).strip().casefold()

    for index, relation in enumerate(semantic_ir.get("relations") or []):
        if not isinstance(relation, Mapping):
            return False, f"invalid_relation_{index}", variable_types
        family = str(relation.get("predicate_family") or "")
        spec = PREDICATE_ENDPOINT_SCHEMA.get(family)
        if spec is None:
            return False, f"unknown_endpoint_schema_op{index}", variable_types
        direction = str(relation.get("semantic_direction") or "")
        expected_input_types = (
            spec.get("left_types") if direction == "canonical"
            else spec.get("right_types")
        ) or ()
        input_variable = str(relation.get("input_variable") or "")
        actual_type = variable_types.get(input_variable, "")
        if not _v2_3_type_is_compatible(actual_type, expected_input_types):
            return (
                False,
                f"semantic_variable_type_flow_mismatch_op{index}",
                variable_types,
            )
        output_variable = str(relation.get("output_variable") or "")
        variable_types[output_variable] = str(
            relation.get("output_entity_type") or "entity"
        ).strip().casefold()
    return True, "validated", variable_types


def normalize_semantic_ir_v2_3(
    question: str,
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """Apply conservative category aliases and cross-edge type-flow checks."""



    structural_ir, reason = normalize_semantic_ir_v2_1(question, payload)
    if not structural_ir:
        return {}, reason
    adapted, category_adaptations, reason = _canonicalize_v2_3_terminal_category(
        question, payload
    )
    if not adapted:
        return {}, reason
    semantic_ir, reason = normalize_semantic_ir_v2_2(question, adapted)
    if not semantic_ir:
        return {}, reason
    valid, reason, variable_types = _validate_v2_3_variable_type_flow(semantic_ir)
    if not valid:
        return {}, reason

    certificate = dict(semantic_ir.get("semantic_requirement_coverage") or {})
    certificate["version"] = "semantic_requirement_coverage_v2_3"
    semantic_ir.update({
        "version": SEMANTIC_QUERY_IR_V2_3_VERSION,
        "semantic_requirement_coverage": certificate,
        "v2_3_category_adaptations": category_adaptations,
        "v2_3_variable_types": variable_types,
    })
    return semantic_ir, "validated"


def _physical_sides(spec: Mapping[str, Any], semantic_direction: str) -> Tuple[str, str, str]:
    layout = str(spec.get("physical_layout") or "either")
    if bool(spec.get("symmetric")) or layout == "either":
        return "either", "opposite_endpoint", "either"
    if layout != "subject_object":
        raise ValueError("unsupported_physical_layout")
    if semantic_direction == "canonical":
        return "subject", "object", "forward"
    return "object", "subject", "reverse"


def lower_semantic_ir_v2(
    question: str,
    semantic_ir: Mapping[str, Any],
    *,
    allow_external_plan_review: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """Lower validated semantic IR into the existing executable plan shape."""
    relations = [dict(row) for row in semantic_ir.get("relations") or []]
    answer_operation = str(semantic_ir.get("answer_operation") or "project")
    is_branch = answer_operation != "project"
    semantic_version = str(semantic_ir.get("version") or SEMANTIC_QUERY_IR_VERSION)
    is_v2_1 = semantic_version == SEMANTIC_QUERY_IR_V2_1_VERSION
    is_v2_2 = semantic_version == SEMANTIC_QUERY_IR_V2_2_VERSION
    is_v2_3 = semantic_version == SEMANTIC_QUERY_IR_V2_3_VERSION
    lowering_version = (
        PHYSICAL_LOWERING_V2_3_VERSION if is_v2_3
        else PHYSICAL_LOWERING_V2_2_VERSION if is_v2_2
        else PHYSICAL_LOWERING_V2_1_VERSION if is_v2_1
        else PHYSICAL_LOWERING_VERSION
    )
    compiler_mode = (
        "hybrid_v2_3_offline_replay" if is_v2_3
        else "hybrid_v2_2" if is_v2_2
        else "hybrid_v2_1" if is_v2_1 else "hybrid_v2"
    )
    starts = list(semantic_ir.get("start_entities") or [])
    start_variables = {
        str(row.get("variable") or "") for row in starts if isinstance(row, Mapping)
    }
    if len(start_variables) != 1:
        return {}, "lowering_requires_one_semantic_root_variable"
    semantic_root = next(iter(start_variables))
    operators = []
    physical_root = "$root_each" if is_branch else "$root"
    variable_to_slot = {semantic_root: physical_root}
    for index, relation in enumerate(relations):
        family = str(relation["predicate_family"])
        spec = PREDICATE_ENDPOINT_SCHEMA[family]
        physical_input, physical_output, physical_direction = _physical_sides(
            spec, str(relation["semantic_direction"])
        )
        input_slot = variable_to_slot[str(relation["input_variable"])]
        if is_branch:
            output_slot = (
                "$comparison_value"
                if index == len(relations) - 1
                else f"$branch_v{index + 1}"
            )
        else:
            output_slot = "$answer" if index == len(relations) - 1 else f"$v{index + 1}"
        variable_to_slot[str(relation["output_variable"])] = output_slot
        subtype = str(relation.get("predicate_subtype") or "")
        canonical_relation = str(relation.get("canonical_relation") or "")
        operator = {
            "operator_id": f"op{index}",
            "operator_type": "indexed_fact_lookup",
            "stage_index": index,
            "slot_plan_stage": "comparison_value" if is_branch else "final_answer" if index == len(relations) - 1 else "intermediate_binding",
            "input_slot": input_slot,
            "output_slot": output_slot,
            "predicate_families": [family],
            "relation_tokens": list(relation["relation_cues"]),
            "required_predicate_subtypes": [subtype] if subtype else [],
            "strict_predicate_subtype": bool(subtype),
            "output_type": str(relation["output_entity_type"]),
            "canonical_relation": canonical_relation,

            "input_role": str(relation["input_semantic_role"]),
            "output_role": str(relation["output_semantic_role"]),
            "semantic_input_role": str(relation["input_semantic_role"]),
            "semantic_output_role": str(relation["output_semantic_role"]),
            "semantic_direction": str(relation["semantic_direction"]),

            "physical_input_side": physical_input,
            "physical_output_side": physical_output,
            "physical_direction": physical_direction,
            "join_key_contract": "entity_identity+predicate_subtype+output_type+title_family" if subtype else "exact_normalized_entity_key",
            "role_direction_required": bool(canonical_relation),
            "role_contract_version": "role_directed_typed_join_v1",
            "lowering_version": lowering_version,
            "predicate_endpoint_schema_version": PREDICATE_ENDPOINT_SCHEMA_VERSION,
        }
        if is_v2_2 or is_v2_3:
            operator["target_entity_categories"] = (
                [str(semantic_ir.get("final_answer_category") or "")]
                if index == len(relations) - 1
                and str(semantic_ir.get("final_answer_category") or "")
                else []
            )
        operators.append(operator)

    terminal_operator = None
    logical_operators = list(operators)
    if is_branch:
        terminal_type = "compare_eq" if answer_operation == "compare_eq" else "argmin" if answer_operation == "argmin_root" else "argmax"
        terminal_operator = {
            "operator_id": "terminal",
            "operator_type": terminal_type,
            "stage_index": len(operators),
            "slot_plan_stage": "final_answer",
            "input_slot": "$comparison_value",
            "input_slots": ["$root_each", "$comparison_value"],
            "output_slot": "$answer",
            "output_type": str(semantic_ir["final_answer_type"]),
            "provenance_mode": "union_input_lineage",
            "requires_complete_branches": True,
        }
        logical_operators.append(terminal_operator)
    plan_kind = (
        "branch_compare" if answer_operation == "compare_eq" else
        "branch_temporal_compare" if is_branch else
        "relation_projection" if len(operators) == 1 else "path_projection"
    )
    slots = ["$root_each" if is_branch else "$root"] + [row["output_slot"] for row in operators]
    if is_branch:
        slots.append("$answer")
    plan: Dict[str, Any] = {
        "version": "slot_dependency_query_plan_v3",
        "logical_plan_revision": lowering_version,
        "enabled": True,
        "question": str(question or ""),
        "plan_kind": plan_kind,
        "answer_operator": answer_operation,
        "final_answer_type": str(semantic_ir["final_answer_type"]),
        "requires_bridge": len(operators) > 1 or is_branch,
        "slots": list(dict.fromkeys(slots)),
        "operators": operators,
        "steps": operators,
        "operator_count": len(operators),
        "physical_operator_count": len(operators),
        "logical_operators": logical_operators,
        "logical_operator_count": len(logical_operators),
        "terminal_operator": terminal_operator,
        "compile_status": "compiled",
        "compile_reason": "compiled_by_semantic_ir_v2",
        "semantic_query_ir": dict(semantic_ir),
        "semantic_ir_version": semantic_version,
        "physical_lowering_version": lowering_version,
        "predicate_endpoint_schema_version": PREDICATE_ENDPOINT_SCHEMA_VERSION,
        "source_support_required": True,
        "obligation_admission_required": True,
        "complete_witness_required": True,
    }
    question_contract = compile_question_obligations(question)
    coverage = assess_question_plan_coverage(question_contract, plan)
    semantic_validation: Dict[str, Any]
    if str(question_contract.get("parse_status") or "") != "parsed":
        if is_v2_2 or is_v2_3:
            semantic_coverage = dict(
                semantic_ir.get("semantic_requirement_coverage") or {}
            )
            if str(semantic_coverage.get("status") or "") != PLAN_COMPLETE:
                return {}, "semantic_requirement_coverage_incomplete"
            coverage = {
                "status": QUESTION_SEMANTIC_NEEDS_REVIEW,
                "reason": "self_declared_requirements_do_not_validate_original_question",
                "evidence": "structural_cross_check_only_not_question_semantic_validation",
                "required_obligation_count": int(
                    semantic_coverage.get("required_requirement_count") or 0
                ),
                "covered_obligation_count": int(
                    semantic_coverage.get("covered_requirement_count") or 0
                ),
                "self_declared_semantic_requirement_coverage": semantic_coverage,
            }
        else:
            coverage = {
                "status": QUESTION_SEMANTIC_NEEDS_REVIEW,
                "reason": "semantic_ir_declared_obligations_require_review",
                "evidence": "semantic_ir_is_not_question_semantic_validation",
                "required_obligation_count": len(operators),
                "covered_obligation_count": len(operators),
            }
        semantic_validation = {
            "version": QUESTION_SEMANTIC_VALIDATION_VERSION,
            "status": QUESTION_SEMANTIC_NEEDS_REVIEW,
            "basis": "self_declared_semantic_requirements_only",
            "independent_question_contract": False,
            "human_reviewed": False,
            "auto_execution_eligible": False,
        }
    elif str(coverage.get("status") or "") != PLAN_COMPLETE:
        if not allow_external_plan_review:
            return {}, "semantic_plan_failed_question_contract"
        coverage = {
            "status": QUESTION_SEMANTIC_NEEDS_REVIEW,
            "reason": "rule_question_contract_conflicts_with_ai_development_spec",
            "evidence": "ai_development_diagnostic_only_not_semantic_proof",
            "required_obligation_count": len(operators),
            "covered_obligation_count": len(operators),
            "rule_question_contract": question_contract,
        }
        semantic_validation = {
            "version": QUESTION_SEMANTIC_VALIDATION_VERSION,
            "status": QUESTION_SEMANTIC_NEEDS_REVIEW,
            "basis": "explicit_ai_development_diagnostic_only",
            "independent_question_contract": False,
            "human_reviewed": False,
            "auto_execution_eligible": False,
        }
    else:
        semantic_validation = {
            "version": QUESTION_SEMANTIC_VALIDATION_VERSION,
            "status": QUESTION_SEMANTIC_CONTRACT_VALIDATED,
            "basis": "independent_rule_question_contract",
            "independent_question_contract": True,
            "human_reviewed": False,
            "auto_execution_eligible": True,
        }
    plan.update({
        "question_obligation_contract": question_contract,
        "question_obligations": list(question_contract.get("obligations") or []),
        "plan_coverage_contract_version": PLAN_COVERAGE_CONTRACT_VERSION,
        "plan_coverage": coverage,
        "plan_coverage_status": str(coverage.get("status") or ""),
        "question_semantic_validation": semantic_validation,
    })
    validation = validate_v2_physical_plan(plan)
    if not validation["valid"]:
        return {}, str(validation["reason"])
    plan = attach_logical_query_plan_identity(
        plan,
        compiler_source="llm_semantic_fallback",
        compiler_mode=compiler_mode,
        attempt_id="attempt1",
    )
    return plan, "validated"


def validate_v2_physical_plan(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply shared executor validation plus V2 semantic/physical invariants."""
    base = validate_logical_query_plan_static(plan)
    if not base["valid"]:
        return base
    semantic_version = str(plan.get("semantic_ir_version") or "")
    if semantic_version not in {
        SEMANTIC_QUERY_IR_VERSION,
        SEMANTIC_QUERY_IR_V2_1_VERSION,
        SEMANTIC_QUERY_IR_V2_2_VERSION,
        SEMANTIC_QUERY_IR_V2_3_VERSION,
    }:
        return {"valid": False, "reason": "semantic_ir_version_missing"}
    expected_lowering_version = (
        PHYSICAL_LOWERING_V2_3_VERSION
        if semantic_version == SEMANTIC_QUERY_IR_V2_3_VERSION
        else PHYSICAL_LOWERING_V2_2_VERSION
        if semantic_version == SEMANTIC_QUERY_IR_V2_2_VERSION
        else PHYSICAL_LOWERING_V2_1_VERSION
        if semantic_version == SEMANTIC_QUERY_IR_V2_1_VERSION
        else PHYSICAL_LOWERING_VERSION
    )
    if str(plan.get("physical_lowering_version") or "") != expected_lowering_version:
        return {"valid": False, "reason": "semantic_physical_version_mismatch"}
    semantic_ir = plan.get("semantic_query_ir") or {}
    if bool(semantic_ir.get("contains_runtime_entity_values")):
        return {"valid": False, "reason": "runtime_entity_value_in_semantic_ir"}
    relations = list(semantic_ir.get("relations") or [])
    operators = list(plan.get("operators") or [])
    if len(relations) != len(operators):
        return {"valid": False, "reason": "semantic_physical_operator_count_mismatch"}
    if semantic_version in {
        SEMANTIC_QUERY_IR_V2_2_VERSION,
        SEMANTIC_QUERY_IR_V2_3_VERSION,
    }:
        requirements = list(semantic_ir.get("question_requirements") or [])
        certificate = dict(semantic_ir.get("semantic_requirement_coverage") or {})
        if len(requirements) != len(relations):
            return {"valid": False, "reason": "semantic_requirement_count_mismatch"}
        if str(certificate.get("status") or "") != PLAN_COMPLETE:
            return {"valid": False, "reason": "semantic_requirement_certificate_missing"}
        final_category = str(semantic_ir.get("final_answer_category") or "")
        for index, operator in enumerate(operators):
            expected_categories = [final_category] if final_category and index == len(operators) - 1 else []
            if list(operator.get("target_entity_categories") or []) != expected_categories:
                return {"valid": False, "reason": f"terminal_category_lowering_mismatch_op{index}"}
    for index, (relation, operator) in enumerate(zip(relations, operators)):
        family = str(relation.get("predicate_family") or "")
        if family not in PREDICATE_ENDPOINT_SCHEMA:
            return {"valid": False, "reason": f"unknown_endpoint_schema_op{index}"}
        if list(operator.get("predicate_families") or []) != [family]:
            return {"valid": False, "reason": f"family_lowering_mismatch_op{index}"}
        spec = PREDICATE_ENDPOINT_SCHEMA[family]
        expected_input, expected_output, expected_direction = _physical_sides(
            spec, str(relation.get("semantic_direction") or "")
        )
        actual = (
            str(operator.get("physical_input_side") or ""),
            str(operator.get("physical_output_side") or ""),
            str(operator.get("physical_direction") or ""),
        )
        if actual != (expected_input, expected_output, expected_direction):
            return {"valid": False, "reason": f"physical_endpoint_lowering_mismatch_op{index}"}
        if operator.get("input_role") != relation.get("input_semantic_role") or operator.get("output_role") != relation.get("output_semantic_role"):
            return {"valid": False, "reason": f"semantic_role_lowering_mismatch_op{index}"}
    for key in ("source_support_required", "obligation_admission_required", "complete_witness_required"):
        if not bool(plan.get(key)):
            return {"valid": False, "reason": f"safety_contract_missing_{key}"}
    return {"valid": True, "reason": "validated", "operator_count": len(operators)}


def compile_semantic_payload_v2(
    question: str,
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    semantic_ir, reason = normalize_semantic_ir_v2(question, payload)
    if not semantic_ir:
        return {}, {}, reason
    physical_plan, reason = lower_semantic_ir_v2(question, semantic_ir)
    return semantic_ir, physical_plan, reason


def compile_semantic_payload_v2_1(
    question: str,
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    semantic_ir, reason = normalize_semantic_ir_v2_1(question, payload)
    if not semantic_ir:
        return {}, {}, reason
    physical_plan, reason = lower_semantic_ir_v2(question, semantic_ir)
    return semantic_ir, physical_plan, reason


def compile_semantic_payload_v2_2(
    question: str,
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    semantic_ir, reason = normalize_semantic_ir_v2_2(question, payload)
    if not semantic_ir:
        return {}, {}, reason
    physical_plan, reason = lower_semantic_ir_v2(question, semantic_ir)
    return semantic_ir, physical_plan, reason


def compile_semantic_payload_v2_3(
    question: str,
    payload: Mapping[str, Any],
    *,
    allow_external_plan_review: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    semantic_ir, reason = normalize_semantic_ir_v2_3(question, payload)
    if not semantic_ir:
        return {}, {}, reason
    physical_plan, reason = lower_semantic_ir_v2(
        question,
        semantic_ir,
        allow_external_plan_review=allow_external_plan_review,
    )
    return semantic_ir, physical_plan, reason
