"""Role-explicit query IR for source-scoped fact lookup.

The compiler sees only an instantiated atomic question and its already-bound
values.  It cannot inspect the source passage or an evaluation answer.  The
executor then treats the compiled roles as immutable and may materialize a
tuple only from an exact source quote.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

from .late_bound_textual_relation_utils import (
    PLACEHOLDER_RE,
    normalize_surface,
    validate_supported_extraction,
)


OUTPUT_TYPES = {
    "person",
    "organization",
    "location",
    "date",
    "number",
    "work",
    "event",
    "entity",
    "text",
    "boolean",
}
CARDINALITIES = {"single", "list"}
DIRECTIONS = {"input_to_output", "output_to_input", "symmetric"}
TYPE_ALIASES = {
    "actor": "person",
    "artist": "person",
    "author": "person",
    "composer": "person",
    "human": "person",
    "politician": "person",
    "writer": "person",
    "company": "organization",
    "institution": "organization",
    "political_party": "organization",
    "sports_team": "organization",
    "team": "organization",
    "city": "location",
    "country": "location",
    "geopolitical_entity": "location",
    "place": "location",
    "state": "location",
    "temporal": "date",
    "year": "date",
    "integer": "number",
    "quantity": "number",
    "book": "work",
    "creative_work": "work",
    "film": "work",
    "song": "work",
    "factoid": "entity",
}


def _compact_text(value: Any, *, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _canonical_output_type(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "entity").lower()).strip("_")
    normalized = TYPE_ALIASES.get(normalized, normalized)
    return normalized if normalized in OUTPUT_TYPES else "entity"


def _semantic_match_flag(value: Any, expected: Any) -> bool:
    """Accept a boolean match or a legacy response naming the matched value."""

    if value is True:
        return True
    if not isinstance(value, str):
        return False
    return normalize_surface(value).replace(" ", "_") == normalize_surface(expected).replace(" ", "_")


def _surface_in_question(surface: str, question: str) -> bool:
    surface_norm = normalize_surface(surface)
    question_norm = normalize_surface(question)
    if not surface_norm:
        return False
    if surface_norm in question_norm:
        return True
    tokens = set(surface_norm.split())
    question_tokens = set(question_norm.split())
    return bool(tokens) and len(tokens & question_tokens) / len(tokens) >= 0.75


def build_role_ir_compiler_messages(
    question: str,
    *,
    bound_surfaces: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Build a query-only semantic compilation request.

    Deliberately absent from this prompt: source text, source title, gold
    answer, dataset name, and any previous extraction.
    """

    binding_text = ", ".join(_compact_text(value) for value in bound_surfaces) or "none"
    system = (
        "Compile one atomic fact question into a role-explicit relational lookup. "
        "You define semantics only; you must not answer the question or invent facts. "
        "Identify every entity/value already present in the question as an input argument, "
        "give each argument its semantic role, define one canonical relation, and define "
        "the exact role and type of the requested output. Preserve relation direction. "
        "For 'Who played Barney Stinson?', Barney Stinson is input role=character, "
        "relation=portrayed_by, output role=actor, output type=person. For "
        "'John Cabot >> child', John Cabot is input role=parent, relation=has_child, "
        "output role=child. For 'when did Birmingham City beat Aston Villa', use two "
        "input arguments with roles winner and opponent and output role=match_date. "
        "Return one JSON object with keys supported, input_arguments, relation, direction, "
        "output_role, output_type, cardinality, granularity, and constraints. "
        "input_arguments must be a JSON list of objects with surface and role. "
        "direction must be input_to_output, output_to_input, or symmetric."
    )
    user = (
        f"Atomic question: { _compact_text(question, limit=500) }\n"
        f"Already-bound values: {binding_text}\n"
        "Compile the semantic contract only. Do not provide an answer."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def normalize_role_explicit_ir(
    payload: Mapping[str, Any],
    *,
    question: str,
    bound_surfaces: Sequence[str] = (),
    stage_index: int = 0,
    final: bool = False,
) -> dict[str, Any]:
    """Validate untrusted compiler output and create an executable query IR."""

    atomic_question = _compact_text(question, limit=500)
    unresolved = sorted({int(value) for value in PLACEHOLDER_RE.findall(atomic_question)})
    raw_arguments = payload.get("input_arguments")
    if not isinstance(raw_arguments, list):
        raw_arguments = []
    arguments: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_arguments[:4]):
        if not isinstance(raw, Mapping):
            continue
        surface = _compact_text(raw.get("surface"))
        role = _compact_text(raw.get("role"))
        if not surface or not role or not _surface_in_question(surface, atomic_question):
            continue
        key = (normalize_surface(surface), normalize_surface(role))
        if key in seen:
            continue
        seen.add(key)
        arguments.append({
            "slot": f"$input{index}",
            "surface": surface,
            "role": role,
        })

    relation = _compact_text(payload.get("relation"))
    output_role = _compact_text(payload.get("output_role"))
    output_type = _canonical_output_type(payload.get("output_type"))
    cardinality = str(payload.get("cardinality") or "single").strip().lower()
    cardinality = cardinality if cardinality in CARDINALITIES else "single"
    direction = str(payload.get("direction") or "input_to_output").strip().lower()
    direction = direction if direction in DIRECTIONS else "input_to_output"
    granularity = _compact_text(payload.get("granularity")) or "minimal_answer_span"
    raw_constraints = payload.get("constraints")
    constraints = [
        _compact_text(value) for value in raw_constraints[:8]
        if _compact_text(value)
    ] if isinstance(raw_constraints, list) else []

    compiler_supported = payload.get("supported") is True
    failures = []
    for condition, reason in (
        (compiler_supported, "compiler_declared_unsupported"),
        (bool(atomic_question), "empty_question"),
        (not unresolved, "unresolved_input_binding"),
        (bool(arguments), "missing_input_arguments"),
        (bool(relation), "missing_relation"),
        (bool(output_role), "missing_output_role"),
    ):
        if not condition:
            failures.append(reason)
    compiled = not failures
    explicit_bindings = [
        _compact_text(value) for value in bound_surfaces if _compact_text(value)
    ]
    input_bindings = explicit_bindings or [argument["surface"] for argument in arguments]
    return {
        "operator_id": f"role_ir_op{int(stage_index)}",
        "operator_type": "role_explicit_source_lookup",
        "stage_index": int(stage_index),
        "compile_status": "compiled" if compiled else "unsupported",
        "compile_failures": failures,
        "atomic_question": atomic_question,
        "input_arguments": arguments,
        "input_bindings": input_bindings,
        "relation": relation,
        "direction": direction,
        "output_slot": "$answer",
        "output_role": output_role,
        "output_type": output_type,
        "cardinality": cardinality,
        "granularity": granularity,
        "constraints": constraints,
        "source_scope": "exact_source_passage",
        "slot_plan_stage": "answer_projection" if final else "intermediate_binding",
        "admission_contract": (
            "role_alignment+relation_entailment+exact_quote+answer_in_quote+type_compatible"
        ),
        "query_only_compilation": True,
    }


def build_role_bound_extraction_messages(
    query_ir: Mapping[str, Any],
    *,
    source_title: str,
    source_text: str,
) -> list[dict[str, str]]:
    """Build an extraction request whose semantic contract is immutable."""

    arguments = list(query_ir.get("input_arguments") or [])
    argument_text = "; ".join(
        f"{item.get('role', '')}={item.get('surface', '')}" for item in arguments
    ) or "none"
    constraints = "; ".join(map(str, query_ir.get("constraints") or [])) or "none"
    system = (
        "Execute the supplied role-explicit lookup against exactly one source passage. "
        "The query IR is immutable: do not reverse its direction, rename its roles, infer a "
        "different relation, or use outside knowledge. First decide whether the passage "
        "grounds the input arguments and entails the stated relation in the stated direction. "
        "Then copy the shortest exact source substring that fills the output role. Copy one "
        "contiguous support quote verbatim. If any semantic condition fails, return supported=false. "
        "Return one JSON object with keys supported, binding_consistent, relation_entailed, "
        "output_role_matched, observed_output_role, cardinality_matched, observed_cardinality, "
        "answer, support_quote, and reason_code. output_role_matched and cardinality_matched "
        "must be JSON booleans, while the observed fields must be strings."
    )
    user = (
        f"Atomic question: {query_ir.get('atomic_question', '')}\n"
        f"Input arguments: {argument_text}\n"
        f"Canonical relation: {query_ir.get('relation', '')}\n"
        f"Direction: {query_ir.get('direction', 'input_to_output')}\n"
        f"Required output role: {query_ir.get('output_role', '')}\n"
        f"Required output type: {query_ir.get('output_type', 'entity')}\n"
        f"Cardinality: {query_ir.get('cardinality', 'single')}\n"
        f"Granularity: {query_ir.get('granularity', 'minimal_answer_span')}\n"
        f"Additional constraints: {constraints}\n"
        f"Source title: {source_title}\n"
        f"Source passage:\n{source_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def validate_role_bound_extraction(
    extraction: Mapping[str, Any],
    query_ir: Mapping[str, Any],
    *,
    source_title: str,
    source_text: str,
) -> dict[str, Any]:
    """Admit a tuple only when semantic roles and source provenance agree."""

    deterministic = validate_supported_extraction(
        {
            "supported": extraction.get("supported") is True,
            "answer": extraction.get("answer", ""),
            "support_quote": extraction.get("support_quote", ""),
        },
        {
            "input_bindings": [],
            "output_type": query_ir.get("output_type", "entity"),
        },
        source_title=source_title,
        source_text=source_text,
    )
    model_supported = extraction.get("supported") is True
    binding_consistent = extraction.get("binding_consistent") is True
    relation_entailed = extraction.get("relation_entailed") is True
    output_role_value = extraction.get("output_role_matched")
    if output_role_value is None:
        output_role_value = extraction.get("output_role_filled")
    cardinality_value = extraction.get("cardinality_matched")
    if cardinality_value is None:
        cardinality_value = extraction.get("cardinality_fit")
    output_role_filled = _semantic_match_flag(
        output_role_value,
        query_ir.get("output_role", ""),
    )
    cardinality_fit = _semantic_match_flag(
        cardinality_value,
        query_ir.get("cardinality", "single"),
    )
    provenance_valid = bool(
        deterministic.get("quote_in_source")
        and deterministic.get("answer_in_quote")
        and deterministic.get("type_compatible")
    )
    accepted = bool(
        model_supported
        and binding_consistent
        and relation_entailed
        and output_role_filled
        and cardinality_fit
        and provenance_valid
    )
    rejection_reasons = []
    for condition, reason in (
        (model_supported, "model_declared_unsupported"),
        (binding_consistent, "input_role_not_grounded"),
        (relation_entailed, "relation_not_entailed"),
        (output_role_filled, "output_role_not_filled"),
        (cardinality_fit, "cardinality_mismatch"),
        (bool(deterministic.get("quote_in_source")), "quote_not_in_source"),
        (bool(deterministic.get("answer_in_quote")), "answer_not_in_quote"),
        (bool(deterministic.get("type_compatible")), "answer_type_mismatch"),
    ):
        if not condition:
            rejection_reasons.append(reason)
    return {
        "accepted": accepted,
        "answer": deterministic.get("answer", ""),
        "support_quote": deterministic.get("support_quote", ""),
        "binding_consistent": binding_consistent,
        "relation_entailed": relation_entailed,
        "output_role_filled": output_role_filled,
        "observed_output_role": _compact_text(
            extraction.get("observed_output_role")
            or (output_role_value if isinstance(output_role_value, str) else "")
        ),
        "cardinality_fit": cardinality_fit,
        "observed_cardinality": _compact_text(
            extraction.get("observed_cardinality")
            or (cardinality_value if isinstance(cardinality_value, str) else "")
        ),
        "provenance_valid": provenance_valid,
        "quote_in_source": bool(deterministic.get("quote_in_source")),
        "quote_recovered_from_source": bool(deterministic.get("quote_recovered_from_source")),
        "answer_in_quote": bool(deterministic.get("answer_in_quote")),
        "type_compatible": bool(deterministic.get("type_compatible")),
        "source_sentence_id": int(deterministic.get("source_sentence_id", -1)),
        "reason_code": _compact_text(extraction.get("reason_code")),
        "rejection_reasons": rejection_reasons,
    }


def build_role_filter_messages(
    query_ir: Mapping[str, Any],
    *,
    candidate_answer: str,
    candidate_quote: str,
    source_title: str,
    source_text: str,
) -> list[dict[str, str]]:
    """Build a filter-only verification request that cannot rewrite a tuple."""

    arguments = list(query_ir.get("input_arguments") or [])
    argument_text = "; ".join(
        f"{item.get('role', '')}={item.get('surface', '')}" for item in arguments
    ) or "none"
    system = (
        "You are a filter-only relational tuple verifier. You may accept or reject the "
        "candidate, but you must never propose, rewrite, or return another answer. Check the "
        "immutable query roles, relation direction, candidate output role, cardinality, and "
        "the supplied source quote. Use only the source passage. Return one JSON object with "
        "keys valid, binding_consistent, relation_entailed, candidate_fills_output_role, "
        "cardinality_fit, and reason_code. Do not include an answer field."
    )
    user = (
        f"Input arguments: {argument_text}\n"
        f"Canonical relation: {query_ir.get('relation', '')}\n"
        f"Direction: {query_ir.get('direction', 'input_to_output')}\n"
        f"Required output role: {query_ir.get('output_role', '')}\n"
        f"Required output type: {query_ir.get('output_type', 'entity')}\n"
        f"Cardinality: {query_ir.get('cardinality', 'single')}\n"
        f"Candidate answer: {candidate_answer}\n"
        f"Candidate support quote: {candidate_quote}\n"
        f"Source title: {source_title}\n"
        f"Source passage:\n{source_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def validate_role_filter(
    verdict: Mapping[str, Any],
    candidate_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply an immutable accept/reject verdict to a provenance-valid candidate."""

    candidate_valid = bool(candidate_validation.get("accepted"))
    filter_valid = verdict.get("valid") is True
    binding_consistent = verdict.get("binding_consistent") is True
    relation_entailed = verdict.get("relation_entailed") is True
    output_role_filled = verdict.get("candidate_fills_output_role") is True
    cardinality_fit = verdict.get("cardinality_fit") is True
    forbidden_rewrite = any(
        key in verdict and _compact_text(verdict.get(key))
        for key in ("answer", "verified_answer", "corrected_answer", "replacement_answer")
    )
    accepted = bool(
        candidate_valid
        and filter_valid
        and binding_consistent
        and relation_entailed
        and output_role_filled
        and cardinality_fit
        and not forbidden_rewrite
    )
    rejection_reasons = []
    for condition, reason in (
        (candidate_valid, "candidate_failed_source_admission"),
        (filter_valid, "filter_declared_invalid"),
        (binding_consistent, "filter_binding_inconsistent"),
        (relation_entailed, "filter_relation_not_entailed"),
        (output_role_filled, "filter_output_role_mismatch"),
        (cardinality_fit, "filter_cardinality_mismatch"),
        (not forbidden_rewrite, "filter_attempted_answer_rewrite"),
    ):
        if not condition:
            rejection_reasons.append(reason)
    return {
        "accepted": accepted,
        "answer": candidate_validation.get("answer", ""),
        "support_quote": candidate_validation.get("support_quote", ""),
        "candidate_valid": candidate_valid,
        "filter_valid": filter_valid,
        "binding_consistent": binding_consistent,
        "relation_entailed": relation_entailed,
        "output_role_filled": output_role_filled,
        "cardinality_fit": cardinality_fit,
        "forbidden_rewrite": forbidden_rewrite,
        "reason_code": _compact_text(verdict.get("reason_code")),
        "rejection_reasons": rejection_reasons,
    }


def materialize_role_explicit_tuple(
    query_ir: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    source_id: Any,
    source_title: str,
) -> dict[str, Any] | None:
    """Materialize the accepted role-explicit result as a query-local tuple."""

    if not validation.get("accepted"):
        return None
    arguments = list(query_ir.get("input_arguments") or [])
    subject = " | ".join(
        f"{item.get('role', '')}={item.get('surface', '')}" for item in arguments
    )
    answer = str(validation.get("answer") or "").strip()
    relation = str(query_ir.get("relation") or "").strip()
    fingerprint = "|".join((str(source_id), subject, relation, answer))
    return {
        "fact_id": "role_ir:" + hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:16],
        "triple": [subject or source_title, relation, answer],
        "input_arguments": arguments,
        "output_role": query_ir.get("output_role", ""),
        "output_type": query_ir.get("output_type", "entity"),
        "source_id": source_id,
        "source_title": source_title,
        "source_sentence_id": validation.get("source_sentence_id", -1),
        "source_sentence_span": validation.get("support_quote", ""),
        "origin": "role_explicit_query_local_overlay",
        "tuple_state": "SUPPORTED",
        "support_status": "SUPPORTED",
        "operator_id": query_ir.get("operator_id"),
        "provenance_scope": "exact_source_quote",
    }
