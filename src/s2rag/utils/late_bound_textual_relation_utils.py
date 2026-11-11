"""Late-bound textual relation operators for imperfect fact indexes.

The operator deliberately carries an atomic natural-language relation instead
of forcing it into a closed predicate vocabulary.  It may only materialize a
query-local tuple when the returned answer and supporting quote are both
grounded in the source passage.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .sentence_evidence_utils import infer_answer_target_type


PLACEHOLDER_RE = re.compile(r"#(\d+)\b")
YEAR_RE = re.compile(
    r"(?:\b\d{3,4}\b|\b\d{1,2}(?:st|nd|rd|th)\s+century\b)",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
BOOLEAN_VALUES = {"yes", "no", "true", "false"}
IDENTITY_STOPWORDS = {
    "a", "an", "and", "company", "corporation", "entertainment", "group",
    "inc", "limited", "of", "the",
}


def normalize_surface(value: Any) -> str:
    """Normalize whitespace and case while retaining Unicode word content."""

    text = str(value or "").casefold().replace("–", "-").replace("—", "-")
    text = re.sub(r"[^\w%+.-]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def instantiate_bound_question(
    question: str,
    prior_answers: Mapping[int, Any],
) -> str:
    """Bind known decomposition variables without touching unresolved slots."""

    return PLACEHOLDER_RE.sub(
        lambda match: str(prior_answers.get(int(match.group(1)), match.group(0))),
        str(question or ""),
    )


def infer_textual_answer_contract(question: str) -> dict[str, str]:
    """Infer only output-shape constraints, not a closed relation vocabulary."""

    normalized = normalize_surface(question)
    target_type = infer_answer_target_type(question) or "entity"
    if re.search(r"\b(?:place of birth|birthplace|located|location)\b", normalized):
        target_type = "location"
    elif re.search(r"\b(?:screenwriter|writer|author|director|composer|child|father|mother|spouse)\b", normalized):
        target_type = "person"
    elif re.search(r"\b(?:what year|which year|in what year|in which year)\b", normalized):
        target_type = "date"
    if re.search(r"\b(?:countries|people|persons|names|writers|authors|children)\b", normalized):
        cardinality = "list"
    else:
        cardinality = "single"
    if re.search(r"\b(?:what|which) country\b", normalized):
        granularity = "country"
    elif re.search(r"\b(?:what|which) (?:city|town)\b", normalized):
        granularity = "city"
    elif re.search(r"\b(?:what|which) state\b", normalized):
        granularity = "state"
    elif re.search(r"\b(?:what|which) year\b", normalized):
        granularity = "year"
    elif target_type == "date":
        granularity = "minimal_temporal_value"
    elif target_type == "location":
        granularity = "minimal_location_value"
    else:
        granularity = "minimal_answer_span"
    return {
        "target_type": target_type,
        "cardinality": cardinality,
        "granularity": granularity,
        "surface_policy": "shortest_exact_source_span",
    }


def compile_late_bound_textual_operator(
    question: str,
    *,
    bound_surfaces: Sequence[str] = (),
    stage_index: int = 0,
    final: bool = False,
) -> dict[str, Any]:
    """Compile any non-empty atomic question into a provenance-safe operator."""

    normalized_question = " ".join(str(question or "").split())
    unresolved = sorted({int(value) for value in PLACEHOLDER_RE.findall(normalized_question)})
    compile_status = "compiled" if normalized_question and not unresolved else "unsupported"
    answer_contract = infer_textual_answer_contract(normalized_question)
    return {
        "operator_id": f"late_bound_op{int(stage_index)}",
        "operator_type": "late_bound_text_lookup",
        "stage_index": int(stage_index),
        "compile_status": compile_status,
        "compile_reason": (
            "atomic_textual_relation"
            if compile_status == "compiled"
            else "unresolved_input_binding" if unresolved else "empty_question"
        ),
        "relation_spec": normalized_question,
        "input_bindings": [str(value) for value in bound_surfaces if str(value).strip()],
        "unresolved_slots": unresolved,
        "input_slot": "$bound_entity" if bound_surfaces else "$source_partition",
        "output_slot": "$answer",
        "output_type": answer_contract["target_type"],
        "answer_contract": answer_contract,
        "slot_plan_stage": "answer_projection" if final else "intermediate_binding",
        "access_path": "source_scoped_text_scan",
        "join_key_contract": "bound_surface_or_source_partition",
        "admission_contract": "answer_in_exact_quote+quote_in_source+type_compatible",
        "output_relation": "query_local_overlay",
    }


def build_late_bound_messages(
    operator: Mapping[str, Any],
    *,
    source_title: str,
    source_text: str,
) -> list[dict[str, str]]:
    """Build a compact extraction request without exposing gold answers."""

    bindings = list(operator.get("input_bindings") or [])
    binding_text = ", ".join(str(value) for value in bindings) or "none"
    answer_contract = dict(operator.get("answer_contract") or {})
    system = (
        "You execute one source-scoped relational lookup. Extract only a value "
        "explicitly supported by the supplied passage. The answer must be the shortest "
        "exact substring that fills the requested slot; do not return a broader location, "
        "a whole list, or an explanatory phrase unless the question asks for it. Copy one "
        "contiguous supporting quote verbatim from the passage: never add ellipses, repair "
        "spelling, or paraphrase it. If the passage does not answer the question, "
        "return supported=false and empty answer/quote. Never use outside knowledge. "
        "Return one JSON object with keys supported, answer, support_quote, subject, "
        "and relation_phrase."
    )
    user = (
        f"Atomic relation question: {operator.get('relation_spec', '')}\n"
        f"Expected answer type: {operator.get('output_type', 'entity')}\n"
        f"Answer cardinality: {answer_contract.get('cardinality', 'single')}\n"
        f"Answer granularity: {answer_contract.get('granularity', 'minimal_answer_span')}\n"
        f"Already bound values: {binding_text}\n"
        f"Source title: {source_title}\n"
        f"Source passage:\n{source_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_relation_verification_messages(
    operator: Mapping[str, Any],
    *,
    candidate_answer: str,
    candidate_quote: str,
    source_title: str,
    source_text: str,
) -> list[dict[str, str]]:
    """Build an independent semantic verification request for one candidate tuple."""

    contract = dict(operator.get("answer_contract") or {})
    bindings = list(operator.get("input_bindings") or [])
    system = (
        "You are a strict relational tuple verifier, not an answer generator. Decide whether "
        "the proposed answer fills the exact output role requested by the atomic question in "
        "the supplied source. Check relation meaning, input-entity binding, output role, "
        "cardinality, and granularity independently. A related entity is not enough: for a "
        "question asking who played a character, the character name is the wrong role; a "
        "former manager is not an owner; a relation label such as 'sister' is not a person's "
        "name. If the candidate is wrong but the same source explicitly contains the correct "
        "value, return that correction. The verified answer must be the shortest exact "
        "contiguous substring copied from the source, and support_quote must be one exact "
        "contiguous source quote without ellipses or paraphrase. Return one JSON object with "
        "keys relation_entailed, binding_consistent, slot_fit, cardinality_fit, "
        "verified_answer, support_quote, and reason_code. Never use outside knowledge."
    )
    user = (
        f"Atomic relation question: {operator.get('relation_spec', '')}\n"
        f"Bound input values: {', '.join(map(str, bindings)) or 'none'}\n"
        f"Expected output type: {operator.get('output_type', 'entity')}\n"
        f"Expected cardinality: {contract.get('cardinality', 'single')}\n"
        f"Expected granularity: {contract.get('granularity', 'minimal_answer_span')}\n"
        f"Candidate answer: {candidate_answer}\n"
        f"Candidate support quote: {candidate_quote}\n"
        f"Source title: {source_title}\n"
        f"Source passage:\n{source_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_json_object(value: Any) -> dict[str, Any]:
    """Parse an LLM JSON object, tolerating a fenced response."""

    if isinstance(value, dict):
        return dict(value)
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
        return dict(payload) if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}


def _type_compatible(answer: str, expected_type: str) -> bool:
    normalized = normalize_surface(answer)
    if not normalized:
        return False
    expected = str(expected_type or "entity")
    if expected == "boolean":
        return normalized in BOOLEAN_VALUES
    if expected == "number":
        return bool(NUMBER_RE.search(answer))
    if expected == "date":
        return bool(YEAR_RE.search(answer))
    return True


def _sentence_id(source_text: str, quote: str) -> int:
    quote_norm = normalize_surface(quote)
    for index, sentence in enumerate(re.split(r"(?<=[.!?])\s+", str(source_text or ""))):
        sentence_norm = normalize_surface(sentence)
        if quote_norm and (quote_norm in sentence_norm or sentence_norm in quote_norm):
            return index
    return -1


def _source_sentences(source_text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", str(source_text or ""))
        if sentence.strip()
    ]


def _recover_exact_source_quote(source_text: str, answer: str) -> str:
    """Find the shortest real source sentence containing the proposed answer."""

    answer_norm = normalize_surface(answer)
    candidates = [
        sentence
        for sentence in _source_sentences(source_text)
        if answer_norm and answer_norm in normalize_surface(sentence)
    ]
    return min(candidates, key=len) if candidates else ""


def _identity_tokens(value: str) -> set[str]:
    return {
        token for token in normalize_surface(value).split()
        if len(token) > 1 and token not in IDENTITY_STOPWORDS
    }


def _binding_matches_surface(binding: str, surface: str) -> bool:
    binding_norm = normalize_surface(binding)
    surface_norm = normalize_surface(surface)
    if not binding_norm:
        return True
    if binding_norm in surface_norm:
        return True
    binding_tokens = _identity_tokens(binding)
    surface_tokens = _identity_tokens(surface)
    return bool(binding_tokens) and len(binding_tokens & surface_tokens) / len(binding_tokens) >= 0.6


def validate_supported_extraction(
    extraction: Mapping[str, Any],
    operator: Mapping[str, Any],
    *,
    source_title: str,
    source_text: str,
) -> dict[str, Any]:
    """Enforce the tuple admission contract against the source passage."""

    answer = " ".join(str(extraction.get("answer") or "").split())
    quote = " ".join(str(extraction.get("support_quote") or "").split())
    source_norm = normalize_surface(source_text)
    quote_norm = normalize_surface(quote)
    answer_norm = normalize_surface(answer)
    quote_in_source = bool(quote_norm and quote_norm in source_norm)
    quote_recovered = False
    if not quote_in_source and answer_norm and answer_norm in source_norm:
        recovered_quote = _recover_exact_source_quote(source_text, answer)
        if recovered_quote:
            quote = recovered_quote
            quote_norm = normalize_surface(quote)
            quote_in_source = True
            quote_recovered = True
    answer_in_quote = bool(answer_norm and quote_norm and answer_norm in quote_norm)
    bindings = [
        normalize_surface(value)
        for value in operator.get("input_bindings") or []
        if normalize_surface(value)
    ]
    title_norm = normalize_surface(source_title)
    binding_surface = f"{source_title} {quote}"
    binding_grounded = not bindings or all(
        _binding_matches_surface(binding, binding_surface) for binding in bindings
    )
    type_compatible = _type_compatible(answer, str(operator.get("output_type") or "entity"))
    model_supported = extraction.get("supported") is True
    accepted = bool(
        model_supported
        and quote_in_source
        and answer_in_quote
        and binding_grounded
        and type_compatible
    )
    rejection_reasons = []
    for condition, reason in (
        (model_supported, "model_declared_unsupported"),
        (quote_in_source, "quote_not_in_source"),
        (answer_in_quote, "answer_not_in_quote"),
        (binding_grounded, "bound_value_not_grounded"),
        (type_compatible, "answer_type_mismatch"),
    ):
        if not condition:
            rejection_reasons.append(reason)
    return {
        "accepted": accepted,
        "answer": answer,
        "support_quote": quote,
        "quote_in_source": quote_in_source,
        "quote_recovered_from_source": quote_recovered,
        "answer_in_quote": answer_in_quote,
        "binding_grounded": binding_grounded,
        "type_compatible": type_compatible,
        "source_sentence_id": _sentence_id(source_text, quote) if quote_in_source else -1,
        "rejection_reasons": rejection_reasons,
    }


def validate_relation_verification(
    verification: Mapping[str, Any],
    operator: Mapping[str, Any],
    *,
    source_title: str,
    source_text: str,
    original_answer: str,
) -> dict[str, Any]:
    """Combine semantic judgments with deterministic provenance admission."""

    verified_answer = " ".join(str(verification.get("verified_answer") or "").split())
    support_quote = " ".join(str(verification.get("support_quote") or "").split())
    deterministic = validate_supported_extraction(
        {
            "supported": True,
            "answer": verified_answer,
            "support_quote": support_quote,
        },
        operator,
        source_title=source_title,
        source_text=source_text,
    )
    relation_entailed = verification.get("relation_entailed") is True
    binding_consistent = verification.get("binding_consistent") is True
    slot_fit = verification.get("slot_fit") is True
    cardinality_fit = verification.get("cardinality_fit") is True
    bindings = list(operator.get("input_bindings") or [])
    semantic_binding_valid = binding_consistent if bindings else True
    provenance_valid = bool(
        deterministic.get("quote_in_source")
        and deterministic.get("answer_in_quote")
        and deterministic.get("type_compatible")
    )
    accepted = bool(
        relation_entailed
        and semantic_binding_valid
        and slot_fit
        and cardinality_fit
        and provenance_valid
    )
    rejection_reasons = []
    for condition, reason in (
        (relation_entailed, "relation_not_entailed"),
        (semantic_binding_valid, "binding_inconsistent"),
        (slot_fit, "output_slot_mismatch"),
        (cardinality_fit, "cardinality_mismatch"),
        (bool(deterministic.get("quote_in_source")), "verified_quote_not_in_source"),
        (bool(deterministic.get("answer_in_quote")), "verified_answer_not_in_quote"),
        (bool(deterministic.get("type_compatible")), "verified_answer_type_mismatch"),
    ):
        if not condition:
            rejection_reasons.append(reason)
    return {
        "accepted": accepted,
        "verified_answer": verified_answer,
        "support_quote": deterministic.get("support_quote", support_quote),
        "relation_entailed": relation_entailed,
        "binding_consistent": binding_consistent,
        "slot_fit": slot_fit,
        "cardinality_fit": cardinality_fit,
        "provenance_valid": provenance_valid,
        "answer_corrected": normalize_surface(verified_answer) != normalize_surface(original_answer),
        "source_sentence_id": deterministic.get("source_sentence_id", -1),
        "reason_code": str(verification.get("reason_code") or ""),
        "rejection_reasons": rejection_reasons,
        "deterministic_validation": deterministic,
    }


def materialize_query_local_tuple(
    operator: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    source_id: Any,
    source_title: str,
) -> dict[str, Any] | None:
    """Create a recovered, supported tuple only after source validation."""

    if not validation.get("accepted"):
        return None
    bindings = list(operator.get("input_bindings") or [])
    subject = str(bindings[-1] if bindings else source_title).strip()
    relation = str(operator.get("relation_spec") or "").strip()
    answer = str(validation.get("answer") or "").strip()
    fingerprint = "|".join((str(source_id), subject, relation, answer))
    fact_id = "late_bound:" + hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:16]
    return {
        "fact_id": fact_id,
        "triple": [subject, relation, answer],
        "source_id": source_id,
        "source_title": source_title,
        "source_sentence_id": int(validation.get("source_sentence_id", -1)),
        "source_sentence_span": validation.get("support_quote", ""),
        "origin": "query_local_overlay",
        "tuple_state": "RECOVERED",
        "support_status": "SUPPORTED",
        "operator_id": operator.get("operator_id"),
        "late_bound_textual_relation": True,
        "provenance_scope": "exact_source_quote",
        "answer_type": operator.get("output_type", "entity"),
    }
