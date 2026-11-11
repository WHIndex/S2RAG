import math
import re
import hashlib
import time
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .relation_fact_join_utils import (
    RELATION_POLARITY_TERMS,
    _answerability_gap_profile,
    _bounded_overlap,
    _clean_triple,
    _clamp,
    _entity_key,
    _entity_type,
    _object_matches_answer_type,
    _relation_family_score,
    _relation_profile,
    _schema_signature,
    _tokens,
)
from .planned_retrieval_state_utils import TERMINAL_EVIDENCE_GAP_REASONS


_PREDICATE_CONCEPTS = {
    "birth": {"birth", "born", "birthplace"},
    "death": {"death", "died"},
    "father": {"father", "paternal"},
    "mother": {"maternal", "mother"},
    "parent": {"parent", "parents"},
    "child": {"child", "children", "daughter", "son"},
    "grandparent": {"grandfather", "grandmother", "grandparent"},
    "sibling": {"brother", "sibling", "sister"},
    "spouse": {"husband", "spouse", "wife"},
    "uncle": {"uncle"},
    "aunt": {"aunt"},
    "authorship": {"author", "authored", "wrote", "writer"},
    "composition": {"composed", "composer", "music"},
    "direction": {"directed", "director"},
    "foundation": {"established", "founded", "founder", "founding"},
    "release": {"came", "out", "release", "released", "publication", "published"},
    "production": {"produced", "producer"},
    "visual_creation": {"artist", "designed", "designer", "painted", "painter"},
    "location": {
        "based", "capital", "city", "country", "headquartered", "located",
        "location", "place", "region", "state",
    },
    "located_between": {"between"},
    "border": {"border", "bordered", "bordering", "borders"},
    "confluence": {"confluence", "meet", "meets"},
    "membership": {"affiliated", "club", "member", "part", "played", "team"},
    "occupation": {"career", "job", "occupation", "profession", "worked"},
}

_TERMINAL_PREDICATE_CONCEPTS = {
    "birth",
    "death",
    "foundation",
    "location",
    "located_between",
    "border",
    "confluence",
    "release",
}
_NON_DISCRIMINATIVE_PREDICATE_TERMS = {
    "area",
    "city",
    "country",
    "location",
    "place",
    "region",
    "state",
    "town",
}
_META_DEFICIT_RE = re.compile(
    r"\b(?:evidence|documents?|passages?|context)\b.{0,80}"
    r"\b(?:does not|do not|cannot|fails? to|missing|insufficient|lack(?:s|ing)?)\b|"
    r"\b(?:does not|do not|cannot|fails? to)\b.{0,50}"
    r"\b(?:provide|specify|confirm|identify|establish|show|state)\b",
    flags=re.IGNORECASE,
)
_SLOT_META_TERMS = {
    "answer",
    "context",
    "document",
    "documents",
    "evidence",
    "information",
    "missing",
    "passage",
    "passages",
    "provide",
    "provided",
    "question",
    "retrieved",
}
_RESIDUAL_DEFICIT_TRIGGER_REASONS = set(TERMINAL_EVIDENCE_GAP_REASONS) - {
    "no_new_planned_queries",
    "residual_deficit_query_appended",
}

_RELATION_SIGNATURE_TERMS = {
    "located_between": {"between"},
    "border_with": {"border", "bordered", "bordering", "borders"},
    "adjacent_to": {"adjacent", "neighbor", "neighboring", "neighbour", "neighbouring"},
    "confluence_at": {"confluence", "meet", "meets"},
    "release_date": {"came", "out", "release", "released", "publication", "published"},
    "foundation_date": {"established", "founded", "founding"},
    "birth": {"birth", "birthplace", "born"},
    "death": {"death", "died"},
    "capital_of": {"capital"},
    "headquarters_location": {"based", "headquartered", "headquarters"},
    "formation_location": {"formed", "formation", "originated"},
    "located_in": {"located", "location"},
    "contains": {"comprises", "contains", "includes"},
    "produced_by": {"produced", "producer"},
    "directed_by": {"directed", "director"},
    "authored_by": {"author", "authored", "writer", "wrote"},
    "composed_by": {"composed", "composer"},
    "played_role": {"acted", "actor", "played", "portrayed", "role"},
    "parent_of": {"parent", "parents"},
    "father_of": {"father", "paternal"},
    "mother_of": {"maternal", "mother"},
    "grandparent_of": {"grandfather", "grandmother", "grandparent"},
    "membership": {"affiliated", "club", "member", "part", "played", "team"},
}
_RELATION_OPERATOR_TERMS = set().union(*_RELATION_SIGNATURE_TERMS.values())
_MULTI_KEY_SIGNATURES = {
    "adjacent_to",
    "border_with",
    "confluence_at",
    "located_between",
}
_GENERIC_RELATION_SIGNATURES = {"contains", "located_in", "membership"}
_PLANNER_SUFFICIENT_REASONS = {"initially_sufficient", "verified_sufficient"}
_OPERATOR_EXHAUSTED_REASONS = (
    _RESIDUAL_DEFICIT_TRIGGER_REASONS | {"no_new_planned_queries"}
)
_SIGNATURE_COMPATIBILITY_FAMILIES = {
    "birth": {"birth", "birth_date", "birth_place"},
    "death": {"death", "death_date", "death_place"},
    "foundation": {"foundation", "foundation_date"},
    "formation": {"formation", "formation_location"},
    "parent": {"father_of", "mother_of", "parent_of"},
}
_SIGNATURE_EXPECTED_TYPES = {
    "birth_date": "date",
    "birth_place": "place",
    "death_date": "date",
    "death_place": "place",
    "foundation_date": "date",
    "formation_location": "place",
}
_DERIVED_SLOT_RE = re.compile(
    r"\b(?:"
    r"historical significance|what else|known for|reason|reasons|why|"
    r"how long|duration|difference|differences|impact|effect|effects|"
    r"details?|context|information about|number of|how many|"
    r"caused?|forced?|constituted?"
    r")\b",
    flags=re.IGNORECASE,
)
_COMPARISON_QUERY_RE = re.compile(
    r"\b(?:earlier|later|first|more recently|most recently|older|younger|born first)\b",
    flags=re.IGNORECASE,
)
_ARGMAX_QUERY_RE = re.compile(
    r"\b(?:highest|largest|longest|most|newest|oldest|tallest)\b",
    flags=re.IGNORECASE,
)
_ARGMIN_QUERY_RE = re.compile(
    r"\b(?:fewest|lowest|shortest|smallest|least|earliest)\b",
    flags=re.IGNORECASE,
)
_SPATIAL_BETWEEN_RE = re.compile(
    r"\b(?:border|boundary|countries|country|lies?|located|locations?|places?|regions?|situated|states?|territory)\b.{0,45}\bbetween\b|"
    r"\bbetween\b.{0,45}\b(?:border|boundary|countries|locations?|places?|regions?|states?)\b",
    flags=re.IGNORECASE,
)
_VALIDATION_DATE_RE = re.compile(
    r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b|"
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b",
    flags=re.IGNORECASE,
)
_VALIDATION_NUMBER_RE = re.compile(r"\b\d+(?:[,.]\d+)*(?:\.\d+)?\b")
_VALIDATION_PROPER_SPAN_RE = re.compile(
    r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'.-]+"
    r"(?:\s+(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'.-]+|of|the|and)){0,5}\b"
)
_VALIDATION_VALUE_CUE_RE = re.compile(
    r"\b(?:as|became|is|was|were)\s+(?:an?\s+|the\s+)?[a-z][a-z'-]{2,}\b",
    flags=re.IGNORECASE,
)
_VALIDATION_ORG_RE = re.compile(
    r"\b(?:association|club|college|company|corporation|foundation|group|"
    r"institute|organization|school|team|university)\b",
    flags=re.IGNORECASE,
)
_VALIDATION_WORK_RE = re.compile(
    r"\b(?:album|book|film|movie|novel|play|series|song|work)\b|[\"“”]",
    flags=re.IGNORECASE,
)
_VALIDATION_NATIONALITY_RE = re.compile(
    r"\b(?:american|australian|british|canadian|chinese|dutch|english|"
    r"french|german|greek|indian|irish|italian|japanese|mexican|polish|"
    r"russian|scottish|spanish|swedish|turkish|welsh)\b",
    flags=re.IGNORECASE,
)

_ANSWER_TYPE_FAMILIES = {
    "boolean": {"boolean", "bool", "yes_no"},
    "date": {"date", "day", "month", "time", "year"},
    "entity": {"entity", "organization", "person", "work"},
    "number": {"count", "number", "quantity"},
    "place": {"city", "country", "location", "place", "region", "state"},
}
_GENERIC_TITLE_SUFFIXES = {
    "biography",
    "career",
    "film",
    "history",
    "life",
    "novel",
    "profile",
    "season",
    "series",
    "song",
}
_SLOT_NAMED_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9À-ÖØ-öø-ÿ'.-]*)"
    r"(?:\s+(?:(?:of|the|and|in|on|for)\s+)?"
    r"[A-Z][A-Za-z0-9À-ÖØ-öø-ÿ'.-]*){0,5}\b"
)
_SLOT_QUOTED_PHRASE_RE = re.compile(r"[\"“]([^\"”]{2,100})[\"”]")


def _canonical_answer_type(value: str) -> str:
    normalized = str(value or "unknown").strip().lower()
    for family, members in _ANSWER_TYPE_FAMILIES.items():
        if normalized in members:
            return family
    return "unknown" if normalized in {"", "unknown"} else normalized


def _question_final_answer_type(question: str) -> str:
    normalized = " ".join(str(question or "").lower().split())
    if normalized.startswith("when") or re.search(
        r"\b(?:what|which)\s+(?:(?:is|was)\s+)?(?:the\s+)?"
        r"(?:date|day|month|year)\b",
        normalized,
    ):
        return "date"
    if normalized.startswith("where") or re.search(
        r"\b(?:what|which)\s+(?:(?:is|was)\s+)?(?:the\s+)?"
        r"(?:city|country|location|place|region|state)\b",
        normalized,
    ):
        return "place"
    if normalized.startswith("who"):
        return "entity"
    if re.search(r"\b(?:how many|how much|what number|which number)\b", normalized) or (
        _ARGMAX_QUERY_RE.search(normalized)
        and re.search(r"\b(?:average|count|score|streak|total|value)\b", normalized)
    ):
        return "number"
    if re.search(r"\bwhich\s+(?:album|book|film|movie|novel|song|work)\b", normalized):
        return "entity"
    if re.match(r"^(?:are|can|did|do|does|has|have|is|was|were)\b", normalized):
        return "boolean"
    return "entity"


def _normalized_surface(value: str) -> str:
    normalized = re.sub(r"\s*\([^)]*\)\s*$", "", str(value or "").strip())
    normalized = re.sub(r"[^A-Za-z0-9À-ÖØ-öø-ÿ]+", " ", normalized.lower())
    return " ".join(normalized.split())


def _accent_folded_surface(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    ascii_like = "".join(
        character for character in decomposed
        if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^\w]+", " ", ascii_like.lower()).split())


def _recover_answer_surface(source_text: str, answer_value: str) -> str:
    """Recover the source spelling of an OpenIE value, including diacritics."""
    source = str(source_text or "")
    target = _accent_folded_surface(answer_value)
    if not source or not target:
        return str(answer_value or "").strip()
    token_matches = list(re.finditer(r"[^\W_]+(?:['’-][^\W_]+)*", source, re.UNICODE))
    target_compact = target.replace(" ", "")
    max_window = min(8, max(1, len(target.split()) + 2))
    for window in range(1, max_window + 1):
        for start in range(0, len(token_matches) - window + 1):
            selected = token_matches[start:start + window]
            surface = source[selected[0].start():selected[-1].end()]
            folded = _accent_folded_surface(surface)
            if folded == target or folded.replace(" ", "") == target_compact:
                return surface.strip()
            folded_compact = folded.replace(" ", "")
            if (
                min(len(folded_compact), len(target_compact)) >= 5
                and abs(len(folded_compact) - len(target_compact)) <= 2
                and SequenceMatcher(
                    None,
                    folded_compact,
                    target_compact,
                ).ratio() >= 0.86
            ):
                return surface.strip()
    return str(answer_value or "").strip()


def _structured_branch_binding(
    active_entity: str,
    source_title: str,
    source_text: str,
    slot: Dict[str, Any],
) -> Dict[str, Any]:
    """Bind a structured fact row to the residual slot's named entity branch."""
    active_surface = _accent_folded_surface(active_entity)
    title_surface = _accent_folded_surface(source_title)
    text_surface = _accent_folded_surface(source_text)
    phrases = list(slot.get("named_anchor_phrases") or [])
    phrase_surfaces = [
        _accent_folded_surface(phrase) for phrase in phrases
        if _accent_folded_surface(phrase)
    ]
    if phrase_surfaces:
        for phrase, phrase_surface in zip(phrases, phrase_surfaces):
            exact_entity = active_surface == phrase_surface
            entity_prefix = bool(
                active_surface.startswith(phrase_surface + " ")
                or phrase_surface.startswith(active_surface + " ")
            )
            if exact_entity or entity_prefix:
                source_bound = bool(
                    title_surface == active_surface
                    or title_surface.startswith(active_surface + " ")
                    or text_surface.startswith(active_surface + " ")
                    or (" " + active_surface + " ") in (" " + text_surface + " ")
                )
                return {
                    "passed": source_bound,
                    "reason": (
                        "named_entity_branch_bound"
                        if source_bound else "named_entity_missing_from_source"
                    ),
                    "anchor_phrase": str(phrase),
                }
        return {
            "passed": False,
            "reason": "named_entity_branch_mismatch",
            "anchor_phrase": "",
        }

    anchor_tokens = set(slot.get("anchor_tokens") or set())
    active_tokens = _tokens(active_entity)
    required = max(1, math.ceil(0.67 * len(active_tokens)))
    anchor_bound = bool(active_tokens and len(active_tokens & anchor_tokens) >= required)
    source_bound = bool(
        active_surface
        and (
            title_surface == active_surface
            or title_surface.startswith(active_surface + " ")
            or active_surface in text_surface
        )
    )
    return {
        "passed": bool(anchor_bound and source_bound),
        "reason": (
            "token_entity_branch_bound"
            if anchor_bound and source_bound else "token_entity_branch_unbound"
        ),
        "anchor_phrase": "",
    }


def _slot_named_anchor_phrases(text: str) -> List[str]:
    phrases = list(_SLOT_QUOTED_PHRASE_RE.findall(str(text or "")))
    phrases.extend(_SLOT_NAMED_PHRASE_RE.findall(str(text or "")))
    expanded = []
    for phrase in phrases:
        expanded.append(phrase)
        expanded.extend(
            value.strip()
            for value in re.split(r"\s+(?:by|for|in|on)\s+", phrase)
            if value.strip() and value.strip() != phrase
        )
    blocked = {"what", "when", "where", "which", "who", "the"}
    result = []
    seen = set()
    for phrase in expanded:
        normalized = _normalized_surface(phrase)
        if not normalized or normalized in blocked or normalized in seen:
            continue
        seen.add(normalized)
        result.append(str(phrase).strip())
    return result[:6]


def _dense_branch_binding(
    title: str,
    grounding_sentence: str,
    slot: Dict[str, Any],
) -> Dict[str, Any]:
    """Require dense support to preserve the residual slot's entity identity."""
    phrases = list(slot.get("named_anchor_phrases") or [])
    title_surface = _normalized_surface(title)
    sentence_surface = _normalized_surface(grounding_sentence)
    for phrase in phrases:
        phrase_surface = _normalized_surface(phrase)
        if not phrase_surface:
            continue
        title_tokens = title_surface.split()
        phrase_tokens = phrase_surface.split()
        exact_title = title_surface == phrase_surface
        generic_suffix = bool(
            len(title_tokens) > len(phrase_tokens)
            and title_tokens[: len(phrase_tokens)] == phrase_tokens
            and set(title_tokens[len(phrase_tokens):]) <= _GENERIC_TITLE_SUFFIXES
        )
        sentence_subject = bool(
            sentence_surface == phrase_surface
            or sentence_surface.startswith(phrase_surface + " ")
        )
        if exact_title or generic_suffix or sentence_subject:
            return {
                "passed": True,
                "reason": "named_anchor_bound",
                "anchor_phrase": phrase,
            }

    if phrases:
        return {
            "passed": False,
            "reason": "named_anchor_unbound",
            "anchor_phrase": "",
        }

    anchor_tokens = set(slot.get("anchor_tokens") or set())
    title_tokens = _tokens(title)
    required = max(1, math.ceil(0.67 * len(anchor_tokens)))
    passed = bool(anchor_tokens and len(anchor_tokens & title_tokens) >= required)
    return {
        "passed": passed,
        "reason": "title_anchor_bound" if passed else "title_anchor_unbound",
        "anchor_phrase": "",
    }


def _annotate_slot_plan_stages(
    question: str,
    slot_contracts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    final_type = _question_final_answer_type(question)
    comparison = bool(_COMPARISON_QUERY_RE.search(str(question or "")))
    annotated = []
    for slot in slot_contracts:
        expected_type = _canonical_answer_type(
            str(slot.get("expected_value_type") or "unknown")
        )
        if comparison:
            stage = "operator_input"
        elif expected_type == final_type or (
            final_type == "entity" and expected_type == "entity"
        ):
            stage = "final_answer"
        elif expected_type != "unknown":
            stage = "intermediate_binding"
        elif len(slot_contracts) == 1:
            stage = "final_answer"
        else:
            stage = "support_only"
        phrases = _slot_named_anchor_phrases(str(slot.get("text") or ""))
        branch_surface = "|".join(
            _normalized_surface(value) for value in phrases if value
        ) or "|".join(sorted(slot.get("anchor_tokens") or []))
        branch_id = hashlib.sha256(branch_surface.encode("utf-8")).hexdigest()[:12]
        annotated.append({
            **slot,
            "final_answer_type": final_type,
            "slot_expected_type_family": expected_type,
            "slot_plan_stage": stage,
            "slot_branch_id": branch_id,
            "named_anchor_phrases": phrases,
        })
    return annotated


def _surface_contains_value(text: str, value: str) -> bool:
    normalized_text = " ".join(str(text or "").lower().split())
    normalized_value = " ".join(str(value or "").lower().split())
    if not normalized_text or not normalized_value:
        return False
    if normalized_value in normalized_text:
        return True
    value_tokens = _tokens(normalized_value)
    return bool(value_tokens) and value_tokens <= _tokens(normalized_text)


def _answer_type_surface_match(
    text: str,
    expected_types: Set[str],
    anchor_tokens: Set[str],
) -> Tuple[bool, str]:
    """Check that a grounding sentence exposes a plausible residual value."""
    expected = {str(value or "").lower() for value in expected_types if value}
    if not expected or expected <= {"entity", "unknown"}:
        expected = {"entity"}
    checks = []
    if "date" in expected:
        checks.append((bool(_VALIDATION_DATE_RE.search(text)), "date"))
    if "number" in expected:
        checks.append((bool(_VALIDATION_NUMBER_RE.search(text)), "number"))
    if "nationality" in expected:
        checks.append((bool(_VALIDATION_NATIONALITY_RE.search(text)), "nationality"))
    if "organization" in expected:
        checks.append((bool(_VALIDATION_ORG_RE.search(text)), "organization"))
    if "work" in expected:
        checks.append((bool(_VALIDATION_WORK_RE.search(text)), "work"))
    if expected & {"entity", "person", "place", "location"}:
        spans = _VALIDATION_PROPER_SPAN_RE.findall(text)
        has_unbound_span = any(
            bool(_tokens(span) - anchor_tokens)
            for span in spans
        )
        checks.append((
            bool(has_unbound_span or _VALIDATION_VALUE_CUE_RE.search(text)),
            "entity_surface",
        ))
    if not checks:
        return True, "unconstrained"
    matched = [name for passed, name in checks if passed]
    return bool(matched), matched[0] if matched else "type_surface_missing"


def _grounding_sentences(text: str) -> List[str]:
    values = [
        value.strip()
        for value in re.split(r"(?<=[.!?])\s+|[\r\n]+", str(text or ""))
        if value.strip()
    ]
    return values[:96]


def _validate_dense_operator_candidate(
    content: str,
    slot_contracts: List[Dict[str, Any]],
    *,
    require_branch_binding: bool = False,
) -> Dict[str, Any]:
    """Validate dense support against one residual slot in one sentence."""
    best: Dict[str, Any] = {
        "passed": False,
        "score": 0.0,
        "reason": "no_grounded_residual_slot",
        "slot_id": "",
        "slot_text": "",
        "anchor_overlap": 0.0,
        "predicate_overlap": 0.0,
        "answer_type_surface": "",
        "grounding_sentence": "",
        "branch_binding_passed": False,
        "branch_binding_reason": "not_evaluated",
        "branch_anchor_phrase": "",
    }
    for slot in slot_contracts:
        if bool(slot.get("meta_negative")):
            continue
        anchor_tokens = set(slot.get("anchor_tokens") or set())
        expected_types = set(slot.get("answer_types") or set())
        for sentence in _grounding_sentences(content):
            sentence_tokens = _tokens(sentence)
            anchor_hits = len(anchor_tokens & sentence_tokens)
            anchor_required = min(2, len(anchor_tokens))
            anchor_passed = bool(anchor_tokens) and anchor_hits >= max(1, anchor_required)
            anchor_overlap = (
                anchor_hits / float(max(1, min(3, len(anchor_tokens))))
                if anchor_tokens else 0.0
            )
            predicate = _predicate_contract(
                sentence_tokens,
                slot,
                "subject_to_object",
                strict_signature=False,
            )
            type_passed, type_surface = _answer_type_surface_match(
                sentence,
                expected_types,
                anchor_tokens,
            )
            branch_binding = _dense_branch_binding(
                str(content or "").split("\n", 1)[0].strip(),
                sentence,
                slot,
            )
            score = _clamp(
                0.42 * min(1.0, anchor_overlap)
                + 0.38 * float(predicate["score"])
                + 0.15 * (1.0 if type_passed else 0.0)
                + 0.05
            )
            passed = bool(
                anchor_passed
                and predicate["matched"]
                and type_passed
                and (
                    not require_branch_binding
                    or bool(branch_binding["passed"])
                )
                and score >= 0.68
            )
            candidate = {
                "passed": passed,
                "score": round(float(score), 6),
                "reason": "validated_residual_surface" if passed else (
                    "anchor_missing" if not anchor_passed
                    else "predicate_missing" if not predicate["matched"]
                    else "answer_type_surface_missing" if not type_passed
                    else "branch_anchor_unbound" if not branch_binding["passed"]
                    else "validation_score_low"
                ),
                "slot_id": str(slot.get("slot_id") or ""),
                "slot_text": str(slot.get("text") or ""),
                "slot_source": str(slot.get("source") or "unknown"),
                "slot_answer_types": sorted(expected_types),
                "slot_anchor_tokens": sorted(anchor_tokens),
                "slot_relation_tokens": sorted(slot.get("relation_tokens") or []),
                "slot_materialization_safe": bool(
                    slot.get("materialization_safe")
                    or slot.get("join_executable")
                    or slot.get("execution_operator") == "text_retrieval"
                ),
                "slot_plan_stage": str(
                    slot.get("slot_plan_stage") or "unclassified"
                ),
                "slot_branch_id": str(slot.get("slot_branch_id") or ""),
                "slot_final_answer_type": str(
                    slot.get("final_answer_type") or "unknown"
                ),
                "slot_expected_type_family": str(
                    slot.get("slot_expected_type_family") or "unknown"
                ),
                "branch_binding_passed": bool(branch_binding["passed"]),
                "branch_binding_reason": str(branch_binding["reason"]),
                "branch_anchor_phrase": str(
                    branch_binding.get("anchor_phrase") or ""
                ),
                "anchor_overlap": round(float(anchor_overlap), 6),
                "predicate_overlap": round(float(predicate["score"]), 6),
                "predicate_match_mode": str(predicate["mode"]),
                "answer_type_surface": type_surface,
                "grounding_sentence": sentence[:500],
            }
            if (bool(candidate["passed"]), float(candidate["score"])) > (
                bool(best["passed"]), float(best["score"])
            ):
                best = candidate
    return best


def _validate_structured_operator_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
    expected_signature = str(item.get("expected_relation_signature") or "")
    checks = {
        "entity_binding": bool(item.get("entity_bound") and item.get("join_key_exact")),
        "predicate": bool(item.get("predicate_contract_match")),
        "signature": bool(
            item.get("relation_signature_match") if expected_signature else True
        ),
        "answer_type": bool(item.get("answer_type_match", True)),
        "dependency": bool(item.get("dependency_binding_safe", True)),
        "answer_surface": bool(item.get("answer_surface_grounded")),
        "slot_safe": bool(item.get("slot_materialization_safe")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    score = sum(checks.values()) / float(len(checks))
    return {
        "passed": not failed,
        "score": round(float(score), 6),
        "reason": "validated_structured_fact" if not failed else "failed:" + ",".join(failed),
        "checks": checks,
    }


def _signature_family(signature: str) -> str:
    value = str(signature or "")
    for family, members in _SIGNATURE_COMPATIBILITY_FAMILIES.items():
        if value in members:
            return family
    return value


def _relation_signature_compatibility(
    expected_signature: str,
    candidate_signature: str,
    *,
    candidate_answer_type: str = "",
    expected_answer_types: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Classify a signature pair without collapsing granularity into conflict."""
    expected = str(expected_signature or "")
    candidate = str(candidate_signature or "")
    if expected and candidate and expected == candidate:
        return {"tier": "exact", "compatible": True, "score": 1.0}
    if not expected or not candidate:
        return {"tier": "unknown", "compatible": False, "score": 0.0}
    if _signature_family(expected) != _signature_family(candidate):
        return {"tier": "conflict", "compatible": False, "score": 0.0}

    expected_types = set(expected_answer_types or set())
    required_type = _SIGNATURE_EXPECTED_TYPES.get(expected, "")
    if required_type and expected_types and required_type not in expected_types:
        return {"tier": "conflict", "compatible": False, "score": 0.0}
    if required_type and candidate_answer_type not in {"", required_type, "entity"}:
        return {"tier": "conflict", "compatible": False, "score": 0.0}



    if candidate in {"birth", "death", "foundation", "formation"}:
        return {"tier": "compatible", "compatible": True, "score": 0.94}
    if expected in {"birth", "death", "foundation", "formation", "parent_of"}:
        return {"tier": "compatible", "compatible": True, "score": 0.92}
    return {"tier": "conflict", "compatible": False, "score": 0.0}


def remove_entity_predicate_passages(
    passages: List[str],
    titles: List[str],
    fingerprints: List[str],
) -> Tuple[List[str], List[str]]:
    """Remove only admitted fact-join passages from a finalized reader slate."""
    remaining = []
    removed = []
    title_budget = Counter(
        " ".join(str(title or "").lower().split()) for title in titles
    )
    fingerprint_budget = Counter(str(value or "") for value in fingerprints)
    for passage in passages or []:
        text = str(passage or "")
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        title = " ".join(text.split("\n", 1)[0].strip().lower().split())
        remove = False
        if fingerprint_budget.get(fingerprint, 0) > 0:
            fingerprint_budget[fingerprint] -= 1
            remove = True
        elif title_budget.get(title, 0) > 0:
            title_budget[title] -= 1
            remove = True
        if remove:
            removed.append(text.split("\n", 1)[0].strip())
        else:
            remaining.append(passage)
    return remaining, removed


def _iter_planned_slot_candidates(
    question: str,
    planned_retrieval: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return deficit probes in planner priority order with their provenance."""
    planned = planned_retrieval or {}
    values: List[Dict[str, Any]] = []

    def add_many(raw_values: Any, source: str, hop_index: int = -1) -> None:
        for value in raw_values or []:
            values.append({
                "text": value,
                "source": source,
                "hop_index": int(hop_index),
            })

    add_many(planned.get("residual_deficit_queries"), "residual_deficit_query")
    add_many(
        planned.get("residual_deficit_replaced_queries"),
        "residual_deficit_replaced_query",
    )
    rounds = list(planned.get("rounds") or [])
    for round_index in reversed(range(len(rounds))):
        round_info = rounds[round_index] if isinstance(rounds[round_index], dict) else {}
        verifier = round_info.get("verifier") or {}
        if isinstance(verifier, dict):
            add_many(verifier.get("queries"), "verifier_query", round_index)
        add_many(round_info.get("verifier_queries"), "verifier_query", round_index)
        add_many(round_info.get("queries"), "planner_query", round_index)
        if isinstance(verifier, dict):
            add_many(verifier.get("missing_slots"), "verifier_missing_slot", round_index)
            if verifier.get("missing"):
                values.append({
                    "text": verifier.get("missing"),
                    "source": "verifier_missing_text",
                    "hop_index": round_index,
                })
    if not values:
        values.append({"text": question, "source": "original_query", "hop_index": -1})

    seen = set()
    result = []
    for value in values:
        text = " ".join(str(value.get("text") or "").split()).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append({**value, "text": text[:320]})
        if len(result) >= 8:
            break
    return result


def _planned_slot_texts(question: str, planned_retrieval: Optional[Dict[str, Any]]) -> List[str]:
    return [
        str(item.get("text") or "")
        for item in _iter_planned_slot_candidates(question, planned_retrieval)
    ]


def _iter_residual_deficit_slot_candidates(
    planned_retrieval: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Read unresolved verifier slots before any free-form retrieval queries."""
    planned = planned_retrieval or {}
    values: List[Dict[str, Any]] = []
    rounds = list(planned.get("rounds") or [])
    for round_index in reversed(range(len(rounds))):
        round_info = rounds[round_index] if isinstance(rounds[round_index], dict) else {}
        verifier = round_info.get("verifier") or {}
        if not isinstance(verifier, dict):
            continue
        for value in verifier.get("missing_slots") or []:
            values.append({
                "text": value,
                "source": "verifier_missing_slot",
                "hop_index": int(round_index),
            })
        if len(values) >= 8:
            break

    seen = set()
    result = []
    for value in values:
        text = " ".join(str(value.get("text") or "").split()).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append({**value, "text": text[:320]})
        if len(result) >= 8:
            break
    return result


def _relation_signature(
    text: str,
    tokens: Set[str],
    answer_types: Optional[Set[str]] = None,
) -> str:
    """Compile a relation surface into an executable, arity-aware signature."""
    lowered = " ".join(str(text or "").lower().split())
    answer_types = set(answer_types or set())


    if tokens & _RELATION_SIGNATURE_TERMS["release_date"]:
        return "release_date"
    if tokens & _RELATION_SIGNATURE_TERMS["foundation_date"]:
        return "foundation_date" if "date" in answer_types else "foundation"
    if tokens & _RELATION_SIGNATURE_TERMS["birth"]:
        if "date" in answer_types or re.search(r"\b(?:date|day|year)\b", lowered):
            return "birth_date"
        if "place" in answer_types or re.search(r"\b(?:birthplace|born (?:at|in))\b", lowered):
            return "birth_place"
        return "birth"
    if tokens & _RELATION_SIGNATURE_TERMS["death"]:
        if "date" in answer_types or re.search(r"\b(?:date|day|year)\b", lowered):
            return "death_date"
        if "place" in answer_types or re.search(r"\b(?:death place|died (?:at|in))\b", lowered):
            return "death_place"
        return "death"
    if tokens & _RELATION_SIGNATURE_TERMS["border_with"]:
        return "border_with"
    if tokens & _RELATION_SIGNATURE_TERMS["adjacent_to"]:
        return "adjacent_to"
    if tokens & _RELATION_SIGNATURE_TERMS["confluence_at"]:
        return "confluence_at"
    if ("between" in tokens or " between " in f" {lowered} ") and _SPATIAL_BETWEEN_RE.search(lowered):
        return "located_between"
    if "capital" in tokens:
        return "capital_of"
    if tokens & _RELATION_SIGNATURE_TERMS["headquarters_location"]:
        return "headquarters_location"
    if tokens & _RELATION_SIGNATURE_TERMS["formation_location"]:
        if "date" in answer_types:
            return "foundation_date"
        if "place" not in answer_types and not re.search(r"\bformed (?:at|in)\b", lowered):
            return "formation"
        return "formation_location"
    if tokens & _RELATION_SIGNATURE_TERMS["produced_by"]:
        return "produced_by"
    if tokens & _RELATION_SIGNATURE_TERMS["directed_by"]:
        return "directed_by"
    if tokens & _RELATION_SIGNATURE_TERMS["authored_by"]:
        return "authored_by"
    if tokens & _RELATION_SIGNATURE_TERMS["composed_by"]:
        return "composed_by"
    if tokens & _RELATION_SIGNATURE_TERMS["father_of"]:
        return "father_of"
    if tokens & _RELATION_SIGNATURE_TERMS["mother_of"]:
        return "mother_of"
    if tokens & _RELATION_SIGNATURE_TERMS["grandparent_of"]:
        return "grandparent_of"
    if tokens & _RELATION_SIGNATURE_TERMS["parent_of"]:
        return "parent_of"
    if tokens & _RELATION_SIGNATURE_TERMS["played_role"] and tokens & {"acted", "actor", "portrayed", "role"}:
        return "played_role"
    if tokens & _RELATION_SIGNATURE_TERMS["contains"]:
        return "contains"
    if tokens & _RELATION_SIGNATURE_TERMS["membership"]:
        return "membership"
    if tokens & _RELATION_SIGNATURE_TERMS["located_in"]:
        return "located_in"
    if "place" in answer_types and tokens & _NON_DISCRIMINATIVE_PREDICATE_TERMS:
        return "located_in"
    return ""


def _slot_execution_operator(
    text: str,
    relation_signature: str,
    *,
    catalog_guided: bool = False,
) -> str:
    if _DERIVED_SLOT_RE.search(str(text or "")):
        return "unsafe_derived"
    if catalog_guided and _ARGMAX_QUERY_RE.search(str(text or "")):
        return "argmax"
    if catalog_guided and _ARGMIN_QUERY_RE.search(str(text or "")):
        return "argmin"
    if relation_signature in _MULTI_KEY_SIGNATURES:
        return "multi_key_join"
    if catalog_guided and not relation_signature:
        return "text_retrieval"
    return "lookup"


def _residual_query_plan_type(
    question: str,
    slot_contracts: List[Dict[str, Any]],
) -> str:
    operators = {
        str(slot.get("execution_operator") or "") for slot in slot_contracts
    }
    if "argmax" in operators:
        return "argmax"
    if "argmin" in operators:
        return "argmin"
    executable = [
        slot for slot in slot_contracts if bool(slot.get("join_executable"))
    ]
    if not executable:
        if "text_retrieval" in operators:
            return "text_retrieval"
        return "unsafe_derived"
    if _COMPARISON_QUERY_RE.search(str(question or "")) and any(
        str(slot.get("relation_signature") or "").endswith("_date")
        for slot in executable
    ):
        return "compare"
    if any(
        slot.get("execution_operator") == "multi_key_join"
        for slot in executable
    ):
        return "multi_key_join"
    return "lookup"


def _route_residual_executor(
    *,
    plan_type: str,
    slot_contracts: List[Dict[str, Any]],
    planned_reason: str,
    residual_deficit_trigger: bool,
    operator_faithful: bool = False,
    evidence_deficit: bool = False,
) -> Dict[str, str]:
    """Choose one physical operator from the unresolved evidence state."""
    if not residual_deficit_trigger:
        return {
            "state": "residual_routing_disabled",
            "operator": "skip",
            "reason": "residual_deficit_trigger_disabled",
        }
    if (
        operator_faithful
        and planned_reason in _PLANNER_SUFFICIENT_REASONS
        and not evidence_deficit
    ):
        return {
            "state": "support_verified_sufficient",
            "operator": "skip",
            "reason": "planner_verified_current_support",
        }
    eligible_reasons = (
        _OPERATOR_EXHAUSTED_REASONS
        if operator_faithful
        else _RESIDUAL_DEFICIT_TRIGGER_REASONS
    )
    if planned_reason not in eligible_reasons and not (
        operator_faithful and evidence_deficit
    ):
        return {
            "state": "support_not_exhausted",
            "operator": "skip",
            "reason": "planned_reason_not_eligible",
        }
    if not slot_contracts:
        return {
            "state": "missing_residual_contract",
            "operator": "skip",
            "reason": "no_verifier_missing_slot",
        }
    if plan_type in {"argmax", "argmin"}:
        return {
            "state": "aggregate_deficit",
            "operator": plan_type,
            "reason": "aggregate_requires_branch_collection",
        }
    if plan_type == "text_retrieval":
        return {
            "state": "untyped_residual_deficit",
            "operator": "text_retrieval",
            "reason": "no_catalog_predicate_signature",
        }
    if plan_type == "unsafe_derived":
        return {
            "state": "derived_residual_deficit",
            "operator": "text_retrieval",
            "reason": "derived_slot_not_fact_join_safe",
        }
    if plan_type == "compare":
        return {
            "state": "typed_comparison_deficit",
            "operator": "compare",
            "reason": "typed_branch_values_required",
        }
    if plan_type == "multi_key_join":
        return {
            "state": "typed_multi_key_deficit",
            "operator": "multi_key_fact_join",
            "reason": "typed_multi_key_slot_ready",
        }
    return {
        "state": "typed_lookup_deficit",
        "operator": "fact_lookup",
        "reason": "typed_lookup_slot_ready",
    }


def _target_predicate_concepts(text: str, tokens: Set[str]) -> Set[str]:
    concepts = _predicate_concepts(tokens)
    terminal = concepts & _TERMINAL_PREDICATE_CONCEPTS
    if terminal:
        return terminal
    if len(concepts) <= 1:
        return concepts

    lowered = str(text or "").lower()
    positions = {}
    for concept in concepts:
        positions[concept] = max(
            (lowered.rfind(term) for term in _PREDICATE_CONCEPTS[concept]),
            default=-1,
        )
    return {max(positions, key=positions.get)}


def _compile_slot_contracts(
    slot_candidates: List[Dict[str, Any]],
    *,
    typed_plan: bool = False,
    catalog_guided: bool = False,
) -> List[Dict[str, Any]]:
    contracts = []
    for index, item in enumerate(slot_candidates):
        text = str(item.get("text") or "")
        profile = _relation_profile(text)
        text_tokens = _tokens(text)
        answer_types = set(profile.get("answer_types") or [])
        relation_signature = (
            _relation_signature(text, text_tokens, answer_types)
            if typed_plan else ""
        )
        execution_operator = (
            _slot_execution_operator(
                text,
                relation_signature,
                catalog_guided=catalog_guided,
            )
            if typed_plan else "lookup"
        )
        concept_terms = set().union(*_PREDICATE_CONCEPTS.values())
        target_concepts = _target_predicate_concepts(text, text_tokens)
        target_terms = set().union(
            *(_PREDICATE_CONCEPTS[name] for name in target_concepts)
        ) if target_concepts else set()
        raw_relation_tokens = (
            set(profile.get("relation_tokens") or [])
            | (text_tokens & concept_terms)
        )
        relation_tokens = (
            (raw_relation_tokens & target_terms)
            if target_terms
            else raw_relation_tokens
        )
        if typed_plan:
            relation_tokens |= text_tokens & _RELATION_OPERATOR_TERMS
        anchor_tokens = (
            set(profile.get("anchor_tokens") or [])
            - concept_terms
            - _SLOT_META_TERMS
        )
        if typed_plan:
            anchor_tokens -= _RELATION_OPERATOR_TERMS
        meta_negative = bool(_META_DEFICIT_RE.search(text))
        unsafe_reasons = []
        if str(item.get("source") or "") == "original_query":
            unsafe_reasons.append("original_query_not_atomic_deficit")
        if meta_negative:
            unsafe_reasons.append("meta_negative_deficit_text")
        if len(text_tokens) > 20:
            unsafe_reasons.append("deficit_slot_too_long")
        if not anchor_tokens:
            unsafe_reasons.append("deficit_slot_missing_anchor")
        if not relation_tokens and not set(profile.get("polarity") or []):
            unsafe_reasons.append("deficit_slot_missing_predicate")
        if typed_plan and execution_operator == "unsafe_derived":
            unsafe_reasons.append("derived_slot_requires_non_join_operator")
        join_executable = bool(
            not meta_negative
            and anchor_tokens
            and (
                relation_signature
                if catalog_guided
                else relation_signature or relation_tokens or set(profile.get("polarity") or [])
            )
            and execution_operator in {"lookup", "multi_key_join"}
        )
        contracts.append({
            "slot_id": f"entity_predicate_slot_{index}",
            "text": text,
            "source": str(item.get("source") or "unknown"),
            "hop_index": int(item.get("hop_index", -1)),
            "anchor_tokens": anchor_tokens,
            "relation_tokens": relation_tokens,
            "relation_families": set(profile.get("relation_families") or []),
            "polarity": set(profile.get("polarity") or []),
            "answer_types": answer_types,
            "target_predicate_concepts": target_concepts,
            "expected_value_type": next(
                iter(sorted(profile.get("answer_types") or [])),
                "unknown",
            ),
            "direction": "bound_entity_to_value",
            "relation_signature": relation_signature,
            "relation_arity": 2 if relation_signature in _MULTI_KEY_SIGNATURES else 1,
            "execution_operator": execution_operator,
            "join_executable": join_executable if typed_plan else not unsafe_reasons,
            "materialization_safe": not unsafe_reasons,
            "unsafe_reasons": unsafe_reasons,
            "meta_negative": meta_negative,
        })
    return contracts


def _slot_contracts(
    question: str,
    planned_retrieval: Optional[Dict[str, Any]],
    *,
    typed_plan: bool = False,
    catalog_guided: bool = False,
) -> List[Dict[str, Any]]:
    return _compile_slot_contracts(
        _iter_planned_slot_candidates(question, planned_retrieval),
        typed_plan=typed_plan,
        catalog_guided=catalog_guided,
    )


def _residual_deficit_slot_contracts(
    planned_retrieval: Optional[Dict[str, Any]],
    *,
    typed_plan: bool = False,
    catalog_guided: bool = False,
) -> List[Dict[str, Any]]:
    return _compile_slot_contracts(
        _iter_residual_deficit_slot_candidates(planned_retrieval),
        typed_plan=typed_plan,
        catalog_guided=catalog_guided,
    )


def _operator_faithful_slot_contracts(
    question: str,
    planned_retrieval: Optional[Dict[str, Any]],
    residual_contracts: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str]:
    """Backfill executable planner slots when the verifier has no atomic slot."""
    useful_residual = [
        slot
        for slot in residual_contracts
        if not bool(slot.get("meta_negative"))
        and (
            bool(slot.get("join_executable"))
            or slot.get("execution_operator")
            in {"argmax", "argmin", "text_retrieval"}
        )
    ]
    if useful_residual:
        return useful_residual, "verifier_residual_slots"

    planned_contracts = _slot_contracts(
        question,
        planned_retrieval,
        typed_plan=True,
        catalog_guided=True,
    )
    allowed_sources = {
        "residual_deficit_query",
        "residual_deficit_replaced_query",
        "verifier_query",
        "planner_query",
        "verifier_missing_slot",
    }
    fallback = [
        slot
        for slot in planned_contracts
        if slot.get("source") in allowed_sources
        and not bool(slot.get("meta_negative"))
        and (
            bool(slot.get("join_executable"))
            or slot.get("execution_operator")
            in {"argmax", "argmin", "text_retrieval"}
        )
    ]
    return fallback, "planned_slot_fallback" if fallback else "no_executable_slot"


def _slot_contract_diagnostics(
    slot_contracts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "slot_id": slot["slot_id"],
            "text": slot["text"],
            "source": slot["source"],
            "hop_index": int(slot["hop_index"]),
            "anchor_tokens": sorted(slot["anchor_tokens"]),
            "relation_tokens": sorted(slot["relation_tokens"]),
            "relation_families": sorted(slot["relation_families"]),
            "polarity": sorted(slot["polarity"]),
            "answer_types": sorted(slot["answer_types"]),
            "target_predicate_concepts": sorted(slot["target_predicate_concepts"]),
            "expected_value_type": slot["expected_value_type"],
            "direction": slot["direction"],
            "relation_signature": slot.get("relation_signature", ""),
            "relation_arity": int(slot.get("relation_arity", 1) or 1),
            "execution_operator": slot.get("execution_operator", "lookup"),
            "join_executable": bool(slot.get("join_executable")),
            "materialization_safe": bool(slot["materialization_safe"]),
            "slot_plan_stage": str(
                slot.get("slot_plan_stage") or "unclassified"
            ),
            "slot_branch_id": str(slot.get("slot_branch_id") or ""),
            "final_answer_type": str(
                slot.get("final_answer_type") or "unknown"
            ),
            "slot_expected_type_family": str(
                slot.get("slot_expected_type_family") or "unknown"
            ),
            "named_anchor_phrases": list(
                slot.get("named_anchor_phrases") or []
            ),
            "unsafe_reasons": list(slot["unsafe_reasons"]),
            "meta_negative": bool(slot["meta_negative"]),
        }
        for slot in slot_contracts
    ]


def _anchor_match(entity_tokens: Set[str], anchor_tokens: Set[str], entity: str, text: str) -> float:
    if not entity_tokens or not anchor_tokens:
        return 0.0
    overlap = len(entity_tokens & anchor_tokens)
    denominator = max(1, max(len(entity_tokens), min(3, len(anchor_tokens))))
    score = overlap / float(denominator)
    entity_surface = " ".join(str(entity or "").lower().split())
    text_surface = " ".join(str(text or "").lower().split())
    if entity_surface and entity_surface in text_surface:
        score = max(score, 1.0)
    return _clamp(score)


def _predicate_concepts(tokens: Set[str]) -> Set[str]:
    return {
        concept
        for concept, terms in _PREDICATE_CONCEPTS.items()
        if tokens & terms
    }


def _predicate_contract(
    relation_tokens: Set[str],
    slot: Dict[str, Any],
    direction: str,
    *,
    candidate_signature: str = "",
    strict_signature: bool = False,
    allow_signature_compatibility: bool = False,
    candidate_answer_type: str = "",
    expected_answer_types: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    expected_tokens = set(slot.get("relation_tokens") or set())
    expected_polarity = set(slot.get("polarity") or set())
    direct_tokens = relation_tokens & expected_tokens
    discriminative_expected = expected_tokens - _NON_DISCRIMINATIVE_PREDICATE_TERMS
    discriminative_candidate = relation_tokens - _NON_DISCRIMINATIVE_PREDICATE_TERMS
    discriminative_direct_tokens = discriminative_expected & discriminative_candidate
    direct_overlap = (
        _bounded_overlap(
            discriminative_candidate,
            discriminative_expected,
            max_required=3,
        )
        if discriminative_expected else 0.0
    )
    expected_concepts = _predicate_concepts(expected_tokens)
    candidate_concepts = _predicate_concepts(relation_tokens)
    expected_signature = str(slot.get("relation_signature") or "")
    signature_match = bool(
        expected_signature
        and candidate_signature
        and expected_signature == candidate_signature
    )
    signature_compatibility = _relation_signature_compatibility(
        expected_signature,
        candidate_signature,
        candidate_answer_type=candidate_answer_type,
        expected_answer_types=expected_answer_types,
    )
    signature_compatible = bool(
        allow_signature_compatibility
        and signature_compatibility["tier"] == "compatible"
    )
    concept_overlap = bool(expected_concepts & candidate_concepts)
    polarity_match = any(
        relation_tokens & RELATION_POLARITY_TERMS.get(name, set())
        for name in expected_polarity
    )
    inverse_match = bool(
        direction == "subject_to_object"
        and (
            (
                bool(expected_concepts & {"father", "mother", "parent", "grandparent"})
                and "child" in candidate_concepts
            )
            or (
                "child" in expected_concepts
                and bool(candidate_concepts & {"father", "mother", "parent"})
            )
        )
    )
    matched = bool(
        signature_match
        or signature_compatible
        or (
            not (strict_signature and expected_signature)
            and (direct_overlap > 0.0 or concept_overlap or polarity_match or inverse_match)
        )
    )
    if signature_match:
        mode = "exact_relation_signature"
        score = 1.0
    elif signature_compatible:
        mode = "compatible_relation_signature"
        score = float(signature_compatibility["score"])
    elif strict_signature and expected_signature:
        mode = "relation_signature_mismatch"
        score = 0.0
    elif direct_overlap > 0.0:
        mode = "direct_token"
        score = max(0.86, direct_overlap)
    elif polarity_match:
        mode = "polarity_concept"
        score = 0.92
    elif inverse_match:
        mode = "inverse_role"
        score = 0.88
    elif concept_overlap:
        mode = "normalized_concept"
        score = 0.86
    else:
        mode = "family_only"
        score = 0.0
    return {
        "matched": matched,
        "mode": mode,
        "score": _clamp(score),
        "direct_overlap": _clamp(direct_overlap),
        "direct_tokens": sorted(direct_tokens),
        "discriminative_direct_tokens": sorted(discriminative_direct_tokens),
        "discriminative_expected_tokens": sorted(discriminative_expected),
        "expected_concepts": sorted(expected_concepts),
        "candidate_concepts": sorted(candidate_concepts),
        "expected_signature": expected_signature,
        "candidate_signature": str(candidate_signature or ""),
        "signature_match": signature_match,
        "signature_compatible": signature_compatible,
        "signature_compatibility_tier": str(signature_compatibility["tier"]),
    }


def _dependency_binding(endpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that a join endpoint is the terminal node of one rooted fact path."""
    active_key = str(endpoint.get("entity_key") or "")
    root_key = str(endpoint.get("root_entity_key") or "")
    lineage_path = list(endpoint.get("lineage_path") or [])
    reasons: List[str] = []
    path_entity_keys: List[List[str]] = []

    for raw_fact in lineage_path:
        fact = _clean_triple(raw_fact)
        if fact is None:
            reasons.append("invalid_lineage_fact")
            continue
        path_entity_keys.append([_entity_key(fact[0]), _entity_key(fact[2])])

    if not bool(endpoint.get("lineage_bound")):
        reasons.append("lineage_unbound")
    if not active_key or not root_key:
        reasons.append("missing_endpoint_identity")
    if path_entity_keys:
        if root_key not in path_entity_keys[0]:
            reasons.append("root_not_in_first_dependency")
        if active_key not in path_entity_keys[-1]:
            reasons.append("active_entity_not_terminal")
        for previous, current in zip(path_entity_keys, path_entity_keys[1:]):
            if not set(previous) & set(current):
                reasons.append("disconnected_dependency_path")
                break
    elif active_key and root_key and active_key != root_key:
        reasons.append("zero_hop_identity_mismatch")

    return {
        "safe": not reasons,
        "reasons": reasons,
        "root_entity_key": root_key,
        "terminal_entity_key": active_key,
        "path_entity_keys": path_entity_keys,
        "dependency_hops": len(path_entity_keys),
    }


def _active_entity_endpoints(
    question: str,
    top_k_facts: List[Tuple],
    candidate_facts: Optional[List[Tuple]],
    max_active_facts: int,
    slot_contracts: List[Dict[str, Any]],
    lineage_constrained: bool,
    max_lineage_hops: int,
    require_atomic_slot: bool,
) -> List[Dict[str, Any]]:
    facts = list(top_k_facts or [])
    if len(facts) < max_active_facts:
        facts.extend(list(candidate_facts or [])[: max_active_facts - len(facts)])

    cleaned_facts = []
    adjacency: Dict[str, List[Tuple[Tuple[str, str, str], str]]] = {}
    entity_surfaces: Dict[str, str] = {}
    for raw_fact in facts[: max(1, int(max_active_facts))]:
        fact = _clean_triple(raw_fact)
        if fact is None:
            continue
        subject_key = _entity_key(fact[0])
        object_key = _entity_key(fact[2])
        if not subject_key or not object_key:
            continue
        cleaned_facts.append(fact)
        entity_surfaces.setdefault(subject_key, str(fact[0]))
        entity_surfaces.setdefault(object_key, str(fact[2]))
        adjacency.setdefault(subject_key, []).append((fact, object_key))
        adjacency.setdefault(object_key, []).append((fact, subject_key))

    question_profile = _relation_profile(question)
    question_anchors = set(question_profile.get("anchor_tokens") or [])
    reached: Dict[str, Dict[str, Any]] = {}
    if lineage_constrained:
        for entity_key, entity in entity_surfaces.items():
            entity_tokens = _tokens(entity)
            score = _anchor_match(entity_tokens, question_anchors, entity, question)
            hit_count = len(entity_tokens & question_anchors)
            if score >= 0.67 and (hit_count >= 2 or len(entity_tokens) == 1):
                reached[entity_key] = {
                    "root_entity": entity,
                    "root_entity_key": entity_key,
                    "lineage_depth": 0,
                    "lineage_path": [],
                    "root_anchor_score": score,
                }
        frontier = list(reached)
        for _ in range(max(0, int(max_lineage_hops))):
            next_frontier = []
            for entity_key in frontier:
                lineage = reached[entity_key]
                for fact, neighbor_key in adjacency.get(entity_key, []):
                    if neighbor_key in reached:
                        continue
                    reached[neighbor_key] = {
                        "root_entity": lineage["root_entity"],
                        "root_entity_key": lineage["root_entity_key"],
                        "lineage_depth": int(lineage["lineage_depth"]) + 1,
                        "lineage_path": list(lineage["lineage_path"]) + [list(fact)],
                        "root_anchor_score": lineage["root_anchor_score"],
                    }
                    next_frontier.append(neighbor_key)
            frontier = next_frontier
            if not frontier:
                break
    else:
        reached = {
            key: {
                "root_entity": entity,
                "root_entity_key": key,
                "lineage_depth": 0,
                "lineage_path": [],
                "root_anchor_score": 1.0,
            }
            for key, entity in entity_surfaces.items()
        }

    endpoints: List[Dict[str, Any]] = []
    seen = set()
    fallback_contract = len(slot_contracts) == 1 and slot_contracts[0]["text"] == question
    for fact in cleaned_facts:
        signature = _schema_signature(fact)
        for role, value in (("subject", fact[0]), ("object", fact[2])):
            key = _entity_key(value)
            if not key or key not in reached or len(_tokens(value)) == 0:
                continue
            slot_matches = []
            for slot in slot_contracts:
                score = _anchor_match(
                    _tokens(value),
                    set(slot.get("anchor_tokens") or set()),
                    str(value),
                    str(slot.get("text") or ""),
                )
                if score >= 0.67:
                    slot_matches.append((score, slot))
            if not slot_matches and fallback_contract and reached[key]["lineage_depth"] > 0:
                slot_matches.append((reached[key]["root_anchor_score"], slot_contracts[0]))
            if not slot_matches:
                continue
            slot_score, slot = max(slot_matches, key=lambda pair: pair[0])
            identity = (key, role, fact)
            if identity in seen:
                continue
            seen.add(identity)
            endpoints.append(
                {
                    "entity": str(value),
                    "entity_key": key,
                    "entity_tokens": _tokens(value),
                    "active_role": role,
                    "active_triple": fact,
                    "active_schema": signature,
                    "lineage_bound": True,
                    "root_entity": reached[key]["root_entity"],
                    "root_entity_key": reached[key]["root_entity_key"],
                    "root_anchor_score": round(float(reached[key]["root_anchor_score"]), 6),
                    "lineage_depth": int(reached[key]["lineage_depth"]),
                    "lineage_path": list(reached[key]["lineage_path"]),
                    "slot_id": str(slot.get("slot_id") or ""),
                    "slot_text": str(slot.get("text") or ""),
                    "slot_anchor_overlap": round(float(slot_score), 6),
                    "slot_anchor_tokens": sorted(slot.get("anchor_tokens") or []),
                    "slot_relation_tokens": sorted(slot.get("relation_tokens") or []),
                    "slot_relation_families": sorted(slot.get("relation_families") or []),
                    "slot_polarity": sorted(slot.get("polarity") or []),
                    "slot_answer_types": sorted(slot.get("answer_types") or []),
                    "slot_source": str(slot.get("source") or "unknown"),
                    "slot_hop_index": int(slot.get("hop_index", -1)),
                    "slot_target_predicate_concepts": sorted(
                        slot.get("target_predicate_concepts") or []
                    ),
                    "slot_expected_value_type": str(
                        slot.get("expected_value_type") or "unknown"
                    ),
                    "slot_direction": str(
                        slot.get("direction") or "bound_entity_to_value"
                    ),
                    "slot_relation_signature": str(
                        slot.get("relation_signature") or ""
                    ),
                    "slot_relation_arity": int(slot.get("relation_arity", 1) or 1),
                    "slot_execution_operator": str(
                        slot.get("execution_operator") or "lookup"
                    ),
                    "slot_join_executable": bool(slot.get("join_executable")),
                    "slot_materialization_safe": bool(
                        slot.get("materialization_safe") or not require_atomic_slot
                    ),
                    "slot_plan_stage": str(
                        slot.get("slot_plan_stage") or "unclassified"
                    ),
                    "slot_branch_id": str(slot.get("slot_branch_id") or ""),
                    "slot_final_answer_type": str(
                        slot.get("final_answer_type") or "unknown"
                    ),
                    "slot_expected_type_family": str(
                        slot.get("slot_expected_type_family") or "unknown"
                    ),
                    "slot_named_anchor_phrases": list(
                        slot.get("named_anchor_phrases") or []
                    ),
                    "slot_unsafe_reasons": list(slot.get("unsafe_reasons") or []),
                    "slot_meta_negative": bool(slot.get("meta_negative")),
                }
            )
    return endpoints


def _candidate_answer_side(
    item: Dict[str, Any],
    entity_key: str,
) -> Optional[Dict[str, str]]:
    subject_key = str(item.get("subject_key") or "")
    object_key = str(item.get("object_key") or "")
    if subject_key == entity_key and object_key and object_key != entity_key:
        return {
            "candidate_join_role": "subject",
            "answer_role": "object",
            "answer_value": str(item.get("object") or ""),
            "answer_key": object_key,
        }
    if object_key == entity_key and subject_key and subject_key != entity_key:
        return {
            "candidate_join_role": "object",
            "answer_role": "subject",
            "answer_value": str(item.get("subject") or ""),
            "answer_key": subject_key,
        }
    return None


def _select_operator_candidates(
    operator: str,
    all_ranked: List[Dict[str, Any]],
    slot_contracts: List[Dict[str, Any]],
    *,
    candidate_top_k: int,
    query_candidate_budget: int,
    max_candidates_per_slot: int,
    require_final_stage: bool = False,
) -> Dict[str, Any]:
    query_limit = min(
        max(0, int(candidate_top_k)),
        max(0, int(query_candidate_budget)),
    )
    if query_limit <= 0:
        return {
            "candidates": [],
            "complete": False,
            "required_bindings": 0,
            "bound_bindings": 0,
            "physical_operator": "budget_exhausted",
            "rejected_by_budget": 0,
        }

    if operator == "compare":
        selected = []
        seen_bindings = set()
        for row in all_ranked:
            binding = (
                str(row.get("slot_id") or "unknown"),
                str(
                    row.get("root_entity_key")
                    or row.get("active_entity_key")
                    or "unknown"
                ),
            )
            if binding in seen_bindings:
                continue
            seen_bindings.add(binding)
            selected.append(row)
            if len(selected) >= query_limit:
                break
        required = max(
            2,
            min(
                query_limit,
                len({
                    str(slot.get("slot_id") or "")
                    for slot in slot_contracts
                    if bool(slot.get("join_executable"))
                }),
            ),
        )
        root_count = len({
            str(
                row.get("root_entity_key")
                or row.get("active_entity_key")
                or ""
            )
            for row in selected
            if str(
                row.get("root_entity_key")
                or row.get("active_entity_key")
                or ""
            )
        })
        slot_count = len({
            str(row.get("slot_id") or "")
            for row in selected
            if str(row.get("slot_id") or "")
        })
        bound = max(root_count, slot_count)
        return {
            "candidates": selected,
            "complete": bound >= required,
            "required_bindings": required,
            "bound_bindings": bound,
            "physical_operator": "branch_fact_lookup",
            "rejected_by_budget": max(0, len(all_ranked) - len(selected)),
        }

    if operator == "multi_key_fact_join":
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in all_ranked:
            answer_key = str(row.get("answer_key") or "")
            if answer_key:
                groups.setdefault(answer_key, []).append(row)
        required = max(
            [int(slot.get("relation_arity", 1) or 1) for slot in slot_contracts]
            or [2]
        )
        required = max(2, required)

        def group_key(rows: List[Dict[str, Any]]) -> Tuple[int, float, float]:
            endpoints = {
                str(row.get("active_entity_key") or "")
                for row in rows
                if str(row.get("active_entity_key") or "")
            }
            return (
                len(endpoints),
                max(float(row.get("net_utility", 0.0) or 0.0) for row in rows),
                max(float(row.get("relation_overlap", 0.0) or 0.0) for row in rows),
            )

        best_rows = max(groups.values(), key=group_key, default=[])
        selected = []
        seen_endpoints = set()
        for row in best_rows:
            endpoint = str(row.get("active_entity_key") or "")
            if endpoint in seen_endpoints:
                continue
            seen_endpoints.add(endpoint)
            selected.append(row)
            if len(selected) >= query_limit:
                break
        if not selected:
            selected = all_ranked[:query_limit]
            seen_endpoints = {
                str(row.get("active_entity_key") or "") for row in selected
            }
        return {
            "candidates": selected,
            "complete": len(seen_endpoints) >= required,
            "required_bindings": required,
            "bound_bindings": len(seen_endpoints),
            "physical_operator": "shared_key_fact_join",
            "rejected_by_budget": max(0, len(all_ranked) - len(selected)),
        }

    per_slot_counts: Counter = Counter()
    selected = []
    for row in all_ranked:
        slot_id = str(row.get("slot_id") or "unknown")
        if per_slot_counts[slot_id] >= max(1, int(max_candidates_per_slot)):
            continue
        selected.append(row)
        per_slot_counts[slot_id] += 1
        if len(selected) >= query_limit:
            break
    return {
        "candidates": selected,
        "complete": bool(selected) and (
            not require_final_stage
            or any(
                str(row.get("slot_plan_stage") or "") == "final_answer"
                for row in selected
            )
        ),
        "required_bindings": 1,
        "bound_bindings": int(bool(selected)),
        "physical_operator": "indexed_fact_lookup",
        "rejected_by_budget": max(0, len(all_ranked) - len(selected)),
    }


def _operator_request_texts(
    operator: str,
    slot_contracts: List[Dict[str, Any]],
    limit: int,
) -> List[str]:
    preferred = {
        "argmax": {"argmax"},
        "argmin": {"argmin"},
        "text_retrieval": {"text_retrieval"},
    }.get(operator, set())
    values = [
        str(slot.get("text") or "").strip()
        for slot in slot_contracts
        if str(slot.get("text") or "").strip()
        and (
            not preferred
            or str(slot.get("execution_operator") or "") in preferred
        )
    ]
    if not values:
        values = [
            str(slot.get("text") or "").strip()
            for slot in slot_contracts
            if str(slot.get("text") or "").strip()
        ]
    return list(dict.fromkeys(values))[: max(1, int(limit))]


def _run_operator_retrieval(
    operator: str,
    slot_contracts: List[Dict[str, Any]],
    operator_retriever: Optional[Callable[[str, int], Any]],
    *,
    doc_ids: List[int],
    get_doc_content: Callable[[int], str],
    qa_top_k: int,
    candidate_budget: int,
    validate_completion: bool = False,
    validated_top_k: int = 1,
    plan_stage_gate: bool = False,
) -> Dict[str, Any]:
    requests = _operator_request_texts(
        operator,
        slot_contracts,
        max(1, int(candidate_budget)),
    )
    result = {
        "requests": requests,
        "candidates": [],
        "complete": False,
        "physical_operator": "dense_text_retrieval",
        "outside_top_k": 0,
        "raw_candidate_count": 0,
        "validation_enabled": bool(validate_completion),
        "validation_passed_count": 0,
        "validation_reject_reason_counts": {},
    }
    if operator_retriever is None or not requests:
        return result

    original_rank = {int(doc_id): rank for rank, doc_id in enumerate(doc_ids)}
    current_top = {int(doc_id) for doc_id in doc_ids[: max(1, int(qa_top_k))]}
    by_doc: Dict[int, Dict[str, Any]] = {}
    per_query_k = max(4, int(candidate_budget) * 2)
    for request in requests:
        raw = operator_retriever(request, per_query_k)
        if isinstance(raw, tuple) and len(raw) == 2:
            result_ids, result_scores = raw
        else:
            result_ids, result_scores = raw or [], []
        for operator_rank, raw_doc_id in enumerate(list(result_ids)[:per_query_k]):
            try:
                doc_id = int(raw_doc_id)
                score = float(list(result_scores)[operator_rank])
                content = str(get_doc_content(doc_id) or "")
            except (IndexError, TypeError, ValueError):
                continue
            if not content:
                continue
            candidate = {
                "doc_id": doc_id,
                "title": content.split("\n", 1)[0].strip(),
                "score": round(score, 6),
                "risk": 0.18,
                "net_utility": round(score - 0.063, 6),
                "original_rank": int(original_rank.get(doc_id, len(doc_ids))),
                "retrieval_pool_member": doc_id in original_rank,
                "outside_retrieval_pool": doc_id not in original_rank,
                "already_top5": doc_id in current_top,
                "operator_candidate_kind": "dense_text_retrieval",
                "retrieval_query": request,
                "operator_retrieval_rank": operator_rank,
                "answer_value": "",
                "predicate_match_mode": "operator_text_retrieval",
                "relation_signature_match": False,
                "relation_signature_compatible": False,
                "slot_materialization_safe": False,
            }
            if validate_completion:
                validation = _validate_dense_operator_candidate(
                    content,
                    slot_contracts,
                    require_branch_binding=plan_stage_gate,
                )
                candidate.update({
                    "operator_validation_passed": bool(validation["passed"]),
                    "operator_validation_score": float(validation["score"]),
                    "operator_validation_reason": str(validation["reason"]),
                    "operator_validation_grounding_sentence": str(
                        validation.get("grounding_sentence") or ""
                    ),
                    "operator_validation_answer_type_surface": str(
                        validation.get("answer_type_surface") or ""
                    ),
                    "slot_id": str(validation.get("slot_id") or ""),
                    "slot_text": str(validation.get("slot_text") or ""),
                    "slot_source": str(validation.get("slot_source") or "unknown"),
                    "slot_answer_types": list(
                        validation.get("slot_answer_types") or []
                    ),
                    "slot_anchor_tokens": list(
                        validation.get("slot_anchor_tokens") or []
                    ),
                    "slot_relation_tokens": list(
                        validation.get("slot_relation_tokens") or []
                    ),
                    "slot_anchor_overlap": float(
                        validation.get("anchor_overlap", 0.0) or 0.0
                    ),
                    "relation_overlap": float(
                        validation.get("predicate_overlap", 0.0) or 0.0
                    ),
                    "predicate_contract_match": bool(validation["passed"]),
                    "predicate_match_mode": str(
                        validation.get("predicate_match_mode") or ""
                    ),
                    "slot_materialization_safe": bool(
                        validation.get("slot_materialization_safe")
                    ),
                    "slot_plan_stage": str(
                        validation.get("slot_plan_stage") or "unclassified"
                    ),
                    "slot_branch_id": str(
                        validation.get("slot_branch_id") or ""
                    ),
                    "slot_final_answer_type": str(
                        validation.get("slot_final_answer_type") or "unknown"
                    ),
                    "slot_expected_type_family": str(
                        validation.get("slot_expected_type_family") or "unknown"
                    ),
                    "branch_binding_passed": bool(
                        validation.get("branch_binding_passed")
                    ),
                    "branch_binding_reason": str(
                        validation.get("branch_binding_reason") or ""
                    ),
                    "branch_anchor_phrase": str(
                        validation.get("branch_anchor_phrase") or ""
                    ),
                    "materialization_role": "support_only",
                    "source_surface_grounded": bool(validation["passed"]),
                })
            current = by_doc.get(doc_id)
            if current is None or (
                not bool(current.get("already_top5")),
                float(current.get("score", 0.0)),
            ) < (
                not bool(candidate.get("already_top5")),
                float(candidate.get("score", 0.0)),
            ):
                by_doc[doc_id] = candidate

    raw_ranked = sorted(
        by_doc.values(),
        key=lambda row: (
            not bool(row.get("already_top5")),
            float(row.get("score", 0.0)),
            -int(row.get("operator_retrieval_rank", 0)),
        ),
        reverse=True,
    )[: max(0, int(candidate_budget))]
    result["raw_candidate_count"] = len(raw_ranked)
    if validate_completion:
        validation_rejects: Counter = Counter(
            str(row.get("operator_validation_reason") or "unknown")
            for row in raw_ranked
            if not bool(row.get("operator_validation_passed"))
        )
        ranked = [
            row for row in raw_ranked
            if bool(row.get("operator_validation_passed"))
        ][: max(1, int(validated_top_k))]
        result["validation_passed_count"] = sum(
            bool(row.get("operator_validation_passed")) for row in raw_ranked
        )
        result["validation_reject_reason_counts"] = dict(validation_rejects)
    else:
        ranked = raw_ranked
    result["candidates"] = ranked
    result["complete"] = bool(ranked) and (
        not plan_stage_gate
        or any(
            str(row.get("slot_plan_stage") or "") == "final_answer"
            and bool(row.get("branch_binding_passed"))
            for row in ranked
        )
    )
    result["outside_top_k"] = sum(not row["already_top5"] for row in ranked)
    return result


def _build_entity_predicate_fact_join_candidates_impl(
    *,
    question: str,
    sorted_doc_ids: Any,
    top_k_facts: List[Tuple],
    candidate_facts: Optional[List[Tuple]],
    relation_index: Dict[str, Any],
    get_doc_content: Callable[[int], str],
    qa_top_k: int,
    enabled: bool = False,
    bridge_only: bool = True,
    require_evidence_gap: bool = False,
    max_active_facts: int = 16,
    max_entity_fanout: int = 240,
    candidate_top_k: int = 48,
    max_original_rank: int = 200,
    min_relation_overlap: float = 0.34,
    min_score: float = 0.58,
    planned_retrieval: Optional[Dict[str, Any]] = None,
    lineage_constrained: bool = True,
    max_lineage_hops: int = 2,
    require_atomic_slot: bool = True,
    residual_deficit_trigger: bool = False,
    typed_residual_plan: bool = False,
    max_exact_signature_fanout: int = 32,
    catalog_guided_executor: bool = False,
    operator_faithful_executor: bool = False,
    validate_operator_completion: bool = False,
    validated_dense_top_k: int = 1,
    plan_stage_gate: bool = False,
    signature_compatibility_fallback: bool = False,
    operator_retriever: Optional[Callable[[str, int], Any]] = None,
    max_candidates_per_slot: int = 2,
    query_candidate_budget: int = 4,
    generic_signature_max_fanout: int = 4,
) -> Dict[str, Any]:
    """Create rank-stable fact candidates through an exact entity-predicate join.

    Unlike query schema memory, this access path does not require the next fact
    to share the active fact's schema. It binds one concrete endpoint from an
    active fact, probes the provenance-aware fact store by that entity key, and
    scores the opposite endpoint against the requested predicate and answer
    type. Reader admission happens later and never changes retrieval ranking.
    """
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled" if not enabled else "not_evaluated",
        "rank_stable": True,
        "operator": "entity_predicate_fact_join",
        "index_stats": dict((relation_index or {}).get("stats") or {}),
        "bridge_only": bool(bridge_only),
        "require_evidence_gap": bool(require_evidence_gap),
        "active_endpoint_count": 0,
        "probed_entity_count": 0,
        "raw_fact_count": 0,
        "candidate_count": 0,
        "lineage_bound_candidate_count": 0,
        "predicate_contract_candidate_count": 0,
        "dependency_safe_candidate_count": 0,
        "family_only_candidate_count": 0,
        "cross_schema_candidate_count": 0,
        "already_visible_count": 0,
        "candidate_items": [],
        "lineage_constrained": bool(lineage_constrained),
        "require_atomic_slot": bool(require_atomic_slot),
        "residual_deficit_trigger_enabled": bool(residual_deficit_trigger),
        "residual_deficit_triggered": False,
        "residual_deficit_trigger_reason": "disabled",
        "residual_deficit_planned_reason": str(
            (planned_retrieval or {}).get("reason") or ""
        ),
        "residual_deficit_atomic_slot_count": 0,
        "typed_residual_plan_enabled": bool(typed_residual_plan),
        "catalog_guided_executor_enabled": bool(catalog_guided_executor),
        "operator_faithful_executor_enabled": bool(operator_faithful_executor),
        "validated_completion_enabled": bool(validate_operator_completion),
        "plan_stage_gate_enabled": bool(plan_stage_gate),
        "final_answer_type": _question_final_answer_type(question),
        "signature_compatibility_fallback_enabled": bool(
            signature_compatibility_fallback
        ),
        "executor_state": "not_routed",
        "selected_operator": "not_routed",
        "physical_operator": "not_routed",
        "operator_selection_reason": "not_routed",
        "executor_triggered": False,
        "operator_complete": False,
        "operator_execution_succeeded": False,
        "operator_validation_candidate_count": 0,
        "operator_validation_passed_count": 0,
        "operator_validation_reject_reason_counts": {},
        "operator_required_binding_count": 0,
        "operator_bound_binding_count": 0,
        "operator_request_count": 0,
        "operator_requests": [],
        "operator_retrieval_candidate_count": 0,
        "operator_retrieval_outside_top_k_count": 0,
        "operator_final_binding_count": 0,
        "operator_intermediate_binding_count": 0,
        "operator_support_only_count": 0,
        "operator_requires_followup": False,
        "slot_contract_source": "not_compiled",
        "residual_query_plan_type": "not_compiled",
        "residual_executable_slot_count": 0,
        "residual_unsafe_derived_slot_count": 0,
        "bridge_gate_bypassed": False,
        "exact_signature_candidate_count": 0,
        "compatible_signature_candidate_count": 0,
        "hard_signature_reject_count": 0,
        "selective_global_access_count": 0,
        "selective_global_access_raw_count": 0,
        "selective_global_access_unique_count": 0,
        "catalog_lookup_count": 0,
        "catalog_estimated_scan_rows": 0,
        "slot_contracts": [],
        "reject_reason_counts": {},
        "thresholds": {
            "max_active_facts": int(max_active_facts),
            "max_entity_fanout": int(max_entity_fanout),
            "candidate_top_k": int(candidate_top_k),
            "max_original_rank": int(max_original_rank),
            "min_relation_overlap": round(float(min_relation_overlap), 6),
            "min_score": round(float(min_score), 6),
            "max_lineage_hops": int(max_lineage_hops),
            "max_exact_signature_fanout": int(max_exact_signature_fanout),
            "max_candidates_per_slot": int(max_candidates_per_slot),
            "query_candidate_budget": int(query_candidate_budget),
            "generic_signature_max_fanout": int(generic_signature_max_fanout),
            "validated_dense_top_k": int(validated_dense_top_k),
        },
    }
    if not enabled:
        return info
    items = list((relation_index or {}).get("items") or [])
    entity_to_items = (relation_index or {}).get("entity_to_items") or {}
    if not items or not entity_to_items:
        info["reason"] = "no_fact_store_index"
        return info

    doc_ids = sorted_doc_ids.tolist() if hasattr(sorted_doc_ids, "tolist") else list(sorted_doc_ids or [])
    if not doc_ids:
        info["reason"] = "no_retrieved_docs"
        return info
    current_docs: List[str] = []
    for doc_id in doc_ids[: max(1, int(qa_top_k))]:
        try:
            current_docs.append(str(get_doc_content(int(doc_id)) or ""))
        except Exception:
            continue
    gap = _answerability_gap_profile(
        question,
        current_docs,
        qa_top_k=max(1, int(qa_top_k)),
        min_query_coverage=0.15,
    )
    info["gap"] = gap

    planned_reason = str((planned_retrieval or {}).get("reason") or "")
    residual_slot_contracts = _residual_deficit_slot_contracts(
        planned_retrieval,
        typed_plan=typed_residual_plan,
        catalog_guided=catalog_guided_executor,
    )
    slot_contract_source = "verifier_residual_slots"
    if operator_faithful_executor and typed_residual_plan:
        residual_slot_contracts, slot_contract_source = (
            _operator_faithful_slot_contracts(
                question,
                planned_retrieval,
                residual_slot_contracts,
            )
        )
    if plan_stage_gate:
        residual_slot_contracts = _annotate_slot_plan_stages(
            question,
            residual_slot_contracts,
        )
    info["slot_contract_source"] = slot_contract_source
    safe_residual_slot_contracts = [
        slot
        for slot in residual_slot_contracts
        if bool(
            slot.get("join_executable")
            if typed_residual_plan
            else slot.get("materialization_safe")
        )
    ]
    residual_query_plan_type = _residual_query_plan_type(
        question,
        residual_slot_contracts,
    ) if typed_residual_plan else "legacy_fact_join"
    info["residual_query_plan_type"] = residual_query_plan_type
    executor_route = _route_residual_executor(
        plan_type=residual_query_plan_type,
        slot_contracts=residual_slot_contracts,
        planned_reason=planned_reason,
        residual_deficit_trigger=residual_deficit_trigger,
        operator_faithful=operator_faithful_executor,
        evidence_deficit=bool(gap.get("answerability_gap")),
    ) if catalog_guided_executor else {
        "state": "legacy_typed_execution",
        "operator": residual_query_plan_type,
        "reason": "catalog_guided_executor_disabled",
    }
    info["executor_state"] = executor_route["state"]
    info["selected_operator"] = executor_route["operator"]
    info["operator_selection_reason"] = executor_route["reason"]
    info["residual_executable_slot_count"] = sum(
        bool(slot.get("join_executable")) for slot in residual_slot_contracts
    )
    info["residual_unsafe_derived_slot_count"] = sum(
        slot.get("execution_operator") == "unsafe_derived"
        for slot in residual_slot_contracts
    )
    eligible_reasons = (
        _OPERATOR_EXHAUSTED_REASONS
        if operator_faithful_executor
        else _RESIDUAL_DEFICIT_TRIGGER_REASONS
    )
    residual_eligible = bool(
        residual_deficit_trigger
        and (
            planned_reason in eligible_reasons
            or (
                operator_faithful_executor
                and bool(gap.get("answerability_gap"))
            )
        )
        and safe_residual_slot_contracts
    )
    info["residual_deficit_atomic_slot_count"] = len(safe_residual_slot_contracts)
    info["slot_contracts"] = _slot_contract_diagnostics(residual_slot_contracts)
    if not residual_deficit_trigger:
        info["residual_deficit_trigger_reason"] = "disabled"
    elif planned_reason not in eligible_reasons and not (
        operator_faithful_executor and bool(gap.get("answerability_gap"))
    ):
        info["residual_deficit_trigger_reason"] = "planned_reason_not_eligible"
    elif not residual_slot_contracts:
        info["residual_deficit_trigger_reason"] = "no_verifier_missing_slot"
    elif not safe_residual_slot_contracts:
        info["residual_deficit_trigger_reason"] = "no_atomic_residual_slot"
    else:
        info["residual_deficit_trigger_reason"] = (
            "typed_residual_plan_compiled"
            if typed_residual_plan
            else "atomic_residual_slot_compiled"
        )
    executable_operators = {"compare", "fact_lookup", "multi_key_fact_join"}
    retrieval_operators = {"argmax", "argmin", "text_retrieval"}
    info["executor_triggered"] = bool(
        catalog_guided_executor
        and executor_route["operator"] in executable_operators | retrieval_operators
    )
    if (
        catalog_guided_executor
        and operator_faithful_executor
        and executor_route["operator"] in retrieval_operators
    ):
        retrieval = _run_operator_retrieval(
            executor_route["operator"],
            residual_slot_contracts,
            operator_retriever,
            doc_ids=[int(doc_id) for doc_id in doc_ids],
            get_doc_content=get_doc_content,
            qa_top_k=qa_top_k,
            candidate_budget=query_candidate_budget,
            validate_completion=validate_operator_completion,
            validated_top_k=validated_dense_top_k,
            plan_stage_gate=plan_stage_gate,
        )
        ranked = retrieval["candidates"]
        info.update({
            "applied": bool(ranked),
            "reason": (
                "operator_retrieval_candidates_generated"
                if ranked else f"executor_routed_to_{executor_route['operator']}"
            ),
            "physical_operator": retrieval["physical_operator"],
            "operator_complete": bool(retrieval["complete"]),
            "operator_execution_succeeded": bool(
                retrieval["raw_candidate_count"]
            ),
            "operator_required_binding_count": 1,
            "operator_bound_binding_count": int(bool(ranked)),
            "operator_request_count": len(retrieval["requests"]),
            "operator_requests": retrieval["requests"],
            "operator_retrieval_candidate_count": len(ranked),
            "operator_retrieval_outside_top_k_count": int(
                retrieval["outside_top_k"]
            ),
            "operator_validation_candidate_count": int(
                retrieval["raw_candidate_count"]
            ),
            "operator_validation_passed_count": int(
                retrieval["validation_passed_count"]
            ),
            "operator_validation_reject_reason_counts": dict(
                retrieval["validation_reject_reason_counts"]
            ),
            "candidate_count": len(ranked),
            "candidate_items": ranked,
            "operator_final_binding_count": sum(
                str(row.get("slot_plan_stage") or "") == "final_answer"
                for row in ranked
            ),
            "operator_intermediate_binding_count": sum(
                str(row.get("slot_plan_stage") or "") == "intermediate_binding"
                for row in ranked
            ),
            "operator_support_only_count": sum(
                str(row.get("slot_plan_stage") or "") == "support_only"
                for row in ranked
            ),
            "operator_requires_followup": bool(
                ranked and not retrieval["complete"]
            ),
        })
        return info
    if catalog_guided_executor and executor_route["operator"] not in executable_operators:
        info["reason"] = f"executor_routed_to_{executor_route['operator']}"
        return info

    bridge_gate_bypassed = bool(
        bridge_only
        and not bool(gap.get("bridge_cues"))
        and residual_eligible
    )
    info["residual_deficit_triggered"] = bridge_gate_bypassed
    info["bridge_gate_bypassed"] = bridge_gate_bypassed
    if bridge_only and not bool(gap.get("bridge_cues")) and not bridge_gate_bypassed:
        info["reason"] = "no_bridge_cues"
        return info
    if require_evidence_gap and not bool(gap.get("answerability_gap")):
        info["reason"] = "current_evidence_sufficient"
        return info

    compiled_slot_contracts = (
        residual_slot_contracts
        if bridge_gate_bypassed or catalog_guided_executor
        else _slot_contracts(
            question,
            planned_retrieval,
            typed_plan=typed_residual_plan,
            catalog_guided=catalog_guided_executor,
        )
    )
    if plan_stage_gate and compiled_slot_contracts is not residual_slot_contracts:
        compiled_slot_contracts = _annotate_slot_plan_stages(
            question,
            compiled_slot_contracts,
        )
    slot_contracts = [
        slot
        for slot in compiled_slot_contracts
        if not typed_residual_plan or bool(slot.get("join_executable"))
    ]
    info["slot_contracts"] = _slot_contract_diagnostics(compiled_slot_contracts)
    info["atomic_slot_count"] = sum(
        bool(slot.get("materialization_safe")) for slot in compiled_slot_contracts
    )
    info["unsafe_slot_count"] = len(compiled_slot_contracts) - int(info["atomic_slot_count"])
    if typed_residual_plan and not slot_contracts:
        info["reason"] = "no_executable_residual_slot"
        return info
    endpoints = _active_entity_endpoints(
        question,
        top_k_facts,
        candidate_facts,
        max_active_facts,
        slot_contracts,
        lineage_constrained,
        max_lineage_hops,
        require_atomic_slot,
    )
    info["active_endpoint_count"] = len(endpoints)
    if not endpoints:
        if catalog_guided_executor:
            info["executor_state"] = "typed_deficit_without_join_key"
            info["operator_selection_reason"] = "no_bound_entity_endpoint"
        if catalog_guided_executor and operator_faithful_executor:
            retrieval = _run_operator_retrieval(
                "text_retrieval",
                compiled_slot_contracts,
                operator_retriever,
                doc_ids=[int(doc_id) for doc_id in doc_ids],
                get_doc_content=get_doc_content,
                qa_top_k=qa_top_k,
                candidate_budget=query_candidate_budget,
                validate_completion=validate_operator_completion,
                validated_top_k=validated_dense_top_k,
                plan_stage_gate=plan_stage_gate,
            )
            ranked = retrieval["candidates"]
            info.update({
                "applied": bool(ranked),
                "reason": (
                    "endpoint_fallback_retrieval_candidates_generated"
                    if ranked else "no_active_entity_endpoint"
                ),
                "physical_operator": "dense_endpoint_fallback",
                "operator_complete": bool(ranked),
                "operator_execution_succeeded": bool(
                    retrieval["raw_candidate_count"]
                ),
                "operator_required_binding_count": 1,
                "operator_bound_binding_count": int(bool(ranked)),
                "operator_request_count": len(retrieval["requests"]),
                "operator_requests": retrieval["requests"],
                "operator_retrieval_candidate_count": len(ranked),
                "operator_retrieval_outside_top_k_count": int(
                    retrieval["outside_top_k"]
                ),
                "operator_validation_candidate_count": int(
                    retrieval["raw_candidate_count"]
                ),
                "operator_validation_passed_count": int(
                    retrieval["validation_passed_count"]
                ),
                "operator_validation_reject_reason_counts": dict(
                    retrieval["validation_reject_reason_counts"]
                ),
                "candidate_count": len(ranked),
                "candidate_items": ranked,
                "operator_final_binding_count": sum(
                    str(row.get("slot_plan_stage") or "") == "final_answer"
                    for row in ranked
                ),
                "operator_intermediate_binding_count": sum(
                    str(row.get("slot_plan_stage") or "") == "intermediate_binding"
                    for row in ranked
                ),
                "operator_support_only_count": sum(
                    str(row.get("slot_plan_stage") or "") == "support_only"
                    for row in ranked
                ),
                "operator_requires_followup": bool(
                    ranked and not retrieval["complete"]
                ),
            })
            return info
        info["reason"] = "no_active_entity_endpoint"
        return info

    rank_map = {int(doc_id): rank for rank, doc_id in enumerate(doc_ids)}
    current_top = {int(doc_id) for doc_id in doc_ids[: max(1, int(qa_top_k))]}
    profile = _relation_profile(question)
    tuple_stats = ((relation_index or {}).get("fact_reliability") or {}).get("tuple_stats") or {}
    predicate_catalog = (relation_index or {}).get("predicate_catalog") or {}
    active_schemas: Set[Tuple[str, str, str]] = {
        endpoint["active_schema"] for endpoint in endpoints if endpoint.get("active_schema") is not None
    }
    rejects: Counter = Counter()
    per_doc: Dict[Any, Dict[str, Any]] = {}
    probed_entities = set()
    raw_fact_ids = set()
    catalog_lookup_keys = set()
    catalog_estimated_scan_rows = 0
    selective_global_access_raw_count = 0

    for endpoint in endpoints:
        entity_key = str(endpoint["entity_key"])
        dependency_binding = _dependency_binding(endpoint)
        if not dependency_binding["safe"]:
            rejects.update(
                f"dependency_binding:{reason}"
                for reason in dependency_binding["reasons"]
            )
            continue
        item_ids = list(entity_to_items.get(entity_key) or [])
        if not item_ids:
            rejects["entity_key_not_indexed"] += 1
            continue
        expected_signature = str(endpoint.get("slot_relation_signature") or "")
        exact_signature_item_ids = set()
        compatible_signature_item_ids = set()
        if typed_residual_plan and expected_signature:
            for raw_item_id in item_ids:
                try:
                    signature_item = items[int(raw_item_id)]
                except (IndexError, TypeError, ValueError):
                    continue
                signature_answer_side = _candidate_answer_side(signature_item, entity_key)
                if signature_answer_side is None:
                    continue
                signature_answer_type = _entity_type(
                    signature_answer_side.get("answer_value")
                )
                candidate_signature = _relation_signature(
                    str(signature_item.get("relation") or ""),
                    set(signature_item.get("relation_tokens") or set()),
                    {signature_answer_type},
                )
                if candidate_signature == expected_signature:
                    exact_signature_item_ids.add(int(raw_item_id))
                elif signature_compatibility_fallback:
                    compatibility = _relation_signature_compatibility(
                        expected_signature,
                        candidate_signature,
                        candidate_answer_type=signature_answer_type,
                        expected_answer_types=set(
                            endpoint.get("slot_answer_types") or []
                        ),
                    )
                    if compatibility["compatible"]:
                        compatible_signature_item_ids.add(int(raw_item_id))
        access_signature_item_ids = (
            exact_signature_item_ids | compatible_signature_item_ids
        )
        exact_signature_fanout = len(exact_signature_item_ids)
        access_signature_fanout = len(access_signature_item_ids)
        if (
            catalog_guided_executor
            and expected_signature in _GENERIC_RELATION_SIGNATURES
            and access_signature_fanout > int(generic_signature_max_fanout)
        ):
            rejects["generic_signature_fanout_too_high"] += 1
            continue
        if max_entity_fanout > 0 and len(item_ids) > int(max_entity_fanout):
            if (
                typed_residual_plan
                and access_signature_item_ids
                and access_signature_fanout <= int(max_exact_signature_fanout)
            ):
                item_ids = sorted(access_signature_item_ids)
            else:
                rejects["entity_fanout_too_high"] += 1
                continue
        if catalog_guided_executor:
            catalog_estimated_scan_rows += len(item_ids)
        probed_entities.add(entity_key)
        for item_id in item_ids:
            try:
                item_id = int(item_id)
                item = items[item_id]
            except (IndexError, TypeError, ValueError):
                rejects["invalid_fact_row"] += 1
                continue
            raw_fact_ids.add(item_id)
            answer_side = _candidate_answer_side(item, entity_key)
            if answer_side is None:
                rejects["join_direction_unresolved"] += 1
                continue
            if tuple(item.get("triple") or ()) == tuple(endpoint.get("active_triple") or ()):
                rejects["active_fact_self_join"] += 1
                continue
            relation_tokens = set(item.get("relation_tokens") or set())
            slot_gap = {
                "query_relation_families": endpoint.get("slot_relation_families") or [],
                "query_polarity": endpoint.get("slot_polarity") or [],
            }
            family_relation = _relation_family_score(relation_tokens, slot_gap)
            direction = f"{answer_side['candidate_join_role']}_to_{answer_side['answer_role']}"
            answer_value = str(answer_side["answer_value"] or "")
            answer_type = _entity_type(answer_value)
            answer_types = set(endpoint.get("slot_answer_types") or [])
            candidate_signature = (
                _relation_signature(
                    str(item.get("relation") or ""),
                    relation_tokens,
                    {answer_type},
                )
                if typed_residual_plan else ""
            )
            predicate_contract = _predicate_contract(
                relation_tokens,
                {
                    "relation_tokens": endpoint.get("slot_relation_tokens") or [],
                    "polarity": endpoint.get("slot_polarity") or [],
                    "relation_signature": expected_signature,
                },
                direction,
                candidate_signature=candidate_signature,
                strict_signature=typed_residual_plan,
                allow_signature_compatibility=bool(
                    operator_faithful_executor
                    and signature_compatibility_fallback
                ),
                candidate_answer_type=answer_type,
                expected_answer_types=answer_types,
            )
            if (
                typed_residual_plan
                and expected_signature
                and not (
                    predicate_contract["signature_match"]
                    or predicate_contract["signature_compatible"]
                )
            ):
                rejects["hard_relation_signature_mismatch"] += 1
                continue
            relation_overlap = (
                float(predicate_contract["score"])
                if typed_residual_plan and expected_signature
                else max(
                    float(predicate_contract["score"]),
                    0.5 * float(family_relation),
                )
            )
            if relation_overlap < float(min_relation_overlap):
                rejects["predicate_mismatch"] += 1
                continue

            answer_type_match = _object_matches_answer_type(answer_value, answer_types, "")
            if answer_types and not answer_type_match:
                rejects["answer_type_mismatch"] += 1
                continue

            doc_id = int(item.get("doc_idx"))
            retrieval_pool_member = doc_id in rank_map
            fallback_rank = (
                int(max_original_rank) + max(1, len(doc_ids))
                if typed_residual_plan else max(1, int(max_original_rank))
            )
            rank = int(rank_map.get(doc_id, fallback_rank))
            selective_exact_access = bool(
                typed_residual_plan
                and (
                    predicate_contract["signature_match"]
                    or predicate_contract["signature_compatible"]
                )
                and access_signature_fanout > 0
                and access_signature_fanout <= int(max_exact_signature_fanout)
            )
            if rank > int(max_original_rank) and not selective_exact_access:
                rejects["original_rank_too_deep"] += 1
                continue
            if selective_exact_access and (
                rank > int(max_original_rank) or not retrieval_pool_member
            ):
                selective_global_access_raw_count += 1

            try:
                content = str(get_doc_content(doc_id) or "")
            except Exception:
                rejects["doc_lookup_failed"] += 1
                continue
            if not content:
                rejects["empty_doc"] += 1
                continue
            structured_branch = _structured_branch_binding(
                str(endpoint.get("entity") or ""),
                content.split("\n", 1)[0].strip(),
                content,
                {
                    "named_anchor_phrases": list(
                        endpoint.get("slot_named_anchor_phrases") or []
                    ),
                    "anchor_tokens": set(
                        endpoint.get("slot_anchor_tokens") or []
                    ),
                },
            )
            recovered_answer_surface = _recover_answer_surface(
                content,
                answer_value,
            )
            recovered_surface_grounded = bool(
                recovered_answer_surface
                and _accent_folded_surface(recovered_answer_surface)
                in _accent_folded_surface(content)
            )
            query_overlap = len(_tokens(content[:2200]) & set(profile.get("query_tokens") or [])) / max(
                1,
                len(set(profile.get("query_tokens") or [])),
            )
            stats = tuple_stats.get(str(item.get("tuple_key") or "")) or {}
            predicate_key = str(item.get("relation_key") or "")
            catalog_entry = predicate_catalog.get(predicate_key) or {}
            if catalog_guided_executor and predicate_key:
                catalog_lookup_keys.add(predicate_key)
            catalog_row_count = int(catalog_entry.get("row_count", 0) or 0)
            catalog_doc_count = int(catalog_entry.get("doc_count", 0) or 0)
            catalog_selectivity = 1.0 / (1.0 + math.log1p(max(0, catalog_row_count)))
            provenance_score = _clamp(
                0.65 * min(1.0, float(stats.get("doc_count", 0) or 0) / 2.0)
                + 0.35 * min(1.0, float(stats.get("triple_count", 0) or 0) / 3.0)
            )
            already_visible = doc_id in current_top
            novelty = 0.0 if already_visible else 1.0
            rank_prior = 1.0 / (1.0 + math.log1p(rank + 1))
            cross_schema = item.get("signature") not in active_schemas
            if typed_residual_plan and catalog_guided_executor:
                signature_accepted = bool(
                    predicate_contract["signature_match"]
                    or predicate_contract["signature_compatible"]
                )
                score = _clamp(
                    0.22
                    + 0.40 * relation_overlap
                    + 0.14 * (1.0 if answer_type_match else 0.0)
                    + 0.08 * provenance_score
                    + 0.04 * query_overlap
                    + 0.03 * rank_prior
                    + 0.02 * float(endpoint.get("slot_anchor_overlap", 0.0) or 0.0)
                    + 0.07 * catalog_selectivity
                )
                risk = _clamp(
                    0.40 * (1.0 - relation_overlap)
                    + 0.20 * (0.0 if answer_type_match else 1.0)
                    + 0.08 * (1.0 - provenance_score)
                    + 0.08 * (0.0 if selective_exact_access or retrieval_pool_member else 1.0)
                    + 0.18 * (0.0 if signature_accepted else 1.0)
                    + 0.06 * (1.0 - catalog_selectivity)
                )
            elif typed_residual_plan:
                score = _clamp(
                    0.24
                    + 0.42 * relation_overlap
                    + 0.14 * (1.0 if answer_type_match else 0.0)
                    + 0.10 * provenance_score
                    + 0.05 * query_overlap
                    + 0.03 * rank_prior
                    + 0.02 * float(endpoint.get("slot_anchor_overlap", 0.0) or 0.0)
                )
                risk = _clamp(
                    0.42 * (1.0 - relation_overlap)
                    + 0.20 * (0.0 if answer_type_match else 1.0)
                    + 0.10 * (1.0 - provenance_score)
                    + 0.08 * (0.0 if selective_exact_access or retrieval_pool_member else 1.0)
                    + 0.20 * (0.0 if predicate_contract["signature_match"] else 1.0)
                )
            else:
                score = _clamp(
                    0.22
                    + 0.34 * relation_overlap
                    + 0.14 * (1.0 if answer_type_match else 0.0)
                    + 0.10 * provenance_score
                    + 0.08 * novelty
                    + 0.06 * query_overlap
                    + 0.04 * rank_prior
                    + 0.02 * float(endpoint.get("slot_anchor_overlap", 0.0) or 0.0)
                )
                risk = _clamp(
                    0.30 * (1.0 - relation_overlap)
                    + 0.18 * (0.0 if answer_type_match else 1.0)
                    + 0.12 * (1.0 if already_visible else 0.0)
                    + 0.08 * (1.0 - provenance_score)
                    + 0.08 * (0.0 if retrieval_pool_member else 1.0)
                    + 0.24 * (0.0 if predicate_contract["matched"] else 1.0)
                )
            net_utility = score - 0.35 * risk
            if score < float(min_score):
                rejects["score_below_threshold"] += 1
                continue

            candidate = {
                "doc_id": doc_id,
                "title": content.split("\n", 1)[0].strip(),
                "score": round(float(score), 6),
                "risk": round(float(risk), 6),
                "net_utility": round(float(net_utility), 6),
                "original_rank": rank,
                "retrieval_pool_member": bool(retrieval_pool_member),
                "outside_retrieval_pool": not bool(retrieval_pool_member),
                "already_top5": bool(already_visible),
                "entity_bound": True,
                "join_key_exact": True,
                "join_key_overlap": 1.0,
                "active_entity": str(endpoint.get("entity") or ""),
                "active_entity_key": entity_key,
                "active_join_role": str(endpoint.get("active_role") or ""),
                "candidate_join_role": str(answer_side["candidate_join_role"]),
                "answer_role": str(answer_side["answer_role"]),
                "answer_value": answer_value,
                "answer_surface": recovered_answer_surface,
                "answer_surface_recovered": recovered_surface_grounded,
                "answer_key": str(answer_side["answer_key"]),
                "answer_type": answer_type,
                "answer_type_match": bool(answer_type_match),
                "answer_surface_grounded": bool(
                    _surface_contains_value(content, answer_value)
                    or recovered_surface_grounded
                ),
                "predicate": str(item.get("relation") or ""),
                "predicate_key": predicate_key,
                "catalog_row_count": catalog_row_count,
                "catalog_doc_count": catalog_doc_count,
                "catalog_selectivity": round(float(catalog_selectivity), 6),
                "estimated_endpoint_scan_rows": len(item_ids),
                "relation_overlap": round(float(relation_overlap), 6),
                "query_overlap": round(float(query_overlap), 6),
                "provenance_score": round(float(provenance_score), 6),
                "provenance_doc_ids": list(stats.get("doc_ids") or [doc_id])[:8],
                "triple": list(item.get("triple") or []),
                "schema": list(item.get("signature") or []),
                "active_schema": list(endpoint.get("active_schema") or []),
                "cross_schema": bool(cross_schema),
                "direction": direction,
                "lineage_bound": bool(endpoint.get("lineage_bound")),
                "root_entity": str(endpoint.get("root_entity") or ""),
                "root_entity_key": str(endpoint.get("root_entity_key") or ""),
                "root_anchor_score": float(endpoint.get("root_anchor_score", 0.0) or 0.0),
                "lineage_depth": int(endpoint.get("lineage_depth", 0) or 0),
                "lineage_path": list(endpoint.get("lineage_path") or []),
                "slot_id": str(endpoint.get("slot_id") or ""),
                "slot_text": str(endpoint.get("slot_text") or ""),
                "slot_anchor_overlap": float(endpoint.get("slot_anchor_overlap", 0.0) or 0.0),
                "slot_anchor_tokens": list(endpoint.get("slot_anchor_tokens") or []),
                "slot_relation_tokens": list(endpoint.get("slot_relation_tokens") or []),
                "slot_polarity": list(endpoint.get("slot_polarity") or []),
                "predicate_contract_match": bool(predicate_contract["matched"]),
                "predicate_match_mode": str(predicate_contract["mode"]),
                "expected_relation_signature": str(
                    predicate_contract.get("expected_signature") or ""
                ),
                "candidate_relation_signature": str(
                    predicate_contract.get("candidate_signature") or ""
                ),
                "relation_signature_match": bool(
                    predicate_contract.get("signature_match")
                ),
                "relation_signature_compatible": bool(
                    predicate_contract.get("signature_compatible")
                ),
                "relation_signature_compatibility_tier": str(
                    predicate_contract.get("signature_compatibility_tier")
                    or "unknown"
                ),
                "relation_signature_fanout": int(access_signature_fanout),
                "exact_relation_signature_fanout": int(exact_signature_fanout),
                "compatible_relation_signature_fanout": int(
                    len(compatible_signature_item_ids)
                ),
                "selective_access_eligible": bool(selective_exact_access),
                "selective_global_access": bool(
                    selective_exact_access and not retrieval_pool_member
                ),
                "direct_predicate_overlap": round(float(predicate_contract["direct_overlap"]), 6),
                "direct_predicate_tokens": list(predicate_contract["direct_tokens"]),
                "discriminative_predicate_tokens": list(
                    predicate_contract["discriminative_direct_tokens"]
                ),
                "expected_discriminative_predicate_tokens": list(
                    predicate_contract["discriminative_expected_tokens"]
                ),
                "expected_predicate_concepts": list(predicate_contract["expected_concepts"]),
                "candidate_predicate_concepts": list(predicate_contract["candidate_concepts"]),
                "dependency_binding_safe": bool(dependency_binding["safe"]),
                "dependency_binding_reasons": list(dependency_binding["reasons"]),
                "dependency_root_entity_key": str(
                    dependency_binding["root_entity_key"]
                ),
                "dependency_terminal_entity_key": str(
                    dependency_binding["terminal_entity_key"]
                ),
                "dependency_path_entity_keys": list(
                    dependency_binding["path_entity_keys"]
                ),
                "dependency_hops": int(dependency_binding["dependency_hops"]),
                "slot_source": str(endpoint.get("slot_source") or "unknown"),
                "slot_hop_index": int(endpoint.get("slot_hop_index", -1)),
                "slot_target_predicate_concepts": list(
                    endpoint.get("slot_target_predicate_concepts") or []
                ),
                "slot_expected_value_type": str(
                    endpoint.get("slot_expected_value_type") or "unknown"
                ),
                "slot_direction": str(
                    endpoint.get("slot_direction") or "bound_entity_to_value"
                ),
                "slot_relation_arity": int(
                    endpoint.get("slot_relation_arity", 1) or 1
                ),
                "slot_execution_operator": str(
                    endpoint.get("slot_execution_operator") or "lookup"
                ),
                "slot_join_executable": bool(
                    endpoint.get("slot_join_executable")
                ),
                "slot_materialization_safe": bool(
                    endpoint.get("slot_materialization_safe")
                ),
                "slot_plan_stage": str(
                    endpoint.get("slot_plan_stage") or "unclassified"
                ),
                "slot_branch_id": str(endpoint.get("slot_branch_id") or ""),
                "branch_binding_passed": bool(structured_branch["passed"]),
                "branch_binding_reason": str(structured_branch["reason"]),
                "branch_anchor_phrase": str(
                    structured_branch.get("anchor_phrase") or ""
                ),
                "slot_final_answer_type": str(
                    endpoint.get("slot_final_answer_type") or "unknown"
                ),
                "slot_expected_type_family": str(
                    endpoint.get("slot_expected_type_family") or "unknown"
                ),
                "materialization_role": (
                    "answer_binding"
                    if str(endpoint.get("slot_plan_stage") or "") == "final_answer"
                    else "intermediate_state"
                ),
                "operator_candidate_kind": "structured_fact_join",
                "slot_unsafe_reasons": list(
                    endpoint.get("slot_unsafe_reasons") or []
                ),
                "slot_meta_negative": bool(endpoint.get("slot_meta_negative")),
            }
            per_doc_key: Any = (
                (doc_id, str(candidate.get("slot_id") or ""))
                if catalog_guided_executor else doc_id
            )
            current = per_doc.get(per_doc_key)
            candidate_key = (
                bool(candidate.get("relation_signature_match"))
                    if typed_residual_plan else True,
                bool(candidate.get("relation_signature_compatible"))
                    if typed_residual_plan else False,
                bool(candidate.get("predicate_contract_match")),
                float(candidate.get("relation_overlap", 0.0)),
                float(candidate.get("catalog_selectivity", 0.0))
                    if catalog_guided_executor else 0.0,
                float(candidate.get("provenance_score", 0.0)),
                float(candidate.get("net_utility", 0.0)),
                -int(candidate.get("original_rank", 0)),
            )
            current_key = (
                bool(current.get("relation_signature_match"))
                    if typed_residual_plan and current is not None else True,
                bool(current.get("relation_signature_compatible"))
                    if typed_residual_plan and current is not None else False,
                bool(current.get("predicate_contract_match"))
                    if current is not None else False,
                float(current.get("relation_overlap", 0.0))
                    if current is not None else 0.0,
                float(current.get("catalog_selectivity", 0.0))
                    if catalog_guided_executor and current is not None else 0.0,
                float(current.get("provenance_score", 0.0))
                    if current is not None else 0.0,
                float(current.get("net_utility", 0.0))
                    if current is not None else 0.0,
                -int(current.get("original_rank", 0))
                    if current is not None else 0,
            )
            if current is None or candidate_key > current_key:
                per_doc[per_doc_key] = candidate

    all_ranked = sorted(
        per_doc.values(),
        key=lambda row: (
            bool(row.get("relation_signature_match"))
                if typed_residual_plan else True,
            bool(row.get("relation_signature_compatible"))
                if typed_residual_plan else False,
            bool(row.get("predicate_contract_match")),
            float(row["relation_overlap"]),
            float(row.get("catalog_selectivity", 0.0))
                if catalog_guided_executor else 0.0,
            float(row.get("provenance_score", 0.0)),
            float(row.get("slot_anchor_overlap", 0.0)),
            float(row["net_utility"]),
            -int(row["original_rank"]),
        ),
        reverse=True,
    )
    validation_rejects: Counter = Counter()
    if validate_operator_completion and operator_faithful_executor:
        validated_ranked = []
        for row in all_ranked:
            validation = _validate_structured_operator_candidate(row)
            row = {
                **row,
                "operator_validation_passed": bool(validation["passed"]),
                "operator_validation_score": float(validation["score"]),
                "operator_validation_reason": str(validation["reason"]),
                "operator_validation_checks": dict(validation["checks"]),
            }
            if validation["passed"]:
                validated_ranked.append(row)
            else:
                validation_rejects.update(
                    reason
                    for reason, passed in validation["checks"].items()
                    if not passed
                )
        operator_ranked = validated_ranked
    else:
        operator_ranked = all_ranked
    operator_execution = None
    if catalog_guided_executor and operator_faithful_executor:
        operator_execution = _select_operator_candidates(
            executor_route["operator"],
            operator_ranked,
            compiled_slot_contracts,
            candidate_top_k=candidate_top_k,
            query_candidate_budget=query_candidate_budget,
            max_candidates_per_slot=max_candidates_per_slot,
            require_final_stage=plan_stage_gate,
        )
        ranked = operator_execution["candidates"]
        rejects["operator_candidate_budget_exhausted"] += int(
            operator_execution["rejected_by_budget"]
        )
    elif catalog_guided_executor:
        per_slot_counts: Counter = Counter()
        ranked = []
        query_limit = min(
            max(0, int(candidate_top_k)),
            max(0, int(query_candidate_budget)),
        )
        if query_limit > 0:
            for row in all_ranked:
                slot_id = str(row.get("slot_id") or "unknown")
                if per_slot_counts[slot_id] >= max(1, int(max_candidates_per_slot)):
                    rejects["slot_candidate_budget_exhausted"] += 1
                    continue
                ranked.append(row)
                per_slot_counts[slot_id] += 1
                if len(ranked) >= query_limit:
                    break
    else:
        ranked = all_ranked[: max(0, int(candidate_top_k))]
    selective_global_access_unique_count = sum(
        bool(row.get("selective_global_access")) for row in ranked
    )
    info.update(
        {
            "applied": bool(ranked),
            "reason": "candidates_generated" if ranked else "no_candidate_accepted",
            "probed_entity_count": len(probed_entities),
            "raw_fact_count": len(raw_fact_ids),
            "candidate_count": len(ranked),
            "lineage_bound_candidate_count": sum(
                bool(row.get("lineage_bound")) for row in ranked
            ),
            "predicate_contract_candidate_count": sum(
                bool(row.get("predicate_contract_match")) for row in ranked
            ),
            "dependency_safe_candidate_count": sum(
                bool(row.get("dependency_binding_safe")) for row in ranked
            ),
            "family_only_candidate_count": sum(
                str(row.get("predicate_match_mode") or "") == "family_only"
                for row in ranked
            ),
            "cross_schema_candidate_count": sum(bool(row.get("cross_schema")) for row in ranked),
            "already_visible_count": sum(bool(row.get("already_top5")) for row in ranked),
            "exact_signature_candidate_count": sum(
                bool(row.get("relation_signature_match")) for row in ranked
            ),
            "compatible_signature_candidate_count": sum(
                bool(row.get("relation_signature_compatible")) for row in ranked
            ),
            "hard_signature_reject_count": int(
                rejects.get("hard_relation_signature_mismatch", 0)
            ),
            "selective_global_access_count": int(
                selective_global_access_unique_count
            ),
            "selective_global_access_raw_count": int(
                selective_global_access_raw_count
            ),
            "selective_global_access_unique_count": int(
                selective_global_access_unique_count
            ),
            "catalog_lookup_count": len(catalog_lookup_keys),
            "catalog_estimated_scan_rows": int(catalog_estimated_scan_rows),
            "candidate_items": ranked,
            "reject_reason_counts": dict(rejects),
            "operator_execution_succeeded": bool(all_ranked),
            "operator_validation_candidate_count": len(all_ranked),
            "operator_validation_passed_count": len(operator_ranked),
            "operator_validation_reject_reason_counts": dict(
                validation_rejects
            ),
            "operator_final_binding_count": sum(
                str(row.get("slot_plan_stage") or "") == "final_answer"
                for row in ranked
            ),
            "operator_intermediate_binding_count": sum(
                str(row.get("slot_plan_stage") or "") == "intermediate_binding"
                for row in ranked
            ),
            "operator_support_only_count": sum(
                str(row.get("slot_plan_stage") or "") == "support_only"
                for row in ranked
            ),
        }
    )
    if operator_execution is not None:
        info.update({
            "physical_operator": operator_execution["physical_operator"],
            "operator_complete": bool(operator_execution["complete"]),
            "operator_required_binding_count": int(
                operator_execution["required_bindings"]
            ),
            "operator_bound_binding_count": int(
                operator_execution["bound_bindings"]
            ),
            "operator_requires_followup": bool(
                ranked and not operator_execution["complete"]
            ),
        })
    return info


def build_entity_predicate_fact_join_candidates(*args, **kwargs) -> Dict[str, Any]:
    """Measure the fact-store access path independently from end-to-end QA time."""
    started = time.perf_counter()
    info = _build_entity_predicate_fact_join_candidates_impl(*args, **kwargs)
    info["operator_latency_ms"] = round(
        1000.0 * (time.perf_counter() - started),
        6,
    )
    return info
