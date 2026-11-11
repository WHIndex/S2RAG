import hashlib
import json
import math
import os
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from .learned_cost_optimizer_utils import (
    annotate_slate_relative_features,
    load_learned_cost_optimizer_model,
    score_candidate_with_counterfactual_slate_model,
    score_candidate_with_learned_cost_model,
)
from .cost_based_fact_join_plan_utils import optimize_cost_based_fact_join_plan
from .planned_retrieval_state_utils import planned_retrieval_terminal_state


_COST_ADMISSION_POLICIES = {
    "learned",
    "learned_veto",
    "rule",
    "fixed_order",
    "always_on",
    "counterfactual_drop",
    "counterfactual_slate_veto",
}


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "with",
}

_HIGH_RISK_PLANNED_REASONS = {
    "max_rounds_reached",
    "verifier_insufficient_no_retrieval",
    "routed_long_answer_or_narrative",
    "routed_long_context_evidence",
}

_DEFICIT_PLANNED_REASONS = {
    "max_rounds_reached",
    "verifier_insufficient_no_retrieval",
}

_DEFICIT_STATE_REASONS = {
    "relation_join_gap",
    "query_coverage_gap",
    "missing_query_terms",
}

_PROTECTED_EXTRA_SOURCES = {
    "planned_retrieval",
    "dual_channel_projection",
    "query_schema_memory",
    "relation_fact_join",
    "relation_fact_table",
    "support_set_completion",
    "path_evidence",
    "structured_chain",
    "multigranular_memory",
}

_SHADOW_SAFE_COMPANION_SOURCES = {
    "path_evidence",
}

_SHADOW_UNSAFE_COMPANION_SOURCES = {
    "planned_retrieval",
    "dual_channel_projection",
    "relation_fact_join",
    "support_set_completion",
    "structured_chain",
    "multigranular_memory",
    "extra_evidence",
}

_CALIBRATED_PORTFOLIO_COMPANION_SOURCES = {
    "planned_retrieval",
    "path_evidence",
}

_DATE_VALUE_RE = re.compile(
    r"\b(?:1[5-9]\d{2}|20\d{2}|2100|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
    r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
_NUMBER_VALUE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?(?:\s*(?:%|percent|million|billion|thousand))?\b",
    re.IGNORECASE,
)
_DATE_REQUEST_RE = re.compile(
    r"^\s*(?:when\b|on\s+what\s+date\b|what\s+date\b)|"
    r"\b(?:what|which|in\s+what|in\s+which)\s+year\b",
    re.IGNORECASE,
)
_NUMBER_REQUEST_RE = re.compile(
    r"^\s*(?:how\s+many\b|how\s+much\b|how\s+old\b|how\s+long\b|"
    r"what\s+(?:number|population|percentage|percent|age|total|count)\b|"
    r"which\s+(?:number|population|percentage|percent|age|total|count)\b)",
    re.IGNORECASE,
)
_PERCENT_REQUEST_RE = re.compile(
    r"\b(?:percent|percentage|per\s+cent|share|rate)\b|%",
    re.IGNORECASE,
)
_PERCENT_VALUE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|percent(?:age)?|per\s+cent)\b",
    re.IGNORECASE,
)
_MONTH_REQUEST_RE = re.compile(
    r"\b(?:what|which|in\s+what|in\s+which)\s+month\b",
    re.IGNORECASE,
)
_MONTH_VALUE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
_CAUSE_REQUEST_RE = re.compile(
    r"\b(?:cause\s+of\s+death|cause|reason|die\s+from|died\s+from|death\s+from)\b",
    re.IGNORECASE,
)
_OPEN_ENDED_REQUEST_RE = re.compile(
    r"^\s*(?:why\b|how\b(?!\s+(?:many|much|old|long)\b))|"
    r"\b(?:describe|explain|summari[sz]e|what\s+happens|what\s+happened)\b",
    re.IGNORECASE,
)

_SLOT_PREDICATE_TERMS = {
    "age",
    "birth",
    "born",
    "burial",
    "buried",
    "capital",
    "cause",
    "caused",
    "country",
    "creator",
    "date",
    "death",
    "die",
    "died",
    "directed",
    "director",
    "father",
    "founded",
    "founder",
    "location",
    "mother",
    "nationality",
    "place",
    "percent",
    "percentage",
    "population",
    "producer",
    "spouse",
    "year",
}

_SLOT_NON_ANCHOR_TERMS = _SLOT_PREDICATE_TERMS | {
    "answer",
    "earlier",
    "first",
    "later",
    "missing",
    "needed",
    "older",
    "required",
    "value",
    "younger",
}

_SCHEMA_JOIN_GENERIC_TERMS = _SLOT_NON_ANCHOR_TERMS | {
    "actor",
    "author",
    "belong",
    "body",
    "book",
    "committee",
    "conference",
    "control",
    "controlled",
    "country",
    "date",
    "determines",
    "director",
    "draft",
    "drafted",
    "film",
    "first",
    "gain",
    "group",
    "house",
    "last",
    "majority",
    "model",
    "movie",
    "organization",
    "party",
    "person",
    "place",
    "published",
    "released",
    "representatives",
    "rules",
    "senate",
    "specific",
    "team",
    "timeline",
    "work",
}

_SLOT_PREDICATE_CANONICAL = {
    "birth": "birth",
    "born": "birth",
    "burial": "burial",
    "buried": "burial",
    "creator": "create",
    "created": "create",
    "cause": "cause",
    "caused": "cause",
    "death": "death",
    "die": "death",
    "died": "death",
    "directed": "direct",
    "director": "direct",
    "founded": "found",
    "founder": "found",
    "date": "temporal",
    "percent": "number",
    "percentage": "number",
    "year": "temporal",
}

_SCHEMA_TUPLE_TERMINAL_RELATION_TERMS = _SLOT_PREDICATE_TERMS | {
    "actor",
    "actress",
    "brother",
    "capitol",
    "child",
    "children",
    "city",
    "composer",
    "county",
    "daughter",
    "girlfriend",
    "grandfather",
    "grandmother",
    "grandparent",
    "headquarters",
    "husband",
    "law",
    "located",
    "meaning",
    "played",
    "region",
    "screenwriter",
    "sister",
    "son",
    "state",
    "wife",
}

_SCHEMA_TUPLE_TERMINAL_ROLES = {
    "answer",
    "answer_binding",
    "final_answer",
    "terminal",
    "terminal_answer",
    "terminal_tuple",
}

_SCHEMA_TUPLE_SUPPORT_ROLES = {
    "bridge",
    "bridge_only",
    "bridge_support",
    "intermediate",
    "support",
    "support_only",
}

_SCHEMA_TUPLE_TYPE_ALIASES = {
    "actor": "person",
    "actress": "person",
    "author": "person",
    "city": "location",
    "country": "location",
    "county": "location",
    "date": "date",
    "entity": "entity",
    "film": "work",
    "human": "person",
    "location": "location",
    "movie": "work",
    "number": "number",
    "organization": "organization",
    "person": "person",
    "place": "location",
    "province": "location",
    "region": "location",
    "state": "location",
    "temporal": "date",
    "time": "date",
    "unknown": "unknown",
    "work": "work",
    "year": "date",
}


def _tokens(text: Any) -> set:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _capitalized_anchor_groups(text: str) -> List[set]:
    groups: List[set] = []
    current: List[str] = []
    connectors = {"and", "de", "del", "la", "of", "the", "van", "von"}
    leading = {"are", "did", "does", "how", "is", "what", "when", "where", "which", "who"}
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*", str(text or "")):
        lowered = token.lower().strip(".-'")
        is_capitalized = bool(token[:1].isupper())
        if is_capitalized or current and lowered in connectors:
            current.append(token)
            continue
        if current:
            group = _tokens(" ".join(current)) - _SCHEMA_JOIN_GENERIC_TERMS
            if group and not group <= leading:
                groups.append(group)
            current = []
    if current:
        group = _tokens(" ".join(current)) - _SCHEMA_JOIN_GENERIC_TERMS
        if group and not group <= leading:
            groups.append(group)
    return groups


def _schema_join_anchor_groups(
    question: str,
    lineage_context: Dict[str, Any],
) -> List[set]:
    groups = _capitalized_anchor_groups(question)
    uncovered = set(lineage_context.get("required_uncovered_slot_ids") or [])
    for slot in lineage_context.get("required_slots") or []:
        slot_id = str(slot.get("slot_id") or "")
        if uncovered and slot_id not in uncovered:
            continue
        slot_groups = _capitalized_anchor_groups(str(slot.get("text") or ""))
        groups.extend(slot_groups)
        fallback = {
            str(token)
            for token in (slot.get("anchor_tokens") or [])
            if str(token) not in _SCHEMA_JOIN_GENERIC_TERMS
            and not str(token).isdigit()
        }
        if fallback:
            groups.append(fallback)
    if not groups:
        fallback = {
            token
            for token in _tokens(question)
            if token not in _SCHEMA_JOIN_GENERIC_TERMS and not token.isdigit()
        }
        if 0 < len(fallback) <= 6:
            groups.append(fallback)

    deduplicated: List[set] = []
    seen = set()
    for group in groups:
        key = tuple(sorted(group))
        if not key or key in seen:
            continue
        seen.add(key)
        deduplicated.append(set(group))
    return deduplicated


def _schema_join_anchor_binding(
    item: Dict[str, Any],
    anchor_groups: List[set],
) -> Dict[str, Any]:
    active_tokens = {
        str(token)
        for token in (
            item.get("active_join_key_tokens")
            or item.get("matched_join_tokens")
            or []
        )
        if str(token)
    }
    best = {
        "bound": False,
        "score": 0.0,
        "matched_tokens": [],
        "anchor_tokens": [],
        "active_key_tokens": sorted(active_tokens),
    }
    for group in anchor_groups:
        intersection = set(group) & active_tokens
        coverage = len(intersection) / float(max(1, len(group)))
        overlap = _jaccard(set(group), active_tokens)
        bound = bool(
            group <= active_tokens
            or len(group) == 1 and len(intersection) == 1
            or len(intersection) >= 2 and max(coverage, overlap) >= 0.50
        )
        score = max(coverage, overlap)
        candidate = {
            "bound": bound,
            "score": _round_float(score),
            "matched_tokens": sorted(intersection),
            "anchor_tokens": sorted(group),
            "active_key_tokens": sorted(active_tokens),
        }
        if (int(bound), score, len(intersection)) > (
            int(bool(best["bound"])),
            float(best["score"]),
            len(best["matched_tokens"]),
        ):
            best = candidate
    return best


def _canonical_predicate_tokens(tokens: Any, surface: Any = "") -> set:
    canonical = {
        _SLOT_PREDICATE_CANONICAL.get(str(token), str(token))
        for token in (tokens or [])
        if str(token) in _SLOT_PREDICATE_TERMS
    }
    text = str(surface or "")
    if _DATE_VALUE_RE.search(text):
        canonical.add("temporal")
    if _NUMBER_VALUE_RE.search(text):
        canonical.add("number")
    return canonical


def _comparison_branch_id(slot: Dict[str, Any]) -> str:
    anchors = sorted(set(slot.get("anchor_tokens") or []))
    return "branch:" + "+".join(anchors) if anchors else ""


def _ensure_comparison_branches(lineage_context: Dict[str, Any]) -> Dict[str, Any]:
    slots = list(lineage_context.get("required_slots") or [])
    if not bool(lineage_context.get("comparison_query")):
        return lineage_context
    for slot in slots:
        if not str(slot.get("branch_id") or ""):
            slot["branch_id"] = _comparison_branch_id(slot)
    lineage_context["required_slots"] = slots
    lineage_context["required_branch_ids"] = sorted(
        {
            str(slot.get("branch_id") or "")
            for slot in slots
            if str(slot.get("branch_id") or "")
        }
    )
    return lineage_context


def _title(doc: Any) -> str:
    return str(doc or "").split("\n", 1)[0].strip()


def _doc_key(doc: Any) -> str:
    return re.sub(r"\s+", " ", str(doc or "")).strip().lower()


def _split_projection_sentences(text: str) -> List[str]:
    body = str(text or "").split("\n", 1)[-1]
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", body)
        if sentence.strip()
    ]


def _normalize_schema_tuple_type(label: Any) -> str:
    tokens = _tokens(label)
    if not tokens:
        return "unknown"
    if tokens & {"date", "year", "temporal", "time"}:
        return "date"
    if tokens & {"number", "count", "population", "percent", "percentage"}:
        return "number"
    if tokens & {
        "city",
        "country",
        "county",
        "location",
        "place",
        "province",
        "region",
        "state",
    }:
        return "location"
    if tokens & {"actor", "actress", "author", "human", "person", "people"}:
        return "person"
    if tokens & {"film", "movie", "song", "work", "book"}:
        return "work"
    for token in tokens:
        if token in _SCHEMA_TUPLE_TYPE_ALIASES:
            return _SCHEMA_TUPLE_TYPE_ALIASES[token]
    return "entity" if "entity" in tokens else "unknown"


def _schema_tuple_schema_value_type(item: Dict[str, Any]) -> str:
    schema = item.get("schema") or []
    if not isinstance(schema, (list, tuple)) or len(schema) < 3:
        return "unknown"
    direction = str(item.get("direction") or "").strip().lower()
    labels: List[Any] = []
    if direction == "object_to_subject":
        labels.extend([schema[0], schema[2]])
    else:
        labels.extend([schema[2], schema[0]])
    for label in labels:
        normalized = _normalize_schema_tuple_type(label)
        if normalized not in {"unknown", "entity"}:
            return normalized
    return "entity" if any(_normalize_schema_tuple_type(label) == "entity" for label in labels) else "unknown"


def _infer_schema_tuple_value_type(item: Dict[str, Any], value: str) -> str:
    value_text = str(value or "")
    if _DATE_VALUE_RE.search(value_text):
        return "date"
    if _NUMBER_VALUE_RE.search(value_text):
        return "number"
    labels = [
        item.get("slot_expected_value_type"),
        item.get("slot_final_answer_type"),
        item.get("slot_expected_type_family"),
        item.get("answer_type"),
        _schema_tuple_schema_value_type(item),
    ]
    for label in labels:
        normalized = _normalize_schema_tuple_type(label)
        if normalized not in {"unknown", "entity"}:
            return normalized
    return "entity" if any(_normalize_schema_tuple_type(label) == "entity" for label in labels) else "unknown"


def _schema_tuple_value_matches_answer_slot(answer_slot: str, value_type: str) -> bool:
    slot = _normalize_schema_tuple_type(answer_slot)
    value = _normalize_schema_tuple_type(value_type)
    if slot in {"unknown", "entity"}:
        return value not in {"unknown"}
    if slot == "work":
        return value in {"work", "entity"}
    return slot == value


def _expand_relation_tokens(tokens: set) -> set:
    expanded = set(tokens)
    for token in list(tokens):
        canonical = _SLOT_PREDICATE_CANONICAL.get(token)
        if canonical:
            expanded.add(canonical)
    return expanded


def _schema_tuple_question_relation_tokens(question: str) -> set:
    return _expand_relation_tokens(
        _tokens(question) & _SCHEMA_TUPLE_TERMINAL_RELATION_TERMS
    )


def _schema_tuple_candidate_relation_tokens(item: Dict[str, Any]) -> set:
    triple = item.get("triple") or []
    parts: List[str] = [
        str(item.get("predicate") or ""),
        str(item.get("candidate_relation_signature") or ""),
        " ".join(str(token) for token in item.get("direct_predicate_tokens") or []),
        " ".join(
            str(token) for token in item.get("discriminative_predicate_tokens") or []
        ),
    ]
    if isinstance(triple, (list, tuple)) and len(triple) >= 2:
        parts.append(str(triple[1] or ""))
    schema = item.get("schema") or []
    if isinstance(schema, (list, tuple)) and len(schema) >= 2:
        parts.append(str(schema[1] or ""))
    return _expand_relation_tokens(_tokens(" ".join(parts)))


def _schema_tuple_is_boolean_or_comparison_question(
    question: str,
    lineage_context: Dict[str, Any],
) -> bool:
    normalized = str(question or "").strip().lower()
    if bool(lineage_context.get("comparison_query")):
        return True
    if re.match(r"^(?:are|is|was|were|do|does|did|has|have|had|can|could|should|would)\b", normalized):
        return True
    return bool(
        re.search(
            r"\b(?:both|same|different|earlier|later|older|younger|first|last|more|less)\b",
            normalized,
        )
    )


def _schema_tuple_question_contract(question: str) -> Dict[str, bool]:
    text = str(question or "")
    percent_requested = bool(_PERCENT_REQUEST_RE.search(text))
    month_requested = bool(_MONTH_REQUEST_RE.search(text))
    cause_requested = bool(_CAUSE_REQUEST_RE.search(text))
    date_requested = bool(_DATE_REQUEST_RE.search(text) or month_requested)
    number_requested = bool(_NUMBER_REQUEST_RE.search(text) or percent_requested)
    open_ended = bool(_OPEN_ENDED_REQUEST_RE.search(text)) and not (
        date_requested or number_requested or cause_requested
    )
    return {
        "date_requested": date_requested,
        "number_requested": number_requested,
        "percent_requested": percent_requested,
        "month_requested": month_requested,
        "cause_requested": cause_requested,
        "open_ended": open_ended,
    }


def _schema_tuple_answer_contract_reasons(
    *,
    question: str,
    item: Dict[str, Any],
    value: str,
    value_type: str,
    answer_slot: str,
    source_surface: str = "",
) -> Tuple[List[str], Dict[str, Any]]:
    contract = _schema_tuple_question_contract(question)
    triple = item.get("triple") or []
    surface = " ".join(
        [
            str(value or ""),
            " ".join(str(part or "") for part in triple)
            if isinstance(triple, (list, tuple))
            else "",
            str(item.get("predicate") or ""),
            str(item.get("slot_text") or ""),
            str(source_surface or ""),
        ]
    )
    normalized_value_type = _normalize_schema_tuple_type(value_type)
    normalized_answer_slot = _normalize_schema_tuple_type(answer_slot)
    candidate_relation_tokens = _schema_tuple_candidate_relation_tokens(item)
    reasons: List[str] = []

    if contract["percent_requested"] and not _PERCENT_VALUE_RE.search(surface):
        reasons.append("answer_contract_percentage_unbound")
    if contract["month_requested"] and not _MONTH_VALUE_RE.search(surface):
        reasons.append("answer_contract_month_unbound")
    if contract["cause_requested"]:
        if normalized_value_type in {"date", "number"}:
            reasons.append("answer_contract_cause_value_type_mismatch")
        if not (
            candidate_relation_tokens
            & {"cause", "death", "die", "died", "kill", "killed", "disease", "illness"}
        ):
            reasons.append("answer_contract_cause_relation_unbound")
    if contract["open_ended"] and normalized_answer_slot in {"unknown", "entity"}:
        reasons.append("answer_contract_open_ended_tuple_projection")

    metadata = dict(contract)
    metadata.update({
        "answer_contract_value_type": normalized_value_type,
        "answer_contract_answer_slot": normalized_answer_slot,
    })
    return list(dict.fromkeys(reasons)), metadata


def _schema_tuple_reader_visible_projection_reasons(
    *,
    question: str,
    item: Dict[str, Any],
    value: str,
    terminal_metadata: Dict[str, Any],
    projection_metadata: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    answer_slot = str(terminal_metadata.get("terminal_answer_slot") or "")
    value_type = str(terminal_metadata.get("terminal_value_type") or "")
    contract_reasons, contract_metadata = _schema_tuple_answer_contract_reasons(
        question=question,
        item=item,
        value=value,
        value_type=value_type,
        answer_slot=answer_slot,
        source_surface=str(projection_metadata.get("provenance_sentence") or ""),
    )
    predicate_contract = bool(
        item.get("predicate_contract_match")
        or item.get("relation_signature_match")
        or item.get("relation_signature_compatible")
    )
    relation_intersection = list(
        terminal_metadata.get("terminal_relation_intersection") or []
    )
    source_relation_bound = bool(
        projection_metadata.get("source_sentence_relation_bound")
    )
    strong_relation_binding = bool(
        predicate_contract or (relation_intersection and source_relation_bound)
    )

    reasons = list(contract_reasons)
    normalized_answer_slot = _normalize_schema_tuple_type(answer_slot)
    if not strong_relation_binding:
        reasons.append("reader_visible_relation_contract_missing")
    if normalized_answer_slot in {"unknown", "entity"} and not strong_relation_binding:
        reasons.append("reader_visible_weak_entity_contract")
    if contract_metadata.get("open_ended") and not (
        normalized_answer_slot in {"date", "location", "person"}
        and strong_relation_binding
    ):
        reasons.append("reader_visible_open_ended_blocked")

    metadata = dict(contract_metadata)
    metadata.update({
        "reader_visible_predicate_contract": bool(predicate_contract),
        "reader_visible_strong_relation_binding": bool(strong_relation_binding),
        "reader_visible_source_relation_bound": bool(source_relation_bound),
    })
    return list(dict.fromkeys(reasons)), metadata


def _schema_tuple_terminal_slot_decision(
    *,
    question: str,
    item: Dict[str, Any],
    value: str,
    lineage_context: Dict[str, Any],
    lineage_metadata: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    answer_slot = str(lineage_context.get("answer_slot") or _question_answer_slot(question))
    value_type = _infer_schema_tuple_value_type(item, value)
    stage = str(item.get("slot_plan_stage") or "unclassified").strip().lower()
    role = str(item.get("materialization_role") or "").strip().lower()
    explicit_terminal = bool(
        item.get("terminal_tuple_eligible")
        or stage == "final_answer"
        or role in _SCHEMA_TUPLE_TERMINAL_ROLES
    )
    explicit_support = bool(
        stage in {"bridge", "intermediate", "support", "support_only"}
        or role in _SCHEMA_TUPLE_SUPPORT_ROLES
    )
    question_relation_tokens = _schema_tuple_question_relation_tokens(question)
    candidate_relation_tokens = _schema_tuple_candidate_relation_tokens(item)
    relation_intersection = sorted(question_relation_tokens & candidate_relation_tokens)
    relation_direct = bool(relation_intersection) or not question_relation_tokens
    relation_signature_ok = bool(
        item.get("relation_signature_match")
        or item.get("relation_signature_compatible")
    )
    predicate_contract_ok = bool(item.get("predicate_contract_match"))
    type_ok = _schema_tuple_value_matches_answer_slot(answer_slot, value_type)
    answer_contract_reasons, answer_contract_metadata = (
        _schema_tuple_answer_contract_reasons(
            question=question,
            item=item,
            value=value,
            value_type=value_type,
            answer_slot=answer_slot,
        )
    )
    required_uncovered = set(lineage_context.get("required_uncovered_slot_ids") or [])
    item_slot_ids = {
        str(item.get("slot_id") or ""),
        str(item.get("entity_predicate_target_slot_id") or ""),
    }
    item_slot_ids.discard("")
    lineage_uncovered = set(lineage_metadata.get("uncovered_slot_ids") or [])
    active_tuple_anchor_bound = bool(
        item.get("active_schema_bridge_tuple_executor")
        and (
            item.get("residual_anchor_bound")
            or item.get("branch_binding_passed") is True
            or _as_float(item.get("slot_anchor_overlap"), 0.0) >= 0.67
        )
    )
    slot_bound = bool(
        not required_uncovered
        or lineage_uncovered & required_uncovered
        or item_slot_ids & required_uncovered
        or active_tuple_anchor_bound
    )
    boolean_or_comparison = _schema_tuple_is_boolean_or_comparison_question(
        question, lineage_context
    )

    reasons: List[str] = []
    if explicit_support:
        reasons.append("tuple_marked_support_only")
    if boolean_or_comparison and not explicit_terminal:
        reasons.append("non_terminal_boolean_or_comparison_query")
    if item.get("slot_materialization_safe") is False:
        reasons.append("slot_materialization_unsafe")
    if item.get("slot_meta_negative") is True:
        reasons.append("slot_meta_negative")
    if item.get("answer_type_match") is False:
        reasons.append("answer_type_mismatch")
    if not type_ok:
        reasons.append("terminal_answer_type_mismatch")
    reasons.extend(answer_contract_reasons)
    if not slot_bound:
        reasons.append("terminal_slot_not_uncovered")
    if not (
        explicit_terminal
        or relation_direct
        or (relation_signature_ok and predicate_contract_ok)
    ):
        reasons.append("terminal_relation_mismatch")
    if (
        _normalize_schema_tuple_type(answer_slot) in {"unknown", "entity"}
        and not explicit_terminal
        and not relation_intersection
        and not (relation_signature_ok and predicate_contract_ok)
    ):
        reasons.append("terminal_weak_entity_contract")

    terminal = not reasons
    return terminal, {
        "terminal_slot_gate_passed": bool(terminal),
        "terminal_slot_gate_reasons": list(dict.fromkeys(reasons)),
        "terminal_answer_slot": answer_slot,
        "terminal_value_type": value_type,
        "terminal_slot_stage": stage,
        "terminal_materialization_role": role,
        "terminal_explicit": bool(explicit_terminal),
        "terminal_relation_direct": bool(relation_direct),
        "terminal_relation_signature_ok": bool(relation_signature_ok),
        "terminal_predicate_contract_ok": bool(predicate_contract_ok),
        "terminal_question_relation_tokens": sorted(question_relation_tokens),
        "terminal_candidate_relation_tokens": sorted(candidate_relation_tokens)[:20],
        "terminal_relation_intersection": relation_intersection,
        "terminal_slot_bound": bool(slot_bound),
        "terminal_boolean_or_comparison": bool(boolean_or_comparison),
        **answer_contract_metadata,
    }


def _compact_schema_bridge_doc(
    doc: str,
    item: Dict[str, Any],
    *,
    question_tokens: set,
    deficit_tokens: set,
    relation_tokens: set,
    max_sentences: int = 2,
) -> str:
    title = str(item.get("title") or _title(doc) or "Schema bridge evidence").strip()
    schema = item.get("schema") or []
    if isinstance(schema, (list, tuple)) and len(schema) >= 3:
        schema_text = f"{schema[0]} -- {schema[1]} -- {schema[2]}"
    else:
        schema_text = str(schema or "")
    matched_tokens = _tokens(" ".join(str(token) for token in item.get("matched_join_tokens") or []))
    target_tokens = set(question_tokens) | set(deficit_tokens) | set(relation_tokens) | matched_tokens
    scored_sentences: List[Tuple[float, int, str]] = []
    for idx, sentence in enumerate(_split_projection_sentences(doc)):
        sentence_tokens = _tokens(sentence)
        score = (
            _coverage(target_tokens, sentence_tokens)
            + _coverage(matched_tokens, sentence_tokens)
            + _coverage(relation_tokens, sentence_tokens)
        )
        scored_sentences.append((float(score), idx, sentence))
    scored_sentences.sort(key=lambda row: (-row[0], row[1]))
    selected = [
        sentence
        for score, _idx, sentence in scored_sentences[: max(1, int(max_sentences))]
        if score > 0.0
    ]
    if not selected:
        selected = [sentence for _score, _idx, sentence in scored_sentences[:1]]
    if not selected:
        selected = [str(doc or "").strip()[:700]]
    lines = [
        title,
        f"Schema bridge: {schema_text}",
    ]
    triple = item.get("triple") or []
    if isinstance(triple, (list, tuple)) and len(triple) == 3:
        lines.append(
            "Fact join row: "
            + " -- ".join(str(part).strip() for part in triple)
        )
    lines.append("Evidence: " + " ".join(selected)[:900])
    return "\n".join(line for line in lines if line.strip())


def _format_validated_schema_tuple_projection(
    item: Dict[str, Any],
    source_doc: str,
    *,
    question: str,
    question_tokens: set,
    deficit_tokens: set,
    relation_tokens: set,
    anchor_binding: Dict[str, Any],
    max_sentences: int = 1,
) -> Tuple[str, Dict[str, Any]]:
    title = str(item.get("title") or _title(source_doc) or "Schema tuple evidence").strip()
    triple = list(item.get("triple") or [])
    subject = str(triple[0] if len(triple) > 0 else item.get("active_entity") or "").strip()
    predicate = str(triple[1] if len(triple) > 1 else item.get("predicate") or "").strip()
    value = str(
        item.get("answer_value")
        or (
            triple[2]
            if len(triple) > 2
            else item.get("value") or ""
        )
    ).strip()
    schema = item.get("schema") or []
    if isinstance(schema, (list, tuple)) and len(schema) >= 3:
        schema_text = f"{schema[0]} -- {schema[1]} -- {schema[2]}"
    else:
        schema_text = str(schema or "")
    matched_tokens = _tokens(" ".join(str(token) for token in item.get("matched_join_tokens") or []))
    tuple_tokens = _tokens(" ".join([subject, predicate, value]))
    slot_tokens = _tokens(
        " ".join(
            [
                str(item.get("slot_text") or ""),
                " ".join(str(token) for token in item.get("slot_anchor_tokens") or []),
                " ".join(str(token) for token in item.get("slot_relation_tokens") or []),
            ]
        )
    )
    target_tokens = (
        set(question_tokens)
        | set(deficit_tokens)
        | set(relation_tokens)
        | matched_tokens
        | tuple_tokens
        | slot_tokens
    )
    scored_sentences: List[Tuple[float, int, str]] = []
    for idx, sentence in enumerate(_split_projection_sentences(source_doc)):
        sentence_tokens = _tokens(sentence)
        score = (
            _coverage(target_tokens, sentence_tokens)
            + _coverage(tuple_tokens, sentence_tokens)
            + _coverage(matched_tokens, sentence_tokens)
            + _coverage(relation_tokens, sentence_tokens)
        )
        scored_sentences.append((float(score), idx, sentence))
    scored_sentences.sort(key=lambda row: (-row[0], row[1]))
    selected = [
        sentence
        for score, _idx, sentence in scored_sentences[: max(1, int(max_sentences))]
        if score > 0.0
    ]
    if not selected and scored_sentences:
        selected = [scored_sentences[0][2]]
    if not selected:
        selected = [str(source_doc or "").strip()[:700]]

    slot_text = str(item.get("slot_text") or "").strip()
    slot_id = str(item.get("slot_id") or item.get("entity_predicate_target_slot_id") or "").strip()
    tuple_text = " -- ".join(part for part in [subject, predicate, value] if part)
    lines = [
        title,
        "Validated schema tuple evidence",
        f"Question: {str(question or '').strip()[:220]}",
    ]
    if schema_text:
        lines.append(f"Schema signature: {schema_text}")
    if tuple_text:
        lines.append(f"Fact tuple: {tuple_text}")
    if slot_text or slot_id:
        slot_prefix = f"{slot_id}: " if slot_id else ""
        lines.append(f"Answer slot: {slot_prefix}{slot_text}"[:320])
    if item.get("predicate_contract_match") or item.get("relation_signature_match") or item.get("relation_signature_compatible"):
        lines.append("Predicate contract: matched")
    if anchor_binding.get("bound") or item.get("branch_binding_passed") is True:
        matched = sorted(
            set(anchor_binding.get("matched_tokens") or [])
            | set(str(token) for token in item.get("slot_anchor_tokens") or [])
        )
        if matched:
            lines.append("Join key binding: " + ", ".join(matched[:8]))
    provenance_sentence = " ".join(selected)
    lines.append("Provenance sentence: " + provenance_sentence[:900])

    projected_doc = "\n".join(line for line in lines if str(line).strip())
    original_tokens = len(re.findall(r"\S+", str(source_doc or "")))
    projected_tokens = len(re.findall(r"\S+", projected_doc))
    provenance_tokens = _tokens(provenance_sentence)
    value_tokens = _tokens(value)
    join_tokens = (
        matched_tokens
        | set(str(token) for token in anchor_binding.get("matched_tokens") or [])
        | set(str(token) for token in item.get("active_join_key_tokens") or [])
        | set(str(token) for token in item.get("candidate_join_key_tokens") or [])
    )
    relation_probe_tokens = _schema_tuple_candidate_relation_tokens(item)
    return projected_doc, {
        "original_doc_token_count": int(original_tokens),
        "projected_doc_token_count": int(projected_tokens),
        "projection_token_savings": max(0, int(original_tokens) - int(projected_tokens)),
        "provenance_sentence": provenance_sentence[:900],
        "source_sentence_value_bound": bool(
            not value_tokens or value_tokens & provenance_tokens
        ),
        "source_sentence_join_bound": bool(
            not join_tokens or join_tokens & provenance_tokens
        ),
        "source_sentence_relation_bound": bool(
            not relation_probe_tokens or relation_probe_tokens & provenance_tokens
        ),
    }


def _merge_cross_schema_schema_memory_candidates(
    query_schema_memory: Dict[str, Any],
    entity_predicate_fact_join: Dict[str, Any],
    *,
    enabled: bool,
    max_candidates: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Expose provenance fact-store joins through the Schema Memory interface.

    The original query schema path only probes passages that instantiate the
    active fact's exact schema signature. A multi-hop join normally changes the
    predicate/schema while retaining one concrete endpoint. This adapter keeps
    the rank-stable cross-schema rows produced by the provenance fact store and
    lets the residual/anchor/lineage admission path decide reader visibility.
    """
    base = dict(query_schema_memory or {})
    base_items = [
        dict(item)
        for item in (base.get("candidate_items") or [])
        if isinstance(item, dict)
    ]
    source_items = list(
        (entity_predicate_fact_join or {}).get("candidate_items") or []
    )
    raw_cross_items = [
        item
        for item in source_items
        if isinstance(item, dict) and bool(item.get("cross_schema"))
    ]
    pre_reject_counts: Counter = Counter()
    pre_reject_counts["invalid_candidate"] = sum(
        not isinstance(item, dict) for item in source_items
    )
    pre_reject_counts["same_schema_row"] = sum(
        isinstance(item, dict) and not bool(item.get("cross_schema"))
        for item in source_items
    )
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "reason": "disabled" if not enabled else "not_evaluated",
        "base_candidate_count": len(base_items),
        "source_candidate_count": len(source_items),
        "raw_cross_schema_candidate_count": len(raw_cross_items),
        "eligible_cross_schema_candidate_count": 0,
        "merged_cross_schema_candidate_count": 0,
        "merged_candidate_count": len(base_items),
        "reject_reason_counts": {
            key: int(value)
            for key, value in pre_reject_counts.items()
            if int(value) > 0
        },
        "source": "provenance_fact_store",
    }
    base["cross_schema_join_enabled"] = bool(enabled)
    if not enabled:
        return base, info
    if not raw_cross_items:
        info["reason"] = "no_cross_schema_fact_candidates"
        base["cross_schema_join_reason"] = info["reason"]
        base["cross_schema_join_candidate_count"] = 0
        return base, info

    reject_counts: Counter = Counter(pre_reject_counts)
    converted: List[Dict[str, Any]] = []
    for item in raw_cross_items:
        if not bool(item.get("entity_bound")) or not bool(
            item.get("join_key_exact")
        ):
            reject_counts["endpoint_not_exact"] += 1
            continue
        try:
            doc_id = int(item.get("doc_id"))
        except (TypeError, ValueError):
            reject_counts["invalid_doc_id"] += 1
            continue
        active_entity = str(
            item.get("active_entity_key") or item.get("active_entity") or ""
        ).strip()
        endpoint_tokens = sorted(_tokens(active_entity))
        if not endpoint_tokens:
            reject_counts["empty_join_endpoint"] += 1
            continue
        schema = item.get("schema") or []
        if not isinstance(schema, (list, tuple)) or len(schema) != 3:
            reject_counts["invalid_schema"] += 1
            continue
        catalog_selectivity = _clamp(item.get("catalog_selectivity", 0.0))
        endpoint_scan_rows = max(
            1, _as_int(item.get("estimated_endpoint_scan_rows"), 1)
        )
        endpoint_access_selectivity = 1.0 / (
            1.0 + math.log1p(endpoint_scan_rows)
        )




        schema_idf = max(catalog_selectivity, endpoint_access_selectivity)
        provenance_score = _clamp(item.get("provenance_score", 0.0))
        passthrough_fields = (
            "slot_id",
            "slot_text",
            "slot_anchor_overlap",
            "slot_anchor_tokens",
            "slot_relation_tokens",
            "slot_polarity",
            "slot_answer_types",
            "slot_source",
            "slot_hop_index",
            "slot_target_predicate_concepts",
            "slot_expected_value_type",
            "slot_direction",
            "slot_relation_signature",
            "slot_relation_arity",
            "slot_execution_operator",
            "slot_join_executable",
            "slot_materialization_safe",
            "slot_plan_stage",
            "slot_branch_id",
            "slot_final_answer_type",
            "slot_expected_type_family",
            "slot_unsafe_reasons",
            "slot_meta_negative",
            "materialization_role",
            "branch_binding_passed",
            "branch_binding_reason",
            "branch_anchor_phrase",
            "expected_relation_signature",
            "candidate_relation_signature",
            "relation_signature_match",
            "relation_signature_compatible",
            "relation_signature_compatibility_tier",
            "relation_signature_fanout",
            "exact_relation_signature_fanout",
            "compatible_relation_signature_fanout",
            "selective_access_eligible",
            "selective_global_access",
            "dependency_binding_safe",
            "dependency_binding_reasons",
            "dependency_root_entity_key",
            "dependency_terminal_entity_key",
            "dependency_path_entity_keys",
            "dependency_hops",
            "operator_candidate_kind",
            "operator_validation_passed",
            "operator_validation_score",
            "operator_validation_reason",
        )
        converted_item = {
            "doc_id": doc_id,
            "title": str(item.get("title") or ""),
            "score": _clamp(item.get("score", 0.0)),
            "net_utility": _as_float(item.get("net_utility"), 0.0),
            "risk": _clamp(item.get("risk", 0.0)),
            "original_rank": _as_int(item.get("original_rank"), 10**9),
            "schema": list(schema),
            "active_schema": list(item.get("active_schema") or []),
            "schema_frequency": _as_int(item.get("catalog_doc_count"), 0),
            "schema_idf": schema_idf,
            "catalog_selectivity": catalog_selectivity,
            "endpoint_access_selectivity": endpoint_access_selectivity,
            "estimated_endpoint_scan_rows": endpoint_scan_rows,
            "operator_latency_ms": _round_float(
                item.get(
                    "operator_latency_ms",
                    (entity_predicate_fact_join or {}).get(
                        "operator_latency_ms",
                        0.0,
                    ),
                )
            ),
            "operator_name": str(
                item.get("operator_name")
                or (entity_predicate_fact_join or {}).get("physical_operator")
                or "entity_predicate_fact_join"
            ),
            "schema_support": provenance_score,
            "context_overlap": _clamp(item.get("query_overlap", 0.0)),
            "query_overlap": _clamp(item.get("query_overlap", 0.0)),
            "join_key_overlap": 1.0,
            "join_key_containment": 1.0,
            "join_key_exact": True,
            "entity_bound": True,
            "active_join_role": str(item.get("active_join_role") or ""),
            "candidate_join_role": str(item.get("candidate_join_role") or ""),
            "active_join_key_tokens": endpoint_tokens,
            "candidate_join_key_tokens": endpoint_tokens,
            "matched_join_tokens": endpoint_tokens,
            "join_key_matched": True,
            "already_top5": bool(
                item.get("already_top5") or item.get("already_visible")
            ),
            "candidate_source": "cross_schema_residual_fact_join",
            "cross_schema_join": True,
            "triple": list(item.get("triple") or []),
            "direction": str(item.get("direction") or ""),
            "active_entity": str(item.get("active_entity") or active_entity),
            "answer_value": str(item.get("answer_value") or ""),
            "answer_type": str(item.get("answer_type") or "unknown"),
            "answer_type_match": bool(item.get("answer_type_match")),
            "relation_overlap": _clamp(item.get("relation_overlap", 0.0)),
            "predicate": str(item.get("predicate") or ""),
            "predicate_contract_match": bool(
                item.get("predicate_contract_match")
            ),
            "predicate_match_mode": str(
                item.get("predicate_match_mode") or ""
            ),
            "lineage_bound": bool(item.get("lineage_bound")),
            "lineage_depth": _as_int(item.get("lineage_depth"), 0),
            "provenance_score": provenance_score,
            "provenance_doc_ids": list(item.get("provenance_doc_ids") or []),
        }
        for field in passthrough_fields:
            if field in item:
                converted_item[field] = item.get(field)
        converted.append(converted_item)

    converted.sort(
        key=lambda row: (
            bool(row.get("already_top5")),
            -float(row.get("net_utility", 0.0)),
            -float(row.get("score", 0.0)),
            -float(row.get("relation_overlap", 0.0)),
            int(row.get("original_rank", 10**9)),
        )
    )
    converted = converted[: max(0, int(max_candidates))]
    info["eligible_cross_schema_candidate_count"] = len(converted)

    by_doc: Dict[int, Dict[str, Any]] = {}
    for item in base_items + converted:
        try:
            doc_id = int(item.get("doc_id"))
        except (TypeError, ValueError):
            continue
        current = by_doc.get(doc_id)
        priority = (
            int(bool(item.get("cross_schema_join"))),
            int(not bool(item.get("already_top5"))),
            float(item.get("relation_overlap", 0.0)),
            float(item.get("score", 0.0)),
        )
        current_priority = (
            int(bool((current or {}).get("cross_schema_join"))),
            int(not bool((current or {}).get("already_top5"))),
            float((current or {}).get("relation_overlap", 0.0)),
            float((current or {}).get("score", 0.0)),
        )
        if current is None or priority > current_priority:
            by_doc[doc_id] = item

    merged = sorted(
        by_doc.values(),
        key=lambda row: (
            int(bool(row.get("cross_schema_join"))),
            int(not bool(row.get("already_top5"))),
            float(row.get("score", 0.0)),
            float(row.get("relation_overlap", 0.0)),
            -int(row.get("original_rank", 10**9)),
        ),
        reverse=True,
    )
    merged_cross_count = sum(bool(row.get("cross_schema_join")) for row in merged)
    info.update({
        "reason": (
            "cross_schema_candidates_merged"
            if merged_cross_count
            else "no_eligible_cross_schema_candidate"
        ),
        "merged_cross_schema_candidate_count": int(merged_cross_count),
        "merged_candidate_count": len(merged),
        "reject_reason_counts": dict(reject_counts),
    })
    base.update({
        "candidate_items": merged,
        "candidate_count": len(merged),
        "admitted_candidate_count": len(merged),
        "cross_schema_join_reason": info["reason"],
        "cross_schema_join_candidate_count": int(merged_cross_count),
        "cross_schema_join_raw_candidate_count": len(raw_cross_items),
        "cross_schema_join_reject_reason_counts": dict(reject_counts),
        "reason": (
            "candidates_generated"
            if merged
            else str(base.get("reason") or "no_accepted_candidates")
        ),
    })
    return base, info


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _coverage(query_tokens: set, doc_tokens: set) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / float(len(query_tokens))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _round_float(value: Any) -> float:
    return round(float(value or 0.0), 6)


def _question_answer_slot(question: str) -> str:
    required_type = _question_requires_value_type(question)
    if required_type:
        return required_type
    normalized = str(question or "").strip().lower()
    if (
        normalized.startswith("where")
        or re.search(
            r"\b(?:what|which)\s+(?:city|country|county|location|place|region|state)\b",
            normalized,
        )
    ):
        return "location"
    if normalized.startswith("who"):
        return "person"
    if re.search(r"\bwhich\s+(?:film|movie|book|song|work)\b", normalized):
        return "work"
    if normalized.startswith("which") or normalized.startswith("what"):
        return "entity"
    return "unknown"


def _iter_required_slot_texts(planned_retrieval: Dict[str, Any]):
    for key in ("residual_deficit_queries", "residual_deficit_replaced_queries"):
        for value in planned_retrieval.get(key) or []:
            if str(value or "").strip():
                yield str(value).strip()
    for round_info in planned_retrieval.get("rounds") or []:
        if not isinstance(round_info, dict):
            continue
        verifier = round_info.get("verifier") or {}
        if isinstance(verifier, dict):
            for key in ("missing_slots", "queries"):
                for value in verifier.get(key) or []:
                    if str(value or "").strip():
                        yield str(value).strip()
        for key in ("queries", "verifier_queries"):
            for value in round_info.get(key) or []:
                if str(value or "").strip():
                    yield str(value).strip()


def _slot_match_score(slot: Dict[str, Any], doc: str) -> float:
    title_tokens = _tokens(_title(doc))
    doc_tokens = _tokens(doc)
    anchor_tokens = set(slot.get("anchor_tokens") or [])
    predicate_tokens = set(slot.get("predicate_tokens") or [])
    if anchor_tokens:
        title_hits = len(anchor_tokens & title_tokens)
        body_hits = len(anchor_tokens & doc_tokens)
        anchor_denominator = float(max(1, len(anchor_tokens)))
        title_score = title_hits / anchor_denominator
        body_score = body_hits / anchor_denominator
        anchor_match = bool(
            title_hits >= 2
            or title_score >= 0.50
            or body_hits >= 2 and body_score >= 0.50
            or len(anchor_tokens) == 1 and title_hits == 1
        )
        if not anchor_match:
            return 0.0
    else:
        title_score = 0.0
        body_score = 0.0
    required_predicates = _canonical_predicate_tokens(predicate_tokens)
    document_predicates = _canonical_predicate_tokens(doc_tokens, doc)
    predicate_score = (
        len(required_predicates & document_predicates)
        / float(max(1, len(required_predicates)))
        if required_predicates
        else 0.0
    )
    if predicate_tokens and predicate_score <= 0.0:
        return 0.0
    return _clamp(0.62 * max(title_score, body_score) + 0.38 * predicate_score)


def _build_lineage_context(
    question: str,
    planned_retrieval: Dict[str, Any],
    base_docs: List[str],
) -> Dict[str, Any]:
    seen = set()
    slots = []
    for text in _iter_required_slot_texts(planned_retrieval):
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        slot_tokens = _tokens(text)
        anchor_tokens = sorted(slot_tokens - _SLOT_NON_ANCHOR_TERMS)
        predicate_tokens = sorted(slot_tokens & _SLOT_PREDICATE_TERMS)
        if not anchor_tokens and not predicate_tokens:
            continue
        slots.append({
            "slot_id": f"slot_{len(slots)}",
            "text": text[:240],
            "anchor_tokens": anchor_tokens,
            "predicate_tokens": predicate_tokens,
        })
        if len(slots) >= 12:
            break

    base_covered = set()
    for slot in slots:
        if any(_slot_match_score(slot, doc) >= 0.34 for doc in base_docs):
            base_covered.add(str(slot["slot_id"]))
    required_ids = [str(slot["slot_id"]) for slot in slots]
    uncovered_ids = [slot_id for slot_id in required_ids if slot_id not in base_covered]
    question_lower = str(question or "").lower()
    comparison = bool(
        len(slots) >= 2
        and (
            re.search(r"\b(?:which|who|what)\b.*\bor\b", question_lower)
            or any(token in question_lower for token in ("earlier", "later", "first", "older", "younger"))
        )
    )
    return _ensure_comparison_branches({
        "answer_slot": _question_answer_slot(question),
        "required_slots": slots,
        "required_slot_ids": required_ids,
        "base_covered_slot_ids": sorted(base_covered),
        "required_uncovered_slot_ids": uncovered_ids,
        "comparison_query": comparison,
    })


def _candidate_lineage_metadata(
    doc: str,
    source: str,
    lineage_context: Dict[str, Any],
) -> Dict[str, Any]:
    matched = []
    scores = {}
    for slot in lineage_context.get("required_slots") or []:
        score = _slot_match_score(slot, doc)
        if score < 0.34:
            continue
        slot_id = str(slot.get("slot_id") or "")
        if not slot_id:
            continue
        matched.append(slot_id)
        scores[slot_id] = _round_float(score)
    base_covered = set(lineage_context.get("base_covered_slot_ids") or [])
    uncovered = [slot_id for slot_id in matched if slot_id not in base_covered]
    required_uncovered = lineage_context.get("required_uncovered_slot_ids") or []
    best_slot = ""
    if matched:
        best_slot = max(matched, key=lambda slot_id: scores.get(slot_id, 0.0))
    slot_lookup = {
        str(slot.get("slot_id") or ""): slot
        for slot in lineage_context.get("required_slots") or []
    }
    best_spec = slot_lookup.get(best_slot) or {}
    covered_branch_ids = sorted(
        {
            str(slot_lookup.get(slot_id, {}).get("branch_id") or "")
            for slot_id in matched
            if str(slot_lookup.get(slot_id, {}).get("branch_id") or "")
        }
    )
    coverage_gain = len(uncovered) / float(max(1, len(required_uncovered)))
    lineage_confidence = max((scores.get(slot_id, 0.0) for slot_id in matched), default=0.0)
    lineage_group_id = (
        "slot:" + "+".join(sorted(uncovered or matched))
        if matched
        else f"operator:{source}"
    )
    return {
        "answer_slot": str(lineage_context.get("answer_slot") or "unknown"),
        "required_slot_ids": list(lineage_context.get("required_slot_ids") or []),
        "covered_slot_ids": sorted(matched),
        "uncovered_slot_ids": sorted(uncovered),
        "coverage_gain": _round_float(coverage_gain),
        "lineage_confidence": _round_float(lineage_confidence),
        "lineage_group_id": lineage_group_id,
        "comparison_arm": str(best_spec.get("text") or "")[:160],
        "covered_branch_ids": covered_branch_ids,
        "comparison_branch_id": str(best_spec.get("branch_id") or ""),
        "predicate_tokens": list(best_spec.get("predicate_tokens") or []),
        "comparison_query": bool(lineage_context.get("comparison_query")),
    }


def _entity_predicate_contract_lineage_metadata(
    item: Dict[str, Any],
    lineage_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Bind a fact-join row to the planner slot that created its probe."""
    slot_text = str(item.get("slot_text") or "")
    slot_tokens = _tokens(slot_text)
    slot_anchor_tokens = set(item.get("slot_anchor_tokens") or [])
    slot_predicate_tokens = set(item.get("slot_relation_tokens") or [])
    required_slots = list(lineage_context.get("required_slots") or [])
    best_slot: Dict[str, Any] = {}
    best_score = 0.0
    for slot in required_slots:
        required_tokens = _tokens(slot.get("text") or "")
        required_anchors = set(slot.get("anchor_tokens") or [])
        required_predicates = set(slot.get("predicate_tokens") or [])
        text_overlap = _coverage(required_tokens, slot_tokens)
        anchor_overlap = _coverage(required_anchors, slot_anchor_tokens)
        predicate_overlap = _coverage(required_predicates, slot_predicate_tokens)
        exact_text = bool(
            slot_text
            and re.sub(r"\s+", " ", slot_text).strip().lower()
            == re.sub(r"\s+", " ", str(slot.get("text") or "")).strip().lower()
        )
        score = max(
            1.0 if exact_text else 0.0,
            0.52 * text_overlap + 0.30 * anchor_overlap + 0.18 * predicate_overlap,
        )
        if score > best_score:
            best_score = score
            best_slot = slot

    if not best_slot and bool(lineage_context.get("entity_predicate_virtual_slot")):
        best_slot = (required_slots or [{}])[0]
        best_score = 1.0

    target_slot_id = str(best_slot.get("slot_id") or "") if best_score >= 0.50 else ""
    base_covered = set(lineage_context.get("base_covered_slot_ids") or [])
    covered = [target_slot_id] if target_slot_id else []
    uncovered = (
        [target_slot_id]
        if target_slot_id and target_slot_id not in base_covered
        else []
    )
    required_uncovered = list(
        lineage_context.get("required_uncovered_slot_ids") or []
    )
    predicate_contract_score = 1.0 if item.get("predicate_contract_match") else 0.0
    lineage_confidence = _clamp(
        0.34 * best_score
        + 0.24 * _as_float(item.get("root_anchor_score"), 0.0)
        + 0.22 * _as_float(item.get("slot_anchor_overlap"), 0.0)
        + 0.20 * predicate_contract_score
    )
    return {
        "answer_slot": str(lineage_context.get("answer_slot") or "unknown"),
        "required_slot_ids": list(lineage_context.get("required_slot_ids") or []),
        "covered_slot_ids": covered,
        "uncovered_slot_ids": uncovered,
        "coverage_gain": _round_float(
            len(uncovered) / float(max(1, len(required_uncovered)))
        ),
        "lineage_confidence": _round_float(lineage_confidence),
        "lineage_group_id": (
            f"entity_predicate:{target_slot_id}"
            if target_slot_id
            else "entity_predicate:unbound"
        ),
        "comparison_arm": str(best_slot.get("text") or slot_text)[:160],
        "covered_branch_ids": (
            [str(best_slot.get("branch_id"))]
            if str(best_slot.get("branch_id") or "")
            else []
        ),
        "comparison_branch_id": str(best_slot.get("branch_id") or ""),
        "predicate_tokens": list(
            best_slot.get("predicate_tokens") or slot_predicate_tokens
        )[:12],
        "comparison_query": bool(lineage_context.get("comparison_query")),
        "entity_predicate_slot_id": str(item.get("slot_id") or ""),
        "entity_predicate_slot_text": slot_text[:240],
        "entity_predicate_target_slot_id": target_slot_id,
        "entity_predicate_slot_match": _round_float(best_score),
        "lineage_bound": bool(item.get("lineage_bound")),
        "root_entity": str(item.get("root_entity") or "")[:160],
        "root_anchor_score": _round_float(item.get("root_anchor_score", 0.0)),
        "lineage_depth": _as_int(item.get("lineage_depth"), 0),
        "lineage_path": list(item.get("lineage_path") or [])[:3],
        "predicate_contract_match": bool(item.get("predicate_contract_match")),
        "predicate_match_mode": str(item.get("predicate_match_mode") or ""),
        "direct_predicate_overlap": _round_float(
            item.get("direct_predicate_overlap", 0.0)
        ),
    }


def _normalize_cost_admission_policy(policy: Any) -> str:
    normalized = str(policy or "learned").strip().lower()
    if normalized not in _COST_ADMISSION_POLICIES:
        return "learned"
    return normalized


def _deficit_replacement_priority(
    candidate: Dict[str, Any],
    marginal_slot_ids: set,
) -> float:
    """Score marginal evidence value for a budget-preserving replacement."""
    return (
        _as_float(candidate.get("net_utility"), 0.0)
        + 0.14 * _as_float(candidate.get("lineage_confidence"), 0.0)
        + 0.12 * _as_float(candidate.get("coverage_gain"), 0.0)
        + 0.08 * min(2, len(marginal_slot_ids))
    )


def _candidate_trace_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Return policy-independent candidate fields used to verify pool parity."""
    fields = (
        "source",
        "title",
        "input_extra_rank",
        "token_count",
        "query_overlap",
        "missing_gain",
        "novelty",
        "utility",
        "risk",
        "token_cost",
        "compute_cost",
        "net_utility",
        "plan_type",
        "plan_utility",
        "plan_net_utility",
        "plan_risk",
        "endpoint_overlap",
        "relation_overlap",
        "schema_score",
        "entity_predicate_score",
        "schema_idf",
        "join_key_overlap",
        "schema_frequency",
        "schema_selectivity",
        "provenance_score",
        "compact_table",
        "best_rank",
        "worst_rank",
        "original_rank",
        "answer_slot",
        "active_entity",
        "predicate",
        "direction",
        "answer_value",
        "answer_type_match",
        "cross_schema",
        "retrieval_pool_member",
        "required_slot_ids",
        "covered_slot_ids",
        "uncovered_slot_ids",
        "coverage_gain",
        "lineage_confidence",
        "lineage_group_id",
        "comparison_arm",
        "covered_branch_ids",
        "comparison_branch_id",
        "predicate_tokens",
        "comparison_query",
        "compact_projection",
        "projection_reason",
        "original_doc_token_count",
        "projected_doc_token_count",
        "projection_token_savings",
        "lineage_bound",
        "root_entity",
        "root_anchor_score",
        "lineage_depth",
        "lineage_path",
        "entity_predicate_slot_id",
        "entity_predicate_slot_text",
        "entity_predicate_target_slot_id",
        "entity_predicate_slot_match",
        "predicate_contract_match",
        "predicate_match_mode",
        "direct_predicate_overlap",
        "direct_predicate_tokens",
        "discriminative_predicate_tokens",
        "expected_discriminative_predicate_tokens",
        "dependency_binding_safe",
        "dependency_binding_reasons",
        "dependency_root_entity_key",
        "dependency_terminal_entity_key",
        "dependency_path_entity_keys",
        "dependency_hops",
        "slot_id",
        "slot_text",
        "slot_source",
        "slot_hop_index",
        "slot_target_predicate_concepts",
        "slot_expected_value_type",
        "slot_direction",
        "slot_materialization_safe",
        "slot_unsafe_reasons",
        "slot_meta_negative",
        "terminal_slot_gate_passed",
        "terminal_slot_gate_reasons",
        "terminal_answer_slot",
        "terminal_value_type",
        "terminal_slot_stage",
        "terminal_materialization_role",
        "terminal_relation_direct",
        "terminal_relation_intersection",
        "source_sentence_value_bound",
        "source_sentence_join_bound",
        "source_sentence_relation_bound",
        "estimated_endpoint_scan_rows",
        "operator_latency_ms",
        "operator_name",
        "early_schema_bridge_materialization",
        "admission_min_score",
        "admission_min_join_key_overlap",
        "admission_min_relation_overlap",
    )
    return {key: candidate.get(key) for key in fields}


def _candidate_id(candidate: Dict[str, Any]) -> str:
    payload = json.dumps(
        _candidate_trace_payload(candidate),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _candidate_pool_fingerprint(candidates: List[Dict[str, Any]]) -> str:
    payloads = [_candidate_trace_payload(candidate) for candidate in candidates]
    payload = json.dumps(
        payloads,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_pool_snapshot_path(
    snapshot_dir: str,
    question: str,
    snapshot_key: str = "",
) -> str:
    stable_key = str(snapshot_key or question or "")
    key_hash = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()
    return os.path.join(os.path.abspath(os.path.expanduser(snapshot_dir)), f"{key_hash}.json")


def _load_candidate_pool_snapshot(
    snapshot_dir: str,
    question: str,
    snapshot_key: str = "",
) -> Dict[str, Any]:
    if not snapshot_dir:
        return {}
    path = _candidate_pool_snapshot_path(snapshot_dir, question, snapshot_key)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or int(payload.get("version", 0) or 0) != 2:
        return {}
    if str(payload.get("question") or "") != str(question or ""):
        return {}
    if str(payload.get("snapshot_key") or "") != str(snapshot_key or question or ""):
        return {}
    payload["snapshot_path"] = path
    return payload


def _write_candidate_pool_snapshot(
    snapshot_dir: str,
    question: str,
    snapshot_key: str,
    payload: Dict[str, Any],
) -> str:
    stable_key = str(snapshot_key or question or "")
    path = _candidate_pool_snapshot_path(snapshot_dir, question, stable_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serialized = {
        "version": 2,
        "question": str(question or ""),
        "snapshot_key": stable_key,
        **dict(payload),
    }
    temporary_path = f"{path}.{os.getpid()}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(serialized, handle, ensure_ascii=False, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary_path, path)
    return path


def _sort_candidates_for_policy(
    candidates: List[Dict[str, Any]],
    policy: str,
) -> List[Dict[str, Any]]:
    policy = _normalize_cost_admission_policy(policy)
    if policy in {
        "fixed_order",
        "always_on",
        "learned_veto",
        "counterfactual_drop",
        "counterfactual_slate_veto",
    }:
        return sorted(candidates, key=lambda row: int(row.get("input_extra_rank", 0)))
    if policy == "rule":
        return sorted(
            candidates,
            key=lambda row: (
                -len(row.get("uncovered_slot_ids") or []),
                -float(row.get("lineage_confidence", 0.0)),
                -float(row.get("net_utility", 0.0)),
                -float(row.get("utility", 0.0)),
                float(row.get("risk", 0.0)),
                int(row.get("input_extra_rank", 0)),
            ),
        )
    return sorted(
        candidates,
        key=lambda row: (
            -len(row.get("uncovered_slot_ids") or []),
            -float(row.get("lineage_confidence", 0.0)),
            -float(row.get("admission_sort_utility", row.get("net_utility", 0.0))),
            -float(row.get("learned_expected_utility", 0.0)),
            -float(row.get("utility", 0.0)),
            float(row.get("risk", 0.0)),
            int(row.get("input_extra_rank", 0)),
        ),
    )


def _shadow_plan_surface(plan: Dict[str, Any]) -> str:
    parts = [
        str(plan.get("candidate_answer") or ""),
        str(plan.get("answer_type") or ""),
    ]
    parts.extend(str(item or "") for item in list(plan.get("triple_path") or [])[:3])
    parts.extend(str(item or "") for item in list(plan.get("schema_signatures") or [])[:3])
    return " ".join(parts)


def _question_requires_value_type(question: str) -> str:
    question = str(question or "")
    if _DATE_REQUEST_RE.search(question) or _MONTH_REQUEST_RE.search(question):
        return "date"
    if _NUMBER_REQUEST_RE.search(question) or _PERCENT_REQUEST_RE.search(question):
        return "number"
    return ""


def _shadow_plan_has_required_value_type(question: str, plan: Dict[str, Any]) -> Tuple[bool, str]:
    required = _question_requires_value_type(question)
    if not required:
        return True, "not_value_request"
    surface = _shadow_plan_surface(plan)
    if required == "date":
        return bool(_DATE_VALUE_RE.search(surface)), (
            "date_value_present" if _DATE_VALUE_RE.search(surface) else "missing_date_value"
        )
    if required == "number":
        return bool(_NUMBER_VALUE_RE.search(surface)), (
            "number_value_present" if _NUMBER_VALUE_RE.search(surface) else "missing_number_value"
        )
    return True, "not_value_request"


def _shadow_plan_relation_safe(
    plan: Dict[str, Any],
    *,
    question: str,
    state: Optional[Dict[str, Any]] = None,
    calibrated_shadow_admission: bool = False,
) -> Tuple[bool, List[str]]:
    """Reject typed shadow joins that look structural but are likely distractors."""
    state = state or {}
    reasons: List[str] = []
    plan_type = str(plan.get("plan_type") or "")
    endpoint = _as_float(plan.get("endpoint_overlap"), 0.0)
    relation = _as_float(plan.get("relation_overlap"), 0.0)
    schema = _as_float(plan.get("schema_selectivity"), 0.0)
    provenance = _as_float(plan.get("provenance_score"), 0.0)
    best_rank = _as_int(plan.get("best_rank"), 999999)
    worst_rank = _as_int(plan.get("worst_rank"), best_rank)
    evidence_insufficient = bool(state.get("evidence_insufficient"))
    planned_reason = str(state.get("reason") or "")
    hard_deficit = evidence_insufficient or planned_reason in _DEFICIT_PLANNED_REASONS

    value_ok, value_reason = _shadow_plan_has_required_value_type(question, plan)
    if not value_ok:
        reasons.append(value_reason)

    if plan_type == "two_hop_fact_join":
        if endpoint < 0.34:
            reasons.append("counterfactual_endpoint_low")
        if schema < 0.58:
            reasons.append("counterfactual_schema_low")
        if relation < 0.08:
            strong_zero_relation_lineage = (
                bool(calibrated_shadow_admission)
                and hard_deficit
                and endpoint >= 0.95
                and schema >= 0.98
                and provenance >= 0.425
                and best_rank <= 2
                and worst_rank <= 12
            )
            if not strong_zero_relation_lineage:
                reasons.append("zero_relation_without_strong_lineage")
        elif relation < 0.18 and not hard_deficit and provenance < 0.425:
            reasons.append("weak_relation_without_hard_deficit")
        if worst_rank > 24 and relation < 0.18 and provenance < 0.425:
            reasons.append("deep_weak_relation_join")
    elif plan_type == "one_hop_fact_access":
        if not hard_deficit and relation < 0.18:
            reasons.append("one_hop_without_hard_deficit")

    return not reasons, list(dict.fromkeys(reasons))


def _iter_planned_verdicts(planned_retrieval: Dict[str, Any]):
    for round_info in planned_retrieval.get("rounds") or []:
        if not isinstance(round_info, dict):
            continue
        verdict = round_info.get("verifier")
        if isinstance(verdict, dict):
            yield verdict


def _missing_text(planned_retrieval: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in (
        "residual_deficit_queries",
        "residual_deficit_replaced_queries",
    ):
        value = planned_retrieval.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if str(item).strip())
    for verdict in _iter_planned_verdicts(planned_retrieval):
        for key in ("missing", "missing_slots", "queries", "uncertain_anchors"):
            value = verdict.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value if str(item).strip())
            elif value:
                parts.append(str(value))
    return " ".join(parts)


def _planned_state(planned_retrieval: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    planned_retrieval = planned_retrieval or {}
    router = planned_retrieval.get("router") or {}
    if not isinstance(router, dict):
        router = {}
    reason = str(planned_retrieval.get("reason", "") or "")
    question_type = str(
        planned_retrieval.get("question_type")
        or router.get("question_type")
        or ""
    )
    route_profile = str(
        planned_retrieval.get("routing_profile")
        or router.get("profile")
        or ""
    )
    bridge_strength = _as_int(router.get("bridge_strength"), 0)
    terminal_state = planned_retrieval_terminal_state(planned_retrieval)
    evidence_insufficient = bool(
        planned_retrieval.get("evidence_insufficient", False)
        or reason in _HIGH_RISK_PLANNED_REASONS
        or terminal_state["terminal_evidence_gap"]
    )
    bridge_cues = bool(
        planned_retrieval.get("bridge_cues", False)
        or question_type == "bridge_or_compositional"
        or bridge_strength > 0
    )
    long_context = question_type in {
        "long_answer_or_narrative",
        "long_context_evidence",
        "long_context_factoid",
    } or reason in {
        "routed_long_answer_or_narrative",
        "routed_long_context_evidence",
    }
    top2_changed = _as_int(planned_retrieval.get("top2_override_applied"), 0) > 0
    return {
        "reason": reason or "unavailable",
        "question_type": question_type or "unavailable",
        "route_profile": route_profile or "unavailable",
        "bridge_strength": bridge_strength,
        "bridge_cues": bridge_cues,
        "evidence_insufficient": evidence_insufficient,
        "terminal_evidence_gap": terminal_state["terminal_evidence_gap"],
        "expansion_stopped_by_optimizer": terminal_state[
            "expansion_stopped_by_optimizer"
        ],
        "long_context": bool(long_context),
        "top2_changed": bool(top2_changed),
        "round_count": _as_int(planned_retrieval.get("round_count"), 0),
        "query_count": len(planned_retrieval.get("queries") or []),
        "missing_tokens": _tokens(_missing_text(planned_retrieval)),
    }


def _candidate_table_score(doc: str) -> Tuple[float, int]:
    scores = []
    for match in re.finditer(r"\bscore=([0-9]*\.?[0-9]+)", str(doc or "")):
        scores.append(_as_float(match.group(1), 0.0))
    if not scores:
        return 0.0, 0
    return max(scores), len(scores)


def _reason_from(info: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = info.get(key)
        if isinstance(value, dict):
            nested = _reason_from(value, "reason", "state_reason")
            if nested:
                return nested
        elif value:
            return str(value)
    return ""


def _has_relation_gap(relation_fact_join: Dict[str, Any]) -> bool:
    return bool(
        relation_fact_join.get("answerability_gap")
        or relation_fact_join.get("relation_gap")
        or relation_fact_join.get("polarity_gap")
    )


def _has_selective_support_gap(selective_support_join: Dict[str, Any]) -> bool:
    reason = _reason_from(selective_support_join, "state_reason", "reason")
    if reason in _DEFICIT_STATE_REASONS:
        return True
    reader_projection = selective_support_join.get("reader_projection") or {}
    reader_reason = _reason_from(reader_projection, "reason")
    return reader_reason in {"no_candidate_accepted", "no_join_docs"}


def _admission_deficit_signal(
    *,
    state: Dict[str, Any],
    relation_fact_join: Dict[str, Any],
    selective_support_join: Dict[str, Any],
    query_schema_memory: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Return whether extra shadow evidence is justified by current evidence state.

    Cost admission is intentionally conservative: it should repair a concrete
    evidence deficit, not re-optimize every reader context. This protects runs
    where the original retrieval already provides a stable top-k.
    """
    reasons: List[str] = []
    if state.get("reason") in _DEFICIT_PLANNED_REASONS:
        reasons.append(f"planned:{state.get('reason')}")
    if bool(state.get("evidence_insufficient")) and bool(state.get("bridge_cues")):
        reasons.append("planned:evidence_insufficient_bridge")
    if state.get("missing_tokens"):
        reasons.append("planned:missing_slots")
    if _has_relation_gap(relation_fact_join):
        reasons.append("relation_fact_join:gap")
    if _has_selective_support_gap(selective_support_join):
        reasons.append("selective_support_join:gap")
    qschema_reason = _reason_from(query_schema_memory, "reason")
    if qschema_reason in {"no_accepted_candidates", "candidates_generated"} and bool(state.get("bridge_cues")):
        reasons.append(f"query_schema_memory:{qschema_reason}")
    return bool(reasons), reasons


def _shadow_portfolio_decision(
    source_counts: Counter,
    *,
    max_extra_docs: int,
) -> Tuple[bool, str, List[str]]:
    """Decide whether a typed shadow table should join the reader context.

    The previous admission model judged a shadow table in isolation. Logs showed
    that the table is helpful when it is the only repair channel, but harmful
    when stacked on top of planned retrieval or unstructured extra passages.
    This portfolio gate keeps the top-k stable and only allows the low-cost
    typed table when the existing evidence context has room for that channel.
    """
    counts = Counter({str(k): int(v) for k, v in (source_counts or {}).items() if int(v or 0) > 0})
    total_existing = sum(counts.values())
    active_sources = {source for source, count in counts.items() if count > 0}
    unsafe_sources = sorted(active_sources & _SHADOW_UNSAFE_COMPANION_SOURCES)
    has_relation_table = counts.get("relation_fact_table", 0) > 0

    if total_existing <= 0:
        profile = "no_existing_extra"
    elif active_sources <= _SHADOW_SAFE_COMPANION_SOURCES:
        profile = "path_evidence_only"
    elif has_relation_table and active_sources <= {"relation_fact_table"}:
        profile = "relation_table_already_visible"
    else:
        profile = "mixed_or_unstructured_extra"

    reject_reasons: List[str] = []
    if unsafe_sources:
        reject_reasons.append("unstructured_extra_present:" + ",".join(unsafe_sources[:4]))
    if has_relation_table:
        reject_reasons.append("relation_fact_table_already_visible")
    if total_existing > max(0, int(max_extra_docs)) and profile != "path_evidence_only":
        reject_reasons.append("extra_budget_already_occupied")
    if active_sources - _SHADOW_SAFE_COMPANION_SOURCES - {"relation_fact_table"}:
        reject_reasons.append("non_shadow_repair_channel_present")

    return not reject_reasons, profile, list(dict.fromkeys(reject_reasons))


def _best_calibrated_shadow_signal(
    typed_fact_join_shadow: Dict[str, Any],
    *,
    min_plan_utility: float,
    min_plan_net_utility: float,
    max_plan_risk: float,
    question: str = "",
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    plans = list((typed_fact_join_shadow or {}).get("top_plans") or [])
    if not plans:
        return False, "no_shadow_plans", {}

    best: Dict[str, Any] = {}
    best_tuple = (-1.0, -1.0, 1.0)
    for plan in plans[:24]:
        if not isinstance(plan, dict) or not bool(plan.get("accepted", False)):
            continue
        utility = _as_float(plan.get("utility"), 0.0)
        net = _as_float(plan.get("net_utility"), 0.0)
        risk = _as_float(plan.get("risk"), 1.0)
        endpoint = _as_float(plan.get("endpoint_overlap"), 0.0)
        relation = _as_float(plan.get("relation_overlap"), 0.0)
        schema = _as_float(plan.get("schema_selectivity"), 0.0)
        provenance = _as_float(plan.get("provenance_score"), 0.0)
        best_rank = _as_int(plan.get("best_rank"), 999999)
        worst_rank = _as_int(plan.get("worst_rank"), best_rank)
        current_tuple = (net, utility, -risk)
        if current_tuple > best_tuple:
            best_tuple = current_tuple
            best = {
                "plan_type": str(plan.get("plan_type") or ""),
                "utility": _round_float(utility),
                "net_utility": _round_float(net),
                "risk": _round_float(risk),
                "endpoint_overlap": _round_float(endpoint),
                "relation_overlap": _round_float(relation),
                "schema_selectivity": _round_float(schema),
                "provenance_score": _round_float(provenance),
                "best_rank": best_rank,
                "worst_rank": worst_rank,
                "candidate_answer": str(plan.get("candidate_answer") or "")[:120],
                "answer_type": str(plan.get("answer_type") or ""),
                "triple_path": list(plan.get("triple_path") or [])[:3],
                "schema_signatures": list(plan.get("schema_signatures") or [])[:3],
            }

    if not best:
        return False, "no_accepted_shadow_plan", {}

    utility = _as_float(best.get("utility"), 0.0)
    net = _as_float(best.get("net_utility"), 0.0)
    risk = _as_float(best.get("risk"), 1.0)
    endpoint = _as_float(best.get("endpoint_overlap"), 0.0)
    relation = _as_float(best.get("relation_overlap"), 0.0)
    schema = _as_float(best.get("schema_selectivity"), 0.0)

    if utility < max(float(min_plan_utility), 0.58):
        return False, "best_plan_utility_low", best
    if net < max(float(min_plan_net_utility), 0.16):
        return False, "best_plan_net_low", best
    if risk > min(float(max_plan_risk), 0.52):
        return False, "best_plan_risk_high", best
    if endpoint < 0.42:
        return False, "best_plan_endpoint_low", best
    if schema < 0.58:
        return False, "best_plan_schema_low", best
    safe, safety_reasons = _shadow_plan_relation_safe(
        best,
        question=question,
        state=state,
        calibrated_shadow_admission=True,
    )
    if not safe:
        best["counterfactual_reject_reasons"] = list(safety_reasons)
        return False, "best_plan_counterfactual_unsafe", best
    return True, "calibrated_shadow_signal", best


def _calibrated_portfolio_override_decision(
    source_counts: Counter,
    *,
    state: Dict[str, Any],
    shadow_signal: bool,
    max_extra_docs: int,
) -> Tuple[bool, str, List[str]]:
    counts = Counter({str(k): int(v) for k, v in (source_counts or {}).items() if int(v or 0) > 0})
    active_sources = {source for source, count in counts.items() if count > 0}
    total_existing = sum(counts.values())
    reject_reasons: List[str] = []

    if not shadow_signal:
        reject_reasons.append("no_calibrated_shadow_signal")
    if state.get("long_context"):
        reject_reasons.append("long_context")
    if total_existing > max(1, int(max_extra_docs) + 1):
        reject_reasons.append("extra_budget_too_full")
    disallowed = sorted(active_sources - _CALIBRATED_PORTFOLIO_COMPANION_SOURCES)
    if disallowed:
        reject_reasons.append("disallowed_companion_sources:" + ",".join(disallowed[:4]))

    profile = "calibrated_planned_companion" if not reject_reasons else "calibrated_blocked"
    return not reject_reasons, profile, list(dict.fromkeys(reject_reasons))


def _format_typed_shadow_table(question: str, plan: Dict[str, Any]) -> str:
    """Compact reader evidence for a shadow typed fact-join plan.

    We expose the derived join rows and provenance instead of injecting whole
    passages. That keeps the reader evidence structured and avoids replacing a
    precise answer span with a long distractor passage.
    """
    lines = [
        "Typed Fact Join Candidate Table",
        f"Question: {question}",
        "Namespace: typed_fact_join_shadow_table; rows are derived from OpenIE fact triples backed by base_passage lineage.",
        "Use this table only as supplemental support evidence when the visible passages miss a bridge/entity/relation slot.",
        "Do not treat this table as an answer proposal; verify the answer from the fact rows and visible passages.",
        "Support plan:",
        (
            "- expected_type={answer_type} plan={plan_type} utility={utility:.4f} "
            "net_utility={net:.4f} risk={risk:.4f} endpoint={endpoint:.4f} "
            "relation={relation:.4f} schema_selectivity={schema:.4f} provenance={provenance:.4f} "
            "best_rank={best_rank} worst_rank={worst_rank}"
        ).format(
            answer_type=str(plan.get("answer_type") or "entity"),
            plan_type=str(plan.get("plan_type") or "unknown"),
            utility=_as_float(plan.get("utility"), 0.0),
            net=_as_float(plan.get("net_utility"), 0.0),
            risk=_as_float(plan.get("risk"), 0.0),
            endpoint=_as_float(plan.get("endpoint_overlap"), 0.0),
            relation=_as_float(plan.get("relation_overlap"), 0.0),
            schema=_as_float(plan.get("schema_selectivity"), 0.0),
            provenance=_as_float(plan.get("provenance_score"), 0.0),
            best_rank=_as_int(plan.get("best_rank"), 0),
            worst_rank=_as_int(plan.get("worst_rank"), 0),
        ),
        "Rows:",
    ]
    titles = list(plan.get("titles") or [])
    triples = list(plan.get("triple_path") or [])
    for idx, triple in enumerate(triples[:2], start=1):
        title = str(titles[idx - 1] if idx - 1 < len(titles) else "")[:120]
        lines.append(
            f"- {idx}. source_title=\"{title}\" triple=\"{str(triple)[:220]}\""
        )
    if plan.get("bridge_entity"):
        lines.append(f"Bridge entity: {str(plan.get('bridge_entity'))[:120]}")
    return "\n".join(lines)


def _infer_source(
    doc: str,
    *,
    planned_retrieval: Dict[str, Any],
    relation_fact_join: Dict[str, Any],
    dual_channel_projection: Dict[str, Any],
    support_set_completion: Dict[str, Any],
    path_evidence: Dict[str, Any],
) -> str:
    stripped = str(doc or "").strip()
    title = _title(stripped)
    if stripped.startswith("Directional Fact Join Candidate Table"):
        return "relation_fact_table"
    if stripped.startswith("Structured Evidence Chain"):
        return "structured_chain"
    if stripped.startswith("Multi-Granular Evidence Memory"):
        return "multigranular_memory"

    title_sets = {
        "planned_retrieval": set(planned_retrieval.get("appended_titles") or []),
        "dual_channel_projection": set(
            ((dual_channel_projection.get("reader_projection") or {}).get("appended_titles") or [])
        ),
        "relation_fact_join": set(
            ((relation_fact_join.get("reader_projection") or {}).get("appended_titles") or [])
        ),
        "support_set_completion": set(support_set_completion.get("accepted_titles") or []),
        "path_evidence": set(path_evidence.get("appended_titles") or []),
    }
    for source, titles in title_sets.items():
        if title and title in titles:
            return source
    return "extra_evidence"


def _source_prior(source: str) -> float:
    return {
        "typed_fact_join_shadow": 0.14,
        "relation_fact_table": 0.13,
        "query_schema_memory": 0.11,
        "entity_predicate_fact_join": 0.16,
        "relation_fact_join": 0.10,
        "support_set_completion": 0.08,
        "planned_retrieval": 0.07,
        "dual_channel_projection": 0.05,
        "path_evidence": 0.05,
        "structured_chain": 0.05,
        "multigranular_memory": 0.03,
    }.get(source, 0.02)


def _collect_query_schema_memory_candidates(
    *,
    question: str,
    query_schema_memory: Dict[str, Any],
    get_doc_content: Optional[Callable[[int], str]],
    existing_doc_keys: set,
    state: Dict[str, Any],
    lineage_context: Dict[str, Any],
    enabled: bool,
    max_candidates: int,
    max_original_rank: int,
    min_score: float,
    min_schema_idf: float,
    min_query_overlap: float,
    min_join_key_overlap: float,
    min_relation_overlap: float,
    require_answer_type_value: bool,
    require_entity_bound: bool = True,
    require_lineage_binding: bool = True,
    allow_exact_join_key_relation_relaxation: bool = False,
    exact_join_key_min_schema_idf: float = 0.65,
    exact_join_key_min_relation_overlap: float = 0.05,
    compact_projection: bool = False,
    require_residual_anchor_binding: bool = False,
    safe_max_original_rank: int = 48,
    deep_rank_min_score: float = 0.70,
    cross_schema_max_endpoint_scan_rows: int = 48,
    validated_tuple_materialization: bool = False,
    validated_tuple_max_candidates: int = 1,
    validated_tuple_min_score: float = 0.58,
    validated_tuple_min_relation_overlap: float = 0.05,
    validated_tuple_require_predicate_contract: bool = True,
    validated_tuple_require_terminal_slot: bool = True,
    validated_tuple_require_source_sentence: bool = True,
    validated_tuple_min_token_savings: int = 1,
    validated_tuple_block_reader_visible_source: bool = True,
    active_schema_bridge_tuple_executor: bool = False,
    active_schema_bridge_tuple_min_score: float = 0.52,
    active_schema_bridge_tuple_min_bound_signals: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Materialize only high-confidence, join-connected schema candidates.

    Query schema memory remains rank-stable. This collector exposes at most a
    small number of candidates to the reader-side cost model after retrieval
    has ended with a concrete evidence deficit.
    """
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "reason": "disabled" if not enabled else "not_evaluated",
        "raw_candidate_count": len((query_schema_memory or {}).get("candidate_items") or []),
        "candidate_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "reject_reason_counts": {},
        "accepted_doc_ids": [],
        "top_candidates": [],
        "entity_bound_required": bool(require_entity_bound),
        "lineage_binding_required": bool(require_lineage_binding),
        "relation_relaxation_enabled": bool(allow_exact_join_key_relation_relaxation),
        "relation_relaxed_count": 0,
        "relation_relaxed_candidate_count": 0,
        "validated_topk_count": 0,
        "validated_topk_doc_ids": [],
        "validated_topk_candidates": [],
        "compact_projection_enabled": bool(compact_projection),
        "require_residual_anchor_binding": bool(require_residual_anchor_binding),
        "residual_anchor_groups": [],
        "residual_anchor_bound_candidate_count": 0,
        "safe_max_original_rank": int(safe_max_original_rank),
        "deep_rank_min_score": _round_float(deep_rank_min_score),
        "deep_rank_candidate_count": 0,
        "projected_candidate_count": 0,
        "original_doc_token_count": 0,
        "projected_doc_token_count": 0,
        "projection_token_savings": 0,
        "cross_schema_raw_candidate_count": sum(
            bool(item.get("cross_schema_join"))
            for item in (query_schema_memory or {}).get("candidate_items") or []
            if isinstance(item, dict)
        ),
        "cross_schema_candidate_count": 0,
        "cross_schema_accepted_count": 0,
        "cross_schema_slot_bound_count": 0,
        "cross_schema_predicate_contract_count": 0,
        "cross_schema_endpoint_budget": int(cross_schema_max_endpoint_scan_rows),
        "cross_schema_validated_topk_count": 0,
        "validated_tuple_materialization_enabled": bool(validated_tuple_materialization),
        "validated_tuple_candidate_count": 0,
        "validated_tuple_accepted_count": 0,
        "validated_tuple_rejected_count": 0,
        "validated_tuple_reject_reason_counts": {},
        "validated_tuple_doc_ids": [],
        "validated_tuple_top_candidates": [],
        "validated_tuple_projection_token_savings": 0,
        "validated_tuple_original_doc_token_count": 0,
        "validated_tuple_projected_doc_token_count": 0,
        "validated_tuple_min_score": _round_float(validated_tuple_min_score),
        "validated_tuple_min_relation_overlap": _round_float(
            validated_tuple_min_relation_overlap
        ),
        "validated_tuple_require_predicate_contract": bool(
            validated_tuple_require_predicate_contract
        ),
        "validated_tuple_require_terminal_slot": bool(
            validated_tuple_require_terminal_slot
        ),
        "validated_tuple_require_source_sentence": bool(
            validated_tuple_require_source_sentence
        ),
        "validated_tuple_min_token_savings": int(
            max(0, int(validated_tuple_min_token_savings))
        ),
        "validated_tuple_block_reader_visible_source": bool(
            validated_tuple_block_reader_visible_source
        ),
        "validated_tuple_terminal_candidate_count": 0,
        "validated_tuple_bridge_only_candidate_count": 0,
        "validated_tuple_source_sentence_bound_count": 0,
        "validated_tuple_source_visible_reject_count": 0,
        "validated_tuple_reader_visible_projection_count": 0,
        "schema_tuple_terminal_veto_count": 0,
        "schema_tuple_terminal_veto_reason_counts": {},
        "covered_by_reader_visible_count": 0,
        "covered_by_reader_visible_doc_ids": [],
        "covered_by_reader_visible_candidates": [],
        "active_schema_bridge_tuple_executor_enabled": bool(
            active_schema_bridge_tuple_executor
        ),
        "active_schema_bridge_tuple_min_score": _round_float(
            active_schema_bridge_tuple_min_score
        ),
        "active_schema_bridge_tuple_min_bound_signals": int(
            max(1, int(active_schema_bridge_tuple_min_bound_signals))
        ),
        "active_schema_bridge_tuple_candidate_count": 0,
        "active_schema_bridge_tuple_accepted_count": 0,
        "active_schema_bridge_tuple_relaxed_count": 0,
        "active_schema_bridge_tuple_relaxed_reason_counts": {},
        "validated_tuple_source_sentence_trace_fallback_count": 0,
    }
    if not enabled:
        return [], info
    if not query_schema_memory or not get_doc_content:
        info["reason"] = "no_schema_memory_or_doc_lookup"
        return [], info
    if not bool(state.get("evidence_insufficient")) and not lineage_context.get(
        "required_uncovered_slot_ids"
    ):
        info["reason"] = "current_evidence_sufficient"
        return [], info
    if not bool(state.get("bridge_cues")):
        info["reason"] = "no_bridge_cues"
        return [], info
    if bool(state.get("long_context")):
        info["reason"] = "long_context_blocked"
        return [], info

    question_tokens = _tokens(question)
    deficit_tokens = set(state.get("missing_tokens") or set())
    relation_target_tokens = question_tokens | deficit_tokens
    anchor_groups = _schema_join_anchor_groups(question, lineage_context)
    info["residual_anchor_groups"] = [sorted(group) for group in anchor_groups[:12]]
    date_requested = bool(_DATE_REQUEST_RE.search(str(question or "")))
    number_requested = bool(_NUMBER_REQUEST_RE.search(str(question or "")))
    reject_counts: Counter = Counter()
    tuple_reject_counts: Counter = Counter()
    terminal_veto_counts: Counter = Counter()
    candidates: List[Dict[str, Any]] = []
    tuple_candidates: List[Dict[str, Any]] = []
    tuple_doc_keys: set = set()
    validated_topk: List[Dict[str, Any]] = []
    covered_by_reader_visible: List[Dict[str, Any]] = []
    relation_relaxed_count = 0
    active_bridge_relaxed_counts: Counter = Counter()

    for item in (query_schema_memory.get("candidate_items") or []):
        reasons: List[str] = []
        try:
            doc_id = int(item.get("doc_id"))
        except (TypeError, ValueError):
            reject_counts["invalid_doc_id"] += 1
            continue
        score = _as_float(item.get("score"), 0.0)
        schema_idf = _as_float(item.get("schema_idf"), 0.0)
        query_overlap = _as_float(item.get("query_overlap"), 0.0)
        join_key_overlap = _as_float(item.get("join_key_overlap"), 0.0)
        original_rank = _as_int(item.get("original_rank"), 10**9)
        is_cross_schema = bool(item.get("cross_schema_join"))
        schema = item.get("schema") or []
        relation = str(schema[1] if isinstance(schema, (list, tuple)) and len(schema) > 1 else "")
        predicate_text = str(item.get("predicate") or "")
        relation_tokens = _tokens(predicate_text if is_cross_schema and predicate_text else relation)
        computed_relation_overlap = (
            _coverage(relation_tokens, relation_target_tokens)
            if relation_tokens else 0.0
        )
        relation_overlap = (
            max(computed_relation_overlap, _as_float(item.get("relation_overlap"), 0.0))
            if is_cross_schema
            else computed_relation_overlap
        )
        join_key_exact = bool(item.get("join_key_exact"))
        anchor_binding = _schema_join_anchor_binding(item, anchor_groups)
        slot_anchor_bound = bool(
            _as_float(item.get("slot_anchor_overlap"), 0.0) >= 0.67
            or item.get("branch_binding_passed") is True
        )
        anchor_bound = bool(anchor_binding.get("bound") or slot_anchor_bound)
        endpoint_scan_rows = max(
            1, _as_int(item.get("estimated_endpoint_scan_rows"), 1)
        )
        cross_schema_endpoint_budget_passed = bool(
            endpoint_scan_rows <= max(1, int(cross_schema_max_endpoint_scan_rows))
        )
        cross_schema_predicate_contract = bool(
            item.get("predicate_contract_match")
            or item.get("relation_signature_match")
            or item.get("relation_signature_compatible")
        )
        already_reader_visible = bool(item.get("already_top5"))
        source_rank_reader_visible = bool(
            already_reader_visible or original_rank < 5
        )
        deep_rank = original_rank > int(safe_max_original_rank)
        doc: Optional[str] = None
        doc_key = ""
        triple = item.get("triple") or []
        tuple_value = str(
            item.get("answer_value")
            or (
                triple[2]
                if isinstance(triple, (list, tuple)) and len(triple) > 2
                else ""
            )
        ).strip()
        has_fact_tuple = isinstance(triple, (list, tuple)) and len(triple) == 3 and any(
            str(part or "").strip() for part in triple
        )
        tuple_like_candidate = bool(
            tuple_value
            and (has_fact_tuple or str(item.get("answer_value") or "").strip())
        )
        slot_id = str(
            item.get("slot_id") or item.get("entity_predicate_target_slot_id") or ""
        ).strip()
        active_relation_validated = bool(
            cross_schema_predicate_contract
            or item.get("predicate_contract_match")
            or item.get("relation_signature_match")
            or item.get("relation_signature_compatible")
            or relation_overlap >= float(exact_join_key_min_relation_overlap)
        )
        active_bridge_bound_signal_count = sum(
            int(flag)
            for flag in (
                bool(item.get("entity_bound")),
                bool(join_key_exact),
                bool(anchor_bound or slot_id),
                bool(active_relation_validated),
                bool(score >= float(active_schema_bridge_tuple_min_score)),
            )
        )
        active_schema_bridge_tuple = bool(
            active_schema_bridge_tuple_executor
            and validated_tuple_materialization
            and tuple_like_candidate
            and bool(item.get("entity_bound"))
            and bool(join_key_exact)
            and bool(anchor_bound or slot_id)
            and bool(active_relation_validated)
            and score >= float(active_schema_bridge_tuple_min_score)
            and original_rank <= int(max_original_rank)
            and active_bridge_bound_signal_count
            >= max(1, int(active_schema_bridge_tuple_min_bound_signals))
            and not (
                is_cross_schema
                and (
                    item.get("dependency_binding_safe") is False
                    or not cross_schema_endpoint_budget_passed
                    or (
                        validated_tuple_require_predicate_contract
                        and not cross_schema_predicate_contract
                    )
                )
            )
        )
        if active_schema_bridge_tuple:
            info["active_schema_bridge_tuple_candidate_count"] = int(
                info.get("active_schema_bridge_tuple_candidate_count", 0) or 0
            ) + 1
        reader_visible_tuple_projection = bool(
            active_schema_bridge_tuple and source_rank_reader_visible
        )
        if is_cross_schema:
            relation_relaxed = bool(
                allow_exact_join_key_relation_relaxation
                and join_key_exact
                and anchor_bound
                and cross_schema_endpoint_budget_passed
                and cross_schema_predicate_contract
                and relation_overlap >= float(exact_join_key_min_relation_overlap)
                and relation_overlap < float(min_relation_overlap)
            )
        else:
            relation_relaxed = bool(
                allow_exact_join_key_relation_relaxation
                and join_key_exact
                and anchor_bound
                and schema_idf >= float(exact_join_key_min_schema_idf)
                and relation_overlap >= float(exact_join_key_min_relation_overlap)
                and relation_overlap < float(min_relation_overlap)
            )

        if require_entity_bound and not bool(item.get("entity_bound")):
            reasons.append("entity_binding_missing")
        if require_residual_anchor_binding and not anchor_bound:
            reasons.append("residual_anchor_binding_missing")
        if is_cross_schema and not cross_schema_predicate_contract:
            reasons.append("cross_schema_predicate_contract_missing")
        if is_cross_schema and item.get("dependency_binding_safe") is False:
            reasons.append("cross_schema_dependency_binding_unsafe")
        if is_cross_schema and not cross_schema_endpoint_budget_passed:
            reasons.append("cross_schema_endpoint_scan_too_high")
        if original_rank > int(max_original_rank):
            reasons.append("original_rank_too_deep")
        elif deep_rank:
            if score < max(float(min_score), float(deep_rank_min_score)):
                reasons.append("deep_rank_score_low")
            if not is_cross_schema and schema_idf < max(
                float(min_schema_idf),
                float(exact_join_key_min_schema_idf),
            ):
                reasons.append("deep_rank_schema_idf_low")
        if score < float(min_score) and not active_schema_bridge_tuple:
            reasons.append("schema_score_low")
        elif score < float(min_score) and active_schema_bridge_tuple:
            active_bridge_relaxed_counts["schema_score_low"] += 1
        if (
            not is_cross_schema
            and schema_idf < float(min_schema_idf)
            and not active_schema_bridge_tuple
        ):
            reasons.append("schema_idf_low")
        elif (
            not is_cross_schema
            and schema_idf < float(min_schema_idf)
            and active_schema_bridge_tuple
        ):
            active_bridge_relaxed_counts["schema_idf_low"] += 1
        if query_overlap < float(min_query_overlap) and not active_schema_bridge_tuple:
            reasons.append("query_overlap_low")
        elif query_overlap < float(min_query_overlap) and active_schema_bridge_tuple:
            active_bridge_relaxed_counts["query_overlap_low"] += 1
        if join_key_overlap < float(min_join_key_overlap):
            reasons.append("join_key_overlap_low")
        if (
            relation_overlap < float(min_relation_overlap)
            and not relation_relaxed
            and not active_schema_bridge_tuple
        ):
            reasons.append("relation_overlap_low")
        elif (
            relation_overlap < float(min_relation_overlap)
            and not relation_relaxed
            and active_schema_bridge_tuple
        ):
            active_bridge_relaxed_counts["relation_overlap_low"] += 1

        if validated_tuple_materialization and len(tuple_candidates) < max(
            0, int(validated_tuple_max_candidates)
        ):
            tuple_reasons: List[str] = []
            if not has_fact_tuple and not str(item.get("answer_value") or "").strip():
                tuple_reasons.append("missing_fact_tuple")
            if (
                validated_tuple_block_reader_visible_source
                and source_rank_reader_visible
                and not reader_visible_tuple_projection
            ):
                tuple_reasons.append("source_already_reader_visible")
            if require_entity_bound and not bool(item.get("entity_bound")):
                tuple_reasons.append("entity_binding_missing")
            if require_residual_anchor_binding and not (
                anchor_bound or str(item.get("slot_id") or "").strip()
            ):
                tuple_reasons.append("residual_anchor_binding_missing")
            if (
                validated_tuple_require_predicate_contract
                and is_cross_schema
                and not cross_schema_predicate_contract
            ):
                tuple_reasons.append("cross_schema_predicate_contract_missing")
            if is_cross_schema and item.get("dependency_binding_safe") is False:
                tuple_reasons.append("cross_schema_dependency_binding_unsafe")
            if is_cross_schema and not cross_schema_endpoint_budget_passed:
                tuple_reasons.append("cross_schema_endpoint_scan_too_high")
            if (
                query_overlap < float(min_query_overlap)
                and not (anchor_bound or join_key_exact)
                and not active_schema_bridge_tuple
            ):
                tuple_reasons.append("query_overlap_low")
            if join_key_overlap < float(min_join_key_overlap) and not join_key_exact:
                tuple_reasons.append("join_key_overlap_low")
            if (
                relation_overlap < float(validated_tuple_min_relation_overlap)
                and not relation_relaxed
                and not cross_schema_predicate_contract
                and not active_schema_bridge_tuple
            ):
                tuple_reasons.append("relation_overlap_low")
            tuple_score_floor = max(0.0, float(validated_tuple_min_score))
            if score < tuple_score_floor and not (
                (cross_schema_predicate_contract and anchor_bound)
                or active_schema_bridge_tuple
            ) and not (
                cross_schema_predicate_contract
                and anchor_bound
                and score >= max(0.0, tuple_score_floor - 0.08)
            ):
                tuple_reasons.append("schema_score_low")
            tuple_rank_limit = max(
                int(max_original_rank),
                int(safe_max_original_rank) * 2,
            )
            if original_rank > tuple_rank_limit and not (
                ((cross_schema_predicate_contract and anchor_bound) or active_schema_bridge_tuple)
                and score >= max(tuple_score_floor, float(deep_rank_min_score))
            ):
                tuple_reasons.append("original_rank_too_deep")
            elif (
                deep_rank
                and not active_schema_bridge_tuple
                and score < max(tuple_score_floor, float(deep_rank_min_score) - 0.04)
            ):
                tuple_reasons.append("deep_rank_score_low")

            if not tuple_reasons:
                try:
                    doc = str(get_doc_content(doc_id) or "")
                    doc_key = _doc_key(doc)
                except Exception:
                    doc = None
                    tuple_reasons.append("doc_lookup_failed")
            if (
                not tuple_reasons
                and validated_tuple_block_reader_visible_source
                and doc_key
                and doc_key in existing_doc_keys
                and not reader_visible_tuple_projection
            ):
                tuple_reasons.append("source_already_reader_visible")
            tuple_surface = " ".join(str(part) for part in triple) + " " + str(
                item.get("answer_value") or ""
            )
            if require_answer_type_value and doc is not None:
                typed_surface = tuple_surface + " " + doc
                if date_requested and not _DATE_VALUE_RE.search(typed_surface):
                    tuple_reasons.append("missing_date_value")
                if number_requested and not _NUMBER_VALUE_RE.search(typed_surface):
                    tuple_reasons.append("missing_number_value")

            if not tuple_reasons and doc is not None:
                lineage_metadata = (
                    _entity_predicate_contract_lineage_metadata(item, lineage_context)
                    if is_cross_schema
                    else _candidate_lineage_metadata(
                        doc,
                        "query_schema_memory",
                        lineage_context,
                    )
                )
                if (
                    is_cross_schema
                    and require_lineage_binding
                    and lineage_context.get("required_uncovered_slot_ids")
                    and not lineage_metadata.get("uncovered_slot_ids")
                ):
                    fallback_lineage = _candidate_lineage_metadata(
                        doc,
                        "query_schema_memory",
                        lineage_context,
                    )
                    if fallback_lineage.get("uncovered_slot_ids"):
                        lineage_metadata = fallback_lineage
                if (
                    require_lineage_binding
                    and lineage_context.get("required_uncovered_slot_ids")
                    and not lineage_metadata.get("uncovered_slot_ids")
                    and not anchor_bound
                    and not str(item.get("slot_id") or "").strip()
                ):
                    tuple_reasons.append("no_uncovered_slot_binding")

            if not tuple_reasons and doc is not None:
                tuple_value = str(
                    item.get("answer_value")
                    or (
                        triple[2]
                        if isinstance(triple, (list, tuple)) and len(triple) > 2
                        else ""
                    )
                ).strip()
                terminal_item = dict(item)
                if active_schema_bridge_tuple:
                    terminal_item["active_schema_bridge_tuple_executor"] = True
                    terminal_item["residual_anchor_bound"] = bool(anchor_bound)
                terminal_ok, terminal_metadata = _schema_tuple_terminal_slot_decision(
                    question=question,
                    item=terminal_item,
                    value=tuple_value,
                    lineage_context=lineage_context,
                    lineage_metadata=lineage_metadata,
                )
                if terminal_ok:
                    info["validated_tuple_terminal_candidate_count"] += 1
                else:
                    info["validated_tuple_bridge_only_candidate_count"] += 1
                if validated_tuple_require_terminal_slot and not terminal_ok:
                    tuple_reasons.extend(
                        terminal_metadata.get("terminal_slot_gate_reasons")
                        or ["non_terminal_tuple"]
                    )

            if not tuple_reasons and doc is not None:
                materialized_doc, projection_metadata = (
                    _format_validated_schema_tuple_projection(
                        item,
                        doc,
                        question=question,
                        question_tokens=question_tokens,
                        deficit_tokens=deficit_tokens,
                        relation_tokens=relation_tokens,
                        anchor_binding=anchor_binding,
                    )
                )
                if validated_tuple_require_source_sentence:
                    source_sentence_signal_count = sum(
                        int(bool(projection_metadata.get(key)))
                        for key in (
                            "source_sentence_value_bound",
                            "source_sentence_join_bound",
                            "source_sentence_relation_bound",
                        )
                    )
                    trace_fallback = bool(
                        active_schema_bridge_tuple
                        and terminal_ok
                        and source_sentence_signal_count >= 1
                    )
                    projection_metadata["source_sentence_trace_fallback"] = bool(
                        trace_fallback
                    )
                    if trace_fallback:
                        info["validated_tuple_source_sentence_trace_fallback_count"] = int(
                            info.get(
                                "validated_tuple_source_sentence_trace_fallback_count",
                                0,
                            )
                            or 0
                        ) + 1
                    else:
                        if not projection_metadata.get("source_sentence_value_bound"):
                            tuple_reasons.append("source_sentence_value_unbound")
                        if not projection_metadata.get("source_sentence_join_bound"):
                            tuple_reasons.append("source_sentence_join_unbound")
                        if not projection_metadata.get("source_sentence_relation_bound"):
                            tuple_reasons.append("source_sentence_relation_unbound")
                if not tuple_reasons and reader_visible_tuple_projection:
                    reader_visible_reasons, reader_visible_metadata = (
                        _schema_tuple_reader_visible_projection_reasons(
                            question=question,
                            item=item,
                            value=tuple_value,
                            terminal_metadata=terminal_metadata,
                            projection_metadata=projection_metadata,
                        )
                    )
                    projection_metadata.update(reader_visible_metadata)
                    if reader_visible_reasons:
                        tuple_reasons.extend(reader_visible_reasons)
                if (
                    int(projection_metadata.get("projection_token_savings", 0) or 0)
                    < max(0, int(validated_tuple_min_token_savings))
                ):
                    tuple_reasons.append("tuple_projection_token_savings_low")

            if not tuple_reasons and doc is not None:
                projected_key = _doc_key(materialized_doc)
                if not projected_key or projected_key in existing_doc_keys or projected_key in tuple_doc_keys:
                    tuple_reasons.append("duplicate_or_empty_tuple_doc")
                else:
                    tuple_doc_keys.add(projected_key)
                    tuple_lineage_confidence = _clamp(
                        _as_float(lineage_metadata.get("lineage_confidence"), 0.0)
                    )
                    if anchor_bound:
                        tuple_lineage_confidence = max(
                            tuple_lineage_confidence,
                            _clamp(_as_float(anchor_binding.get("score"), 0.0)),
                            0.42,
                        )
                    tuple_coverage_gain = max(
                        _clamp(_as_float(lineage_metadata.get("coverage_gain"), 0.0)),
                        0.38 if (anchor_bound or str(item.get("slot_id") or "")) else 0.18,
                    )
                    metadata = {
                        "schema_score": score,
                        "schema_idf": schema_idf,
                        "query_schema_overlap": query_overlap,
                        "join_key_overlap": join_key_overlap,
                        "relation_overlap": relation_overlap,
                        "relation_relaxed_by_exact_join_key": bool(relation_relaxed),
                        "schema_frequency": _as_int(item.get("schema_frequency"), 0),
                        "catalog_selectivity": _round_float(
                            item.get("catalog_selectivity", 0.0)
                        ),
                        "endpoint_access_selectivity": _round_float(
                            item.get("endpoint_access_selectivity", 0.0)
                        ),
                        "estimated_endpoint_scan_rows": endpoint_scan_rows,
                        "operator_latency_ms": _round_float(
                            item.get("operator_latency_ms", 0.0)
                        ),
                        "operator_name": str(
                            item.get("operator_name")
                            or "schema_hash_lookup_with_join_key_probe"
                        ),
                        "original_rank": original_rank,
                        "schema": list(schema) if isinstance(schema, (list, tuple)) else schema,
                        "entity_bound": bool(item.get("entity_bound")),
                        "join_key_exact": join_key_exact,
                        "active_join_role": str(item.get("active_join_role") or ""),
                        "candidate_join_role": str(item.get("candidate_join_role") or ""),
                        "matched_join_tokens": list(item.get("matched_join_tokens") or []),
                        "active_join_key_tokens": list(
                            item.get("active_join_key_tokens") or []
                        ),
                        "candidate_join_key_tokens": list(
                            item.get("candidate_join_key_tokens") or []
                        ),
                        "residual_anchor_bound": bool(anchor_bound),
                        "residual_anchor_score": _round_float(anchor_binding.get("score")),
                        "residual_anchor_tokens": list(anchor_binding.get("anchor_tokens") or []),
                        "matched_residual_anchor_tokens": list(
                            anchor_binding.get("matched_tokens") or []
                        ),
                        "source_doc_id": doc_id,
                        "source_doc_key": doc_key or _doc_key(doc),
                        "source_rank_reader_visible": bool(source_rank_reader_visible),
                        "reader_visible_tuple_projection": bool(
                            reader_visible_tuple_projection
                        ),
                        "validated_schema_tuple_materialization": True,
                        "active_schema_bridge_tuple_executor": bool(
                            active_schema_bridge_tuple
                        ),
                        "active_schema_bridge_tuple_bound_signal_count": int(
                            active_bridge_bound_signal_count
                        ),
                        **terminal_metadata,
                        "compact_projection": True,
                        "compact_table": True,
                        "candidate_source": str(
                            item.get("candidate_source") or "same_schema_memory"
                        ),
                        "cross_schema_join": bool(item.get("cross_schema_join")),
                        "active_schema": list(item.get("active_schema") or []),
                        "triple": list(item.get("triple") or []),
                        "join_direction": str(item.get("direction") or ""),
                        "active_entity": str(item.get("active_entity") or ""),
                        "answer_value": str(item.get("answer_value") or ""),
                        "answer_type": str(item.get("answer_type") or "unknown"),
                        "answer_type_match": bool(item.get("answer_type_match")),
                        "predicate": str(item.get("predicate") or ""),
                        "predicate_contract_match": bool(
                            item.get("predicate_contract_match")
                        ),
                        "predicate_match_mode": str(item.get("predicate_match_mode") or ""),
                        "slot_id": str(item.get("slot_id") or ""),
                        "slot_text": str(item.get("slot_text") or "")[:240],
                        "slot_anchor_overlap": _round_float(
                            item.get("slot_anchor_overlap", 0.0)
                        ),
                        "slot_anchor_tokens": list(item.get("slot_anchor_tokens") or []),
                        "slot_relation_tokens": list(item.get("slot_relation_tokens") or []),
                        "slot_plan_stage": str(
                            item.get("slot_plan_stage") or "unclassified"
                        ),
                        "slot_branch_id": str(item.get("slot_branch_id") or ""),
                        "slot_expected_value_type": str(
                            item.get("slot_expected_value_type") or "unknown"
                        ),
                        "slot_materialization_safe": bool(
                            item.get("slot_materialization_safe", True)
                        ),
                        "materialization_role": str(
                            item.get("materialization_role") or "validated_tuple"
                        ),
                        "branch_binding_passed": bool(item.get("branch_binding_passed")),
                        "branch_binding_reason": str(
                            item.get("branch_binding_reason") or ""
                        ),
                        "expected_relation_signature": str(
                            item.get("expected_relation_signature") or ""
                        ),
                        "candidate_relation_signature": str(
                            item.get("candidate_relation_signature") or ""
                        ),
                        "relation_signature_match": bool(
                            item.get("relation_signature_match")
                        ),
                        "relation_signature_compatible": bool(
                            item.get("relation_signature_compatible")
                        ),
                        "selective_access_eligible": bool(
                            item.get("selective_access_eligible")
                        ),
                        "selective_global_access": bool(
                            item.get("selective_global_access")
                        ),
                        "dependency_binding_safe": bool(
                            item.get("dependency_binding_safe", True)
                        ),
                        "dependency_binding_reasons": list(
                            item.get("dependency_binding_reasons") or []
                        ),
                        "provenance_score": _round_float(
                            item.get("provenance_score", 0.0)
                        ),
                        "provenance_doc_ids": list(item.get("provenance_doc_ids") or []),
                        "lineage_confidence": _round_float(tuple_lineage_confidence),
                        "coverage_gain": _round_float(tuple_coverage_gain),
                        **projection_metadata,
                        **{
                            key: value
                            for key, value in lineage_metadata.items()
                            if key not in {"lineage_confidence", "coverage_gain"}
                        },
                    }
                    tuple_candidates.append({
                        "doc": materialized_doc,
                        "source": "query_schema_memory",
                        "metadata": metadata,
                        "doc_id": doc_id,
                        "title": str(item.get("title") or _title(doc)),
                        "schema_score": max(score, tuple_score_floor),
                        "schema_idf": schema_idf,
                        "query_overlap": query_overlap,
                        "join_key_overlap": join_key_overlap,
                        "relation_overlap": relation_overlap,
                        "relation_relaxed": bool(relation_relaxed),
                        "original_rank": original_rank,
                    })
                    if (
                        projection_metadata.get("source_sentence_value_bound")
                        and projection_metadata.get("source_sentence_join_bound")
                    ):
                        info["validated_tuple_source_sentence_bound_count"] += 1
            if tuple_reasons:
                tuple_reject_counts.update(tuple_reasons)

        if already_reader_visible and not reasons:
            covered_row = {
                "doc_id": doc_id,
                "title": str(item.get("title") or ""),
                "schema_score": score,
                "schema_idf": schema_idf,
                "query_overlap": query_overlap,
                "join_key_overlap": join_key_overlap,
                "relation_overlap": relation_overlap,
                "relation_relaxed": bool(relation_relaxed),
                "cross_schema_join": bool(is_cross_schema),
                "predicate_contract_match": bool(
                    item.get("predicate_contract_match")
                ),
                "estimated_endpoint_scan_rows": endpoint_scan_rows,
                "slot_id": str(item.get("slot_id") or ""),
                "slot_text": str(item.get("slot_text") or "")[:160],
                "residual_anchor_bound": bool(anchor_bound),
                "residual_anchor_score": _round_float(anchor_binding.get("score")),
                "residual_anchor_tokens": list(anchor_binding.get("anchor_tokens") or []),
                "matched_residual_anchor_tokens": list(
                    anchor_binding.get("matched_tokens") or []
                ),
                "original_rank": original_rank,
                "active_schema_bridge_tuple_executor": bool(
                    active_schema_bridge_tuple
                ),
            }
            validated_topk.append(covered_row)
            covered_by_reader_visible.append(covered_row)
            continue
        if already_reader_visible:
            reasons.append("already_reader_visible")
        if reasons:
            reject_counts.update(reasons)
            continue

        if doc is None:
            try:
                doc = str(get_doc_content(doc_id) or "")
            except Exception:
                reject_counts["doc_lookup_failed"] += 1
                continue
        doc_key = _doc_key(doc)
        if not doc_key or doc_key in existing_doc_keys:
            reject_counts["duplicate_or_empty_doc"] += 1
            continue
        if require_answer_type_value:
            if date_requested and not _DATE_VALUE_RE.search(doc):
                reject_counts["missing_date_value"] += 1
                continue
            if number_requested and not _NUMBER_VALUE_RE.search(doc):
                reject_counts["missing_number_value"] += 1
                continue

        lineage_metadata = (
            _entity_predicate_contract_lineage_metadata(item, lineage_context)
            if is_cross_schema
            else _candidate_lineage_metadata(
                doc,
                "query_schema_memory",
                lineage_context,
            )
        )
        if (
            is_cross_schema
            and require_lineage_binding
            and lineage_context.get("required_uncovered_slot_ids")
            and not lineage_metadata.get("uncovered_slot_ids")
        ):
            fallback_lineage = _candidate_lineage_metadata(
                doc,
                "query_schema_memory",
                lineage_context,
            )
            if fallback_lineage.get("uncovered_slot_ids"):
                lineage_metadata = fallback_lineage
        if (
            require_lineage_binding
            and
            lineage_context.get("required_uncovered_slot_ids")
            and not lineage_metadata.get("uncovered_slot_ids")
            and not active_schema_bridge_tuple
        ):
            reject_counts["no_uncovered_slot_binding"] += 1
            continue

        if (
            validated_tuple_require_terminal_slot
            and (validated_tuple_materialization or not compact_projection)
            and tuple_like_candidate
        ):
            terminal_item = dict(item)
            if active_schema_bridge_tuple:
                terminal_item["active_schema_bridge_tuple_executor"] = True
                terminal_item["residual_anchor_bound"] = bool(anchor_bound)
            terminal_ok, terminal_metadata = _schema_tuple_terminal_slot_decision(
                question=question,
                item=terminal_item,
                value=tuple_value,
                lineage_context=lineage_context,
                lineage_metadata=lineage_metadata,
            )
            if not terminal_ok:
                veto_reasons = (
                    terminal_metadata.get("terminal_slot_gate_reasons")
                    or ["non_terminal_tuple"]
                )
                terminal_veto_counts.update(veto_reasons)
                reject_counts["schema_tuple_terminal_veto"] += 1
                continue

        materialized_doc = (
            _compact_schema_bridge_doc(
                doc,
                item,
                question_tokens=question_tokens,
                deficit_tokens=deficit_tokens,
                relation_tokens=relation_tokens,
            )
            if compact_projection
            else doc
        )
        original_doc_token_count = len(re.findall(r"\S+", doc))
        projected_doc_token_count = len(re.findall(r"\S+", materialized_doc))
        metadata = {
            "schema_score": score,
            "schema_idf": schema_idf,
            "query_schema_overlap": query_overlap,
            "join_key_overlap": join_key_overlap,
            "relation_overlap": relation_overlap,
            "relation_relaxed_by_exact_join_key": bool(relation_relaxed),
            "schema_frequency": _as_int(item.get("schema_frequency"), 0),
            "catalog_selectivity": _round_float(
                item.get("catalog_selectivity", 0.0)
            ),
            "endpoint_access_selectivity": _round_float(
                item.get("endpoint_access_selectivity", 0.0)
            ),
            "estimated_endpoint_scan_rows": _as_int(
                item.get("estimated_endpoint_scan_rows"), 0
            ),
            "operator_latency_ms": _round_float(
                item.get("operator_latency_ms", 0.0)
            ),
            "operator_name": str(
                item.get("operator_name")
                or "schema_hash_lookup_with_join_key_probe"
            ),
            "original_rank": original_rank,
            "schema": list(schema) if isinstance(schema, (list, tuple)) else schema,
            "entity_bound": bool(item.get("entity_bound")),
            "join_key_exact": join_key_exact,
            "active_join_role": str(item.get("active_join_role") or ""),
            "candidate_join_role": str(item.get("candidate_join_role") or ""),
            "matched_join_tokens": list(item.get("matched_join_tokens") or []),
            "active_join_key_tokens": list(
                item.get("active_join_key_tokens") or []
            ),
            "candidate_join_key_tokens": list(
                item.get("candidate_join_key_tokens") or []
            ),
            "residual_anchor_bound": bool(anchor_bound),
            "residual_anchor_score": _round_float(anchor_binding.get("score")),
            "residual_anchor_tokens": list(anchor_binding.get("anchor_tokens") or []),
            "matched_residual_anchor_tokens": list(
                anchor_binding.get("matched_tokens") or []
            ),
            "source_doc_id": doc_id,
            "source_doc_key": doc_key,
            "compact_projection": bool(compact_projection),
            "original_doc_token_count": int(original_doc_token_count),
            "projected_doc_token_count": int(projected_doc_token_count),
            "projection_token_savings": max(
                0, original_doc_token_count - projected_doc_token_count
            ),
            "candidate_source": str(
                item.get("candidate_source") or "same_schema_memory"
            ),
            "cross_schema_join": bool(item.get("cross_schema_join")),
            "active_schema": list(item.get("active_schema") or []),
            "triple": list(item.get("triple") or []),
            "join_direction": str(item.get("direction") or ""),
            "active_entity": str(item.get("active_entity") or ""),
            "answer_value": str(item.get("answer_value") or ""),
            "answer_type": str(item.get("answer_type") or "unknown"),
            "answer_type_match": bool(item.get("answer_type_match")),
            "predicate": str(item.get("predicate") or ""),
            "predicate_contract_match": bool(
                item.get("predicate_contract_match")
            ),
            "predicate_match_mode": str(
                item.get("predicate_match_mode") or ""
            ),
            "slot_id": str(item.get("slot_id") or ""),
            "slot_text": str(item.get("slot_text") or "")[:240],
            "slot_anchor_overlap": _round_float(
                item.get("slot_anchor_overlap", 0.0)
            ),
            "slot_anchor_tokens": list(item.get("slot_anchor_tokens") or []),
            "slot_relation_tokens": list(item.get("slot_relation_tokens") or []),
            "slot_plan_stage": str(
                item.get("slot_plan_stage") or "unclassified"
            ),
            "slot_branch_id": str(item.get("slot_branch_id") or ""),
            "slot_expected_value_type": str(
                item.get("slot_expected_value_type") or "unknown"
            ),
            "slot_materialization_safe": bool(
                item.get("slot_materialization_safe", True)
            ),
            "materialization_role": str(
                item.get("materialization_role") or ""
            ),
            "branch_binding_passed": bool(item.get("branch_binding_passed")),
            "branch_binding_reason": str(
                item.get("branch_binding_reason") or ""
            ),
            "expected_relation_signature": str(
                item.get("expected_relation_signature") or ""
            ),
            "candidate_relation_signature": str(
                item.get("candidate_relation_signature") or ""
            ),
            "relation_signature_match": bool(
                item.get("relation_signature_match")
            ),
            "relation_signature_compatible": bool(
                item.get("relation_signature_compatible")
            ),
            "selective_access_eligible": bool(
                item.get("selective_access_eligible")
            ),
            "selective_global_access": bool(item.get("selective_global_access")),
            "dependency_binding_safe": bool(
                item.get("dependency_binding_safe", True)
            ),
            "dependency_binding_reasons": list(
                item.get("dependency_binding_reasons") or []
            ),
            "provenance_score": _round_float(
                item.get("provenance_score", 0.0)
            ),
            "provenance_doc_ids": list(item.get("provenance_doc_ids") or []),
            **lineage_metadata,
        }
        if active_schema_bridge_tuple:
            metadata.update({
                "active_schema_bridge_tuple_executor": True,
                "active_schema_bridge_tuple_bound_signal_count": int(
                    active_bridge_bound_signal_count
                ),
            })
        if relation_relaxed:
            relation_relaxed_count += 1
        candidates.append({
            "doc": materialized_doc,
            "source": "query_schema_memory",
            "metadata": metadata,
            "doc_id": doc_id,
            "title": str(item.get("title") or _title(doc)),
            "schema_score": score,
            "schema_idf": schema_idf,
            "query_overlap": query_overlap,
            "join_key_overlap": join_key_overlap,
            "relation_overlap": relation_overlap,
            "relation_relaxed": bool(relation_relaxed),
            "original_rank": original_rank,
        })

    candidates.extend(tuple_candidates)
    candidates.sort(
        key=lambda row: (
            -float(row.get("schema_score", 0.0)),
            -int(
                bool(
                    (row.get("metadata") or {}).get(
                        "validated_schema_tuple_materialization"
                    )
                )
            ),
            -float(row.get("relation_overlap", 0.0)),
            -float(row.get("schema_idf", 0.0)),
            int(row.get("original_rank", 10**9)),
        )
    )
    selected = candidates[: max(0, int(max_candidates))]
    selected_metadata = [dict(row.get("metadata") or {}) for row in selected]
    selected_tuple_metadata = [
        metadata
        for metadata in selected_metadata
        if metadata.get("validated_schema_tuple_materialization")
    ]
    tuple_candidate_metadata = [
        dict(row.get("metadata") or {})
        for row in tuple_candidates
        if (row.get("metadata") or {}).get("validated_schema_tuple_materialization")
    ]
    info.update({
        "reason": (
            "validated_tuple_materialized"
            if selected_tuple_metadata
            else "candidates_materialized"
            if selected
            else "no_candidate_accepted"
        ),
        "candidate_count": len(candidates),
        "accepted_count": len(selected),
        "rejected_count": max(0, int(info["raw_candidate_count"]) - len(selected)),
        "reject_reason_counts": dict(reject_counts),
        "accepted_doc_ids": [int(row["doc_id"]) for row in selected],
        "relation_relaxed_count": sum(
            bool(metadata.get("relation_relaxed_by_exact_join_key"))
            for metadata in selected_metadata
        ),
        "relation_relaxed_candidate_count": int(relation_relaxed_count),
        "validated_topk_count": len(validated_topk),
        "validated_topk_doc_ids": [int(row["doc_id"]) for row in validated_topk[:20]],
        "validated_topk_candidates": validated_topk[:5],
        "residual_anchor_bound_candidate_count": sum(
            bool((row.get("metadata") or {}).get("residual_anchor_bound"))
            for row in candidates
        ),
        "deep_rank_candidate_count": sum(
            int(row.get("original_rank", 0) or 0) > int(safe_max_original_rank)
            for row in candidates
        ),
        "projected_candidate_count": sum(
            bool(metadata.get("compact_projection"))
            for metadata in selected_metadata
        ),
        "original_doc_token_count": sum(
            int(metadata.get("original_doc_token_count", 0) or 0)
            for metadata in selected_metadata
        ),
        "projected_doc_token_count": sum(
            int(metadata.get("projected_doc_token_count", 0) or 0)
            for metadata in selected_metadata
        ),
        "projection_token_savings": sum(
            int(metadata.get("projection_token_savings", 0) or 0)
            for metadata in selected_metadata
        ),
        "validated_tuple_candidate_count": len(tuple_candidates),
        "validated_tuple_accepted_count": len(selected_tuple_metadata),
        "validated_tuple_rejected_count": sum(tuple_reject_counts.values()),
        "validated_tuple_reject_reason_counts": dict(tuple_reject_counts),
        "validated_tuple_terminal_candidate_count": int(
            info.get("validated_tuple_terminal_candidate_count", 0) or 0
        ),
        "validated_tuple_bridge_only_candidate_count": int(
            info.get("validated_tuple_bridge_only_candidate_count", 0) or 0
        ),
        "validated_tuple_source_sentence_bound_count": int(
            info.get("validated_tuple_source_sentence_bound_count", 0) or 0
        ),
        "validated_tuple_source_sentence_trace_fallback_count": int(
            info.get("validated_tuple_source_sentence_trace_fallback_count", 0)
            or 0
        ),
        "validated_tuple_source_visible_reject_count": int(
            tuple_reject_counts.get("source_already_reader_visible", 0)
        ),
        "validated_tuple_reader_visible_projection_count": sum(
            bool(metadata.get("reader_visible_tuple_projection"))
            for metadata in selected_tuple_metadata
        ),
        "schema_tuple_terminal_veto_count": int(sum(terminal_veto_counts.values())),
        "schema_tuple_terminal_veto_reason_counts": dict(terminal_veto_counts),
        "validated_tuple_doc_ids": [
            int(metadata.get("source_doc_id", 0) or 0)
            for metadata in selected_tuple_metadata
        ],
        "validated_tuple_projection_token_savings": sum(
            int(metadata.get("projection_token_savings", 0) or 0)
            for metadata in selected_tuple_metadata
        ),
        "validated_tuple_original_doc_token_count": sum(
            int(metadata.get("original_doc_token_count", 0) or 0)
            for metadata in selected_tuple_metadata
        ),
        "validated_tuple_projected_doc_token_count": sum(
            int(metadata.get("projected_doc_token_count", 0) or 0)
            for metadata in selected_tuple_metadata
        ),
        "validated_tuple_top_candidates": [
            {
                "doc_id": int(metadata.get("source_doc_id", 0) or 0),
                "schema_score": _round_float(metadata.get("schema_score")),
                "relation_overlap": _round_float(metadata.get("relation_overlap")),
                "join_key_overlap": _round_float(metadata.get("join_key_overlap")),
                "cross_schema_join": bool(metadata.get("cross_schema_join")),
                "predicate_contract_match": bool(
                    metadata.get("predicate_contract_match")
                ),
                "residual_anchor_bound": bool(
                    metadata.get("residual_anchor_bound")
                ),
                "slot_id": str(metadata.get("slot_id") or ""),
                "answer_value": str(metadata.get("answer_value") or "")[:120],
                "terminal_slot_gate_passed": bool(
                    metadata.get("terminal_slot_gate_passed")
                ),
                "terminal_answer_slot": str(
                    metadata.get("terminal_answer_slot") or ""
                ),
                "terminal_value_type": str(
                    metadata.get("terminal_value_type") or ""
                ),
                "terminal_relation_intersection": list(
                    metadata.get("terminal_relation_intersection") or []
                ),
                "source_sentence_value_bound": bool(
                    metadata.get("source_sentence_value_bound")
                ),
                "source_sentence_join_bound": bool(
                    metadata.get("source_sentence_join_bound")
                ),
                "source_sentence_relation_bound": bool(
                    metadata.get("source_sentence_relation_bound")
                ),
                "source_sentence_trace_fallback": bool(
                    metadata.get("source_sentence_trace_fallback")
                ),
                "active_schema_bridge_tuple_executor": bool(
                    metadata.get("active_schema_bridge_tuple_executor")
                ),
                "projection_token_savings": int(
                    metadata.get("projection_token_savings", 0) or 0
                ),
            }
            for metadata in tuple_candidate_metadata[:5]
        ],
        "cross_schema_candidate_count": sum(
            bool((row.get("metadata") or {}).get("cross_schema_join"))
            for row in candidates
        ),
        "cross_schema_accepted_count": sum(
            bool(metadata.get("cross_schema_join"))
            for metadata in selected_metadata
        ),
        "cross_schema_slot_bound_count": sum(
            bool((row.get("metadata") or {}).get("cross_schema_join"))
            and bool(
                (row.get("metadata") or {}).get("uncovered_slot_ids")
                or (row.get("metadata") or {}).get("residual_anchor_bound")
                or (row.get("metadata") or {}).get("slot_id")
            )
            for row in candidates
        ),
        "cross_schema_predicate_contract_count": sum(
            bool((row.get("metadata") or {}).get("cross_schema_join"))
            and bool(
                (row.get("metadata") or {}).get("predicate_contract_match")
            )
            for row in candidates
        ),
        "cross_schema_validated_topk_count": sum(
            bool(row.get("cross_schema_join")) for row in validated_topk
        ),
        "covered_by_reader_visible_count": len(covered_by_reader_visible),
        "covered_by_reader_visible_doc_ids": [
            int(row["doc_id"]) for row in covered_by_reader_visible[:20]
        ],
        "covered_by_reader_visible_candidates": covered_by_reader_visible[:5],
        "active_schema_bridge_tuple_accepted_count": sum(
            bool(metadata.get("active_schema_bridge_tuple_executor"))
            for metadata in selected_metadata
        ),
        "active_schema_bridge_tuple_relaxed_count": int(
            sum(active_bridge_relaxed_counts.values())
        ),
        "active_schema_bridge_tuple_relaxed_reason_counts": dict(
            active_bridge_relaxed_counts
        ),
        "top_candidates": [
            {
                key: row[key]
                for key in (
                    "doc_id",
                    "title",
                    "schema_score",
                    "schema_idf",
                    "query_overlap",
                    "join_key_overlap",
                    "relation_overlap",
                    "relation_relaxed",
                    "original_rank",
                )
            }
            | {
                "residual_anchor_bound": bool(
                    (row.get("metadata") or {}).get("residual_anchor_bound")
                ),
                "residual_anchor_score": _round_float(
                    (row.get("metadata") or {}).get("residual_anchor_score")
                ),
                "residual_anchor_tokens": list(
                    (row.get("metadata") or {}).get("residual_anchor_tokens") or []
                ),
                "matched_residual_anchor_tokens": list(
                    (row.get("metadata") or {}).get(
                        "matched_residual_anchor_tokens"
                    )
                    or []
                ),
                "compact_projection": bool(
                    (row.get("metadata") or {}).get("compact_projection")
                ),
                "original_doc_token_count": int(
                    (row.get("metadata") or {}).get(
                        "original_doc_token_count", 0
                    )
                    or 0
                ),
                "projected_doc_token_count": int(
                    (row.get("metadata") or {}).get(
                        "projected_doc_token_count", 0
                    )
                    or 0
                ),
                "projection_token_savings": int(
                    (row.get("metadata") or {}).get(
                        "projection_token_savings", 0
                    )
                    or 0
                ),
                "candidate_source": str(
                    (row.get("metadata") or {}).get("candidate_source") or ""
                ),
                "cross_schema_join": bool(
                    (row.get("metadata") or {}).get("cross_schema_join")
                ),
                "triple": list(
                    (row.get("metadata") or {}).get("triple") or []
                ),
                "answer_value": str(
                    (row.get("metadata") or {}).get("answer_value") or ""
                ),
                "answer_type": str(
                    (row.get("metadata") or {}).get("answer_type") or "unknown"
                ),
                "slot_id": str(
                    (row.get("metadata") or {}).get("slot_id") or ""
                ),
                "entity_predicate_target_slot_id": str(
                    (row.get("metadata") or {}).get(
                        "entity_predicate_target_slot_id"
                    )
                    or ""
                ),
                "slot_plan_stage": str(
                    (row.get("metadata") or {}).get("slot_plan_stage")
                    or "unclassified"
                ),
                "predicate_contract_match": bool(
                    (row.get("metadata") or {}).get("predicate_contract_match")
                ),
                "validated_schema_tuple_materialization": bool(
                    (row.get("metadata") or {}).get(
                        "validated_schema_tuple_materialization"
                    )
                ),
                "active_schema_bridge_tuple_executor": bool(
                    (row.get("metadata") or {}).get(
                        "active_schema_bridge_tuple_executor"
                    )
                ),
                "active_schema_bridge_tuple_bound_signal_count": int(
                    (row.get("metadata") or {}).get(
                        "active_schema_bridge_tuple_bound_signal_count", 0
                    )
                    or 0
                ),
                "estimated_endpoint_scan_rows": int(
                    (row.get("metadata") or {}).get(
                        "estimated_endpoint_scan_rows", 0
                    )
                    or 0
                ),
            }
            for row in selected[:5]
        ],
    })
    return selected, info


def _format_entity_predicate_fact_projection(
    item: Dict[str, Any],
    source_doc: str,
) -> Tuple[str, Dict[str, Any]]:
    triple = list(item.get("triple") or [])
    subject = str(triple[0] if len(triple) > 0 else item.get("active_entity") or "").strip()
    predicate = str(triple[1] if len(triple) > 1 else item.get("predicate") or "").strip()
    value = str(triple[2] if len(triple) > 2 else item.get("answer_value") or "").strip()
    answer_surface = str(
        item.get("answer_surface") or item.get("answer_value") or value
    ).strip()
    title = str(item.get("title") or _title(source_doc) or subject).strip()
    if not subject or not predicate or not value:
        return source_doc, {
            "compact_projection": False,
            "projection_reason": "incomplete_fact_tuple",
        }

    lines = [title, f"Provenance-grounded fact: {subject} {predicate} {value}."]
    if bool(item.get("terminal_tuple_eligible")) and answer_surface:
        lines.extend([
            f"Question slot: {str(item.get('slot_text') or 'final answer').strip()}.",
            f"Answer-slot value: {answer_surface}.",
        ])
    lineage_path = [list(edge) for edge in item.get("lineage_path") or []]
    if lineage_path:
        lineage_surface = " ; ".join(
            f"{str(edge[0]).strip()} --{str(edge[1]).strip()}--> {str(edge[2]).strip()}"
            for edge in lineage_path[:2]
            if len(edge) >= 3
        )
        if lineage_surface:
            lines.append(f"Join lineage: {lineage_surface}.")
    projected = "\n".join(lines)
    original_tokens = max(1, len(re.findall(r"\S+", str(source_doc or ""))))
    projected_tokens = max(1, len(re.findall(r"\S+", projected)))
    return projected, {
        "compact_projection": True,
        "projection_reason": (
            "terminal_provenance_answer_tuple"
            if bool(item.get("terminal_tuple_eligible"))
            else "entity_predicate_tuple_projection"
        ),
        "source_passage_title": _title(source_doc),
        "original_doc_token_count": original_tokens,
        "projected_doc_token_count": projected_tokens,
        "projection_token_savings": max(0, original_tokens - projected_tokens),
    }


def _project_entity_predicate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    if str(record.get("source") or "") != "entity_predicate_fact_join":
        return record
    metadata = dict(record.get("metadata") or {})
    if bool(metadata.get("compact_projection")):
        return record
    triple_path = list(metadata.get("triple_path") or [])
    projection_item = {
        **metadata,
        "title": str(record.get("title") or _title(record.get("doc"))),
        "triple": list(triple_path[0]) if triple_path else [],
    }
    projected_doc, projection_metadata = _format_entity_predicate_fact_projection(
        projection_item,
        str(record.get("doc") or ""),
    )
    projected = dict(record)
    projected["doc"] = projected_doc
    projected["metadata"] = {
        **metadata,
        **projection_metadata,
        "compact_table": bool(projection_metadata.get("compact_projection")),
    }
    return projected


def _collect_entity_predicate_fact_join_candidates(
    *,
    entity_predicate_fact_join: Dict[str, Any],
    get_doc_content: Optional[Callable[[int], str]],
    existing_doc_keys: set,
    state: Dict[str, Any],
    lineage_context: Dict[str, Any],
    enabled: bool,
    max_candidates: int,
    min_score: float,
    min_relation_overlap: float,
    min_lineage_confidence: float,
    require_atomic_slot: bool,
    allowed_match_modes: Any,
    compact_projection: bool = False,
    terminal_tuple_materialization: bool = False,
    terminal_tuple_min_lineage_depth: int = 1,
    terminal_tuple_strict_branch_binding: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Materialize exact entity-predicate joins using per-candidate guards."""
    raw_items = list((entity_predicate_fact_join or {}).get("candidate_items") or [])
    require_validated_completion = bool(
        (entity_predicate_fact_join or {}).get("validated_completion_enabled")
    )
    require_plan_stage_binding = bool(
        (entity_predicate_fact_join or {}).get("plan_stage_gate_enabled")
    )
    selected_operator = str(
        (entity_predicate_fact_join or {}).get("selected_operator") or ""
    )
    if isinstance(allowed_match_modes, str):
        allowed_modes = {
            value.strip()
            for value in allowed_match_modes.split(",")
            if value.strip()
        }
    else:
        allowed_modes = {
            str(value).strip()
            for value in (allowed_match_modes or [])
            if str(value).strip()
        }
    if not allowed_modes:
        allowed_modes = {"direct_token", "polarity_concept"}
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "reason": "disabled" if not enabled else "not_evaluated",
        "raw_candidate_count": len(raw_items),
        "candidate_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "reject_reason_counts": {},
        "accepted_doc_ids": [],
        "top_candidates": [],
        "candidate_level_admission": True,
        "rank_stable": True,
        "require_atomic_slot": bool(require_atomic_slot),
        "allowed_match_modes": sorted(allowed_modes),
        "compact_projection_enabled": bool(compact_projection),
        "terminal_tuple_materialization_enabled": bool(
            terminal_tuple_materialization
        ),
        "terminal_tuple_min_lineage_depth": max(
            0, int(terminal_tuple_min_lineage_depth)
        ),
        "terminal_tuple_strict_branch_binding": bool(
            terminal_tuple_strict_branch_binding
        ),
        "terminal_tuple_candidate_count": 0,
        "terminal_tuple_accepted_count": 0,
        "projected_candidate_count": 0,
        "projection_token_savings": 0,
        "validated_completion_required": require_validated_completion,
        "plan_stage_binding_required": require_plan_stage_binding,
        "final_answer_type": str(
            (entity_predicate_fact_join or {}).get("final_answer_type")
            or lineage_context.get("answer_slot")
            or "unknown"
        ),
        "final_binding_candidate_count": int(
            (entity_predicate_fact_join or {}).get(
                "operator_final_binding_count",
                0,
            )
            or 0
        ),
        "intermediate_binding_candidate_count": int(
            (entity_predicate_fact_join or {}).get(
                "operator_intermediate_binding_count",
                0,
            )
            or 0
        ),
        "support_only_candidate_count": int(
            (entity_predicate_fact_join or {}).get(
                "operator_support_only_count",
                0,
            )
            or 0
        ),
        "operator_requires_followup": bool(
            (entity_predicate_fact_join or {}).get("operator_requires_followup")
        ),
        "operator_complete": bool(
            (entity_predicate_fact_join or {}).get("operator_complete")
        ),
        "validated_candidate_count": sum(
            bool(item.get("operator_validation_passed")) for item in raw_items
        ),
    }
    if not enabled:
        return [], info
    if not entity_predicate_fact_join or not get_doc_content:
        info["reason"] = "no_entity_predicate_join_or_doc_lookup"
        return [], info
    if bool(state.get("long_context")):
        info["reason"] = "long_context_blocked"
        return [], info
    if require_validated_completion and not bool(
        entity_predicate_fact_join.get("operator_complete")
    ):
        info["reason"] = (
            "operator_requires_followup"
            if bool(entity_predicate_fact_join.get("operator_requires_followup"))
            else "operator_validation_incomplete"
        )
        return [], info
    if not bool(state.get("evidence_insufficient")) and not lineage_context.get(
        "required_uncovered_slot_ids"
    ):
        info["reason"] = "current_evidence_sufficient"
        return [], info

    reject_counts: Counter = Counter()
    candidates: List[Dict[str, Any]] = []
    required_uncovered = set(lineage_context.get("required_uncovered_slot_ids") or [])
    for item in raw_items:
        reasons: List[str] = []
        candidate_kind = str(item.get("operator_candidate_kind") or "")
        dense_support = candidate_kind == "dense_text_retrieval"
        slot_plan_stage = str(
            item.get("slot_plan_stage") or "unclassified"
        )
        try:
            doc_id = int(item.get("doc_id"))
        except (TypeError, ValueError):
            reject_counts["invalid_doc_id"] += 1
            continue
        score = (
            _as_float(item.get("operator_validation_score"), 0.0)
            if dense_support and require_validated_completion
            else _as_float(item.get("score"), 0.0)
        )
        relation_overlap = _as_float(item.get("relation_overlap"), 0.0)
        terminal_tuple_candidate = bool(
            terminal_tuple_materialization
            and not dense_support
            and selected_operator not in {"compare", "multi_key_fact_join"}
            and slot_plan_stage == "final_answer"
        )
        if terminal_tuple_candidate:
            info["terminal_tuple_candidate_count"] += 1
            if _as_int(item.get("lineage_depth"), 0) < max(
                0, int(terminal_tuple_min_lineage_depth)
            ):
                reasons.append("terminal_tuple_lineage_too_shallow")
            if (
                terminal_tuple_strict_branch_binding
                and not bool(item.get("branch_binding_passed"))
            ):
                reasons.append("terminal_tuple_branch_identity_unbound")
            if not bool(item.get("relation_signature_match")):
                reasons.append("terminal_tuple_signature_not_exact")
            if not bool(item.get("answer_surface_recovered")):
                reasons.append("terminal_tuple_surface_not_recovered")
        elif (
            terminal_tuple_materialization
            and not dense_support
            and selected_operator not in {"compare", "multi_key_fact_join"}
        ):
            reasons.append("not_terminal_answer_tuple")
        if require_validated_completion and not bool(
            item.get("operator_validation_passed")
        ):
            reasons.append("operator_validation_failed")
        if require_plan_stage_binding:
            if dense_support and not bool(item.get("branch_binding_passed")):
                reasons.append("dense_branch_identity_unbound")
            if (
                selected_operator not in {"compare", "multi_key_fact_join"}
                and slot_plan_stage != "final_answer"
            ):
                reasons.append(f"non_final_plan_stage:{slot_plan_stage}")
        if not dense_support:
            if not bool(item.get("entity_bound")) or not bool(item.get("join_key_exact")):
                reasons.append("entity_binding_not_exact")
            if not bool(item.get("lineage_bound")):
                reasons.append("query_root_lineage_unbound")
            if item.get("dependency_binding_safe") is False:
                reasons.append("dependency_binding_unsafe")
        if not bool(item.get("predicate_contract_match")):
            reasons.append("predicate_contract_mismatch")
        match_mode = str(item.get("predicate_match_mode") or "")
        if match_mode not in allowed_modes:
            reasons.append(f"predicate_match_mode_blocked:{match_mode or 'unknown'}")
        if require_atomic_slot and not bool(item.get("slot_materialization_safe")):
            reasons.append("non_atomic_deficit_slot")
            reasons.extend(
                f"slot_compiler:{reason}"
                for reason in item.get("slot_unsafe_reasons") or []
            )
        if match_mode == "inverse_role" and (
            not bool(item.get("answer_type_match", True))
            or str(item.get("direction") or "")
            not in {"subject_to_object", "object_to_subject"}
        ):
            reasons.append("inverse_role_direction_or_type_unsafe")
        if bool(item.get("already_top5")) and not terminal_tuple_candidate:
            reasons.append("already_reader_visible")
        if score < float(min_score):
            reasons.append("entity_predicate_score_low")
        if relation_overlap < float(min_relation_overlap):
            reasons.append("entity_predicate_relation_low")
        if not bool(item.get("answer_type_match", True)):
            reasons.append("answer_type_mismatch")
        if reasons:
            reject_counts.update(reasons)
            continue

        try:
            doc = str(get_doc_content(doc_id) or "")
        except Exception:
            reject_counts["doc_lookup_failed"] += 1
            continue
        doc_key = _doc_key(doc)
        if not doc_key or (
            doc_key in existing_doc_keys and not terminal_tuple_candidate
        ):
            reject_counts["duplicate_or_empty_doc"] += 1
            continue
        materialization_item = {
            **item,
            "terminal_tuple_eligible": bool(terminal_tuple_candidate),
        }
        lineage_metadata = _entity_predicate_contract_lineage_metadata(
            item,
            lineage_context,
        )
        if required_uncovered and not lineage_metadata.get("uncovered_slot_ids"):
            reject_counts["no_uncovered_slot_binding"] += 1
            continue
        if (
            required_uncovered
            and _as_float(lineage_metadata.get("lineage_confidence"), 0.0)
            < float(min_lineage_confidence)
        ):
            reject_counts["lineage_confidence_low"] += 1
            continue

        metadata = {
            "entity_predicate_score": score,
            "plan_utility": score,
            "plan_net_utility": _as_float(item.get("net_utility"), 0.0),
            "plan_risk": _as_float(item.get("risk"), 0.0),
            "join_key_overlap": _as_float(item.get("join_key_overlap"), 1.0),
            "relation_overlap": relation_overlap,
            "provenance_score": _as_float(item.get("provenance_score"), 0.0),
            "original_rank": _as_int(item.get("original_rank"), 0),
            "entity_bound": bool(item.get("entity_bound")),
            "join_key_exact": bool(item.get("join_key_exact")),
            "direct_slot_grounded": bool(
                dense_support and item.get("operator_validation_passed")
            ),
            "operator_candidate_kind": candidate_kind,
            "operator_validation_passed": bool(
                item.get("operator_validation_passed")
            ),
            "operator_validation_score": _as_float(
                item.get("operator_validation_score"),
                0.0,
            ),
            "operator_validation_reason": str(
                item.get("operator_validation_reason") or ""
            ),
            "operator_validation_grounding_sentence": str(
                item.get("operator_validation_grounding_sentence") or ""
            )[:500],
            "active_entity": str(item.get("active_entity") or ""),
            "root_entity": str(item.get("root_entity") or ""),
            "root_anchor_score": _as_float(item.get("root_anchor_score"), 0.0),
            "lineage_depth": _as_int(item.get("lineage_depth"), 0),
            "lineage_path": list(item.get("lineage_path") or [])[:3],
            "active_join_role": str(item.get("active_join_role") or ""),
            "candidate_join_role": str(item.get("candidate_join_role") or ""),
            "answer_role": str(item.get("answer_role") or ""),
            "answer_value": str(item.get("answer_value") or ""),
            "answer_surface": str(
                item.get("answer_surface") or item.get("answer_value") or ""
            ),
            "answer_surface_recovered": bool(
                item.get("answer_surface_recovered")
            ),
            "answer_type": str(item.get("answer_type") or ""),
            "answer_type_match": bool(item.get("answer_type_match", True)),
            "predicate": str(item.get("predicate") or ""),
            "predicate_contract_match": bool(item.get("predicate_contract_match")),
            "predicate_match_mode": str(item.get("predicate_match_mode") or ""),
            "direct_predicate_overlap": _as_float(
                item.get("direct_predicate_overlap"),
                0.0,
            ),
            "direct_predicate_tokens": list(
                item.get("direct_predicate_tokens") or []
            ),
            "discriminative_predicate_tokens": list(
                item.get("discriminative_predicate_tokens") or []
            ),
            "expected_discriminative_predicate_tokens": list(
                item.get("expected_discriminative_predicate_tokens") or []
            ),
            "dependency_binding_safe": bool(
                item.get("dependency_binding_safe", True)
            ),
            "dependency_binding_reasons": list(
                item.get("dependency_binding_reasons") or []
            ),
            "dependency_root_entity_key": str(
                item.get("dependency_root_entity_key") or ""
            ),
            "dependency_terminal_entity_key": str(
                item.get("dependency_terminal_entity_key") or ""
            ),
            "dependency_path_entity_keys": list(
                item.get("dependency_path_entity_keys") or []
            ),
            "dependency_hops": _as_int(
                item.get("dependency_hops"),
                item.get("lineage_depth", 0),
            ),
            "slot_source": str(item.get("slot_source") or "unknown"),
            "slot_hop_index": _as_int(item.get("slot_hop_index"), -1),
            "slot_target_predicate_concepts": list(
                item.get("slot_target_predicate_concepts") or []
            ),
            "slot_expected_value_type": str(
                item.get("slot_expected_value_type") or "unknown"
            ),
            "slot_direction": str(
                item.get("slot_direction") or "bound_entity_to_value"
            ),
            "slot_materialization_safe": bool(
                item.get("slot_materialization_safe")
            ),
            "slot_plan_stage": slot_plan_stage,
            "slot_branch_id": str(item.get("slot_branch_id") or ""),
            "slot_final_answer_type": str(
                item.get("slot_final_answer_type") or "unknown"
            ),
            "slot_expected_type_family": str(
                item.get("slot_expected_type_family") or "unknown"
            ),
            "branch_binding_passed": bool(
                item.get("branch_binding_passed")
            ),
            "branch_binding_reason": str(
                item.get("branch_binding_reason") or ""
            ),
            "branch_anchor_phrase": str(
                item.get("branch_anchor_phrase") or ""
            ),
            "materialization_role": str(
                item.get("materialization_role")
                or ("support_only" if dense_support else "answer_binding")
            ),
            "terminal_tuple_eligible": bool(terminal_tuple_candidate),
            "slot_unsafe_reasons": list(item.get("slot_unsafe_reasons") or []),
            "slot_meta_negative": bool(item.get("slot_meta_negative")),
            "entity_predicate_slot_id": str(item.get("slot_id") or ""),
            "entity_predicate_slot_text": str(item.get("slot_text") or ""),
            "direction": str(item.get("direction") or ""),
            "cross_schema": bool(item.get("cross_schema")),
            "retrieval_pool_member": bool(item.get("retrieval_pool_member", True)),
            "triple_path": [list(item.get("triple") or [])],
            **lineage_metadata,
        }
        if dense_support:
            metadata.update({
                "lineage_bound": bool(item.get("operator_validation_passed")),
                "join_key_overlap": 1.0,
                "join_key_exact": True,
                "answer_type_match": True,
                "predicate_contract_match": bool(
                    item.get("operator_validation_passed")
                ),
            })
        materialized_doc = doc
        if compact_projection:
            materialized_doc, projection_metadata = _format_entity_predicate_fact_projection(
                materialization_item,
                doc,
            )
            metadata.update(projection_metadata)
            metadata["compact_table"] = bool(
                projection_metadata.get("compact_projection")
            )
            if projection_metadata.get("compact_projection"):
                info["projected_candidate_count"] += 1
                info["projection_token_savings"] += int(
                    projection_metadata.get("projection_token_savings", 0) or 0
                )
        candidates.append(
            {
                "doc": materialized_doc,
                "source": "entity_predicate_fact_join",
                "metadata": metadata,
                "doc_id": doc_id,
                "title": str(item.get("title") or _title(doc)),
                "entity_predicate_score": score,
                "relation_overlap": relation_overlap,
                "original_rank": _as_int(item.get("original_rank"), 0),
                "retrieval_pool_member": bool(
                    item.get("retrieval_pool_member", True)
                ),
            }
        )

    candidates.sort(
        key=lambda row: (
            -float(row.get("entity_predicate_score", 0.0)),
            -float(row.get("relation_overlap", 0.0)),
            int(row.get("original_rank", 10**9)),
        )
    )
    required_bindings = int(
        (entity_predicate_fact_join or {}).get(
            "operator_required_binding_count",
            1,
        )
        or 1
    )
    if (
        require_validated_completion
        and selected_operator in {"compare", "multi_key_fact_join"}
        and len(candidates) >= required_bindings
    ):
        grouped = []
        seen_bindings = set()
        for row in candidates:
            metadata = dict(row.get("metadata") or {})
            binding = (
                str(metadata.get("root_entity") or "")
                if selected_operator == "compare"
                else str(metadata.get("active_entity") or "")
            )
            if not binding or binding in seen_bindings:
                continue
            seen_bindings.add(binding)
            grouped.append(row)
            if len(grouped) >= required_bindings:
                break
        if len(grouped) >= required_bindings:
            operator_label = (
                "comparison" if selected_operator == "compare" else "shared-key join"
            )
            table_lines = [
                f"Validated {operator_label} evidence",
                "branch | predicate | value | provenance",
            ]
            source_doc_ids = []
            for row in grouped:
                metadata = dict(row.get("metadata") or {})
                source_doc_ids.append(int(row.get("doc_id")))
                table_lines.append(
                    " | ".join([
                        str(metadata.get("root_entity") or metadata.get("active_entity") or "unknown"),
                        str(metadata.get("predicate") or "related_to"),
                        str(metadata.get("answer_value") or "unknown"),
                        str(row.get("title") or "source passage"),
                    ])
                )
            first = dict(grouped[0])
            first_metadata = dict(first.get("metadata") or {})
            first_metadata.update({
                "compact_projection": True,
                "compact_table": True,
                "projection_reason": f"validated_{selected_operator}_table",
                "physical_operator": str(
                    entity_predicate_fact_join.get("physical_operator") or ""
                ),
                "operator_complete": True,
                "operator_required_binding_count": required_bindings,
                "operator_bound_binding_count": len(grouped),
                "source_doc_ids": source_doc_ids,
                "validated_operator_table": True,
            })
            first.update({
                "doc": "\n".join(table_lines),
                "title": table_lines[0],
                "metadata": first_metadata,
            })
            selected = [first]
        else:
            selected = []
            reject_counts["validated_operator_bindings_incomplete"] += 1
    else:
        selected = candidates[: max(0, int(max_candidates))]
    info.update(
        {
            "reason": "candidates_materialized" if selected else "no_candidate_accepted",
            "candidate_count": len(candidates),
            "accepted_count": len(selected),
            "terminal_tuple_accepted_count": sum(
                bool((row.get("metadata") or {}).get("terminal_tuple_eligible"))
                for row in selected
            ),
            "rejected_count": max(0, len(raw_items) - len(selected)),
            "reject_reason_counts": dict(reject_counts),
            "accepted_doc_ids": [int(row["doc_id"]) for row in selected],
            "top_candidates": [
                {
                    key: row[key]
                    for key in (
                        "doc_id",
                        "title",
                        "entity_predicate_score",
                        "relation_overlap",
                        "original_rank",
                        "retrieval_pool_member",
                    )
                }
                | {
                    "root_entity": str((row.get("metadata") or {}).get("root_entity") or ""),
                    "slot_text": str(
                        (row.get("metadata") or {}).get("entity_predicate_slot_text")
                        or ""
                    ),
                    "predicate_match_mode": str(
                        (row.get("metadata") or {}).get("predicate_match_mode")
                        or ""
                    ),
                    "target_slot_id": str(
                        (row.get("metadata") or {}).get(
                            "entity_predicate_target_slot_id"
                        )
                        or ""
                    ),
                    "slot_plan_stage": str(
                        (row.get("metadata") or {}).get("slot_plan_stage")
                        or "unclassified"
                    ),
                    "slot_branch_id": str(
                        (row.get("metadata") or {}).get("slot_branch_id")
                        or ""
                    ),
                    "branch_binding_passed": bool(
                        (row.get("metadata") or {}).get(
                            "branch_binding_passed"
                        )
                    ),
                    "materialization_role": str(
                        (row.get("metadata") or {}).get("materialization_role")
                        or "unknown"
                    ),
                }
                for row in selected[:5]
            ],
        }
    )
    return selected, info


def _collect_typed_fact_join_shadow_candidates(
    *,
    question: str,
    typed_fact_join_shadow: Dict[str, Any],
    get_doc_content: Optional[Callable[[int], str]],
    existing_doc_keys: set,
    state: Optional[Dict[str, Any]],
    lineage_context: Dict[str, Any],
    require_lineage_binding: bool,
    bridge_cues: bool,
    max_candidates: int,
    max_original_rank: int,
    min_plan_utility: float,
    min_plan_net_utility: float,
    max_plan_risk: float,
    require_two_hop_for_bridge: bool,
    calibrated_shadow_admission: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "enabled": bool(typed_fact_join_shadow and get_doc_content),
        "candidate_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "reason": "not_evaluated",
        "reject_reason_counts": {},
        "top_candidates": [],
    }
    if not typed_fact_join_shadow:
        info["reason"] = "no_shadow_info"
        return [], info
    if get_doc_content is None:
        info["reason"] = "no_doc_loader"
        return [], info
    plans = list(typed_fact_join_shadow.get("top_plans") or [])
    if not plans:
        info["reason"] = "no_shadow_plans"
        return [], info

    max_candidates = max(0, int(max_candidates))
    if max_candidates <= 0:
        info["reason"] = "max_candidates_zero"
        return [], info

    reject_counts: Counter = Counter()
    candidates: List[Dict[str, Any]] = []
    seen = set(existing_doc_keys or set())
    for plan_idx, plan in enumerate(plans):
        plan_type = str(plan.get("plan_type") or "")
        plan_utility = _as_float(plan.get("utility"), 0.0)
        plan_net = _as_float(plan.get("net_utility"), 0.0)
        plan_risk = _as_float(plan.get("risk"), 1.0)
        endpoint_overlap = _as_float(plan.get("endpoint_overlap"), 0.0)
        relation_overlap = _as_float(plan.get("relation_overlap"), 0.0)
        schema_selectivity = _as_float(plan.get("schema_selectivity"), 0.0)
        provenance_score = _as_float(plan.get("provenance_score"), 0.0)
        best_rank = _as_int(plan.get("best_rank"), 999999)
        worst_rank = _as_int(plan.get("worst_rank"), best_rank)
        accepted = bool(plan.get("accepted", False))

        high_conf_one_hop = (
            plan_type == "one_hop_fact_access"
            and plan_utility >= max(float(min_plan_utility) + 0.16, 0.64)
            and plan_net >= max(float(min_plan_net_utility) + 0.18, 0.24)
            and plan_risk <= min(float(max_plan_risk) - 0.12, 0.42)
            and endpoint_overlap >= 0.34
        )
        if not accepted:
            reject_counts["shadow_plan_not_accepted"] += 1
            continue
        if bridge_cues and require_two_hop_for_bridge and plan_type != "two_hop_fact_join" and not high_conf_one_hop:
            reject_counts["bridge_requires_two_hop_or_high_conf"] += 1
            continue
        if plan_utility < float(min_plan_utility):
            reject_counts["low_plan_utility"] += 1
            continue
        if plan_net < float(min_plan_net_utility):
            reject_counts["low_plan_net_utility"] += 1
            continue
        if plan_risk > float(max_plan_risk):
            reject_counts["high_plan_risk"] += 1
            continue
        safety_ok, safety_reasons = _shadow_plan_relation_safe(
            plan,
            question=question,
            state=state,
            calibrated_shadow_admission=bool(calibrated_shadow_admission),
        )
        if not safety_ok:
            for reason in safety_reasons:
                reject_counts[reason] += 1
            continue
        if plan_type == "two_hop_fact_join":
            strong_structural_support = (
                bool(calibrated_shadow_admission)
                and endpoint_overlap >= 0.50
                and schema_selectivity >= 0.58
                and (
                    relation_overlap >= 0.08
                    or (
                        provenance_score >= 0.425
                        and endpoint_overlap >= 0.95
                        and best_rank <= 2
                        and worst_rank <= 12
                    )
                )
            )
            if endpoint_overlap < 0.24:
                reject_counts["low_endpoint_overlap"] += 1
                continue
            if relation_overlap < 0.08 and not strong_structural_support:
                reject_counts["low_relation_overlap"] += 1
                continue
            if schema_selectivity < 0.45 and not strong_structural_support:
                reject_counts["low_schema_selectivity"] += 1
                continue
        if best_rank > 40 and provenance_score < 0.12 and not (
            bool(calibrated_shadow_admission)
            and plan_net >= max(float(min_plan_net_utility), 0.18)
            and endpoint_overlap >= 0.55
            and schema_selectivity >= 0.60
        ):
            reject_counts["late_rank_low_provenance"] += 1
            continue
        if worst_rank > int(max_original_rank):
            reject_counts["rank_too_deep"] += 1
            continue

        compact_doc = _format_typed_shadow_table(question, plan)
        compact_key = _doc_key(compact_doc)
        if not compact_key or compact_key in seen:
            reject_counts["duplicate_doc"] += 1
            continue
        lineage_metadata = _candidate_lineage_metadata(
            _shadow_plan_surface(plan),
            "typed_fact_join_shadow",
            lineage_context,
        )
        if (
            require_lineage_binding
            and
            lineage_context.get("required_uncovered_slot_ids")
            and not lineage_metadata.get("uncovered_slot_ids")
        ):
            reject_counts["no_uncovered_slot_binding"] += 1
            continue
        seen.add(compact_key)
        record = {
            "doc": compact_doc,
            "source": "typed_fact_join_shadow",
            "metadata": {
                "plan_index": int(plan_idx),
                "plan_type": plan_type,
                "plan_utility": plan_utility,
                "plan_net_utility": plan_net,
                "plan_risk": plan_risk,
                "best_rank": best_rank,
                "worst_rank": worst_rank,
                "endpoint_overlap": endpoint_overlap,
                "relation_overlap": relation_overlap,
                "schema_selectivity": schema_selectivity,
                "provenance_score": provenance_score,
                "candidate_answer": str(plan.get("candidate_answer") or "")[:160],
                "answer_type": str(plan.get("answer_type") or ""),
                "triple_path": list(plan.get("triple_path") or [])[:2],
                "schema_signatures": list(plan.get("schema_signatures") or [])[:3],
                "compact_table": True,
                **lineage_metadata,
            },
        }
        candidates.append(record)
        if len(candidates) >= max_candidates:
            break

    info["candidate_count"] = len(candidates)
    info["accepted_count"] = len(candidates)
    info["rejected_count"] = sum(reject_counts.values())
    info["reject_reason_counts"] = dict(reject_counts)
    info["reason"] = "candidates_collected" if candidates else "no_candidate_accepted"
    info["top_candidates"] = [
        {
            "title": _title(item.get("doc")),
            **{
                key: value
                for key, value in (item.get("metadata") or {}).items()
                if key
                in {
                    "plan_type",
                    "plan_utility",
                    "plan_net_utility",
                    "plan_risk",
                    "best_rank",
                    "worst_rank",
                    "endpoint_overlap",
                    "relation_overlap",
                    "schema_selectivity",
                    "provenance_score",
                    "candidate_answer",
                }
            },
        }
        for item in candidates[:5]
    ]
    return candidates, info


def _score_extra_doc(
    *,
    question_tokens: set,
    missing_tokens: set,
    base_token_sets: List[set],
    doc: str,
    source: str,
    token_budget: int,
    min_query_overlap: float,
    token_weight: float,
    risk_weight: float,
    redundancy_penalty: float,
    compute_weight: float,
    candidate_count: int,
    planned_state: Dict[str, Any],
    candidate_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    candidate_metadata = candidate_metadata or {}
    doc_tokens = _tokens(doc)
    title_tokens = _tokens(_title(doc))
    query_overlap = _coverage(question_tokens, doc_tokens)
    title_overlap = _coverage(question_tokens, title_tokens)
    missing_gain = _coverage(missing_tokens, doc_tokens) if missing_tokens else 0.0
    max_base_overlap = max((_jaccard(doc_tokens, base_tokens) for base_tokens in base_token_sets), default=0.0)
    novelty = _clamp(1.0 - max_base_overlap)
    table_score, table_row_count = _candidate_table_score(doc)

    source_prior = _source_prior(source)
    bridge_bonus = 0.05 if planned_state.get("bridge_cues") else 0.0
    deficit_bonus = 0.07 if planned_state.get("evidence_insufficient") else 0.0
    table_bonus = 0.10 * table_score if source == "relation_fact_table" else 0.0
    plan_bonus = 0.0
    compact_table_bonus = 0.0
    coverage_gain = _clamp(_as_float(candidate_metadata.get("coverage_gain"), 0.0))
    lineage_confidence = _clamp(
        _as_float(candidate_metadata.get("lineage_confidence"), 0.0)
    )
    if source == "typed_fact_join_shadow":
        plan_utility = _clamp(_as_float(candidate_metadata.get("plan_utility"), 0.0))
        plan_net = _clamp(_as_float(candidate_metadata.get("plan_net_utility"), 0.0), -1.0, 1.0)
        endpoint_overlap = _clamp(_as_float(candidate_metadata.get("endpoint_overlap"), 0.0))
        relation_overlap = _clamp(_as_float(candidate_metadata.get("relation_overlap"), 0.0))
        schema_selectivity = _clamp(_as_float(candidate_metadata.get("schema_selectivity"), 0.0))
        provenance_score = _clamp(_as_float(candidate_metadata.get("provenance_score"), 0.0))
        plan_bonus = (
            0.10 * plan_utility
            + 0.05 * max(0.0, plan_net)
            + 0.06 * endpoint_overlap
            + 0.06 * relation_overlap
            + 0.04 * schema_selectivity
            + 0.04 * provenance_score
        )
        if candidate_metadata.get("compact_table"):
            compact_table_bonus = 0.04
    elif source == "entity_predicate_fact_join":
        entity_predicate_score = _clamp(
            _as_float(candidate_metadata.get("entity_predicate_score"), 0.0)
        )
        relation_overlap = _clamp(
            _as_float(candidate_metadata.get("relation_overlap"), 0.0)
        )
        provenance_score = _clamp(
            _as_float(candidate_metadata.get("provenance_score"), 0.0)
        )
        plan_bonus = (
            0.10 * entity_predicate_score
            + 0.08 * relation_overlap
            + 0.04 * provenance_score
        )
    elif source == "query_schema_memory" and candidate_metadata.get(
        "validated_schema_tuple_materialization"
    ):
        schema_score = _clamp(_as_float(candidate_metadata.get("schema_score"), 0.0))
        relation_overlap = _clamp(
            _as_float(candidate_metadata.get("relation_overlap"), 0.0)
        )
        join_key_overlap = _clamp(
            _as_float(candidate_metadata.get("join_key_overlap"), 0.0)
        )
        predicate_contract = 1.0 if candidate_metadata.get("predicate_contract_match") else 0.0
        plan_bonus = (
            0.08 * schema_score
            + 0.06 * relation_overlap
            + 0.05 * join_key_overlap
            + 0.04 * predicate_contract
        )
        compact_table_bonus = 0.05
    answer_type_signal = 0.0
    question_text_tokens = question_tokens
    doc_text = str(doc or "").lower()
    if {"date", "year", "when"} & question_text_tokens and re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", doc_text):
        answer_type_signal = 0.04
    elif {"how", "many", "number"} & question_text_tokens and re.search(r"\b[0-9]+([.,][0-9]+)?\b", doc_text):
        answer_type_signal = 0.04

    utility = _clamp(
        0.30 * query_overlap
        + 0.24 * missing_gain
        + 0.16 * novelty
        + 0.10 * title_overlap
        + source_prior
        + bridge_bonus
        + deficit_bonus
        + table_bonus
        + plan_bonus
        + compact_table_bonus
        + answer_type_signal
        + 0.18 * coverage_gain
        + 0.06 * lineage_confidence
    )

    token_count = max(1, len(re.findall(r"\S+", str(doc or ""))))
    token_cost = _clamp(token_count / max(1.0, float(token_budget)))
    compute_cost = _clamp(math.log1p(max(1, candidate_count)) / math.log1p(32.0))
    low_query_risk = _clamp(
        (float(min_query_overlap) - query_overlap)
        / max(float(min_query_overlap), 1e-6)
    )
    low_query_risk *= 1.0 - 0.75 * coverage_gain
    low_missing_risk = 0.0
    if missing_tokens:
        low_missing_risk = _clamp((0.08 - missing_gain) / 0.08)
        low_missing_risk *= 1.0 - 0.75 * coverage_gain
    redundancy_risk = _clamp(max_base_overlap)
    long_context_risk = 0.12 if planned_state.get("long_context") and token_count > token_budget * 0.65 else 0.0
    table_conflict_risk = 0.0
    if source == "relation_fact_table" and table_row_count > 2 and table_score < 0.66:
        table_conflict_risk = 0.10
    generic_extra_risk = 0.08 if source == "extra_evidence" else 0.0
    plan_risk = 0.0
    if source == "typed_fact_join_shadow":
        plan_risk = 0.16 * _clamp(_as_float(candidate_metadata.get("plan_risk"), 0.0))
        if candidate_metadata.get("compact_table"):
            generic_extra_risk = 0.0
    elif source == "entity_predicate_fact_join":
        plan_risk = 0.10 * _clamp(
            _as_float(candidate_metadata.get("plan_risk"), 0.0)
        )
    elif source == "query_schema_memory" and candidate_metadata.get(
        "validated_schema_tuple_materialization"
    ):
        low_query_risk *= 0.55
        low_missing_risk *= 0.55
        redundancy_risk *= 0.75
    risk = _clamp(
        0.26 * low_query_risk
        + 0.18 * low_missing_risk
        + 0.22 * redundancy_risk
        + long_context_risk
        + table_conflict_risk
        + generic_extra_risk
        + plan_risk
    )
    net_utility = (
        utility
        - float(token_weight) * token_cost
        - float(compute_weight) * compute_cost
        - float(risk_weight) * risk
        - float(redundancy_penalty) * redundancy_risk
    )
    return {
        "source": source,
        "title": _title(doc),
        "token_count": token_count,
        "query_overlap": _round_float(query_overlap),
        "missing_gain": _round_float(missing_gain),
        "novelty": _round_float(novelty),
        "table_score": _round_float(table_score),
        "table_row_count": int(table_row_count),
        "utility": _round_float(utility),
        "risk": _round_float(risk),
        "token_cost": _round_float(token_cost),
        "compute_cost": _round_float(compute_cost),
        "net_utility": _round_float(net_utility),
        "plan_type": str(candidate_metadata.get("plan_type") or ""),
        "plan_net_utility": _round_float(candidate_metadata.get("plan_net_utility", 0.0)),
        "plan_utility": _round_float(candidate_metadata.get("plan_utility", 0.0)),
        "plan_risk": _round_float(candidate_metadata.get("plan_risk", 0.0)),
        "endpoint_overlap": _round_float(candidate_metadata.get("endpoint_overlap", 0.0)),
        "relation_overlap": _round_float(candidate_metadata.get("relation_overlap", 0.0)),
        "schema_score": _round_float(candidate_metadata.get("schema_score", 0.0)),
        "entity_predicate_score": _round_float(
            candidate_metadata.get("entity_predicate_score", 0.0)
        ),
        "schema_idf": _round_float(candidate_metadata.get("schema_idf", 0.0)),
        "join_key_overlap": _round_float(candidate_metadata.get("join_key_overlap", 0.0)),
        "schema_frequency": _as_int(candidate_metadata.get("schema_frequency"), 0),
        "original_rank": _as_int(candidate_metadata.get("original_rank"), 0),
        "schema_selectivity": _round_float(candidate_metadata.get("schema_selectivity", 0.0)),
        "provenance_score": _round_float(candidate_metadata.get("provenance_score", 0.0)),
        "compact_table": bool(candidate_metadata.get("compact_table")),
        "best_rank": _as_int(candidate_metadata.get("best_rank"), 0),
        "worst_rank": _as_int(candidate_metadata.get("worst_rank"), 0),
        "candidate_answer": str(candidate_metadata.get("candidate_answer") or "")[:160],
        "answer_type": str(candidate_metadata.get("answer_type") or ""),
        "triple_path": list(candidate_metadata.get("triple_path") or [])[:3],
        "answer_slot": str(candidate_metadata.get("answer_slot") or "unknown"),
        "active_entity": str(candidate_metadata.get("active_entity") or "")[:160],
        "predicate": str(candidate_metadata.get("predicate") or "")[:160],
        "direction": str(candidate_metadata.get("direction") or ""),
        "answer_value": str(candidate_metadata.get("answer_value") or "")[:160],
        "answer_type_match": bool(candidate_metadata.get("answer_type_match")),
        "cross_schema": bool(candidate_metadata.get("cross_schema")),
        "retrieval_pool_member": bool(
            candidate_metadata.get("retrieval_pool_member", True)
        ),
        "required_slot_ids": list(candidate_metadata.get("required_slot_ids") or []),
        "covered_slot_ids": list(candidate_metadata.get("covered_slot_ids") or []),
        "uncovered_slot_ids": list(candidate_metadata.get("uncovered_slot_ids") or []),
        "coverage_gain": _round_float(candidate_metadata.get("coverage_gain", 0.0)),
        "lineage_confidence": _round_float(
            candidate_metadata.get("lineage_confidence", 0.0)
        ),
        "lineage_group_id": str(candidate_metadata.get("lineage_group_id") or ""),
        "comparison_arm": str(candidate_metadata.get("comparison_arm") or "")[:160],
        "covered_branch_ids": list(
            candidate_metadata.get("covered_branch_ids") or []
        ),
        "comparison_branch_id": str(
            candidate_metadata.get("comparison_branch_id") or ""
        ),
        "predicate_tokens": list(candidate_metadata.get("predicate_tokens") or [])[:12],
        "comparison_query": bool(candidate_metadata.get("comparison_query")),
        "compact_projection": bool(candidate_metadata.get("compact_projection")),
        "projection_reason": str(candidate_metadata.get("projection_reason") or ""),
        "original_doc_token_count": _as_int(
            candidate_metadata.get("original_doc_token_count"),
            0,
        ),
        "projected_doc_token_count": _as_int(
            candidate_metadata.get("projected_doc_token_count"),
            0,
        ),
        "projection_token_savings": _as_int(
            candidate_metadata.get("projection_token_savings"),
            0,
        ),
        "lineage_bound": bool(candidate_metadata.get("lineage_bound")),
        "direct_slot_grounded": bool(
            candidate_metadata.get("direct_slot_grounded")
        ),
        "operator_validation_passed": bool(
            candidate_metadata.get("operator_validation_passed")
        ),
        "operator_validation_score": _round_float(
            candidate_metadata.get("operator_validation_score", 0.0)
        ),
        "root_entity": str(candidate_metadata.get("root_entity") or "")[:160],
        "root_anchor_score": _round_float(
            candidate_metadata.get("root_anchor_score", 0.0)
        ),
        "lineage_depth": _as_int(candidate_metadata.get("lineage_depth"), 0),
        "lineage_path": list(candidate_metadata.get("lineage_path") or [])[:3],
        "entity_predicate_slot_id": str(
            candidate_metadata.get("entity_predicate_slot_id") or ""
        ),
        "entity_predicate_slot_text": str(
            candidate_metadata.get("entity_predicate_slot_text") or ""
        )[:240],
        "entity_predicate_target_slot_id": str(
            candidate_metadata.get("entity_predicate_target_slot_id") or ""
        ),
        "entity_predicate_slot_match": _round_float(
            candidate_metadata.get("entity_predicate_slot_match", 0.0)
        ),
        "predicate_contract_match": bool(
            candidate_metadata.get("predicate_contract_match")
        ),
        "predicate_match_mode": str(
            candidate_metadata.get("predicate_match_mode") or ""
        ),
        "direct_predicate_overlap": _round_float(
            candidate_metadata.get("direct_predicate_overlap", 0.0)
        ),
        "slot_source": str(candidate_metadata.get("slot_source") or "unknown"),
        "slot_hop_index": _as_int(candidate_metadata.get("slot_hop_index"), -1),
        "slot_target_predicate_concepts": list(
            candidate_metadata.get("slot_target_predicate_concepts") or []
        ),
        "slot_expected_value_type": str(
            candidate_metadata.get("slot_expected_value_type") or "unknown"
        ),
        "slot_direction": str(
            candidate_metadata.get("slot_direction") or "bound_entity_to_value"
        ),
        "slot_materialization_safe": bool(
            candidate_metadata.get("slot_materialization_safe")
        ),
        "slot_unsafe_reasons": list(
            candidate_metadata.get("slot_unsafe_reasons") or []
        ),
        "slot_meta_negative": bool(candidate_metadata.get("slot_meta_negative")),
        "terminal_slot_gate_passed": bool(
            candidate_metadata.get("terminal_slot_gate_passed")
        ),
        "terminal_slot_gate_reasons": list(
            candidate_metadata.get("terminal_slot_gate_reasons") or []
        ),
        "terminal_answer_slot": str(
            candidate_metadata.get("terminal_answer_slot") or ""
        ),
        "terminal_value_type": str(
            candidate_metadata.get("terminal_value_type") or ""
        ),
        "terminal_slot_stage": str(
            candidate_metadata.get("terminal_slot_stage") or ""
        ),
        "terminal_materialization_role": str(
            candidate_metadata.get("terminal_materialization_role") or ""
        ),
        "terminal_relation_direct": bool(
            candidate_metadata.get("terminal_relation_direct")
        ),
        "terminal_relation_intersection": list(
            candidate_metadata.get("terminal_relation_intersection") or []
        ),
        "source_sentence_value_bound": bool(
            candidate_metadata.get("source_sentence_value_bound")
        ),
        "source_sentence_join_bound": bool(
            candidate_metadata.get("source_sentence_join_bound")
        ),
        "source_sentence_relation_bound": bool(
            candidate_metadata.get("source_sentence_relation_bound")
        ),
        "source_sentence_trace_fallback": bool(
            candidate_metadata.get("source_sentence_trace_fallback")
        ),
        "estimated_endpoint_scan_rows": max(
            1,
            _as_int(
                candidate_metadata.get("estimated_endpoint_scan_rows"),
                _as_int(candidate_metadata.get("schema_frequency"), 1),
            ),
        ),
        "operator_latency_ms": _round_float(
            candidate_metadata.get("operator_latency_ms", 0.0)
        ),
        "operator_name": str(
            candidate_metadata.get("operator_name") or source
        ),
        "early_schema_bridge_materialization": bool(
            candidate_metadata.get("early_schema_bridge_materialization")
        ),
        "admission_min_score": _round_float(
            candidate_metadata.get("admission_min_score", 0.0)
        ),
        "admission_min_join_key_overlap": _round_float(
            candidate_metadata.get("admission_min_join_key_overlap", 0.0)
        ),
        "admission_min_relation_overlap": _round_float(
            candidate_metadata.get("admission_min_relation_overlap", 0.0)
        ),
        "active_schema_bridge_tuple_executor": bool(
            candidate_metadata.get("active_schema_bridge_tuple_executor")
        ),
        "active_schema_bridge_tuple_bound_signal_count": _as_int(
            candidate_metadata.get("active_schema_bridge_tuple_bound_signal_count"),
            0,
        ),
    }


def apply_cost_based_evidence_admission(
    *,
    question: str,
    docs: List[str],
    base_doc_count: int,
    planned_retrieval: Optional[Dict[str, Any]] = None,
    relation_fact_join: Optional[Dict[str, Any]] = None,
    dual_channel_projection: Optional[Dict[str, Any]] = None,
    support_set_completion: Optional[Dict[str, Any]] = None,
    path_evidence: Optional[Dict[str, Any]] = None,
    typed_fact_join_shadow: Optional[Dict[str, Any]] = None,
    query_schema_memory: Optional[Dict[str, Any]] = None,
    entity_predicate_fact_join: Optional[Dict[str, Any]] = None,
    selective_support_join: Optional[Dict[str, Any]] = None,
    get_doc_content: Optional[Callable[[int], str]] = None,
    enabled: bool = False,
    trigger_policy: str = "high_risk_only",
    admission_policy: str = "learned",
    optimize_existing_extra_docs: bool = False,
    max_extra_docs: int = 2,
    token_budget: int = 900,
    min_utility: float = 0.32,
    max_risk: float = 0.62,
    min_net_utility: float = -0.04,
    min_query_overlap: float = 0.05,
    token_weight: float = 0.12,
    compute_weight: float = 0.04,
    risk_weight: float = 0.42,
    redundancy_penalty: float = 0.08,
    preserve_best_on_empty: bool = True,
    deficit_aware_replacement_enabled: bool = False,
    deficit_aware_replacement_min_margin: float = 0.08,
    deficit_aware_replacement_max_evictions: int = 1,
    shadow_candidate_top_k: int = 4,
    shadow_max_original_rank: int = 80,
    shadow_min_plan_utility: float = 0.42,
    shadow_min_plan_net_utility: float = 0.04,
    shadow_max_plan_risk: float = 0.58,
    shadow_require_two_hop_for_bridge: bool = True,
    calibrated_shadow_admission: bool = False,
    source_sufficiency_veto: bool = True,
    learned_cost_optimizer_enabled: bool = False,
    learned_cost_optimizer_model_path: str = "",
    learned_cost_optimizer_min_expected_gain: float = 0.012,
    learned_cost_optimizer_min_confidence: float = 0.30,
    learned_cost_optimizer_min_support: int = 4,
    learned_cost_optimizer_max_degrade: float = 0.46,
    learned_cost_optimizer_veto_min_degrade_margin: float = 0.0,
    counterfactual_drop_rank: int = -1,
    counterfactual_slate_min_harm_signals: int = 2,
    counterfactual_slate_min_stable_datasets: int = 3,
    fact_join_plan_optimizer_enabled: bool = False,
    fact_join_plan_compute_budget: float = 1.0,
    fact_join_plan_min_objective: float = 0.0,
    fact_join_plan_max_enumerated_candidates: int = 12,
    route_early_schema_through_optimizer: bool = False,
    candidate_pool_snapshot_dir: str = "",
    candidate_pool_snapshot_mode: str = "off",
    candidate_pool_snapshot_strict: bool = True,
    candidate_pool_snapshot_key: str = "",
    schema_materialization_enabled: bool = True,
    schema_materialization_max_candidates: int = 1,
    schema_materialization_max_original_rank: int = 32,
    schema_materialization_min_score: float = 0.70,
    schema_materialization_min_schema_idf: float = 0.60,
    schema_materialization_min_query_overlap: float = 0.10,
    schema_materialization_min_join_key_overlap: float = 0.66,
    schema_materialization_min_relation_overlap: float = 0.50,
    schema_materialization_require_answer_type_value: bool = True,
    query_schema_memory_cross_schema_join_enabled: bool = False,
    query_schema_memory_cross_schema_max_candidates: int = 48,
    query_schema_memory_cross_schema_max_endpoint_scan_rows: int = 48,
    early_schema_bridge_materialization_enabled: bool = False,
    early_schema_bridge_materialization_max_candidates: int = 1,
    early_schema_bridge_materialization_max_original_rank: int = 80,
    early_schema_bridge_materialization_min_score: float = 0.66,
    early_schema_bridge_materialization_min_schema_idf: float = 0.35,
    early_schema_bridge_materialization_min_query_overlap: float = 0.08,
    early_schema_bridge_materialization_min_join_key_overlap: float = 0.50,
    early_schema_bridge_materialization_min_relation_overlap: float = 0.20,
    early_schema_bridge_materialization_require_answer_type_value: bool = True,
    early_schema_bridge_materialization_require_lineage_binding: bool = True,
    early_schema_bridge_materialization_allow_relation_relaxation: bool = True,
    early_schema_bridge_materialization_exact_join_key_min_schema_idf: float = 0.65,
    early_schema_bridge_materialization_exact_join_key_min_relation_overlap: float = 0.05,
    early_schema_bridge_materialization_compact_projection: bool = True,
    early_schema_bridge_materialization_require_residual_deficit: bool = True,
    early_schema_bridge_materialization_require_anchor_binding: bool = True,
    early_schema_bridge_materialization_safe_max_original_rank: int = 48,
    early_schema_bridge_materialization_deep_rank_min_score: float = 0.70,
    validated_schema_tuple_materialization_enabled: bool = False,
    validated_schema_tuple_materialization_max_candidates: int = 1,
    validated_schema_tuple_materialization_min_score: float = 0.58,
    validated_schema_tuple_materialization_min_relation_overlap: float = 0.05,
    validated_schema_tuple_materialization_require_predicate_contract: bool = True,
    validated_schema_tuple_materialization_require_terminal_slot: bool = True,
    validated_schema_tuple_materialization_require_source_sentence: bool = True,
    validated_schema_tuple_materialization_min_token_savings: int = 1,
    validated_schema_tuple_materialization_block_reader_visible_source: bool = True,
    active_schema_bridge_tuple_executor_enabled: bool = False,
    active_schema_bridge_tuple_min_score: float = 0.52,
    active_schema_bridge_tuple_min_bound_signals: int = 2,
    entity_predicate_materialization_enabled: bool = False,
    entity_predicate_materialization_max_candidates: int = 1,
    entity_predicate_materialization_min_score: float = 0.64,
    entity_predicate_materialization_min_relation_overlap: float = 0.50,
    entity_predicate_materialization_min_lineage_confidence: float = 0.34,
    entity_predicate_materialization_require_atomic_slot: bool = True,
    entity_predicate_materialization_allowed_match_modes: Any = (
        "direct_token,polarity_concept"
    ),
    entity_predicate_materialization_compact_projection: bool = False,
    entity_predicate_terminal_tuple_materialization: bool = False,
    entity_predicate_terminal_tuple_min_lineage_depth: int = 1,
    entity_predicate_terminal_tuple_strict_branch_binding: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    """Optimize reader-visible appended evidence without touching original top-k.

    This is an admission controller, not a reranker: the first `base_doc_count`
    documents are treated as stable retrieval evidence. Additional reader-only
    evidence from planned retrieval, schema/fact joins, or shadow channels is
    scored with a local cost/selectivity model and either admitted or withheld.
    """
    docs = list(docs or [])
    base_doc_count = max(0, min(int(base_doc_count), len(docs)))
    requested_admission_policy = _normalize_cost_admission_policy(admission_policy)
    effective_admission_policy = requested_admission_policy
    snapshot_mode = str(candidate_pool_snapshot_mode or "off").strip().lower()
    if snapshot_mode not in {"off", "record", "replay"}:
        snapshot_mode = "off"
    if snapshot_mode != "off" and not str(candidate_pool_snapshot_dir or "").strip():
        raise ValueError("candidate_pool_snapshot_dir is required for record/replay mode")
    if requested_admission_policy in {
        "learned",
        "learned_veto",
        "counterfactual_slate_veto",
    } and not learned_cost_optimizer_enabled:
        effective_admission_policy = (
            "always_on"
            if requested_admission_policy in {"learned_veto", "counterfactual_slate_veto"}
            else "rule"
        )
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled" if not enabled else "not_evaluated",
        "trigger_policy": str(trigger_policy or "high_risk_only"),
        "admission_policy": requested_admission_policy,
        "effective_admission_policy": effective_admission_policy,
        "optimize_existing_extra_docs": bool(optimize_existing_extra_docs),
        "base_doc_count": int(base_doc_count),
        "input_doc_count": len(docs),
        "candidate_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "final_doc_count": len(docs),
        "max_extra_docs": int(max_extra_docs),
        "min_utility": _round_float(min_utility),
        "max_risk": _round_float(max_risk),
        "min_net_utility": _round_float(min_net_utility),
        "accepted_titles": [],
        "rejected_titles": [],
        "reject_reason_counts": {},
        "source_counts": {},
        "accepted_source_counts": {},
        "accepted_entity_predicate_titles": [],
        "accepted_entity_predicate_doc_fingerprints": [],
        "top_candidates": [],
        "candidate_pool_fingerprint": _candidate_pool_fingerprint([]),
        "candidate_pool_token_count": 0,
        "admitted_candidate_count": 0,
        "admitted_candidate_tokens": 0,
        "token_budget_enforced": True,
        "token_budget_utilization": 0.0,
        "deficit_aware_replacement": {
            "enabled": bool(deficit_aware_replacement_enabled),
            "replacement_count": 0,
            "attempt_count": 0,
            "slot_loss_veto_count": 0,
            "branch_loss_veto_count": 0,
            "pairs": [],
            "min_margin": _round_float(deficit_aware_replacement_min_margin),
            "max_evictions": max(0, int(deficit_aware_replacement_max_evictions)),
        },
        "decision_trace": [],
        "counterfactual_intervention": {
            "enabled": requested_admission_policy == "counterfactual_drop",
            "requested_drop_rank": int(counterfactual_drop_rank),
            "applied": False,
            "dropped_candidate_id": "",
            "baseline_accepted_candidate_ids": [],
            "baseline_accepted_count": 0,
            "baseline_admitted_tokens": 0,
        },
        "shadow_candidate_admission": {},
        "query_schema_memory_materialization": {},
        "early_schema_bridge_materialization": {},
        "query_schema_memory_cross_schema_join": {},
        "entity_predicate_fact_join_materialization": {},
        "active_schema_bridge_tuple_executor": {
            "enabled": bool(active_schema_bridge_tuple_executor_enabled),
            "min_score": _round_float(active_schema_bridge_tuple_min_score),
            "min_bound_signals": int(
                max(1, int(active_schema_bridge_tuple_min_bound_signals))
            ),
            "candidate_count": 0,
            "accepted_count": 0,
            "relaxed_count": 0,
            "trace_fallback_count": 0,
            "covered_by_reader_visible_count": 0,
        },
        "entity_predicate_compact_projection": bool(
            entity_predicate_materialization_compact_projection
        ),
        "calibrated_shadow_admission": bool(calibrated_shadow_admission),
        "source_sufficiency_veto_enabled": bool(source_sufficiency_veto),
        "learned_cost_optimizer": {
            "enabled": bool(learned_cost_optimizer_enabled),
            "model_path": str(learned_cost_optimizer_model_path or ""),
            "min_expected_gain": _round_float(learned_cost_optimizer_min_expected_gain),
            "min_confidence": _round_float(learned_cost_optimizer_min_confidence),
            "min_support": int(learned_cost_optimizer_min_support),
            "max_degrade": _round_float(learned_cost_optimizer_max_degrade),
            "veto_min_degrade_margin": _round_float(
                learned_cost_optimizer_veto_min_degrade_margin
            ),
            "counterfactual_min_harm_signals": int(counterfactual_slate_min_harm_signals),
            "counterfactual_min_stable_datasets": int(
                counterfactual_slate_min_stable_datasets
            ),
            "model_enabled": False,
            "reason": "disabled" if not learned_cost_optimizer_enabled else "not_loaded",
        },
        "candidate_pool_snapshot": {
            "mode": snapshot_mode,
            "directory": str(candidate_pool_snapshot_dir or ""),
            "key": str(candidate_pool_snapshot_key or question or ""),
            "strict": bool(candidate_pool_snapshot_strict),
            "loaded": False,
            "written": False,
            "path": "",
        },
        "fact_join_plan_optimizer": {
            "enabled": bool(fact_join_plan_optimizer_enabled),
            "reason": (
                "not_evaluated" if fact_join_plan_optimizer_enabled else "disabled"
            ),
            "route_early_schema_through_optimizer": bool(
                route_early_schema_through_optimizer
            ),
            "compute_budget": _round_float(fact_join_plan_compute_budget),
            "min_objective": _round_float(fact_join_plan_min_objective),
        },
        "query_schema_memory_candidate_count": int((query_schema_memory or {}).get("candidate_count", 0) or 0),
        "entity_predicate_fact_join_candidate_count": int(
            (entity_predicate_fact_join or {}).get("candidate_count", 0) or 0
        ),
        "selective_support_join_reason": (selective_support_join or {}).get("reason"),
    }
    if not enabled:
        return docs, info

    learned_model: Dict[str, Any] = {}
    use_learned_policy = bool(
        effective_admission_policy in {
            "learned",
            "learned_veto",
            "counterfactual_slate_veto",
        }
        and learned_cost_optimizer_enabled
    )
    if use_learned_policy:
        learned_model = load_learned_cost_optimizer_model(str(learned_cost_optimizer_model_path or ""))
        training_manifest = learned_model.get("training_manifest") or {}
        info["learned_cost_optimizer"].update(
            {
                "model_enabled": bool(learned_model.get("enabled")),
                "reason": str(learned_model.get("reason") or "loaded"),
                "model_path": str(learned_model.get("model_path") or learned_cost_optimizer_model_path or ""),
                "model_version": int(learned_model.get("version", 0) or 0),
                "training_protocol": str(training_manifest.get("protocol") or "unknown"),
                "training_sample_key_sha256": str(
                    training_manifest.get("sample_key_sha256") or ""
                ),
                "training_excluded_datasets": list(
                    training_manifest.get("excluded_datasets") or []
                ),
            }
        )
        if not bool(learned_model.get("enabled")):
            effective_admission_policy = (
                "always_on"
                if requested_admission_policy in {
                    "learned_veto",
                    "counterfactual_slate_veto",
                }
                else "rule"
            )
            info["effective_admission_policy"] = effective_admission_policy
            use_learned_policy = False

    base_docs = docs[:base_doc_count]
    extra_docs = docs[base_doc_count:]

    planned_retrieval = planned_retrieval or {}
    relation_fact_join = relation_fact_join or {}
    dual_channel_projection = dual_channel_projection or {}
    support_set_completion = support_set_completion or {}
    path_evidence = path_evidence or {}
    typed_fact_join_shadow = typed_fact_join_shadow or {}
    query_schema_memory = query_schema_memory or {}
    entity_predicate_fact_join = entity_predicate_fact_join or {}
    selective_support_join = selective_support_join or {}
    query_schema_memory, cross_schema_merge_info = (
        _merge_cross_schema_schema_memory_candidates(
            query_schema_memory,
            entity_predicate_fact_join,
            enabled=bool(query_schema_memory_cross_schema_join_enabled),
            max_candidates=max(
                0, int(query_schema_memory_cross_schema_max_candidates)
            ),
        )
    )
    info["query_schema_memory_cross_schema_join"] = cross_schema_merge_info
    info["query_schema_memory_candidate_count"] = int(
        query_schema_memory.get("candidate_count", 0) or 0
    )

    state = _planned_state(planned_retrieval)
    info.update(
        {
            "planned_reason": state["reason"],
            "question_type": state["question_type"],
            "bridge_strength": int(state["bridge_strength"]),
            "evidence_insufficient": bool(state["evidence_insufficient"]),
            "long_context": bool(state["long_context"]),
            "top2_changed": bool(state["top2_changed"]),
        }
    )
    question_tokens = _tokens(question)
    base_token_sets = [_tokens(doc) for doc in base_docs]
    missing_tokens = set(state.get("missing_tokens") or set())
    lineage_context = _build_lineage_context(question, planned_retrieval, base_docs)
    if (
        not lineage_context.get("required_slot_ids")
        and bool(state.get("evidence_insufficient"))
        and bool(entity_predicate_materialization_enabled)
        and (entity_predicate_fact_join.get("candidate_items") or [])
    ):
        best_join = (entity_predicate_fact_join.get("candidate_items") or [])[0]
        virtual_slot_id = "slot_entity_predicate_answer"
        virtual_slot = {
            "slot_id": virtual_slot_id,
            "text": (
                f"{str(best_join.get('active_entity') or '')} "
                f"{str(best_join.get('predicate') or '')}"
            ).strip()[:240],
            "anchor_tokens": sorted(_tokens(best_join.get("active_entity") or "")),
            "predicate_tokens": sorted(_tokens(best_join.get("predicate") or "")),
            "virtual": True,
        }
        lineage_context["required_slots"] = [virtual_slot]
        lineage_context["required_slot_ids"] = [virtual_slot_id]
        lineage_context["base_covered_slot_ids"] = []
        lineage_context["required_uncovered_slot_ids"] = [virtual_slot_id]
        lineage_context["entity_predicate_virtual_slot"] = True
    info["answer_slot"] = str(lineage_context.get("answer_slot") or "unknown")
    info["required_slots"] = list(lineage_context.get("required_slots") or [])
    info["required_slot_count"] = len(lineage_context.get("required_slot_ids") or [])
    info["base_covered_slot_ids"] = list(
        lineage_context.get("base_covered_slot_ids") or []
    )
    info["required_uncovered_slot_ids"] = list(
        lineage_context.get("required_uncovered_slot_ids") or []
    )
    info["comparison_query"] = bool(lineage_context.get("comparison_query"))
    reader_repair_veto_reasons: List[str] = []
    relation_join_reason = str((relation_fact_join or {}).get("reason") or "")
    if bool(source_sufficiency_veto):
        if (
            relation_join_reason == "current_evidence_sufficient"
            and not _has_relation_gap(relation_fact_join)
            and not _has_selective_support_gap(selective_support_join)
        ):
            reader_repair_veto_reasons.append("relation_source:evidence_sufficient")
        if not bool(state.get("evidence_insufficient")):
            reader_repair_veto_reasons.append("planner:evidence_sufficient")
        if (
            lineage_context.get("required_slot_ids")
            and not lineage_context.get("required_uncovered_slot_ids")
        ):
            reader_repair_veto_reasons.append("lineage:all_required_slots_covered")
    shadow_source_allowed = not reader_repair_veto_reasons
    info["reader_repair_source_allowed"] = bool(shadow_source_allowed)
    info["shadow_source_allowed"] = bool(shadow_source_allowed)
    info["candidate_level_join_sources_exempt_from_global_veto"] = [
        "query_schema_memory",
        "entity_predicate_fact_join",
    ]
    info["reader_repair_veto_reasons"] = list(reader_repair_veto_reasons)
    deficit_signal, deficit_reasons = _admission_deficit_signal(
        state=state,
        relation_fact_join=relation_fact_join,
        selective_support_join=selective_support_join,
        query_schema_memory=query_schema_memory,
    )
    info["deficit_signal"] = bool(deficit_signal)
    info["deficit_reasons"] = list(deficit_reasons)
    required_slot_ids = set(lineage_context.get("required_slot_ids") or [])
    residual_uncovered_slot_ids = set(
        lineage_context.get("required_uncovered_slot_ids") or []
    )
    residual_deficit_signal = bool(residual_uncovered_slot_ids) or bool(
        not required_slot_ids and state.get("evidence_insufficient")
    )
    residual_deficit_reason = (
        "uncovered_required_slots"
        if residual_uncovered_slot_ids
        else "planner_insufficient_without_slot_contract"
        if not required_slot_ids and state.get("evidence_insufficient")
        else "all_required_slots_covered"
        if required_slot_ids
        else "planner_evidence_sufficient"
    )
    info["residual_deficit_signal"] = bool(residual_deficit_signal)
    info["residual_deficit_reason"] = residual_deficit_reason
    info["residual_uncovered_slot_ids"] = sorted(residual_uncovered_slot_ids)
    shadow_signal, shadow_signal_reason, shadow_signal_plan = _best_calibrated_shadow_signal(
        typed_fact_join_shadow or {},
        min_plan_utility=shadow_min_plan_utility,
        min_plan_net_utility=shadow_min_plan_net_utility,
        max_plan_risk=shadow_max_plan_risk,
        question=question,
        state=state,
    )
    info["calibrated_shadow_signal"] = bool(shadow_signal)
    info["calibrated_shadow_signal_reason"] = shadow_signal_reason
    info["calibrated_shadow_best_plan"] = dict(shadow_signal_plan)

    extra_records: List[Dict[str, Any]] = []
    existing_doc_keys = {_doc_key(doc) for doc in docs}
    for idx, doc in enumerate(extra_docs):
        source = _infer_source(
            doc,
            planned_retrieval=planned_retrieval,
            relation_fact_join=relation_fact_join,
            dual_channel_projection=dual_channel_projection,
            support_set_completion=support_set_completion,
            path_evidence=path_evidence,
        )
        lineage_metadata = _candidate_lineage_metadata(doc, source, lineage_context)
        extra_records.append({
            "doc": doc,
            "source": source,
            "metadata": {"input_extra_rank": idx, **lineage_metadata},
        })

    protected_records = [
        record
        for record in extra_records
        if str(record.get("source") or "") in _PROTECTED_EXTRA_SOURCES
    ]
    protected_source_counts = Counter(str(record.get("source") or "extra_evidence") for record in protected_records)
    source_counts = Counter(str(record.get("source") or "extra_evidence") for record in extra_records)
    info["existing_extra_doc_count"] = len(extra_docs)
    info["protected_existing_extra_count"] = len(protected_records)
    info["protected_source_counts"] = dict(protected_source_counts)
    info["existing_extra_source_counts"] = dict(source_counts)
    portfolio_allow_shadow, portfolio_profile, portfolio_reject_reasons = _shadow_portfolio_decision(
        source_counts,
        max_extra_docs=max_extra_docs,
    )
    info["evidence_portfolio_policy"] = "source_aware_shadow_table_admission_v1"
    info["evidence_portfolio_profile"] = portfolio_profile
    info["evidence_portfolio_allows_shadow"] = bool(portfolio_allow_shadow)
    info["evidence_portfolio_reject_reasons"] = list(portfolio_reject_reasons)
    portfolio_override_allow, portfolio_override_profile, portfolio_override_reject_reasons = (
        _calibrated_portfolio_override_decision(
            source_counts,
            state=state,
            shadow_signal=bool(shadow_signal),
            max_extra_docs=max_extra_docs,
        )
        if bool(calibrated_shadow_admission)
        else (False, "disabled", ["calibrated_shadow_admission_disabled"])
    )
    info["evidence_portfolio_override"] = bool(portfolio_override_allow)
    info["evidence_portfolio_override_profile"] = portfolio_override_profile
    info["evidence_portfolio_override_reject_reasons"] = list(portfolio_override_reject_reasons)

    policy = str(trigger_policy or "high_risk_only").strip().lower()
    if policy not in {"high_risk_only", "always"}:
        policy = "high_risk_only"
    hard_deficit_signal = bool(deficit_signal) or state.get("reason") in {
        "no_new_planned_queries",
        "max_rounds_reached",
        "verifier_insufficient_no_retrieval",
        "residual_deficit_query_appended",
    }
    try:
        bridge_strength = int(state.get("bridge_strength") or 0)
    except (TypeError, ValueError):
        bridge_strength = 0
    simple_factoid_without_deficit = (
        str(state.get("question_type") or "").lower()
        in {"generic_factoid", "single_hop_factoid", "unavailable", "single_hop"}
        and bridge_strength < 2
        and not bool(state.get("bridge_cues"))
        and not bool(hard_deficit_signal)
    )
    info["shadow_simple_factoid_without_deficit"] = bool(simple_factoid_without_deficit)
    calibrated_state_allows_shadow = (
        bool(calibrated_shadow_admission)
        and bool(shadow_signal)
        and not bool(state.get("long_context"))
        and not bool(simple_factoid_without_deficit)
        and (
            bool(deficit_signal)
            or state.get("reason") in {"no_new_planned_queries", "max_rounds_reached", "verifier_insufficient_no_retrieval"}
        )
    )
    base_allow_shadow = (
        bool(deficit_signal)
        and not bool(state.get("long_context"))
        and not bool(simple_factoid_without_deficit)
        and (
            bool(state.get("bridge_cues"))
            or _has_relation_gap(relation_fact_join)
            or _has_selective_support_gap(selective_support_join)
        )
    )
    collection_allow_shadow = bool(
        (base_allow_shadow or calibrated_state_allows_shadow)
        and shadow_source_allowed
    )
    allow_shadow = bool(
        collection_allow_shadow
        and (portfolio_allow_shadow or portfolio_override_allow)
    )
    if not allow_shadow:
        shadow_info = {
            "enabled": bool(typed_fact_join_shadow and get_doc_content),
            "candidate_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "reason": (
                "blocked_long_context"
                if state.get("long_context")
                else "blocked_simple_factoid_without_deficit"
                if simple_factoid_without_deficit
                else "blocked_source_sufficiency_veto"
                if not shadow_source_allowed
                else "blocked_no_deficit"
                if not collection_allow_shadow
                else "blocked_source_portfolio"
                if not portfolio_allow_shadow
                else "blocked_no_bridge_or_join_gap"
            ),
            "portfolio_profile": portfolio_profile,
            "portfolio_reject_reasons": list(portfolio_reject_reasons),
            "portfolio_override": bool(portfolio_override_allow),
            "portfolio_override_profile": portfolio_override_profile,
            "portfolio_override_reject_reasons": list(portfolio_override_reject_reasons),
            "calibrated_signal": bool(shadow_signal),
            "calibrated_signal_reason": shadow_signal_reason,
            "reject_reason_counts": {},
            "top_candidates": [],
        }
        shadow_records: List[Dict[str, Any]] = []
    else:
        shadow_records, shadow_info = _collect_typed_fact_join_shadow_candidates(
            question=question,
            typed_fact_join_shadow=typed_fact_join_shadow,
            get_doc_content=get_doc_content,
            existing_doc_keys=existing_doc_keys,
            state=state,
            lineage_context=lineage_context,
            require_lineage_binding=bool(source_sufficiency_veto),
            bridge_cues=bool(state.get("bridge_cues")),
            max_candidates=shadow_candidate_top_k,
            max_original_rank=shadow_max_original_rank,
            min_plan_utility=shadow_min_plan_utility,
            min_plan_net_utility=shadow_min_plan_net_utility,
            max_plan_risk=shadow_max_plan_risk,
            require_two_hop_for_bridge=shadow_require_two_hop_for_bridge,
            calibrated_shadow_admission=bool(calibrated_shadow_admission),
        )
        shadow_info["portfolio_override"] = bool(portfolio_override_allow and not portfolio_allow_shadow)
        shadow_info["portfolio_override_profile"] = portfolio_override_profile
        shadow_info["portfolio_override_reject_reasons"] = list(portfolio_override_reject_reasons)
        shadow_info["calibrated_signal"] = bool(shadow_signal)
        shadow_info["calibrated_signal_reason"] = shadow_signal_reason

    qschema_reason = _reason_from(query_schema_memory, "reason")
    early_residual_gate_passed = bool(
        residual_deficit_signal
        if early_schema_bridge_materialization_require_residual_deficit
        else (
            bool(deficit_signal)
            or bool(state.get("evidence_insufficient"))
            or bool(residual_uncovered_slot_ids)
        )
    )
    early_schema_bridge_allowed = bool(
        early_schema_bridge_materialization_enabled
        and schema_materialization_enabled
        and query_schema_memory
        and get_doc_content
        and not bool(state.get("long_context"))
        and early_residual_gate_passed
        and (
            bool(state.get("bridge_cues"))
            or bridge_strength >= 2
            or _has_relation_gap(relation_fact_join)
            or _has_selective_support_gap(selective_support_join)
            or qschema_reason in {"candidates_generated", "no_accepted_candidates"}
        )
    )
    if early_schema_bridge_allowed:
        early_state = dict(state)
        early_state["evidence_insufficient"] = True
        early_state["bridge_cues"] = True
        early_schema_records, early_schema_info = _collect_query_schema_memory_candidates(
            question=question,
            query_schema_memory=query_schema_memory,
            get_doc_content=get_doc_content,
            existing_doc_keys=existing_doc_keys,
            state=early_state,
            lineage_context=lineage_context,
            enabled=True,
            max_candidates=max(0, int(early_schema_bridge_materialization_max_candidates)),
            max_original_rank=max(1, int(early_schema_bridge_materialization_max_original_rank)),
            min_score=max(0.0, float(early_schema_bridge_materialization_min_score)),
            min_schema_idf=max(0.0, float(early_schema_bridge_materialization_min_schema_idf)),
            min_query_overlap=max(0.0, float(early_schema_bridge_materialization_min_query_overlap)),
            min_join_key_overlap=max(
                0.0,
                float(early_schema_bridge_materialization_min_join_key_overlap),
            ),
            min_relation_overlap=max(
                0.0,
                float(early_schema_bridge_materialization_min_relation_overlap),
            ),
            require_answer_type_value=bool(
                early_schema_bridge_materialization_require_answer_type_value
            ),
            require_entity_bound=bool(
                (query_schema_memory or {}).get("entity_bound_join_enabled", True)
            ),
            require_lineage_binding=bool(
                early_schema_bridge_materialization_require_lineage_binding
            ),
            allow_exact_join_key_relation_relaxation=bool(
                early_schema_bridge_materialization_allow_relation_relaxation
            ),
            exact_join_key_min_schema_idf=max(
                0.0,
                float(
                    early_schema_bridge_materialization_exact_join_key_min_schema_idf
                ),
            ),
            exact_join_key_min_relation_overlap=max(
                0.0,
                float(
                    early_schema_bridge_materialization_exact_join_key_min_relation_overlap
                ),
            ),
            compact_projection=bool(
                early_schema_bridge_materialization_compact_projection
            ),
            require_residual_anchor_binding=bool(
                early_schema_bridge_materialization_require_anchor_binding
            ),
            safe_max_original_rank=max(
                1,
                int(early_schema_bridge_materialization_safe_max_original_rank),
            ),
            deep_rank_min_score=max(
                0.0,
                float(early_schema_bridge_materialization_deep_rank_min_score),
            ),
            cross_schema_max_endpoint_scan_rows=max(
                1,
                int(query_schema_memory_cross_schema_max_endpoint_scan_rows),
            ),
            validated_tuple_materialization=bool(
                validated_schema_tuple_materialization_enabled
            ),
            validated_tuple_max_candidates=max(
                0,
                int(validated_schema_tuple_materialization_max_candidates),
            ),
            validated_tuple_min_score=max(
                0.0,
                float(validated_schema_tuple_materialization_min_score),
            ),
            validated_tuple_min_relation_overlap=max(
                0.0,
                float(validated_schema_tuple_materialization_min_relation_overlap),
            ),
            validated_tuple_require_predicate_contract=bool(
                validated_schema_tuple_materialization_require_predicate_contract
            ),
            validated_tuple_require_terminal_slot=bool(
                validated_schema_tuple_materialization_require_terminal_slot
            ),
            validated_tuple_require_source_sentence=bool(
                validated_schema_tuple_materialization_require_source_sentence
            ),
            validated_tuple_min_token_savings=max(
                0,
                int(validated_schema_tuple_materialization_min_token_savings),
            ),
            validated_tuple_block_reader_visible_source=bool(
                validated_schema_tuple_materialization_block_reader_visible_source
            ),
            active_schema_bridge_tuple_executor=bool(
                active_schema_bridge_tuple_executor_enabled
            ),
            active_schema_bridge_tuple_min_score=max(
                0.0,
                float(active_schema_bridge_tuple_min_score),
            ),
            active_schema_bridge_tuple_min_bound_signals=max(
                1,
                int(active_schema_bridge_tuple_min_bound_signals),
            ),
        )
        for record in early_schema_records:
            metadata = dict(record.get("metadata") or {})
            metadata["early_schema_bridge_materialization"] = True
            metadata["protected_reader_visible"] = True
            metadata["materialization_stage"] = "post_retrieval_bridge_deficit"
            metadata["admission_min_score"] = max(
                0.0, float(early_schema_bridge_materialization_min_score)
            )
            metadata["admission_min_join_key_overlap"] = max(
                0.0,
                float(early_schema_bridge_materialization_min_join_key_overlap),
            )
            metadata["admission_min_relation_overlap"] = max(
                0.0,
                float(early_schema_bridge_materialization_min_relation_overlap),
            )
            metadata["operator_name"] = "early_schema_bridge_fact_join"
            record["metadata"] = metadata
            existing_doc_keys.add(_doc_key(str(record.get("doc") or "")))
            source_doc_key = str(metadata.get("source_doc_key") or "")
            if source_doc_key:
                existing_doc_keys.add(source_doc_key)
        early_schema_info.update({
            "bridge_allowed": True,
            "deficit_reasons": list(deficit_reasons),
            "materialization_stage": "post_retrieval_bridge_deficit",
            "relation_relaxation_config": {
                "enabled": bool(
                    early_schema_bridge_materialization_allow_relation_relaxation
                ),
                "exact_join_key_min_schema_idf": _round_float(
                    early_schema_bridge_materialization_exact_join_key_min_schema_idf
                ),
                "exact_join_key_min_relation_overlap": _round_float(
                    early_schema_bridge_materialization_exact_join_key_min_relation_overlap
                ),
            },
            "compact_projection": bool(
                early_schema_bridge_materialization_compact_projection
            ),
            "residual_deficit_signal": bool(residual_deficit_signal),
            "residual_deficit_reason": residual_deficit_reason,
            "residual_uncovered_slot_ids": sorted(residual_uncovered_slot_ids),
        })
    else:
        early_schema_records = []
        early_schema_info = {
            "enabled": bool(early_schema_bridge_materialization_enabled),
            "bridge_allowed": False,
            "reason": (
                "disabled"
                if not early_schema_bridge_materialization_enabled
                else "schema_materialization_disabled"
                if not schema_materialization_enabled
                else "no_schema_memory_or_doc_lookup"
                if not query_schema_memory or not get_doc_content
                else "long_context_blocked"
                if bool(state.get("long_context"))
                else "no_residual_deficit"
                if not early_residual_gate_passed
                else "no_bridge_deficit"
            ),
            "accepted_count": 0,
            "accepted_doc_ids": [],
            "validated_tuple_materialization_enabled": bool(
                validated_schema_tuple_materialization_enabled
            ),
            "validated_tuple_candidate_count": 0,
            "validated_tuple_accepted_count": 0,
            "validated_tuple_rejected_count": 0,
            "validated_tuple_reject_reason_counts": {},
            "validated_tuple_doc_ids": [],
            "validated_tuple_projection_token_savings": 0,
            "validated_tuple_require_terminal_slot": bool(
                validated_schema_tuple_materialization_require_terminal_slot
            ),
            "validated_tuple_require_source_sentence": bool(
                validated_schema_tuple_materialization_require_source_sentence
            ),
            "validated_tuple_min_token_savings": int(
                max(0, int(validated_schema_tuple_materialization_min_token_savings))
            ),
            "validated_tuple_block_reader_visible_source": bool(
                validated_schema_tuple_materialization_block_reader_visible_source
            ),
            "validated_tuple_terminal_candidate_count": 0,
            "validated_tuple_bridge_only_candidate_count": 0,
            "validated_tuple_source_sentence_bound_count": 0,
            "validated_tuple_source_sentence_trace_fallback_count": 0,
            "validated_tuple_source_visible_reject_count": 0,
            "covered_by_reader_visible_count": 0,
            "covered_by_reader_visible_doc_ids": [],
            "active_schema_bridge_tuple_executor_enabled": bool(
                active_schema_bridge_tuple_executor_enabled
            ),
            "active_schema_bridge_tuple_candidate_count": 0,
            "active_schema_bridge_tuple_accepted_count": 0,
            "active_schema_bridge_tuple_relaxed_count": 0,
            "active_schema_bridge_tuple_relaxed_reason_counts": {},
            "deficit_reasons": list(deficit_reasons),
            "materialization_stage": "post_retrieval_bridge_deficit",
            "residual_deficit_signal": bool(residual_deficit_signal),
            "residual_deficit_reason": residual_deficit_reason,
            "residual_uncovered_slot_ids": sorted(residual_uncovered_slot_ids),
        }

    schema_records, schema_materialization_info = _collect_query_schema_memory_candidates(
        question=question,
        query_schema_memory=query_schema_memory,
        get_doc_content=get_doc_content,
        existing_doc_keys=existing_doc_keys,
        state=state,
        lineage_context=lineage_context,
        enabled=bool(schema_materialization_enabled),
        max_candidates=max(0, int(schema_materialization_max_candidates)),
        max_original_rank=max(1, int(schema_materialization_max_original_rank)),
        min_score=max(0.0, float(schema_materialization_min_score)),
        min_schema_idf=max(0.0, float(schema_materialization_min_schema_idf)),
        min_query_overlap=max(0.0, float(schema_materialization_min_query_overlap)),
        min_join_key_overlap=max(0.0, float(schema_materialization_min_join_key_overlap)),
        min_relation_overlap=max(0.0, float(schema_materialization_min_relation_overlap)),
        require_answer_type_value=bool(schema_materialization_require_answer_type_value),
        require_entity_bound=bool(
            (query_schema_memory or {}).get("entity_bound_join_enabled", True)
        ),
        require_lineage_binding=bool(source_sufficiency_veto),
        cross_schema_max_endpoint_scan_rows=max(
            1,
            int(query_schema_memory_cross_schema_max_endpoint_scan_rows),
        ),
    )
    entity_predicate_records, entity_predicate_materialization_info = (
        _collect_entity_predicate_fact_join_candidates(
            entity_predicate_fact_join=entity_predicate_fact_join,
            get_doc_content=get_doc_content,
            existing_doc_keys=existing_doc_keys,
            state=state,
            lineage_context=lineage_context,
            enabled=bool(entity_predicate_materialization_enabled),
            max_candidates=max(0, int(entity_predicate_materialization_max_candidates)),
            min_score=max(0.0, float(entity_predicate_materialization_min_score)),
            min_relation_overlap=max(
                0.0,
                float(entity_predicate_materialization_min_relation_overlap),
            ),
            min_lineage_confidence=max(
                0.0,
                float(entity_predicate_materialization_min_lineage_confidence),
            ),
            require_atomic_slot=bool(
                entity_predicate_materialization_require_atomic_slot
            ),
            allowed_match_modes=(
                entity_predicate_materialization_allowed_match_modes
            ),
            compact_projection=bool(
                entity_predicate_materialization_compact_projection
            ),
            terminal_tuple_materialization=bool(
                entity_predicate_terminal_tuple_materialization
            ),
            terminal_tuple_min_lineage_depth=max(
                0, int(entity_predicate_terminal_tuple_min_lineage_depth)
            ),
            terminal_tuple_strict_branch_binding=bool(
                entity_predicate_terminal_tuple_strict_branch_binding
            ),
        )
    )
    generated_records = (
        (list(early_schema_records) if route_early_schema_through_optimizer else [])
        + list(shadow_records)
        + list(schema_records)
        + list(entity_predicate_records)
    )
    for record in generated_records:
        metadata = dict(record.get("metadata") or {})
        if str(record.get("source") or "") != "entity_predicate_fact_join":
            metadata.update(
                _candidate_lineage_metadata(
                    str(record.get("doc") or ""),
                    str(record.get("source") or "extra_evidence"),
                    lineage_context,
                )
            )
        record["metadata"] = metadata
    policy_existing_records = list(extra_records) if optimize_existing_extra_docs else []
    passthrough_extra_records = (
        [] if route_early_schema_through_optimizer else list(early_schema_records)
    ) + (
        [] if optimize_existing_extra_docs else list(extra_records)
    )
    candidate_records = policy_existing_records + generated_records
    deduplicated_candidate_records = []
    candidate_record_keys = set()
    for record in candidate_records:
        metadata = record.get("metadata") or {}
        source_doc_id = record.get("doc_id", metadata.get("source_doc_id"))
        if source_doc_id is not None:
            candidate_key = ("source_doc_id", str(source_doc_id))
        else:
            candidate_key = ("doc", _doc_key(str(record.get("doc") or "")))
        if candidate_key in candidate_record_keys:
            continue
        candidate_record_keys.add(candidate_key)
        deduplicated_candidate_records.append(record)
    info["candidate_record_deduplicated_count"] = max(
        0,
        len(candidate_records) - len(deduplicated_candidate_records),
    )
    candidate_records = deduplicated_candidate_records
    info["shadow_candidate_admission"] = shadow_info
    info["query_schema_memory_materialization"] = schema_materialization_info
    info["early_schema_bridge_materialization"] = early_schema_info
    info["active_schema_bridge_tuple_executor"] = {
        "enabled": bool(active_schema_bridge_tuple_executor_enabled),
        "min_score": _round_float(active_schema_bridge_tuple_min_score),
        "min_bound_signals": int(
            max(1, int(active_schema_bridge_tuple_min_bound_signals))
        ),
        "candidate_count": int(
            early_schema_info.get("active_schema_bridge_tuple_candidate_count", 0)
            or 0
        ),
        "accepted_count": int(
            early_schema_info.get("active_schema_bridge_tuple_accepted_count", 0)
            or 0
        ),
        "relaxed_count": int(
            early_schema_info.get("active_schema_bridge_tuple_relaxed_count", 0)
            or 0
        ),
        "relaxed_reason_counts": dict(
            early_schema_info.get(
                "active_schema_bridge_tuple_relaxed_reason_counts",
                {},
            )
            or {}
        ),
        "trace_fallback_count": int(
            early_schema_info.get(
                "validated_tuple_source_sentence_trace_fallback_count",
                0,
            )
            or 0
        ),
        "covered_by_reader_visible_count": int(
            early_schema_info.get("covered_by_reader_visible_count", 0) or 0
        ),
    }
    info["entity_predicate_fact_join_materialization"] = (
        entity_predicate_materialization_info
    )
    source_counts.update(
        str(record.get("source") or "extra_evidence")
        for record in early_schema_records
    )
    source_counts.update(
        str(record.get("source") or "extra_evidence")
        for record in generated_records
    )
    info["candidate_count"] = len(passthrough_extra_records) + len(candidate_records)
    info["shadow_candidate_count"] = len(shadow_records)
    info["schema_memory_candidate_count"] = len(schema_records)
    info["entity_predicate_fact_join_materialized_candidate_count"] = len(
        entity_predicate_records
    )
    info["source_counts"] = dict(source_counts)

    if snapshot_mode == "replay":
        snapshot = _load_candidate_pool_snapshot(
            candidate_pool_snapshot_dir,
            question,
            candidate_pool_snapshot_key,
        )
        if not snapshot:
            if candidate_pool_snapshot_strict:
                raise RuntimeError(
                    "candidate pool snapshot missing or invalid for strict replay: "
                    + _candidate_pool_snapshot_path(
                        candidate_pool_snapshot_dir,
                        question,
                        candidate_pool_snapshot_key,
                    )
                )
            info["candidate_pool_snapshot"]["reason"] = "missing_fallback_to_live_pool"
        else:
            base_docs = list(snapshot.get("base_docs") or [])
            extra_records = list(snapshot.get("extra_records") or [])
            passthrough_extra_records = list(snapshot.get("passthrough_extra_records") or [])
            candidate_records = list(snapshot.get("candidate_records") or [])
            shadow_records = list(snapshot.get("shadow_records") or [])
            early_schema_records = list(snapshot.get("early_schema_records") or [])
            schema_records = list(snapshot.get("schema_records") or [])
            entity_predicate_records = list(
                snapshot.get("entity_predicate_records") or []
            )
            state = dict(snapshot.get("state") or state)
            lineage_context = _ensure_comparison_branches(
                dict(snapshot.get("lineage_context") or lineage_context)
            )
            deficit_signal = bool(snapshot.get("deficit_signal", deficit_signal))
            deficit_reasons = list(snapshot.get("deficit_reasons") or deficit_reasons)
            shadow_info = dict(snapshot.get("shadow_info") or shadow_info)
            early_schema_info = dict(snapshot.get("early_schema_info") or early_schema_info)
            schema_materialization_info = dict(
                snapshot.get("schema_materialization_info") or schema_materialization_info
            )
            entity_predicate_materialization_info = dict(
                snapshot.get("entity_predicate_materialization_info")
                or entity_predicate_materialization_info
            )
            source_counts = Counter(snapshot.get("source_counts") or {})
            if not bool(entity_predicate_materialization_enabled):
                removed_entity_predicate_count = sum(
                    str(record.get("source") or "")
                    == "entity_predicate_fact_join"
                    for record in candidate_records
                )
                entity_predicate_records = []
                candidate_records = [
                    record
                    for record in candidate_records
                    if str(record.get("source") or "")
                    != "entity_predicate_fact_join"
                ]
                source_counts.pop("entity_predicate_fact_join", None)
                entity_predicate_materialization_info = {
                    **entity_predicate_materialization_info,
                    "enabled": False,
                    "reason": "disabled_in_fixed_pool_replay",
                    "accepted_count": 0,
                    "accepted_doc_ids": [],
                }
                info["candidate_pool_snapshot"].update(
                    {
                        "intervention": "remove_entity_predicate_fact_join",
                        "removed_candidate_count": int(
                            removed_entity_predicate_count
                        ),
                    }
                )
            portfolio_allow_shadow = bool(
                snapshot.get("portfolio_allow_shadow", portfolio_allow_shadow)
            )
            portfolio_override_allow = bool(
                snapshot.get("portfolio_override_allow", portfolio_override_allow)
            )
            calibrated_state_allows_shadow = bool(
                snapshot.get(
                    "calibrated_state_allows_shadow",
                    calibrated_state_allows_shadow,
                )
            )
            allow_shadow = bool(snapshot.get("allow_shadow", allow_shadow))
            rebound_candidate_records = []
            for record in candidate_records:
                record = dict(record)
                source = str(record.get("source") or "extra_evidence")
                metadata = dict(record.get("metadata") or {})
                if source == "entity_predicate_fact_join":
                    if entity_predicate_materialization_compact_projection:
                        record = _project_entity_predicate_record(record)
                        metadata = dict(record.get("metadata") or {})
                    target_slot_id = str(
                        metadata.get("entity_predicate_target_slot_id") or ""
                    )
                    slot_lookup = {
                        str(slot.get("slot_id") or ""): slot
                        for slot in lineage_context.get("required_slots") or []
                    }
                    target_slot = slot_lookup.get(target_slot_id) or {}
                    branch_id = str(target_slot.get("branch_id") or "")
                    metadata["covered_branch_ids"] = [branch_id] if branch_id else []
                    metadata["comparison_branch_id"] = branch_id
                else:
                    metadata.update(
                        _candidate_lineage_metadata(
                            str(record.get("doc") or ""),
                            source,
                            lineage_context,
                        )
                    )
                record["metadata"] = metadata
                rebound_candidate_records.append(record)
            candidate_records = rebound_candidate_records
            if entity_predicate_materialization_compact_projection:
                entity_predicate_records = [
                    _project_entity_predicate_record(dict(record))
                    for record in entity_predicate_records
                ]
                projected = [
                    record
                    for record in candidate_records
                    if bool((record.get("metadata") or {}).get("compact_projection"))
                ]
                entity_predicate_materialization_info.update(
                    {
                        "compact_projection_enabled": True,
                        "projected_candidate_count": len(projected),
                        "projection_token_savings": sum(
                            int(
                                (record.get("metadata") or {}).get(
                                    "projection_token_savings",
                                    0,
                                )
                                or 0
                            )
                            for record in projected
                        ),
                    }
                )
            docs = base_docs + [
                str(record.get("doc") or "") for record in extra_records
            ]
            base_token_sets = [_tokens(doc) for doc in base_docs]
            missing_tokens = set(state.get("missing_tokens") or set())
            info["candidate_pool_snapshot"].update(
                {
                    "loaded": True,
                    "path": str(snapshot.get("snapshot_path") or ""),
                    "reason": "materialized_pool_replayed",
                }
            )
    elif snapshot_mode == "record":
        snapshot_state = dict(state)
        snapshot_state["missing_tokens"] = sorted(state.get("missing_tokens") or [])
        snapshot_path = _write_candidate_pool_snapshot(
            candidate_pool_snapshot_dir,
            question,
            candidate_pool_snapshot_key,
            {
                "base_docs": base_docs,
                "extra_records": extra_records,
                "passthrough_extra_records": passthrough_extra_records,
                "candidate_records": candidate_records,
                "shadow_records": shadow_records,
                "early_schema_records": early_schema_records,
                "schema_records": schema_records,
                "entity_predicate_records": entity_predicate_records,
                "state": snapshot_state,
                "lineage_context": lineage_context,
                "deficit_signal": bool(deficit_signal),
                "deficit_reasons": list(deficit_reasons),
                "shadow_info": shadow_info,
                "early_schema_info": early_schema_info,
                "schema_materialization_info": schema_materialization_info,
                "entity_predicate_materialization_info": (
                    entity_predicate_materialization_info
                ),
                "source_counts": dict(source_counts),
                "portfolio_allow_shadow": bool(portfolio_allow_shadow),
                "portfolio_override_allow": bool(portfolio_override_allow),
                "calibrated_state_allows_shadow": bool(calibrated_state_allows_shadow),
                "allow_shadow": bool(allow_shadow),
            },
        )
        info["candidate_pool_snapshot"].update(
            {
                "written": True,
                "path": snapshot_path,
                "reason": "materialized_pool_recorded",
            }
        )

    info["base_doc_count"] = len(base_docs)
    info["candidate_count"] = len(passthrough_extra_records) + len(candidate_records)
    info["shadow_candidate_count"] = len(shadow_records)
    info["schema_memory_candidate_count"] = len(schema_records)
    info["entity_predicate_fact_join_materialized_candidate_count"] = len(
        entity_predicate_records
    )
    info["source_counts"] = dict(source_counts)
    info["planned_reason"] = state.get("reason")
    info["question_type"] = state.get("question_type")
    info["evidence_insufficient"] = bool(state.get("evidence_insufficient"))
    info["answer_slot"] = str(lineage_context.get("answer_slot") or "unknown")
    info["required_slots"] = list(lineage_context.get("required_slots") or [])
    info["required_slot_count"] = len(lineage_context.get("required_slot_ids") or [])
    info["base_covered_slot_ids"] = list(
        lineage_context.get("base_covered_slot_ids") or []
    )
    info["required_uncovered_slot_ids"] = list(
        lineage_context.get("required_uncovered_slot_ids") or []
    )
    info["deficit_signal"] = bool(deficit_signal)
    info["deficit_reasons"] = list(deficit_reasons)
    info["shadow_candidate_admission"] = shadow_info
    info["early_schema_bridge_materialization"] = early_schema_info
    info["query_schema_memory_materialization"] = schema_materialization_info
    info["active_schema_bridge_tuple_executor"] = {
        "enabled": bool(active_schema_bridge_tuple_executor_enabled),
        "min_score": _round_float(active_schema_bridge_tuple_min_score),
        "min_bound_signals": int(
            max(1, int(active_schema_bridge_tuple_min_bound_signals))
        ),
        "candidate_count": int(
            early_schema_info.get("active_schema_bridge_tuple_candidate_count", 0)
            or 0
        ),
        "accepted_count": int(
            early_schema_info.get("active_schema_bridge_tuple_accepted_count", 0)
            or 0
        ),
        "relaxed_count": int(
            early_schema_info.get("active_schema_bridge_tuple_relaxed_count", 0)
            or 0
        ),
        "relaxed_reason_counts": dict(
            early_schema_info.get(
                "active_schema_bridge_tuple_relaxed_reason_counts",
                {},
            )
            or {}
        ),
        "trace_fallback_count": int(
            early_schema_info.get(
                "validated_tuple_source_sentence_trace_fallback_count",
                0,
            )
            or 0
        ),
        "covered_by_reader_visible_count": int(
            early_schema_info.get("covered_by_reader_visible_count", 0) or 0
        ),
    }
    info["entity_predicate_fact_join_materialization"] = (
        entity_predicate_materialization_info
    )

    if not passthrough_extra_records and not candidate_records:
        info["reason"] = "no_extra_docs"
        return docs, info

    optimize_requested = policy == "always" or bool(deficit_signal) or bool(calibrated_state_allows_shadow)
    if policy == "high_risk_only" and not optimize_requested:
        info["reason"] = "pass_through_low_risk"
        info["accepted_count"] = len(extra_docs)
        info["accepted_titles"] = [_title(doc) for doc in extra_docs]
        info["accepted_source_counts"] = dict(source_counts)
        info["final_doc_count"] = len(docs)
        return docs, info

    if not candidate_records:
        blocked_reason = str(shadow_info.get("reason") or "")
        passthrough_docs = [str(record.get("doc") or "") for record in passthrough_extra_records]
        final_docs = base_docs + passthrough_docs
        passthrough_sources = Counter(
            str(record.get("source") or "extra_evidence")
            for record in passthrough_extra_records
        )
        early_schema_count = sum(
            str(record.get("source") or "") == "query_schema_memory"
            and bool((record.get("metadata") or {}).get("early_schema_bridge_materialization"))
            for record in passthrough_extra_records
        )
        info["reason"] = (
            "early_schema_bridge_materialized"
            if early_schema_count
            else "pass_through_no_shadow_candidates"
            if allow_shadow
            else "pass_through_no_deficit"
            if blocked_reason == "blocked_no_deficit"
            else f"pass_through_{blocked_reason or 'no_shadow_allowed'}"
        )
        info["applied"] = bool(early_schema_count)
        info["accepted_count"] = len(passthrough_extra_records)
        info["accepted_titles"] = [_title(record.get("doc")) for record in passthrough_extra_records]
        info["accepted_source_counts"] = dict(passthrough_sources)
        info["accepted_schema_memory_count"] = int(early_schema_count)
        info["early_schema_bridge_materialized_count"] = int(early_schema_count)
        info["final_doc_count"] = len(final_docs)
        return final_docs, info

    scored = []
    slate_source_counts = Counter(
        str(record.get("source") or "extra_evidence") for record in candidate_records
    )
    slate_token_total = sum(
        max(1, len(re.findall(r"\S+", str(record.get("doc") or ""))))
        for record in candidate_records
    )
    pre_scored = []
    for idx, record in enumerate(candidate_records):
        doc = str(record.get("doc") or "")
        source = str(record.get("source") or "extra_evidence")
        metadata = dict(record.get("metadata") or {})
        score = _score_extra_doc(
            question_tokens=question_tokens,
            missing_tokens=missing_tokens,
            base_token_sets=base_token_sets,
            doc=doc,
            source=source,
            token_budget=max(1, int(token_budget)),
            min_query_overlap=float(min_query_overlap),
            token_weight=float(token_weight),
            compute_weight=float(compute_weight),
            risk_weight=float(risk_weight),
            redundancy_penalty=float(redundancy_penalty),
            candidate_count=len(candidate_records),
            planned_state=state,
            candidate_metadata=metadata,
        )
        score["input_extra_rank"] = (
            idx if optimize_existing_extra_docs else len(extra_records) + idx
        )
        score["doc"] = doc
        score["slate_size"] = len(candidate_records)
        score["slate_rank"] = idx
        score["slate_token_share"] = _round_float(
            float(score.get("token_count", 0.0)) / float(max(1, slate_token_total))
        )
        score["slate_source_count"] = int(slate_source_counts.get(source, 0))
        pre_scored.append(score)
    annotate_slate_relative_features(pre_scored)

    for record, preliminary_score in zip(candidate_records, pre_scored):
        score = dict(preliminary_score)
        source = str(score.get("source") or record.get("source") or "extra_evidence")
        if use_learned_policy and effective_admission_policy == "counterfactual_slate_veto":
            learned_score = score_candidate_with_counterfactual_slate_model(
                score,
                learned_model,
                state=state,
                min_support=int(learned_cost_optimizer_min_support),
                min_harm_signals=int(counterfactual_slate_min_harm_signals),
                min_stable_datasets=int(counterfactual_slate_min_stable_datasets),
                min_effect=float(learned_cost_optimizer_min_expected_gain),
            )
            score["learned_cost"] = learned_score
            score["learned_expected_gain"] = _round_float(
                learned_score.get("expected_gain", 0.0)
            )
            score["learned_expected_utility"] = score["learned_expected_gain"]
            score["learned_p_improve"] = _round_float(
                1.0 - float(learned_score.get("p_degrade", 0.0))
            )
            score["learned_p_degrade"] = _round_float(
                learned_score.get("p_degrade", 0.0)
            )
            score["learned_confidence"] = _round_float(
                learned_score.get("confidence", 0.0)
            )
            score["learned_harm_expected_gain"] = score["learned_expected_gain"]
            score["learned_harm_p_improve"] = score["learned_p_improve"]
            score["learned_harm_p_degrade"] = score["learned_p_degrade"]
            score["learned_harm_margin"] = _round_float(
                float(score["learned_harm_p_degrade"])
                - float(score["learned_harm_p_improve"])
            )
            score["learned_harm_confidence"] = score["learned_confidence"]
            score["learned_harm_support"] = _round_float(
                learned_score.get("support", 0.0)
            )
            score["learned_harm_feature_count"] = int(
                learned_score.get("harm_signal_count", 0) or 0
            )
            top_harm_features = list(learned_score.get("top_features") or [])
            score["learned_harm_primary_feature"] = str(
                (top_harm_features[0] if top_harm_features else {}).get("feature") or ""
            )
            score["counterfactual_quality_upper_ci"] = _round_float(
                learned_score.get("quality_upper_ci", 0.0)
            )
            score["counterfactual_stable_dataset_count"] = int(
                learned_score.get("stable_dataset_count", 0) or 0
            )
            score["counterfactual_harm_family_count"] = int(
                learned_score.get("harm_family_count", 0) or 0
            )
            score["counterfactual_coverage_protected"] = bool(
                learned_score.get("coverage_protected")
            )
            score["counterfactual_net_drop_utility"] = _round_float(
                learned_score.get("net_drop_utility", 0.0)
            )
            conditional_decision = (
                str(learned_score.get("decision_type") or "")
                == "conditional_cost_effect"
            )
            if conditional_decision:
                learned_harm_veto = bool(
                    learned_score.get("veto_recommended")
                    and float(score["learned_harm_confidence"])
                    >= float(learned_cost_optimizer_min_confidence)
                    and not bool(score["counterfactual_coverage_protected"])
                )
            else:
                learned_harm_veto = bool(
                    learned_score.get("veto_recommended")
                    and float(score["learned_harm_confidence"])
                    >= float(learned_cost_optimizer_min_confidence)
                    and float(score["counterfactual_quality_upper_ci"])
                    <= -float(learned_cost_optimizer_min_expected_gain)
                    and int(score["counterfactual_harm_family_count"])
                    >= max(2, int(counterfactual_slate_min_harm_signals))
                    and int(score["counterfactual_stable_dataset_count"])
                    >= int(counterfactual_slate_min_stable_datasets)
                    and not bool(score["counterfactual_coverage_protected"])
                )
            score["learned_harm_veto"] = learned_harm_veto
            score["learned_decision"] = "veto" if learned_harm_veto else "admit"
            score["admission_sort_utility"] = float(score["net_utility"])
        elif use_learned_policy:
            learned_score = score_candidate_with_learned_cost_model(
                score,
                learned_model,
                state=state,
                min_support=int(learned_cost_optimizer_min_support),
            )
            score["learned_cost"] = learned_score
            score["learned_expected_gain"] = _round_float(learned_score.get("expected_gain", 0.0))
            score["learned_expected_utility"] = _round_float(
                learned_score.get("expected_utility", learned_score.get("expected_gain", 0.0))
            )
            score["learned_p_improve"] = _round_float(learned_score.get("p_improve", 0.0))
            score["learned_p_degrade"] = _round_float(learned_score.get("p_degrade", 1.0))
            score["learned_confidence"] = _round_float(learned_score.get("confidence", 0.0))
            score["learned_harm_expected_gain"] = _round_float(
                learned_score.get("harm_expected_gain", learned_score.get("expected_gain", 0.0))
            )
            score["learned_harm_p_improve"] = _round_float(
                learned_score.get("harm_p_improve", learned_score.get("p_improve", 0.0))
            )
            score["learned_harm_p_degrade"] = _round_float(
                learned_score.get("harm_p_degrade", learned_score.get("p_degrade", 1.0))
            )
            score["learned_harm_margin"] = _round_float(
                learned_score.get("harm_margin", 0.0)
            )
            score["learned_harm_confidence"] = _round_float(
                learned_score.get("harm_confidence", learned_score.get("confidence", 0.0))
            )
            score["learned_harm_support"] = _round_float(
                learned_score.get("harm_support", 0.0)
            )
            score["learned_harm_feature_count"] = int(
                learned_score.get("harm_feature_count", 0) or 0
            )
            score["learned_harm_primary_feature"] = str(
                learned_score.get("harm_primary_feature") or ""
            )
            learned_available = bool(learned_score.get("enabled")) and int(
                learned_score.get("matched_feature_count") or 0
            ) > 0
            learned_accept = bool(
                learned_available
                and float(score["learned_expected_utility"]) >= float(learned_cost_optimizer_min_expected_gain)
                and float(score["learned_confidence"]) >= float(learned_cost_optimizer_min_confidence)
                and float(score["learned_p_degrade"]) <= float(learned_cost_optimizer_max_degrade)
            )
            learned_reject = bool(
                learned_available
                and float(score["learned_confidence"]) >= float(learned_cost_optimizer_min_confidence)
                and (
                    float(score["learned_expected_utility"]) < -float(learned_cost_optimizer_min_expected_gain)
                    or float(score["learned_p_degrade"]) > float(learned_cost_optimizer_max_degrade)
                )
                and not learned_accept
            )
            learned_harm_veto = bool(
                effective_admission_policy == "learned_veto"
                and learned_available
                and int(score["learned_harm_feature_count"]) > 0
                and float(score["learned_harm_confidence"])
                >= float(learned_cost_optimizer_min_confidence)
                and float(score["learned_harm_expected_gain"])
                <= -float(learned_cost_optimizer_min_expected_gain)
                and float(score["learned_harm_margin"])
                >= float(learned_cost_optimizer_veto_min_degrade_margin)
            )
            score["learned_harm_veto"] = learned_harm_veto
            if effective_admission_policy == "learned_veto":
                score["learned_decision"] = "veto" if learned_harm_veto else "admit"
                score["admission_sort_utility"] = float(score["net_utility"])
            else:
                score["learned_decision"] = (
                    "accept"
                    if learned_accept
                    else "reject"
                    if learned_reject
                    else str(learned_score.get("reason") or "abstain")
                )
                score["admission_sort_utility"] = (
                    float(score["learned_expected_utility"])
                    if learned_available
                    else float(score["net_utility"])
                )
        else:
            score["admission_sort_utility"] = float(score["net_utility"])
            score["learned_decision"] = "not_used"
            score["learned_harm_veto"] = False
        score["candidate_id"] = _candidate_id(score)
        score["coverage_rescue_eligible"] = bool(
            score.get("uncovered_slot_ids")
            and source
            in (
                _PROTECTED_EXTRA_SOURCES
                | {"query_schema_memory", "entity_predicate_fact_join"}
            )
            and float(score.get("lineage_confidence", 0.0)) >= 0.34
        )
        scored.append(score)

    info["candidate_pool_fingerprint"] = _candidate_pool_fingerprint(scored)
    info["candidate_pool_token_count"] = sum(
        max(0, int(row.get("token_count", 0) or 0)) for row in scored
    )

    accepted = []
    rejected = []
    reject_reasons = Counter()
    max_extra_docs = max(0, int(max_extra_docs))
    candidate_budget = max_extra_docs
    admitted_candidate_tokens = 0
    remaining_required_slots = set(
        lineage_context.get("required_uncovered_slot_ids") or []
    )
    admitted_slot_ids = set(lineage_context.get("base_covered_slot_ids") or [])
    coverage_rescued_ids = set()
    replacement_info = info["deficit_aware_replacement"]

    def recompute_admission_state() -> Tuple[int, set, set]:
        tokens = sum(
            max(0, int(row.get("token_count", 0) or 0)) for row in accepted
        )
        covered = set(lineage_context.get("base_covered_slot_ids") or [])
        for row in accepted:
            covered.update(row.get("covered_slot_ids") or [])
            covered.update(row.get("uncovered_slot_ids") or [])
        required = set(lineage_context.get("required_uncovered_slot_ids") or [])
        return tokens, covered, required - covered

    for item in _sort_candidates_for_policy(scored, effective_admission_policy):
        reasons = []
        learned_accept = str(item.get("learned_decision") or "") == "accept"
        learned_reject = str(item.get("learned_decision") or "") == "reject"
        learned_harm_veto = bool(item.get("learned_harm_veto"))
        always_on = effective_admission_policy in {
            "always_on",
            "learned_veto",
            "counterfactual_drop",
            "counterfactual_slate_veto",
        }
        newly_covered_slots = remaining_required_slots & set(
            item.get("uncovered_slot_ids") or []
        )
        coverage_rescue = bool(
            effective_admission_policy in {
                "rule",
                "learned",
                "learned_veto",
                "counterfactual_slate_veto",
            }
            and newly_covered_slots
            and item.get("coverage_rescue_eligible")
            and float(item.get("risk", 0.0)) <= float(max_risk)
            and float(item.get("net_utility", 0.0)) >= min(
                float(min_net_utility),
                0.02,
            )
        )
        if not fact_join_plan_optimizer_enabled and len(accepted) >= candidate_budget:
            reasons.append("admission_budget_full")
        candidate_tokens = max(0, int(item.get("token_count", 0) or 0))
        if (
            not fact_join_plan_optimizer_enabled
            and admitted_candidate_tokens + candidate_tokens > max(1, int(token_budget))
        ):
            reasons.append("token_budget_exceeded")
        if learned_harm_veto and not coverage_rescue:
            reasons.append("learned_harm_veto")
        if learned_reject and not always_on and not coverage_rescue:
            reasons.append("learned_expected_utility_low_or_degrade_risk")
        if (
            float(item["utility"]) < float(min_utility)
            and not fact_join_plan_optimizer_enabled
            and not learned_accept
            and not always_on
            and not coverage_rescue
        ):
            reasons.append("utility_below_threshold")
        if float(item["risk"]) > float(max_risk) and not always_on:
            reasons.append("risk_above_threshold")
        if (
            float(item["net_utility"]) < float(min_net_utility)
            and not fact_join_plan_optimizer_enabled
            and not learned_accept
            and not always_on
            and not coverage_rescue
        ):
            reasons.append("net_utility_below_threshold")
        if item["source"] == "typed_fact_join_shadow":
            safety_ok, safety_reasons = _shadow_plan_relation_safe(
                item,
                question=question,
                state=state,
                calibrated_shadow_admission=bool(calibrated_shadow_admission),
            )
            if not safety_ok:
                reasons.extend(f"shadow_counterfactual_{reason}" for reason in safety_reasons)
            calibrated_structural_support = (
                bool(calibrated_shadow_admission)
                and float(item.get("endpoint_overlap", 0.0)) >= 0.50
                and float(item.get("schema_selectivity", 0.0)) >= 0.58
                and (
                    float(item.get("relation_overlap", 0.0)) >= 0.08
                    or (
                        float(item.get("provenance_score", 0.0)) >= 0.425
                        and float(item.get("endpoint_overlap", 0.0)) >= 0.95
                        and int(item.get("best_rank", 0) or 0) <= 2
                        and int(item.get("worst_rank", 0) or 0) <= 12
                    )
                )
            )
            if (
                float(item.get("plan_utility", 0.0)) < max(float(shadow_min_plan_utility), 0.50)
                and not learned_accept
                and not always_on
            ):
                reasons.append("shadow_plan_utility_below_calibrated_min")
            if (
                float(item.get("plan_net_utility", 0.0)) < max(float(shadow_min_plan_net_utility), 0.08)
                and not learned_accept
                and not always_on
            ):
                reasons.append("shadow_plan_net_below_calibrated_min")
            if (
                float(item.get("plan_risk", 0.0)) > min(float(shadow_max_plan_risk), 0.52)
                and not always_on
            ):
                reasons.append("shadow_plan_risk_above_calibrated_max")
            if str(item.get("plan_type") or "") == "two_hop_fact_join":
                if float(item.get("endpoint_overlap", 0.0)) < 0.24:
                    reasons.append("shadow_endpoint_overlap_low")
                if float(item.get("relation_overlap", 0.0)) < 0.08 and not calibrated_structural_support:
                    reasons.append("shadow_relation_overlap_low")
            if float(item.get("schema_selectivity", 0.0)) < 0.45:
                reasons.append("shadow_schema_selectivity_low")
            if (
                int(item.get("best_rank", 0) or 0) > 40
                and float(item.get("provenance_score", 0.0)) < 0.12
                and not always_on
            ):
                reasons.append("shadow_late_rank_low_provenance")
            if bool(portfolio_override_allow and not portfolio_allow_shadow) and not always_on:
                if float(item.get("plan_utility", 0.0)) < 0.60:
                    reasons.append("portfolio_override_plan_utility_low")
                if float(item.get("plan_net_utility", 0.0)) < 0.18:
                    reasons.append("portfolio_override_plan_net_low")
                if float(item.get("plan_risk", 0.0)) > 0.50:
                    reasons.append("portfolio_override_plan_risk_high")
                if float(item.get("endpoint_overlap", 0.0)) < 0.42:
                    reasons.append("portfolio_override_endpoint_low")
                if float(item.get("schema_selectivity", 0.0)) < 0.58:
                    reasons.append("portfolio_override_schema_low")
                if (
                    float(item.get("relation_overlap", 0.0)) < 0.08
                    and not (
                        float(item.get("provenance_score", 0.0)) >= 0.425
                        and float(item.get("endpoint_overlap", 0.0)) >= 0.95
                        and int(item.get("best_rank", 0) or 0) <= 2
                        and int(item.get("worst_rank", 0) or 0) <= 12
                    )
                ):
                    reasons.append("portfolio_override_relation_or_provenance_low")
        if item["source"] == "query_schema_memory":
            item_min_join_key_overlap = float(
                item.get("admission_min_join_key_overlap", 0.0)
                or schema_materialization_min_join_key_overlap
            )
            item_min_relation_overlap = float(
                item.get("admission_min_relation_overlap", 0.0)
                or schema_materialization_min_relation_overlap
            )
            item_min_schema_score = float(
                item.get("admission_min_score", 0.0)
                or schema_materialization_min_score
            )
            if float(item.get("join_key_overlap", 0.0) or 0.0) < float(
                item_min_join_key_overlap
            ):
                reasons.append("schema_materialization_join_key_low")
            if float(item.get("relation_overlap", 0.0) or 0.0) < float(
                item_min_relation_overlap
            ):
                reasons.append("schema_materialization_relation_low")
            if float(item.get("schema_score", 0.0) or 0.0) < float(
                item_min_schema_score
            ):
                reasons.append("schema_materialization_score_low")
        if item["source"] == "entity_predicate_fact_join":
            direct_slot_grounded = bool(item.get("direct_slot_grounded"))
            if not bool(item.get("lineage_bound")) and not direct_slot_grounded:
                reasons.append("entity_predicate_materialization_lineage_unbound")
            if item.get("dependency_binding_safe") is False:
                reasons.append("entity_predicate_materialization_dependency_unsafe")
            if not bool(item.get("predicate_contract_match")):
                reasons.append("entity_predicate_materialization_predicate_contract_mismatch")
            if float(item.get("entity_predicate_score", 0.0) or 0.0) < float(
                entity_predicate_materialization_min_score
            ):
                reasons.append("entity_predicate_materialization_score_low")
            if float(item.get("relation_overlap", 0.0) or 0.0) < float(
                entity_predicate_materialization_min_relation_overlap
            ):
                reasons.append("entity_predicate_materialization_relation_low")
            if (
                float(item.get("join_key_overlap", 0.0) or 0.0) < 1.0
                and not direct_slot_grounded
            ):
                reasons.append("entity_predicate_materialization_join_not_exact")
            if not bool(item.get("answer_type_match", True)):
                reasons.append("entity_predicate_materialization_answer_type_mismatch")
            if (
                lineage_context.get("required_uncovered_slot_ids")
                and not item.get("uncovered_slot_ids")
            ):
                reasons.append("entity_predicate_materialization_no_slot_binding")
            if (
                lineage_context.get("required_uncovered_slot_ids")
                and float(item.get("coverage_gain", 0.0) or 0.0) <= 0.0
            ):
                reasons.append("entity_predicate_materialization_no_slot_gain")
            if (
                lineage_context.get("required_uncovered_slot_ids")
                and float(item.get("lineage_confidence", 0.0) or 0.0)
                < float(entity_predicate_materialization_min_lineage_confidence)
            ):
                reasons.append("entity_predicate_materialization_lineage_low")
        if (
            float(item["query_overlap"]) < float(min_query_overlap)
            and float(item["missing_gain"]) <= 0.0
            and item["source"] not in {"relation_fact_table"}
            and not learned_accept
            and not always_on
            and not coverage_rescue
        ):
            reasons.append("low_query_and_missing_overlap")
        if newly_covered_slots and any(
            reason in {"admission_budget_full", "token_budget_exceeded"}
            for reason in reasons
        ):
            reasons.append("required_slot_budget_blocked")
        budget_reasons = {
            "admission_budget_full",
            "token_budget_exceeded",
            "required_slot_budget_blocked",
        }
        non_budget_reasons = [reason for reason in reasons if reason not in budget_reasons]
        replacement_safe = bool(
            not fact_join_plan_optimizer_enabled
            and
            deficit_aware_replacement_enabled
            and str(item.get("source") or "") == "entity_predicate_fact_join"
            and newly_covered_slots
            and not non_budget_reasons
            and bool(set(reasons) & budget_reasons)
            and bool(item.get("lineage_bound"))
            and item.get("dependency_binding_safe") is not False
            and bool(item.get("predicate_contract_match"))
            and str(item.get("predicate_match_mode") or "")
            in {"direct_token", "polarity_concept"}
            and bool(item.get("slot_materialization_safe"))
            and bool(item.get("answer_type_match", True))
            and float(item.get("risk", 0.0)) <= float(max_risk)
        )
        if replacement_safe:
            replacement_info["attempt_count"] += 1
            candidate_slots = set(item.get("covered_slot_ids") or []) | set(
                item.get("uncovered_slot_ids") or []
            )
            candidate_branches = set(item.get("covered_branch_ids") or [])
            victim_rows = []
            for victim in accepted:
                other_covered = set(lineage_context.get("base_covered_slot_ids") or [])
                other_branches = set()
                for other in accepted:
                    if other is victim:
                        continue
                    other_covered.update(other.get("covered_slot_ids") or [])
                    other_covered.update(other.get("uncovered_slot_ids") or [])
                    other_branches.update(other.get("covered_branch_ids") or [])
                victim_slots = set(victim.get("covered_slot_ids") or []) | set(
                    victim.get("uncovered_slot_ids") or []
                )
                if victim_slots - candidate_slots - other_covered:
                    replacement_info["slot_loss_veto_count"] += 1
                    continue
                victim_branches = set(victim.get("covered_branch_ids") or [])
                if (
                    bool(lineage_context.get("comparison_query"))
                    and victim_branches - candidate_branches - other_branches
                ):
                    replacement_info["branch_loss_veto_count"] += 1
                    continue
                victim_priority = _deficit_replacement_priority(
                    victim,
                    victim_slots - other_covered,
                )
                victim_rows.append((victim_priority, victim))
            victim_rows.sort(
                key=lambda pair: (
                    float(pair[0]),
                    int(pair[1].get("token_count", 0) or 0),
                )
            )
            selected_victims = []
            candidate_priority = _deficit_replacement_priority(
                item,
                newly_covered_slots,
            )
            for victim_priority, victim in victim_rows:
                if len(selected_victims) >= max(
                    0,
                    int(deficit_aware_replacement_max_evictions),
                ):
                    break
                if candidate_priority < (
                    float(victim_priority)
                    + float(deficit_aware_replacement_min_margin)
                ):
                    continue
                selected_victims.append((victim_priority, victim))
                retained = [
                    row
                    for row in accepted
                    if all(row is not selected for _, selected in selected_victims)
                ]
                retained_tokens = sum(
                    max(0, int(row.get("token_count", 0) or 0))
                    for row in retained
                )
                if (
                    len(retained) < candidate_budget
                    and retained_tokens + candidate_tokens
                    <= max(1, int(token_budget))
                ):
                    break
            if selected_victims:
                selected_ids = {id(victim) for _, victim in selected_victims}
                retained = [row for row in accepted if id(row) not in selected_ids]
                retained_tokens = sum(
                    max(0, int(row.get("token_count", 0) or 0))
                    for row in retained
                )
                replacement_fits = bool(
                    len(retained) < candidate_budget
                    and retained_tokens + candidate_tokens
                    <= max(1, int(token_budget))
                )
                if replacement_fits:
                    accepted[:] = retained
                    for victim_priority, victim in selected_victims:
                        victim = dict(victim)
                        victim["reject_reasons"] = ["replaced_by_deficit_join"]
                        rejected.append(victim)
                        reject_reasons["replaced_by_deficit_join"] += 1
                        coverage_rescued_ids.discard(
                            str(victim.get("candidate_id") or "")
                        )
                        replacement_info["pairs"].append(
                            {
                                "admitted_candidate_id": str(
                                    item.get("candidate_id") or ""
                                ),
                                "admitted_title": str(item.get("title") or ""),
                                "admitted_source": str(item.get("source") or ""),
                                "admitted_priority": _round_float(candidate_priority),
                                "admitted_tokens": int(candidate_tokens),
                                "covered_slot_ids": sorted(newly_covered_slots),
                                "evicted_candidate_id": str(
                                    victim.get("candidate_id") or ""
                                ),
                                "evicted_title": str(victim.get("title") or ""),
                                "evicted_source": str(victim.get("source") or ""),
                                "evicted_priority": _round_float(victim_priority),
                                "evicted_tokens": int(
                                    victim.get("token_count", 0) or 0
                                ),
                                "evicted_doc": str(victim.get("doc") or ""),
                            }
                        )
                    item = dict(item)
                    item["accepted_by_deficit_replacement"] = True
                    item["replaced_candidate_ids"] = [
                        str(victim.get("candidate_id") or "")
                        for _, victim in selected_victims
                    ]
                    accepted.append(item)
                    replacement_info["replacement_count"] += 1
                    coverage_rescued_ids.add(str(item.get("candidate_id") or ""))
                    (
                        admitted_candidate_tokens,
                        admitted_slot_ids,
                        remaining_required_slots,
                    ) = recompute_admission_state()
                    continue
        if reasons:
            item = dict(item)
            item["reject_reasons"] = reasons
            rejected.append(item)
            for reason in reasons:
                reject_reasons[reason] += 1
            continue
        accepted.append(item)
        admitted_candidate_tokens += candidate_tokens
        admitted_slot_ids.update(item.get("covered_slot_ids") or [])
        admitted_slot_ids.update(item.get("uncovered_slot_ids") or [])
        remaining_required_slots.difference_update(newly_covered_slots)
        if coverage_rescue:
            coverage_rescued_ids.add(str(item.get("candidate_id") or ""))

    if fact_join_plan_optimizer_enabled:
        locally_eligible = list(accepted)
        plan_result = optimize_cost_based_fact_join_plan(
            locally_eligible,
            required_slot_ids=lineage_context.get("required_uncovered_slot_ids") or [],
            max_candidates=candidate_budget,
            token_budget=max(1, int(token_budget)),
            compute_budget=max(0.0, float(fact_join_plan_compute_budget)),
            token_weight=float(token_weight),
            compute_weight=float(compute_weight),
            risk_weight=float(risk_weight),
            redundancy_weight=float(redundancy_penalty),
            min_objective=float(fact_join_plan_min_objective),
            max_enumerated_candidates=max(
                1, int(fact_join_plan_max_enumerated_candidates)
            ),
        )
        selected_plan_ids = set(plan_result.get("selected_candidate_ids") or [])
        accepted = [
            row
            for row in locally_eligible
            if str(row.get("candidate_id") or "") in selected_plan_ids
        ]
        for row in locally_eligible:
            candidate_id = str(row.get("candidate_id") or "")
            if candidate_id in selected_plan_ids:
                continue
            rejected_row = dict(row)
            rejected_row["reject_reasons"] = ["not_selected_by_cost_plan"]
            rejected.append(rejected_row)
            reject_reasons["not_selected_by_cost_plan"] += 1
            coverage_rescued_ids.discard(candidate_id)
        (
            admitted_candidate_tokens,
            admitted_slot_ids,
            remaining_required_slots,
        ) = recompute_admission_state()
        plan_result["locally_eligible_candidate_count"] = len(locally_eligible)
        plan_result["locally_rejected_candidate_count"] = max(
            0, len(scored) - len(locally_eligible)
        )
        info["fact_join_plan_optimizer"] = plan_result

    if (
        not fact_join_plan_optimizer_enabled
        and not accepted
        and preserve_best_on_empty
        and scored
        and candidate_budget > 0
        and deficit_signal
    ):
        best = max(
            scored,
            key=lambda row: (
                float(row.get("net_utility", 0.0)),
                float(row.get("utility", 0.0)),
                -float(row.get("risk", 0.0)),
            ),
        )
        if (
            not bool(best.get("learned_harm_veto"))
            and float(best.get("utility", 0.0)) >= max(0.52, float(min_utility))
            and float(best.get("net_utility", 0.0)) >= max(0.08, float(min_net_utility))
            and float(best.get("risk", 0.0)) <= min(0.52, float(max_risk))
            and max(0, int(best.get("token_count", 0) or 0)) <= max(1, int(token_budget))
        ):
            accepted.append(best)
            rejected = [row for row in rejected if row.get("input_extra_rank") != best.get("input_extra_rank")]
            reject_reasons["preserve_best_on_empty"] += 1

    accepted.sort(key=lambda row: int(row.get("input_extra_rank", 0)))
    baseline_accepted = list(accepted)
    intervention = dict(info.get("counterfactual_intervention") or {})
    intervention.update(
        {
            "baseline_accepted_candidate_ids": [
                str(row.get("candidate_id") or "") for row in baseline_accepted
            ],
            "baseline_accepted_count": len(baseline_accepted),
            "baseline_admitted_tokens": int(
                sum(max(0, int(row.get("token_count", 0) or 0)) for row in baseline_accepted)
            ),
        }
    )
    if effective_admission_policy == "counterfactual_drop":
        drop_rank = int(counterfactual_drop_rank)
        if 0 <= drop_rank < len(accepted):
            dropped = accepted.pop(drop_rank)
            dropped_id = str(dropped.get("candidate_id") or "")
            rejected.append(
                {**dropped, "reject_reasons": ["counterfactual_leave_one_out"]}
            )
            reject_reasons["counterfactual_leave_one_out"] += 1
            coverage_rescued_ids.discard(dropped_id)
            intervention.update(
                {
                    "applied": True,
                    "dropped_candidate_id": dropped_id,
                    "dropped_title": str(dropped.get("title") or ""),
                    "dropped_token_count": int(dropped.get("token_count", 0) or 0),
                }
            )
            admitted_slot_ids = set(lineage_context.get("base_covered_slot_ids") or [])
            for admitted in accepted:
                admitted_slot_ids.update(admitted.get("covered_slot_ids") or [])
                admitted_slot_ids.update(admitted.get("uncovered_slot_ids") or [])
        else:
            intervention["reason"] = "drop_rank_out_of_range"
    info["counterfactual_intervention"] = intervention
    accepted_docs = [str(row.get("doc") or "") for row in accepted]
    passthrough_docs = [str(row.get("doc") or "") for row in passthrough_extra_records]
    final_docs = base_docs + passthrough_docs + accepted_docs
    cost_accepted_schema_rows = [
        row
        for row in accepted
        if str(row.get("source") or "") == "query_schema_memory"
    ]
    cost_accepted_schema_count = len(cost_accepted_schema_rows)
    passthrough_early_schema_count = sum(
        str(row.get("source") or "") == "query_schema_memory"
        and bool((row.get("metadata") or {}).get("early_schema_bridge_materialization"))
        for row in passthrough_extra_records
    )
    planned_early_schema_count = sum(
        bool(row.get("early_schema_bridge_materialization"))
        for row in cost_accepted_schema_rows
    )
    early_accepted_schema_count = (
        passthrough_early_schema_count + planned_early_schema_count
    )
    accepted_schema_count = cost_accepted_schema_count + passthrough_early_schema_count
    accepted_entity_predicate_count = sum(
        str(row.get("source") or "") == "entity_predicate_fact_join"
        for row in accepted
    )
    accepted_entity_predicate_rows = [
        row
        for row in accepted
        if str(row.get("source") or "") == "entity_predicate_fact_join"
    ]
    accepted_shadow_count = sum(
        str(row.get("source") or "") == "typed_fact_join_shadow" for row in accepted
    )
    info["applied"] = (
        bool(accepted_docs)
        or bool(early_accepted_schema_count)
        or bool(intervention.get("applied"))
    )
    info["reason"] = (
        "counterfactual_candidate_removed"
        if bool(intervention.get("applied"))
        else "deficit_budget_replacement_applied"
        if int(replacement_info.get("replacement_count", 0) or 0) > 0
        else "entity_predicate_support_appended"
        if accepted_entity_predicate_count
        else "schema_support_view_appended"
        if cost_accepted_schema_count
        else "early_schema_bridge_materialized"
        if early_accepted_schema_count
        else "coverage_constrained_evidence_appended"
        if coverage_rescued_ids
        else "shadow_table_appended"
        if accepted_docs
        else "learned_harm_vetoed_all"
        if effective_admission_policy in {"learned_veto", "counterfactual_slate_veto"}
        and reject_reasons.get("learned_harm_veto", 0) > 0
        else "pass_through_no_admissible_candidate"
    )
    info["accepted_count"] = len(passthrough_extra_records) + len(accepted)
    info["admitted_candidate_count"] = len(accepted)
    info["admitted_candidate_tokens"] = int(sum(
        max(0, int(row.get("token_count", 0) or 0)) for row in accepted
    ))
    info["token_budget_utilization"] = _round_float(
        info["admitted_candidate_tokens"] / float(max(1, int(token_budget)))
    )
    required_slot_ids = set(lineage_context.get("required_slot_ids") or [])
    info["coverage_constraint_enabled"] = bool(
        effective_admission_policy in {
            "rule",
            "learned",
            "learned_veto",
            "counterfactual_slate_veto",
        }
        and lineage_context.get("required_uncovered_slot_ids")
    )
    info["coverage_rescued_count"] = len(coverage_rescued_ids)
    info["admitted_covered_slot_ids"] = sorted(admitted_slot_ids)
    info["remaining_uncovered_slot_ids"] = sorted(
        required_slot_ids - admitted_slot_ids
    )
    info["required_slot_coverage"] = _round_float(
        len(required_slot_ids & admitted_slot_ids) / float(max(1, len(required_slot_ids)))
    )
    info["accepted_shadow_count"] = int(accepted_shadow_count)
    info["accepted_schema_memory_count"] = int(accepted_schema_count)
    info["early_schema_bridge_materialized_count"] = int(early_accepted_schema_count)
    info["cost_planned_early_schema_count"] = int(planned_early_schema_count)
    info["passthrough_early_schema_count"] = int(passthrough_early_schema_count)
    info["accepted_entity_predicate_fact_join_count"] = int(
        accepted_entity_predicate_count
    )
    info["accepted_entity_predicate_titles"] = [
        str(row.get("title") or _title(row.get("doc")))
        for row in accepted_entity_predicate_rows
    ]
    info["accepted_entity_predicate_doc_fingerprints"] = [
        hashlib.sha256(str(row.get("doc") or "").encode("utf-8")).hexdigest()[:16]
        for row in accepted_entity_predicate_rows
    ]
    schema_materialization_result = dict(info.get("query_schema_memory_materialization") or {})
    schema_materialization_result["early_accepted_count"] = int(early_accepted_schema_count)
    schema_materialization_result["cost_accepted_count"] = int(cost_accepted_schema_count)
    schema_materialization_result["cost_rejected_count"] = max(
        0,
        len(schema_records) - int(cost_accepted_schema_count),
    )
    if schema_records and not cost_accepted_schema_count:
        schema_materialization_result["reason"] = "rejected_by_cost_model"
    elif cost_accepted_schema_count:
        schema_materialization_result["reason"] = "reader_support_materialized"
    info["query_schema_memory_materialization"] = schema_materialization_result
    early_schema_result = dict(
        info.get("early_schema_bridge_materialization") or {}
    )
    early_schema_result["route_through_cost_optimizer"] = bool(
        route_early_schema_through_optimizer
    )
    early_schema_result["cost_accepted_count"] = int(planned_early_schema_count)
    early_schema_result["passthrough_accepted_count"] = int(
        passthrough_early_schema_count
    )
    early_schema_result["cost_rejected_count"] = max(
        0,
        len(early_schema_records)
        - int(planned_early_schema_count)
        - int(passthrough_early_schema_count),
    )
    if route_early_schema_through_optimizer and early_schema_records:
        early_schema_result["execution_reason"] = (
            "selected_by_cost_plan"
            if planned_early_schema_count
            else "rejected_by_cost_plan"
        )
    info["early_schema_bridge_materialization"] = early_schema_result
    entity_predicate_result = dict(
        info.get("entity_predicate_fact_join_materialization") or {}
    )
    entity_predicate_result["cost_accepted_count"] = int(
        accepted_entity_predicate_count
    )
    entity_predicate_result["cost_rejected_count"] = max(
        0,
        len(entity_predicate_records) - int(accepted_entity_predicate_count),
    )
    if entity_predicate_records and not accepted_entity_predicate_count:
        entity_predicate_result["reason"] = "rejected_by_cost_model"
    elif accepted_entity_predicate_count:
        entity_predicate_result["reason"] = "reader_support_materialized"
    info["entity_predicate_fact_join_materialization"] = entity_predicate_result
    info["preserved_existing_extra_count"] = len(passthrough_extra_records)
    info["rejected_count"] = max(0, len(candidate_records) - len(accepted))
    info["final_doc_count"] = len(final_docs)
    info["accepted_titles"] = [_title(row.get("doc")) for row in passthrough_extra_records] + [
        str(row.get("title") or "") for row in accepted
    ]
    info["rejected_titles"] = [str(row.get("title") or "") for row in rejected[:8]]
    info["reject_reason_counts"] = dict(reject_reasons)
    info["learned_harm_veto_count"] = int(
        sum(bool(row.get("learned_harm_veto")) for row in scored)
    )
    info["learned_harm_veto_tokens"] = int(
        sum(
            max(0, int(row.get("token_count", 0) or 0))
            for row in scored
            if bool(row.get("learned_harm_veto"))
        )
    )
    accepted_sources = Counter(
        str(record.get("source") or "extra_evidence")
        for record in passthrough_extra_records
    )
    accepted_sources.update(str(row.get("source") or "extra_evidence") for row in accepted)
    info["accepted_source_counts"] = dict(accepted_sources)
    accepted_ids = {str(row.get("candidate_id") or "") for row in accepted}
    rejected_by_id = {
        str(row.get("candidate_id") or ""): list(row.get("reject_reasons") or [])
        for row in rejected
    }
    info["decision_trace"] = [
        {
            **_candidate_trace_payload(row),
            "candidate_id": str(row.get("candidate_id") or _candidate_id(row)),
            "accepted": str(row.get("candidate_id") or "") in accepted_ids,
            "accepted_by_coverage_constraint": str(row.get("candidate_id") or "")
            in coverage_rescued_ids,
            "reject_reasons": rejected_by_id.get(str(row.get("candidate_id") or ""), []),
            "learned_decision": str(row.get("learned_decision") or "not_used"),
            "learned_expected_gain": _round_float(row.get("learned_expected_gain", 0.0)),
            "learned_expected_utility": _round_float(
                row.get("learned_expected_utility", 0.0)
            ),
            "learned_p_improve": _round_float(row.get("learned_p_improve", 0.0)),
            "learned_p_degrade": _round_float(row.get("learned_p_degrade", 0.0)),
            "learned_confidence": _round_float(row.get("learned_confidence", 0.0)),
            "learned_harm_expected_gain": _round_float(
                row.get("learned_harm_expected_gain", 0.0)
            ),
            "learned_harm_p_improve": _round_float(
                row.get("learned_harm_p_improve", 0.0)
            ),
            "learned_harm_p_degrade": _round_float(
                row.get("learned_harm_p_degrade", 0.0)
            ),
            "learned_harm_margin": _round_float(row.get("learned_harm_margin", 0.0)),
            "learned_harm_confidence": _round_float(
                row.get("learned_harm_confidence", 0.0)
            ),
            "learned_harm_support": _round_float(row.get("learned_harm_support", 0.0)),
            "learned_harm_feature_count": int(
                row.get("learned_harm_feature_count", 0) or 0
            ),
            "learned_harm_primary_feature": str(
                row.get("learned_harm_primary_feature") or ""
            ),
            "learned_harm_veto": bool(row.get("learned_harm_veto")),
            "counterfactual_quality_upper_ci": _round_float(
                row.get("counterfactual_quality_upper_ci", 0.0)
            ),
            "counterfactual_stable_dataset_count": int(
                row.get("counterfactual_stable_dataset_count", 0) or 0
            ),
            "counterfactual_harm_family_count": int(
                row.get("counterfactual_harm_family_count", 0) or 0
            ),
            "counterfactual_coverage_protected": bool(
                row.get("counterfactual_coverage_protected")
            ),
            "counterfactual_net_drop_utility": _round_float(
                row.get("counterfactual_net_drop_utility", 0.0)
            ),
            "slate_size": int(row.get("slate_size", 0) or 0),
            "slate_rank": int(row.get("slate_rank", 0) or 0),
            "slate_token_share": _round_float(row.get("slate_token_share", 0.0)),
            "slate_source_count": int(row.get("slate_source_count", 0) or 0),
            "slate_trace_size": int(row.get("slate_trace_size", 0) or 0),
            "slate_source_diversity": int(
                row.get("slate_source_diversity", 0) or 0
            ),
            "slate_same_lineage_count": int(
                row.get("slate_same_lineage_count", 0) or 0
            ),
            "slate_coverage_redundancy": int(
                row.get("slate_coverage_redundancy", 0) or 0
            ),
            **{
                key: _round_float(row.get(key, 0.0))
                for key in (
                    "slate_net_utility_minus_mean",
                    "slate_net_utility_minus_max",
                    "slate_net_utility_percentile",
                    "slate_utility_minus_mean",
                    "slate_utility_minus_max",
                    "slate_utility_percentile",
                    "slate_risk_minus_mean",
                    "slate_risk_minus_max",
                    "slate_risk_percentile",
                    "slate_relation_overlap_minus_mean",
                    "slate_relation_overlap_minus_max",
                    "slate_relation_overlap_percentile",
                    "slate_schema_selectivity_minus_mean",
                    "slate_schema_selectivity_minus_max",
                    "slate_schema_selectivity_percentile",
                    "slate_provenance_score_minus_mean",
                    "slate_provenance_score_minus_max",
                    "slate_provenance_score_percentile",
                )
            },
        }
        for row in _sort_candidates_for_policy(scored, effective_admission_policy)
    ]
    info["top_candidates"] = [
        {
            key: value
            for key, value in row.items()
            if key
            in {
                "source",
                "candidate_id",
                "title",
                "input_extra_rank",
                "token_count",
                "query_overlap",
                "missing_gain",
                "novelty",
                "table_score",
                "table_row_count",
                "utility",
                "risk",
                "token_cost",
                "compute_cost",
                "net_utility",
                "admission_sort_utility",
                "learned_decision",
                "learned_expected_gain",
                "learned_expected_utility",
                "learned_p_improve",
                "learned_p_degrade",
                "learned_confidence",
                "learned_harm_expected_gain",
                "learned_harm_p_improve",
                "learned_harm_p_degrade",
                "learned_harm_margin",
                "learned_harm_confidence",
                "learned_harm_support",
                "learned_harm_feature_count",
                "learned_harm_primary_feature",
                "learned_harm_veto",
                "counterfactual_quality_upper_ci",
                "counterfactual_stable_dataset_count",
                "counterfactual_harm_family_count",
                "counterfactual_coverage_protected",
                "plan_type",
                "plan_utility",
                "plan_net_utility",
                "plan_risk",
                "endpoint_overlap",
                "relation_overlap",
                "schema_score",
                "schema_idf",
                "join_key_overlap",
                "schema_frequency",
                "schema_selectivity",
                "provenance_score",
                "compact_table",
                "best_rank",
                "worst_rank",
                "original_rank",
                "answer_slot",
                "required_slot_ids",
                "covered_slot_ids",
                "uncovered_slot_ids",
                "coverage_gain",
                "lineage_confidence",
                "lineage_group_id",
                "comparison_arm",
                "covered_branch_ids",
                "comparison_branch_id",
                "predicate_tokens",
                "comparison_query",
                "compact_projection",
                "projection_reason",
                "original_doc_token_count",
                "projected_doc_token_count",
                "projection_token_savings",
                "coverage_rescue_eligible",
                "slate_size",
                "slate_rank",
                "slate_token_share",
                "slate_source_count",
            }
        }
        for row in sorted(scored, key=lambda row: -float(row.get("net_utility", 0.0)))[:5]
    ]
    top_decisions = {
        str(row.get("candidate_id") or ""): row for row in info["decision_trace"]
    }
    for row in info["top_candidates"]:
        decision = top_decisions.get(str(row.get("candidate_id") or ""), {})
        row["accepted"] = bool(decision.get("accepted"))
        row["reject_reasons"] = list(decision.get("reject_reasons") or [])
    return final_docs, info
