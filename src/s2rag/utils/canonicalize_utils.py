import re
from datetime import date
from typing import Any, Dict, List, Tuple


YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")
MONTH_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
FULL_DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)
MONTH_NAME_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
MONTH_FIRST_COMPLETE_DATE_RE = re.compile(
    rf"\b(?P<month>{MONTH_NAME_PATTERN})\.?\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,\s*|\s+)"
    r"(?P<year>\d{4})\b",
    re.IGNORECASE,
)
DAY_FIRST_COMPLETE_DATE_RE = re.compile(
    r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+"
    rf"(?P<month>{MONTH_NAME_PATTERN})\.?(?:,\s*|\s+)"
    r"(?P<year>\d{4})\b",
    re.IGNORECASE,
)
QUESTION_COMPLETE_DATE_RE = re.compile(
    r"\b(?:birth\s*date|date\s+of\s+birth|death\s*date|date\s+of\s+death|"
    r"what\s+date|which\s+date|on\s+what\s+date|when)\b",
    re.IGNORECASE,
)
MONTH_NUMBERS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

TRAILING_PAREN_RE = re.compile(r"^(?P<base>.+?)\s*\((?P<paren>[^)]{1,60})\)\s*$")
WORD_RE = re.compile(r"[a-z0-9]+")

QUESTION_CITY_RE = re.compile(r"\b(?:what|which|where|in what)\s+(?:city|town)\b", re.IGNORECASE)
QUESTION_COUNTRY_RE = re.compile(r"\b(?:what|which|where|in what)\s+country\b", re.IGNORECASE)
QUESTION_COUNTY_RE = re.compile(r"\b(?:what|which|where|in what)\s+county\b", re.IGNORECASE)
QUESTION_YEAR_RE = re.compile(r"\b(?:what|which|in what|in which)\s+year\b", re.IGNORECASE)
QUESTION_DATE_RE = re.compile(r"\b(?:what|which|on what)\s+date\b|\bwhen\b", re.IGNORECASE)
QUESTION_LOCATION_RE = re.compile(
    r"\b(?:where|place|location|located|city|town|county|country|state|province)\b",
    re.IGNORECASE,
)
QUESTION_DEATH_LOCATION_RE = re.compile(
    r"\b(?:where\s+did\b.+\b(?:die|died)|where\s+was\b.+\bdeath|place\s+of\s+death|died\s+in)\b",
    re.IGNORECASE,
)
QUESTION_BIRTH_LOCATION_RE = re.compile(
    r"\b(?:where\s+(?:was|were)\b.+\bborn|place\s+of\s+birth|birthplace|born\s+in)\b",
    re.IGNORECASE,
)
QUESTION_OCCUPATION_RE = re.compile(
    r"\b(?:occupation|profession|job|role|position)\b|"
    r"\bwhat\s+(?:does|did|do|is|was|are|were)\b.+\b(?:do|known as|work as)\b",
    re.IGNORECASE,
)

SPLIT_RE = re.compile(r"\s*(?:;|,|\band\b|/)\s*", re.IGNORECASE)
LEADING_DESCRIPTOR_RE = re.compile(
    r"^(?:an?|the)\s+|"
    r"^(?:american|british|english|australian|canadian|dutch|french|german|"
    r"irish|italian|japanese|spanish|swedish|norwegian|danish|finnish|"
    r"russian|indian|chinese|filipino|south african|new zealand|scottish|"
    r"welsh|mexican|brazilian|argentine|polish|czech|hungarian|greek|"
    r"romanian|ukrainian|israeli|turkish|korean)\s+",
    re.IGNORECASE,
)

OCCUPATION_TERMS = {
    "actor",
    "actress",
    "artist",
    "astronomer",
    "athlete",
    "author",
    "banker",
    "baseball player",
    "basketball player",
    "batsman",
    "biologist",
    "businessman",
    "businessperson",
    "businesswoman",
    "cartoonist",
    "chemist",
    "composer",
    "cricketer",
    "dentist",
    "designer",
    "diplomat",
    "director",
    "economist",
    "engineer",
    "entrepreneur",
    "filmmaker",
    "footballer",
    "historian",
    "illustrator",
    "journalist",
    "lawyer",
    "mathematician",
    "military officer",
    "model",
    "musician",
    "novelist",
    "painter",
    "physician",
    "poet",
    "politician",
    "printer",
    "producer",
    "professor",
    "psychologist",
    "scientist",
    "singer",
    "soccer player",
    "soldier",
    "songwriter",
    "writer",
}

HIGH_SPECIFICITY_OCCUPATIONS = {
    "actor",
    "actress",
    "artist",
    "astronomer",
    "baseball player",
    "basketball player",
    "batsman",
    "biologist",
    "cartoonist",
    "chemist",
    "composer",
    "cricketer",
    "dentist",
    "designer",
    "diplomat",
    "director",
    "economist",
    "engineer",
    "entrepreneur",
    "filmmaker",
    "footballer",
    "historian",
    "journalist",
    "lawyer",
    "mathematician",
    "model",
    "musician",
    "novelist",
    "painter",
    "physician",
    "politician",
    "producer",
    "professor",
    "psychologist",
    "scientist",
    "singer",
    "soccer player",
    "songwriter",
}

LOW_PRIORITY_OCCUPATIONS = {
    "businessman",
    "businessperson",
    "businesswoman",
    "illustrator",
    "military officer",
    "poet",
    "printer",
    "soldier",
    "writer",
}

LOCATION_SLOT_TERMS = {
    "location",
    "place",
    "city",
    "town",
    "county",
    "country",
    "state",
    "province",
    "region",
}
SPECIFIC_LOCATION_SLOT_TERMS = {"city", "town"}
DEATH_RELATION_RE = re.compile(r"\b(?:die|died|death|dead|place of death)\b", re.IGNORECASE)


def _normalize(text: str) -> str:
    return " ".join(WORD_RE.findall((text or "").lower()))


def _evidence_text(qa_docs: List[str]) -> str:
    return "\n".join(str(doc) for doc in qa_docs or [])


def _doc_titles(qa_docs: List[str]) -> List[str]:
    titles = []
    for doc in qa_docs or []:
        first_line = str(doc).splitlines()[0].strip()
        if first_line:
            titles.append(first_line)
    return titles


def _new_meta(enabled: bool, before: str) -> Dict[str, Any]:
    return {
        "enabled": enabled,
        "changed": False,
        "action": "none" if enabled else "disabled",
        "before": before,
        "after": before,
        "confidence": 0.0,
    }


def _finish(meta: Dict[str, Any], action: str, before: str, after: str, confidence: float, **extra: Any) -> Dict[str, Any]:
    meta.update({
        "changed": after != before,
        "action": action,
        "before": before,
        "after": after,
        "confidence": round(float(confidence), 4),
    })
    meta.update(extra)
    return meta


def _split_candidate_answer(answer: str) -> List[str]:
    parts = [part.strip(" \t\r\n.;:-") for part in SPLIT_RE.split(answer) if part.strip(" \t\r\n.;:-")]
    seen = set()
    unique_parts = []
    for part in parts:
        norm = _normalize(part)
        if norm and norm not in seen:
            seen.add(norm)
            unique_parts.append(part)
    return unique_parts


def _canonical_occupation_candidate(candidate: str) -> str:
    cleaned = candidate.strip(" \t\r\n.;:-")
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = LEADING_DESCRIPTOR_RE.sub("", cleaned).strip()
    return cleaned


def _occupation_head(candidate: str) -> str:
    norm = _normalize(candidate)
    if norm in OCCUPATION_TERMS:
        return norm
    tokens = norm.split()
    for width in (3, 2, 1):
        if len(tokens) >= width:
            tail = " ".join(tokens[-width:])
            if tail in OCCUPATION_TERMS:
                return tail
    return norm


def _looks_like_occupation(candidate: str) -> bool:
    head = _occupation_head(candidate)
    if head in OCCUPATION_TERMS:
        return True
    return bool(re.search(r"(?:ist|ian|er|or|ant|man|woman)$", head))


def _score_occupation_candidate(candidate: str, index: int, total: int, evidence_norm: str, title_norms: List[str]) -> Dict[str, Any]:
    clean = _canonical_occupation_candidate(candidate)
    head = _occupation_head(clean)
    score = 0.55
    reasons = []

    if head in OCCUPATION_TERMS:
        score += 0.08
        reasons.append("known_occupation")
    if head in HIGH_SPECIFICITY_OCCUPATIONS:
        score += 0.15
        reasons.append("high_specificity")
    if head in LOW_PRIORITY_OCCUPATIONS:
        score -= 0.12
        reasons.append("low_priority")
    if _normalize(clean) and _normalize(clean) in evidence_norm:
        score += 0.05
        reasons.append("evidence_mention")
    if any(_normalize(clean) and _normalize(clean) in title for title in title_norms):
        score += 0.08
        reasons.append("title_mention")
    if index == 0:
        score += 0.02
    if index == total - 1:
        score += 0.05
    score -= 0.02 * max(0, len(clean.split()) - 1)

    return {
        "raw": candidate,
        "clean": clean,
        "head": head,
        "score": round(score, 4),
        "reasons": reasons,
    }


def _parse_complete_date_surface(value: str) -> Tuple[int, int, int] | None:
    cleaned = (value or "").strip(" \t\r\n.;:()[]{}")
    match = MONTH_FIRST_COMPLETE_DATE_RE.fullmatch(cleaned)
    if match is None:
        match = DAY_FIRST_COMPLETE_DATE_RE.fullmatch(cleaned)
    if match is None:
        return None

    month = MONTH_NUMBERS.get(match.group("month").casefold()[:3])
    if month is None:
        return None
    try:
        parsed = date(
            int(match.group("year")),
            month,
            int(match.group("day")),
        )
    except ValueError:
        return None
    return parsed.year, parsed.month, parsed.day


def equivalent_complete_date_surfaces(left: str, right: str) -> bool:
    """Return whether two complete date surfaces denote the same day.

    Partial dates and non-date strings deliberately return ``False``.  This
    keeps answer admission from treating a shared year or month as sufficient
    evidence for replacing an incumbent answer.
    """
    left_date = _parse_complete_date_surface(left)
    return bool(
        left_date is not None
        and left_date == _parse_complete_date_surface(right)
    )


def _source_grounded_complete_date_surface(
    question: str,
    answer: str,
    qa_docs: List[str],
) -> Tuple[str, str, float]:
    """Return the source spelling of an equivalent complete date.

    This is deliberately evidence-bound: it never invents a date format and
    never consults a gold answer.  A rewrite is allowed only when the complete
    predicted date and a complete date span in the reader-visible evidence
    resolve to the same valid calendar day.
    """
    if not QUESTION_COMPLETE_DATE_RE.search(question or ""):
        return answer, "", 0.0
    parsed_answer = _parse_complete_date_surface(answer)
    if parsed_answer is None:
        return answer, "", 0.0

    evidence = _evidence_text(qa_docs)
    candidates = []
    for pattern in (DAY_FIRST_COMPLETE_DATE_RE, MONTH_FIRST_COMPLETE_DATE_RE):
        for match in pattern.finditer(evidence):
            surface = match.group(0).strip(" ,.;")
            if _parse_complete_date_surface(surface) == parsed_answer:
                candidates.append((match.start(), surface))
    if not candidates:
        return answer, "", 0.0

    _, source_surface = min(candidates, key=lambda item: item[0])
    if source_surface == answer:
        return answer, "", 0.0
    return source_surface, "source_grounded_complete_date_surface", 0.97


def _canonicalize_year_or_date(
    question: str,
    answer: str,
    qa_docs: List[str],
) -> Tuple[str, str, float]:
    grounded, action, confidence = _source_grounded_complete_date_surface(
        question,
        answer,
        qa_docs,
    )
    if action:
        return grounded, action, confidence

    if QUESTION_YEAR_RE.search(question):
        years = YEAR_RE.findall(answer)
        unique_years = list(dict.fromkeys(years))
        if len(unique_years) == 1:
            return unique_years[0], "year_extracted", 0.95

    if QUESTION_DATE_RE.search(question) and (";" in answer or " and " in answer.lower()):
        date_match = FULL_DATE_RE.search(answer)
        if date_match:
            return date_match.group(0).strip(" ,.;"), "date_extracted", 0.85
        if MONTH_RE.search(answer):
            parts = _split_candidate_answer(answer)
            month_parts = [part for part in parts if MONTH_RE.search(part)]
            if len(month_parts) == 1:
                return month_parts[0], "date_phrase_extracted", 0.8

    return answer, "", 0.0


def _canonicalize_parenthetical(question: str, answer: str, evidence_norm: str) -> Tuple[str, str, float]:
    match = TRAILING_PAREN_RE.match(answer)
    if not match:
        return answer, "", 0.0

    base = match.group("base").strip()
    paren = match.group("paren").strip().lower()
    if not base:
        return answer, "", 0.0

    title_like_question = re.search(
        r"\b(?:what|which)\b.+\b(?:song|album|film|movie|book|novel|play|series|screenplay|title)\b",
        question,
        re.IGNORECASE,
    )
    year_disambiguator = bool(re.fullmatch(r"\d{4}", paren))
    media_disambiguator = bool(re.search(r"\b(?:film|movie|song|album|novel|play|series|book)\b", paren))
    evidence_supports_base = _normalize(base) and _normalize(base) in evidence_norm

    if year_disambiguator or media_disambiguator or (title_like_question and evidence_supports_base):
        return base, "parenthetical_disambiguation_stripped", 0.82
    return answer, "", 0.0


def _canonicalize_location(question: str, answer: str, evidence_norm: str) -> Tuple[str, str, float]:
    parts = [part.strip(" \t\r\n.;:-") for part in answer.split(",") if part.strip(" \t\r\n.;:-")]
    if QUESTION_CITY_RE.search(question) and len(parts) >= 2 and not re.search(r"\d", answer):
        return parts[0], "explicit_city_location_shortened", 0.82
    if QUESTION_COUNTRY_RE.search(question) and len(parts) >= 2 and not re.search(r"\d", answer):
        return parts[-1], "explicit_country_location_shortened", 0.82

    if QUESTION_COUNTY_RE.search(question):
        norm_answer = _normalize(answer)
        if norm_answer and "county" not in norm_answer:
            county_candidate = f"{answer.strip(' ,.;')} county"
            if _normalize(county_candidate) in evidence_norm:
                return county_candidate, "county_suffix_added_from_evidence", 0.78

    return answer, "", 0.0


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _positive_count(container: Dict[str, Any], *keys: str) -> bool:
    for key in keys:
        try:
            if float(container.get(key, 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _schema_memory_touched(evidence_admission: Dict[str, Any], query_schema_memory: Dict[str, Any]) -> bool:
    evidence_admission = _as_dict(evidence_admission)
    query_schema_memory = _as_dict(query_schema_memory)
    early = _as_dict(evidence_admission.get("early_schema_bridge_materialization"))
    materialized = _as_dict(evidence_admission.get("query_schema_memory_materialization"))
    cross_schema = _as_dict(evidence_admission.get("query_schema_memory_cross_schema_join"))
    active_tuple = _as_dict(evidence_admission.get("active_schema_bridge_tuple_executor"))
    entity_predicate_materialized = _as_dict(
        evidence_admission.get("entity_predicate_fact_join_materialization")
    )

    return any(
        [
            bool(query_schema_memory.get("applied")),
            bool(query_schema_memory.get("top2_changed")),
            _positive_count(
                query_schema_memory,
                "accepted_count",
                "materialized_count",
                "projected_candidate_count",
            ),
            _positive_count(
                early,
                "accepted_count",
                "validated_tuple_accepted_count",
                "validated_tuple_reader_visible_projection_count",
                "active_schema_bridge_tuple_accepted_count",
            ),
            _positive_count(
                materialized,
                "accepted_count",
                "validated_topk_count",
                "validated_tuple_accepted_count",
                "validated_tuple_reader_visible_projection_count",
                "active_schema_bridge_tuple_accepted_count",
                "cross_schema_accepted_count",
                "cross_schema_validated_topk_count",
            ),
            _positive_count(
                cross_schema,
                "merged_candidate_count",
                "cross_schema_accepted_count",
            ),
            _positive_count(
                active_tuple,
                "accepted_count",
                "covered_by_reader_visible_count",
            ),
            _positive_count(
                entity_predicate_materialized,
                "terminal_tuple_accepted_count",
                "projected_candidate_count",
            ),
        ]
    )


def _candidate_metadata(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    metadata = dict(item)
    nested = item.get("metadata")
    if isinstance(nested, dict):
        metadata.update(nested)
    return metadata


def _collect_schema_surface_candidates(
    evidence_admission: Dict[str, Any],
    query_schema_memory: Dict[str, Any],
) -> List[Dict[str, Any]]:
    evidence_admission = _as_dict(evidence_admission)
    query_schema_memory = _as_dict(query_schema_memory)
    containers = [
        query_schema_memory,
        evidence_admission,
        _as_dict(evidence_admission.get("early_schema_bridge_materialization")),
        _as_dict(evidence_admission.get("query_schema_memory_materialization")),
        _as_dict(evidence_admission.get("query_schema_memory_cross_schema_join")),
        _as_dict(evidence_admission.get("active_schema_bridge_tuple_executor")),
        _as_dict(evidence_admission.get("entity_predicate_fact_join_materialization")),
    ]
    candidate_keys = (
        "validated_tuple_top_candidates",
        "validated_topk_candidates",
        "covered_by_reader_visible_candidates",
        "top_candidates",
        "accepted_candidates",
        "accepted_rows",
    )
    candidates: List[Dict[str, Any]] = []
    seen = set()
    for container in containers:
        for key in candidate_keys:
            for item in _as_list(container.get(key)):
                metadata = _candidate_metadata(item)
                if not metadata:
                    continue
                value = str(
                    metadata.get("answer_surface")
                    or metadata.get("answer_value")
                    or metadata.get("terminal_answer")
                    or metadata.get("terminal_value")
                    or ""
                ).strip()
                if not value:
                    continue
                fingerprint = (
                    _normalize(value),
                    str(metadata.get("doc_id") or metadata.get("candidate_id") or ""),
                    str(metadata.get("source_sentence") or "")[:80],
                )
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                metadata["_candidate_list"] = key
                candidates.append(metadata)
    return candidates


def _answer_has_multi_surface(answer: str) -> bool:
    return bool(re.search(r"\s*(?:;|/|\band\b)\s*", answer, re.IGNORECASE))


def _comma_location_parts(answer: str) -> List[str]:
    return [part.strip(" \t\r\n.;:-") for part in (answer or "").split(",") if part.strip(" \t\r\n.;:-")]


def _collapse_duplicate_location_surface(answer: str) -> Tuple[str, str, float, Dict[str, Any]]:
    parts = _comma_location_parts(answer)
    if len(parts) < 2 or any(re.search(r"\d", part) for part in parts):
        return answer, "", 0.0, {}

    unique_parts: List[str] = []
    seen = set()
    duplicate_count = 0
    for part in parts:
        norm = _normalize(part)
        if not norm:
            continue
        if norm in seen:
            duplicate_count += 1
            continue
        seen.add(norm)
        unique_parts.append(part)

    if duplicate_count <= 0 or len(unique_parts) >= len(parts):
        return answer, "", 0.0, {"parts": parts}
    if len(unique_parts) == 1:
        return unique_parts[0], "schema_duplicate_location_component_collapsed", 0.92, {"parts": parts}
    return ", ".join(unique_parts), "schema_duplicate_location_component_collapsed", 0.88, {"parts": parts}


def _schema_candidate_value(candidate: Dict[str, Any]) -> str:
    return str(
        candidate.get("answer_surface")
        or candidate.get("answer_value")
        or candidate.get("terminal_answer")
        or candidate.get("terminal_value")
        or ""
    ).strip(" \t\r\n.;:-")


def _schema_candidate_slot(candidate: Dict[str, Any]) -> str:
    slot = (
        candidate.get("terminal_answer_slot")
        or candidate.get("answer_contract_answer_slot")
        or candidate.get("answer_slot")
        or candidate.get("terminal_value_type")
        or candidate.get("answer_contract_value_type")
        or candidate.get("value_type")
        or ""
    )
    return _normalize(str(slot))


def _schema_candidate_relation_text(candidate: Dict[str, Any]) -> str:
    fields = [
        candidate.get("predicate"),
        candidate.get("relation"),
        candidate.get("relation_text"),
        candidate.get("terminal_relation"),
        candidate.get("source_sentence"),
        candidate.get("sentence_preview"),
    ]
    for key in ("terminal_relation_intersection", "predicate_tokens", "relation_tokens", "triple_path"):
        value = candidate.get(key)
        if isinstance(value, list):
            fields.extend(str(item) for item in value)
        elif value:
            fields.append(value)
    return " ".join(str(item) for item in fields if item)


def _best_matching_schema_location_candidate(
    answer: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    answer_norm = _normalize(answer)
    if not answer_norm:
        return {}
    first_part = _comma_location_parts(answer)[0] if _comma_location_parts(answer) else ""
    first_norm = _normalize(first_part)
    best: Dict[str, Any] = {}
    best_score = -1.0
    for candidate in candidates:
        value = _schema_candidate_value(candidate)
        value_norm = _normalize(value)
        if not value_norm:
            continue
        if value_norm != first_norm and value_norm not in answer_norm:
            continue
        slot = _schema_candidate_slot(candidate)
        location_like = (
            any(term in slot.split() for term in LOCATION_SLOT_TERMS)
            or slot in LOCATION_SLOT_TERMS
        )
        if not location_like:
            continue
        source_bound = candidate.get("source_sentence_value_bound")
        if source_bound is False:
            continue
        score = 0.0
        if value_norm == first_norm:
            score += 0.45
        if slot in SPECIFIC_LOCATION_SLOT_TERMS:
            score += 0.30
        elif slot in LOCATION_SLOT_TERMS:
            score += 0.16
        if bool(candidate.get("predicate_contract_match")):
            score += 0.18
        if candidate.get("terminal_relation_intersection"):
            score += 0.14
        if candidate.get("source_sentence_value_bound") is True:
            score += 0.08
        try:
            score += min(0.10, max(0.0, float(candidate.get("score", 0.0) or 0.0)) * 0.10)
        except (TypeError, ValueError):
            pass
        score = min(1.0, score)
        if score > best_score:
            best_score = score
            best = dict(candidate)
            best["_surface_score"] = round(score, 4)
    return best


def _schema_surface_location_rewrite(
    question: str,
    answer: str,
    candidates: List[Dict[str, Any]],
) -> Tuple[str, str, float, Dict[str, Any]]:
    parts = _comma_location_parts(answer)
    if len(parts) < 2 or _answer_has_multi_surface(answer) or re.search(r"\d", answer):
        return answer, "", 0.0, {}
    if not QUESTION_LOCATION_RE.search(question):
        return answer, "", 0.0, {"reason": "question_not_location_slot"}
    if QUESTION_COUNTRY_RE.search(question) or QUESTION_COUNTY_RE.search(question):
        return answer, "", 0.0, {"reason": "question_requests_coarser_location"}

    candidate = _best_matching_schema_location_candidate(answer, candidates)
    if not candidate:
        return answer, "", 0.0, {"reason": "no_matching_schema_location_candidate"}

    first_part = parts[0]
    candidate_slot = _schema_candidate_slot(candidate)
    relation_text = _schema_candidate_relation_text(candidate)
    predicate_bound = bool(candidate.get("predicate_contract_match")) or bool(
        candidate.get("terminal_relation_intersection")
    )
    source_bound = candidate.get("source_sentence_value_bound") is True

    if QUESTION_CITY_RE.search(question) and _normalize(first_part):
        return first_part, "schema_explicit_city_location_shortened", 0.88, {
            "selected_candidate": _schema_candidate_value(candidate),
            "selected_candidate_slot": candidate_slot,
            "selected_candidate_score": candidate.get("_surface_score", 0.0),
        }

    if candidate_slot in SPECIFIC_LOCATION_SLOT_TERMS and predicate_bound and source_bound:
        return first_part, "schema_specific_location_slot_shortened", 0.86, {
            "selected_candidate": _schema_candidate_value(candidate),
            "selected_candidate_slot": candidate_slot,
            "selected_candidate_score": candidate.get("_surface_score", 0.0),
        }

    if (
        QUESTION_DEATH_LOCATION_RE.search(question)
        and DEATH_RELATION_RE.search(relation_text)
        and predicate_bound
        and source_bound
    ):
        return first_part, "schema_death_location_shortened", 0.84, {
            "selected_candidate": _schema_candidate_value(candidate),
            "selected_candidate_slot": candidate_slot,
            "selected_candidate_score": candidate.get("_surface_score", 0.0),
        }

    if QUESTION_BIRTH_LOCATION_RE.search(question):
        return answer, "", 0.0, {
            "reason": "birth_location_granularity_ambiguous",
            "selected_candidate": _schema_candidate_value(candidate),
            "selected_candidate_slot": candidate_slot,
        }

    return answer, "", 0.0, {
        "reason": "schema_location_candidate_not_specific_enough",
        "selected_candidate": _schema_candidate_value(candidate),
        "selected_candidate_slot": candidate_slot,
        "selected_candidate_score": candidate.get("_surface_score", 0.0),
    }


def canonicalize_schema_touched_answer_surface(
    question: str,
    answer: str,
    qa_docs: List[str] = None,
    evidence_admission: Dict[str, Any] = None,
    query_schema_memory: Dict[str, Any] = None,
    answer_style: str = None,
    enabled: bool = False,
    min_confidence: float = 0.78,
) -> Tuple[str, Dict[str, Any]]:
    """Normalize answer surface granularity only when schema-memory evidence touched the query."""
    before = (answer or "").strip()
    meta = _new_meta(enabled=enabled, before=before)
    meta.update({
        "source": "schema_touched_answer_surface",
        "schema_touched": False,
        "candidate_count": 0,
        "reason": "disabled" if not enabled else "not_evaluated",
    })
    if not enabled or not before:
        return before, meta

    style = (answer_style or "").strip().lower()
    if style and style not in {"short_factoid", "multi_slot"}:
        meta["reason"] = "non_factoid_answer_style"
        return before, meta
    if len(WORD_RE.findall(before)) > 12:
        meta["reason"] = "answer_too_long"
        return before, meta

    touched = _schema_memory_touched(evidence_admission or {}, query_schema_memory or {})
    meta["schema_touched"] = touched
    if not touched:
        meta["reason"] = "schema_not_touched"
        return before, meta

    candidates = _collect_schema_surface_candidates(evidence_admission or {}, query_schema_memory or {})
    meta["candidate_count"] = len(candidates)

    after, action, confidence, detail = _collapse_duplicate_location_surface(before)
    if action and confidence >= min_confidence:
        return after, _finish(
            meta,
            action,
            before,
            after,
            confidence,
            reason=action,
            **detail,
        )

    after, action, confidence, detail = _schema_surface_location_rewrite(
        question,
        before,
        candidates,
    )
    if action and confidence >= min_confidence:
        return after, _finish(
            meta,
            action,
            before,
            after,
            confidence,
            reason=action,
            **detail,
        )

    if detail:
        meta.update(detail)
    meta.setdefault("reason", "no_safe_schema_surface_rewrite")
    if meta.get("reason") == "not_evaluated":
        meta["reason"] = "no_safe_schema_surface_rewrite"
    return before, meta


def _canonicalize_occupation(question: str, answer: str, qa_docs: List[str], min_confidence: float) -> Tuple[str, str, float, Dict[str, Any]]:
    if not QUESTION_OCCUPATION_RE.search(question):
        return answer, "", 0.0, {}
    if not re.search(r";|,|\band\b|/", answer, re.IGNORECASE):
        return answer, "", 0.0, {}

    parts = _split_candidate_answer(answer)
    if not (2 <= len(parts) <= 6):
        return answer, "", 0.0, {}

    evidence_norm = _normalize(_evidence_text(qa_docs))
    title_norms = [_normalize(title) for title in _doc_titles(qa_docs)]
    scored = [
        _score_occupation_candidate(part, idx, len(parts), evidence_norm, title_norms)
        for idx, part in enumerate(parts)
    ]
    viable = [item for item in scored if item["clean"] and _looks_like_occupation(item["clean"])]
    if not viable:
        return answer, "", 0.0, {"candidates": scored}

    viable.sort(key=lambda item: item["score"], reverse=True)
    top = viable[0]
    second_score = viable[1]["score"] if len(viable) > 1 else 0.0
    margin = top["score"] - second_score
    confidence = min(0.95, max(0.0, top["score"]))
    detail = {"candidates": scored, "margin": round(margin, 4)}

    if confidence >= min_confidence and (margin >= 0.10 or "title_mention" in top["reasons"]):
        return top["clean"], "occupation_list_canonicalized", confidence, detail

    return answer, "", 0.0, detail


def canonicalize_answer(
    question: str,
    answer: str,
    qa_docs: List[str] = None,
    enabled: bool = False,
    min_confidence: float = 0.65,
) -> Tuple[str, Dict[str, Any]]:
    """Canonicalize high-confidence answer-span variants without using gold answers."""
    before = (answer or "").strip()
    meta = _new_meta(enabled=enabled, before=before)
    if not enabled or not before:
        return before, meta

    qa_docs = qa_docs or []
    evidence_norm = _normalize(_evidence_text(qa_docs))

    after, action, confidence = _canonicalize_year_or_date(
        question,
        before,
        qa_docs,
    )
    if action and confidence >= min_confidence:
        return after, _finish(meta, action, before, after, confidence)

    after, action, confidence = _canonicalize_parenthetical(question, before, evidence_norm)
    if action and confidence >= min_confidence:
        return after, _finish(meta, action, before, after, confidence)

    after, action, confidence = _canonicalize_location(question, before, evidence_norm)
    if action and confidence >= min_confidence:
        return after, _finish(meta, action, before, after, confidence)

    after, action, confidence, detail = _canonicalize_occupation(question, before, qa_docs, min_confidence)
    if action and confidence >= min_confidence:
        return after, _finish(meta, action, before, after, confidence, **detail)

    if detail:
        meta.update(detail)
    return before, meta
