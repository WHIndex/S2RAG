import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .answer_verifier_utils import (
    ABSTAIN_RE,
    OCCUPATION_HEAD_RE,
    _candidate_key,
    _is_supported,
    _rank_candidates,
)
from .candidate_answer_selector_utils import _is_synthetic_doc
from .eval_utils import normalize_answer
from .evidence_object_arbitration_utils import (
    _filter_candidates,
    _is_containment_variant,
    _looks_verbose_or_multi_answer,
)
from .sentence_evidence_utils import infer_answer_target_type
from .typed_evidence_table_utils import _build_typed_table, _same_answer, _token_count


BOOLEAN_ANSWER_RE = re.compile(r"^\s*(yes|no|true|false)\s*\.?\s*$", re.I)
BOOLEAN_QUESTION_RE = re.compile(r"^\s*(is|are|was|were|do|does|did|has|have|had|can|could|will|would)\b", re.I)
WORD_RE = re.compile(r"[a-z0-9]+", re.I)
MONTH_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.I,
)
FULL_DATE_VALUE_RE = re.compile(
    r"\b(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?|"
    r"\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+\d{4})?)\b",
    re.I,
)
YEAR_ONLY_RE = re.compile(r"^\s*(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\s*$")
LATIN_SPECIES_RE = re.compile(r"^(?:[A-Z][a-z]+|[A-Z]\.)\s+[a-z][a-z-]+$")
INSTITUTION_RE = re.compile(r"\b(?:university|college|school|institute|academy|polytechnic|conservatory)\b", re.I)
INTERSECTION_RE = re.compile(r"\b(?:avenue|street|road|boulevard|drive|lane|parkway|highway|route|intersection)\b", re.I)
ORGANIZATION_RE = re.compile(
    r"\b(?:company|co\.?|corporation|corp\.?|inc\.?|ltd\.?|limited|group|"
    r"association|society|organization|organisation|institution|institute|"
    r"foundation|agency|manufacturer|publisher|press|department|ministry|"
    r"committee|council|club|team|band|label|watch\s+co)\b",
    re.I,
)
COUNTY_RE = re.compile(r"\b(?:county|parish|borough|district|shire|canton)\b", re.I)
REGION_RE = re.compile(r"\b(?:state|province|district|county|region|territory|prefecture|canton)\b", re.I)
CITY_RE = re.compile(r"\b(?:city|town|village|municipality|borough)\b", re.I)
CITY_PREFIX_RE = re.compile(
    r"^(?:new|los|las|san|santa|santo|saint|st|fort|port|lake|mount|"
    r"east|west|north|south|rio|sao|são|buenos)\b",
    re.I,
)
LOCATION_NOUN_RE = re.compile(
    r"\b(?:city|town|village|municipality|county|state|province|district|region|"
    r"country|kingdom|republic|island|ocean|sea|river|lake|mountain|valley|"
    r"territory|prefecture|canton|capital)\b",
    re.I,
)
ROLE_TITLE_RE = re.compile(
    r"\b(?:governor[-\s]?general|viceroy|governor|president|minister|king|queen|"
    r"emperor|empress|prince|princess|duke|duchess|chair(?:man|woman)?|"
    r"captain|coach|leader|director|officer|secretary|mayor|senator|"
    r"representative|judge|professor|commander|chief|title|position|office)\b",
    re.I,
)
HIGH_LEVEL_LOCATION_KEYS = {
    "england", "scotland", "wales", "ireland", "switzerland", "georgia", "oregon",
    "maryland", "british columbia", "canada", "united states", "united states america",
    "usa", "us", "france", "germany", "spain", "italy", "portugal", "india", "china",
    "japan", "australia", "brazil", "mexico",
}
PERSON_ROLE_EVIDENCE_PATTERNS = {
    "person_mother": re.compile(r"\b(?:mother|maternal|daughter\s+of)\b", re.I),
    "person_father": re.compile(r"\b(?:father|paternal|son\s+of)\b", re.I),
    "person_spouse": re.compile(r"\b(?:spouse|wife|husband|married|marriage)\b", re.I),
    "person_child": re.compile(r"\b(?:child|children|son|daughter)\b", re.I),
}
SINGLE_HOP_ATTRIBUTE_RE = re.compile(
    r"\b(?:born|birthplace|city|country|nationality|occupation|profession|"
    r"date of birth|birthdate|founded|founded by|manufacturer|located|located in|"
    r"capital|headquarters|language|genre|publisher|author|director|played|"
    r"portrayed|starred|won|winner|scored|most)\b",
    re.I,
)
BRIDGE_QUERY_RE = re.compile(
    r"\b(?:husband|wife|spouse|father|mother|parent|child|children|son|daughter|"
    r"performer of|director of|author of|composer of|producer of|company that|"
    r"person who|county where|state where|country where|where the|which.*where|"
    r"shares a border|iso code|1st|first|second|third)\b",
    re.I,
)
LOCATION_CITY_GRANULARITY_QUERY_RE = re.compile(
    r"\b(?:place of birth|birthplace|born|die|died|death|where did|where was|"
    r"where were)\b",
    re.I,
)
FULL_LOCATION_SURFACE_QUERY_RE = re.compile(
    r"\b(?:based|headquarters|located|situated|headquartered|where is|where are)\b",
    re.I,
)
ALIAS_NAME_QUERY_RE = re.compile(
    r"\b(?:also known|known as|alternate(?:ive)? name|other name|what name)\b",
    re.I,
)
COORDINATED_OR_RANGE_SURFACE_RE = re.compile(
    r"\b(?:between|from|through|until|and|or)\b|[-–—]|/",
    re.I,
)
NATIONALITY_ADJECTIVE_KEYS = {
    "afghan", "albanian", "algerian", "american", "argentine", "armenian", "australian",
    "austrian", "belgian", "brazilian", "british", "bulgarian", "canadian", "chilean",
    "chinese", "colombian", "croatian", "cuban", "czech", "danish", "dutch", "egyptian",
    "english", "finnish", "french", "german", "ghanaian", "greek", "hungarian", "indian",
    "indonesian", "iranian", "irish", "israeli", "italian", "jamaican", "japanese",
    "korean", "malaysian", "mexican", "moroccan", "nigerian", "norwegian", "pakistani",
    "polish", "portuguese", "romanian", "russian", "scottish", "serbian", "singaporean",
    "spanish", "swedish", "swiss", "syrian", "taiwanese", "thai", "turkish",
    "ukrainian", "vietnamese", "welsh",
}
GENERIC_HEAD_KEYS = {
    "american", "comics", "vertigo", "appliances", "artist", "director", "producer",
    "writer", "actor", "actress", "model", "city", "country", "state", "team", "band",
    "album", "film", "book", "company", "organization", "novel", "series",
}
OCCUPATION_MODIFIER_KEYS = {
    "film", "movie", "manga", "screen", "fashion", "baseball", "basketball", "football",
    "racing", "television", "tv", "jazz", "rock", "opera", "stage",
}
DATE_NUMBER_TYPES = {"date", "number", "year"}
SUPPORTED_TARGET_TYPES = {
    "date",
    "number",
    "year",
    "occupation",
    "nationality",
    "person",
    "location",
    "organization",
    "work",
}
STRICT_ABSTAIN_KEYS = {
    "unknown",
    "not specified",
    "not provided",
    "not mentioned",
    "cannot determine",
    "cannot be determined",
    "insufficient information",
    "no information",
    "not available",
    "no answer",
    "none",
    "n a",
    "na",
    "i dont know",
    "i do not know",
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))


def _question_contains_candidate(question: str, candidate: str, target_type: str) -> bool:
    cand_norm = normalize_answer(candidate or "")
    if not cand_norm:
        return False
    if target_type in DATE_NUMBER_TYPES:
        return False
    q_norm = normalize_answer(question or "")
    if len(cand_norm.split()) <= 1 and len(cand_norm) <= 3:
        return False
    return cand_norm in q_norm


def _content_tokens(text: str) -> List[str]:
    return [tok for tok in WORD_RE.findall(normalize_answer(text or "")) if len(tok) > 1]


def _candidate_evidence_text(candidate: Dict[str, Any]) -> str:
    examples = candidate.get("examples") or []
    if isinstance(examples, (list, tuple)):
        example_text = " ".join(str(item or "") for item in examples[:3])
    else:
        example_text = str(examples or "")
    return " ".join(
        str(part or "")
        for part in (
            candidate.get("title"),
            candidate.get("sentence_preview"),
            candidate.get("lineage_source_title"),
            candidate.get("lineage_sentence_preview"),
            example_text,
        )
    )


def _candidate_is_high_level_location(candidate: str) -> bool:
    norm = normalize_answer(candidate or "")
    return norm in HIGH_LEVEL_LOCATION_KEYS


def _looks_like_title_metadata(candidate: str) -> bool:
    text = str(candidate or "").strip()
    return bool(re.search(r"\s[-–]\s|\([^)]*\)|\b(?:early life|career|discography|filmography)\b", text, re.I))


def _looks_like_person_surface_for_city(candidate: str) -> bool:
    text = str(candidate or "").strip()
    tokens = _content_tokens(text)
    if len(tokens) < 2:
        return False
    if LOCATION_NOUN_RE.search(text) or _candidate_is_high_level_location(text) or CITY_PREFIX_RE.search(text):
        return False
    return bool(re.fullmatch(r"(?:[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:\s+|$)){2,5}", text))


def _has_full_date_value(text: str) -> bool:
    return bool(FULL_DATE_VALUE_RE.search(str(text or "")))


def _date_key(text: str) -> Tuple[str, str, str]:
    surface = str(text or "").lower()
    year_match = re.search(r"\b(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", surface)
    month_match = MONTH_RE.search(surface)
    day_match = re.search(r"\b([0-3]?\d)(?:st|nd|rd|th)?\b", surface)
    return (
        year_match.group(0) if year_match else "",
        month_match.group(0)[:3] if month_match else "",
        day_match.group(1).lstrip("0") if day_match else "",
    )


def _same_date_value(left: str, right: str) -> bool:
    left_key = _date_key(left)
    right_key = _date_key(right)
    if not any(left_key) or not any(right_key):
        return False
    return left_key == right_key


def _is_strict_abstain_surface(answer: Any) -> bool:
    text = str(answer or "").strip()
    key = normalize_answer(text)
    if key in STRICT_ABSTAIN_KEYS:
        return True
    return bool(
        re.fullmatch(
            r"\s*(?:n/?a|none|unknown|not\s+(?:specified|provided|mentioned|available)|"
            r"cannot\s+(?:determine|be\s+determined)|insufficient\s+information|"
            r"no\s+(?:information|answer)|i\s+(?:do\s+not|don't)\s+know)\s*\.?\s*",
            text,
            flags=re.I,
        )
    )


def _surface_rewrite_loss_reason(
    *,
    question: str,
    before: str,
    after: str,
    slot: str,
    target_type: str,
) -> str:
    """Return a reason when a candidate rewrite drops answer-bearing surface."""
    if not before or not after or _same_answer(before, after):
        return ""
    before_norm = normalize_answer(before)
    after_norm = normalize_answer(after)
    if not before_norm or not after_norm:
        return "empty_surface"
    if ALIAS_NAME_QUERY_RE.search(question or ""):
        alias_parts = [
            normalize_answer(part)
            for part in re.split(r"\s*,?\s+or\s+|\s*;\s*", str(before or ""), flags=re.I)
            if normalize_answer(part)
        ]
        if after_norm in alias_parts:
            return ""

    temporal_slot = slot in {"full_date", "date", "year"} or target_type in DATE_NUMBER_TYPES
    if temporal_slot:
        if _same_date_value(before, after):
            return ""
        if _has_full_date_value(before) and YEAR_ONLY_RE.match(after):
            return "temporal_granularity_loss"
        if _is_containment_variant(before, after) and COORDINATED_OR_RANGE_SURFACE_RE.search(str(before or "")):
            return "temporal_range_truncation"

    if not _is_containment_variant(before, after):
        return ""

    before_tokens = _content_tokens(before)
    after_tokens = _content_tokens(after)
    if not before_tokens or not after_tokens:
        return ""

    if slot in {"city", "county", "region", "location"} or target_type == "location":
        if "," in str(before or ""):
            parts = [
                normalize_answer(part)
                for part in str(before or "").split(",")
                if normalize_answer(part)
            ]
            first_part = parts[0] if parts else ""
            if after_norm != first_part:
                return "location_parent_truncation"
            if FULL_LOCATION_SURFACE_QUERY_RE.search(question or "") and not LOCATION_CITY_GRANULARITY_QUERY_RE.search(question or ""):
                return "location_full_surface_required"
        if _candidate_is_high_level_location(after) and not _candidate_is_high_level_location(before):
            return "location_granularity_loss"

    if COORDINATED_OR_RANGE_SURFACE_RE.search(str(before or "")) and len(after_tokens) < len(before_tokens):
        return "coordinated_answer_truncation"

    short_entity_slot = (
        target_type in {"person", "organization", "work"}
        or slot in {"named_entity", "organization", "institution", "work", "role_title"}
        or slot.startswith("person_")
    )
    if short_entity_slot and len(before_tokens) <= 6 and len(after_tokens) < len(before_tokens):
        return "short_entity_truncation"

    return ""


def _is_single_hop_attribute_query(question: str, slot: str) -> bool:
    question_text = str(question or "")
    if slot in PERSON_ROLE_EVIDENCE_PATTERNS:
        return False
    if slot in {"institution", "organization"} and re.search(r"\b(?:work(?:s|ed)?\s+(?:at|for)|manufacturer|company|publisher|founded)\b", question_text, re.I):
        return not BRIDGE_QUERY_RE.search(question_text)
    if slot in {"city", "region", "location", "nationality", "occupation", "full_date", "year"}:
        return bool(SINGLE_HOP_ATTRIBUTE_RE.search(question_text)) and not BRIDGE_QUERY_RE.search(question_text)
    if slot == "person" and re.search(r"^\s*who\b", question_text, re.I):
        return bool(SINGLE_HOP_ATTRIBUTE_RE.search(question_text)) and not BRIDGE_QUERY_RE.search(question_text)
    return False


def _candidate_target_type_for_slot(slot: str, fallback: str) -> str:
    if slot in {"city", "county", "region", "location", "intersection"}:
        return "location"
    if slot in {"institution", "organization"}:
        return "organization"
    if slot in {"full_date", "date", "year"}:
        return "date"
    if slot == "number":
        return "number"
    if slot in PERSON_ROLE_EVIDENCE_PATTERNS:
        return "person"
    if slot == "occupation":
        return "occupation"
    if slot == "role_title":
        return "person"
    return fallback


def _infer_answer_slot(question: str, target_type: str) -> str:
    q = normalize_answer(question or "")
    leading_person_question = bool(re.match(r"^(?:who|whose)\b", q))
    if re.match(r"^(?:how many|how much|how tall|how long|how old)\b", q):
        return "number"
    if re.match(
        r"^(?:what|which)\b.{0,48}\b(?:salary|salaries|income|earnings|wage|wages|pay)\b",
        q,
    ):
        return "number"
    if re.search(r"\b(?:date of birth|birthdate|birthday|born on|date of death|died on)\b", q):
        return "full_date"
    if q.startswith("when "):
        return "year" if re.search(r"\b(?:what|which|in what|in which)\s+year\b", q) else "date"
    if leading_person_question:
        if "mother" in q or "maternal" in q:
            return "person_mother"
        if "father" in q or "paternal" in q:
            return "person_father"
        if "spouse" in q or "wife" in q or "husband" in q:
            return "person_spouse"
        if " child " in f" {q} " or " children " in f" {q} " or " son " in f" {q} " or " daughter " in f" {q} ":
            return "person_child"
        return "person"
    if re.search(r"\b(?:what|which|in what|in which)\s+year\b", q):
        return "year"
    if "occupation" in q or "profession" in q:
        return "occupation"
    if "nationality" in q or re.search(r"\b(?:citizen|citizenship)\b", q):
        return "nationality"
    if target_type == "work":
        return "work"
    if target_type == "organization":
        return "organization"
    if target_type == "number":
        return "number"
    if re.search(r"\b(?:position|office|role|title)\b", q):
        return "role_title"
    if "species" in q or "within its genus" in q or "binomial" in q:
        return "species"
    if (
        "graduate from" in q
        or "graduated from" in q
        or "alma mater" in q
        or "studied at" in q
        or "study at" in q
        or "educated at" in q
        or "attended" in q
        or "work at" in q
        or "works at" in q
        or "worked at" in q
        or "work for" in q
        or "works for" in q
        or "worked for" in q
        or "employed by" in q
        or re.search(r"\bwhere\b.{0,80}\b(?:study|studied|attend(?:ed)?|educated)\b", q)
        or re.search(r"\b(?:which|what)\s+(?:school|college|university|institution)\b", q)
        or re.search(r"\bwhere\b.{0,80}\bwork(?:s|ed)?\b", q)
    ):
        return "institution"
    if re.search(r"\b(?:manufacturer|company|corporation|organization|organisation|publisher|press)\b", q):
        return "organization"
    if "intersection" in q:
        return "intersection"
    if re.search(r"\b(?:what|which|in what|in which)\s+county\b", q) or " county " in f" {q} ":
        return "county"
    if re.search(r"\b(?:what|which|in what|in which)\s+(?:state|province|district|region)\b", q):
        return "region"
    if re.search(r"\b(?:what|which|in what|in which)\s+city\b", q):
        return "city"
    if re.search(r"\b(?:what|which|in what|in which)\s+country\b", q):
        return "location"
    if (
        "place of birth" in q
        or "birthplace" in q
        or "born in" in q
        or "located" in q
        or q.startswith("where ")
    ):
        return "location"
    if "mother" in q or "maternal" in q:
        return "person_mother"
    if "father" in q or "paternal" in q:
        return "person_father"
    if "spouse" in q or "wife" in q or "husband" in q:
        return "person_spouse"
    if " child " in f" {q} " or " children " in f" {q} " or " son " in f" {q} " or " daughter " in f" {q} ":
        return "person_child"
    if "language family" in q or "family of languages" in q:
        return "class_phrase"
    if target_type in DATE_NUMBER_TYPES:
        return target_type
    if q.startswith("who ") or " paternal " in q or " maternal " in q or "sibling" in q:
        return "person"
    if any(term in q for term in ("company", "team", "band", "album", "film", "book", "novel", "name of")):
        return "named_entity"
    return target_type if target_type in SUPPORTED_TARGET_TYPES else "named_entity"


def _infer_answer_arity(question: str, answer: str, target_type: str) -> str:
    q = normalize_answer(question or "")
    if re.search(r"\b(?:who|what|which)\s+are\b", q):
        return "multi"
    if re.search(
        r"\b(?:people|persons|players|teams|countries|states|cities|members|"
        r"children|sons|daughters|companies|organizations|organisations|films|"
        r"movies|books|albums|schools|universities)\s+"
        r"(?:who|that|which|with|without|have|has|had|are|were)\b",
        q,
    ):
        return "multi"
    if any(term in q for term in (" cast ", " characters ", " members ", "names of", "list of")):
        return "multi"
    if re.search(r"\band\s+(?:when|where|what|who|which)\b", q):
        return "multi_slot"
    if target_type != "location":
        answer_text = str(answer or "")
        if ";" in answer_text or re.search(r"\s+(?:or|and)\s+", answer_text, re.I):
            return "multi_value"
        if answer_text.count(",") >= 2:
            return "multi_value"
    return "single"


def _candidate_passes_answer_slot(
    candidate: str,
    slot: str,
    target_type: str,
    question: str = "",
    evidence_text: str = "",
) -> Tuple[bool, str]:
    candidate = str(candidate or "").strip()
    norm = normalize_answer(candidate)
    tokens = _content_tokens(candidate)
    if not norm:
        return False, "empty_candidate"
    if _is_strict_abstain_surface(candidate):
        return False, "abstain_candidate"
    if slot == "full_date":
        if not _has_full_date_value(candidate):
            return False, "full_date_slot_required"
        return True, ""
    if slot == "date":
        if not (YEAR_ONLY_RE.search(candidate) or MONTH_RE.search(candidate) or _has_full_date_value(candidate)):
            return False, "date_slot_required"
        return True, ""
    if slot == "year":
        if not YEAR_ONLY_RE.match(candidate):
            return False, "year_slot_required"
        return True, ""
    if slot == "number":
        if not re.search(r"\d", candidate):
            return False, "number_slot_required"
        return True, ""
    if slot == "location":
        if _looks_like_title_metadata(candidate):
            return False, "location_slot_title_metadata"
        if norm in NATIONALITY_ADJECTIVE_KEYS:
            return False, "location_slot_nationality_adjective"
        if OCCUPATION_HEAD_RE.fullmatch(candidate.strip()):
            return False, "location_slot_occupation_head"
        if norm in GENERIC_HEAD_KEYS:
            return False, "location_slot_generic_head"
        if ORGANIZATION_RE.search(candidate) and not LOCATION_NOUN_RE.search(candidate):
            return False, "location_slot_organization_surface"
    elif slot == "city":
        if _looks_like_title_metadata(candidate):
            return False, "city_slot_title_metadata"
        if _looks_like_person_surface_for_city(candidate):
            return False, "city_slot_person_surface"
        if norm in NATIONALITY_ADJECTIVE_KEYS:
            return False, "city_slot_nationality_adjective"
        if norm in GENERIC_HEAD_KEYS:
            return False, "city_slot_generic_head"
        if YEAR_ONLY_RE.match(candidate) or _has_full_date_value(candidate):
            return False, "city_slot_temporal_candidate"
        if ORGANIZATION_RE.search(candidate) or INSTITUTION_RE.search(candidate):
            return False, "city_slot_organization_surface"
        if OCCUPATION_HEAD_RE.fullmatch(candidate.strip()):
            return False, "city_slot_occupation_head"
    elif slot == "county":
        if not COUNTY_RE.search(candidate):
            return False, "county_slot_required"
    elif slot == "region":
        if ORGANIZATION_RE.search(candidate) and not REGION_RE.search(candidate):
            return False, "region_slot_organization_surface"
        if not (REGION_RE.search(candidate) or LOCATION_NOUN_RE.search(candidate)):
            return False, "region_slot_required"
    elif slot == "institution":
        if not (INSTITUTION_RE.search(candidate) or ORGANIZATION_RE.search(candidate)):
            return False, "institution_slot_required"
    elif slot == "organization":
        if not (ORGANIZATION_RE.search(candidate) or INSTITUTION_RE.search(candidate)):
            return False, "organization_slot_required"
    elif slot == "intersection":
        if not INTERSECTION_RE.search(candidate):
            return False, "intersection_slot_required"
    elif slot == "occupation":
        if not OCCUPATION_HEAD_RE.search(candidate):
            return False, "occupation_slot_required"
    elif slot == "role_title":
        if not (ROLE_TITLE_RE.search(candidate) or OCCUPATION_HEAD_RE.search(candidate)):
            return False, "role_title_slot_required"
    elif slot == "species":
        if not LATIN_SPECIES_RE.match(candidate.strip()):
            return False, "species_slot_binomial_required"
    elif slot in PERSON_ROLE_EVIDENCE_PATTERNS:
        if norm in NATIONALITY_ADJECTIVE_KEYS:
            return False, f"{slot}_nationality_adjective"
        if ORGANIZATION_RE.search(candidate) and not ROLE_TITLE_RE.search(candidate):
            return False, f"{slot}_organization_surface"
        evidence_pattern = PERSON_ROLE_EVIDENCE_PATTERNS[slot]
        if evidence_text and not evidence_pattern.search(evidence_text):
            return False, f"{slot}_missing_role_evidence"
    elif slot in {"person", "named_entity", "organization", "work"}:
        if norm in NATIONALITY_ADJECTIVE_KEYS:
            return False, f"{slot}_slot_nationality_adjective"
        if norm in GENERIC_HEAD_KEYS and len(tokens) <= 1:
            return False, f"{slot}_slot_generic_head"
    return True, ""


def _has_material_information_loss(before: str, after: str, slot: str, target_type: str) -> bool:
    if not _is_containment_variant(before, after):
        return False
    before_tokens = _content_tokens(before)
    after_tokens = set(_content_tokens(after))
    if not before_tokens or not after_tokens:
        return False
    lost = [tok for tok in before_tokens if tok not in after_tokens]
    if not lost:
        return False
    if slot == "species" and len(before_tokens) >= 2 and len(after_tokens) <= 1:
        return True
    if slot == "occupation" and any(tok in OCCUPATION_MODIFIER_KEYS for tok in lost):
        return True
    if target_type == "person" and len(before_tokens) >= 2 and len(after_tokens) <= 1:
        return True
    if slot in {"named_entity", "organization", "work", "institution"}:
        if len(before_tokens) >= 2 and len(after_tokens) <= 1:
            return True
    if len(before_tokens) <= 4 and len(after_tokens) < len(before_tokens):
        return True
    return False


def _incumbent_slot_for_check(slot: str) -> str:
    if slot in PERSON_ROLE_EVIDENCE_PATTERNS:
        return "person"
    return slot


def _incumbent_is_slot_compatible(answer: str, slot: str, target_type: str) -> bool:
    check_slot = _incumbent_slot_for_check(slot)
    ok, _ = _candidate_passes_answer_slot(answer, check_slot, target_type)
    return ok


def _is_unsafe_temporal_rewrite(before: str, after: str, slot: str, target_type: str) -> bool:
    if slot not in {"full_date", "date", "year"} and target_type not in DATE_NUMBER_TYPES:
        return False
    if _has_full_date_value(before):
        if YEAR_ONLY_RE.match(after):
            return True
        if _has_full_date_value(after) and not _same_date_value(before, after):
            return True
    if MONTH_RE.search(before or "") and YEAR_ONLY_RE.match(after):
        return True
    return False


def _is_unsafe_slot_granularity_rewrite(before: str, after: str, slot: str, target_type: str) -> bool:
    if not before or not after:
        return False
    if slot in {"city", "county", "region", "location"}:
        if _candidate_is_high_level_location(after) and not _candidate_is_high_level_location(before):
            return True
    if slot in {"institution", "organization"}:
        before_ok, _ = _candidate_passes_answer_slot(before, slot, target_type)
        after_ok, _ = _candidate_passes_answer_slot(after, slot, target_type)
        if before_ok and not after_ok:
            return True
    if slot == "role_title":
        before_ok, _ = _candidate_passes_answer_slot(before, slot, target_type)
        after_ok, _ = _candidate_passes_answer_slot(after, slot, target_type)
        if before_ok and not after_ok:
            return True
    return False


def _format_repair_slot_allowed(slot: str, target_type: str) -> bool:
    if slot in {
        "occupation",
        "nationality",
        "institution",
        "organization",
        "county",
        "region",
        "city",
        "location",
        "intersection",
        "species",
        "role_title",
        "full_date",
        "year",
    }:
        return True
    if slot in PERSON_ROLE_EVIDENCE_PATTERNS:
        return True
    return target_type in DATE_NUMBER_TYPES


def _is_safe_format_projection(before: str, after: str, slot: str, target_type: str) -> bool:
    if _is_strict_abstain_surface(before):
        return True
    if not before or not after:
        return False
    if _same_answer(before, after):
        return True

    temporal_slot = slot in {"full_date", "date", "year"} or target_type in DATE_NUMBER_TYPES
    if temporal_slot:
        return _same_date_value(before, after)

    if not _is_containment_variant(before, after):
        return False
    if _token_count(after) > max(1, _token_count(before)):
        return False

    before_tokens = set(_content_tokens(before))
    after_tokens = set(_content_tokens(after))
    if not before_tokens or not after_tokens:
        return False
    if not after_tokens.issubset(before_tokens):
        overlap = len(before_tokens.intersection(after_tokens)) / max(1, len(after_tokens))
        if overlap < 0.86:
            return False

    if slot in {"city", "county", "region", "location"} or target_type == "location":
        if _candidate_is_high_level_location(after) and not _candidate_is_high_level_location(before):
            return False
    return True


def _candidate_rows_by_key(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = _candidate_key(str(row.get("candidate", "")))
        if not key:
            continue
        if key not in by_key or float(row.get("table_score", 0.0) or 0.0) > float(by_key[key].get("table_score", 0.0) or 0.0):
            by_key[key] = row
    return by_key


def _collect_source_counts(candidates: Sequence[Dict[str, Any]]) -> Tuple[Counter, Dict[str, List[str]]]:
    source_counts: Counter = Counter()
    examples: Dict[str, List[str]] = defaultdict(list)
    seen_sources: Dict[str, set] = defaultdict(set)
    for cand in candidates:
        key = _candidate_key(str(cand.get("candidate", "")))
        if not key:
            continue
        title = str(cand.get("title", "") or "")
        source = title or str(cand.get("doc_idx", "") or "")
        if source and source not in seen_sources[key]:
            seen_sources[key].add(source)
            source_counts[key] += 1
        preview = str(cand.get("sentence_preview", "") or "")
        if preview and len(examples[key]) < 3:
            examples[key].append(preview)
    return source_counts, examples


def _answer_state(
    answer: str,
    answer_supported: bool,
    answer_support_score: float,
    max_answer_tokens: int,
) -> Dict[str, Any]:
    answer = answer or ""
    strict_abstain = _is_strict_abstain_surface(answer)
    return {
        "abstain": strict_abstain,
        "abstain_like": bool(ABSTAIN_RE.search(answer)),
        "verbose_or_multi": _looks_verbose_or_multi_answer(answer, max_answer_tokens),
        "supported": bool(answer_supported),
        "support_score": round(float(answer_support_score or 0.0), 4),
        "weak": (not answer_supported) or float(answer_support_score or 0.0) < 0.34,
        "token_count": _token_count(answer),
    }


def _score_candidates(
    *,
    question: str,
    current_answer: str,
    source_docs: Sequence[str],
    target_type: str,
    filtered_candidates: Sequence[Dict[str, Any]],
    raw_candidates: Sequence[Dict[str, Any]],
    max_docs: int,
    min_question_overlap: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = _build_typed_table(list(filtered_candidates))
    row_by_key = _candidate_rows_by_key(rows)
    source_counts, examples = _collect_source_counts(filtered_candidates)
    raw_source_counts, raw_examples = _collect_source_counts(raw_candidates)

    scored: List[Dict[str, Any]] = []
    seen = set()
    for cand in filtered_candidates:
        candidate = str(cand.get("candidate", "") or "").strip()
        key = _candidate_key(candidate)
        if not candidate or not key or key in seen:
            continue
        seen.add(key)

        row = row_by_key.get(key, {})
        supported, support_score = _is_supported(question, candidate, source_docs, target_type, max_docs=max_docs)
        relation_score = _clamp01(cand.get("relation_score", cand.get("relation_bonus", 0.0)))
        question_overlap = _clamp01(cand.get("question_overlap", 0.0))
        type_score = _clamp01(cand.get("target_signal", 0.0))
        provenance_count = int(max(
            int(cand.get("frequency", 1) or 1),
            int(row.get("provenance_count", 0) or 0),
            int(source_counts.get(key, 0) or 0),
            int(raw_source_counts.get(key, 0) or 0),
        ))
        source_count = int(max(int(source_counts.get(key, 0) or 0), int(raw_source_counts.get(key, 0) or 0)))

        risk = 0.0
        if _question_contains_candidate(question, candidate, target_type):
            risk += 0.28
        if question_overlap < min_question_overlap and relation_score < 0.05:
            risk += 0.08
        if _same_answer(candidate, current_answer):
            risk += 0.20

        base_score = max(
            float(cand.get("confidence", 0.0) or 0.0),
            float(row.get("table_score", 0.0) or 0.0),
            float(support_score or 0.0),
        )
        provenance_bonus = min(0.10, 0.035 * max(0, provenance_count - 1))
        source_bonus = min(0.06, 0.025 * max(0, source_count - 1))
        role_bonus = 0.06 * relation_score + 0.03 * type_score
        evidence_score = _clamp01(base_score + provenance_bonus + source_bonus + role_bonus - risk)

        scored.append({
            "candidate": candidate,
            "key": key,
            "score": round(evidence_score, 4),
            "base_score": round(float(base_score or 0.0), 4),
            "support_score": round(float(support_score or 0.0), 4),
            "supported": bool(supported),
            "type_score": round(type_score, 4),
            "relation_score": round(relation_score, 4),
            "question_overlap": round(question_overlap, 4),
            "provenance_count": provenance_count,
            "source_count": source_count,
            "risk": round(risk, 4),
            "examples": examples.get(key) or raw_examples.get(key) or [],
        })

    scored.sort(key=lambda x: (
        float(x.get("score", 0.0) or 0.0),
        int(x.get("source_count", 0) or 0),
        int(x.get("provenance_count", 0) or 0),
    ), reverse=True)

    return scored, {
        "typed_rows": rows[:10],
        "source_count_keys": len(source_counts),
        "raw_source_count_keys": len(raw_source_counts),
    }


def arbitrate_answer_with_evidence_grounded_extractor(
    *,
    question: str,
    answer: str,
    qa_docs: Sequence[str],
    enabled: bool = False,
    apply_rewrite: bool = True,
    min_score: float = 0.84,
    generic_min_score: float = 0.84,
    min_margin: float = 0.08,
    min_support_gain: float = 0.06,
    alias_min_support_gain: float = 0.01,
    min_provenance: int = 1,
    min_question_overlap: float = 0.02,
    max_docs: int = 8,
    max_answer_tokens: int = 14,
    max_candidates: int = 12,
    require_role_constraint: bool = True,
    skip_supported_answers: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    before = answer or ""
    if not enabled:
        return before, {
            "enabled": False,
            "changed": False,
            "action": "disabled",
            "reason": "disabled",
            "before": before,
            "after": before,
        }

    source_docs = [doc for doc in (qa_docs or []) if isinstance(doc, str) and doc.strip() and not _is_synthetic_doc(doc)]
    target_type = infer_answer_target_type(question or "")
    candidate_target_type = "date" if target_type == "year" else target_type
    candidate_target_type = candidate_target_type if candidate_target_type in SUPPORTED_TARGET_TYPES else "person"
    answer_slot = _infer_answer_slot(question or "", candidate_target_type)
    candidate_target_type = _candidate_target_type_for_slot(answer_slot, candidate_target_type)

    meta: Dict[str, Any] = {
        "enabled": True,
        "changed": False,
        "action": "keep",
        "reason": "init",
        "before": before,
        "after": before,
        "target_type": target_type,
        "candidate_target_type": candidate_target_type,
        "source_doc_count": len(source_docs),
    }

    if not source_docs:
        meta["reason"] = "no_source_docs"
        return before, meta

    if BOOLEAN_ANSWER_RE.match(before) or BOOLEAN_QUESTION_RE.match(question or ""):
        meta["reason"] = "boolean_protected"
        return before, meta

    answer_supported, answer_support_score = _is_supported(question, before, source_docs, candidate_target_type, max_docs=max_docs)
    state = _answer_state(before, answer_supported, answer_support_score, max_answer_tokens)
    meta["answer_state"] = state
    answer_arity = _infer_answer_arity(question or "", before, candidate_target_type)
    meta["answer_slot"] = answer_slot
    meta["answer_arity"] = answer_arity
    if answer_arity != "single":
        meta["reason"] = "multi_answer_projection_protected"
        return before, meta

    raw_candidates = _rank_candidates(question, before, source_docs, candidate_target_type, max_docs=max_docs)
    meta["raw_candidate_count"] = len(raw_candidates)
    if not raw_candidates:
        meta["reason"] = "no_raw_candidates"
        return before, meta

    filtered_candidates, reject_counts = _filter_candidates(
        raw_candidates,
        question=question,
        target_type=candidate_target_type,
        min_question_overlap=min_question_overlap,
        require_role_constraint=require_role_constraint,
    )
    meta["candidate_reject_counts"] = dict(reject_counts)
    meta["candidate_count"] = len(filtered_candidates)
    if not filtered_candidates:
        meta["reason"] = "no_filtered_candidates"
        return before, meta

    scored, score_meta = _score_candidates(
        question=question,
        current_answer=before,
        source_docs=source_docs,
        target_type=candidate_target_type,
        filtered_candidates=filtered_candidates[: max(1, max_candidates * 4)],
        raw_candidates=raw_candidates[: max(1, max_candidates * 6)],
        max_docs=max_docs,
        min_question_overlap=min_question_overlap,
    )
    scored = scored[:max(1, max_candidates)]
    meta["store_size"] = len(scored)
    meta.update(score_meta)
    if not scored:
        meta["reason"] = "no_scored_candidates"
        return before, meta
    slot_filtered: List[Dict[str, Any]] = []
    slot_reject_counts: Counter = Counter()
    for candidate in scored:
        ok, reason = _candidate_passes_answer_slot(
            str(candidate.get("candidate", "") or ""),
            answer_slot,
            candidate_target_type,
            question=question or "",
            evidence_text=_candidate_evidence_text(candidate),
        )
        if ok:
            slot_filtered.append(candidate)
        else:
            slot_reject_counts[reason] += 1
    meta["slot_reject_counts"] = dict(slot_reject_counts)
    meta["slot_filtered_count"] = len(slot_filtered)
    if not slot_filtered:
        meta["reason"] = "slot_no_candidates"
        return before, meta
    scored = slot_filtered[:max(1, max_candidates)]

    top = scored[0]
    second = scored[1] if len(scored) > 1 else None
    margin = float(top.get("score", 0.0) or 0.0) - float((second or {}).get("score", 0.0) or 0.0)
    support_gain = float(top.get("support_score", 0.0) or 0.0) - float(answer_support_score or 0.0)
    after = str(top.get("candidate", "") or "").strip()
    same = _same_answer(before, after)
    containment_alias = bool(_is_containment_variant(before, after) and _token_count(after) <= max(1, _token_count(before)))
    lossy_alias = bool(containment_alias and _has_material_information_loss(before, after, answer_slot, candidate_target_type))
    surface_loss_reason = _surface_rewrite_loss_reason(
        question=question or "",
        before=before,
        after=after,
        slot=answer_slot,
        target_type=candidate_target_type,
    )
    if surface_loss_reason:
        lossy_alias = True
    verbose_repair = bool(state.get("abstain") or ((not state.get("supported")) and state.get("weak")))
    weak_repair = bool(state.get("weak"))
    incumbent_slot_compatible = _incumbent_is_slot_compatible(before, answer_slot, candidate_target_type)
    unsafe_temporal_rewrite = _is_unsafe_temporal_rewrite(before, after, answer_slot, candidate_target_type)
    unsafe_slot_granularity_rewrite = _is_unsafe_slot_granularity_rewrite(
        before,
        after,
        answer_slot,
        candidate_target_type,
    )
    single_hop_attribute_query = _is_single_hop_attribute_query(question or "", answer_slot)
    if state.get("abstain") and single_hop_attribute_query:
        unsafe_slot_granularity_rewrite = False

    floor = float(min_score if candidate_target_type in DATE_NUMBER_TYPES else generic_min_score)
    if state.get("abstain") and single_hop_attribute_query and answer_slot in {"person", "date", "number"}:
        floor = min(floor, 0.48)
    elif state.get("abstain") and (single_hop_attribute_query or answer_slot in {"institution", "intersection", "species", "location"}):
        floor = min(floor, 0.55)
    elif state.get("abstain"):
        floor = min(floor, 0.68)
    format_repair_allowed = _format_repair_slot_allowed(answer_slot, candidate_target_type)
    safe_format_projection = bool(
        not surface_loss_reason
        and _is_safe_format_projection(before, after, answer_slot, candidate_target_type)
    )
    incumbent_needs_format_repair = bool(
        state.get("abstain")
        or state.get("verbose_or_multi")
        or not incumbent_slot_compatible
        or (
            candidate_target_type in DATE_NUMBER_TYPES
            and answer_slot in {"full_date", "date", "year"}
            and not re.search(r"\d", before or "")
        )
    )
    is_alias_repair = (
        containment_alias
        and not lossy_alias
        and float(top.get("score", 0.0) or 0.0) >= max(0.0, floor - 0.04)
        and support_gain >= float(alias_min_support_gain)
    )
    strong_repair = (
        float(top.get("score", 0.0) or 0.0) >= floor
        and margin >= float(min_margin)
        and support_gain >= float(min_support_gain)
        and weak_repair
    )
    format_repair = (
        format_repair_allowed
        and verbose_repair
        and incumbent_needs_format_repair
        and safe_format_projection
        and not unsafe_temporal_rewrite
        and not unsafe_slot_granularity_rewrite
        and float(top.get("score", 0.0) or 0.0) >= max(0.0, floor - 0.04)
        and support_gain >= float(min_support_gain)
    )

    meta.update({
        "selected_candidate": after,
        "selected_score": round(float(top.get("score", 0.0) or 0.0), 4),
        "selected_support_score": round(float(top.get("support_score", 0.0) or 0.0), 4),
        "selected_provenance_count": int(top.get("provenance_count", 0) or 0),
        "selected_source_count": int(top.get("source_count", 0) or 0),
        "selected_question_overlap": round(float(top.get("question_overlap", 0.0) or 0.0), 4),
        "selected_relation_score": round(float(top.get("relation_score", 0.0) or 0.0), 4),
        "margin": round(margin, 4),
        "support_gain": round(support_gain, 4),
        "same_as_current": same,
        "containment_alias": containment_alias,
        "lossy_alias": lossy_alias,
        "surface_loss_reason": surface_loss_reason,
        "verbose_repair": verbose_repair,
        "weak_repair": weak_repair,
        "incumbent_slot_compatible": incumbent_slot_compatible,
        "format_repair_allowed": format_repair_allowed,
        "safe_format_projection": safe_format_projection,
        "incumbent_needs_format_repair": incumbent_needs_format_repair,
        "unsafe_temporal_rewrite": unsafe_temporal_rewrite,
        "unsafe_slot_granularity_rewrite": unsafe_slot_granularity_rewrite,
        "single_hop_attribute_query": single_hop_attribute_query,
        "top_candidates": scored[:5],
    })

    if same:
        meta["reason"] = "same_as_current"
        return before, meta
    if not after:
        meta["reason"] = "empty_candidate"
        return before, meta
    if _token_count(after) > max_answer_tokens and candidate_target_type not in DATE_NUMBER_TYPES:
        meta["reason"] = "candidate_too_long"
        return before, meta
    if int(top.get("provenance_count", 0) or 0) < int(min_provenance):
        meta["reason"] = "insufficient_provenance"
        return before, meta
    if _question_contains_candidate(question or "", after, candidate_target_type):
        meta["reason"] = "question_anchor_candidate"
        return before, meta
    if surface_loss_reason:
        meta["reason"] = surface_loss_reason
        return before, meta
    if lossy_alias:
        meta["reason"] = "lossy_alias_protected"
        return before, meta
    if unsafe_temporal_rewrite:
        meta["reason"] = "unsafe_temporal_rewrite_protected"
        return before, meta
    if unsafe_slot_granularity_rewrite:
        meta["reason"] = "unsafe_slot_granularity_protected"
        return before, meta
    if incumbent_slot_compatible and state.get("supported") and not is_alias_repair:
        meta["reason"] = "slot_compatible_supported_answer_protected"
        return before, meta
    if skip_supported_answers and state.get("supported") and not (is_alias_repair or format_repair):
        meta["reason"] = "supported_answer_protected"
        return before, meta
    if format_repair_allowed and verbose_repair and incumbent_needs_format_repair and not safe_format_projection:
        meta["reason"] = "unsafe_format_projection_protected"
        return before, meta
    if not (strong_repair or is_alias_repair or format_repair or weak_repair and strong_repair):
        meta["reason"] = "below_rewrite_threshold"
        return before, meta

    rewrite_reason = (
        "alias_repair" if is_alias_repair
        else "format_repair" if format_repair
        else "unsupported_answer_repair"
    )
    meta["reason"] = rewrite_reason
    risky_unsupported_rewrite = rewrite_reason == "unsupported_answer_repair"
    meta["action"] = "rewrite" if apply_rewrite and not risky_unsupported_rewrite else "shadow"
    if apply_rewrite and not risky_unsupported_rewrite:
        meta["changed"] = True
        meta["after"] = after
        return after, meta
    if risky_unsupported_rewrite:
        meta["reason"] = "unsupported_answer_shadow_only"
        meta["shadow_changed"] = True
        meta["shadow_after"] = after
        meta["shadow_reason"] = rewrite_reason
        return before, meta
    meta["after"] = before
    return before, meta
