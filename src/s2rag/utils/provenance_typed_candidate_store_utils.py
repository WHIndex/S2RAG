import re
from typing import Any, Dict, List, Optional, Tuple

from .answer_verifier_utils import _is_supported, _rank_candidates
from .candidate_answer_selector_utils import _has_main_numeric_temporal_request, _is_synthetic_doc
from .eval_utils import normalize_answer
from .evidence_object_arbitration_utils import (
    ABSTAIN_RE,
    _filter_candidates,
    _candidate_passes_role_constraint,
    _is_containment_variant,
    _looks_verbose_or_multi_answer,
)
from .provenance_answer_lattice_utils import (
    _collect_from_rows,
    _collect_module_candidates,
    _is_synthetic_title,
    _materialize_lattice,
    _same_answer,
    _source_docs,
    _token_count,
)
from .sentence_evidence_utils import infer_answer_target_type
from .typed_evidence_table_utils import _build_typed_table


DATE_VALUE_RE = re.compile(
    r"\b(?:\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2})\b",
    re.IGNORECASE,
)
NUMBER_VALUE_RE = re.compile(r"\b\d+(?:\.\d+)?(?:\s*(?:%|percent|million|billion|km|kg|miles?))?\b", re.IGNORECASE)
BOOLEAN_ANSWER_RE = re.compile(r"^(?:yes|no|true|false)$", re.IGNORECASE)
BOOLEAN_QUESTION_RE = re.compile(
    r"^\s*(?:are|is|was|were|am|do|does|did|can|could|should|would|will|has|have|had)\b",
    re.IGNORECASE,
)
UNIT_VALUE_RE = re.compile(
    r"(?P<prefix>[$€£])?\s*(?P<number>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>%|percent|percentage|million|billion|trillion|thousand|"
    r"milligrams?|mg|grams?|g|kilograms?|kg|pounds?|lbs?|ounces?|oz|"
    r"millimeters?|millimetres?|mm|centimeters?|centimetres?|cm|"
    r"meters?|metres?|m|kilometers?|kilometres?|km|miles?|"
    r"inches?|feet|foot|ft|seconds?|minutes?|hours?|days?|weeks?|months?|years?|"
    r"dollars?|usd|euros?|eur|pounds?|gbp)?\b",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
GENERIC_ENTITY_RE = re.compile(
    r"\b(?:person|people|city|country|state|company|organization|film|book|album|"
    r"team|school|university|unknown|none|not specified)\b",
    re.IGNORECASE,
)
LOCATION_CUE_RE = re.compile(
    r"\b(?:city|town|village|county|state|province|country|region|district|"
    r"capital|port|island|kingdom|republic|territory|municipality|prefecture|"
    r"born|birthplace|place\s+of\s+birth|died|death|headquartered|located)\b",
    re.IGNORECASE,
)
ORG_CUE_RE = re.compile(
    r"\b(?:university|college|school|institute|company|corporation|inc\.?|ltd\.?|"
    r"club|team|party|association|organization|organisation|agency|committee|"
    r"ministry|department|foundation|league|group|band)\b",
    re.IGNORECASE,
)
WORK_CUE_RE = re.compile(
    r"\b(?:film|movie|album|song|single|book|novel|poem|play|series|episode|"
    r"opera|work|publication|magazine|newspaper|game|directed|written|published|released)\b",
    re.IGNORECASE,
)
OCCUPATION_CUE_RE = re.compile(
    r"\b(?:actor|actress|singer|writer|author|director|producer|composer|poet|"
    r"politician|scientist|engineer|artist|painter|footballer|player|coach|"
    r"journalist|historian|professor|mathematician|physician|lawyer|designer|"
    r"architect|musician|novelist|screenwriter|businessperson)\b",
    re.IGNORECASE,
)
NATIONALITY_OR_COUNTRY_RE = re.compile(
    r"\b(?:afghan|albanian|algerian|american|andorran|angolan|argentine|armenian|"
    r"australian|austrian|azerbaijani|bahamian|bahraini|bangladeshi|barbadian|"
    r"belarusian|belgian|belizean|beninese|bhutanese|bolivian|bosnian|brazilian|"
    r"british|bulgarian|burmese|cambodian|cameroonian|canadian|chilean|chinese|"
    r"colombian|congolese|croatian|cuban|cypriot|czech|danish|dominican|dutch|"
    r"ecuadorean|egyptian|emirati|english|estonian|ethiopian|fijian|filipino|"
    r"finnish|french|georgian|german|ghanaian|greek|guatemalan|haitian|honduran|"
    r"hungarian|icelandic|indian|indonesian|iranian|iraqi|irish|israeli|italian|"
    r"jamaican|japanese|jordanian|kazakh|kenyan|korean|kuwaiti|latvian|lebanese|"
    r"liberian|libyan|lithuanian|malaysian|mexican|moroccan|nepali|new\s+zealander|"
    r"nigerian|norwegian|pakistani|palestinian|panamanian|peruvian|polish|portuguese|"
    r"romanian|russian|saudi|scottish|serbian|singaporean|slovak|slovenian|somali|"
    r"south\s+african|spanish|sri\s+lankan|sudanese|swedish|swiss|syrian|taiwanese|"
    r"thai|tunisian|turkish|ukrainian|uruguayan|venezuelan|vietnamese|welsh|"
    r"zambian|zimbabwean|"
    r"afghanistan|albania|algeria|argentina|armenia|australia|austria|azerbaijan|"
    r"bahamas|bahrain|bangladesh|belarus|belgium|belize|benin|bhutan|bolivia|"
    r"bosnia|brazil|bulgaria|burma|cambodia|cameroon|canada|chile|china|colombia|"
    r"congo|croatia|cuba|cyprus|denmark|dominica|ecuador|egypt|england|estonia|"
    r"ethiopia|fiji|finland|france|georgia|germany|ghana|greece|guatemala|haiti|"
    r"honduras|hungary|iceland|india|indonesia|iran|iraq|ireland|israel|italy|"
    r"jamaica|japan|jordan|kazakhstan|kenya|korea|kuwait|latvia|lebanon|liberia|"
    r"libya|lithuania|malaysia|mexico|morocco|nepal|nigeria|norway|pakistan|"
    r"palestine|panama|peru|poland|portugal|romania|russia|scotland|serbia|"
    r"singapore|slovakia|slovenia|somalia|spain|sudan|sweden|switzerland|syria|"
    r"taiwan|thailand|tunisia|turkey|ukraine|uruguay|venezuela|vietnam|wales)\b",
    re.IGNORECASE,
)
QUESTION_ANCHOR_STOPWORDS = {
    "about", "after", "again", "also", "answer", "before", "began", "begin",
    "born", "city", "country", "date", "death", "did", "died", "does", "film",
    "first", "from", "has", "have", "many", "movie", "name", "nationality",
    "number", "place", "second", "song", "that", "their", "this", "what",
    "when", "where", "which", "who", "whom", "whose", "with", "year",
}
DERIVED_EVIDENCE_SOURCES = {
    "anchored_typed_candidate_table",
    "candidate_answer_selector",
    "contrastive_answer_arbitration",
    "contrastive_answer_arbitration:selected",
    "evidence_answer_arbiter",
    "evidence_answer_arbiter:selected",
    "evidence_consistent_reader",
    "evidence_consistent_reader:selected",
    "evidence_object_arbitration",
    "evidence_object_arbitration:selected",
    "provenance_answer_lattice",
    "provenance_answer_lattice:selected",
    "relation_fact_join",
    "relation_fact_join_reader_projection",
    "typed_evidence_table",
    "typed_evidence_table:selected",
    "typed_table_from_reader",
}
DERIVED_TITLE_RE = re.compile(
    r"^(?:anchored typed candidate table|typed evidence grounding|"
    r"relation fact join|query schema memory|slot-projected provenance|"
    r"slot-separated fact join view|provenance answer lattice|candidate answer table)\b",
    re.IGNORECASE,
)


def _clamp(value: float, low: float = 0.0, high: float = 0.99) -> float:
    return max(low, min(high, float(value or 0.0)))


def _new_meta(enabled: bool, before: str) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "changed": False,
        "action": "none" if enabled else "disabled",
        "reason": "disabled" if not enabled else "not_evaluated",
        "before": before,
        "after": before,
        "target_type": None,
        "answer_supported": False,
        "answer_support_score": 0.0,
        "raw_candidate_count": 0,
        "filtered_candidate_count": 0,
        "store_size": 0,
        "selected_candidate": "",
        "selected_score": 0.0,
        "selected_net_utility": 0.0,
        "selected_risk": 0.0,
        "selected_source_count": 0,
        "selected_provenance_count": 0,
        "selected_base_source_count": 0,
        "selected_base_provenance_count": 0,
        "selected_derived_provenance_count": 0,
        "margin": 0.0,
        "support_gain": 0.0,
        "base_supported_candidate": False,
        "current_answer_in_base_docs": False,
        "candidate_in_base_docs": False,
        "base_comentioned_with_current": False,
        "shadow_changed": False,
        "shadow_action": "none",
        "shadow_after": before,
    }


def _same_or_empty(left: str, right: str) -> bool:
    return not str(left or "").strip() or not str(right or "").strip() or _same_answer(left, right)


def _is_shorter_alias(before: str, after: str) -> bool:
    if not _is_containment_variant(before, after):
        return False
    return 0 < _token_count(after) < _token_count(before)


def _type_value_signal(candidate: str, target_type: str) -> float:
    text = str(candidate or "")
    if target_type == "date":
        return 1.0 if DATE_VALUE_RE.search(text) else 0.0
    if target_type == "number":
        return 1.0 if NUMBER_VALUE_RE.search(text) else 0.0
    if target_type == "nationality":
        if NATIONALITY_OR_COUNTRY_RE.search(text):
            return 1.0
        return 0.0
    if target_type == "occupation":
        return 1.0 if OCCUPATION_CUE_RE.search(text) else 0.0
    if target_type == "work":
        if WORK_CUE_RE.search(text) or '"' in text or "'" in text:
            return 1.0
        return 0.45 if _token_count(text) <= 8 else 0.15
    if target_type in {"person", "organization", "location", "entity"}:
        if not text.strip() or GENERIC_ENTITY_RE.fullmatch(text.strip()):
            return 0.0
        return 0.55 if _token_count(text) <= 8 else 0.20
    return 0.35 if text.strip() else 0.0


def _content_tokens(text: Any) -> List[str]:
    return [
        token
        for token in TOKEN_RE.findall(normalize_answer(str(text or "")))
        if len(token) > 1 and token not in QUESTION_ANCHOR_STOPWORDS
    ]


def _norm_contains(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return re.search(rf"(?:^|\s){re.escape(needle)}(?:\s|$)", haystack) is not None


def _candidate_reuses_question_surface(candidate: str, question: str) -> bool:
    candidate_norm = normalize_answer(candidate)
    question_norm = normalize_answer(re.sub(r"['’]s\b", " ", str(question or ""), flags=re.IGNORECASE))
    if not candidate_norm or not question_norm:
        return False
    if _norm_contains(question_norm, candidate_norm):
        return True
    candidate_tokens = set(_content_tokens(candidate))
    question_tokens = set(_content_tokens(question))
    if candidate_tokens and candidate_tokens.issubset(question_tokens):
        return True
    return False


def _slot_type_for_question(question: str, fallback: str) -> str:
    q = re.sub(r"\s+", " ", str(question or "").lower()).strip()
    if re.search(r"\b(?:nationality|citizenship|what country was .{0,80} from)\b", q):
        return "nationality"
    if re.search(r"\b(?:which|what)\s+(?:film|movie|album|book|novel|song|play|series|episode|work)\b", q):
        return "work"
    if re.search(r"\b(?:occupation|profession|job|career)\b", q):
        return "occupation"
    if re.search(
        r"\b(?:where|which city|which country|which state|which county|birthplace|"
        r"place of birth|place of death|headquarters|headquartered|located|capital)\b",
        q,
    ):
        return "location"
    if re.search(r"\b(?:who|whose|which person|father|mother|parent|spouse|wife|husband|son|daughter)\b", q):
        return "person"
    return str(fallback or "factoid")


def _row_evidence_text(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    for value in row.get("titles") or []:
        parts.append(str(value or ""))
    for item in row.get("evidence") or []:
        if isinstance(item, dict):
            parts.append(str(item.get("title") or ""))
            parts.append(str(item.get("sentence_preview") or item.get("sentence") or ""))
        else:
            parts.append(str(item or ""))
    return " ".join(part for part in parts if part).strip()


def _row_role_proxy(row: Dict[str, Any]) -> Dict[str, Any]:
    evidence = row.get("evidence") or []
    first = evidence[0] if evidence and isinstance(evidence[0], dict) else {}
    titles = row.get("titles") or []
    return {
        "candidate": row.get("candidate", ""),
        "title": first.get("title") or (titles[0] if titles else ""),
        "sentence_preview": first.get("sentence_preview") or first.get("sentence") or _row_evidence_text(row)[:360],
    }


def _source_root(source: str) -> str:
    text = str(source or "").strip()
    if ":lineage:" in text:
        text = text.split(":lineage:", 1)[0]
    return text


def _is_derived_evidence_item(item: Dict[str, Any]) -> bool:
    source = _source_root(str(item.get("source") or ""))
    title = str(item.get("title") or "")
    if source in DERIVED_EVIDENCE_SOURCES:
        return True
    if DERIVED_TITLE_RE.search(title.strip()):
        return True
    return bool(title and _is_synthetic_title(title))


def _evidence_sentence(item: Dict[str, Any]) -> str:
    return str(item.get("sentence_preview") or item.get("sentence") or "")


def _source_documents_contain_answer(source_docs: List[str], answer: str) -> bool:
    answer_norm = normalize_answer(answer)
    if not answer_norm:
        return False
    return any(answer_norm in normalize_answer(doc) for doc in source_docs)


def _answers_comentioned_in_base_evidence(
    row: Dict[str, Any],
    before: str,
    after: str,
) -> bool:
    before_norm = normalize_answer(before)
    after_norm = normalize_answer(after)
    if not before_norm or not after_norm:
        return False
    for item in row.get("evidence") or []:
        if not isinstance(item, dict) or _is_derived_evidence_item(item):
            continue
        sentence_norm = normalize_answer(_evidence_sentence(item))
        if before_norm in sentence_norm and after_norm in sentence_norm:
            return True
    return False


def _candidate_matches_slot_type(candidate: str, row: Dict[str, Any], slot_type: str) -> Tuple[bool, str]:
    text = str(candidate or "").strip()
    evidence_text = _row_evidence_text(row)
    joined = f"{text} {evidence_text}"
    token_count = _token_count(text)
    if not text:
        return False, "empty_candidate"
    if slot_type == "date":
        return (True, "") if DATE_VALUE_RE.search(text) else (False, "target_type_date_mismatch")
    if slot_type == "number":
        return (True, "") if NUMBER_VALUE_RE.search(text) else (False, "target_type_number_mismatch")
    if slot_type == "nationality":
        if token_count > 4 or re.search(r"[,;/]|\band\b|\bor\b", text, re.IGNORECASE):
            return False, "target_type_nationality_composite"
        if NATIONALITY_OR_COUNTRY_RE.search(text):
            return True, ""
        return False, "target_type_nationality_mismatch"
    if slot_type == "occupation":
        return (True, "") if OCCUPATION_CUE_RE.search(text) else (False, "target_type_occupation_mismatch")
    if slot_type == "work":
        if DATE_VALUE_RE.search(text) or NUMBER_VALUE_RE.fullmatch(text):
            return False, "target_type_work_numeric_or_date"
        if NATIONALITY_OR_COUNTRY_RE.fullmatch(text.strip()):
            return False, "target_type_work_demonym_or_country"
        if WORK_CUE_RE.search(joined) or '"' in joined or "'" in joined or float(row.get("answer_type_score", 0.0) or 0.0) >= 0.55:
            return True, ""
        return False, "target_type_work_mismatch"
    if slot_type == "location":
        if DATE_VALUE_RE.search(text) or NUMBER_VALUE_RE.fullmatch(text):
            return False, "target_type_location_numeric_or_date"
        if LOCATION_CUE_RE.search(joined) or "," in text or float(row.get("answer_type_score", 0.0) or 0.0) >= 0.55:
            return True, ""
        return False, "target_type_location_mismatch"
    if slot_type == "organization":
        if ORG_CUE_RE.search(joined) or float(row.get("answer_type_score", 0.0) or 0.0) >= 0.55:
            return True, ""
        return False, "target_type_organization_mismatch"
    if slot_type == "person":
        if DATE_VALUE_RE.search(text) or NUMBER_VALUE_RE.fullmatch(text):
            return False, "target_type_person_numeric_or_date"
        if NATIONALITY_OR_COUNTRY_RE.fullmatch(text.strip()):
            return False, "target_type_person_demonym_or_country"
        if token_count <= 6:
            return True, ""
        return False, "target_type_person_too_long"
    return True, ""


def _is_boolean_answer(text: str) -> bool:
    return bool(BOOLEAN_ANSWER_RE.fullmatch(normalize_answer(str(text or ""))))


def _looks_boolean_question(question: str) -> bool:
    q = str(question or "").strip()
    if BOOLEAN_QUESTION_RE.search(q):
        return True
    q_norm = normalize_answer(q)
    return bool(
        re.search(r"\b(?:same|both|either|neither|also|as well as)\b", q_norm)
        and not re.search(r"\b(?:what|which|who|where|when|how many|how much)\b", q_norm)
    )


def _numeric_values(text: str) -> List[str]:
    values: List[str] = []
    for match in UNIT_VALUE_RE.finditer(str(text or "")):
        raw = match.group("number")
        if not raw:
            continue
        try:
            value = float(raw)
            normalized = f"{value:.8f}".rstrip("0").rstrip(".")
        except ValueError:
            normalized = raw.strip().lower()
        if normalized:
            values.append(normalized)
    return values


def _numeric_units(text: str) -> List[str]:
    units: List[str] = []
    for match in UNIT_VALUE_RE.finditer(str(text or "")):
        prefix = str(match.group("prefix") or "").strip().lower()
        unit = str(match.group("unit") or "").strip().lower()
        if prefix:
            units.append(prefix)
        if unit:
            units.append(unit)
    return units


def _date_values(text: str) -> List[str]:
    return [normalize_answer(match.group(0)) for match in DATE_VALUE_RE.finditer(str(text or ""))]


def _rewrite_safety_block_reason(
    *,
    question: str,
    before: str,
    after: str,
    target_type: str,
    answer_supported: bool,
) -> str:
    before_norm = normalize_answer(before)
    after_norm = normalize_answer(after)
    if not after_norm or before_norm == after_norm:
        return ""

    before_boolean = _is_boolean_answer(before)
    after_boolean = _is_boolean_answer(after)
    if before_boolean or _looks_boolean_question(question):
        if before_boolean and not after_boolean:
            return "boolean_answer_protected"
        if before_boolean and after_boolean and before_norm != after_norm and answer_supported:
            return "boolean_conflict_supported_answer"

    before_numbers = set(_numeric_values(before))
    after_numbers = set(_numeric_values(after))
    before_units = set(_numeric_units(before))
    after_units = set(_numeric_units(after))
    if before_numbers and after_numbers:
        if before_units and not after_units and before_numbers & after_numbers:
            return "numeric_unit_protected"
        if target_type == "number" and answer_supported and not (before_numbers & after_numbers):
            return "numeric_conflict_supported_answer"

    before_dates = set(_date_values(before))
    after_dates = set(_date_values(after))
    if target_type == "date" and answer_supported and before_dates and after_dates and not (before_dates & after_dates):
        return "date_conflict_supported_answer"

    return ""


def _is_generic_bad_candidate(candidate: str, target_type: str) -> bool:
    text = str(candidate or "").strip()
    if not text:
        return True
    if target_type in {"entity", "person", "organization", "location"} and GENERIC_ENTITY_RE.fullmatch(text):
        return True
    return False


def _row_cost_features(
    row: Dict[str, Any],
    *,
    question: str,
    source_docs: List[str],
    target_type: str,
    slot_type: str,
    answer_support_score: float,
    max_docs: int,
    max_answer_tokens: int,
    token_weight: float,
    provenance_weight: float,
    risk_weight: float,
    current_answer_supported: bool,
) -> Dict[str, Any]:
    candidate = str(row.get("candidate") or "").strip()
    candidate_supported, candidate_support_score = _is_supported(
        question=question,
        answer=candidate,
        qa_docs=source_docs,
        target_type=target_type,
        max_docs=max_docs,
    )
    row_score = float(row.get("score", row.get("table_score", row.get("confidence", 0.0))) or 0.0)
    support_score = max(row_score, float(candidate_support_score or 0.0))
    source_count = max(
        int(row.get("source_count", 0) or 0),
        int(row.get("title_count", 0) or 0),
    )
    module_source_count = int(row.get("module_source_count", 0) or 0)
    provenance_count = int(row.get("provenance_count", 0) or 0)
    module_provenance_count = int(row.get("module_provenance_count", 0) or 0)
    evidence_items = row.get("evidence") or []
    base_sentence_keys = set()
    derived_sentence_keys = set()
    base_titles = set()
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        sentence_key = normalize_answer(_evidence_sentence(item)[:220])
        title_key = normalize_answer(str(item.get("title") or ""))
        if _is_derived_evidence_item(item):
            if sentence_key:
                derived_sentence_keys.add(sentence_key)
            continue
        if sentence_key:
            base_sentence_keys.add(sentence_key)
        if title_key:
            base_titles.add(title_key)
    sentence_provenance_count = len(base_sentence_keys)
    derived_provenance_count = len(derived_sentence_keys)
    base_source_count = len(base_titles)
    base_provenance_count = len(base_sentence_keys)
    if sentence_provenance_count:
        provenance_count = max(provenance_count, sentence_provenance_count)
        source_count = max(source_count, min(max(base_source_count, sentence_provenance_count), module_source_count, 4))
    if not source_count and evidence_items:
        source_count = min(max(1, module_source_count), 3)
    if not provenance_count and evidence_items:
        provenance_count = min(max(1, module_provenance_count), 3)
    type_signal = max(float(row.get("answer_type_score", 0.0) or 0.0), _type_value_signal(candidate, slot_type))
    question_overlap = float(row.get("question_overlap", 0.0) or 0.0)
    relation_score = float(row.get("relation_score", row.get("relation_role_score", 0.0)) or 0.0)
    evidence_count = max(source_count, provenance_count)
    independent_bonus = min(
        0.10,
        provenance_weight * max(0, provenance_count - 1)
        + 0.020 * max(0, source_count - 1),
    )
    module_bonus = min(0.02, 0.006 * max(0, module_source_count - 1)) if source_count > 0 else 0.0
    utility = (
        0.52 * support_score
        + 0.13 * type_signal
        + 0.10 * question_overlap
        + 0.08 * relation_score
        + independent_bonus
        + module_bonus
    )

    token_count = _token_count(candidate)
    length_cost = token_weight * min(1.0, max(0, token_count - max_answer_tokens) / max(1, max_answer_tokens))
    single_source_risk = 0.16 if source_count <= 1 else 0.0
    weak_independent_provenance_risk = 0.10 if provenance_count <= 1 else 0.0
    low_overlap_risk = 0.08 if question_overlap < 0.04 and source_count <= 1 else 0.0
    generic_risk = 0.12 if _is_generic_bad_candidate(candidate, target_type) else 0.0
    supported_current_risk = 0.12 if current_answer_supported else 0.0
    question_anchor_risk = 0.30 if _candidate_reuses_question_surface(candidate, question) else 0.0
    slot_ok, _ = _candidate_matches_slot_type(candidate, row, slot_type)
    slot_mismatch_risk = 0.35 if not slot_ok else 0.0
    risk = (
        length_cost
        + single_source_risk
        + weak_independent_provenance_risk
        + low_overlap_risk
        + generic_risk
        + supported_current_risk
        + question_anchor_risk
        + slot_mismatch_risk
    )
    net_utility = utility - risk_weight * risk
    return {
        "candidate": candidate,
        "support_score": round(_clamp(support_score), 4),
        "candidate_supported": bool(candidate_supported),
        "candidate_support_score": round(float(candidate_support_score or 0.0), 4),
        "type_signal": round(_clamp(type_signal), 4),
        "question_overlap": round(question_overlap, 4),
        "relation_score": round(relation_score, 4),
        "source_count": source_count,
        "module_source_count": module_source_count,
        "provenance_count": provenance_count,
        "module_provenance_count": module_provenance_count,
        "sentence_provenance_count": sentence_provenance_count,
        "base_source_count": base_source_count,
        "base_provenance_count": base_provenance_count,
        "derived_provenance_count": derived_provenance_count,
        "evidence_count": evidence_count,
        "utility": round(_clamp(utility, 0.0, 1.25), 4),
        "risk": round(_clamp(risk, 0.0, 1.0), 4),
        "net_utility": round(_clamp(net_utility, -1.0, 1.25), 4),
        "slot_type": slot_type,
        "slot_type_match": bool(slot_ok),
        "support_gain": round(float(support_score) - float(answer_support_score), 4),
        "token_count": token_count,
        "sources": row.get("sources", [])[:5],
        "titles": row.get("titles", [])[:5],
        "evidence": row.get("evidence", [])[:3],
    }


def _collect_all_candidate_sources(
    *,
    candidates: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    evidence_consistent_reader: Optional[Dict[str, Any]],
    evidence_object_arbitration: Optional[Dict[str, Any]],
    evidence_answer_arbiter: Optional[Dict[str, Any]],
    typed_evidence_table: Optional[Dict[str, Any]],
    contrastive_answer_arbitration: Optional[Dict[str, Any]],
    provenance_answer_lattice: Optional[Dict[str, Any]],
    anchored_typed_candidate_table: Optional[Dict[str, Any]],
    candidate_answer_selector: Optional[Dict[str, Any]],
    relation_fact_join: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    store: Dict[str, Dict[str, Any]] = {}
    _collect_from_rows(store, candidates, "reader_evidence_candidates")
    _collect_from_rows(store, evidence_rows, "typed_table_from_reader")
    _collect_module_candidates(store, "evidence_consistent_reader", evidence_consistent_reader)
    _collect_module_candidates(store, "evidence_object_arbitration", evidence_object_arbitration)
    _collect_module_candidates(store, "evidence_answer_arbiter", evidence_answer_arbiter)
    _collect_module_candidates(store, "typed_evidence_table", typed_evidence_table)
    _collect_module_candidates(store, "contrastive_answer_arbitration", contrastive_answer_arbitration)
    _collect_module_candidates(store, "candidate_answer_selector", candidate_answer_selector)
    _collect_from_rows(
        store,
        (provenance_answer_lattice or {}).get("lattice_rows") or [],
        "provenance_answer_lattice",
    )
    _collect_from_rows(
        store,
        (anchored_typed_candidate_table or {}).get("candidate_rows") or [],
        "anchored_typed_candidate_table",
    )
    _collect_from_rows(
        store,
        (relation_fact_join or {}).get("candidate_table_rows") or [],
        "relation_fact_join",
    )
    _collect_from_rows(
        store,
        (relation_fact_join or {}).get("reader_candidate_table_rows") or [],
        "relation_fact_join_reader_projection",
    )
    return store


def arbitrate_answer_with_provenance_typed_candidate_store(
    question: str,
    answer: str,
    qa_docs: Optional[List[str]] = None,
    answer_style: str = "",
    evidence_consistent_reader: Optional[Dict[str, Any]] = None,
    evidence_object_arbitration: Optional[Dict[str, Any]] = None,
    evidence_answer_arbiter: Optional[Dict[str, Any]] = None,
    typed_evidence_table: Optional[Dict[str, Any]] = None,
    contrastive_answer_arbitration: Optional[Dict[str, Any]] = None,
    provenance_answer_lattice: Optional[Dict[str, Any]] = None,
    anchored_typed_candidate_table: Optional[Dict[str, Any]] = None,
    candidate_answer_selector: Optional[Dict[str, Any]] = None,
    relation_fact_join: Optional[Dict[str, Any]] = None,
    enabled: bool = False,
    apply_rewrite: bool = False,
    min_score: float = 0.88,
    generic_min_score: float = 0.92,
    min_net_utility: float = 0.46,
    min_margin: float = 0.08,
    min_support_gain: float = 0.08,
    alias_min_support_gain: float = 0.01,
    min_question_overlap: float = 0.06,
    max_docs: int = 10,
    max_answer_tokens: int = 12,
    max_candidates: int = 16,
    min_sources: int = 1,
    min_provenance: int = 1,
    high_confidence_single_source: float = 0.94,
    token_weight: float = 0.08,
    provenance_weight: float = 0.035,
    risk_weight: float = 0.60,
    require_role_constraint: bool = True,
    skip_supported_answers: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Cost-based selector over a provenance-typed answer candidate store.

    Unlike the older lattice arbiter, this materializes candidates from all
    answer-side operators as a tiny typed table and admits rewrites by a
    utility-cost objective. It is intentionally rank-stable: retrieval order is
    not changed, and concise supported reader answers are protected unless a
    candidate has a clear provenance/type/support advantage.
    """
    before = str(answer or "").strip()
    meta = _new_meta(enabled=enabled, before=before)
    meta.update({
        "apply_rewrite": bool(apply_rewrite),
        "require_role_constraint": bool(require_role_constraint),
        "skip_supported_answers": bool(skip_supported_answers),
        "min_sources": int(min_sources),
        "min_provenance": int(min_provenance),
        "min_net_utility": float(min_net_utility),
    })
    if not enabled or not before:
        return before, meta

    source_docs = _source_docs(qa_docs)
    if not source_docs:
        meta["reason"] = "no_source_docs"
        return before, meta

    target_type = infer_answer_target_type(question)
    slot_type = _slot_type_for_question(question, target_type)
    is_numeric_temporal = target_type in {"date", "number"}
    meta["target_type"] = target_type
    meta["slot_type"] = slot_type
    if is_numeric_temporal and not _has_main_numeric_temporal_request(question, target_type):
        meta["reason"] = "not_main_numeric_temporal_question"
        return before, meta

    short_style = answer_style in {"", None, "short_factoid", "multi_slot"}
    verbose_or_multi = _looks_verbose_or_multi_answer(before, max_answer_tokens)
    if not short_style and not verbose_or_multi and not is_numeric_temporal:
        meta.update({"reason": "non_short_answer_style", "answer_style": answer_style})
        return before, meta

    answer_supported, answer_support_score = _is_supported(
        question=question,
        answer=before,
        qa_docs=source_docs,
        target_type=target_type,
        max_docs=max_docs,
    )
    raw_candidates = _rank_candidates(
        question=question,
        answer=before,
        qa_docs=source_docs,
        target_type=target_type,
        max_docs=max_docs,
    )
    candidates, reject_counts = _filter_candidates(
        raw_candidates,
        question=question,
        target_type=target_type,
        min_question_overlap=min_question_overlap,
        require_role_constraint=require_role_constraint,
    )
    evidence_rows = _build_typed_table(candidates)
    store = _collect_all_candidate_sources(
        candidates=candidates,
        evidence_rows=evidence_rows,
        evidence_consistent_reader=evidence_consistent_reader,
        evidence_object_arbitration=evidence_object_arbitration,
        evidence_answer_arbiter=evidence_answer_arbiter,
        typed_evidence_table=typed_evidence_table,
        contrastive_answer_arbitration=contrastive_answer_arbitration,
        provenance_answer_lattice=provenance_answer_lattice,
        anchored_typed_candidate_table=anchored_typed_candidate_table,
        candidate_answer_selector=candidate_answer_selector,
        relation_fact_join=relation_fact_join,
    )
    rows = _materialize_lattice(store, current_answer=before, min_sources=min_sources)
    scored_rows = [
        _row_cost_features(
            row,
            question=question,
            source_docs=source_docs,
            target_type=target_type,
            slot_type=slot_type,
            answer_support_score=float(answer_support_score),
            max_docs=max_docs,
            max_answer_tokens=max_answer_tokens,
            token_weight=token_weight,
            provenance_weight=provenance_weight,
            risk_weight=risk_weight,
            current_answer_supported=bool(answer_supported),
        )
        for row in rows
        if not _same_or_empty(before, row.get("candidate", ""))
    ]
    scored_rows.sort(
        key=lambda row: (
            -float(row.get("net_utility", 0.0) or 0.0),
            -float(row.get("support_score", 0.0) or 0.0),
            -int(row.get("evidence_count", 0) or 0),
            str(row.get("candidate", "")).lower(),
        )
    )
    scored_rows = scored_rows[:max(1, int(max_candidates))]

    top = scored_rows[0] if scored_rows else None
    second_net = float(scored_rows[1].get("net_utility", 0.0)) if len(scored_rows) > 1 else 0.0
    after = str(top.get("candidate", "") or "").strip() if top else ""
    top_score = float(top.get("support_score", 0.0) or 0.0) if top else 0.0
    net_utility = float(top.get("net_utility", 0.0) or 0.0) if top else 0.0
    support_gain = top_score - float(answer_support_score or 0.0)
    margin = net_utility - second_net
    source_count = int(top.get("source_count", 0) or 0) if top else 0
    provenance_count = int(top.get("provenance_count", 0) or 0) if top else 0
    evidence_count = int(top.get("evidence_count", 0) or 0) if top else 0
    base_source_count = int(top.get("base_source_count", 0) or 0) if top else 0
    base_provenance_count = int(top.get("base_provenance_count", 0) or 0) if top else 0
    derived_provenance_count = int(top.get("derived_provenance_count", 0) or 0) if top else 0
    base_supported_candidate = bool(base_source_count >= 1 and base_provenance_count >= 1)
    current_answer_in_base_docs = _source_documents_contain_answer(source_docs, before)
    candidate_in_base_docs = _source_documents_contain_answer(source_docs, after)
    base_comentioned_with_current = bool(top and _answers_comentioned_in_base_evidence(top, before, after))
    is_abstain = bool(ABSTAIN_RE.search(before))
    shorter_alias = bool(after and _is_shorter_alias(before, after))
    same_answer = bool(after and _same_answer(before, after))
    before_multi_answer = answer_style == "multi_slot" or (
        re.search(r"(?:;|,|\band\b|\bor\b|/)", before, re.IGNORECASE) is not None
        and _token_count(before) >= 4
    )
    after_multi_answer = (
        re.search(r"(?:;|,|\band\b|\bor\b|/)", after, re.IGNORECASE) is not None
        and _token_count(after) >= 4
    )
    candidate_reuses_question_surface = bool(after and _candidate_reuses_question_surface(after, question))
    slot_type_match, slot_type_reason = (
        _candidate_matches_slot_type(after, top, slot_type) if top else (False, "no_candidate")
    )
    role_target_type = target_type if slot_type == "nationality" else slot_type
    role_ok, role_reason = (
        _candidate_passes_role_constraint(
            candidate=_row_role_proxy(top),
            question=question,
            target_type=role_target_type,
            require_role_constraint=require_role_constraint,
        )
        if top
        else (False, "no_candidate")
    )
    confidence_floor = float(min_score) if is_numeric_temporal else float(generic_min_score)
    alias_canonicalization = (
        bool(shorter_alias)
        and base_supported_candidate
        and not base_comentioned_with_current
        and top_score >= max(0.84, confidence_floor - 0.08)
        and net_utility >= max(0.34, float(min_net_utility) - 0.12)
        and margin >= max(0.04, float(min_margin) * 0.5)
    )
    role_bound_disambiguation = (
        bool(answer_supported)
        and not bool(shorter_alias)
        and base_source_count >= 2
        and base_provenance_count >= 2
        and not current_answer_in_base_docs
        and candidate_in_base_docs
        and not base_comentioned_with_current
        and top_score >= max(0.96, confidence_floor + 0.03)
        and net_utility >= max(0.60, float(min_net_utility) + 0.12)
        and margin >= max(0.22, float(min_margin) + 0.12)
        and support_gain >= max(0.30, float(min_support_gain) + 0.18)
    )
    unsupported_answer_repair = (
        not bool(answer_supported)
        and base_supported_candidate
        and candidate_in_base_docs
        and top_score >= max(0.86, confidence_floor - 0.06)
        and net_utility >= max(0.38, float(min_net_utility) - 0.08)
        and support_gain >= max(0.08, float(min_support_gain) * 0.50)
    )

    meta.update({
        "answer_style": answer_style,
        "answer_supported": bool(answer_supported),
        "answer_support_score": round(float(answer_support_score), 4),
        "raw_candidate_count": len(raw_candidates),
        "filtered_candidate_count": len(candidates),
        "candidate_reject_counts": reject_counts,
        "store_size": len(rows),
        "candidate_store_rows": scored_rows[:8],
        "selected_candidate": after,
        "selected_score": round(top_score, 4),
        "selected_net_utility": round(net_utility, 4),
        "selected_risk": round(float(top.get("risk", 0.0) or 0.0), 4) if top else 0.0,
        "selected_source_count": source_count,
        "selected_provenance_count": provenance_count,
        "selected_evidence_count": evidence_count,
        "selected_base_source_count": base_source_count,
        "selected_base_provenance_count": base_provenance_count,
        "selected_derived_provenance_count": derived_provenance_count,
        "margin": round(margin, 4),
        "support_gain": round(support_gain, 4),
        "verbose_or_multi": bool(verbose_or_multi),
        "shorter_alias": bool(shorter_alias),
        "is_abstain": bool(is_abstain),
        "candidate_reuses_question_surface": candidate_reuses_question_surface,
        "slot_type_match": bool(slot_type_match),
        "role_constraint_match": bool(role_ok),
        "base_supported_candidate": bool(base_supported_candidate),
        "current_answer_in_base_docs": bool(current_answer_in_base_docs),
        "candidate_in_base_docs": bool(candidate_in_base_docs),
        "base_comentioned_with_current": bool(base_comentioned_with_current),
        "alias_canonicalization": bool(alias_canonicalization),
        "role_bound_disambiguation": bool(role_bound_disambiguation),
        "unsupported_answer_repair": bool(unsupported_answer_repair),
    })

    if not top:
        meta["reason"] = "no_candidates"
        return before, meta
    if not after or same_answer:
        meta["reason"] = "same_as_current"
        return before, meta

    rewrite_safety_reason = _rewrite_safety_block_reason(
        question=question,
        before=before,
        after=after,
        target_type=target_type,
        answer_supported=bool(answer_supported),
    )
    meta["rewrite_safety_reason"] = rewrite_safety_reason

    required_sources = max(int(min_sources), 2)
    required_provenance = max(int(min_provenance), 2)
    single_source_allowed = (
        not bool(answer_supported)
        and (is_abstain or verbose_or_multi)
        and base_source_count >= 1
        and base_provenance_count >= 1
        and candidate_in_base_docs
        and top_score >= float(high_confidence_single_source)
        and net_utility >= float(min_net_utility)
        and bool(top.get("candidate_supported"))
    )
    relaxed_single_evidence_repair = alias_canonicalization or unsupported_answer_repair
    enough_sources = base_source_count >= required_sources or single_source_allowed or relaxed_single_evidence_repair
    enough_provenance = base_provenance_count >= required_provenance or single_source_allowed or relaxed_single_evidence_repair
    enough_support_gain = (
        support_gain >= float(min_support_gain)
        or (shorter_alias and support_gain >= float(alias_min_support_gain))
        or is_abstain
        or verbose_or_multi
        or alias_canonicalization
        or role_bound_disambiguation
        or unsupported_answer_repair
    )
    protected_supported_answer = (
        skip_supported_answers
        and answer_supported
        and not is_abstain
        and not alias_canonicalization
        and not role_bound_disambiguation
        and not (
            shorter_alias
            and base_source_count >= required_sources
            and base_provenance_count >= required_provenance
            and top_score >= max(confidence_floor, 0.95)
            and margin >= max(float(min_margin), 0.14)
            and support_gain >= float(alias_min_support_gain)
        )
        and not (
            verbose_or_multi
            and _is_containment_variant(before, after)
            and base_source_count >= required_sources
            and base_provenance_count >= required_provenance
            and top_score >= max(confidence_floor, 0.95)
        )
    )

    if not base_supported_candidate and derived_provenance_count > 0:
        meta["reason"] = "derived_only_candidate_protected"
        return before, meta
    if not enough_sources:
        meta["reason"] = "insufficient_sources"
        return before, meta
    if not enough_provenance:
        meta["reason"] = "insufficient_provenance"
        return before, meta
    if base_comentioned_with_current and answer_supported and not alias_canonicalization:
        meta["reason"] = "comentioned_answer_protected"
        return before, meta
    if candidate_reuses_question_surface:
        meta["reason"] = "candidate_is_question_anchor"
        return before, meta
    if rewrite_safety_reason:
        meta["reason"] = rewrite_safety_reason
        return before, meta
    if after_multi_answer and answer_style != "multi_slot":
        meta["reason"] = "composite_candidate_protected"
        return before, meta
    if not slot_type_match:
        meta["reason"] = slot_type_reason or "target_type_mismatch"
        return before, meta
    if not role_ok:
        meta["reason"] = role_reason or "role_constraint_mismatch"
        return before, meta
    score_floor_met = (
        top_score >= confidence_floor
        or single_source_allowed
        or alias_canonicalization
        or role_bound_disambiguation
        or unsupported_answer_repair
    )
    net_floor_met = (
        net_utility >= float(min_net_utility)
        or alias_canonicalization
        or role_bound_disambiguation
        or unsupported_answer_repair
    )
    margin_met = (
        margin >= float(min_margin)
        or single_source_allowed
        or alias_canonicalization
        or role_bound_disambiguation
        or unsupported_answer_repair
    )
    if not score_floor_met:
        meta["reason"] = "low_score"
        return before, meta
    if not net_floor_met:
        meta["reason"] = "low_net_utility"
        return before, meta
    if not margin_met:
        meta["reason"] = "low_margin"
        return before, meta
    if not enough_support_gain:
        meta["reason"] = "low_support_gain"
        return before, meta
    if before_multi_answer and not after_multi_answer and not is_abstain:
        meta["reason"] = "multi_answer_span_protected"
        return before, meta
    if protected_supported_answer:
        meta["reason"] = "supported_answer_protected"
        return before, meta

    meta.update({
        "shadow_changed": True,
        "shadow_action": "rewrite",
        "shadow_after": after,
        "reason": (
            "accepted_alias_canonicalization"
            if alias_canonicalization
            else "accepted_role_bound_disambiguation"
            if role_bound_disambiguation
            else "accepted_unsupported_answer_repair"
            if unsupported_answer_repair
            else "accepted"
        ),
    })
    if not apply_rewrite:
        meta["action"] = "shadow_rewrite"
        return before, meta

    meta.update({
        "changed": True,
        "action": "rewrite",
        "after": after,
    })
    return after, meta
