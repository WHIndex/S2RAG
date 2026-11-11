import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .answer_verifier_utils import (
    ABSTAIN_RE,
    _candidate_key,
    _doc_body,
    _doc_title,
    _is_supported,
    _rank_candidates,
    _split_sentences,
)
from .candidate_answer_selector_utils import _is_synthetic_doc
from .eval_utils import normalize_answer
from .evidence_grounded_answer_extractor_utils import (
    BOOLEAN_ANSWER_RE,
    BOOLEAN_QUESTION_RE,
    DATE_NUMBER_TYPES,
    FULL_DATE_VALUE_RE,
    MONTH_RE,
    SUPPORTED_TARGET_TYPES,
    YEAR_ONLY_RE,
    _candidate_evidence_text,
    _candidate_passes_answer_slot,
    _candidate_target_type_for_slot,
    _content_tokens,
    _infer_answer_arity,
    _infer_answer_slot,
)
from .evidence_object_arbitration_utils import (
    _filter_candidates,
    _is_containment_variant,
    _looks_verbose_or_multi_answer,
)
from .sentence_evidence_utils import infer_answer_target_type
from .typed_evidence_table_utils import _same_answer, _token_count


WORD_RE = re.compile(r"[a-z0-9]+", re.I)
YEAR_VALUE_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b")
DECADE_VALUE_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})s\b", re.I)
MONTH_VALUE_RE = re.compile(
    r"\b(?:(?:early|mid|late)[-\s]?)?"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.I,
)
PARENTHETICAL_SUFFIX_RE = re.compile(r"\s*\([^)]{1,80}\)\s*$")

OPTION_CUE_RE = re.compile(
    r"\b(?:which|what)\b.{0,80}\b(?:first|last|earlier|later|older|younger|larger|smaller|"
    r"more|less|higher|lower|longer|shorter|before|after|same|between|born|died)\b",
    re.I,
)
OPTION_SELECTION_RE = re.compile(
    r"\bwhich\b.{0,120}\b(?:is|was|were|has|had|contains|features|located|born|died|"
    r"released|published|opened|completed|built|directed|written)\b",
    re.I,
)
SHARED_WORK_RE = re.compile(
    r"\b(?:worked on by both|both worked on|both .*? worked on|shared by both|common to both)\b",
    re.I,
)
INTERSECTION_QUESTION_RE = re.compile(r"\b(?:intersection|junction|crossing)\b", re.I)
STREET_INTERSECTION_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9'. -]{1,70}\b(?:Avenue|Ave\.?|Street|St\.?|Road|Rd\.?|"
    r"Boulevard|Blvd\.?|Drive|Dr\.?|Lane|Ln\.?|Way|Place|Pl\.?)\s+and\s+"
    r"[A-Z][A-Za-z0-9'. -]{1,70}\b(?:Avenue|Ave\.?|Street|St\.?|Road|Rd\.?|"
    r"Boulevard|Blvd\.?|Drive|Dr\.?|Lane|Ln\.?|Way|Place|Pl\.?))\b",
    re.I,
)
ROLE_SPAN_QUESTION_RE = re.compile(
    r"\b(?:right-handed|left-handed|all-round|middle order|opening)\b.{0,80}\bwhat\b|"
    r"\bwhat\b.{0,80}\b(?:right-handed|left-handed|batsman|bowler|all-rounder|wicket-keeper)\b",
    re.I,
)
ROLE_SPAN_VALUE_RE = re.compile(
    r"\b(?:(?:right|left)[-\s]?handed\s+)?(?:(?:middle|top|lower)[-\s]?order\s+|"
    r"opening\s+|wicket[-\s]?keeper\s+|fast\s+|medium[-\s]?fast\s+|spin\s+|"
    r"leg[-\s]?spin\s+|off[-\s]?spin\s+)?"
    r"(?:batsman|batter|bowler|all[-\s]?rounder|wicket[-\s]?keeper)\b",
    re.I,
)
SETTING_QUESTION_RE = re.compile(r"\b(?:take place|takes place|took place|set in|setting)\b", re.I)
COMPLETION_QUESTION_RE = re.compile(
    r"\b(?:completed|built|constructed|opened|finished)\b.{0,40}\b(?:what|which|in what|in which)?\s*year\b|"
    r"\b(?:what|which|in what|in which)\s+year\b.{0,80}\b(?:completed|built|constructed|opened|finished)\b",
    re.I,
)
TEMPORAL_RELATION_RE = re.compile(
    r"\b(?:take(?:s)? place|took place|set in|setting|began|begin|started|commenced|"
    r"released|published|premiered|founded|established|formed|opened|born|died)\b",
    re.I,
)
WORK_RELATION_RE = re.compile(
    r"\b(?:film|movie|screenplay|novel|book|album|song|work|series|episode|play)\b",
    re.I,
)
OCCUPATION_CONTEXT_RE = re.compile(
    r"\b(?:occupation|profession|career|worked as|known as|served as|is an?|was an?|"
    r"became an?|composer|author|writer|actor|actress|musician|artist|scientist)\b",
    re.I,
)
OCCUPATION_VALUE_RE = re.compile(
    r"\b(?:disc jockey|screenwriter|cinematographer|director of photography|film director|"
    r"television director|record producer|music producer|film producer|composer|author|writer|"
    r"novelist|poet|playwright|journalist|editor|publisher|filmmaker|director|producer|"
    r"actor|actress|singer|musician|artist|painter|photographer|designer|architect|"
    r"mathematician|economist|scientist|engineer|physician|doctor|lawyer|attorney|"
    r"politician|minister|governor|president|mayor|diplomat|businessperson|businessman|"
    r"businesswoman|entrepreneur|professor|teacher|researcher|scholar|historian|"
    r"activist|soldier|officer|general|admiral|judge|footballer|cricketer|"
    r"basketball player|baseball player|racing driver|driver|athlete|wrestler|boxer|"
    r"dancer|model|presenter|broadcaster)\b",
    re.I,
)
BROAD_OCCUPATION_KEYS = {
    "artist",
    "writer",
    "author",
    "musician",
    "filmmaker",
    "director",
    "producer",
    "actor",
    "actress",
    "politician",
    "judge",
    "lawyer",
    "businessperson",
}
SPECIFIC_OCCUPATION_KEYS = {
    "disc jockey",
    "screenwriter",
    "cinematographer",
    "director of photography",
    "film director",
    "television director",
    "record producer",
    "music producer",
    "film producer",
    "novelist",
    "poet",
    "playwright",
    "journalist",
    "mathematician",
    "economist",
    "racing driver",
    "basketball player",
    "baseball player",
}
MONTH_TO_INT = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
RELATION_PROFILES: Tuple[Tuple[str, re.Pattern, re.Pattern], ...] = (
    (
        "setting_time",
        re.compile(r"\b(?:take place|takes place|took place|set in|setting)\b", re.I),
        re.compile(r"\b(?:take(?:s)? place|took place|set in|setting|story is set|events occur)\b", re.I),
    ),
    (
        "start_time",
        re.compile(r"\b(?:start|starts|started|begin|began|commence|commenced)\b", re.I),
        re.compile(r"\b(?:start(?:s|ed)?|begin(?:s|ning)?|began|commence(?:d)?)\b", re.I),
    ),
    (
        "release_time",
        re.compile(r"\b(?:released|published|publication|premiered)\b", re.I),
        re.compile(r"\b(?:released|published|publication|premiered|came out)\b", re.I),
    ),
    (
        "birth_time",
        re.compile(r"\b(?:born|date of birth|birthdate|birthday)\b", re.I),
        re.compile(r"\b(?:born|date of birth|birthdate|birthday)\b", re.I),
    ),
    (
        "death_time",
        re.compile(r"\b(?:died|date of death|death)\b", re.I),
        re.compile(r"\b(?:died|date of death|death)\b", re.I),
    ),
    (
        "nationality",
        re.compile(r"\b(?:nationality|citizenship|citizen|national|country)\b", re.I),
        re.compile(r"\b(?:nationality|citizenship|citizen|national|born|from)\b", re.I),
    ),
    (
        "occupation",
        re.compile(r"\b(?:occupation|profession|career|worked as|what does .* do)\b", re.I),
        OCCUPATION_CONTEXT_RE,
    ),
    (
        "creator",
        re.compile(r"\b(?:author|writer|wrote|written|director|directed|composer|composed)\b", re.I),
        re.compile(r"\b(?:author|writer|wrote|written|director|directed|composer|composed|by)\b", re.I),
    ),
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))


def _tokens(text: Any) -> List[str]:
    return [tok for tok in WORD_RE.findall(normalize_answer(str(text or ""))) if len(tok) > 1]


def _token_set(text: Any) -> set:
    return set(_tokens(text))


def _overlap(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def _clean_surface(text: Any) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip(" \t\n\r\"'`.,;:?")
    value = value.replace("'S", "'s").replace("’S", "'s").replace("’", "'")
    return value


def _clean_work_title(title: str) -> str:
    title = _clean_surface(title)
    title = PARENTHETICAL_SUFFIX_RE.sub("", title).strip()
    return title


def _extract_options(question: str) -> List[str]:
    text = _clean_surface(question)
    options: List[str] = []
    patterns = (
        r"^.*?,\s*(.{2,180})\s+or\s+(.{2,140})\s*\??$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        for group in match.groups():
            candidate = _clean_work_title(group)
            if 1 <= len(_tokens(candidate)) <= 12:
                options.append(candidate)
        break

    deduped: List[str] = []
    seen = set()
    for option in options:
        key = _candidate_key(option)
        if key and key not in seen:
            seen.add(key)
            deduped.append(option)
    return deduped[:4]


def _infer_temporal_granularity(question: str, slot: str) -> str:
    q = normalize_answer(question or "")
    if COMPLETION_QUESTION_RE.search(question or ""):
        return "year"
    if SETTING_QUESTION_RE.search(question or ""):
        return "setting_period"
    if re.search(r"\bwhat month\b|\bwhich month\b", q):
        return "month"
    if re.search(r"\bwhat year\b|\bwhich year\b|\bin what year\b|\bin which year\b", q):
        return "year"
    if slot == "full_date":
        return "full_date"
    if slot == "year":
        return "year"
    return "date"


def _active_relations(question: str) -> List[Tuple[str, re.Pattern]]:
    active: List[Tuple[str, re.Pattern]] = []
    targets_completion = bool(COMPLETION_QUESTION_RE.search(question or ""))
    for label, q_pattern, evidence_pattern in RELATION_PROFILES:
        if targets_completion and label == "setting_time":
            continue
        if q_pattern.search(question or ""):
            active.append((label, evidence_pattern))
    if targets_completion:
        active.append((
            "completion_time",
            re.compile(r"\b(?:completed|built|constructed|opened|finished)\b", re.I),
        ))
    if (
        not targets_completion
        and SETTING_QUESTION_RE.search(question or "")
        and not any(label == "setting_time" for label, _ in active)
    ):
        active.append(("setting_time", TEMPORAL_RELATION_RE))
    return active


def _comparison_direction(question: str) -> Optional[str]:
    q = normalize_answer(question or "")
    if re.search(r"\b(?:earlier|first|before|older|oldest|earliest)\b", q):
        return "min"
    if re.search(r"\b(?:later|after|younger|newer|more recent|came last|comes last|released last)\b", q):
        return "max"
    return None


def _parse_temporal_value(value: str) -> Optional[Tuple[int, int, int]]:
    text = _clean_surface(value)
    if not text:
        return None
    year_match = YEAR_VALUE_RE.search(text)
    if not year_match:
        return None
    year = int(year_match.group(0))
    month = 0
    day = 0
    month_match = MONTH_VALUE_RE.search(text)
    if month_match:
        month_key = normalize_answer(month_match.group(0)).split()[-1]
        month = MONTH_TO_INT.get(month_key[:3], MONTH_TO_INT.get(month_key, 0))
    day_match = re.search(r"\b([0-3]?[0-9])(?:st|nd|rd|th)?\b", text)
    if day_match:
        try:
            maybe_day = int(day_match.group(1))
            if 1 <= maybe_day <= 31 and maybe_day != year:
                day = maybe_day
        except ValueError:
            day = 0
    return year, month, day


def _question_role_terms(question: str) -> set:
    q = normalize_answer(question or "")
    role_terms = set()
    if re.search(r"\bdirector|directed\b", q):
        role_terms.update({"director", "directed"})
    if re.search(r"\bauthor|writer|written|wrote\b", q):
        role_terms.update({"author", "writer", "written", "wrote"})
    if re.search(r"\bcomposer|composed\b", q):
        role_terms.update({"composer", "composed"})
    if re.search(r"\bborn|birth\b", q):
        role_terms.update({"born", "birth"})
    if re.search(r"\bdied|death\b", q):
        role_terms.update({"died", "death"})
    return role_terms


def _surface_hits_option(surface_norm: str, option: str) -> bool:
    option_key = normalize_answer(option)
    if not option_key:
        return False
    if option_key in surface_norm:
        return True
    clean_key = normalize_answer(_clean_work_title(option))
    return bool(clean_key and clean_key in surface_norm)


def _compile_answer_contract(question: str, answer_style: Optional[str] = None) -> Dict[str, Any]:
    boolean_answer = bool(BOOLEAN_QUESTION_RE.match(question or ""))
    if boolean_answer:
        return {
            "active": True,
            "target_type": "boolean",
            "candidate_target_type": "boolean",
            "answer_slot": "boolean",
            "answer_style": answer_style,
            "answer_domain": ["yes", "no"],
            "boolean_answer": True,
            "options": [],
            "requires_option": False,
            "comparison_direction": None,
            "shared_work": False,
            "intersection": False,
            "role_span": False,
            "temporal_granularity": "none",
            "relations": [],
        }
    target_type = infer_answer_target_type(question or "")
    candidate_target_type = "date" if target_type == "year" else target_type
    candidate_target_type = candidate_target_type if candidate_target_type in SUPPORTED_TARGET_TYPES else "person"
    answer_slot = _infer_answer_slot(question or "", candidate_target_type)
    candidate_target_type = _candidate_target_type_for_slot(answer_slot, candidate_target_type)
    options = _extract_options(question or "")
    relations = _active_relations(question or "")
    q_norm = normalize_answer(question or "")
    shared_work = bool(SHARED_WORK_RE.search(question or "") and WORK_RELATION_RE.search(question or ""))
    temporal_granularity = _infer_temporal_granularity(question or "", answer_slot)
    is_temporal = candidate_target_type in DATE_NUMBER_TYPES or bool(
        re.search(r"\b(?:when|what month|what year|which year|in what year|in which year)\b", q_norm)
    )
    is_option = bool(options and (OPTION_CUE_RE.search(question or "") or OPTION_SELECTION_RE.search(question or "")))
    is_occupation = answer_slot == "occupation" or "occupation" in q_norm or "profession" in q_norm
    is_intersection = bool(INTERSECTION_QUESTION_RE.search(question or ""))
    is_role_span = bool(ROLE_SPAN_QUESTION_RE.search(question or ""))
    active = bool(is_temporal or is_option or is_occupation or shared_work or is_intersection or is_role_span)
    return {
        "active": active,
        "boolean_answer": False,
        "target_type": target_type,
        "candidate_target_type": candidate_target_type,
        "answer_slot": answer_slot,
        "answer_style": answer_style,
        "options": options,
        "requires_option": is_option,
        "comparison_direction": _comparison_direction(question or "") if is_option else None,
        "shared_work": shared_work,
        "intersection": is_intersection,
        "role_span": is_role_span,
        "temporal_granularity": temporal_granularity,
        "relations": [label for label, _ in relations],
    }


def _subject_terms(question: str, docs: Sequence[str]) -> set:
    q_tokens = set(tok for tok in _tokens(question) if len(tok) > 2)
    title_terms = set()
    for doc in docs[:3]:
        title_terms.update(tok for tok in _tokens(_doc_title(doc)) if len(tok) > 2)
    terms = q_tokens | (title_terms & q_tokens)
    return {tok for tok in terms if tok not in {"what", "which", "when", "where", "does", "did", "both"}}


def _relation_score(contract: Dict[str, Any], text: str) -> float:
    active_labels = set(contract.get("relations") or [])
    if not active_labels:
        return 0.0
    score = 0.0
    for label, _, evidence_pattern in RELATION_PROFILES:
        if label in active_labels and evidence_pattern.search(text or ""):
            score = max(score, 1.0)
    if "setting_time" in active_labels and TEMPORAL_RELATION_RE.search(text or ""):
        score = max(score, 1.0)
    if "completion_time" in active_labels and re.search(r"\b(?:completed|built|constructed|opened|finished)\b", text or "", re.I):
        score = max(score, 1.0)
    return score


def _candidate_question_overlap(question_tokens: set, text: str) -> float:
    return _overlap(question_tokens, _token_set(text))


def _add_candidate(
    bucket: Dict[str, Dict[str, Any]],
    *,
    candidate: str,
    score: float,
    reason: str,
    title: str = "",
    sentence: str = "",
    source_rank: int = 0,
    relation_score: float = 0.0,
    question_overlap: float = 0.0,
    source: str = "contract_scan",
) -> None:
    candidate = _clean_surface(candidate)
    if not candidate:
        return
    key = _candidate_key(candidate)
    if not key:
        return
    entry = bucket.get(key)
    example = {
        "title": title,
        "sentence_preview": sentence[:260],
        "source_rank": source_rank,
        "reason": reason,
        "score": round(float(score or 0.0), 4),
    }
    if entry is None:
        bucket[key] = {
            "candidate": candidate,
            "key": key,
            "score": round(float(score or 0.0), 4),
            "support_score": round(float(score or 0.0), 4),
            "relation_score": round(float(relation_score or 0.0), 4),
            "question_overlap": round(float(question_overlap or 0.0), 4),
            "source_count": 1,
            "provenance_count": 1,
            "source": source,
            "reasons": [reason],
            "examples": [example],
        }
        return
    entry["score"] = round(max(float(entry.get("score", 0.0) or 0.0), float(score or 0.0)), 4)
    entry["support_score"] = round(max(float(entry.get("support_score", 0.0) or 0.0), float(score or 0.0)), 4)
    entry["relation_score"] = round(max(float(entry.get("relation_score", 0.0) or 0.0), float(relation_score or 0.0)), 4)
    entry["question_overlap"] = round(max(float(entry.get("question_overlap", 0.0) or 0.0), float(question_overlap or 0.0)), 4)
    entry["source_count"] = int(entry.get("source_count", 0) or 0) + 1
    entry["provenance_count"] = int(entry.get("provenance_count", 0) or 0) + 1
    if reason not in entry["reasons"]:
        entry["reasons"].append(reason)
    if len(entry["examples"]) < 3:
        entry["examples"].append(example)


def _extract_temporal_candidates(
    question: str,
    source_docs: Sequence[str],
    contract: Dict[str, Any],
    max_docs: int,
) -> List[Dict[str, Any]]:
    granularity = str(contract.get("temporal_granularity") or "date")
    question_tokens = _token_set(question)
    subject_tokens = _subject_terms(question, source_docs)
    bucket: Dict[str, Dict[str, Any]] = {}
    question_years = {match.group(0) for match in YEAR_VALUE_RE.finditer(question or "")}

    for doc_rank, doc in enumerate(source_docs[:max(1, max_docs)]):
        title = _doc_title(doc)
        body = _doc_body(doc)
        for sentence_rank, sentence in enumerate(_split_sentences(body)):
            surface = f"{title} {sentence}"
            rel = _relation_score(contract, surface)
            subj = _candidate_question_overlap(subject_tokens, surface)
            q_overlap = _candidate_question_overlap(question_tokens, surface)
            if rel <= 0.0 and granularity in {"setting_period", "month"} and subj < 0.18:
                continue
            rank_bonus = max(0.0, 0.08 - 0.012 * doc_rank - 0.004 * sentence_rank)
            base = 0.28 + 0.34 * rel + 0.18 * subj + 0.12 * q_overlap + rank_bonus
            if granularity == "setting_period":
                for match in DECADE_VALUE_RE.finditer(sentence):
                    value = match.group(0)
                    if value in question_years:
                        continue
                    _add_candidate(
                        bucket,
                        candidate=f"the {value.lower()}",
                        score=base + 0.09,
                        reason="setting_period_decade",
                        title=title,
                        sentence=sentence,
                        source_rank=doc_rank,
                        relation_score=rel,
                        question_overlap=q_overlap,
                    )
                for match in YEAR_VALUE_RE.finditer(sentence):
                    value = match.group(0)
                    if value in question_years:
                        continue
                    _add_candidate(
                        bucket,
                        candidate=value,
                        score=base,
                        reason="setting_period_year",
                        title=title,
                        sentence=sentence,
                        source_rank=doc_rank,
                        relation_score=rel,
                        question_overlap=q_overlap,
                    )
            elif granularity == "month":
                for match in MONTH_VALUE_RE.finditer(sentence):
                    value = _clean_surface(match.group(0))
                    if not value:
                        continue
                    _add_candidate(
                        bucket,
                        candidate=value,
                        score=base + (0.08 if "-" in value or value.lower().startswith(("early", "mid", "late")) else 0.0),
                        reason="month_value",
                        title=title,
                        sentence=sentence,
                        source_rank=doc_rank,
                        relation_score=rel,
                        question_overlap=q_overlap,
                    )
            elif granularity == "year":
                for match in YEAR_VALUE_RE.finditer(sentence):
                    value = match.group(0)
                    if value in question_years:
                        continue
                    _add_candidate(
                        bucket,
                        candidate=value,
                        score=base,
                        reason="year_value",
                        title=title,
                        sentence=sentence,
                        source_rank=doc_rank,
                        relation_score=rel,
                        question_overlap=q_overlap,
                    )
            else:
                date_matches = list(FULL_DATE_VALUE_RE.finditer(sentence))
                if date_matches:
                    for match in date_matches:
                        _add_candidate(
                            bucket,
                            candidate=match.group(0),
                            score=base + 0.06,
                            reason="full_date_value",
                            title=title,
                            sentence=sentence,
                            source_rank=doc_rank,
                            relation_score=rel,
                            question_overlap=q_overlap,
                        )
                else:
                    for match in YEAR_VALUE_RE.finditer(sentence):
                        value = match.group(0)
                        if value in question_years:
                            continue
                        _add_candidate(
                            bucket,
                            candidate=value,
                            score=base,
                            reason="date_year_value",
                            title=title,
                            sentence=sentence,
                            source_rank=doc_rank,
                            relation_score=rel,
                            question_overlap=q_overlap,
                        )
    return sorted(bucket.values(), key=lambda x: (x["score"], x["source_count"], x["provenance_count"]), reverse=True)


def _option_candidates(
    question: str,
    answer: str,
    source_docs: Sequence[str],
    contract: Dict[str, Any],
    max_docs: int,
) -> List[Dict[str, Any]]:
    options = contract.get("options") or []
    if not options:
        return []
    question_tokens = _token_set(question)
    bucket: Dict[str, Dict[str, Any]] = {}
    answer_key = normalize_answer(answer or "")
    for option in options:
        option_key = normalize_answer(option)
        if option_key and option_key in answer_key:
            _add_candidate(
                bucket,
                candidate=option,
                score=0.84,
                reason="option_in_answer",
                relation_score=0.0,
                question_overlap=1.0,
                source="option_contract",
            )
    for doc_rank, doc in enumerate(source_docs[:max(1, max_docs)]):
        title = _doc_title(doc)
        body = _doc_body(doc)
        for sentence_rank, sentence in enumerate(_split_sentences(body)):
            surface_norm = normalize_answer(f"{title} {sentence}")
            rel = _relation_score(contract, sentence)
            q_overlap = _candidate_question_overlap(question_tokens, sentence)
            for option in options:
                option_key = normalize_answer(option)
                if not option_key or option_key not in surface_norm:
                    continue
                score = 0.55 + 0.16 * rel + 0.13 * q_overlap + max(0.0, 0.08 - 0.012 * doc_rank)
                _add_candidate(
                    bucket,
                    candidate=option,
                    score=score,
                    reason="option_evidence_match",
                    title=title,
                    sentence=sentence,
                    source_rank=doc_rank,
                    relation_score=rel,
                    question_overlap=q_overlap,
                    source="option_contract",
                )
    return sorted(bucket.values(), key=lambda x: (x["score"], x["source_count"], x["provenance_count"]), reverse=True)


def _extract_both_entities(question: str) -> List[str]:
    match = re.search(r"\bboth\s+(.+?)\s+and\s+(.+?)(?:\?|$)", question or "", re.I)
    if not match:
        match = re.search(r"\bby both\s+(.+?)\s+and\s+(.+?)(?:\?|$)", question or "", re.I)
    if not match:
        return []
    entities = []
    for group in match.groups():
        value = _clean_surface(re.sub(r"\b(?:worked|on|by|the|a|an)\b", " ", group, flags=re.I))
        value = re.sub(r"\s+", " ", value).strip()
        if len(_tokens(value)) >= 1:
            entities.append(value)
    return entities[:2]


def _shared_work_candidates(
    question: str,
    source_docs: Sequence[str],
    max_docs: int,
) -> List[Dict[str, Any]]:
    entities = _extract_both_entities(question)
    if len(entities) < 2:
        return []
    entity_keys = [normalize_answer(entity) for entity in entities]
    entity_last_keys = [key.split()[-1] for key in entity_keys if key.split()]
    question_tokens = _token_set(question)
    bucket: Dict[str, Dict[str, Any]] = {}

    for doc_rank, doc in enumerate(source_docs[:max(1, max_docs)]):
        title = _doc_title(doc)
        body = _doc_body(doc)
        title_key = normalize_answer(title)
        if not title_key or any(key and key in title_key for key in entity_keys):
            continue
        body_key = normalize_answer(body)
        full_hits = sum(1 for key in entity_keys if key and key in body_key)
        last_hits = sum(1 for key in entity_last_keys if key and re.search(rf"\b{re.escape(key)}\b", body_key))
        if full_hits < 2 and last_hits < 2:
            continue
        title_candidate = _clean_work_title(title)
        best_sentence = ""
        best_sentence_score = 0.0
        for sentence_rank, sentence in enumerate(_split_sentences(body)):
            sent_key = normalize_answer(sentence)
            hit_count = sum(1 for key in entity_keys if key and key in sent_key)
            hit_count = max(hit_count, sum(1 for key in entity_last_keys if key and re.search(rf"\b{re.escape(key)}\b", sent_key)))
            rel = 1.0 if re.search(r"\b(?:screenplay|film|music|art direction|worked|by|composer|director)\b", sentence, re.I) else 0.0
            q_overlap = _candidate_question_overlap(question_tokens, sentence)
            sentence_score = 0.48 + 0.18 * hit_count + 0.12 * rel + 0.10 * q_overlap - 0.01 * doc_rank
            if sentence_score > best_sentence_score:
                best_sentence_score = sentence_score
                best_sentence = sentence
        score = max(0.0, min(1.0, 0.58 + 0.14 * full_hits + 0.10 * last_hits - 0.015 * doc_rank))
        if best_sentence_score:
            score = max(score, best_sentence_score)
        _add_candidate(
            bucket,
            candidate=title_candidate,
            score=score,
            reason="shared_work_intersection",
            title=title,
            sentence=best_sentence,
            source_rank=doc_rank,
            relation_score=1.0,
            question_overlap=_candidate_question_overlap(question_tokens, body),
            source="shared_work_contract",
        )
    return sorted(bucket.values(), key=lambda x: (x["score"], x["source_count"], x["provenance_count"]), reverse=True)


def _is_broad_occupation(answer: str) -> bool:
    key = _candidate_key(answer)
    if key in BROAD_OCCUPATION_KEYS:
        return True
    tokens = set(_tokens(answer))
    return bool(tokens & BROAD_OCCUPATION_KEYS and not tokens & SPECIFIC_OCCUPATION_KEYS)


def _is_specific_occupation(answer: str) -> bool:
    key = _candidate_key(answer)
    if key in SPECIFIC_OCCUPATION_KEYS:
        return True
    return bool(set(_tokens(answer)) & SPECIFIC_OCCUPATION_KEYS)


def _occupation_candidates(
    question: str,
    answer: str,
    source_docs: Sequence[str],
    contract: Dict[str, Any],
    max_docs: int,
    min_question_overlap: float,
    require_role_constraint: bool,
) -> List[Dict[str, Any]]:
    question_tokens = _token_set(question)
    bucket: Dict[str, Dict[str, Any]] = {}
    raw_candidates = _rank_candidates(question, answer, source_docs, "occupation", max_docs=max_docs)
    filtered, _ = _filter_candidates(
        raw_candidates,
        question=question,
        target_type="occupation",
        min_question_overlap=max(0.0, min_question_overlap),
        require_role_constraint=require_role_constraint,
    )
    for raw in filtered[:24]:
        candidate = _clean_surface(raw.get("candidate", ""))
        if not OCCUPATION_VALUE_RE.search(candidate):
            evidence = _candidate_evidence_text(raw)
            match = OCCUPATION_VALUE_RE.search(evidence or "")
            if match:
                candidate = match.group(0)
            else:
                continue
        support = float(raw.get("support_score", raw.get("score", 0.0)) or 0.0)
        relation = float(raw.get("relation_score", 0.0) or 0.0)
        q_overlap = float(raw.get("question_overlap", 0.0) or 0.0)
        score = 0.48 + 0.34 * support + 0.08 * relation + 0.06 * q_overlap
        if _is_specific_occupation(candidate):
            score += 0.06
        _add_candidate(
            bucket,
            candidate=candidate,
            score=score,
            reason="ranked_occupation_candidate",
            title=str(raw.get("title", "") or ""),
            sentence=str(raw.get("sentence_preview", "") or ""),
            relation_score=relation,
            question_overlap=q_overlap,
            source="occupation_contract",
        )

    for doc_rank, doc in enumerate(source_docs[:max(1, max_docs)]):
        title = _doc_title(doc)
        body = _doc_body(doc)
        title_match = OCCUPATION_VALUE_RE.fullmatch(_clean_surface(title))
        if title_match:
            score = 0.66 + (0.08 if _is_specific_occupation(title) else 0.0) - 0.012 * doc_rank
            _add_candidate(
                bucket,
                candidate=title_match.group(0),
                score=score,
                reason="occupation_title_candidate",
                title=title,
                sentence="",
                source_rank=doc_rank,
                relation_score=0.5,
                question_overlap=0.0,
                source="occupation_contract",
            )
        for sentence_rank, sentence in enumerate(_split_sentences(body)):
            if not OCCUPATION_CONTEXT_RE.search(sentence):
                continue
            rel = 1.0
            q_overlap = _candidate_question_overlap(question_tokens, sentence)
            for match in OCCUPATION_VALUE_RE.finditer(sentence):
                candidate = match.group(0)
                score = 0.50 + 0.18 * rel + 0.10 * q_overlap + max(0.0, 0.08 - 0.012 * doc_rank)
                if _is_specific_occupation(candidate):
                    score += 0.08
                _add_candidate(
                    bucket,
                    candidate=candidate,
                    score=score,
                    reason="occupation_sentence_candidate",
                    title=title,
                    sentence=sentence,
                    source_rank=doc_rank,
                    relation_score=rel,
                    question_overlap=q_overlap,
                    source="occupation_contract",
                )
    return sorted(bucket.values(), key=lambda x: (x["score"], x["source_count"], x["provenance_count"]), reverse=True)


def _intersection_candidates(
    question: str,
    answer: str,
    source_docs: Sequence[str],
    max_docs: int,
) -> List[Dict[str, Any]]:
    if not INTERSECTION_QUESTION_RE.search(question or ""):
        return []
    bucket: Dict[str, Dict[str, Any]] = {}
    question_tokens = _token_set(question)
    for match in STREET_INTERSECTION_RE.finditer(answer or ""):
        candidate = _clean_surface(match.group(1))
        if not candidate:
            continue
        _add_candidate(
            bucket,
            candidate=candidate,
            score=0.94,
            reason="intersection_surface_projection",
            title="",
            sentence=str(answer or ""),
            relation_score=1.0,
            question_overlap=1.0,
            source="intersection_contract",
        )

    for doc_rank, doc in enumerate(source_docs[:max(1, max_docs)]):
        title = _doc_title(doc)
        body = _doc_body(doc)
        for sentence_rank, sentence in enumerate(_split_sentences(body)):
            if not INTERSECTION_QUESTION_RE.search(sentence) and " and " not in sentence:
                continue
            q_overlap = _candidate_question_overlap(question_tokens, sentence)
            for match in STREET_INTERSECTION_RE.finditer(sentence):
                candidate = _clean_surface(match.group(1))
                if not candidate:
                    continue
                score = 0.70 + 0.12 * q_overlap + max(0.0, 0.08 - 0.012 * doc_rank - 0.004 * sentence_rank)
                _add_candidate(
                    bucket,
                    candidate=candidate,
                    score=score,
                    reason="intersection_evidence_projection",
                    title=title,
                    sentence=sentence,
                    source_rank=doc_rank,
                    relation_score=1.0 if INTERSECTION_QUESTION_RE.search(sentence) else 0.6,
                    question_overlap=q_overlap,
                    source="intersection_contract",
                )
    return sorted(bucket.values(), key=lambda x: (x["score"], x["source_count"], x["provenance_count"]), reverse=True)


def _role_span_candidates(
    question: str,
    answer: str,
    source_docs: Sequence[str],
    max_docs: int,
) -> List[Dict[str, Any]]:
    if not ROLE_SPAN_QUESTION_RE.search(question or ""):
        return []
    bucket: Dict[str, Dict[str, Any]] = {}
    question_tokens = _token_set(question)
    answer_key = normalize_answer(answer or "")
    for doc_rank, doc in enumerate(source_docs[:max(1, max_docs)]):
        title = _doc_title(doc)
        body = _doc_body(doc)
        for sentence_rank, sentence in enumerate(_split_sentences(body)):
            sent_norm = normalize_answer(sentence)
            role_context = bool(
                re.search(r"\b(?:right|left)[-\s]?handed\b", sentence, re.I)
                or re.search(r"\b(?:batsman|batter|bowler|all[-\s]?rounder|wicket[-\s]?keeper)\b", sentence, re.I)
            )
            if not role_context:
                continue
            q_overlap = _candidate_question_overlap(question_tokens, sentence)
            for match in ROLE_SPAN_VALUE_RE.finditer(sentence):
                candidate = _clean_surface(match.group(0))
                candidate = re.sub(r"^(?:right|left)[-\s]?handed\s+", "", candidate, flags=re.I).strip()
                if not candidate:
                    continue
                cand_key = normalize_answer(candidate)
                if cand_key == answer_key:
                    continue
                if answer_key and answer_key not in cand_key and cand_key not in answer_key:
                    continue
                score = 0.78 + 0.10 * q_overlap + max(0.0, 0.08 - 0.012 * doc_rank - 0.004 * sentence_rank)
                if re.search(r"\b(?:middle|top|lower|opening|right|left|wicket|fast|spin|order)\b", candidate, re.I):
                    score += 0.06
                _add_candidate(
                    bucket,
                    candidate=candidate,
                    score=score,
                    reason="role_span_specificity_projection",
                    title=title,
                    sentence=sentence,
                    source_rank=doc_rank,
                    relation_score=1.0,
                    question_overlap=q_overlap,
                    source="role_span_contract",
                )
    return sorted(bucket.values(), key=lambda x: (x["score"], x["source_count"], x["provenance_count"]), reverse=True)


def _collect_contract_candidates(
    question: str,
    answer: str,
    source_docs: Sequence[str],
    contract: Dict[str, Any],
    max_docs: int,
    max_candidates: int,
    min_question_overlap: float,
    require_role_constraint: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    diagnostics: Dict[str, Any] = {}
    parts: List[List[Dict[str, Any]]] = []
    if contract.get("requires_option"):



        comparison_rows: List[Dict[str, Any]] = []
        diagnostics["option_comparison_candidate_count"] = len(comparison_rows)
        diagnostics["option_comparison_status"] = "disabled_untrusted_role_grounding"
        option_rows = _option_candidates(question, answer, source_docs, contract, max_docs)
        parts.append(option_rows)
        diagnostics["option_candidate_count"] = len(option_rows)
    if contract.get("shared_work"):
        shared_rows = _shared_work_candidates(question, source_docs, max_docs)
        parts.append(shared_rows)
        diagnostics["shared_work_candidate_count"] = len(shared_rows)
    if contract.get("candidate_target_type") in DATE_NUMBER_TYPES or contract.get("temporal_granularity") in {
        "setting_period",
        "month",
        "year",
    }:
        temporal_rows = _extract_temporal_candidates(question, source_docs, contract, max_docs)
        parts.append(temporal_rows)
        diagnostics["temporal_candidate_count"] = len(temporal_rows)
    if contract.get("answer_slot") == "occupation":
        occupation_rows = _occupation_candidates(
            question,
            answer,
            source_docs,
            contract,
            max_docs,
            min_question_overlap,
            require_role_constraint,
        )
        parts.append(occupation_rows)
        diagnostics["occupation_candidate_count"] = len(occupation_rows)
    if contract.get("intersection"):
        intersection_rows = _intersection_candidates(question, answer, source_docs, max_docs)
        parts.append(intersection_rows)
        diagnostics["intersection_candidate_count"] = len(intersection_rows)
    if contract.get("role_span"):
        role_rows = _role_span_candidates(question, answer, source_docs, max_docs)
        parts.append(role_rows)
        diagnostics["role_span_candidate_count"] = len(role_rows)

    for rows in parts:
        for row in rows:
            key = row.get("key")
            if not key:
                continue
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = dict(row)
                continue
            existing["score"] = round(max(float(existing.get("score", 0.0) or 0.0), float(row.get("score", 0.0) or 0.0)), 4)
            existing["support_score"] = round(max(float(existing.get("support_score", 0.0) or 0.0), float(row.get("support_score", 0.0) or 0.0)), 4)
            existing["relation_score"] = round(max(float(existing.get("relation_score", 0.0) or 0.0), float(row.get("relation_score", 0.0) or 0.0)), 4)
            existing["question_overlap"] = round(max(float(existing.get("question_overlap", 0.0) or 0.0), float(row.get("question_overlap", 0.0) or 0.0)), 4)
            existing["source_count"] = int(existing.get("source_count", 0) or 0) + int(row.get("source_count", 1) or 1)
            existing["provenance_count"] = int(existing.get("provenance_count", 0) or 0) + int(row.get("provenance_count", 1) or 1)
            for reason in row.get("reasons") or []:
                if reason not in existing.setdefault("reasons", []):
                    existing["reasons"].append(reason)
            existing.setdefault("examples", []).extend((row.get("examples") or [])[: max(0, 3 - len(existing.get("examples") or []))])

    rows = list(by_key.values())
    rows.sort(key=lambda x: (float(x.get("score", 0.0) or 0.0), int(x.get("source_count", 0) or 0)), reverse=True)
    return rows[:max(1, max_candidates)], diagnostics


def _answer_relation_support(question: str, answer: str, source_docs: Sequence[str], contract: Dict[str, Any], max_docs: int) -> float:
    answer_key = _candidate_key(answer)
    if not answer_key:
        return 0.0
    best = 0.0
    for doc in source_docs[:max(1, max_docs)]:
        title = _doc_title(doc)
        for sentence in _split_sentences(_doc_body(doc)):
            surface = f"{title} {sentence}"
            if answer_key not in normalize_answer(surface):
                continue
            best = max(best, _relation_score(contract, surface))
    return best


def _current_contains_single_option(answer: str, options: Sequence[str]) -> Optional[str]:
    answer_key = normalize_answer(answer or "")
    hits = []
    for option in options:
        option_key = normalize_answer(option)
        if option_key and option_key in answer_key:
            hits.append(option)
    if len(hits) == 1:
        return hits[0]
    return None


def _extract_role_entities_for_option(
    option: str,
    source_docs: Sequence[str],
    max_docs: int,
) -> List[Dict[str, Any]]:
    option_key = normalize_answer(option)
    if not option_key:
        return []
    rows: List[Dict[str, Any]] = []
    patterns = (
        re.compile(
            r"\b(?:directed|written|composed|created|produced)\s+by\s+"
            r"([A-Z][A-Za-z0-9'. -]{2,80}?)(?:,|\band\b|\bwith\b|\bstarring\b|\.|$)",
            re.I,
        ),
        re.compile(
            r"\b(?:director|writer|author|composer|creator|producer)\s+(?:is|was|:)?\s*"
            r"([A-Z][A-Za-z0-9'. -]{2,80}?)(?:,|\band\b|\bwith\b|\bstarring\b|\.|$)",
            re.I,
        ),
    )
    seen = set()
    for doc_rank, doc in enumerate(source_docs[:max(1, max_docs)]):
        title = _doc_title(doc)
        body = _doc_body(doc)
        doc_norm = normalize_answer(f"{title} {body[:600]}")
        if option_key not in doc_norm and normalize_answer(_clean_work_title(option)) not in doc_norm:
            continue
        for sentence in _split_sentences(body):
            sent_norm = normalize_answer(sentence)
            if option_key not in doc_norm and option_key not in sent_norm:
                continue
            for pattern in patterns:
                for match in pattern.finditer(sentence):
                    value = _clean_surface(match.group(1))
                    value = re.sub(
                        r"\b(?:and|with|starring|featuring|based on|from)\b.*$",
                        "",
                        value,
                        flags=re.I,
                    ).strip()
                    if not value or len(_tokens(value)) > 6:
                        continue
                    key = _candidate_key(value)
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    rows.append({
                        "entity": value,
                        "key": key,
                        "title": title,
                        "sentence": sentence,
                        "source_rank": doc_rank,
                    })
    return rows[:4]


def _temporal_values_for_option(
    option: str,
    role_entities: Sequence[Dict[str, Any]],
    question: str,
    source_docs: Sequence[str],
    contract: Dict[str, Any],
    max_docs: int,
) -> List[Dict[str, Any]]:
    option_key = normalize_answer(option)
    option_title_key = normalize_answer(_clean_work_title(option))
    role_keys = [str(row.get("key") or "") for row in role_entities if row.get("key")]
    relation_labels = set(contract.get("relations") or [])
    temporal_relation_labels = {
        label
        for label in relation_labels
        if label in {"birth_time", "death_time", "release_time", "completion_time", "setting_time", "start_time"}
    }
    evidence_patterns: List[re.Pattern] = []
    for label, _, evidence_pattern in RELATION_PROFILES:
        if label in temporal_relation_labels:
            evidence_patterns.append(evidence_pattern)
    if not evidence_patterns:
        evidence_patterns = [TEMPORAL_RELATION_RE]

    values: List[Dict[str, Any]] = []
    seen = set()
    for doc_rank, doc in enumerate(source_docs[:max(1, max_docs)]):
        title = _doc_title(doc)
        title_norm = normalize_answer(title)
        body = _doc_body(doc)
        doc_head_norm = normalize_answer(f"{title} {body[:800]}")
        title_hits_option = bool(
            option_key and (option_key in title_norm or option_title_key in title_norm)
        )
        title_hits_role = any(key and key in title_norm for key in role_keys)
        doc_mentions_option = bool(
            option_key and (option_key in doc_head_norm or option_title_key in doc_head_norm)
        )
        doc_mentions_role = any(key and key in doc_head_norm for key in role_keys)
        if not (title_hits_option or title_hits_role or doc_mentions_option or doc_mentions_role):
            continue
        for sentence_rank, sentence in enumerate(_split_sentences(body)):
            sent_norm = normalize_answer(sentence)
            option_hit = bool(option_key and (option_key in sent_norm or option_title_key in sent_norm))
            role_hit = any(key and key in sent_norm for key in role_keys)
            if not (title_hits_option or title_hits_role or option_hit or role_hit):
                continue
            relation_hit = any(pattern.search(sentence) for pattern in evidence_patterns)
            if temporal_relation_labels and not relation_hit:
                continue
            date_surfaces = [match.group(0) for match in FULL_DATE_VALUE_RE.finditer(sentence)]
            if not date_surfaces:
                date_surfaces = [match.group(0) for match in YEAR_VALUE_RE.finditer(sentence)]
            for surface in date_surfaces:
                parsed = _parse_temporal_value(surface)
                if parsed is None:
                    continue
                key = (surface, parsed, title, sentence[:120])
                if key in seen:
                    continue
                seen.add(key)
                support = 0.58
                if title_hits_role or role_hit:
                    support += 0.18
                if title_hits_option or option_hit:
                    support += 0.12
                if relation_hit:
                    support += 0.14
                support += max(0.0, 0.08 - 0.012 * doc_rank - 0.004 * sentence_rank)
                values.append({
                    "value": surface,
                    "parsed": parsed,
                    "score": round(min(1.0, support), 4),
                    "title": title,
                    "sentence": sentence,
                    "source_rank": doc_rank,
                    "relation_hit": bool(relation_hit),
                    "role_hit": bool(title_hits_role or role_hit),
                    "option_hit": bool(title_hits_option or option_hit),
                })
    values.sort(key=lambda row: (row["score"], row["parsed"]), reverse=True)
    return values[:5]


def _option_comparison_candidates(
    question: str,
    answer: str,
    source_docs: Sequence[str],
    contract: Dict[str, Any],
    max_docs: int,
) -> List[Dict[str, Any]]:
    options = contract.get("options") or []
    direction = contract.get("comparison_direction")
    if len(options) != 2 or direction not in {"min", "max"}:
        return []
    if not set(contract.get("relations") or []) & {
        "birth_time",
        "death_time",
        "release_time",
        "completion_time",
        "setting_time",
        "start_time",
    }:
        return []

    option_values: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = []
    for option in options:
        role_entities = _extract_role_entities_for_option(option, source_docs, max_docs)
        values = _temporal_values_for_option(
            option,
            role_entities,
            question,
            source_docs,
            contract,
            max_docs,
        )
        if not values:
            return []
        option_values.append((option, values[0], role_entities))

    if len(option_values) != 2:
        return []
    left, right = option_values
    left_value = left[1]["parsed"]
    right_value = right[1]["parsed"]
    if left_value == right_value:
        return []
    selected = left if (left_value < right_value) == (direction == "min") else right
    other = right if selected is left else left
    selected_option, selected_value, selected_roles = selected
    other_option, other_value, other_roles = other
    score = min(0.98, 0.83 + 0.06 * selected_value.get("score", 0.0) + 0.04 * other_value.get("score", 0.0))
    bucket: Dict[str, Dict[str, Any]] = {}
    _add_candidate(
        bucket,
        candidate=selected_option,
        score=score,
        reason="option_temporal_comparison",
        title=str(selected_value.get("title") or ""),
        sentence=str(selected_value.get("sentence") or ""),
        source_rank=int(selected_value.get("source_rank", 0) or 0),
        relation_score=1.0,
        question_overlap=1.0,
        source="option_comparison_contract",
    )
    row = next(iter(bucket.values()), None)
    if row is not None:
        row["comparison"] = {
            "direction": direction,
            "selected_option": selected_option,
            "selected_value": selected_value.get("value"),
            "selected_parsed": selected_value.get("parsed"),
            "selected_role_entities": [r.get("entity") for r in selected_roles],
            "other_option": other_option,
            "other_value": other_value.get("value"),
            "other_parsed": other_value.get("parsed"),
            "other_role_entities": [r.get("entity") for r in other_roles],
        }
        return [row]
    return []


def _contract_slot_allows_candidate(
    candidate: str,
    contract: Dict[str, Any],
    question: str,
) -> Tuple[bool, str]:
    slot = str(contract.get("answer_slot") or "")
    target_type = str(contract.get("candidate_target_type") or "person")
    if contract.get("temporal_granularity") == "setting_period" and (
        DECADE_VALUE_RE.search(candidate or "") or YEAR_ONLY_RE.match(candidate or "")
    ):
        return True, ""
    if contract.get("temporal_granularity") == "month" and MONTH_VALUE_RE.search(candidate or ""):
        return True, ""
    if contract.get("requires_option"):
        return True, ""
    if contract.get("intersection") and STREET_INTERSECTION_RE.search(candidate or ""):
        return True, ""
    if contract.get("role_span") and ROLE_SPAN_VALUE_RE.search(candidate or ""):
        return True, ""
    ok, reason = _candidate_passes_answer_slot(candidate, slot, target_type, question=question, evidence_text="")
    if slot == "occupation" and OCCUPATION_VALUE_RE.search(candidate or ""):
        return True, ""
    return ok, reason


def _selected_reason(
    before: str,
    top: Dict[str, Any],
    contract: Dict[str, Any],
    answer_supported: bool,
    answer_relation: float,
    max_answer_tokens: int,
) -> str:
    after = str(top.get("candidate", "") or "")
    reasons = set(top.get("reasons") or [])
    if "option_temporal_comparison" in reasons:
        return "option_temporal_comparison_rewrite"
    if contract.get("requires_option"):
        single_option = _current_contains_single_option(before, contract.get("options") or [])
        if single_option and _same_answer(single_option, after):
            return "option_format_repair"
        return "option_contract_rewrite"
    if "intersection_surface_projection" in reasons or "intersection_evidence_projection" in reasons:
        return "intersection_surface_projection"
    if "role_span_specificity_projection" in reasons:
        return "role_span_specificity_projection"
    if contract.get("shared_work"):
        return "shared_work_intersection_rewrite"
    if contract.get("temporal_granularity") == "setting_period":
        if answer_relation <= 0.0:
            return "temporal_relation_repair"
        return "setting_granularity_repair"
    if contract.get("temporal_granularity") == "month":
        before_key = normalize_answer(before)
        after_key = normalize_answer(after)
        if before_key and before_key in after_key and before_key != after_key:
            return "month_specificity_repair"
        return "month_relation_repair"
    if contract.get("answer_slot") == "occupation":
        if _is_broad_occupation(before) and _is_specific_occupation(after):
            return "occupation_specificity_repair"
        return "occupation_contract_rewrite"
    if ABSTAIN_RE.search(before or "") or _looks_verbose_or_multi_answer(before, max_answer_tokens):
        return "format_or_abstain_repair"
    if not answer_supported:
        return "unsupported_answer_repair"
    return "contract_rewrite"


def arbitrate_answer_with_query_answer_contract(
    *,
    question: str,
    answer: str,
    qa_docs: Sequence[str],
    answer_style: Optional[str] = None,
    enabled: bool = False,
    apply_rewrite: bool = True,
    min_score: float = 0.78,
    generic_min_score: float = 0.82,
    min_margin: float = 0.05,
    min_support_gain: float = 0.03,
    min_relation_score: float = 0.50,
    max_docs: int = 8,
    max_answer_tokens: int = 14,
    max_candidates: int = 12,
    min_question_overlap: float = 0.02,
    require_role_constraint: bool = True,
    protect_supported_answers: bool = True,
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

    source_docs = [
        doc for doc in (qa_docs or [])
        if isinstance(doc, str) and doc.strip() and not _is_synthetic_doc(doc)
    ]
    contract = _compile_answer_contract(question or "", answer_style)
    meta: Dict[str, Any] = {
        "enabled": True,
        "changed": False,
        "action": "keep",
        "reason": "init",
        "before": before,
        "after": before,
        "source_doc_count": len(source_docs),
        "contract": contract,
    }
    if not source_docs:
        meta["reason"] = "no_source_docs"
        return before, meta
    if not contract.get("active"):
        meta["reason"] = "inactive_contract"
        return before, meta
    if BOOLEAN_ANSWER_RE.match(before) or BOOLEAN_QUESTION_RE.match(question or ""):
        meta["reason"] = "boolean_protected"
        return before, meta

    answer_supported, answer_support_score = _is_supported(
        question,
        before,
        source_docs,
        str(contract.get("candidate_target_type") or "person"),
        max_docs=max_docs,
    )
    answer_relation = _answer_relation_support(question, before, source_docs, contract, max_docs)
    answer_arity = _infer_answer_arity(question or "", before, str(contract.get("candidate_target_type") or "person"))
    meta["answer_state"] = {
        "supported": bool(answer_supported),
        "support_score": round(float(answer_support_score or 0.0), 4),
        "relation_score": round(float(answer_relation or 0.0), 4),
        "verbose_or_multi": bool(_looks_verbose_or_multi_answer(before, max_answer_tokens)),
        "arity": answer_arity,
        "token_count": _token_count(before),
    }
    if answer_arity not in {"single", "multi_value"} and not contract.get("requires_option"):
        meta["reason"] = "multi_slot_or_list_protected"
        return before, meta

    scored, collect_meta = _collect_contract_candidates(
        question=question or "",
        answer=before,
        source_docs=source_docs,
        contract=contract,
        max_docs=max_docs,
        max_candidates=max_candidates,
        min_question_overlap=min_question_overlap,
        require_role_constraint=require_role_constraint,
    )
    meta.update(collect_meta)
    meta["candidate_count"] = len(scored)
    if not scored:
        meta["reason"] = "no_contract_candidates"
        return before, meta

    slot_filtered: List[Dict[str, Any]] = []
    slot_reject_counts: Counter = Counter()
    for candidate in scored:
        ok, reason = _contract_slot_allows_candidate(
            str(candidate.get("candidate", "") or ""),
            contract,
            question or "",
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
    after = _clean_surface(top.get("candidate", ""))
    margin = float(top.get("score", 0.0) or 0.0) - float((second or {}).get("score", 0.0) or 0.0)
    support_gain = float(top.get("support_score", 0.0) or 0.0) - float(answer_support_score or 0.0)
    top_relation = float(top.get("relation_score", 0.0) or 0.0)
    same = _same_answer(before, after)
    containment_alias = _is_containment_variant(before, after)
    meta.update({
        "selected": top,
        "second": second,
        "selected_score": round(float(top.get("score", 0.0) or 0.0), 4),
        "selected_relation_score": round(top_relation, 4),
        "margin": round(margin, 4),
        "support_gain": round(support_gain, 4),
        "candidates": scored[: min(len(scored), max_candidates)],
    })
    top_reasons = set(top.get("reasons") or [])

    if same:
        meta["reason"] = "same_answer"
        return before, meta

    if protect_supported_answers and answer_supported and not _looks_verbose_or_multi_answer(before, max_answer_tokens):
        if not contract.get("requires_option") and not (
            contract.get("temporal_granularity") in {"setting_period", "month"} and top_relation >= min_relation_score > answer_relation
        ) and not (
            contract.get("intersection") and bool(top_reasons & {"intersection_surface_projection", "intersection_evidence_projection"})
        ) and not (
            contract.get("role_span") and "role_span_specificity_projection" in top_reasons
        ):
            meta["reason"] = "supported_answer_protected"
            return before, meta

    floor = min_score if contract.get("candidate_target_type") in DATE_NUMBER_TYPES else generic_min_score
    rewrite_reason = _selected_reason(
        before,
        top,
        contract,
        bool(answer_supported),
        float(answer_relation or 0.0),
        max_answer_tokens,
    )

    single_option = _current_contains_single_option(before, contract.get("options") or [])
    option_format_repair = bool(
        contract.get("requires_option")
        and single_option
        and _same_answer(single_option, after)
        and not _same_answer(before, single_option)
    )
    option_temporal_comparison_repair = bool(
        contract.get("requires_option")
        and "option_temporal_comparison" in top_reasons
        and float(top.get("score", 0.0) or 0.0) >= max(0.88, generic_min_score)
        and (margin >= min_margin or second is None)
    )
    temporal_relation_repair = bool(
        contract.get("temporal_granularity") in {"setting_period", "month"}
        and top_relation >= min_relation_score
        and float(top.get("score", 0.0) or 0.0) >= min_score
        and margin >= min_margin
        and (answer_relation + 0.01 < top_relation or _looks_verbose_or_multi_answer(before, max_answer_tokens))
    )
    occupation_specificity_repair = bool(
        contract.get("answer_slot") == "occupation"
        and not answer_supported
        and _is_broad_occupation(before)
        and _is_specific_occupation(after)
        and float(top.get("score", 0.0) or 0.0) >= 0.92
        and float(top.get("source_count", 0) or 0) >= 2
    )
    intersection_surface_repair = bool(
        contract.get("intersection")
        and bool(top_reasons & {"intersection_surface_projection", "intersection_evidence_projection"})
        and STREET_INTERSECTION_RE.search(after or "")
        and normalize_answer(after) in normalize_answer(before)
        and normalize_answer(after) != normalize_answer(before)
        and float(top.get("score", 0.0) or 0.0) >= 0.86
    )
    role_span_specificity_repair = bool(
        contract.get("role_span")
        and "role_span_specificity_projection" in top_reasons
        and normalize_answer(before) in normalize_answer(after)
        and normalize_answer(before) != normalize_answer(after)
        and float(top.get("score", 0.0) or 0.0) >= 0.88
    )
    shared_work_repair = bool(
        contract.get("shared_work")
        and top_relation >= min_relation_score
        and float(top.get("score", 0.0) or 0.0) >= floor
        and (margin >= min_margin or second is None)
    )
    unsupported_or_verbose_repair = False
    score_gate = bool(
        float(top.get("score", 0.0) or 0.0) >= floor
        and margin >= min_margin
        and (support_gain >= min_support_gain or containment_alias or option_format_repair)
    )
    rewrite_allowed = bool(
        option_format_repair
        or option_temporal_comparison_repair
        or temporal_relation_repair
        or occupation_specificity_repair
        or intersection_surface_repair
        or role_span_specificity_repair
        or shared_work_repair
        or (unsupported_or_verbose_repair and score_gate)
    )

    if not rewrite_allowed:
        meta["reason"] = "below_contract_rewrite_gate"
        meta["gate"] = {
            "floor": round(float(floor), 4),
            "min_margin": round(float(min_margin), 4),
            "min_support_gain": round(float(min_support_gain), 4),
            "option_format_repair": option_format_repair,
            "option_temporal_comparison_repair": option_temporal_comparison_repair,
            "temporal_relation_repair": temporal_relation_repair,
            "occupation_specificity_repair": occupation_specificity_repair,
            "intersection_surface_repair": intersection_surface_repair,
            "role_span_specificity_repair": role_span_specificity_repair,
            "shared_work_repair": shared_work_repair,
            "unsupported_or_verbose_repair": unsupported_or_verbose_repair,
            "score_gate": score_gate,
        }
        return before, meta

    meta["reason"] = rewrite_reason
    meta["action"] = "rewrite" if apply_rewrite else "shadow_rewrite"
    meta["after"] = after
    if not apply_rewrite:
        meta["shadow_changed"] = True
        meta["shadow_after"] = after
        return before, meta
    meta["changed"] = True
    return after, meta
