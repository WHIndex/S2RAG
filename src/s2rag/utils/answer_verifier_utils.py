import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .eval_utils import normalize_answer
from .sentence_evidence_utils import infer_answer_target_type


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "how", "in",
    "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
    "there", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "whom", "whose", "why", "with",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b")
FULL_DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)
MONTH_YEAR_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{4}\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(
    r"\b(?:\$|€|£)?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:%|percent|million|billion|thousand|km|miles|years?|people))?\b",
    re.IGNORECASE,
)
TITLE_WORD = r"[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:[-'’][A-ZÀ-ÖØ-Þ]?[a-zà-öø-ÿ]+)?"
TITLE_CASE_RE = re.compile(
    rf"\b{TITLE_WORD}(?:\s+(?:of|the|and|de|da|do|dos|das|del|la|le|du|van|von|{TITLE_WORD})){{0,6}}\b"
)
ABSTAIN_RE = re.compile(
    r"\b(?:not specified|not provided|not mentioned|cannot determine|cannot be determined|"
    r"unknown|insufficient information|no information|not available)\b",
    re.IGNORECASE,
)
BOOLEAN_SURFACE_RE = re.compile(r"^(?:yes|no|true|false)$", re.IGNORECASE)
QUESTION_YEAR_RE = re.compile(r"\b(?:what|which|in what|in which)\s+year\b", re.IGNORECASE)
QUESTION_OPTION_SLOT_RE = re.compile(
    r"\bwhich\s+(?:film|movie|song|album|book|novel|work|series|episode|team|company|organization|person|one)\b",
    re.IGNORECASE,
)
MONTH_NAME_KEYS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}
BAD_GENERIC_CANDIDATE_KEYS = {
    "anchored typed candidate table",
    "candidate",
    "candidate answer table",
    "wikipedia title",
    "typed evidence grounding",
    "directional fact join candidate table",
    "multi granular evidence memory",
    "entity state evidence control",
    "candidate answer contrastive selector",
    "structured evidence chain",
    "slot separated fact join view",
    "question",
    "rows",
    "overview",
    "statistics",
    "history",
    "background",
    "current",
    "present",
    "annual",
}
NATIONALITY_OR_COUNTRY_RE = re.compile(
    r"\b(?:afghan|albanian|algerian|american|argentine|armenian|australian|"
    r"austrian|belgian|brazilian|british|bulgarian|canadian|chilean|"
    r"chinese|colombian|croatian|cuban|czech|danish|dutch|egyptian|"
    r"english|finnish|french|german|ghanaian|greek|hungarian|indian|"
    r"indonesian|iranian|irish|israeli|italian|jamaican|japanese|korean|"
    r"malaysian|mexican|moroccan|nigerian|norwegian|pakistani|polish|"
    r"portuguese|romanian|russian|scottish|serbian|singaporean|spanish|"
    r"swedish|swiss|syrian|taiwanese|thai|turkish|ukrainian|vietnamese|"
    r"welsh|afghanistan|albania|algeria|argentina|armenia|australia|"
    r"austria|belgium|brazil|britain|bulgaria|canada|chile|china|"
    r"colombia|croatia|cuba|denmark|egypt|england|finland|france|germany|"
    r"ghana|greece|hungary|india|indonesia|iran|ireland|israel|italy|"
    r"jamaica|japan|korea|malaysia|mexico|morocco|nigeria|norway|pakistan|"
    r"poland|portugal|romania|russia|scotland|serbia|singapore|spain|"
    r"sweden|switzerland|syria|taiwan|thailand|turkey|ukraine|vietnam|wales)\b",
    re.IGNORECASE,
)
OCCUPATION_HEAD_RE = re.compile(
    r"\b(?:director\s+of\s+photography|screenwriter|cinematographer|"
    r"photographer|director|producer|composer|author|writer|novelist|poet|"
    r"playwright|actor|actress|singer|musician|artist|painter|journalist|"
    r"historian|professor|teacher|scientist|engineer|mathematician|physician|"
    r"doctor|lawyer|attorney|designer|architect|politician|minister|governor|"
    r"president|mayor|diplomat|businessperson|businessman|businesswoman|"
    r"entrepreneur|footballer|cricketer|basketball\s+player|baseball\s+player|"
    r"player|coach|athlete|wrestler|boxer|racing\s+driver|driver|dancer|"
    r"model|presenter|broadcaster|editor|publisher|researcher|scholar|"
    r"economist|activist|soldier|officer|general|admiral|judge)\b",
    re.IGNORECASE,
)
OCCUPATION_CONTEXT_RE = re.compile(
    r"\b(?:occupation|profession|job|career|worked\s+as|known\s+as|served\s+as|"
    r"is\s+an?|was\s+an?|became\s+an?)\b",
    re.IGNORECASE,
)
RELATION_CUE_PATTERNS = (
    (
        re.compile(r"\b(?:won|win|winner|victory|champion|championship)\b", re.IGNORECASE),
        re.compile(r"\b(?:won|win|winner|victory|champion|championship|belongs\s+to|awarded\s+to)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\b(?:founded|formed|established|created|opened|open)\b", re.IGNORECASE),
        re.compile(r"\b(?:founded|formed|established|created|opened|open|started|began)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\b(?:born|birthplace|place\s+of\s+birth)\b", re.IGNORECASE),
        re.compile(r"\b(?:born|birthplace|place\s+of\s+birth|native\s+of)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\b(?:died|death|burial|buried)\b", re.IGNORECASE),
        re.compile(r"\b(?:died|death|burial|buried|tomb|interred)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\b(?:signed|signatory|contract)\b", re.IGNORECASE),
        re.compile(r"\b(?:signed|signatory|contract|joined|contracted)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\b(?:author|writer|wrote|written|composer|composed|director|directed)\b", re.IGNORECASE),
        re.compile(r"\b(?:author|writer|wrote|written|composer|composed|director|directed|by)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\b(?:nationality|citizenship|citizen|national)\b", re.IGNORECASE),
        re.compile(r"\b(?:nationality|citizenship|citizen|national|born|from)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\b(?:located|location|where|country|city|state|province)\b", re.IGNORECASE),
        re.compile(r"\b(?:located|location|in|city|country|state|province|region|island)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\b(?:released|published|publication|premiered)\b", re.IGNORECASE),
        re.compile(r"\b(?:released|published|publication|premiered|came\s+out)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\b(?:child|children|father|mother|parent|son|daughter|spouse|wife|husband)\b", re.IGNORECASE),
        re.compile(r"\b(?:child|children|father|mother|parent|son|daughter|spouse|wife|husband|eldest|youngest)\b", re.IGNORECASE),
    ),
)


def _tokens(text: Any) -> List[str]:
    return [
        token
        for token in TOKEN_RE.findall(str(text or "").lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def _token_set(text: Any) -> set:
    return set(_tokens(text))


def _doc_title(doc: str) -> str:
    if not doc:
        return ""
    return str(doc).split("\n", 1)[0].strip()


def _doc_body(doc: str) -> str:
    if not doc:
        return ""
    parts = str(doc).split("\n", 1)
    return parts[1] if len(parts) > 1 else parts[0]


def _split_sentences(text: str, max_body_chars: int = 10000) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "")[:max_body_chars]).strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    sentences = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) <= 440:
            sentences.append(piece)
        else:
            sentences.append(piece[:440].strip())
    return sentences


def _overlap(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def _candidate_key(text: str) -> str:
    return normalize_answer(str(text or ""))


def _active_relation_patterns(question: str) -> List[re.Pattern]:
    patterns = []
    for question_pattern, evidence_pattern in RELATION_CUE_PATTERNS:
        if question_pattern.search(question or ""):
            patterns.append(evidence_pattern)
    return patterns


def _candidate_char_positions(candidate: str, sentence: str) -> List[int]:
    candidate = str(candidate or "").strip()
    sentence = str(sentence or "")
    if not candidate or not sentence:
        return []
    positions = [match.start() for match in re.finditer(re.escape(candidate), sentence, re.IGNORECASE)]
    if positions:
        return positions
    candidate_tokens = [token for token in TOKEN_RE.findall(candidate.lower()) if len(token) > 2]
    if not candidate_tokens:
        return []
    token_positions = []
    sentence_lower = sentence.lower()
    for token in candidate_tokens:
        match = re.search(rf"\b{re.escape(token)}\b", sentence_lower)
        if match:
            token_positions.append(match.start())
    return token_positions


def _relation_proximity_score(question: str, sentence: str, candidate: str) -> float:
    relation_patterns = _active_relation_patterns(question)
    if not relation_patterns:
        return 0.0
    cue_positions = []
    for pattern in relation_patterns:
        cue_positions.extend(match.start() for match in pattern.finditer(sentence or ""))
    if not cue_positions:
        return 0.0
    candidate_positions = _candidate_char_positions(candidate, sentence)
    if not candidate_positions:
        return 0.0
    min_distance = min(abs(candidate_pos - cue_pos) for candidate_pos in candidate_positions for cue_pos in cue_positions)
    if min_distance <= 50:
        return 0.18
    if min_distance <= 100:
        return 0.13
    if min_distance <= 170:
        return 0.08
    if min_distance <= 260:
        return 0.04
    return 0.0


def _norm_contains(haystack_norm: str, needle_norm: str) -> bool:
    if not haystack_norm or not needle_norm:
        return False
    return bool(re.search(rf"(?:^|\s){re.escape(needle_norm)}(?:\s|$)", haystack_norm))


def _is_supported(
    question: str,
    answer: str,
    qa_docs: List[str],
    target_type: str,
    max_docs: int,
) -> Tuple[bool, float]:
    answer_norm = _candidate_key(answer)
    if not answer_norm:
        return False, 0.0

    best_score = 0.0
    evidence_norm = _candidate_key("\n".join(str(doc) for doc in qa_docs or []))
    if _norm_contains(evidence_norm, answer_norm):
        best_score = 0.25

    for item in _score_sentences(question, qa_docs, target_type, max_docs):
        sent_norm = _candidate_key(f"{item['title']} {item['sentence']}")
        if _norm_contains(sent_norm, answer_norm):
            best_score = max(best_score, float(item["score"]))
    return best_score > 0.0, best_score


def _target_signal(sentence: str, title: str, target_type: str) -> float:
    text = f"{title} {sentence}"
    if target_type == "date":
        return 1.0 if (FULL_DATE_RE.search(text) or MONTH_YEAR_RE.search(text) or YEAR_RE.search(text)) else 0.0
    if target_type == "number":
        return 1.0 if NUMBER_RE.search(text) else 0.0
    if target_type == "occupation":
        return 1.0 if OCCUPATION_HEAD_RE.search(text) else 0.0
    if target_type == "nationality":
        return 1.0 if NATIONALITY_OR_COUNTRY_RE.search(text) else 0.0
    if target_type in {"person", "location", "organization", "work"}:
        return 0.75 if TITLE_CASE_RE.search(text) else 0.0
    return 0.35 if (TITLE_CASE_RE.search(text) or YEAR_RE.search(text) or NUMBER_RE.search(text)) else 0.0


def _score_sentences(question: str, qa_docs: List[str], target_type: str, max_docs: int) -> List[Dict[str, Any]]:
    question_tokens = _token_set(question)
    scored = []
    for doc_index, doc in enumerate((qa_docs or [])[:max(1, int(max_docs))]):
        title = _doc_title(doc)
        title_tokens = _token_set(title)
        for sentence_index, sentence in enumerate(_split_sentences(_doc_body(doc))):
            sent_tokens = _token_set(sentence)
            if not sent_tokens:
                continue
            question_overlap = _overlap(question_tokens, sent_tokens)
            title_overlap = _overlap(question_tokens, title_tokens)
            target_signal = _target_signal(sentence, title, target_type)
            rank_prior = 1.0 / (doc_index + 1)
            sentence_prior = 1.0 / (sentence_index + 1)
            score = (
                0.48 * question_overlap
                + 0.16 * title_overlap
                + 0.22 * target_signal
                + 0.09 * rank_prior
                + 0.05 * sentence_prior
            )
            scored.append({
                "doc_index": doc_index,
                "sentence_index": sentence_index,
                "title": title,
                "sentence": sentence,
                "score": score,
                "question_overlap": question_overlap,
                "target_signal": target_signal,
            })
    scored.sort(key=lambda item: (-item["score"], item["doc_index"], item["sentence_index"]))
    return scored


def _extract_numeric_or_date_candidates(question: str, sentence: str, target_type: str) -> List[str]:
    if target_type == "date":
        if QUESTION_YEAR_RE.search(question):
            return YEAR_RE.findall(sentence)
        candidates = []
        candidates.extend(match.group(0).strip(" ,.;") for match in FULL_DATE_RE.finditer(sentence))
        candidates.extend(match.group(0).strip(" ,.;") for match in MONTH_YEAR_RE.finditer(sentence))
        if not candidates:
            candidates.extend(YEAR_RE.findall(sentence))
        return list(dict.fromkeys(candidates))
    if target_type == "number":
        return list(dict.fromkeys(match.group(0).strip(" ,.;") for match in NUMBER_RE.finditer(sentence)))
    return []


def _extract_entity_candidates(sentence: str, title: str, target_type: str) -> List[str]:
    if target_type not in {"person", "location", "organization", "work", "occupation", "nationality"}:
        return []
    candidates = []
    if target_type == "nationality":
        return list(dict.fromkeys(match.group(0).strip(" ,.;:") for match in NATIONALITY_OR_COUNTRY_RE.finditer(sentence or "")))
    if target_type == "occupation":
        for match in OCCUPATION_HEAD_RE.finditer(sentence or ""):
            candidate = match.group(0).strip(" ,.;:")
            if candidate:
                candidates.append(candidate)
        return list(dict.fromkeys(candidate for candidate in candidates if len(candidate) > 1))

    candidates.extend(match.group(0).strip(" ,.;") for match in TITLE_CASE_RE.finditer(sentence))
    if target_type in {"work", "organization"}:
        candidates.extend(match.group(1).strip(" ,.;") for match in re.finditer(r"[\"']([^\"']{2,80})[\"']", sentence))
    if title and target_type != "person":
        candidates.append(title)
    return list(dict.fromkeys(candidate for candidate in candidates if len(candidate) > 1))


def _candidate_token_count(candidate: str) -> int:
    return len(_tokens(candidate))


def _valid_candidate(candidate: str, answer: str, target_type: str) -> bool:
    candidate = str(candidate or "").strip(" \t\r\n.;:")
    key = _candidate_key(candidate)
    if not candidate:
        return False
    if not key:
        return False
    if key == _candidate_key(answer):
        return False
    if target_type in {"date", "number"}:
        return bool(re.search(r"\d", candidate))
    if target_type == "occupation":
        return bool(OCCUPATION_HEAD_RE.search(candidate))
    if target_type == "nationality":
        return bool(NATIONALITY_OR_COUNTRY_RE.search(candidate))
    if _candidate_token_count(candidate) == 0:
        return False
    if _candidate_token_count(candidate) > 8:
        return False
    if key in MONTH_NAME_KEYS:
        return False
    if key in BAD_GENERIC_CANDIDATE_KEYS:
        return False
    return True


def _rank_candidates(
    question: str,
    answer: str,
    qa_docs: List[str],
    target_type: str,
    max_docs: int,
) -> List[Dict[str, Any]]:
    scored_sentences = _score_sentences(question, qa_docs, target_type, max_docs)
    by_key: Dict[str, Dict[str, Any]] = {}
    counts = Counter()

    for item in scored_sentences[:60]:
        if target_type in {"date", "number"}:
            raw_candidates = _extract_numeric_or_date_candidates(question, item["sentence"], target_type)
        else:
            raw_candidates = _extract_entity_candidates(item["sentence"], item["title"], target_type)
        for raw in raw_candidates:
            candidate = str(raw).strip(" \t\r\n.;:")
            if not _valid_candidate(candidate, answer, target_type):
                continue
            key = _candidate_key(candidate)
            counts[key] += 1
            current = by_key.get(key)
            candidate_score = float(item["score"])
            relation_score = _relation_proximity_score(question, item["sentence"], candidate)
            if current is None or candidate_score > current.get("best_sentence_score", 0.0):
                by_key[key] = {
                    "candidate": candidate,
                    "key": key,
                    "best_sentence_score": candidate_score,
                    "best_candidate_score": candidate_score,
                    "relation_score": relation_score,
                    "title": item["title"],
                    "sentence_preview": item["sentence"][:240],
                    "source_namespace": "base_passage",
                    "lineage_source_title": item["title"],
                    "lineage_sentence_preview": item["sentence"][:240],
                    "question_overlap": item["question_overlap"],
                    "target_signal": item["target_signal"],
                }

    ranked = []
    for key, item in by_key.items():
        frequency_bonus = min(0.12, 0.03 * max(0, counts[key] - 1))
        token_penalty = 0.015 * max(0, _candidate_token_count(item["candidate"]) - 3)
        relation_bonus = float(item.get("relation_score", 0.0) or 0.0)
        confidence = item["best_sentence_score"] + frequency_bonus - token_penalty
        if target_type in {"date", "number"}:
            confidence += 0.05
        item = dict(item)
        item.update({
            "frequency": counts[key],
            "relation_bonus": round(relation_bonus, 4),
            "confidence": round(max(0.0, min(0.99, confidence)), 4),
        })
        ranked.append(item)

    ranked.sort(key=lambda item: (-item["confidence"], -item["best_sentence_score"], item["candidate"].lower()))
    return ranked


def _new_meta(enabled: bool, before: str) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "changed": False,
        "action": "none" if enabled else "disabled",
        "before": before,
        "after": before,
        "confidence": 0.0,
    }


def _is_synthetic_verifier_doc(doc: str) -> bool:
    title = _doc_title(doc).lower()
    return any(
        marker in title
        for marker in (
            "candidate answer",
            "candidate table",
            "typed evidence",
            "answer-slot",
            "fact-join",
            "provenance",
            "structured evidence",
        )
    )


def _plan_source_docs(qa_docs: Sequence[str], max_docs: int) -> List[str]:
    docs: List[str] = []
    for doc in qa_docs or []:
        if not isinstance(doc, str) or not doc.strip():
            continue
        if _is_synthetic_verifier_doc(doc):
            continue
        docs.append(doc)
        if len(docs) >= max(1, int(max_docs)):
            break
    return docs


def _infer_plan_answer_slot(question: str, target_type: str) -> str:
    q = str(question or "").lower()
    if re.search(r"\bnationality|citizenship|citizen|national\b", q):
        return "nationality"
    if re.search(r"\bcountry\s+of\s+origin|what\s+country|which\s+country\b", q):
        return "location"
    if re.search(r"\bwhen|what year|which year|date|date\s+of|born|died|released|published|opened|founded|established\b", q):
        return "date" if target_type != "year" else "year"
    if re.search(r"\bwhere|place\s+of|place|city|country|state|province|located|location\b", q):
        return "location"
    if QUESTION_OPTION_SLOT_RE.search(question or ""):
        return "question_option"
    if re.search(r"\bhow many|number|population|age|height|length|count|total\b", q):
        return "number"
    if re.search(r"\boccupation|profession|job|career|worked as|known as\b", q):
        return "occupation"
    if re.search(r"\bwho|whose|person|mother|father|parent|son|daughter|wife|husband|spouse|director|author|writer|composer|actor|actress\b", q):
        return "person"
    return target_type or "entity"


def _question_option_keys(question: str) -> set:
    question = str(question or "")
    keys = set()
    quoted = re.findall(r"[\"“”'‘’]([^\"“”'‘’]{2,80})[\"“”'‘’]", question)
    for value in quoted:
        key = _candidate_key(value)
        if key:
            keys.add(key)
    if "," in question and re.search(r"\b(?:or|and)\b", question, re.IGNORECASE):
        tail = question.rsplit(",", 1)[-1]
        parts = re.split(r"\b(?:or|and)\b", tail, flags=re.IGNORECASE)
        for value in parts:
            key = _candidate_key(value.strip(" ?.,;:"))
            if key and 1 <= len(key.split()) <= 8:
                keys.add(key)
    return keys


def _project_plan_candidate(question: str, candidate: str, slot: str) -> str:
    candidate = str(candidate or "").strip(" \t\r\n.;:")
    if not candidate:
        return ""
    if slot == "year" or QUESTION_YEAR_RE.search(question or ""):
        years = YEAR_RE.findall(candidate)
        if years:
            return years[-1]
    return candidate


def _candidate_fits_plan_slot(candidate: str, slot: str, target_type: str) -> Tuple[bool, str]:
    candidate = str(candidate or "").strip()
    key = _candidate_key(candidate)
    if not key:
        return False, "empty"
    slot = slot or target_type or "entity"
    if slot in {"date", "year"} or target_type in {"date", "year"}:
        if not re.search(r"\d", candidate):
            return False, "date_without_digit"
        return True, "date"
    if slot == "number" or target_type == "number":
        if not NUMBER_RE.search(candidate):
            return False, "number_without_numeric_surface"
        return True, "number"
    if slot == "occupation" or target_type == "occupation":
        if not OCCUPATION_HEAD_RE.search(candidate):
            return False, "occupation_missing_head"
        return True, "occupation"
    if slot == "nationality" or target_type == "nationality":
        if not NATIONALITY_OR_COUNTRY_RE.search(candidate):
            return False, "nationality_missing_surface"
        return True, "nationality"
    if slot == "question_option":


        if _candidate_token_count(candidate) > 9:
            return False, "candidate_too_long"
        return True, "question_option"
    if _candidate_token_count(candidate) > 9:
        return False, "candidate_too_long"
    if key in BAD_GENERIC_CANDIDATE_KEYS:
        return False, "generic_candidate"
    if slot in {"person", "location", "organization", "work", "entity"}:
        if re.search(r"[A-ZÀ-ÖØ-Þ]", candidate) or _candidate_token_count(candidate) <= 4:
            return True, "entity"
    return True, "generic"


def _date_specificity(text: str) -> int:
    text = str(text or "")
    if FULL_DATE_RE.search(text):
        return 3
    if MONTH_YEAR_RE.search(text):
        return 2
    if YEAR_RE.search(text):
        return 1
    return 0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _surface_safety_reason(question: str, before: str, after: str, slot: str, target_type: str) -> str:
    before_key = _candidate_key(before)
    after_key = _candidate_key(after)
    if not after_key:
        return "empty_after"
    if before_key == after_key:
        return "same_answer"
    if slot in {"date", "year"} or target_type in {"date", "year"}:
        before_spec = _date_specificity(before)
        after_spec = _date_specificity(after)
        if before_spec > after_spec and before_spec >= 2 and not QUESTION_YEAR_RE.search(question or ""):
            return "temporal_granularity_loss"
    if slot == "location" or target_type == "location":
        if before_key and _norm_contains(before_key, after_key) and len(after_key.split()) < len(before_key.split()):
            return "location_granularity_loss"
    return ""


def _candidate_plan_support(
    *,
    question: str,
    candidate: str,
    qa_docs: Sequence[str],
    target_type: str,
    max_docs: int,
) -> Dict[str, Any]:
    candidate_norm = _candidate_key(candidate)
    supported, support_score = _is_supported(question, candidate, list(qa_docs or []), target_type, max_docs)
    best: Dict[str, Any] = {
        "supported": bool(supported),
        "support_score": float(support_score or 0.0),
        "relation_score": 0.0,
        "question_overlap": 0.0,
        "target_signal": 0.0,
        "title": "",
        "sentence_preview": "",
    }
    if not candidate_norm:
        return best
    for item in _score_sentences(question, list(qa_docs or []), target_type, max_docs):
        sent_norm = _candidate_key(f"{item.get('title', '')} {item.get('sentence', '')}")
        if not _norm_contains(sent_norm, candidate_norm):
            continue
        relation_score = _relation_proximity_score(question, item.get("sentence", ""), candidate)
        local_support = max(float(item.get("score", 0.0) or 0.0), float(support_score or 0.0))
        plan_score = (
            0.55 * local_support
            + 0.22 * relation_score
            + 0.13 * float(item.get("question_overlap", 0.0) or 0.0)
            + 0.10 * float(item.get("target_signal", 0.0) or 0.0)
        )
        old_plan = (
            0.55 * float(best.get("support_score", 0.0) or 0.0)
            + 0.22 * float(best.get("relation_score", 0.0) or 0.0)
            + 0.13 * float(best.get("question_overlap", 0.0) or 0.0)
            + 0.10 * float(best.get("target_signal", 0.0) or 0.0)
        )
        if plan_score > old_plan:
            best.update({
                "supported": True,
                "support_score": local_support,
                "relation_score": relation_score,
                "question_overlap": float(item.get("question_overlap", 0.0) or 0.0),
                "target_signal": float(item.get("target_signal", 0.0) or 0.0),
                "title": item.get("title", ""),
                "sentence_preview": str(item.get("sentence", ""))[:240],
            })
    return best


def _iter_meta_candidates(value: Any, source: str, depth: int = 0) -> List[Dict[str, Any]]:
    if depth > 4:
        return []
    rows: List[Dict[str, Any]] = []
    if isinstance(value, dict):
        for key in (
            "candidate",
            "selected_candidate",
            "shadow_after",
            "after",
            "answer",
            "answer_surface",
            "surface",
        ):
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip():
                rows.append({
                    "candidate": raw.strip(),
                    "source": source,
                    "evidence": value.get("evidence") or value.get("sentence_preview") or value.get("lineage_sentence_preview") or "",
                    "module_score": _safe_float(
                        value.get("binding_score", value.get("score", value.get("support_score", 0.0))),
                        0.0,
                    ),
                })
        for key in (
            "top_candidates",
            "candidate_rows",
            "candidate_table_rows",
            "reader_candidate_table_rows",
            "answer_slot_store_rows",
            "rows",
            "typed_rows",
        ):
            child = value.get(key)
            rows.extend(_iter_meta_candidates(child, source, depth + 1))
    elif isinstance(value, list):
        for item in value[:80]:
            rows.extend(_iter_meta_candidates(item, source, depth + 1))
    return rows


def _module_candidate_rows(named_modules: Sequence[Tuple[str, Optional[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source, meta in named_modules:
        if not meta:
            continue
        rows.extend(_iter_meta_candidates(meta, source))
    return rows


def verify_answer_with_plan_consistency(
    *,
    question: str,
    answer: str,
    qa_docs: Optional[Sequence[str]] = None,
    answer_style: Optional[str] = None,
    evidence_grounded_answer_extractor: Optional[Dict[str, Any]] = None,
    query_answer_contract: Optional[Dict[str, Any]] = None,
    provenance_answer_slot_store: Optional[Dict[str, Any]] = None,
    answer_slot_binding_optimizer: Optional[Dict[str, Any]] = None,
    answer_surface_lattice_optimizer: Optional[Dict[str, Any]] = None,
    provenance_typed_candidate_store: Optional[Dict[str, Any]] = None,
    typed_evidence_table: Optional[Dict[str, Any]] = None,
    relation_fact_join: Optional[Dict[str, Any]] = None,
    evidence_admission: Optional[Dict[str, Any]] = None,
    enabled: bool = False,
    apply_rewrite: bool = True,
    min_plan_score: float = 0.42,
    min_margin: float = 0.055,
    min_relation_gain: float = 0.035,
    min_support_score: float = 0.20,
    max_docs: int = 10,
    max_candidates: int = 18,
    protect_supported_answers: bool = True,
    allow_non_abstain_unsupported_rewrite: bool = False,
    allow_slot_rewrite: bool = False,
    allow_relation_rewrite: bool = False,
    allow_format_rewrite: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Verify the final answer against a lightweight query plan and provenance.

    Unlike the early evidence verifier, this runs after answer-slot and surface
    modules have produced candidates. It only rewrites when a candidate is
    grounded in reader-visible evidence and better satisfies the question's
    requested relation/slot than the incumbent answer.
    """
    before = str(answer or "").strip()
    meta = _new_meta(enabled=enabled, before=before)
    meta.update({
        "apply_rewrite": bool(apply_rewrite),
        "protect_supported_answers": bool(protect_supported_answers),
        "min_plan_score": float(min_plan_score),
        "min_margin": float(min_margin),
        "min_relation_gain": float(min_relation_gain),
        "allow_non_abstain_unsupported_rewrite": bool(allow_non_abstain_unsupported_rewrite),
        "allow_slot_rewrite": bool(allow_slot_rewrite),
        "allow_relation_rewrite": bool(allow_relation_rewrite),
        "allow_format_rewrite": bool(allow_format_rewrite),
    })
    if not enabled or not before:
        return before, meta
    if not answer_style in {None, "", "short_factoid", "multi_slot"}:
        meta["reason"] = "non_factoid_style_kept"
        return before, meta
    if BOOLEAN_SURFACE_RE.match(before):
        meta["reason"] = "boolean_protected"
        return before, meta

    source_docs = _plan_source_docs(list(qa_docs or []), max_docs=max_docs)
    if not source_docs:
        meta["reason"] = "no_source_docs"
        return before, meta

    target_type = infer_answer_target_type(question or "")
    candidate_target_type = "date" if target_type == "year" else target_type
    answer_slot = _infer_plan_answer_slot(question or "", target_type)
    if answer_slot == "year":
        candidate_target_type = "date"
    elif answer_slot in {"date", "number", "occupation", "nationality", "location"}:
        candidate_target_type = answer_slot
    elif answer_slot == "question_option":
        candidate_target_type = target_type if target_type not in {"year", "date", "number"} else "work"
    option_keys = _question_option_keys(question or "") if answer_slot == "question_option" else set()
    meta.update({
        "target_type": target_type,
        "candidate_target_type": candidate_target_type,
        "answer_slot": answer_slot,
        "question_option_count": len(option_keys),
        "source_doc_count": len(source_docs),
    })

    incumbent_slot_ok, incumbent_slot_reason = _candidate_fits_plan_slot(before, answer_slot, candidate_target_type)
    incumbent_support = _candidate_plan_support(
        question=question or "",
        candidate=before,
        qa_docs=source_docs,
        target_type=candidate_target_type,
        max_docs=max_docs,
    )
    incumbent_plan_score = (
        0.55 * float(incumbent_support.get("support_score", 0.0) or 0.0)
        + 0.22 * float(incumbent_support.get("relation_score", 0.0) or 0.0)
        + 0.13 * float(incumbent_support.get("question_overlap", 0.0) or 0.0)
        + 0.10 * float(incumbent_support.get("target_signal", 0.0) or 0.0)
    )

    rows: List[Dict[str, Any]] = []
    for row in _rank_candidates(question or "", before, source_docs, candidate_target_type, max_docs):
        rows.append({"candidate": row.get("candidate", ""), "source": "reader_evidence_ranker", "module_score": row.get("confidence", 0.0)})
    rows.extend(_module_candidate_rows([
        ("evidence_grounded_answer_extractor", evidence_grounded_answer_extractor),
        ("query_answer_contract", query_answer_contract),
        ("provenance_answer_slot_store", provenance_answer_slot_store),
        ("answer_slot_binding_optimizer", answer_slot_binding_optimizer),
        ("answer_surface_lattice_optimizer", answer_surface_lattice_optimizer),
        ("provenance_typed_candidate_store", provenance_typed_candidate_store),
        ("typed_evidence_table", typed_evidence_table),
        ("relation_fact_join", relation_fact_join),
        ("evidence_admission", evidence_admission),
    ]))

    by_key: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        raw = _project_plan_candidate(question or "", str(row.get("candidate") or ""), answer_slot)
        if not raw or _candidate_key(raw) == _candidate_key(before):
            continue
        if option_keys and _candidate_key(raw) not in option_keys:
            continue
        slot_ok, slot_reason = _candidate_fits_plan_slot(raw, answer_slot, candidate_target_type)
        if not slot_ok:
            continue
        surface_reason = _surface_safety_reason(question or "", before, raw, answer_slot, candidate_target_type)
        if surface_reason and surface_reason not in {"same_answer"}:
            continue
        support = _candidate_plan_support(
            question=question or "",
            candidate=raw,
            qa_docs=source_docs,
            target_type=candidate_target_type,
            max_docs=max_docs,
        )
        if not support.get("supported"):
            continue
        key = _candidate_key(raw)
        current = by_key.setdefault(key, {
            "candidate": raw,
            "sources": set(),
            "module_score": 0.0,
            "slot_reason": slot_reason,
        })
        current["sources"].add(str(row.get("source") or "unknown"))
        current["module_score"] = max(float(current.get("module_score", 0.0) or 0.0), float(row.get("module_score", 0.0) or 0.0))
        for k, v in support.items():
            if isinstance(v, (int, float)):
                current[k] = max(float(current.get(k, 0.0) or 0.0), float(v))
            elif k not in current or not current.get(k):
                current[k] = v

    candidates: List[Dict[str, Any]] = []
    for row in by_key.values():
        module_votes = len(row.get("sources") or [])
        module_bonus = min(0.08, 0.025 * max(0, module_votes - 1))
        plan_score = (
            0.55 * float(row.get("support_score", 0.0) or 0.0)
            + 0.22 * float(row.get("relation_score", 0.0) or 0.0)
            + 0.13 * float(row.get("question_overlap", 0.0) or 0.0)
            + 0.10 * float(row.get("target_signal", 0.0) or 0.0)
            + module_bonus
        )
        item = dict(row)
        item["sources"] = sorted(row.get("sources") or [])
        item["module_votes"] = module_votes
        item["plan_score"] = round(plan_score, 4)
        item["relation_gain"] = round(float(row.get("relation_score", 0.0) or 0.0) - float(incumbent_support.get("relation_score", 0.0) or 0.0), 4)
        item["support_gain"] = round(float(row.get("support_score", 0.0) or 0.0) - float(incumbent_support.get("support_score", 0.0) or 0.0), 4)
        candidates.append(item)
    candidates.sort(key=lambda r: (-float(r.get("plan_score", 0.0) or 0.0), -int(r.get("module_votes", 0) or 0), str(r.get("candidate", "")).lower()))
    candidates = candidates[:max(1, int(max_candidates))]
    top = candidates[0] if candidates else None
    second_score = float(candidates[1].get("plan_score", 0.0) or 0.0) if len(candidates) > 1 else 0.0
    margin = float(top.get("plan_score", 0.0) or 0.0) - second_score if top else 0.0
    support_gain = float(top.get("support_gain", 0.0) or 0.0) if top else 0.0
    relation_gain = float(top.get("relation_gain", 0.0) or 0.0) if top else 0.0
    after = str(top.get("candidate") or "").strip() if top else ""

    answer_supported = bool(incumbent_support.get("supported"))
    answer_relation_score = float(incumbent_support.get("relation_score", 0.0) or 0.0)
    answer_is_abstain = bool(ABSTAIN_RE.search(before))
    unsupported_repair = bool(
        top
        and not answer_supported
        and (
            answer_is_abstain
            or bool(allow_non_abstain_unsupported_rewrite)
        )
    )
    slot_repair = bool(top and bool(allow_slot_rewrite) and not incumbent_slot_ok)
    relation_bound_repair = bool(
        top
        and bool(allow_relation_rewrite)
        and answer_supported
        and relation_gain >= float(min_relation_gain)
        and float(top.get("support_score", 0.0) or 0.0) >= float(min_support_score)
    )
    format_repair = bool(
        top
        and bool(allow_format_rewrite)
        and answer_supported
        and _candidate_key(before)
        and (
            _norm_contains(_candidate_key(after), _candidate_key(before))
            or _norm_contains(_candidate_key(before), _candidate_key(after))
        )
        and support_gain >= -0.02
    )

    protected_supported = bool(
        protect_supported_answers
        and answer_supported
        and incumbent_slot_ok
        and answer_relation_score >= 0.12
        and not relation_bound_repair
        and not format_repair
    )

    meta.update({
        "answer_supported": answer_supported,
        "answer_is_abstain": answer_is_abstain,
        "answer_support_score": round(float(incumbent_support.get("support_score", 0.0) or 0.0), 4),
        "answer_relation_score": round(answer_relation_score, 4),
        "incumbent_plan_score": round(incumbent_plan_score, 4),
        "incumbent_slot_ok": bool(incumbent_slot_ok),
        "incumbent_slot_reason": incumbent_slot_reason,
        "raw_candidate_count": len(rows),
        "candidate_count": len(candidates),
        "selected_candidate": after,
        "selected_plan_score": round(float(top.get("plan_score", 0.0) or 0.0), 4) if top else 0.0,
        "selected_support_score": round(float(top.get("support_score", 0.0) or 0.0), 4) if top else 0.0,
        "selected_relation_score": round(float(top.get("relation_score", 0.0) or 0.0), 4) if top else 0.0,
        "margin": round(margin, 4),
        "support_gain": round(support_gain, 4),
        "relation_gain": round(relation_gain, 4),
        "unsupported_repair": bool(unsupported_repair),
        "slot_repair": bool(slot_repair),
        "relation_bound_repair": bool(relation_bound_repair),
        "format_repair": bool(format_repair),
        "protected_supported_answer": bool(protected_supported),
        "top_candidates": candidates[:6],
    })

    if not top:
        meta["reason"] = "no_candidates"
        return before, meta
    if protected_supported:
        meta["reason"] = "supported_relation_bound_answer_protected"
        return before, meta
    if float(top.get("plan_score", 0.0) or 0.0) < float(min_plan_score):
        meta["reason"] = "low_plan_score"
        return before, meta
    if top and not answer_supported and not answer_is_abstain and not bool(allow_non_abstain_unsupported_rewrite):
        meta["reason"] = "unsupported_non_abstain_answer_protected"
        return before, meta
    if margin < float(min_margin) and not (unsupported_repair or slot_repair or format_repair):
        meta["reason"] = "low_margin"
        return before, meta
    if not (unsupported_repair or slot_repair or relation_bound_repair or format_repair):
        meta["reason"] = "no_verified_rewrite_case"
        return before, meta

    reason = (
        "unsupported_plan_repair"
        if unsupported_repair
        else "slot_plan_repair"
        if slot_repair
        else "relation_bound_plan_repair"
        if relation_bound_repair
        else "format_plan_repair"
    )
    meta.update({
        "shadow_changed": True,
        "shadow_after": after,
        "shadow_reason": reason,
        "reason": reason,
    })
    if not apply_rewrite:
        meta["action"] = "shadow_rewrite"
        return before, meta

    meta.update({
        "changed": after != before,
        "action": "rewrite" if after != before else "keep",
        "after": after,
    })
    return after, meta


def verify_answer_with_evidence(
    question: str,
    answer: str,
    qa_docs: List[str] = None,
    answer_style: str = "",
    enabled: bool = False,
    min_confidence: float = 0.72,
    min_margin: float = 0.12,
    generic_min_confidence: float = 0.84,
    max_docs: int = 8,
) -> Tuple[str, Dict[str, Any]]:
    """Conservatively replace unsupported answers with evidence-backed spans."""
    before = (answer or "").strip()
    meta = _new_meta(enabled=enabled, before=before)
    if not enabled or not before:
        return before, meta

    qa_docs = qa_docs or []
    target_type = infer_answer_target_type(question)
    answer_supported, answer_support_score = _is_supported(question, before, qa_docs, target_type, max_docs)
    candidates = _rank_candidates(question, before, qa_docs, target_type, max_docs)
    top = candidates[0] if candidates else None
    second_confidence = candidates[1]["confidence"] if len(candidates) > 1 else 0.0
    margin = (top["confidence"] - second_confidence) if top else 0.0

    meta.update({
        "target_type": target_type,
        "answer_style": answer_style,
        "answer_supported": answer_supported,
        "answer_support_score": round(answer_support_score, 4),
        "candidate_count": len(candidates),
        "candidates": candidates[:5],
        "margin": round(margin, 4),
    })

    if not top:
        meta["action"] = "no_candidates"
        return before, meta

    top_confidence = float(top["confidence"])
    after = top["candidate"]
    is_numeric_temporal = target_type in {"date", "number"}
    is_abstain = bool(ABSTAIN_RE.search(before))
    short_answer_style = answer_style in {"", None, "short_factoid", "multi_slot"}

    if is_numeric_temporal:
        enough_margin = margin >= min_margin or second_confidence == 0.0
        answer_lacks_required_type = not re.search(r"\d", before)
        better_supported = top_confidence >= answer_support_score + 0.16
        if top_confidence >= min_confidence and enough_margin and (
            is_abstain or answer_lacks_required_type or not answer_supported or better_supported
        ):
            meta.update({
                "changed": after != before,
                "action": "numeric_temporal_evidence_rewrite",
                "after": after,
                "confidence": round(top_confidence, 4),
            })
            return after, meta
        meta["action"] = "numeric_temporal_kept"
        return before, meta

    if not short_answer_style:
        meta["action"] = "non_factoid_style_kept"
        return before, meta

    if answer_supported and not is_abstain:
        meta["action"] = "answer_supported_kept"
        return before, meta

    if top_confidence >= generic_min_confidence and (margin >= min_margin or second_confidence == 0.0):
        meta.update({
            "changed": after != before,
            "action": "unsupported_answer_evidence_rewrite",
            "after": after,
            "confidence": round(top_confidence, 4),
        })
        return after, meta

    meta["action"] = "low_confidence_kept"
    return before, meta
