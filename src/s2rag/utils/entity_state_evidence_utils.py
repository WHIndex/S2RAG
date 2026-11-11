import re
from typing import Any, Dict, List, Tuple

from .sentence_evidence_utils import (
    _doc_body,
    _doc_title,
    _overlap_ratio,
    _split_sentences,
    _target_signal,
    _token_set,
    infer_answer_target_type,
)


PLACE_SUFFIX_RE = re.compile(
    r"\b[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,3},\s+"
    r"(?:[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,3}|[A-Z]{2})\b"
)
PAREN_PLACE_RE = re.compile(r"\(([^()]{3,80})\)")
TITLE_PHRASE_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9.'-]+(?:\s+(?:of|the|and|de|la|le|du|[A-Z][A-Za-z0-9.'-]+)){1,6}\b"
)
LOCATION_HINT_RE = re.compile(
    r"\b(?:city|town|village|county|state|province|country|region|district|"
    r"capital|birthplace|born|located|based|in|from)\b",
    re.IGNORECASE,
)
QUESTION_LOCATION_CONSTRAINT_RE = re.compile(
    r"\b(?:birthplace|born in|born near|located in|based in|headquartered in|"
    r"set in|which city|which country|which state|which county|which province|"
    r"city|town|village|county|state|province|country|region|district|capital)\b",
    re.IGNORECASE,
)
PROPER_LOCATION_PREP_RE = re.compile(
    r"\b(?:in|from|near|around)\s+(?:the\s+)?[A-Z][A-Za-z.'-]+"
    r"(?:\s+[A-Z][A-Za-z.'-]+){0,3}\b"
)
COMPARISON_OR_MULTI_BRANCH_RE = re.compile(
    r"\b(?:or|either|both|between|besides|what else|earlier|later|first|last|"
    r"most|least|longest|shortest|older|younger|highest|lowest)\b|"
    r"\bwhich\s+[^?]{0,80}\b(?:earlier|later|first|last|most|least|longest|"
    r"shortest|older|younger|highest|lowest)\b",
    re.IGNORECASE,
)
MAIN_DATE_REQUEST_RE = re.compile(
    r"^\s*(?:when\b|on\s+what\s+date\b|what\s+date\b|"
    r"(?:what|which|in\s+what|in\s+which)\s+year\b)",
    re.IGNORECASE,
)
YEAR_OR_NUMBER_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2}|\d+(?:,\d{3})*)\b")

SYNTHETIC_DOC_TITLES = {
    "anchored typed candidate table",
    "candidate answer table",
    "candidate answer contrastive selector",
    "directional fact join candidate table",
    "typed evidence grounding",
    "entity-state evidence control",
    "multi-granular evidence memory",
    "structured evidence chain",
    "slot-separated fact join view",
    "evidence snippets from",
}

GENERIC_ANCHOR_TOKENS = {
    "answer",
    "born",
    "built",
    "city",
    "country",
    "date",
    "film",
    "location",
    "monument",
    "performer",
    "question",
    "song",
    "state",
    "title",
    "year",
}

LOCATION_SUFFIXES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming", "united states",
    "united kingdom", "england", "scotland", "wales", "ireland",
    "canada", "france", "germany", "italy", "spain", "portugal",
    "australia", "new zealand", "india", "china", "japan", "mexico",
    "brazil", "argentina", "russia", "poland", "netherlands",
    "belgium", "sweden", "norway", "denmark", "finland", "greece",
}


def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _anchor_key(phrase: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", phrase.lower()).strip()


def _is_synthetic_doc(doc: str) -> bool:
    title = _doc_title(doc).lower().strip()
    return any(title.startswith(prefix) for prefix in SYNTHETIC_DOC_TITLES)


def _has_strict_temporal_location_shape(question: str, target_type: str) -> bool:
    """Only guard date answers when the question has an explicit place/state cue."""
    question = _normalize_space(question)
    if target_type != "date":
        return False
    if not MAIN_DATE_REQUEST_RE.search(question):
        return False
    if COMPARISON_OR_MULTI_BRANCH_RE.search(question):
        return False
    return bool(
        QUESTION_LOCATION_CONSTRAINT_RE.search(question)
        or PROPER_LOCATION_PREP_RE.search(question)
    )


def _compatible_location_keys(left: str, right: str) -> bool:
    left = _anchor_key(left)
    right = _anchor_key(right)
    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return False
    return left_tokens <= right_tokens or right_tokens <= left_tokens


def _select_dominant_location_anchors(anchors: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    location_anchors = [
        anchor
        for anchor in anchors
        if anchor.get("kind") == "location" or _looks_like_place_anchor(anchor.get("text", ""))
    ]
    if not location_anchors:
        return [], "no_location_state_anchor"

    top_anchor = location_anchors[0]
    top_text = str(top_anchor.get("text", ""))
    top_score = max(0.000001, float(top_anchor.get("score", 0.0)))
    selected: List[Dict[str, Any]] = []
    ambiguous: List[Dict[str, Any]] = []
    for anchor in location_anchors:
        text = str(anchor.get("text", ""))
        score = float(anchor.get("score", 0.0))
        if _compatible_location_keys(text, top_text):
            selected.append(anchor)
        elif score >= top_score * 0.65:
            ambiguous.append(anchor)

    if ambiguous:
        return [], "ambiguous_location_state"
    return selected[:2], ""


def _is_useful_anchor(phrase: str, question_tokens: set) -> bool:
    phrase = _normalize_space(phrase).strip(" ,.;:")
    if len(phrase) < 3 or len(phrase) > 80:
        return False
    tokens = _token_set(phrase)
    if len(tokens) == 0:
        return False
    if len(tokens) == 1 and next(iter(tokens)) in GENERIC_ANCHOR_TOKENS:
        return False
    if tokens and tokens <= GENERIC_ANCHOR_TOKENS:
        return False

    if tokens and tokens <= (question_tokens | GENERIC_ANCHOR_TOKENS) and len(tokens) < 2:
        return False
    return True


def _extract_place_phrases(text: str) -> List[str]:
    text = _normalize_space(text)
    phrases = []
    for match in PLACE_SUFFIX_RE.finditer(text):
        phrases.append(match.group(0).strip(" ,.;:"))
    for match in PAREN_PLACE_RE.finditer(text):
        inner = match.group(1).strip()
        if "," in inner or LOCATION_HINT_RE.search(inner):
            for place in PLACE_SUFFIX_RE.findall(inner):
                phrases.append(place.strip(" ,.;:"))
    return phrases


def _looks_like_place_anchor(phrase: str) -> bool:
    phrase = _normalize_space(phrase).strip(" ,.;:")
    if "," not in phrase:
        return bool(LOCATION_HINT_RE.search(phrase))
    suffix = _normalize_space(phrase.rsplit(",", 1)[-1]).strip(" .;:").lower()
    suffix = re.sub(r"[^a-z0-9 ]+", " ", suffix).strip()
    if suffix in LOCATION_SUFFIXES:
        return True
    if re.fullmatch(r"[a-z]{2}", suffix):
        return True
    return False


def _extract_title_phrases(text: str) -> List[str]:
    text = _normalize_space(text)
    phrases = []
    for match in TITLE_PHRASE_RE.finditer(text):
        phrase = match.group(0).strip(" ,.;:")
        if phrase.lower().startswith(("the ", "a ", "an ")):
            phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase, flags=re.IGNORECASE)
        phrases.append(phrase)
    return phrases


def _sentence_for_phrase(text: str, phrase: str) -> str:
    phrase_key = _anchor_key(phrase)
    for sentence in _split_sentences(text, max_body_chars=4000):
        if phrase_key and phrase_key in _anchor_key(sentence):
            return sentence
    return _normalize_space(text)[:300]


def _doc_contains_anchor(doc: str, anchor_keys: List[str]) -> bool:
    doc_key = _anchor_key(doc)
    return any(anchor and anchor in doc_key for anchor in anchor_keys)


def _collect_state_anchors(
    question: str,
    docs: List[str],
    planned_retrieval: Dict[str, Any],
    max_anchor_docs: int,
    max_anchors: int,
    min_source_overlap: float,
) -> List[Dict[str, Any]]:
    question_tokens = _token_set(question)
    planned_queries = planned_retrieval.get("queries") if isinstance(planned_retrieval, dict) else []
    planned_text = " ".join(str(q) for q in planned_queries or [])
    planned_tokens = _token_set(planned_text)
    source_tokens = question_tokens | planned_tokens

    candidates: Dict[str, Dict[str, Any]] = {}
    for doc_index, doc in enumerate(docs[:max(1, max_anchor_docs)]):
        title = _doc_title(doc)
        body = _doc_body(doc)
        doc_tokens = _token_set(f"{title} {body[:1600]}")
        title_tokens = _token_set(title)
        source_overlap = max(
            _overlap_ratio(question_tokens, doc_tokens | title_tokens),
            _overlap_ratio(planned_tokens, doc_tokens | title_tokens) if planned_tokens else 0.0,
        )
        if source_overlap < min_source_overlap and doc_index > 1:
            continue

        source_score = source_overlap + 0.15 / (doc_index + 1)
        phrase_sources = []
        phrase_sources.extend(_extract_place_phrases(title))
        if _overlap_ratio(question_tokens, title_tokens) > 0.0:
            phrase_sources.extend(_extract_title_phrases(title))
        for sentence in _split_sentences(body, max_body_chars=1800)[:8]:
            sentence_tokens = _token_set(sentence)
            if _overlap_ratio(source_tokens, sentence_tokens) <= 0.0 and not LOCATION_HINT_RE.search(sentence):
                continue
            phrase_sources.extend(_extract_place_phrases(sentence))
            if _overlap_ratio(question_tokens, sentence_tokens) > 0.08:
                phrase_sources.extend(_extract_title_phrases(sentence))

        for phrase in phrase_sources:
            phrase = _normalize_space(phrase).strip(" ,.;:")
            if " The " in phrase:
                before, after = phrase.split(" The ", 1)
                if _anchor_key(before) and _anchor_key(before) == _anchor_key(after):
                    phrase = before.strip()
            if not _is_useful_anchor(phrase, question_tokens):
                continue
            phrase_tokens = _token_set(phrase)
            is_place_anchor = _looks_like_place_anchor(phrase)
            if not is_place_anchor and len(source_tokens & phrase_tokens) < 2:
                continue
            key = _anchor_key(phrase)
            if not key:
                continue
            entry = candidates.setdefault(
                key,
                {
                    "text": phrase,
                    "score": 0.0,
                    "count": 0,
                    "source_titles": [],
                    "kind": "location" if "," in phrase else "entity",
                },
            )
            entry["score"] += source_score
            entry["count"] += 1
            if title and title not in entry["source_titles"]:
                entry["source_titles"].append(title)

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            -float(item.get("score", 0.0)),
            -int(item.get("count", 0)),
            str(item.get("text", "")),
        ),
    )
    return ranked[:max(1, max_anchors)]


def _score_doc_sentences(
    question: str,
    doc: str,
    target_type: str,
    anchor_keys: List[str],
    max_sentences_per_doc: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    title = _doc_title(doc)
    body = _doc_body(doc)
    question_tokens = _token_set(question)
    title_tokens = _token_set(title)
    doc_has_anchor = _doc_contains_anchor(f"{title} {body[:2500]}", anchor_keys)
    place_phrases = []
    for phrase in _extract_place_phrases(f"{title}. {body[:2500]}"):
        key = _anchor_key(phrase)
        if key and key not in anchor_keys and _looks_like_place_anchor(phrase):
            place_phrases.append(phrase)
    conflicting_places = [
        phrase
        for phrase in place_phrases
        if not any(_compatible_location_keys(phrase, anchor_key) for anchor_key in anchor_keys)
    ]

    support_items = []
    distractor_items = []
    for sent_index, sentence in enumerate(_split_sentences(body, max_body_chars=5000)):
        sent_tokens = _token_set(sentence)
        if not sent_tokens:
            continue
        sent_text = f"{title} {sentence}"
        sent_has_anchor = _doc_contains_anchor(sent_text, anchor_keys)
        target_signal = _target_signal(sentence, title, target_type)
        question_overlap = _overlap_ratio(question_tokens, sent_tokens | title_tokens)
        if not sent_has_anchor and target_signal <= 0 and question_overlap <= 0:
            continue
        score = (
            0.42 * (1.0 if sent_has_anchor else 0.0)
            + 0.24 * target_signal
            + 0.20 * question_overlap
            + 0.08 * (1.0 if doc_has_anchor else 0.0)
            + 0.06 / (sent_index + 1)
        )
        item = {
            "title": title,
            "sentence": sentence,
            "score": round(score, 6),
            "has_state_anchor": bool(sent_has_anchor or doc_has_anchor),
            "target_signal": round(float(target_signal), 6),
            "question_overlap": round(float(question_overlap), 6),
        }
        if (sent_has_anchor or doc_has_anchor) and score >= 0.24:
            support_items.append(item)
        elif target_signal > 0.0 and conflicting_places and not (sent_has_anchor or doc_has_anchor):
            item["conflicting_places"] = conflicting_places[:3]
            distractor_items.append(item)

    support_items.sort(key=lambda x: (-x["score"], x["title"], x["sentence"]))
    distractor_items.sort(key=lambda x: (-x["target_signal"], -x["question_overlap"], x["title"]))
    return support_items[:max_sentences_per_doc], distractor_items[:max_sentences_per_doc], place_phrases


def _format_entity_state_doc(
    target_type: str,
    anchors: List[Dict[str, Any]],
    support: List[Dict[str, Any]],
    distractors: List[Dict[str, Any]],
) -> str:
    lines = [
        "Entity-State Evidence Control",
        f"Target answer type: {target_type}",
        "Resolved location/state anchor for this date answer:",
    ]
    for anchor in anchors:
        lines.append(f"- {anchor['text']}")
    lines.extend(
        [
            "Soft answer-selection note:",
            "- Prefer answer-bearing date evidence that matches the resolved location/state anchor.",
            "- Be cautious with date evidence from a different location/state unless the passage explicitly links it back to the asked entity.",
        ]
    )
    if support:
        lines.append("State-consistent date evidence:")
        for item in support:
            lines.append(f"- {item['title']}: {item['sentence']}")
    if distractors:
        lines.append("Potential cross-state date evidence:")
        for item in distractors:
            places = ", ".join(item.get("conflicting_places") or [])
            suffix = f" [different state: {places}]" if places else ""
            lines.append(f"- {item['title']}: {item['sentence']}{suffix}")
    return "\n".join(lines)


def build_entity_state_evidence_control(
    question: str,
    docs: List[str],
    planned_retrieval: Dict[str, Any] = None,
    evidence_selection: Dict[str, Any] = None,
    enabled: bool = False,
    bridge_only: bool = True,
    conflict_only: bool = True,
    max_anchor_docs: int = 5,
    max_anchors: int = 6,
    max_support: int = 4,
    max_distractors: int = 3,
    max_sentences_per_doc: int = 2,
    min_source_overlap: float = 0.08,
) -> Tuple[List[str], Dict[str, Any]]:
    docs = docs or []
    planned_retrieval = planned_retrieval or {}
    evidence_selection = evidence_selection or {}
    target_type = infer_answer_target_type(question)
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "target_type": target_type,
        "original_doc_count": len(docs),
        "final_doc_count": len(docs),
        "anchor_count": 0,
        "support_sentence_count": 0,
        "distractor_sentence_count": 0,
        "bridge_only": bool(bridge_only),
        "conflict_only": bool(conflict_only),
        "selected_anchors": [],
        "support_titles": [],
        "distractor_titles": [],
    }
    if not enabled:
        return docs, info
    if not docs:
        info["reason"] = "no_docs"
        return docs, info
    if not _has_strict_temporal_location_shape(question, target_type):
        info["reason"] = "not_temporal_location_state_shape"
        return docs, info

    bridge_state_signal = bool(
        planned_retrieval.get("applied")
        or planned_retrieval.get("round_count")
        or evidence_selection.get("reason") == "expanded"
    )
    if bridge_only and not bridge_state_signal:
        info["reason"] = "no_bridge_state_signal"
        return docs, info

    source_docs = [doc for doc in docs if not _is_synthetic_doc(doc)]
    if not source_docs:
        info["reason"] = "no_source_docs"
        return docs, info

    anchors = _collect_state_anchors(
        question=question,
        docs=source_docs,
        planned_retrieval=planned_retrieval,
        max_anchor_docs=max_anchor_docs,
        max_anchors=max_anchors,
        min_source_overlap=min_source_overlap,
    )
    if not anchors:
        info["reason"] = "no_state_anchors"
        return docs, info

    anchors, anchor_reason = _select_dominant_location_anchors(anchors)
    if not anchors:
        info["reason"] = anchor_reason or "no_location_state_anchor"
        return docs, info

    anchor_keys = [_anchor_key(anchor["text"]) for anchor in anchors if _anchor_key(anchor["text"])]
    support = []
    distractors = []
    for doc in source_docs[:max(1, max_anchor_docs + 3)]:
        support_items, distractor_items, _ = _score_doc_sentences(
            question=question,
            doc=doc,
            target_type=target_type,
            anchor_keys=anchor_keys,
            max_sentences_per_doc=max(1, max_sentences_per_doc),
        )
        support.extend(support_items)
        distractors.extend(distractor_items)

    support = [item for item in support if float(item.get("target_signal", 0.0)) > 0.0]
    support.sort(key=lambda x: (-x["score"], x["title"], x["sentence"]))
    distractors.sort(key=lambda x: (-x["target_signal"], -x["question_overlap"], x["title"]))
    support = support[:max(0, max_support)]
    distractors = distractors[:max(0, max_distractors)]

    if not support:
        info.update({
            "reason": "no_state_answer_support",
            "anchor_count": len(anchors),
            "selected_anchors": [anchor["text"] for anchor in anchors],
            "distractor_sentence_count": len(distractors),
        })
        return docs, info

    if conflict_only and not distractors:
        info.update({
            "reason": "no_state_conflict",
            "anchor_count": len(anchors),
            "selected_anchors": [anchor["text"] for anchor in anchors],
            "support_sentence_count": len(support),
        })
        return docs, info

    control_doc = _format_entity_state_doc(
        target_type=target_type,
        anchors=anchors,
        support=support,
        distractors=distractors,
    )
    insert_after = min(2, len(docs))
    final_docs = docs[:insert_after] + [control_doc] + docs[insert_after:]
    support_titles = []
    distractor_titles = []
    for item in support:
        if item["title"] not in support_titles:
            support_titles.append(item["title"])
    for item in distractors:
        if item["title"] not in distractor_titles:
            distractor_titles.append(item["title"])

    info.update({
        "applied": True,
        "reason": "state_conflict_control" if distractors else "state_anchor_control",
        "final_doc_count": len(final_docs),
        "anchor_count": len(anchors),
        "support_sentence_count": len(support),
        "distractor_sentence_count": len(distractors),
        "selected_anchors": [anchor["text"] for anchor in anchors],
        "support_titles": support_titles,
        "distractor_titles": distractor_titles,
        "top_support": [
            {
                "title": item["title"],
                "score": item["score"],
                "sentence_preview": item["sentence"][:220],
            }
            for item in support[:4]
        ],
        "top_distractors": [
            {
                "title": item["title"],
                "conflicting_places": item.get("conflicting_places", [])[:3],
                "sentence_preview": item["sentence"][:220],
            }
            for item in distractors[:4]
        ],
    })
    return final_docs, info
