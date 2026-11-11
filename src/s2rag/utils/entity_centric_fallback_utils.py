import math
import re
from typing import Any, Dict, List, Sequence, Set, Tuple

from .evidence_utils import has_bridge_cues
from .sentence_evidence_utils import (
    _doc_body,
    _doc_title,
    _split_sentences,
    _target_signal,
    _token_set,
    infer_answer_target_type,
)


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
TITLE_PHRASE_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9'.-]*(?:\s+(?:of|the|and|de|del|da|la|le|van|von|"
    r"[A-Z][A-Za-z0-9'.-]*)){0,7}\b"
)
POSSESSIVE_RE = re.compile(r"\b([A-Z][^?]{1,90}?)'s\b")
BOOLEAN_OR_COMPARISON_RE = re.compile(
    r"^\s*(?:do|does|did|is|are|was|were|can|could|has|have|had)\b|"
    r"\b(?:both|same|different|compare|compared|comparison|lowest|highest|"
    r"largest|smallest|most|least|earliest|latest|first|last|older|younger|"
    r"before|after)\b",
    re.IGNORECASE,
)
LIST_LIKE_TITLE_RE = re.compile(
    r"\b(?:disambiguation|list of|index of|outline of|timeline of|"
    r"people with the last name|people with the surname)\b|\(name\)",
    re.IGNORECASE,
)
GENERIC_TOPIC_TITLE_RE = re.compile(
    r"\b(?:commission|committee|department|office|agency|ministry|government|"
    r"network|broadcast network|radio network|cheerleaders|hall of fame|"
    r"legislative assembly|general assembly|house of representatives|senate|"
    r"elections?|post office|railway station)\b|"
    r"^\s*president of\b|\bhistory of\b|\bmilitary history\b",
    re.IGNORECASE,
)
WH_RE = re.compile(r"\b(?:what|who|where|when|which|whom|whose)\b", re.IGNORECASE)

QUESTION_STOPWORDS = {
    "a", "an", "and", "answer", "are", "as", "at", "be", "by", "did",
    "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "his", "how", "in", "is", "it", "its", "name", "named", "of", "on",
    "or", "question", "she", "that", "the", "their", "there", "this",
    "to", "was", "were", "what", "when", "where", "which", "who",
    "whom", "whose", "why", "with",
}
TYPE_AND_RELATION_WORDS = {
    "album", "area", "author", "book", "born", "birth", "birthplace",
    "capital", "city", "company", "country", "county", "date", "day",
    "death", "died", "director", "district", "film", "founder", "founded",
    "language", "location", "month", "nationality", "number", "occupation",
    "person", "place", "population", "profession", "publisher", "record",
    "region", "released", "state", "team", "town", "village", "writer",
    "year",
}
RELATION_HINTS = {
    "author", "born", "birth", "birthplace", "capital", "citizenship",
    "creator", "date", "death", "died", "director", "founded", "founder",
    "language", "located", "location", "nationality", "occupation",
    "profession", "published", "publisher", "released", "spouse", "wife",
    "husband", "mother", "father", "record", "label",
}
BRIDGE_LIKE_ENTITY_FALLBACK_RE = re.compile(
    r"\b(?:whose|performer of|responsible for the performer|sister|brother|"
    r"father-in-law|mother-in-law|played\s+.+?\s+in|appeared\s+.+?\s+in|"
    r"associated with|work whose|person whose)\b",
    re.IGNORECASE,
)
SUBJECT_OF_RE = re.compile(
    r"\b(?:population|capital|chairman|director|author|writer|founder|"
    r"occupation|profession|birthplace|location|record label|label|"
    r"publisher|composer|producer|creator|owner|president|mayor)\s+"
    r"(?:of|for|by)\s+(.+?)(?:\?|$)",
    re.IGNORECASE,
)
PASSIVE_SUBJECT_RE = re.compile(
    r"\b(?:where|when)\s+(?:was|were|is|are)\s+(.+?)\s+"
    r"(?:filmed|shot|released|born|located|published|founded|set|made|"
    r"produced|created|written|directed)\b",
    re.IGNORECASE,
)
SLOT_TERM_HINTS = {
    "date": {
        "date", "day", "month", "year", "when", "born", "birth", "died",
        "death", "released", "published", "founded", "established",
    },
    "number": {
        "number", "population", "count", "total", "percentage", "percent",
        "age", "census", "rank", "score", "amount",
    },
    "location": {
        "where", "location", "located", "place", "city", "town", "village",
        "county", "state", "province", "country", "region", "district",
        "capital", "birthplace", "filmed", "filming", "shot",
    },
    "person": {
        "person", "who", "director", "author", "writer", "founder",
        "composer", "producer", "creator", "designer", "chairman", "chair",
        "president", "mother", "father", "spouse", "wife", "husband",
        "sister", "brother",
    },
    "organization": {
        "company", "organization", "organisation", "team", "university",
        "school", "label", "record", "publisher", "agency", "committee",
        "party", "club", "institute",
    },
    "work": {
        "film", "movie", "album", "book", "novel", "song", "single",
        "work", "episode", "series", "publication",
    },
    "occupation": {
        "occupation", "profession", "job", "known", "actor", "actress",
        "politician", "journalist", "writer", "author", "director",
        "producer", "composer", "artist", "painter", "player", "coach",
        "footballer", "singer", "musician", "professor", "scientist",
        "engineer", "lawyer", "businessperson", "screenwriter",
    },
    "nationality": {
        "nationality", "citizenship", "citizen", "national",
    },
    "factoid": set(),
}


def _doc_key(doc: Any) -> str:
    return re.sub(r"\s+", " ", str(doc or "")).strip().lower()


def _tokens(text: Any) -> Set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(str(text or ""))
        if len(token) > 1
    }


def _bounded(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _overlap_ratio(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def _clean_phrase(phrase: str) -> str:
    phrase = re.sub(r"\s+", " ", str(phrase or "")).strip(" ,:;?-.")
    phrase = re.sub(
        r"^(?:what|who|where|when|which|whose|is|are|was|were|did|does|do|"
        r"the|a|an|in|of|for)\s+",
        "",
        phrase,
        flags=re.IGNORECASE,
    ).strip(" ,:;?-.")
    return phrase


def _anchor_phrase_sets(question: str) -> List[Set[str]]:
    raw = str(question or "")
    phrases: List[Set[str]] = []

    for quoted_a, quoted_b in re.findall(r'"([^"]+)"|\'([^\']+)\'', raw):
        tokens = _token_set(_clean_phrase(quoted_a or quoted_b))
        if tokens:
            phrases.append(tokens)

    for match in POSSESSIVE_RE.finditer(raw):
        tokens = _token_set(_clean_phrase(match.group(1)))
        if tokens:
            phrases.append(tokens)

    for phrase in TITLE_PHRASE_RE.findall(raw):
        tokens = _token_set(_clean_phrase(phrase))
        tokens = {
            token
            for token in tokens
            if token not in QUESTION_STOPWORDS and token not in TYPE_AND_RELATION_WORDS
        }
        if tokens:
            phrases.append(tokens)

    deduped: List[Set[str]] = []
    seen = set()
    for tokens in phrases:
        key = tuple(sorted(tokens))
        if key and key not in seen:
            seen.add(key)
            deduped.append(tokens)
    return deduped[:5]


def _add_phrase_set(phrases: List[Set[str]], phrase: str) -> None:
    tokens = {
        token
        for token in _token_set(_clean_phrase(phrase))
        if token not in QUESTION_STOPWORDS
    }
    if not tokens or tokens <= TYPE_AND_RELATION_WORDS:
        return
    phrases.append(tokens)


def _subject_phrase_sets(question: str, fallback_phrase_sets: Sequence[Set[str]]) -> List[Set[str]]:
    raw = str(question or "")
    explicit_phrases: List[Set[str]] = []

    for quoted_a, quoted_b in re.findall(r'"([^"]+)"|\'([^\']+)\'', raw):
        _add_phrase_set(explicit_phrases, quoted_a or quoted_b)

    for match in POSSESSIVE_RE.finditer(raw):
        _add_phrase_set(explicit_phrases, match.group(1))

    for regex in (PASSIVE_SUBJECT_RE, SUBJECT_OF_RE):
        for match in regex.finditer(raw):
            _add_phrase_set(explicit_phrases, match.group(1))

    phrases: List[Set[str]] = []
    if explicit_phrases:
        phrases.extend(explicit_phrases)
    else:
        for phrase_tokens in fallback_phrase_sets:
            if phrase_tokens:
                phrases.append(set(phrase_tokens))

    deduped: List[Set[str]] = []
    seen = set()
    for tokens in sorted(phrases, key=lambda item: (-len(item), sorted(item))):
        key = tuple(sorted(tokens))
        if key and key not in seen:
            seen.add(key)
            deduped.append(tokens)
    return deduped[:6]


def _query_anchor_tokens(question: str, phrase_sets: Sequence[Set[str]]) -> Set[str]:
    anchors: Set[str] = set()
    for phrase_tokens in phrase_sets:
        anchors.update(phrase_tokens)
    anchors.update(
        token
        for token in _tokens(question)
        if token not in QUESTION_STOPWORDS
        and token not in TYPE_AND_RELATION_WORDS
        and len(token) > 2
    )
    return anchors


def _query_relation_terms(question: str) -> Set[str]:
    q_tokens = _tokens(question)
    relation_terms = q_tokens & RELATION_HINTS
    q = str(question or "").lower()
    if "record label" in q:
        relation_terms.update({"record", "label"})
    if "place of birth" in q:
        relation_terms.update({"birth", "born"})
    if "date of death" in q:
        relation_terms.update({"death", "died"})
    return relation_terms


def _query_slot_terms(question: str, target_type: str, relation_terms: Set[str]) -> Set[str]:
    q_tokens = _tokens(question)
    slot_terms = set(relation_terms or set())
    slot_terms.update(q_tokens & SLOT_TERM_HINTS.get(target_type, set()))
    slot_terms.update(
        token
        for token in q_tokens
        if token in RELATION_HINTS or token in TYPE_AND_RELATION_WORDS
    )
    if target_type in {"occupation", "nationality"}:
        slot_terms.update(SLOT_TERM_HINTS.get(target_type, set()))
    if target_type == "number" and ("population" in q_tokens or "census" in q_tokens):
        slot_terms.update({"population", "census"})
    if target_type == "location" and ("filmed" in q_tokens or "shot" in q_tokens):
        slot_terms.update({"filmed", "filming", "shot", "location", "place"})
    return {
        token
        for token in slot_terms
        if token not in QUESTION_STOPWORDS and len(token) > 1
    }


def _max_phrase_overlap(phrase_sets: Sequence[Set[str]], tokens: Set[str]) -> float:
    best = 0.0
    for phrase_tokens in phrase_sets:
        best = max(best, _overlap_ratio(phrase_tokens, tokens))
    return best


def _best_subject_token_count(phrase_sets: Sequence[Set[str]], tokens: Set[str]) -> int:
    best_count = 0
    best_overlap = 0.0
    for phrase_tokens in phrase_sets:
        overlap = _overlap_ratio(phrase_tokens, tokens)
        if overlap > best_overlap or (overlap == best_overlap and len(phrase_tokens) > best_count):
            best_overlap = overlap
            best_count = len(phrase_tokens)
    return best_count


def _slot_supported(
    target_type: str,
    slot_terms: Set[str],
    sentence: str,
    title: str,
    target_signal: float,
    relation_overlap: float,
) -> Tuple[bool, float]:
    sent_tokens = _token_set(sentence)
    slot_overlap = _overlap_ratio(slot_terms, sent_tokens) if slot_terms else 0.0
    if slot_overlap > 0.0 or relation_overlap > 0.0:
        return True, slot_overlap
    if target_type in {"occupation", "nationality"} and target_signal >= 0.80:
        return True, slot_overlap
    if not slot_terms and target_signal >= 0.80:
        return True, slot_overlap
    if target_type == "factoid" and target_signal >= 0.35:
        title_tokens = _token_set(title)
        if _overlap_ratio(title_tokens, sent_tokens) > 0.0:
            return True, slot_overlap
    return False, slot_overlap


def _is_generic_topic_candidate(title: str, target_type: str) -> bool:
    title_text = str(title or "").strip()
    if not title_text:
        return False
    title_lower = title_text.lower()
    if target_type == "person" and GENERIC_TOPIC_TITLE_RE.search(title_text):
        return True
    if target_type in {"location", "number"} and re.search(
        r"\b(?:commission|committee|department|office|agency|government|"
        r"assembly|senate|election|network)\b",
        title_lower,
    ):
        return True
    return False


def _planned_retrieval_text(planned_retrieval: Dict[str, Any], *, include_keys: bool = True) -> str:
    parts: List[str] = []

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 3 or value is None:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if include_keys:
                    parts.append(str(key))
                visit(item, depth + 1)
            return
        if isinstance(value, (list, tuple, set)):
            for item in list(value)[:24]:
                visit(item, depth + 1)
            return
        if isinstance(value, (str, int, float, bool)):
            parts.append(str(value))

    visit(planned_retrieval or {})
    return " ".join(parts).lower()


def _route_admission_profile(
    question: str,
    planned_retrieval: Dict[str, Any],
    enabled: bool,
) -> Dict[str, Any]:
    planned_text = _planned_retrieval_text(planned_retrieval, include_keys=True)
    planned_value_text = _planned_retrieval_text(planned_retrieval, include_keys=False)
    question_text = str(question or "")
    bridge_signal = bool(
        has_bridge_cues(question_text)
        or BRIDGE_LIKE_ENTITY_FALLBACK_RE.search(question_text)
        or re.search(r"\b(?:bridge|multi[_ -]?hop|multihop|two[_ -]?hop|compositional)\b", planned_text)
    )
    comparison_signal = bool(
        BOOLEAN_OR_COMPARISON_RE.search(question_text)
        or re.search(
            r"\b(?:comparison|compare|compared|both|same|different)\b",
            planned_value_text,
        )
    )
    deficit_signal = bool(
        re.search(
            r"\b(?:max_rounds_reached|evidence_insufficient|insufficient|"
            r"missing_slots?|missing evidence|deficit|no_answer|closed_loop)\b",
            planned_text,
        )
    )
    route_aware = bool(enabled and bridge_signal and deficit_signal and not comparison_signal)
    return {
        "route_aware_relaxation": route_aware,
        "route_bridge_signal": bridge_signal,
        "route_deficit_signal": deficit_signal,
        "route_comparison_signal": comparison_signal,
    }


def _current_evidence_has_slot_support(
    question: str,
    current_docs: Sequence[str],
    subject_phrase_sets: Sequence[Set[str]],
    slot_terms: Set[str],
    target_type: str,
    min_subject_doc_overlap: float,
    min_target_signal: float,
) -> bool:
    question_tokens = _token_set(question)
    for doc in list(current_docs or [])[:8]:
        title = _doc_title(doc)
        body = _doc_body(doc)
        title_tokens = _token_set(title)
        for sentence in _split_sentences(body, max_body_chars=5000)[:60]:
            sent_tokens = _token_set(sentence)
            if not sent_tokens:
                continue
            evidence_tokens = title_tokens | sent_tokens
            subject_overlap = _max_phrase_overlap(subject_phrase_sets, evidence_tokens)
            if subject_overlap < min_subject_doc_overlap:
                continue
            target_signal = _bounded(_target_signal(sentence, title, target_type))
            relation_overlap = _overlap_ratio(slot_terms, sent_tokens) if slot_terms else 0.0
            slot_ok, _ = _slot_supported(
                target_type=target_type,
                slot_terms=slot_terms,
                sentence=sentence,
                title=title,
                target_signal=target_signal,
                relation_overlap=relation_overlap,
            )
            question_overlap = _overlap_ratio(question_tokens, evidence_tokens)
            if slot_ok and target_signal >= min_target_signal and question_overlap >= 0.28:
                return True
    return False


def _is_single_hop_factoid(
    question: str,
    planned_retrieval: Dict[str, Any],
    block_bridge: bool,
    block_bridge_like_relation: bool,
    allow_route_aware_bridge: bool = False,
) -> Tuple[bool, str]:
    question_text = str(question or "")
    q_lower = question_text.lower()
    if not WH_RE.search(question_text):
        return False, "not_wh_question"
    if BOOLEAN_OR_COMPARISON_RE.search(question_text):
        return False, "boolean_or_comparison"
    if block_bridge and has_bridge_cues(question_text) and not allow_route_aware_bridge:
        return False, "bridge_cues"
    if (
        block_bridge_like_relation
        and BRIDGE_LIKE_ENTITY_FALLBACK_RE.search(question_text)
        and not allow_route_aware_bridge
    ):
        return False, "bridge_like_relation"
    if re.search(r"\b(?:both|same|which of|who among|what among)\b", q_lower):
        return False, "set_or_comparison"

    planned_retrieval = planned_retrieval or {}
    route_type = str(planned_retrieval.get("question_type", "") or "").lower()
    route_reason = str(planned_retrieval.get("reason", "") or "").lower()
    router = planned_retrieval.get("router") or {}
    router_type = str(router.get("question_type", "") or "").lower() if isinstance(router, dict) else ""
    type_blob = f"{route_type} {router_type} {route_reason}"
    if "comparison" in type_blob:
        return False, "routed_non_single_hop"
    if (
        any(term in type_blob for term in ("multi_hop", "multihop", "bridge"))
        and not allow_route_aware_bridge
    ):
        return False, "routed_non_single_hop"

    target_type = infer_answer_target_type(question_text)
    if target_type in {
        "date",
        "factoid",
        "location",
        "nationality",
        "number",
        "occupation",
        "organization",
        "person",
        "work",
    }:
        return True, "single_hop_factoid"
    return False, "unsupported_target_type"


def _best_sentence_profile(
    question_tokens: Set[str],
    anchor_tokens: Set[str],
    subject_phrase_sets: Sequence[Set[str]],
    relation_terms: Set[str],
    slot_terms: Set[str],
    title: str,
    body: str,
    target_type: str,
) -> Dict[str, Any]:
    best = {
        "sentence": "",
        "score": 0.0,
        "target_signal": 0.0,
        "query_overlap": 0.0,
        "anchor_overlap": 0.0,
        "subject_overlap": 0.0,
        "relation_overlap": 0.0,
        "slot_overlap": 0.0,
        "slot_supported": False,
    }
    for sentence in _split_sentences(body, max_body_chars=7000)[:80]:
        sent_tokens = _token_set(sentence)
        if not sent_tokens:
            continue
        target_signal = _bounded(_target_signal(sentence, title, target_type))
        query_overlap = _overlap_ratio(question_tokens, sent_tokens)
        anchor_overlap = _overlap_ratio(anchor_tokens, sent_tokens)
        subject_overlap = _max_phrase_overlap(subject_phrase_sets, sent_tokens)
        relation_overlap = _overlap_ratio(relation_terms, sent_tokens) if relation_terms else 0.0
        slot_supported, slot_overlap = _slot_supported(
            target_type=target_type,
            slot_terms=slot_terms,
            sentence=sentence,
            title=title,
            target_signal=target_signal,
            relation_overlap=relation_overlap,
        )
        score = (
            0.24 * query_overlap
            + 0.20 * anchor_overlap
            + 0.18 * subject_overlap
            + 0.18 * target_signal
            + 0.12 * relation_overlap
            + 0.08 * slot_overlap
        )
        if score > best["score"]:
            best = {
                "sentence": sentence,
                "score": score,
                "target_signal": target_signal,
                "query_overlap": query_overlap,
                "anchor_overlap": anchor_overlap,
                "subject_overlap": subject_overlap,
                "relation_overlap": relation_overlap,
                "slot_overlap": slot_overlap,
                "slot_supported": bool(slot_supported),
            }
    return best


def _rank_entity_fallback_candidates(
    question: str,
    current_docs: List[str],
    ranked_docs: List[str],
    candidate_top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    target_type = infer_answer_target_type(question)
    question_tokens = _token_set(question)
    phrase_sets = _anchor_phrase_sets(question)
    subject_phrase_sets = _subject_phrase_sets(question, phrase_sets)
    anchor_tokens = _query_anchor_tokens(question, phrase_sets)
    relation_terms = _query_relation_terms(question)
    slot_terms = _query_slot_terms(question, target_type, relation_terms)
    current_keys = {_doc_key(doc) for doc in current_docs}
    current_titles = {_doc_title(doc).strip().lower() for doc in current_docs if _doc_title(doc)}

    profile = {
        "target_type": target_type,
        "anchor_tokens": sorted(anchor_tokens)[:20],
        "relation_terms": sorted(relation_terms),
        "slot_terms": sorted(slot_terms)[:20],
        "anchor_phrase_count": len(phrase_sets),
        "subject_phrase_count": len(subject_phrase_sets),
    }
    if not question_tokens or not anchor_tokens or not subject_phrase_sets:
        return [], profile

    candidates: List[Dict[str, Any]] = []
    for rank, doc in enumerate(ranked_docs[: max(0, int(candidate_top_k))]):
        doc_key = _doc_key(doc)
        if not doc_key or doc_key in current_keys:
            continue
        title = _doc_title(doc)
        if not title:
            continue
        normalized_title = title.strip().lower()
        if normalized_title in current_titles:
            continue
        if LIST_LIKE_TITLE_RE.search(title):
            continue
        body = _doc_body(doc)
        title_tokens = _token_set(title)
        body_tokens = _token_set(body[:5000])
        doc_tokens = title_tokens | body_tokens
        if not doc_tokens:
            continue

        phrase_title_overlap = _max_phrase_overlap(phrase_sets, title_tokens)
        phrase_doc_overlap = _max_phrase_overlap(phrase_sets, doc_tokens)
        subject_title_overlap = _max_phrase_overlap(subject_phrase_sets, title_tokens)
        subject_doc_overlap = _max_phrase_overlap(subject_phrase_sets, doc_tokens)
        best_subject_token_count = _best_subject_token_count(subject_phrase_sets, title_tokens)
        informative_question_tokens = question_tokens - QUESTION_STOPWORDS - TYPE_AND_RELATION_WORDS
        short_subject_ambiguous = (
            best_subject_token_count <= 1
            and bool(title_tokens)
            and len(title_tokens) > max(1, best_subject_token_count)
            and not bool((title_tokens - anchor_tokens) & informative_question_tokens)
        )
        title_anchor_overlap = max(
            phrase_title_overlap,
            _overlap_ratio(anchor_tokens, title_tokens),
            _overlap_ratio(title_tokens, anchor_tokens),
        )
        doc_anchor_overlap = max(
            phrase_doc_overlap,
            _overlap_ratio(anchor_tokens, doc_tokens),
        )
        question_overlap = _overlap_ratio(question_tokens, doc_tokens)
        relation_overlap = _overlap_ratio(relation_terms, doc_tokens) if relation_terms else 0.0
        exact_anchor_hit = any(
            " ".join(sorted(phrase_tokens)) == " ".join(sorted(title_tokens & phrase_tokens))
            for phrase_tokens in phrase_sets
            if phrase_tokens
        )
        sentence_profile = _best_sentence_profile(
            question_tokens=question_tokens,
            anchor_tokens=anchor_tokens,
            subject_phrase_sets=subject_phrase_sets,
            relation_terms=relation_terms,
            slot_terms=slot_terms,
            title=title,
            body=body,
            target_type=target_type,
        )
        target_signal = max(
            _bounded(sentence_profile.get("target_signal", 0.0)),
            _bounded(_target_signal(body[:1600], title, target_type)),
        )
        slot_supported, slot_overlap = _slot_supported(
            target_type=target_type,
            slot_terms=slot_terms,
            sentence=str(sentence_profile.get("sentence", "") or ""),
            title=title,
            target_signal=target_signal,
            relation_overlap=float(sentence_profile.get("relation_overlap", 0.0) or 0.0),
        )
        slot_overlap = max(slot_overlap, float(sentence_profile.get("slot_overlap", 0.0) or 0.0))
        rank_prior = 1.0 / (rank + 1)
        score = (
            0.22 * subject_title_overlap
            + 0.13 * subject_doc_overlap
            + 0.13 * title_anchor_overlap
            + 0.14 * _bounded(sentence_profile.get("score", 0.0))
            + 0.11 * target_signal
            + 0.09 * question_overlap
            + 0.07 * slot_overlap
            + 0.05 * relation_overlap
            + 0.04 * (1.0 if exact_anchor_hit else 0.0)
            + 0.02 * rank_prior
        )
        candidates.append({
            "rank": rank,
            "title": title,
            "doc": doc,
            "score": score,
            "title_anchor_overlap": title_anchor_overlap,
            "doc_anchor_overlap": doc_anchor_overlap,
            "subject_title_overlap": subject_title_overlap,
            "subject_doc_overlap": subject_doc_overlap,
            "subject_sentence_overlap": _bounded(sentence_profile.get("subject_overlap", 0.0)),
            "question_overlap": question_overlap,
            "relation_overlap": relation_overlap,
            "slot_overlap": slot_overlap,
            "slot_supported": bool(slot_supported),
            "target_signal": target_signal,
            "sentence_score": _bounded(sentence_profile.get("score", 0.0)),
            "sentence": sentence_profile.get("sentence", ""),
            "exact_anchor_hit": bool(exact_anchor_hit),
            "short_subject_ambiguous": bool(short_subject_ambiguous),
            "generic_topic_candidate": _is_generic_topic_candidate(title, target_type),
        })

    candidates.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            -float(item.get("subject_title_overlap", 0.0)),
            -float(item.get("title_anchor_overlap", 0.0)),
            -float(item.get("sentence_score", 0.0)),
            int(item.get("rank", 0)),
        )
    )
    return candidates, profile


def _format_entity_fallback_doc(candidate: Dict[str, Any], target_type: str) -> str:
    title = str(candidate.get("title", "") or "").strip()
    sentence = re.sub(r"\s+", " ", str(candidate.get("sentence", "") or "")).strip()
    lines = [
        "Entity-Centric Deep Evidence Fallback",
        f"Source title: {title}",
        f"Target answer type: {target_type}",
        "Evidence:",
    ]
    if sentence:
        lines.append(f"- {sentence}")
    else:
        body = _doc_body(str(candidate.get("doc", "") or ""))
        excerpt = re.sub(r"\s+", " ", body[:480]).strip()
        if excerpt:
            lines.append(f"- {excerpt}")
    return "\n".join(lines)


def append_entity_centric_deep_evidence(
    question: str,
    current_docs: List[str],
    ranked_docs: List[str],
    planned_retrieval: Dict[str, Any] = None,
    enabled: bool = False,
    candidate_top_k: int = 80,
    max_append: int = 1,
    min_score: float = 0.44,
    min_title_anchor_overlap: float = 0.58,
    min_subject_title_overlap: float = 0.85,
    min_subject_doc_overlap: float = 0.90,
    min_sentence_score: float = 0.20,
    min_target_signal: float = 0.20,
    insert_after: int = 5,
    block_bridge: bool = True,
    block_bridge_like_relation: bool = True,
    strict_subject_match: bool = True,
    require_slot_signal: bool = True,
    block_ambiguous_short_subject: bool = True,
    current_sufficiency_guard: bool = True,
    block_empty_factoid_slot: bool = True,
    quarantine_generic_titles: bool = True,
    route_aware_relaxation: bool = True,
    route_aware_max_rank: int = 24,
    route_aware_min_score: float = 0.54,
    route_aware_min_sentence_score: float = 0.32,
    route_aware_min_question_overlap: float = 0.20,
) -> Tuple[List[str], Dict[str, Any]]:
    """Append one entity-title-matched deep passage for single-hop factoid QA.

    This is reader-only: it never changes retrieval ranking or Recall@k. It
    targets cases where the correct entity page is present deeper in the dense
    retrieval list but absent from the QA top-k context.
    """
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "candidate_top_k": int(candidate_top_k),
        "max_append": int(max_append),
        "min_score": float(min_score),
        "min_title_anchor_overlap": float(min_title_anchor_overlap),
        "min_subject_title_overlap": float(min_subject_title_overlap),
        "min_subject_doc_overlap": float(min_subject_doc_overlap),
        "min_sentence_score": float(min_sentence_score),
        "min_target_signal": float(min_target_signal),
        "insert_after": int(insert_after),
        "block_bridge": bool(block_bridge),
        "block_bridge_like_relation": bool(block_bridge_like_relation),
        "strict_subject_match": bool(strict_subject_match),
        "require_slot_signal": bool(require_slot_signal),
        "block_ambiguous_short_subject": bool(block_ambiguous_short_subject),
        "current_sufficiency_guard": bool(current_sufficiency_guard),
        "block_empty_factoid_slot": bool(block_empty_factoid_slot),
        "quarantine_generic_titles": bool(quarantine_generic_titles),
        "route_aware_relaxation_enabled": bool(route_aware_relaxation),
        "route_aware_max_rank": int(route_aware_max_rank),
        "route_aware_min_score": float(route_aware_min_score),
        "route_aware_min_sentence_score": float(route_aware_min_sentence_score),
        "route_aware_min_question_overlap": float(route_aware_min_question_overlap),
        "candidate_count": 0,
        "accepted_count": 0,
        "accepted_titles": [],
        "top_candidates": [],
    }
    current_docs = list(current_docs or [])
    ranked_docs = list(ranked_docs or [])
    if not enabled:
        return current_docs, info
    if max_append <= 0:
        info["reason"] = "max_append_zero"
        return current_docs, info
    if not ranked_docs:
        info["reason"] = "no_ranked_docs"
        return current_docs, info

    route_profile = _route_admission_profile(
        question=question,
        planned_retrieval=planned_retrieval or {},
        enabled=route_aware_relaxation,
    )
    info.update(route_profile)
    is_single_hop, route_reason = _is_single_hop_factoid(
        question=question,
        planned_retrieval=planned_retrieval or {},
        block_bridge=block_bridge,
        block_bridge_like_relation=block_bridge_like_relation,
        allow_route_aware_bridge=bool(route_profile.get("route_aware_relaxation")),
    )
    info["route_reason"] = route_reason
    if not is_single_hop:
        info["reason"] = route_reason
        return current_docs, info

    candidates, profile = _rank_entity_fallback_candidates(
        question=question,
        current_docs=current_docs,
        ranked_docs=ranked_docs,
        candidate_top_k=candidate_top_k,
    )
    info.update(profile)
    info["candidate_count"] = len(candidates)
    target_type = str(profile.get("target_type", "factoid") or "factoid")
    slot_terms = set(profile.get("slot_terms", []) or [])
    route_relaxed = bool(route_profile.get("route_aware_relaxation"))
    route_relaxed_empty_factoid = bool(
        route_relaxed and target_type == "factoid" and not slot_terms
    )
    info["route_aware_empty_factoid_relaxed"] = route_relaxed_empty_factoid
    current_slot_covered = False
    if current_sufficiency_guard and target_type != "factoid":
        anchor_phrase_sets = _anchor_phrase_sets(question)
        subject_phrase_sets = _subject_phrase_sets(question, anchor_phrase_sets)
        current_slot_covered = _current_evidence_has_slot_support(
            question=question,
            current_docs=current_docs,
            subject_phrase_sets=subject_phrase_sets,
            slot_terms=slot_terms,
            target_type=target_type,
            min_subject_doc_overlap=float(min_subject_doc_overlap),
            min_target_signal=float(min_target_signal),
        )
    info["current_slot_covered"] = bool(current_slot_covered)
    info["top_candidates"] = [
        {
            "rank": int(item.get("rank", 0)),
            "title": item.get("title", ""),
            "score": round(float(item.get("score", 0.0)), 6),
            "title_anchor_overlap": round(float(item.get("title_anchor_overlap", 0.0)), 6),
            "doc_anchor_overlap": round(float(item.get("doc_anchor_overlap", 0.0)), 6),
            "subject_title_overlap": round(float(item.get("subject_title_overlap", 0.0)), 6),
            "subject_doc_overlap": round(float(item.get("subject_doc_overlap", 0.0)), 6),
            "question_overlap": round(float(item.get("question_overlap", 0.0)), 6),
            "relation_overlap": round(float(item.get("relation_overlap", 0.0)), 6),
            "slot_overlap": round(float(item.get("slot_overlap", 0.0)), 6),
            "slot_supported": bool(item.get("slot_supported", False)),
            "target_signal": round(float(item.get("target_signal", 0.0)), 6),
            "sentence_score": round(float(item.get("sentence_score", 0.0)), 6),
            "short_subject_ambiguous": bool(item.get("short_subject_ambiguous", False)),
            "generic_topic_candidate": bool(item.get("generic_topic_candidate", False)),
        }
        for item in candidates[:5]
    ]
    if not candidates:
        info["reason"] = "no_candidates"
        return current_docs, info
    if (
        block_empty_factoid_slot
        and target_type == "factoid"
        and not slot_terms
        and not route_relaxed_empty_factoid
    ):
        info["reason"] = "empty_factoid_slot_quarantined"
        return current_docs, info
    if current_slot_covered and not route_relaxed:
        info["reason"] = "current_slot_evidence_sufficient"
        return current_docs, info

    accepted: List[Dict[str, Any]] = []
    reject_counts: Dict[str, int] = {}
    for item in candidates:
        if len(accepted) >= max_append:
            break
        reasons = []
        if float(item.get("score", 0.0)) < float(min_score):
            reasons.append("low_score")
        if float(item.get("title_anchor_overlap", 0.0)) < float(min_title_anchor_overlap):
            reasons.append("low_title_anchor_overlap")
        if strict_subject_match:
            subject_title = float(item.get("subject_title_overlap", 0.0))
            subject_doc = float(item.get("subject_doc_overlap", 0.0))
            if (
                subject_title < float(min_subject_title_overlap)
                and subject_doc < float(min_subject_doc_overlap)
            ):
                reasons.append("low_subject_match")
        if (
            block_ambiguous_short_subject
            and bool(item.get("short_subject_ambiguous", False))
        ):
            reasons.append("ambiguous_short_subject")
        if quarantine_generic_titles and bool(item.get("generic_topic_candidate", False)):
            reasons.append("generic_topic_quarantined")
        if require_slot_signal and not bool(item.get("slot_supported", False)):
            reasons.append("missing_slot_signal")
        if route_relaxed_empty_factoid:
            if int(item.get("rank", 0)) > int(route_aware_max_rank):
                reasons.append("route_aware_rank_too_deep")
            if float(item.get("score", 0.0)) < float(route_aware_min_score):
                reasons.append("route_aware_low_score")
            if float(item.get("sentence_score", 0.0)) < float(route_aware_min_sentence_score):
                reasons.append("route_aware_low_sentence_score")
            if float(item.get("question_overlap", 0.0)) < float(route_aware_min_question_overlap):
                reasons.append("route_aware_low_question_overlap")
        if float(item.get("sentence_score", 0.0)) < float(min_sentence_score):
            reasons.append("low_sentence_score")
        if float(item.get("target_signal", 0.0)) < float(min_target_signal):
            reasons.append("low_target_signal")
        if reasons:
            for reason in reasons:
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
            continue
        accepted.append(item)

    info["reject_reason_counts"] = reject_counts
    if not accepted:
        info["reason"] = "no_candidate_accepted"
        return current_docs, info

    appended_docs = [
        _format_entity_fallback_doc(item, str(profile.get("target_type", "factoid")))
        for item in accepted
    ]
    insert_at = max(0, min(len(current_docs), int(insert_after)))
    updated_docs = current_docs[:insert_at] + appended_docs + current_docs[insert_at:]
    info["applied"] = True
    info["reason"] = "appended_entity_deep_evidence"
    info["accepted_count"] = len(appended_docs)
    info["accepted_titles"] = [str(item.get("title", "") or "") for item in accepted]
    info["accepted_ranks"] = [int(item.get("rank", 0)) for item in accepted]
    return updated_docs, info
