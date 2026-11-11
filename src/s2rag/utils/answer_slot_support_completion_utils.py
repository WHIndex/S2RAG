import math
import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Set, Tuple

from .candidate_answer_selector_utils import _is_synthetic_doc
from .evidence_utils import has_bridge_cues
from .sentence_evidence_utils import infer_answer_target_type
from .support_rerank_utils import (
    _doc_body,
    _doc_title,
    _split_sentences,
    _target_signal,
    _token_set,
)


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
TITLE_PHRASE_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9'.-]*(?:\s+(?:of|the|and|de|la|[A-Z][A-Za-z0-9'.-]*)){0,6}\b"
)
POSSESSIVE_RE = re.compile(r"\b([A-Z][^?]{1,90}?)'s\b")
DATE_RE = re.compile(
    r"\b(?:[1-9][0-9]{2,3}s|[1-9][0-9]{2,3}|20[0-9]{2}|21[0-9]{2}|"
    r"[0-9]{1,2}(?:st|nd|rd|th)\s+century|"
    r"january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")
SLOT_PREFIX = "Answer-Slot Support Completion View"
LINEAGE_SAFE_SLOT_PREFIX = "Lineage-Safe Answer-Slot Support View"

GENERIC_ANCHOR_STOPWORDS = {
    "answer", "called", "current", "following", "given", "known", "list",
    "name", "named", "question", "title", "type",
}
RELATION_STOPWORDS = {
    "answer", "date", "fact", "location", "name", "number", "person",
    "place", "type", "year",
}
BOOLEAN_COMPARISON_RE = re.compile(
    r"^\s*(?:do|does|did|is|are|was|were|can|could|has|have|had)\b|"
    r"\b(?:both|same|different|compare|compared|comparison|lowest|highest|"
    r"largest|smallest|most|least|earliest|latest|first|last|older|younger)\b",
    re.IGNORECASE,
)
LIST_LIKE_TITLE_RE = re.compile(
    r"\b(?:disambiguation|people with the last name|people with the surname|"
    r"list of|index of)\b|\(name\)",
    re.IGNORECASE,
)
LIST_LIKE_SENTENCE_RE = re.compile(
    r"\b(?:several people|may refer to|include:|including:)\b",
    re.IGNORECASE,
)

SLOT_PROFILES = (
    {
        "slot": "mother",
        "pattern": re.compile(r"\bmother\b", re.IGNORECASE),
        "relations": {"mother", "maternal", "parent"},
    },
    {
        "slot": "father",
        "pattern": re.compile(r"\bfather\b", re.IGNORECASE),
        "relations": {"father", "paternal", "parent"},
    },
    {
        "slot": "spouse",
        "pattern": re.compile(r"\b(?:spouse|wife|husband|married)\b", re.IGNORECASE),
        "relations": {"spouse", "wife", "husband", "married"},
    },
    {
        "slot": "director",
        "pattern": re.compile(r"\b(?:director|directed)\b", re.IGNORECASE),
        "relations": {"director", "directed"},
    },
    {
        "slot": "author",
        "pattern": re.compile(r"\b(?:author|writer|written|wrote)\b", re.IGNORECASE),
        "relations": {"author", "writer", "written", "wrote"},
    },
    {
        "slot": "creator",
        "pattern": re.compile(
            r"\b(?:creator|created|composer|composed|producer|produced|founder|founded)\b",
            re.IGNORECASE,
        ),
        "relations": {"creator", "created", "composer", "producer", "founder", "founded"},
    },
    {
        "slot": "occupation",
        "pattern": re.compile(r"\b(?:occupation|profession|job|known for)\b", re.IGNORECASE),
        "relations": {"occupation", "profession", "job", "known"},
    },
    {
        "slot": "nationality",
        "pattern": re.compile(r"\b(?:nationality|citizenship|citizen|national)\b", re.IGNORECASE),
        "relations": {"nationality", "citizenship", "citizen", "national", "country"},
    },
    {
        "slot": "birth",
        "pattern": re.compile(r"\b(?:birthplace|place of birth|born)\b", re.IGNORECASE),
        "relations": {"birthplace", "birth", "born"},
    },
    {
        "slot": "death",
        "pattern": re.compile(r"\b(?:died|death|date of death)\b", re.IGNORECASE),
        "relations": {"died", "death"},
    },
    {
        "slot": "foundation",
        "pattern": re.compile(r"\b(?:founded|established|formed|abolished|created)\b", re.IGNORECASE),
        "relations": {"founded", "established", "formed", "abolished", "created"},
    },
    {
        "slot": "setting",
        "pattern": re.compile(r"\b(?:take place|takes place|took place|set in|setting)\b", re.IGNORECASE),
        "relations": {"take", "takes", "took", "place", "set", "setting", "occur", "occurs"},
    },
    {
        "slot": "location",
        "pattern": re.compile(r"\b(?:where|located|city|country|state|county|capital)\b", re.IGNORECASE),
        "relations": {"where", "located", "city", "country", "state", "county", "capital"},
    },
)


def _doc_key(doc: Any) -> str:
    return re.sub(r"\s+", " ", str(doc or "")).strip().lower()


def _tokens(text: Any) -> Set[str]:
    return set(TOKEN_RE.findall(str(text or "").lower()))


def _overlap_ratio(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def _bounded(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _clean_subject_phrase(text: str) -> str:
    phrase = re.sub(r"\s+", " ", str(text or "")).strip(" ,:;?-")
    phrase = re.sub(
        r"^(?:(?:what|who|where|when|which|year|date|time|period|era|"
        r"is|are|was|were|does|do|did|the|a|an)\s+)+",
        "",
        phrase,
        flags=re.IGNORECASE,
    ).strip(" ,:;?-")
    phrase = re.sub(
        r"\b(?:occupation|profession|nationality|citizenship|birthplace|"
        r"place of birth|date of death|father|mother|spouse|wife|husband)\b.*$",
        "",
        phrase,
        flags=re.IGNORECASE,
    ).strip(" ,:;?-")
    phrase = re.sub(
        r"\s+(?:take|takes|took)\s+place\b.*$|\s+(?:set|located|born)\b.*$",
        "",
        phrase,
        flags=re.IGNORECASE,
    ).strip(" ,:;?-")
    return phrase


def _subject_phrases(question: str) -> List[str]:
    question_text = str(question or "")
    phrases: List[str] = []
    patterns = (
        r"\b(?:what|who|where|when|which)\s+(?:is|are|was|were|does|did)?\s*([A-Z][^?]{1,90}?)'s\b",
        r"\b(?:what|who|where|when|which)\s+(?:is|are|was|were)\s+([A-Z][A-Za-z0-9'. -]{2,80}?)(?:\s+(?:occupation|profession|nationality|citizenship|birthplace|place of birth)\b|\?)",
        r"\b(?:what\s+(?:year|date|time|period|era)|when|where)\s+(?:does|did|do|was|is)\s+([A-Z][A-Za-z0-9'. -]{2,80}?)(?:\s+(?:take|takes|took)\s+place\b|\s+(?:set|located|born)\b|\?)",
        r"\b(?:where|when)\s+(?:does|did|was|is)\s+([A-Z][A-Za-z0-9'. -]{2,80}?)(?:\s+(?:take|takes|took|set|located|born)\b|\?)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, question_text, flags=re.IGNORECASE):
            phrase = _clean_subject_phrase(match.group(1))
            if phrase and len(_token_set(phrase)) >= 1:
                phrases.append(phrase)
    for match in POSSESSIVE_RE.finditer(question_text):
        phrase = _clean_subject_phrase(match.group(1))
        if phrase and len(_token_set(phrase)) >= 1:
            phrases.append(phrase)
    seen = set()
    unique = []
    for phrase in phrases:
        key = " ".join(sorted(_token_set(phrase)))
        if key and key not in seen:
            seen.add(key)
            unique.append(phrase)
    return unique[:4]


def _is_list_like_source(title: str, sentence: str) -> bool:
    title_text = str(title or "")
    sentence_text = str(sentence or "")
    if LIST_LIKE_TITLE_RE.search(title_text):
        return True
    if LIST_LIKE_SENTENCE_RE.search(sentence_text):
        return True
    if sentence_text.count(";") >= 2:
        return True
    return False


def _infer_slot_profile(question: str) -> Dict[str, Any]:
    question_text = str(question or "")
    target_type = infer_answer_target_type(question_text)
    slots = []
    relation_terms: Set[str] = set()
    for profile in SLOT_PROFILES:
        if profile["pattern"].search(question_text):
            slots.append(profile["slot"])
            relation_terms.update(profile["relations"])
    if target_type in {"date", "number", "location", "nationality", "occupation"}:
        slots.append(target_type)
        relation_terms.add(target_type)
    if not slots:
        slots.append(target_type if target_type != "factoid" else "generic")
    relation_terms = {tok for tok in relation_terms if tok not in RELATION_STOPWORDS}
    subject_phrases = _subject_phrases(question_text)
    subject_tokens = set()
    for phrase in subject_phrases:
        subject_tokens.update(_token_set(phrase))
    return {
        "target_type": target_type,
        "answer_slot": "+".join(dict.fromkeys(slots)),
        "relation_terms": relation_terms,
        "subject_phrases": subject_phrases,
        "subject_tokens": subject_tokens,
        "boolean_or_comparison": bool(BOOLEAN_COMPARISON_RE.search(question_text)),
        "single_hop_attribute": _is_single_hop_attribute_query(question_text, target_type, relation_terms),
    }


def _is_single_hop_attribute_query(question: str, target_type: str, relation_terms: Set[str]) -> bool:
    q = str(question or "").lower()
    if not re.search(r"\b(?:what|who|where|when|which)\b", q):
        return False
    if has_bridge_cues(question):
        return False
    if re.search(r"\b(?: of | by | from | whose | that | which )\b.{0,100}\b(?:of|by|from|whose|that|which)\b", f" {q} "):
        return False
    if target_type in {"occupation", "nationality", "location", "date", "number"}:
        return True
    return bool(relation_terms and re.search(r"\b(?:occupation|profession|nationality|citizenship|born|located)\b", q))


def _anchor_tokens(question: str, relation_terms: Set[str]) -> Set[str]:
    anchors: Set[str] = set()
    for match in POSSESSIVE_RE.finditer(str(question or "")):
        anchors.update(_token_set(match.group(1)))
    for phrase in TITLE_PHRASE_RE.findall(str(question or "")):
        phrase_tokens = _token_set(phrase)
        if phrase_tokens and not (phrase_tokens <= relation_terms):
            anchors.update(phrase_tokens)
    q_tokens = _token_set(question)
    anchors.update(
        token
        for token in q_tokens
        if token not in relation_terms
        and token not in GENERIC_ANCHOR_STOPWORDS
        and len(token) > 2
    )
    return anchors


def _slot_value_signal(text: str, title: str, target_type: str) -> float:
    base = _target_signal(text, title, target_type)
    if target_type == "date" and DATE_RE.search(f"{title} {text}"):
        return 1.0
    if target_type == "number" and NUMBER_RE.search(f"{title} {text}"):
        return 1.0
    return _bounded(base)


def _best_slot_sentence(
    question_tokens: Set[str],
    anchor_tokens: Set[str],
    subject_tokens: Set[str],
    relation_terms: Set[str],
    current_tokens: Set[str],
    current_title_tokens: Set[str],
    title: str,
    body: str,
    target_type: str,
) -> Dict[str, Any]:
    best: Dict[str, Any] = {}
    title_tokens = _token_set(title)
    missing_tokens = question_tokens - current_tokens
    if not missing_tokens:
        missing_tokens = question_tokens

    for sent_rank, sentence in enumerate(_split_sentences(body, max_body_chars=7000)):
        sent_tokens = _token_set(sentence)
        if not sent_tokens:
            continue
        doc_sentence_tokens = sent_tokens | title_tokens
        query_overlap = _overlap_ratio(question_tokens, doc_sentence_tokens)
        missing_gain = _overlap_ratio(missing_tokens, doc_sentence_tokens)
        anchor_overlap = max(
            _overlap_ratio(anchor_tokens, doc_sentence_tokens),
            _overlap_ratio(title_tokens, anchor_tokens),
        )
        subject_overlap = max(
            _overlap_ratio(subject_tokens, doc_sentence_tokens),
            _overlap_ratio(subject_tokens, title_tokens),
            _overlap_ratio(subject_tokens, sent_tokens),
        )
        relation_overlap = _overlap_ratio(relation_terms, doc_sentence_tokens)
        slot_signal = _slot_value_signal(sentence, title, target_type)
        chain_link = max(
            _overlap_ratio(title_tokens, current_tokens),
            _overlap_ratio(current_title_tokens, sent_tokens),
            _overlap_ratio(sent_tokens, current_title_tokens),
        )
        early = 1.0 / (sent_rank + 1)
        score = (
            0.22 * query_overlap
            + 0.18 * missing_gain
            + 0.12 * anchor_overlap
            + 0.12 * subject_overlap
            + 0.16 * relation_overlap
            + 0.18 * slot_signal
            + 0.06 * chain_link
            + 0.02 * early
        )
        item = {
            "sentence": sentence,
            "sentence_rank": sent_rank,
            "query_overlap": query_overlap,
            "missing_gain": missing_gain,
            "anchor_overlap": anchor_overlap,
            "subject_overlap": subject_overlap,
            "relation_overlap": relation_overlap,
            "slot_signal": slot_signal,
            "chain_link": chain_link,
            "list_like_source": _is_list_like_source(title, sentence),
            "score": score,
        }
        if not best or score > float(best.get("score", 0.0)):
            best = item
    return best


def _rank_rows(
    question: str,
    current_docs: Sequence[str],
    ranked_docs: Sequence[str],
    candidate_top_k: int,
    include_current: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    profile = _infer_slot_profile(question)
    question_tokens = _token_set(question)
    relation_terms = set(profile.get("relation_terms") or set())
    subject_tokens = set(profile.get("subject_tokens") or set())
    anchor_tokens = _anchor_tokens(question, relation_terms)
    skip_counts: Counter = Counter()
    current_source_docs = [
        str(doc or "")
        for doc in current_docs or []
        if doc and not _is_synthetic_doc(str(doc or ""))
    ]
    current_tokens = _token_set(" ".join(current_source_docs)[:18000])
    current_title_tokens: Set[str] = set()
    for doc in current_source_docs:
        current_title_tokens.update(_token_set(_doc_title(doc)))
    seen_keys = {_doc_key(doc) for doc in current_docs or [] if _doc_key(doc)}

    pools: List[Tuple[str, int, str]] = []
    if include_current:
        pools.extend(("current", rank, str(doc)) for rank, doc in enumerate(current_source_docs[: max(1, len(current_source_docs))]))
    pools.extend(
        ("ranked", rank, str(doc))
        for rank, doc in enumerate((ranked_docs or [])[: max(0, int(candidate_top_k))])
    )

    rows: List[Dict[str, Any]] = []
    row_keys = set()
    for source, rank, doc in pools:
        doc_key = _doc_key(doc)
        if not doc_key:
            continue
        if _is_synthetic_doc(doc):
            skip_counts["synthetic_doc"] += 1
            continue
        if source == "ranked" and doc_key in seen_keys:
            continue
        title = _doc_title(doc)
        body = _doc_body(doc)
        best = _best_slot_sentence(
            question_tokens=question_tokens,
            anchor_tokens=anchor_tokens,
            subject_tokens=subject_tokens,
            relation_terms=relation_terms,
            current_tokens=current_tokens,
            current_title_tokens=current_title_tokens,
            title=title,
            body=body,
            target_type=str(profile.get("target_type") or "factoid"),
        )
        if not best:
            continue
        key = (title.lower(), best["sentence"].lower()[:180])
        if key in row_keys:
            continue
        row_keys.add(key)
        rank_prior = 1.0 / (rank + 1)
        best["score"] = _bounded(float(best["score"]) + (0.04 * rank_prior if source == "current" else 0.02 * rank_prior))
        best.update({
            "source": source,
            "rank": rank,
            "title": title,
        })
        rows.append(best)

    rows.sort(
        key=lambda row: (
            bool(row.get("list_like_source", False)),
            -float(row.get("score", 0.0)),
            -float(row.get("subject_overlap", 0.0)),
            -float(row.get("slot_signal", 0.0)),
            -float(row.get("chain_link", 0.0)),
            int(row.get("rank", 0)),
        )
    )
    profile["question_token_count"] = len(question_tokens)
    profile["anchor_token_count"] = len(anchor_tokens)
    profile["current_source_doc_count"] = len(current_source_docs)
    profile["skip_reason_counts"] = dict(skip_counts)
    return rows, profile


def _format_view(profile: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    title = LINEAGE_SAFE_SLOT_PREFIX if profile.get("lineage_safe") else SLOT_PREFIX
    lines = [
        title,
        f"Expected answer slot: {profile.get('answer_slot', 'unknown')}",
        f"Expected answer type: {profile.get('target_type', 'factoid')}",
        f"Subject lineage: {', '.join(profile.get('subject_phrases') or []) or 'unavailable'}",
        "Use these rows only when the source, relation role, and answer slot agree:",
    ]
    for row in rows:
        sentence = re.sub(r"\s+", " ", str(row.get("sentence") or "")).strip()
        if len(sentence) > 360:
            sentence = sentence[:357].rstrip() + "..."
        lines.append(
            "- "
            f"Source={row.get('title', '')}; "
            f"role={row.get('relation_overlap', 0.0):.2f}; "
            f"slot={row.get('slot_signal', 0.0):.2f}; "
            f"subject={row.get('subject_overlap', 0.0):.2f}; "
            f"chain={row.get('chain_link', 0.0):.2f}; "
            f"evidence: {sentence}"
        )
    return "\n".join(lines)


def append_answer_slot_support_completion(
    question: str,
    current_docs: List[str],
    ranked_docs: List[str],
    planned_retrieval: Dict[str, Any] = None,
    enabled: bool = False,
    bridge_or_attribute_only: bool = True,
    candidate_top_k: int = 40,
    max_rows: int = 4,
    min_row_score: float = 0.40,
    min_slot_signal: float = 0.30,
    min_anchor_or_chain: float = 0.08,
    min_missing_gain: float = 0.00,
    insert_after: int = 2,
    include_current_rows: bool = True,
    lineage_safe: bool = True,
    allow_bridge: bool = False,
    min_relation_overlap: float = 0.08,
    min_subject_overlap: float = 0.55,
    conflict_margin: float = 0.05,
    block_boolean_comparison: bool = True,
    block_list_like_sources: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    """Append a compact answer-slot support view without changing retrieval ranks.

    This is a reader-side completion operator: it materializes high-confidence
    role/slot evidence rows from current evidence plus deeper ranked passages.
    It is deliberately conservative about admission but does not rewrite or
    promote retrieval results, so Recall@2/5 remains stable.
    """
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "bridge_cues": bool(has_bridge_cues(question)),
        "bridge_or_attribute_only": bool(bridge_or_attribute_only),
        "lineage_safe": bool(lineage_safe),
        "allow_bridge": bool(allow_bridge),
        "min_relation_overlap": float(min_relation_overlap),
        "min_subject_overlap": float(min_subject_overlap),
        "conflict_margin": float(conflict_margin),
        "block_boolean_comparison": bool(block_boolean_comparison),
        "block_list_like_sources": bool(block_list_like_sources),
        "candidate_top_k": int(candidate_top_k),
        "max_rows": int(max_rows),
        "min_row_score": float(min_row_score),
        "min_slot_signal": float(min_slot_signal),
        "min_anchor_or_chain": float(min_anchor_or_chain),
        "min_missing_gain": float(min_missing_gain),
        "candidate_count": 0,
        "accepted_count": 0,
        "accepted_titles": [],
        "reject_reason_counts": {},
        "top_rows": [],
    }
    docs = list(current_docs or [])
    if not enabled:
        return docs, info
    if max_rows <= 0:
        info["reason"] = "max_rows_zero"
        return docs, info
    if not docs:
        info["reason"] = "no_current_docs"
        return docs, info

    rows, profile = _rank_rows(
        question=question,
        current_docs=docs,
        ranked_docs=ranked_docs or [],
        candidate_top_k=candidate_top_k,
        include_current=include_current_rows,
    )
    info.update({
        "answer_slot": profile.get("answer_slot"),
        "target_type": profile.get("target_type"),
        "subject_phrases": profile.get("subject_phrases"),
        "boolean_or_comparison": bool(profile.get("boolean_or_comparison")),
        "single_hop_attribute": bool(profile.get("single_hop_attribute")),
        "question_token_count": profile.get("question_token_count"),
        "anchor_token_count": profile.get("anchor_token_count"),
        "skip_reason_counts": profile.get("skip_reason_counts") or {},
    })
    if bridge_or_attribute_only and not (info["bridge_cues"] or info["single_hop_attribute"]):
        info["reason"] = "no_bridge_or_attribute_signal"
        return docs, info
    if lineage_safe and block_boolean_comparison and info["boolean_or_comparison"]:
        info["reason"] = "blocked_boolean_or_comparison"
        return docs, info
    if lineage_safe and info["bridge_cues"] and not allow_bridge:
        info["reason"] = "blocked_bridge_lineage_safe"
        return docs, info

    info["candidate_count"] = len(rows)
    info["top_rows"] = [
        {
            "title": row.get("title"),
            "source": row.get("source"),
            "rank": int(row.get("rank", 0)),
            "score": round(float(row.get("score", 0.0)), 6),
            "slot_signal": round(float(row.get("slot_signal", 0.0)), 6),
            "anchor_overlap": round(float(row.get("anchor_overlap", 0.0)), 6),
            "subject_overlap": round(float(row.get("subject_overlap", 0.0)), 6),
            "chain_link": round(float(row.get("chain_link", 0.0)), 6),
            "relation_overlap": round(float(row.get("relation_overlap", 0.0)), 6),
            "missing_gain": round(float(row.get("missing_gain", 0.0)), 6),
            "list_like_source": bool(row.get("list_like_source", False)),
        }
        for row in rows[:5]
    ]

    accepted: List[Dict[str, Any]] = []
    reject_counts: Counter = Counter()
    relation_terms_for_gate = set(profile.get("relation_terms") or set())
    for row in rows:
        reasons = []
        if float(row.get("score", 0.0)) < min_row_score:
            reasons.append("low_row_score")
        if float(row.get("slot_signal", 0.0)) < min_slot_signal:
            reasons.append("low_slot_signal")
        if max(float(row.get("anchor_overlap", 0.0)), float(row.get("chain_link", 0.0))) < min_anchor_or_chain:
            reasons.append("low_anchor_or_chain")
        if float(row.get("missing_gain", 0.0)) < min_missing_gain:
            reasons.append("low_missing_gain")
        if lineage_safe:
            if block_list_like_sources and bool(row.get("list_like_source", False)):
                reasons.append("list_like_source")
            if info["single_hop_attribute"]:
                if float(row.get("subject_overlap", 0.0)) < min_subject_overlap:
                    reasons.append("low_subject_overlap")
                if (
                    str(info.get("target_type") or "") in {"date", "location", "number"}
                    and relation_terms_for_gate
                ):
                    if float(row.get("relation_overlap", 0.0)) < min_relation_overlap:
                        reasons.append("low_relation_overlap")
            elif info["bridge_cues"]:
                if float(row.get("relation_overlap", 0.0)) < min_relation_overlap:
                    reasons.append("low_relation_overlap")
                if float(row.get("anchor_overlap", 0.0)) < min_anchor_or_chain:
                    reasons.append("low_anchor_overlap")
                if float(row.get("chain_link", 0.0)) < min_anchor_or_chain:
                    reasons.append("low_chain_link")
        if reasons:
            reject_counts.update(reasons)
            continue
        accepted.append(row)
        if len(accepted) >= max_rows:
            break

    info["reject_reason_counts"] = dict(reject_counts)
    if not accepted:
        info["reason"] = "no_rows_accepted" if rows else "no_candidate_rows"
        return docs, info
    if lineage_safe and len(accepted) == 1:
        winner = accepted[0]
        for row in rows:
            if row is winner:
                continue
            if bool(row.get("list_like_source", False)) and block_list_like_sources:
                continue
            if float(row.get("score", 0.0)) < min_row_score:
                continue
            if float(row.get("slot_signal", 0.0)) < min_slot_signal:
                continue
            if max(float(row.get("anchor_overlap", 0.0)), float(row.get("chain_link", 0.0))) < min_anchor_or_chain:
                continue
            if info["single_hop_attribute"]:
                if float(row.get("subject_overlap", 0.0)) < min_subject_overlap:
                    continue
                if (
                    str(info.get("target_type") or "") in {"date", "location", "number"}
                    and relation_terms_for_gate
                    and float(row.get("relation_overlap", 0.0)) < min_relation_overlap
                ):
                    continue
            elif info["bridge_cues"]:
                if float(row.get("relation_overlap", 0.0)) < min_relation_overlap:
                    continue
                if float(row.get("anchor_overlap", 0.0)) < min_anchor_or_chain:
                    continue
                if float(row.get("chain_link", 0.0)) < min_anchor_or_chain:
                    continue
            if abs(float(winner.get("score", 0.0)) - float(row.get("score", 0.0))) <= conflict_margin:
                info["reason"] = "blocked_ambiguous_slot_rows"
                info["ambiguous_title"] = row.get("title")
                info["ambiguous_score"] = round(float(row.get("score", 0.0)), 6)
                info["reject_reason_counts"] = dict(reject_counts)
                return docs, info
            break

    profile["lineage_safe"] = bool(lineage_safe)
    view = _format_view(profile, accepted)
    insert_at = min(max(0, int(insert_after)), len(docs))
    result_docs = docs[:insert_at] + [view] + docs[insert_at:]
    info["applied"] = True
    info["reason"] = "inserted_answer_slot_support_view"
    info["accepted_count"] = len(accepted)
    info["accepted_titles"] = [str(row.get("title") or "") for row in accepted]
    info["accepted_rows"] = [
        {
            "title": row.get("title"),
            "source": row.get("source"),
            "rank": int(row.get("rank", 0)),
            "score": round(float(row.get("score", 0.0)), 6),
            "slot_signal": round(float(row.get("slot_signal", 0.0)), 6),
            "subject_overlap": round(float(row.get("subject_overlap", 0.0)), 6),
            "anchor_overlap": round(float(row.get("anchor_overlap", 0.0)), 6),
            "relation_overlap": round(float(row.get("relation_overlap", 0.0)), 6),
            "chain_link": round(float(row.get("chain_link", 0.0)), 6),
            "sentence_preview": re.sub(r"\s+", " ", str(row.get("sentence") or "")).strip()[:240],
        }
        for row in accepted
    ]
    info["view_insert_after"] = insert_at
    return result_docs, info
