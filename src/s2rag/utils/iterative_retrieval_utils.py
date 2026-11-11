import re
from typing import Any, Dict, List, Tuple

import numpy as np

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
NUMBER_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")
TITLE_CASE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+(?:of|the|and|de|la|[A-Z][a-z]+)){0,5}\b")
BRIDGE_WORD_RE = re.compile(
    r"\b(?:father|mother|spouse|child|director|author|writer|founder|composer|"
    r"producer|creator|birthplace|capital|located|country|city|county|state|"
    r"same|both|where|when|which)\b",
    re.IGNORECASE,
)


def _doc_title(doc: str) -> str:
    if not doc:
        return ""
    return str(doc).split("\n", 1)[0].strip()


def _doc_body(doc: str) -> str:
    if not doc:
        return ""
    parts = str(doc).split("\n", 1)
    return parts[1] if len(parts) > 1 else parts[0]


def _tokens(text: Any) -> List[str]:
    return [
        token
        for token in TOKEN_RE.findall(str(text or "").lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def _token_set(text: Any) -> set:
    return set(_tokens(text))


def _overlap_ratio(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def _split_sentences(text: str, max_body_chars: int = 4500) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "")[:max_body_chars]).strip()
    if not text:
        return []
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
        if item.strip()
    ][:24]


def _target_signal(text: str, target_type: str) -> float:
    if target_type == "date":
        return 1.0 if YEAR_RE.search(text) else 0.0
    if target_type == "number":
        return 1.0 if NUMBER_RE.search(text) else 0.0
    if target_type in {"person", "location", "organization", "work", "factoid"}:
        return 0.7 if TITLE_CASE_RE.search(text) else 0.0
    return 0.5 if TITLE_CASE_RE.search(text) or NUMBER_RE.search(text) else 0.0


def _best_question_sentence(question_tokens: set, doc: str, target_type: str) -> Tuple[str, float]:
    title = _doc_title(doc)
    best_sentence = ""
    best_score = 0.0
    for idx, sentence in enumerate(_split_sentences(_doc_body(doc))):
        sentence_tokens = _token_set(sentence)
        if not sentence_tokens:
            continue
        overlap = _overlap_ratio(question_tokens, sentence_tokens | _token_set(title))
        target = _target_signal(sentence, target_type)
        early = 1.0 / (idx + 1)
        score = 0.62 * overlap + 0.26 * target + 0.12 * early
        if score > best_score:
            best_sentence = sentence
            best_score = score
    return best_sentence, best_score


def assess_evidence_sufficiency(
    question: str,
    docs: List[str],
    qa_top_k: int = 5,
    bridge_cues: bool = False,
    bridge_threshold: float = 0.62,
    non_bridge_threshold: float = 0.48,
    min_query_coverage: float = 0.50,
    min_best_sentence_score: float = 0.24,
) -> Dict[str, Any]:
    """Heuristically decide whether the current top-k evidence is enough.

    This is intentionally label-free and deterministic. It estimates whether
    the top evidence covers the question terms, contains answer-type signals,
    and for bridge questions has enough cross-document support diversity.
    """
    docs = docs[:max(1, qa_top_k)] if docs else []
    target_type = infer_answer_target_type(question)
    question_tokens = _token_set(question)
    if not docs or not question_tokens:
        return {
            "sufficient": False,
            "reason": "no_docs_or_question_tokens",
            "score": 0.0,
            "target_type": target_type,
            "query_coverage": 0.0,
            "target_signal": 0.0,
            "bridge_support": 0.0,
        }

    title_tokens = set()
    body_tokens = set()
    best_sentence_scores = []
    bridge_title_hits = 0
    for doc in docs:
        title = _doc_title(doc)
        body = _doc_body(doc)
        title_token_set = _token_set(title)
        body_token_set = _token_set(body[:2500])
        title_tokens.update(title_token_set)
        body_tokens.update(body_token_set)
        _, sentence_score = _best_question_sentence(question_tokens, doc, target_type)
        best_sentence_scores.append(sentence_score)
        if title_token_set & question_tokens or BRIDGE_WORD_RE.search(title):
            bridge_title_hits += 1

    evidence_tokens = title_tokens | body_tokens
    query_coverage = _overlap_ratio(question_tokens, evidence_tokens)
    target_signal = max((_target_signal(f"{_doc_title(doc)} {_doc_body(doc)[:1600]}", target_type) for doc in docs), default=0.0)
    best_sentence = max(best_sentence_scores) if best_sentence_scores else 0.0
    bridge_support = min(1.0, bridge_title_hits / 2.0) if bridge_cues else 1.0

    score = (
        0.38 * query_coverage
        + 0.28 * best_sentence
        + 0.22 * target_signal
        + 0.12 * bridge_support
    )
    threshold = bridge_threshold if bridge_cues else non_bridge_threshold
    has_target_signal = target_signal > 0 or target_type == "factoid"
    sufficient = (
        score >= threshold
        and query_coverage >= min_query_coverage
        and best_sentence >= min_best_sentence_score
        and has_target_signal
    )
    if bridge_cues:
        sufficient = sufficient and bridge_support >= 1.0

    reason = "sufficient" if sufficient else "insufficient"
    if not sufficient and bridge_cues and bridge_support < 1.0:
        reason = "insufficient_bridge_support"
    elif not sufficient and best_sentence < min_best_sentence_score:
        reason = "insufficient_sentence_support"
    elif not sufficient and target_signal <= 0:
        reason = "insufficient_answer_type_signal"
    elif not sufficient and query_coverage < min_query_coverage:
        reason = "insufficient_query_coverage"

    return {
        "sufficient": sufficient,
        "reason": reason,
        "score": round(score, 6),
        "threshold": threshold,
        "min_query_coverage": min_query_coverage,
        "min_best_sentence_score": min_best_sentence_score,
        "target_type": target_type,
        "query_coverage": round(query_coverage, 6),
        "target_signal": round(target_signal, 6),
        "best_sentence_score": round(best_sentence, 6),
        "bridge_support": round(bridge_support, 6),
        "doc_count": len(docs),
    }


def build_followup_query(
    question: str,
    docs: List[str],
    max_titles: int = 5,
    max_snippets: int = 3,
) -> str:
    target_type = infer_answer_target_type(question)
    question_tokens = _token_set(question)

    titles = []
    snippets = []
    seen_titles = set()
    for doc in docs[:max_titles]:
        title = _doc_title(doc)
        if title and title not in seen_titles:
            titles.append(title)
            seen_titles.add(title)
        sentence, score = _best_question_sentence(question_tokens, doc, target_type)
        if sentence and score > 0.12:
            snippets.append(f"{title}: {sentence[:240]}")
        if len(snippets) >= max_snippets:
            break

    pieces = [
        f"Original question: {question}",
        f"Answer type needed: {target_type}",
    ]
    if titles:
        pieces.append("Current evidence titles: " + "; ".join(titles))
    if snippets:
        pieces.append("Current evidence snippets: " + " ".join(snippets))
    pieces.append("Retrieve the missing supporting passage that connects the current evidence and directly answers the original question.")
    return "\n".join(pieces)


def merge_iterative_retrieval_results(
    base_doc_ids: np.ndarray,
    base_doc_scores: np.ndarray,
    expansion_doc_ids: np.ndarray,
    expansion_doc_scores: np.ndarray,
    target_k: int = 5,
    max_promotions: int = 1,
    protect_top1: bool = True,
    protect_top_k: int = 2,
    expansion_weight: float = 0.72,
    min_expansion_score: float = 0.40,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Merge a follow-up retrieval ranking into the current retrieval ranking."""
    target_k = max(1, int(target_k))
    max_promotions = max(0, int(max_promotions))
    if protect_top1:
        protected_prefix = max(1, int(protect_top_k))
    else:
        protected_prefix = max(0, int(protect_top_k))
    protected_prefix = min(protected_prefix, target_k, len(base_doc_ids))

    base_ids = [int(item) for item in base_doc_ids.tolist()]
    base_scores = [float(item) for item in base_doc_scores.tolist()]
    expansion_ids = [int(item) for item in expansion_doc_ids.tolist()]
    expansion_scores = [float(item) for item in expansion_doc_scores.tolist()]

    base_rank = {doc_id: rank for rank, doc_id in enumerate(base_ids)}
    base_score = {doc_id: base_scores[rank] for rank, doc_id in enumerate(base_ids)}
    expansion_score = {doc_id: expansion_scores[rank] for rank, doc_id in enumerate(expansion_ids)}

    old_top = set(base_ids[:target_k])
    new_top = base_ids[:protected_prefix]
    seen = set(new_top)
    promoted = []

    for doc_id in expansion_ids:
        if len(promoted) >= max_promotions or len(new_top) >= target_k:
            break
        if expansion_score.get(doc_id, 0.0) < min_expansion_score:
            continue
        if doc_id in seen or doc_id in old_top:
            continue
        new_top.append(doc_id)
        seen.add(doc_id)
        promoted.append({
            "doc_id": doc_id,
            "from_rank": base_rank.get(doc_id),
            "expansion_rank": expansion_ids.index(doc_id),
            "expansion_score": round(expansion_score.get(doc_id, 0.0), 6),
        })

    for doc_id in base_ids[:target_k]:
        if len(new_top) >= target_k:
            break
        if doc_id in seen:
            continue
        new_top.append(doc_id)
        seen.add(doc_id)

    combined_ids = []
    for doc_id in expansion_ids + base_ids:
        if doc_id not in combined_ids:
            combined_ids.append(doc_id)

    remaining = [doc_id for doc_id in combined_ids if doc_id not in seen]
    new_ids = new_top + remaining

    new_scores = []
    for idx, doc_id in enumerate(new_ids):
        score = base_score.get(doc_id, 0.0) + expansion_weight * expansion_score.get(doc_id, 0.0)
        if doc_id in new_top:
            score += 1.0 / (idx + 1)
        new_scores.append(score)

    old_top2 = set(base_ids[:min(2, len(base_ids))])
    old_top5 = set(base_ids[:min(5, len(base_ids))])
    new_top2 = set(new_ids[:min(2, len(new_ids))])
    new_top5 = set(new_ids[:min(5, len(new_ids))])
    info = {
        "promoted_count": len(promoted),
        "promoted_to_top2": len(new_top2 - old_top2),
        "promoted_to_top5": len(new_top5 - old_top5),
        "protect_top_k": protected_prefix,
        "min_expansion_score": min_expansion_score,
        "promotions": promoted[:8],
    }
    return np.array(new_ids, dtype=int), np.array(new_scores, dtype=float), info
