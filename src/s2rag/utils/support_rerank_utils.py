import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
MONTH_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan\.|feb\.|mar\.|apr\.|jun\.|jul\.|"
    r"aug\.|sep\.|sept\.|oct\.|nov\.|dec\.)\b",
    re.IGNORECASE,
)
TITLE_CASE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+(?:of|the|and|de|la|[A-Z][a-z]+)){0,5}\b")
LOCATION_WORD_RE = re.compile(
    r"\b(?:city|town|village|county|state|province|country|region|district|"
    r"capital|river|lake|mountain|island|municipality|prefecture|kingdom|"
    r"republic|territory)\b",
    re.IGNORECASE,
)
ORG_WORD_RE = re.compile(
    r"\b(?:university|college|school|institute|company|corporation|inc\.?|ltd\.?|"
    r"club|team|party|association|organization|organisation|agency|committee|"
    r"ministry|department|foundation|league|group|band)\b",
    re.IGNORECASE,
)
WORK_WORD_RE = re.compile(
    r"\b(?:film|movie|album|song|single|book|novel|poem|play|series|episode|"
    r"opera|work|publication|magazine|newspaper|game)\b",
    re.IGNORECASE,
)
OCCUPATION_WORD_RE = re.compile(
    r"\b(?:actor|actress|singer|writer|author|director|producer|poet|"
    r"politician|scientist|engineer|artist|painter|footballer|player|coach|"
    r"journalist|historian|professor|mathematician|physician|lawyer|designer|"
    r"architect|musician|rapper|novelist|screenwriter|businessperson)\b",
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


def _normalize_text(text: Any) -> str:
    return " ".join(TOKEN_RE.findall(str(text or "").lower()))


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


def _normalize_values(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v <= min_v:
        return [1.0 for _ in values]
    return [(value - min_v) / (max_v - min_v) for value in values]


def _safe_score(scores: Optional[np.ndarray], index: int) -> float:
    if scores is None:
        return 0.0
    try:
        score = float(scores[index])
    except Exception:
        return 0.0
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return score


def _split_sentences(text: str, max_body_chars: int = 5000) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "")[:max_body_chars]).strip()
    if not text:
        return []
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
        if item.strip()
    ][:30]


def _target_signal(text: str, title: str, target_type: str) -> float:
    joined = f"{title} {text}"
    if target_type == "date":
        return 1.0 if (YEAR_RE.search(joined) or MONTH_RE.search(joined)) else 0.0
    if target_type == "number":
        return 1.0 if NUMBER_RE.search(joined) else 0.0
    if target_type == "location":
        if LOCATION_WORD_RE.search(joined):
            return 1.0
        return 0.45 if TITLE_CASE_RE.search(text) else 0.0
    if target_type == "person":
        return 1.0 if TITLE_CASE_RE.search(text) else 0.0
    if target_type == "organization":
        if ORG_WORD_RE.search(joined):
            return 1.0
        return 0.45 if TITLE_CASE_RE.search(text) else 0.0
    if target_type == "work":
        if WORK_WORD_RE.search(joined) or '"' in text or "'" in text:
            return 1.0
        return 0.45 if TITLE_CASE_RE.search(text) else 0.0
    if target_type == "occupation":
        return 1.0 if OCCUPATION_WORD_RE.search(joined) else 0.0
    return 0.35 if (TITLE_CASE_RE.search(text) or YEAR_RE.search(text) or NUMBER_RE.search(text)) else 0.0


def _best_sentence_score(question_tokens: set, title: str, body: str, target_type: str) -> float:
    best = 0.0
    for sent_idx, sentence in enumerate(_split_sentences(body)):
        sent_tokens = _token_set(sentence)
        if not sent_tokens:
            continue
        question_overlap = _overlap_ratio(question_tokens, sent_tokens)
        target = _target_signal(sentence, title, target_type)
        early = 1.0 / (sent_idx + 1)
        score = 0.58 * question_overlap + 0.30 * target + 0.12 * early
        best = max(best, score)
    return best


def support_aware_rerank_docs(
    query: str,
    sorted_doc_ids: np.ndarray,
    sorted_doc_scores: np.ndarray,
    candidate_docs: List[str],
    enabled: bool = False,
    rerank_top_k: int = 50,
    target_k: int = 5,
    weight: float = 0.55,
    min_support_score: float = 0.34,
    max_promotions: int = 3,
    protect_top1: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Conservatively promote likely support passages into top-k retrieval.

    This module is retrieval-stage: it changes the ranked document ids used by
    retrieval recall evaluation. It uses only the query and candidate passages,
    not labels or dataset names.
    """
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "rerank_top_k": rerank_top_k,
        "target_k": target_k,
        "weight": weight,
        "min_support_score": min_support_score,
        "max_promotions": max_promotions,
        "protect_top1": bool(protect_top1),
        "target_type": infer_answer_target_type(query),
        "promoted_to_top2": 0,
        "promoted_to_top5": 0,
        "promotions": [],
    }

    if not enabled:
        return sorted_doc_ids, sorted_doc_scores, info
    if sorted_doc_ids is None or len(sorted_doc_ids) == 0:
        info["reason"] = "no_candidates"
        return sorted_doc_ids, sorted_doc_scores, info

    top_n = min(max(target_k, rerank_top_k), len(sorted_doc_ids), len(candidate_docs))
    if top_n <= target_k:
        info["reason"] = "not_enough_candidates"
        return sorted_doc_ids, sorted_doc_scores, info

    target_k = min(max(1, target_k), top_n)
    protected_prefix = 1 if protect_top1 else 0
    question_tokens = _token_set(query)
    query_norm = _normalize_text(query)
    if not question_tokens:
        info["reason"] = "no_question_tokens"
        return sorted_doc_ids, sorted_doc_scores, info

    original_scores = [_safe_score(sorted_doc_scores, rank) for rank in range(top_n)]
    normalized_original_scores = _normalize_values(original_scores)

    top_title_tokens = set()
    for doc in candidate_docs[:target_k]:
        top_title_tokens.update(_token_set(_doc_title(doc)))

    scored = []
    for rank in range(top_n):
        doc_id = int(sorted_doc_ids[rank])
        doc = candidate_docs[rank]
        title = _doc_title(doc)
        body = _doc_body(doc)
        title_tokens = _token_set(title)
        body_tokens = _token_set(body[:3000])
        doc_tokens = title_tokens | body_tokens

        query_overlap = _overlap_ratio(question_tokens, doc_tokens)
        title_query_overlap = _overlap_ratio(question_tokens, title_tokens)
        title_coverage = _overlap_ratio(title_tokens, question_tokens)
        title_norm = _normalize_text(title)
        exact_title = bool(title_norm and len(title_norm) >= 4 and title_norm in query_norm)
        if not exact_title and len(title_tokens) >= 2 and title_coverage >= 0.80:
            exact_title = True

        sentence_score = _best_sentence_score(question_tokens, title, body, info["target_type"])
        target_signal = _target_signal(body[:1200], title, info["target_type"])
        chain_overlap = _overlap_ratio(title_tokens, top_title_tokens)
        rank_prior = 1.0 / (rank + 1)

        title_anchor = 1.0 if exact_title else max(title_query_overlap, title_coverage)
        support_score = (
            0.26 * query_overlap
            + 0.22 * title_anchor
            + 0.22 * sentence_score
            + 0.14 * target_signal
            + 0.08 * chain_overlap
            + 0.08 * rank_prior
        )
        final_score = normalized_original_scores[rank] + weight * support_score

        scored.append({
            "doc_id": doc_id,
            "rank": rank,
            "title": title,
            "original_score": original_scores[rank],
            "normalized_original_score": normalized_original_scores[rank],
            "support_score": support_score,
            "final_score": final_score,
            "query_overlap": query_overlap,
            "title_anchor": title_anchor,
            "sentence_score": sentence_score,
            "target_signal": target_signal,
            "chain_overlap": chain_overlap,
        })

    original_top_ids = [int(doc_id) for doc_id in sorted_doc_ids[:target_k].tolist()]
    original_top_set = set(original_top_ids)
    promotable = [
        item for item in scored[target_k:]
        if item["support_score"] >= min_support_score
    ]
    promotable.sort(
        key=lambda item: (
            -item["support_score"],
            -item["sentence_score"],
            -item["title_anchor"],
            item["rank"],
        )
    )

    new_top = []
    if protected_prefix:
        new_top.append(int(sorted_doc_ids[0]))

    promoted = []
    seen = set(new_top)
    for item in promotable:
        if len(promoted) >= max(0, max_promotions):
            break
        if len(new_top) >= target_k:
            break
        if item["doc_id"] in seen:
            continue
        new_top.append(item["doc_id"])
        seen.add(item["doc_id"])
        if item["doc_id"] not in original_top_set:
            promoted.append(item)

    for doc_id in original_top_ids:
        if len(new_top) >= target_k:
            break
        if doc_id in seen:
            continue
        new_top.append(doc_id)
        seen.add(doc_id)

    for item in sorted(scored[:top_n], key=lambda item: (-item["final_score"], item["rank"])):
        if len(new_top) >= target_k:
            break
        if item["doc_id"] in seen:
            continue
        new_top.append(item["doc_id"])
        seen.add(item["doc_id"])

    remaining = []
    for doc_id in [int(doc_id) for doc_id in sorted_doc_ids.tolist()]:
        if doc_id in seen:
            continue
        remaining.append(doc_id)

    new_doc_ids = np.array(new_top + remaining, dtype=int)
    score_by_id = {item["doc_id"]: item["final_score"] for item in scored}
    new_scores = []
    for rank, doc_id in enumerate(new_doc_ids.tolist()):
        if doc_id in score_by_id:
            new_scores.append(score_by_id[doc_id])
        else:
            try:
                original_rank = int(np.where(sorted_doc_ids == doc_id)[0][0])
                new_scores.append(float(sorted_doc_scores[original_rank]))
            except Exception:
                new_scores.append(0.0)

    new_top2 = set(new_doc_ids[:min(2, len(new_doc_ids))].tolist())
    new_top5 = set(new_doc_ids[:min(5, len(new_doc_ids))].tolist())
    old_top2 = set(sorted_doc_ids[:min(2, len(sorted_doc_ids))].tolist())
    old_top5 = set(sorted_doc_ids[:min(5, len(sorted_doc_ids))].tolist())

    info.update({
        "applied": True,
        "reason": "applied",
        "candidate_count": top_n,
        "promoted_to_top2": len(new_top2 - old_top2),
        "promoted_to_top5": len(new_top5 - old_top5),
        "promotions": [
            {
                "doc_id": item["doc_id"],
                "from_rank": item["rank"],
                "title": item["title"],
                "support_score": round(item["support_score"], 6),
                "final_score": round(item["final_score"], 6),
                "query_overlap": round(item["query_overlap"], 6),
                "title_anchor": round(item["title_anchor"], 6),
                "sentence_score": round(item["sentence_score"], 6),
                "target_signal": round(item["target_signal"], 6),
            }
            for item in promoted[:8]
        ],
    })
    return new_doc_ids, np.array(new_scores, dtype=float), info
