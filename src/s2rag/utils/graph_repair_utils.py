import math
import re
from typing import Any, Dict, List, Optional, Tuple

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
LOCATION_RE = re.compile(
    r"\b(?:city|town|village|county|state|province|country|region|capital|river|"
    r"island|municipality|republic|kingdom)\b",
    re.IGNORECASE,
)
ORG_RE = re.compile(
    r"\b(?:university|college|school|company|corporation|club|team|association|"
    r"organization|organisation|agency|committee|foundation|league|group)\b",
    re.IGNORECASE,
)
WORK_RE = re.compile(
    r"\b(?:film|movie|album|song|book|novel|poem|play|series|episode|opera|game)\b",
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


def _overlap(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _safe_score(scores: Optional[np.ndarray], rank: int) -> float:
    if scores is None:
        return 0.0
    try:
        return _safe_float(scores[rank])
    except Exception:
        return 0.0


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v <= min_v:
        return [1.0 for _ in values]
    return [(value - min_v) / (max_v - min_v) for value in values]


def _target_signal(doc: str, target_type: str) -> float:
    title = _doc_title(doc)
    body = _doc_body(doc)[:3000]
    text = f"{title} {body}"
    if target_type == "date":
        return 1.0 if YEAR_RE.search(text) else 0.0
    if target_type == "number":
        return 1.0 if NUMBER_RE.search(text) else 0.0
    if target_type == "location":
        if LOCATION_RE.search(text):
            return 1.0
        return 0.45 if TITLE_CASE_RE.search(text) else 0.0
    if target_type == "person":
        return 0.85 if TITLE_CASE_RE.search(text) else 0.0
    if target_type == "organization":
        if ORG_RE.search(text):
            return 1.0
        return 0.45 if TITLE_CASE_RE.search(text) else 0.0
    if target_type == "work":
        if WORK_RE.search(text) or '"' in text or "'" in text:
            return 1.0
        return 0.45 if TITLE_CASE_RE.search(text) else 0.0
    return 0.35 if (TITLE_CASE_RE.search(text) or YEAR_RE.search(text) or NUMBER_RE.search(text)) else 0.0


def graph_repair_trigger_reason(
    fact_rerank_info: Optional[Dict[str, Any]],
    min_top_facts: int = 1,
    min_candidate_facts: int = 2,
    trigger_on_dense_fallback: bool = True,
) -> str:
    fact_rerank_info = fact_rerank_info or {}
    if trigger_on_dense_fallback and bool(fact_rerank_info.get("fallback_to_dense", False)):
        return "dense_fallback"

    top_fact_count = int(fact_rerank_info.get("top_fact_count", 0) or 0)
    if top_fact_count < max(0, int(min_top_facts)):
        return "insufficient_top_facts"

    candidate_fact_count = int(fact_rerank_info.get("candidate_fact_count", 0) or 0)
    if candidate_fact_count < max(0, int(min_candidate_facts)):
        return "insufficient_candidate_facts"

    if fact_rerank_info.get("error"):
        return "fact_rerank_error"

    return "graph_fact_signal_sufficient"


def graph_repair_retrieve(
    question: str,
    base_doc_ids: np.ndarray,
    base_doc_scores: np.ndarray,
    base_docs: List[str],
    dense_doc_ids: np.ndarray,
    dense_doc_scores: np.ndarray,
    dense_docs: List[str],
    fact_rerank_info: Optional[Dict[str, Any]],
    qa_top_k: int = 5,
    enabled: bool = False,
    dense_top_k: int = 80,
    max_promotions: int = 2,
    protect_top_k: int = 1,
    min_score: float = 0.22,
    min_top_facts: int = 1,
    min_candidate_facts: int = 2,
    trigger_on_dense_fallback: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Repair graph retrieval when the fact channel is too sparse.

    The module is intentionally non-destructive: it preserves the leading
    retrieval prefix and can promote only a small number of dense candidates
    whose text also matches query, anchor, and target-type signals.
    """
    base_ids = list(base_doc_ids.tolist() if hasattr(base_doc_ids, "tolist") else base_doc_ids)
    base_scores = list(base_doc_scores.tolist() if hasattr(base_doc_scores, "tolist") else base_doc_scores)
    dense_ids = list(dense_doc_ids.tolist() if hasattr(dense_doc_ids, "tolist") else dense_doc_ids)
    dense_scores = list(dense_doc_scores.tolist() if hasattr(dense_doc_scores, "tolist") else dense_doc_scores)
    dense_top_k = max(1, int(dense_top_k))
    max_promotions = max(0, int(max_promotions))
    protect_top_k = max(0, int(protect_top_k))
    qa_top_k = max(1, int(qa_top_k))

    trigger_reason = graph_repair_trigger_reason(
        fact_rerank_info=fact_rerank_info,
        min_top_facts=min_top_facts,
        min_candidate_facts=min_candidate_facts,
        trigger_on_dense_fallback=trigger_on_dense_fallback,
    )
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "trigger_reason": trigger_reason,
        "dense_top_k": dense_top_k,
        "max_promotions": max_promotions,
        "protect_top_k": protect_top_k,
        "min_score": min_score,
        "target_type": infer_answer_target_type(question),
        "candidate_count": 0,
        "promoted_to_top2": 0,
        "promoted_to_top5": 0,
        "promotions": [],
    }

    if not enabled:
        return np.array(base_ids), np.array(base_scores), info
    if trigger_reason == "graph_fact_signal_sufficient":
        info["reason"] = trigger_reason
        return np.array(base_ids), np.array(base_scores), info
    if max_promotions <= 0:
        info["reason"] = "no_promotion_budget"
        return np.array(base_ids), np.array(base_scores), info
    if not base_ids or not dense_ids:
        info["reason"] = "no_candidates"
        return np.array(base_ids), np.array(base_scores), info

    question_tokens = _token_set(question)
    protected_ids = set(base_ids[:protect_top_k])
    base_rank = {doc_id: rank for rank, doc_id in enumerate(base_ids)}
    base_score = {doc_id: _safe_score(np.array(base_scores), rank) for rank, doc_id in enumerate(base_ids)}
    dense_score = {doc_id: _safe_score(np.array(dense_scores), rank) for rank, doc_id in enumerate(dense_ids)}

    protected_anchor_tokens = set()
    top_anchor_tokens = set()
    top_docs = base_docs[:max(qa_top_k, protect_top_k)]
    for doc in top_docs:
        title_tokens = _token_set(_doc_title(doc))
        protected_anchor_tokens.update(title_tokens)
        top_anchor_tokens.update(title_tokens)
        top_anchor_tokens.update(_token_set(_doc_body(doc)[:1000]))

    raw_scores = [_safe_float(score) for score in dense_scores[:dense_top_k]]
    norm_dense_scores = _normalize(raw_scores)
    seen_titles = {_doc_title(doc).lower() for doc in base_docs[:qa_top_k]}
    candidates = []
    for dense_rank, doc_id in enumerate(dense_ids[:dense_top_k]):
        if doc_id in protected_ids:
            continue
        doc = dense_docs[dense_rank] if dense_rank < len(dense_docs) else ""
        title = _doc_title(doc)
        if not title:
            continue
        title_key = title.lower()
        already_in_top = doc_id in set(base_ids[:qa_top_k]) or title_key in seen_titles
        if already_in_top:
            continue

        title_tokens = _token_set(title)
        body_tokens = _token_set(_doc_body(doc)[:3500])
        all_tokens = title_tokens | body_tokens
        dense_norm = norm_dense_scores[dense_rank] if dense_rank < len(norm_dense_scores) else 0.0
        query_overlap = _overlap(question_tokens, all_tokens)
        title_query_overlap = _overlap(question_tokens, title_tokens)
        anchor_overlap = _overlap((protected_anchor_tokens | question_tokens), all_tokens | title_tokens)
        top_context_overlap = _overlap(title_tokens | question_tokens, top_anchor_tokens)
        target_signal = _target_signal(doc, info["target_type"])
        rank_prior = 1.0 / (dense_rank + 1)
        score = (
            0.36 * dense_norm
            + 0.20 * query_overlap
            + 0.14 * title_query_overlap
            + 0.14 * anchor_overlap
            + 0.08 * top_context_overlap
            + 0.06 * target_signal
            + 0.02 * rank_prior
        )
        if score < float(min_score):
            continue
        candidates.append({
            "doc_id": doc_id,
            "dense_rank": dense_rank,
            "original_rank": base_rank.get(doc_id),
            "title": title,
            "score": score,
            "dense_score": dense_score.get(doc_id, 0.0),
            "query_overlap": query_overlap,
            "anchor_overlap": anchor_overlap,
            "target_signal": target_signal,
        })

    candidates.sort(
        key=lambda item: (
            -item["score"],
            item["original_rank"] if item["original_rank"] is not None else 10**9,
            item["dense_rank"],
        )
    )
    promotions = candidates[:max_promotions]
    info["candidate_count"] = len(candidates)

    if not promotions:
        info["reason"] = "no_repair_candidate_above_threshold"
        return np.array(base_ids), np.array(base_scores), info

    promoted_ids = [item["doc_id"] for item in promotions]
    promoted_set = set(promoted_ids)
    repaired_ids = base_ids[:protect_top_k] + promoted_ids + [
        doc_id for doc_id in base_ids[protect_top_k:] if doc_id not in promoted_set
    ]

    repaired_scores = []
    for rank, doc_id in enumerate(repaired_ids):
        if doc_id in promoted_set:
            repair_item = next(item for item in promotions if item["doc_id"] == doc_id)
            score = max(base_score.get(doc_id, 0.0), repair_item["score"], repair_item["dense_score"])
        else:
            score = base_score.get(doc_id, 0.0)
        repaired_scores.append(score)

    promotion_logs = []
    for item in promotions:
        new_rank = repaired_ids.index(item["doc_id"])
        original_rank = item["original_rank"]
        if new_rank < 2 and (original_rank is None or original_rank >= 2):
            info["promoted_to_top2"] += 1
        if new_rank < 5 and (original_rank is None or original_rank >= 5):
            info["promoted_to_top5"] += 1
        promotion_logs.append({
            "doc_id": int(item["doc_id"]) if isinstance(item["doc_id"], (int, np.integer)) else item["doc_id"],
            "title": item["title"],
            "dense_rank": int(item["dense_rank"]),
            "original_rank": None if original_rank is None else int(original_rank),
            "new_rank": int(new_rank),
            "score": round(item["score"], 6),
            "dense_score": round(item["dense_score"], 6),
            "query_overlap": round(item["query_overlap"], 6),
            "anchor_overlap": round(item["anchor_overlap"], 6),
            "target_signal": round(item["target_signal"], 6),
        })

    info.update({
        "applied": True,
        "reason": "repaired",
        "promotions": promotion_logs,
    })
    return np.array(repaired_ids), np.array(repaired_scores), info
