import re
from typing import Any, Dict, List, Tuple

from .evidence_utils import has_bridge_cues
from .support_rerank_utils import (
    _best_sentence_score,
    _doc_body,
    _doc_title,
    _overlap_ratio,
    _target_signal,
    _token_set,
)
from .sentence_evidence_utils import infer_answer_target_type


def _doc_key(doc: Any) -> str:
    return re.sub(r"\s+", " ", str(doc or "")).strip().lower()


def _rank_support_completion_candidates(
    question: str,
    current_docs: List[str],
    candidate_docs: List[str],
    candidate_top_k: int,
) -> List[Dict[str, Any]]:
    question_tokens = _token_set(question)
    if not question_tokens:
        return []

    target_type = infer_answer_target_type(question)
    current_text = " ".join(str(doc or "")[:2500] for doc in current_docs)
    current_tokens = _token_set(current_text)
    missing_tokens = question_tokens - current_tokens
    if not missing_tokens:
        missing_tokens = question_tokens

    current_title_tokens = set()
    current_doc_keys = {_doc_key(doc) for doc in current_docs}
    current_titles = {_doc_title(doc).strip().lower() for doc in current_docs if _doc_title(doc)}
    for doc in current_docs:
        current_title_tokens.update(_token_set(_doc_title(doc)))

    ranked: List[Dict[str, Any]] = []
    for rank, doc in enumerate(candidate_docs[: max(0, int(candidate_top_k))]):
        doc_key = _doc_key(doc)
        if not doc_key or doc_key in current_doc_keys:
            continue
        title = _doc_title(doc)
        if title.strip().lower() in current_titles:
            continue
        body = _doc_body(doc)
        doc_tokens = _token_set(f"{title} {body[:3500]}")
        title_tokens = _token_set(title)
        if not doc_tokens:
            continue

        query_overlap = _overlap_ratio(question_tokens, doc_tokens)
        missing_gain = _overlap_ratio(missing_tokens, doc_tokens)
        title_anchor = max(
            _overlap_ratio(question_tokens, title_tokens),
            _overlap_ratio(title_tokens, question_tokens),
        )
        sentence_score = _best_sentence_score(question_tokens, title, body, target_type)
        target_signal = _target_signal(body[:1500], title, target_type)
        chain_overlap = _overlap_ratio(title_tokens, current_title_tokens)
        rank_prior = 1.0 / (rank + 1)

        score = (
            0.30 * missing_gain
            + 0.22 * sentence_score
            + 0.18 * query_overlap
            + 0.12 * target_signal
            + 0.10 * title_anchor
            + 0.05 * chain_overlap
            + 0.03 * rank_prior
        )
        ranked.append({
            "rank": rank,
            "title": title,
            "doc": doc,
            "score": score,
            "missing_gain": missing_gain,
            "query_overlap": query_overlap,
            "title_anchor": title_anchor,
            "sentence_score": sentence_score,
            "target_signal": target_signal,
            "chain_overlap": chain_overlap,
        })

    ranked.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            -float(item.get("missing_gain", 0.0)),
            -float(item.get("sentence_score", 0.0)),
            int(item.get("rank", 0)),
        )
    )
    return ranked


def append_support_set_completion_evidence(
    question: str,
    current_docs: List[str],
    ranked_docs: List[str],
    planned_retrieval: Dict[str, Any] = None,
    enabled: bool = False,
    bridge_only: bool = True,
    candidate_top_k: int = 20,
    max_append: int = 1,
    min_score: float = 0.54,
    min_missing_gain: float = 0.12,
    min_sentence_score: float = 0.30,
) -> Tuple[List[str], Dict[str, Any]]:
    """Append missing-slot support evidence for the reader without reranking retrieval.

    The module is intentionally reader-only: it never changes the retrieved
    ranking used for Recall@k. It targets cases where top-20 evidence already
    contains a likely bridge/support passage but top-5 context is incomplete.
    """
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "bridge_cues": bool(has_bridge_cues(question)),
        "candidate_top_k": int(candidate_top_k),
        "max_append": int(max_append),
        "min_score": float(min_score),
        "min_missing_gain": float(min_missing_gain),
        "min_sentence_score": float(min_sentence_score),
        "candidate_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "accepted_titles": [],
        "top_candidates": [],
    }
    current_docs = list(current_docs or [])
    ranked_docs = list(ranked_docs or [])
    if not enabled:
        return current_docs, info
    if bridge_only and not info["bridge_cues"]:
        info["reason"] = "no_bridge_cues"
        return current_docs, info
    if not ranked_docs:
        info["reason"] = "no_ranked_docs"
        return current_docs, info
    if max_append <= 0:
        info["reason"] = "max_append_zero"
        return current_docs, info

    planned_retrieval = planned_retrieval or {}
    route_reason = str(planned_retrieval.get("reason", "") or "")
    route_question_type = str(planned_retrieval.get("question_type", "") or "")
    if route_reason == "routed_single_hop_attribute" or route_question_type == "single_hop_attribute":
        info["reason"] = "single_hop_route"
        return current_docs, info

    candidates = _rank_support_completion_candidates(
        question=question,
        current_docs=current_docs,
        candidate_docs=ranked_docs,
        candidate_top_k=candidate_top_k,
    )
    info["candidate_count"] = len(candidates)
    info["top_candidates"] = [
        {
            "rank": int(item["rank"]),
            "title": item["title"],
            "score": round(float(item["score"]), 6),
            "missing_gain": round(float(item["missing_gain"]), 6),
            "query_overlap": round(float(item["query_overlap"]), 6),
            "sentence_score": round(float(item["sentence_score"]), 6),
            "target_signal": round(float(item["target_signal"]), 6),
        }
        for item in candidates[:5]
    ]

    appended: List[str] = []
    reject_reasons: Dict[str, int] = {}
    for item in candidates:
        if len(appended) >= max_append:
            break
        reasons = []
        if float(item["score"]) < float(min_score):
            reasons.append("low_score")
        if float(item["missing_gain"]) < float(min_missing_gain):
            reasons.append("low_missing_gain")
        if float(item["sentence_score"]) < float(min_sentence_score):
            reasons.append("low_sentence_score")
        if reasons:
            for reason in reasons:
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            continue
        appended.append(str(item["doc"]))

    info["rejected_count"] = sum(reject_reasons.values())
    info["reject_reason_counts"] = reject_reasons
    if not appended:
        info["reason"] = "no_candidate_accepted" if candidates else "no_candidates"
        return current_docs, info

    info["applied"] = True
    info["reason"] = "appended_support_completion"
    info["accepted_count"] = len(appended)
    info["accepted_titles"] = [_doc_title(doc) for doc in appended]
    return current_docs + appended, info
