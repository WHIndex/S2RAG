import re
from typing import Any, Dict, List, Tuple

from .evidence_utils import has_bridge_cues
from .planned_retrieval_state_utils import planned_retrieval_terminal_state
from .sentence_evidence_utils import (
    _doc_body,
    _doc_title,
    _overlap_ratio,
    _split_sentences,
    _target_signal,
    _token_set,
    infer_answer_target_type,
)


LONG_CONTEXT_REASONS = {
    "routed_long_answer_or_narrative",
    "routed_long_context_evidence",
}

LONG_CONTEXT_QUESTION_TYPES = {
    "long_answer_or_narrative",
    "long_context_evidence",
    "long_context_factoid",
}

LONG_CONTEXT_PROFILES = {
    "long_answer",
    "long_context",
    "long_context_evidence_only",
}


def _doc_key(doc: Any) -> str:
    return re.sub(r"\s+", " ", str(doc or "")).strip().lower()


def _sentence_key(sentence: Any) -> str:
    return re.sub(r"\W+", " ", str(sentence or "").lower()).strip()


def _source_docs_with_candidates(
    current_docs: List[str],
    ranked_docs: List[str],
    candidate_top_k: int,
) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen = set()
    for idx, doc in enumerate(current_docs or []):
        key = _doc_key(doc)
        if not key or key in seen:
            continue
        seen.add(key)
        sources.append({"doc": str(doc), "source": "current", "rank": idx})

    for idx, doc in enumerate((ranked_docs or [])[: max(0, int(candidate_top_k))]):
        key = _doc_key(doc)
        if not key or key in seen:
            continue
        seen.add(key)
        sources.append({"doc": str(doc), "source": "candidate", "rank": idx})
    return sources


def _score_chain_sentences(
    question: str,
    current_docs: List[str],
    ranked_docs: List[str],
    candidate_top_k: int,
    min_sentence_score: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    question_tokens = _token_set(question)
    target_type = infer_answer_target_type(question)
    current_text = " ".join(str(doc or "")[:2200] for doc in current_docs or [])
    current_tokens = _token_set(current_text)
    missing_tokens = question_tokens - current_tokens
    if not missing_tokens:
        missing_tokens = question_tokens

    current_title_tokens = set()
    for doc in current_docs or []:
        current_title_tokens.update(_token_set(_doc_title(doc)))

    sources = _source_docs_with_candidates(current_docs, ranked_docs, candidate_top_k)
    scored: List[Dict[str, Any]] = []
    min_sentence_score = max(0.0, float(min_sentence_score))

    for source in sources:
        doc = source["doc"]
        title = _doc_title(doc)
        body = _doc_body(doc)
        title_tokens = _token_set(title)
        rank_prior = 1.0 / (int(source["rank"]) + 1)

        for sentence_index, sentence in enumerate(_split_sentences(body, max_body_chars=6000)):
            sentence_tokens = _token_set(sentence)
            if not sentence_tokens:
                continue
            support_tokens = sentence_tokens | title_tokens
            query_overlap = _overlap_ratio(question_tokens, support_tokens)
            missing_gain = _overlap_ratio(missing_tokens, support_tokens)
            title_anchor = max(
                _overlap_ratio(question_tokens, title_tokens),
                _overlap_ratio(title_tokens, question_tokens),
            )
            chain_link = max(
                _overlap_ratio(title_tokens, current_title_tokens),
                _overlap_ratio(current_title_tokens, sentence_tokens),
            )
            target = _target_signal(sentence, title, target_type)
            early_sentence = 1.0 / (sentence_index + 1)
            external_bonus = 0.05 if source["source"] == "candidate" and missing_gain >= 0.08 else 0.0
            score = (
                0.28 * query_overlap
                + 0.20 * missing_gain
                + 0.20 * target
                + 0.14 * title_anchor
                + 0.10 * chain_link
                + 0.05 * rank_prior
                + 0.03 * early_sentence
                + external_bonus
            )
            if score < min_sentence_score:
                continue
            roles = []
            if query_overlap >= 0.16 or title_anchor >= 0.28:
                roles.append("anchor")
            if chain_link >= 0.14 and (missing_gain >= 0.06 or query_overlap >= 0.12):
                roles.append("bridge")
            if target >= 0.45 or (target_type == "factoid" and target >= 0.35):
                roles.append("answer")
            if source["source"] == "candidate" and missing_gain >= 0.08:
                roles.append("missing_slot")
            if not roles:
                continue
            scored.append({
                "title": title,
                "sentence": sentence,
                "source": source["source"],
                "rank": int(source["rank"]),
                "sentence_index": int(sentence_index),
                "score": float(score),
                "query_overlap": float(query_overlap),
                "missing_gain": float(missing_gain),
                "title_anchor": float(title_anchor),
                "chain_link": float(chain_link),
                "target_signal": float(target),
                "roles": roles,
            })

    scored.sort(
        key=lambda item: (
            -float(item["score"]),
            0 if item["source"] == "current" else 1,
            int(item["rank"]),
            int(item["sentence_index"]),
        )
    )
    state = {
        "target_type": target_type,
        "question_token_count": len(question_tokens),
        "missing_token_count": len(missing_tokens),
        "source_doc_count": len(sources),
    }
    return scored, state


def _best_for_role(
    scored: List[Dict[str, Any]],
    role: str,
    used_sentence_keys: set,
    prefer_external: bool = False,
) -> Dict[str, Any]:
    candidates = [item for item in scored if role in item.get("roles", [])]
    if prefer_external:
        candidates.sort(
            key=lambda item: (
                0 if item.get("source") == "candidate" else 1,
                -float(item.get("score", 0.0)),
                int(item.get("rank", 0)),
            )
        )
    for item in candidates:
        key = _sentence_key(item.get("sentence", ""))
        if key and key not in used_sentence_keys:
            return item
    return {}


def _format_chain_doc(target_type: str, selected: List[Dict[str, Any]]) -> str:
    role_order = [
        ("anchor", "Anchor evidence"),
        ("bridge", "Bridge evidence"),
        ("missing_slot", "Missing-slot support"),
        ("answer", "Answer evidence"),
    ]
    lines = [
        "Structured Evidence Chain",
        f"Target answer type: {target_type}",
        "Evidence chain:",
    ]
    emitted = set()
    for role, label in role_order:
        for item in selected:
            key = _sentence_key(item.get("sentence", ""))
            if key in emitted or role not in item.get("roles", []):
                continue
            emitted.add(key)
            lines.append(f"- {label} [{item['title']}]: {item['sentence']}")
            break
    for item in selected:
        key = _sentence_key(item.get("sentence", ""))
        if key in emitted:
            continue
        emitted.add(key)
        lines.append(f"- Supporting evidence [{item['title']}]: {item['sentence']}")
    return "\n".join(lines)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _planned_route_state(planned_retrieval: Dict[str, Any]) -> Dict[str, Any]:
    router = planned_retrieval.get("router") or {}
    route_reason = str(planned_retrieval.get("reason", "") or "")
    router_reason = str(router.get("reason", "") or "")
    question_type = str(
        router.get("question_type", planned_retrieval.get("question_type", "")) or ""
    )
    routing_profile = str(
        router.get("routing_profile", planned_retrieval.get("routing_profile", "")) or ""
    )
    bridge_strength = _as_int(
        router.get("bridge_strength", planned_retrieval.get("bridge_strength", 0)),
        default=0,
    )
    terminal_state = planned_retrieval_terminal_state(planned_retrieval)
    evidence_gap_state = terminal_state["terminal_evidence_gap"]
    long_context_state = (
        _as_bool(router.get("long_context_shape", False))
        or route_reason in LONG_CONTEXT_REASONS
        or router_reason in LONG_CONTEXT_REASONS
        or question_type in LONG_CONTEXT_QUESTION_TYPES
        or routing_profile in LONG_CONTEXT_PROFILES
    )
    return {
        "route_reason": route_reason,
        "router_reason": router_reason,
        "question_type": question_type,
        "routing_profile": routing_profile,
        "bridge_strength": bridge_strength,
        "evidence_gap_state": bool(evidence_gap_state),
        "expansion_stopped_by_optimizer": terminal_state[
            "expansion_stopped_by_optimizer"
        ],
        "terminal_materialization_required": terminal_state[
            "terminal_materialization_required"
        ],
        "long_context_state": bool(long_context_state),
    }


def _planned_top2_changed(planned_retrieval: Dict[str, Any]) -> bool:
    for key in (
        "promoted_to_top2",
        "top2_override_applied",
        "support_set_rerank_promoted_to_top2",
    ):
        if _as_int(planned_retrieval.get(key, 0), default=0) > 0:
            return True
    for round_info in planned_retrieval.get("rounds") or []:
        if _as_int((round_info or {}).get("promoted_to_top2", 0), default=0) > 0:
            return True
    merge_entries = planned_retrieval.get("merge_history") or planned_retrieval.get("merge_info") or []
    if isinstance(merge_entries, dict):
        merge_entries = [merge_entries]
    for merge_info in merge_entries:
        if not isinstance(merge_info, dict):
            continue
        if _as_int(merge_info.get("promoted_to_top2", 0), default=0) > 0:
            return True
        if _as_int(merge_info.get("top2_override_applied", 0), default=0) > 0:
            return True
    return False


def build_structured_evidence_chain(
    question: str,
    docs: List[str],
    ranked_docs: List[str] = None,
    planned_retrieval: Dict[str, Any] = None,
    enabled: bool = False,
    bridge_only: bool = True,
    candidate_top_k: int = 24,
    max_sentences: int = 4,
    max_candidate_sentences: int = 1,
    min_sentence_score: float = 0.27,
    min_chain_score: float = 0.52,
    insert_after: int = 2,
    skip_single_hop: bool = True,
    require_evidence_gap: bool = True,
    allow_long_context: bool = False,
    min_bridge_strength: int = 3,
    skip_if_top2_changed: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    """Insert a compact, role-ordered support chain for the reader.

    Unlike support-set completion, this module does not append full passages.
    It composes a synthetic evidence chain from the current QA evidence and at
    most a few high-scoring candidate sentences, preserving retrieval ranks
    while giving the reader an explicit anchor/bridge/answer structure.
    """
    docs = list(docs or [])
    planned_retrieval = planned_retrieval or {}
    route_state = _planned_route_state(planned_retrieval)
    routed_bridge_cues = bool(
        has_bridge_cues(question)
        or route_state["question_type"] == "bridge_or_compositional"
        or route_state["bridge_strength"] >= 2
    )
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "bridge_cues": routed_bridge_cues,
        "candidate_top_k": int(candidate_top_k),
        "max_sentences": int(max_sentences),
        "max_candidate_sentences": int(max_candidate_sentences),
        "min_sentence_score": float(min_sentence_score),
        "min_chain_score": float(min_chain_score),
        "insert_after": int(insert_after),
        "require_evidence_gap": bool(require_evidence_gap),
        "allow_long_context": bool(allow_long_context),
        "min_bridge_strength": int(min_bridge_strength),
        "skip_if_top2_changed": bool(skip_if_top2_changed),
        "original_doc_count": len(docs),
        "final_doc_count": len(docs),
        "selected_sentence_count": 0,
        "candidate_sentence_count": 0,
        "used_external_candidate": False,
        "roles_present": [],
        "selected_titles": [],
        "chain_score": 0.0,
        "route_reason": "",
        "router_reason": "",
        "question_type": "",
        "routing_profile": "",
        "bridge_strength": 0,
        "evidence_gap_state": False,
        "long_context_state": False,
        "top2_changed_state": False,
    }
    info.update(route_state)
    if not enabled:
        return docs, info
    if not docs:
        info["reason"] = "no_docs"
        return docs, info
    if bridge_only and not info["bridge_cues"]:
        info["reason"] = "no_bridge_cues"
        return docs, info

    top2_changed_state = _planned_top2_changed(planned_retrieval)
    info["top2_changed_state"] = bool(top2_changed_state)
    route_reason = route_state["route_reason"]
    route_question_type = route_state["question_type"]
    if skip_if_top2_changed and top2_changed_state:
        info["reason"] = "retrieval_top2_already_changed"
        return docs, info
    if skip_single_hop and (
        route_reason == "routed_single_hop_attribute"
        or route_question_type == "single_hop_attribute"
    ):
        info["reason"] = "single_hop_route"
        return docs, info
    if not allow_long_context and route_state["long_context_state"]:
        info["reason"] = "long_context_or_narrative_route"
        return docs, info
    if require_evidence_gap and not route_state["evidence_gap_state"]:
        info["reason"] = "no_evidence_gap_state"
        return docs, info
    min_bridge_strength = max(0, int(min_bridge_strength))
    if (
        bridge_only
        and min_bridge_strength > 0
        and route_question_type == "bridge_or_compositional"
        and route_state["bridge_strength"] < min_bridge_strength
    ):
        info["reason"] = "weak_bridge_strength"
        return docs, info

    scored, state = _score_chain_sentences(
        question=question,
        current_docs=docs,
        ranked_docs=ranked_docs or [],
        candidate_top_k=candidate_top_k,
        min_sentence_score=min_sentence_score,
    )
    info.update(state)
    if not scored:
        info["reason"] = "no_scored_sentences"
        return docs, info

    selected: List[Dict[str, Any]] = []
    used_keys = set()
    external_count = 0
    for role, prefer_external in (
        ("anchor", False),
        ("bridge", False),
        ("missing_slot", True),
        ("answer", False),
    ):
        if len(selected) >= max(1, int(max_sentences)):
            break
        item = _best_for_role(scored, role, used_keys, prefer_external=prefer_external)
        if not item:
            continue
        if item.get("source") == "candidate":
            if external_count >= max(0, int(max_candidate_sentences)):
                continue
            external_count += 1
        key = _sentence_key(item.get("sentence", ""))
        if key:
            used_keys.add(key)
        selected.append(item)

    if len(selected) < 2:
        info["reason"] = "insufficient_chain_roles"
        return docs, info

    roles_present = sorted({role for item in selected for role in item.get("roles", [])})
    if "answer" not in roles_present:
        info["reason"] = "missing_answer_role"
        info["roles_present"] = roles_present
        return docs, info
    if info["bridge_cues"] and "bridge" not in roles_present and "missing_slot" not in roles_present:
        info["reason"] = "missing_bridge_role"
        info["roles_present"] = roles_present
        return docs, info

    best_by_role = {
        role: max(
            [float(item.get("score", 0.0)) for item in selected if role in item.get("roles", [])],
            default=0.0,
        )
        for role in ("anchor", "bridge", "missing_slot", "answer")
    }
    avg_score = sum(float(item.get("score", 0.0)) for item in selected) / max(1, len(selected))
    chain_score = (
        0.25 * best_by_role["anchor"]
        + 0.25 * max(best_by_role["bridge"], best_by_role["missing_slot"])
        + 0.30 * best_by_role["answer"]
        + 0.20 * avg_score
    )
    info["chain_score"] = round(float(chain_score), 6)
    if chain_score < float(min_chain_score):
        info["reason"] = "low_chain_score"
        info["roles_present"] = roles_present
        return docs, info

    chain_doc = _format_chain_doc(state["target_type"], selected)
    insert_after = min(max(0, int(insert_after)), len(docs))
    final_docs = docs[:insert_after] + [chain_doc] + docs[insert_after:]

    selected_titles = []
    seen_titles = set()
    for item in selected:
        title = item.get("title", "")
        if title and title not in seen_titles:
            selected_titles.append(title)
            seen_titles.add(title)

    info.update({
        "applied": True,
        "reason": "structured_chain_inserted",
        "final_doc_count": len(final_docs),
        "selected_sentence_count": len(selected),
        "candidate_sentence_count": external_count,
        "used_external_candidate": external_count > 0,
        "roles_present": roles_present,
        "selected_titles": selected_titles,
        "top_chain_items": [
            {
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "rank": int(item.get("rank", 0)),
                "roles": item.get("roles", []),
                "score": round(float(item.get("score", 0.0)), 6),
                "missing_gain": round(float(item.get("missing_gain", 0.0)), 6),
                "target_signal": round(float(item.get("target_signal", 0.0)), 6),
                "sentence_preview": str(item.get("sentence", ""))[:240],
            }
            for item in selected
        ],
    })
    return final_docs, info
