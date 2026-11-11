import math
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .evidence_utils import has_bridge_cues
from .relation_fact_join_utils import _relation_profile, _tokens


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: Any) -> float:
    return round(_as_float(value), 6)


def _doc_title(doc: Any) -> str:
    return str(doc or "").split("\n", 1)[0].strip()


def _doc_key(doc: Any) -> str:
    return re.sub(r"\s+", " ", str(doc or "")).strip().lower()


def _safe_doc_content(get_doc_content: Callable[[int], str], doc_id: int) -> str:
    try:
        return str(get_doc_content(int(doc_id)) or "")
    except Exception:
        return ""


def _estimate_tokens(text: Any) -> int:
    return max(1, len(re.findall(r"\S+", str(text or ""))))


def _overlap(left: Set[str], right: Set[str], max_required: int = 6) -> float:
    if not left or not right:
        return 0.0
    return min(1.0, len(left & right) / max(1, min(len(left), int(max_required))))


def _count_bucket(value: Any) -> str:
    count = _as_int(value, 0)
    if count <= 0:
        return "0"
    if count <= 2:
        return "1-2"
    if count <= 5:
        return "3-5"
    if count <= 10:
        return "6-10"
    if count <= 50:
        return "11-50"
    if count <= 100:
        return "51-100"
    if count <= 200:
        return "101-200"
    return "201+"


def _current_evidence_state(
    *,
    question: str,
    current_docs: Sequence[str],
    relation_fact_join: Optional[Dict[str, Any]],
    min_query_coverage: float,
    min_relation_coverage: float,
) -> Dict[str, Any]:
    relation_fact_join = relation_fact_join or {}
    profile = _relation_profile(question)
    question_tokens = set(_tokens(question))
    query_relation_tokens = set(profile.get("relation_tokens") or [])
    current_text = " ".join(str(doc or "")[:2500] for doc in current_docs)
    current_tokens = set(_tokens(current_text))
    missing_tokens = question_tokens - current_tokens
    query_coverage = _overlap(question_tokens, current_tokens, max_required=10)
    relation_coverage = (
        _overlap(query_relation_tokens, current_tokens, max_required=4)
        if query_relation_tokens else 1.0
    )
    explicit_gap = any(
        bool(relation_fact_join.get(key))
        for key in (
            "evidence_gap",
            "answerability_gap",
            "relation_gap",
            "bridge_gap",
            "type_gap",
            "polarity_gap",
        )
    )
    current_reason = str(relation_fact_join.get("reason", "") or "")
    optimistic_sufficient = current_reason == "current_evidence_sufficient"
    deficit = (
        explicit_gap
        or query_coverage < float(min_query_coverage)
        or relation_coverage < float(min_relation_coverage)
        or len(missing_tokens) >= 4
    )
    if optimistic_sufficient and not deficit:
        deficit_reason = "current_evidence_sufficient"
    elif explicit_gap:
        deficit_reason = "relation_join_gap"
    elif relation_coverage < float(min_relation_coverage):
        deficit_reason = "relation_coverage_gap"
    elif query_coverage < float(min_query_coverage):
        deficit_reason = "query_coverage_gap"
    elif len(missing_tokens) >= 4:
        deficit_reason = "missing_query_terms"
    else:
        deficit_reason = "no_deficit"
    return {
        "query_coverage": _round(query_coverage),
        "relation_coverage": _round(relation_coverage),
        "missing_token_count": len(missing_tokens),
        "missing_tokens": sorted(missing_tokens)[:20],
        "query_relation_tokens": sorted(query_relation_tokens)[:20],
        "explicit_gap": bool(explicit_gap),
        "optimistic_sufficient": bool(optimistic_sufficient),
        "deficit": bool(deficit),
        "deficit_reason": deficit_reason,
    }


def select_selective_support_join_evidence(
    *,
    question: str,
    sorted_doc_ids: Sequence[Any],
    typed_fact_join_shadow: Optional[Dict[str, Any]],
    relation_fact_join: Optional[Dict[str, Any]],
    get_doc_content: Callable[[int], str],
    qa_top_k: int,
    enabled: bool = False,
    bridge_only: bool = True,
    max_append: int = 1,
    max_candidate_plans: int = 12,
    max_original_rank: int = 80,
    min_plan_utility: float = 0.48,
    min_plan_net_utility: float = 0.22,
    high_conf_net_utility: float = 0.42,
    max_plan_risk: float = 0.48,
    min_missing_gain: float = 0.04,
    min_relation_overlap: float = 0.08,
    min_endpoint_overlap: float = 0.22,
    min_schema_selectivity: float = 0.62,
    min_admission_score: float = 0.28,
    max_accepted_plan_count: int = 96,
    require_two_hop_for_bridge: bool = True,
    require_deficit: bool = True,
    min_current_query_coverage: float = 0.62,
    min_current_relation_coverage: float = 0.50,
    block_long_context: bool = True,
    long_context_avg_chars: int = 1400,
    token_budget: int = 900,
) -> Tuple[List[int], List[str], Dict[str, Any]]:
    """Select reader-only support evidence from typed fact-join plans.

    This is a physical evidence operator: shadow plans enumerate possible fact
    joins, while this selector admits only low-risk plans that fill a current
    support-set deficit. It never changes retrieval ranking.
    """
    typed_fact_join_shadow = typed_fact_join_shadow or {}
    sorted_doc_ids = [] if sorted_doc_ids is None else list(sorted_doc_ids)
    current_doc_ids = [int(doc_id) for doc_id in sorted_doc_ids[: max(1, int(qa_top_k))]]
    current_docs = [_safe_doc_content(get_doc_content, doc_id) for doc_id in current_doc_ids]
    bridge_cues = bool(has_bridge_cues(question))
    avg_doc_chars = (
        sum(len(str(doc or "")) for doc in current_docs) / max(1, len(current_docs))
        if current_docs else 0.0
    )
    state = _current_evidence_state(
        question=question,
        current_docs=current_docs,
        relation_fact_join=relation_fact_join,
        min_query_coverage=min_current_query_coverage,
        min_relation_coverage=min_current_relation_coverage,
    )
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "shadow_only": False,
        "rank_stable": True,
        "bridge_cues": bridge_cues,
        "qa_top_k": int(qa_top_k),
        "max_append": int(max_append),
        "max_candidate_plans": int(max_candidate_plans),
        "max_original_rank": int(max_original_rank),
        "candidate_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "selected_doc_ids": [],
        "selected_titles": [],
        "top_candidates": [],
        "reject_reason_counts": {},
        "state": state,
        "typed_shadow_reason": typed_fact_join_shadow.get("reason", "unavailable"),
        "typed_shadow_accepted_plan_bucket": _count_bucket(
            typed_fact_join_shadow.get("accepted_plan_count", 0)
        ),
        "policy": {
            "bridge_only": bool(bridge_only),
            "require_deficit": bool(require_deficit),
            "require_two_hop_for_bridge": bool(require_two_hop_for_bridge),
            "min_plan_utility": float(min_plan_utility),
            "min_plan_net_utility": float(min_plan_net_utility),
            "high_conf_net_utility": float(high_conf_net_utility),
            "max_plan_risk": float(max_plan_risk),
            "min_missing_gain": float(min_missing_gain),
            "min_relation_overlap": float(min_relation_overlap),
            "min_endpoint_overlap": float(min_endpoint_overlap),
            "min_schema_selectivity": float(min_schema_selectivity),
            "min_admission_score": float(min_admission_score),
            "max_accepted_plan_count": int(max_accepted_plan_count),
            "block_long_context": bool(block_long_context),
            "long_context_avg_chars": int(long_context_avg_chars),
            "token_budget": int(token_budget),
        },
    }
    if not enabled:
        return [], [], info
    if max_append <= 0:
        info["reason"] = "max_append_zero"
        return [], [], info
    if bridge_only and not bridge_cues:
        info["reason"] = "no_bridge_cues"
        return [], [], info
    if block_long_context and avg_doc_chars >= float(long_context_avg_chars) and not bridge_cues:
        info["reason"] = "long_context_blocked"
        info["avg_doc_chars"] = _round(avg_doc_chars)
        return [], [], info
    if require_deficit and not state.get("deficit"):
        info["reason"] = str(state.get("deficit_reason") or "no_deficit")
        return [], [], info
    plans = list(typed_fact_join_shadow.get("top_plans") or [])
    if not plans:
        info["reason"] = "no_shadow_plans"
        return [], [], info

    question_tokens = set(_tokens(question))
    current_tokens = set(_tokens(" ".join(current_docs)))
    missing_tokens = set(state.get("missing_tokens") or []) or (question_tokens - current_tokens)
    current_doc_keys = {_doc_key(doc) for doc in current_docs}
    current_titles = {_doc_title(doc).strip().lower() for doc in current_docs if _doc_title(doc)}
    rank_map = {}
    for rank, doc_id in enumerate(sorted_doc_ids, start=1):
        try:
            rank_map.setdefault(int(doc_id), rank)
        except Exception:
            continue

    accepted_plan_count = _as_int(typed_fact_join_shadow.get("accepted_plan_count"), 0)
    overgenerated = accepted_plan_count > int(max_accepted_plan_count)
    reject_counts: Counter = Counter()
    candidates: List[Dict[str, Any]] = []

    for plan_idx, plan in enumerate(plans[: max(1, int(max_candidate_plans))]):
        plan_type = str(plan.get("plan_type") or "")
        plan_net = _as_float(plan.get("net_utility"), 0.0)
        plan_utility = _as_float(plan.get("utility"), 0.0)
        plan_risk = _as_float(plan.get("risk"), 1.0)
        relation_overlap = _as_float(plan.get("relation_overlap"), 0.0)
        endpoint_overlap = _as_float(plan.get("endpoint_overlap"), 0.0)
        schema_selectivity = _as_float(plan.get("schema_selectivity"), 0.0)
        provenance_score = _as_float(plan.get("provenance_score"), 0.0)
        plan_doc_ids = []
        for raw_doc_id in plan.get("doc_ids") or []:
            doc_id = _as_int(raw_doc_id, -1)
            if doc_id >= 0 and doc_id not in plan_doc_ids:
                plan_doc_ids.append(doc_id)
        if not plan_doc_ids:
            reject_counts["plan_without_docs"] += 1
            continue
        if require_two_hop_for_bridge and bridge_cues and plan_type != "two_hop_fact_join":
            reject_counts["not_two_hop_bridge_plan"] += 1
            continue
        if plan_utility < float(min_plan_utility):
            reject_counts["low_plan_utility"] += 1
            continue
        if plan_net < float(min_plan_net_utility):
            reject_counts["low_plan_net_utility"] += 1
            continue
        if plan_risk > float(max_plan_risk):
            reject_counts["high_plan_risk"] += 1
            continue
        if relation_overlap < float(min_relation_overlap):
            reject_counts["low_relation_overlap"] += 1
            continue
        if endpoint_overlap < float(min_endpoint_overlap):
            reject_counts["low_endpoint_overlap"] += 1
            continue
        if schema_selectivity < float(min_schema_selectivity):
            reject_counts["low_schema_selectivity"] += 1
            continue
        if overgenerated and plan_net < float(high_conf_net_utility):
            reject_counts["overgenerated_low_confidence"] += 1
            continue

        for doc_id in plan_doc_ids:
            rank = int(rank_map.get(doc_id, 999999))
            if rank <= int(qa_top_k):
                reject_counts["already_in_reader_topk"] += 1
                continue
            if rank > int(max_original_rank):
                reject_counts["rank_too_deep"] += 1
                continue
            doc = _safe_doc_content(get_doc_content, doc_id)
            doc_key = _doc_key(doc)
            title = _doc_title(doc)
            if not doc_key:
                reject_counts["empty_doc"] += 1
                continue
            if doc_key in current_doc_keys or title.strip().lower() in current_titles:
                reject_counts["duplicate_doc"] += 1
                continue
            doc_tokens = set(_tokens(f"{title} {doc[:2500]}"))
            missing_gain = _overlap(missing_tokens, doc_tokens, max_required=6)
            query_overlap = _overlap(question_tokens, doc_tokens, max_required=10)
            token_cost = min(1.0, _estimate_tokens(doc) / max(1.0, float(token_budget)))
            rank_prior = 1.0 / (1.0 + math.log1p(max(1, rank)))
            support_gain = (
                0.34 * missing_gain
                + 0.20 * query_overlap
                + 0.18 * relation_overlap
                + 0.14 * endpoint_overlap
                + 0.08 * provenance_score
                + 0.06 * rank_prior
            )
            admission_score = (
                support_gain
                + 0.34 * plan_net
                + 0.16 * schema_selectivity
                - 0.18 * plan_risk
                - 0.08 * token_cost
                - (0.06 if overgenerated else 0.0)
            )
            reasons = []
            if missing_gain < float(min_missing_gain):
                reasons.append("low_missing_gain")
            if admission_score < float(min_admission_score):
                reasons.append("low_admission_score")
            if reasons:
                reject_counts.update(reasons)
                continue
            candidates.append({
                "doc_id": int(doc_id),
                "title": title,
                "doc": doc,
                "rank": int(rank),
                "plan_index": int(plan_idx),
                "plan_type": plan_type,
                "plan_net_utility": _round(plan_net),
                "plan_utility": _round(plan_utility),
                "plan_risk": _round(plan_risk),
                "relation_overlap": _round(relation_overlap),
                "endpoint_overlap": _round(endpoint_overlap),
                "schema_selectivity": _round(schema_selectivity),
                "provenance_score": _round(provenance_score),
                "missing_gain": _round(missing_gain),
                "query_overlap": _round(query_overlap),
                "support_gain": _round(support_gain),
                "admission_score": _round(admission_score),
                "token_cost": _round(token_cost),
                "candidate_answer": str(plan.get("candidate_answer") or "")[:160],
                "triple_path": list(plan.get("triple_path") or [])[:2],
                "schema_signatures": list(plan.get("schema_signatures") or [])[:2],
            })

    candidates.sort(
        key=lambda item: (
            -float(item.get("admission_score", 0.0)),
            -float(item.get("missing_gain", 0.0)),
            int(item.get("rank", 999999)),
        )
    )
    info["candidate_count"] = len(candidates)
    info["top_candidates"] = [
        {key: value for key, value in item.items() if key != "doc"}
        for item in candidates[:5]
    ]
    selected_doc_ids: List[int] = []
    selected_docs: List[str] = []
    seen_doc_keys = set(current_doc_keys)
    for item in candidates:
        if len(selected_doc_ids) >= int(max_append):
            break
        doc_key = _doc_key(item.get("doc"))
        if not doc_key or doc_key in seen_doc_keys:
            continue
        selected_doc_ids.append(int(item["doc_id"]))
        selected_docs.append(str(item["doc"]))
        seen_doc_keys.add(doc_key)

    info["rejected_count"] = sum(reject_counts.values())
    info["reject_reason_counts"] = dict(reject_counts)
    if not selected_docs:
        info["reason"] = "no_candidate_accepted" if plans else "no_shadow_plans"
        return [], [], info

    info["applied"] = True
    info["reason"] = "selected_reader_support_join"
    info["accepted_count"] = len(selected_docs)
    info["selected_doc_ids"] = selected_doc_ids
    info["selected_titles"] = [_doc_title(doc) for doc in selected_docs]
    return selected_doc_ids, selected_docs, info


def append_selective_support_join_evidence(
    *,
    docs: Sequence[str],
    join_docs: Sequence[str],
    join_info: Optional[Dict[str, Any]],
    enabled: bool = False,
    max_append: int = 1,
) -> Tuple[List[str], Dict[str, Any]]:
    current_docs = list(docs or [])
    join_docs = list(join_docs or [])
    info = dict(join_info or {})
    reader_info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "appended": 0,
        "reason": "disabled",
        "appended_titles": [],
    }
    if not enabled:
        info["reader_projection"] = reader_info
        return current_docs, info
    if not info.get("applied") or not join_docs:
        reader_info["reason"] = str(info.get("reason") or "no_selected_join_docs")
        info["reader_projection"] = reader_info
        return current_docs, info
    seen = {_doc_key(doc) for doc in current_docs}
    appended: List[str] = []
    for doc in join_docs:
        doc_key = _doc_key(doc)
        if not doc_key or doc_key in seen:
            continue
        appended.append(str(doc))
        seen.add(doc_key)
        if len(appended) >= int(max_append):
            break
    if not appended:
        reader_info["reason"] = "selected_docs_already_visible"
        info["reader_projection"] = reader_info
        return current_docs, info
    reader_info["reason"] = "reader_append"
    reader_info["appended"] = len(appended)
    reader_info["appended_titles"] = [_doc_title(doc) for doc in appended]
    info["reader_projection"] = reader_info
    return current_docs + appended, info
