import math
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .relation_fact_join_utils import (
    _active_fact_signals,
    _clamp,
    _entity_key,
    _entity_type,
    _relation_profile,
    _schema_signature,
    _tokens,
)


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


def _norm_text(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _estimate_tokens(text: Any) -> int:
    return max(1, len(re.findall(r"\S+", str(text or ""))))


def _safe_doc_content(get_doc_content: Callable[[int], str], doc_id: int) -> str:
    try:
        return str(get_doc_content(int(doc_id)) or "")
    except Exception:
        return ""


def _doc_keys(doc: Any) -> Set[str]:
    text = str(doc or "")
    title = _norm_text(_doc_title(text))
    body = _norm_text(text[:1500])
    keys = {key for key in (title, body) if key}
    if title:
        keys.add(title[:160])
    return keys


def _gold_keys(gold_docs: Optional[Sequence[Any]]) -> Set[str]:
    keys: Set[str] = set()
    for doc in gold_docs or []:
        keys.update(_doc_keys(doc))
    return {key for key in keys if key}


def _doc_matches_gold(content: str, gold_keys: Set[str]) -> bool:
    if not gold_keys:
        return False
    keys = _doc_keys(content)
    for left in keys:
        for right in gold_keys:
            if not left or not right:
                continue
            if left == right:
                return True
            if len(left) >= 40 and left in right:
                return True
            if len(right) >= 40 and right in left:
                return True
    return False


def _retrieval_oracle(
    *,
    sorted_doc_ids: Sequence[Any],
    get_doc_content: Callable[[int], str],
    gold_docs: Optional[Sequence[Any]],
    k_values: Sequence[int] = (2, 5, 10, 20, 50, 100, 200),
) -> Dict[str, Any]:
    gold = _gold_keys(gold_docs)
    info: Dict[str, Any] = {
        "gold_available": bool(gold),
        "first_gold_rank": None,
    }
    if not gold:
        for k in k_values:
            info[f"gold_at_{k}"] = None
        return info

    first_rank = None
    max_k = min(max(k_values), len(sorted_doc_ids))
    for rank, doc_id in enumerate(sorted_doc_ids[:max_k], start=1):
        content = _safe_doc_content(get_doc_content, int(doc_id))
        if _doc_matches_gold(content, gold):
            first_rank = rank
            break
    info["first_gold_rank"] = first_rank
    for k in k_values:
        info[f"gold_at_{k}"] = bool(first_rank is not None and first_rank <= int(k))
    return info


def _bounded_overlap(left: Set[str], right: Set[str], max_required: int = 4) -> float:
    if not left or not right:
        return 0.0
    return _clamp(len(left & right) / max(1, min(len(right), int(max_required))))


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _rank_map(sorted_doc_ids: Sequence[Any]) -> Dict[int, int]:
    ranks: Dict[int, int] = {}
    for rank, doc_id in enumerate(sorted_doc_ids or [], start=1):
        try:
            ranks.setdefault(int(doc_id), rank)
        except Exception:
            continue
    return ranks


def _lookup_items(
    relation_index: Dict[str, Any],
    keys: Iterable[Any],
    *,
    max_per_key: int,
    max_fanout: int,
) -> List[Dict[str, Any]]:
    items = relation_index.get("items") or []
    entity_to_items = relation_index.get("entity_to_items") or {}
    seen: Set[int] = set()
    out: List[Dict[str, Any]] = []
    for raw_key in keys:
        key = str(raw_key or "").strip()
        if not key:
            continue
        ids = list(entity_to_items.get(key, []) or [])
        if len(ids) > int(max_fanout):
            continue
        for item_id in ids[: max(1, int(max_per_key))]:
            if item_id in seen:
                continue
            seen.add(int(item_id))
            if 0 <= int(item_id) < len(items):
                item = items[int(item_id)]
                if isinstance(item, dict):
                    out.append(item)
    return out


def _lookup_relation_items(
    relation_index: Dict[str, Any],
    keys: Iterable[Any],
    *,
    max_per_key: int,
    max_fanout: int,
) -> List[Dict[str, Any]]:
    items = relation_index.get("items") or []
    relation_to_items = relation_index.get("relation_to_items") or {}
    seen: Set[int] = set()
    out: List[Dict[str, Any]] = []
    for raw_key in keys:
        key = str(raw_key or "").strip()
        if not key:
            continue
        ids = list(relation_to_items.get(key, []) or [])
        if len(ids) > int(max_fanout):
            continue
        for item_id in ids[: max(1, int(max_per_key))]:
            if item_id in seen:
                continue
            seen.add(int(item_id))
            if 0 <= int(item_id) < len(items):
                item = items[int(item_id)]
                if isinstance(item, dict):
                    out.append(item)
    return out


def _schema_frequency(relation_index: Dict[str, Any], signature: Any) -> int:
    if signature is None:
        return 0
    return len((relation_index.get("schema_to_items") or {}).get(signature, []) or [])


def _tuple_reliability(relation_index: Dict[str, Any], tuple_key: str) -> Dict[str, Any]:
    stats = ((relation_index.get("fact_reliability") or {}).get("tuple_stats") or {}).get(
        str(tuple_key or ""),
        {},
    )
    doc_count = _as_int(stats.get("doc_count"), 0)
    triple_count = _as_int(stats.get("triple_count"), 0)
    score = _clamp(0.45 * min(1.0, doc_count / 3.0) + 0.25 * min(1.0, triple_count / 4.0))
    return {
        "doc_count": doc_count,
        "triple_count": triple_count,
        "score": round(float(score), 6),
    }


def _item_doc_id(item: Dict[str, Any]) -> Optional[int]:
    try:
        return int(item.get("doc_idx"))
    except Exception:
        return None


def _endpoint_sides(item: Dict[str, Any]) -> List[Tuple[str, str, Set[str]]]:
    return [
        ("subject", str(item.get("subject") or ""), set(item.get("subject_tokens") or set())),
        ("object", str(item.get("object") or ""), set(item.get("object_tokens") or set())),
    ]


def _answer_type_bonus(value: str, answer_types: Set[str]) -> float:
    if not answer_types:
        return 0.0
    value_type = _entity_type(value)
    if value_type in answer_types:
        return 0.08
    if "place" in answer_types and value_type == "entity" and (_tokens(value) & {"city", "country", "state"}):
        return 0.04
    return 0.0


def _score_plan(
    *,
    plan_type: str,
    question_tokens: Set[str],
    query_anchor_tokens: Set[str],
    query_relation_tokens: Set[str],
    answer_types: Set[str],
    relation_index: Dict[str, Any],
    doc_ids: Sequence[int],
    endpoint_tokens: Set[str],
    relation_tokens: Set[str],
    answer_value: str,
    signatures: Sequence[Any],
    rank_map: Dict[int, int],
    get_doc_content: Callable[[int], str],
    raw_candidate_count: int,
    token_budget: int,
    token_weight: float,
    compute_weight: float,
    risk_weight: float,
) -> Dict[str, Any]:
    ranks = [int(rank_map.get(int(doc_id), 999999)) for doc_id in doc_ids]
    best_rank = min(ranks) if ranks else 999999
    worst_rank = max(ranks) if ranks else 999999
    rank_prior = 1.0 / (1.0 + math.log1p(max(1, best_rank)))
    endpoint_overlap = _bounded_overlap(endpoint_tokens, query_anchor_tokens, max_required=5)
    relation_overlap = (
        _bounded_overlap(relation_tokens, query_relation_tokens, max_required=4)
        if query_relation_tokens else 0.0
    )
    query_overlap = _bounded_overlap(endpoint_tokens | relation_tokens, question_tokens, max_required=6)
    schema_freqs = [_schema_frequency(relation_index, signature) for signature in signatures if signature]
    min_schema_freq = min(schema_freqs) if schema_freqs else 0
    schema_selectivity = 1.0 if not min_schema_freq else _clamp(1.0 - min_schema_freq / 1200.0)
    provenance_scores = []
    for doc_id in doc_ids:
        for item_id in (relation_index.get("doc_to_items") or {}).get(int(doc_id), [])[:12]:
            items = relation_index.get("items") or []
            if 0 <= int(item_id) < len(items):
                provenance_scores.append(
                    _tuple_reliability(relation_index, str(items[int(item_id)].get("tuple_key") or "")).get(
                        "score",
                        0.0,
                    )
                )
    provenance = max(provenance_scores or [0.0])
    answer_bonus = _answer_type_bonus(answer_value, answer_types)
    utility = _clamp(
        0.24 * query_overlap
        + 0.20 * endpoint_overlap
        + 0.18 * relation_overlap
        + 0.12 * rank_prior
        + 0.10 * schema_selectivity
        + 0.08 * provenance
        + answer_bonus
        + (0.06 if plan_type == "two_hop_fact_join" else 0.0)
    )

    docs_text = [_safe_doc_content(get_doc_content, int(doc_id)) for doc_id in doc_ids]
    token_count = sum(_estimate_tokens(text) for text in docs_text)
    token_cost = _clamp(token_count / max(1.0, float(token_budget)))
    compute_cost = _clamp(math.log1p(max(1, raw_candidate_count)) / math.log1p(800.0))
    fanout_risk = _clamp((max(schema_freqs or [0]) or 0) / 2000.0)
    low_relation_risk = 0.0
    if query_relation_tokens:
        low_relation_risk = _clamp((0.20 - relation_overlap) / 0.20)
    low_anchor_risk = _clamp((0.20 - endpoint_overlap) / 0.20)
    late_rank_risk = 0.12 if best_rank > 20 else 0.04 if best_rank > 5 else 0.0
    risk = _clamp(
        0.24 * fanout_risk
        + 0.22 * low_relation_risk
        + 0.20 * low_anchor_risk
        + late_rank_risk
    )
    net_utility = (
        utility
        - float(token_weight) * token_cost
        - float(compute_weight) * compute_cost
        - float(risk_weight) * risk
    )
    return {
        "utility": _round(utility),
        "risk": _round(risk),
        "net_utility": _round(net_utility),
        "token_count": int(token_count),
        "token_cost": _round(token_cost),
        "compute_cost": _round(compute_cost),
        "rank_prior": _round(rank_prior),
        "best_rank": int(best_rank),
        "worst_rank": int(worst_rank),
        "endpoint_overlap": _round(endpoint_overlap),
        "relation_overlap": _round(relation_overlap),
        "query_overlap": _round(query_overlap),
        "schema_selectivity": _round(schema_selectivity),
        "provenance_score": _round(provenance),
        "answer_type_bonus": _round(answer_bonus),
    }


def _sanitize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    keep = {
        "plan_type",
        "join_order",
        "candidate_answer",
        "answer_type",
        "doc_ids",
        "titles",
        "best_rank",
        "worst_rank",
        "utility",
        "risk",
        "net_utility",
        "token_cost",
        "compute_cost",
        "endpoint_overlap",
        "relation_overlap",
        "schema_selectivity",
        "provenance_score",
        "accepted",
        "reject_reason",
        "gold_doc_hit",
        "schema_signatures",
        "triple_path",
        "bridge_entity",
        "bridge_overlap",
    }
    out = {key: plan.get(key) for key in keep if key in plan}
    if "triple_path" in out:
        out["triple_path"] = out["triple_path"][:2]
    if "titles" in out:
        out["titles"] = out["titles"][:3]
    return out


def build_typed_fact_join_shadow_optimizer(
    *,
    question: str,
    sorted_doc_ids: Sequence[Any],
    top_k_facts: Sequence[Any],
    candidate_facts: Optional[Sequence[Any]],
    relation_index: Optional[Dict[str, Any]],
    get_doc_content: Callable[[int], str],
    gold_docs: Optional[Sequence[Any]] = None,
    enabled: bool = False,
    candidate_top_k: int = 200,
    max_anchor_items: int = 48,
    max_second_hop_items: int = 64,
    max_plans: int = 12,
    max_key_fanout: int = 800,
    min_utility: float = 0.34,
    max_risk: float = 0.70,
    min_net_utility: float = -0.06,
    token_budget: int = 1200,
    token_weight: float = 0.08,
    compute_weight: float = 0.05,
    risk_weight: float = 0.36,
) -> Dict[str, Any]:
    """Shadow-only optimizer over typed OpenIE facts.

    The output is intentionally diagnostic: it does not change retrieval order
    or reader evidence. It exposes whether the persistent typed fact store can
    compile promising one-hop/two-hop join plans inside the retrieved candidate
    pool, plus oracle signals for later activation decisions.
    """
    relation_index = relation_index or {}
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "shadow_only": True,
        "applied": False,
        "reason": "disabled" if not enabled else "not_evaluated",
        "index_stats": dict(relation_index.get("stats", {}) or {}),
        "candidate_top_k": int(candidate_top_k),
        "candidate_item_count": 0,
        "raw_plan_count": 0,
        "accepted_plan_count": 0,
        "rejected_plan_count": 0,
        "best_plan_score": 0.0,
        "best_plan_type": "",
        "best_plan_gold_doc_hit": None,
        "oracle": {},
        "fact_store": {},
        "reject_reason_counts": {},
        "plan_type_counts": {},
        "accepted_plan_type_counts": {},
        "top_plans": [],
    }
    if not enabled:
        return info
    if not relation_index.get("items"):
        info["reason"] = "no_fact_store"
        return info

    sorted_doc_ids = [] if sorted_doc_ids is None else list(sorted_doc_ids)
    rank_map = _rank_map(sorted_doc_ids)
    oracle = _retrieval_oracle(
        sorted_doc_ids=sorted_doc_ids,
        get_doc_content=get_doc_content,
        gold_docs=gold_docs,
    )
    info["oracle"] = oracle

    profile = _relation_profile(question)
    question_tokens = _tokens(question)
    query_anchor_tokens = set(profile.get("anchor_tokens") or []) | question_tokens
    query_relation_tokens = set(profile.get("relation_tokens") or [])
    answer_types = set(profile.get("answer_types") or [])
    active = _active_fact_signals(
        question,
        list(top_k_facts or []),
        list(candidate_facts or []),
        max_facts=24,
    )

    active_entity_keys = set(active.get("entity_keys") or set())
    active_endpoint_tokens = set(active.get("endpoint_tokens") or set())
    active_relation_tokens = set(active.get("relation_tokens") or set())
    active_signatures = set(active.get("signatures") or set())
    lookup_keys = list(dict.fromkeys(
        list(active_entity_keys)
        + list(active_endpoint_tokens)
        + list(query_anchor_tokens)
    ))[:96]
    relation_keys = list(dict.fromkeys(list(query_relation_tokens | active_relation_tokens)))[:48]

    anchor_items = _lookup_items(
        relation_index,
        lookup_keys,
        max_per_key=max_anchor_items,
        max_fanout=max_key_fanout,
    )
    relation_items = _lookup_relation_items(
        relation_index,
        relation_keys,
        max_per_key=max_anchor_items,
        max_fanout=max_key_fanout,
    )
    by_id: Dict[int, Dict[str, Any]] = {}
    for item in anchor_items + relation_items:
        item_id = _as_int(item.get("item_id"), -1)
        if item_id >= 0:
            by_id[item_id] = item
    candidate_items = list(by_id.values())
    info["candidate_item_count"] = len(candidate_items)
    info["fact_store"] = {
        "active_fact_count": len(active.get("active_triples") or []),
        "active_entity_key_count": len(active_entity_keys),
        "active_relation_token_count": len(active_relation_tokens),
        "active_schema_count": len(active_signatures),
        "query_anchor_token_count": len(query_anchor_tokens),
        "query_relation_token_count": len(query_relation_tokens),
        "answer_types": sorted(answer_types),
        "retrieved_doc_fact_coverage_top2": sum(
            1 for doc_id in sorted_doc_ids[:2]
            if int(doc_id) in (relation_index.get("doc_to_items") or {})
        ),
        "retrieved_doc_fact_coverage_top5": sum(
            1 for doc_id in sorted_doc_ids[:5]
            if int(doc_id) in (relation_index.get("doc_to_items") or {})
        ),
        "retrieved_doc_fact_coverage_top20": sum(
            1 for doc_id in sorted_doc_ids[:20]
            if int(doc_id) in (relation_index.get("doc_to_items") or {})
        ),
    }
    if not candidate_items:
        info["reason"] = "no_candidate_items"
        return info

    gold_keys = _gold_keys(gold_docs)
    plans: List[Dict[str, Any]] = []
    raw_candidate_count = len(candidate_items)

    for item in candidate_items[: max(1, int(candidate_top_k))]:
        doc_id = _item_doc_id(item)
        if doc_id is None:
            continue
        triple = item.get("triple") or ("", "", "")
        endpoint_tokens = set(item.get("endpoint_tokens") or set())
        relation_tokens = set(item.get("relation_tokens") or set())
        signature = item.get("signature") or _schema_signature(triple)
        candidate_answer = str(triple[2] if isinstance(triple, (list, tuple)) and len(triple) == 3 else "")
        score = _score_plan(
            plan_type="one_hop_fact_access",
            question_tokens=question_tokens,
            query_anchor_tokens=query_anchor_tokens,
            query_relation_tokens=query_relation_tokens,
            answer_types=answer_types,
            relation_index=relation_index,
            doc_ids=[doc_id],
            endpoint_tokens=endpoint_tokens,
            relation_tokens=relation_tokens,
            answer_value=candidate_answer,
            signatures=[signature],
            rank_map=rank_map,
            get_doc_content=get_doc_content,
            raw_candidate_count=raw_candidate_count,
            token_budget=token_budget,
            token_weight=token_weight,
            compute_weight=compute_weight,
            risk_weight=risk_weight,
        )
        content = _safe_doc_content(get_doc_content, doc_id)
        plan = {
            "plan_type": "one_hop_fact_access",
            "join_order": ["fact"],
            "candidate_answer": candidate_answer,
            "answer_type": _entity_type(candidate_answer),
            "doc_ids": [doc_id],
            "titles": [_doc_title(content)],
            "schema_signatures": [str(signature)],
            "triple_path": [" | ".join(str(part) for part in triple)],
            "gold_doc_hit": _doc_matches_gold(content, gold_keys),
            **score,
        }
        plans.append(plan)


    second_seen: Set[Tuple[int, int]] = set()
    for first in candidate_items[: max(1, int(max_anchor_items))]:
        first_id = _as_int(first.get("item_id"), -1)
        first_doc = _item_doc_id(first)
        if first_doc is None:
            continue
        first_relation = set(first.get("relation_tokens") or set())
        first_signature = first.get("signature")
        for side, bridge_text, bridge_tokens in _endpoint_sides(first):
            if not bridge_tokens:
                continue
            if _bounded_overlap(bridge_tokens, query_anchor_tokens, max_required=4) >= 0.67:
                continue
            second_items = _lookup_items(
                relation_index,
                list(bridge_tokens)[:8],
                max_per_key=max_second_hop_items,
                max_fanout=max_key_fanout,
            )
            for second in second_items[: max(1, int(max_second_hop_items))]:
                second_id = _as_int(second.get("item_id"), -1)
                if second_id < 0 or second_id == first_id or (first_id, second_id) in second_seen:
                    continue
                second_seen.add((first_id, second_id))
                second_doc = _item_doc_id(second)
                if second_doc is None:
                    continue
                best_bridge_overlap = 0.0
                answer_value = ""
                for second_side, second_text, second_tokens in _endpoint_sides(second):
                    overlap = _bounded_overlap(second_tokens, bridge_tokens, max_required=4)
                    if overlap > best_bridge_overlap:
                        best_bridge_overlap = overlap
                        other_side = "object" if second_side == "subject" else "subject"
                        answer_value = str(second.get(other_side) or "")
                if best_bridge_overlap < 0.50 or not answer_value:
                    continue
                second_relation = set(second.get("relation_tokens") or set())
                endpoint_tokens = (
                    set(first.get("endpoint_tokens") or set())
                    | set(second.get("endpoint_tokens") or set())
                    | bridge_tokens
                )
                relation_tokens = first_relation | second_relation
                score = _score_plan(
                    plan_type="two_hop_fact_join",
                    question_tokens=question_tokens,
                    query_anchor_tokens=query_anchor_tokens,
                    query_relation_tokens=query_relation_tokens,
                    answer_types=answer_types,
                    relation_index=relation_index,
                    doc_ids=[first_doc, second_doc],
                    endpoint_tokens=endpoint_tokens,
                    relation_tokens=relation_tokens,
                    answer_value=answer_value,
                    signatures=[first_signature, second.get("signature")],
                    rank_map=rank_map,
                    get_doc_content=get_doc_content,
                    raw_candidate_count=raw_candidate_count + len(second_seen),
                    token_budget=token_budget,
                    token_weight=token_weight,
                    compute_weight=compute_weight,
                    risk_weight=risk_weight,
                )
                first_content = _safe_doc_content(get_doc_content, first_doc)
                second_content = _safe_doc_content(get_doc_content, second_doc)
                plan = {
                    "plan_type": "two_hop_fact_join",
                    "join_order": ["anchor_fact", "bridge_fact"],
                    "candidate_answer": answer_value,
                    "answer_type": _entity_type(answer_value),
                    "doc_ids": list(dict.fromkeys([first_doc, second_doc])),
                    "titles": [_doc_title(first_content), _doc_title(second_content)],
                    "schema_signatures": [str(first_signature), str(second.get("signature"))],
                    "triple_path": [
                        " | ".join(str(part) for part in (first.get("triple") or [])),
                        " | ".join(str(part) for part in (second.get("triple") or [])),
                    ],
                    "bridge_entity": bridge_text,
                    "bridge_overlap": _round(best_bridge_overlap),
                    "gold_doc_hit": (
                        _doc_matches_gold(first_content, gold_keys)
                        or _doc_matches_gold(second_content, gold_keys)
                    ),
                    **score,
                }
                plans.append(plan)
                if len(plans) >= int(candidate_top_k) * 4:
                    break
            if len(plans) >= int(candidate_top_k) * 4:
                break
        if len(plans) >= int(candidate_top_k) * 4:
            break

    reject_counts = Counter()
    plan_type_counts = Counter()
    accepted_type_counts = Counter()
    accepted = []
    for plan in plans:
        plan_type_counts.update([plan.get("plan_type", "unknown")])
        reasons = []
        if float(plan.get("utility", 0.0)) < float(min_utility):
            reasons.append("utility_below_threshold")
        if float(plan.get("risk", 0.0)) > float(max_risk):
            reasons.append("risk_above_threshold")
        if float(plan.get("net_utility", 0.0)) < float(min_net_utility):
            reasons.append("net_utility_below_threshold")
        if reasons:
            plan["accepted"] = False
            plan["reject_reason"] = "+".join(reasons)
            reject_counts.update(reasons)
        else:
            plan["accepted"] = True
            plan["reject_reason"] = "accepted"
            accepted.append(plan)
            accepted_type_counts.update([plan.get("plan_type", "unknown")])

    plans.sort(
        key=lambda row: (
            -float(row.get("accepted", False)),
            -float(row.get("net_utility", 0.0)),
            -float(row.get("utility", 0.0)),
            float(row.get("risk", 0.0)),
            int(row.get("best_rank", 999999)),
        )
    )
    info["raw_plan_count"] = len(plans)
    info["accepted_plan_count"] = len(accepted)
    info["rejected_plan_count"] = max(0, len(plans) - len(accepted))
    info["reject_reason_counts"] = dict(reject_counts)
    info["plan_type_counts"] = dict(plan_type_counts)
    info["accepted_plan_type_counts"] = dict(accepted_type_counts)
    info["top_plans"] = [_sanitize_plan(plan) for plan in plans[: max(1, int(max_plans))]]
    if plans:
        best = plans[0]
        info["best_plan_score"] = _round(best.get("net_utility", 0.0))
        info["best_plan_type"] = str(best.get("plan_type", ""))
        info["best_plan_gold_doc_hit"] = bool(best.get("gold_doc_hit")) if gold_keys else None
        info["reason"] = "plans_generated"
    else:
        info["reason"] = "no_join_plans"
    info["applied"] = False
    return info
