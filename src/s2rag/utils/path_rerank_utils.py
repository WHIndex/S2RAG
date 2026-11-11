import math
import re
from collections import defaultdict
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
TITLE_CASE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+(?:of|the|and|de|la|[A-Z][a-z]+)){0,5}\b")
LOCATION_WORD_RE = re.compile(
    r"\b(?:city|town|village|county|state|province|country|region|district|"
    r"capital|river|lake|mountain|island|municipality|prefecture|kingdom|"
    r"republic|territory)\b",
    re.IGNORECASE,
)
OCCUPATION_WORD_RE = re.compile(
    r"\b(?:actor|actress|singer|writer|author|director|producer|poet|"
    r"politician|scientist|engineer|artist|painter|footballer|player|coach|"
    r"journalist|historian|professor|mathematician|physician|lawyer|designer|"
    r"architect|musician|rapper|novelist|screenwriter|businessperson)\b",
    re.IGNORECASE,
)


def normalize_phrase(text: Any) -> str:
    return " ".join(TOKEN_RE.findall(str(text or "").lower()))


def tokenize(text: Any) -> List[str]:
    return [
        token
        for token in TOKEN_RE.findall(str(text or "").lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def _doc_title(doc: str) -> str:
    if not doc:
        return ""
    return str(doc).split("\n", 1)[0].strip()


def _doc_body(doc: str) -> str:
    if not doc:
        return ""
    parts = str(doc).split("\n", 1)
    return parts[1] if len(parts) > 1 else parts[0]


def _target_signal(text: str, target_type: str) -> float:
    if target_type == "date":
        return 1.0 if YEAR_RE.search(text) else 0.0
    if target_type == "number":
        return 1.0 if NUMBER_RE.search(text) else 0.0
    if target_type == "location":
        return 1.0 if LOCATION_WORD_RE.search(text) else (0.45 if TITLE_CASE_RE.search(text) else 0.0)
    if target_type == "occupation":
        return 1.0 if OCCUPATION_WORD_RE.search(text) else 0.0
    if target_type in {"person", "organization", "work", "factoid"}:
        return 0.75 if TITLE_CASE_RE.search(text) else 0.0
    return 0.35 if (TITLE_CASE_RE.search(text) or YEAR_RE.search(text) or NUMBER_RE.search(text)) else 0.0


def _entity_jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


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


def _normalize_values(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v <= min_v:
        return [1.0 for _ in values]
    return [(value - min_v) / (max_v - min_v) for value in values]


def build_chunk_path_info(openie_info: List[dict]) -> Dict[str, Dict[str, Any]]:
    """Build lightweight relation/entity summaries for path-aware reranking."""
    chunk_info: Dict[str, Dict[str, Any]] = {}

    for item in openie_info or []:
        chunk_id = item.get("idx")
        if not chunk_id:
            continue

        entities = set()
        entity_tokens = set()
        relation_tokens = set()

        for entity in item.get("extracted_entities", []) or []:
            norm_entity = normalize_phrase(entity)
            if norm_entity:
                entities.add(norm_entity)
                entity_tokens.update(tokenize(entity))

        for triple in item.get("extracted_triples", []) or []:
            if len(triple) != 3:
                continue
            subject, predicate, obj = triple
            for entity in (subject, obj):
                norm_entity = normalize_phrase(entity)
                if norm_entity:
                    entities.add(norm_entity)
                    entity_tokens.update(tokenize(entity))
            relation_tokens.update(tokenize(predicate))

        chunk_info[chunk_id] = {
            "entities": entities,
            "entity_tokens": entity_tokens,
            "relation_tokens": relation_tokens,
        }

    return chunk_info


def _fact_anchor_weights(
    top_k_facts: List[Tuple],
    top_k_fact_indices: List[int],
    query_fact_scores: np.ndarray,
) -> Tuple[Dict[str, float], set]:
    anchor_weights = defaultdict(float)
    predicate_tokens = set()

    for rank, fact in enumerate(top_k_facts or []):
        if len(fact) != 3:
            continue
        fact_index = top_k_fact_indices[rank] if rank < len(top_k_fact_indices or []) else rank
        fact_score = _safe_score(query_fact_scores, fact_index)
        if fact_score <= 0:
            fact_score = 1.0 / (rank + 1)

        subject, predicate, obj = fact
        predicate_tokens.update(tokenize(predicate))
        for entity in (subject, obj):
            norm_entity = normalize_phrase(entity)
            if norm_entity:
                anchor_weights[norm_entity] = max(anchor_weights[norm_entity], fact_score)

    if not anchor_weights:
        return {}, predicate_tokens

    max_score = max(anchor_weights.values())
    if max_score > 0:
        for entity in list(anchor_weights.keys()):
            anchor_weights[entity] = anchor_weights[entity] / max_score

    return dict(anchor_weights), predicate_tokens


def _chunk_info_for_doc(
    doc_id: int,
    passage_node_keys: List[str],
    chunk_path_info: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        chunk_key = passage_node_keys[int(doc_id)]
    except Exception:
        return {"entities": set(), "entity_tokens": set(), "relation_tokens": set()}
    return chunk_path_info.get(chunk_key, {"entities": set(), "entity_tokens": set(), "relation_tokens": set()})


def _overlap_ratio(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def path_aware_rerank_docs(
    query: str,
    sorted_doc_ids: np.ndarray,
    sorted_doc_scores: np.ndarray,
    passage_node_keys: List[str],
    chunk_path_info: Dict[str, Dict[str, Any]],
    top_k_facts: List[Tuple],
    top_k_fact_indices: List[int],
    query_fact_scores: np.ndarray,
    qa_top_k: int = 5,
    rerank_top_k: int = 200,
    weight: float = 0.30,
    enabled: bool = False,
    bridge_only: bool = True,
    has_bridge_cues: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Promote top-N candidates that complete query-conditioned entity/relation paths."""
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "bridge_cues": bool(has_bridge_cues),
        "rerank_top_k": rerank_top_k,
        "weight": weight,
        "promoted_to_top5": 0,
        "top_promotions": [],
    }

    if not enabled:
        return sorted_doc_ids, sorted_doc_scores, info
    if bridge_only and not has_bridge_cues:
        info["reason"] = "no_bridge_cues"
        return sorted_doc_ids, sorted_doc_scores, info
    if sorted_doc_ids is None or len(sorted_doc_ids) == 0:
        info["reason"] = "no_candidates"
        return sorted_doc_ids, sorted_doc_scores, info

    top_n = min(max(qa_top_k, rerank_top_k), len(sorted_doc_ids))
    if top_n <= qa_top_k:
        info["reason"] = "not_enough_candidates"
        return sorted_doc_ids, sorted_doc_scores, info

    anchor_weights, predicate_tokens = _fact_anchor_weights(top_k_facts, top_k_fact_indices, query_fact_scores)
    if not anchor_weights and not predicate_tokens:
        info["reason"] = "no_fact_anchors"
        return sorted_doc_ids, sorted_doc_scores, info

    query_tokens = set(tokenize(query))
    candidate_ids = [int(doc_id) for doc_id in sorted_doc_ids[:top_n].tolist()]
    original_scores = [float(score) for score in sorted_doc_scores[:top_n].tolist()]
    normalized_original_scores = _normalize_values(original_scores)

    seed_k = min(max(1, qa_top_k), len(candidate_ids))
    base_entities = set()
    base_entity_tokens = set()
    for doc_id in candidate_ids[:seed_k]:
        chunk_info = _chunk_info_for_doc(doc_id, passage_node_keys, chunk_path_info)
        base_entities.update(chunk_info.get("entities", set()))
        base_entity_tokens.update(chunk_info.get("entity_tokens", set()))

    anchor_total = sum(anchor_weights.values()) or 1.0
    scored_candidates = []

    for rank, doc_id in enumerate(candidate_ids):
        chunk_info = _chunk_info_for_doc(doc_id, passage_node_keys, chunk_path_info)
        entities = chunk_info.get("entities", set())
        entity_tokens = chunk_info.get("entity_tokens", set())
        relation_tokens = chunk_info.get("relation_tokens", set())

        anchor_overlap_weight = sum(anchor_weights.get(entity, 0.0) for entity in entities & set(anchor_weights))
        anchor_score = min(1.0, anchor_overlap_weight / anchor_total)

        novel_anchor_overlap_weight = sum(
            anchor_weights.get(entity, 0.0)
            for entity in (entities - base_entities) & set(anchor_weights)
        )
        novel_anchor_score = min(1.0, novel_anchor_overlap_weight / anchor_total)

        relation_score = max(
            _overlap_ratio(query_tokens, relation_tokens),
            _overlap_ratio(predicate_tokens, relation_tokens),
        )
        entity_query_score = _overlap_ratio(query_tokens, entity_tokens)

        base_overlap = _overlap_ratio(entities, base_entities)
        complement = min(1.0, len(entities - base_entities) / 8.0) if entities else 0.0
        bridge_score = math.sqrt(base_overlap * complement) if base_overlap > 0 and complement > 0 else 0.0

        rank_prior = 1.0 / (rank + 1)
        path_score = (
            0.34 * anchor_score
            + 0.22 * novel_anchor_score
            + 0.18 * relation_score
            + 0.14 * entity_query_score
            + 0.08 * bridge_score
            + 0.04 * rank_prior
        )
        final_score = normalized_original_scores[rank] + weight * path_score

        scored_candidates.append({
            "doc_id": doc_id,
            "rank": rank,
            "original_score": original_scores[rank],
            "normalized_original_score": normalized_original_scores[rank],
            "path_score": path_score,
            "final_score": final_score,
            "anchor_score": anchor_score,
            "novel_anchor_score": novel_anchor_score,
            "relation_score": relation_score,
            "entity_query_score": entity_query_score,
            "bridge_score": bridge_score,
        })

    reranked = sorted(scored_candidates, key=lambda item: (-item["final_score"], item["rank"]))
    new_doc_ids = [item["doc_id"] for item in reranked] + [int(doc_id) for doc_id in sorted_doc_ids[top_n:].tolist()]
    final_scores = [item["final_score"] for item in reranked]
    final_scores = _normalize_values(final_scores)
    new_doc_scores = final_scores + [float(score) for score in sorted_doc_scores[top_n:].tolist()]

    original_top5 = set(candidate_ids[:qa_top_k])
    new_top5 = [item["doc_id"] for item in reranked[:qa_top_k]]
    promoted = [doc_id for doc_id in new_top5 if doc_id not in original_top5]

    promotion_details = []
    original_rank = {doc_id: rank for rank, doc_id in enumerate(candidate_ids)}
    for new_rank, item in enumerate(reranked[:qa_top_k]):
        if item["doc_id"] in promoted:
            promotion_details.append({
                "doc_id": item["doc_id"],
                "from_rank": original_rank.get(item["doc_id"]),
                "to_rank": new_rank,
                "path_score": round(item["path_score"], 6),
                "anchor_score": round(item["anchor_score"], 6),
                "novel_anchor_score": round(item["novel_anchor_score"], 6),
                "relation_score": round(item["relation_score"], 6),
                "bridge_score": round(item["bridge_score"], 6),
            })

    info.update({
        "applied": True,
        "reason": "applied",
        "candidate_count": top_n,
        "anchor_count": len(anchor_weights),
        "predicate_token_count": len(predicate_tokens),
        "promoted_to_top5": len(promoted),
        "top_promotions": promotion_details[:5],
        "mean_path_score_top5": round(float(np.mean([item["path_score"] for item in reranked[:qa_top_k]])), 6),
    })

    return np.array(new_doc_ids, dtype=int), np.array(new_doc_scores, dtype=float), info


def constrained_chain_rerank_docs(
    query: str,
    sorted_doc_ids: np.ndarray,
    sorted_doc_scores: np.ndarray,
    candidate_docs: List[str],
    passage_node_keys: List[str],
    chunk_path_info: Dict[str, Dict[str, Any]],
    top_k_facts: List[Tuple],
    top_k_fact_indices: List[int],
    query_fact_scores: np.ndarray,
    qa_top_k: int = 5,
    candidate_top_k: int = 200,
    protect_top_k: int = 1,
    max_promotions: int = 2,
    min_chain_score: float = 0.30,
    enabled: bool = False,
    bridge_only: bool = True,
    has_bridge_cues: bool = False,
    require_insufficient: bool = True,
    evidence_insufficient: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Constrained top-k reconstruction for missing bridge evidence.

    Unlike the older path reranker, this does not globally sort the top-N list.
    It keeps a protected prefix, then rebuilds the remaining QA prefix with
    candidates that have query-fact anchors, relation/path support, and evidence
    diversity. This is meant to affect Recall@2/5 without the broad top-k drift
    that hurt earlier runs.
    """
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "bridge_cues": bool(has_bridge_cues),
        "evidence_insufficient": bool(evidence_insufficient),
        "candidate_top_k": candidate_top_k,
        "protect_top_k": protect_top_k,
        "max_promotions": max_promotions,
        "min_chain_score": min_chain_score,
        "promoted_to_top2": 0,
        "promoted_to_top5": 0,
        "promotions": [],
    }

    if not enabled:
        return sorted_doc_ids, sorted_doc_scores, info
    if bridge_only and not has_bridge_cues:
        info["reason"] = "no_bridge_cues"
        return sorted_doc_ids, sorted_doc_scores, info
    if require_insufficient and not evidence_insufficient:
        info["reason"] = "evidence_not_marked_insufficient"
        return sorted_doc_ids, sorted_doc_scores, info
    if sorted_doc_ids is None or len(sorted_doc_ids) == 0:
        info["reason"] = "no_candidates"
        return sorted_doc_ids, sorted_doc_scores, info

    target_k = min(max(1, qa_top_k), len(sorted_doc_ids), len(candidate_docs))
    top_n = min(max(target_k, candidate_top_k), len(sorted_doc_ids), len(candidate_docs))
    if top_n <= target_k:
        info["reason"] = "not_enough_candidates"
        return sorted_doc_ids, sorted_doc_scores, info

    anchor_weights, predicate_tokens = _fact_anchor_weights(top_k_facts, top_k_fact_indices, query_fact_scores)
    if not anchor_weights:
        info["reason"] = "no_entity_anchors"
        return sorted_doc_ids, sorted_doc_scores, info

    query_tokens = set(tokenize(query))
    target_type = infer_answer_target_type(query)
    candidate_ids = [int(doc_id) for doc_id in sorted_doc_ids[:top_n].tolist()]
    original_scores = [float(score) for score in sorted_doc_scores[:top_n].tolist()]
    normalized_original_scores = _normalize_values(original_scores)
    original_top = candidate_ids[:target_k]
    original_top_set = set(original_top)
    protected_prefix = min(max(0, protect_top_k), target_k, len(candidate_ids))
    protected_ids = candidate_ids[:protected_prefix]

    protected_entities = set()
    protected_entity_tokens = set()
    for doc_id in protected_ids:
        chunk_info = _chunk_info_for_doc(doc_id, passage_node_keys, chunk_path_info)
        protected_entities.update(chunk_info.get("entities", set()))
        protected_entity_tokens.update(chunk_info.get("entity_tokens", set()))

    anchor_entities = set(anchor_weights)
    anchor_total = sum(anchor_weights.values()) or 1.0
    scored = []

    for rank, doc_id in enumerate(candidate_ids):
        if rank < protected_prefix:
            continue

        chunk_info = _chunk_info_for_doc(doc_id, passage_node_keys, chunk_path_info)
        entities = chunk_info.get("entities", set())
        entity_tokens = chunk_info.get("entity_tokens", set())
        relation_tokens = chunk_info.get("relation_tokens", set())
        if not entities:
            continue

        doc = candidate_docs[rank]
        title = _doc_title(doc)
        body = _doc_body(doc)
        text_head = f"{title} {body[:1800]}"

        anchor_overlap_weight = sum(anchor_weights.get(entity, 0.0) for entity in entities & anchor_entities)
        anchor_score = min(1.0, anchor_overlap_weight / anchor_total)
        novel_anchor_overlap_weight = sum(
            anchor_weights.get(entity, 0.0)
            for entity in (entities - protected_entities) & anchor_entities
        )
        novel_anchor_score = min(1.0, novel_anchor_overlap_weight / anchor_total)
        relation_score = max(
            _overlap_ratio(query_tokens, relation_tokens),
            _overlap_ratio(predicate_tokens, relation_tokens),
        )
        entity_query_score = _overlap_ratio(query_tokens, entity_tokens)
        protected_overlap = _overlap_ratio(entities, protected_entities)
        complement = min(1.0, len(entities - protected_entities) / 8.0)
        bridge_score = math.sqrt(protected_overlap * complement) if protected_overlap > 0 and complement > 0 else 0.0
        target_score = _target_signal(text_head, target_type)
        rank_prior = 1.0 / (rank + 1)
        retrieval_score = normalized_original_scores[rank]

        chain_score = (
            0.24 * anchor_score
            + 0.20 * novel_anchor_score
            + 0.18 * relation_score
            + 0.14 * entity_query_score
            + 0.12 * bridge_score
            + 0.08 * target_score
            + 0.04 * rank_prior
        )
        final_score = 0.52 * retrieval_score + 0.48 * chain_score
        is_new_promotion = doc_id not in original_top_set

        strong_path_signal = (
            chain_score >= min_chain_score
            and (anchor_score >= 0.18 or novel_anchor_score >= 0.06)
            and (relation_score >= 0.08 or entity_query_score >= 0.12 or bridge_score >= 0.10)
        )
        if is_new_promotion and not strong_path_signal:
            continue

        scored.append({
            "doc_id": doc_id,
            "rank": rank,
            "title": title,
            "entities": set(entities),
            "chain_score": chain_score,
            "final_score": final_score,
            "retrieval_score": retrieval_score,
            "anchor_score": anchor_score,
            "novel_anchor_score": novel_anchor_score,
            "relation_score": relation_score,
            "entity_query_score": entity_query_score,
            "bridge_score": bridge_score,
            "target_score": target_score,
            "is_new_promotion": is_new_promotion,
        })

    if not scored:
        info["reason"] = "no_chain_candidates"
        return sorted_doc_ids, sorted_doc_scores, info

    scored.sort(key=lambda item: (-item["final_score"], -item["chain_score"], item["rank"]))
    new_top = list(protected_ids)
    seen = set(new_top)
    seen_entity_sets = []
    promoted = []

    for item in scored:
        if len(new_top) >= target_k:
            break
        if item["doc_id"] in seen:
            continue
        if item["is_new_promotion"] and len(promoted) >= max(0, max_promotions):
            continue
        if any(_entity_jaccard(item["entities"], prior) > 0.88 for prior in seen_entity_sets):
            continue

        new_top.append(item["doc_id"])
        seen.add(item["doc_id"])
        seen_entity_sets.append(item["entities"])
        if item["is_new_promotion"]:
            promoted.append(item)

    for doc_id in original_top:
        if len(new_top) >= target_k:
            break
        if doc_id in seen:
            continue
        new_top.append(doc_id)
        seen.add(doc_id)

    for doc_id in candidate_ids:
        if len(new_top) >= target_k:
            break
        if doc_id in seen:
            continue
        new_top.append(doc_id)
        seen.add(doc_id)

    remaining = [int(doc_id) for doc_id in sorted_doc_ids.tolist() if int(doc_id) not in seen]
    new_doc_ids = np.array(new_top + remaining, dtype=int)

    score_by_id = {item["doc_id"]: item["final_score"] for item in scored}
    original_score_by_id = {int(doc_id): float(score) for doc_id, score in zip(sorted_doc_ids.tolist(), sorted_doc_scores.tolist())}
    new_scores = [
        score_by_id.get(int(doc_id), original_score_by_id.get(int(doc_id), 0.0))
        for doc_id in new_doc_ids.tolist()
    ]

    old_top2 = set(sorted_doc_ids[:min(2, len(sorted_doc_ids))].tolist())
    old_top5 = set(sorted_doc_ids[:min(5, len(sorted_doc_ids))].tolist())
    new_top2 = set(new_doc_ids[:min(2, len(new_doc_ids))].tolist())
    new_top5 = set(new_doc_ids[:min(5, len(new_doc_ids))].tolist())

    info.update({
        "applied": True,
        "reason": "applied",
        "candidate_count": top_n,
        "anchor_count": len(anchor_weights),
        "predicate_token_count": len(predicate_tokens),
        "promoted_to_top2": len(new_top2 - old_top2),
        "promoted_to_top5": len(new_top5 - old_top5),
        "promotions": [
            {
                "doc_id": item["doc_id"],
                "from_rank": item["rank"],
                "title": item["title"],
                "chain_score": round(item["chain_score"], 6),
                "final_score": round(item["final_score"], 6),
                "anchor_score": round(item["anchor_score"], 6),
                "novel_anchor_score": round(item["novel_anchor_score"], 6),
                "relation_score": round(item["relation_score"], 6),
                "entity_query_score": round(item["entity_query_score"], 6),
                "bridge_score": round(item["bridge_score"], 6),
                "target_score": round(item["target_score"], 6),
            }
            for item in promoted[:8]
        ],
    })
    return new_doc_ids, np.array(new_scores, dtype=float), info


def select_path_aware_extra_doc_ids(
    query: str,
    sorted_doc_ids: np.ndarray,
    sorted_doc_scores: np.ndarray,
    passage_node_keys: List[str],
    chunk_path_info: Dict[str, Dict[str, Any]],
    top_k_facts: List[Tuple],
    top_k_fact_indices: List[int],
    query_fact_scores: np.ndarray,
    qa_top_k: int = 5,
    candidate_top_k: int = 100,
    max_extra: int = 2,
    min_path_score: float = 0.22,
    enabled: bool = False,
    bridge_only: bool = True,
    has_bridge_cues: bool = False,
) -> Tuple[List[int], Dict[str, Any]]:
    """Select high-confidence path candidates without changing the retrieval ranking."""
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "bridge_cues": bool(has_bridge_cues),
        "candidate_top_k": candidate_top_k,
        "max_extra": max_extra,
        "min_path_score": min_path_score,
        "selected_count": 0,
        "selected": [],
    }

    if not enabled:
        return [], info
    if bridge_only and not has_bridge_cues:
        info["reason"] = "no_bridge_cues"
        return [], info
    if sorted_doc_ids is None or len(sorted_doc_ids) == 0:
        info["reason"] = "no_candidates"
        return [], info

    top_n = min(max(qa_top_k, candidate_top_k), len(sorted_doc_ids))
    if top_n <= qa_top_k:
        info["reason"] = "not_enough_candidates"
        return [], info

    anchor_weights, predicate_tokens = _fact_anchor_weights(top_k_facts, top_k_fact_indices, query_fact_scores)
    if not anchor_weights:
        info["reason"] = "no_entity_anchors"
        return [], info

    query_tokens = set(tokenize(query))
    candidate_ids = [int(doc_id) for doc_id in sorted_doc_ids[:top_n].tolist()]
    original_scores = [float(score) for score in sorted_doc_scores[:top_n].tolist()]
    normalized_original_scores = _normalize_values(original_scores)

    seed_k = min(max(1, qa_top_k), len(candidate_ids))
    base_doc_ids = set(candidate_ids[:seed_k])
    base_entities = set()
    for doc_id in candidate_ids[:seed_k]:
        chunk_info = _chunk_info_for_doc(doc_id, passage_node_keys, chunk_path_info)
        base_entities.update(chunk_info.get("entities", set()))

    anchor_entities = set(anchor_weights)
    anchor_total = sum(anchor_weights.values()) or 1.0
    selected = []
    seen_entity_sets = []

    for rank, doc_id in enumerate(candidate_ids):
        if doc_id in base_doc_ids:
            continue

        chunk_info = _chunk_info_for_doc(doc_id, passage_node_keys, chunk_path_info)
        entities = chunk_info.get("entities", set())
        entity_tokens = chunk_info.get("entity_tokens", set())
        relation_tokens = chunk_info.get("relation_tokens", set())
        if not entities:
            continue

        anchor_overlap_weight = sum(anchor_weights.get(entity, 0.0) for entity in entities & anchor_entities)
        anchor_score = min(1.0, anchor_overlap_weight / anchor_total)
        novel_anchor_overlap_weight = sum(
            anchor_weights.get(entity, 0.0)
            for entity in (entities - base_entities) & anchor_entities
        )
        novel_anchor_score = min(1.0, novel_anchor_overlap_weight / anchor_total)

        relation_score = max(
            _overlap_ratio(query_tokens, relation_tokens),
            _overlap_ratio(predicate_tokens, relation_tokens),
        )
        entity_query_score = _overlap_ratio(query_tokens, entity_tokens)
        base_overlap = _overlap_ratio(entities, base_entities)
        complement = min(1.0, len(entities - base_entities) / 8.0)
        bridge_score = math.sqrt(base_overlap * complement) if base_overlap > 0 and complement > 0 else 0.0
        rank_prior = 1.0 / (rank + 1)

        path_score = (
            0.36 * anchor_score
            + 0.24 * novel_anchor_score
            + 0.16 * relation_score
            + 0.12 * entity_query_score
            + 0.08 * bridge_score
            + 0.04 * normalized_original_scores[rank]
            + 0.02 * rank_prior
        )

        has_anchor_signal = anchor_score >= 0.25 or novel_anchor_score >= 0.05
        has_bridge_or_relation = bridge_score >= 0.12 or relation_score >= 0.15 or entity_query_score >= 0.18
        if path_score < min_path_score or not (has_anchor_signal and has_bridge_or_relation):
            continue

        if any(entities and len(entities & prior) / max(1, len(entities | prior)) > 0.85 for prior in seen_entity_sets):
            continue

        selected.append({
            "doc_id": doc_id,
            "rank": rank,
            "path_score": path_score,
            "anchor_score": anchor_score,
            "novel_anchor_score": novel_anchor_score,
            "relation_score": relation_score,
            "entity_query_score": entity_query_score,
            "bridge_score": bridge_score,
            "original_score": original_scores[rank],
        })
        seen_entity_sets.append(set(entities))

    selected.sort(key=lambda item: (-item["path_score"], item["rank"]))
    selected = selected[:max(0, max_extra)]
    selected_ids = [item["doc_id"] for item in selected]

    info.update({
        "applied": True,
        "reason": "selected" if selected else "no_high_confidence_path_candidates",
        "candidate_count": top_n,
        "anchor_count": len(anchor_weights),
        "predicate_token_count": len(predicate_tokens),
        "selected_count": len(selected),
        "selected": [
            {
                "doc_id": item["doc_id"],
                "rank": item["rank"],
                "path_score": round(item["path_score"], 6),
                "anchor_score": round(item["anchor_score"], 6),
                "novel_anchor_score": round(item["novel_anchor_score"], 6),
                "relation_score": round(item["relation_score"], 6),
                "entity_query_score": round(item["entity_query_score"], 6),
                "bridge_score": round(item["bridge_score"], 6),
                "original_score": round(item["original_score"], 6),
            }
            for item in selected
        ],
    })
    return selected_ids, info
