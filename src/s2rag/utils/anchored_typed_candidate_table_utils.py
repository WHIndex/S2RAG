import math
import re
from typing import Any, Dict, List, Tuple

from .answer_verifier_utils import _rank_candidates
from .candidate_answer_selector_utils import _has_main_numeric_temporal_request, _is_synthetic_doc
from .evidence_object_arbitration_utils import _filter_candidates
from .sentence_evidence_utils import infer_answer_target_type
from .typed_evidence_table_utils import _build_typed_table


TABLE_TITLE = "Anchored Typed Candidate Table"
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
BRIDGE_OR_CHAIN_RE = re.compile(
    r"\b(?:"
    r"both|same|compare|which of|"
    r"father|mother|grandfather|grandmother|spouse|wife|husband|son|daughter|"
    r"director|writer|author|composer|producer|creator|founder|designer|"
    r"where .+ (?:born|located|died|founded|created|released|published)|"
    r"when .+ (?:born|founded|created|released|published|abolished)|"
    r"(?:that|which|who|where) (?:was|were|is|are|has|have|had|located|based|born|died|created|released|published)|"
    r"of the (?:director|writer|author|composer|producer|creator|founder|designer|father|mother|spouse)"
    r")\b",
    re.IGNORECASE,
)
SIMPLE_ATTRIBUTE_RE = re.compile(
    r"^(?:what|who|where|when|in what|in which)\b.{0,160}\b(?:"
    r"occupation|profession|nationality|born|died|birthplace|"
    r"date of birth|date of death|place of birth|place of death|"
    r"located|headquartered|population|capital"
    r")\??$",
    re.IGNORECASE,
)
LONG_ANSWER_RE = re.compile(
    r"^(?:why|how did|how does|how do|what happened|what caused|describe|explain)\b",
    re.IGNORECASE,
)
BROAD_TARGET_TYPES = {"factoid", "person", "location", "work", "organization"}


def _new_info(enabled: bool, original_doc_count: int) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled" if not enabled else "not_evaluated",
        "target_type": None,
        "original_doc_count": int(original_doc_count),
        "final_doc_count": int(original_doc_count),
        "source_doc_count": 0,
        "raw_candidate_count": 0,
        "candidate_count": 0,
        "table_size": 0,
        "selected_count": 0,
        "insert_after": 0,
        "selected_candidates": [],
        "candidate_rows": [],
        "cost_model": {},
    }


def _format_table_doc(question: str, target_type: str, rows: List[Dict[str, Any]]) -> str:
    lines = [
        TABLE_TITLE,
        f"Question: {question}",
        f"Target answer type: {target_type}",
        "Namespace: derived_candidate_table; every row must be grounded by lineage_source from base passages.",
        "Use this table as candidate evidence only; select the final answer from supported passages.",
        "Rows:",
    ]
    for index, row in enumerate(rows, start=1):
        candidate = str(row.get("candidate", "") or "").strip()
        title = str(row.get("title", "") or "").strip()
        sentence = str(row.get("sentence_preview", "") or "").strip()
        lines.append(
            "- {idx}. candidate=\"{candidate}\" type={target_type} score={score:.4f} "
            "provenance={prov} question_overlap={overlap:.4f} relation_role={relation:.4f} "
            "namespace=derived_candidate_table lineage_source=\"{title}\" evidence=\"{sentence}\"".format(
                idx=index,
                candidate=candidate,
                target_type=target_type,
                score=float(row.get("table_score", 0.0) or 0.0),
                prov=int(row.get("provenance_count", 0) or 0),
                overlap=float(row.get("question_overlap", 0.0) or 0.0),
                relation=float(row.get("relation_role_score", 0.0) or 0.0),
                title=title[:120],
                sentence=sentence[:260],
            )
        )
    return "\n".join(lines)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(float(value)):
        return low
    return min(high, max(low, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _estimate_added_tokens(text: str) -> int:


    return max(1, int(len(TOKEN_RE.findall(str(text or ""))) * 1.25))


def _question_cost_profile(question: str, target_type: str) -> Dict[str, Any]:
    q = str(question or "").strip()
    q_lower = q.lower()
    bridge_matches = BRIDGE_OR_CHAIN_RE.findall(q)
    bridge_strength = _clamp(0.24 * len(bridge_matches))
    if re.search(r"\b(?:and|or)\b", q_lower) and len(TOKEN_RE.findall(q_lower)) >= 9:
        bridge_strength = _clamp(bridge_strength + 0.12)
    if re.search(r"\b(?:of|by|from|in)\b", q_lower) and target_type in BROAD_TARGET_TYPES:
        bridge_strength = _clamp(bridge_strength + 0.10)
    simple_attribute = bool(SIMPLE_ATTRIBUTE_RE.search(q)) and bridge_strength < 0.30
    long_answer = bool(LONG_ANSWER_RE.search(q))
    return {
        "bridge_strength": round(bridge_strength, 6),
        "bridge_cue_count": len(bridge_matches),
        "simple_attribute": simple_attribute,
        "long_answer": long_answer,
        "broad_target_type": target_type in BROAD_TARGET_TYPES,
    }


def _estimate_candidate_table_cost_decision(
    *,
    question: str,
    target_type: str,
    selected_rows: List[Dict[str, Any]],
    all_rows: List[Dict[str, Any]],
    table_doc: str,
    source_doc_count: int,
    raw_candidate_count: int,
    candidate_count: int,
    insert_after: int,
    enabled: bool,
    token_budget: int,
    min_utility: float,
    max_risk: float,
    min_net_utility: float,
    token_weight: float,
    compute_weight: float,
    risk_weight: float,
) -> Dict[str, Any]:
    row_scores = sorted(
        [_safe_float(row.get("table_score")) for row in selected_rows],
        reverse=True,
    )
    top_score = row_scores[0] if row_scores else 0.0
    second_score = row_scores[1] if len(row_scores) > 1 else 0.0
    score_margin = _clamp(top_score - second_score)
    max_overlap = max(
        [_safe_float(row.get("question_overlap")) for row in selected_rows] or [0.0]
    )
    max_relation = max(
        [_safe_float(row.get("relation_role_score")) for row in selected_rows] or [0.0]
    )
    max_provenance = max(
        [int(row.get("provenance_count", 0) or 0) for row in selected_rows] or [0]
    )
    provenance_signal = _clamp(math.log1p(max_provenance) / math.log(5.0))
    selected_count = len(selected_rows)
    table_size = len(all_rows)
    added_tokens = _estimate_added_tokens(table_doc)
    token_cost = _clamp(added_tokens / max(1.0, float(token_budget)))
    compute_cost = _clamp(
        0.04
        + 0.018 * min(max(0, source_doc_count), 12)
        + 0.0012 * min(max(0, raw_candidate_count), 240)
        + 0.004 * min(max(0, candidate_count), 80)
    )
    q_profile = _question_cost_profile(question, target_type)
    single_hop_bonus = 0.10 if q_profile["simple_attribute"] else 0.0
    type_bonus = 0.08 if target_type in {"date", "number", "occupation"} else 0.03
    coverage_bonus = _clamp(selected_count / 4.0) * 0.06
    utility = _clamp(
        0.38 * top_score
        + 0.13 * score_margin
        + 0.14 * max_overlap
        + 0.13 * max_relation
        + 0.10 * provenance_signal
        + type_bonus
        + single_hop_bonus
        + coverage_bonus
    )

    ambiguity_risk = _clamp((0.16 - score_margin) / 0.16) if selected_count > 1 else 0.0
    early_insert_risk = 0.16 if insert_after <= 2 else 0.06
    broad_bridge_risk = (
        0.18 if q_profile["broad_target_type"] and q_profile["bridge_strength"] >= 0.24 else 0.0
    )
    low_grounding_risk = 0.12 if max_overlap < 0.10 or max_relation < 0.10 else 0.0
    many_rows_risk = _clamp(max(0, selected_count - 2) / 4.0) * 0.08
    long_answer_risk = 0.20 if q_profile["long_answer"] else 0.0
    bridge_risk = 0.36 * _safe_float(q_profile["bridge_strength"])
    risk = _clamp(
        bridge_risk
        + ambiguity_risk * 0.16
        + early_insert_risk
        + broad_bridge_risk
        + low_grounding_risk
        + many_rows_risk
        + long_answer_risk
    )
    total_cost = _clamp(token_weight * token_cost + compute_weight * compute_cost)
    net_utility = utility - risk_weight * risk - total_cost
    reject_reasons = []
    if not enabled:
        reject_reasons.append("cost_model_disabled")
    if utility < min_utility:
        reject_reasons.append("utility_below_threshold")
    if risk > max_risk:
        reject_reasons.append("risk_above_threshold")
    if net_utility < min_net_utility:
        reject_reasons.append("net_utility_below_threshold")
    accepted = enabled and not reject_reasons
    return {
        "enabled": bool(enabled),
        "operator": "anchored_typed_candidate_table",
        "accepted": bool(accepted),
        "reason": "accepted" if accepted else "+".join(reject_reasons),
        "utility": round(float(utility), 6),
        "risk": round(float(risk), 6),
        "token_cost": round(float(token_cost), 6),
        "compute_cost": round(float(compute_cost), 6),
        "total_cost": round(float(total_cost), 6),
        "net_utility": round(float(net_utility), 6),
        "thresholds": {
            "min_utility": round(float(min_utility), 6),
            "max_risk": round(float(max_risk), 6),
            "min_net_utility": round(float(min_net_utility), 6),
            "token_budget": int(token_budget),
            "token_weight": round(float(token_weight), 6),
            "compute_weight": round(float(compute_weight), 6),
            "risk_weight": round(float(risk_weight), 6),
        },
        "features": {
            "target_type": target_type,
            "top_score": round(float(top_score), 6),
            "score_margin": round(float(score_margin), 6),
            "max_question_overlap": round(float(max_overlap), 6),
            "max_relation_role": round(float(max_relation), 6),
            "max_provenance": int(max_provenance),
            "selected_count": int(selected_count),
            "table_size": int(table_size),
            "estimated_added_tokens": int(added_tokens),
            "source_doc_count": int(source_doc_count),
            "raw_candidate_count": int(raw_candidate_count),
            "candidate_count": int(candidate_count),
            "insert_after": int(insert_after),
            **q_profile,
        },
    }


def build_anchored_typed_candidate_table(
    question: str,
    docs: List[str],
    enabled: bool = False,
    max_docs: int = 8,
    max_rows: int = 5,
    min_table_score: float = 0.72,
    min_question_overlap: float = 0.08,
    min_provenance: int = 1,
    require_role_constraint: bool = True,
    insert_after: int = 2,
    cost_model_enabled: bool = False,
    cost_model_token_budget: int = 220,
    cost_model_min_utility: float = 0.50,
    cost_model_max_risk: float = 0.55,
    cost_model_min_net_utility: float = 0.08,
    cost_model_token_weight: float = 0.18,
    cost_model_compute_weight: float = 0.08,
    cost_model_risk_weight: float = 0.62,
) -> Tuple[List[str], Dict[str, Any]]:
    """Append a compact, provenance-aware typed candidate table for the reader.

    This is a reader-evidence module, not a post-answer rewrite. It extracts
    candidate answers only from real source passages, keeps typed provenance
    and relation cues, and appends one synthetic table document to help the
    reader resolve cases where support is visible but the final span is wrong.
    """
    docs = list(docs or [])
    info = _new_info(enabled=enabled, original_doc_count=len(docs))
    info["insert_after"] = int(insert_after)
    if not enabled:
        return docs, info
    if not docs:
        info["reason"] = "no_docs"
        return docs, info

    target_type = infer_answer_target_type(question)
    info["target_type"] = target_type
    if target_type in {"date", "number"} and not _has_main_numeric_temporal_request(question, target_type):
        info["reason"] = "not_main_numeric_temporal_question"
        return docs, info

    source_docs = [doc for doc in docs if not _is_synthetic_doc(doc)]
    info["source_doc_count"] = len(source_docs)
    if not source_docs:
        info["reason"] = "no_source_docs"
        return docs, info

    raw_candidates = _rank_candidates(
        question=question,
        answer="",
        qa_docs=source_docs,
        target_type=target_type,
        max_docs=max_docs,
    )
    candidates, reject_counts = _filter_candidates(
        raw_candidates,
        question=question,
        target_type=target_type,
        min_question_overlap=min_question_overlap,
        require_role_constraint=require_role_constraint,
    )
    table_rows = _build_typed_table(candidates)
    selected_rows = [
        row
        for row in table_rows
        if float(row.get("table_score", 0.0) or 0.0) >= float(min_table_score)
        and int(row.get("provenance_count", 0) or 0) >= max(1, int(min_provenance))
    ][:max(1, int(max_rows))]

    info.update({
        "raw_candidate_count": len(raw_candidates),
        "candidate_count": len(candidates),
        "candidate_reject_counts": reject_counts,
        "table_size": len(table_rows),
        "selected_count": len(selected_rows),
        "candidate_rows": table_rows[:max(1, int(max_rows))],
        "selected_candidates": [row.get("candidate", "") for row in selected_rows],
    })

    if not raw_candidates:
        info["reason"] = "no_raw_candidates"
        return docs, info
    if not candidates:
        info["reason"] = "no_filtered_candidates"
        return docs, info
    if not selected_rows:
        info["reason"] = "no_high_confidence_rows"
        return docs, info

    table_doc = _format_table_doc(question=question, target_type=target_type, rows=selected_rows)
    cost_model = _estimate_candidate_table_cost_decision(
        question=question,
        target_type=target_type,
        selected_rows=selected_rows,
        all_rows=table_rows,
        table_doc=table_doc,
        source_doc_count=len(source_docs),
        raw_candidate_count=len(raw_candidates),
        candidate_count=len(candidates),
        insert_after=insert_after,
        enabled=cost_model_enabled,
        token_budget=cost_model_token_budget,
        min_utility=cost_model_min_utility,
        max_risk=cost_model_max_risk,
        min_net_utility=cost_model_min_net_utility,
        token_weight=cost_model_token_weight,
        compute_weight=cost_model_compute_weight,
        risk_weight=cost_model_risk_weight,
    )
    info["cost_model"] = cost_model
    info["cost_model_enabled"] = bool(cost_model_enabled)
    info["cost_model_accepted"] = bool(cost_model.get("accepted", False))
    info["cost_model_utility"] = cost_model.get("utility")
    info["cost_model_risk"] = cost_model.get("risk")
    info["cost_model_total_cost"] = cost_model.get("total_cost")
    info["cost_model_net_utility"] = cost_model.get("net_utility")
    if cost_model_enabled and not cost_model.get("accepted", False):
        info["reason"] = "cost_model_rejected"
        info["cost_model_reason"] = cost_model.get("reason")
        return docs, info

    position = max(0, min(len(docs), int(insert_after)))
    updated_docs = docs[:position] + [table_doc] + docs[position:]
    info.update({
        "applied": True,
        "reason": "table_inserted",
        "final_doc_count": len(updated_docs),
        "insert_after": position,
    })
    return updated_docs, info
