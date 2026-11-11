import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .answer_slot_binding_optimizer_utils import (
    _answer_slot_contract,
    _candidate_passes_location_granularity,
    _collect_extended_module_candidates,
    _incumbent_slot_state,
    _is_supported_format_or_alias_repair,
    _module_roots,
    _relation_entailment_state,
    _row_evidence_text,
)
from .answer_verifier_utils import _is_supported, _rank_candidates
from .candidate_answer_selector_utils import _has_main_numeric_temporal_request
from .evidence_grounded_answer_extractor_utils import (
    BOOLEAN_ANSWER_RE,
    BOOLEAN_QUESTION_RE,
    DATE_NUMBER_TYPES,
    _candidate_passes_answer_slot,
    _infer_answer_arity,
)
from .evidence_object_arbitration_utils import (
    _candidate_passes_role_constraint,
    _filter_candidates,
    _looks_verbose_or_multi_answer,
)
from .provenance_answer_lattice_utils import (
    _collect_from_rows,
    _materialize_lattice,
    _same_answer,
    _source_docs,
    _token_count,
)
from .provenance_typed_candidate_store_utils import (
    _answers_comentioned_in_base_evidence,
    _candidate_matches_slot_type,
    _rewrite_safety_block_reason,
    _row_cost_features,
    _row_role_proxy,
    _source_documents_contain_answer,
)
from .typed_evidence_table_utils import _build_typed_table


def _new_meta(enabled: bool, before: str) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "changed": False,
        "action": "none" if enabled else "disabled",
        "reason": "disabled" if not enabled else "not_evaluated",
        "before": before,
        "after": before,
        "target_type": None,
        "candidate_target_type": None,
        "answer_slot": None,
        "slot_type": None,
        "answer_supported": False,
        "answer_support_score": 0.0,
        "incumbent_slot_match": False,
        "incumbent_slot_reason": "",
        "candidate_count": 0,
        "slot_candidate_count": 0,
        "selected_candidate": "",
        "selected_slot_store_score": 0.0,
        "selected_support_score": 0.0,
        "selected_net_utility": 0.0,
        "selected_source_count": 0,
        "selected_provenance_count": 0,
        "selected_base_source_count": 0,
        "selected_base_provenance_count": 0,
        "selected_cross_module_votes": 0,
        "margin": 0.0,
        "support_gain": 0.0,
        "shadow_changed": False,
        "shadow_action": "none",
        "shadow_after": before,
    }


def arbitrate_answer_with_provenance_answer_slot_store(
    *,
    question: str,
    answer: str,
    qa_docs: Optional[Sequence[str]] = None,
    answer_style: Optional[str] = None,
    evidence_consistent_reader: Optional[Dict[str, Any]] = None,
    evidence_object_arbitration: Optional[Dict[str, Any]] = None,
    evidence_answer_arbiter: Optional[Dict[str, Any]] = None,
    typed_evidence_table: Optional[Dict[str, Any]] = None,
    contrastive_answer_arbitration: Optional[Dict[str, Any]] = None,
    provenance_answer_lattice: Optional[Dict[str, Any]] = None,
    provenance_typed_candidate_store: Optional[Dict[str, Any]] = None,
    evidence_grounded_answer_extractor: Optional[Dict[str, Any]] = None,
    query_answer_contract: Optional[Dict[str, Any]] = None,
    anchored_typed_candidate_table: Optional[Dict[str, Any]] = None,
    candidate_answer_selector: Optional[Dict[str, Any]] = None,
    relation_fact_join: Optional[Dict[str, Any]] = None,
    enabled: bool = False,
    apply_rewrite: bool = True,
    min_score: float = 0.80,
    generic_min_score: float = 0.86,
    min_slot_store_score: float = 0.44,
    min_margin: float = 0.04,
    min_support_gain: float = 0.05,
    alias_min_support_gain: float = 0.01,
    min_question_overlap: float = 0.02,
    max_docs: int = 10,
    max_answer_tokens: int = 14,
    max_candidates: int = 18,
    min_base_provenance: int = 1,
    min_source_diversity: int = 1,
    require_role_constraint: bool = True,
    protect_supported_answers: bool = True,
    min_relation_entailment_score: float = 0.20,
    min_relation_question_overlap: float = 0.10,
) -> Tuple[str, Dict[str, Any]]:
    """Materialize a provenance-constrained, slot-safe answer candidate store.

    The goal of this operator is different from retrieval reranking: retrieval
    stays fixed, while answer-side candidates from the reader and downstream
    arbiters are compiled into a small table with explicit slot, provenance,
    source-diversity, and relation-entailment features. Rewrites are admitted
    only when the candidate is both answer-slot compatible and provenance-safe.
    """
    before = str(answer or "").strip()
    meta = _new_meta(enabled=enabled, before=before)
    meta.update({
        "apply_rewrite": bool(apply_rewrite),
        "require_role_constraint": bool(require_role_constraint),
        "protect_supported_answers": bool(protect_supported_answers),
        "min_slot_store_score": float(min_slot_store_score),
        "min_base_provenance": int(min_base_provenance),
        "min_source_diversity": int(min_source_diversity),
        "min_relation_entailment_score": float(min_relation_entailment_score),
        "min_relation_question_overlap": float(min_relation_question_overlap),
    })
    if not enabled or not before:
        return before, meta

    source_docs = _source_docs(list(qa_docs or []))
    if not source_docs:
        meta["reason"] = "no_source_docs"
        return before, meta
    if BOOLEAN_ANSWER_RE.match(before) or BOOLEAN_QUESTION_RE.match(question or ""):
        meta["reason"] = "boolean_protected"
        return before, meta

    contract = _answer_slot_contract(question or "", answer_style)
    target_type = str(contract["target_type"])
    candidate_target_type = str(contract["candidate_target_type"])
    answer_slot = str(contract["answer_slot"])
    slot_type = str(contract["slot_type"])
    is_numeric_temporal = candidate_target_type in DATE_NUMBER_TYPES
    meta.update(contract)
    if is_numeric_temporal and not _has_main_numeric_temporal_request(question, candidate_target_type):
        meta["reason"] = "not_main_numeric_temporal_question"
        return before, meta

    answer_arity = _infer_answer_arity(question or "", before, candidate_target_type)
    meta["answer_arity"] = answer_arity
    if answer_arity not in {"single", "multi_value"}:
        meta["reason"] = "multi_slot_or_list_protected"
        return before, meta

    short_style = answer_style in {"", None, "short_factoid", "multi_slot"}
    verbose_or_multi = _looks_verbose_or_multi_answer(before, max_answer_tokens)
    if not short_style and not verbose_or_multi and not is_numeric_temporal:
        meta.update({"reason": "non_short_answer_style", "answer_style": answer_style})
        return before, meta

    answer_supported, answer_support_score = _is_supported(
        question=question,
        answer=before,
        qa_docs=source_docs,
        target_type=candidate_target_type,
        max_docs=max_docs,
    )
    incumbent_slot_ok, incumbent_slot_reason = _incumbent_slot_state(
        question=question or "",
        before=before,
        answer_slot=answer_slot,
        candidate_target_type=candidate_target_type,
    )
    raw_candidates = _rank_candidates(
        question=question,
        answer=before,
        qa_docs=source_docs,
        target_type=candidate_target_type,
        max_docs=max_docs,
    )
    filtered_candidates, reject_counts = _filter_candidates(
        raw_candidates,
        question=question,
        target_type=candidate_target_type,
        min_question_overlap=min_question_overlap,
        require_role_constraint=require_role_constraint,
    )
    evidence_rows = _build_typed_table(filtered_candidates)

    lattice: Dict[str, Dict[str, Any]] = {}
    _collect_from_rows(lattice, filtered_candidates, "reader_evidence_candidates")
    _collect_from_rows(lattice, evidence_rows, "typed_table_from_reader")
    _collect_extended_module_candidates(lattice, "evidence_consistent_reader", evidence_consistent_reader)
    _collect_extended_module_candidates(lattice, "evidence_object_arbitration", evidence_object_arbitration)
    _collect_extended_module_candidates(lattice, "evidence_answer_arbiter", evidence_answer_arbiter)
    _collect_extended_module_candidates(lattice, "typed_evidence_table", typed_evidence_table)
    _collect_extended_module_candidates(lattice, "contrastive_answer_arbitration", contrastive_answer_arbitration)
    _collect_extended_module_candidates(lattice, "provenance_answer_lattice", provenance_answer_lattice)
    _collect_extended_module_candidates(
        lattice,
        "provenance_typed_candidate_store",
        provenance_typed_candidate_store,
    )
    _collect_extended_module_candidates(
        lattice,
        "evidence_grounded_answer_extractor",
        evidence_grounded_answer_extractor,
    )
    _collect_extended_module_candidates(lattice, "query_answer_contract", query_answer_contract)
    _collect_extended_module_candidates(
        lattice,
        "anchored_typed_candidate_table",
        anchored_typed_candidate_table,
    )
    _collect_extended_module_candidates(
        lattice,
        "candidate_answer_selector",
        candidate_answer_selector,
    )
    _collect_from_rows(
        lattice,
        (relation_fact_join or {}).get("candidate_table_rows") or [],
        "relation_fact_join",
    )
    _collect_from_rows(
        lattice,
        (relation_fact_join or {}).get("reader_candidate_table_rows") or [],
        "relation_fact_join_reader_projection",
    )

    rows = _materialize_lattice(lattice, current_answer=before, min_sources=1)
    scored_rows: List[Dict[str, Any]] = []
    slot_reject_counts: Counter = Counter()
    role_reject_counts: Counter = Counter()
    for row in rows:
        candidate = str(row.get("candidate") or "").strip()
        if not candidate or _same_answer(before, candidate):
            continue
        slot_ok, slot_reason = _candidate_passes_answer_slot(
            candidate,
            answer_slot,
            candidate_target_type,
            question=question or "",
            evidence_text=_row_evidence_text(row),
        )
        broad_slot_ok, broad_slot_reason = _candidate_matches_slot_type(candidate, row, slot_type)
        role_target_type = candidate_target_type if slot_type == "nationality" else slot_type
        role_ok, role_reason = _candidate_passes_role_constraint(
            candidate=_row_role_proxy(row),
            question=question or "",
            target_type=role_target_type,
            require_role_constraint=require_role_constraint,
        )
        if not slot_ok:
            slot_reject_counts[slot_reason] += 1
            continue
        if not broad_slot_ok:
            slot_reject_counts[broad_slot_reason] += 1
            continue
        granularity_ok, granularity_reason = _candidate_passes_location_granularity(
            before=before,
            candidate=candidate,
            answer_slot=answer_slot,
        )
        if not granularity_ok:
            slot_reject_counts[granularity_reason] += 1
            continue
        if not role_ok:
            role_reject_counts[role_reason] += 1
            continue

        features = _row_cost_features(
            row,
            question=question or "",
            source_docs=source_docs,
            target_type=candidate_target_type,
            slot_type=slot_type,
            answer_support_score=float(answer_support_score or 0.0),
            max_docs=max_docs,
            max_answer_tokens=max_answer_tokens,
            token_weight=0.06,
            provenance_weight=0.030,
            risk_weight=0.62,
            current_answer_supported=bool(answer_supported),
        )
        module_roots = _module_roots(row)
        cross_module_votes = len(module_roots)
        candidate_in_base_docs = _source_documents_contain_answer(source_docs, candidate)
        base_comentioned = _answers_comentioned_in_base_evidence(row, before, candidate)
        support_score = float(features.get("support_score", 0.0) or 0.0)
        net_utility = float(features.get("net_utility", 0.0) or 0.0)
        relation_score = float(features.get("relation_score", 0.0) or 0.0)
        type_signal = float(features.get("type_signal", 0.0) or 0.0)
        question_overlap = float(features.get("question_overlap", 0.0) or 0.0)
        base_source_count = int(features.get("base_source_count", 0) or 0)
        base_provenance_count = int(features.get("base_provenance_count", 0) or 0)
        relation_entailment_ok, relation_entailment_reason, relation_entailment = _relation_entailment_state(
            question=question or "",
            before=before,
            candidate=candidate,
            answer_slot=answer_slot,
            candidate_target_type=candidate_target_type,
            row=row,
            relation_score=relation_score,
            question_overlap=question_overlap,
            min_relation_score=min_relation_entailment_score,
            min_question_overlap=min_relation_question_overlap,
        )
        diversity_signal = min(1.0, base_source_count / max(1, int(min_source_diversity)))
        provenance_signal = min(1.0, base_provenance_count / max(1, int(min_base_provenance)))
        consensus_signal = min(1.0, cross_module_votes / 3.0)
        slot_store_score = (
            0.40 * support_score
            + 0.18 * net_utility
            + 0.10 * type_signal
            + 0.08 * relation_score
            + 0.05 * question_overlap
            + 0.07 * diversity_signal
            + 0.07 * provenance_signal
            + 0.05 * consensus_signal
        )
        features.update({
            "candidate": candidate,
            "slot_store_score": round(max(0.0, min(1.30, slot_store_score)), 4),
            "module_roots": module_roots,
            "cross_module_votes": cross_module_votes,
            "explicit_slot_match": bool(slot_ok),
            "broad_slot_match": bool(broad_slot_ok),
            "role_constraint_match": bool(role_ok),
            "relation_entailment_match": bool(relation_entailment_ok),
            "relation_entailment_reason": relation_entailment_reason,
            "relation_entailment": relation_entailment,
            "candidate_in_base_docs": bool(candidate_in_base_docs),
            "base_comentioned_with_current": bool(base_comentioned),
            "diversity_signal": round(diversity_signal, 4),
            "provenance_signal": round(provenance_signal, 4),
            "consensus_signal": round(consensus_signal, 4),
            "sources": row.get("sources", [])[:6],
            "titles": row.get("titles", [])[:5],
            "evidence": row.get("evidence", [])[:3],
        })
        scored_rows.append(features)

    scored_rows.sort(
        key=lambda row: (
            -float(row.get("slot_store_score", 0.0) or 0.0),
            -float(row.get("support_score", 0.0) or 0.0),
            -int(row.get("base_provenance_count", 0) or 0),
            -int(row.get("cross_module_votes", 0) or 0),
            str(row.get("candidate", "")).lower(),
        )
    )
    scored_rows = scored_rows[:max(1, int(max_candidates))]

    top = scored_rows[0] if scored_rows else None
    second = scored_rows[1] if len(scored_rows) > 1 else None
    after = str(top.get("candidate") or "").strip() if top else ""
    slot_store_score = float(top.get("slot_store_score", 0.0) or 0.0) if top else 0.0
    second_score = float((second or {}).get("slot_store_score", 0.0) or 0.0)
    margin = slot_store_score - second_score
    top_support = float(top.get("support_score", 0.0) or 0.0) if top else 0.0
    support_gain = top_support - float(answer_support_score or 0.0)
    candidate_in_base_docs = bool(top.get("candidate_in_base_docs")) if top else False
    base_source_count = int(top.get("base_source_count", 0) or 0) if top else 0
    base_provenance_count = int(top.get("base_provenance_count", 0) or 0) if top else 0
    cross_module_votes = int(top.get("cross_module_votes", 0) or 0) if top else 0
    relation_entailment_ok = bool(top.get("relation_entailment_match")) if top else False
    relation_entailment_reason = str(top.get("relation_entailment_reason") or "") if top else ""
    base_source_diversity_ok = base_source_count >= max(1, int(min_source_diversity))
    base_provenance_ok = base_provenance_count >= max(1, int(min_base_provenance))
    current_answer_in_base_docs = _source_documents_contain_answer(source_docs, before)
    format_or_alias_repair = bool(
        top
        and _is_supported_format_or_alias_repair(
            question=question or "",
            before=before,
            after=after,
            answer_slot=answer_slot,
            candidate_target_type=candidate_target_type,
            candidate_in_base_docs=candidate_in_base_docs,
            support_gain=support_gain,
            alias_min_support_gain=alias_min_support_gain,
        )
    )
    slot_violation_repair = bool(
        top
        and not incumbent_slot_ok
        and top.get("explicit_slot_match")
        and candidate_in_base_docs
        and support_gain >= float(min_support_gain)
    )
    unsupported_answer_repair = bool(
        top
        and not answer_supported
        and candidate_in_base_docs
        and support_gain >= float(min_support_gain)
    )
    source_diverse_disambiguation = bool(
        top
        and answer_supported
        and candidate_in_base_docs
        and not current_answer_in_base_docs
        and base_source_diversity_ok
        and base_provenance_ok
        and cross_module_votes >= 2
        and relation_entailment_ok
        and support_gain >= float(min_support_gain)
    )
    confidence_floor = float(min_score if is_numeric_temporal else generic_min_score)
    rewrite_safety_reason = _rewrite_safety_block_reason(
        question=question or "",
        before=before,
        after=after,
        target_type=candidate_target_type,
        answer_supported=bool(answer_supported),
    )
    protected_supported_answer = bool(
        protect_supported_answers
        and answer_supported
        and incumbent_slot_ok
        and not format_or_alias_repair
        and not source_diverse_disambiguation
    )

    meta.update({
        "answer_supported": bool(answer_supported),
        "answer_support_score": round(float(answer_support_score or 0.0), 4),
        "incumbent_slot_match": bool(incumbent_slot_ok),
        "incumbent_slot_reason": incumbent_slot_reason,
        "raw_candidate_count": len(raw_candidates),
        "filtered_candidate_count": len(filtered_candidates),
        "candidate_reject_counts": dict(reject_counts),
        "slot_reject_counts": dict(slot_reject_counts),
        "role_reject_counts": dict(role_reject_counts),
        "candidate_count": len(rows),
        "slot_candidate_count": len(scored_rows),
        "selected_candidate": after,
        "selected_slot_store_score": round(slot_store_score, 4),
        "selected_support_score": round(top_support, 4),
        "selected_net_utility": round(float(top.get("net_utility", 0.0) or 0.0), 4) if top else 0.0,
        "selected_source_count": int(top.get("source_count", 0) or 0) if top else 0,
        "selected_provenance_count": int(top.get("provenance_count", 0) or 0) if top else 0,
        "selected_base_source_count": base_source_count,
        "selected_base_provenance_count": base_provenance_count,
        "selected_cross_module_votes": cross_module_votes,
        "selected_candidate_in_base_docs": bool(candidate_in_base_docs),
        "selected_relation_entailment_match": bool(relation_entailment_ok),
        "selected_relation_entailment_reason": relation_entailment_reason,
        "selected_relation_entailment": top.get("relation_entailment", {}) if top else {},
        "selected_base_source_diversity_ok": bool(base_source_diversity_ok),
        "selected_base_provenance_ok": bool(base_provenance_ok),
        "margin": round(margin, 4),
        "support_gain": round(support_gain, 4),
        "rewrite_safety_reason": rewrite_safety_reason,
        "format_or_alias_repair": bool(format_or_alias_repair),
        "slot_violation_repair": bool(slot_violation_repair),
        "unsupported_answer_repair": bool(unsupported_answer_repair),
        "source_diverse_disambiguation": bool(source_diverse_disambiguation),
        "protected_supported_answer": bool(protected_supported_answer),
        "top_candidates": scored_rows[:6],
        "candidate_store_rows": scored_rows[:8],
        "answer_slot_store_rows": scored_rows[:8],
    })

    if not top:
        meta["reason"] = "no_candidates"
        return before, meta
    if not after or _same_answer(before, after):
        meta["reason"] = "same_as_current"
        return before, meta
    if not candidate_in_base_docs:
        meta["reason"] = "candidate_not_in_base_evidence"
        return before, meta
    if not base_source_diversity_ok and not (format_or_alias_repair or unsupported_answer_repair):
        meta["reason"] = "insufficient_base_source_diversity"
        return before, meta
    if not base_provenance_ok and not (format_or_alias_repair or unsupported_answer_repair):
        meta["reason"] = "insufficient_base_provenance"
        return before, meta
    if rewrite_safety_reason and not format_or_alias_repair:
        meta["reason"] = rewrite_safety_reason
        return before, meta
    if protected_supported_answer:
        meta["reason"] = "supported_slot_compatible_answer_protected"
        return before, meta
    if _token_count(after) > max_answer_tokens and candidate_target_type not in DATE_NUMBER_TYPES:
        meta["reason"] = "candidate_too_long"
        return before, meta
    if slot_store_score < float(min_slot_store_score):
        meta["reason"] = "low_slot_store_score"
        return before, meta
    if top_support < confidence_floor and not (format_or_alias_repair or slot_violation_repair):
        meta["reason"] = "low_support_score"
        return before, meta
    if margin < float(min_margin) and not (format_or_alias_repair or slot_violation_repair):
        meta["reason"] = "low_margin"
        return before, meta
    if not (
        format_or_alias_repair
        or slot_violation_repair
        or unsupported_answer_repair
        or source_diverse_disambiguation
    ):
        meta["reason"] = "no_safe_rewrite_case"
        return before, meta

    shadow_reason = (
        "format_or_alias_slot_store"
        if format_or_alias_repair
        else "slot_violation_slot_store"
        if slot_violation_repair
        else "unsupported_answer_slot_store"
        if unsupported_answer_repair
        else "source_diverse_slot_store"
    )
    meta.update({
        "shadow_changed": True,
        "shadow_action": "rewrite",
        "shadow_after": after,
        "shadow_reason": shadow_reason,
        "reason": shadow_reason,
    })
    if not apply_rewrite:
        meta["action"] = "shadow_rewrite"
        return before, meta

    meta.update({
        "changed": True,
        "action": "rewrite",
        "after": after,
    })
    return after, meta
