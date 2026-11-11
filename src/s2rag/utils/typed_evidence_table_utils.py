import re
from typing import Any, Dict, List, Tuple

from .answer_verifier_utils import _is_supported, _rank_candidates
from .candidate_answer_selector_utils import _has_main_numeric_temporal_request, _is_synthetic_doc
from .eval_utils import normalize_answer
from .evidence_object_arbitration_utils import (
    ABSTAIN_RE,
    _filter_candidates,
    _is_containment_variant,
    _looks_verbose_or_multi_answer,
)
from .sentence_evidence_utils import infer_answer_target_type


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _token_count(text: str) -> int:
    return len(TOKEN_RE.findall(str(text or "")))


def _same_answer(left: str, right: str) -> bool:
    return normalize_answer(left) == normalize_answer(right)


def _is_shorter_alias(before: str, after: str) -> bool:
    if not _is_containment_variant(before, after):
        return False
    return 0 < _token_count(after) < _token_count(before)


def _new_meta(enabled: bool, before: str) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "changed": False,
        "action": "none" if enabled else "disabled",
        "reason": "disabled" if not enabled else "not_evaluated",
        "before": before,
        "after": before,
        "target_type": None,
        "answer_supported": False,
        "answer_support_score": 0.0,
        "raw_candidate_count": 0,
        "candidate_count": 0,
        "table_size": 0,
        "confidence": 0.0,
        "raw_confidence": 0.0,
        "margin": 0.0,
        "support_gain": 0.0,
        "selected_candidate": "",
        "selected_provenance_count": 0,
        "shadow_changed": False,
        "shadow_action": "none",
        "shadow_after": before,
    }


def _build_typed_table(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        frequency = max(1, int(candidate.get("frequency", 1) or 1))
        raw_confidence = float(candidate.get("confidence", 0.0) or 0.0)
        relation_bonus = float(candidate.get("relation_bonus", 0.0) or 0.0)
        provenance_bonus = min(0.04, 0.015 * max(0, frequency - 1))
        schema_role_bonus = min(0.03, 0.20 * max(0.0, relation_bonus))
        table_score = max(0.0, min(0.99, raw_confidence + provenance_bonus + schema_role_bonus))
        rows.append({
            "candidate": candidate.get("candidate", ""),
            "key": candidate.get("key", ""),
            "answer_type_score": round(float(candidate.get("target_signal", 0.0) or 0.0), 4),
            "question_overlap": round(float(candidate.get("question_overlap", 0.0) or 0.0), 4),
            "relation_role_score": round(relation_bonus, 4),
            "provenance_count": frequency,
            "raw_confidence": round(raw_confidence, 4),
            "table_score": round(table_score, 4),
            "title": candidate.get("title", ""),
            "sentence_preview": candidate.get("sentence_preview", ""),
            "source_namespace": candidate.get("source_namespace", "base_passage"),
            "lineage_source_title": candidate.get("lineage_source_title", candidate.get("title", "")),
            "lineage_sentence_preview": candidate.get("lineage_sentence_preview", candidate.get("sentence_preview", "")),
        })
    rows.sort(
        key=lambda row: (
            -float(row.get("table_score", 0.0) or 0.0),
            -int(row.get("provenance_count", 1) or 1),
            str(row.get("candidate", "")).lower(),
        )
    )
    return rows


def arbitrate_answer_with_typed_evidence_table(
    question: str,
    answer: str,
    qa_docs: List[str] = None,
    answer_style: str = "",
    enabled: bool = False,
    apply_rewrite: bool = False,
    min_confidence: float = 0.86,
    generic_min_confidence: float = 0.92,
    min_margin: float = 0.12,
    min_support_gain: float = 0.12,
    alias_min_support_gain: float = 0.02,
    min_question_overlap: float = 0.08,
    max_docs: int = 10,
    max_answer_tokens: int = 12,
    min_provenance: int = 1,
    require_role_constraint: bool = True,
    skip_supported_answers: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Typed evidence-table answer arbitration.

    This rank-stable module builds a typed table of candidate answers from the
    final reader evidence and rewrites only when the table shows a strong,
    type-consistent, role-consistent candidate that better explains the
    retrieved evidence than the current answer.
    """
    before = str(answer or "").strip()
    meta = _new_meta(enabled=enabled, before=before)
    meta.update({
        "apply_rewrite": bool(apply_rewrite),
        "require_role_constraint": bool(require_role_constraint),
        "skip_supported_answers": bool(skip_supported_answers),
        "min_provenance": int(min_provenance),
    })
    if not enabled or not before:
        return before, meta

    source_docs = [doc for doc in (qa_docs or []) if not _is_synthetic_doc(doc)]
    if not source_docs:
        meta["reason"] = "no_source_docs"
        return before, meta

    target_type = infer_answer_target_type(question)
    is_numeric_temporal = target_type in {"date", "number"}
    meta["target_type"] = target_type

    if is_numeric_temporal and not _has_main_numeric_temporal_request(question, target_type):
        meta["reason"] = "not_main_numeric_temporal_question"
        return before, meta

    short_style = answer_style in {"", None, "short_factoid", "multi_slot"}
    verbose_or_multi = _looks_verbose_or_multi_answer(before, max_answer_tokens)
    if not short_style and not verbose_or_multi and not is_numeric_temporal:
        meta.update({
            "reason": "non_short_answer_style",
            "answer_style": answer_style,
        })
        return before, meta

    answer_supported, answer_support_score = _is_supported(
        question=question,
        answer=before,
        qa_docs=source_docs,
        target_type=target_type,
        max_docs=max_docs,
    )
    raw_candidates = _rank_candidates(
        question=question,
        answer=before,
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

    top = table_rows[0] if table_rows else None
    second_score = float(table_rows[1].get("table_score", 0.0)) if len(table_rows) > 1 else 0.0
    top_score = float(top.get("table_score", 0.0)) if top else 0.0
    raw_top_score = float(top.get("raw_confidence", 0.0)) if top else 0.0
    margin = top_score - second_score
    after = str(top.get("candidate", "") or "").strip() if top else ""
    support_gain = top_score - float(answer_support_score)
    provenance_count = int(top.get("provenance_count", 0) or 0) if top else 0

    is_abstain = bool(ABSTAIN_RE.search(before))
    containment_variant = bool(after and _is_containment_variant(before, after))
    shorter_alias = bool(after and _is_shorter_alias(before, after))
    same_answer = bool(after and _same_answer(before, after))

    meta.update({
        "answer_style": answer_style,
        "answer_supported": bool(answer_supported),
        "answer_support_score": round(float(answer_support_score), 4),
        "raw_candidate_count": len(raw_candidates),
        "candidate_count": len(candidates),
        "candidate_reject_counts": reject_counts,
        "table_size": len(table_rows),
        "table_rows": table_rows[:5],
        "selected_candidate": after,
        "selected_provenance_count": provenance_count,
        "confidence": round(top_score, 4),
        "raw_confidence": round(raw_top_score, 4),
        "margin": round(margin, 4),
        "support_gain": round(support_gain, 4),
        "verbose_or_multi": bool(verbose_or_multi),
        "containment_variant": bool(containment_variant),
        "shorter_alias": bool(shorter_alias),
        "is_abstain": bool(is_abstain),
    })

    if not top:
        meta["reason"] = "no_candidates"
        return before, meta
    if not after or same_answer:
        meta["reason"] = "same_as_current"
        return before, meta
    if provenance_count < max(1, int(min_provenance)):
        meta["reason"] = "insufficient_provenance"
        return before, meta

    confidence_floor = float(min_confidence) if is_numeric_temporal else float(generic_min_confidence)
    enough_margin = margin >= float(min_margin) or second_score <= 0.0
    strong_candidate = top_score >= confidence_floor and enough_margin

    should_rewrite = False
    action = "kept"
    if strong_candidate:
        if is_abstain:
            should_rewrite = True
            action = "abstain_to_typed_evidence"
        elif not answer_supported and support_gain >= max(0.0, float(min_support_gain) - 0.04):
            should_rewrite = True
            action = "unsupported_to_typed_evidence"
        elif verbose_or_multi and support_gain >= max(0.05, float(min_support_gain) * 0.5):
            should_rewrite = True
            action = "verbose_to_typed_evidence"
        elif shorter_alias and support_gain >= float(alias_min_support_gain):
            should_rewrite = True
            action = "granularity_alias_rewrite"
        elif is_numeric_temporal and not answer_supported and support_gain >= max(0.04, float(min_support_gain) * 0.5):
            should_rewrite = True
            action = "numeric_temporal_typed_evidence"

    if answer_supported and skip_supported_answers and not (
        is_abstain or verbose_or_multi or shorter_alias or (is_numeric_temporal and support_gain >= float(min_support_gain))
    ):
        should_rewrite = False
        action = "supported_answer_protected"

    if should_rewrite:
        meta.update({
            "shadow_changed": True,
            "shadow_action": action,
            "shadow_after": after,
            "shadow_reason": action,
        })
        if not apply_rewrite:
            meta.update({
                "changed": False,
                "action": "shadow_kept",
                "reason": "shadow_only",
                "after": before,
            })
            return before, meta
        meta.update({
            "changed": True,
            "action": action,
            "reason": action,
            "after": after,
        })
        return after, meta

    meta.update({
        "action": "kept",
        "reason": action if action != "kept" else "kept_low_confidence_or_margin",
        "after": before,
    })
    return before, meta
