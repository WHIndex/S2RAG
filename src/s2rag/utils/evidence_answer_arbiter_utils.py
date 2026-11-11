import re
from typing import Any, Dict, List, Tuple

from .answer_verifier_utils import _is_supported, _rank_candidates
from .candidate_answer_selector_utils import (
    _has_main_numeric_temporal_request,
    _is_synthetic_doc,
)
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
        "candidate_count": 0,
        "raw_candidate_count": 0,
        "confidence": 0.0,
        "margin": 0.0,
        "support_gain": 0.0,
        "shadow_changed": False,
        "shadow_after": before,
        "shadow_action": "none",
    }


def _answer_is_too_long(answer: str, max_answer_tokens: int) -> bool:
    return _token_count(answer) > max(1, int(max_answer_tokens))


def _same_answer(a: str, b: str) -> bool:
    return normalize_answer(a) == normalize_answer(b)


def _is_shorter_alias(before: str, after: str) -> bool:
    if not _is_containment_variant(before, after):
        return False
    return 0 < _token_count(after) < _token_count(before)


def arbitrate_answer_with_evidence_grounding(
    question: str,
    answer: str,
    qa_docs: List[str] = None,
    answer_style: str = "",
    enabled: bool = False,
    apply_rewrite: bool = False,
    min_confidence: float = 0.84,
    generic_min_confidence: float = 0.92,
    min_margin: float = 0.14,
    min_support_gain: float = 0.18,
    alias_min_support_gain: float = 0.04,
    min_question_overlap: float = 0.08,
    max_docs: int = 10,
    max_answer_tokens: int = 12,
    require_role_constraint: bool = True,
    skip_supported_answers: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Evidence-grounded final answer arbitration.

    The arbiter is intentionally rank-stable: it never changes retrieved
    passages. It builds a small answer-object lattice from the final reader
    evidence and only rewrites when the evidence state strongly suggests that
    the reader output is unsupported, over-granular, verbose, or type-mismatched.
    """
    before = str(answer or "").strip()
    meta = _new_meta(enabled=enabled, before=before)
    meta["apply_rewrite"] = bool(apply_rewrite)
    meta["require_role_constraint"] = bool(require_role_constraint)
    meta["skip_supported_answers"] = bool(skip_supported_answers)
    if not enabled or not before:
        return before, meta

    docs = [doc for doc in (qa_docs or []) if not _is_synthetic_doc(doc)]
    if not docs:
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
    too_long = _answer_is_too_long(before, max_answer_tokens)
    if not short_style and not verbose_or_multi and not too_long and not is_numeric_temporal:
        meta.update({
            "reason": "non_short_answer_style",
            "answer_style": answer_style,
        })
        return before, meta

    answer_supported, answer_support_score = _is_supported(
        question=question,
        answer=before,
        qa_docs=docs,
        target_type=target_type,
        max_docs=max_docs,
    )
    raw_candidates = _rank_candidates(
        question=question,
        answer=before,
        qa_docs=docs,
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

    top = candidates[0] if candidates else None
    second_confidence = float(candidates[1].get("confidence", 0.0)) if len(candidates) > 1 else 0.0
    top_confidence = float(top.get("confidence", 0.0)) if top else 0.0
    margin = top_confidence - second_confidence
    after = str(top.get("candidate", "") or "").strip() if top else ""
    support_gain = top_confidence - float(answer_support_score)

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
        "candidates": candidates[:5],
        "selected_candidate": after,
        "confidence": round(top_confidence, 4),
        "margin": round(margin, 4),
        "support_gain": round(support_gain, 4),
        "verbose_or_multi": bool(verbose_or_multi),
        "too_long": bool(too_long),
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

    confidence_floor = float(min_confidence) if is_numeric_temporal else float(generic_min_confidence)
    enough_margin = margin >= float(min_margin) or second_confidence <= 0.0
    strong_candidate = top_confidence >= confidence_floor and enough_margin

    should_rewrite = False
    action = "kept"
    if strong_candidate:
        if is_abstain:
            should_rewrite = True
            action = "abstain_to_evidence_object"
        elif not answer_supported and support_gain >= 0.0:
            should_rewrite = True
            action = "unsupported_to_evidence_object"
        elif verbose_or_multi and support_gain >= max(0.06, float(min_support_gain) * 0.5):
            should_rewrite = True
            action = "verbose_to_evidence_object"
        elif shorter_alias and support_gain >= float(alias_min_support_gain):
            should_rewrite = True
            action = "granularity_alias_rewrite"
        elif is_numeric_temporal and support_gain >= float(min_support_gain):
            should_rewrite = True
            action = "numeric_temporal_evidence_object"
        elif not skip_supported_answers and support_gain >= float(min_support_gain) + 0.08:
            should_rewrite = True
            action = "high_gain_supported_answer_rewrite"

    if answer_supported and skip_supported_answers and not (
        is_abstain or verbose_or_multi or shorter_alias or is_numeric_temporal
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
