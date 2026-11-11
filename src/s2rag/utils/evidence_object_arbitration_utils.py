import re
from typing import Any, Dict, List, Tuple

from .answer_verifier_utils import _is_supported, _rank_candidates
from .candidate_answer_selector_utils import (
    _candidate_contains_question_anchor,
    _candidate_is_disallowed_date_range,
    _candidate_is_year_like_count_answer,
    _candidate_looks_like_secondary_version,
    _candidate_matches_question_unit,
    _candidate_matches_required_relation,
    _candidate_reuses_question_number,
    _has_main_numeric_temporal_request,
    _is_synthetic_doc,
    _required_relation_pattern,
)
from .eval_utils import normalize_answer
from .sentence_evidence_utils import infer_answer_target_type


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
ABSTAIN_RE = re.compile(
    r"\b(?:not specified|not provided|not mentioned|cannot determine|cannot be determined|"
    r"unknown|insufficient information|no information|not available)\b",
    re.IGNORECASE,
)
MULTI_ANSWER_RE = re.compile(r"(?:;|,|\band\b|\bor\b|/)")
KINSHIP_RE = re.compile(
    r"\b(?:father|mother|parent|parents|son|daughter|wife|husband|spouse|"
    r"brother|sister)\b",
    re.IGNORECASE,
)
ROLE_QUERY_RE = re.compile(
    r"\b(?:father|mother|parent|parents|son|daughter|wife|husband|spouse|"
    r"brother|sister|author|writer|director|composer|founder|owner|"
    r"president|minister|governor|coach|captain|leader)\b",
    re.IGNORECASE,
)
SIGNED_QUERY_RE = re.compile(r"\b(?:who|which\s+person).{0,80}\b(?:signed|signatory)\b", re.IGNORECASE)
WHERE_STUDY_RE = re.compile(r"\bwhere\b.{0,80}\b(?:study|studied|educated|attend|attended)\b", re.IGNORECASE)
STUDY_EVIDENCE_RE = re.compile(r"\b(?:studied|educated|attended|graduated|college|university|school)\b", re.IGNORECASE)
LOCATION_ANSWER_RE = re.compile(
    r"\b(?:college|university|school|city|town|village|county|state|province|"
    r"country|kingdom|republic|port|island|district|region)\b",
    re.IGNORECASE,
)


def _token_count(text: str) -> int:
    return len(TOKEN_RE.findall(str(text or "")))


def _new_meta(enabled: bool, before: str) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "changed": False,
        "action": "none" if enabled else "disabled",
        "reason": "disabled" if not enabled else "not_evaluated",
        "apply_rewrite": False,
        "shadow_changed": False,
        "shadow_action": "none",
        "shadow_after": before,
        "before": before,
        "after": before,
        "target_type": None,
        "candidate_count": 0,
        "raw_candidate_count": 0,
        "confidence": 0.0,
        "margin": 0.0,
    }


def _filter_candidates(
    candidates: List[Dict[str, Any]],
    question: str,
    target_type: str,
    min_question_overlap: float,
    require_role_constraint: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    relation_pattern = _required_relation_pattern(question)
    filtered: List[Dict[str, Any]] = []
    reject_counts: Dict[str, int] = {}
    for candidate in candidates:
        candidate_text = str(candidate.get("candidate", "") or "")
        role_ok, role_reason = _candidate_passes_role_constraint(
            candidate=candidate,
            question=question,
            target_type=target_type,
            require_role_constraint=require_role_constraint,
        )
        reason = ""
        if float(candidate.get("question_overlap", 0.0) or 0.0) < float(min_question_overlap):
            reason = "low_question_overlap"
        elif not role_ok:
            reason = role_reason
        elif _candidate_reuses_question_number(candidate_text, question):
            reason = "candidate_reuses_question_number"
        elif _candidate_is_year_like_count_answer(candidate_text, question, target_type):
            reason = "year_like_count_answer"
        elif _candidate_is_disallowed_date_range(candidate_text, question, target_type):
            reason = "disallowed_date_range"
        elif not _candidate_matches_required_relation(candidate, relation_pattern):
            reason = "missing_relation_cue"
        elif not _candidate_matches_question_unit(candidate, question, target_type):
            reason = "missing_question_unit"
        elif not _candidate_contains_question_anchor(candidate, question):
            reason = "missing_question_anchor"
        elif _candidate_looks_like_secondary_version(candidate, question):
            reason = "secondary_version_candidate"
        if reason:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            continue
        filtered.append(candidate)
    return filtered, reject_counts


def _candidate_is_question_anchor(candidate_text: str, question: str) -> bool:
    candidate_norm = normalize_answer(candidate_text)
    question_norm = normalize_answer(question)
    if not candidate_norm or not question_norm:
        return False
    candidate_tokens = TOKEN_RE.findall(candidate_norm)
    if not candidate_tokens or len(candidate_tokens) > 3:
        return False
    return re.search(rf"(?:^|\s){re.escape(candidate_norm)}(?:\s|$)", question_norm) is not None


def _candidate_passes_role_constraint(
    candidate: Dict[str, Any],
    question: str,
    target_type: str,
    require_role_constraint: bool,
) -> Tuple[bool, str]:
    if not require_role_constraint:
        return True, ""
    candidate_text = str(candidate.get("candidate", "") or "")
    title = str(candidate.get("title", "") or "")
    sentence = str(candidate.get("sentence_preview", "") or "")
    evidence_text = f"{title} {sentence}"
    question_text = str(question or "")

    if _candidate_is_question_anchor(candidate_text, question_text):
        return False, "candidate_is_question_anchor"

    if KINSHIP_RE.search(question_text) and not KINSHIP_RE.search(evidence_text):
        return False, "missing_kinship_role_cue"

    if SIGNED_QUERY_RE.search(question_text):
        if _candidate_is_question_anchor(candidate_text, question_text):
            return False, "signed_question_anchor_candidate"
        if re.search(r"\b(?:country|nation|government|political leadership)\b", sentence, re.IGNORECASE):
            if normalize_answer(candidate_text) in {"germany", "france", "britain", "united states"}:
                return False, "signed_question_collective_candidate"

    if WHERE_STUDY_RE.search(question_text):
        if not (STUDY_EVIDENCE_RE.search(evidence_text) or LOCATION_ANSWER_RE.search(candidate_text)):
            return False, "missing_study_location_cue"

    if ROLE_QUERY_RE.search(question_text) and target_type in {"person", "entity", "organization"}:
        if not ROLE_QUERY_RE.search(evidence_text) and _candidate_is_question_anchor(candidate_text, question_text):
            return False, "role_query_anchor_candidate"

    return True, ""


def _looks_verbose_or_multi_answer(answer: str, max_tokens: int) -> bool:
    text = str(answer or "").strip()
    if _token_count(text) > max_tokens:
        return True
    return bool(MULTI_ANSWER_RE.search(text) and _token_count(text) >= 4)


def _is_containment_variant(before: str, after: str) -> bool:
    before_norm = normalize_answer(before)
    after_norm = normalize_answer(after)
    if not before_norm or not after_norm or before_norm == after_norm:
        return False
    return (
        re.search(rf"(?:^|\s){re.escape(after_norm)}(?:\s|$)", before_norm) is not None
        or re.search(rf"(?:^|\s){re.escape(before_norm)}(?:\s|$)", after_norm) is not None
    )


def arbitrate_answer_with_evidence_objects(
    question: str,
    answer: str,
    qa_docs: List[str] = None,
    answer_style: str = "",
    enabled: bool = False,
    min_confidence: float = 0.82,
    generic_min_confidence: float = 0.88,
    min_margin: float = 0.12,
    min_support_gain: float = 0.18,
    min_question_overlap: float = 0.08,
    max_docs: int = 10,
    max_answer_tokens: int = 12,
    apply_rewrite: bool = False,
    require_role_constraint: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Conservatively arbitrate final answers against evidence-backed objects.

    This module is intentionally post-reader and non-retrieval. It targets the
    dominant failure modes where support is already visible but the reader picks
    the wrong object, overly verbose answer, or wrong numeric/date value.
    """
    before = str(answer or "").strip()
    meta = _new_meta(enabled=enabled, before=before)
    meta["apply_rewrite"] = bool(apply_rewrite)
    meta["require_role_constraint"] = bool(require_role_constraint)
    if not enabled or not before:
        return before, meta

    qa_docs = qa_docs or []
    source_docs = [doc for doc in qa_docs if not _is_synthetic_doc(doc)]
    target_type = infer_answer_target_type(question)
    meta["target_type"] = target_type
    if not source_docs:
        meta["reason"] = "no_source_docs"
        return before, meta

    is_numeric_temporal = target_type in {"date", "number"}
    if is_numeric_temporal and not _has_main_numeric_temporal_request(question, target_type):
        meta["reason"] = "not_main_numeric_temporal_question"
        return before, meta

    short_style = answer_style in {"", None, "short_factoid"}
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
    top = candidates[0] if candidates else None
    second_confidence = float(candidates[1].get("confidence", 0.0)) if len(candidates) > 1 else 0.0
    top_confidence = float(top.get("confidence", 0.0)) if top else 0.0
    margin = top_confidence - second_confidence

    meta.update({
        "answer_style": answer_style,
        "answer_supported": bool(answer_supported),
        "answer_support_score": round(float(answer_support_score), 4),
        "raw_candidate_count": len(raw_candidates),
        "candidate_count": len(candidates),
        "candidate_reject_counts": reject_counts,
        "candidates": candidates[:5],
        "confidence": round(top_confidence, 4),
        "margin": round(margin, 4),
        "verbose_or_multi": bool(verbose_or_multi),
    })

    if not top:
        meta["reason"] = "no_candidates"
        return before, meta

    after = str(top.get("candidate", "") or "").strip()
    if not after or normalize_answer(after) == normalize_answer(before):
        meta.update({
            "reason": "same_as_current",
            "selected_candidate": after,
            "after": before,
        })
        return before, meta

    confidence_floor = float(min_confidence) if is_numeric_temporal else float(generic_min_confidence)
    enough_margin = margin >= float(min_margin) or second_confidence <= 0.0
    support_gain = top_confidence - float(answer_support_score)
    is_abstain = bool(ABSTAIN_RE.search(before))
    containment_variant = _is_containment_variant(before, after)

    should_rewrite = False
    action = "kept"
    if top_confidence >= confidence_floor and enough_margin:
        if is_abstain:
            should_rewrite = True
            action = "abstain_to_supported_object"
        elif verbose_or_multi and support_gain >= max(0.06, float(min_support_gain) * 0.5):
            should_rewrite = True
            action = "verbose_to_supported_object"
        elif not answer_supported and support_gain >= 0.0:
            should_rewrite = True
            action = "unsupported_to_supported_object"
        elif is_numeric_temporal and support_gain >= float(min_support_gain):
            should_rewrite = True
            action = "numeric_temporal_object_rewrite"
        elif containment_variant and support_gain >= max(0.04, float(min_support_gain) * 0.5):
            should_rewrite = True
            action = "granularity_object_rewrite"

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
                "selected_candidate": after,
                "support_gain": round(support_gain, 4),
            })
            return before, meta
        meta.update({
            "changed": True,
            "action": action,
            "reason": action,
            "after": after,
            "selected_candidate": after,
            "support_gain": round(support_gain, 4),
        })
        return after, meta

    meta.update({
        "action": "kept",
        "reason": "kept_low_margin_or_supported",
        "after": before,
        "selected_candidate": after,
        "support_gain": round(support_gain, 4),
    })
    return before, meta
