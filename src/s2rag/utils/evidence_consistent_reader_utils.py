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
    _required_relation_pattern,
)
from .eval_utils import normalize_answer
from .sentence_evidence_utils import infer_answer_target_type


ABSTAIN_RE = re.compile(
    r"\b(?:not specified|not provided|not mentioned|cannot determine|cannot be determined|"
    r"unknown|insufficient information|no information|not available)\b",
    re.IGNORECASE,
)
FULL_DATE_ANSWER_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)
YEAR_ONLY_RE = re.compile(r"^\s*(?:1[5-9]\d{2}|20\d{2}|2100)\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _token_count(text: str) -> int:
    return len(TOKEN_RE.findall(str(text or "")))


def _new_meta(enabled: bool, before: str) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "changed": False,
        "action": "none" if enabled else "disabled",
        "before": before,
        "after": before,
        "confidence": 0.0,
        "reason": "disabled" if not enabled else "not_evaluated",
    }


def _is_short_reader_style(answer_style: str, answer: str, max_answer_tokens: int) -> bool:
    if answer_style in {"complete_phrase", "multi_slot"}:
        return False
    return _token_count(answer) <= max(1, int(max_answer_tokens))


def _is_granularity_rewrite(before: str, after: str, target_type: str) -> bool:
    before_norm = normalize_answer(before)
    after_norm = normalize_answer(after)
    if not before_norm or not after_norm or before_norm == after_norm:
        return False
    if not re.search(rf"(?:^|\s){re.escape(after_norm)}(?:\s|$)", before_norm):
        return False
    if target_type == "date" and FULL_DATE_ANSWER_RE.search(before) and YEAR_ONLY_RE.match(after):
        return False
    return _token_count(before) >= _token_count(after) + 2


def _candidate_passes_reader_guards(candidate: Dict[str, Any], question: str, target_type: str) -> Tuple[bool, str]:
    candidate_text = str(candidate.get("candidate", ""))
    relation_pattern = _required_relation_pattern(question)
    if _candidate_reuses_question_number(candidate_text, question):
        return False, "candidate_reuses_question_number"
    if _candidate_is_year_like_count_answer(candidate_text, question, target_type):
        return False, "year_like_count_answer"
    if _candidate_is_disallowed_date_range(candidate_text, question, target_type):
        return False, "disallowed_date_range"
    if not _candidate_matches_required_relation(candidate, relation_pattern):
        return False, "missing_relation_cue"
    if not _candidate_matches_question_unit(candidate, question, target_type):
        return False, "missing_question_unit"
    if not _candidate_contains_question_anchor(candidate, question):
        return False, "missing_question_anchor"
    if _candidate_looks_like_secondary_version(candidate, question):
        return False, "secondary_version_candidate"
    return True, "accepted"


def _filter_candidates(candidates: List[Dict[str, Any]], question: str, target_type: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    filtered: List[Dict[str, Any]] = []
    reject_counts: Dict[str, int] = {}
    for candidate in candidates:
        ok, reason = _candidate_passes_reader_guards(candidate, question, target_type)
        if ok:
            filtered.append(candidate)
        else:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
    return filtered, reject_counts


def refine_answer_with_evidence_consistent_reader(
    question: str,
    answer: str,
    qa_docs: List[str] = None,
    answer_style: str = "",
    enabled: bool = False,
    min_confidence: float = 0.84,
    min_margin: float = 0.10,
    generic_min_confidence: float = 0.92,
    min_support_gain: float = 0.18,
    max_docs: int = 8,
    max_answer_tokens: int = 10,
) -> Tuple[str, Dict[str, Any]]:
    """Conservatively rewrite short answers using evidence-consistent spans.

    The module is intentionally stricter than the older verifier: it requires
    candidate evidence to match the question anchor, relation cue, answer type,
    and numeric unit when such cues are present.
    """
    before = (answer or "").strip()
    meta = _new_meta(enabled=enabled, before=before)
    if not enabled or not before:
        return before, meta

    qa_docs = qa_docs or []
    target_type = infer_answer_target_type(question)
    short_reader_style = _is_short_reader_style(answer_style, before, max_answer_tokens)
    is_numeric_temporal = target_type in {"date", "number"}
    if is_numeric_temporal and not _has_main_numeric_temporal_request(question, target_type):
        meta.update({
            "reason": "not_main_numeric_temporal_question",
            "target_type": target_type,
            "answer_style": answer_style,
        })
        return before, meta
    if not short_reader_style and not is_numeric_temporal:
        meta.update({
            "reason": "non_short_answer_style",
            "target_type": target_type,
            "answer_style": answer_style,
        })
        return before, meta

    answer_supported, answer_support_score = _is_supported(
        question=question,
        answer=before,
        qa_docs=qa_docs,
        target_type=target_type,
        max_docs=max_docs,
    )
    raw_candidates = _rank_candidates(
        question=question,
        answer=before,
        qa_docs=qa_docs,
        target_type=target_type,
        max_docs=max_docs,
    )
    candidates, reject_counts = _filter_candidates(raw_candidates, question, target_type)
    top = candidates[0] if candidates else None
    second_confidence = float(candidates[1].get("confidence", 0.0)) if len(candidates) > 1 else 0.0
    top_confidence = float(top.get("confidence", 0.0)) if top else 0.0
    margin = top_confidence - second_confidence

    meta.update({
        "target_type": target_type,
        "answer_style": answer_style,
        "answer_supported": bool(answer_supported),
        "answer_support_score": round(float(answer_support_score), 4),
        "candidate_count": len(candidates),
        "raw_candidate_count": len(raw_candidates),
        "candidate_reject_counts": reject_counts,
        "candidates": candidates[:5],
        "margin": round(margin, 4),
    })

    if not top:
        meta["reason"] = "no_candidates"
        meta["action"] = "no_candidates"
        return before, meta

    after = str(top.get("candidate", "")).strip()
    is_abstain = bool(ABSTAIN_RE.search(before))
    enough_margin = margin >= float(min_margin) or second_confidence <= 0.0
    confidence_floor = float(min_confidence) if is_numeric_temporal else float(generic_min_confidence)
    answer_lacks_required_type = is_numeric_temporal and not re.search(r"\d", before)
    granularity_rewrite = _is_granularity_rewrite(before, after, target_type)

    should_rewrite = False
    action = "kept"
    if top_confidence >= confidence_floor and enough_margin:
        if is_abstain or answer_lacks_required_type:
            if not answer_supported:
                should_rewrite = True
                action = "evidence_reader_missing_type_rewrite"
        elif not answer_supported:
            should_rewrite = True
            action = "evidence_reader_unsupported_rewrite"
        elif granularity_rewrite:
            should_rewrite = True
            action = "evidence_reader_granularity_rewrite"

    if should_rewrite and normalize_answer(after) != normalize_answer(before):
        meta.update({
            "changed": True,
            "action": action,
            "reason": action,
            "after": after,
            "confidence": round(top_confidence, 4),
            "selected_candidate": after,
        })
        return after, meta

    meta.update({
        "action": "kept",
        "reason": "kept_low_gain_or_supported",
        "after": before,
        "confidence": round(top_confidence, 4),
        "selected_candidate": after,
    })
    return before, meta
