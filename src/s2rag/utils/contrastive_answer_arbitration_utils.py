import re
from typing import Any, Dict, List, Tuple

from .answer_verifier_utils import OCCUPATION_HEAD_RE, _is_supported, _rank_candidates
from .candidate_answer_selector_utils import _has_main_numeric_temporal_request, _is_synthetic_doc
from .eval_utils import normalize_answer
from .evidence_object_arbitration_utils import _filter_candidates, _is_containment_variant, _looks_verbose_or_multi_answer
from .sentence_evidence_utils import infer_answer_target_type
from .typed_evidence_table_utils import _build_typed_table


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
QUERY_ANCHOR_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "how", "in",
    "is", "it", "its", "of", "on", "or", "s", "she", "that", "the",
    "their", "there", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "whom", "whose", "why", "with",
}
DATE_VALUE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})|"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?))\b",
    re.IGNORECASE,
)
NUMBER_VALUE_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")


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
        "challenger_count": 0,
        "selected_challenger": "",
        "challenger_score": 0.0,
        "incumbent_score": 0.0,
        "contrast_gain": 0.0,
        "margin": 0.0,
        "shadow_changed": False,
        "shadow_action": "none",
        "shadow_after": before,
    }


def _norm_contains(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return re.search(rf"(?:^|\s){re.escape(needle)}(?:\s|$)", haystack) is not None


def _content_tokens(text: str) -> List[str]:
    return [
        token
        for token in TOKEN_RE.findall(normalize_answer(text))
        if len(token) > 1 and token not in QUERY_ANCHOR_STOPWORDS
    ]


def _candidate_is_query_anchor(candidate: str, question: str) -> bool:
    candidate_norm = normalize_answer(candidate)
    question_norm = normalize_answer(question)
    if not candidate_norm or not question_norm:
        return False
    question_no_possessive = normalize_answer(re.sub(r"['’]s\b", " ", str(question or ""), flags=re.IGNORECASE))
    if _norm_contains(question_no_possessive, candidate_norm):
        return True
    candidate_tokens = TOKEN_RE.findall(candidate_norm)
    if len(candidate_tokens) >= 2:
        if _norm_contains(question_norm, candidate_norm):
            return True
        candidate_content = set(_content_tokens(candidate))
        question_content = set(_content_tokens(question))
        if candidate_content and candidate_content.issubset(question_content):
            return True
    if len(candidate_norm) >= 8:
        return _norm_contains(question_norm, candidate_norm)
    return False


def _same_answer(left: str, right: str) -> bool:
    return normalize_answer(left) == normalize_answer(right)


def _candidate_matches_target_type(candidate: str, row: Dict[str, Any], target_type: str) -> bool:
    candidate = str(candidate or "").strip()
    if not candidate:
        return False
    if target_type == "date":
        return bool(DATE_VALUE_RE.search(candidate))
    if target_type == "number":
        return bool(NUMBER_VALUE_RE.search(candidate))
    if target_type == "occupation":
        return bool(OCCUPATION_HEAD_RE.search(candidate))
    return True


def _filter_challengers(
    rows: List[Dict[str, Any]],
    question: str,
    incumbent: str,
    target_type: str,
    min_provenance: int,
    allow_alias_challengers: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    filtered: List[Dict[str, Any]] = []
    reject_counts: Dict[str, int] = {}
    for row in rows:
        candidate = str(row.get("candidate", "") or "").strip()
        reason = ""
        if not candidate:
            reason = "empty_candidate"
        elif _same_answer(candidate, incumbent):
            reason = "same_as_incumbent"
        elif _candidate_is_query_anchor(candidate, question):
            reason = "candidate_is_query_anchor"
        elif not _candidate_matches_target_type(candidate, row, target_type):
            reason = "target_type_mismatch"
        elif not allow_alias_challengers and _is_containment_variant(incumbent, candidate):
            reason = "containment_alias_challenger"
        elif int(row.get("provenance_count", 0) or 0) < max(1, int(min_provenance)):
            reason = "insufficient_provenance"
        if reason:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            continue
        filtered.append(row)
    return filtered, reject_counts


def arbitrate_answer_with_contrastive_candidates(
    question: str,
    answer: str,
    qa_docs: List[str] = None,
    answer_style: str = "",
    enabled: bool = False,
    apply_rewrite: bool = False,
    min_confidence: float = 0.90,
    generic_min_confidence: float = 0.94,
    min_contrast_gain: float = 0.30,
    min_margin: float = 0.14,
    min_question_overlap: float = 0.08,
    max_docs: int = 10,
    max_answer_tokens: int = 12,
    min_provenance: int = 2,
    require_role_constraint: bool = True,
    require_supported_incumbent: bool = True,
    allow_alias_challengers: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Contrast the current answer against competing typed evidence candidates."""
    before = str(answer or "").strip()
    meta = _new_meta(enabled=enabled, before=before)
    meta.update({
        "apply_rewrite": bool(apply_rewrite),
        "require_role_constraint": bool(require_role_constraint),
        "require_supported_incumbent": bool(require_supported_incumbent),
        "allow_alias_challengers": bool(allow_alias_challengers),
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
        meta.update({"reason": "non_short_answer_style", "answer_style": answer_style})
        return before, meta

    answer_supported, answer_support_score = _is_supported(
        question=question,
        answer=before,
        qa_docs=source_docs,
        target_type=target_type,
        max_docs=max_docs,
    )
    if require_supported_incumbent and not answer_supported:
        meta.update({
            "reason": "unsupported_incumbent_defer_to_typed_table",
            "answer_supported": False,
            "answer_support_score": round(float(answer_support_score), 4),
            "incumbent_score": round(float(answer_support_score), 4),
        })
        return before, meta

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
    challengers, challenger_reject_counts = _filter_challengers(
        rows=table_rows,
        question=question,
        incumbent=before,
        target_type=target_type,
        min_provenance=min_provenance,
        allow_alias_challengers=allow_alias_challengers,
    )

    top = challengers[0] if challengers else None
    second_score = float(challengers[1].get("table_score", 0.0)) if len(challengers) > 1 else 0.0
    challenger_score = float(top.get("table_score", 0.0)) if top else 0.0
    margin = challenger_score - second_score
    incumbent_score = float(answer_support_score)
    contrast_gain = challenger_score - incumbent_score
    after = str(top.get("candidate", "") or "").strip() if top else ""

    meta.update({
        "answer_style": answer_style,
        "answer_supported": bool(answer_supported),
        "answer_support_score": round(float(answer_support_score), 4),
        "raw_candidate_count": len(raw_candidates),
        "candidate_count": len(candidates),
        "candidate_reject_counts": reject_counts,
        "challenger_count": len(challengers),
        "challenger_reject_counts": challenger_reject_counts,
        "challengers": challengers[:5],
        "selected_challenger": after,
        "selected_provenance_count": int(top.get("provenance_count", 0) or 0) if top else 0,
        "challenger_score": round(challenger_score, 4),
        "incumbent_score": round(incumbent_score, 4),
        "contrast_gain": round(contrast_gain, 4),
        "margin": round(margin, 4),
        "verbose_or_multi": bool(verbose_or_multi),
    })

    if not top:
        meta["reason"] = "no_challengers"
        return before, meta

    confidence_floor = float(min_confidence) if is_numeric_temporal else float(generic_min_confidence)
    enough_margin = margin >= float(min_margin) or second_score <= 0.0
    strong_challenger = challenger_score >= confidence_floor
    sufficient_gain = contrast_gain >= float(min_contrast_gain)
    if not (strong_challenger and enough_margin and sufficient_gain):
        meta["reason"] = "kept_low_contrast_or_margin"
        return before, meta

    action = "contrastive_challenger_rewrite"
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
