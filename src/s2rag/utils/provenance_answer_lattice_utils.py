import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .answer_verifier_utils import _is_supported, _rank_candidates
from .candidate_answer_selector_utils import _has_main_numeric_temporal_request, _is_synthetic_doc
from .evidence_grounded_answer_extractor_utils import (
    BOOLEAN_ANSWER_RE,
    BOOLEAN_QUESTION_RE,
)
from .eval_utils import normalize_answer
from .evidence_object_arbitration_utils import (
    ABSTAIN_RE,
    _filter_candidates,
    _is_containment_variant,
    _looks_verbose_or_multi_answer,
)
from .sentence_evidence_utils import infer_answer_target_type
from .typed_evidence_table_utils import _build_typed_table


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
LIST_SEPARATOR_RE = re.compile(r"[,;]|\band\b", re.IGNORECASE)
SYNTHETIC_TITLE_KEYS = {
    "anchored typed candidate table",
    "candidate answer contrastive selector",
    "typed evidence grounding",
    "directional fact join candidate table",
    "entity state evidence control",
    "multi granular evidence memory",
    "structured evidence chain",
    "slot separated fact join view",
    "candidate answer table",
    "evidence snippets from",
}


def _token_count(text: str) -> int:
    return len(TOKEN_RE.findall(str(text or "")))


def _is_multi_answer_like(text: str) -> bool:
    text = str(text or "")
    return bool(LIST_SEPARATOR_RE.search(text)) and _token_count(text) >= 3


def _same_answer(left: str, right: str) -> bool:
    return normalize_answer(left) == normalize_answer(right)


def _title_key(title: str) -> str:
    return normalize_answer(title)


def _is_synthetic_title(title: str) -> bool:
    key = _title_key(title)
    return not key or any(key.startswith(prefix) for prefix in SYNTHETIC_TITLE_KEYS)


def _sentence_key(title: str, sentence: str) -> str:
    title_key = _title_key(title)
    sent_key = normalize_answer(sentence)[:180]
    if not title_key or not sent_key or _is_synthetic_title(title):
        return ""
    return f"{title_key}::{sent_key}"


def _fact_provenance_key(row: Dict[str, Any], candidate: Any) -> str:
    explicit = str(row.get("canonical_fact_key") or "").strip()
    triple = row.get("triple")
    if isinstance(triple, (list, tuple)):
        triple = " | ".join(str(part or "").strip() for part in triple)
    triple_key = normalize_answer(str(triple or explicit))
    candidate_key = _candidate_key(candidate)
    if not triple_key:
        return ""
    return f"fact::{triple_key}::{candidate_key}"


def _source_docs(qa_docs: Optional[List[str]]) -> List[str]:
    return [doc for doc in (qa_docs or []) if not _is_synthetic_doc(doc)]


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
        "lattice_size": 0,
        "selected_candidate": "",
        "selected_score": 0.0,
        "selected_source_count": 0,
        "selected_provenance_count": 0,
        "margin": 0.0,
        "support_gain": 0.0,
        "conflict_group_count": 0,
        "shadow_changed": False,
        "shadow_action": "none",
        "shadow_after": before,
    }


def _candidate_key(candidate: Any) -> str:
    return normalize_answer(str(candidate or ""))


def _candidate_title(item: Dict[str, Any]) -> str:
    title = str(item.get("title") or item.get("source_title") or "").strip()
    if title and not _is_synthetic_title(title):
        return title
    return str(item.get("lineage_source_title") or "").strip()


def _candidate_sentence(item: Dict[str, Any]) -> str:
    return str(
        item.get("sentence_preview")
        or item.get("sentence")
        or item.get("lineage_sentence_preview")
        or item.get("evidence")
        or ""
    ).strip()


def _candidate_source_titles(item: Dict[str, Any]) -> List[str]:
    titles: List[str] = []
    raw_titles = item.get("source_titles")
    if isinstance(raw_titles, (list, tuple)):
        titles.extend(str(title or "").strip() for title in raw_titles)
    elif raw_titles:
        titles.extend(str(raw_titles).split(" ; "))
    lineage_title = str(item.get("lineage_source_title") or "").strip()
    if lineage_title:
        titles.extend(lineage_title.split(" ; "))
    title = _candidate_title(item)
    if title:
        titles.extend(title.split(" ; "))
    deduped = []
    seen = set()
    for title in titles:
        title = str(title or "").strip()
        if not title or _is_synthetic_title(title):
            continue
        key = _title_key(title)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(title)
    return deduped


def _add_candidate(
    lattice: Dict[str, Dict[str, Any]],
    *,
    candidate: Any,
    source: str,
    score: float = 0.0,
    raw_score: float = 0.0,
    provenance_count: int = 1,
    question_overlap: float = 0.0,
    relation_score: float = 0.0,
    answer_type_score: float = 0.0,
    title: str = "",
    sentence_preview: str = "",
    provenance_key: str = "",
) -> None:
    text = str(candidate or "").strip(" \t\r\n.;:")
    key = _candidate_key(text)
    if not text or not key:
        return
    row = lattice.get(key)
    if row is None:
        row = {
            "candidate": text,
            "key": key,
            "sources": set(),
            "source_scores": {},
            "max_score": 0.0,
            "raw_score": 0.0,
            "provenance_count": 0,
            "titles": set(),
            "evidence_keys": set(),
            "source_keys": set(),
            "question_overlap": 0.0,
            "relation_score": 0.0,
            "answer_type_score": 0.0,
            "evidence": [],
        }
        lattice[key] = row

    row["sources"].add(source)
    row["source_scores"][source] = max(float(row["source_scores"].get(source, 0.0) or 0.0), float(score or 0.0))
    row["max_score"] = max(float(row["max_score"]), float(score or raw_score or 0.0))
    row["raw_score"] = max(float(row["raw_score"]), float(raw_score or score or 0.0))
    row["provenance_count"] += max(1, int(provenance_count or 1))
    row["question_overlap"] = max(float(row["question_overlap"]), float(question_overlap or 0.0))
    row["relation_score"] = max(float(row["relation_score"]), float(relation_score or 0.0))
    row["answer_type_score"] = max(float(row["answer_type_score"]), float(answer_type_score or 0.0))
    if title and not _is_synthetic_title(title):
        row["titles"].add(title)
    evidence_key = str(provenance_key or "").strip() or _sentence_key(
        title, sentence_preview
    )
    new_evidence = bool(evidence_key and evidence_key not in row["evidence_keys"])
    if evidence_key:
        row["evidence_keys"].add(evidence_key)
        row["source_keys"].add(evidence_key)
    elif title and not _is_synthetic_title(title):
        row["source_keys"].add(f"title::{_title_key(title)}")
    if sentence_preview and (not evidence_key or new_evidence):
        row["evidence"].append({
            "source": source,
            "title": title,
            "score": round(float(score or raw_score or 0.0), 4),
            "sentence_preview": sentence_preview[:240],
        })


def _collect_from_rows(
    lattice: Dict[str, Dict[str, Any]],
    rows: List[Dict[str, Any]],
    source: str,
) -> None:
    for row in rows or []:
        if bool(row.get("support_only")) or row.get("answer_candidate_eligible") is False:
            continue
        score = float(
            row.get("table_score", row.get("confidence", row.get("score", row.get("raw_confidence", 0.0)))) or 0.0
        )
        titles = _candidate_source_titles(row)
        provenance_count = int(row.get("provenance_count", row.get("frequency", 1)) or 1)
        if not titles:
            titles = [_candidate_title(row)]
        provenance_key = _fact_provenance_key(
            row,
            row.get("candidate") or row.get("selected_candidate"),
        )
        for index, title in enumerate(titles):
            _add_candidate(
                lattice,
                candidate=row.get("candidate") or row.get("selected_candidate"),
                source=source if len(titles) <= 1 else f"{source}:lineage:{index + 1}",
                score=score,
                raw_score=float(row.get("raw_confidence", score) or score),
                provenance_count=1 if len(titles) > 1 else provenance_count,
                question_overlap=float(row.get("question_overlap", 0.0) or 0.0),
                relation_score=float(row.get("relation_role_score", row.get("relation_bonus", row.get("role_score", 0.0))) or 0.0),
                answer_type_score=float(row.get("answer_type_score", row.get("target_signal", 0.0)) or 0.0),
                title=title,
                sentence_preview=_candidate_sentence(row),
                provenance_key=provenance_key,
            )


def _collect_module_candidates(
    lattice: Dict[str, Dict[str, Any]],
    module_name: str,
    module_info: Optional[Dict[str, Any]],
) -> None:
    info = module_info or {}
    _collect_from_rows(lattice, info.get("candidates") or [], module_name)
    _collect_from_rows(lattice, info.get("table_rows") or [], module_name)
    _collect_from_rows(lattice, info.get("challengers") or [], module_name)
    selected = info.get("selected_candidate")
    if selected:
        _add_candidate(
            lattice,
            candidate=selected,
            source=f"{module_name}:selected",
            score=float(info.get("confidence", info.get("selected_confidence", 0.0)) or 0.0),
            raw_score=float(info.get("raw_confidence", 0.0) or 0.0),
            provenance_count=max(1, int(info.get("selected_provenance_count", 1) or 1)),
        )


def _materialize_lattice(
    lattice: Dict[str, Dict[str, Any]],
    *,
    current_answer: str,
    min_sources: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    current_key = _candidate_key(current_answer)
    for key, item in lattice.items():
        if not key or key == current_key:
            continue
        sources = sorted(item["sources"])
        titles = sorted(item["titles"])
        module_source_count = len(sources)
        title_count = len(titles)
        independent_evidence_count = len(item.get("evidence_keys") or [])
        source_count = len(item.get("source_keys") or [])
        provenance_count = independent_evidence_count
        module_provenance_count = int(item["provenance_count"])
        max_score = float(item["max_score"])
        answer_type_score = float(item["answer_type_score"])
        relation_score = float(item["relation_score"])
        question_overlap = float(item["question_overlap"])
        provenance_bonus = min(0.12, 0.035 * max(0, provenance_count - 1))
        source_bonus = min(0.14, 0.055 * max(0, source_count - 1))
        title_bonus = min(0.10, 0.04 * max(0, title_count - 1))
        relation_bonus = min(0.08, 0.12 * relation_score)
        type_bonus = min(0.06, 0.05 * answer_type_score)
        overlap_bonus = min(0.06, 0.10 * question_overlap)
        short_penalty = 0.0
        if _token_count(str(item["candidate"])) > 8:
            short_penalty += 0.05
        if source_count < int(min_sources):
            short_penalty += 0.04
        score = max(
            0.0,
            min(
                0.99,
                max_score
                + provenance_bonus
                + source_bonus
                + title_bonus
                + relation_bonus
                + type_bonus
                + overlap_bonus
                - short_penalty,
            ),
        )
        rows.append({
            "candidate": item["candidate"],
            "key": key,
            "score": round(score, 4),
            "max_score": round(max_score, 4),
            "source_count": source_count,
            "module_source_count": module_source_count,
            "sources": sources,
            "provenance_count": provenance_count,
            "module_provenance_count": module_provenance_count,
            "title_count": title_count,
            "titles": titles[:5],
            "evidence_fingerprint_count": independent_evidence_count,
            "canonical_fact_deduplicated_support_count": max(
                0, title_count - source_count
            ),
            "question_overlap": round(question_overlap, 4),
            "relation_score": round(relation_score, 4),
            "answer_type_score": round(answer_type_score, 4),
            "evidence": item["evidence"][:5],
        })
    rows.sort(
        key=lambda row: (
            -float(row.get("score", 0.0) or 0.0),
            -int(row.get("source_count", 0) or 0),
            -int(row.get("provenance_count", 0) or 0),
            str(row.get("candidate", "")).lower(),
        )
    )
    return rows


def arbitrate_answer_with_provenance_lattice(
    question: str,
    answer: str,
    qa_docs: Optional[List[str]] = None,
    answer_style: str = "",
    evidence_consistent_reader: Optional[Dict[str, Any]] = None,
    evidence_object_arbitration: Optional[Dict[str, Any]] = None,
    typed_evidence_table: Optional[Dict[str, Any]] = None,
    contrastive_answer_arbitration: Optional[Dict[str, Any]] = None,
    anchored_typed_candidate_table: Optional[Dict[str, Any]] = None,
    relation_fact_join: Optional[Dict[str, Any]] = None,
    enabled: bool = False,
    apply_rewrite: bool = False,
    min_score: float = 0.88,
    generic_min_score: float = 0.93,
    min_margin: float = 0.12,
    min_support_gain: float = 0.14,
    alias_min_support_gain: float = 0.02,
    min_sources: int = 2,
    min_provenance: int = 2,
    min_question_overlap: float = 0.08,
    max_docs: int = 10,
    max_answer_tokens: int = 12,
    require_role_constraint: bool = True,
    skip_supported_answers: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Provenance-grounded answer lattice arbitration.

    The module treats candidate answers as tuples in a tiny provenance table:
    answer key, support sources, provenance snippets, typed score, relation
    score, and conflict margin. It rewrites only when independent evidence
    sources agree on a higher-scoring candidate than the current reader answer.
    """
    before = str(answer or "").strip()
    meta = _new_meta(enabled=enabled, before=before)
    meta.update({
        "apply_rewrite": bool(apply_rewrite),
        "min_sources": int(min_sources),
        "min_provenance": int(min_provenance),
        "require_role_constraint": bool(require_role_constraint),
        "skip_supported_answers": bool(skip_supported_answers),
    })
    if not enabled or not before:
        return before, meta

    source_docs = _source_docs(qa_docs)
    if not source_docs:
        meta["reason"] = "no_source_docs"
        return before, meta

    target_type = infer_answer_target_type(question)
    boolean_contract = bool(
        BOOLEAN_QUESTION_RE.match(question or "")
        or BOOLEAN_ANSWER_RE.match(before)
    )
    meta["boolean_answer_contract"] = boolean_contract
    if boolean_contract:
        meta.update({
            "target_type": "boolean",
            "answer_domain": ["yes", "no"],
            "reason": "boolean_answer_contract_protected",
            "action": "kept",
        })
        return before, meta
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
    evidence_rows = _build_typed_table(candidates)

    lattice: Dict[str, Dict[str, Any]] = {}
    _collect_from_rows(lattice, candidates, "reader_evidence_candidates")
    _collect_from_rows(lattice, evidence_rows, "typed_table_from_reader")
    _collect_module_candidates(lattice, "evidence_consistent_reader", evidence_consistent_reader)
    _collect_module_candidates(lattice, "evidence_object_arbitration", evidence_object_arbitration)
    _collect_module_candidates(lattice, "typed_evidence_table", typed_evidence_table)
    _collect_module_candidates(lattice, "contrastive_answer_arbitration", contrastive_answer_arbitration)
    _collect_from_rows(
        lattice,
        (anchored_typed_candidate_table or {}).get("candidate_rows") or [],
        "anchored_typed_candidate_table",
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
    rows = _materialize_lattice(lattice, current_answer=before, min_sources=min_sources)

    top = rows[0] if rows else None
    top_info = top or {}
    second_score = float(rows[1].get("score", 0.0)) if len(rows) > 1 else 0.0
    top_score = float(top_info.get("score", 0.0) or 0.0)
    margin = top_score - second_score
    after = str(top_info.get("candidate", "") or "").strip()
    source_count = int(top_info.get("source_count", 0) or 0)
    provenance_count = int(top_info.get("provenance_count", 0) or 0)
    support_gain = top_score - float(answer_support_score)
    is_abstain = bool(ABSTAIN_RE.search(before))
    containment_variant = bool(after and _is_containment_variant(before, after))
    shorter_alias = bool(
        containment_variant and 0 < _token_count(after) < _token_count(before)
    )
    concise_answer_span_protected = bool(
        shorter_alias and not verbose_or_multi and not is_abstain
    )
    before_multi_answer = _is_multi_answer_like(before) or answer_style == "multi_slot"
    after_multi_answer = _is_multi_answer_like(after)
    multi_answer_span_protected = bool(
        before_multi_answer
        and not after_multi_answer
        and _token_count(after) < _token_count(before)
        and not is_abstain
    )
    full_name_to_single_token_protected = bool(
        target_type == "person"
        and _token_count(before) >= 2
        and _token_count(after) <= 1
        and not is_abstain
    )
    same_answer = bool(after and _same_answer(before, after))

    conflict_groups = defaultdict(int)
    for row in rows[:8]:
        if float(row.get("score", 0.0) or 0.0) >= max(0.0, top_score - 0.08):
            conflict_groups[str(row.get("key", ""))] += 1

    meta.update({
        "answer_style": answer_style,
        "answer_supported": bool(answer_supported),
        "answer_support_score": round(float(answer_support_score), 4),
        "raw_candidate_count": len(raw_candidates),
        "filtered_candidate_count": len(candidates),
        "candidate_reject_counts": reject_counts,
        "lattice_size": len(rows),
        "lattice_rows": rows[:8],
        "canonical_fact_deduplicated_support_count": sum(
            int(row.get("canonical_fact_deduplicated_support_count", 0) or 0)
            for row in rows
        ),
        "selected_candidate": after,
        "selected_score": round(top_score, 4),
        "selected_source_count": source_count,
        "selected_provenance_count": provenance_count,
        "score": round(top_score, 4),
        "source_count": source_count,
        "provenance_count": provenance_count,
        "question_overlap": round(float(top_info.get("question_overlap", 0.0) or 0.0), 4),
        "relation_score": round(float(top_info.get("relation_score", 0.0) or 0.0), 4),
        "answer_type_score": round(float(top_info.get("answer_type_score", 0.0) or 0.0), 4),
        "margin": round(margin, 4),
        "support_gain": round(support_gain, 4),
        "conflict_group_count": len(conflict_groups),
        "verbose_or_multi": bool(verbose_or_multi),
        "containment_variant": bool(containment_variant),
        "shorter_alias": bool(shorter_alias),
        "concise_answer_span_protected": bool(concise_answer_span_protected),
        "multi_answer_span_protected": bool(multi_answer_span_protected),
        "full_name_to_single_token_protected": bool(full_name_to_single_token_protected),
        "is_abstain": bool(is_abstain),
    })

    if not top:
        meta["reason"] = "no_candidates"
        return before, meta
    if not after or same_answer:
        meta["reason"] = "same_as_current"
        return before, meta
    if source_count < max(1, int(min_sources)):
        meta["reason"] = "insufficient_independent_sources"
        return before, meta
    if provenance_count < max(1, int(min_provenance)):
        meta["reason"] = "insufficient_provenance"
        return before, meta
    if concise_answer_span_protected:
        meta["reason"] = "concise_answer_span_protected"
        return before, meta
    if multi_answer_span_protected:
        meta["reason"] = "multi_answer_span_protected"
        return before, meta
    if full_name_to_single_token_protected:
        meta["reason"] = "full_name_to_single_token_protected"
        return before, meta

    score_floor = float(min_score) if is_numeric_temporal else float(generic_min_score)
    strong_candidate = top_score >= score_floor and (margin >= float(min_margin) or second_score <= 0.0)

    should_rewrite = False
    action = "kept"
    if strong_candidate:
        if is_abstain:
            should_rewrite = True
            action = "abstain_to_lattice_answer"
        elif (
            not is_numeric_temporal
            and not answer_supported
            and support_gain >= max(0.0, float(min_support_gain) - 0.04)
        ):
            should_rewrite = True
            action = "unsupported_to_lattice_answer"
        elif (
            not is_numeric_temporal
            and verbose_or_multi
            and support_gain >= max(0.05, float(min_support_gain) * 0.5)
        ):
            should_rewrite = True
            action = "verbose_to_lattice_answer"
        elif verbose_or_multi and shorter_alias and support_gain >= float(alias_min_support_gain):
            should_rewrite = True
            action = "granularity_lattice_alias"

    if answer_supported and skip_supported_answers and not (
        is_abstain or verbose_or_multi
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
        "reason": action if action != "kept" else "kept_low_score_or_margin",
        "after": before,
    })
    return before, meta
