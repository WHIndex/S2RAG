import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .answer_verifier_utils import _is_supported
from .eval_utils import normalize_answer
from .evidence_grounded_answer_extractor_utils import (
    BOOLEAN_ANSWER_RE,
    BOOLEAN_QUESTION_RE,
    _candidate_evidence_text,
    _candidate_passes_answer_slot,
    _candidate_target_type_for_slot,
    _content_tokens,
    _infer_answer_arity,
    _infer_answer_slot,
    _same_date_value,
    _surface_rewrite_loss_reason,
)
from .evidence_object_arbitration_utils import _is_containment_variant
from .sentence_evidence_utils import infer_answer_target_type
from .typed_evidence_table_utils import _same_answer, _token_count


QUESTION_ANCHOR_MIN_TOKENS = 2
CHOICE_SURFACE_MAX_TOKENS = 12
STRICT_ABSTAIN_KEYS = {
    "unknown",
    "not specified",
    "not provided",
    "not mentioned",
    "cannot determine",
    "cannot be determined",
    "insufficient information",
    "no information",
    "not available",
    "no answer",
    "i dont know",
    "i do not know",
}
SOURCE_WEIGHTS = {
    "relation_fact_join": 0.14,
    "query_answer_contract": 0.13,
    "query_answer_contract_shadow": 0.12,
    "provenance_answer_slot_store": 0.11,
    "provenance_answer_slot_store_shadow": 0.10,
    "answer_slot_binding_optimizer": 0.10,
    "answer_slot_binding_optimizer_shadow": 0.09,
    "evidence_grounded_answer_extractor": 0.08,
    "evidence_grounded_answer_extractor_shadow": 0.07,
    "provenance_typed_candidate_store": 0.06,
    "reader_answer": 0.05,
    "stage_answer": 0.04,
    "current_answer": 0.02,
}

LINEAGE_FLAG_KEYS = (
    "explicit_slot_match",
    "role_constraint_match",
    "relation_entailment_match",
    "candidate_in_base_docs",
    "terminal_slot_match",
    "residual_predicate_match",
    "answer_candidate_eligible",
)


def _clean_answer_surface(answer: Any) -> str:
    text = str(answer or "").strip()
    text = re.sub(r"^\s*(?:answer|final answer)\s*:\s*", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip(" \t\n\r\"'")


def _surface_key(answer: Any) -> str:
    return normalize_answer(_clean_answer_surface(answer))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))


def _is_abstain_surface(answer: Any) -> bool:
    key = normalize_answer(str(answer or ""))
    if key in STRICT_ABSTAIN_KEYS:
        return True
    return bool(
        re.fullmatch(
            r"\s*(?:n/?a|none|unknown|not\s+(?:specified|provided|mentioned|available)|"
            r"cannot\s+(?:determine|be\s+determined)|insufficient\s+information|"
            r"no\s+(?:information|answer)|i\s+(?:do\s+not|don't)\s+know)\s*\.?\s*",
            str(answer or ""),
            flags=re.I,
        )
    )


def _new_meta(enabled: bool, answer: str) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "changed": False,
        "action": "none" if enabled else "disabled",
        "reason": "not_evaluated" if enabled else "disabled",
        "before": answer,
        "after": answer,
        "target_type": "",
        "candidate_target_type": "",
        "answer_slot": "",
        "answer_arity": "",
        "lattice_size": 0,
        "selected_candidate": None,
        "selected_source": None,
        "selected_score": 0.0,
        "selected_net_utility": 0.0,
        "selected_support_score": 0.0,
        "selected_risk": 0.0,
        "margin": 0.0,
        "runner_up_margin": 0.0,
        "runner_up_candidate": None,
        "lineage_terminal_key_count": 0,
        "selected_lineage_terminal_match": False,
        "support_gain": 0.0,
        "current_supported": False,
        "current_support_score": 0.0,
        "shadow_changed": False,
        "shadow_after": None,
        "shadow_reason": "",
        "boolean_contract": False,
        "explicit_choice_count": 0,
        "explicit_choices": [],
        "current_slot_compatible": False,
        "candidate_reject_counts": {},
        "top_candidates": [],
    }


def _looks_like_question_anchor(question: str, candidate: str, target_type: str) -> bool:
    if target_type in {"date", "year", "number"}:
        return False
    candidate_key = _surface_key(candidate)
    if not candidate_key:
        return False
    tokens = _content_tokens(candidate)
    if len(tokens) < QUESTION_ANCHOR_MIN_TOKENS:
        return False
    return candidate_key in normalize_answer(question or "")


def _clean_choice_surface(choice: str) -> str:
    text = re.sub(r"\s+", " ", str(choice or "")).strip(" \t\n\r\"'.,;:()[]")
    text = re.sub(r"^\s*(?:the|a|an)\s+", "", text, flags=re.I)
    return text.strip(" \t\n\r\"'.,;:()[]")


def _valid_choice_surface(choice: str) -> bool:
    norm = normalize_answer(choice or "")
    if not norm:
        return False
    if BOOLEAN_ANSWER_RE.match(choice or ""):
        return False
    tokens = _content_tokens(choice)
    if not tokens or len(tokens) > CHOICE_SURFACE_MAX_TOKENS:
        return False
    if norm in {"which", "what", "who", "where", "when", "film", "movie", "person"}:
        return False
    return True


def _split_choice_tail(tail: str, *, allow_and: bool = True) -> List[str]:
    split_re = r"\s+(?:or|and)\s+" if allow_and else r"\s+or\s+"
    pieces = [
        _clean_choice_surface(piece)
        for piece in re.split(split_re, str(tail or ""), flags=re.I)
    ]
    return [piece for piece in pieces if _valid_choice_surface(piece)]


def _extract_explicit_choices(question: str) -> List[str]:
    """Extract answer-domain choices from comparison questions.

    The surface lattice must not replace a comparison answer with a bridge
    entity.  For questions like "Which film ..., A or B?" the final answer
    should be one of the explicit choices, while directors, dates, countries,
    and other evidence entities are only intermediate values.
    """
    text = str(question or "").strip().rstrip("?.! ")
    if not text:
        return []
    norm = normalize_answer(text)
    wh_or_choice = bool(re.match(r"^\s*(?:which|what|who)\b", text, re.I))
    if not wh_or_choice and " out of " not in norm and " between " not in norm:
        return []

    candidates: List[str] = []
    for pattern in (
        r"\bout of\s+(.+?)\s+(?:and|or)\s+(.+)$",
        r"\bbetween\s+(.+?)\s+and\s+(.+)$",
    ):
        match = re.search(pattern, text, flags=re.I)
        if match:
            candidates.extend(_clean_choice_surface(part) for part in match.groups())

    if "," in text and re.search(r"\s+or\s+", text.rsplit(",", 1)[-1], re.I):
        candidates.extend(_split_choice_tail(text.rsplit(",", 1)[-1], allow_and=False))
    elif wh_or_choice and re.search(r"\s+or\s+", text, re.I):
        candidates.extend(_split_choice_tail(text, allow_and=False))

    deduped: List[str] = []
    seen = set()
    for choice in candidates:
        choice = _clean_choice_surface(choice)
        key = _surface_key(choice)
        if not key or key in seen or not _valid_choice_surface(choice):
            continue
        seen.add(key)
        deduped.append(choice)
    return deduped if len(deduped) >= 2 else []


def _candidate_matches_explicit_choice(candidate: str, choices: List[str]) -> bool:
    cand_key = _surface_key(candidate)
    if not cand_key:
        return False
    cand_tokens = _content_tokens(candidate)
    for choice in choices:
        choice_key = _surface_key(choice)
        if not choice_key:
            continue
        if cand_key == choice_key or _same_answer(candidate, choice):
            return True
        if len(cand_tokens) >= 2 and (cand_key in choice_key or choice_key in cand_key):
            return True
    return False


def _candidate_rows_from_meta(meta: Optional[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    if not isinstance(meta, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for key in ("top_candidates", "candidates", "contract_candidates", "slot_candidates", "candidate_rows", "lattice_rows"):
        value = meta.get(key)
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, dict))
    return rows


def _add_candidate(
    lattice: Dict[str, Dict[str, Any]],
    *,
    answer: Any,
    source: str,
    source_weight: float = 0.0,
    module_score: float = 0.0,
    support_score_hint: float = 0.0,
    source_count: int = 0,
    provenance_count: int = 0,
    question_overlap: float = 0.0,
    relation_score: float = 0.0,
    evidence_text: str = "",
    constraint_meta: Optional[Dict[str, Any]] = None,
) -> None:
    surface = _clean_answer_surface(answer)
    key = _surface_key(surface)
    if not key:
        return
    entry = lattice.setdefault(
        key,
        {
            "candidate": surface,
            "sources": set(),
            "source_weight": 0.0,
            "module_score": 0.0,
            "support_score_hint": 0.0,
            "source_count": 0,
            "provenance_count": 0,
            "question_overlap": 0.0,
            "relation_score": 0.0,
            "evidence_text": "",
            "base_provenance_count": 0,
            "lineage_terminal_match": False,
            "answer_types": set(),
            "predicate_families": set(),
            "branch_ids": set(),
            "lineage_keys": set(),
            **{key: False for key in LINEAGE_FLAG_KEYS},
        },
    )
    entry["sources"].add(source)
    entry["source_weight"] = max(
        float(entry.get("source_weight", 0.0) or 0.0),
        float(source_weight or 0.0),
        SOURCE_WEIGHTS.get(source, 0.0),
    )
    entry["module_score"] = max(float(entry.get("module_score", 0.0) or 0.0), _float(module_score))
    entry["support_score_hint"] = max(
        float(entry.get("support_score_hint", 0.0) or 0.0),
        _float(support_score_hint),
    )
    entry["source_count"] = max(_int(entry.get("source_count")), _int(source_count))
    entry["provenance_count"] = max(_int(entry.get("provenance_count")), _int(provenance_count))
    entry["question_overlap"] = max(_float(entry.get("question_overlap")), _float(question_overlap))
    entry["relation_score"] = max(_float(entry.get("relation_score")), _float(relation_score))
    constraint_meta = constraint_meta if isinstance(constraint_meta, dict) else {}
    entry["base_provenance_count"] = max(
        _int(entry.get("base_provenance_count")),
        _int(constraint_meta.get("base_provenance_count")),
        _int(constraint_meta.get("base_source_count")),
        _int(constraint_meta.get("selected_base_provenance_count")),
        _int(constraint_meta.get("selected_base_source_count")),
    )
    for flag in LINEAGE_FLAG_KEYS:
        if flag in constraint_meta:
            entry[flag] = bool(entry.get(flag)) or bool(constraint_meta.get(flag))
    entry["lineage_terminal_match"] = bool(entry.get("lineage_terminal_match")) or bool(
        constraint_meta.get("lineage_terminal_match")
    )
    answer_type = str(
        constraint_meta.get("answer_type")
        or constraint_meta.get("terminal_answer_type")
        or ""
    ).strip().lower()
    if answer_type:
        entry["answer_types"].add(answer_type)
    predicate_families = constraint_meta.get("predicate_families") or []
    if isinstance(predicate_families, str):
        predicate_families = [predicate_families]
    predicate_family = str(
        constraint_meta.get("predicate_family") or ""
    ).strip().lower()
    if predicate_family:
        entry["predicate_families"].add(predicate_family)
    entry["predicate_families"].update(
        str(value).strip().lower()
        for value in predicate_families
        if str(value).strip()
    )
    branch_id = str(constraint_meta.get("branch_id") or "").strip()
    if branch_id:
        entry["branch_ids"].add(branch_id)
    lineage_key = str(constraint_meta.get("lineage_key") or "").strip()
    if lineage_key:
        entry["lineage_keys"].add(lineage_key)
    if evidence_text and len(str(evidence_text)) > len(str(entry.get("evidence_text") or "")):
        entry["evidence_text"] = str(evidence_text)


def _add_stage_candidate(lattice: Dict[str, Dict[str, Any]], answer: Any, source: str) -> None:
    _add_candidate(
        lattice,
        answer=answer,
        source=source,
        source_weight=SOURCE_WEIGHTS.get(source, SOURCE_WEIGHTS["stage_answer"]),
    )


def _add_module_candidates(
    lattice: Dict[str, Dict[str, Any]],
    *,
    meta: Optional[Dict[str, Any]],
    source: str,
) -> None:
    if not isinstance(meta, dict):
        return
    source_weight = SOURCE_WEIGHTS.get(source, 0.04)
    selected_for_evidence = meta.get("selected_candidate")
    selected_evidence_text = (
        _candidate_evidence_text(selected_for_evidence)
        if isinstance(selected_for_evidence, dict)
        else ""
    )
    if meta.get("changed"):
        _add_candidate(
            lattice,
            answer=meta.get("after"),
            source=source,
            source_weight=source_weight,
            module_score=max(_float(meta.get("selected_score")), _float(meta.get("selected_binding_score"))),
            support_score_hint=_float(meta.get("selected_support_score")),
            source_count=_int(meta.get("selected_source_count")),
            provenance_count=_int(meta.get("selected_provenance_count")),
            question_overlap=_float(meta.get("selected_question_overlap")),
            relation_score=max(_float(meta.get("selected_relation_score")), _float(meta.get("selected_relation_entailment_score"))),
            evidence_text=selected_evidence_text,
            constraint_meta=meta,
        )
    if meta.get("shadow_changed"):
        _add_candidate(
            lattice,
            answer=meta.get("shadow_after"),
            source=f"{source}_shadow",
            source_weight=SOURCE_WEIGHTS.get(f"{source}_shadow", source_weight * 0.9),
            module_score=max(_float(meta.get("selected_score")), _float(meta.get("selected_binding_score"))),
            support_score_hint=_float(meta.get("selected_support_score")),
            source_count=_int(meta.get("selected_source_count")),
            provenance_count=_int(meta.get("selected_provenance_count")),
            question_overlap=_float(meta.get("selected_question_overlap")),
            relation_score=max(_float(meta.get("selected_relation_score")), _float(meta.get("selected_relation_entailment_score"))),
            constraint_meta=meta,
        )
    selected = meta.get("selected_candidate")
    if selected is not None:
        if isinstance(selected, dict):
            selected_answer = selected.get("candidate") or selected.get("answer") or selected.get("surface")
            selected_evidence = _candidate_evidence_text(selected)
        else:
            selected_answer = selected
            selected_evidence = ""
        _add_candidate(
            lattice,
            answer=selected_answer,
            source=source,
            source_weight=source_weight,
            module_score=max(_float(meta.get("selected_score")), _float(meta.get("selected_binding_score"))),
            support_score_hint=_float(meta.get("selected_support_score")),
            source_count=_int(meta.get("selected_source_count")),
            provenance_count=_int(meta.get("selected_provenance_count")),
            question_overlap=_float(meta.get("selected_question_overlap")),
            relation_score=max(_float(meta.get("selected_relation_score")), _float(meta.get("selected_relation_entailment_score"))),
            evidence_text=selected_evidence,
            constraint_meta=(selected if isinstance(selected, dict) else meta),
        )
    for row in _candidate_rows_from_meta(meta):
        row_answer = row.get("candidate") or row.get("answer") or row.get("surface")
        _add_candidate(
            lattice,
            answer=row_answer,
            source=source,
            source_weight=source_weight * 0.75,
            module_score=max(
                _float(row.get("score")),
                _float(row.get("binding_score")),
                _float(row.get("net_utility")),
            ),
            support_score_hint=max(_float(row.get("support_score")), _float(row.get("score"))),
            source_count=_int(row.get("source_count")),
            provenance_count=_int(row.get("provenance_count")),
            question_overlap=_float(row.get("question_overlap")),
            relation_score=max(_float(row.get("relation_score")), _float(row.get("relation_entailment_score"))),
            evidence_text=_candidate_evidence_text(row),
            constraint_meta=row,
        )


def _relation_terminal_rows(relation_fact_join: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(relation_fact_join, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for key in (
        "validated_terminal_candidate_rows",
        "reader_candidate_table_rows",
        "candidate_table_rows",
    ):
        value = relation_fact_join.get(key)
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, dict))
    intervention = relation_fact_join.get("counterfactual_reader_intervention")
    if isinstance(intervention, dict) and isinstance(intervention.get("row"), dict):
        row = intervention["row"]
        if row.get("production_eligible") or (
            row.get("schema_bound_semijoin")
            and row.get("terminal_slot_match")
            and row.get("residual_predicate_match")
        ):
            rows.append(row)
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        identity = (
            str(row.get("lineage_key") or ""),
            _surface_key(
                row.get("candidate")
                or row.get("terminal_value")
                or row.get("answer")
            ),
            str(row.get("lineage_source_doc_id") or row.get("doc_id") or ""),
            str(row.get("triple") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(row)
    return deduped


def _lineage_terminal_keys(
    relation_fact_join: Optional[Dict[str, Any]],
) -> set:
    keys = set()
    for row in _relation_terminal_rows(relation_fact_join):
        if row.get("support_only") or row.get("answer_candidate_eligible") is False:
            continue
        terminal_match = bool(
            row.get("production_eligible")
            and row.get("terminal_slot_match")
            and row.get("residual_predicate_match")
        ) or bool(
            row.get("terminal_slot_match")
            and row.get("residual_predicate_match")
            and row.get("schema_bound_semijoin")
        )
        key = _surface_key(
            row.get("candidate") or row.get("terminal_value") or row.get("answer")
        )
        if key and terminal_match:
            keys.add(key)
    return keys


def _add_relation_fact_join_candidates(
    lattice: Dict[str, Dict[str, Any]],
    relation_fact_join: Optional[Dict[str, Any]],
) -> set:
    terminal_keys = set()
    for row in _relation_terminal_rows(relation_fact_join):
        if row.get("support_only") or row.get("answer_candidate_eligible") is False:
            continue
        answer = row.get("candidate") or row.get("terminal_value") or row.get("answer")
        key = _surface_key(answer)
        terminal_match = bool(
            row.get("production_eligible")
            and row.get("terminal_slot_match")
            and row.get("residual_predicate_match")
        ) or bool(
            row.get("terminal_slot_match")
            and row.get("residual_predicate_match")
            and row.get("schema_bound_semijoin")
        )
        if key and terminal_match:
            terminal_keys.add(key)
        annotated = dict(row)
        annotated["lineage_terminal_match"] = terminal_match
        _add_candidate(
            lattice,
            answer=answer,
            source="relation_fact_join",
            source_weight=SOURCE_WEIGHTS["relation_fact_join"],
            module_score=_float(row.get("table_score")),
            support_score_hint=_float(row.get("table_score")),
            provenance_count=_int(row.get("provenance_count")),
            question_overlap=_float(row.get("question_overlap")),
            relation_score=_float(row.get("role_score")),
            evidence_text=str(row.get("evidence") or ""),
            constraint_meta=annotated,
        )
    return terminal_keys


def _build_lattice(
    *,
    answer: str,
    answer_before_evidence_grounded_extractor: Optional[str],
    answer_before_query_answer_contract: Optional[str],
    answer_before_provenance_answer_slot_store: Optional[str],
    answer_before_answer_slot_binding_optimizer: Optional[str],
    evidence_grounded_answer_extractor: Optional[Dict[str, Any]],
    query_answer_contract: Optional[Dict[str, Any]],
    provenance_answer_slot_store: Optional[Dict[str, Any]],
    answer_slot_binding_optimizer: Optional[Dict[str, Any]],
    provenance_typed_candidate_store: Optional[Dict[str, Any]],
    relation_fact_join: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], set]:
    lattice: Dict[str, Dict[str, Any]] = {}
    _add_stage_candidate(lattice, answer, "current_answer")
    _add_stage_candidate(lattice, answer_before_evidence_grounded_extractor, "reader_answer")
    _add_stage_candidate(lattice, answer_before_query_answer_contract, "stage_answer")
    _add_stage_candidate(lattice, answer_before_provenance_answer_slot_store, "stage_answer")
    _add_stage_candidate(lattice, answer_before_answer_slot_binding_optimizer, "stage_answer")
    _add_module_candidates(
        lattice,
        meta=provenance_typed_candidate_store,
        source="provenance_typed_candidate_store",
    )
    _add_module_candidates(
        lattice,
        meta=evidence_grounded_answer_extractor,
        source="evidence_grounded_answer_extractor",
    )
    _add_module_candidates(
        lattice,
        meta=query_answer_contract,
        source="query_answer_contract",
    )
    _add_module_candidates(
        lattice,
        meta=provenance_answer_slot_store,
        source="provenance_answer_slot_store",
    )
    _add_module_candidates(
        lattice,
        meta=answer_slot_binding_optimizer,
        source="answer_slot_binding_optimizer",
    )
    terminal_keys = _add_relation_fact_join_candidates(lattice, relation_fact_join)
    for key in terminal_keys:
        if key in lattice:
            lattice[key]["lineage_terminal_match"] = True
    return lattice, terminal_keys


def _score_candidate(
    *,
    question: str,
    candidate: Dict[str, Any],
    current_answer: str,
    current_supported: bool,
    current_support_score: float,
    qa_docs: List[str],
    target_type: str,
    candidate_target_type: str,
    answer_slot: str,
    answer_arity: str,
    boolean_contract: bool,
    explicit_choices: List[str],
    max_docs: int,
    max_answer_tokens: int,
    risk_weight: float,
) -> Tuple[Optional[Dict[str, Any]], str]:
    surface = str(candidate.get("candidate") or "").strip()
    if not surface:
        return None, "empty_candidate"
    if _is_abstain_surface(surface) and not _is_abstain_surface(current_answer):
        return None, "candidate_abstain"
    if boolean_contract and not BOOLEAN_ANSWER_RE.match(surface):
        return None, "boolean_contract_non_boolean"
    choice_match = bool(explicit_choices and _candidate_matches_explicit_choice(surface, explicit_choices))
    if explicit_choices and not choice_match:
        return None, "choice_contract_mismatch"
    token_count = _token_count(surface)
    if answer_arity == "single" and token_count > max_answer_tokens:
        return None, "candidate_too_verbose"
    if not choice_match and _looks_like_question_anchor(question, surface, target_type):
        return None, "question_anchor"

    evidence_text = " ".join(str(doc or "") for doc in qa_docs[:max_docs])
    if candidate.get("evidence_text"):
        evidence_text = f"{evidence_text} {candidate.get('evidence_text')}"
    slot_ok, slot_reason = _candidate_passes_answer_slot(
        surface,
        answer_slot,
        candidate_target_type,
        question=question,
        evidence_text=evidence_text,
    )
    if not slot_ok:
        return None, slot_reason or "slot_mismatch"

    supported, support_score = _is_supported(
        question,
        surface,
        qa_docs,
        candidate_target_type,
        max_docs=max_docs,
    )
    support_score = max(_float(support_score), _float(candidate.get("support_score_hint")))
    support_gain = support_score - current_support_score
    sources = candidate.get("sources") or set()
    source_count = max(len(sources), _int(candidate.get("source_count")))
    provenance_count = _int(candidate.get("provenance_count"))
    module_score = _bounded(_float(candidate.get("module_score")))
    question_overlap = _bounded(_float(candidate.get("question_overlap")))
    relation_score = _bounded(_float(candidate.get("relation_score")))
    surface_loss = _surface_rewrite_loss_reason(
        question=question,
        before=current_answer,
        after=surface,
        slot=answer_slot,
        target_type=candidate_target_type,
    )

    risk = 0.0
    if not supported:
        risk += 0.35
    if surface_loss:
        risk += 0.38
    if _is_containment_variant(current_answer, surface) and not _same_answer(current_answer, surface):
        if not _same_date_value(current_answer, surface):
            risk += 0.12
    if current_supported and support_gain < 0.015:
        risk += 0.10
    if source_count <= 1 and provenance_count <= 1 and not supported:
        risk += 0.12
    if answer_slot.startswith("person_") and relation_score < 0.12 and source_count <= 1:
        risk += 0.08

    utility = (
        support_score
        + 0.05 * min(source_count, 3)
        + 0.015 * min(provenance_count, 4)
        + _float(candidate.get("source_weight"))
        + 0.08 * module_score
        + 0.04 * question_overlap
        + 0.04 * relation_score
        + 0.18 * max(0.0, support_gain)
    )
    if _is_abstain_surface(current_answer) and supported:
        utility += 0.16
    if not current_supported and supported:
        utility += 0.10

    net_utility = utility - risk_weight * risk
    row = {
        "candidate": surface,
        "sources": sorted(str(src) for src in sources),
        "score": round(float(utility), 4),
        "net_utility": round(float(net_utility), 4),
        "support_score": round(float(support_score), 4),
        "support_gain": round(float(support_gain), 4),
        "risk": round(float(risk), 4),
        "source_count": source_count,
        "provenance_count": provenance_count,
        "module_score": round(float(module_score), 4),
        "question_overlap": round(float(question_overlap), 4),
        "relation_score": round(float(relation_score), 4),
        "supported": bool(supported),
        "surface_loss": surface_loss,
        "base_provenance_count": _int(candidate.get("base_provenance_count")),
        "lineage_terminal_match": bool(candidate.get("lineage_terminal_match")),
        "answer_types": sorted(candidate.get("answer_types") or []),
        "predicate_families": sorted(
            candidate.get("predicate_families") or []
        ),
        "branch_ids": sorted(candidate.get("branch_ids") or []),
        "lineage_keys": sorted(candidate.get("lineage_keys") or []),
        **{key: bool(candidate.get(key)) for key in LINEAGE_FLAG_KEYS},
    }
    return row, ""


def _runner_is_compatible(
    best: Dict[str, Any], runner: Dict[str, Any]
) -> Tuple[bool, str]:
    """Compare only candidates from the same typed terminal domain."""
    for field, reason in (
        ("answer_types", "answer_type_mismatch"),
        ("predicate_families", "predicate_family_mismatch"),
        ("branch_ids", "branch_mismatch"),
    ):
        left = set(best.get(field) or [])
        right = set(runner.get(field) or [])
        if left and right and left.isdisjoint(right):
            return False, reason
    return True, "compatible"


def _adaptive_runner_margin(
    best: Dict[str, Any], default_margin: float
) -> Tuple[float, str]:
    if best.get("lineage_terminal_match"):
        return min(float(default_margin), 0.01), "terminal_lineage"
    base_provenance = _int(best.get("base_provenance_count"))
    if (
        base_provenance > 0
        and best.get("supported")
        and _float(best.get("relation_score")) >= 0.20
    ):
        return min(float(default_margin), 0.015), "supported_relation"
    if base_provenance > 0:
        return min(float(default_margin), 0.03), "base_provenance"
    return float(default_margin), "unbound"


def _candidate_pool_hash(rows: List[Dict[str, Any]]) -> str:
    payload = [
        {
            "candidate": _surface_key(row.get("candidate")),
            "net_utility": round(_float(row.get("net_utility")), 6),
            "support_score": round(_float(row.get("support_score")), 6),
            "answer_types": sorted(row.get("answer_types") or []),
            "predicate_families": sorted(
                row.get("predicate_families") or []
            ),
            "branch_ids": sorted(row.get("branch_ids") or []),
        }
        for row in rows
    ]
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def optimize_answer_surface_lattice(
    *,
    question: str,
    answer: str,
    qa_docs: List[str],
    answer_style: str = "",
    answer_before_evidence_grounded_extractor: Optional[str] = None,
    answer_before_query_answer_contract: Optional[str] = None,
    answer_before_provenance_answer_slot_store: Optional[str] = None,
    answer_before_answer_slot_binding_optimizer: Optional[str] = None,
    evidence_grounded_answer_extractor: Optional[Dict[str, Any]] = None,
    query_answer_contract: Optional[Dict[str, Any]] = None,
    provenance_answer_slot_store: Optional[Dict[str, Any]] = None,
    answer_slot_binding_optimizer: Optional[Dict[str, Any]] = None,
    provenance_typed_candidate_store: Optional[Dict[str, Any]] = None,
    relation_fact_join: Optional[Dict[str, Any]] = None,
    enabled: bool = False,
    apply_rewrite: bool = True,
    min_net_utility: float = 0.08,
    min_margin: float = 0.035,
    min_support_gain: float = 0.02,
    max_risk: float = 0.48,
    max_docs: int = 10,
    max_answer_tokens: int = 16,
    risk_weight: float = 0.65,
    prefer_contract_margin: float = 0.015,
    abstain_min_runner_up_margin: float = 0.08,
    require_lineage_for_relational_abstain: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    original_answer = str(answer or "").strip()
    current_answer = _clean_answer_surface(answer)
    meta = _new_meta(enabled, original_answer)
    if not enabled:
        return original_answer, meta
    if not current_answer:
        meta.update({"reason": "empty_answer", "after": original_answer})
        return original_answer, meta

    target_type = infer_answer_target_type(question or "")
    answer_slot = _infer_answer_slot(question or "", target_type)
    candidate_target_type = _candidate_target_type_for_slot(answer_slot, target_type)
    answer_arity = _infer_answer_arity(question or "", current_answer, target_type)
    boolean_contract = bool(
        BOOLEAN_QUESTION_RE.match(question or "") or BOOLEAN_ANSWER_RE.match(current_answer or "")
    )
    explicit_choices = _extract_explicit_choices(question or "")
    current_slot_compatible, _current_slot_reason = _candidate_passes_answer_slot(
        current_answer,
        answer_slot,
        candidate_target_type,
        question=question,
        evidence_text=" ".join(str(doc or "") for doc in (qa_docs or [])[:max_docs]),
    )
    meta.update(
        {
            "target_type": target_type,
            "candidate_target_type": candidate_target_type,
            "answer_slot": answer_slot,
            "answer_arity": answer_arity,
            "boolean_contract": boolean_contract,
            "explicit_choice_count": len(explicit_choices),
            "explicit_choices": explicit_choices[:4],
            "current_slot_compatible": bool(current_slot_compatible),
        }
    )

    current_supported, current_support_score = _is_supported(
        question,
        current_answer,
        qa_docs,
        candidate_target_type,
        max_docs=max_docs,
    )
    current_support_score = _float(current_support_score)
    meta.update(
        {
            "current_supported": bool(current_supported),
            "current_support_score": round(float(current_support_score), 4),
        }
    )
    current_abstain = _is_abstain_surface(current_answer)
    precompiled_lineage_terminal_keys = _lineage_terminal_keys(
        relation_fact_join
    )
    meta["lineage_terminal_key_count"] = len(
        precompiled_lineage_terminal_keys
    )
    if (
        boolean_contract
        and BOOLEAN_ANSWER_RE.match(current_answer or "")
        and not current_abstain
    ):
        meta.update({"reason": "boolean_current_protected", "after": original_answer})
        return original_answer, meta
    if (
        bool(current_supported)
        and bool(current_slot_compatible)
        and not current_abstain
    ):
        meta.update(
            {
                "reason": "supported_slot_compatible_current_protected",
                "after": original_answer,
            }
        )
        return original_answer, meta
    if not current_abstain and not (
        boolean_contract and not BOOLEAN_ANSWER_RE.match(current_answer or "")
    ):
        meta.update({"reason": "non_abstain_surface_shadow_only", "after": original_answer})
        return original_answer, meta

    lattice, lineage_terminal_keys = _build_lattice(
        answer=current_answer,
        answer_before_evidence_grounded_extractor=answer_before_evidence_grounded_extractor,
        answer_before_query_answer_contract=answer_before_query_answer_contract,
        answer_before_provenance_answer_slot_store=answer_before_provenance_answer_slot_store,
        answer_before_answer_slot_binding_optimizer=answer_before_answer_slot_binding_optimizer,
        evidence_grounded_answer_extractor=evidence_grounded_answer_extractor,
        query_answer_contract=query_answer_contract,
        provenance_answer_slot_store=provenance_answer_slot_store,
        answer_slot_binding_optimizer=answer_slot_binding_optimizer,
        provenance_typed_candidate_store=provenance_typed_candidate_store,
        relation_fact_join=relation_fact_join,
    )
    meta["lattice_size"] = len(lattice)
    meta["lineage_terminal_key_count"] = len(lineage_terminal_keys)
    if len(lattice) <= 1:
        meta.update({"reason": "single_surface_lattice", "after": original_answer})
        return original_answer, meta

    reject_counts: Counter = Counter()
    scored: List[Dict[str, Any]] = []
    for candidate in lattice.values():
        row, reject_reason = _score_candidate(
            question=question,
            candidate=candidate,
            current_answer=current_answer,
            current_supported=bool(current_supported),
            current_support_score=current_support_score,
            qa_docs=qa_docs,
            target_type=target_type,
            candidate_target_type=candidate_target_type,
            answer_slot=answer_slot,
            answer_arity=answer_arity,
            boolean_contract=boolean_contract,
            explicit_choices=explicit_choices,
            max_docs=max_docs,
            max_answer_tokens=max_answer_tokens,
            risk_weight=risk_weight,
        )
        if row is None:
            reject_counts[reject_reason or "rejected"] += 1
            continue
        scored.append(row)

    scored.sort(key=lambda row: (
        -_float(row.get("net_utility")),
        -_float(row.get("support_score")),
        -_int(row.get("source_count")),
        _surface_key(row.get("candidate")),
        str(row.get("candidate") or ""),
    ))
    meta["candidate_reject_counts"] = dict(reject_counts)
    meta["top_candidates"] = scored[:6]
    meta["candidate_pool_hash"] = _candidate_pool_hash(scored)
    if not scored:
        meta.update({"reason": "no_valid_surface_candidate", "after": original_answer})
        return original_answer, meta

    incumbent_key = _surface_key(current_answer)
    incumbent = next((row for row in scored if _surface_key(row.get("candidate")) == incumbent_key), None)
    if incumbent is None:
        incumbent = {
            "candidate": current_answer,
            "net_utility": current_support_score,
            "support_score": current_support_score,
            "risk": 0.0 if current_supported else 0.35,
            "source_count": 1,
            "sources": ["current_answer"],
            "support_gain": 0.0,
            "surface_loss": "",
        }
    best = scored[0]
    best_key = _surface_key(best.get("candidate"))
    competing = [
        row for row in scored[1:]
        if _surface_key(row.get("candidate")) != incumbent_key
    ]
    compatible_competing = []
    incompatible_runner_counts: Counter = Counter()
    for row in competing:
        compatible, compatibility_reason = _runner_is_compatible(best, row)
        if compatible:
            compatible_competing.append(row)
        else:
            incompatible_runner_counts[compatibility_reason] += 1
    runner_up = compatible_competing[0] if compatible_competing else None
    runner_up_margin = (
        _float(best.get("net_utility")) - _float(runner_up.get("net_utility"))
        if runner_up is not None else 1.0
    )
    margin = _float(best.get("net_utility")) - _float(incumbent.get("net_utility"))
    support_gain = _float(best.get("support_score")) - current_support_score
    required_runner_margin, runner_margin_tier = _adaptive_runner_margin(
        best, abstain_min_runner_up_margin
    )
    meta.update(
        {
            "selected_candidate": best.get("candidate"),
            "selected_source": ",".join(best.get("sources") or []),
            "selected_score": best.get("score", 0.0),
            "selected_net_utility": best.get("net_utility", 0.0),
            "selected_support_score": best.get("support_score", 0.0),
            "selected_risk": best.get("risk", 0.0),
            "margin": round(float(margin), 4),
            "runner_up_margin": round(float(runner_up_margin), 4),
            "runner_up_candidate": runner_up.get("candidate") if runner_up else None,
            "runner_up_total_count": len(competing),
            "compatible_runner_count": len(compatible_competing),
            "incompatible_runner_counts": dict(incompatible_runner_counts),
            "required_runner_up_margin": round(
                float(required_runner_margin), 4
            ),
            "runner_margin_tier": runner_margin_tier,
            "selected_lineage_terminal_match": bool(best.get("lineage_terminal_match")),
            "support_gain": round(float(support_gain), 4),
        }
    )

    if best_key == incumbent_key or _same_answer(best.get("candidate"), current_answer):
        meta.update({"reason": "current_best", "after": original_answer})
        return original_answer, meta
    current_abstain = _is_abstain_surface(current_answer)
    if (
        boolean_contract
        and BOOLEAN_ANSWER_RE.match(current_answer or "")
        and not current_abstain
    ):
        meta.update(
            {
                "reason": "boolean_current_protected",
                "after": original_answer,
                "shadow_changed": True,
                "shadow_after": best.get("candidate"),
                "shadow_reason": "boolean_current_protected",
            }
        )
        return original_answer, meta
    if (
        bool(current_supported)
        and bool(current_slot_compatible)
        and not current_abstain
    ):
        meta.update(
            {
                "reason": "supported_slot_compatible_current_protected",
                "after": original_answer,
                "shadow_changed": True,
                "shadow_after": best.get("candidate"),
                "shadow_reason": "supported_slot_compatible_current_protected",
            }
        )
        return original_answer, meta
    if not current_abstain and not (
        boolean_contract and not BOOLEAN_ANSWER_RE.match(current_answer or "")
    ):
        meta.update(
            {
                "reason": "non_abstain_surface_shadow_only",
                "after": original_answer,
                "shadow_changed": True,
                "shadow_after": best.get("candidate"),
                "shadow_reason": "non_abstain_surface_shadow_only",
            }
        )
        return original_answer, meta
    if _float(best.get("net_utility")) < min_net_utility:
        meta.update({"reason": "low_net_utility", "after": original_answer})
        return original_answer, meta
    if _float(best.get("risk")) > max_risk:
        meta.update({"reason": "risk_veto", "after": original_answer})
        return original_answer, meta
    if best.get("surface_loss"):
        meta.update({"reason": f"surface_loss:{best.get('surface_loss')}", "after": original_answer})
        return original_answer, meta

    if current_abstain and runner_up_margin < float(required_runner_margin):
        meta.update({
            "reason": "abstain_ambiguous_runner_up",
            "after": original_answer,
            "shadow_changed": True,
            "shadow_after": best.get("candidate"),
            "shadow_reason": "abstain_ambiguous_runner_up",
        })
        return original_answer, meta

    relational_abstain = bool(
        current_abstain
        and (
            answer_slot.startswith("person_")
            or re.search(
                r"\b(?:who|whose|wife|husband|spouse|mother|father|parent|child|director|composer|author)\b",
                question or "",
                flags=re.I,
            )
        )
    )
    if (
        relational_abstain
        and require_lineage_for_relational_abstain
        and lineage_terminal_keys
        and best_key not in lineage_terminal_keys
    ):
        meta.update({
            "reason": "abstain_lineage_constraint_failed",
            "after": original_answer,
            "shadow_changed": True,
            "shadow_after": best.get("candidate"),
            "shadow_reason": "abstain_lineage_constraint_failed",
        })
        return original_answer, meta

    sources = set(best.get("sources") or [])
    contract_preferred = bool(sources & {"query_answer_contract", "query_answer_contract_shadow"})
    accept = False
    reason = "insufficient_margin"
    if current_abstain and best.get("supported") and margin >= 0.0:
        accept = True
        reason = "replace_abstain"
    elif (not current_supported) and best.get("supported") and margin >= min_margin * 0.5:
        accept = True
        reason = "unsupported_current_surface"
    elif contract_preferred and margin >= prefer_contract_margin and support_gain >= 0.0:
        accept = True
        reason = "contract_preferred_surface"
    elif support_gain >= min_support_gain and margin >= min_margin:
        accept = True
        reason = "support_gain_surface"

    if not accept:
        meta.update(
            {
                "reason": reason,
                "after": original_answer,
                "shadow_changed": True,
                "shadow_after": best.get("candidate"),
                "shadow_reason": reason,
            }
        )
        return original_answer, meta

    if not apply_rewrite:
        meta.update(
            {
                "action": "shadow",
                "reason": reason,
                "after": original_answer,
                "shadow_changed": True,
                "shadow_after": best.get("candidate"),
                "shadow_reason": reason,
            }
        )
        return original_answer, meta

    new_answer = str(best.get("candidate") or original_answer).strip()
    meta.update(
        {
            "changed": not _same_answer(new_answer, original_answer),
            "action": "rewrite",
            "reason": reason,
            "after": new_answer,
        }
    )
    return new_answer, meta
