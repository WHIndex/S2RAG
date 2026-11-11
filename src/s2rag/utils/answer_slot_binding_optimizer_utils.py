import hashlib
import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .answer_verifier_utils import NATIONALITY_OR_COUNTRY_RE, _is_supported, _rank_candidates
from .candidate_answer_selector_utils import _has_main_numeric_temporal_request, _is_synthetic_doc
from .eval_utils import normalize_answer
from .evidence_grounded_answer_extractor_utils import (
    BOOLEAN_ANSWER_RE,
    BOOLEAN_QUESTION_RE,
    DATE_NUMBER_TYPES,
    SUPPORTED_TARGET_TYPES,
    _candidate_evidence_text,
    _candidate_passes_answer_slot,
    _candidate_target_type_for_slot,
    HIGH_LEVEL_LOCATION_KEYS,
    _infer_answer_arity,
    _infer_answer_slot,
    _same_date_value,
    _surface_rewrite_loss_reason,
)
from .evidence_object_arbitration_utils import (
    ABSTAIN_RE,
    _candidate_passes_role_constraint,
    _filter_candidates,
    _is_containment_variant,
    _looks_verbose_or_multi_answer,
)
from .provenance_answer_lattice_utils import (
    _collect_from_rows,
    _collect_module_candidates,
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
    _slot_type_for_question,
    _source_documents_contain_answer,
)
from .sentence_evidence_utils import infer_answer_target_type
from .typed_evidence_table_utils import _build_typed_table


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
RELATION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "does", "for", "from",
    "had", "has", "have", "he", "her", "his", "how", "in", "is", "it", "its", "of",
    "on", "or", "she", "that", "the", "their", "they", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "whom", "whose", "with",
}
ROLE_CUES_BY_SLOT = {
    "person_spouse": {"spouse", "husband", "wife", "married", "marriage", "widow", "widower"},
    "person_mother": {"mother", "parent", "daughter", "son", "child", "children"},
    "person_father": {"father", "parent", "daughter", "son", "child", "children"},
    "person_grandfather": {"grandfather", "grandparent", "father", "mother"},
    "person_grandmother": {"grandmother", "grandparent", "father", "mother"},
    "person_sibling": {"sibling", "brother", "sister"},
    "person_child": {"child", "children", "son", "daughter"},
    "institution": {"work", "worked", "works", "employed", "professor", "university", "college", "alma", "studied"},
    "organization": {"company", "organization", "label", "network", "team", "club", "publisher", "studio"},
    "occupation": {"occupation", "profession", "job", "known", "worked", "served"},
    "nationality": {"nationality", "national", "citizen", "country", "born", "from"},
    "city": {"city", "born", "birth", "birthplace", "located"},
    "location": {"place", "city", "born", "birth", "birthplace", "located"},
    "date": {"date", "year", "born", "died", "released", "founded", "opened"},
    "year": {"date", "year", "born", "died", "released", "founded", "opened"},
}
LOCATION_GRANULARITY_QUERY_RE = re.compile(
    r"\b(?:place of birth|birthplace|born in|where (?:was|were|is|are).{0,80}\bborn|"
    r"where .{0,80}\b(?:die|died|death)|place of death|which city|what city|city of|located in)\b",
    re.IGNORECASE,
)


def _content_tokens(text: str) -> List[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(str(text or ""))]
    return [token for token in tokens if len(token) > 2 and token not in RELATION_STOPWORDS]


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
        "candidate_count": 0,
        "slot_candidate_count": 0,
        "selected_candidate": "",
        "selected_binding_score": 0.0,
        "selected_net_utility": 0.0,
        "selected_support_score": 0.0,
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


def _module_roots(row: Dict[str, Any]) -> List[str]:
    roots: List[str] = []
    for source in row.get("sources") or []:
        root = str(source or "").split(":lineage:", 1)[0].split(":", 1)[0]
        if root and root not in roots:
            roots.append(root)
    return roots


def _row_evidence_text(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    for value in row.get("titles") or []:
        parts.append(str(value or ""))
    for item in row.get("evidence") or []:
        if isinstance(item, dict):
            parts.append(str(item.get("title") or ""))
            parts.append(str(item.get("sentence_preview") or item.get("sentence") or ""))
        else:
            parts.append(str(item or ""))
    return " ".join(part for part in parts if part).strip()


def _candidate_pool_hash(rows: Sequence[Dict[str, Any]]) -> str:
    payload = [
        {
            "candidate": normalize_answer(str(row.get("candidate") or "")),
            "binding_score": round(
                float(row.get("binding_score", 0.0) or 0.0), 6
            ),
            "support_score": round(
                float(row.get("support_score", 0.0) or 0.0), 6
            ),
            "sources": sorted(str(value) for value in row.get("sources") or []),
            "titles": sorted(str(value) for value in row.get("titles") or []),
        }
        for row in rows
    ]
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _candidate_passes_location_granularity(
    *,
    before: str,
    candidate: str,
    answer_slot: str,
) -> Tuple[bool, str]:
    if answer_slot not in {"city", "location"}:
        return True, ""
    candidate_norm = normalize_answer(candidate)
    if not candidate_norm:
        return False, "empty_location_candidate"
    if answer_slot == "city":
        if candidate_norm in HIGH_LEVEL_LOCATION_KEYS:
            return False, "city_slot_high_level_location"
        if NATIONALITY_OR_COUNTRY_RE.fullmatch(str(candidate or "").strip()):
            return False, "city_slot_country_surface"
    before_parts = [
        normalize_answer(part)
        for part in str(before or "").split(",")
        if normalize_answer(part)
    ]
    if answer_slot == "city" and len(before_parts) >= 2 and candidate_norm in before_parts[1:]:
        if candidate_norm in HIGH_LEVEL_LOCATION_KEYS or NATIONALITY_OR_COUNTRY_RE.fullmatch(str(candidate or "").strip()):
            return False, "city_slot_geographic_parent"
    return True, ""


def _answer_slot_contract(question: str, answer_style: Optional[str]) -> Dict[str, Any]:
    target_type = infer_answer_target_type(question or "")
    candidate_target_type = "date" if target_type == "year" else target_type
    if candidate_target_type not in SUPPORTED_TARGET_TYPES:
        candidate_target_type = "person"
    answer_slot = _infer_answer_slot(question or "", candidate_target_type)
    candidate_target_type = _candidate_target_type_for_slot(answer_slot, candidate_target_type)
    slot_type = _slot_type_for_question(question or "", candidate_target_type)
    if answer_slot in {"full_date", "date", "year"}:
        slot_type = "date"
    elif answer_slot in {"number"}:
        slot_type = "number"
    elif answer_slot in {"occupation", "nationality", "organization", "institution"}:
        slot_type = "organization" if answer_slot == "institution" else answer_slot
    elif answer_slot in {"city", "county", "region", "location", "intersection"}:
        slot_type = "location"
    elif answer_slot.startswith("person_"):
        slot_type = "person"
    return {
        "target_type": target_type,
        "candidate_target_type": candidate_target_type,
        "answer_slot": answer_slot,
        "slot_type": slot_type,
        "answer_style": answer_style,
    }


def _collect_extended_module_candidates(
    lattice: Dict[str, Dict[str, Any]],
    module_name: str,
    module_info: Optional[Dict[str, Any]],
) -> None:
    info = module_info or {}
    if not isinstance(info, dict):
        return
    _collect_module_candidates(lattice, module_name, info)
    for field in (
        "top_candidates",
        "candidate_store_rows",
        "lattice_rows",
        "candidate_rows",
        "contract_candidates",
        "slot_candidates",
    ):
        rows = info.get(field) or []
        if isinstance(rows, list):
            _collect_from_rows(lattice, rows, f"{module_name}:{field}")
    selected = str(info.get("selected_candidate") or "").strip()
    candidate_rows = [row for row in (info.get("candidates") or []) if isinstance(row, dict)]
    if selected and not any(_same_answer(selected, str(row.get("candidate") or "")) for row in candidate_rows):
        _collect_from_rows(
            lattice,
            [{
                "candidate": selected,
                "score": info.get("selected_score", info.get("confidence", 0.0)),
                "provenance_count": info.get("selected_provenance_count", 1),
                "question_overlap": info.get("selected_question_overlap", info.get("question_overlap", 0.0)),
                "relation_role_score": info.get("selected_relation_score", info.get("relation_score", 0.0)),
                "answer_type_score": info.get("selected_answer_type_score", info.get("answer_type_score", 0.0)),
            }],
            f"{module_name}:selected",
        )


def _is_supported_format_or_alias_repair(
    *,
    question: str,
    before: str,
    after: str,
    answer_slot: str,
    candidate_target_type: str,
    candidate_in_base_docs: bool,
    support_gain: float,
    alias_min_support_gain: float,
) -> bool:
    if not before or not after or not candidate_in_base_docs:
        return False
    if candidate_target_type in DATE_NUMBER_TYPES and _same_date_value(before, after):
        return True
    if not _is_containment_variant(before, after):
        return False
    before_token_count = _token_count(before)
    after_token_count = _token_count(after)
    if after_token_count >= before_token_count:
        return False
    if support_gain < float(alias_min_support_gain):
        return False
    if _surface_rewrite_loss_reason(
        question=question or "",
        before=before,
        after=after,
        slot=answer_slot,
        target_type=candidate_target_type,
    ):
        return False
    before_norm = normalize_answer(before)
    after_norm = normalize_answer(after)
    if answer_slot in {"city", "location"}:
        if "," not in str(before or "") and before_norm != after_norm:
            return False
        if "," in str(before or ""):
            first_component = normalize_answer(str(before or "").split(",", 1)[0])
            if after_norm != first_component:
                return False
            if not LOCATION_GRANULARITY_QUERY_RE.search(question or ""):
                return False
    if candidate_target_type == "person" or answer_slot.startswith("person_"):
        if after_token_count < 2:
            return False
        if before_token_count <= 6 and not _looks_verbose_or_multi_answer(before, max(12, before_token_count + 2)):
            return False
    if answer_slot == "nationality" and before_token_count <= 2:
        if NATIONALITY_OR_COUNTRY_RE.fullmatch(str(before or "").strip()) and NATIONALITY_OR_COUNTRY_RE.fullmatch(str(after or "").strip()):
            return False
    if answer_slot in {"city", "location"} and ("," in before or LOCATION_GRANULARITY_QUERY_RE.search(question or "")):
        return after_token_count >= 1
    if answer_slot in {"occupation", "role_title"}:
        return after_token_count >= 2
    return bool(_looks_verbose_or_multi_answer(before, max(12, before_token_count - 1)))


def _relation_entailment_state(
    *,
    question: str,
    before: str,
    candidate: str,
    answer_slot: str,
    candidate_target_type: str,
    row: Dict[str, Any],
    relation_score: float,
    question_overlap: float,
    min_relation_score: float,
    min_question_overlap: float,
) -> Tuple[bool, str, Dict[str, Any]]:
    evidence_text = _row_evidence_text(row)
    evidence_tokens = set(_content_tokens(evidence_text))
    question_tokens = set(_content_tokens(question or ""))
    answer_tokens = set(_content_tokens(before or "")) | set(_content_tokens(candidate or ""))
    relation_tokens = {
        token for token in question_tokens
        if token not in answer_tokens
    }
    relation_overlap_count = len(relation_tokens & evidence_tokens)
    relation_denom = max(1, min(8, len(relation_tokens)))
    lexical_relation_overlap = relation_overlap_count / relation_denom
    slot_cues = set(ROLE_CUES_BY_SLOT.get(answer_slot, set()))
    if not slot_cues and candidate_target_type in {"date", "year", "number", "nationality", "organization"}:
        slot_cues = set(ROLE_CUES_BY_SLOT.get(candidate_target_type, set()))
    role_cue_hit = bool(slot_cues & evidence_tokens) if slot_cues else True
    candidate_tokens = set(_content_tokens(candidate or ""))
    candidate_mentioned = bool(candidate_tokens & evidence_tokens)
    strong_relation_score = float(relation_score or 0.0) >= float(min_relation_score)
    strong_question_overlap = float(question_overlap or 0.0) >= float(min_question_overlap)
    strong_lexical_overlap = lexical_relation_overlap >= 0.18 or relation_overlap_count >= 2
    role_specific = (
        answer_slot.startswith("person_")
        or answer_slot in {"institution", "organization", "occupation", "nationality", "city", "location"}
    )
    ok = bool(
        candidate_mentioned
        and (strong_relation_score or strong_question_overlap or strong_lexical_overlap)
        and (role_cue_hit or not role_specific or strong_relation_score)
    )
    if not candidate_mentioned:
        reason = "candidate_not_mentioned_in_relation_evidence"
    elif role_specific and not role_cue_hit and not strong_relation_score:
        reason = "missing_relation_role_cue"
    elif not (strong_relation_score or strong_question_overlap or strong_lexical_overlap):
        reason = "low_relation_entailment"
    else:
        reason = "relation_entailment_ok"
    return ok, reason, {
        "relation_score": round(float(relation_score or 0.0), 4),
        "question_overlap": round(float(question_overlap or 0.0), 4),
        "lexical_relation_overlap": round(float(lexical_relation_overlap), 4),
        "relation_overlap_count": relation_overlap_count,
        "role_cue_hit": bool(role_cue_hit),
        "candidate_mentioned": bool(candidate_mentioned),
        "role_specific": bool(role_specific),
    }


def _incumbent_slot_state(
    *,
    question: str,
    before: str,
    answer_slot: str,
    candidate_target_type: str,
) -> Tuple[bool, str]:
    if not before:
        return False, "empty_answer"
    ok, reason = _candidate_passes_answer_slot(
        before,
        answer_slot,
        candidate_target_type,
        question=question or "",
        evidence_text="",
    )
    return bool(ok), reason


def optimize_answer_slot_binding(
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
    provenance_answer_slot_store: Optional[Dict[str, Any]] = None,
    anchored_typed_candidate_table: Optional[Dict[str, Any]] = None,
    candidate_answer_selector: Optional[Dict[str, Any]] = None,
    relation_fact_join: Optional[Dict[str, Any]] = None,
    enabled: bool = False,
    apply_rewrite: bool = True,
    min_score: float = 0.78,
    generic_min_score: float = 0.84,
    min_binding_score: float = 0.42,
    min_margin: float = 0.04,
    min_support_gain: float = 0.04,
    min_slot_mismatch_gain: float = 0.08,
    alias_min_support_gain: float = 0.01,
    max_docs: int = 10,
    max_answer_tokens: int = 14,
    max_candidates: int = 18,
    min_base_provenance: int = 1,
    require_role_constraint: bool = True,
    protect_supported_answers: bool = True,
    enable_format_alias_rewrite: bool = True,
    enable_slot_mismatch_rewrite: bool = False,
    enable_unsupported_rewrite: bool = False,
    enable_cross_module_rewrite: bool = False,
    min_relation_entailment_score: float = 0.20,
    min_relation_question_overlap: float = 0.10,
) -> Tuple[str, Dict[str, Any]]:
    """Query-conditioned answer-slot binding over materialized evidence candidates.

    This operator is deliberately rank-stable: it does not alter retrieval. It
    compiles a lightweight answer contract from the question, materializes all
    answer-side candidates as a tiny provenance table, then rewrites only when
    the best candidate satisfies the requested slot and has enough base-evidence
    support to justify overriding the reader output.
    """
    before = str(answer or "").strip()
    meta = _new_meta(enabled=enabled, before=before)
    meta.update({
        "apply_rewrite": bool(apply_rewrite),
        "require_role_constraint": bool(require_role_constraint),
        "protect_supported_answers": bool(protect_supported_answers),
        "enable_format_alias_rewrite": bool(enable_format_alias_rewrite),
        "enable_slot_mismatch_rewrite": bool(enable_slot_mismatch_rewrite),
        "enable_unsupported_rewrite": bool(enable_unsupported_rewrite),
        "enable_cross_module_rewrite": bool(enable_cross_module_rewrite),
        "min_relation_entailment_score": float(min_relation_entailment_score),
        "min_relation_question_overlap": float(min_relation_question_overlap),
        "min_binding_score": float(min_binding_score),
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
        min_question_overlap=0.02,
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
    _collect_extended_module_candidates(lattice, "provenance_typed_candidate_store", provenance_typed_candidate_store)
    _collect_extended_module_candidates(lattice, "evidence_grounded_answer_extractor", evidence_grounded_answer_extractor)
    _collect_extended_module_candidates(lattice, "query_answer_contract", query_answer_contract)
    _collect_extended_module_candidates(lattice, "provenance_answer_slot_store", provenance_answer_slot_store)
    _collect_extended_module_candidates(lattice, "anchored_typed_candidate_table", anchored_typed_candidate_table)
    _collect_extended_module_candidates(lattice, "candidate_answer_selector", candidate_answer_selector)
    _collect_from_rows(lattice, (relation_fact_join or {}).get("candidate_table_rows") or [], "relation_fact_join")
    _collect_from_rows(lattice, (relation_fact_join or {}).get("reader_candidate_table_rows") or [], "relation_fact_join_reader_projection")
    _collect_from_rows(
        lattice,
        (relation_fact_join or {}).get("validated_terminal_candidate_rows") or [],
        "relation_fact_join_terminal_contract",
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
            risk_weight=0.66,
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
        base_source_count = int(features.get("base_source_count", 0) or 0)
        base_provenance_count = int(features.get("base_provenance_count", 0) or 0)
        binding_score = (
            0.48 * support_score
            + 0.20 * net_utility
            + 0.08 * min(1.0, cross_module_votes / 3.0)
            + 0.08 * type_signal
            + 0.06 * relation_score
            + 0.04 * question_overlap
            + (0.04 if candidate_in_base_docs else 0.0)
            + min(0.04, 0.015 * max(0, base_provenance_count - 1))
        )
        features.update({
            "candidate": candidate,
            "binding_score": round(max(0.0, min(1.25, binding_score)), 4),
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
            "sources": row.get("sources", [])[:6],
            "titles": row.get("titles", [])[:5],
            "evidence": row.get("evidence", [])[:3],
        })
        scored_rows.append(features)

    scored_rows.sort(
        key=lambda row: (
            -float(row.get("binding_score", 0.0) or 0.0),
            -float(row.get("support_score", 0.0) or 0.0),
            -int(row.get("cross_module_votes", 0) or 0),
            str(row.get("candidate", "")).lower(),
        )
    )
    scored_rows = scored_rows[:max(1, int(max_candidates))]
    candidate_pool_hash = _candidate_pool_hash(scored_rows)

    top = scored_rows[0] if scored_rows else None
    second = scored_rows[1] if len(scored_rows) > 1 else None
    after = str(top.get("candidate") or "").strip() if top else ""
    binding_score = float(top.get("binding_score", 0.0) or 0.0) if top else 0.0
    second_binding = float((second or {}).get("binding_score", 0.0) or 0.0)
    margin = binding_score - second_binding
    top_support = float(top.get("support_score", 0.0) or 0.0) if top else 0.0
    support_gain = top_support - float(answer_support_score or 0.0)
    candidate_in_base_docs = bool(top.get("candidate_in_base_docs")) if top else False
    base_source_count = int(top.get("base_source_count", 0) or 0) if top else 0
    base_provenance_count = int(top.get("base_provenance_count", 0) or 0) if top else 0
    cross_module_votes = int(top.get("cross_module_votes", 0) or 0) if top else 0
    relation_entailment_ok = bool(top.get("relation_entailment_match")) if top else False
    relation_entailment_reason = str(top.get("relation_entailment_reason") or "") if top else ""
    required_base_provenance = max(1, int(min_base_provenance))
    base_provenance_ok = bool(
        base_provenance_count >= required_base_provenance
        or (candidate_in_base_docs and cross_module_votes >= 2)
    )
    slot_mismatch_repair = bool(
        top
        and not incumbent_slot_ok
        and top.get("explicit_slot_match")
        and candidate_in_base_docs
        and support_gain >= float(min_slot_mismatch_gain)
    )
    unsupported_answer_repair = bool(
        top
        and not answer_supported
        and candidate_in_base_docs
        and support_gain >= float(min_support_gain)
    )
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
    cross_module_contract_repair = bool(
        top
        and candidate_in_base_docs
        and cross_module_votes >= 2
        and base_provenance_ok
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
        "candidate_pool_hash": candidate_pool_hash,
        "selected_candidate": after,
        "selected_binding_score": round(binding_score, 4),
        "selected_net_utility": round(float(top.get("net_utility", 0.0) or 0.0), 4) if top else 0.0,
        "selected_support_score": round(top_support, 4),
        "selected_source_count": int(top.get("source_count", 0) or 0) if top else 0,
        "selected_provenance_count": int(top.get("provenance_count", 0) or 0) if top else 0,
        "selected_base_source_count": base_source_count,
        "selected_base_provenance_count": base_provenance_count,
        "required_base_provenance": required_base_provenance,
        "base_provenance_ok": bool(base_provenance_ok),
        "selected_cross_module_votes": cross_module_votes,
        "selected_candidate_in_base_docs": bool(candidate_in_base_docs),
        "selected_relation_entailment_match": bool(relation_entailment_ok),
        "selected_relation_entailment_reason": relation_entailment_reason,
        "selected_relation_entailment": top.get("relation_entailment", {}) if top else {},
        "margin": round(margin, 4),
        "support_gain": round(support_gain, 4),
        "rewrite_safety_reason": rewrite_safety_reason,
        "slot_mismatch_repair": bool(slot_mismatch_repair),
        "unsupported_answer_repair": bool(unsupported_answer_repair),
        "format_or_alias_repair": bool(format_or_alias_repair),
        "cross_module_contract_repair": bool(cross_module_contract_repair),
        "protected_supported_answer": bool(protected_supported_answer),
        "top_candidates": scored_rows[:6],
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
    if not base_provenance_ok:
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
    if binding_score < float(min_binding_score):
        meta["reason"] = "low_binding_score"
        return before, meta
    if top_support < confidence_floor and not (format_or_alias_repair or slot_mismatch_repair):
        meta["reason"] = "low_support_score"
        return before, meta
    if margin < float(min_margin) and not (format_or_alias_repair or slot_mismatch_repair):
        meta["reason"] = "low_margin"
        return before, meta
    if not (
        format_or_alias_repair
        or slot_mismatch_repair
        or unsupported_answer_repair
        or cross_module_contract_repair
    ):
        meta["reason"] = "no_safe_rewrite_case"
        return before, meta

    shadow_reason = (
        "format_or_alias_slot_binding"
        if format_or_alias_repair
        else "slot_mismatch_binding"
        if slot_mismatch_repair
        else "unsupported_answer_binding"
        if unsupported_answer_repair
        else "cross_module_contract_binding"
    )
    blocked_rewrite_reason = ""
    if shadow_reason == "format_or_alias_slot_binding" and not enable_format_alias_rewrite:
        blocked_rewrite_reason = "format_or_alias_shadow_only"
    elif shadow_reason == "slot_mismatch_binding":
        if not enable_slot_mismatch_rewrite:
            blocked_rewrite_reason = "slot_mismatch_shadow_only"
        elif not relation_entailment_ok:
            blocked_rewrite_reason = f"slot_mismatch_{relation_entailment_reason or 'relation_entailment_failed'}"
    elif shadow_reason == "unsupported_answer_binding":
        if not enable_unsupported_rewrite:
            blocked_rewrite_reason = "unsupported_answer_shadow_only"
        elif not relation_entailment_ok:
            blocked_rewrite_reason = f"unsupported_answer_{relation_entailment_reason or 'relation_entailment_failed'}"
    elif shadow_reason == "cross_module_contract_binding":
        if not enable_cross_module_rewrite:
            blocked_rewrite_reason = "cross_module_contract_shadow_only"
        elif not relation_entailment_ok:
            blocked_rewrite_reason = f"cross_module_contract_{relation_entailment_reason or 'relation_entailment_failed'}"
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
    if blocked_rewrite_reason:
        meta["action"] = "shadow_rewrite"
        meta["reason"] = blocked_rewrite_reason
        return before, meta

    meta.update({
        "changed": True,
        "action": "rewrite",
        "after": after,
    })
    return after, meta
