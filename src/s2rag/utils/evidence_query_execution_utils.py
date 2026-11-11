from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from .support_aware_cbo_utils import (
    SampledActionCostModel,
    support_aware_action_score,
)
from .bswc_optimizer_utils import (
    BSWC_POLICIES,
    adaptive_risk_bounded_witness_first_action,
    binding_qualified_continuation_first_action,
    binding_aware_bounded_first_action,
    continuation_safe_witness_first_action,
    deterministic_fixed_order_first_action,
    deterministic_random_source_first_action,
    exact_adaptive_supported_witness_first_action,
    feasibility_preserving_witness_first_action,
    frozen_source_frontier_snapshot,
    precedence_aware_witness_first_action,
    quality_first_precedence_witness_first_action,
    risk_aware_witness_search_first_action,
    source_action_score,
)
from .bswc_feedback_utils import (
    CALIBRATED_POLICIES,
    FEEDBACK_POLICIES,
    STOPPING_POLICIES,
    WITNESS_GAIN_POLICIES,
    HierarchicalSourceValueModel,
    action_features,
    feedback_adjusted_probability,
    should_stop_low_value,
    source_action_success,
    update_query_feedback,
)
from .bswc_support_oracle_utils import support_oracle_first_action
from .bswc_frontier_exposure_oracle_utils import (
    annotate_identity_safe_budget_actions,
    expose_identity_safe_source_actions,
    identity_safe_reserved_scan_limit,
)
from .relation_fact_join_utils import (
    PLAN_COMPLETE,
    PLAN_COVERAGE_CONTRACT_VERSION,
    TERMINAL_TYPE_INCOMPATIBLE,
    role_aware_predicate_contract_compatibility,
    source_local_kinship_role_conflict,
    terminal_answer_domain_compatibility,
)
from .canonicalize_utils import equivalent_complete_date_surfaces


LOOKUP_JOIN = "LOOKUP_JOIN"
COMPLETE = "COMPLETE"
VERIFY = "VERIFY"
BOUND_DENSE_PROBE = "BOUND_DENSE_PROBE"
SOURCE_INSPECT = "SOURCE_INSPECT"
STOP = "STOP"

UNVERIFIED = "UNVERIFIED"
SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
INSUFFICIENT = "INSUFFICIENT"
UNKNOWN = "UNKNOWN"

TUPLE_INDEXED = "INDEXED"
TUPLE_RECOVERED = "RECOVERED"
TUPLE_REJECTED = "REJECTED"
TUPLE_DEFERRED = "DEFERRED"

OBLIGATION_OPEN = "OPEN"
OBLIGATION_SATISFIED = "SATISFIED"
OBLIGATION_INVALIDATED = "INVALIDATED"
OBLIGATION_REOPENED = "REOPENED"
OBLIGATION_COMPLETED = "COMPLETED"

POLICIES = {
    "fixed_early_verify",
    "fixed_late_verify",
    "greedy_immediate",
    "adaptive_dependency",
    "adaptive_dependency_frozen",
    "feasibility_constrained_adaptive",
    "support_aware_cbo",
    *BSWC_POLICIES,
}


EXECUTION_PHASE_STATUS_VERSION = "query_execution_phase_status_v1"


def _execution_phase_status(
    plan: Mapping[str, Any],
    *,
    root_binding_succeeded: bool = False,
    execution_entered: bool = False,
    supported_witness_complete: bool = False,
    structured_answer_admitted: bool = False,
    structured_answer_replaced_reader: bool = False,
    failure_stage: str = "",
    failure_reason: str = "",
) -> Dict[str, Any]:
    compile_phase = dict(plan.get("compile_phase_status") or {})
    generated = bool(compile_phase.get("plan_generated", bool(plan)))
    static_valid = bool(
        compile_phase.get(
            "plan_static_valid",
            str(plan.get("compile_status") or "") == "compiled",
        )
    )
    return {
        "version": EXECUTION_PHASE_STATUS_VERSION,
        "plan_id": str(plan.get("plan_id") or compile_phase.get("plan_id") or ""),
        "plan_generated": generated,
        "plan_static_valid": static_valid,
        "root_binding_succeeded": bool(root_binding_succeeded),
        "execution_entered": bool(execution_entered),
        "supported_witness_complete": bool(supported_witness_complete),
        "structured_answer_admitted": bool(structured_answer_admitted),
        "structured_answer_replaced_reader": bool(
            structured_answer_replaced_reader
        ),
        "failure_stage": str(failure_stage or ""),
        "failure_reason": str(failure_reason or ""),
    }


def update_answer_phase_status(
    trace: Mapping[str, Any],
    *,
    structured_answer_replaced_reader: bool,
) -> Dict[str, Any]:
    """Update only the answer-application phase of an execution trace."""
    result = dict(trace or {})
    phases = dict(result.get("phase_status") or {})
    phases["structured_answer_replaced_reader"] = bool(
        structured_answer_replaced_reader
    )
    if (
        phases.get("structured_answer_admitted")
        and not structured_answer_replaced_reader
        and not phases.get("failure_stage")
    ):
        phases["failure_stage"] = "structured_answer_application"
        phases["failure_reason"] = "admitted_answer_not_selected"
    result["phase_status"] = phases
    return result


def promote_plan_bound_evidence_docs(
    sorted_doc_ids: Sequence[int],
    sorted_doc_scores: Sequence[float],
    relation_fact_join: Mapping[str, Any],
    paper_evidence_execution: Mapping[str, Any],
    *,
    enabled: bool = False,
    protect_dense_top_k: int = 1,
    max_title_docs: int = 3,
    selection_mode: str = "legacy",
    target_top_k: int = 5,
    dense_candidate_top_k: int = 10,
) -> Tuple[List[int], List[float], Dict[str, Any]]:
    """Project plan-bound fact partitions into the final retrieval order.

    COMPLETE already probes an entity-title partition while executing a
    logical fact-join plan.  Historically those document ids were only kept
    in the execution trace, so retrieval Recall@k still reflected the dense
    list alone.  This projection preserves the strongest dense hit, then
    places provenance-verified join sources and critical title partitions in
    front of the remaining dense candidates.  It never promotes documents
    from an unbound dense completion probe.
    """
    doc_ids = [
        int(value)
        for value in list(sorted_doc_ids if sorted_doc_ids is not None else [])
    ]
    scores = [
        float(value)
        for value in list(sorted_doc_scores if sorted_doc_scores is not None else [])
    ]
    if len(scores) < len(doc_ids):
        scores.extend([0.0] * (len(doc_ids) - len(scores)))
    selection_mode = str(selection_mode or "legacy").strip().lower()
    if selection_mode not in {"legacy", "support_aware_topk"}:
        selection_mode = "legacy"
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "protect_dense_top_k": max(0, int(protect_dense_top_k)),
        "max_title_docs": max(0, int(max_title_docs)),
        "selection_mode": selection_mode,
        "target_top_k": max(1, int(target_top_k)),
        "dense_candidate_top_k": max(1, int(dense_candidate_top_k)),
        "verified_doc_ids": [],
        "title_partition_doc_ids": [],
        "selected_title_doc_ids": [],
        "promoted_doc_ids": [],
        "displaced_top_k_doc_ids": [],
        "covered_obligations": [],
        "uncovered_obligations": [],
        "topk_selection_trace": [],
        "candidate_rows": [],
        "reason": "disabled" if not enabled else "no_plan_bound_candidates",
    }
    if not enabled or not doc_ids:
        return doc_ids, scores[:len(doc_ids)], info

    valid_doc_ids = set(doc_ids)
    verified_doc_ids: List[int] = []
    for raw_doc_id in relation_fact_join.get("reader_candidate_doc_ids") or []:
        try:
            doc_id = int(raw_doc_id)
        except (TypeError, ValueError):
            continue
        if doc_id in valid_doc_ids and doc_id not in verified_doc_ids:
            verified_doc_ids.append(doc_id)

    logical_plan = paper_evidence_execution.get("logical_plan") or {}
    operators = {
        str(operator.get("operator_id") or ""): operator
        for operator in logical_plan.get("operators") or []
    }
    critical_stages = {"intermediate_binding", "comparison_value", "final_answer"}
    title_doc_ids: List[int] = []
    title_rows: List[Dict[str, Any]] = []
    for trace in paper_evidence_execution.get("operator_trace") or []:
        if str(trace.get("operator") or "") != COMPLETE:
            continue
        obligation_id = str(trace.get("target_obligation") or "")
        operator = operators.get(obligation_id, {})
        stage = str(operator.get("slot_plan_stage") or "")
        if stage not in critical_stages:
            continue
        physical_access = trace.get("physical_access") or {}
        if not bool(physical_access.get("title_catalog_hit")):
            continue
        for raw_doc_id in physical_access.get("title_catalog_doc_ids") or []:
            try:
                doc_id = int(raw_doc_id)
            except (TypeError, ValueError):
                continue
            if doc_id not in valid_doc_ids:
                continue
            if not any(row["doc_id"] == doc_id for row in title_rows):
                title_rows.append({
                    "doc_id": doc_id,
                    "obligation_id": obligation_id,
                    "stage": stage,
                    "stage_index": _as_int(operator.get("stage_index"), 10**6),
                    "verified_match": _as_int(
                        physical_access.get("title_probe_verified_match_count")
                    ) > 0,
                })
            if doc_id not in title_doc_ids and len(title_doc_ids) < info["max_title_docs"]:
                title_doc_ids.append(doc_id)

    if selection_mode == "support_aware_topk":
        candidate_rows: Dict[int, Dict[str, Any]] = {}

        def mark_candidate(
            doc_id: int,
            source: str,
            *,
            obligation_id: str = "",
            stage: str = "",
            stage_index: int = 10**6,
            supported: bool = False,
            selected_witness: bool = False,
            title_verified: bool = False,
        ) -> None:
            if doc_id not in valid_doc_ids:
                return
            row = candidate_rows.setdefault(doc_id, {
                "doc_id": doc_id,
                "sources": [],
                "obligation_ids": [],
                "stages": [],
                "stage_index": stage_index,
                "supported": False,
                "selected_witness": False,
                "title_verified": False,
            })
            if source and source not in row["sources"]:
                row["sources"].append(source)
            if obligation_id and obligation_id not in row["obligation_ids"]:
                row["obligation_ids"].append(obligation_id)
            if stage and stage not in row["stages"]:
                row["stages"].append(stage)
            row["stage_index"] = min(_as_int(row["stage_index"], 10**6), stage_index)
            row["supported"] = bool(row["supported"] or supported)
            row["selected_witness"] = bool(
                row["selected_witness"] or selected_witness
            )
            row["title_verified"] = bool(row["title_verified"] or title_verified)

        original_rank = {doc_id: index for index, doc_id in enumerate(doc_ids)}
        dense_limit = min(len(doc_ids), info["dense_candidate_top_k"])
        for doc_id in doc_ids[:dense_limit]:
            mark_candidate(doc_id, "dense_topk")

        reader_fact_ids = {
            str(value)
            for value in paper_evidence_execution.get("reader_evidence_fact_ids") or []
        }
        verified_doc_set = set(verified_doc_ids)
        for fact in paper_evidence_execution.get("facts") or []:
            if str(fact.get("support_status") or "") != SUPPORTED:
                continue
            try:
                doc_id = int(fact.get("source_id"))
            except (TypeError, ValueError):
                continue
            obligation_id = str(fact.get("operator_id") or "")
            operator = operators.get(obligation_id, {})
            stage = str(operator.get("slot_plan_stage") or "")
            selected_witness = bool(
                str(fact.get("fact_id") or "") in reader_fact_ids
                or doc_id in verified_doc_set
            )
            mark_candidate(
                doc_id,
                "reader_witness" if selected_witness else "supported_fact",
                obligation_id=obligation_id,
                stage=stage,
                stage_index=_as_int(operator.get("stage_index"), 10**6),
                supported=True,
                selected_witness=selected_witness,
            )

        for doc_id in verified_doc_ids:
            mark_candidate(doc_id, "reader_witness", supported=True, selected_witness=True)
        for row in title_rows:
            mark_candidate(
                row["doc_id"],
                "title_partition",
                obligation_id=row["obligation_id"],
                stage=row["stage"],
                stage_index=row["stage_index"],
                title_verified=row["verified_match"],
            )

        target_count = min(len(doc_ids), info["target_top_k"])
        protect_count = min(info["protect_dense_top_k"], target_count)
        selected = list(doc_ids[:protect_count])
        selected_set = set(selected)
        covered = set()
        selected_title_count = 0
        selection_trace: List[Dict[str, Any]] = []

        for doc_id in selected:
            row = candidate_rows.get(doc_id, {})
            obligations = list(row.get("obligation_ids") or [])
            covered.update(obligations)
            selection_trace.append({
                "doc_id": doc_id,
                "reason": "protected_dense_prefix",
                "new_obligations": obligations,
                "dense_rank_before": original_rank[doc_id] + 1,
            })

        def decision_class(row: Mapping[str, Any]) -> Tuple[int, str]:
            obligations = set(row.get("obligation_ids") or [])
            adds_coverage = bool(obligations - covered)
            sources = set(row.get("sources") or [])
            title_only = sources == {"title_partition"}
            if title_only and selected_title_count >= info["max_title_docs"]:
                return 99, "title_budget_exhausted"
            if row.get("selected_witness"):
                return 0, "selected_supported_witness"
            if row.get("supported") and adds_coverage:
                return 1, "supported_new_obligation"
            if row.get("title_verified") and adds_coverage:
                return 2, "verified_title_new_obligation"
            if "title_partition" in sources and adds_coverage:
                return 3, "title_new_obligation"
            if "dense_topk" in sources:
                return 4, "dense_residual"
            if row.get("supported"):
                return 5, "supported_redundant_obligation"
            if row.get("title_verified"):
                return 6, "verified_title_redundant_obligation"
            return 7, "title_redundant_obligation"

        while len(selected) < target_count:
            ranked = []
            for doc_id, row in candidate_rows.items():
                if doc_id in selected_set:
                    continue
                class_rank, reason = decision_class(row)
                if class_rank >= 99:
                    continue
                stage_rank = (
                    _as_int(row.get("stage_index"), 10**6)
                    if class_rank < 4 else 10**6
                )
                ranked.append((
                    class_rank,
                    stage_rank,
                    original_rank.get(doc_id, 10**9),
                    doc_id,
                    reason,
                ))
            if not ranked:
                fallback = next(
                    (doc_id for doc_id in doc_ids if doc_id not in selected_set),
                    None,
                )
                if fallback is None:
                    break
                ranked = [(8, 10**6, original_rank[fallback], fallback, "dense_fallback")]
            _, _, _, doc_id, reason = min(ranked)
            row = candidate_rows.get(doc_id, {})
            obligations = list(row.get("obligation_ids") or [])
            new_obligations = sorted(set(obligations) - covered)
            selected.append(doc_id)
            selected_set.add(doc_id)
            covered.update(obligations)
            if "title_partition" in set(row.get("sources") or []):
                selected_title_count += 1
            selection_trace.append({
                "doc_id": doc_id,
                "reason": reason,
                "new_obligations": new_obligations,
                "dense_rank_before": original_rank.get(doc_id, -1) + 1,
            })

        reordered = selected + [doc_id for doc_id in doc_ids if doc_id not in selected_set]
        required_obligations = {
            operator_id
            for operator_id, operator in operators.items()
            if str(operator.get("slot_plan_stage") or "") in critical_stages
        }
        selected_title_doc_ids = [
            row["doc_id"]
            for row in selection_trace
            if "title_partition" in set(
                (candidate_rows.get(row["doc_id"]) or {}).get("sources") or []
            )
        ]
        for row in candidate_rows.values():
            row["dense_rank_before"] = original_rank.get(row["doc_id"], -1) + 1
            row["selected_top_k"] = row["doc_id"] in selected_set
        info.update({
            "selected_title_doc_ids": selected_title_doc_ids,
            "covered_obligations": sorted(covered),
            "uncovered_obligations": sorted(required_obligations - covered),
            "topk_selection_trace": selection_trace,
            "candidate_rows": sorted(
                candidate_rows.values(),
                key=lambda row: original_rank.get(row["doc_id"], 10**9),
            ),
            "displaced_top_k_doc_ids": [
                doc_id for doc_id in doc_ids[:target_count] if doc_id not in selected_set
            ],
        })
    else:
        protect_count = min(info["protect_dense_top_k"], len(doc_ids))
        protected = doc_ids[:protect_count]
        reordered = []
        reordered_set = set()
        for group in (protected, verified_doc_ids, title_doc_ids, doc_ids[protect_count:]):
            for doc_id in group:
                if doc_id in reordered_set:
                    continue
                reordered.append(doc_id)
                reordered_set.add(doc_id)

    score_by_doc = dict(zip(doc_ids, scores))



    high = scores[0] if scores else 1.0
    epsilon = max(abs(high) * 1e-7, 1e-7)
    reordered_scores = [high - epsilon * index for index in range(len(reordered))]
    protect_count = min(info["protect_dense_top_k"], len(doc_ids))
    protected_set = set(doc_ids[:protect_count])
    for index, doc_id in enumerate(reordered):
        if doc_id in protected_set:
            reordered_scores[index] = score_by_doc.get(doc_id, reordered_scores[index])

    original_rank = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    promoted = [
        doc_id for index, doc_id in enumerate(reordered)
        if index < original_rank[doc_id]
    ]
    info.update({
        "applied": reordered != doc_ids,
        "verified_doc_ids": verified_doc_ids,
        "title_partition_doc_ids": title_doc_ids,
        "promoted_doc_ids": promoted,
        "reason": (
            f"{selection_mode}_plan_bound_projection"
            if reordered != doc_ids else "already_ranked"
        ),
    })
    return reordered, reordered_scores, info


def _tokens(value: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").lower())


def _evidence_token_count(value: Any) -> int:
    """Count source-text units separately from tokenizer-specific API tokens."""
    return len(re.findall(r"\S+", str(value or "")))


def _key(value: Any) -> str:
    return " ".join(_tokens(value))


def _is_abstaining_answer(value: Any) -> bool:
    text = str(value or "").strip().casefold()
    return bool(re.match(
        r"^(?:n/?a|none|unknown|not\s+(?:specified|provided|mentioned|"
        r"available|given)|cannot\s+(?:determine|answer)|insufficient\b|"
        r"no\s+(?:answer|information)\b)",
        text,
    ))


def _year_values(value: Any) -> set[str]:
    """Extract plausible four-digit years for lineage qualification."""
    return set(re.findall(r"(?<!\d)(?:1[0-9]{3}|20[0-9]{2})(?!\d)", str(value or "")))


def _ordinal_identity_qualifiers(value: Any) -> set[str]:
    """Return title ordinals that distinguish otherwise identical names."""
    text = str(value or "")
    numeric = {
        token.casefold()
        for token in re.findall(
            r"(?<![a-z0-9])\d+(?:st|nd|rd|th)(?![a-z0-9])",
            text,
            flags=re.IGNORECASE,
        )
    }
    roman = set()
    for match in re.finditer(r"\b[IVXLCDM]+\b", text):
        token = match.group(0)




        if len(token) == 1 and not text[:match.start()].strip():
            continue



        before = text[match.start() - 1] if match.start() > 0 else ""
        after = text[match.end()] if match.end() < len(text) else ""
        if (before and before in "-\u2013\u2014") or (
            after and after in "-\u2013\u2014"
        ):
            continue
        roman.add(token.casefold())
    return numeric | roman


_IDENTITY_TITLE_TERMS = {
    "archbishop", "baron", "baroness", "bishop", "count", "countess",
    "duchess", "duke", "earl", "emperor", "empress", "king", "lady",
    "lord", "margrave", "margravine", "marquess", "marchioness", "pope",
    "prince", "princess", "queen", "saint", "viscount", "viscountess",
}


def _identity_title_qualifiers(value: Any) -> set[str]:
    """Return identity-bearing ordinals and titled-name qualifiers.

    Normalized entity keys intentionally collapse punctuation and honorific
    appositives. That is useful for recall, but ``Thomas Howard, 2nd Earl of
    Effingham`` and ``Thomas Howard, 2nd Duke of Norfolk`` must not become the
    same downstream binding merely because both normalize to ``howard
    thomas``. Descriptive occupations remain excluded.
    """
    text = str(value or "")
    qualifiers = set(_ordinal_identity_qualifiers(text))
    title_pattern = "|".join(
        re.escape(value) for value in sorted(_IDENTITY_TITLE_TERMS, key=len, reverse=True)
    )
    for match in re.finditer(rf"\b(?P<title>{title_pattern})\b", text, re.IGNORECASE):
        qualifiers.add(match.group("title").casefold())
        suffix = text[match.end():]
        of_match = re.match(r"\s+of\s+(?P<complement>.+)", suffix, re.IGNORECASE)
        if of_match is None:
            continue



        complement = re.split(
            r"[,;()]|\s+(?:and|but|who|which|that|was|is|were|are|became|"
            r"died|born|married)\b",
            of_match.group("complement"),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        qualifiers.update(_tokens(complement)[:6])
    return qualifiers


def _local_identity_qualifiers(
    source_sentence: Any,
    verified_surface: Any,
) -> set[str]:
    """Extract qualifiers attached to the verified output entity only.

    A source sentence can mention several people and works.  Treating every
    ordinal in that sentence as a qualifier of the output entity incorrectly
    couples unrelated mentions.  We keep qualifiers present in the verified
    surface itself and in an immediately following appositive title.
    """
    source = str(source_sentence or "")
    surface = str(verified_surface or "").strip()
    qualifiers = set(_identity_title_qualifiers(surface))
    if not source or not surface:
        return qualifiers

    surface_pattern = re.escape(surface).replace(r"\ ", r"\s+")
    match = re.search(surface_pattern, source, flags=re.IGNORECASE)
    if match is None:
        return qualifiers

    suffix = source[match.end():]
    appositive = re.match(r"\s*[,()]\s*([^.;)]{0,160})", suffix)
    if appositive is None:
        return qualifiers





    if re.search(
        r"\b(?:also\s+known\s+as|known\s+as|alias(?:ed)?\s+as|"
        r"persona\s+as|stage\s+name|pen\s+name|birth\s+name|"
        r"born\s+as|n[\u00e9e]e)\b",
        appositive.group(1),
        flags=re.IGNORECASE,
    ):
        return qualifiers
    local_title = re.split(
        r"(?:^|\s+)(?:and|but|who|which|that|was|is|became|died|born|married)\b",
        appositive.group(1),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    qualifiers.update(_identity_title_qualifiers(local_title))
    return qualifiers


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, _as_float(value)))


def _sentences(text: str) -> List[Tuple[int, str]]:



    marker = "<S2RAG_INITIAL_DOT>"
    protected = re.sub(
        r"\b([A-Z])\.(?=\s*(?:[A-Z]\.|[A-Z][a-z]))",
        rf"\1{marker}",
        str(text or ""),
    )
    parts = re.split(r"(?<=[.!?])\s+|[\r\n]+", protected)
    return [
        (idx, part.replace(marker, ".").strip())
        for idx, part in enumerate(parts)
        if part.strip()
    ]


def _source_surface(text: str, value: str) -> str:
    """Recover the source-preserving surface for a normalized fact endpoint."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    pattern = r"\s+".join(re.escape(part) for part in raw.split())
    match = re.search(pattern, str(text or ""), re.IGNORECASE)
    return match.group(0) if match else ""


def _exact_source_endpoint_surface(text: Any, value: Any) -> str:
    """Locate a tuple endpoint on token boundaries in its provenance text."""
    tokens = re.findall(r"[^\W_]+", str(value or ""), flags=re.UNICODE)
    if not tokens:
        return ""
    pattern = r"(?<!\w)" + r"[\W_]+".join(
        re.escape(token) for token in tokens
    ) + r"(?!\w)"
    match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    return match.group(0) if match else ""


def _binding_to_obligation_contract(
    fact: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate query-binding evidence before a tuple can extend a lineage.

    Predicate, role, direction, and broad output type are checked by canonical
    obligation admission.  This narrower contract closes the remaining gap
    between a normalized input key and the concrete query entity that key is
    intended to denote, and between an extracted entity surface and a usable
    downstream join key.
    """
    contract = dict(fact.get("obligation_contract") or {})
    if not bool(contract.get("binding_contract_enforced")):
        return {
            "status": "OFF",
            "reason": "binding_contract_disabled",
        }
    triple = list(fact.get("triple") or [])
    input_side = str(contract.get("bound_input_side") or "")
    output_side = str(fact.get("bound_output_role") or "")
    if len(triple) != 3 or input_side not in {"subject", "object"}:
        return {"status": "REJECTED", "reason": "invalid_binding_contract"}
    if output_side not in {"subject", "object"} or output_side == input_side:
        return {"status": "REJECTED", "reason": "invalid_binding_direction"}

    input_index = 0 if input_side == "subject" else 2
    output_index = 0 if output_side == "subject" else 2
    required_qualifiers = {
        str(value).strip().lower()
        for value in contract.get("required_input_qualifier_tokens") or []
        if str(value).strip()
    }
    identity_text = " ".join([
        str(triple[input_index] or ""),
        str(fact.get("source_title") or fact.get("completion_source_title") or ""),
    ])
    identity_tokens = set(_tokens(identity_text))
    if required_qualifiers and not required_qualifiers.issubset(identity_tokens):
        return {
            "status": "REJECTED",
            "reason": "input_identity_qualifier_mismatch",
            "required_input_qualifier_tokens": sorted(required_qualifiers),
            "observed_input_identity_tokens": sorted(identity_tokens),
        }

    if bool(contract.get("output_source_span_required")):
        source_text = str(
            fact.get("source_sentence_span")
            or fact.get("completion_sentence")
            or fact.get("source_text")
            or ""
        )
        source_surface = _exact_source_endpoint_surface(
            source_text, triple[output_index]
        )
        if not source_surface:
            return {
                "status": "REJECTED",
                "reason": "output_endpoint_not_source_aligned",
            }

    if (
        bool(contract.get("requires_downstream_join"))
        and bool(contract.get("output_joinability_checked"))
        and not bool(contract.get("output_joinable"))
    ):
        return {
            "status": "DEFERRED",
            "reason": "output_binding_not_joinable",
        }
    return {
        "status": "ADMITTED",
        "reason": "binding_obligation_contract_satisfied",
        "required_input_qualifier_tokens": sorted(required_qualifiers),
    }


def _clean_root_answer_surface(value: Any) -> str:
    """Remove query scaffolding accidentally absorbed into a title span."""
    surface = " ".join(str(value or "").strip().split())
    if not surface:
        return ""
    previous = ""
    while previous != surface:
        previous = surface
        surface = re.sub(
            r"^(?:was|is|are|were|did|does|do|of|out\s+of)\s+",
            "",
            surface,
            count=1,
            flags=re.IGNORECASE,
        ).strip(" ,:;-?")
    return surface


def _identity_initials(value: Any) -> Tuple[str, ...]:
    """Preserve one-letter initials that the bag-of-token entity key drops."""
    words = re.findall(r"[^\W_]+", str(value or ""), flags=re.UNICODE)
    return tuple(word.casefold() for word in words if len(word) == 1 and word.isalpha())


def _initial_sensitive_identity_matches(bound_surface: Any, endpoint_surface: Any) -> bool:
    required = _identity_initials(bound_surface)
    if len(required) < 2:
        return True
    observed = _identity_initials(endpoint_surface)
    return observed[:len(required)] == required


def _json_safe(value: Any) -> Any:
    """Return a deterministic JSON-safe copy of an execution artifact."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe(item) for item in value), key=lambda item: str(item))
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    return value


@dataclass
class ExecutionBudget:
    max_actions: int = 12
    max_retrieval_calls: int = 2
    max_model_calls: int = 2
    max_scan_rows: int = 5000
    max_tokens: Optional[int] = None
    reserve_model_calls: int = 0
    reserve_tokens: int = 0


@dataclass
class CostLedger:
    actions: int = 0
    scan_rows: int = 0
    retrieval_calls: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    evidence_tokens: int = 0
    unknown_token_events: int = 0
    operator_wall_ms: float = 0.0
    planning_wall_ms: float = 0.0
    cache_hits: int = 0
    budget_violations: List[str] = field(default_factory=list)

    def public(self) -> Dict[str, Any]:
        return {
            "actions": self.actions,
            "scan_rows": self.scan_rows,
            "retrieval_calls": self.retrieval_calls,
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "evidence_tokens": self.evidence_tokens,
            "total_tokens": (
                self.input_tokens + self.output_tokens + self.evidence_tokens
            ),
            "unknown_token_events": self.unknown_token_events,
            "operator_wall_ms": round(self.operator_wall_ms, 6),
            "planning_wall_ms": round(self.planning_wall_ms, 6),
            "cache_hits": self.cache_hits,
            "budget_violations": list(self.budget_violations),
        }


class EvidenceBackend(Protocol):
    verifier_version: str

    def initial_bindings(
        self, question: str, logical_plan: Mapping[str, Any]
    ) -> List[Dict[str, Any]]: ...

    def lookup_join(
        self, binding: Mapping[str, Any], operator: Mapping[str, Any]
    ) -> Dict[str, Any]: ...

    def complete(
        self, binding: Mapping[str, Any], operator: Mapping[str, Any]
    ) -> Dict[str, Any]: ...

    def enumerate_source_candidates(
        self, binding: Mapping[str, Any], operator: Mapping[str, Any]
    ) -> List[Dict[str, Any]]: ...

    def bound_dense_probe(
        self, binding: Mapping[str, Any], operator: Mapping[str, Any]
    ) -> Dict[str, Any]: ...

    def inspect_source(
        self,
        binding: Mapping[str, Any],
        operator: Mapping[str, Any],
        source_candidate: Mapping[str, Any],
    ) -> Dict[str, Any]: ...

    def verify(self, fact: Mapping[str, Any]) -> Dict[str, Any]: ...

    def estimate(
        self,
        action: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> Dict[str, Any]: ...


class RuleBasedSourceSupportVerifier:
    """Conservative, injectable source-support verifier.

    This verifier establishes a local text contract, not semantic truth. It is
    deliberately strict: uncertain passive, cross-sentence, or implicit facts
    are INSUFFICIENT and cannot enter the structured answer path.
    """





    version = "provenance_scoped_source_support_v14"
    _exclusive_families = {
        "birth", "death", "direction", "composition", "authorship",
        "distribution", "geopolitical_location",
    }
    _family_cues = {
        "birth": {"birth", "birthplace", "born"},
        "death": {"death", "died", "dead"},
        "direction": {"directed", "director"},
        "composition": {"composed", "composer", "music"},
        "authorship": {"author", "authored", "wrote", "written"},
        "distribution": {"released", "release", "premiered", "premiere"},
        "parent": {"parent", "father", "mother", "son", "daughter"},
        "spouse": {"spouse", "wife", "husband", "married"},
        "nationality": {"national", "nationality", "citizen", "citizenship"},
    }

    def __init__(self, *, enable_provenance_scope: bool = True) -> None:
        self.enable_provenance_scope = bool(enable_provenance_scope)
        if not self.enable_provenance_scope:
            self.version = "lexical_directional_source_support_v3_compat"

    @classmethod
    def _relation_is_negated(
        cls,
        sentence: str,
        subject: str,
        obj: str,
        relation_tokens: set,
    ) -> bool:
        """Detect negation local to the asserted relation.

        A sentence-wide check incorrectly treats unrelated uses such as
        ``ranked world No. 3`` as negating a nationality assertion.  Endpoint
        names are removed first so titles such as ``Never the Twain`` remain
        safe, and only a short prefix of an expressed relation cue is tested.
        """
        scoped = _key(sentence)
        for endpoint in sorted(
            {_key(subject), _key(obj)}, key=len, reverse=True
        ):
            if endpoint:
                scoped = scoped.replace(endpoint, " ")
        tokens = _tokens(scoped)
        relation_positions = [
            index for index, token in enumerate(tokens)
            if token in relation_tokens
        ]
        if not relation_positions:
            return False
        for relation_at in relation_positions:
            prefix = tokens[max(0, relation_at - 4):relation_at]
            if "not" in prefix and "only" in prefix:
                continue
            if set(prefix) & {"not", "never", "neither", "no"}:
                return True
        return False

    @staticmethod
    def _unresolved_kinship_description(value: Any) -> bool:
        """Return whether ``value`` is a role phrase, not a bound entity.

        A recovered parent endpoint must identify a person.  Phrases such as
        ``the sister of Emperor Nicholas II`` only describe an unresolved
        person and cannot safely become a join key.  Named titles such as
        ``Sister Mary`` are intentionally unaffected because the guard
        requires a determiner/possessive and an explicit ``of`` complement.
        """
        return bool(re.match(
            r"^\s*(?:the|his|her|their|a|an)\s+"
            r"(?:(?:elder|eldest|older|younger|youngest|first|second|third)\s+)?"
            r"(?:mother|father|parent|sister|brother|son|daughter|wife|"
            r"husband|spouse|consort)\s+of\b",
            str(value or ""),
            re.IGNORECASE,
        ))

    @staticmethod
    def _title_describes_entity(title: str, entity: str) -> bool:
        title_key, entity_key = _key(title), _key(entity)
        if not title_key or not entity_key:
            return False
        if title_key == entity_key:
            return True
        title_base = re.sub(r"\s*\([^)]*\)\s*$", "", str(title or "")).strip()
        if _key(title_base) == entity_key:
            return True
        entity_tokens = _tokens(entity)
        title_tokens = _tokens(title_base)
        if bool(
            len(entity_tokens) >= 2
            and len(title_tokens) >= 2
            and (
                entity_tokens == title_tokens[:len(entity_tokens)]
                or title_tokens == entity_tokens[:len(title_tokens)]
            )
        ):
            return True






        shorter, longer = (
            (title_tokens, entity_tokens)
            if len(title_tokens) <= len(entity_tokens)
            else (entity_tokens, title_tokens)
        )
        if (
            len(shorter) < 2
            or shorter[0] != longer[0]
            or shorter[-1] != longer[-1]
        ):
            return False
        cursor = 0
        for token in longer:
            if cursor < len(shorter) and token == shorter[cursor]:
                cursor += 1
        return cursor == len(shorter)

    @staticmethod
    def _title_reference_in_sentence(sentence: str, title: str) -> bool:
        raw_title = re.sub(r"\([^)]*\)", " ", str(title or "")).strip()




        identity_title = raw_title.split(",", 1)[0].strip()
        sentence_key = _key(sentence)
        identity_key = _key(identity_title)
        raw_title_key = _key(raw_title)
        if identity_key and (
            identity_key in sentence_key
            or (raw_title_key and raw_title_key in sentence_key)
        ):
            return True
        sentence_tokens = set(_tokens(sentence))
        if sentence_tokens & {"he", "she", "his", "her", "they", "their", "it", "its"}:
            return True





        if re.search(
            r"\b(?:the|this)\s+(?:album|book|film|novel|play|record|song|"
            r"subject|work)\b",
            str(sentence or ""),
            re.IGNORECASE,
        ):
            return True
        return bool(re.match(r"^\s*(?:is|was|are|were)\b", str(sentence or ""), re.I))

    @staticmethod
    def _title_pronoun_possessive_spouse_conflict(
        sentence: str, obj: str
    ) -> bool:
        """Reject a spouse phrase owned by a different explicit pronoun."""
        title_pronoun = re.match(
            r"^\s*(?P<pronoun>he|she|they)\b",
            str(sentence or ""),
            re.IGNORECASE,
        )
        if title_pronoun is None:
            return False
        possessive = re.search(
            rf"\b(?P<possessive>his|her|their)\s+"
            rf"(?:(?:former|first|second|third)\s+)?"
            rf"(?:wife|husband|spouse|consort)\b.{{0,80}}?"
            rf"\b{re.escape(_key(obj))}\b",
            _key(sentence),
            re.IGNORECASE,
        )
        if possessive is None:
            return False
        expected = {"he": "his", "she": "her", "they": "their"}
        return possessive.group("possessive").casefold() != expected[
            title_pronoun.group("pronoun").casefold()
        ]

    @staticmethod
    def _expanded_kinship_parent_surface(
        fact: Mapping[str, Any], sentence: str, subject: str
    ) -> str:
        canonical_relation = str(fact.get("canonical_relation") or "").casefold()
        if canonical_relation not in {"father_of", "mother_of", "parent_of"}:
            return ""
        match = re.search(
            r"\b(?:son|daughter|child)\s+of\s+"
            r"(?P<parent>[^.;]{1,160}?)"
            r"(?=\s*\([^)]*\b\d{3,4}\b[^)]*\)|"
            r"\s+(?:by|and)\s+(?:(?:his|her|the)\s+)?"
            r"(?:first\s+|second\s+)?(?:wife|mother)\b|"
            r",\s+(?:and|who)\b|[.;]|$)",
            str(sentence or ""),
            re.IGNORECASE,
        )
        if match is None:
            return ""
        candidate = match.group("parent").strip(" ,")
        subject_key = _key(subject)
        candidate_key = _key(candidate)
        subject_words = [value.casefold() for value in re.findall(
            r"[^\W_]+", str(subject or ""), flags=re.UNICODE
        )]
        candidate_words = [value.casefold() for value in re.findall(
            r"[^\W_]+", candidate, flags=re.UNICODE
        )]
        if (
            not subject_key
            or not candidate_key
            or candidate_words[:len(subject_words)] != subject_words
        ):
            return ""
        if len(_tokens(candidate)) <= len(_tokens(subject)) or len(_tokens(candidate)) > 10:
            return ""
        return candidate

    def _lifespan_support(
        self,
        *,
        fact: Mapping[str, Any],
        source_text: str,
        source_title: str,
        subject: str,
        obj: str,
        predicate_families: set,
    ) -> Optional[Dict[str, Any]]:
        if not self.enable_provenance_scope:
            return None
        family = "death" if "death" in predicate_families else (
            "birth" if "birth" in predicate_families else ""
        )
        if not family:
            return None
        title_subject_scope = self._title_describes_entity(source_title, subject)
        object_key = _key(obj)
        if not object_key:
            return None
        for sentence_id, sentence in _sentences(source_text):
            for match in re.finditer(r"\((?P<value>[^)]{3,120})\)", sentence):
                parenthetical = match.group("value")
                prefix = sentence[:match.start()]
                subject_tokens = re.findall(
                    r"[^\W_]+", str(subject or ""), flags=re.UNICODE
                )
                subject_pattern = (
                    r"(?<!\w)"
                    + r"[\W_]+".join(
                        re.escape(token) for token in subject_tokens
                    )
                    + r"[\s\"'\u2019\u201d]*$"
                    if subject_tokens
                    else r"(?!)"
                )
                subject_attached_scope = bool(re.search(
                    subject_pattern,
                    prefix,
                    re.IGNORECASE,
                ))
                if not (title_subject_scope or subject_attached_scope):
                    continue
                if (
                    _key(source_title) != _key(subject)
                    and _key(subject) not in _key(sentence)
                ):
                    continue
                parenthetical_key = _key(parenthetical)
                if object_key not in parenthetical_key:
                    continue
                endpoints = re.split(r"\s+[\u2013\u2014-]\s+", parenthetical, maxsplit=1)
                endpoint_supported = bool(
                    len(endpoints) == 2
                    and object_key in _key(
                        endpoints[1] if family == "death" else endpoints[0]
                    )
                )
                if not endpoint_supported:
                    years = re.findall(
                        r"(?<!\d)(?:1[0-9]{3}|20[0-9]{2})(?!\d)",
                        parenthetical,
                    )
                    object_years = re.findall(
                        r"(?<!\d)(?:1[0-9]{3}|20[0-9]{2})(?!\d)", obj
                    )
                    explicit_family = bool(re.search(
                        r"\b(?:born|birth)\b" if family == "birth"
                        else r"\b(?:died|death|passed\s+away)\b",
                        parenthetical,
                        re.IGNORECASE,
                    ))
                    endpoint_supported = bool(
                        explicit_family
                        or (
                            len(years) >= 2
                            and object_years
                            and object_years[-1]
                            == (years[0] if family == "birth" else years[-1])
                        )
                    )
                if not endpoint_supported:
                    continue
                provenance_scope = (
                    "title_described_entity"
                    if title_subject_scope
                    else "subject_attached_parenthetical"
                )
                reason_prefix = (
                    "title" if title_subject_scope else "subject_attached"
                )
                return self._result(
                    SUPPORTED,
                    (
                        f"{reason_prefix}_lifespan_{family}_support"
                        if len(endpoints) == 2
                        else f"{reason_prefix}_biographical_{family}_support"
                    ),
                    sentence_id,
                    sentence,
                    source_subject_surface=(
                        source_title
                        if title_subject_scope
                        else (_source_surface(prefix, subject) or subject)
                    ),
                    source_object_surface=_source_surface(sentence, obj) or obj,
                    provenance_scope=provenance_scope,
                )
        return None

    @staticmethod
    def _ordered_pattern(*parts: str, max_gap: int = 100) -> re.Pattern:
        escaped = [re.escape(_key(part)) for part in parts if _key(part)]
        return re.compile(
            (rf".{{0,{max_gap}}}").join(rf"\b{part}\b" for part in escaped),
            re.IGNORECASE,
        )

    def _semantic_source_pattern(
        self,
        *,
        fact: Mapping[str, Any],
        sentence: str,
        source_title: str,
        subject: str,
        obj: str,
        predicate_families: set,
    ) -> Tuple[bool, str]:
        sentence_key = _key(sentence)
        subject_key, object_key = _key(subject), _key(obj)
        canonical_relation = str(fact.get("canonical_relation") or "").casefold()
        if source_local_kinship_role_conflict(
            subject, sentence, canonical_relation
        ):
            return False, "kinship_role_descriptor_conflict"
        if "nationality" in predicate_families:
            pattern = re.compile(
                rf"\b{re.escape(subject_key)}\b.{{0,60}}\b(?:is|was|are|were)\b"
                rf".{{0,40}}\b{re.escape(object_key)}\b",
                re.IGNORECASE,
            )
            if pattern.search(sentence_key):
                return True, "nationality_copula_support"
        if canonical_relation in {"father_of", "mother_of", "parent_of"}:
            direct_role = {
                "father_of": "father",
                "mother_of": "mother",
                "parent_of": "parent",
            }[canonical_relation]
            direct = re.compile(
                rf"\b{re.escape(subject_key)}\b.{{0,80}}\b{direct_role}\b"
                rf".{{0,80}}\b{re.escape(object_key)}\b",
                re.IGNORECASE,
            )
            inverse = re.compile(
                rf"\b{re.escape(object_key)}\b.{{0,80}}\b(?:son|daughter|child)\s+of\b"
                rf".{{0,80}}\b{re.escape(subject_key)}\b",
                re.IGNORECASE,
            )
            if direct.search(sentence_key) or inverse.search(sentence_key):
                return True, "inverse_kinship_direction_support"
            title_scoped_inverse = re.compile(
                rf"\b(?:son|daughter|child)\s+of\b"
                rf".{{0,100}}\b{re.escape(subject_key)}\b",
                re.IGNORECASE,
            )
            subject_surface_pattern = re.escape(subject).replace(r"\ ", r"\s+")
            title_scoped_possessive = re.compile(
                rf"\b{subject_surface_pattern}\s*(?:'s|’s|['’])"
                rf"(?:\s+and\s+[^,.;]{{1,80}}?\s*(?:'s|’s|['’]))?"
                rf"\s+(?:(?:eldest|youngest|older|younger|only|first|second)\s+)?"
                rf"(?:son|daughter|child)\b",
                re.IGNORECASE,
            )
            if (
                self.enable_provenance_scope
                and self._title_describes_entity(source_title, obj)
                and self._title_reference_in_sentence(sentence, source_title)
                and (
                    title_scoped_inverse.search(sentence_key)
                    or title_scoped_possessive.search(sentence)
                )
            ):
                return True, "title_scoped_inverse_kinship_direction_support"
        physical_families = {
            str(value or "").strip().casefold()
            for value in fact.get("predicate_families") or []
        }
        physical_relation = str(
            fact.get("canonical_role_relation") or ""
        ).strip().casefold()
        if (
            canonical_relation in {"child_of", "son_of", "daughter_of"}
            or physical_relation in {"child_of", "son_of", "daughter_of"}
            or physical_families
            & {"child_of", "son_of", "daughter_of"}
        ):







            subject_surface = re.escape(subject).replace(r"\ ", r"\s+")
            object_surface = re.escape(obj).replace(r"\ ", r"\s+")
            possessive_parent = re.compile(
                rf"\b{subject_surface}\b"
                rf"(?P<context>[^.;]{{0,180}}?)"
                rf"\b(?P<possessive>his|her)\s+"
                rf"(?P<role>father|mother|parent)\s+"
                rf"(?:was\s+|is\s+)?\b{object_surface}\b",
                re.IGNORECASE,
            ).search(str(sentence or ""))
            if possessive_parent is not None:
                context = possessive_parent.group("context")
                intervening_name = re.search(
                    r"\b[A-Z][\w'\u2019-]+(?:\s+[A-Z][\w'\u2019-]+)+\b",
                    context,
                )
                possessive = possessive_parent.group("possessive").casefold()
                role_compatible = not (
                    (physical_relation == "son_of" or "son_of" in physical_families)
                    and possessive != "his"
                ) and not (
                    (
                        physical_relation == "daughter_of"
                        or "daughter_of" in physical_families
                    )
                    and possessive != "her"
                )
                if intervening_name is None and role_compatible:
                    return True, "possessive_child_parent_support"
        return False, ""

    def __call__(
        self, fact: Mapping[str, Any], source_text: str
    ) -> Dict[str, Any]:
        triple = list(fact.get("triple") or [])
        if len(triple) != 3 or not str(source_text or "").strip():
            return self._result(INSUFFICIENT, "missing_fact_or_source")
        subject, relation, obj = [str(value or "").strip() for value in triple]
        subject_key, object_key = _key(subject), _key(obj)
        canonical_relation = str(
            fact.get("canonical_relation")
            or fact.get("canonical_role_relation")
            or ""
        ).strip().casefold()
        if (
            canonical_relation in {"father_of", "mother_of", "parent_of"}
            and self._unresolved_kinship_description(subject)
        ):
            return self._result(
                INSUFFICIENT,
                "unresolved_kinship_description_not_entity",
            )
        source_title = str(fact.get("source_title") or "").strip()
        predicate_families = set(fact.get("predicate_families") or [])
        predicate_stop_tokens = {
            "a", "an", "and", "are", "as", "at", "be", "been", "being",
            "by", "for", "from", "in", "is", "of", "on", "the", "to",
            "was", "were", "with",
        }
        raw_relation_tokens = {
            token for token in _tokens(relation)
            if token not in predicate_stop_tokens
        }
        endpoint_tokens = set(_tokens(subject)) | set(_tokens(obj))





        predicate_evidence_tokens = raw_relation_tokens - endpoint_tokens
        if not predicate_evidence_tokens:
            predicate_evidence_tokens = set(raw_relation_tokens)
        relation_tokens = set(predicate_evidence_tokens)
        for family in predicate_families:
            relation_tokens.update(self._family_cues.get(str(family), set()))
        if not subject_key or not object_key or not relation_tokens:
            return self._result(INSUFFICIENT, "incomplete_fact_contract")

        lifespan_support = self._lifespan_support(
            fact=fact,
            source_text=source_text,
            source_title=source_title,
            subject=subject,
            obj=obj,
            predicate_families=predicate_families,
        )
        if lifespan_support is not None:
            return lifespan_support

        subject_relation_sentences: List[Tuple[int, str]] = []
        title_subject_scope = bool(
            self.enable_provenance_scope
            and self._title_describes_entity(source_title, subject)
        )
        for sentence_id, sentence in _sentences(source_text):
            sentence_key = _key(sentence)
            sentence_tokens = set(_tokens(sentence))
            subject_match = subject_key in sentence_key
            object_match = object_key in sentence_key
            semantic_match, semantic_reason = self._semantic_source_pattern(
                fact=fact,
                sentence=sentence,
                source_title=source_title,
                subject=subject,
                obj=obj,
                predicate_families=predicate_families,
            )
            relation_match = bool(relation_tokens & sentence_tokens) or semantic_match
            title_scoped_subject_match = bool(
                title_subject_scope
                and self._title_reference_in_sentence(sentence, source_title)
            )
            title_scoped_object_match = bool(
                semantic_reason == "title_scoped_inverse_kinship_direction_support"
            )
            effective_subject_match = subject_match or title_scoped_subject_match
            effective_object_match = object_match or title_scoped_object_match
            if subject_match and relation_match:
                subject_relation_sentences.append((sentence_id, sentence))
            if not (
                effective_subject_match
                and effective_object_match
                and relation_match
            ):
                continue
            if (
                bool(
                    predicate_families
                    & {"spouse", "spouse_of", "wife_of", "husband_of"}
                )
                and title_scoped_subject_match
                and not subject_match
                and self._title_pronoun_possessive_spouse_conflict(
                    sentence, obj
                )
            ):
                return self._result(
                    INSUFFICIENT,
                    "possessive_spouse_owner_conflicts_with_title_reference",
                    sentence_id,
                    sentence,
                )
            if self._lifespan_role_attachment_conflict(
                sentence=sentence,
                source_title=source_title,
                subject=subject,
                obj=obj,
                predicate_families=predicate_families,
            ):
                return self._result(
                    INSUFFICIENT,
                    "lifespan_value_attached_to_different_source_entity",
                    sentence_id,
                    sentence,
                )
            negated = self._relation_is_negated(
                sentence,
                subject,
                obj,
                relation_tokens,
            )
            if negated:
                return self._result(
                    CONTRADICTED,
                    "negated_source_relation",
                    sentence_id,
                    sentence,
                )
            direction_required = bool(fact.get("role_direction_required"))
            if (
                direction_required
                and not semantic_match
                and not (title_scoped_subject_match and object_match)
            ):
                lowered = sentence.lower()
                subject_at = lowered.find(subject.lower())
                relation_at = min(
                    (lowered.find(token) for token in relation_tokens
                     if lowered.find(token) >= 0),
                    default=-1,
                )
                object_at = lowered.find(obj.lower())
                if not (0 <= subject_at <= relation_at <= object_at):
                    return self._result(
                        INSUFFICIENT,
                        "direction_not_explicitly_supported",
                        sentence_id,
                        sentence,
                    )
            return self._result(
                SUPPORTED,
                semantic_reason or (
                    "title_scoped_relation_support"
                    if title_scoped_subject_match and not subject_match
                    else "explicit_same_sentence_support"
                ),
                sentence_id,
                sentence,
                source_subject_surface=(
                    self._expanded_kinship_parent_surface(fact, sentence, subject)
                    or _source_surface(sentence, subject)
                    or (source_title if title_scoped_subject_match else "")
                ),
                source_object_surface=(
                    _source_surface(sentence, obj)
                    or (source_title if title_scoped_object_match else "")
                ),
                provenance_scope=(
                    "title_described_object"
                    if title_scoped_object_match and not object_match
                    else (
                        "title_described_entity"
                        if title_scoped_subject_match and not subject_match
                        else "same_sentence"
                    )
                ),
                certified_assertion={
                    "triple": [subject, relation, obj],
                    "canonical_relation": str(
                        fact.get("canonical_relation") or ""
                    ),
                    "predicate_families": sorted(
                        str(value) for value in predicate_families
                    ),
                    "predicate_evidence_tokens": sorted(
                        predicate_evidence_tokens
                    ),
                    "source_sentence_id": sentence_id,
                    "source_sentence_span": sentence[:480],
                },
            )

        if subject_relation_sentences and predicate_families & self._exclusive_families:
            sentence_id, sentence = subject_relation_sentences[0]
            return self._result(
                CONTRADICTED,
                "exclusive_relation_mentions_different_or_missing_value",
                sentence_id,
                sentence,
            )
        return self._result(INSUFFICIENT, "no_explicit_same_sentence_support")

    def _result(
        self,
        status: str,
        reason: str,
        sentence_id: int = -1,
        sentence: str = "",
        **details: Any,
    ) -> Dict[str, Any]:
        return {
            "support_status": status,
            "support_reason": reason,
            "source_sentence_id": sentence_id,
            "source_sentence_span": sentence[:480],
            "verifier_version": self.version,
            "guarantee_scope": "explicit_local_source_support_only",
            **details,
        }

    @staticmethod
    def _lifespan_role_attachment_conflict(
        *,
        sentence: str,
        source_title: str,
        subject: str,
        obj: str,
        predicate_families: set[str],
    ) -> bool:
        """Reject a lifespan value grammatically attached to another entity.

        Co-occurrence is insufficient for biographical dates.  In a sentence
        such as ``Issac Booth (born 1971) ... played for Cleveland``, the date
        belongs to the title entity, not to Cleveland.  This guard is source-
        local and family-generic; it does not depend on query or entity names.
        """

        if not predicate_families & {"birth", "death"}:
            return False
        title = str(source_title or "").strip()
        subject_text = str(subject or "").strip()
        object_text = str(obj or "").strip()
        if not title or not subject_text or not object_text:
            return False
        if _key(title) == _key(subject_text):
            return False
        lowered = str(sentence or "").lower()
        subject_at = lowered.find(subject_text.lower())
        object_at = lowered.find(object_text.lower())
        if min(subject_at, object_at) < 0:
            return False
        cue_pattern = (
            r"\b(?:born|birth|birthday)\b"
            if "birth" in predicate_families
            else r"\b(?:died|death|deceased)\b"
        )
        cue = re.search(cue_pattern, sentence, re.IGNORECASE)
        if cue is None:
            return False
        cue_at = cue.start()
        title_at = lowered.find(title.lower())
        if title_at < 0:




            title_tokens = _tokens(title)
            prefix_tokens = _tokens(sentence[:cue_at])
            cursor = 0
            for token in prefix_tokens:
                if cursor < len(title_tokens) and token == title_tokens[cursor]:
                    cursor += 1
            if not title_tokens or cursor != len(title_tokens):
                return False
            title_at = 0



        return title_at <= cue_at <= object_at < subject_at


class SourceSupportObligationAdmissionVerifier(RuleBasedSourceSupportVerifier):
    """Two-gate verifier for text-derived tuples used by a query operator.

    The source-support gate asks whether the tuple is licensed by its exact
    provenance partition.  The obligation-admission gate separately checks
    whether that supported tuple is a legal output of the current physical
    operator.  Keeping the gates separate prevents a source-supported but
    directionally or type-incompatible tuple from entering a witness.
    """




    version = "source_support_obligation_admission_v18"
    contract_version = "obligation_admission_contract_v7"
    accepted_contract_versions = {
        "obligation_admission_contract_v1",
        "obligation_admission_contract_v2",
        "obligation_admission_contract_v3",
        "obligation_admission_contract_v4",
        "obligation_admission_contract_v5",
        "obligation_admission_contract_v6",
        "obligation_admission_contract_v7",
    }
    _work_terms = {
        "film", "movie", "documentary", "short", "feature", "comedy",
        "drama", "thriller", "western", "musical", "animation",
    }
    _family_aliases = {
        "spouse_of": "spouse",
        "wife_of": "spouse",
        "husband_of": "spouse",
        "father_of": "parent",
        "mother_of": "parent",
        "parent_of": "parent",
    }
    _parent_predicates = {"father_of", "mother_of", "parent_of"}

    def __init__(
        self,
        *,
        enable_enhanced_source_support: bool = True,
        enable_obligation_admission: bool = True,
        enable_predicate_entailment_admission: bool = False,
        enable_predicate_subtype_constraints: bool = True,
        enable_type_role_constraints: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.enable_enhanced_source_support = bool(
            enable_enhanced_source_support
        )
        self.enable_obligation_admission = bool(enable_obligation_admission)
        self.enable_predicate_entailment_admission = bool(
            enable_predicate_entailment_admission
        )
        self.enable_predicate_subtype_constraints = bool(
            enable_predicate_subtype_constraints
        )
        self.enable_type_role_constraints = bool(enable_type_role_constraints)
        source_mode = (
            "enhanced_source"
            if self.enable_enhanced_source_support
            else "legacy_source"
        )
        admission_mode = (
            "canonical_obligation"
            if self.enable_obligation_admission
            else "no_obligation"
        )
        entailment_mode = (
            "predicate_entailment"
            if self.enable_predicate_entailment_admission
            else "endpoint_cooccurrence"
        )
        base_version = (
            f"source_support_obligation_admission_v18:{source_mode}:"
            f"{admission_mode}:{entailment_mode}"
        )
        self.version = (
            base_version
            if (
                self.enable_predicate_subtype_constraints
                and self.enable_type_role_constraints
            )
            else (
                f"{base_version}:"
                f"subtype_{'on' if self.enable_predicate_subtype_constraints else 'off'}:"
                f"type_role_{'on' if self.enable_type_role_constraints else 'off'}"
            )
        )

    @staticmethod
    def _verification_sentences(text: str) -> List[Tuple[int, str]]:
        """Sentence split that preserves common biographical abbreviations."""
        marker = "<S2RAG_ABBREVIATION_DOT>"
        protected = re.sub(
            r"\b(St|Mt|Dr|Mr|Mrs|Ms|Prof|Sr|Jr|d|b)\."
            r"(?=\s*(?:[A-Z0-9]))",
            rf"\1{marker}",
            str(text or ""),
        )
        return [
            (index, sentence.replace(marker, "."))
            for index, sentence in _sentences(protected)
        ]

    @staticmethod
    def _without_era_tokens(value: Any) -> str:
        return " ".join(
            token for token in _tokens(value)
            if token not in {"ad", "ce", "bc", "bce"}
        )

    @classmethod
    def _value_in_text(cls, value: Any, text: Any) -> bool:
        value_key = _key(value)
        text_key = _key(text)
        if value_key and value_key in text_key:
            return True
        value_without_era = cls._without_era_tokens(value)
        text_without_era = cls._without_era_tokens(text)
        return bool(
            value_without_era
            and len(value_without_era) >= 4
            and value_without_era in text_without_era
        )

    @classmethod
    def _relation_is_negated(
        cls,
        sentence: str,
        subject: str,
        obj: str,
        relation_tokens: set,
    ) -> bool:

        scoped = _key(sentence)
        for endpoint in sorted(
            {_key(subject), _key(obj)}, key=len, reverse=True
        ):
            if endpoint:
                scoped = scoped.replace(endpoint, " ")
        tokens = _tokens(scoped)
        relation_positions = [
            index for index, token in enumerate(tokens)
            if token in relation_tokens
        ]
        if not relation_positions:
            return False
        for relation_at in relation_positions:
            prefix = tokens[max(0, relation_at - 4):relation_at]
            if "not" in prefix and "only" in prefix:
                continue
            if set(prefix) & {"not", "never", "neither", "no"}:
                return True
        return False

    @classmethod
    def _relation_tokens(cls, fact: Mapping[str, Any]) -> set:
        triple = list(fact.get("triple") or [])
        relation = str(triple[1] if len(triple) == 3 else "")
        families = {str(value) for value in fact.get("predicate_families") or []}
        expanded_families = families | {
            cls._family_aliases.get(value, value) for value in families
        }
        canonical = str(fact.get("canonical_relation") or "")
        if canonical:
            expanded_families.add(cls._family_aliases.get(canonical, canonical))
        relation_tokens = set(_tokens(relation))
        for family in expanded_families:
            relation_tokens.update(cls._family_cues.get(family, set()))
        if "spouse" in expanded_families:
            relation_tokens.update({
                "married", "marriage", "spouse", "wife", "husband", "queen",
            })
        return relation_tokens

    @staticmethod
    def _surface_pattern(value: Any) -> str:
        tokens = _tokens(value)
        if not tokens:
            return ""
        return r"\b" + r"[\s\W]+".join(
            re.escape(token) for token in tokens
        ) + r"\b"

    @classmethod
    def _predicate_entailment_admission(
        cls,
        fact: Mapping[str, Any],
        source_result: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Check that a recovered tuple's predicate is expressed by its span.

        Endpoint co-occurrence is not relation evidence.  In particular, a
        sentence saying that the title entity is the wife or consort of X
        must not license an extracted ``X father_of title`` tuple.  This gate
        is deliberately scoped to query-local recovered tuples: indexed facts
        retain their existing source-verification path, while repaired facts
        must carry a locally entailed role and direction before admission.
        """
        is_recovered = bool(
            fact.get("bound_completion_fact")
            or fact.get("predicate_gap_sentence_repair")
            or str(fact.get("origin") or "") == "query_local_overlay"
        )
        if not is_recovered:
            return {
                "status": "NOT_APPLICABLE",
                "reason": "indexed_tuple_uses_existing_verification_contract",
                "target_predicate": "",
                "evidence_predicate": "",
            }

        contract = dict(fact.get("obligation_contract") or {})
        target = str(
            fact.get("canonical_relation")
            or contract.get("canonical_relation")
            or ""
        ).strip().casefold()
        if not target:
            target = next((
                str(value).strip().casefold()
                for value in contract.get("predicate_families") or []
                if str(value).strip().casefold() in cls._parent_predicates
            ), "")
        if target not in cls._parent_predicates:
            return {
                "status": "NOT_APPLICABLE",
                "reason": "predicate_family_not_yet_role_sensitive",
                "target_predicate": target,
                "evidence_predicate": "",
            }

        triple = list(fact.get("triple") or [])
        if len(triple) != 3:
            return {
                "status": "REJECTED",
                "reason": "invalid_tuple_shape",
                "target_predicate": target,
                "evidence_predicate": "",
            }
        subject, _, obj = [str(value or "").strip() for value in triple]
        sentence = str(
            source_result.get("source_sentence_span")
            or fact.get("source_sentence_span")
            or fact.get("completion_sentence")
            or ""
        ).strip()
        subject_pattern = cls._surface_pattern(subject)
        object_pattern = cls._surface_pattern(obj)
        subject_core_pattern = cls._surface_pattern(
            re.sub(r"\([^)]*\)", " ", subject)
        )
        object_core_pattern = cls._surface_pattern(
            re.sub(r"\([^)]*\)", " ", obj)
        )
        subject_endpoint_pattern = (
            rf"(?:{subject_pattern}|{subject_core_pattern})"
            if subject_core_pattern and subject_core_pattern != subject_pattern
            else subject_pattern
        )
        object_endpoint_pattern = (
            rf"(?:{object_pattern}|{object_core_pattern})"
            if object_core_pattern and object_core_pattern != object_pattern
            else object_pattern
        )
        if not sentence or not subject_pattern or not object_pattern:
            return {
                "status": "REJECTED",
                "reason": "missing_local_predicate_evidence",
                "target_predicate": target,
                "evidence_predicate": "",
            }



        match_sentence = _key(sentence)





        spouse_between_endpoints = bool(
            re.search(
                rf"(?:{object_endpoint_pattern}.{{0,160}}\b"
                rf"(?:wife|husband|spouse|consort)\s+of\b"
                rf".{{0,96}}{subject_endpoint_pattern}|"
                rf"{subject_endpoint_pattern}.{{0,160}}\b"
                rf"(?:wife|husband|spouse|consort)\s+of\b"
                rf".{{0,96}}{object_endpoint_pattern})",
                match_sentence,
                re.IGNORECASE,
            )
        )
        if spouse_between_endpoints:
            return {
                "status": "REJECTED",
                "reason": "incompatible_spouse_relation",
                "target_predicate": target,
                "evidence_predicate": "spouse_of",
            }

        direct_role = {
            "father_of": "father",
            "mother_of": "mother",
            "parent_of": "(?:father|mother|parent)",
        }[target]
        direct = re.compile(
            rf"{subject_endpoint_pattern}.{{0,64}}\b{direct_role}\b\s+of\s+"
            rf".{{0,32}}{object_endpoint_pattern}",
            re.IGNORECASE,
        )
        if direct.search(match_sentence):
            return {
                "status": "ADMITTED",
                "reason": "direct_role_predicate_entailed",
                "target_predicate": target,
                "evidence_predicate": target,
                "matched_pattern": "subject_role_of_object",
            }




        child_prefix = re.compile(
            rf"(?:{object_endpoint_pattern}.{{0,120}})?"
            rf"\b(?:son|daughter|child)\s+of\s+"
            rf"(?P<parents>[^.;]{{1,220}})",
            re.IGNORECASE,
        )
        source_title = str(fact.get("source_title") or "")
        title_scoped_object = bool(
            source_title and cls._title_describes_entity(source_title, obj)
        )
        inverse_match = child_prefix.search(match_sentence)
        if inverse_match and (
            re.search(object_endpoint_pattern, match_sentence, re.IGNORECASE)
            or title_scoped_object
        ):
            relation_prefix = match_sentence[
                max(0, inverse_match.start() - 160):inverse_match.start()
            ]
            nested_kinship_subject = bool(re.search(
                r"\b(?:his|her|their)\s+"
                r"(?:mother|father|parent|wife|husband|son|daughter)\b"
                r".{0,96}\b(?:is|was|are|were)\b",
                relation_prefix,
                re.IGNORECASE,
            ))
            if (
                title_scoped_object
                and not re.search(
                    object_endpoint_pattern, match_sentence, re.IGNORECASE
                )
                and nested_kinship_subject
            ):
                return {
                    "status": "REJECTED",
                    "reason": "nested_kinship_subject_mismatch",
                    "target_predicate": target,
                    "evidence_predicate": "ancestor_of",
                    "matched_pattern": "nested_child_of_parent_list",
                }
            parent_clause = inverse_match.group("parents")
            parent_clause = re.split(
                r"\b(?:and\s+)?(?:the\s+)?(?:wife|husband|consort)\s+of\b|"
                r",\s*(?:who|which|but|while)\b",
                parent_clause,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            subject_match = re.search(
                subject_pattern, parent_clause, re.IGNORECASE
            )
            if subject_match:
                prefix = parent_clause[max(0, subject_match.start() - 48):
                                       subject_match.start()]
                suffix = parent_clause[
                    subject_match.end():subject_match.end() + 72
                ]
                subject_tokens = set(_tokens(subject))
                local_suffix = re.split(
                    r"\b(?:and|but|by)\b",
                    suffix,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0]
                local_role_tokens = subject_tokens | set(_tokens(local_suffix))
                maternal = bool(re.search(
                    r"\b(?:mother|his\s+wife|her\s+wife|wife)\b",
                    prefix,
                    re.IGNORECASE,
                )) or bool(local_role_tokens & {
                    "archduchess", "infanta", "princess", "duchess",
                    "queen", "countess", "lady", "baroness", "empress",
                }) or bool(re.search(
                    r"\b(?:and\s+)?her\s+(?:third\s+|second\s+|first\s+)?"
                    r"husband\b",
                    suffix,
                    re.IGNORECASE,
                ))
                paternal = bool(re.search(
                    r"\b(?:father|her\s+husband|his\s+husband|husband)\b",
                    prefix,
                    re.IGNORECASE,
                )) or bool(local_role_tokens & {
                    "archduke", "hetman", "peshwa", "prince", "duke",
                    "king", "lord", "baron", "emperor", "sir",
                }) or bool(re.search(
                    r"\b(?:and\s+)?his\s+(?:third\s+|second\s+|first\s+)?"
                    r"wife\b|\bby\s+his\s+wife\b",
                    suffix,
                    re.IGNORECASE,
                ))
                if target in {"father_of", "mother_of"} and (
                    maternal == paternal
                ):
                    return {
                        "status": "DEFERRED",
                        "reason": "source_parent_role_underspecified",
                        "target_predicate": target,
                        "evidence_predicate": "parent_of",
                        "matched_pattern": "child_of_parent_list",
                    }
                role_compatible = (
                    target == "parent_of"
                    or (target == "mother_of" and maternal and not paternal)
                    or (target == "father_of" and paternal and not maternal)
                )
                if role_compatible:
                    evidence_predicate = (
                        "mother_of" if maternal
                        else "father_of" if paternal or target == "father_of"
                        else "parent_of"
                    )
                    return {
                        "status": "ADMITTED",
                        "reason": "inverse_child_parent_predicate_entailed",
                        "target_predicate": target,
                        "evidence_predicate": evidence_predicate,
                        "matched_pattern": "child_of_parent_list",
                    }
                return {
                    "status": "REJECTED",
                    "reason": "source_parent_role_mismatch",
                    "target_predicate": target,
                    "evidence_predicate": (
                        "mother_of" if maternal else "father_of"
                    ),
                }




        possessive_inverse = re.compile(
            rf"{object_endpoint_pattern}.{{0,120}}?"
            rf"{subject_endpoint_pattern}\s*[\'’](?:s)?"
            rf".{{0,120}}?\b(?:son|daughter|child)\b",
            re.IGNORECASE,
        ).search(sentence)
        if possessive_inverse:
            return {
                "status": (
                    "ADMITTED" if target == "parent_of" else "DEFERRED"
                ),
                "reason": (
                    "possessive_parent_predicate_entailed"
                    if target == "parent_of"
                    else "source_parent_role_underspecified"
                ),
                "target_predicate": target,
                "evidence_predicate": "parent_of",
                "matched_pattern": "possessive_parent_child",
            }

        return {
            "status": "DEFERRED",
            "reason": "source_predicate_not_explicitly_entailed",
            "target_predicate": target,
            "evidence_predicate": "",
        }

    def _typed_source_support(
        self,
        fact: Mapping[str, Any],
        source_text: str,
    ) -> Optional[Dict[str, Any]]:
        triple = list(fact.get("triple") or [])
        if len(triple) != 3:
            return None
        subject, _, obj = [str(value or "").strip() for value in triple]
        source_title = str(fact.get("source_title") or "").strip()
        families = {str(value) for value in fact.get("predicate_families") or []}
        canonical = str(fact.get("canonical_relation") or "").strip()
        expanded_families = families | {
            self._family_aliases.get(value, value) for value in families
        }
        if canonical:
            expanded_families.add(self._family_aliases.get(canonical, canonical))
        relation_tokens = self._relation_tokens(fact)
        title_subject_scope = self._title_describes_entity(source_title, subject)
        sentences = self._verification_sentences(source_text)

        for sentence_id, sentence in sentences:
            sentence_key = _key(sentence)
            sentence_tokens = set(_tokens(sentence))
            subject_match = _key(subject) in sentence_key
            object_match = self._value_in_text(obj, sentence)
            title_reference = self._title_reference_in_sentence(
                sentence, source_title
            ) or bool(re.search(
                r"\b(?:the|this)\s+(?:movie|motion\s+picture)\b",
                sentence,
                re.IGNORECASE,
            ))
            effective_subject = subject_match or (
                title_subject_scope and title_reference
            )
            relation_match = bool(relation_tokens & sentence_tokens)
            if self._lifespan_role_attachment_conflict(
                sentence=sentence,
                source_title=source_title,
                subject=subject,
                obj=obj,
                predicate_families=families,
            ):
                continue
            if (
                "spouse" in expanded_families
                and title_subject_scope
                and not subject_match
                and self._title_pronoun_possessive_spouse_conflict(
                    sentence, obj
                )
            ):
                continue
            if effective_subject and object_match and relation_match:
                if self._relation_is_negated(
                    sentence, subject, obj, relation_tokens
                ):
                    return None
                return self._result(
                    SUPPORTED,
                    "scoped_explicit_source_support",
                    sentence_id,
                    sentence,
                    source_subject_surface=(
                        _source_surface(sentence, subject)
                        or (source_title if title_subject_scope else subject)
                    ),
                    source_object_surface=_source_surface(sentence, obj) or obj,
                    provenance_scope=(
                        "title_described_entity"
                        if title_subject_scope and not subject_match
                        else "same_sentence"
                    ),
                )





            contract = dict(fact.get("obligation_contract") or {})
            if (
                "distribution" in expanded_families
                and str(contract.get("output_type") or "") == "date"
                and title_subject_scope
                and effective_subject
                and object_match
                and re.fullmatch(r"(?:1[0-9]{3}|20[0-9]{2})", _key(obj))
                and bool(sentence_tokens & self._work_terms)
                and bool(re.search(
                    rf"\b(?:is|was)\b.{{0,24}}\b{re.escape(_key(obj))}\b"
                    rf".{{0,80}}\b(?:film|movie|documentary|short|feature)\b",
                    sentence_key,
                    re.IGNORECASE,
                ))
            ):
                return self._result(
                    SUPPORTED,
                    "typed_title_year_source_support",
                    sentence_id,
                    sentence,
                    source_subject_surface=source_title,
                    source_object_surface=_source_surface(sentence, obj) or obj,
                    provenance_scope="typed_title_partition",
                )

            if (
                "spouse" in expanded_families
                and title_subject_scope
                and effective_subject
                and object_match
                and re.search(
                    r"\b(?:married|marriage\s+with|wife\s+of|husband\s+of|"
                    r"queen\s+of|consort\s+of)\b",
                    sentence,
                    re.IGNORECASE,
                )
                and not self._relation_is_negated(
                    sentence, subject, obj, relation_tokens
                )
                and not self._title_pronoun_possessive_spouse_conflict(
                    sentence, obj
                )
            ):
                return self._result(
                    SUPPORTED,
                    "title_scoped_spouse_source_support",
                    sentence_id,
                    sentence,
                    source_subject_surface=source_title,
                    source_object_surface=_source_surface(sentence, obj) or obj,
                    provenance_scope="title_described_entity",
                )

        contract = dict(fact.get("obligation_contract") or {})
        output_type = str(contract.get("output_type") or "")
        lifespan_family = (
            "death" if "death" in expanded_families
            else "birth" if "birth" in expanded_families
            else ""
        )
        if lifespan_family and title_subject_scope:
            for sentence_id, sentence in sentences:
                if not self._value_in_text(obj, sentence):
                    continue
                parentheticals = re.findall(r"\(([^)]{3,180})\)", sentence)
                for parenthetical in parentheticals:
                    if not self._value_in_text(obj, parenthetical):
                        continue
                    halves = re.split(r"\s*[\u2013\u2014-]\s*", parenthetical, maxsplit=1)
                    selected = (
                        halves[-1] if lifespan_family == "death" else halves[0]
                    )
                    explicit = bool(re.search(
                        r"\b(?:died|death|d)\."
                        if lifespan_family == "death"
                        else r"\b(?:born|birth|b)\.",
                        parenthetical,
                        re.IGNORECASE,
                    ))
                    endpoint_match = self._value_in_text(obj, selected)
                    if not (explicit or endpoint_match):
                        continue
                    if output_type == "place" and not endpoint_match:
                        continue
                    return self._result(
                        SUPPORTED,
                        f"typed_title_lifespan_{lifespan_family}_support",
                        sentence_id,
                        sentence,
                        source_subject_surface=source_title,
                        source_object_surface=_source_surface(sentence, obj) or obj,
                        provenance_scope="typed_title_lifespan",
                    )
        return None

    @staticmethod
    def _obligation_admission(
        fact: Mapping[str, Any],
        *,
        enforce_predicate_subtype: bool = True,
        enforce_type_role: bool = True,
    ) -> Dict[str, Any]:
        contract = dict(fact.get("obligation_contract") or {})
        if contract.get("version") not in (
            SourceSupportObligationAdmissionVerifier.accepted_contract_versions
        ):
            return {"status": "REJECTED", "reason": "missing_operator_contract"}
        operator_id = str(fact.get("operator_id") or "")
        if not operator_id or operator_id != str(contract.get("operator_id") or ""):
            return {"status": "REJECTED", "reason": "operator_id_mismatch"}
        triple = list(fact.get("triple") or [])
        if len(triple) != 3:
            return {"status": "REJECTED", "reason": "invalid_tuple_shape"}

        from .relation_fact_join_utils import (
            canonical_entity_key,
            canonical_operator_predicate_contract,
            canonical_predicate_contract,
            terminal_value_contract_compatibility,
        )

        expected_families = {
            str(value) for value in contract.get("predicate_families") or []
        }
        predicate_contract = canonical_predicate_contract(
            triple[1],
            triple[2],
            fact.get("source_sentence_span") or fact.get("source_text") or "",
        )
        operator_predicate_contract = canonical_operator_predicate_contract(
            contract.get("relation_tokens") or [], expected_families
        )
        actual_families = set(
            predicate_contract["canonical_families"]
        )
        predicate_details = {
            "raw_predicate": predicate_contract["raw_relation"],
            "canonical_predicate_families": list(
                predicate_contract["canonical_families"]
            ),
            "canonical_predicate_subtypes": list(
                predicate_contract["semantic_subtypes"]
            ),
            "allowed_predicate_subtypes": list(
                operator_predicate_contract["allowed_semantic_subtypes"]
            ),
            "predicate_contract_version": predicate_contract["version"],
        }
        role_compatibility = role_aware_predicate_contract_compatibility(
            {
                **dict(fact),
                "predicate_families": sorted(actual_families),
                "predicate_subtypes": sorted(
                    set(fact.get("predicate_subtypes") or [])
                    | set(predicate_contract["semantic_subtypes"])
                ),
            },
            contract,
            input_side=str(contract.get("bound_input_side") or ""),
            output_side=str(fact.get("bound_output_role") or ""),
        )
        predicate_details.update({
            "predicate_compatibility_reason": str(
                role_compatibility.get("reason") or ""
            ),
            "predicate_compatibility_orientations": list(
                role_compatibility.get("orientations") or []
            ),
        })
        if (
            expected_families
            and not expected_families & actual_families
            and not role_compatibility.get("compatible")
        ):
            return {
                "status": "REJECTED",
                "reason": "predicate_family_mismatch",
                **predicate_details,
            }
        required_subtypes = {
            str(value) for value in contract.get("required_predicate_subtypes") or []
        }
        actual_subtypes = {
            str(value) for value in fact.get("predicate_subtypes") or []
        } | set(predicate_contract["semantic_subtypes"])
        if (
            enforce_predicate_subtype
            and
            contract.get("strict_predicate_subtype")
            and required_subtypes
            and not required_subtypes & actual_subtypes
        ):
            return {
                "status": "REJECTED",
                "reason": "predicate_subtype_mismatch",
                **predicate_details,
            }

        semantic_subtypes = {
            value for value in predicate_contract["semantic_subtypes"]
            if str(value).startswith("distribution_")
        }
        allowed_semantic_subtypes = set(
            operator_predicate_contract["allowed_semantic_subtypes"]
        )
        if enforce_predicate_subtype and expected_families & {"distribution"}:
            if semantic_subtypes & {
                "distribution_conflicting_release",
                "distribution_intended_release",
            } and not semantic_subtypes & allowed_semantic_subtypes:
                return {
                    "status": "REJECTED",
                    "reason": "predicate_semantic_subtype_mismatch",
                    **predicate_details,
                }
            if (
                allowed_semantic_subtypes
                and semantic_subtypes
                and not semantic_subtypes & allowed_semantic_subtypes
            ):
                return {
                    "status": "REJECTED",
                    "reason": "predicate_semantic_subtype_mismatch",
                    **predicate_details,
                }

        output_side = str(fact.get("bound_output_role") or "")
        input_side = str(contract.get("bound_input_side") or "")
        if output_side not in {"subject", "object"}:
            return {"status": "REJECTED", "reason": "missing_output_role"}
        if input_side not in {"subject", "object"} or input_side == output_side:
            return {"status": "REJECTED", "reason": "direction_mismatch"}
        output_index = 0 if output_side == "subject" else 2
        input_index = 0 if input_side == "subject" else 2
        output_key = canonical_entity_key(triple[output_index])
        input_key = canonical_entity_key(triple[input_index])
        if output_key != str(fact.get("bound_output_key") or ""):
            return {"status": "REJECTED", "reason": "output_binding_mismatch"}
        if output_key != str(contract.get("bound_output_key") or ""):
            return {"status": "REJECTED", "reason": "contract_output_mismatch"}
        accepted_input_keys = {
            str(value) for value in contract.get("accepted_input_keys") or []
        }
        if accepted_input_keys and input_key not in accepted_input_keys:
            return {"status": "REJECTED", "reason": "input_binding_mismatch"}
        if enforce_type_role and not bool(contract.get("role_direction_validated")):
            return {"status": "REJECTED", "reason": "unvalidated_direction"}

        binding_contract = (
            _binding_to_obligation_contract(fact)
            if enforce_type_role
            else {"status": "OFF", "reason": "type_role_constraints_disabled"}
        )
        binding_contract_details = {
            "binding_contract_status": binding_contract["status"],
            "binding_contract_reason": binding_contract["reason"],
            "required_input_qualifier_tokens": binding_contract.get(
                "required_input_qualifier_tokens", []
            ),
        }
        if binding_contract["status"] in {"REJECTED", "DEFERRED"}:
            return {
                "status": binding_contract["status"],
                "reason": binding_contract["reason"],
                **binding_contract_details,
                **predicate_details,
            }

        target_entity_categories = (
            contract.get("target_entity_categories") or []
            if "target_entity_categories" in contract
            else contract.get("relation_tokens") or []
        )
        terminal_type = terminal_value_contract_compatibility(
            triple[output_index],
            str(contract.get("output_type") or ""),
            fact.get("relation_tokens")
            or contract.get("physical_relation_tokens")
            or _tokens(triple[1]),
            target_category_tokens=target_entity_categories,
            output_role=output_side,
            source_text=fact.get("source_sentence_span")
            or fact.get("source_text")
            or "",
            relation_families=sorted(
                set(fact.get("relation_families") or [])
                | set(fact.get("predicate_families") or [])
                | set(contract.get("relation_families") or [])
                | set(contract.get("predicate_families") or [])
            ),
        )
        terminal_type_details = {
            "terminal_type_status": terminal_type["status"],
            "terminal_type_reason": terminal_type["reason"],
            "terminal_value_type": terminal_type.get("value_type", ""),
            "terminal_value_categories": terminal_type.get(
                "value_categories", []
            ),
            "terminal_target_categories": terminal_type.get(
                "target_categories", []
            ),
            "terminal_relation_families": terminal_type.get(
                "relation_families", []
            ),
        }
        if enforce_type_role and terminal_type["status"] == "INCOMPATIBLE":
            reason = (
                "output_category_mismatch"
                if "category" in str(terminal_type.get("reason") or "")
                else "output_type_mismatch"
            )
            return {
                "status": "REJECTED",
                "reason": reason,
                **terminal_type_details,
                **predicate_details,
            }
        if enforce_type_role and terminal_type["status"] == "UNKNOWN":
            return {
                "status": "DEFERRED",
                "reason": "output_type_unknown",
                **terminal_type_details,
                **predicate_details,
            }
        return {
            "status": "ADMITTED",
            "reason": "operator_contract_satisfied",
            "operator_id": operator_id,
            "output_role": output_side,
            "output_type": str(contract.get("output_type") or ""),
            **binding_contract_details,
            **terminal_type_details,
            **predicate_details,
        }

    def __call__(
        self, fact: Mapping[str, Any], source_text: str
    ) -> Dict[str, Any]:





        prevalidated_source = dict(
            fact.get("_prevalidated_source_support") or {}
        )
        source_result = (
            prevalidated_source
            if prevalidated_source.get("support_status")
            in {SUPPORTED, CONTRADICTED, INSUFFICIENT, UNKNOWN}
            else dict(super().__call__(fact, source_text) or {})
        )
        if (
            self.enable_enhanced_source_support
            and source_result.get("support_status") != SUPPORTED
            and source_result.get("support_reason")
            != "unresolved_kinship_description_not_entity"
        ):
            recovered = self._typed_source_support(fact, source_text)
            if recovered is not None:
                source_result = recovered
        output_role = str(fact.get("bound_output_role") or "")
        source_output_surface = str(
            source_result.get(f"source_{output_role}_surface") or ""
        ).strip()
        source_result["source_identity_qualifier_tokens"] = sorted(
            _local_identity_qualifiers(
                source_result.get("source_sentence_span"),
                source_output_surface,
            )
        )
        source_status = str(source_result.get("support_status") or INSUFFICIENT)
        source_reason = str(source_result.get("support_reason") or "unspecified")
        if source_status == SUPPORTED:
            triple = [str(value or "") for value in fact.get("triple") or []]
            if len(triple) == 3:
                canonical_relation = str(
                    fact.get("canonical_relation")
                    or fact.get("canonical_role_relation")
                    or triple[1]
                )
                certificate = dict(
                    source_result.get("certified_assertion") or {}
                )
                certificate.update({
                    "triple": certificate.get("triple") or triple,
                    "raw_triple": certificate.get("raw_triple") or triple,
                    "canonical_triple": certificate.get("canonical_triple")
                    or [triple[0], canonical_relation, triple[2]],
                    "canonical_relation": certificate.get(
                        "canonical_relation"
                    ) or canonical_relation,
                    "predicate_families": certificate.get(
                        "predicate_families"
                    ) or sorted(
                        str(value)
                        for value in fact.get("predicate_families") or []
                    ),
                    "source_id": certificate.get(
                        "source_id", fact.get("source_id", fact.get("doc_idx"))
                    ),
                    "source_title": certificate.get("source_title")
                    or str(fact.get("source_title") or ""),
                    "source_sentence_id": certificate.get(
                        "source_sentence_id",
                        source_result.get("source_sentence_id", -1),
                    ),
                    "source_sentence_span": certificate.get(
                        "source_sentence_span"
                    ) or str(source_result.get("source_sentence_span") or "")[:480],
                })
                source_result["certified_assertion"] = certificate
        admission_fact = dict(fact)
        admission_fact["source_sentence_span"] = str(
            source_result.get("source_sentence_span")
            or fact.get("source_sentence_span")
            or ""
        )
        admission = (
            self._obligation_admission(
                admission_fact,
                enforce_predicate_subtype=(
                    self.enable_predicate_subtype_constraints
                ),
                enforce_type_role=self.enable_type_role_constraints,
            )
            if self.enable_obligation_admission
            else {
                "status": "OFF",
                "reason": "obligation_admission_disabled",
            }
        )
        predicate_entailment = (
            self._predicate_entailment_admission(
                admission_fact, source_result
            )
            if self.enable_predicate_entailment_admission
            else {
                "status": "OFF",
                "reason": "predicate_entailment_admission_disabled",
                "target_predicate": "",
                "evidence_predicate": "",
            }
        )
        source_result.update({
            "verifier_version": self.version,
            "admission_policy": self.version,
            "source_support_gate_status": source_status,
            "source_support_gate_reason": source_reason,
            "obligation_admission_status": admission["status"],
            "obligation_admission_reason": admission["reason"],
            "predicate_entailment_admission_status": predicate_entailment[
                "status"
            ],
            "predicate_entailment_admission_reason": predicate_entailment[
                "reason"
            ],
            "predicate_entailment_target": predicate_entailment.get(
                "target_predicate", ""
            ),
            "predicate_entailment_evidence": predicate_entailment.get(
                "evidence_predicate", ""
            ),
            "predicate_entailment_pattern": predicate_entailment.get(
                "matched_pattern", ""
            ),
            "obligation_contract_version": self.contract_version,
            "obligation_operator_id": admission.get("operator_id", ""),
            "obligation_output_role": admission.get("output_role", ""),
            "obligation_output_type": admission.get("output_type", ""),
            "obligation_raw_predicate": admission.get("raw_predicate", ""),
            "obligation_canonical_predicate_families": admission.get(
                "canonical_predicate_families", []
            ),
            "obligation_canonical_predicate_subtypes": admission.get(
                "canonical_predicate_subtypes", []
            ),
            "obligation_allowed_predicate_subtypes": admission.get(
                "allowed_predicate_subtypes", []
            ),
            "obligation_predicate_compatibility_reason": admission.get(
                "predicate_compatibility_reason", ""
            ),
            "obligation_predicate_compatibility_orientations": admission.get(
                "predicate_compatibility_orientations", []
            ),
            "predicate_contract_version": admission.get(
                "predicate_contract_version", ""
            ),
            "obligation_terminal_type_status": admission.get(
                "terminal_type_status", ""
            ),
            "obligation_terminal_type_reason": admission.get(
                "terminal_type_reason", ""
            ),
            "obligation_terminal_value_type": admission.get(
                "terminal_value_type", ""
            ),
            "obligation_terminal_value_categories": admission.get(
                "terminal_value_categories", []
            ),
            "obligation_terminal_target_categories": admission.get(
                "terminal_target_categories", []
            ),
            "binding_obligation_contract_status": admission.get(
                "binding_contract_status", ""
            ),
            "binding_obligation_contract_reason": admission.get(
                "binding_contract_reason", ""
            ),
            "binding_required_input_qualifier_tokens": admission.get(
                "required_input_qualifier_tokens", []
            ),
        })
        if source_status not in {SUPPORTED, "OFF"}:
            return source_result
        if predicate_entailment["status"] == "REJECTED":
            source_result.update({
                "support_status": INSUFFICIENT,
                "support_reason": (
                    "predicate_entailment:"
                    f"{predicate_entailment['reason']}"
                ),
                "guarantee_scope": (
                    "source_supported_but_predicate_not_entailed"
                ),
            })
            return source_result




        if not self.enable_obligation_admission:
            source_result.update({
                "support_status": SUPPORTED,
                "support_reason": f"source_only:{source_reason}",
                "guarantee_scope": (
                    "enhanced_source_support_only"
                    if self.enable_enhanced_source_support
                    else "legacy_source_support_only"
                ),
            })
            return source_result
        if admission["status"] == "DEFERRED":
            source_result.update({
                "support_status": UNKNOWN,
                "support_reason": f"obligation_admission:{admission['reason']}",
                "guarantee_scope": "source_supported_obligation_type_unknown",
            })
            return source_result
        if admission["status"] != "ADMITTED":
            source_result.update({
                "support_status": INSUFFICIENT,
                "support_reason": f"obligation_admission:{admission['reason']}",
                "guarantee_scope": "source_supported_but_obligation_rejected",
            })
            return source_result
        source_result.update({
            "support_status": SUPPORTED,
            "support_reason": f"source+obligation:{source_reason}",
            "guarantee_scope": "source_support+typed_obligation_admission",
        })
        return source_result


class CallbackEvidenceBackend:
    """Small adapter used by both production wiring and deterministic tests."""

    def __init__(
        self,
        *,
        initial_bindings: Sequence[Mapping[str, Any]],
        lookup: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
        complete: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
        verify: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        estimate: Optional[
            Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
        ] = None,
        verifier_version: str = "injected_verifier_v1",
    ) -> None:
        self._initial_bindings = [dict(value) for value in initial_bindings]
        self._lookup = lookup
        self._complete = complete
        self._verify = verify
        self._estimate = estimate
        self.verifier_version = verifier_version

    def initial_bindings(self, question, logical_plan):
        return [dict(value) for value in self._initial_bindings]

    def lookup_join(self, binding, operator):
        return dict(self._lookup(binding, operator) or {})

    def complete(self, binding, operator):
        return dict(self._complete(binding, operator) or {})

    def verify(self, fact):
        return dict(self._verify(fact) or {})

    def estimate(self, action, state):
        if self._estimate is not None:
            return dict(self._estimate(action, state) or {})
        operator = str(action.get("operator") or "")
        defaults = {
            LOOKUP_JOIN: {
                "expected_progress": 0.65,
                "expected_scan_rows": 20,
                "expected_retrieval_calls": 0,
                "expected_model_calls": 0,
                "expected_tokens": 0,
                "support_probability": 0.65,
            },
            COMPLETE: {
                "expected_progress": 0.72,
                "expected_scan_rows": 0,
                "expected_retrieval_calls": 1,
                "expected_model_calls": 0,
                "expected_tokens": None,
                "support_probability": 0.60,
            },
            VERIFY: {
                "expected_progress": 0.40,
                "expected_scan_rows": 0,
                "expected_retrieval_calls": 0,
                "expected_model_calls": 0,
                "expected_tokens": 0,
                "support_probability": 0.70,
            },
        }
        return dict(defaults.get(operator, {}))


class RelationIndexEvidenceBackend:
    """Adapter from the existing relation index to the paper executor.

    The adapter intentionally calls one edge at a time. LOOKUP_JOIN never
    invokes completion or verification, and COMPLETE writes only query-local
    fact rows returned to the executor.
    """

    def __init__(
        self,
        *,
        relation_index: Mapping[str, Any],
        get_source_content: Callable[[int], str],
        completion_lookup: Callable[[str, int], Sequence[Mapping[str, Any]]],
        obligation_source_lookup: Optional[
            Callable[
                [str, Mapping[str, Any], int],
                Sequence[Mapping[str, Any]],
            ]
        ] = None,
        relation_materializer: Optional[Callable[..., Mapping[str, Any]]] = None,
        get_source_title: Optional[Callable[[int], str]] = None,
        current_source_candidates: Optional[
            Sequence[Mapping[str, Any]]
        ] = None,
        completion_top_k: int = 6,
        completion_mode: str = "lazy",
        completion_query_mode: str = "obligation_bound",
        verification_mode: str = "strict",
        repair_scheduler_mode: str = "eager",
        execution_mode: str = "saqe",
        index_delete_rate: float = 0.0,
        index_corrupt_rate: float = 0.0,
        index_corrupt_mode: str = "object",
        perturbation_seed: int = 0,
        max_facts_per_key: int = 60,
        max_key_fanout: int = 600,
        enable_title_catalog_completion: bool = True,
        enable_provenance_source_verifier: bool = True,
        enable_identity_safe_composite_join: bool = True,
        relation_matching_mode: str = "canonical",
        enable_predicate_subtype_constraints: bool = True,
        enable_type_role_constraints: bool = True,
        enable_composite_index_access: bool = True,
        enable_schema_plan_coverage_v7: bool = False,
        enable_lineage_terminal_semijoin_v7_1: bool = False,
        enable_provenance_grouped_terminal_v7_2: bool = False,
        enable_schema_exact_nested_join_v9: bool = False,
        enable_collision_aware_schema_identity: bool = False,
        enable_safe_title_family_root_binding: bool = True,
        enable_source_grounded_terminal_tuple_recovery: bool = False,
        enable_obligation_bound_repair_certificate: bool = False,
        enable_terminal_candidate_priority: bool = False,
        enable_schema_compiled_source_certificate: bool = False,
        enable_source_support_obligation_admission: bool = False,
        enable_enhanced_source_support: bool = False,
        enable_obligation_admission: bool = False,
        enable_query_local_overlay_reuse: bool = True,
        enable_predicate_entailment_admission: bool = False,
        enable_binding_obligation_contract_validation: bool = False,
        enable_query_time_relation_materialization: bool = False,
        enable_binding_conditioned_typed_materialization: bool = False,
        cbo_calibration_path: str = "",
        cbo_min_samples: int = 8,
        cbo_risk_weight: float = 0.8,
        cbo_cost_weight: float = 0.25,
        bswc_beam_width: int = 8,
        bswc_search_horizon: int = 3,
        bswc_source_calibration_path: str = "",
        bswc_source_min_samples: int = 8,
        bswc_feedback_strength: float = 3.0,
        enable_bswc_query_feedback: bool = True,
        bswc_stop_threshold: float = 0.035,
        bswc_min_exploration_actions: int = 1,
        bswc_arwc_partial_value: float = 0.10,
        bswc_arwc_budget_penalty: float = 0.05,
        bswc_arwc_min_utility: float = 0.0,
        bswc_fpawc_stop_upper_bound: float = 0.0,
        verifier: Optional[Callable[[Mapping[str, Any], str], Mapping[str, Any]]] = None,
    ) -> None:
        self.relation_index = dict(relation_index or {})
        self._perturbation_doc_ids = tuple(sorted(
            _as_int(doc_id, -1)
            for doc_id in (self.relation_index.get("doc_title_surfaces") or {})
            if _as_int(doc_id, -1) >= 0
        ))
        self.get_source_content = get_source_content
        self.get_source_title = get_source_title
        self.completion_lookup = completion_lookup
        self.obligation_source_lookup = obligation_source_lookup
        self.relation_materializer = relation_materializer
        self.enable_query_time_relation_materialization = bool(
            enable_query_time_relation_materialization and relation_materializer
        )
        self.enable_binding_conditioned_typed_materialization = bool(
            self.enable_query_time_relation_materialization
            and enable_binding_conditioned_typed_materialization
        )
        self.current_source_candidates = [
            dict(value) for value in (current_source_candidates or [])
        ]
        self.completion_top_k = max(1, int(completion_top_k))
        self.completion_mode = str(completion_mode or "lazy").strip().lower()
        if self.completion_mode not in {"lazy", "eager", "off"}:
            raise ValueError(
                f"Unknown paper evidence completion mode: {completion_mode}"
            )
        self.completion_query_mode = str(
            completion_query_mode or "obligation_bound"
        ).strip().lower()
        if self.completion_query_mode not in {"obligation_bound", "entity_only"}:
            raise ValueError(
                "Unknown paper evidence completion query mode: "
                f"{completion_query_mode}"
            )
        self.verification_mode = str(
            verification_mode or "strict"
        ).strip().lower()
        if self.verification_mode not in {
            "strict", "tri_state_strict", "obligation_only", "unsafe_admit"
        }:
            raise ValueError(
                f"Unknown paper evidence verification mode: {verification_mode}"
            )
        self.repair_scheduler_mode = str(
            repair_scheduler_mode or "eager"
        ).strip().lower()
        if self.repair_scheduler_mode not in {"eager", "terminal_critical"}:
            raise ValueError(
                "Unknown paper evidence repair scheduler mode: "
                f"{repair_scheduler_mode}"
            )
        self.execution_mode = str(execution_mode or "saqe").strip().lower()
        if self.execution_mode not in {
            "saqe", "no_reopen", "no_invalidation", "restart",
        }:
            raise ValueError(
                f"Unknown paper evidence execution mode: {execution_mode}"
            )
        self.index_delete_rate = _bounded(index_delete_rate)
        self.index_corrupt_rate = _bounded(index_corrupt_rate)
        self.index_corrupt_mode = str(
            index_corrupt_mode or "object"
        ).strip().lower()
        if self.index_corrupt_mode not in {
            "object", "relation", "subtype", "direction", "role",
            "provenance", "mixed",
        }:
            raise ValueError(
                f"Unknown paper evidence index corruption mode: {index_corrupt_mode}"
            )
        self.perturbation_seed = int(perturbation_seed)
        self.max_facts_per_key = max(1, int(max_facts_per_key))
        self.max_key_fanout = max(1, int(max_key_fanout))
        self.enable_title_catalog_completion = bool(enable_title_catalog_completion)
        self.enable_provenance_source_verifier = bool(
            enable_provenance_source_verifier
        )
        self.enable_identity_safe_composite_join = bool(
            enable_identity_safe_composite_join
        )
        self.relation_matching_mode = str(
            relation_matching_mode or "canonical"
        ).strip().lower()
        if self.relation_matching_mode not in {"canonical", "exact_raw"}:
            raise ValueError(
                "Unknown paper evidence relation matching mode: "
                f"{relation_matching_mode}"
            )
        self.enable_predicate_subtype_constraints = bool(
            enable_predicate_subtype_constraints
        )
        self.enable_type_role_constraints = bool(enable_type_role_constraints)
        self.enable_composite_index_access = bool(
            enable_composite_index_access
        )
        self.enable_schema_plan_coverage_v7 = bool(
            enable_schema_plan_coverage_v7
        )
        self.enable_lineage_terminal_semijoin_v7_1 = bool(
            enable_lineage_terminal_semijoin_v7_1
        )
        self.enable_provenance_grouped_terminal_v7_2 = bool(
            enable_provenance_grouped_terminal_v7_2
        )
        self.enable_schema_exact_nested_join_v9 = bool(
            enable_schema_exact_nested_join_v9
        )
        self.enable_collision_aware_schema_identity = bool(
            enable_collision_aware_schema_identity
        )
        self.enable_safe_title_family_root_binding = bool(
            enable_safe_title_family_root_binding
        )
        self.enable_source_grounded_terminal_tuple_recovery = bool(
            enable_source_grounded_terminal_tuple_recovery
        )
        self.enable_obligation_bound_repair_certificate = bool(
            enable_obligation_bound_repair_certificate
        )
        self.enable_terminal_candidate_priority = bool(
            enable_terminal_candidate_priority
        )
        self.enable_schema_compiled_source_certificate = bool(
            enable_schema_compiled_source_certificate
        )
        self.enable_source_support_obligation_admission = bool(
            enable_source_support_obligation_admission
        )
        self.enable_enhanced_source_support = bool(
            enable_enhanced_source_support
            or self.enable_source_support_obligation_admission
        )
        self.enable_obligation_admission = bool(
            enable_obligation_admission
            or self.enable_source_support_obligation_admission
        )
        self.enable_query_local_overlay_reuse = bool(
            enable_query_local_overlay_reuse
        )
        self.enable_predicate_entailment_admission = bool(
            enable_predicate_entailment_admission
        )
        self.enable_binding_obligation_contract_validation = bool(
            enable_binding_obligation_contract_validation
        )
        self.cbo_calibration_path = str(cbo_calibration_path or "")
        self.cbo_min_samples = max(1, int(cbo_min_samples))
        self.cbo_risk_weight = max(0.0, float(cbo_risk_weight))
        self.cbo_cost_weight = max(0.0, float(cbo_cost_weight))
        self.bswc_beam_width = max(1, int(bswc_beam_width))
        self.bswc_search_horizon = max(1, int(bswc_search_horizon))
        self.bswc_source_calibration_path = str(bswc_source_calibration_path or "")
        self.bswc_source_min_samples = max(1, int(bswc_source_min_samples))
        self.bswc_feedback_strength = max(0.1, float(bswc_feedback_strength))
        self.enable_bswc_query_feedback = bool(enable_bswc_query_feedback)
        self.bswc_stop_threshold = max(0.0, float(bswc_stop_threshold))
        self.bswc_min_exploration_actions = max(0, int(bswc_min_exploration_actions))
        self.bswc_arwc_partial_value = max(0.0, float(bswc_arwc_partial_value))
        self.bswc_arwc_budget_penalty = max(
            0.0, float(bswc_arwc_budget_penalty)
        )
        self.bswc_arwc_min_utility = float(bswc_arwc_min_utility)
        self.bswc_fpawc_stop_upper_bound = min(
            1.0, max(0.0, float(bswc_fpawc_stop_upper_bound))
        )
        self.bswc_source_value_model = HierarchicalSourceValueModel.from_path(
            self.bswc_source_calibration_path,
            min_samples=self.bswc_source_min_samples,
        )
        self.cbo_cost_model = SampledActionCostModel.from_path(
            self.cbo_calibration_path,
            min_samples=self.cbo_min_samples,
        )
        verifier_class = (
            SourceSupportObligationAdmissionVerifier
            if (
                self.enable_enhanced_source_support
                or self.enable_obligation_admission
            )
            else RuleBasedSourceSupportVerifier
        )
        verifier_kwargs = {
            "enable_provenance_scope": self.enable_provenance_source_verifier,
        }
        if verifier_class is SourceSupportObligationAdmissionVerifier:
            verifier_kwargs.update({
                "enable_enhanced_source_support": (
                    self.enable_enhanced_source_support
                ),
                "enable_obligation_admission": (
                    self.enable_obligation_admission
                ),
                "enable_predicate_entailment_admission": (
                    self.enable_predicate_entailment_admission
                ),
                "enable_predicate_subtype_constraints": (
                    self.enable_predicate_subtype_constraints
                ),
                "enable_type_role_constraints": self.enable_type_role_constraints,
            })
        self.verifier = verifier or verifier_class(**verifier_kwargs)
        self.verifier_version = str(
            getattr(self.verifier, "version", "injected_source_verifier_v1")
        )
        if self.verification_mode == "unsafe_admit":
            self.verifier_version = "unsafe_source_pointer_v1"
        elif self.verification_mode == "obligation_only":
            self.verifier_version = "obligation_only_ablation_v1"
        self._verify_cache: Dict[str, Dict[str, Any]] = {}
        self._source_support_cache: Dict[str, Dict[str, Any]] = {}



        self._source_document_cache: Dict[int, Dict[str, Any]] = {}
        self._inspected_source_ids: set[int] = set()

    def _effective_schema_operator(
        self, operator: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Apply experiment-only schema controls without mutating the plan."""
        effective = dict(operator or {})
        if not self.enable_predicate_subtype_constraints:
            effective["required_predicate_subtypes"] = []
            effective["strict_predicate_subtype"] = False
            if str(effective.get("join_key_contract") or "").startswith(
                "entity_identity+predicate_subtype"
            ):
                effective["join_key_contract"] = "exact_normalized_entity_key"
        effective["schema_predicate_subtype_constraints"] = bool(
            self.enable_predicate_subtype_constraints
        )
        effective["schema_type_role_constraints"] = bool(
            self.enable_type_role_constraints
        )
        return effective

    def initial_bindings(self, question, logical_plan):
        from .relation_fact_join_utils import (
            _bind_explicit_query_entities,
            _query_root_surface_match,
            canonical_root_entity_categories,
            canonical_entity_identity_key,
            infer_question_root_entity_categories,
        )

        operators = list((logical_plan or {}).get("operators") or [])
        if not operators:
            return []
        branch_plan = str((logical_plan or {}).get("plan_kind") or "").startswith(
            "branch_"
        )
        inferred_root_categories = infer_question_root_entity_categories(
            question, branch_plan=branch_plan
        )
        semantic_ir = dict(
            (logical_plan or {}).get("semantic_query_ir") or {}
        )
        if semantic_ir:
            relations = list(semantic_ir.get("relations") or [])
            produced_variables = {
                str(relation.get("output_variable") or "")
                for relation in relations
                if str(relation.get("output_variable") or "")
            }
            root_variables = {
                str(relation.get("input_variable") or "")
                for relation in relations
                if (
                    str(relation.get("input_variable") or "")
                    and str(relation.get("input_variable") or "")
                    not in produced_variables
                )
            }
            semantic_starts = [
                dict(start)
                for start in (semantic_ir.get("start_entities") or [])
                if (
                    str(start.get("mention") or "").strip()
                    and str(start.get("variable") or "") in root_variables
                )
            ]
            expected_roots = 2 if branch_plan else 1
            if len(semantic_starts) != expected_roots:
                return []

            roots: List[Dict[str, Any]] = []
            for start in semantic_starts:
                mention = str(start.get("mention") or "").strip()
                semantic_root_categories = canonical_root_entity_categories([
                    start.get("semantic_type") or ""
                ]) or set(inferred_root_categories)




                if not _query_root_surface_match(question, mention):
                    return []
                matched = _bind_explicit_query_entities(
                    mention,
                    self.relation_index,
                    operator=dict(operators[0]),
                    expected_count=1,
                    max_bindings=1,
                    require_operator_match=False,
                    canonicalize_surface=not branch_plan,
                    enforce_root_contract=True,
                    use_title_entity_keys=True,
                    enable_collision_aware_title_identity=(
                        self.enable_collision_aware_schema_identity
                    ),
                    enable_safe_title_family_query_span=(
                        self.enable_safe_title_family_root_binding
                    ),
                    expected_root_categories=semantic_root_categories,
                )
                if len(matched) != 1:
                    return []
                root = dict(matched[0])




                mention_identity = canonical_entity_identity_key(
                    re.sub(r"\([^)]*\)", " ", mention)
                )
                bound_mention_identity = canonical_entity_identity_key(
                    re.sub(
                        r"\([^)]*\)",
                        " ",
                        str(root.get("mention_surface") or ""),
                    )
                )
                if not mention_identity or mention_identity != bound_mention_identity:
                    return []
                if bool(root.get("surface_canonicalized")) and not bool(
                    root.get("title_catalog_exact")
                    or root.get("collision_aware_title_identity")
                    or root.get("collision_safe_legacy_fallback")
                    or str(root.get("binding_method") or "").startswith(
                        "provenance_alias_"
                    )
                ):
                    return []
                root["semantic_start_variable"] = str(
                    start.get("variable") or ""
                )
                root["semantic_start_type"] = str(
                    start.get("semantic_type") or ""
                )
                root["binding_method"] = (
                    "semantic_plan_start_entity:"
                    + str(root.get("binding_method") or "exact_query_span")
                )
                roots.append(root)
            if len({str(root.get("key") or "") for root in roots}) != len(roots):
                return []
            return roots

        return _bind_explicit_query_entities(
            question,
            self.relation_index,
            operator=dict(operators[0]),
            expected_count=2 if branch_plan else 1,
            max_bindings=2 if branch_plan else 4,
            require_operator_match=False,
            canonicalize_surface=not branch_plan,
            enforce_root_contract=True,
            use_title_entity_keys=True,
            enable_collision_aware_title_identity=(
                self.enable_collision_aware_schema_identity
            ),
            enable_safe_title_family_query_span=(
                self.enable_safe_title_family_root_binding
            ),
            expected_root_categories=inferred_root_categories,
        )

    def _bound_input(self, binding, operator) -> Tuple[str, str]:
        input_slot = str(operator.get("input_slot") or "$root")
        slot = dict((binding.get("slot_bindings") or {}).get(input_slot) or {})
        return str(slot.get("surface") or ""), str(slot.get("key") or "")

    def _bound_slot(self, binding, operator) -> Dict[str, Any]:
        input_slot = str(operator.get("input_slot") or "$root")
        return dict((binding.get("slot_bindings") or {}).get(input_slot) or {})

    def _category_partition_scope(
        self,
        binding: Mapping[str, Any],
        operator: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return the document scope for the currently consumed binding.

        A category partition disambiguates one title family (for example a
        film from an album with the same title).  It is not a global source
        whitelist: a different page may still support the relation when its
        text preserves the bound entity identity.
        """
        slot = self._bound_slot(binding, operator)
        partition_doc_ids = {
            _as_int(value, -1)
            for value in slot.get("root_category_partition_doc_ids") or []
            if _as_int(value, -1) >= 0
        }
        expected_categories = {
            str(value).strip().casefold()
            for value in slot.get("root_category_partition_categories") or []
            if str(value).strip()
        }
        bound_surface, bound_key = self._bound_input(binding, operator)
        title_key_to_family = self.relation_index.get(
            "title_key_to_family_key"
        ) or {}
        family_key = str(
            title_key_to_family.get(bound_key)
            or title_key_to_family.get(_key(bound_surface))
            or ""
        )
        family_doc_ids = {
            _as_int(value, -1)
            for value in (
                (self.relation_index.get("title_family_doc_ids") or {}).get(
                    family_key
                )
                or []
            )
            if _as_int(value, -1) >= 0
        }
        return {
            "active": bool(
                slot.get("root_category_partition_validated")
                and partition_doc_ids
            ),
            "bound_surface": bound_surface,
            "bound_key": bound_key,
            "partition_doc_ids": partition_doc_ids,
            "expected_categories": expected_categories,
            "title_family_doc_ids": family_doc_ids,
        }

    def _category_partition_source_decision(
        self,
        binding: Mapping[str, Any],
        operator: Mapping[str, Any],
        doc_id: int,
        *,
        source_text: str = "",
    ) -> Tuple[bool, str]:
        """Reject identity conflicts without blocking legitimate cross-pages."""
        from .relation_fact_join_utils import (
            _equivalent_entity_keys,
            canonical_root_entity_categories,
        )

        scope = self._category_partition_scope(binding, operator)
        if not scope["active"]:
            return True, "partition_inactive"
        doc_id = _as_int(doc_id, -1)
        if doc_id in scope["partition_doc_ids"]:
            return True, "exact_category_partition"
        if doc_id in scope["title_family_doc_ids"]:
            return False, "same_title_category_conflict"




        text = re.sub(r"\s+", " ", str(source_text or "")).strip()
        identity_surfaces = {str(scope["bound_surface"] or "").strip()}
        equivalent_keys = _equivalent_entity_keys(
            self.relation_index,
            str(scope["bound_key"] or ""),
            allow_title_provenance_alias=True,
        )
        canonical_surfaces = self.relation_index.get(
            "canonical_entity_surfaces"
        ) or {}
        title_surfaces = self.relation_index.get("title_entity_surfaces") or {}
        for key in equivalent_keys:
            identity_surfaces.update({
                str(key or "").strip(),
                str(canonical_surfaces.get(key) or "").strip(),
                str(title_surfaces.get(key) or "").strip(),
            })
        identity_patterns = []
        for surface in sorted(identity_surfaces, key=lambda value: -len(value)):
            tokens = _tokens(surface)
            if not tokens:
                continue
            identity_patterns.append(
                r"\b" + r"[\s'\u2019\-]+".join(
                    re.escape(token) for token in tokens
                ) + r"\b"
            )
        identity_matches = [
            match
            for pattern in identity_patterns
            for match in re.finditer(pattern, text, flags=re.IGNORECASE)
        ]
        if text and not identity_matches:
            return False, "cross_page_bound_identity_absent"
        if text:
            for match in identity_matches:
                window = text[
                    max(0, match.start() - 80):min(len(text), match.end() + 80)
                ]
                mentioned_categories = canonical_root_entity_categories([window])
                if (
                    mentioned_categories
                    and scope["expected_categories"]
                    and not mentioned_categories.intersection(
                        scope["expected_categories"]
                    )
                ):
                    return False, "cross_page_bound_category_conflict"
        return True, "identity_preserving_cross_page"

    @staticmethod
    def _query_binding_qualifiers(
        binding: Mapping[str, Any], operator: Mapping[str, Any]
    ) -> List[str]:
        """Return explicit parenthetical/comma qualifiers from a root mention."""
        input_slot = str(operator.get("input_slot") or "$root")
        slot = dict((binding.get("slot_bindings") or {}).get(input_slot) or {})
        if slot.get("fact_id"):
            return sorted({
                str(value).strip().casefold()
                for value in slot.get("source_identity_qualifier_tokens") or []
                if str(value).strip()
            })
        metadata = dict(binding.get("root_metadata") or {})
        mention = str(metadata.get("mention_surface") or "")
        if not mention:
            return []
        qualifier_segments = re.findall(r"\(([^)]*)", mention)
        if "," in mention:
            qualifier_segments.append(mention.split(",", 1)[1])
        qualifier_segments.extend(
            str(value)
            for value in metadata.get("canonical_title_family_qualifiers") or []
            if str(value).strip()
        )
        explicit_tokens = set(_tokens(" ".join(qualifier_segments)))
        mention_tokens = set(_tokens(metadata.get("mention_key") or mention))
        bound_tokens = set(_tokens(slot.get("key") or metadata.get("key") or ""))
        return sorted(explicit_tokens & (mention_tokens - bound_tokens))

    def _output_binding_joinability(
        self, output_key: str, operator: Mapping[str, Any]
    ) -> Tuple[bool, bool]:
        requires_downstream = bool(
            str(operator.get("slot_plan_stage") or "") == "intermediate_binding"
            and str(operator.get("output_type") or "entity").lower()
            in {"entity", "person", "organization", "organisation", "work"}
        )
        if not requires_downstream:
            return False, True
        indexes = [
            self.relation_index.get("subject_to_items") or {},
            self.relation_index.get("object_to_items") or {},
            self.relation_index.get("title_entity_doc_ids") or {},
        ]
        checked = any(bool(index) for index in indexes)
        return checked, bool(output_key and any(output_key in index for index in indexes))

    def _source_title_for_item(self, item: Mapping[str, Any]) -> str:
        source_id = item.get(
            "source_id", item.get("completion_source_doc_id", item.get("doc_idx"))
        )
        try:
            source_id = int(source_id)
        except (TypeError, ValueError):
            return ""
        if self.get_source_title is not None:
            try:
                title = str(self.get_source_title(source_id) or "").strip()
                if title:
                    return title
            except (TypeError, ValueError, KeyError, IndexError):
                pass
        try:
            content = str(self.get_source_content(source_id) or "")
        except (TypeError, ValueError, KeyError, IndexError):
            return ""
        return content.splitlines()[0].strip() if content else ""

    def _identity_scoped_rows(
        self,
        binding: Mapping[str, Any],
        operator: Mapping[str, Any],
        stats: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Probe the canonical title's fact partition before mention aliases."""
        raw_scan_limit = operator.get("execution_scan_limit")
        scan_limit = (
            None if raw_scan_limit is None else max(0, _as_int(raw_scan_limit))
        )
        row_cap = self.max_facts_per_key
        if scan_limit is not None:
            row_cap = min(row_cap, scan_limit)
            if row_cap <= 0:
                stats["budget_truncated_rows"] = int(
                    stats.get("budget_truncated_rows", 0) or 0
                ) + 1
                return []
        if not self.enable_identity_safe_composite_join:
            return []
        slot = self._bound_slot(binding, operator)
        identity_key = str(slot.get("canonical_identity_key") or "").strip()
        if not slot.get("canonical_identity_validated") or not identity_key:
            return []
        direct_root_partition = bool(
            not slot.get("fact_id")
            and slot.get("allow_title_partition_direct_lookup")
            and (
                slot.get("canonical_title_family_unambiguous")
                or slot.get("root_category_partition_validated")
            )
            and (
                str(operator.get("slot_plan_stage") or "") == "final_answer"
                or str(operator.get("output_slot") or "") == "$answer"
                or (
                    slot.get("root_category_partition_validated")
                    and str(operator.get("slot_plan_stage") or "")
                    == "comparison_value"
                )
            )
        )
        if (
            self.enable_schema_plan_coverage_v7
            and not self.enable_schema_exact_nested_join_v9
        ):



            if not slot.get("fact_id") and not direct_root_partition:
                stats["identity_probe_skipped_unbound_root"] = 1
                return []
            if not (
                str(operator.get("slot_plan_stage") or "") == "final_answer"
                or str(operator.get("output_slot") or "") == "$answer"
            ):
                stats["identity_probe_skipped_nonterminal"] = 1
                return []
        if direct_root_partition:
            doc_ids = list(slot.get("canonical_title_doc_ids") or [])
            if not doc_ids:
                doc_ids = list(
                    slot.get("canonical_title_family_doc_ids") or []
                )
            doc_ids = list(dict.fromkeys(doc_ids))
            stats["title_partition_direct_root"] = 1
        else:
            doc_ids = list(
                (self.relation_index.get("title_entity_doc_ids") or {}).get(identity_key)
                or []
            )
        if (
            self.enable_schema_exact_nested_join_v9
            and slot.get("canonical_title_family_unambiguous")
        ):
            doc_ids = list(dict.fromkeys(
                doc_ids
                + list(slot.get("canonical_title_family_doc_ids") or [])
            ))
        if self.enable_schema_plan_coverage_v7:
            from .relation_fact_join_utils import (
                _equivalent_entity_keys,
                _oriented_endpoint_for_operator_binding,
                entity_predicate_index_key,
            )

            predicate_families = set(operator.get("predicate_families") or [])
            required_subtypes = set(
                operator.get("required_predicate_subtypes") or []
            )
            output_type = str(operator.get("output_type") or "").strip().lower()
            predicate_keys = set(predicate_families)
            index_suffix = "predicate_family_to_items"
            if required_subtypes:
                predicate_keys = set(required_subtypes)
                index_suffix = "predicate_subtype_to_items"
            elif predicate_families & {"birth", "death"} and output_type in {
                "date", "place",
            }:
                predicate_keys = {
                    f"{family}_{output_type}"
                    for family in predicate_families & {"birth", "death"}
                }
                index_suffix = "predicate_subtype_to_items"
            semantic_predicate_keys = set(predicate_keys)
            if not self.enable_composite_index_access:
                predicate_keys = {"__entity_only__"}
                index_suffix = "to_items"
            if not doc_ids or not predicate_keys:
                stats["title_terminal_direct_miss"] = 1
                return []

            accepted_doc_ids = {
                int(value) for value in doc_ids
                if str(value).lstrip("-").isdigit()
            }
            probe_keys = _equivalent_entity_keys(
                self.relation_index,
                identity_key,
                allow_title_provenance_alias=True,
            )
            probe_keys.add(identity_key)
            items = list(self.relation_index.get("items") or [])
            rows: List[Dict[str, Any]] = []
            seen = set()
            for entity_key in sorted(
                probe_keys, key=lambda value: (value != identity_key, value)
            ):
                for side in ("subject", "object"):
                    index = self.relation_index.get(
                        f"{side}_{index_suffix}"
                    ) or {}
                    for predicate_key in sorted(predicate_keys):
                        composite_key = (
                            entity_key
                            if predicate_key == "__entity_only__"
                            else entity_predicate_index_key(
                                entity_key, predicate_key
                            )
                        )
                        raw_item_ids = list(index.get(composite_key) or [])
                        stats["lookup_keys"] = int(
                            stats.get("lookup_keys", 0) or 0
                        ) + 1
                        stats["loaded_doc_rows"] = int(
                            stats.get("loaded_doc_rows", 0) or 0
                        ) + len(raw_item_ids)
                        for raw_item_id in raw_item_ids:
                            try:
                                item_id = int(raw_item_id)
                                item = items[item_id]
                                doc_id = int(item.get("doc_idx", -1))
                            except (IndexError, TypeError, ValueError):
                                continue
                            if doc_id not in accepted_doc_ids or item_id in seen:
                                continue
                            seen.add(item_id)
                            direct_item = dict(item)
                            direct_item["access_path"] = (
                                "canonical_title_terminal_composite_index"
                                if self.enable_composite_index_access
                                else "canonical_title_terminal_entity_index"
                            )
                            direct_item["schema_terminal_direct_access"] = True
                            rows.append(direct_item)
                            if len(rows) >= row_cap:
                                break
                        if len(rows) >= row_cap:
                            break
                    if len(rows) >= row_cap:
                        break
                if len(rows) >= row_cap:
                    break
            if scan_limit is not None and len(rows) >= scan_limit:
                stats["budget_truncated_rows"] = int(
                    stats.get("budget_truncated_rows", 0) or 0
                ) + 1
            if (
                self.enable_lineage_terminal_semijoin_v7_1
                and len(rows) > 1
                and output_type == "date"
            ):
                lineage_years = {
                    str(value)
                    for value in slot.get("lineage_source_years") or []
                    if str(value)
                }
                if lineage_years:
                    _, bound_key = self._bound_input(binding, operator)
                    equivalent_keys = _equivalent_entity_keys(
                        self.relation_index,
                        bound_key,
                        allow_title_provenance_alias=True,
                    )
                    terminal_values = set()
                    qualified_rows = []
                    for item in rows:
                        oriented = _oriented_endpoint_for_operator_binding(
                            item,
                            dict(operator),
                            bound_key,
                            equivalent_keys=equivalent_keys,
                        )
                        output_surface = str(
                            (oriented or {}).get("surface") or ""
                        ).strip()
                        if output_surface:
                            terminal_values.add(_key(output_surface))
                        if _year_values(output_surface).intersection(lineage_years):
                            qualified_rows.append(item)
                    if len(terminal_values) > 1:
                        stats["terminal_conflict_count"] = 1
                    if qualified_rows:
                        stats["lineage_qualifier_available"] = 1
                        stats["lineage_qualifier_pruned_rows"] = max(
                            0, len(rows) - len(qualified_rows)
                        )
                        rows = qualified_rows
            stats["input_scan_rows"] = int(
                stats.get("input_scan_rows", 0) or 0
            ) + len(rows)
            stats["rows_after_cap"] = int(
                stats.get("rows_after_cap", 0) or 0
            ) + len(rows)
            stats["identity_scoped_doc_count"] = len(accepted_doc_ids)
            stats["identity_scoped_rows"] = len(rows)
            stats["identity_scoped_key"] = identity_key
            stats["title_terminal_direct_actions"] = 1
            stats["title_terminal_direct_rows"] = len(rows)
            stats["title_terminal_direct_predicates"] = sorted(
                semantic_predicate_keys
            )
            stats["composite_index_access"] = int(
                self.enable_composite_index_access
            )
            return rows
        items = list(self.relation_index.get("items") or [])
        doc_to_items = self.relation_index.get("doc_to_items") or {}
        rows: List[Dict[str, Any]] = []
        seen = set()
        for raw_doc_id in doc_ids:
            try:
                doc_id = int(raw_doc_id)
            except (TypeError, ValueError):
                continue
            raw_item_ids = list(doc_to_items.get(doc_id) or [])
            stats["loaded_doc_rows"] = int(
                stats.get("loaded_doc_rows", 0) or 0
            ) + len(raw_item_ids)
            remaining = max(0, row_cap - len(rows))
            selected_item_ids = raw_item_ids[:remaining]
            stats["input_scan_rows"] = int(
                stats.get("input_scan_rows", 0) or 0
            ) + len(selected_item_ids)
            stats["rows_after_cap"] = int(
                stats.get("rows_after_cap", 0) or 0
            ) + len(selected_item_ids)
            stats["truncated_rows"] = int(
                stats.get("truncated_rows", 0) or 0
            ) + max(0, len(raw_item_ids) - len(selected_item_ids))
            for raw_item_id in selected_item_ids:
                try:
                    item_id = int(raw_item_id)
                except (TypeError, ValueError):
                    continue
                if item_id in seen or item_id < 0 or item_id >= len(items):
                    continue
                seen.add(item_id)
                rows.append(items[item_id])
            if len(rows) >= row_cap:
                break
        if scan_limit is not None and len(rows) >= scan_limit:
            stats["budget_truncated_rows"] = int(
                stats.get("budget_truncated_rows", 0) or 0
            ) + 1
        stats["identity_scoped_doc_count"] = len(set(doc_ids))
        stats["identity_scoped_rows"] = len(rows)
        stats["identity_scoped_key"] = identity_key
        return rows

    def _matches(self, items, bound_surface, bound_key, operator, binding=None):
        from .relation_fact_join_utils import (
            TERMINAL_TYPE_INCOMPATIBLE,
            _completion_title_identity_decision,
            _equivalent_entity_keys,
            _oriented_endpoint_for_operator_binding,
            terminal_candidate_category_priority,
            terminal_value_contract_compatibility,
        )

        equivalent = _equivalent_entity_keys(
            self.relation_index,
            bound_key,
            allow_title_provenance_alias=True,
        )
        facts = []
        matches = []
        diagnostics = {
            "lineage_identity_rejected_count": 0,
            "identity_scoped_match_count": 0,
        }
        bound_slot = self._bound_slot(binding or {}, operator)
        lineage_bound = bool(bound_slot.get("fact_id"))
        category_scope = self._category_partition_scope(binding or {}, operator)
        root_category_doc_ids = set(category_scope["partition_doc_ids"])
        root_category_partition_active = bool(
            not lineage_bound and category_scope["active"]
        )
        weak_lineage_surface = bool(lineage_bound and len(_tokens(bound_surface)) == 1)
        canonical_identity_surface = str(
            bound_slot.get("canonical_identity_surface") or ""
        ).strip()
        canonical_identity_key = str(
            bound_slot.get("canonical_identity_key") or ""
        ).strip()
        identity_contract_active = bool(
            self.enable_identity_safe_composite_join
            and lineage_bound
            and bound_slot.get("canonical_identity_validated")
            and canonical_identity_surface
            and canonical_identity_key
        )
        if (
            lineage_bound
            and bound_slot.get("canonical_identity_qualifier_conflict")
        ):
            diagnostics["lineage_identity_qualifier_conflict_count"] = len(items)
            return facts, matches, diagnostics
        for raw_item in items:
            item = dict(raw_item)
            if root_category_partition_active:
                item_doc_id = _as_int(
                    item.get("source_id", item.get("doc_idx")), -1
                )
                source_text = str(item.get("source_sentence") or "")
                if not source_text and item_doc_id >= 0:
                    try:
                        source_text = str(
                            self.get_source_content(item_doc_id) or ""
                        )
                    except (KeyError, IndexError, TypeError, ValueError):
                        source_text = ""
                partition_allowed, partition_reason = (
                    self._category_partition_source_decision(
                        binding or {},
                        operator,
                        item_doc_id,
                        source_text=source_text,
                    )
                )
                if not partition_allowed:
                    diagnostics["root_category_partition_rejected_count"] = int(
                        diagnostics.get(
                            "root_category_partition_rejected_count", 0
                        )
                    ) + 1
                    diagnostics[
                        f"root_category_partition_{partition_reason}_count"
                    ] = int(diagnostics.get(
                        f"root_category_partition_{partition_reason}_count", 0
                    )) + 1
                    continue
            match_operator = dict(operator)
            if self.relation_matching_mode == "exact_raw":
                triple = list(item.get("triple") or [])
                raw_relation = str(
                    item.get("relation")
                    or (triple[1] if len(triple) == 3 else "")
                    or ""
                )
                expected_relation = str(
                    operator.get("canonical_relation") or ""
                ).replace("_", " ")
                raw_relation_key = " ".join(_tokens(raw_relation))
                expected_relation_key = " ".join(_tokens(expected_relation))
                if (
                    not raw_relation_key
                    or not expected_relation_key
                    or raw_relation_key != expected_relation_key
                ):
                    diagnostics["exact_raw_relation_rejected_count"] = int(
                        diagnostics.get("exact_raw_relation_rejected_count", 0)
                    ) + 1
                    continue



                match_operator["relation_tokens"] = list(
                    item.get("relation_tokens") or _tokens(raw_relation)
                )
                match_operator["predicate_families"] = []
                match_operator["relation_families"] = []
                match_operator["required_predicate_subtypes"] = []
            oriented = _oriented_endpoint_for_operator_binding(
                item,
                match_operator,
                bound_key,
                equivalent_keys=equivalent,
            )
            if oriented is None:
                continue
            input_side = str(oriented.get("input_side") or "")
            input_surface = str(item.get(input_side) or "")
            input_key = str(item.get(f"{input_side}_key") or _key(input_surface))
            if weak_lineage_surface and _key(input_surface) != _key(bound_surface):
                diagnostics["lineage_identity_rejected_count"] += 1
                continue
            if not _initial_sensitive_identity_matches(
                bound_surface, input_surface
            ):
                continue
            identity_join_mode = "mention_key"
            identity_confidence = 0.5
            identity_source_title = ""
            if identity_contract_active:
                if input_key == canonical_identity_key:
                    identity_join_mode = "exact_canonical_entity_key"
                    identity_confidence = 1.0
                else:
                    identity_source_title = self._source_title_for_item(item)
                    mode, reject_reason = _completion_title_identity_decision(
                        bound_surface=canonical_identity_surface,
                        bound_identity_keys={canonical_identity_key},
                        title=identity_source_title,
                        enabled=True,
                    )
                    if not mode:
                        diagnostics["lineage_identity_rejected_count"] += 1
                        diagnostics[
                            f"identity_rejected_{reject_reason or 'unresolved'}"
                        ] = int(diagnostics.get(
                            f"identity_rejected_{reject_reason or 'unresolved'}", 0
                        )) + 1
                        continue
                    identity_join_mode = f"canonical_title_doc:{mode}"
                    identity_confidence = 0.98
                    diagnostics["identity_scoped_match_count"] += 1
            output_surface = str(oriented.get("surface") or "")
            target_entity_categories = (
                operator.get("target_entity_categories") or []
                if "target_entity_categories" in operator
                else operator.get("relation_tokens") or []
            )
            if self.enable_type_role_constraints:
                terminal_type = terminal_value_contract_compatibility(
                    output_surface,
                    str(operator.get("output_type") or "entity"),
                    item.get("relation_tokens") or [],
                    target_category_tokens=target_entity_categories,
                    output_role=str(oriented.get("output_side") or ""),
                    source_text=str(item.get("source_sentence") or ""),
                    relation_families=sorted(
                        set(item.get("relation_families") or [])
                        | set(item.get("predicate_families") or [])
                        | set(operator.get("relation_families") or [])
                        | set(operator.get("predicate_families") or [])
                    ),
                )
                if terminal_type["status"] == TERMINAL_TYPE_INCOMPATIBLE:
                    continue
            else:
                terminal_type = {
                    "status": "OFF",
                    "reason": "type_role_constraints_disabled",
                }
            item["terminal_type_prefilter_status"] = terminal_type["status"]
            item["terminal_type_prefilter_reason"] = terminal_type["reason"]
            item["source_id"] = item.get(
                "completion_source_doc_id", item.get("doc_idx")
            )
            item["role_direction_required"] = bool(
                operator.get("role_direction_required")
            )
            item["canonical_relation"] = str(
                operator.get("canonical_relation") or ""
            )
            item["operator_id"] = str(operator.get("operator_id") or "")
            item["relation_matching_mode"] = self.relation_matching_mode
            item["composite_index_access"] = bool(
                self.enable_composite_index_access
            )
            subject_key = str(
                item.get("subject_key")
                or _key((item.get("triple") or ["", "", ""])[0])
            )
            object_key = str(
                item.get("object_key")
                or _key((item.get("triple") or ["", "", ""])[2])
            )
            output_key = str(oriented.get("key") or _key(output_surface))
            if output_key == subject_key:
                item["bound_output_role"] = "subject"
            elif output_key == object_key:
                item["bound_output_role"] = "object"
            else:
                item["bound_output_role"] = ""
            item["bound_output_key"] = output_key
            item["source_title"] = str(
                item.get("source_title")
                or item.get("completion_source_title")
                or self._source_title_for_item(item)
                or ""
            ).strip()
            joinability_checked, output_joinable = (
                self._output_binding_joinability(output_key, operator)
            )
            required_input_qualifiers = self._query_binding_qualifiers(
                binding or {}, operator
            )
            from .relation_fact_join_utils import (
                canonical_operator_predicate_contract,
            )
            operator_predicate_contract = canonical_operator_predicate_contract(
                operator.get("relation_tokens") or [],
                operator.get("predicate_families") or [],
            )
            item["obligation_contract"] = {
                "version": SourceSupportObligationAdmissionVerifier.contract_version,
                "operator_id": str(operator.get("operator_id") or ""),
                "predicate_families": sorted(
                    str(value) for value in operator.get("predicate_families") or []
                ),
                "relation_families": sorted(
                    str(value) for value in operator.get("relation_families") or []
                ),
                "required_predicate_subtypes": sorted(
                    str(value)
                    for value in operator.get("required_predicate_subtypes") or []
                ),
                "strict_predicate_subtype": bool(
                    operator.get("strict_predicate_subtype")
                ),
                "relation_tokens": sorted(
                    str(value) for value in operator.get("relation_tokens") or []
                ),
                "target_entity_categories": sorted(
                    str(value)
                    for value in operator.get("target_entity_categories") or []
                ),
                "physical_relation_tokens": sorted(
                    str(value) for value in item.get("relation_tokens") or []
                ),
                "allowed_predicate_subtypes": list(
                    operator_predicate_contract["allowed_semantic_subtypes"]
                ),
                "predicate_contract_version": (
                    operator_predicate_contract["version"]
                ),
                "output_type": str(operator.get("output_type") or "entity"),
                "input_role": str(operator.get("input_role") or ""),
                "output_role": str(operator.get("output_role") or ""),
                "role_direction_required": bool(
                    operator.get("role_direction_required")
                ),
                "canonical_relation": str(
                    operator.get("canonical_relation") or ""
                ),
                "bound_input_surface": str(bound_surface or ""),
                "bound_input_key": str(bound_key or ""),
                "accepted_input_keys": sorted(
                    str(value) for value in equivalent | {str(bound_key or "")}
                    if str(value)
                ),
                "bound_input_side": input_side,
                "bound_output_role": str(oriented.get("output_side") or ""),
                "bound_output_key": output_key,
                "role_direction_validated": bool(
                    oriented.get("role_direction_validated")
                ),
                "role_direction_confidence": _as_float(
                    oriented.get("role_direction_confidence")
                ),
                "binding_contract_enforced": bool(
                    self.enable_binding_obligation_contract_validation
                ),
                "required_input_qualifier_tokens": required_input_qualifiers,
                "input_identity_scope": (
                    "query_root_qualified_identity"
                    if required_input_qualifiers else "normalized_binding_key"
                ),
                "root_category_partition_enforced": bool(
                    root_category_partition_active
                ),
                "root_category_partition_doc_ids": sorted(
                    root_category_doc_ids
                ),
                "output_source_span_required": bool(
                    item.get("bound_completion_fact")
                ),
                "requires_downstream_join": bool(
                    str(operator.get("slot_plan_stage") or "")
                    == "intermediate_binding"
                ),
                "output_joinability_checked": joinability_checked,
                "output_joinable": output_joinable,
            }
            item["lineage_identity_validated"] = bool(identity_contract_active)
            item["lineage_identity_key"] = canonical_identity_key
            item["lineage_identity_surface"] = canonical_identity_surface
            item["lineage_identity_join_mode"] = identity_join_mode
            item["lineage_identity_confidence"] = identity_confidence
            item["lineage_identity_source_title"] = identity_source_title
            item["fact_id"] = _fact_id(item)
            terminal_priority = {
                "score": 0.0,
                "reason": "disabled",
                "target_categories": [],
                "value_categories": [],
                "descriptor_categories": [],
            }
            if (
                self.enable_terminal_candidate_priority
                and target_entity_categories
                and str(operator.get("slot_plan_stage") or "") == "final_answer"
            ):
                priority_source_text = str(item.get("source_sentence") or "")
                priority_source_id = _as_int(
                    item.get("source_id", item.get("doc_idx")), -1
                )
                if priority_source_id >= 0:
                    try:
                        full_source_text = str(
                            self.get_source_content(priority_source_id) or ""
                        )
                    except (KeyError, IndexError, TypeError, ValueError):
                        full_source_text = ""
                    if full_source_text:
                        priority_source_text = full_source_text
                terminal_priority = terminal_candidate_category_priority(
                    output_surface,
                    target_category_tokens=target_entity_categories,
                    source_text=priority_source_text,
                )
            item["terminal_candidate_priority_score"] = float(
                terminal_priority["score"]
            )
            item["terminal_candidate_priority_reason"] = str(
                terminal_priority["reason"]
            )
            title_surface = str(
                (self.relation_index.get("title_entity_surfaces") or {}).get(
                    output_key
                )
                or ""
            ).strip()
            facts.append(item)
            matches.append({
                "fact_id": item["fact_id"],
                "output_surface": output_surface,
                "output_key": output_key,
                "canonical_identity_surface": title_surface,
                "canonical_identity_key": output_key if title_surface else "",
                "canonical_identity_validated": bool(title_surface),
                "identity_join_mode": identity_join_mode,
                "terminal_candidate_priority_score": float(
                    terminal_priority["score"]
                ),
                "terminal_candidate_priority_reason": str(
                    terminal_priority["reason"]
                ),
                "terminal_candidate_descriptor_categories": list(
                    terminal_priority["descriptor_categories"]
                ),
            })
        if self.enable_terminal_candidate_priority and len(matches) > 1:
            ranked = sorted(
                enumerate(zip(facts, matches)),
                key=lambda pair: (
                    -_as_float(
                        pair[1][1].get("terminal_candidate_priority_score")
                    ),
                    pair[0],
                ),
            )
            facts = [pair[1][0] for pair in ranked]
            matches = [pair[1][1] for pair in ranked]
            diagnostics["terminal_candidate_priority_applied_count"] = 1
            diagnostics["terminal_candidate_priority_promoted_count"] = sum(
                1 for new_index, (old_index, _) in enumerate(ranked)
                if new_index < old_index
            )
            diagnostics["terminal_candidate_priority_top"] = [
                {
                    "output_surface": str(match.get("output_surface") or ""),
                    "score": _as_float(
                        match.get("terminal_candidate_priority_score")
                    ),
                    "reason": str(
                        match.get("terminal_candidate_priority_reason") or ""
                    ),
                }
                for match in matches[:5]
            ]
        return facts, matches, diagnostics

    def _perturbation_score(self, kind: str, fact_id: str) -> float:
        material = f"{self.perturbation_seed}\x1e{kind}\x1e{fact_id}"
        digest = hashlib.sha1(material.encode()).hexdigest()[:15]
        return int(digest, 16) / float(16 ** len(digest))

    @staticmethod
    def _corrupted_surface(surface: str, output_type: str) -> str:
        raw = str(surface or "").strip()
        kind = str(output_type or "entity").strip().lower()
        year_match = re.fullmatch(r"(1[0-9]{3}|20[0-9]{2})", raw)
        if year_match:
            return str(int(year_match.group(1)) + 1)
        if kind in {"date", "year"}:
            return "1 January 1900" if raw != "1 January 1900" else "2 January 1900"
        if kind in {"number", "count", "quantity"}:
            number_match = re.search(r"-?\d+(?:\.\d+)?", raw)
            if number_match:
                value = float(number_match.group(0)) + 1.0
                return str(int(value)) if value.is_integer() else str(value)
            return "1"
        if kind in {"boolean", "bool"}:
            return "no" if raw.casefold() == "yes" else "yes"
        replacements = {
            "person": "Alex Morgan",
            "place": "Springfield",
            "location": "Springfield",
            "city": "Springfield",
            "country": "Canada",
            "occupation": "teacher",
            "organization": "Example Organization",
            "organisation": "Example Organisation",
        }
        replacement = replacements.get(kind, "Alternate Entity")
        return replacement if _key(replacement) != _key(raw) else f"{replacement} II"

    @staticmethod
    def _corrupted_subtype_predicate(
        relation: str,
        value: str,
        declared_subtypes: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        """Return a same-family predicate with a conflicting semantic subtype.

        Predicate subtype admission is intentionally compiled from the raw
        predicate instead of trusting stored subtype metadata.  A useful fault
        must therefore change the predicate semantics while preserving its
        canonical family and both tuple endpoints.  Facts whose family has no
        modeled sibling subtype are ineligible for this isolated fault.
        """
        from .relation_fact_join_utils import canonical_predicate_contract

        original = canonical_predicate_contract(relation, value)
        original_families = set(original["canonical_families"])
        original_subtypes = set(original["semantic_subtypes"]) | {
            str(item) for item in declared_subtypes if str(item)
        }
        sibling_predicates = {
            "birth_place": "born on",
            "birth_date": "born in",
            "birth_name": "born in",
            "death_place": "died on",
            "death_date": "died in",
            "death_cause": "died in",
            "occupation_role": "worked for",
            "employment_organization": "worked in",
            "employment_location": "worked as",
            "employment_generic": "worked as",
        }
        for subtype in sorted(original_subtypes):
            replacement = sibling_predicates.get(subtype)
            if not replacement:
                continue
            candidate = canonical_predicate_contract(replacement, value)
            candidate_families = set(candidate["canonical_families"])
            candidate_subtypes = set(candidate["semantic_subtypes"])
            if (
                candidate_families == original_families
                and candidate_subtypes
                and candidate_subtypes.isdisjoint(original_subtypes)
            ):
                return {
                    "relation": replacement,
                    "predicate_families": sorted(candidate_families),
                    "predicate_subtypes": sorted(candidate_subtypes),
                    "predicate_contract": candidate,
                }
        return None

    def _perturb_lookup_results(
        self,
        facts: Sequence[Mapping[str, Any]],
        matches: Sequence[Mapping[str, Any]],
        operator: Mapping[str, Any],
        stats: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply deterministic query-time faults to matched index tuples only."""
        facts_by_id = {
            str(fact.get("fact_id") or ""): dict(fact) for fact in facts
        }
        output_facts: List[Dict[str, Any]] = []
        output_matches: List[Dict[str, Any]] = []
        deleted = 0
        corrupted = 0
        deleted_fact_ids: List[str] = []
        corrupted_fact_ids: List[str] = []
        for raw_match in matches:
            match = dict(raw_match)
            fact_id = str(match.get("fact_id") or "")
            fact = dict(facts_by_id.get(fact_id) or {})
            if not fact:
                continue
            if self._perturbation_score("delete", fact_id) < self.index_delete_rate:
                deleted += 1
                deleted_fact_ids.append(fact_id)
                continue
            if self._perturbation_score("corrupt", fact_id) < self.index_corrupt_rate:
                corrupt_mode = self.index_corrupt_mode
                if corrupt_mode == "mixed":
                    mode_score = self._perturbation_score("corrupt_mode", fact_id)
                    corrupt_mode = (
                        "object" if mode_score < (1.0 / 3.0)
                        else "relation" if mode_score < (2.0 / 3.0)
                        else "direction"
                    )
                output_role = str(fact.get("bound_output_role") or "")
                triple = list(fact.get("triple") or [])
                fact["perturbation_original_fact_id"] = fact_id
                fact["perturbation_original_output_surface"] = str(
                    match.get("output_surface") or ""
                )
                fact["perturbation_original_output_key"] = str(
                    match.get("output_key") or ""
                )
                fact["perturbation_original_triple"] = list(triple)
                fact["perturbation_corrupt_mode"] = corrupt_mode
                corruption_applied = True
                if corrupt_mode == "object":
                    output_surface = self._corrupted_surface(
                        str(match.get("output_surface") or ""),
                        str(operator.get("output_type") or "entity"),
                    )
                    output_key = _key(output_surface)
                    triple_index = 0 if output_role == "subject" else 2
                    fact["perturbation_kind"] = "corrupt_object"
                    if len(triple) == 3 and output_role in {"subject", "object"}:
                        triple[triple_index] = output_surface
                    fact["bound_output_key"] = output_key
                    match.update({
                        "output_surface": output_surface,
                        "output_key": output_key,
                    })
                elif corrupt_mode == "relation":
                    fact["perturbation_kind"] = "corrupt_relation"
                    fact["perturbation_original_predicate_families"] = list(
                        fact.get("predicate_families") or []
                    )
                    if len(triple) == 3:
                        triple[1] = "unrelated to"
                    fact["predicate_families"] = ["unrelated"]
                    fact["relation_tokens"] = ["unrelated"]
                    fact["canonical_relation"] = "unrelated_to"
                elif corrupt_mode == "subtype":
                    fact["perturbation_kind"] = "corrupt_subtype"
                    fact["perturbation_original_predicate_families"] = list(
                        fact.get("predicate_families") or []
                    )
                    fact["perturbation_original_predicate_subtypes"] = list(
                        fact.get("predicate_subtypes") or []
                    )
                    subtype_fault = self._corrupted_subtype_predicate(
                        str(triple[1]) if len(triple) == 3 else "",
                        str(triple[2]) if len(triple) == 3 else "",
                        list(fact.get("predicate_subtypes") or []),
                    )
                    if subtype_fault is None:
                        corruption_applied = False
                    else:
                        triple[1] = subtype_fault["relation"]
                        fact["predicate_families"] = subtype_fault[
                            "predicate_families"
                        ]
                        fact["predicate_subtypes"] = subtype_fault[
                            "predicate_subtypes"
                        ]
                        fact["predicate_contract"] = subtype_fault[
                            "predicate_contract"
                        ]
                        fact["relation_tokens"] = sorted(_tokens(triple[1]))
                        fact["canonical_relation"] = "_".join(_tokens(triple[1]))
                elif corrupt_mode == "direction":
                    fact["perturbation_kind"] = "corrupt_direction"
                    if len(triple) == 3:
                        triple[0], triple[2] = triple[2], triple[0]
                    if output_role == "subject":
                        fact["bound_output_role"] = "object"
                    elif output_role == "object":
                        fact["bound_output_role"] = "subject"
                    fact["role_direction_required"] = True
                elif corrupt_mode == "role":
                    fact["perturbation_kind"] = "corrupt_role"
                    fact["perturbation_original_output_role"] = output_role
                    if output_role == "subject":
                        fact["bound_output_role"] = "object"
                    elif output_role == "object":
                        fact["bound_output_role"] = "subject"
                    else:
                        corruption_applied = False
                    fact["role_direction_required"] = True
                else:
                    fact["perturbation_kind"] = "corrupt_provenance"
                    original_source_id = _as_int(
                        fact.get("source_id", fact.get("doc_idx")), -1
                    )
                    fact["perturbation_original_source_id"] = original_source_id
                    doc_titles = self.relation_index.get("doc_title_surfaces") or {}
                    endpoint_tokens = set()
                    if len(triple) == 3:
                        endpoint_tokens = set(_tokens(triple[0])) | set(
                            _tokens(triple[2])
                        )
                    candidate_doc_ids = self._perturbation_doc_ids
                    if len(candidate_doc_ids) > 1:
                        source_score = self._perturbation_score(
                            "provenance_source", fact_id
                        )
                        source_offset = min(
                            len(candidate_doc_ids) - 1,
                            int(source_score * len(candidate_doc_ids)),
                        )
                        corrupted_source_id = -1



                        for delta in range(min(64, len(candidate_doc_ids))):
                            candidate = candidate_doc_ids[
                                (source_offset + delta) % len(candidate_doc_ids)
                            ]
                            if candidate == original_source_id:
                                continue
                            title_tokens = _tokens(str(doc_titles.get(candidate) or ""))
                            if not endpoint_tokens.intersection(title_tokens):
                                corrupted_source_id = candidate
                                break
                        if corrupted_source_id < 0:
                            corrupted_source_id = next(
                                (
                                    candidate for candidate in candidate_doc_ids
                                    if candidate != original_source_id
                                ),
                                original_source_id + 1,
                            )
                    else:



                        corrupted_source_id = original_source_id + 1
                    fact["source_id"] = corrupted_source_id
                    fact["doc_idx"] = corrupted_source_id
                    fact["perturbation_corrupted_source_id"] = (
                        corrupted_source_id
                    )
                if corruption_applied:
                    fact["triple"] = triple
                    fact["fact_id"] = "perturbed:" + hashlib.sha1(
                        (
                            f"{fact_id}\x1e{corrupt_mode}\x1e{triple}\x1e"
                            f"{fact.get('source_id')}\x1e"
                            f"{fact.get('bound_output_role')}\x1e"
                            f"{fact.get('predicate_subtypes')}"
                        ).encode()
                    ).hexdigest()[:20]
                    match.update({
                        "fact_id": fact["fact_id"],
                        "perturbation_kind": fact["perturbation_kind"],
                        "perturbation_corrupt_mode": corrupt_mode,
                    })
                    corrupted += 1
                    corrupted_fact_ids.append(fact_id)
                else:
                    fact = dict(facts_by_id[fact_id])
            output_facts.append(fact)
            output_matches.append(match)
        stats.update({
            "perturbation_input_match_count": len(matches),
            "perturbation_deleted_match_count": deleted,
            "perturbation_corrupted_match_count": corrupted,
            "perturbation_deleted_fact_ids": sorted(set(deleted_fact_ids)),
            "perturbation_corrupted_original_fact_ids": sorted(
                set(corrupted_fact_ids)
            ),
            "perturbation_output_match_count": len(output_matches),
            "perturbation_corrupt_mode": self.index_corrupt_mode,
            "perturbation_seed": self.perturbation_seed,
        })
        return output_facts, output_matches

    def lookup_join(self, binding, operator):
        from .relation_fact_join_utils import (
            _lookup_items_by_exact_entity_key,
            _new_index_access_stats,
        )

        operator = self._effective_schema_operator(operator)
        bound_surface, bound_key = self._bound_input(binding, operator)
        stats = _new_index_access_stats()
        raw_scan_limit = operator.get("execution_scan_limit")
        scan_limit = (
            None if raw_scan_limit is None else max(0, _as_int(raw_scan_limit))
        )
        scoped_rows = self._identity_scoped_rows(binding, operator, stats)
        if self.enable_schema_plan_coverage_v7 and scoped_rows:
            facts, matches, match_diagnostics = self._matches(
                scoped_rows, bound_surface, bound_key, operator, binding
            )
            facts, matches = self._perturb_lookup_results(
                facts, matches, operator, stats
            )
            rejected_fact_ids = set(binding.get("rejected_index_fact_ids") or [])
            if rejected_fact_ids:
                facts = [
                    fact for fact in facts
                    if str(fact.get("fact_id") or "") not in rejected_fact_ids
                ]
                matches = [
                    match for match in matches
                    if str(match.get("fact_id") or "") not in rejected_fact_ids
                ]
                stats["rejected_lineage_fact_filter_count"] = (
                    len(rejected_fact_ids)
                )
            stats["title_terminal_direct_match_count"] = len(matches)
            if matches:
                return {
                    "facts": facts,
                    "matches": matches,
                    "reason": "canonical_title_terminal_direct_match",
                    "cost": {
                        "scan_rows": int(stats.get("input_scan_rows", 0) or 0),
                        "retrieval_calls": 0,
                        "model_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    },
                    "index_access": dict(stats),
                    "identity_diagnostics": match_diagnostics,
                    "scan_complete": not bool(stats.get("budget_truncated_rows")),
                }
        alias_scan_limit = None
        if scan_limit is not None:
            alias_scan_limit = max(
                0, scan_limit - int(stats.get("input_scan_rows", 0) or 0)
            )
        alias_rows = _lookup_items_by_exact_entity_key(
            self.relation_index,
            bound_key,
            max_per_key=self.max_facts_per_key,
            max_fanout=self.max_key_fanout,
            access_stats=stats,
            allow_title_provenance_alias=True,
            max_total_rows=alias_scan_limit,
        )
        rows = []
        seen_item_ids = set()
        for item in (
            list(alias_rows)
            if self.enable_schema_plan_coverage_v7
            else list(scoped_rows) + list(alias_rows)
        ):
            identity = item.get("item_id")
            identity = identity if identity is not None else id(item)
            if identity in seen_item_ids:
                continue
            seen_item_ids.add(identity)
            rows.append(item)
        facts, matches, match_diagnostics = self._matches(
            rows, bound_surface, bound_key, operator, binding
        )
        facts, matches = self._perturb_lookup_results(
            facts, matches, operator, stats
        )
        rejected_fact_ids = set(binding.get("rejected_index_fact_ids") or [])
        if rejected_fact_ids:
            facts = [
                fact for fact in facts
                if str(fact.get("fact_id") or "") not in rejected_fact_ids
            ]
            matches = [
                match for match in matches
                if str(match.get("fact_id") or "") not in rejected_fact_ids
            ]
            stats["rejected_lineage_fact_filter_count"] = len(rejected_fact_ids)
        return {
            "facts": facts,
            "matches": matches,
            "reason": (
                "index_match"
                if matches
                else "index_scan_partial_budget_limited"
                if stats.get("budget_truncated_rows")
                else "index_miss_not_absence"
            ),
            "cost": {
                "scan_rows": int(stats.get("input_scan_rows", 0) or 0),
                "retrieval_calls": 0,
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
            "index_access": dict(stats),
            "identity_diagnostics": match_diagnostics,
            "scan_complete": not bool(stats.get("budget_truncated_rows")),
        }

    def _source_predicate_metadata(self, doc_id: int) -> Dict[str, Any]:
        items = list(self.relation_index.get("items") or [])
        item_ids = list(
            (self.relation_index.get("doc_to_items") or {}).get(int(doc_id))
            or []
        )
        families = set()
        subtypes = set()
        for raw_item_id in item_ids:
            item_id = _as_int(raw_item_id, -1)
            if item_id < 0 or item_id >= len(items):
                continue
            item = items[item_id]
            families.update(str(value) for value in item.get("predicate_families") or [])
            subtypes.update(str(value) for value in item.get("predicate_subtypes") or [])
        return {
            "indexed_fact_count": len(item_ids),
            "indexed_predicate_families": sorted(value for value in families if value),
            "indexed_predicate_subtypes": sorted(value for value in subtypes if value),
        }

    def enumerate_source_candidates(self, binding, operator):
        """Enumerate already-available and title-partition source metadata.

        This method never reads passage text and never performs retrieval.
        Sources already present in the query's initial evidence pool are
        therefore visible to every optimizer arm without being charged as a
        new dense probe.
        """
        from .relation_fact_join_utils import _equivalent_entity_keys

        bound_surface, bound_key = self._bound_input(binding, operator)
        bound_slot = self._bound_slot(binding, operator)
        keys = {str(bound_key or "")}
        if not (bound_slot.get("fact_id") and len(_tokens(bound_surface)) == 1):
            keys.update(_equivalent_entity_keys(
                self.relation_index,
                bound_key,
                allow_title_provenance_alias=True,
            ))
        title_doc_ids = self.relation_index.get("title_entity_doc_ids") or {}
        title_surfaces = self.relation_index.get("title_entity_surfaces") or {}
        category_scope = self._category_partition_scope(binding, operator)
        root_category_doc_ids = sorted(category_scope["partition_doc_ids"])
        category_partition_active = bool(category_scope["active"])
        candidates = []
        seen = set()
        if self.enable_title_catalog_completion:
            source_groups = (
                [(bound_key, root_category_doc_ids)]
                if category_partition_active else
                [
                    (key, list(title_doc_ids.get(key) or []))
                    for key in sorted(
                        keys, key=lambda value: (value != bound_key, value)
                    )
                ]
            )
            for key, source_doc_ids in source_groups:
                for raw_doc_id in source_doc_ids:
                    doc_id = _as_int(raw_doc_id, -1)
                    if doc_id < 0 or doc_id in seen:
                        continue
                    seen.add(doc_id)
                    candidates.append({
                        "source_id": doc_id,
                        "doc_id": doc_id,
                        "source_type": "title_entity_catalog",
                        "completion_source": "title_entity_catalog",
                        "rank": len(candidates),
                        "score": 1.0 if key == bound_key else 0.92,
                        "title": str(
                            (self.relation_index.get("doc_title_surfaces") or {}).get(
                                doc_id
                            )
                            or title_surfaces.get(key)
                            or bound_surface
                        ),
                        "bound_surface": bound_surface,
                        "bound_key": bound_key,
                        "candidate_entity_key": key,
                        "root_category_partition": bool(
                            category_partition_active
                        ),
                        "evidence_token_count": _as_int(
                            (
                                self.relation_index.get(
                                    "doc_evidence_token_counts"
                                ) or {}
                            ).get(doc_id),
                            0,
                        ),
                        **self._source_predicate_metadata(doc_id),
                    })
                    if len(candidates) >= self.completion_top_k:
                        break
                if len(candidates) >= self.completion_top_k:
                    break
        for raw in self.current_source_candidates:
            doc_id = _as_int(raw.get("source_id", raw.get("doc_id")), -1)
            if doc_id < 0 or doc_id in seen:
                continue
            partition_allowed, _ = self._category_partition_source_decision(
                binding, operator, doc_id
            )
            if not partition_allowed:
                continue
            seen.add(doc_id)
            candidates.append({
                **{
                    key: _json_safe(value)
                    for key, value in raw.items()
                    if key != "content"
                },
                "source_id": doc_id,
                "doc_id": doc_id,
                "source_type": "current_evidence_pool",
                "completion_source": "current_evidence_pool",
                "rank": _as_int(raw.get("rank"), len(candidates)),
                "score": _as_float(raw.get("score"), 0.0),
                "bound_surface": bound_surface,
                "bound_key": bound_key,
                "evidence_token_count": (
                    _evidence_token_count(raw.get("content"))
                    or _as_int(
                        (
                            self.relation_index.get(
                                "doc_evidence_token_counts"
                            ) or {}
                        ).get(doc_id),
                        0,
                    )
                ),
                **self._source_predicate_metadata(doc_id),
            })
            if len(candidates) >= 2 * self.completion_top_k:
                break
        return candidates

    def bound_dense_probe(self, binding, operator):
        """Retrieve source candidates but defer extraction to SOURCE_INSPECT."""
        from .relation_fact_join_utils import _bound_completion_query

        bound_surface, _ = self._bound_input(binding, operator)
        query = (
            bound_surface
            if self.completion_query_mode == "entity_only"
            else _bound_completion_query(
                bound_surface, dict(operator), structured=True
            )
        )
        documents = list(self.completion_lookup(query, self.completion_top_k) or [])
        candidates = []
        seen = set()
        for document in documents:
            doc_id = _as_int(document.get("doc_id"), -1)
            if doc_id < 0 or doc_id in seen:
                continue
            partition_allowed, _ = self._category_partition_source_decision(
                binding,
                operator,
                doc_id,
                source_text=str(document.get("content") or ""),
            )
            if not partition_allowed:
                continue
            seen.add(doc_id)
            cached = dict(document)
            cached["doc_id"] = doc_id
            self._source_document_cache[doc_id] = cached
            source_type = str(
                document.get("completion_source") or "dense_completion_probe"
            )
            if source_type == "current_evidence_pool":



                continue
            candidates.append({
                "source_id": doc_id,
                "doc_id": doc_id,
                "source_type": source_type,
                "completion_source": source_type,
                "rank": _as_int(document.get("rank"), len(candidates)),
                "score": _as_float(document.get("score"), 0.0),
                "title": str(document.get("title") or ""),
                "bound_surface": bound_surface,
                "evidence_token_count": _evidence_token_count(
                    document.get("content")
                ),
                **self._source_predicate_metadata(doc_id),
            })
        return {
            "source_candidates": candidates,
            "completion_query": query,
            "completion_query_mode": self.completion_query_mode,
            "bounded_source_doc_ids": [row["doc_id"] for row in candidates],
            "dense_probe_called": True,
            "physical_join_plan": "bswc_bound_dense_candidate_probe",
            "candidate_scan_count": 0,
            "reason": (
                "dense_source_candidates" if candidates
                else "dense_source_candidates_empty"
            ),
            "cost": {
                "scan_rows": 0,
                "retrieval_calls": 1,
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
        }

    def _query_time_materialize(
        self,
        *,
        binding: Mapping[str, Any],
        operator: Mapping[str, Any],
        documents: Sequence[Mapping[str, Any]],
        bound_surface: str,
        bound_key: str,
    ) -> Dict[str, Any]:
        """Materialize source-grounded candidates only after deterministic miss."""

        if not self.enable_query_time_relation_materialization:
            return {
                "attempted": False,
                "facts": [],
                "matches": [],
                "diagnostics": {"disabled": 1},
                "cost": {"model_calls": 0, "input_tokens": 0, "output_tokens": 0},
            }
        if _as_int(operator.get("execution_model_limit"), 0) <= 0:
            return {
                "attempted": False,
                "facts": [],
                "matches": [],
                "diagnostics": {"model_budget_exhausted": 1},
                "cost": {"model_calls": 0, "input_tokens": 0, "output_tokens": 0},
            }
        active_documents = [
            dict(value) for value in documents
            if _as_int(value.get("doc_id"), -1) >= 0
            and str(value.get("content") or "").strip()
        ]
        if not active_documents:
            return {
                "attempted": False,
                "facts": [],
                "matches": [],
                "diagnostics": {"empty_source_pool": 1},
                "cost": {"model_calls": 0, "input_tokens": 0, "output_tokens": 0},
            }
        try:
            result = dict(self.relation_materializer(
                bound_surface=bound_surface,
                bound_key=bound_key,
                operator=dict(operator),
                documents=active_documents,
                binding=dict(binding),
            ) or {})
        except Exception as exc:
            return {
                "attempted": True,
                "facts": [],
                "matches": [],
                "diagnostics": {
                    "materializer_error": 1,
                    "materializer_error_type": exc.__class__.__name__,
                },
                "error": f"{exc.__class__.__name__}: {str(exc)[:240]}",
                "cost": {"model_calls": 1, "input_tokens": None, "output_tokens": None},
            }
        facts, matches, identity_diagnostics = self._matches(
            list(result.get("facts") or []),
            bound_surface,
            bound_key,
            dict(operator),
            binding,
        )
        diagnostics = dict(result.get("diagnostics") or {})
        diagnostics.update({
            "candidate_count": _as_int(result.get("candidate_count")),
            "deterministically_admitted_count": _as_int(result.get("accepted_count")),
            "binding_identity_admitted_count": _as_int(
                result.get("binding_identity_admitted_count")
            ),
            "typed_contract_admitted_count": _as_int(
                result.get("typed_contract_admitted_count")
            ),
            "binding_conditioned_typed_materialization": int(
                self.enable_binding_conditioned_typed_materialization
            ),
            "operator_match_count": len(matches),
        })
        return {
            "attempted": True,
            "facts": facts,
            "matches": matches,
            "validations": list(result.get("validations") or []),
            "diagnostics": diagnostics,
            "identity_diagnostics": identity_diagnostics,
            "cost": dict(result.get("cost") or {
                "model_calls": 1, "input_tokens": None, "output_tokens": None,
            }),
            "cache_hit": bool(result.get("cache_hit")),
        }

    def inspect_source(self, binding, operator, source_candidate):
        """Inspect exactly one selected source for one bound obligation."""
        from collections import Counter
        from .relation_fact_join_utils import _extract_bound_completion_items

        doc_id = _as_int(
            source_candidate.get("source_id", source_candidate.get("doc_id")),
            -1,
        )
        if doc_id < 0:
            return {
                "facts": [],
                "matches": [],
                "reason": "invalid_source_candidate",
                "cost": {"scan_rows": 0, "retrieval_calls": 0, "model_calls": 0},
            }
        cached = dict(self._source_document_cache.get(doc_id) or {})
        content = str(cached.get("content") or "")
        if not content:
            try:
                content = str(self.get_source_content(doc_id) or "")
            except (TypeError, ValueError, KeyError, IndexError):
                content = ""
        title = str(cached.get("title") or source_candidate.get("title") or "").strip()
        if not title and self.get_source_title is not None:
            try:
                title = str(self.get_source_title(doc_id) or "").strip()
            except (TypeError, ValueError, KeyError, IndexError):
                title = ""
        if not title and content:
            title = content.splitlines()[0].strip()
        source_type = str(
            source_candidate.get("source_type")
            or source_candidate.get("completion_source")
            or cached.get("completion_source")
            or "selected_source"
        )
        document = {
            **cached,
            "doc_id": doc_id,
            "rank": _as_int(source_candidate.get("rank"), 0),
            "score": _as_float(source_candidate.get("score"), 0.0),
            "content": content,
            "title": title,
            "completion_source": source_type,
        }
        self._source_document_cache[doc_id] = dict(document)
        already_inspected = doc_id in self._inspected_source_ids
        self._inspected_source_ids.add(doc_id)
        evidence_tokens = (
            0 if already_inspected else _evidence_token_count(content)
        )
        if not content:
            return {
                "facts": [],
                "matches": [],
                "reason": "selected_source_unavailable",
                "selected_source_doc_id": doc_id,
                "selected_source_type": source_type,
                "source_reused": already_inspected,
                "cost": {
                    "scan_rows": 0,
                    "retrieval_calls": 0,
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "evidence_tokens": evidence_tokens,
                    "cache_hit": already_inspected,
                },
            }

        bound_surface, bound_key = self._bound_input(binding, operator)
        partition_allowed, partition_reason = (
            self._category_partition_source_decision(
                binding,
                operator,
                doc_id,
                source_text=content,
            )
        )
        if not partition_allowed:
            return {
                "facts": [],
                "matches": [],
                "reason": "selected_source_identity_conflict",
                "selected_source_doc_id": doc_id,
                "selected_source_type": source_type,
                "source_reused": already_inspected,
                "identity_diagnostics": {
                    "root_category_partition_rejected_count": 1,
                    f"root_category_partition_{partition_reason}_count": 1,
                },
                "cost": {
                    "scan_rows": 0 if already_inspected else 1,
                    "retrieval_calls": 0,
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "evidence_tokens": evidence_tokens,
                    "cache_hit": already_inspected,
                },
            }
        diagnostics = Counter()

        def extract(*, continue_after_rejected_indexed_candidates=False):
            attempt_diagnostics = Counter()
            extracted_items = _extract_bound_completion_items(
                bound_surface=bound_surface,
                bound_key=bound_key,
                operator=dict(operator),
                documents=[document],
                item_id_seed=-3000000 - doc_id * 100,
                diagnostics=attempt_diagnostics,
                relation_index=self.relation_index,
                enable_predicate_gap_repair=True,
                enable_inverse_relation_completion=True,
                enable_title_grounded_completion=True,
                enable_source_grounded_terminal_repair=True,
                enable_source_grounded_terminal_tuple_recovery=(
                    self.enable_source_grounded_terminal_tuple_recovery
                ),
                enable_obligation_bound_repair_certificate=(
                    self.enable_obligation_bound_repair_certificate
                ),
                enable_schema_compiled_source_certificate=(
                    self.enable_schema_compiled_source_certificate
                ),
                continue_sentence_repair_after_indexed_candidates=bool(
                    continue_after_rejected_indexed_candidates
                ),
            )
            attempt = self._matches(
                extracted_items, bound_surface, bound_key, operator, binding
            )
            return (*attempt, attempt_diagnostics)

        continue_after_indexed = bool(
            source_candidate.get(
                "continue_sentence_repair_after_indexed_candidates"
            )
        )
        facts, matches, match_diagnostics, diagnostics = extract(
            continue_after_rejected_indexed_candidates=continue_after_indexed
        )
        if not continue_after_indexed and not matches and facts:
            facts, matches, match_diagnostics, diagnostics = extract(
                continue_after_rejected_indexed_candidates=True
            )
            diagnostics["post_index_sentence_repair_attempted"] += 1
        materialized = {
            "attempted": False,
            "facts": [],
            "matches": [],
            "diagnostics": {},
            "cost": {},
        }
        if not matches:
            materialized = self._query_time_materialize(
                binding=binding,
                operator=operator,
                documents=[document],
                bound_surface=bound_surface,
                bound_key=bound_key,
            )
            if materialized.get("matches"):
                facts = list(materialized.get("facts") or [])
                matches = list(materialized.get("matches") or [])
                match_diagnostics = dict(
                    materialized.get("identity_diagnostics") or {}
                )
        materializer_cost = dict(materialized.get("cost") or {})
        model_calls = (
            _as_int(document.get("model_calls"))
            + _as_int(materializer_cost.get("model_calls"))
        )
        return {
            "facts": facts,
            "matches": matches,
            "reason": "selected_source_match" if matches else "selected_source_miss",
            "selected_source_doc_id": doc_id,
            "selected_source_type": source_type,
            "source_reused": already_inspected,
            "bounded_source_doc_ids": [doc_id],
            "candidate_scan_count": 1,
            "physical_join_plan": "bswc_selected_source_inspection",
            "diagnostics": dict(diagnostics),
            "identity_diagnostics": match_diagnostics,
            "query_time_relation_materialization": {
                "attempted": bool(materialized.get("attempted")),
                "matched": bool(materialized.get("matches")),
                "diagnostics": dict(materialized.get("diagnostics") or {}),
                "validations": list(materialized.get("validations") or []),
                "error": str(materialized.get("error") or ""),
            },
            "cost": {
                "scan_rows": 0 if already_inspected else 1,
                "retrieval_calls": 0,
                "model_calls": model_calls,
                "input_tokens": (
                    _as_int(document.get("input_tokens"))
                    + _as_int(materializer_cost.get("input_tokens"))
                ),
                "output_tokens": (
                    _as_int(document.get("output_tokens"))
                    + _as_int(materializer_cost.get("output_tokens"))
                ),
                "evidence_tokens": evidence_tokens,
                "cache_hit": bool(
                    already_inspected or materialized.get("cache_hit")
                ),
            },
        }

    def _title_catalog_documents(
        self,
        binding: Mapping[str, Any],
        operator: Mapping[str, Any],
        bound_surface: str,
        bound_key: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not self.enable_title_catalog_completion or limit <= 0:
            return []
        from .relation_fact_join_utils import _equivalent_entity_keys

        bound_slot = self._bound_slot(binding, operator)
        keys = {str(bound_key or "")}
        if not (bound_slot.get("fact_id") and len(_tokens(bound_surface)) == 1):
            keys.update(_equivalent_entity_keys(
                self.relation_index,
                bound_key,
                allow_title_provenance_alias=True,
            ))
        title_doc_ids = self.relation_index.get("title_entity_doc_ids") or {}
        category_scope = self._category_partition_scope(binding, operator)
        source_groups = (
            [(bound_key, sorted(category_scope["partition_doc_ids"]))]
            if category_scope["active"] else
            [
                (key, list(title_doc_ids.get(key) or []))
                for key in sorted(keys, key=lambda value: (value != bound_key, value))
            ]
        )
        documents: List[Dict[str, Any]] = []
        seen = set()
        for key, source_doc_ids in source_groups:
            for raw_doc_id in source_doc_ids:
                try:
                    doc_id = int(raw_doc_id)
                    content = str(self.get_source_content(doc_id) or "")
                except (TypeError, ValueError, KeyError, IndexError):
                    continue
                if doc_id in seen or not content:
                    continue
                seen.add(doc_id)
                title = ""
                if self.get_source_title is not None:
                    try:
                        title = str(self.get_source_title(doc_id) or "").strip()
                    except (TypeError, ValueError, KeyError, IndexError):
                        title = ""
                if not title:
                    title = content.splitlines()[0].strip()
                documents.append({
                    "doc_id": doc_id,
                    "rank": len(documents),
                    "score": 1.0,
                    "content": content,
                    "title": title,
                    "completion_source": "title_entity_catalog",
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                })
                if len(documents) >= limit:
                    return documents
        return documents

    def complete(self, binding, operator):
        from collections import Counter
        from .relation_fact_join_utils import (
            _bound_completion_query,
            _extract_bound_completion_items,
        )

        operator = self._effective_schema_operator(operator)
        bound_surface, bound_key = self._bound_input(binding, operator)
        raw_scan_limit = operator.get("execution_scan_limit")
        scan_limit = (
            2 * self.completion_top_k
            if raw_scan_limit is None
            else max(0, _as_int(raw_scan_limit))
        )
        retrieval_limit = max(
            0,
            _as_int(operator.get("execution_retrieval_limit"), 1),
        )
        query = (
            bound_surface
            if self.completion_query_mode == "entity_only"
            else _bound_completion_query(
                bound_surface, dict(operator), structured=True
            )
        )





        prefetched_external_rows: List[Dict[str, Any]] = []
        prefetched_external_reject_reasons = Counter()
        if (
            self.obligation_source_lookup is not None
            and retrieval_limit > 0
            and scan_limit > 0
        ):
            raw_external_rows = list(self.obligation_source_lookup(
                bound_surface,
                dict(operator),
                1,
            ) or [])
            bound_slot = self._bound_slot(binding, operator)
            root_qualifier_tokens = {
                token
                for qualifier in (
                    (binding.get("root_metadata") or {}).get(
                        "canonical_title_family_qualifiers"
                    ) or []
                )
                for token in _tokens(qualifier)
            } - {
                "album", "book", "film", "movie", "novel", "song", "work"
            }
            lineage_consistent = bool(
                not bound_slot.get("fact_id")
                or not root_qualifier_tokens
                or root_qualifier_tokens & set(_tokens(bound_surface))
            )
            for raw in raw_external_rows:
                doc_id = _as_int(raw.get("doc_id", raw.get("source_id")), -1)
                partition_allowed, partition_reason = (
                    self._category_partition_source_decision(
                        binding,
                        operator,
                        doc_id,
                        source_text=str(raw.get("content") or ""),
                    )
                )
                if not partition_allowed:
                    prefetched_external_reject_reasons[
                        f"external_{partition_reason}_rejected"
                    ] += 1
                    continue
                if not lineage_consistent:
                    prefetched_external_reject_reasons[
                        "external_root_qualifier_lineage_mismatch_rejected"
                    ] += 1
                    continue
                prefetched_external_rows.append(dict(raw))
        external_reserved_scan_rows = min(
            scan_limit,
            2 if prefetched_external_rows else 0,
        )
        catalog_documents = self._title_catalog_documents(
            binding,
            operator,
            bound_surface,
            bound_key,
            min(
                self.completion_top_k,
                max(0, scan_limit - external_reserved_scan_rows),
            ),
        )
        documents = list(catalog_documents)
        diagnostics = Counter(prefetched_external_reject_reasons)

        def extract(
            active_documents,
            *,
            continue_after_rejected_indexed_candidates=False,
        ):
            attempt_diagnostics = Counter()
            extracted_items = _extract_bound_completion_items(
                bound_surface=bound_surface,
                bound_key=bound_key,
                operator=dict(operator),
                documents=[dict(value) for value in active_documents],
                item_id_seed=-2000000,
                diagnostics=attempt_diagnostics,
                relation_index=self.relation_index,
                enable_predicate_gap_repair=True,
                enable_inverse_relation_completion=True,
                enable_title_grounded_completion=True,
                enable_source_grounded_terminal_repair=True,
                enable_source_grounded_terminal_tuple_recovery=(
                    self.enable_source_grounded_terminal_tuple_recovery
                ),
                enable_obligation_bound_repair_certificate=(
                    self.enable_obligation_bound_repair_certificate
                ),
                enable_schema_compiled_source_certificate=(
                    self.enable_schema_compiled_source_certificate
                ),
                continue_sentence_repair_after_indexed_candidates=bool(
                    continue_after_rejected_indexed_candidates
                ),
            )
            attempt = self._matches(
                extracted_items, bound_surface, bound_key, operator, binding
            )
            return (*attempt, attempt_diagnostics)





        facts, matches, match_diagnostics, diagnostics = extract(documents)
        diagnostics.update(prefetched_external_reject_reasons)
        if not matches and facts:
            facts, matches, match_diagnostics, diagnostics = extract(
                documents,
                continue_after_rejected_indexed_candidates=True,
            )
            diagnostics.update(prefetched_external_reject_reasons)
            diagnostics["post_index_sentence_repair_attempted"] += 1
        title_probe_match_count = len(matches)
        fact_by_id = {
            str(value.get("fact_id") or ""): value for value in facts
        }
        verification_capacity = max(
            0,
            scan_limit - len(documents) - external_reserved_scan_rows,
        )
        title_matches_to_verify = list(matches[:verification_capacity])
        verified_match_ids = set()
        title_probe_verified_match_count = 0
        for match in title_matches_to_verify:
            fact_id = str(match.get("fact_id") or "")
            verified_match_ids.add(fact_id)
            if self.verify(fact_by_id.get(fact_id, {})).get(
                "support_status"
            ) == SUPPORTED:
                title_probe_verified_match_count += 1
        internal_verification_calls = len(title_matches_to_verify)






        remaining_verification_capacity = max(
            0, verification_capacity - internal_verification_calls
        )
        verified_category_scope = self._category_partition_scope(binding, operator)
        if (
            matches
            and title_probe_verified_match_count == 0
            and remaining_verification_capacity > 0
            and documents
            and verified_category_scope["active"]
        ):
            repaired_facts, repaired_matches, repaired_identity, repaired_diag = (
                extract(
                    documents,
                    continue_after_rejected_indexed_candidates=True,
                )
            )
            repaired_fact_by_id = {
                str(value.get("fact_id") or ""): value
                for value in repaired_facts
            }
            new_matches = [
                match for match in repaired_matches
                if str(match.get("fact_id") or "") not in verified_match_ids
            ][:remaining_verification_capacity]
            for match in new_matches:
                fact_id = str(match.get("fact_id") or "")
                verified_match_ids.add(fact_id)
                if self.verify(repaired_fact_by_id.get(fact_id, {})).get(
                    "support_status"
                ) == SUPPORTED:
                    title_probe_verified_match_count += 1
            internal_verification_calls += len(new_matches)
            facts = repaired_facts
            matches = repaired_matches
            fact_by_id = repaired_fact_by_id
            match_diagnostics = repaired_identity
            diagnostics.update(repaired_diag)
            diagnostics["post_verify_sentence_repair_attempted"] += 1
            diagnostics["post_verify_sentence_repair_candidate_count"] += len(
                new_matches
            )
            diagnostics["post_verify_sentence_repair_supported_count"] += (
                title_probe_verified_match_count
            )






        external_source_catalog_called = False
        external_source_documents: List[Dict[str, Any]] = []
        external_source_supported_match_count = 0
        external_scan_capacity = max(
            0, scan_limit - len(documents) - internal_verification_calls
        )
        if (
            title_probe_verified_match_count == 0
            and self.obligation_source_lookup is not None
            and retrieval_limit > 0
            and external_scan_capacity > 0
        ):
            external_rows = prefetched_external_rows[:external_scan_capacity]
            seen_doc_ids = {value.get("doc_id") for value in documents}
            for raw in external_rows:
                doc_id = _as_int(raw.get("doc_id", raw.get("source_id")), -1)
                content = str(raw.get("content") or "")
                title = str(raw.get("title") or "").strip()
                if doc_id < 0 or doc_id in seen_doc_ids or not content or not title:
                    continue
                partition_allowed, partition_reason = (
                    self._category_partition_source_decision(
                        binding,
                        operator,
                        doc_id,
                        source_text=content,
                    )
                )
                if not partition_allowed:
                    diagnostics[
                        f"external_{partition_reason}_rejected"
                    ] += 1
                    continue
                document = {
                    **dict(raw),
                    "doc_id": doc_id,
                    "source_id": doc_id,
                    "rank": len(external_source_documents),
                    "score": _as_float(raw.get("score"), 1.0),
                    "content": content,
                    "title": title,
                    "completion_source": "frozen_external_title_catalog",
                }
                seen_doc_ids.add(doc_id)
                documents.append(document)
                external_source_documents.append(document)
                self._source_document_cache[doc_id] = dict(document)
                if len(external_source_documents) >= external_scan_capacity:
                    break
            external_source_catalog_called = bool(external_source_documents)
            if external_source_catalog_called:
                facts, matches, match_diagnostics, external_diagnostics = extract(
                    documents,
                    continue_after_rejected_indexed_candidates=True,
                )
                diagnostics.update(external_diagnostics)
                fact_by_id = {
                    str(value.get("fact_id") or ""): value for value in facts
                }
                remaining_verification_capacity = max(
                    0,
                    scan_limit
                    - len(documents)
                    - internal_verification_calls,
                )
                new_matches = [
                    match for match in matches
                    if str(match.get("fact_id") or "") not in verified_match_ids
                ][:remaining_verification_capacity]
                for match in new_matches:
                    fact_id = str(match.get("fact_id") or "")
                    verified_match_ids.add(fact_id)
                    if self.verify(fact_by_id.get(fact_id, {})).get(
                        "support_status"
                    ) == SUPPORTED:
                        external_source_supported_match_count += 1
                internal_verification_calls += len(new_matches)
                diagnostics["external_source_catalog_documents"] += len(
                    external_source_documents
                )
                diagnostics["external_source_catalog_candidates"] += len(
                    new_matches
                )
                diagnostics["external_source_catalog_supported"] += (
                    external_source_supported_match_count
                )
        verification_truncated = len(matches) > internal_verification_calls
        dense_probe_requested = bool(
            self.completion_mode == "eager"
            or (
                title_probe_verified_match_count == 0
                and external_source_supported_match_count == 0
            )
        )
        scan_remaining = max(
            0, scan_limit - len(documents) - internal_verification_calls
        )
        dense_probe_called = bool(
            dense_probe_requested
            and retrieval_limit > int(external_source_catalog_called)
            and scan_remaining > 0
        )
        budget_limited_partial = bool(
            scan_limit < 2 * self.completion_top_k
            or verification_truncated
            or (dense_probe_requested and not dense_probe_called)
        )
        physical_join_plan = (
            "eager_title_plus_dense_reextract"
            if self.completion_mode == "eager"
            else "title_then_external_bound_semijoin"
            if external_source_catalog_called and not dense_probe_called
            else "title_external_then_dense_bound_semijoin"
            if external_source_catalog_called
            else "title_catalog_index_nested_loop"
            if not dense_probe_called
            else "title_then_dense_bound_semijoin"
        )
        if dense_probe_called:



            diagnostics = Counter({
                key: value for key, value in diagnostics.items()
                if str(key).startswith("external_")
            })
            seen_doc_ids = {value.get("doc_id") for value in documents}
            retrieved = list(self.completion_lookup(
                query,
                min(self.completion_top_k, scan_remaining),
            ) or [])
            for document in retrieved:
                doc_id = document.get("doc_id")
                if doc_id in seen_doc_ids:
                    continue
                partition_allowed, partition_reason = (
                    self._category_partition_source_decision(
                        binding,
                        operator,
                        _as_int(doc_id, -1),
                        source_text=str(document.get("content") or ""),
                    )
                )
                if not partition_allowed:
                    diagnostics[
                        f"dense_{partition_reason}_rejected"
                    ] += 1
                    continue
                seen_doc_ids.add(doc_id)
                documents.append(dict(document))
                if len(documents) >= scan_limit:
                    break
            facts, matches, match_diagnostics, dense_diagnostics = extract(
                documents,
                continue_after_rejected_indexed_candidates=bool(
                    verified_category_scope["active"]
                ),
            )
            diagnostics.update(dense_diagnostics)
            if verified_category_scope["active"]:
                diagnostics["dense_sentence_repair_enabled"] += 1
            elif not matches and facts:
                facts, matches, match_diagnostics, dense_diagnostics = extract(
                    documents,
                    continue_after_rejected_indexed_candidates=True,
                )
                diagnostics.update(dense_diagnostics)
                diagnostics["post_index_sentence_repair_attempted"] += 1
        materialized = {
            "attempted": False,
            "facts": [],
            "matches": [],
            "diagnostics": {},
            "cost": {},
        }
        if not matches:
            materialized = self._query_time_materialize(
                binding=binding,
                operator=operator,
                documents=documents,
                bound_surface=bound_surface,
                bound_key=bound_key,
            )
            if materialized.get("matches"):
                facts = list(materialized.get("facts") or [])
                matches = list(materialized.get("matches") or [])
                match_diagnostics = dict(
                    materialized.get("identity_diagnostics") or {}
                )
                physical_join_plan += "+query_time_relation_materialization"
        materializer_cost = dict(materialized.get("cost") or {})
        model_calls = (
            sum(_as_int(value.get("model_calls")) for value in documents)
            + _as_int(materializer_cost.get("model_calls"))
        )
        input_tokens = [value.get("input_tokens") for value in documents]
        output_tokens = [value.get("output_tokens") for value in documents]
        input_tokens.append(materializer_cost.get("input_tokens", 0))
        output_tokens.append(materializer_cost.get("output_tokens", 0))
        return {
            "facts": facts,
            "matches": matches,
            "reason": (
                "completion_match"
                if matches
                else "completion_partial_budget_limited"
                if budget_limited_partial
                else "bounded_source_scope_exhausted"
            ),
            "completion_query": query,
            "completion_query_mode": self.completion_query_mode,
            "bounded_source_doc_ids": [value.get("doc_id") for value in documents],
            "title_catalog_doc_ids": [
                value.get("doc_id") for value in catalog_documents
            ],
            "title_catalog_hit": bool(catalog_documents),
            "external_source_catalog_called": external_source_catalog_called,
            "external_source_reserved_scan_rows": external_reserved_scan_rows,
            "external_source_doc_ids": [
                value.get("doc_id") for value in external_source_documents
            ],
            "external_source_supported_match_count": (
                external_source_supported_match_count
            ),
            "external_source_provenance": [
                {
                    "doc_id": value.get("doc_id"),
                    "title": value.get("title"),
                    "source_url": value.get("source_url"),
                    "revision_id": value.get("revision_id"),
                    "content_sha256": value.get("content_sha256"),
                    "catalog_version": value.get("catalog_version"),
                }
                for value in external_source_documents
            ],
            "dense_probe_called": dense_probe_called,
            "dense_probe_requested": dense_probe_requested,
            "scan_complete": not budget_limited_partial,
            "budget_limited_partial_scan": budget_limited_partial,
            "physical_join_plan": physical_join_plan,
            "candidate_scan_count": len(documents),
            "title_probe_match_count": title_probe_match_count,
            "title_probe_verified_match_count": title_probe_verified_match_count,
            "internal_verification_calls": internal_verification_calls,
            "internal_verification_scan_rows": internal_verification_calls,
            "internal_verification_truncated": verification_truncated,
            "online_cardinality": {
                "title_partition_rows": len(catalog_documents),
                "candidate_rows": len(documents),
                "output_rows": len(matches),
            },
            "diagnostics": dict(diagnostics),
            "identity_diagnostics": match_diagnostics,
            "query_time_relation_materialization": {
                "attempted": bool(materialized.get("attempted")),
                "matched": bool(materialized.get("matches")),
                "diagnostics": dict(materialized.get("diagnostics") or {}),
                "validations": list(materialized.get("validations") or []),
                "error": str(materialized.get("error") or ""),
            },
            "cost": {
                "scan_rows": len(documents) + internal_verification_calls,
                "retrieval_calls": (
                    int(external_source_catalog_called) + int(dense_probe_called)
                ),
                "model_calls": model_calls,
                "input_tokens": (
                    None if model_calls and any(value is None for value in input_tokens)
                    else sum(_as_int(value) for value in input_tokens)
                ),
                "output_tokens": (
                    None if model_calls and any(value is None for value in output_tokens)
                    else sum(_as_int(value) for value in output_tokens)
                ),
            },
        }

    def _verify_obligation_bound_certificate(
        self,
        fact: Mapping[str, Any],
        record: Mapping[str, Any],
        source_text: str,
        source_title: str,
        source_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Revalidate an exact source-span certificate before admission."""
        if (
            not self.enable_obligation_bound_repair_certificate
            or record.get("version") != "obligation_bound_repair_certificate_v1"
        ):
            return None
        try:
            start = int(record.get("source_span_start", -1))
            end = int(record.get("source_span_end", -1))
            record_source_id = int(record.get("source_document_id", -1))
        except (TypeError, ValueError):
            return None
        sentence = str(record.get("source_sentence") or "")
        source_subject = str(record.get("source_subject_surface") or "").strip()
        source_relation = str(record.get("source_relation_surface") or "").strip()
        source_object = str(record.get("source_object_surface") or "").strip()
        if (
            record_source_id != source_id
            or start < 0
            or end <= start
            or end > len(source_text)
            or source_text[start:end] != sentence
            or not source_subject
            or not source_relation
            or not source_object
        ):
            return None
        if str(record.get("source_title") or "").strip() != source_title:
            return None
        if str(record.get("certified_tuple_key") or "") != str(
            fact.get("tuple_key") or ""
        ):
            return None
        obligation_id = str(record.get("obligation_id") or "")
        fact_operator_id = str(fact.get("operator_id") or "")
        if fact_operator_id and obligation_id != fact_operator_id:
            return None
        sentence_key = sentence.casefold()
        relation_at = sentence_key.find(source_relation.casefold())
        object_at = sentence_key.find(source_object.casefold())
        subject_at = sentence_key.find(source_subject.casefold())
        title_scoped = str(record.get("source_subject_scope") or "") == (
            "exact_document_title"
        )
        if relation_at < 0 or object_at < relation_at:
            return None
        if subject_at < 0 and not (title_scoped and _key(source_subject) == _key(source_title)):
            return None
        if subject_at >= 0 and subject_at > relation_at:
            return None
        prefix_tokens = _tokens(sentence[:relation_at])[-3:]
        if set(prefix_tokens) & {"neither", "never", "no", "not"}:
            return None
        return {
            "support_status": SUPPORTED,
            "support_reason": "obligation_bound_exact_source_certificate",
            "source_sentence_id": _as_int(record.get("source_sentence_id"), -1),
            "source_sentence_span": sentence[:480],
            "source_subject_surface": source_subject,
            "source_object_surface": source_object,
            "verifier_version": "obligation_bound_repair_certificate_v1",
            "guarantee_scope": (
                "typed_obligation+exact_source_span+predicate_direction+title_identity"
            ),
            "provenance_scope": (
                "exact_title_bound_sentence"
                if title_scoped else "exact_same_sentence"
            ),
            "recovery_contract_version": str(record.get("version") or ""),
            "verified_source_triple": [
                source_subject, source_relation, source_object
            ],
        }

    def _verify_schema_compiled_source_certificate(
        self,
        fact: Mapping[str, Any],
        record: Mapping[str, Any],
        source_text: str,
        source_title: str,
        source_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Re-evaluate a direction- and type-aware schema source proof."""
        from .relation_fact_join_utils import canonical_entity_key

        if (
            not self.enable_schema_compiled_source_certificate
            or record.get("version") != "schema_compiled_source_certificate_v2"
        ):
            return None
        try:
            start = int(record.get("source_span_start", -1))
            end = int(record.get("source_span_end", -1))
            record_source_id = int(record.get("source_document_id", -1))
        except (TypeError, ValueError):
            return None
        sentence = str(record.get("source_sentence") or "")
        source_subject = str(record.get("source_subject_surface") or "").strip()
        source_relation = str(record.get("source_relation_surface") or "").strip()
        source_object = str(record.get("source_object_surface") or "").strip()
        if (
            record_source_id != source_id
            or start < 0
            or end <= start
            or end > len(source_text)
            or source_text[start:end] != sentence
            or str(record.get("source_title") or "").strip() != source_title
            or not source_subject
            or not source_object
        ):
            return None
        if str(record.get("certified_tuple_key") or "") != str(
            fact.get("tuple_key") or ""
        ):
            return None
        obligation_id = str(record.get("obligation_id") or "")
        fact_operator_id = str(fact.get("operator_id") or "")
        if fact_operator_id and obligation_id != fact_operator_id:
            return None

        contract = dict(fact.get("obligation_contract") or {})
        accepted_input_keys = {
            str(value) for value in contract.get("accepted_input_keys") or []
            if str(value)
        }
        bound_input_key = str(record.get("bound_input_key") or "")
        bound_output_key = str(record.get("bound_output_key") or "")
        contract_output_key = str(contract.get("bound_output_key") or "")
        if (
            not bound_input_key
            or (accepted_input_keys and bound_input_key not in accepted_input_keys)
            or not bound_output_key
            or (contract_output_key and bound_output_key != contract_output_key)
            or bound_output_key != canonical_entity_key(
                (record.get("typed_object") or {}).get("surface")
            )
        ):
            return None
        contract_families = {
            str(value) for value in contract.get("predicate_families") or []
        }
        record_families = {
            str(value) for value in record.get("predicate_families") or []
        }
        if not contract_families or record_families != contract_families:
            return None

        record_direction = str(record.get("relation_direction") or "")
        expected_input_side = str(contract.get("bound_input_side") or "")
        expected_output_side = str(contract.get("bound_output_role") or "")
        if record_direction != f"{expected_input_side}_to_{expected_output_side}":
            return None

        title_scoped = str(record.get("source_subject_scope") or "") == (
            "exact_document_title"
        )
        bound_input_surface = str(record.get("bound_input_surface") or "")
        sentence_key = sentence.casefold()
        subject_at = sentence_key.find(source_subject.casefold())
        object_at = sentence_key.find(source_object.casefold())
        if object_at < 0:
            return None
        if subject_at < 0 and not (
            title_scoped
            and canonical_entity_key(bound_input_surface)
            == canonical_entity_key(source_title)
        ):
            return None

        evidence_mode = str(record.get("schema_evidence_mode") or "")
        if evidence_mode.startswith("typed_biographical_"):
            family = (
                "birth" if "birth" in evidence_mode
                else "death" if "death" in evidence_mode else ""
            )
            if (
                family not in contract_families
                or str(contract.get("output_type") or "").lower() != "date"
            ):
                return None
            date_pattern = re.compile(
                r"(?<!\w)(?:"
                r"(?:January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s+[0-3]?\d,?\s+"
                r"[1-9]\d{2,3}|[0-3]?\d\s+(?:January|February|March|"
                r"April|May|June|July|August|September|October|November|"
                r"December)\s+[1-9]\d{2,3}|[1-9]\d{2,3})(?!\w)",
                flags=re.IGNORECASE,
            )
            endpoint_valid = False
            for parenthetical in re.findall(r"\(([^)]{3,160})\)", sentence):
                dates = [
                    match.group(0).strip()
                    for match in date_pattern.finditer(parenthetical)
                ]
                if len(dates) < 2:
                    continue
                endpoint = dates[0] if family == "birth" else dates[-1]
                if canonical_entity_key(endpoint) == bound_output_key:
                    endpoint_valid = True
                    break
            if not endpoint_valid:
                return None
        else:
            relation_at = sentence_key.find(source_relation.casefold())
            if not source_relation or relation_at < 0 or object_at < relation_at:
                return None
            if subject_at >= 0 and subject_at > relation_at:
                return None
            lexical_prefix = sentence[:relation_at]
            if source_subject:
                lexical_prefix = re.sub(
                    re.escape(source_subject),
                    " ",
                    lexical_prefix,
                    flags=re.IGNORECASE,
                )
            relation_window = " ".join([
                lexical_prefix[-48:], sentence[relation_at:object_at]
            ])
            if set(_tokens(relation_window)) & {"neither", "never", "no", "not"}:
                return None

        return {
            "support_status": SUPPORTED,
            "support_reason": "schema_compiled_exact_source_certificate",
            "source_sentence_id": _as_int(record.get("source_sentence_id"), -1),
            "source_sentence_span": sentence[:480],
            "source_subject_surface": source_subject,
            "source_object_surface": source_object,
            "verifier_version": "schema_compiled_source_certificate_v2",
            "guarantee_scope": (
                "typed_obligation+direction+exact_source_span+bound_identity"
            ),
            "provenance_scope": (
                "exact_title_bound_sentence" if title_scoped else "exact_same_sentence"
            ),
            "recovery_contract_version": str(record.get("version") or ""),
            "schema_evidence_mode": evidence_mode,
            "verified_source_triple": [
                source_subject, source_relation, source_object
            ],
        }

    def _verify_source_grounded_recovery_certificate(
        self,
        fact: Mapping[str, Any],
        record: Mapping[str, Any],
        source_text: str,
        source_title: str,
        source_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Validate the lexical evidence view without rewriting its tuple.

        COMPLETE materializes a canonical tuple for query execution and keeps
        the exact sentence-level wording in a recovery record.  This method
        proves only source support.  Predicate, direction, role, binding and
        output type remain the responsibility of obligation admission over
        the canonical tuple.
        """
        from .relation_fact_join_utils import canonical_entity_key

        if (
            not self.enable_source_grounded_terminal_tuple_recovery
            or record.get("version")
            != "source_grounded_terminal_tuple_recovery_v1"
        ):
            return None
        try:
            start = int(record.get("source_span_start", -1))
            end = int(record.get("source_span_end", -1))
            record_source_id = int(record.get("source_document_id", -1))
        except (TypeError, ValueError):
            return None
        sentence = str(record.get("source_sentence") or "")
        source_subject = str(
            record.get("source_subject_surface") or ""
        ).strip()
        source_relation = str(
            record.get("source_relation_surface") or ""
        ).strip()
        source_object = str(
            record.get("source_object_surface") or ""
        ).strip()
        if (
            record_source_id != source_id
            or start < 0
            or end <= start
            or end > len(source_text)
            or source_text[start:end] != sentence
            or str(record.get("source_title") or "").strip() != source_title
            or not source_subject
            or not source_relation
            or not source_object
        ):
            return None

        obligation_id = str(record.get("obligation_id") or "")
        fact_operator_id = str(fact.get("operator_id") or "")
        if fact_operator_id and obligation_id != fact_operator_id:
            return None

        contract = dict(fact.get("obligation_contract") or {})
        accepted_input_keys = {
            str(value)
            for value in contract.get("accepted_input_keys") or []
            if str(value)
        }
        canonical_subject_key = str(
            record.get("canonical_subject_key") or ""
        )
        if (
            accepted_input_keys
            and canonical_subject_key not in accepted_input_keys
        ):
            return None

        typed_object = dict(record.get("typed_object") or {})
        typed_object_surface = str(typed_object.get("surface") or "").strip()
        typed_object_key = str(typed_object.get("key") or "")
        bound_output_key = str(fact.get("bound_output_key") or "")
        contract_output_key = str(contract.get("bound_output_key") or "")
        if (
            not typed_object_surface
            or not typed_object_key
            or typed_object_key != canonical_entity_key(typed_object_surface)
            or (bound_output_key and typed_object_key != bound_output_key)
            or (contract_output_key and typed_object_key != contract_output_key)
        ):
            return None

        expected_output_role = str(fact.get("bound_output_role") or "")
        record_direction = str(record.get("relation_direction") or "")
        if (
            record_direction == "subject_to_object"
            and expected_output_role != "object"
        ) or (
            record_direction == "object_to_subject"
            and expected_output_role != "subject"
        ):
            return None

        sentence_key = sentence.casefold()
        relation_at = sentence_key.find(source_relation.casefold())
        object_at = sentence_key.find(source_object.casefold())
        subject_at = sentence_key.find(source_subject.casefold())
        if relation_at < 0 or object_at < relation_at:
            return None
        identity_mode = str(record.get("instance_identity_mode") or "")
        title_scoped = identity_mode.startswith("title_")
        if subject_at < 0 and not (
            title_scoped
            and canonical_entity_key(record.get("canonical_subject"))
            == canonical_entity_key(source_title)
        ):
            return None
        if subject_at >= 0 and subject_at > relation_at:
            return None




        lexical_prefix = sentence[:relation_at]
        if source_subject:
            lexical_prefix = re.sub(
                re.escape(source_subject),
                " ",
                lexical_prefix,
                flags=re.IGNORECASE,
            )
        relation_window = " ".join([
            lexical_prefix[-48:],
            sentence[relation_at:object_at],
        ])
        if set(_tokens(relation_window)) & {"neither", "never", "no", "not"}:
            return None

        return {
            "support_status": SUPPORTED,
            "support_reason": "exact_lexical_source_schema_contract",
            "source_sentence_id": _as_int(
                record.get("source_sentence_id"), -1
            ),
            "source_sentence_span": sentence[:480],
            "source_subject_surface": source_subject,
            "source_object_surface": source_object,
            "verifier_version": "executable_schema_source_contract_v1",
            "guarantee_scope": "exact_source_span_for_canonical_join_tuple",
            "provenance_scope": (
                "title_bound_exact_sentence"
                if title_scoped else "exact_same_sentence"
            ),
            "recovery_contract_version": str(record.get("version") or ""),
            "verified_source_triple": [
                source_subject, source_relation, source_object
            ],
        }

    def _verify_binding_conditioned_typed_materialization_certificate(
        self,
        fact: Mapping[str, Any],
        record: Mapping[str, Any],
        source_text: str,
        source_title: str,
        source_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Replay the exact-span typed compilation used by online QTRM.

        This certificate is intentionally narrower than a generic LLM claim.
        The verifier independently checks the immutable source span, current
        operator contract, bound input key, output endpoint, identity mode and
        lexical-to-canonical compilation before source support can become
        SUPPORTED.  Obligation admission still runs afterwards.
        """

        from .query_time_relation_materialization_utils import (
            BINDING_CONDITIONED_TYPED_MATERIALIZATION_VERSION,
            _source_title_expanded_alias_identity,
            compile_typed_candidate_contract,
            source_surface_in_text,
        )
        from .relation_fact_join_utils import canonical_entity_key

        if (
            not self.enable_binding_conditioned_typed_materialization
            or record.get("version")
            != BINDING_CONDITIONED_TYPED_MATERIALIZATION_VERSION
        ):
            return None
        try:
            start = int(record.get("source_span_start", -1))
            end = int(record.get("source_span_end", -1))
            record_source_id = int(record.get("source_document_id", -1))
        except (TypeError, ValueError):
            return None
        sentence = str(record.get("source_sentence") or "")
        physical_subject = str(
            record.get("physical_subject_surface")
            or record.get("source_subject_surface")
            or ""
        ).strip()
        physical_object = str(
            record.get("physical_object_surface")
            or record.get("source_object_surface")
            or ""
        ).strip()
        lexical_relation = str(record.get("source_relation_surface") or "").strip()
        bound_surface = str(record.get("canonical_subject") or "").strip()
        bound_key = str(record.get("canonical_subject_key") or "")
        lexical_bound_surface = str(
            record.get("source_lexical_bound_surface") or bound_surface
        ).strip()
        lexical_answer_surface = str(
            record.get("source_lexical_answer_surface")
            or (record.get("typed_object") or {}).get("surface")
            or ""
        ).strip()
        if (
            record_source_id != source_id
            or start < 0
            or end <= start
            or end > len(source_text)
            or source_text[start:end] != sentence
            or str(record.get("source_title") or "").strip() != source_title
            or not physical_subject
            or not physical_object
            or not lexical_relation
            or not lexical_answer_surface
            or not bound_surface
            or not bound_key
        ):
            return None
        if str(record.get("certified_tuple_key") or "") != str(
            fact.get("tuple_key") or ""
        ):
            return None
        if str(record.get("obligation_id") or "") != str(
            fact.get("operator_id") or ""
        ):
            return None

        contract = dict(fact.get("obligation_contract") or {})
        accepted_input_keys = {
            str(value) for value in contract.get("accepted_input_keys") or []
            if str(value)
        }
        if accepted_input_keys and bound_key not in accepted_input_keys:
            return None
        triple = list(fact.get("triple") or [])
        if len(triple) != 3:
            return None
        if (
            canonical_entity_key(triple[0])
            != canonical_entity_key(physical_subject)
            or canonical_entity_key(triple[2])
            != canonical_entity_key(physical_object)
        ):
            return None
        output_role = str(
            fact.get("bound_output_role")
            or contract.get("bound_output_role")
            or ""
        )
        output_index = 0 if output_role == "subject" else 2 if output_role == "object" else -1
        candidate_output = (
            physical_subject if output_role == "subject" else physical_object
            if output_role == "object" else ""
        )
        candidate_key = canonical_entity_key(candidate_output)
        output_key = str(
            fact.get("bound_output_key")
            or contract.get("bound_output_key")
            or ""
        )
        if (
            output_index < 0
            or not candidate_key
            or (output_key and candidate_key != output_key)
            or canonical_entity_key(triple[output_index]) != candidate_key
        ):
            return None

        identity_reason = str(record.get("binding_identity_reason") or "")
        if identity_reason == "candidate_source_title_exact_identity":
            if bound_key != canonical_entity_key(source_title):
                return None
        elif identity_reason == "candidate_source_title_core_identity":
            title_core = re.sub(
                r"\s*\([^()]{1,120}\)\s*$", "", source_title
            ).strip()
            if (
                bound_key != canonical_entity_key(title_core)
                or not source_surface_in_text(bound_surface, sentence)
            ):
                return None
        elif identity_reason == "source_title_expanded_alias_identity":






            if not _source_title_expanded_alias_identity(
                surface=bound_surface,
                title=source_title,
                quote=sentence,
            ):
                return None
        elif identity_reason != "validated_binding_identity":
            return None
        if (
            not source_surface_in_text(lexical_answer_surface, sentence)
            or lexical_relation.casefold() not in sentence.casefold()
            or (
                not source_surface_in_text(lexical_bound_surface, sentence)
                and bound_key != canonical_entity_key(source_title)
            )
        ):
            return None

        operator = {
            "operator_id": str(contract.get("operator_id") or ""),
            "canonical_relation": str(contract.get("canonical_relation") or ""),
            "predicate_families": list(contract.get("predicate_families") or []),
            "relation_families": list(contract.get("predicate_families") or []),
            "required_predicate_subtypes": list(
                contract.get("required_predicate_subtypes") or []
            ),
            "strict_predicate_subtype": bool(
                contract.get("strict_predicate_subtype")
            ),
            "relation_tokens": list(contract.get("relation_tokens") or []),
            "output_type": str(contract.get("output_type") or "entity"),
        }
        compiled = compile_typed_candidate_contract(
            validation={
                "answer": lexical_answer_surface,
                "relation_phrase": lexical_relation,
                "support_quote": sentence,
                "bound_surface": bound_surface,
                "source_anchored_predicate_families": list(
                    record.get("source_anchored_predicate_families") or []
                ),
                "source_relation_canonicalization_reason": str(
                    record.get("source_relation_canonicalization_reason") or ""
                ),
                "source_relation_canonicalization_version": str(
                    record.get("source_relation_canonicalization_version") or ""
                ),
                "bound_identity_mode": str(
                    record.get("instance_identity_mode") or ""
                ),
            },
            operator=operator,
        )
        if (
            not compiled.get("admitted")
            or str(compiled.get("reason") or "")
            != str(record.get("predicate_compilation_reason") or "")
            or canonical_entity_key(compiled.get("canonical_relation"))
            != canonical_entity_key(triple[1])
        ):
            return None







        if str(compiled.get("reason") or "") == "lexical_predicate_contract_match":
            return None



        negation_scope = sentence
        for surface in (lexical_bound_surface, lexical_answer_surface):
            if surface:
                negation_scope = re.sub(
                    re.escape(surface), " ", negation_scope, flags=re.IGNORECASE
                )
        relation_at = negation_scope.casefold().find(lexical_relation.casefold())
        if relation_at >= 0:
            prefix = _tokens(negation_scope[max(0, relation_at - 48):relation_at])
            if set(prefix) & {"neither", "never", "no", "not"}:
                return None

        return {
            "support_status": SUPPORTED,
            "support_reason": (
                "binding_conditioned_typed_materialization_source_certificate"
            ),
            "source_sentence_id": _as_int(record.get("source_sentence_id"), -1),
            "source_sentence_span": sentence[:480],
            "source_subject_surface": physical_subject,
            "source_object_surface": physical_object,
            "verifier_version": BINDING_CONDITIONED_TYPED_MATERIALIZATION_VERSION,
            "guarantee_scope": (
                "exact_source_span+binding_identity+typed_relation_contract"
            ),
            "provenance_scope": (
                "exact_title_bound_sentence"
                if bound_key == canonical_entity_key(source_title)
                else "exact_same_sentence"
            ),
            "recovery_contract_version": str(record.get("version") or ""),
            "verified_source_triple": [
                physical_subject, lexical_relation, physical_object
            ],
            "typed_predicate_compilation_reason": str(compiled.get("reason") or ""),
        }

    def verify(self, fact):
        source_id = fact.get("source_id")
        try:
            source_id_int = int(source_id)
            cached_source = dict(
                self._source_document_cache.get(source_id_int) or {}
            )
            source_text = str(cached_source.get("content") or "")
            if not source_text:
                source_text = str(self.get_source_content(source_id_int) or "")
            source_title = str(cached_source.get("title") or "").strip()
            if not source_title and self.get_source_title is not None:
                source_title = str(
                    self.get_source_title(source_id_int) or ""
                ).strip()
            if not source_title:
                source_title = source_text.splitlines()[0].strip() if source_text else ""
        except (TypeError, ValueError, KeyError, IndexError):
            source_id_int = -1
            source_text = ""
            source_title = ""
        if self.verification_mode == "unsafe_admit":
            triple = list(fact.get("triple") or [])
            subject = str(triple[0] if len(triple) == 3 else "").strip()
            object_ = str(triple[2] if len(triple) == 3 else "").strip()
            sentences = _sentences(source_text)
            selected = next(
                (
                    (sentence_id, sentence)
                    for sentence_id, sentence in sentences
                    if (subject and _key(subject) in _key(sentence))
                    or (object_ and _key(object_) in _key(sentence))
                ),
                sentences[0] if sentences else (-1, ""),
            )
            if source_id_int < 0 or not selected[1]:
                return {
                    "support_status": INSUFFICIENT,
                    "support_reason": "unsafe_ablation_missing_source_pointer",
                    "source_sentence_id": -1,
                    "source_sentence_span": "",
                    "verifier_version": self.verifier_version,
                    "guarantee_scope": "none_unsafe_ablation",
                    "source_id": source_id_int,
                    "source_title": source_title,
                    "cost": {
                        "scan_rows": 0,
                        "retrieval_calls": 0,
                        "model_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_hit": False,
                    },
                }
            return {
                "support_status": SUPPORTED,
                "support_reason": "unsafe_source_pointer_admission_ablation",
                "source_sentence_id": selected[0],
                "source_sentence_span": selected[1][:480],
                "source_subject_surface": subject,
                "source_object_surface": object_,
                "verifier_version": self.verifier_version,
                "guarantee_scope": "none_unsafe_ablation",
                "provenance_scope": "source_pointer_only_not_fact_support",
                "source_id": source_id_int,
                "source_title": source_title,
                "unsafe_admission": True,
                "cost": {
                    "scan_rows": 1,
                    "retrieval_calls": 0,
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_hit": False,
                },
            }
        if self.verification_mode == "obligation_only":
            triple = list(fact.get("triple") or [])
            subject = str(triple[0] if len(triple) == 3 else "").strip()
            object_ = str(triple[2] if len(triple) == 3 else "").strip()
            sentences = _sentences(source_text)
            selected = next(
                (
                    (sentence_id, sentence)
                    for sentence_id, sentence in sentences
                    if (subject and _key(subject) in _key(sentence))
                    or (object_ and _key(object_) in _key(sentence))
                ),
                sentences[0] if sentences else (-1, ""),
            )
            if source_id_int < 0 or not source_text:
                return {
                    "support_status": INSUFFICIENT,
                    "support_reason": "obligation_only_missing_source_pointer",
                    "source_support_gate_status": "OFF",
                    "source_support_gate_reason": "disabled_for_ablation",
                    "obligation_admission_status": "NOT_EVALUATED",
                    "obligation_admission_reason": "missing_source_pointer",
                    "source_sentence_id": -1,
                    "source_sentence_span": "",
                    "verifier_version": self.verifier_version,
                    "guarantee_scope": "typed_obligation_without_source_support",
                    "source_id": source_id_int,
                    "source_title": source_title,
                    "cost": {
                        "scan_rows": 0,
                        "retrieval_calls": 0,
                        "model_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_hit": False,
                    },
                }
            admission_fact = dict(fact)
            admission_fact["source_title"] = source_title
            admission_fact["source_text"] = source_text
            admission_fact["source_sentence_span"] = str(
                fact.get("source_sentence_span") or source_text
            )
            admission = (
                SourceSupportObligationAdmissionVerifier._obligation_admission(
                    admission_fact
                )
            )
            admission_status = str(admission.get("status") or "REJECTED")
            if admission_status == "ADMITTED":
                support_status = SUPPORTED
                guarantee_scope = "typed_obligation_without_source_support"
            elif admission_status == "DEFERRED":
                support_status = UNKNOWN
                guarantee_scope = "deferred_obligation_without_source_support"
            else:
                support_status = INSUFFICIENT
                guarantee_scope = "rejected_obligation_without_source_support"
            result = {
                "support_status": support_status,
                "support_reason": (
                    "obligation_only:" + str(admission.get("reason") or "unspecified")
                ),
                "source_support_gate_status": "OFF",
                "source_support_gate_reason": "disabled_for_ablation",
                "obligation_admission_status": admission_status,
                "obligation_admission_reason": str(
                    admission.get("reason") or "unspecified"
                ),
                "obligation_contract_version": (
                    SourceSupportObligationAdmissionVerifier.contract_version
                ),
                "obligation_operator_id": admission.get("operator_id", ""),
                "obligation_output_role": admission.get("output_role", ""),
                "obligation_output_type": admission.get("output_type", ""),
                "obligation_raw_predicate": admission.get("raw_predicate", ""),
                "obligation_canonical_predicate_families": admission.get(
                    "canonical_predicate_families", []
                ),
                "obligation_canonical_predicate_subtypes": admission.get(
                    "canonical_predicate_subtypes", []
                ),
                "obligation_allowed_predicate_subtypes": admission.get(
                    "allowed_predicate_subtypes", []
                ),
                "predicate_contract_version": admission.get(
                    "predicate_contract_version", ""
                ),
                "obligation_terminal_type_status": admission.get(
                    "terminal_type_status", ""
                ),
                "obligation_terminal_type_reason": admission.get(
                    "terminal_type_reason", ""
                ),
                "obligation_terminal_value_type": admission.get(
                    "terminal_value_type", ""
                ),
                "obligation_terminal_value_categories": admission.get(
                    "terminal_value_categories", []
                ),
                "obligation_terminal_target_categories": admission.get(
                    "terminal_target_categories", []
                ),
                "binding_obligation_contract_status": admission.get(
                    "binding_contract_status", ""
                ),
                "binding_obligation_contract_reason": admission.get(
                    "binding_contract_reason", ""
                ),
                "binding_required_input_qualifier_tokens": admission.get(
                    "required_input_qualifier_tokens", []
                ),
                "source_sentence_id": selected[0],
                "source_sentence_span": selected[1][:480],
                "source_subject_surface": subject,
                "source_object_surface": object_,
                "verifier_version": self.verifier_version,
                "admission_policy": self.verifier_version,
                "guarantee_scope": guarantee_scope,
                "provenance_scope": "source_pointer_only_not_fact_support",
                "source_id": source_id_int,
                "source_title": source_title,
                "unsafe_admission": support_status == SUPPORTED,
                "cost": {
                    "scan_rows": 1,
                    "retrieval_calls": 0,
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_hit": False,
                },
            }
            return result
        verifier_fact = dict(fact)
        verifier_fact["source_title"] = source_title
        recovery_record = dict(
            fact.get("source_grounded_recovery_record") or {}
        )
        recovery_contract_version = ""
        certificate_result = self._verify_obligation_bound_certificate(
            fact,
            recovery_record,
            source_text,
            source_title,
            source_id_int,
        )
        schema_compiled_result = self._verify_schema_compiled_source_certificate(
            fact,
            recovery_record,
            source_text,
            source_title,
            source_id_int,
        )
        typed_materialization_result = (
            self._verify_binding_conditioned_typed_materialization_certificate(
                fact,
                recovery_record,
                source_text,
                source_title,
                source_id_int,
            )
        )
        source_grounded_result = (
            self._verify_source_grounded_recovery_certificate(
                fact,
                recovery_record,
                source_text,
                source_title,
                source_id_int,
            )
        )
        prevalidated_source_result = (
            typed_materialization_result
            or certificate_result
            or schema_compiled_result
            or source_grounded_result
        )
        if prevalidated_source_result is not None:
            recovery_contract_version = str(recovery_record.get("version") or "")
            verifier_fact["_prevalidated_source_support"] = dict(
                prevalidated_source_result
            )
        source_material = json.dumps(
            {
                "triple": list(verifier_fact.get("triple") or []),
                "predicate_families": sorted(
                    str(value)
                    for value in verifier_fact.get("predicate_families") or []
                ),
                "predicate_subtypes": sorted(
                    str(value)
                    for value in verifier_fact.get("predicate_subtypes") or []
                ),
                "canonical_relation": str(
                    verifier_fact.get("canonical_relation") or ""
                ),
                "asserted_output_type": str(
                    verifier_fact.get("asserted_output_type")
                    or verifier_fact.get("object_type")
                    or ""
                ),
                "recovery_contract_version": recovery_contract_version,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        source_cache_material = "\x1e".join([
            str(fact.get("physical_fact_id") or fact.get("fact_id") or ""),
            str(source_id_int),
            hashlib.sha1(source_text.encode()).hexdigest(),
            hashlib.sha1(source_title.encode()).hexdigest(),
            str(getattr(RuleBasedSourceSupportVerifier, "version", "")),
            self.verifier_version,
            hashlib.sha1(source_material.encode()).hexdigest(),
        ])
        source_cache_key = hashlib.sha1(source_cache_material.encode()).hexdigest()
        source_cache_hit = source_cache_key in self._source_support_cache
        if source_cache_hit:
            verifier_fact["_prevalidated_source_support"] = dict(
                self._source_support_cache[source_cache_key]
            )
        verifier_material = json.dumps(
            {
                "triple": list(verifier_fact.get("triple") or []),
                "predicate_families": sorted(
                    str(value)
                    for value in verifier_fact.get("predicate_families") or []
                ),
                "canonical_relation": str(
                    verifier_fact.get("canonical_relation") or ""
                ),
                "bound_output_role": str(
                    verifier_fact.get("bound_output_role") or ""
                ),
                "bound_output_key": str(
                    verifier_fact.get("bound_output_key") or ""
                ),
                "obligation_contract": dict(
                    verifier_fact.get("obligation_contract") or {}
                ),
                "recovery_contract_version": recovery_contract_version,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cache_material = "\x1e".join([
            source_cache_key,
            str(source_id_int),
            hashlib.sha1(source_text.encode()).hexdigest(),
            hashlib.sha1(source_title.encode()).hexdigest(),
            self.verifier_version,
            hashlib.sha1(verifier_material.encode()).hexdigest(),
        ])
        cache_key = hashlib.sha1(cache_material.encode()).hexdigest()
        if cache_key in self._verify_cache:
            cached = dict(self._verify_cache[cache_key])
            cached["cost"] = {
                "scan_rows": 0,
                "retrieval_calls": 0,
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_hit": True,
                "source_cache_hit": True,
                "admission_cache_hit": True,
            }
            return cached
        if isinstance(self.verifier, SourceSupportObligationAdmissionVerifier):
            result = dict(self.verifier(verifier_fact, source_text) or {})
        else:
            result = dict(
                prevalidated_source_result
                or self.verifier(verifier_fact, source_text)
                or {}
            )
        if recovery_contract_version:
            result["recovery_contract_version"] = recovery_contract_version
            if prevalidated_source_result is not None:
                result["verified_source_triple"] = list(
                    prevalidated_source_result.get("verified_source_triple")
                    or []
                )
        if result.get("support_status") == SUPPORTED:
            from .relation_fact_join_utils import canonical_entity_key

            triple = list(fact.get("triple") or [])
            title_keys = set(self.relation_index.get("title_backed_entity_keys") or [])
            title_keys.update((self.relation_index.get("title_entity_surfaces") or {}).keys())
            for role, index in (("subject", 0), ("object", 2)):
                candidate = str(result.get(f"source_{role}_surface") or "").strip()
                original = str(triple[index] if len(triple) == 3 else "").strip()
                if not candidate or len(_tokens(candidate)) <= len(_tokens(original)):
                    continue
                if canonical_entity_key(candidate) not in title_keys:
                    result[f"source_{role}_surface"] = (
                        _source_surface(source_text, original) or original
                    )
                    result["identity_surface_refinement_rejected"] = True
        result.setdefault("verifier_version", self.verifier_version)
        result["source_id"] = source_id_int
        result["source_title"] = source_title
        source_certificate = {
            key: value
            for key, value in result.items()
            if not (
                key.startswith("obligation_")
                or key.startswith("predicate_entailment_")
                or key.startswith("binding_obligation_")
                or key == "cost"
            )
        }
        source_certificate["support_status"] = str(
            result.get("source_support_gate_status")
            or result.get("support_status")
            or INSUFFICIENT
        )
        source_certificate["support_reason"] = str(
            result.get("source_support_gate_reason")
            or result.get("support_reason")
            or "unspecified"
        )
        self._source_support_cache[source_cache_key] = source_certificate
        result["cost"] = {
            "scan_rows": 0 if source_cache_hit else (1 if source_text else 0),
            "retrieval_calls": 0,
            "model_calls": _as_int(result.get("model_calls")),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "cache_hit": False,
            "source_cache_hit": source_cache_hit,
            "admission_cache_hit": False,
        }
        self._verify_cache[cache_key] = {
            key: value for key, value in result.items() if key != "cost"
        }
        return result

    def _calibrated_estimate(self, action, state, estimate):
        if self.cbo_cost_model is None:
            return dict(estimate or {})
        return self.cbo_cost_model.estimate(action, state, estimate)

    def _source_value_estimate(self, action, state, estimate):
        result = dict(estimate or {})
        policy = str(state.get("bswc_policy") or "")
        operator_name = str(action.get("operator") or "")
        if policy not in CALIBRATED_POLICIES or operator_name not in {
            BOUND_DENSE_PROBE, SOURCE_INSPECT, VERIFY,
        }:
            return result
        features = action_features(action, state)
        prediction = self.bswc_source_value_model.predict(
            features,
            sample_key=str(state.get("query_id") or ""),
            default=_as_float(result.get("support_probability"), 0.5),
            coarse_only=policy == "bswc_empirical_source_type",
        )
        probability = _bounded(prediction.get("probability"), 0.0, 1.0)
        feedback = {
            "probability": probability,
            "prior_probability": probability,
            "query_feedback_count": 0,
            "query_feedback_success": 0,
            "query_feedback_failure": 0,
            "query_feedback_key": "",
        }
        if policy in FEEDBACK_POLICIES and self.enable_bswc_query_feedback:
            feedback = feedback_adjusted_probability(
                probability,
                features,
                state,
                strength=self.bswc_feedback_strength,
            )
            probability = _bounded(feedback.get("probability"), 0.0, 1.0)

        result.update({
            "support_probability": probability,
            "expected_progress": min(
                0.98,
                max(0.02, probability * (1.0 if operator_name != VERIFY else 0.65)),
            ),
            "bswc_calibrated": bool(prediction.get("covered")),
            "bswc_calibration_model_loaded": bool(
                self.bswc_source_value_model.loaded
            ),
            "bswc_calibration_target": str(features.get("event") or ""),
            "bswc_calibration_probability": probability,
            "bswc_global_prior_probability": _bounded(
                prediction.get("probability"), 0.0, 1.0
            ),
            "bswc_calibration_sample_count": _as_int(
                prediction.get("sample_count")
            ),
            "bswc_calibration_success_count": _as_int(
                prediction.get("success_count")
            ),
            "bswc_calibration_backoff_level": str(
                prediction.get("backoff_level") or "static_prior"
            ),
            "bswc_calibration_uncertainty": _as_float(
                prediction.get("uncertainty"), 1.0
            ),
            "bswc_calibration_model_split": str(
                prediction.get("model_split") or ""
            ),
            "bswc_calibration_features": dict(features),
            "query_feedback_count": _as_int(feedback.get("query_feedback_count")),
            "query_feedback_success": _as_int(feedback.get("query_feedback_success")),
            "query_feedback_failure": _as_int(feedback.get("query_feedback_failure")),
            "query_feedback_key": str(feedback.get("query_feedback_key") or ""),
            "estimate_source": (
                "feedback_calibrated_source_value"
                if prediction.get("covered") else "feedback_static_prior_fallback"
            ),
        })
        if operator_name == SOURCE_INSPECT:
            witness_features = action_features(action, state, witness=True)
            witness_prediction = self.bswc_source_value_model.predict(
                witness_features,
                sample_key=str(state.get("query_id") or ""),
                default=probability * 0.5,
                coarse_only=policy == "bswc_empirical_source_type",
            )
            witness_probability = _bounded(
                witness_prediction.get("probability"), 0.0, 1.0
            )
            if policy in FEEDBACK_POLICIES and self.enable_bswc_query_feedback:
                witness_probability = _bounded(
                    feedback_adjusted_probability(
                        witness_probability,
                        witness_features,
                        state,
                        strength=self.bswc_feedback_strength,
                    ).get("probability"),
                    0.0,
                    1.0,
                )
            result.update({
                "expected_witness_progress": witness_probability,
                "witness_calibration_sample_count": _as_int(
                    witness_prediction.get("sample_count")
                ),
                "witness_calibration_backoff_level": str(
                    witness_prediction.get("backoff_level") or "static_prior"
                ),
                "reader_evidence_novelty": 0.0 if action.get("source_reused") else 1.0,
            })
        return result

    def estimate(self, action, state):
        operator_name = str(action.get("operator") or "")
        operators = list((state.get("logical_plan") or {}).get("operators") or [])
        if operator_name == VERIFY:
            fact_usage_id = str(
                action.get("fact_usage_id") or action.get("fact_id") or ""
            )
            fact = dict((state.get("facts") or {}).get(fact_usage_id) or {})
            dependent_rows = [
                row for row in _active_rows(state)
                if fact_usage_id in set(
                    row.get("path_fact_usage_ids")
                    or row.get("path_fact_ids")
                    or []
                )
            ]
            remaining = max(
                [
                    len(operators) - _as_int(row.get("next_operator_index"))
                    for row in dependent_rows
                ]
                or [0]
            )
            return self._source_value_estimate(action, state, self._calibrated_estimate(action, state, {
                "expected_progress": 0.45,
                "support_probability": (
                    0.78 if fact.get("origin") == "query_local_overlay" else 0.64
                ),
                "downstream_cost": 0.55 * remaining,
                "expected_scan_rows": 1,
                "expected_retrieval_calls": 0,
                "expected_model_calls": 0,
                "expected_tokens": 0,
                "estimate_source": "transparent_source_origin_heuristic",
            }))
        binding = _binding_by_id(state, str(action.get("binding_id")))
        operator = dict(operators[_as_int(action.get("operator_index"))])
        _, bound_key = self._bound_input(binding, operator)
        subject_rows = len(
            (self.relation_index.get("subject_to_items") or {}).get(bound_key) or []
        )
        object_rows = len(
            (self.relation_index.get("object_to_items") or {}).get(bound_key) or []
        )
        fanout = subject_rows + object_rows
        if operator_name == BOUND_DENSE_PROBE:
            return self._source_value_estimate(action, state, self._calibrated_estimate(action, state, {
                "expected_progress": 0.42,
                "unlock_value": 0.75,
                "support_probability": 0.58,
                "binding_compatible_coverage": max(
                    1, _as_int(action.get("binding_compatible_coverage"), 1)
                ),
                "source_relevance": _as_float(
                    action.get("source_relevance"), 0.55
                ),
                "expected_scan_rows": 0,
                "expected_retrieval_calls": 1,
                "expected_model_calls": 0,
                "expected_tokens": 0,
                "estimate_source": "bswc_bound_dense_probe",
            }))
        if operator_name == SOURCE_INSPECT:
            candidate = dict(action.get("source_candidate") or {})
            source_type = str(
                candidate.get("source_type")
                or action.get("source_type")
                or "unknown"
            )
            support_prior = {
                "title_entity_catalog": 0.80,
                "current_evidence_pool": 0.68,
                "dense_completion_probe": 0.60,
            }.get(source_type, 0.55)
            relevance = _bounded(
                action.get("source_relevance"), 0.0, 1.0
            )
            coverage = max(
                1, _as_int(action.get("binding_compatible_coverage"), 1)
            )
            source_id = _as_int(action.get("source_id"), -1)
            source_reused = source_id in {
                _as_int(value, -2)
                for value in state.get("bswc_inspected_source_ids") or []
            }
            evidence_tokens = (
                0
                if source_reused
                else max(0, _as_int(candidate.get("evidence_token_count")))
            )
            return self._source_value_estimate(action, state, self._calibrated_estimate(action, state, {
                "expected_progress": min(
                    0.95, support_prior * (0.70 + 0.20 * relevance)
                ),
                "unlock_value": 1.0 + 0.25 * max(0, coverage - 1),
                "support_probability": support_prior,
                "binding_compatible_coverage": coverage,
                "source_relevance": relevance,
                "source_reused": source_reused,
                "shared_source_reuse_benefit": max(0, coverage - 1),
                "expected_scan_rows": 0 if source_reused else 1,
                "expected_retrieval_calls": 0,
                "expected_model_calls": 0,
                "expected_tokens": evidence_tokens,
                "expected_evidence_tokens": evidence_tokens,
                "estimate_source": "bswc_source_metadata_prior",
            }))
        if operator_name == COMPLETE:
            return self._calibrated_estimate(action, state, {
                "expected_progress": 0.70,
                "unlock_value": 1.0 + 0.25 * max(
                    0, len(operators) - _as_int(action.get("operator_index")) - 1
                ),
                "support_probability": 0.70,
                "expected_scan_rows": self.completion_top_k,
                "expected_retrieval_calls": 1,
                "expected_model_calls": 0,
                "expected_tokens": 0,
                "estimate_source": "bounded_completion_contract",
            })
        estimate = self._calibrated_estimate(action, state, {
            "expected_progress": 0.72 if fanout else 0.05,
            "expected_pruning": 1.0 / float(max(1, fanout)),
            "unlock_value": 0.65,
            "support_probability": 0.64,
            "expected_scan_rows": min(fanout, self.max_facts_per_key * 2),
            "expected_retrieval_calls": 0,
            "expected_model_calls": 0,
            "expected_tokens": 0,
            "estimate_source": "relation_index_cardinality",
        })



        estimate["index_fanout_rows"] = fanout
        estimate["index_has_candidate_rows"] = bool(fanout)
        return estimate


def _fact_id(raw: Mapping[str, Any]) -> str:
    existing = str(raw.get("fact_id") or raw.get("completion_fact_id") or "")
    if existing:
        return existing
    triple = "\x1f".join(str(value or "") for value in raw.get("triple") or [])
    source_id = str(raw.get("source_id", raw.get("doc_idx", "")))
    return "fact:" + hashlib.sha1(f"{triple}\x1e{source_id}".encode()).hexdigest()[:20]


def _admission_context_key(
    fact: Mapping[str, Any],
    *,
    binding_id: str = "",
    obligation_id: str = "",
) -> str:
    """Identify one query-local use of a physical source assertion."""
    material = {
        "binding_id": str(binding_id or ""),
        "obligation_id": str(
            obligation_id or fact.get("operator_id") or ""
        ),
        "operator_id": str(fact.get("operator_id") or ""),
        "bound_output_role": str(fact.get("bound_output_role") or ""),
        "bound_output_key": str(fact.get("bound_output_key") or ""),
        "obligation_contract": dict(fact.get("obligation_contract") or {}),
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(encoded.encode()).hexdigest()


def _register_fact_usage(
    state: Dict[str, Any],
    raw_fact: Mapping[str, Any],
    *,
    binding_id: str,
    obligation_id: str,
) -> Dict[str, Any]:
    """Store admission state per binding/obligation without cloning support truth."""
    fact = dict(raw_fact)
    physical_fact_id = str(
        fact.get("physical_fact_id") or fact.get("fact_id") or _fact_id(fact)
    )
    context_key = _admission_context_key(
        fact,
        binding_id=binding_id,
        obligation_id=obligation_id,
    )
    usage_lookup_key = f"{physical_fact_id}:{context_key}"
    existing_usage_id = str(
        (state.setdefault("fact_usage_lookup", {})).get(usage_lookup_key) or ""
    )
    if existing_usage_id:
        return state["facts"][existing_usage_id]

    usage_id = physical_fact_id
    if usage_id in state["facts"]:
        usage_id = f"{physical_fact_id}#use:{context_key[:12]}"
    fact.update({
        "fact_id": usage_id,
        "physical_fact_id": physical_fact_id,
        "fact_usage_id": usage_id,
        "admission_context_key": context_key,
        "admission_binding_id": str(binding_id or ""),
        "admission_obligation_id": str(obligation_id or ""),
        "source_support_status": str(
            fact.get("source_support_status") or UNVERIFIED
        ),
        "source_support_reason": str(
            fact.get("source_support_reason") or "not_checked"
        ),
        "obligation_admission_status": str(
            fact.get("obligation_admission_status") or "NOT_EVALUATED"
        ),
        "obligation_admission_reason": str(
            fact.get("obligation_admission_reason") or "not_checked"
        ),
    })
    state["facts"][usage_id] = fact
    state["fact_usage_lookup"][usage_lookup_key] = usage_id
    return fact


def _normalise_fact(raw: Mapping[str, Any], origin: str) -> Dict[str, Any]:
    row = _json_safe(dict(raw))
    row["fact_id"] = _fact_id(row)
    row["origin"] = str(row.get("origin") or origin)
    row["source_id"] = row.get(
        "source_id", row.get("completion_source_doc_id", row.get("doc_idx"))
    )
    row["support_status"] = str(row.get("support_status") or UNVERIFIED)
    row["tuple_origin_state"] = str(
        row.get("tuple_origin_state")
        or (TUPLE_INDEXED if row["origin"] == "fact_index" else TUPLE_RECOVERED)
    )
    row["tuple_state"] = str(row.get("tuple_state") or UNVERIFIED)
    row["support_reason"] = str(row.get("support_reason") or "not_checked")
    row["verifier_version"] = str(row.get("verifier_version") or "")
    return row


def _normalise_cost(cost: Mapping[str, Any] | None, elapsed_ms: float) -> Dict[str, Any]:
    raw = dict(cost or {})
    return {
        "scan_rows": max(0, _as_int(raw.get("scan_rows"))),
        "retrieval_calls": max(0, _as_int(raw.get("retrieval_calls"))),
        "model_calls": max(0, _as_int(raw.get("model_calls"))),
        "input_tokens": (
            None if raw.get("input_tokens") is None else max(0, _as_int(raw.get("input_tokens")))
        ),
        "output_tokens": (
            None if raw.get("output_tokens") is None else max(0, _as_int(raw.get("output_tokens")))
        ),
        "evidence_tokens": max(0, _as_int(raw.get("evidence_tokens"))),
        "wall_ms": max(0.0, _as_float(raw.get("wall_ms"), elapsed_ms)),
        "cache_hit": bool(raw.get("cache_hit")),
    }


def _budget_snapshot(ledger: CostLedger, budget: ExecutionBudget) -> Dict[str, Any]:
    known_tokens = (
        ledger.input_tokens + ledger.output_tokens + ledger.evidence_tokens
    )
    return {
        "actions_remaining": max(0, budget.max_actions - ledger.actions),
        "retrieval_calls_remaining": max(
            0, budget.max_retrieval_calls - ledger.retrieval_calls
        ),
        "model_calls_remaining": max(
            0, budget.max_model_calls - budget.reserve_model_calls - ledger.model_calls
        ),
        "scan_rows_remaining": max(0, budget.max_scan_rows - ledger.scan_rows),
        "known_tokens_remaining": (
            None
            if budget.max_tokens is None
            else max(0, budget.max_tokens - budget.reserve_tokens - known_tokens)
        ),
        "token_accounting_complete": ledger.unknown_token_events == 0,
        "evidence_tokens_used": ledger.evidence_tokens,
    }


def _hard_budget_requirements(
    action: Mapping[str, Any],
    estimate: Mapping[str, Any],
    backend: EvidenceBackend,
) -> Dict[str, Any]:
    """Return physical upper bounds used only for hard-budget admission.

    Learned/empirical estimates remain useful for ranking, but they are not
    safety bounds.  Batch index and completion scans are explicitly marked as
    truncatable; atomic source reads and verification are admitted only when
    their complete upper bound fits.
    """
    operator = str(action.get("operator") or "")
    if operator == LOOKUP_JOIN:
        return {
            "hard_scan_rows": max(
                1, 2 * _as_int(getattr(backend, "max_facts_per_key", 20), 20)
            ),
            "hard_retrieval_calls": 0,
            "hard_model_calls": 0,
            "scan_rows_truncatable": True,
            "retrieval_optional": False,
        }
    if operator == COMPLETE:
        top_k = max(1, _as_int(getattr(backend, "completion_top_k", 8), 8))
        relation_backend = isinstance(backend, RelationIndexEvidenceBackend)
        return {
            "hard_scan_rows": (
                2 * top_k
                if relation_backend
                else max(0, _as_int(estimate.get("expected_scan_rows")))
            ),
            "hard_retrieval_calls": max(
                1, _as_int(estimate.get("expected_retrieval_calls"))
            ),
            "hard_model_calls": (
                0
                if relation_backend
                else max(0, _as_int(estimate.get("expected_model_calls")))
            ),
            "scan_rows_truncatable": relation_backend,



            "retrieval_optional": relation_backend,
        }
    if operator == SOURCE_INSPECT:
        return {
            "hard_scan_rows": 0 if estimate.get("source_reused") else 1,
            "hard_retrieval_calls": 0,
            "hard_model_calls": 0,
            "scan_rows_truncatable": False,
            "retrieval_optional": False,
        }
    if operator == BOUND_DENSE_PROBE:
        return {
            "hard_scan_rows": 0,
            "hard_retrieval_calls": 1,
            "hard_model_calls": 0,
            "scan_rows_truncatable": False,
            "retrieval_optional": False,
        }
    if operator == VERIFY:
        return {
            "hard_scan_rows": 1,
            "hard_retrieval_calls": 0,
            "hard_model_calls": 0,
            "scan_rows_truncatable": False,
            "retrieval_optional": False,
        }
    return {
        "hard_scan_rows": max(0, _as_int(estimate.get("expected_scan_rows"))),
        "hard_retrieval_calls": max(
            0, _as_int(estimate.get("expected_retrieval_calls"))
        ),
        "hard_model_calls": max(0, _as_int(estimate.get("expected_model_calls"))),
        "scan_rows_truncatable": False,
        "retrieval_optional": False,
    }


def _affordable(
    estimate: Mapping[str, Any], ledger: CostLedger, budget: ExecutionBudget
) -> Tuple[bool, str]:
    if ledger.actions >= budget.max_actions:
        return False, "action_budget_exhausted"
    hard_retrieval = _as_int(
        estimate.get("hard_retrieval_calls"),
        estimate.get("expected_retrieval_calls"),
    )
    if (
        not estimate.get("retrieval_optional")
        and ledger.retrieval_calls + hard_retrieval > budget.max_retrieval_calls
    ):
        return False, "retrieval_call_budget_exhausted"
    hard_model = _as_int(
        estimate.get("hard_model_calls"), estimate.get("expected_model_calls")
    )
    if ledger.model_calls + hard_model > (
        budget.max_model_calls - budget.reserve_model_calls
    ):
        return False, "model_call_budget_exhausted"
    hard_scan = _as_int(
        estimate.get("hard_scan_rows"), estimate.get("expected_scan_rows")
    )
    scan_remaining = budget.max_scan_rows - ledger.scan_rows
    if hard_scan > scan_remaining:
        if not estimate.get("scan_rows_truncatable") or scan_remaining <= 0:
            return False, "scan_budget_exhausted"
    expected_tokens = estimate.get("expected_tokens")
    if budget.max_tokens is not None and expected_tokens is not None:
        known = (
            ledger.input_tokens + ledger.output_tokens + ledger.evidence_tokens
        )
        if known + _as_int(expected_tokens) > budget.max_tokens - budget.reserve_tokens:
            return False, "token_budget_exhausted"
    return True, "affordable"


def _add_cost(ledger: CostLedger, cost: Mapping[str, Any], budget: ExecutionBudget) -> None:
    next_values = {
        "actions": ledger.actions + 1,
        "scan_rows": ledger.scan_rows + _as_int(cost.get("scan_rows")),
        "retrieval_calls": ledger.retrieval_calls + _as_int(cost.get("retrieval_calls")),
        "model_calls": ledger.model_calls + _as_int(cost.get("model_calls")),
        "tokens": (
            ledger.input_tokens
            + ledger.output_tokens
            + ledger.evidence_tokens
            + _as_int(cost.get("input_tokens"))
            + _as_int(cost.get("output_tokens"))
            + _as_int(cost.get("evidence_tokens"))
        ),
    }
    limits = {
        "actions": budget.max_actions,
        "scan_rows": budget.max_scan_rows,
        "retrieval_calls": budget.max_retrieval_calls,
        "model_calls": budget.max_model_calls - budget.reserve_model_calls,
    }
    if budget.max_tokens is not None:
        limits["tokens"] = max(0, budget.max_tokens - budget.reserve_tokens)
    exceeded = {
        name: (next_values[name], limit)
        for name, limit in limits.items()
        if next_values[name] > limit
    }
    if exceeded:
        raise RuntimeError(
            "hard_budget_contract_violation:" + ",".join(
                f"{name}={actual}>{limit}"
                for name, (actual, limit) in sorted(exceeded.items())
            )
        )
    ledger.actions += 1
    ledger.scan_rows += _as_int(cost.get("scan_rows"))
    ledger.retrieval_calls += _as_int(cost.get("retrieval_calls"))
    ledger.model_calls += _as_int(cost.get("model_calls"))
    ledger.evidence_tokens += _as_int(cost.get("evidence_tokens"))
    if cost.get("input_tokens") is None:
        ledger.unknown_token_events += int(_as_int(cost.get("model_calls")) > 0)
    else:
        ledger.input_tokens += _as_int(cost.get("input_tokens"))
    if cost.get("output_tokens") is None:
        ledger.unknown_token_events += int(_as_int(cost.get("model_calls")) > 0)
    else:
        ledger.output_tokens += _as_int(cost.get("output_tokens"))
    ledger.operator_wall_ms += _as_float(cost.get("wall_ms"))
    ledger.cache_hits += int(bool(cost.get("cache_hit")))
    checks = {
        "action_budget_exceeded": ledger.actions > budget.max_actions,
        "retrieval_budget_exceeded": ledger.retrieval_calls > budget.max_retrieval_calls,
        "model_budget_exceeded": ledger.model_calls > budget.max_model_calls,
        "scan_budget_exceeded": ledger.scan_rows > budget.max_scan_rows,
    }
    if budget.max_tokens is not None and ledger.unknown_token_events == 0:
        checks["token_budget_exceeded"] = (
            ledger.input_tokens + ledger.output_tokens + ledger.evidence_tokens
            > budget.max_tokens - budget.reserve_tokens
        )
    for name, breached in checks.items():
        if breached and name not in ledger.budget_violations:
            ledger.budget_violations.append(name)


def _active_rows(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        row for row in state.get("bindings") or []
        if row.get("active", True) and not row.get("invalidated")
    ]


def _row_complete(row: Mapping[str, Any], operator_count: int) -> bool:
    return _as_int(row.get("next_operator_index")) >= operator_count


def _row_supported(row: Mapping[str, Any], facts: Mapping[str, Any]) -> bool:
    fact_ids = list(
        row.get("path_fact_usage_ids") or row.get("path_fact_ids") or []
    )
    if not fact_ids:
        return False
    for fact_id in fact_ids:
        fact = dict(facts.get(fact_id) or {})
        if fact.get("support_status") != SUPPORTED:
            return False
        source_status = str(
            fact.get("source_support_status") or fact.get("support_status") or ""
        )
        if source_status != SUPPORTED:
            return False
        admission_status = str(
            fact.get("obligation_admission_status") or "ADMITTED"
        )
        if admission_status not in {"ADMITTED", "OFF"}:
            return False
        if fact.get("source_id") is None:
            return False
        if not str(fact.get("source_sentence_span") or "").strip():
            return False
        if not str(fact.get("verifier_version") or "").strip():
            return False
    return True


def _structured_witnesses(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    operators = list((state.get("logical_plan") or {}).get("operators") or [])
    facts = dict(state.get("facts") or {})
    return [
        row for row in _active_rows(state)
        if _row_complete(row, len(operators)) and _row_supported(row, facts)
    ]


def _reader_evidence_selection(
    state: Mapping[str, Any],
    selected_witness: Optional[Mapping[str, Any]],
    provisional_witness: Optional[Mapping[str, Any]] = None,
    partial_witness: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[str], str]:
    """Expose only verified facts that belong to a live execution lineage."""
    if selected_witness:
        return (
            list(dict.fromkeys(
                selected_witness.get("path_fact_usage_ids")
                or selected_witness.get("path_fact_ids")
                or []
            )),
            "complete_verified_witness",
        )
    if provisional_witness:
        return [], "conflict_quarantined_trace_only"
    facts = dict(state.get("facts") or {})
    if partial_witness:
        supported_path_fact_ids = []
        for fact_id in (
            partial_witness.get("path_fact_usage_ids")
            or partial_witness.get("path_fact_ids")
            or []
        ):
            fact = dict(facts.get(str(fact_id)) or {})
            if (
                fact.get("support_status") == SUPPORTED
                and fact.get("source_id") is not None
                and str(fact.get("source_sentence_span") or "").strip()
                and str(fact.get("verifier_version") or "").strip()
            ):
                supported_path_fact_ids.append(str(fact_id))
        if supported_path_fact_ids:
            return list(dict.fromkeys(supported_path_fact_ids)), "verified_partial_completion"
    rows = sorted(
        _active_rows(state),
        key=lambda row: (
            _as_int(row.get("next_operator_index")),
            len(row.get("path_fact_ids") or []),
        ),
        reverse=True,
    )
    fact_ids: List[str] = []
    for row in rows:
        for fact_id in (
            row.get("path_fact_usage_ids") or row.get("path_fact_ids") or []
        ):
            fact = dict(facts.get(fact_id) or {})
            if (
                fact.get("origin") != "query_local_overlay"
                or fact.get("support_status") != SUPPORTED
                or fact.get("source_id") is None
                or not str(fact.get("source_sentence_span") or "").strip()
            ):
                continue
            if fact_id not in fact_ids:
                fact_ids.append(str(fact_id))
    return fact_ids, ("verified_partial_completion" if fact_ids else "none")


def assess_category_affine_supported_partial(
    trace: Mapping[str, Any],
    *,
    min_priority_score: float = 3.0,
    max_terminal_candidates: int = 2,
) -> Dict[str, Any]:
    """Select a bounded, source-supported partial path for Reader evidence.

    This is deliberately weaker than structured witness admission.  A terminal
    tuple whose relation is source-supported but whose answer category remains
    unknown may be shown to the Reader together with its fully supported path;
    it is never promoted to a supported tuple or a structured answer here.
    """
    result: Dict[str, Any] = {
        "eligible": False,
        "reason": "not_evaluated",
        "fact_ids": [],
        "prefix_fact_ids": [],
        "terminal_fact_ids": [],
        "terminal_candidate_count": 0,
        "top_priority_score": 0.0,
        "certificate_document": "",
    }
    if trace.get("structured_admission") or trace.get("selected_witness"):
        result["reason"] = "complete_witness_already_available"
        return result
    plan = dict(trace.get("logical_plan") or {})
    operators = list(plan.get("operators") or [])
    if not operators:
        result["reason"] = "missing_logical_plan"
        return result
    terminal_operator = dict(operators[-1] or {})
    terminal_operator_id = str(terminal_operator.get("operator_id") or "")
    target_categories = [
        str(value).strip().casefold()
        for value in terminal_operator.get("target_entity_categories") or []
        if str(value).strip()
    ]
    if not terminal_operator_id or not target_categories:
        result["reason"] = "terminal_category_contract_absent"
        return result

    facts = {
        str(row.get("fact_id") or ""): dict(row)
        for row in trace.get("facts") or []
        if str(row.get("fact_id") or "")
    }
    candidates: Dict[str, Dict[str, Any]] = {}
    candidate_paths: Dict[str, List[str]] = {}
    for row in trace.get("bindings") or []:
        path = [str(value) for value in row.get("path_fact_ids") or []]
        if not path:
            continue
        terminal_fact_id = path[-1]
        fact = dict(facts.get(terminal_fact_id) or {})
        if str(fact.get("operator_id") or "") != terminal_operator_id:
            continue
        if str(fact.get("source_support_status") or "") != SUPPORTED:
            continue
        if str(fact.get("obligation_admission_status") or "") != "DEFERRED":
            continue
        if str(fact.get("obligation_admission_reason") or "") != "output_type_unknown":
            continue
        if str(fact.get("binding_obligation_contract_status") or "") != "ADMITTED":
            continue
        if not str(fact.get("verified_by_action_key") or "").strip():
            continue
        if fact.get("source_id") is None:
            continue
        if not str(fact.get("source_sentence_span") or "").strip():
            continue
        score = _as_float(fact.get("terminal_candidate_priority_score"), 0.0)
        if score < float(min_priority_score):
            continue
        prefix = path[:-1]
        if not prefix:
            continue
        prefix_supported = True
        for fact_id in prefix:
            prefix_fact = dict(facts.get(fact_id) or {})
            if (
                prefix_fact.get("support_status") != SUPPORTED
                or str(prefix_fact.get("source_support_status") or "") != SUPPORTED
                or str(prefix_fact.get("obligation_admission_status") or "ADMITTED")
                not in {"ADMITTED", "OFF"}
                or prefix_fact.get("source_id") is None
                or not str(prefix_fact.get("source_sentence_span") or "").strip()
            ):
                prefix_supported = False
                break
        if not prefix_supported:
            continue
        candidates[terminal_fact_id] = fact
        candidate_paths[terminal_fact_id] = prefix

    if not candidates:
        result["reason"] = "no_source_supported_category_affine_terminal"
        return result
    top_score = max(
        _as_float(fact.get("terminal_candidate_priority_score"), 0.0)
        for fact in candidates.values()
    )
    top_ids = sorted(
        fact_id for fact_id, fact in candidates.items()
        if _as_float(fact.get("terminal_candidate_priority_score"), 0.0)
        == top_score
    )
    result["terminal_candidate_count"] = len(top_ids)
    result["top_priority_score"] = top_score
    if len(top_ids) > max(0, int(max_terminal_candidates)):
        result["reason"] = "too_many_tied_terminal_candidates"
        return result

    prefix_fact_ids: List[str] = []
    for terminal_fact_id in top_ids:
        for fact_id in candidate_paths[terminal_fact_id]:
            if fact_id not in prefix_fact_ids:
                prefix_fact_ids.append(fact_id)
    selected_fact_ids = prefix_fact_ids + top_ids
    lines = [
        "Source-Supported Partial Relational Witness",
        (
            "Status: incomplete; terminal category remains unresolved. "
            "The following rows are evidence for the Reader, not a structured answer."
        ),
        "Requested terminal category: " + ", ".join(target_categories),
    ]
    for index, fact_id in enumerate(prefix_fact_ids, start=1):
        fact = facts[fact_id]
        lines.append(
            '- supported_step={step} normalized_relation="{triple}" '
            'source="{title}" evidence="{sentence}"'.format(
                step=index,
                triple=" | ".join(str(value) for value in fact.get("triple") or [])[:320],
                title=str(fact.get("source_title") or "")[:160],
                sentence=str(fact.get("source_sentence_span") or "")[:640],
            )
        )
    for index, fact_id in enumerate(top_ids, start=1):
        fact = facts[fact_id]
        lines.append(
            '- unresolved_terminal_candidate={candidate} normalized_relation="{triple}" '
            'source="{title}" evidence="{sentence}"'.format(
                candidate=index,
                triple=" | ".join(str(value) for value in fact.get("triple") or [])[:320],
                title=str(fact.get("source_title") or "")[:160],
                sentence=str(fact.get("source_sentence_span") or "")[:640],
            )
        )
    result.update({
        "eligible": True,
        "reason": "bounded_source_supported_category_affine_partial",
        "fact_ids": selected_fact_ids,
        "prefix_fact_ids": prefix_fact_ids,
        "terminal_fact_ids": top_ids,
        "certificate_document": "\n".join(lines),
    })
    return result


def assess_terminal_answer_determinacy(
    trace: Mapping[str, Any],
) -> Dict[str, Any]:
    """Separate source-supported evidence from answer-determining evidence.

    A text-derived tuple can be true without denoting the attribute requested
    by the query.  For example, ``worked in London`` is source-supported but
    cannot fill an occupation slot.  This audit applies the terminal
    operator's predicate-subtype contract, provenance identity, and terminal
    cardinality before evidence becomes reader-visible or answer-visible.
    """
    plan = dict(trace.get("logical_plan") or {})
    operators = list(plan.get("operators") or [])
    terminal_operator = dict(operators[-1] if operators else {})
    schema_controls = dict(plan.get("schema_constraint_controls") or {})
    enforce_predicate_subtype = bool(
        schema_controls.get("predicate_subtype_constraints", True)
    )
    witness = dict(trace.get("selected_witness") or {})
    slots = dict(witness.get("slot_bindings") or {})
    root_slot = dict(slots.get("$root") or {})
    answer_slot = dict(slots.get("$answer") or {})
    answer_fact_id = str(answer_slot.get("fact_id") or "")
    facts = [dict(row) for row in trace.get("facts") or []]
    facts_by_id = {
        str(row.get("fact_id") or ""): row
        for row in facts
        if str(row.get("fact_id") or "")
    }
    selected_fact = dict(facts_by_id.get(answer_fact_id) or {})
    required_subtypes = {
        str(value)
        for value in terminal_operator.get("required_predicate_subtypes") or []
        if str(value)
    }
    final_answer_type = str(plan.get("final_answer_type") or "").casefold()
    terminal_predicates = {
        str(value)
        for value in terminal_operator.get("predicate_families") or []
        if str(value)
    }


    if enforce_predicate_subtype and not required_subtypes:
        if final_answer_type == "occupation":
            required_subtypes.add("occupation_role")
        elif final_answer_type == "nationality":
            required_subtypes.add("nationality")
        elif final_answer_type in {"date", "place", "location"}:
            for family in terminal_predicates & {"birth", "death"}:
                suffix = "place" if final_answer_type in {"place", "location"} else "date"
                required_subtypes.add(f"{family}_{suffix}")

    terminal_input_slot = str(
        terminal_operator.get("input_slot") or "$root"
    ).strip() or "$root"
    terminal_input_binding = dict(
        slots.get(terminal_input_slot) or root_slot
    )
    terminal_input_key = str(
        terminal_input_binding.get("canonical_identity_key")
        or terminal_input_binding.get("key")
        or ""
    ).strip()
    terminal_input_surface = str(
        terminal_input_binding.get("canonical_identity_surface")
        or terminal_input_binding.get("surface")
        or ""
    ).strip()
    root_metadata = dict(witness.get("root_metadata") or {})
    canonical_title_doc_ids = {
        _as_int(value, -1)
        for value in root_metadata.get("canonical_title_doc_ids") or []
    }

    def fact_subtypes(fact: Mapping[str, Any]) -> set[str]:
        values = {
            str(value)
            for value in fact.get("predicate_subtypes") or []
            if str(value)
        }
        try:
            from .relation_fact_join_utils import (
                canonical_schema_predicate_subtypes,
            )

            triple = list(fact.get("triple") or [])
            values.update(canonical_schema_predicate_subtypes(
                fact.get("relation") or (triple[1] if len(triple) == 3 else ""),
                fact.get("object") or (triple[2] if len(triple) == 3 else ""),
            ))
        except (ImportError, TypeError, ValueError):
            pass
        return values

    def identity_matches(fact: Mapping[str, Any]) -> bool:
        fact_identity_key = str(fact.get("lineage_identity_key") or "").strip()
        if terminal_input_key and fact_identity_key:
            return terminal_input_key == fact_identity_key
        source_id = _as_int(fact.get("source_id"), -1)
        if (
            terminal_input_slot == "$root"
            and root_metadata.get("title_catalog_exact")
            and canonical_title_doc_ids
        ):
            return source_id in canonical_title_doc_ids
        source_span = str(fact.get("source_sentence_span") or "")
        input_tokens = set(_tokens(terminal_input_surface))
        return bool(
            input_tokens
            and input_tokens.issubset(set(_tokens(source_span)))
        )

    def predicate_compatibility(fact: Mapping[str, Any]) -> Dict[str, Any]:
        if enforce_predicate_subtype and required_subtypes:
            matched = bool(fact_subtypes(fact) & required_subtypes)
            return {
                "compatible": matched,
                "direct": matched,
                "reason": (
                    "direct_predicate_subtype_match"
                    if matched else "predicate_subtype_mismatch"
                ),
                "orientations": [],
            }
        fact_predicates = {
            str(value)
            for value in fact.get("predicate_families") or []
            if str(value)
        }
        direct = bool(terminal_predicates and fact_predicates & terminal_predicates)
        if direct:
            return {
                "compatible": True,
                "direct": True,
                "reason": "direct_predicate_family_match",
                "orientations": [],
            }
        contract = dict(fact.get("obligation_contract") or {})
        input_side = str(contract.get("bound_input_side") or "").strip()
        output_side = str(fact.get("bound_output_role") or "").strip()




        if not (
            input_side in {"subject", "object"}
            and output_side in {"subject", "object"}
            and fact.get("obligation_admission_status") == "ADMITTED"
            and fact.get("binding_obligation_contract_status") == "ADMITTED"
        ):
            return {
                "compatible": False,
                "direct": False,
                "reason": "predicate_family_mismatch",
                "orientations": [],
            }
        role_compatibility = role_aware_predicate_contract_compatibility(
            fact,
            terminal_operator,
            input_side=input_side,
            output_side=output_side,
        )
        return {
            "compatible": bool(role_compatibility.get("compatible")),
            "direct": False,
            "reason": str(role_compatibility.get("reason") or ""),
            "orientations": list(role_compatibility.get("orientations") or []),
        }

    def predicate_matches(fact: Mapping[str, Any]) -> bool:
        return bool(predicate_compatibility(fact).get("compatible"))

    eligible_facts: List[Dict[str, Any]] = []
    for fact in facts:
        if fact.get("support_status") != SUPPORTED:
            continue
        if fact.get("source_id") is None:
            continue
        if not str(fact.get("source_sentence_span") or "").strip():
            continue
        if terminal_operator.get("operator_id") and str(
            fact.get("operator_id") or ""
        ) != str(terminal_operator.get("operator_id")):
            continue
        if not predicate_matches(fact) or not identity_matches(fact):
            continue
        eligible_facts.append(fact)

    eligible_fact_ids = [
        str(fact.get("fact_id"))
        for fact in eligible_facts
        if str(fact.get("fact_id") or "")
    ]
    terminal_values = {
        str(
            fact.get("verified_output_key")
            or fact.get("bound_output_key")
            or fact.get("object_key")
            or fact.get("verified_output_surface")
            or fact.get("object")
            or ""
        ).strip()
        for fact in eligible_facts
    }
    terminal_values.discard("")
    selected_exact = bool(answer_fact_id and answer_fact_id in eligible_fact_ids)
    selected_source_supported = bool(
        selected_fact.get("support_status") == SUPPORTED
        and selected_fact.get("source_id") is not None
        and str(selected_fact.get("source_sentence_span") or "").strip()
    )
    selected_predicate_contract = (
        predicate_compatibility(selected_fact) if selected_fact else {}
    )
    selected_predicate_exact = bool(
        selected_predicate_contract.get("compatible")
        and selected_predicate_contract.get("direct")
    )
    selected_predicate_compatible = bool(
        selected_predicate_contract.get("compatible")
    )
    selected_identity_match = bool(
        selected_fact and identity_matches(selected_fact)
    )
    single_value = len(terminal_values) == 1
    functional_subtypes = {
        "birth_date", "birth_place", "death_date", "death_place",
    }
    functional_contract = bool(required_subtypes & functional_subtypes)
    reduction = dict(trace.get("terminal_reduction") or {})
    complete_reduction = str(reduction.get("status") or "") in {
        "branch_reduction_complete",
    }
    answer_determining = bool(
        trace.get("stop_reason") == "verified_complete_witness"
        and selected_exact
        and single_value
        and (functional_contract or complete_reduction)
    )
    reader_eligible = bool(eligible_facts)
    if not reader_eligible:
        reason = "no_predicate_bound_supported_terminal_fact"
    elif not selected_exact:
        reason = "supported_terminal_fact_not_selected_witness"
    elif not single_value:
        reason = "multiple_supported_terminal_values"
    elif not (functional_contract or complete_reduction):
        reason = "supported_but_set_valued_terminal"
    else:
        reason = "unique_functional_supported_terminal"
    return {
        "version": "terminal_answer_determinacy_v2",
        "reader_evidence_eligible": reader_eligible,
        "answer_materialization_eligible": answer_determining,
        "reason": reason,
        "answer_fact_id": answer_fact_id,
        "selected_terminal_exact": selected_exact,
        "selected_source_supported": selected_source_supported,
        "selected_predicate_exact": selected_predicate_exact,
        "selected_predicate_compatible": selected_predicate_compatible,
        "selected_predicate_compatibility_reason": str(
            selected_predicate_contract.get("reason") or ""
        ),
        "selected_predicate_compatibility_orientations": list(
            selected_predicate_contract.get("orientations") or []
        ),
        "selected_identity_match": selected_identity_match,
        "required_predicate_subtypes": sorted(required_subtypes),
        "eligible_fact_ids": eligible_fact_ids,
        "eligible_fact_count": len(eligible_fact_ids),
        "terminal_value_count": len(terminal_values),
        "terminal_values": sorted(terminal_values),
        "functional_contract": functional_contract,
        "identity_slot": terminal_input_slot,
        "identity_key": terminal_input_key,
        "identity_surface": terminal_input_surface,
    }


def assess_terminal_answer_domain(trace: Mapping[str, Any]) -> Dict[str, Any]:
    """Audit whether the selected terminal surface answers the asked domain."""
    plan = dict(trace.get("logical_plan") or {})
    operators = list(plan.get("operators") or [])
    terminal_operator = dict(operators[-1] if operators else {})
    witness = dict(trace.get("selected_witness") or {})
    answer_slot = dict(
        (witness.get("slot_bindings") or {}).get("$answer") or {}
    )
    answer_fact_id = str(answer_slot.get("fact_id") or "")
    facts_by_id = {
        str(row.get("fact_id") or ""): dict(row)
        for row in trace.get("facts") or []
        if str(row.get("fact_id") or "")
    }
    answer_fact = dict(facts_by_id.get(answer_fact_id) or {})
    value = str(
        answer_slot.get("surface")
        or answer_fact.get("verified_output_surface")
        or answer_fact.get("bound_output_surface")
        or answer_fact.get("object")
        or ""
    ).strip()
    families = {
        str(item)
        for item in terminal_operator.get("predicate_families") or []
        if str(item)
    }
    families.update(
        str(item)
        for item in answer_fact.get("predicate_families") or []
        if str(item)
    )
    result = terminal_answer_domain_compatibility(
        str(plan.get("question") or ""),
        value,
        final_answer_type=str(plan.get("final_answer_type") or ""),
        relation_families=families,
    )
    return {
        "version": "terminal_answer_domain_contract_v1",
        **result,
        "answer_fact_id": answer_fact_id,
        "reader_evidence_eligible": result.get("status")
        != TERMINAL_TYPE_INCOMPATIBLE,
        "answer_materialization_eligible": result.get("status")
        != TERMINAL_TYPE_INCOMPATIBLE,
    }


def _source_grounded_terminal_surface(
    *,
    question: str,
    candidate: str,
    source_span: str,
    final_answer_type: str,
    relation_families: Sequence[str],
) -> Tuple[str, Dict[str, Any]]:
    """Project a verified terminal location to a source-explicit short surface.

    This operator is deliberately narrow.  It never invents an alias and only
    handles a terminal value that was already admitted from the cited source.
    """
    before = str(candidate or "").strip()
    source = str(source_span or "").strip()
    meta = {
        "version": "source_grounded_terminal_surface_v1",
        "applied": False,
        "reason": "no_safe_projection",
        "before": before,
        "after": before,
    }
    if not before or not source or final_answer_type not in {"place", "location"}:
        meta["reason"] = "non_location_or_missing_source"
        return before, meta
    if _key(before) not in _key(source):
        meta["reason"] = "terminal_surface_not_in_source"
        return before, meta
    if not re.search(r"\b(?:where|place|location)\b", question, re.IGNORECASE):
        meta["reason"] = "question_not_generic_location"
        return before, meta
    if re.search(
        r"\b(?:country|county|state|province|region|district)\b",
        question,
        re.IGNORECASE,
    ):
        meta["reason"] = "question_requests_coarse_location"
        return before, meta

    families = {str(item).casefold() for item in relation_families if str(item)}
    if not families.intersection({"birth", "death", "location"}):
        meta["reason"] = "terminal_relation_not_location_bearing"
        return before, meta

    escaped = re.escape(before).replace(r"\ ", r"\s+")
    alias_match = re.search(
        rf"(?<!\w){escaped}\s*\((?P<alias>[^)]{{1,60}})\)",
        source,
        re.IGNORECASE,
    )
    if alias_match:
        alias = alias_match.group("alias").strip(" \t\r\n.;:-")
        alias_tokens = _tokens(alias)
        unsafe_alias = bool(
            not alias_tokens
            or len(alias_tokens) > 5
            or re.search(r"\d", alias)
            or re.search(
                r"\b(?:film|movie|song|album|novel|series|book)\b",
                alias,
                re.IGNORECASE,
            )
        )
        if not unsafe_alias and _key(alias) != _key(before):
            after = f"{before} ({alias})"
            return after, {
                **meta,
                "applied": True,
                "reason": "source_explicit_location_alias",
                "after": after,
                "alias": alias,
            }

    parts = [
        part.strip(" \t\r\n.;:-")
        for part in before.split(",")
        if part.strip(" \t\r\n.;:-")
    ]
    if not (2 <= len(parts) <= 4):
        meta["reason"] = "no_location_hierarchy"
        return before, meta
    if any(
        re.search(r"\d|;|/|\b(?:and|or)\b", part, re.IGNORECASE)
        or len(_tokens(part)) > 5
        for part in parts
    ):
        meta["reason"] = "unsafe_location_hierarchy"
        return before, meta

    relation_cue = ""
    if "birth" in families:
        relation_cue = r"\bborn\s+in\s+"
    elif "death" in families:
        relation_cue = r"\b(?:died|death)\b.{0,28}?\b(?:in|at|near|outside|of)\s+"
    elif "location" in families:
        relation_cue = r"\b(?:located|situated)\s+(?:in|at|near)\s+"
    first_escaped = re.escape(parts[0]).replace(r"\ ", r"\s+")
    if not relation_cue or not re.search(
        relation_cue + first_escaped,
        source,
        re.IGNORECASE,
    ):
        meta["reason"] = "location_hierarchy_not_relation_anchored"
        return before, meta






    return before, {
        **meta,
        "reason": "nonredundant_location_hierarchy_preserved",
        "location_parts": parts,
    }


def select_paper_answer(
    *,
    question: str,
    reader_answer: Any,
    trace: Mapping[str, Any],
    incumbent_guard: bool = True,
    v8_safe_materialization: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Choose between the Reader incumbent and a verified structured result."""
    incumbent = str(reader_answer or "").strip()
    candidate = str(trace.get("structured_answer") or "").strip()
    base = {
        "reader_answer_before_admission": incumbent,
        "candidate_structured_answer": candidate,
        "incumbent_guard_enabled": bool(incumbent_guard),
    }
    answer_domain = dict(trace.get("terminal_answer_domain") or {})
    if answer_domain.get("status") == TERMINAL_TYPE_INCOMPATIBLE:
        return incumbent, {
            **base,
            "path": "terminal_answer_domain_guard",
            "answer_after_admission": incumbent,
            "reason": str(
                answer_domain.get("reason")
                or "terminal_answer_domain_incompatible"
            ),
            "replaced": False,
        }
    plan = dict(trace.get("logical_plan") or {})
    coverage_enforced = bool(
        plan.get("plan_coverage_contract_version")
        == PLAN_COVERAGE_CONTRACT_VERSION
    )
    coverage_status = str(
        (plan.get("plan_coverage") or {}).get("status")
        or plan.get("plan_coverage_status")
        or ""
    )
    if coverage_enforced and coverage_status != PLAN_COMPLETE:
        return incumbent, {
            **base,
            "path": "question_coverage_fallback",
            "reason": f"structured_answer_blocked_{coverage_status.lower() or 'unknown'}",
            "replaced": False,
            "answer_after_admission": incumbent,
            "plan_coverage_status": coverage_status,
        }
    if not trace.get("structured_admission") or not candidate:
        return incumbent, {
            **base,
            "path": str(trace.get("output_path") or "ordinary_baseline_fallback"),
            "reason": str(trace.get("stop_reason") or "no_structured_answer"),
            "replaced": False,
            "answer_after_admission": incumbent,
        }
    if not incumbent_guard:
        return candidate, {
            **base,
            "path": "verified_structured_admission",
            "reason": "incumbent_guard_disabled",
            "replaced": _key(candidate) != _key(incumbent),
            "answer_after_admission": candidate,
        }
    incumbent_key, candidate_key = _key(incumbent), _key(candidate)
    raw_candidate = candidate
    raw_candidate_key = candidate_key
    v8_safe_plan = bool(
        v8_safe_materialization and plan.get("newly_compiled_v8")
    )
    v9_safe_plan = bool(
        v8_safe_materialization and plan.get("newly_compiled_v9")
    )
    safe_new_plan = bool(v8_safe_plan or v9_safe_plan)
    if not incumbent_key and not safe_new_plan:
        return candidate, {
            **base,
            "path": "verified_structured_admission",
            "reason": "empty_reader_incumbent",
            "replaced": True,
            "answer_after_admission": candidate,
        }
    final_answer_type = str(plan.get("final_answer_type") or "").casefold()
    incumbent_location_parts = [
        part.strip()
        for part in re.split(r"\s*,\s*", incumbent)
        if part.strip()
    ]
    witness = dict(trace.get("selected_witness") or {})
    answer_slot = dict((witness.get("slot_bindings") or {}).get("$answer") or {})
    answer_fact_id = str(answer_slot.get("fact_id") or "")
    facts_by_id = {
        str(row.get("fact_id") or ""): dict(row)
        for row in trace.get("facts") or []
    }
    answer_fact = facts_by_id.get(answer_fact_id, {})
    verified_terminal_surface = str(
        answer_fact.get("verified_output_surface")
        or answer_slot.get("surface")
        or ""
    ).strip()
    source_span = str(
        answer_fact.get("source_sentence_span")
        or answer_slot.get("lineage_source_sentence_span")
        or ""
    ).strip()
    terminal_supported = bool(
        answer_fact.get("support_status") == SUPPORTED
        and source_span
        and _key(verified_terminal_surface) == raw_candidate_key
    )
    relation_families = {
        str(item).casefold()
        for item in answer_fact.get("predicate_families") or []
        if str(item).strip()
    }
    surface_projection = {
        "applied": False,
        "reason": "terminal_fact_not_projection_eligible",
        "before": raw_candidate,
        "after": raw_candidate,
    }
    if terminal_supported:
        candidate, surface_projection = _source_grounded_terminal_surface(
            question=question,
            candidate=raw_candidate,
            source_span=source_span,
            final_answer_type=final_answer_type,
            relation_families=relation_families,
        )
        candidate_key = _key(candidate)
    base.update({
        "candidate_structured_answer_raw": raw_candidate,
        "candidate_structured_answer": candidate,
        "source_grounded_terminal_surface": surface_projection,
        "v9_safe_materialization_enabled": v9_safe_plan,
        "v9_plan_validation_status": str(
            plan.get("plan_validation_status") or "unavailable"
        ) if v9_safe_plan else "not_applicable",
    })
    if incumbent_key == candidate_key:
        return incumbent, {
            **base,
            "path": "incumbent_preserved",
            "reason": "normalized_answers_equivalent",
            "replaced": False,
            "answer_after_admission": incumbent,
        }
    if v8_safe_plan:
        determinacy = assess_terminal_answer_determinacy(trace)
        v8_base = {
            **base,
            "v8_safe_materialization_enabled": True,
            "terminal_answer_determinacy": determinacy,
            "v8_compile_guard_reason": str(
                plan.get("v8_compile_guard_reason") or "unavailable"
            ),
        }
        if incumbent_key and not _is_abstaining_answer(incumbent):
            return incumbent, {
                **v8_base,
                "path": "incumbent_preserved",
                "reason": "v8_non_abstaining_incumbent_shadow_only",
                "replaced": False,
                "answer_after_admission": incumbent,
            }
        if not determinacy.get("answer_materialization_eligible"):
            root_metadata_for_reason = dict(witness.get("root_metadata") or {})
            if not determinacy.get("selected_identity_match"):
                determinacy_reason = (
                    "v8_terminal_fact_outside_root_identity_partition"
                    if root_metadata_for_reason.get("canonical_title_doc_ids")
                    else "v8_root_entity_not_explicit_in_source"
                )
            elif not determinacy.get("selected_source_supported"):
                determinacy_reason = "v8_terminal_fact_not_source_verified"
            else:
                determinacy_reason = "v8_terminal_not_answer_determining"
            return incumbent, {
                **v8_base,
                "path": "incumbent_preserved",
                "reason": determinacy_reason,
                "replaced": False,
                "answer_after_admission": incumbent,
            }
        root_slot = dict((witness.get("slot_bindings") or {}).get("$root") or {})
        root_metadata = dict(witness.get("root_metadata") or {})
        root_surface = str(
            root_slot.get("canonical_identity_surface")
            or root_slot.get("surface")
            or ""
        ).strip()
        root_tokens = set(_tokens(root_surface))
        root_explicit = bool(
            root_tokens and root_tokens.issubset(set(_tokens(source_span)))
        )
        canonical_title_doc_ids = {
            _as_int(value, -1)
            for value in root_metadata.get("canonical_title_doc_ids") or []
        }
        source_id = _as_int(
            answer_fact.get("source_id", answer_slot.get("lineage_source_id")),
            -1,
        )
        identity_partition_match = bool(
            determinacy.get("selected_terminal_exact")
            or (
                root_metadata.get("title_catalog_exact")
                and canonical_title_doc_ids
                and source_id in canonical_title_doc_ids
            )
        )
        v8_terminal_supported = bool(
            answer_fact.get("support_status") == SUPPORTED
            and source_span
            and _key(verified_terminal_surface) == raw_candidate_key
        )
        if not v8_terminal_supported:
            return incumbent, {
                **v8_base,
                "path": "incumbent_preserved",
                "reason": "v8_terminal_fact_not_source_verified",
                "replaced": False,
                "answer_after_admission": incumbent,
            }
        if not root_explicit and not determinacy.get("identity_key"):
            return incumbent, {
                **v8_base,
                "path": "incumbent_preserved",
                "reason": "v8_root_entity_not_explicit_in_source",
                "replaced": False,
                "answer_after_admission": incumbent,
                "terminal_fact_id": answer_fact_id,
            }
        if not identity_partition_match:
            return incumbent, {
                **v8_base,
                "path": "incumbent_preserved",
                "reason": "v8_terminal_fact_outside_root_identity_partition",
                "replaced": False,
                "answer_after_admission": incumbent,
                "terminal_fact_id": answer_fact_id,
                "terminal_source_id": source_id,
                "canonical_title_doc_ids": sorted(canonical_title_doc_ids),
            }
        if _is_abstaining_answer(candidate):
            return incumbent, {
                **v8_base,
                "path": "incumbent_preserved",
                "reason": "v8_candidate_is_abstention",
                "replaced": False,
                "answer_after_admission": incumbent,
            }

    plan_kind = str(trace.get("query_shape") or "")
    reduction_status = str(
        (trace.get("terminal_reduction") or {}).get("status") or ""
    )
    if (
        plan_kind == "branch_temporal_compare"
        and reduction_status == "branch_reduction_complete"
    ):
        return incumbent, {
            **base,
            "path": "incumbent_preserved",
            "reason": "temporal_branch_shadow_only_pending_order_validation",
            "replaced": False,
            "answer_after_admission": incumbent,
        }

    if surface_projection.get("applied"):
        return candidate, {
            **base,
            "path": "verified_structured_admission",
            "reason": str(surface_projection.get("reason")),
            "replaced": candidate_key != incumbent_key,
            "answer_after_admission": candidate,
            "terminal_fact_id": answer_fact_id,
        }
    duplicate_location_parts = bool(
        len(incumbent_location_parts) >= 2
        and len({_key(part) for part in incumbent_location_parts}) == 1
    )
    verified_duplicate_location = bool(
        final_answer_type in {"place", "location"}
        and trace.get("stop_reason") == "verified_complete_witness"
        and answer_fact.get("support_status") == SUPPORTED
        and str(answer_fact.get("source_sentence_span") or "").strip()
        and duplicate_location_parts
        and _key(incumbent_location_parts[0]) == candidate_key
        and _key(verified_terminal_surface) == candidate_key
    )
    if verified_duplicate_location:
        return candidate, {
            **base,
            "path": "verified_structured_admission",
            "reason": "verified_duplicate_location_canonicalization",
            "replaced": True,
            "answer_after_admission": candidate,
            "final_answer_type": final_answer_type,
            "terminal_fact_id": answer_fact_id,
        }
    administrative_markers = {
        "prefecture", "province", "municipality", "district", "region"
    }
    first_location_tokens = _tokens(
        incumbent_location_parts[0] if incumbent_location_parts else ""
    )
    second_location_tokens = _tokens(
        incumbent_location_parts[1] if len(incumbent_location_parts) > 1 else ""
    )
    redundant_administrative_location = bool(
        trace.get("version") == "paper_evidence_executor_v7_1"
        and final_answer_type in {"place", "location"}
        and trace.get("stop_reason") == "verified_complete_witness"
        and answer_fact.get("support_status") == SUPPORTED
        and str(answer_fact.get("source_sentence_span") or "").strip()
        and len(incumbent_location_parts) >= 2
        and _key(incumbent_location_parts[0]) == candidate_key
        and _key(verified_terminal_surface) == candidate_key
        and first_location_tokens
        and second_location_tokens[:len(first_location_tokens)] == first_location_tokens
        and bool(
            set(second_location_tokens[len(first_location_tokens):])
            & administrative_markers
        )
        and incumbent_key in _key(answer_fact.get("source_sentence_span"))
    )
    if redundant_administrative_location:
        return candidate, {
            **base,
            "path": "verified_structured_admission",
            "reason": "verified_redundant_administrative_location",
            "replaced": True,
            "answer_after_admission": candidate,
            "final_answer_type": final_answer_type,
            "terminal_fact_id": answer_fact_id,
            "redundant_location_canonicalization": True,
        }
    incumbent_year_match = re.fullmatch(r"\s*(\d{4})[.]?\s*", incumbent)
    candidate_complete_date = re.fullmatch(
        r"\s*(?:"
        r"\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?,?\s+(\d{4})"
        r"|"
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+"
        r"\d{1,2}(?:st|nd|rd|th)?,?\s+(\d{4})"
        r")\s*",
        candidate,
        re.IGNORECASE,
    )
    candidate_year = (
        next(
            (
                group for group in candidate_complete_date.groups()
                if group
            ),
            "",
        )
        if candidate_complete_date
        else ""
    )
    if (
        final_answer_type == "date"
        and terminal_supported
        and incumbent_year_match
        and candidate_year == incumbent_year_match.group(1)
    ):
        return candidate, {
            **base,
            "path": "verified_structured_admission",
            "reason": "source_grounded_complete_date_specificity_upgrade",
            "replaced": True,
            "answer_after_admission": candidate,
            "terminal_fact_id": answer_fact_id,
        }
    if (
        incumbent_key
        and candidate_key
        and (
            incumbent_key in candidate_key
            or candidate_key in incumbent_key
        )
    ):
        return incumbent, {
            **base,
            "path": "incumbent_preserved",
            "reason": "surface_granularity_containment",
            "replaced": False,
            "answer_after_admission": incumbent,
        }

    source_spans = [
        str(facts_by_id.get(str(fact_id), {}).get("source_sentence_span") or "")
        for fact_id in (
            witness.get("path_fact_usage_ids")
            or witness.get("path_fact_ids")
            or []
        )
    ]
    candidate_supported = any(
        candidate_key and candidate_key in _key(span) for span in source_spans
    )
    incumbent_supported = any(
        incumbent_key and incumbent_key in _key(span) for span in source_spans
    )
    if candidate_supported and not incumbent_supported:
        equivalent_complete_date = bool(
            terminal_supported
            and equivalent_complete_date_surfaces(incumbent, candidate)
        )
        if equivalent_complete_date:
            return candidate, {
                **base,
                "path": "verified_structured_admission",
                "reason": "source_grounded_equivalent_date_canonicalization",
                "replaced": True,
                "answer_after_admission": candidate,
                "terminal_fact_id": answer_fact_id,
            }
        if incumbent_key and not _is_abstaining_answer(incumbent):
            return incumbent, {
                **base,
                "path": "incumbent_preserved",
                "reason": "non_equivalent_terminal_candidate_shadow_only",
                "replaced": False,
                "answer_after_admission": incumbent,
                "terminal_fact_id": answer_fact_id,
            }
        return candidate, {
            **base,
            "path": "verified_structured_admission",
            "reason": (
                "v8_verified_abstention_repair"
                if v8_safe_materialization and bool(plan.get("newly_compiled_v8"))
                else "terminal_witness_supports_candidate_only"
            ),
            "replaced": True,
            "answer_after_admission": candidate,
        }
    return incumbent, {
        **base,
        "path": "incumbent_preserved",
        "reason": (
            "terminal_witness_also_supports_incumbent"
            if incumbent_supported else "structured_replacement_not_decisive"
        ),
        "replaced": False,
        "answer_after_admission": incumbent,
    }


def _branch_witness_groups(
    witnesses: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for witness in witnesses:
        root_id = str(witness.get("root_binding_id") or "")
        if root_id:
            groups.setdefault(root_id, []).append(dict(witness))
    return groups


def _reduce_witnesses(
    witnesses: Sequence[Mapping[str, Any]],
    logical_plan: Mapping[str, Any],
    root_binding_ids: Sequence[str],
    facts: Optional[Mapping[str, Any]] = None,
    conflict_safe_terminal: bool = False,
    provenance_grouped_terminal: bool = False,
) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
    """Materialize a projection or a provenance-preserving branch reduction."""
    answer_operator = str(logical_plan.get("answer_operator") or "project")
    if answer_operator == "project":
        if not witnesses:
            return None, "path_witness_incomplete", {}
        facts = dict(facts or {})
        def witness_rank(row: Mapping[str, Any]) -> Tuple[Any, ...]:
            path_facts = [
                dict(facts.get(str(fact_id)) or {})
                for fact_id in (
                    row.get("path_fact_usage_ids")
                    or row.get("path_fact_ids")
                    or []
                )
            ]
            identity_confidence = sum(
                _as_float(fact.get("lineage_identity_confidence"), 0.0)
                for fact in path_facts
            )
            identity_validated = sum(
                bool(fact.get("lineage_identity_validated"))
                for fact in path_facts
            )
            answer = str(
                ((row.get("slot_bindings") or {}).get("$answer") or {}).get(
                    "surface"
                )
                or ""
            )
            return (
                identity_validated,
                round(identity_confidence, 6),
                len(_tokens(answer)),
                len(answer),
                -_as_int(str(row.get("binding_id") or "").lstrip("b"), 10**9),
            )

        def identity_tier(row: Mapping[str, Any]) -> Tuple[int, float]:
            path_facts = [
                dict(facts.get(str(fact_id)) or {})
                for fact_id in (
                    row.get("path_fact_usage_ids")
                    or row.get("path_fact_ids")
                    or []
                )
            ]
            return (
                sum(bool(fact.get("lineage_identity_validated")) for fact in path_facts),
                round(sum(
                    _as_float(fact.get("lineage_identity_confidence"), 0.0)
                    for fact in path_facts
                ), 6),
            )

        def terminal_provenance(row: Mapping[str, Any]) -> Tuple[str, str, str]:
            fact_ids = list(
                row.get("path_fact_usage_ids")
                or row.get("path_fact_ids")
                or []
            )
            terminal_fact = dict(facts.get(str(fact_ids[-1])) or {}) if fact_ids else {}
            return (
                str(terminal_fact.get("source_id")),
                str(terminal_fact.get("source_sentence_id") or ""),
                _key(terminal_fact.get("source_sentence_span")),
            )

        candidate_witnesses = [dict(row) for row in witnesses]
        identity_dominated_count = 0
        if provenance_grouped_terminal and candidate_witnesses:
            strongest_tier = max(identity_tier(row) for row in candidate_witnesses)
            strongest = [
                row for row in candidate_witnesses
                if identity_tier(row) == strongest_tier
            ]
            identity_dominated_count = len(candidate_witnesses) - len(strongest)
            candidate_witnesses = strongest

        answer_groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in candidate_witnesses:
            answer = str(
                ((row.get("slot_bindings") or {}).get("$answer") or {}).get(
                    "surface"
                )
                or ""
            ).strip()
            if _key(answer):
                answer_groups.setdefault(_key(answer), []).append(dict(row))

        reduction_detail = {
            "candidate_witness_count": len(witnesses),
            "identity_dominated_terminal_witness_count": identity_dominated_count,
        }
        if provenance_grouped_terminal and len(answer_groups) > 1:
            provenance_groups = {
                terminal_provenance(row) for row in candidate_witnesses
            }
            if len(provenance_groups) == 1:
                witness = dict(max(candidate_witnesses, key=witness_rank))
                answer = str(
                    ((witness.get("slot_bindings") or {}).get("$answer") or {}).get(
                        "surface"
                    )
                    or ""
                ).strip()
                return witness, "provenance_grouped_projection", {
                    **reduction_detail,
                    "answer": answer,
                    "distinct_terminal_value_count": len(answer_groups),
                    "co_provenant_terminal_group_count": 1,
                    "terminal_values": sorted(answer_groups),
                    "selection_policy": (
                        "identity_dominance_then_provenance_group_specificity"
                    ),
                }
            provisional_witness = dict(max(candidate_witnesses, key=witness_rank))
            return None, "ambiguous_terminal_conflict", {
                **reduction_detail,
                "distinct_terminal_value_count": len(answer_groups),
                "cross_provenance_terminal_conflict_count": 1,
                "terminal_values": sorted(answer_groups),
                "selection_policy": "cross_provenance_conflict_quarantine",
                "provisional_witness": provisional_witness,
            }
        if conflict_safe_terminal and len(answer_groups) > 1:
            return None, "ambiguous_terminal_conflict", {
                **reduction_detail,
                "distinct_terminal_value_count": len(answer_groups),
                "terminal_values": sorted(answer_groups),
                "selection_policy": "conflict_safe_reader_fallback",
            }

        witness = dict(max(candidate_witnesses, key=witness_rank))
        answer = str(
            ((witness.get("slot_bindings") or {}).get("$answer") or {}).get(
                "surface"
            )
            or ""
        ).strip()
        return (
            (witness if answer else None),
            ("projected_answer" if answer else "answer_slot_empty"),
            {
                **reduction_detail,
                "answer": answer,
                "selection_policy": "identity_consistency_then_specificity",
            },
        )

    ordered_roots = [str(value) for value in root_binding_ids if str(value)]
    groups = _branch_witness_groups(witnesses)
    if len(ordered_roots) != 2 or any(root not in groups for root in ordered_roots):
        return None, "incomplete_verified_branches", {
            "required_root_count": 2,
            "verified_root_count": sum(root in groups for root in ordered_roots),
        }

    value_rows: Dict[str, List[Tuple[Dict[str, Any], str]]] = {}
    for root_id in ordered_roots:
        rows = []
        for witness in groups[root_id]:
            value = str(
                ((witness.get("slot_bindings") or {}).get("$comparison_value") or {}).get(
                    "surface"
                )
                or ""
            ).strip()
            if value:
                rows.append((witness, value))
        if not rows:
            return None, "comparison_value_missing", {"root_binding_id": root_id}
        value_rows[root_id] = rows

    chosen_rows: List[Dict[str, Any]] = []
    answer = ""
    reduction_detail: Dict[str, Any] = {
        "answer_operator": answer_operator,
        "branch_values": {
            root_id: [value for _, value in rows]
            for root_id, rows in value_rows.items()
        },
    }
    if answer_operator == "compare_eq":
        keyed_values = {
            root_id: {_key(value): row for row, value in rows if _key(value)}
            for root_id, rows in value_rows.items()
        }
        shared = sorted(
            set(keyed_values[ordered_roots[0]])
            & set(keyed_values[ordered_roots[1]])
        )
        if not shared:


            return None, "no_positive_equality_witness", reduction_detail
        shared_key = shared[0]
        chosen_rows = [keyed_values[root][shared_key] for root in ordered_roots]
        answer = "yes"
        reduction_detail["shared_value_key"] = shared_key
    elif answer_operator in {"argmin_root", "argmax_root"}:
        from .relation_fact_join_utils import _temporal_value_key

        temporal_rows: Dict[
            str, List[Tuple[Tuple[int, int, int], Dict[str, Any], str]]
        ] = {}
        for root_id, rows in value_rows.items():
            parsed = [
                (key, row, value)
                for row, value in rows
                if (key := _temporal_value_key(value)) is not None
            ]
            if not parsed:
                return None, "temporal_value_unparseable", reduction_detail
            temporal_rows[root_id] = parsed
        first_keys = [item[0] for item in temporal_rows[ordered_roots[0]]]
        second_keys = [item[0] for item in temporal_rows[ordered_roots[1]]]
        if answer_operator == "argmin_root":
            winner_index = (
                0 if max(first_keys) < min(second_keys)
                else 1 if max(second_keys) < min(first_keys)
                else -1
            )
            choose = min
        else:
            winner_index = (
                0 if min(first_keys) > max(second_keys)
                else 1 if min(second_keys) > max(first_keys)
                else -1
            )
            choose = max
        if winner_index < 0:
            return None, "temporal_branch_order_ambiguous", reduction_detail
        winner_root = ordered_roots[winner_index]
        chosen_rows = [
            choose(temporal_rows[root_id], key=lambda item: item[0])[1]
            for root_id in ordered_roots
        ]
        winner = chosen_rows[winner_index]
        answer = str(
            winner.get("root_surface")
            or (winner.get("root_metadata") or {}).get("mention_surface")
            or (winner.get("root_metadata") or {}).get("surface")
            or ""
        ).strip()
        reduction_detail["winner_root_binding_id"] = winner_root
    else:
        return None, "unsupported_answer_operator", reduction_detail

    if not answer:
        return None, "reducer_answer_empty", reduction_detail
    fact_usage_ids = list(dict.fromkeys(
        fact_id
        for row in chosen_rows
        for fact_id in (
            row.get("path_fact_usage_ids")
            or row.get("path_fact_ids")
            or []
        )
    ))
    fact_ids = list(dict.fromkeys(
        fact_id
        for row in chosen_rows
        for fact_id in row.get("path_fact_ids") or []
    ))
    combined = {
        "binding_id": "terminal_reduction",
        "parent_binding_id": "",
        "active": True,
        "invalidated": False,
        "slot_bindings": {"$answer": {"surface": answer, "key": _key(answer)}},
        "path_fact_ids": fact_ids,
        "path_fact_usage_ids": fact_usage_ids,
        "next_operator_index": len(logical_plan.get("operators") or []),
        "branch_binding_ids": [str(row.get("binding_id") or "") for row in chosen_rows],
        "root_binding_ids": ordered_roots,
        "terminal_operator": answer_operator,
        "terminal_provenance_mode": "union_input_lineage",
    }
    reduction_detail["answer"] = answer
    reduction_detail["path_fact_ids"] = fact_ids
    reduction_detail["path_fact_usage_ids"] = fact_usage_ids
    return combined, "branch_reduction_complete", reduction_detail


def _pending_verify_actions(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    facts = dict(state.get("facts") or {})
    actions: List[Dict[str, Any]] = []
    seen = set()
    for row in _active_rows(state):
        usage_ids = list(
            row.get("path_fact_usage_ids") or row.get("path_fact_ids") or []
        )
        for usage_id in usage_ids:
            if usage_id in seen:
                continue
            seen.add(usage_id)
            fact = dict(facts.get(usage_id) or {})
            if fact.get("support_status") != UNVERIFIED:
                continue
            actions.append({
                "operator": VERIFY,
                "fact_id": str(fact.get("physical_fact_id") or usage_id),
                "fact_usage_id": usage_id,
                "binding_id": row.get("binding_id"),
                "root_binding_id": row.get("root_binding_id"),
                "obligation_id": str(
                    fact.get("admission_obligation_id")
                    or fact.get("operator_id")
                    or ""
                ),
                "operator_index": max(
                    0, _as_int(row.get("next_operator_index")) - 1
                ),
            })
    return actions


def _operator_reaches_terminal(
    logical_plan: Mapping[str, Any], operator_index: int
) -> bool:
    """Return whether an operator participates in the compiled answer path."""
    operators = list((logical_plan or {}).get("operators") or [])
    if operator_index < 0 or operator_index >= len(operators):
        return False
    for index in range(operator_index, len(operators) - 1):
        output_slot = str(operators[index].get("output_slot") or "")
        next_input_slot = str(operators[index + 1].get("input_slot") or "")
        if output_slot and next_input_slot and output_slot != next_input_slot:
            return False
    return True


def _repair_request_decision(
    state: Mapping[str, Any], request: Mapping[str, Any]
) -> Dict[str, Any]:
    """Decide whether a failed obligation currently blocks a terminal witness."""
    operator_index = _as_int(request.get("operator_index"), -1)
    if not _operator_reaches_terminal(
        state.get("logical_plan") or {}, operator_index
    ):
        return {
            "eligible": False,
            "reason": "skipped_noncritical_obligation",
            "terminal_relevant": False,
            "alternative_binding_ids": [],
        }

    root_binding_id = str(request.get("root_binding_id") or "")
    parent_binding_id = str(request.get("parent_binding_id") or "")
    alternatives = []
    facts = dict(state.get("facts") or {})
    for row in _active_rows(state):
        binding_id = str(row.get("binding_id") or "")
        if binding_id == parent_binding_id:
            continue
        if root_binding_id and str(row.get("root_binding_id") or "") != root_binding_id:
            continue
        if _as_int(row.get("next_operator_index")) <= operator_index:
            continue
        if _row_lineage_state(row, facts) == "blocked":
            continue
        alternatives.append(binding_id)
    if alternatives:
        return {
            "eligible": False,
            "reason": "deferred_live_alternative",
            "terminal_relevant": True,
            "alternative_binding_ids": alternatives,
        }
    return {
        "eligible": True,
        "reason": "scheduled_terminal_blocker",
        "terminal_relevant": True,
        "alternative_binding_ids": [],
    }


def _record_repair_scheduler_event(
    state: Dict[str, Any], request: Mapping[str, Any], decision: Mapping[str, Any]
) -> None:
    request_key = str(request.get("request_key") or "")
    reason = str(decision.get("reason") or "")
    event_key = f"{request_key}:{reason}"
    seen = state.setdefault("repair_scheduler_event_keys", set())
    if event_key in seen:
        return
    seen.add(event_key)
    state.setdefault("repair_scheduler_events", []).append({
        "request_key": request_key,
        "parent_binding_id": str(request.get("parent_binding_id") or ""),
        "root_binding_id": str(request.get("root_binding_id") or ""),
        "obligation_id": str(request.get("obligation_id") or ""),
        "operator_index": _as_int(request.get("operator_index"), -1),
        "source_fact_ids": list(request.get("source_fact_ids") or []),
        "source_statuses": list(request.get("source_statuses") or []),
        **dict(decision),
    })


def _register_bswc_source_candidates(
    state: Dict[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    row: Mapping[str, Any],
    operator: Mapping[str, Any],
    *,
    discovery_action: str,
) -> int:
    """Add physical source alternatives without exposing passage content."""
    obligation_id = str(operator.get("operator_id") or "")
    binding_id = str(row.get("binding_id") or "")
    operator_index = _as_int(row.get("next_operator_index"), -1)
    table = state.setdefault("bswc_source_candidates", [])
    existing = {
        str(value.get("candidate_key") or ""): value for value in table
    }
    added = 0
    for raw in candidates or []:
        source_id = _as_int(raw.get("source_id", raw.get("doc_id")), -1)
        if source_id < 0 or not binding_id or not obligation_id:
            continue
        candidate_key = f"{binding_id}:{obligation_id}:{source_id}"
        candidate = {
            key: _json_safe(value)
            for key, value in dict(raw).items()
            if key != "content"
        }
        candidate.update({
            "candidate_key": candidate_key,
            "source_id": source_id,
            "doc_id": source_id,
            "binding_id": binding_id,
            "obligation_id": obligation_id,
            "operator_index": operator_index,
            "coverage_key": f"{binding_id}:{obligation_id}",
            "root_binding_id": str(row.get("root_binding_id") or binding_id),
            "discovery_action": str(discovery_action),
        })
        if candidate_key in existing:
            current = existing[candidate_key]
            if _as_float(candidate.get("score")) > _as_float(current.get("score")):
                current.update(candidate)
            continue
        table.append(candidate)
        existing[candidate_key] = candidate
        added += 1
    return added


def _bswc_source_actions(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    plan = dict(state.get("logical_plan") or {})
    operators = list(plan.get("operators") or [])
    attempted = set(state.get("attempted_action_keys") or [])
    active = {
        str(row.get("binding_id") or ""): row for row in _active_rows(state)
    }
    eligible_candidates = []
    for raw in state.get("bswc_source_candidates") or []:
        candidate = dict(raw)
        binding_id = str(candidate.get("binding_id") or "")
        row = active.get(binding_id)
        if row is None:
            continue
        operator_index = _as_int(candidate.get("operator_index"), -1)
        if operator_index != _as_int(row.get("next_operator_index"), -2):
            continue
        if operator_index < 0 or operator_index >= len(operators):
            continue
        eligible_candidates.append(candidate)

    binding_qualified_policy = str(state.get("bswc_policy") or "") in {
        "bswc_fixed_source_order",
        "bswc_binding_qualified_continuation_cover",
        "bswc_rank_consistent_binding_qualified_cover",
    }
    coverage_by_source: Dict[int, set[str]] = {}
    qualified_coverage_by_source_identity: Dict[Tuple[int, str], set[str]] = {}
    for candidate in eligible_candidates:
        source_id = _as_int(candidate.get("source_id"), -1)
        if source_id >= 0:
            coverage_key = str(candidate.get("coverage_key") or "")
            coverage_by_source.setdefault(source_id, set()).add(coverage_key)
            bound_key = str(candidate.get("bound_key") or "")
            candidate_entity_key = str(
                candidate.get("candidate_entity_key") or ""
            )
            identity_qualified = bool(
                bound_key
                and candidate_entity_key
                and bound_key == candidate_entity_key
                and str(candidate.get("source_type") or "")
                == "title_entity_catalog"
            )
            if identity_qualified:
                qualified_coverage_by_source_identity.setdefault(
                    (source_id, candidate_entity_key), set()
                ).add(coverage_key)

    actions: List[Dict[str, Any]] = []
    for candidate in eligible_candidates:
        source_id = _as_int(candidate.get("source_id"), -1)
        bound_key = str(candidate.get("bound_key") or "")
        candidate_entity_key = str(candidate.get("candidate_entity_key") or "")
        identity_qualified = bool(
            bound_key
            and candidate_entity_key
            and bound_key == candidate_entity_key
            and str(candidate.get("source_type") or "")
            == "title_entity_catalog"
        )
        own_coverage_key = str(candidate.get("coverage_key") or "")
        if binding_qualified_policy:
            predicted_coverage = (
                qualified_coverage_by_source_identity.get(
                    (source_id, candidate_entity_key), set()
                )
                if identity_qualified else {own_coverage_key}
            )
        else:
            predicted_coverage = coverage_by_source.get(source_id, set())
        source_candidate = dict(candidate)
        if binding_qualified_policy:
            source_candidate.update({
                "binding_identity_qualified": identity_qualified,
                "binding_identity_key": (
                    candidate_entity_key if identity_qualified else ""
                ),
                "continue_sentence_repair_after_indexed_candidates": True,
            })
        action = {
            "operator": SOURCE_INSPECT,
            "binding_id": str(candidate.get("binding_id") or ""),
            "obligation_id": str(candidate.get("obligation_id") or ""),
            "operator_index": _as_int(candidate.get("operator_index"), -1),
            "source_id": source_id,
            "source_type": str(candidate.get("source_type") or "unknown"),
            "source_rank": _as_int(candidate.get("rank"), 0),
            "source_relevance": _as_float(candidate.get("score"), 0.0),
            "source_candidate": source_candidate,
            "predicted_coverage_keys": sorted(
                value for value in predicted_coverage if value
            ),
            "binding_compatible_coverage": len(
                {value for value in predicted_coverage if value}
            ),
            "binding_identity_qualified": identity_qualified,
            "binding_identity_key": (
                candidate_entity_key if identity_qualified else ""
            ),
            "binding_qualified_coverage": binding_qualified_policy,
            "source_reused": source_id in {
                _as_int(value, -1)
                for value in state.get("bswc_inspected_source_ids") or []
            },
        }
        if _action_key(action) not in attempted:
            actions.append(action)

    if not state.get("completion_enabled", True):
        return actions
    for binding_id, row in active.items():
        operator_index = _as_int(row.get("next_operator_index"), -1)
        if operator_index < 0 or operator_index >= len(operators):
            continue
        operator = operators[operator_index]
        obligation_id = str(operator.get("operator_id") or f"op{operator_index}")
        lookup_key = f"{LOOKUP_JOIN}:{binding_id}:{obligation_id}"
        if lookup_key not in attempted:
            continue
        if _as_int((state.get("lookup_match_counts") or {}).get(lookup_key)) != 0:
            continue
        dense_action = {
            "operator": BOUND_DENSE_PROBE,
            "binding_id": binding_id,
            "obligation_id": obligation_id,
            "operator_index": operator_index,
            "predicted_coverage_keys": [f"{binding_id}:{obligation_id}"],
            "binding_compatible_coverage": 1,
            "source_relevance": 0.55,
        }
        if _action_key(dense_action) not in attempted:
            actions.append(dense_action)
    return actions


def _candidate_actions(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    plan = dict(state.get("logical_plan") or {})
    operators = list(plan.get("operators") or [])
    actions = _pending_verify_actions(state)
    attempted = set(state.get("attempted_action_keys") or [])
    for row in _active_rows(state):
        if row.get("overlay_reuse_blocked"):
            continue
        index = _as_int(row.get("next_operator_index"))
        if index >= len(operators):
            continue
        operator = dict(operators[index])
        obligation_id = str(operator.get("operator_id") or f"op{index}")
        lookup_key = f"{LOOKUP_JOIN}:{row['binding_id']}:{obligation_id}"
        if lookup_key not in attempted:
            actions.append({
                "operator": LOOKUP_JOIN,
                "binding_id": row["binding_id"],
                "obligation_id": obligation_id,
                "operator_index": index,
            })
            continue
        if state.get("bswc_enabled"):
            continue
        complete_key = f"{COMPLETE}:{row['binding_id']}:{obligation_id}"
        if (
            state.get("completion_enabled", True)
            and complete_key not in attempted
            and (
                state.get("completion_mode") == "eager"
                or _as_int(
                    (state.get("lookup_match_counts") or {}).get(lookup_key)
                ) == 0
            )
        ):
            if (
                state.get("repair_scheduler_mode") == "terminal_critical"
                and not _operator_reaches_terminal(plan, index)
            ):
                continue
            actions.append({
                "operator": COMPLETE,
                "binding_id": row["binding_id"],
                "obligation_id": obligation_id,
                "operator_index": index,
            })
    if (
        state.get("repair_scheduler_mode") == "terminal_critical"
        and not state.get("bswc_enabled")
    ):
        for request in (state.get("repair_requests") or {}).values():
            if str(request.get("status") or "pending") != "pending":
                continue
            complete_key = ":".join((
                COMPLETE,
                str(request.get("parent_binding_id") or ""),
                str(request.get("obligation_id") or ""),
            ))
            if complete_key in attempted:
                continue
            decision = _repair_request_decision(state, request)
            _record_repair_scheduler_event(state, request, decision)
            if not decision.get("eligible"):
                if decision.get("reason") == "skipped_noncritical_obligation":
                    request["status"] = "skipped_noncritical"
                continue
            actions.append({
                "operator": COMPLETE,
                "binding_id": str(request.get("parent_binding_id") or ""),
                "obligation_id": str(request.get("obligation_id") or ""),
                "operator_index": _as_int(request.get("operator_index")),
                "verification_repair_required": True,
                "repair_request_key": str(request.get("request_key") or ""),
                "repair_scheduler_reason": str(decision.get("reason") or ""),
                "rejected_index_fact_ids": list(
                    request.get("source_fact_ids") or []
                ),
            })
    if state.get("bswc_enabled"):
        source_actions = _bswc_source_actions(state)
        actions.extend(source_actions)
        if (
            state.get("bswc_policy")
            == "bswc_identity_safe_scan_reservation_oracle"
        ):



            exposed, exposure_detail = expose_identity_safe_source_actions(state)
            for action in exposed:
                action["frontier_exposure_detail"] = dict(exposure_detail)
            actions.extend(exposed)
            actions, reservation_detail = (
                annotate_identity_safe_budget_actions(actions, state)
            )
            for action in actions:
                if action.get("identity_safe_budget_reservation"):
                    action["identity_safe_budget_detail"] = dict(
                        reservation_detail
                    )
        if (
            state.get("bswc_policy")
            == "bswc_identity_safe_frontier_exposure_oracle"
        ):
            exposed, exposure_detail = expose_identity_safe_source_actions(state)
            for action in exposed:
                action["frontier_exposure_detail"] = dict(exposure_detail)
            actions.extend(exposed)
    return actions


def _action_key(action: Mapping[str, Any]) -> str:
    if str(action.get("operator") or "") == VERIFY:
        return f"{VERIFY}:{action.get('fact_usage_id') or action.get('fact_id', '')}"
    if str(action.get("operator") or "") == SOURCE_INSPECT:
        return ":".join(str(action.get(key, "")) for key in (
            "operator", "binding_id", "obligation_id", "source_id"
        ))
    return ":".join(str(action.get(key, "")) for key in (
        "operator", "binding_id", "obligation_id"
    ))


def _dependency_score(
    action: Mapping[str, Any], estimate: Mapping[str, Any], state: Mapping[str, Any]
) -> Tuple[float, Dict[str, float]]:
    operator = str(action.get("operator") or "")
    progress = _bounded(estimate.get("expected_progress"))
    support_probability = _bounded(estimate.get("support_probability"), 0.0, 1.0)
    downstream_cost = max(0.0, _as_float(estimate.get("downstream_cost")))
    expected_pruning = _bounded(estimate.get("expected_pruning"))
    unlock_value = max(0.0, _as_float(estimate.get("unlock_value"), progress))
    scan_cost = min(1.0, _as_float(estimate.get("expected_scan_rows")) / 500.0)
    call_cost = 0.35 * _as_int(estimate.get("expected_retrieval_calls"))
    call_cost += 0.55 * _as_int(estimate.get("expected_model_calls"))
    components = {
        "progress": progress,
        "unlock_value": unlock_value,
        "expected_avoided_downstream": 0.0,
        "expected_pruning": expected_pruning,
        "scan_cost": scan_cost,
        "call_cost": call_cost,
    }
    if operator == VERIFY:
        components["expected_avoided_downstream"] = (
            (1.0 - support_probability) * downstream_cost
        )
        score = (
            0.65 * progress
            + components["expected_avoided_downstream"]
            - scan_cost
            - call_cost
        )
    elif operator == COMPLETE:
        score = 1.20 * unlock_value + 0.35 * progress - scan_cost - call_cost
    else:
        score = (
            0.85 * progress
            + 0.85 * expected_pruning
            + 0.25 * unlock_value
            - scan_cost
            - call_cost
        )
    return round(score, 8), components


def _action_rows(
    action: Mapping[str, Any], state: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """Return the live lineages whose feasibility depends on an action."""
    if str(action.get("operator") or "") == VERIFY:
        fact_id = str(
            action.get("fact_usage_id") or action.get("fact_id") or ""
        )
        return [
            row for row in _active_rows(state)
            if fact_id in set(
                row.get("path_fact_usage_ids")
                or row.get("path_fact_ids")
                or []
            )
        ]
    binding_id = str(action.get("binding_id") or "")
    return [
        row for row in _active_rows(state)
        if str(row.get("binding_id") or "") == binding_id
    ]


def _row_lineage_state(
    row: Mapping[str, Any], facts: Mapping[str, Any]
) -> str:
    fact_ids = list(
        row.get("path_fact_usage_ids") or row.get("path_fact_ids") or []
    )
    if not fact_ids:
        return "root"
    statuses = [
        str((facts.get(fact_id) or {}).get("support_status") or UNVERIFIED)
        for fact_id in fact_ids
    ]
    if any(status in {CONTRADICTED, INSUFFICIENT, UNKNOWN} for status in statuses):
        return "blocked"
    if all(status == SUPPORTED for status in statuses):
        return "verified"
    return "unverified"


def _constrained_action_frontier(
    actions: Sequence[Mapping[str, Any]], state: Mapping[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Keep only actions that can contribute to a complete verified plan.

    The constraint is deliberately lexicographic. Branch coverage and verified
    lineage feasibility are enforced before utility/cost ranking is consulted.
    """
    facts = dict(state.get("facts") or {})
    plan = dict(state.get("logical_plan") or {})
    branch_plan = str(plan.get("plan_kind") or "").startswith("branch_")
    candidates = [dict(action) for action in actions]
    detail: Dict[str, Any] = {
        "input_action_count": len(candidates),
        "constraint_sequence": [],
    }

    if branch_plan and candidates:
        witness_roots = {
            str(row.get("root_binding_id") or "")
            for row in _structured_witnesses(state)
            if str(row.get("root_binding_id") or "")
        }
        missing_roots = [
            str(root_id) for root_id in state.get("root_binding_ids") or []
            if str(root_id) and str(root_id) not in witness_roots
        ]
        rooted_actions: List[Tuple[Dict[str, Any], set[str]]] = []
        for action in candidates:
            roots = {
                str(row.get("root_binding_id") or "")
                for row in _action_rows(action, state)
                if str(row.get("root_binding_id") or "")
            }
            rooted_actions.append((action, roots))
        missing_actions = [
            action for action, roots in rooted_actions
            if roots.intersection(missing_roots)
        ]
        if missing_actions:
            candidates = missing_actions
            detail["constraint_sequence"].append("unresolved_branch_roots_only")

        root_progress: Dict[str, int] = {}
        for root_id in missing_roots:
            stages = [
                _as_int(row.get("next_operator_index"))
                for row in _active_rows(state)
                if str(row.get("root_binding_id") or "") == root_id
            ]
            if stages:
                root_progress[root_id] = max(stages)
        action_roots = {
            root
            for action in candidates
            for row in _action_rows(action, state)
            if (root := str(row.get("root_binding_id") or ""))
        }
        available_progress = {
            root: progress for root, progress in root_progress.items()
            if root in action_roots
        }
        if available_progress:
            minimum_progress = min(available_progress.values())
            lagging_roots = {
                root for root, progress in available_progress.items()
                if progress == minimum_progress
            }
            lagging_actions = [
                action for action in candidates
                if any(
                    str(row.get("root_binding_id") or "") in lagging_roots
                    for row in _action_rows(action, state)
                )
            ]
            if lagging_actions:
                candidates = lagging_actions
                detail["constraint_sequence"].append(
                    "least_advanced_branch_frontier"
                )
                detail["lagging_root_ids"] = sorted(lagging_roots)
                detail["minimum_root_progress"] = minimum_progress

    feasible: List[Dict[str, Any]] = []
    deferred_unverified = 0
    blocked = 0
    for action in candidates:
        operator = str(action.get("operator") or "")
        if operator == VERIFY:
            feasible.append(action)
            continue
        rows = _action_rows(action, state)
        lineage_states = {_row_lineage_state(row, facts) for row in rows}
        if "blocked" in lineage_states:
            blocked += 1
            continue
        if _as_int(action.get("operator_index")) > 0 and "unverified" in lineage_states:
            deferred_unverified += 1
            continue
        feasible.append(action)
    candidates = feasible
    detail["deferred_unverified_lineage_count"] = deferred_unverified
    detail["blocked_lineage_count"] = blocked
    if deferred_unverified:
        detail["constraint_sequence"].append("verify_before_downstream_join")

    def feasibility_tier(action: Mapping[str, Any]) -> int:
        operator = str(action.get("operator") or "")
        rows = _action_rows(action, state)
        lineage_states = {_row_lineage_state(row, facts) for row in rows}
        if operator == LOOKUP_JOIN and "verified" in lineage_states:
            return 0
        if operator == VERIFY:
            return 1
        if operator == LOOKUP_JOIN:
            return 2
        if operator == COMPLETE:
            return 3
        return 9

    if candidates:
        minimum_tier = min(feasibility_tier(action) for action in candidates)
        tier_actions = [
            action for action in candidates
            if feasibility_tier(action) == minimum_tier
        ]
        if tier_actions:
            candidates = tier_actions
        detail["feasibility_tier"] = minimum_tier
        detail["constraint_sequence"].append("feasibility_before_cost")
    detail["eligible_action_count"] = len(candidates)
    detail["deferred_action_count"] = max(0, len(actions) - len(candidates))
    return candidates, detail


def _greedy_score(
    action: Mapping[str, Any], estimate: Mapping[str, Any]
) -> float:
    progress = _bounded(estimate.get("expected_progress"))
    scan_cost = min(1.0, _as_float(estimate.get("expected_scan_rows")) / 500.0)
    calls = _as_int(estimate.get("expected_retrieval_calls"))
    calls += _as_int(estimate.get("expected_model_calls"))
    return round(progress - 0.15 * scan_cost - 0.20 * calls, 8)


def _choose_action(
    *,
    policy: str,
    actions: Sequence[Mapping[str, Any]],
    estimates: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if not actions:
        return None, {"reason": "no_executable_action", "ranked": []}

    constraint_detail: Dict[str, Any] = {}
    if policy == "feasibility_constrained_adaptive":
        actions, constraint_detail = _constrained_action_frontier(actions, state)
        if not actions:
            return None, {
                "reason": "no_feasible_action",
                "ranked": [],
                "feasibility_constraints": constraint_detail,
            }

    bounded_selected: Optional[Mapping[str, Any]] = None
    bounded_detail: Dict[str, Any] = {}
    if policy in {"bswc_bounded_search", "bswc_full"}:
        bounded_selected, bounded_detail = binding_aware_bounded_first_action(
            actions,
            estimates,
            state,
            beam_width=_as_int(state.get("bswc_beam_width"), 8),
            horizon=_as_int(state.get("bswc_search_horizon"), 3),
        )
    elif policy == "bswc_risk_aware_search":
        bounded_selected, bounded_detail = risk_aware_witness_search_first_action(
            actions,
            estimates,
            state,
            beam_width=_as_int(state.get("bswc_beam_width"), 16),
            horizon=_as_int(state.get("bswc_search_horizon"), 3),
        )
    elif policy == "bswc_adaptive_risk_cover":
        bounded_selected, bounded_detail = (
            adaptive_risk_bounded_witness_first_action(
                actions,
                estimates,
                state,
                beam_width=_as_int(state.get("bswc_beam_width"), 16),
                horizon=_as_int(state.get("bswc_search_horizon"), 4),
                partial_value=_as_float(
                    state.get("bswc_arwc_partial_value"), 0.10
                ),
                budget_penalty=_as_float(
                    state.get("bswc_arwc_budget_penalty"), 0.05
                ),
                min_utility=_as_float(
                    state.get("bswc_arwc_min_utility"), 0.0
                ),
                min_exploration_actions=_as_int(
                    state.get("bswc_min_exploration_actions"), 1
                ),
            )
        )
    elif policy == "bswc_feasibility_preserving_cover":
        bounded_selected, bounded_detail = (
            feasibility_preserving_witness_first_action(
                actions,
                estimates,
                state,
                beam_width=_as_int(state.get("bswc_beam_width"), 16),
                horizon=_as_int(state.get("bswc_search_horizon"), 4),
                stop_upper_bound=_as_float(
                    state.get("bswc_fpawc_stop_upper_bound"), 0.0
                ),
                min_exploration_actions=_as_int(
                    state.get("bswc_min_exploration_actions"), 1
                ),
            )
        )
    elif policy == "bswc_continuation_safe_cover":
        bounded_selected, bounded_detail = (
            continuation_safe_witness_first_action(
                actions,
                estimates,
                state,
                beam_width=_as_int(state.get("bswc_beam_width"), 16),
                horizon=_as_int(state.get("bswc_search_horizon"), 4),
                stop_upper_bound=_as_float(
                    state.get("bswc_fpawc_stop_upper_bound"), 0.0
                ),
                min_exploration_actions=_as_int(
                    state.get("bswc_min_exploration_actions"), 1
                ),
            )
        )
    elif policy == "bswc_binding_qualified_continuation_cover":
        bounded_selected, bounded_detail = (
            binding_qualified_continuation_first_action(
                actions,
                estimates,
                state,
                beam_width=_as_int(state.get("bswc_beam_width"), 16),
                horizon=_as_int(state.get("bswc_search_horizon"), 4),
                stop_upper_bound=_as_float(
                    state.get("bswc_fpawc_stop_upper_bound"), 0.0
                ),
                min_exploration_actions=_as_int(
                    state.get("bswc_min_exploration_actions"), 1
                ),
            )
        )
    elif policy == "bswc_rank_consistent_binding_qualified_cover":
        bounded_selected, bounded_detail = (
            binding_qualified_continuation_first_action(
                actions,
                estimates,
                state,
                beam_width=_as_int(state.get("bswc_beam_width"), 16),
                horizon=_as_int(state.get("bswc_search_horizon"), 4),
                stop_upper_bound=_as_float(
                    state.get("bswc_fpawc_stop_upper_bound"), 0.0
                ),
                min_exploration_actions=_as_int(
                    state.get("bswc_min_exploration_actions"), 1
                ),
                rank_consistent_source_selection=True,
            )
        )
    elif policy == "bswc_precedence_aware_cover":
        bounded_selected, bounded_detail = precedence_aware_witness_first_action(
            actions,
            estimates,
            state,
            beam_width=_as_int(state.get("bswc_beam_width"), 16),
            horizon=_as_int(state.get("bswc_search_horizon"), 4),
            stop_upper_bound=_as_float(
                state.get("bswc_fpawc_stop_upper_bound"), 0.0
            ),
            min_exploration_actions=_as_int(
                state.get("bswc_min_exploration_actions"), 1
            ),
        )
    elif policy == "bswc_quality_first_precedence_cover":
        bounded_selected, bounded_detail = (
            quality_first_precedence_witness_first_action(
                actions,
                estimates,
                state,
                beam_width=_as_int(state.get("bswc_beam_width"), 16),
                horizon=_as_int(state.get("bswc_search_horizon"), 4),
                stop_upper_bound=_as_float(
                    state.get("bswc_fpawc_stop_upper_bound"), 0.0
                ),
                min_exploration_actions=_as_int(
                    state.get("bswc_min_exploration_actions"), 1
                ),
            )
        )
    elif policy == "bswc_global_witness_reservation_cover":
        bounded_selected, bounded_detail = (
            quality_first_precedence_witness_first_action(
                actions,
                estimates,
                state,
                beam_width=_as_int(state.get("bswc_beam_width"), 16),
                horizon=_as_int(state.get("bswc_search_horizon"), 4),
                stop_upper_bound=_as_float(
                    state.get("bswc_fpawc_stop_upper_bound"), 0.0
                ),
                min_exploration_actions=_as_int(
                    state.get("bswc_min_exploration_actions"), 1
                ),
                global_witness_reservation=True,
            )
        )
    elif policy == "bswc_exact_adaptive_witness":
        bounded_selected, bounded_detail = (
            exact_adaptive_supported_witness_first_action(
                actions,
                estimates,
                state,
            )
        )
    elif policy == "bswc_fixed_source_order":
        bounded_selected, bounded_detail = deterministic_fixed_order_first_action(
            actions, state
        )
    elif policy == "bswc_random_source":
        bounded_selected, bounded_detail = deterministic_random_source_first_action(
            actions, state
        )
    elif policy == "bswc_support_oracle_early":
        bounded_selected, bounded_detail = support_oracle_first_action(
            actions, estimates, state
        )
    elif policy == "bswc_identity_safe_frontier_exposure_oracle":
        exposed = [
            dict(action) for action in actions
            if action.get("identity_safe_frontier_exposure")
        ]
        if exposed:
            ranked_exposed = []
            for position, action in enumerate(exposed):
                score, components = source_action_score(
                    "bswc_set_cover_greedy",
                    action,
                    estimates.get(str(action.get("action_key") or ""), {}),
                    state,
                )
                ranked_exposed.append((score, -position, action, components))
            ranked_exposed.sort(
                key=lambda item: (item[0], item[1]), reverse=True
            )
            bounded_selected = dict(ranked_exposed[0][2])
            bounded_detail = {
                **dict(
                    bounded_selected.get("frontier_exposure_detail") or {}
                ),
                "reason": "selected_identity_safe_exposed_source",
                "selected_exposure_action_key": str(
                    bounded_selected.get("action_key") or ""
                ),
                "selected_exposure_record_id": str(
                    bounded_selected.get("frontier_exposure_record_id") or ""
                ),
                "exposed_action_count": len(exposed),
                "selection_policy": "frozen_supported_source_upper_bound",
                "candidate_action_injected": True,
                "tuple_injected": False,
                "qa_gold_used_for_source_selection": False,
            }
    elif policy == "bswc_identity_safe_scan_reservation_oracle":
        budget_targets = [
            dict(action) for action in actions
            if action.get("identity_safe_budget_reservation")
        ]
        if budget_targets:



            budget_targets.sort(key=lambda action: (
                0
                if action.get("identity_safe_budget_target_source")
                else 1
                if action.get("identity_safe_budget_target_verify")
                else 2,
                str(action.get("action_key") or ""),
            ))
            bounded_selected = dict(budget_targets[0])
            bounded_detail = {
                **dict(
                    bounded_selected.get("identity_safe_budget_detail") or {}
                ),
                "reason": "selected_identity_safe_budget_target",
                "selected_exposure_action_key": str(
                    bounded_selected.get("action_key") or ""
                ),
                "selected_exposure_record_id": str(
                    bounded_selected.get("frontier_exposure_record_id") or ""
                ),
                "selected_target_kind": (
                    "source"
                    if bounded_selected.get("identity_safe_budget_target_source")
                    else "verify"
                    if bounded_selected.get("identity_safe_budget_target_verify")
                    else "lookup"
                ),
                "selection_policy": (
                    "frozen_identity_safe_budget_allocation_upper_bound"
                ),
                "candidate_action_injected": bool(
                    bounded_selected.get("identity_safe_frontier_exposure")
                ),
                "tuple_injected": False,
                "qa_gold_used_for_source_selection": False,
            }

    complete_rows = [
        row for row in _active_rows(state)
        if _row_complete(row, len((state.get("logical_plan") or {}).get("operators") or []))
    ]
    ranked: List[Tuple[float, int, Dict[str, Any], Dict[str, Any]]] = []
    priority = {
        "fixed_early_verify": {VERIFY: 0, LOOKUP_JOIN: 1, COMPLETE: 2},
        "fixed_late_verify": (
            {VERIFY: 0, LOOKUP_JOIN: 1, COMPLETE: 2}
            if complete_rows else {LOOKUP_JOIN: 0, COMPLETE: 1, VERIFY: 2}
        ),
    }
    for action in actions:
        action = dict(action)
        action_key = str(action["action_key"])
        estimate = dict(estimates.get(action_key) or {})
        detail: Dict[str, Any] = {}
        if policy in priority:
            order = priority[policy].get(str(action.get("operator")), 99)
            score = float(100 - order)
            detail = {"fixed_priority": order}
        elif policy == "greedy_immediate":
            score = _greedy_score(action, estimate)
            detail = {"greedy_immediate_score": score}
        elif policy in BSWC_POLICIES:
            scoring_policy = (
                "bswc_adaptive_greedy"
                if policy in {"bswc_bounded_search", "bswc_full"}
                else "bswc_set_cover_greedy"
                if policy in {
                    "bswc_identity_safe_frontier_exposure_oracle",
                    "bswc_identity_safe_scan_reservation_oracle",
                }
                else policy
            )
            score, components = source_action_score(
                scoring_policy,
                action,
                estimate,
                state,
            )
            detail = {
                "bswc_score": score,
                "components": components,
            }
        elif policy == "support_aware_cbo":
            score, components = support_aware_action_score(
                action,
                estimate,
                state,
                risk_weight=_as_float(
                    state.get("cbo_risk_weight"), 0.8
                ),
                cost_weight=_as_float(
                    state.get("cbo_cost_weight"), 0.25
                ),
            )
            detail = {
                "support_aware_cbo_score": score,
                "components": components,
                "calibration": {
                    "calibrated": bool(estimate.get("cbo_calibrated")),
                    "backoff_level": str(
                        estimate.get("cbo_backoff_level") or "base_estimate"
                    ),
                    "sample_count": _as_int(estimate.get("cbo_sample_count")),
                    "confidence": _as_float(estimate.get("cbo_confidence")),
                },
            }
        else:
            score, components = _dependency_score(action, estimate, state)
            detail = {"dependency_score": score, "components": components}
        ranked.append((score, -len(ranked), action, detail))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = dict(ranked[0][2])
    if bounded_selected is not None:
        bounded_key = str(bounded_selected.get("action_key") or "")
        bounded_row = next(
            (
                row for row in ranked
                if str(row[2].get("action_key") or "") == bounded_key
            ),
            None,
        )
        if policy in {
            "bswc_bounded_search", "bswc_risk_aware_search",
            "bswc_adaptive_risk_cover",
            "bswc_feasibility_preserving_cover",
            "bswc_continuation_safe_cover",
            "bswc_binding_qualified_continuation_cover",
            "bswc_rank_consistent_binding_qualified_cover",
            "bswc_precedence_aware_cover",
            "bswc_quality_first_precedence_cover",
            "bswc_global_witness_reservation_cover",
            "bswc_exact_adaptive_witness",
            "bswc_fixed_source_order",
            "bswc_random_source",
            "bswc_support_oracle_early",
            "bswc_identity_safe_frontier_exposure_oracle",
            "bswc_identity_safe_scan_reservation_oracle",
        }:
            selected = dict(bounded_selected)
        elif bounded_row is not None:


            if bounded_row[0] >= ranked[0][0] * 0.85:
                selected = dict(bounded_selected)
    frozen_frontier = frozen_source_frontier_snapshot(actions, estimates, state)
    if (
        policy in {
            "bswc_adaptive_risk_cover",
            "bswc_feasibility_preserving_cover",
            "bswc_continuation_safe_cover",
            "bswc_binding_qualified_continuation_cover",
            "bswc_rank_consistent_binding_qualified_cover",
            "bswc_precedence_aware_cover",
            "bswc_quality_first_precedence_cover",
            "bswc_global_witness_reservation_cover",
            "bswc_exact_adaptive_witness",
        }
        and bounded_selected is None
        and bounded_detail.get("stop")
            and (
                policy in {
                    "bswc_exact_adaptive_witness",
                    "bswc_continuation_safe_cover",
                    "bswc_binding_qualified_continuation_cover",
                    "bswc_rank_consistent_binding_qualified_cover",
                }
                or all(
                str(action.get("operator") or "") == SOURCE_INSPECT
                for action in actions
            )
        )
    ):
        return None, {
            "reason": f"selected_stop_by_{policy}",
            "selected_action_key": "STOP",
            "open_coverage_keys": sorted(
                str(value)
                for value in state.get("bswc_open_coverage_keys") or []
            ),
            "feasibility_constraints": constraint_detail,
            "bounded_search": bounded_detail,
            "frozen_source_frontier": frozen_frontier,
            "ranked": [
                {
                    "action_key": item[2]["action_key"],
                    "operator": item[2]["operator"],
                    "score": item[0],
                    **item[3],
                }
                for item in ranked
            ],
        }
    return selected, {
        "reason": f"selected_by_{policy}",
        "selected_action_key": selected["action_key"],
        "open_coverage_keys": sorted(
            str(value) for value in state.get("bswc_open_coverage_keys") or []
        ),
        "feasibility_constraints": constraint_detail,
        "bounded_search": bounded_detail,
        "frozen_source_frontier": frozen_frontier,
        "ranked": [
            {
                "action_key": item[2]["action_key"],
                "operator": item[2]["operator"],
                "score": item[0],
                **item[3],
            }
            for item in ranked
        ],
    }


def _binding_by_id(state: Mapping[str, Any], binding_id: str) -> Dict[str, Any]:
    return next(
        row for row in state.get("bindings") or []
        if str(row.get("binding_id")) == str(binding_id)
    )


def _effective_rejected_fact_ids(
    binding: Mapping[str, Any],
    obligation_id: str,
    additional_fact_ids: Sequence[Any] = (),
) -> List[str]:
    """Return physical facts rejected for this exact binding use.

    A failed source-support certificate is assertion-wide and remains in the
    legacy ``rejected_index_fact_ids`` set.  An obligation-admission failure is
    narrower: it only blocks the same physical fact for the obligation whose
    direction, role, subtype, or output contract rejected it.
    """
    contextual = dict(
        binding.get("rejected_fact_ids_by_obligation") or {}
    )
    return list(dict.fromkeys(
        str(value)
        for value in (
            list(binding.get("rejected_index_fact_ids") or [])
            + list(contextual.get(str(obligation_id or "")) or [])
            + list(additional_fact_ids or [])
        )
        if str(value)
    ))


def _record_binding_fact_rejection(
    binding: Dict[str, Any],
    *,
    obligation_id: str,
    physical_fact_id: str,
    assertion_wide: bool,
) -> None:
    """Persist a rejection at assertion or query-usage scope."""
    fact_id = str(physical_fact_id or "")
    if not fact_id:
        return
    if assertion_wide:
        binding["rejected_index_fact_ids"] = list(dict.fromkeys(
            list(binding.get("rejected_index_fact_ids") or []) + [fact_id]
        ))
        return
    contextual = dict(
        binding.get("rejected_fact_ids_by_obligation") or {}
    )
    key = str(obligation_id or "")
    contextual[key] = list(dict.fromkeys(
        list(contextual.get(key) or []) + [fact_id]
    ))
    binding["rejected_fact_ids_by_obligation"] = contextual


def _invalidate_dependent_rows(
    state: Dict[str, Any], fact_usage_id: str
) -> List[str]:
    invalidated: List[str] = []
    for row in state.get("bindings") or []:
        usage_ids = set(
            row.get("path_fact_usage_ids") or row.get("path_fact_ids") or []
        )
        if fact_usage_id in usage_ids and not row.get("invalidated"):
            row["invalidated"] = True
            row["active"] = False
            row["invalidation_reason"] = (
                f"rejected_fact_usage:{fact_usage_id}"
            )
            invalidated.append(str(row.get("binding_id") or ""))
    return invalidated


def _restart_root_after_rejection(
    state: Dict[str, Any],
    *,
    rejected_fact_id: str,
    affected_rows: Sequence[Mapping[str, Any]],
    obligation_id: str = "",
    assertion_wide: bool = True,
) -> List[str]:
    """Restart one affected root while retaining only its root binding.

    This is an intentionally coarser baseline for the SAQE local-repair
    transition.  The rejected fact id is carried into the restarted lineage so
    the same bad index tuple is filtered and the ordinary miss -> COMPLETE path
    remains available under the same budget.
    """
    root_ids = list(dict.fromkeys(
        str(row.get("root_binding_id") or "")
        for row in affected_rows
        if str(row.get("root_binding_id") or "")
    ))
    restarted_ids: List[str] = []
    for old_root_id in root_ids:
        root = next(
            (
                row for row in state.get("bindings") or []
                if str(row.get("binding_id") or "") == old_root_id
            ),
            None,
        )
        if root is None:
            continue
        rejected_ids = list(root.get("rejected_index_fact_ids") or [])
        contextual_rejections = dict(
            root.get("rejected_fact_ids_by_obligation") or {}
        )
        if assertion_wide:
            rejected_ids = list(dict.fromkeys(
                rejected_ids + [str(rejected_fact_id)]
            ))
        else:
            key = str(obligation_id or "")
            contextual_rejections[key] = list(dict.fromkeys(
                list(contextual_rejections.get(key) or [])
                + [str(rejected_fact_id)]
            ))
        for row in state.get("bindings") or []:
            if str(row.get("root_binding_id") or "") != old_root_id:
                continue
            row["invalidated"] = True
            row["active"] = False
            row["invalidation_reason"] = (
                f"restart_after_rejected_fact:{rejected_fact_id}"
            )
        root_slots = dict(root.get("slot_bindings") or {})
        root_slot_name = str(
            (((state.get("logical_plan") or {}).get("operators") or [{}])[0])
            .get("input_slot")
            or "$root"
        )
        new_binding_id = f"b{state['next_binding_id']}"
        state["next_binding_id"] += 1
        restarted = {
            "binding_id": new_binding_id,
            "parent_binding_id": "",
            "root_binding_id": new_binding_id,
            "root_surface": str(root.get("root_surface") or ""),
            "root_key": str(root.get("root_key") or ""),
            "active": True,
            "invalidated": False,
            "slot_bindings": {
                root_slot_name: dict(root_slots.get(root_slot_name) or {})
            },
            "path_fact_ids": [],
            "path_fact_usage_ids": [],
            "next_operator_index": 0,
            "root_metadata": dict(root.get("root_metadata") or {}),
            "rejected_index_fact_ids": rejected_ids,
            "rejected_fact_ids_by_obligation": contextual_rejections,
            "verification_repair_required": True,
            "restart_parent_root_binding_id": old_root_id,
        }
        state["bindings"].append(restarted)
        state["root_binding_ids"] = [
            new_binding_id if value == old_root_id else value
            for value in state.get("root_binding_ids") or []
        ]
        restarted_ids.append(new_binding_id)
    return restarted_ids


def _execute_binding_action(
    state: Dict[str, Any], action: Mapping[str, Any], backend: EvidenceBackend
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    row = _binding_by_id(state, str(action["binding_id"]))
    operators = list((state.get("logical_plan") or {}).get("operators") or [])
    operator = dict(operators[_as_int(action.get("operator_index"))])
    operator["execution_scan_limit"] = max(
        0, _as_int(action.get("execution_scan_limit"), 0)
    )
    operator["execution_retrieval_limit"] = max(
        0, _as_int(action.get("execution_retrieval_limit"), 0)
    )
    operator["execution_model_limit"] = max(
        0, _as_int(action.get("execution_model_limit"), 0)
    )
    effective_rejected_fact_ids = _effective_rejected_fact_ids(
        row,
        str(action.get("obligation_id") or operator.get("operator_id") or ""),
        action.get("rejected_index_fact_ids") or [],
    )
    execution_row = {
        **row,
        "rejected_index_fact_ids": effective_rejected_fact_ids,
    }
    if action["operator"] == LOOKUP_JOIN:
        result = dict(backend.lookup_join(execution_row, operator) or {})
        origin = "fact_index"
    elif action["operator"] == BOUND_DENSE_PROBE:
        result = dict(backend.bound_dense_probe(execution_row, operator) or {})
        added = _register_bswc_source_candidates(
            state,
            list(result.get("source_candidates") or []),
            row,
            operator,
            discovery_action=BOUND_DENSE_PROBE,
        )
        result["discovered_source_candidate_count"] = added
        state["bswc_dense_probe_count"] = (
            _as_int(state.get("bswc_dense_probe_count")) + 1
        )
        return result, {
            "created_binding_ids": [],
            "fact_ids": [],
            "logical_progress": bool(added),
            "source_candidate_count_added": added,
            "source_candidate_ids": [
                _as_int(value.get("source_id", value.get("doc_id")), -1)
                for value in result.get("source_candidates") or []
            ],
            "obligation_state_before": OBLIGATION_OPEN,
            "obligation_state_after": OBLIGATION_OPEN,
            "obligation_events": [{
                "obligation_id": str(action.get("obligation_id") or ""),
                "binding_id": str(row.get("binding_id") or ""),
                "from": OBLIGATION_OPEN,
                "to": OBLIGATION_OPEN,
                "cause": BOUND_DENSE_PROBE,
            }],
        }
    elif action["operator"] == SOURCE_INSPECT:
        result = dict(backend.inspect_source(
            execution_row,
            operator,
            dict(action.get("source_candidate") or {}),
        ) or {})
        source_id = _as_int(action.get("source_id"), -1)
        if source_id >= 0:
            inspected = state.setdefault("bswc_inspected_source_ids", set())
            reused = source_id in inspected
            inspected.add(source_id)
            result.setdefault("source_reused", reused)
        origin = "query_local_overlay"
    else:
        result = dict(backend.complete(execution_row, operator) or {})
        origin = "query_local_overlay"
    obligation_before = (
        OBLIGATION_REOPENED
        if row.get("verification_repair_required")
        or row.get("perturbation_repair_required")
        or action.get("verification_repair_required")
        else OBLIGATION_OPEN
    )
    verification_repair_candidate = bool(
        action["operator"] in {COMPLETE, SOURCE_INSPECT}
        and (
            row.get("verification_repair_required")
            or action.get("verification_repair_required")
        )
    )
    perturbation_repair_candidate = bool(
        action["operator"] in {COMPLETE, SOURCE_INSPECT}
        and row.get("perturbation_repair_required")
    )


    known_rejected_fact_ids = {
        str(fact.get("physical_fact_id") or fact_id)
        for fact_id, fact in state["facts"].items()
        if str(fact.get("source_support_status") or "")
        in {CONTRADICTED, INSUFFICIENT}
    }
    known_rejected_fact_ids.update(effective_rejected_fact_ids)
    facts = []
    usage_ids_by_physical: Dict[str, str] = {}
    for raw in result.get("facts") or []:
        fact = _normalise_fact(raw, origin)
        physical_fact_id = str(fact["fact_id"])
        if physical_fact_id in known_rejected_fact_ids:
            continue
        fact.setdefault("operator_id", str(operator.get("operator_id") or ""))
        fact["created_by_action_key"] = str(action.get("action_key") or "")
        fact["verification_repair_candidate"] = verification_repair_candidate
        fact["perturbation_repair_candidate"] = perturbation_repair_candidate
        usage = _register_fact_usage(
            state,
            fact,
            binding_id=str(row.get("binding_id") or ""),
            obligation_id=str(action.get("obligation_id") or ""),
        )
        usage_ids_by_physical[physical_fact_id] = str(usage["fact_usage_id"])
        facts.append(usage)
    matches = list(result.get("matches") or [])
    known_rejected_match_ids = [
        str(match.get("fact_id") or "")
        for match in matches
        if str(match.get("fact_id") or "") in known_rejected_fact_ids
    ]
    created = []
    binding_contract_counts = Counter()
    for match_index, match in enumerate(matches):
        physical_fact_id = str(match.get("fact_id") or "")
        if physical_fact_id in known_rejected_fact_ids:
            continue
        fact_id = str(usage_ids_by_physical.get(physical_fact_id) or "")
        if fact_id not in state["facts"]:
            continue
        binding_contract = _binding_to_obligation_contract(
            state["facts"][fact_id]
        )
        state["facts"][fact_id]["binding_obligation_contract_status"] = (
            binding_contract["status"]
        )
        state["facts"][fact_id]["binding_obligation_contract_reason"] = (
            binding_contract["reason"]
        )
        binding_contract_counts[binding_contract["status"]] += 1
        binding_contract_counts[f"reason:{binding_contract['reason']}"] += 1
        if binding_contract["status"] in {"REJECTED", "DEFERRED"}:
            continue
        output_slot = str(operator.get("output_slot") or "")
        output_surface = str(match.get("output_surface") or "").strip()
        output_key = str(match.get("output_key") or _key(output_surface))
        if not output_slot or not output_surface or not output_key:
            continue
        child = {
            **row,
            "binding_id": f"b{state['next_binding_id']}",
            "parent_binding_id": row["binding_id"],
            "active": True,
            "invalidated": False,
            "slot_bindings": {
                **dict(row.get("slot_bindings") or {}),
                output_slot: {
                    "surface": output_surface,
                    "key": output_key,
                    "fact_id": fact_id,
                    "physical_fact_id": physical_fact_id,
                    "fact_usage_id": fact_id,
                    "source_id": state["facts"][fact_id].get("source_id"),
                    "origin": state["facts"][fact_id].get("origin"),
                    "surface_source": "fact_endpoint",
                    "lineage_key": f"{fact_id}:{output_slot}",
                    "canonical_identity_surface": str(
                        match.get("canonical_identity_surface") or ""
                    ),
                    "canonical_identity_key": str(
                        match.get("canonical_identity_key") or ""
                    ),
                    "canonical_identity_validated": bool(
                        match.get("canonical_identity_validated")
                    ),
                    "identity_join_mode": str(
                        match.get("identity_join_mode") or ""
                    ),
                },
            },
            "path_fact_ids": list(row.get("path_fact_ids") or []) + [
                physical_fact_id
            ],
            "path_fact_usage_ids": list(
                row.get("path_fact_usage_ids") or []
            ) + [fact_id],
            "next_operator_index": _as_int(row.get("next_operator_index")) + 1,
            "last_action": str(action["operator"]),
            "created_by_action_key": str(action.get("action_key") or ""),
        }
        child["overlay_reuse_blocked"] = bool(
            not state.get("query_local_overlay_reuse", True)
            and state["facts"][fact_id].get("origin") == "query_local_overlay"
            and _as_int(child.get("next_operator_index")) < len(
                (state.get("logical_plan") or {}).get("operators") or []
            )
        )
        state["next_binding_id"] += 1
        state["bindings"].append(child)
        created.append(child["binding_id"])
    lookup_key = f"{LOOKUP_JOIN}:{row['binding_id']}:{action['obligation_id']}"
    if action["operator"] == LOOKUP_JOIN:
        state["lookup_match_counts"][lookup_key] = len(created)
        index_access = dict(result.get("index_access") or {})
        deleted_count = _as_int(
            index_access.get("perturbation_deleted_match_count")
        )
        corrupted_count = _as_int(
            index_access.get("perturbation_corrupted_match_count")
        )
        if deleted_count or corrupted_count:
            row["perturbed_lookup_deleted_count"] = deleted_count
            row["perturbed_lookup_corrupted_count"] = corrupted_count
            row["perturbation_repair_required"] = True
        if not created and state.get("bswc_enabled"):
            source_candidates = list(
                backend.enumerate_source_candidates(row, operator) or []
            )
            added = _register_bswc_source_candidates(
                state,
                source_candidates,
                row,
                operator,
                discovery_action=LOOKUP_JOIN,
            )
            result["source_candidates"] = source_candidates
            result["discovered_source_candidate_count"] = added
            result["title_catalog_doc_ids"] = [
                _as_int(value.get("source_id", value.get("doc_id")), -1)
                for value in source_candidates
            ]
            result["title_catalog_hit"] = bool(source_candidates)
    if action["operator"] == SOURCE_INSPECT:
        state["bswc_source_inspection_count"] = (
            _as_int(state.get("bswc_source_inspection_count")) + 1
        )
        if result.get("source_reused"):
            state["bswc_shared_source_reuse_count"] = (
                _as_int(state.get("bswc_shared_source_reuse_count")) + 1
            )
    keep_open_for_eager_completion = bool(
        created
        and action["operator"] == LOOKUP_JOIN
        and state.get("completion_mode") == "eager"
    )
    if created and not keep_open_for_eager_completion:
        row["active"] = False
        row["expanded_by"] = str(action["operator"])
    repair_request_key = str(action.get("repair_request_key") or "")
    if repair_request_key:
        request = (state.get("repair_requests") or {}).get(repair_request_key)
        if request is not None:
            request["status"] = "executed"
            request["created_binding_ids"] = list(created)
    obligation_after = OBLIGATION_SATISFIED if created else OBLIGATION_OPEN
    obligation_events = [{
        "obligation_id": str(action.get("obligation_id") or ""),
        "binding_id": str(row.get("binding_id") or ""),
        "from": obligation_before,
        "to": obligation_after,
        "cause": str(action.get("operator") or ""),
    }]
    return result, {
        "created_binding_ids": created,
        "fact_ids": [fact["physical_fact_id"] for fact in facts],
        "fact_usage_ids": [fact["fact_usage_id"] for fact in facts],
        "known_rejected_fact_filtered_count": len(known_rejected_match_ids),
        "known_rejected_fact_filtered_ids": list(dict.fromkeys(
            known_rejected_match_ids
        )),
        "binding_contract_admitted_count": binding_contract_counts["ADMITTED"],
        "binding_contract_rejected_count": binding_contract_counts["REJECTED"],
        "binding_contract_deferred_count": binding_contract_counts["DEFERRED"],
        "binding_contract_off_count": binding_contract_counts["OFF"],
        "binding_contract_reason_counts": {
            key.split(":", 1)[1]: value
            for key, value in binding_contract_counts.items()
            if key.startswith("reason:")
        },
        "logical_progress": bool(created),
        "scan_complete": bool(result.get("scan_complete", True)),
        "budget_limited_partial_scan": bool(
            result.get("budget_limited_partial_scan")
            or not result.get("scan_complete", True)
        ),
        "query_local_overlay_fact_count": sum(
            fact.get("origin") == "query_local_overlay" for fact in facts
        ),
        "verification_repair_triggered": bool(
            verification_repair_candidate
        ),
        "perturbation_repair_triggered": bool(
            perturbation_repair_candidate
        ),
        "rejected_index_fact_ids": list(
            effective_rejected_fact_ids
        ),
        "repair_request_key": repair_request_key,
        "repair_scheduler_reason": str(
            action.get("repair_scheduler_reason") or ""
        ),
        "obligation_state_before": obligation_before,
        "obligation_state_after": obligation_after,
        "obligation_events": obligation_events,
    }


def _execute_verify_action(
    state: Dict[str, Any], action: Mapping[str, Any], backend: EvidenceBackend
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fact_usage_id = str(
        action.get("fact_usage_id") or action.get("fact_id") or ""
    )
    fact = state["facts"][fact_usage_id]
    fact_id = str(fact.get("physical_fact_id") or fact.get("fact_id") or "")
    tuple_state_before = str(fact.get("tuple_state") or UNVERIFIED)
    verification_fact = dict(fact)
    verification_fact["fact_usage_id"] = fact_usage_id
    verification_fact["fact_id"] = fact_id
    result = dict(backend.verify(verification_fact) or {})
    status = str(result.get("support_status") or INSUFFICIENT)
    if status not in {SUPPORTED, CONTRADICTED, INSUFFICIENT, UNKNOWN}:
        status = INSUFFICIENT
    if (
        status == INSUFFICIENT
        and str(getattr(backend, "verification_mode", ""))
        == "tri_state_strict"
    ):
        status = UNKNOWN
    raw_source_status = str(result.get("source_support_gate_status") or "")
    source_status = (
        raw_source_status
        if raw_source_status in {
            SUPPORTED, CONTRADICTED, INSUFFICIENT, UNKNOWN, "OFF"
        }
        else status
    )
    admission_status = str(result.get("obligation_admission_status") or "")
    if not admission_status:
        admission_status = (
            "ADMITTED" if status == SUPPORTED
            else "DEFERRED" if status == UNKNOWN
            else "REJECTED"
        )
    assertion_wide_rejection = source_status in {
        CONTRADICTED,
        INSUFFICIENT,
    }
    fact.update({
        "support_status": status,
        "support_reason": str(result.get("support_reason") or "unspecified"),
        "source_support_status": source_status,
        "source_support_reason": str(
            result.get("source_support_gate_reason")
            or result.get("support_reason")
            or "unspecified"
        ),
        "source_sentence_id": _as_int(result.get("source_sentence_id"), -1),
        "source_sentence_span": str(result.get("source_sentence_span") or "")[:480],
        "verifier_version": str(
            result.get("verifier_version") or backend.verifier_version
        ),
        "guarantee_scope": str(result.get("guarantee_scope") or "source_support_only"),
        "provenance_scope": str(result.get("provenance_scope") or ""),
        "source_title": str(result.get("source_title") or ""),
        "identity_surface_refinement_rejected": bool(
            result.get("identity_surface_refinement_rejected")
        ),
        "unsafe_admission": bool(result.get("unsafe_admission")),
        "admission_policy": str(result.get("admission_policy") or ""),
        "source_support_gate_status": str(
            result.get("source_support_gate_status") or ""
        ),
        "source_support_gate_reason": str(
            result.get("source_support_gate_reason") or ""
        ),
        "obligation_admission_status": admission_status,
        "obligation_admission_reason": str(
            result.get("obligation_admission_reason") or ""
        ),
        "predicate_entailment_admission_status": str(
            result.get("predicate_entailment_admission_status") or ""
        ),
        "predicate_entailment_admission_reason": str(
            result.get("predicate_entailment_admission_reason") or ""
        ),
        "predicate_entailment_target": str(
            result.get("predicate_entailment_target") or ""
        ),
        "predicate_entailment_evidence": str(
            result.get("predicate_entailment_evidence") or ""
        ),
        "predicate_entailment_pattern": str(
            result.get("predicate_entailment_pattern") or ""
        ),
        "binding_obligation_contract_status": str(
            result.get("binding_obligation_contract_status")
            or fact.get("binding_obligation_contract_status")
            or ""
        ),
        "binding_obligation_contract_reason": str(
            result.get("binding_obligation_contract_reason")
            or fact.get("binding_obligation_contract_reason")
            or ""
        ),
        "obligation_contract_version": str(
            result.get("obligation_contract_version") or ""
        ),
        "certified_assertion": dict(
            result.get("certified_assertion") or {}
        ),
        "verified_by_action_key": str(action.get("action_key") or ""),
    })
    if status == SUPPORTED and admission_status in {"ADMITTED", "OFF"}:
        fact["tuple_state"] = (
            TUPLE_RECOVERED
            if fact.get("origin") == "query_local_overlay"
            else SUPPORTED
        )
    elif status == UNKNOWN or admission_status == "DEFERRED":
        fact["tuple_state"] = TUPLE_DEFERRED
        fact["deferred_by_action_key"] = str(action.get("action_key") or "")
    else:
        fact["tuple_state"] = TUPLE_REJECTED
        fact["rejected_by_action_key"] = str(action.get("action_key") or "")
    fact["rejection_scope"] = (
        (
            "assertion_wide_source_support"
            if assertion_wide_rejection
            else "binding_obligation_usage"
        )
        if fact.get("tuple_state") == TUPLE_REJECTED
        else ""
    )
    operators = list((state.get("logical_plan") or {}).get("operators") or [])
    operator_id = str(fact.get("operator_id") or "")
    output_slot = next(
        (
            str(operator.get("output_slot") or "")
            for operator in operators
            if str(operator.get("operator_id") or "") == operator_id
        ),
        "",
    )
    output_role = str(fact.get("bound_output_role") or "")
    verified_surface = str(
        result.get(f"source_{output_role}_surface") or ""
    ).strip()
    if status == SUPPORTED and output_role in {"subject", "object"} and verified_surface:
        from .relation_fact_join_utils import canonical_entity_key

        verified_key = canonical_entity_key(verified_surface)
        triple = list(fact.get("triple") or [])
        triple_index = 0 if output_role == "subject" else 2
        previous_output_surface = str(
            triple[triple_index] if len(triple) == 3 else ""
        ).strip()
        if len(triple) == 3:
            triple[triple_index] = verified_surface
            fact["triple"] = triple
        fact["verified_output_surface"] = verified_surface
        fact["verified_output_key"] = verified_key
        title_surfaces = (
            getattr(backend, "relation_index", {}).get("title_entity_surfaces") or {}
        )
        canonical_identity_surface = str(
            title_surfaces.get(verified_key) or verified_surface
        ).strip()
        canonical_identity_validated = bool(
            verified_key in title_surfaces
            or len(_tokens(verified_surface))
            > len(_tokens(previous_output_surface))
        )
        source_qualifiers = _local_identity_qualifiers(
            fact.get("source_sentence_span"),
            verified_surface,
        )
        identity_qualifiers = _identity_title_qualifiers(
            canonical_identity_surface
        )
        qualifier_conflict = bool(
            source_qualifiers
            and not source_qualifiers.issubset(identity_qualifiers)
        )
        if qualifier_conflict:
            canonical_identity_validated = False
            fact["canonical_identity_qualifier_conflict"] = True
            fact["source_identity_qualifiers"] = sorted(source_qualifiers)
            fact["canonical_identity_qualifiers"] = sorted(identity_qualifiers)
            fact["identity_qualifier_scope"] = (
                "verified_output_surface_and_local_appositive"
            )
        if output_slot:
            for row in state.get("bindings") or []:
                if fact_usage_id not in set(
                    row.get("path_fact_usage_ids")
                    or row.get("path_fact_ids")
                    or []
                ):
                    continue
                slot = dict((row.get("slot_bindings") or {}).get(output_slot) or {})
                if not slot:
                    continue
                row["slot_bindings"][output_slot] = {
                    **slot,
                    "surface": verified_surface,
                    "key": verified_key,
                    "surface_source": "verified_source_sentence",
                    "verified_source_id": fact.get("source_id"),
                    "verified_fact_id": fact_id,
                    "verified_fact_usage_id": fact_usage_id,
                    "lineage_key": f"{fact_usage_id}:{output_slot}:verified",
                    "canonical_identity_surface": canonical_identity_surface,
                    "canonical_identity_key": (
                        verified_key if canonical_identity_validated else ""
                    ),
                    "canonical_identity_validated": canonical_identity_validated,
                    "canonical_identity_qualifier_conflict": qualifier_conflict,
                    "source_identity_qualifier_tokens": sorted(
                        source_qualifiers
                    ),
                    "canonical_identity_source": (
                        "verified_source_title_or_appositive"
                        if canonical_identity_validated else ""
                    ),
                }
    if status == SUPPORTED and output_slot:
        lineage_years = sorted(_year_values(fact.get("source_sentence_span")))
        for row in state.get("bindings") or []:
            if fact_usage_id not in set(
                row.get("path_fact_usage_ids")
                or row.get("path_fact_ids")
                or []
            ):
                continue
            slot = dict((row.get("slot_bindings") or {}).get(output_slot) or {})
            if not slot:
                continue
            row["slot_bindings"][output_slot] = {
                **slot,
                "lineage_source_years": lineage_years,
                "lineage_source_sentence_span": str(
                    fact.get("source_sentence_span") or ""
                ),
                "lineage_source_id": fact.get("source_id"),
            }
    invalidated_binding_ids: List[str] = []
    reopened_parent_binding_ids: List[str] = []
    restarted_root_binding_ids: List[str] = []
    obligation_events: List[Dict[str, Any]] = []
    if status == CONTRADICTED or (
        status in {INSUFFICIENT, UNKNOWN}
        and state.get("prune_insufficient_lineage")
    ):
        execution_mode = str(state.get("execution_mode") or "saqe")
        affected_rows = [
            row for row in state.get("bindings") or []
            if fact_usage_id in set(
                row.get("path_fact_usage_ids")
                or row.get("path_fact_ids")
                or []
            )
        ]
        if execution_mode != "no_invalidation":
            invalidated_binding_ids = _invalidate_dependent_rows(
                state, fact_usage_id
            )
        obligation_id = str(fact.get("operator_id") or "")
        obligation_events.append({
            "obligation_id": obligation_id,
            "binding_id": str(action.get("binding_id") or ""),
            "from": OBLIGATION_SATISFIED,
            "to": OBLIGATION_INVALIDATED,
            "cause": f"verification_{status.lower()}",
            "fact_id": fact_id,
            "fact_usage_id": fact_usage_id,
        })
        if execution_mode == "restart":
            restarted_root_binding_ids = _restart_root_after_rejection(
                state,
                rejected_fact_id=fact_id,
                affected_rows=affected_rows,
                obligation_id=obligation_id,
                assertion_wide=assertion_wide_rejection,
            )
            for binding_id in restarted_root_binding_ids:
                obligation_events.append({
                    "obligation_id": obligation_id,
                    "binding_id": binding_id,
                    "from": OBLIGATION_INVALIDATED,
                    "to": OBLIGATION_REOPENED,
                    "cause": "root_restart_after_verification_failure",
                    "fact_id": fact_id,
                })
        elif (
            fact.get("origin") == "fact_index"
            and state.get("completion_enabled", True)
            and execution_mode == "saqe"
        ):
            created_by_action_key = str(fact.get("created_by_action_key") or "")
            affected_rows = [
                row for row in state.get("bindings") or []
                if fact_usage_id in set(
                    row.get("path_fact_usage_ids")
                    or row.get("path_fact_ids")
                    or []
                )
                and str(row.get("parent_binding_id") or "")
            ]
            for parent_id in dict.fromkeys(
                str(row.get("parent_binding_id") or "") for row in affected_rows
            ):
                siblings = [
                    row for row in state.get("bindings") or []
                    if str(row.get("parent_binding_id") or "") == parent_id
                    and str(row.get("created_by_action_key") or "")
                    == created_by_action_key
                ]
                if not siblings or any(
                    not sibling.get("invalidated") for sibling in siblings
                ):
                    continue
                parent = _binding_by_id(state, parent_id)
                if parent.get("invalidated"):
                    continue
                _record_binding_fact_rejection(
                    parent,
                    obligation_id=obligation_id,
                    physical_fact_id=fact_id,
                    assertion_wide=assertion_wide_rejection,
                )
                if state.get("repair_scheduler_mode") == "terminal_critical":
                    operator_index = next(
                        (
                            index for index, operator in enumerate(operators)
                            if str(operator.get("operator_id") or "")
                            == obligation_id
                        ),
                        _as_int(action.get("operator_index"), -1),
                    )
                    request_key = f"{parent_id}:{obligation_id}"
                    request = state.setdefault("repair_requests", {}).setdefault(
                        request_key,
                        {
                            "request_key": request_key,
                            "parent_binding_id": parent_id,
                            "root_binding_id": str(
                                parent.get("root_binding_id") or ""
                            ),
                            "obligation_id": obligation_id,
                            "operator_index": operator_index,
                            "source_fact_ids": [],
                            "source_statuses": [],
                            "status": "pending",
                        },
                    )
                    request["source_fact_ids"] = list(dict.fromkeys(
                        list(request.get("source_fact_ids") or []) + [fact_id]
                    ))
                    request["source_statuses"] = list(dict.fromkeys(
                        list(request.get("source_statuses") or []) + [status]
                    ))
                    obligation_events.append({
                        "obligation_id": obligation_id,
                        "binding_id": parent_id,
                        "from": OBLIGATION_INVALIDATED,
                        "to": OBLIGATION_REOPENED,
                        "cause": "repair_request_deferred_to_scheduler",
                        "fact_id": fact_id,
                        "request_key": request_key,
                    })
                else:
                    parent["active"] = True
                    parent["verification_repair_required"] = True
                    lookup_key = f"{LOOKUP_JOIN}:{parent_id}:{obligation_id}"
                    state["lookup_match_counts"][lookup_key] = 0
                    reopened_parent_binding_ids.append(parent_id)
                    obligation_events.append({
                        "obligation_id": obligation_id,
                        "binding_id": parent_id,
                        "from": OBLIGATION_INVALIDATED,
                        "to": OBLIGATION_REOPENED,
                        "cause": "all_index_siblings_rejected",
                        "fact_id": fact_id,
                    })
    elif status == SUPPORTED:
        obligation_events.append({
            "obligation_id": str(fact.get("operator_id") or ""),
            "binding_id": str(action.get("binding_id") or ""),
            "from": OBLIGATION_SATISFIED,
            "to": OBLIGATION_COMPLETED,
            "cause": "source_supported",
            "fact_id": fact_id,
        })
    return result, {
        "verified_fact_id": fact_id,
        "verified_fact_usage_id": fact_usage_id,
        "support_status": status,
        "source_support_gate_status": str(
            result.get("source_support_gate_status") or ""
        ),
        "source_support_gate_reason": str(
            result.get("source_support_gate_reason") or ""
        ),
        "obligation_admission_status": str(
            result.get("obligation_admission_status") or ""
        ),
        "obligation_admission_reason": str(
            result.get("obligation_admission_reason") or ""
        ),
        "predicate_entailment_admission_status": str(
            result.get("predicate_entailment_admission_status") or ""
        ),
        "predicate_entailment_admission_reason": str(
            result.get("predicate_entailment_admission_reason") or ""
        ),
        "binding_obligation_contract_status": str(
            result.get("binding_obligation_contract_status") or ""
        ),
        "binding_obligation_contract_reason": str(
            result.get("binding_obligation_contract_reason") or ""
        ),
        "tuple_state_before": tuple_state_before,
        "tuple_state_after": str(fact.get("tuple_state") or ""),
        "rejected_tuple_ids": (
            [fact_id] if fact.get("tuple_state") == TUPLE_REJECTED else []
        ),
        "deferred_tuple_ids": (
            [fact_id] if fact.get("tuple_state") == TUPLE_DEFERRED else []
        ),
        "invalidated_binding_count": len(invalidated_binding_ids),
        "invalidated_binding_ids": invalidated_binding_ids,
        "reopened_parent_binding_ids": reopened_parent_binding_ids,
        "restarted_root_binding_ids": restarted_root_binding_ids,
        "verification_repair_required": bool(
            reopened_parent_binding_ids or restarted_root_binding_ids
        ),
        "obligation_events": obligation_events,
        "logical_progress": status == SUPPORTED or bool(invalidated_binding_ids),
    }


def _apply_schema_ablation_controls(
    logical_plan: Mapping[str, Any], backend: EvidenceBackend
) -> Dict[str, Any]:
    """Return an execution-local plan reflecting isolated schema ablations.

    The original compiler output is retained under ``unablated_operators`` for
    audit. Defaults are both enabled, so canonical runs are byte-for-byte
    equivalent apart from the explicit controls recorded in the trace.
    """
    plan = copy.deepcopy(dict(logical_plan or {}))
    enforce_subtype = bool(getattr(
        backend, "enable_predicate_subtype_constraints", True
    ))
    enforce_type_role = bool(getattr(
        backend, "enable_type_role_constraints", True
    ))
    plan["schema_constraint_controls"] = {
        "predicate_subtype_constraints": enforce_subtype,
        "type_role_constraints": enforce_type_role,
    }
    if enforce_subtype:
        return plan
    operators = [dict(row) for row in plan.get("operators") or []]
    plan["unablated_operators"] = copy.deepcopy(operators)
    for operator in operators:
        operator["required_predicate_subtypes"] = []
        operator["strict_predicate_subtype"] = False
        if str(operator.get("join_key_contract") or "").startswith(
            "entity_identity+predicate_subtype"
        ):
            operator["join_key_contract"] = "exact_normalized_entity_key"
    plan["operators"] = operators
    return plan


def execute_evidence_query(
    *,
    query_id: str,
    question: str,
    logical_plan: Mapping[str, Any],
    backend: EvidenceBackend,
    policy: str = "adaptive_dependency",
    budget: Optional[ExecutionBudget] = None,
    max_no_progress: int = 3,
    allow_external_plan_review: bool = False,
    allow_model_based_semantic_admission: bool = False,
    require_semantic_execution_admission: bool = False,
) -> Dict[str, Any]:
    """Execute LOOKUP_JOIN, COMPLETE and VERIFY in one observable loop."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown evidence policy: {policy}")
    budget = budget or ExecutionBudget()
    plan = _apply_schema_ablation_controls(logical_plan, backend)
    plan_coverage = dict(plan.get("plan_coverage") or {})
    plan_coverage_enforced = bool(
        plan.get("plan_coverage_contract_version")
        == PLAN_COVERAGE_CONTRACT_VERSION
    )
    plan_coverage_status = str(
        plan_coverage.get("status")
        or plan.get("plan_coverage_status")
        or ""
    )
    question_semantically_complete = bool(
        not plan_coverage_enforced or plan_coverage_status == PLAN_COMPLETE
    )
    started = time.perf_counter()
    constrained_policy = policy == "feasibility_constrained_adaptive"
    bswc_enabled = policy in BSWC_POLICIES
    if bswc_enabled:
        executor_version = "paper_evidence_executor_bswc_v1"
    elif constrained_policy and bool(plan.get("schema_exact_nested_join_v9_enabled")):
        executor_version = "paper_evidence_executor_v9"
    elif constrained_policy and bool(plan.get("general_plan_coverage_v8_enabled")):
        executor_version = "paper_evidence_executor_v8_1"
    elif constrained_policy and bool(getattr(
        backend, "enable_provenance_grouped_terminal_v7_2", False
    )):
        executor_version = "paper_evidence_executor_v7_2"
    elif constrained_policy and bool(getattr(
        backend, "enable_lineage_terminal_semijoin_v7_1", False
    )):
        executor_version = "paper_evidence_executor_v7_1"
    elif constrained_policy and bool(getattr(
        backend, "enable_schema_plan_coverage_v7", False
    )):
        executor_version = "paper_evidence_executor_v7"
    elif constrained_policy and bool(getattr(
        backend, "enable_identity_safe_composite_join", False
    )):
        executor_version = "paper_evidence_executor_v6"
    elif constrained_policy:
        executor_version = "paper_evidence_executor_v5"
    else:
        executor_version = "paper_evidence_executor_v4"
    conflict_safe_terminal = bool(
        getattr(backend, "enable_lineage_terminal_semijoin_v7_1", False)
    )
    provenance_grouped_terminal = bool(
        getattr(backend, "enable_provenance_grouped_terminal_v7_2", False)
    )
    supported_shapes = {
        ("relation_projection", "project"),
        ("path_projection", "project"),
        ("branch_compare", "compare_eq"),
        ("branch_temporal_compare", "argmin_root"),
        ("branch_temporal_compare", "argmax_root"),
    }
    supported_shape = bool(
        plan.get("compile_status") == "compiled"
        and plan.get("operators")
        and (
            str(plan.get("plan_kind") or ""),
            str(plan.get("answer_operator") or ""),
        ) in supported_shapes
    )
    if not supported_shape:
        return _json_safe({
            "version": executor_version,
            "query_id": str(query_id),
            "question": question,
            "policy": policy,
            "query_shape": str(plan.get("plan_kind") or "unsupported"),
            "supported": False,
            "structured_admission": False,
            "plan_coverage_enforced": plan_coverage_enforced,
            "plan_coverage_status": plan_coverage_status,
            "question_complete_supported_witness": False,
            "partial_supported_path": False,
            "output_path": "unsupported_baseline_fallback",
            "stop_reason": "unsupported_query_shape",
            "logical_plan": plan,
            "bindings": [],
            "facts": [],
            "operator_trace": [],
            "cost_ledger": CostLedger().public(),
            "terminal_reduction": {"status": "unsupported_query_shape"},
            "phase_status": _execution_phase_status(
                plan,
                failure_stage=(
                    "plan_static_validation"
                    if not bool(
                        (plan.get("compile_phase_status") or {}).get(
                            "plan_static_valid",
                            plan.get("compile_status") == "compiled",
                        )
                    )
                    else "execution_admission"
                ),
                failure_reason="unsupported_query_shape",
            ),
            "end_to_end_wall_ms": round((time.perf_counter() - started) * 1000, 6),
        })

    semantic_validation = plan.get("question_semantic_validation")
    ai_provenance = plan.get("ai_plan_provenance")
    ai_diagnostic_admitted = bool(
        allow_external_plan_review
        and isinstance(semantic_validation, Mapping)
        and str(semantic_validation.get("status") or "")
        == "AI_DEVELOPMENT_DIAGNOSTIC"
        and bool(semantic_validation.get("diagnostic_execution_authorized"))
        and str(semantic_validation.get("review_origin") or "")
        == "AI_DEVELOPMENT_ONLY"
        and str(semantic_validation.get("human_review_status") or "")
        == "NOT_PERFORMED"
        and isinstance(ai_provenance, Mapping)
        and bool(ai_provenance.get("diagnostic_execution_authorized"))
        and not bool(ai_provenance.get("is_human_gold"))
    )
    model_admission = plan.get("model_based_semantic_admission")
    model_based_semantic_admitted = bool(
        allow_model_based_semantic_admission
        and isinstance(semantic_validation, Mapping)
        and str(semantic_validation.get("status") or "")
        == "MODEL_BASED_SEMANTIC_ADMISSION_DEV"
        and str(semantic_validation.get("admission_source") or "")
        == "MODEL_BASED_SEMANTIC_ADMISSION_DEV"
        and str(semantic_validation.get("review_decision") or "") == "PASS"
        and bool(semantic_validation.get("experimental_execution_eligible"))
        and not bool(semantic_validation.get("human_reviewed"))
        and not bool(semantic_validation.get("is_human_gold"))
        and isinstance(model_admission, Mapping)
        and str(model_admission.get("admission_source") or "")
        == "MODEL_BASED_SEMANTIC_ADMISSION_DEV"
        and bool(model_admission.get("experimental_execution_eligible"))
        and not bool(model_admission.get("human_reviewed"))
        and not bool(model_admission.get("is_human_gold"))
    )
    semantic_admitted = bool(
        isinstance(semantic_validation, Mapping)
        and (
            bool(semantic_validation.get("auto_execution_eligible"))
            or ai_diagnostic_admitted
            or model_based_semantic_admitted
        )
    )
    if (
        require_semantic_execution_admission and not semantic_admitted
    ) or (
        isinstance(semantic_validation, Mapping) and not semantic_admitted
    ):
        return _json_safe({
            "version": executor_version,
            "query_id": str(query_id),
            "question": question,
            "policy": policy,
            "query_shape": str(plan.get("plan_kind") or "unsupported"),
            "supported": False,
            "structured_admission": False,
            "plan_coverage_enforced": plan_coverage_enforced,
            "plan_coverage_status": plan_coverage_status,
            "question_complete_supported_witness": False,
            "partial_supported_path": False,
            "output_path": "question_semantic_review_fallback",
            "stop_reason": "question_semantic_review_required",
            "question_semantic_validation": (
                dict(semantic_validation)
                if isinstance(semantic_validation, Mapping)
                else {}
            ),
            "semantic_execution_admission_required": bool(
                require_semantic_execution_admission
            ),
            "logical_plan": plan,
            "bindings": [],
            "facts": [],
            "operator_trace": [],
            "cost_ledger": CostLedger().public(),
            "terminal_reduction": {
                "status": "question_semantic_review_required"
            },
            "phase_status": _execution_phase_status(
                plan,
                failure_stage="question_semantic_admission",
                failure_reason="question_semantic_review_required",
            ),
            "end_to_end_wall_ms": round(
                (time.perf_counter() - started) * 1000, 6
            ),
        })

    roots = backend.initial_bindings(question, plan)
    operators = list(plan.get("operators") or [])
    root_slot = str((operators[0] if operators else {}).get("input_slot") or "$root")
    branch_plan = str(plan.get("plan_kind") or "").startswith("branch_")
    bindings = []
    for index, root in enumerate(roots):
        surface = str(root.get("surface") or "").strip()
        key = str(root.get("key") or _key(surface))
        if not surface or not key:
            continue
        binding_id = f"b{index}"
        category_partition_validated = bool(
            root.get("root_category_partition_validated")
        )
        category_partition_doc_ids = list(dict.fromkeys(
            root.get("root_category_partition_doc_ids") or []
        ))
        root_identity_validated = bool(
            root.get("title_catalog_exact")
            or category_partition_validated
        )
        identity_doc_ids = (
            category_partition_doc_ids
            if category_partition_validated else
            list(root.get("canonical_title_doc_ids") or [])
        )
        bindings.append({
            "binding_id": binding_id,
            "parent_binding_id": "",
            "root_binding_id": binding_id,
            "root_surface": _clean_root_answer_surface(
                root.get("mention_surface") or surface
            ),
            "root_key": key,
            "active": True,
            "invalidated": False,
            "slot_bindings": {
                root_slot: {
                    "surface": surface,
                    "key": key,
                    "canonical_identity_surface": (
                        surface if root_identity_validated else ""
                    ),
                    "canonical_identity_key": (
                        key if root_identity_validated else ""
                    ),
                    "canonical_identity_validated": bool(
                        root_identity_validated
                    ),
                    "canonical_identity_source": (
                        "query_title_category_partition"
                        if category_partition_validated else
                        "query_title_catalog"
                        if root.get("title_catalog_exact") else ""
                    ),
                    "canonical_title_family_key": str(
                        root.get("canonical_title_family_key") or ""
                    ),
                    "canonical_title_doc_ids": list(
                        identity_doc_ids
                    ),
                    "canonical_title_family_doc_ids": list(
                        root.get("canonical_title_family_doc_ids") or []
                    ),
                    "canonical_title_family_unambiguous": bool(
                        len(identity_doc_ids) == 1
                        if category_partition_validated else
                        root.get("canonical_title_family_unambiguous")
                    ),
                    "root_category_partition_validated": bool(
                        category_partition_validated
                    ),
                    "root_category_partition_doc_ids": list(
                        category_partition_doc_ids
                    ),
                    "root_category_partition_categories": list(
                        root.get("root_category_partition_categories") or []
                    ),
                    "allow_title_partition_direct_lookup": bool(
                        category_partition_validated
                        or (
                            plan.get("newly_compiled_v8")
                            and root.get("title_catalog_exact")
                            and root.get("canonical_title_family_unambiguous")
                        )
                    ),
                }
            },
            "path_fact_ids": [],
            "path_fact_usage_ids": [],
            "next_operator_index": 0,
            "root_metadata": dict(root),
        })
    expected_root_count = 2 if branch_plan else 1
    if len(bindings) != expected_root_count:
        return _json_safe({
            "version": executor_version,
            "query_id": str(query_id),
            "question": question,
            "policy": policy,
            "query_shape": str(plan.get("plan_kind") or ""),
            "supported": True,
            "structured_admission": False,
            "plan_coverage_enforced": plan_coverage_enforced,
            "plan_coverage_status": plan_coverage_status,
            "question_complete_supported_witness": False,
            "partial_supported_path": False,
            "output_path": "ordinary_baseline_fallback",
            "stop_reason": (
                "branch_root_arity_unavailable"
                if branch_plan else "root_binding_unavailable"
            ),
            "logical_plan": plan,
            "bindings": bindings,
            "facts": [],
            "operator_trace": [],
            "cost_ledger": CostLedger().public(),
            "terminal_reduction": {
                "status": "root_binding_unavailable",
                "expected_root_count": expected_root_count,
                "actual_root_count": len(bindings),
            },
            "phase_status": _execution_phase_status(
                plan,
                failure_stage="root_binding",
                failure_reason=(
                    "branch_root_arity_unavailable"
                    if branch_plan else "root_binding_unavailable"
                ),
            ),
            "end_to_end_wall_ms": round((time.perf_counter() - started) * 1000, 6),
        })

    state: Dict[str, Any] = {
        "query_id": str(query_id),
        "question": question,
        "logical_plan": plan,
        "bindings": bindings,
        "facts": {},
        "fact_usage_lookup": {},
        "root_binding_ids": [row["binding_id"] for row in bindings],
        "next_binding_id": len(bindings),
        "attempted_action_keys": [],
        "lookup_match_counts": {},
        "observations": [],
        "obligation_history": [],
        "prune_insufficient_lineage": bool(
            policy == "feasibility_constrained_adaptive"
            or bswc_enabled
            or str(getattr(backend, "verification_mode", ""))
            in {"strict", "tri_state_strict", "obligation_only"}
        ),
        "completion_enabled": str(
            getattr(backend, "completion_mode", "lazy")
        ) != "off",
        "completion_mode": str(
            getattr(backend, "completion_mode", "lazy") or "lazy"
        ),
        "terminal_sibling_verification_fact_ids": set(),
        "repair_scheduler_mode": str(
            getattr(backend, "repair_scheduler_mode", "eager") or "eager"
        ),
        "execution_mode": str(getattr(backend, "execution_mode", "saqe") or "saqe"),
        "query_local_overlay_reuse": bool(getattr(
            backend, "enable_query_local_overlay_reuse", True
        )),
        "repair_requests": {},
        "repair_scheduler_events": [],
        "repair_scheduler_event_keys": set(),
        "cbo_risk_weight": _as_float(
            getattr(backend, "cbo_risk_weight", 0.8), 0.8
        ),
        "cbo_cost_weight": _as_float(
            getattr(backend, "cbo_cost_weight", 0.25), 0.25
        ),
        "bswc_enabled": bswc_enabled,
        "bswc_policy": policy if bswc_enabled else "",
        "bswc_support_oracle_path": (
            os.environ.get("S2RAG_BSWC_SUPPORT_ORACLE_PATH", "")
            if policy == "bswc_support_oracle_early" else ""
        ),
        "bswc_frontier_exposure_oracle_path": (
            os.environ.get("S2RAG_BSWC_FRONTIER_EXPOSURE_ORACLE_PATH", "")
            if policy in {
                "bswc_identity_safe_frontier_exposure_oracle",
                "bswc_identity_safe_scan_reservation_oracle",
            } else ""
        ),
        "bswc_source_candidates": [],
        "bswc_inspected_source_ids": set(),
        "bswc_open_coverage_keys": set(),
        "bswc_beam_width": max(
            1, _as_int(getattr(backend, "bswc_beam_width", 8), 8)
        ),
        "bswc_search_horizon": max(
            1, _as_int(getattr(backend, "bswc_search_horizon", 3), 3)
        ),
        "bswc_dense_probe_count": 0,
        "bswc_source_inspection_count": 0,
        "bswc_shared_source_reuse_count": 0,
        "bswc_feedback_stats": {},
        "bswc_feedback_events": [],
        "bswc_pending_source_feedback": {},
        "bswc_stopping_events": [],
        "bswc_source_model_loaded": bool(getattr(
            getattr(backend, "bswc_source_value_model", None), "loaded", False
        )),
        "bswc_feedback_strength": _as_float(
            getattr(backend, "bswc_feedback_strength", 3.0), 3.0
        ),
        "bswc_stop_threshold": _as_float(
            getattr(backend, "bswc_stop_threshold", 0.035), 0.035
        ),
        "bswc_min_exploration_actions": _as_int(
            getattr(backend, "bswc_min_exploration_actions", 1), 1
        ),
        "bswc_arwc_partial_value": _as_float(
            getattr(backend, "bswc_arwc_partial_value", 0.10), 0.10
        ),
        "bswc_arwc_budget_penalty": _as_float(
            getattr(backend, "bswc_arwc_budget_penalty", 0.05), 0.05
        ),
        "bswc_arwc_min_utility": _as_float(
            getattr(backend, "bswc_arwc_min_utility", 0.0), 0.0
        ),
        "bswc_fpawc_stop_upper_bound": _as_float(
            getattr(backend, "bswc_fpawc_stop_upper_bound", 0.0), 0.0
        ),
    }
    ledger = CostLedger()
    traces: List[Dict[str, Any]] = []
    no_progress = 0
    stop_reason = "no_executable_action"
    frozen_estimates: Dict[str, Dict[str, Any]] = {}
    selected_witness: Optional[Dict[str, Any]] = None
    reduction_reason = "not_attempted"
    reduction_detail: Dict[str, Any] = {}

    while True:
        if bswc_enabled:
            state["bswc_open_coverage_keys"] = {
                ":".join((
                    str(row.get("binding_id") or ""),
                    str(
                        operators[_as_int(row.get("next_operator_index"))].get(
                            "operator_id"
                        )
                        if 0 <= _as_int(row.get("next_operator_index")) < len(operators)
                        else ""
                    ),
                ))
                for row in _active_rows(state)
                if 0 <= _as_int(row.get("next_operator_index")) < len(operators)
            }
        witnesses = _structured_witnesses(state)
        selected_witness, reduction_reason, reduction_detail = _reduce_witnesses(
            witnesses,
            plan,
            state["root_binding_ids"],
            state["facts"],
            conflict_safe_terminal,
            provenance_grouped_terminal,
        )
        if selected_witness and not question_semantically_complete:
            stop_reason = (
                "plan_partial_supported_path"
                if plan_coverage_status == "PLAN_PARTIAL"
                else "plan_coverage_unknown_supported_path"
            )
            break
        if selected_witness:
            pending_terminal_verifies = []
            if conflict_safe_terminal:
                complete_rows = [
                    row for row in _active_rows(state)
                    if _row_complete(row, len(operators))
                ]
                complete_fact_usage_ids = {
                    str((
                        row.get("path_fact_usage_ids")
                        or row.get("path_fact_ids")
                        or [""]
                    )[-1])
                    for row in complete_rows
                    if row.get("path_fact_usage_ids") or row.get("path_fact_ids")
                }
                pending_terminal_verifies = [
                    action for action in _pending_verify_actions(state)
                    if str(
                        action.get("fact_usage_id")
                        or action.get("fact_id")
                        or ""
                    ) in complete_fact_usage_ids
                ]
            pending_eager_completion = bool(
                state.get("completion_mode") == "eager"
                and any(
                    action.get("operator") == COMPLETE
                    for action in _candidate_actions(state)
                )
            )
            if not pending_terminal_verifies and not pending_eager_completion:
                stop_reason = "verified_complete_witness"
                break
            state["terminal_sibling_verification_fact_ids"].update(
                str(action.get("fact_id") or "")
                for action in pending_terminal_verifies
            )
        actions = _candidate_actions(state)
        if not actions:
            stop_reason = (
                reduction_reason
                if reduction_reason == "ambiguous_terminal_conflict"
                else "no_executable_action"
            )
            break

        planning_started = time.perf_counter()
        affordable_actions = []
        estimates: Dict[str, Dict[str, Any]] = {}
        rejected_for_budget = []
        for action_index, raw_action in enumerate(actions):
            action = dict(raw_action)
            action_key = _action_key(action)
            action["action_key"] = action_key
            if (
                policy == "adaptive_dependency_frozen"
                and action_key in frozen_estimates
            ):
                estimate = dict(frozen_estimates[action_key])
            else:
                estimate = dict(backend.estimate(action, state) or {})
                if policy == "adaptive_dependency_frozen":



                    frozen_estimates[action_key] = dict(estimate)
            estimate.update(_hard_budget_requirements(action, estimate, backend))
            estimates[action_key] = estimate
            allowed, reason = _affordable(estimate, ledger, budget)
            if allowed:
                scan_rows_remaining = max(
                    0, budget.max_scan_rows - ledger.scan_rows
                )
                action["execution_scan_limit"] = scan_rows_remaining
                if (
                    policy == "bswc_identity_safe_scan_reservation_oracle"
                    and action.get("identity_safe_budget_target_lookup")
                ):
                    (
                        action["execution_scan_limit"],
                        action["scan_budget_reservation"],
                    ) = identity_safe_reserved_scan_limit(
                        action, scan_rows_remaining
                    )
                action["execution_retrieval_limit"] = max(
                    0, budget.max_retrieval_calls - ledger.retrieval_calls
                )
                action["execution_model_limit"] = max(
                    0,
                    budget.max_model_calls
                    - budget.reserve_model_calls
                    - ledger.model_calls,
                )
                affordable_actions.append(action)
            else:
                rejected_for_budget.append({"action_key": action_key, "reason": reason})
        if not affordable_actions:
            stop_reason = (
                "budget_insufficient" if rejected_for_budget else "no_executable_action"
            )
            break
        if policy in STOPPING_POLICIES:
            stop_low_value, stopping_detail = should_stop_low_value(
                affordable_actions,
                estimates,
                state,
                threshold=_as_float(state.get("bswc_stop_threshold"), 0.035),
                min_exploration=_as_int(
                    state.get("bswc_min_exploration_actions"), 1
                ),
            )
            stopping_event = {
                "decision_index": len(traces),
                "stopped": bool(stop_low_value),
                "remaining_budget": _budget_snapshot(ledger, budget),
                "open_coverage_keys": sorted(
                    str(value) for value in state.get("bswc_open_coverage_keys") or []
                ),
                **dict(stopping_detail),
            }
            state.setdefault("bswc_stopping_events", []).append(stopping_event)
            if stop_low_value:
                if state.get("bswc_feedback_events"):
                    latest_feedback = state["bswc_feedback_events"][-1]
                    latest_feedback.setdefault(
                        "ranking_after",
                        [str(action.get("action_key") or "") for action in affordable_actions],
                    )
                    latest_feedback.setdefault(
                        "next_selected_action_key", "STOP"
                    )
                stop_reason = "bswc_low_value_stop"
                break
        state["bswc_remaining_budget"] = _budget_snapshot(ledger, budget)
        selected, decision = _choose_action(
            policy=(
                "adaptive_dependency"
                if policy == "adaptive_dependency_frozen" else policy
            ),
            actions=affordable_actions,
            estimates=estimates,
            state=state,
        )
        if state.get("bswc_feedback_events"):
            latest_feedback = state["bswc_feedback_events"][-1]
            if "ranking_after" not in latest_feedback:
                latest_feedback["ranking_after"] = [
                    str(row.get("action_key") or "")
                    for row in decision.get("ranked") or []
                ]
                latest_feedback["next_selected_action_key"] = str(
                    (selected or {}).get("action_key") or ""
                )
        planning_ms = (time.perf_counter() - planning_started) * 1000.0
        ledger.planning_wall_ms += planning_ms
        if selected is None:
            if decision.get("reason") in {
                "selected_stop_by_bswc_adaptive_risk_cover",
                "selected_stop_by_bswc_feasibility_preserving_cover",
                "selected_stop_by_bswc_continuation_safe_cover",
                "selected_stop_by_bswc_precedence_aware_cover",
                "selected_stop_by_bswc_quality_first_precedence_cover",
                "selected_stop_by_bswc_global_witness_reservation_cover",
                "selected_stop_by_bswc_exact_adaptive_witness",
            }:
                stopping_policy = str(
                    (decision.get("bounded_search") or {}).get("reason") or policy
                )
                state.setdefault("bswc_stopping_events", []).append({
                    "decision_index": len(traces),
                    "stopped": True,
                    "policy": policy,
                    "stopping_policy": stopping_policy,
                    "selected_action_key": "STOP",
                    "frozen_source_frontier": decision.get(
                        "frozen_source_frontier"
                    ),
                    "remaining_budget": _budget_snapshot(ledger, budget),
                    "open_coverage_keys": sorted(
                        str(value)
                        for value in state.get("bswc_open_coverage_keys") or []
                    ),
                    **dict(decision.get("bounded_search") or {}),
                })
            stop_reason = "predicted_continue_value_low"
            break

        before_budget = _budget_snapshot(ledger, budget)
        action_started = time.perf_counter()
        if selected["operator"] == VERIFY:
            raw_result, transition = _execute_verify_action(state, selected, backend)
            if bswc_enabled and state.get("completion_enabled"):
                reopened_ids = list(
                    transition.get("reopened_parent_binding_ids") or []
                ) + list(transition.get("restarted_root_binding_ids") or [])
                discovered = 0
                for reopened_id in reopened_ids:
                    try:
                        reopened_row = _binding_by_id(state, str(reopened_id))
                    except StopIteration:
                        continue
                    operator_index = _as_int(
                        reopened_row.get("next_operator_index"), -1
                    )
                    if operator_index < 0 or operator_index >= len(operators):
                        continue
                    reopened_operator = operators[operator_index]
                    discovered += _register_bswc_source_candidates(
                        state,
                        backend.enumerate_source_candidates(
                            reopened_row, reopened_operator
                        ),
                        reopened_row,
                        reopened_operator,
                        discovery_action=VERIFY,
                    )
                transition["source_candidate_count_added"] = discovered
        else:
            raw_result, transition = _execute_binding_action(state, selected, backend)
        elapsed_ms = (time.perf_counter() - action_started) * 1000.0
        cost = _normalise_cost(raw_result.get("cost"), elapsed_ms)
        _add_cost(ledger, cost, budget)
        state["attempted_action_keys"].append(selected["action_key"])
        state["observations"].append({
            "action_key": selected["action_key"],
            "transition": transition,
            "cost": cost,
        })
        state["obligation_history"].extend(
            list(transition.get("obligation_events") or [])
        )
        if (
            policy in FEEDBACK_POLICIES
            and bool(getattr(backend, "enable_bswc_query_feedback", True))
            and str(selected.get("operator") or "")
            in {BOUND_DENSE_PROBE, SOURCE_INSPECT, VERIFY}
        ):
            selected_operator = str(selected.get("operator") or "")
            selected_estimate = dict(estimates.get(selected["action_key"]) or {})
            feedback_features = dict(
                selected_estimate.get("bswc_calibration_features")
                or action_features(selected, state)
            )
            probability_before = _as_float(
                selected_estimate.get("bswc_calibration_probability"),
                selected_estimate.get("support_probability", 0.5),
            )
            selected_rank_before = next(
                (
                    index
                    for index, ranked in enumerate(
                        (decision.get("ranked") or []), start=1
                    )
                    if str(ranked.get("action_key") or "")
                    == str(selected.get("action_key") or "")
                ),
                0,
            )
            ranking_before = [
                str(ranked.get("action_key") or "")
                for ranked in (decision.get("ranked") or [])
            ]

            def _annotate_feedback_event(feedback_event):
                feedback_event["selected_rank_before"] = selected_rank_before
                feedback_event["ranking_before"] = list(ranking_before)
                feedback_event["binding_count_after"] = len(_active_rows(state))
                feedback_event["open_obligation_count_after"] = len(
                    state.get("bswc_open_coverage_keys") or []
                )

            source_fact_ids = {
                str(value) for value in transition.get("fact_ids") or [] if str(value)
            }
            if (
                selected_operator == SOURCE_INSPECT
                and transition.get("created_binding_ids")
                and source_fact_ids
            ):




                state.setdefault("bswc_pending_source_feedback", {})[
                    str(selected.get("action_key") or "")
                ] = {
                    "features": feedback_features,
                    "probability_before": probability_before,
                    "fact_ids": sorted(source_fact_ids),
                    "resolved_fact_ids": [],
                    "selected_rank_before": selected_rank_before,
                    "ranking_before": list(ranking_before),
                }
            else:
                feedback_event = update_query_feedback(
                    state,
                    features=feedback_features,
                    success=source_action_success(selected_operator, transition),
                    action_key=str(selected.get("action_key") or ""),
                    probability_before=probability_before,
                )
                _annotate_feedback_event(feedback_event)

            if selected_operator == VERIFY:
                verified_fact_id = str(transition.get("verified_fact_id") or "")
                support_observed = str(
                    transition.get("support_status") or ""
                ) == SUPPORTED
                pending_table = state.setdefault("bswc_pending_source_feedback", {})
                for source_action_key, pending in list(pending_table.items()):
                    pending_fact_ids = {
                        str(value) for value in pending.get("fact_ids") or [] if str(value)
                    }
                    if verified_fact_id not in pending_fact_ids:
                        continue
                    resolved = {
                        str(value)
                        for value in pending.get("resolved_fact_ids") or []
                        if str(value)
                    }
                    resolved.add(verified_fact_id)
                    pending["resolved_fact_ids"] = sorted(resolved)
                    if not support_observed and not pending_fact_ids.issubset(resolved):
                        continue
                    source_feedback_event = update_query_feedback(
                        state,
                        features=dict(pending.get("features") or {}),
                        success=support_observed,
                        action_key=source_action_key,
                        probability_before=_as_float(
                            pending.get("probability_before"), 0.5
                        ),
                    )
                    source_feedback_event.update({
                        "deferred_until_verify": True,
                        "resolved_by_fact_id": verified_fact_id,
                        "selected_rank_before": _as_int(
                            pending.get("selected_rank_before"), 0
                        ),
                        "ranking_before": list(pending.get("ranking_before") or []),
                        "binding_count_after": len(_active_rows(state)),
                        "open_obligation_count_after": len(
                            state.get("bswc_open_coverage_keys") or []
                        ),
                    })
                    del pending_table[source_action_key]
        no_progress = 0 if transition.get("logical_progress") else no_progress + 1
        trace = {
            "query_id": str(query_id),
            "action_id": f"a{len(traces)}",
            "action_key": selected["action_key"],
            "operator": selected["operator"],
            "binding_id": selected.get("binding_id"),
            "fact_id": selected.get("fact_id"),
            "fact_usage_id": selected.get("fact_usage_id"),
            "source_id": selected.get("source_id"),
            "source_type": selected.get("source_type"),
            "target_obligation": selected.get("obligation_id"),
            "policy": policy,
            "scan_budget_reservation": _json_safe(
                selected.get("scan_budget_reservation") or {}
            ),
            "decision": decision,
            "estimate": estimates.get(selected["action_key"], {}),
            "actual_cost": cost,
            "transition": transition,
            "failure_reason": str(raw_result.get("reason") or ""),
            "physical_access": {
                key: _json_safe(raw_result.get(key))
                for key in (
                    "completion_query",
                    "completion_query_mode",
                    "bounded_source_doc_ids",
                    "title_catalog_doc_ids",
                    "title_catalog_hit",
                    "dense_probe_called",
                    "physical_join_plan",
                    "candidate_scan_count",
                    "scan_complete",
                    "budget_limited_partial_scan",
                    "dense_probe_requested",
                    "internal_verification_calls",
                    "internal_verification_scan_rows",
                    "internal_verification_truncated",
                    "title_probe_match_count",
                    "title_probe_verified_match_count",
                    "online_cardinality",
                    "index_access",
                    "identity_diagnostics",
                    "diagnostics",
                    "query_time_relation_materialization",
                    "selected_source_doc_id",
                    "selected_source_type",
                    "source_reused",
                    "source_candidates",
                    "discovered_source_candidate_count",
                )
                if key in raw_result
            },
            "budget_before": before_budget,
            "budget_after": _budget_snapshot(ledger, budget),
            "planning_wall_ms": round(planning_ms, 6),
        }
        traces.append(trace)
        effective_no_progress_limit = max(
            max(1, int(max_no_progress)),
            2 * expected_root_count if branch_plan else 1,
        )
        if (
            no_progress >= effective_no_progress_limit
            and policy != "feasibility_constrained_adaptive"
            and not bswc_enabled
        ):
            stop_reason = "no_progress_limit"
            break
        if ledger.budget_violations:
            stop_reason = "actual_budget_violation"
            break

    witnesses = _structured_witnesses(state)
    selected_witness, reduction_reason, reduction_detail = _reduce_witnesses(
        witnesses,
        plan,
        state["root_binding_ids"],
        state["facts"],
        conflict_safe_terminal,
        provenance_grouped_terminal,
    )
    provisional_witness = dict(reduction_detail.pop("provisional_witness", {}) or {})
    if provisional_witness:
        reduction_detail["provisional_witness_binding_id"] = str(
            provisional_witness.get("binding_id") or ""
        )
        reduction_detail["provisional_witness_answer"] = str(
            ((provisional_witness.get("slot_bindings") or {}).get("$answer") or {}).get(
                "surface"
            )
            or ""
        ).strip()
    plan_selected_witness = dict(selected_witness or {})
    question_complete_witnesses = (
        list(witnesses) if question_semantically_complete else []
    )
    question_complete_selected_witness = (
        dict(selected_witness or {}) if question_semantically_complete else {}
    )
    answer = str(
        (((selected_witness or {}).get("slot_bindings") or {}).get("$answer") or {}).get(
            "surface"
        )
        or ""
    ).strip()
    admitted = bool(question_complete_selected_witness and answer)
    verification_repair_fact_ids = {
        str(fact.get("physical_fact_id") or fact_id)
        for fact_id, fact in state["facts"].items()
        if fact.get("verification_repair_candidate")
        and fact.get("support_status") == SUPPORTED
    }
    perturbation_repair_fact_ids = {
        str(fact.get("physical_fact_id") or fact_id)
        for fact_id, fact in state["facts"].items()
        if fact.get("perturbation_repair_candidate")
        and fact.get("support_status") == SUPPORTED
    }
    verification_repair_witnesses = [
        row for row in witnesses
        if verification_repair_fact_ids.intersection(row.get("path_fact_ids") or [])
    ]
    perturbation_repair_witnesses = [
        row for row in witnesses
        if perturbation_repair_fact_ids.intersection(row.get("path_fact_ids") or [])
    ]
    if stop_reason == "verified_complete_witness" and not admitted:
        stop_reason = "complete_witness_projection_failed"
    reader_evidence_fact_ids, reader_evidence_mode = _reader_evidence_selection(
        state,
        question_complete_selected_witness,
        provisional_witness,
        plan_selected_witness if not question_semantically_complete else None,
    )
    support_reason_counts = Counter(
        str(fact.get("support_reason") or "")
        for fact in state["facts"].values()
        if str(fact.get("support_reason") or "")
    )
    title_catalog_actions = [
        row for row in traces
        if bool((row.get("physical_access") or {}).get("title_catalog_hit"))
    ]
    identity_rejected_count = sum(
        _as_int(
            ((row.get("physical_access") or {}).get("identity_diagnostics") or {}).get(
                "lineage_identity_rejected_count"
            )
        )
        for row in traces
    )
    identity_scoped_rows = sum(
        _as_int(
            ((row.get("physical_access") or {}).get("index_access") or {}).get(
                "identity_scoped_rows"
            )
        )
        for row in traces
    )
    identity_scoped_actions = sum(
        _as_int(
            ((row.get("physical_access") or {}).get("index_access") or {}).get(
                "identity_scoped_rows"
            )
        )
        > 0
        for row in traces
    )
    identity_scoped_matches = sum(
        _as_int(
            ((row.get("physical_access") or {}).get("identity_diagnostics") or {}).get(
                "identity_scoped_match_count"
            )
        )
        for row in traces
    )
    title_terminal_direct_actions = sum(
        _as_int(
            ((row.get("physical_access") or {}).get("index_access") or {}).get(
                "title_terminal_direct_actions"
            )
        )
        for row in traces
    )
    title_terminal_direct_rows = sum(
        _as_int(
            ((row.get("physical_access") or {}).get("index_access") or {}).get(
                "title_terminal_direct_rows"
            )
        )
        for row in traces
    )
    title_terminal_direct_matches = sum(
        _as_int(
            ((row.get("physical_access") or {}).get("index_access") or {}).get(
                "title_terminal_direct_match_count"
            )
        )
        for row in traces
    )
    terminal_conflict_count = sum(
        _as_int(
            ((row.get("physical_access") or {}).get("index_access") or {}).get(
                "terminal_conflict_count"
            )
        )
        for row in traces
    )
    lineage_qualifier_pruned_rows = sum(
        _as_int(
            ((row.get("physical_access") or {}).get("index_access") or {}).get(
                "lineage_qualifier_pruned_rows"
            )
        )
        for row in traces
    )
    constrained_decisions = [
        dict((row.get("decision") or {}).get("feasibility_constraints") or {})
        for row in traces
        if (row.get("decision") or {}).get("feasibility_constraints")
    ]
    cbo_decisions = [
        row for row in traces
        if str(row.get("policy") or "") == "support_aware_cbo"
        and len(((row.get("decision") or {}).get("ranked") or [])) >= 2
    ]
    cbo_different_from_fixed_early = 0
    for row in cbo_decisions:
        ranked = list((row.get("decision") or {}).get("ranked") or [])
        fixed_early = min(
            ranked,
            key=lambda item: (
                {VERIFY: 0, LOOKUP_JOIN: 1, COMPLETE: 2}.get(
                    str(item.get("operator") or ""), 99
                ),
                ranked.index(item),
            ),
        )
        if str(fixed_early.get("action_key") or "") != str(row.get("action_key") or ""):
            cbo_different_from_fixed_early += 1
    bswc_decisions = [
        row for row in traces
        if str(row.get("policy") or "") in BSWC_POLICIES
        and len(((row.get("decision") or {}).get("ranked") or [])) >= 2
    ]
    bswc_source_ids = {
        _as_int(row.get("source_id"), -1)
        for row in traces
        if row.get("operator") == SOURCE_INSPECT
        and _as_int(row.get("source_id"), -1) >= 0
    }
    return _json_safe({
        "version": executor_version,
        "query_id": str(query_id),
        "question": question,
        "policy": policy,
        "query_shape": str(plan.get("plan_kind") or ""),
        "supported": True,
        "plan_coverage_enforced": plan_coverage_enforced,
        "plan_coverage_status": plan_coverage_status,
        "question_complete_supported_witness": bool(
            question_complete_selected_witness
        ),
        "partial_supported_path": bool(
            plan_selected_witness and not question_semantically_complete
        ),
        "structured_admission": admitted,
        "structured_answer": answer if admitted else "",
        "phase_status": _execution_phase_status(
            plan,
            root_binding_succeeded=True,
            execution_entered=True,
            supported_witness_complete=bool(
                question_complete_selected_witness
            ),
            structured_answer_admitted=bool(admitted),
            failure_stage=(
                ""
                if admitted
                else (
                    "structured_answer_admission"
                    if question_complete_selected_witness
                    else "supported_witness_completion"
                )
            ),
            failure_reason="" if admitted else str(stop_reason or "not_admitted"),
        ),
        "output_path": (
            "verified_structured_admission" if admitted else "ordinary_baseline_fallback"
        ),
        "stop_reason": stop_reason,
        "execution_summary": {
            "index_unique_deleted_fact_ids": sorted({
                str(fact_id)
                for row in traces
                for fact_id in (
                    ((row.get("physical_access") or {}).get("index_access") or {}).get(
                        "perturbation_deleted_fact_ids"
                    )
                    or []
                )
                if str(fact_id)
            }),
            "index_unique_corrupted_fact_ids": sorted({
                str(fact_id)
                for row in traces
                for fact_id in (
                    ((row.get("physical_access") or {}).get("index_access") or {}).get(
                        "perturbation_corrupted_original_fact_ids"
                    )
                    or []
                )
                if str(fact_id)
            }),
            "index_tuple_deleted_count": sum(
                _as_int(
                    ((row.get("physical_access") or {}).get("index_access") or {}).get(
                        "perturbation_deleted_match_count"
                    )
                )
                for row in traces
            ),
            "index_tuple_corrupted_count": sum(
                _as_int(
                    ((row.get("physical_access") or {}).get("index_access") or {}).get(
                        "perturbation_corrupted_match_count"
                    )
                )
                for row in traces
            ),
            "verification_repair_reopen_count": sum(
                len((row.get("transition") or {}).get("reopened_parent_binding_ids") or [])
                for row in traces
            ),
            "root_restart_count": sum(
                len((row.get("transition") or {}).get("restarted_root_binding_ids") or [])
                for row in traces
            ),
            "verified_tuple_count": sum(
                bool(fact.get("verified_by_action_key"))
                for fact in state["facts"].values()
            ),
            "rejected_tuple_count": sum(
                fact.get("tuple_state") == TUPLE_REJECTED
                for fact in state["facts"].values()
            ),
            "unknown_tuple_count": sum(
                fact.get("support_status") == UNKNOWN
                for fact in state["facts"].values()
            ),
            "deferred_tuple_count": sum(
                fact.get("tuple_state") == TUPLE_DEFERRED
                for fact in state["facts"].values()
            ),
            "recovered_tuple_count": sum(
                fact.get("origin") == "query_local_overlay"
                for fact in state["facts"].values()
            ),
            "supported_recovered_tuple_count": sum(
                fact.get("origin") == "query_local_overlay"
                and fact.get("support_status") == SUPPORTED
                for fact in state["facts"].values()
            ),
            "invalidated_lineage_count": sum(
                _as_int(
                    (row.get("transition") or {}).get(
                        "invalidated_binding_count"
                    )
                )
                for row in traces
            ),
            "known_rejected_fact_filtered_count": sum(
                _as_int(
                    (row.get("transition") or {}).get(
                        "known_rejected_fact_filtered_count"
                    )
                )
                for row in traces
            ),
            "binding_contract_admitted_count": sum(
                _as_int((row.get("transition") or {}).get(
                    "binding_contract_admitted_count"
                ))
                for row in traces
            ),
            "binding_contract_rejected_count": sum(
                _as_int((row.get("transition") or {}).get(
                    "binding_contract_rejected_count"
                ))
                for row in traces
            ),
            "binding_contract_deferred_count": sum(
                _as_int((row.get("transition") or {}).get(
                    "binding_contract_deferred_count"
                ))
                for row in traces
            ),
            "noninvalidated_rejected_lineage_count": sum(
                not row.get("invalidated")
                and any(
                    str((state["facts"].get(fact_id) or {}).get("tuple_state") or "")
                    == TUPLE_REJECTED
                    for fact_id in (
                        row.get("path_fact_usage_ids")
                        or row.get("path_fact_ids")
                        or []
                    )
                )
                for row in state["bindings"]
            ),
            "complete_binding_count": sum(
                not row.get("invalidated") and _row_complete(row, len(operators))
                for row in state["bindings"]
            ),
            "verification_repair_success_count": sum(
                fact.get("verification_repair_candidate")
                and fact.get("support_status") == SUPPORTED
                for fact in state["facts"].values()
            ),
            "perturbation_repair_success_count": sum(
                fact.get("perturbation_repair_candidate")
                and fact.get("support_status") == SUPPORTED
                for fact in state["facts"].values()
            ),
            "verification_repair_candidate_count": sum(
                bool((row.get("transition") or {}).get("verification_repair_triggered"))
                and bool((row.get("transition") or {}).get("created_binding_ids"))
                for row in traces
            ),
            "perturbation_repair_candidate_count": sum(
                bool((row.get("transition") or {}).get("perturbation_repair_triggered"))
                and bool((row.get("transition") or {}).get("created_binding_ids"))
                for row in traces
            ),
            "verification_repair_witness_count": len(
                verification_repair_witnesses
            ),
            "repair_request_count": len(state.get("repair_requests") or {}),
            "terminal_critical_repair_scheduled_count": sum(
                str(row.get("reason") or "") == "scheduled_terminal_blocker"
                for row in state.get("repair_scheduler_events") or []
            ),
            "terminal_critical_repair_executed_count": sum(
                bool((row.get("transition") or {}).get("repair_request_key"))
                for row in traces
                if row.get("operator") == COMPLETE
            ),
            "repair_deferred_live_alternative_count": sum(
                str(row.get("reason") or "") == "deferred_live_alternative"
                for row in state.get("repair_scheduler_events") or []
            ),
            "repair_skipped_noncritical_count": sum(
                str(row.get("reason") or "") == "skipped_noncritical_obligation"
                for row in state.get("repair_scheduler_events") or []
            ),
            "perturbation_repair_witness_count": len(
                perturbation_repair_witnesses
            ),
            "query_local_overlay_fact_count": sum(
                fact.get("origin") == "query_local_overlay"
                for fact in state["facts"].values()
            ),
            "overlay_reuse_blocked_binding_count": sum(
                bool(row.get("overlay_reuse_blocked"))
                for row in state["bindings"]
            ),
            "plan_execution_complete_witness_count": len(witnesses),
            "question_complete_supported_witness_count": len(
                question_complete_witnesses
            ),
            "partial_supported_path_count": int(
                bool(witnesses) and not question_semantically_complete
            ),


            "verified_complete_witness_count": len(
                question_complete_witnesses
            ),
            "plan_coverage_violation_count": int(
                plan_coverage_enforced
                and plan_coverage_status not in {"", PLAN_COMPLETE}
            ),
            "unsafe_admission_count": sum(
                bool(fact.get("unsafe_admission"))
                for fact in state["facts"].values()
            ),
            "title_catalog_hit_actions": len(title_catalog_actions),
            "title_catalog_doc_count": sum(
                len((row.get("physical_access") or {}).get("title_catalog_doc_ids") or [])
                for row in title_catalog_actions
            ),
            "lineage_identity_rejected_count": identity_rejected_count,
            "identity_scoped_lookup_actions": identity_scoped_actions,
            "identity_scoped_rows": identity_scoped_rows,
            "identity_scoped_match_count": identity_scoped_matches,
            "title_terminal_direct_actions": title_terminal_direct_actions,
            "title_terminal_direct_rows": title_terminal_direct_rows,
            "title_terminal_direct_match_count": title_terminal_direct_matches,
            "terminal_conflict_count": terminal_conflict_count,
            "lineage_qualifier_pruned_rows": lineage_qualifier_pruned_rows,
            "ambiguous_terminal_fallback_count": int(
                reduction_reason == "ambiguous_terminal_conflict"
            ),
            "co_provenant_terminal_group_count": _as_int(
                reduction_detail.get("co_provenant_terminal_group_count")
            ),
            "identity_dominated_terminal_witness_count": _as_int(
                reduction_detail.get("identity_dominated_terminal_witness_count")
            ),
            "cross_provenance_terminal_conflict_count": _as_int(
                reduction_detail.get("cross_provenance_terminal_conflict_count")
            ),
            "conflict_quarantined_reader_evidence_count": int(
                reader_evidence_mode == "conflict_quarantined_trace_only"
            ),
            "terminal_sibling_verification_count": len(
                state["terminal_sibling_verification_fact_ids"]
            ),
            "performer_plan": bool(plan.get("performer_plan")),
            "newly_compiled_v7": bool(plan.get("newly_compiled_v7")),
            "feasibility_constrained_decision_count": len(constrained_decisions),
            "feasibility_deferred_action_count": sum(
                _as_int(row.get("deferred_action_count"))
                for row in constrained_decisions
            ),
            "branch_frontier_constraint_count": sum(
                "least_advanced_branch_frontier"
                in set(row.get("constraint_sequence") or [])
                for row in constrained_decisions
            ),
            "verify_before_join_constraint_count": sum(
                "verify_before_downstream_join"
                in set(row.get("constraint_sequence") or [])
                for row in constrained_decisions
            ),
            "cbo_eligible_decision_count": len(cbo_decisions),
            "cbo_different_from_fixed_early_count": (
                cbo_different_from_fixed_early
            ),
            "cbo_replanning_count": max(0, len(cbo_decisions) - 1),
            "cbo_calibrated_action_count": sum(
                bool((row.get("estimate") or {}).get("cbo_calibrated"))
                for row in traces
            ),
            "bswc_enabled": bswc_enabled,
            "bswc_eligible_decision_count": len(bswc_decisions),
            "bswc_replanning_count": max(0, len(bswc_decisions) - 1),
            "bswc_source_candidate_count": len(
                state.get("bswc_source_candidates") or []
            ),
            "bswc_source_inspection_count": _as_int(
                state.get("bswc_source_inspection_count")
            ),
            "bswc_unique_source_inspection_count": len(bswc_source_ids),
            "bswc_shared_source_reuse_count": _as_int(
                state.get("bswc_shared_source_reuse_count")
            ),
            "bswc_dense_probe_count": _as_int(
                state.get("bswc_dense_probe_count")
            ),
            "bswc_bounded_search_decision_count": sum(
                str((row.get("decision") or {}).get("bounded_search", {}).get(
                    "reason"
                ) or "") == "binding_aware_bounded_search"
                for row in traces
            ),
            "bswc_risk_aware_search_decision_count": sum(
                str((row.get("decision") or {}).get("bounded_search", {}).get(
                    "reason"
                ) or "") == "risk_aware_witness_search"
                for row in traces
            ),
            "bswc_adaptive_risk_cover_decision_count": sum(
                str((row.get("decision") or {}).get("bounded_search", {}).get(
                    "reason"
                ) or "") == "adaptive_risk_bounded_witness_cover"
                for row in traces
            ),
            "bswc_adaptive_risk_cover_stop_count": sum(
                bool(row.get("stopped"))
                and str(row.get("policy") or "")
                == "bswc_adaptive_risk_cover"
                for row in state.get("bswc_stopping_events") or []
            ),
            "bswc_feasibility_preserving_cover_decision_count": sum(
                str((row.get("decision") or {}).get("bounded_search", {}).get(
                    "reason"
                ) or "") == "feasibility_preserving_adaptive_witness_cover"
                for row in traces
            ),
            "bswc_feasibility_preserving_cover_stop_count": sum(
                bool(row.get("stopped"))
                and str(row.get("policy") or "")
                == "bswc_feasibility_preserving_cover"
                for row in state.get("bswc_stopping_events") or []
            ),
            "bswc_continuation_safe_cover_decision_count": sum(
                str((row.get("decision") or {}).get("bounded_search", {}).get(
                    "reason"
                ) or "") == "continuation_safe_witness_cover"
                for row in traces
            ),
            "bswc_continuation_safe_index_dominance_count": sum(
                str((row.get("decision") or {}).get("bounded_search", {}).get(
                    "phase"
                ) or "") == "index_dominant_lookup"
                for row in traces
            ),
            "bswc_continuation_safe_reserve_stop_count": sum(
                bool(row.get("stopped"))
                and str(row.get("policy") or "")
                == "bswc_continuation_safe_cover"
                and str(row.get("stop_reason") or "")
                == "continuation_reserve_would_be_violated"
                for row in state.get("bswc_stopping_events") or []
            ),
            "bswc_binding_qualified_continuation_decision_count": sum(
                str((row.get("decision") or {}).get("bounded_search", {}).get(
                    "reason"
                ) or "") == "binding_qualified_continuation_cover"
                for row in traces
            ),
            "bswc_binding_identity_source_dominance_count": sum(
                str((row.get("decision") or {}).get("bounded_search", {}).get(
                    "phase"
                ) or "") == "binding_identity_source_dominance"
                for row in traces
            ),
            "bswc_binding_qualified_dense_fallback_count": sum(
                str((row.get("decision") or {}).get("bounded_search", {}).get(
                    "phase"
                ) or "") == "binding_qualified_dense_fallback"
                for row in traces
            ),
            "bswc_forced_feasible_source_count": sum(
                str((row.get("decision") or {}).get("bounded_search", {}).get(
                    "phase"
                ) or "") == "forced_feasible_source"
                for row in traces
            ),
            "bswc_exact_adaptive_witness_decision_count": sum(
                str((row.get("decision") or {}).get("bounded_search", {}).get(
                    "reason"
                ) or "") == "exact_adaptive_supported_witness_dp"
                for row in traces
            ),
            "bswc_exact_adaptive_witness_stop_count": sum(
                bool(row.get("stopped"))
                and str(row.get("policy") or "")
                == "bswc_exact_adaptive_witness"
                for row in state.get("bswc_stopping_events") or []
            ),
            "bswc_precedence_aware_cover_decision_count": sum(
                str((row.get("decision") or {}).get("bounded_search", {}).get(
                    "reason"
                ) or "") == "precedence_aware_witness_cover"
                for row in traces
            ),
            "bswc_precedence_relational_closure_count": sum(
                str((row.get("decision") or {}).get("bounded_search", {}).get(
                    "phase"
                ) or "") == "relational_closure"
                for row in traces
            ),
            "bswc_precedence_frontier_expansion_count": sum(
                str((row.get("decision") or {}).get("bounded_search", {}).get(
                    "phase"
                ) or "") == "source_frontier_expansion"
                for row in traces
            ),
            "bswc_precedence_source_cover_count": sum(
                str((row.get("decision") or {}).get("bounded_search", {}).get(
                    "phase"
                ) or "") == "source_cover"
                for row in traces
            ),
            "bswc_precedence_dominance_pruned_count": sum(
                _as_int(
                    ((row.get("decision") or {}).get("bounded_search", {}).get(
                        "plan"
                    ) or {}).get("dominance_pruned_count")
                )
                for row in traces
            ),
            "bswc_precedence_budget_pruned_count": sum(
                _as_int(
                    ((row.get("decision") or {}).get("bounded_search", {}).get(
                        "plan"
                    ) or {}).get("budget_pruned_count")
                )
                for row in traces
            ),
            "bswc_precedence_aware_cover_stop_count": sum(
                bool(row.get("stopped"))
                and str(row.get("policy") or "")
                == "bswc_precedence_aware_cover"
                for row in state.get("bswc_stopping_events") or []
            ),
            "bswc_random_source_decision_count": sum(
                str((row.get("decision") or {}).get("bounded_search", {}).get(
                    "reason"
                ) or "") == "deterministic_query_keyed_random_source"
                for row in traces
            ),
            "bswc_calibrated_action_count": sum(
                bool((row.get("estimate") or {}).get("bswc_calibrated"))
                for row in traces
            ),
            "bswc_calibration_fallback_count": sum(
                str((row.get("estimate") or {}).get(
                    "bswc_calibration_backoff_level"
                ) or "") == "static_prior"
                for row in traces
                if str(row.get("operator") or "")
                in {BOUND_DENSE_PROBE, SOURCE_INSPECT, VERIFY}
            ),
            "bswc_feedback_update_count": len(
                state.get("bswc_feedback_events") or []
            ),
            "bswc_feedback_failure_count": sum(
                not bool(row.get("success"))
                for row in state.get("bswc_feedback_events") or []
            ),
            "bswc_stopping_evaluation_count": len(
                state.get("bswc_stopping_events") or []
            ),
            "bswc_low_value_stop_count": sum(
                bool(row.get("stopped"))
                for row in state.get("bswc_stopping_events") or []
            ),
            "support_reason_counts": dict(support_reason_counts),
        },
        "logical_plan": plan,
        "experiment_controls": {
            "completion_mode": str(getattr(backend, "completion_mode", "lazy")),
            "completion_query_mode": str(
                getattr(backend, "completion_query_mode", "obligation_bound")
            ),
            "verification_mode": str(getattr(backend, "verification_mode", "strict")),
            "repair_scheduler_mode": str(
                getattr(backend, "repair_scheduler_mode", "eager")
            ),
            "execution_mode": str(getattr(backend, "execution_mode", "saqe")),
            "relation_matching_mode": str(getattr(
                backend, "relation_matching_mode", "canonical"
            )),
            "predicate_subtype_constraints": bool(getattr(
                backend, "enable_predicate_subtype_constraints", True
            )),
            "type_role_constraints": bool(getattr(
                backend, "enable_type_role_constraints", True
            )),
            "composite_index_access": bool(getattr(
                backend, "enable_composite_index_access", True
            )),
            "index_delete_rate": _as_float(
                getattr(backend, "index_delete_rate", 0.0)
            ),
            "index_corrupt_rate": _as_float(
                getattr(backend, "index_corrupt_rate", 0.0)
            ),
            "index_corrupt_mode": str(
                getattr(backend, "index_corrupt_mode", "object")
            ),
            "perturbation_seed": _as_int(
                getattr(backend, "perturbation_seed", 0)
            ),
            "binding_obligation_contract_validation": bool(getattr(
                backend,
                "enable_binding_obligation_contract_validation",
                False,
            )),
            "query_local_overlay_reuse": bool(getattr(
                backend, "enable_query_local_overlay_reuse", True
            )),
            "cbo_calibration_path": str(getattr(
                backend, "cbo_calibration_path", ""
            )),
            "cbo_model_loaded": bool(getattr(
                backend, "cbo_cost_model", None
            )),
            "cbo_model_version": str(getattr(
                getattr(backend, "cbo_cost_model", None), "version", ""
            )),
            "cbo_min_samples": _as_int(getattr(
                backend, "cbo_min_samples", 0
            )),
            "cbo_risk_weight": _as_float(getattr(
                backend, "cbo_risk_weight", 0.0
            )),
            "cbo_cost_weight": _as_float(getattr(
                backend, "cbo_cost_weight", 0.0
            )),
            "bswc_policy": policy if bswc_enabled else "",
            "bswc_physical_source_selection": bswc_enabled,
            "bswc_source_calibration_path": str(getattr(
                backend, "bswc_source_calibration_path", ""
            )),
            "bswc_source_model_loaded": bool(getattr(
                getattr(backend, "bswc_source_value_model", None), "loaded", False
            )),
            "bswc_source_model_version": str(getattr(
                getattr(backend, "bswc_source_value_model", None), "version", ""
            )),
            "bswc_feedback_enabled": (
                policy in FEEDBACK_POLICIES
                and bool(getattr(backend, "enable_bswc_query_feedback", True))
            ),
            "bswc_stopping_enabled": policy in STOPPING_POLICIES,
            "bswc_witness_gain_enabled": policy in WITNESS_GAIN_POLICIES,
            "bswc_stop_threshold": _as_float(getattr(
                backend, "bswc_stop_threshold", 0.0
            )),
            "bswc_beam_width": _as_int(getattr(
                backend, "bswc_beam_width", 8
            )),
            "bswc_search_horizon": _as_int(getattr(
                backend, "bswc_search_horizon", 3
            )),
            "bswc_arwc_partial_value": _as_float(getattr(
                backend, "bswc_arwc_partial_value", 0.10
            )),
            "bswc_arwc_budget_penalty": _as_float(getattr(
                backend, "bswc_arwc_budget_penalty", 0.05
            )),
            "bswc_arwc_min_utility": _as_float(getattr(
                backend, "bswc_arwc_min_utility", 0.0
            )),
            "bswc_fpawc_stop_upper_bound": _as_float(getattr(
                backend, "bswc_fpawc_stop_upper_bound", 0.0
            )),
        },
        "bindings": [dict(row) for row in state["bindings"]],
        "binding_table": [
            {
                "binding_id": row["binding_id"],
                "parent_binding_id": row.get("parent_binding_id", ""),
                "active": bool(row.get("active")),
                "invalidated": bool(row.get("invalidated")),
                "next_operator_index": _as_int(row.get("next_operator_index")),
                "slot_bindings": dict(row.get("slot_bindings") or {}),
                "path_fact_ids": list(row.get("path_fact_ids") or []),
                "path_fact_usage_ids": list(
                    row.get("path_fact_usage_ids") or []
                ),
                "admission_eligible": bool(
                    question_semantically_complete and row in witnesses
                ),
                "plan_execution_complete": bool(row in witnesses),
            }
            for row in state["bindings"]
        ],
        "facts": [dict(value) for value in state["facts"].values()],
        "fact_usages": [dict(value) for value in state["facts"].values()],
        "obligation_history": list(state["obligation_history"]),
        "repair_scheduler_events": list(
            state.get("repair_scheduler_events") or []
        ),
        "bswc_feedback_events": list(state.get("bswc_feedback_events") or []),
        "bswc_stopping_events": list(state.get("bswc_stopping_events") or []),
        "plan_execution_witness_count": len(witnesses),
        "verified_witness_count": len(question_complete_witnesses),
        "selected_plan_witness": plan_selected_witness,
        "selected_witness": question_complete_selected_witness,
        "reader_evidence_fact_ids": reader_evidence_fact_ids,
        "reader_evidence_mode": reader_evidence_mode,
        "terminal_reduction": {
            "status": reduction_reason,
            **reduction_detail,
        },
        "operator_trace": traces,
        "cost_ledger": ledger.public(),
        "budget": {
            "max_actions": budget.max_actions,
            "max_retrieval_calls": budget.max_retrieval_calls,
            "max_model_calls": budget.max_model_calls,
            "max_scan_rows": budget.max_scan_rows,
            "max_tokens": budget.max_tokens,
            "reserve_model_calls": budget.reserve_model_calls,
            "reserve_tokens": budget.reserve_tokens,
        },
        "end_to_end_wall_ms": round((time.perf_counter() - started) * 1000, 6),
        "guarantee_scope": (
            "all_structured_witness_facts_supported_by_configured_verifier;"
            "question_level_plan_coverage_required_for_terminal_admission"
            + (
                ";verified_intermediate_entities_joined_with_canonical_title_identity"
                if executor_version in {
                    "paper_evidence_executor_v6", "paper_evidence_executor_v7",
                    "paper_evidence_executor_v7_1", "paper_evidence_executor_v7_2",
                }
                else ""
            )
        ),
    })
