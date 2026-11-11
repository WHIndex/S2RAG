"""Fixed-budget, reader-visible Schema Memory execution.

This module deliberately sits outside the broad evidence-admission vetoes.  It
implements one narrow causal path: a stable top-k evidence deficit can be
repaired by one entity-bound Schema Memory fact tuple, but only when the
tuple's cost-adjusted plan is positive.  The returned trace is designed for
paired ablations and does not use gold answers.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from .adaptive_query_executor_utils import lexical_query_coverage
from .cost_based_fact_join_plan_utils import optimize_cost_based_fact_join_plan
from .relation_fact_join_utils import (
    canonical_entity_key,
    canonical_schema_predicate_families,
    canonical_schema_predicate_subtypes,
    compile_stage_aware_fact_join_plan,
    entity_predicate_index_key,
    terminal_value_matches_contract,
)


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "what", "when", "where", "which", "who", "with",
}


def _tokens(value: Any) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: Any) -> float:
    return max(0.0, min(1.0, _as_float(value)))


def _relation_overlap(question: str, item: Dict[str, Any]) -> float:
    explicit = _clamp(item.get("relation_overlap"))
    schema = item.get("schema") or []
    triple = item.get("triple") or []
    predicate = item.get("predicate") or (
        triple[1] if isinstance(triple, (list, tuple)) and len(triple) > 1 else None
    )
    relation = predicate or (schema[1] if isinstance(schema, (list, tuple)) and len(schema) > 1 else "")
    relation_tokens = _tokens(relation)
    question_tokens = _tokens(question)
    lexical = len(relation_tokens & question_tokens) / float(max(1, len(relation_tokens)))
    return max(explicit, lexical)


def _tuple_fields(item: Dict[str, Any]) -> tuple[str, str, str]:
    triple = item.get("triple") or item.get("schema_fact_tuple")
    if not isinstance(triple, (list, tuple)) or len(triple) != 3:
        return "", "", ""
    return tuple(str(value or "").strip() for value in triple)


def _is_complete_instance_bound_lineage(item: Dict[str, Any]) -> bool:
    lineage_depth = int(item.get("lineage_depth", 0) or 0)
    return bool(
        item.get("materialization_role")
        == "instance_bound_terminal_fact_join"
        and item.get("operator_complete")
        and item.get("strict_operator_contract_validated")
        and item.get("rooted_lineage_complete")
        and lineage_depth >= 2
        and len(item.get("lineage") or []) == lineage_depth
        and len(item.get("lineage_source_doc_ids") or []) == lineage_depth
        and bool(item.get("lineage_root_key"))
    )


def _terminal_fact_supported_by_base(
    base_docs: Sequence[str],
    item: Dict[str, Any],
) -> bool:
    """Return whether reader-visible evidence already contains the terminal fact.

    Lexical question coverage is not evidence sufficiency: a top-k passage can
    repeat every query entity while omitting the answer-producing tuple.  For
    a complete instance-bound plan, test the final bound subject and projected
    value together in one reader-visible passage instead.
    """

    lineage = list(item.get("lineage") or [])
    terminal = lineage[-1] if lineage and isinstance(lineage[-1], dict) else {}
    triple = terminal.get("triple") or item.get("triple") or []
    if not isinstance(triple, (list, tuple)) or len(triple) != 3:
        return False
    terminal_subject = str(triple[0] or "")
    answer_value = str(item.get("answer_value") or triple[2] or "")
    subject_tokens = _tokens(terminal_subject)
    answer_tokens = _tokens(answer_value)
    if not subject_tokens or not answer_tokens:
        return False

    for doc in base_docs:
        doc_tokens = _tokens(doc)
        if not answer_tokens.issubset(doc_tokens):
            continue
        subject_overlap = len(subject_tokens & doc_tokens) / float(
            max(1, len(subject_tokens))
        )
        if subject_overlap >= 0.60:
            return True
    return False


def _lineage_supported_by_single_base_doc(
    base_docs: Sequence[str],
    item: Dict[str, Any],
) -> bool:
    """Return whether one reader passage already contains the whole join path."""

    lineage = [
        row for row in item.get("lineage") or []
        if isinstance(row, dict)
    ]
    required_pairs: List[tuple[Set[str], Set[str]]] = []
    for row in lineage:
        triple = row.get("triple") or []
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            return False
        subject_tokens = _tokens(triple[0])
        object_tokens = _tokens(triple[2])
        if not subject_tokens or not object_tokens:
            return False
        required_pairs.append((subject_tokens, object_tokens))
    if len(required_pairs) < 2:
        return False

    for doc in base_docs:
        doc_tokens = _tokens(doc)
        if all(
            subject_tokens.issubset(doc_tokens)
            and object_tokens.issubset(doc_tokens)
            for subject_tokens, object_tokens in required_pairs
        ):
            return True
    return False


def _normalized_terminal_value(item: Dict[str, Any]) -> str:
    value = str(item.get("answer_value") or _tuple_fields(item)[2]).strip()
    return " ".join(sorted(_tokens(value)))


def _terminal_contract_subtypes(
    question: str,
    predicate_families: Iterable[str],
    output_type: str,
) -> List[str]:
    families = {str(value or "").strip().lower() for value in predicate_families}
    output_type = str(output_type or "entity").strip().lower()
    question_text = " ".join(re.findall(r"[a-z0-9]+", str(question or "").lower()))
    result: Set[str] = set(families - {"birth", "death"})
    if "birth" in families:
        if output_type == "date" or re.search(r"\b(?:when|birth date|date of birth)\b", question_text):
            result.add("birth_date")
        elif output_type == "place" or re.search(r"\b(?:where|birthplace|place of birth)\b", question_text):
            result.add("birth_place")
        elif re.search(r"\b(?:birth name|name at birth|born as)\b", question_text):
            result.add("birth_name")
    if "death" in families:
        if re.search(r"\b(?:die|died|death) (?:of|from)\b|\bcause of death\b", question_text):
            result.add("death_cause")
        elif output_type == "date" or re.search(r"\b(?:when|death date|date of death)\b", question_text):
            result.add("death_date")
        elif output_type == "place" or re.search(r"\b(?:where|death place|place of death)\b", question_text):
            result.add("death_place")
    return sorted(result)


def _normalized_doc_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _lineage_source_visibility(
    item: Dict[str, Any],
    reader_visible_docs: Sequence[str],
    get_doc_content: Callable[[int], str],
) -> Dict[str, Any]:
    source_ids = list(dict.fromkeys(
        _as_int(value, -1)
        for value in item.get("lineage_source_doc_ids") or []
        if _as_int(value, -1) >= 0
    ))
    visible_docs = [str(value or "") for value in reader_visible_docs]
    normalized_visible = {_normalized_doc_text(value) for value in visible_docs}
    visible_ids: List[int] = []
    missing_ids: List[int] = []
    lookup_failed_ids: List[int] = []
    for source_id in source_ids:
        marker = re.compile(rf"\bsource_doc\s*=\s*{source_id}\b")
        if any(marker.search(value) for value in visible_docs):
            visible_ids.append(source_id)
            continue
        try:
            source_content = str(get_doc_content(source_id) or "")
        except Exception:
            source_content = ""
        normalized_source = _normalized_doc_text(source_content)
        if not normalized_source:
            lookup_failed_ids.append(source_id)
        elif normalized_source in normalized_visible:
            visible_ids.append(source_id)
        else:
            missing_ids.append(source_id)
    terminal_source_id = _as_int(
        ((item.get("lineage") or [{}])[-1] or {}).get("source_doc_id"),
        -1,
    )
    return {
        "lineage_source_doc_ids": source_ids,
        "lineage_visible_source_doc_ids": visible_ids,
        "lineage_missing_source_doc_ids": missing_ids,
        "lineage_source_lookup_failed_doc_ids": lookup_failed_ids,
        "source_visibility_complete": bool(source_ids) and not lookup_failed_ids,
        "source_novel": bool(missing_ids) and not lookup_failed_ids,
        "terminal_source_doc_id": terminal_source_id,
        "terminal_source_novel": terminal_source_id in missing_ids,
    }


def _source_grounded_terminal_fact_view(
    item: Dict[str, Any],
    *,
    allow_general_terminal_fact_view: bool = False,
) -> bool:
    """Return whether an existing passage yields a newly validated terminal fact.

    Passage-level novelty alone misses a useful case: a document may already be
    visible to the reader while its title/pronoun-bound terminal tuple is absent
    from the compact evidence view.  Keep this exception deliberately narrow so
    ordinary tuples from visible passages remain blocked.
    """
    if not (
        item.get("rooted_lineage_complete")
        and item.get("operator_complete")
        and item.get("strict_operator_contract_validated")
        and item.get("completion_causal_lineage_validated")
        and item.get("source_grounded_terminal_repair_used")
    ):
        return False





    unsafe_terminal_subtypes = {"", "geopolitical_location", "related"}
    legacy_terminal_subtypes = {"birth_place", "death_place"}
    return any(
        str(lineage.get("slot_stage") or "") == "final_answer"
        and bool(lineage.get("source_grounded"))
        and bool(lineage.get("terminal_source_repair"))
        and bool(lineage.get("instance_identity_validated"))
        and not bool(lineage.get("contextual_qualifier_risk"))
        and _as_int(lineage.get("source_doc_id"), -1) >= 0
        and _as_int(lineage.get("source_sentence_id"), -1) >= 0
        and any(
            subtype not in unsafe_terminal_subtypes
            and not subtype.startswith("relation:")
            and (
                allow_general_terminal_fact_view
                or subtype in legacy_terminal_subtypes
            )
            for subtype in {
                str(value or "").strip().lower()
                for value in lineage.get("predicate_subtypes") or []
            }
        )
        for lineage in item.get("bound_completion_lineage") or []
        if isinstance(lineage, dict)
    )


def protect_provenance_fact_view_surface(
    *,
    before: str,
    after: str,
    schema_critical_info: Optional[Dict[str, Any]],
    answer_slot_binding_optimizer: Optional[Dict[str, Any]] = None,
    min_incumbent_support: float = 0.50,
) -> Tuple[str, Dict[str, Any]]:
    """Protect a validated, independently supported full location surface."""
    optimizer = answer_slot_binding_optimizer or {}
    incumbent_support = _as_float(optimizer.get("answer_support_score"), 0.0)
    meta = {
        "applied": False,
        "reason": "not_applicable",
        "before": str(before or ""),
        "proposed_after": str(after or ""),
        "after": str(after or ""),
        "incumbent_support": round(incumbent_support, 4),
        "min_incumbent_support": round(float(min_incumbent_support), 4),
    }
    if (schema_critical_info or {}).get("materialization_enabled") is False:
        meta["reason"] = "schema_critical_materialization_shadow_only"
        return str(after or "").strip(), meta
    before_text = str(before or "").strip()
    after_text = str(after or "").strip()
    if not before_text or not after_text or before_text == after_text:
        return after_text, meta
    before_key = _normalized_doc_text(before_text)
    after_key = _normalized_doc_text(after_text)
    first_part = _normalized_doc_text(before_text.split(",", 1)[0])
    selected = [
        item
        for item in (schema_critical_info or {}).get("selected") or []
        if isinstance(item, dict)
        and item.get("fact_view_novel")
        and _normalized_doc_text(item.get("answer_value")) == before_key
        and {
            str(value or "").strip().lower()
            for value in item.get("predicate_subtypes") or []
        }
        & {"birth_place", "death_place"}
    ]
    if (
        selected
        and "," in before_text
        and after_key == first_part
        and len(_tokens(after_text)) < len(_tokens(before_text))
        and incumbent_support >= float(min_incumbent_support)
    ):
        meta.update({
            "applied": True,
            "reason": "validated_fact_view_full_location_protected",
            "after": before_text,
            "answer_value": str(selected[0].get("answer_value") or ""),
            "lineage_source_doc_ids": list(
                selected[0].get("lineage_source_doc_ids") or []
            ),
        })
        return before_text, meta
    if selected and incumbent_support < float(min_incumbent_support):
        meta["reason"] = "insufficient_independent_incumbent_support"
    return after_text, meta


def _terminal_slot_contract(question: str) -> Dict[str, Any]:
    plan = compile_stage_aware_fact_join_plan(
        question,
        enable_bound_completion=True,
    )
    operators = list(plan.get("operators") or [])
    terminal = next(
        (
            operator
            for operator in reversed(operators)
            if str(operator.get("slot_plan_stage") or "") == "final_answer"
            and str(operator.get("output_slot") or "") == "$answer"
        ),
        {},
    )
    predicate_families = {
        str(value or "").strip().lower()
        for value in terminal.get("predicate_families") or []
        if str(value or "").strip()
    }
    relation_tokens = {
        str(value or "").strip().lower()
        for value in terminal.get("relation_tokens") or []
        if str(value or "").strip()
    }
    valid = bool(
        plan.get("compile_status") == "compiled"
        and plan.get("requires_bridge")
        and terminal
        and (predicate_families or relation_tokens)
    )
    operator_id = str(terminal.get("operator_id") or "terminal")
    output_type = str(terminal.get("output_type") or plan.get("final_answer_type") or "entity")
    return {
        "valid": valid,
        "compile_status": str(plan.get("compile_status") or "unsupported"),
        "compile_reason": str(plan.get("compile_reason") or ""),
        "plan_kind": str(plan.get("plan_kind") or "unsupported"),
        "requires_bridge": bool(plan.get("requires_bridge")),
        "operator_id": operator_id,
        "slot_id": f"terminal:{operator_id}",
        "predicate_families": sorted(predicate_families),
        "predicate_subtypes": _terminal_contract_subtypes(
            question,
            predicate_families,
            output_type,
        ),
        "relation_tokens": sorted(relation_tokens),
        "output_type": output_type,
        "input_slot": str(terminal.get("input_slot") or ""),
        "output_slot": str(terminal.get("output_slot") or ""),
        "logical_plan": plan,
    }


def probe_terminal_schema_candidates(
    *,
    question: str,
    top_k_facts: Sequence[Sequence[Any]],
    relation_index: Mapping[str, Any],
    rank_by_doc: Mapping[int, int],
    top_doc_ids: Sequence[int],
    num_passages: int,
    get_doc_content: Callable[[int], str],
    max_rows_per_lookup: int = 16,
    max_candidates: int = 32,
) -> Dict[str, Any]:
    """Probe terminal tuples through the exact (entity, predicate) index.

    The regular Schema Memory scan starts from schemas observed in retrieved
    facts.  This operator instead binds their endpoints and probes the final
    logical-plan predicate, preventing an intermediate bridge tuple from being
    mistaken for answer-producing evidence.
    """

    contract = _terminal_slot_contract(question)
    info: Dict[str, Any] = {
        "enabled": True,
        "reason": "terminal_contract_unavailable",
        "terminal_contract": {
            key: value for key, value in contract.items() if key != "logical_plan"
        },
        "lookup_count": 0,
        "hit_count": 0,
        "scan_rows": 0,
        "subtype_mismatch_count": 0,
        "physical_access_path": "unavailable",
        "candidate_count": 0,
        "candidates": [],
    }
    if not contract["valid"]:
        return info

    items = list((relation_index or {}).get("items") or [])
    subject_index = (relation_index or {}).get("subject_predicate_family_to_items") or {}
    object_index = (relation_index or {}).get("object_predicate_family_to_items") or {}
    subject_subtype_index = (
        (relation_index or {}).get("subject_predicate_subtype_to_items") or {}
    )
    object_subtype_index = (
        (relation_index or {}).get("object_predicate_subtype_to_items") or {}
    )
    if not items or (not subject_index and not object_index):
        info["reason"] = "terminal_composite_index_unavailable"
        return info

    bound_surfaces: Dict[str, str] = {}
    for fact in top_k_facts or []:
        if not isinstance(fact, (list, tuple)) or len(fact) != 3:
            continue
        for endpoint in (fact[0], fact[2]):
            key = canonical_entity_key(endpoint)
            if key:
                bound_surfaces.setdefault(key, str(endpoint or "").strip())
    if not bound_surfaces:
        info["reason"] = "no_bound_upstream_endpoint"
        return info

    q_tokens = _tokens(question)
    current_top = {int(value) for value in list(top_doc_ids or [])}
    candidate_by_signature: Dict[tuple, Dict[str, Any]] = {}
    total_passages = max(1, int(num_passages))
    row_cap = max(1, int(max_rows_per_lookup))
    use_subtype_index = bool(
        contract["predicate_subtypes"]
        and (subject_subtype_index or object_subtype_index)
    )
    info["physical_access_path"] = (
        "exact_entity_predicate_subtype_index"
        if use_subtype_index else "entity_predicate_family_index_with_subtype_filter"
    )
    access_keys = (
        contract["predicate_subtypes"]
        if use_subtype_index else contract["predicate_families"]
    )
    for bound_key, bound_surface in bound_surfaces.items():
        for access_key in access_keys:
            composite_key = entity_predicate_index_key(bound_key, access_key)
            for probe_role, family_index in (
                (
                    "subject",
                    subject_subtype_index if use_subtype_index else subject_index,
                ),
                (
                    "object",
                    object_subtype_index if use_subtype_index else object_index,
                ),
            ):
                info["lookup_count"] += 1
                item_ids = list(family_index.get(composite_key) or [])
                if not item_ids:
                    continue
                info["hit_count"] += 1
                capped_ids = item_ids[:row_cap]
                info["scan_rows"] += len(capped_ids)
                effective_fanout = max(1, len(item_ids))
                effective_idf = math.log(
                    (1.0 + total_passages) / (1.0 + effective_fanout)
                ) / max(math.log(1.0 + total_passages), 1e-9)
                for raw_item_id in capped_ids:
                    item_id = _as_int(raw_item_id, -1)
                    if item_id < 0 or item_id >= len(items):
                        continue
                    item = dict(items[item_id] or {})
                    doc_id = _as_int(item.get("doc_idx"), -1)
                    if doc_id < 0:
                        continue
                    subject = str(item.get("subject") or "").strip()
                    predicate = str(item.get("relation") or "").strip()
                    object_value = str(item.get("object") or "").strip()
                    if not subject or not predicate or not object_value:
                        continue
                    answer_value = object_value if probe_role == "subject" else subject
                    if canonical_entity_key(answer_value) == bound_key:
                        continue
                    candidate_families = {
                        str(value or "").strip().lower()
                        for value in (
                            item.get("predicate_families")
                            or canonical_schema_predicate_families(predicate)
                        )
                    }
                    if not (candidate_families & set(contract["predicate_families"])):
                        continue
                    candidate_subtypes = set(
                        item.get("predicate_subtypes")
                        or canonical_schema_predicate_subtypes(
                            predicate,
                            answer_value,
                        )
                    )
                    subtype_match = bool(
                        not contract["predicate_subtypes"]
                        or candidate_subtypes & set(contract["predicate_subtypes"])
                    )
                    if not subtype_match:
                        info["subtype_mismatch_count"] += 1
                        continue
                    if not terminal_value_matches_contract(
                        answer_value,
                        contract["output_type"],
                        contract["relation_tokens"],
                    ):
                        continue
                    schemas = [
                        tuple(signature)
                        for signature in item.get("schema_signatures_v2") or []
                        if isinstance(signature, (list, tuple))
                        and len(signature) == 4
                        and str(signature[1]) in candidate_families
                    ]
                    schema = schemas[0] if schemas else (
                        "entity", access_key, "entity", "subject_to_object"
                    )
                    answer_type = str(schema[2] if probe_role == "subject" else schema[0])
                    rank = int(rank_by_doc.get(doc_id, total_passages))
                    rank_prior = 1.0 / (1.0 + math.log1p(rank + 1))
                    tuple_tokens = _tokens(" ".join((subject, predicate, object_value)))
                    query_overlap = len(tuple_tokens & q_tokens) / float(max(1, len(q_tokens)))
                    score = min(
                        1.0,
                        0.66
                        + 0.18 * effective_idf
                        + 0.10 * query_overlap
                        + 0.06 * rank_prior,
                    )
                    try:
                        content = str(get_doc_content(doc_id) or "")
                    except Exception:
                        content = ""
                    title = content.split("\n", 1)[0].strip() if content else ""
                    signature_key = (
                        doc_id,
                        canonical_entity_key(subject),
                        " ".join(sorted(_tokens(predicate))),
                        canonical_entity_key(object_value),
                        probe_role,
                    )
                    candidate = {
                        "doc_id": doc_id,
                        "title": title,
                        "score": round(score, 6),
                        "original_rank": rank,
                        "schema": list(schema),
                        "schema_frequency": effective_fanout,
                        "schema_idf": round(effective_idf, 6),
                        "schema_support": 1.0,
                        "context_overlap": round(query_overlap, 6),
                        "query_overlap": round(query_overlap, 6),
                        "relation_overlap": 1.0,
                        "join_key_overlap": 1.0,
                        "join_key_containment": 1.0,
                        "join_key_exact": True,
                        "entity_bound": True,
                        "active_join_role": probe_role,
                        "candidate_join_role": probe_role,
                        "active_join_key_tokens": sorted(_tokens(bound_surface)),
                        "candidate_join_key_tokens": sorted(_tokens(bound_surface)),
                        "matched_join_tokens": sorted(_tokens(bound_surface)),
                        "join_key_matched": True,
                        "already_top5": doc_id in current_top,
                        "schema_fact_tuple": True,
                        "triple": [subject, predicate, object_value],
                        "predicate": predicate,
                        "answer_value": answer_value,
                        "answer_type": answer_type or "entity",
                        "tuple_selection_score": 1.0,
                        "matched_endpoint_tokens": sorted(_tokens(bound_surface)),
                        "candidate_join_endpoint_role": probe_role,
                        "direction": (
                            "subject_to_object" if probe_role == "subject"
                            else "object_to_subject"
                        ),
                        "materialization_role": "terminal_schema_fact_tuple",
                        "logical_plan_stage": "final_answer",
                        "terminal_operator_id": contract["operator_id"],
                        "predicate_contract_match": True,
                        "predicate_subtype_match": subtype_match,
                        "predicate_subtypes": sorted(candidate_subtypes),
                        "expected_predicate_subtypes": list(
                            contract["predicate_subtypes"]
                        ),
                        "terminal_slot_match": True,
                        "answer_type_match": True,
                        "rooted_lineage_complete": True,
                        "covered_slot_ids": [contract["slot_id"]],
                        "uncovered_slot_ids": [],
                        "physical_access_path": info["physical_access_path"],
                        "estimated_endpoint_scan_rows": effective_fanout,
                    }
                    previous = candidate_by_signature.get(signature_key)
                    if previous is None or candidate["score"] > previous["score"]:
                        candidate_by_signature[signature_key] = candidate

    ranked = sorted(
        candidate_by_signature.values(),
        key=lambda row: (float(row.get("score", 0.0)), -int(row.get("original_rank", 0))),
        reverse=True,
    )[: max(1, int(max_candidates))]
    info["candidates"] = ranked
    info["candidate_count"] = len(ranked)
    info["reason"] = "terminal_candidates_generated" if ranked else "terminal_composite_probe_miss"
    return info


def _sentence_projection(content: str, item: Dict[str, Any], *, max_sentences: int = 2) -> str:
    lineage_projection = str(item.get("lineage_reader_text") or "").strip()
    if (
        lineage_projection
        and item.get("materialization_role")
        == "instance_bound_terminal_fact_join"
        and item.get("rooted_lineage_complete")
        and item.get("operator_complete")
    ):
        return lineage_projection
    subject, predicate, object_value = _tuple_fields(item)
    title = str(item.get("title") or "").splitlines()[0].strip()
    terms = _tokens(" ".join((subject, predicate, object_value, str(item.get("answer_value") or ""))))
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", str(content or "")) if part.strip()]
    ranked = []
    for index, sentence in enumerate(sentences):
        overlap = len(terms & _tokens(sentence))
        ranked.append((overlap, -index, sentence))
    ranked.sort(reverse=True)
    selected = [row[2] for row in ranked[: max(1, int(max_sentences))] if row[2]]
    if not selected:
        selected = [str(content or "").strip()[:1200]]
    prefix = f"Title: {title}\n" if title else ""
    fact = f"Fact tuple: ({subject}, {predicate}, {object_value})\n"
    return (
        f"[Schema Memory fact join; source_doc={_as_int(item.get('doc_id'), -1)}]\n"
        f"{prefix}{fact}"
        + "\n".join(selected)
    ).strip()


def select_schema_critical_evidence(
    *,
    question: str,
    base_docs: Sequence[str],
    reader_visible_docs: Optional[Sequence[str]] = None,
    candidate_items: Iterable[Dict[str, Any]],
    get_doc_content: Callable[[int], str],
    qa_top_k: int,
    max_append: int = 1,
    deficit_coverage_threshold: float = 0.72,
    min_score: float = 0.58,
    min_schema_idf: float = 0.50,
    min_join_key_overlap: float = 0.75,
    min_relation_overlap: float = 0.05,
    max_original_rank: int = 80,
    token_budget: int = 220,
    compute_budget: float = 1.0,
    min_objective: float = 0.08,
    risk_weight: float = 0.42,
    token_weight: float = 0.12,
    compute_weight: float = 0.04,
    require_bridge_cues: bool = False,
    bridge_cues: bool = False,
    require_terminal_closure: bool = True,
    require_instance_bound_lineage: bool = False,
    materialize_distributed_join_view: bool = False,
    require_unique_terminal_value: bool = True,
    require_source_novelty: bool = False,
    require_terminal_subtype_match: bool = False,
    allow_general_terminal_fact_view: bool = False,
) -> Dict[str, Any]:
    """Select at most one strict schema bridge for the reader-visible pool."""

    coverage = lexical_query_coverage(question, list(base_docs)[: max(1, int(qa_top_k))])
    reader_visible_docs = list(
        reader_visible_docs if reader_visible_docs is not None else base_docs
    )
    candidate_items = [item for item in (candidate_items or []) if isinstance(item, dict)]
    terminal_contract = _terminal_slot_contract(question)
    public_terminal_contract = {
        key: value
        for key, value in terminal_contract.items()
        if key != "logical_plan"
    }
    info: Dict[str, Any] = {
        "enabled": True,
        "triggered": False,
        "accepted": False,
        "reason": "not_evaluated",
        "base_query_coverage": round(float(coverage), 6),
        "deficit_coverage_threshold": round(float(deficit_coverage_threshold), 6),
        "deficit": bool(coverage < float(deficit_coverage_threshold)),
        "bridge_cues": bool(bridge_cues),
        "candidate_count": 0,
        "eligible_candidate_count": 0,
        "terminal_closed_candidate_count": 0,
        "accepted_count": 0,
        "rejected_reason_counts": {},
        "selected_candidate_ids": [],
        "selected_doc_ids": [],
        "selected": [],
        "plan": {},
        "reader_projection_token_count": 0,
        "source_doc_token_count": 0,
        "projection_token_savings": 0,
        "require_terminal_closure": bool(require_terminal_closure),
        "require_instance_bound_lineage": bool(
            require_instance_bound_lineage
        ),
        "require_source_novelty": bool(require_source_novelty),
        "require_terminal_subtype_match": bool(
            require_terminal_subtype_match
        ),
        "allow_general_terminal_fact_view": bool(
            allow_general_terminal_fact_view
        ),
        "terminal_contract": public_terminal_contract,
        "logical_plan": terminal_contract.get("logical_plan") or {},
    }
    strict_lineage_items = [
        item
        for item in candidate_items
        if _is_complete_instance_bound_lineage(item)
    ]
    source_visibility_by_id = {
        id(item): _lineage_source_visibility(
            item,
            reader_visible_docs,
            get_doc_content,
        )
        for item in strict_lineage_items
    }
    source_novel_lineage_items = [
        item
        for item in strict_lineage_items
        if source_visibility_by_id[id(item)]["source_novel"]
    ]
    raw_fact_novel_lineage_items = [
        item
        for item in strict_lineage_items
        if _source_grounded_terminal_fact_view(
            item,
            allow_general_terminal_fact_view=allow_general_terminal_fact_view,
        )
    ]
    fact_view_terminal_values = {
        _normalized_terminal_value(item)
        for item in raw_fact_novel_lineage_items
        if _normalized_terminal_value(item)
    }
    unique_fact_view_terminal_value = len(fact_view_terminal_values) == 1
    fact_novel_lineage_items = (
        raw_fact_novel_lineage_items
        if unique_fact_view_terminal_value else []
    )
    novelty_eligible_ids = {
        id(item) for item in source_novel_lineage_items
    } | {
        id(item) for item in fact_novel_lineage_items
    }
    unsupported_strict_lineage_items = [
        item
        for item in strict_lineage_items
        if not _terminal_fact_supported_by_base(
            list(base_docs)[: max(1, int(qa_top_k))], item
        )
    ]
    deficit_lineage_items = [
        item
        for item in unsupported_strict_lineage_items
        if (
            not require_source_novelty
            or id(item) in novelty_eligible_ids
        )
    ]
    distributed_lineage_items = [
        item
        for item in strict_lineage_items
        if len(set(item.get("lineage_source_doc_ids") or [])) >= 2
        and not _lineage_supported_by_single_base_doc(
            list(base_docs)[: max(1, int(qa_top_k))], item
        )
    ]
    source_novel_distributed_lineage_items = [
        item
        for item in distributed_lineage_items
        if source_visibility_by_id[id(item)]["source_novel"]
    ]
    fact_novel_distributed_lineage_items = [
        item
        for item in distributed_lineage_items
        if item in fact_novel_lineage_items
    ]
    eligible_distributed_lineage_items = [
        item
        for item in distributed_lineage_items
        if (
            not require_source_novelty
            or id(item) in novelty_eligible_ids
        )
    ]
    terminal_values = {
        _normalized_terminal_value(item)
        for item in eligible_distributed_lineage_items
        if _normalized_terminal_value(item)
    }
    unique_terminal_value = len(terminal_values) == 1
    schema_fact_view_deficit = bool(
        fact_novel_lineage_items
        and unique_fact_view_terminal_value
    )
    schema_join_view_deficit = bool(
        materialize_distributed_join_view
        and eligible_distributed_lineage_items
        and (unique_terminal_value or not require_unique_terminal_value)
    )
    instance_bound_chain_deficit = bool(deficit_lineage_items)
    info.update({
        "instance_bound_lineage_candidate_count": len(strict_lineage_items),
        "instance_bound_chain_deficit_count": len(
            deficit_lineage_items
        ),
        "instance_bound_chain_deficit": instance_bound_chain_deficit,
        "distributed_lineage_candidate_count": len(
            distributed_lineage_items
        ),
        "source_novel_lineage_candidate_count": len(
            source_novel_lineage_items
        ),
        "source_novel_distributed_lineage_candidate_count": len(
            source_novel_distributed_lineage_items
        ),
        "fact_novel_lineage_candidate_count": len(
            fact_novel_lineage_items
        ),
        "raw_fact_novel_lineage_candidate_count": len(
            raw_fact_novel_lineage_items
        ),
        "fact_view_terminal_value_count": len(
            fact_view_terminal_values
        ),
        "fact_view_terminal_value_unique": bool(
            unique_fact_view_terminal_value
        ),
        "fact_novel_distributed_lineage_candidate_count": len(
            fact_novel_distributed_lineage_items
        ),
        "redundant_distributed_lineage_candidate_count": max(
            0,
            len(distributed_lineage_items)
            - len({
                id(item)
                for item in source_novel_distributed_lineage_items
                + fact_novel_distributed_lineage_items
            }),
        ),
        "lineage_visible_source_count": sum(
            len(value["lineage_visible_source_doc_ids"])
            for value in source_visibility_by_id.values()
        ),
        "lineage_missing_source_count": sum(
            len(value["lineage_missing_source_doc_ids"])
            for value in source_visibility_by_id.values()
        ),
        "lineage_source_lookup_failure_count": sum(
            len(value["lineage_source_lookup_failed_doc_ids"])
            for value in source_visibility_by_id.values()
        ),
        "distributed_terminal_value_count": len(terminal_values),
        "distributed_terminal_value_unique": unique_terminal_value,
        "schema_join_view_deficit": schema_join_view_deficit,
        "schema_fact_view_deficit": schema_fact_view_deficit,
        "materialize_distributed_join_view": bool(
            materialize_distributed_join_view
        ),
        "lexical_sufficiency_bypassed": bool(
            coverage >= float(deficit_coverage_threshold)
            and (
                instance_bound_chain_deficit
                or schema_join_view_deficit
                or schema_fact_view_deficit
            )
        ),
    })
    info["deficit"] = bool(
        coverage < float(deficit_coverage_threshold)
        or instance_bound_chain_deficit
        or schema_join_view_deficit
        or schema_fact_view_deficit
    )
    if (
        coverage >= float(deficit_coverage_threshold)
        and not instance_bound_chain_deficit
        and not schema_join_view_deficit
        and not schema_fact_view_deficit
    ):
        info["reason"] = "base_evidence_sufficient"
        return info
    if require_bridge_cues and not bridge_cues:
        info["reason"] = "bridge_cues_required"
        return info
    if require_terminal_closure and not terminal_contract["valid"]:
        info["triggered"] = True
        info["candidate_count"] = len(candidate_items)
        info["reason"] = "terminal_contract_unavailable"
        info["rejected_reason_counts"] = {
            "terminal_contract_unavailable": len(candidate_items)
        }
        return info

    info["triggered"] = True
    candidates: List[Dict[str, Any]] = []
    reject_counts: Dict[str, int] = {}
    base_doc_text = {str(doc).strip() for doc in reader_visible_docs}
    for item in candidate_items:
        doc_id = _as_int(item.get("doc_id"), -1)
        score = _as_float(item.get("score"))
        schema_idf = _as_float(item.get("schema_idf"))
        join_overlap = _as_float(item.get("join_key_overlap"))
        original_rank = _as_int(item.get("original_rank"), 10**9)
        relation_overlap = _relation_overlap(question, item)
        subject, predicate, object_value = _tuple_fields(item)
        candidate_families = {
            str(value or "").strip().lower()
            for value in canonical_schema_predicate_families(predicate)
            if str(value or "").strip()
        }
        terminal_families = set(terminal_contract["predicate_families"])
        terminal_subtypes = set(terminal_contract["predicate_subtypes"])
        predicate_contract_match = bool(
            candidate_families & terminal_families
            or (
                not terminal_families
                and _tokens(predicate) & set(terminal_contract["relation_tokens"])
            )
        )
        answer_value = str(item.get("answer_value") or object_value).strip()
        candidate_subtypes = set(
            canonical_schema_predicate_subtypes(predicate, answer_value)
        )
        predicate_subtype_match = bool(
            not terminal_subtypes
            or candidate_subtypes & terminal_subtypes
        )
        answer_type_match = terminal_value_matches_contract(
            answer_value,
            terminal_contract["output_type"],
            terminal_contract["relation_tokens"],
        )
        lineage_tokens = set(item.get("matched_endpoint_tokens") or []) | set(
            item.get("matched_join_tokens") or []
        )
        instance_bound_lineage = _is_complete_instance_bound_lineage(item)
        source_visibility = source_visibility_by_id.get(id(item), {
            "lineage_source_doc_ids": [],
            "lineage_visible_source_doc_ids": [],
            "lineage_missing_source_doc_ids": [],
            "lineage_source_lookup_failed_doc_ids": [],
            "source_visibility_complete": not instance_bound_lineage,
            "source_novel": False,
            "terminal_source_doc_id": -1,
            "terminal_source_novel": False,
        })
        fact_view_novel = _source_grounded_terminal_fact_view(
            item,
            allow_general_terminal_fact_view=allow_general_terminal_fact_view,
        )
        join_view_materialization_candidate = bool(
            schema_join_view_deficit
            and instance_bound_lineage
            and item in eligible_distributed_lineage_items
        )
        fact_view_materialization_candidate = bool(
            schema_fact_view_deficit
            and instance_bound_lineage
            and item in fact_novel_lineage_items
        )
        rooted_lineage_complete = bool(
            instance_bound_lineage
            or (
                not require_instance_bound_lineage
                and item.get("rooted_lineage_complete")
            )
            or (
                not require_instance_bound_lineage
                and
                item.get("entity_bound")
                and (item.get("join_key_exact") or join_overlap >= float(min_join_key_overlap))
                and lineage_tokens
            )
        )
        terminal_slot_match = bool(
            predicate_contract_match
            and (
                not require_terminal_subtype_match
                or predicate_subtype_match
            )
            and answer_type_match
            and rooted_lineage_complete
        )
        canonical_relation_bound = bool(
            instance_bound_lineage
            and predicate_contract_match
        )
        effective_relation_overlap = (
            1.0 if canonical_relation_bound else relation_overlap
        )
        checks = [
            (doc_id >= 0, "invalid_doc_id"),
            (
                original_rank >= max(1, int(qa_top_k))
                or join_view_materialization_candidate
                or fact_view_materialization_candidate,
                "already_in_top_k",
            ),
            (
                instance_bound_lineage
                or original_rank <= int(max_original_rank),
                "rank_out_of_bound",
            ),
            (
                not require_instance_bound_lineage
                or instance_bound_lineage,
                "instance_bound_lineage_required",
            ),
            (
                not require_source_novelty
                or not instance_bound_lineage
                or bool(source_visibility["source_visibility_complete"]),
                "lineage_source_lookup_failed",
            ),
            (
                not require_source_novelty
                or not instance_bound_lineage
                or bool(source_visibility["source_novel"])
                or fact_view_novel,
                "lineage_sources_already_visible",
            ),
            (bool(item.get("entity_bound")), "entity_not_bound"),
            (bool(item.get("join_key_exact")) or join_overlap >= float(min_join_key_overlap), "join_key_not_bound"),
            (score >= float(min_score), "score_below_threshold"),
            (schema_idf >= float(min_schema_idf), "schema_not_selective"),
            (
                effective_relation_overlap >= float(min_relation_overlap),
                "relation_not_bound",
            ),
            (bool(subject and predicate and object_value), "fact_tuple_missing"),
            (bool(answer_value), "answer_value_missing"),
        ]
        if require_terminal_closure:
            checks.extend([
                (predicate_contract_match, "terminal_predicate_mismatch"),
                (
                    not require_terminal_subtype_match
                    or predicate_subtype_match,
                    "terminal_predicate_subtype_mismatch",
                ),
                (answer_type_match, "terminal_answer_type_mismatch"),
                (rooted_lineage_complete, "terminal_lineage_unbound"),
                (terminal_slot_match, "terminal_slot_not_closed"),
            ])
        failed = next((reason for passed, reason in checks if not passed), None)
        if failed:
            reject_counts[failed] = reject_counts.get(failed, 0) + 1
            continue
        try:
            content = str(get_doc_content(doc_id) or "").strip()
        except Exception:
            reject_counts["source_content_lookup_failed"] = reject_counts.get("source_content_lookup_failed", 0) + 1
            continue
        if not content:
            reject_counts["source_content_missing"] = reject_counts.get("source_content_missing", 0) + 1
            continue
        projection = _sentence_projection(content, item)
        if projection in base_doc_text or (
            content in base_doc_text
            and not join_view_materialization_candidate
            and not fact_view_materialization_candidate
        ):
            reject_counts["duplicate_source"] = reject_counts.get("duplicate_source", 0) + 1
            continue
        projection_tokens = max(1, len(_tokens(projection)))
        source_tokens = max(1, len(_tokens(content)))
        candidate_id = f"schema_critical:{doc_id}:{original_rank}"
        risk = min(
            1.0,
            0.44 * (1.0 - score)
            + 0.24 * (1.0 - schema_idf)
            + 0.20 * (1.0 - join_overlap)
            + 0.12 * (1.0 - effective_relation_overlap),
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "source": "query_schema_memory",
                "doc_id": doc_id,
                "title": str(item.get("title") or ""),
                "reader_text": projection,
                "source_content": content,
                "utility": score,
                "risk": risk,
                "coverage_gain": max(
                    0.35
                    if instance_bound_chain_deficit
                    else 0.25
                    if (
                        join_view_materialization_candidate
                        or fact_view_materialization_candidate
                    )
                    else 0.05,
                    1.0 - coverage,
                ),
                "relation_overlap": effective_relation_overlap,
                "lexical_relation_overlap": relation_overlap,
                "canonical_relation_bound": canonical_relation_bound,
                "provenance_score": 1.0 if doc_id >= 0 else 0.0,
                "schema_selectivity": schema_idf,
                "join_key_overlap": max(join_overlap, 1.0 if item.get("join_key_exact") else join_overlap),
                "predicate_contract_match": bool(predicate_contract_match),
                "predicate_subtype_match": bool(predicate_subtype_match),
                "predicate_subtypes": sorted(candidate_subtypes),
                "expected_predicate_subtypes": sorted(terminal_subtypes),
                "terminal_slot_match": bool(terminal_slot_match),
                "answer_type_match": bool(answer_type_match),
                "rooted_lineage_complete": bool(rooted_lineage_complete),
                "logical_plan_stage": "final_answer",
                "terminal_operator_id": terminal_contract["operator_id"],
                "token_count": projection_tokens,
                "estimated_endpoint_scan_rows": max(1, _as_int(item.get("estimated_endpoint_scan_rows"), 1)),
                "compute_cost": 0.12,
                "covered_slot_ids": (
                    [terminal_contract["slot_id"]]
                    if terminal_slot_match
                    else list(item.get("covered_slot_ids") or [])
                ),
                "uncovered_slot_ids": list(item.get("uncovered_slot_ids") or []),
                "input_extra_rank": original_rank,
                "original_rank": original_rank,
                "schema_fact_tuple": True,
                "triple": [subject, predicate, object_value],
                "answer_value": answer_value,
                "source_doc_id": doc_id,
                "schema_idf": schema_idf,
                "join_key_exact": bool(item.get("join_key_exact")),
                "entity_bound": True,
                "instance_bound_lineage": bool(instance_bound_lineage),
                "materialization_role": str(
                    item.get("materialization_role") or ""
                ),
                "operator_complete": bool(item.get("operator_complete")),
                "lineage_root_key": str(item.get("lineage_root_key") or ""),
                "lineage_bound_keys": list(
                    item.get("lineage_bound_keys") or []
                ),
                "lineage_depth": int(item.get("lineage_depth", 0) or 0),
                "lineage": list(item.get("lineage") or []),
                "lineage_source_doc_ids": list(
                    item.get("lineage_source_doc_ids") or []
                ),
                "lineage_visible_source_doc_ids": list(
                    source_visibility["lineage_visible_source_doc_ids"]
                ),
                "lineage_missing_source_doc_ids": list(
                    source_visibility["lineage_missing_source_doc_ids"]
                ),
                "source_novel": bool(source_visibility["source_novel"]),
                "terminal_source_novel": bool(
                    source_visibility["terminal_source_novel"]
                ),
                "fact_view_novel": bool(fact_view_novel),
                "source_grounded_terminal_repair_used": bool(
                    item.get("source_grounded_terminal_repair_used")
                ),
                "bound_completion_lineage": list(
                    item.get("bound_completion_lineage") or []
                ),
                "slot_bindings": dict(item.get("slot_bindings") or {}),
                "strict_operator_contract_validated": bool(
                    item.get("strict_operator_contract_validated")
                ),
                "root_plan_variant": str(
                    item.get("root_plan_variant") or ""
                ),
                "root_identity_safe_for_commit": bool(
                    item.get("root_identity_safe_for_commit", True)
                ),
                "bound_completion_used": bool(
                    item.get("bound_completion_used")
                ),
                "completion_causal_lineage_validated": bool(
                    item.get("completion_causal_lineage_validated")
                ),
                "distributed_join_view": bool(
                    join_view_materialization_candidate
                ),
                "schema_terminal_fact_view": bool(
                    fact_view_materialization_candidate
                ),
            }
        )

    info["candidate_count"] = len(candidate_items)
    info["eligible_candidate_count"] = len(candidates)
    info["terminal_closed_candidate_count"] = sum(
        bool(row.get("terminal_slot_match")) for row in candidates
    )
    info["rejected_reason_counts"] = reject_counts
    if not candidates:
        info["reason"] = "no_strict_schema_candidate"
        return info

    required_slots: Set[str] = (
        {terminal_contract["slot_id"]}
        if require_terminal_closure else set()
    )
    if not required_slots:
        for row in candidates:
            required_slots.update(
                str(value)
                for value in row.get("uncovered_slot_ids") or []
                if str(value)
            )
    plan = optimize_cost_based_fact_join_plan(
        candidates,
        required_slot_ids=required_slots,
        max_candidates=max(0, int(max_append)),
        token_budget=max(1, int(token_budget)),
        compute_budget=max(0.0, float(compute_budget)),
        token_weight=float(token_weight),
        compute_weight=float(compute_weight),
        risk_weight=float(risk_weight),
        min_objective=float(min_objective),
        max_enumerated_candidates=max(1, len(candidates)),
    )
    info["plan"] = plan
    selected_ids = set(plan.get("selected_candidate_ids") or plan.get("candidate_ids") or [])
    selected = [row for row in candidates if row.get("candidate_id") in selected_ids]
    selected_objective = float(plan.get("selected_objective", 0.0) or 0.0)
    if not selected or selected_objective < float(min_objective):
        info["reason"] = "cost_plan_rejected"
        return info

    info["accepted"] = True
    info["accepted_count"] = len(selected)
    info["reason"] = (
        "accepted_schema_terminal_fact_view"
        if any(row.get("schema_terminal_fact_view") for row in selected)
        else "accepted_distributed_schema_join_view"
        if any(row.get("distributed_join_view") for row in selected)
        else "accepted_schema_critical_bridge"
    )
    info["selected_candidate_ids"] = [str(row["candidate_id"]) for row in selected]
    info["selected_doc_ids"] = [int(row["doc_id"]) for row in selected]
    info["reader_projection_token_count"] = sum(int(row["token_count"]) for row in selected)
    info["source_doc_token_count"] = sum(len(_tokens(row["source_content"])) for row in selected)
    info["projection_token_savings"] = max(0, info["source_doc_token_count"] - info["reader_projection_token_count"])
    info["selected"] = [
        {key: value for key, value in row.items() if key not in {"source_content"}}
        for row in selected
    ]
    return info
