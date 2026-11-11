"""Schema-aware fixed-budget evidence-set selection.

Most policies only reorder a frozen dense candidate pool.  The
``index_probe_path`` policy may additionally materialize provenance documents
reached through bounded entity-predicate index probes.  No policy receives QA
gold, and every policy keeps the Reader budget fixed at ``target_k``.
"""

from __future__ import annotations

import math
import re
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .relation_fact_join_utils import (
    _bind_explicit_query_entities,
    _equivalent_entity_keys,
    _lookup_items_by_exact_entity_key,
    _oriented_endpoint_for_operator_binding,
    _terminal_value_matches_contract,
    canonical_entity_identity_key,
    canonical_entity_key,
    canonical_title_surface_key,
    entity_predicate_index_key,
)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(value: Any) -> Set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(str(value or "").lower())
        if len(token) > 1
    }


def _title(doc: Any) -> str:
    return str(doc or "").split("\n", 1)[0].strip()


def assess_complete_path_projection_closure(
    evidence_selection: Optional[Dict[str, Any]],
    accepted_witness_rows: Optional[Sequence[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Check whether a complete path and its witness are already reader-visible.

    ``index_probe_path_surface_safe`` may append a compact, source-first witness
    only when it summarizes evidence that the Reader can already inspect.  In
    particular, a path that happened to be present in the original dense top-k
    is just as safe as a path selected from the global index.  The earlier
    policy keyed eligibility on ``selection.applied`` and incorrectly hid the
    former case.

    The returned record is deliberately serializable so a run trace can show
    why a witness projection was admitted or gated.
    """
    selection = dict(evidence_selection or {})
    selected_doc_ids = set()
    required_path_doc_ids = set()
    witness_doc_ids = set()
    missing_witness_provenance = 0

    for value in selection.get("selected_doc_ids") or []:
        try:
            selected_doc_ids.add(int(value))
        except (TypeError, ValueError):
            continue
    for value in selection.get("required_path_doc_ids") or []:
        try:
            required_path_doc_ids.add(int(value))
        except (TypeError, ValueError):
            continue

    rows = list(accepted_witness_rows or [])
    for row in rows:
        if not isinstance(row, dict):
            missing_witness_provenance += 1
            continue
        value = row.get("source_id", row.get("doc_idx"))
        try:
            witness_doc_ids.add(int(value))
        except (TypeError, ValueError):
            missing_witness_provenance += 1

    path_complete = bool(selection.get("binding_path_complete"))
    required_path_visible = bool(required_path_doc_ids) and required_path_doc_ids.issubset(
        selected_doc_ids
    )
    witness_visible = bool(rows) and not missing_witness_provenance and witness_doc_ids.issubset(
        selected_doc_ids
    )
    if not path_complete:
        reason = "no_complete_schema_path"
    elif not selected_doc_ids:
        reason = "missing_reader_doc_ids"
    elif not required_path_doc_ids:
        reason = "missing_required_path_doc_ids"
    elif not required_path_visible:
        reason = "schema_path_outside_reader_evidence"
    elif not rows:
        reason = "missing_complete_witness_rows"
    elif missing_witness_provenance:
        reason = "missing_witness_provenance"
    elif not witness_visible:
        reason = "witness_outside_reader_evidence"
    else:
        reason = "reader_closed_complete_path"

    return {
        "binding_path_complete": path_complete,
        "selected_doc_ids": sorted(selected_doc_ids),
        "required_path_doc_ids": sorted(required_path_doc_ids),
        "witness_source_doc_ids": sorted(witness_doc_ids),
        "missing_witness_provenance_rows": int(missing_witness_provenance),
        "required_path_reader_visible": required_path_visible,
        "witness_reader_visible": witness_visible,
        "eligible": reason == "reader_closed_complete_path",
        "reason": reason,
    }


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _normalize(values: Sequence[Any]) -> List[float]:
    scores = [_safe_float(value) for value in values]
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high <= low:
        return [1.0 for _ in scores]
    return [(value - low) / (high - low) for value in scores]


def _as_strings(values: Iterable[Any]) -> Set[str]:
    return {
        str(value or "").strip().lower()
        for value in values or []
        if str(value or "").strip()
    }


def _logical_operators(logical_plan: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    plan = dict(logical_plan or {})
    raw = plan.get("logical_operators") or plan.get("operators") or plan.get("steps") or []
    operators = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        families = _as_strings(item.get("predicate_families") or [])
        subtypes = _as_strings(item.get("required_predicate_subtypes") or [])
        relation_tokens = _as_strings(item.get("relation_tokens") or [])
        if not (families or subtypes or relation_tokens):
            continue
        operators.append({
            "index": len(operators),
            "operator_id": str(item.get("operator_id") or f"op{index}"),
            "families": families,
            "subtypes": subtypes,
            "relation_tokens": relation_tokens,
            "output_type": str(item.get("output_type") or "").strip().lower(),
        })
    return operators


def _path_operators(logical_plan: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the executable lookup chain, excluding a terminal reducer."""
    plan = dict(logical_plan or {})
    raw = plan.get("operators") or plan.get("steps") or plan.get("logical_operators") or []
    operators: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        operator_type = str(item.get("operator_type") or "indexed_fact_lookup")
        if operator_type != "indexed_fact_lookup":
            continue
        operator = dict(item)
        operator.setdefault("operator_id", f"op{index}")
        operator.setdefault("input_slot", "$root" if index == 0 else f"$v{index}")
        operator.setdefault("output_slot", "$answer" if index == len(raw) - 1 else f"$v{index + 1}")
        operator["predicate_families"] = sorted(
            _as_strings(operator.get("predicate_families") or [])
        )
        operator["required_predicate_subtypes"] = sorted(
            _as_strings(operator.get("required_predicate_subtypes") or [])
        )
        operator["relation_tokens"] = sorted(
            _as_strings(operator.get("relation_tokens") or [])
        )
        operator["relation_families"] = sorted(
            _as_strings(operator.get("relation_families") or [])
        )
        if not (
            operator["predicate_families"]
            or operator["required_predicate_subtypes"]
            or operator["relation_tokens"]
        ):
            continue
        operators.append(operator)
    return operators


def _operators_form_chain(operators: Sequence[Dict[str, Any]]) -> bool:
    return bool(operators) and all(
        str(left.get("output_slot") or "")
        == str(right.get("input_slot") or "")
        for left, right in zip(operators, operators[1:])
    )


def _bind_path_roots(
    *,
    question: str,
    relation_index: Dict[str, Any],
    logical_plan: Optional[Dict[str, Any]],
    operators: Sequence[Dict[str, Any]],
    enforce_exact_title_provenance: bool = False,
) -> Tuple[List[Dict[str, Any]], int]:
    plan_kind = str((logical_plan or {}).get("plan_kind") or "")
    expected_count = 2 if plan_kind in {"branch_compare", "branch_temporal_compare"} else 1
    token_matches = list(re.finditer(
        r"[A-Za-z0-9À-ÖØ-öø-ÿ]+", str(question or "")
    ))
    title_surfaces = relation_index.get("title_identity_surfaces") or {}
    title_doc_ids = relation_index.get("title_identity_doc_ids") or {}
    exact_title_surfaces = relation_index.get("title_surface_surfaces") or {}
    exact_title_doc_ids = relation_index.get("title_surface_doc_ids") or {}
    exact_title_candidates: List[Dict[str, Any]] = []
    for start in range(len(token_matches)):
        for width in range(1, min(10, len(token_matches) - start) + 1):
            end = start + width
            mention = str(question or "")[
                token_matches[start].start():token_matches[end - 1].end()
            ]
            identity_key = canonical_entity_identity_key(mention)
            surface_key = canonical_title_surface_key(mention)
            canonical_surface = str(
                exact_title_surfaces.get(surface_key)
                or title_surfaces.get(identity_key)
                or ""
            ).strip()
            if not canonical_surface:
                continue
            entity_key = canonical_entity_key(canonical_surface)
            if not entity_key:
                continue
            incident = _lookup_items_by_exact_entity_key(
                relation_index,
                entity_key,
                max_per_key=64,
                max_fanout=512,
                allow_title_provenance_alias=True,
                max_total_rows=128,
            )
            equivalent = _equivalent_entity_keys(
                relation_index,
                entity_key,
                allow_title_provenance_alias=True,
            )
            operator_match_count = sum(
                _oriented_endpoint_for_operator_binding(
                    item,
                    dict(operators[0]),
                    entity_key,
                    equivalent_keys=equivalent,
                ) is not None
                for item in incident
            )
            broad_doc_ids = list(dict.fromkeys(
                title_doc_ids.get(identity_key) or []
            ))
            surface_doc_ids = list(dict.fromkeys(
                exact_title_doc_ids.get(surface_key) or []
            ))
            restrict_first_hop = bool(
                enforce_exact_title_provenance
                and surface_doc_ids
                and set(surface_doc_ids) != set(broad_doc_ids)
            )
            exact_title_candidates.append({
                "surface": canonical_surface,
                "mention_surface": mention,
                "key": entity_key,
                "binding_method": "ordered_title_identity_query_span",
                "binding_confidence": 1.0,
                "operator_match_count": int(operator_match_count),
                "token_start": start,
                "token_end": end,
                "token_width": width,
                "canonical_title_doc_ids": broad_doc_ids,
                "exact_title_surface_key": surface_key,
                "exact_title_doc_ids": surface_doc_ids,
                "title_identity_collision": len(broad_doc_ids) > len(surface_doc_ids),
                "restrict_first_hop_doc_ids": (
                    surface_doc_ids if restrict_first_hop else []
                ),
            })
    exact_title_candidates.sort(key=lambda root: (
        -int(root["token_width"]),
        -int(root.get("operator_match_count", 0) or 0),
        int(root["token_start"]),
        str(root["key"]),
    ))
    roots: List[Dict[str, Any]] = []
    occupied: Set[int] = set()
    seen_keys: Set[str] = set()
    for candidate in exact_title_candidates:
        span = set(range(int(candidate["token_start"]), int(candidate["token_end"])))
        if candidate["key"] in seen_keys or span & occupied:
            continue
        roots.append(candidate)
        occupied.update(span)
        seen_keys.add(str(candidate["key"]))
        if len(roots) >= expected_count:
            break
    roots.sort(key=lambda root: int(root["token_start"]))
    if len(roots) == expected_count:
        return roots, expected_count



    if expected_count > 1:
        return [], expected_count
    roots = _bind_explicit_query_entities(
        question,
        relation_index,
        operator=dict(operators[0]),
        expected_count=expected_count,
        max_bindings=expected_count,
        require_operator_match=False,
        canonicalize_surface=expected_count == 1,
        enforce_root_contract=True,



        use_title_entity_keys=True,
        enable_collision_aware_title_identity=True,
    )
    return roots, expected_count


def _new_index_access_stats() -> Dict[str, int]:
    return {
        "lookup_keys": 0,
        "hit_keys": 0,
        "alias_hit_keys": 0,
        "input_scan_rows": 0,
        "rows_after_cap": 0,
        "truncated_rows": 0,
        "budget_truncated_rows": 0,
        "fanout_rejected_keys": 0,
        "fanout_rejected_rows": 0,
        "duplicate_rows": 0,
        "deduplicated_rows": 0,
        "cache_hits": 0,
        "budget_exhausted_probes": 0,
        "oriented_rows": 0,
        "contract_rejected_rows": 0,
        "root_title_provenance_rejected_rows": 0,
    }


def _probe_items_by_bound_operator(
    *,
    relation_index: Dict[str, Any],
    bound_key: str,
    operator: Dict[str, Any],
    access_stats: Dict[str, int],
    probe_cache: Dict[Tuple[str, str], List[Dict[str, Any]]],
    max_rows_per_key: int,
    max_total_rows: int,
    max_fanout: int,
) -> List[Dict[str, Any]]:
    """Probe physical entity-predicate indexes under one shared query budget."""
    operator_id = str(operator.get("operator_id") or "")
    cache_key = (str(bound_key or ""), operator_id)
    if cache_key in probe_cache:
        access_stats["cache_hits"] += 1
        return probe_cache[cache_key]

    equivalent_keys = _equivalent_entity_keys(
        relation_index,
        bound_key,
        allow_title_provenance_alias=True,
    )
    subtypes = sorted(_as_strings(
        operator.get("required_predicate_subtypes") or []
    ))
    families = sorted(_as_strings(operator.get("predicate_families") or []))
    index_contracts: List[Tuple[str, str]] = []
    index_contracts.extend(("subtype", subtype) for subtype in subtypes)
    index_contracts.extend(("family", family) for family in families)




    if not index_contracts:
        remaining = max(0, int(max_total_rows) - access_stats["input_scan_rows"])
        if remaining <= 0:
            access_stats["budget_exhausted_probes"] += 1
            probe_cache[cache_key] = []
            return []
        rows = _lookup_items_by_exact_entity_key(
            relation_index,
            bound_key,
            max_per_key=max_rows_per_key,
            max_fanout=max_fanout,
            access_stats=access_stats,
            allow_title_provenance_alias=True,
            max_total_rows=remaining,
        )
        probe_cache[cache_key] = rows
        return rows

    raw_ids: List[int] = []
    for probe_key in sorted(equivalent_keys):
        for contract_kind, contract_value in index_contracts:
            composite_key = entity_predicate_index_key(
                probe_key,
                contract_value,
            )
            if not composite_key:
                continue
            for side in ("subject", "object"):
                index_name = f"{side}_predicate_{contract_kind}_to_items"
                access_stats["lookup_keys"] += 1
                values = list(
                    (relation_index.get(index_name) or {}).get(composite_key) or []
                )
                if not values:
                    continue
                access_stats["hit_keys"] += 1
                if probe_key != bound_key:
                    access_stats["alias_hit_keys"] += 1
                if max_fanout > 0 and len(values) > max_fanout:
                    access_stats["fanout_rejected_keys"] += 1
                    access_stats["fanout_rejected_rows"] += len(values)
                    continue
                remaining = max(
                    0,
                    int(max_total_rows) - access_stats["input_scan_rows"],
                )
                if remaining <= 0:
                    access_stats["budget_exhausted_probes"] += 1
                    access_stats["budget_truncated_rows"] += len(values)
                    continue
                row_limit = min(max(1, int(max_rows_per_key)), remaining)
                capped = values[:row_limit]
                raw_ids.extend(capped)
                access_stats["input_scan_rows"] += len(capped)
                access_stats["rows_after_cap"] += len(capped)
                truncated = max(0, len(values) - len(capped))
                access_stats["truncated_rows"] += truncated
                if remaining < min(len(values), max(1, int(max_rows_per_key))):
                    access_stats["budget_truncated_rows"] += (
                        min(len(values), max(1, int(max_rows_per_key)))
                        - len(capped)
                    )

    items = relation_index.get("items") or []
    result: List[Dict[str, Any]] = []
    seen: Set[int] = set()
    for raw_id in raw_ids:
        try:
            item_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if item_id in seen or item_id < 0 or item_id >= len(items):
            continue
        seen.add(item_id)
        result.append(items[item_id])
    access_stats["duplicate_rows"] += max(0, len(raw_ids) - len(result))
    access_stats["deduplicated_rows"] += len(result)
    probe_cache[cache_key] = result
    return result


def _search_index_probed_paths(
    *,
    root: Dict[str, Any],
    operators: Sequence[Dict[str, Any]],
    relation_index: Dict[str, Any],
    rank_by_doc_id: Dict[int, int],
    target_k: int,
    access_stats: Dict[str, int],
    probe_cache: Dict[Tuple[str, str], List[Dict[str, Any]]],
    max_rows_per_key: int,
    max_total_rows: int,
    max_fanout: int,
    beam_width: int = 32,
) -> List[Dict[str, Any]]:
    root_key = str(root.get("key") or "")
    if not root_key:
        return []
    states: List[Dict[str, Any]] = [{
        "bound_key": root_key,
        "bound_surface": str(root.get("surface") or ""),
        "visited_keys": {root_key},
        "steps": [],
    }]
    restricted_root_doc_ids = {
        int(doc_id)
        for doc_id in root.get("restrict_first_hop_doc_ids") or []
    }
    for operator_index, operator in enumerate(operators):
        next_states: List[Dict[str, Any]] = []
        for state in states:
            bound_key = str(state.get("bound_key") or "")
            equivalent_keys = _equivalent_entity_keys(
                relation_index,
                bound_key,
                allow_title_provenance_alias=True,
            )
            candidates = _probe_items_by_bound_operator(
                relation_index=relation_index,
                bound_key=bound_key,
                operator=dict(operator),
                access_stats=access_stats,
                probe_cache=probe_cache,
                max_rows_per_key=max_rows_per_key,
                max_total_rows=max_total_rows,
                max_fanout=max_fanout,
            )
            for item in candidates:
                raw_item_id = item.get("item_id", -1)
                try:
                    item_id = int(raw_item_id)
                    doc_id = int(item.get("doc_idx", -1))
                except (TypeError, ValueError):
                    continue
                if doc_id < 0 or any(
                    step["item_id"] == item_id for step in state["steps"]
                ):
                    continue
                if (
                    operator_index == 0
                    and restricted_root_doc_ids
                    and doc_id not in restricted_root_doc_ids
                ):
                    access_stats["root_title_provenance_rejected_rows"] += 1
                    continue
                oriented = _oriented_endpoint_for_operator_binding(
                    item,
                    dict(operator),
                    bound_key,
                    equivalent_keys=equivalent_keys,
                )
                if oriented is None:
                    access_stats["contract_rejected_rows"] += 1
                    continue
                output_surface = str(oriented.get("surface") or "")
                output_key = str(oriented.get("key") or "")
                if not output_key or output_key in state["visited_keys"]:
                    access_stats["contract_rejected_rows"] += 1
                    continue
                if not _terminal_value_matches_contract(
                    output_surface,
                    str(operator.get("output_type") or "entity"),
                    set(item.get("relation_tokens") or set()),
                ):
                    access_stats["contract_rejected_rows"] += 1
                    continue
                access_stats["oriented_rows"] += 1
                dense_rank = rank_by_doc_id.get(doc_id)
                next_states.append({
                    "bound_key": output_key,
                    "bound_surface": output_surface,
                    "visited_keys": set(state["visited_keys"]) | {output_key},
                    "steps": list(state["steps"]) + [{
                        "operator_id": str(operator.get("operator_id") or ""),
                        "item_id": item_id,
                        "doc_id": doc_id,
                        "rank": int(dense_rank) if dense_rank is not None else -1,
                        "dense_rank": (
                            int(dense_rank) if dense_rank is not None else None
                        ),
                        "in_dense_pool": dense_rank is not None,
                        "triple": list(item.get("triple") or []),
                        "input_key": bound_key,
                        "output_key": output_key,
                        "input_side": str(oriented.get("input_side") or ""),
                        "output_side": str(oriented.get("output_side") or ""),
                        "role_direction_required": bool(
                            oriented.get("role_direction_required")
                        ),
                    }],
                })
        deduplicated: Dict[Tuple[str, Tuple[int, ...]], Dict[str, Any]] = {}
        for state in next_states:
            path_item_ids = tuple(step["item_id"] for step in state["steps"])
            deduplicated.setdefault((str(state["bound_key"]), path_item_ids), state)
        dense_pool_size = len(rank_by_doc_id)
        states = sorted(
            deduplicated.values(),
            key=lambda state: (
                sum(
                    step["dense_rank"] is None
                    or int(step["dense_rank"]) >= target_k
                    for step in state["steps"]
                ),
                sum(step["dense_rank"] is None for step in state["steps"]),
                len({step["doc_id"] for step in state["steps"]}),
                sum(
                    int(step["dense_rank"])
                    if step["dense_rank"] is not None
                    else dense_pool_size + 1 + int(step["item_id"])
                    for step in state["steps"]
                ),
                str(state["bound_key"]),
            ),
        )[: max(1, int(beam_width))]
        if not states:
            break
    return states if states and len(states[0]["steps"]) == len(operators) else []


def _search_complete_paths(
    *,
    root: Dict[str, Any],
    operators: Sequence[Dict[str, Any]],
    relation_index: Dict[str, Any],
    allowed_doc_ids: Set[int],
    rank_by_doc_id: Dict[int, int],
    target_k: int,
    beam_width: int = 32,
) -> List[Dict[str, Any]]:
    root_key = str(root.get("key") or "")
    if not root_key:
        return []
    states: List[Dict[str, Any]] = [{
        "bound_key": root_key,
        "bound_surface": str(root.get("surface") or ""),
        "visited_keys": {root_key},
        "steps": [],
    }]
    for operator_index, operator in enumerate(operators):
        next_states: List[Dict[str, Any]] = []
        for state in states:
            bound_key = str(state.get("bound_key") or "")



            allow_title_alias = True
            equivalent_keys = _equivalent_entity_keys(
                relation_index,
                bound_key,
                allow_title_provenance_alias=allow_title_alias,
            )
            candidates = _lookup_items_by_exact_entity_key(
                relation_index,
                bound_key,
                max_per_key=96,
                max_fanout=512,
                allow_title_provenance_alias=allow_title_alias,
                max_total_rows=192,
            )
            for item in candidates:
                try:
                    doc_id = int(item.get("doc_idx", -1))
                except (TypeError, ValueError):
                    continue
                if doc_id not in allowed_doc_ids:
                    continue
                raw_item_id = item.get("item_id", -1)
                try:
                    item_id = int(raw_item_id)
                except (TypeError, ValueError):
                    continue
                if any(step["item_id"] == item_id for step in state["steps"]):
                    continue
                oriented = _oriented_endpoint_for_operator_binding(
                    item,
                    dict(operator),
                    bound_key,
                    equivalent_keys=equivalent_keys,
                )
                if oriented is None:
                    continue
                output_surface = str(oriented.get("surface") or "")
                output_key = str(oriented.get("key") or "")
                if not output_key or output_key in state["visited_keys"]:
                    continue
                if not _terminal_value_matches_contract(
                    output_surface,
                    str(operator.get("output_type") or "entity"),
                    set(item.get("relation_tokens") or set()),
                ):
                    continue
                next_states.append({
                    "bound_key": output_key,
                    "bound_surface": output_surface,
                    "visited_keys": set(state["visited_keys"]) | {output_key},
                    "steps": list(state["steps"]) + [{
                        "operator_id": str(operator.get("operator_id") or ""),
                        "item_id": item_id,
                        "doc_id": doc_id,
                        "rank": int(rank_by_doc_id[doc_id]),
                        "triple": list(item.get("triple") or []),
                        "input_key": bound_key,
                        "output_key": output_key,
                        "input_side": str(oriented.get("input_side") or ""),
                        "output_side": str(oriented.get("output_side") or ""),
                        "role_direction_required": bool(
                            oriented.get("role_direction_required")
                        ),
                    }],
                })
        deduplicated: Dict[Tuple[str, Tuple[int, ...]], Dict[str, Any]] = {}
        for state in next_states:
            path_item_ids = tuple(step["item_id"] for step in state["steps"])
            key = (str(state["bound_key"]), path_item_ids)
            deduplicated.setdefault(key, state)
        states = sorted(
            deduplicated.values(),
            key=lambda state: (
                sum(step["rank"] >= target_k for step in state["steps"]),
                len({step["doc_id"] for step in state["steps"]}),
                sum(step["rank"] for step in state["steps"]),
                str(state["bound_key"]),
            ),
        )[: max(1, int(beam_width))]
        if not states:
            break
    return states if states and len(states[0]["steps"]) == len(operators) else []


def _select_binding_path(
    *,
    question: str,
    docs: Sequence[str],
    doc_ids: Sequence[int],
    profiles: Sequence[Dict[str, Any]],
    relation_index: Dict[str, Any],
    logical_plan: Optional[Dict[str, Any]],
    operators: Sequence[Dict[str, Any]],
    target_k: int,
    protect_top_k: int,
    info: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    base_docs = list(docs[:target_k])
    if not _operators_form_chain(operators):
        info["reason"] = "non_chain_logical_plan"
        return base_docs, info
    roots, expected_root_count = _bind_path_roots(
        question=question,
        relation_index=relation_index,
        logical_plan=logical_plan,
        operators=operators,
    )
    info["expected_root_count"] = expected_root_count
    info["root_binding_count"] = len(roots)
    info["root_bindings"] = [{
        "surface": str(root.get("surface") or ""),
        "key": str(root.get("key") or ""),
        "method": str(root.get("binding_method") or ""),
        "operator_match_count": int(root.get("operator_match_count", 0) or 0),
    } for root in roots]
    if len(roots) != expected_root_count:
        info["reason"] = "root_binding_incomplete"
        return base_docs, info

    rank_by_doc_id = {
        int(doc_id): rank for rank, doc_id in enumerate(doc_ids[: len(profiles)])
    }
    allowed_doc_ids = set(rank_by_doc_id)
    paths_by_root = [
        _search_complete_paths(
            root=root,
            operators=operators,
            relation_index=relation_index,
            allowed_doc_ids=allowed_doc_ids,
            rank_by_doc_id=rank_by_doc_id,
            target_k=target_k,
        )
        for root in roots
    ]
    info["complete_path_counts"] = [len(paths) for paths in paths_by_root]
    if any(not paths for paths in paths_by_root):
        info["reason"] = "no_complete_binding_path"
        return base_docs, info

    combinations = product(*(paths[:8] for paths in paths_by_root))
    chosen_paths = min(
        combinations,
        key=lambda group: (
            len({
                step["doc_id"]
                for path in group for step in path["steps"]
                if step["rank"] >= target_k
            }),
            len({step["doc_id"] for path in group for step in path["steps"]}),
            sum(step["rank"] for path in group for step in path["steps"]),
            tuple(path["bound_key"] for path in group),
        ),
    )
    required_ranks = {
        int(step["rank"])
        for path in chosen_paths for step in path["steps"]
    }
    mandatory_ranks = required_ranks | set(range(min(protect_top_k, len(profiles))))
    if len(mandatory_ranks) > target_k:
        info["reason"] = "complete_path_exceeds_reader_budget"
        return base_docs, info
    selected_ranks = set(mandatory_ranks)
    for rank in range(len(profiles)):
        if len(selected_ranks) >= target_k:
            break
        selected_ranks.add(rank)
    if len(selected_ranks) < min(target_k, len(profiles)):
        info["reason"] = "insufficient_candidate_documents"
        return base_docs, info

    ordered_ranks = sorted(selected_ranks)[:target_k]
    selected = [profiles[rank] for rank in ordered_ranks]
    selected_coverage = _set_coverage(selected, len(operators))
    selected_joinability = _adjacent_joinability(selected, len(operators))
    base_indices = list(range(min(target_k, len(docs))))
    changed_count = len(set(ordered_ranks) ^ set(base_indices)) // 2
    info.update({
        "version": "binding_aware_fact_path_evidence_v2",
        "applied": ordered_ranks != base_indices,
        "reason": (
            "complete_binding_path_selected"
            if ordered_ranks != base_indices else "complete_binding_path_already_dense"
        ),
        "binding_path_complete": True,
        "branch_complete": expected_root_count == 1 or len(chosen_paths) == expected_root_count,
        "required_path_ranks": sorted(required_ranks),
        "selected_indices": ordered_ranks,
        "selected_doc_ids": [int(profiles[rank]["doc_id"]) for rank in ordered_ranks],
        "selected_titles": [profiles[rank]["title"] for rank in ordered_ranks],
        "selected_obligation_coverage": round(selected_coverage, 6),
        "selected_joinability": round(selected_joinability, 6),
        "changed_count": int(changed_count),
        "candidate_profile_count": len(profiles),
        "selected_paths": [{
            "root_surface": str(root.get("surface") or ""),
            "root_key": str(root.get("key") or ""),
            "terminal_surface": str(path.get("bound_surface") or ""),
            "terminal_key": str(path.get("bound_key") or ""),
            "steps": list(path.get("steps") or []),
        } for root, path in zip(roots, chosen_paths)],
    })
    return [profiles[rank]["doc"] for rank in ordered_ranks], info


def _select_index_probe_path(
    *,
    question: str,
    docs: Sequence[str],
    doc_ids: Sequence[int],
    profiles: Sequence[Dict[str, Any]],
    relation_index: Dict[str, Any],
    logical_plan: Optional[Dict[str, Any]],
    operators: Sequence[Dict[str, Any]],
    target_k: int,
    protect_top_k: int,
    info: Dict[str, Any],
    get_doc_content: Optional[Callable[[int], str]],
    max_rows_per_key: int,
    max_total_rows: int,
    max_fanout: int,
    enforce_exact_title_provenance: bool = False,
) -> Tuple[List[str], Dict[str, Any]]:
    """Materialize a complete, contract-valid path from the global fact index."""
    base_docs = list(docs[:target_k])
    base_doc_ids = [int(value) for value in doc_ids[: len(base_docs)]]
    info["version"] = (
        "index_probed_binding_path_evidence_safe_v1"
        if enforce_exact_title_provenance
        else "index_probed_binding_path_evidence_v1"
    )
    info["global_index_probe"] = True
    info["exact_title_provenance"] = bool(enforce_exact_title_provenance)
    info["index_probe_limits"] = {
        "max_rows_per_key": int(max_rows_per_key),
        "max_total_rows": int(max_total_rows),
        "max_fanout": int(max_fanout),
    }
    if not _operators_form_chain(operators):
        info["reason"] = "non_chain_logical_plan"
        return base_docs, info
    roots, expected_root_count = _bind_path_roots(
        question=question,
        relation_index=relation_index,
        logical_plan=logical_plan,
        operators=operators,
        enforce_exact_title_provenance=enforce_exact_title_provenance,
    )
    info["expected_root_count"] = expected_root_count
    info["root_binding_count"] = len(roots)
    info["root_bindings"] = [{
        "surface": str(root.get("surface") or ""),
        "key": str(root.get("key") or ""),
        "method": str(root.get("binding_method") or ""),
        "operator_match_count": int(root.get("operator_match_count", 0) or 0),
        "mention_surface": str(root.get("mention_surface") or ""),
        "title_identity_collision": bool(root.get("title_identity_collision")),
        "exact_title_doc_ids": list(root.get("exact_title_doc_ids") or []),
        "restrict_first_hop_doc_ids": list(
            root.get("restrict_first_hop_doc_ids") or []
        ),
    } for root in roots]
    if len(roots) != expected_root_count:
        info["reason"] = "root_binding_incomplete"
        return base_docs, info

    rank_by_doc_id = {
        int(doc_id): rank for rank, doc_id in enumerate(doc_ids[: len(profiles)])
    }
    access_stats = _new_index_access_stats()
    probe_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    paths_by_root = [
        _search_index_probed_paths(
            root=root,
            operators=operators,
            relation_index=relation_index,
            rank_by_doc_id=rank_by_doc_id,
            target_k=target_k,
            access_stats=access_stats,
            probe_cache=probe_cache,
            max_rows_per_key=max_rows_per_key,
            max_total_rows=max_total_rows,
            max_fanout=max_fanout,
        )
        for root in roots
    ]
    info["index_access"] = dict(access_stats)
    info["complete_path_counts"] = [len(paths) for paths in paths_by_root]
    if any(not paths for paths in paths_by_root):
        info["reason"] = "no_complete_global_index_path"
        return base_docs, info

    combinations = product(*(paths[:8] for paths in paths_by_root))
    dense_pool_size = len(rank_by_doc_id)
    chosen_paths = min(
        combinations,
        key=lambda group: (
            len({
                step["doc_id"]
                for path in group for step in path["steps"]
                if step["dense_rank"] is None
                or int(step["dense_rank"]) >= target_k
            }),
            len({
                step["doc_id"]
                for path in group for step in path["steps"]
                if step["dense_rank"] is None
            }),
            len({step["doc_id"] for path in group for step in path["steps"]}),
            sum(
                int(step["dense_rank"])
                if step["dense_rank"] is not None
                else dense_pool_size + 1 + int(step["item_id"])
                for path in group for step in path["steps"]
            ),
            tuple(path["bound_key"] for path in group),
        ),
    )

    required_doc_ids: List[int] = []
    for path in chosen_paths:
        for step in path["steps"]:
            doc_id = int(step["doc_id"])
            if doc_id not in required_doc_ids:
                required_doc_ids.append(doc_id)
    protected_doc_ids = [
        int(value) for value in doc_ids[: min(protect_top_k, len(doc_ids))]
    ]
    mandatory_doc_ids = list(dict.fromkeys(protected_doc_ids + required_doc_ids))
    if len(mandatory_doc_ids) > target_k:
        info.update({
            "reason": "complete_path_exceeds_reader_budget",
            "binding_path_complete": True,
            "required_path_doc_ids": required_doc_ids,
            "out_of_pool_path_doc_ids": [
                doc_id for doc_id in required_doc_ids if doc_id not in rank_by_doc_id
            ],
        })
        return base_docs, info

    selected_doc_ids = list(mandatory_doc_ids)
    for doc_id in doc_ids:
        doc_id = int(doc_id)
        if len(selected_doc_ids) >= target_k:
            break
        if doc_id not in selected_doc_ids:
            selected_doc_ids.append(doc_id)
    if len(selected_doc_ids) < min(target_k, len(docs)):
        info["reason"] = "insufficient_candidate_documents"
        return base_docs, info

    dense_docs_by_id = {
        int(profile["doc_id"]): str(profile["doc"])
        for profile in profiles
    }
    selected_docs: List[str] = []
    selected_origins: List[str] = []
    for doc_id in selected_doc_ids:
        if doc_id in dense_docs_by_id:
            selected_docs.append(dense_docs_by_id[doc_id])
            selected_origins.append("dense_pool")
            continue
        if get_doc_content is None:
            info["reason"] = "missing_global_document_materializer"
            return base_docs, info
        try:
            content = str(get_doc_content(doc_id) or "")
        except Exception as exc:
            info["reason"] = "global_document_materialization_failed"
            info["materialization_error"] = type(exc).__name__
            return base_docs, info
        if not content.strip():
            info["reason"] = "empty_global_document"
            return base_docs, info
        selected_docs.append(content)
        selected_origins.append("global_index")

    question_tokens = _tokens(question)
    physical_obligations = [{
        "families": set(item.get("predicate_families") or []),
        "subtypes": set(item.get("required_predicate_subtypes") or []),
        "relation_tokens": set(item.get("relation_tokens") or []),
    } for item in operators]
    selected_profiles = [
        _profile_candidate(
            rank=(
                int(rank_by_doc_id[doc_id])
                if doc_id in rank_by_doc_id else dense_pool_size + offset
            ),
            doc_id=doc_id,
            doc=doc,
            dense_score=(
                float(profiles[rank_by_doc_id[doc_id]]["dense_score"])
                if doc_id in rank_by_doc_id else 0.0
            ),
            relation_index=relation_index,
            obligations=physical_obligations,
            question_tokens=question_tokens,
        )
        for offset, (doc_id, doc) in enumerate(zip(selected_doc_ids, selected_docs))
    ]
    selected_coverage = _set_coverage(selected_profiles, len(operators))
    selected_joinability = _adjacent_joinability(selected_profiles, len(operators))
    out_of_pool_doc_ids = [
        doc_id for doc_id in required_doc_ids if doc_id not in rank_by_doc_id
    ]
    changed_count = len(set(selected_doc_ids) - set(base_doc_ids))
    info.update({
        "applied": selected_doc_ids != base_doc_ids,
        "reason": (
            "complete_global_index_path_selected"
            if selected_doc_ids != base_doc_ids
            else "complete_global_index_path_already_dense"
        ),
        "binding_path_complete": True,
        "branch_complete": expected_root_count == 1 or len(chosen_paths) == expected_root_count,
        "required_path_doc_ids": required_doc_ids,
        "required_path_ranks": sorted({
            int(rank_by_doc_id[doc_id])
            for doc_id in required_doc_ids if doc_id in rank_by_doc_id
        }),
        "out_of_pool_path_doc_ids": out_of_pool_doc_ids,
        "out_of_pool_path_doc_count": len(out_of_pool_doc_ids),
        "selected_indices": [
            int(rank_by_doc_id[doc_id]) if doc_id in rank_by_doc_id else -1
            for doc_id in selected_doc_ids
        ],
        "selected_doc_ids": selected_doc_ids,
        "selected_titles": [_title(doc) for doc in selected_docs],
        "selected_origins": selected_origins,
        "selected_obligation_coverage": round(selected_coverage, 6),
        "selected_joinability": round(selected_joinability, 6),
        "changed_count": int(changed_count),
        "candidate_profile_count": len(profiles),
        "selected_paths": [{
            "root_surface": str(root.get("surface") or ""),
            "root_key": str(root.get("key") or ""),
            "terminal_surface": str(path.get("bound_surface") or ""),
            "terminal_key": str(path.get("bound_key") or ""),
            "steps": list(path.get("steps") or []),
        } for root, path in zip(roots, chosen_paths)],
    })
    return selected_docs, info


def _item_obligation_score(item: Dict[str, Any], obligation: Dict[str, Any]) -> float:
    item_families = _as_strings(item.get("predicate_families") or [])
    item_subtypes = _as_strings(item.get("predicate_subtypes") or [])
    item_relation_tokens = _as_strings(item.get("relation_tokens") or [])
    required_subtypes = obligation["subtypes"]
    if required_subtypes:
        return 1.0 if item_subtypes & required_subtypes else 0.0
    if item_families & obligation["families"]:
        return 1.0
    required_tokens = obligation["relation_tokens"]
    if not required_tokens or not item_relation_tokens:
        return 0.0
    overlap = len(required_tokens & item_relation_tokens) / float(len(required_tokens))
    return overlap if overlap >= 0.5 else 0.0


def _profile_candidate(
    *,
    rank: int,
    doc_id: int,
    doc: str,
    dense_score: float,
    relation_index: Dict[str, Any],
    obligations: List[Dict[str, Any]],
    question_tokens: Set[str],
) -> Dict[str, Any]:
    items = relation_index.get("items") or []
    item_ids = (relation_index.get("doc_to_items") or {}).get(int(doc_id), []) or []
    obligation_scores = [0.0 for _ in obligations]
    obligation_entities: List[Set[str]] = [set() for _ in obligations]
    entity_keys: Set[str] = set()
    predicate_families: Set[str] = set()
    for item_id in item_ids:
        try:
            item = items[int(item_id)]
        except (IndexError, TypeError, ValueError):
            continue
        item_entities = {
            str(value or "").strip()
            for value in item.get("entity_keys") or []
            if str(value or "").strip()
        }
        entity_keys.update(item_entities)
        predicate_families.update(_as_strings(item.get("predicate_families") or []))
        for obligation_index, obligation in enumerate(obligations):
            score = _item_obligation_score(item, obligation)
            if score <= 0.0:
                continue
            obligation_scores[obligation_index] = max(
                obligation_scores[obligation_index], score
            )
            obligation_entities[obligation_index].update(item_entities)

    anchor_score = 0.0
    for entity_key in entity_keys:
        key_tokens = _tokens(entity_key)
        if not key_tokens:
            continue
        anchor_score = max(
            anchor_score,
            len(key_tokens & question_tokens) / float(len(key_tokens)),
        )
    return {
        "rank": int(rank),
        "doc_id": int(doc_id),
        "doc": doc,
        "title": _title(doc),
        "dense_score": float(dense_score),
        "obligation_scores": obligation_scores,
        "covered_obligations": {
            index for index, score in enumerate(obligation_scores) if score >= 0.5
        },
        "obligation_entities": obligation_entities,
        "entity_keys": entity_keys,
        "predicate_families": predicate_families,
        "anchor_score": float(anchor_score),
    }


def _set_coverage(selected: Sequence[Dict[str, Any]], obligation_count: int) -> float:
    if obligation_count <= 0:
        return 0.0
    best = [0.0 for _ in range(obligation_count)]
    for profile in selected:
        for index, score in enumerate(profile["obligation_scores"]):
            best[index] = max(best[index], float(score))
    return sum(best) / float(obligation_count)


def _adjacent_joinability(
    selected: Sequence[Dict[str, Any]], obligation_count: int
) -> float:
    if obligation_count <= 1:
        return 0.0
    connected = 0
    for left_index in range(obligation_count - 1):
        right_index = left_index + 1
        found = False
        for left in selected:
            left_keys = left["obligation_entities"][left_index]
            if not left_keys:
                continue
            for right in selected:
                if left is right:
                    continue
                if left_keys & right["obligation_entities"][right_index]:
                    found = True
                    break
            if found:
                break
        connected += int(found)
    return connected / float(obligation_count - 1)


def _marginal_joinability(
    candidate: Dict[str, Any],
    selected: Sequence[Dict[str, Any]],
    obligation_count: int,
) -> float:
    if not selected or obligation_count <= 1:
        return 0.0
    for index in range(obligation_count - 1):
        candidate_left = candidate["obligation_entities"][index]
        candidate_right = candidate["obligation_entities"][index + 1]
        for existing in selected:
            if candidate_left & existing["obligation_entities"][index + 1]:
                return 1.0
            if candidate_right & existing["obligation_entities"][index]:
                return 1.0
    return 0.0


def _redundancy(candidate: Dict[str, Any], selected: Sequence[Dict[str, Any]]) -> float:
    best = 0.0
    candidate_entities = candidate["entity_keys"]
    for existing in selected:
        existing_entities = existing["entity_keys"]
        union = candidate_entities | existing_entities
        if union:
            best = max(best, len(candidate_entities & existing_entities) / float(len(union)))
        if candidate["title"].casefold() == existing["title"].casefold():
            best = 1.0
    return best


def select_schema_aware_evidence_set(
    *,
    question: str,
    docs: Sequence[str],
    doc_ids: Optional[Sequence[int]],
    doc_scores: Optional[Sequence[float]],
    relation_index: Optional[Dict[str, Any]],
    logical_plan: Optional[Dict[str, Any]],
    enabled: bool = False,
    mode: str = "joint",
    candidate_k: int = 20,
    target_k: int = 5,
    protect_top_k: int = 1,
    get_doc_content: Optional[Callable[[int], str]] = None,
    index_max_rows_per_key: int = 64,
    index_max_total_rows: int = 512,
    index_max_fanout: int = 512,
) -> Tuple[List[str], Dict[str, Any]]:
    """Select exactly ``target_k`` passages without using answer supervision."""
    docs = list(docs or [])
    target_k = max(1, int(target_k))
    candidate_k = max(target_k, int(candidate_k))
    protect_top_k = max(0, min(target_k, int(protect_top_k)))
    mode = str(mode or "joint").strip().lower()
    if mode not in {
        "dense",
        "schema_score",
        "greedy_coverage",
        "joint",
        "binding_path",
        "index_probe_path",
        "index_probe_path_safe",
        "index_probe_path_surface_safe",
    }:
        mode = "joint"
    base_docs = docs[:target_k]
    info: Dict[str, Any] = {
        "version": "schema_aware_evidence_set_v1",
        "enabled": bool(enabled),
        "mode": mode,
        "candidate_k": candidate_k,
        "target_k": target_k,
        "protect_top_k": protect_top_k,
        "applied": False,
        "reason": "disabled",
        "selected_indices": list(range(len(base_docs))),
        "selected_doc_ids": [],
        "selected_titles": [_title(doc) for doc in base_docs],
        "obligation_count": 0,
        "base_obligation_coverage": 0.0,
        "selected_obligation_coverage": 0.0,
        "base_joinability": 0.0,
        "selected_joinability": 0.0,
        "changed_count": 0,
        "online_gold_used": False,
    }
    if not enabled or mode == "dense":
        info["reason"] = "dense_control" if enabled else "disabled"
        return base_docs, info
    if not relation_index or not (relation_index.get("items") or []):
        info["reason"] = "missing_relation_index"
        return base_docs, info
    if not doc_ids or len(doc_ids) < min(target_k, len(docs)):
        info["reason"] = "missing_aligned_doc_ids"
        return base_docs, info

    obligations = _logical_operators(logical_plan)
    info["obligation_count"] = len(obligations)
    info["obligations"] = [
        {
            "operator_id": item["operator_id"],
            "families": sorted(item["families"]),
            "subtypes": sorted(item["subtypes"]),
            "relation_tokens": sorted(item["relation_tokens"]),
        }
        for item in obligations
    ]
    if not obligations:
        info["reason"] = "no_relation_obligations"
        return base_docs, info

    pool_size = min(candidate_k, len(docs), len(doc_ids))
    raw_scores = list(doc_scores[:pool_size]) if doc_scores is not None else [0.0] * pool_size
    normalized_scores = _normalize(raw_scores)
    question_tokens = _tokens(question)
    profiles = [
        _profile_candidate(
            rank=rank,
            doc_id=int(doc_ids[rank]),
            doc=docs[rank],
            dense_score=normalized_scores[rank] if rank < len(normalized_scores) else 0.0,
            relation_index=relation_index,
            obligations=obligations,
            question_tokens=question_tokens,
        )
        for rank in range(pool_size)
    ]
    base_profiles = profiles[:target_k]
    base_coverage = _set_coverage(base_profiles, len(obligations))
    base_joinability = _adjacent_joinability(base_profiles, len(obligations))
    info["base_obligation_coverage"] = round(base_coverage, 6)
    info["base_joinability"] = round(base_joinability, 6)

    if mode in {
        "binding_path",
        "index_probe_path",
        "index_probe_path_safe",
        "index_probe_path_surface_safe",
    }:
        path_operators = _path_operators(logical_plan)
        info["obligation_count"] = len(path_operators)
        info["obligations"] = [{
            "operator_id": str(item.get("operator_id") or ""),
            "input_slot": str(item.get("input_slot") or ""),
            "output_slot": str(item.get("output_slot") or ""),
            "families": sorted(item.get("predicate_families") or []),
            "subtypes": sorted(item.get("required_predicate_subtypes") or []),
            "relation_tokens": sorted(item.get("relation_tokens") or []),
            "canonical_relation": str(item.get("canonical_relation") or ""),
            "input_role": str(item.get("input_role") or ""),
            "output_role": str(item.get("output_role") or ""),
            "output_type": str(item.get("output_type") or ""),
        } for item in path_operators]
        if not path_operators:
            info["reason"] = "no_executable_lookup_operators"
            return base_docs, info


        physical_profiles = [
            _profile_candidate(
                rank=profile["rank"],
                doc_id=profile["doc_id"],
                doc=profile["doc"],
                dense_score=profile["dense_score"],
                relation_index=relation_index,
                obligations=[{
                    "families": set(item.get("predicate_families") or []),
                    "subtypes": set(item.get("required_predicate_subtypes") or []),
                    "relation_tokens": set(item.get("relation_tokens") or []),
                } for item in path_operators],
                question_tokens=question_tokens,
            )
            for profile in profiles
        ]
        info["base_obligation_coverage"] = round(
            _set_coverage(physical_profiles[:target_k], len(path_operators)), 6
        )
        info["base_joinability"] = round(
            _adjacent_joinability(physical_profiles[:target_k], len(path_operators)), 6
        )
        if mode == "binding_path":
            return _select_binding_path(
                question=question,
                docs=docs,
                doc_ids=doc_ids[:pool_size],
                profiles=physical_profiles,
                relation_index=relation_index,
                logical_plan=logical_plan,
                operators=path_operators,
                target_k=target_k,
                protect_top_k=protect_top_k,
                info=info,
            )
        return _select_index_probe_path(
            question=question,
            docs=docs,
            doc_ids=doc_ids[:pool_size],
            profiles=physical_profiles,
            relation_index=relation_index,
            logical_plan=logical_plan,
            operators=path_operators,
            target_k=target_k,
            protect_top_k=protect_top_k,
            info=info,
            get_doc_content=get_doc_content,
            max_rows_per_key=max(1, int(index_max_rows_per_key)),
            max_total_rows=max(1, int(index_max_total_rows)),
            max_fanout=max(1, int(index_max_fanout)),
            enforce_exact_title_provenance=(
                mode in {"index_probe_path_safe", "index_probe_path_surface_safe"}
            ),
        )

    selected = list(profiles[:protect_top_k])
    selected_ranks = {item["rank"] for item in selected}
    while len(selected) < min(target_k, len(profiles)):
        current_best = [0.0 for _ in obligations]
        for existing in selected:
            for index, score in enumerate(existing["obligation_scores"]):
                current_best[index] = max(current_best[index], float(score))
        candidates = []
        for candidate in profiles:
            if candidate["rank"] in selected_ranks:
                continue
            marginal = sum(
                max(0.0, float(score) - current_best[index])
                for index, score in enumerate(candidate["obligation_scores"])
            ) / float(len(obligations))
            absolute = sum(candidate["obligation_scores"]) / float(len(obligations))
            multi_cover = max(0, len(candidate["covered_obligations"]) - 1) / float(
                max(1, len(obligations) - 1)
            )
            join_gain = _marginal_joinability(candidate, selected, len(obligations))
            redundancy = _redundancy(candidate, selected)
            if mode == "schema_score":
                utility = (
                    0.55 * candidate["dense_score"]
                    + 0.35 * absolute
                    + 0.10 * candidate["anchor_score"]
                )
            elif mode == "greedy_coverage":
                utility = (
                    0.45 * candidate["dense_score"]
                    + 0.45 * marginal
                    + 0.10 * multi_cover
                    - 0.05 * redundancy
                )
            else:
                utility = (
                    0.42 * candidate["dense_score"]
                    + 0.28 * marginal
                    + 0.16 * join_gain
                    + 0.07 * candidate["anchor_score"]
                    + 0.07 * multi_cover
                    - 0.08 * redundancy
                )
            candidates.append((utility, -candidate["rank"], candidate))
        if not candidates:
            break
        utility, _, winner = max(candidates, key=lambda row: (row[0], row[1]))
        winner = dict(winner)
        winner["selection_utility"] = float(utility)
        selected.append(winner)
        selected_ranks.add(winner["rank"])

    selected = selected[:target_k]
    selected_coverage = _set_coverage(selected, len(obligations))
    selected_joinability = _adjacent_joinability(selected, len(obligations))
    if mode in {"greedy_coverage", "joint"} and (
        selected_coverage <= base_coverage and selected_joinability <= base_joinability
    ):
        info["reason"] = "no_schema_set_gain"
        return base_docs, info

    selected_indices = [item["rank"] for item in selected]
    selected_docs = [item["doc"] for item in selected]
    base_indices = list(range(min(target_k, len(docs))))
    changed_count = len(set(selected_indices) ^ set(base_indices)) // 2
    info.update({
        "applied": bool(selected_indices != base_indices),
        "reason": "selected" if selected_indices != base_indices else "unchanged",
        "selected_indices": selected_indices,
        "selected_doc_ids": [item["doc_id"] for item in selected],
        "selected_titles": [item["title"] for item in selected],
        "selected_obligation_coverage": round(selected_coverage, 6),
        "selected_joinability": round(selected_joinability, 6),
        "changed_count": int(changed_count),
        "candidate_profile_count": len(profiles),
        "selected_profiles": [
            {
                "rank": item["rank"],
                "doc_id": item["doc_id"],
                "title": item["title"],
                "dense_score": round(item["dense_score"], 6),
                "covered_obligations": sorted(item["covered_obligations"]),
                "predicate_families": sorted(item["predicate_families"]),
                "anchor_score": round(item["anchor_score"], 6),
                "selection_utility": round(float(item.get("selection_utility", 0.0)), 6),
            }
            for item in selected
        ],
    })
    return selected_docs, info
