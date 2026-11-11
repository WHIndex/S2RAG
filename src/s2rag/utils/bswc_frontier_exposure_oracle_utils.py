"""Diagnostic exposure oracle for source actions outside the live frontier.

The manifest consumed here is frozen before execution.  It may identify an
existing corpus source, but it may not contain passage text, extracted tuples,
QA answers, or endpoint metrics.  Runtime admission additionally requires the
same query, operator contract, bound entity, and a completed zero-match index
lookup.  The utility therefore exposes a physical SOURCE_INSPECT action; it
does not inject a fact or bypass the normal extractor and verifier.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ORACLE_VERSION = "s2rag_identity_safe_frontier_exposure_oracle_v1"


def _truth(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def normalize_key(value: Any) -> str:
    return " ".join(re.findall(r"[\w]+", str(value or "").casefold()))


def operator_contract(operator: Mapping[str, Any]) -> str:
    """Return the contract hash used by the expanded-frontier audit."""
    payload = {
        "operator_id": str(operator.get("operator_id") or ""),
        "input_slot": str(operator.get("input_slot") or ""),
        "output_slot": str(operator.get("output_slot") or ""),
        "predicate_families": sorted(
            str(value) for value in operator.get("predicate_families") or []
        ),
        "predicate_subtypes": sorted(
            str(value)
            for value in operator.get("required_predicate_subtypes") or []
        ),
        "relation_tokens": sorted(
            str(value) for value in operator.get("relation_tokens") or []
        ),
        "direction": str(operator.get("direction") or ""),
        "input_role": str(operator.get("input_role") or ""),
        "output_role": str(operator.get("output_role") or ""),
        "output_type": str(operator.get("output_type") or ""),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=8)
def load_frontier_exposure_oracle(path_value: str) -> Dict[str, Any]:
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"Missing frontier-exposure oracle: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != ORACLE_VERSION:
        raise ValueError("Unsupported frontier-exposure oracle version")
    if _truth(payload.get("qa_gold_used_for_source_selection")):
        raise ValueError("Frontier-exposure oracle must not use QA gold")
    if _truth(payload.get("tuple_injection_allowed")):
        raise ValueError("Frontier-exposure oracle must not inject tuples")
    if _truth(payload.get("source_passage_text_in_manifest")):
        raise ValueError("Frontier-exposure manifest must not contain passage text")
    return {
        **payload,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _bound_identity(
    row: Mapping[str, Any], operator: Mapping[str, Any]
) -> Tuple[str, str]:
    input_slot = str(operator.get("input_slot") or "$root")
    slot = dict((row.get("slot_bindings") or {}).get(input_slot) or {})
    surface = str(slot.get("surface") or "")
    key = str(slot.get("key") or row.get("current_key") or "")
    return surface, normalize_key(key)


def _query_records(state: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    oracle_path = str(state.get("bswc_frontier_exposure_oracle_path") or "")
    if not oracle_path:
        return [], {"reason": "oracle_path_unset"}
    oracle = load_frontier_exposure_oracle(oracle_path)
    query_id = str(state.get("query_id") or "")
    records = [
        dict(record)
        for record in oracle.get("records") or []
        if str(record.get("sample_key") or "") == query_id
        and _truth(record.get("identity_safe"))
    ]
    return records, {
        "oracle_path": oracle["path"],
        "oracle_sha256": oracle["sha256"],
    }


def _binding_operator(
    state: Mapping[str, Any], binding_id: str, obligation_id: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    row = next(
        (
            dict(value) for value in state.get("bindings") or []
            if str(value.get("binding_id") or "") == binding_id
            and value.get("active", True)
            and not value.get("invalidated")
        ),
        {},
    )
    operators = list((state.get("logical_plan") or {}).get("operators") or [])
    operator_index = int(row.get("next_operator_index") or 0) if row else -1
    if operator_index < 0 or operator_index >= len(operators):
        return {}, {}
    operator = dict(operators[operator_index])
    current_obligation = str(
        operator.get("operator_id") or f"op{operator_index}"
    )
    if current_obligation != obligation_id:
        return {}, {}
    return row, operator


def annotate_identity_safe_budget_actions(
    actions: Sequence[Mapping[str, Any]], state: Mapping[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Mark frozen target LOOKUP_JOIN and live SOURCE_INSPECT actions.

    The marker is diagnostic metadata only.  It neither creates a source
    candidate nor injects a tuple.  A source action is marked only after the
    matching lookup has executed with zero matches.
    """
    query_records, oracle_detail = _query_records(state)
    if not query_records:
        return [dict(action) for action in actions], {
            **oracle_detail,
            "reason": "no_identity_safe_record_for_query",
            "marked_lookup_count": 0,
            "marked_source_count": 0,
        }

    attempted = set(state.get("attempted_action_keys") or [])
    lookup_counts = dict(state.get("lookup_match_counts") or {})
    marked: List[Dict[str, Any]] = []
    lookup_count = 0
    source_count = 0
    verify_count = 0
    for raw_action in actions:
        action = dict(raw_action)
        action_type = str(action.get("operator") or "")
        binding_id = str(action.get("binding_id") or "")
        obligation_id = str(action.get("obligation_id") or "")
        if action_type == "VERIFY":
            fact = dict(
                (state.get("facts") or {}).get(action.get("fact_id")) or {}
            )
            created_by = str(fact.get("created_by_action_key") or "")
            created_parts = created_by.split(":")
            selected_record = next(
                (
                    dict(record) for record in query_records
                    if len(created_parts) == 4
                    and created_parts[0] == "SOURCE_INSPECT"
                    and created_parts[2]
                    == str(record.get("obligation_id") or "")
                    and created_parts[3] == str(record.get("source_id") or "")
                ),
                {},
            )
            if selected_record:
                verify_count += 1
                action.update({
                    "identity_safe_budget_reservation": True,
                    "identity_safe_budget_target_verify": True,
                    "frontier_exposure_record_id": str(
                        selected_record.get("record_id") or ""
                    ),
                    "identity_safe_budget_reserve_scan_rows": 0,
                })
            marked.append(action)
            continue
        row, operator = _binding_operator(
            state, binding_id, obligation_id
        )
        if not row or not operator:
            marked.append(action)
            continue
        _, bound_key = _bound_identity(row, operator)
        contract = operator_contract(operator)
        matches = [
            record for record in query_records
            if str(record.get("obligation_id") or "") == obligation_id
            and str(record.get("operator_contract") or "") == contract
            and normalize_key(record.get("bound_key")) == bound_key
        ]
        selected_record: Dict[str, Any] = {}
        if action_type == "LOOKUP_JOIN" and matches:
            selected_record = dict(matches[0])
            lookup_count += 1
            action["identity_safe_budget_target_lookup"] = True
        elif action_type == "SOURCE_INSPECT":
            lookup_key = f"LOOKUP_JOIN:{binding_id}:{obligation_id}"
            source_id = str(action.get("source_id") or "")
            if (
                lookup_key in attempted
                and int(lookup_counts.get(lookup_key) or 0) == 0
            ):
                selected_record = next(
                    (
                        dict(record) for record in matches
                        if str(record.get("source_id") or "") == source_id
                    ),
                    {},
                )
            if selected_record:
                source_count += 1
                action["identity_safe_budget_target_source"] = True
        if selected_record:
            action.update({
                "identity_safe_budget_reservation": True,
                "frontier_exposure_record_id": str(
                    selected_record.get("record_id") or ""
                ),


                "identity_safe_budget_reserve_scan_rows": 2,
            })
        marked.append(action)
    return marked, {
        **oracle_detail,
        "reason": "identity_safe_budget_actions_marked",
        "diagnostic_only": True,
        "candidate_action_injected": False,
        "tuple_injected": False,
        "qa_gold_used_for_source_selection": False,
        "marked_lookup_count": lookup_count,
        "marked_source_count": source_count,
        "marked_verify_count": verify_count,
    }


def identity_safe_reserved_scan_limit(
    action: Mapping[str, Any], available_scan_rows: int
) -> Tuple[int, Dict[str, Any]]:
    """Reserve existing rows for SOURCE_INSPECT plus mandatory VERIFY."""
    available = max(0, int(available_scan_rows))
    requested = max(
        0, int(action.get("identity_safe_budget_reserve_scan_rows") or 0)
    )
    eligible = bool(
        action.get("identity_safe_budget_target_lookup") and requested > 0
    )
    reserved = min(requested, max(0, available - 1)) if eligible else 0
    limit = max(0, available - reserved)
    return limit, {
        "applied": bool(reserved),
        "diagnostic_only": True,
        "available_scan_rows": available,
        "reserved_scan_rows": reserved,
        "lookup_scan_limit": limit,
        "total_scan_budget_unchanged": True,
        "record_id": str(action.get("frontier_exposure_record_id") or ""),
    }


def expose_identity_safe_source_actions(
    state: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Materialize eligible frozen source actions for the current state.

    Exposure is legal only after the corresponding LOOKUP_JOIN has run and
    returned zero matches.  Every returned action still executes the ordinary
    source reader, tuple extractor, obligation admission, and verifier.
    """
    oracle_path = str(state.get("bswc_frontier_exposure_oracle_path") or "")
    if not oracle_path:
        return [], {"reason": "oracle_path_unset", "exposed_count": 0}
    oracle = load_frontier_exposure_oracle(oracle_path)
    query_id = str(state.get("query_id") or "")
    query_records = [
        dict(record)
        for record in oracle.get("records") or []
        if str(record.get("sample_key") or "") == query_id
        and _truth(record.get("identity_safe"))
    ]
    if not query_records:
        return [], {
            "reason": "no_identity_safe_record_for_query",
            "exposed_count": 0,
            "oracle_path": oracle["path"],
            "oracle_sha256": oracle["sha256"],
        }

    plan = dict(state.get("logical_plan") or {})
    operators = list(plan.get("operators") or [])
    attempted = set(state.get("attempted_action_keys") or [])
    lookup_counts = dict(state.get("lookup_match_counts") or {})
    existing_candidates = {
        (
            str(candidate.get("binding_id") or ""),
            str(candidate.get("obligation_id") or ""),
            str(candidate.get("source_id", candidate.get("doc_id", ""))),
        )
        for candidate in state.get("bswc_source_candidates") or []
    }
    actions: List[Dict[str, Any]] = []
    rejected: Dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for row in state.get("bindings") or []:
        if not row.get("active", True) or row.get("invalidated"):
            continue
        operator_index = int(row.get("next_operator_index") or 0)
        if operator_index < 0 or operator_index >= len(operators):
            continue
        operator = dict(operators[operator_index])
        obligation_id = str(operator.get("operator_id") or f"op{operator_index}")
        binding_id = str(row.get("binding_id") or "")
        lookup_key = f"LOOKUP_JOIN:{binding_id}:{obligation_id}"
        if lookup_key not in attempted:
            reject("lookup_not_attempted")
            continue
        if int(lookup_counts.get(lookup_key) or 0) != 0:
            reject("lookup_not_zero_match")
            continue
        bound_surface, bound_key = _bound_identity(row, operator)
        contract = operator_contract(operator)
        matching = [
            record
            for record in query_records
            if str(record.get("obligation_id") or "") == obligation_id
            and str(record.get("operator_contract") or "") == contract
            and normalize_key(record.get("bound_key")) == bound_key
        ]
        if not matching:
            reject("binding_or_contract_mismatch")
            continue
        for record in matching:
            source_id = str(record.get("source_id") or "")
            try:
                source_id_int = int(source_id)
            except (TypeError, ValueError):
                reject("invalid_source_id")
                continue
            identity = (binding_id, obligation_id, source_id)
            if identity in existing_candidates:
                reject("already_live_candidate")
                continue
            action_key = f"SOURCE_INSPECT:{binding_id}:{obligation_id}:{source_id}"
            if action_key in attempted:
                reject("already_attempted")
                continue
            raw_candidate = dict(record.get("source_candidate") or {})
            candidate = {
                key: value
                for key, value in raw_candidate.items()
                if key not in {"content", "passage", "text", "qa_answer", "gold_answer"}
            }
            candidate.update({
                "source_id": source_id_int,
                "doc_id": source_id_int,
                "binding_id": binding_id,
                "obligation_id": obligation_id,
                "operator_index": operator_index,
                "coverage_key": f"{binding_id}:{obligation_id}",
                "bound_surface": bound_surface,
                "bound_key": bound_key,
                "frontier_exposure_record_id": str(record.get("record_id") or ""),
            })
            actions.append({
                "operator": "SOURCE_INSPECT",
                "binding_id": binding_id,
                "obligation_id": obligation_id,
                "operator_index": operator_index,
                "source_id": source_id_int,
                "source_type": str(candidate.get("source_type") or "frozen_source_oracle"),
                "source_rank": int(candidate.get("rank") or 0),
                "source_relevance": float(candidate.get("score") or 0.0),
                "source_candidate": candidate,
                "predicted_coverage_keys": [f"{binding_id}:{obligation_id}"],
                "binding_compatible_coverage": 1,
                "source_reused": source_id_int in {
                    int(value) for value in state.get("bswc_inspected_source_ids") or []
                },
                "identity_safe_frontier_exposure": True,
                "frontier_exposure_record_id": str(record.get("record_id") or ""),
            })
    return actions, {
        "reason": "identity_safe_actions_exposed" if actions else "no_legal_exposure",
        "diagnostic_only": True,
        "candidate_action_injected": bool(actions),
        "tuple_injected": False,
        "qa_gold_used_for_source_selection": False,
        "exposed_count": len(actions),
        "rejected_counts": rejected,
        "oracle_path": oracle["path"],
        "oracle_sha256": oracle["sha256"],
    }
