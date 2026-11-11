"""Frozen support-oracle policy used only to diagnose source scheduling.

The policy cannot create candidates, inspect passage text, or use QA answers.
It can only prioritize a SOURCE_INSPECT action already admitted by the live
executor when a frozen prior trace observed that exact
(query, obligation, bound entity, source) action produce a supported tuple.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .bswc_optimizer_utils import source_action_score


ORACLE_VERSION = "s2rag_frozen_source_support_oracle_v1"


def normalize_bound_key(value: Any) -> str:
    return " ".join(re.findall(r"[\w]+", str(value or "").casefold()))


def _stable_key(
    query_id: Any, obligation_id: Any, bound_key: Any, source_id: Any
) -> str:
    return "\x1f".join((
        str(query_id or ""),
        str(obligation_id or ""),
        normalize_bound_key(bound_key),
        str(source_id if source_id is not None else ""),
    ))


@lru_cache(maxsize=8)
def load_support_oracle(path_text: str) -> Dict[str, Any]:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Missing frozen support oracle: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != ORACLE_VERSION:
        raise ValueError(f"Unsupported support oracle version: {payload.get('version')}")
    if payload.get("qa_gold_used_for_source_selection") is not False:
        raise ValueError("Support oracle must explicitly prohibit QA-gold source selection")
    if payload.get("candidate_injection_allowed") is not False:
        raise ValueError("Support oracle must explicitly prohibit candidate injection")

    records: Dict[str, Dict[str, Any]] = {}
    relaxed: Dict[str, list[Dict[str, Any]]] = {}
    for raw in payload.get("records") or []:
        row = dict(raw)
        if str(row.get("observed_outcome") or "") != "SUPPORTED":
            continue
        key = _stable_key(
            row.get("sample_key"), row.get("obligation_id"),
            row.get("bound_key"), row.get("source_id"),
        )
        records[key] = row
        relaxed_key = _stable_key(
            row.get("sample_key"), row.get("obligation_id"), "", row.get("source_id")
        )
        relaxed.setdefault(relaxed_key, []).append(row)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "records": records,
        "relaxed": relaxed,
        "record_count": len(records),
    }


def _bound_key(action: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    candidate = dict(action.get("source_candidate") or {})
    direct = (
        candidate.get("bound_key")
        or candidate.get("candidate_entity_key")
        or candidate.get("bound_surface")
    )
    if direct:
        return normalize_bound_key(direct)

    binding_id = str(action.get("binding_id") or "")
    row = next((
        dict(value) for value in state.get("bindings") or []
        if str(value.get("binding_id") or "") == binding_id
    ), {})
    operators = list((state.get("logical_plan") or {}).get("operators") or [])
    operator_index = int(action.get("operator_index", -1) or -1)
    operator = (
        dict(operators[operator_index])
        if 0 <= operator_index < len(operators) else {}
    )
    input_slot = str(operator.get("input_slot") or "")
    slot = dict((row.get("slot_bindings") or {}).get(input_slot) or {})
    return normalize_bound_key(
        slot.get("key") or slot.get("surface")
        or row.get("current_key") or row.get("root_key")
    )


def _match_action(
    action: Mapping[str, Any], state: Mapping[str, Any], oracle: Mapping[str, Any]
) -> Tuple[Optional[Dict[str, Any]], str]:
    if str(action.get("operator") or "") != "SOURCE_INSPECT":
        return None, "not_source_inspect"
    query_id = str(state.get("query_id") or "")
    obligation_id = str(action.get("obligation_id") or "")
    source_id = str(action.get("source_id") or "")
    bound_key = _bound_key(action, state)
    exact = oracle["records"].get(
        _stable_key(query_id, obligation_id, bound_key, source_id)
    )
    if exact is not None:
        return dict(exact), "exact_bound_identity"

    if bound_key:
        return None, "bound_identity_mismatch"



    candidates = oracle["relaxed"].get(
        _stable_key(query_id, obligation_id, "", source_id), []
    )
    unique_bounds = {normalize_bound_key(row.get("bound_key")) for row in candidates}
    if len(candidates) == 1 and len(unique_bounds) == 1:
        return dict(candidates[0]), "unique_query_obligation_source"
    return None, "no_frozen_supported_match"


def support_oracle_first_action(
    actions: Sequence[Mapping[str, Any]],
    estimates: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Choose a currently executable frozen-supported source, else Greedy.

    This is an offline upper-bound diagnostic, not a deployable policy.  The
    fallback is exactly the immediate set-cover score used by the strong
    Greedy baseline, so all differences come from the frozen source ordering.
    """
    path = str(state.get("bswc_support_oracle_path") or "")
    if not path:
        raise ValueError(
            "bswc_support_oracle_early requires S2RAG_BSWC_SUPPORT_ORACLE_PATH"
        )
    oracle = load_support_oracle(path)
    matched = []
    ranked = []
    for action_index, raw in enumerate(actions):
        action = dict(raw)
        key = str(action.get("action_key") or "")
        estimate = dict(estimates.get(key) or {})
        score, _ = source_action_score(
            "bswc_set_cover_greedy", action, estimate, state
        )
        record, match_mode = _match_action(action, state, oracle)


        rank = (float(score), -action_index)
        ranked.append((rank, action))
        if record is not None:
            matched.append((rank, action, record, match_mode))

    greedy_selected = (
        max(ranked, key=lambda item: item[0])[1] if ranked else None
    )
    greedy_action_key = str(
        (greedy_selected or {}).get("action_key") or ""
    )

    if matched:
        _, selected, record, match_mode = max(matched, key=lambda item: item[0])
        selected_action_key = str(selected.get("action_key") or "")
        return dict(selected), {
            "reason": "frozen_support_oracle_current_frontier",
            "diagnostic_only": True,
            "candidate_injected": False,
            "qa_gold_used_for_source_selection": False,
            "oracle_path": oracle["path"],
            "oracle_sha256": oracle["sha256"],
            "oracle_record_count": oracle["record_count"],
            "current_frontier_match_count": len(matched),
            "greedy_action_key": greedy_action_key,
            "oracle_action_key": selected_action_key,
            "oracle_changed_vs_greedy": bool(
                selected_action_key != greedy_action_key
            ),
            "match_mode": match_mode,
            "matched_bound_key": normalize_bound_key(record.get("bound_key")),
            "matched_source_id": str(record.get("source_id") or ""),
            "target_reference_discovered_expanded": bool(
                record.get("target_reference_discovered_expanded")
            ),
        }

    if not ranked:
        return None, {
            "reason": "no_executable_action",
            "diagnostic_only": True,
            "current_frontier_match_count": 0,
            "greedy_action_key": "",
            "oracle_action_key": "",
            "oracle_changed_vs_greedy": False,
        }
    selected = dict(greedy_selected or {})
    selected_action_key = str(selected.get("action_key") or "")
    return dict(selected), {
        "reason": "set_cover_greedy_fallback_no_visible_oracle_match",
        "diagnostic_only": True,
        "candidate_injected": False,
        "qa_gold_used_for_source_selection": False,
        "oracle_path": oracle["path"],
        "oracle_sha256": oracle["sha256"],
        "oracle_record_count": oracle["record_count"],
        "current_frontier_match_count": 0,
        "greedy_action_key": greedy_action_key,
        "oracle_action_key": selected_action_key,
        "oracle_changed_vs_greedy": False,
    }
