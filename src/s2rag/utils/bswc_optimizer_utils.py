"""Physical source-selection policies for Budgeted Witness Completion.

The functions in this module are deliberately independent from the executor.
They consume observable action metadata and never inspect passage text or QA
gold answers.  This keeps source planning replayable and makes the small exact
solver suitable as an offline oracle on deterministic candidate pools.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


BSWC_POLICIES = {
    "bswc_fixed_source_order",
    "bswc_probability_only",
    "bswc_relevance",
    "bswc_lowest_cost",
    "bswc_set_cover_greedy",
    "bswc_adaptive_greedy",
    "bswc_bounded_search",
    "bswc_full",
    "bswc_empirical_source_type",
    "bswc_feedback_calibrated",
    "bswc_feedback_replan",
    "bswc_feedback_stopping",
    "bswc_witness_calibrated",
    "bswc_witness_stopping",
    "bswc_feedback_full",
    "bswc_random_source",
    "bswc_risk_aware_search",
    "bswc_adaptive_risk_cover",
    "bswc_feasibility_preserving_cover",
    "bswc_continuation_safe_cover",
    "bswc_binding_qualified_continuation_cover",
    "bswc_rank_consistent_binding_qualified_cover",
    "bswc_precedence_aware_cover",
    "bswc_quality_first_precedence_cover",
    "bswc_global_witness_reservation_cover",
    "bswc_exact_adaptive_witness",
    "bswc_support_oracle_early",
    "bswc_identity_safe_frontier_exposure_oracle",
    "bswc_identity_safe_scan_reservation_oracle",
}

WITNESS_VALUE_POLICIES = {
    "bswc_witness_calibrated",
    "bswc_witness_stopping",
    "bswc_feedback_full",
}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coverage(action: Mapping[str, Any]) -> Set[str]:
    values = action.get("predicted_coverage_keys") or []
    if values:
        return {str(value) for value in values if str(value)}
    binding = str(action.get("binding_id") or "")
    obligation = str(action.get("obligation_id") or "")
    return {f"{binding}:{obligation}"} if binding and obligation else set()


def _open_coverage_keys(state: Mapping[str, Any]) -> Set[str]:
    keys = {
        str(value)
        for value in state.get("bswc_open_coverage_keys") or []
        if str(value)
    }
    if keys:
        return keys
    return {
        f"{row.get('binding_id', '')}:{row.get('obligation_id', '')}"
        for row in state.get("bswc_source_candidates") or []
        if row.get("binding_id") and row.get("obligation_id")
    }


def normalized_physical_cost(estimate: Mapping[str, Any]) -> float:
    """Convert the multidimensional resource estimate to a stable scalar.

    The scalar is used only for ranking.  Admission still uses the executor's
    hard resource ledger, so this function cannot make an unaffordable action
    executable.
    """
    scans = max(0.0, _float(estimate.get("expected_scan_rows")))
    retrievals = max(0.0, _float(estimate.get("expected_retrieval_calls")))
    models = max(0.0, _float(estimate.get("expected_model_calls")))
    tokens = max(0.0, _float(estimate.get("expected_tokens")))
    return max(0.05, scans + 8.0 * retrievals + 12.0 * models + tokens / 512.0)


_FRONTIER_ACTION_FIELDS = (
    "action_key",
    "operator",
    "binding_id",
    "obligation_id",
    "operator_index",
    "source_id",
    "source_type",
    "source_rank",
    "source_relevance",
    "source_reused",
    "binding_compatible_coverage",
    "binding_identity_qualified",
    "binding_identity_key",
    "binding_qualified_coverage",
)

_FRONTIER_ESTIMATE_FIELDS = (
    "support_probability",
    "expected_progress",
    "expected_witness_progress",
    "binding_compatible_coverage",
    "source_relevance",
    "source_reused",
    "reader_evidence_novelty",
    "expected_scan_rows",
    "expected_retrieval_calls",
    "expected_model_calls",
    "expected_tokens",
    "expected_evidence_tokens",
    "hard_scan_rows",
    "hard_retrieval_calls",
    "hard_model_calls",
    "scan_rows_truncatable",
    "retrieval_optional",
    "bswc_calibrated",
    "bswc_calibration_probability",
    "bswc_calibration_sample_count",
    "bswc_calibration_backoff_level",
    "bswc_calibration_uncertainty",
    "query_feedback_count",
    "query_feedback_success",
    "query_feedback_failure",
    "estimate_source",
)


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _frontier_binding_rows(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return only execution-state fields needed to identify a decision state."""
    rows = []
    for raw in state.get("bindings") or []:
        if not raw.get("active", True) or raw.get("invalidated"):
            continue
        rows.append({
            "binding_id": str(raw.get("binding_id") or ""),
            "root_binding_id": str(raw.get("root_binding_id") or ""),
            "next_operator_index": int(_float(raw.get("next_operator_index"))),
            "current_key": str(
                raw.get("current_key")
                or raw.get("value_key")
                or raw.get("entity_key")
                or ""
            ),
            "path_fact_ids": sorted(
                str(value) for value in raw.get("path_fact_ids") or [] if str(value)
            ),
        })
    return sorted(rows, key=lambda row: (
        row["binding_id"], row["next_operator_index"], row["current_key"]
    ))


def frozen_source_frontier_snapshot(
    actions: Sequence[Mapping[str, Any]],
    estimates: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
) -> Dict[str, Any] | None:
    """Serialize a policy-neutral SOURCE_INSPECT decision frontier.

    The snapshot is captured before physical access.  It intentionally omits
    policy scores, the selected action, realized outcomes, passage text, and QA
    gold fields.  Offline policies can therefore be replayed over exactly the
    same action set and estimates without confusing later execution states for
    the original decision frontier.
    """
    source_actions = [
        dict(action) for action in actions
        if str(action.get("operator") or "") == "SOURCE_INSPECT"
    ]
    if not source_actions:
        return None

    all_open = _open_coverage_keys(state)
    candidates: List[Dict[str, Any]] = []
    structural_rows: List[Dict[str, Any]] = []
    estimate_rows: List[Dict[str, Any]] = []
    grouped: Dict[str, Dict[str, Any]] = {}
    executable: Set[str] = set()

    for raw_action in source_actions:
        action_key = str(raw_action.get("action_key") or "")
        if not action_key:
            continue
        estimate = dict(estimates.get(action_key) or {})
        coverage = _coverage(raw_action)
        if all_open:
            coverage &= all_open
        if not coverage:
            continue
        executable.update(coverage)
        action_row = {
            key: raw_action.get(key)
            for key in _FRONTIER_ACTION_FIELDS
            if key in raw_action
        }
        action_row["action_key"] = action_key
        action_row["operator"] = "SOURCE_INSPECT"
        action_row["predicted_coverage_keys"] = sorted(coverage)
        estimate_row = {
            key: estimate.get(key)
            for key in _FRONTIER_ESTIMATE_FIELDS
            if key in estimate
        }
        estimate_row["normalized_physical_cost"] = round(
            normalized_physical_cost(estimate), 8
        )
        candidate = {**action_row, "estimate": estimate_row}
        candidates.append(candidate)

        structural_rows.append({
            **action_row,
            "hard_scan_rows": estimate.get("hard_scan_rows"),
            "hard_retrieval_calls": estimate.get("hard_retrieval_calls"),
            "hard_model_calls": estimate.get("hard_model_calls"),
        })
        estimate_rows.append({"action_key": action_key, **estimate_row})

        source_id = str(raw_action.get("source_id") or "")
        group = grouped.setdefault(source_id, {
            "source_id": source_id,
            "action_keys": [],
            "predicted_coverage_keys": set(),
            "probability_by_key": {},
            "cost": None,
        })
        group["action_keys"].append(action_key)
        group["predicted_coverage_keys"].update(coverage)
        probability = min(1.0, max(0.0, _float(
            estimate.get("support_probability"), 0.5
        )))
        own_key = ":".join((
            str(raw_action.get("binding_id") or ""),
            str(raw_action.get("obligation_id") or ""),
        ))
        if own_key in coverage:
            group["probability_by_key"][own_key] = max(
                probability,
                _float(group["probability_by_key"].get(own_key)),
            )
        cost = normalized_physical_cost(estimate)
        group["cost"] = cost if group["cost"] is None else min(
            _float(group["cost"]), cost
        )

    if not candidates:
        return None
    candidates.sort(key=lambda row: str(row.get("action_key") or ""))
    structural_rows.sort(key=lambda row: str(row.get("action_key") or ""))
    estimate_rows.sort(key=lambda row: str(row.get("action_key") or ""))

    physical_sources = []
    for source_id, raw_group in sorted(grouped.items()):
        probabilities = dict(raw_group["probability_by_key"])
        source_candidates = [
            row for row in candidates if str(row.get("source_id") or "") == source_id
        ]
        fallback = min(
            (
                _float((row.get("estimate") or {}).get("support_probability"), 0.5)
                for row in source_candidates
            ),
            default=0.0,
        )
        for key in raw_group["predicted_coverage_keys"]:
            probabilities.setdefault(str(key), fallback)
        action_obligations = {
            ":".join((
                str(row.get("binding_id") or ""),
                str(row.get("obligation_id") or ""),
            ))
            for row in source_candidates
            if row.get("binding_id") and row.get("obligation_id")
        }
        resource_cost = {


            "actions": max(1, len(action_obligations)),
            "scan_rows": max(
                (
                    max(0, int(_float(
                        (row.get("estimate") or {}).get(
                            "hard_scan_rows",
                            (row.get("estimate") or {}).get("expected_scan_rows"),
                        )
                    )))
                    for row in source_candidates
                ),
                default=0,
            ),
            "retrieval_calls": max(
                (
                    max(0, int(_float(
                        (row.get("estimate") or {}).get(
                            "hard_retrieval_calls",
                            (row.get("estimate") or {}).get(
                                "expected_retrieval_calls"
                            ),
                        )
                    )))
                    for row in source_candidates
                ),
                default=0,
            ),
            "model_calls": max(
                (
                    max(0, int(_float(
                        (row.get("estimate") or {}).get(
                            "hard_model_calls",
                            (row.get("estimate") or {}).get("expected_model_calls"),
                        )
                    )))
                    for row in source_candidates
                ),
                default=0,
            ),
            "tokens": max(
                (
                    max(0, int(_float(
                        (row.get("estimate") or {}).get("expected_tokens")
                    )))
                    for row in source_candidates
                ),
                default=0,
            ),
        }
        physical_sources.append({
            "name": source_id,
            "source_id": source_id,
            "action_keys": sorted(raw_group["action_keys"]),
            "predicted_coverage_keys": sorted(
                str(value) for value in raw_group["predicted_coverage_keys"]
            ),
            "probability_by_key": {
                str(key): round(_float(value), 10)
                for key, value in sorted(probabilities.items())
            },
            "support_probability": round(fallback, 10),
            "cost": round(max(0.05, _float(raw_group["cost"], 1.0)), 8),
            "resource_cost": resource_cost,
        })

    deferred = all_open - executable
    if len(all_open) >= 2 and len(executable) < 2:
        frontier_class = "precedence_blocked"
    elif len(executable) >= 2 and len(physical_sources) >= 2:
        frontier_class = "nontrivial_source_cover"
    elif len(executable) == 1 and len(physical_sources) >= 2:
        frontier_class = "source_ranking"
    else:
        frontier_class = "trivial_source_frontier"

    candidate_pool_fingerprint = _stable_json_hash(structural_rows)
    estimate_fingerprint = _stable_json_hash(estimate_rows)
    remaining_budget = {
        str(key): value
        for key, value in dict(state.get("bswc_remaining_budget") or {}).items()
        if key in {
            "actions_remaining",
            "scan_rows_remaining",
            "retrieval_calls_remaining",
            "model_calls_remaining",
            "known_tokens_remaining",
        }
    }
    state_payload = {
        "query_id": str(state.get("query_id") or ""),
        "all_open_coverage_keys": sorted(all_open),
        "active_bindings": _frontier_binding_rows(state),
        "remaining_budget": remaining_budget,
        "source_inspection_count": max(
            0, int(_float(state.get("bswc_source_inspection_count")))
        ),
    }
    state_fingerprint = _stable_json_hash(state_payload)
    return {
        "version": "bswc_frozen_source_frontier_v1",
        "query_id": str(state.get("query_id") or ""),
        "all_open_coverage_keys": sorted(all_open),
        "source_executable_coverage_keys": sorted(executable),
        "deferred_open_coverage_keys": sorted(deferred),
        "source_action_count": len(candidates),
        "unique_physical_source_count": len(physical_sources),
        "frontier_class": frontier_class,
        "candidate_pool_fingerprint": candidate_pool_fingerprint,
        "estimate_fingerprint": estimate_fingerprint,
        "state_fingerprint": state_fingerprint,
        "frontier_fingerprint": _stable_json_hash({
            "state_fingerprint": state_fingerprint,
            "candidate_pool_fingerprint": candidate_pool_fingerprint,
            "estimate_fingerprint": estimate_fingerprint,
        }),
        "active_bindings": state_payload["active_bindings"],
        "remaining_budget": remaining_budget,
        "source_inspection_count": state_payload["source_inspection_count"],
        "candidates": candidates,
        "physical_sources": physical_sources,
    }


def source_action_score(
    policy: str,
    action: Mapping[str, Any],
    estimate: Mapping[str, Any],
    state: Mapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Score one executable action using only pre-access metadata."""
    operator = str(action.get("operator") or "")
    support = min(1.0, max(0.0, _float(estimate.get("support_probability"), 0.5)))
    progress = min(1.0, max(0.0, _float(estimate.get("expected_progress"), 0.0)))
    relevance = min(1.0, max(0.0, _float(
        action.get("source_relevance", estimate.get("source_relevance", 0.0))
    )))
    cost = normalized_physical_cost(estimate)
    open_keys = _open_coverage_keys(state)
    uncovered = _coverage(action)
    if open_keys:
        uncovered &= open_keys
    coverage_count = len(uncovered)
    binding_compatible = max(
        0.0,
        _float(
            estimate.get(
                "binding_compatible_coverage",
                action.get("binding_compatible_coverage", coverage_count),
            )
        ),
    )
    reuse = bool(
        action.get("source_reused")
        or estimate.get("source_reused")
        or str(action.get("source_id") or "")
        in {str(value) for value in state.get("bswc_inspected_source_ids") or []}
    )
    remaining = max(1, len(open_keys)) if open_keys else max(1, coverage_count)
    witness_fraction = min(1.0, binding_compatible / float(remaining))

    if policy == "bswc_probability_only":



        score = support
    elif policy == "bswc_lowest_cost":
        score = -cost + 0.001 * relevance
    elif operator == "VERIFY":
        downstream = max(0.0, _float(estimate.get("downstream_cost")))
        downstream += max(
            0.0, _float(estimate.get("expected_downstream_scan_rows")) / 25.0
        )
        score = (progress + (1.0 - support) * downstream) / cost
    elif operator == "LOOKUP_JOIN":
        pruning = max(0.0, _float(estimate.get("expected_pruning")))
        score = (1.4 * progress + 0.4 * pruning) / cost
    elif policy == "bswc_relevance":
        score = relevance + 0.05 * progress - 0.001 * cost
    elif policy == "bswc_empirical_source_type":


        score = support - 0.002 * cost
    elif policy == "bswc_set_cover_greedy":
        score = float(coverage_count) / cost
    else:
        expected_supported_gain = binding_compatible * support
        expected_witness_progress = min(1.0, max(0.0, _float(
            estimate.get("expected_witness_progress"),
            support * witness_fraction,
        )))
        completion_bonus = support * witness_fraction


        feedback_failures = max(0, int(_float(
            estimate.get("query_feedback_failure"), 0.0
        )))
        reuse_bonus = 0.45 if reuse and feedback_failures == 0 else 0.0
        failure_risk = (1.0 - support) * (0.35 + 0.15 * binding_compatible)
        if policy in WITNESS_VALUE_POLICIES:
            reader_novelty = min(1.0, max(0.0, _float(
                estimate.get("reader_evidence_novelty"), 1.0
            )))
            score = (
                expected_supported_gain * expected_witness_progress
                + 0.40 * completion_bonus
                + 0.20 * reader_novelty
                + reuse_bonus
                + 0.05 * relevance
                - failure_risk
                - 0.25 * feedback_failures
            ) / cost
        else:
            score = (
                expected_supported_gain
                + 0.75 * completion_bonus
                + 0.25 * progress
                + reuse_bonus
                + 0.10 * relevance
                - failure_risk
            ) / cost

    return round(score, 10), {
        "policy": policy,
        "operator": operator,
        "coverage_keys": sorted(uncovered),
        "coverage_count": coverage_count,
        "binding_compatible_coverage": binding_compatible,
        "support_probability": support,
        "source_relevance": relevance,
        "source_reused": reuse,
        "witness_fraction": witness_fraction,
        "expected_witness_progress": round(
            min(1.0, max(0.0, _float(
                estimate.get("expected_witness_progress"),
                support * witness_fraction,
            ))),
            8,
        ),
        "query_feedback_count": int(_float(
            estimate.get("query_feedback_count"), 0.0
        )),
        "query_feedback_failure": int(_float(
            estimate.get("query_feedback_failure"), 0.0
        )),
        "normalized_cost": round(cost, 8),
    }


@dataclass(frozen=True)
class _BeamState:
    selected: Tuple[int, ...]
    covered: frozenset[str]
    expected_gain: float
    cost: float


@dataclass(frozen=True)
class _RiskBeamState:
    selected: Tuple[int, ...]
    failure_probabilities: Tuple[Tuple[str, float], ...]
    cost: float


def _risk_adjusted_probability(estimate: Mapping[str, Any]) -> float:
    """Return a conservative calibrated probability available before access."""
    probability = min(
        1.0, max(0.0, _float(estimate.get("support_probability"), 0.5))
    )
    uncertainty = max(
        0.0, _float(estimate.get("bswc_calibration_uncertainty"), 0.0)
    )



    return min(1.0, max(0.0, probability / (1.0 + uncertainty)))


def _source_groups(
    actions: Sequence[Mapping[str, Any]],
    estimates: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Group obligation-specific inspect actions by physical source.

    One passage access may support several live obligations.  The executor
    exposes one SOURCE_INSPECT action per binding-obligation pair, so planning
    over those rows directly would charge and select the same passage multiple
    times.  This grouping is the physical source relation used by BSWC.
    """
    open_keys = _open_coverage_keys(state)
    grouped: Dict[str, Dict[str, Any]] = {}
    for raw in actions:
        if str(raw.get("operator") or "") != "SOURCE_INSPECT":
            continue
        action = dict(raw)
        action_key = str(action.get("action_key") or "")
        raw_source_id = action.get("source_id")
        source_id = "" if raw_source_id is None else str(raw_source_id)
        if not action_key or not source_id:
            continue
        estimate = dict(estimates.get(action_key) or {})
        own_key = ":".join((
            str(action.get("binding_id") or ""),
            str(action.get("obligation_id") or ""),
        ))
        coverage = _coverage(action)
        if open_keys:
            coverage &= open_keys
        if not coverage:
            continue
        row = grouped.setdefault(source_id, {
            "source_id": source_id,
            "actions": [],
            "coverage_keys": set(),
            "probability_by_key": {},
            "cost": None,
        })
        probability = _risk_adjusted_probability(estimate)
        row["actions"].append({
            "action": action,
            "estimate": estimate,
            "own_key": own_key,
            "probability": probability,
        })
        row["coverage_keys"].update(coverage)
        if own_key in coverage:
            row["probability_by_key"][own_key] = max(
                probability,
                _float(row["probability_by_key"].get(own_key)),
            )
        cost = normalized_physical_cost(estimate)
        row["cost"] = cost if row["cost"] is None else min(row["cost"], cost)

    result: List[Dict[str, Any]] = []
    for source_id, row in sorted(grouped.items()):
        probabilities = dict(row["probability_by_key"])



        observed = [item["probability"] for item in row["actions"]]
        fallback = min(observed) if observed else 0.0
        for key in row["coverage_keys"]:
            probabilities.setdefault(key, fallback)
        result.append({
            "source_id": source_id,
            "actions": list(row["actions"]),
            "coverage_keys": set(row["coverage_keys"]),
            "probability_by_key": probabilities,
            "cost": max(0.05, _float(row["cost"], 1.0)),
            "resource_cost": {
                "actions": max(
                    1,
                    len({
                        str(item.get("own_key") or "")
                        for item in row["actions"]
                        if str(item.get("own_key") or "")
                    }),
                ),
                "scan_rows": max(
                    (
                        max(0, int(_float(
                            item["estimate"].get(
                                "hard_scan_rows",
                                item["estimate"].get("expected_scan_rows"),
                            )
                        )))
                        for item in row["actions"]
                    ),
                    default=0,
                ),
                "retrieval_calls": max(
                    (
                        max(0, int(_float(
                            item["estimate"].get(
                                "hard_retrieval_calls",
                                item["estimate"].get("expected_retrieval_calls"),
                            )
                        )))
                        for item in row["actions"]
                    ),
                    default=0,
                ),
                "model_calls": max(
                    (
                        max(0, int(_float(
                            item["estimate"].get(
                                "hard_model_calls",
                                item["estimate"].get("expected_model_calls"),
                            )
                        )))
                        for item in row["actions"]
                    ),
                    default=0,
                ),
                "tokens": max(
                    (
                        max(0, int(_float(
                            item["estimate"].get("expected_tokens")
                        )))
                        for item in row["actions"]
                    ),
                    default=0,
                ),
            },
        })
    return result


def _risk_plan_metrics(
    obligations: Set[str], failure_probabilities: Mapping[str, float], cost: float
) -> Dict[str, float]:
    coverage_probabilities = {
        key: 1.0 - min(1.0, max(0.0, failure_probabilities.get(key, 1.0)))
        for key in obligations
    }
    complete_probability = math.prod(coverage_probabilities.values())\
        if obligations else 0.0
    expected_coverage = sum(coverage_probabilities.values())
    return {
        "complete_probability": complete_probability,
        "expected_coverage": expected_coverage,
        "cost": cost,
    }


def _risk_plan_objective(
    obligations: Set[str], failure_probabilities: Mapping[str, float], cost: float
) -> Tuple[int, float, float, float]:
    metrics = _risk_plan_metrics(obligations, failure_probabilities, cost)
    structurally_complete = int(
        bool(obligations)
        and all(failure_probabilities.get(key, 1.0) < 1.0 for key in obligations)
    )
    return (
        structurally_complete,
        metrics["complete_probability"],
        metrics["expected_coverage"] / max(0.05, cost),
        -cost,
    )


def _source_action_resource_cost(
    estimate: Mapping[str, Any], *, source_already_inspected: bool
) -> Dict[str, float]:
    """Return the executor-aligned hard cost of one SOURCE_INSPECT action."""
    scan_rows = max(0.0, _float(
        estimate.get("hard_scan_rows", estimate.get("expected_scan_rows"))
    ))
    if source_already_inspected:
        scan_rows = 0.0
    return {
        "actions": 1.0,
        "scan_rows": scan_rows,
        "retrieval_calls": max(0.0, _float(estimate.get(
            "hard_retrieval_calls", estimate.get("expected_retrieval_calls")
        ))),
        "model_calls": max(0.0, _float(estimate.get(
            "hard_model_calls", estimate.get("expected_model_calls")
        ))),
        "tokens": max(0.0, _float(estimate.get("expected_tokens"))),
    }


def _resource_scalar_cost(cost: Mapping[str, Any]) -> float:
    return (
        _float(cost.get("scan_rows"))
        + 8.0 * _float(cost.get("retrieval_calls"))
        + 12.0 * _float(cost.get("model_calls"))
        + _float(cost.get("tokens")) / 512.0
    )


def exact_adaptive_supported_witness_plan(
    actions: Sequence[Mapping[str, Any]],
    estimates: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    *,
    max_candidates: int = 18,
) -> Dict[str, Any]:
    """Solve the finite SOURCE_INSPECT policy exactly under a hard budget.

    Each action attempts one binding-specific obligation.  Its support outcome
    is Bernoulli under the frozen pre-access calibration model.  The Bellman
    state records remaining actions, supported obligations, inspected physical
    sources, and every bounded resource dimension.  Reusing a physical source
    therefore consumes an executor action but no second scan, matching
    ``RelationIndexEvidenceBackend.inspect_source``.

    The primary objective is the probability that every currently executable
    obligation becomes supported.  Expected supported-obligation coverage and
    expected physical cost are deterministic tie breakers.  There is no beam,
    candidate pruning, greedy fallback, QA answer, or realized source outcome.
    The result is exact for the supplied finite action set, independent-outcome
    model, and resource ledger; it is not an exact solution for unseen future
    obligations or an unbounded corpus.
    """
    source_actions = [
        dict(action) for action in actions
        if str(action.get("operator") or "") == "SOURCE_INSPECT"
    ]
    source_actions.sort(key=lambda row: str(row.get("action_key") or ""))
    if not source_actions:
        return {"status": "infeasible", "reason": "empty_source_frontier"}
    if len(source_actions) > max(0, int(max_candidates)):
        return {
            "status": "capacity_exceeded",
            "candidate_count": len(source_actions),
            "max_candidates": max(0, int(max_candidates)),
            "heuristic_fallback_used": False,
        }

    obligation_names = sorted({
        f"{action.get('binding_id', '')}:{action.get('obligation_id', '')}"
        for action in source_actions
        if action.get("binding_id") and action.get("obligation_id")
    })
    if not obligation_names:
        return {"status": "infeasible", "reason": "missing_obligation_keys"}
    obligation_bits = {
        name: 1 << index for index, name in enumerate(obligation_names)
    }
    full_coverage_mask = (1 << len(obligation_names)) - 1

    source_names = sorted({
        str(action.get("source_id") or "") for action in source_actions
    })
    source_bits = {name: 1 << index for index, name in enumerate(source_names)}
    initially_inspected = {
        str(value) for value in state.get("bswc_inspected_source_ids") or []
    }
    initial_source_mask = sum(
        bit for name, bit in source_bits.items() if name in initially_inspected
    )

    candidates: List[Dict[str, Any]] = []
    structurally_coverable = 0
    for action in source_actions:
        action_key = str(action.get("action_key") or "")
        own_key = f"{action.get('binding_id', '')}:{action.get('obligation_id', '')}"
        bit = obligation_bits.get(own_key, 0)
        if not action_key or not bit:
            continue
        estimate = dict(estimates.get(action_key) or {})
        source_id = str(action.get("source_id") or "")
        candidates.append({
            "action": action,
            "action_key": action_key,
            "obligation_key": own_key,
            "obligation_bit": bit,
            "source_id": source_id,
            "source_bit": source_bits.get(source_id, 0),
            "probability": _risk_adjusted_probability(estimate),
            "estimate": estimate,
        })
        structurally_coverable |= bit
    if structurally_coverable != full_coverage_mask:
        return {
            "status": "infeasible",
            "reason": "uncoverable_executable_obligation",
            "obligations": obligation_names,
        }

    remaining_budget = dict(state.get("bswc_remaining_budget") or {})
    bounded_resources = tuple(
        resource for resource, remaining_key in _RESOURCE_REMAINING_KEYS.items()
        if remaining_budget.get(remaining_key) is not None
    )
    initial_remaining = tuple(
        round(
            max(0.0, _float(
                remaining_budget.get(_RESOURCE_REMAINING_KEYS[resource]), 0.0
            )),
            8,
        )
        for resource in bounded_resources
    )

    explored_state_count = 0
    feasible_transition_count = 0
    budget_pruned_transition_count = 0

    def affordable(
        candidate: Mapping[str, Any], inspected_mask: int,
        remaining: Tuple[float, ...],
    ) -> Tuple[Tuple[float, ...], Dict[str, float]] | None:
        source_bit = int(candidate.get("source_bit") or 0)
        resource_cost = _source_action_resource_cost(
            candidate.get("estimate") or {},
            source_already_inspected=bool(inspected_mask & source_bit),
        )
        next_remaining = []
        for position, resource in enumerate(bounded_resources):
            value = remaining[position] - _float(resource_cost.get(resource))
            if value < -1e-12:
                return None
            next_remaining.append(round(max(0.0, value), 8))
        return tuple(next_remaining), resource_cost



    @lru_cache(maxsize=None)
    def solve(
        available_mask: int,
        covered_mask: int,
        inspected_mask: int,
        remaining: Tuple[float, ...],
    ) -> Tuple[float, float, float, int]:
        nonlocal explored_state_count
        nonlocal feasible_transition_count
        nonlocal budget_pruned_transition_count
        explored_state_count += 1
        if covered_mask == full_coverage_mask:
            return 1.0, float(len(obligation_names)), 0.0, -1

        best = (
            0.0,
            float(covered_mask.bit_count()),
            0.0,
            -1,
        )
        best_objective = (best[0], best[1], -best[2])
        for index, candidate in enumerate(candidates):
            action_bit = 1 << index
            obligation_bit = int(candidate["obligation_bit"])
            if not available_mask & action_bit or covered_mask & obligation_bit:
                continue
            transition = affordable(candidate, inspected_mask, remaining)
            if transition is None:
                budget_pruned_transition_count += 1
                continue
            feasible_transition_count += 1
            next_remaining, resource_cost = transition
            next_available = available_mask & ~action_bit
            next_inspected = inspected_mask | int(candidate.get("source_bit") or 0)
            success = solve(
                next_available,
                covered_mask | obligation_bit,
                next_inspected,
                next_remaining,
            )
            failure = solve(
                next_available,
                covered_mask,
                next_inspected,
                next_remaining,
            )
            probability = min(1.0, max(0.0, _float(candidate["probability"])))
            completion = probability * success[0] + (1.0 - probability) * failure[0]
            expected_coverage = (
                probability * success[1] + (1.0 - probability) * failure[1]
            )
            expected_cost = _resource_scalar_cost(resource_cost) + (
                probability * success[2] + (1.0 - probability) * failure[2]
            )
            objective = (completion, expected_coverage, -expected_cost)
            if objective > best_objective:
                best = (completion, expected_coverage, expected_cost, index)
                best_objective = objective
        return best

    available_mask = (1 << len(candidates)) - 1
    value = solve(
        available_mask, 0, initial_source_mask, initial_remaining
    )
    if value[3] < 0:
        return {
            "status": "stop",
            "reason": "no_budget_feasible_positive_value_action",
            "completion_probability": round(value[0], 10),
            "expected_coverage": round(value[1], 10),
            "heuristic_fallback_used": False,
            "explored_state_count": explored_state_count,
        }
    selected = candidates[value[3]]
    return {
        "status": "optimal",
        "selected_action": dict(selected["action"]),
        "selected_action_key": selected["action_key"],
        "selected_source_id": selected["source_id"],
        "selected_obligation_key": selected["obligation_key"],
        "completion_probability": round(value[0], 10),
        "expected_coverage": round(value[1], 10),
        "expected_coverage_fraction": round(
            value[1] / max(1, len(obligation_names)), 10
        ),
        "expected_physical_cost": round(value[2], 10),
        "obligations": obligation_names,
        "candidate_count": len(candidates),
        "unique_source_count": len(source_names),
        "bounded_resources": list(bounded_resources),
        "initial_remaining_budget": dict(zip(
            bounded_resources, initial_remaining
        )),
        "explored_state_count": explored_state_count,
        "feasible_transition_count": feasible_transition_count,
        "budget_pruned_transition_count": budget_pruned_transition_count,
        "solver": "exact_adaptive_bellman_dp",
        "objective_order": [
            "supported_executable_witness_probability",
            "expected_supported_obligation_coverage",
            "expected_physical_cost",
        ],
        "probability_model": "calibrated_lower_confidence_independence",
        "exact_on_finite_frontier": True,
        "heuristic_fallback_used": False,
    }


def exact_adaptive_supported_witness_first_action(
    actions: Sequence[Mapping[str, Any]],
    estimates: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    *,
    max_candidates: int = 18,
) -> Tuple[Mapping[str, Any] | None, Dict[str, Any]]:
    """Select the exact quality-first action on a pure source frontier."""
    if any(str(action.get("operator") or "") == "VERIFY" for action in actions):
        return None, {"reason": "defer_to_mandatory_verification", "stop": False}
    source_count = sum(
        str(action.get("operator") or "") == "SOURCE_INSPECT"
        for action in actions
    )
    if not source_count:
        return None, {"reason": "no_source_inspect_frontier", "stop": False}
    plan = exact_adaptive_supported_witness_plan(
        actions, estimates, state, max_candidates=max_candidates
    )
    detail = {
        "reason": "exact_adaptive_supported_witness_dp",
        "stop": plan.get("status") != "optimal",
        "plan": {
            key: value for key, value in plan.items()
            if key != "selected_action"
        },
        "exact_on_finite_frontier": bool(
            plan.get("exact_on_finite_frontier")
        ),
        "heuristic_fallback_used": bool(
            plan.get("heuristic_fallback_used")
        ),
        "receding_horizon": False,
    }
    selected = plan.get("selected_action")
    return (dict(selected) if selected else None), detail


def risk_aware_witness_search_first_action(
    actions: Sequence[Mapping[str, Any]],
    estimates: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    *,
    beam_width: int = 16,
    horizon: int = 3,
) -> Tuple[Mapping[str, Any] | None, Dict[str, Any]]:
    """Plan over unique physical sources for supported witness completion.

    The search uses calibrated lower-confidence support probabilities and
    recomputes a short source plan after every observation.  It differs from
    set-cover greedy in two ways: a source is charged once even when shared by
    obligations, and a nominal cover with weak source support can lose to a
    slightly more expensive but substantially more reliable cover.
    """
    if any(str(action.get("operator") or "") == "VERIFY" for action in actions):
        return None, {"reason": "defer_to_mandatory_verification"}
    obligations = _open_coverage_keys(state)
    groups = _source_groups(actions, estimates, state)
    if len(obligations) < 2 or len(groups) < 2:
        return None, {
            "reason": "trivial_source_frontier",
            "open_obligation_count": len(obligations),
            "unique_source_count": len(groups),
        }

    explored: List[_RiskBeamState] = []
    expanded = 0
    search_depth = max(1, min(int(horizon), len(groups)))
    search_mode = "exact_bounded_subset"
    if len(groups) <= 22 and search_depth <= 4:
        for size in range(1, search_depth + 1):
            for selected in combinations(range(len(groups)), size):
                failures = {key: 1.0 for key in obligations}
                cost = 0.0
                for index in selected:
                    group = groups[index]
                    for key, probability in group["probability_by_key"].items():
                        if key in obligations:
                            failures[key] *= 1.0 - probability
                    cost += _float(group.get("cost"), 1.0)
                explored.append(_RiskBeamState(
                    selected, tuple(sorted(failures.items())), cost
                ))
                expanded += 1
    else:
        search_mode = "beam_bounded_subset"
        initial_failures = tuple((key, 1.0) for key in sorted(obligations))
        beam = [_RiskBeamState((), initial_failures, 0.0)]
        for _ in range(search_depth):
            frontier: List[_RiskBeamState] = []
            for node in beam:
                used = set(node.selected)
                old_failures = dict(node.failure_probabilities)
                for index, group in enumerate(groups):
                    if index in used:
                        continue
                    failures = dict(old_failures)
                    for key, probability in group["probability_by_key"].items():
                        if key in obligations:
                            failures[key] = failures.get(key, 1.0) * (
                                1.0 - probability
                            )
                    frontier.append(_RiskBeamState(
                        node.selected + (index,),
                        tuple(sorted(failures.items())),
                        node.cost + _float(group.get("cost"), 1.0),
                    ))
                    expanded += 1
            if not frontier:
                break
            explored.extend(frontier)
            frontier.sort(
                key=lambda node: _risk_plan_objective(
                    obligations, dict(node.failure_probabilities), node.cost
                ),
                reverse=True,
            )
            beam = frontier[:max(1, int(beam_width))]

    if not explored:
        return None, {"reason": "empty_risk_aware_search"}
    best = max(
        explored,
        key=lambda node: _risk_plan_objective(
            obligations, dict(node.failure_probabilities), node.cost
        ),
    )
    selected_groups = [groups[index] for index in best.selected]
    alternatives_per_key = {
        key: sum(key in group["coverage_keys"] for group in groups)
        for key in obligations
    }

    def first_source_priority(group: Mapping[str, Any]) -> Tuple[float, float, str]:
        critical_gain = sum(
            _float(group["probability_by_key"].get(key))
            / max(1, alternatives_per_key.get(key, 1))
            for key in obligations
        )
        cost = max(0.05, _float(group.get("cost"), 1.0))
        return critical_gain / cost, critical_gain, str(group.get("source_id") or "")

    first_group = max(selected_groups, key=first_source_priority)

    def first_action_priority(item: Mapping[str, Any]) -> Tuple[float, float, str]:
        own_key = str(item.get("own_key") or "")
        criticality = 1.0 / max(1, alternatives_per_key.get(own_key, 1))
        probability = _float(item.get("probability"))
        estimate = dict(item.get("estimate") or {})
        witness = _float(estimate.get("expected_witness_progress"), probability)
        return probability * criticality, witness, str(
            (item.get("action") or {}).get("action_key") or ""
        )

    first_item = max(first_group["actions"], key=first_action_priority)
    metrics = _risk_plan_metrics(
        obligations, dict(best.failure_probabilities), best.cost
    )
    return dict(first_item["action"]), {
        "reason": "risk_aware_witness_search",
        "beam_width": max(1, int(beam_width)),
        "horizon": search_depth,
        "expanded_state_count": expanded,
        "search_mode": search_mode,
        "unique_source_count": len(groups),
        "open_obligation_count": len(obligations),
        "selected_source_ids": [
            str(group.get("source_id") or "") for group in selected_groups
        ],
        "selected_sequence": [
            str(group.get("source_id") or "") for group in selected_groups
        ],
        "first_source_id": str(first_group.get("source_id") or ""),
        "predicted_complete_probability": round(
            metrics["complete_probability"], 10
        ),
        "predicted_expected_coverage": round(metrics["expected_coverage"], 10),
        "predicted_cost": round(metrics["cost"], 8),
        "probability_model": "calibrated_lower_confidence_independence",
    }


def deterministic_random_source_first_action(
    actions: Sequence[Mapping[str, Any]], state: Mapping[str, Any]
) -> Tuple[Mapping[str, Any] | None, Dict[str, Any]]:
    """Stable query-keyed random source baseline without QA-gold access."""
    if any(str(action.get("operator") or "") == "VERIFY" for action in actions):
        return None, {"reason": "defer_to_mandatory_verification"}
    source_actions = [
        dict(action) for action in actions
        if str(action.get("operator") or "") == "SOURCE_INSPECT"
    ]
    if not source_actions:
        return None, {"reason": "no_source_inspect_frontier"}
    query_id = str(state.get("query_id") or "")
    selected = min(
        source_actions,
        key=lambda action: hashlib.sha256(
            f"{query_id}\x1f{action.get('action_key', '')}".encode("utf-8")
        ).hexdigest(),
    )
    return selected, {
        "reason": "deterministic_query_keyed_random_source",
        "candidate_count": len(source_actions),
        "selected_action_key": str(selected.get("action_key") or ""),
    }


def deterministic_fixed_order_first_action(
    actions: Sequence[Mapping[str, Any]], state: Mapping[str, Any]
) -> Tuple[Mapping[str, Any] | None, Dict[str, Any]]:
    """Select the first physical action under an answer-blind fixed order.

    This baseline consumes the same BSWC action frontier as the learned and
    adaptive policies, but deliberately ignores probability, gain, cost, and
    realized query feedback.  Retriever rank is used only to define a stable
    order among source alternatives.
    """
    del state
    if not actions:
        return None, {"reason": "no_executable_action"}
    operator_priority = {
        "VERIFY": 0,
        "LOOKUP_JOIN": 1,
        "SOURCE_INSPECT": 2,
        "BOUND_DENSE_PROBE": 3,
        "COMPLETE": 4,
    }

    def fixed_key(action: Mapping[str, Any]) -> Tuple[Any, ...]:
        operator = str(action.get("operator") or "")
        source_rank = max(0, int(_float(action.get("source_rank"), 0.0)))
        source_id = max(-1, int(_float(action.get("source_id"), -1.0)))
        return (
            operator_priority.get(operator, 99),
            max(-1, int(_float(action.get("operator_index"), -1.0))),
            source_rank if operator == "SOURCE_INSPECT" else 0,
            source_id if operator == "SOURCE_INSPECT" else -1,
            str(action.get("binding_id") or ""),
            str(action.get("obligation_id") or ""),
            str(action.get("action_key") or ""),
        )

    selected = min((dict(action) for action in actions), key=fixed_key)
    return selected, {
        "reason": "deterministic_fixed_physical_order",
        "candidate_count": len(actions),
        "selected_action_key": str(selected.get("action_key") or ""),
        "uses_probability": False,
        "uses_gain": False,
        "uses_cost": False,
        "uses_query_feedback": False,
    }


def risk_aware_witness_plan_exact(
    obligations: Iterable[str],
    candidates: Sequence[Mapping[str, Any]],
    *,
    max_candidates: int = 22,
    max_sources: int = 3,
) -> Dict[str, Any]:
    """Exact small-instance oracle for the risk-aware planning objective.

    This is an offline diagnostic, not the online policy.  Candidate
    probabilities must be available before source access; no realized outcome
    or QA answer is consumed.
    """
    universe = {str(value) for value in obligations if str(value)}
    rows = [dict(row) for row in candidates]
    if len(rows) > int(max_candidates):
        return {
            "status": "too_many_candidates",
            "candidate_count": len(rows),
            "max_candidates": int(max_candidates),
        }
    if not universe or not rows:
        return {"status": "infeasible", "selected": [], "cost": None}
    best: Tuple[Tuple[int, float, float, float], Tuple[int, ...], Dict[str, float]] | None = None
    limit = min(max(1, int(max_sources)), len(rows))
    for size in range(1, limit + 1):
        for selected in combinations(range(len(rows)), size):
            failures = {key: 1.0 for key in universe}
            cost = 0.0
            for index in selected:
                row = rows[index]
                probabilities = dict(row.get("probability_by_key") or {})
                fallback = min(1.0, max(0.0, _float(
                    row.get("support_probability"), 0.0
                )))
                for key in _coverage(row) & universe:
                    probability = min(
                        1.0, max(0.0, _float(probabilities.get(key), fallback))
                    )
                    failures[key] *= 1.0 - probability
                cost += max(0.05, _float(row.get("cost"), 1.0))
            objective = _risk_plan_objective(universe, failures, cost)
            candidate = (objective, selected, failures)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        return {"status": "infeasible", "selected": [], "cost": None}
    objective, selected, failures = best
    metrics = _risk_plan_metrics(universe, failures, -objective[3])
    return {
        "status": "optimal",
        "selected": [dict(rows[index]) for index in selected],
        "selected_indices": list(selected),
        "cost": metrics["cost"],
        "complete_probability": metrics["complete_probability"],
        "expected_coverage": metrics["expected_coverage"],
        "objective": list(objective),
    }


_RESOURCE_REMAINING_KEYS = {
    "actions": "actions_remaining",
    "scan_rows": "scan_rows_remaining",
    "retrieval_calls": "retrieval_calls_remaining",
    "model_calls": "model_calls_remaining",
    "tokens": "known_tokens_remaining",
}


def _candidate_resource_cost(candidate: Mapping[str, Any]) -> Dict[str, float]:
    raw = dict(candidate.get("resource_cost") or {})
    return {
        "actions": max(1.0, _float(raw.get("actions"), 1.0)),
        "scan_rows": max(0.0, _float(
            raw.get("scan_rows"), candidate.get("cost", 0.0)
        )),
        "retrieval_calls": max(0.0, _float(raw.get("retrieval_calls"))),
        "model_calls": max(0.0, _float(raw.get("model_calls"))),
        "tokens": max(0.0, _float(raw.get("tokens"))),
    }


def _add_resource_costs(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> Dict[str, float]:
    return {
        key: _float(left.get(key)) + _float(right.get(key))
        for key in _RESOURCE_REMAINING_KEYS
    }


def _budget_feasible(
    usage: Mapping[str, Any], remaining_budget: Mapping[str, Any]
) -> bool:
    for resource, remaining_key in _RESOURCE_REMAINING_KEYS.items():
        remaining = remaining_budget.get(remaining_key)


        if remaining is None:
            continue
        if _float(usage.get(resource)) > max(0.0, _float(remaining)) + 1e-12:
            return False
    return True


def _budget_pressure(
    usage: Mapping[str, Any], remaining_budget: Mapping[str, Any]
) -> float:
    fractions = []
    for resource, remaining_key in _RESOURCE_REMAINING_KEYS.items():
        used = max(0.0, _float(usage.get(resource)))
        remaining = remaining_budget.get(remaining_key)
        if remaining is None:
            continue
        capacity = max(0.0, _float(remaining))
        if capacity <= 0.0:
            if used > 0.0:
                return float("inf")
            continue
        fractions.append(used / capacity)
    return max(fractions, default=0.0)


def _adaptive_risk_utility(
    obligations: Set[str],
    failure_probabilities: Mapping[str, float],
    usage: Mapping[str, Any],
    remaining_budget: Mapping[str, Any],
    *,
    partial_value: float,
    budget_penalty: float,
) -> Dict[str, float]:
    metrics = _risk_plan_metrics(obligations, failure_probabilities, 0.0)
    obligation_count = max(1, len(obligations))
    expected_fraction = metrics["expected_coverage"] / obligation_count
    pressure = _budget_pressure(usage, remaining_budget)
    utility = (
        metrics["complete_probability"]
        + max(0.0, float(partial_value)) * expected_fraction
        - max(0.0, float(budget_penalty)) * pressure
    )
    return {
        "utility": utility,
        "complete_probability": metrics["complete_probability"],
        "expected_coverage": metrics["expected_coverage"],
        "expected_coverage_fraction": expected_fraction,
        "budget_pressure": pressure,
    }


def adaptive_risk_bounded_witness_plan_exact(
    obligations: Iterable[str],
    candidates: Sequence[Mapping[str, Any]],
    *,
    remaining_budget: Mapping[str, Any],
    max_candidates: int = 22,
    max_sources: int | None = None,
    partial_value: float = 0.10,
    budget_penalty: float = 0.05,
    min_utility: float = 0.0,
) -> Dict[str, Any]:
    """Exact small-frontier oracle with a budget-feasible STOP alternative.

    The empty plan has utility zero.  Every non-empty subset must satisfy the
    executor's *remaining* multidimensional budget.  The objective values the
    probability of a fully supported witness, gives limited credit to partial
    support, and charges the tightest consumed budget dimension.  It never
    rewards nominal structural coverage before evidence support probability.

    This routine is exponential and intentionally restricted to frozen small
    frontiers.  The online policy uses it only inside a bounded horizon.
    """
    universe = {str(value) for value in obligations if str(value)}
    rows = [dict(row) for row in candidates]
    if len(rows) > int(max_candidates):
        return {
            "status": "too_many_candidates",
            "candidate_count": len(rows),
            "max_candidates": int(max_candidates),
        }
    if not universe or not rows:
        return {
            "status": "stop",
            "selected": [],
            "utility": 0.0,
            "stop_reason": "empty_frontier",
        }

    limit = len(rows) if max_sources is None else min(
        max(0, int(max_sources)), len(rows)
    )
    best: Tuple[
        Tuple[float, float, float, float, float],
        Tuple[int, ...],
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
    ] | None = None
    feasible_plan_count = 0
    budget_pruned_count = 0
    for size in range(1, limit + 1):
        for selected in combinations(range(len(rows)), size):
            failures = {key: 1.0 for key in universe}
            usage = {key: 0.0 for key in _RESOURCE_REMAINING_KEYS}
            scalar_cost = 0.0
            for index in selected:
                row = rows[index]
                probabilities = dict(row.get("probability_by_key") or {})
                fallback = min(1.0, max(0.0, _float(
                    row.get("support_probability"), 0.0
                )))
                for key in _coverage(row) & universe:
                    probability = min(
                        1.0, max(0.0, _float(probabilities.get(key), fallback))
                    )
                    failures[key] *= 1.0 - probability
                usage = _add_resource_costs(
                    usage, _candidate_resource_cost(row)
                )
                scalar_cost += max(0.05, _float(row.get("cost"), 1.0))
            if not _budget_feasible(usage, remaining_budget):
                budget_pruned_count += 1
                continue
            feasible_plan_count += 1
            metrics = _adaptive_risk_utility(
                universe,
                failures,
                usage,
                remaining_budget,
                partial_value=partial_value,
                budget_penalty=budget_penalty,
            )
            objective = (
                metrics["utility"],
                metrics["complete_probability"],
                metrics["expected_coverage_fraction"],
                -metrics["budget_pressure"],
                -scalar_cost,
            )
            candidate = (objective, selected, failures, usage, metrics)
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None or best[0][0] <= float(min_utility):
        return {
            "status": "stop",
            "selected": [],
            "utility": 0.0,
            "stop_reason": (
                "no_budget_feasible_plan" if best is None
                else "best_plan_not_better_than_stop"
            ),
            "best_nonstop_utility": (
                None if best is None else round(best[0][0], 10)
            ),
            "feasible_plan_count": feasible_plan_count,
            "budget_pruned_count": budget_pruned_count,
        }

    objective, selected, failures, usage, metrics = best
    return {
        "status": "optimal",
        "selected": [dict(rows[index]) for index in selected],
        "selected_indices": list(selected),
        "utility": round(metrics["utility"], 10),
        "complete_probability": round(metrics["complete_probability"], 10),
        "expected_coverage": round(metrics["expected_coverage"], 10),
        "expected_coverage_fraction": round(
            metrics["expected_coverage_fraction"], 10
        ),
        "budget_pressure": round(metrics["budget_pressure"], 10),
        "resource_usage": {
            key: round(value, 8) for key, value in usage.items()
        },
        "remaining_budget": dict(remaining_budget),
        "objective": list(objective),
        "feasible_plan_count": feasible_plan_count,
        "budget_pruned_count": budget_pruned_count,
    }


def adaptive_risk_bounded_witness_first_action(
    actions: Sequence[Mapping[str, Any]],
    estimates: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    *,
    beam_width: int = 16,
    horizon: int = 4,
    partial_value: float = 0.10,
    budget_penalty: float = 0.05,
    min_utility: float = 0.0,
    min_exploration_actions: int = 1,
) -> Tuple[Mapping[str, Any] | None, Dict[str, Any]]:
    """Choose one source action by adaptive risk-bounded witness cover.

    The executor calls this function again after every observed action, so the
    selected plan is a receding-horizon plan rather than a committed sequence.
    """
    if any(str(action.get("operator") or "") == "VERIFY" for action in actions):
        return None, {"reason": "defer_to_mandatory_verification", "stop": False}
    obligations = _open_coverage_keys(state)
    groups = _source_groups(actions, estimates, state)
    if len(obligations) < 2 or len(groups) < 2:
        return None, {
            "reason": "trivial_source_frontier",
            "stop": False,
            "open_obligation_count": len(obligations),
            "unique_source_count": len(groups),
        }

    remaining_budget = dict(state.get("bswc_remaining_budget") or {})
    candidates = []
    for group in groups:
        candidates.append({
            **dict(group),
            "name": str(group.get("source_id") or ""),
            "predicted_coverage_keys": sorted(group.get("coverage_keys") or []),
            "support_probability": min(
                (_float(value) for value in group["probability_by_key"].values()),
                default=0.0,
            ),
        })



    pruned_candidate_count = 0
    if len(candidates) > 22:
        def preaccess_priority(row: Mapping[str, Any]) -> Tuple[float, float, str]:
            probabilities = dict(row.get("probability_by_key") or {})
            gain = sum(
                _float(probabilities.get(key))
                for key in obligations & _coverage(row)
            )
            cost = max(0.05, _float(row.get("cost"), 1.0))
            return gain / cost, gain, str(row.get("source_id") or "")

        candidates.sort(key=preaccess_priority, reverse=True)
        pruned_candidate_count = len(candidates) - 22
        candidates = candidates[:22]

    search_depth = max(1, min(int(horizon), len(candidates)))
    plan = adaptive_risk_bounded_witness_plan_exact(
        obligations,
        candidates,
        remaining_budget=remaining_budget,
        max_candidates=22,
        max_sources=search_depth,
        partial_value=partial_value,
        budget_penalty=budget_penalty,
        min_utility=min_utility,
    )
    forced_exploration = False
    if (
        plan.get("status") == "stop"
        and int(_float(state.get("bswc_source_inspection_count")))
        < max(0, int(min_exploration_actions))
    ):


        exploratory = adaptive_risk_bounded_witness_plan_exact(
            obligations,
            candidates,
            remaining_budget=remaining_budget,
            max_candidates=22,
            max_sources=search_depth,
            partial_value=partial_value,
            budget_penalty=budget_penalty,
            min_utility=float("-inf"),
        )
        if exploratory.get("status") == "optimal":
            plan = exploratory
            forced_exploration = True
    common_detail = {
        "reason": "adaptive_risk_bounded_witness_cover",
        "stop": plan.get("status") == "stop",
        "open_obligation_count": len(obligations),
        "unique_source_count": len(groups),
        "candidate_source_count": len(candidates),
        "pruned_candidate_count": pruned_candidate_count,
        "beam_width": max(1, int(beam_width)),
        "horizon": search_depth,
        "remaining_budget": remaining_budget,
        "partial_value": float(partial_value),
        "budget_penalty": float(budget_penalty),
        "min_utility": float(min_utility),
        "min_exploration_actions": max(0, int(min_exploration_actions)),
        "source_inspection_count": int(_float(
            state.get("bswc_source_inspection_count")
        )),
        "forced_exploration": forced_exploration,
        "plan": {
            key: value for key, value in plan.items()
            if key not in {"selected"}
        },
    }
    if plan.get("status") != "optimal" or not plan.get("selected"):
        common_detail["stop_reason"] = str(
            plan.get("stop_reason") or plan.get("status") or "unknown"
        )
        return None, common_detail

    selected_groups = [dict(row) for row in plan["selected"]]
    alternatives_per_key = {
        key: sum(key in group["coverage_keys"] for group in groups)
        for key in obligations
    }

    def first_source_priority(group: Mapping[str, Any]) -> Tuple[float, float, str]:
        critical_gain = sum(
            _float((group.get("probability_by_key") or {}).get(key))
            / max(1, alternatives_per_key.get(key, 1))
            for key in obligations
        )
        pressure = _budget_pressure(
            _candidate_resource_cost(group), remaining_budget
        )
        return (
            critical_gain / max(0.01, pressure),
            critical_gain,
            str(group.get("source_id") or ""),
        )

    first_group = max(selected_groups, key=first_source_priority)

    def first_action_priority(item: Mapping[str, Any]) -> Tuple[float, float, str]:
        own_key = str(item.get("own_key") or "")
        criticality = 1.0 / max(1, alternatives_per_key.get(own_key, 1))
        probability = _float(item.get("probability"))
        witness = _float(
            (item.get("estimate") or {}).get("expected_witness_progress"),
            probability,
        )
        return probability * criticality, witness, str(
            (item.get("action") or {}).get("action_key") or ""
        )

    first_item = max(first_group["actions"], key=first_action_priority)
    common_detail.update({
        "selected_source_ids": [
            str(group.get("source_id") or "") for group in selected_groups
        ],
        "first_source_id": str(first_group.get("source_id") or ""),
        "predicted_utility": plan.get("utility"),
        "predicted_complete_probability": plan.get("complete_probability"),
        "predicted_expected_coverage": plan.get("expected_coverage"),
        "predicted_budget_pressure": plan.get("budget_pressure"),
        "predicted_resource_usage": plan.get("resource_usage"),
        "probability_model": "calibrated_lower_confidence_independence",
        "receding_horizon": True,
    })
    return dict(first_item["action"]), common_detail


def feasibility_preserving_witness_plan_exact(
    obligations: Iterable[str],
    candidates: Sequence[Mapping[str, Any]],
    *,
    remaining_budget: Mapping[str, Any],
    max_candidates: int = 22,
    max_sources: int | None = None,
) -> Dict[str, Any]:
    """Choose a hard-budget-feasible plan without pricing feasibility twice.

    The old ARWC objective subtracts budget pressure from a multiplicative
    witness probability.  On sparse text-derived relations this can make every
    non-empty plan look worse than STOP even when a feasible witness plan
    exists.  This planner treats the resource ledger as a hard constraint and
    compares feasible plans lexicographically: complete-witness probability,
    expected obligation coverage, budget pressure, and scalar cost.

    The routine is exact only on the retained small frontier and bounded
    horizon.  It consumes calibrated pre-access probabilities, never realized
    source outcomes or QA answers.
    """
    universe = {str(value) for value in obligations if str(value)}
    rows = [dict(row) for row in candidates]
    if len(rows) > int(max_candidates):
        return {
            "status": "too_many_candidates",
            "candidate_count": len(rows),
            "max_candidates": int(max_candidates),
        }
    if not universe or not rows:
        return {
            "status": "infeasible",
            "selected": [],
            "stop_reason": "empty_frontier",
        }

    limit = len(rows) if max_sources is None else min(
        max(0, int(max_sources)), len(rows)
    )
    best: Tuple[
        Tuple[float, float, float, float, float],
        Tuple[int, ...],
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
    ] | None = None
    feasible_plan_count = 0
    budget_pruned_count = 0
    for size in range(1, limit + 1):
        for selected in combinations(range(len(rows)), size):
            failures = {key: 1.0 for key in universe}
            usage = {key: 0.0 for key in _RESOURCE_REMAINING_KEYS}
            scalar_cost = 0.0
            for index in selected:
                row = rows[index]
                probabilities = dict(row.get("probability_by_key") or {})
                fallback = min(1.0, max(0.0, _float(
                    row.get("support_probability"), 0.0
                )))
                for key in _coverage(row) & universe:
                    probability = min(
                        1.0, max(0.0, _float(probabilities.get(key), fallback))
                    )
                    failures[key] *= 1.0 - probability
                usage = _add_resource_costs(
                    usage, _candidate_resource_cost(row)
                )
                scalar_cost += max(0.05, _float(row.get("cost"), 1.0))
            if not _budget_feasible(usage, remaining_budget):
                budget_pruned_count += 1
                continue
            feasible_plan_count += 1
            metrics = _risk_plan_metrics(universe, failures, scalar_cost)
            expected_fraction = metrics["expected_coverage"] / max(
                1, len(universe)
            )
            pressure = _budget_pressure(usage, remaining_budget)
            objective = (
                metrics["complete_probability"],
                expected_fraction,
                -pressure,
                -scalar_cost,
                -float(size),
            )
            candidate = (objective, selected, failures, usage, metrics)
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        return {
            "status": "infeasible",
            "selected": [],
            "stop_reason": "no_budget_feasible_plan",
            "feasible_plan_count": feasible_plan_count,
            "budget_pruned_count": budget_pruned_count,
        }

    objective, selected, failures, usage, metrics = best
    return {
        "status": "optimal",
        "selected": [dict(rows[index]) for index in selected],
        "selected_indices": list(selected),
        "objective": list(objective),
        "complete_probability": round(metrics["complete_probability"], 10),
        "expected_coverage": round(metrics["expected_coverage"], 10),
        "expected_coverage_fraction": round(
            metrics["expected_coverage"] / max(1, len(universe)), 10
        ),
        "budget_pressure": round(_budget_pressure(
            usage, remaining_budget
        ), 10),
        "scalar_cost": round(metrics["cost"], 8),
        "resource_usage": {
            key: round(value, 8) for key, value in usage.items()
        },
        "feasible_plan_count": feasible_plan_count,
        "budget_pruned_count": budget_pruned_count,
    }


@dataclass(frozen=True)
class _ParetoPlanState:
    selected: Tuple[int, ...]
    failures: Tuple[float, ...]
    usage: Tuple[float, ...]
    scalar_cost: float


def _pareto_state_dominates(
    left: _ParetoPlanState, right: _ParetoPlanState
) -> bool:
    """Return whether left is never worse for any future source suffix."""
    weakly_better = (
        all(a <= b + 1e-12 for a, b in zip(left.failures, right.failures))
        and all(a <= b + 1e-12 for a, b in zip(left.usage, right.usage))
        and left.scalar_cost <= right.scalar_cost + 1e-12
        and len(left.selected) <= len(right.selected)
    )
    strictly_better = (
        any(a < b - 1e-12 for a, b in zip(left.failures, right.failures))
        or any(a < b - 1e-12 for a, b in zip(left.usage, right.usage))
        or left.scalar_cost < right.scalar_cost - 1e-12
        or len(left.selected) < len(right.selected)
    )
    return weakly_better and strictly_better


def feasibility_preserving_witness_plan_pareto(
    obligations: Iterable[str],
    candidates: Sequence[Mapping[str, Any]],
    *,
    remaining_budget: Mapping[str, Any],
    max_candidates: int = 22,
    max_sources: int | None = None,
) -> Dict[str, Any]:
    """Solve a bounded stochastic cover frontier with exact Pareto pruning.

    Weighted Set Cover is recovered when support probabilities are one and the
    only resource is scalar source cost.  The general online problem additionally
    tracks per-obligation failure probabilities and a multi-resource hard budget.
    This dynamic program remains exponential in the worst case, but dominance
    pruning is exact: a discarded state cannot outperform its dominator after any
    common suffix of source choices.  Production execution applies it only to the
    retained query-local frontier and bounded source horizon.
    """
    universe = tuple(sorted({str(value) for value in obligations if str(value)}))
    rows = [dict(row) for row in candidates]
    if len(rows) > int(max_candidates):
        return {
            "status": "too_many_candidates",
            "candidate_count": len(rows),
            "max_candidates": int(max_candidates),
            "solver": "pareto_dynamic_program",
        }
    if not universe or not rows:
        return {
            "status": "infeasible",
            "selected": [],
            "stop_reason": "empty_frontier",
            "solver": "pareto_dynamic_program",
        }

    resource_names = tuple(_RESOURCE_REMAINING_KEYS)
    limit = len(rows) if max_sources is None else min(
        max(0, int(max_sources)), len(rows)
    )
    empty = _ParetoPlanState(
        selected=(),
        failures=tuple(1.0 for _ in universe),
        usage=tuple(0.0 for _ in resource_names),
        scalar_cost=0.0,
    )
    frontier = [empty]
    expanded_state_count = 0
    budget_pruned_count = 0
    dominance_pruned_count = 0
    peak_frontier_size = 1

    for index, row in enumerate(rows):
        additions: List[_ParetoPlanState] = []
        probabilities = dict(row.get("probability_by_key") or {})
        fallback = min(1.0, max(0.0, _float(
            row.get("support_probability"), 0.0
        )))
        coverage = _coverage(row) & set(universe)
        row_resources = _candidate_resource_cost(row)
        row_cost = max(0.05, _float(row.get("cost"), 1.0))
        for current in frontier:
            if len(current.selected) >= limit:
                continue
            expanded_state_count += 1
            usage = tuple(
                current.usage[position] + row_resources[name]
                for position, name in enumerate(resource_names)
            )
            usage_map = dict(zip(resource_names, usage))
            if not _budget_feasible(usage_map, remaining_budget):
                budget_pruned_count += 1
                continue
            failures = list(current.failures)
            for position, key in enumerate(universe):
                if key not in coverage:
                    continue
                probability = min(
                    1.0, max(0.0, _float(probabilities.get(key), fallback))
                )
                failures[position] *= 1.0 - probability
            additions.append(_ParetoPlanState(
                selected=current.selected + (index,),
                failures=tuple(failures),
                usage=usage,
                scalar_cost=current.scalar_cost + row_cost,
            ))

        retained: List[_ParetoPlanState] = []
        for candidate in frontier + additions:
            if any(_pareto_state_dominates(row, candidate) for row in retained):
                dominance_pruned_count += 1
                continue
            survivors = []
            for row in retained:
                if _pareto_state_dominates(candidate, row):
                    dominance_pruned_count += 1
                else:
                    survivors.append(row)
            survivors.append(candidate)
            retained = survivors
        frontier = retained
        peak_frontier_size = max(peak_frontier_size, len(frontier))

    feasible = [row for row in frontier if row.selected]
    if not feasible:
        return {
            "status": "infeasible",
            "selected": [],
            "stop_reason": "no_budget_feasible_plan",
            "solver": "pareto_dynamic_program",
            "expanded_state_count": expanded_state_count,
            "retained_state_count": len(frontier),
            "peak_frontier_size": peak_frontier_size,
            "budget_pruned_count": budget_pruned_count,
            "dominance_pruned_count": dominance_pruned_count,
        }

    def objective(row: _ParetoPlanState) -> Tuple[float, float, float, float, float]:
        failure_map = dict(zip(universe, row.failures))
        usage_map = dict(zip(resource_names, row.usage))
        metrics = _risk_plan_metrics(set(universe), failure_map, row.scalar_cost)
        return (
            metrics["complete_probability"],
            metrics["expected_coverage"] / max(1, len(universe)),
            -_budget_pressure(usage_map, remaining_budget),
            -row.scalar_cost,
            -float(len(row.selected)),
        )

    best = max(feasible, key=lambda row: (objective(row), row.selected))
    failure_map = dict(zip(universe, best.failures))
    usage_map = dict(zip(resource_names, best.usage))
    metrics = _risk_plan_metrics(set(universe), failure_map, best.scalar_cost)
    return {
        "status": "optimal",
        "selected": [dict(rows[index]) for index in best.selected],
        "selected_indices": list(best.selected),
        "objective": list(objective(best)),
        "complete_probability": round(metrics["complete_probability"], 10),
        "expected_coverage": round(metrics["expected_coverage"], 10),
        "expected_coverage_fraction": round(
            metrics["expected_coverage"] / max(1, len(universe)), 10
        ),
        "budget_pressure": round(
            _budget_pressure(usage_map, remaining_budget), 10
        ),
        "scalar_cost": round(best.scalar_cost, 8),
        "resource_usage": {
            key: round(value, 8) for key, value in usage_map.items()
        },
        "solver": "pareto_dynamic_program",
        "exact_on_retained_frontier": True,
        "expanded_state_count": expanded_state_count,
        "retained_state_count": len(frontier),
        "peak_frontier_size": peak_frontier_size,
        "budget_pruned_count": budget_pruned_count,
        "dominance_pruned_count": dominance_pruned_count,
    }


def optimistic_witness_completion_upper_bound(
    obligations: Iterable[str],
    candidates: Sequence[Mapping[str, Any]],
    *,
    remaining_budget: Mapping[str, Any],
) -> float:
    """Return an optimistic pre-access completion bound for safe stopping.

    Each individually affordable source contributes to every compatible open
    obligation, even when all such sources cannot jointly satisfy the budget.  The
    resulting independent-support product is therefore deliberately
    optimistic under the planner's probability model.  A low bound can justify
    STOP; a high bound cannot guarantee success.
    """
    universe = {str(value) for value in obligations if str(value)}
    if not universe:
        return 1.0
    failures = {key: 1.0 for key in universe}
    for row in candidates:
        if not _budget_feasible(
            _candidate_resource_cost(row), remaining_budget
        ):
            continue
        probabilities = dict(row.get("probability_by_key") or {})
        fallback = min(1.0, max(0.0, _float(
            row.get("support_probability"), 0.0
        )))
        for key in _coverage(row) & universe:
            probability = min(
                1.0, max(0.0, _float(probabilities.get(key), fallback))
            )
            failures[key] *= 1.0 - probability
    bound = 1.0
    for key in universe:
        bound *= 1.0 - failures[key]
    return min(1.0, max(0.0, bound))


def feasibility_preserving_witness_first_action(
    actions: Sequence[Mapping[str, Any]],
    estimates: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    *,
    beam_width: int = 16,
    horizon: int = 4,
    stop_upper_bound: float = 0.0,
    min_exploration_actions: int = 1,
) -> Tuple[Mapping[str, Any] | None, Dict[str, Any]]:
    """Select one source using feasibility-preserving receding-horizon cover."""
    if any(str(action.get("operator") or "") == "VERIFY" for action in actions):
        return None, {"reason": "defer_to_mandatory_verification", "stop": False}
    obligations = _open_coverage_keys(state)
    groups = _source_groups(actions, estimates, state)
    if len(obligations) < 2 or len(groups) < 2:
        return None, {
            "reason": "trivial_source_frontier",
            "stop": False,
            "open_obligation_count": len(obligations),
            "unique_source_count": len(groups),
        }

    remaining_budget = dict(state.get("bswc_remaining_budget") or {})
    candidates = [{
        **dict(group),
        "name": str(group.get("source_id") or ""),
        "predicted_coverage_keys": sorted(group.get("coverage_keys") or []),
        "support_probability": min(
            (_float(value) for value in group["probability_by_key"].values()),
            default=0.0,
        ),
    } for group in groups]

    pruned_candidate_count = 0
    if len(candidates) > 22:
        def preaccess_priority(row: Mapping[str, Any]) -> Tuple[float, float, str]:
            gain = sum(
                _float((row.get("probability_by_key") or {}).get(key))
                for key in obligations & _coverage(row)
            )
            cost = max(0.05, _float(row.get("cost"), 1.0))
            return gain / cost, gain, str(row.get("source_id") or "")

        candidates.sort(key=preaccess_priority, reverse=True)
        pruned_candidate_count = len(candidates) - 22
        candidates = candidates[:22]

    upper_bound = optimistic_witness_completion_upper_bound(
        obligations, candidates, remaining_budget=remaining_budget
    )
    inspection_count = int(_float(state.get("bswc_source_inspection_count")))
    threshold = min(1.0, max(0.0, float(stop_upper_bound)))
    if (
        inspection_count >= max(0, int(min_exploration_actions))
        and upper_bound <= threshold
    ):
        return None, {
            "reason": "feasibility_preserving_adaptive_witness_cover",
            "stop": True,
            "stop_reason": "reachable_probability_upper_bound_low",
            "reachable_completion_upper_bound": round(upper_bound, 10),
            "stop_upper_bound": threshold,
            "open_obligation_count": len(obligations),
            "unique_source_count": len(groups),
        }

    search_depth = max(1, min(int(horizon), len(candidates)))
    plan = feasibility_preserving_witness_plan_exact(
        obligations,
        candidates,
        remaining_budget=remaining_budget,
        max_candidates=22,
        max_sources=search_depth,
    )
    detail = {
        "reason": "feasibility_preserving_adaptive_witness_cover",
        "stop": plan.get("status") != "optimal",
        "open_obligation_count": len(obligations),
        "unique_source_count": len(groups),
        "candidate_source_count": len(candidates),
        "pruned_candidate_count": pruned_candidate_count,
        "beam_width": max(1, int(beam_width)),
        "horizon": search_depth,
        "remaining_budget": remaining_budget,
        "reachable_completion_upper_bound": round(upper_bound, 10),
        "stop_upper_bound": threshold,
        "min_exploration_actions": max(0, int(min_exploration_actions)),
        "source_inspection_count": inspection_count,
        "plan": {
            key: value for key, value in plan.items() if key != "selected"
        },
    }
    if plan.get("status") != "optimal" or not plan.get("selected"):
        detail["stop_reason"] = str(
            plan.get("stop_reason") or plan.get("status") or "unknown"
        )
        return None, detail

    selected_groups = [dict(row) for row in plan["selected"]]
    alternatives_per_key = {
        key: sum(key in group["coverage_keys"] for group in groups)
        for key in obligations
    }

    def first_source_priority(group: Mapping[str, Any]) -> Tuple[float, float, str]:
        critical_gain = sum(
            _float((group.get("probability_by_key") or {}).get(key))
            / max(1, alternatives_per_key.get(key, 1))
            for key in obligations
        )
        pressure = _budget_pressure(
            _candidate_resource_cost(group), remaining_budget
        )
        return (
            critical_gain / max(0.01, pressure),
            critical_gain,
            str(group.get("source_id") or ""),
        )

    first_group = max(selected_groups, key=first_source_priority)

    def first_action_priority(item: Mapping[str, Any]) -> Tuple[float, float, str]:
        own_key = str(item.get("own_key") or "")
        criticality = 1.0 / max(1, alternatives_per_key.get(own_key, 1))
        probability = _float(item.get("probability"))
        witness = _float(
            (item.get("estimate") or {}).get("expected_witness_progress"),
            probability,
        )
        return probability * criticality, witness, str(
            (item.get("action") or {}).get("action_key") or ""
        )

    first_item = max(first_group["actions"], key=first_action_priority)
    detail.update({
        "selected_source_ids": [
            str(group.get("source_id") or "") for group in selected_groups
        ],
        "first_source_id": str(first_group.get("source_id") or ""),
        "predicted_complete_probability": plan.get("complete_probability"),
        "predicted_expected_coverage": plan.get("expected_coverage"),
        "predicted_budget_pressure": plan.get("budget_pressure"),
        "predicted_resource_usage": plan.get("resource_usage"),
        "probability_model": "calibrated_preaccess_independence",
        "objective_order": [
            "complete_probability",
            "expected_coverage",
            "budget_pressure",
            "scalar_cost",
        ],
        "receding_horizon": True,
    })
    return dict(first_item["action"]), detail


def continuation_safe_witness_first_action(
    actions: Sequence[Mapping[str, Any]],
    estimates: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    *,
    beam_width: int = 16,
    horizon: int = 4,
    stop_upper_bound: float = 0.0,
    min_exploration_actions: int = 1,
    force_feasible_source: bool = False,
    fallback_scoring_policy: str | None = None,
) -> Tuple[Mapping[str, Any] | None, Dict[str, Any]]:
    """Prefer proven index work and reserve actions needed to finish a witness.

    Source-cover optimization alone can spend every remaining action inspecting
    passages.  A successful inspection is not a query result: the recovered
    tuple must still be verified and any downstream logical operators must run.
    This policy therefore applies two physical invariants before invoking the
    existing source-cover solver:

    1. A zero-call LOOKUP_JOIN with known positive local fanout dominates an
       external source inspection for the same live query state.
    2. Source acquisition may consume only the action slack left after a lower
       bound for verification and the remaining logical chain is reserved.

    The reserve is intentionally a lower bound.  It assumes one surviving path
    per required root and charges one LOOKUP_JOIN plus one VERIFY for each
    downstream relation.  It does not inspect passage text or QA answers.
    """
    indexed_actions = list(enumerate(actions))

    def estimate_for(action: Mapping[str, Any]) -> Dict[str, Any]:
        return dict(estimates.get(str(action.get("action_key") or "")) or {})

    verify_actions = [
        (index, dict(action))
        for index, action in indexed_actions
        if str(action.get("operator") or "") == "VERIFY"
    ]
    if verify_actions:
        def verify_priority(
            item: Tuple[int, Mapping[str, Any]]
        ) -> Tuple[float, float, float, int]:
            index, action = item
            estimate = estimate_for(action)
            return (
                _float(estimate.get("support_probability")),
                _float(estimate.get("expected_progress")),
                -normalized_physical_cost(estimate),
                -index,
            )

        _, selected = max(verify_actions, key=verify_priority)
        return selected, {
            "reason": "continuation_safe_witness_cover",
            "stop": False,
            "phase": "mandatory_verification",
            "selected_prerequisite_action_key": str(
                selected.get("action_key") or ""
            ),
            "receding_horizon": True,
        }

    positive_lookup_actions = []
    for index, action in indexed_actions:
        if str(action.get("operator") or "") != "LOOKUP_JOIN":
            continue
        estimate = estimate_for(action)
        fanout = max(0, int(_float(estimate.get("index_fanout_rows"))))
        known_nonempty = bool(estimate.get("index_has_candidate_rows")) or fanout > 0
        zero_call = (
            _float(estimate.get("hard_retrieval_calls")) <= 0.0
            and _float(estimate.get("hard_model_calls")) <= 0.0
        )
        if known_nonempty and zero_call:
            positive_lookup_actions.append((index, dict(action), estimate, fanout))
    if positive_lookup_actions:
        def lookup_priority(
            item: Tuple[int, Mapping[str, Any], Mapping[str, Any], int]
        ) -> Tuple[float, float, float, float, int]:
            index, _, estimate, fanout = item
            return (
                -float(fanout),
                _float(estimate.get("expected_witness_progress")),
                _float(estimate.get("expected_progress")),
                _float(estimate.get("unlock_value")),
                -index,
            )

        _, selected, selected_estimate, selected_fanout = max(
            positive_lookup_actions, key=lookup_priority
        )
        return selected, {
            "reason": "continuation_safe_witness_cover",
            "stop": False,
            "phase": "index_dominant_lookup",
            "selected_prerequisite_action_key": str(
                selected.get("action_key") or ""
            ),
            "selected_local_index_fanout": selected_fanout,
            "selected_expected_progress": _float(
                selected_estimate.get("expected_progress")
            ),
            "dominance_rule": "known_nonempty_zero_call_index_before_source",
            "receding_horizon": True,
        }

    source_actions = [
        dict(action) for action in actions
        if str(action.get("operator") or "") == "SOURCE_INSPECT"
    ]
    if not source_actions:
        selected, detail = precedence_aware_witness_first_action(
            actions,
            estimates,
            state,
            beam_width=beam_width,
            horizon=horizon,
            stop_upper_bound=stop_upper_bound,
            min_exploration_actions=min_exploration_actions,
        )
        detail = dict(detail)
        detail.setdefault("reason", "continuation_safe_witness_cover")
        detail["continuation_safe_fallback"] = True
        return selected, detail

    plan = dict(state.get("logical_plan") or {})
    operator_count = len(plan.get("operators") or [])
    active_bindings = [
        dict(row) for row in state.get("bindings") or []
        if row.get("active", True) and not row.get("invalidated")
    ]
    bindings_by_id = {
        str(row.get("binding_id") or ""): row
        for row in active_bindings
        if str(row.get("binding_id") or "")
    }
    required_roots = [
        str(value) for value in state.get("root_binding_ids") or [] if str(value)
    ]
    if not required_roots:
        required_roots = sorted({
            str(row.get("root_binding_id") or row.get("binding_id") or "")
            for row in active_bindings
            if str(row.get("root_binding_id") or row.get("binding_id") or "")
        })

    def continuation_action_reserve(action: Mapping[str, Any]) -> int:
        binding_id = str(action.get("binding_id") or "")
        target_binding = bindings_by_id.get(binding_id) or {}
        target_root = str(
            target_binding.get("root_binding_id") or binding_id
        )
        target_index = max(0, int(_float(action.get("operator_index"), 0)))
        reserve = 0
        for root_id in required_roots or [target_root]:
            existing_remaining = [
                max(
                    0,
                    2 * (
                        operator_count
                        - max(0, int(_float(row.get("next_operator_index"), 0)))
                    ),
                )
                for row in active_bindings
                if str(row.get("root_binding_id") or row.get("binding_id") or "")
                == root_id
            ]
            if root_id == target_root:


                source_path_remaining = 1 + 2 * max(
                    0, operator_count - target_index - 1
                )
                existing_remaining.append(source_path_remaining)
            if existing_remaining:
                reserve += min(existing_remaining)
        return max(1, reserve)

    reserve_by_action = {
        str(action.get("action_key") or ""): continuation_action_reserve(action)
        for action in source_actions
    }
    continuation_reserve = min(reserve_by_action.values(), default=1)
    remaining_budget = dict(state.get("bswc_remaining_budget") or {})
    actions_remaining = max(
        0, int(_float(remaining_budget.get("actions_remaining"), 0))
    )
    source_budget = dict(remaining_budget)
    source_budget["actions_remaining"] = max(
        0, actions_remaining - continuation_reserve
    )

    safe_source_actions = []
    for action in source_actions:
        estimate = estimate_for(action)
        action_cost = {
            "actions": 1.0,
            "scan_rows": max(0.0, _float(
                estimate.get("hard_scan_rows"),
                _float(estimate.get("expected_scan_rows")),
            )),
            "retrieval_calls": max(0.0, _float(
                estimate.get("hard_retrieval_calls"),
                _float(estimate.get("expected_retrieval_calls")),
            )),
            "model_calls": max(0.0, _float(
                estimate.get("hard_model_calls"),
                _float(estimate.get("expected_model_calls")),
            )),
            "tokens": max(0.0, _float(estimate.get("expected_tokens"))),
        }
        if _budget_feasible(action_cost, source_budget):
            safe_source_actions.append(action)

    reserve_detail = {
        "continuation_action_reserve": continuation_reserve,
        "continuation_reserve_by_action": reserve_by_action,
        "actions_remaining_before_source": actions_remaining,
        "source_action_slack": source_budget["actions_remaining"],
        "required_root_count": len(required_roots),
        "operator_count": operator_count,
        "safe_source_action_count": len(safe_source_actions),
        "source_action_count": len(source_actions),
        "reserve_model": "one_path_per_required_root",
    }
    if not safe_source_actions:
        return None, {
            "reason": "continuation_safe_witness_cover",
            "stop": True,
            "stop_reason": "continuation_reserve_would_be_violated",
            "phase": "continuation_reserve_stop",
            **reserve_detail,
        }

    planning_state = dict(state)
    planning_state["bswc_remaining_budget"] = source_budget
    non_source_actions = [
        dict(action) for action in actions
        if str(action.get("operator") or "") != "SOURCE_INSPECT"
    ]
    planning_actions = non_source_actions + safe_source_actions
    selected, detail = feasibility_preserving_witness_first_action(
        planning_actions,
        estimates,
        planning_state,
        beam_width=beam_width,
        horizon=horizon,
        stop_upper_bound=stop_upper_bound,
        min_exploration_actions=min_exploration_actions,
    )
    detail = dict(detail)
    if (
        fallback_scoring_policy
        and selected is not None
        and str(selected.get("operator") or "") == "SOURCE_INSPECT"
        and _float((detail.get("plan") or {}).get("complete_probability")) <= 0.0
    ):
        selected_binding = str(selected.get("binding_id") or "")
        selected_obligation = str(selected.get("obligation_id") or "")
        peer_sources = [
            action for action in safe_source_actions
            if str(action.get("binding_id") or "") == selected_binding
            and str(action.get("obligation_id") or "") == selected_obligation
        ]
        ranked_peers = []
        for position, action in enumerate(peer_sources):
            score, components = source_action_score(
                fallback_scoring_policy,
                action,
                estimate_for(action),
                state,
            )
            ranked_peers.append((score, -position, action, components))
        if ranked_peers:
            score, _, replacement, components = max(
                ranked_peers, key=lambda item: (item[0], item[1])
            )
            original_action_key = str(selected.get("action_key") or "")
            selected = replacement
            detail.update({
                "phase": "rank_consistent_incomplete_cover",
                "fallback_rank_consistent": True,
                "fallback_scoring_policy": fallback_scoring_policy,
                "fallback_original_action_key": original_action_key,
                "fallback_selected_score": score,
                "fallback_selected_components": components,
                "fallback_peer_source_count": len(peer_sources),
            })
    if selected is None and (
        not detail.get("stop") or force_feasible_source
    ):
        def source_priority(
            action: Mapping[str, Any],
        ) -> Tuple[float, float, float, str]:
            estimate = estimate_for(action)
            return (
                _float(estimate.get("support_probability")),
                _float(estimate.get("expected_witness_progress")),
                -normalized_physical_cost(estimate),
                str(action.get("action_key") or ""),
            )





        selected = max(safe_source_actions, key=source_priority)
        detail = {
            "reason": "continuation_safe_witness_cover",
            "stop": False,
            "phase": (
                "forced_feasible_source"
                if force_feasible_source and detail.get("stop")
                else "reserved_single_source"
            ),
            "source_cover_reason": str(detail.get("reason") or ""),
        }
    else:
        detail["reason"] = "continuation_safe_witness_cover"
        detail["phase"] = str(detail.get("phase") or "reserved_source_cover")
    detail.update(reserve_detail)
    detail["remaining_budget_after_reserve"] = source_budget
    detail["receding_horizon"] = True
    return selected, detail


def binding_qualified_continuation_first_action(
    actions: Sequence[Mapping[str, Any]],
    estimates: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    *,
    beam_width: int = 16,
    horizon: int = 4,
    stop_upper_bound: float = 0.0,
    min_exploration_actions: int = 1,
    rank_consistent_source_selection: bool = False,
) -> Tuple[Mapping[str, Any] | None, Dict[str, Any]]:
    """Preserve witness feasibility while requiring binding-qualified sources.

    A physical passage is not evidence for every open obligation merely because
    it appears in each binding's candidate pool.  The executor annotates source
    actions whose title/entity partition is compatible with the concrete bound
    identity.  This policy keeps the continuation reserve from
    :func:`continuation_safe_witness_first_action`, but applies two additional
    quality constraints:

    * an identity-qualified source dominates an unqualified shared source; and
    * when no identity-qualified source exists, a bound dense probe dominates a
      low-probability speculative inspection for the same binding.

    The policy never reads passage text or answer labels.  It consumes only
    candidate metadata and calibrated estimates already visible to the physical
    optimizer.
    """
    selected, detail = continuation_safe_witness_first_action(
        actions,
        estimates,
        state,
        beam_width=beam_width,
        horizon=horizon,
        stop_upper_bound=stop_upper_bound,
        min_exploration_actions=min_exploration_actions,
        force_feasible_source=True,
        fallback_scoring_policy=(
            "bswc_rank_consistent_binding_qualified_cover"
            if rank_consistent_source_selection else None
        ),
    )
    detail = dict(detail)
    detail["reason"] = "binding_qualified_continuation_cover"
    if selected is None or str(selected.get("operator") or "") != "SOURCE_INSPECT":
        return selected, detail

    def estimate_for(action: Mapping[str, Any]) -> Dict[str, Any]:
        return dict(estimates.get(str(action.get("action_key") or "")) or {})

    action_positions = {
        str(action.get("action_key") or ""): position
        for position, action in enumerate(actions)
    }

    def source_priority(action: Mapping[str, Any]) -> Tuple[float, float, float, str]:
        estimate = estimate_for(action)
        if rank_consistent_source_selection:
            score, _ = source_action_score(
                "bswc_rank_consistent_binding_qualified_cover",
                action,
                estimate,
                state,
            )
            return (
                score,
                -float(action_positions.get(
                    str(action.get("action_key") or ""), len(actions)
                )),
                0.0,
                "",
            )
        return (
            _float(estimate.get("support_probability")),
            _float(estimate.get("expected_witness_progress")),
            -normalized_physical_cost(estimate),
            str(action.get("action_key") or ""),
        )

    reserve_by_action = {
        str(key): max(1, int(_float(value, 1)))
        for key, value in dict(
            detail.get("continuation_reserve_by_action") or {}
        ).items()
    }
    actions_remaining = max(
        0,
        int(_float(
            (state.get("bswc_remaining_budget") or {}).get(
                "actions_remaining"
            ),
            0,
        )),
    )

    def continuation_feasible(action: Mapping[str, Any]) -> bool:
        action_key = str(action.get("action_key") or "")
        reserve = reserve_by_action.get(
            action_key,
            max(1, int(_float(detail.get("continuation_action_reserve"), 1))),
        )
        return actions_remaining > reserve

    selected_binding = str(selected.get("binding_id") or "")
    qualified_sources = [
        dict(action) for action in actions
        if str(action.get("operator") or "") == "SOURCE_INSPECT"
        and bool(action.get("binding_identity_qualified"))
        and continuation_feasible(action)
        and (
            not selected_binding
            or str(action.get("binding_id") or "") == selected_binding
        )
    ]
    if not qualified_sources:


        qualified_sources = [
            dict(action) for action in actions
            if str(action.get("operator") or "") == "SOURCE_INSPECT"
            and bool(action.get("binding_identity_qualified"))
            and continuation_feasible(action)
        ]
    if qualified_sources and not selected.get("binding_identity_qualified"):
        replacement = max(qualified_sources, key=source_priority)
        detail.update({
            "phase": "binding_identity_source_dominance",
            "replaced_action_key": str(selected.get("action_key") or ""),
            "selected_binding_identity_key": str(
                replacement.get("binding_identity_key") or ""
            ),
            "qualified_replacement_rank_consistent": bool(
                rank_consistent_source_selection
            ),
        })
        return replacement, detail

    if selected.get("binding_identity_qualified"):
        return selected, detail

    dense_candidates = [
        dict(action) for action in actions
        if str(action.get("operator") or "") == "BOUND_DENSE_PROBE"
        and (
            not selected_binding
            or str(action.get("binding_id") or "") == selected_binding
        )
    ]
    if not dense_candidates:
        return selected, detail

    def dense_priority(action: Mapping[str, Any]) -> Tuple[float, float, float, str]:
        estimate = estimate_for(action)
        return (
            _float(estimate.get("support_probability")),
            _float(estimate.get("expected_witness_progress")),
            -normalized_physical_cost(estimate),
            str(action.get("action_key") or ""),
        )

    dense = max(dense_candidates, key=dense_priority)
    source_estimate = estimate_for(selected)
    dense_estimate = estimate_for(dense)
    source_probability = _float(source_estimate.get("support_probability"))
    dense_probability = _float(dense_estimate.get("support_probability"))
    remaining_budget = dict(state.get("bswc_remaining_budget") or {})
    dense_cost = {
        "actions": 1.0,
        "scan_rows": max(0.0, _float(
            dense_estimate.get("hard_scan_rows"),
            _float(dense_estimate.get("expected_scan_rows")),
        )),
        "retrieval_calls": max(0.0, _float(
            dense_estimate.get("hard_retrieval_calls"),
            _float(dense_estimate.get("expected_retrieval_calls"), 1.0),
        )),
        "model_calls": max(0.0, _float(
            dense_estimate.get("hard_model_calls"),
            _float(dense_estimate.get("expected_model_calls")),
        )),
        "tokens": max(0.0, _float(dense_estimate.get("expected_tokens"))),
    }


    dense_continuation_feasible = (
        actions_remaining >= 3
        and _budget_feasible(dense_cost, remaining_budget)
    )
    if dense_probability <= source_probability or not dense_continuation_feasible:
        return selected, detail
    detail.update({
        "phase": "binding_qualified_dense_fallback",
        "replaced_action_key": str(selected.get("action_key") or ""),
        "source_support_probability": source_probability,
        "dense_support_probability": dense_probability,
    })
    return dense, detail


def precedence_aware_witness_first_action(
    actions: Sequence[Mapping[str, Any]],
    estimates: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    *,
    beam_width: int = 16,
    horizon: int = 4,
    stop_upper_bound: float = 0.0,
    min_exploration_actions: int = 1,
) -> Tuple[Mapping[str, Any] | None, Dict[str, Any]]:
    """Close relational prerequisites before optimizing source acquisition.

    Source cover is meaningful only over obligations that currently have source
    alternatives.  LOOKUP_JOIN and BOUND_DENSE_PROBE actions for deferred
    obligations therefore take precedence; after the physical frontier is
    exposed, a Pareto-DP chooses a hard-budget-feasible source set and execution
    replans after every observed outcome.
    """
    if any(str(action.get("operator") or "") == "VERIFY" for action in actions):
        return None, {"reason": "defer_to_mandatory_verification", "stop": False}

    obligations = _open_coverage_keys(state)
    groups = _source_groups(actions, estimates, state)
    source_keys = set().union(
        *(set(group.get("coverage_keys") or []) for group in groups)
    ) if groups else set()
    executable_obligations = obligations & source_keys
    deferred_obligations = obligations - executable_obligations

    def action_coverage_key(action: Mapping[str, Any]) -> str:
        return ":".join((
            str(action.get("binding_id") or ""),
            str(action.get("obligation_id") or ""),
        ))

    def closure_priority(action: Mapping[str, Any]) -> Tuple[float, float, float, int, str]:
        estimate = dict(estimates.get(str(action.get("action_key") or "")) or {})
        return (
            _float(estimate.get("expected_progress")),
            _float(estimate.get("unlock_value")),
            -normalized_physical_cost(estimate),
            -int(_float(action.get("operator_index"), 0)),
            str(action.get("action_key") or ""),
        )

    for operator, phase in (
        ("LOOKUP_JOIN", "relational_closure"),
        ("BOUND_DENSE_PROBE", "source_frontier_expansion"),
    ):
        closure_actions = [
            dict(action) for action in actions
            if str(action.get("operator") or "") == operator
            and action_coverage_key(action) in deferred_obligations
        ]
        if closure_actions:
            selected = max(closure_actions, key=closure_priority)
            return selected, {
                "reason": "precedence_aware_witness_cover",
                "stop": False,
                "phase": phase,
                "all_open_obligation_count": len(obligations),
                "source_executable_obligation_count": len(executable_obligations),
                "deferred_obligation_count": len(deferred_obligations),
                "source_executable_obligations": sorted(executable_obligations),
                "deferred_obligations": sorted(deferred_obligations),
                "selected_prerequisite_action_key": str(
                    selected.get("action_key") or ""
                ),
                "solver": "precedence_then_pareto_dynamic_program",
                "receding_horizon": True,
            }

    if len(executable_obligations) < 2 or len(groups) < 2:
        return None, {
            "reason": "precedence_aware_trivial_source_frontier",
            "stop": False,
            "phase": "fallback_single_source",
            "all_open_obligation_count": len(obligations),
            "source_executable_obligation_count": len(executable_obligations),
            "deferred_obligation_count": len(deferred_obligations),
            "unique_source_count": len(groups),
        }

    remaining_budget = dict(state.get("bswc_remaining_budget") or {})
    candidates = []
    for group in groups:
        coverage = set(group.get("coverage_keys") or []) & executable_obligations
        if not coverage:
            continue
        probabilities = {
            key: value for key, value in dict(
                group.get("probability_by_key") or {}
            ).items() if key in coverage
        }
        candidates.append({
            **dict(group),
            "name": str(group.get("source_id") or ""),
            "coverage_keys": coverage,
            "predicted_coverage_keys": sorted(coverage),
            "probability_by_key": probabilities,
            "support_probability": min(
                (_float(value) for value in probabilities.values()),
                default=0.0,
            ),
        })

    pruned_candidate_count = 0
    if len(candidates) > 22:
        def preaccess_priority(row: Mapping[str, Any]) -> Tuple[float, float, str]:
            gain = sum(
                _float((row.get("probability_by_key") or {}).get(key))
                for key in executable_obligations & _coverage(row)
            )
            cost = max(0.05, _float(row.get("cost"), 1.0))
            return gain / cost, gain, str(row.get("source_id") or "")

        candidates.sort(key=preaccess_priority, reverse=True)
        pruned_candidate_count = len(candidates) - 22
        candidates = candidates[:22]

    upper_bound = optimistic_witness_completion_upper_bound(
        executable_obligations, candidates, remaining_budget=remaining_budget
    )
    inspection_count = int(_float(state.get("bswc_source_inspection_count")))
    threshold = min(1.0, max(0.0, float(stop_upper_bound)))
    if (
        not deferred_obligations
        and inspection_count >= max(0, int(min_exploration_actions))
        and upper_bound <= threshold
    ):
        return None, {
            "reason": "precedence_aware_witness_cover",
            "stop": True,
            "phase": "source_cover",
            "stop_reason": "reachable_probability_upper_bound_low",
            "reachable_completion_upper_bound": round(upper_bound, 10),
            "stop_upper_bound": threshold,
            "all_open_obligation_count": len(obligations),
            "source_executable_obligation_count": len(executable_obligations),
            "deferred_obligation_count": 0,
            "unique_source_count": len(groups),
        }

    search_depth = max(1, min(int(horizon), len(candidates)))
    plan = feasibility_preserving_witness_plan_pareto(
        executable_obligations,
        candidates,
        remaining_budget=remaining_budget,
        max_candidates=22,
        max_sources=search_depth,
    )
    detail = {
        "reason": "precedence_aware_witness_cover",
        "stop": plan.get("status") != "optimal" and not deferred_obligations,
        "phase": "source_cover",
        "all_open_obligation_count": len(obligations),
        "source_executable_obligation_count": len(executable_obligations),
        "deferred_obligation_count": len(deferred_obligations),
        "source_executable_obligations": sorted(executable_obligations),
        "deferred_obligations": sorted(deferred_obligations),
        "unique_source_count": len(groups),
        "candidate_source_count": len(candidates),
        "pruned_candidate_count": pruned_candidate_count,
        "beam_width": max(1, int(beam_width)),
        "horizon": search_depth,
        "remaining_budget": remaining_budget,
        "reachable_completion_upper_bound": round(upper_bound, 10),
        "stop_upper_bound": threshold,
        "min_exploration_actions": max(0, int(min_exploration_actions)),
        "source_inspection_count": inspection_count,
        "plan": {key: value for key, value in plan.items() if key != "selected"},
    }
    if plan.get("status") != "optimal" or not plan.get("selected"):
        detail["stop_reason"] = str(
            plan.get("stop_reason") or plan.get("status") or "unknown"
        )
        return None, detail

    selected_groups = [dict(row) for row in plan["selected"]]
    alternatives_per_key = {
        key: sum(key in set(group.get("coverage_keys") or []) for group in groups)
        for key in executable_obligations
    }

    def first_source_priority(group: Mapping[str, Any]) -> Tuple[float, float, str]:
        critical_gain = sum(
            _float((group.get("probability_by_key") or {}).get(key))
            / max(1, alternatives_per_key.get(key, 1))
            for key in executable_obligations & _coverage(group)
        )
        pressure = _budget_pressure(
            _candidate_resource_cost(group), remaining_budget
        )
        return (
            critical_gain / max(0.01, pressure),
            critical_gain,
            str(group.get("source_id") or ""),
        )

    first_group = max(selected_groups, key=first_source_priority)

    def first_action_priority(item: Mapping[str, Any]) -> Tuple[float, float, str]:
        own_key = str(item.get("own_key") or "")
        criticality = 1.0 / max(1, alternatives_per_key.get(own_key, 1))
        probability = _float(item.get("probability"))
        witness = _float(
            (item.get("estimate") or {}).get("expected_witness_progress"),
            probability,
        )
        return probability * criticality, witness, str(
            (item.get("action") or {}).get("action_key") or ""
        )

    first_item = max(first_group["actions"], key=first_action_priority)
    detail.update({
        "selected_source_ids": [
            str(group.get("source_id") or "") for group in selected_groups
        ],
        "first_source_id": str(first_group.get("source_id") or ""),
        "predicted_frontier_complete_probability": plan.get(
            "complete_probability"
        ),
        "predicted_expected_coverage": plan.get("expected_coverage"),
        "predicted_budget_pressure": plan.get("budget_pressure"),
        "predicted_resource_usage": plan.get("resource_usage"),
        "probability_model": "calibrated_preaccess_independence",
        "objective_order": [
            "source_frontier_complete_probability",
            "expected_coverage",
            "budget_pressure",
            "scalar_cost",
        ],
        "solver": "precedence_then_pareto_dynamic_program",
        "exact_on_retained_frontier": bool(
            plan.get("exact_on_retained_frontier")
        ),
        "receding_horizon": True,
    })
    return dict(first_item["action"]), detail


def quality_first_precedence_witness_first_action(
    actions: Sequence[Mapping[str, Any]],
    estimates: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    *,
    beam_width: int = 16,
    horizon: int = 4,
    stop_upper_bound: float = 0.0,
    min_exploration_actions: int = 1,
    global_witness_reservation: bool = False,
) -> Tuple[Mapping[str, Any] | None, Dict[str, Any]]:
    """Preserve answer reachability before optimizing physical source cost.

    The older precedence policy closes relational prerequisites by expected
    progress.  On branch queries, equally scored LOOKUP_JOIN actions can have
    very different query-local fanout and consume the scan budget needed to
    materialize another required branch.  This policy uses only information
    available before access: mandatory verification, exact local index
    cardinality, remaining hard budget, and stable logical root order.

    The optional global-witness mode prevents an active branch from monopolizing
    the budget.  It exposes low-fanout gaps first, reserves observable work for
    unopened branches, and replans after a failed source access.  The policy
    uses no QA answer and does not alter the source-value model or safety gates.
    """
    indexed_actions = list(enumerate(actions))
    verify_actions = [
        (index, dict(action))
        for index, action in indexed_actions
        if str(action.get("operator") or "") == "VERIFY"
    ]
    if verify_actions:
        def verify_priority(item: Tuple[int, Mapping[str, Any]]) -> Tuple[float, float, float, int]:
            index, action = item
            estimate = dict(
                estimates.get(str(action.get("action_key") or "")) or {}
            )
            return (
                _float(estimate.get("support_probability")),
                _float(estimate.get("expected_progress")),
                -normalized_physical_cost(estimate),
                -index,
            )

        _, selected = max(verify_actions, key=verify_priority)
        return selected, {
            "reason": (
                "global_witness_reservation_cover"
                if global_witness_reservation else
                "quality_first_precedence_witness_cover"
            ),
            "stop": False,
            "phase": "mandatory_verification",
            "selected_prerequisite_action_key": str(
                selected.get("action_key") or ""
            ),
            "objective_order": [
                "mandatory_verification",
                "support_probability",
                "expected_progress",
                "physical_cost",
            ],
            "receding_horizon": True,
        }

    obligations = _open_coverage_keys(state)
    groups = _source_groups(actions, estimates, state)
    source_keys = set().union(
        *(set(group.get("coverage_keys") or []) for group in groups)
    ) if groups else set()
    executable_obligations = obligations & source_keys
    deferred_obligations = obligations - executable_obligations

    bindings = {
        str(row.get("binding_id") or ""): dict(row)
        for row in state.get("bindings") or []
        if str(row.get("binding_id") or "")
    }
    root_order = {
        str(root_id): index
        for index, root_id in enumerate(state.get("root_binding_ids") or [])
    }
    remaining_scan_rows = max(
        0,
        int(_float(
            (state.get("bswc_remaining_budget") or {}).get(
                "scan_rows_remaining"
            )
        )),
    )

    def action_coverage_key(action: Mapping[str, Any]) -> str:
        return ":".join((
            str(action.get("binding_id") or ""),
            str(action.get("obligation_id") or ""),
        ))

    def action_root_order(action: Mapping[str, Any]) -> int:
        binding_id = str(action.get("binding_id") or "")
        row = bindings.get(binding_id) or {}
        root_id = str(row.get("root_binding_id") or binding_id)
        return root_order.get(root_id, len(root_order))

    def quality_closure_priority(
        item: Tuple[int, Mapping[str, Any]]
    ) -> Tuple[float, int, float, float, int, int]:
        input_index, action = item
        estimate = dict(
            estimates.get(str(action.get("action_key") or "")) or {}
        )
        operator = str(action.get("operator") or "")
        local_fanout = max(0, int(_float(estimate.get("index_fanout_rows"))))
        projected_scan_remaining = remaining_scan_rows
        if operator == "LOOKUP_JOIN":
            projected_scan_remaining -= min(local_fanout, remaining_scan_rows)
        first_priority = (
            -float(local_fanout)
            if global_witness_reservation and operator == "LOOKUP_JOIN"
            else float(int(operator != "LOOKUP_JOIN" or local_fanout > 0))
        )
        return (
            first_priority,
            projected_scan_remaining,
            _float(estimate.get("expected_progress")),
            _float(estimate.get("unlock_value")),
            -action_root_order(action),
            -input_index,
        )

    def closure_actions_for(keys: Set[str]) -> List[Tuple[int, Dict[str, Any]]]:
        for operator in ("LOOKUP_JOIN", "BOUND_DENSE_PROBE"):
            rows = [
                (index, dict(action))
                for index, action in indexed_actions
                if str(action.get("operator") or "") == operator
                and action_coverage_key(action) in keys
            ]
            if rows:
                return rows
        return []

    def select_closure(
        keys: Set[str], phase: str
    ) -> Tuple[Mapping[str, Any] | None, Dict[str, Any]]:
        rows = closure_actions_for(keys)
        if not rows:
            return None, {}
        _, selected = max(rows, key=quality_closure_priority)
        selected_estimate = dict(
            estimates.get(str(selected.get("action_key") or "")) or {}
        )
        selected_fanout = max(
            0, int(_float(selected_estimate.get("index_fanout_rows")))
        )
        return selected, {
            "reason": (
                "global_witness_reservation_cover"
                if global_witness_reservation
                else "quality_first_precedence_witness_cover"
            ),
            "stop": False,
            "phase": phase,
            "all_open_obligation_count": len(obligations),
            "source_executable_obligation_count": len(executable_obligations),
            "deferred_obligation_count": len(deferred_obligations),
            "source_executable_obligations": sorted(executable_obligations),
            "deferred_obligations": sorted(deferred_obligations),
            "selected_prerequisite_action_key": str(
                selected.get("action_key") or ""
            ),
            "selected_local_index_fanout": selected_fanout,
            "projected_scan_rows_remaining": (
                remaining_scan_rows
                - min(selected_fanout, remaining_scan_rows)
                if str(selected.get("operator") or "") == "LOOKUP_JOIN"
                else remaining_scan_rows
            ),
            "objective_order": (
                [
                    "minimum_local_index_fanout_including_zero",
                    "preserve_global_witness_budget",
                    "expected_progress",
                    "unlock_value",
                    "logical_root_order",
                ]
                if global_witness_reservation else [
                    "nonempty_local_index_access",
                    "preserve_scan_budget_for_remaining_branches",
                    "expected_progress",
                    "unlock_value",
                    "logical_root_order",
                ]
            ),
            "solver": (
                "global_witness_reservation_then_pareto_dynamic_program"
                if global_witness_reservation else
                "quality_first_precedence_then_pareto_dynamic_program"
            ),
            "receding_horizon": True,
        }

    def failed_source_coverage_keys() -> Set[str]:
        failed: Set[str] = set()
        prefix = "SOURCE_INSPECT:"
        for event in state.get("bswc_feedback_events") or []:
            if bool(event.get("success")):
                continue
            action_key = str(event.get("action_key") or "")
            if not action_key.startswith(prefix):
                continue
            parts = action_key[len(prefix):].split(":", 2)
            if len(parts) >= 2:
                failed.add(f"{parts[0]}:{parts[1]}")
        return failed

    def minimum_action_resource_cost(
        action: Mapping[str, Any]
    ) -> Dict[str, float]:
        estimate = dict(
            estimates.get(str(action.get("action_key") or "")) or {}
        )
        operator = str(action.get("operator") or "")
        if operator == "LOOKUP_JOIN":
            fanout = max(0, int(_float(estimate.get("index_fanout_rows"))))
            scan_rows = float(min(1, fanout))
        else:
            scan_rows = max(0.0, _float(
                estimate.get("hard_scan_rows"),
                _float(estimate.get("expected_scan_rows")),
            ))
        return {
            "actions": 1.0,
            "scan_rows": scan_rows,
            "retrieval_calls": max(0.0, _float(
                estimate.get("hard_retrieval_calls"),
                _float(estimate.get("expected_retrieval_calls")),
            )),
            "model_calls": max(0.0, _float(
                estimate.get("hard_model_calls"),
                _float(estimate.get("expected_model_calls")),
            )),
            "tokens": max(0.0, _float(estimate.get("expected_tokens"))),
        }

    def unopened_branch_reserve() -> Dict[str, float]:
        reserve = {key: 0.0 for key in _RESOURCE_REMAINING_KEYS}
        for coverage_key in sorted(deferred_obligations):
            rows = closure_actions_for({coverage_key})
            if not rows:
                continue
            cheapest = min(
                (row[1] for row in rows),
                key=lambda action: (
                    _budget_pressure(
                        minimum_action_resource_cost(action),
                        dict(state.get("bswc_remaining_budget") or {}),
                    ),
                    normalized_physical_cost(dict(estimates.get(
                        str(action.get("action_key") or "")
                    ) or {})),
                    str(action.get("action_key") or ""),
                ),
            )
            reserve = _add_resource_costs(
                reserve, minimum_action_resource_cost(cheapest)
            )
        return reserve

    failed_source_keys = failed_source_coverage_keys()
    if global_witness_reservation and failed_source_keys:



        selected, detail = select_closure(
            deferred_obligations,
            "global_witness_branch_rebalance",
        )
        if selected is not None:
            detail["failed_source_obligations"] = sorted(failed_source_keys)
            detail["rebalance_reason"] = "source_miss_with_unopened_branch"
            return selected, detail

        untouched = executable_obligations - failed_source_keys
        if not untouched:
            probe_rows = [
                (index, dict(action))
                for index, action in indexed_actions
                if str(action.get("operator") or "") == "BOUND_DENSE_PROBE"
                and action_coverage_key(action) in failed_source_keys
            ]
            if probe_rows:
                _, selected = max(probe_rows, key=quality_closure_priority)
                return selected, {
                    "reason": "global_witness_reservation_cover",
                    "stop": False,
                    "phase": "global_witness_frontier_refresh",
                    "failed_source_obligations": sorted(failed_source_keys),
                    "selected_prerequisite_action_key": str(
                        selected.get("action_key") or ""
                    ),
                    "rebalance_reason": "source_miss_before_additional_retry",
                    "objective_order": [
                        "refresh_source_frontier_after_miss",
                        "avoid_low_probability_source_retries",
                        "preserve_global_witness_budget",
                    ],
                    "solver": (
                        "global_witness_reservation_then_pareto_dynamic_program"
                    ),
                    "receding_horizon": True,
                }







    if executable_obligations and groups:
        remaining_budget = dict(state.get("bswc_remaining_budget") or {})
        candidates = []
        for group in groups:
            coverage = set(group.get("coverage_keys") or []) & executable_obligations
            if not coverage:
                continue
            probabilities = {
                key: value
                for key, value in dict(group.get("probability_by_key") or {}).items()
                if key in coverage
            }
            candidates.append({
                **dict(group),
                "name": str(group.get("source_id") or ""),
                "coverage_keys": coverage,
                "predicted_coverage_keys": sorted(coverage),
                "probability_by_key": probabilities,
                "support_probability": min(
                    (_float(value) for value in probabilities.values()),
                    default=0.0,
                ),
            })

        pruned_candidate_count = 0
        if len(candidates) > 22:
            def quality_source_priority(
                row: Mapping[str, Any]
            ) -> Tuple[float, float, str]:
                gain = sum(
                    _float((row.get("probability_by_key") or {}).get(key))
                    for key in executable_obligations & _coverage(row)
                )
                return (
                    gain,
                    -_budget_pressure(
                        _candidate_resource_cost(row), remaining_budget
                    ),
                    str(row.get("source_id") or ""),
                )

            candidates.sort(key=quality_source_priority, reverse=True)
            pruned_candidate_count = len(candidates) - 22
            candidates = candidates[:22]

        search_depth = max(1, min(int(horizon), len(candidates)))
        plan = feasibility_preserving_witness_plan_pareto(
            executable_obligations,
            candidates,
            remaining_budget=remaining_budget,
            max_candidates=22,
            max_sources=search_depth,
        )
        if plan.get("status") == "optimal" and plan.get("selected"):
            selected_groups = [dict(row) for row in plan["selected"]]
            alternatives_per_key = {
                key: sum(
                    key in set(group.get("coverage_keys") or [])
                    for group in groups
                )
                for key in executable_obligations
            }

            def first_source_priority(
                group: Mapping[str, Any]
            ) -> Tuple[float, float, str]:
                critical_gain = sum(
                    _float((group.get("probability_by_key") or {}).get(key))
                    / max(1, alternatives_per_key.get(key, 1))
                    for key in executable_obligations & _coverage(group)
                )
                return (
                    critical_gain,
                    -_budget_pressure(
                        _candidate_resource_cost(group), remaining_budget
                    ),
                    str(group.get("source_id") or ""),
                )

            first_group = max(selected_groups, key=first_source_priority)

            def first_action_priority(
                item: Mapping[str, Any]
            ) -> Tuple[float, float, float, str]:
                own_key = str(item.get("own_key") or "")
                criticality = 1.0 / max(
                    1, alternatives_per_key.get(own_key, 1)
                )
                probability = _float(item.get("probability"))
                estimate = dict(item.get("estimate") or {})
                return (
                    probability * criticality,
                    _float(
                        estimate.get("expected_witness_progress"), probability
                    ),
                    -normalized_physical_cost(estimate),
                    str((item.get("action") or {}).get("action_key") or ""),
                )

            first_item = max(first_group["actions"], key=first_action_priority)
            if global_witness_reservation and deferred_obligations:
                reserve = unopened_branch_reserve()


                reserve = _add_resource_costs(reserve, {
                    "actions": 1.0,
                    "scan_rows": 1.0,
                    "retrieval_calls": 0.0,
                    "model_calls": 0.0,
                    "tokens": 0.0,
                })
                selected_cost = minimum_action_resource_cost(
                    dict(first_item["action"])
                )
                required = _add_resource_costs(reserve, selected_cost)
                if not _budget_feasible(required, remaining_budget):
                    selected, detail = select_closure(
                        deferred_obligations,
                        "global_witness_budget_reservation",
                    )
                    if selected is not None:
                        detail.update({
                            "reserved_resource_usage": reserve,
                            "candidate_source_resource_usage": selected_cost,
                            "combined_required_resource_usage": required,
                            "rebalance_reason": (
                                "source_access_would_consume_unopened_branch_reserve"
                            ),
                        })
                        return selected, detail
            return dict(first_item["action"]), {
                "reason": (
                    "global_witness_reservation_cover"
                    if global_witness_reservation else
                    "quality_first_precedence_witness_cover"
                ),
                "stop": False,
                "phase": "source_cover",
                "source_cover_mode": "active_witness_completion",
                "global_witness_reservation": bool(global_witness_reservation),
                "failed_source_obligations": sorted(failed_source_keys),
                "reserved_resource_usage": (
                    unopened_branch_reserve()
                    if global_witness_reservation else {}
                ),
                "all_open_obligation_count": len(obligations),
                "source_executable_obligation_count": len(
                    executable_obligations
                ),
                "deferred_obligation_count": len(deferred_obligations),
                "source_executable_obligations": sorted(
                    executable_obligations
                ),
                "deferred_obligations": sorted(deferred_obligations),
                "candidate_source_count": len(candidates),
                "pruned_candidate_count": pruned_candidate_count,
                "selected_source_ids": [
                    str(group.get("source_id") or "")
                    for group in selected_groups
                ],
                "first_source_id": str(first_group.get("source_id") or ""),
                "predicted_frontier_complete_probability": plan.get(
                    "complete_probability"
                ),
                "predicted_expected_coverage": plan.get("expected_coverage"),
                "predicted_budget_pressure": plan.get("budget_pressure"),
                "predicted_resource_usage": plan.get("resource_usage"),
                "plan": {
                    key: value for key, value in plan.items()
                    if key != "selected"
                },
                "objective_order": [
                    (
                        "global_complete_witness_probability"
                        if global_witness_reservation else
                        "finish_active_witness"
                    ),
                    "source_frontier_complete_probability",
                    "expected_coverage",
                    "budget_pressure",
                ],
                "solver": (
                    "global_witness_reservation_then_pareto_dynamic_program"
                    if global_witness_reservation else
                    "quality_first_precedence_then_pareto_dynamic_program"
                ),
                "exact_on_retained_frontier": bool(
                    plan.get("exact_on_retained_frontier")
                ),
                "receding_horizon": True,
            }

    for operator, phase in (
        ("LOOKUP_JOIN", "quality_preserving_relational_closure"),
        ("BOUND_DENSE_PROBE", "quality_preserving_frontier_expansion"),
    ):
        closure_actions = [
            (index, dict(action))
            for index, action in indexed_actions
            if str(action.get("operator") or "") == operator
            and action_coverage_key(action) in deferred_obligations
        ]
        if closure_actions:
            selected, detail = select_closure(deferred_obligations, phase)
            if selected is not None:
                return selected, detail

    selected, detail = precedence_aware_witness_first_action(
        actions,
        estimates,
        state,
        beam_width=beam_width,
        horizon=horizon,
        stop_upper_bound=stop_upper_bound,
        min_exploration_actions=min_exploration_actions,
    )
    detail = dict(detail)
    if str(detail.get("phase") or "") == "source_cover":
        detail["reason"] = (
            "global_witness_reservation_cover"
            if global_witness_reservation else
            "quality_first_precedence_witness_cover"
        )
        detail["solver"] = (
            "global_witness_reservation_then_pareto_dynamic_program"
            if global_witness_reservation else
            "quality_first_precedence_then_pareto_dynamic_program"
        )
    return selected, detail


def binding_aware_bounded_first_action(
    actions: Sequence[Mapping[str, Any]],
    estimates: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    *,
    beam_width: int = 8,
    horizon: int = 3,
) -> Tuple[Mapping[str, Any] | None, Dict[str, Any]]:
    """Return the first action of a small binding-aware source plan.

    This is not claimed to be an exact solver for stochastic BSWC.  It searches
    short source sequences and values only binding-compatible obligation keys,
    thereby avoiding a common greedy failure where unrelated predicate hits
    appear to cover the plan.
    """
    source_actions = [
        dict(action) for action in actions
        if str(action.get("operator") or "") == "SOURCE_INSPECT"
    ]
    if not source_actions:
        return None, {"reason": "no_source_inspect_frontier"}
    open_keys = _open_coverage_keys(state)
    indexed = list(enumerate(source_actions))
    beam = [_BeamState((), frozenset(), 0.0, 0.0)]
    completed: List[_BeamState] = []
    expanded = 0
    for _ in range(max(1, min(int(horizon), len(source_actions)))):
        frontier: List[_BeamState] = []
        for node in beam:
            if open_keys and open_keys.issubset(node.covered):
                completed.append(node)
                continue
            used = set(node.selected)
            for index, action in indexed:
                if index in used:
                    continue
                estimate = estimates.get(str(action.get("action_key") or ""), {})
                coverage = _coverage(action)
                if open_keys:
                    coverage &= open_keys
                marginal = coverage - set(node.covered)
                support = min(1.0, max(0.0, _float(
                    estimate.get("support_probability"), 0.5
                )))
                cost = normalized_physical_cost(estimate)
                gain = support * len(marginal)
                frontier.append(_BeamState(
                    node.selected + (index,),
                    frozenset(set(node.covered) | coverage),
                    node.expected_gain + gain,
                    node.cost + cost,
                ))
                expanded += 1
        if not frontier:
            break

        def objective(node: _BeamState) -> Tuple[float, int, float]:
            complete = int(bool(open_keys) and open_keys.issubset(node.covered))
            utility = (
                -node.cost + 0.01 * node.expected_gain
                if complete
                else node.expected_gain - 0.08 * node.cost
            )
            return complete, utility, len(node.covered)

        frontier.sort(key=objective, reverse=True)
        beam = frontier[:max(1, int(beam_width))]

    candidates = completed + beam
    if not candidates:
        return None, {"reason": "empty_bounded_search"}
    best = max(
        candidates,
        key=objective,
    )
    first = source_actions[best.selected[0]]
    return first, {
        "reason": "binding_aware_bounded_search",
        "beam_width": max(1, int(beam_width)),
        "horizon": max(1, int(horizon)),
        "expanded_state_count": expanded,
        "selected_sequence": [
            str(source_actions[index].get("action_key") or "")
            for index in best.selected
        ],
        "predicted_covered_keys": sorted(best.covered),
        "predicted_expected_gain": round(best.expected_gain, 8),
        "predicted_cost": round(best.cost, 8),
    }


def minimum_cost_obligation_cover_exact(
    obligations: Iterable[str],
    candidates: Sequence[Mapping[str, Any]],
    *,
    max_candidates: int = 22,
) -> Dict[str, Any]:
    """Exact offline oracle for deterministic small MCSOC instances."""
    universe = {str(value) for value in obligations if str(value)}
    rows = [dict(row) for row in candidates]
    if len(rows) > int(max_candidates):
        return {
            "status": "too_many_candidates",
            "candidate_count": len(rows),
            "max_candidates": int(max_candidates),
        }
    ordered = sorted(universe)
    bit_for = {key: 1 << index for index, key in enumerate(ordered)}
    full_mask = (1 << len(ordered)) - 1




    dp: Dict[int, Tuple[float, Tuple[int, ...]]] = {0: (0.0, ())}
    for index, row in enumerate(rows):
        row_mask = 0
        for key in _coverage(row) & universe:
            row_mask |= bit_for[key]
        if row_mask == 0:
            continue
        row_cost = max(0.0, _float(row.get("cost"), 1.0))
        updated = dict(dp)
        for mask, (cost, selected) in dp.items():
            new_mask = mask | row_mask
            candidate = (cost + row_cost, selected + (index,))
            if new_mask not in updated or candidate < updated[new_mask]:
                updated[new_mask] = candidate
        dp = updated
    best = dp.get(full_mask)
    if best is None:
        return {"status": "infeasible", "selected": [], "cost": None}
    return {
        "status": "optimal",
        "selected": [dict(rows[index]) for index in best[1]],
        "selected_indices": list(best[1]),
        "cost": best[0],
    }


def cost_normalized_greedy_cover(
    obligations: Iterable[str], candidates: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Deterministic weighted-set-cover baseline used in offline analysis."""
    uncovered = {str(value) for value in obligations if str(value)}
    remaining = [dict(row) for row in candidates]
    selected: List[Dict[str, Any]] = []
    total_cost = 0.0
    while uncovered:
        ranked = []
        for index, row in enumerate(remaining):
            marginal = _coverage(row) & uncovered
            cost = max(0.05, _float(row.get("cost"), 1.0))
            ranked.append((len(marginal) / cost, len(marginal), -cost, -index, row))
        if not ranked:
            break
        score, marginal_count, _, _, chosen = max(ranked)
        if score <= 0.0 or marginal_count <= 0:
            break
        selected.append(chosen)
        total_cost += max(0.0, _float(chosen.get("cost"), 1.0))
        uncovered -= _coverage(chosen)
        remaining.remove(chosen)
    return {
        "status": "covered" if not uncovered else "infeasible",
        "selected": selected,
        "cost": total_cost,
        "uncovered": sorted(uncovered),
    }
