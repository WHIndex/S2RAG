"""Lossless dependency ordering for generated semantic query relations.

The model owns relation semantics, variables, and endpoint declarations.  This
module may only reorder an already complete linear dependency chain and update
the relation indices referenced by the model-owned requirement certificate. It
never creates or deletes entities, relations, variables, or requirements.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence, Tuple


SEMANTIC_DEPENDENCY_NORMALIZATION_VERSION = (
    "semantic_dependency_normalization_v3_10_20261029"
)


def normalize_semantic_dependency_order(
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Return a uniquely ordered linear semantic chain when one exists.

    A reorder is allowed only when there is one semantic root variable, every
    relation output has one producer, and exactly one relation is executable at
    every dependency step.  The unique-ready-edge requirement deliberately
    rejects branches and independent side constraints: choosing an order for
    those shapes would be query planning, not lossless normalization.
    """
    trace: Dict[str, Any] = {
        "version": SEMANTIC_DEPENDENCY_NORMALIZATION_VERSION,
        "applied": False,
        "lossless": False,
        "original_relation_order": [],
        "normalized_relation_order": [],
    }
    if not isinstance(payload, Mapping):
        return {}, "dependency_normalization_invalid_payload", trace
    if payload.get("abstain") is True:
        return {}, "dependency_normalization_not_applicable_to_abstain", trace
    if str(payload.get("answer_operation") or "project").strip().casefold() != "project":
        return {}, "dependency_normalization_non_project_unsupported", trace

    starts = payload.get("start_entities")
    relations = payload.get("relations")
    requirements = payload.get("question_requirements")
    if not isinstance(starts, Sequence) or isinstance(starts, (str, bytes)):
        return {}, "dependency_normalization_invalid_start_entities", trace
    if not isinstance(relations, list) or not relations:
        return {}, "dependency_normalization_invalid_relations", trace
    if not isinstance(requirements, list) or len(requirements) != len(relations):
        return {}, "dependency_normalization_requirement_count_mismatch", trace

    root_variables = {
        str(row.get("variable") or "").strip()
        for row in starts
        if isinstance(row, Mapping) and str(row.get("variable") or "").strip()
    }
    trace["root_variables"] = sorted(root_variables)
    if len(root_variables) != 1:
        return {}, "dependency_normalization_requires_one_root_variable", trace
    root_variable = next(iter(root_variables))

    output_to_index: Dict[str, int] = {}
    relation_rows = []
    for index, raw in enumerate(relations):
        if not isinstance(raw, Mapping):
            return {}, f"dependency_normalization_invalid_relation_{index}", trace
        input_variable = str(raw.get("input_variable") or "").strip()
        output_variable = str(raw.get("output_variable") or "").strip()
        if not input_variable or not output_variable or input_variable == output_variable:
            return {}, f"dependency_normalization_invalid_edge_{index}", trace
        if output_variable == root_variable:
            return {}, f"dependency_normalization_overwrites_root_{index}", trace
        if output_variable in output_to_index:
            return {}, "dependency_normalization_duplicate_output_producer", trace
        output_to_index[output_variable] = index
        relation_rows.append((input_variable, output_variable))

    final_variable = str(payload.get("final_answer_variable") or "").strip()
    trace["final_answer_variable"] = final_variable
    if final_variable not in output_to_index:
        return {}, "dependency_normalization_final_variable_not_produced", trace

    available = {root_variable}
    unscheduled = set(range(len(relations)))
    schedule = []
    while unscheduled:
        ready = sorted(
            index
            for index in unscheduled
            if relation_rows[index][0] in available
        )
        if not ready:
            trace["blocked_relation_indices"] = sorted(unscheduled)
            return {}, "dependency_normalization_disconnected_or_cyclic", trace
        if len(ready) != 1:
            trace["ambiguous_ready_relation_indices"] = ready
            return {}, "dependency_normalization_non_linear_or_ambiguous", trace
        selected = ready[0]
        schedule.append(selected)
        available.add(relation_rows[selected][1])
        unscheduled.remove(selected)

    trace["original_relation_order"] = list(range(len(relations)))
    trace["normalized_relation_order"] = schedule
    if relation_rows[schedule[-1]][1] != final_variable:
        return {}, "dependency_normalization_final_variable_not_terminal", trace

    old_to_new = {old_index: new_index for new_index, old_index in enumerate(schedule)}
    normalized_requirements = []
    seen_covered_indices = set()
    for requirement_index, raw in enumerate(requirements):
        if not isinstance(raw, Mapping):
            return {}, f"dependency_normalization_invalid_requirement_{requirement_index}", trace
        covered = raw.get("covered_by_relation_index")
        if isinstance(covered, bool) or not isinstance(covered, int):
            return {}, f"dependency_normalization_invalid_requirement_index_{requirement_index}", trace
        if covered not in old_to_new or covered in seen_covered_indices:
            return {}, "dependency_normalization_invalid_requirement_bijection", trace
        seen_covered_indices.add(covered)
        row = deepcopy(dict(raw))
        row["covered_by_relation_index"] = old_to_new[covered]
        normalized_requirements.append(row)
    if len(seen_covered_indices) != len(relations):
        return {}, "dependency_normalization_incomplete_requirement_bijection", trace

    normalized = deepcopy(dict(payload))
    normalized["relations"] = [deepcopy(dict(relations[index])) for index in schedule]
    normalized["question_requirements"] = normalized_requirements
    trace.update({
        "applied": schedule != list(range(len(relations))),
        "lossless": True,
        "requirement_index_map": {
            str(old_index): new_index for old_index, new_index in old_to_new.items()
        },
    })
    reason = (
        "dependency_order_reordered"
        if trace["applied"]
        else "dependency_order_already_valid"
    )
    return normalized, reason, trace
