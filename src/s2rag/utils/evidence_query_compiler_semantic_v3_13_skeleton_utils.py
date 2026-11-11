"""Semantic V3.13 skeleton generation and deterministic direction repair.

V3.13 keeps the V3.12 semantic-only model boundary.  It adds two narrowly
scoped changes motivated by the cross-dataset failure audit:

* nested relational descriptions must be expanded from their innermost named
  instance instead of being emitted as a synthetic start entity; and
* physical lowering may reverse a model-proposed traversal only when the
  endpoint catalog makes the opposite traversal uniquely type-compatible.

The second rule never changes a predicate family, entity mention, output type,
or relation count.  Ambiguous and incompatible candidates remain rejected.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, Mapping

from .evidence_query_compiler_semantic_v3_12_skeleton_utils import (
    _schema_link_family,
    _type_compatible,
    build_semantic_v3_12_skeleton_messages,
    lower_semantic_v3_12_skeleton,
)
from .evidence_query_compiler_v2_utils import PREDICATE_ENDPOINT_SCHEMA


SEMANTIC_V3_13_PROMPT_VERSION = "semantic_skeleton_generation_v3_13_20261030"
SEMANTIC_V3_13_CACHE_VERSION = "semantic_v3_13_skeleton_cache_v1"
SEMANTIC_V3_13_PIPELINE_VERSION = (
    "deterministic_skeleton_lowering_v3_13_20261030"
)


def build_semantic_v3_13_skeleton_messages(
    question: str,
    *,
    max_completion_tokens: int = 650,
) -> list[dict[str, str]]:
    """Build the V3.13 semantic-only request with nested-chain guidance."""
    messages = copy.deepcopy(
        build_semantic_v3_12_skeleton_messages(
            question,
            max_completion_tokens=max_completion_tokens,
        )
    )
    messages[0]["content"] += (
        " A possessive or relational description is never a named start entity. "
        "For example, in 'Erik Hort's birthplace', start from 'Erik Hort' and "
        "emit the birth relation before any relation applied to the birthplace. "
        "For nested 'the X of Y' phrases, start from the innermost explicit named "
        "instance and emit every relation needed to reach the answer. Do not use "
        "a phrase containing a relation cue as the start merely to shorten the plan."
    )
    payload = json.loads(messages[1]["content"])
    payload["protocol"] = SEMANTIC_V3_13_PROMPT_VERSION
    payload["cache_contract"] = SEMANTIC_V3_13_CACHE_VERSION
    payload["nested_description_contract"] = {
        "rule": (
            "Start from the innermost named instance and preserve every relation "
            "in possessive and nested-of descriptions."
        ),
        "forbidden_start_example": "Erik Hort's birthplace",
    }
    payload.setdefault("worked_examples", []).append({
        "question": "What county is Erik Hort's birthplace a part of?",
        "output": {
            "decision": "PLAN",
            "start_entity": {"mention": "Erik Hort", "semantic_type": "person"},
            "steps": [
                {
                    "relation_cue": "birthplace",
                    "predicate_family": "birth",
                    "predicate_subtype": "birth_place",
                    "traversal": "LEFT_TO_RIGHT",
                    "output_entity_type": "place",
                },
                {
                    "relation_cue": "part of",
                    "predicate_family": "containment",
                    "predicate_subtype": "",
                    "traversal": "LEFT_TO_RIGHT",
                    "output_entity_type": "place",
                },
            ],
            "final_answer_type": "place",
        },
    })
    messages[1]["content"] = json.dumps(
        payload, ensure_ascii=False, sort_keys=True
    )
    return messages


def _repair_unique_traversals(
    raw_payload: Mapping[str, Any],
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Reverse only endpoint-incompatible traversals with one valid opposite."""
    payload = copy.deepcopy(dict(raw_payload))
    if str(payload.get("decision") or "").strip().upper() != "PLAN":
        return payload, []
    start = payload.get("start_entity")
    steps = payload.get("steps")
    if not isinstance(start, Mapping) or not isinstance(steps, list):
        return payload, []

    current_type = str(start.get("semantic_type") or "").strip().casefold()
    repairs: list[Dict[str, Any]] = []
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, Mapping):
            break
        step = dict(raw_step)
        cue = str(step.get("relation_cue") or "")
        proposed_family = str(step.get("predicate_family") or "").casefold()
        family, _ = _schema_link_family(cue, proposed_family)
        spec = PREDICATE_ENDPOINT_SCHEMA.get(family)
        traversal = str(step.get("traversal") or "").strip().upper()
        output_type = str(step.get("output_entity_type") or "").strip().casefold()
        if spec is None or traversal not in {"LEFT_TO_RIGHT", "RIGHT_TO_LEFT"}:
            break

        left_to_right = traversal == "LEFT_TO_RIGHT"
        declared_input = spec["left_types"] if left_to_right else spec["right_types"]
        declared_output = spec["right_types"] if left_to_right else spec["left_types"]
        opposite_input = spec["right_types"] if left_to_right else spec["left_types"]
        opposite_output = spec["left_types"] if left_to_right else spec["right_types"]
        declared_valid = bool(
            _type_compatible(current_type, declared_input)
            and _type_compatible(output_type, declared_output)
        )
        opposite_valid = bool(
            _type_compatible(current_type, opposite_input)
            and _type_compatible(output_type, opposite_output)
        )
        if not declared_valid and opposite_valid:
            repaired = "RIGHT_TO_LEFT" if left_to_right else "LEFT_TO_RIGHT"
            step["traversal"] = repaired
            steps[index] = step
            repairs.append({
                "step": index,
                "predicate_family": family,
                "input_type": current_type,
                "output_type": output_type,
                "original_traversal": traversal,
                "repaired_traversal": repaired,
                "basis": "unique_endpoint_type_compatibility",
            })
        current_type = output_type
    return payload, repairs


def lower_semantic_v3_13_skeleton(
    question: str,
    raw_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply the unique direction repair, then reuse the strict V3.12 lowerer."""
    payload, repairs = _repair_unique_traversals(raw_payload)
    result = lower_semantic_v3_12_skeleton(question, payload)
    result["version"] = SEMANTIC_V3_13_PIPELINE_VERSION
    result["deterministic_traversal_repairs"] = repairs
    if isinstance(result.get("plan"), Mapping):
        plan = copy.deepcopy(dict(result["plan"]))
        plan["semantic_skeleton_version"] = SEMANTIC_V3_13_PROMPT_VERSION
        plan["semantic_skeleton_lowering_version"] = SEMANTIC_V3_13_PIPELINE_VERSION
        plan["deterministic_traversal_repairs"] = repairs
        result["plan"] = plan
    return result
