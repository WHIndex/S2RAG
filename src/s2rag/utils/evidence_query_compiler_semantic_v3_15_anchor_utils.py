"""Anchor-first lowering for a narrow class of possessive descriptions.

The V3.14 executable-root contract correctly rejects a surface such as
``Erik Hort's birthplace``: that phrase is a derived value, not an executable
instance binding.  V3.15 repairs only the deterministic subset where the
possessor is itself an executable named surface and the possessive suffix has
one unambiguous endpoint contract.  It rewrites the semantic skeleton before
delegating all type, dependency, coverage, and physical-plan checks to the
frozen V3.14/V3.12 lowering stack.

This module deliberately does not resolve relative clauses or guess missing
relations.  Unsupported or ambiguous possessives retain the V3.14 rejection.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .evidence_query_compiler_semantic_v3_14_root_utils import (
    assess_executable_root_mention,
    lower_semantic_v3_14_root_contract,
)
from .relation_fact_join_utils import canonical_schema_predicate_families


SEMANTIC_V3_15_ANCHOR_CONTRACT_VERSION = (
    "semantic_anchor_first_possessive_v3_15_20261101"
)

_POSSESSIVE_RE = re.compile(
    r"^(?P<owner>.+?)(?:['\u2019]s)\s+(?P<relation>.+?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _AnchorEdge:
    family: str
    traversal: str
    input_type: str
    output_type: str
    subtype: str = ""






_POSSESSIVE_EDGES = {
    "birthplace": _AnchorEdge("birth", "LEFT_TO_RIGHT", "person", "place", "birth_place"),
    "birth place": _AnchorEdge("birth", "LEFT_TO_RIGHT", "person", "place", "birth_place"),
    "place of birth": _AnchorEdge("birth", "LEFT_TO_RIGHT", "person", "place", "birth_place"),
    "birth country": _AnchorEdge("birth", "LEFT_TO_RIGHT", "person", "place", "birth_place"),
    "place of death": _AnchorEdge("death", "LEFT_TO_RIGHT", "person", "place", "death_place"),
    "death place": _AnchorEdge("death", "LEFT_TO_RIGHT", "person", "place", "death_place"),
    "author": _AnchorEdge("authorship", "LEFT_TO_RIGHT", "work", "person"),
    "writer": _AnchorEdge("authorship", "LEFT_TO_RIGHT", "work", "person"),
    "director": _AnchorEdge("direction", "LEFT_TO_RIGHT", "work", "person"),
    "producer": _AnchorEdge("production", "LEFT_TO_RIGHT", "work", "person"),
    "performer": _AnchorEdge("performance", "LEFT_TO_RIGHT", "work", "person"),
    "actor": _AnchorEdge("performance", "LEFT_TO_RIGHT", "work", "person"),
    "spouse": _AnchorEdge("spouse_of", "LEFT_TO_RIGHT", "person", "person"),
    "husband": _AnchorEdge("spouse_of", "LEFT_TO_RIGHT", "person", "person"),
    "wife": _AnchorEdge("spouse_of", "LEFT_TO_RIGHT", "person", "person"),
    "father": _AnchorEdge("father_of", "RIGHT_TO_LEFT", "person", "person"),
    "mother": _AnchorEdge("mother_of", "RIGHT_TO_LEFT", "person", "person"),
    "parent": _AnchorEdge("parent_of", "RIGHT_TO_LEFT", "person", "person"),
    "child": _AnchorEdge("child_of", "RIGHT_TO_LEFT", "person", "person"),
    "son": _AnchorEdge("son_of", "RIGHT_TO_LEFT", "person", "person"),
    "daughter": _AnchorEdge("daughter_of", "RIGHT_TO_LEFT", "person", "person"),
}

_DIRECTIONAL_KINSHIP_FAMILIES = {
    "mother_of",
    "father_of",
    "parent_of",
    "daughter_of",
    "son_of",
    "child_of",
}
_PROJECTED_KINSHIP_RE = re.compile(
    r"\b(?:who|what)\s+is\s+(?:the\s+)?(?:other\s+)?"
    r"(?:mother|father|parent|daughter|son|child)\s+of\b",
    re.IGNORECASE,
)
_OTHER_ENTITY_PROJECTION_RE = re.compile(
    r"\b(?:who|what)\s+is\s+(?:the\s+)?other\s+"
    r"(?:mother|father|parent|daughter|son|child|spouse|husband|wife)\s+of\b",
    re.IGNORECASE,
)


def _normalize_suffix(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _validate_existing_chain(
    question: str,
    steps: list[Any],
    anchor_edge: _AnchorEdge,
) -> Dict[str, Any]:
    """Check that a newly admitted root does not expose an unsafe old chain."""
    if _OTHER_ENTITY_PROJECTION_RE.search(str(question or "")):
        return {
            "valid": False,
            "reason": "unsupported_other_entity_exclusion",
        }
    if steps and isinstance(steps[0], Mapping):
        first_family = str(
            steps[0].get("predicate_family") or ""
        ).strip().casefold()
        first_traversal = str(
            steps[0].get("traversal") or ""
        ).strip().upper()
        if (
            anchor_edge.family in {"mother_of", "father_of", "parent_of"}
            and first_family == "child_of"
            and first_traversal == "RIGHT_TO_LEFT"
        ):
            return {
                "valid": False,
                "reason": "unsupported_root_identity_exclusion",
                "detail": (
                    "parent-to-child traversal can reproduce the original "
                    "child and requires answer != root"
                ),
            }
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            return {"valid": False, "reason": f"invalid_existing_step_{index}"}
        cue = str(step.get("relation_cue") or "").strip()
        family = str(step.get("predicate_family") or "").strip().casefold()
        canonical = set(canonical_schema_predicate_families(cue))
        if family not in canonical:
            return {
                "valid": False,
                "reason": f"existing_cue_family_mismatch_{index}",
                "step": index,
                "relation_cue": cue,
                "proposed_family": family,
                "canonical_families": sorted(canonical),
            }
        traversal = str(step.get("traversal") or "").strip().upper()
        if (
            family in _DIRECTIONAL_KINSHIP_FAMILIES
            and _PROJECTED_KINSHIP_RE.search(str(question or ""))
            and traversal != "RIGHT_TO_LEFT"
        ):
            return {
                "valid": False,
                "reason": f"existing_kinship_direction_mismatch_{index}",
                "step": index,
                "relation_cue": cue,
                "predicate_family": family,
                "expected_traversal": "RIGHT_TO_LEFT",
                "actual_traversal": traversal,
            }
    return {"valid": True, "reason": "validated"}


def rewrite_possessive_root_to_anchor(
    question: str,
    raw_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a certified rewrite record and, when safe, a new skeleton."""
    payload = copy.deepcopy(dict(raw_payload or {}))
    base = {
        "version": SEMANTIC_V3_15_ANCHOR_CONTRACT_VERSION,
        "decision": "NO_REWRITE",
        "reason": "not_applicable",
        "original_root": "",
        "anchor": "",
        "relation_cue": "",
        "rewritten_payload": None,
    }
    if str(payload.get("decision") or "").strip().upper() != "PLAN":
        base["reason"] = "payload_not_plan"
        return base
    start = payload.get("start_entity")
    steps = payload.get("steps")
    if not isinstance(start, Mapping) or not isinstance(steps, list):
        base["reason"] = "invalid_skeleton_shape"
        return base

    mention = str(start.get("mention") or "").strip()
    base["original_root"] = mention
    match = _POSSESSIVE_RE.match(mention)
    if not match:
        base["reason"] = "root_not_possessive"
        return base

    owner = str(match.group("owner") or "").strip()
    relation_cue = str(match.group("relation") or "").strip()
    base["anchor"] = owner
    base["relation_cue"] = relation_cue
    edge: Optional[_AnchorEdge] = _POSSESSIVE_EDGES.get(
        _normalize_suffix(relation_cue)
    )
    if edge is None:
        base["reason"] = "unsupported_or_ambiguous_possessive_relation"
        return base

    owner_contract = assess_executable_root_mention(
        question, owner, edge.input_type
    )
    base["anchor_contract"] = owner_contract
    if owner_contract.get("decision") != "PASS":
        base["reason"] = "possessor_not_executable_anchor"
        return base

    derived_type = str(start.get("semantic_type") or "").strip().casefold()
    if derived_type != edge.output_type:
        base["reason"] = "derived_root_type_mismatch"
        base["expected_derived_type"] = edge.output_type
        base["actual_derived_type"] = derived_type
        return base
    if len(steps) >= 4:
        base["reason"] = "rewritten_plan_exceeds_step_limit"
        return base

    chain_contract = _validate_existing_chain(question, steps, edge)
    base["existing_chain_contract"] = chain_contract
    if not chain_contract["valid"]:
        base["reason"] = str(chain_contract["reason"])
        return base

    normalized_owner = str(owner_contract.get("normalized_mention") or owner)
    rewritten = copy.deepcopy(payload)
    rewritten["start_entity"] = {
        "mention": normalized_owner,
        "semantic_type": edge.input_type,
    }
    rewritten["steps"] = [
        {
            "output_entity_type": edge.output_type,
            "predicate_family": edge.family,
            "predicate_subtype": edge.subtype,
            "relation_cue": relation_cue,
            "traversal": edge.traversal,
        },
        *copy.deepcopy(steps),
    ]
    base.update(
        {
            "decision": "REWRITE",
            "reason": "certified_possessive_anchor_edge",
            "edge": {
                "predicate_family": edge.family,
                "predicate_subtype": edge.subtype,
                "traversal": edge.traversal,
                "input_type": edge.input_type,
                "output_type": edge.output_type,
            },
            "rewritten_payload": rewritten,
        }
    )
    return base


def lower_semantic_v3_15_anchor_first(
    question: str,
    raw_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Rewrite a safe derived root, then apply the strict V3.14 lowerer."""
    rewrite = rewrite_possessive_root_to_anchor(question, raw_payload)
    payload = rewrite.get("rewritten_payload")
    result = lower_semantic_v3_14_root_contract(
        question,
        payload if isinstance(payload, Mapping) else raw_payload,
    )
    result["anchor_first_rewrite"] = {
        key: value for key, value in rewrite.items() if key != "rewritten_payload"
    }
    plan = result.get("plan")
    if isinstance(plan, dict):
        plan["anchor_first_rewrite"] = result["anchor_first_rewrite"]
        if (
            rewrite.get("decision") == "REWRITE"
            and bool(result.get("static_valid"))
            and bool((rewrite.get("existing_chain_contract") or {}).get("valid"))
        ):
            certificate = {
                "version": SEMANTIC_V3_15_ANCHOR_CONTRACT_VERSION,
                "status": "INDEPENDENT_CONTRACT_VALIDATED",
                "basis": "deterministic_anchor_first_complete_chain_contract",
                "independent_question_contract": True,
                "human_reviewed": False,
                "auto_execution_eligible": True,
                "scope": (
                    "explicit_named_possessor+catalogued_possessive_edge+"
                    "cue_family_alignment+endpoint_direction_and_type+"
                    "strict_static_plan_validation"
                ),
            }
            plan["anchor_first_semantic_certificate"] = certificate
            current_validation = plan.get("question_semantic_validation")
            if not bool(
                isinstance(current_validation, Mapping)
                and current_validation.get("auto_execution_eligible")
            ):
                plan["question_semantic_validation"] = dict(certificate)
    return result
