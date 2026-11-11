"""Refined start-grounding prompt for titled proper names versus role descriptions."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .evidence_query_compiler_semantic_v3_4_review_utils import (
    build_v3_4_candidate_facts,
)


SEMANTIC_V3_9_GROUNDING_PROMPT_VERSION = "start_entity_grounding_gate_v3_9"
SEMANTIC_V3_9_GROUNDING_CACHE_VERSION = "semantic_v3_9_grounding_cache_v1"


def build_v3_9_grounding_messages(
    question: str, plan: Mapping[str, Any]
) -> list[dict[str, str]]:
    facts = build_v3_4_candidate_facts(plan)
    system = (
        "Classify only whether the displayed start mention names one explicit entity or is a "
        "description whose bearer must first be resolved. Return exactly one JSON field "
        "start_entity_grounding and nothing else. Use NAMED_INSTANCE for proper names and "
        "titles of individual works, organizations, places, or people. A courtesy, royal, "
        "military, religious, or professional title followed by a person's explicit given or "
        "family name remains a named person, for example Prince Dmitri Alexandrovich of Russia, "
        "Dr Alice Smith, or General George Patton. Use DESCRIPTION_OR_ROLE for a bare office, "
        "relational description, or unbound title whose holder must be found, for example the "
        "director of Inception, the president of a country, or Earl of Merioneth when no personal "
        "name is supplied. Use UNKNOWN only when the wording is insufficient. Do not use world "
        "knowledge and do not answer the question."
    )
    payload = {
        "protocol": SEMANTIC_V3_9_GROUNDING_PROMPT_VERSION,
        "cache_contract": SEMANTIC_V3_9_GROUNDING_CACHE_VERSION,
        "question": str(question or "").strip(),
        "start_entities": facts.get("start_entities") or [],
        "required_exact_output": {
            "start_entity_grounding": "NAMED_INSTANCE|DESCRIPTION_OR_ROLE|UNKNOWN"
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]
