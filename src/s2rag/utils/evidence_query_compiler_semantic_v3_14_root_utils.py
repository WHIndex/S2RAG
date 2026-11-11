"""Executable-root contract for the experimental semantic compiler.

V3.12 verifies that a start mention occurs in the question, but occurrence is
not enough for execution: a phrase such as ``the actor who played X`` still
denotes the result of an omitted relation.  This module adds a deliberately
conservative, question-only boundary.  It can normalize an explicitly quoted
title, certify a surface that looks like a named instance, or reject an
obviously unresolved relational description.  It never invents an entity or
adds a missing relation.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Mapping

from .evidence_query_compiler_semantic_v3_12_skeleton_utils import (
    lower_semantic_v3_12_skeleton,
)


SEMANTIC_V3_14_ROOT_CONTRACT_VERSION = (
    "semantic_executable_root_contract_v3_14_20261101"
)

_QUOTED_SPAN_RE = re.compile(r'["\u201c\u201d]([^"\u201c\u201d]{2,160})["\u201c\u201d]')
_RELATIONAL_MARKER_RE = re.compile(
    r"\b(?:who|whose|whom|which|that|where|located|adjacent|containing|"
    r"contains|performed|played|portraying|produced|directed|wrote|written|"
    r"created|founded|named\s+after|born\s+in|died\s+in)\b",
    re.IGNORECASE,
)
_POSSESSIVE_DERIVED_VALUE_RE = re.compile(
    r"(?:['\u2019]s)\s+(?:birth|birthplace|birth\s+(?:place|city|country|state)|"
    r"death|death\s+(?:place|city|country|date)|spouse|husband|wife|child|"
    r"parent|mother|father|son|daughter|place|country|county|city|state|district|"
    r"occupation|employer|work|director|producer|author|performer)\b",
    re.IGNORECASE,
)
_GENERIC_HEAD_RE = re.compile(
    r"^(?:the|a|an|what|which|who)\s+(?:(?:notable|famous|historical|"
    r"british|american|italian|french|russian|largest|most\s+populous|"
    r"leading|professional|fictional|former|current)\s+){0,3}"
    r"(?:person|individual|man|woman|actor|actress|singer|artist|writer|"
    r"author|director|producer|performer|navigator|founder|plaintiff|lover|"
    r"king|queen|president|minister|county|country|city|state|district|"
    r"province|town|river|group|company|organization|military|capital)\b",
    re.IGNORECASE,
)
_TYPED_QUOTED_PREFIX_RE = re.compile(
    r"^(?:the\s+)?(?:opera|film|movie|song|album|book|novel|series|play|"
    r"painting|work)\s+[\"\u201c\u201d]",
    re.IGNORECASE,
)
_NAME_TOKEN_RE = re.compile(r"^[A-Z\u00c0-\u00de][\w\u00c0-\u024f'\u2019.-]*$")


def assess_executable_root_mention(
    question: str,
    mention: str,
    semantic_type: str,
) -> Dict[str, Any]:
    """Return a conservative execution contract for one start mention."""
    question = str(question or "")
    mention = str(mention or "").strip()
    semantic_type = str(semantic_type or "").strip().casefold()
    if not mention or mention.casefold() not in question.casefold():
        return {
            "version": SEMANTIC_V3_14_ROOT_CONTRACT_VERSION,
            "decision": "REJECT",
            "reason": "root_not_grounded_in_question",
            "original_mention": mention,
            "normalized_mention": "",
        }

    quoted = _QUOTED_SPAN_RE.findall(mention)
    if quoted and _TYPED_QUOTED_PREFIX_RE.search(mention):
        normalized = str(quoted[-1]).strip()
        return {
            "version": SEMANTIC_V3_14_ROOT_CONTRACT_VERSION,
            "decision": "PASS",
            "reason": "explicit_typed_quoted_title",
            "original_mention": mention,
            "normalized_mention": normalized,
            "semantic_type": semantic_type,
        }

    rejection_reasons = []
    if _POSSESSIVE_DERIVED_VALUE_RE.search(mention):
        rejection_reasons.append("possessive_derived_value")
    if _RELATIONAL_MARKER_RE.search(mention):
        rejection_reasons.append("embedded_relational_clause")
    if _GENERIC_HEAD_RE.search(mention):
        rejection_reasons.append("generic_relational_head")
    if rejection_reasons:
        return {
            "version": SEMANTIC_V3_14_ROOT_CONTRACT_VERSION,
            "decision": "REJECT",
            "reason": "unresolved_descriptive_root",
            "signals": rejection_reasons,
            "original_mention": mention,
            "normalized_mention": "",
            "semantic_type": semantic_type,
        }

    tokens = [token.strip("()[]{}.,:;!?") for token in mention.split()]
    name_tokens = [token for token in tokens if token and _NAME_TOKEN_RE.match(token)]
    if name_tokens:
        return {
            "version": SEMANTIC_V3_14_ROOT_CONTRACT_VERSION,
            "decision": "PASS",
            "reason": "explicit_named_surface",
            "original_mention": mention,
            "normalized_mention": mention,
            "semantic_type": semantic_type,
        }

    return {
        "version": SEMANTIC_V3_14_ROOT_CONTRACT_VERSION,
        "decision": "REVIEW",
        "reason": "root_surface_not_deterministically_certified",
        "original_mention": mention,
        "normalized_mention": "",
        "semantic_type": semantic_type,
    }


def lower_semantic_v3_14_root_contract(
    question: str,
    raw_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply the root contract before reusing the frozen V3.12 lowerer."""
    payload = copy.deepcopy(dict(raw_payload or {}))
    if str(payload.get("decision") or "").strip().upper() != "PLAN":
        result = lower_semantic_v3_12_skeleton(question, payload)
        result["executable_root_contract"] = None
        return result

    start = payload.get("start_entity")
    if not isinstance(start, Mapping):
        result = lower_semantic_v3_12_skeleton(question, payload)
        result["executable_root_contract"] = None
        return result

    contract = assess_executable_root_mention(
        question,
        str(start.get("mention") or ""),
        str(start.get("semantic_type") or ""),
    )
    if contract["decision"] != "PASS":
        return {
            "decision": "PLAN",
            "reason": str(contract["reason"]),
            "protocol_valid": True,
            "static_valid": False,
            "semantic_payload": None,
            "analysis": None,
            "plan": None,
            "executable_root_contract": contract,
        }

    normalized = str(contract.get("normalized_mention") or "").strip()
    payload["start_entity"] = dict(start)
    payload["start_entity"]["mention"] = normalized
    result = lower_semantic_v3_12_skeleton(question, payload)
    result["executable_root_contract"] = contract
    plan = result.get("plan")
    if isinstance(plan, dict):
        plan["executable_root_contract"] = contract
    return result
