"""Query-only risk features for deterministic repair admission.

The helpers in this module deliberately avoid gold answers and dataset names.
They derive a relation determinacy class, a small hierarchy of calibration
keys, and provenance-preserving answer projections from executable query IR.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TRAILING_PAREN_RE = re.compile(r"^(?P<base>.+?)\s*\((?P<alternate>[^()]*)\)\s*$")

_ANTI_JOIN_MARKERS = {
    "does_not",
    "did_not",
    "not",
    "without",
    "exclude",
    "excluded",
    "except",
}
_TEMPORAL_OR_MULTIVALUED_MARKERS = {
    "affiliate",
    "affiliated",
    "educated",
    "employ",
    "employed",
    "joined",
    "member",
    "membership",
    "operate",
    "operated",
    "operator",
    "own",
    "owned",
    "owner",
    "play",
    "played",
    "served",
    "sign",
    "signed",
    "team",
    "work",
    "worked",
}
_MEASUREMENT_MARKERS = {
    "age",
    "area",
    "distance",
    "duration",
    "height",
    "income",
    "length",
    "mass",
    "measurement",
    "size",
    "speed",
    "temperature",
    "volume",
    "weight",
}
_UNIT_MARKERS = {
    "c",
    "cm",
    "f",
    "feet",
    "ft",
    "in",
    "inch",
    "inches",
    "kg",
    "kilometer",
    "kilometers",
    "km",
    "lb",
    "lbs",
    "m",
    "meter",
    "meters",
    "mi",
    "mile",
    "miles",
    "mm",
    "percent",
}


def normalize_component(value: Any) -> str:
    tokens = _TOKEN_RE.findall(str(value or "").casefold())
    return "_".join(tokens) or "unknown"


def normalized_query_cluster(question: Any) -> str:
    """Collapse formatting variants of the same atomic query template."""

    return " ".join(_TOKEN_RE.findall(str(question or "").casefold()))


def _semantic_tokens(query_ir: Mapping[str, Any]) -> set[str]:
    values = [
        query_ir.get("relation"),
        query_ir.get("output_role"),
        query_ir.get("granularity"),
    ]
    values.extend(query_ir.get("constraints") or [])
    return {
        token
        for value in values
        for token in _TOKEN_RE.findall(str(value or "").casefold())
    }


def relation_determinacy_class(query_ir: Mapping[str, Any]) -> str:
    """Classify whether one binding should determine one stable output.

    This is intentionally conservative. Temporal affiliations, ownership,
    employment, and participation can legitimately have multiple values even
    when a compiler labels the requested cardinality as ``single``.
    """

    relation = normalize_component(query_ir.get("relation"))
    relation_tokens = set(relation.split("_"))
    tokens = _semantic_tokens(query_ir)
    if "does_not" in relation or "did_not" in relation:
        return "anti_join"
    if relation_tokens & _ANTI_JOIN_MARKERS or tokens & _ANTI_JOIN_MARKERS:
        return "anti_join"
    if tokens & _TEMPORAL_OR_MULTIVALUED_MARKERS:
        return "temporal_or_multivalued"
    return "functional_candidate"


def granularity_class(query_ir: Mapping[str, Any]) -> str:
    tokens = _semantic_tokens(query_ir)
    output_type = normalize_component(query_ir.get("output_type"))
    if tokens & _MEASUREMENT_MARKERS:
        return "measurement"
    if output_type == "date" or tokens & {"date", "day", "month", "year", "time"}:
        return "temporal"
    if output_type == "person" or "individual" in tokens:
        return "individual"
    if output_type == "location":
        return "location"
    if output_type in {"entity", "event", "organization", "work"}:
        return "named_entity"
    return "other"


def risk_hierarchy_keys(query_ir: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return most-specific to broadest gold-free calibration keys."""

    output_type = normalize_component(query_ir.get("output_type"))
    determinacy = relation_determinacy_class(query_ir)
    stage = normalize_component(query_ir.get("slot_plan_stage"))
    granularity = granularity_class(query_ir)
    dimensions = [
        (
            "type_determinacy_stage_granularity",
            (output_type, determinacy, stage, granularity),
        ),
        ("type_determinacy_stage", (output_type, determinacy, stage)),
        ("type_determinacy", (output_type, determinacy)),
        ("output_type", (output_type,)),
    ]
    return [
        {
            "level": level,
            "key": f"{level}|{'|'.join(values)}",
        }
        for level, values in dimensions
    ]


def canonicalize_provenance_safe_answer(
    answer: Any,
    query_ir: Mapping[str, Any],
    support_quote: Any,
) -> dict[str, Any]:
    """Apply a narrow projection while retaining the original surface.

    Only a trailing alternate-unit parenthetical is removed. Both the original
    answer and projected base must occur in the exact support quote, so the
    operation cannot synthesize a value that is absent from provenance.
    """

    original = " ".join(str(answer or "").split())
    quote = " ".join(str(support_quote or "").split())
    result = {
        "answer": original,
        "original_answer": original,
        "changed": False,
        "rule": "identity",
    }
    match = _TRAILING_PAREN_RE.match(original)
    if not match or not quote:
        return result

    base = match.group("base").strip(" ,;")
    alternate = match.group("alternate").strip()
    semantic_tokens = _semantic_tokens(query_ir)
    alternate_tokens = set(_TOKEN_RE.findall(alternate.casefold()))
    measurement_semantics = bool(semantic_tokens & _MEASUREMENT_MARKERS)
    alternate_unit = bool(alternate_tokens & _UNIT_MARKERS) and bool(
        re.search(r"\d", alternate)
    )
    quote_folded = quote.casefold()
    if (
        base
        and (measurement_semantics or alternate_unit)
        and original.casefold() in quote_folded
        and base.casefold() in quote_folded
    ):
        result.update({
            "answer": base,
            "changed": True,
            "rule": "drop_provenance_present_alternate_unit_parenthetical",
        })
    return result


def select_hierarchical_risk_group(
    gate_payload: Mapping[str, Any],
    query_ir: Mapping[str, Any],
) -> dict[str, Any]:
    """Choose the first sufficiently supported group, without gold access."""

    min_clusters = int(gate_payload.get("min_unique_clusters", 1) or 1)
    groups = dict(gate_payload.get("groups") or {})
    determinacy = relation_determinacy_class(query_ir)
    if determinacy == "anti_join":
        return {
            "admitted": False,
            "decision_reason": "anti_join_requires_explicit_operator",
            "determinacy": determinacy,
        }
    for candidate in risk_hierarchy_keys(query_ir):




        if (
            candidate["level"] == "output_type"
            and determinacy != "functional_candidate"
        ):
            continue
        stats = dict(groups.get(candidate["key"]) or {})
        if int(stats.get("unique_clusters", 0) or 0) < min_clusters:
            continue
        stats.setdefault("level", candidate["level"])
        stats.setdefault("key", candidate["key"])
        stats["decision_reason"] = (
            "risk_bound_admitted" if stats.get("admitted") else "risk_bound_veto"
        )
        return stats
    return {
        "admitted": False,
        "decision_reason": "insufficient_cluster_support",
        "determinacy": determinacy,
    }
