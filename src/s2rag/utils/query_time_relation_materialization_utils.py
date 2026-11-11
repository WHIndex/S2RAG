"""Source-constrained query-time materialization for missing relation tuples.

The model is only a candidate extractor.  Every returned value must be tied to
one supplied source and an exact source span before it becomes an UNVERIFIED
query-local tuple.  The evidence executor's existing VERIFY action remains the
only component allowed to mark that tuple SUPPORTED.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .late_bound_textual_relation_utils import parse_json_object
from .relation_fact_join_utils import (
    SCHEMA_PREDICATE_FAMILY_TERMS,
    TERMINAL_TYPE_INCOMPATIBLE,
    _completion_fact_item,
    _operator_requires_role_direction,
    canonical_entity_key,
    canonical_predicate_contract,
    canonical_schema_predicate_subtypes,
    terminal_value_contract_compatibility,
)


BINDING_CONDITIONED_TYPED_MATERIALIZATION_VERSION = (
    "binding_conditioned_typed_materialization_v4"
)
BINDING_IDENTITY_CONTRACT_VERSION = "source_anchored_binding_identity_v3_1"
WINDOWED_EXTRACTIVE_TOPK_VERSION = "windowed_extractive_topk_v2"
SOURCE_ANCHORED_RELATION_CANONICALIZATION_VERSION = (
    "source_anchored_directional_relation_v2"
)


def build_query_time_relation_materialization_messages(
    *,
    bound_surface: str,
    operator: Mapping[str, Any],
    documents: Sequence[Mapping[str, Any]],
    source_windows: Optional[Sequence[Mapping[str, Any]]] = None,
    max_candidates: int = 3,
    enable_directional_surface_canonicalization: bool = False,
) -> list[dict[str, str]]:
    """Build one batched extraction request without exposing a QA gold answer."""

    source_blocks = []
    windowed = source_windows is not None
    if windowed:
        for window in source_windows or []:
            source_blocks.append(
                "\n".join([
                    f"WINDOW_ID: {str(window.get('window_id') or '')}",
                    f"SOURCE_ID: {int(window.get('doc_id', -1))}",
                    f"TITLE: {str(window.get('title') or '')}",
                    "SOURCE_WINDOW:",
                    str(window.get("content") or ""),
                ])
            )
    else:
        for document in documents:
            source_blocks.append(
                "\n".join([
                    f"SOURCE_ID: {int(document.get('doc_id', -1))}",
                    f"TITLE: {str(document.get('title') or '')}",
                    "PASSAGE:",
                    str(document.get("content") or ""),
                ])
            )
    predicate_families = ", ".join(
        sorted(str(value) for value in operator.get("predicate_families") or [])
    ) or "unknown"
    relation_tokens = ", ".join(
        sorted(str(value) for value in operator.get("relation_tokens") or [])
    ) or "unknown"
    relation_surface_instruction = ""
    if enable_directional_surface_canonicalization:
        relation_surface_instruction = (
            " relation_phrase must be the shortest exact source phrase that "
            "expresses the relation and must not include the answer itself "
            "(for example, 'painting by', 'written by', 'performed by', or "
            "'mother of')."
        )
    if windowed:
        system = (
            "You perform one source-scoped relational lookup over independently "
            "ranked source windows. Evaluate every window separately and find values "
            "that fill the requested output role for the already bound entity. Use "
            "only explicit evidence inside that same window and never outside "
            "knowledge. Return at most one candidate per WINDOW_ID and at most "
            f"{max(1, int(max_candidates))} candidates total. Do not combine text "
            "from different windows. support_quote must be a verbatim contiguous "
            "span from the selected window; answer and relation_phrase must both be "
            "exact contiguous substrings of support_quote."
            + relation_surface_instruction
            + " If a window does not "
            "explicitly state the requested relation for the bound entity, omit it. "
            "Return one JSON object with key candidates. Each candidate has "
            "window_id, source_id, answer, support_quote, relation_phrase, and "
            "supported."
        )
    else:
        system = (
            "You perform one source-scoped relational lookup over supplied passages. "
            "Find values that fill the requested output role for the already bound "
            "entity. Use only explicit source evidence and never outside knowledge. "
            "For each source return at most one candidate. support_quote must be one "
            "verbatim contiguous sentence from that source; answer and relation_phrase "
            "must both be exact contiguous substrings of support_quote."
            + relation_surface_instruction
            + " If a source does "
            "not explicitly state the requested relation for the bound entity, omit it. "
            "Return one JSON object with key candidates. Each candidate has source_id, "
            "answer, support_quote, relation_phrase, and supported."
        )
    user = "\n".join([
        f"Bound entity: {str(bound_surface or '').strip()}",
        f"Requested canonical relation: {str(operator.get('canonical_relation') or '')}",
        f"Predicate families: {predicate_families}",
        f"Relation tokens: {relation_tokens}",
        f"Input role: {str(operator.get('input_role') or '')}",
        f"Output role: {str(operator.get('output_role') or '')}",
        f"Expected output type: {str(operator.get('output_type') or 'entity')}",
        "Ranked source windows:" if windowed else "Sources:",
        "\n\n---\n\n".join(source_blocks),
    ])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_materialization_candidates(value: Any) -> list[dict[str, Any]]:
    payload = parse_json_object(value)
    candidates = payload.get("candidates") or []
    if isinstance(candidates, Mapping):
        candidates = [candidates]
    return [dict(row) for row in candidates if isinstance(row, Mapping)]


def _locate_exact_source_span(source: str, quote: str) -> Optional[Tuple[int, int, str]]:
    """Locate a copied quote while tolerating only whitespace normalization."""

    source = str(source or "")
    quote = str(quote or "").strip()
    if not source or not quote:
        return None
    start = source.find(quote)
    if start >= 0:
        return start, start + len(quote), source[start : start + len(quote)]
    start = source.casefold().find(quote.casefold())
    if start >= 0:
        return start, start + len(quote), source[start : start + len(quote)]
    pieces = [re.escape(value) for value in re.split(r"\s+", quote) if value]
    if not pieces:
        return None
    match = re.search(r"\s+".join(pieces), source, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.start(), match.end(), source[match.start() : match.end()]


def _surface_pattern(value: Any) -> str:
    words = re.findall(r"[^\W_]+", str(value or ""), flags=re.UNICODE)
    if not words:
        return r"(?!)"
    return r"(?<!\w)" + r"[\W_]+".join(
        re.escape(word) for word in words
    ) + r"(?!\w)"


def source_surface_in_text(surface: str, text: str) -> bool:
    """Return whether ``surface`` occurs in source order in ``text``.

    Physical entity keys are bags of normalized tokens, which is useful for
    join lookup but invalid for source-span validation.  Comparing those keys
    as strings turns ``10 June 1819`` into ``10 1819 june`` and can reject a
    verbatim source mention.  Source support instead requires the original
    token order while tolerating punctuation and whitespace differences.
    """

    return bool(
        str(surface or "").strip()
        and str(text or "").strip()
        and re.search(_surface_pattern(surface), str(text), re.IGNORECASE)
    )


def _surface_in_text(surface: str, text: str) -> bool:
    return source_surface_in_text(surface, text)


def _source_anchored_relation_surface(
    *,
    quote: str,
    answer: str,
    operator: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover a typed relation phrase from an exact source construction.

    The model occasionally copies an over-wide phrase such as ``by John
    Everett Millais`` or emits a canonical label that is not literally present
    in the source.  This helper may replace that phrase only when the answer is
    immediately governed by a source construction compatible with the current
    obligation.  It never uses the QA gold answer or text outside ``quote``.
    """

    expected = {
        str(value).casefold()
        for value in operator.get("predicate_families") or []
        if str(value).strip()
    }
    if not quote or not answer or not expected:
        return {"matched": False}

    quote_context = quote.casefold()
    contextual_patterns: list[tuple[str, str, str]] = []

    def add(family: str, reason: str, patterns: Sequence[str]) -> None:
        if family not in expected:
            return
        contextual_patterns.extend((family, reason, pattern) for pattern in patterns)

    add("visual_art", "source_visual_art_by_construction", (
        r"painting\s+by", r"painted\s+by", r"portrait\s+by",
        r"artwork\s+by", r"(?:oil|work)\s+by\s+(?:the\s+)?(?:artist|painter)",
    ))
    add("direction", "source_directed_by_construction", (r"directed\s+by",))
    add("production", "source_produced_by_construction", (r"produced\s+by",))
    add("performance", "source_performed_by_construction", (
        r"performed\s+by", r"recorded\s+by", r"sung\s+by", r"song\s+by",
    ))
    if "performance" in expected and re.search(
        r"\b(?:album|single|record|recording|song|track)\b",
        quote_context,
    ):
        contextual_patterns.extend((
            (
                "performance",
                "source_recording_artist_construction",
                r"recording\s+artist",
            ),
            (
                "performance",
                "source_recording_by_construction",
                r"(?:album|single|record|recording|song|track)\s+by",
            ),
            (
                "performance",
                "source_recording_singer_construction",
                r"(?:singer|singer-songwriter|vocalist)",
            ),
            (
                "performance",
                "source_music_artist_construction",
                r"(?:(?:pop|rock|country|folk|jazz|classical|indie)\s+)?"
                r"(?:music\s+)?artist",
            ),
        ))
    add("creation", "source_created_by_construction", (
        r"created\s+by", r"made\s+by",
    ))
    add("design", "source_designed_by_construction", (
        r"designed\s+by", r"architect(?:ed)?\s+by",
    ))
    add("authorship", "source_authorship_by_construction", (
        r"written\s+by", r"authored\s+by", r"book\s+by", r"novel\s+by",
        r"poem\s+by", r"essay\s+by", r"polemical\s+work\s+by",
    ))
    if "authorship" in expected and re.search(
        r"\b(?:encyclical|apostolic\s+letter|papal\s+letter)\b",
        quote_context,
    ):
        contextual_patterns.extend((
            (
                "authorship",
                "source_papal_document_issued_by_construction",
                r"issued\s+by(?:\s+(?:Pope|Saint))?",
            ),
            (
                "authorship",
                "source_papal_document_promulgated_by_construction",
                r"promulgated\s+by(?:\s+(?:Pope|Saint))?",
            ),
            (
                "authorship",
                "source_papal_document_drafted_by_construction",
                r"drafted\s+by(?:\s+(?:Pope|Saint))?",
            ),
        ))
    if "composition" in expected and re.search(
        r"\b(?:song|music|musical|composition|track|score|melody|sonata|"
        r"symphony|concerto|opera|oratorio|cantata|quartet|quintet|"
        r"mass|motet)\b",
        quote_context,
    ):
        contextual_patterns.extend((
            ("composition", "source_music_written_by_construction", r"written\s+by"),
            ("composition", "source_composed_by_construction", r"composed\s+by"),
            ("composition", "source_music_by_construction", r"music\s+by"),
        ))
    elif "composition" in expected:
        contextual_patterns.extend((
            ("composition", "source_composed_by_construction", r"composed\s+by"),
            ("composition", "source_music_by_construction", r"music\s+by"),
        ))



    if "child_of" in expected:
        contextual_patterns.extend((
            ("child_of", "source_parent_to_child_construction", r"mother\s+of"),
            ("child_of", "source_parent_to_child_construction", r"father\s+of"),
            ("child_of", "source_parent_to_child_construction", r"parent\s+of"),
        ))

    answer_pattern = re.compile(_surface_pattern(answer), re.IGNORECASE)
    trailing_separator = r"(?:\s|[,;:()\[\]\-\u2013\u2014])*"
    for answer_match in answer_pattern.finditer(quote):
        prefix_start = max(0, answer_match.start() - 180)
        prefix = quote[prefix_start : answer_match.start()]
        for family, reason, raw_pattern in contextual_patterns:
            relation_match = re.search(
                rf"(?P<relation>{raw_pattern}){trailing_separator}$",
                prefix,
                re.IGNORECASE,
            )
            if relation_match is None:
                continue
            surface = relation_match.group("relation")
            return {
                "matched": True,
                "relation_phrase": surface,
                "predicate_families": [family],
                "reason": reason,
                "version": SOURCE_ANCHORED_RELATION_CANONICALIZATION_VERSION,
            }
    return {"matched": False}


_NON_TERMINAL_ABBREVIATIONS = {
    "dr", "e.g", "etc", "i.e", "jr", "mr", "mrs", "ms", "mt", "prof",
    "sr", "st", "vs",
}


def _protected_sentence_period(text: str, period_index: int) -> bool:
    """Return whether a period belongs to an abbreviation or personal name."""

    if period_index <= 0 or text[period_index] != ".":
        return False
    if (
        period_index + 1 < len(text)
        and text[period_index - 1].isdigit()
        and text[period_index + 1].isdigit()
    ):
        return True
    prefix = text[: period_index + 1]
    token_match = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*)\.$", prefix)
    token = token_match.group(1).casefold() if token_match else ""
    if token in _NON_TERMINAL_ABBREVIATIONS:
        return True

    suffix = text[period_index + 1 :]
    next_token = re.match(r"\s+([A-Z](?:\.|[a-z]))", suffix)
    if re.search(r"\b[A-Z]\.$", prefix) and next_token:
        return True
    if re.search(r"(?:\b[A-Z]\.\s*){2,}$", prefix):
        return True
    return False


def _exact_sentence_spans(text: str) -> list[Tuple[int, int, str]]:
    """Split source text without rewriting the spans sent to the extractor."""

    text = str(text or "")
    if not text.strip():
        return []
    boundaries = [0]
    for match in re.finditer(
        r"(?<=[.!?])\s+(?=(?:[\"'\(\[]*[A-Z0-9]))",
        text,
    ):
        punctuation_index = match.start() - 1
        if (
            punctuation_index >= 0
            and text[punctuation_index] == "."
            and _protected_sentence_period(text, punctuation_index)
        ):
            continue
        boundaries.append(match.end())
    boundaries.append(len(text))
    spans: list[Tuple[int, int, str]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        raw = text[left:right]
        value = raw.strip()
        if not value:
            continue
        start = text.find(value, left, right)
        spans.append((start, start + len(value), value))
    return spans


def _relation_window_terms(operator: Mapping[str, Any]) -> set[str]:
    values = [
        str(operator.get("canonical_relation") or "").replace("_", " "),
        *(str(value) for value in operator.get("relation_tokens") or []),
        *(str(value) for value in operator.get("predicate_families") or []),
    ]
    terms = {
        token.casefold()
        for value in values
        for token in re.findall(r"[^\W_]+", value, flags=re.UNICODE)
        if len(token) > 1
    }
    for family in operator.get("predicate_families") or []:
        terms.update(
            str(value).casefold()
            for value in SCHEMA_PREDICATE_FAMILY_TERMS.get(
                str(family).casefold(), set()
            )
        )
    return terms - {
        "and", "by", "entity", "for", "from", "in", "of", "on",
        "relation", "the", "to", "unknown", "with",
    }


def rank_binding_predicate_source_windows(
    *,
    bound_surface: str,
    operator: Mapping[str, Any],
    documents: Sequence[Mapping[str, Any]],
    max_windows: int = 8,
    max_sentences_per_window: int = 2,
) -> list[dict[str, Any]]:
    """Rank exact source windows for one bound predicate obligation.

    Ranking only controls which source text is shown to the candidate extractor.
    It never admits a tuple; the existing identity, source-support and typed
    obligation gates still decide whether a returned candidate is executable.
    """

    relation_terms = _relation_window_terms(operator)
    candidates: list[dict[str, Any]] = []
    width_limit = max(1, min(2, int(max_sentences_per_window)))
    for doc_rank, document in enumerate(documents):
        doc_id = int(document.get("doc_id", -1))
        title = str(document.get("title") or "")
        bound_key = canonical_entity_key(bound_surface)
        title_match = bool(
            bound_key
            and bound_key in {
                canonical_entity_key(title),
                _title_core_entity_key(title),
            }
        )
        spans = _exact_sentence_spans(str(document.get("content") or ""))
        for sentence_index in range(len(spans)):
            for width in range(1, width_limit + 1):
                end_index = sentence_index + width - 1
                if end_index >= len(spans):
                    continue
                start = spans[sentence_index][0]
                end = spans[end_index][1]
                content = str(document.get("content") or "")[start:end].strip()
                if not content:
                    continue
                tokens = {
                    token.casefold()
                    for token in re.findall(
                        r"[^\W_]+", content, flags=re.UNICODE
                    )
                }
                bound_match = source_surface_in_text(bound_surface, content)
                relation_hits = sorted(tokens & relation_terms)





                if (
                    "performance" in {
                        str(value).casefold()
                        for value in operator.get("predicate_families") or []
                    }
                    and re.search(
                        r"\b(?:album|single|record|recording|song|track)\b",
                        content,
                        flags=re.IGNORECASE,
                    )
                    and re.search(
                        r"\b(?:music\s+artist|recording\s+artist)\b",
                        content,
                        flags=re.IGNORECASE,
                    )
                ):
                    relation_hits = sorted(set(relation_hits) | {"music_artist"})
                score = (
                    8.0 * int(bound_match)
                    + 5.0 * int(title_match)
                    + 2.0 * len(relation_hits)
                    + 1.5 * int(bool(relation_hits) and (bound_match or title_match))
                    + 0.25 * int(width > 1 and bool(relation_hits))
                    - 0.05 * doc_rank
                    - 0.000001 * start
                )
                candidates.append({
                    "window_id": f"d{doc_id}:s{sentence_index}:n{width}",
                    "doc_id": doc_id,
                    "title": title,
                    "content": content,
                    "source_span_start": start,
                    "source_span_end": end,
                    "sentence_start": sentence_index,
                    "sentence_end": end_index,
                    "score": score,
                    "bound_surface_match": bool(bound_match),
                    "title_identity_match": bool(title_match),
                    "relation_term_hits": relation_hits,
                    "document_rank": doc_rank,
                })


    unique_candidates: list[dict[str, Any]] = []
    seen_content: set[tuple[int, str]] = set()
    for row in candidates:
        content_key = " ".join(str(row["content"]).casefold().split())
        key = (int(row["doc_id"]), content_key)
        if key in seen_content:
            continue
        seen_content.add(key)
        unique_candidates.append(row)
    candidates = unique_candidates
    candidates.sort(
        key=lambda row: (
            -float(row["score"]),
            int(row["document_rank"]),
            int(row["source_span_start"]),
            str(row["window_id"]),
        )
    )
    limit = max(1, int(max_windows))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()


    seen_docs: set[int] = set()
    for row in candidates:
        if int(row["doc_id"]) in seen_docs:
            continue
        selected.append(row)
        selected_ids.add(str(row["window_id"]))
        seen_docs.add(int(row["doc_id"]))
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for row in candidates:
            if str(row["window_id"]) in selected_ids:
                continue
            dominated = False
            for previous in selected:
                if int(previous["doc_id"]) != int(row["doc_id"]):
                    continue
                overlap = max(
                    0,
                    min(
                        int(previous["source_span_end"]),
                        int(row["source_span_end"]),
                    )
                    - max(
                        int(previous["source_span_start"]),
                        int(row["source_span_start"]),
                    ),
                )
                shorter = max(
                    1,
                    min(
                        int(previous["source_span_end"])
                        - int(previous["source_span_start"]),
                        int(row["source_span_end"])
                        - int(row["source_span_start"]),
                    ),
                )
                if overlap / shorter >= 0.8:
                    dominated = True
                    break
            if dominated:
                continue
            selected.append(row)
            selected_ids.add(str(row["window_id"]))
            if len(selected) >= limit:
                break
    selected.sort(
        key=lambda row: (
            -float(row["score"]),
            int(row["document_rank"]),
            int(row["source_span_start"]),
        )
    )
    for rank, row in enumerate(selected):
        row["window_rank"] = rank
    return selected


def merge_materialization_results(
    primary: Mapping[str, Any],
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge two independently validated materialization passes.

    This function only unions facts that have already passed the same source,
    identity, and typed-obligation gates. It does not admit a rejected tuple.
    """

    merged = dict(primary)
    facts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in (primary, fallback):
        for raw_fact in result.get("facts") or []:
            fact = dict(raw_fact)
            key = str(fact.get("tuple_key") or repr(fact.get("triple")))
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)
    merged["facts"] = facts
    merged["validations"] = [
        *list(primary.get("validations") or []),
        *list(fallback.get("validations") or []),
    ]
    for key in (
        "candidate_count",
        "binding_identity_admitted_count",
        "typed_contract_admitted_count",
    ):
        merged[key] = int(primary.get(key, 0) or 0) + int(
            fallback.get(key, 0) or 0
        )
    merged["accepted_count"] = len(facts)
    merged["binding_conditioned_typed_materialization"] = bool(
        primary.get("binding_conditioned_typed_materialization")
        or fallback.get("binding_conditioned_typed_materialization")
    )
    return merged


def should_run_whole_source_fallback(
    *,
    primary_accepted_count: int,
    execution_model_limit: int,
    max_passes_per_obligation: int,
) -> bool:
    """Return whether one obligation may spend a second QTRM model pass."""

    return bool(
        int(primary_accepted_count) <= 0
        and int(execution_model_limit) >= 2
        and int(max_passes_per_obligation) >= 2
    )


def _matches_any(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _title_core_entity_key(title: Any) -> str:
    """Return the entity key before a trailing disambiguation parenthetical."""

    core = _title_identity_surface(title)
    return canonical_entity_key(core)


def _surface_tokens(value: Any) -> list[str]:
    return [
        token.casefold()
        for token in re.findall(r"[^\W_]+", str(value or ""), flags=re.UNICODE)
        if token
    ]


def _title_identity_surface(title: Any) -> str:
    """Remove document-section suffixes while retaining the entity title."""

    core = re.sub(r"\s*\([^()]{1,120}\)\s*$", "", str(title or "")).strip()
    return re.split(r"\s+(?:-|\u2013|\u2014)\s+", core, maxsplit=1)[0].strip()


def _query_scaffold_binding(surface: Any) -> bool:
    """Identify compiler fragments that are not executable entity bindings."""

    value = " ".join(str(surface or "").casefold().split())
    return bool(
        not value
        or re.match(r"^(?:who|what|which|where|when|why|how)\b", value)
    )


def _source_title_expanded_alias_identity(
    *, surface: str, title: str, quote: str
) -> bool:
    """Prove an alias from the title entity's expanded lead mention.

    A short alias is accepted only when the source lead spells out both the
    page-title entity and the bound alias inside one uninterrupted name span,
    e.g. ``Beyonce Giselle Knowles-Carter``.  Merely seeing a surname elsewhere
    in a title-scoped passage is intentionally insufficient.
    """

    title_surface = _title_identity_surface(title)
    if not surface or not title_surface or not quote:
        return False
    lead = re.split(
        r"\s+(?:is|was|are|were|has|had|became)\s+|[;(]",
        quote,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    title_match = re.search(_surface_pattern(title_surface), lead, re.IGNORECASE)
    alias_match = re.search(_surface_pattern(surface), lead, re.IGNORECASE)
    if not title_match or not alias_match or title_match.start() > 16:
        return False
    left, right = sorted((title_match, alias_match), key=lambda value: value.start())
    bridge = lead[left.end() : right.start()]
    return bool(
        right.end() - left.start() <= 96
        and not re.search(r"[,;:!?]", bridge)
        and not re.search(
            r"\b(?:and|or|with|from|by|of|is|was|are|were)\b",
            bridge,
            re.IGNORECASE,
        )
    )


def _source_defined_description_identity(
    *, surface: str, validation: Mapping[str, Any], operator: Mapping[str, Any]
) -> bool:
    """Prove that a named output is explicitly defined by a description.

    This covers source statements such as ``Gregor Mendel ... is often called
    the father of genetics``.  It does not turn arbitrary co-occurrence into an
    identity edge: the named candidate must precede the description and the
    intervening clause must contain an explicit naming predicate.
    """

    answer = str(validation.get("answer") or "").strip()
    quote = str(validation.get("support_quote") or "")
    relation = str(validation.get("relation_phrase") or "").strip()
    if not answer or not surface or not quote:
        return False
    if str(operator.get("output_type") or "entity").casefold() not in {
        "entity", "person", "organization", "work", "place",
    }:
        return False
    answer_tokens = re.findall(r"[^\W_]+", answer, flags=re.UNICODE)
    if not answer_tokens or not any(
        token[:1].isupper() and len(token) > 1 for token in answer_tokens
    ):
        return False
    if canonical_entity_key(relation) != canonical_entity_key(surface):
        return False
    answer_match = re.search(_surface_pattern(answer), quote, re.IGNORECASE)
    surface_match = re.search(_surface_pattern(surface), quote, re.IGNORECASE)
    if not answer_match or not surface_match or answer_match.end() > surface_match.start():
        return False
    bridge = quote[answer_match.end() : surface_match.start()]
    return bool(
        len(bridge) <= 180
        and not re.search(r"[.!?]", bridge)
        and re.search(
            r"\b(?:called|known as|referred to as|dubbed|nicknamed)\b",
            bridge,
            re.IGNORECASE,
        )
    )


def _non_instance_intermediate_candidate(
    *, validation: Mapping[str, Any], operator: Mapping[str, Any]
) -> bool:
    """Detect a class label that cannot become an executable join binding."""

    intermediate = bool(
        str(operator.get("slot_plan_stage") or "") == "intermediate_binding"
        and str(operator.get("output_slot") or "") != "$answer"
    )
    if not intermediate:
        return False
    answer_tokens = re.findall(
        r"[^\W_]+", str(validation.get("answer") or ""), flags=re.UNICODE
    )
    return bool(
        answer_tokens
        and not any(token[:1].isupper() for token in answer_tokens)
    )


def _ambiguous_single_token_title_alias(
    *, surface: str, title: str, quote: str
) -> bool:
    """Identify an unproven surname/title fragment that may name many entities."""

    surface_tokens = _surface_tokens(surface)
    title_tokens = _surface_tokens(_title_identity_surface(title))
    return bool(
        len(surface_tokens) == 1
        and len(title_tokens) >= 2
        and surface_tokens[0] in title_tokens
        and source_surface_in_text(surface, quote)
    )


def binding_identity_decision(
    *,
    binding: Mapping[str, Any],
    operator: Mapping[str, Any],
    validation: Mapping[str, Any],
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a query-state entity binding before typed materialization.

    A lexical mention in a passage is not enough: the input slot must either
    carry a canonical identity established by the executor, or the bounded
    source partition must have an exact title identity.  This prevents query
    scaffold fragments such as ``Which film has the`` from becoming join keys.
    """

    input_slot = str(operator.get("input_slot") or "$root")
    slot = dict((binding.get("slot_bindings") or {}).get(input_slot) or {})
    surface = str(slot.get("surface") or "").strip()
    key = str(slot.get("key") or canonical_entity_key(surface))
    title = str(document.get("title") or "").strip()
    quote = str(validation.get("support_quote") or "")
    if bool(slot.get("canonical_identity_validated")):
        return {
            "admitted": True,
            "reason": "validated_binding_identity",
            "bound_surface": surface,
            "bound_key": key,
            "intentional_safety_rejection": False,
        }
    if _query_scaffold_binding(surface):
        return {
            "admitted": False,
            "reason": "query_scaffold_binding",
            "bound_surface": surface,
            "bound_key": key,
            "intentional_safety_rejection": True,
        }
    title_exact = bool(
        validation.get("bound_identity_mode") == "title_exact_entity"
        and key
        and canonical_entity_key(title) == key
    )
    if title_exact:
        return {
            "admitted": True,
            "reason": "candidate_source_title_exact_identity",
            "bound_surface": surface,
            "bound_key": key,
            "intentional_safety_rejection": False,
        }
    title_core_exact = bool(
        key
        and key == _title_core_entity_key(title)
        and source_surface_in_text(
            surface, str(validation.get("support_quote") or "")
        )
    )
    if title_core_exact:
        return {
            "admitted": True,
            "reason": "candidate_source_title_core_identity",
            "bound_surface": surface,
            "bound_key": key,
            "intentional_safety_rejection": False,
        }



    if _ambiguous_single_token_title_alias(
        surface=surface,
        title=title,
        quote=quote,
    ):
        return {
            "admitted": False,
            "reason": "ambiguous_single_token_title_alias",
            "bound_surface": surface,
            "bound_key": key,
            "intentional_safety_rejection": True,
        }
    if _source_title_expanded_alias_identity(
        surface=surface,
        title=title,
        quote=quote,
    ):
        return {
            "admitted": True,
            "reason": "source_title_expanded_alias_identity",
            "bound_surface": surface,
            "bound_key": key,
            "intentional_safety_rejection": False,
        }
    if _source_defined_description_identity(
        surface=surface,
        validation=validation,
        operator=operator,
    ):
        return {
            "admitted": True,
            "reason": "source_defined_description_identity",
            "bound_surface": surface,
            "bound_key": key,
            "intentional_safety_rejection": False,
        }
    if _non_instance_intermediate_candidate(
        validation=validation,
        operator=operator,
    ):
        return {
            "admitted": False,
            "reason": "non_instance_intermediate_candidate",
            "bound_surface": surface,
            "bound_key": key,
            "intentional_safety_rejection": True,
        }
    return {
        "admitted": False,
        "reason": "unvalidated_or_non_instance_binding",
        "bound_surface": surface,
        "bound_key": key,
        "intentional_safety_rejection": False,
    }


def _bound_aware_nationality_support(
    *,
    quote: str,
    bound_surface: str,
    answer: str,
    title_bound: bool,
) -> bool:
    """Require nationality evidence to be grammatically local to the binding.

    A nationality adjective elsewhere in the sentence is insufficient.  For
    example, ``German film directed by Louis Ralph`` describes the film, not
    Louis Ralph.  The common bound-before-value forms are accepted, while a
    title-bound pronoun form remains available for biography passages.
    """

    bound_match = (
        re.search(_surface_pattern(bound_surface), quote, re.IGNORECASE)
        if bound_surface else None
    )
    answer_match = (
        re.search(_surface_pattern(answer), quote, re.IGNORECASE)
        if answer else None
    )
    if not answer_match:
        return False
    if bound_match and bound_match.end() <= answer_match.start():
        bridge = quote[bound_match.end() : answer_match.start()]
        if (
            len(bridge) <= 120
            and not re.search(r"[.;!?]", bridge)
            and re.search(
                r"\b(?:is|was|are|were|nationality|citizen|citizenship|"
                r"origin|origins|from)\b",
                bridge,
                re.IGNORECASE,
            )
        ):
            return True
    if title_bound:
        answer_pattern = _surface_pattern(answer)
        return _matches_any(
            quote,
            (
                rf"^\s*(?:he|she|they|the subject)\s+"
                rf"(?:is|was|are|were).{{0,80}}{answer_pattern}",
                rf"^\s*(?:his|her|their)\s+(?:nationality|citizenship|"
                rf"origin)\s+(?:is|was).{{0,48}}{answer_pattern}",
            ),
        )
    return False


def _canonical_relation_for_operator(
    operator: Mapping[str, Any],
    matched_families: Sequence[str] = (),
) -> str:
    required_subtypes = {
        str(value) for value in operator.get("required_predicate_subtypes") or []
    }
    subtype_relations = {
        "birth_place": "born in",
        "birth_date": "born on",
        "death_place": "died in",
        "death_date": "died on",
        "death_cause": "died of",
        "occupation_role": "worked as",
        "employment_organization": "worked for",
        "employment_location": "worked in",
    }
    for subtype in sorted(required_subtypes):
        if subtype in subtype_relations:
            return subtype_relations[subtype]
    canonical = str(operator.get("canonical_relation") or "").strip()
    if canonical:
        return canonical.replace("_", " ")
    family_relations = {
        "nationality": "nationality",
        "direction": "director",
        "composition": "composer",
        "authorship": "author",
        "performance": "performer",
        "production": "producer",
        "visual_art": "painter",
        "creation": "creator",
        "design": "designer",
        "distribution": "released",
        "containment": "located in",
    }
    family_order = list(dict.fromkeys([
        *(str(value) for value in matched_families),
        *(str(value) for value in operator.get("predicate_families") or []),
    ]))
    for family in family_order:
        if str(family) in family_relations:
            return family_relations[str(family)]
    return " ".join(str(value) for value in operator.get("relation_tokens") or [])


def compile_typed_candidate_contract(
    *,
    validation: Mapping[str, Any],
    operator: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile exact lexical evidence into the operator's typed predicate.

    The compiler consumes only the candidate's exact source span and the
    already compiled operator.  It never observes a QA gold answer.  Generic
    canonical predicate matching is tried first; the source-local patterns
    cover relation realizations that the OpenIE predicate mapper currently
    misses while preserving output-type and subtype checks.
    """

    answer = str(validation.get("answer") or "").strip()
    relation = str(validation.get("relation_phrase") or "").strip()
    quote = str(validation.get("support_quote") or "").strip()
    expected_families = {
        str(value) for value in operator.get("predicate_families") or []
    }
    required_subtypes = {
        str(value) for value in operator.get("required_predicate_subtypes") or []
    }
    lexical = canonical_predicate_contract(relation, answer, quote)
    source_anchored_families = list(dict.fromkeys(
        str(value)
        for value in validation.get("source_anchored_predicate_families") or []
        if str(value).strip()
    ))
    lexical_families = set(lexical["canonical_families"])
    lexical_families.update(source_anchored_families)
    lexical_subtypes = set(lexical["semantic_subtypes"])
    family_match = bool(
        not expected_families or expected_families & lexical_families
    )
    subtype_match = bool(
        not operator.get("strict_predicate_subtype")
        or not required_subtypes
        or required_subtypes & lexical_subtypes
    )
    relation_match = re.search(
        _surface_pattern(relation), quote, re.IGNORECASE
    ) if relation else None
    answer_match = re.search(
        _surface_pattern(answer), quote, re.IGNORECASE
    ) if answer else None
    bound_surface = str(validation.get("bound_surface") or "").strip()
    bound_match = re.search(
        _surface_pattern(bound_surface), quote, re.IGNORECASE
    ) if bound_surface else None
    title_bound = str(validation.get("bound_identity_mode") or "") == (
        "title_exact_entity"
    )
    role_direction_required = _operator_requires_role_direction(dict(operator))
    source_anchored_arguments = bool(
        relation_match
        and answer_match
        and relation_match.start() <= answer_match.start()
        and (
            title_bound
            or (bound_match and bound_match.start() <= relation_match.start())
        )
        and validation.get("source_relation_canonicalization_reason")
    )
    ordered_source_arguments = bool(
        relation_match
        and answer_match
        and relation_match.start() <= answer_match.start()
        and (
            title_bound
            or (bound_match and bound_match.start() <= relation_match.start())
        )
        and not role_direction_required
    )
    if family_match and subtype_match:
        compilation_reason = str(
            validation.get("source_relation_canonicalization_reason") or ""
        ) if source_anchored_arguments else (
            "source_ordered_predicate_contract_match"
            if ordered_source_arguments
            else "lexical_predicate_contract_match"
        )
        return {
            "admitted": True,
            "reason": compilation_reason,
            "canonical_relation": _canonical_relation_for_operator(
                operator,
                [
                    *source_anchored_families,
                    *sorted(lexical_families - set(source_anchored_families)),
                ],
            ),
            "lexical_families": sorted(lexical_families),
            "source_anchored_families": source_anchored_families,
            "lexical_subtypes": sorted(lexical_subtypes),
            "source_argument_order_validated": bool(
                ordered_source_arguments or source_anchored_arguments
            ),
        }

    target_entity_categories = (
        operator.get("target_entity_categories") or []
        if "target_entity_categories" in operator
        else operator.get("relation_tokens") or []
    )
    terminal_type = terminal_value_contract_compatibility(
        answer,
        str(operator.get("output_type") or "entity"),
        re.findall(r"[a-z0-9]+", relation.casefold()),
        target_category_tokens=target_entity_categories,
        source_text=quote,
        relation_families=operator.get("relation_families") or [],
    )
    if terminal_type.get("status") == TERMINAL_TYPE_INCOMPATIBLE:
        return {
            "admitted": False,
            "reason": "output_type_mismatch",
            "canonical_relation": "",
            "lexical_families": sorted(lexical_families),
            "lexical_subtypes": sorted(lexical_subtypes),
        }

    answer_pattern = _surface_pattern(answer)
    typed_reason = ""
    if "nationality" in expected_families and _bound_aware_nationality_support(
        quote=quote,
        bound_surface=bound_surface,
        answer=answer,
        title_bound=title_bound,
    ):
        typed_reason = "bound_local_copula_or_origin_nationality"
    elif "birth_place" in required_subtypes and _matches_any(
        quote,
        (
            rf"\bborn\b.{{0,96}}\b(?:in|at)\b.{{0,40}}{answer_pattern}",
            rf"{answer_pattern}.{{0,64}}\bwhere\b.{{0,32}}\bborn\b",
        ),
    ):
        typed_reason = "source_birth_place_preposition"
    elif "death_place" in required_subtypes and _matches_any(
        quote,
        (
            rf"\bdied\b.{{0,48}}\b(?:in|at)\b.{{0,40}}{answer_pattern}",
            rf"{answer_pattern}.{{0,64}}\bwhere\b.{{0,32}}"
            rf"\b(?:he|she|they)?\s*died\b",
        ),
    ):
        typed_reason = "source_death_place_preposition"

    if not typed_reason:
        return {
            "admitted": False,
            "reason": (
                "strict_predicate_subtype_mismatch"
                if family_match and required_subtypes
                else "predicate_family_mismatch"
            ),
            "canonical_relation": "",
            "lexical_families": sorted(lexical_families),
            "lexical_subtypes": sorted(lexical_subtypes),
        }
    return {
        "admitted": True,
        "reason": typed_reason,
        "canonical_relation": _canonical_relation_for_operator(
            operator, sorted(lexical_families or expected_families)
        ),
        "lexical_families": sorted(lexical_families),
        "lexical_subtypes": sorted(lexical_subtypes),
        "source_argument_order_validated": True,
    }


def _answer_type_compatible(answer: str, expected_type: str) -> bool:
    expected = str(expected_type or "entity").strip().lower()
    if not str(answer or "").strip():
        return False
    if expected in {"date", "year"}:
        return bool(re.search(r"(?<!\d)(?:1[0-9]{3}|20[0-9]{2})(?!\d)", answer))
    if expected in {"number", "numeric", "quantity"}:
        return bool(re.search(r"\d", answer))
    if expected in {"boolean", "bool"}:
        return canonical_entity_key(answer) in {"yes", "no", "true", "false"}
    return True


def validate_materialization_candidate(
    candidate: Mapping[str, Any],
    *,
    bound_surface: str,
    operator: Mapping[str, Any],
    document: Mapping[str, Any],
    enable_directional_surface_canonicalization: bool = False,
) -> dict[str, Any]:
    """Apply provenance, identity, lexical relation, and type checks."""

    answer = " ".join(str(candidate.get("answer") or "").split()).strip()
    model_relation_phrase = " ".join(
        str(candidate.get("relation_phrase") or "").split()
    ).strip()
    relation_phrase = model_relation_phrase
    source_text = str(document.get("content") or "")
    source_title = str(document.get("title") or "").strip()
    located = _locate_exact_source_span(
        source_text, str(candidate.get("support_quote") or "")
    )
    quote = located[2] if located else ""
    answer_in_quote = _surface_in_text(answer, quote)
    source_relation = {"matched": False}
    if enable_directional_surface_canonicalization and answer_in_quote:
        lexical = canonical_predicate_contract(
            relation_phrase, answer, quote
        )
        expected_families = {
            str(value)
            for value in operator.get("predicate_families") or []
        }
        current_families = set(lexical.get("canonical_families") or [])
        current_relation_in_quote = bool(
            relation_phrase
            and relation_phrase.casefold() in quote.casefold()
        )
        if not current_relation_in_quote or not (
            not expected_families or expected_families & current_families
        ):
            source_relation = _source_anchored_relation_surface(
                quote=quote,
                answer=answer,
                operator=operator,
            )
            if source_relation.get("matched"):
                relation_phrase = str(
                    source_relation.get("relation_phrase") or relation_phrase
                ).strip()
    relation_in_quote = bool(
        relation_phrase
        and relation_phrase.casefold() in quote.casefold()
    )
    bound_in_quote = _surface_in_text(bound_surface, quote)
    bound_title_match = bool(
        canonical_entity_key(bound_surface)
        and canonical_entity_key(bound_surface) == canonical_entity_key(source_title)
    )
    type_compatible = _answer_type_compatible(
        answer, str(operator.get("output_type") or "entity")
    )
    model_supported = candidate.get("supported") is True
    accepted = bool(
        model_supported
        and located
        and answer_in_quote
        and relation_in_quote
        and (bound_in_quote or bound_title_match)
        and type_compatible
    )
    reasons = []
    for condition, reason in (
        (model_supported, "model_declared_unsupported"),
        (bool(located), "quote_not_in_source"),
        (answer_in_quote, "answer_not_in_quote"),
        (relation_in_quote, "relation_not_in_quote"),
        (bound_in_quote or bound_title_match, "bound_entity_not_source_grounded"),
        (type_compatible, "answer_type_mismatch"),
    ):
        if not condition:
            reasons.append(reason)
    return {
        "accepted": accepted,
        "answer": answer,
        "model_relation_phrase": model_relation_phrase,
        "relation_phrase": relation_phrase,
        "source_anchored_relation_phrase": (
            relation_phrase if source_relation.get("matched") else ""
        ),
        "source_anchored_predicate_families": list(
            source_relation.get("predicate_families") or []
        ),
        "source_relation_canonicalization_reason": str(
            source_relation.get("reason") or ""
        ),
        "source_relation_canonicalization_version": str(
            source_relation.get("version") or ""
        ),
        "support_quote": quote,
        "source_span_start": located[0] if located else -1,
        "source_span_end": located[1] if located else -1,
        "bound_identity_mode": (
            "explicit_bound_source" if bound_in_quote else
            "title_exact_entity" if bound_title_match else ""
        ),
        "bound_surface": str(bound_surface or "").strip(),
        "rejection_reasons": reasons,
    }


def materialize_unverified_operator_fact(
    *,
    validation: Mapping[str, Any],
    bound_surface: str,
    bound_key: str,
    operator: Mapping[str, Any],
    document: Mapping[str, Any],
    item_id: int,
) -> Optional[Dict[str, Any]]:
    """Create an operator-compatible query-local row, never a supported row."""

    if not validation.get("accepted"):
        return None
    answer = str(validation.get("answer") or "").strip()
    canonical_relation = str(
        operator.get("canonical_relation")
        or validation.get("relation_phrase")
        or "related to"
    ).replace("_", " ").strip()
    role_direction_required = _operator_requires_role_direction(dict(operator))





    relation = str(validation.get("relation_phrase") or "").strip()
    subject, obj = (
        (answer, bound_surface)
        if role_direction_required else (bound_surface, answer)
    )
    doc_id = int(document.get("doc_id", -1))
    quote = str(validation.get("support_quote") or "")
    item = _completion_fact_item(
        subject=subject,
        relation=relation,
        obj=obj,
        doc_id=doc_id,
        item_id=int(item_id),
        sentence=quote,
        source_sentence_id=-1,
        bound_input_key=bound_key,
        extraction_direction="llm_source_scoped",
        instance_identity_mode=str(validation.get("bound_identity_mode") or ""),
        instance_identity_validated=True,
    )
    if item is None:
        return None
    if role_direction_required:
        item["object_key"] = str(bound_key)
    else:
        item["subject_key"] = str(bound_key)
    item["entity_keys"] = {
        str(item.get("subject_key") or ""), str(item.get("object_key") or "")
    }
    item["tuple_key"] = "|".join([
        str(item.get("subject_key") or ""),
        str(item.get("relation_key") or ""),
        str(item.get("object_key") or ""),
    ])
    item.update({
        "access_path": "query_time_relation_materialization",
        "support_status": "UNVERIFIED",
        "predicate_subtypes": set(canonical_schema_predicate_subtypes(
            item.get("relation"), answer
        )),
        "completion_source_title": str(document.get("title") or ""),
        "completion_source_grounded": True,
        "completion_source_doc_id": doc_id,
        "completion_sentence": quote[:480],
        "source_sentence": quote[:480],
        "query_time_relation_materialization": True,
        "query_time_relation_materialization_version": "qtrm_v1",
    })
    source_subject = (
        bound_surface
        if _surface_in_text(bound_surface, quote)
        else str(document.get("title") or bound_surface)
    )
    relation_direction = (
        "object_to_subject" if role_direction_required else "subject_to_object"
    )




    item["source_grounded_recovery_record"] = {
        "version": "query_time_relation_materialization_v1",
        "canonical_subject": str(bound_surface),
        "canonical_subject_key": str(bound_key),
        "canonical_predicate": canonical_relation,
        "typed_object": {
            "surface": answer,
            "key": canonical_entity_key(answer),
            "type": str(operator.get("output_type") or "entity"),
        },
        "relation_direction": relation_direction,
        "source_document_id": doc_id,
        "source_title": str(document.get("title") or ""),
        "source_sentence_id": -1,
        "source_sentence": quote,
        "source_span_start": int(validation.get("source_span_start", -1)),
        "source_span_end": int(validation.get("source_span_end", -1)),
        "source_subject_surface": source_subject,
        "source_relation_surface": str(validation.get("relation_phrase") or ""),
        "source_object_surface": answer,
        "obligation_id": str(operator.get("operator_id") or ""),
        "query_local_provenance": "query_time_materialization_overlay",
        "candidate_support_status": "UNVERIFIED",
        "instance_identity_mode": str(validation.get("bound_identity_mode") or ""),
    }
    item["source_grounded_terminal_tuple_recovery"] = False
    return item


def materialize_binding_conditioned_typed_fact(
    *,
    validation: Mapping[str, Any],
    identity: Mapping[str, Any],
    typed_contract: Mapping[str, Any],
    operator: Mapping[str, Any],
    document: Mapping[str, Any],
    item_id: int,
) -> Optional[Dict[str, Any]]:
    """Create a canonical row plus a source-replayable typed certificate."""

    typed_validation = dict(validation)
    lexical_relation = str(validation.get("relation_phrase") or "").strip()
    typed_validation["relation_phrase"] = str(
        typed_contract.get("canonical_relation") or ""
    ).strip()
    item = materialize_unverified_operator_fact(
        validation=typed_validation,
        bound_surface=str(identity.get("bound_surface") or ""),
        bound_key=str(identity.get("bound_key") or ""),
        operator=operator,
        document=document,
        item_id=item_id,
    )
    if item is None:
        return None
    answer = str(validation.get("answer") or "").strip()
    quote = str(validation.get("support_quote") or "")
    source_title = str(document.get("title") or "").strip()
    bound_surface = str(identity.get("bound_surface") or "").strip()
    bound_key = str(identity.get("bound_key") or "")
    role_direction_required = _operator_requires_role_direction(dict(operator))
    triple = list(item.get("triple") or [])
    if len(triple) != 3:
        return None
    physical_subject = str(triple[0] or "").strip()
    physical_object = str(triple[2] or "").strip()
    lexical_bound_surface = (
        bound_surface if _surface_in_text(bound_surface, quote) else source_title
    )
    item.update({
        "lexical_relation_phrase": lexical_relation,
        "model_relation_phrase": str(
            validation.get("model_relation_phrase") or lexical_relation
        ),
        "source_relation_canonicalization_reason": str(
            validation.get("source_relation_canonicalization_reason") or ""
        ),
        "source_relation_canonicalization_version": str(
            validation.get("source_relation_canonicalization_version") or ""
        ),
        "typed_contract_reason": str(typed_contract.get("reason") or ""),
        "binding_identity_reason": str(identity.get("reason") or ""),
        "binding_identity_contract_version": BINDING_IDENTITY_CONTRACT_VERSION,
        "query_time_relation_materialization_version": (
            BINDING_CONDITIONED_TYPED_MATERIALIZATION_VERSION
        ),
        "binding_conditioned_typed_materialization": True,
    })
    item["source_grounded_recovery_record"] = {
        "version": BINDING_CONDITIONED_TYPED_MATERIALIZATION_VERSION,
        "canonical_subject": bound_surface,
        "canonical_subject_key": bound_key,
        "canonical_predicate": str(typed_contract.get("canonical_relation") or ""),
        "typed_object": {
            "surface": answer,
            "key": canonical_entity_key(answer),
            "type": str(operator.get("output_type") or "entity"),
        },
        "relation_direction": (
            "object_to_subject" if role_direction_required else "subject_to_object"
        ),
        "source_document_id": int(document.get("doc_id", -1)),
        "source_title": source_title,
        "source_sentence_id": -1,
        "source_sentence": quote,
        "source_span_start": int(validation.get("source_span_start", -1)),
        "source_span_end": int(validation.get("source_span_end", -1)),



        "source_subject_surface": physical_subject,
        "source_object_surface": physical_object,
        "physical_subject_surface": physical_subject,
        "physical_object_surface": physical_object,
        "source_lexical_bound_surface": lexical_bound_surface,
        "source_lexical_answer_surface": answer,
        "source_relation_surface": lexical_relation,
        "model_relation_surface": str(
            validation.get("model_relation_phrase") or lexical_relation
        ),
        "source_anchored_predicate_families": list(
            validation.get("source_anchored_predicate_families") or []
        ),
        "source_relation_canonicalization_reason": str(
            validation.get("source_relation_canonicalization_reason") or ""
        ),
        "source_relation_canonicalization_version": str(
            validation.get("source_relation_canonicalization_version") or ""
        ),
        "obligation_id": str(operator.get("operator_id") or ""),
        "query_local_provenance": "typed_materialization_overlay",
        "candidate_support_status": "UNVERIFIED",
        "instance_identity_mode": str(validation.get("bound_identity_mode") or ""),
        "binding_identity_reason": str(identity.get("reason") or ""),
        "binding_identity_contract_version": BINDING_IDENTITY_CONTRACT_VERSION,
        "predicate_compilation_reason": str(typed_contract.get("reason") or ""),
        "source_argument_order_validated": bool(
            typed_contract.get("source_argument_order_validated")
        ),
        "lexical_predicate_families": list(
            typed_contract.get("lexical_families") or []
        ),
        "lexical_predicate_subtypes": list(
            typed_contract.get("lexical_subtypes") or []
        ),
        "certified_tuple_key": str(item.get("tuple_key") or ""),
    }
    return item


def decode_and_materialize_response(
    response: Any,
    *,
    bound_surface: str,
    bound_key: str,
    operator: Mapping[str, Any],
    documents: Sequence[Mapping[str, Any]],
    source_windows: Optional[Sequence[Mapping[str, Any]]] = None,
    binding: Optional[Mapping[str, Any]] = None,
    enable_binding_conditioned_typed_materialization: bool = False,
    enable_directional_surface_canonicalization: bool = False,
) -> dict[str, Any]:
    """Turn one model response into deterministically validated fact rows."""

    docs_by_id = {
        int(document.get("doc_id", -1)): dict(document)
        for document in documents
        if int(document.get("doc_id", -1)) >= 0
    }
    candidates = parse_materialization_candidates(response)
    windows_by_id = {
        str(window.get("window_id") or ""): dict(window)
        for window in source_windows or []
        if str(window.get("window_id") or "")
    }
    facts = []
    validations = []
    seen = set()
    binding_identity_admitted_count = 0
    typed_contract_admitted_count = 0
    for ordinal, candidate in enumerate(candidates):
        try:
            source_id = int(candidate.get("source_id", -1))
        except (TypeError, ValueError):
            source_id = -1
        document = docs_by_id.get(source_id)
        if document is None:
            validations.append({
                "accepted": False,
                "source_id": source_id,
                "rejection_reasons": ["source_id_not_in_candidate_pool"],
            })
            continue
        validation = validate_materialization_candidate(
            candidate,
            bound_surface=bound_surface,
            operator=operator,
            document=document,
            enable_directional_surface_canonicalization=(
                enable_directional_surface_canonicalization
            ),
        )
        if source_windows is not None:
            window_id = str(candidate.get("window_id") or "")
            window = windows_by_id.get(window_id)
            window_reason = ""
            if window is None:
                window_reason = "window_id_not_in_ranked_pool"
            elif int(window.get("doc_id", -1)) != source_id:
                window_reason = "window_source_mismatch"
            elif _locate_exact_source_span(
                str(window.get("content") or ""),
                str(candidate.get("support_quote") or ""),
            ) is None:
                window_reason = "quote_not_in_ranked_window"
            validation["window_id"] = window_id
            validation["ranked_window_validated"] = not bool(window_reason)
            if window_reason:
                validation["accepted"] = False
                validation["rejection_reasons"] = [
                    *list(validation.get("rejection_reasons") or []),
                    window_reason,
                ]
        validation["source_id"] = source_id
        fact = None
        if enable_binding_conditioned_typed_materialization:
            identity = binding_identity_decision(
                binding=dict(binding or {}),
                operator=operator,
                validation=validation,
                document=document,
            )
            typed_contract = (
                compile_typed_candidate_contract(
                    validation=validation,
                    operator=operator,
                )
                if validation.get("accepted") and identity.get("admitted")
                else {
                    "admitted": False,
                    "reason": (
                        "source_validation_rejected"
                        if not validation.get("accepted")
                        else "identity_gate_rejected"
                    ),
                    "canonical_relation": "",
                    "lexical_families": [],
                    "lexical_subtypes": [],
                }
            )
            validation.update({
                "binding_identity_contract_version": (
                    BINDING_IDENTITY_CONTRACT_VERSION
                ),
                "binding_identity_admitted": bool(identity.get("admitted")),
                "binding_identity_reason": str(identity.get("reason") or ""),
                "binding_identity_intentional_safety_rejection": bool(
                    identity.get("intentional_safety_rejection")
                ),
                "typed_contract_admitted": bool(typed_contract.get("admitted")),
                "typed_contract_reason": str(typed_contract.get("reason") or ""),
                "typed_canonical_relation": str(
                    typed_contract.get("canonical_relation") or ""
                ),
                "lexical_predicate_families": list(
                    typed_contract.get("lexical_families") or []
                ),
                "lexical_predicate_subtypes": list(
                    typed_contract.get("lexical_subtypes") or []
                ),
                "source_argument_order_validated": bool(
                    typed_contract.get("source_argument_order_validated")
                ),
            })
            binding_identity_admitted_count += int(bool(identity.get("admitted")))
            typed_contract_admitted_count += int(
                bool(typed_contract.get("admitted"))
            )
            if (
                validation.get("accepted")
                and identity.get("admitted")
                and typed_contract.get("admitted")
            ):
                fact = materialize_binding_conditioned_typed_fact(
                    validation=validation,
                    identity=identity,
                    typed_contract=typed_contract,
                    operator=operator,
                    document=document,
                    item_id=-4000000 - ordinal,
                )
        else:
            fact = materialize_unverified_operator_fact(
                validation=validation,
                bound_surface=bound_surface,
                bound_key=bound_key,
                operator=operator,
                document=document,
                item_id=-4000000 - ordinal,
            )
        validations.append(validation)
        if fact is None or fact["tuple_key"] in seen:
            continue
        seen.add(fact["tuple_key"])
        facts.append(fact)
    return {
        "facts": facts,
        "validations": validations,
        "candidate_count": len(candidates),
        "accepted_count": len(facts),
        "binding_identity_admitted_count": binding_identity_admitted_count,
        "typed_contract_admitted_count": typed_contract_admitted_count,
        "binding_conditioned_typed_materialization": bool(
            enable_binding_conditioned_typed_materialization
        ),
        "directional_surface_canonicalization": bool(
            enable_directional_surface_canonicalization
        ),
        "binding_identity_contract_version": (
            BINDING_IDENTITY_CONTRACT_VERSION
            if enable_binding_conditioned_typed_materialization else ""
        ),
    }
