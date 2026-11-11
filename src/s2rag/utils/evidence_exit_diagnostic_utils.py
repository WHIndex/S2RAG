"""Read-only helpers for the visited-source evidence-exit diagnostic.

The helpers consume frozen executor traces and Reader records.  They never
run retrieval, extraction, verification, admission, or witness reduction.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping, Sequence

from s2rag.utils.recovery_to_answer_closure_utils import (
    RENDERER_VERSION,
    REQUEST_VERSION,
    SourceDocument,
    build_exact_request,
    conservative_message_token_upper_bound,
    document_from_content,
    sha256_text,
    sha256_value,
    truncate_documents_to_token_budget,
)


DIAGNOSTIC_VERSION = "lazy_visited_source_evidence_v1"
DIAGNOSTIC_ARM = "LAZY_VISITED_SOURCE_EVIDENCE"
SELECTIVE_DIAGNOSTIC_VERSION = "selective_dual_evidence_exit_v2"
SELECTIVE_DIAGNOSTIC_ARM = "SELECTIVE_DUAL_EVIDENCE_EXIT"
OBLIGATION_SCOPED_DIAGNOSTIC_VERSION = "obligation_scoped_evidence_exit_v3_1"
OBLIGATION_SCOPED_DIAGNOSTIC_ARM = "OBLIGATION_SCOPED_EVIDENCE_EXIT_V3_1"


_WORD_RE = re.compile(r"[a-z0-9]+")
_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)\s*$")
_DATE_VALUE_RE = re.compile(
    r"(?:\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b|\b(?:1[0-9]{3}|20[0-9]{2})\b)",
    re.IGNORECASE,
)
_NUMBER_VALUE_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_UNCERTAIN_ANSWER_RE = re.compile(
    r"(?:\bcannot\s+(?:be\s+)?determin(?:e|ed)\b|"
    r"\bunable\s+to\s+determine\b|\bunknown\b|\bnot\s+(?:provided|specified|available)\b|"
    r"\bno\s+(?:relevant\s+)?(?:answer|date|location|place|information|evidence)\b|"
    r"\binsufficient\s+(?:information|evidence)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceVisit:
    source_id: int
    event_order: int
    proof: str
    operator: str
    obligation_id: str
    fact_usage_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "event_order": self.event_order,
            "proof": self.proof,
            "operator": self.operator,
            "obligation_id": self.obligation_id,
            "fact_usage_id": self.fact_usage_id,
        }


def _fact_identifier(fact: Mapping[str, Any]) -> str:
    return str(fact.get("fact_usage_id") or fact.get("fact_id") or "")


def _fact_source_id(fact: Mapping[str, Any]) -> int | None:
    value = fact.get("source_id")
    if value is None:
        value = fact.get("completion_source_doc_id", fact.get("doc_idx"))
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def extract_actual_source_visits(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return source-body reads supported by the frozen trace.

    The harness completion callback materializes source content before adding
    an id to ``returned_source_ids``.  Recovered facts and VERIFY records are
    additional proof when a source did not appear in that callback log.
    Merely occurring in the fact index is deliberately insufficient.
    """
    facts = {
        _fact_identifier(fact): dict(fact)
        for fact in trace.get("facts") or []
        if _fact_identifier(fact)
    }
    ordered: list[SourceVisit] = []
    seen: set[int] = set()

    def append(
        source_id: int | None,
        *,
        proof: str,
        operator: str,
        obligation_id: str = "",
        fact_usage_id: str = "",
    ) -> None:
        if source_id is None or source_id in seen:
            return
        seen.add(source_id)
        ordered.append(
            SourceVisit(
                source_id=source_id,
                event_order=len(ordered),
                proof=proof,
                operator=operator,
                obligation_id=obligation_id,
                fact_usage_id=fact_usage_id,
            )
        )

    complete_events = [
        event
        for event in trace.get("operator_trace") or []
        if str(event.get("operator") or "") == "COMPLETE"
    ]
    for lookup_index, lookup in enumerate(
        (trace.get("harness") or {}).get("completion_retrieval_log") or []
    ):
        complete_event = (
            complete_events[lookup_index]
            if lookup_index < len(complete_events)
            else {}
        )
        for source_id in lookup.get("returned_source_ids") or []:
            append(
                int(source_id),
                proof=f"completion_lookup_body_read:{lookup_index}",
                operator="COMPLETE",
                obligation_id=str(complete_event.get("obligation_id") or ""),
            )








    for event in complete_events:
        physical_access = event.get("physical_access") or {}
        obligation_id = str(
            event.get("target_obligation")
            or event.get("obligation_id")
            or ""
        )
        for source_id in physical_access.get("bounded_source_doc_ids") or []:
            append(
                source_id,
                proof="completion_physical_access_body_read",
                operator="COMPLETE",
                obligation_id=obligation_id,
            )

    for fact in trace.get("facts") or []:
        if str(fact.get("origin") or "") != "query_local_overlay" and not str(
            fact.get("fact_id") or ""
        ).startswith("completion:"):
            continue
        append(
            _fact_source_id(fact),
            proof="recovered_tuple_requires_source_body",
            operator="COMPLETE",
            obligation_id=str(fact.get("operator_id") or ""),
            fact_usage_id=_fact_identifier(fact),
        )

    for event in trace.get("operator_trace") or []:
        if str(event.get("operator") or "") != "VERIFY":
            continue
        usage_id = str(event.get("fact_usage_id") or event.get("fact_id") or "")
        fact = facts.get(usage_id)
        append(
            _fact_source_id(fact or {}),
            proof="verify_source_body_read",
            operator="VERIFY",
            obligation_id=str(event.get("obligation_id") or ""),
            fact_usage_id=usage_id,
        )
    return [visit.as_dict() for visit in ordered]


def _normalized_words(value: str) -> tuple[str, ...]:
    return tuple(_WORD_RE.findall(str(value or "").lower()))


def _normalized_phrase(value: str) -> str:
    return " ".join(_normalized_words(value))


def _source_title(content: str) -> str:
    lines = str(content or "").splitlines()
    return lines[0].strip() if lines else ""


def _title_aliases(title: str) -> list[str]:
    values = [str(title or "").strip()]
    without_parenthetical = _PARENTHETICAL_RE.sub("", values[0]).strip()
    if without_parenthetical and without_parenthetical != values[0]:
        values.append(without_parenthetical)
    output: list[str] = []
    for value in values:
        normalized = _normalized_phrase(value)
        if normalized and normalized not in output:
            output.append(normalized)
    return output


def _contains_entity_phrase(haystack: str, phrase: str) -> bool:
    phrase_tokens = tuple(phrase.split())
    if len(phrase_tokens) < 2:
        return False
    padded_haystack = f" {_normalized_phrase(haystack)} "
    return f" {phrase} " in padded_haystack


def _binding_surfaces(
    trace: Mapping[str, Any], *, require_fact_lineage: bool = False
) -> set[str]:
    values: set[str] = set()
    for binding in trace.get("bindings") or []:
        if require_fact_lineage and not (
            binding.get("path_fact_usage_ids") or binding.get("path_fact_ids")
        ):
            continue
        for item in (binding.get("slot_bindings") or {}).values():
            if not isinstance(item, Mapping):
                continue
            for field in ("surface", "canonical_identity_surface"):
                normalized = _normalized_phrase(str(item.get(field) or ""))
                if normalized:
                    values.add(normalized)
    return values


def _operator_relation_words(
    trace: Mapping[str, Any], obligation_id: str
) -> set[str]:
    plan = trace.get("logical_plan") or {}
    operators = plan.get("operators") or plan.get("logical_operators") or []
    selected: list[Mapping[str, Any]] = []
    for operator in operators:
        if str(operator.get("operator_id") or "") == str(obligation_id or ""):
            selected.append(operator)
    terminal = plan.get("terminal_operator") or {}
    if terminal:
        selected.append(terminal)
    values: set[str] = set()
    for operator in selected:
        for field in (
            "relation_tokens",
            "predicate_families",
            "relation_families",
            "target_entity_categories",
        ):
            for value in operator.get(field) or []:
                values.update(_normalized_words(str(value)))
        values.update(_normalized_words(str(operator.get("canonical_relation") or "")))
        values.update(_normalized_words(str(operator.get("output_role") or "")))
    return values


def _content_has_relation_signal(content: str, relation_words: set[str]) -> bool:
    content_words = set(_normalized_words(content))
    for relation_word in relation_words:
        if len(relation_word) < 4:
            continue
        stem = relation_word[:-1] if relation_word.endswith("s") else relation_word
        if any(word == relation_word or word.startswith(stem) for word in content_words):
            return True
    return False


def _content_has_answer_value_signal(
    content: str, answer_contract: Mapping[str, Any]
) -> bool:
    answer_type = str(answer_contract.get("expected_answer_type") or "")
    if answer_type == "date":
        return bool(_DATE_VALUE_RE.search(content))
    if answer_type in {"number", "count", "quantity"}:
        return bool(_NUMBER_VALUE_RE.search(content))
    return False


def _facts_by_source(trace: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for fact in trace.get("facts") or []:
        source_id = _fact_source_id(fact)
        if source_id is None:
            continue
        result.setdefault(source_id, []).append(dict(fact))
    return result


def is_reader_prediction_uncertain(prediction: str) -> bool:
    """Classify explicit Reader abstentions without using QA gold."""
    value = str(prediction or "").strip()
    return not value or bool(_UNCERTAIN_ANSWER_RE.search(value))


def _fact_operator_id(fact: Mapping[str, Any]) -> str:
    contract = fact.get("obligation_contract") or {}
    return str(
        fact.get("admission_obligation_id")
        or fact.get("operator_id")
        or contract.get("operator_id")
        or ""
    )


def _is_supported_admitted(fact: Mapping[str, Any]) -> bool:
    return (
        str(fact.get("support_status") or "") == "SUPPORTED"
        and str(fact.get("obligation_admission_status") or "") == "ADMITTED"
    )


def _fact_matches_visit_obligation(
    fact: Mapping[str, Any], visit_obligation_id: str
) -> bool:
    visit_obligation_id = str(visit_obligation_id or "")
    return not visit_obligation_id or _fact_operator_id(fact) == visit_obligation_id


def _entity_token_key(value: str) -> str:
    return " ".join(sorted(_normalized_words(value)))


def _fact_bound_input_key(fact: Mapping[str, Any]) -> str:
    contract = fact.get("obligation_contract") or {}
    return _entity_token_key(
        str(contract.get("bound_input_key") or contract.get("bound_input_surface") or "")
    )


def _fact_endpoint_keys(fact: Mapping[str, Any]) -> set[str]:
    values = {
        _entity_token_key(str(fact.get("subject") or "")),
        _entity_token_key(str(fact.get("object") or "")),
        _fact_bound_input_key(fact),
    }
    return {value for value in values if value}


def _source_title_matches_fact(title: str, fact: Mapping[str, Any]) -> bool:
    title_keys = {_entity_token_key(alias) for alias in _title_aliases(title)}
    return bool({value for value in title_keys if value} & _fact_endpoint_keys(fact))


def _comparison_terminal_operator_id(trace: Mapping[str, Any]) -> str:
    plan = trace.get("logical_plan") or {}
    operators = plan.get("operators") or plan.get("logical_operators") or []
    if not operators:
        return ""
    return str(operators[-1].get("operator_id") or "")


def _is_comparison_plan(trace: Mapping[str, Any]) -> bool:
    plan = trace.get("logical_plan") or {}
    if str(plan.get("plan_kind") or "").startswith("branch_"):
        return True
    contract = plan.get("question_obligation_contract") or {}
    return int(contract.get("comparison_obligation_count") or 0) > 0


def _comparison_branch_keys(
    trace: Mapping[str, Any], visible_source_ids: set[int]
) -> set[str]:
    terminal_operator_id = _comparison_terminal_operator_id(trace)
    if not terminal_operator_id:
        return set()
    keys: set[str] = set()
    for fact in trace.get("facts") or []:
        source_id = _fact_source_id(fact)
        if source_id not in visible_source_ids:
            continue
        if not _is_supported_admitted(fact):
            continue
        if _fact_operator_id(fact) != terminal_operator_id:
            continue
        key = _fact_bound_input_key(fact)
        if key:
            keys.add(key)
    return keys


def infer_answer_contract(
    trace: Mapping[str, Any], question: str
) -> dict[str, Any]:
    plan = trace.get("logical_plan") or {}
    answer_type = str(plan.get("final_answer_type") or "").strip().lower()
    if not answer_type:
        answer_type = str(
            (plan.get("question_obligation_contract") or {}).get("answer_type") or ""
        ).strip().lower()
    question_words = set(_normalized_words(question))
    date_question = answer_type == "date" or (
        not answer_type
        and bool(question_words & {"when", "born", "died", "date"})
    )
    return {
        "expected_answer_type": answer_type or "unknown",
        "prefer_most_specific_supported_date": date_question,
        "guard_mode": "metadata_only_no_answer_replacement",
    }


def select_dual_evidence_sources(
    *,
    original_lazy_record: Mapping[str, Any],
    trace: Mapping[str, Any],
    visits: Sequence[Mapping[str, Any]],
    source_contents: Mapping[int, str],
    max_selected_sources: int,
    excluded_source_ids: Sequence[int] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select ordinary-text sidecar sources without certifying new tuples.

    A source is eligible only when the frozen run actually read its body and
    either (a) it already produced a SUPPORTED+ADMITTED partial fact, or (b)
    its title is connected to an existing binding, the question, or the
    frozen base context.  The latter is a textual join signal, not a support
    decision, and the source remains ordinary Reader text.
    """
    existing_ids = {
        int(value) for value in original_lazy_record.get("actual_source_ids") or []
    }
    excluded_ids = {int(value) for value in excluded_source_ids}
    base_contents = [
        str(row.get("content") or "")
        for row in original_lazy_record.get("actual_documents") or []
    ]
    base_text = "\n".join(base_contents)
    question = str(original_lazy_record.get("question") or "")
    lineage_binding_surfaces = _binding_surfaces(
        trace, require_fact_lineage=True
    )
    facts_by_source = _facts_by_source(trace)
    answer_contract = infer_answer_contract(trace, question)
    has_complete_candidate_witness = bool(
        trace.get("question_complete_supported_witness")
        or trace.get("supported_witnesses")
    )
    decisions: list[dict[str, Any]] = []
    eligible: list[tuple[tuple[int, int, int], dict[str, Any]]] = []

    for visit_position, raw_visit in enumerate(visits):
        source_id = int(raw_visit["source_id"])
        content = str(source_contents.get(source_id) or "").strip()
        title = _source_title(content)
        aliases = _title_aliases(title)
        source_facts = facts_by_source.get(source_id, [])
        supported_admitted = [
            fact
            for fact in source_facts
            if str(fact.get("support_status") or "") == "SUPPORTED"
            and str(fact.get("obligation_admission_status") or "") == "ADMITTED"
        ]
        binding_link = any(alias in lineage_binding_surfaces for alias in aliases)
        question_link = any(
            _contains_entity_phrase(question, alias) for alias in aliases
        )
        base_context_link = any(
            _contains_entity_phrase(base_text, alias) for alias in aliases
        )
        relation_words = _operator_relation_words(
            trace, str(raw_visit.get("obligation_id") or "")
        )
        relation_signal = _content_has_relation_signal(content, relation_words)
        answer_value_signal = _content_has_answer_value_signal(
            content, answer_contract
        )
        decision = {
            "source_id": source_id,
            "event_order": int(raw_visit.get("event_order", visit_position)),
            "visit_proof": str(raw_visit.get("proof") or ""),
            "visit_operator": str(raw_visit.get("operator") or ""),
            "visit_obligation_id": str(raw_visit.get("obligation_id") or ""),
            "source_title": title,
            "supported_admitted_fact_count": len(supported_admitted),
            "binding_title_link": binding_link,
            "question_title_link": question_link,
            "base_context_title_link": base_context_link,
            "relation_content_signal": relation_signal,
            "answer_value_signal": answer_value_signal,
            "complete_candidate_witness_exists": has_complete_candidate_witness,
            "ordinary_text_only": True,
            "tuple_certified_by_selection": False,
        }
        if source_id in existing_ids:
            decision["decision"] = "DEDUPLICATED_ALREADY_IN_CONTEXT"
        elif source_id in excluded_ids:
            decision["decision"] = "REJECTED_CONTENT_IDENTITY_CONFLICT"
        elif not content:
            decision["decision"] = "NOT_LOGGED_SOURCE_CONTENT_UNAVAILABLE"
        elif supported_admitted and not has_complete_candidate_witness:
            decision["decision"] = "ELIGIBLE_SUPPORTED_ADMITTED_PARTIAL_SOURCE"
            decision["priority_tier"] = 0
            eligible.append(((0, decision["event_order"], source_id), decision))
        elif (
            not has_complete_candidate_witness
            and (binding_link or question_link or base_context_link)
            and (relation_signal or answer_value_signal)
        ):
            decision["decision"] = "ELIGIBLE_TEXTUAL_ENTITY_JOIN_SOURCE"
            decision["priority_tier"] = 1
            eligible.append(((1, decision["event_order"], source_id), decision))
        else:
            decision["decision"] = "REJECTED_NO_BINDING_OR_OBLIGATION_SIGNAL"
        decisions.append(decision)

    chosen_ids = {
        item[1]["source_id"]
        for item in sorted(eligible, key=lambda item: item[0])[
            : max(0, int(max_selected_sources))
        ]
    }
    selected_visits: list[dict[str, Any]] = []
    for decision in decisions:
        if decision["source_id"] in chosen_ids:
            decision["selection_decision"] = "SELECTED_ORDINARY_TEXT_SIDECAR"
            selected_visits.append(
                next(
                    dict(visit)
                    for visit in visits
                    if int(visit["source_id"]) == decision["source_id"]
                )
            )
        elif str(decision.get("decision") or "").startswith("ELIGIBLE_"):
            decision["selection_decision"] = "ELIGIBLE_BUT_SOURCE_CAP_REACHED"
        else:
            decision["selection_decision"] = "NOT_SELECTED"
    selected_visits.sort(
        key=lambda visit: next(
            key
            for key, decision in eligible
            if decision["source_id"] == int(visit["source_id"])
        )
    )
    return selected_visits, decisions


def build_selective_dual_evidence_reader_record(
    *,
    original_lazy_record: Mapping[str, Any],
    trace: Mapping[str, Any],
    visits: Sequence[Mapping[str, Any]],
    source_contents: Mapping[int, str],
    render_messages: Callable[[str, Sequence[str]], Sequence[Mapping[str, str]]],
    count_message_tokens: Callable[[Sequence[Mapping[str, str]]], int],
    token_counter_version: str,
    max_selected_sources: int,
    max_input_tokens: int,
    excluded_source_ids: Sequence[int] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    before = state_fingerprint(trace)
    selected_visits, selection_decisions = select_dual_evidence_sources(
        original_lazy_record=original_lazy_record,
        trace=trace,
        visits=visits,
        source_contents=source_contents,
        max_selected_sources=max_selected_sources,
        excluded_source_ids=excluded_source_ids,
    )
    record, append_decisions = build_visited_source_reader_record(
        original_lazy_record=original_lazy_record,
        visits=selected_visits,
        source_contents=source_contents,
        render_messages=render_messages,
        count_message_tokens=count_message_tokens,
        token_counter_version=token_counter_version,
        max_extra_sources=max_selected_sources,
        max_input_tokens=max_input_tokens,
        excluded_source_ids=excluded_source_ids,
    )
    append_by_source = {
        int(row["source_id"]): dict(row) for row in append_decisions
    }
    for decision in selection_decisions:
        append_decision = append_by_source.get(int(decision["source_id"]))
        if append_decision:
            decision["render_decision"] = append_decision.get("decision")
            if append_decision.get("content_sha256"):
                decision["content_sha256"] = append_decision["content_sha256"]
    record.update(
        {
            "version": SELECTIVE_DIAGNOSTIC_VERSION,
            "reader_arm": SELECTIVE_DIAGNOSTIC_ARM,
            "diagnostic_renderer_version": SELECTIVE_DIAGNOSTIC_VERSION,
            "strict_visited_source_ids": [
                int(row["source_id"]) for row in visits
            ],
            "selected_sidecar_source_ids": [
                int(row["source_id"]) for row in selected_visits
            ],
            "source_selection_decisions": selection_decisions,
            "answer_contract": infer_answer_contract(
                trace, str(original_lazy_record.get("question") or "")
            ),
            "structured_answer_admission": False,
            "ordinary_text_sidecar_only": True,
            "gold_used_online": False,
        }
    )
    if state_fingerprint(trace) != before:
        raise AssertionError("selective_dual_exit_mutated_execution_state")
    return record, selection_decisions


def select_obligation_scoped_evidence_sources(
    *,
    original_lazy_record: Mapping[str, Any],
    trace: Mapping[str, Any],
    visits: Sequence[Mapping[str, Any]],
    source_contents: Mapping[int, str],
    baseline_prediction: str,
    max_selected_sources: int,
    excluded_source_ids: Sequence[int] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select Reader sidecars under an obligation- and branch-scoped contract.

    V2 treated any supported and admitted fact in a visited source as a source-
    level certificate.  V3 requires the fact to match the visit obligation and
    the source to align with the active entity.  Uncertified text is useful only
    as a selective recovery path after an explicit Reader abstention.  For a
    comparison plan, a stable incumbent is never replaced by evidence covering
    fewer than two terminal bound inputs.

    This policy reads only frozen execution state and the baseline Reader
    prediction.  It does not use QA gold and does not mutate execution state.
    """
    existing_ids = {
        int(value) for value in original_lazy_record.get("actual_source_ids") or []
    }
    excluded_ids = {int(value) for value in excluded_source_ids}
    base_contents = [
        str(row.get("content") or "")
        for row in original_lazy_record.get("actual_documents") or []
    ]
    base_text = "\n".join(base_contents)
    question = str(original_lazy_record.get("question") or "")
    lineage_binding_surfaces = _binding_surfaces(trace, require_fact_lineage=True)
    facts_by_source = _facts_by_source(trace)
    answer_contract = infer_answer_contract(trace, question)
    has_complete_candidate_witness = bool(
        trace.get("question_complete_supported_witness")
        or trace.get("supported_witnesses")
    )
    baseline_uncertain = is_reader_prediction_uncertain(baseline_prediction)
    comparison_plan = _is_comparison_plan(trace)
    decisions: list[dict[str, Any]] = []
    eligible: list[tuple[tuple[int, int, int], dict[str, Any]]] = []

    for visit_position, raw_visit in enumerate(visits):
        source_id = int(raw_visit["source_id"])
        visit_obligation_id = str(raw_visit.get("obligation_id") or "")
        content = str(source_contents.get(source_id) or "").strip()
        title = _source_title(content)
        aliases = _title_aliases(title)
        source_facts = facts_by_source.get(source_id, [])
        supported_admitted = [fact for fact in source_facts if _is_supported_admitted(fact)]
        obligation_aligned = [
            fact
            for fact in supported_admitted
            if _fact_matches_visit_obligation(fact, visit_obligation_id)
        ]
        binding_link = any(alias in lineage_binding_surfaces for alias in aliases)
        question_link = any(
            _contains_entity_phrase(question, alias) for alias in aliases
        )
        base_context_link = any(
            _contains_entity_phrase(base_text, alias) for alias in aliases
        )
        fact_endpoint_link = any(
            _source_title_matches_fact(title, fact) for fact in obligation_aligned
        )
        source_identity_aligned = bool(
            binding_link or question_link or base_context_link or fact_endpoint_link
        )
        relation_words = _operator_relation_words(trace, visit_obligation_id)
        relation_signal = _content_has_relation_signal(content, relation_words)
        answer_value_signal = _content_has_answer_value_signal(content, answer_contract)
        decision = {
            "source_id": source_id,
            "event_order": int(raw_visit.get("event_order", visit_position)),
            "visit_proof": str(raw_visit.get("proof") or ""),
            "visit_operator": str(raw_visit.get("operator") or ""),
            "visit_obligation_id": visit_obligation_id,
            "source_title": title,
            "supported_admitted_fact_count": len(supported_admitted),
            "obligation_aligned_supported_admitted_fact_count": len(obligation_aligned),
            "obligation_aligned_fact_operator_ids": sorted(
                {_fact_operator_id(fact) for fact in obligation_aligned if _fact_operator_id(fact)}
            ),
            "binding_title_link": binding_link,
            "question_title_link": question_link,
            "base_context_title_link": base_context_link,
            "fact_endpoint_title_link": fact_endpoint_link,
            "source_identity_aligned": source_identity_aligned,
            "relation_content_signal": relation_signal,
            "answer_value_signal": answer_value_signal,
            "baseline_prediction_uncertain": baseline_uncertain,
            "comparison_plan": comparison_plan,
            "complete_candidate_witness_exists": has_complete_candidate_witness,
            "ordinary_text_only": True,
            "tuple_certified_by_selection": False,
        }
        if source_id in existing_ids:
            decision["decision"] = "DEDUPLICATED_ALREADY_IN_CONTEXT"
        elif source_id in excluded_ids:
            decision["decision"] = "REJECTED_CONTENT_IDENTITY_CONFLICT"
        elif not content:
            decision["decision"] = "NOT_LOGGED_SOURCE_CONTENT_UNAVAILABLE"
        elif has_complete_candidate_witness:
            decision["decision"] = "REJECTED_COMPLETE_WITNESS_ALREADY_EXISTS"
        elif supported_admitted and not obligation_aligned:
            decision["decision"] = "REJECTED_OBLIGATION_MISMATCH"
        elif obligation_aligned and not source_identity_aligned:
            decision["decision"] = "REJECTED_SOURCE_ENTITY_MISMATCH"
        elif obligation_aligned:
            decision["decision"] = "ELIGIBLE_OBLIGATION_ALIGNED_SUPPORTED_SOURCE"
            decision["priority_tier"] = 0
            eligible.append(((0, decision["event_order"], source_id), decision))
        elif (
            source_identity_aligned
            and (relation_signal or answer_value_signal)
            and baseline_uncertain
        ):
            decision["decision"] = "ELIGIBLE_UNCERTAIN_INCUMBENT_TEXTUAL_SOURCE"
            decision["priority_tier"] = 1
            eligible.append(((1, decision["event_order"], source_id), decision))
        elif source_identity_aligned and (relation_signal or answer_value_signal):
            decision["decision"] = "REJECTED_UNCERTIFIED_TEXT_WITH_STABLE_INCUMBENT"
        else:
            decision["decision"] = "REJECTED_NO_BINDING_OR_OBLIGATION_SIGNAL"
        decisions.append(decision)

    chosen = sorted(eligible, key=lambda item: item[0])[
        : max(0, int(max_selected_sources))
    ]
    chosen_ids = {int(item[1]["source_id"]) for item in chosen}
    visible_source_ids = existing_ids | chosen_ids
    branch_keys = _comparison_branch_keys(trace, visible_source_ids)
    branch_complete = not comparison_plan or len(branch_keys) >= 2
    branch_guard_applied = comparison_plan and not branch_complete and not baseline_uncertain
    if branch_guard_applied:
        chosen_ids.clear()

    selected_visits: list[dict[str, Any]] = []
    for decision in decisions:
        decision["visible_comparison_branch_count"] = len(branch_keys)
        decision["visible_comparison_branch_keys"] = sorted(branch_keys)
        decision["comparison_branch_complete"] = branch_complete
        if branch_guard_applied and str(decision.get("decision") or "").startswith(
            "ELIGIBLE_"
        ):
            decision["decision"] = "REJECTED_INCOMPLETE_COMPARISON_BRANCH_COVERAGE"
            decision["selection_decision"] = "NOT_SELECTED"
        elif int(decision["source_id"]) in chosen_ids:
            decision["selection_decision"] = "SELECTED_ORDINARY_TEXT_SIDECAR"
            selected_visits.append(
                next(
                    dict(visit)
                    for visit in visits
                    if int(visit["source_id"]) == int(decision["source_id"])
                )
            )
        elif str(decision.get("decision") or "").startswith("ELIGIBLE_"):
            decision["selection_decision"] = "ELIGIBLE_BUT_SOURCE_CAP_REACHED"
        else:
            decision["selection_decision"] = "NOT_SELECTED"
    priority_by_source = {
        int(decision["source_id"]): key for key, decision in eligible
    }
    selected_visits.sort(
        key=lambda visit: priority_by_source[int(visit["source_id"])]
    )
    return selected_visits, decisions


def build_obligation_scoped_evidence_reader_record(
    *,
    original_lazy_record: Mapping[str, Any],
    trace: Mapping[str, Any],
    visits: Sequence[Mapping[str, Any]],
    source_contents: Mapping[int, str],
    baseline_prediction: str,
    render_messages: Callable[[str, Sequence[str]], Sequence[Mapping[str, str]]],
    count_message_tokens: Callable[[Sequence[Mapping[str, str]]], int],
    token_counter_version: str,
    max_selected_sources: int,
    max_input_tokens: int,
    excluded_source_ids: Sequence[int] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    before = state_fingerprint(trace)
    selected_visits, selection_decisions = select_obligation_scoped_evidence_sources(
        original_lazy_record=original_lazy_record,
        trace=trace,
        visits=visits,
        source_contents=source_contents,
        baseline_prediction=baseline_prediction,
        max_selected_sources=max_selected_sources,
        excluded_source_ids=excluded_source_ids,
    )
    record, append_decisions = build_visited_source_reader_record(
        original_lazy_record=original_lazy_record,
        visits=selected_visits,
        source_contents=source_contents,
        render_messages=render_messages,
        count_message_tokens=count_message_tokens,
        token_counter_version=token_counter_version,
        max_extra_sources=max_selected_sources,
        max_input_tokens=max_input_tokens,
        excluded_source_ids=excluded_source_ids,
    )
    append_by_source = {
        int(row["source_id"]): dict(row) for row in append_decisions
    }
    for decision in selection_decisions:
        append_decision = append_by_source.get(int(decision["source_id"]))
        if append_decision:
            decision["render_decision"] = append_decision.get("decision")
            if append_decision.get("content_sha256"):
                decision["content_sha256"] = append_decision["content_sha256"]
    record.update(
        {
            "version": OBLIGATION_SCOPED_DIAGNOSTIC_VERSION,
            "reader_arm": OBLIGATION_SCOPED_DIAGNOSTIC_ARM,
            "diagnostic_renderer_version": OBLIGATION_SCOPED_DIAGNOSTIC_VERSION,
            "strict_visited_source_ids": [int(row["source_id"]) for row in visits],
            "selected_sidecar_source_ids": [
                int(row["source_id"]) for row in selected_visits
            ],
            "source_selection_decisions": selection_decisions,
            "baseline_prediction_sha256": sha256_text(str(baseline_prediction or "")),
            "baseline_prediction_uncertain": is_reader_prediction_uncertain(
                baseline_prediction
            ),
            "comparison_plan": _is_comparison_plan(trace),
            "answer_contract": infer_answer_contract(
                trace, str(original_lazy_record.get("question") or "")
            ),
            "structured_answer_admission": False,
            "ordinary_text_sidecar_only": True,
            "gold_used_online": False,
        }
    )
    if state_fingerprint(trace) != before:
        raise AssertionError("obligation_scoped_exit_mutated_execution_state")
    return record, selection_decisions


def _documents_from_reader_record(
    record: Mapping[str, Any],
) -> tuple[list[SourceDocument], list[SourceDocument]]:
    base_ids = {int(value) for value in record.get("base_source_ids") or []}
    base: list[SourceDocument] = []
    extra: list[SourceDocument] = []
    for position, raw in enumerate(record.get("actual_documents") or []):
        source_id = int(raw["source_id"])
        content = str(raw.get("content") or "").strip()
        if sha256_text(content) != str(raw.get("content_sha256") or ""):
            raise ValueError(f"historical_document_hash_mismatch:{source_id}")
        document = document_from_content(
            source_id,
            content,
            origin=str(raw.get("origin") or "historical_reader_context"),
            rank=position,
        )
        (base if source_id in base_ids else extra).append(document)
    if [document.source_id for document in base] != [
        int(value) for value in record.get("base_source_ids") or []
    ]:
        raise ValueError("historical_dense_prefix_mismatch")
    return base, extra


def build_visited_source_reader_record(
    *,
    original_lazy_record: Mapping[str, Any],
    visits: Sequence[Mapping[str, Any]],
    source_contents: Mapping[int, str],
    render_messages: Callable[[str, Sequence[str]], Sequence[Mapping[str, str]]],
    count_message_tokens: Callable[[Sequence[Mapping[str, str]]], int],
    token_counter_version: str,
    max_extra_sources: int,
    max_input_tokens: int,
    excluded_source_ids: Sequence[int] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Expose already visited source text under one fixed append-only policy."""
    trace_fingerprint = sha256_value(visits)
    historical_fingerprint = sha256_value(original_lazy_record)
    base, original_extra = _documents_from_reader_record(original_lazy_record)
    final = list(base) + list(original_extra)
    seen = {document.source_id for document in final}
    excluded = {int(value) for value in excluded_source_ids}
    decisions: list[dict[str, Any]] = []
    remaining = max(0, int(max_extra_sources) - len(original_extra))

    for raw_visit in visits:
        source_id = int(raw_visit["source_id"])
        decision = {
            "source_id": source_id,
            "event_order": int(raw_visit.get("event_order", len(decisions))),
            "visit_proof": str(raw_visit.get("proof") or ""),
        }
        if source_id in seen:
            decision["decision"] = "DEDUPLICATED_ALREADY_IN_CONTEXT"
        elif source_id in excluded:
            decision["decision"] = "REJECTED_CONTENT_IDENTITY_CONFLICT"
        elif remaining <= 0:
            decision["decision"] = "SOURCE_CAP_NO_ADDITIONAL_ROOM"
        else:
            content = str(source_contents.get(source_id) or "").strip()
            if not content:
                decision["decision"] = "NOT_LOGGED_SOURCE_CONTENT_UNAVAILABLE"
            else:
                final.append(
                    document_from_content(
                        source_id,
                        content,
                        origin="lazy_visited_source_diagnostic",
                        rank=len(final),
                    )
                )
                seen.add(source_id)
                remaining -= 1
                decision["decision"] = "APPENDED_VISITED_SOURCE"
                decision["content_sha256"] = sha256_text(content)
        decisions.append(decision)

    final, truncated, input_tokens = truncate_documents_to_token_budget(
        question=str(original_lazy_record["question"]),
        documents=final,
        render_messages=render_messages,
        count_message_tokens=count_message_tokens,
        max_input_tokens=max_input_tokens,
        protected_base_count=len(base),
    )
    messages = list(
        render_messages(
            str(original_lazy_record["question"]),
            [document.content for document in final],
        )
    )
    request = build_exact_request(
        model=str(original_lazy_record["model"]),
        messages=messages,
        seed=int(original_lazy_record["seed"]),
        temperature=float(original_lazy_record["temperature"]),
        max_completion_tokens=int(original_lazy_record["max_completion_tokens"]),
        n=int(original_lazy_record.get("n", 1)),
    )
    record = {
        "version": DIAGNOSTIC_VERSION,
        "dataset": original_lazy_record["dataset"],
        "source_index": int(original_lazy_record["source_index"]),
        "frozen_position": int(original_lazy_record["frozen_position"]),
        "query_id": original_lazy_record["query_id"],
        "question": original_lazy_record["question"],
        "question_sha256": original_lazy_record["question_sha256"],
        "plan_hash": original_lazy_record["plan_hash"],
        "reader_arm": DIAGNOSTIC_ARM,
        "execution_arm": original_lazy_record["execution_arm"],
        "base_source_ids": [document.source_id for document in base],
        "historical_lazy_extra_source_ids": [
            document.source_id for document in original_extra
        ],
        "strict_visited_source_ids": [int(row["source_id"]) for row in visits],
        "candidate_extra_source_ids": [
            int(row["source_id"])
            for row in visits
            if int(row["source_id"]) not in {
                document.source_id for document in base + original_extra
            }
        ],
        "source_selection_decisions": decisions,
        "truncated_documents": truncated,
        "actual_documents": [document.as_dict() for document in final],
        "actual_source_ids": [document.source_id for document in final],
        "actual_context_sha256": sha256_value(
            [document.as_dict() for document in final]
        ),
        "messages": request["messages"],
        "request_hash": request["request_hash"],
        "request_version": REQUEST_VERSION,
        "renderer_version": RENDERER_VERSION,
        "diagnostic_renderer_version": DIAGNOSTIC_VERSION,
        "model": original_lazy_record["model"],
        "seed": int(original_lazy_record["seed"]),
        "temperature": float(original_lazy_record["temperature"]),
        "max_completion_tokens": int(original_lazy_record["max_completion_tokens"]),
        "n": int(original_lazy_record.get("n", 1)),
        "input_tokens_estimate": input_tokens,
        "input_token_budget_upper_bound": conservative_message_token_upper_bound(
            messages
        ),
        "token_counter_version": token_counter_version,
        "structured_answer_admission": False,
        "gold_used_online": False,
        "frozen_trace_state_sha256": trace_fingerprint,
        "historical_lazy_record_sha256": historical_fingerprint,
    }
    if sha256_value(visits) != trace_fingerprint:
        raise AssertionError("visited_source_state_mutated")
    if sha256_value(original_lazy_record) != historical_fingerprint:
        raise AssertionError("historical_reader_record_mutated")
    return record, decisions


def reader_context_signature(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_ids": [int(value) for value in record.get("actual_source_ids") or []],
        "document_hashes": [
            str(row.get("content_sha256") or "")
            for row in record.get("actual_documents") or []
        ],
        "request_hash": str(record.get("request_hash") or ""),
    }


def state_fingerprint(trace: Mapping[str, Any]) -> str:
    """Fingerprint every structure the diagnostic is forbidden to mutate."""
    return sha256_value(
        {
            key: deepcopy(trace.get(key))
            for key in (
                "facts",
                "bindings",
                "supported_witnesses",
                "selected_witness",
                "question_complete_supported_witness",
                "structured_admission",
                "structured_answer",
            )
        }
    )
