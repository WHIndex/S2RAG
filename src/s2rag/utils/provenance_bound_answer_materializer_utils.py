import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .answer_verifier_utils import _doc_body, _doc_title, _is_supported, _split_sentences
from .candidate_answer_selector_utils import _is_synthetic_doc
from .cost_gated_reread_utils import (
    _answer_token_count,
    _evidence_ids,
    _extract_json_object,
    _surface_in_text,
)
from .eval_utils import normalize_answer
from .evidence_grounded_answer_extractor_utils import (
    BOOLEAN_QUESTION_RE,
    _candidate_passes_answer_slot,
    _infer_answer_slot,
    _is_containment_variant,
    _same_date_value,
)
from .planned_retrieval_utils import unpack_llm_result
from .sentence_evidence_utils import infer_answer_target_type


WORD_RE = re.compile(r"[a-z0-9]+", re.I)
TITLE_SPAN_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'.-]*(?:\s+|$)){1,7}",
)
VERBOSE_ANSWER_RE = re.compile(
    r"\b(?:because|therefore|according|which|who|that|she|he|they|it)\b|[.;:]",
    re.I,
)
TEMPORAL_QUALIFIER_RE = re.compile(
    r"\b(?:before|after|circa|about|approximately|around|prior\s+to|"
    r"no\s+later\s+than|no\s+earlier\s+than)\b",
    re.I,
)
TEMPORAL_SURFACE_RE = re.compile(
    r"^\s*(?:[0-3]?\d\s+)?(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+(?:[1-9]\d{2,3})\s*$|"
    r"^\s*(?:[1-9]\d{3}|20\d{2}|21\d{2})\s*$",
    re.I,
)



RELATION_PROFILES: Tuple[Tuple[str, re.Pattern, re.Pattern, Optional[str], Optional[str]], ...] = (
    (
        "landmark",
        re.compile(
            r"^(?:what|which)\b.{0,96}\b(?:bridge|river|lake|mountain|airport|"
            r"university|school|museum|building|monument)\b",
            re.I,
        ),
        re.compile(
            r"\b(?:bridge|river|lake|mountain|airport|university|school|"
            r"museum|building|monument)\b",
            re.I,
        ),
        "named_entity",
        "place",
    ),
    (
        "income",
        re.compile(r"^(?:what|which)\b.{0,48}\b(?:salary|salaries|income|earnings|wage|wages|pay)\b", re.I),
        re.compile(r"\b(?:salary|salaries|income|earnings|wage|wages|paid|pay)\b", re.I),
        "number",
        "number",
    ),
    (
        "birth_date",
        re.compile(r"\b(?:date of birth|birthdate|birthday|born on|when .* born)\b", re.I),
        re.compile(r"\b(?:date of birth|birthdate|birthday|born on|born)\b", re.I),
        "full_date",
        "date",
    ),
    (
        "death_date",
        re.compile(r"\b(?:date of death|death date|died on|when .* (?:die|died|dies))\b", re.I),
        re.compile(r"\b(?:date of death|death date|died on|die|died|dies)\b", re.I),
        "full_date",
        "date",
    ),
    (
        "death_cause",
        re.compile(r"\b(?:cause of death|died (?:from|of)|die (?:from|of)|what killed)\b", re.I),
        re.compile(r"\b(?:cause of death|died (?:from|of)|death (?:from|of|was caused by)|killed by)\b", re.I),
        "named_entity",
        "factoid",
    ),
    (
        "detention_facility",
        re.compile(r"\b(?:prison|jail|detained|incarcerated|imprisoned|held captive)\b", re.I),
        re.compile(r"\b(?:prison|jail|detained|incarcerated|imprisoned|held (?:at|in)|correctional)\b", re.I),
        "institution",
        "organization",
    ),
    (
        "birth_place",
        re.compile(r"\b(?:where .* born|birthplace|place of birth|born in)\b", re.I),
        re.compile(r"\b(?:born in|birthplace|place of birth)\b", re.I),
        "location",
        "location",
    ),
    (
        "death_place",
        re.compile(r"\b(?:where .* died|place of death|died in)\b", re.I),
        re.compile(r"\b(?:died in|place of death)\b", re.I),
        "location",
        "location",
    ),
    (
        "location",
        re.compile(r"\b(?:where|located|situated|headquartered|based in|capital of)\b", re.I),
        re.compile(r"\b(?:located|situated|headquartered|based in|capital of|lies in|is in)\b", re.I),
        "location",
        "location",
    ),
    (
        "nationality",
        re.compile(r"\b(?:nationality|citizenship|citizen|national|country .* from)\b", re.I),
        re.compile(r"\b(?:nationality|citizenship|citizen|national|born|from)\b", re.I),
        "nationality",
        "nationality",
    ),
    (
        "creator",
        re.compile(
            r"^(?:who|which person)\b.{0,96}\b(?:wrote|directed|composed|"
            r"created|produced|author|writer|director|composer|creator|producer)\b",
            re.I,
        ),
        re.compile(r"\b(?:written|wrote|directed|composed|created|produced|author|writer|director|composer|creator|producer|by)\b", re.I),
        "person",
        "person",
    ),
    (
        "family_relation",
        re.compile(r"\b(?:mother|father|parent|spouse|wife|husband|son|daughter|child|sibling)\b", re.I),
        re.compile(r"\b(?:mother|father|parent|spouse|wife|husband|married|son|daughter|child|sibling)\b", re.I),
        None,
        "person",
    ),
    (
        "affiliation",
        re.compile(r"\b(?:worked (?:at|for)|works (?:at|for)|employed by|member of|played for|studied at|graduated from)\b", re.I),
        re.compile(r"\b(?:worked (?:at|for)|works (?:at|for)|employed by|member of|played for|studied at|graduated from)\b", re.I),
        "organization",
        "organization",
    ),
    (
        "occupation",
        re.compile(r"\b(?:occupation|profession|job|worked as|known for)\b", re.I),
        re.compile(r"\b(?:occupation|profession|worked as|served as|was an?|is an?|known for)\b", re.I),
        "occupation",
        "occupation",
    ),
    (
        "role_title",
        re.compile(r"\b(?:position|office|role|title)\b", re.I),
        re.compile(r"\b(?:position|office|role|title|served as|appointed|elected)\b", re.I),
        "role_title",
        "person",
    ),
    (
        "work",
        re.compile(r"^(?:which|what)\s+(?:film|movie|book|novel|album|song|work)\b", re.I),
        re.compile(r"\b(?:film|movie|book|novel|album|song|work|released|published)\b", re.I),
        "work",
        "work",
    ),
)


def _new_meta(enabled: bool, answer: str) -> Dict[str, Any]:
    before = str(answer or "").strip()
    return {
        "enabled": bool(enabled),
        "eligible": False,
        "triggered": False,
        "changed": False,
        "action": "none" if enabled else "disabled",
        "reason": "not_evaluated" if enabled else "disabled",
        "before": before,
        "after": before,
    }


def _logical_plan_terminal_operator(logical_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    plan = dict(logical_plan or {})
    if plan.get("compile_status") != "compiled":
        return {}
    explicit_terminal = dict(plan.get("terminal_operator") or {})
    if (
        str(explicit_terminal.get("output_slot") or "") == "$answer"
        and str(explicit_terminal.get("slot_plan_stage") or "") == "final_answer"
    ):
        return explicit_terminal
    if str(plan.get("answer_operator") or "project") != "project":
        return {}
    candidates = [
        dict(operator)
        for operator in (plan.get("operators") or plan.get("steps") or [])
        if isinstance(operator, dict)
        and str(operator.get("output_slot") or "") == "$answer"
        and str(operator.get("slot_plan_stage") or "") == "final_answer"
    ]
    return candidates[-1] if candidates else {}


def _terminal_relation_family(operator: Dict[str, Any], output_type: str) -> str:
    predicates = [str(value) for value in operator.get("predicate_families", []) if value]
    relations = [str(value) for value in operator.get("relation_families", []) if value]
    relation_tokens = {
        str(value).strip().lower()
        for value in operator.get("relation_tokens", [])
        if str(value).strip()
    }
    if relation_tokens & {
        "bridge", "river", "lake", "mountain", "airport", "university",
        "school", "museum", "building", "monument",
    }:
        return "landmark"
    primary = predicates[0] if predicates else (relations[0] if relations else "terminal_relation")
    if primary == "birth":
        return "birth_place" if output_type in {"location", "place"} else "birth_date"
    if primary == "death":
        return "death_place" if output_type in {"location", "place"} else "death_date"
    return primary


def _terminal_relation_pattern(operator: Dict[str, Any]) -> str:
    values = list(operator.get("relation_tokens") or [])
    values.extend(operator.get("predicate_families") or [])
    values = sorted({str(value).strip().lower() for value in values if str(value).strip()})
    if not values:
        return ""
    return r"\b(?:" + "|".join(re.escape(value) for value in values) + r")\b"


def compile_answer_domain(
    question: str,
    answer_style: Optional[str] = None,
    *,
    logical_plan: Optional[Dict[str, Any]] = None,
    require_compiled_terminal_plan: bool = False,
) -> Dict[str, Any]:
    boolean_question = bool(BOOLEAN_QUESTION_RE.match(question or ""))
    target_type = (
        "boolean" if boolean_question
        else infer_answer_target_type(question or "")
    )
    answer_slot = (
        "boolean" if boolean_question
        else _infer_answer_slot(question or "", target_type)
    )
    relation_family = "comparison" if boolean_question else "generic_relation"
    evidence_pattern: Optional[re.Pattern] = None
    if not boolean_question:
        for family, question_pattern, relation_pattern, slot_override, type_override in RELATION_PROFILES:
            if not question_pattern.search(question or ""):
                continue
            relation_family = family
            evidence_pattern = relation_pattern
            if slot_override:
                answer_slot = slot_override
            if type_override:
                target_type = type_override
            break
    contract = {
        "active": not boolean_question,
        "target_type": target_type,
        "answer_slot": answer_slot,
        "relation_family": relation_family,
        "relation_pattern": evidence_pattern.pattern if evidence_pattern else "",
        "answer_style": answer_style,
        "contract_source": "question_surface",
    }
    plan = dict(logical_plan or {})
    terminal_operator = _logical_plan_terminal_operator(plan)
    if terminal_operator:
        plan_target_type = str(
            terminal_operator.get("output_type")
            or plan.get("final_answer_type")
            or target_type
        )
        if not _answer_types_compatible(target_type, plan_target_type):
            terminal_operator = {}
            contract.update({
                "active": False if require_compiled_terminal_plan else contract["active"],
                "contract_source": "logical_plan_rejected",
                "contract_reason": "outer_answer_type_mismatch",
                "plan_target_type": plan_target_type,
                "logical_plan_revision": str(plan.get("logical_plan_revision") or ""),
                "plan_kind": str(plan.get("plan_kind") or ""),
                "answer_operator": str(plan.get("answer_operator") or ""),
            })
            return contract
        target_type = plan_target_type
        contract.update({
            "active": not boolean_question,
            "target_type": target_type,
            "answer_slot": (
                "boolean" if boolean_question
                else _infer_answer_slot(question or "", target_type)
            ),
            "relation_family": _terminal_relation_family(terminal_operator, target_type),
            "relation_pattern": _terminal_relation_pattern(terminal_operator),
            "contract_source": "typed_logical_plan",
            "logical_plan_revision": str(plan.get("logical_plan_revision") or ""),
            "plan_kind": str(plan.get("plan_kind") or ""),
            "answer_operator": str(plan.get("answer_operator") or "project"),
            "terminal_operator_type": str(
                terminal_operator.get("operator_type") or "project"
            ),
            "terminal_provenance_mode": str(
                terminal_operator.get("provenance_mode") or ""
            ),
            "terminal_operator_id": str(terminal_operator.get("operator_id") or ""),
            "terminal_input_slot": str(terminal_operator.get("input_slot") or ""),
            "terminal_output_slot": str(terminal_operator.get("output_slot") or ""),
            "terminal_relation_tokens": list(terminal_operator.get("relation_tokens") or []),
            "terminal_predicate_families": list(
                terminal_operator.get("predicate_families") or []
            ),
            "intermediate_output_slots": [
                str(operator.get("output_slot") or "")
                for operator in (plan.get("operators") or [])
                if isinstance(operator, dict)
                and str(operator.get("slot_plan_stage") or "") == "intermediate_binding"
            ],
        })
        return contract
    if require_compiled_terminal_plan:
        contract.update({
            "active": False,
            "contract_source": "logical_plan_rejected",
            "contract_reason": (
                "non_projection_answer_operator"
                if plan.get("compile_status") == "compiled"
                and str(plan.get("answer_operator") or "project") != "project"
                else "compiled_terminal_operator_missing"
            ),
            "logical_plan_revision": str(plan.get("logical_plan_revision") or ""),
            "plan_kind": str(plan.get("plan_kind") or ""),
            "answer_operator": str(plan.get("answer_operator") or ""),
        })
    return contract


def _answer_types_compatible(outer_type: str, plan_type: str) -> bool:
    outer = str(outer_type or "").strip().lower()
    planned = str(plan_type or "").strip().lower()
    if not outer or outer in {"factoid", "entity", "named_entity"}:
        return True
    if outer == planned:
        return True
    aliases = (
        {"location", "place"},
        {"date", "year"},
        {"organization", "institution"},
    )
    return any(outer in group and planned in group for group in aliases)


def _terminal_relation_matches_outer_contract(
    outer_contract: Dict[str, Any],
    terminal_operator: Dict[str, Any],
    plan_type: str,
) -> bool:
    """Reject a plan that ends at an intermediate relation of the question.

    Most legacy contracts are intentionally permissive because comparison and
    reduction plans project a root value after evaluating another relation.
    Landmark questions are different: the requested object class is explicit,
    so a birthplace tuple cannot satisfy a bridge obligation even though both
    values share the broad ``place`` type.
    """
    outer_family = str(outer_contract.get("relation_family") or "").strip().lower()
    if outer_family not in {"landmark"}:
        return True
    terminal_family = _terminal_relation_family(terminal_operator, plan_type)
    return terminal_family == outer_family


def select_authoritative_logical_query_plan(
    question: str,
    answer_style: Optional[str],
    plan_candidates: Sequence[Tuple[str, Optional[Dict[str, Any]]]],
) -> Dict[str, Any]:
    """Choose one typed plan whose terminal output agrees with the outer question.

    Schema execution and the adaptive executor historically compiled separate
    plans.  This selector makes their hand-off explicit and rejects a nested
    relation plan when it does not produce the question's outer answer column.
    """
    outer_contract = compile_answer_domain(question, answer_style)
    diagnostics: List[Dict[str, Any]] = []
    for source, raw_plan in plan_candidates:
        plan = dict(raw_plan or {})
        terminal = _logical_plan_terminal_operator(plan)
        if not terminal:
            diagnostics.append({
                "source": str(source),
                "accepted": False,
                "reason": "compiled_terminal_operator_missing",
                "compile_status": str(plan.get("compile_status") or "missing"),
            })
            continue
        plan_type = str(
            terminal.get("output_type") or plan.get("final_answer_type") or ""
        ).strip().lower()
        if not _answer_types_compatible(outer_contract.get("target_type"), plan_type):
            diagnostics.append({
                "source": str(source),
                "accepted": False,
                "reason": "outer_answer_type_mismatch",
                "outer_target_type": str(outer_contract.get("target_type") or ""),
                "plan_target_type": plan_type,
            })
            continue
        if not _terminal_relation_matches_outer_contract(
            outer_contract,
            terminal,
            plan_type,
        ):
            diagnostics.append({
                "source": str(source),
                "accepted": False,
                "reason": "outer_answer_relation_mismatch",
                "outer_relation_family": str(
                    outer_contract.get("relation_family") or ""
                ),
                "plan_terminal_relation_family": _terminal_relation_family(
                    terminal,
                    plan_type,
                ),
                "outer_target_type": str(outer_contract.get("target_type") or ""),
                "plan_target_type": plan_type,
            })
            continue
        selected = dict(plan)
        selected.update({
            "query_plan_ir_version": "unified_query_plan_ir_v2",
            "query_plan_source": str(source),
            "outer_contract_validated": True,
            "outer_answer_contract": dict(outer_contract),
            "plan_candidate_diagnostics": diagnostics + [{
                "source": str(source),
                "accepted": True,
                "reason": "outer_answer_contract_validated",
                "outer_target_type": str(outer_contract.get("target_type") or ""),
                "plan_target_type": plan_type,
            }],
        })
        return selected
    return {
        "enabled": True,
        "compile_status": "unsupported",
        "compile_reason": "no_outer_contract_compatible_plan",
        "query_plan_ir_version": "unified_query_plan_ir_v2",
        "query_plan_source": "none",
        "outer_contract_validated": False,
        "outer_answer_contract": dict(outer_contract),
        "plan_candidate_diagnostics": diagnostics,
        "operators": [],
        "steps": [],
        "operator_count": 0,
    }


def _terminal_provenance_docs(
    rows: Optional[Sequence[Dict[str, Any]]],
    contract: Dict[str, Any],
    max_rows: int = 3,
) -> Tuple[List[str], List[str]]:
    docs: List[str] = []
    candidates: List[str] = []
    target_type = str(contract.get("target_type") or "")
    terminal_predicates = set(contract.get("terminal_predicate_families") or [])
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("slot_plan_stage") or "") != "final_answer":
            continue
        if not bool(row.get("operator_complete") or row.get("slot_dependency_validated")):
            continue
        if not bool(row.get("rooted_lineage_complete")):
            continue
        if not bool(row.get("production_eligible")):
            continue
        if target_type and str(row.get("final_answer_type") or "") not in {"", target_type}:
            continue
        row_predicates = set(row.get("candidate_predicate_families") or [])
        if terminal_predicates and not (terminal_predicates & row_predicates):
            continue
        candidate = str(row.get("candidate") or row.get("raw_candidate") or "").strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
        evidence = re.sub(r"\s+", " ", str(row.get("evidence") or "")).strip()
        if (
            evidence
            and evidence not in docs
            and len(docs) < max(1, int(max_rows))
        ):
            docs.append(evidence)
    return docs, candidates


def _schema_terminal_provenance_docs(
    rows: Optional[Sequence[Dict[str, Any]]],
    *,
    question: str = "",
    contract: Optional[Dict[str, Any]] = None,
    min_score: float = 0.80,
    max_rows: int = 3,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Build a late-materialization domain from executed Schema Memory tuples."""
    docs: List[str] = []
    candidates: List[str] = []
    rejected: Dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    contract = dict(contract or {})
    target_type = str(contract.get("target_type") or "").strip().lower()
    terminal_operator_id = str(contract.get("terminal_operator_id") or "").strip()
    terminal_predicates = {
        str(value or "").strip().lower()
        for value in contract.get("terminal_predicate_families") or []
        if str(value or "").strip()
    }
    for raw_row in rows or []:
        if not isinstance(raw_row, dict):
            continue
        row = dict(raw_row)
        if str(row.get("materialization_role") or "") != "instance_bound_terminal_fact_join":
            reject("not_instance_bound_terminal")
            continue
        if str(row.get("logical_plan_stage") or "") != "final_answer":
            reject("not_final_answer_stage")
            continue
        if terminal_operator_id and str(row.get("terminal_operator_id") or "") != terminal_operator_id:
            reject("terminal_operator_mismatch")
            continue
        row_type = str(row.get("answer_type") or row.get("final_answer_type") or "").strip().lower()
        if target_type and row_type and not _answer_types_compatible(target_type, row_type):
            reject("outer_answer_type_mismatch")
            continue
        row_predicates = {
            str(value or "").strip().lower()
            for value in (
                list(row.get("predicate_families") or [])
                + list(row.get("candidate_predicate_families") or [])
                + list(row.get("predicate_subtypes") or [])
            )
            if str(value or "").strip()
        }
        row_predicate_roots = {value.split("_", 1)[0] for value in row_predicates}
        if terminal_predicates and not (
            terminal_predicates & row_predicates
            or terminal_predicates & row_predicate_roots
        ):
            reject("terminal_predicate_mismatch")
            continue
        if not all(bool(row.get(field)) for field in (
            "operator_complete",
            "strict_operator_contract_validated",
            "rooted_lineage_complete",
            "terminal_slot_match",
            "answer_type_match",
            "root_identity_safe_for_commit",
        )):
            reject("incomplete_terminal_contract")
            continue
        if float(row.get("score", 0.0) or 0.0) < float(min_score):
            reject("score_below_threshold")
            continue
        candidate = str(row.get("answer_value") or "").strip()
        evidence = re.sub(
            r"\s+",
            " ",
            str(row.get("schema_terminal_evidence") or row.get("lineage_reader_text") or ""),
        ).strip()
        if not candidate or not evidence:
            reject("candidate_or_evidence_missing")
            continue
        slot_ok, slot_reason = _candidate_passes_answer_slot(
            candidate,
            str(contract.get("answer_slot") or "named_entity"),
            target_type or "factoid",
            question=question,
            evidence_text=evidence,
        )
        if not slot_ok:
            reject(f"answer_slot_rejected:{slot_reason or 'unknown'}")
            continue
        if not _surface_in_text(candidate, evidence, str(row.get("answer_type") or "")):
            reject("candidate_not_in_schema_evidence")
            continue
        if candidate not in candidates:
            candidates.append(candidate)
        if evidence not in docs and len(docs) < max(1, int(max_rows)):
            docs.append(evidence)

    candidate_keys = {normalize_answer(value) for value in candidates if normalize_answer(value)}
    return docs, candidates, {
        "input_row_count": len(list(rows or [])),
        "accepted_row_count": len(docs),
        "candidate_class_count": len(candidate_keys),
        "unique_terminal_value": len(candidate_keys) == 1,
        "reject_reason_counts": rejected,
    }


def _terminal_completion_provenance_docs(
    rows: Optional[Sequence[Dict[str, Any]]],
    max_rows: int = 4,
) -> Tuple[List[str], List[str]]:
    """Expose only executor-produced, lineage-complete near-terminal tuples."""
    docs: List[str] = []
    candidates: List[str] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if not bool(row.get("terminal_tuple_completion_candidate")):
            continue
        if not all(bool(row.get(field)) for field in (
            "operator_complete",
            "slot_dependency_validated",
            "rooted_lineage_complete",
            "bridge_bound",
            "final_predicate_match",
            "root_identity_safe_for_commit",
            "completion_lineage_source_valid",
        )):
            continue
        candidate = str(
            row.get("candidate") or row.get("raw_candidate") or ""
        ).strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
        evidence = re.sub(r"\s+", " ", str(row.get("evidence") or "")).strip()
        if evidence and evidence not in docs and len(docs) < max(1, int(max_rows)):
            docs.append(evidence)
    return docs, candidates


def _deterministic_schema_terminal_projection_gate(
    *,
    question: str,
    before: str,
    contract: Dict[str, Any],
    rows: Optional[Sequence[Dict[str, Any]]],
    candidates: Sequence[str],
    docs: Sequence[str],
    trace: Dict[str, Any],
    incumbent_supported: bool,
    incumbent_slot_valid: bool,
) -> Dict[str, Any]:
    decision: Dict[str, Any] = {
        "accepted": False,
        "reason": "schema_terminal_tuple_unavailable",
        "decision_source": "schema_terminal_tuple_projection_v22",
        "input_row_count": int(trace.get("input_row_count", 0) or 0),
        "accepted_row_count": int(trace.get("accepted_row_count", 0) or 0),
    }
    candidate_classes = {
        normalize_answer(value): str(value).strip()
        for value in candidates
        if normalize_answer(value)
    }
    if len(candidate_classes) != 1:
        decision["reason"] = "schema_terminal_value_not_unique"
        return decision
    candidate = next(iter(candidate_classes.values()))
    if normalize_answer(candidate) == normalize_answer(before):
        decision["reason"] = "schema_terminal_same_as_incumbent"
        return decision
    target_type = str(contract.get("target_type") or "factoid")
    answer_slot = str(contract.get("answer_slot") or "named_entity")
    slot_ok, slot_reason = _candidate_passes_answer_slot(
        candidate,
        answer_slot,
        target_type,
        question=question,
        evidence_text="\n".join(docs),
    )
    if not slot_ok:
        decision["reason"] = f"schema_terminal_slot_rejected:{slot_reason}"
        return decision
    candidate_supported, candidate_support = _is_supported(
        question,
        candidate,
        docs,
        target_type,
        len(docs),
    )
    if not candidate_supported:
        decision["reason"] = "schema_terminal_candidate_not_supported"
        return decision
    if (
        incumbent_supported
        and incumbent_slot_valid
        and target_type == "date"
        and _same_date_value(before, candidate)
    ):
        decision["reason"] = "equivalent_supported_incumbent_protected"
        return decision
    containment_sensitive_slots = {
        "city", "county", "region", "location", "full_date", "date", "year",
        "person", "organization", "institution", "work", "named_entity",
    }
    if (
        incumbent_supported
        and incumbent_slot_valid
        and answer_slot in containment_sensitive_slots
        and _is_containment_variant(before, candidate)
    ):
        decision["reason"] = "containment_supported_incumbent_protected"
        return decision
    matching_scores = [
        float(row.get("score", row.get("utility", 0.0)) or 0.0)
        for row in rows or []
        if isinstance(row, dict)
        and normalize_answer(row.get("answer_value") or "")
        == normalize_answer(candidate)
    ]
    decision.update({
        "accepted": True,
        "reason": "strict_unique_schema_terminal_tuple",
        "candidate": candidate,
        "candidate_support_score": round(float(candidate_support or 0.0), 4),
        "table_score": round(max(matching_scores or [0.0]), 6),
        "evidence_ids": ["E1"] if docs else [],
    })
    return decision


def _terminal_surface_equivalence_key(row: Dict[str, Any]) -> str:
    """Cluster raw/materialized aliases from the same physical tuple.

    The surface materializer records both values explicitly.  Using that
    lineage-backed alias is safer than globally collapsing location prefixes
    such as ``Cambridge`` across unrelated passages.
    """
    values = [
        str(row.get("candidate") or "").strip(),
        str(row.get("raw_candidate") or "").strip(),
    ]
    surface = dict(row.get("surface_materializer") or {})
    values.extend([
        str(surface.get("raw_candidate") or "").strip(),
        str(surface.get("surface_candidate") or "").strip(),
    ])
    normalized = sorted({normalize_answer(value) for value in values if value})
    return normalized[0] if normalized else ""


def _provenance_terminal_tuple_completion_gate(
    *,
    question: str,
    before: str,
    contract: Dict[str, Any],
    rows: Optional[Sequence[Dict[str, Any]]],
    validated_terminal_row_count: int,
    incumbent_supported: bool,
    incumbent_binding: float,
    terminal_docs: Sequence[str],
    min_net_utility: float,
) -> Dict[str, Any]:
    """Complete $answer from an executed bridge-to-terminal tuple.

    These rows are not arbitrary answer candidates: the fact-join executor
    reached the terminal predicate from a bound bridge and retained the exact
    passage/triple lineage.  This gate rechecks the answer slot, ambiguity,
    provenance, and physical utility before allowing a projection.
    """
    decision: Dict[str, Any] = {
        "enabled": True,
        "accepted": False,
        "reason": "no_terminal_tuple_completion_candidates",
        "input_row_count": len(list(rows or [])),
        "validated_terminal_row_count": int(validated_terminal_row_count),
        "validated_terminal_rows_present": bool(validated_terminal_row_count),
        "structural_row_count": 0,
        "slot_valid_row_count": 0,
        "candidate_class_count": 0,
        "min_net_utility": round(float(min_net_utility), 6),
    }
    target_type = str(contract.get("target_type") or "factoid")
    answer_slot = str(contract.get("answer_slot") or "named_entity")
    terminal_predicates = set(contract.get("terminal_predicate_families") or [])
    structural_rows: List[Dict[str, Any]] = []
    for raw_row in rows or []:
        if not isinstance(raw_row, dict):
            continue
        row = dict(raw_row)
        if not bool(row.get("terminal_tuple_completion_candidate")):
            continue
        if not all(bool(row.get(field)) for field in (
            "operator_complete",
            "slot_dependency_validated",
            "rooted_lineage_complete",
            "bridge_bound",
            "final_predicate_match",
            "root_identity_safe_for_commit",
            "completion_lineage_source_valid",
        )):
            continue
        if str(row.get("terminal_output_slot") or "") != "$answer":
            continue
        if int(row.get("terminal_source_doc_id", -1) or -1) < 0:
            continue
        if not str(row.get("terminal_source_sentence_span") or "").strip():
            continue
        if bool(row.get("terminal_compound_recovery_required")) and not bool(
            row.get("terminal_compound_surface_recovered")
        ):
            continue
        if target_type and str(row.get("final_answer_type") or "") not in {
            "", target_type,
        }:
            continue
        predicates = set(row.get("candidate_predicate_families") or [])
        if terminal_predicates and not (terminal_predicates & predicates):
            continue
        candidate = str(
            row.get("candidate") or row.get("raw_candidate") or ""
        ).strip()
        if target_type in {"place", "location"} and TEMPORAL_SURFACE_RE.match(
            candidate
        ):
            continue
        context = _terminal_row_context(row)
        if not candidate or not _surface_in_text(candidate, context, target_type):
            continue
        if _drops_terminal_qualifier(candidate, context):
            continue
        structural_rows.append(row)

    decision["structural_row_count"] = len(structural_rows)
    if not structural_rows:
        decision["reason"] = (
            "validated_terminal_rows_present"
            if validated_terminal_row_count > 0
            else "no_lineage_complete_terminal_tuple"
        )
        return decision

    scored_rows: List[Dict[str, Any]] = []
    slot_reasons: Dict[str, int] = {}
    for row in structural_rows:
        candidate = str(
            row.get("candidate") or row.get("raw_candidate") or ""
        ).strip()
        context = _terminal_row_context(row)
        slot_ok, slot_reason = _candidate_passes_answer_slot(
            candidate,
            answer_slot,
            target_type,
            question=question,
            evidence_text=context,
        )
        if not slot_ok:
            slot_reasons[slot_reason] = slot_reasons.get(slot_reason, 0) + 1
            continue
        role_confidence = min([
            float(match.get("role_direction_confidence", 1.0) or 1.0)
            for match in (row.get("role_direction_matches") or [])
        ] or [1.0])
        utility = min(1.0, (
            0.35 * float(row.get("table_score", 0.0) or 0.0)
            + 0.25 * float(row.get("root_binding_confidence", 0.0) or 0.0)
            + 0.25 * role_confidence
            + 0.15
        ))
        risk = 0.08
        risk += 0.10 if int(row.get("terminal_source_sentence_id", -1) or -1) < 0 else 0.0
        risk += 0.08 if bool(row.get("surface_materialized")) else 0.0
        risk += 0.06 if bool(row.get("bound_completion_used")) else 0.0
        normalized_cost = min(1.0, float(row.get("cost", 0.0) or 0.0))
        net_utility = utility - 0.62 * risk - 0.08 * normalized_cost
        candidate_binding = _binding_score(
            question,
            candidate,
            [context],
            contract,
        )
        scored = dict(row)
        scored.update({
            "completion_utility": round(float(utility), 6),
            "completion_risk": round(float(risk), 6),
            "completion_net_utility": round(float(net_utility), 6),
            "completion_binding_score": round(float(candidate_binding), 6),
            "completion_equivalence_key": _terminal_surface_equivalence_key(row),
        })
        scored_rows.append(scored)

    decision["slot_valid_row_count"] = len(scored_rows)
    decision["slot_reject_reason_counts"] = slot_reasons
    if not scored_rows:
        decision["reason"] = "terminal_tuple_answer_slot_rejected"
        return decision

    candidate_classes: Dict[str, List[Dict[str, Any]]] = {}
    for row in scored_rows:
        key = str(row.get("completion_equivalence_key") or "")
        if key:
            candidate_classes.setdefault(key, []).append(row)
    decision["candidate_class_count"] = len(candidate_classes)
    decision["candidate_surfaces"] = sorted({
        str(row.get("candidate") or row.get("raw_candidate") or "").strip()
        for row in scored_rows
    })
    if len(candidate_classes) != 1:
        decision["reason"] = "terminal_tuple_candidate_conflict"
        return decision

    candidate_rows = next(iter(candidate_classes.values()))
    admissible = [
        row for row in candidate_rows
        if float(row.get("completion_net_utility", -1.0) or -1.0)
        >= float(min_net_utility)
    ]
    if not admissible:
        decision["reason"] = "terminal_tuple_risk_cost_rejected"
        return decision
    selected = max(
        admissible,
        key=lambda row: (
            float(row.get("completion_net_utility", 0.0) or 0.0),
            float(row.get("completion_binding_score", 0.0) or 0.0),
            -int(row.get("answer_rank", 999999) or 999999),
        ),
    )
    candidate = str(
        selected.get("candidate") or selected.get("raw_candidate") or ""
    ).strip()
    candidate_binding = float(
        selected.get("completion_binding_score", 0.0) or 0.0
    )
    if normalize_answer(candidate) == normalize_answer(before):
        decision["reason"] = "terminal_tuple_same_as_incumbent"
        return decision
    if incumbent_supported:
        decision["reason"] = "supported_incumbent_protected"
        return decision

    evidence = re.sub(r"\s+", " ", str(selected.get("evidence") or "")).strip()
    evidence_id = next((
        index + 1
        for index, value in enumerate(terminal_docs)
        if re.sub(r"\s+", " ", str(value or "")).strip() == evidence
    ), 1)
    decision.update({
        "accepted": True,
        "reason": "unique_provenance_bound_terminal_tuple",
        "candidate": candidate,
        "evidence_ids": [int(evidence_id)],
        "net_utility": round(
            float(selected.get("completion_net_utility", 0.0) or 0.0), 6
        ),
        "utility": round(
            float(selected.get("completion_utility", 0.0) or 0.0), 6
        ),
        "risk": round(
            float(selected.get("completion_risk", 0.0) or 0.0), 6
        ),
        "table_score": round(float(selected.get("table_score", 0.0) or 0.0), 6),
        "binding_score": round(candidate_binding, 6),
        "source_doc_ids": list(selected.get("source_doc_ids") or []),
        "source_sentence_id": int(
            selected.get("terminal_source_sentence_id", -1) or -1
        ),
        "source_sentence_span": str(
            selected.get("terminal_source_sentence_span") or ""
        ),
        "terminal_bound_surface": str(
            selected.get("terminal_bound_surface") or ""
        ),
        "terminal_operator_id": str(contract.get("terminal_operator_id") or ""),
        "decision_source": "provenance_terminal_tuple_completion_v12_3",
    })
    return decision


def _terminal_row_context(row: Dict[str, Any]) -> str:
    return "\n".join(
        value for value in (
            str(row.get("evidence") or "").strip(),
            str(row.get("triple") or "").strip(),
            str(row.get("lineage_sentence_preview") or "").strip(),
        )
        if value
    )


def _drops_terminal_qualifier(candidate: str, context: str) -> bool:
    """Reject a projected value that strips an evidential qualifier.

    OpenIE can represent ``born before 1264`` as relation ``born before`` and
    object ``1264``.  Projecting only the object is not provenance faithful,
    even when a benchmark happens to use the unqualified year.
    """
    candidate = str(candidate or "").strip()
    if not candidate or not context:
        return False
    candidate_pattern = re.escape(candidate).replace(r"\ ", r"\s+")
    qualified_candidate = re.compile(
        TEMPORAL_QUALIFIER_RE.pattern
        + r"(?:[\s|:;,()\[\]-]{0,24})"
        + candidate_pattern,
        re.I,
    )
    return bool(qualified_candidate.search(context))


def _deterministic_terminal_projection_gate(
    *,
    question: str,
    before: str,
    contract: Dict[str, Any],
    rows: Optional[Sequence[Dict[str, Any]]],
    incumbent_slot_valid: bool,
    terminal_docs: Sequence[str],
    min_net_utility: float,
) -> Dict[str, Any]:
    """Admit a unique, fully validated terminal tuple without another LLM.

    This is intentionally stricter than the LLM fallback.  Every structurally
    valid row participates in ambiguity detection, while the selected row must
    also pass the physical cost decision and preserve any evidence qualifier.
    """
    decision: Dict[str, Any] = {
        "enabled": True,
        "accepted": False,
        "reason": "no_validated_terminal_rows",
        "input_row_count": len(list(rows or [])),
        "structural_row_count": 0,
        "candidate_class_count": 0,
        "min_net_utility": round(float(min_net_utility), 6),
    }
    target_type = str(contract.get("target_type") or "factoid")
    answer_slot = str(contract.get("answer_slot") or "named_entity")
    terminal_predicates = set(contract.get("terminal_predicate_families") or [])
    structural_rows: List[Dict[str, Any]] = []
    for raw_row in rows or []:
        if not isinstance(raw_row, dict):
            continue
        row = dict(raw_row)
        candidate = str(row.get("candidate") or row.get("raw_candidate") or "").strip()
        predicates = set(row.get("candidate_predicate_families") or [])
        if not candidate:
            continue
        if str(row.get("slot_plan_stage") or "") != "final_answer":
            continue
        if not all(bool(row.get(field)) for field in (
            "operator_complete",
            "slot_dependency_validated",
            "rooted_lineage_complete",
            "production_eligible",
            "final_predicate_match",
            "final_answer_type_match",
            "root_identity_safe_for_commit",
        )):
            continue
        if target_type and str(row.get("final_answer_type") or "") not in {"", target_type}:
            continue
        if terminal_predicates and not (terminal_predicates & predicates):
            continue
        structural_rows.append(row)

    decision["structural_row_count"] = len(structural_rows)
    if not structural_rows:
        return decision

    candidate_classes: Dict[str, List[Dict[str, Any]]] = {}
    for row in structural_rows:
        candidate = str(row.get("candidate") or row.get("raw_candidate") or "").strip()
        key = normalize_answer(candidate)
        if key:
            candidate_classes.setdefault(key, []).append(row)
    decision["candidate_class_count"] = len(candidate_classes)
    decision["candidate_surfaces"] = sorted({
        str(row.get("candidate") or row.get("raw_candidate") or "").strip()
        for row in structural_rows
        if str(row.get("candidate") or row.get("raw_candidate") or "").strip()
    })
    if len(candidate_classes) != 1:
        decision["reason"] = "terminal_candidate_conflict"
        return decision
    if incumbent_slot_valid:
        decision["reason"] = "incumbent_slot_already_valid"
        return decision

    candidate_rows = next(iter(candidate_classes.values()))
    if any(
        bool((row.get("provenance_store") or {}).get("ambiguous_conflict"))
        or int((row.get("provenance_store") or {}).get("conflict_candidate_count", 0) or 0) > 1
        for row in candidate_rows
    ):
        decision["reason"] = "terminal_provenance_conflict"
        return decision
    if any(
        _drops_terminal_qualifier(
            str(row.get("candidate") or row.get("raw_candidate") or ""),
            _terminal_row_context(row),
        )
        for row in candidate_rows
    ):
        decision["reason"] = "terminal_qualifier_loss"
        return decision

    admissible_rows = [
        row for row in candidate_rows
        if row.get("cost_accepted") is True
        and float(row.get("net_utility", float("-inf")) or float("-inf"))
        >= float(min_net_utility)
    ]
    if not admissible_rows:
        decision["reason"] = "terminal_cost_or_utility_rejected"
        return decision
    selected = max(
        admissible_rows,
        key=lambda row: (
            float(row.get("net_utility", 0.0) or 0.0),
            float(row.get("table_score", 0.0) or 0.0),
            -int(row.get("rank", 999999) or 999999),
        ),
    )
    candidate = str(
        selected.get("candidate") or selected.get("raw_candidate") or ""
    ).strip()
    context = _terminal_row_context(selected)
    if normalize_answer(candidate) == normalize_answer(before):
        decision["reason"] = "terminal_candidate_same_as_incumbent"
        return decision
    if not _surface_in_text(candidate, context, target_type):
        decision["reason"] = "terminal_candidate_not_in_provenance"
        return decision
    slot_ok, slot_reason = _candidate_passes_answer_slot(
        candidate,
        answer_slot,
        target_type,
        question=question,
        evidence_text=context,
    )
    if not slot_ok:
        decision.update({
            "reason": "terminal_candidate_slot_rejected",
            "slot_reject_reason": slot_reason,
        })
        return decision

    evidence = re.sub(r"\s+", " ", str(selected.get("evidence") or "")).strip()
    evidence_id = next(
        (
            index + 1
            for index, value in enumerate(terminal_docs)
            if re.sub(r"\s+", " ", str(value or "")).strip() == evidence
        ),
        1,
    )
    decision.update({
        "accepted": True,
        "reason": "unique_cost_accepted_terminal_tuple",
        "candidate": candidate,
        "evidence_ids": [int(evidence_id)],
        "net_utility": round(float(selected.get("net_utility", 0.0) or 0.0), 6),
        "table_score": round(float(selected.get("table_score", 0.0) or 0.0), 6),
        "source_doc_ids": list(selected.get("source_doc_ids") or []),
        "terminal_operator_id": str(contract.get("terminal_operator_id") or ""),
    })
    return decision


def _relation_pattern(contract: Dict[str, Any]) -> Optional[re.Pattern]:
    pattern = str(contract.get("relation_pattern") or "")
    return re.compile(pattern, re.I) if pattern else None


def _question_anchor_tokens(question: str) -> set:
    spans = []
    for match in TITLE_SPAN_RE.finditer(question or ""):
        value = match.group(0).strip()
        value = re.sub(r"^(?:Who|What|Which|Where|When|How|The|A|An)\s+", "", value)
        spans.append(value)
    tokens = {
        token.lower()
        for span in spans
        for token in WORD_RE.findall(span)
        if len(token) > 2
    }
    return tokens


def _compact_docs(
    docs: Sequence[str],
    max_docs: int,
    max_doc_chars: int,
    max_total_chars: int,
) -> Tuple[List[str], List[str]]:
    compact: List[str] = []
    rendered: List[str] = []
    used = 0
    for doc in docs or []:
        if len(compact) >= max(1, int(max_docs)):
            break
        if not isinstance(doc, str) or not doc.strip() or _is_synthetic_doc(doc):
            continue
        clean = re.sub(r"\s+", " ", doc).strip()
        remaining = int(max_total_chars) - used
        if remaining <= 0:
            break
        clean = clean[: min(int(max_doc_chars), remaining)]
        compact.append(clean)
        rendered.append(f"[E{len(compact)}] {clean}")
        used += len(clean)
    return compact, rendered


def _binding_score(
    question: str,
    candidate: str,
    docs: Sequence[str],
    contract: Dict[str, Any],
) -> float:
    if not candidate:
        return 0.0
    pattern = _relation_pattern(contract)
    anchors = _question_anchor_tokens(question)
    best = 0.0
    for doc in docs or []:
        title = _doc_title(doc)
        title_tokens = {token.lower() for token in WORD_RE.findall(title) if len(token) > 2}
        for sentence in _split_sentences(_doc_body(doc)):
            if not _surface_in_text(candidate, f"{title} {sentence}", str(contract.get("target_type") or "")):
                continue
            relation_hit = bool(pattern and pattern.search(sentence))
            surface_tokens = {token.lower() for token in WORD_RE.findall(f"{title} {sentence}") if len(token) > 2}
            anchor_hit = bool(anchors & (surface_tokens | title_tokens))
            score = 0.34 + (0.40 if relation_hit else 0.0) + (0.18 if anchor_hit else 0.0)
            if contract.get("relation_family") == "generic_relation":
                score = 0.34 + (0.20 if anchor_hit else 0.0)
            best = max(best, min(1.0, score))
    return best


def _candidate_surface_count(raw_response: str) -> int:
    values = {
        normalize_answer(match.group(0))
        for match in TITLE_SPAN_RE.finditer(raw_response or "")
        if len(normalize_answer(match.group(0)).split()) <= 7
    }
    return len({value for value in values if value})


def prepare_provenance_bound_answer_materialization(
    *,
    question: str,
    answer: str,
    raw_response: str,
    qa_docs: Sequence[str],
    answer_style: Optional[str] = None,
    logical_plan: Optional[Dict[str, Any]] = None,
    terminal_candidate_rows: Optional[Sequence[Dict[str, Any]]] = None,
    terminal_tuple_completion_rows: Optional[Sequence[Dict[str, Any]]] = None,
    schema_terminal_rows: Optional[Sequence[Dict[str, Any]]] = None,
    schema_terminal_min_score: float = 0.80,
    require_compiled_terminal_plan: bool = False,
    enable_deterministic_terminal_projection: bool = False,
    deterministic_terminal_min_net_utility: float = 0.70,
    enable_terminal_tuple_completion: bool = False,
    terminal_tuple_completion_min_net_utility: float = 0.42,
    enabled: bool = False,
    min_trigger_risk: float = 0.34,
    max_docs: int = 6,
    max_doc_chars: int = 1050,
    max_total_chars: int = 6200,
) -> Tuple[Optional[List[Dict[str, str]]], Dict[str, Any]]:
    before = str(answer or "").strip()
    meta = _new_meta(enabled, before)
    if not enabled:
        return None, meta
    if not before:
        meta["reason"] = "empty_incumbent"
        return None, meta

    contract = compile_answer_domain(
        question,
        answer_style,
        logical_plan=logical_plan,
        require_compiled_terminal_plan=require_compiled_terminal_plan,
    )
    meta["contract"] = contract
    if not contract.get("active"):
        meta["reason"] = str(
            contract.get("contract_reason") or "boolean_or_inactive_contract"
        )
        return None, meta
    if answer_style and answer_style not in {"short_factoid", "short", "factoid"}:
        meta["reason"] = "non_factoid_answer_style"
        return None, meta

    terminal_docs, terminal_candidates = _terminal_provenance_docs(
        terminal_candidate_rows,
        contract,
    )
    schema_terminal_docs, schema_terminal_candidates, schema_terminal_trace = (
        _schema_terminal_provenance_docs(
            schema_terminal_rows,
            question=question,
            contract=contract,
            min_score=float(schema_terminal_min_score),
        )
    )
    completion_docs, completion_candidates = _terminal_completion_provenance_docs(
        terminal_tuple_completion_rows,
    )
    if (
        str(contract.get("terminal_operator_type") or "")
        in {"argmin", "argmax", "compare_eq", "count", "boolean_exists"}
        and not terminal_docs
    ):
        meta.update({
            "eligible": False,
            "logical_plan_bound": True,
            "terminal_operator_id": contract.get("terminal_operator_id", ""),
            "terminal_operator_type": contract.get(
                "terminal_operator_type", ""
            ),
            "terminal_provenance_row_count": 0,
            "reason": "terminal_reduction_not_executed",
        })
        return None, meta
    compact_docs, rendered_docs = _compact_docs(
        schema_terminal_docs + terminal_docs + completion_docs + list(qa_docs or []),
        max_docs=max_docs,
        max_doc_chars=max_doc_chars,
        max_total_chars=max_total_chars,
    )
    if not compact_docs:
        meta["reason"] = "no_source_docs"
        return None, meta

    target_type = str(contract.get("target_type") or "factoid")
    answer_slot = str(contract.get("answer_slot") or "named_entity")
    incumbent_supported, incumbent_support = _is_supported(
        question,
        before,
        compact_docs,
        target_type,
        len(compact_docs),
    )
    incumbent_binding = _binding_score(question, before, compact_docs, contract)
    incumbent_slot_valid, incumbent_slot_reason = _candidate_passes_answer_slot(
        before,
        answer_slot,
        target_type,
        question=question,
        evidence_text="\n".join(compact_docs),
    )
    verbose = bool(VERBOSE_ANSWER_RE.search(before) or _answer_token_count(before) > 7)
    alternative_count = _candidate_surface_count(raw_response)
    schema_candidate_keys = {
        normalize_answer(value)
        for value in schema_terminal_candidates
        if normalize_answer(value)
    }
    unique_schema_candidate = (
        next(iter(schema_candidate_keys)) if len(schema_candidate_keys) == 1 else ""
    )
    unique_schema_terminal_conflict = bool(
        unique_schema_candidate
        and unique_schema_candidate != normalize_answer(before)
    )
    generic_slot = answer_slot in {"named_entity", "person", "location"}
    risk = 0.08
    risk += 0.36 if incumbent_binding < 0.50 else 0.0
    risk += 0.28 if not incumbent_slot_valid else 0.0
    risk += 0.18 if not incumbent_supported else 0.0
    risk += 0.16 if verbose else 0.0
    risk += 0.10 if generic_slot else 0.0
    risk += 0.10 if alternative_count >= 3 else 0.0
    risk += 0.28 if unique_schema_terminal_conflict else 0.0
    risk = min(1.0, risk)
    meta.update({
        "eligible": True,
        "logical_plan_bound": contract.get("contract_source") == "typed_logical_plan",
        "terminal_operator_id": contract.get("terminal_operator_id", ""),
        "terminal_input_slot": contract.get("terminal_input_slot", ""),
        "terminal_output_slot": contract.get("terminal_output_slot", ""),
        "terminal_provenance_row_count": len(terminal_docs),
        "terminal_candidate_surfaces": terminal_candidates,
        "schema_terminal_candidate_surfaces": schema_terminal_candidates,
        "schema_terminal_trace": schema_terminal_trace,
        "unique_schema_terminal_conflict": unique_schema_terminal_conflict,
        "terminal_tuple_completion_row_count": len(
            list(terminal_tuple_completion_rows or [])
        ),
        "terminal_tuple_completion_surfaces": completion_candidates,
        "source_doc_count": len(compact_docs),
        "incumbent_supported": bool(incumbent_supported),
        "incumbent_support_score": round(float(incumbent_support or 0.0), 4),
        "incumbent_binding_score": round(incumbent_binding, 4),
        "incumbent_slot_valid": bool(incumbent_slot_valid),
        "incumbent_slot_reason": incumbent_slot_reason,
        "incumbent_verbose": verbose,
        "raw_candidate_surface_count": alternative_count,
        "trigger_risk": round(risk, 4),
    })
    if enable_deterministic_terminal_projection:
        schema_projection = _deterministic_schema_terminal_projection_gate(
            question=question,
            before=before,
            contract=contract,
            rows=schema_terminal_rows,
            candidates=schema_terminal_candidates,
            docs=schema_terminal_docs,
            trace=schema_terminal_trace,
            incumbent_supported=bool(incumbent_supported),
            incumbent_slot_valid=bool(incumbent_slot_valid),
        )
        meta["deterministic_schema_terminal_projection"] = schema_projection
        if schema_projection.get("accepted"):
            meta["deterministic_terminal_projection"] = schema_projection
            meta.update({
                "triggered": True,
                "action": "deterministic_schema_materialize",
                "reason": "deterministic_schema_terminal_projection_ready",
                "candidate": str(schema_projection.get("candidate") or ""),
                "evidence_ids": list(schema_projection.get("evidence_ids") or []),
            })
            return None, meta
        deterministic_projection = _deterministic_terminal_projection_gate(
            question=question,
            before=before,
            contract=contract,
            rows=terminal_candidate_rows,
            incumbent_slot_valid=bool(incumbent_slot_valid),
            terminal_docs=terminal_docs,
            min_net_utility=float(deterministic_terminal_min_net_utility),
        )
        meta["deterministic_terminal_projection"] = deterministic_projection
        if deterministic_projection.get("accepted"):
            meta.update({
                "triggered": True,
                "action": "deterministic_materialize",
                "reason": "deterministic_terminal_projection_ready",
                "candidate": str(deterministic_projection.get("candidate") or ""),
                "evidence_ids": list(
                    deterministic_projection.get("evidence_ids") or []
                ),
            })
            return None, meta
    if enable_terminal_tuple_completion:
        validated_terminal_row_count = sum(
            bool(row.get("production_eligible"))
            and bool(row.get("operator_complete"))
            and bool(row.get("slot_dependency_validated"))
            and bool(row.get("rooted_lineage_complete"))
            and bool(row.get("final_predicate_match"))
            and bool(row.get("final_answer_type_match"))
            for row in (terminal_candidate_rows or [])
            if isinstance(row, dict)
        )
        terminal_tuple_completion = _provenance_terminal_tuple_completion_gate(
            question=question,
            before=before,
            contract=contract,
            rows=terminal_tuple_completion_rows,
            validated_terminal_row_count=validated_terminal_row_count,
            incumbent_supported=bool(incumbent_supported),
            incumbent_binding=float(incumbent_binding),
            terminal_docs=terminal_docs + completion_docs,
            min_net_utility=float(terminal_tuple_completion_min_net_utility),
        )
        meta["terminal_tuple_completion"] = terminal_tuple_completion
        if terminal_tuple_completion.get("accepted"):


            meta["deterministic_terminal_projection"] = dict(
                terminal_tuple_completion
            )
            meta.update({
                "triggered": True,
                "action": "terminal_tuple_completion",
                "reason": "provenance_terminal_tuple_completion_ready",
                "candidate": str(
                    terminal_tuple_completion.get("candidate") or ""
                ),
                "evidence_ids": list(
                    terminal_tuple_completion.get("evidence_ids") or []
                ),
            })
            return None, meta
    if risk < float(min_trigger_risk):
        meta["reason"] = "risk_below_trigger"
        return None, meta

    prior_trace = re.sub(r"\s+", " ", str(raw_response or "")).strip()[:1800]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a provenance-bound terminal projection operator. Execute only the final $answer slot. "
                "Use the prior reader trace only as a hint and use only the numbered evidence as authority. "
                "Return exactly one JSON object with keys answer, evidence_ids, relation, subject, confidence. "
                "The answer must be the shortest complete surface that directly fills the requested slot, copied "
                "verbatim from cited evidence. Preserve an official full name when shortening would change identity. "
                "evidence_ids must be quoted strings such as [\"E1\"]. If no cited row establishes the requested "
                "relation, return an empty answer and empty evidence_ids. Never return an entity bound to an "
                "intermediate slot such as $v1."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Current answer: {before}\n"
                f"Required output type: {target_type}\n"
                f"Required output slot: {answer_slot}\n"
                f"Required relation family: {contract.get('relation_family')}\n"
                f"Logical plan kind: {contract.get('plan_kind', '')}\n"
                f"Terminal operator: {contract.get('terminal_operator_id', '')} "
                f"{contract.get('terminal_input_slot', '')} -> "
                f"{contract.get('terminal_output_slot', '')}\n"
                f"Terminal predicate families: "
                f"{contract.get('terminal_predicate_families', [])}\n"
                f"Schema terminal candidate domain: {schema_terminal_candidates}\n"
                f"Forbidden intermediate output slots: "
                f"{contract.get('intermediate_output_slots', [])}\n"
                f"Prior reader trace: {prior_trace}\n\n"
                "Evidence rows:\n" + "\n".join(rendered_docs)
            ),
        },
    ]
    meta.update({
        "triggered": True,
        "action": "materialize",
        "reason": "typed_terminal_projection_materialization",
        "compact_docs": compact_docs,
    })
    return messages, meta


def finalize_deterministic_terminal_projection(
    *,
    answer: str,
    prepared_meta: Dict[str, Any],
    apply_rewrite: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Commit or shadow a tuple already admitted by the strict fast path."""
    before = str(answer or "").strip()
    meta = dict(prepared_meta or {})
    decision = dict(meta.get("deterministic_terminal_projection") or {})
    candidate = str(decision.get("candidate") or "").strip()
    meta.update({
        "before": before,
        "after": before,
        "candidate": candidate,
        "candidate_confidence": round(
            float(decision.get("table_score", 0.0) or 0.0), 4
        ),
        "evidence_ids": list(decision.get("evidence_ids") or []),
        "materialization_seconds": 0.0,
    })
    if not decision.get("accepted") or not candidate:
        meta.update({
            "changed": False,
            "action": "keep",
            "reason": str(decision.get("reason") or "deterministic_projection_rejected"),
        })
        return before, meta

    decision_source = str(decision.get("decision_source") or "")
    meta.update({
        "reason": (
            "provenance_bound_terminal_tuple_completion"
            if decision_source == "provenance_terminal_tuple_completion_v12_3"
            else "deterministic_provenance_terminal_tuple"
        ),
        "shadow_changed": True,
        "shadow_after": candidate,
        "action": "replace" if apply_rewrite else "shadow_replace",
    })
    if not apply_rewrite:
        return before, meta
    meta["changed"] = True
    meta["after"] = candidate
    return candidate, meta


def finalize_provenance_bound_answer_materialization(
    *,
    question: str,
    answer: str,
    prepared_meta: Dict[str, Any],
    llm_result: Any,
    apply_rewrite: bool = False,
    min_candidate_confidence: float = 0.72,
    min_binding_score: float = 0.50,
    min_binding_gain: float = 0.08,
    schema_conflict_min_confidence: float = 0.85,
    max_answer_tokens: int = 14,
) -> Tuple[str, Dict[str, Any]]:
    before = str(answer or "").strip()
    meta = dict(prepared_meta or {})
    meta["before"] = before
    meta["after"] = before
    if not meta.get("triggered"):
        return before, meta

    started = time.perf_counter()
    try:
        response, llm_metadata = unpack_llm_result(llm_result)
    except Exception as exc:
        meta.update({
            "reason": "materialization_result_invalid",
            "error_type": exc.__class__.__name__,
            "error_preview": str(exc)[:300],
        })
        return before, meta

    parsed, parse_mode = _extract_json_object(response)
    candidate = str(parsed.get("answer", "") or "").strip(" \t\r\n.;")
    compact_docs = list(meta.pop("compact_docs", []) or [])
    cited_ids = _evidence_ids(parsed.get("evidence_ids"), len(compact_docs))
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    cited_docs = [compact_docs[index - 1] for index in cited_ids]
    cited_text = "\n".join(cited_docs)
    contract = dict(meta.get("contract") or {})
    target_type = str(contract.get("target_type") or "factoid")
    answer_slot = str(contract.get("answer_slot") or "named_entity")
    meta.update({
        "llm_metadata": dict(llm_metadata or {}),
        "raw_response_preview": str(response or "")[:600],
        "response_parse_mode": parse_mode,
        "candidate": candidate,
        "candidate_confidence": round(confidence, 4),
        "evidence_ids": cited_ids,
        "reported_relation": str(parsed.get("relation", "") or "")[:180],
        "reported_subject": str(parsed.get("subject", "") or "")[:180],
        "materialization_seconds": round(time.perf_counter() - started, 6),
    })
    if not parsed:
        meta["reason"] = "invalid_json_response"
        return before, meta
    if not candidate or not cited_ids:
        meta["reason"] = "empty_candidate_or_provenance"
        return before, meta
    if normalize_answer(candidate) == normalize_answer(before):
        meta["reason"] = "same_answer"
        return before, meta
    if confidence < float(min_candidate_confidence):
        meta["reason"] = "candidate_confidence_below_threshold"
        return before, meta
    if _answer_token_count(candidate) > int(max_answer_tokens):
        meta["reason"] = "candidate_too_long"
        return before, meta
    if not _surface_in_text(candidate, cited_text, target_type):
        meta["reason"] = "candidate_not_in_cited_evidence"
        return before, meta
    terminal_candidates = list(meta.get("terminal_candidate_surfaces") or [])
    terminal_candidates.extend(meta.get("schema_terminal_candidate_surfaces") or [])
    if terminal_candidates:
        normalized_candidate = normalize_answer(candidate)
        terminal_match = any(
            normalized_candidate == normalize_answer(value)
            or normalized_candidate in normalize_answer(value)
            or normalize_answer(value) in normalized_candidate
            for value in terminal_candidates
            if normalize_answer(value)
        )
        meta["terminal_tuple_value_match"] = bool(terminal_match)
        if not terminal_match:
            meta["reason"] = "candidate_not_bound_to_terminal_tuple"
            return before, meta

    slot_ok, slot_reason = _candidate_passes_answer_slot(
        candidate,
        answer_slot,
        target_type,
        question=question,
        evidence_text=cited_text,
    )
    meta["slot_valid"] = bool(slot_ok)
    meta["slot_reject_reason"] = slot_reason
    if not slot_ok:
        meta["reason"] = "answer_slot_rejected"
        return before, meta

    candidate_binding = _binding_score(question, candidate, cited_docs, contract)
    incumbent_binding = float(meta.get("incumbent_binding_score", 0.0) or 0.0)
    binding_gain = candidate_binding - incumbent_binding
    candidate_supported, candidate_support = _is_supported(
        question,
        candidate,
        cited_docs,
        target_type,
        len(cited_docs),
    )
    equivalent_supported_incumbent = bool(
        meta.get("incumbent_supported")
        and meta.get("incumbent_slot_valid", True)
        and target_type == "date"
        and _same_date_value(before, candidate)
    )
    containment_supported_incumbent = bool(
        meta.get("incumbent_supported")
        and meta.get("incumbent_slot_valid", True)
        and answer_slot in {
            "city", "county", "region", "location", "full_date", "date", "year",
            "person", "organization", "institution", "work", "named_entity",
        }
        and _is_containment_variant(before, candidate)
        and normalize_answer(before) != normalize_answer(candidate)
        and not meta.get("incumbent_verbose")
    )
    if equivalent_supported_incumbent or containment_supported_incumbent:
        meta.update({
            "candidate_supported": bool(candidate_supported),
            "candidate_support_score": round(float(candidate_support or 0.0), 4),
            "equivalent_supported_incumbent": equivalent_supported_incumbent,
            "containment_supported_incumbent": containment_supported_incumbent,
            "reason": (
                "equivalent_supported_incumbent_protected"
                if equivalent_supported_incumbent
                else "containment_supported_incumbent_protected"
            ),
        })
        return before, meta
    containment_repair = bool(
        normalize_answer(candidate) in normalize_answer(before)
        and normalize_answer(candidate) != normalize_answer(before)
        and meta.get("incumbent_verbose")
    )
    schema_conflict_override = bool(
        meta.get("unique_schema_terminal_conflict")
        and confidence >= float(schema_conflict_min_confidence)
        and terminal_candidates
        and meta.get("terminal_tuple_value_match")
    )
    dominance = bool(
        candidate_binding >= float(min_binding_score)
        and (
            binding_gain >= float(min_binding_gain)
            or containment_repair
            or not bool(meta.get("incumbent_supported"))
            or not bool(meta.get("incumbent_slot_valid", True))
            or schema_conflict_override
        )
    )
    meta.update({
        "candidate_supported": bool(candidate_supported),
        "candidate_support_score": round(float(candidate_support or 0.0), 4),
        "candidate_binding_score": round(candidate_binding, 4),
        "binding_gain": round(binding_gain, 4),
        "containment_repair": containment_repair,
        "schema_conflict_override": schema_conflict_override,
        "dominance_gate": dominance,
    })
    if not candidate_supported:
        meta["reason"] = "candidate_not_supported_by_citation"
        return before, meta
    if not dominance:
        meta["reason"] = "candidate_does_not_dominate_incumbent"
        return before, meta

    meta.update({
        "action": "replace" if apply_rewrite else "shadow_replace",
        "reason": "provenance_bound_domain_candidate",
        "shadow_changed": True,
        "shadow_after": candidate,
    })
    if not apply_rewrite:
        return before, meta
    meta["changed"] = True
    meta["after"] = candidate
    return candidate, meta
