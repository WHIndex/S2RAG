import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .candidate_answer_selector_utils import _is_synthetic_doc
from .sentence_evidence_utils import infer_answer_target_type
from .planned_retrieval_state_utils import planned_retrieval_terminal_state


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
DATE_VALUE_RE = re.compile(
    r"\b(?:\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2})\b",
    re.IGNORECASE,
)
NUMBER_VALUE_RE = re.compile(r"\b\d+(?:[,.]\d+)*(?:\.\d+)?(?:\s*(?:%|percent|million|billion))?\b")
NATIONALITY_OR_COUNTRY_RE = re.compile(
    r"\b(?:afghan|albanian|algerian|american|argentine|armenian|australian|"
    r"austrian|belgian|brazilian|british|bulgarian|canadian|chilean|chinese|"
    r"colombian|croatian|cuban|czech|danish|dutch|egyptian|english|finnish|"
    r"french|german|ghanaian|greek|hungarian|indian|indonesian|iranian|"
    r"irish|israeli|italian|jamaican|japanese|korean|malaysian|mexican|"
    r"moroccan|nigerian|norwegian|pakistani|polish|portuguese|romanian|"
    r"russian|scottish|serbian|singaporean|spanish|swedish|swiss|syrian|"
    r"taiwanese|thai|turkish|ukrainian|vietnamese|welsh|"
    r"afghanistan|albania|algeria|argentina|armenia|australia|austria|"
    r"belgium|brazil|britain|bulgaria|canada|chile|china|colombia|croatia|"
    r"cuba|denmark|egypt|england|finland|france|germany|ghana|greece|"
    r"hungary|india|indonesia|iran|ireland|israel|italy|jamaica|japan|"
    r"korea|malaysia|mexico|morocco|nigeria|norway|pakistan|poland|"
    r"portugal|romania|russia|scotland|serbia|singapore|spain|sweden|"
    r"switzerland|syria|taiwan|thailand|turkey|ukraine|vietnam|wales)\b",
    re.IGNORECASE,
)
OCCUPATION_RE = re.compile(
    r"\b(?:actor|actress|singer|writer|author|director|producer|composer|poet|"
    r"politician|scientist|engineer|artist|painter|footballer|player|coach|"
    r"journalist|historian|professor|mathematician|physician|lawyer|designer|"
    r"architect|musician|novelist|screenwriter|businessperson|occupation)\b",
    re.IGNORECASE,
)
BRIDGE_CUE_RE = re.compile(
    r"\b(?:of|by|from|whose|that|which)\b.{0,140}\b(?:director|author|writer|"
    r"composer|producer|creator|founder|designer|parent|father|mother|spouse|"
    r"birthplace|nationality|citizenship|country|capital)\b|"
    r"\b(?:director|author|writer|composer|producer|creator|founder|designer|"
    r"parent|father|mother|spouse)\s+of\b",
    re.IGNORECASE,
)
SYNTHETIC_TITLE_PREFIX = "slot-separated fact join view"
BOOLEAN_QUESTION_RE = re.compile(
    r"^\s*(?:are|is|was|were|do|does|did|can|could|would|will|has|have|had)\b"
    r".*\?",
    re.IGNORECASE,
)
COMPARISON_QUESTION_RE = re.compile(
    r"\b(?:both|same|different|earlier|later|older|younger|more|less|larger|"
    r"smaller|higher|lower|first|last)\b|"
    r"\b(?:which|who|what)\b.{0,120}\b(?:or|than)\b",
    re.IGNORECASE,
)
PREDICATE_SPECS = (
    {
        "slot": "in_law_relation",
        "question": re.compile(r"\b(?:mother|father|parent|brother|sister|son|daughter)-?in-?law\b", re.IGNORECASE),
        "block": True,
    },
    {
        "slot": "death_date",
        "question": re.compile(r"\b(?:date of death|death date|when did .{0,80} die|when was .{0,80} death|died)\b", re.IGNORECASE),
        "relation": re.compile(r"\b(?:died|die|death|date of death|deceased)\b", re.IGNORECASE),
        "answer_side": "object",
    },
    {
        "slot": "birth_date",
        "question": re.compile(r"\b(?:date of birth|birth date|when was .{0,80} born|born)\b", re.IGNORECASE),
        "relation": re.compile(r"\b(?:born|birth|date of birth)\b", re.IGNORECASE),
        "answer_side": "object",
    },
    {
        "slot": "foundation_date",
        "question": re.compile(r"\b(?:founded|foundation|established|formed|incorporated|launched)\b", re.IGNORECASE),
        "relation": re.compile(r"\b(?:founded|foundation|established|formed|incorporated|launched)\b", re.IGNORECASE),
        "answer_side": "object",
    },
    {
        "slot": "publication_date",
        "question": re.compile(r"\b(?:published|released|aired|premiered|opened)\b", re.IGNORECASE),
        "relation": re.compile(r"\b(?:published|released|aired|premiered|opened)\b", re.IGNORECASE),
        "answer_side": "object",
    },
    {
        "slot": "adapted_work",
        "question": re.compile(r"\b(?:adaptation|adaption|adapted)\b", re.IGNORECASE),
        "relation": re.compile(r"\b(?:adaptation|adaption|adapted|based on)\b", re.IGNORECASE),
        "answer_side": "subject",
    },
    {
        "slot": "director_person",
        "question": re.compile(r"\b(?:director|directed)\b", re.IGNORECASE),
        "relation": re.compile(r"\b(?:director|directed)\b", re.IGNORECASE),
        "answer_side": "actor",
    },
    {
        "slot": "author_person",
        "question": re.compile(r"\b(?:author|writer|written by|wrote)\b", re.IGNORECASE),
        "relation": re.compile(r"\b(?:author|writer|written|wrote)\b", re.IGNORECASE),
        "answer_side": "actor",
    },
    {
        "slot": "creator_person",
        "question": re.compile(r"\b(?:creator|created by|composer|composed by|producer|produced by|founder)\b", re.IGNORECASE),
        "relation": re.compile(r"\b(?:creator|created|composer|composed|producer|produced|founder|founded by)\b", re.IGNORECASE),
        "answer_side": "actor",
    },
    {
        "slot": "kinship_person",
        "question": re.compile(r"\b(?:mother|father|parent|grandfather|grandmother|spouse|wife|husband|child|son|daughter)\b", re.IGNORECASE),
        "relation": re.compile(r"\b(?:mother|father|parent|grandfather|grandmother|spouse|wife|husband|child|son|daughter|married)\b", re.IGNORECASE),
        "answer_side": "kinship",
    },
    {
        "slot": "nationality",
        "question": re.compile(r"\b(?:nationality|citizenship|citizen|national)\b", re.IGNORECASE),
        "relation": re.compile(r"\b(?:nationality|citizenship|citizen|national|country)\b", re.IGNORECASE),
        "answer_side": "object",
    },
    {
        "slot": "occupation",
        "question": re.compile(r"\b(?:occupation|profession|job|known for)\b", re.IGNORECASE),
        "relation": re.compile(r"\b(?:occupation|profession|job|known for|worked as)\b", re.IGNORECASE),
        "answer_side": "object",
    },
    {
        "slot": "religion",
        "question": re.compile(r"\b(?:religion|auspices|denomination)\b", re.IGNORECASE),
        "relation": re.compile(r"\b(?:religion|auspices|denomination|affiliated|affiliation)\b", re.IGNORECASE),
        "answer_side": "object",
    },
    {
        "slot": "county_seat",
        "question": re.compile(r"\b(?:seat of|county seat|seat)\b", re.IGNORECASE),
        "relation": re.compile(r"\b(?:seat|county seat)\b", re.IGNORECASE),
        "answer_side": "object",
    },
    {
        "slot": "capital_location",
        "question": re.compile(r"\bcapital\b", re.IGNORECASE),
        "relation": re.compile(r"\bcapital\b", re.IGNORECASE),
        "answer_side": "object",
    },
    {
        "slot": "located_place",
        "question": re.compile(r"\b(?:where|located|city|country|state|county|castle)\b", re.IGNORECASE),
        "relation": re.compile(r"\b(?:located|location|city|country|state|county|capital|in)\b", re.IGNORECASE),
        "answer_side": "object",
    },
    {
        "slot": "commentary_affiliation",
        "question": re.compile(r"\b(?:co-?commentator|commentator)\b", re.IGNORECASE),
        "relation": re.compile(r"\b(?:co-?commentator|commentator|commentary)\b", re.IGNORECASE),
        "answer_side": "object",
    },
)
STRICT_PREDICATE_FINAL_TYPES = {"date", "location", "nationality", "occupation"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, _as_float(value, low)))


def _doc_title(doc: Any) -> str:
    return str(doc or "").split("\n", 1)[0].strip()


def _source_docs(docs: List[str]) -> List[str]:
    return [str(doc or "") for doc in docs or [] if not _is_synthetic_doc(str(doc or ""))]


def _avg_source_doc_chars(docs: List[str]) -> float:
    source_docs = _source_docs(docs)
    if not source_docs:
        return 0.0
    return sum(len(doc) for doc in source_docs) / max(1, len(source_docs))


def _tokens(text: Any) -> List[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def _candidate_key(text: Any) -> str:
    return " ".join(_tokens(text))


def _triple_parts(triple_text: Any) -> Tuple[str, str, str]:
    parts = [part.strip() for part in str(triple_text or "").split("|")]
    if len(parts) < 3:
        return "", "", ""
    return parts[0], parts[1], " | ".join(parts[2:]).strip()


def _infer_predicate_spec(question: str) -> Dict[str, Any]:
    question = str(question or "")
    for spec in PREDICATE_SPECS:
        if spec.get("question") and spec["question"].search(question):
            return dict(spec)
    return {"slot": "generic", "answer_side": "any"}


def _question_has_boolean_or_comparison_shape(question: str) -> bool:
    question = str(question or "")
    return bool(BOOLEAN_QUESTION_RE.search(question) or COMPARISON_QUESTION_RE.search(question))


def _key_overlaps(left: Any, right: Any) -> bool:
    left_key = _candidate_key(left)
    right_key = _candidate_key(right)
    if not left_key or not right_key:
        return False
    return left_key == right_key or left_key in right_key or right_key in left_key


def _candidate_side(candidate: str, triple_text: Any) -> str:
    subject, _, obj = _triple_parts(triple_text)
    subject_hit = _key_overlaps(candidate, subject)
    object_hit = _key_overlaps(candidate, obj)
    if subject_hit and object_hit:
        return "both"
    if subject_hit:
        return "subject"
    if object_hit:
        return "object"
    return "unknown"


def _side_compatible(answer_side: str, candidate_side: str, relation_text: str) -> bool:
    answer_side = str(answer_side or "any").lower()
    candidate_side = str(candidate_side or "unknown").lower()
    relation_text = str(relation_text or "").lower()
    if answer_side == "any" or candidate_side == "both":
        return True
    if candidate_side == "unknown":
        return False
    if answer_side == candidate_side:
        return True
    if answer_side == "actor":
        if " by" in relation_text or relation_text.endswith("by"):
            return candidate_side == "object"
        return candidate_side in {"subject", "object"}
    if answer_side == "kinship":
        return candidate_side in {"subject", "object"}
    return False


def _predicate_compatible(
    *,
    final_slot_type: str,
    candidate: str,
    triple_path: List[str],
    predicate_spec: Dict[str, Any],
    require_predicate_match: bool,
) -> Tuple[bool, str]:
    if not require_predicate_match:
        return True, "predicate_gate_disabled"
    if predicate_spec.get("block"):
        return False, f"blocked_predicate_slot:{predicate_spec.get('slot', 'unknown')}"
    slot = str(predicate_spec.get("slot") or "generic")
    if slot == "generic" and str(final_slot_type or "") not in STRICT_PREDICATE_FINAL_TYPES:
        return True, "generic_slot_allowed"
    if not triple_path:
        return False, "missing_triple_path"
    final_triple = triple_path[-1]
    _, relation_text, _ = _triple_parts(final_triple)
    relation_re = predicate_spec.get("relation")
    if relation_re is None:
        return False, "missing_predicate_constraint"
    if not relation_re.search(relation_text):
        return False, f"predicate_relation_mismatch:{slot}"
    side = _candidate_side(candidate, final_triple)
    if not _side_compatible(str(predicate_spec.get("answer_side") or "any"), side, relation_text):
        return False, f"predicate_answer_side_mismatch:{slot}"
    return True, f"predicate_match:{slot}"


def _infer_final_slot_type(question: str) -> str:
    q = re.sub(r"\s+", " ", str(question or "").lower()).strip()
    if re.search(r"\b(?:nationality|citizenship|citizen|national)\b", q):
        return "nationality"
    return infer_answer_target_type(question)


def _question_state(question: str, planned_retrieval: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    planned_retrieval = planned_retrieval or {}
    router = planned_retrieval.get("router") or {}
    verifier = planned_retrieval.get("verifier") or planned_retrieval.get("verification") or {}
    if not isinstance(router, dict):
        router = {}
    if not isinstance(verifier, dict):
        verifier = {}
    reason = str(
        planned_retrieval.get("reason")
        or planned_retrieval.get("stop_reason")
        or verifier.get("reason")
        or ""
    )
    question_type = str(
        planned_retrieval.get("question_type")
        or planned_retrieval.get("routing_profile")
        or router.get("question_type")
        or router.get("profile")
        or ""
    )
    bridge_strength = max(
        _as_float(planned_retrieval.get("bridge_strength"), 0.0),
        _as_float(router.get("bridge_strength"), 0.0),
    )
    bridge_cues = (
        bool(BRIDGE_CUE_RE.search(question or ""))
        or "bridge" in question_type.lower()
        or "multi" in question_type.lower()
        or bridge_strength >= 2.0
    )
    terminal_state = planned_retrieval_terminal_state(planned_retrieval)
    hard_deficit = bool(terminal_state["terminal_evidence_gap"])
    return {
        "question_type": question_type,
        "reason": reason,
        "bridge_strength": bridge_strength,
        "bridge_cues": bridge_cues,
        "hard_deficit": hard_deficit,
        "expansion_stopped_by_optimizer": terminal_state[
            "expansion_stopped_by_optimizer"
        ],
    }


def _value_type(value: str, plan_type: str) -> str:
    value = str(value or "").strip()
    plan_type = str(plan_type or "").lower()
    if not value:
        return "unknown"
    if DATE_VALUE_RE.search(value):
        return "date"
    if NUMBER_VALUE_RE.fullmatch(value):
        return "number"
    if NATIONALITY_OR_COUNTRY_RE.search(value) or "national" in plan_type or "citizen" in plan_type:
        return "nationality"
    if OCCUPATION_RE.search(value):
        return "occupation"
    return plan_type or "entity"


def _looks_like_person_or_entity(value: str) -> bool:
    value = str(value or "").strip()
    if not value or len(value) <= 2:
        return False
    if re.search(r"\d", value):
        return False
    tokens = _tokens(value)
    if not tokens or len(tokens) > 8:
        return False
    if value.lower() in {"unknown", "none", "person", "people", "director", "author"}:
        return False
    return True


def _is_type_compatible(candidate: str, plan_answer_type: str, final_slot_type: str, triple_path: List[str]) -> bool:
    candidate = str(candidate or "").strip()
    plan_answer_type = str(plan_answer_type or "").lower()
    final_slot_type = str(final_slot_type or "").lower()
    path_text = " ".join(str(item or "") for item in triple_path).lower()
    value_type = _value_type(candidate, plan_answer_type)
    if not candidate:
        return False
    if final_slot_type == "date":
        return value_type == "date"
    if final_slot_type == "number":
        return value_type == "number"
    if final_slot_type == "nationality":
        return value_type == "nationality" or (
            bool(NATIONALITY_OR_COUNTRY_RE.search(candidate))
            and re.search(r"\b(?:nationality|citizenship|citizen|national|country)\b", path_text)
        )
    if final_slot_type == "occupation":
        return value_type == "occupation" or bool(
            re.search(r"\b(?:occupation|profession|known for)\b", path_text)
        )
    if final_slot_type == "location":
        return value_type in {"nationality", "place", "location"} or bool(
            re.search(r"\b(?:city|country|state|place|located|capital)\b", f"{candidate} {path_text}")
        )
    if final_slot_type == "organization":
        return plan_answer_type in {"organization", "entity"} and _looks_like_person_or_entity(candidate)
    if final_slot_type == "person":
        return plan_answer_type in {"person", "entity"} and _looks_like_person_or_entity(candidate)
    if final_slot_type == "work":
        return plan_answer_type in {"work", "entity"} and _looks_like_person_or_entity(candidate)
    return _looks_like_person_or_entity(candidate) or value_type in {"date", "number", "nationality", "occupation"}


def _plan_admitted(
    plan: Dict[str, Any],
    *,
    min_utility: float,
    max_risk: float,
    min_net_utility: float,
) -> Tuple[bool, str]:
    if not bool(plan.get("accepted", False)):
        return False, str(plan.get("reject_reason") or "shadow_plan_rejected")
    if _as_float(plan.get("utility"), 0.0) < float(min_utility):
        return False, "utility_below_slot_threshold"
    if _as_float(plan.get("risk"), 1.0) > float(max_risk):
        return False, "risk_above_slot_threshold"
    if _as_float(plan.get("net_utility"), -1.0) < float(min_net_utility):
        return False, "net_utility_below_slot_threshold"
    if _as_int(plan.get("worst_rank"), 999999) > 120:
        return False, "source_rank_too_deep"
    return True, "accepted"


def _row_score(plan: Dict[str, Any]) -> float:
    utility = _as_float(plan.get("utility"), 0.0)
    net_utility = _as_float(plan.get("net_utility"), 0.0)
    risk = _as_float(plan.get("risk"), 1.0)
    relation = _as_float(plan.get("relation_overlap"), 0.0)
    endpoint = _as_float(plan.get("endpoint_overlap"), 0.0)
    provenance = _as_float(plan.get("provenance_score"), 0.0)
    score = (
        0.34 * utility
        + 0.28 * max(0.0, net_utility)
        + 0.14 * relation
        + 0.10 * endpoint
        + 0.14 * provenance
        - 0.18 * risk
    )
    return round(_clamp(score), 6)


def _format_doc(final_slot_type: str, rows: List[Dict[str, Any]]) -> str:
    predicate_slot = str(rows[0].get("predicate_slot") or "generic") if rows else "generic"
    lines = [
        "Slot-Separated Fact Join View",
        f"Final answer slot: {final_slot_type}",
        f"Predicate slot: {predicate_slot}",
        "The bridge_entity column is intermediate context. Use final_answer_candidate for the final answer.",
        "Rows:",
    ]
    for idx, row in enumerate(rows, start=1):
        path_text = " -> ".join(str(item or "") for item in row.get("triple_path", [])[:2])
        lines.append(
            "- row={idx} bridge_entity=\"{bridge}\" final_answer_candidate=\"{answer}\" "
            "final_answer_type={answer_type} confidence={score:.3f} sources=\"{sources}\" "
            "path=\"{path}\"".format(
                idx=idx,
                bridge=str(row.get("bridge_entity") or "")[:120],
                answer=str(row.get("final_answer_candidate") or "")[:120],
                answer_type=str(row.get("final_answer_type") or final_slot_type),
                score=float(row.get("slot_score", 0.0) or 0.0),
                sources="; ".join(str(title or "")[:80] for title in row.get("titles", [])[:2]),
                path=path_text[:360],
            )
        )
    lines.extend([
        "Reader guidance:",
        "- Resolve the bridge entity first, then read the final answer slot.",
        "- Do not output a bridge_entity unless the question asks directly for that entity.",
    ])
    return "\n".join(lines)


def build_slot_separated_fact_join_view(
    *,
    question: str,
    docs: List[str],
    planned_retrieval: Optional[Dict[str, Any]] = None,
    typed_fact_join_shadow: Optional[Dict[str, Any]] = None,
    enabled: bool = False,
    max_rows: int = 3,
    min_utility: float = 0.54,
    max_risk: float = 0.50,
    min_net_utility: float = 0.12,
    insert_after: int = 2,
    require_answer_slot: bool = True,
    bridge_or_deficit_only: bool = True,
    block_long_context: bool = True,
    long_context_avg_chars: int = 1400,
    allow_one_hop_answer_rows: bool = False,
    require_predicate_match: bool = True,
    block_boolean_comparison: bool = True,
    min_slot_score: float = 0.50,
) -> Tuple[List[str], Dict[str, Any]]:
    docs = docs or []
    state = _question_state(question, planned_retrieval)
    final_slot_type = _infer_final_slot_type(question)
    predicate_spec = _infer_predicate_spec(question)
    boolean_or_comparison = _question_has_boolean_or_comparison_shape(question)
    avg_chars = _avg_source_doc_chars(docs)
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled" if not enabled else "not_evaluated",
        "final_slot_type": final_slot_type,
        "predicate_slot": str(predicate_spec.get("slot") or "generic"),
        "predicate_gate_enabled": bool(require_predicate_match),
        "boolean_or_comparison": bool(boolean_or_comparison),
        "block_boolean_comparison": bool(block_boolean_comparison),
        "min_slot_score": float(min_slot_score),
        "question_type": state["question_type"],
        "bridge_strength": state["bridge_strength"],
        "bridge_cues": state["bridge_cues"],
        "hard_deficit": state["hard_deficit"],
        "avg_source_doc_chars": round(avg_chars, 2),
        "raw_plan_count": 0,
        "accepted_plan_count": 0,
        "candidate_row_count": 0,
        "row_count": 0,
        "reject_reason_counts": {},
        "rows": [],
        "original_doc_count": len(docs),
        "final_doc_count": len(docs),
    }
    if not enabled:
        return docs, info
    if not docs:
        info["reason"] = "no_docs"
        return docs, info
    if any(_doc_title(doc).lower().startswith(SYNTHETIC_TITLE_PREFIX) for doc in docs):
        info["reason"] = "already_inserted"
        return docs, info
    if block_long_context and avg_chars >= float(long_context_avg_chars) and not state["bridge_cues"]:
        info["reason"] = "blocked_long_context_without_bridge"
        return docs, info
    if block_boolean_comparison and boolean_or_comparison:
        info["reason"] = "blocked_boolean_or_comparison"
        return docs, info
    if bridge_or_deficit_only and not (state["bridge_cues"] or state["hard_deficit"]):
        info["reason"] = "blocked_no_bridge_or_deficit"
        return docs, info

    shadow = typed_fact_join_shadow or {}
    plans = list(shadow.get("top_plans") or [])
    info["raw_plan_count"] = int(shadow.get("raw_plan_count", len(plans)) or len(plans))
    info["accepted_plan_count"] = int(shadow.get("accepted_plan_count", 0) or 0)
    if not plans:
        info["reason"] = "no_shadow_plans"
        return docs, info

    reject_counts = Counter()
    by_answer: Dict[str, Dict[str, Any]] = {}
    for plan in plans:
        if not isinstance(plan, dict):
            reject_counts["malformed_plan"] += 1
            continue
        plan_type = str(plan.get("plan_type") or "")
        is_two_hop = "two_hop" in plan_type
        if not is_two_hop and not allow_one_hop_answer_rows:
            reject_counts["one_hop_blocked"] += 1
            continue
        admitted, reason = _plan_admitted(
            plan,
            min_utility=min_utility,
            max_risk=max_risk,
            min_net_utility=min_net_utility,
        )
        if not admitted:
            reject_counts[reason] += 1
            continue
        candidate = str(plan.get("candidate_answer") or "").strip()
        triple_path = [str(item or "") for item in plan.get("triple_path") or [] if str(item or "").strip()]
        if not _is_type_compatible(candidate, str(plan.get("answer_type") or ""), final_slot_type, triple_path):
            reject_counts["final_slot_type_mismatch"] += 1
            continue
        predicate_ok, predicate_reason = _predicate_compatible(
            final_slot_type=final_slot_type,
            candidate=candidate,
            triple_path=triple_path,
            predicate_spec=predicate_spec,
            require_predicate_match=require_predicate_match,
        )
        if not predicate_ok:
            reject_counts[predicate_reason] += 1
            continue
        slot_score = _row_score(plan)
        if slot_score < float(min_slot_score):
            reject_counts["slot_score_below_threshold"] += 1
            continue
        key = _candidate_key(candidate)
        if not key:
            reject_counts["empty_candidate"] += 1
            continue
        bridge_entity = str(plan.get("bridge_entity") or "").strip()
        if not bridge_entity and is_two_hop and triple_path:
            _, _, bridge_entity = _triple_parts(triple_path[0])
        first = by_answer.get(key)
        row = {
            "bridge_entity": bridge_entity,
            "final_answer_candidate": candidate,
            "final_answer_type": final_slot_type,
            "predicate_slot": str(predicate_spec.get("slot") or "generic"),
            "predicate_match_reason": predicate_reason,
            "plan_answer_type": str(plan.get("answer_type") or ""),
            "plan_type": plan_type,
            "slot_score": slot_score,
            "utility": round(_as_float(plan.get("utility"), 0.0), 4),
            "net_utility": round(_as_float(plan.get("net_utility"), 0.0), 4),
            "risk": round(_as_float(plan.get("risk"), 0.0), 4),
            "best_rank": _as_int(plan.get("best_rank"), 999999),
            "worst_rank": _as_int(plan.get("worst_rank"), 999999),
            "titles": list(plan.get("titles") or [])[:2],
            "triple_path": triple_path[:2],
        }
        if first is None or (
            float(row["slot_score"]),
            -int(row["worst_rank"]),
            float(row["net_utility"]),
        ) > (
            float(first.get("slot_score", 0.0) or 0.0),
            -int(first.get("worst_rank", 999999) or 999999),
            float(first.get("net_utility", 0.0) or 0.0),
        ):
            by_answer[key] = row

    rows = list(by_answer.values())
    rows.sort(
        key=lambda row: (
            -float(row.get("slot_score", 0.0) or 0.0),
            -float(row.get("net_utility", 0.0) or 0.0),
            float(row.get("risk", 1.0) or 1.0),
            int(row.get("worst_rank", 999999) or 999999),
        )
    )
    rows = rows[: max(1, int(max_rows))]
    info["candidate_row_count"] = len(by_answer)
    info["row_count"] = len(rows)
    info["reject_reason_counts"] = dict(reject_counts)
    info["rows"] = rows
    if require_answer_slot and not rows:
        info["reason"] = "no_compatible_answer_slot"
        return docs, info
    if not rows:
        info["reason"] = "no_rows"
        return docs, info

    slot_doc = _format_doc(final_slot_type, rows)
    insert_at = min(max(0, int(insert_after)), len(docs))
    final_docs = docs[:insert_at] + [slot_doc] + docs[insert_at:]
    info.update({
        "applied": True,
        "reason": "slot_separated_fact_join_inserted",
        "insert_at": insert_at,
        "final_doc_count": len(final_docs),
        "selected_candidate": rows[0].get("final_answer_candidate"),
        "selected_score": rows[0].get("slot_score"),
    })
    return final_docs, info
