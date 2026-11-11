import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from .answer_verifier_utils import _rank_candidates
from .candidate_answer_selector_utils import (
    KINSHIP_RE,
    _candidate_contains_question_anchor,
    _candidate_is_disallowed_date_range,
    _candidate_is_year_like_count_answer,
    _candidate_looks_like_secondary_version,
    _candidate_matches_question_unit,
    _candidate_matches_required_relation,
    _candidate_reuses_question_number,
    _has_main_numeric_temporal_request,
    _is_synthetic_doc,
    _required_relation_pattern,
)
from .sentence_evidence_utils import infer_answer_target_type
from .planned_retrieval_state_utils import planned_retrieval_terminal_state


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TITLE_CASE_RE = re.compile(
    r"\b[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:[-'’][A-ZÀ-ÖØ-Þ]?[a-zà-öø-ÿ]+)?"
    r"(?:\s+(?:of|the|and|de|da|do|dos|das|del|la|le|du|van|von|"
    r"[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:[-'’][A-ZÀ-ÖØ-Þ]?[a-zà-öø-ÿ]+)?)){0,5}\b"
)
_QUESTION_STOPWORDS = {
    "about",
    "after",
    "also",
    "answer",
    "before",
    "between",
    "both",
    "current",
    "date",
    "does",
    "from",
    "have",
    "many",
    "much",
    "name",
    "number",
    "only",
    "their",
    "this",
    "what",
    "when",
    "where",
    "which",
    "whose",
    "with",
    "year",
}
_BAD_PERSON_SLOT_KEYS = {
    "american",
    "australian",
    "british",
    "canadian",
    "dutch",
    "english",
    "french",
    "german",
    "irish",
    "italian",
    "japanese",
    "no",
    "none",
    "spanish",
    "swedish",
    "the who",
    "unknown",
}
_BAD_GENERIC_SLOT_KEYS = {
    "answer",
    "candidate",
    "evidence",
    "fact",
    "information",
    "no",
    "none",
    "question",
    "unknown",
}
_CHAIN_KINSHIP_RE = re.compile(
    r"\b(?:father|mother|parent)s?-in-law\b|\b(?:paternal|maternal)\s+grand(?:father|mother|parent)\b",
    re.IGNORECASE,
)
_DIRECT_KINSHIP_RE = re.compile(
    r"\b(?:father|mother|parent|parents|son|daughter|wife|husband|spouse|child|children)\b",
    re.IGNORECASE,
)
_CREATIVE_ACTOR_Q_RE = re.compile(
    r"\b(?:who|whom|whose|which\s+(?:person|people|artist|company|organization|organisation))\b"
    r".{0,120}\b(?:produced?|producer|directed?|director|wrote|written|writer|author|"
    r"composed?|composer|created?|creator|developed?|developer|designed?|designer|"
    r"founded?|founder|performed?|credited)\b|"
    r"\b(?:producer|director|writer|author|composer|creator|developer|designer|founder)s?\s+of\b",
    re.IGNORECASE,
)
_CREATIVE_ACTOR_EVIDENCE_RE = re.compile(
    r"\b(?:produced?|producer|directed?|director|wrote|written|writer|author|"
    r"composed?|composer|created?|creator|developed?|developer|designed?|designer|"
    r"founded?|founder|performed?|credited)\b|\bby\b",
    re.IGNORECASE,
)
_CREATIVE_BY_RE = re.compile(
    r"\b(?:produced|directed|written|composed|created|developed|designed|founded|performed|"
    r"credited)\s+by\s+([^.;:\n]{3,180})",
    re.IGNORECASE,
)
_CREATIVE_FOLLOWUP_RELATION_RE = re.compile(
    r"\s+and\s+(?:produced|directed|written|composed|created|developed|designed|founded|performed|"
    r"credited|starring|released|published)\b",
    re.IGNORECASE,
)
_CAPITAL_Q_RE = re.compile(r"\bcapital\b", re.IGNORECASE)
_CAPITAL_EVIDENCE_RE = re.compile(r"\bcapital\b", re.IGNORECASE)
_LOCATION_Q_RE = re.compile(
    r"^\s*where\b|\b(?:located|location|birthplace|place\s+of\s+birth|which\s+(?:city|country|state|county))\b",
    re.IGNORECASE,
)
_LOCATION_EVIDENCE_RE = re.compile(
    r"\b(?:located|location|in|birthplace|born\s+in|city|country|state|county|province|region)\b",
    re.IGNORECASE,
)
_EVENT_TERMS_RE = re.compile(
    r"\b(?:battle|war|event|match|game|election|tournament|film|movie|song|album|book|"
    r"novel|episode|season|publication|release|opened|founded|established|formed)\b",
    re.IGNORECASE,
)
_BIRTH_DEATH_RE = re.compile(r"\b(?:born|birth|died|death)\b", re.IGNORECASE)
_CONJUNCTION_RE = re.compile(r"\b(?:and|with|,)\b", re.IGNORECASE)


def _normalize_candidate(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"^[\"'`]+|[\"'`.,;:]+$", "", text)


def _token_set(text: Any) -> set:
    return {
        token
        for token in _TOKEN_RE.findall(str(text or "").lower())
        if len(token) > 1 and token not in _QUESTION_STOPWORDS
    }


def _question_content_terms(question: str) -> set:
    return {
        token
        for token in _token_set(question)
        if len(token) >= 4 and not token.isdigit()
    }


def _candidate_reuses_question_surface(candidate_text: str, question: str) -> bool:
    candidate_key = _normalize_candidate(candidate_text)
    if not candidate_key:
        return False
    question_key = _normalize_candidate(question)
    if re.search(rf"(?:^|\s){re.escape(candidate_key)}(?:\s|$)", question_key):
        return True
    candidate_tokens = _token_set(candidate_text)
    if not candidate_tokens:
        return False
    question_tokens = _token_set(question)
    return bool(candidate_tokens) and candidate_tokens <= question_tokens


def _candidate_has_person_shape(candidate_text: str) -> bool:
    key = _normalize_candidate(candidate_text)
    if key in _BAD_PERSON_SLOT_KEYS or key in _BAD_GENERIC_SLOT_KEYS:
        return False
    if len(candidate_text.strip()) <= 2:
        return False
    if re.search(r"\d", candidate_text):
        return False
    title_chunks = _TITLE_CASE_RE.findall(candidate_text)
    if len(title_chunks) >= 1 and len(_token_set(candidate_text)) >= 2:
        return True
    if "'" in candidate_text or "’" in candidate_text or "-" in candidate_text:
        return True
    return False


def _candidate_has_reasonable_shape(candidate_text: str, target_type: str) -> bool:
    key = _normalize_candidate(candidate_text)
    if not key or key in _BAD_GENERIC_SLOT_KEYS:
        return False
    if target_type == "person":
        return _candidate_has_person_shape(candidate_text)
    if target_type in {"location", "organization", "work"}:
        if key in {"current", "capital", "city", "country", "state", "district"}:
            return False
        return len(_token_set(candidate_text)) >= 1
    return True


def _candidate_positions(candidate_text: str, sentence: str) -> List[int]:
    candidate_text = str(candidate_text or "").strip()
    sentence = str(sentence or "")
    if not candidate_text or not sentence:
        return []
    positions = [match.start() for match in re.finditer(re.escape(candidate_text), sentence, re.IGNORECASE)]
    if positions:
        return positions
    positions = []
    sentence_lower = sentence.lower()
    for token in _token_set(candidate_text):
        match = re.search(rf"\b{re.escape(token)}\b", sentence_lower)
        if match:
            positions.append(match.start())
    return positions


def _near_relation(candidate_text: str, sentence: str, relation_pattern, max_distance: int = 150) -> bool:
    if relation_pattern is None:
        return True
    candidate_positions = _candidate_positions(candidate_text, sentence)
    if not candidate_positions:
        return False
    relation_positions = [match.start() for match in relation_pattern.finditer(sentence or "")]
    if not relation_positions:
        return False
    return min(abs(c_pos - r_pos) for c_pos in candidate_positions for r_pos in relation_positions) <= max_distance


def _capital_sentence_fills_candidate_slot(candidate_text: str, sentence: str) -> bool:
    candidate_text = str(candidate_text or "").strip()
    sentence = re.sub(r"\s+", " ", str(sentence or "")).strip()
    if not candidate_text or not sentence or not _CAPITAL_EVIDENCE_RE.search(sentence):
        return False
    candidate_re = re.escape(candidate_text)
    patterns = (
        rf"\b{candidate_re}\b.{{0,90}}\b(?:is|was|became|serves\s+as|served\s+as)\b.{{0,35}}\bcapital\b",
        rf"\bcapital(?:\s+city)?\s+(?:is|was|became)\s+\b{candidate_re}\b",
        rf"\b(?:has|had)\s+\b{candidate_re}\b\s+as\s+(?:its\s+)?capital\b",
    )
    return any(re.search(pattern, sentence, re.IGNORECASE) for pattern in patterns)


def _infer_role_slot_plan(question: str, target_type: str) -> Dict[str, Any]:
    q = re.sub(r"\s+", " ", str(question or "")).strip()
    if _CHAIN_KINSHIP_RE.search(q):
        return {
            "supported": False,
            "name": "chain_kinship_blocked",
            "reason": "chain_kinship_needs_explicit_path_not_slot_projection",
            "relation_pattern": _DIRECT_KINSHIP_RE,
            "expected_target_types": ("person",),
        }
    if target_type == "location" and _CAPITAL_Q_RE.search(q):
        return {
            "supported": True,
            "name": "capital_location_slot",
            "reason": "explicit_capital_slot",
            "relation_pattern": _CAPITAL_EVIDENCE_RE,
            "expected_target_types": ("location",),
            "requires_capital_slot": True,
            "max_relation_distance": 120,
        }
    if target_type in {"person", "organization"} and _CREATIVE_ACTOR_Q_RE.search(q):
        return {
            "supported": True,
            "name": "creative_actor_slot",
            "reason": "explicit_actor_relation_slot",
            "relation_pattern": _CREATIVE_ACTOR_EVIDENCE_RE,
            "expected_target_types": ("person", "organization"),
            "allow_multi_answer": True,
            "max_relation_distance": 150,
        }
    if target_type == "location" and _LOCATION_Q_RE.search(q):
        return {
            "supported": True,
            "name": "location_attribute_slot",
            "reason": "explicit_location_slot",
            "relation_pattern": _LOCATION_EVIDENCE_RE,
            "expected_target_types": ("location",),
            "max_relation_distance": 150,
        }
    if target_type == "date":
        relation_pattern = _required_relation_pattern(q)
        if relation_pattern is not None:
            return {
                "supported": True,
                "name": "temporal_relation_slot",
                "reason": "explicit_temporal_relation_slot",
                "relation_pattern": relation_pattern,
                "expected_target_types": ("date",),
                "requires_event_guard": bool(_EVENT_TERMS_RE.search(q)),
                "max_relation_distance": 130,
            }
    if target_type == "number":
        relation_pattern = _required_relation_pattern(q)
        if relation_pattern is not None:
            return {
                "supported": True,
                "name": "numeric_relation_slot",
                "reason": "explicit_numeric_relation_slot",
                "relation_pattern": relation_pattern,
                "expected_target_types": ("number",),
                "max_relation_distance": 130,
            }
    return {
        "supported": False,
        "name": "unsupported_role_slot",
        "reason": "no_explicit_answer_slot_plan",
        "relation_pattern": None,
        "expected_target_types": (),
    }


def _sentence_has_question_event_anchor(question: str, sentence: str) -> bool:
    question_terms = _question_content_terms(question)
    if not question_terms:
        return True
    sentence_terms = _token_set(sentence)
    event_terms = {term for term in question_terms if len(term) >= 5}
    return bool(event_terms & sentence_terms)


def _candidate_satisfies_role_slot(
    candidate: Dict[str, Any],
    question: str,
    target_type: str,
    slot_plan: Dict[str, Any],
) -> Tuple[bool, str]:
    if not slot_plan.get("supported", False):
        return False, str(slot_plan.get("name") or "unsupported_role_slot")
    expected_types = set(slot_plan.get("expected_target_types") or [])
    if expected_types and target_type not in expected_types:
        return False, "target_type_mismatch"

    candidate_text = str(candidate.get("candidate", "") or "").strip()
    sentence = str(candidate.get("sentence_preview", "") or "")
    title = str(candidate.get("title", "") or "")
    evidence_text = f"{title} {sentence}"
    if not _candidate_has_reasonable_shape(candidate_text, target_type):
        return False, "bad_candidate_shape"
    if _candidate_reuses_question_surface(candidate_text, question):
        return False, "candidate_is_question_or_bridge_anchor"

    relation_pattern = slot_plan.get("relation_pattern")
    if relation_pattern is not None and not relation_pattern.search(evidence_text):
        return False, "slot_relation_absent"
    if not _near_relation(
        candidate_text=candidate_text,
        sentence=sentence,
        relation_pattern=relation_pattern,
        max_distance=int(slot_plan.get("max_relation_distance", 150) or 150),
    ):
        return False, "candidate_not_near_slot_relation"
    if slot_plan.get("requires_capital_slot") and not _capital_sentence_fills_candidate_slot(candidate_text, sentence):
        return False, "candidate_not_capital_slot"
    if slot_plan.get("requires_event_guard"):
        relation_matches_birth_death = bool(_BIRTH_DEATH_RE.search(sentence or ""))
        question_matches_birth_death = bool(_BIRTH_DEATH_RE.search(question or ""))
        if relation_matches_birth_death and not question_matches_birth_death:
            return False, "temporal_birth_death_distractor"
        if not _sentence_has_question_event_anchor(question, sentence):
            return False, "temporal_event_anchor_mismatch"
    return True, "role_slot_matched"


def _clean_extracted_name(name: str) -> str:
    name = re.sub(r"\s+", " ", str(name or "")).strip(" \t\r\n,.;:()[]")
    name = re.sub(r"^(?:and|with|by)\s+", "", name, flags=re.IGNORECASE).strip()
    return name


def _extract_title_names_from_coordination(phrase: str) -> List[str]:
    phrase = _CREATIVE_FOLLOWUP_RELATION_RE.split(str(phrase or ""), maxsplit=1)[0]
    pieces = re.split(r"\s*(?:,|&|\band\b|\bwith\b)\s*", phrase)
    names: List[str] = []
    seen = set()
    for piece in pieces:
        piece = _clean_extracted_name(piece)
        if not piece:
            continue
        match = _TITLE_CASE_RE.search(piece)
        if not match:
            continue
        name = _clean_extracted_name(match.group(0))
        key = _normalize_candidate(name)
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def _extract_multi_answer_slot_rows(
    question: str,
    candidates: List[Dict[str, Any]],
    target_type: str,
    slot_plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if target_type != "person" or not slot_plan.get("allow_multi_answer", False):
        return []
    rows = []
    seen = set()
    for candidate in candidates:
        sentence = str(candidate.get("sentence_preview", "") or "")
        for match in _CREATIVE_BY_RE.finditer(sentence):
            phrase = match.group(1)
            if not _CONJUNCTION_RE.search(phrase):
                continue
            names = []
            for raw_name in _extract_title_names_from_coordination(phrase):
                name = _clean_extracted_name(raw_name)
                key = _normalize_candidate(name)
                if not name or key in seen:
                    continue
                if _candidate_reuses_question_surface(name, question):
                    continue
                if not _candidate_has_person_shape(name):
                    continue
                seen.add(key)
                names.append(name)
            if len(names) < 2 or len(names) > 4:
                continue
            group_candidate = " and ".join(names)
            group_key = _normalize_candidate(group_candidate)
            if not group_key or group_key in seen:
                continue
            seen.add(group_key)
            row = dict(candidate)
            row.update({
                "candidate": group_candidate,
                "key": group_key,
                "confidence": min(0.99, float(candidate.get("confidence", 0.0) or 0.0) + 0.08),
                "best_sentence_score": float(candidate.get("best_sentence_score", 0.0) or 0.0),
                "question_overlap": float(candidate.get("question_overlap", 0.0) or 0.0),
                "provenance_count": max(1, len(names)),
                "answer_arity": len(names),
                "slot_plan": slot_plan.get("name"),
                "role_slot_score": 1.0,
            })
            rows.append(row)
    return rows


def _question_state(planned_retrieval: Dict[str, Any]) -> Dict[str, Any]:
    planned_retrieval = planned_retrieval or {}
    router = planned_retrieval.get("router") if isinstance(planned_retrieval.get("router"), dict) else {}
    verifier = planned_retrieval.get("verifier") if isinstance(planned_retrieval.get("verifier"), dict) else {}
    question_type = (
        planned_retrieval.get("question_type")
        or router.get("question_type")
        or router.get("route")
        or "unavailable"
    )
    try:
        bridge_strength = int(router.get("bridge_strength", planned_retrieval.get("bridge_strength", 0)) or 0)
    except (TypeError, ValueError):
        bridge_strength = 0
    reason = str(planned_retrieval.get("reason") or verifier.get("reason") or "")
    evidence_insufficient = bool(
        planned_retrieval.get("evidence_insufficient")
        or verifier.get("evidence_insufficient")
        or verifier.get("insufficient")
    )
    bridge_cues = (
        str(question_type).lower() in {"bridge_or_compositional", "multi_hop", "comparison"}
        or bridge_strength >= 2
    )
    terminal_state = planned_retrieval_terminal_state(planned_retrieval)
    hard_deficit = bool(
        evidence_insufficient or terminal_state["terminal_evidence_gap"]
    )
    return {
        "question_type": str(question_type),
        "bridge_strength": bridge_strength,
        "bridge_cues": bool(bridge_cues),
        "hard_deficit": bool(hard_deficit),
        "reason": reason,
        "expansion_stopped_by_optimizer": terminal_state[
            "expansion_stopped_by_optimizer"
        ],
    }


def _avg_source_doc_chars(docs: List[str]) -> float:
    source_docs = [str(doc or "") for doc in docs if not _is_synthetic_doc(str(doc or ""))]
    if not source_docs:
        return 0.0
    return sum(len(doc) for doc in source_docs) / max(1, len(source_docs))


def _filter_candidate(
    candidate: Dict[str, Any],
    question: str,
    target_type: str,
    relation_pattern,
    min_question_overlap: float,
    require_role_constraint: bool,
    slot_plan: Dict[str, Any] = None,
) -> Tuple[bool, str]:
    candidate_text = str(candidate.get("candidate", "") or "")
    if not candidate_text:
        return False, "empty_candidate"
    if float(candidate.get("question_overlap", 0.0) or 0.0) < float(min_question_overlap):
        return False, "low_question_overlap"
    if _candidate_reuses_question_number(candidate_text, question):
        return False, "question_number_reuse"
    if _candidate_is_year_like_count_answer(candidate_text, question, target_type):
        return False, "year_like_count_answer"
    if _candidate_is_disallowed_date_range(candidate_text, question, target_type):
        return False, "date_range_without_request"
    if require_role_constraint and not _candidate_matches_required_relation(candidate, relation_pattern):
        return False, "relation_mismatch"
    if require_role_constraint:
        keep, reason = _candidate_satisfies_role_slot(
            candidate=candidate,
            question=question,
            target_type=target_type,
            slot_plan=slot_plan or {},
        )
        if not keep:
            return False, reason
    if not _candidate_matches_question_unit(candidate, question, target_type):
        return False, "unit_mismatch"
    if not _candidate_contains_question_anchor(candidate, question):
        return False, "anchor_mismatch"
    if _candidate_looks_like_secondary_version(candidate, question):
        return False, "secondary_version_distractor"
    return True, "accepted"


def _format_slot_projection_doc(
    target_type: str,
    rows: List[Dict[str, Any]],
) -> str:
    slot_plan = str(rows[0].get("slot_plan") or "role_arity_slot") if rows else "role_arity_slot"
    lines = [
        "Role-Arity Slot-Projected Provenance View",
        f"Target answer type: {target_type}",
        f"Answer-slot plan: {slot_plan}",
        "Answer-slot candidates projected from the retrieved evidence:",
    ]
    for row in rows:
        lines.append(
            "- candidate=\"{candidate}\" type={answer_type} arity={answer_arity} provenance={provenance_count} "
            "confidence={confidence:.3f} source=\"{title}\" evidence=\"{sentence_preview}\"".format(
                candidate=row.get("candidate", ""),
                answer_type=row.get("answer_type", target_type),
                answer_arity=int(row.get("answer_arity", 1) or 1),
                provenance_count=int(row.get("provenance_count", 1) or 1),
                confidence=float(row.get("confidence", 0.0) or 0.0),
                title=str(row.get("title", ""))[:120],
                sentence_preview=str(row.get("sentence_preview", ""))[:240],
            )
        )
    lines.extend([
        "Reader guidance:",
        "- Use a candidate only when its evidence sentence directly fills the requested answer slot.",
        "- Keep the original top evidence as primary; this view only disambiguates answer spans.",
    ])
    return "\n".join(lines)


def build_slot_projected_provenance_view(
    question: str,
    docs: List[str],
    planned_retrieval: Dict[str, Any] = None,
    evidence_admission: Dict[str, Any] = None,
    enabled: bool = False,
    max_docs: int = 8,
    max_rows: int = 3,
    min_confidence: float = 0.70,
    min_margin: float = 0.06,
    min_question_overlap: float = 0.08,
    min_provenance: int = 1,
    insert_after: int = 2,
    bridge_or_deficit_only: bool = True,
    block_simple_factoid_without_deficit: bool = True,
    block_long_context: bool = True,
    long_context_avg_chars: int = 1400,
    require_role_constraint: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    docs = docs or []
    target_type = infer_answer_target_type(question)
    state = _question_state(planned_retrieval or {})
    avg_doc_chars = _avg_source_doc_chars(docs)
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "target_type": target_type,
        "slot_plan": "unavailable",
        "slot_plan_supported": False,
        "slot_plan_reason": "unavailable",
        "original_doc_count": len(docs),
        "final_doc_count": len(docs),
        "candidate_count": 0,
        "row_count": 0,
        "selected_candidate": None,
        "selected_confidence": 0.0,
        "margin": 0.0,
        "question_type": state["question_type"],
        "bridge_strength": state["bridge_strength"],
        "bridge_cues": state["bridge_cues"],
        "hard_deficit": state["hard_deficit"],
        "avg_source_doc_chars": round(avg_doc_chars, 2),
    }

    if not enabled:
        return docs, info
    if not docs:
        info["reason"] = "no_docs"
        return docs, info
    if block_long_context and avg_doc_chars >= float(long_context_avg_chars):
        info["reason"] = "blocked_long_context"
        return docs, info
    if bridge_or_deficit_only and not (state["bridge_cues"] or state["hard_deficit"]):
        info["reason"] = "blocked_no_bridge_or_deficit"
        return docs, info
    if block_simple_factoid_without_deficit and not state["bridge_cues"] and not state["hard_deficit"]:
        info["reason"] = "blocked_simple_factoid_without_deficit"
        return docs, info
    if target_type in {"date", "number"} and not _has_main_numeric_temporal_request(question, target_type):
        info["reason"] = "not_main_numeric_temporal_question"
        return docs, info
    if target_type in {"date", "number"} and KINSHIP_RE.search(question or ""):
        info["reason"] = "kinship_numeric_temporal_question"
        return docs, info

    slot_plan = _infer_role_slot_plan(question=question, target_type=target_type)
    info["slot_plan"] = slot_plan.get("name", "unavailable")
    info["slot_plan_supported"] = bool(slot_plan.get("supported", False))
    info["slot_plan_reason"] = slot_plan.get("reason", "unavailable")
    if require_role_constraint and not slot_plan.get("supported", False):
        info["reason"] = slot_plan.get("name", "unsupported_role_slot")
        return docs, info

    source_docs = [doc for doc in docs if not _is_synthetic_doc(doc)]
    if not source_docs:
        info["reason"] = "no_source_docs"
        return docs, info

    candidates = _rank_candidates(
        question=question,
        answer="",
        qa_docs=source_docs,
        target_type=target_type,
        max_docs=max(1, int(max_docs)),
    )
    relation_pattern = _required_relation_pattern(question)
    if require_role_constraint and slot_plan.get("relation_pattern") is not None:
        relation_pattern = slot_plan.get("relation_pattern")
    multi_answer_rows = _extract_multi_answer_slot_rows(
        question=question,
        candidates=candidates,
        target_type=target_type,
        slot_plan=slot_plan,
    )
    reject_counts = defaultdict(int)
    grouped: Dict[str, Dict[str, Any]] = {}
    for candidate in multi_answer_rows + candidates:
        keep, reason = _filter_candidate(
            candidate=candidate,
            question=question,
            target_type=target_type,
            relation_pattern=relation_pattern,
            min_question_overlap=min_question_overlap,
            require_role_constraint=require_role_constraint,
            slot_plan=slot_plan,
        )
        if not keep:
            reject_counts[reason] += 1
            continue
        key = _normalize_candidate(str(candidate.get("candidate", "")))
        if not key:
            reject_counts["empty_normalized_candidate"] += 1
            continue
        current = grouped.get(key)
        confidence = float(candidate.get("confidence", 0.0) or 0.0)
        if current is None:
            grouped[key] = dict(
                candidate,
                provenance_count=int(candidate.get("provenance_count", 1) or 1),
                _best_confidence=confidence,
            )
        else:
            current["provenance_count"] = int(current.get("provenance_count", 1) or 1) + int(
                candidate.get("provenance_count", 1) or 1
            )
            if confidence > float(current.get("_best_confidence", 0.0) or 0.0):
                current.update(candidate)
                current["_best_confidence"] = confidence

    rows = []
    for item in grouped.values():
        provenance_count = int(item.get("provenance_count", 1) or 1)
        if provenance_count < int(min_provenance):
            reject_counts["low_provenance"] += 1
            continue
        item = dict(item)
        item["confidence"] = float(item.get("_best_confidence", item.get("confidence", 0.0)) or 0.0)
        item.pop("_best_confidence", None)
        rows.append(item)

    rows.sort(
        key=lambda item: (
            float(item.get("confidence", 0.0) or 0.0),
            int(item.get("provenance_count", 1) or 1),
            float(item.get("question_overlap", 0.0) or 0.0),
        ),
        reverse=True,
    )
    rows = rows[:max(1, int(max_rows))]
    info["candidate_count"] = len(candidates)
    info["multi_answer_candidate_count"] = len(multi_answer_rows)
    info["row_count"] = len(rows)
    info["reject_reason_counts"] = dict(reject_counts)
    info["rows"] = rows

    if not rows:
        info["reason"] = "no_rows"
        return docs, info

    top = rows[0]
    confidence = float(top.get("confidence", 0.0) or 0.0)
    second_confidence = float(rows[1].get("confidence", 0.0) or 0.0) if len(rows) > 1 else 0.0
    margin = confidence - second_confidence
    info.update({
        "selected_candidate": top.get("candidate"),
        "selected_confidence": round(confidence, 4),
        "margin": round(margin, 4),
        "selected_provenance_count": int(top.get("provenance_count", 1) or 1),
        "selected_answer_arity": int(top.get("answer_arity", 1) or 1),
    })
    if confidence < float(min_confidence):
        info["reason"] = "low_confidence"
        return docs, info
    if len(rows) > 1 and margin < float(min_margin):
        info["reason"] = "low_margin"
        return docs, info

    projection_doc = _format_slot_projection_doc(target_type=target_type, rows=rows)
    insert_at = min(max(0, int(insert_after)), len(docs))
    final_docs = docs[:insert_at] + [projection_doc] + docs[insert_at:]
    info.update({
        "applied": True,
        "reason": (
            "multi_answer_slot_projection_inserted"
            if int(top.get("answer_arity", 1) or 1) > 1
            else "role_arity_slot_projection_inserted"
        ),
        "insert_at": insert_at,
        "final_doc_count": len(final_docs),
    })
    return final_docs, info
