import re
from typing import Any, Dict, List, Tuple

from .answer_verifier_utils import _rank_candidates
from .sentence_evidence_utils import _doc_title, infer_answer_target_type


SYNTHETIC_DOC_TITLES = {
    "anchored typed candidate table",
    "answer-slot support completion view",
    "candidate answer table",
    "candidate answer contrastive selector",
    "directional fact join candidate table",
    "entity-state evidence control",
    "evidence-grounded answer extractor",
    "evidence answer arbiter",
    "lineage-safe answer-slot support view",
    "multi-granular evidence memory",
    "provenance answer lattice",
    "provenance typed candidate store",
    "role-arity slot-projected provenance view",
    "slot-projected provenance view",
    "slot-separated fact join view",
    "structured evidence chain",
    "support-set completion evidence",
    "typed evidence grounding",
    "typed fact join candidate table",
    "evidence snippets from",
}

MAIN_DATE_REQUEST_RE = re.compile(
    r"^\s*(?:when\b|on\s+what\s+date\b|what\s+date\b)|"
    r"\b(?:what|which|in\s+what|in\s+which)\s+year\b",
    re.IGNORECASE,
)
MAIN_NUMBER_REQUEST_RE = re.compile(
    r"^\s*(?:how\s+many\b|how\s+much\b|how\s+old\b|how\s+long\b|"
    r"what\s+(?:number|population|percentage|percent|age|total|count)\b|"
    r"which\s+(?:number|population|percentage|percent|age|total|count)\b)",
    re.IGNORECASE,
)
NON_NUMERIC_MAIN_RE = re.compile(
    r"^\s*(?:who|where|which\s+(?:person|film|movie|book|song|country|city|state|county)|"
    r"what\s+(?:is|are|was|were)\s+(?:the\s+)?(?:meaning|name|title|occupation|profession|role|job))\b",
    re.IGNORECASE,
)
KINSHIP_RE = re.compile(r"\b(?:father|mother|parent|parents|son|daughter|wife|husband|spouse)\b", re.IGNORECASE)
SECONDARY_VERSION_RE = re.compile(
    r"\b(?:film|movie|adaptation|screenplay|directed\s+by|starring|"
    r"under\s+the\s+title|edition|translated|translation|version|"
    r"united\s+states|u\.s\.|us\s+release|scholastic)\b",
    re.IGNORECASE,
)
QUESTION_SECONDARY_VERSION_RE = re.compile(
    r"\b(?:film|movie|adaptation|screenplay|edition|translated|translation|version|"
    r"united\s+states|u\.s\.|us\s+release)\b",
    re.IGNORECASE,
)
RELATION_CUE_PATTERNS = (
    (re.compile(r"\b(?:born|birthday|birth date)\b", re.IGNORECASE), re.compile(r"\b(?:born|birth|birthday)\b", re.IGNORECASE)),
    (re.compile(r"\b(?:die|died|death)\b", re.IGNORECASE), re.compile(r"\b(?:died|death|dead|deceased)\b", re.IGNORECASE)),
    (re.compile(r"\b(?:release|released|came out|come out)\b", re.IGNORECASE), re.compile(r"\b(?:release|released|came out)\b", re.IGNORECASE)),
    (re.compile(r"\b(?:publish|published|publication)\b", re.IGNORECASE), re.compile(r"\b(?:publish|published|publication)\b", re.IGNORECASE)),
    (re.compile(r"\b(?:founded|formed|established|created)\b", re.IGNORECASE), re.compile(r"\b(?:founded|formed|established|created|founded in|formed in)\b", re.IGNORECASE)),
    (re.compile(r"\b(?:begin|began|started|start)\b", re.IGNORECASE), re.compile(r"\b(?:begin|began|started|start|founded)\b", re.IGNORECASE)),
    (re.compile(r"\b(?:charter|chartered)\b", re.IGNORECASE), re.compile(r"\b(?:charter|chartered|incorporated)\b", re.IGNORECASE)),
    (re.compile(r"\b(?:dissolved|dissolution)\b", re.IGNORECASE), re.compile(r"\b(?:dissolved|dissolution|ended)\b", re.IGNORECASE)),
    (re.compile(r"\b(?:won|win|winner)\b", re.IGNORECASE), re.compile(r"\b(?:won|win|winner|victory)\b", re.IGNORECASE)),
    (re.compile(r"\b(?:wrote|written|composed|composer)\b", re.IGNORECASE), re.compile(r"\b(?:wrote|written|composed|composer|composed by)\b", re.IGNORECASE)),
)
NUMBER_TOKEN_RE = re.compile(r"\d+")
YEAR_LIKE_RE = re.compile(r"^\s*(?:1[5-9]\d{2}|20\d{2}|2100)\s*$")
DATE_RANGE_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)\s*(?:-|–|—|/|to)\s*(?:\d{2}|1[5-9]\d{2}|20\d{2}|2100)\b", re.IGNORECASE)
DATE_RANGE_QUESTION_RE = re.compile(
    r"\b(?:between|from\s+when\s+to\s+when|what\s+years|which\s+years|"
    r"range|period|tenure|term|season|during)\b",
    re.IGNORECASE,
)
NUMBER_UNIT_PATTERNS = (
    (re.compile(r"\b(?:episode|episodes)\b", re.IGNORECASE), re.compile(r"\b(?:episode|episodes)\b", re.IGNORECASE)),
    (re.compile(r"\b(?:room|rooms)\b", re.IGNORECASE), re.compile(r"\b(?:room|rooms)\b", re.IGNORECASE)),
    (re.compile(r"\b(?:season|seasons)\b", re.IGNORECASE), re.compile(r"\b(?:season|seasons)\b", re.IGNORECASE)),
    (re.compile(r"\b(?:age|old)\b", re.IGNORECASE), re.compile(r"\b(?:years?\s+old|age)\b", re.IGNORECASE)),
    (re.compile(r"\b(?:population|people|residents)\b", re.IGNORECASE), re.compile(r"\b(?:population|people|residents|inhabitants)\b", re.IGNORECASE)),
    (re.compile(r"\b(?:percent|percentage|%)\b", re.IGNORECASE), re.compile(r"\b(?:percent|percentage|%)\b", re.IGNORECASE)),
    (re.compile(r"\b(?:mile|miles|km|kilometer|kilometers)\b", re.IGNORECASE), re.compile(r"\b(?:mile|miles|km|kilometer|kilometers)\b", re.IGNORECASE)),
    (re.compile(r"\b(?:dollar|dollars|cost|price|budget|gross|revenue)\b", re.IGNORECASE), re.compile(r"\b(?:\$|dollar|dollars|cost|price|budget|gross|revenue)\b", re.IGNORECASE)),
)
QUERY_ANCHOR_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "answer",
    "before",
    "began",
    "begin",
    "born",
    "bowl",
    "came",
    "charter",
    "chartered",
    "come",
    "composed",
    "created",
    "date",
    "died",
    "does",
    "established",
    "first",
    "formed",
    "founded",
    "from",
    "have",
    "into",
    "last",
    "latest",
    "many",
    "much",
    "number",
    "only",
    "out",
    "published",
    "release",
    "released",
    "started",
    "super",
    "that",
    "their",
    "there",
    "these",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "written",
    "wrote",
    "year",
}


def _is_synthetic_doc(doc: str) -> bool:
    title = _doc_title(doc).lower().strip()
    return any(title.startswith(prefix) for prefix in SYNTHETIC_DOC_TITLES)


def _has_main_numeric_temporal_request(question: str, target_type: str) -> bool:
    question = re.sub(r"\s+", " ", str(question or "")).strip()
    if NON_NUMERIC_MAIN_RE.search(question):
        return False
    if target_type == "date":
        return bool(MAIN_DATE_REQUEST_RE.search(question))
    if target_type == "number":
        return bool(MAIN_NUMBER_REQUEST_RE.search(question))
    return False


def _required_relation_pattern(question: str):
    for question_pattern, evidence_pattern in RELATION_CUE_PATTERNS:
        if question_pattern.search(question):
            return evidence_pattern
    return None


def _candidate_reuses_question_number(candidate: str, question: str) -> bool:
    question_numbers = set(NUMBER_TOKEN_RE.findall(question or ""))
    if not question_numbers:
        return False
    return any(number in question_numbers for number in NUMBER_TOKEN_RE.findall(candidate or ""))


def _candidate_is_year_like_count_answer(candidate: str, question: str, target_type: str) -> bool:
    if target_type != "number":
        return False
    if re.search(r"\b(?:year|date|when)\b", question or "", re.IGNORECASE):
        return False
    return bool(YEAR_LIKE_RE.match(str(candidate or "")))


def _candidate_is_disallowed_date_range(candidate: str, question: str, target_type: str) -> bool:
    if target_type != "date":
        return False
    if DATE_RANGE_QUESTION_RE.search(question or ""):
        return False
    return bool(DATE_RANGE_RE.search(str(candidate or "")))


def _candidate_matches_question_unit(candidate: Dict[str, Any], question: str, target_type: str) -> bool:
    if target_type != "number":
        return True
    evidence_text = f"{candidate.get('title', '')} {candidate.get('sentence_preview', '')}"
    for question_pattern, evidence_pattern in NUMBER_UNIT_PATTERNS:
        if question_pattern.search(question or ""):
            return bool(evidence_pattern.search(evidence_text))
    return True


def _candidate_matches_required_relation(candidate: Dict[str, Any], relation_pattern) -> bool:
    if relation_pattern is None:
        return True
    evidence_text = f"{candidate.get('title', '')} {candidate.get('sentence_preview', '')}"
    return bool(relation_pattern.search(evidence_text))


def _question_anchor_terms(question: str) -> List[str]:
    terms: List[str] = []
    seen = set()
    for token in re.findall(r"[a-z0-9]+", str(question or "").lower()):
        if len(token) < 4:
            continue
        if token in QUERY_ANCHOR_STOPWORDS:
            continue
        if token.isdigit():
            continue
        if token not in seen:
            seen.add(token)
            terms.append(token)
    return terms


def _candidate_contains_question_anchor(candidate: Dict[str, Any], question: str) -> bool:
    anchors = _question_anchor_terms(question)
    if not anchors:
        return True
    evidence_text = f"{candidate.get('title', '')} {candidate.get('sentence_preview', '')}".lower()
    for anchor in anchors:
        variants = {anchor}
        if anchor.endswith("s") and len(anchor) > 4:
            variants.add(anchor[:-1])
        else:
            variants.add(f"{anchor}s")
        if any(re.search(rf"\b{re.escape(variant)}\b", evidence_text) for variant in variants):
            return True
    return False


def _candidate_looks_like_secondary_version(candidate: Dict[str, Any], question: str) -> bool:
    if QUESTION_SECONDARY_VERSION_RE.search(question or ""):
        return False
    if not re.search(r"\b(?:publish|published|publication|wrote|written|release|released)\b", question or "", re.IGNORECASE):
        return False
    evidence_text = f"{candidate.get('title', '')} {candidate.get('sentence_preview', '')}"
    return bool(SECONDARY_VERSION_RE.search(evidence_text))


def _format_selector_doc(
    target_type: str,
    candidates: List[Dict[str, Any]],
) -> str:
    top = candidates[0]
    lines = [
        "Candidate Answer Contrastive Selector",
        f"Target answer type: {target_type}",
        "Best supported answer candidate:",
        f"- {top['candidate']} [{top['title']}]: {top['sentence_preview']}",
        "Selection note:",
        "- Use this candidate only if its evidence sentence directly answers the question.",
        "- Prefer the candidate whose type and relation match the question over nearby numeric/date distractors.",
    ]
    if len(candidates) > 1:
        lines.append("Lower-confidence alternatives:")
        for item in candidates[1:]:
            lines.append(f"- {item['candidate']} [{item['title']}]: {item['sentence_preview']}")
    return "\n".join(lines)


def build_candidate_answer_contrastive_selector(
    question: str,
    docs: List[str],
    enabled: bool = False,
    numeric_temporal_only: bool = True,
    min_confidence: float = 0.76,
    min_margin: float = 0.08,
    min_question_overlap: float = 0.08,
    max_docs: int = 8,
    max_candidates: int = 3,
    insert_after: int = 2,
) -> Tuple[List[str], Dict[str, Any]]:
    docs = docs or []
    target_type = infer_answer_target_type(question)
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "target_type": target_type,
        "original_doc_count": len(docs),
        "final_doc_count": len(docs),
        "candidate_count": 0,
        "selected_candidate": None,
        "selected_confidence": 0.0,
        "margin": 0.0,
        "numeric_temporal_only": bool(numeric_temporal_only),
    }

    if not enabled:
        return docs, info
    if not docs:
        info["reason"] = "no_docs"
        return docs, info
    if numeric_temporal_only and target_type not in {"date", "number"}:
        info["reason"] = "unsupported_target_type"
        return docs, info
    if target_type in {"date", "number"} and not _has_main_numeric_temporal_request(question, target_type):
        info["reason"] = "not_main_numeric_temporal_question"
        return docs, info
    if target_type in {"date", "number"} and KINSHIP_RE.search(question):
        info["reason"] = "kinship_numeric_temporal_question"
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
    candidates = [
        item for item in candidates
        if float(item.get("question_overlap", 0.0)) >= float(min_question_overlap)
        and not _candidate_reuses_question_number(str(item.get("candidate", "")), question)
        and not _candidate_is_year_like_count_answer(str(item.get("candidate", "")), question, target_type)
        and not _candidate_is_disallowed_date_range(str(item.get("candidate", "")), question, target_type)
        and _candidate_matches_required_relation(item, relation_pattern)
        and _candidate_matches_question_unit(item, question, target_type)
        and _candidate_contains_question_anchor(item, question)
        and not _candidate_looks_like_secondary_version(item, question)
    ]
    max_candidates = max(1, int(max_candidates))
    candidates = candidates[:max_candidates]
    info["candidate_count"] = len(candidates)
    info["candidates"] = candidates

    if not candidates:
        info["reason"] = "no_candidates"
        return docs, info

    top = candidates[0]
    second_confidence = float(candidates[1].get("confidence", 0.0)) if len(candidates) > 1 else 0.0
    confidence = float(top.get("confidence", 0.0))
    margin = confidence - second_confidence
    info.update({
        "selected_candidate": top.get("candidate"),
        "selected_confidence": round(confidence, 4),
        "margin": round(margin, 4),
    })

    if confidence < float(min_confidence):
        info["reason"] = "low_confidence"
        return docs, info
    if len(candidates) > 1 and margin < float(min_margin):
        info["reason"] = "low_margin"
        return docs, info

    selector_doc = _format_selector_doc(target_type=target_type, candidates=candidates)
    insert_at = min(max(0, int(insert_after)), len(docs))
    final_docs = docs[:insert_at] + [selector_doc] + docs[insert_at:]
    info.update({
        "applied": True,
        "reason": "candidate_selector_inserted",
        "final_doc_count": len(final_docs),
        "insert_at": insert_at,
    })
    return final_docs, info
