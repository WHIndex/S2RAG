import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from .sentence_evidence_utils import infer_answer_target_type


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "how", "in",
    "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
    "there", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "whom", "whose", "why", "with",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
TITLE_CASE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+(?:of|the|and|de|la|[A-Z][a-z]+)){0,5}\b")
LONG_ANSWER_RE = re.compile(
    r"\b(?:why|how|explain|describe|summari[sz]e|what happened|what happens|"
    r"what is the relationship|relationship between|main reason|story|plot|"
    r"event|events|cause|effect|motivation|role does|role did)\b",
    re.IGNORECASE,
)


def _doc_title(doc: str) -> str:
    if not doc:
        return ""
    return str(doc).split("\n", 1)[0].strip()


def _doc_body(doc: str) -> str:
    if not doc:
        return ""
    parts = str(doc).split("\n", 1)
    return parts[1] if len(parts) > 1 else parts[0]


def _tokens(text: Any) -> List[str]:
    return [
        token
        for token in TOKEN_RE.findall(str(text or "").lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def _token_set(text: Any) -> set:
    return set(_tokens(text))


def _overlap(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def _split_sentences(text: str, max_body_chars: int = 10000) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "")[:max_body_chars]).strip()
    if not text:
        return []

    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    sentences = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) <= 440:
            sentences.append(piece)
            continue
        clauses = re.split(r"(?<=[,;:])\s+", piece)
        current = ""
        for clause in clauses:
            if len(current) + len(clause) + 1 <= 440:
                current = f"{current} {clause}".strip()
            else:
                if current:
                    sentences.append(current)
                current = clause
        if current:
            sentences.append(current[:440].strip())
    return sentences


def _question_shape(question: str, docs: List[str], min_avg_doc_chars: int) -> Tuple[str, bool, Dict[str, Any]]:
    body_lengths = [len(_doc_body(doc)) for doc in docs[: max(1, min(5, len(docs)))]]
    avg_doc_chars = sum(body_lengths) / len(body_lengths) if body_lengths else 0.0
    long_cue = bool(LONG_ANSWER_RE.search(str(question or "")))
    target_type = infer_answer_target_type(question)
    factoid_target = target_type in {"date", "number", "location", "person", "organization", "work", "occupation"}
    long_context = avg_doc_chars >= max(0, int(min_avg_doc_chars))

    if long_cue:
        return "long_answer_cue", True, {
            "avg_doc_chars": round(avg_doc_chars, 2),
            "long_answer_cue": long_cue,
            "long_context": long_context,
            "target_type": target_type,
        }
    if long_context and not factoid_target:
        return "long_context_non_factoid", True, {
            "avg_doc_chars": round(avg_doc_chars, 2),
            "long_answer_cue": long_cue,
            "long_context": long_context,
            "target_type": target_type,
        }
    if long_context and str(question or "").strip().lower().startswith(("what", "how")):
        return "long_context_open_question", True, {
            "avg_doc_chars": round(avg_doc_chars, 2),
            "long_answer_cue": long_cue,
            "long_context": long_context,
            "target_type": target_type,
        }
    return "factoid_or_short_context", False, {
        "avg_doc_chars": round(avg_doc_chars, 2),
        "long_answer_cue": long_cue,
        "long_context": long_context,
        "target_type": target_type,
    }


def _normalized(value: Any) -> str:
    return str(value or "").strip().lower()


def _selective_controller_decision(
    shape: str,
    should_apply_shape: bool,
    shape_info: Dict[str, Any],
    question_type: str = "",
    route_profile: str = "",
    evidence_selection_reason: str = "",
    long_answer_only: bool = True,
    controller_mode: str = "selective",
    controller_min_avg_doc_chars: int = 500,
    skip_no_useful_extra_candidates: bool = True,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Gate the memory document with query-level retrieval diagnostics.

    The first MGM version helped long-context datasets but hurt short-answer
    bridge/factoid questions. This controller keeps the shape detector as a
    cheap first pass, then requires either long-source evidence or a route that
    historically benefits from compact cross-granularity memory.
    """
    mode = _normalized(controller_mode) or "shape"
    question_type = _normalized(question_type)
    route_profile = _normalized(route_profile)
    evidence_selection_reason = _normalized(evidence_selection_reason)
    avg_doc_chars = float(shape_info.get("avg_doc_chars", 0.0) or 0.0)
    min_avg_chars = max(0, int(controller_min_avg_doc_chars))
    long_doc_signal = avg_doc_chars >= min_avg_chars
    expanded_signal = evidence_selection_reason == "expanded"
    long_context_signal = (
        shape in {"long_context_non_factoid", "long_context_open_question"}
        or question_type == "long_context_evidence"
        or route_profile == "long_context"
    )

    controller_info = {
        "controller_mode": mode,
        "controller_min_avg_doc_chars": min_avg_chars,
        "controller_long_doc_signal": long_doc_signal,
        "controller_expanded_signal": expanded_signal,
        "controller_long_context_signal": long_context_signal,
        "controller_question_type": question_type,
        "controller_route_profile": route_profile,
        "controller_evidence_selection_reason": evidence_selection_reason,
        "controller_skip_no_useful_extra_candidates": bool(skip_no_useful_extra_candidates),
    }

    if long_answer_only and not should_apply_shape:
        return False, "not_long_answer_shape", controller_info

    if mode in {"off", "none", "shape", "legacy"}:
        return True, "shape_gate_passed", controller_info

    if mode != "selective":
        controller_info["controller_unknown_mode"] = mode
        return True, "shape_gate_passed", controller_info

    if skip_no_useful_extra_candidates and evidence_selection_reason == "no_useful_extra_candidates":
        return False, "controller_no_useful_extra_candidates", controller_info

    if long_doc_signal:
        return True, "controller_long_doc_signal", controller_info

    if question_type == "long_context_evidence" or long_context_signal:
        return True, "controller_long_context_signal", controller_info

    if question_type == "long_answer_or_narrative" and shape == "long_answer_cue":
        return True, "controller_narrative_cue", controller_info

    if question_type == "single_hop_attribute" and (expanded_signal or shape != "long_answer_cue"):
        return True, "controller_single_hop_attribute", controller_info

    mid_context_threshold = min_avg_chars * 0.7
    if expanded_signal and avg_doc_chars >= mid_context_threshold:
        if question_type in {"bridge_or_compositional", "numeric_or_temporal", "generic_factoid"}:
            return True, "controller_expanded_mid_context", controller_info

    return False, "controller_low_memory_gain_signal", controller_info


def _score_sentence(
    question_tokens: set,
    title_tokens: set,
    doc_anchor_tokens: set,
    sentence: str,
    sentence_index: int,
    doc_index: int,
) -> Dict[str, Any]:
    sent_tokens = _token_set(sentence)
    question_overlap = _overlap(question_tokens, sent_tokens)
    title_overlap = _overlap(question_tokens, title_tokens)
    anchor_overlap = _overlap(doc_anchor_tokens | question_tokens, sent_tokens | title_tokens)
    entity_count = len(TITLE_CASE_RE.findall(sentence))
    entity_signal = min(1.0, entity_count / 3.0)
    rank_prior = 1.0 / (doc_index + 1)
    sentence_prior = 1.0 / (sentence_index + 1)
    score = (
        0.42 * question_overlap
        + 0.18 * title_overlap
        + 0.18 * anchor_overlap
        + 0.10 * entity_signal
        + 0.07 * rank_prior
        + 0.05 * sentence_prior
    )
    return {
        "score": score,
        "question_overlap": question_overlap,
        "title_overlap": title_overlap,
        "anchor_overlap": anchor_overlap,
        "entity_signal": entity_signal,
    }


def _collect_cross_doc_anchors(selected: List[Dict[str, Any]], max_anchors: int = 10) -> List[str]:
    anchor_docs = defaultdict(set)
    for item in selected:
        text = f"{item['title']} {item['sentence']}"
        for match in TITLE_CASE_RE.findall(text):
            anchor = re.sub(r"\s+", " ", match.strip())
            if len(anchor) <= 2:
                continue
            anchor_docs[anchor].add(item["doc_index"])
    ranked = sorted(
        anchor_docs.items(),
        key=lambda pair: (-len(pair[1]), -len(pair[0].split()), pair[0].lower()),
    )
    return [anchor for anchor, doc_ids in ranked[:max_anchors] if len(doc_ids) >= 1]


def _format_memory_doc(
    question: str,
    target_type: str,
    selected: List[Dict[str, Any]],
    cross_doc_anchors: List[str],
    max_chars: int,
) -> str:
    lines = [
        "Multi-Granular Evidence Memory",
        f"Question focus: {question}",
        f"Expected answer type: {target_type}",
    ]
    if cross_doc_anchors:
        lines.append("Cross-document anchors: " + "; ".join(cross_doc_anchors))
    lines.append("Evidence map:")
    for item in selected:
        lines.append(f"- [{item['title']}] {item['sentence']}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 3)].rstrip() + "..."
    return text


def build_multigranular_evidence_memory(
    question: str,
    docs: List[str],
    enabled: bool = False,
    max_docs: int = 8,
    max_sentences_per_doc: int = 2,
    max_total_sentences: int = 14,
    keep_full_top_docs: int = 2,
    min_sentence_score: float = 0.12,
    max_memory_chars: int = 3600,
    long_answer_only: bool = True,
    min_avg_doc_chars: int = 1400,
    question_type: str = "",
    route_profile: str = "",
    evidence_selection_reason: str = "",
    controller_mode: str = "selective",
    controller_min_avg_doc_chars: int = 500,
    skip_no_useful_extra_candidates: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    """Add a compact cross-granularity memory document for long-answer QA.

    The memory is built from the retrieved passages only. It keeps full leading
    passages intact, then inserts a synthetic evidence map that connects document
    titles, high-signal sentences, and repeated anchors.
    """
    docs = docs or []
    max_docs = max(1, int(max_docs))
    max_sentences_per_doc = max(1, int(max_sentences_per_doc))
    max_total_sentences = max(1, int(max_total_sentences))
    keep_full_top_docs = max(0, int(keep_full_top_docs))
    max_memory_chars = max(600, int(max_memory_chars))
    min_sentence_score = max(0.0, float(min_sentence_score))
    shape, should_apply_shape, shape_info = _question_shape(question, docs, min_avg_doc_chars)

    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "question_shape": shape,
        "target_type": shape_info.get("target_type"),
        "long_answer_only": bool(long_answer_only),
        "original_doc_count": len(docs),
        "final_doc_count": len(docs),
        "max_docs": max_docs,
        "max_sentences_per_doc": max_sentences_per_doc,
        "max_total_sentences": max_total_sentences,
        "keep_full_top_docs": keep_full_top_docs,
        "min_sentence_score": min_sentence_score,
        "max_memory_chars": max_memory_chars,
        "min_avg_doc_chars": min_avg_doc_chars,
        "selected_sentence_count": 0,
        "selected_titles": [],
        "cross_doc_anchors": [],
        "question_type": _normalized(question_type),
        "route_profile": _normalized(route_profile),
        "evidence_selection_reason": _normalized(evidence_selection_reason),
        **shape_info,
    }

    if not enabled:
        return docs, info
    if not docs:
        info["reason"] = "no_docs"
        return docs, info
    should_apply_controller, controller_reason, controller_info = _selective_controller_decision(
        shape=shape,
        should_apply_shape=should_apply_shape,
        shape_info=shape_info,
        question_type=question_type,
        route_profile=route_profile,
        evidence_selection_reason=evidence_selection_reason,
        long_answer_only=long_answer_only,
        controller_mode=controller_mode,
        controller_min_avg_doc_chars=controller_min_avg_doc_chars,
        skip_no_useful_extra_candidates=skip_no_useful_extra_candidates,
    )
    info.update(controller_info)
    if not should_apply_controller:
        info["reason"] = controller_reason
        return docs, info

    question_tokens = _token_set(question)
    if not question_tokens:
        info["reason"] = "no_question_tokens"
        return docs, info

    doc_anchor_tokens = set()
    for doc in docs[:max_docs]:
        doc_anchor_tokens.update(_token_set(_doc_title(doc)))

    scored = []
    per_doc_seen = Counter()
    for doc_index, doc in enumerate(docs[:max_docs]):
        title = _doc_title(doc)
        title_tokens = _token_set(title)
        body = _doc_body(doc)
        for sentence_index, sentence in enumerate(_split_sentences(body)):
            if not _token_set(sentence):
                continue
            score_info = _score_sentence(
                question_tokens=question_tokens,
                title_tokens=title_tokens,
                doc_anchor_tokens=doc_anchor_tokens,
                sentence=sentence,
                sentence_index=sentence_index,
                doc_index=doc_index,
            )
            if score_info["score"] < min_sentence_score:
                continue
            scored.append({
                "doc_index": doc_index,
                "sentence_index": sentence_index,
                "title": title,
                "sentence": sentence,
                **score_info,
            })

    if not scored:
        info["reason"] = "no_high_signal_sentences"
        return docs, info

    scored.sort(key=lambda item: (-item["score"], item["doc_index"], item["sentence_index"]))
    selected = []
    seen_sentence_keys = set()
    for item in scored:
        if len(selected) >= max_total_sentences:
            break
        if per_doc_seen[item["doc_index"]] >= max_sentences_per_doc:
            continue
        sentence_key = re.sub(r"\W+", " ", item["sentence"].lower()).strip()
        if not sentence_key or sentence_key in seen_sentence_keys:
            continue
        selected.append(item)
        seen_sentence_keys.add(sentence_key)
        per_doc_seen[item["doc_index"]] += 1

    if not selected:
        info["reason"] = "no_selected_sentences"
        return docs, info

    selected.sort(key=lambda item: (item["doc_index"], item["sentence_index"]))
    cross_doc_anchors = _collect_cross_doc_anchors(selected)
    memory_doc = _format_memory_doc(
        question=question,
        target_type=shape_info.get("target_type", "factoid"),
        selected=selected,
        cross_doc_anchors=cross_doc_anchors,
        max_chars=max_memory_chars,
    )

    insert_at = min(keep_full_top_docs, len(docs))
    final_docs = docs[:insert_at] + [memory_doc] + docs[insert_at:]

    selected_titles = []
    seen_titles = set()
    for item in selected:
        if item["title"] and item["title"] not in seen_titles:
            selected_titles.append(item["title"])
            seen_titles.add(item["title"])

    info.update({
        "applied": True,
        "reason": "memory_inserted",
        "final_doc_count": len(final_docs),
        "selected_sentence_count": len(selected),
        "selected_titles": selected_titles,
        "cross_doc_anchors": cross_doc_anchors,
        "memory_chars": len(memory_doc),
        "insert_at": insert_at,
        "top_sentences": [
            {
                "doc_index": item["doc_index"],
                "sentence_index": item["sentence_index"],
                "title": item["title"],
                "score": round(item["score"], 6),
                "question_overlap": round(item["question_overlap"], 6),
                "anchor_overlap": round(item["anchor_overlap"], 6),
                "sentence_preview": item["sentence"][:240],
            }
            for item in sorted(selected, key=lambda x: -x["score"])[:8]
        ],
    })
    return final_docs, info
