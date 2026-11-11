import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "how", "in",
    "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
    "there", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "whom", "whose", "why", "with",
}

ANSWER_TYPE_TERMS = {
    "album", "area", "author", "book", "borough", "capital", "city",
    "college", "company", "country", "county", "creator", "date",
    "day", "department", "director", "district", "employer", "film",
    "founder", "island", "language", "location", "mountain", "municipality",
    "name", "nation", "occupation", "person", "place", "population",
    "province", "publisher", "region", "river", "school", "seat", "song",
    "state", "team", "town", "university", "village", "writer", "year",
}

GENERIC_BRIDGE_TERMS = {
    "also", "became", "become", "birth", "born", "called", "county", "died",
    "famous", "known", "located", "made", "named", "place", "seat", "team",
    "used", "where", "which", "whose", "would", "east", "north", "south", "west",
}

GEO_TITLE_TERMS = {
    "borough", "city", "county", "district", "municipality", "province",
    "region", "state", "town", "village",
}

BRIDGE_CONTRASTIVE_RISK_TYPE_TERMS = {
    "borough", "capital", "city", "country", "county", "department",
    "district", "island", "location", "mountain", "municipality", "place",
    "province", "region", "river", "seat", "state", "town", "village",
}

BRIDGE_CONTRASTIVE_EXPLICIT_SURFACE_TERMS = {
    "bay", "canal", "creek", "gulf", "lake", "mountain", "river", "sea",
}

PLACE_SUFFIX_TERMS = {
    "alabama", "alaska", "arizona", "arkansas", "california", "carolina",
    "colorado", "connecticut", "dakota", "delaware", "florida", "georgia",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada", "ohio",
    "oklahoma", "oregon", "pennsylvania", "tennessee", "texas", "utah",
    "vermont", "virginia", "washington", "wisconsin", "wyoming",
}

DATE_VALUE_RE = re.compile(
    r"\b(?:\d{1,2}\s+)?(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{1,2},?\s+\d{3,4}\b|"
    r"\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{3,4}\b|"
    r"\b(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b",
    re.IGNORECASE,
)
NUMBER_VALUE_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")

BRIDGE_CUE_RE = re.compile(
    r"\b(?:"
    r"the\s+(?:city|country|region|state|county|film|album|book|team|company|school|university)\s+(?:where|that|which)|"
    r"(?:city|country|region|state|county|film|album|book|team|company|school|university)\s+where|"
    r"(?:director|writer|author|composer|producer|creator|founder|designer|employer|birthplace|capital|location)\s+of|"
    r"where\s+.+\s+(?:born|located|died|founded|created|released|published)|"
    r"when\s+.+\s+(?:born|founded|created|released|published|abolished)|"
    r"immediately\s+(?:north|south|east|west)\s+of|"
    r"same\s+(?:country|city|state|county)|"
    r"both\s+.+\s+and|"
    r"worked\s+on\s+by\s+both|"
    r"born\s+in\s+.+\s+what|"
    r"in\s+which\s+.+\s+was"
    r")\b",
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


def _tokens(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def _token_set(text: str) -> set:
    return set(_tokens(text))


def _normalized_surface(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _raw_distinctive_tokens(text: str) -> set:
    raw_tokens = re.findall(r"[\w]+", str(text or "").lower(), flags=re.UNICODE)
    blocked = STOPWORDS | ANSWER_TYPE_TERMS | GENERIC_BRIDGE_TERMS | PLACE_SUFFIX_TERMS
    return {
        token
        for token in raw_tokens
        if len(token) > 1 and token not in blocked
    }


def _overlap_ratio(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def _capitalized_phrase_token_sets(text: str) -> List[set]:
    phrase_sets: List[set] = []
    raw_text = str(text or "")
    quoted_spans = re.findall(r'"([^"]+)"|\'([^\']+)\'', raw_text)
    for span_a, span_b in quoted_spans:
        tokens = set(_tokens(span_a or span_b))
        tokens = {
            token
            for token in tokens
            if token not in ANSWER_TYPE_TERMS and token not in PLACE_SUFFIX_TERMS
        }
        if len(tokens) >= 1:
            phrase_sets.append(tokens)


    for match in re.finditer(
        r"\b(?:[A-Z][A-Za-z0-9'’.-]+(?:\s+(?:of|the|and|de|del|da|la|le|[A-Z][A-Za-z0-9'’.-]+)){0,5})",
        raw_text,
    ):
        tokens = set(_tokens(match.group(0)))
        tokens = {
            token
            for token in tokens
            if token not in ANSWER_TYPE_TERMS and token not in PLACE_SUFFIX_TERMS
        }
        if len(tokens) >= 1:
            phrase_sets.append(tokens)

    deduped: List[set] = []
    seen = set()
    for tokens in phrase_sets:
        key = tuple(sorted(tokens))
        if key and key not in seen:
            deduped.append(tokens)
            seen.add(key)
    return deduped


def _bridge_anchor_profile(question: str, docs: List[str], protect_top_k: int) -> Dict[str, Any]:
    question_tokens = _token_set(question)
    question_lower = str(question or "").lower()
    requires_date_value = bool(
        re.search(r"\b(?:when|date|year|month|day|born|died|released|published|founded)\b", question_lower)
    )
    requires_number_value = bool(
        re.search(r"\b(?:how many|how much|number|population|total|count|age|percent|percentage)\b", question_lower)
    )
    anchor_tokens = {
        token
        for token in question_tokens
        if (
            token not in ANSWER_TYPE_TERMS
            and token not in GENERIC_BRIDGE_TERMS
            and token not in PLACE_SUFFIX_TERMS
        )
    }
    phrase_sets = _capitalized_phrase_token_sets(question)
    for tokens in phrase_sets:
        anchor_tokens.update(tokens)

    protected_docs = docs[:max(1, protect_top_k)]
    protected_title_tokens = set()
    protected_body_tokens = set()
    for doc in protected_docs:
        protected_title_tokens.update(_token_set(_doc_title(doc)))
        protected_body_tokens.update(_token_set(_doc_body(doc)[:1800]))

    protected_title_tokens = {
        token
        for token in protected_title_tokens
        if (
            token not in ANSWER_TYPE_TERMS
            and token not in GENERIC_BRIDGE_TERMS
            and token not in PLACE_SUFFIX_TERMS
        )
    }
    anchor_tokens.update(protected_title_tokens)

    return {
        "question_tokens": question_tokens,
        "type_tokens": question_tokens & ANSWER_TYPE_TERMS,
        "anchor_tokens": anchor_tokens,
        "anchor_phrase_sets": phrase_sets,
        "protected_title_tokens": protected_title_tokens,
        "protected_body_tokens": protected_body_tokens,
    }


def _doc_contrast_features(
    question_tokens: set,
    type_tokens: set,
    anchor_tokens: set,
    anchor_phrase_sets: List[set],
    trusted_title_tokens: set,
    doc: str,
) -> Dict[str, Any]:
    title = _doc_title(doc)
    title_tokens = _token_set(title)
    body_tokens = _token_set(_doc_body(doc)[:2600])
    all_tokens = title_tokens | body_tokens
    non_type_title_tokens = {
        token
        for token in title_tokens
        if (
            token not in ANSWER_TYPE_TERMS
            and token not in GENERIC_BRIDGE_TERMS
            and token not in PLACE_SUFFIX_TERMS
        )
    }
    anchor_overlap = _overlap_ratio(anchor_tokens, all_tokens)
    title_anchor_overlap = _overlap_ratio(non_type_title_tokens, anchor_tokens | trusted_title_tokens)
    query_overlap = _overlap_ratio(question_tokens, all_tokens)
    type_overlap = _overlap_ratio(type_tokens, all_tokens)
    phrase_hits = 0
    for phrase_tokens in anchor_phrase_sets:
        if phrase_tokens and _overlap_ratio(phrase_tokens, all_tokens) >= 0.67:
            phrase_hits += 1
    bridge_link = max(anchor_overlap, title_anchor_overlap, 1.0 if phrase_hits else 0.0)
    title_geo_like = bool(title_tokens & (GEO_TITLE_TERMS | PLACE_SUFFIX_TERMS))
    answer_type_like = bool(type_tokens and type_overlap > 0.0)
    return {
        "title": title,
        "title_tokens": title_tokens,
        "raw_distinctive_title_tokens": _raw_distinctive_tokens(title),
        "all_tokens": all_tokens,
        "non_type_title_tokens": non_type_title_tokens,
        "anchor_overlap": anchor_overlap,
        "title_anchor_overlap": title_anchor_overlap,
        "query_overlap": query_overlap,
        "type_overlap": type_overlap,
        "phrase_hits": phrase_hits,
        "bridge_link": bridge_link,
        "title_geo_like": title_geo_like,
        "answer_type_like": answer_type_like,
    }


def _chain_role_context(
    question: str,
    planned_retrieval: Optional[Dict[str, Any]] = None,
    path_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    planned_retrieval = planned_retrieval or {}
    path_evidence = path_evidence or {}

    planned_text_parts: List[str] = []
    for key in ("queries", "appended_titles"):
        values = planned_retrieval.get(key) if isinstance(planned_retrieval, dict) else None
        if isinstance(values, list):
            planned_text_parts.extend(str(value) for value in values if value)
        elif values:
            planned_text_parts.append(str(values))

    path_titles: List[str] = []
    for key in ("selected_titles", "appended_titles"):
        values = path_evidence.get(key) if isinstance(path_evidence, dict) else None
        if isinstance(values, list):
            path_titles.extend(str(value) for value in values if value)
        elif values:
            path_titles.append(str(values))

    planned_text = " ".join(planned_text_parts)
    return {
        "question_surface": _normalized_surface(question),
        "planned_surface": _normalized_surface(planned_text),
        "planned_distinctive_tokens": _raw_distinctive_tokens(planned_text),
        "path_title_surfaces": {
            _normalized_surface(title)
            for title in path_titles
            if _normalized_surface(title)
        },
    }


def _chain_role_protection_reason(
    title: str,
    features: Dict[str, Any],
    chain_context: Dict[str, Any],
) -> Optional[str]:
    title_surface = _normalized_surface(title)
    if not title_surface:
        return None

    raw_title_tokens = features.get("raw_distinctive_title_tokens") or set()
    title_tokens = features.get("title_tokens") or set()
    planned_tokens = chain_context.get("planned_distinctive_tokens") or set()
    planned_surface = chain_context.get("planned_surface", "")
    question_surface = chain_context.get("question_surface", "")
    path_title_surfaces = chain_context.get("path_title_surfaces") or set()

    has_named_title_signal = bool(raw_title_tokens)
    exact_surface_bridge = bool(title_tokens & BRIDGE_CONTRASTIVE_EXPLICIT_SURFACE_TERMS)

    if has_named_title_signal and title_surface in planned_surface:
        return "planned_query_or_append_title"

    if has_named_title_signal and raw_title_tokens and planned_tokens:
        overlap = len(raw_title_tokens & planned_tokens) / max(1, len(raw_title_tokens))
        if overlap >= 0.67:
            return "planned_distinctive_title_overlap"

    if has_named_title_signal and title_surface in path_title_surfaces and planned_tokens:
        overlap = len(raw_title_tokens & planned_tokens) / max(1, len(raw_title_tokens))
        if overlap >= 0.50:
            return "path_selected_planned_title_overlap"

    if title_surface in question_surface:
        if has_named_title_signal:
            return "question_explicit_named_title"
        if exact_surface_bridge:
            return "question_explicit_bridge_surface"

    return None


def _bridge_state_metrics(
    order_indices: List[int],
    feature_by_idx: Dict[int, Dict[str, Any]],
    risky_indices: set,
    chain_context: Dict[str, Any],
    safety_top_k: int,
    min_bridge_link: float,
    min_title_anchor_overlap: float,
) -> Dict[str, Any]:
    top_indices = order_indices[:max(1, int(safety_top_k))]
    planned_tokens = chain_context.get("planned_distinctive_tokens") or set()

    chain_role_count = 0
    bridge_connected_count = 0
    risky_count = 0
    anchor_score = 0.0
    planned_title_tokens = set()
    chain_role_reasons = Counter()
    titles = []

    for idx in top_indices:
        features = feature_by_idx.get(idx)
        if not features:
            continue
        titles.append(features.get("title", ""))
        bridge_link = max(
            float(features.get("anchor_overlap", 0.0) or 0.0),
            float(features.get("title_anchor_overlap", 0.0) or 0.0),
            1.0 if int(features.get("phrase_hits", 0) or 0) else 0.0,
        )
        anchor_score += bridge_link
        if (
            int(features.get("phrase_hits", 0) or 0) > 0
            or float(features.get("anchor_overlap", 0.0) or 0.0) >= min_bridge_link
            or float(features.get("title_anchor_overlap", 0.0) or 0.0) >= min_title_anchor_overlap
        ):
            bridge_connected_count += 1

        raw_title_tokens = features.get("raw_distinctive_title_tokens") or set()
        planned_title_tokens.update(raw_title_tokens & planned_tokens)

        chain_role_reason = _chain_role_protection_reason(
            title=features.get("title", ""),
            features=features,
            chain_context=chain_context,
        )
        if chain_role_reason:
            chain_role_count += 1
            chain_role_reasons[chain_role_reason] += 1

        if idx in risky_indices:
            risky_count += 1

    planned_title_token_hits = len(planned_title_tokens)
    score = (
        2.00 * chain_role_count
        + 0.75 * bridge_connected_count
        + 0.60 * anchor_score
        + 0.12 * min(planned_title_token_hits, 6)
        - 0.18 * risky_count
    )
    return {
        "top_indices": top_indices,
        "top_titles": titles[:8],
        "chain_role_count": chain_role_count,
        "bridge_connected_count": bridge_connected_count,
        "risky_count": risky_count,
        "anchor_score": round(anchor_score, 6),
        "planned_title_token_hits": planned_title_token_hits,
        "chain_role_reasons": dict(chain_role_reasons),
        "score": round(score, 6),
    }


def _bridge_counterfactual_safety_gate(
    old_order_indices: List[int],
    new_order_indices: List[int],
    feature_by_idx: Dict[int, Dict[str, Any]],
    risky_indices: set,
    chain_context: Dict[str, Any],
    enabled: bool,
    safety_top_k: int,
    min_score_drop: float,
    min_bridge_link: float,
    min_title_anchor_overlap: float,
) -> Dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "reason": "disabled",
            "vetoed": False,
        }
    if old_order_indices == new_order_indices:
        return {
            "enabled": True,
            "reason": "not_reordered",
            "vetoed": False,
        }

    safety_top_k = max(2, int(safety_top_k))
    min_score_drop = max(0.0, float(min_score_drop))
    old_state = _bridge_state_metrics(
        order_indices=old_order_indices,
        feature_by_idx=feature_by_idx,
        risky_indices=risky_indices,
        chain_context=chain_context,
        safety_top_k=safety_top_k,
        min_bridge_link=min_bridge_link,
        min_title_anchor_overlap=min_title_anchor_overlap,
    )
    new_state = _bridge_state_metrics(
        order_indices=new_order_indices,
        feature_by_idx=feature_by_idx,
        risky_indices=risky_indices,
        chain_context=chain_context,
        safety_top_k=safety_top_k,
        min_bridge_link=min_bridge_link,
        min_title_anchor_overlap=min_title_anchor_overlap,
    )

    score_delta = float(new_state["score"]) - float(old_state["score"])
    chain_role_loss = int(old_state["chain_role_count"]) - int(new_state["chain_role_count"])
    bridge_loss = (
        int(old_state["bridge_connected_count"])
        - int(new_state["bridge_connected_count"])
    )
    planned_title_loss = (
        int(old_state["planned_title_token_hits"])
        - int(new_state["planned_title_token_hits"])
    )

    veto_reason = None
    if chain_role_loss > 0:
        veto_reason = "chain_role_coverage_loss"
    elif bridge_loss >= 2 and score_delta < 0.0:
        veto_reason = "bridge_coverage_loss"
    elif planned_title_loss >= 2 and score_delta < -min_score_drop:
        veto_reason = "planned_title_signal_loss"
    elif score_delta < -min_score_drop:
        veto_reason = "evidence_state_score_drop"

    return {
        "enabled": True,
        "reason": veto_reason or "accepted",
        "vetoed": bool(veto_reason),
        "safety_top_k": safety_top_k,
        "min_score_drop": min_score_drop,
        "score_delta": round(score_delta, 6),
        "chain_role_loss": chain_role_loss,
        "bridge_connected_loss": bridge_loss,
        "planned_title_token_loss": planned_title_loss,
        "old_state": old_state,
        "new_state": new_state,
    }


def apply_bridge_contrastive_evidence_guard(
    question: str,
    docs: List[str],
    evidence_selection: Optional[Dict[str, Any]] = None,
    planned_retrieval: Optional[Dict[str, Any]] = None,
    path_evidence: Optional[Dict[str, Any]] = None,
    enabled: bool = False,
    protect_top_k: int = 1,
    min_bridge_link: float = 0.34,
    min_title_anchor_overlap: float = 0.50,
    enable_counterfactual_safety: bool = True,
    counterfactual_safety_top_k: int = 8,
    counterfactual_safety_min_score_drop: float = 0.35,
) -> Tuple[List[str], Dict[str, Any]]:
    """Demote high-risk answer-type distractors behind bridge-connected evidence.

    This is deliberately non-destructive: it preserves all QA documents but
    demotes documents that look like answer-type matches without a bridge link.
    Non-risky documents keep their original relative order, so this guard cannot
    front-load merely "trusted" evidence and disturb a working multi-hop chain.
    Retrieval rankings and Recall@k are not changed.
    """
    docs = docs or []
    protect_top_k = max(1, int(protect_top_k))
    min_bridge_link = max(0.0, float(min_bridge_link))
    min_title_anchor_overlap = max(0.0, float(min_title_anchor_overlap))
    counterfactual_safety_top_k = max(2, int(counterfactual_safety_top_k))
    counterfactual_safety_min_score_drop = max(
        0.0, float(counterfactual_safety_min_score_drop)
    )
    evidence_selection = evidence_selection or {}
    info = {
        "enabled": bool(enabled),
        "counterfactual_safety_enabled": bool(enable_counterfactual_safety),
        "bridge_cues": has_bridge_cues(question),
        "protect_top_k": protect_top_k,
        "min_bridge_link": min_bridge_link,
        "min_title_anchor_overlap": min_title_anchor_overlap,
        "counterfactual_safety_top_k": counterfactual_safety_top_k,
        "counterfactual_safety_min_score_drop": counterfactual_safety_min_score_drop,
        "reason": "disabled",
        "reordered": False,
        "trusted_count": 0,
        "neutral_count": 0,
        "risky_candidate_count": 0,
        "risky_demoted_count": 0,
        "chain_role_protected_count": 0,
        "chain_role_protect_reason_counts": {},
        "chain_role_protected_titles": [],
        "candidate_demote_reason_counts": {},
        "candidate_demoted_titles": [],
        "demote_reason_counts": {},
        "demoted_titles": [],
        "counterfactual_safety": {
            "enabled": bool(enable_counterfactual_safety),
            "reason": "not_evaluated",
            "vetoed": False,
        },
        "kept_titles": [_doc_title(doc) for doc in docs[:protect_top_k]],
    }
    if not enabled:
        return docs, info
    if len(docs) <= protect_top_k:
        info["reason"] = "not_enough_docs"
        return docs, info
    if not info["bridge_cues"]:
        info["reason"] = "no_bridge_cues"
        return docs, info

    profile = _bridge_anchor_profile(question, docs, protect_top_k)
    question_tokens = profile["question_tokens"]
    type_tokens = profile["type_tokens"]
    anchor_tokens = profile["anchor_tokens"]
    anchor_phrase_sets = profile["anchor_phrase_sets"]
    if not type_tokens or not anchor_tokens:
        info["reason"] = "missing_type_or_anchor_signal"
        return docs, info
    info["anchor_tokens"] = sorted(anchor_tokens)[:24]
    info["type_tokens"] = sorted(type_tokens)[:12]
    if not (type_tokens & BRIDGE_CONTRASTIVE_RISK_TYPE_TERMS):
        info["reason"] = "unsupported_type_signal"
        return docs, info



    trusted_title_tokens = set(profile["protected_title_tokens"])
    feature_cache: List[Dict[str, Any]] = []
    feature_by_idx: Dict[int, Dict[str, Any]] = {}
    for idx, doc in enumerate(docs):
        features = _doc_contrast_features(
            question_tokens=question_tokens,
            type_tokens=type_tokens,
            anchor_tokens=anchor_tokens,
            anchor_phrase_sets=anchor_phrase_sets,
            trusted_title_tokens=trusted_title_tokens,
            doc=doc,
        )
        features["idx"] = idx
        feature_cache.append(features)
        feature_by_idx[idx] = features
        if idx < protect_top_k:
            continue
        if (
            features["phrase_hits"] > 0
            or features["anchor_overlap"] >= min_bridge_link
        ):
            trusted_title_tokens.update(features["non_type_title_tokens"])

    trusted_items = []
    neutral_items = []
    risky_items = []
    chain_role_protected_items = []
    chain_role_protect_reason_counts = Counter()
    demote_reason_counts = Counter()
    selected_extra = set(evidence_selection.get("selected_extra_indices", []) or [])
    chain_context = _chain_role_context(
        question=question,
        planned_retrieval=planned_retrieval,
        path_evidence=path_evidence,
    )
    for idx, doc in enumerate(docs[protect_top_k:], start=protect_top_k):
        features = feature_cache[idx]

        features = _doc_contrast_features(
            question_tokens=question_tokens,
            type_tokens=type_tokens,
            anchor_tokens=anchor_tokens,
            anchor_phrase_sets=anchor_phrase_sets,
            trusted_title_tokens=trusted_title_tokens,
            doc=doc,
        )
        features["idx"] = idx
        feature_by_idx[idx] = features
        trusted = (
            features["phrase_hits"] > 0
            or features["anchor_overlap"] >= min_bridge_link
            or features["title_anchor_overlap"] >= min_title_anchor_overlap
        )
        answer_type_only = (
            features["answer_type_like"]
            and features["query_overlap"] >= 0.18
            and features["bridge_link"] < min_bridge_link
            and features["title_anchor_overlap"] < min_title_anchor_overlap
        )
        geo_distractor = (
            answer_type_only
            and features["title_geo_like"]
            and not trusted
        )
        if trusted:
            trusted_items.append((idx, doc, features))
        elif geo_distractor:
            chain_role_reason = _chain_role_protection_reason(
                title=features["title"],
                features=features,
                chain_context=chain_context,
            )
            if chain_role_reason:
                chain_role_protect_reason_counts[chain_role_reason] += 1
                chain_role_protected_items.append((idx, doc, features, chain_role_reason))
                neutral_items.append((idx, doc, features))
                continue
            reason = "answer_type_only_no_bridge_anchor"
            if idx in selected_extra:
                reason = "extra_answer_type_only_no_bridge_anchor"
            demote_reason_counts[reason] += 1
            risky_items.append((idx, doc, features, reason))
        else:
            neutral_items.append((idx, doc, features))

    if not risky_items:
        info.update({
            "reason": "no_risky_distractor",
            "trusted_count": len(trusted_items),
            "neutral_count": len(neutral_items),
            "risky_candidate_count": 0,
            "risky_demoted_count": 0,
            "chain_role_protected_count": len(chain_role_protected_items),
            "chain_role_protect_reason_counts": dict(chain_role_protect_reason_counts),
            "chain_role_protected_titles": [
                {
                    "rank": idx,
                    "title": features["title"],
                    "reason": reason,
                    "query_overlap": round(features["query_overlap"], 6),
                    "type_overlap": round(features["type_overlap"], 6),
                }
                for idx, _, features, reason in chain_role_protected_items[:12]
            ],
            "demote_reason_counts": {},
            "demoted_titles": [],
            "kept_titles": [_doc_title(doc) for doc in docs[:min(len(docs), 12)]],
        })
        return docs, info

    risky_indices = {idx for idx, _, _, _ in risky_items}
    kept_items = [
        (idx, doc)
        for idx, doc in enumerate(docs)
        if idx not in risky_indices
    ]
    new_order_indices = [idx for idx, _ in kept_items] + [
        idx for idx, _, _, _ in risky_items
    ]
    new_docs = [doc for _, doc in kept_items] + [item[1] for item in risky_items]
    old_order_indices = list(range(len(docs)))
    reordered = new_order_indices != old_order_indices
    new_titles = [_doc_title(doc) for doc in new_docs]
    candidate_demoted_titles = [
        {
            "rank": idx,
            "title": features["title"],
            "reason": reason,
            "anchor_overlap": round(features["anchor_overlap"], 6),
            "title_anchor_overlap": round(features["title_anchor_overlap"], 6),
            "query_overlap": round(features["query_overlap"], 6),
            "type_overlap": round(features["type_overlap"], 6),
        }
        for idx, _, features, reason in risky_items[:12]
    ]
    counterfactual_safety = _bridge_counterfactual_safety_gate(
        old_order_indices=old_order_indices,
        new_order_indices=new_order_indices,
        feature_by_idx=feature_by_idx,
        risky_indices=risky_indices,
        chain_context=chain_context,
        enabled=bool(enable_counterfactual_safety),
        safety_top_k=counterfactual_safety_top_k,
        min_score_drop=counterfactual_safety_min_score_drop,
        min_bridge_link=min_bridge_link,
        min_title_anchor_overlap=min_title_anchor_overlap,
    )

    if counterfactual_safety.get("vetoed"):
        info.update({
            "reason": "counterfactual_safety_veto",
            "reordered": False,
            "trusted_count": len(trusted_items),
            "neutral_count": len(neutral_items),
            "risky_candidate_count": len(risky_items),
            "risky_demoted_count": 0,
            "chain_role_protected_count": len(chain_role_protected_items),
            "chain_role_protect_reason_counts": dict(chain_role_protect_reason_counts),
            "chain_role_protected_titles": [
                {
                    "rank": idx,
                    "title": features["title"],
                    "reason": reason,
                    "query_overlap": round(features["query_overlap"], 6),
                    "type_overlap": round(features["type_overlap"], 6),
                }
                for idx, _, features, reason in chain_role_protected_items[:12]
            ],
            "candidate_demote_reason_counts": dict(demote_reason_counts),
            "candidate_demoted_titles": candidate_demoted_titles,
            "demote_reason_counts": {},
            "demoted_titles": [],
            "counterfactual_safety": counterfactual_safety,
            "kept_titles": [_doc_title(doc) for doc in docs[:min(len(docs), 12)]],
        })
        return docs, info

    info.update({
        "reason": "demoted" if reordered else "risky_already_last",
        "reordered": reordered,
        "trusted_count": len(trusted_items),
        "neutral_count": len(neutral_items),
        "risky_candidate_count": len(risky_items),
        "risky_demoted_count": len(risky_items),
        "chain_role_protected_count": len(chain_role_protected_items),
        "chain_role_protect_reason_counts": dict(chain_role_protect_reason_counts),
        "chain_role_protected_titles": [
            {
                "rank": idx,
                "title": features["title"],
                "reason": reason,
                "query_overlap": round(features["query_overlap"], 6),
                "type_overlap": round(features["type_overlap"], 6),
            }
            for idx, _, features, reason in chain_role_protected_items[:12]
        ],
        "candidate_demote_reason_counts": dict(demote_reason_counts),
        "candidate_demoted_titles": candidate_demoted_titles,
        "demote_reason_counts": dict(demote_reason_counts),
        "demoted_titles": candidate_demoted_titles,
        "counterfactual_safety": counterfactual_safety,
        "kept_titles": new_titles[:min(len(new_titles), 12)],
    })
    return new_docs, info


def has_bridge_cues(question: str) -> bool:
    """Detect questions that likely need more than the first local fact span."""
    q = re.sub(r"\s+", " ", str(question).strip())
    if BRIDGE_CUE_RE.search(q):
        return True


    interrogatives = re.findall(r"\b(?:who|what|where|when|which|how many|how long)\b", q.lower())
    if len(interrogatives) >= 2:
        return True


    if len(re.findall(r"'s\b", q)) >= 2:
        return True

    return False


def _safe_doc_score(doc_scores: Optional[np.ndarray], rank: int) -> float:
    if doc_scores is None:
        return 0.0
    try:
        score = float(doc_scores[rank])
    except Exception:
        return 0.0
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return score


def _normalize_scores(values: List[float]) -> List[float]:
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v <= min_v:
        return [1.0 for _ in values]
    return [(value - min_v) / (max_v - min_v) for value in values]


def select_adaptive_evidence_passages(
    question: str,
    docs: List[str],
    doc_scores: Optional[np.ndarray] = None,
    qa_top_k: int = 5,
    enabled: bool = False,
    candidate_k: int = 20,
    target_k: int = 8,
    bridge_only: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    """Select a compact evidence pack for QA.

    The selector keeps the original top-k passages first, then appends a small
    number of high-signal bridge candidates from the wider retrieval pool. It is
    intentionally deterministic and does not use gold labels or dataset names.
    """
    docs = docs or []
    base_k = max(1, qa_top_k)
    final_target_k = max(base_k, target_k)
    candidate_k = max(final_target_k, candidate_k)

    base_docs = docs[:base_k]
    info = {
        "enabled": bool(enabled),
        "bridge_cues": has_bridge_cues(question),
        "candidate_k": candidate_k,
        "base_qa_top_k": base_k,
        "target_k": final_target_k,
        "selected_indices": list(range(len(base_docs))),
        "selected_extra_indices": [],
        "selected_titles": [_doc_title(doc) for doc in base_docs],
        "reason": "disabled",
    }

    if not enabled:
        return base_docs, info
    if bridge_only and not info["bridge_cues"]:
        info["reason"] = "no_bridge_cues"
        return base_docs, info
    if len(docs) <= base_k or final_target_k <= base_k:
        info["reason"] = "no_extra_candidates"
        return base_docs, info

    question_tokens = _token_set(question)
    base_title_tokens = set()
    base_body_tokens = set()
    for doc in base_docs:
        base_title_tokens.update(_token_set(_doc_title(doc)))
        base_body_tokens.update(_token_set(_doc_body(doc)[:2000]))

    candidates = []
    raw_doc_scores = [_safe_doc_score(doc_scores, rank) for rank in range(min(candidate_k, len(docs)))]
    norm_doc_scores = _normalize_scores(raw_doc_scores)
    seen_titles = {_doc_title(doc).lower() for doc in base_docs}

    for rank, doc in enumerate(docs[:candidate_k]):
        if rank < base_k:
            continue

        title = _doc_title(doc)
        lowered_title = title.lower()
        if lowered_title in seen_titles:
            continue

        title_tokens = _token_set(title)
        body_tokens = _token_set(_doc_body(doc)[:2500])
        all_tokens = title_tokens | body_tokens

        query_overlap = len(question_tokens & all_tokens) / max(1, len(question_tokens))
        title_query_overlap = len(question_tokens & title_tokens) / max(1, len(title_tokens))
        bridge_overlap = len((base_title_tokens | question_tokens) & all_tokens) / max(1, len(base_title_tokens | question_tokens))
        body_to_base_overlap = len(base_body_tokens & title_tokens) / max(1, len(title_tokens))
        rank_score = 1.0 / (rank + 1)
        retrieval_score = norm_doc_scores[rank] if rank < len(norm_doc_scores) else 0.0

        score = (
            0.36 * query_overlap
            + 0.20 * title_query_overlap
            + 0.18 * bridge_overlap
            + 0.12 * body_to_base_overlap
            + 0.10 * retrieval_score
            + 0.04 * rank_score
        )
        candidates.append({
            "rank": rank,
            "doc": doc,
            "title": title,
            "score": score,
            "query_overlap": query_overlap,
            "title_query_overlap": title_query_overlap,
            "bridge_overlap": bridge_overlap,
            "body_to_base_overlap": body_to_base_overlap,
            "retrieval_score": retrieval_score,
        })

    candidates.sort(key=lambda item: (-item["score"], item["rank"]))
    extra_slots = final_target_k - len(base_docs)
    selected_extra = candidates[:extra_slots]
    selected_docs = base_docs + [item["doc"] for item in selected_extra]

    info.update({
        "selected_indices": list(range(len(base_docs))) + [item["rank"] for item in selected_extra],
        "selected_extra_indices": [item["rank"] for item in selected_extra],
        "selected_titles": [_doc_title(doc) for doc in selected_docs],
        "extra_candidates": [
            {
                "rank": item["rank"],
                "title": item["title"],
                "score": round(item["score"], 6),
                "query_overlap": round(item["query_overlap"], 6),
                "title_query_overlap": round(item["title_query_overlap"], 6),
                "bridge_overlap": round(item["bridge_overlap"], 6),
                "body_to_base_overlap": round(item["body_to_base_overlap"], 6),
                "retrieval_score": round(item["retrieval_score"], 6),
            }
            for item in selected_extra
        ],
        "reason": "expanded" if selected_extra else "no_useful_extra_candidates",
    })
    return selected_docs, info


def _list_from_array(values: Any) -> List[Any]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        return list(values.tolist())
    return list(values)


def _current_evidence_state(
    question: str,
    docs: List[str],
    qa_top_k: int,
) -> Dict[str, Any]:
    top_docs = (docs or [])[:max(1, int(qa_top_k))]
    question_tokens = _token_set(question)
    anchor_tokens = {
        token
        for token in question_tokens
        if (
            token not in ANSWER_TYPE_TERMS
            and token not in GENERIC_BRIDGE_TERMS
            and token not in PLACE_SUFFIX_TERMS
        )
    }
    top_tokens = set()
    bridge_connected_docs = 0
    title_tokens_all = set()
    for doc in top_docs:
        title_tokens = _token_set(_doc_title(doc))
        body_tokens = _token_set(_doc_body(doc)[:2600])
        doc_tokens = title_tokens | body_tokens
        title_tokens_all.update(title_tokens)
        top_tokens.update(doc_tokens)
        anchor_hit = _overlap_ratio(anchor_tokens, doc_tokens)
        title_hit = _overlap_ratio(title_tokens, question_tokens)
        if anchor_hit >= 0.20 or title_hit >= 0.20:
            bridge_connected_docs += 1

    query_coverage = _overlap_ratio(question_tokens, top_tokens)
    anchor_coverage = _overlap_ratio(anchor_tokens, top_tokens)
    title_query_coverage = _overlap_ratio(question_tokens, title_tokens_all)
    return {
        "query_coverage": round(query_coverage, 6),
        "anchor_coverage": round(anchor_coverage, 6),
        "title_query_coverage": round(title_query_coverage, 6),
        "bridge_connected_docs": bridge_connected_docs,
        "question_token_count": len(question_tokens),
        "anchor_token_count": len(anchor_tokens),
        "missing_query_tokens": sorted(question_tokens - top_tokens)[:24],
        "missing_anchor_tokens": sorted(anchor_tokens - top_tokens)[:24],
        "top_titles": [_doc_title(doc) for doc in top_docs],
    }


def apply_dual_channel_retrieval_projection(
    question: str,
    base_doc_ids: Any,
    base_doc_scores: Any,
    base_docs: List[str],
    candidate_doc_ids: Any,
    candidate_doc_scores: Any,
    candidate_docs: List[str],
    qa_top_k: int = 5,
    enabled: bool = False,
    bridge_only: bool = True,
    candidate_top_k: int = 80,
    max_promotions: int = 1,
    protect_top_k: int = 1,
    min_projection_score: float = 0.66,
    min_query_overlap: float = 0.18,
    min_missing_gain: float = 0.10,
    min_bridge_link: float = 0.06,
    shadow_top_k: int = 8,
    shadow_when_sufficient: bool = True,
    projection_mode: str = "append_only_or_top5",
    trigger_min_query_coverage: float = 0.50,
    trigger_min_bridge_docs: int = 2,
    reader_soft_append_enabled: bool = False,
    reader_soft_append_require_insufficient: bool = True,
    reader_soft_append_min_score: float = 0.42,
    reader_soft_append_min_query_overlap: float = 0.06,
    reader_soft_append_min_missing_gain: float = 0.05,
    reader_soft_append_min_bridge_link: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Promote one high-recall dense candidate only when the graph channel is weak.

    This creates a retrieval-side high-recall channel without blindly replacing
    the graph ranking. It protects the leading evidence prefix and requires a
    candidate to agree with query anchors and the current evidence state.
    """
    base_ids = _list_from_array(base_doc_ids)
    base_scores = _list_from_array(base_doc_scores)
    candidate_ids = _list_from_array(candidate_doc_ids)
    candidate_scores = _list_from_array(candidate_doc_scores)
    qa_top_k = max(1, int(qa_top_k))
    candidate_top_k = max(qa_top_k, int(candidate_top_k))
    max_promotions = max(0, int(max_promotions))
    protect_top_k = max(0, int(protect_top_k))
    min_projection_score = max(0.0, float(min_projection_score))
    min_query_overlap = max(0.0, float(min_query_overlap))
    min_missing_gain = max(0.0, float(min_missing_gain))
    min_bridge_link = max(0.0, float(min_bridge_link))
    shadow_top_k = max(0, int(shadow_top_k))
    projection_mode = str(projection_mode or "append_only_or_top5").strip().lower()
    if projection_mode not in {"aggressive_promote", "append_only_or_top5", "append_only"}:
        projection_mode = "append_only_or_top5"
    trigger_min_query_coverage = max(0.0, float(trigger_min_query_coverage))
    trigger_min_bridge_docs = max(0, int(trigger_min_bridge_docs))
    reader_soft_append_min_score = max(0.0, float(reader_soft_append_min_score))
    reader_soft_append_min_query_overlap = max(0.0, float(reader_soft_append_min_query_overlap))
    reader_soft_append_min_missing_gain = max(0.0, float(reader_soft_append_min_missing_gain))
    reader_soft_append_min_bridge_link = max(0.0, float(reader_soft_append_min_bridge_link))

    current_state = _current_evidence_state(
        question=question,
        docs=base_docs,
        qa_top_k=qa_top_k,
    )
    top_docs_for_shape = base_docs[:qa_top_k]
    avg_top_doc_chars = (
        sum(len(_doc_body(doc)) for doc in top_docs_for_shape) / max(1, len(top_docs_for_shape))
        if top_docs_for_shape
        else 0.0
    )
    empty_title_ratio = (
        sum(1 for doc in top_docs_for_shape if not _doc_title(doc)) / max(1, len(top_docs_for_shape))
        if top_docs_for_shape
        else 0.0
    )
    long_context_shape = bool(avg_top_doc_chars >= 1200.0 and empty_title_ratio >= 0.50)
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "bridge_cues": has_bridge_cues(question),
        "candidate_top_k": candidate_top_k,
        "max_promotions": max_promotions,
        "protect_top_k": protect_top_k,
        "min_projection_score": min_projection_score,
        "min_query_overlap": min_query_overlap,
        "min_missing_gain": min_missing_gain,
        "min_bridge_link": min_bridge_link,
        "shadow_top_k": shadow_top_k,
        "shadow_when_sufficient": bool(shadow_when_sufficient),
        "projection_mode": projection_mode,
        "trigger_min_query_coverage": trigger_min_query_coverage,
        "trigger_min_bridge_docs": trigger_min_bridge_docs,
        "reader_soft_append_enabled": bool(reader_soft_append_enabled),
        "reader_soft_append_require_insufficient": bool(reader_soft_append_require_insufficient),
        "reader_soft_append_min_score": reader_soft_append_min_score,
        "reader_soft_append_min_query_overlap": reader_soft_append_min_query_overlap,
        "reader_soft_append_min_missing_gain": reader_soft_append_min_missing_gain,
        "reader_soft_append_min_bridge_link": reader_soft_append_min_bridge_link,
        "current_state": current_state,
        "trigger_reason": "not_evaluated",
        "retrieval_applied": False,
        "reader_candidate_count": 0,
        "reader_candidate_doc_ids": [],
        "reader_candidate_titles": [],
        "reader_soft_append_candidate_count": 0,
        "reader_soft_append_rejected_count": 0,
        "reader_soft_append_reject_reason_counts": {},
        "reader_soft_append_long_context_blocked": long_context_shape,
        "avg_top_doc_chars": round(avg_top_doc_chars, 3),
        "empty_title_ratio": round(empty_title_ratio, 6),
        "shadow_candidate_count": 0,
        "candidate_count": 0,
        "shadow_rejected_count": 0,
        "rejection_reason_counts": {},
        "shadow_top_candidates": [],
        "promoted_to_top2": 0,
        "promoted_to_top5": 0,
        "promotions": [],
    }

    if not enabled:
        return np.array(base_ids), np.array(base_scores), info
    if bridge_only and not info["bridge_cues"]:
        info["reason"] = "no_bridge_cues"
        return np.array(base_ids), np.array(base_scores), info
    if max_promotions <= 0:
        info["reason"] = "no_promotion_budget"
        return np.array(base_ids), np.array(base_scores), info
    if not base_ids or not candidate_ids or not base_docs or not candidate_docs:
        info["reason"] = "no_candidates"
        return np.array(base_ids), np.array(base_scores), info

    current_query_coverage = float(current_state.get("query_coverage", 0.0) or 0.0)
    current_bridge_docs = int(current_state.get("bridge_connected_docs", 0) or 0)
    if (
        current_query_coverage >= trigger_min_query_coverage
        and current_bridge_docs >= trigger_min_bridge_docs
    ):
        info["trigger_reason"] = "current_evidence_sufficient"
        if not shadow_when_sufficient:
            info["reason"] = "current_evidence_sufficient"
            return np.array(base_ids), np.array(base_scores), info
    else:
        info["trigger_reason"] = (
            "low_query_coverage"
            if current_query_coverage < trigger_min_query_coverage
            else "low_bridge_connected_docs"
        )

    question_tokens = _token_set(question)
    anchor_tokens = {
        token
        for token in question_tokens
        if (
            token not in ANSWER_TYPE_TERMS
            and token not in GENERIC_BRIDGE_TERMS
            and token not in PLACE_SUFFIX_TERMS
        )
    }
    phrase_sets = _capitalized_phrase_token_sets(question)
    for tokens in phrase_sets:
        anchor_tokens.update(tokens)
    protected_ids = set(base_ids[:protect_top_k])
    base_top_ids = set(base_ids[:qa_top_k])
    base_rank = {doc_id: rank for rank, doc_id in enumerate(base_ids)}
    base_score = {
        doc_id: _safe_doc_score(np.array(base_scores), rank)
        for rank, doc_id in enumerate(base_ids)
    }
    raw_candidate_scores = [
        _safe_doc_score(np.array(candidate_scores), rank)
        for rank in range(min(candidate_top_k, len(candidate_ids)))
    ]
    normalized_candidate_scores = _normalize_scores(raw_candidate_scores)

    base_title_tokens = set()
    base_body_tokens = set()
    base_top_tokens = set()
    seen_titles = set()
    seen_doc_surfaces = set()
    for doc in base_docs[:qa_top_k]:
        title = _doc_title(doc).lower()
        if title:
            seen_titles.add(title)
        title_tokens = _token_set(_doc_title(doc))
        body_tokens = _token_set(_doc_body(doc)[:2200])
        base_title_tokens.update(title_tokens)
        base_body_tokens.update(body_tokens)
        base_top_tokens.update(title_tokens | body_tokens)
        surface = _normalized_surface(doc)
        if surface:
            seen_doc_surfaces.add(surface[:500])

    missing_anchor_tokens = anchor_tokens - base_top_tokens
    missing_query_tokens = question_tokens - base_top_tokens
    target_missing_tokens = (
        missing_anchor_tokens
        or missing_query_tokens
        or anchor_tokens
        or question_tokens
    )
    question_lower = str(question or "").lower()
    requires_date_value = bool(
        re.search(r"\b(?:when|date|year|month|day|born|died|released|published|founded)\b", question_lower)
    )
    requires_number_value = bool(
        re.search(r"\b(?:how many|how much|number|population|total|count|age|percent|percentage)\b", question_lower)
    )
    weak_current = (
        current_query_coverage < trigger_min_query_coverage
        or current_bridge_docs < trigger_min_bridge_docs
    )

    all_candidates = []
    eligible_candidates = []
    rejection_reason_counts = Counter()
    top_n = min(candidate_top_k, len(candidate_ids), len(candidate_docs))
    for dense_rank in range(top_n):
        doc_id = candidate_ids[dense_rank]
        if doc_id in protected_ids:
            rejection_reason_counts["protected_doc"] += 1
            continue
        doc = candidate_docs[dense_rank]
        title = _doc_title(doc)
        title_key = title.lower()
        surface = _normalized_surface(doc)
        if doc_id in base_top_ids:
            rejection_reason_counts["already_in_topk"] += 1
            continue
        if title_key and title_key in seen_titles:
            rejection_reason_counts["duplicate_title"] += 1
            continue
        if surface and surface[:500] in seen_doc_surfaces:
            rejection_reason_counts["duplicate_surface"] += 1
            continue
        title_tokens = _token_set(title)
        body_text = _doc_body(doc)[:3200]
        body_tokens = _token_set(body_text)
        all_tokens = title_tokens | body_tokens
        if not all_tokens:
            rejection_reason_counts["empty_candidate_tokens"] += 1
            continue
        candidate_surface = f"{title}\n{body_text}"
        answer_type_value_signal = 1.0
        if requires_date_value and not DATE_VALUE_RE.search(candidate_surface):
            answer_type_value_signal = 0.0
        if requires_number_value and not NUMBER_VALUE_RE.search(candidate_surface):
            answer_type_value_signal = 0.0

        dense_norm = (
            normalized_candidate_scores[dense_rank]
            if dense_rank < len(normalized_candidate_scores)
            else 0.0
        )
        query_overlap = _overlap_ratio(question_tokens, all_tokens)
        anchor_overlap = _overlap_ratio(anchor_tokens, all_tokens)
        title_query_overlap = _overlap_ratio(question_tokens, title_tokens)
        base_bridge_overlap = _overlap_ratio(base_title_tokens | anchor_tokens, all_tokens)
        missing_anchor_gain = _overlap_ratio(missing_anchor_tokens, all_tokens)
        missing_query_gain = _overlap_ratio(missing_query_tokens, all_tokens)
        missing_slot_gain = _overlap_ratio(target_missing_tokens, all_tokens)
        context_complement = max(
            _overlap_ratio(title_tokens | anchor_tokens, base_body_tokens),
            _overlap_ratio(base_title_tokens, all_tokens),
            _overlap_ratio(title_tokens, base_body_tokens | question_tokens),
        )
        bridge_link = max(
            context_complement,
            base_bridge_overlap,
            _overlap_ratio(base_top_tokens & anchor_tokens, all_tokens),
        )
        rank_prior = 1.0 / (dense_rank + 1)
        score = (
            0.24 * dense_norm
            + 0.30 * missing_slot_gain
            + 0.16 * anchor_overlap
            + 0.12 * bridge_link
            + 0.10 * query_overlap
            + 0.06 * title_query_overlap
            + 0.02 * rank_prior
        )
        if weak_current:
            score += 0.03

        passes_missing = (
            missing_slot_gain >= min_missing_gain
            or (weak_current and anchor_overlap >= min_query_overlap)
        )
        passes_bridge = (
            bridge_link >= min_bridge_link
            or (anchor_overlap >= max(min_query_overlap, 0.20) and query_overlap >= 0.08)
        )
        passes_score = score >= min_projection_score
        reject_reason = ""
        if not passes_missing:
            reject_reason = "low_missing_slot_gain"
        elif not passes_bridge:
            reject_reason = "low_bridge_link"
        elif not passes_score:
            reject_reason = "low_shadow_score"

        candidate = {
            "doc_id": doc_id,
            "dense_rank": dense_rank,
            "original_rank": base_rank.get(doc_id),
            "title": title or _doc_body(doc)[:80].replace("\n", " "),
            "surface": candidate_surface,
            "score": score,
            "dense_norm": dense_norm,
            "query_overlap": query_overlap,
            "anchor_overlap": anchor_overlap,
            "base_bridge_overlap": base_bridge_overlap,
            "bridge_link": bridge_link,
            "missing_slot_gain": missing_slot_gain,
            "missing_anchor_gain": missing_anchor_gain,
            "missing_query_gain": missing_query_gain,
            "title_query_overlap": title_query_overlap,
            "answer_type_value_signal": answer_type_value_signal,
            "accepted": not reject_reason,
            "reject_reason": reject_reason,
        }
        all_candidates.append(candidate)
        if reject_reason:
            rejection_reason_counts[reject_reason] += 1
        else:
            eligible_candidates.append(candidate)

    all_candidates.sort(
        key=lambda item: (
            -item["score"],
            -item["missing_slot_gain"],
            -item["bridge_link"],
            item["original_rank"] if item["original_rank"] is not None else 10**9,
            item["dense_rank"],
        )
    )
    eligible_candidates.sort(
        key=lambda item: (
            -item["score"],
            -item["missing_slot_gain"],
            -item["bridge_link"],
            item["original_rank"] if item["original_rank"] is not None else 10**9,
            item["dense_rank"],
        )
    )

    soft_append_candidates = []
    soft_append_rejection_counts = Counter()
    if reader_soft_append_enabled:
        for item in all_candidates:
            title_text = str(item.get("title", "") or "")
            doc_surface = str(item.get("surface", "") or "")
            if long_context_shape:
                soft_append_rejection_counts["long_context_soft_append_blocked"] += 1
                continue
            if (
                title_text.strip().lower() == "typed evidence grounding"
                or doc_surface.lstrip().lower().startswith("typed evidence grounding")
            ):
                soft_append_rejection_counts["synthetic_evidence_blocked"] += 1
                continue
            if reader_soft_append_require_insufficient and not weak_current:
                soft_append_rejection_counts["current_evidence_sufficient"] += 1
                continue
            overlap_signal = max(item["query_overlap"], item["anchor_overlap"])
            missing_signal = max(
                item["missing_slot_gain"],
                item["missing_anchor_gain"],
                item["missing_query_gain"],
            )
            bridge_signal = item["bridge_link"]
            if item["score"] < reader_soft_append_min_score:
                soft_append_rejection_counts["low_soft_score"] += 1
                continue
            if overlap_signal < reader_soft_append_min_query_overlap:
                soft_append_rejection_counts["low_soft_query_overlap"] += 1
                continue
            if missing_signal < reader_soft_append_min_missing_gain:
                soft_append_rejection_counts["low_soft_missing_gain"] += 1
                continue
            if bridge_signal < reader_soft_append_min_bridge_link:
                soft_append_rejection_counts["low_soft_bridge_link"] += 1
                continue
            if float(item.get("answer_type_value_signal", 1.0) or 0.0) <= 0.0:
                soft_append_rejection_counts["missing_required_answer_type_value"] += 1
                continue
            soft_item = dict(item)
            soft_item["soft_append_reason"] = "soft_shadow_bridge_deficit"
            soft_item["soft_append_overlap_signal"] = overlap_signal
            soft_item["soft_append_missing_signal"] = missing_signal
            soft_item["soft_append_bridge_signal"] = bridge_signal
            soft_append_candidates.append(soft_item)
        soft_append_candidates.sort(
            key=lambda item: (
                -item["score"],
                -item["soft_append_missing_signal"],
                -item["soft_append_bridge_signal"],
                -item["soft_append_overlap_signal"],
                item["original_rank"] if item["original_rank"] is not None else 10**9,
                item["dense_rank"],
            )
        )

    info["shadow_candidate_count"] = len(all_candidates)
    info["candidate_count"] = len(eligible_candidates)
    info["shadow_rejected_count"] = max(0, len(all_candidates) - len(eligible_candidates))
    info["rejection_reason_counts"] = dict(rejection_reason_counts)
    info["reader_soft_append_candidate_count"] = len(soft_append_candidates)
    info["reader_soft_append_rejected_count"] = sum(soft_append_rejection_counts.values())
    info["reader_soft_append_reject_reason_counts"] = dict(soft_append_rejection_counts)
    info["shadow_top_candidates"] = [
        {
            "doc_id": int(item["doc_id"]) if isinstance(item["doc_id"], (int, np.integer)) else item["doc_id"],
            "title": item["title"],
            "dense_rank": int(item["dense_rank"]),
            "original_rank": None if item["original_rank"] is None else int(item["original_rank"]),
            "score": round(item["score"], 6),
            "missing_slot_gain": round(item["missing_slot_gain"], 6),
            "missing_anchor_gain": round(item["missing_anchor_gain"], 6),
            "missing_query_gain": round(item["missing_query_gain"], 6),
            "bridge_link": round(item["bridge_link"], 6),
            "query_overlap": round(item["query_overlap"], 6),
            "anchor_overlap": round(item["anchor_overlap"], 6),
            "accepted": bool(item["accepted"]),
            "reject_reason": item["reject_reason"],
        }
        for item in all_candidates[:shadow_top_k]
    ]
    promotions = eligible_candidates[:max_promotions]
    if not promotions:
        soft_promotions = soft_append_candidates[:max_promotions]
        if soft_promotions:
            promoted_ids = [item["doc_id"] for item in soft_promotions]
            info["reader_candidate_count"] = len(soft_promotions)
            info["reader_candidate_doc_ids"] = [
                int(doc_id) if isinstance(doc_id, (int, np.integer)) else doc_id
                for doc_id in promoted_ids
            ]
            info["reader_candidate_titles"] = [item["title"] for item in soft_promotions]
            info.update({
                "applied": True,
                "retrieval_applied": False,
                "reason": "shadow_reader_soft_append_only",
                "promotions": [
                    {
                        "doc_id": (
                            int(item["doc_id"])
                            if isinstance(item["doc_id"], (int, np.integer))
                            else item["doc_id"]
                        ),
                        "title": item["title"],
                        "dense_rank": int(item["dense_rank"]),
                        "original_rank": (
                            None
                            if item["original_rank"] is None
                            else int(item["original_rank"])
                        ),
                        "new_rank": None,
                        "action": "reader_soft_append_only",
                        "score": round(item["score"], 6),
                        "dense_norm": round(item["dense_norm"], 6),
                        "missing_slot_gain": round(item["missing_slot_gain"], 6),
                        "missing_anchor_gain": round(item["missing_anchor_gain"], 6),
                        "missing_query_gain": round(item["missing_query_gain"], 6),
                        "bridge_link": round(item["bridge_link"], 6),
                        "query_overlap": round(item["query_overlap"], 6),
                        "anchor_overlap": round(item["anchor_overlap"], 6),
                        "answer_type_value_signal": round(
                            float(item.get("answer_type_value_signal", 1.0) or 0.0),
                            6,
                        ),
                        "soft_append_missing_signal": round(
                            item["soft_append_missing_signal"], 6
                        ),
                        "soft_append_bridge_signal": round(
                            item["soft_append_bridge_signal"], 6
                        ),
                        "soft_append_overlap_signal": round(
                            item["soft_append_overlap_signal"], 6
                        ),
                        "soft_append_reason": item["soft_append_reason"],
                    }
                    for item in soft_promotions
                ],
            })
            return np.array(base_ids), np.array(base_scores), info
        info["reason"] = (
            "shadow_candidates_rejected"
            if all_candidates
            else "no_shadow_candidates"
        )
        return np.array(base_ids), np.array(base_scores), info

    promoted_ids = [item["doc_id"] for item in promotions]
    info["reader_candidate_count"] = len(promotions)
    info["reader_candidate_doc_ids"] = [
        int(doc_id) if isinstance(doc_id, (int, np.integer)) else doc_id
        for doc_id in promoted_ids
    ]
    info["reader_candidate_titles"] = [item["title"] for item in promotions]

    append_only = projection_mode == "append_only" or (
        projection_mode == "append_only_or_top5"
        and info.get("trigger_reason") == "current_evidence_sufficient"
    )

    if append_only:
        promotion_logs = []
        for item in promotions:
            original_rank = item["original_rank"]
            promotion_logs.append({
                "doc_id": int(item["doc_id"]) if isinstance(item["doc_id"], (int, np.integer)) else item["doc_id"],
                "title": item["title"],
                "dense_rank": int(item["dense_rank"]),
                "original_rank": None if original_rank is None else int(original_rank),
                "new_rank": None,
                "action": "reader_append_only",
                "score": round(item["score"], 6),
                "dense_norm": round(item["dense_norm"], 6),
                "missing_slot_gain": round(item["missing_slot_gain"], 6),
                "missing_anchor_gain": round(item["missing_anchor_gain"], 6),
                "missing_query_gain": round(item["missing_query_gain"], 6),
                "bridge_link": round(item["bridge_link"], 6),
                "query_overlap": round(item["query_overlap"], 6),
                "anchor_overlap": round(item["anchor_overlap"], 6),
                "base_bridge_overlap": round(item["base_bridge_overlap"], 6),
                "title_query_overlap": round(item["title_query_overlap"], 6),
            })
        info.update({
            "applied": True,
            "retrieval_applied": False,
            "reason": "shadow_reader_append_only",
            "promotions": promotion_logs,
        })
        return np.array(base_ids), np.array(base_scores), info

    promoted_set = set(promoted_ids)
    insertion_prefix = protect_top_k
    if projection_mode == "append_only_or_top5":
        insertion_prefix = max(insertion_prefix, 2)
    insertion_prefix = min(insertion_prefix, len(base_ids))
    projected_ids = base_ids[:insertion_prefix] + promoted_ids + [
        doc_id for doc_id in base_ids[insertion_prefix:] if doc_id not in promoted_set
    ]

    candidate_score_by_id = {
        item["doc_id"]: max(
            item["score"],
            _safe_doc_score(np.array(candidate_scores), item["dense_rank"]),
            base_score.get(item["doc_id"], 0.0),
        )
        for item in promotions
    }
    projected_scores = [
        candidate_score_by_id.get(doc_id, base_score.get(doc_id, 0.0))
        for doc_id in projected_ids
    ]

    promotion_logs = []
    for item in promotions:
        new_rank = projected_ids.index(item["doc_id"])
        original_rank = item["original_rank"]
        if new_rank < 2 and (original_rank is None or original_rank >= 2):
            info["promoted_to_top2"] += 1
        if new_rank < 5 and (original_rank is None or original_rank >= 5):
            info["promoted_to_top5"] += 1
        promotion_logs.append({
            "doc_id": int(item["doc_id"]) if isinstance(item["doc_id"], (int, np.integer)) else item["doc_id"],
            "title": item["title"],
            "dense_rank": int(item["dense_rank"]),
            "original_rank": None if original_rank is None else int(original_rank),
            "new_rank": int(new_rank),
            "score": round(item["score"], 6),
            "dense_norm": round(item["dense_norm"], 6),
            "missing_slot_gain": round(item["missing_slot_gain"], 6),
            "missing_anchor_gain": round(item["missing_anchor_gain"], 6),
            "missing_query_gain": round(item["missing_query_gain"], 6),
            "bridge_link": round(item["bridge_link"], 6),
            "query_overlap": round(item["query_overlap"], 6),
            "anchor_overlap": round(item["anchor_overlap"], 6),
            "base_bridge_overlap": round(item["base_bridge_overlap"], 6),
            "title_query_overlap": round(item["title_query_overlap"], 6),
        })

    info.update({
        "applied": True,
        "retrieval_applied": True,
        "reason": (
            "projected_dense_candidate_top5"
            if projection_mode == "append_only_or_top5"
            else "projected_dense_candidate"
        ),
        "promotions": promotion_logs,
    })
    return np.array(projected_ids), np.array(projected_scores), info


def assess_dual_channel_projection_trigger(
    question: str,
    docs: List[str],
    qa_top_k: int = 5,
    enabled: bool = False,
    bridge_only: bool = True,
    shadow_when_sufficient: bool = True,
    trigger_min_query_coverage: float = 0.50,
    trigger_min_bridge_docs: int = 2,
) -> Dict[str, Any]:
    current_state = _current_evidence_state(
        question=question,
        docs=docs or [],
        qa_top_k=qa_top_k,
    )
    bridge_cues = has_bridge_cues(question)
    info = {
        "enabled": bool(enabled),
        "should_project": False,
        "reason": "disabled",
        "bridge_cues": bridge_cues,
        "shadow_when_sufficient": bool(shadow_when_sufficient),
        "trigger_min_query_coverage": trigger_min_query_coverage,
        "trigger_min_bridge_docs": trigger_min_bridge_docs,
        "current_state": current_state,
    }
    if not enabled:
        return info
    if bridge_only and not bridge_cues:
        info["reason"] = "no_bridge_cues"
        return info
    current_query_coverage = float(current_state.get("query_coverage", 0.0) or 0.0)
    current_bridge_docs = int(current_state.get("bridge_connected_docs", 0) or 0)
    if (
        current_query_coverage >= max(0.0, float(trigger_min_query_coverage))
        and current_bridge_docs >= max(0, int(trigger_min_bridge_docs))
    ):
        info["reason"] = "current_evidence_sufficient"
        info["should_project"] = bool(shadow_when_sufficient)
        return info
    info["should_project"] = True
    if current_query_coverage < max(0.0, float(trigger_min_query_coverage)):
        info["reason"] = "low_query_coverage"
    else:
        info["reason"] = "low_bridge_connected_docs"
    return info


def apply_dual_channel_reader_projection(
    docs: List[str],
    projection_docs: List[str],
    retrieval_projection: Optional[Dict[str, Any]] = None,
    enabled: bool = False,
    max_append: int = 1,
) -> Tuple[List[str], Dict[str, Any]]:
    """Append accepted high-recall channel docs to QA context without reordering."""
    docs = docs or []
    projection_docs = projection_docs or []
    max_append = max(0, int(max_append))
    retrieval_projection = retrieval_projection or {}
    info = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "max_append": max_append,
        "retrieval_reason": retrieval_projection.get("reason", "unavailable"),
        "retrieval_applied": bool(retrieval_projection.get("applied", False)),
        "appended_count": 0,
        "appended_titles": [],
    }
    if not enabled:
        return docs, info
    if max_append <= 0:
        info["reason"] = "no_append_budget"
        return docs, info
    if not projection_docs or not retrieval_projection.get("applied"):
        info["reason"] = "no_projected_retrieval_docs"
        return docs, info

    seen = {str(doc).strip() for doc in docs}
    out_docs = list(docs)
    appended = []
    for doc in projection_docs:
        key = str(doc).strip()
        if not key or key in seen:
            continue
        out_docs.append(doc)
        appended.append(doc)
        seen.add(key)
        if len(appended) >= max_append:
            break

    info.update({
        "applied": bool(appended),
        "reason": "projected_to_reader" if appended else "already_visible",
        "appended_count": len(appended),
        "appended_titles": [_doc_title(doc) for doc in appended],
    })
    return out_docs, info
