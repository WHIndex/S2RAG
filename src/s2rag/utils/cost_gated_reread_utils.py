import json
import re
from typing import Any, Callable, Dict, List, Sequence, Tuple

from .answer_verifier_utils import _is_supported
from .eval_utils import normalize_answer
from .evidence_grounded_answer_extractor_utils import (
    _candidate_passes_answer_slot,
    _infer_answer_slot,
)
from .evidence_object_arbitration_utils import _candidate_passes_role_constraint
from .planned_retrieval_utils import unpack_llm_result
from .sentence_evidence_utils import infer_answer_target_type


ABSTAIN_RE = re.compile(
    r"^(?:unknown|none|n/?a|not (?:known|provided|specified|mentioned)|"
    r"cannot (?:determine|be determined)|insufficient (?:evidence|information))$",
    re.IGNORECASE,
)
MONTH_NUMBER = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
MONTH_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)


def _new_meta(enabled: bool, answer: str) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "triggered": False,
        "changed": False,
        "action": "none" if enabled else "disabled",
        "reason": "not_evaluated" if enabled else "disabled",
        "before": str(answer or "").strip(),
        "after": str(answer or "").strip(),
        "estimated_utility": 0.0,
        "estimated_risk": 0.0,
        "estimated_cost": 0.0,
        "net_utility": 0.0,
    }


def _doc_title(doc: str) -> str:
    text = str(doc or "")
    return text.split("\n", 1)[0].strip()


def _compact_docs(
    docs: Sequence[str],
    max_docs: int,
    max_doc_chars: int,
    max_total_chars: int,
) -> Tuple[List[str], List[str]]:
    compact = []
    rendered = []
    used_chars = 0
    for doc in list(docs or [])[: max(1, int(max_docs))]:
        clean = re.sub(r"\s+", " ", str(doc or "")).strip()
        if not clean:
            continue
        remaining = max(0, int(max_total_chars) - used_chars)
        if remaining <= 0:
            break
        clean = clean[: min(int(max_doc_chars), remaining)]
        compact.append(clean)
        rendered.append(f"[E{len(compact)}] {clean}")
        used_chars += len(clean)
    return compact, rendered


def _extract_json_object(text: Any) -> Tuple[Dict[str, Any], str]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    candidates = [(raw, "strict_json")]
    start = raw.find("{")
    end = raw.rfind("}")
    if 0 <= start < end:
        candidates.append((raw[start : end + 1], "extracted_json"))
    repaired = []
    for candidate, mode in candidates:
        fixed = re.sub(
            r'("evidence_ids"\s*:\s*\[)([^\]]*)(\])',
            lambda match: match.group(1)
            + re.sub(r'(?<!["\w])E(\d+)(?!["\w])', r'"E\1"', match.group(2))
            + match.group(3),
            candidate,
            flags=re.IGNORECASE,
        )
        if fixed != candidate:
            repaired.append((fixed, "repaired_bare_evidence_ids"))
    candidates.extend(repaired)
    for candidate, mode in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return parsed, mode
    return {}, "invalid"


def _date_keys(text: str) -> set:
    normalized = re.sub(r"[–—−]", "-", str(text or "").lower())
    keys = set()

    def add(year: str, month: str, day: str) -> None:
        month_number = MONTH_NUMBER.get(str(month or "")[:3])
        if year and month_number and day:
            keys.add((int(year), month_number, int(day)))

    day_first = re.compile(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*-\s*(\d{{1,2}})(?:st|nd|rd|th)?)?\s+"
        rf"({MONTH_PATTERN})\s*,?\s*(\d{{4}})\b",
        re.IGNORECASE,
    )
    month_first = re.compile(
        rf"\b({MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?"
        rf"(?:\s*-\s*(\d{{1,2}})(?:st|nd|rd|th)?)?\s*,?\s*(\d{{4}})\b",
        re.IGNORECASE,
    )
    for match in day_first.finditer(normalized):
        add(match.group(4), match.group(3), match.group(1))
        if match.group(2):
            add(match.group(4), match.group(3), match.group(2))
    for match in month_first.finditer(normalized):
        add(match.group(4), match.group(1), match.group(2))
        if match.group(3):
            add(match.group(4), match.group(1), match.group(3))
    for match in re.finditer(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", normalized):
        keys.add((int(match.group(1)), int(match.group(2)), int(match.group(3))))
    return keys


def _surface_in_text(surface: str, text: str, target_type: str = "") -> bool:
    surface_norm = normalize_answer(surface)
    text_norm = normalize_answer(text)
    if not surface_norm or not text_norm:
        return False
    if re.search(rf"(?:^|\s){re.escape(surface_norm)}(?:\s|$)", text_norm) is not None:
        return True
    if target_type in {"date", "year"}:
        return bool(_date_keys(surface) & _date_keys(text))
    return False


def _occupation_surface_only_rewrite(before: str, candidate: str) -> bool:
    ignored = {"and", "or"}
    before_tokens = {
        token for token in re.findall(r"[a-z0-9]+", str(before or "").lower())
        if token not in ignored
    }
    candidate_tokens = {
        token for token in re.findall(r"[a-z0-9]+", str(candidate or "").lower())
        if token not in ignored
    }
    return bool(before_tokens) and before_tokens == candidate_tokens


def _answer_token_count(answer: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", str(answer or "")))


def _evidence_ids(value: Any, doc_count: int) -> List[int]:
    if not isinstance(value, list):
        return []
    ids = []
    for item in value:
        try:
            evidence_id = int(re.sub(r"^E", "", str(item).strip(), flags=re.IGNORECASE))
        except (TypeError, ValueError):
            continue
        if 1 <= evidence_id <= doc_count and evidence_id not in ids:
            ids.append(evidence_id)
    return ids


def _contract_state(
    question: str,
    query_answer_contract: Dict[str, Any],
) -> Tuple[str, str, Dict[str, Any]]:
    contract = dict((query_answer_contract or {}).get("contract") or {})
    target_type = str(
        contract.get("candidate_target_type")
        or contract.get("target_type")
        or infer_answer_target_type(question)
    )
    answer_slot = str(contract.get("answer_slot") or _infer_answer_slot(question, target_type))
    return target_type, answer_slot, contract


def cost_gated_constrained_reread(
    *,
    question: str,
    answer: str,
    qa_docs: Sequence[str],
    query_answer_contract: Dict[str, Any],
    answer_slot_binding_optimizer: Dict[str, Any],
    evidence_admission: Dict[str, Any],
    llm_infer: Callable[..., Any],
    enabled: bool = False,
    apply_rewrite: bool = True,
    min_net_utility: float = 0.42,
    min_candidate_support: float = 0.25,
    min_support_gain: float = 0.05,
    max_docs: int = 8,
    max_doc_chars: int = 900,
    max_total_chars: int = 6200,
    max_answer_tokens: int = 16,
    max_completion_tokens: int = 120,
    require_role_constraint: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Run a second reader only for unsupported answers with sufficient evidence."""
    before = str(answer or "").strip()
    meta = _new_meta(enabled=enabled, answer=before)
    if not enabled:
        return before, meta
    if not before:
        meta["reason"] = "empty_incumbent"
        return before, meta

    query_answer_contract = dict(query_answer_contract or {})
    answer_state = dict(query_answer_contract.get("answer_state") or {})
    asbo = dict(answer_slot_binding_optimizer or {})
    admission = dict(evidence_admission or {})
    target_type, answer_slot, contract = _contract_state(question, query_answer_contract)
    meta.update({
        "target_type": target_type,
        "answer_slot": answer_slot,
        "contract_active": bool(contract.get("active", False)),
        "contract_reason": str(query_answer_contract.get("reason", "") or ""),
        "contract_answer_supported": answer_state.get("supported"),
        "slot_optimizer_answer_supported": asbo.get("answer_supported"),
        "evidence_insufficient": admission.get("evidence_insufficient"),
    })

    if answer_state.get("supported") is not False:
        meta["reason"] = "contract_does_not_flag_unsupported"
        return before, meta
    if asbo.get("answer_supported") is True:
        meta["reason"] = "later_supported_answer_protected"
        return before, meta
    if admission.get("evidence_insufficient") is not False:
        meta["reason"] = "evidence_not_sufficient"
        return before, meta
    if not bool(contract.get("active", False)):
        meta["reason"] = "inactive_answer_contract"
        return before, meta

    compact_docs, rendered_docs = _compact_docs(
        qa_docs,
        max_docs=max_docs,
        max_doc_chars=max_doc_chars,
        max_total_chars=max_total_chars,
    )
    if not compact_docs:
        meta["reason"] = "no_evidence_docs"
        return before, meta

    prompt_chars = sum(len(doc) for doc in compact_docs)
    temporal_or_numeric = target_type in {"date", "year", "number"}
    estimated_utility = 0.56 + (0.10 if temporal_or_numeric else 0.06)
    estimated_utility += 0.08 if query_answer_contract.get("reason") == "below_contract_rewrite_gate" else 0.04
    estimated_risk = 0.08 + (0.08 if answer_slot in {"named_entity", "person"} else 0.03)
    estimated_cost = 0.06 + 0.14 * min(1.0, prompt_chars / max(1, int(max_total_chars)))
    net_utility = estimated_utility - estimated_risk - estimated_cost
    meta.update({
        "evidence_doc_count": len(compact_docs),
        "prompt_chars": prompt_chars,
        "estimated_prompt_tokens": int(round(prompt_chars / 4.0)),
        "estimated_utility": round(estimated_utility, 4),
        "estimated_risk": round(estimated_risk, 4),
        "estimated_cost": round(estimated_cost, 4),
        "net_utility": round(net_utility, 4),
    })
    if net_utility < float(min_net_utility):
        meta["reason"] = "cost_gate_rejected"
        return before, meta

    meta["triggered"] = True
    meta["action"] = "reread"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a provenance-constrained QA operator. Use only the numbered evidence rows. "
                "Return exactly one JSON object with keys answer, evidence_ids, and answer_type. "
                'Write evidence_ids as quoted strings, for example ["E1", "E3"]. '
                "The answer must directly answer the requested slot and must be copied from the cited evidence. "
                "If the evidence is insufficient, return an empty answer and empty evidence_ids."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Current unsupported answer: {before}\n"
                f"Required answer slot: {answer_slot}\n"
                f"Required answer type: {target_type}\n\n"
                "Evidence:\n"
                + "\n".join(rendered_docs)
            ),
        },
    ]
    try:
        try:
            result = llm_infer(
                messages,
                max_completion_tokens=int(max_completion_tokens),
                temperature=0.0,
            )
        except TypeError:
            result = llm_infer(messages)
    except Exception as exc:
        meta.update({
            "reason": "reread_inference_failed",
            "error_type": exc.__class__.__name__,
            "error_preview": str(exc)[:300],
        })
        return before, meta

    response, llm_metadata = unpack_llm_result(result)
    parsed, response_parse_mode = _extract_json_object(response)
    candidate = str(parsed.get("answer", "") or "").strip(" \t\r\n.;")
    cited_ids = _evidence_ids(parsed.get("evidence_ids"), len(compact_docs))
    meta.update({
        "llm_metadata": dict(llm_metadata or {}),
        "raw_response_preview": str(response or "")[:500],
        "response_parse_mode": response_parse_mode,
        "candidate": candidate,
        "candidate_answer_type": str(parsed.get("answer_type", "") or ""),
        "evidence_ids": cited_ids,
    })
    if not parsed:
        meta["reason"] = "invalid_json_response"
        return before, meta
    if not candidate or ABSTAIN_RE.fullmatch(normalize_answer(candidate)):
        meta["reason"] = "empty_or_abstain_candidate"
        return before, meta
    if normalize_answer(candidate) == normalize_answer(before):
        meta["reason"] = "same_answer"
        return before, meta
    if target_type == "occupation" and _occupation_surface_only_rewrite(before, candidate):
        meta["reason"] = "occupation_surface_only_rewrite"
        return before, meta
    if _answer_token_count(candidate) > int(max_answer_tokens):
        meta["reason"] = "candidate_too_long"
        return before, meta
    if not cited_ids:
        meta["reason"] = "missing_provenance"
        return before, meta

    cited_docs = [compact_docs[index - 1] for index in cited_ids]
    cited_text = "\n".join(cited_docs)
    candidate_surface_supported = _surface_in_text(candidate, cited_text, target_type)
    if not candidate_surface_supported:
        meta["reason"] = "candidate_not_in_cited_evidence"
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

    role_ok, role_reason = _candidate_passes_role_constraint(
        {
            "candidate": candidate,
            "title": _doc_title(cited_docs[0]),
            "sentence_preview": cited_text[:1200],
        },
        question,
        target_type,
        require_role_constraint,
    )
    meta["role_valid"] = bool(role_ok)
    meta["role_reject_reason"] = role_reason
    if not role_ok:
        meta["reason"] = "role_constraint_rejected"
        return before, meta

    incumbent_supported, incumbent_support = _is_supported(
        question,
        before,
        cited_docs,
        target_type,
        len(cited_docs),
    )
    candidate_supported, candidate_support = _is_supported(
        question,
        candidate,
        cited_docs,
        target_type,
        len(cited_docs),
    )
    if target_type in {"date", "year"} and candidate_surface_supported:
        candidate_supported = True
        candidate_support = max(float(candidate_support), 0.35)
    support_gain = float(candidate_support) - float(incumbent_support)
    meta.update({
        "incumbent_lexically_supported": bool(incumbent_supported),
        "incumbent_support_score": round(float(incumbent_support), 4),
        "candidate_supported": bool(candidate_supported),
        "candidate_support_score": round(float(candidate_support), 4),
        "support_gain": round(support_gain, 4),
    })
    if not candidate_supported or float(candidate_support) < float(min_candidate_support):
        meta["reason"] = "candidate_support_below_threshold"
        return before, meta
    if support_gain < float(min_support_gain):
        meta["reason"] = "candidate_support_gain_too_small"
        return before, meta

    meta.update({
        "changed": bool(apply_rewrite),
        "action": "replace" if apply_rewrite else "shadow_replace",
        "reason": "provenance_slot_support_verified",
        "after": candidate if apply_rewrite else before,
        "shadow_after": candidate,
    })
    return (candidate if apply_rewrite else before), meta
