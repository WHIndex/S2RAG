import re
from typing import Tuple


SHORT_FACTOID_STYLE = ""
EXPLICIT_DATE_PRECISION_STYLE = (
    "Answer precision: when the evidence explicitly gives a complete date, "
    "preserve the day, month, and year in the final Answer instead of reducing "
    "it to the year alone. Do not invent a missing day or month."
)
COMPLETE_PHRASE_STYLE = (
    "Answer style: complete_phrase. After \"Answer:\", return a brief complete phrase or one short sentence "
    "that preserves the requested reason, event, role, relationship, or required qualifier."
)
MULTI_SLOT_STYLE = (
    "Answer style: multi_slot. After \"Answer:\", include every requested piece in one compact answer, "
    "separating pieces with semicolons when helpful."
)
ANSWER_MARKER_RE = re.compile(r"\bAnswer\s*:\s*", re.IGNORECASE)
LEADING_ANSWER_RE = re.compile(
    r"^\s*(?:answer\s*[:\-]\s*|the\s+answer\s+is\s+)",
    re.IGNORECASE,
)
TRAILING_ASSUMPTION_RE = re.compile(
    r"\s*\((?:assuming|based on|according to|if|since|from the context)[^)]*\)\s*\.?\s*$",
    re.IGNORECASE,
)
TRAILING_EXPLANATION_RE = re.compile(
    r"\s*(?:,?\s+(?:according to|based on|from)\s+the\s+(?:text|passage|context).*)$",
    re.IGNORECASE,
)
ABSTAIN_PHRASE_RE = re.compile(
    r"\b(?:not specified|not provided|not mentioned|cannot determine|cannot be determined|"
    r"unknown|insufficient information|no information)\b",
    re.IGNORECASE,
)
YES_NO_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
SURROUNDING_QUOTES = "\"'` \t\r\n"


def classify_question_answer_style(question: str) -> str:
    """Route questions to a compact answer style without using dataset names."""
    q = f" {question.strip().lower()} "
    q = re.sub(r"\s+", " ", q)





    asks_second_question = re.search(
        r"\b(?:who|what|where|when|which|how long|how many)\b.+(?:,| and )\s*"
        r"(?:who|what|where|when|which|how long|how many)\s+"
        r"(?:is|are|was|were|did|does|do|has|have|had|can|could|would|will)\b",
        q,
    )
    asks_entity_and_attribute = re.search(
        r"\bwho\b.+\b(?:job|role|occupation|profession|position)\b"
        r"|\b(?:job|role|occupation|profession|position)\b.+\bwho\b",
        q,
    )
    asks_named_list = re.search(r"\bwhat (?:are|were) the names?\b|\bwhich (?:two|three|four)\b", q)
    if asks_second_question or asks_entity_and_attribute or asks_named_list:
        return "multi_slot"

    if re.search(r"\bwhy\b|\bwhat happened\b|\bwhat is ironic\b|\bwhat was ironic\b", q):
        return "complete_phrase"
    if re.search(r"\bhow (?:did|does|do|was|were|is|are|can|could|would|will|should)\b", q):
        if not re.search(r"\bhow (?:many|much|long|old|far)\b", q):
            return "complete_phrase"
    if re.search(r"\b(?:reason|cause|caused|because)\b", q):
        return "complete_phrase"

    return "short_factoid"


def _asks_for_explicit_date(question: str) -> bool:
    q = " ".join(str(question or "").casefold().split())
    if re.search(r"\b(?:what|which|in) year\b", q):
        return False
    if re.search(r"\b(?:what date|date of (?:birth|death)|birthday)\b", q):
        return True
    return bool(
        re.search(r"\bwhen\b.*\b(?:born|birth|died|die|death)\b", q)
        or re.search(r"\b(?:born|birth|died|die|death)\b.*\bwhen\b", q)
    )


def get_answer_style_instruction(
    question: str,
    *,
    preserve_explicit_date_precision: bool = False,
) -> Tuple[str, str]:
    style = classify_question_answer_style(question)
    if style == "multi_slot":
        return style, MULTI_SLOT_STYLE
    if style == "complete_phrase":
        return style, COMPLETE_PHRASE_STYLE
    if preserve_explicit_date_precision and _asks_for_explicit_date(question):
        return style, EXPLICIT_DATE_PRECISION_STYLE


    return style, SHORT_FACTOID_STYLE


def extract_answer_from_response(response_content: str) -> Tuple[str, bool]:
    """Extract the final answer segment from a CoT-style response."""
    if response_content is None:
        return "", False

    matches = list(ANSWER_MARKER_RE.finditer(response_content))
    if not matches:
        return response_content.strip(), False


    return response_content[matches[-1].end():].strip(), True


def _strip_wrapping_quotes(answer: str) -> str:
    stripped = answer.strip()
    while len(stripped) >= 2 and stripped[0] in SURROUNDING_QUOTES and stripped[-1] == stripped[0]:
        stripped = stripped[1:-1].strip()
    return stripped


def _strip_sentence_noise(answer: str) -> str:
    answer = TRAILING_ASSUMPTION_RE.sub("", answer).strip()
    answer = TRAILING_EXPLANATION_RE.sub("", answer).strip()
    return answer


def _token_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _is_short_informative_candidate(text: str) -> bool:
    candidate = text.strip(" .;:-")
    if not candidate or ABSTAIN_PHRASE_RE.search(candidate):
        return False
    if _token_count(candidate) > 12:
        return False
    return True


def _recover_informative_uncertainty_answer(answer: str) -> str:
    """Recover the supported part from mixed answers like 'X; date not specified'."""
    stripped = answer.strip()
    if not ABSTAIN_PHRASE_RE.search(stripped):
        return stripped

    for separator in (";", " - ", " – ", " — "):
        if separator not in stripped:
            continue
        parts = [part.strip(" .;:-") for part in stripped.split(separator)]
        informative = [part for part in parts if _is_short_informative_candidate(part)]
        if len(informative) == 1:
            return informative[0]

    match = re.match(
        r"^\s*(?:not specified|not provided|not mentioned|cannot determine|cannot be determined|unknown)"
        r"\s*(?:,|;)?\s*(?:but|however)\s+(?P<candidate>.+)$",
        stripped,
        re.IGNORECASE,
    )
    if match:
        candidate = match.group("candidate").strip(" .;:-")
        if _is_short_informative_candidate(candidate):
            return candidate

    match = re.match(
        r"^(?P<candidate>.+?)\s*,?\s+but\s+.+\b(?:not specified|not provided|unknown|cannot determine)\b.*$",
        stripped,
        re.IGNORECASE,
    )
    if match:
        candidate = match.group("candidate").strip(" .;:-")
        if _is_short_informative_candidate(candidate):
            return candidate

    return stripped


def _strip_terminal_period(answer: str) -> str:
    stripped = answer.strip()
    if stripped.endswith(".") and not re.search(r"\b[A-Z]\.$", stripped):
        return stripped[:-1].strip()
    return stripped


def _is_yes_no_question(question: str) -> bool:
    return bool(re.match(r"\s*(?:is|are|was|were|do|does|did|has|have|had|can|could|would|will|should)\b", question, re.IGNORECASE))


def _collapse_answer_lines(answer: str) -> str:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not lines:
        return answer.strip()
    return " ".join(lines)


def _compact_location_answer(question: str, answer: str) -> str:
    lowered_question = question.lower()
    if "," not in answer:
        return answer
    if re.search(r"\d", answer):
        return answer

    parts = [part.strip() for part in answer.split(",") if part.strip()]
    if len(parts) < 2:
        return answer

    if re.search(r"\b(?:which|what)\s+country\b", lowered_question):
        return parts[-1]
    return answer


def postprocess_answer(question: str, answer: str) -> str:
    """Conservatively normalize a predicted answer for exact-span QA metrics.

    This does not use gold answers. It removes common formatting noise after
    the final "Answer:" marker while keeping entity names and titles intact.
    """
    if answer is None:
        return ""

    cleaned = answer.strip()
    if not cleaned:
        return cleaned

    cleaned = _collapse_answer_lines(cleaned)
    cleaned = LEADING_ANSWER_RE.sub("", cleaned).strip()
    cleaned = _strip_wrapping_quotes(cleaned)
    cleaned = _strip_sentence_noise(cleaned)
    cleaned = _strip_terminal_period(cleaned)
    cleaned = _recover_informative_uncertainty_answer(cleaned)

    if _is_yes_no_question(question):
        match = YES_NO_RE.match(cleaned)
        if match:
            return match.group(1).lower()

    cleaned = _compact_location_answer(question, cleaned)
    cleaned = _strip_terminal_period(cleaned)
    cleaned = _strip_wrapping_quotes(cleaned)
    return cleaned.strip()
