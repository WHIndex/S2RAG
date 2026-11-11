import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from .adaptive_query_executor_utils import estimate_inverted_index_access
from .misc_utils import flatten_facts, text_processing


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
BOOLEAN_QUESTION_RE = re.compile(
    r"^\s*(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would)\b",
    re.IGNORECASE,
)


QUERY_PLAN_LIFECYCLE_VERSION = "query_plan_lifecycle_v1"


def validate_logical_query_plan_static(
    plan: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate only the executable IR shape, independent of answer quality.

    This deliberately does not inspect retrieval output, reader answers, or QA
    gold.  Both rule and LLM-assisted compilers use this validator before a
    plan can enter execution.
    """
    operators = [dict(row) for row in plan.get("operators") or []]
    if str(plan.get("compile_status") or "") != "compiled":
        return {
            "valid": False,
            "reason": "compile_status_not_compiled",
            "operator_count": len(operators),
        }
    if not operators:
        return {
            "valid": False,
            "reason": "operator_list_empty",
            "operator_count": 0,
        }
    supported_shapes = {
        ("relation_projection", "project"),
        ("path_projection", "project"),
        ("branch_compare", "compare_eq"),
        ("branch_temporal_compare", "argmin_root"),
        ("branch_temporal_compare", "argmax_root"),
    }
    shape = (
        str(plan.get("plan_kind") or ""),
        str(plan.get("answer_operator") or ""),
    )
    if shape not in supported_shapes:
        return {
            "valid": False,
            "reason": "executor_unsupported_plan_shape",
            "operator_count": len(operators),
        }
    operator_ids: Set[str] = set()
    previous_output = "$root_each" if shape[0].startswith("branch_") else "$root"
    for index, row in enumerate(operators):
        operator_id = str(row.get("operator_id") or "")
        if not operator_id or operator_id in operator_ids:
            return {
                "valid": False,
                "reason": f"invalid_or_duplicate_operator_id_op{index}",
                "operator_count": len(operators),
            }
        operator_ids.add(operator_id)
        if str(row.get("operator_type") or "") != "indexed_fact_lookup":
            return {
                "valid": False,
                "reason": f"unsupported_operator_type_op{index}",
                "operator_count": len(operators),
            }
        actual_input = str(row.get("input_slot") or "")
        actual_output = str(row.get("output_slot") or "")
        if actual_input != previous_output:
            return {
                "valid": False,
                "reason": f"disconnected_input_slot_op{index}",
                "operator_count": len(operators),
            }
        if not actual_output or actual_output == actual_input:
            return {
                "valid": False,
                "reason": f"invalid_output_slot_op{index}",
                "operator_count": len(operators),
            }
        if not (
            row.get("predicate_families")
            or row.get("relation_tokens")
            or row.get("canonical_relation")
        ):
            return {
                "valid": False,
                "reason": f"missing_relation_contract_op{index}",
                "operator_count": len(operators),
            }
        previous_output = actual_output
    if shape[1] == "project" and previous_output != "$answer":
        return {
            "valid": False,
            "reason": "terminal_answer_slot_missing",
            "operator_count": len(operators),
        }
    return {
        "valid": True,
        "reason": "validated",
        "operator_count": len(operators),
    }


def attach_logical_query_plan_identity(
    plan: Mapping[str, Any],
    *,
    compiler_source: str,
    compiler_mode: str,
    attempt_id: str = "attempt0",
) -> Dict[str, Any]:
    """Attach immutable compiler provenance and a stable content identity."""
    result = dict(plan or {})
    validation = validate_logical_query_plan_static(result)
    identity_payload = {
        key: value
        for key, value in result.items()
        if key not in {
            "plan_id",
            "compiler_provenance",
            "compile_phase_status",


            "query_compiler_trace",
        }
    }
    digest = hashlib.sha256(
        json.dumps(
            identity_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:20]
    plan_id = f"plan:{digest}"
    result["plan_id"] = plan_id
    result["compiler_provenance"] = {
        "version": QUERY_PLAN_LIFECYCLE_VERSION,
        "plan_id": plan_id,
        "attempt_id": str(attempt_id),
        "compiler_source": str(compiler_source),
        "compiler_mode": str(compiler_mode),
    }
    result["compile_phase_status"] = {
        "version": QUERY_PLAN_LIFECYCLE_VERSION,
        "plan_id": plan_id,
        "plan_generated": True,
        "plan_static_valid": bool(validation["valid"]),
        "static_validation_reason": str(validation["reason"]),
    }
    return result
BRIDGE_CUE_RE = re.compile(
    r"\b(?:"
    r"both|same|compare|which of|"
    r"father|mother|grandfather|grandmother|spouse|wife|husband|son|daughter|"
    r"director|writer|author|composer|producer|performer|singer|creator|founder|designer|"
    r"where .+ (?:born|located|died|founded|created|released|published)|"
    r"when .+ (?:born|founded|created|released|published|abolished)|"
    r"(?:that|which|who|where) (?:was|were|is|are|has|have|had|located|based|born|died|created|released|published)|"
    r"of the (?:director|writer|author|composer|producer|performer|singer|creator|founder|designer|father|mother|spouse)"
    r")\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "in", "is", "it", "its",
    "of", "on", "or", "she", "that", "the", "their", "they", "this", "to",
    "was", "were", "which", "who", "with", "what", "when", "where", "why",
    "how", "did", "does", "do", "were", "was", "name", "called", "known",
}

PLACE_TERMS = {
    "airport", "area", "bay", "borough", "capital", "city", "continent",
    "country", "county", "district", "island", "lake", "mountain", "province",
    "region", "republic", "river", "sea", "state", "territory", "town",
    "village",
}
ORG_TERMS = {
    "academy", "agency", "airlines", "association", "bank", "band", "club",
    "college", "company", "conference", "corporation", "department", "division",
    "festival", "foundation", "group", "institute", "label", "league", "ministry",
    "museum", "party", "records", "school", "team", "university",
}
WORK_TERMS = {
    "album", "book", "film", "magazine", "movie", "novel", "opera", "play",
    "series", "song", "tournament", "work",
}
NATIONALITY_TERMS = {
    "nationality", "citizenship", "citizen", "national",
    "american", "british", "english", "french", "german", "chinese", "japanese",
    "indian", "canadian", "australian", "italian", "spanish", "russian",
    "irish", "scottish", "welsh", "dutch", "swedish", "polish", "brazilian",
    "mexican", "korean", "turkish", "ukrainian", "israeli", "iranian",
    "pakistani", "nigerian", "danish", "greek", "panamanian", "venezuelan",
    "georgian", "norwegian", "finnish", "icelandic", "belgian", "swiss",
    "austrian", "hungarian", "romanian", "bulgarian", "serbian", "croatian",
    "slovenian", "slovak", "czech", "portuguese", "argentine", "argentinian",
    "colombian", "chilean", "peruvian", "cuban", "egyptian", "moroccan",
    "algerian", "tunisian", "ethiopian", "kenyan", "ghanaian", "south african",
    "new zealander", "indonesian", "malaysian", "filipino", "thai",
    "vietnamese", "singaporean", "bangladeshi", "sri lankan", "nepali",
    "afghan", "iraqi", "syrian", "lebanese", "jordanian", "saudi",
    "emirati", "armenian", "azerbaijani", "kazakh", "uzbek", "american",
    "country",
}

RELATION_FAMILY_TERMS = {
    "kinship": {
        "aunt", "brother", "child", "children", "cousin", "daughter",
        "father", "grandchild", "granddaughter", "grandfather", "grandmother",
        "grandparent", "husband", "mother", "parent", "sibling", "sister",
        "son", "spouse", "uncle", "wife",
    },
    "creator": {
        "architect", "artist", "author", "composer", "creator", "designer",
        "directed", "director", "founded", "founder", "invented", "inventor",
        "painted", "painter", "produced", "producer", "wrote", "writer",
    },
    "location": {
        "based", "born", "capital", "city", "country", "died", "headquartered",
        "located", "location", "place", "region", "state", "where",
    },
    "temporal": {
        "abolished", "created", "date", "died", "founded", "released",
        "published", "started", "when", "year",
    },
    "membership": {
        "affiliated", "belong", "belongs", "club", "conference", "division",
        "league", "member", "part", "party", "played", "team",
    },
    "draft_selection": {"draft", "drafted", "selected", "selection"},
    "identity": {
        "citizen", "citizenship", "national", "nationality",
    },
}

RELATION_POLARITY_TERMS = {
    "paternal": {"father", "paternal", "grandfather"},
    "maternal": {"mother", "maternal", "grandmother"},
    "birth": {"birth", "birthplace", "born", "native"},
    "death": {"death", "die", "died", "dies"},
}

PARENT_RELATION_TERMS = {"father", "mother", "parent"}
CHILD_RELATION_TERMS = {"son", "daughter", "child", "children"}
DIRECTOR_RELATION_TERMS = {"directed", "director"}
BIRTH_PLACE_RELATION_TERMS = {"born", "birthplace"}
BIRTH_DATE_RELATION_TERMS = {"born", "birth", "birthday", "date"}
SURFACE_MONTH_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b",
    re.IGNORECASE,
)
SURFACE_PROPER_SEGMENT = r"(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'.-]*|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'.-]*|[A-Z]{2,})){0,4}"

ANSWER_TYPE_TERMS = {
    "person": {"who", "person", "people", "father", "mother", "author", "director", "writer"},
    "place": {"where", "place", "city", "country", "state", "location"},
    "nationality": {"nationality", "citizenship", "citizen", "national"},
    "date": {"when", "date", "year"},
    "number": {"how", "many", "number", "population", "age"},
    "work": {"film", "book", "album", "song", "novel", "series", "work"},
    "organization": {"company", "organization", "team", "university", "club"},
}






TERMINAL_ENTITY_CATEGORY_TERMS = {
    "airport": {"airport", "airports", "aerodrome"},
    "bridge": {"bridge", "bridges"},
    "building": {"building", "buildings"},
    "castle": {"castle", "castles", "fort", "fortress"},
    "city": {"city", "cities", "town", "towns", "village", "villages"},
    "country": {"country", "countries", "nation", "nations"},
    "county": {"county", "counties"},
    "lake": {"lake", "lakes"},
    "monument": {"monument", "monuments", "memorial", "memorials"},
    "mountain": {"mountain", "mountains", "mount", "peak", "peaks"},
    "museum": {"museum", "museums"},
    "river": {"river", "rivers"},
    "region": {"region", "regions", "territory", "territories"},
    "school": {"school", "schools", "college", "colleges"},
    "state": {"state", "states", "province", "provinces"},
    "university": {"university", "universities"},
}
TERMINAL_ENTITY_CATEGORY_PARENTS = {
    "airport": {"entity", "facility", "place"},
    "bridge": {"entity", "place", "structure"},
    "building": {"entity", "place", "structure"},
    "castle": {"building", "entity", "place", "structure"},
    "city": {"entity", "place"},
    "country": {"entity", "place"},
    "county": {"entity", "place"},
    "lake": {"entity", "natural_feature", "place"},
    "monument": {"entity", "place", "structure"},
    "mountain": {"entity", "natural_feature", "place"},
    "museum": {"entity", "organization", "place"},
    "river": {"entity", "natural_feature", "place"},
    "region": {"entity", "place"},
    "school": {"entity", "organization", "place"},
    "state": {"entity", "place"},
    "university": {"entity", "organization", "place"},
}




TERMINAL_ENTITY_DESCRIPTOR_TERMS = {
    **TERMINAL_ENTITY_CATEGORY_TERMS,
    "building": {
        *TERMINAL_ENTITY_CATEGORY_TERMS["building"],
        "house",
        "houses",
        "manor",
        "manors",
        "mansion",
        "mansions",
        "palace",
        "palaces",
        "structure",
        "structures",
        "tower",
        "towers",
    },
}




SOURCE_LOCAL_PERSON_DESCRIPTOR_TERMS = {
    "actor", "actress", "architect", "artist", "author", "composer",
    "director", "filmmaker", "historian", "journalist", "musician",
    "novelist", "painter", "philosopher", "physician", "politician",
    "producer", "professor", "psychologist", "researcher", "scientist",
    "singer", "songwriter", "writer",
}
TERMINAL_TYPE_COMPATIBLE = "COMPATIBLE"
TERMINAL_TYPE_UNKNOWN = "UNKNOWN"
TERMINAL_TYPE_INCOMPATIBLE = "INCOMPATIBLE"

PLAN_COMPLETE = "PLAN_COMPLETE"
PLAN_PARTIAL = "PLAN_PARTIAL"
PLAN_COVERAGE_UNKNOWN = "PLAN_COVERAGE_UNKNOWN"
QUESTION_OBLIGATION_CONTRACT_VERSION = "question_obligation_contract_v1"
PLAN_COVERAGE_CONTRACT_VERSION = "question_level_plan_coverage_v1"

GENERIC_ANCHOR_STOPWORDS = {
    "answer", "called", "current", "following", "given", "known", "list",
    "name", "named", "question", "title", "type",
}
GENERIC_RESIDUAL_RELATION_TOKENS = {
    "city", "country", "date", "located", "location", "place", "state", "when", "where",
    "who", "year",
}





QUERY_SLOT_PHRASE_KEYS = {
    "birth city", "birth country", "birth date", "birth place",
    "death city", "death country", "death date", "death place",
    "director film", "producer film", "writer film", "director movie",
    "movie producer", "movie writer", "composer song", "creator work",
}





PREDICATE_FAMILY_TERMS = {
    "birth": {"birth", "birthday", "birthplace", "born", "native"},
    "death": {"death", "die", "died", "dies"},
    "capital": {"capital"},
    "containment": {
        "based", "contained", "contains", "located", "location", "situated",
    },
    "geopolitical_location": {
        "city", "country", "county", "province", "region", "state", "territory",
    },
    "founding": {
        "established", "formed", "formation", "founded", "founding", "incorporated",
    },
    "leadership": {
        "chair", "chairman", "chairwoman", "founder", "headed", "leader", "led",
        "president",
    },
    "creation": {
        "architect", "artist", "author", "composer", "created", "creator",
        "designed", "designer", "directed", "director", "painted", "painter",
        "produced", "producer", "wrote", "writer",
    },
    "distribution": {
        "distributed", "distributing", "distribution", "published", "publisher",
        "released",
    },
    "education": {
        "attended", "educated", "education", "graduated", "school", "studied",
        "university",
    },
    "employment": {"employed", "employment", "occupation", "worked"},
    "nationality": {"citizen", "citizenship", "from", "national", "nationality"},
    "membership": {
        "affiliated", "belong", "belongs", "conference", "division", "joined",
        "league", "member", "membership", "part", "played", "team",
    },
    "draft_selection": {"draft", "drafted", "selected", "selection"},
    "portrayal": {"acted", "actor", "actress", "portrayed", "portrays"},
    "border": {"adjacent", "border", "bordered", "borders", "neighbor", "neighbour"},
    "hydrology_outflow": {
        "drain", "drained", "drains", "empties", "empty", "flows", "mouth", "outflow",
    },
    "start_time": {"began", "begin", "commenced", "start", "started"},
}






SCHEMA_PREDICATE_FAMILY_TERMS = {
    "birth": {"birth", "birthday", "birthplace", "born", "native"},
    "death": {"death", "die", "died", "dies"},
    "capital": {"capital"},
    "containment": {
        "based", "contained", "contains", "located", "location", "situated",
    },
    "geopolitical_location": {
        "city", "country", "county", "province", "region", "state", "territory",
    },
    "founding": {
        "established", "formed", "formation", "founded", "founding", "incorporated",
    },
    "leadership": {
        "chair", "chairman", "chairwoman", "headed", "leader", "led", "president",
    },
    "authorship": {"author", "authored", "wrote", "writer", "written"},
    "direction": {"directed", "director"},
    "composition": {"composed", "composer", "composition"},
    "performance": {
        "performed", "performer", "recorded", "sang", "singer", "vocalist",
    },
    "production": {"produced", "producer"},
    "visual_art": {"artist", "painted", "painter"},
    "design": {"architect", "designed", "designer"},
    "creation": {"created", "creator"},
    "distribution": {
        "distributed", "distributing", "distribution", "published", "publisher",
        "released",
    },
    "education": {
        "attended", "educated", "education", "graduated", "school", "studied",
        "university",
    },
    "employment": {"employed", "employment", "occupation", "worked"},
    "nationality": {"citizen", "citizenship", "from", "national", "nationality"},
    "membership": {
        "affiliated", "belong", "belongs", "conference", "division", "joined",
        "league", "member", "membership", "part", "played", "team",
    },
    "draft_selection": {"draft", "drafted", "selected", "selection"},
    "border": {"adjacent", "border", "bordered", "borders", "neighbor", "neighbour"},
    "hydrology_outflow": {
        "drain", "drained", "drains", "empties", "empty", "flows", "mouth", "outflow",
    },
    "start_time": {"began", "begin", "commenced", "start", "started"},
    "end_time": {
        "abolished", "ceased", "closed", "dissolved", "ended", "ending",
        "finished", "terminated",
    },
    "kinship": {
        "child", "daughter", "father", "mother", "parent", "son", "spouse",
        "married", "marriage", "partner", "consort",
    },


    "mother_of": {"maternal", "mother"},
    "father_of": {"father", "paternal"},
    "parent_of": {"parent"},
    "daughter_of": {"daughter"},
    "son_of": {"son"},
    "child_of": {"child"},
    "spouse_of": {
        "husband", "spouse", "wife", "married", "marriage", "partner",
        "consort",
    },
    "identity": {"alias", "called", "known", "named"},
}

DATE_VALUE_RE = re.compile(
    r"(?:\b(?:[1-9]\d{3}|20\d{2}|21\d{2})\b|"
    r"\b(?:[1-9]\d{0,3})\s*(?:bc|bce|ad|ce)\b)",
    re.IGNORECASE,
)
NUMBER_VALUE_RE = re.compile(r"\b\d+(?:[,.]\d+)*(?:\.\d+)?\b")

MONTH_TO_NUMBER = {
    name: index
    for index, name in enumerate(
        (
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november",
            "december",
        ),
        start=1,
    )
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low
    if not math.isfinite(value):
        return low
    return min(high, max(low, value))


def _tokens(text: Any, *, keep_numbers: bool = True) -> Set[str]:
    result = set()
    for token in TOKEN_RE.findall(str(text or "").lower()):
        if token in STOPWORDS:
            continue
        if len(token) <= 1:
            continue
        if not keep_numbers and token.isdigit():
            continue
        result.add(token)
    return result


def _entity_key(text: Any) -> str:
    return " ".join(sorted(_tokens(text)))[:160]


def canonical_entity_key(text: Any) -> str:
    """Return the normalized entity key used by physical fact indexes."""
    return _entity_key(text)


def canonical_entity_identity_key(text: Any) -> str:
    """Return an ordered, lossless-enough key for title identity checks.

    The physical join key deliberately removes stopwords and token order to
    tolerate harmless surface variation.  That representation is unsafe for
    deciding whether a question span names a passage title: ``With Me`` and
    ``Why Did It Have to Be Me?`` can both collapse to ``me``.  Identity keys
    preserve token order and function words, while the existing entity key
    remains the value used by fact joins.
    """
    tokens = _surface_word_tokens(text)
    return " ".join(tokens)[:240]


def canonical_title_surface_key(text: Any) -> str:
    """Normalize a title while preserving identity-bearing punctuation.

    Fact joins intentionally use a permissive bag-of-words entity key.  A
    passage-title binding needs a stricter representation because titles such
    as ``Johnny-on-the-Spot`` and ``Johnny on the Spot`` may name different
    works.  Case and Unicode dash variants are harmless; the presence of the
    dash is not.
    """
    value = str(text or "").casefold().strip()
    value = re.sub(r"[‐‑‒–—―]", "-", value)
    value = re.sub(r"\s*-\s*", "-", value)
    value = re.sub(r"\s+", " ", value)
    return value[:300]


def _surface_word_tokens(text: Any) -> List[str]:
    """Tokenize an entity surface while preserving order and Unicode letters."""
    return [
        token.casefold()
        for token in re.findall(r"[^\W_]+", str(text or ""), flags=re.UNICODE)
        if token
    ]


def _explicit_age_comparison_direction(question: Any) -> str:
    """Return the direction of an explicit age comparison.

    Bare ``older``/``younger`` tokens are unsafe because they can be part of
    an entity name, as in ``Charles the Younger``.  Requiring a copular
    comparison (or an explicit ``than`` construction) keeps named entities
    from changing the logical predicate selected by the compiler.
    """
    text = " ".join(str(question or "").casefold().split())
    if not text:
        return ""
    if re.search(
        r"\b(?:is|was|are|were)\s+(?:the\s+)?(?:older|oldest)\b|"
        r"\b(?:older|oldest)\s+than\b",
        text,
    ):
        return "argmin_root"
    if re.search(
        r"\b(?:is|was|are|were)\s+(?:the\s+)?(?:younger|youngest)\b|"
        r"\b(?:younger|youngest)\s+than\b",
        text,
    ):
        return "argmax_root"
    return ""


def _temporal_comparison_scope(question: Any) -> str:
    """Return the question span allowed to define a temporal predicate.

    The comparison templates used by the relation workload place candidate
    entities after a comma.  Relation words inside those titles are data, not
    query operators (for example, ``Love & Death in Saigon``).  When the
    prefix already contains a complete temporal comparison, mask the suffix
    from predicate compilation.  Questions without this syntax retain their
    full surface and are handled by relation-local patterns below.
    """
    text = " ".join(str(question or "").casefold().split())
    if not text or "," not in text:
        return text
    prefix = text.split(",", 1)[0].strip()
    has_relation_comparison = bool(re.search(
        r"\b(?:born|birth|birthday|died|death|released|release|"
        r"premiered|premiere)\b[^,?]{0,32}"
        r"\b(?:first|earlier|earliest|last|later|latest|most recent|"
        r"most recently)\b",
        prefix,
    ))
    if has_relation_comparison or _explicit_age_comparison_direction(prefix):
        return prefix
    return text


def _temporal_comparison_direction(question: Any) -> str:
    """Resolve a temporal comparator without reading cue words from titles.

    A global search for ``first`` miscompiled questions such as "born later,
    The First Day of Freedom or ...".  Prefer relation-local comparator
    phrases, then fall back to unambiguous age cues.  The compiler can safely
    reject a query when no local comparator exists; guessing the wrong
    direction is substantially more harmful than retaining the Reader path.
    """
    text = _temporal_comparison_scope(question)
    if not text:
        return ""
    relation = (
        r"born|birth|birthday|died|death|released|release|premiered|premiere"
    )
    minimum = r"first|earlier|earliest|older|oldest"
    maximum = r"last|later|latest|most recent|most recently|younger|youngest"
    local_patterns = (
        ("argmin_root", rf"\b(?:{relation})\b[^,?]{{0,32}}\b(?:{minimum})\b"),
        ("argmax_root", rf"\b(?:{relation})\b[^,?]{{0,32}}\b(?:{maximum})\b"),
        ("argmin_root", rf"\b(?:{minimum})\b[^,?]{{0,32}}\b(?:{relation})\b"),
        ("argmax_root", rf"\b(?:{maximum})\b[^,?]{{0,32}}\b(?:{relation})\b"),
    )
    for direction, pattern in local_patterns:
        if re.search(pattern, text):
            return direction
    return _explicit_age_comparison_direction(text)


def _is_query_slot_phrase(surface: Any) -> bool:
    """Return whether a query mention names an answer/relation slot.

    Entity keys are bag-of-token keys, so compare both the ordered surface and
    the normalized key.  The list is deliberately narrow: rejecting a valid
    named work is worse than leaving an unusual schema phrase unclassified.
    """
    words = _surface_word_tokens(surface)
    if not words:
        return False
    ordered = " ".join(words)
    key = _entity_key(surface)
    if key in QUERY_SLOT_PHRASE_KEYS:
        return True





    if re.fullmatch(
        r"(?:is|was|were|are|has|have|had|did|does|do)\s+.+",
        ordered,
    ):
        return True
    if re.search(r"\b(?:a|an|of|the|that|was|were|which)$", ordered):
        return True
    if re.match(r"^(?:as\s+(?:an?\s+)?|of\s+the\s+)", ordered):
        return True
    descriptor_roles = {"composer", "musician", "singer", "songwriter"}
    descriptor_modifiers = set(NATIONALITY_TERMS) | {
        "electronic", "independent", "recording",
    }
    if (
        set(words) & descriptor_roles
        and set(words).issubset(descriptor_roles | descriptor_modifiers)
    ):
        return True
    return bool(re.fullmatch(
        r"(?:place|date|city|country) of (?:birth|death)",
        ordered,
    ))


def _query_root_surface_match(question: str, entity: Any) -> bool:
    """Return whether ``entity`` is an explicit root in the question body.

    Parenthetical text commonly qualifies another entity (for example,
    ``Rio Verde (Mato Grosso do Sul)``).  Treating that qualifier as a root
    permits reverse sibling joins, so lineage compilation deliberately ignores
    parenthetical spans.
    """




    query_tokens = _surface_word_tokens(
        re.sub(r"\([^)]*\)", " ", str(question or ""))
    )
    entity_tokens = _surface_word_tokens(
        re.sub(r"\([^)]*\)", " ", str(entity or ""))
    )
    if not query_tokens or not entity_tokens:
        return False
    width = len(entity_tokens)
    return any(
        query_tokens[offset:offset + width] == entity_tokens
        for offset in range(0, len(query_tokens) - width + 1)
    )


ROOT_ENTITY_CATEGORY_ALIASES = {
    "actor": "person",
    "actress": "person",
    "album": "album",
    "albums": "album",
    "band": "organization",
    "book": "book",
    "books": "book",
    "club": "team",
    "clubs": "team",
    "film": "film",
    "films": "film",
    "movie": "film",
    "movies": "film",
    "novel": "book",
    "novels": "book",
    "person": "person",
    "record": "album",
    "records": "album",
    "series": "series",
    "song": "song",
    "songs": "song",
    "team": "team",
    "teams": "team",
    "university": "organization",
    "universities": "organization",
}


def canonical_root_entity_categories(values: Iterable[Any]) -> Set[str]:
    """Normalize question and title categories used for root identity.

    These categories identify an entity partition, not a terminal answer type.
    In particular, ``film`` and ``album`` must remain distinct even though
    both are broad ``work`` values in the logical schema.
    """
    result: Set[str] = set()
    for value in values or []:
        for token in _surface_word_tokens(value):
            category = ROOT_ENTITY_CATEGORY_ALIASES.get(token)
            if category:
                result.add(category)
    return result


def infer_question_root_entity_categories(
    question: Any,
    *,
    branch_plan: bool = False,
) -> Set[str]:
    """Infer only explicit root categories from a branch question.

    The inference is intentionally limited to branch comparisons where the
    question names the compared entity class (``Which movie ...`` or
    ``Who ...``).  For nested path questions, the answer type is not the root
    type, so guessing here would erase an upstream relation obligation.
    """
    if not branch_plan:
        return set()
    text = re.sub(r"\s+", " ", str(question or "")).strip().casefold()
    if not text:
        return set()
    scope = re.split(r"\s*,\s*|\bout\s+of\b", text, maxsplit=1)[0]
    if re.match(r"^(?:who|which\s+(?:person|actor|actress))\b", scope):
        return {"person"}
    category_match = re.match(
        r"^(?:which|what)\s+"
        r"(?:(?:of|the|two|other|both|[12][0-9]{3}['\u2019]?s?)\s+){0,4}"
        r"(?P<category>films?|movies?|albums?|books?|novels?|records?|"
        r"songs?|series|teams?|clubs?|universities)\b",
        scope,
    )
    if category_match is None:
        return set()
    return canonical_root_entity_categories([category_match.group("category")])


def _title_document_categories(title: Any, passage: Any) -> Set[str]:
    """Infer a conservative entity category for one titled passage.

    Parenthetical title qualifiers are strongest.  Unqualified titles use
    only the lead definitional clause, so an album page mentioning a film
    later in the passage cannot become a film partition accidentally.
    """
    title_text = re.sub(r"\s+", " ", str(title or "")).strip()
    passage_text = str(passage or "")
    categories = canonical_root_entity_categories(
        re.findall(r"\(([^()]*)\)", title_text)
    )
    lines = [line.strip() for line in passage_text.splitlines() if line.strip()]
    lead = " ".join(lines[1:2] if len(lines) > 1 else lines[:1]).casefold()
    lead = lead[:320]
    definitional_patterns = {
        "film": r"\b(?:is|was)\s+(?:an?\s+)?[^.;]{0,80}\b(?:film|movie)\b",
        "album": r"\b(?:is|was)\s+(?:an?\s+)?[^.;]{0,80}\b(?:album|record)\b",
        "book": r"\b(?:is|was)\s+(?:an?\s+)?[^.;]{0,80}\b(?:book|novel)\b",
        "song": r"\b(?:is|was)\s+(?:an?\s+)?[^.;]{0,80}\bsong\b",
        "series": r"\b(?:is|was)\s+(?:an?\s+)?[^.;]{0,80}\bseries\b",
        "team": r"\b(?:is|was|are|were)\s+(?:an?\s+)?[^.;]{0,80}\b(?:team|club)\b",
        "organization": r"\b(?:is|was|are|were)\s+(?:an?\s+)?[^.;]{0,80}\b(?:band|company|organization|university)\b",
    }
    for category, pattern in definitional_patterns.items():
        if re.search(pattern, lead):
            categories.add(category)
    if re.search(r"\b(?:born|was born)\b", lead) and re.search(
        r"\b(?:actor|actress|artist|author|composer|director|person|singer|writer)\b",
        lead,
    ):
        categories.add("person")
    return categories


def _canonical_entity_surface_for_binding(
    relation_index: Mapping[str, Any],
    entity_key: str,
    *,
    equivalent_keys: Optional[Iterable[str]] = None,
    fallback: str = "",
) -> str:
    """Resolve a query span to the fact store's canonical endpoint surface."""
    accepted_keys = {
        str(value or "").strip()
        for value in (equivalent_keys or [])
        if str(value or "").strip()
    }
    accepted_keys.add(str(entity_key or "").strip())
    canonical_surfaces = relation_index.get("canonical_entity_surfaces") or {}
    preferred = str(canonical_surfaces.get(entity_key) or "").strip()
    if preferred:
        return preferred

    items = relation_index.get("items") or []
    subject_index = relation_index.get("subject_to_items") or {}
    object_index = relation_index.get("object_to_items") or {}
    surface_counts: Counter = Counter()
    for key in accepted_keys:
        for item_id in dict.fromkeys(
            list(subject_index.get(key) or [])
            + list(object_index.get(key) or [])
        ):
            try:
                item = items[int(item_id)]
            except (IndexError, TypeError, ValueError):
                continue
            if str(item.get("subject_key") or "") in accepted_keys:
                surface_counts[str(item.get("subject") or "").strip()] += 1
            if str(item.get("object_key") or "") in accepted_keys:
                surface_counts[str(item.get("object") or "").strip()] += 1
    surface_counts.pop("", None)
    if surface_counts:
        return max(
            surface_counts,
            key=lambda surface: (
                int(surface_counts[surface]),
                len(_surface_word_tokens(surface)),
                len(surface),
                surface,
            ),
        )
    return str(fallback or "").strip()


def _bind_explicit_query_entities(
    question: str,
    relation_index: Dict[str, Any],
    *,
    operator: Optional[Dict[str, Any]] = None,
    expected_count: Optional[int] = None,
    max_bindings: int = 4,
    require_operator_match: bool = True,
    canonicalize_surface: bool = True,
    enforce_root_contract: bool = True,
    use_title_entity_keys: bool = False,
    enable_collision_aware_title_identity: bool = False,
    enable_safe_title_family_query_span: bool = True,
    expected_root_categories: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Bind longest non-overlapping question mentions to exact fact keys.

    Endpoint-driven substring matching made short qualifiers such as
    ``Patiala`` compete with the actual roots ``St. Mary's School`` and
    ``Apeejay Education Society``.  V3 instead enumerates question spans,
    keeps only spans backed by the fact indexes, and gives longer mentions
    priority before enforcing the requested root arity.
    """
    matches = list(re.finditer(r"[A-Za-z0-9À-ÖØ-öø-ÿ]+", str(question or "")))
    if not matches:
        return []
    subject_index = relation_index.get("subject_to_items") or {}
    object_index = relation_index.get("object_to_items") or {}
    alias_equivalents = dict(
        relation_index.get("alias_equivalent_keys") or {}
    )
    if use_title_entity_keys:
        for key, values in (
            relation_index.get("title_alias_equivalent_keys") or {}
        ).items():
            alias_equivalents.setdefault(key, [])
            alias_equivalents[key] = list(dict.fromkeys(
                list(alias_equivalents[key] or []) + list(values or [])
            ))
    indexed_keys = set(subject_index.keys())
    indexed_keys.update(object_index.keys())
    indexed_keys.update(alias_equivalents.keys())
    title_entity_surfaces = (
        relation_index.get("title_entity_surfaces") or {}
        if use_title_entity_keys else {}
    )
    title_entity_doc_ids = (
        relation_index.get("title_entity_doc_ids") or {}
        if use_title_entity_keys else {}
    )
    title_identity_surfaces = (
        relation_index.get("title_identity_surfaces") or {}
        if use_title_entity_keys and enable_collision_aware_title_identity
        else {}
    )
    title_identity_doc_ids = (
        relation_index.get("title_identity_doc_ids") or {}
        if use_title_entity_keys and enable_collision_aware_title_identity
        else {}
    )
    title_key_to_family_key = (
        relation_index.get("title_key_to_family_key") or {}
        if use_title_entity_keys else {}
    )
    title_family_doc_ids = (
        relation_index.get("title_family_doc_ids") or {}
        if use_title_entity_keys else {}
    )
    title_family_qualifiers = (
        relation_index.get("title_family_qualifiers") or {}
        if use_title_entity_keys else {}
    )
    title_doc_categories = (
        relation_index.get("title_doc_categories") or {}
        if use_title_entity_keys else {}
    )
    expected_categories = canonical_root_entity_categories(
        expected_root_categories or []
    )
    indexed_keys.update(title_entity_surfaces.keys())






    if enable_safe_title_family_query_span:
        indexed_keys.update(title_family_doc_ids.keys())
    if not indexed_keys:
        return []

    generic_tokens = set(GENERIC_ANCHOR_STOPWORDS) | set(
        GENERIC_RESIDUAL_RELATION_TOKENS
    )
    generic_tokens.update({
        "famous", "many", "much", "name", "number", "place", "time",
        "times", "type", "year",
    })
    for terms in RELATION_FAMILY_TERMS.values():
        generic_tokens.update(terms)
    schema_phrase_tokens = generic_tokens | {
        "book", "city", "country", "date", "director", "entity", "film",
        "location", "movie", "person", "record", "song", "state",
        "work", "writer",
    }
    title_backed_entity_keys = set(
        relation_index.get("title_backed_entity_keys") or []
    )
    candidates: List[Dict[str, Any]] = []




    max_width = min(24, len(matches))
    for start in range(len(matches)):
        for width in range(1, min(max_width, len(matches) - start) + 1):
            end = start + width
            surface = str(question or "")[
                matches[start].start():matches[end - 1].end()
            ]
            key = _entity_key(surface)
            identity_key = canonical_entity_identity_key(surface)
            if not key or key not in indexed_keys:
                continue
            key_tokens = set(key.split())
            if len(key_tokens) == 1:
                raw_token = matches[start].group(0)
                if (
                    next(iter(key_tokens)) in generic_tokens
                    or not raw_token[:1].isupper()
                ):
                    continue
            equivalent_keys = [key] + list(alias_equivalents.get(key) or [])
            resolved_key = max(
                dict.fromkeys(equivalent_keys),
                key=lambda value: (
                    len(subject_index.get(value) or [])
                    + len(object_index.get(value) or []),
                    value == key,
                    value,
                ),
            )
            alias_resolved = resolved_key != key
            incident_ids = list(subject_index.get(resolved_key) or [])
            incident_ids.extend(object_index.get(resolved_key) or [])
            incident_items = relation_index.get("items") or []
            operator_match_count = 0
            if operator:
                operator_match_count = sum(
                    1
                    for item_id in dict.fromkeys(incident_ids)
                    if 0 <= int(item_id) < len(incident_items)
                    and _oriented_endpoint_for_operator_binding(
                        incident_items[int(item_id)],
                        operator,
                        resolved_key,
                        equivalent_keys=set(equivalent_keys),
                    ) is not None
                )
                if require_operator_match and operator_match_count <= 0:
                    continue
            collision_aware_title_exact = bool(
                identity_key and identity_key in title_identity_surfaces
            )
            legacy_title_doc_ids = list(dict.fromkeys(
                int(doc_id)
                for doc_id in (
                    title_entity_doc_ids.get(key)
                    or title_entity_doc_ids.get(resolved_key)
                    or []
                )
                if str(doc_id).lstrip("-").isdigit()
            ))
            canonical_title_family_key = str(
                title_key_to_family_key.get(key)
                or title_key_to_family_key.get(resolved_key)
                or ""
            )
            canonical_title_family_doc_ids = list(dict.fromkeys(
                int(doc_id)
                for doc_id in title_family_doc_ids.get(
                    canonical_title_family_key, []
                )
                if str(doc_id).lstrip("-").isdigit()
            ))
            family_qualifiers = list(
                title_family_qualifiers.get(canonical_title_family_key) or []
            )
            category_partition_base_doc_ids = list(dict.fromkeys(
                canonical_title_family_doc_ids or legacy_title_doc_ids
            ))
            known_title_categories = {
                str(category)
                for doc_id in category_partition_base_doc_ids
                for category in title_doc_categories.get(
                    int(doc_id), title_doc_categories.get(str(doc_id), [])
                )
                if str(category)
            }
            category_partition_doc_ids = [
                int(doc_id)
                for doc_id in category_partition_base_doc_ids
                if expected_categories.intersection({
                    str(category)
                    for category in title_doc_categories.get(
                        int(doc_id), title_doc_categories.get(str(doc_id), [])
                    )
                    if str(category)
                })
            ]
            category_partition_match = bool(
                expected_categories and category_partition_doc_ids
            )
            category_partition_conflict = bool(
                expected_categories
                and known_title_categories
                and not category_partition_match
            )
            title_family_query_span = bool(
                enable_safe_title_family_query_span
                and
                use_title_entity_keys
                and key in title_family_doc_ids
            )
            safe_title_family_query_span = bool(
                title_family_query_span
                and (
                    len(set(canonical_title_family_doc_ids)) == 1
                    or (
                        category_partition_match
                        and len(set(category_partition_doc_ids)) == 1
                    )
                )
            )
            direct_title_catalog_span = bool(
                use_title_entity_keys
                and (
                    key in title_entity_surfaces
                    or resolved_key in title_entity_surfaces
                )
            )









            collision_safe_legacy_fallback = bool(
                enable_collision_aware_title_identity
                and not collision_aware_title_exact
                and use_title_entity_keys
                and int(expected_count or 0) == 1
                and (
                    key in title_entity_surfaces
                    or resolved_key in title_entity_surfaces
                )
                and len(legacy_title_doc_ids) == 1
                and len(canonical_title_family_doc_ids) == 1
                and legacy_title_doc_ids == canonical_title_family_doc_ids
            )
            legacy_title_resolution_allowed = bool(
                not enable_collision_aware_title_identity
                or collision_safe_legacy_fallback
            )
            catalog_surface = (
                title_identity_surfaces.get(identity_key)
                if collision_aware_title_exact
                else (
                    title_entity_surfaces.get(key)
                    or title_entity_surfaces.get(resolved_key)
                    or surface
                )
                if safe_title_family_query_span
                else (
                    title_entity_surfaces.get(key)
                    or title_entity_surfaces.get(resolved_key)
                )
                if legacy_title_resolution_allowed
                else None
            )
            canonical_surface = (
                str(
                    catalog_surface
                    or _canonical_entity_surface_for_binding(
                        relation_index,
                        resolved_key,
                        equivalent_keys=equivalent_keys,
                        fallback=surface,
                    )
                ).strip()
                if canonicalize_surface else surface
            )
            schema_phrase = bool(
                _is_query_slot_phrase(surface)
                or (
                    key_tokens
                    and key_tokens.issubset(schema_phrase_tokens)
                    and key_tokens & generic_tokens
                    and len(key_tokens) <= 3
                )
            )
            title_backed = bool(
                resolved_key in title_backed_entity_keys
                or key in title_backed_entity_keys
                or resolved_key in title_entity_surfaces
                or key in title_entity_surfaces
                or safe_title_family_query_span
            )
            title_catalog_exact = bool(
                (
                    collision_aware_title_exact
                    or safe_title_family_query_span
                    or collision_safe_legacy_fallback
                )
                if enable_collision_aware_title_identity
                else use_title_entity_keys and key in title_entity_surfaces
            )
            canonical_title_doc_ids = list(dict.fromkeys(
                int(doc_id)
                for doc_id in (
                    title_identity_doc_ids.get(identity_key)
                    if collision_aware_title_exact
                    else (
                        category_partition_doc_ids
                        if category_partition_match
                        else canonical_title_family_doc_ids
                    )
                    if safe_title_family_query_span
                    else legacy_title_doc_ids
                    if legacy_title_resolution_allowed
                    else []
                )
                if str(doc_id).lstrip("-").isdigit()
            ))
            span_start = matches[start].start()
            starts_inside_parenthetical = any(
                match.start() < span_start < match.end()
                for match in re.finditer(r"\([^)]*\)", str(question or ""))
            )
            qualified_title_root = bool(
                title_catalog_exact and not starts_inside_parenthetical
            )
            named_title_override = bool(
                schema_phrase
                and title_backed
                and operator_match_count > 0
                and (
                    not expected_categories or category_partition_match
                )
            )
            connector_spanning = bool(
                re.search(r"\b(?:and|or|versus|vs)\b", surface, re.IGNORECASE)
            )
            connector_internal_to_surface = bool(re.search(
                r"[A-Za-z0-9]\s+(?:and|or|versus|vs)\s+[A-Za-z0-9]",
                surface,
                re.IGNORECASE,
            ))
            right_context = str(question or "")[matches[end - 1].end():]
            connector_continuation = bool(re.match(
                r"\s+(?:and|or|versus|vs)\s+[A-Z0-9]",
                right_context,
            ))
            single_root_connector_conflict = bool(
                expected_count is not None
                and int(expected_count) == 1
                and (


                    connector_continuation



                    or (
                        connector_spanning
                        and not (
                            title_catalog_exact
                            and connector_internal_to_surface
                        )
                    )
                )
            )
            root_contract_valid = bool(
                (
                    _query_root_surface_match(question, surface)
                    or qualified_title_root
                )
                and (not schema_phrase or named_title_override)
                and not single_root_connector_conflict
                and not category_partition_conflict
            )
            binding_key = resolved_key
            if title_catalog_exact and canonical_title_family_key:
                canonical_surface_key = _entity_key(canonical_surface)
                if canonical_surface_key == canonical_title_family_key:





                    binding_key = canonical_surface_key
            candidates.append({
                "surface": canonical_surface,
                "mention_surface": surface,
                "canonical_surface": canonical_surface,
                "surface_canonicalized": canonical_surface.casefold() != surface.casefold(),
                "connector_spanning": connector_spanning,
                "connector_continuation": connector_continuation,
                "connector_internal_to_surface": connector_internal_to_surface,
                "mention_key": key,
                "key": binding_key,
                "token_start": start,
                "token_end": end,
                "token_width": width,
                "key_width": len(key_tokens),
                "binding_method": (
                    "ordered_title_identity_query_span"
                    if collision_aware_title_exact else
                    "title_catalog_query_span"
                    if direct_title_catalog_span else
                    "unique_title_family_query_span"
                    if safe_title_family_query_span else
                    "unique_title_family_fallback_query_span"
                    if collision_safe_legacy_fallback else
                    "title_catalog_query_span"
                    if title_catalog_exact else
                    "provenance_alias_query_span"
                    if alias_resolved else "longest_exact_query_span"
                ),
                "binding_confidence": (
                    1.0 if collision_aware_title_exact else
                    1.0 if direct_title_catalog_span else
                    0.99 if safe_title_family_query_span else
                    0.98 if collision_safe_legacy_fallback else
                    1.0 if title_catalog_exact else
                    0.92 if alias_resolved else 1.0
                ),
                "operator_match_count": int(operator_match_count),
                "operator_match_required": bool(require_operator_match),
                "schema_phrase": schema_phrase,
                "title_backed": title_backed,
                "title_catalog_exact": title_catalog_exact,
                "collision_aware_title_identity": bool(
                    collision_aware_title_exact
                ),
                "collision_safe_legacy_fallback": bool(
                    collision_safe_legacy_fallback
                ),
                "title_family_query_span": bool(title_family_query_span),
                "safe_title_family_query_span": bool(
                    safe_title_family_query_span
                ),
                "title_identity_resolution_mode": (
                    "ordered_identity"
                    if collision_aware_title_exact else
                    "legacy_title_key"
                    if direct_title_catalog_span else
                    "unique_or_category_resolved_title_family"
                    if safe_title_family_query_span else
                    "unique_legacy_family_fallback"
                    if collision_safe_legacy_fallback else
                    "legacy_title_key"
                    if title_catalog_exact else "none"
                ),
                "title_identity_key": identity_key,
                "canonical_title_doc_ids": canonical_title_doc_ids,
                "canonical_title_doc_count": len(canonical_title_doc_ids),
                "canonical_title_family_key": canonical_title_family_key,
                "canonical_title_family_doc_ids": canonical_title_family_doc_ids,
                "canonical_title_family_doc_count": len(
                    canonical_title_family_doc_ids
                ),
                "canonical_title_family_qualifiers": family_qualifiers,
                "expected_root_categories": sorted(expected_categories),
                "root_category_partition_doc_ids": list(dict.fromkeys(
                    category_partition_doc_ids
                )),
                "root_category_partition_doc_count": len(set(
                    category_partition_doc_ids
                )),
                "root_category_partition_categories": sorted(
                    expected_categories & known_title_categories
                ),
                "root_category_partition_validated": bool(
                    category_partition_match
                ),
                "root_category_partition_unambiguous": bool(
                    category_partition_match
                    and len(set(category_partition_doc_ids)) == 1
                ),
                "root_category_partition_status": (
                    "MATCHED" if category_partition_match else
                    "MISMATCH" if category_partition_conflict else
                    "UNRESOLVED" if expected_categories else
                    "NOT_REQUIRED"
                ),
                "canonical_title_family_unambiguous": bool(
                    canonical_title_family_key
                    and len(family_qualifiers) <= 1
                ),
                "root_contract_valid": root_contract_valid,
                "root_contract_reason": (
                    "instance_query_mention"
                    if root_contract_valid and not named_title_override else
                    "title_instance_with_operator_edge"
                    if root_contract_valid else
                    "query_slot_phrase"
                    if schema_phrase else
                    "root_category_mismatch"
                    if category_partition_conflict else
                    "connector_spanning_single_root"
                    if single_root_connector_conflict else
                    "not_explicit_query_instance"
                ),
                "instance_binding_score": round(
                    (1.0 if operator_match_count > 0 else 0.0)
                    + (0.30 if title_backed else 0.0)
                    + (0.20 if root_contract_valid else 0.0)
                    - (0.80 if schema_phrase else 0.0),
                    4,
                ),
            })

    if expected_count is not None and int(expected_count) > 1:
        candidates.sort(key=lambda item: (





            int(bool(item.get("connector_spanning"))),
            -int(bool(item.get("root_contract_valid"))),
            -int(bool(item.get("root_category_partition_validated"))),
            -int(bool(item.get("title_catalog_exact"))),
            -int(item["key_width"]),
            -int(item["token_width"]),
            -int(item.get("operator_match_count", 0) or 0),
            int(item["token_start"]),
            str(item["key"]),
        ))
    elif (
        canonicalize_surface
        and operator
        and not require_operator_match
        and enable_collision_aware_title_identity
    ):
        candidates.sort(key=lambda item: (
            -int(bool(item.get("root_contract_valid"))),
            -int(bool(item.get("root_category_partition_validated"))),
            -int(bool(item.get("title_catalog_exact"))),
            int(bool(item.get("schema_phrase"))),
            -int(item["key_width"]),
            -int(bool(item.get("collision_aware_title_identity"))),




            abs(int(item["token_width"]) - int(item["key_width"])),
            -int(item["token_width"]),
            -int(int(item.get("operator_match_count", 0) or 0) > 0),
            -int(bool(item.get("title_backed"))),
            int(item["token_start"]),
            str(item["key"]),
        ))
    elif canonicalize_surface and operator and not require_operator_match:
        candidates.sort(key=lambda item: (




            -int(bool(item.get("root_contract_valid"))),
            -int(bool(item.get("root_category_partition_validated"))),
            -int(bool(item.get("title_catalog_exact"))),
            -int(int(item.get("operator_match_count", 0) or 0) > 0),
            int(bool(item.get("schema_phrase"))),
            -int(bool(item.get("title_backed"))),
            -int(item["key_width"]),
            abs(int(item["token_width"]) - int(item["key_width"])),
            -int(item["token_width"]),
            -int(item.get("operator_match_count", 0) or 0),
            int(item["token_start"]),
            str(item["key"]),
        ))
    else:
        candidates.sort(key=lambda item: (


            -int(item["key_width"]),
            -int(item["token_width"]),
            -int(item.get("operator_match_count", 0) or 0),
            int(item["token_start"]),
            str(item["key"]),
        ))
    selection_candidates = candidates
    if enforce_root_contract:
        selection_candidates = [
            candidate for candidate in candidates
            if bool(candidate.get("root_contract_valid"))
        ]
    selected: List[Dict[str, Any]] = []
    occupied: Set[int] = set()
    seen_keys: Set[str] = set()
    target = max(1, int(expected_count or max_bindings))
    for candidate in selection_candidates:
        span = set(range(
            int(candidate["token_start"]), int(candidate["token_end"])
        ))
        if candidate["key"] in seen_keys or span & occupied:
            continue
        selected.append(candidate)
        seen_keys.add(str(candidate["key"]))
        occupied.update(span)
        if len(selected) >= min(target, max(1, int(max_bindings))):
            break
    selected.sort(key=lambda item: int(item["token_start"]))
    if expected_count is not None and len(selected) != int(expected_count):
        return []
    return selected


def _entity_type(entity: Any) -> str:
    text = str(entity or "").strip().lower()
    if not text:
        return "unknown"
    if DATE_VALUE_RE.search(text) or SURFACE_MONTH_RE.search(text):
        return "date"
    if re.fullmatch(r"[\d,.\-/% ]+", text):
        return "number"
    tokens = _tokens(text)
    if tokens & NATIONALITY_TERMS:
        return "nationality"
    if tokens & PLACE_TERMS:
        return "place"
    if tokens & ORG_TERMS:
        return "organization"
    if tokens & WORK_TERMS:
        return "work"
    return "entity"


def _relation_key(relation: Any) -> str:
    relation_tokens = [
        token
        for token in TOKEN_RE.findall(str(relation or "").lower())
        if token not in STOPWORDS and len(token) > 1
    ]
    return " ".join(relation_tokens[:8])


def canonical_fact_tuple_key(fact: Any) -> str:
    """Return the exact normalized tuple key used by the provenance index."""
    if not isinstance(fact, (list, tuple)) or len(fact) != 3:
        return ""
    subject, relation, obj = fact
    keys = (_entity_key(subject), _relation_key(relation), _entity_key(obj))
    if not all(keys):
        return ""
    return "|".join(keys)


def _title_family_parts(title: Any) -> Tuple[str, Tuple[str, ...]]:
    """Return a stable title family and optional disambiguating qualifiers.

    Passage titles such as ``Chris Brown (singer)`` and ``Chris Brown - Early
    life`` should share an entity family without being treated as the same raw
    document.  The family key supports provenance partitions; qualifiers remain
    available so query-time code can reject ambiguous same-name partitions.
    """
    text = re.sub(r"\s+", " ", str(title or "").strip())
    if not text:
        return "", ()
    qualifiers = tuple(
        key
        for key in (_entity_key(value) for value in re.findall(r"\(([^()]*)\)", text))
        if key
    )
    base = re.sub(r"\s*\([^()]*\)\s*", " ", text).strip()
    base = re.split(r"\s+(?:-|–|—)\s+", base, maxsplit=1)[0].strip()
    return _entity_key(base), tuple(dict.fromkeys(qualifiers))


def _schema_signature(triple: Tuple[Any, Any, Any]) -> Optional[Tuple[str, str, str]]:
    if not isinstance(triple, (list, tuple)) or len(triple) != 3:
        return None
    relation = _relation_key(triple[1])
    if not relation:
        return None
    return (_entity_type(triple[0]), relation, _entity_type(triple[2]))


PREDICATE_SEMANTIC_CONTRACT_VERSION = "predicate_semantic_contract_v1"
_DISTRIBUTION_RELEASE_CUES = {
    "release", "released", "releases", "releasing",
}
_DISTRIBUTION_PREMIERE_CUES = {
    "debut", "debuted", "premiere", "premiered", "premieres",
}
_DISTRIBUTION_INTENT_CUES = {
    "anticipated", "expected", "intended", "planned", "proposed",
    "scheduled", "slated",
}
_DISTRIBUTION_CONFLICT_CUES = {
    "alternative", "conflicting", "disputed", "reported", "uncertain",
}


def _distribution_semantic_subtypes(
    relation: Any,
    source_text: Any = "",
) -> Tuple[str, ...]:
    """Return release-time semantics without conflating plans with events."""
    relation_tokens = set(TOKEN_RE.findall(str(relation or "").lower()))
    context_tokens = set(TOKEN_RE.findall(str(source_text or "").lower()))
    tokens = relation_tokens | context_tokens
    has_release = bool(relation_tokens & _DISTRIBUTION_RELEASE_CUES)
    has_premiere = bool(relation_tokens & _DISTRIBUTION_PREMIERE_CUES)
    if not (has_release or has_premiere):
        return ()
    if tokens & _DISTRIBUTION_INTENT_CUES:
        return ("distribution_intended_release",)
    if tokens & _DISTRIBUTION_CONFLICT_CUES:
        return ("distribution_conflicting_release",)
    if has_premiere:
        return ("distribution_premiere",)
    return ("distribution_actual_release",)


def canonical_schema_predicate_families(relation: Any) -> Tuple[str, ...]:
    """Return selective, deterministic predicate families for Schema Memory.

    Known aliases share a family (``born`` and ``birthplace``); uncovered
    predicates retain an exact normalized family instead of being discarded.
    This makes catalog coverage measurable without merging unrelated roles.
    """
    relation_text = " ".join(TOKEN_RE.findall(str(relation or "").lower()))
    relation_tokens = _tokens(relation)




    if re.search(
        r"\b(?:also known as|alias(?:ed)? (?:as|to)|"
        r"(?:artistic|stage|pen|real|birth) name (?:is |was )?(?:of|for)|"
        r"is the (?:artistic|stage|pen|real|birth) name of|"
        r"was the (?:artistic|stage|pen|real|birth) name of)\b",
        relation_text,
    ):
        return ("identity",)



    if re.search(
        r"\b(?:(?:originally\s+)?(?:performed|recorded|sung)\s+by|"
        r"(?:is\s+)?(?:a\s+)?song\s+by)\b",
        relation_text,
    ):
        return ("performance",)
    families = {
        family
        for family, terms in SCHEMA_PREDICATE_FAMILY_TERMS.items()
        if relation_tokens & terms
    }
    if _distribution_semantic_subtypes(relation):
        families.add("distribution")
    if families:
        return tuple(sorted(families))
    relation_key = _relation_key(relation)
    return (f"relation:{relation_key}",) if relation_key else ()


def canonical_schema_predicate_subtypes(
    relation: Any,
    value: Any = "",
) -> Tuple[str, ...]:
    """Return terminal-safe predicate subtypes for physical fact access.

    Schema families intentionally merge aliases for recall, but ``birth`` and
    ``death`` are too broad for final-answer execution: place, date, name, and
    cause facts are not interchangeable.  This second key keeps family recall
    while giving terminal operators an exact attribute-level access path.
    """

    relation_text = " ".join(TOKEN_RE.findall(str(relation or "").lower()))
    families = set(canonical_schema_predicate_families(relation))
    value_text = str(value or "")
    subtypes: Set[str] = set()

    if "birth" in families:
        if re.search(r"\b(?:born as|birth name|name at birth|nee)\b", relation_text):
            subtypes.add("birth_name")
        elif re.search(
            r"\b(?:born on|date of birth|birth date|birthday)\b",
            relation_text,
        ):
            subtypes.add("birth_date")
        elif re.search(
            r"\b(?:born in|born at|birthplace|birth place|place of birth|"
            r"native city|native town|native place)\b",
            relation_text,
        ):
            subtypes.add("birth_place")
        elif DATE_VALUE_RE.search(value_text) or SURFACE_MONTH_RE.search(value_text):
            subtypes.add("birth_date")

    if "death" in families:
        if re.search(
            r"\b(?:died of|died from|cause of death|death cause|killed by)\b",
            relation_text,
        ):
            subtypes.add("death_cause")
        elif re.search(r"\b(?:died on|date of death|death date)\b", relation_text):
            subtypes.add("death_date")
        elif re.search(
            r"\b(?:died in|died at|place of death|death place)\b",
            relation_text,
        ):
            subtypes.add("death_place")
        elif DATE_VALUE_RE.search(value_text) or SURFACE_MONTH_RE.search(value_text):
            subtypes.add("death_date")

    if "employment" in families:



        if re.search(
            r"\b(?:worked|works|working|served|serves|serving)\s+as\b|"
            r"\b(?:occupation|profession|job|career)\b",
            relation_text,
        ):
            subtypes.add("occupation_role")
        elif re.search(
            r"\b(?:worked|works|working|employed|served|serves|serving)\s+"
            r"(?:for|by)\b",
            relation_text,
        ):
            subtypes.add("employment_organization")
        elif re.search(
            r"\b(?:worked|works|working|employed)\s+(?:in|at)\b",
            relation_text,
        ):
            subtypes.add("employment_location")
        else:
            subtypes.add("employment_generic")

    distribution_subtypes = set(_distribution_semantic_subtypes(relation))
    subtypes.update(distribution_subtypes)

    residual_families = families - {"birth", "death"}
    if families & {"birth", "death"}:
        residual_families -= {"geopolitical_location"}
    for family in residual_families:
        subtypes.add(family)
    return tuple(sorted(subtypes))


def canonical_predicate_contract(
    relation: Any,
    value: Any = "",
    source_text: Any = "",
) -> Dict[str, Any]:
    """Compile the shared index/admission predicate representation.

    The raw predicate remains available for provenance and auditing.  Families
    are physical access keys; semantic subtypes retain distinctions that a
    family key deliberately merges for recall.
    """
    semantic_subtypes = set(canonical_schema_predicate_subtypes(
        relation, value
    ))
    contextual_distribution = set(_distribution_semantic_subtypes(
        relation, source_text
    ))
    if contextual_distribution:
        semantic_subtypes = {
            value for value in semantic_subtypes
            if not str(value).startswith("distribution_")
        }
        semantic_subtypes.update(contextual_distribution)
    return {
        "version": PREDICATE_SEMANTIC_CONTRACT_VERSION,
        "raw_relation": str(relation or "").strip(),
        "canonical_families": canonical_schema_predicate_families(relation),
        "semantic_subtypes": tuple(sorted(semantic_subtypes)),
    }


def canonical_operator_predicate_contract(
    relation_tokens: Iterable[str],
    predicate_families: Iterable[str] = (),
) -> Dict[str, Any]:
    """Compile allowed predicate semantics for one physical obligation."""
    tokens = {
        str(token or "").strip().lower()
        for token in relation_tokens
        if str(token or "").strip()
    }
    families = {
        str(value or "").strip()
        for value in predicate_families
        if str(value or "").strip()
    }
    allowed_subtypes: Set[str] = set()
    if families & {"distribution"}:
        if tokens & _DISTRIBUTION_INTENT_CUES and tokens & (
            _DISTRIBUTION_RELEASE_CUES | _DISTRIBUTION_PREMIERE_CUES
        ):
            allowed_subtypes.add("distribution_intended_release")
        else:
            if tokens & _DISTRIBUTION_RELEASE_CUES:
                allowed_subtypes.add("distribution_actual_release")
            if tokens & _DISTRIBUTION_PREMIERE_CUES:
                allowed_subtypes.add("distribution_premiere")
    return {
        "version": PREDICATE_SEMANTIC_CONTRACT_VERSION,
        "relation_tokens": tuple(sorted(tokens)),
        "canonical_families": tuple(sorted(families)),
        "allowed_semantic_subtypes": tuple(sorted(allowed_subtypes)),
    }


def _canonical_operator_role_contract(
    relation_tokens: Iterable[str],
) -> Dict[str, Any]:
    """Compile a query relation into an oriented endpoint contract.

    The contract describes traversal from the entity already bound by the
    query to the entity produced by the operator.  It is intentionally more
    selective than the broad ``kinship`` family.
    """
    tokens = {str(token or "").strip().lower() for token in relation_tokens}
    specs = (
        ("mother", "mother_of", "child", "mother"),
        ("maternal", "mother_of", "child", "mother"),
        ("father", "father_of", "child", "father"),
        ("paternal", "father_of", "child", "father"),
        ("daughter", "daughter_of", "parent", "daughter"),
        ("son", "son_of", "parent", "son"),
        ("parent", "parent_of", "child", "parent"),
        ("child", "child_of", "parent", "child"),
        ("wife", "wife_of", "spouse", "wife"),
        ("husband", "husband_of", "spouse", "husband"),
        ("spouse", "spouse_of", "spouse", "spouse"),
    )
    for cue, canonical_relation, input_role, output_role in specs:
        if cue in tokens:
            return {
                "canonical_relation": canonical_relation,
                "input_role": input_role,
                "output_role": output_role,
                "role_direction_required": True,
                "role_contract_version": "role_directed_typed_join_v1",
            }
    return {
        "canonical_relation": "",
        "input_role": "entity",
        "output_role": "entity",
        "role_direction_required": False,
        "role_contract_version": "role_directed_typed_join_v1",
    }


def _operator_requires_role_direction(operator: Mapping[str, Any]) -> bool:
    if "role_direction_required" in operator:
        return bool(operator.get("role_direction_required"))
    return bool(
        _canonical_operator_role_contract(
            operator.get("relation_tokens") or []
        ).get("role_direction_required")
    )


def _canonical_fact_role_relation(item: Mapping[str, Any]) -> str:
    """Return the directional relation expressed by one OpenIE fact."""
    explicit = str(item.get("canonical_role_relation") or "").strip().lower()
    if explicit:
        return explicit
    tokens = set(item.get("relation_tokens") or [])
    if tokens & {"mother", "maternal"}:
        return "mother_of"
    if tokens & {"father", "paternal"}:
        return "father_of"
    if "daughter" in tokens:
        return "daughter_of"
    if "son" in tokens:
        return "son_of"
    if "parent" in tokens:
        return "parent_of"
    if tokens & {"child", "children"}:
        return "child_of"
    if "wife" in tokens:
        return "wife_of"
    if "husband" in tokens:
        return "husband_of"
    if "spouse" in tokens:
        return "spouse_of"
    if tokens & {"married", "marriage", "partner", "consort"}:
        return "spouse_of"
    return ""


def _role_orientations_for_operator_item(
    item: Mapping[str, Any],
    operator: Mapping[str, Any],
) -> List[Tuple[str, str, str]]:
    """Return valid ``(input_side, output_side, fact_relation)`` traversals."""
    requested = str(operator.get("canonical_relation") or "").strip().lower()
    if not requested:
        requested = str(
            _canonical_operator_role_contract(
                operator.get("relation_tokens") or []
            ).get("canonical_relation")
            or ""
        )
    if not requested:
        return []
    fact_relation = _canonical_fact_role_relation(item)
    parent_facts = {"mother_of", "father_of", "parent_of"}
    child_facts = {"daughter_of", "son_of", "child_of"}

    if requested == "mother_of" and fact_relation == "mother_of":
        return [("object", "subject", fact_relation)]
    if requested == "father_of" and fact_relation == "father_of":
        return [("object", "subject", fact_relation)]
    if requested == "parent_of":
        if fact_relation in parent_facts:
            return [("object", "subject", fact_relation)]
        if fact_relation in child_facts:
            return [("subject", "object", fact_relation)]
    if requested == "daughter_of" and fact_relation == "daughter_of":
        return [("object", "subject", fact_relation)]
    if requested == "son_of" and fact_relation == "son_of":
        return [("object", "subject", fact_relation)]
    if requested == "child_of":
        if fact_relation in parent_facts:
            return [("subject", "object", fact_relation)]
        if fact_relation in child_facts:
            return [("object", "subject", fact_relation)]
    if requested == "wife_of":
        if fact_relation == "wife_of":
            return [("object", "subject", fact_relation)]
        if fact_relation == "husband_of":
            return [("subject", "object", fact_relation)]
        if fact_relation == "spouse_of":




            return [
                ("subject", "object", fact_relation),
                ("object", "subject", fact_relation),
            ]
    if requested == "husband_of":
        if fact_relation == "husband_of":
            return [("object", "subject", fact_relation)]
        if fact_relation == "wife_of":
            return [("subject", "object", fact_relation)]
        if fact_relation == "spouse_of":
            return [
                ("subject", "object", fact_relation),
                ("object", "subject", fact_relation),
            ]
    if requested == "spouse_of" and fact_relation in {
        "spouse_of", "wife_of", "husband_of",
    }:
        return [
            ("subject", "object", fact_relation),
            ("object", "subject", fact_relation),
        ]
    return []


def typed_schema_signatures_v2(
    triple: Tuple[Any, Any, Any],
) -> Tuple[Tuple[str, str, str, str], ...]:
    """Compile one fact into directional, typed schema catalog signatures."""
    if not isinstance(triple, (list, tuple)) or len(triple) != 3:
        return ()
    subject_type = _entity_type(triple[0])
    object_type = _entity_type(triple[2])
    return tuple(
        (
            subject_type,
            family,
            object_type,
            "subject_to_object",
        )
        for family in canonical_schema_predicate_families(triple[1])
    )


def _fact_slot_key(label: str, bridge_key: str) -> str:
    label = str(label or "").strip().lower()
    bridge_key = str(bridge_key or "").strip()
    if not label or not bridge_key:
        return ""
    return f"{label}|{bridge_key}"


def _fact_slot_answer_key(label: str, bridge_key: str, answer_key: str) -> str:
    slot_key = _fact_slot_key(label, bridge_key)
    answer_key = str(answer_key or "").strip()
    if not slot_key or not answer_key:
        return ""
    return f"{slot_key}|{answer_key}"


def _fact_slot_labels_for_relation_tokens(relation_tokens: Set[str]) -> List[str]:
    labels: List[str] = []
    if relation_tokens & (BIRTH_PLACE_RELATION_TERMS | BIRTH_DATE_RELATION_TERMS):
        labels.append("birth")
    if relation_tokens & {"died", "death"}:
        labels.append("death")
    if relation_tokens & PARENT_RELATION_TERMS or relation_tokens & CHILD_RELATION_TERMS:
        labels.append("parent")
        if relation_tokens & {"father", "paternal"}:
            labels.append("father")
        if relation_tokens & {"mother", "maternal"}:
            labels.append("mother")
    return list(dict.fromkeys(labels))


def _row_fact_slot_labels(row: Dict[str, Any]) -> List[str]:
    template = str(row.get("template") or "")
    labels: List[str] = []
    if template in {"creator_birth_place_2hop", "creator_birth_date_2hop", "birth_place_join", "birth_date_join", "birth_relation_join"}:
        labels.append("birth")
    elif template in {"death_place_join", "death_date_join", "death_relation_join"}:
        labels.append("death")
    elif template in {"kinship_grandparent_2hop", "kinship_inverse_join"}:
        labels.append("parent")
        relation_text = str(row.get("triple") or "").lower()
        if "father" in relation_text or "paternal" in relation_text:
            labels.append("father")
        if "mother" in relation_text or "maternal" in relation_text:
            labels.append("mother")
    return list(dict.fromkeys(labels))


def _safe_rank(doc_idx: int, rank_map: Dict[int, int], fallback_rank: int) -> int:
    try:
        return int(rank_map.get(int(doc_idx), fallback_rank))
    except Exception:
        return fallback_rank


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(TOKEN_RE.findall(str(text or ""))) * 1.25))


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _overlap_over_query(left: Set[str], query_tokens: Set[str]) -> float:
    if not left or not query_tokens:
        return 0.0
    return len(left & query_tokens) / max(1, len(query_tokens))


def _bounded_overlap(left: Set[str], required: Set[str], max_required: int = 3) -> float:
    if not left or not required:
        return 0.0
    return _clamp(len(left & required) / max(1, min(len(required), int(max_required))))


def _entity_bridge_score(left: Set[str], right: Set[str], *, max_required: int = 4) -> float:
    """Measure whether two endpoints likely denote the same bridge entity."""
    if not left or not right:
        return 0.0
    denominator = max(1, min(len(left), len(right), int(max_required)))
    return _clamp(len(left & right) / denominator)


def _relation_profile(question: str) -> Dict[str, Any]:
    q_tokens = _tokens(question)
    relation_tokens: Set[str] = set()
    relation_families: Set[str] = set()
    for family, terms in RELATION_FAMILY_TERMS.items():
        hits = q_tokens & terms
        if hits:
            relation_families.add(family)
            relation_tokens.update(hits)

    polarity: Set[str] = set()
    polarity_tokens: Set[str] = set()
    for name, terms in RELATION_POLARITY_TERMS.items():
        hits = q_tokens & terms
        if hits:
            polarity.add(name)
            polarity_tokens.update(hits)
            relation_tokens.update(hits)

    answer_types: Set[str] = set()
    answer_type_tokens: Set[str] = set()
    lower_question = str(question or "").lower()
    if re.search(r"\bhow many\b|\bnumber of\b", lower_question):
        answer_types.add("number")
    for answer_type, terms in ANSWER_TYPE_TERMS.items():
        hits = q_tokens & terms
        if hits:
            answer_types.add(answer_type)
            answer_type_tokens.update(hits)

    role_and_type_tokens = set(relation_tokens) | set(polarity_tokens) | set(answer_type_tokens)
    for terms in RELATION_FAMILY_TERMS.values():
        role_and_type_tokens.update(terms)
    role_and_type_tokens.update(GENERIC_ANCHOR_STOPWORDS)

    anchor_tokens = {
        token
        for token in q_tokens
        if token not in role_and_type_tokens and not token.isdigit() and len(token) > 2
    }
    return {
        "query_tokens": sorted(q_tokens),
        "relation_tokens": sorted(relation_tokens),
        "relation_families": sorted(relation_families),
        "polarity": sorted(polarity),
        "answer_types": sorted(answer_types),
        "anchor_tokens": sorted(anchor_tokens),
        "has_relation_need": bool(relation_tokens or relation_families or polarity),
    }


def _strict_terminal_answer_type(question: str) -> str:
    """Compile an unambiguous terminal slot from the question surface."""
    text = str(question or "").strip().lower()
    if not text:
        return ""
    if re.search(r"\bhow long\b", text):
        return "duration"
    if re.search(r"\b(?:how many|number of)\b", text):
        return "number"
    if re.match(
        r"^(?:what|which)\b.{0,48}\b(?:salary|salaries|income|earnings|wage|wages|pay)\b",
        text,
    ):
        return "number"
    if re.search(
        r"\b(?:date|year|day|month)\s+of\s+(?:birth|death)\b|"
        r"\b(?:birth\s+date|birthdate|birthday|death\s+date|deathdate)\b",
        text,
    ):
        return "date"
    if re.search(
        r"\b(?:place|city|country|state|location)\s+of\s+"
        r"(?:birth|death)\b|\bbirthplace\b",
        text,
    ):
        return "place"
    if re.search(r"\b(?:nationality|citizenship)\b", text) or re.search(
        r"\b(?:what|which)\s+country\b.*\b(?:from|citizen|national)\b",
        text,
    ):
        return "nationality"
    if re.search(r"\b(?:when|what year|which year|what date|which date)\b", text):
        return "date"
    if re.search(
        r"\b(?:where|what (?:city|country|state|place|location|capital|county)|"
        r"which (?:city|country|state|place|location|capital|county)|"
        r"what is the capital)\b",
        text,
    ):
        return "place"


    if re.search(
        r"\b(?:same|both)\b.*\b(?:city|country|state|place|location|county)\b",
        text,
    ):
        return "place"
    if re.search(
        r"^(?:what|which)\s+(?:film|movie|book|album|song|novel|series|work)\b",
        text,
    ):
        return "work"
    if re.search(r"\b(?:who|whom|whose)\b", text):
        return "person"
    if re.search(
        r"\b(?:what|which) (?:company|organization|team|university|club)\b",
        text,
    ):
        return "organization"
    return ""


def _question_source_span(question: str, pattern: str) -> Dict[str, Any]:
    match = re.search(pattern, str(question or ""), flags=re.IGNORECASE)
    if not match:
        return {"text": "", "start": -1, "end": -1}
    return {
        "text": str(question or "")[match.start():match.end()],
        "start": int(match.start()),
        "end": int(match.end()),
    }


def _branch_comparison_role_label(question: str) -> str:
    """Return a role only when a branch comparison states a work-role path.

    The role must be syntactically connected to a work mention.  This avoids
    treating role words inside film or book titles as query operators while
    covering both ``directors of films X and Y`` and ``films X and Y have the
    directors``.
    """
    text = re.sub(r"\s+", " ", str(question or "")).strip().lower()
    role = r"director|writer|author|composer|producer|creator"
    work = r"films?|movies?|books?|novels?|albums?|songs?|works?"
    patterns = (
        rf"\b(?P<role>{role})s?\s+of\s+(?:(?:the|both)\s+)?(?:{work})\b",
        rf"\b(?:{work})\b.{{0,120}}?\b(?:have|with)\s+"
        rf"(?:the\s+)?(?P<role>{role})s?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return str(match.group("role") or "")
    return ""


_EXACT_CHAIN_ROLE_PATTERN = (
    r"mother|father|parent|child|son|daughter|spouse|husband|wife|director|"
    r"writer|author|composer|producer|creator|performer|singer|painter|artist|"
    r"designer|founder"
)


def _safe_atomic_question_anchor(anchor: str) -> bool:
    """Return whether one surface span is a literal, non-compositional anchor.

    This is a syntax guard, not entity linking.  It deliberately rejects
    relative clauses and role descriptions because compiling those spans as a
    root entity would silently delete a relation from the question.
    """
    value = re.sub(r"\s+", " ", str(anchor or "")).strip(" ?.,")
    if not value or len(value) > 180:
        return False
    if re.search(r"\b(?:who|whom|whose|that|which|where)\b", value):
        return False
    if re.search(
        r"\b(?:person|man|woman|actor|author|composer|director|performer|"
        r"producer|singer|writer|film|movie|book|song|album|city|country|"
        r"organization|company|school|university)\s+(?:from|in|of)\b",
        value,
    ):
        return False
    if re.search(
        r"\b(?:member|owner|founder|capital|headquarters|birthplace|place)\s+of\b",
        value,
    ):
        return False
    if "," in value or re.search(r"\s+and\s+", value):
        return False
    if re.search(
        r"\b(?:acquired|adaptation|adjacent|appears|born|composed|created|"
        r"directed|died|established|founded|headquartered|performed|played|"
        r"produced|recorded|stars?|succeeded|won|winner|written)\b",
        value,
    ):
        return False
    if re.match(
        r"^(?:the\s+)?(?:company|country|film|group|man|movie|organization|"
        r"political\s+party|party|person|place|province|region|seminar|team|"
        r"woman)\b",
        value,
    ):
        return False
    return True


def _safe_role_chain_anchor(anchor: str) -> bool:
    """Accept a literal tail after an explicit all-role dependency chain."""
    value = re.sub(r"\s+", " ", str(anchor or "")).strip(" ?.")
    return bool(
        value
        and len(value) <= 180
        and not re.search(r"\b(?:who|whom|whose|that|which|where)\b", value)
    )


def _exact_role_chain(question: str) -> Tuple[str, ...]:
    """Return all roles in execution order for an exact nested role chain.

    ``mother of composer of Work`` is written outside-in, while execution
    starts at ``Work``.  The returned sequence is therefore
    ``("composer", "mother")``.  Only complete, anchored question templates
    are accepted; free relative clauses remain unsupported.
    """
    text = re.sub(r"\s+", " ", str(question or "")).strip().lower()
    match = re.match(
        r"^who\s+(?:is|was)\s+(?:the\s+)?(?P<body>.+?)\s*\??$",
        text,
    )
    if not match:
        return ()
    body = str(match.group("body") or "").strip(" ?")
    parts = [re.sub(r"^the\s+", "", part.strip()) for part in body.split(" of ")]
    if not 3 <= len(parts) <= 5:
        return ()
    roles = parts[:-1]
    anchor = parts[-1]
    if not all(re.fullmatch(_EXACT_CHAIN_ROLE_PATTERN, role) for role in roles):
        return ()
    if not _safe_role_chain_anchor(anchor):
        return ()


    if roles and all(role == "father" for role in roles):
        return ()
    return tuple(reversed(roles))


def _exact_nested_role_path(question: str) -> Optional[Tuple[str, str]]:
    """Return execution-order roles for an unambiguous ``role of role`` path.

    The returned tuple is ``(inner_role, outer_role)`` because execution starts
    from the named entity at the end of the question.  Relative clauses are
    deliberately excluded: phrases such as ``the actor who played X`` require
    an additional portrayal constraint and cannot safely be lowered to a
    two-edge role path.
    """
    text = re.sub(r"\s+", " ", str(question or "")).strip().lower()
    chain = _exact_role_chain(question)
    if len(chain) == 2:
        return chain[0], chain[1]

    outer = r"mother|father|parent|child|son|daughter|spouse|husband|wife"
    inner = _EXACT_CHAIN_ROLE_PATTERN
    patterns = (
        rf"^who\s+(?:is|was)\s+(?:the\s+)?(?P<outer>{outer})\s+of\s+"
        rf"(?:the\s+)?(?P<inner>{inner})\s+of\s+(?P<anchor>.+?)\s*\??$",
        rf"^who\s+(?:is|was)\s+(?:the\s+)?(?P<outer>{outer})\s+of\s+"
        rf"(?P<anchor>.+?)[\'\u2019]s\s+(?P<inner>{inner})\s*\??$",
    )
    for pattern in patterns:
        match = re.match(pattern, text)
        if not match:
            continue
        anchor = str(match.group("anchor") or "").strip(" ?")
        if not _safe_role_chain_anchor(anchor):
            return None
        return str(match.group("inner") or ""), str(match.group("outer") or "")
    return None


def _exact_atomic_relation_projection(question: str) -> Optional[Dict[str, Any]]:
    """Parse exact one-edge surfaces already supported by LOOKUP_JOIN.

    The matcher is intentionally anchored to the whole question.  It expands
    compiler coverage without treating a relation buried in a relative clause
    as the answer-producing edge.
    """
    text = re.sub(r"\s+", " ", str(question or "")).strip().lower()
    profiles = (
        (
            "location", "geopolitical_location",
            {"based", "headquartered", "located", "situated"}, {"location"},
            "place", (),
            (
                r"^(?:where|in what (?:city|country|state|county|province|"
                r"region|place))\s+(?:is|was|are|were)\s+(?P<anchor>.+?)\s+"
                r"(?:located|based|headquartered|situated)\s*\??$"
            ),
        ),
        (
            "founding_time", "founding",
            {"established", "formed", "founded"}, {"temporal"},
            "date", (),
            r"^when\s+(?:was|is|were)\s+(?P<anchor>.+?)\s+(?:founded|formed|established)\s*\??$",
        ),
        (
            "founder", "founding",
            {"established", "founded", "founder"}, {"creator"},
            "person", (),
            r"^who\s+(?:founded|established)\s+(?P<anchor>.+?)\s*\??$",
        ),
        (
            "leader", "leadership",
            {"chairman", "ceo", "governor", "head", "leader", "president"},
            {"leadership"}, "person", (),
            (
                r"^who\s+(?:is|was)\s+(?:the\s+)?(?:leader|president|governor|"
                r"head|chairman|ceo)\s+of\s+(?P<anchor>.+?)\s*\??$"
            ),
        ),
        (
            "capital", "capital", {"capital"}, {"location"}, "place", (),
            r"^what\s+(?:is|was)\s+(?:the\s+)?capital\s+of\s+(?P<anchor>.+?)\s*\??$",
        ),
        (
            "education", "education",
            {"attend", "attended", "graduate", "graduated", "study", "studied"},
            {"education"}, "organization", (),
            r"^where\s+did\s+(?P<anchor>.+?)\s+(?:study|attend|graduate)\s*\??$",
        ),
        (
            "education", "education",
            {"attend", "attended", "graduate", "graduated", "study", "studied"},
            {"education"}, "organization", (),
            (
                r"^what\s+(?:school|university|college)\s+did\s+(?P<anchor>.+?)\s+"
                r"(?:attend|graduate from|study at)\s*\??$"
            ),
        ),
        (
            "distribution_time", "distribution",
            {"premiere", "premiered", "published", "release", "released"},
            {"temporal"}, "date", (),
            r"^when\s+(?:was|is)\s+(?P<anchor>.+?)\s+(?:released|premiered|published)\s*\??$",
        ),
        (
            "nationality", "nationality",
            {"country", "from", "nationality"}, {"nationality"},
            "nationality", ("nationality",),
            r"^(?:what|which)\s+(?:country|nationality)\s+(?:is|was)\s+(?P<anchor>.+?)\s+from\s*\??$",
        ),
    )
    for label, family, tokens, relation_families, output_type, subtypes, pattern in profiles:
        match = re.match(pattern, text)
        if not match or not _safe_atomic_question_anchor(match.group("anchor")):
            continue
        return {
            "semantic_label": label,
            "predicate_families": {family},
            "relation_tokens": set(tokens),
            "relation_families": set(relation_families),
            "output_type": output_type,
            "required_predicate_subtypes": set(subtypes),
            "source_pattern": pattern,
        }
    return None


def _exact_role_education_path(question: str) -> str:
    """Return a role for exact ``role of entity -> education`` questions."""
    text = re.sub(r"\s+", " ", str(question or "")).strip().lower()
    role = (
        r"director|writer|author|composer|producer|creator|performer|singer|"
        r"painter|artist|designer|founder"
    )
    patterns = (
        rf"^where\s+did\s+(?:the\s+)?(?P<role>{role})\s+of\s+"
        rf"(?P<anchor>.+?)\s+(?:study|attend|graduate)\s*\??$",
        rf"^where\s+was\s+(?:the\s+)?(?P<role>{role})\s+of\s+"
        rf"(?P<anchor>.+?)\s+(?:educated|trained)\s*\??$",
        rf"^what\s+(?:music\s+)?(?:school|university|college)\s+did\s+"
        rf"(?:the\s+)?(?P<role>{role})\s+of\s+(?P<anchor>.+?)\s+"
        rf"(?:attend|graduate from|study at)\s*\??$",
        rf"^(?:the\s+)?(?P<role>{role})\s+of\s+(?P<anchor>.+?)\s+"
        rf"(?:attended|graduated from|studied at)\s+which\s+"
        rf"(?:[a-z]+\s+)?(?:school|university|college)\s*\??$",
    )
    for pattern in patterns:
        match = re.match(pattern, text)
        if not match:
            continue
        anchor = str(match.group("anchor") or "").strip(" ?")
        if anchor and not re.search(r"\b(?:who|that|which|where|whose)\b", anchor):
            return str(match.group("role") or "")
    return ""


def _exact_role_temporal_attribute_path(question: str) -> str:
    """Return a role only when birth/death is the explicit outer answer edge."""
    text = re.sub(r"\s+", " ", str(question or "")).strip().lower()
    role = (
        r"director|writer|author|composer|producer|creator|performer|singer|"
        r"painter|artist|designer|founder"
    )
    patterns = (
        rf"^(?:where|when)\s+(?:was|is)\s+(?:the\s+)?(?P<role>{role})\s+"
        rf"of\s+(?P<anchor>.+?)\s+(?:born|dead)\s*\??$",
        rf"^(?:where|when)\s+did\s+(?:the\s+)?(?P<role>{role})\s+"
        rf"of\s+(?P<anchor>.+?)\s+(?:die|died)\s*\??$",
        rf"^what\s+(?:is|was)\s+(?:the\s+)?(?P<anchor>.+?)\s+"
        rf"(?P<role>{role})['\u2019]s\s+(?:birth\s+date|birthdate|birthday|"
        rf"death\s+date)\s*\??$",
    )
    for pattern in patterns:
        match = re.match(pattern, text)
        if not match:
            continue
        anchor = str(match.group("anchor") or "").strip(" ?")
        if anchor and not re.search(r"\b(?:who|that|which|where|whose)\b", anchor):
            return str(match.group("role") or "")
    passive = re.match(
        r"^(?P<anchor>.+?)\s+was\s+"
        r"(?P<verb>composed|directed|performed|produced|recorded|written)\s+by\s+"
        r"(?:a|an|the)\s+(?:man|woman|person)\s+(?:born|who\s+died)\s+"
        r"(?:in|on)?\s*what\s+(?:year|date)\s*\??$",
        text,
    )
    if passive and _safe_role_chain_anchor(str(passive.group("anchor") or "")):
        return {
            "composed": "composer",
            "directed": "director",
            "performed": "performer",
            "produced": "producer",
            "recorded": "performer",
            "written": "author",
        }.get(str(passive.group("verb") or ""), "")
    return ""


def _nested_location_landmark_projection(
    question: str,
) -> Optional[Dict[str, str]]:
    """Parse a complete role -> life-event place -> landmark projection.

    The parser is intentionally narrow: it accepts only an explicitly named
    landmark category and an explicit birth/death location clause.  It does
    not infer an outer relation from a generic ``where`` clause.
    """
    text = re.sub(r"\s+", " ", str(question or "")).strip().lower()
    role = (
        r"director|writer|author|composer|producer|creator|performer|singer|"
        r"painter|artist|designer|founder"
    )
    target = (
        r"airport|bridge|building|castle|lake|monument|mountain|museum|river|"
        r"school|university"
    )
    birth_of = re.search(
        rf"(?:name\s+of\s+the\s+)?(?:famous\s+)?(?P<target>{target})\s+"
        rf"(?:located\s+)?in\s+the\s+(?:birth\s+city|birthplace|"
        rf"place\s+of\s+birth)\s+of\s+the\s+(?P<role>{role})\s+of\b",
        text,
    )
    if birth_of:
        return {
            "target": str(birth_of.group("target") or ""),
            "role": str(birth_of.group("role") or ""),
            "event": "birth",
        }

    relative = re.search(
        rf"(?:name\s+of\s+the\s+)?(?:famous\s+)?(?P<target>{target})\s+"
        rf"(?:located\s+)?in\s+the\s+(?:city|place)\s+where\s+the\s+"
        rf"(?P<role>{role})\s+of\b.{{1,200}}?\b"
        rf"(?P<event>born|died|dies)\b",
        text,
    )
    if not relative:
        return None
    return {
        "target": str(relative.group("target") or ""),
        "role": str(relative.group("role") or ""),
        "event": (
            "birth" if str(relative.group("event") or "") == "born" else "death"
        ),
    }


def compile_question_obligations(question: str) -> Dict[str, Any]:
    """Parse high-confidence question semantics independently of a query plan.

    This parser is deliberately conservative.  It emits a contract only for
    relation compositions whose dependency order can be recovered from the
    question surface without consulting the operators produced by the query
    compiler.  Unknown means that structured execution cannot prove terminal
    completeness; it does not mean that the question has no answer.
    """
    text = re.sub(r"\s+", " ", str(question or "")).strip()
    lowered = text.lower()
    answer_type = (
        "boolean"
        if BOOLEAN_QUESTION_RE.match(text)
        else (_strict_terminal_answer_type(text) or "entity")
    )
    obligations: List[Dict[str, Any]] = []

    role_specs = {
        "mother": ({"mother_of"}, {"mother"}, "child", "mother", "person"),
        "father": ({"father_of"}, {"father"}, "child", "father", "person"),
        "parent": ({"parent_of"}, {"parent"}, "child", "parent", "person"),
        "child": ({"child_of"}, {"child"}, "parent", "child", "person"),
        "son": ({"son_of"}, {"son"}, "parent", "son", "person"),
        "daughter": ({"daughter_of"}, {"daughter"}, "parent", "daughter", "person"),
        "spouse": ({"spouse_of"}, {"spouse", "husband", "wife"}, "spouse", "spouse", "person"),
        "husband": ({"spouse_of"}, {"husband", "spouse"}, "spouse", "husband", "person"),
        "wife": ({"spouse_of"}, {"wife", "spouse"}, "spouse", "wife", "person"),
        "director": ({"direction"}, {"directed", "director"}, "work", "person", "person"),
        "writer": ({"authorship"}, {"author", "wrote", "writer"}, "work", "person", "person"),
        "author": ({"authorship"}, {"author", "wrote", "writer"}, "work", "person", "person"),
        "composer": ({"composition"}, {"composed", "composer"}, "work", "person", "person"),
        "producer": ({"production"}, {"produced", "producer"}, "work", "person", "person"),
        "performer": ({"performance"}, {"performed", "performer", "singer"}, "work", "person", "person"),
        "singer": ({"performance"}, {"performed", "sang", "singer"}, "work", "person", "person"),
        "painter": ({"visual_art"}, {"artist", "painted", "painter"}, "work", "person", "person"),
        "artist": ({"visual_art"}, {"artist", "painted", "painter"}, "work", "person", "person"),
        "designer": ({"design"}, {"designed", "designer"}, "work", "person", "person"),
        "founder": ({"founding"}, {"cofounder", "founded", "founder"}, "organization", "person", "person"),
        "creator": (
            {"authorship", "composition", "creation", "design", "direction", "production", "visual_art"},
            {"author", "composer", "created", "creator", "designer", "directed", "director", "producer", "writer"},
            "work", "person", "person",
        ),
    }

    def add_relation(
        label: str,
        predicate_families: Iterable[str],
        relation_tokens: Iterable[str],
        *,
        expected_output_type: str = "entity",
        required_subtypes: Iterable[str] = (),
        input_role: str = "entity",
        output_role: str = "entity",
        canonical_relation: str = "",
        source_pattern: str = "",
        branch_input: bool = False,
    ) -> None:
        index = sum(row.get("kind") == "relation" for row in obligations)
        input_variable = "$root_each" if branch_input and index == 0 else (
            "$root" if index == 0 else f"$q{index}"
        )
        output_variable = f"$q{index + 1}"
        obligations.append({
            "obligation_id": f"qrel{index}",
            "kind": "relation",
            "semantic_label": str(label),
            "predicate_families": sorted(set(predicate_families)),
            "required_predicate_subtypes": sorted(set(required_subtypes)),
            "relation_tokens": sorted(set(relation_tokens)),
            "input_role": str(input_role),
            "output_role": str(output_role),
            "canonical_relation": str(canonical_relation),
            "direction": "bound_input_to_output",
            "input_variable": input_variable,
            "output_variable": output_variable,
            "depends_on": [obligations[-1]["obligation_id"]] if obligations else [],
            "expected_output_type": str(expected_output_type or "entity"),
            "required": True,
            "final_projection": False,
            "source_span": _question_source_span(text, source_pattern) if source_pattern else {
                "text": "", "start": -1, "end": -1,
            },
        })

    def add_role(label: str, *, branch_input: bool = False) -> bool:
        spec = role_specs.get(label)
        if not spec:
            return False
        predicates, tokens, input_role, output_role, output_type = spec
        canonical_relation = (
            next(iter(predicates)) if label in {"mother", "father"} else ""
        )
        add_relation(
            label,
            predicates,
            tokens,
            expected_output_type=output_type,
            input_role=input_role,
            output_role=output_role,
            canonical_relation=canonical_relation,
            source_pattern=rf"\b{re.escape(label)}(?:'s)?\b",
            branch_input=branch_input,
        )
        return True

    def finalize(reason: str, confidence: str = "high") -> Dict[str, Any]:
        nonlocal answer_type
        relation_rows = [row for row in obligations if row.get("kind") == "relation"]
        comparison_rows = [row for row in obligations if row.get("kind") == "comparison"]
        if comparison_rows:
            comparison_rows[-1]["final_projection"] = True
            comparison_rows[-1]["output_variable"] = "$answer"
        elif relation_rows:
            relation_rows[-1]["final_projection"] = True
            relation_rows[-1]["output_variable"] = "$answer"




            education_place_surface = (
                str(relation_rows[-1].get("semantic_label") or "") == "education"
                and answer_type == "place"
            )
            if education_place_surface:


                answer_type = str(
                    relation_rows[-1].get("expected_output_type")
                    or "organization"
                )
            else:
                relation_rows[-1]["expected_output_type"] = answer_type
        return {
            "version": QUESTION_OBLIGATION_CONTRACT_VERSION,
            "parse_status": "parsed" if obligations else "unknown",
            "parse_reason": reason if obligations else "no_high_confidence_relation_parse",
            "parse_confidence": confidence if obligations else "unknown",
            "answer_type": answer_type,
            "obligations": obligations,
            "required_obligation_count": sum(bool(row.get("required")) for row in obligations),
            "relation_obligation_count": len(relation_rows),
            "comparison_obligation_count": len(comparison_rows),
        }

    temporal_direction = _temporal_comparison_direction(text)
    if temporal_direction:
        role_match = re.search(
            r"\b(?:whose|the)\s+(director|writer|author|composer|producer|creator)\b|"
            r"\b(director|writer|author|composer|producer|creator)\s+of\b",
            lowered,
        )
        if role_match:
            add_role(str(role_match.group(1) or role_match.group(2)), branch_input=True)
        scope = _temporal_comparison_scope(text)
        if _explicit_age_comparison_direction(scope) or re.search(r"\b(?:birth|birthday|born)\b", scope):
            label, family, tokens, subtype = "birth_date", "birth", {"birth", "birthday", "born"}, {"birth_date"}
        elif re.search(r"\b(?:death|die|died)\b", scope):
            label, family, tokens, subtype = "death_date", "death", {"death", "die", "died"}, {"death_date"}
        elif re.search(r"\b(?:release|released|premiere|premiered)\b", scope):
            label, family, tokens, subtype = "release_time", "distribution", {"release", "released", "premiere", "premiered"}, set()
        elif re.search(r"\b(?:formed|founded|established|created)\b", scope):
            label, family, tokens, subtype = "founding_time", "founding", {"formed", "founded", "established", "created"}, set()
        else:
            return finalize("temporal_comparison_relation_unknown", "unknown")
        add_relation(
            label, {family}, tokens, expected_output_type="date",
            required_subtypes=subtype, source_pattern=r"\b(?:birth|born|death|died|release|released|premiere|premiered|formed|founded|established|created)\b",
            branch_input=not obligations,
        )
        obligations.append({
            "obligation_id": "qcompare0",
            "kind": "comparison",
            "semantic_label": temporal_direction,
            "operator_type": "argmin" if temporal_direction == "argmin_root" else "argmax",
            "input_variable": "$comparison_value",
            "output_variable": "$answer",
            "depends_on": [obligations[-1]["obligation_id"]],
            "required": True,
            "final_projection": True,
            "source_span": _question_source_span(text, r"\b(?:earlier|later|older|oldest|younger|youngest|first|last)\b"),
        })
        return finalize("parsed_temporal_comparison")

    same_location_match = re.search(
        r"\b(?:the\s+)?same\s+"
        r"(?P<category>city|country|state|place|location|county|region|nationality)\b",
        lowered,
    )
    if (
        answer_type == "boolean"
        and re.search(r"\b(?:same|both)\b", lowered)
        and same_location_match
    ):
        category = str(same_location_match.group("category") or "")
        branch_role = _branch_comparison_role_label(text)
        if category in {"country", "nationality"} and branch_role:
            add_role(branch_role, branch_input=True)
            add_relation(
                "nationality",
                {"nationality"},
                {"citizen", "citizenship", "country", "from", "national", "nationality"},
                expected_output_type="nationality",
                required_subtypes={"nationality"},
                source_pattern=r"\b(?:country|from|nationality|citizenship)\b",
            )
        elif category == "nationality":
            add_relation(
                "nationality",
                {"nationality"},
                {"citizen", "citizenship", "country", "from", "national", "nationality"},
                expected_output_type="nationality",
                required_subtypes={"nationality"},
                source_pattern=r"\b(?:nationality|citizenship)\b",
                branch_input=True,
            )
        else:
            add_relation(
                "location", {"containment", "geopolitical_location"},
                {"based", "country", "located", "location", "state"},
                expected_output_type="place", source_pattern=r"\b(?:city|country|state|place|location|county)\b",
                branch_input=True,
            )
        obligations.append({
            "obligation_id": "qcompare0", "kind": "comparison",
            "semantic_label": "equality", "operator_type": "compare_eq",
            "input_variable": "$comparison_value", "output_variable": "$answer",
            "depends_on": [obligations[-1]["obligation_id"]], "required": True,
            "final_projection": True,
            "source_span": _question_source_span(text, r"\b(?:same|both)\b"),
        })
        return finalize(
            "parsed_role_nationality_branch_equality"
            if branch_role and category in {"country", "nationality"}
            else (
                "parsed_nationality_branch_equality"
                if category == "nationality"
                else "parsed_branch_equality"
            )
        )

    grandparent_match = re.search(
        r"\b(?P<lineage>paternal|maternal)\s+grand(?P<kind>father|mother)\b",
        lowered,
    )
    if grandparent_match:



        first_role = (
            "father"
            if str(grandparent_match.group("lineage") or "") == "paternal"
            else "mother"
        )
        second_role = str(grandparent_match.group("kind") or "")
        add_role(first_role)
        add_role(second_role)
        source_span = _question_source_span(
            text,
            r"\b(?:paternal|maternal)\s+grand(?:father|mother)\b",
        )
        for row in obligations:
            row["source_span"] = dict(source_span)
        return finalize("parsed_explicit_grandparent_path")

    exact_role_chain = _exact_role_chain(text)
    if exact_role_chain:
        for chain_role in exact_role_chain:
            add_role(chain_role)
        return finalize("parsed_exact_nested_role_chain")

    exact_nested_roles = _exact_nested_role_path(text)
    if exact_nested_roles:
        inner_role, outer_role = exact_nested_roles
        add_role(inner_role)
        add_role(outer_role)
        return finalize("parsed_exact_nested_role_path")

    repeated_fathers = len(re.findall(r"\bfather(?:'s)?\b", lowered))
    if repeated_fathers >= 2:
        for _ in range(repeated_fathers):
            add_role("father")
        return finalize("parsed_repeated_kinship_path")

    nested_landmark = _nested_location_landmark_projection(text)
    if nested_landmark and add_role(nested_landmark["role"]):
        event = nested_landmark["event"]
        add_relation(
            f"{event}_place",
            {event},
            {
                "birth", "birthday", "birthplace", "born"
            } if event == "birth" else {
                "death", "die", "died", "dies"
            },
            expected_output_type="place",
            required_subtypes={f"{event}_place"},
            source_pattern=(
                r"\b(?:birth city|birth place|birthplace|born)\b"
                if event == "birth"
                else r"\b(?:death place|place of death|died|dies)\b"
            ),
        )
        add_relation(
            nested_landmark["target"],
            set(),
            {nested_landmark["target"]},
            expected_output_type="entity",
            source_pattern=rf"\b{re.escape(nested_landmark['target'])}\b",
        )
        return finalize("parsed_nested_location_landmark_path")

    role_match = re.search(
        r"\b(director|writer|author|composer|producer|creator|performer|singer|painter|artist|designer|founder|mother|father|parent|child|son|daughter|spouse|husband|wife)(?:'s)?\b",
        lowered,
    )
    role_label = str(role_match.group(1) or "") if role_match else ""
    if not role_label:
        surface_role_patterns = (
            ("director", r"\bdirected\b"),
            ("writer", r"\b(?:wrote|written by)\b"),
            ("composer", r"\bcomposed\b"),
            ("producer", r"\bproduced\b"),
            ("performer", r"\b(?:performed|sang)\b"),
        )
        role_label = next(
            (
                label for label, pattern in surface_role_patterns
                if re.search(pattern, lowered)
            ),
            "",
        )
    has_birth = bool(re.search(r"\b(?:birth|birthday|birthplace|born)\b", lowered))
    has_death = bool(re.search(r"\b(?:death|die|died|dies)\b", lowered))
    education_match = re.search(
        r"\b(?:study|studied|graduate|graduated|attend|attended|educated)\b",
        lowered,
    )
    containment_pattern = (
        r"\b(?:county|state|country|province|region)\b.{0,48}"
        r"\b(?:having|containing|contains|part of|located in|in which|where)\b.{0,48}"
        r"\b(?:city|town|place|birthplace)\b|"
        r"\b(?:county|state|country|province|region)\b.{0,24}"
        r"\b(?:city|town|place|birthplace)\b.{0,24}"
        r"\b(?:in which|where)\b|"
        r"\b(?:city|town|place|birthplace)\b.{0,48}"
        r"\b(?:part of|located in|contained in)\b.{0,48}"
        r"\b(?:county|state|country|province|region)\b"
    )
    has_nested_containment = bool(re.search(containment_pattern, lowered))
    border_match = re.search(
        r"\b(?:shares? (?:a )?border with|borders?|bordering|adjacent to|neighbou?r(?:s|ing)?)\b",
        lowered,
    )
    hydrology_match = re.search(
        r"\b(?:flows? into|drains? into|empt(?:y|ies) into|outflow)\b", lowered
    )





    if (
        role_label
        and role_match
        and education_match
        and role_match.start() < education_match.start()
    ):
        add_role(role_label)
        add_relation(
            "education",
            {"education"},
            {"attend", "attended", "educated", "graduate", "graduated", "study", "studied"},
            expected_output_type="organization",
            input_role="person",
            output_role="organization",
            source_pattern=r"\b(?:study|studied|graduate|graduated|attend|attended|educated)\b",
        )
        return finalize("parsed_role_education_path")

    if role_label and (has_birth or has_death or has_nested_containment or border_match or hydrology_match):
        add_role(role_label)
    if has_birth and (role_label or has_nested_containment):
        add_relation(
            "birth_place" if answer_type == "place" or has_nested_containment else "birth",
            {"birth"}, {"birth", "birthday", "birthplace", "born"},
            expected_output_type="place" if answer_type == "place" or has_nested_containment else answer_type,
            required_subtypes={"birth_place"} if answer_type == "place" or has_nested_containment else (),
            source_pattern=r"\b(?:birth city|birth place|birthplace|born)\b",
        )
    elif has_death and (role_label or has_nested_containment):
        add_relation(
            "death_place" if answer_type == "place" or has_nested_containment else "death",
            {"death"}, {"death", "die", "died", "dies"},
            expected_output_type="place" if answer_type == "place" or has_nested_containment else answer_type,
            required_subtypes={"death_place"} if answer_type == "place" or has_nested_containment else (),
            source_pattern=r"\b(?:death place|place of death|died|dies)\b",
        )
    if has_nested_containment:
        add_relation(
            "administrative_containment",
            {"containment", "geopolitical_location", "membership"},
            {"contained", "county", "country", "located", "part", "province", "region", "state"},
            expected_output_type="place",
            source_pattern=containment_pattern,
        )
    if border_match:
        add_relation(
            "border", {"border"}, {"adjacent", "border", "bordered", "borders", "neighbor", "neighbour"},
            expected_output_type="place", source_pattern=r"\b(?:shares? (?:a )?border with|borders?|bordering|adjacent to|neighbou?r(?:s|ing)?)\b",
        )
    if hydrology_match:
        add_relation(
            "hydrology_outflow", {"hydrology_outflow"},
            {"drain", "drained", "drains", "empties", "flows", "mouth", "outflow"},
            expected_output_type="place", source_pattern=r"\b(?:flows? into|drains? into|empt(?:y|ies) into|outflow)\b",
        )
    if obligations:
        return finalize("parsed_compositional_relation_path")

    if re.search(r"\b(?:draft|drafted|selected)\b", lowered) and re.search(
        r"\b(?:belong|belongs|conference|division|league)\b", lowered
    ):
        add_relation("draft_selection", {"draft_selection"}, {"draft", "drafted", "selected"}, expected_output_type="organization")
        add_relation("membership", {"membership"}, {"belong", "belongs", "conference", "division", "league"}, expected_output_type="organization")
        return finalize("parsed_draft_membership_path")

    if has_birth:
        subtype = {f"birth_{answer_type}"} if answer_type in {"date", "place"} else set()
        add_relation("birth", {"birth"}, {"birth", "birthday", "birthplace", "born"}, expected_output_type=answer_type, required_subtypes=subtype, source_pattern=r"\b(?:birth|birthday|birthplace|born)\b")
        return finalize("parsed_atomic_birth")
    if has_death:
        subtype = {f"death_{answer_type}"} if answer_type in {"date", "place"} else set()
        add_relation("death", {"death"}, {"death", "die", "died", "dies"}, expected_output_type=answer_type, required_subtypes=subtype, source_pattern=r"\b(?:death|die|died|dies)\b")
        return finalize("parsed_atomic_death")
    if role_label:
        add_role(role_label)
        return finalize("parsed_atomic_role_relation")

    exact_atomic = _exact_atomic_relation_projection(text)
    if exact_atomic:
        add_relation(
            str(exact_atomic["semantic_label"]),
            exact_atomic["predicate_families"],
            exact_atomic["relation_tokens"],
            expected_output_type=str(exact_atomic["output_type"]),
            required_subtypes=exact_atomic["required_predicate_subtypes"],
            source_pattern=str(exact_atomic["source_pattern"]),
        )
        return finalize("parsed_exact_atomic_relation_projection")

    atomic_specs = (
        ("occupation", r"\b(?:occupation|profession|job|worked as)\b", {"employment", "occupation"}, {"occupation", "profession", "job", "worked"}, "occupation", {"occupation_role"}),
        ("nationality", r"\b(?:nationality|citizenship|citizen|national)\b", {"nationality"}, {"nationality", "citizenship", "citizen", "national"}, "nationality", {"nationality"}),
        ("end_time", r"\b(?:abolished|ceased|dissolved|ended)\b", {"end_time"}, {"abolished", "ceased", "dissolved", "ended"}, answer_type, set()),
        ("start_time", r"\b(?:began|commenced|started)\b", {"start_time"}, {"began", "commenced", "started"}, answer_type, set()),
    )
    matches = [spec for spec in atomic_specs if re.search(spec[1], lowered)]
    if len(matches) == 1:
        label, pattern, families, tokens, output_type, subtypes = matches[0]
        add_relation(label, families, tokens, expected_output_type=output_type, required_subtypes=subtypes, source_pattern=pattern)
        return finalize("parsed_atomic_relation")
    return finalize("no_high_confidence_relation_parse", "unknown")


def _coverage_type_compatible(expected: str, actual: str) -> bool:
    expected = str(expected or "entity").strip().lower()
    actual = str(actual or "entity").strip().lower()
    if expected in {"", "entity", "factoid", "named_entity"}:
        return True
    if actual in {"", "entity", "factoid", "named_entity"}:



        return True
    if expected == actual:
        return True
    aliases = (
        {"location", "place", "city", "county", "country", "state"},
        {"date", "year"},
        {"organization", "institution", "school", "university"},
        {"person", "human"},
    )
    return any(expected in group and actual in group for group in aliases)


def _unmodeled_nested_location_outer_projection_reason(
    question: str,
    plan: Mapping[str, Any],
) -> str:
    """Reject a two-hop attribute plan that terminates inside a location clause.

    Questions such as ``When did the city where X died become independent?``
    contain a person-to-place relation inside a relative clause and then ask
    for another relation of that place.  The generic two-stage role/attribute
    compiler used to lower only ``X -> person -> death_date`` and incorrectly
    certify the inner value as the final answer.  A direct request for the
    relative-clause location itself remains a valid two-hop projection.
    """
    if str(plan.get("compile_reason") or "") != "compiled_two_stage_attribute_path":
        return ""
    if len(list(plan.get("operators") or [])) != 2:
        return ""

    normalized = re.sub(r"\s+", " ", str(question or "").strip().casefold())
    location_relative = re.search(
        r"\b(?:city|town|state|country|county|province|region|capital|place)\s+"
        r"(?:in which|where)\b",
        normalized,
    )
    if not location_relative:
        return ""
    if not re.search(r"\b(?:birth|born|death|died|dies)\b", normalized):
        return ""

    direct_location_projection = re.match(
        r"^(?:what|which)\s+"
        r"(?:(?:is|was|are|were)\s+(?:the\s+)?)?"
        r"(?:name\s+of\s+(?:the\s+)?)?"
        r"(?:city|town|state|country|county|province|region|capital|place)\s+"
        r"(?:in which|where)\b",
        normalized,
    )
    if direct_location_projection:
        return ""
    return "nested_location_outer_projection_not_lowered"


def assess_question_plan_coverage(
    question_contract: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> Dict[str, Any]:
    """Prove whether a concrete logical plan covers parsed question semantics."""
    obligations = [
        dict(row) for row in question_contract.get("obligations") or []
        if row.get("required")
    ]
    base = {
        "version": PLAN_COVERAGE_CONTRACT_VERSION,
        "question_obligation_version": str(question_contract.get("version") or ""),
        "required_obligation_count": len(obligations),
        "covered_obligation_count": 0,
        "missing_obligation_ids": [],
        "mismatch_reasons": [],
        "operator_matches": [],
        "terminal_projection_match": False,
        "variable_dependency_match": False,
    }
    if str(question_contract.get("parse_status") or "") != "parsed" or not obligations:
        return {
            **base,
            "status": PLAN_COVERAGE_UNKNOWN,
            "reason": str(question_contract.get("parse_reason") or "question_parse_unknown"),
        }
    if str(plan.get("compile_status") or "") != "compiled":
        return {
            **base,
            "status": PLAN_PARTIAL,
            "reason": "logical_plan_not_compiled",
            "missing_obligation_ids": [row["obligation_id"] for row in obligations],
        }

    nested_outer_reason = _unmodeled_nested_location_outer_projection_reason(
        str(plan.get("question") or ""),
        plan,
    )
    if nested_outer_reason:
        return {
            **base,
            "status": PLAN_PARTIAL,
            "reason": nested_outer_reason,
            "missing_obligation_ids": ["unmodeled_outer_projection"],
            "mismatch_reasons": [nested_outer_reason],
        }

    operators = [dict(row) for row in plan.get("operators") or []]
    relation_obligations = [row for row in obligations if row.get("kind") == "relation"]
    comparison_obligations = [row for row in obligations if row.get("kind") == "comparison"]
    mismatches: List[str] = []
    matches: List[Dict[str, Any]] = []
    covered_ids: Set[str] = set()
    previous_output = ""

    for index, obligation in enumerate(relation_obligations):
        obligation_id = str(obligation.get("obligation_id") or f"qrel{index}")
        if index >= len(operators):
            mismatches.append(f"{obligation_id}:missing_operator")
            continue
        operator_row = operators[index]
        required_families = set(obligation.get("predicate_families") or [])
        required_subtypes = set(obligation.get("required_predicate_subtypes") or [])
        required_tokens = set(obligation.get("relation_tokens") or [])
        actual_families = set(operator_row.get("predicate_families") or [])
        actual_subtypes = set(operator_row.get("required_predicate_subtypes") or [])
        actual_tokens = set(operator_row.get("relation_tokens") or [])
        family_match = bool(required_families & actual_families)
        subtype_match = (
            not required_subtypes
            or not actual_subtypes
            or bool(required_subtypes & actual_subtypes)
        )
        token_match = bool(required_tokens & actual_tokens)
        semantic_match = (family_match or token_match) and subtype_match
        required_relation = str(obligation.get("canonical_relation") or "")
        actual_relation = str(operator_row.get("canonical_relation") or "")
        role_match = not required_relation or required_relation == actual_relation
        actual_input = str(operator_row.get("input_slot") or "")
        actual_output = str(operator_row.get("output_slot") or "")
        if index == 0:
            expected_roots = {"$root_each"} if str(plan.get("plan_kind") or "").startswith("branch_") else {"$root"}
            dependency_match = actual_input in expected_roots
        else:
            dependency_match = bool(previous_output and actual_input == previous_output)
        previous_output = actual_output
        output_type_match = _coverage_type_compatible(
            obligation.get("expected_output_type"), operator_row.get("output_type")
        )
        if str(obligation.get("semantic_label") or "") == "education":
            education_endpoint_types = {
                str(obligation.get("expected_output_type") or "").lower(),
                str(operator_row.get("output_type") or "").lower(),
            }
            if education_endpoint_types <= {
                "organization", "institution", "school", "university",
                "location", "place",
            }:
                output_type_match = True
        reasons = []
        if not semantic_match:
            reasons.append("predicate_or_subtype_mismatch")
        if not role_match:
            reasons.append("role_or_direction_mismatch")
        if not dependency_match:
            reasons.append("variable_dependency_mismatch")
        if not output_type_match:
            reasons.append("output_type_mismatch")
        if reasons:
            mismatches.extend(f"{obligation_id}:{reason}" for reason in reasons)
        else:
            covered_ids.add(obligation_id)
        matches.append({
            "obligation_id": obligation_id,
            "operator_id": str(operator_row.get("operator_id") or f"op{index}"),
            "matched": not reasons,
            "reasons": reasons,
        })

    comparison_match = True
    if comparison_obligations:
        terminal = dict(plan.get("terminal_operator") or {})
        expected_type = str(comparison_obligations[-1].get("operator_type") or "")
        actual_type = str(terminal.get("operator_type") or "")
        comparison_match = bool(expected_type and expected_type == actual_type)
        if comparison_match:
            covered_ids.update(str(row.get("obligation_id")) for row in comparison_obligations)
        else:
            mismatches.append(
                f"{comparison_obligations[-1].get('obligation_id')}:comparison_operator_missing"
            )

    final_obligation = obligations[-1]
    if comparison_obligations:
        terminal_output = str((plan.get("terminal_operator") or {}).get("output_slot") or "")
    else:
        terminal_output = str((operators[-1] if operators else {}).get("output_slot") or "")
    terminal_projection_match = bool(
        final_obligation.get("final_projection") and terminal_output == "$answer"
    )
    if not terminal_projection_match:
        mismatches.append("terminal_projection_mismatch")
    if len(operators) < len(relation_obligations):
        missing = [
            str(row.get("obligation_id"))
            for row in relation_obligations[len(operators):]
        ]
    else:
        missing = []
    missing.extend(
        str(row.get("obligation_id")) for row in obligations
        if str(row.get("obligation_id")) not in covered_ids
    )
    missing = list(dict.fromkeys(missing))
    complete = bool(
        not missing and not mismatches and terminal_projection_match and comparison_match
    )
    return {
        **base,
        "status": PLAN_COMPLETE if complete else PLAN_PARTIAL,
        "reason": "all_question_obligations_covered" if complete else "question_obligation_not_covered",
        "covered_obligation_count": len(covered_ids),
        "missing_obligation_ids": missing,
        "mismatch_reasons": list(dict.fromkeys(mismatches)),
        "operator_matches": matches,
        "terminal_projection_match": terminal_projection_match,
        "variable_dependency_match": not any(
            "variable_dependency_mismatch" in reason for reason in mismatches
        ),
    }


def compile_stage_aware_fact_join_plan(
    question: str,
    gap: Optional[Dict[str, Any]] = None,
    *,
    enable_bound_completion: bool = False,
    enable_role_directed_join: bool = True,
    enable_schema_plan_coverage_v7: bool = False,
    enable_general_plan_coverage_v8: bool = False,
    enable_schema_exact_nested_join_v9: bool = False,
) -> Dict[str, Any]:
    """Compile a schema-constrained, variable-length logical fact-join plan.

    V3 keeps named slot dependencies from V2 but no longer assumes that every
    supported question has exactly two edges.  High-confidence nested query
    forms produce complete operator chains; unsupported forms are rejected
    instead of silently dropping a terminal relation.
    """
    del gap
    lowered = str(question or "").strip().lower()
    question_tokens = _tokens(question)
    final_answer_type = _strict_terminal_answer_type(question) or "entity"
    if BOOLEAN_QUESTION_RE.match(str(question or "")):
        final_answer_type = "boolean"

    def operator(
        stage_index: int,
        *,
        input_slot: str,
        output_slot: str,
        relation_tokens: Iterable[str],
        predicate_families: Iterable[str],
        relation_families: Iterable[str] = (),
        target_entity_categories: Iterable[str] = (),
        required_predicate_subtypes: Iterable[str] = (),
        output_type: str = "entity",
        slot_plan_stage: str = "intermediate_binding",
    ) -> Dict[str, Any]:
        relation_tokens = set(relation_tokens)
        predicate_families = set(predicate_families)
        relation_families = set(relation_families)
        target_entity_categories = set(target_entity_categories)
        required_predicate_subtypes = set(required_predicate_subtypes)
        compiled = {
            "operator_id": f"op{stage_index}",
            "operator_type": "indexed_fact_lookup",
            "stage_index": int(stage_index),
            "slot_plan_stage": slot_plan_stage,
            "input_slot": input_slot,
            "output_slot": output_slot,
            "relation_tokens": sorted(relation_tokens),
            "predicate_families": sorted(predicate_families),
            "relation_families": sorted(relation_families),
            "target_entity_categories": sorted(target_entity_categories),
            "output_type": output_type,
            "join_key_contract": "exact_normalized_entity_key",
        }
        if required_predicate_subtypes:
            compiled["required_predicate_subtypes"] = sorted(
                required_predicate_subtypes
            )
            compiled["strict_predicate_subtype"] = True
            compiled["join_key_contract"] = (
                "entity_identity+predicate_subtype+output_type+title_family"
            )
        if (
            enable_schema_plan_coverage_v7
            and output_type in {"date", "place"}
            and predicate_families & {"birth", "death"}
        ):





            compiled["required_predicate_subtypes"] = sorted(
                f"{family}_{output_type}"
                for family in predicate_families & {"birth", "death"}
            )
            compiled["strict_predicate_subtype"] = True
        if enable_role_directed_join:
            compiled.update(_canonical_operator_role_contract(relation_tokens))
        else:
            compiled.update({
                "canonical_relation": "",
                "input_role": "entity",
                "output_role": "entity",
                "role_direction_required": False,
                "role_contract_version": "family_only_ablation",
            })
        return compiled

    operators: List[Dict[str, Any]] = []
    relation_obligations: List[str] = []
    plan_kind = "unsupported"
    answer_operator = "project"
    compile_reason = "unsupported_query_shape"
    v8_compile_guard_reason = "not_evaluated"
    newly_compiled_v9 = False

    role_specs = {
        "mother": (
            {"mother"},
            {"mother_of"} if enable_role_directed_join else {"kinship"},
            {"kinship"},
        ),
        "father": (
            {"father"},
            {"father_of"} if enable_role_directed_join else {"kinship"},
            {"kinship"},
        ),





        "husband": ({"husband"}, {"spouse_of"}, {"kinship"}),
        "wife": ({"wife"}, {"spouse_of"}, {"kinship"}),
        "spouse": ({"spouse"}, {"spouse_of"}, {"kinship"}),
        "director": ({"directed", "director"}, {"direction"}, {"creator"}),
        "writer": ({"author", "wrote", "writer"}, {"authorship"}, {"creator"}),
        "author": ({"author", "wrote", "writer"}, {"authorship"}, {"creator"}),
        "composer": ({"composed", "composer"}, {"composition"}, {"creator"}),
        "producer": ({"produced", "producer"}, {"production"}, {"creator"}),



        "creator": (
            {
                "author", "composed", "composer", "created", "creator",
                "designed", "designer", "directed", "director", "painted",
                "painter", "produced", "producer", "wrote", "writer",
            },
            {
                "authorship", "composition", "creation", "design",
                "direction", "production", "visual_art",
            },
            {"creator"},
        ),
    }
    if enable_schema_plan_coverage_v7:
        performer_spec = (
            {
                "performed", "performer", "recorded", "sang", "singer",
                "vocalist",
            },
            {"performance"},
            {"creator"},
        )
        role_specs["performer"] = performer_spec
        role_specs["singer"] = performer_spec
    if enable_schema_exact_nested_join_v9:
        role_specs.update({
            "parent": (
                {"parent"}, {"parent_of"}, {"kinship"},
            ),
            "child": (
                {"child"}, {"child_of"}, {"kinship"},
            ),
            "son": (
                {"son"}, {"son_of"}, {"kinship"},
            ),
            "daughter": (
                {"daughter"}, {"daughter_of"}, {"kinship"},
            ),
            "painter": (
                {"artist", "painted", "painter"},
                {"visual_art"},
                {"creator"},
            ),
            "artist": (
                {"artist", "painted", "painter"},
                {"visual_art"},
                {"creator"},
            ),
            "designer": (
                {"designed", "designer"},
                {"design"},
                {"creator"},
            ),
            "founder": (
                {"cofounder", "founded", "founder"},
                {"founding"},
                {"creator"},
            ),
        })

    def role_spec(cue: str) -> Optional[Tuple[Set[str], Set[str], Set[str]]]:
        value = role_specs.get(str(cue or "").lower())
        return tuple(set(part) for part in value) if value else None

    def path_operator(
        stage_index: int,
        spec: Tuple[Set[str], Set[str], Set[str]],
        *,
        output_type: str = "entity",
        final: bool = False,
    ) -> Dict[str, Any]:
        input_slot = "$root" if stage_index == 0 else f"$v{stage_index}"
        output_slot = "$answer" if final else f"$v{stage_index + 1}"
        tokens, predicates, families = spec
        return operator(
            stage_index,
            input_slot=input_slot,
            output_slot=output_slot,
            relation_tokens=tokens,
            predicate_families=predicates,
            relation_families=families,
            output_type=output_type,
            slot_plan_stage="final_answer" if final else "intermediate_binding",
        )

    temporal_comparison_scope = _temporal_comparison_scope(question)
    temporal_comparison_direction = _temporal_comparison_direction(question)
    age_comparison_direction = _explicit_age_comparison_direction(
        temporal_comparison_scope
    )

    temporal_comparison_spec: Optional[
        Tuple[Set[str], Set[str], Set[str]]
    ] = None
    temporal_comparison_label = ""
    if temporal_comparison_direction:
        if age_comparison_direction:


            temporal_comparison_label = "birth_date"
            temporal_comparison_spec = (
                {"birth", "birthday", "born"},
                {"birth"},
                {"temporal"},
            )
        elif re.search(
            r"\b(?:release|released|premiere|premiered)\b",
            temporal_comparison_scope,
        ):
            temporal_comparison_label = "release_time"
            temporal_comparison_spec = (
                {"release", "released", "premiere", "premiered"},
                {"distribution"},
                {"temporal"},
            )
        elif re.search(
            r"\b(?:birth|birthday|born)\b", temporal_comparison_scope
        ):
            temporal_comparison_label = "birth_date"
            temporal_comparison_spec = (
                {"birth", "birthday", "born"},
                {"birth"},
                {"temporal"},
            )
        elif re.search(
            r"\b(?:death|die|died)\b", temporal_comparison_scope
        ):
            temporal_comparison_label = "death_date"
            temporal_comparison_spec = (
                {"death", "die", "died"},
                {"death"},
                {"temporal"},
            )
        elif re.search(
            r"\b(?:formed|founded|established|created)\b",
            temporal_comparison_scope,
        ):
            temporal_comparison_label = "founding_time"
            temporal_comparison_spec = (
                {"formed", "founded", "established", "created"},
                {"founding"},
                {"temporal"},
            )

    if temporal_comparison_spec is not None:
        nested_role_match = re.search(
            r"\b(?:whose|the)\s+"
            r"(?P<role>director|writer|author|composer|producer|creator)\b|"
            r"\b(?P<role_of>director|writer|author|composer|producer|creator)"
            r"\s+of\b",
            lowered,
        )
        branch_specs: List[Tuple[Set[str], Set[str], Set[str]]] = []
        branch_labels: List[str] = []
        if nested_role_match:
            nested_role = str(
                nested_role_match.group("role")
                or nested_role_match.group("role_of")
                or ""
            )
            nested_role_spec = role_spec(nested_role)
            if nested_role_spec is not None:
                branch_specs.append(nested_role_spec)
                branch_labels.append(nested_role)
        branch_specs.append(temporal_comparison_spec)
        branch_labels.append(temporal_comparison_label)
        operators = []
        for index, spec in enumerate(branch_specs):
            tokens, predicates, families = spec
            final_branch_operator = index == len(branch_specs) - 1
            operators.append(operator(
                index,
                input_slot="$root_each" if index == 0 else f"$branch_v{index}",
                output_slot=(
                    "$comparison_value"
                    if final_branch_operator else f"$branch_v{index + 1}"
                ),
                relation_tokens=tokens,
                predicate_families=predicates,
                relation_families=families,
                output_type="date" if final_branch_operator else "entity",
                slot_plan_stage=(
                    "comparison_value"
                    if final_branch_operator else "intermediate_binding"
                ),
            ))
        plan_kind = "branch_temporal_compare"
        answer_operator = temporal_comparison_direction
        compile_reason = "compiled_branch_temporal_comparison"
        relation_obligations = branch_labels

    same_location_match = re.search(
        r"\b(?:the\s+)?same\s+"
        r"(?P<category>city|country|state|place|location|county|region|nationality)\b",
        lowered,
    )
    same_location = bool(
        final_answer_type == "boolean"
        and re.search(r"\b(?:same|both)\b", lowered)
        and same_location_match
    )
    if not operators and same_location:
        plan_kind = "branch_compare"
        answer_operator = "compare_eq"
        category = str(same_location_match.group("category") or "")
        branch_role = _branch_comparison_role_label(question)
        branch_role_spec = role_spec(branch_role)
        if branch_role_spec is not None and category in {"country", "nationality"}:
            role_tokens, role_predicates, role_families = branch_role_spec
            compile_reason = "compiled_role_nationality_branch_comparison"
            relation_obligations = [branch_role, "nationality"]
            operators = [
                operator(
                    0,
                    input_slot="$root_each",
                    output_slot="$branch_v1",
                    relation_tokens=role_tokens,
                    predicate_families=role_predicates,
                    relation_families=role_families,
                    output_type="person",
                    slot_plan_stage="intermediate_binding",
                ),
                operator(
                    1,
                    input_slot="$branch_v1",
                    output_slot="$comparison_value",
                    relation_tokens={
                        "citizen", "citizenship", "country", "from",
                        "national", "nationality",
                    },
                    predicate_families={"nationality"},
                    relation_families={"nationality"},
                    required_predicate_subtypes={"nationality"},
                    output_type="nationality",
                    slot_plan_stage="comparison_value",
                ),
            ]
        elif category == "nationality":
            compile_reason = "compiled_nationality_branch_comparison"
            relation_obligations = ["nationality"]
            operators = [operator(
                0,
                input_slot="$root_each",
                output_slot="$comparison_value",
                relation_tokens={
                    "citizen", "citizenship", "country", "from",
                    "national", "nationality",
                },
                predicate_families={"nationality"},
                relation_families={"nationality"},
                required_predicate_subtypes={"nationality"},
                output_type="nationality",
                slot_plan_stage="comparison_value",
            )]
        else:
            compile_reason = "compiled_exact_branch_comparison"
            relation_obligations = ["location"]
            operators = [operator(
                0,
                input_slot="$root_each",
                output_slot="$comparison_value",
                relation_tokens={"based", "country", "located", "location", "state"},
                predicate_families={"containment", "geopolitical_location"},
                relation_families={"location"},
                target_entity_categories=(
                    {category}
                    if category in TERMINAL_ENTITY_CATEGORY_TERMS
                    else set()
                ),
                output_type="place",
                slot_plan_stage="support_only",
            )]
    elif not operators:
        nested_landmark = _nested_location_landmark_projection(question)
        nested_occurrence = re.search(
            r"how\s+many\s+times\s+did\s+(?P<event>[a-z0-9 -]{2,48}?)\s+"
            r"occur\s+in\s+the\s+place\s+where\s+.+?'s\s+"
            r"(?P<role>director|writer|author|composer|producer|creator)\s+"
            r"(?:died|death)\b",
            lowered,
        )
        if nested_landmark:
            role = nested_landmark["role"]
            target = nested_landmark["target"]
            event = nested_landmark["event"]
            first = role_spec(role)
            if first:
                specs = [
                    first,
                    (
                        (
                            {"birth", "birthday", "birthplace", "born"}
                            if event == "birth"
                            else {"death", "die", "died", "dies"}
                        ),
                        {event},
                        {"location"},
                    ),
                    ({target}, set(), {"location"}),
                ]
                operators = [
                    path_operator(
                        index,
                        spec,
                        output_type=(
                            final_answer_type if index == len(specs) - 1
                            else "place" if index == 1 else "person"
                        ),
                        final=index == len(specs) - 1,
                    )
                    for index, spec in enumerate(specs)
                ]
                operators[1]["required_predicate_subtypes"] = [f"{event}_place"]
                operators[1]["strict_predicate_subtype"] = True
                operators[-1]["target_entity_categories"] = [target]
                plan_kind = "path_projection"
                compile_reason = "compiled_nested_landmark_path"
                relation_obligations = [role, event, target]
        elif nested_occurrence:
            role = str(nested_occurrence.group("role") or "")
            event_tokens = _tokens(nested_occurrence.group("event"))
            first = role_spec(role)
            if first and event_tokens:
                specs = [
                    first,
                    ({"death", "died"}, {"death"}, {"location"}),
                    (
                        set(event_tokens) | {"occur", "occurred", "times"},
                        set(),
                        set(),
                    ),
                ]
                operators = [
                    path_operator(
                        index,
                        spec,
                        output_type=(
                            "number" if index == len(specs) - 1
                            else "place" if index == 1 else "entity"
                        ),
                        final=index == len(specs) - 1,
                    )
                    for index, spec in enumerate(specs)
                ]
                final_answer_type = "number"
                plan_kind = "path_projection"
                compile_reason = "compiled_nested_occurrence_path"
                relation_obligations = [role, "death", "occurrence_count"]

    if not operators and re.search(r"\bbirthplace\b.{0,80}\babolished\b", lowered):
        plan_kind = "path_projection"
        compile_reason = "compiled_birthplace_end_time_path"
        relation_obligations = ["birth", "end_time"]
        specs = [
            (
                {"birth", "birthplace", "born"},
                {"birth"},
                {"location"},
            ),
            (
                {"abolished", "ceased", "dissolved", "ended"},
                {"end_time"},
                {"temporal"},
            ),
        ]
        operators = [
            path_operator(0, specs[0], output_type="place"),
            path_operator(1, specs[1], output_type="date", final=True),
        ]
        final_answer_type = "date"
    if (
        not operators
        and re.search(r"\b(?:draft|drafted|selected)\b", lowered)
        and re.search(r"\b(?:belong|belongs|conference|division|league)\b", lowered)
    ):




        plan_kind = "path_projection"
        compile_reason = "compiled_draft_team_membership_path"
        relation_obligations = ["draft_selection", "membership"]
        operators = [
            path_operator(
                0,
                (
                    {"draft", "drafted", "selected", "selection"},
                    {"draft_selection"},
                    {"draft_selection"},
                ),
                output_type="organization",
            ),
            path_operator(
                1,
                (
                    {
                        "affiliated", "belong", "belongs", "conference",
                        "division", "league", "member", "part",
                    },
                    {"membership"},
                    {"membership"},
                ),
                output_type="organization",
                final=True,
            ),
        ]
        final_answer_type = "organization"
    if not operators:




        v9_extension_roles = {
            "artist", "child", "daughter", "designer", "founder", "painter",
            "parent", "son",
        }
        exact_v9_attribute_role = _exact_role_temporal_attribute_path(question)
        exact_attribute_spec = role_spec(exact_v9_attribute_role)
        bridge_spec = (
            (exact_v9_attribute_role, exact_attribute_spec)
            if exact_v9_attribute_role and exact_attribute_spec is not None
            else next(
                (
                    (cue, spec)
                    for cue, spec in role_specs.items()
                    if re.search(rf"\b{re.escape(cue)}(?:'s)?\b", lowered)
                    and cue not in v9_extension_roles
                ),
                None,
            )
        )
        terminal_spec: Optional[Tuple[Set[str], Set[str], Set[str], str]] = None
        if (
            final_answer_type in {"date", "place"}
            and re.search(r"\b(?:birth|birthday|birthplace|born)\b", lowered)
        ):
            terminal_spec = (
                {"birth", "birthday", "birthplace", "born"},
                {"birth"},
                {"location", "temporal"},
                final_answer_type,
            )
        elif (
            final_answer_type in {"date", "place"}
            and re.search(r"\b(?:death|die|died|dies)\b", lowered)
        ):
            terminal_spec = (
                {"death", "die", "died", "dies"},
                {"death"},
                {"location", "temporal"},
                final_answer_type,
            )
        elif re.search(r"\b(?:played|member|joined)\b", lowered):
            terminal_spec = (
                {"joined", "member", "played"}, {"membership"}, {"membership"},
                final_answer_type,
            )
        elif (
            final_answer_type == "nationality"
            and (
            re.search(r"\b(?:nationality|citizenship|citizen|national)\b", lowered)
            or re.search(r"\b(?:what|which)\s+country\b.*\bfrom\b", lowered)
            )
        ):
            terminal_spec = (
                {"citizen", "citizenship", "from", "national", "nationality"},
                {"nationality"},
                {"nationality"},
                "nationality",
            )
        elif re.search(r"\b(?:abolished|ceased|dissolved|ended)\b", lowered):
            terminal_spec = (
                {"abolished", "ceased", "dissolved", "ended"},
                {"end_time"}, {"temporal"}, final_answer_type,
            )

        if bridge_spec and terminal_spec:
            bridge_cue, bridge_values = bridge_spec
            bridge_tokens, bridge_predicates, bridge_families = (
                set(part) for part in bridge_values
            )
            terminal_tokens, terminal_predicates, terminal_families, output_type = terminal_spec
            plan_kind = "path_projection"
            compile_reason = "compiled_two_stage_attribute_path"
            relation_obligations = [
                str(bridge_cue),
                next(iter(sorted(terminal_predicates or terminal_tokens)), "terminal"),
            ]
            operators = [
                path_operator(
                    0,
                    (bridge_tokens, bridge_predicates, bridge_families),
                    output_type="person",
                ),
                path_operator(
                    1,
                    (terminal_tokens, terminal_predicates, terminal_families),
                    output_type=output_type,
                    final=True,
                ),
            ]
            if (
                enable_schema_exact_nested_join_v9
                and bridge_cue in {"artist", "designer", "founder", "painter"}
            ):
                compile_reason = "compiled_schema_exact_role_attribute_path_v9"
                newly_compiled_v9 = True

    if not operators and enable_schema_exact_nested_join_v9:



        normalized_question = re.sub(r"\s+", " ", lowered).strip()
        region_slot = r"(?:county|state|country|province|region)"
        nested_birth_county = bool(
            re.match(
                rf"^in which {region_slot} is the (?:town|city|place) "
                rf"in which .+ (?:was|is) born\s*\??$",
                normalized_question,
            )
            or re.match(
                rf"^what {region_slot} is .+?'s birthplace (?:a )?part of\s*\??$",
                normalized_question,
            )
            or re.match(
                rf"^what {region_slot} is the (?:town|city|place) "
                rf"(?:in which|where) .+ (?:was|is) born (?:located in|part of)\s*\??$",
                normalized_question,
            )
        )
        if nested_birth_county:
            requested_region_match = re.search(
                r"\b(?:in\s+which|what|which)\s+"
                r"(?P<region>county|state|country|province|region)\b",
                normalized_question,
            )
            requested_region = str(
                requested_region_match.group("region")
                if requested_region_match else ""
            )
            operators = [
                operator(
                    0,
                    input_slot="$root",
                    output_slot="$v1",
                    relation_tokens={"birth", "birthplace", "born"},
                    predicate_families={"birth"},
                    relation_families={"location"},
                    required_predicate_subtypes={"birth_place"},
                    output_type="place",
                ),
                operator(
                    1,
                    input_slot="$v1",
                    output_slot="$answer",
                    relation_tokens={
                        "contained", "county", "country", "located", "part",
                        "province", "region", "state",
                    },
                    predicate_families={
                        "containment", "geopolitical_location", "membership",
                    },
                    relation_families={"location", "membership"},
                    required_predicate_subtypes={
                        "containment", "geopolitical_location", "membership",
                    },
                    target_entity_categories=(
                        {requested_region} if requested_region else set()
                    ),
                    output_type="place",
                    slot_plan_stage="final_answer",
                ),
            ]
            final_answer_type = "place"
            plan_kind = "path_projection"
            compile_reason = "compiled_schema_exact_birthplace_region_path_v9"
            relation_obligations = ["birth_place", "administrative_containment"]
            newly_compiled_v9 = True

        exact_role_chain = _exact_role_chain(question)
        if not operators and exact_role_chain:
            chain_specs = [role_spec(role) for role in exact_role_chain]
            if all(spec is not None for spec in chain_specs):
                operators = [
                    path_operator(
                        index,
                        spec,
                        output_type="person",
                        final=index == len(chain_specs) - 1,
                    )
                    for index, spec in enumerate(chain_specs)
                    if spec is not None
                ]
                final_answer_type = "person"
                plan_kind = "path_projection"
                compile_reason = "compiled_schema_exact_role_chain_v10"
                relation_obligations = list(exact_role_chain)
                newly_compiled_v9 = True

        exact_nested_roles = _exact_nested_role_path(question)
        if (
            not operators
            and exact_nested_roles
            and exact_nested_roles != ("father", "father")
        ):
            inner_role, outer_role = exact_nested_roles
            inner_spec = role_spec(inner_role)
            outer_spec = role_spec(outer_role)
            if inner_spec is not None and outer_spec is not None:
                operators = [
                    path_operator(0, inner_spec, output_type="person"),
                    path_operator(1, outer_spec, output_type="person", final=True),
                ]
                final_answer_type = "person"
                plan_kind = "path_projection"
                compile_reason = "compiled_schema_exact_nested_role_path_v9"
                relation_obligations = [inner_role, outer_role]
                newly_compiled_v9 = True

        exact_education_role = _exact_role_education_path(question)
        if not operators and exact_education_role:
            education_role_spec = role_spec(exact_education_role)
            if education_role_spec is not None:



                education_output_type = "organization"
                operators = [
                    path_operator(0, education_role_spec, output_type="person"),
                    path_operator(
                        1,
                        (
                            {
                                "attend", "attended", "educated", "graduate",
                                "graduated", "study", "studied", "trained",
                            },
                            {"education"},
                            {"education"},
                        ),
                        output_type=education_output_type,
                        final=True,
                    ),
                ]
                final_answer_type = education_output_type
                plan_kind = "path_projection"
                compile_reason = "compiled_schema_exact_role_education_path_v9"
                relation_obligations = [exact_education_role, "education"]
                newly_compiled_v9 = True

        exact_atomic = _exact_atomic_relation_projection(question)
        if not operators and exact_atomic:
            operators = [operator(
                0,
                input_slot="$root",
                output_slot="$answer",
                relation_tokens=exact_atomic["relation_tokens"],
                predicate_families=exact_atomic["predicate_families"],
                relation_families=exact_atomic["relation_families"],
                required_predicate_subtypes=exact_atomic[
                    "required_predicate_subtypes"
                ],
                output_type=str(exact_atomic["output_type"]),
                slot_plan_stage="final_answer",
            )]
            final_answer_type = str(exact_atomic["output_type"])
            plan_kind = "relation_projection"
            compile_reason = "compiled_schema_exact_atomic_projection_v10"
            relation_obligations = [str(exact_atomic["semantic_label"])]
            newly_compiled_v9 = True

        repeated_father_count = len(re.findall(r"\bfather(?:'s)?\b", lowered))
        if not operators and repeated_father_count >= 2:
            operators = [
                operator(
                    index,
                    input_slot="$root" if index == 0 else f"$v{index}",
                    output_slot=(
                        "$answer" if index == repeated_father_count - 1
                        else f"$v{index + 1}"
                    ),
                    relation_tokens={"father"},
                    predicate_families={"father_of"},
                    relation_families={"kinship"},
                    output_type="person",
                    slot_plan_stage=(
                        "final_answer"
                        if index == repeated_father_count - 1
                        else "intermediate_binding"
                    ),
                )
                for index in range(repeated_father_count)
            ]
            final_answer_type = "person"
            plan_kind = "path_projection"
            compile_reason = "compiled_schema_exact_repeated_kinship_path_v9"
            relation_obligations = ["father_of"] * repeated_father_count
            newly_compiled_v9 = True

        atomic_occupation = bool(re.match(
            r"^(?:what|which)\s+(?:is|was|are|were)\s+.+?'s\s+"
            r"(?:occupation|profession|job)\s*\??$",
            normalized_question,
        ))
        if not operators and atomic_occupation:
            operators = [operator(
                0,
                input_slot="$root",
                output_slot="$answer",
                relation_tokens={"job", "occupation", "profession", "worked"},
                predicate_families={"employment"},
                relation_families={"occupation"},
                required_predicate_subtypes={"occupation_role"},
                output_type="occupation",
                slot_plan_stage="final_answer",
            )]
            final_answer_type = "occupation"
            plan_kind = "relation_projection"
            compile_reason = "compiled_schema_exact_occupation_projection_v9"
            relation_obligations = ["occupation_role"]
            newly_compiled_v9 = True

    if not operators and enable_general_plan_coverage_v8:





        direct_specs: List[
            Tuple[str, Tuple[Set[str], Set[str], Set[str]], str]
        ] = []
        if re.search(r"\b(?:birth|birthday|birthplace|born)\b", lowered):
            direct_specs.append((
                "birth",
                (
                    {"birth", "birthday", "birthplace", "born"},
                    {"birth"},
                    {"location", "temporal"},
                ),
                final_answer_type,
            ))
        if re.search(r"\b(?:death|die|died|dies)\b", lowered):
            direct_specs.append((
                "death",
                (
                    {"death", "die", "died", "dies"},
                    {"death"},
                    {"location", "temporal"},
                ),
                final_answer_type,
            ))
        if re.search(
            r"\b(?:occupation|profession|job|worked as|known for)\b",
            lowered,
        ):
            direct_specs.append((
                "occupation",
                (
                    {"job", "known", "occupation", "profession", "worked"},
                    {"occupation"},
                    {"occupation"},
                ),
                "occupation",
            ))
        if re.search(
            r"\b(?:nationality|citizenship|citizen|national)\b",
            lowered,
        ):
            direct_specs.append((
                "nationality",
                (
                    {"citizen", "citizenship", "national", "nationality"},
                    {"nationality"},
                    {"nationality"},
                ),
                "nationality",
            ))
        creator_profiles = (
            ("direction", r"\b(?:directed|director)\b", {"directed", "director"}, {"direction"}),
            ("authorship", r"\b(?:author|wrote|writer|written by)\b", {"author", "wrote", "writer"}, {"authorship"}),
            ("composition", r"\b(?:composed|composer)\b", {"composed", "composer"}, {"composition"}),
            ("production", r"\b(?:produced|producer)\b", {"produced", "producer"}, {"production"}),
            ("performance", r"\b(?:performed|performer|sang|singer|vocalist)\b", {"performed", "performer", "sang", "singer", "vocalist"}, {"performance"}),
        )
        for label, pattern, tokens, predicates in creator_profiles:
            if re.search(pattern, lowered):
                direct_specs.append((
                    label,
                    (set(tokens), set(predicates), {"creator"}),
                    "person",
                ))
        kinship_profiles = (
            ("mother", r"\bmother\b"),
            ("father", r"\bfather\b"),
            ("spouse", r"\b(?:spouse|husband|wife)\b"),
            ("child", r"\b(?:child|son|daughter)\b"),
        )
        for label, pattern in kinship_profiles:
            if not re.search(pattern, lowered):
                continue
            spec = role_spec(label)
            if spec is None and label == "child":
                spec = (
                    {"child", "daughter", "son"},
                    {"child_of"} if enable_role_directed_join else {"kinship"},
                    {"kinship"},
                )
            if spec is not None:
                direct_specs.append((label, spec, "person"))




        unique_specs = []
        seen_labels = set()
        for label, spec, output_type in direct_specs:
            if label in seen_labels:
                continue
            seen_labels.add(label)
            unique_specs.append((label, spec, output_type))
        def direct_output_compatible(expected: str, produced: str) -> bool:
            expected = str(expected or "entity").strip().lower()
            produced = str(produced or "entity").strip().lower()
            if expected in {"entity", "factoid", "named_entity"}:
                return True
            if expected == produced:
                return True
            aliases = (
                {"location", "place"},
                {"date", "year"},
                {"organization", "institution"},
                {"person", "human"},
            )
            return any(expected in group and produced in group for group in aliases)

        def direct_projection_guard(label: str) -> Tuple[bool, str]:
            """Admit only surface forms whose answer is the direct relation value.

            Relation-token cardinality alone cannot distinguish ``father`` from
            ``father-in-law`` or a director from an award earned by that
            director.  V8.1 therefore uses a data-independent surface contract
            before compiling a one-hop projection.  Rejected questions remain
            available to later multi-operator compilers instead of silently
            dropping their outer relation.
            """
            text = re.sub(r"\s+", " ", lowered).strip()
            if re.search(r"\b(?:in[- ]law)\b", text):
                return False, "compound_kinship"
            if re.search(r"\b(?:how many|number of)\b", text):
                return False, "aggregation"
            if re.search(
                r"\b(?:first|last|earlier|later|older|oldest|younger|"
                r"youngest|more|less|same|both|compare|difference)\b",
                text,
            ):
                return False, "comparison_or_superlative"
            if re.search(
                r"\b(?:town|city|country|state|place)\s+in\s+which\b|"
                r"\b(?:who|which|that)\s+also\b|,\s*(?:who|which|that)\b",
                text,
            ):
                return False, "relative_clause"
            if re.search(
                r"\b(?:person|actor|author|composer|director|performer|"
                r"producer|singer|writer|film|movie|book|song|city|town|"
                r"country|state|place)\b.{0,100}\b(?:who|that|which|whose|where)\b",
                text,
            ):
                return False, "nested_relative_entity"
            if re.search(
                r"\bthe\s+(?:actor|author|composer|director|performer|"
                r"producer|singer|writer)\s+(?:in|of|who|that|which)\b",
                text,
            ):
                return False, "nested_role_description"

            if label in {"birth", "death"}:
                relation = (
                    r"(?:birth|birthday|birthplace|born)"
                    if label == "birth"
                    else r"(?:death|die|died|dies)"
                )
                direct = bool(
                    re.match(
                        rf"^(?:(?:in\s+)?what\s+"
                        rf"(?:city|country|state|place|location|county|year|date)"
                        rf"|where|when)\s+(?:was|is|were|did)\s+.+\b{relation}\b"
                        rf"(?:\s+(?:in|on|at))?\s*\??$",
                        text,
                    )
                    or re.match(
                        rf"^what\s+(?:is|was)\s+.+?'s\s+"
                        rf"(?:place|city|country|state|date|year)\s+of\s+{label}\b",
                        text,
                    )
                    or re.match(
                        rf"^what\s+(?:is|was)\s+.+?'s\s+"
                        rf"(?:birthplace|birthdate|birthday|death date)\b",
                        text,
                    )
                )
                return direct, (
                    "atomic_relation_surface" if direct
                    else "non_atomic_attribute_surface"
                )

            if label == "occupation":
                direct = bool(re.match(
                    r"^(?:what|which)\s+(?:is|was|are|were)\s+.+?'s\s+"
                    r"(?:occupation|profession|job)\s*\??$",
                    text,
                ))
                return direct, (
                    "atomic_relation_surface" if direct
                    else "non_atomic_occupation_surface"
                )

            if label == "nationality":
                direct = bool(re.match(
                    r"^(?:what|which)\s+(?:is|was|are|were)\s+.+?'s\s+"
                    r"(?:nationality|citizenship)\s*\??$",
                    text,
                ))
                return direct, (
                    "atomic_relation_surface" if direct
                    else "non_atomic_nationality_surface"
                )

            creator_patterns = {
                "direction": r"direct(?:ed|or)",
                "authorship": r"(?:author|wrote|writer|written)",
                "composition": r"compos(?:ed|er)",
                "production": r"produc(?:ed|er)",
                "performance": r"(?:performed|performer|sang|singer|vocalist)",
            }
            if label in creator_patterns:
                cue = creator_patterns[label]
                direct = bool(
                    re.match(rf"^who\s+{cue}\b", text)
                    or re.match(
                        rf"^who\s+(?:is|was)\s+(?:the\s+)?{cue}\s+of\b",
                        text,
                    )
                )
                return direct, (
                    "atomic_relation_surface" if direct
                    else "outer_answer_relation_missing"
                )

            if label in {"mother", "father", "spouse", "husband", "wife", "child"}:
                cue = r"(?:mother|father|spouse|husband|wife|child|son|daughter)"
                if len(re.findall(rf"\b{cue}\b", text)) > 1:
                    return False, "repeated_kinship_path"
                direct = bool(
                    re.match(rf"^who\s+(?:is|was)\s+(?:the\s+)?{cue}\s+of\b", text)
                    or re.match(rf"^who\s+(?:is|was)\s+.+?'s\s+{cue}\s*\??$", text)
                )
                return direct, (
                    "atomic_relation_surface" if direct
                    else "non_atomic_kinship_surface"
                )

            return False, "unsupported_direct_relation"

        if len(unique_specs) == 1:
            label, spec, output_type = unique_specs[0]
            guard_passed, v8_compile_guard_reason = direct_projection_guard(label)
            output_compatible = direct_output_compatible(
                final_answer_type, output_type
            )
            if guard_passed and output_compatible:
                tokens, predicates, families = spec
                required_subtypes: Set[str] = set()
                if label == "occupation":
                    required_subtypes.add("occupation_role")
                elif label == "nationality":
                    required_subtypes.add("nationality")
                operators = [operator(
                    0,
                    input_slot="$root",
                    output_slot="$answer",
                    relation_tokens=tokens,
                    predicate_families=predicates,
                    relation_families=families,
                    required_predicate_subtypes=required_subtypes,
                    output_type=output_type,
                    slot_plan_stage="final_answer",
                )]
                final_answer_type = output_type
                plan_kind = "relation_projection"
                answer_operator = "project"
                compile_reason = "compiled_semantics_safe_relation_projection_v8_1"
                relation_obligations = [label]
            elif not guard_passed:
                compile_reason = f"v8_rejected_{v8_compile_guard_reason}"
            else:
                v8_compile_guard_reason = "answer_type_mismatch"
                compile_reason = "v8_rejected_answer_type_mismatch"
        elif unique_specs:
            v8_compile_guard_reason = "multiple_relation_families"
            compile_reason = "v8_rejected_multiple_relation_families"

    plan_validation_status = "not_requested"
    plan_validation_reason = "v9_disabled"
    if enable_schema_exact_nested_join_v9:
        plan_validation_status = "unsupported" if not operators else "validated"
        plan_validation_reason = "no_compiled_plan" if not operators else "validated"
        if operators and plan_kind in {"path_projection", "relation_projection"}:
            previous_output = "$root"
            for index, compiled_operator in enumerate(operators):
                expected_input = "$root" if index == 0 else previous_output
                actual_input = str(compiled_operator.get("input_slot") or "")
                actual_output = str(compiled_operator.get("output_slot") or "")
                if actual_input != expected_input:
                    plan_validation_status = "rejected"
                    plan_validation_reason = f"disconnected_input_slot_op{index}"
                    break
                if not actual_output or actual_output == actual_input:
                    plan_validation_status = "rejected"
                    plan_validation_reason = f"invalid_output_slot_op{index}"
                    break
                previous_output = actual_output
            if plan_validation_status == "validated" and previous_output != "$answer":
                plan_validation_status = "rejected"
                plan_validation_reason = "terminal_answer_slot_missing"
            if (
                plan_validation_status == "validated"
                and len(relation_obligations) != len(operators)
            ):
                plan_validation_status = "rejected"
                plan_validation_reason = "relation_obligation_arity_mismatch"
        if plan_validation_status == "rejected":
            operators = []
            plan_kind = "unsupported"
            compile_reason = f"v9_rejected_{plan_validation_reason}"
            newly_compiled_v9 = False

    requires_bridge = plan_kind in {
        "path_projection", "branch_compare", "branch_temporal_compare",
    }
    bridge_op = operators[0] if operators and requires_bridge else {}
    terminal_op = operators[-1] if operators else {}
    if plan_kind == "branch_compare":
        terminal_predicates = set(terminal_op.get("predicate_families") or [])
        terminal_tokens = set(terminal_op.get("relation_tokens") or [])
    else:
        terminal_predicates = set(terminal_op.get("predicate_families") or []) if operators else set()
        terminal_tokens = set(terminal_op.get("relation_tokens") or []) if operators else set()
    slot_names = []
    for compiled_operator in operators:
        slot_names.extend([
            str(compiled_operator.get("input_slot") or ""),
            str(compiled_operator.get("output_slot") or ""),
        ])
    terminal_operator: Dict[str, Any] = {}
    if operators and answer_operator == "project":
        terminal_operator = dict(operators[-1])
        terminal_operator["operator_type"] = "project"
        terminal_operator["provenance_mode"] = "inherit_input_lineage"
    elif operators and answer_operator in {
        "argmin_root", "argmax_root", "compare_eq",
    }:
        reducer_types = {
            "argmin_root": "argmin",
            "argmax_root": "argmax",
            "compare_eq": "compare_eq",
        }
        terminal_operator = {
            "operator_id": "terminal",
            "operator_type": reducer_types[answer_operator],
            "stage_index": len(operators),
            "slot_plan_stage": "final_answer",
            "input_slot": "$comparison_value",
            "input_slots": ["$root_each", "$comparison_value"],
            "output_slot": "$answer",
            "relation_tokens": list(
                operators[-1].get("relation_tokens") or []
            ),
            "predicate_families": list(
                operators[-1].get("predicate_families") or []
            ),
            "relation_families": list(
                operators[-1].get("relation_families") or []
            ),
            "target_entity_categories": list(
                operators[-1].get("target_entity_categories") or []
            ),
            "output_type": final_answer_type,
            "provenance_mode": "union_input_lineage",
            "requires_complete_branches": True,
        }
        slot_names.append("$answer")
    logical_operators = list(operators)
    if terminal_operator and answer_operator != "project":
        logical_operators.append(terminal_operator)
    compiled_plan = {
        "version": (
            "schema_constrained_query_plan_v4"
            if enable_bound_completion else "slot_dependency_query_plan_v3"
        ),
        "logical_plan_revision": (
            "schema_exact_nested_join_v9"
            if enable_schema_exact_nested_join_v9
            else (
                "semantics_safe_general_plan_coverage_v8_1"
                if enable_general_plan_coverage_v8
                else (
                    "schema_driven_plan_coverage_v7"
                    if enable_schema_plan_coverage_v7
                    else "typed_branch_query_plan_v6"
                )
            )
        ),
        "role_directed_revision": "role_directed_typed_join_v10",
        "role_contract_version": (
            "role_directed_typed_join_v1"
            if enable_role_directed_join else "family_only_ablation"
        ),
        "role_directed_join_enabled": bool(enable_role_directed_join),
        "enabled": True,
        "question": str(question or ""),
        "plan_kind": plan_kind,
        "answer_operator": answer_operator,
        "final_answer_type": final_answer_type,
        "requires_bridge": bool(requires_bridge),
        "bridge_relation_tokens": list(bridge_op.get("relation_tokens") or []),
        "bridge_predicate_families": list(bridge_op.get("predicate_families") or []),
        "bridge_relation_families": list(bridge_op.get("relation_families") or []),
        "terminal_relation_tokens": sorted(terminal_tokens),
        "terminal_predicate_families": sorted(terminal_predicates),
        "terminal_target_entity_categories": sorted(
            terminal_op.get("target_entity_categories") or []
        ),
        "slots": list(dict.fromkeys(slot for slot in slot_names if slot)),
        "operators": operators,
        "steps": operators,
        "operator_count": len(operators),
        "physical_operator_count": len(operators),
        "logical_operators": logical_operators,
        "logical_operator_count": len(logical_operators),
        "terminal_operator": terminal_operator,
        "compile_status": "compiled" if operators else "unsupported",
        "compile_reason": compile_reason,
        "schema_plan_coverage_v7_enabled": bool(enable_schema_plan_coverage_v7),
        "general_plan_coverage_v8_enabled": bool(
            enable_general_plan_coverage_v8
        ),
        "schema_exact_nested_join_v9_enabled": bool(
            enable_schema_exact_nested_join_v9
        ),
        "newly_compiled_v9": bool(newly_compiled_v9 and operators),
        "plan_validation_status": plan_validation_status,
        "plan_validation_reason": plan_validation_reason,
        "v8_compile_guard_reason": v8_compile_guard_reason,
        "newly_compiled_v8": bool(
            enable_general_plan_coverage_v8
            and plan_kind == "relation_projection"
            and operators
        ),
        "newly_compiled_v7": bool(
            enable_schema_plan_coverage_v7
            and operators
            and any(
                "performance" in set(step.get("predicate_families") or [])
                for step in operators
            )
        ),
        "performer_plan": bool(
            operators
            and any(
                "performance" in set(step.get("predicate_families") or [])
                for step in operators
            )
        ),
        "relation_obligations": relation_obligations,
        "covered_relation_obligation_count": len(relation_obligations),
        "query_predicate_families": sorted(
            _canonical_predicate_families_for_tokens(question_tokens)
        ),
    }
    question_obligation_contract = compile_question_obligations(question)
    plan_coverage = assess_question_plan_coverage(
        question_obligation_contract, compiled_plan
    )
    compiled_plan.update({
        "question_obligation_contract": question_obligation_contract,
        "question_obligations": list(
            question_obligation_contract.get("obligations") or []
        ),
        "plan_coverage_contract_version": PLAN_COVERAGE_CONTRACT_VERSION,
        "plan_coverage": plan_coverage,
        "plan_coverage_status": str(plan_coverage.get("status") or ""),
    })
    return attach_logical_query_plan_identity(
        compiled_plan,
        compiler_source="rule",
        compiler_mode="rule",
    )


def _stage_aware_row_contract(
    *,
    relation_tokens: Set[str],
    candidate: Any,
    plan: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not plan:
        return {
            "slot_plan_stage": "unclassified",
            "final_answer_type": "",
            "final_predicate_match": False,
            "bridge_predicate_match": False,
            "final_answer_type_match": False,
        }
    predicate_families = _canonical_predicate_families_for_tokens(
        relation_tokens
    )
    relation_families = _relation_families_for_tokens(relation_tokens)
    terminal_families = set(plan.get("terminal_predicate_families") or [])
    terminal_tokens = set(plan.get("terminal_relation_tokens") or [])
    bridge_families = set(plan.get("bridge_predicate_families") or [])
    bridge_relation_families = set(
        plan.get("bridge_relation_families") or []
    )
    bridge_tokens = set(plan.get("bridge_relation_tokens") or [])
    final_type = str(plan.get("final_answer_type") or "entity")
    final_predicate_match = bool(
        predicate_families & terminal_families
        or relation_tokens & terminal_tokens
        or (
            not terminal_families
            and not terminal_tokens
            and not bool(plan.get("requires_bridge"))
        )
    )
    bridge_predicate_match = bool(
        predicate_families & bridge_families
        or relation_families & bridge_relation_families
        or relation_tokens & bridge_tokens
    )
    final_type_match = (
        False
        if final_type == "boolean"
        else _terminal_value_matches_contract(
            candidate,
            final_type,
            relation_tokens,
        )
    )
    if final_type == "boolean":
        stage = "support_only"
    elif final_predicate_match and final_type_match:
        stage = "final_answer"
    elif bridge_predicate_match:
        stage = "intermediate_binding"
    else:
        stage = "support_only"
    return {
        "slot_plan_stage": stage,
        "final_answer_type": final_type,
        "final_predicate_match": bool(final_predicate_match),
        "bridge_predicate_match": bool(bridge_predicate_match),
        "final_answer_type_match": bool(final_type_match),
        "candidate_predicate_families": sorted(predicate_families),
    }


def _relation_families_for_tokens(tokens: Set[str]) -> Set[str]:
    return {
        family
        for family, terms in RELATION_FAMILY_TERMS.items()
        if tokens & terms
    }


def _predicate_families_for_tokens(tokens: Set[str]) -> Set[str]:
    return {
        family
        for family, terms in PREDICATE_FAMILY_TERMS.items()
        if tokens & terms
    }


def _canonical_predicate_families_for_tokens(tokens: Set[str]) -> Set[str]:
    """Return only known selective Schema Memory predicate families."""
    return {
        family
        for family, terms in SCHEMA_PREDICATE_FAMILY_TERMS.items()
        if tokens & terms
    }


def _role_predicate_index_key(entity_key: str, predicate_family: str) -> str:
    entity_key = str(entity_key or "").strip()
    predicate_family = str(predicate_family or "").strip()
    if not entity_key or not predicate_family:
        return ""
    return f"{entity_key}\t{predicate_family}"


def entity_predicate_index_key(entity_key: str, predicate_family: str) -> str:
    """Return the composite physical key used by entity-predicate probes."""
    return _role_predicate_index_key(entity_key, predicate_family)


def _terminal_target_categories(tokens: Iterable[str]) -> Set[str]:
    normalized = {
        str(token or "").strip().lower()
        for token in tokens
        if str(token or "").strip()
    }
    return {
        category
        for category, category_terms in TERMINAL_ENTITY_CATEGORY_TERMS.items()
        if normalized & category_terms
    }


def _terminal_value_categories(value: Any, source_text: str = "") -> Set[str]:
    """Infer nominal categories from the value or a mention-local type clause."""
    text = str(value or "").strip()
    value_tokens = _tokens(text)
    categories = {
        category
        for category, category_terms in TERMINAL_ENTITY_CATEGORY_TERMS.items()
        if value_tokens & category_terms
    }
    source = str(source_text or "").strip()
    if categories or not text or not source:
        return categories




    mention_pattern = r"\b" + r"\s+".join(
        re.escape(token) for token in TOKEN_RE.findall(text)
    ) + r"\b"
    if mention_pattern == r"\b\b":
        return categories
    category_to_pattern = {
        category: r"(?:" + "|".join(
            re.escape(term) for term in sorted(category_terms, key=len, reverse=True)
        ) + r")"
        for category, category_terms in TERMINAL_ENTITY_CATEGORY_TERMS.items()
    }
    for category, category_pattern in category_to_pattern.items():
        after = re.search(
            mention_pattern
            + r"(?:\s*\([^)]{0,80}\))?\s*(?:,|\bis\b|\bwas\b|\bare\b|\bwere\b)"
            + r".{0,72}?\b"
            + category_pattern
            + r"\b",
            source,
            re.IGNORECASE,
        )
        before = re.search(
            r"\b" + category_pattern + r"\b(?:\s+(?:named|called))?\s+"
            + mention_pattern,
            source,
            re.IGNORECASE,
        )
        if after or before:
            categories.add(category)
    return categories


def _terminal_value_descriptor_categories(
    value: Any,
    source_text: str = "",
) -> Set[str]:
    """Infer source-local entity descriptors for soft candidate ordering."""
    text = str(value or "").strip()
    value_tokens = _tokens(text)
    categories = {
        category
        for category, terms in TERMINAL_ENTITY_DESCRIPTOR_TERMS.items()
        if value_tokens & terms
    }
    source = str(source_text or "").strip()
    if not text or not source:
        return categories
    mention_tokens = TOKEN_RE.findall(text)
    if not mention_tokens:
        return categories
    mention_pattern = r"\b" + r"\s+".join(
        re.escape(token) for token in mention_tokens
    ) + r"\b"
    for category, terms in TERMINAL_ENTITY_DESCRIPTOR_TERMS.items():
        pattern = r"(?:" + "|".join(
            re.escape(term) for term in sorted(terms, key=len, reverse=True)
        ) + r")"
        after = re.search(
            mention_pattern
            + r"(?:\s*\([^)]{0,80}\))?\s*(?:,|\bis\b|\bwas\b|\bare\b|\bwere\b)"
            + r".{0,96}?\b"
            + pattern
            + r"\b",
            source,
            re.IGNORECASE,
        )
        before = re.search(
            r"\b" + pattern + r"\b(?:\s+(?:named|called))?\s+" + mention_pattern,
            source,
            re.IGNORECASE,
        )
        if after or before:
            categories.add(category)
    return categories


def _source_local_entity_type(value: Any, source_text: str = "") -> str:
    """Infer a broad type only from a descriptor adjacent to the value.

    The relation endpoint itself is not enough to prove a concrete type: a
    work may be authored by a person or an organization.  An exact source
    phrase such as ``psychologist Erik Erikson`` does provide that missing
    evidence.  This helper intentionally recognizes only person descriptors
    with a strict mention-local pattern.
    """
    text = str(value or "").strip()
    source = str(source_text or "").strip()
    mention_tokens = TOKEN_RE.findall(text)
    if not mention_tokens or not source:
        return "unknown"
    mention_pattern = r"\b" + r"[\W_]+".join(
        re.escape(token) for token in mention_tokens
    ) + r"\b"
    descriptor_pattern = r"(?:" + "|".join(
        re.escape(term)
        for term in sorted(
            SOURCE_LOCAL_PERSON_DESCRIPTOR_TERMS, key=len, reverse=True
        )
    ) + r")"
    before = re.search(
        r"\b" + descriptor_pattern + r"\b\s+" + mention_pattern,
        source,
        re.IGNORECASE,
    )
    after = re.search(
        mention_pattern
        + r"\s*,?\s*(?:is|was)?\s*(?:an?\s+|the\s+)?\b"
        + descriptor_pattern
        + r"\b",
        source,
        re.IGNORECASE,
    )
    return "person" if before or after else "unknown"


def source_local_kinship_role_conflict(
    value: Any,
    source_text: str,
    canonical_relation: str,
) -> bool:
    """Reject a parent role contradicted by an adjacent source descriptor.

    A coordinated parent phrase does not imply that its second member is the
    mother.  For example, ``son of Queen Margrethe II and Henrik, the Prince
    Consort`` explicitly marks Henrik as male.  This is a contradiction guard,
    not a gender classifier: an unmarked name remains unknown.
    """
    relation = str(canonical_relation or "").strip().casefold()
    if relation not in {"mother_of", "father_of"}:
        return False
    text = str(value or "").strip()
    source = str(source_text or "").strip()
    tokens = TOKEN_RE.findall(text)
    if not tokens or not source:
        return False
    mention = r"\b" + r"[\W_]+".join(
        re.escape(token) for token in tokens
    ) + r"\b"
    conflicting_terms = (
        {
            "baron", "boy", "brother", "count", "duke", "emperor",
            "father", "husband", "king", "man", "prince", "son",
        }
        if relation == "mother_of"
        else {
            "baroness", "countess", "daughter", "duchess", "empress",
            "girl", "lady", "mother", "princess", "queen", "wife",
            "woman",
        }
    )
    descriptor = r"(?:" + "|".join(
        re.escape(term)
        for term in sorted(conflicting_terms, key=len, reverse=True)
    ) + r")"
    before = re.search(
        r"\b" + descriptor + r"\b\s+" + mention,
        source,
        re.IGNORECASE,
    )
    after = re.search(
        mention
        + r"\s*,?\s*(?:the\s+|an?\s+)?\b"
        + descriptor
        + r"\b",
        source,
        re.IGNORECASE,
    )
    return bool(before or after)


def terminal_candidate_category_priority(
    value: Any,
    *,
    target_category_tokens: Iterable[str] = (),
    source_text: str = "",
) -> Dict[str, Any]:
    """Rank terminal candidates without changing their admission decision.

    The score is deliberately ordinal.  Exact target-category evidence comes
    first, followed by a source-local parent descriptor (for example a
    ``mansion`` when the requested category is ``castle``), then candidates
    whose category is unresolved.  Explicitly different categories remain at
    the tail.  VERIFY and obligation admission still decide correctness.
    """
    target_categories = _terminal_target_categories(target_category_tokens)
    if not target_categories:
        return {
            "score": 0.0,
            "reason": "no_terminal_target_category",
            "target_categories": [],
            "value_categories": [],
            "descriptor_categories": [],
        }
    value_categories = _terminal_value_categories(value, source_text)
    descriptor_categories = _terminal_value_descriptor_categories(
        value, source_text
    )
    exact = target_categories & value_categories
    if exact:
        score = 4.0
        reason = "exact_target_category"
    else:
        target_parents = set().union(*(
            TERMINAL_ENTITY_CATEGORY_PARENTS.get(category, set())
            for category in target_categories
        ))
        specific_target_parents = target_parents - {"entity", "place"}
        parent_descriptors = descriptor_categories & specific_target_parents
        descriptor_parents = set().union(*(
            TERMINAL_ENTITY_CATEGORY_PARENTS.get(category, set())
            for category in descriptor_categories
        )) if descriptor_categories else set()
        shared_specific_parents = (
            specific_target_parents & descriptor_parents
        ) - {"organization"}
        if parent_descriptors:
            score = 3.0
            reason = "target_parent_descriptor"
        elif shared_specific_parents:
            score = 2.0
            reason = "shared_specific_category_parent"
        elif value_categories or descriptor_categories:
            score = 0.0
            reason = "different_resolved_category"
        else:
            score = 1.0
            reason = "category_unresolved"
    return {
        "score": score,
        "reason": reason,
        "target_categories": sorted(target_categories),
        "value_categories": sorted(value_categories),
        "descriptor_categories": sorted(descriptor_categories),
    }


def terminal_value_contract_compatibility(
    value: Any,
    answer_type: str,
    relation_tokens: Iterable[str],
    *,
    target_category_tokens: Iterable[str] = (),
    output_role: str = "",
    source_text: str = "",
    relation_families: Iterable[str] = (),
) -> Dict[str, Any]:
    """Return a direction-aware, three-state terminal type decision.

    ``relation_tokens`` describe the physical fact edge.  In contrast,
    ``target_category_tokens`` describe a requested entity category such as
    bridge or river.  Keeping them separate is essential for reverse joins:
    ``Rialto Bridge --located in--> Venice`` returns a bridge from the subject
    side even though the edge itself belongs to the location family.
    """
    answer_type = str(answer_type or "").strip().lower()
    if not answer_type:
        return {
            "status": TERMINAL_TYPE_COMPATIBLE,
            "reason": "untyped_output_contract",
            "answer_type": "",
            "value_type": _entity_type(value),
            "value_categories": [],
            "target_categories": [],
            "output_role": str(output_role or ""),
        }
    text = str(value or "").strip()
    if not text:
        return {
            "status": TERMINAL_TYPE_INCOMPATIBLE,
            "reason": "empty_output_value",
            "answer_type": answer_type,
            "value_type": "unknown",
            "value_categories": [],
            "target_categories": [],
            "output_role": str(output_role or ""),
        }
    value_tokens = _tokens(text)
    if value_tokens and value_tokens <= {
        "dr", "mr", "mrs", "ms", "prof", "sir", "sr", "jr",
    }:
        return {
            "status": TERMINAL_TYPE_INCOMPATIBLE,
            "reason": "honorific_only_output",
            "answer_type": answer_type,
            "value_type": "entity",
            "value_categories": [],
            "target_categories": [],
            "output_role": str(output_role or ""),
        }
    value_type = _entity_type(text)
    source_local_type = _source_local_entity_type(text, source_text)
    normalized_relation_tokens = {
        str(token or "").strip().lower()
        for token in relation_tokens
        if str(token or "").strip()
    }
    families = _relation_families_for_tokens(normalized_relation_tokens) | {
        str(family or "").strip().lower()
        for family in relation_families
        if str(family or "").strip()
    }
    target_categories = _terminal_target_categories(target_category_tokens)
    value_categories = _terminal_value_categories(text, source_text)
    broad_types = {value_type, "entity"}
    for category in value_categories:
        broad_types.update(TERMINAL_ENTITY_CATEGORY_PARENTS.get(category, set()))

    details = {
        "answer_type": answer_type,
        "value_type": value_type,
        "source_local_type": source_local_type,
        "value_categories": sorted(value_categories),
        "target_categories": sorted(target_categories),
        "relation_families": sorted(families),
        "output_role": str(output_role or ""),
    }
    if target_categories:
        matched_categories = target_categories & value_categories
        if matched_categories:
            if answer_type not in broad_types and answer_type != "entity":
                return {
                    "status": TERMINAL_TYPE_INCOMPATIBLE,
                    "reason": "category_matches_but_parent_type_mismatch",
                    **details,
                }
            return {
                "status": TERMINAL_TYPE_COMPATIBLE,
                "reason": "target_entity_category_match",
                **details,
            }
        if value_categories:
            return {
                "status": TERMINAL_TYPE_INCOMPATIBLE,
                "reason": "target_entity_category_mismatch",
                **details,
            }
        return {
            "status": TERMINAL_TYPE_UNKNOWN,
            "reason": "target_entity_category_unresolved",
            **details,
        }

    compatible = False
    reason = "broad_output_type_match"
    if answer_type == "date":
        compatible = value_type == "date" or bool(DATE_VALUE_RE.search(text))
    elif answer_type == "number":
        compatible = value_type == "number" or bool(NUMBER_VALUE_RE.fullmatch(text))
    elif answer_type == "duration":
        compatible = bool(NUMBER_VALUE_RE.search(text)) and bool(
            _tokens(text)
            & {
                "day", "days", "hour", "hours", "month", "months", "week",
                "weeks", "year", "years",
            }
        )
    elif answer_type == "nationality":
        compatible = value_type == "nationality" or bool(
            _tokens(text) & NATIONALITY_TERMS
        )
    elif answer_type == "place":





        compatible = value_type == "place" or (
            value_type == "entity"
            and (
                ("location" in families and str(output_role or "") != "subject")
                or bool(
                    normalized_relation_tokens
                    & (BIRTH_PLACE_RELATION_TERMS | {"died", "death"})
                )
            )
        )
    elif answer_type == "person":
        compatible = source_local_type == "person" or (
            value_type == "entity"
            and bool(families & {"creator", "kinship"})
        )
        if source_local_type == "person":
            reason = "source_grounded_person_descriptor"
    elif answer_type == "organization":
        compatible = (
            value_type in {"organization", "entity"}
            or bool(_tokens(text) & ORG_TERMS)
        ) and bool(families & {"education", "identity", "membership"})
    elif answer_type == "work":
        compatible = value_type in {"work", "entity"} and "creator" in families
    elif answer_type == "entity":



        compatible = value_type not in {"date", "number"}
    else:
        compatible = True
        reason = "unrecognized_output_type_preserved"

    if compatible:
        return {
            "status": TERMINAL_TYPE_COMPATIBLE,
            "reason": reason,
            **details,
        }
    if value_type == "entity" and not value_categories:
        return {
            "status": TERMINAL_TYPE_UNKNOWN,
            "reason": "broad_output_type_unresolved",
            **details,
        }
    return {
        "status": TERMINAL_TYPE_INCOMPATIBLE,
        "reason": "broad_output_type_mismatch",
        **details,
    }


def terminal_answer_domain_compatibility(
    question: str,
    value: Any,
    *,
    final_answer_type: str = "",
    relation_families: Iterable[str] = (),
) -> Dict[str, Any]:
    """Check the answer surface domain separately from relation semantics.

    A nationality relation can legitimately produce a demonym (``English``),
    while a question phrased as ``which country ... from`` requests a country
    entity (``United Kingdom``).  The tuple remains source-supported and may
    participate in execution, but the demonym must not be promoted as final
    Reader evidence for the country-valued question.

    This intentionally implements only surface distinctions stated explicitly
    by the question.  Unknown proper names are preserved; Source Support and
    Obligation Admission remain responsible for relation correctness.
    """
    text = str(question or "").strip().lower()
    surface = str(value or "").strip()
    families = {
        str(family or "").strip().lower()
        for family in relation_families
        if str(family or "").strip()
    }
    answer_type = str(final_answer_type or "").strip().lower()
    country_entity_requested = bool(
        re.search(r"\b(?:what|which)\s+country\b", text)
        and not re.search(r"\b(?:nationality|citizenship)\b", text)
    )
    nationality_requested = bool(
        re.search(r"\b(?:nationality|citizenship)\b", text)
    )
    expected_domain = (
        "country_entity"
        if country_entity_requested
        else "nationality_value"
        if nationality_requested
        else "relation_value"
    )
    observed_type = _entity_type(surface)
    details = {
        "question": str(question or ""),
        "value": surface,
        "expected_domain": expected_domain,
        "observed_value_type": observed_type,
        "final_answer_type": answer_type,
        "relation_families": sorted(families),
    }
    if not surface:
        return {
            "status": TERMINAL_TYPE_UNKNOWN,
            "reason": "empty_terminal_surface",
            **details,
        }
    if expected_domain != "country_entity":
        return {
            "status": TERMINAL_TYPE_COMPATIBLE,
            "reason": "no_country_entity_surface_contract",
            **details,
        }
    if answer_type != "nationality" and "nationality" not in families:
        return {
            "status": TERMINAL_TYPE_COMPATIBLE,
            "reason": "country_contract_not_over_nationality_relation",
            **details,
        }
    if observed_type == "nationality":
        return {
            "status": TERMINAL_TYPE_INCOMPATIBLE,
            "reason": "country_question_received_nationality_demonym",
            **details,
        }
    if observed_type in {"date", "number", "organization", "work"}:
        return {
            "status": TERMINAL_TYPE_INCOMPATIBLE,
            "reason": "country_question_received_non_place_value",
            **details,
        }
    return {
        "status": TERMINAL_TYPE_COMPATIBLE,
        "reason": "country_entity_surface_not_demonym",
        **details,
    }


def _terminal_value_matches_contract(
    value: Any,
    answer_type: str,
    relation_tokens: Set[str],
) -> bool:
    """Validate the terminal value itself, never an incidental passage value."""
    return terminal_value_contract_compatibility(
        value,
        answer_type,
        relation_tokens,
    )["status"] == TERMINAL_TYPE_COMPATIBLE


def terminal_value_matches_contract(
    value: Any,
    answer_type: str,
    relation_tokens: Iterable[str],
) -> bool:
    """Public terminal-value validator shared by reader admission paths."""
    return _terminal_value_matches_contract(
        value,
        answer_type,
        {str(token or "").strip().lower() for token in relation_tokens},
    )


def _strict_instance_operator_contracts_match(
    *,
    operators: Sequence[Mapping[str, Any]],
    path_triples: Sequence[Sequence[Any]],
    slot_bindings: Mapping[str, Any],
) -> Tuple[bool, str]:
    """Validate types and direction for a production instance-bound path.

    Broad predicate families are useful for catalog recall, but ``birth`` can
    describe a place, a date, or a birth name.  Production materialization
    therefore validates each bound slot against the physical tuple instead of
    treating all members of one family as interchangeable.
    """

    if len(operators) != len(path_triples):
        return False, "strict_operator_arity_mismatch"
    for index, (operator, triple) in enumerate(zip(operators, path_triples)):
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            return False, f"operator_{index}_strict_tuple_missing"
        input_binding = dict(
            slot_bindings.get(str(operator.get("input_slot") or "")) or {}
        )
        output_binding = dict(
            slot_bindings.get(str(operator.get("output_slot") or "")) or {}
        )
        input_key = str(input_binding.get("key") or "")
        output_key = str(output_binding.get("key") or "")
        output_surface = str(output_binding.get("surface") or "")
        output_type = str(operator.get("output_type") or "entity").lower()
        predicate = str(triple[1] or "")
        predicate_text = " ".join(TOKEN_RE.findall(predicate.lower()))
        predicate_families = set(
            canonical_schema_predicate_families(predicate)
        )



        if output_type in {"entity", "person", "organization", "work"}:
            if re.fullmatch(r"\s*(?:[1-9]\d{3}|20\d{2}|21\d{2})\s*", output_surface):
                return False, f"operator_{index}_attribute_bound_as_entity"
            if (
                SURFACE_MONTH_RE.search(output_surface)
                and NUMBER_VALUE_RE.search(output_surface)
            ):
                return False, f"operator_{index}_attribute_bound_as_entity"

        is_birth_or_death = bool(
            predicate_families & {"birth", "death"}
        )
        if is_birth_or_death and output_type in {"place", "date"}:
            if re.search(r"\bborn\s+as\b|\bbirth\s+name\b|\bn[ée]e\b", predicate_text):
                return False, f"operator_{index}_terminal_attribute_mismatch"

            subject_key = canonical_entity_key(triple[0])
            object_key = canonical_entity_key(triple[2])
            inverse_birthplace = bool(
                re.search(r"\bbirth\s*place\s+of\b|\bbirthplace\s+of\b", predicate_text)
            )
            direction_valid = (
                input_key == object_key and output_key == subject_key
                if inverse_birthplace
                else input_key == subject_key and output_key == object_key
            )
            if not direction_valid:
                return False, f"operator_{index}_terminal_direction_mismatch"
            expected_subtypes = {
                f"{family}_{output_type}"
                for family in predicate_families & {"birth", "death"}
            }
            candidate_subtypes = set(
                canonical_schema_predicate_subtypes(
                    predicate,
                    output_surface,
                )
            )
            if not (expected_subtypes & candidate_subtypes):
                return False, f"operator_{index}_terminal_attribute_mismatch"
    return True, "validated"


_COMPOUND_TERMINAL_DATE_RE = re.compile(
    r"\b(?:[0-3]?\d\s+)?(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+(?:[1-9]\d{2,3})\b|"
    r"\b(?:[1-9]\d{3}|20\d{2}|21\d{2})\b",
    re.IGNORECASE,
)


def _extract_compound_terminal_place_recovery(
    *,
    candidate: str,
    answer_type: str,
    relation_tokens: Set[str],
    predicate_families: Iterable[str],
    bound_surface: str,
    source_doc_id: int,
    get_doc_content: Callable[[int], str],
) -> Optional[Dict[str, Any]]:
    """Recover a place from a date-bearing sentence without changing its type.

    OpenIE frequently extracts the date from sentences such as ``born on DATE
    in PLACE`` and attaches that date to the terminal slot.  The old near-miss
    path forwarded the date as a place candidate.  This helper only creates a
    recovery row when the same source sentence explicitly contains the bound
    entity, the extracted date, and a birth/death place complement.
    """
    if str(answer_type or "").strip().lower() != "place":
        return None
    candidate = str(candidate or "").strip()
    bound_surface = str(bound_surface or "").strip()
    if not candidate or not bound_surface or not _COMPOUND_TERMINAL_DATE_RE.search(candidate):
        return None
    families = {str(value or "").strip().lower() for value in predicate_families}
    tokens = {str(value or "").strip().lower() for value in relation_tokens}
    is_birth = "birth" in families or bool(tokens & BIRTH_PLACE_RELATION_TERMS)
    is_death = "death" in families or bool(tokens & {"died", "death", "die"})
    if not (is_birth or is_death):
        return None
    try:
        content = str(get_doc_content(int(source_doc_id)) or "")
    except Exception:
        content = ""
    if not content or _entity_key(bound_surface) not in _entity_key(content):
        return None
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", content)
        if part.strip()
    ]
    candidate_parts = re.findall(r"[A-Za-z0-9]+", candidate)
    if not candidate_parts:
        return None
    date_text = r"\b" + r"[\s,./-]+".join(
        re.escape(part) for part in candidate_parts
    ) + r"\b"
    candidate_presence = re.compile(date_text, re.IGNORECASE)
    patterns: List[re.Pattern] = []
    if is_birth:
        patterns.append(re.compile(
            rf"\b(?:was\s+)?born\b[^.;()\n]{{0,100}}?{date_text}"
            rf"[^.;()\n]{{0,80}}?\b(?:in|at)\s+"
            r"(?P<place>[^.;()\n]+)",
            re.IGNORECASE,
        ))
    if is_death:
        patterns.append(re.compile(
            rf"\b(?:died|death)\b[^.;()\n]{{0,100}}?{date_text}"
            rf"[^.;()\n]{{0,80}}?\b(?:in|at)\s+"
            r"(?P<place>[^.;()\n]+)",
            re.IGNORECASE,
        ))
        patterns.append(re.compile(
            rf"[-–]\s*(?P<place>[^,.;()\n]+(?:,\s*[^,.;()\n]+)?)\s*,\s*"
            rf"{date_text}\b",
            re.IGNORECASE,
        ))
    for sentence_id, sentence in enumerate(sentences):
        if _entity_key(bound_surface) not in _entity_key(sentence):
            continue
        if not candidate_presence.search(sentence):
            continue
        for pattern in patterns:
            match = pattern.search(sentence)
            if not match:
                continue
            place = re.sub(r"\s+", " ", str(match.group("place") or ""))
            place = re.split(
                r"\s*,\s*(?:and|which|who|where|was|is|born|died)\b",
                place,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" \t,;:-")
            if not place or not _terminal_value_matches_contract(
                place, "place", tokens | {"born" if is_birth else "died"}
            ):
                continue
            return {
                "candidate": place,
                "source_doc_id": int(source_doc_id),
                "source_sentence_id": int(sentence_id),
                "source_sentence_span": sentence[:480],
                "recovered_relation": "birth_place" if is_birth else "death_place",
                "recovered_from": candidate,
            }
    return None


def _temporal_value_key(value: Any) -> Optional[Tuple[int, int, int]]:
    """Convert a supported date surface into a sortable calendar key."""
    text = str(value or "").strip().casefold()
    if not text:
        return None
    iso_match = re.search(
        r"\b(?P<year>\d{3,4})[-/](?P<month>\d{1,2})"
        r"(?:[-/](?P<day>\d{1,2}))?\b",
        text,
    )
    if iso_match:
        return (
            int(iso_match.group("year")),
            int(iso_match.group("month")),
            int(iso_match.group("day") or 1),
        )

    month_matches = [
        number for name, number in MONTH_TO_NUMBER.items() if name in text
    ]
    month = month_matches[0] if month_matches else 1
    has_month = bool(month_matches)
    era_match = re.search(
        r"\b(?:(?P<prefix>bc|bce|ad|ce)\s*)?"
        r"(?P<year>\d{1,4})\s*(?P<suffix>bc|bce|ad|ce)?\b",
        text,
    )
    year: Optional[int] = None
    era = ""
    if has_month:
        numbers = [int(number) for number in re.findall(r"\b\d{1,4}\b", text)]
        year_candidates = [number for number in numbers if number > 31]
        if year_candidates:
            year = year_candidates[-1]
        elif numbers:
            year = numbers[-1]
        era_tokens = re.findall(r"\b(?:bc|bce|ad|ce)\b", text)
        era = era_tokens[-1] if era_tokens else ""
    elif era_match:
        candidate_year = int(era_match.group("year"))
        if candidate_year >= 1000 or era_match.group("prefix") or era_match.group("suffix"):
            year = candidate_year
            era = str(
                era_match.group("prefix") or era_match.group("suffix") or ""
            )
    if year is None:
        four_digit = re.search(r"\b([1-9]\d{3}|20\d{2}|21\d{2})\b", text)
        if four_digit:
            year = int(four_digit.group(1))
    if year is None:
        return None
    if era in {"bc", "bce"}:
        year = -year

    day = 1
    if has_month:
        day_candidates = [
            int(number)
            for number in re.findall(r"\b\d{1,2}\b", text)
            if 1 <= int(number) <= 31 and int(number) != abs(year)
        ]
        if day_candidates:
            day = day_candidates[0]
    return year, month, day


def _compile_residual_predicate_contract(
    source_triple: Tuple[str, str, str],
    gap: Dict[str, Any],
    question: str,
) -> Dict[str, Any]:
    source_relation_tokens = _tokens(source_triple[1])
    source_families = _relation_families_for_tokens(source_relation_tokens)
    source_predicate_families = set(
        canonical_schema_predicate_families(source_triple[1])
    )
    source_polarities = {
        polarity
        for polarity, terms in RELATION_POLARITY_TERMS.items()
        if source_relation_tokens & terms
    }
    query_families = set(gap.get("query_relation_families") or [])
    query_polarities = set(gap.get("query_polarity") or [])
    query_relation_tokens = set(gap.get("query_relation_tokens") or [])
    query_predicate_families = _canonical_predicate_families_for_tokens(
        _tokens(question)
    )
    residual_families = query_families - source_families
    residual_polarities = query_polarities - source_polarities
    residual_predicate_families = (
        query_predicate_families - source_predicate_families
    )
    residual_tokens = {
        token
        for token in query_relation_tokens
        if not any(
            token == source_token
            or (
                min(len(token), len(source_token)) >= 5
                and (
                    token.startswith(source_token[:5])
                    or source_token.startswith(token[:5])
                )
            )
            for source_token in source_relation_tokens
        )
    }
    for polarity in source_polarities:
        residual_tokens -= RELATION_POLARITY_TERMS.get(polarity, set())
    terminal_answer_type = _strict_terminal_answer_type(question)
    residual_specific_tokens = (
        residual_tokens - GENERIC_RESIDUAL_RELATION_TOKENS
    )
    return {
        "source_relation_tokens": sorted(source_relation_tokens),
        "source_relation_families": sorted(source_families),
        "source_predicate_families": sorted(source_predicate_families),
        "source_polarities": sorted(source_polarities),
        "residual_relation_tokens": sorted(residual_tokens),
        "residual_specific_relation_tokens": sorted(residual_specific_tokens),
        "residual_relation_families": sorted(residual_families),
        "residual_predicate_families": sorted(residual_predicate_families),
        "residual_polarities": sorted(residual_polarities),
        "terminal_answer_type": terminal_answer_type,
        "has_residual_predicate": bool(
            residual_tokens
            or residual_families
            or residual_predicate_families
            or residual_polarities
        ),
    }


def _evaluate_residual_predicate(
    relation_tokens: Set[str],
    binding: Dict[str, Any],
    *,
    enable_predicate_family_match: bool = False,
) -> Dict[str, Any]:
    residual_tokens = set(binding.get("residual_relation_tokens") or [])
    residual_specific_tokens = set(
        binding.get("residual_specific_relation_tokens") or []
    )
    residual_families = set(binding.get("residual_relation_families") or [])
    residual_predicate_families = set(
        binding.get("residual_predicate_families") or []
    )
    residual_polarities = set(binding.get("residual_polarities") or [])
    candidate_families = _relation_families_for_tokens(relation_tokens)
    candidate_predicate_families = _canonical_predicate_families_for_tokens(
        relation_tokens
    )
    direct_match = bool(relation_tokens & residual_tokens)
    family_match = bool(candidate_families & residual_families)
    predicate_family_match = bool(
        candidate_predicate_families & residual_predicate_families
    )
    polarity_match = all(
        bool(relation_tokens & RELATION_POLARITY_TERMS.get(polarity, set()))
        or bool(
            enable_predicate_family_match
            and polarity in candidate_predicate_families
        )
        for polarity in residual_polarities
    )
    has_contract = bool(
        residual_tokens
        or residual_families
        or residual_predicate_families
        or residual_polarities
    )
    specific_match = bool(relation_tokens & residual_specific_tokens)
    semantic_match = bool(
        direct_match
        or family_match
        or (
            enable_predicate_family_match
            and predicate_family_match
        )
        or bool(residual_polarities)
    )
    predicate_match = bool(
        has_contract
        and polarity_match
        and semantic_match
        and (
            not residual_specific_tokens
            or specific_match
            or (
                enable_predicate_family_match
                and predicate_family_match
            )
            or bool(residual_polarities)
        )
    )
    return {
        "has_contract": has_contract,
        "match": predicate_match,
        "direct_match": direct_match,
        "specific_match": specific_match,
        "family_match": family_match,
        "predicate_family_match": predicate_family_match,
        "polarity_match": polarity_match,
        "candidate_relation_families": sorted(candidate_families),
        "candidate_predicate_families": sorted(
            candidate_predicate_families
        ),
    }


def _tokens_have_polarity(tokens: Set[str], polarity_name: str) -> bool:
    return bool(tokens & RELATION_POLARITY_TERMS.get(polarity_name, set()))


def _opposite_polarities(polarity_name: str) -> Set[str]:
    if polarity_name == "paternal":
        return {"maternal"}
    if polarity_name == "maternal":
        return {"paternal"}
    if polarity_name == "birth":
        return {"death"}
    if polarity_name == "death":
        return {"birth"}
    return set()


def _object_matches_answer_type(obj: Any, answer_types: Set[str], content: str) -> bool:
    if not answer_types:
        return True
    object_type = _entity_type(obj)
    content_text = str(content or "")
    if "date" in answer_types:
        return object_type == "date" or bool(DATE_VALUE_RE.search(str(obj or ""))) or bool(DATE_VALUE_RE.search(content_text[:2200]))
    if "number" in answer_types:
        return object_type == "number" or bool(NUMBER_VALUE_RE.search(str(obj or "")))
    if "nationality" in answer_types:
        obj_tokens = _tokens(obj)
        content_tokens = _tokens(content_text[:2200])
        return (
            object_type == "nationality"
            or bool(obj_tokens & NATIONALITY_TERMS)
            or bool(content_tokens & {"nationality", "citizenship", "citizen", "national"})
        )
    if "place" in answer_types:
        return object_type in {"place", "entity"}
    if "organization" in answer_types:
        return object_type in {"organization", "entity"}
    if "work" in answer_types:
        return object_type in {"work", "entity"}
    if "person" in answer_types:
        return object_type == "entity"
    return True


def _relation_family_score(relation_tokens: Set[str], gap: Dict[str, Any]) -> float:
    if not relation_tokens:
        return 0.0
    scores = []
    for family in gap.get("query_relation_families") or []:
        family_terms = RELATION_FAMILY_TERMS.get(str(family), set())
        if family_terms:
            scores.append(_bounded_overlap(relation_tokens, family_terms, max_required=2))
    for polarity in gap.get("query_polarity") or []:
        polarity_terms = RELATION_POLARITY_TERMS.get(str(polarity), set())
        if polarity_terms:
            scores.append(_bounded_overlap(relation_tokens, polarity_terms, max_required=1))
    return max(scores or [0.0])


def _answer_type_label(answer_types: Set[str], fallback: Any) -> str:
    if answer_types:
        for label in ("person", "place", "date", "number", "work", "organization"):
            if label in answer_types:
                return label
    return _entity_type(fallback)


def _relation_template_name(relation_tokens: Set[str], gap: Dict[str, Any]) -> str:
    polarities = set(gap.get("query_polarity") or [])
    answer_types = set(gap.get("query_answer_types") or [])
    families = set(gap.get("query_relation_families") or [])
    if "birth" in polarities:
        if "place" in answer_types:
            return "birth_place_join"
        if "date" in answer_types:
            return "birth_date_join"
        return "birth_relation_join"
    if "death" in polarities:
        if "place" in answer_types:
            return "death_place_join"
        if "date" in answer_types:
            return "death_date_join"
        return "death_relation_join"
    if "kinship" in families:
        return "kinship_inverse_join"
    if "creator" in families:
        return "creator_work_join"
    if "location" in families:
        return "location_attribute_join"
    if "temporal" in families:
        return "temporal_attribute_join"
    if relation_tokens:
        return "relation_attribute_join"
    return "generic_fact_join"


def _fact_doc_id(item: Dict[str, Any]) -> Optional[int]:
    value = item.get("doc_idx")
    if value is None:
        value = item.get("doc_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_directional_candidate_table(
    *,
    question: str,
    rows: List[Dict[str, Any]],
) -> str:
    lines = [
        "Directional Fact Join Candidate Table",
        f"Question: {question}",
        "Namespace: relation_fact_join_candidate_table; rows are derived views backed by base_passage lineage.",
        "Use this table as structured evidence only. Prefer candidates whose anchor, relation role, answer type, and provenance match the question.",
        "Rows:",
    ]
    if rows and all(bool(row.get("support_only")) for row in rows):
        lines.insert(
            -1,
            "Answer contract: boolean; rows are support-only and must not replace a yes/no answer.",
        )
    for idx, row in enumerate(rows, start=1):
        raw_candidate = str(row.get("raw_candidate") or "").strip()
        materialized_note = ""
        if row.get("surface_materialized") and raw_candidate and raw_candidate.lower() != str(row.get("candidate", "")).lower():
            materialized_note = f" raw_candidate=\"{raw_candidate[:120]}\" surface=materialized"
        lines.append(
            "- {idx}. candidate=\"{candidate}\" type={answer_type} template={template} "
            "score={score:.4f} anchor={anchor:.4f} role={role:.4f} direction={direction} "
            "validation={validation}{materialized_note} "
            "cost={cost:.4f} namespace=derived_candidate_table source_namespace=base_passage "
            "source_doc_id={source_doc_id} lineage_source=\"{title}\" triple=\"{triple}\" evidence=\"{evidence}\"".format(
                idx=idx,
                candidate=str(row.get("candidate", "") or "")[:120],
                answer_type=str(row.get("answer_type", "entity")),
                template=str(row.get("template", "generic_fact_join")),
                score=float(row.get("table_score", 0.0) or 0.0),
                anchor=float(row.get("anchor_score", 0.0) or 0.0),
                role=float(row.get("role_score", 0.0) or 0.0),
                direction=str(row.get("direction", "unknown")),
                validation=str(row.get("validation_tier", "strict")),
                materialized_note=materialized_note,
                cost=float(row.get("cost", 0.0) or 0.0),
                source_doc_id=str(row.get("doc_id", "")),
                title=str(row.get("title", "") or "")[:100],
                triple=str(row.get("triple", "") or "")[:180],
                evidence=str(row.get("evidence", "") or "")[:220],
            )
        )
    return "\n".join(lines)


def _directional_candidate_row(
    *,
    question: str,
    item: Dict[str, Any],
    content: str,
    title: str,
    rank: int,
    gap: Dict[str, Any],
    hard_anchor_min_overlap: float,
    stage_aware_plan: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    triple = item.get("triple") or ()
    if not isinstance(triple, (list, tuple)) or len(triple) != 3:
        return None
    subject, relation, obj = triple
    subject_tokens = set(item.get("subject_tokens") or _tokens(subject))
    object_tokens = set(item.get("object_tokens") or _tokens(obj))
    relation_tokens = set(item.get("relation_tokens") or _tokens(relation))
    query_anchor_tokens = set(gap.get("query_anchor_tokens") or [])
    query_relation_tokens = set(gap.get("query_relation_tokens") or [])
    answer_types = set(gap.get("query_answer_types") or [])
    bound_match = dict(item.get("schema_bound_semijoin") or {})

    subject_anchor = _bounded_overlap(subject_tokens, query_anchor_tokens, max_required=3)
    object_anchor = _bounded_overlap(object_tokens, query_anchor_tokens, max_required=3)
    anchor_side = ""
    if bound_match:
        anchor_side = str(bound_match.get("candidate_probe_role") or "")
        terminal_role = str(bound_match.get("terminal_role") or "")
        if anchor_side not in {"subject", "object"} or terminal_role not in {"subject", "object"}:
            return None
        anchor_entity = subject if anchor_side == "subject" else obj
        candidate = subject if terminal_role == "subject" else obj
        candidate_tokens = subject_tokens if terminal_role == "subject" else object_tokens
        direction = f"schema_bound_{anchor_side}_to_{terminal_role}_answer"
        subject_anchor = 1.0 if anchor_side == "subject" else subject_anchor
        object_anchor = 1.0 if anchor_side == "object" else object_anchor
    else:
        if not query_anchor_tokens:
            return None
        if subject_anchor >= float(hard_anchor_min_overlap) and subject_anchor >= object_anchor:
            anchor_side = "subject"
            anchor_entity = subject
            candidate = obj
            candidate_tokens = object_tokens
            direction = "subject_anchor_to_object_answer"
        elif object_anchor >= float(hard_anchor_min_overlap):
            anchor_side = "object"
            anchor_entity = obj
            candidate = subject
            candidate_tokens = subject_tokens
            direction = "object_anchor_to_subject_answer"
        else:
            return None

    role_overlap = (
        _bounded_overlap(relation_tokens, query_relation_tokens, max_required=3)
        if query_relation_tokens else 0.0
    )
    family_score = _relation_family_score(relation_tokens, gap)
    role_score = max(
        role_overlap,
        family_score,
        float(bound_match.get("relation_role_score", 0.0) or 0.0),
    )
    if query_relation_tokens and role_score < 0.34:
        return None

    candidate_predicate_families = _predicate_families_for_tokens(
        relation_tokens
    )
    stage_contract = _stage_aware_row_contract(
        relation_tokens=relation_tokens,
        candidate=candidate,
        plan=stage_aware_plan,
    )
    for required_polarity in gap.get("query_polarity") or []:
        if (
            stage_aware_plan
            and stage_contract["slot_plan_stage"] == "intermediate_binding"
            and required_polarity
            in set(stage_aware_plan.get("terminal_predicate_families") or [])
        ):
            continue
        if any(_tokens_have_polarity(relation_tokens, opposite) for opposite in _opposite_polarities(required_polarity)):
            return None
        if (
            not _tokens_have_polarity(relation_tokens, required_polarity)
            and not (
                bound_match
                and required_polarity in candidate_predicate_families
            )
        ):
            return None

    terminal_answer_type = str(
        bound_match.get("terminal_answer_type") or ""
    ).strip().lower()
    if (
        stage_aware_plan
        and stage_contract["slot_plan_stage"] == "final_answer"
    ):
        terminal_answer_type = str(
            stage_contract.get("final_answer_type") or terminal_answer_type
        )
    if (
        stage_aware_plan
        and stage_contract["slot_plan_stage"] == "intermediate_binding"
    ):
        answer_type_match = bool(
            str(candidate or "").strip()
            and _entity_type(candidate) not in {"date", "number"}
        )
    elif bound_match and terminal_answer_type:
        answer_type_match = _terminal_value_matches_contract(
            candidate,
            terminal_answer_type,
            relation_tokens,
        )
    else:
        answer_type_match = _object_matches_answer_type(
            candidate,
            answer_types,
            content,
        )
    if not answer_type_match:
        return None

    candidate_text = str(candidate or "").strip()
    if not candidate_text or len(candidate_text) > 140:
        return None

    doc_tokens = _tokens(str(content)[:2200])
    query_tokens = set(gap.get("query_tokens") or [])
    question_overlap = _overlap_over_query(doc_tokens, query_tokens)
    rank_prior = 1.0 / (1.0 + math.log1p(max(0, int(rank)) + 1))
    answer_type_score = 1.0 if answer_type_match else 0.0
    direction_bonus = 0.10 if bound_match else 0.08 if anchor_side == "subject" else 0.04
    anchor_score = max(subject_anchor, object_anchor)
    cost = _clamp(0.03 + 0.004 * max(0, int(rank)) + 0.001 * _estimate_tokens(content[:1200]))
    table_score = _clamp(
        0.30 * anchor_score
        + 0.24 * role_score
        + 0.15 * answer_type_score
        + 0.13 * question_overlap
        + 0.10 * rank_prior
        + direction_bonus
        - 0.08 * cost
    )
    doc_id = _fact_doc_id(item)
    if doc_id is None:
        return None
    row = {
        "candidate": candidate_text,
        "key": " ".join(sorted(_tokens(candidate_text)))[:120],
        "anchor_entity": str(anchor_entity or "").strip(),
        "anchor_key": _entity_key(anchor_entity),
        "bridge_entity": str(anchor_entity or "").strip(),
        "bridge_key": _entity_key(anchor_entity),
        "answer_key": _entity_key(candidate_text),
        "answer_type": (
            terminal_answer_type
            if bound_match and terminal_answer_type
            else _answer_type_label(answer_types, candidate)
        ),
        "template": _relation_template_name(relation_tokens, gap),
        "direction": direction,
        "anchor_side": anchor_side,
        "anchor_score": round(float(anchor_score), 6),
        "role_score": round(float(role_score), 6),
        "question_overlap": round(float(question_overlap), 6),
        "rank_prior": round(float(rank_prior), 6),
        "cost": round(float(cost), 6),
        "table_score": round(float(table_score), 6),
        "rank": int(rank),
        "doc_id": doc_id,
        "title": title,
        "source_namespace": "base_passage",
        "lineage_source_title": title,
        "lineage_source_doc_id": doc_id,
        "lineage_sentence_preview": str(content).replace("\n", " ")[:320],
        "derived_namespace": "relation_fact_join_candidate_table",
        "triple": " | ".join(str(part) for part in triple),
        "evidence": str(content).replace("\n", " ")[:320],
    }
    if bound_match:
        row.update({
            "schema_bound_semijoin": True,
            "schema_binding_id": int(bound_match.get("binding_id", 0) or 0),
            "schema_bound_key": str(bound_match.get("bound_key") or ""),
            "schema_source_triple": " | ".join(
                str(part) for part in (bound_match.get("source_triple") or [])
            ),
            "residual_predicate_match": bool(
                bound_match.get("residual_predicate_match")
            ),
            "terminal_slot_match": bool(bound_match.get("terminal_slot_match")),
            "terminal_answer_type": str(
                bound_match.get("terminal_answer_type") or ""
            ),
        })
    predicate_families = sorted(
        _predicate_families_for_tokens(relation_tokens)
    )
    predicate_family = predicate_families[0] if predicate_families else "generic"
    branch_id = str(
        bound_match.get("binding_id")
        if bound_match
        else f"{row['anchor_key']}::{predicate_family}"
    )
    row.update({
        "terminal_contract_version": (
            "stage_aware_typed_terminal_candidate_v2"
            if stage_aware_plan else "typed_terminal_candidate_v1"
        ),
        "validation_tier": "strict",
        "shadow_only": False,
        "production_eligible": True,
        "base_validation_passed": True,
        "anchor_match": True,
        "relation_match": True,
        "answer_type_match": True,
        "terminal_slot_match": bool(
            bound_match.get("terminal_slot_match", True)
        ),
        "residual_predicate_match": bool(
            bound_match.get("residual_predicate_match", True)
        ),
        "predicate_families": predicate_families,
        "predicate_family": predicate_family,
        "branch_id": branch_id,
        "lineage_key": (
            f"doc:{doc_id}|branch:{branch_id}|answer:{row['answer_key']}"
        ),
    })
    if stage_aware_plan:
        row.update(stage_contract)
        row["bridge_eligible"] = bool(
            stage_contract["slot_plan_stage"] == "intermediate_binding"
        )
        row["rooted_lineage_complete"] = bool(
            not stage_aware_plan.get("requires_bridge")
            or bound_match
        )
        row["production_eligible"] = bool(
            stage_contract["slot_plan_stage"] == "final_answer"
            and stage_contract["final_answer_type_match"]
            and row["rooted_lineage_complete"]
        )
        if not row["production_eligible"]:
            row["validation_tier"] = (
                "bridge" if row["bridge_eligible"] else "shadow"
            )
            row["shadow_only"] = True
    return row


def _terminal_candidate_contract_row(
    *,
    question: str,
    item: Dict[str, Any],
    content: str,
    title: str,
    rank: int,
    gap: Dict[str, Any],
    hard_anchor_min_overlap: float,
    stage_aware_plan: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Freeze a typed terminal candidate without weakening production gates.

    Strict directional rows remain production eligible.  When one of those
    checks fails, this function still emits a typed, provenance-backed shadow
    row for paired reader replay.  This separates causal coverage from answer
    admission and prevents the diagnostic path from inflating QA metrics.
    """
    strict = _directional_candidate_row(
        question=question,
        item=item,
        content=content,
        title=title,
        rank=rank,
        gap=gap,
        hard_anchor_min_overlap=hard_anchor_min_overlap,
        stage_aware_plan=stage_aware_plan,
    )
    if strict is not None:
        return strict

    triple = item.get("triple") or ()
    if not isinstance(triple, (list, tuple)) or len(triple) != 3:
        return None
    subject, relation, obj = triple
    subject_tokens = set(item.get("subject_tokens") or _tokens(subject))
    object_tokens = set(item.get("object_tokens") or _tokens(obj))
    relation_tokens = set(item.get("relation_tokens") or _tokens(relation))
    query_anchor_tokens = set(gap.get("query_anchor_tokens") or [])
    query_relation_tokens = set(gap.get("query_relation_tokens") or [])
    answer_types = set(gap.get("query_answer_types") or [])
    bound_match = dict(item.get("schema_bound_semijoin") or {})

    subject_anchor = _bounded_overlap(
        subject_tokens, query_anchor_tokens, max_required=3
    )
    object_anchor = _bounded_overlap(
        object_tokens, query_anchor_tokens, max_required=3
    )
    subject_type_match = _object_matches_answer_type(
        subject, answer_types, content
    )
    object_type_match = _object_matches_answer_type(obj, answer_types, content)

    anchor_side = ""
    terminal_role = ""
    if bound_match:
        anchor_side = str(bound_match.get("candidate_probe_role") or "")
        terminal_role = str(bound_match.get("terminal_role") or "")
        if anchor_side not in {"subject", "object"}:
            anchor_side = str(bound_match.get("probe_role") or "")
        if terminal_role not in {"subject", "object"}:
            terminal_role = "object" if anchor_side == "subject" else "subject"
    elif subject_type_match != object_type_match:
        terminal_role = "subject" if subject_type_match else "object"
        anchor_side = "object" if terminal_role == "subject" else "subject"
    elif max(subject_anchor, object_anchor) > 0.0:
        anchor_side = (
            "subject" if subject_anchor >= object_anchor else "object"
        )
        terminal_role = "object" if anchor_side == "subject" else "subject"
    else:
        return None

    if anchor_side not in {"subject", "object"}:
        return None
    if terminal_role not in {"subject", "object"}:
        return None
    anchor_entity = subject if anchor_side == "subject" else obj
    candidate = subject if terminal_role == "subject" else obj
    candidate_text = str(candidate or "").strip()
    if not candidate_text or len(candidate_text) > 140:
        return None

    anchor_score = subject_anchor if anchor_side == "subject" else object_anchor
    if bound_match:
        anchor_score = max(
            anchor_score,
            float(bound_match.get("anchor_overlap", 0.0) or 0.0),
        )
    family_score = _relation_family_score(relation_tokens, gap)
    relation_overlap = (
        _bounded_overlap(
            relation_tokens, query_relation_tokens, max_required=3
        )
        if query_relation_tokens else 0.0
    )
    role_score = max(
        relation_overlap,
        family_score,
        float(bound_match.get("relation_role_score", 0.0) or 0.0),
    )
    relation_match = bool(not query_relation_tokens or role_score >= 0.34)
    polarity_match = True
    for required_polarity in gap.get("query_polarity") or []:
        if any(
            _tokens_have_polarity(relation_tokens, opposite)
            for opposite in _opposite_polarities(required_polarity)
        ):
            polarity_match = False
            break
    stage_contract = _stage_aware_row_contract(
        relation_tokens=relation_tokens,
        candidate=candidate,
        plan=stage_aware_plan,
    )
    terminal_answer_type = str(
        bound_match.get("terminal_answer_type") or ""
    ).strip().lower()
    if (
        stage_aware_plan
        and stage_contract["slot_plan_stage"] == "final_answer"
    ):
        terminal_answer_type = str(
            stage_contract.get("final_answer_type") or terminal_answer_type
        )
    answer_type_match = (
        _terminal_value_matches_contract(
            candidate, terminal_answer_type, relation_tokens
        )
        if terminal_answer_type
        else _object_matches_answer_type(candidate, answer_types, content)
    )
    doc_id = _fact_doc_id(item)
    if doc_id is None:
        return None
    doc_tokens = _tokens(str(content)[:2200])
    query_tokens = set(gap.get("query_tokens") or [])
    question_overlap = _overlap_over_query(doc_tokens, query_tokens)
    rank_prior = 1.0 / (1.0 + math.log1p(max(0, int(rank)) + 1))
    cost = _clamp(
        0.03 + 0.004 * max(0, int(rank))
        + 0.001 * _estimate_tokens(content[:1200])
    )
    table_score = _clamp(
        0.28 * anchor_score
        + 0.24 * role_score
        + 0.16 * float(answer_type_match)
        + 0.12 * question_overlap
        + 0.10 * rank_prior
        + 0.06 * float(bool(bound_match))
        - 0.08 * cost
    )
    predicate_families = sorted(
        _predicate_families_for_tokens(relation_tokens)
    )
    predicate_family = predicate_families[0] if predicate_families else "generic"
    anchor_key = _entity_key(anchor_entity)
    answer_key = _entity_key(candidate_text)
    branch_id = str(
        bound_match.get("binding_id")
        if bound_match
        else f"{anchor_key}::{predicate_family}"
    )
    terminal_slot_match = bool(
        bound_match.get("terminal_slot_match", answer_type_match)
    )
    residual_predicate_match = bool(
        bound_match.get(
            "residual_predicate_match", relation_match and polarity_match
        )
    )
    row = {
        "candidate": candidate_text,
        "key": " ".join(sorted(_tokens(candidate_text)))[:120],
        "anchor_entity": str(anchor_entity or "").strip(),
        "anchor_key": anchor_key,
        "bridge_entity": str(anchor_entity or "").strip(),
        "bridge_key": anchor_key,
        "answer_key": answer_key,
        "answer_type": (
            terminal_answer_type
            or _answer_type_label(answer_types, candidate_text)
        ),
        "template": _relation_template_name(relation_tokens, gap),
        "direction": f"shadow_{anchor_side}_to_{terminal_role}_answer",
        "anchor_side": anchor_side,
        "anchor_score": round(float(anchor_score), 6),
        "role_score": round(float(role_score), 6),
        "question_overlap": round(float(question_overlap), 6),
        "rank_prior": round(float(rank_prior), 6),
        "cost": round(float(cost), 6),
        "table_score": round(float(table_score), 6),
        "rank": int(rank),
        "doc_id": doc_id,
        "title": title,
        "source_namespace": "base_passage",
        "lineage_source_title": title,
        "lineage_source_doc_id": doc_id,
        "lineage_sentence_preview": str(content).replace("\n", " ")[:320],
        "derived_namespace": "relation_fact_join_terminal_contract",
        "triple": " | ".join(str(part) for part in triple),
        "evidence": str(content).replace("\n", " ")[:320],
        "schema_bound_semijoin": bool(bound_match),
        "schema_binding_id": int(bound_match.get("binding_id", 0) or 0),
        "schema_bound_key": str(bound_match.get("bound_key") or ""),
        "terminal_contract_version": (
            "stage_aware_typed_terminal_candidate_v2"
            if stage_aware_plan else "typed_terminal_candidate_v1"
        ),
        "validation_tier": "shadow",
        "shadow_only": True,
        "production_eligible": False,
        "base_validation_passed": False,
        "anchor_match": bool(
            bound_match or anchor_score >= float(hard_anchor_min_overlap)
        ),
        "relation_match": bool(relation_match and polarity_match),
        "answer_type_match": bool(answer_type_match),
        "terminal_slot_match": terminal_slot_match,
        "residual_predicate_match": residual_predicate_match,
        "predicate_families": predicate_families,
        "predicate_family": predicate_family,
        "branch_id": branch_id,
        "lineage_key": (
            f"doc:{doc_id}|branch:{branch_id}|answer:{answer_key}"
        ),
    }
    if stage_aware_plan:
        row.update(stage_contract)
        row["bridge_eligible"] = False
        row["rooted_lineage_complete"] = False
        row["production_eligible"] = False
    return row


def _row_source_doc_ids(row: Dict[str, Any]) -> List[int]:
    values = list(row.get("source_doc_ids") or []) + list(
        row.get("path_doc_ids") or []
    )
    if row.get("doc_id") is not None:
        values.append(row.get("doc_id"))
    result: List[int] = []
    for value in values:
        try:
            doc_id = int(value)
        except (TypeError, ValueError):
            continue
        if doc_id >= 0 and doc_id not in result:
            result.append(doc_id)
    return result


def _merge_directional_rows(rows: List[Dict[str, Any]], max_rows: int) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("key") or row.get("candidate") or "").lower()
        if not key:
            continue
        current = by_key.get(key)
        if current is None:
            merged = dict(row)
            source_doc_ids = _row_source_doc_ids(row)
            source_titles = [
                title for title in list(row.get("source_titles") or [])
                + [row.get("title", "")]
                if title
            ]
            base_score = _clamp(float(row.get("table_score", 0.0) or 0.0))
            merged["base_table_score"] = round(base_score, 6)
            merged["table_score"] = round(base_score, 6)
            merged["provenance_count"] = len(source_doc_ids)
            merged["source_titles"] = list(dict.fromkeys(source_titles))
            merged["source_doc_ids"] = source_doc_ids
            by_key[key] = merged
            continue
        source_doc_ids = list(dict.fromkeys(
            _row_source_doc_ids(current) + _row_source_doc_ids(row)
        ))
        provenance_count = len(source_doc_ids)
        source_titles = list(dict.fromkeys(
            list(current.get("source_titles") or [])
            + list(row.get("source_titles") or [])
            + [row.get("title", "")]
        ))
        current_base_score = float(
            current.get("base_table_score", current.get("table_score", 0.0))
            or 0.0
        )
        row_base_score = float(
            row.get("base_table_score", row.get("table_score", 0.0)) or 0.0
        )
        base_score = _clamp(max(current_base_score, row_base_score))
        replace = bool(
            row.get("production_eligible")
            and not current.get("production_eligible")
        ) or (
            bool(row.get("production_eligible"))
            == bool(current.get("production_eligible"))
            and float(row.get("anchor_score", 0.0) or 0.0)
            > float(current.get("anchor_score", 0.0) or 0.0)
        )
        if replace:
            current.update({
                key: value for key, value in row.items()
                if key not in {
                    "provenance_count", "source_titles", "source_doc_ids",
                    "base_table_score", "table_score",
                }
            })
        completion_lineage_by_id = {
            str(lineage.get("completion_fact_id") or ""): dict(lineage)
            for lineage in (
                list(current.get("bound_completion_lineage") or [])
                + list(row.get("bound_completion_lineage") or [])
            )
            if lineage.get("completion_fact_id")
        }
        if completion_lineage_by_id:
            current["bound_completion_used"] = True
            current["bound_completion_lineage"] = list(
                completion_lineage_by_id.values()
            )
            current["bound_completion_fact_ids"] = list(
                completion_lineage_by_id
            )
            current["bound_completion_doc_ids"] = sorted({
                int(doc_id)
                for doc_id in (
                    list(current.get("bound_completion_doc_ids") or [])
                    + list(row.get("bound_completion_doc_ids") or [])
                )
                if doc_id is not None and int(doc_id) >= 0
            })
            current["bound_completion_tuple_keys"] = list(dict.fromkeys(
                list(current.get("bound_completion_tuple_keys") or [])
                + list(row.get("bound_completion_tuple_keys") or [])
            ))
            current["completion_causal_lineage_validated"] = all(
                lineage.get("completion_fact_id")
                and lineage.get("tuple_key")
                and int(lineage.get("source_doc_id", -1)) >= 0
                and lineage.get("bound_input_key")
                and lineage.get("instance_identity_validated")
                for lineage in completion_lineage_by_id.values()
            )
            current["completion_chain_depth"] = len(
                completion_lineage_by_id
            )
            current["residual_terminal_completion_used"] = any(
                lineage.get("slot_stage") == "final_answer"
                and lineage.get("causal_predecessor")
                for lineage in completion_lineage_by_id.values()
            )
        current["path_doc_ids"] = list(dict.fromkeys(
            list(current.get("path_doc_ids") or [])
            + list(row.get("path_doc_ids") or [])
        ))
        evidence_bonus = min(0.04, 0.012 * max(0, provenance_count - 1))
        current["base_table_score"] = round(base_score, 6)
        current["table_score"] = round(_clamp(base_score + evidence_bonus), 6)
        current["provenance_count"] = provenance_count
        current["source_titles"] = source_titles
        current["source_doc_ids"] = source_doc_ids
    merged_rows = sorted(
        by_key.values(),
        key=lambda row: (
            -int(bool(row.get("production_eligible"))),
            -float(row.get("table_score", 0.0) or 0.0),
            -float(row.get("anchor_score", 0.0) or 0.0),
            int(row.get("rank", 999999) or 999999),
            str(row.get("candidate", "")).lower(),
        ),
    )
    return merged_rows[: max(1, int(max_rows))]


def _endpoint_overlap(tokens: Set[str], required: Set[str], max_required: int = 4) -> float:
    return _bounded_overlap(tokens, required, max_required=max_required)


def _is_parent_edge(item: Dict[str, Any]) -> bool:
    rel = set(item.get("relation_tokens") or set())
    return bool(rel & PARENT_RELATION_TERMS or rel & CHILD_RELATION_TERMS)


def _parent_edge_orientation(item: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    """Return (parent_side, child_side, parent_kind) for kinship parent facts."""
    rel = set(item.get("relation_tokens") or set())
    if rel & {"father", "paternal"}:
        return "subject", "object", "father"
    if rel & {"mother", "maternal"}:
        return "subject", "object", "mother"
    if rel & {"parent"}:
        return "subject", "object", "parent"
    if rel & CHILD_RELATION_TERMS:
        return "object", "subject", "parent"
    return None


def _edge_side_text(item: Dict[str, Any], side: str) -> str:
    if side == "subject":
        return str(item.get("subject") or "")
    return str(item.get("object") or "")


def _edge_side_tokens(item: Dict[str, Any], side: str) -> Set[str]:
    if side == "subject":
        return set(item.get("subject_tokens") or set())
    return set(item.get("object_tokens") or set())


def _relation_has_any(item: Dict[str, Any], terms: Set[str]) -> bool:
    return bool(set(item.get("relation_tokens") or set()) & terms)


def _candidate_surface_pattern(candidate: str) -> Optional[re.Pattern]:
    tokens = [
        token
        for token in TOKEN_RE.findall(str(candidate or "").lower())
        if token not in STOPWORDS and len(token) > 1
    ]
    if not tokens:
        return None


    pattern = r"\b" + r"[^A-Za-z0-9]+".join(re.escape(token) for token in tokens) + r"\b"
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None


def _split_surface_sentences(text: str) -> List[str]:
    sentences = []
    for piece in re.split(r"(?<=[.!?])\s+|\n+", str(text or "")):
        piece = piece.strip()
        if 20 <= len(piece) <= 1200:
            sentences.append(piece)
    return sentences


def _clean_surface_value(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n,;:.()[]{}\"'")


def _expand_place_surface(sentence: str, match: re.Match) -> str:
    start, end = match.span()
    left = start
    right = end



    for _ in range(2):
        prefix = sentence[:left]
        m = re.search(r"(" + SURFACE_PROPER_SEGMENT + r"),\s*$", prefix)
        if not m:
            break
        segment = m.group(1)
        if DATE_VALUE_RE.search(segment) or NUMBER_VALUE_RE.search(segment):
            break
        left = m.start(1)




    for _ in range(3):
        suffix = sentence[right:]
        m = re.match(r"\s*,\s*(" + SURFACE_PROPER_SEGMENT + r")", suffix)
        if not m:
            break
        segment = m.group(1).strip()
        if not segment:
            break
        if DATE_VALUE_RE.search(segment) or NUMBER_VALUE_RE.search(segment):
            break

        segment_tokens = segment.split()
        if not any(token[:1].isupper() or token.isupper() for token in segment_tokens):
            break
        right += m.end()

    surface = _clean_surface_value(sentence[left:right])
    if len(surface) > 96:
        return ""
    return surface


def _extract_date_surface(sentence: str, candidate: str) -> str:
    year_match = DATE_VALUE_RE.search(str(candidate or ""))
    year = year_match.group(0) if year_match else ""
    date_patterns = [
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        r"\b\d{4}\b",
    ]
    for pattern in date_patterns:
        for match in re.finditer(pattern, sentence, re.IGNORECASE):
            surface = _clean_surface_value(match.group(0))
            if not surface:
                continue
            if year and year not in surface:
                continue
            if not year and not (_tokens(candidate) & _tokens(surface)):
                continue
            return surface
    return ""


def _materialize_candidate_surface(
    *,
    candidate: str,
    answer_type: str,
    path_items: List[Dict[str, Any]],
    get_doc_content: Callable[[int], str],
) -> Tuple[str, Dict[str, Any]]:
    info = {
        "enabled": True,
        "applied": False,
        "reason": "not_materialized",
        "raw_candidate": str(candidate or "").strip(),
        "surface_candidate": "",
    }
    raw_candidate = str(candidate or "").strip()
    if not raw_candidate:
        info["reason"] = "empty_candidate"
        return raw_candidate, info
    if answer_type not in {"place", "date"}:
        info["reason"] = "unsupported_answer_type"
        return raw_candidate, info
    pattern = _candidate_surface_pattern(raw_candidate)
    if pattern is None and answer_type != "date":
        info["reason"] = "no_candidate_pattern"
        return raw_candidate, info

    candidate_tokens = _tokens(raw_candidate)
    best_surface = ""
    best_doc_id: Optional[int] = None
    for item in reversed(path_items or []):
        doc_id = _fact_doc_id(item)
        if doc_id is None:
            continue
        try:
            content = get_doc_content(doc_id)
        except Exception:
            continue
        for sentence in _split_surface_sentences(content):
            if answer_type == "date":
                surface = _extract_date_surface(sentence, raw_candidate)
            else:
                if pattern is None:
                    continue
                match = pattern.search(sentence)
                if not match:

                    if candidate_tokens and not candidate_tokens.issubset(_tokens(sentence)):
                        continue
                    continue
                surface = _expand_place_surface(sentence, match)
            surface = _clean_surface_value(surface)
            if not surface:
                continue
            surface_tokens = _tokens(surface)
            if candidate_tokens and not (candidate_tokens & surface_tokens):
                continue
            if len(surface) < len(raw_candidate):
                continue
            if not best_surface or len(surface) > len(best_surface):
                best_surface = surface
                best_doc_id = doc_id
    if not best_surface or best_surface.lower() == raw_candidate.lower():
        info["reason"] = "no_better_surface"
        return raw_candidate, info

    if answer_type == "date" and not SURFACE_MONTH_RE.search(best_surface) and len(_tokens(raw_candidate)) >= len(_tokens(best_surface)):
        info["reason"] = "date_surface_not_more_specific"
        return raw_candidate, info
    info.update({
        "applied": True,
        "reason": "surface_expanded",
        "surface_candidate": best_surface,
        "source_doc_id": best_doc_id,
    })
    return best_surface, info


_INDEX_ACCESS_COUNT_FIELDS = (
    "lookup_keys",
    "hit_keys",
    "alias_hit_keys",
    "input_scan_rows",
    "rows_after_cap",
    "truncated_rows",
    "fanout_rejected_keys",
    "fanout_rejected_rows",
    "duplicate_rows",
    "deduplicated_rows",
    "loaded_doc_rows",
    "budget_truncated_rows",
)


def _new_index_access_stats() -> Dict[str, int]:
    return {field: 0 for field in _INDEX_ACCESS_COUNT_FIELDS}


def _merge_index_access_stats(*stats_rows: Mapping[str, Any]) -> Dict[str, int]:
    return {
        field: sum(int(row.get(field, 0) or 0) for row in stats_rows)
        for field in _INDEX_ACCESS_COUNT_FIELDS
    }


def _lookup_items_by_entity_tokens(
    relation_index: Dict[str, Any],
    entity_tokens: Set[str],
    *,
    max_per_key: int,
    max_fanout: int,
    access_stats: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    entity_to_items = relation_index.get("entity_to_items") or {}
    items = relation_index.get("items") or []
    raw_ids: List[int] = []
    for key in sorted(entity_tokens):
        if access_stats is not None:
            access_stats["lookup_keys"] += 1
        values = list(entity_to_items.get(key) or [])
        if not values:
            continue
        if access_stats is not None:
            access_stats["hit_keys"] += 1
        if max_fanout > 0 and len(values) > max_fanout:
            if access_stats is not None:
                access_stats["fanout_rejected_keys"] += 1
                access_stats["fanout_rejected_rows"] += len(values)
            continue
        capped = values[: max(1, int(max_per_key))]
        raw_ids.extend(capped)
        if access_stats is not None:
            access_stats["input_scan_rows"] += len(capped)
            access_stats["rows_after_cap"] += len(capped)
            access_stats["truncated_rows"] += max(0, len(values) - len(capped))
    result = []
    seen = set()
    for item_id in raw_ids:
        try:
            idx = int(item_id)
        except Exception:
            continue
        if idx in seen or idx < 0 or idx >= len(items):
            continue
        seen.add(idx)
        result.append(items[idx])
    if access_stats is not None:
        access_stats["duplicate_rows"] += max(0, len(raw_ids) - len(result))
        access_stats["deduplicated_rows"] += len(result)
    return result


def _lookup_items_by_exact_entity_key(
    relation_index: Dict[str, Any],
    entity_key: str,
    *,
    max_per_key: int,
    max_fanout: int,
    access_stats: Optional[Dict[str, int]] = None,
    allow_title_provenance_alias: bool = True,
    max_total_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Probe endpoint indexes using one key and bounded provenance aliases.

    Explicit identity facts are safe for the normal physical plan.  Title
    aliases are a speculative fallback and must be opt-in at the call site;
    the default remains permissive for backwards-compatible diagnostic tests.
    """
    entity_key = str(entity_key or "").strip()
    if not entity_key:
        return []
    probe_keys = _equivalent_entity_keys(
        relation_index,
        entity_key,
        allow_title_provenance_alias=allow_title_provenance_alias,
    )
    raw_ids: List[int] = []
    for probe_key in probe_keys:
        for index_name in ("subject_to_items", "object_to_items"):
            if access_stats is not None:
                access_stats["lookup_keys"] += 1
            values = list(
                (relation_index.get(index_name) or {}).get(probe_key) or []
            )
            if not values:
                continue
            if access_stats is not None:
                access_stats["hit_keys"] += 1
                if probe_key != entity_key:
                    access_stats["alias_hit_keys"] = int(
                        access_stats.get("alias_hit_keys", 0) or 0
                    ) + 1
            if max_fanout > 0 and len(values) > max_fanout:
                if access_stats is not None:
                    access_stats["fanout_rejected_keys"] += 1
                    access_stats["fanout_rejected_rows"] += len(values)
                continue
            per_key_limit = max(1, int(max_per_key))
            if max_total_rows is not None:
                remaining = max(0, int(max_total_rows) - len(raw_ids))
                if remaining <= 0:
                    if access_stats is not None:
                        access_stats["budget_truncated_rows"] += len(values)
                    continue
                per_key_limit = min(per_key_limit, remaining)
            capped = values[:per_key_limit]
            raw_ids.extend(capped)
            if access_stats is not None:
                access_stats["input_scan_rows"] += len(capped)
                access_stats["rows_after_cap"] += len(capped)
                access_stats["truncated_rows"] += max(
                    0, len(values) - len(capped)
                )
                if max_total_rows is not None and len(capped) < min(
                    len(values), max(1, int(max_per_key))
                ):
                    access_stats["budget_truncated_rows"] += min(
                        len(values), max(1, int(max_per_key))
                    ) - len(capped)
    items = relation_index.get("items") or []
    result: List[Dict[str, Any]] = []
    seen: Set[int] = set()
    for raw_id in raw_ids:
        try:
            item_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if item_id in seen or item_id < 0 or item_id >= len(items):
            continue
        seen.add(item_id)
        result.append(items[item_id])
    if access_stats is not None:
        access_stats["duplicate_rows"] += max(0, len(raw_ids) - len(result))
        access_stats["deduplicated_rows"] += len(result)
    return result


def _equivalent_entity_keys(
    relation_index: Dict[str, Any],
    entity_key: str,
    *,
    max_keys: int = 16,
    allow_title_provenance_alias: bool = False,
) -> Set[str]:
    """Return a bounded alias class for an entity key.

    Explicit identity aliases are part of the ordinary fact-store contract.
    Title-derived aliases are deliberately kept behind an opt-in switch so a
    speculative title fallback cannot silently change the legacy root binding.
    """
    entity_key = str(entity_key or "").strip()
    if not entity_key:
        return set()
    alias_maps = [relation_index.get("alias_equivalent_keys") or {}]
    if allow_title_provenance_alias:
        alias_maps.append(
            relation_index.get("title_alias_equivalent_keys") or {}
        )
    found = {entity_key}
    frontier = [entity_key]
    while frontier and len(found) < max(1, int(max_keys)):
        current = frontier.pop(0)
        aliases: List[str] = []
        for alias_map in alias_maps:
            aliases.extend(alias_map.get(current) or [])
        for alias in dict.fromkeys(aliases):
            alias = str(alias or "").strip()
            if not alias or alias in found:
                continue
            found.add(alias)
            frontier.append(alias)
            if len(found) >= max(1, int(max_keys)):
                break
    return found


def role_aware_predicate_contract_compatibility(
    item: Mapping[str, Any],
    operator: Mapping[str, Any],
    *,
    input_side: str = "",
    output_side: str = "",
) -> Dict[str, Any]:
    """Evaluate one fact against the predicate and directional role contract.

    LOOKUP_JOIN and VERIFY must use the same compatibility semantics.  A
    canonical ``child_of`` request, for example, may consume a source tuple
    expressed as ``mother_of`` or ``son_of`` only when its physical endpoints
    agree with the bound input/output sides.  Broader relation-family overlap
    alone is never sufficient.
    """
    relation_tokens = set(item.get("relation_tokens") or set())
    predicate_families = set(
        item.get("predicate_families")
        or _canonical_predicate_families_for_tokens(relation_tokens)
    )
    relation_families = _relation_families_for_tokens(relation_tokens)
    required_tokens = set(operator.get("relation_tokens") or [])
    required_predicates = set(operator.get("predicate_families") or [])
    required_families = set(operator.get("relation_families") or [])
    required_subtypes = set(
        operator.get("required_predicate_subtypes") or []
    )
    if required_subtypes:
        item_subtypes = set(item.get("predicate_subtypes") or [])
        if not item_subtypes:
            triple = list(item.get("triple") or [])
            item_subtypes = set(canonical_schema_predicate_subtypes(
                item.get("relation") or (triple[1] if len(triple) == 3 else ""),
                item.get("object") or (triple[2] if len(triple) == 3 else ""),
            ))
        if not item_subtypes & required_subtypes:
            return {
                "compatible": False,
                "reason": "predicate_subtype_mismatch",
                "orientations": [],
            }
    if required_predicates:



        direct_family_match = bool(
            predicate_families & required_predicates
            or relation_tokens & required_tokens
        )
        if direct_family_match:
            return {
                "compatible": True,
                "reason": "direct_predicate_family_match",
                "orientations": [],
            }







        orientations = (
            _role_orientations_for_operator_item(item, operator)
            if _operator_requires_role_direction(operator)
            else []
        )
        if input_side or output_side:
            orientations = [
                orientation for orientation in orientations
                if (not input_side or orientation[0] == input_side)
                and (not output_side or orientation[1] == output_side)
            ]
        if orientations:
            return {
                "compatible": True,
                "reason": "directional_role_compatible",
                "orientations": [list(value) for value in orientations],
            }
        return {
            "compatible": False,
            "reason": (
                "directional_role_mismatch"
                if _operator_requires_role_direction(operator)
                else "predicate_family_mismatch"
            ),
            "orientations": [],
        }
    fallback_match = bool(
        relation_tokens & required_tokens
        or relation_families & required_families
    )
    return {
        "compatible": fallback_match,
        "reason": (
            "relation_token_or_family_match"
            if fallback_match
            else "predicate_family_mismatch"
        ),
        "orientations": [],
    }


def _plan_operator_predicate_matches_item(
    item: Dict[str, Any],
    operator: Dict[str, Any],
) -> bool:
    return bool(
        role_aware_predicate_contract_compatibility(item, operator).get(
            "compatible"
        )
    )


def _plan_operator_matches_item(
    item: Dict[str, Any],
    operator: Dict[str, Any],
) -> bool:
    """Match both the predicate contract and any directional role contract."""
    if not _plan_operator_predicate_matches_item(item, operator):
        return False
    role_required = _operator_requires_role_direction(operator)
    if not role_required:
        return True
    return bool(_role_orientations_for_operator_item(item, operator))


def _operator_uses_explicit_physical_endpoint_contract(
    operator: Mapping[str, Any],
) -> bool:
    """Return whether an operator claims a compiler-owned V2 tuple contract.

    Field validity is checked by the explicit execution path.  A malformed V2
    operator must be rejected there instead of silently falling back to legacy
    direction inference.
    """
    lowering_version = str(operator.get("lowering_version") or "")
    schema_version = str(operator.get("predicate_endpoint_schema_version") or "")
    return bool(
        lowering_version.startswith("deterministic_physical_lowering_v2")
        and schema_version.startswith("predicate_endpoint_schema_v2")
    )


def _explicit_physical_endpoint_for_operator_binding(
    item: Dict[str, Any],
    operator: Dict[str, Any],
    input_key: str,
    equivalent_keys: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Execute a V2 lowering contract against the stored tuple endpoints.

    V2 lowering derives tuple sides from the endpoint schema.  Reinterpreting
    those sides through the legacy relation-token role table can reverse a
    correctly lowered operator, so this path consumes the physical contract
    directly.  Operators without the V2 marker continue to use the legacy
    behavior below.
    """
    if not _plan_operator_predicate_matches_item(item, operator):
        return None
    accepted_keys = set(equivalent_keys or set()) | {str(input_key or "")}
    accepted_keys.discard("")
    subject_key = str(item.get("subject_key") or "")
    object_key = str(item.get("object_key") or "")
    input_side = str(operator.get("physical_input_side") or "")
    output_side = str(operator.get("physical_output_side") or "")
    if input_side not in {"subject", "object", "either"}:
        return None
    if output_side not in {"subject", "object", "opposite_endpoint"}:
        return None

    if input_side == "either":
        if output_side != "opposite_endpoint":
            return None
        output = _other_endpoint_for_exact_binding(
            item,
            input_key,
            equivalent_keys=accepted_keys,
        )
        if output is None:
            return None
        output_surface, output_key = output
        bound_side = "subject" if subject_key in accepted_keys else "object"
        resolved_output_side = "object" if bound_side == "subject" else "subject"
    else:
        if output_side not in {"subject", "object"} or input_side == output_side:
            return None
        bound_key = subject_key if input_side == "subject" else object_key
        if not bound_key or bound_key not in accepted_keys:
            return None
        output_key = subject_key if output_side == "subject" else object_key
        output_surface = _edge_side_text(item, output_side)
        if not output_key or not output_surface:
            return None
        bound_side = input_side
        resolved_output_side = output_side

    return {
        "surface": output_surface,
        "key": output_key,
        "input_side": bound_side,
        "output_side": resolved_output_side,
        "fact_relation": _canonical_fact_role_relation(item),
        "requested_relation": str(operator.get("canonical_relation") or ""),
        "role_direction_required": bool(
            operator.get("role_direction_required")
        ),
        "role_direction_validated": True,
        "role_direction_confidence": 1.0,
        "physical_endpoint_contract_used": True,
        "physical_lowering_version": str(operator.get("lowering_version") or ""),
    }


def _other_endpoint_for_exact_binding(
    item: Dict[str, Any],
    input_key: str,
    equivalent_keys: Optional[Set[str]] = None,
) -> Optional[Tuple[str, str]]:
    """Return (surface, key) only when exactly one endpoint binds input_key."""
    subject_key = str(item.get("subject_key") or "")
    object_key = str(item.get("object_key") or "")
    accepted_keys = set(equivalent_keys or set()) | {str(input_key or "")}
    subject_match = bool(subject_key and subject_key in accepted_keys)
    object_match = bool(object_key and object_key in accepted_keys)
    if subject_match == object_match:
        return None
    if subject_match:
        return str(item.get("object") or ""), object_key
    return str(item.get("subject") or ""), subject_key


def _oriented_endpoint_for_operator_binding(
    item: Dict[str, Any],
    operator: Dict[str, Any],
    input_key: str,
    equivalent_keys: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Bind one fact according to the operator's endpoint-role contract."""
    if _operator_uses_explicit_physical_endpoint_contract(operator):
        return _explicit_physical_endpoint_for_operator_binding(
            item,
            operator,
            input_key,
            equivalent_keys=equivalent_keys,
        )
    if not _plan_operator_matches_item(item, operator):
        return None
    accepted_keys = set(equivalent_keys or set()) | {str(input_key or "")}
    subject_key = str(item.get("subject_key") or "")
    object_key = str(item.get("object_key") or "")
    role_required = _operator_requires_role_direction(operator)
    if role_required:
        for input_side, output_side, fact_relation in (
            _role_orientations_for_operator_item(item, operator)
        ):
            bound_key = subject_key if input_side == "subject" else object_key
            if not bound_key or bound_key not in accepted_keys:
                continue
            output_surface = _edge_side_text(item, output_side)
            output_key = subject_key if output_side == "subject" else object_key
            if not output_surface or not output_key:
                continue
            return {
                "surface": output_surface,
                "key": output_key,
                "input_side": input_side,
                "output_side": output_side,
                "fact_relation": fact_relation,
                "requested_relation": str(
                    operator.get("canonical_relation") or ""
                ),
                "role_direction_required": True,
                "role_direction_validated": True,
                "role_direction_confidence": 1.0,
            }
        return None

    output = _other_endpoint_for_exact_binding(
        item,
        input_key,
        equivalent_keys=accepted_keys,
    )
    if output is None:
        return None
    surface, output_key = output
    return {
        "surface": surface,
        "key": output_key,
        "input_side": "subject" if subject_key in accepted_keys else "object",
        "output_side": "object" if subject_key in accepted_keys else "subject",
        "fact_relation": "",
        "requested_relation": "",
        "role_direction_required": False,
        "role_direction_validated": True,
        "role_direction_confidence": 0.65,
    }


def _multi_hop_plan_names(
    question: str,
    gap: Dict[str, Any],
    stage_aware_plan: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if (
        stage_aware_plan
        and stage_aware_plan.get("version") in {
            "slot_dependency_query_plan_v3",
            "schema_constrained_query_plan_v4",
        }
        and stage_aware_plan.get("compile_status") == "compiled"
    ):
        return [str(stage_aware_plan.get("version"))]
    q = str(question or "").lower()
    plans: List[str] = []
    if re.search(r"\b(?:paternal|maternal)?\s*grand(?:father|mother|parent)\b", q):
        plans.append("kinship_grandparent_2hop")
    if re.search(r"\b(?:place of birth|birthplace|born in|where .{0,80}born)\b", q) and re.search(
        r"\b(?:director|writer|author|composer|producer|creator|founder|designer)\b",
        q,
    ):
        plans.append("creator_birth_place_2hop")
    if re.search(r"\b(?:date of birth|born when|when .{0,80}born)\b", q) and re.search(
        r"\b(?:director|writer|author|composer|producer|creator|founder|designer)\b",
        q,
    ):
        plans.append("creator_birth_date_2hop")
    if (
        stage_aware_plan
        and stage_aware_plan.get("requires_bridge")
        and stage_aware_plan.get("terminal_predicate_families")
    ):
        plans.append("stage_aware_nested_attribute_2hop")
    return list(dict.fromkeys(plans))


def _kinship_parent_kind_allowed(kind: str, required: str) -> bool:
    if required == "father":
        return kind == "father"
    if required == "mother":
        return kind == "mother"
    return kind in {"father", "mother", "parent"}


def _build_multi_hop_row(
    *,
    candidate: str,
    answer_type: str,
    plan: str,
    anchor_entity: str,
    bridge_entity: str,
    path_items: List[Dict[str, Any]],
    path_roles: List[str],
    anchor_score: float,
    role_score: float,
    rank_map: Dict[int, int],
    get_doc_content: Callable[[int], str],
    enable_surface_materializer: bool,
) -> Optional[Dict[str, Any]]:
    raw_candidate_text = str(candidate or "").strip()
    candidate_text = raw_candidate_text
    surface_info: Dict[str, Any] = {
        "enabled": bool(enable_surface_materializer),
        "applied": False,
        "reason": "disabled" if not enable_surface_materializer else "not_evaluated",
        "raw_candidate": raw_candidate_text,
        "surface_candidate": "",
    }
    if enable_surface_materializer:
        candidate_text, surface_info = _materialize_candidate_surface(
            candidate=raw_candidate_text,
            answer_type=answer_type,
            path_items=path_items,
            get_doc_content=get_doc_content,
        )
    if not candidate_text or len(candidate_text) > 160:
        return None
    titles = []
    triples = []
    evidence_parts = []
    ranks = []
    doc_ids = []
    for item in path_items:
        doc_id = _fact_doc_id(item)
        if doc_id is None:
            return None
        doc_ids.append(doc_id)
        ranks.append(int(rank_map.get(doc_id, len(rank_map) + 100)))
        triples.append(" | ".join(str(part) for part in (item.get("triple") or [])))
        try:
            content = get_doc_content(doc_id)
        except Exception:
            content = ""
        title = str(content).split("\n", 1)[0].strip()
        if title:
            titles.append(title)
        if content:
            evidence_parts.append(str(content).replace("\n", " ")[:180])
    best_rank = min(ranks or [999999])
    answer_doc_id = doc_ids[-1] if doc_ids else -1
    answer_rank = ranks[-1] if ranks else 999999
    rank_prior = 1.0 / (1.0 + math.log1p(best_rank + 1))
    answer_rank_prior = 1.0 / (1.0 + math.log1p(answer_rank + 1))
    provenance = len(set(doc_ids))
    cost = _clamp(0.04 + 0.01 * len(path_items) + 0.004 * best_rank)
    table_score = _clamp(
        0.34 * anchor_score
        + 0.26 * role_score
        + 0.14 * min(1.0, provenance / 2.0)
        + 0.12 * rank_prior
        + 0.08 * (1.0 if len(path_items) >= 2 else 0.0)
        - 0.06 * cost
    )
    return {
        "candidate": candidate_text,
        "key": " ".join(sorted(_tokens(candidate_text)))[:120],
        "raw_candidate": raw_candidate_text,
        "anchor_entity": str(anchor_entity or "").strip(),
        "anchor_key": _entity_key(anchor_entity),
        "bridge_entity": str(bridge_entity or "").strip(),
        "bridge_key": _entity_key(bridge_entity),
        "answer_key": _entity_key(raw_candidate_text or candidate_text),
        "answer_type": answer_type,
        "template": plan,
        "surface_materializer": surface_info,
        "surface_materialized": bool(surface_info.get("applied")),
        "direction": "multi_hop_path",
        "path_roles": path_roles,
        "anchor_score": round(float(anchor_score), 6),
        "role_score": round(float(role_score), 6),
        "question_overlap": 1.0,
        "rank_prior": round(float(rank_prior), 6),
        "answer_rank_prior": round(float(answer_rank_prior), 6),
        "cost": round(float(cost), 6),
        "table_score": round(float(table_score), 6),
        "rank": best_rank,
        "answer_rank": int(answer_rank),
        "doc_id": doc_ids[0] if doc_ids else -1,
        "answer_doc_id": int(answer_doc_id),
        "path_doc_ids": doc_ids,
        "path_ranks": ranks,
        "title": " ; ".join(titles[:2]),
        "source_titles": titles[:4],
        "source_doc_ids": list(dict.fromkeys(doc_ids)),
        "path_item_ids": [
            int(item.get("item_id", -1)) for item in path_items
        ],
        "path_tuple_keys": [
            str(item.get("tuple_key") or "") for item in path_items
        ],
        "path_triples": [
            [str(value or "") for value in (item.get("triple") or [])]
            for item in path_items
        ],
        "source_namespace": "base_passage",
        "lineage_source_title": " ; ".join(titles[:2]),
        "lineage_source_doc_id": answer_doc_id,
        "lineage_sentence_preview": " || ".join(evidence_parts[:2])[:320],
        "derived_namespace": "relation_fact_join_candidate_table",
        "provenance_count": provenance,
        "triple": " -> ".join(triples[:3]),
        "evidence": " || ".join(evidence_parts[:2]),
    }


def _bound_completion_query(
    bound_surface: str,
    operator: Dict[str, Any],
    *,
    structured: bool = True,
) -> str:
    bound_surface = str(bound_surface or "").strip()
    canonical_relation = str(
        operator.get("canonical_relation") or ""
    ).strip().lower()
    output_type = str(operator.get("output_type") or "entity").strip().lower()
    role_queries = {
        "mother_of": f"Who was the mother of {bound_surface}?",
        "father_of": f"Who was the father of {bound_surface}?",
        "parent_of": f"Who was a parent of {bound_surface}?",
        "daughter_of": f"Who was a daughter of {bound_surface}?",
        "son_of": f"Who was a son of {bound_surface}?",
        "child_of": f"Who was a child of {bound_surface}?",
        "spouse_of": f"Who was the spouse of {bound_surface}?",
        "wife_of": f"Who was the wife of {bound_surface}?",
        "husband_of": f"Who was the husband of {bound_surface}?",
    }
    if structured and canonical_relation in role_queries:
        return role_queries[canonical_relation]

    predicate_families = set(operator.get("predicate_families") or [])
    if structured and "birth" in predicate_families:
        if output_type == "date":
            return f"When was {bound_surface} born?"
        return f"Where was {bound_surface} born?"
    if structured and "death" in predicate_families:
        if output_type == "date":
            return f"When did {bound_surface} die?"
        return f"Where did {bound_surface} die?"






    if structured:
        creator_families = predicate_families & {
            "authorship", "composition", "creation", "design", "direction",
            "production", "visual_art",
        }
        if (
            len(creator_families) > 1
            or "creator" in set(operator.get("relation_tokens") or [])
        ):
            return f"Who created {bound_surface}?"
        if "authorship" in predicate_families:
            return f"Who wrote {bound_surface}?"
        if "composition" in predicate_families:
            return f"Who composed {bound_surface}?"
        if "creation" in predicate_families:
            return f"Who created {bound_surface}?"
        if "design" in predicate_families:
            return f"Who designed {bound_surface}?"
        if "direction" in predicate_families:
            return f"Who directed {bound_surface}?"
        if "production" in predicate_families:
            return f"Who produced {bound_surface}?"
        if "visual_art" in predicate_families:
            return f"Who painted {bound_surface}?"
        if "performance" in predicate_families:
            return f"Who performed or recorded {bound_surface}?"
        if "founding" in predicate_families:
            if output_type == "date":
                return f"When was {bound_surface} founded?"
            return f"Who founded {bound_surface}?"
        if "education" in predicate_families:
            return f"Where was {bound_surface} educated?"
        if predicate_families & {"containment", "geopolitical_location"}:
            target_categories = [
                str(value).strip().lower()
                for value in operator.get("target_entity_categories") or []
                if str(value).strip()
            ]
            if len(target_categories) == 1:
                return (
                    f"Which {target_categories[0]} is {bound_surface} "
                    "located in?"
                )
            return f"Where is {bound_surface} located?"
        if "nationality" in predicate_families:
            return f"What is the nationality of {bound_surface}?"
        if "membership" in predicate_families:
            return f"Which organization was {bound_surface} a member of?"
        if "distribution" in predicate_families and output_type == "date":
            return f"When was {bound_surface} released or premiered?"

    relation_terms = list(operator.get("relation_tokens") or [])
    if not relation_terms:
        relation_terms = list(operator.get("predicate_families") or [])
    return " ".join(
        part
        for part in (
            bound_surface,
            " ".join(sorted(set(relation_terms))),
            output_type,
        )
        if part
    )


def _bound_completion_admission(
    *,
    bound_surface: str,
    operator: Dict[str, Any],
    call_count: int,
    max_calls: int,
    top_k: int,
    max_risk: float,
    min_net_utility: float,
    compute_weight: float,
    risk_weight: float,
    causal_predecessor: bool = False,
    terminal_operator: bool = False,
) -> Dict[str, Any]:
    """Choose a completion probe under explicit benefit/risk/cost bounds."""
    predicate_families = set(operator.get("predicate_families") or [])
    relation_tokens = set(operator.get("relation_tokens") or [])
    output_type = str(operator.get("output_type") or "entity")
    bound_width = len(_tokens(bound_surface))
    role_contract = _canonical_operator_role_contract(relation_tokens)
    role_direction_required = _operator_requires_role_direction(operator)
    utility_components = {
        "index_miss": 0.34,
        "predicate_contract": 0.20 if predicate_families else 0.08,
        "bound_entity": 0.16 if bound_width >= 2 else 0.08,
        "typed_output": 0.12 if output_type != "entity" else 0.05,
        "role_direction_contract": 0.10 if role_direction_required else 0.0,
        "causal_residual_chain": 0.12 if causal_predecessor else 0.0,
        "terminal_projection": 0.08 if terminal_operator else 0.0,
    }
    risk_components = {
        "untyped_predicate": 0.0 if predicate_families else 0.20,
        "generic_output": 0.08 if output_type == "entity" else 0.02,
        "short_binding": 0.10 if bound_width <= 1 else 0.02,
        "weak_relation_contract": 0.12 if not relation_tokens else 0.02,
        "missing_role_direction": (
            0.18
            if "kinship" in set(operator.get("relation_families") or [])
            and not role_direction_required
            else 0.0
        ),
    }
    utility = _clamp(sum(utility_components.values()))
    risk = _clamp(sum(risk_components.values()))
    normalized_cost = _clamp(0.05 + 0.015 * max(1, int(top_k)))
    net_utility = (
        utility
        - float(risk_weight) * risk
        - float(compute_weight) * normalized_cost
    )
    reasons = []
    if int(call_count) >= max(0, int(max_calls)):
        reasons.append("completion_call_budget_exhausted")
    if not str(bound_surface or "").strip():
        reasons.append("missing_bound_surface")
    if not relation_tokens and not predicate_families:
        reasons.append("missing_predicate_contract")
    if risk > float(max_risk):
        reasons.append("risk_budget_exceeded")
    if net_utility < float(min_net_utility):
        reasons.append("net_utility_below_threshold")
    return {
        "accepted": not reasons,
        "reason": "accepted" if not reasons else "+".join(reasons),
        "utility": round(float(utility), 6),
        "risk": round(float(risk), 6),
        "normalized_cost": round(float(normalized_cost), 6),
        "net_utility": round(float(net_utility), 6),
        "utility_components": utility_components,
        "risk_components": risk_components,
        "call_count_before": int(call_count),
        "max_calls": int(max_calls),
        "top_k": int(top_k),
        "causal_predecessor": bool(causal_predecessor),
        "terminal_operator": bool(terminal_operator),
        "canonical_relation": str(
            operator.get("canonical_relation")
            or role_contract.get("canonical_relation")
            or ""
        ),
        "role_direction_required": role_direction_required,
        "api_calls": 0,
    }


def _completion_fact_item(
    *,
    subject: str,
    relation: str,
    obj: str,
    doc_id: int,
    item_id: int,
    sentence: str,
    source_sentence_id: int = -1,
    bound_input_key: str = "",
    extraction_direction: str = "forward",
    instance_identity_mode: str = "explicit_bound_mention",
    instance_identity_validated: bool = True,
) -> Optional[Dict[str, Any]]:
    subject = str(subject or "").strip(" \t\n\r,;:-")
    relation = str(relation or "").strip()
    obj = str(obj or "").strip(" \t\n\r,;:-")
    if not subject or not relation or not obj:
        return None
    subject_key = _entity_key(subject)
    object_key = _entity_key(obj)
    if not subject_key or not object_key or subject_key == object_key:
        return None
    relation_tokens = _tokens(relation)
    if not relation_tokens:
        return None
    triple = (subject, relation, obj)
    tuple_key = "|".join(
        [subject_key, _relation_key(relation), object_key]
    )
    completion_fact_id = ":".join([
        "completion",
        str(int(doc_id)),
        str(int(source_sentence_id)),
        tuple_key,
    ])
    item = {
        "item_id": int(item_id),
        "doc_idx": int(doc_id),
        "chunk_id": f"bound_completion:{doc_id}",
        "triple": triple,
        "subject": subject,
        "relation": relation,
        "object": obj,
        "subject_tokens": _tokens(subject),
        "object_tokens": _tokens(obj),
        "endpoint_tokens": _tokens(subject) | _tokens(obj),
        "relation_tokens": relation_tokens,
        "predicate_families": set(
            canonical_schema_predicate_families(relation)
        ),
        "signature": _schema_signature(triple),
        "schema_signatures_v2": typed_schema_signatures_v2(triple),
        "subject_key": subject_key,
        "relation_key": _relation_key(relation),
        "object_key": object_key,
        "tuple_key": tuple_key,
        "entity_keys": {subject_key, object_key},
        "access_path": "bound_text_completion",
        "completion_sentence": str(sentence or "")[:320],
        "bound_completion_fact": True,
        "completion_fact_id": completion_fact_id,
        "completion_bound_input_key": str(bound_input_key or ""),
        "completion_source_doc_id": int(doc_id),
        "completion_source_sentence_id": int(source_sentence_id),
        "completion_extraction_direction": str(
            extraction_direction or "forward"
        ),
        "completion_instance_identity_mode": str(
            instance_identity_mode or "explicit_bound_mention"
        ),
        "completion_instance_identity_validated": bool(
            instance_identity_validated
        ),
    }
    item["canonical_role_relation"] = _canonical_fact_role_relation(item)
    return item


def _clean_completion_value(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" \t\n\r,;:-\"'")
    text = re.split(
        r"\s+(?:and|but|which|who|whose|where|when)\s+",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return text.strip(" \t\n\r,;:-\"'")[:120]


_ROMAN_IDENTITY_TOKENS = {
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii",
    "xix", "xx",
}


def _completion_identity_tokens(surface: Any) -> List[str]:
    """Return conservative name tokens for title-backed identity checks."""
    text = re.sub(r"\([^)]*\)", " ", str(surface or ""))
    tokens = _surface_word_tokens(text)
    return [
        token for token in tokens
        if not re.fullmatch(r"\d{3,4}", token)
        and not (len(token) == 1 and token not in _ROMAN_IDENTITY_TOKENS)
    ]


def _completion_identity_roman_tokens(surface: Any) -> Set[str]:
    return {
        token for token in _surface_word_tokens(surface)
        if token in _ROMAN_IDENTITY_TOKENS
    }


def _completion_identity_year_tokens(surface: Any) -> Set[str]:
    return set(re.findall(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b", str(surface or "")))


def _completion_title_identity_decision(
    *,
    bound_surface: str,
    bound_identity_keys: Set[str],
    title: str,
    enabled: bool,
) -> Tuple[str, str]:
    """Return the title identity mode and a machine-readable rejection reason."""
    title_key = _entity_key(title)
    if not enabled:
        if title_key and title_key in bound_identity_keys:
            return "title_or_alias_scoped", ""
        return "", "disabled"

    bound_roman = _completion_identity_roman_tokens(bound_surface)
    title_roman = _completion_identity_roman_tokens(title)
    if (bound_roman or title_roman) and bound_roman != title_roman:
        return "", "roman_conflict"

    bound_years = _completion_identity_year_tokens(bound_surface)
    title_years = _completion_identity_year_tokens(title)
    if bound_years and title_years and bound_years.isdisjoint(title_years):
        return "", "year_conflict"

    if title_key and title_key in bound_identity_keys:
        return "title_or_alias_scoped", ""

    title_tokens = _completion_identity_tokens(title)
    bound_tokens = _completion_identity_tokens(bound_surface)
    if len(title_tokens) < 2 or len(bound_tokens) < 2:
        return "", "insufficient_name_tokens"

    def contains(outer: List[str], inner: List[str]) -> bool:
        width = len(inner)
        return any(
            outer[offset:offset + width] == inner
            for offset in range(0, len(outer) - width + 1)
        )

    if contains(bound_tokens, title_tokens) or contains(title_tokens, bound_tokens):
        return "title_grounded_name_containment", ""




    shorter, longer = (
        (title_tokens, bound_tokens)
        if len(title_tokens) <= len(bound_tokens)
        else (bound_tokens, title_tokens)
    )
    if (
        len(shorter) >= 2
        and shorter[0] == longer[0]
        and shorter[-1] == longer[-1]
    ):
        cursor = 0
        for token in longer:
            if cursor < len(shorter) and token == shorter[cursor]:
                cursor += 1
        if cursor == len(shorter):
            return "title_grounded_name_subsequence", ""
    return "", "name_mismatch"


def _refine_legacy_root_with_title_identity(
    legacy_root: Dict[str, Any],
    title_root: Dict[str, Any],
) -> Dict[str, Any]:
    """Classify a nested legacy root as a title alias or an identity conflict.

    The legacy fact index often stores a useful edge under a shortened page
    title (for example ``Goodbye, Franziska``), while the title catalog binds
    the full query mention (``Goodbye, Franziska (1941 film)``).  Span
    containment alone cannot distinguish that benign alias from a one-token
    homonym such as ``Elizabeth`` inside ``Anne Elizabeth Rector``.  Reuse the
    conservative title identity contract so multi-token aliases keep their
    fact edges, while Roman numeral, year, and low-specificity conflicts remain
    vetoed.
    """
    legacy_surface = str(
        legacy_root.get("mention_surface")
        or legacy_root.get("surface")
        or legacy_root.get("canonical_surface")
        or ""
    ).strip()
    title_surface = str(
        title_root.get("canonical_surface")
        or title_root.get("surface")
        or title_root.get("mention_surface")
        or ""
    ).strip()
    identity_keys = {
        str(legacy_root.get("key") or "").strip(),
        str(legacy_root.get("mention_key") or "").strip(),
    }
    identity_keys.discard("")
    mode, reject_reason = _completion_title_identity_decision(
        bound_surface=legacy_surface,
        bound_identity_keys=identity_keys,
        title=title_surface,
        enabled=True,
    )
    compatible = bool(mode)
    return {
        "compatible": compatible,
        "mode": mode,
        "reason": "compatible_title_alias" if compatible else reject_reason,
        "legacy_surface": legacy_surface,
        "title_surface": title_surface,
        "canonical_identity_key": str(title_root.get("key") or ""),
    }


def _completion_title_identity_mode(
    *,
    bound_surface: str,
    bound_identity_keys: Set[str],
    title: str,
    enabled: bool,
) -> str:
    """Validate a retrieved page title as the currently bound instance.

    Exact fact-store aliases remain the strongest identity signal.  The
    title-grounded fallback only admits multi-token names whose normalized
    title is a contiguous part of the noisy bound surface (or vice versa).
    Roman numerals are retained, preventing Frederick II/III collisions.
    """
    mode, _ = _completion_title_identity_decision(
        bound_surface=bound_surface,
        bound_identity_keys=bound_identity_keys,
        title=title,
        enabled=enabled,
    )
    return mode


def _split_completion_sentences(content: str) -> List[str]:
    """Split completion passages without truncating initialed person names."""
    marker = "<S2RAG_INITIAL_DOT>"
    protected = re.sub(
        r"\b([A-Z])\.(?=\s*(?:[A-Z]\.|[A-Z][a-z]))",
        rf"\1{marker}",
        str(content or ""),
    )




    protected = re.sub(
        r"\b(ca?)\.(?=\s*\d)",
        rf"\1{marker}",
        protected,
        flags=re.IGNORECASE,
    )
    return [
        part.replace(marker, ".").strip()
        for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", protected)
        if part.strip()
    ]


def _split_completion_sentence_records(content: str) -> List[Dict[str, Any]]:
    """Return completion sentences with stable offsets into the source text."""
    source = str(content or "")
    records: List[Dict[str, Any]] = []
    cursor = 0
    for sentence_id, sentence in enumerate(_split_completion_sentences(source)):
        start = source.find(sentence, cursor)
        if start < 0:
            start = source.find(sentence)
        if start < 0:
            start = cursor
            end = min(len(source), start + len(sentence))
        else:
            end = start + len(sentence)
        records.append({
            "sentence_id": sentence_id,
            "sentence": sentence,
            "span_start": int(start),
            "span_end": int(end),
        })
        cursor = max(cursor, end)
    return records


def _completion_source_surface(sentence: Any, surface: Any) -> str:
    """Recover an exact source surface without fuzzy entity matching."""
    tokens = _surface_word_tokens(surface)
    if not tokens:
        return ""
    pattern = r"\b" + r"[\s'\u2019\-]+".join(
        re.escape(token) for token in tokens
    ) + r"\b"
    match = re.search(pattern, str(sentence or ""), flags=re.IGNORECASE)
    return match.group(0) if match else ""


def _completion_flexible_source_span(
    sentence: Any,
    surface: Any,
) -> Optional[Tuple[int, int, str]]:
    """Locate a source surface while tolerating punctuation between tokens.

    This is intentionally stricter than entity linking: every token must occur
    contiguously and in order in the same source sentence.  The punctuation
    tolerance is needed for initialed names such as ``Hans J. Salter``.
    """
    tokens = _surface_word_tokens(surface)
    if not tokens:
        return None
    pattern = r"\b" + r"[\W_]+".join(
        re.escape(token) for token in tokens
    ) + r"\b"
    match = re.search(pattern, str(sentence or ""), flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.start()), int(match.end()), match.group(0)


def _canonicalize_intermediate_completion_endpoint(
    *,
    value: str,
    sentence: str,
    operator: Mapping[str, Any],
    relation_index: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve a noisy intermediate COMPLETE value to an indexed endpoint.

    Sentence patterns can conservatively recover the requested relation while
    still capturing an appositive or role descriptor around its entity value,
    for example ``Wimbledon champion Gerald Patterson``.  Such a value is
    source-supported but cannot become the input binding of the next join.

    Canonicalization is allowed only when one unique indexed entity surface is
    an exact source span contained by the extracted value.  A second safe case
    completes a truncated initialed name from the same source start.  The
    helper never selects a different nearby entity and never changes terminal
    answers or scalar/place values.
    """
    raw_value = str(value or "").strip()
    decision: Dict[str, Any] = {
        "applied": False,
        "surface": raw_value,
        "key": _entity_key(raw_value),
        "reason": "unchanged",
        "candidate_count": 0,
    }
    if not raw_value or not relation_index:
        decision["reason"] = "missing_value_or_index"
        return decision
    slot_plan_stage = str(operator.get("slot_plan_stage") or "").strip()
    output_slot = str(operator.get("output_slot") or "").strip()
    is_intermediate = (
        slot_plan_stage.startswith("intermediate")
        or (bool(output_slot) and output_slot != "$answer")
    )
    if not is_intermediate or slot_plan_stage == "final_answer":
        decision["reason"] = "non_intermediate_output"
        return decision
    output_type = str(operator.get("output_type") or "entity").strip().lower()
    if output_type not in {
        "entity", "person", "organization", "work", "film", "book",
        "song", "album",
    }:
        decision["reason"] = "non_entity_output"
        return decision

    subject_index = relation_index.get("subject_to_items") or {}
    object_index = relation_index.get("object_to_items") or {}
    title_doc_ids = relation_index.get("title_entity_doc_ids") or {}
    title_surfaces = relation_index.get("title_entity_surfaces") or {}
    canonical_surfaces = relation_index.get("canonical_entity_surfaces") or {}
    indexed_keys = set(subject_index) | set(object_index) | set(title_doc_ids)
    raw_key = _entity_key(raw_value)
    raw_span = _completion_flexible_source_span(sentence, raw_value)
    if raw_span is None:
        decision["reason"] = "raw_value_not_source_aligned"
        return decision
    raw_start, raw_end, raw_source = raw_span
    raw_tokens = _surface_word_tokens(raw_source)
    short_person_endpoint = output_type == "person" and len(raw_tokens) == 1
    if raw_key and raw_key in indexed_keys and not short_person_endpoint:
        decision["reason"] = "already_joinable"
        return decision
    if len(raw_tokens) < 2 and not short_person_endpoint:
        decision["reason"] = "insufficient_raw_tokens"
        return decision




    token_matches = list(
        re.finditer(r"[^\W_]+", raw_source, flags=re.UNICODE)
    )
    candidate_keys: Set[str] = set()
    if len(raw_tokens) >= 2:
        for start in range(len(token_matches)):
            for end in range(start + 2, min(len(token_matches), start + 8) + 1):
                surface = raw_source[
                    token_matches[start].start():token_matches[end - 1].end()
                ]
                key = _entity_key(surface)
                if key and key != raw_key and key in indexed_keys:
                    candidate_keys.add(key)



    sentence_token_matches = list(
        re.finditer(r"[^\W_]+", str(sentence or ""), flags=re.UNICODE)
    )
    raw_token_start = next(
        (
            index
            for index, match in enumerate(sentence_token_matches)
            if match.start() == raw_start
        ),
        None,
    )
    if raw_token_start is not None:
        max_end = min(
            len(sentence_token_matches),
            raw_token_start + len(raw_tokens) + 4,
        )
        for end in range(raw_token_start + len(raw_tokens) + 1, max_end + 1):
            surface = str(sentence or "")[
                sentence_token_matches[raw_token_start].start():
                sentence_token_matches[end - 1].end()
            ]
            key = _entity_key(surface)
            if key and key in indexed_keys:
                candidate_keys.add(key)

    candidates: List[Dict[str, Any]] = []
    for key in sorted(candidate_keys):
        known_surfaces = list(dict.fromkeys(
            str(surface or "").strip()
            for surface in (
                title_surfaces.get(key),
                canonical_surfaces.get(key),
            )
            if str(surface or "").strip()
        ))
        for known_surface in known_surfaces:
            source_span = _completion_flexible_source_span(
                sentence, known_surface
            )
            if source_span is None:
                continue
            start, end, source_surface = source_span
            source_tokens = _surface_word_tokens(source_surface)
            contained = raw_start <= start and end <= raw_end
            same_start_extension = (
                start == raw_start
                and end > raw_end
                and source_tokens[:len(raw_tokens)] == raw_tokens
            )
            if not contained and not same_start_extension:
                continue
            title_backed = key in title_doc_ids
            if output_type == "person":




                if len(source_tokens) < 2 or any(
                    re.fullmatch(r"\d+(?:st|nd|rd|th)?", token)
                    for token in source_tokens
                ):
                    continue
                person_title_extension = bool(
                    title_backed
                    and same_start_extension
                    and re.search(
                        r"\b(?:baron|count|countess|crown\s+prince|"
                        r"crown\s+princess|duke|duchess|emperor|empress|"
                        r"king|prince|princess|queen|sir)\b",
                        source_surface,
                        re.IGNORECASE,
                    )
                )
                if not person_title_extension and _entity_type(source_surface) in {
                    "date", "number", "nationality", "place",
                    "organization", "work",
                }:
                    continue
                if contained and start > raw_start and end < raw_end:
                    prefix = str(sentence or "")[raw_start:start].strip()
                    suffix = str(sentence or "")[end:raw_end].strip()
                    person_titles = {
                        "dr", "doctor", "duke", "duchess", "emperor",
                        "empress", "king", "lady", "lord", "president",
                        "prince", "princess", "professor", "queen",
                        "regent", "sir",
                    }
                    prefix_tokens = set(_surface_word_tokens(prefix))
                    titled_geographic_suffix = (
                        bool(prefix_tokens)
                        and prefix_tokens <= person_titles
                        and bool(re.match(r"^of\b", suffix, re.IGNORECASE))
                    )
                    if not titled_geographic_suffix:
                        continue
            candidates.append({
                "key": key,
                "surface": source_surface,
                "start": int(start),
                "end": int(end),
                "mode": (
                    "same_start_truncated_name_extension"
                    if same_start_extension
                    else "contained_indexed_entity"
                ),
                "score": (
                    (100 if title_backed else 0)
                    + (30 if start == raw_start else 0)
                    + 3 * len(source_tokens)
                    - max(0, start - raw_start)
                ),
            })


    by_key: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        key = str(candidate["key"])
        previous = by_key.get(key)
        if previous is None or (
            int(candidate["score"]),
            len(str(candidate["surface"])),
        ) > (
            int(previous["score"]),
            len(str(previous["surface"])),
        ):
            by_key[key] = candidate
    ranked = sorted(
        by_key.values(),
        key=lambda candidate: (
            -int(candidate["score"]),
            -len(_surface_word_tokens(candidate["surface"])),
            str(candidate["key"]),
        ),
    )
    decision["candidate_count"] = len(ranked)
    if not ranked:
        decision["reason"] = "no_source_aligned_indexed_endpoint"
        return decision
    if len(ranked) > 1 and int(ranked[0]["score"]) == int(ranked[1]["score"]):
        decision["reason"] = "ambiguous_source_aligned_endpoints"
        return decision

    selected = ranked[0]
    decision.update({
        "applied": True,
        "surface": str(selected["surface"]),
        "key": str(selected["key"]),
        "reason": str(selected["mode"]),
    })
    return decision


def _canonical_completion_predicate(
    operator: Dict[str, Any], relation: str
) -> str:
    """Compile a surface relation to the predicate required by the plan."""
    explicit = str(operator.get("canonical_relation") or "").strip()
    if explicit:
        return explicit
    families = set(operator.get("predicate_families") or [])
    output_type = str(operator.get("output_type") or "entity").strip().lower()
    if "birth" in families:
        return "birth_date" if output_type == "date" else "birth_place"
    if "death" in families:
        return "death_date" if output_type == "date" else "death_place"
    if "nationality" in families:
        return "nationality"
    return _relation_key(relation).replace(" ", "_")


def _source_relation_surface(
    sentence: str,
    relation: str,
    value: str,
    predicate_families: Set[str],
) -> str:
    """Locate the lexical relation used by an extracted source certificate."""
    text = str(sentence or "")
    if "nationality" in predicate_families:
        object_surface = _completion_source_surface(text, value)
        if object_surface:
            prefix = text[: text.lower().find(object_surface.lower())]
            copulas = list(re.finditer(r"\b(?:is|was|are|were)\b", prefix, re.I))
            if copulas:
                return copulas[-1].group(0)
    if "performance" in predicate_families:
        object_surface = _completion_source_surface(text, value)
        if object_surface:
            prefix = text[: text.lower().find(object_surface.lower())]
            performance_relations = list(re.finditer(
                r"\b(?:performed\s+by|recorded\s+by|sung\s+by|"
                r"(?:music\s+)?artist|singer-songwriter|singer|performer|"
                r"vocalist)\b",
                prefix,
                flags=re.IGNORECASE,
            ))
            if performance_relations:
                return performance_relations[-1].group(0)
    if "visual_art" in predicate_families:
        object_surface = _completion_source_surface(text, value)
        if object_surface:
            prefix = text[: text.lower().find(object_surface.lower())]
            visual_art_relations = list(re.finditer(
                r"\b(?:painted\s+by|painting\s+by|artwork\s+by|"
                r"portrait\s+by|artist|painter)\b",
                prefix,
                flags=re.IGNORECASE,
            ))
            if visual_art_relations:
                return visual_art_relations[-1].group(0)
    relation_tokens = _surface_word_tokens(relation)
    if relation_tokens:
        pattern = r"\b" + r"\s+".join(
            re.escape(token) for token in relation_tokens
        ) + r"\b"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    if predicate_families & {"birth", "death"}:
        object_surface = _completion_source_surface(text, value)
        if object_surface:
            prefix = text[: text.lower().find(object_surface.lower())]
            locators = list(re.finditer(r"\b(?:in|at|on)\b", prefix, re.I))
            if locators:
                return locators[-1].group(0)
    return ""


def _certificate_source_surface(text: Any, surface: Any) -> str:
    """Locate an endpoint while preserving harmless source punctuation.

    OpenIE canonicalization removes commas and periods from names such as
    ``Robert A. Stemmle`` and ``Tabitha, Stand Up``.  A recovery certificate
    may bridge those separators, but it may not skip lexical tokens.
    """
    tokens = _surface_word_tokens(surface)
    if not tokens:
        return ""
    pattern = r"\b" + r"[\W_]+".join(
        re.escape(token) for token in tokens
    ) + r"\b"
    match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    return match.group(0) if match else ""


def _certificate_relation_is_negated(sentence: str, relation_surface: str) -> bool:
    """Reject negation local to the certified relation, not words in titles."""
    text = str(sentence or "")
    relation = str(relation_surface or "").strip()
    if not text or not relation:
        return True
    relation_at = text.casefold().find(relation.casefold())
    if relation_at < 0:
        return True
    prefix_tokens = _surface_word_tokens(text[:relation_at])[-3:]
    return bool(set(prefix_tokens) & {"neither", "never", "no", "not"})


def _schema_compiled_source_certificate(
    *,
    item: Mapping[str, Any],
    document: Mapping[str, Any],
    operator: Mapping[str, Any],
    doc_id: int,
    bound_surface: str,
) -> Dict[str, Any]:
    """Compile a typed source proof from the active schema obligation.

    The fact index stores canonical tuples whose relation or endpoint surface
    may not occur verbatim in the passage.  This proof therefore uses the
    *active* predicate family, direction and output type to recognize a small
    set of source constructions, while retaining exact document and sentence
    provenance.  It is deliberately fail-closed: no fuzzy document identity,
    untyped date, or relation-free endpoint co-occurrence is sufficient.
    """
    triple = [str(value or "").strip() for value in item.get("triple") or []]
    role_match = dict(item.get("role_direction_match") or {})
    if len(triple) != 3 or not all(triple) or not role_match:
        return {}

    input_side = str(role_match.get("input_side") or "")
    output_side = str(role_match.get("output_side") or "")
    if (input_side, output_side) not in {
        ("subject", "object"), ("object", "subject")
    }:
        return {}
    output_surface = str(role_match.get("surface") or "").strip()
    output_key = str(role_match.get("key") or "").strip()
    bound_key = str(item.get("completion_bound_input_key") or "").strip()
    if not output_surface or not output_key or not bound_key:
        return {}

    content = str(document.get("content") or "")
    title = str(document.get("title") or "").strip()
    if not title and content:
        title = content.splitlines()[0].strip()
    if not content or not title:
        return {}

    normalized_bound_surface = str(bound_surface or "").strip()
    title_binds_input = bool(
        normalized_bound_surface
        and _entity_key(title) == _entity_key(normalized_bound_surface)
    )
    predicate_families = set(operator.get("predicate_families") or [])
    output_type = str(operator.get("output_type") or "entity").strip().lower()

    role_phrases: List[str] = []
    role_family_phrases = {
        "authorship": ["written by", "authored by"],
        "composition": ["composed by", "music by", "score by"],
        "creation": ["created by"],
        "design": ["designed by", "architect was"],
        "direction": ["directed by"],
        "performance": ["performed by", "recorded by", "sung by"],
        "production": ["produced by"],
        "visual_art": ["painted by"],
    }
    for family, phrases in role_family_phrases.items():
        if family in predicate_families:
            role_phrases.extend(phrases)

    geographic_heads = (
        "airport", "city", "country", "district", "island", "mountain",
        "municipality", "province", "region", "river", "school", "station",
        "town", "university", "village",
    )
    date_pattern = re.compile(
        r"(?<!\w)(?:"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+[0-3]?\d,?\s+[1-9]\d{2,3}|"
        r"[0-3]?\d\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+[1-9]\d{2,3}|"
        r"[1-9]\d{2,3}"
        r")(?!\w)",
        flags=re.IGNORECASE,
    )

    for sentence_record in _split_completion_sentence_records(content):
        sentence = str(sentence_record.get("sentence") or "")
        explicit_bound = _certificate_source_surface(
            sentence, normalized_bound_surface
        )
        if not explicit_bound and not title_binds_input:
            continue
        source_bound = explicit_bound or title
        evidence_mode = ""
        source_relation = ""
        source_output = ""




        for phrase in role_phrases:
            phrase_match = re.search(rf"\b{re.escape(phrase)}\b", sentence, re.I)
            if phrase_match is None:
                continue
            suffix = sentence[phrase_match.end():]
            candidate = re.match(
                r"\s*(?P<value>[^,;.]{1,100}?)(?=\s+(?:and|who|whose)\b|[,;.]|$)",
                suffix,
                flags=re.IGNORECASE,
            )
            if candidate is None:
                continue
            candidate_surface = candidate.group("value").strip()
            if _entity_key(candidate_surface) != output_key:
                continue
            source_relation = phrase_match.group(0)
            source_output = candidate_surface
            evidence_mode = "directed_role_clause"
            break




        if (
            not evidence_mode
            and output_type == "date"
            and len(predicate_families & {"birth", "death"}) == 1
            and int(sentence_record.get("sentence_id", -1)) <= 2
        ):
            family = next(iter(predicate_families & {"birth", "death"}))
            for parenthetical in re.findall(r"\(([^)]{3,160})\)", sentence):
                dates = [
                    match.group(0).strip()
                    for match in date_pattern.finditer(parenthetical)
                ]
                if len(dates) < 2:
                    continue
                endpoint = dates[0] if family == "birth" else dates[-1]
                if _entity_key(endpoint) != output_key:
                    continue
                source_output = endpoint
                evidence_mode = f"typed_biographical_{family}_endpoint"
                break






        if (
            not evidence_mode
            and predicate_families & {"containment", "geopolitical_location"}
            and output_type == "place"
            and int(sentence_record.get("sentence_id", -1)) <= 2
        ):
            source_output = _certificate_source_surface(sentence, output_surface)
            if source_output:
                output_at = sentence.casefold().find(source_output.casefold())
                prefix = sentence[:output_at]
                explicit_location = list(re.finditer(
                    r"\b(?:located|situated|based)\b[^.;]{0,100}?"
                    r"\b(?:in|at|within|near)\b",
                    prefix,
                    flags=re.IGNORECASE,
                ))
                nominal_location = list(re.finditer(
                    r"\b(?:is|was|are|were)\b[^.;]{0,80}?\b(?:"
                    + "|".join(geographic_heads)
                    + r")\b[^.;]{0,80}?\b(?:in|at|within|near|of)\b",
                    prefix,
                    flags=re.IGNORECASE,
                ))
                relation_matches = explicit_location or nominal_location
                if relation_matches:
                    source_relation = relation_matches[-1].group(0)
                    evidence_mode = (
                        "explicit_location_clause"
                        if explicit_location else "typed_nominal_location_clause"
                    )

        if not evidence_mode:
            continue
        if source_relation and _certificate_relation_is_negated(
            sentence, source_relation
        ):
            continue
        return {
            "version": "schema_compiled_source_certificate_v2",
            "schema_evidence_mode": evidence_mode,
            "canonical_subject": triple[0],
            "canonical_subject_key": str(item.get("subject_key") or ""),
            "canonical_predicate": _canonical_completion_predicate(
                dict(operator), triple[1]
            ),
            "typed_object": {
                "surface": output_surface,
                "key": output_key,
                "type": output_type,
            },
            "bound_input_surface": normalized_bound_surface,
            "bound_input_key": bound_key,
            "bound_output_key": output_key,
            "relation_direction": f"{input_side}_to_{output_side}",
            "predicate_families": sorted(predicate_families),
            "source_document_id": int(doc_id),
            "source_title": title,
            "source_sentence_id": int(sentence_record.get("sentence_id", -1)),
            "source_sentence": sentence,
            "source_span_start": int(sentence_record.get("span_start", -1)),
            "source_span_end": int(sentence_record.get("span_end", -1)),
            "source_subject_surface": source_bound,
            "source_subject_scope": (
                "same_sentence" if explicit_bound else "exact_document_title"
            ),
            "source_relation_surface": source_relation,
            "source_object_surface": source_output,
            "obligation_id": str(operator.get("operator_id") or ""),
            "certified_tuple_key": str(item.get("tuple_key") or ""),
            "query_local_provenance": "completion_overlay",
            "candidate_support_status": "UNVERIFIED",
            "instance_identity_mode": (
                "explicit_bound_mention"
                if explicit_bound else "exact_title_bound_input"
            ),
        }
    return {}


def _indexed_obligation_repair_certificate(
    *,
    item: Mapping[str, Any],
    document: Mapping[str, Any],
    operator: Mapping[str, Any],
    doc_id: int,
    bound_surface: str = "",
    enable_schema_compiled_source_certificate: bool = False,
) -> Dict[str, Any]:
    """Build a strict source certificate for an indexed COMPLETE tuple.

    Predicate, role direction, and output type have already been checked by
    the physical semijoin.  This function additionally requires an exact
    source sentence containing the lexical relation and output endpoint.  The
    input endpoint must occur in that sentence or equal the document title.
    """
    triple = [str(value or "").strip() for value in item.get("triple") or []]
    if len(triple) != 3 or not all(triple):
        return {}
    content = str(document.get("content") or "")
    title = str(document.get("title") or "").strip()
    if not title and content:
        title = content.splitlines()[0].strip()
    if not content or not title:
        return {}
    title_binds_subject = _entity_key(title) == _entity_key(triple[0])
    predicate_families = set(operator.get("predicate_families") or [])
    for sentence_record in _split_completion_sentence_records(content):
        sentence = str(sentence_record.get("sentence") or "")
        source_subject = _certificate_source_surface(sentence, triple[0])
        source_object = _certificate_source_surface(sentence, triple[2])
        source_relation = _source_relation_surface(
            sentence,
            triple[1],
            triple[2],
            predicate_families,
        )
        if not source_object or not source_relation:
            continue
        if not source_subject and not title_binds_subject:
            continue
        lowered = sentence.casefold()
        relation_at = lowered.find(source_relation.casefold())
        object_at = lowered.find(source_object.casefold())
        subject_at = (
            lowered.find(source_subject.casefold()) if source_subject else -1
        )
        if relation_at < 0 or object_at < relation_at:
            continue
        if subject_at >= 0 and subject_at > relation_at:
            continue
        if _certificate_relation_is_negated(sentence, source_relation):
            continue
        output_type = str(operator.get("output_type") or "entity")
        return {
            "version": "obligation_bound_repair_certificate_v1",
            "canonical_subject": triple[0],
            "canonical_subject_key": str(item.get("subject_key") or ""),
            "canonical_predicate": _canonical_completion_predicate(
                dict(operator), triple[1]
            ),
            "typed_object": {
                "surface": triple[2],
                "key": str(item.get("object_key") or ""),
                "type": output_type,
            },
            "relation_direction": "subject_to_object",
            "source_document_id": int(doc_id),
            "source_title": title,
            "source_sentence_id": int(sentence_record.get("sentence_id", -1)),
            "source_sentence": sentence,
            "source_span_start": int(sentence_record.get("span_start", -1)),
            "source_span_end": int(sentence_record.get("span_end", -1)),
            "source_subject_surface": source_subject or title,
            "source_subject_scope": (
                "same_sentence" if source_subject else "exact_document_title"
            ),
            "source_relation_surface": source_relation,
            "source_object_surface": source_object,
            "obligation_id": str(operator.get("operator_id") or ""),
            "certified_tuple_key": str(item.get("tuple_key") or ""),
            "query_local_provenance": "completion_overlay",
            "candidate_support_status": "UNVERIFIED",
            "instance_identity_mode": (
                "explicit_subject_mention"
                if source_subject else "exact_title_subject"
            ),
        }
    if enable_schema_compiled_source_certificate:
        return _schema_compiled_source_certificate(
            item=item,
            document=document,
            operator=operator,
            doc_id=doc_id,
            bound_surface=bound_surface,
        )
    return {}


def _extract_indexed_bound_completion_items(
    *,
    bound_surface: str,
    bound_key: str,
    operator: Dict[str, Any],
    documents: List[Dict[str, Any]],
    relation_index: Dict[str, Any],
    diagnostics: Counter,
    allow_title_provenance_alias: bool = False,
    enable_obligation_bound_repair_certificate: bool = False,
    enable_schema_compiled_source_certificate: bool = False,
) -> List[Dict[str, Any]]:
    """Semijoin retrieved document ids with typed OpenIE fact rows.

    Dense retrieval identifies candidate documents.  The fact store then
    performs the physical selection: the row must bind the requested entity,
    satisfy the canonical predicate family, and produce the operator's output
    type.  This avoids reparsing passages with a small set of surface regexes.
    """
    items = list((relation_index or {}).get("items") or [])
    doc_to_items = (relation_index or {}).get("doc_to_items") or {}
    if not items or not doc_to_items:
        diagnostics["indexed_fact_store_unavailable"] += 1
        return []

    equivalent_keys = _equivalent_entity_keys(
        relation_index,
        bound_key,
        allow_title_provenance_alias=bool(allow_title_provenance_alias),
    )
    equivalent_keys.add(str(bound_key or ""))
    output_type = str(operator.get("output_type") or "entity")
    role_direction_required = _operator_requires_role_direction(operator)
    extracted: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, str]] = set()
    for document in documents or []:
        diagnostics["indexed_documents_seen"] += 1
        try:
            doc_id = int(document.get("doc_id", -1))
        except (TypeError, ValueError):
            diagnostics["indexed_invalid_doc_id"] += 1
            continue
        item_ids = list(doc_to_items.get(doc_id) or [])
        if not item_ids:
            diagnostics["indexed_doc_miss"] += 1
            continue
        diagnostics["indexed_doc_hit"] += 1
        for item_id in item_ids:
            try:
                item = items[int(item_id)]
            except (IndexError, TypeError, ValueError):
                diagnostics["indexed_invalid_item_id"] += 1
                continue
            diagnostics["indexed_fact_rows"] += 1
            output = _other_endpoint_for_exact_binding(
                item,
                bound_key,
                equivalent_keys=equivalent_keys,
            )
            if output is None:
                diagnostics["indexed_bound_mismatch"] += 1
                continue
            diagnostics["indexed_bound_match"] += 1
            if not _plan_operator_predicate_matches_item(item, operator):
                diagnostics["indexed_predicate_mismatch"] += 1
                continue
            diagnostics["indexed_predicate_match"] += 1
            oriented_output = _oriented_endpoint_for_operator_binding(
                item,
                operator,
                bound_key,
                equivalent_keys=equivalent_keys,
            )
            if oriented_output is None:
                diagnostics["indexed_role_direction_mismatch"] += 1
                continue
            diagnostics["indexed_role_direction_match"] += 1
            output_surface = str(oriented_output.get("surface") or "")
            relation_tokens = set(item.get("relation_tokens") or set())
            output_contract = terminal_value_contract_compatibility(
                output_surface,
                output_type,
                relation_tokens,
                target_category_tokens=(
                    operator.get("target_entity_categories") or []
                ),
                output_role=str(oriented_output.get("output_side") or ""),
                source_text=str(item.get("source_sentence") or ""),
                relation_families=sorted(
                    set(item.get("relation_families") or [])
                    | set(item.get("predicate_families") or [])
                    | set(operator.get("relation_families") or [])
                    | set(operator.get("predicate_families") or [])
                ),
            )
            if output_contract["status"] == TERMINAL_TYPE_INCOMPATIBLE:
                diagnostics["indexed_output_type_mismatch"] += 1
                continue
            diagnostics["indexed_output_type_match"] += 1
            key = (doc_id, str(item.get("tuple_key") or item_id))
            if key in seen:
                diagnostics["indexed_duplicate_fact"] += 1
                continue
            seen.add(key)
            completion_item = dict(item)
            tuple_key = str(completion_item.get("tuple_key") or "")
            completion_item.update({
                "access_path": "bound_document_fact_semijoin",
                "bound_completion_fact": True,
                "completion_fact_id": ":".join([
                    "completion-indexed",
                    str(doc_id),
                    tuple_key,
                ]),
                "completion_bound_input_key": str(bound_key),
                "completion_source_doc_id": int(doc_id),
                "completion_source_sentence_id": -1,
                "completion_extraction_direction": "indexed",
                "completion_instance_identity_mode": (
                    "indexed_exact_entity_semijoin"
                ),
                "completion_instance_identity_validated": True,
                "completion_retrieval_rank": int(
                    document.get("rank", len(documents)) or 0
                ),
                "completion_retrieval_score": float(
                    document.get("score", 0.0) or 0.0
                ),
                "completion_typed_access_key": {
                    "doc_id": doc_id,
                    "bound_key": str(bound_key),
                    "predicate_families": sorted(
                        operator.get("predicate_families") or []
                    ),
                    "output_type": output_type,
                    "canonical_relation": str(
                        operator.get("canonical_relation") or ""
                    ),
                    "input_role": str(operator.get("input_role") or ""),
                    "output_role": str(operator.get("output_role") or ""),
                },
                "role_direction_match": oriented_output,
                "completion_sentence": " | ".join(
                    str(value) for value in item.get("triple") or []
                )[:320],
            })
            if (
                enable_obligation_bound_repair_certificate
                or enable_schema_compiled_source_certificate
            ):
                certificate = _indexed_obligation_repair_certificate(
                    item=completion_item,
                    document=document,
                    operator=operator,
                    doc_id=doc_id,
                    bound_surface=bound_surface,
                    enable_schema_compiled_source_certificate=bool(
                        enable_schema_compiled_source_certificate
                    ),
                )
                if certificate:
                    completion_item["source_grounded_recovery_record"] = certificate
                    completion_item["obligation_bound_repair_certificate"] = True
                    diagnostics["obligation_bound_repair_certificate"] += 1
                else:
                    diagnostics["obligation_bound_repair_certificate_miss"] += 1
            extracted.append(completion_item)
            diagnostics["indexed_accepted_fact"] += 1
    return extracted


def _extract_bound_completion_items(
    *,
    bound_surface: str,
    bound_key: str,
    operator: Dict[str, Any],
    documents: List[Dict[str, Any]],
    item_id_seed: int,
    diagnostics: Optional[Counter] = None,
    relation_index: Optional[Dict[str, Any]] = None,
    enable_predicate_gap_repair: bool = True,
    enable_inverse_relation_completion: bool = True,
    enable_title_grounded_completion: bool = False,
    enable_source_grounded_terminal_repair: bool = False,
    enable_source_grounded_terminal_tuple_recovery: bool = False,
    enable_obligation_bound_repair_certificate: bool = False,
    enable_schema_compiled_source_certificate: bool = False,
    continue_sentence_repair_after_indexed_candidates: bool = False,
) -> List[Dict[str, Any]]:
    """Extract explicit operator facts from predicate-bound passages.

    Completion passages use several equivalent surface orders.  Normalize
    those orders into the operator's canonical ``bound -> value`` direction
    before matching the physical predicate contract.  This keeps extraction
    strict without assuming that the bound entity must occur before the
    predicate in the sentence.
    """
    diagnostics = diagnostics if diagnostics is not None else Counter()
    enable_source_grounded_terminal_tuple_recovery = bool(
        enable_source_grounded_terminal_tuple_recovery
        and str(operator.get("slot_plan_stage") or "") == "final_answer"
    )
    predicate_gap_repair = False
    bound_tokens = _surface_word_tokens(bound_surface)
    if not bound_tokens:
        diagnostics["missing_bound_tokens"] += 1
        return []
    bound_pattern = r"\b" + r"[\s'\u2019\-]+".join(
        re.escape(token) for token in bound_tokens
    ) + r"\b"
    bound_identity_keys = {str(bound_key or "")}
    if relation_index:
        bound_identity_keys.update(
            _equivalent_entity_keys(
                relation_index,
                bound_key,
                allow_title_provenance_alias=bool(
                    enable_title_grounded_completion
                ),
            )
        )
    document_identity_decisions: List[Tuple[str, str]] = []
    document_identity_titles: List[str] = []
    for document in documents or []:
        content = str(document.get("content") or "")
        title = str(document.get("title") or "").strip()
        if not title:
            title = content.splitlines()[0].strip() if content else ""
        document_identity_titles.append(title)
        identity_decision = _completion_title_identity_decision(
            bound_surface=bound_surface,
            bound_identity_keys=bound_identity_keys,
            title=title,
            enabled=bool(enable_title_grounded_completion),
        )
        if (
            enable_source_grounded_terminal_tuple_recovery
            and not identity_decision[0]
            and re.search(bound_pattern, content, flags=re.IGNORECASE)
        ):
            identity_decision = ("explicit_bound_source", "")
            diagnostics["source_grounded_explicit_bound_document"] += 1
        document_identity_decisions.append(identity_decision)

    strong_identity_present = any(
        mode == "title_or_alias_scoped"
        for mode, _ in document_identity_decisions
    )
    fuzzy_title_keys = {
        _entity_key(title)
        for title, (mode, _) in zip(
            document_identity_titles, document_identity_decisions
        )
        if mode == "title_grounded_name_containment" and _entity_key(title)
    }
    normalized_identity_decisions: List[Tuple[str, str]] = []
    for mode, reason in document_identity_decisions:
        if mode == "title_grounded_name_containment" and strong_identity_present:
            normalized_identity_decisions.append(
                ("", "stronger_identity_candidate_present")
            )
        elif (
            mode == "title_grounded_name_containment"
            and len(fuzzy_title_keys) > 1
        ):
            normalized_identity_decisions.append(("", "ambiguous_title_match"))
        else:
            normalized_identity_decisions.append((mode, reason))
    document_identity_decisions = normalized_identity_decisions
    for _, reason in document_identity_decisions:
        diagnostic_key = {
            "roman_conflict": "title_identity_roman_conflict_rejected",
            "year_conflict": "title_identity_year_conflict_rejected",
            "stronger_identity_candidate_present": (
                "title_identity_stronger_candidate_rejected"
            ),
            "ambiguous_title_match": "title_identity_ambiguous_rejected",
        }.get(reason)
        if diagnostic_key:
            diagnostics[diagnostic_key] += 1
    title_grounded_fallback = any(
        bool(mode) for mode, _ in document_identity_decisions
    )
    if relation_index:
        before = Counter(diagnostics)
        indexed_items = _extract_indexed_bound_completion_items(
            bound_surface=bound_surface,
            bound_key=bound_key,
            operator=operator,
            documents=documents,
            relation_index=relation_index,
            diagnostics=diagnostics,
            allow_title_provenance_alias=bool(enable_title_grounded_completion),
            enable_obligation_bound_repair_certificate=bool(
                enable_obligation_bound_repair_certificate
            ),
            enable_schema_compiled_source_certificate=bool(
                enable_schema_compiled_source_certificate
            ),
        )
        if indexed_items and not continue_sentence_repair_after_indexed_candidates:
            return indexed_items
        fact_rows = int(diagnostics.get("indexed_fact_rows", 0)) - int(
            before.get("indexed_fact_rows", 0)
        )
        bound_matches = int(diagnostics.get("indexed_bound_match", 0)) - int(
            before.get("indexed_bound_match", 0)
        )
        predicate_matches = int(
            diagnostics.get("indexed_predicate_match", 0)
        ) - int(before.get("indexed_predicate_match", 0))
        predicate_gap_repair = bool(
            enable_predicate_gap_repair
            and (
                (
                    fact_rows > 0
                    and bound_matches > 0
                    and predicate_matches == 0
                )
                or (
                    continue_sentence_repair_after_indexed_candidates
                    and bool(indexed_items)
                )
            )
        )
        if predicate_gap_repair:
            diagnostics["predicate_gap_sentence_repair_triggered"] += 1
        elif fact_rows > 0 and not title_grounded_fallback:
            return indexed_items



        if not predicate_gap_repair and not title_grounded_fallback:
            diagnostics["indexed_absent_regex_fallback"] += 1
        elif title_grounded_fallback:
            diagnostics["title_grounded_regex_fallback"] += 1
    predicate_families = set(operator.get("predicate_families") or [])
    relation_tokens = set(operator.get("relation_tokens") or [])
    output_type = str(operator.get("output_type") or "entity")
    role_direction_required = _operator_requires_role_direction(operator)
    role_contract = _canonical_operator_role_contract(relation_tokens)
    extracted: List[Dict[str, Any]] = list(
        indexed_items
        if relation_index and continue_sentence_repair_after_indexed_candidates
        else []
    )
    seen: Set[Tuple[int, str, str]] = {
        (
            int(item.get("doc_idx", -1)),
            str(item.get("relation_key") or ""),
            str(item.get("object_key") or ""),
        )
        for item in extracted
    }
    if extracted:
        diagnostics["indexed_candidates_retained_for_sentence_repair"] += len(
            extracted
        )
    documents_by_id: Dict[int, Dict[str, Any]] = {}
    for raw_document in documents or []:
        try:
            raw_doc_id = int(raw_document.get("doc_id", -1))
        except (TypeError, ValueError):
            continue
        if raw_doc_id >= 0:
            documents_by_id[raw_doc_id] = dict(raw_document)

    def emit(
        doc_id: int,
        sentence_id: int,
        sentence: str,
        relation: str,
        value: str,
        *,
        inverse_match: bool,
        identity_mode: str,
        source_title: str,
        source_span_start: int = -1,
        source_span_end: int = -1,
        source_relation_surface: str = "",
    ) -> bool:
        value = _clean_completion_value(value)
        source_grounded_relation_contract_eligible = not (
            enable_source_grounded_terminal_tuple_recovery
            and str(relation or "").strip().casefold() == "native city"
        )
        if not source_grounded_relation_contract_eligible:





            diagnostics["source_grounded_relation_contract_rejected"] += 1
        if output_type == "place":
            value = re.split(
                r"\s+(?:in|on)\s+(?=(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b)",
                value,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            value = re.split(
                r",\s+(?:(?:he|she|they|the\s+subject|[A-Z][\w'\-]+)\s+(?:is|was)\b)",
                value,
                maxsplit=1,
            )[0].strip()




            value = re.split(
                r",\s+(?:(?:he|she|they|the\s+subject)|[A-Z][\w'\-]+)\s+"
                r"(?:grew|moved|lived|worked|served|studied|attended|began|"
                r"became|died|married|joined|returned|settled)\b",
                value,
                maxsplit=1,
            )[0].strip()
            value = re.split(
                r"\s+in\s+(?=(?:(?:[0-3]?\d)\s+)?(?:January|February|March|"
                r"April|May|June|July|August|September|October|November|"
                r"December)\s+(?:[1-9]\d{2,3})\b)",
                value,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            value = re.split(
                r",\s+as\s+(?:the|an?|one)\b",
                value,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
        if not value:
            diagnostics["empty_value"] += 1
            return False
        canonical_relation = str(
            operator.get("canonical_relation")
            or role_contract.get("canonical_relation")
            or ""
        ).strip().casefold()
        if source_local_kinship_role_conflict(
            value, sentence, canonical_relation
        ):
            diagnostics["source_local_kinship_role_conflict"] += 1
            return False
        raw_completion_value = value
        endpoint_decision = _canonicalize_intermediate_completion_endpoint(
            value=value,
            sentence=sentence,
            operator=operator,
            relation_index=relation_index,
        )
        if endpoint_decision["applied"]:
            value = str(endpoint_decision["surface"])
            diagnostics["completion_endpoint_canonicalized"] += 1
            diagnostics[
                "completion_endpoint_canonicalized_"
                + str(endpoint_decision["reason"])
            ] += 1
        canonical_bound_surface = str(bound_surface or "").strip()
        exact_bound_surface = _completion_source_surface(sentence, bound_surface)
        if (
            enable_source_grounded_terminal_tuple_recovery
            and source_grounded_relation_contract_eligible
        ):
            if exact_bound_surface:
                canonical_bound_surface = exact_bound_surface
            elif identity_mode.startswith("title_") and source_title:
                canonical_bound_surface = str(source_title).strip()
            canonical_bound_surface = canonical_bound_surface.strip(
                " \t\n\r,;:-."
            )
        fact_subject = canonical_bound_surface
        fact_object = value
        fact_relation = relation
        if role_direction_required:



            fact_subject = value
            fact_object = canonical_bound_surface
            fact_relation = str(
                operator.get("canonical_relation")
                or role_contract.get("canonical_relation")
                or relation
            ).replace("_", " ")
        item = _completion_fact_item(
            subject=fact_subject,
            relation=fact_relation,
            obj=fact_object,
            doc_id=doc_id,
            item_id=item_id_seed - len(extracted) - 1,
            sentence=sentence,
            source_sentence_id=sentence_id,
            bound_input_key=bound_key,
            extraction_direction=("inverse" if inverse_match else "forward"),
            instance_identity_mode=identity_mode,
            instance_identity_validated=True,
        )
        if item is None:
            diagnostics["invalid_fact"] += 1
            return False




        if role_direction_required:
            item["object_key"] = str(bound_key)
        else:
            item["subject_key"] = str(bound_key)
        item["entity_keys"] = {
            str(item.get("subject_key") or ""),
            str(item.get("object_key") or ""),
        }
        item["tuple_key"] = "|".join([
            str(item.get("subject_key") or ""),
            str(item.get("relation_key") or ""),
            str(item.get("object_key") or ""),
        ])
        item["completion_fact_id"] = ":".join([
            "completion",
            str(int(doc_id)),
            str(int(sentence_id)),
            str(item["tuple_key"]),
        ])
        oriented_output = _oriented_endpoint_for_operator_binding(
            item,
            operator,
            bound_key,
            equivalent_keys={bound_key},
        )
        if oriented_output is None:
            diagnostics["predicate_contract_mismatch"] += 1
            return False
        if str(oriented_output.get("key") or "") != _entity_key(value):
            diagnostics["oriented_output_mismatch"] += 1
            return False
        item["role_direction_match"] = oriented_output
        item["completion_raw_output_surface"] = raw_completion_value
        item["completion_canonical_output_surface"] = value
        item["completion_endpoint_canonicalization_applied"] = bool(
            endpoint_decision["applied"]
        )
        item["completion_endpoint_canonicalization_reason"] = str(
            endpoint_decision["reason"]
        )
        item["completion_endpoint_canonicalization_candidate_count"] = int(
            endpoint_decision["candidate_count"]
        )
        item["predicate_subtypes"] = set(
            canonical_schema_predicate_subtypes(
                item.get("relation"),
                oriented_output.get("surface"),
            )
        )
        item["completion_source_title"] = str(source_title or "")
        item["completion_source_grounded"] = bool(
            identity_mode and int(doc_id) >= 0 and int(sentence_id) >= 0
        )





        if (
            enable_obligation_bound_repair_certificate
            and item["completion_source_grounded"]
            and int(doc_id) in documents_by_id
        ):
            certificate = _indexed_obligation_repair_certificate(
                item=item,
                document=documents_by_id[int(doc_id)],
                operator=operator,
                doc_id=int(doc_id),
                bound_surface=bound_surface,
                enable_schema_compiled_source_certificate=False,
            )
            if certificate:
                item["source_grounded_recovery_record"] = certificate
                item["obligation_bound_repair_certificate"] = True
                diagnostics["obligation_bound_repair_certificate"] += 1
            else:
                diagnostics["obligation_bound_repair_certificate_miss"] += 1
        item["completion_contextual_qualifier_risk"] = bool(
            output_type == "place"
            and re.search(
                r"\b(?:which|that)\b.{0,100}\b(?:part\s+of|within|"
                r"located\s+in|incorporated\s+into|merged\s+into)\b",
                str(sentence or ""),
                flags=re.IGNORECASE,
            )
        )
        if item["completion_contextual_qualifier_risk"]:
            diagnostics["contextual_qualifier_risk_fact"] += 1
        item["completion_terminal_source_repair"] = bool(
            (
                enable_source_grounded_terminal_repair
                or enable_source_grounded_terminal_tuple_recovery
            )
            and str(operator.get("slot_plan_stage") or "") == "final_answer"
        )
        if (
            enable_source_grounded_terminal_tuple_recovery
            and source_grounded_relation_contract_eligible
        ):
            lexical_relation = str(
                source_relation_surface
                or _source_relation_surface(
                    sentence,
                    relation,
                    value,
                    predicate_families,
                )
            ).strip()
            source_subject = (
                _completion_source_surface(sentence, canonical_bound_surface)
                or canonical_bound_surface
            )
            source_object = _completion_source_surface(sentence, value) or value
            canonical_predicate = _canonical_completion_predicate(
                operator, relation
            )
            item["source_grounded_recovery_record"] = {
                "version": "source_grounded_terminal_tuple_recovery_v1",
                "canonical_subject": canonical_bound_surface,
                "canonical_subject_key": str(bound_key or ""),
                "canonical_predicate": canonical_predicate,
                "typed_object": {
                    "surface": value,
                    "key": _entity_key(value),
                    "type": output_type,
                },
                "relation_direction": (
                    "object_to_subject" if role_direction_required else "subject_to_object"
                ),
                "source_document_id": int(doc_id),
                "source_title": str(source_title or ""),
                "source_sentence_id": int(sentence_id),
                "source_sentence": str(sentence or ""),
                "source_span_start": int(source_span_start),
                "source_span_end": int(source_span_end),
                "source_subject_surface": source_subject,
                "source_relation_surface": lexical_relation,
                "source_object_surface": source_object,
                "obligation_id": str(operator.get("operator_id") or ""),
                "query_local_provenance": "completion_overlay",
                "candidate_support_status": "UNVERIFIED",
                "instance_identity_mode": identity_mode,
            }
            item["source_grounded_terminal_tuple_recovery"] = True
            diagnostics["source_grounded_recovery_record"] += 1
        if predicate_gap_repair:
            item["access_path"] = "predicate_gap_sentence_repair"
            item["predicate_gap_sentence_repair"] = True
        key = (
            int(doc_id),
            str(item.get("relation_key") or ""),
            str(item.get("object_key") or ""),
        )
        if key in seen:
            diagnostics["duplicate_fact"] += 1
            return False
        seen.add(key)
        extracted.append(item)
        diagnostics["accepted_fact"] += 1
        if predicate_gap_repair:
            diagnostics["predicate_gap_sentence_repair_fact"] += 1
        if item["completion_source_grounded"]:
            diagnostics["source_grounded_fact"] += 1
        if item["completion_terminal_source_repair"]:
            diagnostics["source_grounded_terminal_repair_fact"] += 1
        return True



    value_tail = (
        r"(?P<value>[^.;]{1,180}?)"
        r"(?=,\s+(?:(?:he|she|they|the\s+subject|[A-Z][\w'\-]+)\s+"
        r"(?:is|was|became|served)\b)|"
        r"\s+(?:in|on)\s+(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b|[.;]|$)"
    )
    reverse_value = r"(?P<value>[^.;]{1,100}?)"
    date_value = (
        r"(?P<value>(?:(?:[0-3]?\d)\s+)?"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+(?:[1-9]\d{2,3})|"
        r"(?:[1-9]\d{2,3}))"
    )
    date_qualifier = (
        r"(?:(?:[0-3]?\d)\s+)?"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+(?:[1-9]\d{2,3})|"
        r"(?:[1-9]\d{2,3})"
    )
    date_surface_pattern = re.compile(
        r"(?<!\w)(?:"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+[0-3]?\d,?\s+[1-9]\d{2,3}|"
        r"[0-3]?\d\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+[1-9]\d{2,3}|"
        r"[1-9]\d{2,3}"
        r")(?!\w)",
        flags=re.IGNORECASE,
    )
    nationality_values = sorted(
        NATIONALITY_TERMS
        - {"citizen", "citizenship", "country", "national", "nationality"},
        key=lambda value: (-len(value), value),
    )
    nationality_value = (
        r"(?P<value>" + "|".join(re.escape(value) for value in nationality_values) + r")"
    )
    parent_value = (
        r"(?P<value>[^,;.]{1,90}?)"
        r"(?=\s+(?:and|by)\s+|[,;.]|$)"
    )

    for document_index, document in enumerate(documents or []):
        diagnostics["documents_seen"] += 1
        try:
            doc_id = int(document.get("doc_id", -1))
        except (TypeError, ValueError):
            continue
        content = str(document.get("content") or "")
        if doc_id < 0 or not content:
            continue
        title = str(document.get("title") or "").strip()
        if not title:
            title = content.splitlines()[0].strip() if content else ""
        title_key = _entity_key(title)
        title_identity_mode = document_identity_decisions[document_index][0]
        title_scoped_document = bool(
            title_identity_mode
            and title_identity_mode != "explicit_bound_source"
        )
        if title_scoped_document:
            diagnostics["title_scoped_documents"] += 1
            if title_identity_mode == "title_grounded_name_containment":
                diagnostics["title_grounded_documents"] += 1
        elif title_identity_mode == "explicit_bound_source":
            diagnostics["explicit_bound_source_documents"] += 1
        sentence_records = _split_completion_sentence_records(content)
        for sentence_record in sentence_records:
            sentence_id = int(sentence_record["sentence_id"])
            sentence = str(sentence_record["sentence"])
            source_span_start = int(sentence_record["span_start"])
            source_span_end = int(sentence_record["span_end"])
            diagnostics["sentences_seen"] += 1
            explicit_bound = bool(
                re.search(bound_pattern, sentence, flags=re.IGNORECASE)
            )
            title_scoped_sentence = bool(
                title_scoped_document
                and _entity_key(sentence) != title_key
            )
            if not explicit_bound and not title_scoped_sentence:
                continue
            if explicit_bound:
                diagnostics["bound_sentences"] += 1
            else:
                diagnostics["title_scoped_sentences"] += 1
            patterns: List[Tuple[str, str]] = []
            inverse_patterns: Set[str] = set()
            if (
                (title_scoped_sentence or explicit_bound)
                and output_type == "date"
                and predicate_families & {"birth", "death"}
                and sentence_id <= 2
            ):




                family = "birth" if "birth" in predicate_families else "death"
                for parenthetical in re.findall(r"\(([^)]{3,140})\)", sentence):
                    date_surfaces = [
                        match.group(0).strip()
                        for match in date_surface_pattern.finditer(parenthetical)
                    ]
                    explicit_family = bool(re.search(
                        r"\b(?:born|birth)\b" if family == "birth"
                        else r"\b(?:died|death|passed\s+away)\b",
                        parenthetical,
                        flags=re.IGNORECASE,
                    ))
                    if not date_surfaces or (
                        len(date_surfaces) < 2 and not explicit_family
                    ):
                        continue
                    selected_date = (
                        date_surfaces[0]
                        if family == "birth" else date_surfaces[-1]
                    )
                    if emit(
                        doc_id,
                        sentence_id,
                        sentence,
                        f"{family} date",
                        selected_date,
                        inverse_match=False,
                        identity_mode=title_identity_mode,
                        source_title=title,
                        source_span_start=source_span_start,
                        source_span_end=source_span_end,
                    ):
                        diagnostics["title_biographical_date_fact"] += 1
                    break
            if (
                enable_source_grounded_terminal_tuple_recovery
                and (title_scoped_sentence or explicit_bound)
                and output_type == "place"
                and predicate_families & {"birth", "death"}
                and sentence_id <= 2
            ):
                family = "birth" if "birth" in predicate_families else "death"
                for parenthetical in re.findall(r"\(([^)]{8,180})\)", sentence):
                    endpoints = re.split(
                        r"\s+[\u2013\u2014-]\s+", parenthetical, maxsplit=1
                    )
                    if len(endpoints) != 2:
                        continue
                    if not all(date_surface_pattern.search(value) for value in endpoints):
                        continue
                    endpoint = endpoints[0] if family == "birth" else endpoints[1]
                    place_match = re.search(
                        r"\b(?:in|at)\s+(?P<value>[^,;.()]{1,100}?)(?=\s*$)",
                        endpoint,
                        flags=re.IGNORECASE,
                    )
                    if place_match is None:
                        continue
                    if emit(
                        doc_id,
                        sentence_id,
                        sentence,
                        f"{family} place",
                        place_match.group("value"),
                        inverse_match=False,
                        identity_mode=(
                            "explicit_bound_mention"
                            if explicit_bound else title_identity_mode
                        ),
                        source_title=title,
                        source_span_start=source_span_start,
                        source_span_end=source_span_end,
                        source_relation_surface="in",
                    ):
                        diagnostics[f"source_grounded_lifespan_{family}_place_fact"] += 1
                    break
            if "birth" in predicate_families:
                if output_type == "date":
                    patterns.extend([
                        (
                            "birth date",
                            rf"{bound_pattern}.{{0,36}}?\b(?:was\s+)?born\s+(?:on\s+)?{date_value}",
                        ),
                        (
                            "birth date",
                            rf"{bound_pattern}\s*\(\s*born\s+{date_value}",
                        ),
                    ])
                    if enable_title_grounded_completion:
                        patterns.append((
                            "birth date",
                            rf"{bound_pattern}(?:\s*\([^)]{{1,40}}\))?\s*\(\s*{date_value}\s*(?:[\u2013\u2014-]|to\b)",
                        ))
                else:
                    patterns.extend([
                        (
                            "born in",
                            rf"{bound_pattern}.{{0,48}}?\b(?:was\s+)?born(?:\s+on\s+[^,;.]+)?\s+(?:in|at)\s+{value_tail}",
                        ),
                        (
                            "born in",
                            rf"{bound_pattern}(?:['\u2019]s)?\s+(?:birthplace|place\s+of\s+birth)\s+(?:is|was)\s+{value_tail}",
                        ),
                        (
                            "born in",
                            rf"(?:the\s+)?(?:birthplace|place\s+of\s+birth)\s+of\s+{bound_pattern}\s+(?:is|was)\s+{value_tail}",
                        ),
                        (
                            "born in",
                            rf"{reverse_value}\s+(?:is|was)\s+(?:the\s+)?(?:birthplace|place\s+of\s+birth)\s+of\s+{bound_pattern}",
                        ),
                    ])
                if title_scoped_sentence:
                    if output_type == "date":
                        patterns.extend([
                            (
                                "birth date",
                                rf"\b(?:he|she|they|the\s+subject)\s+(?:was\s+)?born\s+(?:on\s+)?{date_value}",
                            ),
                            (
                                "birth date",
                                rf"\bborn\s+(?:on\s+)?{date_value}",
                            ),
                        ])
                    else:
                        patterns.extend([
                            (
                                "born in",
                                rf"\b(?:he|she|they|the\s+subject)\s+(?:was\s+)?born(?:\s+on\s+[^,;.]+)?\s+(?:in|at)\s+{value_tail}",
                            ),
                            (
                                "born in",
                                rf"\b(?:his|her|their)\s+(?:birthplace|place\s+of\s+birth)\s+(?:is|was)\s+{value_tail}",
                            ),
                        ])
                        if enable_title_grounded_completion:
                            patterns.append((
                                "born in",
                                rf"\bBorn\s+(?:in|at)\s+{value_tail}",
                            ))
                        if enable_source_grounded_terminal_repair:
                            patterns.extend([
                                (
                                    "born in",
                                    rf"^(?:{SURFACE_PROPER_SEGMENT}|he|she|they|the\s+subject)"
                                    rf"(?:(?!\b(?:who|whose|which|whom)\b)[^.;]){{0,140}}?"
                                    rf"\bborn\s+(?:(?:on\s+)?(?:{date_qualifier})\s+)?"
                                    rf"(?:in|at)\s+{value_tail}",
                                ),
                                (
                                    "native city",
                                    rf"\b(?:his|her|their)\s+native\s+"
                                    rf"(?:city|town|place)(?:\s+of)?\s+{value_tail}",
                                ),
                            ])
                        if enable_source_grounded_terminal_tuple_recovery:
                            patterns.append((
                                "born in",
                                r"^\s*Born\s+into\b[^.;]{0,140}?\bin\s+"
                                r"(?P<value>[^,;.]{1,100}?)(?=\s*[,;.]|$)",
                            ))
            if "death" in predicate_families:
                if output_type == "date":
                    patterns.append((
                        "death date",
                        rf"{bound_pattern}.{{0,36}}?\b(?:died|passed\s+away)\s+(?:on\s+)?{date_value}",
                    ))
                else:
                    patterns.extend([
                        (
                            "died in",
                            rf"{bound_pattern}.{{0,36}}?\b(?:died|passed\s+away)\s+(?:in|at)\s+{value_tail}",
                        ),
                        (
                            "died in",
                            rf"{bound_pattern}(?:['\u2019]s)?\s+(?:place\s+of\s+death|deathplace)\s+(?:is|was)\s+{value_tail}",
                        ),
                        (
                            "died in",
                            rf"{reverse_value}\s+(?:is|was)\s+(?:the\s+)?(?:place\s+of\s+death|place\s+where\s+{bound_pattern}\s+died)",
                        ),
                    ])
                if title_scoped_sentence and output_type == "date":
                    patterns.extend([
                        (
                            "death date",
                            rf"\b(?:he|she|they|the\s+subject)\s+(?:died|passed\s+away)\s+(?:on\s+)?{date_value}",
                        ),
                        (
                            "death date",
                            rf"\b(?:died|passed\s+away)\s+(?:on\s+)?{date_value}",
                        ),
                    ])
                if title_scoped_sentence:
                    if output_type == "date":
                        patterns.append((
                            "death date",
                            rf"\b(?:he|she|they|the\s+subject)\s+(?:died|passed\s+away)\s+(?:on\s+)?{date_value}",
                        ))
                    else:
                        patterns.extend([
                            (
                                "died in",
                                rf"\b(?:he|she|they|the\s+subject)\s+(?:died|passed\s+away)\s+(?:in|at)\s+{value_tail}",
                            ),
                            (
                                "died in",
                                rf"\b(?:his|her|their)\s+(?:place\s+of\s+death|deathplace)\s+(?:is|was)\s+{value_tail}",
                            ),
                        ])
                        if enable_source_grounded_terminal_repair:
                            patterns.append((
                                "died in",
                                rf"\b(?:he|she|they|the\s+subject)\b"
                                rf"(?:(?!\b(?:who|whose|which|whom)\b)[^.;]){{0,140}}?"
                                rf"\b(?:died|passed\s+away)\s+(?:in|at)\s+{value_tail}",
                            ))
            if "nationality" in predicate_families:
                patterns.extend([
                    (
                        "nationality",
                        rf"{bound_pattern}.{{0,80}}?\b(?:is|was)\s+(?:an?\s+)?"
                        rf"{nationality_value}\b",
                    ),
                    (
                        "nationality",
                        rf"{bound_pattern}.{{0,80}}?\b(?:nationality|citizenship)\s+"
                        rf"(?:is|was)\s+{nationality_value}\b",
                    ),
                ])
                if title_scoped_sentence and enable_source_grounded_terminal_repair:
                    patterns.append((
                        "nationality",
                        rf"^(?:{SURFACE_PROPER_SEGMENT}|he|she|they|the\s+subject)"
                        rf"(?:(?!\b(?:who|whose|which|whom)\b)[^.;]){{0,100}}?"
                        rf"\b(?:is|was)\s+(?:an?\s+)?{nationality_value}\b",
                    ))
            role_verbs = {
                "authorship": "written by",
                "composition": "composed by",
                "creation": "created by",
                "direction": "directed by",
                "production": "produced by",
                "visual_art": "painted by",
                "design": "designed by",
            }
            for family, relation in role_verbs.items():
                if family not in predicate_families:
                    continue
                verb = relation.split()[0]
                role_noun = {
                    "authorship": "author|writer",
                    "composition": "composer",
                    "creation": "creator",
                    "direction": "director",
                    "production": "producer",
                    "visual_art": "artist|painter",
                    "design": "architect|designer",
                }[family]
                patterns.extend([
                    (
                        relation,
                        rf"{bound_pattern}.{{0,36}}?\b(?:was\s+)?{re.escape(verb)}\s+by\s+{value_tail}",
                    ),
                    (
                        relation,
                        rf"(?:the\s+)?(?:{role_noun})\s+of\s+{bound_pattern}\s+(?:is|was)\s+{value_tail}",
                    ),
                    (
                        relation,
                        rf"{bound_pattern}(?:['\u2019]s)?\s+(?:{role_noun})\s+(?:is|was)\s+{value_tail}",
                    ),
                    (
                        relation,
                        rf"{reverse_value}\s+{re.escape(verb)}\s+{bound_pattern}",
                    ),
                ])
                if title_scoped_sentence:
                    patterns.extend([
                        (
                            relation,
                            rf"\b(?:he|she|it|they|the\s+film|the\s+work|the\s+book)\s+(?:was\s+)?{re.escape(verb)}\s+by\s+{value_tail}",
                        ),
                        (
                            relation,
                            rf"\b(?:his|her|its|their)\s+(?:{role_noun})\s+(?:is|was)\s+{value_tail}",
                        ),
                    ])
                    if family == "visual_art":
                        visual_credit_name = (
                            rf"(?P<value>(?-i:{SURFACE_PROPER_SEGMENT}))(?<!\.)"
                        )
                        patterns.append((
                            relation,
                            rf"\b(?:painting|artwork|portrait)\s+by\s+"
                            rf"{visual_credit_name}"
                            rf"(?=\s+(?:who|whose|and|exhibited|displayed|"
                            rf"shown|held|located|at|in|on)\b|[,;.]|$)",
                        ))
                    if enable_source_grounded_terminal_repair:




                        credit_name = rf"(?P<value>{SURFACE_PROPER_SEGMENT})"
                        patterns.append((
                            relation,
                            rf"\b(?:{role_noun})\s*(?::|is\s+|was\s+)?"
                            rf"{credit_name}(?=\s+(?:who|whose|and)\b|[,;.]|$)",
                        ))
                        if sentence_id <= 2:




                            patterns.append((
                                relation,
                                rf"^(?:(?![.;]).){{0,220}}?\b"
                                rf"(?:{re.escape(verb)})\s+by\s+{value_tail}",
                            ))
                    if enable_inverse_relation_completion and family == "composition":
                        inverse_composer_patterns = [
                            rf"{bound_pattern}.{{0,100}}?\b(?:music|score)\s+(?:was\s+)?by\s+{value_tail}",
                            rf"{bound_pattern}.{{0,100}}?\b(?:music|score)\s+(?:was\s+)?composed\s+by\s+{value_tail}",
                        ]
                        patterns.extend(
                            (relation, pattern)
                            for pattern in inverse_composer_patterns
                        )
                        inverse_patterns.update(inverse_composer_patterns)
                        if title_scoped_sentence:
                            inverse_pattern = rf"\b(?:with\s+)?(?:music|score)\s+(?:was\s+)?(?:composed\s+)?by\s+{value_tail}"
                            patterns.append((relation, inverse_pattern))
                            inverse_patterns.add(inverse_pattern)
            if "performance" in predicate_families:
                patterns.extend([
                    (
                        "performed by",
                        rf"{bound_pattern}.{{0,48}}?\b(?:was\s+)?"
                        rf"(?:performed|recorded|sung)\s+by\s+{value_tail}",
                    ),
                    (
                        "performed by",
                        rf"(?:the\s+)?(?:performer|singer|vocalist)\s+of\s+"
                        rf"{bound_pattern}\s+(?:is|was)\s+{value_tail}",
                    ),
                ])
                if title_scoped_sentence:
                    patterns.append((
                        "performed by",
                        rf"\b(?:it|the\s+(?:song|work|recording))\s+(?:was\s+)?"
                        rf"(?:performed|recorded|sung)\s+by\s+{value_tail}",
                    ))












                    performance_credit = (
                        rf"(?P<value>(?-i:{SURFACE_PROPER_SEGMENT}))(?<!\.)"
                    )
                    patterns.append((
                        "performed by",
                        rf"\b(?:album|single|record|recording|song|track)\s+by\s+"
                        rf"(?:[A-Za-zÀ-ÖØ-öø-ÿ'\-]+\s+){{0,4}}"
                        rf"(?:music\s+)?(?:artist|singer-songwriter|singer|"
                        rf"performer|vocalist)\s+{performance_credit}"
                        rf"(?=\s+(?:who|whose|and|from|for|on|with)\b|[,;.]|$)",
                    ))
            if "founding" in predicate_families and output_type != "date":
                patterns.extend([
                    (
                        "founded by",
                        rf"{bound_pattern}.{{0,48}}?\b(?:was\s+)?"
                        rf"(?:co-founded|founded|established)\s+by\s+{value_tail}",
                    ),
                    (
                        "founded by",
                        rf"(?:the\s+)?(?:co-?founder|founder)\s+of\s+"
                        rf"{bound_pattern}\s+(?:is|was)\s+{value_tail}",
                    ),
                ])
            if "education" in predicate_families:
                education_value = r"(?P<value>[^,;.]{1,120}?)"
                patterns.extend([
                    (
                        "educated at",
                        rf"{bound_pattern}.{{0,60}}?\b(?:attended|studied\s+at|"
                        rf"graduated\s+from|was\s+educated\s+at|trained\s+at)\s+"
                        rf"{education_value}(?=[,;.]|$)",
                    ),
                    (
                        "educated at",
                        rf"{bound_pattern}(?:['\u2019]s)?\s+"
                        rf"(?:college|school|university)\s+(?:is|was)\s+"
                        rf"{education_value}(?=[,;.]|$)",
                    ),
                ])
            if "membership" in predicate_families:
                patterns.extend([
                    (
                        "member of",
                        rf"{bound_pattern}.{{0,36}}?\b(?:played\s+for|played\s+on|joined|was\s+(?:a\s+)?member\s+of)\s+{value_tail}",
                    ),
                    (
                        "member of",
                        rf"{bound_pattern}(?:['\u2019]s)?\s+(?:club|team)\s+(?:is|was)\s+{value_tail}",
                    ),
                ])
            if role_direction_required or "kinship" in predicate_families:
                kinship_terms = sorted(
                    set(operator.get("relation_tokens") or [])
                    & {"child", "daughter", "father", "mother", "parent", "son", "spouse"}
                ) or ["father", "mother", "parent", "spouse"]
                relation_group = "|".join(re.escape(term) for term in kinship_terms)
                patterns.extend([
                    (
                        kinship_terms[0],
                        rf"{bound_pattern}(?:['\u2019]s)?\s+(?:{relation_group})\s+(?:is|was)\s+{value_tail}",
                    ),
                    (
                        kinship_terms[0],
                        rf"{reverse_value}\s+(?:is|was)\s+(?:the\s+)?(?:{relation_group})\s+of\s+{bound_pattern}",
                    ),
                ])
                if title_scoped_sentence:
                    patterns.extend([
                        (
                            kinship_terms[0],
                            rf"\b(?:his|her|their)\s+(?:{relation_group})\s+(?:is|was)\s+{value_tail}",
                        ),
                        (
                            kinship_terms[0],
                            rf"{reverse_value}\s+(?:is|was)\s+(?:his|her|their)\s+(?:{relation_group})\b",
                        ),
                    ])
                    if enable_source_grounded_terminal_repair:
                        person_name = rf"(?P<value>{SURFACE_PROPER_SEGMENT})"
                        spouse_name = (
                            r"(?P<value>(?-i:(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'.-]*)"
                            r"(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'.-]*){0,4}))"
                        )
                        profession = (
                            r"(?:actor|actress|artist|author|composer|director|"
                            r"educator|filmmaker|journalist|politician|producer|"
                            r"screenwriter|singer|writer)"
                        )
                        patterns.extend([
                            (
                                kinship_terms[0],
                                rf"\b(?:he|she|they|the\s+subject)\b"
                                rf"(?:(?!\b(?:who|whose|which|whom)\b)[^.;]){{0,120}}?"
                                rf"\b(?:met|married)\s+(?:his|her|their)\s+"
                                rf"(?:future\s+)?(?:husband|wife|spouse),?\s+"
                                rf"(?:(?:[A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s+)?"
                                rf"{profession}(?:\s+and\s+{profession})?\s+"
                                rf"{person_name}(?=\s+(?:who|whom|whose)\b|[,.;]|$)",
                            ),
                            (
                                kinship_terms[0],
                                rf"\b(?:he|she|they|the\s+subject)\s+"
                                rf"(?:was\s+)?(?:married|wed)\s+(?:to\s+)?{person_name}"
                                rf"(?=\s+(?:who|whom|whose)\b|[,.;]|$)",
                            ),
                            (
                                kinship_terms[0],
                                rf"\b(?:princess|prince|queen|king|duchess|duke)?"
                                rf"\s*by\s+marriage\s+to\s+{person_name}"
                                rf"(?=\s+(?:who|whom|whose)\b|[,.;]|$)",
                            ),
                            (
                                kinship_terms[0],
                                rf"\b(?:wife|husband|spouse|consort|chief\s+queen)\s+"
                                rf"of\s+(?:(?:King|Queen|Prince|Princess)\s+)?"
                                rf"{spouse_name}(?=\s+(?:who|whom|whose)\b|[,.;]|$)",
                            ),
                            (
                                kinship_terms[0],
                                rf"\b(?:wife|husband|spouse|consort)\s+of\s+"
                                rf"[^,;.]{{0,80}}?{spouse_name}"
                                rf"(?=\s+(?:who|whom|whose)\b|[,.;]|$)",
                            ),
                        ])
                if enable_inverse_relation_completion:
                    canonical_relation = str(
                        operator.get("canonical_relation")
                        or role_contract.get("canonical_relation")
                        or ""
                    )
                    if canonical_relation == "father_of":
                        if title_scoped_document:



                            appositive_parent = (
                                r"^.{0,140}?\b(?:was\s+)?(?:an?\s+)?"
                                r"(?:(?:eldest|first|illegitimate|older|only|"
                                r"second|younger|youngest)\s+){0,2}"
                                r"(?:son|daughter|child)\s+of\s+"
                                r"(?P<value>[^.;]{1,120}?)"
                                r"(?=\s+and\s+(?:(?:his|her|the)\s+)?"
                                r"(?:(?:first|second)\s+)?(?:wife|mother)\b)"
                            )
                            patterns.append(("father", appositive_parent))
                            inverse_patterns.add(appositive_parent)
                            inverse_pattern = (
                                r"^.{0,140}?\b(?:was\s+)?(?:an?\s+)?"
                                r"(?:(?:eldest|first|illegitimate|older|only|"
                                r"second|younger|youngest)\s+){0,2}"
                                r"(?:son|daughter|child)\s+of\s+"
                                r"(?P<value>[^,;.]{1,100}?)"
                                r"(?=\s+(?:and|by)\s+(?:(?:his|her|the)\s+)?"
                                r"(?:(?:first|second)\s+)?(?:wife|mother)\b|"
                                r"[,;.]|$)"
                            )
                            patterns.append(("father", inverse_pattern))
                            inverse_patterns.add(inverse_pattern)
                            possessive_name = (
                                r"(?-i:(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ.-]*)"
                                r"(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ.-]*){0,4})"
                            )
                            possessive_parent = (
                                rf"(?P<value>{possessive_name})['\u2019](?:s)?\s+"
                                rf"and\s+{possessive_name}['\u2019](?:s)?\s+"
                                r"(?:(?:eldest|first|only|younger|youngest)\s+)?"
                                r"(?:son|daughter|child)\b"
                            )
                            patterns.append(("father", possessive_parent))
                            inverse_patterns.add(possessive_parent)
                        if enable_title_grounded_completion:
                            inverse_pattern = rf"{bound_pattern}.{{0,100}}?\b(?:son|daughter|child)\s+of\s+(?P<value>[^.;]{{1,120}}?)(?=\s+and\s+(?:(?:his|her|the)\s+)?(?:(?:first|second)\s+)?(?:wife|mother)\b)"
                            patterns.append(("father", inverse_pattern))
                            inverse_patterns.add(inverse_pattern)
                            if title_scoped_sentence:
                                inverse_pattern = rf"\b(?:he|she|the\s+subject)\s+.{{0,80}}?\b(?:son|daughter|child)\s+of\s+(?P<value>[^.;]{{1,120}}?)(?=\s+and\s+(?:(?:his|her|the)\s+)?(?:wife|mother)\b)"
                                patterns.append(("father", inverse_pattern))
                                inverse_patterns.add(inverse_pattern)
                        inverse_pattern = rf"{bound_pattern}.{{0,100}}?\b(?:son|daughter|child)\s+of\s+{parent_value}"
                        patterns.append(("father", inverse_pattern))
                        inverse_patterns.add(inverse_pattern)
                        if title_scoped_sentence:
                            inverse_pattern = rf"\b(?:he|she|the\s+subject)\s+.{{0,80}}?\b(?:son|daughter|child)\s+of\s+{parent_value}"
                            patterns.append(("father", inverse_pattern))
                            inverse_patterns.add(inverse_pattern)
                    elif canonical_relation == "mother_of":






                        mother_value = (
                            r"(?P<value>[^,;.()]{1,90}?)"
                            r"(?=\s*\(|\s+and\s+(?:(?:the|his|her|their)\s+)?"
                            r"(?:sister|brother|son|daughter|wife|husband|"
                            r"spouse|consort)\s+of\b|[,;.]|$)"
                        )
                        inverse_pattern = rf"{bound_pattern}.{{0,120}}?\b(?:son|daughter|child)\s+of\s+[^.;]{{1,100}}?\s+(?:and|by)\s+(?:(?:his|her|the)\s+)?(?:(?:first|second)\s+)?(?:wife|mother),?\s+{mother_value}"
                        patterns.append(("mother", inverse_pattern))
                        inverse_patterns.add(inverse_pattern)
                        if title_scoped_document:
                            explicit_maternal_first = (
                                r"^.{0,140}?\b(?:was\s+)?(?:an?\s+)?"
                                r"(?:(?:eldest|first|second|third|fourth|fifth|"
                                r"older|only|younger|youngest)\s+){0,2}"
                                r"(?:son|daughter|child)\s+of\s+"
                                r"(?P<value>(?:Queen|Princess|Duchess|Empress|"
                                r"Countess|Lady)\s+[^,;.]{1,90}?)\s+and\s+"
                                r"[^,;.]{1,100}?(?=[,;.]|$)"
                            )
                            patterns.append(("mother", explicit_maternal_first))
                            inverse_patterns.add(explicit_maternal_first)
                            inverse_pattern = rf"\b(?:he|she|the\s+subject)\s+.{{0,100}}?\b(?:son|daughter|child)\s+of\s+[^.;]{{1,100}}?\s+(?:and|by)\s+(?:(?:his|her|the)\s+)?(?:(?:first|second)\s+)?(?:wife|mother),?\s+{mother_value}"
                            patterns.append(("mother", inverse_pattern))
                            inverse_patterns.add(inverse_pattern)
                            generic_mother = (
                                r"^.{0,140}?\b(?:was\s+)?(?:an?\s+)?"
                                r"(?:(?:eldest|first|older|only|second|younger|"
                                r"youngest)\s+){0,2}(?:son|daughter|child)\s+of\s+"
                                r"[^.;]{1,100}?\s+(?:and|by)\s+"
                                r"(?:(?:his|her|the)\s+)?"
                                r"(?:(?:first|second)\s+)?(?:wife|mother),?\s+"
                                + mother_value
                            )
                            patterns.append(("mother", generic_mother))
                            inverse_patterns.add(generic_mother)






                            paired_parent_mother = (
                                r"^.{0,140}?\b(?:was\s+)?(?:an?\s+)?"
                                r"(?:(?:eldest|first|second|third|fourth|fifth|"
                                r"older|only|younger|youngest)\s+){0,2}"
                                r"(?:son|daughter|child)\s+of\s+"
                                r"[^,;.]{1,100}?\s+and\s+"
                                + mother_value
                            )
                            patterns.append(("mother", paired_parent_mother))
                            inverse_patterns.add(paired_parent_mother)
                            possessive_name = (
                                r"(?-i:(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ.-]*)"
                                r"(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ.-]*){0,4})"
                            )
                            possessive_mother = (
                                rf"{possessive_name}['\u2019](?:s)?\s+and\s+"
                                rf"(?P<value>{possessive_name})['\u2019](?:s)?\s+"
                                r"(?:(?:eldest|first|only|younger|youngest)\s+)?"
                                r"(?:son|daughter|child)\b"
                            )
                            patterns.append(("mother", possessive_mother))
                            inverse_patterns.add(possessive_mother)
            if predicate_families & {"containment", "geopolitical_location"}:
                patterns.extend([
                    (
                        "located in",
                        rf"{bound_pattern}.{{0,36}}?\b(?:is\s+|was\s+)?(?:located|situated|based)\s+in\s+{value_tail}",
                    ),
                    (
                        "located in",
                        rf"{bound_pattern}\s+(?:is|was)\s+in\s+{value_tail}",
                    ),
                ])
            if "distribution" in predicate_families and output_type == "date":



                patterns.extend([
                    (
                        "released in",
                        rf"{bound_pattern}.{{0,48}}?\b(?:released|published|premiered)\s+"
                        rf"(?:in|on)\s+{date_value}",
                    ),
                    (
                        "released in",
                        rf"{bound_pattern}.{{0,36}}?\b(?:is|was)\s+(?:an?\s+)?"
                        rf"{date_value}\s+(?:[A-Za-z-]+\s+){{0,4}}"
                        rf"(?:film|movie|novel|album|book|song|work)\b",
                    ),
                ])
                if title_scoped_sentence and enable_source_grounded_terminal_repair:
                    patterns.extend([
                        (
                            "released in",
                            rf"\b(?:it|the\s+(?:film|movie|novel|album|book|song|work))\s+"
                            rf"(?:was\s+)?(?:released|published|premiered)\s+"
                            rf"(?:in|on)\s+{date_value}",
                        ),
                        (
                            "released in",
                            rf"^(?:an?\s+)?{date_value}\s+(?:[A-Za-z-]+\s+){{0,4}}"
                            rf"(?:film|movie|novel|album|book|song|work)\b",
                        ),
                    ])
            if "end_time" in predicate_families:
                patterns.append((
                    "ended in",
                    rf"{bound_pattern}.{{0,36}}?\b(?:was\s+)?(?:abolished|ceased|closed|dissolved|ended)\b\s*(?:in|on)?\s*{value_tail}",
                ))
            if output_type == "number" and relation_tokens & {"occur", "occurred", "times"}:
                patterns.append((
                    "occurred times",
                    r"\b(?P<value>\d+(?:[,.]\d+)*)\s+times\b",
                ))
            matched_pattern = False
            clause_boundary_counted = False
            for relation, pattern in patterns:
                match = re.search(pattern, sentence, flags=re.IGNORECASE)
                if match:
                    matched_pattern = True
                    diagnostics["pattern_match"] += 1
                    if (
                        title_scoped_sentence
                        and not clause_boundary_counted
                        and re.search(
                            r"\b(?:born|died)\s+(?:in|at)\s+[^.;]{1,180}?,\s+"
                            r"(?:he|she|they|the\s+subject)\s+"
                            r"(?:is|was|became|served)\b",
                            sentence,
                            flags=re.IGNORECASE,
                        )
                    ):
                        diagnostics["title_lead_clause_boundary_match"] += 1
                        clause_boundary_counted = True
                    inverse_match = pattern in inverse_patterns
                    if inverse_match:
                        diagnostics["inverse_relation_pattern_match"] += 1






                        if not title_scoped_document:
                            diagnostics[
                                "inverse_relation_instance_identity_rejected"
                            ] += 1
                            continue
                        identity_mode = (
                            f"{title_identity_mode}_inverse"
                            if title_identity_mode
                            else "title_or_alias_scoped_inverse"
                        )
                    else:
                        identity_mode = (
                            title_identity_mode
                            if title_scoped_document
                            else "explicit_bound_mention"
                        )
                    emitted = emit(
                        doc_id,
                        sentence_id,
                        sentence,
                        relation,
                        match.group("value"),
                        inverse_match=inverse_match,
                        identity_mode=identity_mode,
                        source_title=title,
                        source_span_start=source_span_start,
                        source_span_end=source_span_end,
                    )
                    if emitted and title_scoped_document:
                        diagnostics["title_partition_compiled_fact"] += 1
                        for family in predicate_families:
                            diagnostics[
                                f"title_partition_compiled_{family}"
                            ] += 1
                    if inverse_match and emitted:
                        diagnostics["inverse_relation_accepted_fact"] += 1
            if not matched_pattern:
                diagnostics["bound_sentence_no_pattern"] += 1
    if predicate_gap_repair and not extracted:
        diagnostics["predicate_gap_sentence_repair_failed"] += 1
    return extracted


def _execute_slot_dependency_query_plan(
    *,
    question: str,
    plan: Dict[str, Any],
    relation_index: Dict[str, Any],
    anchor_items: List[Dict[str, Any]],
    rank_map: Dict[int, int],
    get_doc_content: Callable[[int], str],
    max_facts_per_key: int,
    max_key_fanout: int,
    access_stats: Dict[str, int],
    enable_surface_materializer: bool,
    enable_bound_completion: bool = False,
    bound_completion_lookup: Optional[
        Callable[[str, int], List[Dict[str, Any]]]
    ] = None,
    bound_completion_top_k: int = 6,
    bound_completion_max_calls: int = 2,
    enable_bound_completion_cost_admission: bool = True,
    bound_completion_max_risk: float = 0.42,
    bound_completion_min_net_utility: float = 0.20,
    compute_weight: float = 0.08,
    risk_weight: float = 0.62,
    enable_canonical_bound_completion: bool = True,
    enable_completion_selectivity_feedback: bool = True,
    enable_query_plan_root_contract: bool = True,
    enable_inverse_relation_completion: bool = True,
    enable_title_grounded_completion: bool = False,
    enable_source_grounded_terminal_repair: bool = False,
    enable_title_catalog_root_plan: Optional[bool] = None,
    commit_title_root_plan: bool = True,
    enable_identity_safe_root_commit: bool = False,
) -> Tuple[List[Dict[str, Any]], Counter, Dict[str, Any]]:
    """Execute slot plans with exact joins and bounded miss completion."""
    title_catalog_root_plan_enabled = (
        bool(enable_title_grounded_completion)
        if enable_title_catalog_root_plan is None
        else bool(enable_title_catalog_root_plan)
    )
    rows: List[Dict[str, Any]] = []
    rejects: Counter = Counter()
    operators = list(plan.get("operators") or [])
    plan_kind = str(plan.get("plan_kind") or "")
    final_answer_type = str(plan.get("final_answer_type") or "entity")
    execution_trace: Dict[str, Any] = {
        "version": str(plan.get("version") or ""),
        "root_bindings": [],
        "physical_operator_trace": [],
        "exact_probe_count": 0,
        "exact_probe_hit_count": 0,
        "completion_enabled": bool(
            enable_bound_completion and bound_completion_lookup
        ),
        "completion_call_count": 0,
        "completion_doc_count": 0,
        "completion_extracted_fact_count": 0,
        "completion_path_row_count": 0,
        "residual_terminal_completion_path_row_count": 0,
        "completion_scheduler": (
            "canonical_bound_selectivity_feedback_v2"
            if enable_completion_selectivity_feedback
            else "role_consistent_risk_cost_best_first_v1"
        ),
        "canonical_bound_completion_enabled": bool(
            enable_canonical_bound_completion
        ),
        "completion_selectivity_feedback_enabled": bool(
            enable_completion_selectivity_feedback
        ),
        "query_plan_root_contract_enabled": bool(
            enable_query_plan_root_contract
        ),
        "inverse_relation_completion_enabled": bool(
            enable_inverse_relation_completion
        ),
        "title_grounded_completion_enabled": bool(
            enable_title_grounded_completion
        ),
        "source_grounded_terminal_repair_enabled": bool(
            enable_source_grounded_terminal_repair
        ),
        "title_catalog_root_plan_enabled": bool(
            title_catalog_root_plan_enabled
        ),
        "title_root_plan_commit_enabled": bool(commit_title_root_plan),
        "identity_safe_root_commit_enabled": bool(
            enable_identity_safe_root_commit
        ),
        "root_plan_policy": (
            "provenance_alias_refined_terminal_commit_v12_2"
            if enable_identity_safe_root_commit
            else "clean_speculative_late_commit_v11_8"
        ),
        "root_plan_alternatives": [],
        "root_plan_selected": {},
        "completion_selectivity_feedback": [],
        "completion_probe_trace": [],
        "terminal_tuple_completion_candidates": [],
        "completion_extraction_diagnostics": {},
        "completion_reject_reason_counts": {},
    }
    completion_rejects: Counter = Counter()
    completion_extraction_diagnostics: Counter = Counter()
    completion_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    completion_feedback_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    if plan_kind == "path_projection" and len(operators) >= 2:
        legacy_root_bindings = _bind_explicit_query_entities(
            question,
            relation_index,
            operator=operators[0],
            expected_count=(1 if enable_canonical_bound_completion else None),
            max_bindings=(1 if enable_canonical_bound_completion else 4),
            require_operator_match=not bool(enable_bound_completion),
            canonicalize_surface=bool(enable_canonical_bound_completion),
            enforce_root_contract=bool(enable_query_plan_root_contract),
            use_title_entity_keys=False,
        )
        title_root_bindings: List[Dict[str, Any]] = []
        title_identity_bindings: List[Dict[str, Any]] = []
        if title_catalog_root_plan_enabled:
            title_root_bindings = _bind_explicit_query_entities(
                question,
                relation_index,
                operator=operators[0],
                expected_count=(1 if enable_canonical_bound_completion else None),
                max_bindings=(1 if enable_canonical_bound_completion else 4),
                require_operator_match=not bool(enable_bound_completion),
                canonicalize_surface=bool(enable_canonical_bound_completion),
                enforce_root_contract=bool(enable_query_plan_root_contract),
                use_title_entity_keys=True,
            )
            if enable_identity_safe_root_commit:




                title_identity_bindings = _bind_explicit_query_entities(
                    question,
                    relation_index,
                    operator=None,
                    expected_count=1,
                    max_bindings=1,
                    require_operator_match=False,
                    canonicalize_surface=True,
                    enforce_root_contract=bool(enable_query_plan_root_contract),
                    use_title_entity_keys=True,
                )

        def rank_root_bindings(
            bindings: List[Dict[str, Any]], variant: str,
        ) -> List[Dict[str, Any]]:
            ranked = [dict(root, root_plan_variant=variant) for root in bindings]
            ranked.sort(key=lambda root: (
                -float(_bound_completion_admission(
                    bound_surface=str(root.get("surface") or ""),
                    operator=operators[0],
                    call_count=0,
                    max_calls=max(1, int(bound_completion_max_calls)),
                    top_k=int(bound_completion_top_k),
                    max_risk=1.0,
                    min_net_utility=-1.0,
                    compute_weight=float(compute_weight),
                    risk_weight=float(risk_weight),
                ).get("net_utility", 0.0) or 0.0),
                -int(root.get("token_width", 0) or 0),
                str(root.get("key") or ""),
            ))
            return ranked

        legacy_root_bindings = rank_root_bindings(
            legacy_root_bindings, "legacy_operator_root"
        )
        title_root_bindings = rank_root_bindings(
            title_root_bindings, "title_catalog_root"
        )
        legacy_root_identities = {
            (str(root.get("key") or ""), str(root.get("surface") or "").casefold())
            for root in legacy_root_bindings
        }
        title_root_bindings = [
            root for root in title_root_bindings
            if (
                str(root.get("key") or ""),
                str(root.get("surface") or "").casefold(),
            ) not in legacy_root_identities
        ]
        for root in legacy_root_bindings + title_root_bindings:
            root["identity_safe_for_commit"] = True
            root["identity_conflict_title_surface"] = ""
            root["identity_refinement_mode"] = ""
            root["identity_refinement_reason"] = "not_required"
            root["identity_alias_refined"] = False
            root["canonical_identity_key"] = str(root.get("key") or "")
        if enable_identity_safe_root_commit and title_identity_bindings:
            for legacy_root in legacy_root_bindings:
                legacy_start_value = legacy_root.get("token_start")
                legacy_end_value = legacy_root.get("token_end")
                legacy_start = int(
                    legacy_start_value if legacy_start_value is not None else -1
                )
                legacy_end = int(
                    legacy_end_value if legacy_end_value is not None else -1
                )
                conflicting_title = next((
                    title_root
                    for title_root in title_identity_bindings
                    if int(
                        title_root.get("token_start")
                        if title_root.get("token_start") is not None else -1
                    ) <= legacy_start
                    and int(
                        title_root.get("token_end")
                        if title_root.get("token_end") is not None else -1
                    ) >= legacy_end
                    and int(title_root.get("token_width", 0) or 0)
                    > int(legacy_root.get("token_width", 0) or 0)
                    and bool(title_root.get("title_catalog_exact"))
                ), None)
                if conflicting_title:
                    refinement = _refine_legacy_root_with_title_identity(
                        legacy_root,
                        conflicting_title,
                    )
                    legacy_root["identity_safe_for_commit"] = bool(
                        refinement["compatible"]
                    )
                    legacy_root["identity_alias_refined"] = bool(
                        refinement["compatible"]
                    )
                    legacy_root["identity_refinement_mode"] = str(
                        refinement["mode"]
                    )
                    legacy_root["identity_refinement_reason"] = str(
                        refinement["reason"]
                    )
                    legacy_root["canonical_identity_key"] = str(
                        refinement["canonical_identity_key"]
                    )
                    legacy_root["identity_conflict_title_surface"] = (
                        "" if refinement["compatible"]
                        else str(refinement["title_surface"])
                    )
                    legacy_root["identity_refined_title_surface"] = str(
                        refinement["title_surface"]
                    )
        root_bindings = legacy_root_bindings + title_root_bindings
        if not root_bindings:
            rejects["slot_root_not_bound_from_query_mention"] += 1
            return rows, rejects, execution_trace

        base_rejects = Counter(rejects)
        root_plan_outcomes: List[Dict[str, Any]] = []
        for root in root_bindings:
            root_plan_variant = str(
                root.get("root_plan_variant") or "legacy_operator_root"
            )
            if (
                root_plan_variant == "title_catalog_root"
                and any(
                    outcome["variant"] == "legacy_operator_root"
                    and outcome["terminal_rows"]
                    and bool(
                        outcome["root"].get(
                            "identity_safe_for_commit", True
                        )
                    )
                    for outcome in root_plan_outcomes
                )
            ):
                root_plan_outcomes.append({
                    "variant": root_plan_variant,
                    "root": dict(root),
                    "rows": [],
                    "terminal_rows": [],
                    "rejects": Counter(),
                    "completion_rejects": Counter(),
                    "completion_call_count": 0,
                    "terminal_tuple_completion_candidates": [],
                    "skipped_reason": "legacy_terminal_complete",
                })
                continue
            root_row_start = len(rows)
            root_reject_start = Counter(rejects)
            root_completion_reject_start = Counter(completion_rejects)
            root_completion_call_start = int(
                execution_trace["completion_call_count"]
            )
            root_terminal_tuple_candidates: List[Dict[str, Any]] = []
            seen_paths: Set[Tuple[Tuple[int, ...], str]] = set()
            root_key = str(root.get("key") or "")
            root_surface = str(root.get("surface") or "")
            frontier = [{
                "bound_key": root_key,
                "bound_surface": root_surface,
                "lineage_root_key": root_key,
                "lineage_bound_keys": [root_key],
                "path_items": [],
                "visited_keys": {root_key},
                "slot_bindings": {
                    "$root": {"surface": root_surface, "key": root_key},
                },
                "binding_confidence": float(
                    root.get("binding_confidence", 0.0) or 0.0
                ),
                "role_direction_matches": [],
                "root_plan_variant": root_plan_variant,
            }]
            for operator_index, compiled_operator in enumerate(operators):
                next_frontier = []
                final_operator = operator_index == len(operators) - 1



                def completion_priority(state: Dict[str, Any]) -> Tuple[Any, ...]:
                    causal_predecessor = any(
                        bool(item.get("bound_completion_fact"))
                        for item in state.get("path_items") or []
                    )
                    estimate = _bound_completion_admission(
                        bound_surface=str(state.get("bound_surface") or ""),
                        operator=compiled_operator,
                        call_count=0,
                        max_calls=max(1, int(bound_completion_max_calls)),
                        top_k=int(bound_completion_top_k),
                        max_risk=1.0,
                        min_net_utility=-1.0,
                        compute_weight=float(compute_weight),
                        risk_weight=float(risk_weight),
                        causal_predecessor=causal_predecessor,
                        terminal_operator=bool(final_operator),
                    )
                    return (
                        -float(estimate.get("net_utility", 0.0) or 0.0),
                        -min(
                            [
                                float(match.get("role_direction_confidence", 0.0) or 0.0)
                                for match in state.get("role_direction_matches") or []
                            ]
                            or [float(state.get("binding_confidence", 0.0) or 0.0)]
                        ),
                        sum(
                            int(rank_map.get(int(item.get("doc_idx", -1)), 999999))
                            for item in state.get("path_items") or []
                        ),
                        -float(state.get("binding_confidence", 0.0) or 0.0),
                        str(state.get("bound_key") or ""),
                    )

                frontier = sorted(frontier, key=completion_priority)
                for scheduler_rank, state in enumerate(frontier):
                    lineage_keys = list(state.get("lineage_bound_keys") or [])
                    if (
                        not lineage_keys
                        or str(lineage_keys[-1]) != str(state.get("bound_key") or "")
                    ):
                        rejects[
                            f"operator_{operator_index}_lineage_input_mismatch"
                        ] += 1
                        continue
                    candidates = _lookup_items_by_exact_entity_key(
                        relation_index,
                        str(state["bound_key"]),
                        max_per_key=max_facts_per_key,
                        max_fanout=max_key_fanout,
                        access_stats=access_stats,
                        allow_title_provenance_alias=(
                            operator_index > 0
                            or str(state.get("root_plan_variant") or "")
                            == "title_catalog_root"
                        ),
                    )
                    execution_trace["exact_probe_count"] += 1
                    matching_candidates = [
                        item
                        for item in candidates
                        if item not in state["path_items"]
                        and _plan_operator_matches_item(item, compiled_operator)
                    ]
                    if matching_candidates:
                        execution_trace["exact_probe_hit_count"] += 1
                    access_path = "exact_entity_predicate_index"
                    if (
                        not matching_candidates
                        and enable_bound_completion
                        and bound_completion_lookup is not None
                    ):
                        decision = _bound_completion_admission(
                            bound_surface=str(state["bound_surface"]),
                            operator=compiled_operator,
                            call_count=max(
                                0,
                                int(execution_trace["completion_call_count"])
                                - root_completion_call_start,
                            ),
                            max_calls=int(bound_completion_max_calls),
                            top_k=int(bound_completion_top_k),
                            max_risk=float(bound_completion_max_risk),
                            min_net_utility=float(
                                bound_completion_min_net_utility
                            ),
                            compute_weight=float(compute_weight),
                            risk_weight=float(risk_weight),
                            causal_predecessor=any(
                                bool(item.get("bound_completion_fact"))
                                for item in state.get("path_items") or []
                            ),
                            terminal_operator=bool(final_operator),
                        )
                        if not enable_bound_completion_cost_admission:
                            decision["accepted"] = (
                                max(
                                    0,
                                    int(execution_trace["completion_call_count"])
                                    - root_completion_call_start,
                                )
                                < max(0, int(bound_completion_max_calls))
                            )
                            decision["reason"] = (
                                "cost_admission_disabled"
                                if decision["accepted"]
                                else "completion_call_budget_exhausted"
                            )
                        decision.update({
                            "operator_id": str(
                                compiled_operator.get("operator_id") or ""
                            ),
                            "operator_index": int(operator_index),
                            "bound_surface": str(state["bound_surface"]),
                            "bound_key": str(state["bound_key"]),
                            "scheduler_rank": int(scheduler_rank),
                        })
                        if decision["accepted"]:
                            cache_key = (
                                str(state["bound_key"]),
                                str(compiled_operator.get("operator_id") or ""),
                            )
                            if cache_key not in completion_cache:
                                completion_query = _bound_completion_query(
                                    str(state["bound_surface"]),
                                    compiled_operator,
                                    structured=bool(
                                        enable_canonical_bound_completion
                                    ),
                                )
                                documents = bound_completion_lookup(
                                    completion_query,
                                    max(1, int(bound_completion_top_k)),
                                )
                                bound_preview_tokens = set(
                                    _surface_word_tokens(
                                        str(state["bound_surface"])
                                    )
                                )
                                relation_preview_tokens = set(
                                    compiled_operator.get("relation_tokens")
                                    or []
                                )
                                probe_trace = {
                                    "query": completion_query,
                                    "operator_id": str(
                                        compiled_operator.get("operator_id")
                                        or ""
                                    ),
                                    "predicate_families": sorted(
                                        str(value)
                                        for value in (
                                            compiled_operator.get(
                                                "predicate_families"
                                            )
                                            or []
                                        )
                                    ),
                                    "bound_key": str(state["bound_key"]),
                                    "bound_surface": str(
                                        state["bound_surface"]
                                    ),
                                    "documents": [],
                                }
                                for completion_document in documents or []:
                                    try:
                                        completion_doc_id = int(
                                            completion_document.get("doc_id", -1)
                                        )
                                        completion_rank = int(
                                            completion_document.get(
                                                "rank", bound_completion_top_k
                                            )
                                        )
                                    except (TypeError, ValueError):
                                        continue
                                    if completion_doc_id >= 0:
                                        rank_map[completion_doc_id] = min(
                                            int(
                                                rank_map.get(
                                                    completion_doc_id,
                                                    completion_rank,
                                                )
                                            ),
                                            completion_rank,
                                        )
                                    content = str(
                                        completion_document.get("content")
                                        or ""
                                    )
                                    title = str(
                                        completion_document.get("title")
                                        or (
                                            content.splitlines()[0].strip()
                                            if content else ""
                                        )
                                    )
                                    sentences = [
                                        part.strip()
                                        for part in re.split(
                                            r"(?<=[.!?])\s+|[\r\n]+",
                                            content,
                                        )
                                        if part.strip()
                                    ]
                                    bound_previews = [
                                        sentence[:240]
                                        for sentence in sentences
                                        if bound_preview_tokens
                                        and bound_preview_tokens.issubset(
                                            set(_surface_word_tokens(sentence))
                                        )
                                    ][:2]
                                    relation_previews = [
                                        sentence[:240]
                                        for sentence in sentences
                                        if relation_preview_tokens
                                        & _tokens(sentence)
                                    ][:2]
                                    probe_trace["documents"].append({
                                        "doc_id": completion_doc_id,
                                        "rank": completion_rank,
                                        "score": round(
                                            float(
                                                completion_document.get(
                                                    "score", 0.0
                                                )
                                                or 0.0
                                            ),
                                            6,
                                        ),
                                        "source": str(
                                            completion_document.get(
                                                "completion_source", "unknown"
                                            )
                                        ),
                                        "title": title[:160],
                                        "bound_sentence_previews": (
                                            bound_previews
                                        ),
                                        "relation_sentence_previews": (
                                            relation_previews
                                        ),
                                    })
                                execution_trace["completion_call_count"] += 1
                                execution_trace["completion_doc_count"] += len(
                                    documents or []
                                )
                                diagnostics_before = Counter(
                                    completion_extraction_diagnostics
                                )
                                completion_cache[cache_key] = (
                                    _extract_bound_completion_items(
                                        bound_surface=str(state["bound_surface"]),
                                        bound_key=str(state["bound_key"]),
                                        operator=compiled_operator,
                                        documents=list(documents or []),
                                        item_id_seed=-1000000 - 1000 * operator_index,
                                        diagnostics=(
                                            completion_extraction_diagnostics
                                        ),
                                        relation_index=relation_index,
                                        enable_predicate_gap_repair=bool(
                                            enable_completion_selectivity_feedback
                                        ),
                                        enable_inverse_relation_completion=bool(
                                            enable_inverse_relation_completion
                                        ),
                                        enable_title_grounded_completion=bool(
                                            enable_title_grounded_completion
                                        ),
                                        enable_source_grounded_terminal_repair=bool(
                                            enable_source_grounded_terminal_repair
                                        ),
                                    )
                                )
                                for completion_item in completion_cache[
                                    cache_key
                                ]:
                                    completion_item.update({
                                        "completion_operator_id": str(
                                            compiled_operator.get(
                                                "operator_id"
                                            )
                                            or ""
                                        ),
                                        "completion_operator_index": int(
                                            operator_index
                                        ),
                                        "completion_slot_stage": str(
                                            compiled_operator.get(
                                                "slot_plan_stage"
                                            )
                                            or (
                                                "final_answer"
                                                if final_operator
                                                else "intermediate_binding"
                                            )
                                        ),
                                        "completion_causal_predecessor": any(
                                            bool(
                                                path_item.get(
                                                    "bound_completion_fact"
                                                )
                                            )
                                            for path_item in state.get(
                                                "path_items"
                                            )
                                            or []
                                        ),
                                    })
                                probe_trace["extracted_facts"] = [
                                    {
                                        "completion_fact_id": str(
                                            item.get("completion_fact_id")
                                            or ""
                                        ),
                                        "triple": list(
                                            item.get("triple") or []
                                        ),
                                        "tuple_key": str(
                                            item.get("tuple_key") or ""
                                        ),
                                        "operator_index": int(
                                            item.get(
                                                "completion_operator_index",
                                                operator_index,
                                            )
                                        ),
                                        "slot_stage": str(
                                            item.get(
                                                "completion_slot_stage"
                                            )
                                            or ""
                                        ),
                                        "source_doc_id": int(
                                            item.get(
                                                "completion_source_doc_id",
                                                item.get("doc_idx", -1),
                                            )
                                        ),
                                        "source_sentence_id": int(
                                            item.get(
                                                "completion_source_sentence_id",
                                                -1,
                                            )
                                        ),
                                        "direction": str(
                                            item.get(
                                                "completion_extraction_direction"
                                            )
                                            or ""
                                        ),
                                        "identity_mode": str(
                                            item.get(
                                                "completion_instance_identity_mode"
                                            )
                                            or ""
                                        ),
                                    }
                                    for item in completion_cache[cache_key]
                                ]
                                diagnostics_after = Counter(
                                    completion_extraction_diagnostics
                                )
                                observed = {
                                    key: int(diagnostics_after.get(key, 0))
                                    - int(diagnostics_before.get(key, 0))
                                    for key in (
                                        "indexed_fact_rows",
                                        "indexed_bound_match",
                                        "indexed_predicate_match",
                                        "indexed_accepted_fact",
                                        "predicate_gap_sentence_repair_fact",
                                        "inverse_relation_pattern_match",
                                        "inverse_relation_accepted_fact",
                                        "inverse_relation_instance_identity_rejected",
                                        "title_scoped_documents",
                                        "title_scoped_sentences",
                                        "title_grounded_documents",
                                        "title_grounded_regex_fallback",
                                        "title_identity_roman_conflict_rejected",
                                        "title_identity_year_conflict_rejected",
                                        "title_identity_stronger_candidate_rejected",
                                        "title_identity_ambiguous_rejected",
                                        "title_lead_clause_boundary_match",
                                        "source_grounded_fact",
                                        "source_grounded_terminal_repair_fact",
                                    )
                                }
                                scanned_rows = max(
                                    0, observed["indexed_fact_rows"]
                                )
                                bound_rows = max(
                                    0, observed["indexed_bound_match"]
                                )
                                predicate_rows = max(
                                    0, observed["indexed_predicate_match"]
                                )
                                extracted_rows = len(completion_cache[cache_key])
                                causal_rows = sum(
                                    bool(item.get("bound_completion_fact"))
                                    and bool(item.get("completion_fact_id"))
                                    and bool(
                                        item.get(
                                            "completion_instance_identity_validated"
                                        )
                                    )
                                    and int(
                                        item.get(
                                            "completion_source_doc_id",
                                            item.get("doc_idx", -1),
                                        )
                                    ) >= 0
                                    for item in completion_cache[cache_key]
                                )
                                indexed_rows = max(
                                    0, observed["indexed_accepted_fact"]
                                )
                                repaired_rows = max(
                                    0,
                                    observed[
                                        "predicate_gap_sentence_repair_fact"
                                    ],
                                )
                                text_predicate_rows = max(
                                    0,
                                    extracted_rows
                                    - indexed_rows
                                    - repaired_rows,
                                )
                                ambiguity_risk = _clamp(
                                    0.08 * max(0, extracted_rows - 1),
                                    0.0,
                                    0.32,
                                )
                                realized_risk = _clamp(
                                    (
                                        0.06
                                        if indexed_rows > 0
                                        else 0.18
                                        if repaired_rows > 0
                                        else 0.26
                                        if text_predicate_rows > 0
                                        else 0.58
                                    )
                                    + ambiguity_risk
                                )
                                realized_utility = _clamp(
                                    (
                                        0.78
                                        if indexed_rows > 0
                                        else 0.66
                                        if repaired_rows > 0
                                        else 0.58
                                        if text_predicate_rows > 0
                                        else 0.0
                                    )
                                    + 0.06 * min(1, extracted_rows)
                                )
                                normalized_scan_cost = _clamp(
                                    scanned_rows
                                    / max(1.0, float(len(documents or []) * 24))
                                )
                                realized_net_utility = (
                                    realized_utility
                                    - float(risk_weight) * realized_risk
                                    - float(compute_weight)
                                    * normalized_scan_cost
                                )
                                post_probe_accepted = bool(
                                    extracted_rows > 0
                                    and causal_rows == extracted_rows
                                    and (
                                        not enable_completion_selectivity_feedback
                                        or (
                                            realized_risk
                                            <= float(bound_completion_max_risk)
                                            and realized_net_utility
                                            >= float(
                                                bound_completion_min_net_utility
                                            )
                                        )
                                    )
                                )
                                observed.update({
                                    "bound_selectivity": round(
                                        bound_rows / max(1, scanned_rows), 6
                                    ),
                                    "predicate_selectivity": round(
                                        predicate_rows / max(1, bound_rows), 6
                                    ),
                                    "output_cardinality": int(extracted_rows),
                                    "causal_lineage_fact_count": int(
                                        causal_rows
                                    ),
                                    "operator_index": int(operator_index),
                                    "slot_stage": str(
                                        compiled_operator.get(
                                            "slot_plan_stage"
                                        )
                                        or (
                                            "final_answer"
                                            if final_operator
                                            else "intermediate_binding"
                                        )
                                    ),
                                    "residual_terminal_probe": bool(
                                        final_operator
                                        and any(
                                            bool(
                                                path_item.get(
                                                    "bound_completion_fact"
                                                )
                                            )
                                            for path_item in state.get(
                                                "path_items"
                                            )
                                            or []
                                        )
                                    ),
                                    "text_predicate_tuple_count": int(
                                        text_predicate_rows
                                    ),
                                    "ambiguity_risk": round(
                                        float(ambiguity_risk), 6
                                    ),
                                    "realized_risk": round(
                                        float(realized_risk), 6
                                    ),
                                    "realized_utility": round(
                                        float(realized_utility), 6
                                    ),
                                    "normalized_scan_cost": round(
                                        float(normalized_scan_cost), 6
                                    ),
                                    "realized_net_utility": round(
                                        float(realized_net_utility), 6
                                    ),
                                    "post_probe_accepted": post_probe_accepted,
                                    "physical_action": (
                                        "accept_without_selectivity_feedback"
                                        if extracted_rows > 0
                                        and not enable_completion_selectivity_feedback
                                        else
                                        "reject_post_probe_risk_cost"
                                        if extracted_rows > 0
                                        and not post_probe_accepted
                                        else "accept_indexed_tuple"
                                        if indexed_rows > 0
                                        else "repair_openie_predicate_gap"
                                        if repaired_rows > 0
                                        else "accept_text_predicate_tuple"
                                        if text_predicate_rows > 0
                                        else "stop_zero_predicate_selectivity"
                                        if bound_rows > 0 and predicate_rows == 0
                                        else "stop_no_bound_match"
                                    ),
                                })
                                source_counts = Counter(
                                    str(
                                        document.get(
                                            "completion_source", "unknown"
                                        )
                                    )
                                    for document in documents or []
                                )
                                observed["document_source_counts"] = dict(
                                    source_counts
                                )
                                if (
                                    enable_completion_selectivity_feedback
                                    and not post_probe_accepted
                                ):
                                    completion_cache[cache_key] = []
                                    if extracted_rows > 0:
                                        completion_rejects[
                                            "completion_post_probe_risk_cost_rejected"
                                        ] += 1
                                completion_feedback_cache[cache_key] = dict(
                                    observed
                                )
                                execution_trace[
                                    "completion_selectivity_feedback"
                                ].append(observed)
                                probe_trace["observed_selectivity"] = dict(
                                    observed
                                )
                                execution_trace["completion_probe_trace"].append(
                                    probe_trace
                                )
                            matching_candidates = list(
                                completion_cache.get(cache_key) or []
                            )
                            execution_trace[
                                "completion_extracted_fact_count"
                            ] += len(matching_candidates)
                            decision["completion_query"] = _bound_completion_query(
                                str(state["bound_surface"]),
                                compiled_operator,
                                structured=bool(
                                    enable_canonical_bound_completion
                                ),
                            )
                            decision["extracted_fact_count"] = len(
                                matching_candidates
                            )
                            decision["observed_selectivity"] = dict(
                                completion_feedback_cache.get(cache_key) or {}
                            )
                            decision["post_probe_accepted"] = bool(
                                matching_candidates
                            )
                            access_path = "bounded_dense_text_completion"
                            if not matching_candidates:
                                completion_rejects[
                                    "completion_no_operator_matching_fact"
                                ] += 1
                        else:
                            completion_rejects[str(decision["reason"])] += 1
                        if len(execution_trace["physical_operator_trace"]) < 64:
                            execution_trace["physical_operator_trace"].append(
                                decision
                            )
                    for item in matching_candidates:
                        equivalent_input_keys = _equivalent_entity_keys(
                            relation_index,
                            str(state["bound_key"]),
                            allow_title_provenance_alias=(
                                operator_index > 0
                                or root_plan_variant == "title_catalog_root"
                            ),
                        )
                        oriented_output = _oriented_endpoint_for_operator_binding(
                            item,
                            compiled_operator,
                            str(state["bound_key"]),
                            equivalent_keys=equivalent_input_keys,
                        )
                        if oriented_output is None:
                            rejects[
                                f"operator_{operator_index}_role_direction_mismatch"
                            ] += 1
                            continue
                        surface = str(oriented_output.get("surface") or "")
                        output_key = str(oriented_output.get("key") or "")
                        if (
                            not output_key
                            or output_key in state["visited_keys"]
                        ):
                            rejects[f"operator_{operator_index}_reuses_bound_slot"] += 1
                            continue
                        relation_tokens = set(item.get("relation_tokens") or set())
                        operator_output_type = str(
                            compiled_operator.get("output_type") or "entity"
                        )
                        if (
                            not final_operator
                            and operator_output_type
                            in {"entity", "person", "organization", "work"}
                            and (
                                re.fullmatch(
                                    r"\s*(?:[1-9]\d{3}|20\d{2}|21\d{2})\s*",
                                    surface,
                                )
                                or (
                                    SURFACE_MONTH_RE.search(surface)
                                    and NUMBER_VALUE_RE.search(surface)
                                )
                            )
                        ):
                            rejects[
                                f"operator_{operator_index}_attribute_bound_as_entity"
                            ] += 1
                            continue
                        if terminal_value_contract_compatibility(
                            surface,
                            (
                                final_answer_type
                                if final_operator else operator_output_type
                            ),
                            relation_tokens,
                            output_role=str(
                                oriented_output.get("output_side") or ""
                            ),
                            source_text=str(item.get("source_sentence") or ""),
                            relation_families=sorted(
                                set(item.get("relation_families") or [])
                                | set(item.get("predicate_families") or [])
                                | set(compiled_operator.get("relation_families") or [])
                                | set(compiled_operator.get("predicate_families") or [])
                            ),
                        )["status"] != TERMINAL_TYPE_COMPATIBLE:
                            rejects[
                                "terminal_answer_type_mismatch"
                                if final_operator
                                else f"operator_{operator_index}_output_type_mismatch"
                            ] += 1





                            if final_operator and state.get("path_items"):
                                source_doc_id = int(
                                    item.get(
                                        "completion_source_doc_id",
                                        item.get("doc_idx", -1),
                                    )
                                )
                                compound_recovery = (
                                    _extract_compound_terminal_place_recovery(
                                        candidate=surface,
                                        answer_type=final_answer_type,
                                        relation_tokens=relation_tokens,
                                        predicate_families=(
                                            compiled_operator.get(
                                                "predicate_families"
                                            )
                                            or []
                                        ),
                                        bound_surface=str(
                                            state.get("bound_surface") or root_surface
                                        ),
                                        source_doc_id=source_doc_id,
                                        get_doc_content=get_doc_content,
                                    )
                                )
                                if not compound_recovery:
                                    continue
                                recovered_surface = str(
                                    compound_recovery.get("candidate") or ""
                                ).strip()
                                if not recovered_surface:
                                    continue
                                completion_path = list(state["path_items"]) + [item]
                                bridge_entity = str(
                                    state.get("bound_surface") or root_surface
                                )
                                completion_row = _build_multi_hop_row(
                                    candidate=recovered_surface,
                                    answer_type=final_answer_type,
                                    plan=(
                                        "provenance_bound_terminal_tuple_completion"
                                    ),
                                    anchor_entity=root_surface,
                                    bridge_entity=bridge_entity,
                                    path_items=completion_path,
                                    path_roles=[
                                        str(
                                            operator.get("operator_id")
                                            or f"op{index}"
                                        )
                                        for index, operator in enumerate(operators)
                                    ],
                                    anchor_score=float(
                                        root.get("binding_confidence", 1.0) or 1.0
                                    ),
                                    role_score=1.0,
                                    rank_map=rank_map,
                                    get_doc_content=get_doc_content,
                                    enable_surface_materializer=bool(
                                        enable_surface_materializer
                                    ),
                                )
                                if completion_row is not None:
                                    role_matches = list(
                                        state.get("role_direction_matches") or []
                                    ) + [oriented_output]
                                    completion_facts = [
                                        path_item
                                        for path_item in completion_path
                                        if path_item.get("bound_completion_fact")
                                        or str(
                                            path_item.get("access_path") or ""
                                        ).startswith("bound_")
                                    ]
                                    completion_lineage_valid = all(
                                        bool(
                                            path_item.get("completion_fact_id")
                                        )
                                        and int(
                                            path_item.get(
                                                "completion_source_doc_id",
                                                path_item.get("doc_idx", -1),
                                            )
                                        ) >= 0
                                        and bool(
                                            path_item.get(
                                                "completion_instance_identity_validated"
                                            )
                                        )
                                        for path_item in completion_facts
                                    )
                                    terminal_triple = list(item.get("triple") or [])
                                    completion_row.update({
                                        "terminal_tuple_completion_candidate": True,
                                        "terminal_tuple_completion_reason": (
                                            "compound_sentence_terminal_place_recovery"
                                        ),
                                        "terminal_tuple_completion_version": (
                                            "v12_4_compound_sentence_recovery"
                                        ),
                                        "terminal_compound_recovery_required": True,
                                        "terminal_compound_surface_recovered": True,
                                        "terminal_recovered_from": str(
                                            compound_recovery.get("recovered_from") or ""
                                        ),
                                        "terminal_recovered_relation": str(
                                            compound_recovery.get("recovered_relation") or ""
                                        ),
                                        "terminal_contract_version": str(
                                            plan.get("version") or ""
                                        ),
                                        "query_plan_version": str(
                                            plan.get("version") or ""
                                        ),
                                        "query_plan_kind": plan_kind,
                                        "slot_plan_stage": "final_answer",
                                        "final_answer_type": final_answer_type,
                                        "final_predicate_match": True,
                                        "final_answer_type_match": True,
                                        "bridge_predicate_match": True,
                                        "bridge_bound": True,
                                        "operator_complete": (
                                            len(completion_path) == len(operators)
                                        ),
                                        "rooted_lineage_complete": True,
                                        "slot_dependency_validated": bool(
                                            len(role_matches) == len(operators)
                                            and all(
                                                bool(
                                                    match.get(
                                                        "role_direction_validated"
                                                    )
                                                )
                                                for match in role_matches
                                            )
                                        ),
                                        "role_direction_matches": role_matches,
                                        "production_eligible": False,
                                        "root_identity_safe_for_commit": bool(
                                            root.get(
                                                "identity_safe_for_commit", True
                                            )
                                        ),
                                        "root_binding_confidence": round(
                                            float(
                                                root.get(
                                                    "binding_confidence", 0.0
                                                )
                                                or 0.0
                                            ),
                                            6,
                                        ),
                                        "root_plan_variant": root_plan_variant,
                                        "lineage_root_key": str(
                                            state.get("lineage_root_key")
                                            or root_key
                                        ),
                                        "lineage_bound_keys": lineage_keys
                                        + [output_key],
                                        "candidate_predicate_families": list(
                                            compiled_operator.get(
                                                "predicate_families"
                                            )
                                            or []
                                        ),
                                        "terminal_input_slot": str(
                                            compiled_operator.get("input_slot")
                                            or ""
                                        ),
                                        "terminal_output_slot": str(
                                            compiled_operator.get("output_slot")
                                            or ""
                                        ),
                                        "terminal_bound_surface": bridge_entity,
                                        "terminal_bound_key": str(
                                            state.get("bound_key") or ""
                                        ),
                                        "terminal_source_doc_id": int(
                                            compound_recovery.get(
                                                "source_doc_id", source_doc_id
                                            )
                                        ),
                                        "terminal_source_sentence_id": int(
                                            compound_recovery.get(
                                                "source_sentence_id", -1
                                            )
                                        ),
                                        "terminal_source_sentence_span": (
                                            str(
                                                compound_recovery.get(
                                                    "source_sentence_span", ""
                                                )
                                            )
                                        ),
                                        "terminal_recovered_triple": [
                                            bridge_entity,
                                            str(
                                                compound_recovery.get(
                                                    "recovered_relation", ""
                                                )
                                            ),
                                            recovered_surface,
                                        ],
                                        "completion_lineage_source_valid": bool(
                                            completion_lineage_valid
                                        ),
                                        "bound_completion_used": bool(
                                            completion_facts
                                        ),
                                    })
                                    if len(root_terminal_tuple_candidates) < 64:
                                        root_terminal_tuple_candidates.append(
                                            completion_row
                                        )
                            continue
                        slot_bindings = dict(state["slot_bindings"])
                        slot_bindings[str(compiled_operator.get("output_slot") or "")] = {
                            "surface": surface,
                            "key": output_key,
                        }
                        next_frontier.append({
                            "bound_key": output_key,
                            "bound_surface": surface,
                            "lineage_root_key": str(
                                state.get("lineage_root_key") or root_key
                            ),
                            "lineage_bound_keys": lineage_keys + [output_key],
                            "path_items": list(state["path_items"]) + [item],
                            "visited_keys": set(state["visited_keys"]) | {output_key},
                            "slot_bindings": slot_bindings,
                            "binding_confidence": float(
                                state.get("binding_confidence", 0.0) or 0.0
                            ),
                            "role_direction_matches": list(
                                state.get("role_direction_matches") or []
                            ) + [oriented_output],
                            "physical_access_paths": list(
                                state.get("physical_access_paths") or []
                            ) + [
                                str(item.get("access_path") or access_path)
                            ],
                            "root_plan_variant": str(
                                state.get("root_plan_variant")
                                or root_plan_variant
                            ),
                        })
                if not next_frontier:
                    rejects[f"operator_{operator_index}_no_exact_match"] += 1
                    frontier = []
                    break
                next_frontier.sort(key=lambda state: (
                    sum(
                        int(rank_map.get(int(item.get("doc_idx", -1)), 999999))
                        for item in state["path_items"]
                    ),
                    str(state["bound_key"]),
                ))
                frontier = next_frontier[: max(4, int(max_facts_per_key))]

            for state in frontier:
                path_items = list(state["path_items"])
                candidate = str(state["bound_surface"] or "")
                candidate_key = str(state["bound_key"] or "")
                path_key = (
                    tuple(int(item.get("item_id", -1)) for item in path_items),
                    candidate_key,
                )
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)
                bridge_entity = (
                    str(path_items[-2].get("object") or "")
                    if len(path_items) >= 2 else root_surface
                )
                row = _build_multi_hop_row(
                    candidate=candidate,
                    answer_type=final_answer_type,
                    plan=f"slot_dependency_path_{len(path_items)}hop",
                    anchor_entity=root_surface,
                    bridge_entity=bridge_entity,
                    path_items=path_items,
                    path_roles=[
                        str(compiled_operator.get("operator_id") or f"op{index}")
                        for index, compiled_operator in enumerate(operators)
                    ],
                    anchor_score=float(root.get("binding_confidence", 1.0) or 1.0),
                    role_score=1.0,
                    rank_map=rank_map,
                    get_doc_content=get_doc_content,
                    enable_surface_materializer=bool(enable_surface_materializer),
                )
                if row is None:
                    continue
                completion_path_items = [
                    item for item in path_items
                    if item.get("bound_completion_fact")
                    or str(item.get("access_path") or "").startswith("bound_")
                ]
                completion_lineage = [
                    {
                        "completion_fact_id": str(
                            item.get("completion_fact_id") or ""
                        ),
                        "tuple_key": str(item.get("tuple_key") or ""),
                        "bound_input_key": str(
                            item.get("completion_bound_input_key") or ""
                        ),
                        "source_doc_id": int(
                            item.get(
                                "completion_source_doc_id",
                                item.get("doc_idx", -1),
                            )
                        ),
                        "source_sentence_id": int(
                            item.get("completion_source_sentence_id", -1)
                        ),
                        "source_title": str(
                            item.get("completion_source_title") or ""
                        ),
                        "source_grounded": bool(
                            item.get("completion_source_grounded")
                        ),
                        "terminal_source_repair": bool(
                            item.get("completion_terminal_source_repair")
                        ),
                        "contextual_qualifier_risk": bool(
                            item.get("completion_contextual_qualifier_risk")
                        ),
                        "predicate_subtypes": list(
                            item.get("predicate_subtypes") or []
                        ),
                        "direction": str(
                            item.get("completion_extraction_direction") or ""
                        ),
                        "operator_id": str(
                            item.get("completion_operator_id") or ""
                        ),
                        "operator_index": int(
                            item.get("completion_operator_index", -1)
                        ),
                        "slot_stage": str(
                            item.get("completion_slot_stage") or ""
                        ),
                        "causal_predecessor": bool(
                            item.get("completion_causal_predecessor")
                        ),
                        "identity_mode": str(
                            item.get("completion_instance_identity_mode") or ""
                        ),
                        "instance_identity_validated": bool(
                            item.get(
                                "completion_instance_identity_validated"
                            )
                        ),
                    }
                    for item in completion_path_items
                ]
                completion_causal_lineage_validated = bool(
                    completion_lineage
                    and all(
                        lineage.get("completion_fact_id")
                        and lineage.get("tuple_key")
                        and int(lineage.get("source_doc_id", -1)) >= 0
                        and lineage.get("bound_input_key")
                        and lineage.get("instance_identity_validated")
                        for lineage in completion_lineage
                    )
                )
                residual_terminal_completion_used = bool(
                    completion_causal_lineage_validated
                    and any(
                        lineage.get("slot_stage") == "final_answer"
                        and lineage.get("causal_predecessor")
                        for lineage in completion_lineage
                    )
                )
                source_grounded_completion_used = bool(
                    completion_causal_lineage_validated
                    and any(
                        lineage.get("source_grounded")
                        for lineage in completion_lineage
                    )
                )
                source_grounded_terminal_repair_used = bool(
                    completion_causal_lineage_validated
                    and any(
                        lineage.get("slot_stage") == "final_answer"
                        and lineage.get("source_grounded")
                        and lineage.get("terminal_source_repair")
                        and int(lineage.get("source_sentence_id", -1)) >= 0
                        for lineage in completion_lineage
                    )
                )
                role_direction_required = any(
                    bool(operator.get("role_direction_required"))
                    for operator in operators
                )
                role_direction_matches = list(
                    state.get("role_direction_matches") or []
                )
                role_direction_validated = bool(
                    len(role_direction_matches) == len(operators)
                    and all(
                        bool(match.get("role_direction_validated"))
                        for match in role_direction_matches
                    )
                )
                row.update({
                    "terminal_contract_version": str(plan.get("version") or ""),
                    "query_plan_version": str(plan.get("version") or ""),
                    "query_plan_kind": plan_kind,
                    "validation_tier": (
                        "schema_constrained_binding_completion_v4"
                        if str(plan.get("version") or "").endswith("_v4")
                        else "exact_slot_binding_v3"
                    ),
                    "shadow_only": False,
                    "production_eligible": True,
                    "base_validation_passed": True,
                    "slot_plan_stage": "final_answer",
                    "final_answer_type": final_answer_type,
                    "final_predicate_match": True,
                    "bridge_predicate_match": True,
                    "final_answer_type_match": True,
                    "bridge_eligible": False,
                    "rooted_lineage_complete": True,
                    "operator_complete": len(path_items) == len(operators),
                    "slot_dependency_validated": bool(
                        not role_direction_required
                        or role_direction_validated
                    ),
                    "role_direction_required": role_direction_required,
                    "role_direction_validated": role_direction_validated,
                    "role_direction_matches": role_direction_matches,
                    "exact_join_key_count": max(1, len(path_items) - 1),
                    "lineage_depth": len(path_items),
                    "plan_stage_count": len(operators),
                    "relation_obligations": list(
                        plan.get("relation_obligations") or []
                    ),
                    "root_binding_method": str(
                        root.get("binding_method") or ""
                    ),
                    "root_plan_variant": root_plan_variant,
                    "root_binding_confidence": round(
                        float(root.get("binding_confidence", 0.0) or 0.0), 6
                    ),
                    "root_contract_valid": bool(
                        root.get("root_contract_valid")
                    ),
                    "root_contract_reason": str(
                        root.get("root_contract_reason") or ""
                    ),
                    "root_identity_safe_for_commit": bool(
                        root.get("identity_safe_for_commit", True)
                    ),
                    "root_identity_conflict_title_surface": str(
                        root.get("identity_conflict_title_surface") or ""
                    ),
                    "lineage_root_key": str(
                        state.get("lineage_root_key") or root_key
                    ),
                    "lineage_bound_keys": list(
                        state.get("lineage_bound_keys") or []
                    ),
                    "slot_bindings": state["slot_bindings"],
                    "physical_access_paths": list(
                        state.get("physical_access_paths") or []
                    ),
                    "bound_completion_used": bool(completion_path_items),
                    "completion_causal_lineage_validated": bool(
                        completion_causal_lineage_validated
                    ),
                    "residual_terminal_completion_used": bool(
                        residual_terminal_completion_used
                    ),
                    "source_grounded_completion_used": bool(
                        source_grounded_completion_used
                    ),
                    "source_grounded_terminal_repair_used": bool(
                        source_grounded_terminal_repair_used
                    ),
                    "completion_chain_depth": len(completion_lineage),
                    "bound_completion_lineage": completion_lineage,
                    "bound_completion_fact_ids": list(dict.fromkeys(
                        str(item.get("completion_fact_id") or "")
                        for item in completion_path_items
                        if item.get("completion_fact_id")
                    )),
                    "bound_completion_doc_ids": sorted({
                        int(item.get("doc_idx"))
                        for item in completion_path_items
                        if item.get("doc_idx") is not None
                    }),
                    "bound_completion_tuple_keys": list(dict.fromkeys(
                        str(item.get("tuple_key") or "")
                        for item in completion_path_items
                        if item.get("tuple_key")
                    )),
                })
                if row.get("completion_causal_lineage_validated"):
                    execution_trace["completion_path_row_count"] += 1
                if row.get("residual_terminal_completion_used"):
                    execution_trace[
                        "residual_terminal_completion_path_row_count"
                    ] = int(
                        execution_trace.get(
                            "residual_terminal_completion_path_row_count",
                            0,
                        )
                        or 0
                    ) + 1
                rows.append(row)
            root_rows = list(rows[root_row_start:])
            terminal_rows = [
                row for row in root_rows
                if bool(row.get("production_eligible"))
                and bool(row.get("operator_complete"))
                and bool(row.get("rooted_lineage_complete"))
                and bool(row.get("slot_dependency_validated"))
                and bool(row.get("final_predicate_match"))
                and bool(row.get("final_answer_type_match"))
                and (
                    not bool(row.get("bound_completion_used"))
                    or bool(row.get("completion_causal_lineage_validated"))
                )
            ]
            root_plan_outcomes.append({
                "variant": root_plan_variant,
                "root": dict(root),
                "rows": root_rows,
                "terminal_rows": terminal_rows,
                "rejects": rejects - root_reject_start,
                "completion_rejects": (
                    completion_rejects - root_completion_reject_start
                ),
                "completion_call_count": max(
                    0,
                    int(execution_trace["completion_call_count"])
                    - root_completion_call_start,
                ),
                "terminal_tuple_completion_candidates": list(
                    root_terminal_tuple_candidates
                ),
            })

        legacy_outcome = next((
            outcome for outcome in root_plan_outcomes
            if outcome["variant"] == "legacy_operator_root"
        ), None)
        title_outcome = next((
            outcome for outcome in root_plan_outcomes
            if outcome["variant"] == "title_catalog_root"
        ), None)
        selected_outcome = legacy_outcome
        legacy_identity_safe = bool(
            not legacy_outcome
            or legacy_outcome["root"].get(
                "identity_safe_for_commit", True
            )
        )
        legacy_terminal_usable = bool(
            legacy_outcome
            and legacy_outcome["terminal_rows"]
            and legacy_identity_safe
        )
        selection_reason = (
            "legacy_alias_refined_terminal_complete"
            if (
                legacy_terminal_usable
                and bool(
                    legacy_outcome["root"].get("identity_alias_refined")
                )
            )
            else "legacy_terminal_complete"
            if legacy_terminal_usable
            else "legacy_identity_conflict_pending"
            if (
                legacy_outcome
                and legacy_outcome["terminal_rows"]
                and not legacy_identity_safe
            )
            else "legacy_fallback"
            if legacy_outcome else "no_legacy_root"
        )
        if (
            commit_title_root_plan
            and title_outcome
            and title_outcome["terminal_rows"]





            and (
                not legacy_outcome
                or not (legacy_outcome.get("rows") or [])
                or not legacy_identity_safe
            )
        ):
            selected_outcome = title_outcome
            selection_reason = (
                "title_identity_conflict_recovery"
                if not legacy_identity_safe
                else "title_terminal_recovery"
            )
        elif (
            title_outcome
            and title_outcome["terminal_rows"]
            and not commit_title_root_plan
        ):
            selection_reason = "title_terminal_shadow_only"
        elif (
            enable_identity_safe_root_commit
            and legacy_outcome
            and legacy_outcome["terminal_rows"]
            and not legacy_identity_safe
        ):
            selected_outcome = None
            selection_reason = "legacy_root_identity_conflict_veto"

        rows = list((selected_outcome or {}).get("rows") or [])
        rejects = Counter(base_rejects)
        rejects.update((selected_outcome or {}).get("rejects") or {})
        if selection_reason == "legacy_root_identity_conflict_veto":
            rejects["legacy_root_identity_conflict_veto"] += len(
                legacy_outcome["terminal_rows"]
            )
        selected_completion_rejects = Counter(
            (selected_outcome or {}).get("completion_rejects") or {}
        )
        execution_trace["root_bindings"] = (
            [dict((selected_outcome or {}).get("root") or {})]
            if selected_outcome else []
        )
        execution_trace["root_plan_alternatives"] = [
            {
                "variant": str(outcome["variant"]),
                "root_key": str(outcome["root"].get("key") or ""),
                "root_surface": str(outcome["root"].get("surface") or ""),
                "binding_method": str(
                    outcome["root"].get("binding_method") or ""
                ),
                "operator_match_count": int(
                    outcome["root"].get("operator_match_count", 0) or 0
                ),
                "identity_safe_for_commit": bool(
                    outcome["root"].get("identity_safe_for_commit", True)
                ),
                "identity_conflict_title_surface": str(
                    outcome["root"].get(
                        "identity_conflict_title_surface"
                    ) or ""
                ),
                "identity_alias_refined": bool(
                    outcome["root"].get("identity_alias_refined")
                ),
                "identity_refinement_mode": str(
                    outcome["root"].get("identity_refinement_mode") or ""
                ),
                "identity_refinement_reason": str(
                    outcome["root"].get("identity_refinement_reason") or ""
                ),
                "canonical_identity_key": str(
                    outcome["root"].get("canonical_identity_key") or ""
                ),
                "row_count": len(outcome["rows"]),
                "terminal_row_count": len(outcome["terminal_rows"]),
                "terminal_tuple_completion_candidate_count": len(
                    outcome.get("terminal_tuple_completion_candidates") or []
                ),
                "completion_call_count": int(
                    outcome["completion_call_count"]
                ),
                "reject_reason_counts": dict(outcome["rejects"]),
                "skipped_reason": str(
                    outcome.get("skipped_reason") or ""
                ),
                "selected": outcome is selected_outcome,
            }
            for outcome in root_plan_outcomes
        ]
        execution_trace["root_plan_selected"] = {
            "variant": str(
                (selected_outcome or {}).get("variant") or "none"
            ),
            "reason": selection_reason,
            "terminal_row_count": len(
                (selected_outcome or {}).get("terminal_rows") or []
            ),
            "title_commit_enabled": bool(commit_title_root_plan),
        }
        execution_trace["terminal_tuple_completion_candidates"] = list(
            (selected_outcome or {}).get(
                "terminal_tuple_completion_candidates"
            ) or []
        )
        execution_trace["terminal_tuple_completion_candidate_count"] = len(
            execution_trace["terminal_tuple_completion_candidates"]
        )
        execution_trace["completion_path_row_count"] = sum(
            bool(row.get("completion_causal_lineage_validated"))
            for row in rows
        )
        execution_trace[
            "residual_terminal_completion_path_row_count"
        ] = sum(
            bool(row.get("residual_terminal_completion_used"))
            for row in rows
        )
        execution_trace["completion_reject_reason_counts"] = dict(
            selected_completion_rejects
        )
        execution_trace["completion_extraction_diagnostics"] = dict(
            completion_extraction_diagnostics
        )
        return rows, rejects, execution_trace

    if plan_kind == "branch_temporal_compare" and operators:
        direction = str(plan.get("answer_operator") or "argmin_root")
        explicit_roots = _bind_explicit_query_entities(
            question,
            relation_index,
            operator=operators[0],
            expected_count=2,
            max_bindings=2,
            canonicalize_surface=False,
        )
        execution_trace["root_bindings"] = [
            dict(root) for root in explicit_roots
        ]
        if len(explicit_roots) != 2:
            rejects["comparison_requires_exactly_two_roots"] += 1
            return rows, rejects, execution_trace

        branch_results: List[Dict[str, Any]] = []
        for root in explicit_roots:
            root_key = str(root.get("key") or "")
            root_surface = str(root.get("surface") or "")
            frontier = [{
                "bound_key": root_key,
                "bound_surface": root_surface,
                "path_items": [],
                "visited_keys": {root_key},
                "role_direction_matches": [],
            }]
            for operator_index, compiled_operator in enumerate(operators):
                next_frontier: List[Dict[str, Any]] = []
                for state in frontier:
                    bound_key = str(state.get("bound_key") or "")
                    candidates = _lookup_items_by_exact_entity_key(
                        relation_index,
                        bound_key,
                        max_per_key=max_facts_per_key,
                        max_fanout=max_key_fanout,
                        access_stats=access_stats,
                    )
                    execution_trace["exact_probe_count"] += 1
                    matched = [
                        item
                        for item in candidates
                        if item not in state["path_items"]
                        and _plan_operator_matches_item(
                            item, compiled_operator
                        )
                    ]
                    if matched:
                        execution_trace["exact_probe_hit_count"] += 1
                    for item in matched:
                        oriented_output = _oriented_endpoint_for_operator_binding(
                            item,
                            compiled_operator,
                            bound_key,
                            equivalent_keys=_equivalent_entity_keys(
                                relation_index,
                                bound_key,
                                allow_title_provenance_alias=(
                                    operator_index > 0
                                    or str(
                                        state.get("root_plan_variant") or ""
                                    ) == "title_catalog_root"
                                ),
                            ),
                        )
                        if oriented_output is None:
                            rejects[
                                f"comparison_operator_{operator_index}_role_direction_mismatch"
                            ] += 1
                            continue
                        output_surface = str(
                            oriented_output.get("surface") or ""
                        )
                        output_key = str(oriented_output.get("key") or "")
                        if (
                            not output_key
                            or output_key in state["visited_keys"]
                        ):
                            continue
                        if not _terminal_value_matches_contract(
                            output_surface,
                            str(
                                compiled_operator.get("output_type")
                                or "entity"
                            ),
                            set(item.get("relation_tokens") or set()),
                        ):
                            rejects[
                                f"comparison_operator_{operator_index}_output_type_mismatch"
                            ] += 1
                            continue
                        next_frontier.append({
                            "bound_key": output_key,
                            "bound_surface": output_surface,
                            "path_items": list(state["path_items"]) + [item],
                            "visited_keys": (
                                set(state["visited_keys"]) | {output_key}
                            ),
                            "role_direction_matches": list(
                                state.get("role_direction_matches") or []
                            ) + [oriented_output],
                        })
                if not next_frontier:
                    rejects[
                        f"comparison_operator_{operator_index}_no_exact_match"
                    ] += 1
                    frontier = []
                    break
                next_frontier.sort(key=lambda state: (
                    sum(
                        int(
                            rank_map.get(
                                int(item.get("doc_idx", -1)), 999999
                            )
                        )
                        for item in state["path_items"]
                    ),
                    str(state["bound_key"]),
                ))
                frontier = next_frontier[: max(4, int(max_facts_per_key))]

            dated_states = []
            for state in frontier:
                temporal_key = _temporal_value_key(state["bound_surface"])
                if temporal_key is not None:
                    dated_states.append((temporal_key, state))
            if not dated_states:
                rejects["comparison_terminal_date_unparseable"] += 1
                continue
            dated_states.sort(key=lambda pair: (
                pair[0],
                len(pair[1]["path_items"]),
                str(pair[1]["bound_key"]),
            ))
            selected_key, selected_state = (
                dated_states[-1]
                if direction == "argmax_root" else dated_states[0]
            )
            branch_results.append({
                "root": root,
                "temporal_key": selected_key,
                "terminal_surface": str(selected_state["bound_surface"]),
                "path_items": list(selected_state["path_items"]),
                "role_direction_matches": list(
                    selected_state.get("role_direction_matches") or []
                ),
            })

        if len(branch_results) != 2:
            rejects["comparison_incomplete_branch_evidence"] += 1
            return rows, rejects, execution_trace
        reverse = direction == "argmax_root"
        ordered = sorted(
            branch_results,
            key=lambda result: (
                result["temporal_key"],
                str(result["root"].get("key") or ""),
            ),
            reverse=reverse,
        )
        if ordered[0]["temporal_key"] == ordered[1]["temporal_key"]:
            rejects["comparison_tied_terminal_values"] += 1
            return rows, rejects, execution_trace

        winner = ordered[0]
        all_path_items = [
            item
            for branch in branch_results
            for item in branch["path_items"]
        ]
        winner_surface = str(winner["root"].get("surface") or "")
        role_direction_required = any(
            bool(operator.get("role_direction_required"))
            for operator in operators
        )
        role_direction_matches = [
            match
            for branch in branch_results
            for match in branch.get("role_direction_matches") or []
        ]
        role_direction_validated = bool(
            len(role_direction_matches) == len(operators) * len(branch_results)
            and all(
                bool(match.get("role_direction_validated"))
                for match in role_direction_matches
            )
        )
        row = _build_multi_hop_row(
            candidate=winner_surface,
            answer_type=final_answer_type,
            plan=(
                f"slot_dependency_{direction}_"
                f"{len(operators)}hop_per_branch"
            ),
            anchor_entity=" ; ".join(
                str(result["root"].get("surface") or "")
                for result in branch_results
            ),
            bridge_entity=str(winner["terminal_surface"]),
            path_items=all_path_items,
            path_roles=[
                str(operator.get("operator_id") or "comparison_lookup")
                for _branch in branch_results
                for operator in operators
            ],
            anchor_score=min(
                float(
                    result["root"].get("binding_confidence", 1.0) or 1.0
                )
                for result in branch_results
            ),
            role_score=1.0,
            rank_map=rank_map,
            get_doc_content=get_doc_content,
            enable_surface_materializer=False,
        )
        if row is not None:
            terminal_operator = dict(plan.get("terminal_operator") or {})
            terminal_predicate_families = sorted({
                str(family)
                for item in all_path_items
                for family in (item.get("predicate_families") or [])
                if str(family)
            })
            row.update({
                "terminal_contract_version": str(plan.get("version") or ""),
                "query_plan_version": str(plan.get("version") or ""),
                "logical_plan_revision": str(
                    plan.get("logical_plan_revision") or ""
                ),
                "query_plan_kind": plan_kind,
                "validation_tier": "exact_branch_temporal_comparison_v6",
                "shadow_only": False,
                "production_eligible": True,
                "base_validation_passed": True,
                "slot_plan_stage": "final_answer",
                "support_only": False,
                "answer_candidate_eligible": True,
                "final_answer_type": final_answer_type,
                "final_predicate_match": True,
                "bridge_predicate_match": True,
                "final_answer_type_match": True,
                "bridge_eligible": False,
                "rooted_lineage_complete": True,
                "operator_complete": True,
                "slot_dependency_validated": bool(
                    not role_direction_required or role_direction_validated
                ),
                "role_direction_required": role_direction_required,
                "role_direction_validated": role_direction_validated,
                "role_direction_matches": role_direction_matches,
                "exact_join_key_count": max(2, len(all_path_items)),
                "lineage_depth": len(all_path_items),
                "plan_stage_count": len(operators),
                "comparison_operator": direction,
                "terminal_operator_id": str(
                    terminal_operator.get("operator_id") or "terminal"
                ),
                "terminal_operator_type": str(
                    terminal_operator.get("operator_type")
                    or direction.replace("_root", "")
                ),
                "terminal_input_slot": str(
                    terminal_operator.get("input_slot") or "$comparison_value"
                ),
                "terminal_output_slot": "$answer",
                "terminal_reduction_executed": True,
                "terminal_provenance_mode": "union_input_lineage",
                "terminal_input_provenance_doc_ids": list(
                    row.get("source_doc_ids") or []
                ),
                "candidate_predicate_families": (
                    terminal_predicate_families
                    or list(terminal_operator.get("predicate_families") or [])
                ),
                "comparison_values": {
                    str(result["root"].get("surface") or ""): str(
                        result["terminal_surface"]
                    )
                    for result in branch_results
                },
            })
            rows.append(row)
        return rows, rejects, execution_trace

    if plan_kind != "branch_compare" or not operators:
        rejects["unsupported_slot_dependency_plan"] += 1
        return rows, rejects, execution_trace

    lookup_operator = operators[0]
    explicit_roots = _bind_explicit_query_entities(
        question,
        relation_index,
        operator=lookup_operator,
        expected_count=2,
        max_bindings=2,
        canonicalize_surface=False,
    )
    execution_trace["root_bindings"] = [dict(root) for root in explicit_roots]
    roots = {
        str(binding.get("key") or ""): str(binding.get("surface") or "")
        for binding in explicit_roots
        if binding.get("key")
    }
    if len(roots) != 2:
        rejects["comparison_requires_exactly_two_roots"] += 1
        return rows, rejects, execution_trace

    def reachable_values(root_key: str) -> Dict[str, Tuple[str, List[Dict[str, Any]]]]:
        found: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}
        frontier: List[Tuple[str, List[Dict[str, Any]]]] = [(root_key, [])]
        visited = {root_key}
        for _depth in range(2):
            next_frontier: List[Tuple[str, List[Dict[str, Any]]]] = []
            for bound_key, path in frontier:
                candidates = _lookup_items_by_exact_entity_key(
                    relation_index,
                    bound_key,
                    max_per_key=max_facts_per_key,
                    max_fanout=max_key_fanout,
                    access_stats=access_stats,
                    allow_title_provenance_alias=False,
                )
                for item in candidates:
                    if not _plan_operator_matches_item(item, lookup_operator):
                        continue
                    oriented_output = _oriented_endpoint_for_operator_binding(
                        item,
                        lookup_operator,
                        bound_key,
                        equivalent_keys=_equivalent_entity_keys(
                            relation_index,
                            bound_key,
                            allow_title_provenance_alias=False,
                        ),
                    )
                    if oriented_output is None:
                        continue
                    surface = str(oriented_output.get("surface") or "")
                    output_key = str(oriented_output.get("key") or "")
                    if not output_key or output_key in visited:
                        continue
                    visited.add(output_key)
                    output_path = path + [item]
                    found.setdefault(output_key, (surface, output_path))
                    next_frontier.append((output_key, output_path))
            frontier = next_frontier
        return found

    root_pairs = list(roots.items())
    left_values = reachable_values(root_pairs[0][0])
    right_values = reachable_values(root_pairs[1][0])
    common_keys = set(left_values) & set(right_values)
    common_keys -= set(roots)
    if not common_keys:

        rejects["comparison_no_provable_common_value"] += 1
        return rows, rejects, execution_trace
    common_key = min(
        common_keys,
        key=lambda key: (
            len(left_values[key][1]) + len(right_values[key][1]),
            key,
        ),
    )
    common_surface = left_values[common_key][0] or right_values[common_key][0]
    path_items = left_values[common_key][1] + right_values[common_key][1]
    row = _build_multi_hop_row(
        candidate="yes",
        answer_type="boolean",
        plan="slot_dependency_compare_eq_2hop",
        anchor_entity=" ; ".join(surface for _, surface in root_pairs),
        bridge_entity=common_surface,
        path_items=path_items,
        path_roles=["branch_lookup"] * len(path_items),
        anchor_score=1.0,
        role_score=1.0,
        rank_map=rank_map,
        get_doc_content=get_doc_content,
        enable_surface_materializer=False,
    )
    if row is not None:
        row.update({
            "terminal_contract_version": str(plan.get("version") or ""),
            "query_plan_version": str(plan.get("version") or ""),
            "query_plan_kind": plan_kind,
            "validation_tier": "exact_branch_comparison",
            "shadow_only": False,
            "production_eligible": True,
            "base_validation_passed": True,
            "slot_plan_stage": "support_only",
            "support_only": True,
            "answer_candidate_eligible": False,
            "final_answer_type": "boolean",
            "final_predicate_match": True,
            "bridge_predicate_match": True,
            "final_answer_type_match": True,
            "bridge_eligible": False,
            "rooted_lineage_complete": True,
            "operator_complete": True,
            "slot_dependency_validated": True,
            "exact_join_key_count": max(2, len(path_items)),
            "lineage_depth": len(path_items),
            "plan_stage_count": 1,
            "comparison_operator": "eq",
            "comparison_value": common_surface,
            "branch_bindings": {
                root_key: {
                    "root_surface": root_surface,
                    "comparison_key": common_key,
                    "comparison_surface": common_surface,
                }
                for root_key, root_surface in root_pairs
            },
        })
        rows.append(row)
    execution_trace["completion_reject_reason_counts"] = dict(
        completion_rejects
    )
    execution_trace["completion_extraction_diagnostics"] = dict(
        completion_extraction_diagnostics
    )
    return rows, rejects, execution_trace


def execute_instance_bound_fact_join_plan(
    *,
    question: str,
    relation_index: Mapping[str, Any],
    rank_by_doc: Mapping[int, int],
    get_doc_content: Callable[[int], str],
    max_facts_per_key: int = 24,
    max_key_fanout: int = 256,
    max_candidates: int = 32,
    enable_bound_completion: bool = False,
    bound_completion_lookup: Optional[
        Callable[[str, int], List[Dict[str, Any]]]
    ] = None,
    bound_completion_top_k: int = 6,
    bound_completion_max_calls: int = 2,
    enable_source_grounded_terminal_repair: bool = False,
) -> Dict[str, Any]:
    """Execute a complete instance-bound path plan over the fact indexes.

    Unlike a terminal-predicate probe, every operator consumes only bindings
    emitted by its direct predecessor.  A candidate is production eligible
    only when the explicit query root, every intermediate slot, the terminal
    value, and every source passage form one complete provenance chain.
    """
    plan = compile_stage_aware_fact_join_plan(
        question,
        enable_bound_completion=True,
        enable_role_directed_join=True,
    )
    operators = list(plan.get("operators") or [])
    info: Dict[str, Any] = {
        "enabled": True,
        "executed": False,
        "reason": "unsupported_logical_plan",
        "plan_kind": str(plan.get("plan_kind") or "unsupported"),
        "compile_status": str(plan.get("compile_status") or "unsupported"),
        "compile_reason": str(plan.get("compile_reason") or ""),
        "operator_count": len(operators),
        "plan_coverage_status": str(
            (plan.get("plan_coverage") or {}).get("status") or ""
        ),
        "plan_coverage": dict(plan.get("plan_coverage") or {}),
        "root_binding_count": 0,
        "complete_lineage_count": 0,
        "candidate_count": 0,
        "reject_reason_counts": {},
        "index_access": {},
        "execution_trace": {},
        "candidates": [],
    }
    if (
        plan.get("compile_status") != "compiled"
        or plan.get("plan_kind") != "path_projection"
        or len(operators) < 2
        or str(plan.get("answer_operator") or "") != "project"
    ):
        return info

    if str((plan.get("plan_coverage") or {}).get("status") or "") != PLAN_COMPLETE:
        coverage = dict(plan.get("plan_coverage") or {})
        if str(coverage.get("status") or "") == PLAN_PARTIAL:


            info["reason"] = "uncompiled_question_relation"
            obligation_by_id = {
                str(row.get("obligation_id") or ""): dict(row)
                for row in plan.get("question_obligations") or []
            }
            info["uncovered_relation_cues"] = sorted({
                str(obligation_by_id.get(str(obligation_id), {}).get(
                    "semantic_label"
                ) or obligation_id)
                for obligation_id in coverage.get("missing_obligation_ids") or []
                if str(obligation_id)
            })
        else:
            info["reason"] = "question_plan_coverage_unknown"
        return info




    if not re.match(
        r"^\s*(?:who|whom|whose|what|which|where|when|in\s+what|on\s+what)\b",
        str(question or ""),
        flags=re.IGNORECASE,
    ):
        info["reason"] = "unsupported_answer_operator"
        return info
    terminal_operator = operators[-1]
    if (
        str(terminal_operator.get("slot_plan_stage") or "") != "final_answer"
        or str(terminal_operator.get("output_slot") or "") != "$answer"
        or str(plan.get("final_answer_type") or "") == "boolean"
    ):
        info["reason"] = "terminal_projection_contract_unavailable"
        return info
    access_stats = _new_index_access_stats()
    rows, rejects, execution_trace = _execute_slot_dependency_query_plan(
        question=question,
        plan=plan,
        relation_index=dict(relation_index or {}),
        anchor_items=[],
        rank_map={int(key): int(value) for key, value in rank_by_doc.items()},
        get_doc_content=get_doc_content,
        max_facts_per_key=max(1, int(max_facts_per_key)),
        max_key_fanout=max(1, int(max_key_fanout)),
        access_stats=access_stats,
        enable_surface_materializer=False,
        enable_bound_completion=bool(enable_bound_completion),
        bound_completion_lookup=bound_completion_lookup,
        bound_completion_top_k=max(1, int(bound_completion_top_k)),
        bound_completion_max_calls=max(0, int(bound_completion_max_calls)),
        enable_bound_completion_cost_admission=True,
        enable_canonical_bound_completion=True,
        enable_completion_selectivity_feedback=True,
        enable_query_plan_root_contract=True,
        enable_inverse_relation_completion=True,
        enable_title_grounded_completion=bool(enable_bound_completion),
        enable_source_grounded_terminal_repair=bool(
            enable_source_grounded_terminal_repair
        ),
        enable_title_catalog_root_plan=True,
        commit_title_root_plan=True,
        enable_identity_safe_root_commit=True,
    )
    info["executed"] = True
    info["root_binding_count"] = len(
        execution_trace.get("root_bindings") or []
    )
    candidate_rejects = Counter(rejects)
    info["index_access"] = dict(access_stats)
    info["execution_trace"] = execution_trace

    total_passages = max(1, len(rank_by_doc))
    terminal_id = str(terminal_operator.get("operator_id") or "terminal")
    terminal_slot_id = f"terminal:{terminal_id}"
    terminal_families = set(
        terminal_operator.get("predicate_families") or []
    )
    candidates: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, Tuple[int, ...]]] = set()
    for row in rows:
        path_triples = [
            list(triple)
            for triple in row.get("path_triples") or []
            if isinstance(triple, (list, tuple)) and len(triple) == 3
        ]
        path_doc_ids = [
            int(value) for value in row.get("path_doc_ids") or []
        ]
        lineage_bound_keys = [
            str(value or "")
            for value in row.get("lineage_bound_keys") or []
            if str(value or "")
        ]
        strict_contract_valid, strict_contract_reason = (
            _strict_instance_operator_contracts_match(
                operators=operators,
                path_triples=path_triples,
                slot_bindings=dict(row.get("slot_bindings") or {}),
            )
        )
        complete = bool(
            row.get("production_eligible")
            and row.get("operator_complete")
            and row.get("rooted_lineage_complete")
            and row.get("slot_dependency_validated")
            and row.get("final_predicate_match")
            and row.get("final_answer_type_match")
            and len(path_triples) == len(operators)
            and len(path_doc_ids) == len(operators)
            and len(lineage_bound_keys) == len(operators) + 1
            and all(doc_id >= 0 for doc_id in path_doc_ids)
            and strict_contract_valid
        )
        if not complete:
            if not strict_contract_valid:
                candidate_rejects[strict_contract_reason] += 1
            continue
        answer_value = str(row.get("candidate") or "").strip()
        if not answer_value:
            continue
        signature = (canonical_entity_key(answer_value), tuple(path_doc_ids))
        if signature in seen:
            continue
        seen.add(signature)

        terminal_triple = path_triples[-1]
        terminal_predicate = str(terminal_triple[1] or "")
        candidate_families = set(
            canonical_schema_predicate_families(terminal_predicate)
        )
        if terminal_families and not candidate_families & terminal_families:
            continue
        schemas = list(typed_schema_signatures_v2(tuple(terminal_triple)))
        terminal_schema = schemas[0] if schemas else (
            "entity",
            next(iter(sorted(terminal_families)), "relation"),
            str(plan.get("final_answer_type") or "entity"),
            "subject_to_object",
        )
        terminal_input = dict(
            (row.get("slot_bindings") or {}).get(
                str(terminal_operator.get("input_slot") or "")
            )
            or {}
        )
        terminal_input_key = str(terminal_input.get("key") or "")
        terminal_family = next(
            iter(sorted(candidate_families & terminal_families)),
            next(iter(sorted(candidate_families)), ""),
        )
        subject_family_index = (
            relation_index.get("subject_predicate_family_to_items") or {}
        )
        object_family_index = (
            relation_index.get("object_predicate_family_to_items") or {}
        )
        composite_key = entity_predicate_index_key(
            terminal_input_key,
            terminal_family,
        )
        endpoint_scan_rows = len(
            set(subject_family_index.get(composite_key) or [])
            | set(object_family_index.get(composite_key) or [])
        )
        endpoint_scan_rows = max(1, endpoint_scan_rows)
        schema_idf = math.log(
            (1.0 + total_passages) / (1.0 + endpoint_scan_rows)
        ) / max(math.log(1.0 + total_passages), 1e-9)

        lineage = []
        projection_lines = [
            "[Instance-bound Schema Memory fact join]",
            f"Root binding: {row.get('anchor_entity', '')}",
        ]
        for index, (operator, triple, doc_id) in enumerate(
            zip(operators, path_triples, path_doc_ids)
        ):
            tuple_key = str(
                (row.get("path_tuple_keys") or [""] * len(path_triples))[index]
                if index < len(row.get("path_tuple_keys") or []) else ""
            )
            lineage_row = {
                "operator_id": str(operator.get("operator_id") or f"op{index}"),
                "stage_index": index,
                "input_slot": str(operator.get("input_slot") or ""),
                "output_slot": str(operator.get("output_slot") or ""),
                "tuple_key": tuple_key,
                "triple": list(triple),
                "source_doc_id": int(doc_id),
                "input_binding_key": lineage_bound_keys[index],
                "output_binding_key": lineage_bound_keys[index + 1],
            }
            lineage.append(lineage_row)
            projection_lines.append(
                f"{lineage_row['operator_id']}: ({triple[0]}, {triple[1]}, "
                f"{triple[2]}) [source_doc={doc_id}]"
            )
        projection_lines.append(f"Answer projection: {answer_value}")
        answer_doc_id = int(path_doc_ids[-1])
        answer_rank = int(rank_by_doc.get(answer_doc_id, total_passages))
        score = min(
            0.99,
            0.78
            + 0.08 * float(row.get("root_binding_confidence", 1.0) or 1.0)
            + 0.07 * min(1.0, len(path_triples) / 3.0)
            + 0.04 * schema_idf,
        )
        candidates.append({
            "doc_id": answer_doc_id,
            "title": str((row.get("source_titles") or [""])[-1]),
            "score": round(score, 6),
            "original_rank": answer_rank,
            "schema": list(terminal_schema),
            "schema_frequency": endpoint_scan_rows,
            "schema_idf": round(schema_idf, 6),
            "schema_support": 1.0,
            "context_overlap": 1.0,
            "query_overlap": 1.0,
            "relation_overlap": 1.0,
            "join_key_overlap": 1.0,
            "join_key_containment": 1.0,
            "join_key_exact": True,
            "entity_bound": True,
            "active_join_role": "bound_slot",
            "candidate_join_role": "bound_slot",
            "active_join_key_tokens": sorted(_tokens(terminal_input_key)),
            "candidate_join_key_tokens": sorted(_tokens(terminal_input_key)),
            "matched_join_tokens": sorted(_tokens(terminal_input_key)),
            "join_key_matched": True,
            "already_top5": answer_rank < 5,
            "schema_fact_tuple": True,
            "triple": list(terminal_triple),
            "predicate": terminal_predicate,
            "answer_value": answer_value,
            "answer_type": str(plan.get("final_answer_type") or "entity"),
            "tuple_selection_score": 1.0,
            "matched_endpoint_tokens": sorted(_tokens(terminal_input_key)),
            "candidate_join_endpoint_role": "bound_slot",
            "direction": "instance_bound_path",
            "materialization_role": "instance_bound_terminal_fact_join",
            "logical_plan_stage": "final_answer",
            "terminal_operator_id": terminal_id,
            "predicate_contract_match": True,
            "terminal_slot_match": True,
            "answer_type_match": True,
            "rooted_lineage_complete": True,
            "operator_complete": True,
            "strict_operator_contract_validated": True,
            "root_plan_variant": str(row.get("root_plan_variant") or ""),
            "root_binding_method": str(row.get("root_binding_method") or ""),
            "root_binding_confidence": float(
                row.get("root_binding_confidence", 0.0) or 0.0
            ),
            "root_identity_safe_for_commit": bool(
                row.get("root_identity_safe_for_commit", True)
            ),
            "bound_completion_used": bool(row.get("bound_completion_used")),
            "completion_causal_lineage_validated": bool(
                row.get("completion_causal_lineage_validated")
            ),
            "source_grounded_completion_used": bool(
                row.get("source_grounded_completion_used")
            ),
            "source_grounded_terminal_repair_used": bool(
                row.get("source_grounded_terminal_repair_used")
            ),
            "bound_completion_lineage": list(
                row.get("bound_completion_lineage") or []
            ),
            "bound_completion_doc_ids": list(
                row.get("bound_completion_doc_ids") or []
            ),
            "covered_slot_ids": [terminal_slot_id],
            "uncovered_slot_ids": [],
            "physical_access_path": (
                "instance_bound_multi_stage_entity_predicate_index"
            ),
            "estimated_endpoint_scan_rows": endpoint_scan_rows,
            "lineage_root_key": str(row.get("lineage_root_key") or ""),
            "lineage_bound_keys": lineage_bound_keys,
            "lineage_depth": len(lineage),
            "lineage": lineage,
            "lineage_source_doc_ids": path_doc_ids,
            "lineage_reader_text": "\n".join(projection_lines),
            "slot_bindings": dict(row.get("slot_bindings") or {}),
        })

    candidates.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            -int(item.get("estimated_endpoint_scan_rows", 1)),
            -int(item.get("original_rank", total_passages)),
        ),
        reverse=True,
    )
    candidates = candidates[: max(1, int(max_candidates))]
    info["complete_lineage_count"] = len(candidates)
    info["candidate_count"] = len(candidates)
    info["reject_reason_counts"] = dict(candidate_rejects)
    info["candidates"] = candidates
    info["reason"] = (
        "instance_bound_candidates_generated"
        if candidates else "no_complete_instance_bound_chain"
    )
    return info


def _surface_specificity_score(row: Dict[str, Any], question: str) -> float:
    candidate = str(row.get("candidate") or "").strip()
    if not candidate:
        return 0.0
    answer_type = str(row.get("answer_type") or "")
    tokens = _tokens(candidate)
    score = 0.45
    if row.get("surface_materialized"):
        score += 0.12
    if answer_type == "place":
        comma_parts = [
            part.strip()
            for part in candidate.split(",")
            if part.strip()
        ]
        if len(comma_parts) >= 2:
            score += 0.12
        if len(comma_parts) >= 4:
            score -= 0.08
        if re.search(r"\b(?:born|was|were|is|are|he|she|they)\b", candidate, re.IGNORECASE):
            score -= 0.35
        q = str(question or "").lower()
        if re.search(r"\bcity\b", q) and len(comma_parts) > 1:
            score -= 0.06
        if re.search(r"\b(?:state|province|country|full|complete)\b", q) and len(comma_parts) > 1:
            score += 0.08
    elif answer_type == "date":
        if SURFACE_MONTH_RE.search(candidate):
            score += 0.14
        if DATE_VALUE_RE.search(candidate):
            score += 0.08
    if len(tokens) > 9:
        score -= 0.12
    return _clamp(score)


def _relation_support_for_row(
    row: Dict[str, Any],
    relation_index: Dict[str, Any],
    *,
    max_per_key: int,
    max_fanout: int,
) -> Dict[str, Any]:
    bridge_tokens = _tokens(row.get("bridge_entity") or row.get("bridge_key") or "")
    answer_tokens = _tokens(row.get("raw_candidate") or row.get("candidate") or "")
    plan = str(row.get("template") or "")




    terminal_support_doc_ids = list(
        row.get("terminal_support_doc_ids") or []
    )
    if plan.startswith("slot_dependency_path_"):
        if not terminal_support_doc_ids and row.get("answer_doc_id") is not None:
            terminal_support_doc_ids = [row.get("answer_doc_id")]
        support_doc_ids: Set[int] = {
            int(doc_id)
            for doc_id in terminal_support_doc_ids
            if doc_id is not None and int(doc_id) >= 0
        }
    else:
        support_doc_ids = set(_row_source_doc_ids(row))
    support_triples: List[str] = [
        triple.strip()
        for triple in str(row.get("triple") or "").split(" -> ")
        if triple.strip()
    ]
    conflict_answer_keys: Set[str] = set()
    scanned = 0
    if not bridge_tokens or not answer_tokens:
        return {
            "support_doc_count": len(support_doc_ids),
            "support_triple_count": len(set(support_triples)),
            "conflict_candidate_count": 0,
            "support_doc_ids": sorted(support_doc_ids)[:8],
            "support_triples": list(dict.fromkeys(support_triples))[:5],
        }
    candidates = _lookup_items_by_entity_tokens(
        relation_index,
        bridge_tokens,
        max_per_key=max_per_key,
        max_fanout=max_fanout,
    )
    for item in candidates:
        scanned += 1
        subj_tokens = set(item.get("subject_tokens") or set())
        obj_tokens = set(item.get("object_tokens") or set())
        rel_tokens = set(item.get("relation_tokens") or set())
        relation_ok = False
        bridge_ok = False
        answer_ok = False
        conflict_key = ""
        if plan in {"creator_birth_place_2hop", "creator_birth_date_2hop"}:
            relation_ok = bool(rel_tokens & (BIRTH_PLACE_RELATION_TERMS | BIRTH_DATE_RELATION_TERMS))
            bridge_ok = _entity_bridge_score(subj_tokens, bridge_tokens, max_required=4) >= 0.67
            answer_ok = _entity_bridge_score(obj_tokens, answer_tokens, max_required=4) >= 0.67
            conflict_key = _entity_key(item.get("object"))
        elif plan == "kinship_grandparent_2hop":
            orient = _parent_edge_orientation(item)
            if orient is not None:
                parent_side, child_side, _ = orient
                relation_ok = True
                bridge_ok = _entity_bridge_score(_edge_side_tokens(item, child_side), bridge_tokens, max_required=4) >= 0.67
                answer_ok = _entity_bridge_score(_edge_side_tokens(item, parent_side), answer_tokens, max_required=4) >= 0.67
                conflict_key = _entity_key(_edge_side_text(item, parent_side))
        elif plan.startswith("slot_dependency_"):
            bridge_key = str(row.get("bridge_key") or "")
            answer_key = str(row.get("answer_key") or "")
            subject_key = str(item.get("subject_key") or "")
            object_key = str(item.get("object_key") or "")
            relation_ok = bool(
                bridge_key
                and answer_key
                and {bridge_key, answer_key} == {subject_key, object_key}
            )
            bridge_ok = relation_ok
            answer_ok = relation_ok
            conflict_key = (
                object_key if subject_key == bridge_key else subject_key
            )
        if not relation_ok or not bridge_ok:
            continue
        if conflict_key:
            conflict_answer_keys.add(conflict_key)
        if not answer_ok:
            continue
        doc_id = _fact_doc_id(item)
        if doc_id is not None:
            support_doc_ids.add(doc_id)
        triple = item.get("triple") or []
        if triple:
            support_triples.append(" | ".join(str(part) for part in triple))
    scan_support = {
        "support_doc_count": len(support_doc_ids),
        "support_triple_count": len(set(support_triples)),
        "conflict_candidate_count": len({key for key in conflict_answer_keys if key}),
        "support_doc_ids": sorted(support_doc_ids)[:8],
        "support_triples": list(dict.fromkeys(support_triples))[:5],
        "scanned_fact_count": scanned,
    }
    persistent_support = _persistent_fact_support_for_row(row, relation_index)
    if persistent_support.get("support_doc_count", 0) > scan_support["support_doc_count"]:
        merged_doc_ids = sorted(set(scan_support["support_doc_ids"]) | set(persistent_support.get("support_doc_ids", [])))[:8]
        merged_triples = list(dict.fromkeys(
            list(scan_support["support_triples"]) + list(persistent_support.get("support_triples", []))
        ))[:5]
        scan_support.update({
            "support_doc_count": int(persistent_support.get("support_doc_count", 0) or 0),
            "support_triple_count": max(
                int(scan_support.get("support_triple_count", 0) or 0),
                int(persistent_support.get("support_triple_count", 0) or 0),
            ),
            "conflict_candidate_count": max(
                int(scan_support.get("conflict_candidate_count", 0) or 0),
                int(persistent_support.get("conflict_candidate_count", 0) or 0),
            ),
            "support_doc_ids": merged_doc_ids,
            "support_triples": merged_triples,
            "persistent_slot_key": persistent_support.get("slot_key", ""),
            "persistent_answer_key": persistent_support.get("answer_key", ""),
            "persistent_slot_doc_count": persistent_support.get("slot_doc_count", 0),
        })
    elif persistent_support:
        scan_support["conflict_candidate_count"] = max(
            int(scan_support.get("conflict_candidate_count", 0) or 0),
            int(persistent_support.get("conflict_candidate_count", 0) or 0),
        )
        scan_support["persistent_slot_key"] = persistent_support.get("slot_key", "")
        scan_support["persistent_answer_key"] = persistent_support.get("answer_key", "")
        scan_support["persistent_slot_doc_count"] = persistent_support.get("slot_doc_count", 0)
    return scan_support


def _persistent_fact_support_for_row(row: Dict[str, Any], relation_index: Dict[str, Any]) -> Dict[str, Any]:
    reliability = (relation_index or {}).get("fact_reliability") or {}
    slot_answer_stats = reliability.get("slot_answer_stats") or {}
    slot_conflict_counts = reliability.get("slot_conflict_counts") or {}
    slot_doc_counts = reliability.get("slot_doc_counts") or {}
    bridge_key = str(row.get("bridge_key") or row.get("anchor_key") or "").strip()
    answer_key = str(row.get("answer_key") or _row_answer_key(row) or "").strip()
    best: Dict[str, Any] = {}
    if not bridge_key or not answer_key:
        return best
    for label in _row_fact_slot_labels(row):
        slot_key = _fact_slot_key(label, bridge_key)
        slot_answer_key = _fact_slot_answer_key(label, bridge_key, answer_key)
        stats = slot_answer_stats.get(slot_answer_key) or {}
        if not stats:
            continue
        candidate = {
            "slot_key": slot_key,
            "answer_key": answer_key,
            "label": label,
            "support_doc_count": int(stats.get("doc_count", 0) or 0),
            "support_triple_count": int(stats.get("triple_count", 0) or 0),
            "conflict_candidate_count": int(
                stats.get("conflict_candidate_count", slot_conflict_counts.get(slot_key, 0)) or 0
            ),
            "slot_doc_count": int(stats.get("slot_doc_count", slot_doc_counts.get(slot_key, 0)) or 0),
            "support_doc_ids": list(stats.get("doc_ids") or [])[:8],
            "support_triples": list(stats.get("support_triples") or [])[:5],
            "tuple_keys": list(stats.get("tuple_keys") or [])[:5],
        }
        if (
            not best
            or candidate["support_doc_count"] > int(best.get("support_doc_count", 0) or 0)
            or (
                candidate["support_doc_count"] == int(best.get("support_doc_count", 0) or 0)
                and candidate["support_triple_count"] > int(best.get("support_triple_count", 0) or 0)
            )
        ):
            best = candidate
    return best


def _fact_reliability_for_row(
    row: Dict[str, Any],
    support: Dict[str, Any],
    *,
    question: str,
    base_score: float,
    min_support_docs: int,
    min_reliability: float,
) -> Dict[str, Any]:
    support_doc_count = int(support.get("support_doc_count", 0) or 0)
    support_triple_count = int(support.get("support_triple_count", 0) or 0)
    conflict_candidate_count = int(support.get("conflict_candidate_count", 0) or 0)
    surface_score = _surface_specificity_score(row, question)
    support_signal = _clamp(float(support_doc_count) / max(1.0, float(min_support_docs)))
    triple_signal = _clamp(float(support_triple_count) / max(2.0, float(min_support_docs + 1)))
    conflict_signal = 1.0 / (1.0 + math.log1p(max(0, conflict_candidate_count - 1)))
    base_signal = _clamp(float(base_score))
    path_signal = 1.0 if str(row.get("template") or "").endswith("_2hop") else 0.65
    score = _clamp(
        0.34 * support_signal
        + 0.20 * triple_signal
        + 0.17 * conflict_signal
        + 0.12 * surface_score
        + 0.10 * base_signal
        + 0.07 * path_signal
    )
    reject_reasons: List[str] = []
    if support_doc_count < max(1, int(min_support_docs)):
        reject_reasons.append("support_doc_count_below_min")
    if score < float(min_reliability):
        reject_reasons.append("reliability_below_min")
    if conflict_candidate_count >= 4 and support_doc_count <= 1:
        reject_reasons.append("high_conflict_low_support")
    passed = not reject_reasons
    return {
        "score": round(float(score), 6),
        "passed": bool(passed),
        "reason": "passed" if passed else "+".join(reject_reasons),
        "support_doc_count": support_doc_count,
        "support_triple_count": support_triple_count,
        "conflict_candidate_count": conflict_candidate_count,
        "slot_doc_count": int(support.get("persistent_slot_doc_count", 0) or 0),
        "slot_key": support.get("persistent_slot_key", ""),
        "answer_key": support.get("persistent_answer_key", _row_answer_key(row)),
        "support_doc_ids": support.get("support_doc_ids", []),
        "support_triples": support.get("support_triples", []),
        "surface_specificity": round(float(surface_score), 6),
        "min_support_docs": int(min_support_docs),
        "min_reliability": round(float(min_reliability), 6),
    }


def _provenance_witness_query(
    row: Dict[str, Any],
    operator: Dict[str, Any],
) -> str:
    """Build a tuple-conditioned query for an independent terminal witness."""
    bridge = str(row.get("bridge_entity") or "").strip()
    answer = str(row.get("raw_candidate") or row.get("candidate") or "").strip()
    predicates = list(operator.get("relation_tokens") or [])
    if not predicates:
        predicates = list(operator.get("predicate_families") or [])
    return " ".join(
        part
        for part in (
            bridge,
            " ".join(sorted(set(predicates))),
            answer,
        )
        if part
    )


def _completion_item_matches_projected_tuple(
    item: Dict[str, Any],
    *,
    bridge_key: str,
    answer_key: str,
) -> bool:
    if not bridge_key or not answer_key:
        return False
    subject_key = str(item.get("subject_key") or "")
    object_key = str(item.get("object_key") or "")
    if subject_key != bridge_key:
        return False
    if object_key == answer_key:
        return True



    object_tokens = set(object_key.split())
    answer_tokens = set(answer_key.split())
    return bool(
        object_tokens
        and answer_tokens
        and (object_tokens <= answer_tokens or answer_tokens <= object_tokens)
        and len(object_tokens & answer_tokens) >= 1
    )


def _repair_provenance_witnesses(
    rows: List[Dict[str, Any]],
    *,
    question: str,
    plan: Optional[Dict[str, Any]],
    relation_index: Dict[str, Any],
    lookup: Optional[Callable[[str, int], List[Dict[str, Any]]]],
    enabled: bool,
    top_k: int,
    max_calls: int,
    min_support_docs: int,
    enable_cost_admission: bool,
    max_risk: float,
    min_net_utility: float,
    compute_weight: float,
    risk_weight: float,
    max_facts_per_key: int,
    max_key_fanout: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Repair under-supported answer tuples with an independent witness.

    The operator is invoked only for a complete typed path whose projected
    terminal tuple fails the provenance support requirement.  Retrieved text
    is never admitted directly: it must be parsed into the same canonical
    predicate family and bind the exact ``bridge -> answer`` tuple.
    """
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "input_row_count": len(rows or []),
        "eligible_row_count": 0,
        "call_count": 0,
        "doc_count": 0,
        "extracted_fact_count": 0,
        "tuple_match_count": 0,
        "independent_witness_count": 0,
        "repaired_row_count": 0,
        "reason_counts": {},
        "extraction_diagnostics": {},
        "decisions": [],
    }
    if not enabled or not rows:
        return list(rows or []), info
    if lookup is None:
        info["reason_counts"] = {"missing_witness_lookup": len(rows)}
        return list(rows), info
    operators = list((plan or {}).get("operators") or [])
    if not operators:
        info["reason_counts"] = {"missing_terminal_operator": len(rows)}
        return list(rows), info
    terminal_operator = dict(operators[-1])
    reasons: Counter = Counter()
    extraction_diagnostics: Counter = Counter()
    repaired_rows: List[Dict[str, Any]] = []

    for row_index, source_row in enumerate(rows):
        row = dict(source_row)
        template = str(row.get("template") or "")
        if (
            not template.startswith("slot_dependency_")
            or not row.get("rooted_lineage_complete")
            or not row.get("operator_complete")
            or row.get("support_only")
        ):
            reasons["not_terminal_tuple"] += 1
            repaired_rows.append(row)
            continue
        support = _relation_support_for_row(
            row,
            relation_index,
            max_per_key=max_facts_per_key,
            max_fanout=max_key_fanout,
        )
        existing_doc_ids = {
            int(doc_id)
            for doc_id in support.get("support_doc_ids") or []
            if doc_id is not None
        }
        if len(existing_doc_ids) >= max(1, int(min_support_docs)):
            reasons["already_sufficient"] += 1
            repaired_rows.append(row)
            continue
        bridge_surface = str(row.get("bridge_entity") or "").strip()
        bridge_key = str(row.get("bridge_key") or _entity_key(bridge_surface))
        answer_surface = str(
            row.get("raw_candidate") or row.get("candidate") or ""
        ).strip()
        answer_key = str(row.get("answer_key") or _entity_key(answer_surface))
        if not bridge_surface or not bridge_key or not answer_surface or not answer_key:
            reasons["missing_tuple_binding"] += 1
            repaired_rows.append(row)
            continue
        info["eligible_row_count"] += 1
        decision = _bound_completion_admission(
            bound_surface=bridge_surface,
            operator=terminal_operator,
            call_count=int(info["call_count"]),
            max_calls=int(max_calls),
            top_k=int(top_k),
            max_risk=float(max_risk),
            min_net_utility=float(min_net_utility),
            compute_weight=float(compute_weight),
            risk_weight=float(risk_weight),
        )
        if not enable_cost_admission:
            decision["accepted"] = int(info["call_count"]) < max(0, int(max_calls))
            decision["reason"] = (
                "cost_admission_disabled"
                if decision["accepted"]
                else "witness_call_budget_exhausted"
            )
        witness_query = _provenance_witness_query(row, terminal_operator)
        decision.update({
            "operator_type": "provenance_witness_lookup",
            "row_index": int(row_index),
            "query": witness_query,
            "bridge": bridge_surface,
            "answer": answer_surface,
            "support_docs_before": len(existing_doc_ids),
        })
        if not decision.get("accepted"):
            reasons[str(decision.get("reason") or "witness_not_admitted")] += 1
            if len(info["decisions"]) < 16:
                info["decisions"].append(decision)
            repaired_rows.append(row)
            continue

        documents = list(lookup(witness_query, max(1, int(top_k))) or [])
        info["call_count"] += 1
        info["doc_count"] += len(documents)
        extracted = _extract_bound_completion_items(
            bound_surface=bridge_surface,
            bound_key=bridge_key,
            operator=terminal_operator,
            documents=documents,
            item_id_seed=-2000000 - 1000 * int(info["call_count"]),
            diagnostics=extraction_diagnostics,
            relation_index=relation_index,
        )
        info["extracted_fact_count"] += len(extracted)
        matching_items = [
            item for item in extracted
            if _completion_item_matches_projected_tuple(
                item,
                bridge_key=bridge_key,
                answer_key=answer_key,
            )
        ]
        info["tuple_match_count"] += len(matching_items)
        witnesses: List[Dict[str, Any]] = []
        for item in matching_items:
            doc_id = _fact_doc_id(item)
            if doc_id is None or int(doc_id) in existing_doc_ids:
                continue
            existing_doc_ids.add(int(doc_id))
            witnesses.append({
                "doc_id": int(doc_id),
                "triple": list(item.get("triple") or []),
                "sentence": str(item.get("completion_sentence") or "")[:320],
                "predicate_families": sorted(
                    item.get("predicate_families") or []
                ),
            })
            if len(existing_doc_ids) >= max(1, int(min_support_docs)):
                break
        info["independent_witness_count"] += len(witnesses)
        if witnesses:
            row["terminal_support_doc_ids"] = sorted(existing_doc_ids)
            row["source_doc_ids"] = list(dict.fromkeys(
                _row_source_doc_ids(row)
                + [int(witness["doc_id"]) for witness in witnesses]
            ))
            row["provenance_witnesses"] = list(
                row.get("provenance_witnesses") or []
            ) + witnesses
            row["provenance_repair_applied"] = True
            row["provenance_count"] = len(row["source_doc_ids"])
            witness_evidence = " || ".join(
                witness["sentence"] for witness in witnesses
                if witness.get("sentence")
            )
            if witness_evidence:
                row["evidence"] = " || ".join(
                    part for part in (
                        str(row.get("evidence") or ""), witness_evidence,
                    ) if part
                )[:1200]
        repaired = len(existing_doc_ids) >= max(1, int(min_support_docs))
        if repaired:
            info["repaired_row_count"] += 1
            reasons["repaired"] += 1
        else:
            reasons[
                "no_independent_operator_matching_witness"
                if matching_items else "no_operator_matching_witness"
            ] += 1
        decision.update({
            "documents": len(documents),
            "extracted_facts": len(extracted),
            "tuple_matches": len(matching_items),
            "new_witnesses": len(witnesses),
            "support_docs_after": len(existing_doc_ids),
            "repaired": bool(repaired),
        })
        if len(info["decisions"]) < 16:
            info["decisions"].append(decision)
        repaired_rows.append(row)

    info["reason_counts"] = dict(reasons)
    info["extraction_diagnostics"] = dict(extraction_diagnostics)
    return repaired_rows, info


def _row_slot_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("template") or "generic_fact_join"),
        str(row.get("anchor_key") or ""),
        str(row.get("bridge_key") or ""),
        str(row.get("answer_type") or ""),
    )


def _row_answer_key(row: Dict[str, Any]) -> str:
    return _entity_key(row.get("raw_candidate") or row.get("candidate") or "")


def _reader_projection_slot_key(row: Dict[str, Any]) -> str:
    template = str(row.get("template") or "")
    if not template.endswith("_2hop"):
        return ""
    return "|".join(_row_slot_key(row))


def _row_reader_projection_score(row: Dict[str, Any]) -> Tuple[float, float, int, float]:
    reliability = row.get("fact_reliability") or {}
    return (
        float(row.get("table_score", 0.0) or 0.0),
        float(reliability.get("score", 0.0) or 0.0),
        int(reliability.get("support_doc_count", 0) or 0),
        float(row.get("answer_rank_prior", row.get("rank_prior", 0.0)) or 0.0),
    )


def _project_reader_candidate_rows(
    rows: List[Dict[str, Any]],
    *,
    min_margin: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    info = {
        "enabled": True,
        "input_row_count": len(rows or []),
        "output_row_count": 0,
        "conflict_group_count": 0,
        "winner_group_count": 0,
        "suppressed_group_count": 0,
        "dropped_row_count": 0,
        "min_margin": round(float(min_margin), 6),
        "suppressed_all": False,
        "reason": "not_evaluated",
        "examples": [],
    }
    if not rows:
        info["reason"] = "no_rows"
        return [], info

    contract_safe_rows: List[Dict[str, Any]] = []
    contract_dropped = 0
    for row in rows:
        derived_path = bool(
            str(row.get("template") or "").endswith("_2hop")
            or row.get("rooted_lineage_complete")
        )
        if derived_path and row.get("answer_candidate_eligible") is False and not (
            row.get("support_only") and row.get("operator_complete")
        ):
            contract_dropped += 1
            continue
        contract_safe_rows.append(row)
    if not contract_safe_rows:
        info.update({
            "reason": "answer_contract_rejected_all",
            "dropped_row_count": contract_dropped,
            "contract_dropped_row_count": contract_dropped,
        })
        return [], info
    info["contract_dropped_row_count"] = contract_dropped

    passthrough: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in contract_safe_rows:
        key = _reader_projection_slot_key(row)
        if not key:
            passthrough.append(row)
        else:
            grouped[key].append(row)

    projected: List[Dict[str, Any]] = list(passthrough)
    suppressed_groups = 0
    winner_groups = 0
    conflict_groups = 0
    dropped_rows = 0
    examples: List[Dict[str, Any]] = []

    for slot_key, group_rows in grouped.items():
        answer_keys = {_row_answer_key(row) for row in group_rows if _row_answer_key(row)}
        if len(answer_keys) <= 1:
            projected.extend(group_rows)
            continue
        conflict_groups += 1
        ranked = sorted(group_rows, key=_row_reader_projection_score, reverse=True)
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        margin = (
            float(top.get("table_score", 0.0) or 0.0)
            - float(second.get("table_score", 0.0) or 0.0)
            if second is not None else 1.0
        )
        if margin >= float(min_margin):
            winner = dict(top)
            winner["reader_projection"] = {
                "reason": "conflict_winner_projected",
                "slot_key": slot_key,
                "margin": round(float(margin), 6),
                "dropped_conflict_rows": max(0, len(ranked) - 1),
            }
            projected.append(winner)
            winner_groups += 1
            dropped_rows += max(0, len(ranked) - 1)
            if len(examples) < 5:
                examples.append({
                    "slot_key": slot_key,
                    "reason": "conflict_winner_projected",
                    "margin": round(float(margin), 6),
                    "kept": str(winner.get("candidate", ""))[:120],
                    "dropped": [str(row.get("candidate", ""))[:120] for row in ranked[1:4]],
                })
        else:
            suppressed_groups += 1
            dropped_rows += len(ranked)
            if len(examples) < 5:
                examples.append({
                    "slot_key": slot_key,
                    "reason": "unresolved_conflict_suppressed",
                    "margin": round(float(margin), 6),
                    "candidates": [str(row.get("candidate", ""))[:120] for row in ranked[:4]],
                })

    if suppressed_groups:


        info.update({
            "output_row_count": 0,
            "conflict_group_count": conflict_groups,
            "winner_group_count": winner_groups,
            "suppressed_group_count": suppressed_groups,
            "dropped_row_count": len(rows),
            "suppressed_all": True,
            "reason": "unresolved_conflict_suppressed_all",
            "examples": examples,
        })
        return [], info

    projected.sort(key=lambda row: float(row.get("table_score", 0.0) or 0.0), reverse=True)
    info.update({
        "output_row_count": len(projected),
        "conflict_group_count": conflict_groups,
        "winner_group_count": winner_groups,
        "suppressed_group_count": suppressed_groups,
        "dropped_row_count": dropped_rows + contract_dropped,
        "reason": "projected" if conflict_groups else "no_conflict",
        "examples": examples,
    })
    return projected, info


def _optimize_provenance_conflicts(
    rows: List[Dict[str, Any]],
    *,
    question: str,
    relation_index: Dict[str, Any],
    enabled: bool,
    min_margin: float,
    min_support_docs: int,
    min_reliability: float,
    max_bonus: float,
    shadow_only: bool,
    max_facts_per_key: int,
    max_key_fanout: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    info = {
        "enabled": bool(enabled),
        "applied": False,
        "shadow_only": bool(shadow_only),
        "row_count": len(rows or []),
        "optimized_row_count": 0,
        "annotated_row_count": 0,
        "reliability_passed_count": 0,
        "active_row_count": 0,
        "conflict_group_count": 0,
        "ambiguous_group_count": 0,
        "min_support_docs": int(min_support_docs),
        "min_reliability": round(float(min_reliability), 6),
        "min_margin": round(float(min_margin), 6),
        "max_bonus": round(float(max_bonus), 6),
        "reliability_gate_reason_counts": {},
        "reason": "disabled" if not enabled else "not_evaluated",
    }
    if not enabled or not rows:
        return rows, info

    optimized: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    passthrough: List[Dict[str, Any]] = []
    for row in rows:
        template = str(row.get("template") or "")
        if not template.endswith("_2hop"):
            passthrough.append(row)
            continue
        groups[_row_slot_key(row)].append(row)

    conflict_groups = 0
    ambiguous_groups = 0
    annotated_rows = 0
    passed_rows = 0
    active_rows = 0
    gate_reason_counts = Counter()
    for slot_key, group_rows in groups.items():
        answer_keys = {_row_answer_key(row) for row in group_rows if _row_answer_key(row)}
        has_conflict = len(answer_keys) > 1
        if has_conflict:
            conflict_groups += 1
        scored_rows: List[Dict[str, Any]] = []
        for row in group_rows:
            support = _relation_support_for_row(
                row,
                relation_index,
                max_per_key=max_facts_per_key,
                max_fanout=max_key_fanout,
            )
            answer_rank_prior = float(row.get("answer_rank_prior", row.get("rank_prior", 0.0)) or 0.0)
            base_score = float(row.get("table_score", 0.0) or 0.0)
            reliability = _fact_reliability_for_row(
                row,
                support,
                question=question,
                base_score=base_score,
                min_support_docs=max(1, int(min_support_docs)),
                min_reliability=float(min_reliability),
            )
            new_row = dict(row)
            new_row["pre_optimizer_table_score"] = round(base_score, 6)
            new_row["table_score"] = round(base_score, 6)
            new_row["fact_reliability"] = reliability
            new_row["provenance_store"] = {
                "slot_key": "|".join(slot_key),
                "answer_key": _row_answer_key(row),
                "support_doc_count": support.get("support_doc_count", 0),
                "support_triple_count": support.get("support_triple_count", 0),
                "conflict_candidate_count": support.get("conflict_candidate_count", 0),
                "support_doc_ids": support.get("support_doc_ids", []),
                "support_triples": support.get("support_triples", []),
                "surface_specificity": reliability.get("surface_specificity", 0.0),
                "answer_rank_prior": round(float(answer_rank_prior), 6),
                "reliability_score": reliability.get("score", 0.0),
                "reliability_passed": reliability.get("passed", False),
                "reliability_reason": reliability.get("reason", "unknown"),
                "active": False,
                "score_bonus": 0.0,
                "optimized_score": round(base_score, 6),
            }
            annotated_rows += 1
            if reliability.get("passed"):
                passed_rows += 1
            gate_reason_counts.update([str(reliability.get("reason", "unknown"))])
            scored_rows.append(new_row)
        reliability_sorted = sorted(
            scored_rows,
            key=lambda row: (
                float((row.get("fact_reliability") or {}).get("score", 0.0) or 0.0),
                int((row.get("fact_reliability") or {}).get("support_doc_count", 0) or 0),
                float(row.get("pre_optimizer_table_score", row.get("table_score", 0.0)) or 0.0),
            ),
            reverse=True,
        )
        margin = 1.0
        if len(reliability_sorted) > 1:
            margin = (
                float((reliability_sorted[0].get("fact_reliability") or {}).get("score", 0.0) or 0.0)
                - float((reliability_sorted[1].get("fact_reliability") or {}).get("score", 0.0) or 0.0)
            )
            for row in scored_rows:
                row["provenance_store"]["top_margin"] = round(float(margin), 6)
                row["provenance_store"]["ambiguous_conflict"] = bool(has_conflict and margin < float(min_margin))
            if has_conflict and margin < float(min_margin):
                ambiguous_groups += 1
        top_answer_key = ""
        if reliability_sorted:
            top_answer_key = _row_answer_key(reliability_sorted[0])
        for row in scored_rows:
            reliability = row.get("fact_reliability") or {}
            active = (
                bool(reliability.get("passed"))
                and not bool(shadow_only)
                and (
                    not has_conflict
                    or (
                        _row_answer_key(row) == top_answer_key
                        and float(margin) >= float(min_margin)
                    )
                )
            )
            if active:
                bonus = min(float(max_bonus), float(max_bonus) * float(reliability.get("score", 0.0) or 0.0))
                row["table_score"] = round(
                    _clamp(float(row.get("pre_optimizer_table_score", row.get("table_score", 0.0)) or 0.0) + bonus),
                    6,
                )
                row["provenance_store"]["active"] = True
                row["provenance_store"]["score_bonus"] = round(float(bonus), 6)
                row["provenance_store"]["optimized_score"] = row["table_score"]
                active_rows += 1
        optimized.extend(scored_rows)

    final_rows = passthrough + optimized
    final_rows.sort(key=lambda row: float(row.get("table_score", 0.0) or 0.0), reverse=True)
    info.update({
        "applied": bool(active_rows),
        "optimized_row_count": len(optimized),
        "annotated_row_count": annotated_rows,
        "reliability_passed_count": passed_rows,
        "active_row_count": active_rows,
        "conflict_group_count": conflict_groups,
        "ambiguous_group_count": ambiguous_groups,
        "reliability_gate_reason_counts": dict(gate_reason_counts),
        "reason": (
            "shadow_annotated" if bool(shadow_only) and optimized
            else "active_optimized" if active_rows
            else "annotated_no_active_rows" if optimized
            else "no_optimizable_rows"
        ),
    })
    return final_rows, info


def _apply_reader_provenance_hard_gate(
    rows: List[Dict[str, Any]],
    *,
    enabled: bool,
    min_support_docs: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Keep derived join rows reader-visible only after lineage validation."""
    info = {
        "enabled": bool(enabled),
        "input_row_count": len(rows or []),
        "output_row_count": 0,
        "checked_row_count": 0,
        "passed_row_count": 0,
        "rejected_row_count": 0,
        "reason_counts": {},
    }
    if not enabled:
        info["output_row_count"] = len(rows or [])
        return list(rows or []), info

    output: List[Dict[str, Any]] = []
    reasons = Counter()
    for row in rows or []:
        template = str(row.get("template") or "")
        is_derived_path = bool(
            template.endswith("_2hop") or row.get("rooted_lineage_complete")
        )
        if not is_derived_path:
            output.append(row)
            continue
        info["checked_row_count"] += 1
        reliability = row.get("fact_reliability") or {}
        provenance = row.get("provenance_store") or {}
        rejection = ""
        causal_completion = bool(
            row.get("bound_completion_used")
            and row.get("completion_causal_lineage_validated")
        )
        completion_lineage = list(
            row.get("bound_completion_lineage") or []
        )
        path_doc_ids = {
            int(doc_id)
            for doc_id in row.get("path_doc_ids") or []
            if doc_id is not None and int(doc_id) >= 0
        }
        if not row.get("rooted_lineage_complete"):
            rejection = "incomplete_rooted_lineage"
        elif row.get("role_direction_required") and not row.get(
            "role_direction_validated"
        ):
            rejection = "role_direction_not_validated"
        elif template.startswith("slot_dependency_") and not row.get(
            "slot_dependency_validated"
        ):
            rejection = "slot_dependency_not_validated"
        elif row.get("support_only") and not row.get("operator_complete"):
            rejection = "support_operator_incomplete"
        elif causal_completion and (
            not completion_lineage
            or len(path_doc_ids) < 2
            or not all(
                lineage.get("completion_fact_id")
                and lineage.get("tuple_key")
                and int(lineage.get("source_doc_id", -1)) in path_doc_ids
                and lineage.get("instance_identity_validated")
                for lineage in completion_lineage
            )
        ):
            rejection = "causal_completion_direct_provenance_invalid"
        elif causal_completion:





            rejection = ""
        elif int(reliability.get("support_doc_count", 0) or 0) < max(
            1, int(min_support_docs)
        ):
            rejection = "insufficient_distinct_support_docs"
        elif not reliability.get("passed"):
            rejection = "fact_reliability_failed"
        elif not provenance.get("active"):
            rejection = "provenance_candidate_inactive"
        if rejection:
            reasons[rejection] += 1
            continue
        accepted = dict(row)
        accepted["reader_provenance_gate"] = {
            "passed": True,
            "causal_completion": causal_completion,
            "support_doc_count": int(
                reliability.get("support_doc_count", 0) or 0
            ),
            "reliability_score": float(reliability.get("score", 0.0) or 0.0),
        }
        output.append(accepted)
        info["passed_row_count"] += 1
    info["output_row_count"] = len(output)
    info["rejected_row_count"] = sum(reasons.values())
    info["reason_counts"] = dict(reasons)
    return output, info


def _apply_bound_completion_reader_gate(
    rows: List[Dict[str, Any]],
    *,
    materialize: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Gate only rows introduced by a bound completion operator.

    Existing fact-join rows are invariant across the control, shadow, and
    materialized arms.  A completed row becomes reader-visible only when its
    logical plan is complete and every added tuple retains direct document
    provenance from the indexed fact store.
    """
    info: Dict[str, Any] = {
        "enabled": True,
        "materialize": bool(materialize),
        "input_row_count": len(rows or []),
        "completion_row_count": 0,
        "passed_row_count": 0,
        "rejected_row_count": 0,
        "output_row_count": 0,
        "reason_counts": {},
    }
    output: List[Dict[str, Any]] = []
    reasons: Counter = Counter()
    for source_row in rows or []:
        if not source_row.get("bound_completion_used"):
            output.append(source_row)
            continue
        info["completion_row_count"] += 1
        rejection = ""
        completion_doc_ids = {
            int(doc_id)
            for doc_id in source_row.get("bound_completion_doc_ids") or []
            if doc_id is not None and int(doc_id) >= 0
        }
        path_doc_ids = {
            int(doc_id)
            for doc_id in source_row.get("path_doc_ids") or []
            if doc_id is not None and int(doc_id) >= 0
        }
        if not materialize:
            rejection = "completion_shadow_only"
        elif not source_row.get("rooted_lineage_complete"):
            rejection = "completion_incomplete_lineage"
        elif not source_row.get("operator_complete"):
            rejection = "completion_operator_incomplete"
        elif not source_row.get("completion_causal_lineage_validated"):
            rejection = "completion_causal_lineage_invalid"
        elif not source_row.get("slot_dependency_validated"):
            rejection = "completion_slot_dependency_invalid"
        elif source_row.get("role_direction_required") and not source_row.get(
            "role_direction_validated"
        ):
            rejection = "completion_role_direction_invalid"
        elif not source_row.get("final_predicate_match"):
            rejection = "completion_final_predicate_mismatch"
        elif not source_row.get("final_answer_type_match"):
            rejection = "completion_final_type_mismatch"
        elif not completion_doc_ids:
            rejection = "completion_missing_direct_provenance"
        elif len(path_doc_ids | completion_doc_ids) < 2:
            rejection = "completion_single_document_chain"
        if rejection:
            reasons[rejection] += 1
            continue
        accepted = dict(source_row)
        accepted["shadow_only"] = False
        accepted["bound_completion_reader_gate"] = {
            "passed": True,
            "completion_doc_ids": sorted(completion_doc_ids),
            "path_doc_count": len(path_doc_ids | completion_doc_ids),
        }
        output.append(accepted)
        info["passed_row_count"] += 1
    info["rejected_row_count"] = sum(reasons.values())
    info["output_row_count"] = len(output)
    info["reason_counts"] = dict(reasons)
    return output, info


def _semantic_plan_execution_summary(
    *,
    plan: Optional[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    execution_trace: Mapping[str, Any],
    rejects: Mapping[str, int],
) -> Dict[str, Any]:
    """Report logical completion only for fully bound terminal plan rows."""
    compiled_plan = dict(plan or {})
    operators = list(compiled_plan.get("operators") or [])
    plan_kind = str(compiled_plan.get("plan_kind") or "")
    terminal_rows = [
        row
        for row in rows
        if bool(row.get("operator_complete"))
        and bool(row.get("rooted_lineage_complete"))
        and bool(row.get("slot_dependency_validated"))
        and (
            not bool(row.get("role_direction_required"))
            or bool(row.get("role_direction_validated"))
        )
        and bool(row.get("final_predicate_match"))
        and bool(row.get("final_answer_type_match"))
    ]
    root_bindings = list(execution_trace.get("root_bindings") or [])
    expected_root_count = 2 if plan_kind.startswith("branch_") else 1
    exact_probe_count = int(execution_trace.get("exact_probe_count", 0) or 0)
    exact_probe_hit_count = int(
        execution_trace.get("exact_probe_hit_count", 0) or 0
    )
    completion_calls = int(
        execution_trace.get("completion_call_count", 0) or 0
    )
    completion_rows = sum(
        bool(row.get("completion_causal_lineage_validated"))
        for row in terminal_rows
    )
    raw_completion_rows = sum(
        bool(row.get("bound_completion_used")) for row in terminal_rows
    )
    residual_terminal_completion_rows = sum(
        bool(row.get("residual_terminal_completion_used"))
        for row in terminal_rows
    )
    completed = bool(terminal_rows)
    if completed:
        status = "terminal_binding_complete"
    elif len(root_bindings) < expected_root_count:
        status = "root_binding_incomplete"
    elif exact_probe_count and not exact_probe_hit_count and not completion_calls:
        status = "exact_predicate_index_miss"
    elif completion_calls:
        status = "completion_without_terminal_binding"
    else:
        status = "terminal_binding_missing"
    return {
        "required": bool(operators),
        "completed": completed,
        "status": status,
        "plan_kind": plan_kind,
        "required_operator_count": len(operators),
        "root_binding_count": len(root_bindings),
        "expected_root_binding_count": expected_root_count,
        "exact_probe_count": exact_probe_count,
        "exact_probe_hit_count": exact_probe_hit_count,
        "terminal_binding_count": len(terminal_rows),
        "completed_path_count": len(terminal_rows),
        "completion_call_count": completion_calls,
        "completion_terminal_row_count": raw_completion_rows,
        "completion_recovered_terminal_count": completion_rows,
        "residual_terminal_completion_count": (
            residual_terminal_completion_rows
        ),
        "reject_reason_counts": dict(rejects),
    }


def _execute_multi_hop_fact_join(
    *,
    question: str,
    gap: Dict[str, Any],
    relation_index: Dict[str, Any],
    rank_map: Dict[int, int],
    get_doc_content: Callable[[int], str],
    max_facts_per_key: int,
    max_key_fanout: int,
    max_rows: int,
    hard_anchor_min_overlap: float,
    multi_hop_min_bridge_overlap: float,
    enable_surface_materializer: bool,
    stage_aware_plan: Optional[Dict[str, Any]] = None,
    enable_bound_completion: bool = False,
    bound_completion_lookup: Optional[
        Callable[[str, int], List[Dict[str, Any]]]
    ] = None,
    bound_completion_top_k: int = 6,
    bound_completion_max_calls: int = 2,
    enable_bound_completion_cost_admission: bool = True,
    bound_completion_max_risk: float = 0.42,
    bound_completion_min_net_utility: float = 0.20,
    compute_weight: float = 0.08,
    risk_weight: float = 0.62,
    enable_canonical_bound_completion: bool = True,
    enable_completion_selectivity_feedback: bool = True,
    enable_query_plan_root_contract: bool = True,
    enable_inverse_relation_completion: bool = True,
    enable_title_grounded_completion: bool = False,
    enable_source_grounded_terminal_repair: bool = False,
    enable_title_catalog_root_plan: bool = False,
    enable_title_root_plan_commit: bool = False,
    enable_identity_safe_root_commit: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    plans = _multi_hop_plan_names(question, gap, stage_aware_plan)
    info = {
        "enabled": bool(plans),
        "plans": plans,
        "raw_path_count": 0,
        "row_count": 0,
        "surface_materialized_count": 0,
        "min_bridge_overlap": round(float(multi_hop_min_bridge_overlap), 6),
        "surface_materializer_enabled": bool(enable_surface_materializer),
        "reject_reason_counts": {},
        "index_access": _new_index_access_stats(),
        "query_plan_execution_trace": {},
        "semantic_plan_execution": {
            "required": bool(stage_aware_plan),
            "completed": False,
            "status": "plan_not_executed",
            "terminal_binding_count": 0,
        },
    }
    if not plans:
        return [], info
    query_anchor_tokens = set(gap.get("query_anchor_tokens") or [])
    if not query_anchor_tokens:
        info["reject_reason_counts"] = {"no_query_anchor": 1}
        return [], info
    items = relation_index.get("items") or []
    anchor_items = _lookup_items_by_entity_tokens(
        relation_index,
        query_anchor_tokens,
        max_per_key=max_facts_per_key,
        max_fanout=max_key_fanout,
        access_stats=info["index_access"],
    )
    rows: List[Dict[str, Any]] = []
    rejects = Counter()

    if (
        set(plans)
        & {"slot_dependency_query_plan_v3", "schema_constrained_query_plan_v4"}
        and stage_aware_plan
    ):
        rows, slot_rejects, execution_trace = _execute_slot_dependency_query_plan(
            question=question,
            plan=stage_aware_plan,
            relation_index=relation_index,
            anchor_items=anchor_items,
            rank_map=rank_map,
            get_doc_content=get_doc_content,
            max_facts_per_key=max_facts_per_key,
            max_key_fanout=max_key_fanout,
            access_stats=info["index_access"],
            enable_surface_materializer=bool(enable_surface_materializer),
            enable_bound_completion=bool(enable_bound_completion),
            bound_completion_lookup=bound_completion_lookup,
            bound_completion_top_k=int(bound_completion_top_k),
            bound_completion_max_calls=int(bound_completion_max_calls),
            enable_bound_completion_cost_admission=bool(
                enable_bound_completion_cost_admission
            ),
            bound_completion_max_risk=float(bound_completion_max_risk),
            bound_completion_min_net_utility=float(
                bound_completion_min_net_utility
            ),
            compute_weight=float(compute_weight),
            risk_weight=float(risk_weight),
            enable_canonical_bound_completion=bool(
                enable_canonical_bound_completion
            ),
            enable_completion_selectivity_feedback=bool(
                enable_completion_selectivity_feedback
            ),
            enable_query_plan_root_contract=bool(
                enable_query_plan_root_contract
            ),
            enable_inverse_relation_completion=bool(
                enable_inverse_relation_completion
            ),
            enable_title_grounded_completion=bool(
                enable_title_grounded_completion
            ),
            enable_source_grounded_terminal_repair=bool(
                enable_source_grounded_terminal_repair
            ),
            enable_title_catalog_root_plan=bool(
                enable_title_catalog_root_plan
            ),
            commit_title_root_plan=bool(enable_title_root_plan_commit),
            enable_identity_safe_root_commit=bool(
                enable_identity_safe_root_commit
            ),
        )
        rejects.update(slot_rejects)
        info["raw_path_count"] = len(rows)
        rows = _merge_directional_rows(rows, max_rows=max_rows)
        info["row_count"] = len(rows)
        info["surface_materialized_count"] = sum(
            1 for row in rows if row.get("surface_materialized")
        )
        execution_trace["role_direction_required_row_count"] = sum(
            bool(row.get("role_direction_required")) for row in rows
        )
        execution_trace["role_direction_validated_row_count"] = sum(
            bool(row.get("role_direction_required"))
            and bool(row.get("role_direction_validated"))
            for row in rows
        )
        execution_trace["role_direction_match_count"] = sum(
            len(row.get("role_direction_matches") or []) for row in rows
        )
        info["reject_reason_counts"] = dict(rejects)
        info["slot_dependency_plan_executed"] = True
        semantic_execution = _semantic_plan_execution_summary(
            plan=stage_aware_plan,
            rows=rows,
            execution_trace=execution_trace,
            rejects=rejects,
        )
        execution_trace["semantic_plan_execution"] = semantic_execution
        info["query_plan_execution_trace"] = execution_trace
        info["semantic_plan_execution"] = semantic_execution
        info["exact_join_validated_count"] = sum(
            1 for row in rows if row.get("slot_dependency_validated")
        )
        return rows, info

    if "kinship_grandparent_2hop" in plans:
        q = str(question or "").lower()
        first_required = "father" if "paternal" in q else "mother" if "maternal" in q else "parent"
        second_required = "father" if "grandfather" in q else "mother" if "grandmother" in q else "parent"
        for first in anchor_items:
            orient = _parent_edge_orientation(first)
            if orient is None:
                continue
            parent_side, child_side, parent_kind = orient
            child_score = _endpoint_overlap(_edge_side_tokens(first, child_side), query_anchor_tokens, max_required=4)
            if child_score < float(hard_anchor_min_overlap):
                continue
            if not _kinship_parent_kind_allowed(parent_kind, first_required):
                rejects["first_parent_kind_mismatch"] += 1
                continue
            parent_tokens = _edge_side_tokens(first, parent_side)
            parent_name = _edge_side_text(first, parent_side)
            if not parent_tokens:
                continue
            second_candidates = _lookup_items_by_entity_tokens(
                relation_index,
                parent_tokens,
                max_per_key=max_facts_per_key,
                max_fanout=max_key_fanout,
                access_stats=info["index_access"],
            )
            for second in second_candidates:
                if second is first:
                    continue
                orient2 = _parent_edge_orientation(second)
                if orient2 is None:
                    continue
                grand_side, second_child_side, grand_kind = orient2
                parent_match = _entity_bridge_score(
                    _edge_side_tokens(second, second_child_side),
                    parent_tokens,
                    max_required=4,
                )
                if parent_match < float(multi_hop_min_bridge_overlap):
                    rejects["bridge_entity_mismatch"] += 1
                    continue
                if not _kinship_parent_kind_allowed(grand_kind, second_required):
                    rejects["second_parent_kind_mismatch"] += 1
                    continue
                candidate = _edge_side_text(second, grand_side)
                if not candidate:
                    continue
                if _jaccard(_tokens(candidate), _tokens(parent_name)) > 0.95:
                    continue
                row = _build_multi_hop_row(
                    candidate=candidate,
                    answer_type="person",
                    plan="kinship_grandparent_2hop",
                    anchor_entity=_edge_side_text(first, child_side),
                    bridge_entity=parent_name,
                    path_items=[first, second],
                    path_roles=[f"{parent_kind}_of_anchor", f"{grand_kind}_of_parent"],
                    anchor_score=min(1.0, (child_score + parent_match) / 2.0),
                    role_score=1.0,
                    rank_map=rank_map,
                    get_doc_content=get_doc_content,
                    enable_surface_materializer=bool(enable_surface_materializer),
                )
                if row is not None:
                    rows.append(row)

    creator_plans = {"creator_birth_place_2hop", "creator_birth_date_2hop"} & set(plans)
    if creator_plans:
        for first in anchor_items:
            if not _relation_has_any(first, DIRECTOR_RELATION_TERMS | RELATION_FAMILY_TERMS["creator"]):
                continue
            subj_anchor = _endpoint_overlap(set(first.get("subject_tokens") or set()), query_anchor_tokens, max_required=4)
            obj_anchor = _endpoint_overlap(set(first.get("object_tokens") or set()), query_anchor_tokens, max_required=4)
            if subj_anchor >= float(hard_anchor_min_overlap) and subj_anchor >= obj_anchor:
                anchor_entity = str(first.get("subject") or "")
                creator_tokens = set(first.get("object_tokens") or set())
                creator_name = str(first.get("object") or "")
                anchor_score = subj_anchor
            elif obj_anchor >= float(hard_anchor_min_overlap):
                anchor_entity = str(first.get("object") or "")
                creator_tokens = set(first.get("subject_tokens") or set())
                creator_name = str(first.get("subject") or "")
                anchor_score = obj_anchor
            else:
                continue
            if not creator_tokens:
                continue
            second_candidates = _lookup_items_by_entity_tokens(
                relation_index,
                creator_tokens,
                max_per_key=max_facts_per_key,
                max_fanout=max_key_fanout,
                access_stats=info["index_access"],
            )
            for second in second_candidates:
                if second is first:
                    continue
                if not _relation_has_any(second, BIRTH_PLACE_RELATION_TERMS | BIRTH_DATE_RELATION_TERMS):
                    continue
                subj_creator = _entity_bridge_score(
                    set(second.get("subject_tokens") or set()),
                    creator_tokens,
                    max_required=4,
                )
                obj_creator = _entity_bridge_score(
                    set(second.get("object_tokens") or set()),
                    creator_tokens,
                    max_required=4,
                )
                if subj_creator >= float(multi_hop_min_bridge_overlap):
                    candidate = str(second.get("object") or "")
                    creator_match = subj_creator
                elif obj_creator >= float(multi_hop_min_bridge_overlap):

                    rejects["birth_relation_reverse_not_answer_slot"] += 1
                    continue
                else:
                    rejects["bridge_entity_mismatch"] += 1
                    continue
                for plan in sorted(creator_plans):
                    answer_type = "place" if plan == "creator_birth_place_2hop" else "date"
                    if not _object_matches_answer_type(candidate, {answer_type}, ""):
                        rejects[f"{plan}_answer_type_mismatch"] += 1
                        continue
                    if _jaccard(_tokens(candidate), _tokens(creator_name)) > 0.95:
                        continue
                    row = _build_multi_hop_row(
                        candidate=candidate,
                        answer_type=answer_type,
                        plan=plan,
                        anchor_entity=anchor_entity,
                        bridge_entity=creator_name,
                        path_items=[first, second],
                        path_roles=["anchor_to_creator", "creator_to_birth_answer"],
                        anchor_score=min(1.0, (anchor_score + creator_match) / 2.0),
                        role_score=1.0,
                        rank_map=rank_map,
                        get_doc_content=get_doc_content,
                        enable_surface_materializer=bool(enable_surface_materializer),
                    )
                    if row is not None:
                        rows.append(row)

    if "stage_aware_nested_attribute_2hop" in plans and stage_aware_plan:
        bridge_terms = set(stage_aware_plan.get("bridge_relation_tokens") or [])
        bridge_predicate_families = set(
            stage_aware_plan.get("bridge_predicate_families") or []
        )
        bridge_relation_families = set(
            stage_aware_plan.get("bridge_relation_families") or []
        )
        terminal_terms = set(
            stage_aware_plan.get("terminal_relation_tokens") or []
        )
        terminal_families = set(
            stage_aware_plan.get("terminal_predicate_families") or []
        )
        final_answer_type = str(
            stage_aware_plan.get("final_answer_type") or "entity"
        )
        required_parent_kind = (
            "mother" if "mother" in bridge_terms or "maternal" in bridge_terms
            else "father" if "father" in bridge_terms or "paternal" in bridge_terms
            else "parent"
        )
        for first in anchor_items:
            first_relation_tokens = set(first.get("relation_tokens") or set())
            first_predicate_families = _canonical_predicate_families_for_tokens(
                first_relation_tokens
            )
            first_relation_families = _relation_families_for_tokens(
                first_relation_tokens
            )
            if not (
                first_relation_tokens & bridge_terms
                or first_predicate_families & bridge_predicate_families
                or first_relation_families & bridge_relation_families
            ):
                continue

            anchor_entity = ""
            bridge_entity = ""
            bridge_entity_tokens: Set[str] = set()
            anchor_score = 0.0
            path_role = "anchor_to_bridge"
            parent_orientation = _parent_edge_orientation(first)
            if parent_orientation is not None and (
                first_relation_families & {"kinship"}
                or first_relation_tokens & bridge_terms
            ):
                parent_side, child_side, parent_kind = parent_orientation
                child_score = _endpoint_overlap(
                    _edge_side_tokens(first, child_side),
                    query_anchor_tokens,
                    max_required=4,
                )
                if child_score < float(hard_anchor_min_overlap):
                    continue
                if not _kinship_parent_kind_allowed(
                    parent_kind, required_parent_kind
                ):
                    rejects["stage_bridge_parent_kind_mismatch"] += 1
                    continue
                anchor_entity = _edge_side_text(first, child_side)
                bridge_entity = _edge_side_text(first, parent_side)
                bridge_entity_tokens = _edge_side_tokens(first, parent_side)
                anchor_score = child_score
                path_role = f"{parent_kind}_of_anchor"
            else:
                subject_score = _endpoint_overlap(
                    set(first.get("subject_tokens") or set()),
                    query_anchor_tokens,
                    max_required=4,
                )
                object_score = _endpoint_overlap(
                    set(first.get("object_tokens") or set()),
                    query_anchor_tokens,
                    max_required=4,
                )
                if (
                    subject_score >= float(hard_anchor_min_overlap)
                    and subject_score >= object_score
                ):
                    anchor_entity = str(first.get("subject") or "")
                    bridge_entity = str(first.get("object") or "")
                    bridge_entity_tokens = set(
                        first.get("object_tokens") or set()
                    )
                    anchor_score = subject_score
                elif object_score >= float(hard_anchor_min_overlap):
                    anchor_entity = str(first.get("object") or "")
                    bridge_entity = str(first.get("subject") or "")
                    bridge_entity_tokens = set(
                        first.get("subject_tokens") or set()
                    )
                    anchor_score = object_score
                else:
                    continue

            if not bridge_entity_tokens:
                continue
            second_candidates = _lookup_items_by_entity_tokens(
                relation_index,
                bridge_entity_tokens,
                max_per_key=max_facts_per_key,
                max_fanout=max_key_fanout,
                access_stats=info["index_access"],
            )
            for second in second_candidates:
                if second is first:
                    continue
                second_relation_tokens = set(
                    second.get("relation_tokens") or set()
                )
                second_predicate_families = (
                    _canonical_predicate_families_for_tokens(
                        second_relation_tokens
                    )
                )
                if not (
                    second_predicate_families & terminal_families
                    or second_relation_tokens & terminal_terms
                ):
                    continue
                subject_bridge = _entity_bridge_score(
                    set(second.get("subject_tokens") or set()),
                    bridge_entity_tokens,
                    max_required=4,
                )
                object_bridge = _entity_bridge_score(
                    set(second.get("object_tokens") or set()),
                    bridge_entity_tokens,
                    max_required=4,
                )
                if subject_bridge >= float(multi_hop_min_bridge_overlap):
                    candidate = str(second.get("object") or "")
                    bridge_match = subject_bridge
                elif object_bridge >= float(multi_hop_min_bridge_overlap):
                    candidate = str(second.get("subject") or "")
                    bridge_match = object_bridge
                else:
                    rejects["stage_terminal_bridge_entity_mismatch"] += 1
                    continue
                if not _terminal_value_matches_contract(
                    candidate,
                    final_answer_type,
                    second_relation_tokens,
                ):
                    rejects["stage_terminal_answer_type_mismatch"] += 1
                    continue
                if _jaccard(_tokens(candidate), bridge_entity_tokens) > 0.95:
                    rejects["stage_terminal_repeats_bridge"] += 1
                    continue
                row = _build_multi_hop_row(
                    candidate=candidate,
                    answer_type=final_answer_type,
                    plan="stage_aware_nested_attribute_2hop",
                    anchor_entity=anchor_entity,
                    bridge_entity=bridge_entity,
                    path_items=[first, second],
                    path_roles=[path_role, "bridge_to_typed_terminal"],
                    anchor_score=min(
                        1.0, (anchor_score + bridge_match) / 2.0
                    ),
                    role_score=1.0,
                    rank_map=rank_map,
                    get_doc_content=get_doc_content,
                    enable_surface_materializer=bool(
                        enable_surface_materializer
                    ),
                )
                if row is not None:
                    row.update({
                        "terminal_contract_version": (
                            "stage_aware_typed_terminal_candidate_v2"
                        ),
                        "validation_tier": "strict",
                        "shadow_only": False,
                        "production_eligible": True,
                        "base_validation_passed": True,
                        "slot_plan_stage": "final_answer",
                        "final_answer_type": final_answer_type,
                        "final_predicate_match": True,
                        "bridge_predicate_match": False,
                        "final_answer_type_match": True,
                        "bridge_eligible": False,
                        "rooted_lineage_complete": True,
                        "lineage_depth": 2,
                        "plan_stage_count": 2,
                    })
                    rows.append(row)

    if stage_aware_plan:
        final_answer_type = str(
            stage_aware_plan.get("final_answer_type") or "entity"
        )
        for row in rows:
            if row.get("slot_plan_stage"):
                continue
            final_type_match = bool(
                str(row.get("answer_type") or "") == final_answer_type
            )
            row.update({
                "terminal_contract_version": (
                    "stage_aware_typed_terminal_candidate_v2"
                ),
                "validation_tier": (
                    "strict" if final_type_match else "shadow"
                ),
                "shadow_only": not final_type_match,
                "production_eligible": final_type_match,
                "base_validation_passed": True,
                "slot_plan_stage": (
                    "final_answer" if final_type_match else "support_only"
                ),
                "final_answer_type": final_answer_type,
                "final_predicate_match": final_type_match,
                "bridge_predicate_match": False,
                "final_answer_type_match": final_type_match,
                "bridge_eligible": False,
                "rooted_lineage_complete": bool(final_type_match),
                "lineage_depth": len(row.get("path_doc_ids") or []),
                "plan_stage_count": len(
                    stage_aware_plan.get("steps") or []
                ),
            })

    info["raw_path_count"] = len(rows)
    rows = _merge_directional_rows(rows, max_rows=max_rows)
    info["row_count"] = len(rows)
    info["surface_materialized_count"] = sum(1 for row in rows if row.get("surface_materialized"))
    info["reject_reason_counts"] = dict(rejects)
    return rows, info


def _clean_triple(triple: Any) -> Optional[Tuple[str, str, str]]:
    if not isinstance(triple, (list, tuple)) or len(triple) != 3:
        return None
    processed = tuple(text_processing(list(triple)))
    if len(processed) != 3:
        return None
    if not processed[0] or not processed[1] or not processed[2]:
        return None
    return processed


def _iter_clean_triples(raw_triples: Any) -> Iterable[Tuple[str, str, str]]:
    for triple in flatten_facts([raw_triples or []]):
        clean = _clean_triple(triple)
        if clean is not None:
            yield clean


def build_relation_fact_join_index(
    all_openie_info: List[dict],
    passage_node_key_to_idx: Dict[str, int],
) -> Dict[str, Any]:
    """Build a typed fact-join index over OpenIE triples.

    The index is intentionally local and deterministic. It lets query-time
    retrieval look up passages by shared endpoints, relation tokens, and typed
    relation signatures without issuing an extra LLM call.
    """
    items: List[Dict[str, Any]] = []
    entity_to_items: Dict[str, List[int]] = defaultdict(list)
    entity_surface_counts: Dict[str, Counter] = defaultdict(Counter)
    subject_to_items: Dict[str, List[int]] = defaultdict(list)
    object_to_items: Dict[str, List[int]] = defaultdict(list)
    subject_predicate_family_to_items: Dict[str, List[int]] = defaultdict(list)
    object_predicate_family_to_items: Dict[str, List[int]] = defaultdict(list)
    subject_predicate_subtype_to_items: Dict[str, List[int]] = defaultdict(list)
    object_predicate_subtype_to_items: Dict[str, List[int]] = defaultdict(list)
    relation_to_items: Dict[str, List[int]] = defaultdict(list)
    schema_to_items: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    schema_v2_to_items: Dict[Tuple[str, str, str, str], List[int]] = defaultdict(list)
    doc_to_items: Dict[int, List[int]] = defaultdict(list)
    tuple_stats: Dict[str, Dict[str, Any]] = {}
    slot_answer_stats: Dict[str, Dict[str, Any]] = {}
    slot_answers: Dict[str, Set[str]] = defaultdict(set)
    slot_doc_ids: Dict[str, Set[int]] = defaultdict(set)
    predicate_catalog_stats: Dict[str, Dict[str, Any]] = {}
    doc_title_surfaces: Dict[int, str] = {}
    doc_evidence_token_counts: Dict[int, int] = {}
    title_entity_doc_ids: Dict[str, List[int]] = defaultdict(list)
    title_identity_doc_ids: Dict[str, List[int]] = defaultdict(list)
    title_family_doc_ids: Dict[str, List[int]] = defaultdict(list)
    title_family_qualifiers: Dict[str, Set[str]] = defaultdict(set)
    title_doc_categories: Dict[int, Set[str]] = defaultdict(set)
    title_family_category_doc_ids: Dict[
        str, Dict[str, List[int]]
    ] = defaultdict(lambda: defaultdict(list))
    title_key_to_family_key: Dict[str, str] = {}
    title_backed_entity_keys: Set[str] = set()
    title_entity_surfaces: Dict[str, str] = {}
    title_identity_surfaces: Dict[str, str] = {}
    title_surface_surfaces: Dict[str, str] = {}
    title_surface_doc_ids: Dict[str, List[int]] = defaultdict(list)
    doc_passage_surfaces: Dict[int, str] = {}
    doc_copular_identity_pairs: Dict[int, Set[Tuple[str, str]]] = defaultdict(set)

    def update_tuple_stats(item: Dict[str, Any]) -> None:
        tuple_key = str(item.get("tuple_key") or "")
        if not tuple_key:
            return
        stats = tuple_stats.setdefault(
            tuple_key,
            {
                "doc_ids": set(),
                "triple_count": 0,
                "subject_key": item.get("subject_key", ""),
                "relation_key": item.get("relation_key", ""),
                "object_key": item.get("object_key", ""),
                "signature": item.get("signature"),
                "relation_tokens": set(),
            },
        )
        stats["doc_ids"].add(int(item["doc_idx"]))
        stats["triple_count"] = int(stats.get("triple_count", 0) or 0) + 1
        stats["relation_tokens"].update(set(item.get("relation_tokens") or set()))

    def update_slot_stats(label: str, bridge_key: str, answer_key: str, item: Dict[str, Any]) -> None:
        slot_key = _fact_slot_key(label, bridge_key)
        slot_answer_key = _fact_slot_answer_key(label, bridge_key, answer_key)
        if not slot_key or not slot_answer_key:
            return
        slot_answers[slot_key].add(str(answer_key))
        slot_doc_ids[slot_key].add(int(item["doc_idx"]))
        stats = slot_answer_stats.setdefault(
            slot_answer_key,
            {
                "slot_key": slot_key,
                "label": str(label),
                "bridge_key": str(bridge_key),
                "answer_key": str(answer_key),
                "doc_ids": set(),
                "triple_count": 0,
                "support_triples": [],
                "tuple_keys": set(),
            },
        )
        stats["doc_ids"].add(int(item["doc_idx"]))
        stats["triple_count"] = int(stats.get("triple_count", 0) or 0) + 1
        if len(stats["support_triples"]) < 8:
            stats["support_triples"].append(" | ".join(str(part) for part in (item.get("triple") or [])))
        tuple_key = str(item.get("tuple_key") or "")
        if tuple_key:
            stats["tuple_keys"].add(tuple_key)

    def update_predicate_catalog(item: Dict[str, Any]) -> None:
        relation_key = str(item.get("relation_key") or "")
        if not relation_key:
            return
        stats = predicate_catalog_stats.setdefault(
            relation_key,
            {
                "row_count": 0,
                "doc_ids": set(),
                "relation_tokens": set(),
                "subject_types": Counter(),
                "object_types": Counter(),
            },
        )
        stats["row_count"] = int(stats.get("row_count", 0) or 0) + 1
        stats["doc_ids"].add(int(item["doc_idx"]))
        stats["relation_tokens"].update(set(item.get("relation_tokens") or set()))
        stats["subject_types"][_entity_type(item.get("subject"))] += 1
        stats["object_types"][_entity_type(item.get("object"))] += 1

    for doc in all_openie_info or []:
        chunk_id = doc.get("idx")
        if chunk_id not in passage_node_key_to_idx:
            continue
        doc_idx = int(passage_node_key_to_idx[chunk_id])
        passage_surface = str(doc.get("passage") or "")
        doc_passage_surfaces[doc_idx] = passage_surface


        doc_evidence_token_counts[doc_idx] = (
            len(re.findall(r"\S+", passage_surface)) if passage_surface else 0
        )
        passage_lines = passage_surface.splitlines()
        if passage_lines and passage_lines[0].strip():
            doc_title_surfaces[doc_idx] = passage_lines[0].strip()
            document_categories = _title_document_categories(
                doc_title_surfaces[doc_idx], passage_surface
            )
            title_doc_categories[doc_idx].update(document_categories)
            title_key = _entity_key(doc_title_surfaces[doc_idx])
            title_identity_key = canonical_entity_identity_key(
                doc_title_surfaces[doc_idx]
            )
            title_surface_key = canonical_title_surface_key(
                doc_title_surfaces[doc_idx]
            )
            if title_key:
                title_entity_surfaces.setdefault(
                    title_key, doc_title_surfaces[doc_idx]
                )
                title_entity_doc_ids[title_key].append(doc_idx)
                title_backed_entity_keys.add(title_key)
                title_family_key, title_qualifiers = _title_family_parts(
                    doc_title_surfaces[doc_idx]
                )
                if title_family_key:







                    title_key_to_family_key.setdefault(
                        title_family_key, title_family_key
                    )
                    title_key_to_family_key[title_key] = title_family_key
                    title_family_doc_ids[title_family_key].append(doc_idx)
                    title_family_qualifiers[title_family_key].update(
                        title_qualifiers
                    )
                    for category in document_categories:
                        title_family_category_doc_ids[title_family_key][
                            category
                        ].append(doc_idx)
            if title_identity_key:
                title_identity_surfaces.setdefault(
                    title_identity_key, doc_title_surfaces[doc_idx]
                )
                title_identity_doc_ids[title_identity_key].append(doc_idx)
            if title_surface_key:
                title_surface_surfaces.setdefault(
                    title_surface_key, doc_title_surfaces[doc_idx]
                )
                title_surface_doc_ids[title_surface_key].append(doc_idx)
        clean_triples = list(_iter_clean_triples(
            doc.get("extracted_triples", [])
        ))
        for raw_subject, raw_relation, raw_object in clean_triples:
            if str(raw_relation or "").strip().casefold() in {
                "is", "was", "are", "were",
            }:
                doc_copular_identity_pairs[doc_idx].add((
                    _entity_key(raw_subject),
                    _entity_key(raw_object),
                ))
        for triple in clean_triples:
            subject, relation, obj = triple
            relation_tokens = _tokens(relation)
            predicate_contract = canonical_predicate_contract(relation, obj)
            predicate_families = set(
                predicate_contract["canonical_families"]
            )
            predicate_subtypes = set(
                predicate_contract["semantic_subtypes"]
            )
            endpoint_tokens = _tokens(subject) | _tokens(obj)
            if not relation_tokens or not endpoint_tokens:
                continue
            signature = _schema_signature(triple)
            schema_signatures_v2 = typed_schema_signatures_v2(triple)
            subject_key = _entity_key(subject)
            relation_key = _relation_key(relation)
            object_key = _entity_key(obj)
            tuple_key = canonical_fact_tuple_key(triple)
            item_id = len(items)
            item = {
                "item_id": item_id,
                "doc_idx": doc_idx,
                "chunk_id": chunk_id,
                "triple": triple,
                "subject": subject,
                "relation": relation,
                "object": obj,
                "subject_tokens": _tokens(subject),
                "object_tokens": _tokens(obj),
                "endpoint_tokens": endpoint_tokens,
                "relation_tokens": relation_tokens,
                "predicate_families": predicate_families,
                "predicate_subtypes": predicate_subtypes,
                "predicate_contract": predicate_contract,
                "signature": signature,
                "schema_signatures_v2": schema_signatures_v2,
                "subject_key": subject_key,
                "relation_key": relation_key,
                "object_key": object_key,
                "tuple_key": tuple_key,
                "entity_keys": {
                    key for key in (_entity_key(subject), _entity_key(obj)) if key
                },
            }
            item["canonical_role_relation"] = _canonical_fact_role_relation(
                item
            )
            items.append(item)
            if subject_key:
                entity_surface_counts[subject_key][str(subject).strip()] += 1
            if object_key:
                entity_surface_counts[object_key][str(obj).strip()] += 1
            title_tokens = _tokens(doc_title_surfaces.get(doc_idx, ""))
            if title_tokens:
                if item["subject_tokens"] and item["subject_tokens"].issubset(
                    title_tokens
                ):
                    title_backed_entity_keys.add(subject_key)
                if item["object_tokens"] and item["object_tokens"].issubset(
                    title_tokens
                ):
                    title_backed_entity_keys.add(object_key)
            doc_to_items[doc_idx].append(item_id)
            update_tuple_stats(item)
            update_predicate_catalog(item)

            for label in _fact_slot_labels_for_relation_tokens(relation_tokens):
                if label in {"birth", "death"}:
                    update_slot_stats(label, subject_key, object_key, item)
                elif label in {"parent", "father", "mother"}:
                    orient = _parent_edge_orientation(item)
                    if orient is not None:
                        parent_side, child_side, parent_kind = orient
                        if label in {"father", "mother"} and parent_kind != label:
                            continue
                        update_slot_stats(
                            label,
                            _entity_key(_edge_side_text(item, child_side)),
                            _entity_key(_edge_side_text(item, parent_side)),
                            item,
                        )

            for key in item["entity_keys"]:
                entity_to_items[key].append(item_id)
            if subject_key:
                subject_to_items[subject_key].append(item_id)
            if object_key:
                object_to_items[object_key].append(item_id)
            for predicate_family in predicate_families:
                subject_family_key = _role_predicate_index_key(
                    subject_key,
                    predicate_family,
                )
                object_family_key = _role_predicate_index_key(
                    object_key,
                    predicate_family,
                )
                if subject_family_key:
                    subject_predicate_family_to_items[
                        subject_family_key
                    ].append(item_id)
                if object_family_key:
                    object_predicate_family_to_items[
                        object_family_key
                    ].append(item_id)
            for predicate_subtype in predicate_subtypes:
                subject_subtype_key = _role_predicate_index_key(
                    subject_key,
                    predicate_subtype,
                )
                object_subtype_key = _role_predicate_index_key(
                    object_key,
                    predicate_subtype,
                )
                if subject_subtype_key:
                    subject_predicate_subtype_to_items[
                        subject_subtype_key
                    ].append(item_id)
                if object_subtype_key:
                    object_predicate_subtype_to_items[
                        object_subtype_key
                    ].append(item_id)
            for token in endpoint_tokens:
                if len(token) > 2:
                    entity_to_items[token].append(item_id)
            for token in relation_tokens:
                if len(token) > 2:
                    relation_to_items[token].append(item_id)
            relation_key = _relation_key(relation)
            if relation_key:
                relation_to_items[relation_key].append(item_id)
            if signature is not None:
                schema_to_items[signature].append(item_id)
            for schema_signature_v2 in schema_signatures_v2:
                schema_v2_to_items[schema_signature_v2].append(item_id)

    compact_tuple_stats = {}
    for key, stats in tuple_stats.items():
        compact_tuple_stats[key] = {
            "doc_count": len(stats.get("doc_ids") or set()),
            "triple_count": int(stats.get("triple_count", 0) or 0),
            "doc_ids": sorted(stats.get("doc_ids") or set())[:16],
            "subject_key": stats.get("subject_key", ""),
            "relation_key": stats.get("relation_key", ""),
            "object_key": stats.get("object_key", ""),
            "signature": stats.get("signature"),
            "relation_tokens": sorted(stats.get("relation_tokens") or set()),
        }
    compact_slot_answer_stats = {}
    for key, stats in slot_answer_stats.items():
        slot_key = str(stats.get("slot_key") or "")
        compact_slot_answer_stats[key] = {
            "slot_key": slot_key,
            "label": stats.get("label", ""),
            "bridge_key": stats.get("bridge_key", ""),
            "answer_key": stats.get("answer_key", ""),
            "doc_count": len(stats.get("doc_ids") or set()),
            "triple_count": int(stats.get("triple_count", 0) or 0),
            "doc_ids": sorted(stats.get("doc_ids") or set())[:16],
            "support_triples": list(stats.get("support_triples") or [])[:8],
            "tuple_keys": sorted(stats.get("tuple_keys") or set())[:8],
            "conflict_candidate_count": len(slot_answers.get(slot_key, set())),
            "slot_doc_count": len(slot_doc_ids.get(slot_key, set())),
        }
    fact_reliability = {
        "tuple_stats": compact_tuple_stats,
        "slot_answer_stats": compact_slot_answer_stats,
        "slot_conflict_counts": {
            key: len(value)
            for key, value in slot_answers.items()
        },
        "slot_doc_counts": {
            key: len(value)
            for key, value in slot_doc_ids.items()
        },
    }
    predicate_catalog = {
        key: {
            "row_count": int(stats.get("row_count", 0) or 0),
            "doc_count": len(stats.get("doc_ids") or set()),
            "relation_tokens": sorted(stats.get("relation_tokens") or set()),
            "subject_type_counts": dict(stats.get("subject_types") or {}),
            "object_type_counts": dict(stats.get("object_types") or {}),
        }
        for key, stats in predicate_catalog_stats.items()
    }



    alias_equivalent_keys: Dict[str, Set[str]] = defaultdict(set)
    alias_equivalent_sources: Dict[str, Set[str]] = defaultdict(set)
    title_alias_equivalent_keys: Dict[str, Set[str]] = defaultdict(set)
    title_alias_equivalent_sources: Dict[str, Set[str]] = defaultdict(set)

    def add_alias(left_key: str, right_key: str, source: str) -> None:
        if not left_key or not right_key or left_key == right_key:
            return
        alias_equivalent_keys[left_key].add(right_key)
        alias_equivalent_keys[right_key].add(left_key)
        alias_equivalent_sources[f"{left_key}\t{right_key}"].add(source)
        alias_equivalent_sources[f"{right_key}\t{left_key}"].add(source)

    def add_title_alias(left_key: str, right_key: str, source: str) -> None:
        if not left_key or not right_key or left_key == right_key:
            return
        title_alias_equivalent_keys[left_key].add(right_key)
        title_alias_equivalent_keys[right_key].add(left_key)
        title_alias_equivalent_sources[f"{left_key}\t{right_key}"].add(source)
        title_alias_equivalent_sources[f"{right_key}\t{left_key}"].add(source)

    for item in items:
        if "identity" not in set(item.get("predicate_families") or set()):
            continue
        subject_key = str(item.get("subject_key") or "")
        object_key = str(item.get("object_key") or "")
        add_alias(subject_key, object_key, "explicit_identity_fact")








    for doc_idx, title_surface in doc_title_surfaces.items():
        title_words = _surface_word_tokens(title_surface)
        title_key = _entity_key(title_surface)
        if not (2 <= len(title_words) <= 8) or any(
            token.isdigit() for token in title_words
        ):
            continue
        title_has_parenthetical = bool(re.search(r"\([^)]*\)", title_surface))
        title_has_qualifier_marker = bool(
            title_has_parenthetical
            or re.search(r",|\bof\b", str(title_surface), flags=re.IGNORECASE)
        )
        for item_id in doc_to_items.get(doc_idx) or []:
            if item_id < 0 or item_id >= len(items):
                continue
            item = items[item_id]
            for endpoint_name in ("subject", "object"):
                endpoint_surface = str(item.get(endpoint_name) or "")
                endpoint_words = _surface_word_tokens(endpoint_surface)
                endpoint_key = str(item.get(f"{endpoint_name}_key") or "")
                if not endpoint_key or not endpoint_words:
                    continue
                title_token_set = set(title_words)
                endpoint_token_set = set(endpoint_words)
                token_sets_contained = bool(
                    title_token_set.issubset(endpoint_token_set)
                    or endpoint_token_set.issubset(title_token_set)
                )




                if (
                    len(endpoint_words) >= 2
                    and token_sets_contained
                    and endpoint_words[0] == title_words[0]
                    and (
                        endpoint_words[-1] == title_words[-1]
                        or title_has_qualifier_marker
                    )
                ):
                    add_title_alias(
                        title_key,
                        endpoint_key,
                        f"same_passage_title_subset:{doc_idx}",
                    )
                elif (
                    title_has_parenthetical
                    and len(endpoint_words) == 1
                    and endpoint_token_set.issubset(title_token_set)
                    and endpoint_words[0] == title_words[0]
                ):
                    add_title_alias(
                        title_key,
                        endpoint_key,
                        f"same_passage_parenthetical_base:{doc_idx}",
                    )








        passage_lines = str(doc_passage_surfaces.get(doc_idx) or "").splitlines()
        lead_text = " ".join(
            line.strip() for line in passage_lines[1:3] if line.strip()
        )
        subtitle_match = re.match(
            rf"^{re.escape(str(title_surface).strip())}\s*:\s*"
            r"(?P<subtitle>.+?)\s+(?:is|was|are|were)\b",
            lead_text,
            flags=re.IGNORECASE,
        )
        if subtitle_match:
            subtitle_surface = str(
                subtitle_match.group("subtitle") or ""
            ).strip()
            subtitle_words = _surface_word_tokens(subtitle_surface)
            subtitle_key = _entity_key(subtitle_surface)
            copular_bridge = (
                title_key,
                subtitle_key,
            ) in doc_copular_identity_pairs.get(doc_idx, set())
            if (
                subtitle_key
                and 2 <= len(subtitle_words) <= 12
                and not all(token.isdigit() for token in subtitle_words)
                and copular_bridge
            ):
                add_title_alias(
                    title_key,
                    subtitle_key,
                    f"same_passage_title_subtitle:{doc_idx}",
                )

    return {
        "items": items,
        "entity_to_items": dict(entity_to_items),
        "subject_to_items": dict(subject_to_items),
        "object_to_items": dict(object_to_items),
        "canonical_entity_surfaces": {
            key: max(
                counts,
                key=lambda surface: (
                    int(counts[surface]),
                    len(_surface_word_tokens(surface)),
                    len(surface),
                    surface,
                ),
            )
            for key, counts in entity_surface_counts.items()
            if counts
        },
        "title_entity_surfaces": dict(title_entity_surfaces),
        "title_entity_doc_ids": {
            key: list(dict.fromkeys(doc_ids))
            for key, doc_ids in title_entity_doc_ids.items()
        },
        "title_identity_surfaces": dict(title_identity_surfaces),
        "title_identity_doc_ids": {
            key: list(dict.fromkeys(doc_ids))
            for key, doc_ids in title_identity_doc_ids.items()
        },
        "title_surface_surfaces": dict(title_surface_surfaces),
        "title_surface_doc_ids": {
            key: list(dict.fromkeys(doc_ids))
            for key, doc_ids in title_surface_doc_ids.items()
        },
        "doc_title_surfaces": dict(doc_title_surfaces),
        "doc_evidence_token_counts": dict(doc_evidence_token_counts),
        "title_key_to_family_key": dict(title_key_to_family_key),
        "title_family_doc_ids": {
            key: list(dict.fromkeys(doc_ids))
            for key, doc_ids in title_family_doc_ids.items()
        },
        "title_family_qualifiers": {
            key: sorted(values)
            for key, values in title_family_qualifiers.items()
        },
        "title_doc_categories": {
            int(doc_id): sorted(values)
            for doc_id, values in title_doc_categories.items()
            if values
        },
        "title_family_category_doc_ids": {
            family_key: {
                category: list(dict.fromkeys(doc_ids))
                for category, doc_ids in category_rows.items()
            }
            for family_key, category_rows in (
                title_family_category_doc_ids.items()
            )
        },
        "title_backed_entity_keys": sorted(title_backed_entity_keys),
        "subject_predicate_family_to_items": dict(
            subject_predicate_family_to_items
        ),
        "object_predicate_family_to_items": dict(
            object_predicate_family_to_items
        ),
        "subject_predicate_subtype_to_items": dict(
            subject_predicate_subtype_to_items
        ),
        "object_predicate_subtype_to_items": dict(
            object_predicate_subtype_to_items
        ),
        "relation_to_items": dict(relation_to_items),
        "schema_to_items": dict(schema_to_items),
        "schema_v2_to_items": dict(schema_v2_to_items),
        "doc_to_items": dict(doc_to_items),
        "tuple_stats": compact_tuple_stats,
        "fact_reliability": fact_reliability,
        "predicate_catalog": predicate_catalog,
        "alias_equivalent_keys": {
            key: sorted(values)
            for key, values in alias_equivalent_keys.items()
        },
        "alias_equivalent_sources": {
            key: sorted(values)
            for key, values in alias_equivalent_sources.items()
        },
        "title_alias_equivalent_keys": {
            key: sorted(values)
            for key, values in title_alias_equivalent_keys.items()
        },
        "title_alias_equivalent_sources": {
            key: sorted(values)
            for key, values in title_alias_equivalent_sources.items()
        },
        "stats": {
            "num_fact_items": len(items),
            "num_entity_keys": len(entity_to_items),
            "num_subject_keys": len(subject_to_items),
            "num_object_keys": len(object_to_items),
            "num_subject_predicate_family_keys": len(
                subject_predicate_family_to_items
            ),
            "num_object_predicate_family_keys": len(
                object_predicate_family_to_items
            ),
            "num_subject_predicate_subtype_keys": len(
                subject_predicate_subtype_to_items
            ),
            "num_object_predicate_subtype_keys": len(
                object_predicate_subtype_to_items
            ),
            "num_relation_keys": len(relation_to_items),
            "num_schema_keys": len(schema_to_items),
            "num_schema_v2_keys": len(schema_v2_to_items),
            "schema_v2_avg_rows_per_key": round(
                len(items) / float(max(1, len(schema_v2_to_items))),
                6,
            ),
            "num_doc_entries": len(doc_to_items),
            "num_doc_evidence_token_counts": len(doc_evidence_token_counts),
            "num_tuple_keys": len(compact_tuple_stats),
            "num_fact_slot_keys": len(slot_answers),
            "num_fact_slot_answer_keys": len(compact_slot_answer_stats),
            "num_predicate_catalog_entries": len(predicate_catalog),
            "num_alias_keys": len(alias_equivalent_keys),
            "num_title_alias_keys": len(title_alias_equivalent_keys),
            "num_title_backed_entity_keys": len(title_backed_entity_keys),
            "num_title_entity_keys": len(title_entity_surfaces),
            "num_title_entity_doc_keys": len(title_entity_doc_ids),
            "num_title_identity_keys": len(title_identity_doc_ids),
            "num_title_surface_keys": len(title_surface_doc_ids),
            "num_title_family_keys": len(title_family_doc_ids),
            "num_typed_title_docs": len(title_doc_categories),
            "num_typed_title_families": len(
                title_family_category_doc_ids
            ),
        },
    }


def _active_fact_signals(
    question: str,
    top_k_facts: List[Tuple],
    candidate_facts: Optional[List[Tuple]] = None,
    upstream_fact_tuples: Optional[List[Any]] = None,
    max_facts: int = 16,
    max_upstream_facts: int = 16,
) -> Dict[str, Any]:
    q_tokens = _tokens(question)
    facts = list(top_k_facts or [])
    if len(facts) < max_facts and candidate_facts:
        facts.extend(candidate_facts[: max_facts - len(facts)])

    upstream_facts: List[Tuple[str, str, str]] = []
    upstream_seen = set()
    for raw_item in upstream_fact_tuples or []:
        raw_fact = raw_item.get("triple") if isinstance(raw_item, dict) else raw_item
        fact = _clean_triple(raw_fact)
        if fact is None or fact in upstream_seen:
            continue
        upstream_seen.add(fact)
        upstream_facts.append(fact)
        if len(upstream_facts) >= max(0, int(max_upstream_facts)):
            break

    endpoint_tokens: Set[str] = set()
    relation_tokens: Set[str] = set()
    entity_keys: Set[str] = set()
    signatures: Set[Tuple[str, str, str]] = set()
    active_triples: List[Tuple[str, str, str]] = []

    base_fact_count = 0
    upstream_fact_count = 0
    for source, raw_fact in (
        [("base", fact) for fact in facts[:max_facts]]
        + [("upstream", fact) for fact in upstream_facts]
    ):
        fact = _clean_triple(raw_fact)
        if fact is None:
            continue
        subject, relation, obj = fact
        subj_tokens = _tokens(subject)
        obj_tokens = _tokens(obj)
        rel_tokens = _tokens(relation)
        fact_tokens = subj_tokens | obj_tokens | rel_tokens
        if q_tokens and not (fact_tokens & q_tokens) and len(active_triples) >= 4:
            continue
        active_triples.append(fact)
        if source == "upstream":
            upstream_fact_count += 1
        else:
            base_fact_count += 1
        endpoint_tokens.update(subj_tokens)
        endpoint_tokens.update(obj_tokens)
        relation_tokens.update(rel_tokens)
        for key in (_entity_key(subject), _entity_key(obj)):
            if key:
                entity_keys.add(key)
        signature = _schema_signature(fact)
        if signature is not None:
            signatures.add(signature)

    for token in q_tokens:
        if len(token) > 2:
            entity_keys.add(token)

    return {
        "query_tokens": q_tokens,
        "endpoint_tokens": endpoint_tokens,
        "relation_tokens": relation_tokens,
        "entity_keys": entity_keys,
        "signatures": signatures,
        "active_triples": active_triples,
        "base_fact_count": base_fact_count,
        "upstream_input_count": len(upstream_fact_tuples or []),
        "upstream_fact_count": upstream_fact_count,
    }


def _schema_bound_semijoin_bindings(
    upstream_fact_tuples: Optional[List[Any]],
    *,
    gap: Optional[Dict[str, Any]] = None,
    question: str = "",
    max_bindings: int = 16,
    enable_terminal_variable_binding: bool = False,
    enable_lineage_directed_terminal_join: bool = False,
    min_query_anchor_overlap: float = 0.67,
    compile_stats: Optional[Counter] = None,
) -> List[Dict[str, Any]]:
    """Compile schema-memory tuples into role-specific semijoin bindings.

    A schema tuple R(a, b) whose schema lookup matched on ``b`` is continued
    through a subject index probe S(b, c).  Keeping the endpoint role is what
    distinguishes a real fact-table join from adding more lexical lookup keys.
    """
    bindings: List[Dict[str, Any]] = []
    gap = dict(gap or {})
    query_anchor_tokens = set(gap.get("query_anchor_tokens") or [])
    seen = set()
    for raw_item in upstream_fact_tuples or []:
        item = raw_item if isinstance(raw_item, dict) else {}
        triple = _clean_triple(item.get("triple") if item else raw_item)
        if triple is None:
            continue
        subject, _, obj = triple
        bound_role = str(
            item.get("candidate_join_endpoint_role")
            or item.get("active_join_role")
            or ""
        ).strip().lower()
        if bound_role not in {"subject", "object"}:
            direction = str(item.get("direction") or "").strip().lower()
            if direction.startswith("object_to_"):
                bound_role = "object"
            elif direction.startswith("subject_to_"):
                bound_role = "subject"
        if bound_role not in {"subject", "object"}:
            continue

        bound_value = subject if bound_role == "subject" else obj
        source_other_value = obj if bound_role == "subject" else subject
        bound_key = _entity_key(bound_value)
        source_other_key = _entity_key(source_other_value)
        if not bound_key:
            continue
        bound_tokens = _tokens(bound_value)
        source_other_tokens = _tokens(source_other_value)
        bound_anchor_overlap = (
            len(bound_tokens & query_anchor_tokens)
            / float(max(1, len(bound_tokens)))
        )
        source_anchor_overlap = (
            len(source_other_tokens & query_anchor_tokens)
            / float(max(1, len(source_other_tokens)))
        )
        anchor_overlap = max(bound_anchor_overlap, source_anchor_overlap)
        if enable_terminal_variable_binding:
            if not query_anchor_tokens:
                if compile_stats is not None:
                    compile_stats["missing_query_anchor_contract"] += 1
                continue
            if anchor_overlap < max(0.0, float(min_query_anchor_overlap)):
                if compile_stats is not None:
                    compile_stats["source_tuple_not_query_bound"] += 1
                continue
        lineage_root_match = _query_root_surface_match(
            question,
            source_other_value,
        )
        if enable_lineage_directed_terminal_join:
            if bound_role != "object":
                if compile_stats is not None:
                    compile_stats["lineage_reverse_source_role"] += 1
                continue
            if not lineage_root_match:
                if compile_stats is not None:
                    compile_stats["lineage_source_not_query_root"] += 1
                continue
        if source_anchor_overlap >= bound_anchor_overlap:
            query_bound_role = "source_other"
            query_bound_key = source_other_key
            query_bound_value = source_other_value
        else:
            query_bound_role = "join_endpoint"
            query_bound_key = bound_key
            query_bound_value = bound_value
        probe_role = "object" if bound_role == "subject" else "subject"
        binding_key = (bound_key, probe_role, "|".join(triple))
        if binding_key in seen:
            continue
        seen.add(binding_key)
        residual_contract = _compile_residual_predicate_contract(
            triple,
            gap,
            question,
        )
        bindings.append({
            "binding_id": len(bindings),
            "bound_key": bound_key,
            "bound_value": str(bound_value),
            "source_bound_role": bound_role,
            "probe_role": probe_role,
            "terminal_role": "subject" if probe_role == "object" else "object",
            "source_other_key": source_other_key,
            "terminal_variable_bound": bool(enable_terminal_variable_binding),
            "lineage_directed": bool(enable_lineage_directed_terminal_join),
            "lineage_direction": (
                "query_root_subject_to_bridge_object_to_terminal_object"
                if enable_lineage_directed_terminal_join
                else "role_preserving_semijoin"
            ),
            "lineage_root_match": bool(lineage_root_match),
            "lineage_root_key": source_other_key,
            "lineage_root_value": str(source_other_value),
            "parent_binding_id": "|".join([
                _entity_key(subject),
                _relation_key(triple[1]),
                _entity_key(obj),
            ]),
            "hop_id": 1,
            "query_bound_role": query_bound_role,
            "query_bound_key": query_bound_key,
            "query_bound_value": str(query_bound_value),
            "query_anchor_overlap": round(float(anchor_overlap), 6),
            "source_other_anchor_overlap": round(
                float(source_anchor_overlap), 6
            ),
            "join_endpoint_anchor_overlap": round(
                float(bound_anchor_overlap), 6
            ),
            "source_tuple_key": "|".join([
                _entity_key(subject),
                _relation_key(triple[1]),
                _entity_key(obj),
            ]),
            "source_triple": list(triple),
            "source_doc_id": item.get("doc_id"),
            "source_title": str(item.get("title") or ""),
            **residual_contract,
        })
        if len(bindings) >= max(0, int(max_bindings)):
            break
    return bindings


def _schema_bound_semijoin_match(
    item: Dict[str, Any],
    bindings: List[Dict[str, Any]],
    gap: Dict[str, Any],
    *,
    enable_residual_predicate_gate: bool = False,
    enable_terminal_slot_gate: bool = False,
    enable_predicate_family_match: bool = False,
    rejection_counts: Optional[Counter] = None,
) -> Optional[Dict[str, Any]]:
    """Return the strongest exact role-preserving binding for an index row."""
    best: Optional[Dict[str, Any]] = None
    relation_tokens = set(item.get("relation_tokens") or set())
    query_relation_tokens = set(gap.get("query_relation_tokens") or [])
    direct_role_score = (
        _bounded_overlap(relation_tokens, query_relation_tokens, max_required=3)
        if query_relation_tokens else 1.0
    )
    family_score = _relation_family_score(relation_tokens, gap)
    relation_role_score = max(direct_role_score, family_score)
    for binding in bindings:
        probe_role = str(binding.get("probe_role") or "")
        if bool(binding.get("lineage_directed")) and (
            str(binding.get("source_bound_role") or "") != "object"
            or probe_role != "subject"
            or str(binding.get("terminal_role") or "") != "object"
            or not bool(binding.get("lineage_root_match"))
        ):
            if rejection_counts is not None:
                rejection_counts["lineage_direction_mismatch"] += 1
            continue
        probe_key = str(item.get(f"{probe_role}_key") or "")
        if not probe_key or probe_key != str(binding.get("bound_key") or ""):
            continue
        if str(item.get("tuple_key") or "") == str(binding.get("source_tuple_key") or ""):
            continue
        terminal_role = str(binding.get("terminal_role") or "")
        terminal_key = str(item.get(f"{terminal_role}_key") or "")
        if not terminal_key or terminal_key in {
            str(binding.get("bound_key") or ""),
            str(binding.get("source_other_key") or ""),
        }:
            continue
        terminal_value = str(item.get(terminal_role) or "").strip()
        if not terminal_value:
            continue
        residual_check = _evaluate_residual_predicate(
            relation_tokens,
            binding,
            enable_predicate_family_match=bool(
                enable_predicate_family_match
            ),
        )
        if enable_residual_predicate_gate and not residual_check["match"]:
            if rejection_counts is not None:
                reason = (
                    "missing_residual_predicate_contract"
                    if not residual_check["has_contract"]
                    else "residual_predicate_mismatch"
                )
                rejection_counts[reason] += 1
            continue
        terminal_answer_type = str(binding.get("terminal_answer_type") or "")
        terminal_slot_match = _terminal_value_matches_contract(
            terminal_value,
            terminal_answer_type,
            relation_tokens,
        )
        if enable_terminal_slot_gate and terminal_answer_type and not terminal_slot_match:
            if rejection_counts is not None:
                rejection_counts["terminal_slot_mismatch"] += 1
            continue
        effective_relation_role_score = max(
            relation_role_score,
            1.0 if residual_check["match"] else 0.0,
        )
        rank_score = (
            1.0
            + 0.35 * effective_relation_role_score
            + 0.10 * float(terminal_key not in set(gap.get("query_anchor_tokens") or []))
        )
        match = {
            **binding,
            "candidate_probe_role": probe_role,
            "terminal_role": terminal_role,
            "terminal_key": terminal_key,
            "terminal_value": terminal_value,
            "relation_role_score": round(float(effective_relation_role_score), 6),
            "direct_relation_score": round(float(direct_role_score), 6),
            "relation_family_score": round(float(family_score), 6),
            "residual_predicate_match": bool(residual_check["match"]),
            "residual_predicate_direct_match": bool(residual_check["direct_match"]),
            "residual_predicate_specific_match": bool(residual_check["specific_match"]),
            "residual_predicate_family_match": bool(residual_check["family_match"]),
            "residual_predicate_canonical_family_match": bool(
                residual_check["predicate_family_match"]
            ),
            "residual_predicate_polarity_match": bool(residual_check["polarity_match"]),
            "candidate_relation_families": residual_check["candidate_relation_families"],
            "candidate_predicate_families": residual_check[
                "candidate_predicate_families"
            ],
            "terminal_answer_type": terminal_answer_type,
            "terminal_slot_match": bool(terminal_slot_match),
            "binding_score": round(float(rank_score), 6),
        }
        if best is None or float(match["binding_score"]) > float(best["binding_score"]):
            best = match
    return best


def _fact_join_replay_signature(
    item: Dict[str, Any],
    binding: Dict[str, Any],
    match: Dict[str, Any],
) -> str:
    """Identify one bound output tuple independently of its access path."""
    return "\t".join([
        str(binding.get("binding_id", 0)),
        str(item.get("tuple_key") or ""),
        str(match.get("terminal_role") or ""),
        str(match.get("terminal_key") or ""),
    ])


def replay_logical_physical_fact_join(
    *,
    bindings: List[Dict[str, Any]],
    relation_index: Dict[str, Any],
    gap: Dict[str, Any],
    enable_terminal_slot_gate: bool = True,
    max_output_rows: int = 32,
) -> Dict[str, Any]:
    """Replay logical and physical fact-join alternatives on one frozen trace.

    The logical contrast keeps the role/entity probe fixed and changes only
    the residual predicate from lexical token equality to canonical predicate
    families.  The physical contrast keeps the canonical predicate fixed and
    compares a full role-index scan with the composite
    ``(role, entity, predicate_family)`` lookup.  No replay output is admitted
    to the production reader-visible pool.
    """
    items = list((relation_index or {}).get("items") or [])
    subject_index = (relation_index or {}).get("subject_to_items") or {}
    object_index = (relation_index or {}).get("object_to_items") or {}
    subject_family_index = (
        (relation_index or {}).get("subject_predicate_family_to_items") or {}
    )
    object_family_index = (
        (relation_index or {}).get("object_predicate_family_to_items") or {}
    )

    lexical_rows: Dict[str, Dict[str, Any]] = {}
    canonical_scan_rows: Dict[str, Dict[str, Any]] = {}
    canonical_index_rows: Dict[str, Dict[str, Any]] = {}
    binding_rows: List[Dict[str, Any]] = []
    role_scan_rows = 0
    composite_scan_rows = 0
    role_lookup_count = 0
    role_hit_count = 0
    composite_lookup_count = 0
    composite_hit_count = 0
    eligible_binding_count = 0
    physical_mismatch_count = 0

    def materialize(
        item_id: int,
        binding: Dict[str, Any],
        *,
        mode: str,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        if item_id < 0 or item_id >= len(items):
            return None
        item = items[item_id]
        match = _schema_bound_semijoin_match(
            item,
            [binding],
            gap,
            enable_residual_predicate_gate=False,
            enable_terminal_slot_gate=bool(enable_terminal_slot_gate),
            enable_predicate_family_match=True,
        )
        if match is None:
            return None
        if mode == "lexical":
            accepted = bool(
                match.get("residual_predicate_direct_match")
                and match.get("residual_predicate_polarity_match", True)
            )
        else:
            accepted = bool(
                match.get("residual_predicate_canonical_family_match")
                and match.get("residual_predicate_polarity_match", True)
            )
        if not accepted:
            return None
        signature = _fact_join_replay_signature(item, binding, match)
        return signature, {
            "signature": signature,
            "binding_id": int(binding.get("binding_id", 0) or 0),
            "tuple_key": str(item.get("tuple_key") or ""),
            "triple": list(item.get("triple") or []),
            "doc_id": int(
                item.get("doc_idx")
                if item.get("doc_idx") is not None
                else -1
            ),
            "probe_role": str(binding.get("probe_role") or ""),
            "bound_key": str(binding.get("bound_key") or ""),
            "terminal_role": str(match.get("terminal_role") or ""),
            "terminal_key": str(match.get("terminal_key") or ""),
            "terminal_value": str(match.get("terminal_value") or ""),
            "candidate_predicate_families": list(
                match.get("candidate_predicate_families") or []
            ),
        }

    for binding in list(bindings or []):
        predicate_families = sorted(set(
            binding.get("residual_predicate_families") or []
        ))
        if not predicate_families:
            binding_rows.append({
                "binding_id": int(binding.get("binding_id", 0) or 0),
                "eligible": False,
                "reason": "no_canonical_predicate_family_contract",
                "probe_role": str(binding.get("probe_role") or ""),
                "bound_key": str(binding.get("bound_key") or ""),
                "source_triple": list(binding.get("source_triple") or []),
            })
            continue

        eligible_binding_count += 1
        probe_role = str(binding.get("probe_role") or "")
        role_index = subject_index if probe_role == "subject" else object_index
        family_index = (
            subject_family_index
            if probe_role == "subject"
            else object_family_index
        )
        bound_key = str(binding.get("bound_key") or "")
        role_lookup_count += 1
        role_ids = list(dict.fromkeys(
            int(item_id) for item_id in list(role_index.get(bound_key) or [])
        ))
        role_scan_rows += len(role_ids)
        role_hit_count += int(bool(role_ids))

        binding_lexical: Dict[str, Dict[str, Any]] = {}
        binding_canonical_scan: Dict[str, Dict[str, Any]] = {}
        for item_id in role_ids:
            lexical = materialize(item_id, binding, mode="lexical")
            if lexical is not None:
                binding_lexical[lexical[0]] = lexical[1]
                lexical_rows[lexical[0]] = lexical[1]
            canonical = materialize(item_id, binding, mode="canonical")
            if canonical is not None:
                binding_canonical_scan[canonical[0]] = canonical[1]
                canonical_scan_rows[canonical[0]] = canonical[1]

        composite_ids: List[int] = []
        for family in predicate_families:
            composite_lookup_count += 1
            composite_key = _role_predicate_index_key(bound_key, family)
            values = list(family_index.get(composite_key) or [])
            composite_hit_count += int(bool(values))
            composite_scan_rows += len(values)
            composite_ids.extend(int(item_id) for item_id in values)
        composite_ids = list(dict.fromkeys(composite_ids))
        binding_canonical_index: Dict[str, Dict[str, Any]] = {}
        for item_id in composite_ids:
            canonical = materialize(item_id, binding, mode="canonical")
            if canonical is not None:
                binding_canonical_index[canonical[0]] = canonical[1]
                canonical_index_rows[canonical[0]] = canonical[1]

        scan_signatures = set(binding_canonical_scan)
        index_signatures = set(binding_canonical_index)
        missing = sorted(scan_signatures - index_signatures)
        extra = sorted(index_signatures - scan_signatures)
        equivalent = scan_signatures == index_signatures
        physical_mismatch_count += int(not equivalent)
        binding_rows.append({
            "binding_id": int(binding.get("binding_id", 0) or 0),
            "eligible": True,
            "reason": "replayed",
            "probe_role": probe_role,
            "bound_key": bound_key,
            "source_triple": list(binding.get("source_triple") or []),
            "residual_predicate_families": predicate_families,
            "role_scan_rows": len(role_ids),
            "composite_scan_rows": sum(
                len(family_index.get(_role_predicate_index_key(bound_key, family)) or [])
                for family in predicate_families
            ),
            "lexical_output_rows": len(binding_lexical),
            "canonical_output_rows": len(binding_canonical_scan),
            "composite_output_rows": len(binding_canonical_index),
            "physical_exact_tuple_equivalent": equivalent,
            "physical_missing_signatures": missing[:8],
            "physical_extra_signatures": extra[:8],
        })

    lexical_signatures = set(lexical_rows)
    canonical_scan_signatures = set(canonical_scan_rows)
    canonical_index_signatures = set(canonical_index_rows)
    logical_added = sorted(canonical_scan_signatures - lexical_signatures)
    logical_removed = sorted(lexical_signatures - canonical_scan_signatures)
    union = lexical_signatures | canonical_scan_signatures
    physical_eligible = eligible_binding_count > 0
    physical_equivalent = bool(
        physical_eligible
        and physical_mismatch_count == 0
        and canonical_scan_signatures == canonical_index_signatures
    )
    scan_reduction = (
        1.0 - (float(composite_scan_rows) / float(role_scan_rows))
        if role_scan_rows > 0 else 0.0
    )
    index_reference_count = sum(
        len(values) for values in subject_family_index.values()
    ) + sum(len(values) for values in object_family_index.values())
    max_rows = max(0, int(max_output_rows))

    return {
        "enabled": True,
        "executed": True,
        "shadow_only": True,
        "trace_version": "single_trace_logical_physical_fact_join_v1",
        "binding_count": len(bindings or []),
        "eligible_binding_count": eligible_binding_count,
        "ineligible_binding_count": max(0, len(bindings or []) - eligible_binding_count),
        "logical_rewrite": {
            "baseline": "lexical_residual_predicate_over_role_scan",
            "candidate": "canonical_predicate_family_over_role_scan",
            "lexical_output_rows": len(lexical_signatures),
            "canonical_output_rows": len(canonical_scan_signatures),
            "added_tuple_count": len(logical_added),
            "removed_tuple_count": len(logical_removed),
            "result_changed": lexical_signatures != canonical_scan_signatures,
            "tuple_jaccard": round(
                len(lexical_signatures & canonical_scan_signatures)
                / float(max(1, len(union))),
                6,
            ),
            "added_signatures": logical_added[:max_rows],
            "removed_signatures": logical_removed[:max_rows],
            "lexical_rows": [
                lexical_rows[key] for key in sorted(lexical_rows)[:max_rows]
            ],
            "canonical_rows": [
                canonical_scan_rows[key]
                for key in sorted(canonical_scan_rows)[:max_rows]
            ],
            "outputs_truncated": bool(
                len(lexical_rows) > max_rows or len(canonical_scan_rows) > max_rows
            ),
        },
        "physical_rewrite": {
            "baseline": "canonical_predicate_family_role_index_scan",
            "candidate": "canonical_role_entity_predicate_family_lookup",
            "eligible": physical_eligible,
            "status": (
                "exact_tuple_equivalent"
                if physical_equivalent else
                "tuple_mismatch"
                if physical_eligible else
                "no_indexable_contract"
            ),
            "exact_tuple_equivalent": physical_equivalent,
            "mismatched_binding_count": physical_mismatch_count,
            "role_index_lookup_count": role_lookup_count,
            "role_index_hit_count": role_hit_count,
            "composite_index_lookup_count": composite_lookup_count,
            "composite_index_hit_count": composite_hit_count,
            "role_scan_rows": role_scan_rows,
            "composite_scan_rows": composite_scan_rows,
            "scan_row_reduction": role_scan_rows - composite_scan_rows,
            "scan_reduction_rate": round(scan_reduction, 6),
            "role_scan_output_rows": len(canonical_scan_signatures),
            "composite_output_rows": len(canonical_index_signatures),
            "missing_signatures": sorted(
                canonical_scan_signatures - canonical_index_signatures
            )[:max_rows],
            "extra_signatures": sorted(
                canonical_index_signatures - canonical_scan_signatures
            )[:max_rows],
        },
        "index_footprint": {
            "fact_row_count": len(items),
            "composite_key_count": len(subject_family_index) + len(object_family_index),
            "composite_row_reference_count": index_reference_count,
            "references_per_fact_row": round(
                index_reference_count / float(max(1, len(items))),
                6,
            ),
        },
        "bindings": binding_rows[:max_rows],
        "bindings_truncated": len(binding_rows) > max_rows,
    }


def estimate_relation_fact_join_access(
    *,
    question: str,
    top_k_facts: List[Tuple],
    candidate_facts: Optional[List[Tuple]],
    upstream_fact_tuples: Optional[List[Any]] = None,
    relation_index: Dict[str, Any],
    max_facts_per_key: int = 60,
    max_key_fanout: int = 600,
) -> Dict[str, Any]:
    """Estimate rows touched by the same inverted indexes used at execution."""
    active = _active_fact_signals(
        question,
        top_k_facts,
        candidate_facts,
        upstream_fact_tuples=upstream_fact_tuples,
    )
    relation_keys = set(active["relation_tokens"]) | {
        token for token in active["query_tokens"] if len(token) > 3
    }
    access_paths = (
        ("entity", active["entity_keys"], relation_index.get("entity_to_items") or {}),
        ("relation", relation_keys, relation_index.get("relation_to_items") or {}),
        ("schema", active["signatures"], relation_index.get("schema_to_items") or {}),
    )
    estimate = estimate_inverted_index_access(
        access_paths,
        per_key_cap=max_facts_per_key,
        max_key_fanout=max_key_fanout,
    )
    return {
        **estimate,
        "active_entity_key_count": len(active["entity_keys"]),
        "active_relation_key_count": len(relation_keys),
        "active_schema_key_count": len(active["signatures"]),
        "upstream_bind_input_count": int(active.get("upstream_input_count", 0)),
        "upstream_bind_fact_count": int(active.get("upstream_fact_count", 0)),
    }


def _evidence_gap_profile(
    question: str,
    current_docs: List[str],
    qa_top_k: int,
    min_query_coverage: float,
) -> Dict[str, Any]:
    q_tokens = _tokens(question)
    doc_hits = 0
    covered: Set[str] = set()
    for doc in current_docs[: max(1, qa_top_k)]:
        doc_tokens = _tokens(str(doc)[:1800])
        if not doc_tokens:
            continue
        covered.update(q_tokens & doc_tokens)
        if _overlap_over_query(doc_tokens, q_tokens) >= 0.06:
            doc_hits += 1
    query_coverage = len(covered) / max(1, len(q_tokens))
    bridge_cues = bool(BRIDGE_CUE_RE.search(str(question or "")))
    evidence_gap = query_coverage < min_query_coverage or (bridge_cues and doc_hits < 2)
    return {
        "bridge_cues": bridge_cues,
        "doc_hits": doc_hits,
        "query_coverage": round(float(query_coverage), 6),
        "evidence_gap": bool(evidence_gap),
    }


def _answerability_gap_profile(
    question: str,
    current_docs: List[str],
    qa_top_k: int,
    min_query_coverage: float,
) -> Dict[str, Any]:
    base = _evidence_gap_profile(
        question=question,
        current_docs=current_docs,
        qa_top_k=qa_top_k,
        min_query_coverage=min_query_coverage,
    )
    profile = _relation_profile(question)
    doc_tokens: Set[str] = set()
    doc_text_parts: List[str] = []
    for doc in current_docs[: max(1, qa_top_k)]:
        text = str(doc or "")[:2200]
        doc_text_parts.append(text)
        doc_tokens.update(_tokens(text))

    required_relation_tokens = set(profile["relation_tokens"])
    query_anchor_tokens = set(profile["anchor_tokens"])
    answer_types = set(profile["answer_types"])

    relation_role_coverage = (
        _bounded_overlap(doc_tokens, required_relation_tokens, max_required=3)
        if required_relation_tokens else 1.0
    )
    anchor_coverage = (
        _bounded_overlap(doc_tokens, query_anchor_tokens, max_required=4)
        if query_anchor_tokens else 0.0
    )
    joined_doc_text = "\n".join(doc_text_parts)
    answer_type_coverage = 1.0
    if "date" in answer_types:
        answer_type_coverage = 1.0 if DATE_VALUE_RE.search(joined_doc_text) else 0.0
    elif "number" in answer_types:
        answer_type_coverage = 1.0 if NUMBER_VALUE_RE.search(joined_doc_text) else 0.0

    polarity_gap = False
    doc_polarities = {
        name for name in profile["polarity"] if _tokens_have_polarity(doc_tokens, name)
    }
    for required_polarity in profile["polarity"]:
        if required_polarity not in doc_polarities:
            polarity_gap = True

    relation_gap = bool(
        profile["has_relation_need"]
        and (
            relation_role_coverage < 0.50
            or polarity_gap
            or (query_anchor_tokens and anchor_coverage < 0.25)
        )
    )
    bridge_gap = bool(base["bridge_cues"] and (base["doc_hits"] < 2 or relation_role_coverage < 0.34))
    type_gap = bool(answer_types & {"date", "number"} and answer_type_coverage < 0.5)
    answerability_gap = bool(base["evidence_gap"] or relation_gap or bridge_gap or type_gap)

    base.update({
        "answerability_gap": answerability_gap,
        "relation_gap": relation_gap,
        "bridge_gap": bridge_gap,
        "type_gap": type_gap,
        "polarity_gap": polarity_gap,
        "relation_role_coverage": round(float(relation_role_coverage), 6),
        "anchor_coverage": round(float(anchor_coverage), 6),
        "answer_type_coverage": round(float(answer_type_coverage), 6),
        "query_relation_tokens": profile["relation_tokens"],
        "query_relation_families": profile["relation_families"],
        "query_polarity": profile["polarity"],
        "query_answer_types": profile["answer_types"],
        "query_anchor_tokens": profile["anchor_tokens"][:12],
        "query_anchor_count": len(profile["anchor_tokens"]),
        "has_relation_need": bool(profile["has_relation_need"]),
    })
    return base


def _cost_decision(
    *,
    question: str,
    content: str,
    item: Dict[str, Any],
    rank: int,
    raw_candidate_count: int,
    candidate_count: int,
    evidence_gap: bool,
    utility: float,
    risk: float,
    token_budget: int,
    min_utility: float,
    max_risk: float,
    min_net_utility: float,
    token_weight: float,
    compute_weight: float,
    risk_weight: float,
) -> Dict[str, Any]:
    added_tokens = _estimate_tokens(content)
    token_cost = _clamp(added_tokens / max(1.0, float(token_budget)))
    compute_cost = _clamp(
        0.03
        + 0.0015 * min(max(0, raw_candidate_count), 300)
        + 0.004 * min(max(0, candidate_count), 80)
    )
    net_utility = float(utility) - risk_weight * float(risk) - token_weight * token_cost - compute_weight * compute_cost
    reasons = []
    if utility < min_utility:
        reasons.append("utility_below_threshold")
    if risk > max_risk:
        reasons.append("risk_above_threshold")
    if net_utility < min_net_utility:
        reasons.append("net_utility_below_threshold")
    accepted = not reasons
    return {
        "operator": "relation_aware_fact_join_retrieval",
        "accepted": bool(accepted),
        "reason": "accepted" if accepted else "+".join(reasons),
        "utility": round(float(utility), 6),
        "risk": round(float(risk), 6),
        "token_cost": round(float(token_cost), 6),
        "compute_cost": round(float(compute_cost), 6),
        "net_utility": round(float(net_utility), 6),
        "added_tokens": int(added_tokens),
        "rank": int(rank),
        "raw_candidate_count": int(raw_candidate_count),
        "candidate_count": int(candidate_count),
        "evidence_gap": bool(evidence_gap),
        "triple": list(item.get("triple") or []),
        "thresholds": {
            "min_utility": round(float(min_utility), 6),
            "max_risk": round(float(max_risk), 6),
            "min_net_utility": round(float(min_net_utility), 6),
            "token_budget": int(token_budget),
            "token_weight": round(float(token_weight), 6),
            "compute_weight": round(float(compute_weight), 6),
            "risk_weight": round(float(risk_weight), 6),
        },
    }


def _admit_slot_dependency_reader_rows(
    rows: List[Dict[str, Any]],
    *,
    question: str,
    visible_doc_ids: Iterable[int],
    get_doc_content: Callable[[int], str],
    evidence_gap: bool,
    raw_candidate_count: int,
    candidate_count: int,
    token_budget: int,
    min_utility: float,
    max_risk: float,
    min_net_utility: float,
    token_weight: float,
    compute_weight: float,
    risk_weight: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply one risk/cost contract to compiled rows before reader visibility."""
    compiled_versions = {
        "slot_dependency_query_plan_v3",
        "schema_constrained_query_plan_v4",
    }
    visible_ids = {int(value) for value in visible_doc_ids}
    visible_parts = []
    for doc_id in visible_ids:
        try:
            visible_parts.append(str(get_doc_content(doc_id) or ""))
        except Exception:
            continue
    visible_text = " ".join(visible_parts).casefold()
    admitted: List[Dict[str, Any]] = []
    reasons: Counter = Counter()
    checked = 0
    for raw_row in rows:
        row = dict(raw_row)
        if str(row.get("query_plan_version") or "") not in compiled_versions:
            admitted.append(row)
            continue
        checked += 1
        source_doc_ids = {
            int(value)
            for value in (row.get("source_doc_ids") or [])
            if str(value).lstrip("-").isdigit()
        }
        novel_doc_ids = sorted(source_doc_ids - visible_ids)
        candidate_surface = str(row.get("candidate") or "").strip()
        candidate_in_visible = bool(
            candidate_surface
            and candidate_surface.casefold() in visible_text
        )
        support_only = bool(row.get("support_only"))
        non_redundant = bool(novel_doc_ids) and (
            support_only or not candidate_in_visible
        )
        binding_confidence = _clamp(
            row.get("root_binding_confidence", 1.0) or 0.0
        )
        reliability = _clamp(
            (row.get("fact_reliability") or {}).get("score", 0.85) or 0.0
        )
        completeness = 1.0 if (
            row.get("rooted_lineage_complete")
            and row.get("operator_complete")
        ) else 0.0
        utility = _clamp(
            0.72 * float(row.get("table_score", 0.0) or 0.0)
            + 0.12 * float(bool(evidence_gap))
            + 0.10 * float(non_redundant)
            + 0.06 * completeness
        )
        risk = _clamp(
            0.40 * (1.0 - binding_confidence)
            + 0.35 * (1.0 - reliability)
            + 0.20 * (1.0 - completeness)
            + 0.05 * float(not non_redundant)
        )
        decision = _cost_decision(
            question=question,
            content=str(row.get("evidence") or candidate_surface),
            item={"triple": [str(row.get("triple") or "")]},
            rank=int(row.get("rank", 999999) or 999999),
            raw_candidate_count=int(raw_candidate_count),
            candidate_count=int(candidate_count),
            evidence_gap=bool(evidence_gap),
            utility=utility,
            risk=risk,
            token_budget=int(token_budget),
            min_utility=float(min_utility),
            max_risk=float(max_risk),
            min_net_utility=float(min_net_utility),
            token_weight=float(token_weight),
            compute_weight=float(compute_weight),
            risk_weight=float(risk_weight),
        )
        decision["operator"] = "slot_dependency_fact_join_materialization"
        decision.update({
            "non_redundant": bool(non_redundant),
            "candidate_already_visible": bool(candidate_in_visible),
            "novel_source_doc_ids": novel_doc_ids,
            "binding_confidence": round(binding_confidence, 6),
            "provenance_reliability": round(reliability, 6),
            "operator_complete": bool(completeness),
        })
        if not non_redundant:
            decision["accepted"] = False
            decision["reason"] = "reader_evidence_redundant"
        row["cost_model"] = decision
        row["cost_accepted"] = bool(decision.get("accepted"))
        row["cost_reason"] = str(decision.get("reason") or "")
        row["net_utility"] = float(decision.get("net_utility", 0.0) or 0.0)
        if decision.get("accepted"):
            admitted.append(row)
        else:
            reasons[str(decision.get("reason") or "rejected")] += 1
    return admitted, {
        "enabled": True,
        "checked_row_count": int(checked),
        "input_row_count": len(rows),
        "output_row_count": len(admitted),
        "accepted_row_count": sum(
            str(row.get("query_plan_version") or "") in compiled_versions
            for row in admitted
        ),
        "rejected_row_count": int(checked) - sum(
            str(row.get("query_plan_version") or "") in compiled_versions
            for row in admitted
        ),
        "reason_counts": dict(reasons),
    }


def _verify_relation_join_candidate(
    *,
    item: Dict[str, Any],
    content: str,
    gap: Dict[str, Any],
    doc_tokens: Set[str],
    require_relation_role_match: bool,
    require_anchor_match: bool,
    min_anchor_overlap: float,
    min_role_overlap: float,
) -> Dict[str, Any]:
    endpoint_tokens = set(item.get("endpoint_tokens") or set())
    relation_tokens = set(item.get("relation_tokens") or set())
    query_relation_tokens = set(gap.get("query_relation_tokens") or [])
    query_anchor_tokens = set(gap.get("query_anchor_tokens") or [])
    query_polarities = set(gap.get("query_polarity") or [])
    answer_types = set(gap.get("query_answer_types") or [])

    relation_role_overlap = (
        _bounded_overlap(relation_tokens | doc_tokens, query_relation_tokens, max_required=3)
        if query_relation_tokens else 1.0
    )
    bound_match = dict(item.get("schema_bound_semijoin") or {})
    if bound_match:
        relation_role_overlap = max(
            relation_role_overlap,
            _relation_family_score(relation_tokens, gap),
            float(bound_match.get("relation_role_score", 0.0) or 0.0),
        )
    anchor_overlap = (
        _bounded_overlap(endpoint_tokens, query_anchor_tokens, max_required=3)
        if query_anchor_tokens else 0.0
    )
    doc_anchor_overlap = (
        _bounded_overlap(doc_tokens, query_anchor_tokens, max_required=3)
        if query_anchor_tokens else 0.0
    )

    reasons: List[str] = []
    if require_relation_role_match and query_relation_tokens and relation_role_overlap < float(min_role_overlap):
        reasons.append("relation_role_mismatch")

    for required_polarity in query_polarities:
        opposite = _opposite_polarities(required_polarity)
        if opposite and any(_tokens_have_polarity(relation_tokens, polarity) for polarity in opposite):
            reasons.append("relation_polarity_mismatch")
            break
        if not _tokens_have_polarity(relation_tokens | doc_tokens, required_polarity):
            reasons.append("relation_polarity_missing")
            break

    if bound_match:
        anchor_overlap = 1.0
        doc_anchor_overlap = max(doc_anchor_overlap, 1.0)
    elif require_anchor_match and query_anchor_tokens:
        if anchor_overlap < float(min_anchor_overlap) and doc_anchor_overlap < 0.67:
            reasons.append("anchor_mismatch")

    triple = item.get("triple") or ("", "", "")
    answer_value = (
        str(bound_match.get("terminal_value") or "")
        if bound_match else
        triple[2] if isinstance(triple, (list, tuple)) and len(triple) == 3 else ""
    )
    terminal_answer_type = str(
        bound_match.get("terminal_answer_type") or ""
    ).strip().lower()
    if bound_match and terminal_answer_type:
        if not _terminal_value_matches_contract(
            answer_value,
            terminal_answer_type,
            relation_tokens,
        ):
            reasons.append("answer_type_mismatch")
    elif not _object_matches_answer_type(answer_value, answer_types, content):
        reasons.append("answer_type_mismatch")

    return {
        "accepted": not reasons,
        "reason": "accepted" if not reasons else "+".join(reasons),
        "relation_role_overlap": round(float(relation_role_overlap), 6),
        "anchor_overlap": round(float(anchor_overlap), 6),
        "doc_anchor_overlap": round(float(doc_anchor_overlap), 6),
        "query_relation_tokens": sorted(query_relation_tokens),
        "query_polarity": sorted(query_polarities),
        "query_answer_types": sorted(answer_types),
        "schema_bound_semijoin": bool(bound_match),
        "schema_binding_id": int(bound_match.get("binding_id", 0) or 0) if bound_match else None,
        "terminal_value": str(answer_value or ""),
    }


def _build_validated_terminal_projection_rows(
    rows: Sequence[Dict[str, Any]],
    plan: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize complete plan rows into the materializer's terminal contract."""
    compiled_plan = dict(plan or {})
    if (
        compiled_plan.get("compile_status") != "compiled"
        or str(compiled_plan.get("answer_operator") or "") != "project"
    ):
        return []

    terminal_operators = [
        dict(operator)
        for operator in (compiled_plan.get("operators") or [])
        if str(operator.get("output_slot") or "") == "$answer"
        and str(operator.get("slot_plan_stage") or "") == "final_answer"
    ]
    if not terminal_operators:
        return []
    terminal_operator = terminal_operators[-1]
    terminal_predicates = {
        str(value)
        for value in (terminal_operator.get("predicate_families") or [])
        if str(value)
    }
    final_answer_type = str(
        terminal_operator.get("output_type")
        or compiled_plan.get("final_answer_type")
        or ""
    )
    plan_version = str(compiled_plan.get("version") or "")

    projected_rows: List[Dict[str, Any]] = []
    seen = set()
    for source_row in rows or []:
        row = dict(source_row or {})
        candidate = str(
            row.get("candidate") or row.get("raw_candidate") or ""
        ).strip()
        evidence = str(row.get("evidence") or "").strip()
        if not candidate or not evidence:
            continue
        if plan_version and str(row.get("query_plan_version") or "") != plan_version:
            continue
        if str(row.get("slot_plan_stage") or "") != "final_answer":
            continue
        if not bool(row.get("operator_complete")):
            continue
        if not bool(row.get("slot_dependency_validated")):
            continue
        if not bool(row.get("rooted_lineage_complete")):
            continue
        if not bool(row.get("production_eligible")):
            continue
        if not bool(row.get("final_predicate_match")):
            continue
        if not bool(row.get("final_answer_type_match")):
            continue
        if not bool(row.get("root_identity_safe_for_commit", True)):
            continue
        row_answer_type = str(row.get("final_answer_type") or "")
        if final_answer_type and row_answer_type not in {"", final_answer_type}:
            continue

        row_predicates = {
            str(value)
            for value in (row.get("candidate_predicate_families") or [])
            if str(value)
        }


        if not row_predicates:
            row_predicates = set(terminal_predicates)
        if terminal_predicates and not (terminal_predicates & row_predicates):
            continue

        dedup_key = (
            candidate.casefold(),
            evidence.casefold(),
            str(row.get("lineage_root_key") or ""),
        )
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        row.update({
            "terminal_projection_contract_version": (
                "validated_terminal_projection_v12_2"
            ),
            "terminal_operator_id": str(
                terminal_operator.get("operator_id") or ""
            ),
            "terminal_input_slot": str(
                terminal_operator.get("input_slot") or ""
            ),
            "terminal_output_slot": str(
                terminal_operator.get("output_slot") or ""
            ),
            "terminal_predicate_families": sorted(terminal_predicates),
            "candidate_predicate_families": sorted(row_predicates),
        })
        projected_rows.append(row)
    return projected_rows


def select_relation_fact_join_evidence(
    *,
    question: str,
    sorted_doc_ids: Any,
    sorted_doc_scores: Any,
    top_k_facts: List[Tuple],
    candidate_facts: Optional[List[Tuple]],
    upstream_fact_tuples: Optional[List[Any]],
    relation_index: Dict[str, Any],
    passage_node_keys: List[str],
    get_doc_content: Callable[[int], str],
    qa_top_k: int,
    enabled: bool,
    bridge_only: bool,
    require_evidence_gap: bool,
    candidate_top_k: int,
    max_append: int,
    max_facts_per_key: int,
    max_key_fanout: int,
    protect_top_k: int,
    min_query_overlap: float,
    min_relation_overlap: float,
    min_utility: float,
    max_risk: float,
    min_net_utility: float,
    token_budget: int,
    token_weight: float,
    compute_weight: float,
    risk_weight: float,
    use_answerability_gap: bool = True,
    require_relation_role_match: bool = True,
    require_anchor_match: bool = True,
    min_anchor_overlap: float = 0.34,
    min_role_overlap: float = 0.34,
    enable_candidate_table: bool = True,
    candidate_table_max_rows: int = 4,
    candidate_table_min_score: float = 0.56,
    hard_anchor_min_overlap: float = 0.67,
    enable_multi_hop_planner: bool = True,
    multi_hop_min_bridge_overlap: float = 0.67,
    enable_surface_materializer: bool = True,
    enable_provenance_optimizer: bool = True,
    provenance_conflict_min_margin: float = 0.04,
    provenance_min_support_docs: int = 2,
    provenance_min_reliability: float = 0.72,
    provenance_max_bonus: float = 0.025,
    provenance_shadow_only: bool = False,
    enable_schema_bound_semijoin: bool = False,
    schema_bound_semijoin_fallback: bool = True,
    enable_schema_bound_residual_gate: bool = False,
    enable_schema_bound_terminal_slot_gate: bool = False,
    enable_schema_bound_predicate_family_index: bool = False,
    enable_terminal_variable_bound_fact_join: bool = False,
    terminal_variable_min_anchor_overlap: float = 0.67,
    enable_lineage_directed_terminal_join: bool = False,
    enable_single_trace_fact_join_replay: bool = False,
    enable_stage_aware_fact_join_plan: bool = False,
    enable_role_directed_fact_join: bool = True,
    enable_canonical_bound_completion: bool = True,
    enable_completion_selectivity_feedback: bool = True,
    enable_query_plan_root_contract: bool = True,
    enable_inverse_relation_completion: bool = True,
    enable_title_grounded_completion: bool = False,
    enable_source_grounded_terminal_repair: bool = False,
    enable_title_catalog_root_plan: bool = False,
    enable_title_root_plan_commit: bool = False,
    enable_identity_safe_root_commit: bool = False,
    relation_fact_join_reader_stage_mode: str = "legacy",
    enable_slot_dependency_cost_admission: bool = True,
    enable_bound_completion_fact_join: bool = False,
    bound_completion_lookup: Optional[
        Callable[[str, int], List[Dict[str, Any]]]
    ] = None,
    bound_completion_top_k: int = 6,
    bound_completion_max_calls: int = 2,
    enable_bound_completion_cost_admission: bool = True,
    bound_completion_max_risk: float = 0.42,
    bound_completion_min_net_utility: float = 0.20,
    bound_completion_materialize_reader: bool = False,
    enable_provenance_witness_repair: bool = False,
    provenance_witness_top_k: int = 6,
    provenance_witness_max_calls: int = 2,
) -> Tuple[List[int], Dict[str, Any]]:
    stage_reader_mode = str(
        relation_fact_join_reader_stage_mode or "legacy"
    ).strip().lower()
    valid_stage_reader_modes = {
        "legacy", "control", "bridge_only", "final_slot_validated",
    }
    if stage_reader_mode not in valid_stage_reader_modes:
        stage_reader_mode = "legacy"
    if not enable_stage_aware_fact_join_plan:
        stage_reader_mode = "legacy"
    boolean_answer_contract = bool(BOOLEAN_QUESTION_RE.match(question or ""))
    replay_upstream_fact_tuples = list(upstream_fact_tuples or [])
    production_upstream_fact_tuples = (
        replay_upstream_fact_tuples if enable_schema_bound_semijoin else []
    )
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled" if not enabled else "not_evaluated",
        "bridge_only": bool(bridge_only),
        "answer_contract": "boolean" if boolean_answer_contract else "open_value",
        "support_only": bool(boolean_answer_contract),
        "role_directed_fact_join_enabled": bool(
            enable_role_directed_fact_join
        ),
        "canonical_bound_completion_enabled": bool(
            enable_canonical_bound_completion
        ),
        "completion_selectivity_feedback_enabled": bool(
            enable_completion_selectivity_feedback
        ),
        "query_plan_root_contract_enabled": bool(
            enable_query_plan_root_contract
        ),
        "inverse_relation_completion_enabled": bool(
            enable_inverse_relation_completion
        ),
        "title_grounded_completion_enabled": bool(
            enable_title_grounded_completion
        ),
        "source_grounded_terminal_repair_enabled": bool(
            enable_source_grounded_terminal_repair
        ),
        "title_catalog_root_plan_enabled": bool(
            enable_title_catalog_root_plan
        ),
        "title_root_plan_commit_enabled": bool(
            enable_title_root_plan_commit
        ),
        "identity_safe_root_commit_enabled": bool(
            enable_identity_safe_root_commit
        ),
        "require_evidence_gap": bool(require_evidence_gap),
        "index_stats": dict((relation_index or {}).get("stats", {}) or {}),
        "bridge_cues": False,
        "evidence_gap": False,
        "answerability_gap": False,
        "relation_gap": False,
        "bridge_gap": False,
        "type_gap": False,
        "polarity_gap": False,
        "query_coverage": 0.0,
        "relation_role_coverage": 0.0,
        "anchor_coverage": 0.0,
        "answer_type_coverage": 0.0,
        "use_answerability_gap": bool(use_answerability_gap),
        "require_relation_role_match": bool(require_relation_role_match),
        "require_anchor_match": bool(require_anchor_match),
        "min_anchor_overlap": round(float(min_anchor_overlap), 6),
        "min_role_overlap": round(float(min_role_overlap), 6),
        "enable_candidate_table": bool(enable_candidate_table),
        "candidate_table_max_rows": int(candidate_table_max_rows),
        "candidate_table_min_score": round(float(candidate_table_min_score), 6),
        "hard_anchor_min_overlap": round(float(hard_anchor_min_overlap), 6),
        "enable_multi_hop_planner": bool(enable_multi_hop_planner),
        "multi_hop_min_bridge_overlap": round(float(multi_hop_min_bridge_overlap), 6),
        "enable_surface_materializer": bool(enable_surface_materializer),
        "provenance_min_support_docs": int(provenance_min_support_docs),
        "provenance_min_reliability": round(float(provenance_min_reliability), 6),
        "provenance_max_bonus": round(float(provenance_max_bonus), 6),
        "provenance_shadow_only": bool(provenance_shadow_only),
        "stage_aware_fact_join": {
            "enabled": bool(enable_stage_aware_fact_join_plan),
            "reader_stage_mode": stage_reader_mode,
            "plan": {},
            "bridge_candidate_count": 0,
            "final_slot_candidate_count": 0,
            "support_only_candidate_count": 0,
            "reader_row_count": 0,
        },
        "multi_hop_join": {},
        "semantic_plan_execution": {
            "required": False,
            "completed": False,
            "status": "plan_not_compiled",
            "terminal_binding_count": 0,
        },
        "provenance_optimizer": {},
        "slot_dependency_cost_admission": {
            "enabled": bool(enable_slot_dependency_cost_admission),
            "checked_row_count": 0,
            "input_row_count": 0,
            "output_row_count": 0,
            "accepted_row_count": 0,
            "rejected_row_count": 0,
            "reason_counts": {},
        },
        "bound_completion_fact_join": {
            "enabled": bool(enable_bound_completion_fact_join),
            "materialize_reader": bool(bound_completion_materialize_reader),
            "cost_admission_enabled": bool(
                enable_bound_completion_cost_admission
            ),
            "top_k": int(bound_completion_top_k),
            "max_calls": int(bound_completion_max_calls),
            "max_risk": round(float(bound_completion_max_risk), 6),
            "min_net_utility": round(
                float(bound_completion_min_net_utility), 6
            ),
        },
        "provenance_witness_repair": {
            "enabled": bool(enable_provenance_witness_repair),
            "top_k": int(provenance_witness_top_k),
            "max_calls": int(provenance_witness_max_calls),
            "input_row_count": 0,
            "eligible_row_count": 0,
            "call_count": 0,
            "doc_count": 0,
            "extracted_fact_count": 0,
            "tuple_match_count": 0,
            "independent_witness_count": 0,
            "repaired_row_count": 0,
            "reason_counts": {},
        },
        "active_anchor_count": 0,
        "active_relation_count": 0,
        "active_schema_count": 0,
        "query_anchor_count": 0,
        "upstream_bind_input_count": len(production_upstream_fact_tuples),
        "upstream_bind_fact_count": 0,
        "upstream_bind_active_triples": [],
        "schema_bound_semijoin": {
            "enabled": bool(enable_schema_bound_semijoin),
            "applied": False,
            "reason": "no_upstream_binding",
            "binding_count": 0,
            "probe_key_count": 0,
            "matched_candidate_count": 0,
            "accepted_count": 0,
            "table_row_count": 0,
            "reader_row_count": 0,
            "fallback_enabled": bool(schema_bound_semijoin_fallback),
            "fallback_used": False,
            "fallback_state_restored": False,
            "residual_predicate_gate_enabled": bool(
                enable_schema_bound_residual_gate
            ),
            "terminal_slot_gate_enabled": bool(
                enable_schema_bound_terminal_slot_gate
            ),
            "predicate_family_index_enabled": bool(
                enable_schema_bound_predicate_family_index
            ),
            "terminal_variable_binding_enabled": bool(
                enable_terminal_variable_bound_fact_join
            ),
            "terminal_variable_min_anchor_overlap": round(
                float(terminal_variable_min_anchor_overlap), 6
            ),
            "terminal_variable_binding_count": 0,
            "lineage_directed_terminal_join_enabled": bool(
                enable_lineage_directed_terminal_join
            ),
            "lineage_directed_binding_count": 0,
            "binding_compile_reject_reason_counts": {},
            "probe_executed": False,
            "probe_hit": False,
            "role_index_candidate_count": 0,
            "composite_index_lookup_count": 0,
            "composite_index_hit_count": 0,
            "residual_predicate_match_count": 0,
            "terminal_slot_match_count": 0,
            "residual_predicate_reject_count": 0,
            "terminal_slot_reject_count": 0,
            "gate_reject_reason_counts": {},
        },
        "single_trace_fact_join_replay": {
            "enabled": bool(enable_single_trace_fact_join_replay),
            "executed": False,
            "shadow_only": True,
            "reason": "awaiting_frozen_execution_state",
        },
        "counterfactual_reader_intervention": {
            "enabled": True,
            "eligible": False,
            "reason": "no_directional_candidate",
            "row": {},
            "candidate_table_doc": "",
        },
        "raw_candidate_count": 0,
        "candidate_count": 0,
        "accepted_count": 0,
        "reader_candidate_doc_ids": [],
        "reader_candidate_titles": [],
        "accepted": [],
        "rejected": [],
        "candidate_table_rows": [],
        "candidate_table_row_count": 0,
        "terminal_candidate_rows": [],
        "validated_terminal_candidate_rows": [],
        "validated_terminal_projection_rows": [],
        "terminal_candidate_count": 0,
        "validated_terminal_candidate_count": 0,
        "validated_terminal_projection_row_count": 0,
        "candidate_table_doc": "",
        "reject_reason_counts": {},
        "index_access": _new_index_access_stats(),
    }

    def fallback_to_generic_join(
        semijoin_attempt: Dict[str, Any],
        reason: str,
    ) -> Tuple[List[int], Dict[str, Any]]:
        fallback_doc_ids, fallback_info = select_relation_fact_join_evidence(
            question=question,
            sorted_doc_ids=sorted_doc_ids,
            sorted_doc_scores=sorted_doc_scores,
            top_k_facts=top_k_facts,
            candidate_facts=candidate_facts,
            upstream_fact_tuples=[],
            relation_index=relation_index,
            passage_node_keys=passage_node_keys,
            get_doc_content=get_doc_content,
            qa_top_k=qa_top_k,
            enabled=enabled,
            bridge_only=bridge_only,
            require_evidence_gap=require_evidence_gap,
            candidate_top_k=candidate_top_k,
            max_append=max_append,
            max_facts_per_key=max_facts_per_key,
            max_key_fanout=max_key_fanout,
            protect_top_k=protect_top_k,
            min_query_overlap=min_query_overlap,
            min_relation_overlap=min_relation_overlap,
            min_utility=min_utility,
            max_risk=max_risk,
            min_net_utility=min_net_utility,
            token_budget=token_budget,
            token_weight=token_weight,
            compute_weight=compute_weight,
            risk_weight=risk_weight,
            use_answerability_gap=use_answerability_gap,
            require_relation_role_match=require_relation_role_match,
            require_anchor_match=require_anchor_match,
            min_anchor_overlap=min_anchor_overlap,
            min_role_overlap=min_role_overlap,
            enable_candidate_table=enable_candidate_table,
            candidate_table_max_rows=candidate_table_max_rows,
            candidate_table_min_score=candidate_table_min_score,
            hard_anchor_min_overlap=hard_anchor_min_overlap,
            enable_multi_hop_planner=enable_multi_hop_planner,
            multi_hop_min_bridge_overlap=multi_hop_min_bridge_overlap,
            enable_surface_materializer=enable_surface_materializer,
            enable_provenance_optimizer=enable_provenance_optimizer,
            provenance_conflict_min_margin=provenance_conflict_min_margin,
            provenance_min_support_docs=provenance_min_support_docs,
            provenance_min_reliability=provenance_min_reliability,
            provenance_max_bonus=provenance_max_bonus,
            provenance_shadow_only=provenance_shadow_only,
            enable_schema_bound_semijoin=False,
            schema_bound_semijoin_fallback=False,
            enable_schema_bound_residual_gate=False,
            enable_schema_bound_terminal_slot_gate=False,
            enable_schema_bound_predicate_family_index=False,
            enable_terminal_variable_bound_fact_join=False,
            terminal_variable_min_anchor_overlap=(
                terminal_variable_min_anchor_overlap
            ),
            enable_lineage_directed_terminal_join=False,
            enable_single_trace_fact_join_replay=False,
            enable_stage_aware_fact_join_plan=(
                enable_stage_aware_fact_join_plan
            ),
            enable_role_directed_fact_join=enable_role_directed_fact_join,
            enable_canonical_bound_completion=enable_canonical_bound_completion,
            enable_completion_selectivity_feedback=(
                enable_completion_selectivity_feedback
            ),
            enable_query_plan_root_contract=enable_query_plan_root_contract,
            enable_inverse_relation_completion=(
                enable_inverse_relation_completion
            ),
            enable_title_grounded_completion=(
                enable_title_grounded_completion
            ),
            enable_source_grounded_terminal_repair=(
                enable_source_grounded_terminal_repair
            ),
            enable_title_catalog_root_plan=(
                enable_title_catalog_root_plan
            ),
            enable_title_root_plan_commit=(
                enable_title_root_plan_commit
            ),
            enable_identity_safe_root_commit=(
                enable_identity_safe_root_commit
            ),
            relation_fact_join_reader_stage_mode=stage_reader_mode,
            enable_slot_dependency_cost_admission=(
                enable_slot_dependency_cost_admission
            ),
            enable_bound_completion_fact_join=(
                enable_bound_completion_fact_join
            ),
            bound_completion_lookup=bound_completion_lookup,
            bound_completion_top_k=bound_completion_top_k,
            bound_completion_max_calls=bound_completion_max_calls,
            enable_bound_completion_cost_admission=(
                enable_bound_completion_cost_admission
            ),
            bound_completion_max_risk=bound_completion_max_risk,
            bound_completion_min_net_utility=(
                bound_completion_min_net_utility
            ),
            bound_completion_materialize_reader=(
                bound_completion_materialize_reader
            ),
            enable_provenance_witness_repair=(
                enable_provenance_witness_repair
            ),
            provenance_witness_top_k=provenance_witness_top_k,
            provenance_witness_max_calls=provenance_witness_max_calls,
        )
        fallback_info = dict(fallback_info or {})
        attempt = dict(semijoin_attempt or {})
        attempt["probe_applied"] = bool(
            attempt.get("probe_hit")
            or attempt.get("role_index_candidate_count")
        )
        attempt.update({
            "applied": False,
            "fallback_used": True,
            "fallback_state_restored": True,
            "fallback_input": "pre_schema_relation_snapshot",
            "reason": reason,
        })
        attempt["attempt_index_access"] = dict(info.get("index_access") or {})
        attempt["attempt_scan_rows"] = max(
            int(attempt["attempt_index_access"].get("input_scan_rows", 0) or 0),
            int(attempt["attempt_index_access"].get("rows_after_cap", 0) or 0),
        )
        fallback_info["schema_bound_semijoin"] = attempt
        fallback_info["physical_fallback_reason"] = reason
        fallback_info["upstream_bind_input_count"] = int(
            info.get("upstream_bind_input_count", 0) or 0
        )
        fallback_info["upstream_bind_fact_count"] = int(
            info.get("upstream_bind_fact_count", 0) or 0
        )
        fallback_info["upstream_bind_active_triples"] = list(
            info.get("upstream_bind_active_triples") or []
        )
        fallback_info["single_trace_fact_join_replay"] = dict(
            info.get("single_trace_fact_join_replay") or {}
        )
        return fallback_doc_ids, fallback_info

    if not enabled:
        return [], info
    if int(max_append) <= 0 and not enable_single_trace_fact_join_replay:
        info["reason"] = "max_append_zero"
        return [], info
    if not relation_index or not relation_index.get("items"):
        info["reason"] = "no_relation_fact_join_index"
        return [], info

    doc_ids = sorted_doc_ids.tolist() if hasattr(sorted_doc_ids, "tolist") else list(sorted_doc_ids or [])
    doc_scores = sorted_doc_scores.tolist() if hasattr(sorted_doc_scores, "tolist") else list(sorted_doc_scores or [])
    if not doc_ids:
        info["reason"] = "no_retrieved_docs"
        return [], info

    current_docs = []
    for doc_id in doc_ids[: max(1, qa_top_k)]:
        try:
            current_docs.append(get_doc_content(int(doc_id)))
        except Exception:
            continue
    gap = _answerability_gap_profile(
        question,
        current_docs,
        qa_top_k=qa_top_k,
        min_query_coverage=max(0.0, float(min_query_overlap) * 2.5),
    )
    info.update(gap)
    stage_aware_plan = (
        compile_stage_aware_fact_join_plan(
            question,
            gap,
            enable_bound_completion=bool(enable_bound_completion_fact_join),
            enable_role_directed_join=bool(enable_role_directed_fact_join),
        )
        if enable_stage_aware_fact_join_plan else None
    )
    info["stage_aware_fact_join"]["plan"] = dict(
        stage_aware_plan or {}
    )
    if (
        stage_aware_plan
        and stage_aware_plan.get("compile_status") == "compiled"
    ):
        info["semantic_plan_execution"] = {
            "required": True,
            "completed": False,
            "status": "compiled_plan_not_physically_completed",
            "plan_kind": str(stage_aware_plan.get("plan_kind") or ""),
            "required_operator_count": len(
                stage_aware_plan.get("operators") or []
            ),
            "terminal_binding_count": 0,
        }
    binding_compile_stats = Counter()
    compiled_schema_bound_bindings = (
        _schema_bound_semijoin_bindings(
            replay_upstream_fact_tuples,
            gap=gap,
            question=question,
            enable_terminal_variable_binding=bool(
                enable_terminal_variable_bound_fact_join
            ),
            enable_lineage_directed_terminal_join=bool(
                enable_lineage_directed_terminal_join
            ),
            min_query_anchor_overlap=float(
                terminal_variable_min_anchor_overlap
            ),
            compile_stats=binding_compile_stats,
        )
        if enable_schema_bound_semijoin or enable_single_trace_fact_join_replay
        else []
    )
    if enable_single_trace_fact_join_replay:
        info["single_trace_fact_join_replay"] = (
            replay_logical_physical_fact_join(
                bindings=compiled_schema_bound_bindings,
                relation_index=relation_index,
                gap=gap,
                enable_terminal_slot_gate=True,
            )
        )
    if int(max_append) <= 0:
        info["reason"] = "max_append_zero_replay_only"
        return [], info
    if bridge_only and not gap["bridge_cues"]:
        info["reason"] = "no_bridge_cues"
        return [], info
    effective_gap = bool(gap["answerability_gap"] if use_answerability_gap else gap["evidence_gap"])
    if require_evidence_gap and not effective_gap:
        info["reason"] = "current_evidence_sufficient"
        return [], info

    schema_bound_bindings = (
        compiled_schema_bound_bindings if enable_schema_bound_semijoin else []
    )
    schema_bound_info = info["schema_bound_semijoin"]
    schema_bound_info["binding_count"] = len(schema_bound_bindings)
    schema_bound_info["terminal_variable_binding_count"] = sum(
        bool(binding.get("terminal_variable_bound"))
        for binding in schema_bound_bindings
    )
    schema_bound_info["lineage_directed_binding_count"] = sum(
        bool(binding.get("lineage_directed"))
        for binding in schema_bound_bindings
    )
    schema_bound_info["binding_compile_reject_reason_counts"] = dict(
        binding_compile_stats
    )
    schema_bound_info["probe_key_count"] = len({
        (str(binding.get("probe_role") or ""), str(binding.get("bound_key") or ""))
        for binding in schema_bound_bindings
    })
    schema_bound_requested = bool(schema_bound_bindings)
    if schema_bound_requested:
        schema_bound_info["reason"] = "awaiting_role_index_probe"

    rank_map = {int(doc_id): rank for rank, doc_id in enumerate(doc_ids)}
    direct_index_access = _new_index_access_stats()
    multi_hop_rows: List[Dict[str, Any]] = []
    multi_hop_info: Dict[str, Any] = {
        "enabled": False,
        "plans": [],
        "raw_path_count": 0,
        "row_count": 0,
        "surface_materialized_count": 0,
        "min_bridge_overlap": round(float(multi_hop_min_bridge_overlap), 6),
        "surface_materializer_enabled": bool(enable_surface_materializer),
        "reject_reason_counts": {},
    }
    if (
        enable_candidate_table
        and enable_multi_hop_planner
        and (
            not schema_bound_requested
            or bool(enable_stage_aware_fact_join_plan)
        )
    ):
        multi_hop_rows, multi_hop_info = _execute_multi_hop_fact_join(
            question=question,
            gap=gap,
            relation_index=relation_index,
            rank_map=rank_map,
            get_doc_content=get_doc_content,
            max_facts_per_key=max_facts_per_key,
            max_key_fanout=max_key_fanout,
            max_rows=max(1, int(candidate_table_max_rows)),
            hard_anchor_min_overlap=float(hard_anchor_min_overlap),
            multi_hop_min_bridge_overlap=float(multi_hop_min_bridge_overlap),
            enable_surface_materializer=bool(enable_surface_materializer),
            stage_aware_plan=stage_aware_plan,
            enable_bound_completion=bool(enable_bound_completion_fact_join),
            bound_completion_lookup=bound_completion_lookup,
            bound_completion_top_k=int(bound_completion_top_k),
            bound_completion_max_calls=int(bound_completion_max_calls),
            enable_bound_completion_cost_admission=bool(
                enable_bound_completion_cost_admission
            ),
            bound_completion_max_risk=float(bound_completion_max_risk),
            bound_completion_min_net_utility=float(
                bound_completion_min_net_utility
            ),
            compute_weight=float(compute_weight),
            risk_weight=float(risk_weight),
            enable_canonical_bound_completion=bool(
                enable_canonical_bound_completion
            ),
            enable_completion_selectivity_feedback=bool(
                enable_completion_selectivity_feedback
            ),
            enable_query_plan_root_contract=bool(
                enable_query_plan_root_contract
            ),
            enable_inverse_relation_completion=bool(
                enable_inverse_relation_completion
            ),
            enable_title_grounded_completion=bool(
                enable_title_grounded_completion
            ),
            enable_source_grounded_terminal_repair=bool(
                enable_source_grounded_terminal_repair
            ),
            enable_title_catalog_root_plan=bool(
                enable_title_catalog_root_plan
            ),
            enable_title_root_plan_commit=bool(
                enable_title_root_plan_commit
            ),
            enable_identity_safe_root_commit=bool(
                enable_identity_safe_root_commit
            ),
        )
    elif schema_bound_requested:
        multi_hop_info.update({
            "enabled": False,
            "reason": "replaced_by_schema_bound_semijoin",
        })
    info["multi_hop_join"] = multi_hop_info
    info["semantic_plan_execution"] = dict(
        multi_hop_info.get("semantic_plan_execution")
        or info["semantic_plan_execution"]
    )
    completion_trace = dict(
        multi_hop_info.get("query_plan_execution_trace") or {}
    )
    info["bound_completion_fact_join"].update({
        "call_count": int(
            completion_trace.get("completion_call_count", 0) or 0
        ),
        "doc_count": int(
            completion_trace.get("completion_doc_count", 0) or 0
        ),
        "extracted_fact_count": int(
            completion_trace.get("completion_extracted_fact_count", 0) or 0
        ),
        "path_row_count": int(
            completion_trace.get("completion_path_row_count", 0) or 0
        ),
        "reject_reason_counts": dict(
            completion_trace.get("completion_reject_reason_counts") or {}
        ),
    })
    multi_hop_index_access = dict(multi_hop_info.get("index_access") or {})
    info["index_access"] = _merge_index_access_stats(
        direct_index_access,
        multi_hop_index_access,
    )

    active = _active_fact_signals(
        question,
        top_k_facts,
        candidate_facts,
        upstream_fact_tuples=production_upstream_fact_tuples,
    )
    info["upstream_bind_fact_count"] = int(
        active.get("upstream_fact_count", 0) or 0
    )
    info["upstream_bind_active_triples"] = [
        list(fact)
        for fact in list(active.get("active_triples") or [])[
            int(active.get("base_fact_count", 0) or 0):
        ]
    ]
    info["active_anchor_count"] = len(active["entity_keys"])
    info["active_relation_count"] = len(active["relation_tokens"])
    info["active_schema_count"] = len(active["signatures"])
    if (
        not active["entity_keys"]
        and not active["relation_tokens"]
        and not active["signatures"]
        and not multi_hop_rows
    ):
        info["reason"] = "no_active_fact_join_signal"
        return [], info

    entity_to_items = relation_index.get("entity_to_items") or {}
    subject_to_items = relation_index.get("subject_to_items") or {}
    object_to_items = relation_index.get("object_to_items") or {}
    subject_predicate_family_to_items = (
        relation_index.get("subject_predicate_family_to_items") or {}
    )
    object_predicate_family_to_items = (
        relation_index.get("object_predicate_family_to_items") or {}
    )
    relation_to_items = relation_index.get("relation_to_items") or {}
    schema_to_items = relation_index.get("schema_to_items") or {}
    items = relation_index.get("items") or []

    raw_item_ids: List[int] = []
    key_reject_counts = Counter()

    def add_from_key(
        index: Dict[Any, List[int]],
        key: Any,
        source: str,
        *,
        row_cap: Optional[int] = None,
    ) -> None:
        direct_index_access["lookup_keys"] += 1
        values = list(index.get(key) or [])
        if not values:
            return
        direct_index_access["hit_keys"] += 1
        if max_key_fanout > 0 and len(values) > max_key_fanout:
            key_reject_counts[f"{source}_fanout_too_high"] += 1
            direct_index_access["fanout_rejected_keys"] += 1
            direct_index_access["fanout_rejected_rows"] += len(values)
            return
        effective_cap = max(1, int(row_cap or max_facts_per_key))
        capped = values[:effective_cap]
        raw_item_ids.extend(capped)
        direct_index_access["input_scan_rows"] += len(capped)
        direct_index_access["rows_after_cap"] += len(capped)
        direct_index_access["truncated_rows"] += max(0, len(values) - len(capped))

    schema_bound_matches: Dict[int, Dict[str, Any]] = {}
    schema_bound_gate_rejects = Counter()
    schema_bound_applied = False
    if schema_bound_requested:
        schema_bound_info["probe_executed"] = True
        bound_row_cap = max(1, min(
            max_key_fanout if max_key_fanout > 0 else max_facts_per_key * 4,
            max_facts_per_key * 4,
        ))
        for binding in schema_bound_bindings:
            probe_role = str(binding.get("probe_role") or "")
            probe_index = subject_to_items if probe_role == "subject" else object_to_items
            predicate_families = list(
                binding.get("residual_predicate_families") or []
            )
            predicate_index = (
                subject_predicate_family_to_items
                if probe_role == "subject"
                else object_predicate_family_to_items
            )
            if (
                enable_schema_bound_predicate_family_index
                and predicate_families
            ):
                for predicate_family in predicate_families:
                    composite_key = _role_predicate_index_key(
                        str(binding.get("bound_key") or ""),
                        predicate_family,
                    )
                    schema_bound_info["composite_index_lookup_count"] += 1
                    before_hits = int(direct_index_access["hit_keys"])
                    add_from_key(
                        predicate_index,
                        composite_key,
                        f"schema_bound_{probe_role}_predicate_family",
                        row_cap=bound_row_cap,
                    )
                    if int(direct_index_access["hit_keys"]) > before_hits:
                        schema_bound_info["composite_index_hit_count"] += 1
            else:
                add_from_key(
                    probe_index,
                    str(binding.get("bound_key") or ""),
                    f"schema_bound_{probe_role}",
                    row_cap=bound_row_cap,
                )
        bound_raw_ids = list(dict.fromkeys(
            int(item_id)
            for item_id in raw_item_ids
            if 0 <= int(item_id) < len(items)
        ))
        schema_bound_info["role_index_candidate_count"] = len(bound_raw_ids)
        schema_bound_info["probe_hit"] = bool(bound_raw_ids)
        for item_id in bound_raw_ids:
            match = _schema_bound_semijoin_match(
                items[item_id],
                schema_bound_bindings,
                gap,
                enable_residual_predicate_gate=bool(
                    enable_schema_bound_residual_gate
                ),
                enable_terminal_slot_gate=bool(
                    enable_schema_bound_terminal_slot_gate
                ),
                enable_predicate_family_match=bool(
                    enable_schema_bound_predicate_family_index
                ),
                rejection_counts=schema_bound_gate_rejects,
            )
            if match is not None:
                schema_bound_matches[item_id] = match
        schema_bound_info["matched_candidate_count"] = len(schema_bound_matches)
        schema_bound_info["residual_predicate_match_count"] = sum(
            bool(match.get("residual_predicate_match"))
            for match in schema_bound_matches.values()
        )
        schema_bound_info["terminal_slot_match_count"] = sum(
            bool(match.get("terminal_slot_match"))
            for match in schema_bound_matches.values()
        )
        schema_bound_info["residual_predicate_reject_count"] = int(
            schema_bound_gate_rejects.get("missing_residual_predicate_contract", 0)
            + schema_bound_gate_rejects.get("residual_predicate_mismatch", 0)
        )
        schema_bound_info["terminal_slot_reject_count"] = int(
            schema_bound_gate_rejects.get("terminal_slot_mismatch", 0)
        )
        schema_bound_info["gate_reject_reason_counts"] = dict(
            schema_bound_gate_rejects
        )
        if schema_bound_matches:
            raw_item_ids = list(schema_bound_matches)
            schema_bound_applied = True
            schema_bound_info["applied"] = True
            schema_bound_info["reason"] = "role_index_match"
        elif schema_bound_semijoin_fallback:
            schema_bound_info["fallback_used"] = True
            schema_bound_info["reason"] = (
                "schema_bound_gate_rejected_all_fallback"
                if schema_bound_gate_rejects
                else "no_role_index_candidates_fallback"
            )
            info["index_access"] = _merge_index_access_stats(
                direct_index_access,
                multi_hop_index_access,
            )
            return fallback_to_generic_join(
                schema_bound_info,
                schema_bound_info["reason"],
            )
        else:
            raw_item_ids = []
            schema_bound_info["reason"] = "no_role_index_match"

    if not schema_bound_applied and (not schema_bound_requested or schema_bound_semijoin_fallback):
        for key in sorted(active["entity_keys"]):
            add_from_key(entity_to_items, key, "entity")
        relation_lookup_tokens = active["relation_tokens"] | {
            token for token in active["query_tokens"] if len(token) > 3
        }
        for token in sorted(relation_lookup_tokens):
            add_from_key(relation_to_items, token, "relation")
        for signature in sorted(active["signatures"]):
            add_from_key(schema_to_items, signature, "schema")

    info["index_access"] = _merge_index_access_stats(
        direct_index_access,
        multi_hop_index_access,
    )

    if not raw_item_ids and not multi_hop_rows:
        info["reason"] = "no_join_candidates"
        info["reject_reason_counts"] = dict(key_reject_counts)
        return [], info

    unique_item_ids = list(dict.fromkeys(int(item_id) for item_id in raw_item_ids if 0 <= int(item_id) < len(items)))
    direct_index_access["deduplicated_rows"] = len(unique_item_ids)
    direct_index_access["duplicate_rows"] = max(
        0,
        len(raw_item_ids) - len(unique_item_ids),
    )
    info["index_access"] = _merge_index_access_stats(
        direct_index_access,
        multi_hop_index_access,
    )
    info["raw_candidate_count"] = len(unique_item_ids)

    fallback_rank = len(doc_ids) + 100
    protected = {int(doc_id) for doc_id in doc_ids[: max(0, protect_top_k)]}
    current_top = {int(doc_id) for doc_id in doc_ids[: max(1, qa_top_k)]}
    active_endpoint_tokens = active["endpoint_tokens"]
    active_relation_tokens = active["relation_tokens"]
    q_tokens = active["query_tokens"]

    per_doc: Dict[int, Dict[str, Any]] = {}
    for item_id in unique_item_ids:
        item = items[item_id]
        bound_match = schema_bound_matches.get(item_id)
        if bound_match is not None:
            item = dict(item)
            item["schema_bound_semijoin"] = bound_match
        doc_idx = int(item.get("doc_idx"))
        if doc_idx in protected:
            continue
        endpoint_tokens = set(item.get("endpoint_tokens") or set())
        relation_tokens = set(item.get("relation_tokens") or set())
        join_signal = max(
            _jaccard(endpoint_tokens, active_endpoint_tokens),
            _overlap_over_query(endpoint_tokens, q_tokens),
        )
        relation_signal = max(
            _jaccard(relation_tokens, active_relation_tokens),
            _overlap_over_query(relation_tokens, q_tokens),
        )
        schema_signal = 1.0 if item.get("signature") in active["signatures"] else 0.0
        rank = _safe_rank(doc_idx, rank_map, fallback_rank)
        rank_prior = 1.0 / (1.0 + math.log1p(rank + 1))
        visible_penalty = 0.18 if doc_idx in current_top else 0.0
        if bound_match is not None:
            bound_role_score = float(bound_match.get("relation_role_score", 0.0) or 0.0)
            pre_score = (
                0.44
                + 0.24 * bound_role_score
                + 0.12 * join_signal
                + 0.10 * rank_prior
                + 0.08 * _overlap_over_query(endpoint_tokens | relation_tokens, q_tokens)
                - visible_penalty
            )
        else:
            pre_score = (
                0.33 * join_signal
                + 0.26 * relation_signal
                + 0.16 * schema_signal
                + 0.15 * _overlap_over_query(endpoint_tokens | relation_tokens, q_tokens)
                + 0.10 * rank_prior
                - visible_penalty
            )
        current = per_doc.get(doc_idx)
        if current is None or pre_score > current["pre_score"]:
            per_doc[doc_idx] = {
                "doc_id": doc_idx,
                "item": item,
                "rank": rank,
                "pre_score": float(pre_score),
                "join_signal": float(join_signal),
                "relation_signal": float(relation_signal),
                "schema_signal": float(schema_signal),
                "rank_prior": float(rank_prior),
                "already_visible": doc_idx in current_top,
                "schema_bound_semijoin": bool(bound_match),
            }

    candidates = sorted(
        per_doc.values(),
        key=lambda row: (
            -float(row.get("pre_score", 0.0) or 0.0),
            int(row.get("rank", 999999) or 999999),
            int(row.get("doc_id", 999999) or 999999),
            " | ".join(
                str(value)
                for value in (row.get("item", {}).get("triple") or [])
            ).lower(),
        ),
    )
    candidates = candidates[: max(1, int(candidate_top_k))]
    info["candidate_count"] = len(candidates)
    if not candidates and not multi_hop_rows:
        if schema_bound_applied and schema_bound_semijoin_fallback:
            return fallback_to_generic_join(
                schema_bound_info,
                "no_scored_semijoin_candidates_fallback",
            )
        info["reason"] = "no_scored_candidates"
        info["reject_reason_counts"] = dict(key_reject_counts)
        return [], info

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    directional_rows: List[Dict[str, Any]] = []
    intervention_rows: List[Dict[str, Any]] = []
    reject_counts = Counter(key_reject_counts)
    reject_counts.update((multi_hop_info or {}).get("reject_reason_counts", {}) or {})
    for candidate in candidates:
        doc_id = int(candidate["doc_id"])
        try:
            content = get_doc_content(doc_id)
        except Exception:
            reject_counts["doc_load_failed"] += 1
            continue
        info["index_access"]["loaded_doc_rows"] += 1
        doc_tokens = _tokens(str(content)[:2200])
        title = str(content).split("\n", 1)[0].strip()
        doc_query_overlap = _overlap_over_query(doc_tokens, q_tokens)
        endpoint_doc_overlap = _jaccard(doc_tokens, set(candidate["item"].get("endpoint_tokens") or set()))
        relation_signal = float(candidate["relation_signal"])
        join_signal = float(candidate["join_signal"])
        schema_signal = float(candidate["schema_signal"])
        rank_prior = float(candidate["rank_prior"])
        bound_match = dict(candidate["item"].get("schema_bound_semijoin") or {})
        bound_role_score = float(
            bound_match.get("relation_role_score", 0.0) or 0.0
        )
        effective_relation_signal = max(relation_signal, bound_role_score)
        verifier = _verify_relation_join_candidate(
            item=candidate["item"],
            content=content,
            gap=gap,
            doc_tokens=doc_tokens,
            require_relation_role_match=bool(require_relation_role_match),
            require_anchor_match=bool(require_anchor_match),
            min_anchor_overlap=float(min_anchor_overlap),
            min_role_overlap=float(min_role_overlap),
        )
        evidence_gap_bonus = 0.08 if effective_gap else 0.0
        bridge_bonus = (
            0.05
            if gap["bridge_cues"]
            and effective_relation_signal >= min_relation_overlap
            else 0.0
        )
        verifier_bonus = 0.06 if verifier["accepted"] else 0.0
        schema_bound_bonus = 0.12 if bound_match else 0.0
        utility = _clamp(
            0.28 * join_signal
            + 0.24 * effective_relation_signal
            + 0.18 * doc_query_overlap
            + 0.10 * endpoint_doc_overlap
            + 0.08 * schema_signal
            + 0.07 * rank_prior
            + evidence_gap_bonus
            + bridge_bonus
            + verifier_bonus
            + schema_bound_bonus
        )
        low_query_risk = _clamp((float(min_query_overlap) - doc_query_overlap) / max(float(min_query_overlap), 1e-6))
        low_relation_risk = _clamp(
            (float(min_relation_overlap) - effective_relation_signal)
            / max(float(min_relation_overlap), 1e-6)
        )
        rank_risk = _clamp(max(0, int(candidate["rank"]) - qa_top_k) / max(1.0, float(candidate_top_k)))
        visible_risk = 0.08 if candidate["already_visible"] else 0.0
        non_bridge_risk = (
            0.08
            if not gap["bridge_cues"] and effective_relation_signal < 0.20
            else 0.0
        )
        broad_relation_risk = (
            0.0
            if bound_match else
            0.08 if len(candidate["item"].get("relation_tokens") or []) <= 1 else 0.0
        )
        verifier_risk = 0.28 if not verifier["accepted"] else 0.0
        risk = _clamp(
            0.20 * low_query_risk
            + 0.18 * low_relation_risk
            + 0.18 * rank_risk
            + visible_risk
            + non_bridge_risk
            + broad_relation_risk
            + verifier_risk
        )
        cost_model = _cost_decision(
            question=question,
            content=content,
            item=candidate["item"],
            rank=int(candidate["rank"]),
            raw_candidate_count=info["raw_candidate_count"],
            candidate_count=info["candidate_count"],
            evidence_gap=effective_gap,
            utility=utility,
            risk=risk,
            token_budget=token_budget,
            min_utility=min_utility,
            max_risk=max_risk,
            min_net_utility=min_net_utility,
            token_weight=token_weight,
            compute_weight=compute_weight,
            risk_weight=risk_weight,
        )
        record = {
            "doc_id": doc_id,
            "title": title,
            "rank": int(candidate["rank"]),
            "join_signal": round(join_signal, 6),
            "relation_signal": round(relation_signal, 6),
            "effective_relation_signal": round(effective_relation_signal, 6),
            "schema_signal": round(schema_signal, 6),
            "doc_query_overlap": round(float(doc_query_overlap), 6),
            "endpoint_doc_overlap": round(float(endpoint_doc_overlap), 6),
            "already_visible": bool(candidate["already_visible"]),
            "triple": list(candidate["item"].get("triple") or []),
            "verifier": verifier,
            "cost_model": cost_model,
            "schema_bound_semijoin": bound_match,
            "support_only": bool(boolean_answer_contract),
            "answer_candidate_eligible": not bool(boolean_answer_contract),
        }
        if enable_candidate_table:
            directional_row = _terminal_candidate_contract_row(
                question=question,
                item=candidate["item"],
                content=content,
                title=title,
                rank=int(candidate["rank"]),
                gap=gap,
                hard_anchor_min_overlap=float(hard_anchor_min_overlap),
                stage_aware_plan=stage_aware_plan,
            )
            if directional_row is not None:
                directional_row.update({
                    "verifier_accepted": bool(verifier.get("accepted")),
                    "verifier_reason": str(verifier.get("reason") or ""),
                    "cost_accepted": bool(cost_model.get("accepted")),
                    "cost_reason": str(cost_model.get("reason") or ""),
                    "net_utility": round(
                        float(cost_model.get("net_utility", 0.0) or 0.0), 6
                    ),
                    "support_only": bool(boolean_answer_contract),
                    "answer_candidate_eligible": not bool(boolean_answer_contract),
                    "intervention_only": True,
                })
                intervention_rows.append(dict(directional_row))
                stage_name = str(
                    directional_row.get("slot_plan_stage") or "unclassified"
                )
                reader_stage_eligible = bool(
                    directional_row.get("production_eligible")
                )
                if enable_stage_aware_fact_join_plan:
                    if stage_reader_mode == "control":
                        reader_stage_eligible = False
                    elif stage_reader_mode == "bridge_only":
                        reader_stage_eligible = bool(
                            stage_name == "intermediate_binding"
                            and directional_row.get("bridge_eligible")
                            and directional_row.get("base_validation_passed")
                        )
                    elif stage_reader_mode == "final_slot_validated":
                        reader_stage_eligible = bool(
                            stage_name == "final_answer"
                            and directional_row.get("production_eligible")
                            and directional_row.get("rooted_lineage_complete")
                        )
                if (
                    reader_stage_eligible
                    and float(directional_row.get("table_score", 0.0) or 0.0)
                    >= float(candidate_table_min_score)
                ):
                    reader_row = dict(directional_row)
                    reader_row["reader_stage_eligible"] = True
                    reader_row["shadow_only"] = False
                    directional_rows.append(reader_row)
                elif reader_stage_eligible:
                    reject_counts["directional_table_score_below_threshold"] += 1
        if not verifier["accepted"]:
            reject_counts[verifier["reason"]] += 1
            rejected.append(record)
            continue
        if candidate["already_visible"]:
            reject_counts["already_visible"] += 1
            rejected.append(record)
            continue
        if cost_model["accepted"]:
            if len(accepted) < max(0, int(max_append)):
                accepted.append(record)
            else:
                reject_counts["accepted_passage_budget_full"] += 1
                rejected.append(record)
                continue
            if len(accepted) >= max(0, int(max_append)) and not enable_candidate_table:
                break
        else:
            reject_counts[cost_model["reason"]] += 1
            rejected.append(record)

    selected_multi_hop_rows = list(multi_hop_rows or [])
    if enable_stage_aware_fact_join_plan:
        if stage_reader_mode in {"control", "bridge_only"}:
            selected_multi_hop_rows = []
        elif stage_reader_mode == "final_slot_validated":
            selected_multi_hop_rows = [
                row for row in selected_multi_hop_rows
                if row.get("production_eligible")
                and row.get("rooted_lineage_complete")
                and (
                    row.get("slot_plan_stage") == "final_answer"
                    or (
                        row.get("slot_plan_stage") == "support_only"
                        and row.get("support_only")
                        and row.get("operator_complete")
                    )
                )
            ]
    all_stage_rows = list(multi_hop_rows or []) + intervention_rows
    stage_counts = Counter(
        str(row.get("slot_plan_stage") or "unclassified")
        for row in all_stage_rows
    )
    info["stage_aware_fact_join"].update({
        "bridge_candidate_count": int(
            stage_counts.get("intermediate_binding", 0)
        ),
        "final_slot_candidate_count": int(
            stage_counts.get("final_answer", 0)
        ),
        "support_only_candidate_count": int(
            stage_counts.get("support_only", 0)
        ),
    })
    selected_multi_hop_rows, completion_reader_gate_info = (
        _apply_bound_completion_reader_gate(
            selected_multi_hop_rows,
            materialize=bool(bound_completion_materialize_reader),
        )
    )
    info["bound_completion_fact_join"]["reader_gate"] = (
        completion_reader_gate_info
    )
    table_rows = _merge_directional_rows(
        selected_multi_hop_rows + directional_rows,
        max_rows=max(1, int(candidate_table_max_rows)),
    )
    for row in table_rows:
        row["support_only"] = bool(boolean_answer_contract)
        row["answer_candidate_eligible"] = not bool(boolean_answer_contract)
    table_rows, provenance_witness_info = _repair_provenance_witnesses(
        table_rows,
        question=question,
        plan=stage_aware_plan,
        relation_index=relation_index,
        lookup=bound_completion_lookup,
        enabled=bool(enable_provenance_witness_repair),
        top_k=max(1, int(provenance_witness_top_k)),
        max_calls=max(0, int(provenance_witness_max_calls)),
        min_support_docs=max(1, int(provenance_min_support_docs)),
        enable_cost_admission=bool(enable_bound_completion_cost_admission),
        max_risk=float(bound_completion_max_risk),
        min_net_utility=float(bound_completion_min_net_utility),
        compute_weight=float(compute_weight),
        risk_weight=float(risk_weight),
        max_facts_per_key=max_facts_per_key,
        max_key_fanout=max_key_fanout,
    )
    info["provenance_witness_repair"] = provenance_witness_info
    table_rows, provenance_optimizer_info = _optimize_provenance_conflicts(
        table_rows,
        question=question,
        relation_index=relation_index,
        enabled=bool(enable_provenance_optimizer),
        min_margin=float(provenance_conflict_min_margin),
        min_support_docs=max(1, int(provenance_min_support_docs)),
        min_reliability=float(provenance_min_reliability),
        max_bonus=float(provenance_max_bonus),
        shadow_only=bool(provenance_shadow_only),
        max_facts_per_key=max_facts_per_key,
        max_key_fanout=max_key_fanout,
    )
    table_rows = [
        row for row in table_rows
        if float(row.get("table_score", 0.0) or 0.0) >= float(candidate_table_min_score)
    ]
    table_rows, provenance_hard_gate_info = _apply_reader_provenance_hard_gate(
        table_rows,
        enabled=bool(enable_provenance_optimizer and not provenance_shadow_only),
        min_support_docs=max(1, int(provenance_min_support_docs)),
    )
    provenance_optimizer_info["reader_hard_gate"] = provenance_hard_gate_info
    if enable_slot_dependency_cost_admission:
        table_rows, slot_dependency_cost_info = _admit_slot_dependency_reader_rows(
            table_rows,
            question=question,
            visible_doc_ids=doc_ids[: max(0, int(qa_top_k))],
            get_doc_content=get_doc_content,
            evidence_gap=effective_gap,
            raw_candidate_count=int(info.get("raw_candidate_count", 0) or 0),
            candidate_count=int(info.get("candidate_count", 0) or 0),
            token_budget=int(token_budget),
            min_utility=float(min_utility),
            max_risk=float(max_risk),
            min_net_utility=float(min_net_utility),
            token_weight=float(token_weight),
            compute_weight=float(compute_weight),
            risk_weight=float(risk_weight),
        )
    else:
        slot_dependency_cost_info = {
            "enabled": False,
            "checked_row_count": 0,
            "input_row_count": len(table_rows),
            "output_row_count": len(table_rows),
            "accepted_row_count": 0,
            "rejected_row_count": 0,
            "reason_counts": {},
        }
    info["slot_dependency_cost_admission"] = slot_dependency_cost_info
    validated_terminal_projection_rows = (
        _build_validated_terminal_projection_rows(
            table_rows,
            stage_aware_plan,
        )
    )
    reader_table_rows, reader_projection_info = _project_reader_candidate_rows(
        table_rows,
        min_margin=float(provenance_conflict_min_margin),
    )
    candidate_table_doc = ""
    if reader_table_rows:
        candidate_table_doc = _format_directional_candidate_table(
            question=question,
            rows=reader_table_rows,
        )

    ranked_intervention_rows = _merge_directional_rows(
        intervention_rows,
        max_rows=max(
            1, int(candidate_table_max_rows), int(candidate_top_k)
        ),
    )
    ranked_intervention_rows.sort(key=lambda row: (
        -int(bool(row.get("verifier_accepted"))),
        -int(bool(row.get("cost_accepted"))),
        -float(row.get("net_utility", 0.0) or 0.0),
        -float(row.get("table_score", 0.0) or 0.0),
        int(row.get("rank", 999999) or 999999),
        str(row.get("candidate") or "").lower(),
    ))
    intervention_row = (
        dict(ranked_intervention_rows[0]) if ranked_intervention_rows else {}
    )
    intervention_doc = (
        _format_directional_candidate_table(
            question=question,
            rows=[intervention_row],
        )
        if intervention_row else ""
    )

    has_reader_table = bool(reader_table_rows and candidate_table_doc)
    table_first_reader_projection = bool(enable_candidate_table)
    reader_passage_records = [] if table_first_reader_projection else accepted

    schema_bound_info["accepted_count"] = sum(
        bool(item.get("schema_bound_semijoin")) for item in accepted
    )
    schema_bound_info["table_row_count"] = sum(
        bool(row.get("schema_bound_semijoin")) for row in table_rows
    )
    schema_bound_info["reader_row_count"] = sum(
        bool(row.get("schema_bound_semijoin")) for row in reader_table_rows
    )
    schema_bound_info["reader_materialized"] = bool(
        schema_bound_info["reader_row_count"] > 0
    )
    if (
        schema_bound_applied
        and schema_bound_semijoin_fallback
        and not schema_bound_info["reader_materialized"]
    ):
        return fallback_to_generic_join(
            schema_bound_info,
            "no_reader_visible_semijoin_rows_fallback",
        )

    info["accepted_count"] = len(accepted)
    info["reader_passage_candidate_count"] = len(reader_passage_records)
    info["table_first_reader_projection"] = bool(table_first_reader_projection)
    info["raw_reader_append_suppressed_count"] = (
        len(accepted) if table_first_reader_projection else 0
    )
    info["accepted"] = accepted
    info["rejected"] = rejected[:8]
    info["candidate_table_rows"] = table_rows
    info["candidate_table_row_count"] = len(table_rows)
    info["reader_candidate_table_rows"] = reader_table_rows
    info["reader_candidate_table_row_count"] = len(reader_table_rows)
    info["stage_aware_fact_join"]["reader_row_count"] = len(
        reader_table_rows
    )
    info["reader_candidate_table_projection"] = reader_projection_info
    info["candidate_table_doc"] = candidate_table_doc
    info["terminal_candidate_rows"] = ranked_intervention_rows
    info["validated_terminal_candidate_rows"] = [
        row for row in ranked_intervention_rows
        if row.get("production_eligible")
    ]
    info["terminal_candidate_count"] = len(ranked_intervention_rows)
    info["validated_terminal_candidate_count"] = len(
        info["validated_terminal_candidate_rows"]
    )
    info["validated_terminal_projection_rows"] = (
        validated_terminal_projection_rows
    )
    info["validated_terminal_projection_row_count"] = len(
        validated_terminal_projection_rows
    )
    info["counterfactual_reader_intervention"] = {
        "enabled": True,
        "eligible": bool(intervention_row and intervention_doc),
        "reason": (
            "directional_candidate_frozen"
            if intervention_row and intervention_doc
            else "no_directional_candidate"
        ),
        "row": intervention_row,
        "candidate_table_doc": intervention_doc,
        "candidate_count": len(ranked_intervention_rows),
        "validation_tier": str(
            intervention_row.get("validation_tier") or ""
        ),
        "production_eligible": bool(
            intervention_row.get("production_eligible")
        ),
        "verifier_accepted": bool(intervention_row.get("verifier_accepted")),
        "cost_accepted": bool(intervention_row.get("cost_accepted")),
        "production_materialized": bool(
            intervention_doc and intervention_doc == candidate_table_doc
        ),
    }
    info["provenance_optimizer"] = provenance_optimizer_info
    info["reject_reason_counts"] = dict(reject_counts)
    info["reader_candidate_doc_ids"] = [int(item["doc_id"]) for item in reader_passage_records]
    info["reader_candidate_titles"] = [str(item["title"]) for item in reader_passage_records]
    info["applied"] = bool(reader_passage_records or has_reader_table)
    if schema_bound_info["reader_materialized"]:
        info["reason"] = "accepted_schema_bound_candidate_table"
    elif has_reader_table:
        info["reason"] = "accepted_candidate_table"
    elif table_rows:
        info["reason"] = "candidate_table_suppressed_by_conflict_projection"
    elif accepted and table_first_reader_projection:
        info["reason"] = "candidate_table_required_no_rows"
    elif accepted:
        info["reason"] = "accepted_reader_append"
    else:
        info["reason"] = "no_cost_accepted_candidates"
    return info["reader_candidate_doc_ids"], info


def _build_provenance_bound_witness_certificate(
    join_info: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Render a complete admitted witness without consulting QA gold.

    The certificate is intentionally stricter than ordinary passage projection:
    every logical operator must be represented by a source-supported,
    obligation-admitted fact with a locatable source sentence.  It is a Reader
    input only; it never authorizes direct answer replacement.
    """
    info: Dict[str, Any] = {
        "eligible": False,
        "reason": "not_evaluated",
        "row_count": 0,
        "operator_count": 0,
        "fact_ids": [],
    }
    paper_trace = dict((join_info or {}).get("paper_evidence_execution") or {})
    if not bool(paper_trace.get("question_complete_supported_witness")):
        info["reason"] = "witness_not_complete"
        return "", info
    if not bool(paper_trace.get("structured_admission")):
        info["reason"] = "structured_admission_missing"
        return "", info

    operators = list((paper_trace.get("logical_plan") or {}).get("operators") or [])
    operator_order = {
        str(operator.get("operator_id") or ""): index
        for index, operator in enumerate(operators)
        if str(operator.get("operator_id") or "")
    }
    required_operator_ids = set(operator_order)
    info["operator_count"] = len(required_operator_ids)

    facts_by_id = {
        str(row.get("fact_id") or ""): row
        for row in paper_trace.get("facts") or []
        if str(row.get("fact_id") or "")
    }
    reader_fact_ids = list(
        paper_trace.get("reader_evidence_fact_ids")
        or paper_trace.get("v8_shadow_reader_evidence_fact_ids")
        or []
    )
    if not reader_fact_ids:
        info["reason"] = "reader_witness_fact_ids_missing"
        return "", info

    rows: List[Mapping[str, Any]] = []
    seen_fact_ids: Set[str] = set()
    for raw_fact_id in reader_fact_ids:
        fact_id = str(raw_fact_id or "")
        if not fact_id or fact_id in seen_fact_ids:
            continue
        seen_fact_ids.add(fact_id)
        row = facts_by_id.get(fact_id)
        if not row:
            info["reason"] = "reader_witness_fact_missing"
            return "", info
        if str(row.get("support_status") or "").upper() != "SUPPORTED":
            info["reason"] = "witness_fact_not_supported"
            return "", info
        if str(row.get("source_support_gate_status") or "").upper() != "SUPPORTED":
            info["reason"] = "witness_source_support_not_admitted"
            return "", info
        if str(row.get("obligation_admission_status") or "").upper() != "ADMITTED":
            info["reason"] = "witness_obligation_not_admitted"
            return "", info
        if str(row.get("binding_obligation_contract_status") or "").upper() != "ADMITTED":
            info["reason"] = "witness_binding_contract_not_admitted"
            return "", info
        if bool(row.get("unsafe_admission")):
            info["reason"] = "unsafe_witness_fact"
            return "", info
        if not str(row.get("source_sentence_span") or "").strip():
            info["reason"] = "witness_source_span_missing"
            return "", info
        if not str(row.get("source_title") or "").strip():
            info["reason"] = "witness_source_title_missing"
            return "", info
        rows.append(row)

    covered_operator_ids = {
        str(row.get("operator_id") or "") for row in rows
        if str(row.get("operator_id") or "")
    }
    if required_operator_ids and not required_operator_ids.issubset(
        covered_operator_ids
    ):
        info["reason"] = "witness_operator_coverage_incomplete"
        return "", info

    rows.sort(
        key=lambda row: (
            operator_order.get(str(row.get("operator_id") or ""), 999),
            str(row.get("operator_id") or ""),
            str(row.get("fact_id") or ""),
        )
    )
    lines = [
        "Verified Source-Grounded Relation Path",
        (
            "Every normalized relation below passed source support, query "
            "obligation, direction, role, and binding checks. Follow the steps "
            "in order; use the quoted source sentence as provenance."
        ),
    ]
    for step_index, row in enumerate(rows, start=1):
        triple = " | ".join(
            str(value or "").strip() for value in row.get("triple") or []
        )
        lines.append(
            '- step={step} operator="{operator}" normalized_relation="{triple}" '
            'source="{title}" source_doc_id={doc_id} evidence="{sentence}"'.format(
                step=step_index,
                operator=str(row.get("operator_id") or "lookup"),
                triple=triple[:320],
                title=str(row.get("source_title") or "")[:160],
                doc_id=str(row.get("source_id", row.get("doc_idx", ""))),
                sentence=str(row.get("source_sentence_span") or "")[:640],
            )
        )

    info.update({
        "eligible": True,
        "reason": "complete_admitted_witness",
        "row_count": len(rows),
        "fact_ids": [str(row.get("fact_id") or "") for row in rows],
        "covered_operator_ids": sorted(covered_operator_ids),
    })
    return "\n".join(lines), info


def append_relation_fact_join_evidence(
    docs: List[str],
    join_docs: List[str],
    join_info: Optional[Dict[str, Any]],
    *,
    enabled: bool,
    max_append: int,
    append_mode: str = "table",
    preserve_complete_witness_projection: bool = False,
    complete_witness_projection_eligible: bool = True,
    complete_witness_projection_gate_reason: str = "",
    source_surface_first: bool = False,
) -> Tuple[List[str], Dict[str, Any]]:
    projection = {
        "enabled": bool(enabled),
        "reason": "disabled" if not enabled else "not_evaluated",
        "appended": 0,
        "appended_titles": [],
        "append_mode": str(append_mode or "table"),
        "preserve_complete_witness_projection": bool(
            preserve_complete_witness_projection
        ),
        "complete_witness_projection_eligible": bool(
            complete_witness_projection_eligible
        ),
        "complete_witness_projection_gate_reason": str(
            complete_witness_projection_gate_reason or ""
        ),
        "source_surface_first": bool(source_surface_first),
        "witness_projection_appended": False,
        "witness_certificate": {
            "eligible": False,
            "reason": "mode_not_requested",
            "row_count": 0,
        },
    }
    if not enabled:
        return docs, projection
    if int(max_append) <= 0:
        projection["reason"] = "max_append_zero"
        return docs, projection
    join_info = join_info or {}
    mode = str(append_mode or "table").strip().lower()
    if mode in {"none", "trace_only"}:
        projection["reason"] = "trace_only"
        return docs, projection
    candidate_docs: List[str] = []
    table_doc = str(join_info.get("candidate_table_doc") or "").strip()
    certificate_modes = {
        "witness_certificate",
        "certificate_and_passage",
        "witness_or_bounded_partial_certificate",
    }
    if mode in certificate_modes:
        certificate_doc, certificate_info = (
            _build_provenance_bound_witness_certificate(join_info)
        )
        projection["witness_certificate"] = certificate_info
        if certificate_doc:
            candidate_docs.append(certificate_doc)
        elif (
            mode == "witness_or_bounded_partial_certificate"
            and str(join_info.get("reader_evidence_mode") or "")
            == "category_affine_supported_partial"
        ):
            candidate_docs.extend(join_docs or [])
            projection["witness_certificate"] = {
                "eligible": True,
                "reason": "bounded_partial_certificate",
                "row_count": len(join_info.get("accepted") or []),
            }
    if mode in {"table", "table_only", "table_and_passage"} and table_doc:
        candidate_docs.append(table_doc)
    if mode in {
        "passage",
        "passage_only",
        "table_and_passage",
        "certificate_and_passage",
    }:
        candidate_docs.extend(join_docs or [])
    if mode not in {
        "table",
        "table_only",
        "passage",
        "passage_only",
        "table_and_passage",
        "witness_certificate",
        "certificate_and_passage",
        "witness_or_bounded_partial_certificate",
    }:
        candidate_docs = [table_doc] if table_doc else list(join_docs or [])
    if mode in certificate_modes and not candidate_docs:
        projection["reason"] = str(
            (projection.get("witness_certificate") or {}).get("reason")
            or "witness_certificate_unavailable"
        )
        return docs, projection
    if not join_docs:
        if not candidate_docs:
            projection["reason"] = "no_join_docs"
            return docs, projection
    seen = {str(doc).strip() for doc in docs or []}
    appended = []
    for doc in candidate_docs:
        doc_key = str(doc or "").strip()
        if not doc_key or doc_key in seen:
            continue
        appended.append(doc)
        seen.add(doc_key)
        if len(appended) >= max(0, int(max_append)):
            break
    if (
        not appended
        and preserve_complete_witness_projection
        and complete_witness_projection_eligible
    ):
        paper_trace = dict(
            (join_info or {}).get("paper_evidence_execution") or {}
        )
        accepted = list((join_info or {}).get("accepted") or [])
        complete_supported = bool(
            paper_trace.get("question_complete_supported_witness")
            and paper_trace.get("structured_admission")
            and accepted
            and all(
                str(row.get("support_status") or "").upper() == "SUPPORTED"
                and str(row.get("source_sentence_span") or "").strip()
                for row in accepted
            )
        )
        if complete_supported:
            operator_order = {
                str(operator.get("operator_id") or ""): index
                for index, operator in enumerate(
                    (paper_trace.get("logical_plan") or {}).get("operators")
                    or []
                )
            }
            ordered_rows = sorted(
                accepted,
                key=lambda row: (
                    operator_order.get(str(row.get("operator_id") or ""), 999),
                    str(row.get("operator_id") or ""),
                    int(row.get("source_id", row.get("doc_idx", -1)) or -1),
                ),
            )
            lines = [
                "Supported Relational Witness",
                "Status: complete; every edge below is source-supported and provenance-bound.",
            ]
            if source_surface_first:
                lines.append(
                    "Answer-surface policy: quoted source evidence is authoritative. "
                    "The index fact is a normalized lookup key and may omit accents, "
                    "qualifiers, or geographic detail. Answer using the source-supported "
                    "surface that is compatible with the question."
                )
            for row in ordered_rows:
                triple = " | ".join(
                    str(value or "").strip()
                    for value in row.get("triple") or []
                )
                values = {
                    "operator": str(row.get("operator_id") or "lookup"),
                    "title": str(row.get("source_title") or "")[:120],
                    "doc_id": str(row.get("source_id", row.get("doc_idx", ""))),
                    "triple": triple[:240],
                    "sentence": str(row.get("source_sentence_span") or "")[:480],
                }
                if source_surface_first:
                    lines.append(
                        "- {operator} evidence=\"{sentence}\" source=\"{title}\" "
                        "source_doc_id={doc_id} index_fact=\"{triple}\"".format(**values)
                    )
                else:
                    lines.append(
                        "- {operator} source=\"{title}\" source_doc_id={doc_id} "
                        "fact=\"{triple}\" evidence=\"{sentence}\"".format(**values)
                    )
            witness_doc = "\n".join(lines)
            if witness_doc not in seen:
                appended.append(witness_doc)
                projection["witness_projection_appended"] = True
                projection["witness_projection_row_count"] = len(ordered_rows)
                projection["reason"] = "complete_witness_projection_appended"
    if (
        not appended
        and preserve_complete_witness_projection
        and not complete_witness_projection_eligible
    ):
        projection["reason"] = (
            str(complete_witness_projection_gate_reason or "").strip()
            or "complete_witness_projection_not_selected"
        )
        return docs, projection
    if not appended:
        projection["reason"] = "already_visible"
        return docs, projection
    if not projection["witness_projection_appended"]:
        projection["reason"] = (
            "witness_certificate_appended"
            if mode in certificate_modes
            else "appended"
        )
    projection["appended"] = len(appended)
    projection["appended_titles"] = [str(doc).split("\n", 1)[0].strip() for doc in appended]
    if isinstance(join_info, dict):
        projection["source_reason"] = join_info.get("reason")
        projection["candidate_table_row_count"] = int(join_info.get("candidate_table_row_count", 0) or 0)
    if mode in certificate_modes:
        projection["certificate_prepend"] = True
        return appended + list(docs or []), projection
    return list(docs or []) + appended, projection
