"""Optional V3.15 runtime for strict anchor-first possessive lowering."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from .evidence_query_compiler_semantic_v3_12_runtime_utils import (
    compile_rule_first_semantic_v3_12_query_plan,
)
from .evidence_query_compiler_semantic_v3_15_anchor_utils import (
    SEMANTIC_V3_15_ANCHOR_CONTRACT_VERSION,
    lower_semantic_v3_15_anchor_first,
)
from .relation_fact_join_utils import attach_logical_query_plan_identity


SEMANTIC_V3_15_RUNTIME_VERSION = "rule_first_semantic_v3_15_runtime_v1"


def compile_rule_first_semantic_v3_15_query_plan(
    question: str,
    *,
    llm_infer: Optional[Callable[..., Any]],
    model_name: str,
    rule_options: Optional[Mapping[str, Any]] = None,
    generation_max_completion_tokens: int = 650,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    plan, trace = compile_rule_first_semantic_v3_12_query_plan(
        question,
        llm_infer=llm_infer,
        model_name=model_name,
        rule_options=rule_options,
        generation_max_completion_tokens=generation_max_completion_tokens,
        semantic_skeleton_lowerer=lower_semantic_v3_15_anchor_first,
    )
    trace = dict(trace)
    trace["version"] = SEMANTIC_V3_15_RUNTIME_VERSION
    versions = dict(trace.get("versions") or {})
    versions["anchor_first_contract"] = SEMANTIC_V3_15_ANCHOR_CONTRACT_VERSION
    trace["versions"] = versions
    if str(trace.get("selected_source") or "") == "llm_semantic_skeleton_v3_12":
        trace["selected_source"] = "llm_semantic_skeleton_v3_15"
        plan = attach_logical_query_plan_identity(
            plan,
            compiler_source="llm_semantic_skeleton_v3_15",
            compiler_mode="rule_first_semantic_v3_15",
            attempt_id="canonical",
        )
    plan["query_compiler_trace"] = trace
    return plan, trace
