from pathlib import Path
from typing import Optional

from .utils.config_utils import BaseConfig


def build_s2rag_config(
    *,
    save_dir: str = "outputs",
    dataset: Optional[str] = None,
    llm_name: str = "reader-model",
    llm_base_url: Optional[str] = None,
    retrieval_backend: str = "graph_ppr",
    calibration_path: str = "",
) -> BaseConfig:
    config = BaseConfig(
        save_dir=save_dir,
        dataset=dataset,
        llm_name=llm_name,
        llm_base_url=llm_base_url,
        retrieval_backend=retrieval_backend,
    )
    config.enable_paper_evidence_query_executor = True
    config.enable_relation_fact_join_retrieval = True
    config.enable_stage_aware_fact_join_plan = True
    config.enable_role_directed_fact_join = True
    config.enable_query_plan_root_contract = True
    config.enable_bound_completion_fact_join = True
    config.enable_completion_evidence_pool = True
    config.enable_canonical_bound_completion = True
    config.enable_inverse_relation_completion = True
    config.enable_title_grounded_completion = True
    config.enable_source_grounded_terminal_repair = True

    config.paper_evidence_query_compiler_mode = "rule_first_semantic_v3_15"
    config.paper_evidence_allow_model_based_semantic_admission = True
    config.paper_evidence_require_semantic_execution_admission = True
    config.paper_evidence_policy = "bswc_rank_consistent_binding_qualified_cover"
    config.paper_evidence_execution_mode = "saqe"
    config.paper_evidence_completion_mode = "lazy"
    config.paper_evidence_completion_query_mode = "obligation_bound"
    config.paper_evidence_repair_scheduler_mode = "eager"
    config.paper_evidence_verification_mode = "strict"

    config.paper_evidence_max_actions = 12
    config.paper_evidence_max_retrieval_calls = 2
    config.paper_evidence_max_model_calls = 2
    config.paper_evidence_max_scan_rows = 5000
    config.paper_evidence_reserve_model_calls = 1
    config.paper_evidence_bswc_beam_width = 16
    config.paper_evidence_bswc_search_horizon = 4
    config.paper_evidence_bswc_source_min_samples = 8
    config.paper_evidence_bswc_feedback_strength = 3.0
    config.paper_evidence_bswc_stop_threshold = 0.035
    config.paper_evidence_bswc_min_exploration_actions = 1
    config.paper_evidence_bswc_fpawc_stop_upper_bound = 0.0
    config.paper_evidence_bswc_query_feedback = True
    config.paper_evidence_bswc_source_calibration_path = (
        str(Path(calibration_path).expanduser()) if calibration_path else ""
    )

    config.paper_evidence_provenance_source_verifier = True
    config.paper_evidence_source_support_obligation_admission = True
    config.paper_evidence_enhanced_source_support = True
    config.paper_evidence_obligation_admission = True
    config.paper_evidence_query_local_overlay_reuse = True
    config.paper_evidence_binding_obligation_contract_validation = True
    config.paper_evidence_identity_safe_composite_join = True
    config.paper_evidence_composite_index_access = True
    config.paper_evidence_collision_aware_schema_identity = True
    config.paper_evidence_safe_title_family_root_binding = True
    config.paper_evidence_relation_matching_mode = "canonical"
    config.paper_evidence_predicate_subtype_constraints = True
    config.paper_evidence_type_role_constraints = True

    config.paper_evidence_schema_plan_coverage_v7 = True
    config.paper_evidence_general_plan_coverage_v8 = False
    config.paper_evidence_schema_exact_nested_join_v9 = False
    config.paper_evidence_lineage_terminal_semijoin_v7_1 = True
    config.paper_evidence_provenance_grouped_terminal_v7_2 = True
    config.paper_evidence_v8_safe_materialization = True

    config.paper_evidence_materialize_reader_evidence = True
    config.paper_evidence_materialize_verified_partial = True
    config.paper_evidence_partial_max_docs = 2
    config.paper_evidence_apply_structured_answer = False
    config.paper_evidence_answer_domain_guard = True
    config.paper_evidence_incumbent_guard = True
    return config
