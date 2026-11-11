import os
from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Optional
)

from .logging_utils import get_logger

logger = get_logger(__name__)


PAPER_EVIDENCE_QUERY_COMPILER_MODES = (
    "rule",
    "rule_contract_v1",
    "rule_first_semantic_v3_15",
)


@dataclass
class BaseConfig:
    """One and only configuration."""

    llm_name: str = field(
        default="reader-model",
        metadata={"help": "Class name indicating which LLM model to use."}
    )
    llm_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for the language model service."}
    )
    llm_cache_namespace: str = field(
        default="",
        metadata={"help": "Optional namespace added to LLM cache keys for isolated ablation/reproducibility runs."}
    )
    rerank_cache_namespace: str = field(
        default="",
        metadata={
            "help": (
                "Optional cache namespace used only by the retrieval fact reranker. "
                "An empty value reuses llm_cache_namespace."
            )
        },
    )
    llm_cache_only: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, serve LLM requests exclusively from the SQLite cache. "
                "A cache miss fails before an API client is created or a network request is attempted."
            )
        },
    )
    embedding_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for a remote embedding service."}
    )
    azure_endpoint: str = field(
        default=None,
        metadata={"help": "Optional managed-service endpoint for the language model."}
    )
    azure_embedding_endpoint: str = field(
        default=None,
        metadata={"help": "Optional managed-service endpoint for embeddings."}
    )
    max_new_tokens: Union[None, int] = field(
        default=2048,
        metadata={"help": "Max new tokens to generate in each inference."}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={"help": "How many chat completion choices to generate for each input message."}
    )
    seed: Union[None, int] = field(
        default=None,
        metadata={"help": "Random seed."}
    )
    temperature: float = field(
        default=0,
        metadata={"help": "Temperature for sampling in each inference."}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: { "type": "json_object" },
        metadata={"help": "Specifying the format that the model must output."}
    )


    max_retry_attempts: int = field(
        default=5,
        metadata={"help": "Max number of retry attempts for an asynchronous API calling."}
    )
    retrieval_llm_workers: int = field(
        default=1,
        metadata={
            "help": (
                "Bounded worker threads for retrieval-time LLM reranking and "
                "semantic query compilation. One preserves legacy serial execution."
            )
        },
    )

    force_openie_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing openie files and rebuild them from scratch."}
    )


    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing storage files and graph data and will rebuild from scratch."}
    )
    graph_cache_validate_config: bool = field(
        default=True,
        metadata={"help": "If set to True, only reuse graph.pickle when its graph-construction signature matches the current config."}
    )
    rerank_dspy_file_path: str = field(
        default=None,
        metadata={"help": "Path to the rerank dspy file."}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={"help": "Multiplicative factor that modified the passage node weights in PPR."}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "If set to True, will save the OpenIE model to disk."}
    )


    text_preprocessor_class_name: str = field(
        default="TextPreprocessor",
        metadata={"help": "Name of the text-based preprocessor to use in preprocessing."}
    )
    preprocess_encoder_name: str = field(
        default="preprocessing-model",
        metadata={"help": "Name of the encoder to use in preprocessing (currently implemented specifically for doc chunking)."}
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128,
        metadata={"help": "Number of overlap tokens between neighbouring chunks."}
    )
    preprocess_chunk_max_token_size: int = field(
        default=None,
        metadata={"help": "Max number of tokens each chunk can contain. If set to None, the whole doc will treated as a single chunk."}
    )
    preprocess_chunk_func: Literal["by_token", "by_word"] = field(default='by_token')



    information_extraction_model_name: Literal["relation_extraction_chat", ] = field(
        default="relation_extraction_chat",
        metadata={"help": "Class name indicating which information extraction model to use."}
    )
    openie_mode: Literal["offline", "online"] = field(
        default="online",
        metadata={"help": "Mode of the OpenIE model to use."}
    )
    skip_graph: bool = field(
        default=False,
        metadata={"help": "Whether to skip graph construction or not. Set it to be true when running vllm offline indexing for the first time."}
    )



    embedding_model_name: str = field(
        default="nvidia/NV-Embed-v2",
        metadata={"help": "Class name indicating which embedding model to use."}
    )
    embedding_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "Max sequence length for the embedding model."}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto",
        metadata={"help": "Data type for local embedding model."}
    )




    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "k for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "Batch size for query embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "Batch size for key embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "Similarity threshold to include candidate synonymy nodes."}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "Whether the graph is directed or not."}
    )




    linking_top_k: int = field(
        default=5,
        metadata={"help": "The number of linked nodes at each retrieval step"}
    )
    retrieval_top_k: int = field(
        default=200,
        metadata={"help": "Retrieving k documents at each step"}
    )
    retrieval_backend: Literal["graph_ppr", "nv_embed_dense"] = field(
        default="graph_ppr",
        metadata={
            "help": (
                "Passage retrieval backend. 'nv_embed_dense' ranks passages only by "
                "NV-Embed-v2 similarity and never executes graph propagation or PPR."
            )
        },
    )
    damping: float = field(
        default=0.5,
        metadata={"help": "Damping factor for ppr algorithm."}
    )
    enable_path_aware_rerank: bool = field(
        default=False,
        metadata={"help": "Whether to rerank top retrieved passages with query-conditioned path signals."}
    )
    path_aware_rerank_top_k: int = field(
        default=200,
        metadata={"help": "Number of retrieved passages considered by path-aware reranking."}
    )
    path_aware_rerank_weight: float = field(
        default=0.30,
        metadata={"help": "Boost weight for path-aware reranking scores."}
    )
    path_aware_rerank_bridge_only: bool = field(
        default=True,
        metadata={"help": "Only apply path-aware reranking to questions with bridge or composition cues."}
    )
    enable_path_aware_evidence: bool = field(
        default=False,
        metadata={"help": "Whether to append high-confidence path-aware evidence without changing retrieval ranking."}
    )
    path_aware_evidence_top_k: int = field(
        default=100,
        metadata={"help": "Number of retrieved passages considered for non-destructive path-aware evidence."}
    )
    path_aware_evidence_max_extra: int = field(
        default=2,
        metadata={"help": "Maximum number of path-aware evidence passages appended to QA context."}
    )
    path_aware_evidence_min_score: float = field(
        default=0.22,
        metadata={"help": "Minimum path score for appending path-aware evidence."}
    )
    path_aware_evidence_bridge_only: bool = field(
        default=True,
        metadata={"help": "Only append path-aware evidence for bridge or composition questions."}
    )
    enable_support_aware_rerank: bool = field(
        default=False,
        metadata={"help": "Whether to conservatively promote likely support passages into top-k retrieval."}
    )
    support_aware_rerank_top_k: int = field(
        default=50,
        metadata={"help": "Number of retrieved candidates considered by support-aware top-k reranking."}
    )
    support_aware_rerank_target_k: int = field(
        default=5,
        metadata={"help": "Top-k prefix rebuilt by support-aware reranking."}
    )
    support_aware_rerank_weight: float = field(
        default=0.55,
        metadata={"help": "Weight for support-aware retrieval reranking scores."}
    )
    support_aware_rerank_min_score: float = field(
        default=0.50,
        metadata={"help": "Minimum support score required for a candidate to be promoted."}
    )
    support_aware_rerank_max_promotions: int = field(
        default=1,
        metadata={"help": "Maximum number of non-top-k candidates promoted into the target prefix."}
    )
    support_aware_rerank_protect_top1: bool = field(
        default=True,
        metadata={"help": "Keep the original top-1 passage fixed during support-aware reranking."}
    )
    enable_fact_source_scan: bool = field(
        default=False,
        metadata={"help": "Promote provenance documents for high-scoring exact fact tuples."}
    )
    fact_source_scan_fallback_only: bool = field(
        default=True,
        metadata={"help": "Run fact-source scan only when no verified complete witness is available."}
    )
    fact_source_scan_fact_top_k: int = field(
        default=8,
        metadata={"help": "Maximum semantically retrieved facts probed against the tuple index."}
    )
    fact_source_scan_target_top_k: int = field(
        default=5,
        metadata={"help": "Reader-visible prefix whose tail may receive provenance source documents."}
    )
    fact_source_scan_max_promotions: int = field(
        default=1,
        metadata={"help": "Maximum provenance source documents promoted per query."}
    )
    fact_source_scan_min_fact_score: float = field(
        default=0.60,
        metadata={"help": "Minimum normalized query-to-fact score for source access."}
    )
    fact_source_scan_max_source_docs_per_fact: int = field(
        default=2,
        metadata={"help": "Maximum provenance documents considered for each exact fact tuple."}
    )
    fact_source_scan_selection_mode: str = field(
        default="legacy",
        metadata={"help": "Fact-source selection policy: legacy or obligation_aware."}
    )
    fact_source_scan_protect_dense_top_k: int = field(
        default=1,
        metadata={"help": "Minimum dense-retrieval prefix protected from fact-source replacement."}
    )
    enable_dual_channel_evidence_projection: bool = field(
        default=False,
        metadata={"help": "Whether to use a high-recall dense channel plus risk-aware evidence projection for bridge questions."}
    )
    dual_channel_projection_bridge_only: bool = field(
        default=True,
        metadata={"help": "Only run dual-channel projection for bridge or compositional questions."}
    )
    dual_channel_projection_candidate_top_k: int = field(
        default=80,
        metadata={"help": "Number of dense-channel candidates considered by dual-channel retrieval projection."}
    )
    dual_channel_projection_max_promotions: int = field(
        default=1,
        metadata={"help": "Maximum dense-channel candidates promoted into the retrieval prefix."}
    )
    dual_channel_projection_protect_top_k: int = field(
        default=1,
        metadata={"help": "Number of original top-ranked retrieval passages protected from dual-channel projection."}
    )
    dual_channel_projection_min_score: float = field(
        default=0.50,
        metadata={"help": "Minimum shadow-calibrated dual-channel score required before retrieval promotion."}
    )
    dual_channel_projection_min_query_overlap: float = field(
        default=0.08,
        metadata={"help": "Minimum query or anchor overlap used as a weak fallback for dense-channel candidates."}
    )
    dual_channel_projection_min_missing_gain: float = field(
        default=0.10,
        metadata={"help": "Minimum missing-slot evidence gain required for shadow-calibrated dense-channel projection."}
    )
    dual_channel_projection_min_bridge_link: float = field(
        default=0.06,
        metadata={"help": "Minimum bridge-link signal between projected evidence and the current evidence state."}
    )
    dual_channel_projection_shadow_top_k: int = field(
        default=8,
        metadata={"help": "Number of shadow-scored dense-channel candidates kept in diagnostics."}
    )
    dual_channel_projection_shadow_when_sufficient: bool = field(
        default=False,
        metadata={"help": "Run shadow scoring even when current lexical evidence coverage appears sufficient."}
    )
    dual_channel_projection_mode: str = field(
        default="append_only_or_top5",
        metadata={"help": "Dual-channel projection mode: aggressive_promote, append_only_or_top5, or append_only."}
    )
    dual_channel_projection_trigger_min_query_coverage: float = field(
        default=0.50,
        metadata={"help": "Skip dual-channel projection when current top-k evidence already covers this fraction of query tokens."}
    )
    dual_channel_projection_trigger_min_bridge_docs: int = field(
        default=2,
        metadata={"help": "Skip dual-channel projection when current top-k already has this many bridge-connected docs."}
    )
    dual_channel_projection_reader_max_append: int = field(
        default=1,
        metadata={"help": "Maximum accepted dense-channel projection docs appended to the QA reader context."}
    )
    dual_channel_projection_reader_soft_append_enabled: bool = field(
        default=False,
        metadata={"help": "Allow rejected shadow-channel candidates to be appended to the QA reader when they fill a low-risk evidence deficit without changing retrieval ranks."}
    )
    dual_channel_projection_reader_soft_append_require_insufficient: bool = field(
        default=True,
        metadata={"help": "Only soft-append shadow evidence when the current evidence state is insufficient."}
    )
    dual_channel_projection_reader_soft_append_min_score: float = field(
        default=0.42,
        metadata={"help": "Minimum shadow score for non-destructive reader-only shadow evidence."}
    )
    dual_channel_projection_reader_soft_append_min_query_overlap: float = field(
        default=0.06,
        metadata={"help": "Minimum question/anchor overlap for non-destructive shadow reader evidence."}
    )
    dual_channel_projection_reader_soft_append_min_missing_gain: float = field(
        default=0.05,
        metadata={"help": "Minimum missing-slot gain for non-destructive shadow reader evidence."}
    )
    dual_channel_projection_reader_soft_append_min_bridge_link: float = field(
        default=0.08,
        metadata={"help": "Minimum bridge-link signal for non-destructive shadow reader evidence."}
    )
    enable_constrained_chain_rerank: bool = field(
        default=False,
        metadata={"help": "Whether to rebuild the QA top-k with constrained path/chain evidence candidates."}
    )
    constrained_chain_rerank_candidate_top_k: int = field(
        default=200,
        metadata={"help": "Number of retrieved candidates considered by constrained chain reranking."}
    )
    constrained_chain_rerank_protect_top_k: int = field(
        default=1,
        metadata={"help": "Number of original top-ranked passages fixed during constrained chain reranking."}
    )
    constrained_chain_rerank_max_promotions: int = field(
        default=2,
        metadata={"help": "Maximum number of outside-top-k candidates promoted by constrained chain reranking."}
    )
    constrained_chain_rerank_min_score: float = field(
        default=0.30,
        metadata={"help": "Minimum chain score required for a candidate to be promoted."}
    )
    constrained_chain_rerank_bridge_only: bool = field(
        default=True,
        metadata={"help": "Only apply constrained chain reranking to bridge/composition questions."}
    )
    constrained_chain_rerank_require_insufficient: bool = field(
        default=True,
        metadata={"help": "Only apply constrained chain reranking when evidence sufficiency marks the current top-k insufficient."}
    )
    enable_iterative_retrieval: bool = field(
        default=False,
        metadata={"help": "Whether to run evidence sufficiency guided iterative retrieval."}
    )
    iterative_retrieval_bridge_only: bool = field(
        default=True,
        metadata={"help": "Only run iterative retrieval for bridge/composition questions."}
    )
    iterative_retrieval_max_steps: int = field(
        default=2,
        metadata={"help": "Maximum number of sufficiency-guided retrieval steps."}
    )
    iterative_retrieval_target_k: int = field(
        default=5,
        metadata={"help": "Top-k prefix assessed and rebuilt by iterative retrieval."}
    )
    iterative_retrieval_max_promotions: int = field(
        default=1,
        metadata={"help": "Maximum number of follow-up retrieval candidates promoted per step."}
    )
    iterative_retrieval_protect_top1: bool = field(
        default=True,
        metadata={"help": "Keep the original top-1 passage fixed during iterative retrieval merging."}
    )
    iterative_retrieval_protect_top_k: int = field(
        default=2,
        metadata={"help": "Keep this many original top-ranked passages fixed during iterative retrieval merging."}
    )
    iterative_retrieval_expansion_weight: float = field(
        default=0.72,
        metadata={"help": "Score weight assigned to follow-up retrieval candidates when merging."}
    )
    iterative_retrieval_min_expansion_score: float = field(
        default=0.40,
        metadata={"help": "Minimum follow-up retrieval score required before a passage can be promoted."}
    )
    iterative_retrieval_max_followup_snippets: int = field(
        default=3,
        metadata={"help": "Maximum evidence snippets used to build the follow-up query."}
    )
    iterative_retrieval_bridge_threshold: float = field(
        default=0.62,
        metadata={"help": "Evidence sufficiency threshold for bridge/composition questions."}
    )
    iterative_retrieval_non_bridge_threshold: float = field(
        default=0.48,
        metadata={"help": "Evidence sufficiency threshold for non-bridge questions."}
    )
    iterative_retrieval_min_query_coverage: float = field(
        default=0.50,
        metadata={"help": "Minimum question token coverage required before evidence is considered sufficient."}
    )
    iterative_retrieval_min_best_sentence_score: float = field(
        default=0.24,
        metadata={"help": "Minimum best evidence sentence score required before evidence is considered sufficient."}
    )
    enable_planned_bridge_retrieval: bool = field(
        default=False,
        metadata={"help": "Whether to run LLM-planned follow-up retrieval for bridge/composition evidence gaps."}
    )
    planned_bridge_retrieval_bridge_only: bool = field(
        default=False,
        metadata={"help": "Only run planned retrieval for bridge/composition questions."}
    )
    planned_bridge_retrieval_require_insufficient: bool = field(
        default=False,
        metadata={"help": "Only run planned retrieval when earlier evidence checks mark current evidence insufficient."}
    )
    planned_bridge_retrieval_max_queries: int = field(
        default=3,
        metadata={"help": "Maximum number of planned follow-up retrieval queries to execute."}
    )
    planned_bridge_retrieval_max_snippets: int = field(
        default=3,
        metadata={"help": "Maximum current evidence snippets shown to the planning prompt."}
    )
    planned_bridge_retrieval_max_tokens: int = field(
        default=160,
        metadata={"help": "Maximum completion tokens for the planned retrieval query-generation prompt."}
    )
    planned_bridge_retrieval_protect_top_k: int = field(
        default=1,
        metadata={"help": "Number of original top-ranked passages fixed during planned retrieval merging."}
    )
    planned_bridge_retrieval_max_promotions: int = field(
        default=2,
        metadata={"help": "Maximum number of planned retrieval candidates promoted into the QA prefix."}
    )
    planned_bridge_retrieval_plan_top_k: int = field(
        default=20,
        metadata={"help": "Number of candidates considered from each planned retrieval query."}
    )
    planned_bridge_retrieval_min_score: float = field(
        default=0.18,
        metadata={"help": "Minimum planned retrieval score required before a candidate can be promoted."}
    )
    planned_bridge_retrieval_merge_mode: str = field(
        default="guarded_promote",
        metadata={"help": "How planned retrieval merges evidence: append_only preserves retrieval top-k; guarded_promote safely promotes audited evidence; replace_topk is an aggressive ablation."}
    )
    planned_bridge_retrieval_max_extra_docs: int = field(
        default=3,
        metadata={"help": "Maximum planned retrieval evidence documents appended to the QA context in append_only mode."}
    )
    planned_bridge_retrieval_guarded_min_score: float = field(
        default=0.82,
        metadata={"help": "Minimum guarded evidence score required before planned retrieval can promote a passage into top-k."}
    )
    planned_bridge_retrieval_guarded_max_original_rank: int = field(
        default=80,
        metadata={"help": "Largest original retrieval rank allowed for guarded planned-retrieval promotion."}
    )
    planned_bridge_retrieval_guarded_max_plan_rank: int = field(
        default=8,
        metadata={"help": "Largest rank within a planned retrieval result allowed for guarded promotion."}
    )
    planned_bridge_retrieval_guarded_min_query_overlap: float = field(
        default=0.08,
        metadata={"help": "Minimum content-token overlap between the original question and planned query before guarded promotion."}
    )
    planned_bridge_retrieval_max_rounds: int = field(
        default=5,
        metadata={"help": "Maximum planned retrieval rounds available to iterative planners."}
    )
    planned_bridge_retrieval_verify_each_round: bool = field(
        default=True,
        metadata={"help": "Whether iterative planned retrieval should verify evidence sufficiency after each round."}
    )
    planned_bridge_retrieval_route_mode: str = field(
        default="adaptive",
        metadata={"help": "Question-type routing mode for planned retrieval: off or adaptive."}
    )
    planned_bridge_retrieval_query_workers: int = field(
        default=1,
        metadata={"help": "Number of worker threads used to execute planned retrieval queries inside one retrieval round. Keep at 1 for local embedding models unless query embeddings are precomputed."}
    )
    enable_planned_retrieval_cost_controller: bool = field(
        default=False,
        metadata={"help": "Whether a cost-aware fixed-point controller may stop unproductive planned-retrieval rounds."}
    )
    planned_retrieval_cost_controller_min_rounds: int = field(
        default=2,
        metadata={"help": "Minimum number of completed planned-retrieval rounds before an early stop is admissible."}
    )
    planned_retrieval_cost_controller_patience: int = field(
        default=2,
        metadata={"help": "Consecutive no-progress rounds required to declare a verifier/evidence fixed point."}
    )
    planned_retrieval_cost_controller_resolution_prior: float = field(
        default=0.12,
        metadata={"help": "Prior utility that another planned-retrieval round resolves the remaining evidence deficit."}
    )
    planned_retrieval_cost_controller_no_progress_decay: float = field(
        default=0.50,
        metadata={"help": "Multiplicative resolution-utility decay after each no-progress round."}
    )
    planned_retrieval_cost_controller_planner_cost: float = field(
        default=0.04,
        metadata={"help": "Normalized cost of one planner call in the next retrieval round."}
    )
    planned_retrieval_cost_controller_verifier_cost: float = field(
        default=0.04,
        metadata={"help": "Normalized cost of one verifier call in the next retrieval round."}
    )
    planned_retrieval_cost_controller_retriever_cost: float = field(
        default=0.015,
        metadata={"help": "Normalized cost per planned query retrieval in the next round."}
    )
    planned_retrieval_cost_controller_min_continue_net_benefit: float = field(
        default=0.0,
        metadata={"help": "Minimum expected utility minus normalized cost required to continue after a safe fixed point."}
    )
    planned_retrieval_cost_controller_candidate_frontier_mode: str = field(
        default="raw_candidates",
        metadata={"help": "Candidate stability signal for the planned-retrieval round controller: raw_candidates or decision_relevant."}
    )
    planned_retrieval_cost_controller_shadow_only: bool = field(
        default=False,
        metadata={"help": "Record counterfactual STOP decisions without terminating planned retrieval."}
    )
    enable_planned_residual_deficit_queries: bool = field(
        default=False,
        metadata={"help": "Generate deterministic residual-deficit follow-up queries from verifier missing slots when the LLM planner repeats or reaches its final round."}
    )
    planned_residual_deficit_max_queries: int = field(
        default=1,
        metadata={"help": "Maximum residual-deficit fallback queries added in one planned retrieval round."}
    )
    planned_residual_deficit_last_round_only: bool = field(
        default=False,
        metadata={"help": "Only add residual-deficit queries in the final planned retrieval round instead of whenever the planner produces no new query."}
    )
    planned_residual_deficit_bridge_only: bool = field(
        default=True,
        metadata={"help": "Only add residual-deficit queries for bridge/compositional closed-loop questions."}
    )
    planned_residual_deficit_require_explicit_slot: bool = field(
        default=True,
        metadata={"help": "Require verifier missing slots, missing fields, uncertain anchors, or verifier queries before generating residual-deficit queries."}
    )
    planned_residual_deficit_disable_long_context: bool = field(
        default=True,
        metadata={"help": "Disable residual-deficit queries for long-context and narrative routes to avoid noisy query drift."}
    )
    planned_residual_deficit_min_bridge_strength: int = field(
        default=2,
        metadata={"help": "Minimum router bridge strength required before residual-deficit queries may be added."}
    )
    planned_residual_deficit_reserve_after_round: int = field(
        default=2,
        metadata={"help": "After this 1-indexed closed-loop round, reserve one retrieval query slot for explicit residual deficits when possible."}
    )
    planned_residual_deficit_replace_last_query: bool = field(
        default=True,
        metadata={"help": "When residual-deficit is active and the planner fills the query budget, replace the last planner query with a deficit-targeted query."}
    )
    planned_residual_deficit_append_only_after_round: int = field(
        default=3,
        metadata={"help": "From this 1-indexed closed-loop round onward, residual-deficit retrieval is appended as reader evidence instead of entering rank-editing merger."}
    )
    planned_residual_deficit_append_candidate_top_k: int = field(
        default=3,
        metadata={"help": "Number of late residual retrieval candidates considered for append-only reader evidence."}
    )
    planned_residual_deficit_append_max_docs: int = field(
        default=1,
        metadata={"help": "Maximum append-only reader evidence docs selected from late residual retrieval in one round."}
    )
    planned_residual_deficit_rank_stable_after_round: int = field(
        default=3,
        metadata={"help": "From this 1-indexed closed-loop round onward, disable top-2 rank override to preserve early retrieval precision."}
    )
    enable_planned_top2_override: bool = field(
        default=False,
        metadata={"help": "Whether high-confidence planned retrieval evidence may replace the original rank-2 passage while preserving rank 1."}
    )
    planned_top2_override_min_score: float = field(
        default=0.96,
        metadata={"help": "Minimum planned-evidence score required for rank-2 override."}
    )
    planned_top2_override_max_original_rank: int = field(
        default=30,
        metadata={"help": "Largest original retrieval rank allowed for rank-2 override candidates."}
    )
    planned_top2_override_max_plan_rank: int = field(
        default=2,
        metadata={"help": "Largest rank inside a planned retrieval result allowed for rank-2 override."}
    )
    planned_top2_override_min_query_overlap: float = field(
        default=0.18,
        metadata={"help": "Minimum content-token overlap between original question and planned query for rank-2 override."}
    )
    planned_top2_override_min_query_hits: int = field(
        default=1,
        metadata={"help": "Minimum number of planned queries that must retrieve the same candidate before rank-2 override."}
    )
    planned_top2_intervention_gate_mode: str = field(
        default="off",
        metadata={"help": "Query-level gate for rank-2 interventions: off or context_state."}
    )
    planned_top2_gate_min_bridge_strength: int = field(
        default=2,
        metadata={"help": "Minimum bridge-strength signal required before context-state gating allows rank-2 intervention."}
    )
    planned_top2_gate_long_context_avg_doc_chars: float = field(
        default=1200.0,
        metadata={"help": "Average top-document character length above which rank-2 intervention is frozen unless bridge evidence is strong."}
    )
    planned_top2_gate_long_context_empty_title_ratio: float = field(
        default=0.50,
        metadata={"help": "Empty-title ratio threshold used with long evidence length to freeze rank-2 intervention."}
    )
    planned_top2_gate_long_context_min_bridge_strength: int = field(
        default=4,
        metadata={"help": "Bridge-strength threshold needed to keep rank-2 intervention active under long-context evidence."}
    )
    enable_planned_pairwise_utility_search: bool = field(
        default=False,
        metadata={"help": "Whether rank-2 intervention should search original and planned candidates by top1+candidate evidence-pair utility."}
    )
    planned_pairwise_utility_max_original_rank: int = field(
        default=30,
        metadata={"help": "Largest original retrieval rank considered by pairwise evidence utility search."}
    )
    planned_pairwise_utility_max_candidates: int = field(
        default=32,
        metadata={"help": "Maximum candidate passages evaluated by pairwise evidence utility search."}
    )
    planned_pairwise_utility_min_margin: float = field(
        default=0.12,
        metadata={"help": "Minimum pairwise utility margin over the original top1+rank2 pair."}
    )
    planned_pairwise_utility_min_pair_score: float = field(
        default=0.65,
        metadata={"help": "Minimum candidate top1+rank2 pair utility score required for replacement."}
    )
    planned_pairwise_utility_min_candidate_support: float = field(
        default=0.45,
        metadata={"help": "Minimum question/query support required from a pairwise utility candidate."}
    )
    planned_pairwise_utility_allowed_sources: str = field(
        default="planned_and_original",
        metadata={"help": "Comma-separated candidate sources allowed by pairwise utility search: planned_and_original, planned_retrieval, original_retrieval, or all."}
    )
    planned_pairwise_utility_source_policy: str = field(
        default="fixed",
        metadata={"help": "How to choose pairwise utility candidate sources per query: fixed or adaptive."}
    )
    planned_pairwise_utility_adaptive_min_bridge_strength: int = field(
        default=3,
        metadata={"help": "Minimum bridge-strength signal before adaptive pairwise source policy can recover original-retrieval candidates."}
    )
    planned_pairwise_utility_adaptive_max_avg_doc_chars: float = field(
        default=900.0,
        metadata={"help": "Maximum average top-document length before adaptive pairwise source policy keeps strict sources."}
    )
    planned_pairwise_utility_adaptive_allow_long_answer: bool = field(
        default=False,
        metadata={"help": "Whether adaptive pairwise source policy may recover original-retrieval candidates for long-answer signals."}
    )
    planned_pairwise_utility_original_fallback_only: bool = field(
        default=True,
        metadata={"help": "Use original-retrieval candidates only when no planned-supported pairwise candidate passes."}
    )
    enable_schema_verified_utility_prior: bool = field(
        default=False,
        metadata={"help": "Whether query-schema-memory agreement should provide a gated prior for pairwise utility scoring."}
    )
    schema_verified_prior_bonus: float = field(
        default=0.08,
        metadata={"help": "Maximum utility bonus from schema-verified evidence agreement."}
    )
    schema_verified_prior_min_schema_score: float = field(
        default=0.72,
        metadata={"help": "Minimum query-schema-memory score required before the schema prior can apply."}
    )
    schema_verified_prior_min_schema_idf: float = field(
        default=0.58,
        metadata={"help": "Minimum schema specificity required before the schema prior can apply."}
    )
    schema_verified_prior_min_overlap: float = field(
        default=0.18,
        metadata={"help": "Minimum query/context overlap required before the schema prior can apply."}
    )
    schema_verified_prior_max_original_rank: int = field(
        default=20,
        metadata={"help": "Largest original retrieval rank eligible for schema-verified utility prior."}
    )
    schema_verified_prior_numeric_requires_value: bool = field(
        default=True,
        metadata={"help": "Require date/number evidence in schema-prior candidates for temporal or numeric questions."}
    )
    schema_verified_prior_block_long_context: bool = field(
        default=True,
        metadata={"help": "Disable schema-prior rank intervention in long-context or narrative states."}
    )
    schema_verified_prior_rescue_enabled: bool = field(
        default=False,
        metadata={"help": "Allow strong schema prior to rescue only low pair-score/margin/support rejections. Disabled by default; the prior normally acts as a utility bonus only."}
    )
    schema_verified_prior_min_rescue_bonus: float = field(
        default=0.045,
        metadata={"help": "Minimum schema-prior bonus required before soft pairwise rejections can be rescued."}
    )
    enable_answer_type_conflict_guard: bool = field(
        default=False,
        metadata={"help": "Whether pairwise rank-2 replacement should reject candidates whose answer-type value conflicts with strong original evidence."}
    )
    answer_type_conflict_guard_min_old_signal: float = field(
        default=0.55,
        metadata={"help": "Minimum original rank-2 answer-value support before answer-type conflict guarding is active."}
    )
    answer_type_conflict_guard_min_candidate_signal: float = field(
        default=0.35,
        metadata={"help": "Minimum candidate answer-value support treated as a conflicting answer-type value."}
    )
    answer_type_conflict_guard_min_conflict_margin: float = field(
        default=0.08,
        metadata={"help": "Minimum candidate-vs-original answer-value support margin required to reject a conflicting candidate."}
    )
    answer_type_conflict_guard_numeric_only: bool = field(
        default=True,
        metadata={"help": "Restrict answer-type conflict guarding to date and number questions."}
    )
    enable_planned_support_set_rerank: bool = field(
        default=False,
        metadata={"help": "Whether planned retrieval should rerank the whole top evidence support set instead of only rank-2."}
    )
    planned_support_set_target_k: int = field(
        default=5,
        metadata={"help": "Number of top evidence passages optimized by support-set utility reranking."}
    )
    planned_support_set_protect_top_k: int = field(
        default=2,
        metadata={"help": "Protected prefix size for support-set utility reranking. Defaults to preserving top-2 recall."}
    )
    planned_support_set_max_replacements: int = field(
        default=2,
        metadata={"help": "Maximum number of original top-k passages support-set utility reranking may replace."}
    )
    planned_support_set_max_candidates: int = field(
        default=96,
        metadata={"help": "Maximum candidate passages profiled by support-set utility reranking."}
    )
    planned_support_set_max_original_rank: int = field(
        default=200,
        metadata={"help": "Deepest original retrieval rank admitted into support-set utility reranking."}
    )
    planned_support_set_min_candidate_score: float = field(
        default=0.52,
        metadata={"help": "Minimum utility score for a non-visible passage to enter support-set reranking."}
    )
    planned_support_set_redundancy_penalty: float = field(
        default=0.18,
        metadata={"help": "Penalty applied to redundant passages during greedy support-set utility selection."}
    )
    planned_support_set_preserve_original_topk: bool = field(
        default=True,
        metadata={"help": "Require support-set replacements to preserve original top-k question/anchor coverage."}
    )
    planned_support_set_preserve_min_gain: float = field(
        default=0.03,
        metadata={"help": "Minimum coverage-preserving gain required before support-set reranking may replace an original top-k passage."}
    )
    planned_support_set_preserve_max_coverage_loss: float = field(
        default=0.02,
        metadata={"help": "Maximum query-token coverage loss tolerated when support-set reranking replaces an original top-k passage."}
    )
    enable_support_set_completion_evidence: bool = field(
        default=False,
        metadata={"help": "Append missing-slot support evidence to the reader without changing retrieval ranking."}
    )
    support_set_completion_bridge_only: bool = field(
        default=True,
        metadata={"help": "Only append support-set completion evidence for bridge/compositional questions."}
    )
    support_set_completion_candidate_top_k: int = field(
        default=20,
        metadata={"help": "Number of ranked retrieved passages considered for reader-only support completion."}
    )
    support_set_completion_max_append: int = field(
        default=1,
        metadata={"help": "Maximum support-completion passages appended to the reader."}
    )
    support_set_completion_min_score: float = field(
        default=0.54,
        metadata={"help": "Minimum support-completion utility score required for append-only evidence."}
    )
    support_set_completion_min_missing_gain: float = field(
        default=0.12,
        metadata={"help": "Minimum missing-query-slot coverage required for support-completion evidence."}
    )
    support_set_completion_min_sentence_score: float = field(
        default=0.30,
        metadata={"help": "Minimum best sentence support score required for support-completion evidence."}
    )
    enable_entity_centric_fallback_evidence: bool = field(
        default=False,
        metadata={"help": "Append one deep entity-title-matched evidence view for single-hop factoid questions without changing retrieval ranks."}
    )
    entity_centric_fallback_candidate_top_k: int = field(
        default=80,
        metadata={"help": "Number of stable ranked passages scanned by the entity-centric deep evidence fallback."}
    )
    entity_centric_fallback_max_append: int = field(
        default=1,
        metadata={"help": "Maximum entity-centric fallback evidence views inserted into the reader context."}
    )
    entity_centric_fallback_min_score: float = field(
        default=0.44,
        metadata={"help": "Minimum utility score for entity-centric fallback evidence admission."}
    )
    entity_centric_fallback_min_title_anchor_overlap: float = field(
        default=0.58,
        metadata={"help": "Minimum question entity/title anchor overlap for entity-centric fallback evidence."}
    )
    entity_centric_fallback_min_subject_title_overlap: float = field(
        default=0.85,
        metadata={"help": "Minimum extracted question-subject overlap with fallback title."}
    )
    entity_centric_fallback_min_subject_doc_overlap: float = field(
        default=0.90,
        metadata={"help": "Minimum extracted question-subject overlap with fallback title/body when title overlap is weaker."}
    )
    entity_centric_fallback_min_sentence_score: float = field(
        default=0.20,
        metadata={"help": "Minimum best evidence sentence score for entity-centric fallback evidence."}
    )
    entity_centric_fallback_min_target_signal: float = field(
        default=0.20,
        metadata={"help": "Minimum answer-type signal required by entity-centric fallback evidence."}
    )
    entity_centric_fallback_insert_after: int = field(
        default=5,
        metadata={"help": "Insert entity-centric fallback evidence after this many reader documents."}
    )
    entity_centric_fallback_block_bridge: bool = field(
        default=True,
        metadata={"help": "Disable entity-centric fallback on bridge/compositional questions."}
    )
    entity_centric_fallback_block_bridge_like_relation: bool = field(
        default=True,
        metadata={"help": "Disable entity-centric fallback on relation-chain questions that look single-hop but require bridge resolution."}
    )
    entity_centric_fallback_strict_subject_match: bool = field(
        default=True,
        metadata={"help": "Require fallback evidence to match the extracted question subject, not only loose query anchors."}
    )
    entity_centric_fallback_require_slot_signal: bool = field(
        default=True,
        metadata={"help": "Require the fallback evidence sentence to support the query answer slot/predicate."}
    )
    entity_centric_fallback_block_ambiguous_short_subject: bool = field(
        default=True,
        metadata={"help": "Reject short ambiguous subject matches such as title pages that only share a first name."}
    )
    entity_centric_fallback_current_sufficiency_guard: bool = field(
        default=True,
        metadata={"help": "Skip fallback when current reader evidence already contains subject, slot, and answer-type signals."}
    )
    entity_centric_fallback_block_empty_factoid_slot: bool = field(
        default=True,
        metadata={"help": "Quarantine factoid fallback candidates when the query has no explicit answer-slot/predicate terms."}
    )
    entity_centric_fallback_quarantine_generic_titles: bool = field(
        default=True,
        metadata={"help": "Keep generic topic/role pages as diagnostics instead of appending them to the reader."}
    )
    entity_centric_fallback_route_aware_relaxation: bool = field(
        default=True,
        metadata={"help": "Relax entity fallback only for route-level bridge deficit states."}
    )
    entity_centric_fallback_route_aware_max_rank: int = field(
        default=24,
        metadata={"help": "Maximum original retrieval rank allowed by route-aware entity fallback relaxation."}
    )
    entity_centric_fallback_route_aware_min_score: float = field(
        default=0.54,
        metadata={"help": "Minimum utility score under route-aware entity fallback relaxation."}
    )
    entity_centric_fallback_route_aware_min_sentence_score: float = field(
        default=0.32,
        metadata={"help": "Minimum evidence sentence score under route-aware entity fallback relaxation."}
    )
    entity_centric_fallback_route_aware_min_question_overlap: float = field(
        default=0.20,
        metadata={"help": "Minimum question overlap under route-aware entity fallback relaxation."}
    )
    enable_answer_slot_support_completion: bool = field(
        default=False,
        metadata={"help": "Append a compact answer-slot support view to the reader without changing retrieval ranks."}
    )
    answer_slot_support_completion_bridge_or_attribute_only: bool = field(
        default=True,
        metadata={"help": "Only insert answer-slot support completion for bridge/compositional or single-hop attribute queries."}
    )
    answer_slot_support_completion_candidate_top_k: int = field(
        default=40,
        metadata={"help": "Number of stable ranked passages scanned for answer-slot support completion."}
    )
    answer_slot_support_completion_max_rows: int = field(
        default=1,
        metadata={"help": "Maximum role/slot evidence rows inserted into the answer-slot support view."}
    )
    answer_slot_support_completion_min_row_score: float = field(
        default=0.52,
        metadata={"help": "Minimum row utility score for answer-slot support completion admission."}
    )
    answer_slot_support_completion_min_slot_signal: float = field(
        default=0.60,
        metadata={"help": "Minimum expected answer-slot signal required for a support-completion row."}
    )
    answer_slot_support_completion_min_anchor_or_chain: float = field(
        default=0.18,
        metadata={"help": "Minimum anchor overlap or current-evidence chain link required for a support-completion row."}
    )
    answer_slot_support_completion_min_missing_gain: float = field(
        default=0.00,
        metadata={"help": "Minimum missing-query-token gain required for a support-completion row."}
    )
    answer_slot_support_completion_insert_after: int = field(
        default=2,
        metadata={"help": "Insert the answer-slot support view after this many final QA evidence documents."}
    )
    answer_slot_support_completion_include_current_rows: bool = field(
        default=True,
        metadata={"help": "Allow high-confidence rows from current QA evidence, not only deeper ranked passages."}
    )
    answer_slot_support_completion_lineage_safe: bool = field(
        default=True,
        metadata={"help": "Require source/subject lineage and conflict-safe admission before inserting the answer-slot view."}
    )
    answer_slot_support_completion_allow_bridge: bool = field(
        default=False,
        metadata={"help": "Allow answer-slot support completion on bridge/compositional questions under strict lineage constraints."}
    )
    answer_slot_support_completion_min_relation_overlap: float = field(
        default=0.08,
        metadata={"help": "Minimum relation-role overlap required by lineage-safe answer-slot completion."}
    )
    answer_slot_support_completion_min_subject_overlap: float = field(
        default=0.55,
        metadata={"help": "Minimum question-subject overlap required by lineage-safe single-hop answer-slot completion."}
    )
    answer_slot_support_completion_conflict_margin: float = field(
        default=0.05,
        metadata={"help": "Block insertion when another lineage-compatible slot row is within this score margin."}
    )
    answer_slot_support_completion_block_boolean_comparison: bool = field(
        default=True,
        metadata={"help": "Block answer-slot completion for boolean, comparison, ordinal, or superlative questions."}
    )
    answer_slot_support_completion_block_list_like_sources: bool = field(
        default=True,
        metadata={"help": "Reject disambiguation/list-like passages before answer-slot support admission."}
    )
    enable_structured_evidence_chain: bool = field(
        default=False,
        metadata={"help": "Whether to insert a compact role-ordered evidence chain before QA."}
    )
    structured_evidence_chain_bridge_only: bool = field(
        default=True,
        metadata={"help": "Only build structured evidence chains for bridge or compositional questions."}
    )
    structured_evidence_chain_candidate_top_k: int = field(
        default=24,
        metadata={"help": "Number of ranked passages considered for missing-slot chain sentences."}
    )
    structured_evidence_chain_max_sentences: int = field(
        default=4,
        metadata={"help": "Maximum sentences in the structured evidence chain document."}
    )
    structured_evidence_chain_max_candidate_sentences: int = field(
        default=1,
        metadata={"help": "Maximum external candidate sentences admitted into one chain document."}
    )
    structured_evidence_chain_min_sentence_score: float = field(
        default=0.27,
        metadata={"help": "Minimum sentence utility score for chain role selection."}
    )
    structured_evidence_chain_min_chain_score: float = field(
        default=0.52,
        metadata={"help": "Minimum aggregate chain score required before insertion."}
    )
    structured_evidence_chain_insert_after: int = field(
        default=2,
        metadata={"help": "Insert the structured evidence chain after this many leading QA docs."}
    )
    structured_evidence_chain_skip_single_hop: bool = field(
        default=True,
        metadata={"help": "Skip structured evidence chain insertion for routed single-hop attribute queries."}
    )
    structured_evidence_chain_require_evidence_gap: bool = field(
        default=True,
        metadata={"help": "Only insert structured chains when the retrieval controller reports an unresolved evidence gap."}
    )
    structured_evidence_chain_allow_long_context: bool = field(
        default=False,
        metadata={"help": "Allow structured chains for long-context or narrative-style routed questions."}
    )
    structured_evidence_chain_min_bridge_strength: int = field(
        default=3,
        metadata={"help": "Minimum router bridge-strength level required for bridge-only structured chains."}
    )
    structured_evidence_chain_skip_if_top2_changed: bool = field(
        default=True,
        metadata={"help": "Skip structured reader chains when retrieval has already changed rank-2 evidence."}
    )
    enable_planned_bridge_slot_rescue: bool = field(
        default=False,
        metadata={"help": "Whether rejected pairwise rank-2 candidates can be rescued when they cover a missing bridge slot."}
    )
    planned_bridge_slot_rescue_min_score: float = field(
        default=0.58,
        metadata={"help": "Minimum bridge-slot coverage score required to rescue a pairwise rank-2 candidate."}
    )
    planned_bridge_slot_rescue_min_improvement: float = field(
        default=0.08,
        metadata={"help": "Minimum bridge-slot coverage improvement over the original rank-2 passage."}
    )
    planned_bridge_slot_rescue_min_anchor_overlap: float = field(
        default=0.34,
        metadata={"help": "Minimum entity-anchor overlap required by bridge-slot coverage rescue when anchors are available."}
    )
    planned_bridge_slot_rescue_require_relation: bool = field(
        default=True,
        metadata={"help": "Whether bridge-slot rescue requires a non-generic inferred relation slot."}
    )
    planned_bridge_slot_rescue_require_old_deficit: bool = field(
        default=True,
        metadata={"help": "Only allow bridge-slot rescue to replace rank 2 when the original top-2 pair has an evidence deficit."}
    )
    planned_bridge_slot_rescue_deficit_max_old_score: float = field(
        default=0.50,
        metadata={"help": "Maximum original rank-2 bridge-slot score treated as deficient for replacement."}
    )
    planned_bridge_slot_rescue_min_missing_gain: float = field(
        default=0.15,
        metadata={"help": "Minimum missing-slot support gain required for bridge-slot rescue."}
    )
    planned_bridge_slot_rescue_append_on_no_deficit: bool = field(
        default=True,
        metadata={"help": "Append bridge-slot evidence to QA context when replacement is blocked by a sufficient original top-2 pair."}
    )
    enable_object_consistent_evidence_memory: bool = field(
        default=False,
        metadata={"help": "Whether bridge-slot append evidence is converted into compact answer-object-consistent memory instead of appending full passages."}
    )
    object_consistent_memory_min_score: float = field(
        default=0.42,
        metadata={"help": "Minimum sentence score required by object-consistent evidence memory."}
    )
    object_consistent_memory_max_sentences: int = field(
        default=2,
        metadata={"help": "Maximum selected sentences in one object-consistent evidence memory document."}
    )
    object_consistent_memory_max_objects: int = field(
        default=1,
        metadata={"help": "Maximum distinct answer-object candidates allowed in one object-consistent memory document."}
    )
    object_consistent_memory_max_chars: int = field(
        default=720,
        metadata={"help": "Maximum characters in one object-consistent evidence memory document."}
    )
    object_consistent_memory_append_only: bool = field(
        default=True,
        metadata={"help": "Append object-consistent evidence memory to QA without changing retrieval top-k."}
    )
    object_consistent_memory_append_max_items: int = field(
        default=1,
        metadata={"help": "Maximum append-only object-consistent memory documents per query."}
    )
    object_consistent_memory_append_candidate_top_k: int = field(
        default=12,
        metadata={"help": "Number of planned candidates considered for append-only object memory."}
    )
    object_consistent_memory_append_min_plan_score: float = field(
        default=0.50,
        metadata={"help": "Minimum planned retrieval score for append-only object memory candidates."}
    )
    enable_budgeted_evidence_subgraph: bool = field(
        default=False,
        metadata={"help": "Whether append-only planned evidence should be selected by a budgeted utility subgraph selector."}
    )
    budgeted_evidence_subgraph_max_append: int = field(
        default=1,
        metadata={"help": "Maximum append-only evidence passages selected by the budgeted evidence subgraph."}
    )
    budgeted_evidence_subgraph_max_candidates: int = field(
        default=48,
        metadata={"help": "Maximum candidate passages profiled by the budgeted evidence subgraph selector."}
    )
    budgeted_evidence_subgraph_min_utility: float = field(
        default=0.48,
        metadata={"help": "Minimum utility score required before a candidate can be appended by the budgeted evidence subgraph."}
    )
    budgeted_evidence_subgraph_redundancy_penalty: float = field(
        default=0.16,
        metadata={"help": "Penalty applied to redundant passages during budgeted evidence subgraph selection."}
    )
    budgeted_evidence_subgraph_require_bridge_or_gap: bool = field(
        default=True,
        metadata={"help": "Require a bridge, schema, object-memory, or missing-slot signal before append-only subgraph selection."}
    )
    enable_decoupled_evidence_frontier: bool = field(
        default=False,
        metadata={"help": "Build a query-time evidence frontier for append/promotion decisions even when rank-2 override is disabled."}
    )
    enable_promotion_utility_veto: bool = field(
        default=False,
        metadata={"help": "Use a budgeted evidence-utility check as a veto for rank promotions instead of selecting append-only evidence."}
    )
    promotion_utility_veto_min_utility: float = field(
        default=0.18,
        metadata={"help": "Minimum budgeted evidence utility required before a planned candidate may change retrieval rank."}
    )
    promotion_utility_veto_require_bridge_or_gap: bool = field(
        default=True,
        metadata={"help": "Require bridge, schema, answer-type, or missing-slot signal before allowing a rank promotion."}
    )
    promotion_utility_veto_apply_to_guarded: bool = field(
        default=False,
        metadata={"help": "Apply the promotion utility veto to guarded top-5 promotions in addition to rank-2 override."}
    )
    promotion_utility_veto_high_risk_only: bool = field(
        default=True,
        metadata={"help": "Only apply the promotion utility veto to high-risk sources such as schema-only or shadow candidates."}
    )
    promotion_utility_veto_append_on_reject: bool = field(
        default=True,
        metadata={"help": "Downgrade vetoed promotion candidates to append-only reader evidence instead of discarding them."}
    )
    enable_planned_top2_pair_verifier: bool = field(
        default=False,
        metadata={"help": "Whether rank-2 override must pass a counterfactual top1+rank2 evidence-pair comparison."}
    )
    planned_top2_pair_verifier_mode: str = field(
        default="hard",
        metadata={"help": "How the pair verifier affects rank-2 override: hard or soft."}
    )
    planned_top2_pair_verifier_score_weight: float = field(
        default=0.12,
        metadata={"help": "Soft reranking weight applied to the counterfactual pair margin for rank-2 override."}
    )
    planned_top2_pair_verifier_min_margin: float = field(
        default=0.04,
        metadata={"help": "Minimum pair-score margin required before replacing the original rank-2 passage."}
    )
    planned_top2_pair_verifier_max_coverage_loss: float = field(
        default=0.02,
        metadata={"help": "Maximum allowed question-token coverage loss in the replacement top1+rank2 pair."}
    )
    planned_top2_pair_verifier_min_candidate_query_support: float = field(
        default=0.45,
        metadata={"help": "Minimum planned-query token support required in a rank-2 override candidate passage."}
    )
    enable_planned_relation_object_guard: bool = field(
        default=False,
        metadata={"help": "Whether rank-2 override should reject candidates that conflict with strong original relation-object evidence."}
    )
    planned_relation_object_guard_min_score: float = field(
        default=0.55,
        metadata={"help": "Minimum relation-object cue score for the original rank-2 passage before conflict guarding."}
    )
    planned_relation_object_guard_min_margin: float = field(
        default=0.08,
        metadata={"help": "Minimum relation-object score margin used by the conflict guard."}
    )
    enable_planned_entity_conditioned_relation_guard: bool = field(
        default=False,
        metadata={"help": "Whether the relation-object guard should condition relation objects on question/planned-query entity anchors."}
    )
    planned_entity_relation_anchor_min_overlap: float = field(
        default=0.50,
        metadata={"help": "Minimum token overlap between a relation sentence and an entity anchor before entity-conditioned guarding uses that sentence."}
    )
    enable_graph_repair_retrieval: bool = field(
        default=False,
        metadata={"help": "Whether to repair weak graph retrieval by promoting dense candidates with graph/anchor signals."}
    )
    graph_repair_dense_top_k: int = field(
        default=80,
        metadata={"help": "Number of dense candidates considered by graph repair retrieval."}
    )
    graph_repair_max_promotions: int = field(
        default=2,
        metadata={"help": "Maximum dense candidates promoted by graph repair retrieval."}
    )
    graph_repair_protect_top_k: int = field(
        default=1,
        metadata={"help": "Number of leading retrieval results protected from graph repair promotion."}
    )
    graph_repair_min_score: float = field(
        default=0.22,
        metadata={"help": "Minimum graph repair score required for a dense candidate promotion."}
    )
    graph_repair_min_top_facts: int = field(
        default=1,
        metadata={"help": "Run graph repair when reranked top facts are below this count."}
    )
    graph_repair_min_candidate_facts: int = field(
        default=2,
        metadata={"help": "Run graph repair when fact candidates before rerank are below this count."}
    )
    graph_repair_trigger_on_dense_fallback: bool = field(
        default=True,
        metadata={"help": "Run graph repair when the initial graph retrieval falls back to dense retrieval."}
    )
    enable_schema_memory_graph: bool = field(
        default=True,
        metadata={"help": "Whether to add MemGraphRAG-inspired weak passage links for shared relation schemas during graph construction."}
    )
    schema_memory_edge_weight: float = field(
        default=0.08,
        metadata={"help": "Weight of each schema-memory passage edge."}
    )
    schema_memory_max_passages_per_schema: int = field(
        default=80,
        metadata={"help": "Maximum passages linked for one relation schema."}
    )
    schema_memory_neighbors_per_passage: int = field(
        default=2,
        metadata={"help": "Number of nearby same-schema passages linked from each passage."}
    )
    schema_memory_min_relation_tokens: int = field(
        default=1,
        metadata={"help": "Minimum informative relation tokens required before a triple contributes schema-memory graph edges."}
    )
    enable_query_schema_memory: bool = field(
        default=True,
        metadata={"help": "Whether to retrieve passages from query-conditioned relation-schema memory."}
    )
    enable_schema_catalog_v2: bool = field(
        default=False,
        metadata={
            "help": (
                "Use directional typed signatures with selective canonical "
                "predicate families and catalog cardinality statistics."
            )
        }
    )
    schema_catalog_v2_max_residual_signatures: int = field(
        default=24,
        metadata={
            "help": (
                "Maximum catalog signatures compiled from residual query "
                "predicate families before entity-bound probing."
            )
        }
    )
    query_schema_memory_merge_mode: str = field(
        default="candidate_only",
        metadata={"help": "How query schema memory is merged: candidate_only for pairwise arbitration, guarded_promote for direct ablations."}
    )
    query_schema_memory_bridge_only: bool = field(
        default=True,
        metadata={"help": "Only apply query-conditioned schema memory to questions with bridge or composition cues."}
    )
    query_schema_memory_disable_long_context: bool = field(
        default=True,
        metadata={"help": "Skip schema-memory promotion for long-context evidence unless the question has strong bridge cues."}
    )
    query_schema_memory_long_context_avg_doc_chars: float = field(
        default=1200.0,
        metadata={"help": "Average top-document length threshold used by the long-context schema-memory gate."}
    )
    query_schema_memory_long_context_min_bridge_strength: int = field(
        default=3,
        metadata={"help": "Minimum bridge-strength signal needed to run schema memory under long-context evidence."}
    )
    query_schema_memory_max_schema_frequency: int = field(
        default=120,
        metadata={"help": "Ignore relation schemas that appear in more passages than this query-time frequency cap."}
    )
    query_schema_memory_max_chunks_per_schema: int = field(
        default=32,
        metadata={"help": "Maximum passages considered from one relation schema at query time."}
    )
    query_schema_memory_candidate_top_k: int = field(
        default=48,
        metadata={"help": "Maximum schema-memory candidates scored per query."}
    )
    query_schema_memory_max_promotions: int = field(
        default=1,
        metadata={"help": "Maximum query-schema-memory candidates promoted into the retrieval prefix."}
    )
    query_schema_memory_protect_top_k: int = field(
        default=1,
        metadata={"help": "Number of leading retrieval results protected from schema-memory promotion."}
    )
    query_schema_memory_min_score: float = field(
        default=0.52,
        metadata={"help": "Minimum query-schema-memory utility score required for promotion."}
    )
    query_schema_memory_min_query_overlap: float = field(
        default=0.06,
        metadata={"help": "Minimum question-token overlap for schema-memory promotion candidates."}
    )
    query_schema_memory_min_schema_idf: float = field(
        default=0.18,
        metadata={"help": "Minimum schema specificity score required for schema-memory promotion candidates."}
    )
    query_schema_memory_join_key_guard: bool = field(
        default=True,
        metadata={"help": "Require same-schema candidates to share a fact endpoint join key with the active query facts."}
    )
    query_schema_memory_entity_bound_join: bool = field(
        default=True,
        metadata={"help": "Require schema-memory join keys to bind a concrete endpoint entity using exact or high-Jaccard matching."}
    )
    query_schema_memory_min_join_key_overlap: float = field(
        default=0.50,
        metadata={"help": "Minimum endpoint-token containment required by the query schema-memory join-key guard."}
    )
    enable_query_schema_memory_cross_schema_join: bool = field(
        default=True,
        metadata={"help": "Probe the provenance fact store by bound endpoints and merge cross-schema residual fact rows into Schema Memory."}
    )
    enable_schema_bind_fact_join: bool = field(
        default=False,
        metadata={"help": "Pipeline entity-bound Schema Memory tuples into the relation fact join as physical bind inputs."}
    )
    enable_schema_bound_semijoin: bool = field(
        default=False,
        metadata={"help": "Execute role-indexed semijoins over Schema Memory bind tuples."}
    )
    enable_schema_bound_residual_gate: bool = field(
        default=False,
        metadata={"help": "Require a schema-bound continuation tuple to satisfy the unresolved query predicate."}
    )
    enable_schema_bound_terminal_slot_gate: bool = field(
        default=False,
        metadata={"help": "Require a schema-bound continuation value to satisfy the compiled terminal answer slot."}
    )
    enable_schema_bound_predicate_family_index: bool = field(
        default=False,
        metadata={"help": "Use canonical predicate families in a role/entity/predicate composite index for schema-bound semijoins."}
    )
    enable_terminal_variable_bound_fact_join: bool = field(
        default=False,
        metadata={"help": "Require every schema-bound fact join to originate from a query-bound endpoint and materialize only its terminal variable."}
    )
    terminal_variable_fact_join_min_anchor_overlap: float = field(
        default=0.67,
        metadata={"help": "Minimum query-anchor containment for the source tuple of a terminal-variable fact join."}
    )
    enable_lineage_directed_terminal_join: bool = field(
        default=False,
        metadata={
            "help": (
                "Compile terminal joins as query-root -> bridge -> terminal "
                "lineages and reject reverse sibling expansion."
            )
        }
    )
    enable_single_trace_fact_join_replay: bool = field(
        default=False,
        metadata={"help": "Replay lexical/canonical logical joins and scan/index physical joins on one frozen query trace without changing reader evidence."}
    )
    enable_stage_aware_fact_join_plan: bool = field(
        default=False,
        metadata={"help": "Compile fact joins into typed bridge-binding and final-answer stages."}
    )
    enable_role_directed_fact_join: bool = field(
        default=True,
        metadata={"help": "Enforce directional endpoint-role contracts for typed fact joins."}
    )
    enable_canonical_bound_completion: bool = field(
        default=True,
        metadata={"help": "Canonicalize root bindings and issue structured predicate-bound completion queries."}
    )
    enable_completion_selectivity_feedback: bool = field(
        default=True,
        metadata={"help": "Use observed bound/predicate selectivity to choose predicate-gap tuple repair or stop."}
    )
    enable_query_plan_root_contract: bool = field(
        default=True,
        metadata={"help": "Reject answer-slot/schema phrases when binding the instance consumed by the first query-plan operator."}
    )
    enable_inverse_relation_completion: bool = field(
        default=True,
        metadata={"help": "Recover explicit inverse kinship and creator expressions missed by OpenIE during bound completion."}
    )
    enable_title_grounded_completion: bool = field(
        default=False,
        metadata={"help": "Use a safely matched retrieved page title as instance identity for terminal tuple completion."}
    )
    enable_source_grounded_terminal_repair: bool = field(
        default=False,
        metadata={"help": "Recover typed plan tuples only from title-bound source sentences and preserve document/sentence provenance."}
    )
    enable_title_catalog_root_plan: bool = field(
        default=False,
        metadata={"help": "Compile a speculative query-plan root from safely matched title catalog identities."}
    )
    enable_title_root_plan_commit: bool = field(
        default=False,
        metadata={"help": "Allow a terminal-complete title-catalog root plan to replace an incomplete legacy root plan."}
    )
    enable_identity_safe_root_commit: bool = field(
        default=False,
        metadata={"help": "Veto a partial legacy entity root when a longer exact title identity covers the same query span."}
    )
    enable_completion_evidence_pool: bool = field(
        default=True,
        metadata={"help": "Seed bound completion with current high-ranked evidence before filling the probe budget with dense retrieval."}
    )
    relation_fact_join_reader_stage_mode: str = field(
        default="legacy",
        metadata={"help": "Reader-visible fact-join stage policy: legacy, control, bridge_only, or final_slot_validated."}
    )
    enable_slot_dependency_cost_admission: bool = field(
        default=True,
        metadata={"help": "Apply non-redundancy and risk/cost admission to complete slot-dependency fact-join rows."}
    )
    enable_bound_completion_fact_join: bool = field(
        default=False,
        metadata={"help": "After an exact fact-index miss, issue a bound predicate-aware dense completion probe."}
    )
    bound_completion_top_k: int = field(
        default=6,
        metadata={"help": "Number of passages inspected by each bound completion probe."}
    )
    bound_completion_max_calls: int = field(
        default=2,
        metadata={"help": "Maximum bound completion probes per query plan."}
    )
    enable_bound_completion_cost_admission: bool = field(
        default=True,
        metadata={"help": "Enforce utility, risk, and compute budgets before a bound completion probe."}
    )
    bound_completion_max_risk: float = field(
        default=0.42,
        metadata={"help": "Maximum estimated risk allowed for a bound completion probe."}
    )
    bound_completion_min_net_utility: float = field(
        default=0.20,
        metadata={"help": "Minimum risk- and cost-adjusted utility for a bound completion probe."}
    )
    bound_completion_materialize_reader: bool = field(
        default=False,
        metadata={
            "help": (
                "Expose only complete typed bound-completion paths to the "
                "reader; false keeps those paths in shadow diagnostics."
            )
        }
    )
    enable_provenance_witness_repair: bool = field(
        default=False,
        metadata={
            "help": (
                "For a complete fact-join row with insufficient terminal "
                "support, retrieve and validate an independent tuple witness."
            )
        }
    )
    provenance_witness_top_k: int = field(
        default=6,
        metadata={"help": "Number of passages inspected by each provenance witness lookup."}
    )
    provenance_witness_max_calls: int = field(
        default=2,
        metadata={"help": "Maximum provenance witness lookups per question."}
    )
    query_schema_memory_cross_schema_max_candidates: int = field(
        default=48,
        metadata={"help": "Maximum cross-schema provenance fact rows exposed to Schema Memory admission."}
    )
    query_schema_memory_cross_schema_max_endpoint_scan_rows: int = field(
        default=48,
        metadata={"help": "Maximum bound-endpoint fact rows scanned before a cross-schema Schema Memory candidate is considered too costly."}
    )
    query_schema_memory_cross_schema_max_original_rank: int = field(
        default=200,
        metadata={"help": "Deepest retrieval rank considered by the cross-schema Schema Memory access path."}
    )
    query_schema_memory_cross_schema_min_relation_overlap: float = field(
        default=0.20,
        metadata={"help": "Minimum residual-predicate overlap for cross-schema Schema Memory fact rows."}
    )
    query_schema_memory_cross_schema_min_score: float = field(
        default=0.58,
        metadata={"help": "Minimum local utility score for cross-schema Schema Memory fact rows."}
    )
    enable_query_schema_memory_materialization: bool = field(
        default=True,
        metadata={"help": "Expose strictly join-connected schema-memory candidates to reader-side cost admission."}
    )
    query_schema_memory_materialization_max_candidates: int = field(
        default=1,
        metadata={"help": "Maximum schema-memory support passages considered for reader-only materialization."}
    )
    query_schema_memory_materialization_max_original_rank: int = field(
        default=32,
        metadata={"help": "Deepest original rank allowed for schema-memory support materialization."}
    )
    query_schema_memory_materialization_min_score: float = field(
        default=0.70,
        metadata={"help": "Minimum schema-memory candidate score for reader-only materialization."}
    )
    query_schema_memory_materialization_min_schema_idf: float = field(
        default=0.60,
        metadata={"help": "Minimum schema specificity for reader-only schema materialization."}
    )
    query_schema_memory_materialization_min_query_overlap: float = field(
        default=0.10,
        metadata={"help": "Minimum question overlap for reader-only schema materialization."}
    )
    query_schema_memory_materialization_min_join_key_overlap: float = field(
        default=0.66,
        metadata={"help": "Minimum entity-endpoint Jaccard overlap for reader-only schema materialization."}
    )
    query_schema_memory_materialization_min_relation_overlap: float = field(
        default=0.50,
        metadata={"help": "Minimum schema-relation coverage by question or residual-deficit terms."}
    )
    query_schema_memory_materialization_require_answer_type_value: bool = field(
        default=True,
        metadata={"help": "Require date or numeric values when materializing schema evidence for typed questions."}
    )
    enable_early_schema_bridge_materialization: bool = field(
        default=True,
        metadata={"help": "After retrieval detects a bridge deficit, directly expose high-confidence schema-memory fact-join evidence to the reader."}
    )
    early_schema_bridge_materialization_max_candidates: int = field(
        default=1,
        metadata={"help": "Maximum early schema-memory bridge candidates appended to the reader-visible pool."}
    )
    early_schema_bridge_materialization_max_original_rank: int = field(
        default=80,
        metadata={"help": "Deepest original rank allowed for early schema-memory bridge materialization."}
    )
    early_schema_bridge_materialization_min_score: float = field(
        default=0.66,
        metadata={"help": "Minimum schema-memory candidate score for early bridge materialization."}
    )
    early_schema_bridge_materialization_min_schema_idf: float = field(
        default=0.35,
        metadata={"help": "Minimum schema specificity for early bridge materialization."}
    )
    early_schema_bridge_materialization_min_query_overlap: float = field(
        default=0.08,
        metadata={"help": "Minimum question overlap for early schema-memory bridge materialization."}
    )
    early_schema_bridge_materialization_min_join_key_overlap: float = field(
        default=0.50,
        metadata={"help": "Minimum endpoint-token overlap for early schema-memory bridge materialization."}
    )
    early_schema_bridge_materialization_min_relation_overlap: float = field(
        default=0.20,
        metadata={"help": "Minimum relation-token overlap for early schema-memory bridge materialization."}
    )
    early_schema_bridge_materialization_require_answer_type_value: bool = field(
        default=True,
        metadata={"help": "Require date or numeric values for typed questions in early schema bridge materialization."}
    )
    early_schema_bridge_materialization_require_lineage_binding: bool = field(
        default=True,
        metadata={"help": "Require early schema bridge evidence to bind an uncovered answer slot."}
    )
    early_schema_bridge_materialization_allow_relation_relaxation: bool = field(
        default=True,
        metadata={"help": "Allow low relation-token overlap for early schema candidates when an exact join key and specific schema make the join reliable."}
    )
    early_schema_bridge_materialization_exact_join_key_min_schema_idf: float = field(
        default=0.65,
        metadata={"help": "Minimum schema specificity required before exact join-key candidates can relax the relation-overlap gate."}
    )
    early_schema_bridge_materialization_exact_join_key_min_relation_overlap: float = field(
        default=0.05,
        metadata={"help": "Minimum relation-token overlap retained under exact join-key relation relaxation."}
    )
    early_schema_bridge_materialization_compact_projection: bool = field(
        default=True,
        metadata={"help": "Materialize early schema bridge evidence as a compact title/schema/sentence projection instead of the full passage."}
    )
    early_schema_bridge_materialization_require_residual_deficit: bool = field(
        default=True,
        metadata={"help": "Require a residual evidence deficit after base-passage slot coverage is recomputed."}
    )
    early_schema_bridge_materialization_require_anchor_binding: bool = field(
        default=True,
        metadata={"help": "Require the joined endpoint to bind a question entity or an uncovered planner-slot anchor."}
    )
    early_schema_bridge_materialization_safe_max_original_rank: int = field(
        default=48,
        metadata={"help": "Rank boundary after which schema bridge candidates receive stricter score and schema-specificity gates."}
    )
    early_schema_bridge_materialization_deep_rank_min_score: float = field(
        default=0.70,
        metadata={"help": "Minimum schema score for candidates deeper than the safe early-materialization rank."}
    )
    enable_schema_critical_reader_visible: bool = field(
        default=False,
        metadata={
            "help": (
                "Run a strict fixed-budget Schema Memory bridge path after base "
                "retrieval and expose only a positive-cost-plan tuple to the reader."
            )
        },
    )
    schema_critical_reader_visible_apply_materialization: bool = field(
        default=True,
        metadata={
            "help": (
                "Commit selected Schema-Critical projections to the reader. "
                "When disabled, selection and cost planning still run and are "
                "recorded as a shadow materialization counterfactual."
            )
        },
    )
    schema_critical_reader_visible_max_append: int = field(
        default=1,
        metadata={"help": "Maximum Schema-Critical projections appended per query."},
    )
    schema_critical_reader_visible_deficit_coverage_threshold: float = field(
        default=0.72,
        metadata={"help": "Base top-k lexical coverage below which the strict bridge path may trigger."},
    )
    schema_critical_reader_visible_min_score: float = field(
        default=0.58,
        metadata={"help": "Minimum query-schema candidate score for the strict bridge path."},
    )
    schema_critical_reader_visible_min_schema_idf: float = field(
        default=0.50,
        metadata={"help": "Minimum schema specificity for a strict bridge candidate."},
    )
    schema_critical_reader_visible_min_join_key_overlap: float = field(
        default=0.75,
        metadata={"help": "Minimum endpoint overlap for a strict bridge candidate."},
    )
    schema_critical_reader_visible_min_relation_overlap: float = field(
        default=0.05,
        metadata={"help": "Minimum question-to-predicate overlap for a strict bridge candidate."},
    )
    schema_critical_reader_visible_max_original_rank: int = field(
        default=80,
        metadata={"help": "Maximum dense retrieval rank eligible for strict Schema-Critical execution."},
    )
    schema_critical_reader_visible_token_budget: int = field(
        default=220,
        metadata={"help": "Token budget for the compact reader-visible fact projection."},
    )
    schema_critical_reader_visible_compute_budget: float = field(
        default=1.0,
        metadata={"help": "Local operator compute budget for the strict Schema-Critical plan."},
    )
    schema_critical_reader_visible_min_objective: float = field(
        default=0.08,
        metadata={"help": "Minimum cost-adjusted plan objective required for reader admission."},
    )
    schema_critical_reader_visible_risk_weight: float = field(
        default=0.42,
        metadata={"help": "Risk weight for the strict Schema-Critical plan."},
    )
    schema_critical_reader_visible_token_weight: float = field(
        default=0.12,
        metadata={"help": "Token cost weight for the strict Schema-Critical plan."},
    )
    schema_critical_reader_visible_compute_weight: float = field(
        default=0.04,
        metadata={"help": "Compute cost weight for the strict Schema-Critical plan."},
    )
    schema_critical_reader_visible_require_bridge_cues: bool = field(
        default=False,
        metadata={"help": "Require lexical multi-hop bridge cues before strict Schema-Critical execution."},
    )
    schema_critical_reader_visible_require_instance_bound_lineage: bool = field(
        default=True,
        metadata={
            "help": (
                "Require a complete query-rooted multi-stage fact-join lineage "
                "before Schema-Critical evidence may enter the reader."
            )
        },
    )
    schema_critical_reader_visible_materialize_distributed_join_view: bool = field(
        default=False,
        metadata={
            "help": (
                "Materialize a compact provenance join view when a complete "
                "instance-bound lineage is distributed across reader passages."
            )
        },
    )
    schema_critical_reader_visible_require_unique_terminal_value: bool = field(
        default=True,
        metadata={
            "help": (
                "Require all eligible distributed lineages to agree on one "
                "terminal value before exposing their join view to the reader."
            )
        },
    )
    schema_critical_reader_visible_require_source_novelty: bool = field(
        default=False,
        metadata={
            "help": (
                "Require an admitted instance-bound lineage to add at least "
                "one provenance source not already present in the reader pool."
            )
        },
    )
    schema_critical_reader_visible_require_terminal_subtype_match: bool = field(
        default=False,
        metadata={
            "help": (
                "Require attribute-level terminal predicate agreement, such "
                "as death_place rather than the broad death family."
            )
        },
    )
    schema_critical_reader_visible_allow_general_terminal_fact_view: bool = field(
        default=False,
        metadata={
            "help": (
                "Allow validated terminal fact views beyond the legacy birth/death "
                "place subtypes. This explicit treatment switch supports a clean "
                "Schema Memory ablation."
            )
        },
    )
    enable_validated_schema_tuple_materialization: bool = field(
        default=True,
        metadata={"help": "Materialize verified Schema Memory fact tuples as compact reader-visible evidence under early bridge deficits."}
    )
    validated_schema_tuple_materialization_max_candidates: int = field(
        default=1,
        metadata={"help": "Maximum validated schema tuple projections exposed per query."}
    )
    validated_schema_tuple_materialization_min_score: float = field(
        default=0.58,
        metadata={"help": "Minimum schema score for validated tuple projection materialization."}
    )
    validated_schema_tuple_materialization_min_relation_overlap: float = field(
        default=0.05,
        metadata={"help": "Minimum relation overlap for validated tuple projection when predicate contracts do not already validate it."}
    )
    validated_schema_tuple_materialization_require_predicate_contract: bool = field(
        default=True,
        metadata={"help": "Require cross-schema tuple projections to satisfy a predicate or relation-signature contract."}
    )
    validated_schema_tuple_materialization_require_terminal_slot: bool = field(
        default=True,
        metadata={"help": "Require validated tuple projections to match the terminal answer slot instead of an intermediate bridge slot."}
    )
    validated_schema_tuple_materialization_require_source_sentence: bool = field(
        default=True,
        metadata={"help": "Require the projected provenance sentence to bind the tuple value and join key."}
    )
    validated_schema_tuple_materialization_min_token_savings: int = field(
        default=1,
        metadata={"help": "Minimum token savings required before exposing a validated tuple projection to the reader."}
    )
    validated_schema_tuple_materialization_block_reader_visible_source: bool = field(
        default=True,
        metadata={"help": "Do not expose a tuple projection when its source passage is already reader-visible."}
    )
    enable_active_schema_bridge_tuple_executor: bool = field(
        default=True,
        metadata={"help": "Allow strong Schema Memory fact tuples to execute as early bridge operators under residual deficits."}
    )
    active_schema_bridge_tuple_min_score: float = field(
        default=0.52,
        metadata={"help": "Minimum schema score for the active bridge tuple executor."}
    )
    active_schema_bridge_tuple_min_bound_signals: int = field(
        default=2,
        metadata={"help": "Minimum endpoint/slot/relation binding signals required by the active bridge tuple executor."}
    )
    enable_entity_predicate_fact_join: bool = field(
        default=False,
        metadata={"help": "Generate rank-stable cross-schema fact candidates by joining an active entity endpoint with the requested predicate."}
    )
    entity_predicate_fact_join_bridge_only: bool = field(
        default=True,
        metadata={"help": "Restrict entity-predicate fact joins to bridge or compositional questions."}
    )
    entity_predicate_fact_join_require_evidence_gap: bool = field(
        default=False,
        metadata={"help": "Require the retrieval-side lexical answerability-gap detector before generating join candidates."}
    )
    entity_predicate_fact_join_lineage_constrained: bool = field(
        default=True,
        metadata={"help": "Only probe entity endpoints reachable from a query-rooted active-fact lineage."}
    )
    entity_predicate_fact_join_max_lineage_hops: int = field(
        default=2,
        metadata={"help": "Maximum active-fact hops from a query-root entity to an entity-predicate join key."}
    )
    entity_predicate_fact_join_require_atomic_slot: bool = field(
        default=True,
        metadata={"help": "Require a compiled atomic deficit slot before an entity-predicate candidate may reach the reader."}
    )
    entity_predicate_fact_join_residual_deficit_trigger: bool = field(
        default=False,
        metadata={"help": "Allow an atomic unresolved verifier slot to bypass the bridge-cue gate after planned retrieval is exhausted."}
    )
    entity_predicate_fact_join_typed_residual_plan: bool = field(
        default=False,
        metadata={"help": "Compile verifier deficits into typed lookup, multi-key join, comparison, or unsafe-derived execution plans."}
    )
    entity_predicate_fact_join_max_exact_signature_fanout: int = field(
        default=32,
        metadata={"help": "Maximum exact relation-signature cardinality eligible for selective fact-store access beyond the dense rank cutoff."}
    )
    entity_predicate_fact_join_catalog_guided_executor: bool = field(
        default=False,
        metadata={"help": "Route unresolved typed slots through predicate-catalog-aware physical evidence operators."}
    )
    entity_predicate_fact_join_operator_faithful_executor: bool = field(
        default=False,
        metadata={"help": "Execute routed residual slots with operator-specific lookup, comparison, shared-key join, or dense fallback semantics."}
    )
    entity_predicate_fact_join_validate_operator_completion: bool = field(
        default=False,
        metadata={"help": "Count an operator as complete only after its residual-slot, type, provenance, and surface guards pass."}
    )
    entity_predicate_fact_join_validated_dense_top_k: int = field(
        default=1,
        metadata={"help": "Maximum sentence-grounded dense support candidates retained after operator completion validation."}
    )
    entity_predicate_fact_join_plan_stage_gate: bool = field(
        default=False,
        metadata={"help": "Require branch-bound final-answer slot completion before an operator candidate may be materialized for the reader."}
    )
    entity_predicate_fact_join_signature_compatibility_fallback: bool = field(
        default=False,
        metadata={"help": "After exact signature lookup, admit typed granularity-compatible signatures while retaining hard family conflicts."}
    )
    entity_predicate_fact_join_max_candidates_per_slot: int = field(
        default=2,
        metadata={"help": "Maximum catalog-guided fact candidates retained for one residual answer slot."}
    )
    entity_predicate_fact_join_query_candidate_budget: int = field(
        default=4,
        metadata={"help": "Maximum catalog-guided fact candidates retained across all residual slots in one query."}
    )
    entity_predicate_fact_join_generic_signature_max_fanout: int = field(
        default=4,
        metadata={"help": "Maximum endpoint fanout allowed for generic relation signatures such as located_in."}
    )
    entity_predicate_fact_join_max_active_facts: int = field(
        default=16,
        metadata={"help": "Maximum active retrieved facts whose concrete entity endpoints may probe the fact store."}
    )
    entity_predicate_fact_join_max_entity_fanout: int = field(
        default=240,
        metadata={"help": "Maximum fact-store rows scanned for one bound entity endpoint."}
    )
    entity_predicate_fact_join_candidate_top_k: int = field(
        default=48,
        metadata={"help": "Maximum rank-stable entity-predicate join candidates retained per query."}
    )
    entity_predicate_fact_join_max_original_rank: int = field(
        default=200,
        metadata={"help": "Deepest original passage rank eligible for entity-predicate fact joining."}
    )
    entity_predicate_fact_join_min_relation_overlap: float = field(
        default=0.34,
        metadata={"help": "Minimum requested-predicate overlap for an entity-bound fact-store row."}
    )
    entity_predicate_fact_join_min_score: float = field(
        default=0.58,
        metadata={"help": "Minimum retrieval-side utility score for an entity-predicate join candidate."}
    )
    enable_entity_predicate_fact_join_materialization: bool = field(
        default=False,
        metadata={"help": "Submit entity-predicate join candidates to reader-side candidate-level cost admission."}
    )
    entity_predicate_fact_join_materialization_max_candidates: int = field(
        default=1,
        metadata={"help": "Maximum entity-predicate support passages considered for reader materialization."}
    )
    entity_predicate_fact_join_materialization_min_score: float = field(
        default=0.64,
        metadata={"help": "Minimum join score for reader-side entity-predicate materialization."}
    )
    entity_predicate_fact_join_materialization_min_relation_overlap: float = field(
        default=0.50,
        metadata={"help": "Minimum predicate overlap for reader-side entity-predicate materialization."}
    )
    entity_predicate_fact_join_materialization_min_lineage_confidence: float = field(
        default=0.34,
        metadata={"help": "Minimum residual-slot lineage confidence for reader-side entity-predicate materialization."}
    )
    entity_predicate_fact_join_materialization_allowed_match_modes: str = field(
        default="direct_token,polarity_concept",
        metadata={"help": "Comma-separated predicate match modes eligible for reader materialization; other modes remain diagnostic-only."}
    )
    entity_predicate_fact_join_materialization_compact_projection: bool = field(
        default=False,
        metadata={"help": "Materialize an entity-predicate join as a compact provenance fact projection instead of the full source passage."}
    )
    entity_predicate_fact_join_terminal_tuple_materialization: bool = field(
        default=False,
        metadata={"help": "Materialize only terminal answer tuples with exact schema, rooted lineage, and branch identity."}
    )
    entity_predicate_fact_join_terminal_tuple_min_lineage_depth: int = field(
        default=1,
        metadata={"help": "Minimum rooted fact-join depth required for terminal answer-tuple materialization."}
    )
    entity_predicate_fact_join_terminal_tuple_strict_branch_binding: bool = field(
        default=True,
        metadata={"help": "Require terminal fact rows to bind the residual slot's named entity branch."}
    )
    enable_entity_predicate_final_reader_counterfactual: bool = field(
        default=False,
        metadata={"help": "Run a paired final-reader prompt with admitted entity-predicate passages removed for touched-query causal analysis."}
    )
    entity_predicate_final_reader_counterfactual_max_queries: int = field(
        default=64,
        metadata={"help": "Maximum touched queries per QA batch receiving final-reader counterfactual replay."}
    )
    enable_relation_fact_join_final_reader_counterfactual: bool = field(
        default=False,
        metadata={"help": "Replay one frozen Relation Fact Join tuple as a paired reader intervention on the same retrieval trace."}
    )
    relation_fact_join_final_reader_counterfactual_max_queries: int = field(
        default=64,
        metadata={"help": "Maximum Relation Fact Join reader-intervention pairs per QA batch."}
    )
    query_schema_memory_intervention_policy: str = field(
        default="risk_aware",
        metadata={"help": "How query-schema-memory candidates may affect rank intervention: off, risk_aware, or append_only."}
    )
    query_schema_memory_top2_min_score: float = field(
        default=0.66,
        metadata={"help": "Minimum query-schema-memory score before a schema-only candidate can change rank 2."}
    )
    query_schema_memory_top2_min_schema_idf: float = field(
        default=0.32,
        metadata={"help": "Minimum schema specificity before a schema-only candidate can change rank 2."}
    )
    query_schema_memory_top2_min_overlap: float = field(
        default=0.10,
        metadata={"help": "Minimum query/context overlap before a schema-only candidate can change rank 2."}
    )
    query_schema_memory_top2_max_schema_frequency: int = field(
        default=80,
        metadata={"help": "Maximum schema frequency allowed before a schema-only candidate is downgraded to append-only."}
    )
    query_schema_memory_top2_max_original_rank: int = field(
        default=24,
        metadata={"help": "Largest original retrieval rank from which a schema-only candidate can change rank 2."}
    )
    query_schema_memory_top2_min_bridge_strength: int = field(
        default=3,
        metadata={"help": "Minimum query bridge-strength before a schema-only candidate can change rank 2."}
    )
    enable_schema_deficit_promotion: bool = field(
        default=False,
        metadata={"help": "Allow schema-memory candidates to recover top-2 eligibility when they fill a current evidence deficit."}
    )
    schema_deficit_min_score: float = field(
        default=0.62,
        metadata={"help": "Minimum schema-deficit promotion score."}
    )
    schema_deficit_min_gain: float = field(
        default=0.10,
        metadata={"help": "Minimum measured evidence-deficit gain for schema-deficit promotion unless the old pair is clearly deficient."}
    )
    schema_deficit_min_schema_score: float = field(
        default=0.68,
        metadata={"help": "Minimum schema-memory retrieval score for schema-deficit promotion."}
    )
    schema_deficit_min_schema_idf: float = field(
        default=0.30,
        metadata={"help": "Minimum schema specificity for schema-deficit promotion."}
    )
    schema_deficit_max_original_rank: int = field(
        default=60,
        metadata={"help": "Largest original retrieval rank eligible for schema-deficit promotion."}
    )
    schema_deficit_require_bridge_constraint: bool = field(
        default=True,
        metadata={"help": "Require schema-deficit promotion candidates to pass bridge-slot structural evidence before top-2 intervention."}
    )
    schema_deficit_min_bridge_score: float = field(
        default=0.54,
        metadata={"help": "Minimum bridge-slot structural score for schema-deficit promotion."}
    )
    schema_deficit_min_bridge_margin: float = field(
        default=0.08,
        metadata={"help": "Minimum bridge-slot improvement over the current rank-2 passage for schema-deficit promotion."}
    )
    schema_deficit_min_bridge_missing_gain: float = field(
        default=0.10,
        metadata={"help": "Minimum missing-slot support gain for schema-deficit promotion."}
    )
    schema_deficit_min_bridge_anchor_overlap: float = field(
        default=0.24,
        metadata={"help": "Minimum bridge anchor overlap for schema-deficit promotion unless entity-conditioned evidence is strong."}
    )
    schema_deficit_soft_append_enabled: bool = field(
        default=True,
        metadata={"help": "Append strong schema-deficit candidates as QA evidence when they are too risky to edit top-2 ranking."}
    )
    schema_deficit_soft_append_min_score: float = field(
        default=0.48,
        metadata={"help": "Minimum schema-deficit score for append-only schema evidence."}
    )
    schema_deficit_soft_append_min_schema_score: float = field(
        default=0.58,
        metadata={"help": "Minimum schema-memory score for append-only schema evidence."}
    )
    schema_deficit_soft_append_min_schema_idf: float = field(
        default=0.24,
        metadata={"help": "Minimum schema specificity for append-only schema evidence."}
    )
    schema_deficit_soft_append_min_query_support: float = field(
        default=0.34,
        metadata={"help": "Minimum candidate query-support signal for append-only schema evidence."}
    )
    schema_deficit_soft_append_max_original_rank: int = field(
        default=80,
        metadata={"help": "Largest original retrieval rank eligible for append-only schema evidence."}
    )
    enable_schema_reliability_gate: bool = field(
        default=False,
        metadata={"help": "Calibrate query-schema-memory edges before append-only schema evidence is shown to the reader."}
    )
    schema_reliability_min_score: float = field(
        default=0.62,
        metadata={"help": "Minimum reliability score for passing an already accepted schema append candidate."}
    )
    schema_reliability_rescue_min_score: float = field(
        default=0.76,
        metadata={"help": "Minimum reliability score required to rescue a rejected schema append candidate."}
    )
    schema_reliability_min_anchor_overlap: float = field(
        default=0.08,
        metadata={"help": "Minimum query/context anchor overlap unless relation or query support is strong."}
    )
    schema_reliability_max_schema_frequency: int = field(
        default=80,
        metadata={"help": "Maximum schema frequency accepted by the schema reliability gate; 0 disables the cap."}
    )
    schema_reliability_numeric_requires_value: bool = field(
        default=True,
        metadata={"help": "Require date/number value evidence for schema reliability on numeric or temporal questions."}
    )



    max_qa_steps: int = field(
        default=1,
        metadata={"help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."}
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "Feeding top k documents to the QA model for reading."}
    )
    qa_inference_workers: int = field(
        default=32,
        metadata={"help": "Number of worker threads used for QA LLM inference. Use >1 for API models."}
    )
    qa_inference_recovery_rounds: int = field(
        default=2,
        metadata={"help": "Query-level recovery rounds for transient QA inference failures."}
    )
    qa_inference_recovery_workers: int = field(
        default=1,
        metadata={"help": "Worker threads used only for recovering failed QA queries."}
    )
    qa_inference_recovery_delay_seconds: float = field(
        default=10.0,
        metadata={"help": "Delay before each QA inference recovery round."}
    )
    qa_fail_on_unrecovered: bool = field(
        default=True,
        metadata={"help": "Fail the dataset instead of scoring fallback answers when QA queries remain unrecovered."}
    )
    qa_cache_coverage_preflight: bool = field(
        default=False,
        metadata={
            "help": (
                "Check every final Reader prompt against the configured SQLite "
                "cache before starting Reader inference; fail atomically on a miss."
            )
        },
    )
    qa_cache_coverage_report_path: str = field(
        default="",
        metadata={
            "help": "Optional JSON path for the Reader cache-coverage preflight report."
        },
    )
    enable_adaptive_evidence: bool = field(
        default=False,
        metadata={"help": "Whether to adaptively add bridge evidence from the wider retrieval pool before QA."}
    )
    adaptive_evidence_candidate_k: int = field(
        default=20,
        metadata={"help": "Number of retrieved documents considered by adaptive evidence selection."}
    )
    adaptive_evidence_target_k: int = field(
        default=8,
        metadata={"help": "Final number of documents fed to QA when adaptive evidence expansion is triggered."}
    )
    adaptive_evidence_bridge_only: bool = field(
        default=True,
        metadata={"help": "Only expand evidence for questions with bridge or composition cues."}
    )
    enable_schema_evidence_set_selection: bool = field(
        default=False,
        metadata={"help": "Select a fixed Reader top-k from a frozen dense pool using query obligations and fact-index connectivity."}
    )
    schema_evidence_set_mode: Literal["dense", "schema_score", "greedy_coverage", "joint", "binding_path", "index_probe_path", "index_probe_path_safe", "index_probe_path_surface_safe"] = field(
        default="joint",
        metadata={"help": "Schema-aware fixed-budget evidence-set selection policy."}
    )
    schema_evidence_set_candidate_k: int = field(
        default=20,
        metadata={"help": "Frozen dense prefix considered by schema-aware evidence-set selection."}
    )
    schema_evidence_set_target_k: int = field(
        default=5,
        metadata={"help": "Fixed number of passages emitted by schema-aware evidence-set selection."}
    )
    schema_evidence_set_protect_top_k: int = field(
        default=1,
        metadata={"help": "Dense prefix protected by schema-aware evidence-set selection."}
    )
    schema_evidence_set_index_max_rows_per_key: int = field(
        default=64,
        metadata={"help": "Maximum physical index rows scanned for one entity-predicate key."}
    )
    schema_evidence_set_index_max_total_rows: int = field(
        default=512,
        metadata={"help": "Maximum physical index rows scanned by one query evidence selection."}
    )
    schema_evidence_set_index_max_fanout: int = field(
        default=512,
        metadata={"help": "Reject entity-predicate keys whose raw fanout exceeds this bound."}
    )
    enable_bridge_contrastive_evidence_guard: bool = field(
        default=False,
        metadata={"help": "Whether to reorder QA evidence so answer-type-only distractors without bridge anchors appear after bridge-connected evidence."}
    )
    bridge_contrastive_guard_protect_top_k: int = field(
        default=1,
        metadata={"help": "Number of leading QA evidence documents protected from bridge-contrastive reordering."}
    )
    bridge_contrastive_guard_min_bridge_link: float = field(
        default=0.34,
        metadata={"help": "Minimum bridge-link score for a document to avoid answer-type-only demotion."}
    )
    bridge_contrastive_guard_min_title_anchor_overlap: float = field(
        default=0.50,
        metadata={"help": "Minimum title-to-anchor overlap for treating an evidence document as bridge-connected."}
    )
    enable_bridge_contrastive_counterfactual_safety: bool = field(
        default=True,
        metadata={"help": "Whether bridge-contrastive reordering must pass a counterfactual evidence-state safety check before changing QA evidence order."}
    )
    bridge_contrastive_counterfactual_safety_top_k: int = field(
        default=8,
        metadata={"help": "Top-k evidence window compared by bridge-contrastive counterfactual safety."}
    )
    bridge_contrastive_counterfactual_safety_min_score_drop: float = field(
        default=0.35,
        metadata={"help": "Maximum allowed evidence-state score drop before bridge-contrastive reordering is vetoed."}
    )
    enable_answer_canonicalization: bool = field(
        default=False,
        metadata={"help": "Whether to canonicalize high-confidence answer-span variants after QA."}
    )
    enable_answer_style_routing: bool = field(
        default=True,
        metadata={"help": "Whether to inject question-dependent answer-style instructions before QA."}
    )
    enable_answer_postprocessing: bool = field(
        default=True,
        metadata={"help": "Whether to run lightweight extracted-answer postprocessing before downstream QA arbitration."}
    )
    answer_canonicalization_min_confidence: float = field(
        default=0.65,
        metadata={"help": "Minimum confidence required to apply evidence-grounded answer canonicalization."}
    )
    enable_evidence_answer_verifier: bool = field(
        default=False,
        metadata={"help": "Whether to verify and rewrite unsupported answers with evidence-backed spans after QA."}
    )
    evidence_answer_verifier_min_confidence: float = field(
        default=0.72,
        metadata={"help": "Minimum confidence required for numeric/date answer verifier rewrites."}
    )
    evidence_answer_verifier_min_margin: float = field(
        default=0.12,
        metadata={"help": "Minimum confidence margin over the second candidate for answer verifier rewrites."}
    )
    evidence_answer_verifier_generic_min_confidence: float = field(
        default=0.84,
        metadata={"help": "Minimum confidence for non-numeric unsupported answer verifier rewrites."}
    )
    evidence_answer_verifier_max_docs: int = field(
        default=8,
        metadata={"help": "Maximum QA evidence documents scanned by the evidence answer verifier."}
    )
    enable_evidence_consistent_reader: bool = field(
        default=False,
        metadata={"help": "Whether to run the evidence-consistent answer reader after QA."}
    )
    evidence_consistent_reader_min_confidence: float = field(
        default=0.84,
        metadata={"help": "Minimum confidence for numeric/date evidence-consistent reader rewrites."}
    )
    evidence_consistent_reader_min_margin: float = field(
        default=0.10,
        metadata={"help": "Minimum margin over the second evidence-consistent reader candidate."}
    )
    evidence_consistent_reader_generic_min_confidence: float = field(
        default=0.92,
        metadata={"help": "Minimum confidence for non-numeric evidence-consistent reader rewrites."}
    )
    evidence_consistent_reader_min_support_gain: float = field(
        default=0.18,
        metadata={"help": "Minimum support-score gain over the current answer before rewriting."}
    )
    evidence_consistent_reader_max_docs: int = field(
        default=8,
        metadata={"help": "Maximum QA evidence documents scanned by the evidence-consistent reader."}
    )
    evidence_consistent_reader_max_answer_tokens: int = field(
        default=10,
        metadata={"help": "Maximum current answer token count for short-answer reader rewrites."}
    )
    enable_evidence_object_arbitration: bool = field(
        default=False,
        metadata={"help": "Whether to arbitrate final answers against evidence-supported answer objects."}
    )
    evidence_object_arbitration_min_confidence: float = field(
        default=0.82,
        metadata={"help": "Minimum confidence for numeric/date evidence-object arbitration rewrites."}
    )
    evidence_object_arbitration_generic_min_confidence: float = field(
        default=0.88,
        metadata={"help": "Minimum confidence for generic evidence-object arbitration rewrites."}
    )
    evidence_object_arbitration_min_margin: float = field(
        default=0.12,
        metadata={"help": "Minimum margin over the second evidence-object arbitration candidate."}
    )
    evidence_object_arbitration_min_support_gain: float = field(
        default=0.18,
        metadata={"help": "Minimum support-score gain over the current answer before object arbitration rewrites."}
    )
    evidence_object_arbitration_min_question_overlap: float = field(
        default=0.08,
        metadata={"help": "Minimum question overlap for evidence-object arbitration candidates."}
    )
    evidence_object_arbitration_max_docs: int = field(
        default=10,
        metadata={"help": "Maximum QA evidence documents scanned by evidence-object arbitration."}
    )
    evidence_object_arbitration_max_answer_tokens: int = field(
        default=12,
        metadata={"help": "Maximum current answer token count before treating it as concise."}
    )
    evidence_object_arbitration_apply_rewrite: bool = field(
        default=False,
        metadata={"help": "If false, object arbitration runs in shadow mode and never changes final answers."}
    )
    evidence_object_arbitration_require_role_constraint: bool = field(
        default=True,
        metadata={"help": "Require subject/relation role cues before evidence-object arbitration may rewrite."}
    )
    enable_evidence_answer_arbiter: bool = field(
        default=False,
        metadata={"help": "Whether to run evidence-grounded final answer arbitration."}
    )
    evidence_answer_arbiter_apply_rewrite: bool = field(
        default=False,
        metadata={"help": "If false, evidence-answer arbiter runs in shadow mode and never changes final answers."}
    )
    evidence_answer_arbiter_min_confidence: float = field(
        default=0.84,
        metadata={"help": "Minimum confidence for numeric/date evidence-answer arbitration."}
    )
    evidence_answer_arbiter_generic_min_confidence: float = field(
        default=0.92,
        metadata={"help": "Minimum confidence for generic evidence-answer arbitration."}
    )
    evidence_answer_arbiter_min_margin: float = field(
        default=0.14,
        metadata={"help": "Minimum margin over the second evidence-answer candidate."}
    )
    evidence_answer_arbiter_min_support_gain: float = field(
        default=0.18,
        metadata={"help": "Minimum support-score gain over the current answer before final answer arbitration."}
    )
    evidence_answer_arbiter_alias_min_support_gain: float = field(
        default=0.04,
        metadata={"help": "Minimum support-score gain for shorter containment/alias rewrites."}
    )
    evidence_answer_arbiter_min_question_overlap: float = field(
        default=0.08,
        metadata={"help": "Minimum question overlap for evidence-answer candidates."}
    )
    evidence_answer_arbiter_max_docs: int = field(
        default=10,
        metadata={"help": "Maximum QA evidence documents scanned by evidence-answer arbitration."}
    )
    evidence_answer_arbiter_max_answer_tokens: int = field(
        default=12,
        metadata={"help": "Maximum current answer token count before treating it as verbose."}
    )
    evidence_answer_arbiter_require_role_constraint: bool = field(
        default=True,
        metadata={"help": "Require subject/relation role cues before evidence-answer arbitration may rewrite."}
    )
    evidence_answer_arbiter_skip_supported_answers: bool = field(
        default=True,
        metadata={"help": "Protect already supported concise answers from generic answer arbitration rewrites."}
    )
    enable_typed_evidence_table_arbiter: bool = field(
        default=False,
        metadata={"help": "Whether to run typed evidence-table answer arbitration after QA."}
    )
    typed_evidence_table_apply_rewrite: bool = field(
        default=False,
        metadata={"help": "If false, typed evidence-table arbitration runs in shadow mode."}
    )
    typed_evidence_table_min_confidence: float = field(
        default=0.86,
        metadata={"help": "Minimum confidence for numeric/date typed evidence-table arbitration."}
    )
    typed_evidence_table_generic_min_confidence: float = field(
        default=0.92,
        metadata={"help": "Minimum confidence for generic typed evidence-table arbitration."}
    )
    typed_evidence_table_min_margin: float = field(
        default=0.12,
        metadata={"help": "Minimum typed table score margin over the second candidate."}
    )
    typed_evidence_table_min_support_gain: float = field(
        default=0.12,
        metadata={"help": "Minimum evidence support gain before typed evidence-table rewriting."}
    )
    typed_evidence_table_alias_min_support_gain: float = field(
        default=0.02,
        metadata={"help": "Minimum support gain for shorter alias/granularity rewrites."}
    )
    typed_evidence_table_min_question_overlap: float = field(
        default=0.08,
        metadata={"help": "Minimum question overlap for typed evidence-table rows."}
    )
    typed_evidence_table_max_docs: int = field(
        default=10,
        metadata={"help": "Maximum QA evidence documents scanned by typed evidence-table arbitration."}
    )
    typed_evidence_table_max_answer_tokens: int = field(
        default=12,
        metadata={"help": "Maximum current answer token count before treating it as verbose."}
    )
    typed_evidence_table_min_provenance: int = field(
        default=1,
        metadata={"help": "Minimum supporting provenance count for typed evidence-table candidates."}
    )
    typed_evidence_table_require_role_constraint: bool = field(
        default=True,
        metadata={"help": "Require relation/role cues before typed evidence-table arbitration may rewrite."}
    )
    typed_evidence_table_skip_supported_answers: bool = field(
        default=True,
        metadata={"help": "Protect already supported concise answers from typed evidence-table rewrites."}
    )
    enable_provenance_answer_lattice: bool = field(
        default=False,
        metadata={"help": "Whether to run provenance-grounded answer lattice arbitration after QA."}
    )
    provenance_answer_lattice_apply_rewrite: bool = field(
        default=False,
        metadata={"help": "If false, provenance answer lattice runs in shadow mode."}
    )
    provenance_answer_lattice_min_score: float = field(
        default=0.88,
        metadata={"help": "Minimum score for numeric/date provenance answer lattice rewrites."}
    )
    provenance_answer_lattice_generic_min_score: float = field(
        default=0.93,
        metadata={"help": "Minimum score for generic provenance answer lattice rewrites."}
    )
    provenance_answer_lattice_min_margin: float = field(
        default=0.12,
        metadata={"help": "Minimum margin over the second lattice candidate."}
    )
    provenance_answer_lattice_min_support_gain: float = field(
        default=0.14,
        metadata={"help": "Minimum support gain over the current answer before lattice rewriting."}
    )
    provenance_answer_lattice_alias_min_support_gain: float = field(
        default=0.02,
        metadata={"help": "Minimum support gain for shorter alias/granularity lattice rewrites."}
    )
    provenance_answer_lattice_min_sources: int = field(
        default=2,
        metadata={"help": "Minimum number of independent candidate sources required by the answer lattice."}
    )
    provenance_answer_lattice_min_provenance: int = field(
        default=2,
        metadata={"help": "Minimum total provenance count required by the answer lattice."}
    )
    provenance_answer_lattice_min_question_overlap: float = field(
        default=0.08,
        metadata={"help": "Minimum question overlap for answer lattice evidence candidates."}
    )
    provenance_answer_lattice_max_docs: int = field(
        default=10,
        metadata={"help": "Maximum QA evidence documents scanned by provenance answer lattice."}
    )
    provenance_answer_lattice_max_answer_tokens: int = field(
        default=12,
        metadata={"help": "Maximum current answer token count before treating it as verbose."}
    )
    provenance_answer_lattice_require_role_constraint: bool = field(
        default=True,
        metadata={"help": "Require relation/role cues before provenance answer lattice may rewrite."}
    )
    provenance_answer_lattice_skip_supported_answers: bool = field(
        default=True,
        metadata={"help": "Protect already supported concise answers from lattice rewrites."}
    )
    enable_provenance_typed_candidate_store: bool = field(
        default=False,
        metadata={"help": "Whether to run a cost-based provenance-typed answer candidate store after QA."}
    )
    provenance_typed_candidate_store_apply_rewrite: bool = field(
        default=False,
        metadata={"help": "If false, the provenance-typed candidate store runs in shadow mode."}
    )
    provenance_typed_candidate_store_min_score: float = field(
        default=0.92,
        metadata={"help": "Minimum support score for numeric/date provenance-typed answer rewrites."}
    )
    provenance_typed_candidate_store_generic_min_score: float = field(
        default=0.96,
        metadata={"help": "Minimum support score for generic provenance-typed answer rewrites."}
    )
    provenance_typed_candidate_store_min_net_utility: float = field(
        default=0.58,
        metadata={"help": "Minimum utility-minus-risk objective for candidate-store rewrites."}
    )
    provenance_typed_candidate_store_min_margin: float = field(
        default=0.16,
        metadata={"help": "Minimum net-utility margin over the second answer candidate."}
    )
    provenance_typed_candidate_store_min_support_gain: float = field(
        default=0.20,
        metadata={"help": "Minimum support gain over the current answer before candidate-store rewriting."}
    )
    provenance_typed_candidate_store_alias_min_support_gain: float = field(
        default=0.03,
        metadata={"help": "Minimum support gain for shorter alias/granularity candidate-store rewrites."}
    )
    provenance_typed_candidate_store_min_question_overlap: float = field(
        default=0.08,
        metadata={"help": "Minimum question overlap for candidate-store rows."}
    )
    provenance_typed_candidate_store_max_docs: int = field(
        default=10,
        metadata={"help": "Maximum QA evidence documents scanned by the answer candidate store."}
    )
    provenance_typed_candidate_store_max_answer_tokens: int = field(
        default=12,
        metadata={"help": "Maximum current answer token count before treating it as verbose."}
    )
    provenance_typed_candidate_store_max_candidates: int = field(
        default=16,
        metadata={"help": "Maximum materialized answer candidates scored by the cost model."}
    )
    provenance_typed_candidate_store_min_sources: int = field(
        default=2,
        metadata={"help": "Minimum independent source count for answer candidate-store rewrites."}
    )
    provenance_typed_candidate_store_min_provenance: int = field(
        default=2,
        metadata={"help": "Minimum provenance count for answer candidate-store rewrites."}
    )
    provenance_typed_candidate_store_high_confidence_single_source: float = field(
        default=1.01,
        metadata={"help": "Single-source candidate support score that bypasses multi-source requirements."}
    )
    provenance_typed_candidate_store_token_weight: float = field(
        default=0.08,
        metadata={"help": "Weight applied to candidate answer length cost."}
    )
    provenance_typed_candidate_store_provenance_weight: float = field(
        default=0.02,
        metadata={"help": "Utility bonus per additional candidate provenance item."}
    )
    provenance_typed_candidate_store_risk_weight: float = field(
        default=0.78,
        metadata={"help": "Weight applied to candidate-store rewrite risk."}
    )
    provenance_typed_candidate_store_require_role_constraint: bool = field(
        default=True,
        metadata={"help": "Require relation/role cues before candidate-store rewrites."}
    )
    provenance_typed_candidate_store_skip_supported_answers: bool = field(
        default=True,
        metadata={"help": "Protect already supported concise answers from candidate-store rewrites."}
    )
    enable_evidence_grounded_answer_extractor: bool = field(
        default=False,
        metadata={"help": "Whether to run evidence-grounded answer extraction as the final QA projection operator."}
    )
    evidence_grounded_answer_extractor_apply_rewrite: bool = field(
        default=False,
        metadata={"help": "If false, the evidence-grounded extractor only records shadow diagnostics."}
    )
    evidence_grounded_answer_extractor_min_score: float = field(
        default=0.84,
        metadata={"help": "Minimum support score for numeric/date evidence-grounded rewrites."}
    )
    evidence_grounded_answer_extractor_generic_min_score: float = field(
        default=0.84,
        metadata={"help": "Minimum support score for generic evidence-grounded rewrites."}
    )
    evidence_grounded_answer_extractor_min_margin: float = field(
        default=0.08,
        metadata={"help": "Minimum score margin over the second evidence-grounded candidate."}
    )
    evidence_grounded_answer_extractor_min_support_gain: float = field(
        default=0.06,
        metadata={"help": "Minimum support gain over the current answer before evidence-grounded rewriting."}
    )
    evidence_grounded_answer_extractor_alias_min_support_gain: float = field(
        default=0.01,
        metadata={"help": "Minimum support gain for evidence-grounded alias/granularity rewrites."}
    )
    evidence_grounded_answer_extractor_min_provenance: int = field(
        default=1,
        metadata={"help": "Minimum provenance count for evidence-grounded answer candidates."}
    )
    evidence_grounded_answer_extractor_min_question_overlap: float = field(
        default=0.02,
        metadata={"help": "Minimum question overlap for evidence-grounded answer candidates."}
    )
    evidence_grounded_answer_extractor_max_docs: int = field(
        default=8,
        metadata={"help": "Maximum QA evidence documents scanned by evidence-grounded extraction."}
    )
    evidence_grounded_answer_extractor_max_answer_tokens: int = field(
        default=14,
        metadata={"help": "Maximum current answer token count before treating it as verbose."}
    )
    evidence_grounded_answer_extractor_max_candidates: int = field(
        default=12,
        metadata={"help": "Maximum evidence-grounded answer candidates scored."}
    )
    evidence_grounded_answer_extractor_require_role_constraint: bool = field(
        default=True,
        metadata={"help": "Require relation/role cues before evidence-grounded rewrites."}
    )
    evidence_grounded_answer_extractor_skip_supported_answers: bool = field(
        default=True,
        metadata={"help": "Protect already supported concise answers from evidence-grounded rewrites."}
    )
    enable_query_answer_contract: bool = field(
        default=False,
        metadata={"help": "Whether to run a query-compiled answer contract as the final answer projection step."}
    )
    query_answer_contract_apply_rewrite: bool = field(
        default=False,
        metadata={"help": "If false, the query-answer contract records shadow diagnostics only."}
    )
    query_answer_contract_min_score: float = field(
        default=0.78,
        metadata={"help": "Minimum contract score for temporal/numeric answer-contract rewrites."}
    )
    query_answer_contract_generic_min_score: float = field(
        default=0.82,
        metadata={"help": "Minimum contract score for non-temporal answer-contract rewrites."}
    )
    query_answer_contract_min_margin: float = field(
        default=0.05,
        metadata={"help": "Minimum margin over the second answer-contract candidate."}
    )
    query_answer_contract_min_support_gain: float = field(
        default=0.03,
        metadata={"help": "Minimum support gain over the current answer before contract rewriting."}
    )
    query_answer_contract_min_relation_score: float = field(
        default=0.50,
        metadata={"help": "Minimum relation-match score for relation-constrained contract rewrites."}
    )
    query_answer_contract_max_docs: int = field(
        default=8,
        metadata={"help": "Maximum QA evidence documents scanned by the query-answer contract."}
    )
    query_answer_contract_max_answer_tokens: int = field(
        default=14,
        metadata={"help": "Maximum current answer length before treating it as a verbose answer."}
    )
    query_answer_contract_max_candidates: int = field(
        default=12,
        metadata={"help": "Maximum answer-contract candidates retained for diagnostics."}
    )
    query_answer_contract_min_question_overlap: float = field(
        default=0.02,
        metadata={"help": "Minimum question overlap for reusable candidate extractors inside the answer contract."}
    )
    query_answer_contract_require_role_constraint: bool = field(
        default=True,
        metadata={"help": "Require relation/role constraints for generic answer-contract candidate extraction."}
    )
    query_answer_contract_protect_supported_answers: bool = field(
        default=True,
        metadata={"help": "Protect already supported concise answers unless they violate an explicit answer contract."}
    )
    enable_provenance_answer_slot_store: bool = field(
        default=False,
        metadata={"help": "Materialize a provenance-constrained answer-slot store before final answer binding."}
    )
    provenance_answer_slot_store_apply_rewrite: bool = field(
        default=True,
        metadata={"help": "If false, the provenance answer-slot store records shadow diagnostics only."}
    )
    provenance_answer_slot_store_min_score: float = field(
        default=0.80,
        metadata={"help": "Minimum support score for temporal/numeric answer-slot store rewrites."}
    )
    provenance_answer_slot_store_generic_min_score: float = field(
        default=0.86,
        metadata={"help": "Minimum support score for generic answer-slot store rewrites."}
    )
    provenance_answer_slot_store_min_slot_store_score: float = field(
        default=0.44,
        metadata={"help": "Minimum combined store score for provenance answer-slot store rewrites."}
    )
    provenance_answer_slot_store_min_margin: float = field(
        default=0.04,
        metadata={"help": "Minimum score margin over the second answer-slot store candidate."}
    )
    provenance_answer_slot_store_min_support_gain: float = field(
        default=0.05,
        metadata={"help": "Minimum support gain over the current answer before answer-slot store rewriting."}
    )
    provenance_answer_slot_store_alias_min_support_gain: float = field(
        default=0.01,
        metadata={"help": "Minimum support gain for safe alias/granularity answer-slot store rewrites."}
    )
    provenance_answer_slot_store_min_question_overlap: float = field(
        default=0.02,
        metadata={"help": "Minimum question overlap for base reader candidates inside the answer-slot store."}
    )
    provenance_answer_slot_store_max_docs: int = field(
        default=10,
        metadata={"help": "Maximum QA evidence documents scanned by the provenance answer-slot store."}
    )
    provenance_answer_slot_store_max_answer_tokens: int = field(
        default=14,
        metadata={"help": "Maximum current answer token count before treating it as verbose in the answer-slot store."}
    )
    provenance_answer_slot_store_max_candidates: int = field(
        default=18,
        metadata={"help": "Maximum materialized answer-slot store candidates retained for scoring."}
    )
    provenance_answer_slot_store_min_base_provenance: int = field(
        default=1,
        metadata={"help": "Minimum base-evidence provenance count required by the answer-slot store."}
    )
    provenance_answer_slot_store_min_source_diversity: int = field(
        default=1,
        metadata={"help": "Minimum base-evidence source diversity required by the answer-slot store."}
    )
    provenance_answer_slot_store_require_role_constraint: bool = field(
        default=True,
        metadata={"help": "Require relation/role cues before answer-slot store rewrites."}
    )
    provenance_answer_slot_store_protect_supported_answers: bool = field(
        default=True,
        metadata={"help": "Protect supported, slot-compatible reader answers unless the answer-slot store sees a safe improvement."}
    )
    provenance_answer_slot_store_min_relation_entailment_score: float = field(
        default=0.20,
        metadata={"help": "Minimum relation score for source-diverse answer-slot store rewrites."}
    )
    provenance_answer_slot_store_min_relation_question_overlap: float = field(
        default=0.10,
        metadata={"help": "Minimum question overlap for source-diverse answer-slot store rewrites."}
    )
    enable_answer_slot_binding_optimizer: bool = field(
        default=False,
        metadata={"help": "Whether to run query-conditioned answer-slot binding over materialized evidence candidates."}
    )
    answer_slot_binding_optimizer_apply_rewrite: bool = field(
        default=True,
        metadata={"help": "If false, answer-slot binding records shadow diagnostics only."}
    )
    answer_slot_binding_optimizer_min_score: float = field(
        default=0.78,
        metadata={"help": "Minimum support score for numeric/date answer-slot binding rewrites."}
    )
    answer_slot_binding_optimizer_generic_min_score: float = field(
        default=0.84,
        metadata={"help": "Minimum support score for non-temporal answer-slot binding rewrites."}
    )
    answer_slot_binding_optimizer_min_binding_score: float = field(
        default=0.42,
        metadata={"help": "Minimum combined binding score for answer-slot optimizer rewrites."}
    )
    answer_slot_binding_optimizer_min_margin: float = field(
        default=0.04,
        metadata={"help": "Minimum binding-score margin over the second candidate."}
    )
    answer_slot_binding_optimizer_min_support_gain: float = field(
        default=0.04,
        metadata={"help": "Minimum support gain over the current answer before answer-slot binding."}
    )
    answer_slot_binding_optimizer_min_slot_mismatch_gain: float = field(
        default=0.08,
        metadata={"help": "Minimum support gain when repairing an answer that violates the query slot."}
    )
    answer_slot_binding_optimizer_alias_min_support_gain: float = field(
        default=0.01,
        metadata={"help": "Minimum support gain for high-confidence alias/granularity binding repairs."}
    )
    answer_slot_binding_optimizer_max_docs: int = field(
        default=10,
        metadata={"help": "Maximum QA evidence documents scanned by answer-slot binding."}
    )
    answer_slot_binding_optimizer_max_answer_tokens: int = field(
        default=14,
        metadata={"help": "Maximum candidate answer token count before rejecting generic binding rewrites."}
    )
    answer_slot_binding_optimizer_max_candidates: int = field(
        default=18,
        metadata={"help": "Maximum materialized answer candidates kept by the answer-slot optimizer."}
    )
    answer_slot_binding_optimizer_min_base_provenance: int = field(
        default=1,
        metadata={"help": "Minimum base-evidence provenance count required for answer-slot binding."}
    )
    answer_slot_binding_optimizer_require_role_constraint: bool = field(
        default=True,
        metadata={"help": "Require relation/role cues before answer-slot binding rewrites."}
    )
    answer_slot_binding_optimizer_protect_supported_answers: bool = field(
        default=True,
        metadata={"help": "Protect supported, slot-compatible reader answers from answer-slot binding rewrites."}
    )
    answer_slot_binding_optimizer_enable_format_alias_rewrite: bool = field(
        default=True,
        metadata={"help": "Allow only high-confidence non-lossy format/alias answer-slot rewrites."}
    )
    answer_slot_binding_optimizer_enable_slot_mismatch_rewrite: bool = field(
        default=False,
        metadata={"help": "Allow slot-mismatch answer-slot rewrites after relation-entailment verification."}
    )
    answer_slot_binding_optimizer_enable_unsupported_rewrite: bool = field(
        default=False,
        metadata={"help": "Allow unsupported-answer binding rewrites after relation-entailment verification."}
    )
    answer_slot_binding_optimizer_enable_cross_module_rewrite: bool = field(
        default=False,
        metadata={"help": "Allow cross-module contract binding rewrites after relation-entailment verification."}
    )
    answer_slot_binding_optimizer_min_relation_entailment_score: float = field(
        default=0.20,
        metadata={"help": "Minimum relation score for risky answer-slot binding rewrites."}
    )
    answer_slot_binding_optimizer_min_relation_question_overlap: float = field(
        default=0.10,
        metadata={"help": "Minimum question overlap for risky answer-slot binding rewrites."}
    )
    enable_answer_surface_lattice_optimizer: bool = field(
        default=False,
        metadata={"help": "Run a final cost/risk optimizer over answer surfaces proposed by reader and answer modules."}
    )
    answer_surface_lattice_optimizer_apply_rewrite: bool = field(
        default=True,
        metadata={"help": "If false, answer-surface lattice only records shadow diagnostics."}
    )
    answer_surface_lattice_optimizer_min_net_utility: float = field(
        default=0.08,
        metadata={"help": "Minimum utility-minus-risk objective for answer-surface lattice rewrites."}
    )
    answer_surface_lattice_optimizer_min_margin: float = field(
        default=0.035,
        metadata={"help": "Minimum net-utility margin over the current answer surface."}
    )
    answer_surface_lattice_optimizer_min_support_gain: float = field(
        default=0.02,
        metadata={"help": "Minimum evidence support gain before answer-surface lattice rewriting."}
    )
    answer_surface_lattice_optimizer_max_risk: float = field(
        default=0.48,
        metadata={"help": "Maximum allowed answer-surface rewrite risk."}
    )
    answer_surface_lattice_optimizer_max_docs: int = field(
        default=10,
        metadata={"help": "Maximum QA evidence documents scanned by answer-surface lattice."}
    )
    answer_surface_lattice_optimizer_max_answer_tokens: int = field(
        default=16,
        metadata={"help": "Maximum single-answer candidate token count allowed by answer-surface lattice."}
    )
    answer_surface_lattice_optimizer_risk_weight: float = field(
        default=0.65,
        metadata={"help": "Weight applied to answer-surface rewrite risk in the lattice objective."}
    )
    answer_surface_lattice_optimizer_prefer_contract_margin: float = field(
        default=0.015,
        metadata={"help": "Lower margin threshold for query-answer-contract surfaces with non-negative support gain."}
    )
    answer_surface_lattice_optimizer_abstain_min_runner_up_margin: float = field(
        default=0.08,
        metadata={"help": "Minimum first-vs-second candidate margin when replacing an abstaining answer."}
    )
    answer_surface_lattice_optimizer_require_lineage_for_relational_abstain: bool = field(
        default=True,
        metadata={"help": "Require relational abstain repairs to agree with an available terminal fact-join lineage."}
    )
    enable_schema_touched_answer_surface_canonicalization: bool = field(
        default=False,
        metadata={"help": "Canonicalize high-confidence answer-surface granularity only for schema-memory touched queries."}
    )
    schema_touched_answer_surface_canonicalization_min_confidence: float = field(
        default=0.78,
        metadata={"help": "Minimum confidence for schema-touched answer-surface canonicalization rewrites."}
    )
    enable_plan_consistent_answer_verifier: bool = field(
        default=False,
        metadata={"help": "Run a plan/slot/provenance verifier over final answer candidates after answer-surface lattice."}
    )
    plan_consistent_answer_verifier_apply_rewrite: bool = field(
        default=False,
        metadata={"help": "If false, the plan-consistent answer verifier records shadow diagnostics only."}
    )
    plan_consistent_answer_verifier_min_plan_score: float = field(
        default=0.42,
        metadata={"help": "Minimum plan-consistency score required for verifier rewrites."}
    )
    plan_consistent_answer_verifier_min_margin: float = field(
        default=0.055,
        metadata={"help": "Minimum plan-score margin over the second candidate."}
    )
    plan_consistent_answer_verifier_min_relation_gain: float = field(
        default=0.035,
        metadata={"help": "Minimum relation-cue gain over the incumbent answer for relation-bound rewrites."}
    )
    plan_consistent_answer_verifier_min_support_score: float = field(
        default=0.20,
        metadata={"help": "Minimum reader-visible evidence support required for verifier candidates."}
    )
    plan_consistent_answer_verifier_max_docs: int = field(
        default=10,
        metadata={"help": "Maximum base QA evidence documents scanned by the plan-consistent verifier."}
    )
    plan_consistent_answer_verifier_max_candidates: int = field(
        default=18,
        metadata={"help": "Maximum plan-consistent verifier candidates retained for diagnostics."}
    )
    plan_consistent_answer_verifier_protect_supported_answers: bool = field(
        default=True,
        metadata={"help": "Protect incumbent answers that are already evidence-supported and relation-bound."}
    )
    plan_consistent_answer_verifier_allow_non_abstain_unsupported_rewrite: bool = field(
        default=False,
        metadata={
            "help": "Allow rewriting unsupported non-abstain answers. Disabled by default because weak support checks can miss valid concise answers."
        }
    )
    plan_consistent_answer_verifier_allow_slot_rewrite: bool = field(
        default=False,
        metadata={"help": "Allow answer-slot mismatch rewrites in the plan-consistent verifier."}
    )
    plan_consistent_answer_verifier_allow_relation_rewrite: bool = field(
        default=False,
        metadata={"help": "Allow relation-bound rewrites of already supported answers in the plan-consistent verifier."}
    )
    plan_consistent_answer_verifier_allow_format_rewrite: bool = field(
        default=False,
        metadata={"help": "Allow format/alias rewrites in the plan-consistent verifier."}
    )
    enable_cost_gated_constrained_reread: bool = field(
        default=False,
        metadata={"help": "Run a constrained second reader only for unsupported answers with sufficient evidence."}
    )
    cost_gated_constrained_reread_apply_rewrite: bool = field(
        default=True,
        metadata={"help": "If false, constrained reread records its verified candidate in shadow mode."}
    )
    cost_gated_constrained_reread_min_net_utility: float = field(
        default=0.42,
        metadata={"help": "Minimum estimated utility-minus-risk-and-cost required for a second reader call."}
    )
    cost_gated_constrained_reread_min_candidate_support: float = field(
        default=0.25,
        metadata={"help": "Minimum cited-evidence support score required for the reread answer."}
    )
    cost_gated_constrained_reread_min_support_gain: float = field(
        default=0.05,
        metadata={"help": "Minimum cited-evidence support gain over the incumbent answer."}
    )
    cost_gated_constrained_reread_max_docs: int = field(
        default=8,
        metadata={"help": "Maximum provenance evidence rows shown to the constrained second reader."}
    )
    cost_gated_constrained_reread_max_doc_chars: int = field(
        default=900,
        metadata={"help": "Maximum characters retained from each second-reader evidence row."}
    )
    cost_gated_constrained_reread_max_total_chars: int = field(
        default=6200,
        metadata={"help": "Maximum total evidence characters passed to the constrained second reader."}
    )
    cost_gated_constrained_reread_max_answer_tokens: int = field(
        default=16,
        metadata={"help": "Maximum candidate answer tokens accepted from constrained reread."}
    )
    cost_gated_constrained_reread_max_completion_tokens: int = field(
        default=120,
        metadata={"help": "Maximum completion tokens for the constrained reread JSON response."}
    )
    enable_provenance_bound_answer_materializer: bool = field(
        default=False,
        metadata={"help": "Compile a typed answer domain and materialize a provenance-bound final answer."}
    )
    provenance_bound_answer_materializer_apply_rewrite: bool = field(
        default=False,
        metadata={"help": "Commit verified answer-domain materializations; otherwise retain them as shadow decisions."}
    )
    provenance_bound_answer_materializer_require_terminal_plan: bool = field(
        default=False,
        metadata={"help": "Require a compiled $answer terminal operator and reject intermediate-slot materialization."}
    )
    provenance_bound_answer_materializer_use_terminal_projection_rows: bool = field(
        default=False,
        metadata={"help": "Consume only executor-normalized, identity-safe terminal projection rows."}
    )
    provenance_bound_answer_materializer_use_schema_terminal_rows: bool = field(
        default=False,
        metadata={"help": "Expose unique, lineage-complete Schema Memory terminal tuples to late answer materialization."}
    )
    provenance_bound_answer_materializer_schema_terminal_min_score: float = field(
        default=0.80,
        metadata={"help": "Minimum Schema Memory score for a terminal tuple shown to the answer materializer."}
    )
    provenance_bound_answer_materializer_schema_conflict_min_confidence: float = field(
        default=0.85,
        metadata={"help": "Minimum materializer confidence for overriding a supported incumbent with a unique schema terminal tuple."}
    )
    provenance_bound_answer_materializer_enable_deterministic_terminal_projection: bool = field(
        default=False,
        metadata={"help": "Directly project a unique, cost-accepted, provenance-faithful terminal tuple before the LLM fallback."}
    )
    provenance_bound_answer_materializer_deterministic_min_net_utility: float = field(
        default=0.70,
        metadata={"help": "Minimum physical-plan net utility for deterministic terminal tuple projection."}
    )
    provenance_bound_answer_materializer_enable_terminal_tuple_completion: bool = field(
        default=False,
        metadata={"help": "Complete $answer from a bridge-bound terminal tuple retained by the fact-join executor."}
    )
    provenance_bound_answer_materializer_terminal_tuple_completion_apply_rewrite: bool = field(
        default=False,
        metadata={"help": "Commit provenance-bound terminal tuple completion; otherwise record its counterfactual in shadow mode."}
    )
    provenance_bound_answer_materializer_terminal_tuple_completion_min_net_utility: float = field(
        default=0.42,
        metadata={"help": "Minimum risk-adjusted utility for provenance-bound terminal tuple completion."}
    )
    provenance_bound_answer_materializer_min_trigger_risk: float = field(
        default=0.34,
        metadata={"help": "Minimum incumbent answer risk required to invoke answer-domain materialization."}
    )
    provenance_bound_answer_materializer_min_candidate_confidence: float = field(
        default=0.72,
        metadata={"help": "Minimum model-reported confidence for a provenance-bound candidate."}
    )
    provenance_bound_answer_materializer_min_binding_score: float = field(
        default=0.50,
        metadata={"help": "Minimum cited relation-binding score for a materialized answer."}
    )
    provenance_bound_answer_materializer_min_binding_gain: float = field(
        default=0.08,
        metadata={"help": "Minimum relation-binding gain over the incumbent answer."}
    )
    provenance_bound_answer_materializer_max_docs: int = field(
        default=6,
        metadata={"help": "Maximum reader-visible provenance rows shown to the materializer."}
    )
    provenance_bound_answer_materializer_max_doc_chars: int = field(
        default=1050,
        metadata={"help": "Maximum characters retained per provenance row."}
    )
    provenance_bound_answer_materializer_max_total_chars: int = field(
        default=6200,
        metadata={"help": "Maximum total provenance characters passed to the materializer."}
    )
    provenance_bound_answer_materializer_max_answer_tokens: int = field(
        default=14,
        metadata={"help": "Maximum accepted materialized answer length."}
    )
    provenance_bound_answer_materializer_max_completion_tokens: int = field(
        default=120,
        metadata={"help": "Maximum completion tokens for the materializer JSON response."}
    )
    provenance_bound_answer_materializer_workers: int = field(
        default=4,
        metadata={"help": "Maximum concurrent API calls for provenance-bound answer materialization."}
    )
    cost_gated_constrained_reread_require_role_constraint: bool = field(
        default=True,
        metadata={"help": "Require query-role evidence in the cited reread provenance."}
    )
    enable_anchored_typed_candidate_table: bool = field(
        default=False,
        metadata={"help": "Append an anchored, typed candidate evidence table before reader QA."}
    )
    anchored_typed_candidate_table_max_docs: int = field(
        default=8,
        metadata={"help": "Maximum QA source documents scanned by the anchored typed candidate table."}
    )
    anchored_typed_candidate_table_max_rows: int = field(
        default=5,
        metadata={"help": "Maximum candidate rows inserted into the anchored typed candidate table."}
    )
    anchored_typed_candidate_table_min_score: float = field(
        default=0.72,
        metadata={"help": "Minimum typed table score required for a candidate row to be shown to the reader."}
    )
    anchored_typed_candidate_table_min_question_overlap: float = field(
        default=0.08,
        metadata={"help": "Minimum question overlap for anchored typed candidate rows."}
    )
    anchored_typed_candidate_table_min_provenance: int = field(
        default=1,
        metadata={"help": "Minimum provenance count for anchored typed candidate rows."}
    )
    anchored_typed_candidate_table_require_role_constraint: bool = field(
        default=True,
        metadata={"help": "Require relation/role cues for anchored typed candidate rows."}
    )
    anchored_typed_candidate_table_insert_after: int = field(
        default=2,
        metadata={"help": "Insert the anchored typed candidate table after this many reader evidence docs."}
    )
    enable_evidence_cost_model: bool = field(
        default=False,
        metadata={"help": "Use a cost-based evidence operator optimizer before reader-visible evidence insertion."}
    )
    evidence_cost_model_token_budget: int = field(
        default=220,
        metadata={"help": "Approximate token budget used to normalize reader-visible evidence insertion cost."}
    )
    evidence_cost_model_min_utility: float = field(
        default=0.50,
        metadata={"help": "Minimum estimated evidence utility required by the cost model."}
    )
    evidence_cost_model_max_risk: float = field(
        default=0.55,
        metadata={"help": "Maximum estimated reader/rank intervention risk allowed by the cost model."}
    )
    evidence_cost_model_min_net_utility: float = field(
        default=0.08,
        metadata={"help": "Minimum utility minus cost/risk objective required by the cost model."}
    )
    evidence_cost_model_token_weight: float = field(
        default=0.18,
        metadata={"help": "Weight applied to normalized token cost in the evidence cost objective."}
    )
    evidence_cost_model_compute_weight: float = field(
        default=0.08,
        metadata={"help": "Weight applied to local candidate-generation cost in the evidence cost objective."}
    )
    evidence_cost_model_risk_weight: float = field(
        default=0.62,
        metadata={"help": "Weight applied to estimated reader distraction/intervention risk in the evidence cost objective."}
    )
    enable_relation_fact_join_retrieval: bool = field(
        default=False,
        metadata={"help": "Use relation-aware OpenIE fact joins to select reader-only evidence under the cost optimizer."}
    )
    relation_fact_join_bridge_only: bool = field(
        default=False,
        metadata={"help": "Only run relation-aware fact join retrieval on bridge/compositional questions."}
    )
    relation_fact_join_require_evidence_gap: bool = field(
        default=True,
        metadata={"help": "Require a current evidence gap before appending relation fact-join evidence."}
    )
    relation_fact_join_candidate_top_k: int = field(
        default=72,
        metadata={"help": "Maximum scored relation fact-join candidate passages considered per query."}
    )
    relation_fact_join_max_append: int = field(
        default=1,
        metadata={"help": "Maximum relation fact-join passages appended to the QA reader."}
    )
    relation_fact_join_max_facts_per_key: int = field(
        default=60,
        metadata={"help": "Maximum fact rows scanned per entity/relation/schema join key."}
    )
    relation_fact_join_max_key_fanout: int = field(
        default=600,
        metadata={"help": "Skip relation fact-join keys with fanout above this threshold."}
    )
    relation_fact_join_protect_top_k: int = field(
        default=2,
        metadata={"help": "Treat the first k retrieved passages as protected from relation fact-join replacement."}
    )
    relation_fact_join_min_query_overlap: float = field(
        default=0.04,
        metadata={"help": "Minimum query overlap signal used by relation fact-join cost/risk estimation."}
    )
    relation_fact_join_min_relation_overlap: float = field(
        default=0.04,
        metadata={"help": "Minimum relation overlap signal used by relation fact-join cost/risk estimation."}
    )
    relation_fact_join_min_utility: float = field(
        default=0.50,
        metadata={"help": "Minimum estimated utility for relation fact-join evidence."}
    )
    relation_fact_join_max_risk: float = field(
        default=0.52,
        metadata={"help": "Maximum estimated risk for relation fact-join evidence."}
    )
    relation_fact_join_min_net_utility: float = field(
        default=0.04,
        metadata={"help": "Minimum utility minus cost/risk objective for relation fact-join evidence."}
    )
    relation_fact_join_token_budget: int = field(
        default=850,
        metadata={"help": "Approximate token budget used to normalize relation fact-join evidence cost."}
    )
    relation_fact_join_use_answerability_gap: bool = field(
        default=True,
        metadata={"help": "Use relation-role/anchor/answer-type coverage, not only query token coverage, to decide fact-join evidence gaps."}
    )
    relation_fact_join_require_relation_role_match: bool = field(
        default=True,
        metadata={"help": "Require relation fact-join candidates to match the question's relation role and polarity."}
    )
    relation_fact_join_require_anchor_match: bool = field(
        default=True,
        metadata={"help": "Require relation fact-join candidates to share a query anchor with the candidate fact endpoints or document."}
    )
    relation_fact_join_min_anchor_overlap: float = field(
        default=0.34,
        metadata={"help": "Minimum bounded anchor overlap for verified relation fact-join candidates."}
    )
    relation_fact_join_min_role_overlap: float = field(
        default=0.34,
        metadata={"help": "Minimum bounded relation-role overlap for verified relation fact-join candidates."}
    )
    relation_fact_join_enable_candidate_table: bool = field(
        default=True,
        metadata={"help": "Build a compact directional fact-join candidate answer table for the QA reader."}
    )
    relation_fact_join_candidate_table_max_rows: int = field(
        default=4,
        metadata={"help": "Maximum rows in the directional fact-join candidate answer table."}
    )
    relation_fact_join_candidate_table_min_score: float = field(
        default=0.56,
        metadata={"help": "Minimum row score for the directional fact-join candidate answer table."}
    )
    relation_fact_join_hard_anchor_min_overlap: float = field(
        default=0.67,
        metadata={"help": "Minimum endpoint/query anchor overlap for hard-bound directional fact joins."}
    )
    relation_fact_join_append_mode: str = field(
        default="table",
        metadata={
            "help": (
                "Relation fact-join reader append mode: trace_only, table, "
                "passage, table_and_passage, witness_certificate, or "
                "certificate_and_passage."
            )
        }
    )
    relation_fact_join_enable_multi_hop_planner: bool = field(
        default=True,
        metadata={"help": "Compile selected questions into typed multi-hop fact-join plans before building candidate tables."}
    )
    relation_fact_join_multi_hop_min_bridge_overlap: float = field(
        default=0.67,
        metadata={"help": "Minimum canonical endpoint overlap required between adjacent facts in a typed multi-hop join path."}
    )
    relation_fact_join_enable_surface_materializer: bool = field(
        default=True,
        metadata={"help": "Materialize full answer surface values from evidence sentences for typed multi-hop fact-join rows."}
    )
    relation_fact_join_enable_provenance_optimizer: bool = field(
        default=True,
        metadata={"help": "Optimize typed fact-join candidate tables using provenance support, conflict sets, and source costs."}
    )
    relation_fact_join_provenance_conflict_min_margin: float = field(
        default=0.08,
        metadata={"help": "Minimum optimized-score margin before a provenance conflict set is considered resolved."}
    )
    relation_fact_join_provenance_min_support_docs: int = field(
        default=2,
        metadata={"help": "Minimum corpus provenance documents required before fact-join provenance can actively adjust candidate rows."}
    )
    relation_fact_join_provenance_min_reliability: float = field(
        default=0.72,
        metadata={"help": "Minimum persistent fact reliability score required for active provenance adjustment."}
    )
    relation_fact_join_provenance_max_bonus: float = field(
        default=0.025,
        metadata={"help": "Maximum score bonus allowed for a high-confidence provenance-supported fact-join row."}
    )
    relation_fact_join_provenance_shadow_only: bool = field(
        default=False,
        metadata={"help": "If true, compute fact reliability diagnostics but do not alter candidate-table scores."}
    )
    enable_typed_fact_join_shadow_optimizer: bool = field(
        default=False,
        metadata={"help": "Build shadow-only typed fact-store join plans and oracle diagnostics without changing retrieval or reader evidence."}
    )
    typed_fact_join_shadow_candidate_top_k: int = field(
        default=200,
        metadata={"help": "Maximum typed fact rows considered by the shadow join optimizer per query."}
    )
    typed_fact_join_shadow_max_anchor_items: int = field(
        default=48,
        metadata={"help": "Maximum anchor-side fact rows used when enumerating shadow join plans."}
    )
    typed_fact_join_shadow_max_second_hop_items: int = field(
        default=64,
        metadata={"help": "Maximum second-hop fact rows scanned for each bridge endpoint in shadow join planning."}
    )
    typed_fact_join_shadow_max_plans: int = field(
        default=8,
        metadata={"help": "Maximum shadow join plans stored in diagnostics per query."}
    )
    typed_fact_join_shadow_max_key_fanout: int = field(
        default=800,
        metadata={"help": "Skip shadow join lookup keys whose posting lists exceed this fanout."}
    )
    typed_fact_join_shadow_min_utility: float = field(
        default=0.46,
        metadata={"help": "Minimum estimated utility for a shadow typed fact join plan to be marked accepted."}
    )
    typed_fact_join_shadow_max_risk: float = field(
        default=0.58,
        metadata={"help": "Maximum estimated risk for a shadow typed fact join plan to be marked accepted."}
    )
    typed_fact_join_shadow_min_net_utility: float = field(
        default=0.02,
        metadata={"help": "Minimum cost-adjusted net utility for a shadow typed fact join plan."}
    )
    typed_fact_join_shadow_token_budget: int = field(
        default=1200,
        metadata={"help": "Approximate token budget used to normalize shadow join plan materialization cost."}
    )
    typed_fact_join_shadow_token_weight: float = field(
        default=0.08,
        metadata={"help": "Weight applied to token cost in the shadow typed fact join cost model."}
    )
    typed_fact_join_shadow_compute_weight: float = field(
        default=0.05,
        metadata={"help": "Weight applied to local enumeration cost in the shadow typed fact join cost model."}
    )
    typed_fact_join_shadow_risk_weight: float = field(
        default=0.36,
        metadata={"help": "Weight applied to estimated risk in the shadow typed fact join cost model."}
    )
    enable_slot_separated_fact_join_view: bool = field(
        default=False,
        metadata={"help": "Append a reader-only slot-separated fact-join view from accepted typed shadow plans."}
    )
    slot_separated_fact_join_max_rows: int = field(
        default=3,
        metadata={"help": "Maximum answer-slot fact-join rows shown in the slot-separated view."}
    )
    slot_separated_fact_join_min_utility: float = field(
        default=0.54,
        metadata={"help": "Minimum accepted plan utility required for slot-separated fact-join rows."}
    )
    slot_separated_fact_join_max_risk: float = field(
        default=0.50,
        metadata={"help": "Maximum accepted plan risk allowed for slot-separated fact-join rows."}
    )
    slot_separated_fact_join_min_net_utility: float = field(
        default=0.12,
        metadata={"help": "Minimum cost-adjusted utility required for slot-separated fact-join rows."}
    )
    slot_separated_fact_join_insert_after: int = field(
        default=2,
        metadata={"help": "Insert the slot-separated fact-join view after this many leading QA documents."}
    )
    slot_separated_fact_join_require_answer_slot: bool = field(
        default=True,
        metadata={"help": "Only insert the slot-separated view when at least one final-answer slot row survives."}
    )
    slot_separated_fact_join_bridge_or_deficit_only: bool = field(
        default=True,
        metadata={"help": "Only insert the slot-separated view for bridge/compositional questions or evidence deficits."}
    )
    slot_separated_fact_join_block_long_context: bool = field(
        default=True,
        metadata={"help": "Block the slot-separated view for long-context evidence unless bridge cues are present."}
    )
    slot_separated_fact_join_long_context_avg_chars: int = field(
        default=1400,
        metadata={"help": "Average source-document length threshold for long-context slot-separated blocking."}
    )
    slot_separated_fact_join_allow_one_hop_answer_rows: bool = field(
        default=False,
        metadata={"help": "Allow one-hop shadow plans to appear as answer rows in the slot-separated view."}
    )
    slot_separated_fact_join_require_predicate_match: bool = field(
        default=True,
        metadata={"help": "Require the final fact-join hop predicate to match the query predicate slot."}
    )
    slot_separated_fact_join_block_boolean_comparison: bool = field(
        default=True,
        metadata={"help": "Block slot-separated candidate tables for boolean or comparison questions."}
    )
    slot_separated_fact_join_min_slot_score: float = field(
        default=0.50,
        metadata={"help": "Minimum slot-level confidence required before inserting a slot-separated row."}
    )
    enable_selective_support_join_optimizer: bool = field(
        default=False,
        metadata={"help": "Admit low-risk typed fact-join support evidence as reader-only append without changing retrieval ranking."}
    )
    selective_support_join_bridge_only: bool = field(
        default=True,
        metadata={"help": "Only run selective support-join admission for bridge/compositional query shapes."}
    )
    selective_support_join_max_append: int = field(
        default=1,
        metadata={"help": "Maximum selected support-join passages appended to the reader."}
    )
    selective_support_join_max_candidate_plans: int = field(
        default=12,
        metadata={"help": "Maximum typed fact-join shadow plans considered by selective support admission."}
    )
    selective_support_join_max_original_rank: int = field(
        default=80,
        metadata={"help": "Deepest original retrieval rank allowed for reader-only support-join evidence."}
    )
    selective_support_join_min_plan_utility: float = field(
        default=0.48,
        metadata={"help": "Minimum uncosted plan utility required by selective support-join admission."}
    )
    selective_support_join_min_plan_net_utility: float = field(
        default=0.22,
        metadata={"help": "Minimum cost-adjusted plan utility required by selective support-join admission."}
    )
    selective_support_join_high_conf_net_utility: float = field(
        default=0.42,
        metadata={"help": "Net utility required when shadow plan enumeration is high-fanout."}
    )
    selective_support_join_max_plan_risk: float = field(
        default=0.48,
        metadata={"help": "Maximum plan risk allowed by selective support-join admission."}
    )
    selective_support_join_min_missing_gain: float = field(
        default=0.04,
        metadata={"help": "Minimum missing-query-token gain required from an appended support-join passage."}
    )
    selective_support_join_min_relation_overlap: float = field(
        default=0.08,
        metadata={"help": "Minimum query relation overlap required by selective support-join admission."}
    )
    selective_support_join_min_endpoint_overlap: float = field(
        default=0.22,
        metadata={"help": "Minimum endpoint/anchor overlap required by selective support-join admission."}
    )
    selective_support_join_min_schema_selectivity: float = field(
        default=0.62,
        metadata={"help": "Minimum schema selectivity required by selective support-join admission."}
    )
    selective_support_join_min_admission_score: float = field(
        default=0.28,
        metadata={"help": "Minimum final admission score after support gain, cost, risk, and selectivity."}
    )
    selective_support_join_max_accepted_plan_count: int = field(
        default=96,
        metadata={"help": "Accepted-plan fanout above which stricter high-confidence admission is required."}
    )
    selective_support_join_require_two_hop_for_bridge: bool = field(
        default=True,
        metadata={"help": "Require two-hop fact-join plans for bridge/compositional support admission."}
    )
    selective_support_join_require_deficit: bool = field(
        default=True,
        metadata={"help": "Require a current evidence-state deficit before appending support-join evidence."}
    )
    selective_support_join_min_current_query_coverage: float = field(
        default=0.62,
        metadata={"help": "Minimum query-token coverage before current evidence is considered complete."}
    )
    selective_support_join_min_current_relation_coverage: float = field(
        default=0.50,
        metadata={"help": "Minimum relation-token coverage before current evidence is considered relation-complete."}
    )
    selective_support_join_block_long_context: bool = field(
        default=True,
        metadata={"help": "Block support-join admission for long-context evidence unless the query has bridge cues."}
    )
    selective_support_join_long_context_avg_chars: int = field(
        default=1400,
        metadata={"help": "Average current evidence length treated as long-context for support-join blocking."}
    )
    selective_support_join_token_budget: int = field(
        default=900,
        metadata={"help": "Approximate token budget used to normalize support-join reader admission cost."}
    )
    enable_cost_based_evidence_admission: bool = field(
        default=False,
        metadata={"help": "Use a unified cost/selectivity optimizer to admit reader-only extra evidence without changing retrieval top-k."}
    )
    cost_based_admission_trigger_policy: str = field(
        default="high_risk_only",
        metadata={"help": "When to optimize appended reader evidence: high_risk_only or always."}
    )
    cost_based_admission_policy: str = field(
        default="learned",
        metadata={
            "help": (
                "Evidence plan policy over a fixed candidate pool: learned, rule, "
                "learned_veto, fixed_order, or always_on."
            )
        }
    )
    cost_based_admission_optimize_existing_extras: bool = field(
        default=False,
        metadata={
            "help": (
                "Place existing reader-side operator evidence in the same budgeted "
                "candidate pool as schema/shadow evidence."
            )
        }
    )
    cost_based_admission_max_extra_docs: int = field(
        default=2,
        metadata={"help": "Maximum reader-only extra evidence documents admitted after stable retrieval docs."}
    )
    cost_based_admission_token_budget: int = field(
        default=900,
        metadata={"help": "Approximate token budget used to normalize candidate evidence admission cost."}
    )
    cost_based_admission_min_utility: float = field(
        default=0.50,
        metadata={"help": "Minimum estimated utility required before admitting appended evidence."}
    )
    cost_based_admission_max_risk: float = field(
        default=0.52,
        metadata={"help": "Maximum estimated reader-distraction risk allowed for appended evidence."}
    )
    cost_based_admission_min_net_utility: float = field(
        default=0.08,
        metadata={"help": "Minimum cost-adjusted net utility required for appended evidence."}
    )
    cost_based_admission_min_query_overlap: float = field(
        default=0.05,
        metadata={"help": "Minimum query or missing-slot overlap for non-table appended evidence."}
    )
    cost_based_admission_token_weight: float = field(
        default=0.12,
        metadata={"help": "Weight applied to normalized token cost in evidence admission."}
    )
    cost_based_admission_compute_weight: float = field(
        default=0.04,
        metadata={"help": "Weight applied to local candidate-generation cost in evidence admission."}
    )
    cost_based_admission_risk_weight: float = field(
        default=0.42,
        metadata={"help": "Weight applied to estimated reader-distraction risk in evidence admission."}
    )
    cost_based_admission_redundancy_penalty: float = field(
        default=0.08,
        metadata={"help": "Penalty for admitting evidence redundant with stable retrieval docs."}
    )
    cost_based_admission_preserve_best_on_empty: bool = field(
        default=False,
        metadata={"help": "Keep the best appended evidence when all candidates narrowly fail admission thresholds."}
    )
    cost_based_admission_deficit_replacement_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Allow a safe fact-join candidate that covers an uncovered slot to "
                "replace lower-marginal evidence under the same document/token budget."
            )
        }
    )
    cost_based_admission_deficit_replacement_min_margin: float = field(
        default=0.08,
        metadata={"help": "Minimum marginal-priority gain required for deficit-aware evidence replacement."}
    )
    cost_based_admission_deficit_replacement_max_evictions: int = field(
        default=1,
        metadata={"help": "Maximum accepted evidence rows evicted for one deficit-aware replacement."}
    )
    cost_based_admission_shadow_candidate_top_k: int = field(
        default=1,
        metadata={"help": "Maximum typed fact-join shadow candidate passages exposed to cost-based reader evidence admission."}
    )
    cost_based_admission_shadow_max_original_rank: int = field(
        default=50,
        metadata={"help": "Deepest original retrieval rank allowed for typed fact-join shadow admission candidates."}
    )
    cost_based_admission_shadow_min_plan_utility: float = field(
        default=0.52,
        metadata={"help": "Minimum typed fact-join plan utility before exposing it to cost-based admission."}
    )
    cost_based_admission_shadow_min_plan_net_utility: float = field(
        default=0.10,
        metadata={"help": "Minimum cost-adjusted typed fact-join plan utility before exposing it to admission."}
    )
    cost_based_admission_shadow_max_plan_risk: float = field(
        default=0.50,
        metadata={"help": "Maximum typed fact-join plan risk before exposing it to cost-based admission."}
    )
    cost_based_admission_shadow_require_two_hop_for_bridge: bool = field(
        default=True,
        metadata={"help": "Require two-hop typed fact-join plans for bridge-like questions unless a one-hop plan is high confidence."}
    )
    cost_based_admission_calibrated_shadow_admission: bool = field(
        default=False,
        metadata={"help": "Allow calibrated high-confidence typed shadow evidence to bypass conservative bridge/source-portfolio gates without changing top-k ranking."}
    )
    cost_based_admission_source_sufficiency_veto: bool = field(
        default=True,
        metadata={"help": "Block schema/shadow reader evidence when its source operator or slot lineage says current evidence is already sufficient."}
    )
    cost_based_fact_join_plan_optimizer_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enumerate no-op and multi-operator evidence plans, then choose "
                "the best feasible slate under token and access-cost budgets."
            )
        }
    )
    cost_based_fact_join_plan_compute_budget: float = field(
        default=1.0,
        metadata={"help": "Maximum normalized fact-join access cost per query plan."}
    )
    cost_based_fact_join_plan_min_objective: float = field(
        default=0.0,
        metadata={"help": "Minimum plan objective required to beat the no-op plan."}
    )
    cost_based_fact_join_plan_max_enumerated_candidates: int = field(
        default=12,
        metadata={"help": "Maximum locally eligible candidates used for exact slate enumeration."}
    )
    enable_paper_evidence_query_executor: bool = field(
        default=False,
        metadata={
            "help": (
                "Use the paper-facing LOOKUP_JOIN/COMPLETE/VERIFY execution "
                "loop and bypass legacy eager evidence operators."
            )
        }
    )
    paper_evidence_query_compiler_mode: Literal[
        "rule", "rule_contract_v1", "hybrid_v1", "hybrid_v2", "hybrid_v2_1", "hybrid_v2_2",
        "rule_first_semantic_v3", "rule_first_semantic_v3_1",
        "rule_first_semantic_v3_12", "rule_first_semantic_v3_13",
    ] = field(
        default="rule",
        metadata={
            "help": (
                "Rule-only compilation, the legacy physical-IR LLM fallback, "
                "semantic-IR V2, V2.1 semantic-variable lowering, or V2.2 "
                "semantic-requirement coverage after rule validation fails, or "
                "V3.1 constrained generation, the canonical V3.12 semantic "
                "skeleton, or the V3.13 nested-chain and unique-direction repair "
                "path with deterministic lowering and explicit admission stages."
            )
        },
    )
    paper_evidence_query_compiler_max_completion_tokens: int = field(
        default=700,
        metadata={
            "help": "Maximum completion tokens for one hybrid compiler call."
        },
    )
    paper_evidence_allow_model_based_semantic_admission: bool = field(
        default=False,
        metadata={
            "help": (
                "Development-only execution admission for plans that pass the "
                "rule-first Semantic V3 independent model review. The default "
                "remains false and does not alter human-review admission."
            )
        },
    )
    paper_evidence_require_semantic_execution_admission: bool = field(
        default=False,
        metadata={
            "help": (
                "Require a current automatic or explicitly authorized model "
                "semantic-admission certificate before the paper executor "
                "may consume a compiled plan."
            )
        },
    )
    paper_evidence_baseline_mode: Literal[
        "core", "dense_reader", "static_plan_atomic"
    ] = field(
        default="core",
        metadata={
            "help": (
                "Run the normal support-aware executor or the existing "
                "static plan-derived atomic retrieval baseline through the "
                "same Reader path."
            )
        },
    )
    paper_evidence_atomic_max_retrieval_calls: int = field(
        default=2,
        metadata={"help": "Maximum plan-derived retrieval calls for Atomic."},
    )
    paper_evidence_atomic_max_extra_sources: int = field(
        default=3,
        metadata={"help": "Maximum Atomic sources appended to the Reader."},
    )
    paper_evidence_policy: Literal[
        "fixed_early_verify",
        "fixed_late_verify",
        "greedy_immediate",
        "adaptive_dependency",
        "adaptive_dependency_frozen",
        "feasibility_constrained_adaptive",
        "support_aware_cbo",
        "bswc_probability_only",
        "bswc_relevance",
        "bswc_lowest_cost",
        "bswc_set_cover_greedy",
        "bswc_adaptive_greedy",
        "bswc_bounded_search",
        "bswc_full",
        "bswc_empirical_source_type",
        "bswc_feedback_calibrated",
        "bswc_feedback_replan",
        "bswc_feedback_stopping",
        "bswc_witness_calibrated",
        "bswc_witness_stopping",
        "bswc_feedback_full",
        "bswc_fixed_source_order",
        "bswc_random_source",
        "bswc_risk_aware_search",
        "bswc_adaptive_risk_cover",
        "bswc_feasibility_preserving_cover",
        "bswc_continuation_safe_cover",
        "bswc_binding_qualified_continuation_cover",
        "bswc_rank_consistent_binding_qualified_cover",
        "bswc_precedence_aware_cover",
        "bswc_quality_first_precedence_cover",
        "bswc_global_witness_reservation_cover",
        "bswc_exact_adaptive_witness",
        "bswc_support_oracle_early",
        "bswc_identity_safe_frontier_exposure_oracle",
        "bswc_identity_safe_scan_reservation_oracle",
    ] = field(
        default="adaptive_dependency",
        metadata={"help": "Policy used by the paper evidence executor."}
    )
    paper_evidence_max_actions: int = field(
        default=12,
        metadata={"help": "Maximum evidence actions per query."}
    )
    paper_evidence_max_retrieval_calls: int = field(
        default=2,
        metadata={"help": "Maximum COMPLETE retrieval calls per query."}
    )
    paper_evidence_max_model_calls: int = field(
        default=2,
        metadata={"help": "Maximum model calls inside evidence actions."}
    )
    paper_evidence_max_scan_rows: int = field(
        default=5000,
        metadata={"help": "Maximum fact/source rows scanned by evidence actions."}
    )
    paper_evidence_max_tokens: int = field(
        default=0,
        metadata={
            "help": (
                "Maximum total evidence-action tokens (source evidence plus "
                "known model input/output); zero means unbounded."
            )
        }
    )
    paper_evidence_reserve_model_calls: int = field(
        default=0,
        metadata={"help": "Model calls reserved from the evidence-action budget."}
    )
    paper_evidence_reserve_tokens: int = field(
        default=0,
        metadata={"help": "Tokens reserved from the evidence-action budget."}
    )
    paper_evidence_max_no_progress: int = field(
        default=3,
        metadata={"help": "Consecutive no-progress actions allowed before STOP."}
    )
    paper_evidence_cbo_calibration_path: str = field(
        default="",
        metadata={"help": "Frozen sampled action-cost model for support-aware CBO."}
    )
    paper_evidence_cbo_min_samples: int = field(
        default=8,
        metadata={"help": "Minimum samples required for a calibrated estimate."}
    )
    paper_evidence_cbo_risk_weight: float = field(
        default=0.8,
        metadata={"help": "Unsupported-action risk weight for support-aware CBO."}
    )
    paper_evidence_cbo_cost_weight: float = field(
        default=0.25,
        metadata={"help": "Normalized physical-cost weight for support-aware CBO."}
    )
    paper_evidence_bswc_beam_width: int = field(
        default=8,
        metadata={"help": "Beam width for binding-aware BSWC source search."}
    )
    paper_evidence_bswc_search_horizon: int = field(
        default=3,
        metadata={"help": "Maximum BSWC source actions in bounded lookahead."}
    )
    paper_evidence_bswc_source_calibration_path: str = field(
        default="",
        metadata={"help": "Frozen held-out calibrated source-value calibration model."}
    )
    paper_evidence_bswc_source_min_samples: int = field(
        default=8,
        metadata={"help": "Minimum observations before using a BSWC calibration cell."}
    )
    paper_evidence_bswc_feedback_strength: float = field(
        default=3.0,
        metadata={"help": "Equivalent prior observations for query-local feedback."}
    )
    paper_evidence_bswc_query_feedback: bool = field(
        default=True,
        metadata={
            "help": (
                "Enable query-local BSWC outcome feedback. This switch exists "
                "for a clean same-policy feedback ablation; global calibration, "
                "hard budgets, source candidates, and physical ranking remain fixed."
            )
        },
    )
    paper_evidence_bswc_stop_threshold: float = field(
        default=0.035,
        metadata={"help": "Minimum expected witness gain per physical cost."}
    )
    paper_evidence_bswc_min_exploration_actions: int = field(
        default=1,
        metadata={"help": "Recovery actions retained before low-value stopping is legal."}
    )
    paper_evidence_bswc_fpawc_stop_upper_bound: float = field(
        default=0.0,
        metadata={
            "help": (
                "Optimistic witness-completion upper bound at or below which "
                "feasibility-preserving BSWC may stop after exploration."
            )
        }
    )
    paper_evidence_completion_top_k: int = field(
        default=6,
        metadata={"help": "Bounded source documents inspected by COMPLETE."}
    )
    paper_evidence_completion_mode: str = field(
        default="lazy",
        metadata={
            "help": (
                "Evidence repair mode: 'lazy' executes COMPLETE after an index "
                "miss or rejected tuple; 'eager' re-extracts every executed "
                "obligation for a strong baseline; 'off' disables COMPLETE."
            )
        }
    )
    paper_evidence_completion_query_mode: str = field(
        default="obligation_bound",
        metadata={
            "help": (
                "COMPLETE source-query control: 'obligation_bound' includes the "
                "current predicate contract; 'entity_only' is the naive retry "
                "baseline."
            )
        }
    )
    paper_evidence_query_time_relation_materialization: bool = field(
        default=False,
        metadata={
            "help": (
                "After deterministic COMPLETE misses, extract an exact source-"
                "grounded candidate tuple from the bounded source pool. The tuple "
                "remains UNVERIFIED until normal evidence admission."
            )
        },
    )
    paper_evidence_binding_conditioned_typed_materialization: bool = field(
        default=False,
        metadata={
            "help": (
                "Compile exact source-local QTRM relation phrases into the "
                "current operator's typed predicate only after validated "
                "binding identity; the emitted tuple remains subject to normal "
                "source-certificate replay and obligation admission."
            )
        },
    )
    paper_evidence_qtrm_directional_surface_canonicalization: bool = field(
        default=False,
        metadata={
            "help": (
                "Recover a shortest exact relation phrase from direction-aware "
                "source constructions before typed QTRM admission. Source, "
                "binding, obligation, and VERIFY gates remain unchanged."
            )
        },
    )
    paper_evidence_query_time_relation_materialization_max_docs: int = field(
        default=4,
        metadata={"help": "Maximum bounded passages in one relation-materialization call."},
    )
    paper_evidence_query_time_relation_materialization_max_completion_tokens: int = field(
        default=320,
        metadata={"help": "Maximum output tokens for relation materialization."},
    )
    paper_evidence_qtrm_windowed_extractive_topk: bool = field(
        default=False,
        metadata={
            "help": (
                "Rank exact source windows by the active binding and predicate "
                "before QTRM candidate extraction. This changes candidate recall "
                "only; existing tuple admission gates remain authoritative."
            )
        },
    )
    paper_evidence_qtrm_windowed_max_windows: int = field(
        default=8,
        metadata={"help": "Maximum ranked source windows shown to QTRM."},
    )
    paper_evidence_qtrm_windowed_max_candidates: int = field(
        default=3,
        metadata={"help": "Maximum extractive candidates returned by windowed QTRM."},
    )
    paper_evidence_qtrm_max_passes_per_obligation: int = field(
        default=2,
        metadata={
            "help": (
                "Maximum QTRM extraction passes spent on one binding-obligation. "
                "Set to one to reserve query-level model budget for later hops."
            )
        },
    )
    paper_evidence_reader_preserve_explicit_date_precision: bool = field(
        default=False,
        metadata={
            "help": (
                "Tell the Reader to preserve an explicitly supported full date "
                "for date questions instead of collapsing it to a year."
            )
        },
    )
    paper_evidence_verification_mode: str = field(
        default="strict",
        metadata={
            "help": (
                "Evidence admission mode: 'strict' requires local source support; "
                "'obligation_only' preserves typed predicate, direction, role, "
                "binding, and output contracts while ablating source support; "
                "'unsafe_admit' is an explicitly unsafe source-pointer ablation."
            )
        }
    )
    paper_evidence_repair_scheduler_mode: str = field(
        default="eager",
        metadata={
            "help": (
                "Repair scheduling mode: 'eager' preserves the original "
                "behavior; 'terminal_critical' repairs only failed "
                "obligations that currently block an answer path."
            )
        }
    )
    paper_evidence_execution_mode: str = field(
        default="saqe",
        metadata={
            "help": (
                "SAQE mechanism control: saqe, no_reopen, "
                "no_invalidation, or restart. Intended for controlled "
                "algorithm ablations, not deployment tuning."
            )
        }
    )
    paper_evidence_index_delete_rate: float = field(
        default=0.0,
        metadata={"help": "Deterministic matched fact-index tuple deletion rate."}
    )
    paper_evidence_index_corrupt_rate: float = field(
        default=0.0,
        metadata={"help": "Deterministic matched fact-index tuple corruption rate."}
    )
    paper_evidence_index_corrupt_mode: str = field(
        default="object",
        metadata={
            "help": (
                "Fact-index corruption mode: object, relation, subtype, direction, "
                "role, provenance, or mixed."
            )
        }
    )
    paper_evidence_perturbation_seed: int = field(
        default=0,
        metadata={"help": "Seed for deterministic fact-index perturbations."}
    )
    paper_evidence_title_catalog_completion: bool = field(
        default=True,
        metadata={
            "help": (
                "Probe the exact entity-title catalog before dense COMPLETE "
                "retrieval, enabling an index-nested-loop access path."
            )
        }
    )
    paper_evidence_obligation_source_catalog: bool = field(
        default=False,
        metadata={
            "help": (
                "After a local exact-title miss, inspect a frozen external "
                "page selected only by the current bound entity. Tuples still "
                "require normal source support and obligation admission."
            )
        },
    )
    paper_evidence_obligation_source_catalog_path: str = field(
        default="",
        metadata={
            "help": (
                "JSONL path for the immutable obligation-bound source catalog."
            )
        },
    )
    paper_evidence_promote_plan_bound_docs: bool = field(
        default=False,
        metadata={
            "help": (
                "Promote verified fact-join sources and critical entity-title "
                "partitions into the final retrieval order."
            )
        },
    )
    paper_evidence_promotion_protect_dense_top_k: int = field(
        default=1,
        metadata={"help": "Dense prefix protected from plan-bound promotion."},
    )
    paper_evidence_promotion_max_title_docs: int = field(
        default=3,
        metadata={"help": "Maximum critical title-partition docs promoted per query."},
    )
    paper_evidence_promotion_selection_mode: str = field(
        default="legacy",
        metadata={
            "help": (
                "Plan-bound retrieval selection mode: legacy appends all "
                "eligible plan docs; support_aware_topk fills a bounded top-k "
                "by supported obligation coverage before dense rank."
            )
        },
    )
    paper_evidence_promotion_target_top_k: int = field(
        default=5,
        metadata={"help": "Reader-visible top-k optimized by plan-bound promotion."},
    )
    paper_evidence_promotion_dense_candidate_top_k: int = field(
        default=10,
        metadata={
            "help": "Dense prefix considered by support-aware bounded top-k selection."
        },
    )
    paper_evidence_provenance_source_verifier: bool = field(
        default=True,
        metadata={
            "help": (
                "Allow title-scoped pronoun, lifespan, and identity refinement "
                "rules while retaining source-lineage verification."
            )
        }
    )
    paper_evidence_source_grounded_terminal_tuple_recovery: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable source-grounded terminal tuple recovery records for "
                "COMPLETE. Recovered tuples remain UNVERIFIED until the "
                "unchanged source-support verifier admits them."
            )
        },
    )
    paper_evidence_obligation_bound_repair_certificate: bool = field(
        default=False,
        metadata={
            "help": (
                "Attach an exact source-span and typed-obligation certificate "
                "to indexed tuples produced by COMPLETE. The certificate is "
                "revalidated before a repaired tuple can re-enter execution."
            )
        },
    )
    paper_evidence_terminal_candidate_priority: bool = field(
        default=False,
        metadata={
            "help": (
                "Order final LOOKUP_JOIN candidates by source-local terminal "
                "category affinity before verification without changing the "
                "source-support or obligation-admission decision."
            )
        },
    )
    paper_evidence_category_affine_partial_materialization: bool = field(
        default=False,
        metadata={
            "help": (
                "Expose a bounded certificate of source-supported terminal "
                "candidates whose category admission is deferred. This never "
                "creates a structured witness or replaces the Reader answer."
            )
        },
    )
    paper_evidence_schema_compiled_source_certificate: bool = field(
        default=False,
        metadata={
            "help": (
                "Compile the active predicate family, endpoint direction and "
                "output type into a revalidated source-span certificate for "
                "indexed COMPLETE tuples that lack a verbatim relation surface."
            )
        },
    )
    paper_evidence_source_support_obligation_admission: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable the two-gate verifier that independently requires "
                "source support and compatibility with the active operator's "
                "predicate, direction, output role, and output type contract."
            )
        },
    )
    paper_evidence_enhanced_source_support: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable only the enhanced source-support rules. This switch "
                "is independent from obligation admission for factorial tests."
            )
        },
    )
    paper_evidence_obligation_admission: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable canonicalized operator-obligation admission. This "
                "switch is independent from enhanced source support."
            )
        },
    )
    paper_evidence_query_local_overlay_reuse: bool = field(
        default=True,
        metadata={
            "help": (
                "Allow a tuple recovered into the query-local overlay to "
                "propagate its output binding into later operators. Disable "
                "only for the isolated overlay-reuse ablation; recovered "
                "tuples are still materialized and verified."
            )
        },
    )
    paper_evidence_predicate_entailment_admission: bool = field(
        default=False,
        metadata={
            "help": (
                "Require a query-local recovered tuple's exact source span "
                "to entail its canonical predicate role and direction; "
                "endpoint co-occurrence alone is insufficient."
            )
        },
    )
    paper_evidence_binding_obligation_contract_validation: bool = field(
        default=False,
        metadata={
            "help": (
                "Validate root identity qualifiers, exact completion endpoint "
                "alignment, and downstream joinability before a matched tuple "
                "is allowed to create a child binding."
            )
        },
    )
    paper_evidence_identity_safe_composite_join: bool = field(
        default=True,
        metadata={
            "help": (
                "Carry verified canonical entity identity across joins and "
                "probe its title-backed fact partition before mention aliases."
            )
        }
    )
    paper_evidence_relation_matching_mode: str = field(
        default="canonical",
        metadata={
            "help": (
                "Relation predicate semantics for LOOKUP_JOIN: 'canonical' "
                "uses predicate family/subtype expansion; 'exact_raw' "
                "requires the normalized raw OpenIE relation to equal the "
                "operator's canonical relation. Intended only for a clean "
                "schema-semantic ablation."
            )
        },
    )
    paper_evidence_predicate_subtype_constraints: bool = field(
        default=True,
        metadata={
            "help": (
                "Enforce predicate-subtype constraints in schema-guided "
                "lookup, completion, obligation admission, and terminal "
                "witness projection. Disable only for the isolated V3.19 "
                "schema-subtype ablation."
            )
        },
    )
    paper_evidence_type_role_constraints: bool = field(
        default=True,
        metadata={
            "help": (
                "Enforce semantic output-type, role/direction, and binding "
                "contract checks while retaining the physical input/output "
                "slots needed to execute a join. Disable only for the "
                "isolated V3.19 type-role ablation."
            )
        },
    )
    paper_evidence_composite_index_access: bool = field(
        default=True,
        metadata={
            "help": (
                "Use entity-predicate composite indexes for identity-scoped "
                "LOOKUP_JOIN. Disabling this retains identity, predicate, "
                "direction, type, and provenance checks but scans the "
                "entity-only posting list."
            )
        },
    )
    paper_evidence_collision_aware_schema_identity: bool = field(
        default=False,
        metadata={
            "help": (
                "Use an ordered title identity key to prevent distinct "
                "entities from collapsing onto the same stopword-stripped "
                "physical join key during root binding."
            )
        },
    )
    paper_evidence_safe_title_family_root_binding: bool = field(
        default=True,
        metadata={
            "help": (
                "Allow a unique or category-resolved unqualified title "
                "family mentioned in the question to participate in root "
                "binding. Disable only for the clean short-title ablation."
            )
        },
    )
    paper_evidence_schema_plan_coverage_v7: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable the selective performer plan and lazy canonical-title "
                "terminal composite-index access path."
            )
        }
    )
    paper_evidence_general_plan_coverage_v8: bool = field(
        default=False,
        metadata={
            "help": (
                "Compile conservative one-relation selection/projection plans "
                "in addition to multi-hop joins, without dataset routing."
            )
        }
    )
    paper_evidence_schema_exact_nested_join_v9: bool = field(
        default=False,
        metadata={
            "help": (
                "Compile connected nested joins with exact predicate-subtype "
                "and title-family contracts; newly covered plans are shadow-only."
            )
        }
    )
    paper_evidence_v8_safe_materialization: bool = field(
        default=True,
        metadata={
            "help": (
                "Apply predicate-bound terminal determinacy to newly compiled "
                "V8/V9 plans: supported terminal sources may reach the Reader, "
                "while direct answer commits require a unique functional value."
            )
        }
    )
    paper_evidence_lineage_terminal_semijoin_v7_1: bool = field(
        default=False,
        metadata={
            "help": (
                "Qualify terminal fact joins with verified lineage attributes, "
                "verify sibling terminal candidates, and fall back on conflicts."
            )
        }
    )
    paper_evidence_provenance_grouped_terminal_v7_2: bool = field(
        default=False,
        metadata={
            "help": (
                "Reduce terminal candidates by identity strength and source "
                "provenance, while preserving quarantined evidence for the Reader."
            )
        }
    )
    paper_evidence_apply_structured_answer: bool = field(
        default=True,
        metadata={
            "help": (
                "Use a fully source-supported witness as the sole structured "
                "answer admission; otherwise retain the Reader answer."
            )
        }
    )
    paper_evidence_materialize_reader_evidence: bool = field(
        default=True,
        metadata={
            "help": (
                "Append executor-selected source documents to the Reader. "
                "Disable together with structured answer admission for a "
                "same-execution trace-only ablation."
            )
        }
    )
    paper_evidence_answer_domain_guard: bool = field(
        default=False,
        metadata={
            "help": (
                "Keep source-supported terminal tuples in the execution trace "
                "but prevent relation-value surfaces that violate the question's "
                "explicit answer domain from reaching the Reader or replacing its "
                "answer (for example, English for a which-country question)."
            )
        },
    )
    paper_evidence_incumbent_guard: bool = field(
        default=True,
        metadata={
            "help": (
                "Preserve the Reader answer unless a verified structured "
                "witness provides decisive counter-evidence."
            )
        }
    )
    paper_evidence_materialize_verified_partial: bool = field(
        default=True,
        metadata={
            "help": (
                "Append source documents for verified query-local COMPLETE "
                "facts even when the full logical chain is not yet closed."
            )
        }
    )
    paper_evidence_partial_max_docs: int = field(
        default=2,
        metadata={"help": "Maximum verified partial-completion documents sent to the Reader."}
    )
    enable_adaptive_lazy_query_executor: bool = field(
        default=False,
        metadata={
            "help": (
                "Choose retrieval-time schema/fact operators before invoking "
                "them, and re-optimize from observed cardinalities."
            )
        }
    )
    adaptive_lazy_query_executor_max_steps: int = field(
        default=2,
        metadata={"help": "Maximum physical evidence operators executed per query."}
    )
    adaptive_lazy_query_executor_compute_budget: float = field(
        default=1.0,
        metadata={"help": "Normalized runtime operator budget for lazy execution."}
    )
    adaptive_lazy_query_executor_min_objective: float = field(
        default=0.05,
        metadata={"help": "Minimum objective for executing a lazy physical operator."}
    )
    adaptive_lazy_query_executor_max_predicted_risk: float = field(
        default=0.55,
        metadata={"help": "Maximum uncalibrated phase-2 risk estimate for a lazy operator."}
    )
    adaptive_lazy_query_executor_deficit_coverage_threshold: float = field(
        default=0.72,
        metadata={"help": "Lexical coverage threshold used to initialize bridge deficit state."}
    )
    enable_adaptive_risk_bounded_optimizer: bool = field(
        default=False,
        metadata={
            "help": (
                "Enumerate dependency-aware evidence plans under cumulative "
                "risk, token, and compute bounds before each lazy step."
            )
        }
    )
    adaptive_risk_bounded_plan_horizon: int = field(
        default=2,
        metadata={"help": "Maximum number of future physical operators in a candidate plan."}
    )
    adaptive_risk_bounded_beam_width: int = field(
        default=8,
        metadata={"help": "Maximum risk-feasible plans retained at each search depth."}
    )
    adaptive_risk_bounded_uncertainty: float = field(
        default=0.04,
        metadata={"help": "Additive uncertainty margin used for each operator risk upper bound."}
    )
    adaptive_risk_bounded_max_plan_risk: float = field(
        default=0.55,
        metadata={"help": "Maximum composed risk upper bound for a complete evidence plan."}
    )
    enable_adaptive_operator_value_model: bool = field(
        default=False,
        metadata={
            "help": (
                "Calibrate lazy-operator utility with learned non-empty and "
                "residual-resolution probabilities."
            )
        }
    )
    adaptive_operator_value_model_path: str = field(
        default="artifacts/adaptive_operator_value/model.json",
        metadata={"help": "Output-aware adaptive-operator value model."}
    )
    adaptive_operator_value_min_operator_support: int = field(
        default=25,
        metadata={"help": "Minimum operator examples required for VOI calibration."}
    )
    adaptive_operator_value_min_feature_support: int = field(
        default=20,
        metadata={"help": "Minimum examples required for a contextual VOI feature."}
    )
    adaptive_operator_value_min_confidence: float = field(
        default=0.25,
        metadata={"help": "Minimum confidence required to apply VOI calibration."}
    )
    adaptive_operator_value_resolution_floor: float = field(
        default=0.50,
        metadata={"help": "Partial value assigned to a non-empty unresolved output."}
    )
    adaptive_operator_value_min_utility_multiplier: float = field(
        default=0.35,
        metadata={"help": "Lower bound on learned operator-utility calibration."}
    )
    adaptive_operator_value_max_utility_multiplier: float = field(
        default=1.50,
        metadata={"help": "Upper bound on learned operator-utility calibration."}
    )
    adaptive_operator_value_min_fallback_productivity: float = field(
        default=0.12,
        metadata={"help": "Minimum predicted productivity for a risk fallback."}
    )
    enable_adaptive_incumbent_preserving_admission: bool = field(
        default=False,
        metadata={
            "help": (
                "Bound the estimated harm of stopping a lazy plan and execute "
                "the remaining eager incumbent when the bound is violated."
            )
        }
    )
    adaptive_incumbent_admission_model_path: str = field(
        default="artifacts/adaptive_reader_risk/model.json",
        metadata={"help": "Calibrated reader-evidence stop-risk model."}
    )
    adaptive_incumbent_admission_shadow_only: bool = field(
        default=True,
        metadata={"help": "Trace risk-bound violations without forcing incumbent operators."}
    )
    adaptive_incumbent_admission_max_harm_probability: float = field(
        default=0.025,
        metadata={"help": "Maximum calibrated upper harm probability for stopping a plan."}
    )
    adaptive_incumbent_admission_max_reader_change_probability: float = field(
        default=0.12,
        metadata={"help": "Maximum predicted reader-evidence change probability for stopping."}
    )
    adaptive_incumbent_admission_min_model_support: int = field(
        default=25,
        metadata={"help": "Minimum calibration examples supporting a stop-risk feature."}
    )
    adaptive_incumbent_admission_min_confidence: float = field(
        default=0.25,
        metadata={"help": "Minimum confidence required to enforce a stop-risk bound."}
    )
    cost_based_admission_route_early_schema_through_optimizer: bool = field(
        default=False,
        metadata={
            "help": (
                "Route early schema bridge tuples through the same cost-based "
                "plan search instead of admitting them as protected passthrough rows."
            )
        }
    )
    enable_learned_cost_optimizer: bool = field(
        default=False,
        metadata={"help": "Use a learned weak-supervised cost model to estimate expected evidence admission gain."}
    )
    learned_cost_optimizer_model_path: str = field(
        default="artifacts/learned_cost_optimizer/model.json",
        metadata={"help": "Path to the learned cost optimizer JSON model."}
    )
    learned_cost_optimizer_min_expected_gain: float = field(
        default=0.012,
        metadata={"help": "Minimum learned expected utility before a candidate can bypass soft admission thresholds."}
    )
    learned_cost_optimizer_min_confidence: float = field(
        default=0.30,
        metadata={"help": "Minimum learned optimizer confidence before applying accept/reject decisions."}
    )
    learned_cost_optimizer_min_support: int = field(
        default=4,
        metadata={"help": "Smoothing support target used by the learned evidence cost model."}
    )
    learned_cost_optimizer_max_degrade: float = field(
        default=0.46,
        metadata={"help": "Maximum learned probability of degradation allowed for evidence admission."}
    )
    learned_cost_optimizer_veto_min_degrade_margin: float = field(
        default=0.0,
        metadata={
            "help": (
                "Minimum p(degrade)-p(improve) margin required before the "
                "learned_veto policy rejects otherwise budget-admissible evidence."
            )
        }
    )
    cost_based_admission_counterfactual_drop_rank: int = field(
        default=-1,
        metadata={"help": "Zero-based admitted-slate rank removed by counterfactual_drop."}
    )
    counterfactual_slate_min_harm_signals: int = field(
        default=2,
        metadata={"help": "Minimum independent feature families required for a counterfactual slate veto."}
    )
    counterfactual_slate_min_stable_datasets: int = field(
        default=3,
        metadata={"help": "Minimum training datasets with a consistent harmful marginal effect."}
    )
    cost_based_admission_candidate_pool_snapshot_dir: str = field(
        default="",
        metadata={"help": "Directory containing materialized candidate-pool snapshots."}
    )
    cost_based_admission_candidate_pool_snapshot_mode: str = field(
        default="off",
        metadata={"help": "Candidate-pool snapshot mode: off, record, or replay."}
    )
    cost_based_admission_candidate_pool_snapshot_strict: bool = field(
        default=True,
        metadata={"help": "Fail strict replay when a candidate-pool snapshot is unavailable."}
    )
    enable_contrastive_answer_arbitration: bool = field(
        default=False,
        metadata={"help": "Whether to run contrastive answer arbitration over typed evidence candidates."}
    )
    contrastive_answer_arbitration_apply_rewrite: bool = field(
        default=False,
        metadata={"help": "If false, contrastive answer arbitration runs in shadow mode."}
    )
    contrastive_answer_arbitration_min_confidence: float = field(
        default=0.90,
        metadata={"help": "Minimum confidence for numeric/date contrastive answer arbitration."}
    )
    contrastive_answer_arbitration_generic_min_confidence: float = field(
        default=0.94,
        metadata={"help": "Minimum confidence for generic contrastive answer arbitration."}
    )
    contrastive_answer_arbitration_min_contrast_gain: float = field(
        default=0.30,
        metadata={"help": "Minimum score gain over the current answer support before contrastive rewriting."}
    )
    contrastive_answer_arbitration_min_margin: float = field(
        default=0.14,
        metadata={"help": "Minimum margin over the second challenger."}
    )
    contrastive_answer_arbitration_min_question_overlap: float = field(
        default=0.08,
        metadata={"help": "Minimum question overlap for contrastive answer challengers."}
    )
    contrastive_answer_arbitration_max_docs: int = field(
        default=10,
        metadata={"help": "Maximum QA evidence documents scanned by contrastive answer arbitration."}
    )
    contrastive_answer_arbitration_max_answer_tokens: int = field(
        default=12,
        metadata={"help": "Maximum current answer token count before treating it as verbose."}
    )
    contrastive_answer_arbitration_min_provenance: int = field(
        default=2,
        metadata={"help": "Minimum provenance count for contrastive challenger candidates."}
    )
    contrastive_answer_arbitration_require_role_constraint: bool = field(
        default=True,
        metadata={"help": "Require relation/role cues before contrastive answer arbitration may rewrite."}
    )
    contrastive_answer_arbitration_require_supported_incumbent: bool = field(
        default=True,
        metadata={"help": "Only run contrastive arbitration when the current answer has local evidence support."}
    )
    contrastive_answer_arbitration_allow_alias_challengers: bool = field(
        default=False,
        metadata={"help": "Allow containment/alias challenger candidates in contrastive answer arbitration."}
    )
    enable_sentence_evidence: bool = field(
        default=False,
        metadata={"help": "Whether to build typed sentence-level evidence packs before QA."}
    )
    sentence_evidence_max_sentences_per_doc: int = field(
        default=2,
        metadata={"help": "Maximum high-signal sentences selected from each QA evidence document."}
    )
    sentence_evidence_max_total_sentences: int = field(
        default=16,
        metadata={"help": "Maximum high-signal sentences in the typed evidence pack."}
    )
    sentence_evidence_keep_full_top_docs: int = field(
        default=5,
        metadata={"help": "Number of leading QA documents kept in full after prepending typed sentence evidence."}
    )
    sentence_evidence_min_score: float = field(
        default=0.18,
        metadata={"help": "Minimum sentence score required for typed sentence evidence selection."}
    )
    sentence_evidence_chain_preserve: bool = field(
        default=True,
        metadata={"help": "Whether to preserve full evidence chains for bridge or expanded-evidence questions."}
    )
    sentence_evidence_chain_preserve_doc_threshold: int = field(
        default=5,
        metadata={"help": "Preserve the full evidence chain when the QA evidence pool exceeds this size."}
    )
    enable_entity_state_evidence: bool = field(
        default=False,
        metadata={"help": "Whether to prepend an entity-state consistency control document before QA."}
    )
    entity_state_evidence_bridge_only: bool = field(
        default=True,
        metadata={"help": "Only apply entity-state control to bridge, expanded, or planned evidence states."}
    )
    entity_state_evidence_conflict_only: bool = field(
        default=True,
        metadata={"help": "Only insert entity-state control when conflicting answer-bearing evidence is detected."}
    )
    entity_state_evidence_max_anchor_docs: int = field(
        default=5,
        metadata={"help": "Number of leading QA evidence docs used to infer entity-state anchors."}
    )
    entity_state_evidence_max_anchors: int = field(
        default=6,
        metadata={"help": "Maximum resolved state anchors shown to the QA reader."}
    )
    entity_state_evidence_max_support: int = field(
        default=4,
        metadata={"help": "Maximum state-consistent evidence sentences shown in the control document."}
    )
    entity_state_evidence_max_distractors: int = field(
        default=3,
        metadata={"help": "Maximum potential distractor sentences shown in the control document."}
    )
    entity_state_evidence_max_sentences_per_doc: int = field(
        default=2,
        metadata={"help": "Maximum support/distractor sentences collected from one document."}
    )
    entity_state_evidence_min_source_overlap: float = field(
        default=0.08,
        metadata={"help": "Minimum question/planned-query overlap for a document to seed state anchors."}
    )
    enable_slot_projected_provenance_view: bool = field(
        default=False,
        metadata={"help": "Whether to insert a slot-projected provenance view before QA."}
    )
    slot_projected_provenance_max_docs: int = field(
        default=8,
        metadata={"help": "Maximum QA evidence documents scanned by the slot-projected provenance view."}
    )
    slot_projected_provenance_max_rows: int = field(
        default=3,
        metadata={"help": "Maximum answer-slot rows shown in the slot-projected provenance view."}
    )
    slot_projected_provenance_min_confidence: float = field(
        default=0.70,
        metadata={"help": "Minimum candidate confidence required to insert the slot-projected provenance view."}
    )
    slot_projected_provenance_min_margin: float = field(
        default=0.06,
        metadata={"help": "Minimum confidence margin required over the second projected slot."}
    )
    slot_projected_provenance_min_question_overlap: float = field(
        default=0.08,
        metadata={"help": "Minimum question overlap for projected answer-slot evidence."}
    )
    slot_projected_provenance_min_provenance: int = field(
        default=1,
        metadata={"help": "Minimum provenance count required for a projected answer-slot row."}
    )
    slot_projected_provenance_insert_after: int = field(
        default=2,
        metadata={"help": "Insert the slot-projected provenance view after this many leading QA documents."}
    )
    slot_projected_provenance_bridge_or_deficit_only: bool = field(
        default=True,
        metadata={"help": "Only insert the slot-projected provenance view for bridge or evidence-deficit states."}
    )
    slot_projected_provenance_block_simple_factoid_without_deficit: bool = field(
        default=True,
        metadata={"help": "Block slot projection on simple factoid questions when no evidence deficit is detected."}
    )
    slot_projected_provenance_block_long_context: bool = field(
        default=True,
        metadata={"help": "Block slot projection for long-context evidence pools."}
    )
    slot_projected_provenance_long_context_avg_chars: int = field(
        default=1400,
        metadata={"help": "Average source-document length threshold for long-context slot-projection blocking."}
    )
    slot_projected_provenance_require_role_constraint: bool = field(
        default=True,
        metadata={"help": "Require relation/role match before slot projection may expose candidates."}
    )
    enable_candidate_answer_selector: bool = field(
        default=False,
        metadata={"help": "Whether to insert a contrastive candidate-answer evidence pack before QA."}
    )
    candidate_answer_selector_numeric_temporal_only: bool = field(
        default=True,
        metadata={"help": "Only apply candidate answer selector to numeric/date answer types."}
    )
    candidate_answer_selector_min_confidence: float = field(
        default=0.76,
        metadata={"help": "Minimum confidence for inserting candidate answer selector evidence."}
    )
    candidate_answer_selector_min_margin: float = field(
        default=0.08,
        metadata={"help": "Minimum confidence margin over the second candidate."}
    )
    candidate_answer_selector_min_question_overlap: float = field(
        default=0.08,
        metadata={"help": "Minimum question overlap for candidate supporting sentence."}
    )
    candidate_answer_selector_max_docs: int = field(
        default=8,
        metadata={"help": "Maximum QA evidence documents scanned by candidate answer selector."}
    )
    candidate_answer_selector_max_candidates: int = field(
        default=3,
        metadata={"help": "Maximum candidates shown in candidate answer selector evidence."}
    )
    candidate_answer_selector_insert_after: int = field(
        default=2,
        metadata={"help": "Insert candidate answer selector evidence after this many leading QA documents."}
    )
    enable_multigranular_evidence_memory: bool = field(
        default=False,
        metadata={"help": "Whether to insert a compact multi-granular evidence memory document before QA."}
    )
    multigranular_memory_max_docs: int = field(
        default=8,
        metadata={"help": "Maximum QA evidence documents used to build multi-granular memory."}
    )
    multigranular_memory_max_sentences_per_doc: int = field(
        default=2,
        metadata={"help": "Maximum selected memory sentences per source document."}
    )
    multigranular_memory_max_total_sentences: int = field(
        default=14,
        metadata={"help": "Maximum selected memory sentences across all source documents."}
    )
    multigranular_memory_keep_full_top_docs: int = field(
        default=2,
        metadata={"help": "Insert multi-granular memory after this many full leading QA documents."}
    )
    multigranular_memory_min_sentence_score: float = field(
        default=0.12,
        metadata={"help": "Minimum sentence score for multi-granular evidence memory."}
    )
    multigranular_memory_max_chars: int = field(
        default=3600,
        metadata={"help": "Maximum characters in the synthetic multi-granular memory document."}
    )
    multigranular_memory_long_answer_only: bool = field(
        default=True,
        metadata={"help": "Only apply multi-granular memory to long-answer or long-context query shapes."}
    )
    multigranular_memory_min_avg_doc_chars: int = field(
        default=1400,
        metadata={"help": "Average source document length threshold for long-context memory triggering."}
    )
    multigranular_memory_controller_mode: Literal['shape', 'selective', 'off'] = field(
        default='selective',
        metadata={"help": "Controller for multi-granular memory insertion: legacy shape gate, selective diagnostics gate, or off."}
    )
    multigranular_memory_controller_min_avg_doc_chars: int = field(
        default=500,
        metadata={"help": "Average evidence document length threshold used by the selective multi-granular memory controller."}
    )
    multigranular_memory_skip_no_useful_extra_candidates: bool = field(
        default=True,
        metadata={"help": "Skip multi-granular memory when adaptive evidence found no useful extra candidates."}
    )


    save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."}
    )





    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa']] = field(
        default=None,
        metadata={"help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."}
    )

    graph_type: Literal[
        'dpr_only',
        'entity',
        'passage_entity', 'relation_aware_passage_entity',
        'passage_entity_relation',
        'facts_and_sim_passage_node_unidirectional',
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "Type of graph to use in the experiment."}
    )
    corpus_len: Optional[int] = field(
        default=None,
        metadata={"help": "Length of the corpus to use."}
    )


    def __post_init__(self):
        if self.save_dir is None:
            if self.dataset is None: self.save_dir = 'outputs'
            else: self.save_dir = os.path.join('outputs', self.dataset)
        logger.debug(f"Initializing the highest level of save_dir to be {self.save_dir}")
