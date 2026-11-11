import json
import os
import logging
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import numpy as np
import importlib
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from igraph import Graph
import igraph as ig
from transformers import HfArgumentParser
import numpy as np
import re
import time

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput
from .utils.answer_utils import extract_answer_from_response, get_answer_style_instruction, postprocess_answer
from .utils.answer_slot_support_completion_utils import append_answer_slot_support_completion
from .utils.anchored_typed_candidate_table_utils import build_anchored_typed_candidate_table
from .utils.answer_verifier_utils import (
    verify_answer_with_evidence,
    verify_answer_with_plan_consistency,
)
from .utils.canonicalize_utils import (
    canonicalize_answer,
    canonicalize_schema_touched_answer_surface,
)
from .utils.candidate_answer_selector_utils import build_candidate_answer_contrastive_selector
from .utils.evidence_consistent_reader_utils import refine_answer_with_evidence_consistent_reader
from .utils.evidence_object_arbitration_utils import arbitrate_answer_with_evidence_objects
from .utils.evidence_answer_arbiter_utils import arbitrate_answer_with_evidence_grounding
from .utils.typed_evidence_table_utils import arbitrate_answer_with_typed_evidence_table
from .utils.contrastive_answer_arbitration_utils import arbitrate_answer_with_contrastive_candidates
from .utils.provenance_answer_lattice_utils import arbitrate_answer_with_provenance_lattice
from .utils.provenance_typed_candidate_store_utils import (
    arbitrate_answer_with_provenance_typed_candidate_store,
)
from .utils.provenance_answer_slot_store_utils import (
    arbitrate_answer_with_provenance_answer_slot_store,
)
from .utils.evidence_grounded_answer_extractor_utils import (
    arbitrate_answer_with_evidence_grounded_extractor,
)
from .utils.query_answer_contract_utils import (
    arbitrate_answer_with_query_answer_contract,
)
from .utils.answer_slot_binding_optimizer_utils import optimize_answer_slot_binding
from .utils.answer_surface_lattice_optimizer_utils import optimize_answer_surface_lattice
from .utils.adaptive_query_executor_utils import (
    build_logical_plan_execution_policy,
    build_lazy_operator_spec,
    estimate_index_scan_rows,
    execute_adaptive_query_plan,
    lexical_query_coverage,
)
from .utils.adaptive_reader_risk_utils import (
    assess_adaptive_stop_risk,
    load_adaptive_reader_risk_model,
)
from .utils.adaptive_operator_value_utils import (
    assess_adaptive_operator_value,
    load_adaptive_operator_value_model,
)
from .utils.cost_gated_reread_utils import cost_gated_constrained_reread
from .utils.schema_critical_executor_utils import (
    probe_terminal_schema_candidates,
    protect_provenance_fact_view_surface,
    select_schema_critical_evidence,
)
from .utils.provenance_bound_answer_materializer_utils import (
    finalize_deterministic_terminal_projection,
    finalize_provenance_bound_answer_materialization,
    prepare_provenance_bound_answer_materialization,
    select_authoritative_logical_query_plan,
)
from .utils.evidence_query_execution_utils import (
    ExecutionBudget,
    RelationIndexEvidenceBackend,
    assess_category_affine_supported_partial,
    assess_terminal_answer_domain,
    assess_terminal_answer_determinacy,
    execute_evidence_query,
    promote_plan_bound_evidence_docs,
    select_paper_answer,
    update_answer_phase_status,
)
from .utils.pre_reader_trace_utils import (
    pre_reader_trace_path,
    write_pre_reader_trace,
)
from .utils.atomic_json_utils import atomic_write_json, load_locked_json
from .utils.evidence_utils import (
    apply_dual_channel_reader_projection,
    apply_dual_channel_retrieval_projection,
    apply_bridge_contrastive_evidence_guard,
    assess_dual_channel_projection_trigger,
    has_bridge_cues,
    select_adaptive_evidence_passages,
)
from .utils.evidence_admission_utils import apply_cost_based_evidence_admission
from .utils.schema_evidence_set_utils import (
    assess_complete_path_projection_closure,
    select_schema_aware_evidence_set,
)
from .utils.entity_centric_fallback_utils import append_entity_centric_deep_evidence
from .utils.entity_bound_join_utils import entity_bound_join_key_match
from .utils.entity_predicate_fact_join_utils import (
    build_entity_predicate_fact_join_candidates,
    remove_entity_predicate_passages,
)
from .utils.graph_repair_utils import graph_repair_retrieve, graph_repair_trigger_reason
from .utils.fact_source_scan_utils import promote_fact_source_docs
from .utils.multigranular_memory_utils import build_multigranular_evidence_memory
from .utils.path_rerank_utils import (
    build_chunk_path_info,
    constrained_chain_rerank_docs,
    path_aware_rerank_docs,
    select_path_aware_extra_doc_ids,
)
from .utils.relation_fact_join_utils import (
    append_relation_fact_join_evidence,
    build_relation_fact_join_index,
    canonical_schema_predicate_families,
    compile_stage_aware_fact_join_plan,
    execute_instance_bound_fact_join_plan,
    typed_schema_signatures_v2,
    estimate_relation_fact_join_access,
    select_relation_fact_join_evidence,
)
from .utils.evidence_query_compiler_utils import compile_evidence_query_plan
from .utils.evidence_query_compiler_semantic_v3_utils import (
    compile_rule_first_semantic_v3_query_plan,
)
from .utils.evidence_query_compiler_semantic_v3_1_utils import (
    compile_rule_first_semantic_v3_1_query_plan,
)
from .utils.evidence_query_compiler_semantic_v3_12_runtime_utils import (
    compile_rule_first_semantic_v3_12_query_plan,
)
from .utils.evidence_query_compiler_semantic_v3_13_runtime_utils import (
    compile_rule_first_semantic_v3_13_query_plan,
)
from .utils.evidence_query_compiler_semantic_v3_14_runtime_utils import (
    compile_rule_first_semantic_v3_14_query_plan,
)
from .utils.evidence_query_compiler_semantic_v3_15_runtime_utils import (
    compile_rule_first_semantic_v3_15_query_plan,
)
from .utils.selective_support_join_utils import (
    append_selective_support_join_evidence,
    select_selective_support_join_evidence,
)
from .utils.typed_fact_join_shadow_utils import build_typed_fact_join_shadow_optimizer
from .utils.planned_retrieval_utils import (
    EvidenceMerger,
    PlannedRetrievalCostController,
    RetrievalPlanner,
    QuestionRouter,
    SourceRetriever,
    SourceVerifier,
    build_planned_retrieval_decision_frontier,
    build_planned_retrieval_evidence_signature,
    build_residual_deficit_queries,
    route_context_state_intervention_gate,
    route_pairwise_utility_source_policy,
    route_risk_aware_merge_policy,
)
from .utils.planned_retrieval_state_utils import (
    annotate_planned_retrieval_terminal_state,
    build_terminal_materialization_trace,
)
from .utils.sentence_evidence_utils import build_typed_sentence_evidence
from .utils.slot_projected_provenance_utils import build_slot_projected_provenance_view
from .utils.slot_separated_fact_join_utils import build_slot_separated_fact_join_view
from .utils.structured_evidence_chain_utils import build_structured_evidence_chain
from .utils.support_set_completion_utils import append_support_set_completion_evidence
from .utils.support_rerank_utils import support_aware_rerank_docs
from .utils.iterative_retrieval_utils import (
    assess_evidence_sufficiency,
    build_followup_query,
    merge_iterative_retrieval_results,
)
from .utils.embed_utils import retrieve_knn
from .utils.entity_state_evidence_utils import build_entity_state_evidence_control
from .utils.typing import Triple
from .utils.config_utils import BaseConfig
from .utils.ordered_prefetch_utils import OrderedBoundedPrefetch

logger = logging.getLogger(__name__)


def _collection_size(values: Any) -> int:
    """Return a collection size without truth-testing NumPy arrays."""
    if values is None:
        return 0
    return len(values)


def _is_transient_api_failure(exc: Exception) -> bool:
    transient_names = {
        "APITimeoutError",
        "APIConnectionError",
        "APIError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "InternalServerError",
        "RateLimitError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "RetryError",
        "Timeout",
        "TimeoutException",
    }
    transient_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
    current = exc
    seen = set()
    while isinstance(current, Exception) and id(current) not in seen:
        seen.add(id(current))
        for cls in current.__class__.mro():
            if cls.__name__ in transient_names:
                return True
        module_name = getattr(current.__class__, "__module__", "").lower()
        if any(name in module_name for name in ("httpx", "httpcore", "tenacity")):
            return True
        status_code = getattr(current, "status_code", None)
        response = getattr(current, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        if status_code in transient_statuses:
            return True
        last_attempt = getattr(current, "last_attempt", None)
        if last_attempt is not None:
            try:
                attempted_error = last_attempt.exception()
            except Exception:
                attempted_error = None
            if isinstance(attempted_error, Exception) and attempted_error is not current:
                current = attempted_error
                continue
        nested = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if not isinstance(nested, Exception) or nested is current:
            break
        current = nested
    return False


def _root_error(exc: Exception) -> Exception:
    current = exc
    seen = set()
    while isinstance(current, Exception) and id(current) not in seen:
        seen.add(id(current))
        last_attempt = getattr(current, "last_attempt", None)
        if last_attempt is not None:
            try:
                attempted_error = last_attempt.exception()
            except Exception:
                attempted_error = None
            if isinstance(attempted_error, Exception) and attempted_error is not current:
                current = attempted_error
                continue
        nested = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if not isinstance(nested, Exception) or nested is current:
            break
        current = nested
    return current


def _record_planned_retrieval_api_fallback(
    planned_retrieval_info: Dict[str, Any],
    round_info: Dict[str, Any],
    *,
    stage: str,
    exc: Exception,
) -> None:
    root_error = _root_error(exc)
    failure = {
        "stage": stage,
        "error_type": exc.__class__.__name__,
        "error": str(exc)[:500],
        "root_error_type": root_error.__class__.__name__,
        "root_error": str(root_error)[:500],
    }
    planned_retrieval_info.update({
        "reason": "transient_api_fallback",
        "transient_failure": True,
        "fallback_to_current_evidence": True,
        "transient_failure_stage": stage,
        "transient_failure_error_type": failure["root_error_type"],
    })
    planned_retrieval_info.setdefault("transient_failures", []).append(failure)
    round_info.update({
        "reason": "transient_api_fallback",
        "transient_failure": failure,
        "fallback_to_current_evidence": True,
    })
    planned_retrieval_info.setdefault("rounds", []).append(round_info)
    logger.warning(
        "Planned retrieval %s failed transiently; preserving current evidence. "
        "Error=%s: %s; root=%s: %s",
        stage,
        exc.__class__.__name__,
        str(exc)[:300],
        root_error.__class__.__name__,
        str(root_error)[:300],
    )


class S2RAG:

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific S2RAG instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (Union[OpenIE, VLLMOfflineOpenIE]): The Open Information Extraction module
                configured in either online or offline mode based on the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed LLM model, can be inserted directly as well as through configuration file.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config


        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"S2RAG init with config:\n  {_print_config}\n")


        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        if self.global_config.openie_mode != "online":
            raise ValueError("S2RAG supports online relation extraction only")
        self.openie = OpenIE(llm_model=self.llm_model)

        self.graph = self.initialize_graph()

        self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
            embedding_model_name=self.global_config.embedding_model_name
        )(
            global_config=self.global_config,
            embedding_model_name=self.global_config.embedding_model_name,
        )
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact')

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(self.global_config.save_dir,f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        self.rerank_filter = DSPyFilter(self)

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.ppr_call_count = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0

        self.ent_node_to_chunk_ids = None

    def _uses_dense_retrieval_backend(self) -> bool:
        return str(
            getattr(self.global_config, "retrieval_backend", "graph_ppr")
        ).strip().lower() == "nv_embed_dense"

    def _compile_paper_evidence_plan(
        self,
        query: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        compile_started = time.perf_counter()
        compiler_mode = str(getattr(
            self.global_config,
            "paper_evidence_query_compiler_mode",
            "rule",
        ))
        compiler_kwargs = {
            "llm_infer": self.llm_model.infer,
            "model_name": str(getattr(self.global_config, "llm_name", "")),
            "rule_options": {
                "enable_bound_completion": True,
                "enable_role_directed_join": True,
                "enable_schema_plan_coverage_v7": bool(getattr(
                    self.global_config,
                    "paper_evidence_schema_plan_coverage_v7",
                    False,
                )),
                "enable_general_plan_coverage_v8": bool(getattr(
                    self.global_config,
                    "paper_evidence_general_plan_coverage_v8",
                    False,
                )),
                "enable_schema_exact_nested_join_v9": bool(getattr(
                    self.global_config,
                    "paper_evidence_schema_exact_nested_join_v9",
                    False,
                )),
            },
        }
        compiler_max_tokens = int(getattr(
            self.global_config,
            "paper_evidence_query_compiler_max_completion_tokens",
            700,
        ))
        if compiler_mode == "rule_first_semantic_v3":
            logical_plan, query_compiler_trace = (
                compile_rule_first_semantic_v3_query_plan(
                    query,
                    generation_max_completion_tokens=compiler_max_tokens,
                    review_max_completion_tokens=compiler_max_tokens,
                    **compiler_kwargs,
                )
            )
        elif compiler_mode == "rule_first_semantic_v3_1":
            logical_plan, query_compiler_trace = (
                compile_rule_first_semantic_v3_1_query_plan(
                    query,
                    generation_max_completion_tokens=compiler_max_tokens,
                    review_max_completion_tokens=max(compiler_max_tokens, 900),
                    **compiler_kwargs,
                )
            )
        elif compiler_mode == "rule_first_semantic_v3_12":
            logical_plan, query_compiler_trace = (
                compile_rule_first_semantic_v3_12_query_plan(
                    query,
                    generation_max_completion_tokens=compiler_max_tokens,
                    **compiler_kwargs,
                )
            )
        elif compiler_mode == "rule_first_semantic_v3_13":
            logical_plan, query_compiler_trace = (
                compile_rule_first_semantic_v3_13_query_plan(
                    query,
                    generation_max_completion_tokens=compiler_max_tokens,
                    **compiler_kwargs,
                )
            )
        elif compiler_mode == "rule_first_semantic_v3_14":
            logical_plan, query_compiler_trace = (
                compile_rule_first_semantic_v3_14_query_plan(
                    query,
                    generation_max_completion_tokens=compiler_max_tokens,
                    **compiler_kwargs,
                )
            )
        elif compiler_mode == "rule_first_semantic_v3_15":
            logical_plan, query_compiler_trace = (
                compile_rule_first_semantic_v3_15_query_plan(
                    query,
                    generation_max_completion_tokens=compiler_max_tokens,
                    **compiler_kwargs,
                )
            )
        else:
            logical_plan, query_compiler_trace = compile_evidence_query_plan(
                query,
                mode=compiler_mode,
                max_completion_tokens=compiler_max_tokens,
                **compiler_kwargs,
            )
        compile_wall_ms = max(
            0.0,
            (time.perf_counter() - compile_started) * 1000.0,
        )
        return logical_plan, query_compiler_trace, compile_wall_ms

    def _prefetch_retrieval_llm_stages(self, query: str) -> Dict[str, Any]:
        rerank_started = time.perf_counter()
        query_fact_scores = self.get_fact_scores(query)
        top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(
            query,
            query_fact_scores,
        )
        result = {
            "query_fact_scores": query_fact_scores,
            "top_k_fact_indices": top_k_fact_indices,
            "top_k_facts": top_k_facts,
            "rerank_log": rerank_log,
            "rerank_seconds": max(0.0, time.perf_counter() - rerank_started),
            "compiled_plan": None,
        }
        if bool(getattr(
            self.global_config,
            "enable_paper_evidence_query_executor",
            False,
        )):
            result["compiled_plan"] = self._compile_paper_evidence_plan(query)
        return result

    def _retrieval_backend_diagnostics(self) -> Dict[str, Any]:
        dense_only = self._uses_dense_retrieval_backend()
        return {
            "name": "nv_embed_dense" if dense_only else "graph_ppr",
            "embedding_model": str(self.global_config.embedding_model_name),
            "dense_similarity": "normalized_dot_product",
            "ppr_enabled": not dense_only,
            "ppr_call_count": int(self.ppr_call_count),
            "graph_used_for_passage_ranking": not dense_only,
            "openie_role": (
                "schema_fact_table_only" if dense_only else "schema_fact_table_and_graph"
            ),
        }

    def _graph_cache_config_signature(self) -> Dict[str, Any]:
        """Return the graph-construction config that must match a cached graph."""
        keys = [
            "graph_type",
            "is_directed_graph",
            "synonymy_edge_topk",
            "synonymy_edge_sim_threshold",
            "enable_schema_memory_graph",
            "schema_memory_edge_weight",
            "schema_memory_max_passages_per_schema",
            "schema_memory_neighbors_per_passage",
            "schema_memory_min_relation_tokens",
        ]
        return {
            "signature_version": 1,
            **{key: getattr(self.global_config, key, None) for key in keys},
        }

    def _encoded_graph_cache_config_signature(self) -> str:
        return json.dumps(
            self._graph_cache_config_signature(),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _cached_graph_config_signature(self, graph: ig.Graph) -> Optional[str]:
        try:
            return graph["s2rag_graph_cache_signature"]
        except Exception:
            return None

    def _graph_cache_config_matches(self, graph: ig.Graph) -> bool:
        expected = self._encoded_graph_cache_config_signature()
        cached = self._cached_graph_config_signature(graph)
        if not cached:
            logger.warning(
                "Graph cache %s has no config signature; rebuilding to avoid stale ablation state.",
                self._graph_pickle_filename,
            )
            return False
        if cached != expected:
            logger.warning(
                "Graph cache config mismatch for %s; rebuilding. cached=%s expected=%s",
                self._graph_pickle_filename,
                cached,
                expected,
            )
            return False
        return True

    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a Pickle file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        default_graph_path = os.path.join(self.working_dir, "graph.pickle")
        graph_path_override = os.environ.get(
            "S2RAG_GRAPH_PICKLE_PATH", ""
        ).strip()
        self._graph_pickle_filename = os.path.abspath(
            os.path.expanduser(graph_path_override)
        ) if graph_path_override else default_graph_path
        if graph_path_override:
            os.makedirs(
                os.path.dirname(self._graph_pickle_filename), exist_ok=True
            )
            logger.info(
                "Using experiment-local graph cache: %s",
                self._graph_pickle_filename,
            )

        if self._uses_dense_retrieval_backend():
            logger.info(
                "Retrieval backend is nv_embed_dense; graph cache loading is bypassed."
            )
            return ig.Graph(directed=self.global_config.is_directed_graph)

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)
                if (
                    preloaded_graph is not None
                    and getattr(self.global_config, "graph_cache_validate_config", True)
                    and not self._graph_cache_config_matches(preloaded_graph)
                ):
                    preloaded_graph = None

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def pre_openie(self,  docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')

    def index(self, docs: List[str]):
        """
        Indexes the given documents using S2RAG, which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """

        index_started = time.perf_counter()
        index_profile = {
            "document_count": len(docs),
            "force_index_from_scratch": bool(self.global_config.force_index_from_scratch),
            "force_openie_from_scratch": bool(self.global_config.force_openie_from_scratch),
            "schema_memory_graph_enabled": bool(
                getattr(self.global_config, "enable_schema_memory_graph", False)
            ),
            "retrieval_backend": str(
                getattr(self.global_config, "retrieval_backend", "graph_ppr")
            ),
        }

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        stage_started = time.perf_counter()
        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()
        index_profile["chunk_embedding_seconds"] = round(
            time.perf_counter() - stage_started,
            6,
        )

        stage_started = time.perf_counter()
        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)
        index_profile["openie_seconds"] = round(time.perf_counter() - stage_started, 6)
        index_profile["openie_new_document_count"] = len(new_openie_rows)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict), f"len(chunk_to_rows): {len(chunk_to_rows)}, len(ner_results_dict): {len(ner_results_dict)}, len(triple_results_dict): {len(triple_results_dict)}"


        chunk_ids = list(chunk_to_rows.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.info(f"Encoding Entities")
        stage_started = time.perf_counter()
        self.entity_embedding_store.insert_strings(entity_nodes)
        index_profile["entity_embedding_seconds"] = round(
            time.perf_counter() - stage_started,
            6,
        )

        logger.info(f"Encoding Facts")
        stage_started = time.perf_counter()
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])
        index_profile["fact_embedding_seconds"] = round(
            time.perf_counter() - stage_started,
            6,
        )

        logger.info(f"Constructing Graph")
        stage_started = time.perf_counter()

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        if self._uses_dense_retrieval_backend():
            self._rebuild_fact_lookup_metadata(chunk_ids, chunk_triples)
            num_new_chunks = 0
            logger.info(
                "Skipping entity graph, synonym edges, and graph serialization for "
                "nv_embed_dense retrieval."
            )
        else:
            self.add_fact_edges(chunk_ids, chunk_triples)
            num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

            if num_new_chunks > 0:
                logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
                self.add_schema_memory_edges(chunk_ids, chunk_triples)
                self.add_synonymy_edges()

                self.augment_graph()
                self.save_igraph()

        index_profile.update({
            "graph_construction_seconds": round(time.perf_counter() - stage_started, 6),
            "total_seconds": round(time.perf_counter() - index_started, 6),
            "chunk_count": len(chunk_ids),
            "entity_count": len(entity_nodes),
            "fact_count": len(facts),
            "new_graph_chunk_count": int(num_new_chunks),
            "graph_construction_skipped": self._uses_dense_retrieval_backend(),
            "graph_node_count": int(self.graph.vcount()),
            "graph_edge_count": int(self.graph.ecount()),
            "schema_memory_schema_count": int(
                (getattr(self, "schema_memory_graph_stats", {}) or {}).get(
                    "num_schema_memory_schemas",
                    0,
                )
            ),
            "schema_memory_edge_count": int(
                (getattr(self, "schema_memory_graph_stats", {}) or {}).get(
                    "num_schema_memory_edges",
                    0,
                )
            ),
        })
        logger.info(
            "INDEX_PROFILE_JSON %s",
            json.dumps(index_profile, ensure_ascii=True, sort_keys=True),
        )

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes the given documents from all data structures within the S2RAG class.
        Note that triples and entities which are indexed from chunks that are not being removed will not be removed.

        Parameters:
            docs : List[str]
                A list of documents to be deleted.
        """


        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]


        chunk_ids_to_delete = set(
            [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])


        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)


        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[text_processing(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)

        triple_ids_to_delete = set([self.fact_embedding_store.text_to_hash_id[str(triple)] for triple in processed_true_triples_to_delete])


        ent_ids_to_delete = [self.entity_embedding_store.text_to_hash_id[ent] for ent in entities_to_delete]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")

        self.save_openie_results(all_openie_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)


        self.graph.delete_vertices(list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False

    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None,
                 sample_keys: List[str] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using either S2RAG PPR or a strict dense-passage backend.
        In ``nv_embed_dense`` mode, facts are available only to S2RAG's schema
        and join operators; passage ranking never uses graph propagation or PPR.

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()
        if sample_keys is not None and len(sample_keys) != len(queries):
            raise ValueError("sample_keys must align one-to-one with queries")

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_llm_workers = max(
            1,
            int(getattr(self.global_config, "retrieval_llm_workers", 1)),
        )
        retrieval_llm_prefetch = None
        if retrieval_llm_workers > 1 and queries:
            retrieval_llm_prefetch = OrderedBoundedPrefetch(
                self._prefetch_retrieval_llm_stages,
                queries,
                max_workers=retrieval_llm_workers,
                prefetch_window=retrieval_llm_workers * 2,
            )
            logger.info(
                "Retrieval-time LLM stages use %d workers with a %d-query window; "
                "graph execution remains ordered.",
                retrieval_llm_workers,
                retrieval_llm_workers * 2,
            )

        retrieval_results = []
        typed_fact_join_shadow_counts = Counter()
        selective_support_join_counts = Counter()
        adaptive_query_execution_counts = Counter()
        fact_source_scan_counts = Counter()

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            query_retrieval_started = time.perf_counter()
            dense_only_backend = self._uses_dense_retrieval_backend()
            prefetched_compilation = None
            remote_prefetch_wait_seconds = 0.0
            remote_stage_work_seconds = 0.0
            if retrieval_llm_prefetch is not None:
                remote_prefetch_wait_started = time.perf_counter()
                prefetched_stages = next(retrieval_llm_prefetch)
                remote_prefetch_wait_seconds = max(
                    0.0,
                    time.perf_counter() - remote_prefetch_wait_started,
                )
                query_fact_scores = prefetched_stages["query_fact_scores"]
                top_k_fact_indices = prefetched_stages["top_k_fact_indices"]
                top_k_facts = prefetched_stages["top_k_facts"]
                rerank_log = prefetched_stages["rerank_log"]
                prefetched_compilation = prefetched_stages["compiled_plan"]
                self.rerank_time += float(prefetched_stages["rerank_seconds"])
                remote_stage_work_seconds = float(
                    prefetched_stages["rerank_seconds"]
                )
                if prefetched_compilation is not None:
                    remote_stage_work_seconds += float(
                        prefetched_compilation[2]
                    ) / 1000.0
            else:
                rerank_start = time.time()
                query_fact_scores = self.get_fact_scores(query)
                top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(
                    query,
                    query_fact_scores,
                )
                rerank_end = time.time()
                self.rerank_time += rerank_end - rerank_start
            fact_rerank_info = {
                "mode": "schema_fact_candidates" if dense_only_backend else "graph",
                "fallback_to_dense": False,
                "candidate_fact_count": len((rerank_log or {}).get("facts_before_rerank", [])),
                "reranked_fact_count": len((rerank_log or {}).get("facts_after_rerank", top_k_facts)),
                "top_fact_count": len(top_k_facts),
                "error": (rerank_log or {}).get("error"),
            }

            if dense_only_backend:
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                fact_rerank_info["mode"] = "nv_embed_dense"
                fact_rerank_info["facts_used_for_passage_ranking"] = False
            elif len(top_k_facts) == 0:
                logger.info('No facts found after reranking, return DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                fact_rerank_info["mode"] = "dense_fallback"
                fact_rerank_info["fallback_to_dense"] = True
            else:
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(query=query,
                                                                                         link_top_k=self.global_config.linking_top_k,
                                                                                         query_fact_scores=query_fact_scores,
                                                                                         top_k_facts=top_k_facts,
                                                                                         top_k_fact_indices=top_k_fact_indices,
                                                                                         passage_node_weight=self.global_config.passage_node_weight)
            graph_repair_trigger = graph_repair_trigger_reason(
                fact_rerank_info=fact_rerank_info,
                min_top_facts=getattr(self.global_config, "graph_repair_min_top_facts", 1),
                min_candidate_facts=getattr(self.global_config, "graph_repair_min_candidate_facts", 2),
                trigger_on_dense_fallback=getattr(self.global_config, "graph_repair_trigger_on_dense_fallback", True),
            )
            graph_repair_enabled = bool(
                getattr(self.global_config, "enable_graph_repair_retrieval", False)
            ) and not dense_only_backend
            graph_repair_info = {
                "enabled": graph_repair_enabled,
                "applied": False,
                "reason": (
                    "incompatible_with_nv_embed_dense"
                    if dense_only_backend
                    else "disabled"
                ),
                "trigger_reason": graph_repair_trigger,
            }
            if graph_repair_enabled:
                if graph_repair_trigger == "graph_fact_signal_sufficient":
                    graph_repair_info["reason"] = graph_repair_trigger
                else:
                    dense_top_k = max(1, int(getattr(self.global_config, "graph_repair_dense_top_k", 80)))
                    if fact_rerank_info.get("fallback_to_dense"):
                        repair_dense_doc_ids, repair_dense_doc_scores = sorted_doc_ids, sorted_doc_scores
                    else:
                        repair_dense_doc_ids, repair_dense_doc_scores = self.dense_passage_retrieval(query)
                    base_candidate_k = min(
                        max(num_to_retrieve, self.global_config.qa_top_k, dense_top_k),
                        len(sorted_doc_ids),
                    )
                    repair_base_docs = [
                        self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                        for idx in sorted_doc_ids[:base_candidate_k]
                    ]
                    repair_dense_docs = [
                        self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                        for idx in repair_dense_doc_ids[:dense_top_k]
                    ]
                    sorted_doc_ids, sorted_doc_scores, graph_repair_info = graph_repair_retrieve(
                        question=query,
                        base_doc_ids=sorted_doc_ids,
                        base_doc_scores=sorted_doc_scores,
                        base_docs=repair_base_docs,
                        dense_doc_ids=repair_dense_doc_ids,
                        dense_doc_scores=repair_dense_doc_scores,
                        dense_docs=repair_dense_docs,
                        fact_rerank_info=fact_rerank_info,
                        qa_top_k=self.global_config.qa_top_k,
                        enabled=True,
                        dense_top_k=dense_top_k,
                        max_promotions=getattr(self.global_config, "graph_repair_max_promotions", 2),
                        protect_top_k=getattr(self.global_config, "graph_repair_protect_top_k", 1),
                        min_score=getattr(self.global_config, "graph_repair_min_score", 0.22),
                        min_top_facts=getattr(self.global_config, "graph_repair_min_top_facts", 1),
                        min_candidate_facts=getattr(self.global_config, "graph_repair_min_candidate_facts", 2),
                        trigger_on_dense_fallback=getattr(
                            self.global_config,
                            "graph_repair_trigger_on_dense_fallback",
                            True,
                        ),
                    )
            query_schema_memory_facts = top_k_facts
            query_schema_memory_fact_source = "top_facts"
            if not query_schema_memory_facts:
                query_schema_memory_facts = (rerank_log or {}).get("facts_before_rerank", [])
                query_schema_memory_fact_source = "candidate_facts_before_rerank"

            operator_context = {
                "doc_ids": sorted_doc_ids,
                "doc_scores": sorted_doc_scores,
            }
            relation_base_context = {
                "doc_ids": (
                    sorted_doc_ids.copy()
                    if hasattr(sorted_doc_ids, "copy")
                    else list(sorted_doc_ids or [])
                ),
                "doc_scores": (
                    sorted_doc_scores.copy()
                    if hasattr(sorted_doc_scores, "copy")
                    else list(sorted_doc_scores or [])
                ),
            }
            schema_bind_candidates = []

            def execute_schema_probe():
                started = time.perf_counter()
                new_doc_ids, new_doc_scores, info = self.query_schema_memory_retrieve(
                    query=query,
                    sorted_doc_ids=operator_context["doc_ids"],
                    sorted_doc_scores=operator_context["doc_scores"],
                    top_k_facts=query_schema_memory_facts,
                    num_to_retrieve=num_to_retrieve,
                    fact_source=query_schema_memory_fact_source,
                )
                info = dict(info or {})
                latency_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
                items = list(info.get("candidate_items") or [])
                candidate_latency_ms = latency_ms / float(max(1, len(items)))
                info["operator_latency_ms"] = round(latency_ms, 6)
                info["physical_operator"] = "schema_hash_lookup_with_join_key_probe"
                info["executed_by_lazy_optimizer"] = bool(
                    getattr(self.global_config, "enable_adaptive_lazy_query_executor", False)
                )
                for candidate in items:
                    candidate["operator_latency_ms"] = round(candidate_latency_ms, 6)
                    candidate["operator_name"] = "schema_hash_lookup_with_join_key_probe"
                if (
                    getattr(
                        self.global_config,
                        "enable_schema_bind_fact_join",
                        False,
                    )
                    or getattr(
                        self.global_config,
                        "enable_schema_bound_semijoin",
                        False,
                    )
                    or getattr(
                        self.global_config,
                        "enable_single_trace_fact_join_replay",
                        False,
                    )
                ):
                    schema_bind_candidates[:] = [
                        candidate
                        for candidate in items
                        if bool(candidate.get("schema_fact_tuple"))
                        and bool(
                            candidate.get(
                                "entity_bound",
                                candidate.get("join_key_matched", False),
                            )
                        )
                        and isinstance(candidate.get("triple"), (list, tuple))
                        and len(candidate.get("triple")) == 3
                    ]
                info["bind_output_count"] = len(schema_bind_candidates)
                operator_context["doc_ids"] = new_doc_ids
                operator_context["doc_scores"] = new_doc_scores
                return {
                    "doc_ids": new_doc_ids,
                    "doc_scores": new_doc_scores,
                    "info": info,
                }

            entity_predicate_fact_join_info = {
                "enabled": bool(
                    getattr(
                        self.global_config,
                        "enable_entity_predicate_fact_join",
                        False,
                    )
                    or getattr(
                        self.global_config,
                        "enable_query_schema_memory_cross_schema_join",
                        False,
                    )
                ),
                "applied": False,
                "reason": "awaiting_planned_retrieval",
                "rank_stable": True,
                "operator": "entity_predicate_fact_join",
                "candidate_count": 0,
                "candidate_items": [],
            }

            def execute_relation_fact_join():
                started = time.perf_counter()
                bind_pipeline_enabled = bool(
                    getattr(
                        self.global_config,
                        "enable_schema_bind_fact_join",
                        False,
                    )
                )
                semijoin_enabled = bool(
                    getattr(
                        self.global_config,
                        "enable_schema_bound_semijoin",
                        False,
                    )
                )
                terminal_variable_join_enabled = bool(
                    getattr(
                        self.global_config,
                        "enable_terminal_variable_bound_fact_join",
                        False,
                    )
                )
                single_trace_replay_enabled = bool(
                    getattr(
                        self.global_config,
                        "enable_single_trace_fact_join_replay",
                        False,
                    )
                )
                relation_input_context = (
                    relation_base_context
                    if bind_pipeline_enabled
                    else operator_context
                )
                if str(getattr(
                    self.global_config,
                    "paper_evidence_baseline_mode",
                    "core",
                )) == "dense_reader":
                    raw_base_doc_ids = relation_input_context.get("doc_ids")
                    base_doc_ids = [
                        int(value)
                        for value in list(
                            raw_base_doc_ids
                            if raw_base_doc_ids is not None else []
                        )
                    ]
                    paper_trace = {
                        "version": "dense_reader_observable_baseline_v1",
                        "query_id": f"{self.global_config.dataset}:{q_idx}",
                        "question": query,
                        "policy": "dense_reader",
                        "baseline_mode": "dense_reader",
                        "logical_plan": {},
                        "query_compiler_trace": {},
                        "semantic_execution_admitted": False,
                        "semantic_execution_admission": (
                            "not_applicable_dense_baseline"
                        ),
                        "supported": False,
                        "structured_admission": False,
                        "structured_answer": "",
                        "question_complete_supported_witness": False,
                        "partial_supported_path": False,
                        "facts": [],
                        "bindings": [],
                        "operator_trace": [],
                        "reader_evidence_doc_ids": [],
                        "reader_evidence_fact_ids": [],
                        "reader_evidence_applied": False,
                        "reader_evidence_mode": "dense_retrieval_only",
                        "output_path": "ordinary_baseline_fallback",
                        "stop_reason": "dense_baseline_no_relation_execution",
                        "cost_ledger": {
                            "actions": 0,
                            "scan_rows": 0,
                            "retrieval_calls": 0,
                            "model_calls": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "unknown_token_events": 0,
                            "operator_wall_ms": 0.0,
                            "planning_wall_ms": 0.0,
                            "cache_hits": 0,
                            "budget_violations": [],
                        },
                        "fixed_stages": [
                            {
                                "stage_id": "fixed0",
                                "stage": "BASE_RETRIEVAL",
                                "backend": self._retrieval_backend_diagnostics(),
                                "candidate_doc_count": len(base_doc_ids),
                                "reader_doc_count": int(
                                    self.global_config.qa_top_k
                                ),
                                "retrieval_calls": 1,
                                "model_calls": 0,
                            },
                            {
                                "stage_id": "fixed1",
                                "stage": "QUERY_COMPILATION",
                                "compile_status": "NOT_RUN_DENSE_BASELINE",
                                "model_calls": 0,
                                "wall_ms": 0.0,
                            },
                        ],
                    }
                    return {
                        "doc_ids": [],
                        "docs": [],
                        "info": {
                            "enabled": True,
                            "applied": False,
                            "reason": paper_trace["stop_reason"],
                            "physical_operator": "dense_retrieval_only",
                            "executed_by_lazy_optimizer": False,
                            "accepted_count": 0,
                            "accepted": [],
                            "structured_admission": False,
                            "reader_evidence_mode": "dense_retrieval_only",
                            "candidate_count": 0,
                            "candidate_table_rows": [],
                            "candidate_table_row_count": 0,
                            "candidate_table_doc": "",
                            "reader_candidate_doc_ids": [],
                            "paper_evidence_execution": paper_trace,
                            "operator_latency_ms": max(
                                0.0,
                                (time.perf_counter() - started) * 1000.0,
                            ),
                        },
                    }

                obligation_source_lookup = None
                obligation_source_catalog_path = str(getattr(
                    self.global_config,
                    "paper_evidence_obligation_source_catalog_path",
                    "",
                ) or "").strip()
                if bool(getattr(
                    self.global_config,
                    "paper_evidence_obligation_source_catalog",
                    False,
                )):
                    from .utils.obligation_source_catalog_utils import (
                        FrozenObligationSourceCatalog,
                    )

                    cached_catalog = getattr(
                        self, "_paper_obligation_source_catalog", None
                    )
                    cached_path = str(getattr(
                        self, "_paper_obligation_source_catalog_path", ""
                    ))
                    if (
                        cached_catalog is None
                        or cached_path != obligation_source_catalog_path
                    ):
                        cached_catalog = FrozenObligationSourceCatalog(
                            obligation_source_catalog_path
                        )
                        self._paper_obligation_source_catalog = cached_catalog
                        self._paper_obligation_source_catalog_path = (
                            obligation_source_catalog_path
                        )
                    obligation_source_lookup = cached_catalog.lookup

                def bound_completion_lookup(
                    completion_query: str,
                    top_k: int,
                ) -> List[Dict[str, Any]]:
                    """Merge current evidence with a bounded local dense probe."""
                    limit = max(1, int(top_k))
                    use_evidence_pool = bool(
                        getattr(
                            self.global_config,
                            "enable_completion_evidence_pool",
                            True,
                        )
                    )
                    documents: List[Dict[str, Any]] = []
                    seen_doc_ids = set()

                    def append_document(
                        raw_doc_id: Any,
                        *,
                        rank: int,
                        score: float,
                        source: str,
                    ) -> None:
                        try:
                            doc_id = int(raw_doc_id)
                            content = str(
                                self.chunk_embedding_store.get_row(
                                    self.passage_node_keys[doc_id]
                                )["content"]
                                or ""
                            )
                        except Exception:
                            return
                        if doc_id in seen_doc_ids or not content:
                            return
                        seen_doc_ids.add(doc_id)
                        title = content.splitlines()[0].strip() if content else ""
                        documents.append({
                            "doc_id": doc_id,
                            "rank": int(rank),
                            "score": float(score),
                            "content": content,
                            "title": title,
                            "completion_source": str(source),
                        })

                    if use_evidence_pool:
                        raw_base_doc_ids = relation_input_context.get("doc_ids")
                        raw_base_doc_scores = relation_input_context.get(
                            "doc_scores"
                        )
                        base_doc_ids = list(
                            raw_base_doc_ids
                            if raw_base_doc_ids is not None else []
                        )
                        base_doc_scores = list(
                            raw_base_doc_scores
                            if raw_base_doc_scores is not None else []
                        )



                        base_quota = min(2, limit, len(base_doc_ids))
                        for rank, raw_doc_id in enumerate(
                            base_doc_ids[:base_quota]
                        ):
                            score = (
                                float(base_doc_scores[rank])
                                if rank < len(base_doc_scores)
                                else 0.0
                            )
                            append_document(
                                raw_doc_id,
                                rank=rank,
                                score=score,
                                source="current_evidence_pool",
                            )

                    completion_doc_ids = []
                    completion_doc_scores = []
                    try:
                        self.get_query_embeddings([completion_query])
                        completion_doc_ids, completion_doc_scores = (
                            self.dense_passage_retrieval(completion_query)
                        )
                    except Exception as exc:
                        logger.warning(
                            "Bound completion retrieval failed for %r: %s",
                            completion_query,
                            exc,
                        )
                    for dense_rank, raw_doc_id in enumerate(
                        list(completion_doc_ids)
                    ):
                        if len(documents) >= limit:
                            break
                        score = (
                            float(completion_doc_scores[dense_rank])
                            if dense_rank < len(completion_doc_scores)
                            else 0.0
                        )
                        append_document(
                            raw_doc_id,
                            rank=dense_rank,
                            score=score,
                            source="dense_completion_probe",
                        )
                    return documents

                def query_time_relation_materializer(
                    *,
                    bound_surface: str,
                    bound_key: str,
                    operator: Mapping[str, Any],
                    documents: Sequence[Mapping[str, Any]],
                    binding: Optional[Mapping[str, Any]] = None,
                ) -> Dict[str, Any]:
                    """Extract one obligation from a bounded real source pool."""
                    from .utils.query_time_relation_materialization_utils import (
                        WINDOWED_EXTRACTIVE_TOPK_VERSION,
                        build_query_time_relation_materialization_messages,
                        decode_and_materialize_response,
                        merge_materialization_results,
                        rank_binding_predicate_source_windows,
                        should_run_whole_source_fallback,
                    )

                    max_docs = max(1, int(getattr(
                        self.global_config,
                        "paper_evidence_query_time_relation_materialization_max_docs",
                        4,
                    )))
                    active_documents = [
                        dict(value) for value in list(documents)[:max_docs]
                    ]
                    windowed_topk = bool(getattr(
                        self.global_config,
                        "paper_evidence_qtrm_windowed_extractive_topk",
                        False,
                    ))
                    directional_surface_canonicalization = bool(getattr(
                        self.global_config,
                        "paper_evidence_qtrm_directional_surface_canonicalization",
                        False,
                    ))
                    source_windows = None
                    max_candidates = max(1, int(getattr(
                        self.global_config,
                        "paper_evidence_qtrm_windowed_max_candidates",
                        3,
                    )))
                    max_passes_per_obligation = max(1, int(getattr(
                        self.global_config,
                        "paper_evidence_qtrm_max_passes_per_obligation",
                        2,
                    )))
                    if windowed_topk:
                        source_windows = rank_binding_predicate_source_windows(
                            bound_surface=bound_surface,
                            operator=operator,
                            documents=active_documents,
                            max_windows=max(1, int(getattr(
                                self.global_config,
                                "paper_evidence_qtrm_windowed_max_windows",
                                8,
                            ))),
                            max_sentences_per_window=2,
                        )
                    def run_materialization_pass(pass_windows):
                        messages = build_query_time_relation_materialization_messages(
                            bound_surface=bound_surface,
                            operator=operator,
                            documents=active_documents,
                            source_windows=pass_windows,
                            max_candidates=max_candidates,
                            enable_directional_surface_canonicalization=(
                                directional_surface_canonicalization
                            ),
                        )
                        llm_result = self.llm_model.infer(
                            messages,
                            max_completion_tokens=max(64, int(getattr(
                                self.global_config,
                                "paper_evidence_query_time_relation_materialization_max_completion_tokens",
                                320,
                            ))),
                            temperature=0.0,
                            response_format={"type": "json_object"},
                        )
                        if isinstance(llm_result, tuple):
                            response = llm_result[0]
                            pass_metadata = (
                                dict(llm_result[1] or {})
                                if len(llm_result) > 1 else {}
                            )
                            pass_cache_hit = (
                                bool(llm_result[2])
                                if len(llm_result) > 2
                                else bool(pass_metadata.get("cache_hit"))
                            )
                        else:
                            response = llm_result
                            pass_metadata = {}
                            pass_cache_hit = False
                        pass_decoded = decode_and_materialize_response(
                            response,
                            bound_surface=bound_surface,
                            bound_key=bound_key,
                            operator=operator,
                            documents=active_documents,
                            source_windows=pass_windows,
                            binding=binding,
                            enable_binding_conditioned_typed_materialization=bool(
                                getattr(
                                    self.global_config,
                                    "paper_evidence_binding_conditioned_typed_materialization",
                                    False,
                                )
                            ),
                            enable_directional_surface_canonicalization=(
                                directional_surface_canonicalization
                            ),
                        )
                        return pass_decoded, pass_metadata, pass_cache_hit

                    decoded, metadata, cache_hit = run_materialization_pass(
                        source_windows
                    )
                    pass_metadata = [metadata]
                    pass_cache_hits = [cache_hit]
                    primary_candidate_count = int(
                        decoded.get("candidate_count", 0) or 0
                    )
                    primary_accepted_count = int(
                        decoded.get("accepted_count", 0) or 0
                    )
                    fallback_used = False
                    fallback_candidate_count = 0
                    fallback_accepted_count = 0
                    fallback_skip_reason = ""
                    if windowed_topk:
                        for fact in decoded.get("facts") or []:
                            fact["qtrm_candidate_generation_version"] = (
                                WINDOWED_EXTRACTIVE_TOPK_VERSION
                            )
                            fact["qtrm_candidate_generation_path"] = "ranked_window"
                        model_limit = int(
                            operator.get("execution_model_limit", 0) or 0
                        )
                        if should_run_whole_source_fallback(
                            primary_accepted_count=primary_accepted_count,
                            execution_model_limit=model_limit,
                            max_passes_per_obligation=max_passes_per_obligation,
                        ):
                            fallback, fallback_metadata, fallback_cache_hit = (
                                run_materialization_pass(None)
                            )
                            fallback_used = True
                            fallback_candidate_count = int(
                                fallback.get("candidate_count", 0) or 0
                            )
                            fallback_accepted_count = int(
                                fallback.get("accepted_count", 0) or 0
                            )
                            for fact in fallback.get("facts") or []:
                                fact["qtrm_candidate_generation_version"] = (
                                    "whole_source_fallback_v1"
                                )
                                fact["qtrm_candidate_generation_path"] = (
                                    "whole_source_fallback"
                                )
                            decoded = merge_materialization_results(
                                decoded, fallback
                            )
                            pass_metadata.append(fallback_metadata)
                            pass_cache_hits.append(fallback_cache_hit)
                        elif primary_accepted_count == 0:
                            fallback_skip_reason = (
                                "per_obligation_pass_limit"
                                if max_passes_per_obligation < 2
                                else "insufficient_model_budget"
                            )
                    cache_hit = all(pass_cache_hits)
                    decoded["cache_hit"] = cache_hit
                    decoded["diagnostics"] = {
                        "source_document_count": len(active_documents),
                        "windowed_extractive_topk": int(windowed_topk),
                        "ranked_window_count": len(source_windows or []),
                        "ranked_windows": [
                            {
                                "window_id": str(row.get("window_id") or ""),
                                "doc_id": int(row.get("doc_id", -1)),
                                "score": float(row.get("score", 0.0) or 0.0),
                                "bound_surface_match": bool(
                                    row.get("bound_surface_match")
                                ),
                                "title_identity_match": bool(
                                    row.get("title_identity_match")
                                ),
                                "relation_term_hits": list(
                                    row.get("relation_term_hits") or []
                                ),
                            }
                            for row in source_windows or []
                        ],
                        "max_materialized_candidates": max_candidates,
                        "max_passes_per_obligation": max_passes_per_obligation,
                        "windowed_generation_version": (
                            WINDOWED_EXTRACTIVE_TOPK_VERSION
                            if windowed_topk else ""
                        ),
                        "windowed_primary_candidate_count": primary_candidate_count,
                        "windowed_primary_admitted_count": primary_accepted_count,
                        "whole_source_fallback_used": int(fallback_used),
                        "whole_source_fallback_candidate_count": (
                            fallback_candidate_count
                        ),
                        "whole_source_fallback_admitted_count": (
                            fallback_accepted_count
                        ),
                        "whole_source_fallback_skip_reason": fallback_skip_reason,
                        "parsed_response": int(bool(decoded.get("candidate_count"))),
                        "candidate_count": int(
                            decoded.get("candidate_count", 0) or 0
                        ),
                        "deterministically_admitted_count": int(
                            decoded.get("accepted_count", 0) or 0
                        ),
                        "binding_identity_admitted_count": int(
                            decoded.get("binding_identity_admitted_count", 0) or 0
                        ),
                        "typed_contract_admitted_count": int(
                            decoded.get("typed_contract_admitted_count", 0) or 0
                        ),
                        "binding_conditioned_typed_materialization": int(bool(
                            decoded.get("binding_conditioned_typed_materialization")
                        )),
                        "directional_surface_canonicalization": int(bool(
                            decoded.get("directional_surface_canonicalization")
                        )),
                    }
                    decoded["cost"] = {
                        "model_calls": len(pass_metadata),
                        "input_tokens": sum(
                            int(value.get("prompt_tokens", 0) or 0)
                            for value in pass_metadata
                        ),
                        "output_tokens": sum(
                            int(value.get("completion_tokens", 0) or 0)
                            for value in pass_metadata
                        ),
                        "cache_hit": cache_hit,
                    }
                    return decoded

                if bool(
                    getattr(
                        self.global_config,
                        "enable_paper_evidence_query_executor",
                        False,
                    )
                ):
                    current_source_candidates = []
                    if bool(getattr(
                        self.global_config,
                        "enable_completion_evidence_pool",
                        True,
                    )):
                        raw_current_doc_ids = relation_input_context.get("doc_ids")
                        raw_current_doc_scores = relation_input_context.get("doc_scores")
                        current_doc_ids = list(
                            raw_current_doc_ids
                            if raw_current_doc_ids is not None else []
                        )
                        current_doc_scores = list(
                            raw_current_doc_scores
                            if raw_current_doc_scores is not None else []
                        )
                        current_limit = max(1, int(getattr(
                            self.global_config,
                            "paper_evidence_completion_top_k",
                            6,
                        )))
                        for source_rank, raw_doc_id in enumerate(
                            current_doc_ids[:current_limit]
                        ):
                            try:
                                source_doc_id = int(raw_doc_id)
                            except (TypeError, ValueError):
                                continue
                            current_source_candidates.append({
                                "source_id": source_doc_id,
                                "doc_id": source_doc_id,
                                "rank": source_rank,
                                "score": (
                                    float(current_doc_scores[source_rank])
                                    if source_rank < len(current_doc_scores)
                                    else 0.0
                                ),
                                "source_type": "current_evidence_pool",
                                "completion_source": "current_evidence_pool",
                            })
                    if prefetched_compilation is not None:
                        (
                            logical_plan,
                            query_compiler_trace,
                            compile_wall_ms,
                        ) = prefetched_compilation
                    else:
                        (
                            logical_plan,
                            query_compiler_trace,
                            compile_wall_ms,
                        ) = self._compile_paper_evidence_plan(query)
                    if str(getattr(
                        self.global_config,
                        "paper_evidence_baseline_mode",
                        "core",
                    )) == "static_plan_atomic":
                        from .utils.heldout_atomic_rag_utils import (
                            ATOMIC_QUERY_VERSION,
                            budgeted_unique_source_ids,
                            build_plan_atomic_retrieval_queries,
                        )

                        semantic_validation = dict(
                            logical_plan.get("question_semantic_validation") or {}
                        )
                        auto_admitted = bool(
                            semantic_validation.get("auto_execution_eligible")
                        )
                        model_admitted = bool(
                            getattr(
                                self.global_config,
                                "paper_evidence_allow_model_based_semantic_admission",
                                False,
                            )
                            and semantic_validation.get(
                                "experimental_execution_eligible"
                            )
                            and str(
                                semantic_validation.get("review_decision") or ""
                            ) == "PASS"
                        )
                        semantic_admitted = bool(auto_admitted or model_admitted)
                        atomic_requests = []
                        ranked_source_groups = []
                        atomic_wall_ms = 0.0
                        if semantic_admitted:
                            atomic_requests = build_plan_atomic_retrieval_queries(
                                query,
                                logical_plan,
                                max_queries=max(0, int(getattr(
                                    self.global_config,
                                    "paper_evidence_atomic_max_retrieval_calls",
                                    2,
                                ))),
                            )
                            for request in atomic_requests:
                                atomic_started = time.perf_counter()
                                self.get_query_embeddings([request["query"]])
                                atomic_doc_ids, atomic_doc_scores = (
                                    self.dense_passage_retrieval(request["query"])
                                )
                                request["source_ids"] = [
                                    int(value) for value in list(atomic_doc_ids)[:20]
                                ]
                                request["scores"] = [
                                    float(value) for value in list(atomic_doc_scores)[:20]
                                ]
                                request["wall_ms"] = round(
                                    (time.perf_counter() - atomic_started) * 1000.0,
                                    6,
                                )
                                atomic_wall_ms += float(request["wall_ms"])
                                ranked_source_groups.append(request["source_ids"])

                        raw_base_doc_ids = relation_input_context.get("doc_ids")
                        base_doc_ids = [
                            int(value)
                            for value in list(
                                raw_base_doc_ids
                                if raw_base_doc_ids is not None else []
                            )
                        ]
                        atomic_doc_ids = budgeted_unique_source_ids(
                            ranked_source_groups,
                            excluded_source_ids=base_doc_ids,
                            max_sources=max(0, int(getattr(
                                self.global_config,
                                "paper_evidence_atomic_max_extra_sources",
                                3,
                            ))),
                        )
                        atomic_docs = []
                        valid_atomic_doc_ids = []
                        for doc_id in atomic_doc_ids:
                            try:
                                atomic_docs.append(
                                    self.chunk_embedding_store.get_row(
                                        self.passage_node_keys[int(doc_id)]
                                    )["content"]
                                )
                                valid_atomic_doc_ids.append(int(doc_id))
                            except (KeyError, IndexError, TypeError, ValueError):
                                continue
                        paper_trace = {
                            "version": "static_plan_atomic_reader_baseline_v1",
                            "baseline_version": ATOMIC_QUERY_VERSION,
                            "query_id": f"{self.global_config.dataset}:{q_idx}",
                            "question": query,
                            "policy": "static_plan_atomic",
                            "baseline_mode": "static_plan_atomic",
                            "logical_plan": logical_plan,
                            "query_compiler_trace": query_compiler_trace,
                            "semantic_execution_admitted": semantic_admitted,
                            "supported": semantic_admitted,
                            "structured_admission": False,
                            "structured_answer": "",
                            "question_complete_supported_witness": False,
                            "partial_supported_path": False,
                            "facts": [],
                            "bindings": [],
                            "operator_trace": [],
                            "atomic_retrieval_requests": atomic_requests,
                            "reader_evidence_doc_ids": valid_atomic_doc_ids,
                            "reader_evidence_fact_ids": [],
                            "reader_evidence_applied": bool(valid_atomic_doc_ids),
                            "reader_evidence_mode": "static_plan_atomic_retrieval",
                            "output_path": "ordinary_baseline_fallback",
                            "stop_reason": (
                                "static_plan_atomic_complete"
                                if semantic_admitted
                                else "question_semantic_review_required"
                            ),
                            "cost_ledger": {
                                "actions": len(atomic_requests),
                                "scan_rows": 0,
                                "retrieval_calls": len(atomic_requests),
                                "model_calls": 0,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "unknown_token_events": 0,
                                "operator_wall_ms": round(atomic_wall_ms, 6),
                                "planning_wall_ms": 0.0,
                                "cache_hits": 0,
                                "budget_violations": [],
                            },
                            "fixed_stages": [
                                {
                                    "stage_id": "fixed0",
                                    "stage": "BASE_RETRIEVAL",
                                    "backend": self._retrieval_backend_diagnostics(),
                                    "candidate_doc_count": len(base_doc_ids),
                                    "reader_doc_count": int(
                                        self.global_config.qa_top_k
                                    ),
                                    "retrieval_calls": 1,
                                    "model_calls": 0,
                                },
                                {
                                    "stage_id": "fixed1",
                                    "stage": "QUERY_COMPILATION",
                                    "compile_status": logical_plan.get(
                                        "compile_status"
                                    ),
                                    "plan_id": logical_plan.get("plan_id"),
                                    "compiler_source": query_compiler_trace.get(
                                        "selected_source"
                                    ),
                                    "compiler_mode": query_compiler_trace.get("mode"),
                                    "query_shape": logical_plan.get("plan_kind"),
                                    "operator_count": len(
                                        list(logical_plan.get("operators") or [])
                                    ),
                                    "wall_ms": round(compile_wall_ms, 6),
                                    "model_calls": int(
                                        query_compiler_trace.get("llm_requests", 0)
                                        or 0
                                    ),
                                    "cache_hit": query_compiler_trace.get(
                                        "cache_hit"
                                    ),
                                    "input_tokens": query_compiler_trace.get(
                                        "input_tokens"
                                    ),
                                    "output_tokens": query_compiler_trace.get(
                                        "output_tokens"
                                    ),
                                },
                            ],
                        }
                        return {
                            "doc_ids": valid_atomic_doc_ids,
                            "docs": atomic_docs,
                            "info": {
                                "enabled": True,
                                "applied": bool(valid_atomic_doc_ids),
                                "reason": paper_trace["stop_reason"],
                                "physical_operator": "static_plan_atomic_retrieval",
                                "executed_by_lazy_optimizer": False,
                                "accepted_count": len(valid_atomic_doc_ids),
                                "accepted": [],
                                "structured_admission": False,
                                "reader_evidence_mode": (
                                    "static_plan_atomic_retrieval"
                                ),
                                "candidate_count": sum(
                                    len(request.get("source_ids") or [])
                                    for request in atomic_requests
                                ),
                                "candidate_table_rows": [],
                                "candidate_table_row_count": 0,
                                "candidate_table_doc": "",
                                "reader_candidate_doc_ids": valid_atomic_doc_ids,
                                "paper_evidence_execution": paper_trace,
                                "operator_latency_ms": round(
                                    compile_wall_ms + atomic_wall_ms,
                                    6,
                                ),
                            },
                        }
                    paper_backend = RelationIndexEvidenceBackend(
                        relation_index=getattr(
                            self, "relation_fact_join_index", {}
                        ),
                        get_source_content=lambda doc_idx: (
                            self.chunk_embedding_store.get_row(
                                self.passage_node_keys[int(doc_idx)]
                            )["content"]
                        ),
                        get_source_title=lambda doc_idx: (
                            str(
                                self.chunk_embedding_store.get_row(
                                    self.passage_node_keys[int(doc_idx)]
                                )["content"]
                                or ""
                            ).splitlines()
                            or [""]
                        )[0].strip(),
                        completion_lookup=bound_completion_lookup,
                        obligation_source_lookup=obligation_source_lookup,
                        relation_materializer=query_time_relation_materializer,
                        current_source_candidates=current_source_candidates,
                        completion_top_k=getattr(
                            self.global_config,
                            "paper_evidence_completion_top_k",
                            6,
                        ),
                        completion_mode=getattr(
                            self.global_config,
                            "paper_evidence_completion_mode",
                            "lazy",
                        ),
                        completion_query_mode=getattr(
                            self.global_config,
                            "paper_evidence_completion_query_mode",
                            "obligation_bound",
                        ),
                        verification_mode=getattr(
                            self.global_config,
                            "paper_evidence_verification_mode",
                            "strict",
                        ),
                        repair_scheduler_mode=getattr(
                            self.global_config,
                            "paper_evidence_repair_scheduler_mode",
                            "eager",
                        ),
                        execution_mode=getattr(
                            self.global_config,
                            "paper_evidence_execution_mode",
                            "saqe",
                        ),
                        index_delete_rate=getattr(
                            self.global_config,
                            "paper_evidence_index_delete_rate",
                            0.0,
                        ),
                        index_corrupt_rate=getattr(
                            self.global_config,
                            "paper_evidence_index_corrupt_rate",
                            0.0,
                        ),
                        index_corrupt_mode=getattr(
                            self.global_config,
                            "paper_evidence_index_corrupt_mode",
                            "object",
                        ),
                        perturbation_seed=getattr(
                            self.global_config,
                            "paper_evidence_perturbation_seed",
                            0,
                        ),
                        max_facts_per_key=getattr(
                            self.global_config,
                            "relation_fact_join_max_facts_per_key",
                            60,
                        ),
                        max_key_fanout=getattr(
                            self.global_config,
                            "relation_fact_join_max_key_fanout",
                            600,
                        ),
                        enable_title_catalog_completion=bool(getattr(
                            self.global_config,
                            "paper_evidence_title_catalog_completion",
                            True,
                        )),
                        enable_provenance_source_verifier=bool(getattr(
                            self.global_config,
                            "paper_evidence_provenance_source_verifier",
                            True,
                        )),
                        enable_identity_safe_composite_join=bool(getattr(
                            self.global_config,
                            "paper_evidence_identity_safe_composite_join",
                            True,
                        )),
                        relation_matching_mode=str(getattr(
                            self.global_config,
                            "paper_evidence_relation_matching_mode",
                            "canonical",
                        )),
                        enable_predicate_subtype_constraints=bool(getattr(
                            self.global_config,
                            "paper_evidence_predicate_subtype_constraints",
                            True,
                        )),
                        enable_type_role_constraints=bool(getattr(
                            self.global_config,
                            "paper_evidence_type_role_constraints",
                            True,
                        )),
                        enable_composite_index_access=bool(getattr(
                            self.global_config,
                            "paper_evidence_composite_index_access",
                            True,
                        )),
                        enable_schema_plan_coverage_v7=bool(getattr(
                            self.global_config,
                            "paper_evidence_schema_plan_coverage_v7",
                            False,
                        )),
                        enable_lineage_terminal_semijoin_v7_1=bool(getattr(
                            self.global_config,
                            "paper_evidence_lineage_terminal_semijoin_v7_1",
                            False,
                        )),
                        enable_provenance_grouped_terminal_v7_2=bool(getattr(
                            self.global_config,
                            "paper_evidence_provenance_grouped_terminal_v7_2",
                            False,
                        )),
                        enable_schema_exact_nested_join_v9=bool(getattr(
                            self.global_config,
                            "paper_evidence_schema_exact_nested_join_v9",
                            False,
                        )),
                        enable_collision_aware_schema_identity=bool(getattr(
                            self.global_config,
                            "paper_evidence_collision_aware_schema_identity",
                            False,
                        )),
                        enable_safe_title_family_root_binding=bool(getattr(
                            self.global_config,
                            "paper_evidence_safe_title_family_root_binding",
                            True,
                        )),
                        enable_source_grounded_terminal_tuple_recovery=bool(
                            getattr(
                                self.global_config,
                                "paper_evidence_source_grounded_terminal_tuple_recovery",
                                False,
                            )
                        ),
                        enable_obligation_bound_repair_certificate=bool(
                            getattr(
                                self.global_config,
                                "paper_evidence_obligation_bound_repair_certificate",
                                False,
                            )
                        ),
                        enable_terminal_candidate_priority=bool(
                            getattr(
                                self.global_config,
                                "paper_evidence_terminal_candidate_priority",
                                False,
                            )
                        ),
                        enable_schema_compiled_source_certificate=bool(
                            getattr(
                                self.global_config,
                                "paper_evidence_schema_compiled_source_certificate",
                                False,
                            )
                        ),
                        enable_source_support_obligation_admission=bool(
                            getattr(
                                self.global_config,
                                "paper_evidence_source_support_obligation_admission",
                                False,
                            )
                        ),
                        enable_enhanced_source_support=bool(getattr(
                            self.global_config,
                            "paper_evidence_enhanced_source_support",
                            False,
                        )),
                        enable_obligation_admission=bool(getattr(
                            self.global_config,
                            "paper_evidence_obligation_admission",
                            False,
                        )),
                        enable_query_local_overlay_reuse=bool(getattr(
                            self.global_config,
                            "paper_evidence_query_local_overlay_reuse",
                            True,
                        )),
                        enable_predicate_entailment_admission=bool(getattr(
                            self.global_config,
                            "paper_evidence_predicate_entailment_admission",
                            False,
                        )),
                        enable_binding_obligation_contract_validation=bool(
                            getattr(
                                self.global_config,
                                "paper_evidence_binding_obligation_contract_validation",
                                False,
                            )
                        ),
                        enable_query_time_relation_materialization=bool(getattr(
                            self.global_config,
                            "paper_evidence_query_time_relation_materialization",
                            False,
                        )),
                        enable_binding_conditioned_typed_materialization=bool(
                            getattr(
                                self.global_config,
                                "paper_evidence_binding_conditioned_typed_materialization",
                                False,
                            )
                        ),
                        cbo_calibration_path=str(getattr(
                            self.global_config,
                            "paper_evidence_cbo_calibration_path",
                            "",
                        )),
                        cbo_min_samples=int(getattr(
                            self.global_config,
                            "paper_evidence_cbo_min_samples",
                            8,
                        )),
                        cbo_risk_weight=float(getattr(
                            self.global_config,
                            "paper_evidence_cbo_risk_weight",
                            0.8,
                        )),
                        cbo_cost_weight=float(getattr(
                            self.global_config,
                            "paper_evidence_cbo_cost_weight",
                            0.25,
                        )),
                        bswc_beam_width=int(getattr(
                            self.global_config,
                            "paper_evidence_bswc_beam_width",
                            8,
                        )),
                        bswc_search_horizon=int(getattr(
                            self.global_config,
                            "paper_evidence_bswc_search_horizon",
                            3,
                        )),
                        bswc_source_calibration_path=str(getattr(
                            self.global_config,
                            "paper_evidence_bswc_source_calibration_path",
                            "",
                        )),
                        bswc_source_min_samples=int(getattr(
                            self.global_config,
                            "paper_evidence_bswc_source_min_samples",
                            8,
                        )),
                        bswc_feedback_strength=float(getattr(
                            self.global_config,
                            "paper_evidence_bswc_feedback_strength",
                            3.0,
                        )),
                        enable_bswc_query_feedback=bool(getattr(
                            self.global_config,
                            "paper_evidence_bswc_query_feedback",
                            True,
                        )),
                        bswc_stop_threshold=float(getattr(
                            self.global_config,
                            "paper_evidence_bswc_stop_threshold",
                            0.035,
                        )),
                        bswc_min_exploration_actions=int(getattr(
                            self.global_config,
                            "paper_evidence_bswc_min_exploration_actions",
                            1,
                        )),
                        bswc_fpawc_stop_upper_bound=float(getattr(
                            self.global_config,
                            "paper_evidence_bswc_fpawc_stop_upper_bound",
                            0.0,
                        )),
                    )
                    paper_trace = execute_evidence_query(
                        query_id=f"{self.global_config.dataset}:{q_idx}",
                        question=query,
                        logical_plan=logical_plan,
                        backend=paper_backend,
                        policy=getattr(
                            self.global_config,
                            "paper_evidence_policy",
                            "adaptive_dependency",
                        ),
                        budget=ExecutionBudget(
                            max_actions=getattr(
                                self.global_config,
                                "paper_evidence_max_actions",
                                12,
                            ),
                            max_retrieval_calls=getattr(
                                self.global_config,
                                "paper_evidence_max_retrieval_calls",
                                2,
                            ),
                            max_model_calls=getattr(
                                self.global_config,
                                "paper_evidence_max_model_calls",
                                2,
                            ),
                            max_scan_rows=getattr(
                                self.global_config,
                                "paper_evidence_max_scan_rows",
                                5000,
                            ),
                            max_tokens=(
                                getattr(
                                    self.global_config,
                                    "paper_evidence_max_tokens",
                                    0,
                                )
                                or None
                            ),
                            reserve_model_calls=getattr(
                                self.global_config,
                                "paper_evidence_reserve_model_calls",
                                0,
                            ),
                            reserve_tokens=getattr(
                                self.global_config,
                                "paper_evidence_reserve_tokens",
                                0,
                            ),
                        ),
                        max_no_progress=getattr(
                            self.global_config,
                            "paper_evidence_max_no_progress",
                            3,
                        ),
                        allow_model_based_semantic_admission=bool(getattr(
                            self.global_config,
                            "paper_evidence_allow_model_based_semantic_admission",
                            False,
                        )),
                        require_semantic_execution_admission=bool(getattr(
                            self.global_config,
                            "paper_evidence_require_semantic_execution_admission",
                            False,
                        )),
                    )
                    paper_trace["fixed_stages"] = [
                        {
                            "stage_id": "fixed0",
                            "stage": "BASE_RETRIEVAL",
                            "backend": self._retrieval_backend_diagnostics(),
                            "candidate_doc_count": _collection_size(
                                relation_input_context.get("doc_ids")
                            ),
                            "reader_doc_count": int(self.global_config.qa_top_k),
                            "retrieval_calls": 1,
                            "model_calls": 0,
                        },
                        {
                            "stage_id": "fixed1",
                            "stage": "QUERY_COMPILATION",
                            "compile_status": logical_plan.get("compile_status"),
                            "plan_id": logical_plan.get("plan_id"),
                            "compiler_source": query_compiler_trace.get(
                                "selected_source"
                            ),
                            "compiler_mode": query_compiler_trace.get("mode"),
                            "query_shape": logical_plan.get("plan_kind"),
                            "operator_count": len(
                                list(logical_plan.get("operators") or [])
                            ),
                            "wall_ms": round(compile_wall_ms, 6),
                            "model_calls": int(
                                query_compiler_trace.get("llm_requests", 0) or 0
                            ),
                            "cache_hit": query_compiler_trace.get("cache_hit"),
                            "input_tokens": query_compiler_trace.get("input_tokens"),
                            "output_tokens": query_compiler_trace.get("output_tokens"),
                        },
                    ]
                    facts_by_id = {
                        str(row.get("fact_id")): row
                        for row in paper_trace.get("facts") or []
                    }
                    reader_evidence_mode = str(
                        paper_trace.get("reader_evidence_mode") or "none"
                    )
                    reader_fact_ids = list(
                        paper_trace.get("reader_evidence_fact_ids") or []
                    )
                    v8_safe_reader_shadow = bool(
                        getattr(
                            self.global_config,
                            "paper_evidence_v8_safe_materialization",
                            True,
                        )
                        and bool(
                            (paper_trace.get("logical_plan") or {}).get(
                                "newly_compiled_v8"
                            )
                            or (paper_trace.get("logical_plan") or {}).get(
                                "newly_compiled_v9"
                            )
                        )
                    )
                    if v8_safe_reader_shadow:
                        paper_trace["v8_shadow_reader_evidence_fact_ids"] = list(
                            reader_fact_ids
                        )
                        paper_trace["v8_shadow_reader_evidence_mode"] = (
                            reader_evidence_mode
                        )
                        terminal_determinacy = (
                            assess_terminal_answer_determinacy(paper_trace)
                        )
                        paper_trace["terminal_answer_determinacy"] = (
                            terminal_determinacy
                        )
                        if terminal_determinacy.get(
                            "reader_evidence_eligible"
                        ):
                            reader_fact_ids = list(
                                terminal_determinacy.get("eligible_fact_ids")
                                or []
                            )
                            reader_evidence_mode = (
                                "predicate_bound_supported_terminal"
                            )
                        else:
                            reader_fact_ids = []
                            reader_evidence_mode = "v8_safe_trace_only"
                    if bool(getattr(
                        self.global_config,
                        "paper_evidence_answer_domain_guard",
                        False,
                    )):
                        terminal_answer_domain = assess_terminal_answer_domain(
                            paper_trace
                        )
                        paper_trace["terminal_answer_domain"] = (
                            terminal_answer_domain
                        )
                        if not terminal_answer_domain.get(
                            "reader_evidence_eligible", True
                        ):
                            paper_trace[
                                "answer_domain_shadow_reader_evidence_fact_ids"
                            ] = list(reader_fact_ids)
                            paper_trace[
                                "answer_domain_shadow_reader_evidence_mode"
                            ] = reader_evidence_mode
                            reader_fact_ids = []
                            reader_evidence_mode = (
                                "terminal_answer_domain_incompatible"
                            )
                    category_affine_partial = {
                        "eligible": False,
                        "reason": "disabled",
                    }
                    if bool(getattr(
                        self.global_config,
                        "paper_evidence_category_affine_partial_materialization",
                        False,
                    )):
                        category_affine_partial = (
                            assess_category_affine_supported_partial(
                                paper_trace,
                                max_terminal_candidates=max(
                                    1,
                                    int(getattr(
                                        self.global_config,
                                        "paper_evidence_partial_max_docs",
                                        2,
                                    )),
                                ),
                            )
                        )
                        paper_trace["category_affine_partial_evidence"] = (
                            category_affine_partial
                        )
                        if category_affine_partial.get("eligible"):
                            reader_fact_ids = list(
                                category_affine_partial.get("fact_ids") or []
                            )
                            reader_evidence_mode = (
                                "category_affine_supported_partial"
                            )
                    if (
                        reader_evidence_mode == "verified_partial_completion"
                        and not bool(
                            getattr(
                                self.global_config,
                                "paper_evidence_materialize_verified_partial",
                                True,
                            )
                        )
                    ):
                        reader_fact_ids = []
                        reader_evidence_mode = "partial_materialization_disabled"
                    if not bool(getattr(
                        self.global_config,
                        "paper_evidence_materialize_reader_evidence",
                        True,
                    )):
                        paper_trace["shadow_reader_evidence_fact_ids"] = list(
                            reader_fact_ids
                        )
                        paper_trace["shadow_reader_evidence_mode"] = (
                            reader_evidence_mode
                        )
                        reader_fact_ids = []
                        reader_evidence_mode = "reader_materialization_disabled"
                    evidence_facts = [
                        facts_by_id[fact_id]
                        for fact_id in reader_fact_ids
                        if fact_id in facts_by_id
                    ]
                    paper_trace["reader_evidence_fact_ids"] = list(
                        reader_fact_ids
                    )
                    paper_trace["reader_evidence_mode"] = reader_evidence_mode
                    join_doc_ids = []
                    for fact in evidence_facts:
                        try:
                            doc_id = int(fact.get("source_id"))
                        except (TypeError, ValueError):
                            continue
                        if doc_id not in join_doc_ids:
                            join_doc_ids.append(doc_id)
                    join_docs = []
                    valid_join_doc_ids = []
                    if reader_evidence_mode == "category_affine_supported_partial":
                        certificate_document = str(
                            category_affine_partial.get("certificate_document")
                            or ""
                        ).strip()
                        if certificate_document:
                            join_docs.append(certificate_document)
                            terminal_fact_ids = set(
                                category_affine_partial.get("terminal_fact_ids")
                                or []
                            )
                            terminal_source_ids = [
                                int(fact.get("source_id"))
                                for fact in evidence_facts
                                if str(fact.get("fact_id") or "")
                                in terminal_fact_ids
                                and str(fact.get("source_id") or "").lstrip("-").isdigit()
                            ]
                            if terminal_source_ids:
                                valid_join_doc_ids.append(terminal_source_ids[0])
                            paper_trace["reader_evidence_source_doc_ids"] = list(
                                join_doc_ids
                            )
                    else:
                        for doc_id in join_doc_ids:
                            try:
                                join_docs.append(
                                    self.chunk_embedding_store.get_row(
                                        self.passage_node_keys[doc_id]
                                    )["content"]
                                )
                                valid_join_doc_ids.append(doc_id)
                            except (KeyError, IndexError, TypeError, ValueError):
                                continue
                    admitted = bool(paper_trace.get("structured_admission"))
                    if reader_evidence_mode in {
                        "verified_partial_completion",
                        "category_affine_supported_partial",
                    }:
                        partial_limit = max(
                            0,
                            int(
                                getattr(
                                    self.global_config,
                                    "paper_evidence_partial_max_docs",
                                    2,
                                )
                            ),
                        )
                        valid_join_doc_ids = valid_join_doc_ids[:partial_limit]
                        join_docs = join_docs[:partial_limit]
                    reader_applied = bool(valid_join_doc_ids)
                    paper_trace["reader_evidence_doc_ids"] = list(
                        valid_join_doc_ids
                    )
                    paper_trace["reader_evidence_applied"] = reader_applied
                    paper_trace["reader_evidence_doc_count"] = len(
                        valid_join_doc_ids
                    )
                    info = {
                        "enabled": True,
                        "applied": reader_applied,
                        "reason": reader_evidence_mode,
                        "physical_operator": "paper_evidence_query_executor",
                        "executed_by_lazy_optimizer": False,
                        "accepted_count": len(valid_join_doc_ids),
                        "accepted": evidence_facts if reader_applied else [],
                        "structured_admission": admitted,
                        "reader_evidence_mode": reader_evidence_mode,
                        "candidate_count": len(paper_trace.get("facts") or []),
                        "candidate_table_rows": [],
                        "candidate_table_row_count": 0,
                        "candidate_table_doc": "",
                        "reader_candidate_doc_ids": (
                            valid_join_doc_ids
                        ),
                        "paper_evidence_execution": paper_trace,
                        "operator_latency_ms": round(
                            max(0.0, (time.perf_counter() - started) * 1000.0),
                            6,
                        ),
                    }
                    return {
                        "doc_ids": valid_join_doc_ids,
                        "docs": join_docs,
                        "info": info,
                    }

                join_doc_ids, info = select_relation_fact_join_evidence(
                    question=query,
                    sorted_doc_ids=relation_input_context["doc_ids"],
                    sorted_doc_scores=relation_input_context["doc_scores"],
                    top_k_facts=top_k_facts,
                    candidate_facts=(rerank_log or {}).get("facts_before_rerank", []),
                    upstream_fact_tuples=(
                        schema_bind_candidates
                        if semijoin_enabled or single_trace_replay_enabled
                        else []
                    ),
                    relation_index=getattr(self, "relation_fact_join_index", {}),
                    passage_node_keys=self.passage_node_keys,
                    get_doc_content=lambda doc_idx: self.chunk_embedding_store.get_row(
                        self.passage_node_keys[int(doc_idx)]
                    )["content"],
                    qa_top_k=self.global_config.qa_top_k,
                    enabled=getattr(
                        self.global_config,
                        "enable_relation_fact_join_retrieval",
                        False,
                    ),
                    bridge_only=getattr(
                        self.global_config,
                        "relation_fact_join_bridge_only",
                        False,
                    ),
                    require_evidence_gap=getattr(
                        self.global_config,
                        "relation_fact_join_require_evidence_gap",
                        True,
                    ),
                    candidate_top_k=getattr(
                        self.global_config,
                        "relation_fact_join_candidate_top_k",
                        72,
                    ),
                    max_append=getattr(
                        self.global_config,
                        "relation_fact_join_max_append",
                        1,
                    ),
                    max_facts_per_key=getattr(
                        self.global_config,
                        "relation_fact_join_max_facts_per_key",
                        60,
                    ),
                    max_key_fanout=getattr(
                        self.global_config,
                        "relation_fact_join_max_key_fanout",
                        600,
                    ),
                    protect_top_k=getattr(
                        self.global_config,
                        "relation_fact_join_protect_top_k",
                        2,
                    ),
                    min_query_overlap=getattr(
                        self.global_config,
                        "relation_fact_join_min_query_overlap",
                        0.04,
                    ),
                    min_relation_overlap=getattr(
                        self.global_config,
                        "relation_fact_join_min_relation_overlap",
                        0.04,
                    ),
                    min_utility=getattr(
                        self.global_config,
                        "relation_fact_join_min_utility",
                        0.50,
                    ),
                    max_risk=getattr(
                        self.global_config,
                        "relation_fact_join_max_risk",
                        0.52,
                    ),
                    min_net_utility=getattr(
                        self.global_config,
                        "relation_fact_join_min_net_utility",
                        0.04,
                    ),
                    token_budget=getattr(
                        self.global_config,
                        "relation_fact_join_token_budget",
                        850,
                    ),
                    token_weight=getattr(
                        self.global_config,
                        "evidence_cost_model_token_weight",
                        0.18,
                    ),
                    compute_weight=getattr(
                        self.global_config,
                        "evidence_cost_model_compute_weight",
                        0.08,
                    ),
                    risk_weight=getattr(
                        self.global_config,
                        "evidence_cost_model_risk_weight",
                        0.62,
                    ),
                    use_answerability_gap=getattr(
                        self.global_config,
                        "relation_fact_join_use_answerability_gap",
                        True,
                    ),
                    require_relation_role_match=getattr(
                        self.global_config,
                        "relation_fact_join_require_relation_role_match",
                        True,
                    ),
                    require_anchor_match=getattr(
                        self.global_config,
                        "relation_fact_join_require_anchor_match",
                        True,
                    ),
                    min_anchor_overlap=getattr(
                        self.global_config,
                        "relation_fact_join_min_anchor_overlap",
                        0.34,
                    ),
                    min_role_overlap=getattr(
                        self.global_config,
                        "relation_fact_join_min_role_overlap",
                        0.34,
                    ),
                    enable_candidate_table=getattr(
                        self.global_config,
                        "relation_fact_join_enable_candidate_table",
                        True,
                    ),
                    candidate_table_max_rows=getattr(
                        self.global_config,
                        "relation_fact_join_candidate_table_max_rows",
                        4,
                    ),
                    candidate_table_min_score=getattr(
                        self.global_config,
                        "relation_fact_join_candidate_table_min_score",
                        0.56,
                    ),
                    hard_anchor_min_overlap=getattr(
                        self.global_config,
                        "relation_fact_join_hard_anchor_min_overlap",
                        0.67,
                    ),
                    enable_multi_hop_planner=getattr(
                        self.global_config,
                        "relation_fact_join_enable_multi_hop_planner",
                        True,
                    ),
                    multi_hop_min_bridge_overlap=getattr(
                        self.global_config,
                        "relation_fact_join_multi_hop_min_bridge_overlap",
                        0.67,
                    ),
                    enable_surface_materializer=getattr(
                        self.global_config,
                        "relation_fact_join_enable_surface_materializer",
                        True,
                    ),
                    enable_provenance_optimizer=getattr(
                        self.global_config,
                        "relation_fact_join_enable_provenance_optimizer",
                        True,
                    ),
                    provenance_conflict_min_margin=getattr(
                        self.global_config,
                        "relation_fact_join_provenance_conflict_min_margin",
                        0.04,
                    ),
                    provenance_min_support_docs=getattr(
                        self.global_config,
                        "relation_fact_join_provenance_min_support_docs",
                        2,
                    ),
                    provenance_min_reliability=getattr(
                        self.global_config,
                        "relation_fact_join_provenance_min_reliability",
                        0.72,
                    ),
                    provenance_max_bonus=getattr(
                        self.global_config,
                        "relation_fact_join_provenance_max_bonus",
                        0.025,
                    ),
                    provenance_shadow_only=getattr(
                        self.global_config,
                        "relation_fact_join_provenance_shadow_only",
                        False,
                    ),
                    enable_schema_bound_semijoin=semijoin_enabled,
                    schema_bound_semijoin_fallback=(
                        not terminal_variable_join_enabled
                    ),
                    enable_schema_bound_residual_gate=getattr(
                        self.global_config,
                        "enable_schema_bound_residual_gate",
                        False,
                    ),
                    enable_schema_bound_terminal_slot_gate=getattr(
                        self.global_config,
                        "enable_schema_bound_terminal_slot_gate",
                        False,
                    ),
                    enable_schema_bound_predicate_family_index=getattr(
                        self.global_config,
                        "enable_schema_bound_predicate_family_index",
                        False,
                    ),
                    enable_terminal_variable_bound_fact_join=(
                        terminal_variable_join_enabled
                    ),
                    terminal_variable_min_anchor_overlap=getattr(
                        self.global_config,
                        "terminal_variable_fact_join_min_anchor_overlap",
                        0.67,
                    ),
                    enable_lineage_directed_terminal_join=getattr(
                        self.global_config,
                        "enable_lineage_directed_terminal_join",
                        False,
                    ),
                    enable_single_trace_fact_join_replay=(
                        single_trace_replay_enabled
                    ),
                    enable_stage_aware_fact_join_plan=getattr(
                        self.global_config,
                        "enable_stage_aware_fact_join_plan",
                        False,
                    ),
                    enable_role_directed_fact_join=getattr(
                        self.global_config,
                        "enable_role_directed_fact_join",
                        True,
                    ),
                    enable_canonical_bound_completion=getattr(
                        self.global_config,
                        "enable_canonical_bound_completion",
                        True,
                    ),
                    enable_completion_selectivity_feedback=getattr(
                        self.global_config,
                        "enable_completion_selectivity_feedback",
                        True,
                    ),
                    enable_query_plan_root_contract=getattr(
                        self.global_config,
                        "enable_query_plan_root_contract",
                        True,
                    ),
                    enable_inverse_relation_completion=getattr(
                        self.global_config,
                        "enable_inverse_relation_completion",
                        True,
                    ),
                    enable_title_grounded_completion=getattr(
                        self.global_config,
                        "enable_title_grounded_completion",
                        False,
                    ),
                    enable_source_grounded_terminal_repair=getattr(
                        self.global_config,
                        "enable_source_grounded_terminal_repair",
                        False,
                    ),
                    enable_title_catalog_root_plan=getattr(
                        self.global_config,
                        "enable_title_catalog_root_plan",
                        False,
                    ),
                    enable_title_root_plan_commit=getattr(
                        self.global_config,
                        "enable_title_root_plan_commit",
                        False,
                    ),
                    enable_identity_safe_root_commit=getattr(
                        self.global_config,
                        "enable_identity_safe_root_commit",
                        False,
                    ),
                    relation_fact_join_reader_stage_mode=getattr(
                        self.global_config,
                        "relation_fact_join_reader_stage_mode",
                        "legacy",
                    ),
                    enable_slot_dependency_cost_admission=getattr(
                        self.global_config,
                        "enable_slot_dependency_cost_admission",
                        True,
                    ),
                    enable_bound_completion_fact_join=getattr(
                        self.global_config,
                        "enable_bound_completion_fact_join",
                        False,
                    ),
                    bound_completion_lookup=bound_completion_lookup,
                    bound_completion_top_k=getattr(
                        self.global_config,
                        "bound_completion_top_k",
                        6,
                    ),
                    bound_completion_max_calls=getattr(
                        self.global_config,
                        "bound_completion_max_calls",
                        2,
                    ),
                    enable_bound_completion_cost_admission=getattr(
                        self.global_config,
                        "enable_bound_completion_cost_admission",
                        True,
                    ),
                    bound_completion_max_risk=getattr(
                        self.global_config,
                        "bound_completion_max_risk",
                        0.42,
                    ),
                    bound_completion_min_net_utility=getattr(
                        self.global_config,
                        "bound_completion_min_net_utility",
                        0.20,
                    ),
                    bound_completion_materialize_reader=getattr(
                        self.global_config,
                        "bound_completion_materialize_reader",
                        False,
                    ),
                    enable_provenance_witness_repair=getattr(
                        self.global_config,
                        "enable_provenance_witness_repair",
                        False,
                    ),
                    provenance_witness_top_k=getattr(
                        self.global_config,
                        "provenance_witness_top_k",
                        6,
                    ),
                    provenance_witness_max_calls=getattr(
                        self.global_config,
                        "provenance_witness_max_calls",
                        2,
                    ),
                )
                info = dict(info or {})
                semijoin_info = dict(info.get("schema_bound_semijoin") or {})
                if bind_pipeline_enabled and semijoin_info.get("fallback_used"):
                    semijoin_info["fallback_state_restored"] = True
                    semijoin_info["fallback_input"] = "pre_schema_relation_snapshot"
                    info["schema_bound_semijoin"] = semijoin_info
                latency_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
                info["operator_latency_ms"] = round(latency_ms, 6)
                semijoin_info = dict(info.get("schema_bound_semijoin") or {})
                if semijoin_info.get("applied"):
                    info["physical_operator"] = "schema_bound_role_index_semijoin"
                elif int(info.get("upstream_bind_fact_count", 0) or 0) > 0:
                    info["physical_operator"] = "schema_tuple_bind_relation_fact_join_fallback"
                else:
                    info["physical_operator"] = "relation_fact_index_join"
                info["executed_by_lazy_optimizer"] = bool(
                    getattr(self.global_config, "enable_adaptive_lazy_query_executor", False)
                )
                join_docs = [
                    self.chunk_embedding_store.get_row(
                        self.passage_node_keys[int(idx)]
                    )["content"]
                    for idx in join_doc_ids
                ]
                return {
                    "doc_ids": join_doc_ids,
                    "docs": join_docs,
                    "info": info,
                }

            paper_evidence_executor_enabled = bool(
                getattr(
                    self.global_config,
                    "enable_paper_evidence_query_executor",
                    False,
                )
            )
            adaptive_lazy_enabled = bool(
                getattr(self.global_config, "enable_adaptive_lazy_query_executor", False)
            )
            if paper_evidence_executor_enabled:
                relation_result = execute_relation_fact_join()
                paper_trace = dict(
                    (relation_result.get("info") or {}).get(
                        "paper_evidence_execution"
                    )
                    or {}
                )
                operator_outputs = {
                    "query_schema_memory": {
                        "doc_ids": [],
                        "info": {
                            "enabled": False,
                            "applied": False,
                            "reason": "owned_by_paper_evidence_executor",
                            "candidate_count": 0,
                            "candidate_fact_tuple_count": 0,
                            "candidate_items": [],
                            "executed_by_lazy_optimizer": False,
                        },
                    },
                    "relation_fact_join": relation_result,
                }
                adaptive_query_execution_info = {
                    "enabled": True,
                    "version": "paper_evidence_executor_v1",
                    "planning_scope": "binding_obligation_evidence_actions",
                    "runtime_reoptimization_ready": True,
                    "runtime_reoptimization_performed": bool(
                        paper_trace.get("operator_trace")
                    ),
                    "selected_plan_expression": str(
                        paper_trace.get("policy") or ""
                    ),
                    "executed_operators": [
                        str(row.get("operator") or "")
                        for row in paper_trace.get("operator_trace") or []
                    ],
                    "skipped_operators": [
                        "LEGACY_SCHEMA_PROBE",
                        "LEGACY_RELATION_FACT_JOIN",
                    ],
                    "output_path": paper_trace.get("output_path"),
                    "stop_reason": paper_trace.get("stop_reason"),
                    "cost_ledger": paper_trace.get("cost_ledger"),




                    "logical_plan": dict(
                        paper_trace.get("logical_plan") or {}
                    ),
                }
            elif adaptive_lazy_enabled:
                base_operator_docs = [
                    self.chunk_embedding_store.get_row(
                        self.passage_node_keys[int(idx)]
                    )["content"]
                    for idx in list(operator_context["doc_ids"])[
                        : max(1, int(self.global_config.qa_top_k))
                    ]
                ]
                query_coverage = lexical_query_coverage(query, base_operator_docs)
                relation_operator_enabled = bool(
                    getattr(
                        self.global_config,
                        "enable_relation_fact_join_retrieval",
                        False,
                    )
                    and (getattr(self, "relation_fact_join_index", {}) or {}).get(
                        "items"
                    )
                )
                logical_plan_preview = (
                    compile_stage_aware_fact_join_plan(
                        query,
                        enable_bound_completion=bool(
                            getattr(
                                self.global_config,
                                "enable_bound_completion_fact_join",
                                False,
                            )
                        ),
                        enable_role_directed_join=bool(
                            getattr(
                                self.global_config,
                                "enable_role_directed_fact_join",
                                True,
                            )
                        ),
                        enable_schema_plan_coverage_v7=bool(getattr(
                            self.global_config,
                            "paper_evidence_schema_plan_coverage_v7",
                            False,
                        )),
                        enable_general_plan_coverage_v8=bool(getattr(
                            self.global_config,
                            "paper_evidence_general_plan_coverage_v8",
                            False,
                        )),
                        enable_schema_exact_nested_join_v9=bool(getattr(
                            self.global_config,
                            "paper_evidence_schema_exact_nested_join_v9",
                            False,
                        )),
                    )
                    if getattr(
                        self.global_config,
                        "enable_stage_aware_fact_join_plan",
                        False,
                    )
                    else {}
                )
                logical_plan_compiled = bool(
                    logical_plan_preview.get("compile_status") == "compiled"
                    and int(logical_plan_preview.get("operator_count", 0) or 0) > 0
                )
                schema_catalog_v2_enabled = bool(
                    getattr(self.global_config, "enable_schema_catalog_v2", False)
                )
                active_signatures = (
                    self._query_schema_memory_active_signatures_v2(
                        query_schema_memory_facts,
                        query,
                    )
                    if schema_catalog_v2_enabled
                    else self._query_schema_memory_active_signatures(
                        query_schema_memory_facts
                    )
                )
                schema_index = getattr(
                    self,
                    "query_schema_memory_v2_schema_to_chunks"
                    if schema_catalog_v2_enabled
                    else "query_schema_memory_schema_to_chunks",
                    {},
                ) or {}
                schema_catalog = (
                    getattr(self, "query_schema_memory_v2_catalog", {}) or {}
                    if schema_catalog_v2_enabled
                    else {}
                )
                schema_operator_enabled = bool(
                    getattr(
                        self.global_config,
                        "enable_query_schema_memory",
                        False,
                    )
                    and schema_index
                    and active_signatures
                )
                plan_execution_policy = build_logical_plan_execution_policy(
                    logical_plan=logical_plan_preview,
                    schema_probe_enabled=schema_operator_enabled,
                    relation_fact_join_enabled=relation_operator_enabled,
                )
                required_plan_operators = tuple(
                    plan_execution_policy.get("required_seed_operators") or ()
                )
                logical_plan_preview = {
                    **logical_plan_preview,
                    "physical_execution_policy": plan_execution_policy,
                }
                schema_plan_seed_signal = float(
                    plan_execution_policy.get("physical_seed_operator")
                    == "SCHEMA_PROBE"
                )
                relation_plan_fallback_signal = float(
                    plan_execution_policy.get("conditional_fallback_operator")
                    == "RELATION_FACT_JOIN"
                    or plan_execution_policy.get("physical_seed_operator")
                    == "RELATION_FACT_JOIN"
                )
                relation_index = getattr(self, "relation_fact_join_index", {}) or {}
                schema_scan_rows = estimate_index_scan_rows(
                    active_signatures,
                    schema_index,
                    per_key_cap=getattr(
                        self.global_config,
                        "query_schema_memory_max_chunks_per_schema",
                        32,
                    ),
                )
                relation_access = estimate_relation_fact_join_access(
                    question=query,
                    top_k_facts=top_k_facts,
                    candidate_facts=(rerank_log or {}).get(
                        "facts_before_rerank",
                        [],
                    ),
                    relation_index=relation_index,
                    max_facts_per_key=getattr(
                        self.global_config,
                        "relation_fact_join_max_facts_per_key",
                        60,
                    ),
                    max_key_fanout=getattr(
                        self.global_config,
                        "relation_fact_join_max_key_fanout",
                        600,
                    ),
                )
                relation_scan_rows = int(
                    relation_access.get("estimated_scan_rows", 0) or 0
                )
                deficit = max(0.0, 1.0 - float(query_coverage))
                bridge_strength = self._query_schema_memory_bridge_strength(query)
                bridge_signal = min(1.0, float(bridge_strength) / 6.0)
                active_signal = 1.0 if active_signatures else 0.0
                schema_capacity = max(
                    1,
                    len(active_signatures)
                    * max(
                        1,
                        int(
                            getattr(
                                self.global_config,
                                "query_schema_memory_max_chunks_per_schema",
                                32,
                            )
                        ),
                    ),
                )
                relation_probe_count = sum(
                    int(value or 0)
                    for value in (
                        relation_access.get("probe_counts") or {}
                    ).values()
                )
                relation_capacity = max(
                    1,
                    relation_probe_count
                    * max(
                        1,
                        int(
                            getattr(
                                self.global_config,
                                "relation_fact_join_max_facts_per_key",
                                60,
                            )
                        ),
                    ),
                )
                relation_active_signal = 1.0 if relation_probe_count > 0 else 0.0
                coverage_threshold = float(
                    getattr(
                        self.global_config,
                        "adaptive_lazy_query_executor_deficit_coverage_threshold",
                        0.72,
                    )
                )
                bridge_deficit = bool(
                    has_bridge_cues(query)
                    and query_coverage < coverage_threshold
                )
                risk_context = {
                    "query_coverage": round(float(query_coverage), 6),
                    "bridge_deficit": bridge_deficit,
                    "bridge_strength": int(bridge_strength),
                    "question_token_count": len(str(query or "").split()),
                    "base_doc_count": len(base_operator_docs),
                    "avg_base_doc_chars": round(
                        sum(len(str(doc or "")) for doc in base_operator_docs)
                        / float(max(1, len(base_operator_docs))),
                        3,
                    ),
                    "active_schema_count": len(active_signatures),
                    "schema_catalog_version": (
                        "v2" if schema_catalog_v2_enabled else "v1"
                    ),
                    "schema_catalog_hit_count": sum(
                        bool(schema_catalog.get(signature))
                        for signature in active_signatures
                    ),
                    "schema_catalog_mean_selectivity": round(
                        sum(
                            float(
                                (schema_catalog.get(signature) or {}).get(
                                    "selectivity",
                                    0.0,
                                )
                                or 0.0
                            )
                            for signature in active_signatures
                        )
                        / float(max(1, len(active_signatures))),
                        8,
                    ),
                    "schema_estimated_scan_rows": schema_scan_rows,
                    "schema_capacity": schema_capacity,
                    "relation_probe_count": relation_probe_count,
                    "relation_estimated_scan_rows": relation_scan_rows,
                    "relation_capacity": relation_capacity,
                    "logical_plan_compiled": logical_plan_compiled,
                    "logical_plan_kind": str(
                        logical_plan_preview.get("plan_kind") or ""
                    ),
                    "logical_plan_operator_count": int(
                        logical_plan_preview.get("operator_count", 0) or 0
                    ),
                    "logical_plan_answer_operator": str(
                        logical_plan_preview.get("answer_operator") or ""
                    ),
                    "logical_plan_requires_relation_join": bool(
                        plan_execution_policy.get(
                            "relation_join_unconditionally_required"
                        )
                    ),
                    "logical_plan_physical_seed_operator": str(
                        plan_execution_policy.get("physical_seed_operator") or ""
                    ),
                    "logical_plan_conditional_fallback_operator": str(
                        plan_execution_policy.get(
                            "conditional_fallback_operator"
                        )
                        or ""
                    ),
                }
                incumbent_admission_enabled = bool(
                    getattr(
                        self.global_config,
                        "enable_adaptive_incumbent_preserving_admission",
                        False,
                    )
                )
                reader_risk_model = load_adaptive_reader_risk_model(
                    str(
                        getattr(
                            self.global_config,
                            "adaptive_incumbent_admission_model_path",
                            "",
                        )
                    )
                ) if incumbent_admission_enabled else None
                operator_value_enabled = bool(
                    getattr(
                        self.global_config,
                        "enable_adaptive_operator_value_model",
                        False,
                    )
                )
                operator_value_model = load_adaptive_operator_value_model(
                    str(
                        getattr(
                            self.global_config,
                            "adaptive_operator_value_model_path",
                            "",
                        )
                    )
                ) if operator_value_enabled else None

                def assess_stop_risk(state, remaining, executed):
                    return assess_adaptive_stop_risk(
                        context=risk_context,
                        state=state,
                        remaining_operators=remaining,
                        executed_operators=executed,
                        model=reader_risk_model,
                        min_support=getattr(
                            self.global_config,
                            "adaptive_incumbent_admission_min_model_support",
                            25,
                        ),
                        min_confidence=getattr(
                            self.global_config,
                            "adaptive_incumbent_admission_min_confidence",
                            0.25,
                        ),
                    )

                def assess_operator_value(state, operator):
                    return assess_adaptive_operator_value(
                        context=risk_context,
                        state=state,
                        operator=operator,
                        model=operator_value_model,
                        min_operator_support=getattr(
                            self.global_config,
                            "adaptive_operator_value_min_operator_support",
                            25,
                        ),
                        min_feature_support=getattr(
                            self.global_config,
                            "adaptive_operator_value_min_feature_support",
                            20,
                        ),
                        min_confidence=getattr(
                            self.global_config,
                            "adaptive_operator_value_min_confidence",
                            0.25,
                        ),
                        resolution_floor=getattr(
                            self.global_config,
                            "adaptive_operator_value_resolution_floor",
                            0.50,
                        ),
                        min_utility_multiplier=getattr(
                            self.global_config,
                            "adaptive_operator_value_min_utility_multiplier",
                            0.35,
                        ),
                        max_utility_multiplier=getattr(
                            self.global_config,
                            "adaptive_operator_value_max_utility_multiplier",
                            1.50,
                        ),
                    )

                def assess_relation_plan_obligation(operator_info):
                    semantic_execution = dict(
                        operator_info.get("semantic_plan_execution")
                        or (
                            operator_info.get("multi_hop_join") or {}
                        ).get("semantic_plan_execution")
                        or {}
                    )
                    if semantic_execution:
                        return semantic_execution
                    return {
                        "required": bool(logical_plan_compiled),
                        "completed": False,
                        "status": "semantic_execution_result_missing",
                        "terminal_binding_count": 0,
                    }

                lazy_operators = [
                    build_lazy_operator_spec(
                        operator="SCHEMA_PROBE",
                        source="query_schema_memory",
                        access_path=(
                            "typed_predicate_catalog_index"
                            if schema_catalog_v2_enabled
                            else "schema_signature_hash_index"
                        ),
                        enabled=schema_operator_enabled,
                        estimated_utility=(
                            0.30
                            + 0.26 * deficit
                            + 0.14 * bridge_signal
                            + 0.12 * active_signal
                            + 0.18 * schema_plan_seed_signal
                        ),
                        predicted_risk=(
                            0.08
                            + 0.28 * min(1.0, schema_scan_rows / float(schema_capacity))
                        ),
                        estimated_scan_rows=schema_scan_rows,
                        estimated_tokens=120,
                        startup_cost=0.035,
                        execute=execute_schema_probe,
                        terminal_candidate_fields=("candidate_fact_tuple_count",),
                        terminal_requires_novel_candidate=True,
                        terminal_candidates_resolve_residual=False,
                        incumbent_priority=0,
                    ),
                    build_lazy_operator_spec(
                        operator="RELATION_FACT_JOIN",
                        source="relation_fact_join",
                        access_path="relation_schema_inverted_index",
                        enabled=bool(
                            relation_operator_enabled
                        ),
                        estimated_utility=(
                            0.28
                            + 0.30 * deficit
                            + 0.16 * bridge_signal
                            + 0.10 * relation_active_signal
                            + 0.08 * relation_plan_fallback_signal
                        ),
                        predicted_risk=(
                            0.12
                            + 0.32
                            * min(1.0, relation_scan_rows / float(relation_capacity))
                        ),
                        estimated_scan_rows=relation_scan_rows,
                        estimated_tokens=180,
                        startup_cost=0.060,
                        execute=execute_relation_fact_join,
                        terminal_candidate_fields=("accepted_count",),
                        terminal_candidates_resolve_residual=False,
                        plan_obligation_assessor=(
                            assess_relation_plan_obligation
                        ),
                        requires_executed_operators=(
                            ("SCHEMA_PROBE",)
                            if schema_operator_enabled
                            and (
                                plan_execution_policy.get(
                                    "physical_seed_operator"
                                )
                                == "SCHEMA_PROBE"
                                or (
                                    not logical_plan_compiled
                                    and getattr(
                                        self.global_config,
                                        "enable_schema_bind_fact_join",
                                        False,
                                    )
                                )
                            )
                            else ()
                        ),
                        incumbent_priority=1,
                    ),
                ]
                operator_outputs, adaptive_query_execution_info = (
                    execute_adaptive_query_plan(
                        question=query,
                        base_docs=base_operator_docs,
                        operators=lazy_operators,
                        bridge_deficit=bridge_deficit,
                        max_steps=getattr(
                            self.global_config,
                            "adaptive_lazy_query_executor_max_steps",
                            2,
                        ),
                        token_budget=getattr(
                            self.global_config,
                            "cost_based_admission_token_budget",
                            900,
                        ),
                        compute_budget=getattr(
                            self.global_config,
                            "adaptive_lazy_query_executor_compute_budget",
                            1.0,
                        ),
                        min_objective=getattr(
                            self.global_config,
                            "adaptive_lazy_query_executor_min_objective",
                            0.05,
                        ),
                        max_predicted_risk=getattr(
                            self.global_config,
                            "adaptive_lazy_query_executor_max_predicted_risk",
                            0.55,
                        ),
                        token_weight=getattr(
                            self.global_config,
                            "cost_based_admission_token_weight",
                            0.12,
                        ),
                        compute_weight=getattr(
                            self.global_config,
                            "cost_based_admission_compute_weight",
                            0.10,
                        ),
                        risk_weight=getattr(
                            self.global_config,
                            "cost_based_admission_risk_weight",
                            0.42,
                        ),
                        risk_context=risk_context,
                        stop_risk_assessor=(
                            assess_stop_risk if incumbent_admission_enabled else None
                        ),
                        incumbent_preserving_admission=incumbent_admission_enabled,
                        incumbent_admission_shadow_only=getattr(
                            self.global_config,
                            "adaptive_incumbent_admission_shadow_only",
                            True,
                        ),
                        max_skip_harm_probability=getattr(
                            self.global_config,
                            "adaptive_incumbent_admission_max_harm_probability",
                            0.025,
                        ),
                        max_reader_change_probability=getattr(
                            self.global_config,
                            "adaptive_incumbent_admission_max_reader_change_probability",
                            0.12,
                        ),
                        operator_value_assessor=(
                            assess_operator_value if operator_value_enabled else None
                        ),
                        min_fallback_productivity=getattr(
                            self.global_config,
                            "adaptive_operator_value_min_fallback_productivity",
                            0.12,
                        ),
                        enable_risk_bounded_optimizer=getattr(
                            self.global_config,
                            "enable_adaptive_risk_bounded_optimizer",
                            False,
                        ),
                        risk_bounded_plan_horizon=getattr(
                            self.global_config,
                            "adaptive_risk_bounded_plan_horizon",
                            2,
                        ),
                        risk_bounded_beam_width=getattr(
                            self.global_config,
                            "adaptive_risk_bounded_beam_width",
                            8,
                        ),
                        risk_bounded_uncertainty=getattr(
                            self.global_config,
                            "adaptive_risk_bounded_uncertainty",
                            0.04,
                        ),
                        risk_bounded_max_plan_risk=getattr(
                            self.global_config,
                            "adaptive_risk_bounded_max_plan_risk",
                            0.55,
                        ),
                        logical_plan=logical_plan_preview,
                        required_plan_operators=required_plan_operators,
                        plan_execution_policy=plan_execution_policy,
                    )
                )
            else:
                schema_result = execute_schema_probe()
                relation_result = execute_relation_fact_join()
                operator_outputs = {
                    "query_schema_memory": schema_result,
                    "relation_fact_join": relation_result,
                }
                adaptive_query_execution_info = {
                    "enabled": False,
                    "planning_scope": "legacy_eager_operator_execution",
                    "runtime_reoptimization_ready": False,
                    "runtime_reoptimization_performed": False,
                    "selected_plan_expression": (
                        "EAGER(SCHEMA_PROBE, RELATION_FACT_JOIN)"
                    ),
                    "executed_operators": [
                        "SCHEMA_PROBE",
                        "RELATION_FACT_JOIN",
                    ],
                    "skipped_operators": [],
                }

            sorted_doc_ids = operator_context["doc_ids"]
            sorted_doc_scores = operator_context["doc_scores"]
            schema_result = operator_outputs.get("query_schema_memory") or {}
            query_schema_memory_info = dict(schema_result.get("info") or {
                "enabled": bool(
                    getattr(self.global_config, "enable_query_schema_memory", False)
                ),
                "applied": False,
                "reason": "skipped_by_adaptive_query_executor",
                "candidate_count": 0,
                "candidate_fact_tuple_count": 0,
                "candidate_items": [],
                "executed_by_lazy_optimizer": False,
            })
            relation_result = operator_outputs.get("relation_fact_join") or {}
            relation_fact_join_doc_ids = list(relation_result.get("doc_ids") or [])
            relation_fact_join_docs = list(relation_result.get("docs") or [])
            relation_fact_join_info = dict(relation_result.get("info") or {
                "enabled": bool(
                    getattr(
                        self.global_config,
                        "enable_relation_fact_join_retrieval",
                        False,
                    )
                ),
                "applied": False,
                "reason": "skipped_by_adaptive_query_executor",
                "candidate_count": 0,
                "accepted_count": 0,
                "accepted": [],
                "candidate_table_rows": [],
                "candidate_table_row_count": 0,
                "candidate_table_doc": "",
                "reader_candidate_doc_ids": [],
                "executed_by_lazy_optimizer": False,
            })
            paper_evidence_execution_info = dict(
                relation_fact_join_info.get("paper_evidence_execution") or {}
            )
            typed_fact_join_shadow_info = build_typed_fact_join_shadow_optimizer(
                question=query,
                sorted_doc_ids=sorted_doc_ids,
                top_k_facts=top_k_facts,
                candidate_facts=(rerank_log or {}).get("facts_before_rerank", []),
                relation_index=getattr(self, "relation_fact_join_index", {}),
                get_doc_content=lambda doc_idx: self.chunk_embedding_store.get_row(
                    self.passage_node_keys[int(doc_idx)]
                )["content"],
                gold_docs=(gold_docs[q_idx] if gold_docs is not None else None),
                enabled=getattr(self.global_config, "enable_typed_fact_join_shadow_optimizer", False),
                candidate_top_k=getattr(self.global_config, "typed_fact_join_shadow_candidate_top_k", 200),
                max_anchor_items=getattr(self.global_config, "typed_fact_join_shadow_max_anchor_items", 48),
                max_second_hop_items=getattr(
                    self.global_config,
                    "typed_fact_join_shadow_max_second_hop_items",
                    64,
                ),
                max_plans=getattr(self.global_config, "typed_fact_join_shadow_max_plans", 12),
                max_key_fanout=getattr(self.global_config, "typed_fact_join_shadow_max_key_fanout", 800),
                min_utility=getattr(self.global_config, "typed_fact_join_shadow_min_utility", 0.34),
                max_risk=getattr(self.global_config, "typed_fact_join_shadow_max_risk", 0.70),
                min_net_utility=getattr(self.global_config, "typed_fact_join_shadow_min_net_utility", -0.06),
                token_budget=getattr(self.global_config, "typed_fact_join_shadow_token_budget", 1200),
                token_weight=getattr(self.global_config, "typed_fact_join_shadow_token_weight", 0.08),
                compute_weight=getattr(self.global_config, "typed_fact_join_shadow_compute_weight", 0.05),
                risk_weight=getattr(self.global_config, "typed_fact_join_shadow_risk_weight", 0.36),
            )
            shadow_accepted_count = int(typed_fact_join_shadow_info.get("accepted_plan_count", 0) or 0)
            if shadow_accepted_count <= 0:
                shadow_count_bucket = "0"
            elif shadow_accepted_count <= 2:
                shadow_count_bucket = "1-2"
            elif shadow_accepted_count <= 5:
                shadow_count_bucket = "3-5"
            elif shadow_accepted_count <= 10:
                shadow_count_bucket = "6-10"
            elif shadow_accepted_count <= 50:
                shadow_count_bucket = "11-50"
            elif shadow_accepted_count <= 100:
                shadow_count_bucket = "51-100"
            elif shadow_accepted_count <= 200:
                shadow_count_bucket = "101-200"
            else:
                shadow_count_bucket = "201+"
            typed_fact_join_shadow_counts[
                f"{typed_fact_join_shadow_info.get('reason', 'unavailable')}|accepted={shadow_count_bucket}"
            ] += 1
            selective_support_join_doc_ids, selective_support_join_docs, selective_support_join_info = (
                select_selective_support_join_evidence(
                    question=query,
                    sorted_doc_ids=sorted_doc_ids,
                    typed_fact_join_shadow=typed_fact_join_shadow_info,
                    relation_fact_join=relation_fact_join_info,
                    get_doc_content=lambda doc_idx: self.chunk_embedding_store.get_row(
                        self.passage_node_keys[int(doc_idx)]
                    )["content"],
                    qa_top_k=self.global_config.qa_top_k,
                    enabled=getattr(
                        self.global_config,
                        "enable_selective_support_join_optimizer",
                        False,
                    ),
                    bridge_only=getattr(
                        self.global_config,
                        "selective_support_join_bridge_only",
                        True,
                    ),
                    max_append=getattr(
                        self.global_config,
                        "selective_support_join_max_append",
                        1,
                    ),
                    max_candidate_plans=getattr(
                        self.global_config,
                        "selective_support_join_max_candidate_plans",
                        12,
                    ),
                    max_original_rank=getattr(
                        self.global_config,
                        "selective_support_join_max_original_rank",
                        80,
                    ),
                    min_plan_utility=getattr(
                        self.global_config,
                        "selective_support_join_min_plan_utility",
                        0.48,
                    ),
                    min_plan_net_utility=getattr(
                        self.global_config,
                        "selective_support_join_min_plan_net_utility",
                        0.22,
                    ),
                    high_conf_net_utility=getattr(
                        self.global_config,
                        "selective_support_join_high_conf_net_utility",
                        0.42,
                    ),
                    max_plan_risk=getattr(
                        self.global_config,
                        "selective_support_join_max_plan_risk",
                        0.48,
                    ),
                    min_missing_gain=getattr(
                        self.global_config,
                        "selective_support_join_min_missing_gain",
                        0.04,
                    ),
                    min_relation_overlap=getattr(
                        self.global_config,
                        "selective_support_join_min_relation_overlap",
                        0.08,
                    ),
                    min_endpoint_overlap=getattr(
                        self.global_config,
                        "selective_support_join_min_endpoint_overlap",
                        0.22,
                    ),
                    min_schema_selectivity=getattr(
                        self.global_config,
                        "selective_support_join_min_schema_selectivity",
                        0.62,
                    ),
                    min_admission_score=getattr(
                        self.global_config,
                        "selective_support_join_min_admission_score",
                        0.28,
                    ),
                    max_accepted_plan_count=getattr(
                        self.global_config,
                        "selective_support_join_max_accepted_plan_count",
                        96,
                    ),
                    require_two_hop_for_bridge=getattr(
                        self.global_config,
                        "selective_support_join_require_two_hop_for_bridge",
                        True,
                    ),
                    require_deficit=getattr(
                        self.global_config,
                        "selective_support_join_require_deficit",
                        True,
                    ),
                    min_current_query_coverage=getattr(
                        self.global_config,
                        "selective_support_join_min_current_query_coverage",
                        0.62,
                    ),
                    min_current_relation_coverage=getattr(
                        self.global_config,
                        "selective_support_join_min_current_relation_coverage",
                        0.50,
                    ),
                    block_long_context=getattr(
                        self.global_config,
                        "selective_support_join_block_long_context",
                        True,
                    ),
                    long_context_avg_chars=getattr(
                        self.global_config,
                        "selective_support_join_long_context_avg_chars",
                        1400,
                    ),
                    token_budget=getattr(
                        self.global_config,
                        "selective_support_join_token_budget",
                        900,
                    ),
                )
            )
            selective_support_join_counts[
                f"{selective_support_join_info.get('reason', 'unavailable')}|"
                f"{selective_support_join_info.get('accepted_count', 0)}"
            ] += 1
            support_rerank_top_k = min(
                max(
                    self.global_config.qa_top_k,
                    getattr(self.global_config, "support_aware_rerank_top_k", 50),
                ),
                len(sorted_doc_ids),
            )
            support_candidate_docs = [
                self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                for idx in sorted_doc_ids[:support_rerank_top_k]
            ]
            sorted_doc_ids, sorted_doc_scores, support_rerank_info = support_aware_rerank_docs(
                query=query,
                sorted_doc_ids=sorted_doc_ids,
                sorted_doc_scores=sorted_doc_scores,
                candidate_docs=support_candidate_docs,
                enabled=getattr(self.global_config, "enable_support_aware_rerank", False),
                rerank_top_k=getattr(self.global_config, "support_aware_rerank_top_k", 50),
                target_k=getattr(self.global_config, "support_aware_rerank_target_k", self.global_config.qa_top_k),
                weight=getattr(self.global_config, "support_aware_rerank_weight", 0.55),
                min_support_score=getattr(self.global_config, "support_aware_rerank_min_score", 0.50),
                max_promotions=getattr(self.global_config, "support_aware_rerank_max_promotions", 1),
                protect_top1=getattr(self.global_config, "support_aware_rerank_protect_top1", True),
            )

            dual_channel_projection_info = {
                "enabled": bool(getattr(self.global_config, "enable_dual_channel_evidence_projection", False)),
                "applied": False,
                "reason": "disabled",
                "promotions": [],
                "promoted_to_top2": 0,
                "promoted_to_top5": 0,
            }
            dual_channel_projection_docs = []
            if getattr(self.global_config, "enable_dual_channel_evidence_projection", False):
                projection_candidate_top_k = max(
                    self.global_config.qa_top_k,
                    int(getattr(self.global_config, "dual_channel_projection_candidate_top_k", 80)),
                )
                projection_base_top_n = min(projection_candidate_top_k, len(sorted_doc_ids))
                projection_base_docs = [
                    self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                    for idx in sorted_doc_ids[:projection_base_top_n]
                ]
                projection_trigger = assess_dual_channel_projection_trigger(
                    question=query,
                    docs=projection_base_docs,
                    qa_top_k=self.global_config.qa_top_k,
                    enabled=True,
                    bridge_only=getattr(
                        self.global_config,
                        "dual_channel_projection_bridge_only",
                        True,
                    ),
                    trigger_min_query_coverage=getattr(
                        self.global_config,
                        "dual_channel_projection_trigger_min_query_coverage",
                        0.50,
                    ),
                    trigger_min_bridge_docs=getattr(
                        self.global_config,
                        "dual_channel_projection_trigger_min_bridge_docs",
                        2,
                    ),
                    shadow_when_sufficient=getattr(
                        self.global_config,
                        "dual_channel_projection_shadow_when_sufficient",
                        True,
                    ),
                )
                dual_channel_projection_info.update(projection_trigger)
                if projection_trigger.get("should_project"):
                    dense_doc_ids, dense_doc_scores = self.dense_passage_retrieval(query)
                    dense_top_n = min(projection_candidate_top_k, len(dense_doc_ids))
                    dense_docs = [
                        self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                        for idx in dense_doc_ids[:dense_top_n]
                    ]
                    sorted_doc_ids, sorted_doc_scores, dual_channel_projection_info = (
                        apply_dual_channel_retrieval_projection(
                            question=query,
                            base_doc_ids=sorted_doc_ids,
                            base_doc_scores=sorted_doc_scores,
                            base_docs=projection_base_docs,
                            candidate_doc_ids=dense_doc_ids[:dense_top_n],
                            candidate_doc_scores=dense_doc_scores[:dense_top_n],
                            candidate_docs=dense_docs,
                            qa_top_k=self.global_config.qa_top_k,
                            enabled=True,
                            bridge_only=getattr(
                                self.global_config,
                                "dual_channel_projection_bridge_only",
                                True,
                            ),
                            candidate_top_k=projection_candidate_top_k,
                            max_promotions=getattr(
                                self.global_config,
                                "dual_channel_projection_max_promotions",
                                1,
                            ),
                            protect_top_k=getattr(
                                self.global_config,
                                "dual_channel_projection_protect_top_k",
                                1,
                            ),
                            min_projection_score=getattr(
                                self.global_config,
                                "dual_channel_projection_min_score",
                                0.66,
                            ),
                            min_query_overlap=getattr(
                                self.global_config,
                                "dual_channel_projection_min_query_overlap",
                                0.18,
                            ),
                            min_missing_gain=getattr(
                                self.global_config,
                                "dual_channel_projection_min_missing_gain",
                                0.10,
                            ),
                            min_bridge_link=getattr(
                                self.global_config,
                                "dual_channel_projection_min_bridge_link",
                                0.06,
                            ),
                            shadow_top_k=getattr(
                                self.global_config,
                                "dual_channel_projection_shadow_top_k",
                                8,
                            ),
                            shadow_when_sufficient=getattr(
                                self.global_config,
                                "dual_channel_projection_shadow_when_sufficient",
                                True,
                            ),
                            projection_mode=getattr(
                                self.global_config,
                                "dual_channel_projection_mode",
                                "append_only_or_top5",
                            ),
                            trigger_min_query_coverage=getattr(
                                self.global_config,
                                "dual_channel_projection_trigger_min_query_coverage",
                                0.50,
                            ),
                            trigger_min_bridge_docs=getattr(
                                self.global_config,
                                "dual_channel_projection_trigger_min_bridge_docs",
                                2,
                            ),
                            reader_soft_append_enabled=getattr(
                                self.global_config,
                                "dual_channel_projection_reader_soft_append_enabled",
                                False,
                            ),
                            reader_soft_append_require_insufficient=getattr(
                                self.global_config,
                                "dual_channel_projection_reader_soft_append_require_insufficient",
                                True,
                            ),
                            reader_soft_append_min_score=getattr(
                                self.global_config,
                                "dual_channel_projection_reader_soft_append_min_score",
                                0.42,
                            ),
                            reader_soft_append_min_query_overlap=getattr(
                                self.global_config,
                                "dual_channel_projection_reader_soft_append_min_query_overlap",
                                0.06,
                            ),
                            reader_soft_append_min_missing_gain=getattr(
                                self.global_config,
                                "dual_channel_projection_reader_soft_append_min_missing_gain",
                                0.05,
                            ),
                            reader_soft_append_min_bridge_link=getattr(
                                self.global_config,
                                "dual_channel_projection_reader_soft_append_min_bridge_link",
                                0.08,
                            ),
                        )
                    )
                promoted_doc_ids = list(
                    dual_channel_projection_info.get("reader_candidate_doc_ids", []) or []
                )
                if not promoted_doc_ids:
                    promoted_doc_ids = [
                    item.get("doc_id")
                    for item in dual_channel_projection_info.get("promotions", []) or []
                    if item.get("doc_id") is not None
                    ]
                seen_projection_doc_ids = set()
                promoted_doc_ids = [
                    doc_id
                    for doc_id in promoted_doc_ids
                    if doc_id is not None and not (
                        int(doc_id) in seen_projection_doc_ids
                        or seen_projection_doc_ids.add(int(doc_id))
                    )
                ]
                dual_channel_projection_docs = [
                    self.chunk_embedding_store.get_row(self.passage_node_keys[int(doc_id)])["content"]
                    for doc_id in promoted_doc_ids
                ]

            iterative_retrieval_info = {
                "enabled": bool(getattr(self.global_config, "enable_iterative_retrieval", False)),
                "applied": False,
                "reason": "disabled",
                "iterations": [],
                "promoted_to_top2": 0,
                "promoted_to_top5": 0,
            }
            if getattr(self.global_config, "enable_iterative_retrieval", False):
                bridge_cues = has_bridge_cues(query)
                bridge_only = bool(getattr(self.global_config, "iterative_retrieval_bridge_only", True))
                max_steps = max(1, int(getattr(self.global_config, "iterative_retrieval_max_steps", 2)))
                target_k = max(1, int(getattr(self.global_config, "iterative_retrieval_target_k", self.global_config.qa_top_k)))
                max_promotions = max(0, int(getattr(self.global_config, "iterative_retrieval_max_promotions", 1)))
                protect_top1 = bool(getattr(self.global_config, "iterative_retrieval_protect_top1", True))
                protect_top_k = max(0, int(getattr(self.global_config, "iterative_retrieval_protect_top_k", 2)))
                expansion_weight = float(getattr(self.global_config, "iterative_retrieval_expansion_weight", 0.72))
                min_expansion_score = float(getattr(self.global_config, "iterative_retrieval_min_expansion_score", 0.40))
                iterative_retrieval_info.update({
                    "reason": "started",
                    "bridge_cues": bridge_cues,
                    "bridge_only": bridge_only,
                    "max_steps": max_steps,
                    "target_k": target_k,
                    "max_promotions": max_promotions,
                    "protect_top1": protect_top1,
                    "protect_top_k": protect_top_k,
                    "expansion_weight": expansion_weight,
                    "min_expansion_score": min_expansion_score,
                })

                if bridge_only and not bridge_cues:
                    iterative_retrieval_info["reason"] = "no_bridge_cues"
                else:
                    for step_idx in range(max_steps):
                        current_docs = [
                            self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                            for idx in sorted_doc_ids[:target_k]
                        ]
                        sufficiency = assess_evidence_sufficiency(
                            question=query,
                            docs=current_docs,
                            qa_top_k=target_k,
                            bridge_cues=bridge_cues,
                            bridge_threshold=getattr(self.global_config, "iterative_retrieval_bridge_threshold", 0.62),
                            non_bridge_threshold=getattr(self.global_config, "iterative_retrieval_non_bridge_threshold", 0.48),
                            min_query_coverage=getattr(self.global_config, "iterative_retrieval_min_query_coverage", 0.50),
                            min_best_sentence_score=getattr(
                                self.global_config,
                                "iterative_retrieval_min_best_sentence_score",
                                0.24,
                            ),
                        )
                        step_info = {
                            "step": step_idx + 1,
                            "sufficiency": sufficiency,
                        }
                        iterative_retrieval_info["iterations"].append(step_info)

                        if sufficiency.get("sufficient"):
                            iterative_retrieval_info["reason"] = "sufficient"
                            break
                        if step_idx >= max_steps - 1:
                            iterative_retrieval_info["reason"] = "max_steps_reached"
                            break

                        followup_query = build_followup_query(
                            question=query,
                            docs=current_docs,
                            max_titles=target_k,
                            max_snippets=getattr(self.global_config, "iterative_retrieval_max_followup_snippets", 3),
                        )
                        self.get_query_embeddings([followup_query])
                        expansion_fact_scores = self.get_fact_scores(followup_query)
                        expansion_top_k_fact_indices, expansion_top_k_facts, _ = self.rerank_facts(
                            followup_query,
                            expansion_fact_scores,
                        )
                        if self._uses_dense_retrieval_backend() or len(expansion_top_k_facts) == 0:
                            expansion_doc_ids, expansion_doc_scores = self.dense_passage_retrieval(followup_query)
                        else:
                            expansion_doc_ids, expansion_doc_scores = self.graph_search_with_fact_entities(
                                query=followup_query,
                                link_top_k=self.global_config.linking_top_k,
                                query_fact_scores=expansion_fact_scores,
                                top_k_facts=expansion_top_k_facts,
                                top_k_fact_indices=expansion_top_k_fact_indices,
                                passage_node_weight=self.global_config.passage_node_weight,
                            )
                        sorted_doc_ids, sorted_doc_scores, merge_info = merge_iterative_retrieval_results(
                            base_doc_ids=sorted_doc_ids,
                            base_doc_scores=sorted_doc_scores,
                            expansion_doc_ids=expansion_doc_ids,
                            expansion_doc_scores=expansion_doc_scores,
                            target_k=target_k,
                            max_promotions=max_promotions,
                            protect_top1=protect_top1,
                            protect_top_k=protect_top_k,
                            expansion_weight=expansion_weight,
                            min_expansion_score=min_expansion_score,
                        )
                        step_info.update({
                            "followup_query_preview": followup_query[:500],
                            "merge": merge_info,
                        })
                        iterative_retrieval_info["applied"] = True
                        iterative_retrieval_info["promoted_to_top2"] += merge_info.get("promoted_to_top2", 0)
                        iterative_retrieval_info["promoted_to_top5"] += merge_info.get("promoted_to_top5", 0)

            planned_retrieval_info = {
                "enabled": bool(getattr(self.global_config, "enable_planned_bridge_retrieval", False)),
                "applied": False,
                "reason": "disabled",
                "component_roles": ["planner", "retriever", "verifier", "merger"],
                "queries": [],
                "rounds": [],
                "round_count": 0,
                "planner_calls": 0,
                "retriever_calls": 0,
                "verifier_calls": 0,
                "merger_calls": 0,
                "retriever_dense_fallbacks": 0,
                "retriever_graph_calls": 0,
                "residual_deficit_enabled": bool(
                    getattr(self.global_config, "enable_planned_residual_deficit_queries", False)
                ),
                "residual_deficit_query_count": 0,
                "residual_deficit_round_count": 0,
                "residual_deficit_replaced_query_count": 0,
                "residual_deficit_append_only_doc_count": 0,
                "residual_deficit_append_only_round_count": 0,
                "residual_deficit_queries": [],
                "residual_deficit_replaced_queries": [],
                "residual_deficit_append_only_doc_ids": [],
                "residual_deficit_append_only_reject_reason_counts": {},
                "residual_deficit_gate_reject_reason_counts": {},
                "promoted_to_top2": 0,
                "promoted_to_top5": 0,
                "top2_override_applied": 0,
                "top2_override_rejected_count": 0,
                "bridge_slot_replace_accepted_count": 0,
                "bridge_slot_append_accepted_count": 0,
                "bridge_slot_append_doc_ids": [],
                "bridge_slot_reject_reason_counts": {},
                "object_memory_accepted_count": 0,
                "object_memory_rejected_count": 0,
                "object_memory_minimal_sentence_count": 0,
                "object_memory_reject_reason_counts": {},
                "object_memory_append_only_accepted_count": 0,
                "object_memory_append_only_rejected_count": 0,
                "object_memory_append_only_candidate_count": 0,
                "object_memory_append_only_doc_ids": [],
                "object_memory_append_only_reject_reason_counts": {},
                "query_schema_memory_candidate_count": int(
                    (query_schema_memory_info or {}).get("candidate_count", 0) or 0
                ),
                "query_schema_memory_merge_mode": (
                    (query_schema_memory_info or {}).get("merge_mode")
                ),
                "query_schema_memory_pairwise_candidate_count": 0,
                "query_schema_memory_pairwise_duplicate_count": 0,
                "query_schema_intervention_policy": str(
                    getattr(
                        self.global_config,
                        "query_schema_memory_intervention_policy",
                        "risk_aware",
                    )
                ),
                "query_schema_intervention_allow_top2_count": 0,
                "query_schema_intervention_append_only_count": 0,
                "query_schema_intervention_reject_count": 0,
                "query_schema_intervention_reason_counts": {},
                "schema_verified_prior_applied_count": 0,
                "schema_verified_prior_rescue_count": 0,
                "schema_verified_prior_bonus_total": 0.0,
                "schema_deficit_promotion_accepted_count": 0,
                "schema_deficit_promotion_rejected_count": 0,
                "schema_deficit_promotion_reason_counts": {},
                "schema_deficit_promotion_reject_reason_counts": {},
                "schema_deficit_soft_append_accepted_count": 0,
                "schema_deficit_soft_append_rejected_count": 0,
                "schema_deficit_soft_append_doc_ids": [],
                "schema_deficit_soft_append_reason_counts": {},
                "schema_deficit_soft_append_reject_reason_counts": {},
                "schema_reliability_evaluated_count": 0,
                "schema_reliability_passed_count": 0,
                "schema_reliability_rescued_count": 0,
                "schema_reliability_rejected_count": 0,
                "schema_reliability_reason_counts": {},
                "schema_reliability_reject_reason_counts": {},
                "answer_type_conflict_guard_evaluated_count": 0,
                "answer_type_conflict_guard_rejected_count": 0,
                "answer_type_conflict_guard_reject_reason_counts": {},
                "support_set_rerank_applied_count": 0,
                "support_set_rerank_promoted_to_top2": 0,
                "support_set_rerank_promoted_to_top5": 0,
                "support_set_rerank_reason_counts": {},
                "support_set_coverage_preservation_rejected_count": 0,
                "support_set_coverage_preservation_reject_reason_counts": {},
                "budgeted_evidence_subgraph_applied_count": 0,
                "budgeted_evidence_subgraph_selected_count": 0,
                "budgeted_evidence_subgraph_selected_doc_ids": [],
                "budgeted_evidence_subgraph_reason_counts": {},
                "decoupled_evidence_frontier_enabled": bool(
                    getattr(self.global_config, "enable_decoupled_evidence_frontier", False)
                ),
                "promotion_utility_veto_enabled": bool(
                    getattr(self.global_config, "enable_promotion_utility_veto", False)
                ),
                "promotion_utility_veto_evaluated_count": 0,
                "promotion_utility_veto_rejected_count": 0,
                "promotion_utility_veto_reason_counts": {},
                "promotion_utility_veto_soft_append_count": 0,
                "promotion_utility_veto_high_risk_only": bool(
                    getattr(self.global_config, "promotion_utility_veto_high_risk_only", True)
                ),
                "promotion_utility_veto_append_on_reject": bool(
                    getattr(self.global_config, "promotion_utility_veto_append_on_reject", True)
                ),
            }
            planned_extra_memory_docs = []
            planned_extra_memory_doc_set = set()
            planned_evidence_insufficient = iterative_retrieval_info.get("reason") == "max_steps_reached"
            if getattr(self.global_config, "enable_planned_bridge_retrieval", False):
                bridge_cues = has_bridge_cues(query)
                bridge_only = bool(getattr(self.global_config, "planned_bridge_retrieval_bridge_only", False))
                require_prior_insufficient = bool(
                    getattr(self.global_config, "planned_bridge_retrieval_require_insufficient", False)
                )
                planned_retrieval_info.update({
                    "reason": "started",
                    "bridge_cues": bridge_cues,
                    "bridge_only": bridge_only,
                    "evidence_insufficient": planned_evidence_insufficient,
                    "require_prior_insufficient": require_prior_insufficient,
                    "closed_loop": True,
                    "verification_first": True,
                })
                if bridge_only and not bridge_cues:
                    planned_retrieval_info["reason"] = "no_bridge_cues"
                elif require_prior_insufficient and not planned_evidence_insufficient:
                    planned_retrieval_info["reason"] = "evidence_not_marked_insufficient"
                else:
                    base_max_rounds = max(1, int(getattr(self.global_config, "planned_bridge_retrieval_max_rounds", 5)))
                    base_max_queries = max(1, int(getattr(self.global_config, "planned_bridge_retrieval_max_queries", 3)))
                    max_snippets = max(1, int(getattr(self.global_config, "planned_bridge_retrieval_max_snippets", 3)))
                    max_tokens = max(1, int(getattr(self.global_config, "planned_bridge_retrieval_max_tokens", 160)))
                    base_protect_top_k = max(0, int(getattr(self.global_config, "planned_bridge_retrieval_protect_top_k", 1)))
                    base_max_promotions = max(0, int(getattr(self.global_config, "planned_bridge_retrieval_max_promotions", 2)))
                    base_plan_top_k = max(1, int(getattr(self.global_config, "planned_bridge_retrieval_plan_top_k", 20)))
                    base_min_score = float(getattr(self.global_config, "planned_bridge_retrieval_min_score", 0.18))
                    merge_mode = str(getattr(self.global_config, "planned_bridge_retrieval_merge_mode", "guarded_promote"))
                    max_extra_docs = max(
                        0,
                        int(getattr(self.global_config, "planned_bridge_retrieval_max_extra_docs", 3)),
                    )
                    guarded_min_score = float(
                        getattr(self.global_config, "planned_bridge_retrieval_guarded_min_score", 0.82)
                    )
                    guarded_max_original_rank = max(
                        1,
                        int(getattr(self.global_config, "planned_bridge_retrieval_guarded_max_original_rank", 80)),
                    )
                    guarded_max_plan_rank = max(
                        1,
                        int(getattr(self.global_config, "planned_bridge_retrieval_guarded_max_plan_rank", 8)),
                    )
                    guarded_min_query_overlap = max(
                        0.0,
                        float(getattr(self.global_config, "planned_bridge_retrieval_guarded_min_query_overlap", 0.08)),
                    )
                    top2_override_global_enabled = bool(
                        getattr(self.global_config, "enable_planned_top2_override", False)
                    )
                    top2_override_min_score = float(
                        getattr(self.global_config, "planned_top2_override_min_score", 0.96)
                    )
                    top2_override_max_original_rank = max(
                        2,
                        int(getattr(self.global_config, "planned_top2_override_max_original_rank", 30)),
                    )
                    top2_override_max_plan_rank = max(
                        1,
                        int(getattr(self.global_config, "planned_top2_override_max_plan_rank", 2)),
                    )
                    top2_override_min_query_overlap = max(
                        0.0,
                        float(getattr(self.global_config, "planned_top2_override_min_query_overlap", 0.18)),
                    )
                    top2_override_min_query_hits = max(
                        1,
                        int(getattr(self.global_config, "planned_top2_override_min_query_hits", 1)),
                    )
                    top2_intervention_gate_mode = str(
                        getattr(self.global_config, "planned_top2_intervention_gate_mode", "off")
                    ).strip().lower()
                    top2_gate_min_bridge_strength = max(
                        0,
                        int(getattr(self.global_config, "planned_top2_gate_min_bridge_strength", 2)),
                    )
                    top2_gate_long_context_avg_doc_chars = max(
                        0.0,
                        float(getattr(
                            self.global_config,
                            "planned_top2_gate_long_context_avg_doc_chars",
                            1200.0,
                        )),
                    )
                    top2_gate_long_context_empty_title_ratio = min(
                        1.0,
                        max(
                            0.0,
                            float(getattr(
                                self.global_config,
                                "planned_top2_gate_long_context_empty_title_ratio",
                                0.50,
                            )),
                        ),
                    )
                    top2_gate_long_context_min_bridge_strength = max(
                        0,
                        int(getattr(
                            self.global_config,
                            "planned_top2_gate_long_context_min_bridge_strength",
                            4,
                        )),
                    )
                    pairwise_utility_search_enabled = bool(
                        getattr(self.global_config, "enable_planned_pairwise_utility_search", False)
                    )
                    pairwise_utility_max_original_rank = max(
                        2,
                        int(getattr(self.global_config, "planned_pairwise_utility_max_original_rank", 30)),
                    )
                    pairwise_utility_max_candidates = max(
                        1,
                        int(getattr(self.global_config, "planned_pairwise_utility_max_candidates", 32)),
                    )
                    pairwise_utility_min_margin = max(
                        0.0,
                        float(getattr(self.global_config, "planned_pairwise_utility_min_margin", 0.12)),
                    )
                    pairwise_utility_min_pair_score = max(
                        0.0,
                        float(getattr(self.global_config, "planned_pairwise_utility_min_pair_score", 0.65)),
                    )
                    pairwise_utility_min_candidate_support = max(
                        0.0,
                        float(getattr(
                            self.global_config,
                            "planned_pairwise_utility_min_candidate_support",
                            0.45,
                        )),
                    )
                    pairwise_utility_allowed_sources = str(
                        getattr(
                            self.global_config,
                            "planned_pairwise_utility_allowed_sources",
                            "planned_and_original",
                        )
                    )
                    pairwise_utility_source_policy_mode = str(
                        getattr(
                            self.global_config,
                            "planned_pairwise_utility_source_policy",
                            "fixed",
                        )
                    ).strip().lower()
                    pairwise_utility_adaptive_min_bridge_strength = max(
                        0,
                        int(getattr(
                            self.global_config,
                            "planned_pairwise_utility_adaptive_min_bridge_strength",
                            3,
                        )),
                    )
                    pairwise_utility_adaptive_max_avg_doc_chars = max(
                        0.0,
                        float(getattr(
                            self.global_config,
                            "planned_pairwise_utility_adaptive_max_avg_doc_chars",
                            900.0,
                        )),
                    )
                    pairwise_utility_adaptive_allow_long_answer = bool(
                        getattr(
                            self.global_config,
                            "planned_pairwise_utility_adaptive_allow_long_answer",
                            False,
                        )
                    )
                    pairwise_utility_original_fallback_only = bool(
                        getattr(
                            self.global_config,
                            "planned_pairwise_utility_original_fallback_only",
                            True,
                        )
                    )
                    query_schema_memory_intervention_policy = str(
                        getattr(
                            self.global_config,
                            "query_schema_memory_intervention_policy",
                            "risk_aware",
                        )
                    ).strip().lower()
                    query_schema_memory_top2_min_score = max(
                        0.0,
                        float(getattr(self.global_config, "query_schema_memory_top2_min_score", 0.66)),
                    )
                    query_schema_memory_top2_min_schema_idf = min(
                        1.0,
                        max(
                            0.0,
                            float(getattr(self.global_config, "query_schema_memory_top2_min_schema_idf", 0.32)),
                        ),
                    )
                    query_schema_memory_top2_min_overlap = min(
                        1.0,
                        max(
                            0.0,
                            float(getattr(self.global_config, "query_schema_memory_top2_min_overlap", 0.10)),
                        ),
                    )
                    query_schema_memory_top2_max_schema_frequency = max(
                        0,
                        int(getattr(self.global_config, "query_schema_memory_top2_max_schema_frequency", 80)),
                    )
                    query_schema_memory_top2_max_original_rank = max(
                        2,
                        int(getattr(self.global_config, "query_schema_memory_top2_max_original_rank", 24)),
                    )
                    query_schema_memory_top2_min_bridge_strength = max(
                        0,
                        int(getattr(self.global_config, "query_schema_memory_top2_min_bridge_strength", 3)),
                    )
                    schema_deficit_promotion_enabled = bool(
                        getattr(self.global_config, "enable_schema_deficit_promotion", False)
                    )
                    schema_deficit_min_score = max(
                        0.0,
                        float(getattr(self.global_config, "schema_deficit_min_score", 0.62)),
                    )
                    schema_deficit_min_gain = max(
                        0.0,
                        float(getattr(self.global_config, "schema_deficit_min_gain", 0.10)),
                    )
                    schema_deficit_min_schema_score = max(
                        0.0,
                        float(getattr(self.global_config, "schema_deficit_min_schema_score", 0.68)),
                    )
                    schema_deficit_min_schema_idf = max(
                        0.0,
                        float(getattr(self.global_config, "schema_deficit_min_schema_idf", 0.30)),
                    )
                    schema_deficit_max_original_rank = max(
                        2,
                        int(getattr(self.global_config, "schema_deficit_max_original_rank", 60)),
                    )
                    schema_deficit_require_bridge_constraint = bool(
                        getattr(self.global_config, "schema_deficit_require_bridge_constraint", True)
                    )
                    schema_deficit_min_bridge_score = max(
                        0.0,
                        float(getattr(self.global_config, "schema_deficit_min_bridge_score", 0.54)),
                    )
                    schema_deficit_min_bridge_margin = max(
                        0.0,
                        float(getattr(self.global_config, "schema_deficit_min_bridge_margin", 0.08)),
                    )
                    schema_deficit_min_bridge_missing_gain = max(
                        0.0,
                        float(getattr(self.global_config, "schema_deficit_min_bridge_missing_gain", 0.10)),
                    )
                    schema_deficit_min_bridge_anchor_overlap = min(
                        1.0,
                        max(
                            0.0,
                            float(getattr(self.global_config, "schema_deficit_min_bridge_anchor_overlap", 0.24)),
                        ),
                    )
                    schema_deficit_soft_append_enabled = bool(
                        getattr(self.global_config, "schema_deficit_soft_append_enabled", True)
                    )
                    schema_deficit_soft_append_min_score = max(
                        0.0,
                        float(getattr(self.global_config, "schema_deficit_soft_append_min_score", 0.48)),
                    )
                    schema_deficit_soft_append_min_schema_score = max(
                        0.0,
                        float(getattr(self.global_config, "schema_deficit_soft_append_min_schema_score", 0.58)),
                    )
                    schema_deficit_soft_append_min_schema_idf = max(
                        0.0,
                        float(getattr(self.global_config, "schema_deficit_soft_append_min_schema_idf", 0.24)),
                    )
                    schema_deficit_soft_append_min_query_support = min(
                        1.0,
                        max(
                            0.0,
                            float(getattr(self.global_config, "schema_deficit_soft_append_min_query_support", 0.34)),
                        ),
                    )
                    schema_deficit_soft_append_max_original_rank = max(
                        2,
                        int(getattr(self.global_config, "schema_deficit_soft_append_max_original_rank", 80)),
                    )
                    schema_reliability_gate_enabled = bool(
                        getattr(self.global_config, "enable_schema_reliability_gate", False)
                    )
                    schema_reliability_min_score = min(
                        1.0,
                        max(
                            0.0,
                            float(getattr(self.global_config, "schema_reliability_min_score", 0.62)),
                        ),
                    )
                    schema_reliability_rescue_min_score = min(
                        1.0,
                        max(
                            0.0,
                            float(getattr(self.global_config, "schema_reliability_rescue_min_score", 0.76)),
                        ),
                    )
                    schema_reliability_min_anchor_overlap = min(
                        1.0,
                        max(
                            0.0,
                            float(getattr(self.global_config, "schema_reliability_min_anchor_overlap", 0.08)),
                        ),
                    )
                    schema_reliability_max_schema_frequency = max(
                        0,
                        int(getattr(self.global_config, "schema_reliability_max_schema_frequency", 80)),
                    )
                    schema_reliability_numeric_requires_value = bool(
                        getattr(self.global_config, "schema_reliability_numeric_requires_value", True)
                    )
                    schema_verified_prior_enabled = bool(
                        getattr(self.global_config, "enable_schema_verified_utility_prior", False)
                    )
                    schema_verified_prior_bonus = max(
                        0.0,
                        float(getattr(self.global_config, "schema_verified_prior_bonus", 0.08)),
                    )
                    schema_verified_prior_min_schema_score = max(
                        0.0,
                        float(getattr(self.global_config, "schema_verified_prior_min_schema_score", 0.72)),
                    )
                    schema_verified_prior_min_schema_idf = min(
                        1.0,
                        max(
                            0.0,
                            float(getattr(self.global_config, "schema_verified_prior_min_schema_idf", 0.58)),
                        ),
                    )
                    schema_verified_prior_min_overlap = min(
                        1.0,
                        max(
                            0.0,
                            float(getattr(self.global_config, "schema_verified_prior_min_overlap", 0.18)),
                        ),
                    )
                    schema_verified_prior_max_original_rank = max(
                        2,
                        int(getattr(self.global_config, "schema_verified_prior_max_original_rank", 20)),
                    )
                    schema_verified_prior_numeric_requires_value = bool(
                        getattr(self.global_config, "schema_verified_prior_numeric_requires_value", True)
                    )
                    schema_verified_prior_block_long_context = bool(
                        getattr(self.global_config, "schema_verified_prior_block_long_context", True)
                    )
                    schema_verified_prior_rescue_enabled = bool(
                        getattr(self.global_config, "schema_verified_prior_rescue_enabled", False)
                    )
                    schema_verified_prior_min_rescue_bonus = max(
                        0.0,
                        float(getattr(self.global_config, "schema_verified_prior_min_rescue_bonus", 0.045)),
                    )
                    answer_type_conflict_guard_enabled = bool(
                        getattr(self.global_config, "enable_answer_type_conflict_guard", False)
                    )
                    answer_type_conflict_guard_min_old_signal = max(
                        0.0,
                        float(getattr(
                            self.global_config,
                            "answer_type_conflict_guard_min_old_signal",
                            0.55,
                        )),
                    )
                    answer_type_conflict_guard_min_candidate_signal = max(
                        0.0,
                        float(getattr(
                            self.global_config,
                            "answer_type_conflict_guard_min_candidate_signal",
                            0.35,
                        )),
                    )
                    answer_type_conflict_guard_min_conflict_margin = max(
                        0.0,
                        float(getattr(
                            self.global_config,
                            "answer_type_conflict_guard_min_conflict_margin",
                            0.08,
                        )),
                    )
                    answer_type_conflict_guard_numeric_only = bool(
                        getattr(self.global_config, "answer_type_conflict_guard_numeric_only", True)
                    )
                    support_set_rerank_enabled = bool(
                        getattr(self.global_config, "enable_planned_support_set_rerank", False)
                    )
                    support_set_rerank_target_k = max(
                        1,
                        int(getattr(self.global_config, "planned_support_set_target_k", 5)),
                    )
                    support_set_rerank_protect_top_k = max(
                        0,
                        int(getattr(self.global_config, "planned_support_set_protect_top_k", 2)),
                    )
                    support_set_rerank_max_replacements = max(
                        0,
                        int(getattr(self.global_config, "planned_support_set_max_replacements", 2)),
                    )
                    support_set_rerank_max_candidates = max(
                        support_set_rerank_target_k,
                        int(getattr(self.global_config, "planned_support_set_max_candidates", 96)),
                    )
                    support_set_rerank_max_original_rank = max(
                        support_set_rerank_target_k,
                        int(getattr(self.global_config, "planned_support_set_max_original_rank", 200)),
                    )
                    support_set_rerank_min_candidate_score = max(
                        0.0,
                        float(getattr(
                            self.global_config,
                            "planned_support_set_min_candidate_score",
                            0.52,
                        )),
                    )
                    support_set_rerank_redundancy_penalty = max(
                        0.0,
                        float(getattr(
                            self.global_config,
                            "planned_support_set_redundancy_penalty",
                            0.18,
                        )),
                    )
                    support_set_rerank_preserve_original_topk = bool(
                        getattr(
                            self.global_config,
                            "planned_support_set_preserve_original_topk",
                            True,
                        )
                    )
                    support_set_rerank_preserve_min_gain = max(
                        0.0,
                        float(getattr(
                            self.global_config,
                            "planned_support_set_preserve_min_gain",
                            0.03,
                        )),
                    )
                    support_set_rerank_preserve_max_coverage_loss = max(
                        0.0,
                        float(getattr(
                            self.global_config,
                            "planned_support_set_preserve_max_coverage_loss",
                            0.02,
                        )),
                    )
                    bridge_slot_rescue_enabled = bool(
                        getattr(self.global_config, "enable_planned_bridge_slot_rescue", False)
                    )
                    bridge_slot_rescue_min_score = max(
                        0.0,
                        float(getattr(self.global_config, "planned_bridge_slot_rescue_min_score", 0.58)),
                    )
                    bridge_slot_rescue_min_improvement = max(
                        0.0,
                        float(getattr(self.global_config, "planned_bridge_slot_rescue_min_improvement", 0.08)),
                    )
                    bridge_slot_rescue_min_anchor_overlap = min(
                        1.0,
                        max(
                            0.0,
                            float(getattr(
                                self.global_config,
                                "planned_bridge_slot_rescue_min_anchor_overlap",
                                0.34,
                            )),
                        ),
                    )
                    bridge_slot_rescue_require_relation = bool(
                        getattr(self.global_config, "planned_bridge_slot_rescue_require_relation", True)
                    )
                    bridge_slot_rescue_require_old_deficit = bool(
                        getattr(self.global_config, "planned_bridge_slot_rescue_require_old_deficit", True)
                    )
                    bridge_slot_rescue_deficit_max_old_score = max(
                        0.0,
                        float(getattr(
                            self.global_config,
                            "planned_bridge_slot_rescue_deficit_max_old_score",
                            0.50,
                        )),
                    )
                    bridge_slot_rescue_min_missing_gain = max(
                        0.0,
                        float(getattr(
                            self.global_config,
                            "planned_bridge_slot_rescue_min_missing_gain",
                            0.15,
                        )),
                    )
                    bridge_slot_rescue_append_on_no_deficit = bool(
                        getattr(
                            self.global_config,
                            "planned_bridge_slot_rescue_append_on_no_deficit",
                            True,
                        )
                    )
                    object_memory_enabled = bool(
                        getattr(self.global_config, "enable_object_consistent_evidence_memory", False)
                    )
                    object_memory_min_score = max(
                        0.0,
                        float(getattr(self.global_config, "object_consistent_memory_min_score", 0.42)),
                    )
                    object_memory_max_sentences = max(
                        1,
                        int(getattr(self.global_config, "object_consistent_memory_max_sentences", 2)),
                    )
                    object_memory_max_objects = max(
                        0,
                        int(getattr(self.global_config, "object_consistent_memory_max_objects", 1)),
                    )
                    object_memory_max_chars = max(
                        120,
                        int(getattr(self.global_config, "object_consistent_memory_max_chars", 720)),
                    )
                    object_memory_append_only = bool(
                        getattr(self.global_config, "object_consistent_memory_append_only", True)
                    )
                    object_memory_append_max_items = max(
                        0,
                        int(getattr(self.global_config, "object_consistent_memory_append_max_items", 1)),
                    )
                    object_memory_append_candidate_top_k = max(
                        1,
                        int(getattr(
                            self.global_config,
                            "object_consistent_memory_append_candidate_top_k",
                            12,
                        )),
                    )
                    object_memory_append_min_plan_score = max(
                        0.0,
                        float(getattr(
                            self.global_config,
                            "object_consistent_memory_append_min_plan_score",
                            0.50,
                        )),
                    )
                    budgeted_evidence_subgraph_enabled = bool(
                        getattr(self.global_config, "enable_budgeted_evidence_subgraph", False)
                    )
                    budgeted_evidence_subgraph_max_append = max(
                        0,
                        int(getattr(self.global_config, "budgeted_evidence_subgraph_max_append", 1)),
                    )
                    budgeted_evidence_subgraph_max_candidates = max(
                        1,
                        int(getattr(self.global_config, "budgeted_evidence_subgraph_max_candidates", 48)),
                    )
                    budgeted_evidence_subgraph_min_utility = max(
                        0.0,
                        float(getattr(self.global_config, "budgeted_evidence_subgraph_min_utility", 0.48)),
                    )
                    budgeted_evidence_subgraph_redundancy_penalty = max(
                        0.0,
                        float(getattr(
                            self.global_config,
                            "budgeted_evidence_subgraph_redundancy_penalty",
                            0.16,
                        )),
                    )
                    budgeted_evidence_subgraph_require_bridge_or_gap = bool(
                        getattr(
                            self.global_config,
                            "budgeted_evidence_subgraph_require_bridge_or_gap",
                            True,
                        )
                    )
                    evidence_frontier_enabled = bool(
                        getattr(self.global_config, "enable_decoupled_evidence_frontier", False)
                    )
                    promotion_utility_veto_enabled = bool(
                        getattr(self.global_config, "enable_promotion_utility_veto", False)
                    )
                    promotion_utility_veto_min_utility = max(
                        0.0,
                        float(getattr(
                            self.global_config,
                            "promotion_utility_veto_min_utility",
                            0.18,
                        )),
                    )
                    promotion_utility_veto_require_bridge_or_gap = bool(
                        getattr(
                            self.global_config,
                            "promotion_utility_veto_require_bridge_or_gap",
                            True,
                        )
                    )
                    promotion_utility_veto_apply_to_guarded = bool(
                        getattr(
                            self.global_config,
                            "promotion_utility_veto_apply_to_guarded",
                            False,
                        )
                    )
                    promotion_utility_veto_high_risk_only = bool(
                        getattr(
                            self.global_config,
                            "promotion_utility_veto_high_risk_only",
                            True,
                        )
                    )
                    promotion_utility_veto_append_on_reject = bool(
                        getattr(
                            self.global_config,
                            "promotion_utility_veto_append_on_reject",
                            True,
                        )
                    )
                    top2_pair_verifier_global_enabled = bool(
                        getattr(self.global_config, "enable_planned_top2_pair_verifier", False)
                    )
                    top2_pair_verifier_mode = str(
                        getattr(self.global_config, "planned_top2_pair_verifier_mode", "hard")
                    ).strip().lower()
                    if top2_pair_verifier_mode not in {"hard", "soft"}:
                        top2_pair_verifier_mode = "hard"
                    top2_pair_verifier_score_weight = max(
                        0.0,
                        float(getattr(self.global_config, "planned_top2_pair_verifier_score_weight", 0.12)),
                    )
                    top2_pair_verifier_min_margin = max(
                        0.0,
                        float(getattr(self.global_config, "planned_top2_pair_verifier_min_margin", 0.04)),
                    )
                    top2_pair_verifier_max_coverage_loss = max(
                        0.0,
                        float(getattr(self.global_config, "planned_top2_pair_verifier_max_coverage_loss", 0.02)),
                    )
                    top2_pair_verifier_min_candidate_query_support = max(
                        0.0,
                        float(getattr(
                            self.global_config,
                            "planned_top2_pair_verifier_min_candidate_query_support",
                            0.45,
                        )),
                    )
                    relation_object_guard_global_enabled = bool(
                        getattr(self.global_config, "enable_planned_relation_object_guard", False)
                    )
                    relation_object_guard_min_score = max(
                        0.0,
                        float(getattr(self.global_config, "planned_relation_object_guard_min_score", 0.55)),
                    )
                    relation_object_guard_min_margin = max(
                        0.0,
                        float(getattr(self.global_config, "planned_relation_object_guard_min_margin", 0.08)),
                    )
                    entity_conditioned_relation_guard_enabled = bool(
                        getattr(self.global_config, "enable_planned_entity_conditioned_relation_guard", False)
                    )
                    entity_relation_anchor_min_overlap = min(
                        1.0,
                        max(
                            0.0,
                            float(getattr(self.global_config, "planned_entity_relation_anchor_min_overlap", 0.50)),
                        ),
                    )
                    planned_extra_doc_ids = []
                    planned_extra_doc_id_set = set()
                    planned_extra_memory_docs = []
                    planned_extra_memory_doc_set = set()
                    residual_deficit_enabled = bool(
                        getattr(self.global_config, "enable_planned_residual_deficit_queries", False)
                    )
                    residual_deficit_max_queries = max(
                        0,
                        int(getattr(self.global_config, "planned_residual_deficit_max_queries", 1)),
                    )
                    residual_deficit_last_round_only = bool(
                        getattr(
                            self.global_config,
                            "planned_residual_deficit_last_round_only",
                            False,
                        )
                    )
                    residual_deficit_bridge_only = bool(
                        getattr(self.global_config, "planned_residual_deficit_bridge_only", True)
                    )
                    residual_deficit_require_explicit_slot = bool(
                        getattr(
                            self.global_config,
                            "planned_residual_deficit_require_explicit_slot",
                            True,
                        )
                    )
                    residual_deficit_disable_long_context = bool(
                        getattr(
                            self.global_config,
                            "planned_residual_deficit_disable_long_context",
                            True,
                        )
                    )
                    residual_deficit_min_bridge_strength = max(
                        0,
                        int(
                            getattr(
                                self.global_config,
                                "planned_residual_deficit_min_bridge_strength",
                                2,
                            )
                        ),
                    )
                    residual_deficit_reserve_after_round = max(
                        0,
                        int(
                            getattr(
                                self.global_config,
                                "planned_residual_deficit_reserve_after_round",
                                2,
                            )
                        ),
                    )
                    residual_deficit_replace_last_query = bool(
                        getattr(
                            self.global_config,
                            "planned_residual_deficit_replace_last_query",
                            True,
                        )
                    )
                    residual_deficit_append_only_after_round = max(
                        0,
                        int(
                            getattr(
                                self.global_config,
                                "planned_residual_deficit_append_only_after_round",
                                3,
                            )
                        ),
                    )
                    residual_deficit_append_candidate_top_k = max(
                        1,
                        int(
                            getattr(
                                self.global_config,
                                "planned_residual_deficit_append_candidate_top_k",
                                3,
                            )
                        ),
                    )
                    residual_deficit_append_max_docs = max(
                        0,
                        int(
                            getattr(
                                self.global_config,
                                "planned_residual_deficit_append_max_docs",
                                1,
                            )
                        ),
                    )
                    residual_deficit_rank_stable_after_round = max(
                        0,
                        int(
                            getattr(
                                self.global_config,
                                "planned_residual_deficit_rank_stable_after_round",
                                3,
                            )
                        ),
                    )
                    initial_route_docs = [
                        self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                        for idx in sorted_doc_ids[:self.global_config.qa_top_k]
                    ]
                    question_router = QuestionRouter(
                        route_mode=getattr(self.global_config, "planned_bridge_retrieval_route_mode", "off"),
                        base_max_rounds=base_max_rounds,
                        base_max_queries=base_max_queries,
                        base_max_promotions=base_max_promotions,
                        base_protect_top_k=base_protect_top_k,
                        base_plan_top_k=base_plan_top_k,
                        base_min_score=base_min_score,
                        qa_top_k=self.global_config.qa_top_k,
                    )
                    route_info = question_router.run(
                        question=query,
                        docs=initial_route_docs,
                        bridge_cues=bridge_cues,
                    )
                    planned_retrieval_info["router"] = route_info
                    planned_retrieval_info["budget_policy"] = route_info.get(
                        "budget_policy", "fixed_global_cap_dynamic_use"
                    )
                    planned_retrieval_info["selector_policy"] = route_info.get(
                        "selector_policy", "fixed_global_threshold"
                    )
                    planned_retrieval_info["global_budget_cap"] = {
                        "max_rounds": base_max_rounds,
                        "max_queries": base_max_queries,
                        "max_promotions": base_max_promotions,
                        "protect_top_k": base_protect_top_k,
                        "plan_top_k": base_plan_top_k,
                        "min_plan_score": base_min_score,
                        "merge_mode": merge_mode,
                        "risk_aware_available": merge_mode in {"risk_aware", "risk_aware_guarded"},
                        "max_extra_docs": max_extra_docs,
                        "guarded_min_score": guarded_min_score,
                        "guarded_max_original_rank": guarded_max_original_rank,
                        "guarded_max_plan_rank": guarded_max_plan_rank,
                        "guarded_min_query_overlap": guarded_min_query_overlap,
                        "residual_deficit_enabled": residual_deficit_enabled,
                        "residual_deficit_max_queries": residual_deficit_max_queries,
                        "residual_deficit_last_round_only": residual_deficit_last_round_only,
                        "residual_deficit_bridge_only": residual_deficit_bridge_only,
                        "residual_deficit_require_explicit_slot": (
                            residual_deficit_require_explicit_slot
                        ),
                        "residual_deficit_disable_long_context": (
                            residual_deficit_disable_long_context
                        ),
                        "residual_deficit_min_bridge_strength": (
                            residual_deficit_min_bridge_strength
                        ),
                        "residual_deficit_reserve_after_round": (
                            residual_deficit_reserve_after_round
                        ),
                        "residual_deficit_replace_last_query": (
                            residual_deficit_replace_last_query
                        ),
                        "residual_deficit_append_only_after_round": (
                            residual_deficit_append_only_after_round
                        ),
                        "residual_deficit_append_candidate_top_k": (
                            residual_deficit_append_candidate_top_k
                        ),
                        "residual_deficit_append_max_docs": residual_deficit_append_max_docs,
                        "residual_deficit_rank_stable_after_round": (
                            residual_deficit_rank_stable_after_round
                        ),
                        "top2_override_enabled": top2_override_global_enabled,
                        "top2_override_min_score": top2_override_min_score,
                        "top2_override_max_original_rank": top2_override_max_original_rank,
                        "top2_override_max_plan_rank": top2_override_max_plan_rank,
                        "top2_override_min_query_overlap": top2_override_min_query_overlap,
                        "top2_override_min_query_hits": top2_override_min_query_hits,
                        "top2_intervention_gate_mode": top2_intervention_gate_mode,
                        "top2_gate_min_bridge_strength": top2_gate_min_bridge_strength,
                        "top2_gate_long_context_avg_doc_chars": top2_gate_long_context_avg_doc_chars,
                        "top2_gate_long_context_empty_title_ratio": (
                            top2_gate_long_context_empty_title_ratio
                        ),
                        "top2_gate_long_context_min_bridge_strength": (
                            top2_gate_long_context_min_bridge_strength
                        ),
                        "pairwise_utility_search_enabled": pairwise_utility_search_enabled,
                        "pairwise_utility_max_original_rank": pairwise_utility_max_original_rank,
                        "pairwise_utility_max_candidates": pairwise_utility_max_candidates,
                        "pairwise_utility_min_margin": pairwise_utility_min_margin,
                        "pairwise_utility_min_pair_score": pairwise_utility_min_pair_score,
                        "pairwise_utility_min_candidate_support": (
                            pairwise_utility_min_candidate_support
                        ),
                        "pairwise_utility_allowed_sources": pairwise_utility_allowed_sources,
                        "pairwise_utility_source_policy": pairwise_utility_source_policy_mode,
                        "pairwise_utility_adaptive_min_bridge_strength": (
                            pairwise_utility_adaptive_min_bridge_strength
                        ),
                        "pairwise_utility_adaptive_max_avg_doc_chars": (
                            pairwise_utility_adaptive_max_avg_doc_chars
                        ),
                        "pairwise_utility_adaptive_allow_long_answer": (
                            pairwise_utility_adaptive_allow_long_answer
                        ),
                        "pairwise_utility_original_fallback_only": (
                            pairwise_utility_original_fallback_only
                        ),
                        "support_set_rerank_enabled": support_set_rerank_enabled,
                        "support_set_rerank_target_k": support_set_rerank_target_k,
                        "support_set_rerank_protect_top_k": support_set_rerank_protect_top_k,
                        "support_set_rerank_max_replacements": support_set_rerank_max_replacements,
                        "support_set_rerank_max_candidates": support_set_rerank_max_candidates,
                        "support_set_rerank_max_original_rank": support_set_rerank_max_original_rank,
                        "support_set_rerank_min_candidate_score": (
                            support_set_rerank_min_candidate_score
                        ),
                        "support_set_rerank_redundancy_penalty": (
                            support_set_rerank_redundancy_penalty
                        ),
                        "support_set_rerank_preserve_original_topk": (
                            support_set_rerank_preserve_original_topk
                        ),
                        "support_set_rerank_preserve_min_gain": (
                            support_set_rerank_preserve_min_gain
                        ),
                        "support_set_rerank_preserve_max_coverage_loss": (
                            support_set_rerank_preserve_max_coverage_loss
                        ),
                        "bridge_slot_rescue_enabled": bridge_slot_rescue_enabled,
                        "bridge_slot_rescue_min_score": bridge_slot_rescue_min_score,
                        "bridge_slot_rescue_min_improvement": bridge_slot_rescue_min_improvement,
                        "bridge_slot_rescue_min_anchor_overlap": bridge_slot_rescue_min_anchor_overlap,
                        "bridge_slot_rescue_require_relation": bridge_slot_rescue_require_relation,
                        "bridge_slot_rescue_require_old_deficit": (
                            bridge_slot_rescue_require_old_deficit
                        ),
                        "bridge_slot_rescue_deficit_max_old_score": (
                            bridge_slot_rescue_deficit_max_old_score
                        ),
                        "bridge_slot_rescue_min_missing_gain": (
                            bridge_slot_rescue_min_missing_gain
                        ),
                        "bridge_slot_rescue_append_on_no_deficit": (
                            bridge_slot_rescue_append_on_no_deficit
                        ),
                        "object_memory_enabled": object_memory_enabled,
                        "object_memory_min_score": object_memory_min_score,
                        "object_memory_max_sentences": object_memory_max_sentences,
                        "object_memory_max_objects": object_memory_max_objects,
                        "object_memory_max_chars": object_memory_max_chars,
                        "object_memory_append_only": object_memory_append_only,
                        "object_memory_append_max_items": object_memory_append_max_items,
                        "object_memory_append_candidate_top_k": object_memory_append_candidate_top_k,
                        "object_memory_append_min_plan_score": (
                            object_memory_append_min_plan_score
                        ),
                        "budgeted_evidence_subgraph_enabled": budgeted_evidence_subgraph_enabled,
                        "budgeted_evidence_subgraph_max_append": (
                            budgeted_evidence_subgraph_max_append
                        ),
                        "budgeted_evidence_subgraph_max_candidates": (
                            budgeted_evidence_subgraph_max_candidates
                        ),
                        "budgeted_evidence_subgraph_min_utility": (
                            budgeted_evidence_subgraph_min_utility
                        ),
                        "budgeted_evidence_subgraph_redundancy_penalty": (
                            budgeted_evidence_subgraph_redundancy_penalty
                        ),
                        "budgeted_evidence_subgraph_require_bridge_or_gap": (
                            budgeted_evidence_subgraph_require_bridge_or_gap
                        ),
                        "decoupled_evidence_frontier_enabled": evidence_frontier_enabled,
                        "promotion_utility_veto_enabled": promotion_utility_veto_enabled,
                        "promotion_utility_veto_min_utility": promotion_utility_veto_min_utility,
                        "promotion_utility_veto_require_bridge_or_gap": (
                            promotion_utility_veto_require_bridge_or_gap
                        ),
                        "promotion_utility_veto_apply_to_guarded": (
                            promotion_utility_veto_apply_to_guarded
                        ),
                        "promotion_utility_veto_high_risk_only": (
                            promotion_utility_veto_high_risk_only
                        ),
                        "promotion_utility_veto_append_on_reject": (
                            promotion_utility_veto_append_on_reject
                        ),
                        "top2_pair_verifier_enabled": top2_pair_verifier_global_enabled,
                        "top2_pair_verifier_mode": top2_pair_verifier_mode,
                        "top2_pair_verifier_score_weight": top2_pair_verifier_score_weight,
                        "top2_pair_verifier_min_margin": top2_pair_verifier_min_margin,
                        "top2_pair_verifier_max_coverage_loss": top2_pair_verifier_max_coverage_loss,
                        "top2_pair_verifier_min_candidate_query_support": (
                            top2_pair_verifier_min_candidate_query_support
                        ),
                        "relation_object_guard_enabled": relation_object_guard_global_enabled,
                        "relation_object_guard_min_score": relation_object_guard_min_score,
                        "relation_object_guard_min_margin": relation_object_guard_min_margin,
                        "answer_type_conflict_guard_enabled": answer_type_conflict_guard_enabled,
                        "answer_type_conflict_guard_min_old_signal": (
                            answer_type_conflict_guard_min_old_signal
                        ),
                        "answer_type_conflict_guard_min_candidate_signal": (
                            answer_type_conflict_guard_min_candidate_signal
                        ),
                        "answer_type_conflict_guard_min_conflict_margin": (
                            answer_type_conflict_guard_min_conflict_margin
                        ),
                        "answer_type_conflict_guard_numeric_only": (
                            answer_type_conflict_guard_numeric_only
                        ),
                        "entity_conditioned_relation_guard_enabled": (
                            entity_conditioned_relation_guard_enabled
                        ),
                        "entity_relation_anchor_min_overlap": entity_relation_anchor_min_overlap,
                    }
                    planned_retrieval_info["component_roles"] = [
                        question_router.role,
                        "planner",
                        "retriever",
                        "merger",
                        "verifier",
                    ]
                    if not route_info.get("allow_retrieval", True):
                        planned_retrieval_info["reason"] = f"routed_{route_info.get('question_type', 'skip')}"
                        planned_retrieval_info["routing_profile"] = route_info.get("profile")
                        planned_retrieval_info["question_type"] = route_info.get("question_type")
                    else:
                        max_rounds = max(1, int(route_info.get("effective_max_rounds", base_max_rounds)))
                        max_queries = max(1, int(route_info.get("effective_max_queries", base_max_queries)))
                        protect_top_k = max(0, int(route_info.get("effective_protect_top_k", base_protect_top_k)))
                        max_promotions = max(0, int(route_info.get("effective_max_promotions", base_max_promotions)))
                        plan_top_k = max(1, int(route_info.get("effective_plan_top_k", base_plan_top_k)))
                        min_plan_score = float(route_info.get("effective_min_score", base_min_score))
                        merge_policy = route_risk_aware_merge_policy(
                            route_info=route_info,
                            requested_merge_mode=merge_mode,
                            protect_top_k=protect_top_k,
                            max_promotions=max_promotions,
                            qa_top_k=self.global_config.qa_top_k,
                        )
                        effective_merge_mode = merge_policy.get("effective_merge_mode", merge_mode)
                        protect_top_k = max(0, int(merge_policy.get("protect_top_k", protect_top_k)))
                        max_promotions = max(0, int(merge_policy.get("max_promotions", max_promotions)))
                        top2_override_base_enabled = (
                            top2_override_global_enabled
                            and effective_merge_mode == "guarded_promote"
                            and route_info.get("question_type") == "bridge_or_compositional"
                            and route_info.get("profile") == "full_closed_loop"
                            and merge_policy.get("intervention_risk") == "medium"
                        )
                        intervention_gate = route_context_state_intervention_gate(
                            route_info=route_info,
                            merge_policy=merge_policy,
                            gate_mode=top2_intervention_gate_mode,
                            min_bridge_strength=top2_gate_min_bridge_strength,
                            long_context_avg_doc_chars=top2_gate_long_context_avg_doc_chars,
                            long_context_empty_title_ratio=top2_gate_long_context_empty_title_ratio,
                            long_context_min_bridge_strength=(
                                top2_gate_long_context_min_bridge_strength
                            ),
                        )
                        top2_override_enabled = (
                            top2_override_base_enabled
                            and bool(intervention_gate.get("top2_allowed", True))
                        )
                        pairwise_utility_source_policy = route_pairwise_utility_source_policy(
                            route_info=route_info,
                            intervention_gate=intervention_gate,
                            configured_allowed_sources=pairwise_utility_allowed_sources,
                            source_policy=pairwise_utility_source_policy_mode,
                            adaptive_min_bridge_strength=(
                                pairwise_utility_adaptive_min_bridge_strength
                            ),
                            adaptive_max_avg_doc_chars=(
                                pairwise_utility_adaptive_max_avg_doc_chars
                            ),
                            adaptive_allow_long_answer=(
                                pairwise_utility_adaptive_allow_long_answer
                            ),
                        )
                        effective_pairwise_utility_allowed_sources = str(
                            pairwise_utility_source_policy.get(
                                "effective_allowed_sources",
                                pairwise_utility_allowed_sources,
                            )
                        )
                        merge_policy["top2_intervention_gate"] = intervention_gate
                        merge_policy["pairwise_utility_source_policy"] = pairwise_utility_source_policy
                        merge_policy["can_change_top2"] = bool(top2_override_enabled)
                        merge_policy["rank_intervention"] = intervention_gate.get("rank_intervention")
                        if top2_override_enabled:
                            merge_policy["can_change_top2"] = True
                            merge_policy["top2_override_policy"] = (
                                "pairwise_utility_rank2_search"
                                if pairwise_utility_search_enabled
                                else
                                "soft_pair_reranked_relation_guarded_rank2"
                                if top2_pair_verifier_global_enabled and top2_pair_verifier_mode == "soft"
                                else "counterfactual_pair_verified_rank2_only"
                                if top2_pair_verifier_global_enabled
                                else "evidence_verified_rank2_only"
                            )
                        elif top2_override_base_enabled and intervention_gate.get("enabled"):
                            merge_policy["top2_override_policy"] = "context_state_gate_disabled_rank2"
                        query_workers = max(
                            1,
                            int(getattr(self.global_config, "planned_bridge_retrieval_query_workers", 1)),
                        )

                        verify_each_round = True
                        retrieval_planner = RetrievalPlanner(
                            self.llm_model,
                            max_queries=max_queries,
                            max_snippets=max_snippets,
                            max_tokens=max_tokens,
                        )
                        source_retriever = SourceRetriever(self)
                        source_verifier = SourceVerifier(
                            self.llm_model,
                            max_queries=max_queries,
                            max_snippets=max_snippets,
                            max_tokens=max_tokens,
                        )
                        def planned_doc_lookup(doc_id: int) -> str:
                            return self.chunk_embedding_store.get_row(
                                self.passage_node_keys[int(doc_id)]
                            )["content"]

                        evidence_merger = EvidenceMerger(
                            target_k=self.global_config.qa_top_k,
                            protect_top_k=protect_top_k,
                            max_promotions=max_promotions,
                            plan_top_k=plan_top_k,
                            min_plan_score=min_plan_score,
                            merge_mode=effective_merge_mode,
                            question=query,
                            guarded_min_score=guarded_min_score,
                            guarded_max_original_rank=guarded_max_original_rank,
                            guarded_max_plan_rank=guarded_max_plan_rank,
                            guarded_min_query_overlap=guarded_min_query_overlap,
                            top2_override_enabled=top2_override_enabled,
                            top2_override_min_score=top2_override_min_score,
                            top2_override_max_original_rank=top2_override_max_original_rank,
                            top2_override_max_plan_rank=top2_override_max_plan_rank,
                            top2_override_min_query_overlap=top2_override_min_query_overlap,
                            top2_override_min_query_hits=top2_override_min_query_hits,
                            top2_pair_verifier_enabled=(
                                top2_pair_verifier_global_enabled and top2_override_enabled
                            ),
                            top2_pair_verifier_mode=top2_pair_verifier_mode,
                            top2_pair_verifier_score_weight=top2_pair_verifier_score_weight,
                            top2_pair_verifier_min_margin=top2_pair_verifier_min_margin,
                            top2_pair_verifier_max_coverage_loss=top2_pair_verifier_max_coverage_loss,
                            top2_pair_verifier_min_candidate_query_support=(
                                top2_pair_verifier_min_candidate_query_support
                            ),
                            relation_object_guard_enabled=(
                                relation_object_guard_global_enabled and top2_override_enabled
                            ),
                            relation_object_guard_min_score=relation_object_guard_min_score,
                            relation_object_guard_min_margin=relation_object_guard_min_margin,
                            entity_conditioned_relation_guard_enabled=(
                                entity_conditioned_relation_guard_enabled
                                and relation_object_guard_global_enabled
                                and top2_override_enabled
                            ),
                            entity_relation_anchor_min_overlap=entity_relation_anchor_min_overlap,
                            intervention_gate=intervention_gate,
                            pairwise_utility_search_enabled=(
                                pairwise_utility_search_enabled and top2_override_enabled
                            ),
                            pairwise_utility_max_original_rank=pairwise_utility_max_original_rank,
                            pairwise_utility_max_candidates=pairwise_utility_max_candidates,
                            pairwise_utility_min_margin=pairwise_utility_min_margin,
                            pairwise_utility_min_pair_score=pairwise_utility_min_pair_score,
                            pairwise_utility_min_candidate_support=(
                                pairwise_utility_min_candidate_support
                            ),
                            pairwise_utility_allowed_sources=(
                                effective_pairwise_utility_allowed_sources
                            ),
                            pairwise_utility_original_fallback_only=(
                                pairwise_utility_original_fallback_only
                            ),
                            query_schema_memory_intervention_policy=(
                                query_schema_memory_intervention_policy
                            ),
                            query_schema_memory_top2_min_score=query_schema_memory_top2_min_score,
                            query_schema_memory_top2_min_schema_idf=(
                                query_schema_memory_top2_min_schema_idf
                            ),
                            query_schema_memory_top2_min_overlap=query_schema_memory_top2_min_overlap,
                            query_schema_memory_top2_max_schema_frequency=(
                                query_schema_memory_top2_max_schema_frequency
                            ),
                            query_schema_memory_top2_max_original_rank=(
                                query_schema_memory_top2_max_original_rank
                            ),
                            query_schema_memory_top2_min_bridge_strength=(
                                query_schema_memory_top2_min_bridge_strength
                            ),
                            schema_deficit_promotion_enabled=schema_deficit_promotion_enabled,
                            schema_deficit_min_score=schema_deficit_min_score,
                            schema_deficit_min_gain=schema_deficit_min_gain,
                            schema_deficit_min_schema_score=schema_deficit_min_schema_score,
                            schema_deficit_min_schema_idf=schema_deficit_min_schema_idf,
                            schema_deficit_max_original_rank=schema_deficit_max_original_rank,
                            schema_deficit_require_bridge_constraint=(
                                schema_deficit_require_bridge_constraint
                            ),
                            schema_deficit_min_bridge_score=schema_deficit_min_bridge_score,
                            schema_deficit_min_bridge_margin=schema_deficit_min_bridge_margin,
                            schema_deficit_min_bridge_missing_gain=(
                                schema_deficit_min_bridge_missing_gain
                            ),
                            schema_deficit_min_bridge_anchor_overlap=(
                                schema_deficit_min_bridge_anchor_overlap
                            ),
                            schema_deficit_soft_append_enabled=(
                                schema_deficit_soft_append_enabled
                            ),
                            schema_deficit_soft_append_min_score=(
                                schema_deficit_soft_append_min_score
                            ),
                            schema_deficit_soft_append_min_schema_score=(
                                schema_deficit_soft_append_min_schema_score
                            ),
                            schema_deficit_soft_append_min_schema_idf=(
                                schema_deficit_soft_append_min_schema_idf
                            ),
                            schema_deficit_soft_append_min_query_support=(
                                schema_deficit_soft_append_min_query_support
                            ),
                            schema_deficit_soft_append_max_original_rank=(
                                schema_deficit_soft_append_max_original_rank
                            ),
                            schema_reliability_gate_enabled=schema_reliability_gate_enabled,
                            schema_reliability_min_score=schema_reliability_min_score,
                            schema_reliability_rescue_min_score=(
                                schema_reliability_rescue_min_score
                            ),
                            schema_reliability_min_anchor_overlap=(
                                schema_reliability_min_anchor_overlap
                            ),
                            schema_reliability_max_schema_frequency=(
                                schema_reliability_max_schema_frequency
                            ),
                            schema_reliability_numeric_requires_value=(
                                schema_reliability_numeric_requires_value
                            ),
                            schema_verified_prior_enabled=schema_verified_prior_enabled,
                            schema_verified_prior_bonus=schema_verified_prior_bonus,
                            schema_verified_prior_min_schema_score=(
                                schema_verified_prior_min_schema_score
                            ),
                            schema_verified_prior_min_schema_idf=schema_verified_prior_min_schema_idf,
                            schema_verified_prior_min_overlap=schema_verified_prior_min_overlap,
                            schema_verified_prior_max_original_rank=(
                                schema_verified_prior_max_original_rank
                            ),
                            schema_verified_prior_numeric_requires_value=(
                                schema_verified_prior_numeric_requires_value
                            ),
                            schema_verified_prior_block_long_context=(
                                schema_verified_prior_block_long_context
                            ),
                            schema_verified_prior_rescue_enabled=schema_verified_prior_rescue_enabled,
                            schema_verified_prior_min_rescue_bonus=(
                                schema_verified_prior_min_rescue_bonus
                            ),
                            answer_type_conflict_guard_enabled=(
                                answer_type_conflict_guard_enabled
                                and (top2_override_enabled or evidence_frontier_enabled)
                            ),
                            answer_type_conflict_guard_min_old_signal=(
                                answer_type_conflict_guard_min_old_signal
                            ),
                            answer_type_conflict_guard_min_candidate_signal=(
                                answer_type_conflict_guard_min_candidate_signal
                            ),
                            answer_type_conflict_guard_min_conflict_margin=(
                                answer_type_conflict_guard_min_conflict_margin
                            ),
                            answer_type_conflict_guard_numeric_only=(
                                answer_type_conflict_guard_numeric_only
                            ),
                            support_set_rerank_enabled=support_set_rerank_enabled,
                            support_set_rerank_target_k=support_set_rerank_target_k,
                            support_set_rerank_protect_top_k=support_set_rerank_protect_top_k,
                            support_set_rerank_max_replacements=(
                                support_set_rerank_max_replacements
                            ),
                            support_set_rerank_max_candidates=support_set_rerank_max_candidates,
                            support_set_rerank_max_original_rank=(
                                support_set_rerank_max_original_rank
                            ),
                            support_set_rerank_min_candidate_score=(
                                support_set_rerank_min_candidate_score
                            ),
                            support_set_rerank_redundancy_penalty=(
                                support_set_rerank_redundancy_penalty
                            ),
                            support_set_rerank_preserve_original_topk=(
                                support_set_rerank_preserve_original_topk
                            ),
                            support_set_rerank_preserve_min_gain=(
                                support_set_rerank_preserve_min_gain
                            ),
                            support_set_rerank_preserve_max_coverage_loss=(
                                support_set_rerank_preserve_max_coverage_loss
                            ),
                            bridge_slot_rescue_enabled=(
                                bridge_slot_rescue_enabled
                                and (top2_override_enabled or evidence_frontier_enabled)
                            ),
                            bridge_slot_rescue_min_score=bridge_slot_rescue_min_score,
                            bridge_slot_rescue_min_improvement=bridge_slot_rescue_min_improvement,
                            bridge_slot_rescue_min_anchor_overlap=bridge_slot_rescue_min_anchor_overlap,
                            bridge_slot_rescue_require_relation=bridge_slot_rescue_require_relation,
                            bridge_slot_rescue_require_old_deficit=(
                                bridge_slot_rescue_require_old_deficit
                            ),
                            bridge_slot_rescue_deficit_max_old_score=(
                                bridge_slot_rescue_deficit_max_old_score
                            ),
                            bridge_slot_rescue_min_missing_gain=(
                                bridge_slot_rescue_min_missing_gain
                            ),
                            bridge_slot_rescue_append_on_no_deficit=(
                                bridge_slot_rescue_append_on_no_deficit
                            ),
                            object_memory_enabled=object_memory_enabled,
                            object_memory_min_score=object_memory_min_score,
                            object_memory_max_sentences=object_memory_max_sentences,
                            object_memory_max_objects=object_memory_max_objects,
                            object_memory_max_chars=object_memory_max_chars,
                            object_memory_append_only=object_memory_append_only,
                            object_memory_append_max_items=object_memory_append_max_items,
                            object_memory_append_candidate_top_k=(
                                object_memory_append_candidate_top_k
                            ),
                            object_memory_append_min_plan_score=(
                                object_memory_append_min_plan_score
                            ),
                            budgeted_evidence_subgraph_enabled=(
                                budgeted_evidence_subgraph_enabled
                            ),
                            budgeted_evidence_subgraph_max_append=(
                                budgeted_evidence_subgraph_max_append
                            ),
                            budgeted_evidence_subgraph_max_candidates=(
                                budgeted_evidence_subgraph_max_candidates
                            ),
                            budgeted_evidence_subgraph_min_utility=(
                                budgeted_evidence_subgraph_min_utility
                            ),
                            budgeted_evidence_subgraph_redundancy_penalty=(
                                budgeted_evidence_subgraph_redundancy_penalty
                            ),
                            budgeted_evidence_subgraph_require_bridge_or_gap=(
                                budgeted_evidence_subgraph_require_bridge_or_gap
                            ),
                            evidence_frontier_enabled=evidence_frontier_enabled,
                            promotion_utility_veto_enabled=promotion_utility_veto_enabled,
                            promotion_utility_veto_min_utility=(
                                promotion_utility_veto_min_utility
                            ),
                            promotion_utility_veto_require_bridge_or_gap=(
                                promotion_utility_veto_require_bridge_or_gap
                            ),
                            promotion_utility_veto_apply_to_guarded=(
                                promotion_utility_veto_apply_to_guarded
                            ),
                            promotion_utility_veto_high_risk_only=(
                                promotion_utility_veto_high_risk_only
                            ),
                            promotion_utility_veto_append_on_reject=(
                                promotion_utility_veto_append_on_reject
                            ),
                            doc_lookup=planned_doc_lookup,
                        )
                        next_queries = []
                        missing_instruction = ""
                        planned_retrieval_info.update({
                            "component_roles": [
                                question_router.role,
                                retrieval_planner.role,
                                source_retriever.role,
                                evidence_merger.role,
                                source_verifier.role,
                            ],
                            "max_rounds": max_rounds,
                            "max_queries": max_queries,
                            "verify_each_round": verify_each_round,
                            "routing_profile": route_info.get("profile"),
                            "question_type": route_info.get("question_type"),
                            "protect_top_k": protect_top_k,
                            "max_promotions": max_promotions,
                            "min_plan_score": min_plan_score,
                            "merge_mode": effective_merge_mode,
                            "requested_merge_mode": merge_mode,
                            "risk_aware_merge_policy": merge_policy,
                            "max_extra_docs": max_extra_docs,
                            "guarded_min_score": guarded_min_score,
                            "guarded_max_original_rank": guarded_max_original_rank,
                            "guarded_max_plan_rank": guarded_max_plan_rank,
                            "guarded_min_query_overlap": guarded_min_query_overlap,
                            "top2_override_enabled": top2_override_enabled,
                            "top2_intervention_gate": intervention_gate,
                            "top2_override_policy": {
                                "global_enabled": top2_override_global_enabled,
                                "base_enabled_for_query": top2_override_base_enabled,
                                "enabled_for_query": top2_override_enabled,
                                "intervention_gate_mode": top2_intervention_gate_mode,
                                "intervention_gate_reason": intervention_gate.get("reason"),
                                "rank_intervention": intervention_gate.get("rank_intervention"),
                                "pairwise_utility_search_enabled": (
                                    pairwise_utility_search_enabled and top2_override_enabled
                                ),
                                "pairwise_utility_max_original_rank": pairwise_utility_max_original_rank,
                                "pairwise_utility_max_candidates": pairwise_utility_max_candidates,
                                "pairwise_utility_min_margin": pairwise_utility_min_margin,
                                "pairwise_utility_min_pair_score": pairwise_utility_min_pair_score,
                                "pairwise_utility_min_candidate_support": (
                                    pairwise_utility_min_candidate_support
                                ),
                                "pairwise_utility_configured_allowed_sources": (
                                    pairwise_utility_allowed_sources
                                ),
                                "pairwise_utility_effective_allowed_sources": (
                                    effective_pairwise_utility_allowed_sources
                                ),
                                "pairwise_utility_source_policy": pairwise_utility_source_policy,
                                "pairwise_utility_original_fallback_only": (
                                    pairwise_utility_original_fallback_only
                                ),
                                "query_schema_intervention_policy": (
                                    query_schema_memory_intervention_policy
                                ),
                                "query_schema_top2_min_score": (
                                    query_schema_memory_top2_min_score
                                ),
                                "query_schema_top2_min_schema_idf": (
                                    query_schema_memory_top2_min_schema_idf
                                ),
                                "query_schema_top2_min_overlap": (
                                    query_schema_memory_top2_min_overlap
                                ),
                                "query_schema_top2_max_schema_frequency": (
                                    query_schema_memory_top2_max_schema_frequency
                                ),
                                "query_schema_top2_max_original_rank": (
                                    query_schema_memory_top2_max_original_rank
                                ),
                                "query_schema_top2_min_bridge_strength": (
                                    query_schema_memory_top2_min_bridge_strength
                                ),
                                "schema_deficit_promotion_enabled": schema_deficit_promotion_enabled,
                                "schema_deficit_min_score": schema_deficit_min_score,
                                "schema_deficit_min_gain": schema_deficit_min_gain,
                                "schema_deficit_min_schema_score": schema_deficit_min_schema_score,
                                "schema_deficit_min_schema_idf": schema_deficit_min_schema_idf,
                                "schema_deficit_max_original_rank": (
                                    schema_deficit_max_original_rank
                                ),
                                "schema_deficit_require_bridge_constraint": (
                                    schema_deficit_require_bridge_constraint
                                ),
                                "schema_deficit_min_bridge_score": schema_deficit_min_bridge_score,
                                "schema_deficit_min_bridge_margin": schema_deficit_min_bridge_margin,
                                "schema_deficit_min_bridge_missing_gain": (
                                    schema_deficit_min_bridge_missing_gain
                                ),
                                "schema_deficit_min_bridge_anchor_overlap": (
                                    schema_deficit_min_bridge_anchor_overlap
                                ),
                                "schema_deficit_soft_append_enabled": (
                                    schema_deficit_soft_append_enabled
                                ),
                                "schema_deficit_soft_append_min_score": (
                                    schema_deficit_soft_append_min_score
                                ),
                                "schema_deficit_soft_append_min_schema_score": (
                                    schema_deficit_soft_append_min_schema_score
                                ),
                                "schema_deficit_soft_append_min_schema_idf": (
                                    schema_deficit_soft_append_min_schema_idf
                                ),
                                "schema_deficit_soft_append_min_query_support": (
                                    schema_deficit_soft_append_min_query_support
                                ),
                                "schema_deficit_soft_append_max_original_rank": (
                                    schema_deficit_soft_append_max_original_rank
                                ),
                                "schema_reliability_gate_enabled": schema_reliability_gate_enabled,
                                "schema_reliability_min_score": schema_reliability_min_score,
                                "schema_reliability_rescue_min_score": (
                                    schema_reliability_rescue_min_score
                                ),
                                "schema_reliability_min_anchor_overlap": (
                                    schema_reliability_min_anchor_overlap
                                ),
                                "schema_reliability_max_schema_frequency": (
                                    schema_reliability_max_schema_frequency
                                ),
                                "schema_reliability_numeric_requires_value": (
                                    schema_reliability_numeric_requires_value
                                ),
                                "schema_verified_prior_enabled": schema_verified_prior_enabled,
                                "schema_verified_prior_bonus": schema_verified_prior_bonus,
                                "schema_verified_prior_min_schema_score": (
                                    schema_verified_prior_min_schema_score
                                ),
                                "schema_verified_prior_min_schema_idf": schema_verified_prior_min_schema_idf,
                                "schema_verified_prior_min_overlap": schema_verified_prior_min_overlap,
                                "schema_verified_prior_max_original_rank": (
                                    schema_verified_prior_max_original_rank
                                ),
                                "schema_verified_prior_numeric_requires_value": (
                                    schema_verified_prior_numeric_requires_value
                                ),
                                "schema_verified_prior_block_long_context": (
                                    schema_verified_prior_block_long_context
                                ),
                                "schema_verified_prior_rescue_enabled": schema_verified_prior_rescue_enabled,
                                "schema_verified_prior_min_rescue_bonus": (
                                    schema_verified_prior_min_rescue_bonus
                                ),
                                "answer_type_conflict_guard_enabled": (
                                    answer_type_conflict_guard_enabled
                                    and (top2_override_enabled or evidence_frontier_enabled)
                                ),
                                "answer_type_conflict_guard_min_old_signal": (
                                    answer_type_conflict_guard_min_old_signal
                                ),
                                "answer_type_conflict_guard_min_candidate_signal": (
                                    answer_type_conflict_guard_min_candidate_signal
                                ),
                                "answer_type_conflict_guard_min_conflict_margin": (
                                    answer_type_conflict_guard_min_conflict_margin
                                ),
                                "answer_type_conflict_guard_numeric_only": (
                                    answer_type_conflict_guard_numeric_only
                                ),
                                "bridge_slot_rescue_enabled": (
                                    bridge_slot_rescue_enabled
                                    and (top2_override_enabled or evidence_frontier_enabled)
                                ),
                                "bridge_slot_rescue_min_score": bridge_slot_rescue_min_score,
                                "bridge_slot_rescue_min_improvement": bridge_slot_rescue_min_improvement,
                                "bridge_slot_rescue_min_anchor_overlap": (
                                    bridge_slot_rescue_min_anchor_overlap
                                ),
                                "bridge_slot_rescue_require_relation": (
                                    bridge_slot_rescue_require_relation
                                ),
                                "bridge_slot_rescue_require_old_deficit": (
                                    bridge_slot_rescue_require_old_deficit
                                ),
                                "bridge_slot_rescue_deficit_max_old_score": (
                                    bridge_slot_rescue_deficit_max_old_score
                                ),
                                "bridge_slot_rescue_min_missing_gain": (
                                    bridge_slot_rescue_min_missing_gain
                                ),
                                "bridge_slot_rescue_append_on_no_deficit": (
                                    bridge_slot_rescue_append_on_no_deficit
                                ),
                                "object_memory_enabled": object_memory_enabled,
                                "object_memory_min_score": object_memory_min_score,
                                "object_memory_max_sentences": object_memory_max_sentences,
                                "object_memory_max_objects": object_memory_max_objects,
                                "object_memory_max_chars": object_memory_max_chars,
                                "object_memory_append_only": object_memory_append_only,
                                "object_memory_append_max_items": object_memory_append_max_items,
                                "object_memory_append_candidate_top_k": (
                                    object_memory_append_candidate_top_k
                                ),
                                "object_memory_append_min_plan_score": (
                                    object_memory_append_min_plan_score
                                ),
                                "budgeted_evidence_subgraph_enabled": (
                                    budgeted_evidence_subgraph_enabled
                                ),
                                "budgeted_evidence_subgraph_max_append": (
                                    budgeted_evidence_subgraph_max_append
                                ),
                                "budgeted_evidence_subgraph_max_candidates": (
                                    budgeted_evidence_subgraph_max_candidates
                                ),
                                "budgeted_evidence_subgraph_min_utility": (
                                    budgeted_evidence_subgraph_min_utility
                                ),
                                "budgeted_evidence_subgraph_redundancy_penalty": (
                                    budgeted_evidence_subgraph_redundancy_penalty
                                ),
                                "budgeted_evidence_subgraph_require_bridge_or_gap": (
                                    budgeted_evidence_subgraph_require_bridge_or_gap
                                ),
                                "decoupled_evidence_frontier_enabled": evidence_frontier_enabled,
                                "promotion_utility_veto_enabled": promotion_utility_veto_enabled,
                                "promotion_utility_veto_min_utility": (
                                    promotion_utility_veto_min_utility
                                ),
                                "promotion_utility_veto_require_bridge_or_gap": (
                                    promotion_utility_veto_require_bridge_or_gap
                                ),
                                "promotion_utility_veto_apply_to_guarded": (
                                    promotion_utility_veto_apply_to_guarded
                                ),
                                "promotion_utility_veto_high_risk_only": (
                                    promotion_utility_veto_high_risk_only
                                ),
                                "promotion_utility_veto_append_on_reject": (
                                    promotion_utility_veto_append_on_reject
                                ),
                                "min_score": top2_override_min_score,
                                "max_original_rank": top2_override_max_original_rank,
                                "max_plan_rank": top2_override_max_plan_rank,
                                "min_query_overlap": top2_override_min_query_overlap,
                                "min_query_hits": top2_override_min_query_hits,
                                "pair_verifier_global_enabled": top2_pair_verifier_global_enabled,
                                "pair_verifier_enabled_for_query": (
                                    top2_pair_verifier_global_enabled and top2_override_enabled
                                ),
                                "pair_verifier_mode": top2_pair_verifier_mode,
                                "pair_verifier_score_weight": top2_pair_verifier_score_weight,
                                "pair_verifier_min_margin": top2_pair_verifier_min_margin,
                                "pair_verifier_max_coverage_loss": top2_pair_verifier_max_coverage_loss,
                                "pair_verifier_min_candidate_query_support": (
                                    top2_pair_verifier_min_candidate_query_support
                                ),
                                "relation_object_guard_global_enabled": relation_object_guard_global_enabled,
                                "relation_object_guard_enabled_for_query": (
                                    relation_object_guard_global_enabled and top2_override_enabled
                                ),
                                "relation_object_guard_min_score": relation_object_guard_min_score,
                                "relation_object_guard_min_margin": relation_object_guard_min_margin,
                                "entity_conditioned_relation_guard_global_enabled": (
                                    entity_conditioned_relation_guard_enabled
                                ),
                                "entity_conditioned_relation_guard_enabled_for_query": (
                                    entity_conditioned_relation_guard_enabled
                                    and relation_object_guard_global_enabled
                                    and top2_override_enabled
                                ),
                                "entity_relation_anchor_min_overlap": entity_relation_anchor_min_overlap,
                            },
                            "support_set_rerank_policy": {
                                "enabled": support_set_rerank_enabled,
                                "target_k": support_set_rerank_target_k,
                                "protect_top_k": support_set_rerank_protect_top_k,
                                "max_replacements": support_set_rerank_max_replacements,
                                "max_candidates": support_set_rerank_max_candidates,
                                "max_original_rank": support_set_rerank_max_original_rank,
                                "min_candidate_score": support_set_rerank_min_candidate_score,
                                "redundancy_penalty": support_set_rerank_redundancy_penalty,
                                "preserve_original_topk": (
                                    support_set_rerank_preserve_original_topk
                                ),
                                "preserve_min_gain": support_set_rerank_preserve_min_gain,
                                "preserve_max_coverage_loss": (
                                    support_set_rerank_preserve_max_coverage_loss
                                ),
                                "top5_allowed_for_query": bool(
                                    intervention_gate.get("top5_allowed", True)
                                ),
                            },
                            "query_workers": query_workers,
                            "budget_action": route_info.get("budget_action"),
                        })
                        planned_round_cost_controller = PlannedRetrievalCostController(
                            enabled=bool(
                                getattr(
                                    self.global_config,
                                    "enable_planned_retrieval_cost_controller",
                                    False,
                                )
                            ),
                            min_rounds=int(
                                getattr(
                                    self.global_config,
                                    "planned_retrieval_cost_controller_min_rounds",
                                    2,
                                )
                            ),
                            fixed_point_patience=int(
                                getattr(
                                    self.global_config,
                                    "planned_retrieval_cost_controller_patience",
                                    2,
                                )
                            ),
                            resolution_prior=float(
                                getattr(
                                    self.global_config,
                                    "planned_retrieval_cost_controller_resolution_prior",
                                    0.12,
                                )
                            ),
                            no_progress_decay=float(
                                getattr(
                                    self.global_config,
                                    "planned_retrieval_cost_controller_no_progress_decay",
                                    0.50,
                                )
                            ),
                            planner_cost=float(
                                getattr(
                                    self.global_config,
                                    "planned_retrieval_cost_controller_planner_cost",
                                    0.04,
                                )
                            ),
                            verifier_cost=float(
                                getattr(
                                    self.global_config,
                                    "planned_retrieval_cost_controller_verifier_cost",
                                    0.04,
                                )
                            ),
                            retriever_cost=float(
                                getattr(
                                    self.global_config,
                                    "planned_retrieval_cost_controller_retriever_cost",
                                    0.015,
                                )
                            ),
                            min_continue_net_benefit=float(
                                getattr(
                                    self.global_config,
                                    "planned_retrieval_cost_controller_min_continue_net_benefit",
                                    0.0,
                                )
                            ),
                            candidate_frontier_mode=str(
                                getattr(
                                    self.global_config,
                                    "planned_retrieval_cost_controller_candidate_frontier_mode",
                                    "raw_candidates",
                                )
                            ),
                            shadow_only=bool(
                                getattr(
                                    self.global_config,
                                    "planned_retrieval_cost_controller_shadow_only",
                                    False,
                                )
                            ),
                        )
                        planned_retrieval_info["round_cost_controller"] = (
                            planned_round_cost_controller.summary()
                        )
                        seen_query_norms = set()
                        for round_idx in range(max_rounds):
                            current_doc_ids = [int(idx) for idx in sorted_doc_ids[:self.global_config.qa_top_k]]
                            current_seen_doc_ids = set(current_doc_ids)
                            for doc_id in planned_extra_doc_ids:
                                if doc_id in current_seen_doc_ids:
                                    continue
                                current_doc_ids.append(doc_id)
                                current_seen_doc_ids.add(doc_id)
                            current_docs_memory = list(planned_extra_memory_docs)
                            current_docs = [
                                self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                                for idx in current_doc_ids
                            ] + current_docs_memory
                            before_evidence_signature = build_planned_retrieval_evidence_signature(
                                current_doc_ids,
                                current_docs_memory,
                            )
                            round_info = {
                                "round": round_idx + 1,
                                "queries": [],
                                "promoted_to_top2": 0,
                                "promoted_to_top5": 0,
                            }
                            planned_retrieval_info["round_count"] = round_idx + 1

                            try:
                                verdict = source_verifier.run(question=query, docs=current_docs)
                            except Exception as exc:
                                if not _is_transient_api_failure(exc):
                                    raise
                                _record_planned_retrieval_api_fallback(
                                    planned_retrieval_info,
                                    round_info,
                                    stage="verifier",
                                    exc=exc,
                                )
                                break
                            planned_retrieval_info["verifier_calls"] += 1
                            round_info["verifier"] = verdict
                            if verdict.get("sufficient"):
                                planned_retrieval_info["reason"] = (
                                    "initially_sufficient" if not planned_retrieval_info["applied"] else "verified_sufficient"
                                )
                                planned_retrieval_info["rounds"].append(round_info)
                                break

                            missing_parts = []
                            if verdict.get("missing"):
                                missing_parts.append(f"missing: {verdict.get('missing')}")
                            if verdict.get("missing_slots"):
                                missing_parts.append("missing slots: " + "; ".join(verdict.get("missing_slots", [])))
                            if verdict.get("uncertain_anchors"):
                                missing_parts.append("uncertain anchors: " + "; ".join(verdict.get("uncertain_anchors", [])))
                            if verdict.get("rejected_anchors"):
                                missing_parts.append("avoid distractors: " + "; ".join(verdict.get("rejected_anchors", [])))
                            if verdict.get("queries"):
                                missing_parts.append("verifier suggested queries: " + "; ".join(verdict.get("queries", [])))
                            missing_instruction = " | ".join(missing_parts)

                            try:
                                planner_output = retrieval_planner.run(
                                    question=query,
                                    docs=current_docs,
                                    missing_instruction=missing_instruction,
                                )
                            except Exception as exc:
                                if not _is_transient_api_failure(exc):
                                    raise
                                _record_planned_retrieval_api_fallback(
                                    planned_retrieval_info,
                                    round_info,
                                    stage="planner",
                                    exc=exc,
                                )
                                break
                            planned_retrieval_info["planner_calls"] += 1

                            candidate_queries = list(planner_output.get("queries", [])) + list(verdict.get("queries", []))
                            planned_queries = []
                            for planned_query in candidate_queries:
                                query_text = str(planned_query or "").strip()
                                query_norm = re.sub(r"\s+", " ", " ".join(re.findall(r"[a-z0-9]+", query_text.lower())))
                                if not query_text or len(query_norm.split()) < 2 or query_norm in seen_query_norms:
                                    continue
                                seen_query_norms.add(query_norm)
                                planned_queries.append(query_text)
                                if len(planned_queries) >= max_queries:
                                    break

                            residual_queries = []
                            residual_replaced_queries = []
                            residual_gate_reject_reasons = []
                            route_question_type = str(route_info.get("question_type", "") or "")
                            route_profile = str(route_info.get("profile", "") or "")
                            route_bridge_strength = int(route_info.get("bridge_strength", 0) or 0)
                            route_long_context_state = (
                                bool(route_info.get("long_context_shape", False))
                                or route_question_type
                                in {
                                    "long_answer_or_narrative",
                                    "long_context_evidence",
                                    "long_context_factoid",
                                }
                            )
                            explicit_deficit_values = []
                            for deficit_key in (
                                "missing_slots",
                                "missing",
                                "uncertain_anchors",
                                "queries",
                            ):
                                deficit_value = verdict.get(deficit_key)
                                if isinstance(deficit_value, list):
                                    explicit_deficit_values.extend(
                                        str(item).strip()
                                        for item in deficit_value
                                        if str(item).strip()
                                    )
                                elif deficit_value:
                                    explicit_deficit_values.append(str(deficit_value).strip())
                            explicit_deficit_signal = bool(explicit_deficit_values)
                            if (
                                residual_deficit_bridge_only
                                and (
                                    route_question_type != "bridge_or_compositional"
                                    or route_profile != "full_closed_loop"
                                )
                            ):
                                residual_gate_reject_reasons.append(
                                    "not_bridge_compositional_full_loop"
                                )
                            if residual_deficit_disable_long_context and route_long_context_state:
                                residual_gate_reject_reasons.append("long_context_residual_blocked")
                            if route_bridge_strength < residual_deficit_min_bridge_strength:
                                residual_gate_reject_reasons.append("weak_bridge_strength")
                            if residual_deficit_require_explicit_slot and not explicit_deficit_signal:
                                residual_gate_reject_reasons.append("no_explicit_deficit_slot")
                            should_try_residual = (
                                residual_deficit_enabled
                                and residual_deficit_max_queries > 0
                                and not residual_gate_reject_reasons
                                and (
                                    not planned_queries
                                    or not residual_deficit_last_round_only
                                    or round_idx >= max_rounds - 1
                                )
                            )
                            reserve_deficit_slot = (
                                should_try_residual
                                and residual_deficit_replace_last_query
                                and residual_deficit_reserve_after_round > 0
                                and (round_idx + 1) >= residual_deficit_reserve_after_round
                                and len(planned_queries) >= max_queries
                                and bool(planned_queries)
                            )
                            if (
                                residual_deficit_enabled
                                and residual_gate_reject_reasons
                            ):
                                reject_counts = planned_retrieval_info.setdefault(
                                    "residual_deficit_gate_reject_reason_counts",
                                    {},
                                )
                                for reject_reason in residual_gate_reject_reasons:
                                    reject_counts[reject_reason] = (
                                        int(reject_counts.get(reject_reason, 0)) + 1
                                    )
                            if should_try_residual:
                                residual_budget = min(
                                    residual_deficit_max_queries,
                                    max_queries - len(planned_queries),
                                )
                                if reserve_deficit_slot:
                                    residual_budget = max(
                                        residual_budget,
                                        min(residual_deficit_max_queries, 1),
                                    )
                                if residual_budget > 0:
                                    for residual_query in build_residual_deficit_queries(
                                        question=query,
                                        verdict=verdict,
                                        docs=current_docs,
                                        max_queries=max(3, residual_budget * 3),
                                    ):
                                        query_text = str(residual_query or "").strip()
                                        query_norm = re.sub(
                                            r"\s+",
                                            " ",
                                            " ".join(re.findall(r"[a-z0-9]+", query_text.lower())),
                                        )
                                        if (
                                            not query_text
                                            or len(query_norm.split()) < 2
                                            or query_norm in seen_query_norms
                                        ):
                                            continue
                                        seen_query_norms.add(query_norm)
                                        if len(planned_queries) < max_queries:
                                            planned_queries.append(query_text)
                                        elif reserve_deficit_slot and planned_queries:
                                            replaced_query = planned_queries.pop()
                                            planned_queries.append(query_text)
                                            residual_replaced_queries.append(replaced_query)
                                            reserve_deficit_slot = False
                                        else:
                                            continue
                                        residual_queries.append(query_text)
                                        if (
                                            len(residual_queries) >= residual_budget
                                            or len(planned_queries) >= max_queries
                                        ):
                                            break
                            if residual_queries:
                                planned_retrieval_info["residual_deficit_query_count"] += len(
                                    residual_queries
                                )
                                planned_retrieval_info["residual_deficit_round_count"] += 1
                                for residual_query in residual_queries:
                                    if residual_query not in planned_retrieval_info["residual_deficit_queries"]:
                                        planned_retrieval_info["residual_deficit_queries"].append(
                                            residual_query
                                        )
                                planned_retrieval_info["residual_deficit_replaced_query_count"] += len(
                                    residual_replaced_queries
                                )
                                for replaced_query in residual_replaced_queries:
                                    if replaced_query not in planned_retrieval_info["residual_deficit_replaced_queries"]:
                                        planned_retrieval_info["residual_deficit_replaced_queries"].append(
                                            replaced_query
                                        )

                            round_info.update({
                                "queries": planned_queries,
                                "planner": planner_output,
                                "verifier_queries": verdict.get("queries", []),
                                "residual_deficit_queries": residual_queries,
                                "residual_deficit_replaced_queries": residual_replaced_queries,
                                "residual_deficit_gate_reject_reasons": residual_gate_reject_reasons,
                                "residual_deficit_reserve_active": bool(residual_replaced_queries),
                                "residual_deficit_explicit_slot_count": len(
                                    explicit_deficit_values
                                ),
                            })
                            for planned_query in planned_queries:
                                if planned_query not in planned_retrieval_info["queries"]:
                                    planned_retrieval_info["queries"].append(planned_query)
                            if not planned_queries:
                                round_info["reason"] = "no_new_planned_queries"
                                planned_retrieval_info["rounds"].append(round_info)
                                planned_retrieval_info["reason"] = "no_new_planned_queries"
                                break

                            def normalize_planned_query(query_text: str) -> str:
                                return re.sub(
                                    r"\s+",
                                    " ",
                                    " ".join(re.findall(r"[a-z0-9]+", str(query_text or "").lower())),
                                )

                            residual_query_norms = {
                                normalize_planned_query(residual_query)
                                for residual_query in residual_queries
                            }
                            late_residual_append_only_round = (
                                residual_deficit_append_only_after_round > 0
                                and (round_idx + 1) >= residual_deficit_append_only_after_round
                            )
                            rank_stable_round = (
                                residual_deficit_rank_stable_after_round > 0
                                and (round_idx + 1) >= residual_deficit_rank_stable_after_round
                            )
                            late_residual_appended_doc_ids = []
                            late_residual_append_reject_reasons = {}

                            planned_results = []
                            retriever_outputs = []
                            source_retriever.prepare_queries(planned_queries)

                            def run_planned_query(planned_query):
                                retrieval_output = source_retriever.run(planned_query, ensure_embeddings=False)
                                return {
                                    "role": retrieval_output["role"],
                                    "query": planned_query,
                                    "mode": retrieval_output["mode"],
                                    "fallback_to_dense": retrieval_output.get("fallback_to_dense", False),
                                    "candidate_fact_count": retrieval_output.get("candidate_fact_count", 0),
                                    "top_fact_count": retrieval_output["top_fact_count"],
                                    "top_doc_count": int(len(retrieval_output["doc_ids"])),
                                    "doc_ids": retrieval_output["doc_ids"],
                                    "doc_scores": retrieval_output["doc_scores"],
                                }

                            if query_workers > 1 and len(planned_queries) > 1:
                                with ThreadPoolExecutor(max_workers=min(query_workers, len(planned_queries))) as executor:
                                    retrieval_outputs_raw = list(executor.map(run_planned_query, planned_queries))
                            else:
                                retrieval_outputs_raw = [run_planned_query(planned_query) for planned_query in planned_queries]

                            planned_retrieval_info["retriever_calls"] += len(planned_queries)
                            for retrieval_output in retrieval_outputs_raw:
                                if retrieval_output["mode"] == "dense":
                                    planned_retrieval_info["retriever_dense_fallbacks"] += 1
                                elif retrieval_output["mode"] == "graph":
                                    planned_retrieval_info["retriever_graph_calls"] += 1
                                retrieval_query_norm = normalize_planned_query(retrieval_output["query"])
                                is_residual_query = retrieval_query_norm in residual_query_norms
                                late_residual_append_only = (
                                    late_residual_append_only_round
                                    and is_residual_query
                                    and residual_deficit_append_max_docs > 0
                                )
                                appended_doc_ids = []
                                if late_residual_append_only:
                                    for doc_id_raw in retrieval_output["doc_ids"][:residual_deficit_append_candidate_top_k]:
                                        doc_id = int(doc_id_raw)
                                        if doc_id in current_seen_doc_ids or doc_id in planned_extra_doc_id_set:
                                            reason = "duplicate_or_already_in_context"
                                            late_residual_append_reject_reasons[reason] = (
                                                int(late_residual_append_reject_reasons.get(reason, 0)) + 1
                                            )
                                            continue
                                        if len(planned_extra_doc_ids) + len(planned_extra_memory_docs) >= max_extra_docs:
                                            reason = "extra_evidence_budget_full"
                                            late_residual_append_reject_reasons[reason] = (
                                                int(late_residual_append_reject_reasons.get(reason, 0)) + 1
                                            )
                                            break
                                        if len(appended_doc_ids) >= residual_deficit_append_max_docs:
                                            reason = "round_append_limit_reached"
                                            late_residual_append_reject_reasons[reason] = (
                                                int(late_residual_append_reject_reasons.get(reason, 0)) + 1
                                            )
                                            break
                                        planned_extra_doc_ids.append(doc_id)
                                        planned_extra_doc_id_set.add(doc_id)
                                        current_seen_doc_ids.add(doc_id)
                                        appended_doc_ids.append(doc_id)
                                        late_residual_appended_doc_ids.append(doc_id)
                                    if not appended_doc_ids:
                                        late_residual_append_reject_reasons.setdefault("no_append_candidate", 0)
                                        late_residual_append_reject_reasons["no_append_candidate"] += 1
                                else:
                                    planned_results.append((
                                        retrieval_output["query"],
                                        retrieval_output["doc_ids"],
                                        retrieval_output["doc_scores"],
                                    ))
                                retriever_outputs.append({
                                    "role": retrieval_output["role"],
                                    "query": retrieval_output["query"],
                                    "mode": retrieval_output["mode"],
                                    "residual_deficit_query": bool(is_residual_query),
                                    "late_residual_append_only": bool(late_residual_append_only),
                                    "late_residual_appended_doc_ids": appended_doc_ids,
                                    "fallback_to_dense": retrieval_output.get("fallback_to_dense", False),
                                    "candidate_fact_count": retrieval_output.get("candidate_fact_count", 0),
                                    "top_fact_count": retrieval_output["top_fact_count"],
                                    "top_doc_count": retrieval_output["top_doc_count"],
                                })
                            round_info["retriever"] = retriever_outputs
                            retrieval_candidate_signature = tuple(sorted({
                                int(doc_id)
                                for retrieval_output in retrieval_outputs_raw
                                for doc_id in retrieval_output.get("doc_ids", [])[:5]
                            }))
                            round_info["retriever_dense_fallbacks"] = sum(
                                1 for output in retriever_outputs if output.get("fallback_to_dense")
                            )
                            round_info["retriever_graph_calls"] = sum(
                                1 for output in retriever_outputs if output.get("mode") == "graph"
                            )
                            round_info["late_residual_append_only_round"] = bool(
                                late_residual_append_only_round
                            )
                            round_info["rank_stable_round"] = bool(rank_stable_round)
                            round_info["late_residual_appended_doc_ids"] = late_residual_appended_doc_ids
                            round_info["late_residual_append_reject_reasons"] = (
                                late_residual_append_reject_reasons
                            )
                            if late_residual_appended_doc_ids:
                                planned_retrieval_info["residual_deficit_append_only_doc_count"] += len(
                                    late_residual_appended_doc_ids
                                )
                                planned_retrieval_info["residual_deficit_append_only_round_count"] += 1
                                for doc_id in late_residual_appended_doc_ids:
                                    if doc_id not in planned_retrieval_info["residual_deficit_append_only_doc_ids"]:
                                        planned_retrieval_info["residual_deficit_append_only_doc_ids"].append(
                                            int(doc_id)
                                        )
                            append_reject_counts = planned_retrieval_info.setdefault(
                                "residual_deficit_append_only_reject_reason_counts",
                                {},
                            )
                            for reason, count in late_residual_append_reject_reasons.items():
                                append_reject_counts[str(reason)] = (
                                    int(append_reject_counts.get(str(reason), 0)) + int(count or 0)
                                )

                            planned_retrieval_info["merger_calls"] += 1
                            original_top2_override_enabled = evidence_merger.top2_override_enabled
                            if rank_stable_round:
                                evidence_merger.top2_override_enabled = False
                            try:
                                sorted_doc_ids, sorted_doc_scores, planned_merge_info = evidence_merger.run(
                                    base_doc_ids=sorted_doc_ids,
                                    base_doc_scores=sorted_doc_scores,
                                    planned_results=planned_results,
                                    schema_memory_candidates=(
                                        (query_schema_memory_info or {}).get("candidate_items", [])
                                    ),
                                )
                            finally:
                                evidence_merger.top2_override_enabled = original_top2_override_enabled
                            planned_merge_info["rank_stable_round"] = bool(rank_stable_round)
                            planned_merge_info["late_residual_append_only_doc_count"] = len(
                                late_residual_appended_doc_ids
                            )
                            round_info["merge"] = planned_merge_info
                            round_info["promoted_to_top2"] = planned_merge_info.get("promoted_to_top2", 0)
                            round_info["promoted_to_top5"] = planned_merge_info.get("promoted_to_top5", 0)
                            memory_doc_by_id = {
                                int(item.get("doc_id")): str(item.get("memory_doc") or "")
                                for item in planned_merge_info.get("expansion_memory_docs", []) or []
                                if item.get("memory_doc")
                            }
                            for doc_id in planned_merge_info.get("expansion_doc_ids", []):
                                doc_id = int(doc_id)
                                memory_doc = memory_doc_by_id.get(doc_id)
                                if memory_doc:
                                    memory_key = memory_doc.strip()
                                    if not memory_key or memory_key in planned_extra_memory_doc_set:
                                        continue
                                    if len(planned_extra_doc_ids) + len(planned_extra_memory_docs) >= max_extra_docs:
                                        break
                                    planned_extra_memory_docs.append(memory_doc)
                                    planned_extra_memory_doc_set.add(memory_key)
                                    continue
                                if doc_id in planned_extra_doc_id_set:
                                    continue
                                if len(planned_extra_doc_ids) + len(planned_extra_memory_docs) >= max_extra_docs:
                                    break
                                planned_extra_doc_ids.append(doc_id)
                                planned_extra_doc_id_set.add(doc_id)
                            applied_this_round = (
                                planned_merge_info.get("promoted_count", 0) > 0
                                or planned_merge_info.get("expansion_count", 0) > 0
                                or planned_merge_info.get("support_set_rerank_applied", 0) > 0
                                or bool(late_residual_appended_doc_ids)
                            )
                            if applied_this_round:
                                planned_retrieval_info["applied"] = True
                                planned_retrieval_info["reason"] = "applied"
                            else:
                                round_info["reason"] = "no_new_evidence_candidates"
                            planned_retrieval_info["promoted_to_top2"] += planned_merge_info.get("promoted_to_top2", 0)
                            planned_retrieval_info["promoted_to_top5"] += planned_merge_info.get("promoted_to_top5", 0)
                            planned_retrieval_info["top2_override_applied"] += planned_merge_info.get(
                                "top2_override_applied", 0
                            )
                            planned_retrieval_info["top2_override_rejected_count"] += planned_merge_info.get(
                                "top2_override_rejected_count", 0
                            )
                            planned_retrieval_info["promotion_utility_veto_evaluated_count"] += int(
                                planned_merge_info.get(
                                    "promotion_utility_veto_evaluated_count",
                                    0,
                                ) or 0
                            )
                            planned_retrieval_info["promotion_utility_veto_rejected_count"] += int(
                                planned_merge_info.get(
                                    "promotion_utility_veto_rejected_count",
                                    0,
                                ) or 0
                            )
                            veto_reason_counts = planned_retrieval_info.setdefault(
                                "promotion_utility_veto_reason_counts",
                                {},
                            )
                            for reason, count in (
                                planned_merge_info.get(
                                    "promotion_utility_veto_reason_counts",
                                    {},
                                ) or {}
                            ).items():
                                reason_key = str(reason)
                                veto_reason_counts[reason_key] = (
                                    int(veto_reason_counts.get(reason_key, 0) or 0)
                                    + int(count or 0)
                                )
                            planned_retrieval_info[
                                "promotion_utility_veto_soft_append_count"
                            ] += int(
                                planned_merge_info.get(
                                    "promotion_utility_veto_soft_append_count",
                                    0,
                                ) or 0
                            )
                            support_set_info = planned_merge_info.get("support_set_rerank") or {}
                            if support_set_info.get("applied"):
                                planned_retrieval_info["support_set_rerank_applied_count"] += 1
                            planned_retrieval_info["support_set_rerank_promoted_to_top2"] += int(
                                support_set_info.get("promoted_to_top2", 0) or 0
                            )
                            planned_retrieval_info["support_set_rerank_promoted_to_top5"] += int(
                                support_set_info.get("promoted_to_top5", 0) or 0
                            )
                            support_reason = str(support_set_info.get("reason", "unavailable"))
                            support_reason_counts = planned_retrieval_info.setdefault(
                                "support_set_rerank_reason_counts",
                                {},
                            )
                            support_reason_counts[support_reason] = (
                                int(support_reason_counts.get(support_reason, 0) or 0) + 1
                            )
                            subgraph_info = planned_merge_info.get("budgeted_evidence_subgraph") or {}
                            if subgraph_info.get("applied"):
                                planned_retrieval_info["budgeted_evidence_subgraph_applied_count"] += 1
                            planned_retrieval_info["budgeted_evidence_subgraph_selected_count"] += int(
                                subgraph_info.get("selected_count", 0) or 0
                            )
                            for doc_id in subgraph_info.get("selected_doc_ids", []) or []:
                                if doc_id not in planned_retrieval_info[
                                    "budgeted_evidence_subgraph_selected_doc_ids"
                                ]:
                                    planned_retrieval_info[
                                        "budgeted_evidence_subgraph_selected_doc_ids"
                                    ].append(int(doc_id))
                            subgraph_reason = str(subgraph_info.get("reason", "unavailable"))
                            subgraph_reason_counts = planned_retrieval_info.setdefault(
                                "budgeted_evidence_subgraph_reason_counts",
                                {},
                            )
                            subgraph_reason_counts[subgraph_reason] = (
                                int(subgraph_reason_counts.get(subgraph_reason, 0) or 0) + 1
                            )
                            planned_retrieval_info[
                                "support_set_coverage_preservation_rejected_count"
                            ] += int(
                                support_set_info.get(
                                    "coverage_preservation_rejected_count",
                                    0,
                                ) or 0
                            )
                            support_coverage_reason_counts = planned_retrieval_info.setdefault(
                                "support_set_coverage_preservation_reject_reason_counts",
                                {},
                            )
                            for reason, count in (
                                support_set_info.get(
                                    "coverage_preservation_reject_reason_counts",
                                    {},
                                ) or {}
                            ).items():
                                support_coverage_reason_counts[str(reason)] = (
                                    int(support_coverage_reason_counts.get(str(reason), 0) or 0)
                                    + int(count or 0)
                                )
                            top2_info = planned_merge_info.get("top2_override") or {}
                            pairwise_info = top2_info.get("pairwise_utility_search") or {}
                            planned_retrieval_info["query_schema_memory_pairwise_candidate_count"] += int(
                                pairwise_info.get("query_schema_memory_candidate_count", 0) or 0
                            )
                            planned_retrieval_info["query_schema_memory_pairwise_duplicate_count"] += int(
                                pairwise_info.get("query_schema_memory_duplicate_count", 0) or 0
                            )
                            planned_retrieval_info["query_schema_intervention_allow_top2_count"] += int(
                                pairwise_info.get("query_schema_intervention_allow_top2_count", 0) or 0
                            )
                            planned_retrieval_info["query_schema_intervention_append_only_count"] += int(
                                pairwise_info.get("query_schema_intervention_append_only_count", 0) or 0
                            )
                            planned_retrieval_info["query_schema_intervention_reject_count"] += int(
                                pairwise_info.get("query_schema_intervention_reject_count", 0) or 0
                            )
                            for reason, count in (
                                pairwise_info.get(
                                    "query_schema_intervention_reason_counts",
                                    {},
                                ) or {}
                            ).items():
                                reason_key = str(reason)
                                planned_retrieval_info["query_schema_intervention_reason_counts"][reason_key] = (
                                    int(
                                        planned_retrieval_info[
                                            "query_schema_intervention_reason_counts"
                                        ].get(reason_key, 0)
                                        or 0
                                    )
                                    + int(count or 0)
                                )
                            planned_retrieval_info["schema_verified_prior_applied_count"] += int(
                                pairwise_info.get("schema_verified_prior_applied_count", 0) or 0
                            )
                            planned_retrieval_info["schema_verified_prior_rescue_count"] += int(
                                pairwise_info.get("schema_verified_prior_rescue_count", 0) or 0
                            )
                            planned_retrieval_info["schema_verified_prior_bonus_total"] += float(
                                pairwise_info.get("schema_verified_prior_bonus_total", 0.0) or 0.0
                            )
                            planned_retrieval_info["schema_deficit_promotion_accepted_count"] += int(
                                pairwise_info.get("schema_deficit_promotion_accepted_count", 0) or 0
                            )
                            planned_retrieval_info["schema_deficit_promotion_rejected_count"] += int(
                                pairwise_info.get("schema_deficit_promotion_rejected_count", 0) or 0
                            )
                            for reason, count in (
                                pairwise_info.get("schema_deficit_promotion_reason_counts", {}) or {}
                            ).items():
                                reason_key = str(reason)
                                current = planned_retrieval_info[
                                    "schema_deficit_promotion_reason_counts"
                                ].get(reason_key, 0)
                                planned_retrieval_info[
                                    "schema_deficit_promotion_reason_counts"
                                ][reason_key] = int(current or 0) + int(count or 0)
                            for reason, count in (
                                pairwise_info.get("schema_deficit_promotion_reject_reason_counts", {}) or {}
                            ).items():
                                reason_key = str(reason)
                                current = planned_retrieval_info[
                                    "schema_deficit_promotion_reject_reason_counts"
                                ].get(reason_key, 0)
                                planned_retrieval_info[
                                    "schema_deficit_promotion_reject_reason_counts"
                                ][reason_key] = int(current or 0) + int(count or 0)
                            planned_retrieval_info["schema_deficit_soft_append_accepted_count"] += int(
                                pairwise_info.get("schema_deficit_soft_append_accepted_count", 0) or 0
                            )
                            planned_retrieval_info["schema_deficit_soft_append_rejected_count"] += int(
                                pairwise_info.get("schema_deficit_soft_append_rejected_count", 0) or 0
                            )
                            for doc_id in pairwise_info.get("schema_deficit_soft_append_doc_ids", []) or []:
                                if doc_id not in planned_retrieval_info["schema_deficit_soft_append_doc_ids"]:
                                    planned_retrieval_info["schema_deficit_soft_append_doc_ids"].append(int(doc_id))
                            for reason, count in (
                                pairwise_info.get("schema_deficit_soft_append_reason_counts", {}) or {}
                            ).items():
                                reason_key = str(reason)
                                current = planned_retrieval_info[
                                    "schema_deficit_soft_append_reason_counts"
                                ].get(reason_key, 0)
                                planned_retrieval_info[
                                    "schema_deficit_soft_append_reason_counts"
                                ][reason_key] = int(current or 0) + int(count or 0)
                            for reason, count in (
                                pairwise_info.get("schema_deficit_soft_append_reject_reason_counts", {}) or {}
                            ).items():
                                reason_key = str(reason)
                                current = planned_retrieval_info[
                                    "schema_deficit_soft_append_reject_reason_counts"
                                ].get(reason_key, 0)
                                planned_retrieval_info[
                                    "schema_deficit_soft_append_reject_reason_counts"
                                ][reason_key] = int(current or 0) + int(count or 0)
                            planned_retrieval_info["schema_reliability_evaluated_count"] += int(
                                pairwise_info.get("schema_reliability_evaluated_count", 0) or 0
                            )
                            planned_retrieval_info["schema_reliability_passed_count"] += int(
                                pairwise_info.get("schema_reliability_passed_count", 0) or 0
                            )
                            planned_retrieval_info["schema_reliability_rescued_count"] += int(
                                pairwise_info.get("schema_reliability_rescued_count", 0) or 0
                            )
                            planned_retrieval_info["schema_reliability_rejected_count"] += int(
                                pairwise_info.get("schema_reliability_rejected_count", 0) or 0
                            )
                            for reason, count in (
                                pairwise_info.get("schema_reliability_reason_counts", {}) or {}
                            ).items():
                                reason_key = str(reason)
                                current = planned_retrieval_info[
                                    "schema_reliability_reason_counts"
                                ].get(reason_key, 0)
                                planned_retrieval_info[
                                    "schema_reliability_reason_counts"
                                ][reason_key] = int(current or 0) + int(count or 0)
                            for reason, count in (
                                pairwise_info.get("schema_reliability_reject_reason_counts", {}) or {}
                            ).items():
                                reason_key = str(reason)
                                current = planned_retrieval_info[
                                    "schema_reliability_reject_reason_counts"
                                ].get(reason_key, 0)
                                planned_retrieval_info[
                                    "schema_reliability_reject_reason_counts"
                                ][reason_key] = int(current or 0) + int(count or 0)
                            planned_retrieval_info["answer_type_conflict_guard_evaluated_count"] += int(
                                pairwise_info.get("answer_type_conflict_guard_evaluated_count", 0) or 0
                            )
                            planned_retrieval_info["answer_type_conflict_guard_rejected_count"] += int(
                                pairwise_info.get("answer_type_conflict_guard_rejected_count", 0) or 0
                            )
                            answer_type_reject_counts = pairwise_info.get(
                                "answer_type_conflict_guard_reject_reason_counts"
                            ) or {}
                            if answer_type_reject_counts:
                                merged_answer_type_counts = dict(
                                    planned_retrieval_info.get(
                                        "answer_type_conflict_guard_reject_reason_counts",
                                        {},
                                    )
                                )
                                for reason, count in answer_type_reject_counts.items():
                                    merged_answer_type_counts[reason] = (
                                        int(merged_answer_type_counts.get(reason, 0) or 0)
                                        + int(count or 0)
                                    )
                                planned_retrieval_info[
                                    "answer_type_conflict_guard_reject_reason_counts"
                                ] = merged_answer_type_counts
                            planned_retrieval_info["bridge_slot_replace_accepted_count"] += int(
                                pairwise_info.get("bridge_slot_rescue_accepted_count", 0) or 0
                            )
                            planned_retrieval_info["bridge_slot_append_accepted_count"] += int(
                                pairwise_info.get("bridge_slot_append_accepted_count", 0) or 0
                            )
                            planned_retrieval_info["object_memory_accepted_count"] = (
                                planned_retrieval_info.get("object_memory_accepted_count", 0)
                                + int(pairwise_info.get("object_memory_accepted_count", 0) or 0)
                            )
                            planned_retrieval_info["object_memory_rejected_count"] = (
                                planned_retrieval_info.get("object_memory_rejected_count", 0)
                                + int(pairwise_info.get("object_memory_rejected_count", 0) or 0)
                            )
                            planned_retrieval_info["object_memory_minimal_sentence_count"] = (
                                planned_retrieval_info.get("object_memory_minimal_sentence_count", 0)
                                + int(pairwise_info.get("object_memory_minimal_sentence_count", 0) or 0)
                            )
                            append_memory_info = planned_merge_info.get("object_memory_append_only") or {}
                            planned_retrieval_info["object_memory_append_only_accepted_count"] = (
                                planned_retrieval_info.get("object_memory_append_only_accepted_count", 0)
                                + int(append_memory_info.get("accepted_count", 0) or 0)
                            )
                            planned_retrieval_info["object_memory_append_only_rejected_count"] = (
                                planned_retrieval_info.get("object_memory_append_only_rejected_count", 0)
                                + int(append_memory_info.get("rejected_count", 0) or 0)
                            )
                            planned_retrieval_info["object_memory_append_only_candidate_count"] = (
                                planned_retrieval_info.get("object_memory_append_only_candidate_count", 0)
                                + int(append_memory_info.get("candidate_count", 0) or 0)
                            )
                            for doc_id in append_memory_info.get("accepted_doc_ids", []) or []:
                                if doc_id not in planned_retrieval_info["object_memory_append_only_doc_ids"]:
                                    planned_retrieval_info["object_memory_append_only_doc_ids"].append(
                                        int(doc_id)
                                    )
                            for doc_id in pairwise_info.get("bridge_slot_append_doc_ids", []) or []:
                                if doc_id not in planned_retrieval_info["bridge_slot_append_doc_ids"]:
                                    planned_retrieval_info["bridge_slot_append_doc_ids"].append(int(doc_id))
                            for reason, count in (
                                append_memory_info.get("reject_reason_counts", {}) or {}
                            ).items():
                                append_counts = planned_retrieval_info.setdefault(
                                    "object_memory_append_only_reject_reason_counts",
                                    {},
                                )
                                append_counts[reason] = append_counts.get(reason, 0) + int(count or 0)
                            for reason, count in (
                                pairwise_info.get("object_memory_reject_reason_counts", {}) or {}
                            ).items():
                                object_counts = planned_retrieval_info.setdefault(
                                    "object_memory_reject_reason_counts",
                                    {},
                                )
                                object_counts[reason] = object_counts.get(reason, 0) + int(count or 0)
                            for reason, count in (
                                pairwise_info.get("bridge_slot_rescue_reject_reason_counts", {}) or {}
                            ).items():
                                current = planned_retrieval_info["bridge_slot_reject_reason_counts"].get(
                                    reason,
                                    0,
                                )
                                planned_retrieval_info["bridge_slot_reject_reason_counts"][reason] = (
                                    current + int(count or 0)
                                )

                            next_doc_ids = [
                                int(idx)
                                for idx in sorted_doc_ids[:self.global_config.qa_top_k]
                            ]
                            next_seen_doc_ids = set(next_doc_ids)
                            for doc_id in planned_extra_doc_ids:
                                if doc_id in next_seen_doc_ids:
                                    continue
                                next_doc_ids.append(int(doc_id))
                                next_seen_doc_ids.add(int(doc_id))
                            after_evidence_signature = build_planned_retrieval_evidence_signature(
                                next_doc_ids,
                                planned_extra_memory_docs,
                            )
                            decision_frontier_info = (
                                build_planned_retrieval_decision_frontier(
                                    planned_merge_info,
                                    materialized_doc_ids=next_doc_ids,
                                    late_residual_doc_ids=(
                                        late_residual_appended_doc_ids
                                    ),
                                )
                            )
                            round_info["decision_relevant_frontier"] = {
                                key: value
                                for key, value in decision_frontier_info.items()
                                if key != "signature"
                            }
                            if (
                                planned_round_cost_controller.candidate_frontier_mode
                                == "decision_relevant"
                            ):
                                controller_candidate_signature = tuple(
                                    decision_frontier_info.get("signature", ())
                                )
                            else:
                                controller_candidate_signature = (
                                    retrieval_candidate_signature
                                )
                            round_cost_decision = planned_round_cost_controller.observe(
                                round_number=round_idx + 1,
                                max_rounds=max_rounds,
                                verdict=verdict,
                                before_evidence_signature=before_evidence_signature,
                                after_evidence_signature=after_evidence_signature,
                                applied=applied_this_round,
                                query_count=len(planned_queries),
                                candidate_signature=controller_candidate_signature,
                                actionable_candidate_count=int(
                                    decision_frontier_info.get(
                                        "actionable_candidate_count", 0
                                    )
                                    or 0
                                ),
                                best_candidate_utility=float(
                                    decision_frontier_info.get(
                                        "best_candidate_utility", 0.0
                                    )
                                    or 0.0
                                ),
                            )
                            round_info["round_cost_decision"] = round_cost_decision
                            planned_retrieval_info["round_cost_controller"] = (
                                planned_round_cost_controller.summary()
                            )
                            planned_retrieval_info["rounds"].append(round_info)
                            if round_cost_decision.get("decision") == "STOP":
                                planned_retrieval_info["reason"] = (
                                    "cost_controller_fixed_point_stop"
                                )
                                planned_retrieval_info[
                                    "expansion_stopped_by_optimizer"
                                ] = True
                                planned_retrieval_info[
                                    "terminal_evidence_gap"
                                ] = True
                                planned_retrieval_info[
                                    "terminal_materialization_required"
                                ] = True
                                round_info["execution_action"] = "STOP_EXPANSION"
                                round_info["next_execution_phase"] = "FINALIZE"
                                round_info["reason"] = (
                                    "cost_controller_fixed_point_stop"
                                )
                                break
                        else:
                            if planned_retrieval_info["applied"]:
                                planned_retrieval_info["reason"] = "max_rounds_reached"
                            else:
                                planned_retrieval_info["reason"] = "verifier_insufficient_no_retrieval"

                    planned_retrieval_info = (
                        annotate_planned_retrieval_terminal_state(
                            planned_retrieval_info
                        )
                    )
                    planned_retrieval_info["extra_evidence_doc_count"] = len(planned_extra_doc_ids)
                    planned_retrieval_info["extra_evidence_doc_ids"] = planned_extra_doc_ids[:8]
                    planned_retrieval_info["extra_evidence_memory_count"] = len(planned_extra_memory_docs)
                    planned_retrieval_info["extra_evidence_memory_titles"] = [
                        doc.split("\n", 1)[0].strip() for doc in planned_extra_memory_docs[:8]
                    ]

            entity_predicate_join_enabled = bool(
                getattr(
                    self.global_config,
                    "enable_entity_predicate_fact_join",
                    False,
                )
            )
            cross_schema_memory_join_enabled = bool(
                getattr(
                    self.global_config,
                    "enable_query_schema_memory_cross_schema_join",
                    False,
                )
            )
            fact_join_executor_enabled = bool(
                entity_predicate_join_enabled or cross_schema_memory_join_enabled
            )
            entity_predicate_join_started = time.perf_counter()
            entity_predicate_fact_join_info = build_entity_predicate_fact_join_candidates(
                question=query,
                sorted_doc_ids=sorted_doc_ids,
                top_k_facts=top_k_facts,
                candidate_facts=(rerank_log or {}).get("facts_before_rerank", []),
                relation_index=getattr(self, "relation_fact_join_index", {}),
                get_doc_content=lambda doc_idx: self.chunk_embedding_store.get_row(
                    self.passage_node_keys[int(doc_idx)]
                )["content"],
                qa_top_k=self.global_config.qa_top_k,
                enabled=bool(
                    fact_join_executor_enabled
                ),
                bridge_only=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_bridge_only",
                    True,
                ),
                require_evidence_gap=(
                    getattr(
                        self.global_config,
                        "entity_predicate_fact_join_require_evidence_gap",
                        False,
                    )
                    if entity_predicate_join_enabled
                    else False
                ),
                max_active_facts=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_max_active_facts",
                    16,
                ),
                max_entity_fanout=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_max_entity_fanout",
                    240,
                ),
                candidate_top_k=(
                    getattr(
                        self.global_config,
                        "entity_predicate_fact_join_candidate_top_k",
                        48,
                    )
                    if entity_predicate_join_enabled
                    else getattr(
                        self.global_config,
                        "query_schema_memory_cross_schema_max_candidates",
                        48,
                    )
                ),
                max_original_rank=(
                    getattr(
                        self.global_config,
                        "entity_predicate_fact_join_max_original_rank",
                        200,
                    )
                    if entity_predicate_join_enabled
                    else getattr(
                        self.global_config,
                        "query_schema_memory_cross_schema_max_original_rank",
                        200,
                    )
                ),
                min_relation_overlap=(
                    getattr(
                        self.global_config,
                        "entity_predicate_fact_join_min_relation_overlap",
                        0.34,
                    )
                    if entity_predicate_join_enabled
                    else getattr(
                        self.global_config,
                        "query_schema_memory_cross_schema_min_relation_overlap",
                        0.20,
                    )
                ),
                min_score=(
                    getattr(
                        self.global_config,
                        "entity_predicate_fact_join_min_score",
                        0.58,
                    )
                    if entity_predicate_join_enabled
                    else getattr(
                        self.global_config,
                        "query_schema_memory_cross_schema_min_score",
                        0.58,
                    )
                ),
                planned_retrieval=planned_retrieval_info,
                lineage_constrained=(
                    getattr(
                        self.global_config,
                        "entity_predicate_fact_join_lineage_constrained",
                        True,
                    )
                    if fact_join_executor_enabled
                    else False
                ),
                max_lineage_hops=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_max_lineage_hops",
                    2,
                ),
                require_atomic_slot=(
                    getattr(
                        self.global_config,
                        "entity_predicate_fact_join_require_atomic_slot",
                        True,
                    )
                    if fact_join_executor_enabled
                    else False
                ),
                residual_deficit_trigger=(
                    getattr(
                        self.global_config,
                        "entity_predicate_fact_join_residual_deficit_trigger",
                        False,
                    )
                    if fact_join_executor_enabled
                    else False
                ),
                typed_residual_plan=(
                    getattr(
                        self.global_config,
                        "entity_predicate_fact_join_typed_residual_plan",
                        False,
                    )
                    if fact_join_executor_enabled
                    else False
                ),
                max_exact_signature_fanout=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_max_exact_signature_fanout",
                    32,
                ),
                catalog_guided_executor=(
                    getattr(
                        self.global_config,
                        "entity_predicate_fact_join_catalog_guided_executor",
                        False,
                    )
                    if fact_join_executor_enabled
                    else False
                ),
                operator_faithful_executor=(
                    getattr(
                        self.global_config,
                        "entity_predicate_fact_join_operator_faithful_executor",
                        False,
                    )
                    if fact_join_executor_enabled
                    else False
                ),
                validate_operator_completion=(
                    getattr(
                        self.global_config,
                        "entity_predicate_fact_join_validate_operator_completion",
                        False,
                    )
                    if fact_join_executor_enabled
                    else False
                ),
                validated_dense_top_k=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_validated_dense_top_k",
                    1,
                ),
                plan_stage_gate=(
                    getattr(
                        self.global_config,
                        "entity_predicate_fact_join_plan_stage_gate",
                        False,
                    )
                    if fact_join_executor_enabled
                    else False
                ),
                signature_compatibility_fallback=(
                    getattr(
                        self.global_config,
                        "entity_predicate_fact_join_signature_compatibility_fallback",
                        False,
                    )
                    if fact_join_executor_enabled
                    else False
                ),
                operator_retriever=(
                    lambda operator_query, top_k: tuple(
                        values[: max(1, int(top_k))]
                        for values in self.dense_passage_retrieval(operator_query)
                    )
                ),
                max_candidates_per_slot=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_max_candidates_per_slot",
                    2,
                ),
                query_candidate_budget=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_query_candidate_budget",
                    4,
                ),
                generic_signature_max_fanout=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_generic_signature_max_fanout",
                    4,
                ),
            )
            entity_predicate_join_latency_ms = max(
                0.0,
                (time.perf_counter() - entity_predicate_join_started) * 1000.0,
            )
            entity_predicate_items = list(
                (entity_predicate_fact_join_info or {}).get("candidate_items") or []
            )
            entity_predicate_candidate_latency_ms = (
                entity_predicate_join_latency_ms
                / float(max(1, len(entity_predicate_items)))
            )
            entity_predicate_fact_join_info["operator_latency_ms"] = round(
                entity_predicate_join_latency_ms,
                6,
            )
            entity_predicate_fact_join_info["physical_operator"] = (
                "entity_predicate_index_nested_loop_join"
            )
            for candidate in entity_predicate_items:
                candidate["operator_latency_ms"] = round(
                    entity_predicate_candidate_latency_ms,
                    6,
                )
                candidate["operator_name"] = (
                    "entity_predicate_index_nested_loop_join"
                )
            entity_predicate_fact_join_info[
                "schema_memory_cross_schema_access"
            ] = bool(cross_schema_memory_join_enabled)
            entity_predicate_fact_join_info[
                "standalone_entity_predicate_enabled"
            ] = bool(entity_predicate_join_enabled)

            chain_candidate_top_k = min(
                max(
                    self.global_config.qa_top_k,
                    getattr(self.global_config, "constrained_chain_rerank_candidate_top_k", 200),
                ),
                len(sorted_doc_ids),
            )
            chain_candidate_docs = [
                self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                for idx in sorted_doc_ids[:chain_candidate_top_k]
            ]
            chain_evidence_insufficient = (
                iterative_retrieval_info.get("reason") == "max_steps_reached"
                and planned_retrieval_info.get("reason") != "verified_sufficient"
            )
            sorted_doc_ids, sorted_doc_scores, chain_rerank_info = constrained_chain_rerank_docs(
                query=query,
                sorted_doc_ids=sorted_doc_ids,
                sorted_doc_scores=sorted_doc_scores,
                candidate_docs=chain_candidate_docs,
                passage_node_keys=self.passage_node_keys,
                chunk_path_info=getattr(self, "chunk_path_info", {}),
                top_k_facts=top_k_facts,
                top_k_fact_indices=top_k_fact_indices,
                query_fact_scores=query_fact_scores,
                qa_top_k=self.global_config.qa_top_k,
                candidate_top_k=getattr(self.global_config, "constrained_chain_rerank_candidate_top_k", 200),
                protect_top_k=getattr(self.global_config, "constrained_chain_rerank_protect_top_k", 1),
                max_promotions=getattr(self.global_config, "constrained_chain_rerank_max_promotions", 2),
                min_chain_score=getattr(self.global_config, "constrained_chain_rerank_min_score", 0.30),
                enabled=getattr(self.global_config, "enable_constrained_chain_rerank", False),
                bridge_only=getattr(self.global_config, "constrained_chain_rerank_bridge_only", True),
                has_bridge_cues=has_bridge_cues(query),
                require_insufficient=getattr(self.global_config, "constrained_chain_rerank_require_insufficient", True),
                evidence_insufficient=chain_evidence_insufficient,
            )
            path_evidence_doc_ids, path_evidence_info = select_path_aware_extra_doc_ids(
                query=query,
                sorted_doc_ids=sorted_doc_ids,
                sorted_doc_scores=sorted_doc_scores,
                passage_node_keys=self.passage_node_keys,
                chunk_path_info=getattr(self, "chunk_path_info", {}),
                top_k_facts=top_k_facts,
                top_k_fact_indices=top_k_fact_indices,
                query_fact_scores=query_fact_scores,
                qa_top_k=self.global_config.qa_top_k,
                candidate_top_k=getattr(self.global_config, "path_aware_evidence_top_k", 100),
                max_extra=getattr(self.global_config, "path_aware_evidence_max_extra", 2),
                min_path_score=getattr(self.global_config, "path_aware_evidence_min_score", 0.22),
                enabled=getattr(self.global_config, "enable_path_aware_evidence", False),
                bridge_only=getattr(self.global_config, "path_aware_evidence_bridge_only", True),
                has_bridge_cues=has_bridge_cues(query),
            )
            path_evidence_docs = [
                self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                for idx in path_evidence_doc_ids
            ]
            path_evidence_info["selected_titles"] = [
                doc.split("\n", 1)[0].strip() for doc in path_evidence_docs
            ]

            sorted_doc_ids, sorted_doc_scores, path_rerank_info = path_aware_rerank_docs(
                query=query,
                sorted_doc_ids=sorted_doc_ids,
                sorted_doc_scores=sorted_doc_scores,
                passage_node_keys=self.passage_node_keys,
                chunk_path_info=getattr(self, "chunk_path_info", {}),
                top_k_facts=top_k_facts,
                top_k_fact_indices=top_k_fact_indices,
                query_fact_scores=query_fact_scores,
                qa_top_k=self.global_config.qa_top_k,
                rerank_top_k=getattr(self.global_config, "path_aware_rerank_top_k", 200),
                weight=getattr(self.global_config, "path_aware_rerank_weight", 0.30),
                enabled=getattr(self.global_config, "enable_path_aware_rerank", False),
                bridge_only=getattr(self.global_config, "path_aware_rerank_bridge_only", True),
                has_bridge_cues=has_bridge_cues(query),
            )

            (
                sorted_doc_ids,
                sorted_doc_scores,
                fact_source_scan_info,
            ) = promote_fact_source_docs(
                sorted_doc_ids,
                sorted_doc_scores,
                top_k_facts=top_k_facts,
                top_k_fact_indices=top_k_fact_indices,
                query_fact_scores=query_fact_scores,
                relation_index=getattr(self, "relation_fact_join_index", {}),
                executor_stop_reason=str(
                    (paper_evidence_execution_info or {}).get("stop_reason")
                    or ""
                ),
                enabled=bool(
                    getattr(self.global_config, "enable_fact_source_scan", False)
                ),
                fallback_only=bool(
                    getattr(
                        self.global_config,
                        "fact_source_scan_fallback_only",
                        True,
                    )
                ),
                fact_top_k=getattr(
                    self.global_config,
                    "fact_source_scan_fact_top_k",
                    8,
                ),
                target_top_k=getattr(
                    self.global_config,
                    "fact_source_scan_target_top_k",
                    self.global_config.qa_top_k,
                ),
                max_promotions=getattr(
                    self.global_config,
                    "fact_source_scan_max_promotions",
                    1,
                ),
                min_fact_score=getattr(
                    self.global_config,
                    "fact_source_scan_min_fact_score",
                    0.60,
                ),
                max_source_docs_per_fact=getattr(
                    self.global_config,
                    "fact_source_scan_max_source_docs_per_fact",
                    2,
                ),
                logical_plan=(paper_evidence_execution_info or {}).get(
                    "logical_plan"
                ) or {},
                selection_mode=getattr(
                    self.global_config,
                    "fact_source_scan_selection_mode",
                    "legacy",
                ),
                protect_dense_top_k=getattr(
                    self.global_config,
                    "fact_source_scan_protect_dense_top_k",
                    1,
                ),
            )
            fact_source_scan_counts[
                f"{fact_source_scan_info.get('reason', 'unavailable')}|"
                f"{len(fact_source_scan_info.get('promoted_doc_ids') or [])}"
            ] += 1

            (
                sorted_doc_ids,
                sorted_doc_scores,
                plan_bound_retrieval_projection,
            ) = promote_plan_bound_evidence_docs(
                sorted_doc_ids,
                sorted_doc_scores,
                relation_fact_join_info,
                paper_evidence_execution_info,
                enabled=bool(getattr(
                    self.global_config,
                    "paper_evidence_promote_plan_bound_docs",
                    False,
                )),
                protect_dense_top_k=getattr(
                    self.global_config,
                    "paper_evidence_promotion_protect_dense_top_k",
                    1,
                ),
                max_title_docs=getattr(
                    self.global_config,
                    "paper_evidence_promotion_max_title_docs",
                    3,
                ),
                selection_mode=getattr(
                    self.global_config,
                    "paper_evidence_promotion_selection_mode",
                    "legacy",
                ),
                target_top_k=getattr(
                    self.global_config,
                    "paper_evidence_promotion_target_top_k",
                    5,
                ),
                dense_candidate_top_k=getattr(
                    self.global_config,
                    "paper_evidence_promotion_dense_candidate_top_k",
                    10,
                ),
            )
            relation_fact_join_info["retrieval_projection"] = dict(
                plan_bound_retrieval_projection
            )
            paper_evidence_execution_info["retrieval_projection"] = dict(
                plan_bound_retrieval_projection
            )

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]
            planned_evidence_docs = [
                self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                for idx in (planned_retrieval_info.get("extra_evidence_doc_ids") or [])
            ]
            planned_evidence_docs = list(planned_extra_memory_docs) + planned_evidence_docs
            planned_retrieval_info["extra_evidence_titles"] = [
                doc.split("\n", 1)[0].strip() for doc in planned_evidence_docs
            ]

            retrieval_results.append(QuerySolution(
                question=query,
                docs=top_k_docs,
                retrieved_doc_ids=[int(idx) for idx in sorted_doc_ids[:num_to_retrieve]],
                sample_key=(
                    str(sample_keys[q_idx])
                    if sample_keys is not None
                    else f"{self.global_config.dataset}:{q_idx}"
                ),
                doc_scores=sorted_doc_scores[:num_to_retrieve],
                path_rerank=path_rerank_info,
                fact_rerank=fact_rerank_info,
                fact_source_scan=fact_source_scan_info,
                retrieval_backend=self._retrieval_backend_diagnostics(),
                graph_repair=graph_repair_info,
                query_schema_memory=query_schema_memory_info,
                adaptive_query_execution=adaptive_query_execution_info,
                paper_evidence_execution=paper_evidence_execution_info,
                entity_predicate_fact_join=entity_predicate_fact_join_info,
                relation_fact_join=relation_fact_join_info,
                typed_fact_join_shadow=typed_fact_join_shadow_info,
                selective_support_join=selective_support_join_info,
                selective_support_join_docs=selective_support_join_docs,
                relation_fact_join_docs=relation_fact_join_docs,
                support_rerank=support_rerank_info,
                dual_channel_projection=dual_channel_projection_info,
                dual_channel_projection_docs=dual_channel_projection_docs,
                chain_rerank=chain_rerank_info,
                planned_retrieval=planned_retrieval_info,
                iterative_retrieval=iterative_retrieval_info,
                path_evidence=path_evidence_info,
                path_evidence_docs=path_evidence_docs,
                planned_evidence_docs=planned_evidence_docs,
                qa_timing={
                    "retrieval_seconds": round(
                        time.perf_counter() - query_retrieval_started,
                        6,
                    ),
                    "retrieval_llm_prefetch_enabled": bool(
                        retrieval_llm_prefetch is not None
                    ),
                    "retrieval_llm_prefetch_wait_seconds": round(
                        remote_prefetch_wait_seconds,
                        6,
                    ),
                    "retrieval_llm_stage_work_seconds": round(
                        remote_stage_work_seconds,
                        6,
                    ),
                },
            ))

        if retrieval_llm_prefetch is not None:
            retrieval_llm_prefetch.close()

        logger.info(f"Typed fact-join shadow optimizer counts: {dict(typed_fact_join_shadow_counts)}")
        logger.info(f"Selective support-join optimizer counts: {dict(selective_support_join_counts)}")
        logger.info(f"Fact-source scan counts: {dict(fact_source_scan_counts)}")
        for result in retrieval_results:
            trace = result.adaptive_query_execution or {}
            adaptive_query_execution_counts[
                f"{trace.get('selected_plan_expression', 'unavailable')}|"
                f"{trace.get('stop_reason', 'unavailable')}"
            ] += 1
        logger.info(
            "Adaptive lazy query executor counts: %s",
            dict(adaptive_query_execution_counts),
        )

        retrieve_end_time = time.time()

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")
        logger.info(f"Total Recognition Memory Time {self.rerank_time:.2f}s")
        logger.info(
            "Retrieval backend %s; Total PPR Time %.2fs; PPR calls %d",
            getattr(self.global_config, "retrieval_backend", "graph_ppr"),
            self.ppr_time,
            self.ppr_call_count,
        )
        logger.info(f"Total Misc Time {self.all_retrieval_time - (self.rerank_time + self.ppr_time):.2f}s")


        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results], k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None,
               sample_keys: List[str] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using S2RAG.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)


        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(
                    queries=queries,
                    gold_docs=gold_docs,
                    sample_keys=sample_keys,
                )
            else:
                queries = self.retrieve(queries=queries, sample_keys=sample_keys)


        queries_solutions, all_response_message, all_metadata = self.qa(queries)


        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)


            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")


            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def retrieve_dpr(self,
                     queries: List[str],
                     num_to_retrieve: int = None,
                     gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using a DPR framework, which consists of several steps:
        - Dense passage scoring

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            logger.info('No facts found after reranking, return DPR results')
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            fact_rerank_info = {
                "mode": "dense_only",
                "fallback_to_dense": True,
                "candidate_fact_count": 0,
                "reranked_fact_count": 0,
                "top_fact_count": 0,
                "error": None,
            }

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(
                QuerySolution(
                    question=query,
                    docs=top_k_docs,
                    retrieved_doc_ids=[int(idx) for idx in sorted_doc_ids[:num_to_retrieve]],
                    sample_key=f"{self.global_config.dataset}:{q_idx}",
                    doc_scores=sorted_doc_scores[:num_to_retrieve],
                    fact_rerank=fact_rerank_info,
                ))

        retrieve_end_time = time.time()

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")


        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa_dpr(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)


        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve_dpr(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve_dpr(queries=queries)


        queries_solutions, all_response_message, all_metadata = self.qa(queries)


        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)


            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")


            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """

        all_qa_messages = []
        answer_style_counts = Counter()
        evidence_selection_counts = Counter()
        bridge_contrastive_evidence_counts = Counter()
        path_evidence_counts = Counter()
        dual_channel_projection_counts = Counter()
        relation_fact_join_counts = Counter()
        selective_support_join_counts = Counter()
        multigranular_memory_counts = Counter()
        sentence_evidence_counts = Counter()
        anchored_typed_candidate_table_counts = Counter()
        structured_evidence_chain_counts = Counter()
        entity_state_evidence_counts = Counter()
        slot_projected_provenance_counts = Counter()
        slot_separated_fact_join_counts = Counter()
        entity_centric_fallback_counts = Counter()
        answer_slot_support_completion_counts = Counter()
        candidate_answer_selector_counts = Counter()
        support_set_completion_counts = Counter()
        evidence_admission_counts = Counter()
        schema_critical_evidence_counts = Counter()
        evidence_object_arbitration_counts = Counter()
        evidence_answer_arbiter_counts = Counter()
        typed_evidence_table_counts = Counter()
        contrastive_answer_arbitration_counts = Counter()
        provenance_answer_lattice_counts = Counter()
        provenance_typed_candidate_store_counts = Counter()
        evidence_grounded_answer_extractor_counts = Counter()
        query_answer_contract_counts = Counter()
        provenance_answer_slot_store_counts = Counter()
        answer_slot_binding_optimizer_counts = Counter()
        answer_surface_lattice_optimizer_counts = Counter()
        schema_surface_canonicalization_counts = Counter()
        plan_consistent_answer_verifier_counts = Counter()
        cost_gated_reread_counts = Counter()
        provenance_bound_answer_materializer_counts = Counter()
        final_reader_counterfactual_specs = []

        if self.prompt_template_manager.is_template_name_valid(
            name=f'rag_qa_{self.global_config.dataset}'
        ):
            prompt_dataset_name = self.global_config.dataset
        else:
            logger.debug(
                f"rag_qa_{self.global_config.dataset} does not have a customized "
                "prompt template. Using MUSIQUE's prompt template instead."
            )
            prompt_dataset_name = 'musique'

        def _render_reader_prompt(query_solution, passages):
            prompt_user = ''.join(
                f'Wikipedia Title: {passage}\n\n' for passage in passages
            )
            prompt_user += 'Question: ' + query_solution.question + '\n'
            if query_solution.answer_style_instruction:
                prompt_user += query_solution.answer_style_instruction + '\n'
            prompt_user += 'Thought: '
            return self.prompt_template_manager.render(
                name=f'rag_qa_{prompt_dataset_name}',
                prompt_user=prompt_user,
            )

        for query_solution_idx, query_solution in tqdm(
            enumerate(queries),
            total=len(queries),
            desc="Collecting QA prompts",
        ):
            evidence_preparation_started = time.perf_counter()
            if not query_solution.sample_key:
                query_solution.sample_key = (
                    f"{self.global_config.dataset}:{query_solution_idx}"
                )



            logical_plan_for_selection = dict(
                query_solution.logical_query_plan
                or (query_solution.paper_evidence_execution or {}).get("logical_plan")
                or {}
            )
            retrieved_passages, schema_evidence_set = select_schema_aware_evidence_set(
                question=query_solution.question,
                docs=query_solution.docs,
                doc_ids=query_solution.retrieved_doc_ids,
                doc_scores=query_solution.doc_scores,
                relation_index=getattr(self, "relation_fact_join_index", {}) or {},
                logical_plan=logical_plan_for_selection,
                enabled=getattr(
                    self.global_config, "enable_schema_evidence_set_selection", False
                ),
                mode=getattr(self.global_config, "schema_evidence_set_mode", "joint"),
                candidate_k=getattr(
                    self.global_config, "schema_evidence_set_candidate_k", 20
                ),
                target_k=getattr(
                    self.global_config,
                    "schema_evidence_set_target_k",
                    self.global_config.qa_top_k,
                ),
                protect_top_k=getattr(
                    self.global_config, "schema_evidence_set_protect_top_k", 1
                ),
                get_doc_content=lambda doc_idx: self.chunk_embedding_store.get_row(
                    self.passage_node_keys[int(doc_idx)]
                )["content"],
                index_max_rows_per_key=getattr(
                    self.global_config,
                    "schema_evidence_set_index_max_rows_per_key",
                    64,
                ),
                index_max_total_rows=getattr(
                    self.global_config,
                    "schema_evidence_set_index_max_total_rows",
                    512,
                ),
                index_max_fanout=getattr(
                    self.global_config,
                    "schema_evidence_set_index_max_fanout",
                    512,
                ),
            )
            query_solution.schema_evidence_set = schema_evidence_set
            if getattr(
                self.global_config, "enable_schema_evidence_set_selection", False
            ):
                evidence_selection = dict(schema_evidence_set)
            else:
                retrieved_passages, evidence_selection = select_adaptive_evidence_passages(
                    question=query_solution.question,
                    docs=query_solution.docs,
                    doc_scores=query_solution.doc_scores,
                    qa_top_k=self.global_config.qa_top_k,
                    enabled=self.global_config.enable_adaptive_evidence,
                    candidate_k=self.global_config.adaptive_evidence_candidate_k,
                    target_k=self.global_config.adaptive_evidence_target_k,
                    bridge_only=self.global_config.adaptive_evidence_bridge_only,
                )
            query_solution.evidence_selection = evidence_selection
            evidence_selection_counts[evidence_selection.get("reason", "unknown")] += 1
            stable_reader_doc_count = len(retrieved_passages)
            stable_base_passages = list(retrieved_passages)

            planned_info = dict(query_solution.planned_retrieval or {})
            planned_extra_docs = query_solution.planned_evidence_docs or []
            appended_planned_docs = []
            if planned_extra_docs:
                seen_docs = {str(doc).strip() for doc in retrieved_passages}
                for doc in planned_extra_docs:
                    doc_key = str(doc).strip()
                    if not doc_key or doc_key in seen_docs:
                        continue
                    appended_planned_docs.append(doc)
                    seen_docs.add(doc_key)
                retrieved_passages = retrieved_passages + appended_planned_docs
            planned_info["appended_to_qa"] = len(appended_planned_docs)
            planned_info["appended_titles"] = [
                str(doc).split("\n", 1)[0].strip() for doc in appended_planned_docs
            ]
            query_solution.planned_retrieval = planned_info

            path_evidence_info = dict(query_solution.path_evidence or {})
            path_extra_docs = query_solution.path_evidence_docs or []
            appended_path_docs = []
            if getattr(self.global_config, "enable_path_aware_evidence", False) and path_extra_docs:
                seen_docs = {str(doc).strip() for doc in retrieved_passages}
                max_extra = max(0, getattr(self.global_config, "path_aware_evidence_max_extra", 2))
                for doc in path_extra_docs:
                    doc_key = str(doc).strip()
                    if not doc_key or doc_key in seen_docs:
                        continue
                    appended_path_docs.append(doc)
                    seen_docs.add(doc_key)
                    if len(appended_path_docs) >= max_extra:
                        break
                retrieved_passages = retrieved_passages + appended_path_docs

            path_evidence_info["appended_to_qa"] = len(appended_path_docs)
            path_evidence_info["appended_titles"] = [
                str(doc).split("\n", 1)[0].strip() for doc in appended_path_docs
            ]
            query_solution.path_evidence = path_evidence_info
            path_evidence_counts[path_evidence_info.get("reason", "unavailable")] += 1

            retrieved_passages, bridge_contrastive_evidence = apply_bridge_contrastive_evidence_guard(
                question=query_solution.question,
                docs=retrieved_passages,
                evidence_selection=evidence_selection,
                planned_retrieval=planned_info,
                path_evidence=path_evidence_info,
                enabled=getattr(self.global_config, "enable_bridge_contrastive_evidence_guard", False),
                protect_top_k=getattr(
                    self.global_config,
                    "bridge_contrastive_guard_protect_top_k",
                    1,
                ),
                min_bridge_link=getattr(
                    self.global_config,
                    "bridge_contrastive_guard_min_bridge_link",
                    0.34,
                ),
                min_title_anchor_overlap=getattr(
                    self.global_config,
                    "bridge_contrastive_guard_min_title_anchor_overlap",
                    0.50,
                ),
                enable_counterfactual_safety=getattr(
                    self.global_config,
                    "enable_bridge_contrastive_counterfactual_safety",
                    True,
                ),
                counterfactual_safety_top_k=getattr(
                    self.global_config,
                    "bridge_contrastive_counterfactual_safety_top_k",
                    8,
                ),
                counterfactual_safety_min_score_drop=getattr(
                    self.global_config,
                    "bridge_contrastive_counterfactual_safety_min_score_drop",
                    0.35,
                ),
            )
            query_solution.bridge_contrastive_evidence = bridge_contrastive_evidence
            bridge_contrastive_evidence_counts[
                bridge_contrastive_evidence.get("reason", "unavailable")
            ] += 1

            dual_channel_info = dict(query_solution.dual_channel_projection or {})
            retrieved_passages, reader_projection = apply_dual_channel_reader_projection(
                docs=retrieved_passages,
                projection_docs=query_solution.dual_channel_projection_docs or [],
                retrieval_projection=dual_channel_info,
                enabled=getattr(self.global_config, "enable_dual_channel_evidence_projection", False),
                max_append=getattr(self.global_config, "dual_channel_projection_reader_max_append", 1),
            )
            dual_channel_info["reader_projection"] = reader_projection
            query_solution.dual_channel_projection = dual_channel_info
            dual_channel_projection_counts[
                f"{dual_channel_info.get('reason', 'unavailable')}|{reader_projection.get('reason', 'unavailable')}"
            ] += 1

            relation_fact_join_info = dict(query_solution.relation_fact_join or {})
            schema_evidence_mode = str(getattr(
                self.global_config,
                "schema_evidence_set_mode",
                "",
            )).strip().lower()
            preserve_complete_witness_projection = schema_evidence_mode in {
                "index_probe_path_safe",
                "index_probe_path_surface_safe",
            }
            selection_coupled_witness_projection = (
                schema_evidence_mode == "index_probe_path_surface_safe"
            )
            projection_closure = assess_complete_path_projection_closure(
                evidence_selection,
                relation_fact_join_info.get("accepted") or [],
            )
            complete_witness_projection_eligible = bool(
                not selection_coupled_witness_projection
                or projection_closure.get("eligible")
            )
            complete_witness_projection_gate_reason = ""
            if (
                selection_coupled_witness_projection
                and not complete_witness_projection_eligible
            ):
                complete_witness_projection_gate_reason = (
                    "complete_witness_projection_"
                    + str(projection_closure.get("reason") or "not_reader_closed")
                )
            relation_fact_join_info["reader_projection_closure"] = projection_closure
            retrieved_passages, relation_fact_join_reader = append_relation_fact_join_evidence(
                docs=retrieved_passages,
                join_docs=query_solution.relation_fact_join_docs or [],
                join_info=relation_fact_join_info,
                enabled=getattr(self.global_config, "enable_relation_fact_join_retrieval", False),
                max_append=getattr(self.global_config, "relation_fact_join_max_append", 1),
                append_mode=getattr(
                    self.global_config,
                    "relation_fact_join_append_mode",
                    "table",
                ),
                preserve_complete_witness_projection=(
                    preserve_complete_witness_projection
                ),
                complete_witness_projection_eligible=(
                    complete_witness_projection_eligible
                ),
                complete_witness_projection_gate_reason=(
                    complete_witness_projection_gate_reason
                ),
                source_surface_first=(
                    schema_evidence_mode == "index_probe_path_surface_safe"
                ),
            )
            relation_fact_join_info["reader_projection"] = relation_fact_join_reader
            query_solution.relation_fact_join = relation_fact_join_info
            relation_fact_join_counts[
                f"{relation_fact_join_info.get('reason', 'unavailable')}|"
                f"{relation_fact_join_reader.get('reason', 'unavailable')}"
            ] += 1

            selective_support_join_info = dict(query_solution.selective_support_join or {})
            retrieved_passages, selective_support_join_info = append_selective_support_join_evidence(
                docs=retrieved_passages,
                join_docs=query_solution.selective_support_join_docs or [],
                join_info=selective_support_join_info,
                enabled=getattr(
                    self.global_config,
                    "enable_selective_support_join_optimizer",
                    False,
                ),
                max_append=getattr(
                    self.global_config,
                    "selective_support_join_max_append",
                    1,
                ),
            )
            query_solution.selective_support_join = selective_support_join_info
            selective_support_reader = selective_support_join_info.get("reader_projection") or {}
            selective_support_join_counts[
                f"{selective_support_join_info.get('reason', 'unavailable')}|"
                f"{selective_support_reader.get('reason', 'unavailable')}|"
                f"{selective_support_reader.get('appended', 0)}"
            ] += 1

            retrieved_passages, support_set_completion = append_support_set_completion_evidence(
                question=query_solution.question,
                current_docs=retrieved_passages,
                ranked_docs=query_solution.docs or [],
                planned_retrieval=planned_info,
                enabled=getattr(self.global_config, "enable_support_set_completion_evidence", False),
                bridge_only=getattr(self.global_config, "support_set_completion_bridge_only", True),
                candidate_top_k=getattr(self.global_config, "support_set_completion_candidate_top_k", 20),
                max_append=getattr(self.global_config, "support_set_completion_max_append", 1),
                min_score=getattr(self.global_config, "support_set_completion_min_score", 0.54),
                min_missing_gain=getattr(self.global_config, "support_set_completion_min_missing_gain", 0.12),
                min_sentence_score=getattr(self.global_config, "support_set_completion_min_sentence_score", 0.30),
            )
            query_solution.support_set_completion = support_set_completion
            support_set_completion_counts[
                support_set_completion.get("reason", "unavailable")
            ] += 1

            retrieved_passages, evidence_admission = apply_cost_based_evidence_admission(
                question=query_solution.question,
                docs=retrieved_passages,
                base_doc_count=stable_reader_doc_count,
                planned_retrieval=planned_info,
                relation_fact_join=relation_fact_join_info,
                dual_channel_projection=dual_channel_info,
                support_set_completion=support_set_completion,
                path_evidence=path_evidence_info,
                typed_fact_join_shadow=query_solution.typed_fact_join_shadow,
                query_schema_memory=query_solution.query_schema_memory,
                entity_predicate_fact_join=query_solution.entity_predicate_fact_join,
                selective_support_join=selective_support_join_info,
                get_doc_content=lambda doc_idx: self.chunk_embedding_store.get_row(
                    self.passage_node_keys[int(doc_idx)]
                )["content"],
                enabled=getattr(self.global_config, "enable_cost_based_evidence_admission", False),
                trigger_policy=getattr(
                    self.global_config,
                    "cost_based_admission_trigger_policy",
                    "high_risk_only",
                ),
                admission_policy=getattr(
                    self.global_config,
                    "cost_based_admission_policy",
                    "learned",
                ),
                optimize_existing_extra_docs=getattr(
                    self.global_config,
                    "cost_based_admission_optimize_existing_extras",
                    False,
                ),
                max_extra_docs=getattr(self.global_config, "cost_based_admission_max_extra_docs", 2),
                token_budget=getattr(self.global_config, "cost_based_admission_token_budget", 900),
                min_utility=getattr(self.global_config, "cost_based_admission_min_utility", 0.32),
                max_risk=getattr(self.global_config, "cost_based_admission_max_risk", 0.62),
                min_net_utility=getattr(self.global_config, "cost_based_admission_min_net_utility", -0.04),
                min_query_overlap=getattr(self.global_config, "cost_based_admission_min_query_overlap", 0.05),
                token_weight=getattr(self.global_config, "cost_based_admission_token_weight", 0.12),
                compute_weight=getattr(self.global_config, "cost_based_admission_compute_weight", 0.04),
                risk_weight=getattr(self.global_config, "cost_based_admission_risk_weight", 0.42),
                redundancy_penalty=getattr(
                    self.global_config,
                    "cost_based_admission_redundancy_penalty",
                    0.08,
                ),
                preserve_best_on_empty=getattr(
                    self.global_config,
                    "cost_based_admission_preserve_best_on_empty",
                    True,
                ),
                deficit_aware_replacement_enabled=getattr(
                    self.global_config,
                    "cost_based_admission_deficit_replacement_enabled",
                    False,
                ),
                deficit_aware_replacement_min_margin=getattr(
                    self.global_config,
                    "cost_based_admission_deficit_replacement_min_margin",
                    0.08,
                ),
                deficit_aware_replacement_max_evictions=getattr(
                    self.global_config,
                    "cost_based_admission_deficit_replacement_max_evictions",
                    1,
                ),
                shadow_candidate_top_k=getattr(
                    self.global_config,
                    "cost_based_admission_shadow_candidate_top_k",
                    4,
                ),
                shadow_max_original_rank=getattr(
                    self.global_config,
                    "cost_based_admission_shadow_max_original_rank",
                    80,
                ),
                shadow_min_plan_utility=getattr(
                    self.global_config,
                    "cost_based_admission_shadow_min_plan_utility",
                    0.42,
                ),
                shadow_min_plan_net_utility=getattr(
                    self.global_config,
                    "cost_based_admission_shadow_min_plan_net_utility",
                    0.04,
                ),
                shadow_max_plan_risk=getattr(
                    self.global_config,
                    "cost_based_admission_shadow_max_plan_risk",
                    0.58,
                ),
                shadow_require_two_hop_for_bridge=getattr(
                    self.global_config,
                    "cost_based_admission_shadow_require_two_hop_for_bridge",
                    True,
                ),
                calibrated_shadow_admission=getattr(
                    self.global_config,
                    "cost_based_admission_calibrated_shadow_admission",
                    False,
                ),
                source_sufficiency_veto=getattr(
                    self.global_config,
                    "cost_based_admission_source_sufficiency_veto",
                    True,
                ),
                learned_cost_optimizer_enabled=getattr(
                    self.global_config,
                    "enable_learned_cost_optimizer",
                    False,
                ),
                learned_cost_optimizer_model_path=getattr(
                    self.global_config,
                    "learned_cost_optimizer_model_path",
                    "artifacts/learned_cost_optimizer/model.json",
                ),
                learned_cost_optimizer_min_expected_gain=getattr(
                    self.global_config,
                    "learned_cost_optimizer_min_expected_gain",
                    0.012,
                ),
                learned_cost_optimizer_min_confidence=getattr(
                    self.global_config,
                    "learned_cost_optimizer_min_confidence",
                    0.30,
                ),
                learned_cost_optimizer_min_support=getattr(
                    self.global_config,
                    "learned_cost_optimizer_min_support",
                    4,
                ),
                learned_cost_optimizer_max_degrade=getattr(
                    self.global_config,
                    "learned_cost_optimizer_max_degrade",
                    0.46,
                ),
                learned_cost_optimizer_veto_min_degrade_margin=getattr(
                    self.global_config,
                    "learned_cost_optimizer_veto_min_degrade_margin",
                    0.0,
                ),
                counterfactual_drop_rank=getattr(
                    self.global_config,
                    "cost_based_admission_counterfactual_drop_rank",
                    -1,
                ),
                counterfactual_slate_min_harm_signals=getattr(
                    self.global_config,
                    "counterfactual_slate_min_harm_signals",
                    2,
                ),
                counterfactual_slate_min_stable_datasets=getattr(
                    self.global_config,
                    "counterfactual_slate_min_stable_datasets",
                    3,
                ),
                fact_join_plan_optimizer_enabled=getattr(
                    self.global_config,
                    "cost_based_fact_join_plan_optimizer_enabled",
                    False,
                ),
                fact_join_plan_compute_budget=getattr(
                    self.global_config,
                    "cost_based_fact_join_plan_compute_budget",
                    1.0,
                ),
                fact_join_plan_min_objective=getattr(
                    self.global_config,
                    "cost_based_fact_join_plan_min_objective",
                    0.0,
                ),
                fact_join_plan_max_enumerated_candidates=getattr(
                    self.global_config,
                    "cost_based_fact_join_plan_max_enumerated_candidates",
                    12,
                ),
                route_early_schema_through_optimizer=getattr(
                    self.global_config,
                    "cost_based_admission_route_early_schema_through_optimizer",
                    False,
                ),
                candidate_pool_snapshot_dir=getattr(
                    self.global_config,
                    "cost_based_admission_candidate_pool_snapshot_dir",
                    "",
                ),
                candidate_pool_snapshot_mode=getattr(
                    self.global_config,
                    "cost_based_admission_candidate_pool_snapshot_mode",
                    "off",
                ),
                candidate_pool_snapshot_strict=getattr(
                    self.global_config,
                    "cost_based_admission_candidate_pool_snapshot_strict",
                    True,
                ),
                candidate_pool_snapshot_key=query_solution.sample_key or "",
                schema_materialization_enabled=getattr(
                    self.global_config,
                    "enable_query_schema_memory_materialization",
                    True,
                ),
                schema_materialization_max_candidates=getattr(
                    self.global_config,
                    "query_schema_memory_materialization_max_candidates",
                    1,
                ),
                schema_materialization_max_original_rank=getattr(
                    self.global_config,
                    "query_schema_memory_materialization_max_original_rank",
                    32,
                ),
                schema_materialization_min_score=getattr(
                    self.global_config,
                    "query_schema_memory_materialization_min_score",
                    0.70,
                ),
                schema_materialization_min_schema_idf=getattr(
                    self.global_config,
                    "query_schema_memory_materialization_min_schema_idf",
                    0.60,
                ),
                schema_materialization_min_query_overlap=getattr(
                    self.global_config,
                    "query_schema_memory_materialization_min_query_overlap",
                    0.10,
                ),
                schema_materialization_min_join_key_overlap=getattr(
                    self.global_config,
                    "query_schema_memory_materialization_min_join_key_overlap",
                    0.66,
                ),
                schema_materialization_min_relation_overlap=getattr(
                    self.global_config,
                    "query_schema_memory_materialization_min_relation_overlap",
                    0.50,
                ),
                schema_materialization_require_answer_type_value=getattr(
                    self.global_config,
                    "query_schema_memory_materialization_require_answer_type_value",
                    True,
                ),
                query_schema_memory_cross_schema_join_enabled=getattr(
                    self.global_config,
                    "enable_query_schema_memory_cross_schema_join",
                    False,
                ),
                query_schema_memory_cross_schema_max_candidates=getattr(
                    self.global_config,
                    "query_schema_memory_cross_schema_max_candidates",
                    48,
                ),
                query_schema_memory_cross_schema_max_endpoint_scan_rows=getattr(
                    self.global_config,
                    "query_schema_memory_cross_schema_max_endpoint_scan_rows",
                    48,
                ),
                early_schema_bridge_materialization_enabled=getattr(
                    self.global_config,
                    "enable_early_schema_bridge_materialization",
                    False,
                ),
                early_schema_bridge_materialization_max_candidates=getattr(
                    self.global_config,
                    "early_schema_bridge_materialization_max_candidates",
                    1,
                ),
                early_schema_bridge_materialization_max_original_rank=getattr(
                    self.global_config,
                    "early_schema_bridge_materialization_max_original_rank",
                    80,
                ),
                early_schema_bridge_materialization_min_score=getattr(
                    self.global_config,
                    "early_schema_bridge_materialization_min_score",
                    0.66,
                ),
                early_schema_bridge_materialization_min_schema_idf=getattr(
                    self.global_config,
                    "early_schema_bridge_materialization_min_schema_idf",
                    0.35,
                ),
                early_schema_bridge_materialization_min_query_overlap=getattr(
                    self.global_config,
                    "early_schema_bridge_materialization_min_query_overlap",
                    0.08,
                ),
                early_schema_bridge_materialization_min_join_key_overlap=getattr(
                    self.global_config,
                    "early_schema_bridge_materialization_min_join_key_overlap",
                    0.50,
                ),
                early_schema_bridge_materialization_min_relation_overlap=getattr(
                    self.global_config,
                    "early_schema_bridge_materialization_min_relation_overlap",
                    0.20,
                ),
                early_schema_bridge_materialization_require_answer_type_value=getattr(
                    self.global_config,
                    "early_schema_bridge_materialization_require_answer_type_value",
                    True,
                ),
                early_schema_bridge_materialization_require_lineage_binding=getattr(
                    self.global_config,
                    "early_schema_bridge_materialization_require_lineage_binding",
                    True,
                ),
                early_schema_bridge_materialization_allow_relation_relaxation=getattr(
                    self.global_config,
                    "early_schema_bridge_materialization_allow_relation_relaxation",
                    True,
                ),
                early_schema_bridge_materialization_exact_join_key_min_schema_idf=getattr(
                    self.global_config,
                    "early_schema_bridge_materialization_exact_join_key_min_schema_idf",
                    0.65,
                ),
                early_schema_bridge_materialization_exact_join_key_min_relation_overlap=getattr(
                    self.global_config,
                    "early_schema_bridge_materialization_exact_join_key_min_relation_overlap",
                    0.05,
                ),
                early_schema_bridge_materialization_compact_projection=getattr(
                    self.global_config,
                    "early_schema_bridge_materialization_compact_projection",
                    True,
                ),
                early_schema_bridge_materialization_require_residual_deficit=getattr(
                    self.global_config,
                    "early_schema_bridge_materialization_require_residual_deficit",
                    True,
                ),
                early_schema_bridge_materialization_require_anchor_binding=getattr(
                    self.global_config,
                    "early_schema_bridge_materialization_require_anchor_binding",
                    True,
                ),
                early_schema_bridge_materialization_safe_max_original_rank=getattr(
                    self.global_config,
                    "early_schema_bridge_materialization_safe_max_original_rank",
                    48,
                ),
                early_schema_bridge_materialization_deep_rank_min_score=getattr(
                    self.global_config,
                    "early_schema_bridge_materialization_deep_rank_min_score",
                    0.70,
                ),
                validated_schema_tuple_materialization_enabled=getattr(
                    self.global_config,
                    "enable_validated_schema_tuple_materialization",
                    False,
                ),
                validated_schema_tuple_materialization_max_candidates=getattr(
                    self.global_config,
                    "validated_schema_tuple_materialization_max_candidates",
                    1,
                ),
                validated_schema_tuple_materialization_min_score=getattr(
                    self.global_config,
                    "validated_schema_tuple_materialization_min_score",
                    0.58,
                ),
                validated_schema_tuple_materialization_min_relation_overlap=getattr(
                    self.global_config,
                    "validated_schema_tuple_materialization_min_relation_overlap",
                    0.05,
                ),
                validated_schema_tuple_materialization_require_predicate_contract=getattr(
                    self.global_config,
                    "validated_schema_tuple_materialization_require_predicate_contract",
                    True,
                ),
                validated_schema_tuple_materialization_require_terminal_slot=getattr(
                    self.global_config,
                    "validated_schema_tuple_materialization_require_terminal_slot",
                    True,
                ),
                validated_schema_tuple_materialization_require_source_sentence=getattr(
                    self.global_config,
                    "validated_schema_tuple_materialization_require_source_sentence",
                    True,
                ),
                validated_schema_tuple_materialization_min_token_savings=getattr(
                    self.global_config,
                    "validated_schema_tuple_materialization_min_token_savings",
                    1,
                ),
                validated_schema_tuple_materialization_block_reader_visible_source=getattr(
                    self.global_config,
                    "validated_schema_tuple_materialization_block_reader_visible_source",
                    True,
                ),
                active_schema_bridge_tuple_executor_enabled=getattr(
                    self.global_config,
                    "enable_active_schema_bridge_tuple_executor",
                    True,
                ),
                active_schema_bridge_tuple_min_score=getattr(
                    self.global_config,
                    "active_schema_bridge_tuple_min_score",
                    0.52,
                ),
                active_schema_bridge_tuple_min_bound_signals=getattr(
                    self.global_config,
                    "active_schema_bridge_tuple_min_bound_signals",
                    2,
                ),
                entity_predicate_materialization_enabled=getattr(
                    self.global_config,
                    "enable_entity_predicate_fact_join_materialization",
                    False,
                ),
                entity_predicate_materialization_max_candidates=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_materialization_max_candidates",
                    1,
                ),
                entity_predicate_materialization_min_score=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_materialization_min_score",
                    0.64,
                ),
                entity_predicate_materialization_min_relation_overlap=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_materialization_min_relation_overlap",
                    0.50,
                ),
                entity_predicate_materialization_min_lineage_confidence=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_materialization_min_lineage_confidence",
                    0.34,
                ),
                entity_predicate_materialization_require_atomic_slot=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_require_atomic_slot",
                    True,
                ),
                entity_predicate_materialization_allowed_match_modes=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_materialization_allowed_match_modes",
                    "direct_token,polarity_concept",
                ),
                entity_predicate_materialization_compact_projection=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_materialization_compact_projection",
                    False,
                ),
                entity_predicate_terminal_tuple_materialization=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_terminal_tuple_materialization",
                    False,
                ),
                entity_predicate_terminal_tuple_min_lineage_depth=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_terminal_tuple_min_lineage_depth",
                    1,
                ),
                entity_predicate_terminal_tuple_strict_branch_binding=getattr(
                    self.global_config,
                    "entity_predicate_fact_join_terminal_tuple_strict_branch_binding",
                    True,
                ),
            )
            query_solution.evidence_admission = evidence_admission
            evidence_admission_counts[
                f"{evidence_admission.get('reason', 'unavailable')}|"
                f"{evidence_admission.get('accepted_count', 0)}"
            ] += 1

            schema_critical_info = {
                "enabled": bool(
                    getattr(self.global_config, "enable_schema_critical_reader_visible", False)
                ),
                "reason": "disabled",
                "accepted": False,
                "accepted_count": 0,
            }
            if schema_critical_info["enabled"]:
                schema_memory_info = dict(query_solution.query_schema_memory or {})
                schema_critical_info = select_schema_critical_evidence(
                    question=query_solution.question,
                    base_docs=stable_base_passages,
                    reader_visible_docs=retrieved_passages,
                    candidate_items=schema_memory_info.get("candidate_items") or [],
                    get_doc_content=lambda doc_idx: self.chunk_embedding_store.get_row(
                        self.passage_node_keys[int(doc_idx)]
                    )["content"],
                    qa_top_k=self.global_config.qa_top_k,
                    max_append=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_max_append",
                        1,
                    ),
                    deficit_coverage_threshold=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_deficit_coverage_threshold",
                        0.72,
                    ),
                    min_score=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_min_score",
                        0.58,
                    ),
                    min_schema_idf=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_min_schema_idf",
                        0.50,
                    ),
                    min_join_key_overlap=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_min_join_key_overlap",
                        0.75,
                    ),
                    min_relation_overlap=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_min_relation_overlap",
                        0.05,
                    ),
                    max_original_rank=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_max_original_rank",
                        80,
                    ),
                    token_budget=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_token_budget",
                        220,
                    ),
                    compute_budget=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_compute_budget",
                        1.0,
                    ),
                    min_objective=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_min_objective",
                        0.08,
                    ),
                    risk_weight=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_risk_weight",
                        0.42,
                    ),
                    token_weight=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_token_weight",
                        0.12,
                    ),
                    compute_weight=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_compute_weight",
                        0.04,
                    ),
                    require_bridge_cues=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_require_bridge_cues",
                        False,
                    ),
                    bridge_cues=has_bridge_cues(query_solution.question),
                    require_instance_bound_lineage=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_require_instance_bound_lineage",
                        True,
                    ),
                    materialize_distributed_join_view=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_materialize_distributed_join_view",
                        False,
                    ),
                    require_unique_terminal_value=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_require_unique_terminal_value",
                        True,
                    ),
                    require_source_novelty=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_require_source_novelty",
                        False,
                    ),
                    require_terminal_subtype_match=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_require_terminal_subtype_match",
                        False,
                    ),
                    allow_general_terminal_fact_view=getattr(
                        self.global_config,
                        "schema_critical_reader_visible_allow_general_terminal_fact_view",
                        False,
                    ),
                )
                apply_schema_critical_materialization = bool(
                    getattr(
                        self.global_config,
                        "schema_critical_reader_visible_apply_materialization",
                        True,
                    )
                )
                schema_critical_info["materialization_enabled"] = (
                    apply_schema_critical_materialization
                )
                schema_critical_info["would_accept"] = bool(
                    schema_critical_info.get("accepted")
                )
                schema_critical_info["would_accept_count"] = int(
                    schema_critical_info.get("accepted_count", 0) or 0
                )
                appended = 0
                seen_reader_docs = {str(doc).strip() for doc in retrieved_passages}
                for selected in schema_critical_info.get("selected") or []:
                    selected["reader_visible_appended"] = False
                    if not apply_schema_critical_materialization:
                        continue
                    reader_text = str(selected.get("reader_text") or "").strip()
                    if not reader_text or reader_text in seen_reader_docs:
                        continue
                    retrieved_passages.append(reader_text)
                    seen_reader_docs.add(reader_text)
                    selected["reader_visible_appended"] = True
                    appended += 1
                schema_critical_info["reader_visible_appended_count"] = appended
                if (
                    not apply_schema_critical_materialization
                    and schema_critical_info.get("would_accept")
                ):
                    schema_critical_info["accepted"] = False
                    schema_critical_info["reason"] = "selected_shadow_only"
                    schema_critical_info["accepted_count"] = 0
                elif appended == 0 and schema_critical_info.get("accepted"):
                    schema_critical_info["accepted"] = False
                    schema_critical_info["reason"] = "selected_projection_not_appended"
                    schema_critical_info["accepted_count"] = 0
                schema_critical_info["reader_visible"] = bool(appended)
            query_solution.schema_critical_evidence = schema_critical_info
            schema_critical_evidence_counts[
                f"{schema_critical_info.get('reason', 'unavailable')}|"
                f"{schema_critical_info.get('reader_visible_appended_count', 0)}"
            ] += 1

            retrieved_passages, entity_centric_fallback = append_entity_centric_deep_evidence(
                question=query_solution.question,
                current_docs=retrieved_passages,
                ranked_docs=query_solution.docs or [],
                planned_retrieval=planned_info,
                enabled=getattr(
                    self.global_config,
                    "enable_entity_centric_fallback_evidence",
                    False,
                ),
                candidate_top_k=getattr(
                    self.global_config,
                    "entity_centric_fallback_candidate_top_k",
                    80,
                ),
                max_append=getattr(
                    self.global_config,
                    "entity_centric_fallback_max_append",
                    1,
                ),
                min_score=getattr(
                    self.global_config,
                    "entity_centric_fallback_min_score",
                    0.44,
                ),
                min_title_anchor_overlap=getattr(
                    self.global_config,
                    "entity_centric_fallback_min_title_anchor_overlap",
                    0.58,
                ),
                min_subject_title_overlap=getattr(
                    self.global_config,
                    "entity_centric_fallback_min_subject_title_overlap",
                    0.85,
                ),
                min_subject_doc_overlap=getattr(
                    self.global_config,
                    "entity_centric_fallback_min_subject_doc_overlap",
                    0.90,
                ),
                min_sentence_score=getattr(
                    self.global_config,
                    "entity_centric_fallback_min_sentence_score",
                    0.20,
                ),
                min_target_signal=getattr(
                    self.global_config,
                    "entity_centric_fallback_min_target_signal",
                    0.20,
                ),
                insert_after=getattr(
                    self.global_config,
                    "entity_centric_fallback_insert_after",
                    self.global_config.qa_top_k,
                ),
                block_bridge=getattr(
                    self.global_config,
                    "entity_centric_fallback_block_bridge",
                    True,
                ),
                block_bridge_like_relation=getattr(
                    self.global_config,
                    "entity_centric_fallback_block_bridge_like_relation",
                    True,
                ),
                strict_subject_match=getattr(
                    self.global_config,
                    "entity_centric_fallback_strict_subject_match",
                    True,
                ),
                require_slot_signal=getattr(
                    self.global_config,
                    "entity_centric_fallback_require_slot_signal",
                    True,
                ),
                block_ambiguous_short_subject=getattr(
                    self.global_config,
                    "entity_centric_fallback_block_ambiguous_short_subject",
                    True,
                ),
                current_sufficiency_guard=getattr(
                    self.global_config,
                    "entity_centric_fallback_current_sufficiency_guard",
                    True,
                ),
                block_empty_factoid_slot=getattr(
                    self.global_config,
                    "entity_centric_fallback_block_empty_factoid_slot",
                    True,
                ),
                quarantine_generic_titles=getattr(
                    self.global_config,
                    "entity_centric_fallback_quarantine_generic_titles",
                    True,
                ),
                route_aware_relaxation=getattr(
                    self.global_config,
                    "entity_centric_fallback_route_aware_relaxation",
                    True,
                ),
                route_aware_max_rank=getattr(
                    self.global_config,
                    "entity_centric_fallback_route_aware_max_rank",
                    24,
                ),
                route_aware_min_score=getattr(
                    self.global_config,
                    "entity_centric_fallback_route_aware_min_score",
                    0.54,
                ),
                route_aware_min_sentence_score=getattr(
                    self.global_config,
                    "entity_centric_fallback_route_aware_min_sentence_score",
                    0.32,
                ),
                route_aware_min_question_overlap=getattr(
                    self.global_config,
                    "entity_centric_fallback_route_aware_min_question_overlap",
                    0.20,
                ),
            )
            query_solution.entity_centric_fallback = entity_centric_fallback
            entity_centric_fallback_counts[
                f"{entity_centric_fallback.get('reason', 'unavailable')}|"
                f"{entity_centric_fallback.get('accepted_count', 0)}"
            ] += 1

            router_info = planned_info.get("router", {}) if isinstance(planned_info, dict) else {}
            if not isinstance(router_info, dict):
                router_info = {}
            route_question_type = planned_info.get("question_type") or router_info.get("question_type")
            route_profile = planned_info.get("routing_profile") or router_info.get("profile")
            retrieved_passages, multigranular_memory = build_multigranular_evidence_memory(
                question=query_solution.question,
                docs=retrieved_passages,
                enabled=getattr(self.global_config, "enable_multigranular_evidence_memory", False),
                max_docs=getattr(self.global_config, "multigranular_memory_max_docs", 8),
                max_sentences_per_doc=getattr(
                    self.global_config,
                    "multigranular_memory_max_sentences_per_doc",
                    2,
                ),
                max_total_sentences=getattr(self.global_config, "multigranular_memory_max_total_sentences", 14),
                keep_full_top_docs=getattr(self.global_config, "multigranular_memory_keep_full_top_docs", 2),
                min_sentence_score=getattr(self.global_config, "multigranular_memory_min_sentence_score", 0.12),
                max_memory_chars=getattr(self.global_config, "multigranular_memory_max_chars", 3600),
                long_answer_only=getattr(self.global_config, "multigranular_memory_long_answer_only", True),
                min_avg_doc_chars=getattr(self.global_config, "multigranular_memory_min_avg_doc_chars", 1400),
                question_type=route_question_type or "",
                route_profile=route_profile or "",
                evidence_selection_reason=evidence_selection.get("reason", ""),
                controller_mode=getattr(self.global_config, "multigranular_memory_controller_mode", "selective"),
                controller_min_avg_doc_chars=getattr(
                    self.global_config,
                    "multigranular_memory_controller_min_avg_doc_chars",
                    500,
                ),
                skip_no_useful_extra_candidates=getattr(
                    self.global_config,
                    "multigranular_memory_skip_no_useful_extra_candidates",
                    True,
                ),
            )
            query_solution.multigranular_memory = multigranular_memory
            multigranular_memory_counts[multigranular_memory.get("reason", "unavailable")] += 1

            query_solution.qa_source_docs = retrieved_passages
            retrieved_passages, sentence_evidence = build_typed_sentence_evidence(
                question=query_solution.question,
                docs=retrieved_passages,
                enabled=getattr(self.global_config, "enable_sentence_evidence", False),
                max_sentences_per_doc=getattr(self.global_config, "sentence_evidence_max_sentences_per_doc", 2),
                max_total_sentences=getattr(self.global_config, "sentence_evidence_max_total_sentences", 16),
                keep_full_top_docs=getattr(
                    self.global_config,
                    "sentence_evidence_keep_full_top_docs",
                    self.global_config.qa_top_k,
                ),
                min_sentence_score=getattr(self.global_config, "sentence_evidence_min_score", 0.18),
                bridge_cues=has_bridge_cues(query_solution.question),
                chain_preserve=getattr(self.global_config, "sentence_evidence_chain_preserve", True),
                chain_preserve_doc_threshold=getattr(
                    self.global_config,
                    "sentence_evidence_chain_preserve_doc_threshold",
                    self.global_config.qa_top_k,
                ),
            )
            query_solution.sentence_evidence = sentence_evidence
            sentence_evidence_counts[sentence_evidence.get("reason", "unavailable")] += 1

            retrieved_passages, anchored_typed_candidate_table = build_anchored_typed_candidate_table(
                question=query_solution.question,
                docs=retrieved_passages,
                enabled=getattr(self.global_config, "enable_anchored_typed_candidate_table", False),
                max_docs=getattr(self.global_config, "anchored_typed_candidate_table_max_docs", 8),
                max_rows=getattr(self.global_config, "anchored_typed_candidate_table_max_rows", 5),
                min_table_score=getattr(
                    self.global_config,
                    "anchored_typed_candidate_table_min_score",
                    0.72,
                ),
                min_question_overlap=getattr(
                    self.global_config,
                    "anchored_typed_candidate_table_min_question_overlap",
                    0.08,
                ),
                min_provenance=getattr(
                    self.global_config,
                    "anchored_typed_candidate_table_min_provenance",
                    1,
                ),
                require_role_constraint=getattr(
                    self.global_config,
                    "anchored_typed_candidate_table_require_role_constraint",
                    True,
                ),
                insert_after=getattr(self.global_config, "anchored_typed_candidate_table_insert_after", 2),
                cost_model_enabled=getattr(self.global_config, "enable_evidence_cost_model", False),
                cost_model_token_budget=getattr(
                    self.global_config,
                    "evidence_cost_model_token_budget",
                    220,
                ),
                cost_model_min_utility=getattr(
                    self.global_config,
                    "evidence_cost_model_min_utility",
                    0.50,
                ),
                cost_model_max_risk=getattr(
                    self.global_config,
                    "evidence_cost_model_max_risk",
                    0.55,
                ),
                cost_model_min_net_utility=getattr(
                    self.global_config,
                    "evidence_cost_model_min_net_utility",
                    0.08,
                ),
                cost_model_token_weight=getattr(
                    self.global_config,
                    "evidence_cost_model_token_weight",
                    0.18,
                ),
                cost_model_compute_weight=getattr(
                    self.global_config,
                    "evidence_cost_model_compute_weight",
                    0.08,
                ),
                cost_model_risk_weight=getattr(
                    self.global_config,
                    "evidence_cost_model_risk_weight",
                    0.62,
                ),
            )
            query_solution.anchored_typed_candidate_table = anchored_typed_candidate_table
            anchored_typed_candidate_table_counts[
                anchored_typed_candidate_table.get("reason", "unavailable")
            ] += 1

            retrieved_passages, structured_evidence_chain = build_structured_evidence_chain(
                question=query_solution.question,
                docs=retrieved_passages,
                ranked_docs=query_solution.docs or [],
                planned_retrieval=planned_info,
                enabled=getattr(self.global_config, "enable_structured_evidence_chain", False),
                bridge_only=getattr(self.global_config, "structured_evidence_chain_bridge_only", True),
                candidate_top_k=getattr(self.global_config, "structured_evidence_chain_candidate_top_k", 24),
                max_sentences=getattr(self.global_config, "structured_evidence_chain_max_sentences", 4),
                max_candidate_sentences=getattr(
                    self.global_config,
                    "structured_evidence_chain_max_candidate_sentences",
                    1,
                ),
                min_sentence_score=getattr(
                    self.global_config,
                    "structured_evidence_chain_min_sentence_score",
                    0.27,
                ),
                min_chain_score=getattr(
                    self.global_config,
                    "structured_evidence_chain_min_chain_score",
                    0.52,
                ),
                insert_after=getattr(self.global_config, "structured_evidence_chain_insert_after", 2),
                skip_single_hop=getattr(self.global_config, "structured_evidence_chain_skip_single_hop", True),
                require_evidence_gap=getattr(
                    self.global_config,
                    "structured_evidence_chain_require_evidence_gap",
                    True,
                ),
                allow_long_context=getattr(
                    self.global_config,
                    "structured_evidence_chain_allow_long_context",
                    False,
                ),
                min_bridge_strength=getattr(
                    self.global_config,
                    "structured_evidence_chain_min_bridge_strength",
                    3,
                ),
                skip_if_top2_changed=getattr(
                    self.global_config,
                    "structured_evidence_chain_skip_if_top2_changed",
                    True,
                ),
            )
            query_solution.structured_evidence_chain = structured_evidence_chain
            structured_evidence_chain_counts[
                structured_evidence_chain.get("reason", "unavailable")
            ] += 1

            retrieved_passages, entity_state_evidence = build_entity_state_evidence_control(
                question=query_solution.question,
                docs=retrieved_passages,
                planned_retrieval=planned_info,
                evidence_selection=evidence_selection,
                enabled=getattr(self.global_config, "enable_entity_state_evidence", False),
                bridge_only=getattr(self.global_config, "entity_state_evidence_bridge_only", True),
                conflict_only=getattr(self.global_config, "entity_state_evidence_conflict_only", True),
                max_anchor_docs=getattr(self.global_config, "entity_state_evidence_max_anchor_docs", 5),
                max_anchors=getattr(self.global_config, "entity_state_evidence_max_anchors", 6),
                max_support=getattr(self.global_config, "entity_state_evidence_max_support", 4),
                max_distractors=getattr(self.global_config, "entity_state_evidence_max_distractors", 3),
                max_sentences_per_doc=getattr(
                    self.global_config,
                    "entity_state_evidence_max_sentences_per_doc",
                    2,
                ),
                min_source_overlap=getattr(
                    self.global_config,
                    "entity_state_evidence_min_source_overlap",
                    0.08,
                ),
            )
            query_solution.entity_state_evidence = entity_state_evidence
            entity_state_evidence_counts[entity_state_evidence.get("reason", "unavailable")] += 1

            retrieved_passages, slot_projected_provenance = build_slot_projected_provenance_view(
                question=query_solution.question,
                docs=retrieved_passages,
                planned_retrieval=query_solution.planned_retrieval,
                evidence_admission=query_solution.evidence_admission,
                enabled=getattr(self.global_config, "enable_slot_projected_provenance_view", False),
                max_docs=getattr(self.global_config, "slot_projected_provenance_max_docs", 8),
                max_rows=getattr(self.global_config, "slot_projected_provenance_max_rows", 3),
                min_confidence=getattr(self.global_config, "slot_projected_provenance_min_confidence", 0.70),
                min_margin=getattr(self.global_config, "slot_projected_provenance_min_margin", 0.06),
                min_question_overlap=getattr(
                    self.global_config,
                    "slot_projected_provenance_min_question_overlap",
                    0.08,
                ),
                min_provenance=getattr(self.global_config, "slot_projected_provenance_min_provenance", 1),
                insert_after=getattr(self.global_config, "slot_projected_provenance_insert_after", 2),
                bridge_or_deficit_only=getattr(
                    self.global_config,
                    "slot_projected_provenance_bridge_or_deficit_only",
                    True,
                ),
                block_simple_factoid_without_deficit=getattr(
                    self.global_config,
                    "slot_projected_provenance_block_simple_factoid_without_deficit",
                    True,
                ),
                block_long_context=getattr(
                    self.global_config,
                    "slot_projected_provenance_block_long_context",
                    True,
                ),
                long_context_avg_chars=getattr(
                    self.global_config,
                    "slot_projected_provenance_long_context_avg_chars",
                    1400,
                ),
                require_role_constraint=getattr(
                    self.global_config,
                    "slot_projected_provenance_require_role_constraint",
                    True,
                ),
            )
            query_solution.slot_projected_provenance = slot_projected_provenance
            slot_projected_provenance_counts[
                slot_projected_provenance.get("reason", "unavailable")
            ] += 1

            retrieved_passages, candidate_answer_selector = build_candidate_answer_contrastive_selector(
                question=query_solution.question,
                docs=retrieved_passages,
                enabled=getattr(self.global_config, "enable_candidate_answer_selector", False),
                numeric_temporal_only=getattr(
                    self.global_config,
                    "candidate_answer_selector_numeric_temporal_only",
                    True,
                ),
                min_confidence=getattr(self.global_config, "candidate_answer_selector_min_confidence", 0.76),
                min_margin=getattr(self.global_config, "candidate_answer_selector_min_margin", 0.08),
                min_question_overlap=getattr(
                    self.global_config,
                    "candidate_answer_selector_min_question_overlap",
                    0.08,
                ),
                max_docs=getattr(self.global_config, "candidate_answer_selector_max_docs", 8),
                max_candidates=getattr(self.global_config, "candidate_answer_selector_max_candidates", 3),
                insert_after=getattr(self.global_config, "candidate_answer_selector_insert_after", 2),
            )
            query_solution.candidate_answer_selector = candidate_answer_selector
            candidate_answer_selector_counts[candidate_answer_selector.get("reason", "unavailable")] += 1

            retrieved_passages, slot_separated_fact_join = build_slot_separated_fact_join_view(
                question=query_solution.question,
                docs=retrieved_passages,
                planned_retrieval=query_solution.planned_retrieval,
                typed_fact_join_shadow=query_solution.typed_fact_join_shadow,
                enabled=getattr(self.global_config, "enable_slot_separated_fact_join_view", False),
                max_rows=getattr(self.global_config, "slot_separated_fact_join_max_rows", 3),
                min_utility=getattr(self.global_config, "slot_separated_fact_join_min_utility", 0.54),
                max_risk=getattr(self.global_config, "slot_separated_fact_join_max_risk", 0.50),
                min_net_utility=getattr(
                    self.global_config,
                    "slot_separated_fact_join_min_net_utility",
                    0.12,
                ),
                insert_after=getattr(self.global_config, "slot_separated_fact_join_insert_after", 2),
                require_answer_slot=getattr(
                    self.global_config,
                    "slot_separated_fact_join_require_answer_slot",
                    True,
                ),
                bridge_or_deficit_only=getattr(
                    self.global_config,
                    "slot_separated_fact_join_bridge_or_deficit_only",
                    True,
                ),
                block_long_context=getattr(
                    self.global_config,
                    "slot_separated_fact_join_block_long_context",
                    True,
                ),
                long_context_avg_chars=getattr(
                    self.global_config,
                    "slot_separated_fact_join_long_context_avg_chars",
                    1400,
                ),
                allow_one_hop_answer_rows=getattr(
                    self.global_config,
                    "slot_separated_fact_join_allow_one_hop_answer_rows",
                    False,
                ),
                require_predicate_match=getattr(
                    self.global_config,
                    "slot_separated_fact_join_require_predicate_match",
                    True,
                ),
                block_boolean_comparison=getattr(
                    self.global_config,
                    "slot_separated_fact_join_block_boolean_comparison",
                    True,
                ),
                min_slot_score=getattr(
                    self.global_config,
                    "slot_separated_fact_join_min_slot_score",
                    0.50,
                ),
            )
            query_solution.slot_separated_fact_join = slot_separated_fact_join

            retrieved_passages, answer_slot_support_completion = append_answer_slot_support_completion(
                question=query_solution.question,
                current_docs=retrieved_passages,
                ranked_docs=query_solution.docs or [],
                planned_retrieval=query_solution.planned_retrieval,
                enabled=getattr(
                    self.global_config,
                    "enable_answer_slot_support_completion",
                    False,
                ),
                bridge_or_attribute_only=getattr(
                    self.global_config,
                    "answer_slot_support_completion_bridge_or_attribute_only",
                    True,
                ),
                candidate_top_k=getattr(
                    self.global_config,
                    "answer_slot_support_completion_candidate_top_k",
                    40,
                ),
                max_rows=getattr(
                    self.global_config,
                    "answer_slot_support_completion_max_rows",
                    4,
                ),
                min_row_score=getattr(
                    self.global_config,
                    "answer_slot_support_completion_min_row_score",
                    0.40,
                ),
                min_slot_signal=getattr(
                    self.global_config,
                    "answer_slot_support_completion_min_slot_signal",
                    0.30,
                ),
                min_anchor_or_chain=getattr(
                    self.global_config,
                    "answer_slot_support_completion_min_anchor_or_chain",
                    0.08,
                ),
                min_missing_gain=getattr(
                    self.global_config,
                    "answer_slot_support_completion_min_missing_gain",
                    0.00,
                ),
                insert_after=getattr(
                    self.global_config,
                    "answer_slot_support_completion_insert_after",
                    2,
                ),
                include_current_rows=getattr(
                    self.global_config,
                    "answer_slot_support_completion_include_current_rows",
                    True,
                ),
                lineage_safe=getattr(
                    self.global_config,
                    "answer_slot_support_completion_lineage_safe",
                    True,
                ),
                allow_bridge=getattr(
                    self.global_config,
                    "answer_slot_support_completion_allow_bridge",
                    False,
                ),
                min_relation_overlap=getattr(
                    self.global_config,
                    "answer_slot_support_completion_min_relation_overlap",
                    0.08,
                ),
                min_subject_overlap=getattr(
                    self.global_config,
                    "answer_slot_support_completion_min_subject_overlap",
                    0.55,
                ),
                conflict_margin=getattr(
                    self.global_config,
                    "answer_slot_support_completion_conflict_margin",
                    0.05,
                ),
                block_boolean_comparison=getattr(
                    self.global_config,
                    "answer_slot_support_completion_block_boolean_comparison",
                    True,
                ),
                block_list_like_sources=getattr(
                    self.global_config,
                    "answer_slot_support_completion_block_list_like_sources",
                    True,
                ),
            )
            query_solution.answer_slot_support_completion = answer_slot_support_completion
            answer_slot_support_completion_counts[
                f"{answer_slot_support_completion.get('reason', 'unavailable')}|"
                f"{answer_slot_support_completion.get('accepted_count', 0)}"
            ] += 1

            planned_info = annotate_planned_retrieval_terminal_state(
                query_solution.planned_retrieval or planned_info
            )
            planned_info["terminal_materialization"] = (
                build_terminal_materialization_trace(
                    planned_info,
                    source_docs=query_solution.qa_source_docs or [],
                    reader_docs=retrieved_passages,
                    components={
                        "relation_fact_join": query_solution.relation_fact_join,
                        "anchored_typed_candidate_table": (
                            anchored_typed_candidate_table
                        ),
                        "structured_evidence_chain": structured_evidence_chain,
                        "slot_projected_provenance": slot_projected_provenance,
                        "slot_separated_fact_join": slot_separated_fact_join,
                        "answer_slot_support_completion": (
                            answer_slot_support_completion
                        ),
                    },
                )
            )
            query_solution.planned_retrieval = planned_info

            query_solution.qa_docs = retrieved_passages
            slot_separated_fact_join_counts[
                slot_separated_fact_join.get("reason", "unavailable")
            ] += 1

            preserve_explicit_date_precision = bool(getattr(
                self.global_config,
                "paper_evidence_reader_preserve_explicit_date_precision",
                False,
            ))
            if (
                getattr(self.global_config, "enable_answer_style_routing", True)
                or preserve_explicit_date_precision
            ):
                answer_style, answer_style_instruction = get_answer_style_instruction(
                    query_solution.question,
                    preserve_explicit_date_precision=preserve_explicit_date_precision,
                )
            else:
                answer_style, answer_style_instruction = "short_factoid", ""
            query_solution.answer_style = answer_style
            query_solution.answer_style_instruction = answer_style_instruction
            answer_style_counts[answer_style] += 1
            all_qa_messages.append(
                _render_reader_prompt(query_solution, retrieved_passages)
            )
            qa_timing = dict(query_solution.qa_timing or {})
            qa_timing["evidence_preparation_seconds"] = round(
                time.perf_counter() - evidence_preparation_started,
                6,
            )
            query_solution.qa_timing = qa_timing

            replay_enabled = bool(getattr(
                self.global_config,
                "enable_entity_predicate_final_reader_counterfactual",
                False,
            ))
            replay_limit = max(0, int(getattr(
                self.global_config,
                "entity_predicate_final_reader_counterfactual_max_queries",
                64,
            )))
            accepted_join_count = int(
                (evidence_admission or {}).get(
                    "accepted_entity_predicate_fact_join_count",
                    0,
                )
                or 0
            )
            if (
                replay_enabled
                and accepted_join_count > 0
                and len(final_reader_counterfactual_specs) < replay_limit
            ):
                accepted_titles = list(
                    (evidence_admission or {}).get(
                        "accepted_entity_predicate_titles",
                        [],
                    )
                    or []
                )
                accepted_fingerprints = list(
                    (evidence_admission or {}).get(
                        "accepted_entity_predicate_doc_fingerprints",
                        [],
                    )
                    or []
                )
                counterfactual_docs, removed_titles = (
                    remove_entity_predicate_passages(
                        retrieved_passages,
                        accepted_titles,
                        accepted_fingerprints,
                    )
                )
                replacement_pairs = list(
                    ((evidence_admission or {}).get("deficit_aware_replacement") or {}).get(
                        "pairs",
                        [],
                    )
                    or []
                )
                restored_docs = []
                restored_titles = []
                for pair in replacement_pairs:
                    restored_doc = str(pair.get("evicted_doc") or "").strip()
                    if restored_doc and restored_doc not in counterfactual_docs:
                        restored_docs.append(restored_doc)
                        restored_titles.append(
                            str(pair.get("evicted_title") or restored_doc.split("\n", 1)[0])
                        )
                counterfactual_docs.extend(restored_docs)
                if removed_titles and counterfactual_docs:
                    counterfactual_query = deepcopy(query_solution)
                    counterfactual_query.qa_docs = counterfactual_docs
                    counterfactual_source_docs, _ = remove_entity_predicate_passages(
                        counterfactual_query.qa_source_docs or [],
                        accepted_titles,
                        accepted_fingerprints,
                    )
                    counterfactual_query.qa_source_docs = counterfactual_source_docs
                    for restored_doc in restored_docs:
                        if restored_doc not in counterfactual_query.qa_source_docs:
                            counterfactual_query.qa_source_docs.append(restored_doc)
                    counterfactual_admission = dict(
                        counterfactual_query.evidence_admission or {}
                    )
                    counterfactual_admission.update({
                        "reason": (
                            "final_reader_counterfactual_budget_swap_reversed"
                            if restored_docs
                            else "final_reader_counterfactual_entity_predicate_removed"
                        ),
                        "accepted_entity_predicate_fact_join_count": 0,
                        "accepted_entity_predicate_titles": [],
                        "accepted_entity_predicate_doc_fingerprints": [],
                        "final_reader_counterfactual": True,
                    })
                    counterfactual_query.evidence_admission = counterfactual_admission
                    final_reader_counterfactual_specs.append({
                        "operator": "entity_predicate_fact_join",
                        "original_role": "treatment",
                        "original_index": len(all_qa_messages) - 1,
                        "query": counterfactual_query,
                        "removed_titles": removed_titles,
                        "restored_titles": restored_titles,
                        "full_doc_count": len(retrieved_passages),
                        "counterfactual_doc_count": len(counterfactual_docs),
                    })

            relation_replay_enabled = bool(getattr(
                self.global_config,
                "enable_relation_fact_join_final_reader_counterfactual",
                False,
            ))
            relation_replay_limit = max(0, int(getattr(
                self.global_config,
                "relation_fact_join_final_reader_counterfactual_max_queries",
                64,
            )))
            relation_intervention = dict(
                (query_solution.relation_fact_join or {}).get(
                    "counterfactual_reader_intervention"
                ) or {}
            )
            frozen_join_doc = str(
                relation_intervention.get("candidate_table_doc") or ""
            ).strip()
            already_has_pair = any(
                int(item.get("original_index", -1)) == len(all_qa_messages) - 1
                for item in final_reader_counterfactual_specs
            )
            relation_pair_count = sum(
                item.get("operator") == "relation_fact_join"
                for item in final_reader_counterfactual_specs
            )
            if (
                relation_replay_enabled
                and relation_intervention.get("eligible")
                and frozen_join_doc
                and not already_has_pair
                and relation_pair_count < relation_replay_limit
            ):
                relation_table_docs = [
                    doc for doc in retrieved_passages
                    if str(doc or "").strip().startswith(
                        "Directional Fact Join Candidate Table"
                    )
                ]
                paired_query = deepcopy(query_solution)
                if relation_table_docs:
                    paired_docs = [
                        doc for doc in retrieved_passages
                        if doc not in relation_table_docs
                    ]
                    original_role = "treatment"
                    intervention = "remove_frozen_fact_join_view"
                else:
                    paired_docs = list(retrieved_passages) + [frozen_join_doc]
                    original_role = "control"
                    intervention = "add_frozen_fact_join_view"
                if paired_docs and paired_docs != list(retrieved_passages):
                    paired_query.qa_docs = paired_docs
                    paired_query.qa_source_docs = list(paired_docs)
                    final_reader_counterfactual_specs.append({
                        "operator": "relation_fact_join",
                        "original_role": original_role,
                        "intervention": intervention,
                        "original_index": len(all_qa_messages) - 1,
                        "query": paired_query,
                        "removed_titles": (
                            ["Directional Fact Join Candidate Table"]
                            if relation_table_docs else []
                        ),
                        "restored_titles": [],
                        "full_doc_count": len(retrieved_passages),
                        "counterfactual_doc_count": len(paired_docs),
                        "intervention_metadata": relation_intervention,
                    })

        original_query_count = len(queries)
        processing_queries = list(queries)
        for spec in final_reader_counterfactual_specs:
            spec["processing_index"] = len(processing_queries)
            counterfactual_query = spec["query"]
            processing_queries.append(counterfactual_query)
            all_qa_messages.append(
                _render_reader_prompt(
                    counterfactual_query,
                    counterfactual_query.qa_docs or [],
                )
            )
        logger.info(
            "QA final-reader entity-predicate counterfactual pairs: %s",
            len(final_reader_counterfactual_specs),
        )

        logger.info(f"QA answer style routing counts: {dict(answer_style_counts)}")
        logger.info(f"QA evidence selection counts: {dict(evidence_selection_counts)}")
        logger.info(f"QA bridge contrastive evidence counts: {dict(bridge_contrastive_evidence_counts)}")
        logger.info(f"QA dual-channel projection counts: {dict(dual_channel_projection_counts)}")
        logger.info(f"QA relation fact-join counts: {dict(relation_fact_join_counts)}")
        logger.info(f"QA selective support-join counts: {dict(selective_support_join_counts)}")
        logger.info(f"QA path evidence counts: {dict(path_evidence_counts)}")
        logger.info(f"QA multi-granular memory counts: {dict(multigranular_memory_counts)}")
        logger.info(f"QA sentence evidence counts: {dict(sentence_evidence_counts)}")
        logger.info(f"QA anchored typed candidate table counts: {dict(anchored_typed_candidate_table_counts)}")
        logger.info(f"QA structured evidence chain counts: {dict(structured_evidence_chain_counts)}")
        logger.info(f"QA entity-state evidence counts: {dict(entity_state_evidence_counts)}")
        logger.info(f"QA slot-projected provenance counts: {dict(slot_projected_provenance_counts)}")
        logger.info(f"QA slot-separated fact-join counts: {dict(slot_separated_fact_join_counts)}")
        logger.info(f"QA entity-centric fallback counts: {dict(entity_centric_fallback_counts)}")
        logger.info(f"QA answer-slot support completion counts: {dict(answer_slot_support_completion_counts)}")
        logger.info(f"QA support-set completion counts: {dict(support_set_completion_counts)}")
        logger.info(f"QA cost-based evidence admission counts: {dict(evidence_admission_counts)}")
        logger.info(f"QA Schema-Critical reader-visible counts: {dict(schema_critical_evidence_counts)}")
        logger.info(f"QA candidate answer selector counts: {dict(candidate_answer_selector_counts)}")

        if bool(getattr(
            self.global_config, "qa_cache_coverage_preflight", False
        )):
            inspect_cache_coverage = getattr(
                self.llm_model, "inspect_cache_coverage", None
            )
            if not callable(inspect_cache_coverage):
                raise RuntimeError(
                    "Reader cache coverage preflight requested, but the "
                    "configured LLM does not expose inspect_cache_coverage()."
                )
            cache_coverage = dict(
                inspect_cache_coverage(all_qa_messages) or {}
            )
            for record in cache_coverage.get("records", []):
                query_idx = int(record.get("index", -1))
                if query_idx < 0 or query_idx >= len(processing_queries):
                    continue
                query = processing_queries[query_idx]
                record.update({
                    "question": str(query.question or ""),
                    "qa_evidence_signature": str(
                        getattr(query, "qa_evidence_signature", "") or ""
                    ),
                    "qa_doc_titles": list(
                        getattr(query, "qa_doc_titles", []) or []
                    ),
                    "counterfactual_reader_prompt": bool(
                        query_idx >= original_query_count
                    ),
                })
            report_path = str(getattr(
                self.global_config,
                "qa_cache_coverage_report_path",
                "",
            ) or "").strip()
            if report_path:
                report_path = os.path.abspath(report_path)
                report_parent = os.path.dirname(report_path)
                if report_parent:
                    os.makedirs(report_parent, exist_ok=True)
                trace_path = write_pre_reader_trace(
                    pre_reader_trace_path(report_path),
                    processing_queries,
                    cache_coverage.get("records", []),
                    original_query_count=original_query_count,
                    reader_messages=all_qa_messages,
                )
                cache_coverage["pre_reader_trace_path"] = trace_path
                with open(report_path, "w", encoding="utf-8") as handle:
                    json.dump(
                        cache_coverage,
                        handle,
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )
            logger.info(
                "Reader cache coverage preflight: total=%s present=%s "
                "missing=%s report=%s trace=%s",
                cache_coverage.get("total", 0),
                cache_coverage.get("present", 0),
                cache_coverage.get("missing", 0),
                report_path or "<disabled>",
                cache_coverage.get("pre_reader_trace_path", "<disabled>"),
            )
            if not bool(cache_coverage.get("complete")):
                missing_records = list(
                    cache_coverage.get("records", [])
                )
                missing_records = [
                    record for record in missing_records
                    if not bool(record.get("present"))
                ]
                missing_preview = ", ".join(
                    f"query_idx={record.get('index')} key={record.get('key')}"
                    for record in missing_records[:8]
                )
                raise RuntimeError(
                    "Reader cache coverage preflight failed before inference: "
                    f"missing={len(missing_records)}/"
                    f"{cache_coverage.get('total', 0)} "
                    f"cache={cache_coverage.get('cache_file')} "
                    f"report={report_path or '<disabled>'}; "
                    f"{missing_preview}"
                )

        def _safe_qa_infer(indexed_qa_messages):
            query_idx, qa_messages = indexed_qa_messages
            reader_started = time.perf_counter()
            try:
                response, metadata, cache_hit = self.llm_model.infer(qa_messages)
                metadata = dict(metadata or {})
                metadata["qa_reader_seconds"] = round(
                    time.perf_counter() - reader_started,
                    6,
                )
                metadata["qa_reader_cache_hit"] = bool(cache_hit)
                return response, metadata, cache_hit
            except Exception as exc:
                if not _is_transient_api_failure(exc):
                    raise
                error_message = str(exc)
                root_error = _root_error(exc)
                root_error_name = root_error.__class__.__name__
                root_error_message = str(root_error)
                logger.warning(
                    "Transient QA inference failure for query_idx=%s with %s; queued for recovery. "
                    "Root error: %s: %s; Error: %s",
                    query_idx,
                    exc.__class__.__name__,
                    root_error_name,
                    root_error_message[:500],
                    error_message[:500],
                )
                return (
                    "Thought: QA inference failed after retries.\nAnswer: Unknown",
                    {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "finish_reason": "qa_inference_error",
                        "qa_inference_error": exc.__class__.__name__,
                        "qa_inference_error_message": error_message[:500],
                        "qa_inference_root_error": root_error_name,
                        "qa_inference_root_error_message": root_error_message[:500],
                        "qa_inference_recovery_status": "pending",
                        "qa_reader_seconds": round(
                            time.perf_counter() - reader_started,
                            6,
                        ),
                    },
                    False,
                )

        def _qa_result_failed(result) -> bool:
            if not isinstance(result, (list, tuple)) or len(result) < 2:
                return True
            metadata = result[1] if isinstance(result[1], dict) else {}
            return str(metadata.get("finish_reason") or "") == "qa_inference_error"

        qa_inference_workers = max(1, int(getattr(self.global_config, "qa_inference_workers", 1)))
        if qa_inference_workers > 1 and len(all_qa_messages) > 1:
            logger.info(f"QA Reading with {qa_inference_workers} worker threads.")
            with ThreadPoolExecutor(max_workers=qa_inference_workers) as executor:
                all_qa_results = list(
                    tqdm(
                        executor.map(_safe_qa_infer, enumerate(all_qa_messages)),
                        total=len(all_qa_messages),
                        desc=f"QA Reading x{qa_inference_workers}",
                    )
                )
        else:
            all_qa_results = [
                _safe_qa_infer((query_idx, qa_messages))
                for query_idx, qa_messages in tqdm(enumerate(all_qa_messages), total=len(all_qa_messages), desc="QA Reading")
            ]

        initial_failed_indices = [
            query_idx
            for query_idx, result in enumerate(all_qa_results)
            if _qa_result_failed(result)
        ]
        remaining_failed_indices = list(initial_failed_indices)
        recovery_rounds = max(
            0,
            int(getattr(self.global_config, "qa_inference_recovery_rounds", 2)),
        )
        recovery_workers = max(
            1,
            int(getattr(self.global_config, "qa_inference_recovery_workers", 1)),
        )
        recovery_delay_seconds = max(
            0.0,
            float(getattr(self.global_config, "qa_inference_recovery_delay_seconds", 10.0)),
        )
        for recovery_round in range(1, recovery_rounds + 1):
            if not remaining_failed_indices:
                break
            if recovery_delay_seconds > 0:
                time.sleep(recovery_delay_seconds)
            logger.warning(
                "QA inference recovery round %s/%s: retrying %s queries with %s worker(s).",
                recovery_round,
                recovery_rounds,
                len(remaining_failed_indices),
                recovery_workers,
            )
            reset_connection_pool = getattr(self.llm_model, "reset_connection_pool", None)
            if callable(reset_connection_pool):
                try:
                    reset_connection_pool()
                except Exception as exc:
                    logger.warning(
                        "Could not recreate the LLM connection pool before recovery round %s: %r",
                        recovery_round,
                        exc,
                    )
            recovery_inputs = [
                (query_idx, all_qa_messages[query_idx])
                for query_idx in remaining_failed_indices
            ]
            if recovery_workers > 1 and len(recovery_inputs) > 1:
                with ThreadPoolExecutor(max_workers=recovery_workers) as executor:
                    recovery_results = list(executor.map(_safe_qa_infer, recovery_inputs))
            else:
                recovery_results = [_safe_qa_infer(item) for item in recovery_inputs]

            next_failed_indices = []
            for query_idx, result in zip(remaining_failed_indices, recovery_results):
                if _qa_result_failed(result):
                    next_failed_indices.append(query_idx)
                    continue
                response, metadata, cache_hit = result
                metadata = dict(metadata or {})
                metadata.update({
                    "qa_inference_recovered": True,
                    "qa_inference_recovery_round": recovery_round,
                })
                all_qa_results[query_idx] = (response, metadata, cache_hit)
            remaining_failed_indices = next_failed_indices

        recovered_count = len(initial_failed_indices) - len(remaining_failed_indices)
        logger.info(
            "QA inference recovery summary: initial_failed=%s recovered=%s unrecovered=%s rounds=%s.",
            len(initial_failed_indices),
            recovered_count,
            len(remaining_failed_indices),
            recovery_rounds,
        )
        if remaining_failed_indices:
            for query_idx in remaining_failed_indices:
                logger.error(
                    "QA inference failed for query_idx=%s after %s recovery round(s); fallback answer remains.",
                    query_idx,
                    recovery_rounds,
                )
                response, metadata, cache_hit = all_qa_results[query_idx]
                metadata = dict(metadata or {})
                metadata["qa_inference_recovery_status"] = "unrecovered"
                metadata["qa_inference_recovery_rounds"] = recovery_rounds
                all_qa_results[query_idx] = (response, metadata, cache_hit)
            if bool(getattr(self.global_config, "qa_fail_on_unrecovered", True)):
                preview = ",".join(str(index) for index in remaining_failed_indices[:20])
                raise RuntimeError(
                    "QA inference left "
                    f"{len(remaining_failed_indices)} unrecovered query failure(s) after "
                    f"{recovery_rounds} recovery round(s); indices={preview}"
                )

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)


        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(
            enumerate(processing_queries),
            total=len(processing_queries),
            desc="Extraction Answers from LLM Response",
        ):
            answer_materialization_started = time.perf_counter()
            response_content = all_response_message[query_solution_idx]
            pred_ans, has_answer_marker = extract_answer_from_response(response_content)
            query_solution.has_answer_marker = has_answer_marker
            if not has_answer_marker:
                logger.warning("Error in parsing the answer from the raw LLM QA inference response: missing Answer marker!")
            paper_trace = dict(query_solution.paper_evidence_execution or {})
            if paper_trace:
                metadata = dict(all_metadata[query_solution_idx] or {})
                fixed_stages = list(paper_trace.get("fixed_stages") or [])
                fixed_stages.append({
                    "stage_id": "fixed2",
                    "stage": "READER",
                    "model_calls": 0 if all_cache_hit[query_solution_idx] else 1,
                    "cache_hit": bool(all_cache_hit[query_solution_idx]),
                    "input_tokens": metadata.get("prompt_tokens"),
                    "output_tokens": metadata.get("completion_tokens"),
                })
                paper_trace["fixed_stages"] = fixed_stages
                reader_answer = pred_ans
                replacement_enabled = bool(getattr(
                    self.global_config,
                    "paper_evidence_apply_structured_answer",
                    True,
                ))
                paper_trace["reader_raw_answer"] = reader_answer
                paper_trace["structured_candidate"] = str(
                    paper_trace.get("structured_answer") or ""
                )
                paper_trace["replacement_enabled"] = replacement_enabled
                if replacement_enabled:
                    pred_ans, answer_application = select_paper_answer(
                        question=query_solution.question,
                        reader_answer=reader_answer,
                        trace=paper_trace,
                        incumbent_guard=bool(getattr(
                            self.global_config,
                            "paper_evidence_incumbent_guard",
                            True,
                        )),
                        v8_safe_materialization=bool(getattr(
                            self.global_config,
                            "paper_evidence_v8_safe_materialization",
                            True,
                        )),
                    )
                    paper_trace["answer_application"] = answer_application
                    paper_trace = update_answer_phase_status(
                        paper_trace,
                        structured_answer_replaced_reader=bool(
                            answer_application.get("replaced")
                        ),
                    )
                else:
                    paper_trace["answer_application"] = {
                        "path": "structured_answer_application_disabled",
                        "reader_answer_before_admission": reader_answer,
                        "candidate_structured_answer": str(
                            paper_trace.get("structured_answer") or ""
                        ),
                        "answer_after_admission": reader_answer,
                        "reason": "configuration_disabled",
                        "replaced": False,
                    }
                    paper_trace = update_answer_phase_status(
                        paper_trace,
                        structured_answer_replaced_reader=False,
                    )
                query_solution.paper_evidence_execution = paper_trace
            if getattr(self.global_config, "enable_answer_postprocessing", True):
                pred_ans = postprocess_answer(query_solution.question, pred_ans)
            else:
                pred_ans = pred_ans.strip() if pred_ans is not None else ""
            query_solution.answer_before_canonicalization = pred_ans
            pred_ans, canonicalization = canonicalize_answer(
                question=query_solution.question,
                answer=pred_ans,
                qa_docs=query_solution.qa_docs,
                enabled=getattr(self.global_config, "enable_answer_canonicalization", False),
                min_confidence=getattr(self.global_config, "answer_canonicalization_min_confidence", 0.65),
            )
            query_solution.canonicalization = canonicalization
            query_solution.answer_before_verifier = pred_ans
            pred_ans, answer_verifier = verify_answer_with_evidence(
                question=query_solution.question,
                answer=pred_ans,
                qa_docs=query_solution.qa_docs,
                answer_style=query_solution.answer_style,
                enabled=getattr(self.global_config, "enable_evidence_answer_verifier", False),
                min_confidence=getattr(self.global_config, "evidence_answer_verifier_min_confidence", 0.72),
                min_margin=getattr(self.global_config, "evidence_answer_verifier_min_margin", 0.12),
                generic_min_confidence=getattr(
                    self.global_config,
                    "evidence_answer_verifier_generic_min_confidence",
                    0.84,
                ),
                max_docs=getattr(self.global_config, "evidence_answer_verifier_max_docs", 8),
            )
            query_solution.answer_verifier = answer_verifier
            query_solution.answer_before_consistent_reader = pred_ans
            pred_ans, evidence_consistent_reader = refine_answer_with_evidence_consistent_reader(
                question=query_solution.question,
                answer=pred_ans,
                qa_docs=query_solution.qa_docs,
                answer_style=query_solution.answer_style,
                enabled=getattr(self.global_config, "enable_evidence_consistent_reader", False),
                min_confidence=getattr(self.global_config, "evidence_consistent_reader_min_confidence", 0.84),
                min_margin=getattr(self.global_config, "evidence_consistent_reader_min_margin", 0.10),
                generic_min_confidence=getattr(
                    self.global_config,
                    "evidence_consistent_reader_generic_min_confidence",
                    0.92,
                ),
                min_support_gain=getattr(self.global_config, "evidence_consistent_reader_min_support_gain", 0.18),
                max_docs=getattr(self.global_config, "evidence_consistent_reader_max_docs", 8),
                max_answer_tokens=getattr(self.global_config, "evidence_consistent_reader_max_answer_tokens", 10),
            )
            query_solution.evidence_consistent_reader = evidence_consistent_reader
            query_solution.answer_before_object_arbitration = pred_ans
            pred_ans, evidence_object_arbitration = arbitrate_answer_with_evidence_objects(
                question=query_solution.question,
                answer=pred_ans,
                qa_docs=query_solution.qa_docs,
                answer_style=query_solution.answer_style,
                enabled=getattr(self.global_config, "enable_evidence_object_arbitration", False),
                min_confidence=getattr(self.global_config, "evidence_object_arbitration_min_confidence", 0.82),
                generic_min_confidence=getattr(
                    self.global_config,
                    "evidence_object_arbitration_generic_min_confidence",
                    0.88,
                ),
                min_margin=getattr(self.global_config, "evidence_object_arbitration_min_margin", 0.12),
                min_support_gain=getattr(
                    self.global_config,
                    "evidence_object_arbitration_min_support_gain",
                    0.18,
                ),
                min_question_overlap=getattr(
                    self.global_config,
                    "evidence_object_arbitration_min_question_overlap",
                    0.08,
                ),
                max_docs=getattr(self.global_config, "evidence_object_arbitration_max_docs", 10),
                max_answer_tokens=getattr(
                    self.global_config,
                    "evidence_object_arbitration_max_answer_tokens",
                    12,
                ),
                apply_rewrite=getattr(
                    self.global_config,
                    "evidence_object_arbitration_apply_rewrite",
                    False,
                ),
                require_role_constraint=getattr(
                    self.global_config,
                    "evidence_object_arbitration_require_role_constraint",
                    True,
                ),
            )
            query_solution.evidence_object_arbitration = evidence_object_arbitration
            evidence_object_arbitration_counts[
                evidence_object_arbitration.get("reason", "unavailable")
            ] += 1

            query_solution.answer_before_evidence_answer_arbiter = pred_ans
            pred_ans, evidence_answer_arbiter = arbitrate_answer_with_evidence_grounding(
                question=query_solution.question,
                answer=pred_ans,
                qa_docs=query_solution.qa_docs,
                answer_style=query_solution.answer_style,
                enabled=getattr(self.global_config, "enable_evidence_answer_arbiter", False),
                apply_rewrite=getattr(
                    self.global_config,
                    "evidence_answer_arbiter_apply_rewrite",
                    False,
                ),
                min_confidence=getattr(self.global_config, "evidence_answer_arbiter_min_confidence", 0.84),
                generic_min_confidence=getattr(
                    self.global_config,
                    "evidence_answer_arbiter_generic_min_confidence",
                    0.92,
                ),
                min_margin=getattr(self.global_config, "evidence_answer_arbiter_min_margin", 0.14),
                min_support_gain=getattr(
                    self.global_config,
                    "evidence_answer_arbiter_min_support_gain",
                    0.18,
                ),
                alias_min_support_gain=getattr(
                    self.global_config,
                    "evidence_answer_arbiter_alias_min_support_gain",
                    0.04,
                ),
                min_question_overlap=getattr(
                    self.global_config,
                    "evidence_answer_arbiter_min_question_overlap",
                    0.08,
                ),
                max_docs=getattr(self.global_config, "evidence_answer_arbiter_max_docs", 10),
                max_answer_tokens=getattr(
                    self.global_config,
                    "evidence_answer_arbiter_max_answer_tokens",
                    12,
                ),
                require_role_constraint=getattr(
                    self.global_config,
                    "evidence_answer_arbiter_require_role_constraint",
                    True,
                ),
                skip_supported_answers=getattr(
                    self.global_config,
                    "evidence_answer_arbiter_skip_supported_answers",
                    True,
                ),
            )
            query_solution.evidence_answer_arbiter = evidence_answer_arbiter
            evidence_answer_arbiter_counts[
                evidence_answer_arbiter.get("reason", "unavailable")
            ] += 1

            query_solution.answer_before_typed_evidence_table = pred_ans
            pred_ans, typed_evidence_table = arbitrate_answer_with_typed_evidence_table(
                question=query_solution.question,
                answer=pred_ans,
                qa_docs=query_solution.qa_docs,
                answer_style=query_solution.answer_style,
                enabled=getattr(self.global_config, "enable_typed_evidence_table_arbiter", False),
                apply_rewrite=getattr(
                    self.global_config,
                    "typed_evidence_table_apply_rewrite",
                    False,
                ),
                min_confidence=getattr(self.global_config, "typed_evidence_table_min_confidence", 0.86),
                generic_min_confidence=getattr(
                    self.global_config,
                    "typed_evidence_table_generic_min_confidence",
                    0.92,
                ),
                min_margin=getattr(self.global_config, "typed_evidence_table_min_margin", 0.12),
                min_support_gain=getattr(
                    self.global_config,
                    "typed_evidence_table_min_support_gain",
                    0.12,
                ),
                alias_min_support_gain=getattr(
                    self.global_config,
                    "typed_evidence_table_alias_min_support_gain",
                    0.02,
                ),
                min_question_overlap=getattr(
                    self.global_config,
                    "typed_evidence_table_min_question_overlap",
                    0.08,
                ),
                max_docs=getattr(self.global_config, "typed_evidence_table_max_docs", 10),
                max_answer_tokens=getattr(
                    self.global_config,
                    "typed_evidence_table_max_answer_tokens",
                    12,
                ),
                min_provenance=getattr(self.global_config, "typed_evidence_table_min_provenance", 1),
                require_role_constraint=getattr(
                    self.global_config,
                    "typed_evidence_table_require_role_constraint",
                    True,
                ),
                skip_supported_answers=getattr(
                    self.global_config,
                    "typed_evidence_table_skip_supported_answers",
                    True,
                ),
            )
            query_solution.typed_evidence_table = typed_evidence_table
            typed_evidence_table_counts[
                typed_evidence_table.get("reason", "unavailable")
            ] += 1

            query_solution.answer_before_contrastive_arbitration = pred_ans
            pred_ans, contrastive_answer_arbitration = arbitrate_answer_with_contrastive_candidates(
                question=query_solution.question,
                answer=pred_ans,
                qa_docs=query_solution.qa_docs,
                answer_style=query_solution.answer_style,
                enabled=getattr(self.global_config, "enable_contrastive_answer_arbitration", False),
                apply_rewrite=getattr(
                    self.global_config,
                    "contrastive_answer_arbitration_apply_rewrite",
                    False,
                ),
                min_confidence=getattr(
                    self.global_config,
                    "contrastive_answer_arbitration_min_confidence",
                    0.90,
                ),
                generic_min_confidence=getattr(
                    self.global_config,
                    "contrastive_answer_arbitration_generic_min_confidence",
                    0.94,
                ),
                min_contrast_gain=getattr(
                    self.global_config,
                    "contrastive_answer_arbitration_min_contrast_gain",
                    0.30,
                ),
                min_margin=getattr(
                    self.global_config,
                    "contrastive_answer_arbitration_min_margin",
                    0.14,
                ),
                min_question_overlap=getattr(
                    self.global_config,
                    "contrastive_answer_arbitration_min_question_overlap",
                    0.08,
                ),
                max_docs=getattr(self.global_config, "contrastive_answer_arbitration_max_docs", 10),
                max_answer_tokens=getattr(
                    self.global_config,
                    "contrastive_answer_arbitration_max_answer_tokens",
                    12,
                ),
                min_provenance=getattr(
                    self.global_config,
                    "contrastive_answer_arbitration_min_provenance",
                    2,
                ),
                require_role_constraint=getattr(
                    self.global_config,
                    "contrastive_answer_arbitration_require_role_constraint",
                    True,
                ),
                require_supported_incumbent=getattr(
                    self.global_config,
                    "contrastive_answer_arbitration_require_supported_incumbent",
                    True,
                ),
                allow_alias_challengers=getattr(
                    self.global_config,
                    "contrastive_answer_arbitration_allow_alias_challengers",
                    False,
                ),
            )
            query_solution.contrastive_answer_arbitration = contrastive_answer_arbitration
            contrastive_answer_arbitration_counts[
                contrastive_answer_arbitration.get("reason", "unavailable")
            ] += 1

            query_solution.answer_before_provenance_lattice = pred_ans
            pred_ans, provenance_answer_lattice = arbitrate_answer_with_provenance_lattice(
                question=query_solution.question,
                answer=pred_ans,
                qa_docs=query_solution.qa_docs,
                answer_style=query_solution.answer_style,
                evidence_consistent_reader=query_solution.evidence_consistent_reader,
                evidence_object_arbitration=query_solution.evidence_object_arbitration,
                typed_evidence_table=query_solution.typed_evidence_table,
                contrastive_answer_arbitration=query_solution.contrastive_answer_arbitration,
                anchored_typed_candidate_table=query_solution.anchored_typed_candidate_table,
                relation_fact_join=query_solution.relation_fact_join,
                enabled=getattr(self.global_config, "enable_provenance_answer_lattice", False),
                apply_rewrite=getattr(
                    self.global_config,
                    "provenance_answer_lattice_apply_rewrite",
                    False,
                ),
                min_score=getattr(self.global_config, "provenance_answer_lattice_min_score", 0.88),
                generic_min_score=getattr(
                    self.global_config,
                    "provenance_answer_lattice_generic_min_score",
                    0.93,
                ),
                min_margin=getattr(self.global_config, "provenance_answer_lattice_min_margin", 0.12),
                min_support_gain=getattr(
                    self.global_config,
                    "provenance_answer_lattice_min_support_gain",
                    0.14,
                ),
                alias_min_support_gain=getattr(
                    self.global_config,
                    "provenance_answer_lattice_alias_min_support_gain",
                    0.02,
                ),
                min_sources=getattr(self.global_config, "provenance_answer_lattice_min_sources", 2),
                min_provenance=getattr(self.global_config, "provenance_answer_lattice_min_provenance", 2),
                min_question_overlap=getattr(
                    self.global_config,
                    "provenance_answer_lattice_min_question_overlap",
                    0.08,
                ),
                max_docs=getattr(self.global_config, "provenance_answer_lattice_max_docs", 10),
                max_answer_tokens=getattr(
                    self.global_config,
                    "provenance_answer_lattice_max_answer_tokens",
                    12,
                ),
                require_role_constraint=getattr(
                    self.global_config,
                    "provenance_answer_lattice_require_role_constraint",
                    True,
                ),
                skip_supported_answers=getattr(
                    self.global_config,
                    "provenance_answer_lattice_skip_supported_answers",
                    True,
                ),
            )
            query_solution.provenance_answer_lattice = provenance_answer_lattice
            provenance_answer_lattice_counts[
                provenance_answer_lattice.get("reason", "unavailable")
            ] += 1

            query_solution.answer_before_provenance_typed_candidate_store = pred_ans
            pred_ans, provenance_typed_candidate_store = arbitrate_answer_with_provenance_typed_candidate_store(
                question=query_solution.question,
                answer=pred_ans,
                qa_docs=query_solution.qa_docs,
                answer_style=query_solution.answer_style,
                evidence_consistent_reader=query_solution.evidence_consistent_reader,
                evidence_object_arbitration=query_solution.evidence_object_arbitration,
                evidence_answer_arbiter=query_solution.evidence_answer_arbiter,
                typed_evidence_table=query_solution.typed_evidence_table,
                contrastive_answer_arbitration=query_solution.contrastive_answer_arbitration,
                provenance_answer_lattice=query_solution.provenance_answer_lattice,
                anchored_typed_candidate_table=query_solution.anchored_typed_candidate_table,
                candidate_answer_selector=query_solution.candidate_answer_selector,
                relation_fact_join=query_solution.relation_fact_join,
                enabled=getattr(
                    self.global_config,
                    "enable_provenance_typed_candidate_store",
                    False,
                ),
                apply_rewrite=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_apply_rewrite",
                    False,
                ),
                min_score=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_min_score",
                    0.88,
                ),
                generic_min_score=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_generic_min_score",
                    0.92,
                ),
                min_net_utility=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_min_net_utility",
                    0.46,
                ),
                min_margin=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_min_margin",
                    0.08,
                ),
                min_support_gain=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_min_support_gain",
                    0.08,
                ),
                alias_min_support_gain=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_alias_min_support_gain",
                    0.01,
                ),
                min_question_overlap=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_min_question_overlap",
                    0.06,
                ),
                max_docs=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_max_docs",
                    10,
                ),
                max_answer_tokens=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_max_answer_tokens",
                    12,
                ),
                max_candidates=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_max_candidates",
                    16,
                ),
                min_sources=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_min_sources",
                    1,
                ),
                min_provenance=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_min_provenance",
                    1,
                ),
                high_confidence_single_source=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_high_confidence_single_source",
                    0.94,
                ),
                token_weight=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_token_weight",
                    0.08,
                ),
                provenance_weight=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_provenance_weight",
                    0.035,
                ),
                risk_weight=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_risk_weight",
                    0.60,
                ),
                require_role_constraint=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_require_role_constraint",
                    True,
                ),
                skip_supported_answers=getattr(
                    self.global_config,
                    "provenance_typed_candidate_store_skip_supported_answers",
                    True,
                ),
            )
            query_solution.provenance_typed_candidate_store = provenance_typed_candidate_store
            provenance_typed_candidate_store_counts[
                provenance_typed_candidate_store.get("reason", "unavailable")
            ] += 1

            query_solution.answer_before_evidence_grounded_extractor = pred_ans
            pred_ans, evidence_grounded_answer_extractor = arbitrate_answer_with_evidence_grounded_extractor(
                question=query_solution.question,
                answer=pred_ans,
                qa_docs=query_solution.qa_docs,
                enabled=getattr(
                    self.global_config,
                    "enable_evidence_grounded_answer_extractor",
                    False,
                ),
                apply_rewrite=getattr(
                    self.global_config,
                    "evidence_grounded_answer_extractor_apply_rewrite",
                    False,
                ),
                min_score=getattr(
                    self.global_config,
                    "evidence_grounded_answer_extractor_min_score",
                    0.84,
                ),
                generic_min_score=getattr(
                    self.global_config,
                    "evidence_grounded_answer_extractor_generic_min_score",
                    0.90,
                ),
                min_margin=getattr(
                    self.global_config,
                    "evidence_grounded_answer_extractor_min_margin",
                    0.08,
                ),
                min_support_gain=getattr(
                    self.global_config,
                    "evidence_grounded_answer_extractor_min_support_gain",
                    0.06,
                ),
                alias_min_support_gain=getattr(
                    self.global_config,
                    "evidence_grounded_answer_extractor_alias_min_support_gain",
                    0.01,
                ),
                min_provenance=getattr(
                    self.global_config,
                    "evidence_grounded_answer_extractor_min_provenance",
                    1,
                ),
                min_question_overlap=getattr(
                    self.global_config,
                    "evidence_grounded_answer_extractor_min_question_overlap",
                    0.02,
                ),
                max_docs=getattr(
                    self.global_config,
                    "evidence_grounded_answer_extractor_max_docs",
                    8,
                ),
                max_answer_tokens=getattr(
                    self.global_config,
                    "evidence_grounded_answer_extractor_max_answer_tokens",
                    14,
                ),
                max_candidates=getattr(
                    self.global_config,
                    "evidence_grounded_answer_extractor_max_candidates",
                    12,
                ),
                require_role_constraint=getattr(
                    self.global_config,
                    "evidence_grounded_answer_extractor_require_role_constraint",
                    True,
                ),
                skip_supported_answers=getattr(
                    self.global_config,
                    "evidence_grounded_answer_extractor_skip_supported_answers",
                    True,
                ),
            )
            if not getattr(self.global_config, "enable_evidence_grounded_answer_extractor", False):
                evidence_grounded_answer_extractor.update({
                    "enabled": False,
                    "changed": False,
                    "action": "disabled",
                    "reason": "disabled",
                    "after": query_solution.answer_before_evidence_grounded_extractor,
                })
                pred_ans = query_solution.answer_before_evidence_grounded_extractor
            query_solution.evidence_grounded_answer_extractor = evidence_grounded_answer_extractor
            evidence_grounded_answer_extractor_counts[
                evidence_grounded_answer_extractor.get("reason", "unavailable")
            ] += 1

            query_solution.answer_before_query_answer_contract = pred_ans
            pred_ans, query_answer_contract = arbitrate_answer_with_query_answer_contract(
                question=query_solution.question,
                answer=pred_ans,
                qa_docs=query_solution.qa_docs,
                answer_style=query_solution.answer_style,
                enabled=getattr(
                    self.global_config,
                    "enable_query_answer_contract",
                    False,
                ),
                apply_rewrite=getattr(
                    self.global_config,
                    "query_answer_contract_apply_rewrite",
                    False,
                ),
                min_score=getattr(
                    self.global_config,
                    "query_answer_contract_min_score",
                    0.78,
                ),
                generic_min_score=getattr(
                    self.global_config,
                    "query_answer_contract_generic_min_score",
                    0.82,
                ),
                min_margin=getattr(
                    self.global_config,
                    "query_answer_contract_min_margin",
                    0.05,
                ),
                min_support_gain=getattr(
                    self.global_config,
                    "query_answer_contract_min_support_gain",
                    0.03,
                ),
                min_relation_score=getattr(
                    self.global_config,
                    "query_answer_contract_min_relation_score",
                    0.50,
                ),
                max_docs=getattr(
                    self.global_config,
                    "query_answer_contract_max_docs",
                    8,
                ),
                max_answer_tokens=getattr(
                    self.global_config,
                    "query_answer_contract_max_answer_tokens",
                    14,
                ),
                max_candidates=getattr(
                    self.global_config,
                    "query_answer_contract_max_candidates",
                    12,
                ),
                min_question_overlap=getattr(
                    self.global_config,
                    "query_answer_contract_min_question_overlap",
                    0.02,
                ),
                require_role_constraint=getattr(
                    self.global_config,
                    "query_answer_contract_require_role_constraint",
                    True,
                ),
                protect_supported_answers=getattr(
                    self.global_config,
                    "query_answer_contract_protect_supported_answers",
                    True,
                ),
            )
            query_solution.query_answer_contract = query_answer_contract
            query_answer_contract_counts[
                query_answer_contract.get("reason", "unavailable")
            ] += 1

            query_solution.answer_before_provenance_answer_slot_store = pred_ans
            pred_ans, provenance_answer_slot_store = arbitrate_answer_with_provenance_answer_slot_store(
                question=query_solution.question,
                answer=pred_ans,
                qa_docs=query_solution.qa_docs,
                answer_style=query_solution.answer_style,
                evidence_consistent_reader=query_solution.evidence_consistent_reader,
                evidence_object_arbitration=query_solution.evidence_object_arbitration,
                evidence_answer_arbiter=query_solution.evidence_answer_arbiter,
                typed_evidence_table=query_solution.typed_evidence_table,
                contrastive_answer_arbitration=query_solution.contrastive_answer_arbitration,
                provenance_answer_lattice=query_solution.provenance_answer_lattice,
                provenance_typed_candidate_store=query_solution.provenance_typed_candidate_store,
                evidence_grounded_answer_extractor=query_solution.evidence_grounded_answer_extractor,
                query_answer_contract=query_solution.query_answer_contract,
                anchored_typed_candidate_table=query_solution.anchored_typed_candidate_table,
                candidate_answer_selector=query_solution.candidate_answer_selector,
                relation_fact_join=query_solution.relation_fact_join,
                enabled=getattr(
                    self.global_config,
                    "enable_provenance_answer_slot_store",
                    False,
                ),
                apply_rewrite=getattr(
                    self.global_config,
                    "provenance_answer_slot_store_apply_rewrite",
                    True,
                ),
                min_score=getattr(
                    self.global_config,
                    "provenance_answer_slot_store_min_score",
                    0.80,
                ),
                generic_min_score=getattr(
                    self.global_config,
                    "provenance_answer_slot_store_generic_min_score",
                    0.86,
                ),
                min_slot_store_score=getattr(
                    self.global_config,
                    "provenance_answer_slot_store_min_slot_store_score",
                    0.44,
                ),
                min_margin=getattr(
                    self.global_config,
                    "provenance_answer_slot_store_min_margin",
                    0.04,
                ),
                min_support_gain=getattr(
                    self.global_config,
                    "provenance_answer_slot_store_min_support_gain",
                    0.05,
                ),
                alias_min_support_gain=getattr(
                    self.global_config,
                    "provenance_answer_slot_store_alias_min_support_gain",
                    0.01,
                ),
                min_question_overlap=getattr(
                    self.global_config,
                    "provenance_answer_slot_store_min_question_overlap",
                    0.02,
                ),
                max_docs=getattr(
                    self.global_config,
                    "provenance_answer_slot_store_max_docs",
                    10,
                ),
                max_answer_tokens=getattr(
                    self.global_config,
                    "provenance_answer_slot_store_max_answer_tokens",
                    14,
                ),
                max_candidates=getattr(
                    self.global_config,
                    "provenance_answer_slot_store_max_candidates",
                    18,
                ),
                min_base_provenance=getattr(
                    self.global_config,
                    "provenance_answer_slot_store_min_base_provenance",
                    1,
                ),
                min_source_diversity=getattr(
                    self.global_config,
                    "provenance_answer_slot_store_min_source_diversity",
                    1,
                ),
                require_role_constraint=getattr(
                    self.global_config,
                    "provenance_answer_slot_store_require_role_constraint",
                    True,
                ),
                protect_supported_answers=getattr(
                    self.global_config,
                    "provenance_answer_slot_store_protect_supported_answers",
                    True,
                ),
                min_relation_entailment_score=getattr(
                    self.global_config,
                    "provenance_answer_slot_store_min_relation_entailment_score",
                    0.20,
                ),
                min_relation_question_overlap=getattr(
                    self.global_config,
                    "provenance_answer_slot_store_min_relation_question_overlap",
                    0.10,
                ),
            )
            query_solution.provenance_answer_slot_store = provenance_answer_slot_store
            provenance_answer_slot_store_counts[
                provenance_answer_slot_store.get("reason", "unavailable")
            ] += 1

            query_solution.answer_before_answer_slot_binding_optimizer = pred_ans
            pred_ans, answer_slot_binding_optimizer = optimize_answer_slot_binding(
                question=query_solution.question,
                answer=pred_ans,
                qa_docs=query_solution.qa_docs,
                answer_style=query_solution.answer_style,
                evidence_consistent_reader=query_solution.evidence_consistent_reader,
                evidence_object_arbitration=query_solution.evidence_object_arbitration,
                evidence_answer_arbiter=query_solution.evidence_answer_arbiter,
                typed_evidence_table=query_solution.typed_evidence_table,
                contrastive_answer_arbitration=query_solution.contrastive_answer_arbitration,
                provenance_answer_lattice=query_solution.provenance_answer_lattice,
                provenance_typed_candidate_store=query_solution.provenance_typed_candidate_store,
                evidence_grounded_answer_extractor=query_solution.evidence_grounded_answer_extractor,
                query_answer_contract=query_solution.query_answer_contract,
                provenance_answer_slot_store=query_solution.provenance_answer_slot_store,
                anchored_typed_candidate_table=query_solution.anchored_typed_candidate_table,
                candidate_answer_selector=query_solution.candidate_answer_selector,
                relation_fact_join=query_solution.relation_fact_join,
                enabled=getattr(
                    self.global_config,
                    "enable_answer_slot_binding_optimizer",
                    False,
                ),
                apply_rewrite=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_apply_rewrite",
                    True,
                ),
                min_score=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_min_score",
                    0.78,
                ),
                generic_min_score=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_generic_min_score",
                    0.84,
                ),
                min_binding_score=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_min_binding_score",
                    0.42,
                ),
                min_margin=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_min_margin",
                    0.04,
                ),
                min_support_gain=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_min_support_gain",
                    0.04,
                ),
                min_slot_mismatch_gain=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_min_slot_mismatch_gain",
                    0.08,
                ),
                alias_min_support_gain=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_alias_min_support_gain",
                    0.01,
                ),
                max_docs=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_max_docs",
                    10,
                ),
                max_answer_tokens=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_max_answer_tokens",
                    14,
                ),
                max_candidates=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_max_candidates",
                    18,
                ),
                min_base_provenance=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_min_base_provenance",
                    1,
                ),
                require_role_constraint=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_require_role_constraint",
                    True,
                ),
                protect_supported_answers=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_protect_supported_answers",
                    True,
                ),
                enable_format_alias_rewrite=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_enable_format_alias_rewrite",
                    True,
                ),
                enable_slot_mismatch_rewrite=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_enable_slot_mismatch_rewrite",
                    False,
                ),
                enable_unsupported_rewrite=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_enable_unsupported_rewrite",
                    False,
                ),
                enable_cross_module_rewrite=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_enable_cross_module_rewrite",
                    False,
                ),
                min_relation_entailment_score=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_min_relation_entailment_score",
                    0.20,
                ),
                min_relation_question_overlap=getattr(
                    self.global_config,
                    "answer_slot_binding_optimizer_min_relation_question_overlap",
                    0.10,
                ),
            )
            pred_ans, fact_view_surface_guard = (
                protect_provenance_fact_view_surface(
                    before=(
                        query_solution.answer_before_answer_slot_binding_optimizer
                    ),
                    after=pred_ans,
                    schema_critical_info=(
                        query_solution.schema_critical_evidence
                    ),
                    answer_slot_binding_optimizer=(
                        answer_slot_binding_optimizer
                    ),
                )
            )
            answer_slot_binding_optimizer[
                "provenance_fact_view_surface_guard"
            ] = fact_view_surface_guard
            if fact_view_surface_guard.get("applied"):
                answer_slot_binding_optimizer.update({
                    "changed": False,
                    "action": "keep",
                    "reason": "provenance_fact_view_surface_protected",
                    "after": pred_ans,
                })
            query_solution.answer_slot_binding_optimizer = answer_slot_binding_optimizer
            answer_slot_binding_optimizer_counts[
                answer_slot_binding_optimizer.get("reason", "unavailable")
            ] += 1

            query_solution.answer_before_answer_surface_lattice_optimizer = pred_ans
            pred_ans, answer_surface_lattice_optimizer = optimize_answer_surface_lattice(
                question=query_solution.question,
                answer=pred_ans,
                qa_docs=query_solution.qa_docs,
                answer_style=query_solution.answer_style,
                answer_before_evidence_grounded_extractor=(
                    query_solution.answer_before_evidence_grounded_extractor
                ),
                answer_before_query_answer_contract=(
                    query_solution.answer_before_query_answer_contract
                ),
                answer_before_provenance_answer_slot_store=(
                    query_solution.answer_before_provenance_answer_slot_store
                ),
                answer_before_answer_slot_binding_optimizer=(
                    query_solution.answer_before_answer_slot_binding_optimizer
                ),
                evidence_grounded_answer_extractor=query_solution.evidence_grounded_answer_extractor,
                query_answer_contract=query_solution.query_answer_contract,
                provenance_answer_slot_store=query_solution.provenance_answer_slot_store,
                answer_slot_binding_optimizer=query_solution.answer_slot_binding_optimizer,
                provenance_typed_candidate_store=query_solution.provenance_typed_candidate_store,
                relation_fact_join=query_solution.relation_fact_join,
                enabled=getattr(
                    self.global_config,
                    "enable_answer_surface_lattice_optimizer",
                    False,
                ),
                apply_rewrite=getattr(
                    self.global_config,
                    "answer_surface_lattice_optimizer_apply_rewrite",
                    True,
                ),
                min_net_utility=getattr(
                    self.global_config,
                    "answer_surface_lattice_optimizer_min_net_utility",
                    0.08,
                ),
                min_margin=getattr(
                    self.global_config,
                    "answer_surface_lattice_optimizer_min_margin",
                    0.035,
                ),
                min_support_gain=getattr(
                    self.global_config,
                    "answer_surface_lattice_optimizer_min_support_gain",
                    0.02,
                ),
                max_risk=getattr(
                    self.global_config,
                    "answer_surface_lattice_optimizer_max_risk",
                    0.48,
                ),
                max_docs=getattr(
                    self.global_config,
                    "answer_surface_lattice_optimizer_max_docs",
                    10,
                ),
                max_answer_tokens=getattr(
                    self.global_config,
                    "answer_surface_lattice_optimizer_max_answer_tokens",
                    16,
                ),
                risk_weight=getattr(
                    self.global_config,
                    "answer_surface_lattice_optimizer_risk_weight",
                    0.65,
                ),
                prefer_contract_margin=getattr(
                    self.global_config,
                    "answer_surface_lattice_optimizer_prefer_contract_margin",
                    0.015,
                ),
                abstain_min_runner_up_margin=getattr(
                    self.global_config,
                    "answer_surface_lattice_optimizer_abstain_min_runner_up_margin",
                    0.08,
                ),
                require_lineage_for_relational_abstain=getattr(
                    self.global_config,
                    "answer_surface_lattice_optimizer_require_lineage_for_relational_abstain",
                    True,
                ),
            )
            query_solution.answer_surface_lattice_optimizer = answer_surface_lattice_optimizer
            answer_surface_lattice_optimizer_counts[
                answer_surface_lattice_optimizer.get("reason", "unavailable")
            ] += 1

            query_solution.answer_before_schema_surface_canonicalization = pred_ans
            pred_ans, schema_surface_canonicalization = (
                canonicalize_schema_touched_answer_surface(
                    question=query_solution.question,
                    answer=pred_ans,
                    qa_docs=query_solution.qa_docs,
                    evidence_admission=query_solution.evidence_admission,
                    query_schema_memory=query_solution.query_schema_memory,
                    answer_style=query_solution.answer_style,
                    enabled=getattr(
                        self.global_config,
                        "enable_schema_touched_answer_surface_canonicalization",
                        False,
                    ),
                    min_confidence=getattr(
                        self.global_config,
                        "schema_touched_answer_surface_canonicalization_min_confidence",
                        0.78,
                    ),
                )
            )
            query_solution.schema_surface_canonicalization = (
                schema_surface_canonicalization
            )
            schema_surface_canonicalization_counts[
                schema_surface_canonicalization.get(
                    "reason",
                    schema_surface_canonicalization.get("action", "unavailable"),
                )
            ] += 1

            query_solution.answer_before_plan_consistent_answer_verifier = pred_ans
            pred_ans, plan_consistent_answer_verifier = verify_answer_with_plan_consistency(
                question=query_solution.question,
                answer=pred_ans,
                qa_docs=(query_solution.qa_source_docs or query_solution.qa_docs or []),
                answer_style=query_solution.answer_style,
                evidence_grounded_answer_extractor=query_solution.evidence_grounded_answer_extractor,
                query_answer_contract=query_solution.query_answer_contract,
                provenance_answer_slot_store=query_solution.provenance_answer_slot_store,
                answer_slot_binding_optimizer=query_solution.answer_slot_binding_optimizer,
                answer_surface_lattice_optimizer=query_solution.answer_surface_lattice_optimizer,
                provenance_typed_candidate_store=query_solution.provenance_typed_candidate_store,
                typed_evidence_table=query_solution.typed_evidence_table,
                relation_fact_join=query_solution.relation_fact_join,
                evidence_admission=query_solution.evidence_admission,
                enabled=getattr(
                    self.global_config,
                    "enable_plan_consistent_answer_verifier",
                    False,
                ),
                apply_rewrite=getattr(
                    self.global_config,
                    "plan_consistent_answer_verifier_apply_rewrite",
                    False,
                ),
                min_plan_score=getattr(
                    self.global_config,
                    "plan_consistent_answer_verifier_min_plan_score",
                    0.42,
                ),
                min_margin=getattr(
                    self.global_config,
                    "plan_consistent_answer_verifier_min_margin",
                    0.055,
                ),
                min_relation_gain=getattr(
                    self.global_config,
                    "plan_consistent_answer_verifier_min_relation_gain",
                    0.035,
                ),
                min_support_score=getattr(
                    self.global_config,
                    "plan_consistent_answer_verifier_min_support_score",
                    0.20,
                ),
                max_docs=getattr(
                    self.global_config,
                    "plan_consistent_answer_verifier_max_docs",
                    10,
                ),
                max_candidates=getattr(
                    self.global_config,
                    "plan_consistent_answer_verifier_max_candidates",
                    18,
                ),
                protect_supported_answers=getattr(
                    self.global_config,
                    "plan_consistent_answer_verifier_protect_supported_answers",
                    True,
                ),
                allow_non_abstain_unsupported_rewrite=getattr(
                    self.global_config,
                    "plan_consistent_answer_verifier_allow_non_abstain_unsupported_rewrite",
                    False,
                ),
                allow_slot_rewrite=getattr(
                    self.global_config,
                    "plan_consistent_answer_verifier_allow_slot_rewrite",
                    False,
                ),
                allow_relation_rewrite=getattr(
                    self.global_config,
                    "plan_consistent_answer_verifier_allow_relation_rewrite",
                    False,
                ),
                allow_format_rewrite=getattr(
                    self.global_config,
                    "plan_consistent_answer_verifier_allow_format_rewrite",
                    False,
                ),
            )
            query_solution.plan_consistent_answer_verifier = plan_consistent_answer_verifier
            plan_consistent_answer_verifier_counts[
                plan_consistent_answer_verifier.get("reason", "unavailable")
            ] += 1

            query_solution.answer_before_cost_gated_reread = pred_ans
            pred_ans, cost_gated_reread = cost_gated_constrained_reread(
                question=query_solution.question,
                answer=pred_ans,
                qa_docs=(query_solution.qa_source_docs or query_solution.qa_docs or []),
                query_answer_contract=query_solution.query_answer_contract,
                answer_slot_binding_optimizer=query_solution.answer_slot_binding_optimizer,
                evidence_admission=query_solution.evidence_admission,
                llm_infer=self.llm_model.infer,
                enabled=getattr(
                    self.global_config,
                    "enable_cost_gated_constrained_reread",
                    False,
                ),
                apply_rewrite=getattr(
                    self.global_config,
                    "cost_gated_constrained_reread_apply_rewrite",
                    True,
                ),
                min_net_utility=getattr(
                    self.global_config,
                    "cost_gated_constrained_reread_min_net_utility",
                    0.42,
                ),
                min_candidate_support=getattr(
                    self.global_config,
                    "cost_gated_constrained_reread_min_candidate_support",
                    0.25,
                ),
                min_support_gain=getattr(
                    self.global_config,
                    "cost_gated_constrained_reread_min_support_gain",
                    0.05,
                ),
                max_docs=getattr(
                    self.global_config,
                    "cost_gated_constrained_reread_max_docs",
                    8,
                ),
                max_doc_chars=getattr(
                    self.global_config,
                    "cost_gated_constrained_reread_max_doc_chars",
                    900,
                ),
                max_total_chars=getattr(
                    self.global_config,
                    "cost_gated_constrained_reread_max_total_chars",
                    6200,
                ),
                max_answer_tokens=getattr(
                    self.global_config,
                    "cost_gated_constrained_reread_max_answer_tokens",
                    16,
                ),
                max_completion_tokens=getattr(
                    self.global_config,
                    "cost_gated_constrained_reread_max_completion_tokens",
                    120,
                ),
                require_role_constraint=getattr(
                    self.global_config,
                    "cost_gated_constrained_reread_require_role_constraint",
                    True,
                ),
            )
            query_solution.cost_gated_constrained_reread = cost_gated_reread
            cost_gated_reread_counts[
                cost_gated_reread.get("reason", "unavailable")
            ] += 1

            qa_timing = dict(query_solution.qa_timing or {})
            reader_metadata = all_metadata[query_solution_idx] or {}
            qa_timing["reader_seconds"] = round(
                float(reader_metadata.get("qa_reader_seconds", 0.0) or 0.0),
                6,
            )
            qa_timing["answer_materialization_seconds"] = round(
                time.perf_counter() - answer_materialization_started,
                6,
            )
            qa_timing["online_total_seconds"] = round(
                sum(
                    float(qa_timing.get(field, 0.0) or 0.0)
                    for field in (
                        "retrieval_seconds",
                        "evidence_preparation_seconds",
                        "reader_seconds",
                        "answer_materialization_seconds",
                    )
                ),
                6,
            )
            query_solution.qa_timing = qa_timing
            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)




        materializer_jobs = []
        materializer_enabled = bool(
            getattr(self.global_config, "enable_provenance_bound_answer_materializer", False)
        )
        for query_idx, query_solution in enumerate(queries_solutions):
            query_solution.answer_before_provenance_bound_materializer = query_solution.answer
            adaptive_trace = dict(query_solution.adaptive_query_execution or {})
            relation_trace = dict(query_solution.relation_fact_join or {})
            stage_trace = dict(relation_trace.get("stage_aware_fact_join") or {})
            schema_critical_trace = dict(query_solution.schema_critical_evidence or {})
            paper_trace = dict(query_solution.paper_evidence_execution or {})
            logical_plan = select_authoritative_logical_query_plan(
                query_solution.question,
                query_solution.answer_style,
                (
                    (
                        "paper_evidence_executor",
                        paper_trace.get("logical_plan"),
                    ),
                    ("schema_critical_executor", schema_critical_trace.get("logical_plan")),
                    ("adaptive_query_executor", adaptive_trace.get("logical_plan")),
                    ("relation_fact_join", stage_trace.get("plan")),
                ),
            )
            physical_execution_policy = dict(
                adaptive_trace.get("logical_plan_execution_policy")
                or (
                    adaptive_trace.get("logical_plan") or {}
                ).get("physical_execution_policy")
                or {}
            )
            if physical_execution_policy:
                logical_plan["physical_execution_policy"] = (
                    physical_execution_policy
                )
                logical_plan["physical_plan_trace_source"] = (
                    "adaptive_query_executor"
                )
            query_solution.logical_query_plan = logical_plan
            if paper_trace:




                paper_trace.setdefault(
                    "executed_logical_plan",
                    dict(paper_trace.get("logical_plan") or {}),
                )
                paper_trace["published_logical_plan"] = dict(logical_plan)
                paper_trace["logical_plan_source"] = str(
                    logical_plan.get("query_plan_source") or "none"
                )
                paper_trace["logical_plan_ir_version"] = str(
                    logical_plan.get("query_plan_ir_version") or ""
                )
                query_solution.paper_evidence_execution = paper_trace
            if bool(getattr(
                self.global_config,
                "provenance_bound_answer_materializer_use_terminal_projection_rows",
                False,
            )):
                terminal_candidate_rows = list(
                    relation_trace.get("validated_terminal_projection_rows") or []
                )
            else:
                terminal_candidate_rows = list(
                    relation_trace.get("validated_terminal_candidate_rows") or []
                )
                terminal_candidate_rows.extend(
                    relation_trace.get("candidate_table_rows") or []
                )
            multi_hop_trace = dict(relation_trace.get("multi_hop_join") or {})
            query_plan_execution_trace = dict(
                multi_hop_trace.get("query_plan_execution_trace") or {}
            )
            terminal_tuple_completion_rows = list(
                query_plan_execution_trace.get(
                    "terminal_tuple_completion_candidates"
                ) or []
            )
            schema_terminal_rows = []
            if bool(getattr(
                self.global_config,
                "provenance_bound_answer_materializer_use_schema_terminal_rows",
                False,
            )):
                for raw_item in schema_critical_trace.get("selected") or []:
                    if not isinstance(raw_item, dict):
                        continue
                    item = dict(raw_item)
                    item["score"] = float(
                        item.get("score", item.get("utility", 0.0)) or 0.0
                    )
                    try:
                        source_doc_id = int(item.get("doc_id", -1))
                    except (TypeError, ValueError):
                        source_doc_id = -1
                    if source_doc_id < 0:
                        continue
                    try:
                        source_content = str(
                            self.chunk_embedding_store.get_row(
                                self.passage_node_keys[source_doc_id]
                            )["content"]
                            or ""
                        ).strip()
                    except Exception:
                        source_content = ""
                    lineage_view = str(
                        item.get("reader_text") or item.get("lineage_reader_text") or ""
                    ).strip()
                    if lineage_view and source_content:
                        item["schema_terminal_evidence"] = (
                            f"{lineage_view}\nSource passage: {source_content}"
                        )
                    else:
                        item["schema_terminal_evidence"] = lineage_view or source_content
                    schema_terminal_rows.append(item)
            messages, prepared_meta = prepare_provenance_bound_answer_materialization(
                question=query_solution.question,
                answer=query_solution.answer,
                raw_response=all_response_message[query_idx],
                qa_docs=(query_solution.qa_source_docs or query_solution.qa_docs or []),
                answer_style=query_solution.answer_style,
                logical_plan=logical_plan,
                terminal_candidate_rows=terminal_candidate_rows,
                terminal_tuple_completion_rows=terminal_tuple_completion_rows,
                schema_terminal_rows=schema_terminal_rows,
                schema_terminal_min_score=float(
                    getattr(
                        self.global_config,
                        "provenance_bound_answer_materializer_schema_terminal_min_score",
                        0.80,
                    )
                ),
                require_compiled_terminal_plan=bool(
                    getattr(
                        self.global_config,
                        "provenance_bound_answer_materializer_require_terminal_plan",
                        False,
                    )
                ),
                enable_deterministic_terminal_projection=bool(
                    getattr(
                        self.global_config,
                        "provenance_bound_answer_materializer_enable_deterministic_terminal_projection",
                        False,
                    )
                ),
                deterministic_terminal_min_net_utility=float(
                    getattr(
                        self.global_config,
                        "provenance_bound_answer_materializer_deterministic_min_net_utility",
                        0.70,
                    )
                ),
                enable_terminal_tuple_completion=bool(
                    getattr(
                        self.global_config,
                        "provenance_bound_answer_materializer_enable_terminal_tuple_completion",
                        False,
                    )
                ),
                terminal_tuple_completion_min_net_utility=float(
                    getattr(
                        self.global_config,
                        "provenance_bound_answer_materializer_terminal_tuple_completion_min_net_utility",
                        0.42,
                    )
                ),
                enabled=materializer_enabled,
                min_trigger_risk=getattr(
                    self.global_config,
                    "provenance_bound_answer_materializer_min_trigger_risk",
                    0.34,
                ),
                max_docs=getattr(
                    self.global_config,
                    "provenance_bound_answer_materializer_max_docs",
                    6,
                ),
                max_doc_chars=getattr(
                    self.global_config,
                    "provenance_bound_answer_materializer_max_doc_chars",
                    1050,
                ),
                max_total_chars=getattr(
                    self.global_config,
                    "provenance_bound_answer_materializer_max_total_chars",
                    6200,
                ),
            )
            query_solution.provenance_bound_answer_materializer = prepared_meta
            deterministic_projection = dict(
                prepared_meta.get("deterministic_terminal_projection") or {}
            )
            if deterministic_projection.get("accepted"):
                terminal_tuple_completion = dict(
                    prepared_meta.get("terminal_tuple_completion") or {}
                )
                completion_decision = bool(
                    terminal_tuple_completion.get("accepted")
                    and deterministic_projection.get("decision_source")
                    == "provenance_terminal_tuple_completion_v12_3"
                )
                apply_fast_path_rewrite = bool(
                    getattr(
                        self.global_config,
                        "provenance_bound_answer_materializer_apply_rewrite",
                        False,
                    )
                )
                if completion_decision:
                    apply_fast_path_rewrite = bool(
                        apply_fast_path_rewrite
                        and getattr(
                            self.global_config,
                            "provenance_bound_answer_materializer_terminal_tuple_completion_apply_rewrite",
                            False,
                        )
                    )
                final_answer, final_meta = finalize_deterministic_terminal_projection(
                    answer=query_solution.answer,
                    prepared_meta=prepared_meta,
                    apply_rewrite=apply_fast_path_rewrite,
                )
                query_solution.answer = final_answer
                query_solution.provenance_bound_answer_materializer = final_meta
            elif messages:
                materializer_jobs.append((query_idx, messages, prepared_meta))

        def _safe_answer_materializer_infer(job):
            query_idx, messages, prepared_meta = job
            started = time.perf_counter()
            try:
                result = self.llm_model.infer(
                    messages,
                    max_completion_tokens=int(
                        getattr(
                            self.global_config,
                            "provenance_bound_answer_materializer_max_completion_tokens",
                            120,
                        )
                    ),
                    temperature=0.0,
                )
                return query_idx, result, prepared_meta, "", time.perf_counter() - started
            except Exception as exc:
                return (
                    query_idx,
                    None,
                    prepared_meta,
                    f"{exc.__class__.__name__}: {str(exc)[:300]}",
                    time.perf_counter() - started,
                )

        materializer_results = []
        materializer_workers = max(
            1,
            int(
                getattr(
                    self.global_config,
                    "provenance_bound_answer_materializer_workers",
                    qa_inference_workers,
                )
            ),
        )
        if materializer_jobs:
            logger.info(
                "Provenance-bound answer materialization: triggered=%s workers=%s.",
                len(materializer_jobs),
                materializer_workers,
            )
            if materializer_workers > 1 and len(materializer_jobs) > 1:
                with ThreadPoolExecutor(max_workers=materializer_workers) as executor:
                    materializer_results = list(
                        tqdm(
                            executor.map(_safe_answer_materializer_infer, materializer_jobs),
                            total=len(materializer_jobs),
                            desc=f"Answer-domain materialization x{materializer_workers}",
                        )
                    )
            else:
                materializer_results = [
                    _safe_answer_materializer_infer(job)
                    for job in tqdm(materializer_jobs, desc="Answer-domain materialization")
                ]

        for query_idx, result, prepared_meta, error, elapsed in materializer_results:
            query_solution = queries_solutions[query_idx]
            if error:
                final_meta = dict(prepared_meta or {})
                final_meta.update({
                    "changed": False,
                    "action": "keep",
                    "reason": "materialization_inference_failed",
                    "error_preview": error,
                    "materialization_seconds": round(float(elapsed), 6),
                })
                query_solution.provenance_bound_answer_materializer = final_meta
                continue
            final_answer, final_meta = finalize_provenance_bound_answer_materialization(
                question=query_solution.question,
                answer=query_solution.answer,
                prepared_meta=prepared_meta,
                llm_result=result,
                apply_rewrite=getattr(
                    self.global_config,
                    "provenance_bound_answer_materializer_apply_rewrite",
                    False,
                ),
                min_candidate_confidence=getattr(
                    self.global_config,
                    "provenance_bound_answer_materializer_min_candidate_confidence",
                    0.72,
                ),
                min_binding_score=getattr(
                    self.global_config,
                    "provenance_bound_answer_materializer_min_binding_score",
                    0.50,
                ),
                min_binding_gain=getattr(
                    self.global_config,
                    "provenance_bound_answer_materializer_min_binding_gain",
                    0.08,
                ),
                schema_conflict_min_confidence=getattr(
                    self.global_config,
                    "provenance_bound_answer_materializer_schema_conflict_min_confidence",
                    0.85,
                ),
                max_answer_tokens=getattr(
                    self.global_config,
                    "provenance_bound_answer_materializer_max_answer_tokens",
                    14,
                ),
            )
            final_meta["materialization_seconds"] = round(float(elapsed), 6)
            query_solution.answer = final_answer
            query_solution.provenance_bound_answer_materializer = final_meta

        for query_solution in queries_solutions:
            materializer_meta = dict(query_solution.provenance_bound_answer_materializer or {})
            protected_answer, final_fact_view_surface_guard = (
                protect_provenance_fact_view_surface(
                    before=(
                        query_solution.answer_before_answer_slot_binding_optimizer
                    ),
                    after=query_solution.answer,
                    schema_critical_info=(
                        query_solution.schema_critical_evidence
                    ),
                    answer_slot_binding_optimizer=(
                        query_solution.answer_slot_binding_optimizer
                    ),
                )
            )
            query_solution.answer = protected_answer
            paper_trace = dict(query_solution.paper_evidence_execution or {})
            if paper_trace:
                qa_timing = dict(query_solution.qa_timing or {})
                fixed_stages = [
                    dict(stage) for stage in paper_trace.get("fixed_stages") or []
                ]
                for stage in fixed_stages:
                    if stage.get("stage") == "BASE_RETRIEVAL":
                        stage["wall_ms"] = round(
                            1000.0 * float(
                                qa_timing.get("retrieval_seconds", 0.0) or 0.0
                            ),
                            6,
                        )
                    elif stage.get("stage") == "READER":
                        stage["wall_ms"] = round(
                            1000.0 * float(
                                qa_timing.get("reader_seconds", 0.0) or 0.0
                            ),
                            6,
                        )
                paper_trace["fixed_stages"] = fixed_stages
                evidence_cost = dict(paper_trace.get("cost_ledger") or {})
                fixed_model_calls = sum(
                    int(stage.get("model_calls", 0) or 0) for stage in fixed_stages
                )
                fixed_retrieval_calls = sum(
                    int(stage.get("retrieval_calls", 0) or 0)
                    for stage in fixed_stages
                )
                fixed_input_tokens = sum(
                    int(stage.get("input_tokens", 0) or 0)
                    for stage in fixed_stages
                    if stage.get("input_tokens") is not None
                )
                fixed_output_tokens = sum(
                    int(stage.get("output_tokens", 0) or 0)
                    for stage in fixed_stages
                    if stage.get("output_tokens") is not None
                )
                fixed_unknown_token_events = sum(
                    int(
                        int(stage.get("model_calls", 0) or 0) > 0
                        and (
                            stage.get("input_tokens") is None
                            or stage.get("output_tokens") is None
                        )
                    )
                    for stage in fixed_stages
                )
                paper_trace["total_cost_ledger"] = {
                    "evidence_actions": int(evidence_cost.get("actions", 0) or 0),
                    "scan_rows": int(evidence_cost.get("scan_rows", 0) or 0),
                    "retrieval_calls": (
                        int(evidence_cost.get("retrieval_calls", 0) or 0)
                        + fixed_retrieval_calls
                    ),
                    "model_calls": (
                        int(evidence_cost.get("model_calls", 0) or 0)
                        + fixed_model_calls
                    ),
                    "input_tokens": (
                        int(evidence_cost.get("input_tokens", 0) or 0)
                        + fixed_input_tokens
                    ),
                    "output_tokens": (
                        int(evidence_cost.get("output_tokens", 0) or 0)
                        + fixed_output_tokens
                    ),
                    "unknown_token_events": (
                        int(evidence_cost.get("unknown_token_events", 0) or 0)
                        + fixed_unknown_token_events
                    ),
                    "planning_wall_ms": float(
                        evidence_cost.get("planning_wall_ms", 0.0) or 0.0
                    ),
                    "operator_wall_ms": float(
                        evidence_cost.get("operator_wall_ms", 0.0) or 0.0
                    ),
                    "online_end_to_end_wall_ms": round(
                        1000.0 * float(
                            qa_timing.get("online_total_seconds", 0.0) or 0.0
                        ),
                        6,
                    ),
                    "budget_violations": list(
                        evidence_cost.get("budget_violations") or []
                    ),
                }
                paper_trace["final_output"] = {
                    "path": str(
                        (paper_trace.get("answer_application") or {}).get("path")
                        or paper_trace.get("output_path")
                        or "ordinary_baseline_fallback"
                    ),
                    "answer": query_solution.answer,
                    "postprocessing_enabled": bool(
                        getattr(
                            self.global_config, "enable_answer_postprocessing", True
                        )
                    ),
                    "canonicalization": dict(query_solution.canonicalization or {}),
                }
                answer_application = dict(
                    paper_trace.get("answer_application") or {}
                )
                paper_trace["final_answer"] = query_solution.answer
                paper_trace["final_answer_origin"] = (
                    "structured_candidate"
                    if bool(answer_application.get("replaced"))
                    else "reader_pipeline"
                )
                query_solution.paper_evidence_execution = paper_trace
            materializer_meta["provenance_fact_view_surface_guard"] = (
                final_fact_view_surface_guard
            )
            provenance_bound_answer_materializer_counts[
                materializer_meta.get("reason", "unavailable")
            ] += 1
            materializer_seconds = float(
                materializer_meta.get("materialization_seconds", 0.0) or 0.0
            )
            qa_timing = dict(query_solution.qa_timing or {})
            qa_timing["provenance_bound_materializer_seconds"] = round(
                materializer_seconds,
                6,
            )
            qa_timing["online_total_seconds"] = round(
                float(qa_timing.get("online_total_seconds", 0.0) or 0.0)
                + materializer_seconds,
                6,
            )
            query_solution.qa_timing = qa_timing
            query_solution.provenance_bound_answer_materializer = materializer_meta

        logger.info(f"QA evidence-object arbitration counts: {dict(evidence_object_arbitration_counts)}")
        logger.info(f"QA evidence-answer arbiter counts: {dict(evidence_answer_arbiter_counts)}")
        logger.info(f"QA typed evidence-table counts: {dict(typed_evidence_table_counts)}")
        logger.info(f"QA contrastive answer arbitration counts: {dict(contrastive_answer_arbitration_counts)}")
        logger.info(f"QA provenance answer-lattice counts: {dict(provenance_answer_lattice_counts)}")
        logger.info(
            f"QA provenance-typed candidate store counts: {dict(provenance_typed_candidate_store_counts)}"
        )
        logger.info(
            f"QA evidence-grounded answer extractor counts: {dict(evidence_grounded_answer_extractor_counts)}"
        )
        logger.info(f"QA query-answer contract counts: {dict(query_answer_contract_counts)}")
        logger.info(
            f"QA provenance answer-slot store counts: {dict(provenance_answer_slot_store_counts)}"
        )
        logger.info(
            f"QA answer-slot binding optimizer counts: {dict(answer_slot_binding_optimizer_counts)}"
        )
        logger.info(
            f"QA answer-surface lattice optimizer counts: {dict(answer_surface_lattice_optimizer_counts)}"
        )
        logger.info(
            "QA schema-touched answer-surface canonicalization counts: "
            f"{dict(schema_surface_canonicalization_counts)}"
        )
        logger.info(
            f"QA plan-consistent answer verifier counts: {dict(plan_consistent_answer_verifier_counts)}"
        )
        logger.info(
            f"QA cost-gated constrained reread counts: {dict(cost_gated_reread_counts)}"
        )
        logger.info(
            "QA provenance-bound answer materializer counts: "
            f"{dict(provenance_bound_answer_materializer_counts)}"
        )
        original_results = queries_solutions[:original_query_count]
        for spec in final_reader_counterfactual_specs:
            original_index = int(spec["original_index"])
            counterfactual_index = int(spec["processing_index"])
            original_result = original_results[original_index]
            counterfactual_result = queries_solutions[counterfactual_index]
            full_metadata = dict(all_metadata[original_index] or {})
            counterfactual_metadata = dict(all_metadata[counterfactual_index] or {})
            operator = str(spec.get("operator") or "entity_predicate_fact_join")
            original_role = str(spec.get("original_role") or "treatment")
            if original_role == "treatment":
                treatment_result, treatment_metadata = original_result, full_metadata
                control_result, control_metadata = counterfactual_result, counterfactual_metadata
            else:
                treatment_result, treatment_metadata = counterfactual_result, counterfactual_metadata
                control_result, control_metadata = original_result, full_metadata
            original_result.final_reader_counterfactual = {
                "enabled": True,
                "applied": True,
                "operator": operator,
                "original_role": original_role,
                "same_retrieval_trace": True,
                "intervention": spec.get("intervention") or (
                    "reverse_deficit_budget_replacement"
                    if spec.get("restored_titles")
                    else "remove_from_final_reader_prompt"
                ),
                "intervention_metadata": dict(
                    spec.get("intervention_metadata") or {}
                ),
                "removed_titles": list(spec.get("removed_titles") or []),
                "restored_titles": list(spec.get("restored_titles") or []),
                "full_doc_count": int(spec.get("full_doc_count", 0) or 0),
                "counterfactual_doc_count": int(
                    spec.get("counterfactual_doc_count", 0) or 0
                ),
                "full_reader_answer": str(
                    original_result.answer_before_canonicalization or ""
                ),
                "counterfactual_reader_answer": str(
                    counterfactual_result.answer_before_canonicalization or ""
                ),
                "full_final_answer": str(original_result.answer or ""),
                "counterfactual_final_answer": str(
                    counterfactual_result.answer or ""
                ),
                "reader_answer_changed": bool(
                    str(original_result.answer_before_canonicalization or "").strip()
                    != str(
                        counterfactual_result.answer_before_canonicalization or ""
                    ).strip()
                ),
                "final_answer_changed": bool(
                    str(original_result.answer or "").strip()
                    != str(counterfactual_result.answer or "").strip()
                ),
                "full_prompt_tokens": int(
                    full_metadata.get("prompt_tokens", 0) or 0
                ),
                "counterfactual_prompt_tokens": int(
                    counterfactual_metadata.get("prompt_tokens", 0) or 0
                ),
                "control_reader_answer": str(
                    control_result.answer_before_canonicalization or ""
                ),
                "treatment_reader_answer": str(
                    treatment_result.answer_before_canonicalization or ""
                ),
                "control_final_answer": str(control_result.answer or ""),
                "treatment_final_answer": str(treatment_result.answer or ""),
                "control_prompt_tokens": int(
                    control_metadata.get("prompt_tokens", 0) or 0
                ),
                "treatment_prompt_tokens": int(
                    treatment_metadata.get("prompt_tokens", 0) or 0
                ),
            }
        return (
            original_results,
            all_response_message[:original_query_count],
            all_metadata[:original_query_count],
        )

    def _rebuild_fact_lookup_metadata(
        self,
        chunk_ids: List[str],
        chunk_triples: List[Tuple],
    ) -> None:
        """Build fact/entity lookup metadata without constructing a retrieval graph."""
        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}
        for chunk_key, triples in zip(chunk_ids, chunk_triples):
            entities_in_chunk = set()
            for triple in triples:
                if len(triple) != 3:
                    continue
                node_key = compute_mdhash_id(content=triple[0], prefix="entity-")
                node_2_key = compute_mdhash_id(content=triple[2], prefix="entity-")
                self.node_to_node_stats[(node_key, node_2_key)] = (
                    self.node_to_node_stats.get((node_key, node_2_key), 0.0) + 1
                )
                self.node_to_node_stats[(node_2_key, node_key)] = (
                    self.node_to_node_stats.get((node_2_key, node_key), 0.0) + 1
                )
                entities_in_chunk.update((node_key, node_2_key))
            for node_key in entities_in_chunk:
                self.ent_node_to_chunk_ids.setdefault(node_key, set()).add(chunk_key)

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    _SCHEMA_MEMORY_RELATION_STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
        "from", "had", "has", "have", "he", "her", "his", "in", "is", "it", "its",
        "of", "on", "or", "she", "that", "the", "their", "they", "this", "to",
        "was", "were", "which", "who", "with",
    }

    _SCHEMA_MEMORY_PLACE_TERMS = {
        "airport", "area", "bay", "borough", "capital", "city", "continent",
        "country", "county", "district", "island", "lake", "mountain", "province",
        "region", "republic", "river", "sea", "state", "territory", "town",
        "village",
    }

    _SCHEMA_MEMORY_ORG_TERMS = {
        "academy", "agency", "airlines", "association", "bank", "band", "club",
        "college", "company", "corporation", "department", "festival", "foundation",
        "group", "institute", "label", "league", "ministry", "museum", "party",
        "records", "school", "team", "university",
    }

    _SCHEMA_MEMORY_WORK_TERMS = {
        "album", "book", "film", "magazine", "movie", "novel", "opera", "play",
        "series", "song", "tournament", "work",
    }

    def _schema_memory_entity_type(self, entity: str) -> str:
        text = str(entity or "").strip().lower()
        if not text:
            return "unknown"
        if re.search(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b", text):
            return "date"
        if re.fullmatch(r"[\d,.\-/% ]+", text):
            return "number"
        tokens = set(re.findall(r"[a-z0-9]+", text))
        if tokens & self._SCHEMA_MEMORY_PLACE_TERMS:
            return "place"
        if tokens & self._SCHEMA_MEMORY_ORG_TERMS:
            return "organization"
        if tokens & self._SCHEMA_MEMORY_WORK_TERMS:
            return "work"
        return "entity"

    def _schema_memory_relation_key(self, relation: str) -> str:
        min_tokens = max(1, int(getattr(self.global_config, "schema_memory_min_relation_tokens", 1)))
        tokens = re.findall(r"[a-z0-9]+", str(relation or "").lower())
        informative_tokens = [
            token for token in tokens
            if len(token) > 1 and token not in self._SCHEMA_MEMORY_RELATION_STOPWORDS
        ]
        if len(informative_tokens) < min_tokens:
            return ""
        return " ".join(informative_tokens[:8])

    def _schema_memory_signature(self, triple: Tuple) -> Optional[Tuple[str, str, str]]:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            return None
        relation_key = self._schema_memory_relation_key(triple[1])
        if not relation_key:
            return None
        return (
            self._schema_memory_entity_type(triple[0]),
            relation_key,
            self._schema_memory_entity_type(triple[2]),
        )

    def add_schema_memory_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """Add weak passage-passage links for chunks sharing an OpenIE relation schema.

        This is a lightweight MemGraphRAG-inspired schema layer: rather than creating
        explicit schema nodes, it materializes bounded weak edges between passages that
        express the same coarse (head_type, relation, tail_type) pattern. PPR can then
        propagate from dense/fact seeds through relation-level evidence bridges without
        an extra LLM schema-extraction pass.
        """
        if not getattr(self.global_config, "enable_schema_memory_graph", False):
            self.schema_memory_graph_stats = {
                "num_schema_memory_schemas": 0,
                "num_schema_memory_edges": 0,
            }
            return

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        schema_to_chunks: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        seen_pairs = set()
        for chunk_key, triples in zip(chunk_ids, chunk_triples):
            chunk_schemas = set()
            for triple in triples:
                signature = self._schema_memory_signature(tuple(triple))
                if signature is not None:
                    chunk_schemas.add(signature)
            for signature in sorted(chunk_schemas):
                pair = (signature, chunk_key)
                if pair in seen_pairs:
                    continue
                schema_to_chunks[signature].append(chunk_key)
                seen_pairs.add(pair)

        max_passages = max(2, int(getattr(self.global_config, "schema_memory_max_passages_per_schema", 80)))
        neighbors = max(1, int(getattr(self.global_config, "schema_memory_neighbors_per_passage", 2)))
        edge_weight = max(0.0, float(getattr(self.global_config, "schema_memory_edge_weight", 0.08)))
        if edge_weight <= 0:
            self.schema_memory_graph_stats = {
                "num_schema_memory_schemas": len(schema_to_chunks),
                "num_schema_memory_edges": 0,
            }
            return

        added_edges = 0
        active_schemas = 0
        for schema, chunks in schema_to_chunks.items():
            unique_chunks = list(dict.fromkeys(chunks))[:max_passages]
            if len(unique_chunks) < 2:
                continue
            active_schemas += 1
            for idx, source_chunk in enumerate(unique_chunks):
                for target_chunk in unique_chunks[idx + 1: idx + 1 + neighbors]:
                    if source_chunk == target_chunk:
                        continue
                    if source_chunk in current_graph_nodes and target_chunk in current_graph_nodes:
                        continue
                    edge = (source_chunk, target_chunk)
                    old_weight = self.node_to_node_stats.get(edge, 0.0)
                    self.node_to_node_stats[edge] = max(old_weight, edge_weight)
                    added_edges += 1

        self.schema_memory_graph_stats = {
            "num_schema_memory_schemas": active_schemas,
            "num_schema_memory_edges": added_edges,
        }
        logger.info(
            "Schema-memory graph added %d weak passage edges across %d relation schemas.",
            added_edges,
            active_schemas,
        )

    def build_query_schema_memory_index(self, all_openie_info: List[dict]):
        """
        Build a query-time relation-schema memory without modifying the graph. The
        index lets retrieval selectively pull passages that instantiate schemas
        already suggested by the current question's top facts.
        """
        schema_to_chunks = defaultdict(Counter)
        chunk_to_schemas = defaultdict(Counter)
        schema_chunk_join_keys = defaultdict(lambda: defaultdict(list))
        schema_chunk_role_join_keys = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        schema_chunk_fact_rows = defaultdict(lambda: defaultdict(list))
        schema_v2_to_chunks = defaultdict(Counter)
        schema_v2_chunk_join_keys = defaultdict(lambda: defaultdict(list))
        schema_v2_chunk_role_join_keys = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        schema_v2_chunk_fact_rows = defaultdict(lambda: defaultdict(list))
        seen_fact_rows = set()
        seen_v2_fact_rows = set()

        for doc in all_openie_info or []:
            chunk_id = doc.get("idx")
            if not chunk_id:
                continue
            triples = flatten_facts([doc.get("extracted_triples", [])])
            for triple in triples:
                if len(triple) != 3:
                    continue
                proc_triple = tuple(text_processing(list(triple)))
                signature = self._schema_memory_signature(proc_triple)
                if signature is None:
                    continue
                signatures_v2 = typed_schema_signatures_v2(proc_triple)
                schema_to_chunks[signature][chunk_id] += 1
                chunk_to_schemas[chunk_id][signature] += 1
                fact_row_key = (signature, chunk_id, proc_triple)
                if fact_row_key not in seen_fact_rows:
                    seen_fact_rows.add(fact_row_key)
                    schema_chunk_fact_rows[signature][chunk_id].append({
                        "triple": list(proc_triple),
                        "subject": str(proc_triple[0]),
                        "predicate": str(proc_triple[1]),
                        "object": str(proc_triple[2]),
                        "subject_tokens": sorted(
                            self._schema_memory_tokens(proc_triple[0])
                        ),
                        "predicate_tokens": sorted(
                            self._schema_memory_tokens(proc_triple[1])
                        ),
                        "object_tokens": sorted(
                            self._schema_memory_tokens(proc_triple[2])
                        ),
                        "subject_type": self._schema_memory_entity_type(proc_triple[0]),
                        "object_type": self._schema_memory_entity_type(proc_triple[2]),
                    })
                for role, endpoint in (
                    ("subject", proc_triple[0]),
                    ("object", proc_triple[2]),
                ):
                    join_key = tuple(sorted(self._schema_memory_tokens(endpoint)))
                    if join_key and join_key not in schema_chunk_join_keys[signature][chunk_id]:
                        schema_chunk_join_keys[signature][chunk_id].append(join_key)
                    if (
                        join_key
                        and join_key
                        not in schema_chunk_role_join_keys[signature][chunk_id][role]
                    ):
                        schema_chunk_role_join_keys[signature][chunk_id][role].append(join_key)
                for signature_v2 in signatures_v2:
                    schema_v2_to_chunks[signature_v2][chunk_id] += 1
                    v2_fact_row_key = (signature_v2, chunk_id, proc_triple)
                    if v2_fact_row_key not in seen_v2_fact_rows:
                        seen_v2_fact_rows.add(v2_fact_row_key)
                        schema_v2_chunk_fact_rows[signature_v2][chunk_id].append({
                            "triple": list(proc_triple),
                            "subject": str(proc_triple[0]),
                            "predicate": str(proc_triple[1]),
                            "object": str(proc_triple[2]),
                            "subject_tokens": sorted(
                                self._schema_memory_tokens(proc_triple[0])
                            ),
                            "predicate_tokens": sorted(
                                self._schema_memory_tokens(proc_triple[1])
                            ),
                            "object_tokens": sorted(
                                self._schema_memory_tokens(proc_triple[2])
                            ),
                            "subject_type": self._schema_memory_entity_type(proc_triple[0]),
                            "object_type": self._schema_memory_entity_type(proc_triple[2]),
                            "predicate_family": str(signature_v2[1]),
                            "direction": str(signature_v2[3]),
                        })
                    for role, endpoint in (
                        ("subject", proc_triple[0]),
                        ("object", proc_triple[2]),
                    ):
                        join_key = tuple(sorted(self._schema_memory_tokens(endpoint)))
                        if (
                            join_key
                            and join_key
                            not in schema_v2_chunk_join_keys[signature_v2][chunk_id]
                        ):
                            schema_v2_chunk_join_keys[signature_v2][chunk_id].append(join_key)
                        if (
                            join_key
                            and join_key
                            not in schema_v2_chunk_role_join_keys[signature_v2][chunk_id][role]
                        ):
                            schema_v2_chunk_role_join_keys[signature_v2][chunk_id][role].append(
                                join_key
                            )

        self.query_schema_memory_schema_to_chunks = dict(schema_to_chunks)
        self.query_schema_memory_chunk_to_schemas = dict(chunk_to_schemas)
        self.query_schema_memory_schema_chunk_join_keys = {
            signature: {
                chunk_id: list(join_keys)
                for chunk_id, join_keys in chunk_map.items()
            }
            for signature, chunk_map in schema_chunk_join_keys.items()
        }
        self.query_schema_memory_schema_chunk_role_join_keys = {
            signature: {
                chunk_id: {
                    role: list(join_keys)
                    for role, join_keys in role_map.items()
                }
                for chunk_id, role_map in chunk_map.items()
            }
            for signature, chunk_map in schema_chunk_role_join_keys.items()
        }
        self.query_schema_memory_schema_chunk_fact_rows = {
            signature: {
                chunk_id: list(rows)
                for chunk_id, rows in chunk_map.items()
            }
            for signature, chunk_map in schema_chunk_fact_rows.items()
        }
        self.query_schema_memory_v2_schema_to_chunks = dict(schema_v2_to_chunks)
        self.query_schema_memory_v2_schema_chunk_join_keys = {
            signature: {
                chunk_id: list(join_keys)
                for chunk_id, join_keys in chunk_map.items()
            }
            for signature, chunk_map in schema_v2_chunk_join_keys.items()
        }
        self.query_schema_memory_v2_schema_chunk_role_join_keys = {
            signature: {
                chunk_id: {
                    role: list(join_keys)
                    for role, join_keys in role_map.items()
                }
                for chunk_id, role_map in chunk_map.items()
            }
            for signature, chunk_map in schema_v2_chunk_role_join_keys.items()
        }
        self.query_schema_memory_v2_schema_chunk_fact_rows = {
            signature: {
                chunk_id: list(rows)
                for chunk_id, rows in chunk_map.items()
            }
            for signature, chunk_map in schema_v2_chunk_fact_rows.items()
        }
        num_passages = max(1, len(getattr(self, "passage_node_keys", [])))
        self.query_schema_memory_v2_catalog = {
            signature: {
                "row_count": int(sum(chunks.values())),
                "chunk_count": int(len(chunks)),
                "selectivity": round(len(chunks) / float(num_passages), 8),
                "idf": round(
                    float(
                        np.log((1.0 + num_passages) / (1.0 + len(chunks)))
                        / max(np.log(1.0 + num_passages), 1e-9)
                    ),
                    8,
                ),
            }
            for signature, chunks in schema_v2_to_chunks.items()
        }
        self.query_schema_memory_stats = {
            "num_query_schema_memory_schemas": len(schema_to_chunks),
            "num_query_schema_memory_chunk_entries": sum(len(v) for v in schema_to_chunks.values()),
            "num_query_schema_memory_join_keys": sum(
                len(join_keys)
                for chunk_map in schema_chunk_join_keys.values()
                for join_keys in chunk_map.values()
            ),
            "num_query_schema_memory_role_join_keys": sum(
                len(join_keys)
                for chunk_map in schema_chunk_role_join_keys.values()
                for role_map in chunk_map.values()
                for join_keys in role_map.values()
            ),
            "num_query_schema_memory_fact_rows": sum(
                len(rows)
                for chunk_map in schema_chunk_fact_rows.values()
                for rows in chunk_map.values()
            ),
            "num_query_schema_memory_v2_schemas": len(schema_v2_to_chunks),
            "num_query_schema_memory_v2_chunk_entries": sum(
                len(value) for value in schema_v2_to_chunks.values()
            ),
            "num_query_schema_memory_v2_fact_rows": sum(
                len(rows)
                for chunk_map in schema_v2_chunk_fact_rows.values()
                for rows in chunk_map.values()
            ),
        }
        logger.info("Query schema memory stats: %s", self.query_schema_memory_stats)

    def _schema_memory_tokens(self, text: str) -> Set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", str(text).lower())
            if len(token) > 2 and token not in self._SCHEMA_MEMORY_RELATION_STOPWORDS
        }

    def _query_schema_memory_bridge_strength(self, query: str) -> int:
        query_l = str(query).lower()
        strength = 2 if has_bridge_cues(query_l) else 0
        cue_terms = (
            "where", "when", "which", "whose", "that", "born", "died",
            "located", "founded", "created", "released", "published",
            "director", "author", "composer", "producer", "founder",
            "capital", "country", "city", "company", "school", "university",
        )
        strength += sum(1 for term in cue_terms if term in query_l)
        return min(strength, 6)

    def _query_schema_memory_active_signatures(self, top_k_facts: List[Tuple]) -> List[Tuple[str, str, str]]:
        signatures = []
        seen = set()
        for fact in top_k_facts or []:
            if not isinstance(fact, (list, tuple)) or len(fact) != 3:
                continue
            signature = self._schema_memory_signature(tuple(text_processing(list(fact))))
            if signature is None or signature in seen:
                continue
            seen.add(signature)
            signatures.append(signature)
        return signatures

    def _query_schema_memory_active_signatures_v2(
        self,
        top_k_facts: List[Tuple],
        query: str = "",
    ) -> List[Tuple[str, str, str, str]]:
        signatures = []
        seen = set()
        for fact in top_k_facts or []:
            if not isinstance(fact, (list, tuple)) or len(fact) != 3:
                continue
            proc_fact = tuple(text_processing(list(fact)))
            for signature in typed_schema_signatures_v2(proc_fact):
                if signature in seen:
                    continue
                seen.add(signature)
                signatures.append(signature)
        query_families = {
            family
            for family in canonical_schema_predicate_families(query)
            if not str(family).startswith("relation:")
        }
        if query_families:
            catalog = getattr(self, "query_schema_memory_v2_catalog", {}) or {}
            residual_signatures = [
                signature
                for signature in catalog
                if isinstance(signature, (list, tuple))
                and len(signature) == 4
                and str(signature[1]) in query_families
                and signature not in seen
            ]
            residual_signatures.sort(
                key=lambda signature: (
                    float((catalog.get(signature) or {}).get("selectivity", 1.0)),
                    str(signature),
                )
            )
            max_residual = max(
                1,
                int(
                    getattr(
                        self.global_config,
                        "schema_catalog_v2_max_residual_signatures",
                        24,
                    )
                ),
            )
            signatures.extend(residual_signatures[:max_residual])
        return signatures

    def _query_schema_memory_active_join_keys(
        self,
        top_k_facts: List[Tuple],
        *,
        schema_catalog_v2: bool = False,
        active_signatures: Optional[List[Tuple]] = None,
    ) -> Dict[Tuple[str, str, str], Dict[str, List[Set[str]]]]:
        join_keys = defaultdict(lambda: defaultdict(list))
        for fact in top_k_facts or []:
            if not isinstance(fact, (list, tuple)) or len(fact) != 3:
                continue
            proc_fact = tuple(text_processing(list(fact)))
            signatures = (
                typed_schema_signatures_v2(proc_fact)
                if schema_catalog_v2
                else (self._schema_memory_signature(proc_fact),)
            )
            for signature in signatures:
                if signature is None:
                    continue
                for role, endpoint in (
                    ("subject", proc_fact[0]),
                    ("object", proc_fact[2]),
                ):
                    key = set(self._schema_memory_tokens(endpoint))
                    if key and key not in join_keys[signature][role]:
                        join_keys[signature][role].append(key)
        if schema_catalog_v2 and active_signatures:
            all_endpoint_keys = []
            for role_map in join_keys.values():
                for keys in role_map.values():
                    for key in keys:
                        if key and key not in all_endpoint_keys:
                            all_endpoint_keys.append(key)
            for signature in active_signatures:
                if signature not in join_keys and all_endpoint_keys:
                    join_keys[signature]["upstream_endpoint"] = list(
                        all_endpoint_keys
                    )
        return {
            signature: {
                role: list(keys)
                for role, keys in role_map.items()
            }
            for signature, role_map in join_keys.items()
        }

    @staticmethod
    def _query_schema_memory_join_key_overlap(
        active_keys: Any,
        candidate_keys: Any,
    ) -> Dict[str, Any]:
        """Return a role-aware entity binding instead of token containment.

        The old denominator used the shorter endpoint. A generic one-token key
        could therefore obtain overlap 1.0 with an unrelated multi-token entity.
        Entity-bound joins require either an exact normalized endpoint or at
        least two shared tokens with high Jaccard similarity.
        """
        return entity_bound_join_key_match(active_keys, candidate_keys)

    def _query_schema_memory_best_fact_row(
        self,
        fact_rows: Any,
        *,
        query_tokens: Set[str],
        fact_tokens: Set[str],
        join_binding: Dict[str, Any],
        signature: Tuple[str, str, str],
    ) -> Dict[str, Any]:
        rows = list(fact_rows or [])
        if not rows:
            return {}

        context_tokens = set(query_tokens or set()) | set(fact_tokens or set())
        matched_tokens = set(join_binding.get("matched_tokens") or [])
        candidate_role = str(join_binding.get("candidate_role") or "").strip().lower()
        signature_relation_tokens = self._schema_memory_tokens(
            signature[1] if isinstance(signature, (list, tuple)) and len(signature) > 1 else ""
        )

        best_row: Dict[str, Any] = {}
        best_score = -1.0
        for row in rows:
            subject_tokens = set(row.get("subject_tokens") or [])
            predicate_tokens = set(row.get("predicate_tokens") or [])
            object_tokens = set(row.get("object_tokens") or [])
            endpoint_tokens = {
                "subject": subject_tokens,
                "object": object_tokens,
            }.get(candidate_role, subject_tokens | object_tokens)
            answer_value = (
                row.get("subject")
                if candidate_role == "object"
                else row.get("object")
            )
            answer_type = (
                row.get("subject_type")
                if candidate_role == "object"
                else row.get("object_type")
            )
            endpoint_bound = len(matched_tokens & endpoint_tokens) / max(
                1,
                len(matched_tokens) if matched_tokens else len(endpoint_tokens),
            )
            predicate_match = len(predicate_tokens & signature_relation_tokens) / max(
                1,
                len(signature_relation_tokens),
            )
            if (
                len(signature) >= 4
                and str(row.get("predicate_family") or "") == str(signature[1])
            ):
                predicate_match = 1.0
            context_match = len(
                (subject_tokens | predicate_tokens | object_tokens) & context_tokens
            ) / max(1, len(context_tokens))
            query_match = len(
                (subject_tokens | predicate_tokens | object_tokens) & set(query_tokens or set())
            ) / max(1, len(query_tokens or set()))
            score = (
                0.38 * endpoint_bound
                + 0.26 * predicate_match
                + 0.22 * context_match
                + 0.14 * query_match
            )
            if score > best_score:
                best_score = float(score)
                best_row = {
                    **row,
                    "answer_value": str(answer_value or ""),
                    "answer_type": str(answer_type or "unknown"),
                    "tuple_selection_score": round(float(score), 6),
                    "matched_endpoint_tokens": sorted(matched_tokens & endpoint_tokens),
                    "candidate_join_endpoint_role": candidate_role,
                }
        return best_row

    def query_schema_memory_retrieve(
        self,
        query: str,
        sorted_doc_ids: np.ndarray,
        sorted_doc_scores: np.ndarray,
        top_k_facts: List[Tuple],
        num_to_retrieve: int,
        fact_source: str = "top_facts",
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        info = {
            "enabled": bool(getattr(self.global_config, "enable_query_schema_memory", False)),
            "applied": False,
            "reason": "disabled",
            "fact_source": fact_source,
            "merge_mode": str(getattr(self.global_config, "query_schema_memory_merge_mode", "candidate_only")),
            "bridge_strength": 0,
            "avg_top_doc_chars": 0.0,
            "schema_count": 0,
            "active_signature_count": 0,
            "residual_signature_count": 0,
            "schema_fact_row_count": 0,
            "raw_candidate_count": 0,
            "candidate_count": 0,
            "candidate_fact_tuple_count": 0,
            "admitted_candidate_count": 0,
            "join_key_rejected_count": 0,
            "entity_binding_rejected_count": 0,
            "accepted_count": 0,
            "promoted_to_top2": 0,
            "promoted_to_top5": 0,
            "selected_schemas": [],
            "schema_catalog_version": "v1",
            "catalog_lookup_count": 0,
            "catalog_hit_count": 0,
            "catalog_estimated_rows": 0,
            "catalog_mean_selectivity": 0.0,
            "accepted": [],
            "candidate_items": [],
            "terminal_composite_probe": {
                "enabled": False,
                "reason": "not_evaluated",
                "candidate_count": 0,
            },
            "instance_bound_fact_join": {
                "enabled": False,
                "reason": "not_evaluated",
                "candidate_count": 0,
            },
            "index_access": {
                "lookup_keys": 0,
                "hit_keys": 0,
                "input_scan_rows": 0,
                "rows_after_cap": 0,
                "truncated_rows": 0,
                "fanout_rejected_keys": 0,
                "fanout_rejected_rows": 0,
                "duplicate_rows": 0,
                "deduplicated_rows": 0,
                "loaded_doc_rows": 0,
            },
            "join_key_guard_enabled": bool(
                getattr(self.global_config, "query_schema_memory_join_key_guard", True)
            ),
            "entity_bound_join_enabled": bool(
                getattr(self.global_config, "query_schema_memory_entity_bound_join", True)
            ),
            "min_join_key_overlap": float(
                getattr(self.global_config, "query_schema_memory_min_join_key_overlap", 0.50)
            ),
        }
        if not info["enabled"]:
            return sorted_doc_ids, sorted_doc_scores, info

        schema_catalog_v2 = bool(
            getattr(self.global_config, "enable_schema_catalog_v2", False)
        )
        info["schema_catalog_version"] = "v2" if schema_catalog_v2 else "v1"
        schema_to_chunks = getattr(
            self,
            "query_schema_memory_v2_schema_to_chunks"
            if schema_catalog_v2
            else "query_schema_memory_schema_to_chunks",
            None,
        )
        if not schema_to_chunks:
            info["reason"] = "no_schema_memory_index"
            return sorted_doc_ids, sorted_doc_scores, info
        info["schema_fact_row_count"] = int(
            (getattr(self, "query_schema_memory_stats", {}) or {}).get(
                "num_query_schema_memory_fact_rows",
                0,
            )
            or 0
        )

        doc_ids = sorted_doc_ids.tolist() if hasattr(sorted_doc_ids, "tolist") else list(sorted_doc_ids)
        doc_scores = sorted_doc_scores.tolist() if hasattr(sorted_doc_scores, "tolist") else list(sorted_doc_scores)
        if not doc_ids:
            info["reason"] = "no_retrieved_docs"
            return sorted_doc_ids, sorted_doc_scores, info

        bridge_strength = self._query_schema_memory_bridge_strength(query)
        info["bridge_strength"] = bridge_strength
        if getattr(self.global_config, "query_schema_memory_bridge_only", True) and not has_bridge_cues(query):
            info["reason"] = "no_bridge_cues"
            return sorted_doc_ids, sorted_doc_scores, info

        qa_top_k = max(1, int(getattr(self.global_config, "qa_top_k", 5)))
        top_doc_chars = []
        for idx in doc_ids[:qa_top_k]:
            try:
                content = self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                top_doc_chars.append(len(content or ""))
            except Exception:
                continue
        avg_top_doc_chars = float(np.mean(top_doc_chars)) if top_doc_chars else 0.0
        info["avg_top_doc_chars"] = round(avg_top_doc_chars, 2)
        if getattr(self.global_config, "query_schema_memory_disable_long_context", True):
            long_context_threshold = float(
                getattr(self.global_config, "query_schema_memory_long_context_avg_doc_chars", 1200.0)
            )
            min_long_bridge = int(
                getattr(self.global_config, "query_schema_memory_long_context_min_bridge_strength", 3)
            )
            if avg_top_doc_chars >= long_context_threshold and bridge_strength < min_long_bridge:
                info["reason"] = "long_context_gate"
                return sorted_doc_ids, sorted_doc_scores, info

        active_signatures = (
            self._query_schema_memory_active_signatures_v2(top_k_facts, query)
            if schema_catalog_v2
            else self._query_schema_memory_active_signatures(top_k_facts)
        )
        info["active_signature_count"] = len(active_signatures)
        if schema_catalog_v2:
            fact_signatures = set(
                self._query_schema_memory_active_signatures_v2(top_k_facts, "")
            )
            info["residual_signature_count"] = sum(
                signature not in fact_signatures for signature in active_signatures
            )
        if not active_signatures:
            info["reason"] = "no_active_schema"
            return sorted_doc_ids, sorted_doc_scores, info

        active_join_keys = self._query_schema_memory_active_join_keys(
            top_k_facts,
            schema_catalog_v2=schema_catalog_v2,
            active_signatures=active_signatures,
        )
        schema_chunk_join_keys = getattr(
            self,
            "query_schema_memory_v2_schema_chunk_join_keys"
            if schema_catalog_v2
            else "query_schema_memory_schema_chunk_join_keys",
            {},
        ) or {}
        schema_chunk_role_join_keys = getattr(
            self,
            "query_schema_memory_v2_schema_chunk_role_join_keys"
            if schema_catalog_v2
            else "query_schema_memory_schema_chunk_role_join_keys",
            {},
        ) or {}
        schema_chunk_fact_rows = getattr(
            self,
            "query_schema_memory_v2_schema_chunk_fact_rows"
            if schema_catalog_v2
            else "query_schema_memory_schema_chunk_fact_rows",
            {},
        ) or {}
        schema_catalog = (
            getattr(self, "query_schema_memory_v2_catalog", {}) or {}
            if schema_catalog_v2
            else {}
        )
        info["catalog_lookup_count"] = len(active_signatures)
        catalog_rows = [
            dict(schema_catalog.get(signature) or {})
            for signature in active_signatures
            if schema_catalog.get(signature)
        ]
        info["catalog_hit_count"] = len(catalog_rows)
        info["catalog_estimated_rows"] = sum(
            int(row.get("chunk_count", 0) or 0) for row in catalog_rows
        )
        if catalog_rows:
            info["catalog_mean_selectivity"] = round(
                sum(float(row.get("selectivity", 0.0) or 0.0) for row in catalog_rows)
                / float(len(catalog_rows)),
                8,
            )
        join_key_guard_enabled = bool(info["join_key_guard_enabled"])
        entity_bound_join_enabled = bool(info["entity_bound_join_enabled"])
        min_join_key_overlap = max(0.0, float(info["min_join_key_overlap"]))

        num_passages = max(1, len(getattr(self, "passage_node_keys", [])))
        max_schema_frequency = int(getattr(self.global_config, "query_schema_memory_max_schema_frequency", 120))
        max_chunks_per_schema = max(1, int(getattr(self.global_config, "query_schema_memory_max_chunks_per_schema", 32)))
        candidate_top_k = max(1, int(getattr(self.global_config, "query_schema_memory_candidate_top_k", 48)))
        min_schema_idf = float(getattr(self.global_config, "query_schema_memory_min_schema_idf", 0.18))
        min_query_overlap = float(getattr(self.global_config, "query_schema_memory_min_query_overlap", 0.06))
        min_score = float(getattr(self.global_config, "query_schema_memory_min_score", 0.52))
        protect_top_k = max(0, int(getattr(self.global_config, "query_schema_memory_protect_top_k", 1)))
        max_promotions = max(0, int(getattr(self.global_config, "query_schema_memory_max_promotions", 1)))
        merge_mode = str(
            getattr(self.global_config, "query_schema_memory_merge_mode", "candidate_only") or "candidate_only"
        ).strip().lower()
        if merge_mode not in {"candidate_only", "guarded_promote", "direct_promote"}:
            merge_mode = "candidate_only"
        info["merge_mode"] = merge_mode

        query_tokens = self._schema_memory_tokens(query)
        fact_tokens = set()
        for fact in top_k_facts or []:
            if isinstance(fact, (list, tuple)):
                fact_tokens.update(self._schema_memory_tokens(" ".join(str(part) for part in fact)))
        context_tokens = query_tokens | fact_tokens
        old_rank = {int(doc_id): rank for rank, doc_id in enumerate(doc_ids)}
        protected = {int(doc_id) for doc_id in doc_ids[:protect_top_k]}
        already_top5 = {int(doc_id) for doc_id in doc_ids[:qa_top_k]}

        schema_infos = []
        candidates = {}
        qualified_candidate_rows = 0
        index_access = info["index_access"]
        for signature in active_signatures:
            index_access["lookup_keys"] += 1
            chunks_counter = schema_to_chunks.get(signature)
            if not chunks_counter:
                continue
            index_access["hit_keys"] += 1
            schema_frequency = len(chunks_counter)
            if max_schema_frequency > 0 and schema_frequency > max_schema_frequency:
                index_access["fanout_rejected_keys"] += 1
                index_access["fanout_rejected_rows"] += schema_frequency
                continue
            scanned_rows = min(schema_frequency, max_chunks_per_schema)
            index_access["input_scan_rows"] += scanned_rows
            index_access["rows_after_cap"] += scanned_rows
            index_access["truncated_rows"] += max(
                0,
                schema_frequency - max_chunks_per_schema,
            )
            schema_idf = np.log((1.0 + num_passages) / (1.0 + schema_frequency)) / max(
                np.log(1.0 + num_passages),
                1e-9,
            )
            if schema_idf < min_schema_idf:
                continue
            max_count = max(chunks_counter.values()) if chunks_counter else 1
            schema_infos.append({
                "schema": signature,
                "frequency": schema_frequency,
                "idf": round(float(schema_idf), 4),
            })
            for chunk_id, support_count in chunks_counter.most_common(max_chunks_per_schema):
                doc_idx = getattr(self, "passage_node_key_to_idx", {}).get(chunk_id)
                if doc_idx is None or doc_idx in protected:
                    continue
                rank = old_rank.get(int(doc_idx), num_passages)
                try:
                    content = self.chunk_embedding_store.get_row(chunk_id)["content"]
                except Exception:
                    continue
                index_access["loaded_doc_rows"] += 1
                doc_tokens = self._schema_memory_tokens(content[:1600])
                if not doc_tokens:
                    continue
                context_overlap = len(context_tokens & doc_tokens) / max(len(context_tokens), 1)
                query_overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
                if max(context_overlap, query_overlap) < min_query_overlap:
                    continue
                candidate_join_keys = (
                    (schema_chunk_role_join_keys.get(signature) or {}).get(chunk_id)
                    if entity_bound_join_enabled
                    else list((schema_chunk_join_keys.get(signature) or {}).get(chunk_id) or [])
                ) or {}
                join_binding = self._query_schema_memory_join_key_overlap(
                    active_join_keys.get(signature, []),
                    candidate_join_keys,
                )
                join_key_overlap = float(
                    join_binding.get(
                        "overlap" if entity_bound_join_enabled else "containment",
                        0.0,
                    )
                )
                fact_row = self._query_schema_memory_best_fact_row(
                    (schema_chunk_fact_rows.get(signature) or {}).get(chunk_id) or [],
                    query_tokens=query_tokens,
                    fact_tokens=fact_tokens,
                    join_binding=join_binding,
                    signature=signature,
                )
                info["raw_candidate_count"] += 1
                if (
                    join_key_guard_enabled
                    and active_join_keys.get(signature)
                    and (
                        not bool(join_binding.get("bound"))
                        if entity_bound_join_enabled
                        else join_key_overlap < min_join_key_overlap
                    )
                ):
                    info["join_key_rejected_count"] += 1
                    if entity_bound_join_enabled:
                        info["entity_binding_rejected_count"] += 1
                    continue
                support = float(support_count) / max(float(max_count), 1.0)
                rank_prior = 1.0 / (1.0 + np.log1p(rank + 1))
                if active_join_keys.get(signature):
                    score = (
                        0.24 * float(schema_idf)
                        + 0.18 * support
                        + 0.20 * max(context_overlap, query_overlap)
                        + 0.23 * join_key_overlap
                        + 0.15 * rank_prior
                    )
                else:
                    score = (
                        0.30 * float(schema_idf)
                        + 0.25 * support
                        + 0.30 * max(context_overlap, query_overlap)
                        + 0.15 * rank_prior
                    )
                if fact_row:
                    score = min(
                        1.0,
                        float(score)
                        + 0.03 * float(fact_row.get("tuple_selection_score", 0.0)),
                    )
                if score < min_score:
                    continue
                qualified_candidate_rows += 1
                current = candidates.get(int(doc_idx))
                candidate = {
                    "doc_id": int(doc_idx),
                    "title": str(content).split("\n", 1)[0].strip(),
                    "score": float(score),
                    "original_rank": int(rank),
                    "schema": signature,
                    "schema_frequency": int(schema_frequency),
                    "schema_idf": float(schema_idf),
                    "schema_support": support,
                    "context_overlap": float(context_overlap),
                    "query_overlap": float(query_overlap),
                    "join_key_overlap": float(join_key_overlap),
                    "join_key_containment": float(join_binding.get("containment", 0.0)),
                    "join_key_exact": bool(join_binding.get("exact")),
                    "entity_bound": bool(join_binding.get("bound")),
                    "active_join_role": str(join_binding.get("active_role") or ""),
                    "candidate_join_role": str(join_binding.get("candidate_role") or ""),
                    "active_join_key_tokens": list(
                        join_binding.get("active_key_tokens") or []
                    ),
                    "candidate_join_key_tokens": list(
                        join_binding.get("candidate_key_tokens") or []
                    ),
                    "matched_join_tokens": list(join_binding.get("matched_tokens") or []),
                    "join_key_matched": bool(
                        join_binding.get("bound")
                        if entity_bound_join_enabled
                        else join_key_overlap >= min_join_key_overlap
                    ),
                    "already_top5": int(doc_idx) in already_top5,
                    "schema_fact_tuple": bool(fact_row),
                    "triple": list(fact_row.get("triple") or []),
                    "predicate": str(fact_row.get("predicate") or signature[1]),
                    "answer_value": str(fact_row.get("answer_value") or ""),
                    "answer_type": str(fact_row.get("answer_type") or "unknown"),
                    "tuple_selection_score": float(
                        fact_row.get("tuple_selection_score", 0.0)
                    ),
                    "matched_endpoint_tokens": list(
                        fact_row.get("matched_endpoint_tokens") or []
                    ),
                    "candidate_join_endpoint_role": str(
                        fact_row.get("candidate_join_endpoint_role") or ""
                    ),
                    "direction": "object_to_subject"
                    if str(fact_row.get("candidate_join_endpoint_role") or "").lower() == "object"
                    else "subject_to_object",
                    "materialization_role": "schema_memory_fact_tuple"
                    if fact_row
                    else "",
                }
                if current is None or candidate["score"] > current["score"]:
                    candidates[int(doc_idx)] = candidate

        terminal_probe = {
            "enabled": False,
            "reason": "schema_critical_reader_visible_disabled",
            "candidate_count": 0,
            "candidates": [],
        }
        if getattr(
            self.global_config,
            "enable_schema_critical_reader_visible",
            False,
        ):
            instance_completion_enabled = bool(
                getattr(
                    self.global_config,
                    "enable_bound_completion_fact_join",
                    False,
                )
            )

            def instance_bound_completion_lookup(
                completion_query: str,
                top_k: int,
            ) -> List[Dict[str, Any]]:
                """Retrieve a bounded evidence pool for one missing plan slot."""
                limit = max(1, int(top_k))
                documents: List[Dict[str, Any]] = []
                seen_doc_ids = set()

                def append_document(
                    raw_doc_id: Any,
                    *,
                    rank: int,
                    score: float,
                    source: str,
                ) -> None:
                    try:
                        doc_id = int(raw_doc_id)
                        content = str(
                            self.chunk_embedding_store.get_row(
                                self.passage_node_keys[doc_id]
                            )["content"]
                            or ""
                        )
                    except Exception:
                        return
                    if doc_id in seen_doc_ids or not content:
                        return
                    seen_doc_ids.add(doc_id)
                    documents.append({
                        "doc_id": doc_id,
                        "rank": int(rank),
                        "score": float(score),
                        "title": content.splitlines()[0].strip(),
                        "content": content,
                        "completion_source": source,
                    })

                for rank, raw_doc_id in enumerate(doc_ids[: min(3, limit)]):
                    append_document(
                        raw_doc_id,
                        rank=rank,
                        score=float(doc_scores[rank]),
                        source="current_evidence_pool",
                    )
                try:
                    dense_ids, dense_scores = self.dense_passage_retrieval(
                        completion_query
                    )
                except Exception as exc:
                    logger.warning(
                        "Instance-bound completion retrieval failed for %r: %s",
                        completion_query,
                        exc,
                    )
                    dense_ids, dense_scores = [], []
                for rank, raw_doc_id in enumerate(list(dense_ids)):
                    if len(documents) >= limit:
                        break
                    score = (
                        float(dense_scores[rank])
                        if rank < len(dense_scores) else 0.0
                    )
                    append_document(
                        raw_doc_id,
                        rank=rank,
                        score=score,
                        source="dense_completion_probe",
                    )
                return documents

            terminal_probe = probe_terminal_schema_candidates(
                question=query,
                top_k_facts=top_k_facts,
                relation_index=getattr(self, "relation_fact_join_index", {}) or {},
                rank_by_doc=old_rank,
                top_doc_ids=doc_ids[:qa_top_k],
                num_passages=num_passages,
                get_doc_content=lambda doc_idx: self.chunk_embedding_store.get_row(
                    self.passage_node_keys[int(doc_idx)]
                )["content"],
                max_rows_per_lookup=max(1, min(max_chunks_per_schema, 24)),
                max_candidates=candidate_top_k,
            )
            instance_bound_execution = execute_instance_bound_fact_join_plan(
                question=query,
                relation_index=getattr(self, "relation_fact_join_index", {}) or {},
                rank_by_doc=old_rank,
                get_doc_content=lambda doc_idx: self.chunk_embedding_store.get_row(
                    self.passage_node_keys[int(doc_idx)]
                )["content"],
                max_facts_per_key=max(1, min(max_chunks_per_schema, 24)),
                max_key_fanout=max(64, max_chunks_per_schema * 4),
                max_candidates=candidate_top_k,
                enable_bound_completion=instance_completion_enabled,
                bound_completion_lookup=(
                    instance_bound_completion_lookup
                    if instance_completion_enabled else None
                ),
                bound_completion_top_k=int(
                    getattr(self.global_config, "bound_completion_top_k", 6)
                ),
                bound_completion_max_calls=int(
                    getattr(self.global_config, "bound_completion_max_calls", 2)
                ),
                enable_source_grounded_terminal_repair=bool(
                    getattr(
                        self.global_config,
                        "enable_source_grounded_terminal_repair",
                        False,
                    )
                ),
            )
        else:
            instance_bound_execution = {
                "enabled": False,
                "reason": "schema_critical_reader_visible_disabled",
                "candidate_count": 0,
                "candidates": [],
            }
        info["terminal_composite_probe"] = {
            key: value
            for key, value in terminal_probe.items()
            if key != "candidates"
        }




        info["instance_bound_fact_join"] = {
            key: value
            for key, value in instance_bound_execution.items()
            if key != "candidates"
        }
        info["instance_bound_fact_join"]["candidate_summaries"] = [
            {
                "doc_id": int(item.get("doc_id", -1)),
                "original_rank": int(item.get("original_rank", num_passages)),
                "answer_value": str(item.get("answer_value") or ""),
                "materialization_role": str(
                    item.get("materialization_role") or ""
                ),
                "operator_complete": bool(item.get("operator_complete")),
                "strict_operator_contract_validated": bool(
                    item.get("strict_operator_contract_validated")
                ),
                "root_plan_variant": str(
                    item.get("root_plan_variant") or ""
                ),
                "root_binding_method": str(
                    item.get("root_binding_method") or ""
                ),
                "root_binding_confidence": float(
                    item.get("root_binding_confidence", 0.0) or 0.0
                ),
                "root_identity_safe_for_commit": bool(
                    item.get("root_identity_safe_for_commit", True)
                ),
                "bound_completion_used": bool(
                    item.get("bound_completion_used")
                ),
                "completion_causal_lineage_validated": bool(
                    item.get("completion_causal_lineage_validated")
                ),
                "source_grounded_completion_used": bool(
                    item.get("source_grounded_completion_used")
                ),
                "source_grounded_terminal_repair_used": bool(
                    item.get("source_grounded_terminal_repair_used")
                ),
                "bound_completion_lineage": list(
                    item.get("bound_completion_lineage") or []
                ),
                "bound_completion_doc_ids": list(
                    item.get("bound_completion_doc_ids") or []
                ),
                "rooted_lineage_complete": bool(
                    item.get("rooted_lineage_complete")
                ),
                "lineage_root_key": str(item.get("lineage_root_key") or ""),
                "lineage_bound_keys": list(
                    item.get("lineage_bound_keys") or []
                ),
                "lineage_depth": int(item.get("lineage_depth", 0) or 0),
                "lineage": list(item.get("lineage") or []),
                "lineage_source_doc_ids": list(
                    item.get("lineage_source_doc_ids") or []
                ),
            }
            for item in (instance_bound_execution.get("candidates") or [])[:16]
        ]
        for terminal_candidate in instance_bound_execution.get("candidates") or []:
            doc_idx = int(terminal_candidate.get("doc_id", -1))
            if doc_idx < 0 or doc_idx in protected:
                continue
            current = candidates.get(doc_idx)
            if (
                current is None
                or float(terminal_candidate.get("score", 0.0))
                > float(current.get("score", 0.0))
            ):
                candidates[doc_idx] = dict(terminal_candidate)

        index_access["deduplicated_rows"] = len(candidates)
        index_access["duplicate_rows"] = max(
            0,
            qualified_candidate_rows - len(candidates),
        )
        info["schema_count"] = len(schema_infos)
        info["selected_schemas"] = schema_infos[:8]
        ranked_candidates = sorted(
            candidates.values(),
            key=lambda item: (item["score"], -item["original_rank"]),
            reverse=True,
        )[:candidate_top_k]
        info["candidate_count"] = len(ranked_candidates)
        info["candidate_fact_tuple_count"] = sum(
            1 for item in ranked_candidates if item.get("schema_fact_tuple")
        )
        info["admitted_candidate_count"] = len(ranked_candidates)
        info["candidate_items"] = [
            {
                "doc_id": int(item["doc_id"]),
                "title": item["title"],
                "score": round(float(item["score"]), 6),
                "original_rank": int(item["original_rank"]),
                "schema": list(item["schema"]) if isinstance(item.get("schema"), tuple) else item.get("schema"),
                "schema_frequency": int(item["schema_frequency"]),
                "schema_idf": round(float(item["schema_idf"]), 6),
                "schema_support": round(float(item["schema_support"]), 6),
                "context_overlap": round(float(item["context_overlap"]), 6),
                "query_overlap": round(float(item["query_overlap"]), 6),
                "join_key_overlap": round(float(item["join_key_overlap"]), 6),
                "join_key_containment": round(float(item["join_key_containment"]), 6),
                "join_key_exact": bool(item["join_key_exact"]),
                "entity_bound": bool(item["entity_bound"]),
                "active_join_role": item["active_join_role"],
                "candidate_join_role": item["candidate_join_role"],
                "active_join_key_tokens": list(item["active_join_key_tokens"]),
                "candidate_join_key_tokens": list(item["candidate_join_key_tokens"]),
                "matched_join_tokens": list(item["matched_join_tokens"]),
                "join_key_matched": bool(item["join_key_matched"]),
                "already_top5": bool(item["already_top5"]),
                "schema_fact_tuple": bool(item.get("schema_fact_tuple")),
                "triple": list(item.get("triple") or []),
                "predicate": str(item.get("predicate") or ""),
                "answer_value": str(item.get("answer_value") or ""),
                "answer_type": str(item.get("answer_type") or "unknown"),
                "tuple_selection_score": round(
                    float(item.get("tuple_selection_score", 0.0)),
                    6,
                ),
                "matched_endpoint_tokens": list(
                    item.get("matched_endpoint_tokens") or []
                ),
                "candidate_join_endpoint_role": str(
                    item.get("candidate_join_endpoint_role") or ""
                ),
                "direction": str(item.get("direction") or ""),
                "materialization_role": str(item.get("materialization_role") or ""),
                "logical_plan_stage": str(item.get("logical_plan_stage") or ""),
                "terminal_operator_id": str(item.get("terminal_operator_id") or ""),
                "predicate_contract_match": bool(
                    item.get("predicate_contract_match")
                ),
                "terminal_slot_match": bool(item.get("terminal_slot_match")),
                "answer_type_match": bool(item.get("answer_type_match")),
                "rooted_lineage_complete": bool(
                    item.get("rooted_lineage_complete")
                ),
                "covered_slot_ids": list(item.get("covered_slot_ids") or []),
                "uncovered_slot_ids": list(item.get("uncovered_slot_ids") or []),
                "physical_access_path": str(item.get("physical_access_path") or ""),
                "estimated_endpoint_scan_rows": int(
                    item.get("estimated_endpoint_scan_rows", 1) or 1
                ),
                "operator_complete": bool(item.get("operator_complete")),
                "strict_operator_contract_validated": bool(
                    item.get("strict_operator_contract_validated")
                ),
                "root_plan_variant": str(
                    item.get("root_plan_variant") or ""
                ),
                "root_binding_method": str(
                    item.get("root_binding_method") or ""
                ),
                "root_binding_confidence": round(
                    float(item.get("root_binding_confidence", 0.0) or 0.0),
                    6,
                ),
                "root_identity_safe_for_commit": bool(
                    item.get("root_identity_safe_for_commit", True)
                ),
                "bound_completion_used": bool(
                    item.get("bound_completion_used")
                ),
                "completion_causal_lineage_validated": bool(
                    item.get("completion_causal_lineage_validated")
                ),
                "source_grounded_completion_used": bool(
                    item.get("source_grounded_completion_used")
                ),
                "source_grounded_terminal_repair_used": bool(
                    item.get("source_grounded_terminal_repair_used")
                ),
                "bound_completion_lineage": list(
                    item.get("bound_completion_lineage") or []
                ),
                "bound_completion_doc_ids": list(
                    item.get("bound_completion_doc_ids") or []
                ),
                "lineage_root_key": str(item.get("lineage_root_key") or ""),
                "lineage_bound_keys": list(
                    item.get("lineage_bound_keys") or []
                ),
                "lineage_depth": int(item.get("lineage_depth", 0) or 0),
                "lineage": list(item.get("lineage") or []),
                "lineage_source_doc_ids": list(
                    item.get("lineage_source_doc_ids") or []
                ),
                "lineage_reader_text": str(
                    item.get("lineage_reader_text") or ""
                ),
                "slot_bindings": dict(item.get("slot_bindings") or {}),
            }
            for item in ranked_candidates[:candidate_top_k]
        ]
        if not ranked_candidates:
            info["reason"] = "no_accepted_candidates"
            return sorted_doc_ids, sorted_doc_scores, info
        if merge_mode == "candidate_only":
            info["reason"] = "candidates_generated"
            return sorted_doc_ids, sorted_doc_scores, info
        if max_promotions <= 0:
            info["reason"] = "promotion_disabled"
            return sorted_doc_ids, sorted_doc_scores, info

        accepted = ranked_candidates[:max_promotions]
        accepted_ids = [item["doc_id"] for item in accepted]
        max_base_score = float(max(doc_scores)) if doc_scores else 1.0
        score_by_id = {int(doc_id): float(score) for doc_id, score in zip(doc_ids, doc_scores)}
        for rank_offset, item in enumerate(accepted):
            score_by_id[item["doc_id"]] = max(
                score_by_id.get(item["doc_id"], 0.0),
                max_base_score * (0.985 - 0.005 * rank_offset),
            )

        accepted_set = set(accepted_ids)
        protected_prefix = doc_ids[:protect_top_k]
        suffix = [doc_id for doc_id in doc_ids[protect_top_k:] if int(doc_id) not in accepted_set]
        new_doc_ids = protected_prefix + accepted_ids + suffix
        new_doc_scores = [score_by_id.get(int(doc_id), 0.0) for doc_id in new_doc_ids]
        new_rank = {int(doc_id): rank for rank, doc_id in enumerate(new_doc_ids)}

        promoted_to_top2 = sum(
            1 for item in accepted
            if new_rank.get(item["doc_id"], num_passages) < 2
            and old_rank.get(item["doc_id"], num_passages) >= 2
        )
        promoted_to_top5 = sum(
            1 for item in accepted
            if new_rank.get(item["doc_id"], num_passages) < 5
            and old_rank.get(item["doc_id"], num_passages) >= 5
        )
        info.update({
            "applied": True,
            "reason": "applied",
            "accepted_count": len(accepted),
            "promoted_to_top2": promoted_to_top2,
            "promoted_to_top5": promoted_to_top5,
            "accepted": [
                {
                    "title": item["title"],
                    "score": round(item["score"], 4),
                    "original_rank": item["original_rank"],
                    "new_rank": new_rank.get(item["doc_id"]),
                    "schema_frequency": item["schema_frequency"],
                    "schema_idf": round(item["schema_idf"], 4),
                    "context_overlap": round(item["context_overlap"], 4),
                    "query_overlap": round(item["query_overlap"], 4),
                    "already_top5": item["already_top5"],
                }
                for item in accepted
            ],
        })
        return np.array(new_doc_ids), np.array(new_doc_scores), info

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)


        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """


        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            try:
                openie_results = load_locked_json(self.openie_results_path)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "OpenIE cache is not valid JSON and must be restored before indexing: "
                    f"path={self.openie_results_path} byte_offset={exc.pos}"
                ) from exc
            all_openie_info = openie_results.get('docs', [])



            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            try:
                chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_results_dict[chunk_key].triples}
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_key}: {e}")
                chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': [],
                                 'extracted_triples': []}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:

            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0

            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }

            atomic_write_json(self.openie_results_path, openie_dict)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph["s2rag_graph_cache_signature"] = (
            self._encoded_graph_cache_config_signature()
        )
        self.graph["s2rag_graph_cache_config"] = json.dumps(
            self._graph_cache_config_signature(),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}


        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))


        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))


        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]


        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node
        schema_memory_stats = getattr(self, "schema_memory_graph_stats", None)
        if schema_memory_stats is not None:
            graph_info.update(schema_memory_stats)


        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids())
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids())
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())
        self.passage_node_key_to_idx = {key: idx for idx, key in enumerate(self.passage_node_keys)}

        if self._uses_dense_retrieval_backend():
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []
            logger.info(
                "nv_embed_dense preparation skips all graph-node consistency and mapping work."
            )
        else:

            expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
            actual_node_count = self.graph.vcount()

            if expected_node_count != actual_node_count:
                logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")

                if actual_node_count == 0 and expected_node_count > 0:
                    logger.info(f"Initializing graph with {expected_node_count} nodes")
                    self.add_new_nodes()
                    self.save_igraph()


            try:
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx


                missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
                missing_passage_nodes = [node_key for node_key in self.passage_node_keys if node_key not in igraph_name_to_idx]

                if missing_entity_nodes or missing_passage_nodes:
                    logger.warning(f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_passage_nodes)} passage nodes")

                    self.add_new_nodes()
                    self.save_igraph()

                    igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                    self.node_name_to_vertex_idx = igraph_name_to_idx

                self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys]
                self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys]
            except Exception as e:
                logger.error(f"Error creating node index mapping: {str(e)}")

                self.node_name_to_vertex_idx = {}
                self.entity_node_idxs = []
                self.passage_node_idxs = []

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        self.chunk_path_info = build_chunk_path_info(all_openie_info)
        if getattr(self.global_config, "enable_query_schema_memory", False):
            self.build_query_schema_memory_index(all_openie_info)
        else:
            self.query_schema_memory_schema_to_chunks = {}
            self.query_schema_memory_chunk_to_schemas = {}
            self.query_schema_memory_schema_chunk_join_keys = {}
            self.query_schema_memory_schema_chunk_role_join_keys = {}
            self.query_schema_memory_schema_chunk_fact_rows = {}
            self.query_schema_memory_v2_schema_to_chunks = {}
            self.query_schema_memory_v2_schema_chunk_join_keys = {}
            self.query_schema_memory_v2_schema_chunk_role_join_keys = {}
            self.query_schema_memory_v2_schema_chunk_fact_rows = {}
            self.query_schema_memory_v2_catalog = {}
            self.query_schema_memory_stats = {
                "num_query_schema_memory_schemas": 0,
                "num_query_schema_memory_chunk_entries": 0,
                "num_query_schema_memory_join_keys": 0,
                "num_query_schema_memory_role_join_keys": 0,
                "num_query_schema_memory_fact_rows": 0,
                "num_query_schema_memory_v2_schemas": 0,
                "num_query_schema_memory_v2_chunk_entries": 0,
                "num_query_schema_memory_v2_fact_rows": 0,
            }

        if (
            getattr(self.global_config, "enable_relation_fact_join_retrieval", False)
            or getattr(self.global_config, "enable_entity_predicate_fact_join", False)
            or getattr(
                self.global_config,
                "enable_query_schema_memory_cross_schema_join",
                False,
            )
        ):
            self.relation_fact_join_index = build_relation_fact_join_index(
                all_openie_info,
                self.passage_node_key_to_idx,
            )
            logger.info(
                "Relation fact join index stats: %s",
                (self.relation_fact_join_index or {}).get("stats", {}),
            )
        else:
            self.relation_fact_join_index = {
                "items": [],
                "entity_to_items": {},
                "relation_to_items": {},
                "schema_to_items": {},
                "doc_to_items": {},
                "stats": {
                    "num_fact_items": 0,
                    "num_entity_keys": 0,
                    "num_relation_keys": 0,
                    "num_schema_keys": 0,
                    "num_doc_entries": 0,
                },
            }

        self.proc_triples_to_docs = {}

        for doc in all_openie_info:
            triples = flatten_facts([doc['extracted_triples']])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([doc['idx']]))

        if self.ent_node_to_chunk_ids is None or self._uses_dense_retrieval_backend():
            ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)


            if not (len(self.passage_node_keys) == len(ner_results_dict) == len(triple_results_dict)):
                logger.warning(f"Length mismatch: passage_node_keys={len(self.passage_node_keys)}, ner_results_dict={len(ner_results_dict)}, triple_results_dict={len(triple_results_dict)}")


                for chunk_id in self.passage_node_keys:
                    if chunk_id not in ner_results_dict:
                        ner_results_dict[chunk_id] = NerRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            unique_entities=[]
                        )
                    if chunk_id not in triple_results_dict:
                        triple_results_dict[chunk_id] = TripleRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            triples=[]
                        )


            chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in self.passage_node_keys]

            if self._uses_dense_retrieval_backend():
                self._rebuild_fact_lookup_metadata(
                    self.passage_node_keys,
                    chunk_triples,
                )
            else:
                self.node_to_node_stats = {}
                self.ent_node_to_chunk_ids = {}
                self.add_fact_edges(self.passage_node_keys, chunk_triples)

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)


        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])

        try:
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T)
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores


    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """

        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])


        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """

        if self._uses_dense_retrieval_backend():
            logger.info(
                "Blocked graph_search_with_fact_entities under nv_embed_dense; "
                "using dense passage retrieval instead."
            )
            return self.dense_passage_retrieval(query)


        linking_score_map = {}
        phrase_scores = {}
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        number_of_occurs = np.zeros(len(self.graph.vs['name']))

        phrases_and_ids = set()

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    weighted_fact_score = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weighted_fact_score /= len(self.ent_node_to_chunk_ids[phrase_key])

                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1

                phrases_and_ids.add((phrase, phrase_id))

        occurred_phrase_mask = number_of_occurs > 0
        phrase_weights[occurred_phrase_mask] = (
            phrase_weights[occurred_phrase_mask] / number_of_occurs[occurred_phrase_mask]
        )

        for phrase, phrase_id in phrases_and_ids:
            if phrase_id is None:
                continue
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []

            phrase_scores[phrase].append(float(phrase_weights[phrase_id]))


        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)


        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight


        node_weights = phrase_weights + passage_weights


        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'


        ppr_start = time.time()
        self.ppr_call_count += 1
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()

        self.ppr_time += (ppr_end - ppr_start)

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores


    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """

        link_top_k: int = self.global_config.linking_top_k
        if bool(getattr(self.global_config, "enable_fact_source_scan", False)):
            link_top_k = max(
                link_top_k,
                int(getattr(
                    self.global_config,
                    "fact_source_scan_fact_top_k",
                    link_top_k,
                )),
            )


        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}

        try:

            if len(query_fact_scores) <= link_top_k:

                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:

                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()


            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]


            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                                candidate_facts,
                                                                                candidate_fact_indices,
                                                                                len_after_rerank=link_top_k)

            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}

            return top_k_fact_indices, top_k_facts, rerank_log

        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}

    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float =0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None: damping = 0.5
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
