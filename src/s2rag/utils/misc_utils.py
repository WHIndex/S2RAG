from argparse import ArgumentTypeError
from dataclasses import dataclass
from hashlib import md5
from typing import Dict, Any, List, Tuple, Literal, Union, Optional
import numpy as np
import re
import logging

from .typing import Triple
from .llm_utils import filter_invalid_triples

logger = logging.getLogger(__name__)

@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]

@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal['node', 'dpr']

@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    retrieved_doc_ids: Optional[List[int]] = None
    doc_scores: np.ndarray = None
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None
    answer_style: str = None
    answer_style_instruction: str = None
    has_answer_marker: bool = None
    qa_docs: Optional[List[str]] = None
    qa_source_docs: Optional[List[str]] = None
    evidence_selection: Optional[Dict[str, Any]] = None
    schema_evidence_set: Optional[Dict[str, Any]] = None
    evidence_admission: Optional[Dict[str, Any]] = None
    final_reader_counterfactual: Optional[Dict[str, Any]] = None
    bridge_contrastive_evidence: Optional[Dict[str, Any]] = None
    sentence_evidence: Optional[Dict[str, Any]] = None
    anchored_typed_candidate_table: Optional[Dict[str, Any]] = None
    entity_state_evidence: Optional[Dict[str, Any]] = None
    slot_projected_provenance: Optional[Dict[str, Any]] = None
    slot_separated_fact_join: Optional[Dict[str, Any]] = None
    entity_centric_fallback: Optional[Dict[str, Any]] = None
    answer_slot_support_completion: Optional[Dict[str, Any]] = None
    candidate_answer_selector: Optional[Dict[str, Any]] = None
    answer_before_canonicalization: str = None
    answer_before_verifier: str = None
    answer_before_consistent_reader: str = None
    answer_before_object_arbitration: str = None
    answer_before_evidence_answer_arbiter: str = None
    answer_before_typed_evidence_table: str = None
    answer_before_contrastive_arbitration: str = None
    answer_before_provenance_lattice: str = None
    answer_before_provenance_typed_candidate_store: str = None
    answer_before_evidence_grounded_extractor: str = None
    answer_before_query_answer_contract: str = None
    answer_before_provenance_answer_slot_store: str = None
    answer_before_answer_slot_binding_optimizer: str = None
    answer_before_answer_surface_lattice_optimizer: str = None
    answer_before_schema_surface_canonicalization: str = None
    answer_before_plan_consistent_answer_verifier: str = None
    answer_before_cost_gated_reread: str = None
    answer_before_provenance_bound_materializer: str = None
    canonicalization: Optional[Dict[str, Any]] = None
    answer_verifier: Optional[Dict[str, Any]] = None
    evidence_consistent_reader: Optional[Dict[str, Any]] = None
    evidence_object_arbitration: Optional[Dict[str, Any]] = None
    evidence_answer_arbiter: Optional[Dict[str, Any]] = None
    typed_evidence_table: Optional[Dict[str, Any]] = None
    contrastive_answer_arbitration: Optional[Dict[str, Any]] = None
    provenance_answer_lattice: Optional[Dict[str, Any]] = None
    provenance_typed_candidate_store: Optional[Dict[str, Any]] = None
    evidence_grounded_answer_extractor: Optional[Dict[str, Any]] = None
    query_answer_contract: Optional[Dict[str, Any]] = None
    provenance_answer_slot_store: Optional[Dict[str, Any]] = None
    answer_slot_binding_optimizer: Optional[Dict[str, Any]] = None
    answer_surface_lattice_optimizer: Optional[Dict[str, Any]] = None
    schema_surface_canonicalization: Optional[Dict[str, Any]] = None
    plan_consistent_answer_verifier: Optional[Dict[str, Any]] = None
    cost_gated_constrained_reread: Optional[Dict[str, Any]] = None
    provenance_bound_answer_materializer: Optional[Dict[str, Any]] = None
    fact_rerank: Optional[Dict[str, Any]] = None
    fact_source_scan: Optional[Dict[str, Any]] = None
    retrieval_backend: Optional[Dict[str, Any]] = None
    graph_repair: Optional[Dict[str, Any]] = None
    query_schema_memory: Optional[Dict[str, Any]] = None
    schema_critical_evidence: Optional[Dict[str, Any]] = None
    logical_query_plan: Optional[Dict[str, Any]] = None
    adaptive_query_execution: Optional[Dict[str, Any]] = None
    paper_evidence_execution: Optional[Dict[str, Any]] = None
    entity_predicate_fact_join: Optional[Dict[str, Any]] = None
    relation_fact_join: Optional[Dict[str, Any]] = None
    typed_fact_join_shadow: Optional[Dict[str, Any]] = None
    selective_support_join: Optional[Dict[str, Any]] = None
    selective_support_join_docs: Optional[List[str]] = None
    relation_fact_join_docs: Optional[List[str]] = None
    support_rerank: Optional[Dict[str, Any]] = None
    dual_channel_projection: Optional[Dict[str, Any]] = None
    dual_channel_projection_docs: Optional[List[str]] = None
    chain_rerank: Optional[Dict[str, Any]] = None
    planned_retrieval: Optional[Dict[str, Any]] = None
    iterative_retrieval: Optional[Dict[str, Any]] = None
    path_rerank: Optional[Dict[str, Any]] = None
    path_evidence: Optional[Dict[str, Any]] = None
    path_evidence_docs: Optional[List[str]] = None
    planned_evidence_docs: Optional[List[str]] = None
    support_set_completion: Optional[Dict[str, Any]] = None
    structured_evidence_chain: Optional[Dict[str, Any]] = None
    multigranular_memory: Optional[Dict[str, Any]] = None
    qa_timing: Optional[Dict[str, Any]] = None
    sample_key: Optional[str] = None


    def to_dict(self):
        return {
            "question": self.question,
            "sample_key": self.sample_key,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "retrieved_doc_ids": self.retrieved_doc_ids[:20] if self.retrieved_doc_ids else None,
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()[:5]]  if self.doc_scores is not None else None,
            "gold_docs": self.gold_docs,
            "answer_style": self.answer_style,
            "has_answer_marker": self.has_answer_marker,
            "qa_docs": self.qa_docs[:5] if self.qa_docs else None,
            "qa_source_docs": self.qa_source_docs[:5] if self.qa_source_docs else None,
            "evidence_selection": self.evidence_selection,
            "schema_evidence_set": self.schema_evidence_set,
            "evidence_admission": self.evidence_admission,
            "final_reader_counterfactual": self.final_reader_counterfactual,
            "bridge_contrastive_evidence": self.bridge_contrastive_evidence,
            "sentence_evidence": self.sentence_evidence,
            "anchored_typed_candidate_table": self.anchored_typed_candidate_table,
            "entity_state_evidence": self.entity_state_evidence,
            "slot_projected_provenance": self.slot_projected_provenance,
            "slot_separated_fact_join": self.slot_separated_fact_join,
            "entity_centric_fallback": self.entity_centric_fallback,
            "answer_slot_support_completion": self.answer_slot_support_completion,
            "candidate_answer_selector": self.candidate_answer_selector,
            "answer_before_canonicalization": self.answer_before_canonicalization,
            "answer_before_verifier": self.answer_before_verifier,
            "answer_before_consistent_reader": self.answer_before_consistent_reader,
            "answer_before_object_arbitration": self.answer_before_object_arbitration,
            "answer_before_evidence_answer_arbiter": self.answer_before_evidence_answer_arbiter,
            "answer_before_typed_evidence_table": self.answer_before_typed_evidence_table,
            "answer_before_contrastive_arbitration": self.answer_before_contrastive_arbitration,
            "answer_before_provenance_lattice": self.answer_before_provenance_lattice,
            "answer_before_provenance_typed_candidate_store": self.answer_before_provenance_typed_candidate_store,
            "answer_before_evidence_grounded_extractor": self.answer_before_evidence_grounded_extractor,
            "answer_before_query_answer_contract": self.answer_before_query_answer_contract,
            "answer_before_provenance_answer_slot_store": self.answer_before_provenance_answer_slot_store,
            "answer_before_answer_slot_binding_optimizer": self.answer_before_answer_slot_binding_optimizer,
            "answer_before_answer_surface_lattice_optimizer": self.answer_before_answer_surface_lattice_optimizer,
            "answer_before_schema_surface_canonicalization": self.answer_before_schema_surface_canonicalization,
            "answer_before_plan_consistent_answer_verifier": self.answer_before_plan_consistent_answer_verifier,
            "answer_before_cost_gated_reread": self.answer_before_cost_gated_reread,
            "answer_before_provenance_bound_materializer": self.answer_before_provenance_bound_materializer,
            "canonicalization": self.canonicalization,
            "answer_verifier": self.answer_verifier,
            "evidence_consistent_reader": self.evidence_consistent_reader,
            "evidence_object_arbitration": self.evidence_object_arbitration,
            "evidence_answer_arbiter": self.evidence_answer_arbiter,
            "typed_evidence_table": self.typed_evidence_table,
            "contrastive_answer_arbitration": self.contrastive_answer_arbitration,
            "provenance_answer_lattice": self.provenance_answer_lattice,
            "provenance_typed_candidate_store": self.provenance_typed_candidate_store,
            "evidence_grounded_answer_extractor": self.evidence_grounded_answer_extractor,
            "query_answer_contract": self.query_answer_contract,
            "provenance_answer_slot_store": self.provenance_answer_slot_store,
            "answer_slot_binding_optimizer": self.answer_slot_binding_optimizer,
            "answer_surface_lattice_optimizer": self.answer_surface_lattice_optimizer,
            "schema_surface_canonicalization": self.schema_surface_canonicalization,
            "plan_consistent_answer_verifier": self.plan_consistent_answer_verifier,
            "cost_gated_constrained_reread": self.cost_gated_constrained_reread,
            "provenance_bound_answer_materializer": self.provenance_bound_answer_materializer,
            "fact_rerank": self.fact_rerank,
            "fact_source_scan": self.fact_source_scan,
            "retrieval_backend": self.retrieval_backend,
            "graph_repair": self.graph_repair,
            "query_schema_memory": self.query_schema_memory,
            "schema_critical_evidence": self.schema_critical_evidence,
            "logical_query_plan": self.logical_query_plan,
            "adaptive_query_execution": self.adaptive_query_execution,
            "paper_evidence_execution": self.paper_evidence_execution,
            "entity_predicate_fact_join": self.entity_predicate_fact_join,
            "relation_fact_join": self.relation_fact_join,
            "typed_fact_join_shadow": self.typed_fact_join_shadow,
            "selective_support_join": self.selective_support_join,
            "selective_support_join_docs": self.selective_support_join_docs[:3] if self.selective_support_join_docs else None,
            "relation_fact_join_docs": self.relation_fact_join_docs[:3] if self.relation_fact_join_docs else None,
            "support_rerank": self.support_rerank,
            "dual_channel_projection": self.dual_channel_projection,
            "dual_channel_projection_docs": self.dual_channel_projection_docs[:3] if self.dual_channel_projection_docs else None,
            "chain_rerank": self.chain_rerank,
            "planned_retrieval": self.planned_retrieval,
            "iterative_retrieval": self.iterative_retrieval,
            "path_rerank": self.path_rerank,
            "path_evidence": self.path_evidence,
            "path_evidence_docs": self.path_evidence_docs[:3] if self.path_evidence_docs else None,
            "planned_evidence_docs": self.planned_evidence_docs[:3] if self.planned_evidence_docs else None,
            "support_set_completion": self.support_set_completion,
            "structured_evidence_chain": self.structured_evidence_chain,
            "multigranular_memory": self.multigranular_memory,
            "qa_timing": self.qa_timing,
        }

def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()

def reformat_openie_results(corpus_openie_results) -> (Dict[str, NerRawOutput], Dict[str, TripleRawOutput]):

    ner_output_dict = {
        chunk_item['idx']: NerRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            unique_entities=list(np.unique(chunk_item['extracted_entities']))
        )
        for chunk_item in corpus_openie_results
    }
    triple_output_dict = {
        chunk_item['idx']: TripleRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            triples=filter_invalid_triples(triples=chunk_item['extracted_triples'])
        )
        for chunk_item in corpus_openie_results
    }

    return ner_output_dict, triple_output_dict

def extract_entity_nodes(chunk_triples: List[List[Triple]]) -> (List[str], List[List[str]]):
    chunk_triple_entities = []
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
            else:
                logger.warning(f"During graph construction, invalid triple is found: {t}")
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities

def flatten_facts(chunk_triples: List[Triple]) -> List[Triple]:
    graph_triples = []
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    graph_triples = list(set(graph_triples))
    return graph_triples

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val


    if range_val == 0:
        return np.ones_like(x)

    return (x - min_val) / range_val

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()


def all_values_of_same_length(data: dict) -> bool:
    """
    Return True if all values in 'data' have the same length or data is an empty dict,
    otherwise return False.
    """

    value_iter = iter(data.values())


    try:
        first_length = len(next(value_iter))
    except StopIteration:

        return True


    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )
