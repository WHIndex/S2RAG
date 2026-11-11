# S²RAG

S²RAG (`S2RAG` in code and command-line interfaces) executes relational evidence queries over text-derived facts while accounting for missing tuples, unsupported tuples, and bounded source access.

The implementation contains three coupled stages:

1. **Ontology-guided query planning.** A rule-first compiler produces semantic requirements, lowers them into a physical relation plan, and validates predicate, direction, endpoint role, dependency, answer-slot, and plan-coverage constraints.
2. **Support-aware lazy relational execution.** Indexed lookup is attempted first. Missing obligations trigger binding-guided recovery from bounded source passages. Recovered tuples enter a query-local overlay. Source support and obligation compatibility are checked before a tuple can extend a binding. Lineage invalidation and obligation reopening revise only dependent state.
3. **Budgeted witness completion.** BSWC ranks executable source and relation actions using expected supported-witness progress, source suitability, physical cost, hard budgets, and query-local feedback.

The reader receives the initial retrieval context plus complete supported witnesses and bounded verified partial evidence. Structured candidates do not replace the reader answer in the default profile.

## Installation

```bash
python -m pip install -e .
export S2RAG_API_KEY="..."
```

Local embedding weights can be selected with `S2RAG_NVEMBED_MODEL_PATH`. A compatible chat-completion endpoint is configured with `--base-url`.

## Input

```json
{
  "documents": [
    "Document title\nDocument text"
  ],
  "questions": [
    "Which city is associated with the person who directed the film?"
  ]
}
```

## Run

```bash
s2rag input.json \
  --model reader-model \
  --base-url https://service.example/v1 \
  --retrieval-backend graph_ppr \
  --save-dir outputs/run1 \
  --output answers.json
```

Use `nv_embed_dense` to run without graph propagation. Pass a precomputed BSWC calibration file with `--calibration-path`; when omitted, the executor uses its static hierarchical priors and query-local feedback.

## Core modules

- `relation_fact_join_utils.py`: ontology catalog, normalized relation facts, composite indexes, signatures, and indexed relation access.
- `evidence_query_compiler_semantic_v3_15_runtime_utils.py`: rule-first semantic compilation, deterministic lowering, and plan admission.
- `evidence_query_execution_utils.py`: lookup, recovery, verification, lineage revision, query-local overlay, and witness construction.
- `bswc_optimizer_utils.py` and `bswc_feedback_utils.py`: budget-aware physical action selection and feedback updates.
- `profile.py`: the default S2RAG execution profile.
