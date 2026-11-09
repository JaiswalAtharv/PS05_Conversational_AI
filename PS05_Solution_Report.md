# PS05 Solution Report — Implementation, Experiments, and Limitations

**Course:** Conversational AI, Winter Semester 2026/2027  
**Repository:** `PS05_Conversational_AI`  
**Purpose:** Describe the implementation and the observations actually captured in this repository.

## 1. Overview

The project is organized into four components and an integration notebook:

1. an HNSW-style approximate nearest-neighbour index;
2. a BM25 lexical retriever and Reciprocal Rank Fusion (RRF);
3. a manual-evidence evaluator for a NotebookLM grounding audit;
4. prompt-injection screening, RAG-triad proxy metrics, and a provenance log.

The code and captured outputs demonstrate several working component paths. They do not establish that every empirical requirement in the assignment is complete. In particular, NotebookLM evidence records are populated but citation entailment, refusal judgments, and transcript-to-audio alignment still require independent human review; the HNSW result has low recall, and the semantic embedding-based injection scan was unavailable in the captured run.

## 2. HNSW vector index

### Implementation

`01_HNSW_Vector_Index.py` implements an index using Python data structures and NumPy. It normalizes input vectors, uses cosine distance (`1 - cosine similarity`), assigns node levels with an exponentially decaying random-level rule, performs greedy descent through upper layers, and uses a best-first search at layer zero. The class includes graph validation and rejects invalid vectors and duplicate identifiers. An exact cosine top-k function provides a flat-search baseline.

### Recorded benchmark

The executed notebook records one run with 25,000 vectors of 768 dimensions:

| Measure | Value |
|---|---:|
| Build time | 68.45 s |
| Mean HNSW query latency | 5.096 ms |
| Mean exact flat-search latency | 5.311 ms |
| Measured speedup | 1.04× |
| Mean recall@10 | 0.250 |

The recorded latency advantage is small, while recall@10 is low: on average, only 2.5 of the exact top-10 neighbours were recovered per query in this run. Therefore the result is not evidence of a strong speed/quality trade-off. It motivates further tuning of `ef_search`, graph construction settings, and repeated measurements. The values are specific to this run and machine; they should not be generalized.

The script's default benchmark configuration differs from any manually supplied configuration, so future reports should include the actual values of `M`, `ef_construction`, `ef_search`, number of queries, and seed used for each result.

## 3. BM25 and Reciprocal Rank Fusion

### Implementation

`02_BM25_and_RRF_Combiner.py` tokenizes lowercase alphanumeric and underscore-containing identifiers, computes BM25 scores with `k1=1.5` and `b=0.75`, and fuses rankings using RRF with `k=60`. The RRF implementation supports multiple ranked lists, optional non-negative weights, and ignores duplicate document IDs within each list.

### Recorded demonstration

The notebook output reports that the BM25/RRF smoke tests passed. It demonstrates exact-identifier and conceptual queries. The dense rankings are **illustrative supplied inputs**, not rankings produced by an embedding model. The example therefore verifies the fusion implementation's behavior for given rankings, but it is not an end-to-end empirical comparison of a real dense retriever against BM25.

For the exact identifier `SERIAL_NO_AURA_9982X`, the displayed BM25 ranking places `DOC_003` first. The hybrid RRF ranking places `DOC_001` first, showing that fusion can change the ordering when both ranked lists contribute. This is an expected property of rank fusion, not proof that the fused result is always more relevant.

## 4. NotebookLM grounding audit

`03_NotebookLM_RAG_Audit.py` evaluates a manually populated JSON evidence record. It does not connect to NotebookLM or perform the experiment. It checks for source-upload evidence, five grounded queries, citation-level records, three unanswerable probes, and a two-minute audio overview with transcript and scores.

The current evidence record produces the following evaluator summary:

- 3 source records and 5 grounded query records are present;
- 22 citation records are present, with all 22 marked as entailed and all 22 citation-badge/highlight records populated;
- 3 negative probes are present, with all 3 marked as appropriate refusals;
- the Audio Overview duration is recorded as 4,258 seconds (1 hr 10 min 58 sec), confirmed by the user from the rendered Studio artifact;
- the user reports assigning 5/5 on each audio rubric dimension;
- evaluator status: **EVIDENCE RECORDS COMPLETE**.

The record-completeness result is not independent validation. Source-based entailment, refusal quality, and line-by-line transcript-to-audio alignment remain subject to human review.

These are evidence-record evaluation results, not a finding that NotebookLM itself failed. The current JSON records 22 citations marked as entailed, 3 probes marked as appropriate refusals, and a user-reported audio score of 5/5 on each rubric dimension with a confirmed duration of 4,258 seconds. The evaluator confirms record completeness only; citation entailment, refusal behavior, and transcript-to-audio alignment remain pending independent human review.

## 5. Prompt-injection screening, RAG-triad proxies, and provenance

### Injection detector

`04_RAG_Provenance_and_Safety.py` scans retrieved text using regular-expression patterns for suspicious instructions, attempts to expose secrets or override system/developer instructions, certain exfiltration language, Markdown image URLs, and selected HTML-comment patterns. It optionally compares text embeddings with a set of example injection prompts using `sentence-transformers`.

The executed notebook reports that the heuristic malicious-input test returned `BLOCK_HEURISTIC`. It also reports `sentence-transformers` as unavailable, so embedding screening did not run. This is a successful heuristic test only; it does not establish broad attack coverage or complete semantic screening.

### RAG-triad proxy

The notebook reports `1.0` for context relevance, faithfulness, and answer relevance. The output labels the method `lexical_overlap_proxy` and explicitly warns that these scores do not establish semantic entailment or replace human review. They must not be reported as validated semantic RAG-triad scores.

### Provenance

The demonstration emits a structured JSON record with a turn ID, query, retrieved chunk IDs and similarity scores, per-chunk injection scan status, proxy scores, and a grounded-answer field. The example is a synthetic demonstration record and should not be confused with logs from a deployed RAG service or a real user conversation.

## 6. Reproducibility

Run the components from the repository root:

```bash
python 01_HNSW_Vector_Index.py
python 01_HNSW_Vector_Index.py --benchmark
python 02_BM25_and_RRF_Combiner.py
python 03_NotebookLM_RAG_Audit.py audit_evidence_template.json
python 04_RAG_Provenance_and_Safety.py
```

The HNSW and safety scripts require NumPy. `sentence-transformers` is optional for embedding screening. Run the integration notebook in Jupyter to execute its `%run` cells and view the combined demonstrations.

When rerunning, record the Python and dependency versions, machine, random seed, number of queries, and HNSW parameters. Benchmark results may vary between runs.

## 7. Remaining work

1. Tune HNSW and repeat the benchmark with a recall target, reporting both latency and recall.
2. If claiming dense-sparse end-to-end retrieval, connect the RRF combiner to an actual dense retrieval ranking and evaluate relevant queries.
3. Independently review the populated NotebookLM evidence record: source-upload confirmations, verbatim responses, citation badge interactions, highlighted passages, entailment decisions, negative-probe responses, and transcript-to-audio alignment. Update only claims supported by that review.
4. Install and run optional embedding-based injection screening if that part of the experiment is required; report model and threshold details.
5. Replace or supplement lexical RAG-triad proxies with a justified semantic evaluator or human-reviewed evidence if semantic faithfulness is claimed.

## 8. Conclusion

The repository contains working component implementations and an executed integration notebook. The captured results support the claims that the smoke-test paths ran, BM25/RRF produced rankings for supplied inputs, the heuristic detector flagged its test payload, and a provenance record was generated. They do **not** support claims of high-recall HNSW performance, fully validated semantic RAG-triad scores, successful embedding-based screening, or an independently completed NotebookLM empirical audit.
