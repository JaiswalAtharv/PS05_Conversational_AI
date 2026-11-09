# Integrated Mission 5 (PS05): Grounding AI
## From Vector Indexing to Hybrid Retrieval, NotebookLM, and RAG Safety

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Mission:** Week 5 — Grounding AI: From Vector Indexing to Google NotebookLM & GraphRAG  
**Galactic Waypoint:** Crab Pulsar Remnant / Messier 1 (`SEC-05-M01`)  
**Primary Objective:** Restore Subsystem 05, the Dual-Path Associative Memory Core  
**Deliverable:** Integrated Mission 5 Solution Workbook

---

# Executive Summary and Station Sign-offs

During transit through Sector `SEC-05-M01`, synchrotron radiation from the Crab Pulsar Remnant disrupted AURA's associative memory subsystem. The mission scenario identifies two principal retrieval challenges: recovering exact identifiers and retrieving information that requires semantic or corpus-level context.

The USSE Noosphere Bridge Crew addressed these challenges through four complementary implementation modules. Together, they demonstrate approximate nearest-neighbour indexing, hybrid lexical and dense retrieval, retrieval-grounding evaluation, and prompt-injection detection with structured provenance logging.

### Bridge Station Responsibilities

| Station | Responsibility | Implementation Module |
|---|---|---|
| Systems Architect | Hierarchical vector indexing and approximate nearest-neighbour retrieval | `01-hnsw-vector-index.py` |
| Communications & Protocol Officer | BM25 lexical retrieval and Reciprocal Rank Fusion (RRF) | `02-bm25-rrf-combiner.py` |
| Data & Payload Specialist | NotebookLM grounding and citation-audit workflow | `03-notebooklm-rag-audit.py` |
| Security & Ethics Sentinel | Prompt-injection detection, provenance logging, and RAG Triad scoring | `04-rag-provenance-and-safety.py` |

### Mission Outcome

All four Python modules executed successfully in the configured Python virtual environment. The HNSW module passed its included smoke tests, and a separate benchmark measured its query performance against exact brute-force cosine search.

The benchmark also revealed an important engineering result: for the tested dataset and implementation, HNSW was slower than the vectorized exact-search baseline. This finding is retained in the report rather than replaced with an unsupported performance claim.

**Narrative status:** Within the fictional mission scenario, Subsystem 05 is described as recovering from 28.4% to 35.5% integrity, an increase of 7.1 percentage points. These integrity values represent the mission narrative, not measured software-performance metrics.

---

## Verified execution update — 4 November 2026

The user's latest local terminal output confirms: (1) HNSW smoke tests passed; (2) BM25/RRF smoke tests passed and printed the rankings recorded below; (3) the NotebookLM evaluator returned `INCOMPLETE`, with 5 citation records but 0 verified entailments, 3 negative probes but 0 verified refusals, and no audio score/duration; and (4) the safety/provenance tests passed, the heuristic test returned `BLOCK_HEURISTIC`, the embedding scan loaded, and the benign example's injection similarity was approximately `0.108908` versus threshold `0.72`.

The required HNSW benchmark at 25,000 vectors × 768 dimensions is **not yet completed**. An attempt to run the full pure-Python benchmark in this hosted execution environment exceeded the available execution window; no assignment-scale timing or recall figure is reported. The earlier 5,000 × 128 figures remain supplementary only.

# Part 1: Systems Architect — HNSW Multi-Layer Vector Index

## 1.1 Objective

The Systems Architect developed an educational implementation of a Hierarchical Navigable Small World (HNSW)-style index for approximate nearest-neighbour retrieval.

The purpose is to organize vectors in a multi-layer proximity graph so that queries can explore promising regions of the index without necessarily comparing the query against every stored vector.

The implementation uses Python, NumPy, and cosine distance.

## 1.2 Core Implementation

The module implements the following operations:

1. **Vector validation and normalization:** Converts input vectors to `float32`, checks dimensional consistency, rejects non-finite values, and prevents cosine normalization of zero vectors.
2. **Hierarchical level assignment:** Assigns nodes to graph layers using a logarithmic probability distribution.
3. **Greedy upper-layer traversal:** Moves towards closer nodes while descending through the hierarchy.
4. **Best-first graph search:** Uses priority queues to explore promising candidates.
5. **Neighbour selection and pruning:** Connects newly inserted nodes and limits graph degree.
6. **Approximate nearest-neighbour retrieval:** Returns node identifiers ranked by cosine similarity.
7. **Graph validation:** Checks layer membership and maximum-degree constraints.
8. **Benchmarking:** Compares approximate retrieval with exact brute-force cosine search.

The implementation is educational. Its neighbour-selection and graph-maintenance procedures are simplified and should not be assumed to reproduce every optimization in a production HNSW library.

## 1.3 Mathematical Formulations

### Cosine Similarity

For non-zero vectors \(u\) and \(v\), cosine similarity is defined as

\[
\operatorname{sim}(u,v)
=
\frac{u^\top v}
{\|u\|_2\|v\|_2}.
\]

The implementation normalizes stored vectors and uses the corresponding cosine distance:

\[
d_{\cos}(u,v)
=
1-\operatorname{sim}(u,v).
\]

Smaller cosine distances indicate greater similarity.

### Hierarchical Level Assignment

The maximum layer of a newly inserted node is sampled using

\[
l=
\left\lfloor
-\ln(U)\,m_L
\right\rfloor,
\]

where

\[
m_L=\frac{1}{\ln M},
\]

\(U\) is sampled from a uniform distribution on \((0,1)\), and \(M\) is the configured graph-degree parameter.

This distribution makes higher layers progressively less populated.

### Search Parameters

The implementation exposes three important parameters:

- \(M\): Controls the target neighbour degree in upper layers; layer 0 permits up to \(2M\) neighbours.
- \(ef_{\text{construction}}\): Controls the candidate-search breadth during insertion.
- \(ef_{\text{search}}\): Controls the candidate-search breadth during queries.

These parameters affect graph structure, retrieval quality, memory usage, and query performance.

## 1.4 Experimental Configuration

The benchmark compared HNSW retrieval with exact brute-force cosine search on the same collection of normalized vectors.

| Parameter | Value |
|---|---:|
| Indexed vectors | 5,000 |
| Vector dimensionality | 128 |
| Number of queries | 10 |
| \(M\) | 16 |
| \(ef_{\text{construction}}\) | 64 |
| \(ef_{\text{search}}\) | 64 |
| Exact-search method | NumPy vectorized cosine similarity |

The benchmark generated synthetic vectors and query vectors using a fixed seed for reproducibility.

## 1.5 Measured Results

| Metric | Observed result |
|---|---:|
| Index construction time | 6.67 seconds |
| Mean HNSW query latency | 1.184 ms |
| Mean exact-search query latency | 0.528 ms |
| Measured speedup | 0.45× |
| Mean Recall@10 | 0.790 |

### Speedup Calculation

The measured speedup is calculated as

\[
\operatorname{Speedup}
=
\frac{T_{\text{exact}}}{T_{\text{HNSW}}}.
\]

Substituting the measured mean latencies gives

\[
\operatorname{Speedup}
=
\frac{0.528}{1.184}
\approx 0.446
\approx 0.45.
\]

Thus, HNSW achieved approximately \(0.45\times\) the speed of the exact-search baseline in this experiment. Equivalently, its mean query latency was approximately 2.24 times higher.

### Recall@10

Recall@10 measures the overlap between the approximate and exact top-10 result sets:

\[
\operatorname{Recall@10}
=
\frac{
\left|
A_{10}\cap E_{10}
\right|
}{10},
\]

where \(A_{10}\) is the approximate top-10 set and \(E_{10}\) is the exact top-10 set.

The measured mean Recall@10 was

\[
\operatorname{Recall@10}=0.790.
\]

This corresponds to an average of 7.9 overlapping results out of the exact top 10 per query.

## 1.6 Interpretation

The benchmark did not demonstrate a latency advantage for this HNSW implementation under the tested conditions.

One plausible explanation is implementation overhead: the educational index performs graph traversal and heap operations in Python, while the exact-search baseline benefits from NumPy's optimized vectorized operations.

The experiment nevertheless demonstrates the construction and use of a hierarchical proximity graph, as well as the importance of measuring approximate retrieval against a meaningful baseline.

The observed performance should not be generalized to all HNSW implementations or dataset sizes. A production implementation, different graph parameters, or a larger dataset may produce different results.

## 1.7 Validation

The module's included smoke tests passed.

They check basic retrieval behaviour, duplicate-node rejection, zero-query rejection, and graph invariants.

**Station status:** Implementation and smoke tests successful; benchmark completed; no speedup over exact search demonstrated for the measured configuration.

---

# Part 2: Communications & Protocol Officer — BM25 and Reciprocal Rank Fusion

## 2.1 Objective

The Communications & Protocol Officer implemented a hybrid retrieval workflow that combines lexical retrieval with dense-vector retrieval.

These retrieval paths address different query characteristics:

- **Dense retrieval** uses vector representations to find semantically similar content.
- **Lexical retrieval** uses token-based matching to identify documents containing relevant terms, identifiers, or phrases.
- **Reciprocal Rank Fusion** combines ranked results from multiple retrieval systems without requiring their raw scores to share the same scale.

This combination is particularly useful when a query includes both conceptual language and exact technical identifiers.

## 2.2 BM25 Lexical Retrieval

Okapi BM25 is a lexical ranking method that scores documents according to query-term matches while accounting for term frequency, document length, and the rarity of terms across the corpus.

A common BM25 formulation is

\[
\operatorname{BM25}(D,Q)
=
\sum_{q_i\in Q}
\operatorname{IDF}(q_i)
\frac{
f(q_i,D)(k_1+1)
}{
f(q_i,D)+
k_1\left(
1-b+b\frac{|D|}{\operatorname{avgdl}}
\right)
},
\]

where:

- \(D\) is a document.
- \(Q\) is the query.
- \(f(q_i,D)\) is the frequency of term \(q_i\) in document \(D\).
- \(|D|\) is the document length.
- \(\operatorname{avgdl}\) is the average document length.
- \(k_1\) controls term-frequency saturation.
- \(b\) controls document-length normalization.
- \(\operatorname{IDF}(q_i)\) measures the discriminative value of a query term.

The BM25 engine provides a lexical ranking that complements dense-vector retrieval.

## 2.3 Reciprocal Rank Fusion

The RRF combiner assigns a score to each document based on its rank in each retrieval list.

For a document \(d\), the fusion score is

\[
\operatorname{RRF}(d)
=
\sum_{r\in R}
\frac{1}{k+\operatorname{rank}_r(d)},
\]

where:

- \(R\) is the set of retrieval rankings.
- \(\operatorname{rank}_r(d)\) is the rank of document \(d\) in ranking \(r\).
- \(k\) is a smoothing constant, set to 60 in this module.

Documents that rank highly in multiple retrieval lists accumulate larger fusion scores.

## 2.4 Implementation

The implemented combiner follows this procedure:

1. Initialize a score for each retrieved document.
2. Traverse the dense retrieval ranking.
3. Add the reciprocal-rank contribution for each document.
4. Traverse the sparse retrieval ranking.
5. Add the corresponding reciprocal-rank contribution.
6. Sort documents by their combined scores.
7. Return the highest-ranked documents up to the requested result count.

The implementation uses ranked document lists as inputs. It does not require direct comparison between dense similarity scores and BM25 scores.

## 2.5 Execution Results

The module completed two test scenarios.

| Scenario | BM25 top document | Hybrid RRF top document |
|---|---|---|
| Exact serial lookup | `DOC_003` | `DOC_001` |
| Conceptual retrieval | `DOC_004` | `DOC_004` |

The actual terminal output reported the following hybrid RRF scores:

| Scenario | Hybrid top document | RRF score |
|---|---|---:|
| Exact serial lookup | `DOC_001` | 0.0325224749 |
| Conceptual retrieval | `DOC_004` | 0.0327868852 |

The module reported successful validation across both query scenarios.

## 2.6 Interpretation

The observed results demonstrate that the hybrid ranking pipeline completed both an exact-identifier query and a conceptual query. For the exact identifier, BM25 alone ranks `DOC_003` first, but the fused hybrid RRF ranking places `DOC_001` first. The supplied dense rankings are illustrative inputs, not measured embedding retrieval results.

However, two successful scenarios do not establish 100% retrieval precision across a general evaluation dataset.

A defensible precision measurement would require labelled relevance judgments for the retrieved documents. Such an evaluation should specify the number of test queries, the relevance criteria, and the precision metric used.

**Station status:** Hybrid ranking workflow executed successfully; broader retrieval precision remains unverified.

---

# Part 3: Data & Payload Specialist — NotebookLM Grounding Audit

## 3.1 Objective and evidence standard

The NotebookLM audit is intended to assess citation entailment, epistemic restraint on unanswerable questions, and Audio Overview quality. The evaluator reads a manually completed evidence file; it does not connect to NotebookLM or independently retrieve its responses.

The current terminal run of `03-notebooklm-rag-audit.py` reported:

| Field | Current evaluator output |
|---|---:|
| Source records | 3 |
| Grounded queries recorded | 5 |
| Citation records | 5 |
| Entailed citations verified | 0 |
| Citation precision | 0.0 |
| Citation badge interactions verified | 0 |
| Highlighted source spans recorded | 0 |
| Negative probes recorded | 3 |
| Appropriate refusals verified | 0 |
| Epistemic refusal rate | 0.0 |
| Audio average score | Not available |
| Audio duration | Not available |
| Overall audit status | **INCOMPLETE** |

The evaluator listed 47 evidence or correction items. The source upload and ingestion notes are not confirmed; the five grounded-query records lack actual query/response and claim-level citation verification; the three negative probes lack verified actual responses and justifications; and the Audio Overview evidence, duration, transcript, and scores are missing.

## 3.2 Interpretation

The current output does **not** support claims of 100% citation precision, 100% epistemic refusal rate, or a 5.0/5.0 Audio Overview score. Those numbers appeared in an earlier narrative report, but they are not verified by the current evidence template and evaluator run. They must not be presented as measured NotebookLM results unless the underlying evidence is supplied and independently checked.

To complete the audit, record the actual three source uploads and ingestion evidence; five real NotebookLM queries and responses; each citation's associated claim, interactive badge check, highlighted source span, and entailment judgment; three real unanswerable probes with responses and source-limitation explanations; and the actual Audio Overview reference, duration of at least 120 seconds, verbatim transcript, three rubric scores, and evidence-based notes.

**Station status:** Evaluator executed successfully; evidence incomplete; empirical NotebookLM metrics remain unverified.

# Part 4: Security & Ethics Sentinel — Prompt Injection, Provenance, and RAG Triad

## 4.1 Objective

The Security & Ethics Sentinel implemented a basic indirect prompt-injection detector and a structured provenance record for a retrieval-augmented generation workflow.

The module aims to demonstrate three complementary safeguards:

1. Detect suspicious content in retrieved chunks.
2. Record information about a retrieval-and-generation turn.
3. Evaluate the relevance and faithfulness of a generated response.

These mechanisms serve different purposes and should not be treated as interchangeable.

## 4.2 Indirect Prompt-Injection Detection

Retrieved documents may contain instructions intended to manipulate the language model rather than provide relevant factual information.

The detector combines a heuristic/regular-expression screen with an embedding-based similarity screen using Sentence Transformers. The terminal run confirmed that the embedding model loaded and the embedding scan was available. These are complementary signals, and the output should be interpreted per test input.

The module reported detection of the test pattern:

```text
ignore all previous instructions
```

The latest terminal output reported `Heuristic malicious-input test: BLOCK_HEURISTIC`. This confirms detection of the configured test pattern. It does not yet demonstrate detection of a paraphrased attack that avoids the configured heuristic rules.

### Security Limitation

This is a pattern-based detector, not a comprehensive prompt-injection defence.

It may fail to detect attacks expressed through paraphrasing, encoding, indirect instructions, or patterns not covered by its rules. Conversely, benign text may match a suspicious pattern.

Detection alone also does not neutralize malicious instructions. A complete security design should combine detection with instruction hierarchy enforcement, separation of trusted instructions from retrieved content, controlled tool permissions, and appropriate handling of suspicious documents.

## 4.3 Structured Provenance Logging

The latest terminal output generated a structured JSON record containing `turn_id`, the query, retrieved chunk IDs and text, similarity scores, per-chunk heuristic and embedding scan status, the embedding threshold, RAG Triad proxy scores, and the grounded answer. For the benign example, `CHUNK_001` had a retrieval similarity score of `0.8731` and an injection similarity of approximately `0.108908` against the `0.72` threshold.

This record links the query, retrieved document identifiers, security status, evaluation scores, and generated response.

Such records can support debugging and auditing by preserving information about the retrieval-and-generation process.

### Interpreting the Security Status

The terminal output includes a heuristic test returning `BLOCK_HEURISTIC` and a separate benign provenance example classified `CLEAN`. These are different test inputs and are not contradictory. In the benign example, the maximum injection similarity was approximately `0.108908`, below the configured threshold of `0.72`. A separate paraphrased-injection test is still needed to demonstrate that the embedding pathway can detect an attack that does not match the heuristic patterns.

## 4.4 RAG Triad Evaluation

The module reported three scores:

| Dimension | Script-reported score |
|---|---:|
| Context relevance | 1.0 |
| Faithfulness | 1.0 |
| Answer relevance | 1.0 |

The three dimensions represent different evaluation questions:

- **Context relevance:** Does the retrieved context relate to the user's query?
- **Faithfulness:** Is the generated response supported by the available context?
- **Answer relevance:** Does the response address the user's question?

A score of 1.0 indicates the maximum score under the configured scoring scheme. It does not, on its own, demonstrate perfect semantic quality.

The evaluation methodology should explain how each score is computed, identify the test cases, and distinguish heuristic or rule-based scores from human- or model-judged semantic evaluations.

## 4.5 Interpretation

The module successfully executed its configured prompt-injection detection and generated a structured provenance record.

These results demonstrate the operation of the implemented mechanisms, not complete protection against prompt injection or independently verified semantic correctness.

**Station status:** Detection test passed; provenance record generated; broader security coverage and RAG Triad validity require evaluation against documented test cases.

---

# Part 5: Integrated System Assessment

## 5.1 Complementary Retrieval and Safety Functions

The four modules address different aspects of a retrieval-grounded AI system.

| Module | Primary function | Demonstrated output |
|---|---|---|
| HNSW index | Approximate dense-vector retrieval | Nearest-neighbour results, graph validation, latency and recall measurements |
| BM25 + RRF | Lexical and dense-ranking combination | Ranked results for exact-identifier and conceptual queries |
| NotebookLM audit | Grounding and refusal evaluation | Script-reported citation, refusal, and audio-rating metrics |
| Provenance and safety | Suspicious-pattern detection and turn logging | Detection warning, JSON provenance record, and RAG Triad scores |

The architecture illustrates why no single mechanism is sufficient for a reliable retrieval-grounded assistant.

Dense retrieval can support semantic matching, lexical retrieval can help recover exact identifiers, rank fusion can combine complementary rankings, grounding evaluation can assess source support, and security controls can help identify potentially malicious retrieved content.

The quality of the overall system depends on the correctness of each component and the reliability of the evidence used to evaluate it.

## 5.2 Experimental Evidence Versus Mission Narrative

For clarity, the workbook distinguishes three kinds of claims:

1. **Execution results:** Outputs directly observed when the Python modules were run.
2. **Evaluation results:** Metrics that require documented test data and a defined scoring procedure.
3. **Mission narrative:** Fictional AURA subsystem integrity values used to frame the assignment.

This distinction prevents narrative values or unverified script outputs from being misrepresented as measured system performance.

---

# Final System Restoration Metrics

## A. Implementation and Execution Results

| Component | Final recorded result | Status |
|---|---|---|
| HNSW smoke tests | Passed | Confirmed by terminal output |
| HNSW index construction | 5,000 vectors indexed in 6.67 s (earlier run) | Preliminary smaller-scale run only |
| HNSW query latency | 1.184 ms mean (earlier run) | Measured over 10 queries on 5,000 × 128 |
| Exact-search latency | 0.528 ms mean (earlier run) | Measured over 10 queries on 5,000 × 128 |
| HNSW speedup | 0.45× (earlier run) | HNSW was slower in that run |
| HNSW Recall@10 | 0.790 (earlier run) | Measured on 5,000 × 128; not assignment-scale |
| Assignment-scale HNSW benchmark | 25,000 × 768 | **Pending; no completed result recorded** |
| BM25 + RRF | Both test scenarios completed | Confirmed by terminal output |
| NotebookLM audit | Current evaluator: 5 citation records, 0 verified entailments, 3 negative probes, 0 verified refusals, audio score unavailable | **INCOMPLETE; actual evidence required** |
| Prompt-injection test | Heuristic test returned `BLOCK_HEURISTIC`; embedding scan loaded | Confirmed for configured test; paraphrased-injection detection not yet demonstrated |
| Provenance and RAG Triad | JSON record generated; three lexical proxy scores reported as 1.0 | Confirmed as program output; scores are not semantic verification |

## B. Mission-Narrative Integrity

Within the fictional USSE Noosphere mission, Subsystem 05 is described as improving from

\[
I_{\text{initial}}=28.4\%
\]

to

\[
I_{\text{final}}=35.5\%.
\]

The increase is

\[
\Delta I
=
35.5\%-28.4\%
=
7.1\text{ percentage points}.
\]

These figures represent the mission narrative. They are not derived from the HNSW benchmark or from a measured software-integrity calculation.

---

# Conclusion

Mission 5 brings together four important concepts in retrieval-grounded conversational AI: hierarchical vector indexing, hybrid lexical and dense retrieval, grounding evaluation, and security-aware provenance logging.

The implementation work demonstrates functioning HNSW-style graph operations, successful BM25 and RRF test scenarios, execution of the grounding-audit workflow, and detection of a configured adversarial pattern.

The HNSW benchmark provides a particularly useful engineering lesson. Despite the algorithm's purpose of accelerating approximate nearest-neighbour retrieval, the educational implementation did not outperform NumPy's exact-search baseline under the tested configuration. The measured latency and Recall@10 provide a concrete basis for discussing the trade-off between retrieval quality and performance.

The remaining evaluation claims should be supported by reproducible evidence, documented test cases, and transparent scoring procedures.

Within the mission narrative, Subsystem 05 is recorded as restored to 35.5% integrity. From an engineering perspective, the four modules establish a foundation for further testing and refinement of the Noosphere's retrieval-grounding and safety capabilities.

**Final Mission Status:** All four Python modules executed successfully in the user's local environment; HNSW and BM25/RRF smoke tests passed, both hybrid-retrieval scenarios ran successfully, the NotebookLM evaluator returned `INCOMPLETE` as expected for missing evidence, and the safety/provenance module generated its record. The assignment-scale HNSW benchmark and complete NotebookLM evidence audit remain outstanding.

**USSE Noosphere Bridge Crew — Mission 5 Workbook Complete**