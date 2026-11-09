# NotebookLM Grounding and Audio Overview Audit — Evidence Status

**Course:** Conversational AI, Winter Semester 2026/2027  
**Assignment:** PS05  
**Status:** INCOMPLETE — empirical results not yet verified  
**Last evaluator output supplied:** 4 November 2026

## 1. Purpose

This audit is intended to evaluate NotebookLM on three dimensions: citation entailment for grounded answers, epistemic restraint on questions that cannot be answered from the supplied sources, and the factual/pedagogical quality of a NotebookLM Audio Overview.

The Python evaluator assesses evidence manually entered into `audit_evidence_template.json`. It does not connect to NotebookLM and cannot independently confirm that source files were uploaded, citation badges were clicked, highlighted passages appeared, or an Audio Overview was generated.

## 2. Current evaluator results

The user's actual terminal output reported the following:

| Measure | Current result |
|---|---:|
| Source records | 3 |
| Grounded queries recorded | 5 |
| Citation records | 5 |
| Entailed citations verified | 0 |
| Citation precision | 0.0 |
| Citation badge interactions verified | 0 |
| Highlighted source spans recorded | 0 |
| Negative probes recorded | 3 |
| Proper refusals verified | 0 |
| Epistemic refusal rate | 0.0 |
| Audio average score | Not available (`null`) |
| Audio duration | Not available (`null`) |
| Overall audit status | **INCOMPLETE** |

The evaluator reported 47 items requiring evidence or correction. These include missing source upload/ingestion confirmations, missing actual query/response text, unverified citation entailment and citation badge interactions, missing highlighted source spans, incomplete negative-probe records, and missing Audio Overview evidence, duration, transcript, rubric scores, and notes.

## 3. Interpretation and reporting rule

Earlier draft material stated 14/14 entailed citations (100%), 3/3 appropriate refusals (100%), and a 5.0/5.0 Audio Overview score. The current evidence file and evaluator output do **not** substantiate those figures. Do not report them as measured results unless the raw supporting evidence is completed and checked.

A successful run of the evaluator demonstrates that the code executed; it does not itself prove that NotebookLM performed well. Until evidence is entered, citation precision, refusal rate, and audio quality should be reported as **not verified**, not as a pass.

## 4. Evidence required to complete the audit

### 4.1 Three source documents

For each source, record the exact document title, whether it was uploaded to NotebookLM, and an evidence note confirming that NotebookLM ingested it. The intended corpus is:

1. Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.
2. Edge et al. (2024), *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*.
3. The fictional USSE Noosphere technical archive/PS05 briefing used for the assignment.

### 4.2 Five grounded queries

For each query, preserve the actual query and full NotebookLM response. For every citation, record the claim it is meant to support, the visible citation badge, whether the badge was clicked, the highlighted source span, and an independently justified entailment decision. Count only verified citation decisions in the metric.

### 4.3 Three unanswerable probes

Preserve each exact query and actual response. Explain which required fact is absent from the supplied sources and whether the response appropriately refused, qualified its answer, or identified the evidence gap. Do not count an answer as a proper refusal merely because it contains a disclaimer.

### 4.4 Audio Overview

Record the actual Audio Overview file or evidence reference and its measured duration. The assignment requires at least 120 seconds. Include a verbatim two-minute dialogue transcript, then score factual fidelity, persona differentiation, and pedagogical value from 1 to 5 with brief evidence-based justifications. A synthetic or paraphrased summary must not be labelled as a verbatim NotebookLM transcript.

## 5. Metric definitions

- **Citation precision:** verified entailed citations divided by all citations evaluated.
- **Epistemic refusal rate:** appropriate refusals/limitations divided by the three evaluated unanswerable probes.
- **Audio score:** arithmetic mean of the three rubric scores (factual fidelity, persona differentiation, pedagogical value), each from 1 to 5, only after the actual audio and transcript have been reviewed.

Report numerator and denominator alongside percentages, and retain evidence for each decision.

## 6. Conclusion

The evaluator ran successfully, but the NotebookLM audit is not yet complete. The current verified status is **INCOMPLETE**. No citation-precision percentage, refusal-rate percentage, or Audio Overview rating is established by the supplied evidence. Update this report after the evidence template is completed from actual NotebookLM outputs and the evaluator is rerun.
