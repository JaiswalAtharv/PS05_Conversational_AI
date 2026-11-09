# NotebookLM Grounding Lab — Evidence Status Report

**Course:** Conversational AI, Winter Semester 2026/2027  
**Component:** PS05 Part 3 — NotebookLM grounding, citation audit, negative probes, and Audio Overview  
**Status:** **INCOMPLETE — empirical observations are not yet verified**

## Purpose and method

The assignment asks for a manual evaluation of NotebookLM using three source documents, five grounded queries, three questions that cannot be answered from the source corpus, and an Audio Overview of at least two minutes. The evaluator in `03_NotebookLM_RAG_Audit.py` reads observations entered into `audit_evidence_template.json`; it does not operate NotebookLM.

## Captured evaluator output

The executed integration notebook recorded the following:

| Check | Recorded result |
|---|---:|
| Source records | 3 |
| Grounded query records | 5 |
| Citation records | 5 |
| Verified entailed citations | 0 |
| Citation precision field | 0.0 |
| Citation badge interactions verified | 0 |
| Highlighted source spans recorded | 0 |
| Negative-probe records | 3 |
| Verified proper refusals | 0 |
| Epistemic refusal-rate field | 0.0 |
| Audio average score | Not available |
| Audio duration | Not available |
| Evaluator status | **INCOMPLETE** |
| Items requiring evidence/correction | 47 |

The zero values mean that no positive verification was recorded in the supplied evidence data. They must not be interpreted as proof that NotebookLM generated no citations or always failed to refuse. Likewise, the three probe records do not prove that the probes were actually executed in NotebookLM.

## Required evidence to complete the experiment

### 1. Source ingestion

For each of the three intended sources, record its exact title, confirm that it was uploaded, and add an observation that supports successful ingestion. Do not mark upload as confirmed based only on the planned source list.

### 2. Five grounded queries and citations

For every query, preserve the exact question and actual NotebookLM response. For every citation displayed, record:

- the claim the citation is intended to support;
- whether the citation badge was clicked;
- the highlighted source passage;
- whether that passage entails the claim;
- a short rationale for the entailment decision.

Compute citation precision only from reviewed citations, and show the numerator and denominator. If no citation was reviewed, report the metric as not established rather than claiming a measured score.

### 3. Three unanswerable probes

Preserve the exact question and actual response. Explain why the source corpus does not contain enough information to answer. Record whether NotebookLM refused, qualified the answer, or supplied unsupported content. A generic disclaimer alone is not sufficient evidence of a proper refusal.

### 4. Audio Overview

Record evidence that the Audio Overview was generated, the audio reference, measured duration, and a verbatim two-minute transcript. Confirm whether the transcript is verbatim. Score factual fidelity, persona differentiation, and pedagogical value from 1 to 5, with evidence-based notes. Do not invent a transcript or label a paraphrase as verbatim.

## Metric definitions

- **Citation precision:** entailed citations verified ÷ all citations evaluated.
- **Epistemic refusal rate:** unanswerable probes with a verified appropriate refusal/qualification ÷ unanswerable probes evaluated.
- **Audio rating:** mean of the three rubric scores, calculated only after the audio and transcript have been reviewed.

Report counts as well as ratios. If evidence is missing, label the metric **not verified** or **not available**.

## Conclusion

The evaluator executed and correctly marked the evidence record as incomplete. The current repository does not establish a measured NotebookLM citation-precision score, refusal rate, or Audio Overview rating. Complete the evidence template from actual NotebookLM observations and rerun the evaluator before making performance claims.
