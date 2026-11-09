# NotebookLM Grounding Lab — Evidence Status Report

**Course:** Conversational AI, Winter Semester 2026/2027  
**Component:** PS05 Part 3 — NotebookLM grounding, citation audit, negative probes, and Audio Overview  
**Status:** **Evidence records populated — independent human review pending**

## Purpose and scope

`03_NotebookLM_RAG_Audit.py` evaluates manually recorded observations in `audit_evidence_template.json`. It does not operate NotebookLM and cannot independently confirm that a citation entails a claim, that a refusal is appropriate, or that an audio transcript matches the rendered audio. The evaluator checks record completeness and required fields.

## Current recorded results

The evaluator was rerun against the current JSON evidence record. It reports:

| Check | Recorded result | Interpretation |
|---|---:|---|
| Source records | 3 | Three source records entered |
| Grounded query records | 5 | Five query records entered |
| Citation records | 22 | 22 citation-level records entered |
| Citations marked as entailed | 22/22 (100%) | Recorded judgments; independent claim-to-passage review pending |
| Citation badge interactions | 22 | Interactions recorded in the evidence JSON |
| Highlighted source spans | 22 | Source-span records entered |
| Negative probes | 3 | Three unanswerable-probe records entered |
| Probes marked as appropriate refusals | 3/3 (100%) | Recorded judgments; independent response review pending |
| Audio rubric average | 5.0/5 | User-reported score: 5/5 for factual fidelity, persona differentiation, and pedagogical value |
| Audio duration | 4,258 seconds (1 hr 10 min 58 sec) | Confirmed by the user from the rendered NotebookLM Studio artifact |
| Evaluator result | `EVIDENCE RECORDS COMPLETE` | Structural completeness only; not an independent empirical validation |

## Audio Overview verification note

The user reports that the rendered Studio artifact is titled *How Structured Memory Prevents AI Hallucinations*, with reference ID `3127ae4d-ccc5-41f9-afb0-5bc62196c3f5`, and confirmed its duration as 1 hr 10 min 58 sec (4,258 seconds). The user also reports having evaluated the artifact and assigned 5/5 for factual fidelity, persona differentiation, and pedagogical value. The recorded rubric rationale discusses RAG parametric/non-parametric memory, DPR, BM25/RRF, GraphRAG and Leiden community detection, epistemic limitations, and distinct dialogue roles.

These details are documented as user-confirmed observations. **Independent line-by-line comparison of the supplied dialogue transcript with the rendered audio remains pending**, as does independent review of the source-based claims and scoring. The JSON field values should not be represented as externally or independently audited merely because the evaluator accepts them.

## Required human review before claiming validated metrics

1. **Citation entailment:** inspect all 22 citation badges and highlighted spans, and verify that each cited passage supports the specific associated claim. Record corrections and rationales where needed.
2. **Negative probes:** inspect the actual NotebookLM response to each of U1–U3 and confirm that each appropriately acknowledges missing information rather than guessing or using unrelated evidence.
3. **Audio/transcript alignment:** compare the supplied transcript with the rendered artifact line by line. If it is not verbatim, correct the transcript and its status rather than claiming verbatim alignment.
4. **Audio scoring:** retain the 5/5 rubric scores as the user's evaluation; label them as independently reviewed only after a reviewer checks the audio and rubric evidence.
5. Rerun `python 03_NotebookLM_RAG_Audit.py audit_evidence_template.json` after any evidence changes and refresh the saved notebook output.

## Conclusion

The evidence JSON is populated and the evaluator's structural checks pass. The recorded metrics are 22/22 citations marked entailed, 3/3 negative probes marked as appropriate refusals, and a user-reported audio average of 5.0/5 for a confirmed 4,258-second artifact. These values are **recorded observations, not yet independently validated findings**. The remaining work is a human audit of claim-to-passage entailment, refusal responses, and transcript-to-audio alignment.
