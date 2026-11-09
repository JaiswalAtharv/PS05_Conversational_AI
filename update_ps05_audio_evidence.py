
#!/usr/bin/env python3
"""Update PS05 audio evidence while preserving existing audit records."""

import json
import shutil
from datetime import datetime
from pathlib import Path

TRANSCRIPT = r"""#### Section 1: Foundations of RAG — Parametric vs. Non-Parametric Memory

Dr. Aris: Welcome to this technical audit on Retrieval-Augmented Generation (RAG), GraphRAG, and epistemic safety. We will examine how language models combine internal parameter weights with external knowledge structures to generate grounded answers.

Elena: In the original 2020 paper by Lewis and colleagues, RAG pairs parametric memory with non-parametric memory. What does that mean?

Dr. Aris: Parametric memory refers to the pretrained neural parameters of a sequence-to-sequence transformer, specifically BART-large. The model parameters act as a probabilistic generator, calculating conditional probability distributions to generate text tokens.

Elena: And where does non-parametric memory fit in?

Dr. Aris: Non-parametric memory is an explicit external text index, such as a dense vector index of Wikipedia passages. A Dense Passage Retriever (DPR) uses dual neural encoders to embed queries and document chunks, performing Maximum Inner Product Search (MIPS) to retrieve relevant passages.

Elena: So the generator can retrieve factual details at query time instead of relying only on memorized information?

Dr. Aris: Exactly. Another advantage is index hot-swapping. You can update or replace the document index without retraining the model parameters.

Elena: Lewis et al. also proposed RAG-Sequence and RAG-Token. How do they differ?

Dr. Aris: RAG-Sequence uses the same retrieved passage to condition the probability of every generated token across the output sequence. RAG-Token can marginalize over different passages for individual tokens, allowing details from multiple passages to contribute to one answer.

#### Section 2: Dense Retrieval Failures, Sparse Matching, and Hybrid RRF

Elena: What happens when you query a vector RAG system for exact technical codes, serial numbers, or acronyms?

Dr. Aris: Dense semantic retrieval maps text into continuous vector spaces, prioritizing conceptual similarity. This can create lexical mismatch: rare alphanumeric strings may be overlooked in favor of broader semantic context.

Elena: Can you give an example?

Dr. Aris: Suppose you query a maintenance log for a valve identifier such as PR-882-HEX. A vector retriever might return general passages about hydraulic valves or thruster maintenance while missing the exact document containing the code.

Elena: How do hybrid architectures address that problem?

Dr. Aris: They combine sparse lexical retrieval with dense vector search. BM25 uses term-frequency and inverse-document-frequency signals to identify exact token matches, making it useful for serial numbers, acronyms, and technical codes.

Elena: But dense search and BM25 produce different score distributions. How can they be combined?

Dr. Aris: Reciprocal Rank Fusion (RRF) combines rank positions rather than raw scores:

RRF(d) = sum over retrievers m of 1 / (k + r_m(d))

Here, r_m(d) is the rank of document d in retriever m, and k is a smoothing constant commonly set around 60. An exact keyword match can therefore contribute strongly to the fused ranking even when its dense-search rank is low. A cross-encoder reranker can then assess candidate passages before they enter the prompt.

#### Section 3: GraphRAG, Community Detection, and Global Sensemaking

Elena: What about broad questions such as, “What are the main systemic themes across all 500 operational logs?” Why can standard passage retrieval fail?

Dr. Aris: Standard top-k retrieval returns isolated snippets. It can retrieve individual facts but may not connect information across an entire collection. Global sensemaking is a query-focused summarization task, not simply a point-fact retrieval task.

Elena: How does GraphRAG index a collection differently?

Dr. Aris: Its pipeline includes entity and relationship extraction, Leiden community detection, hierarchical community summaries, and map-reduce querying. The Leiden algorithm identifies communities in the entity graph. Language models generate summaries for communities, and a global query can use a map-reduce workflow to combine relevant partial responses.

Elena: And GraphRAG can improve global question answering?

Dr. Aris: Research reports improvements in answer comprehensiveness and diversity on evaluated global question-answering benchmarks. Results depend on the task and evaluation setting, so they should not be treated as universal guarantees.

#### Section 4: Epistemic Safety, Audit Probes, and Fictional Boundaries

Elena: Does the literature support the claim that GraphRAG or hybrid RAG completely eliminates hallucinations?

Dr. Aris: No. Retrieval augmentation can provide explicit external evidence, but retrieval noise and reasoning errors can still reduce accuracy. Hallucination elimination is not guaranteed.

Elena: NotebookLM should therefore acknowledge when information is missing and provide inspectable citations rather than pretending to know every answer.

Dr. Aris: Correct. Let us examine three PS05 audit probes.

Elena: First: What is the exact serial number of the auxiliary plasma injector installed on AURA Subsystem 05?

Dr. Aris: This information is unavailable in the supplied sources. The fictional archive discusses Subsystem 05, Sector SEC-05-M01 near the Crab Pulsar, and valve code PR-882-HEX, but it does not mention an auxiliary plasma injector or its serial number.

Elena: Second: What are the exact Right Ascension and Declination coordinates of the fictional telemetry anomaly near the Crab Pulsar Remnant?

Dr. Aris: The numerical coordinates are unspecified. The archive places the anomaly near Messier 1 in Taurus but provides no exact RA or Dec values.

Elena: Third: What exact measured percentage reduction in hallucinations was achieved in the Stage 4 GraphRAG controlled experiment?

Dr. Aris: That specific result is unreported in the supplied sources. The archive states that Stage 4 GraphRAG community summaries helped restore total Subsystem 05 integrity from +7.1% to 35.5%, but it does not report a Stage 4-specific hallucination-reduction percentage.

Elena: We must not substitute unrelated statistics from other studies for missing experimental evidence.

#### Section 5: Citation Verification Protocol and Conclusion

Dr. Aris: Citation verification requires more than counting citation markers.

Elena: A citation can be fabricated or attached to a passage that discusses the same general topic without supporting the specific claim.

Dr. Aris: Correct. Auditors should inspect citation badges and highlighted source passages. Citation recall measures whether substantive claims receive supporting citations. Citation precision assesses whether cited passages genuinely support the associated claims and are relevant rather than unhelpful or redundant.

Elena: To summarize:
1. Lewis et al. (2020) established the RAG framework combining parametric generators with non-parametric retrieval.
2. Hybrid BM25 and dense retrieval can address lexical mismatch for technical codes.
3. GraphRAG uses entity graphs, Leiden community detection, summaries, and map-reduce synthesis for global sensemaking.
4. Fictional case studies illustrate system failure modes but must not be confused with peer-reviewed empirical evidence.
5. Trustworthy AI evaluation requires epistemic refusal when evidence is missing and human-inspected source citations.

Dr. Aris: Excellent synthesis. That concludes our audit of RAG frameworks and epistemic safety."""

SCORING_NOTES = """Factual fidelity (5/5): The supplied dialogue explains parametric and non-parametric memory, DPR/MIPS, BM25/RRF, GraphRAG community summaries, and limitations on claims about eliminating hallucinations. It identifies missing evidence in all three negative probes rather than inventing answers. This score is based on the supplied transcript and should be checked against the source materials and audio.

Persona differentiation (5/5): Dr. Aris presents technical explanations while Elena asks critical questions, challenges assumptions, and synthesizes implications. This assessment is based on the supplied dialogue text.

Pedagogical value (5/5): The dialogue progresses from RAG foundations through retrieval failures, GraphRAG, epistemic safety, and citation verification. It introduces terminology through examples and practical audit guidance. This assessment is based on the supplied dialogue text.

These are the supplied evaluation scores, not independent proof of audio quality. Compare the transcript with the actual NotebookLM Studio audio before setting transcript_is_verbatim to true."""

def main():
    path = Path("audit_evidence_template.json")

    if not path.exists():
        raise SystemExit(
            f"ERROR: {path.resolve()} not found. "
            "Run this script from ~/Downloads/PS05."
        )

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    audio = data.setdefault("audio_overview", {})

    backup = path.with_name(
        f"{path.stem}.backup_audio_evidence_"
        f"{datetime.now():%Y%m%d_%H%M%S}{path.suffix}"
    )
    shutil.copy2(path, backup)

    audio.update({
        "generated_in_notebooklm": True,
        "audio_file_or_link": (
            "NotebookLM Studio artifact reference: "
            "3127ae4d-ccc5-41f9-afb0-5bc62196c3f5"
        ),
        "duration_seconds": 4258,
        "dialogue_transcript": TRANSCRIPT,
        "transcript_is_verbatim": None,
        "scores_1_to_5": {
            "factual_fidelity": 5,
            "persona_differentiation": 5,
            "pedagogical_value": 5
        },
        "scoring_notes": SCORING_NOTES,
        "artifact_reference_id": (
            "3127ae4d-ccc5-41f9-afb0-5bc62196c3f5"
        ),
        "studio_panel_title": (
            "How Structured Memory Prevents AI Hallucinations"
        )
    })

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Validate the saved JSON and confirm existing audit sections remain.
    with path.open(encoding="utf-8") as f:
        saved = json.load(f)

    a = saved["audio_overview"]
    assert a["duration_seconds"] == 4258
    assert a["dialogue_transcript"].strip()
    assert all(
        1 <= score <= 5
        for score in a["scores_1_to_5"].values()
    )
    assert a["transcript_is_verbatim"] is None
    assert "grounded_queries" in saved
    assert "unanswerable_probes" in saved

    print("Audio evidence saved; JSON validation: PASS")
    print(f"Backup: {backup}")
    print(f"Transcript characters: {len(a['dialogue_transcript'])}")
    print(f"Duration recorded: {a['duration_seconds']} seconds")
    print(f"Scores: {a['scores_1_to_5']}")
    print("Verbatim confirmation: PENDING audio comparison")
    print("Other existing audit evidence preserved.")

if __name__ == "__main__":
    main()
