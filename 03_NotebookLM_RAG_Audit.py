#!/usr/bin/env python3
"""Evidence-based NotebookLM RAG audit evaluator for PS05.

This program evaluates manually recorded NotebookLM observations. It does not
connect to NotebookLM and does not invent answers, citations, refusals, or scores.

Usage:
    python 03-notebooklm-rag-audit.py --create-template
    # Complete audit_evidence_template.json using actual NotebookLM observations.
    python 03-notebooklm-rag-audit.py audit_evidence_template.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXPECTED_SOURCE_COUNT = 3
EXPECTED_GROUNDED_QUERIES = 5
EXPECTED_NEGATIVE_PROBES = 3
AUDIO_MIN_SECONDS = 120


def template() -> dict[str, Any]:
    """Blank evidence template. Values are deliberately not pre-scored."""
    return {
        "sources": [
            {
                "source_id": "SOURCE_1",
                "title": "Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "uploaded_to_notebooklm": None,
                "evidence_note": ""
            },
            {
                "source_id": "SOURCE_2",
                "title": "Edge et al. (2024) — From Local to Global: A Graph RAG Approach",
                "uploaded_to_notebooklm": None,
                "evidence_note": ""
            },
            {
                "source_id": "SOURCE_3",
                "title": "USSE Noosphere technical archive / PS05 source",
                "uploaded_to_notebooklm": None,
                "evidence_note": ""
            }
        ],
        "grounded_queries": [
            {
                "query_id": f"Q{i}",
                "query": "",
                "notebooklm_response": "",
                "citations": [
                    {
                        "source_id": "",
                        "claim_supported": "",
                        "citation_badge_verified": None,
                        "highlighted_source_span": "",
                        "entailment_verified": None,
                        "review_note": ""
                    }
                ],
                "reviewer": "",
                "review_note": ""
            } for i in range(1, EXPECTED_GROUNDED_QUERIES + 1)
        ],
        "unanswerable_probes": [
            {
                "probe_id": f"U{i}",
                "query": "",
                "notebooklm_response": "",
                "refused_or_explicitly_qualified": None,
                "reviewer": "",
                "reason_source_does_not_answer": ""
            } for i in range(1, EXPECTED_NEGATIVE_PROBES + 1)
        ],
        "audio_overview": {
            "generated_in_notebooklm": None,
            "audio_file_or_link": "",
            "duration_seconds": None,
            "dialogue_transcript": "",
            "transcript_is_verbatim": None,
            "scores_1_to_5": {
                "factual_fidelity": None,
                "persona_differentiation": None,
                "pedagogical_value": None
            },
            "scoring_notes": ""
        }
    }


def require_bool(value: Any) -> bool:
    return isinstance(value, bool)


def evaluate(data: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []

    sources = data.get("sources", [])
    if len(sources) != EXPECTED_SOURCE_COUNT:
        issues.append(f"Expected {EXPECTED_SOURCE_COUNT} source records; found {len(sources)}.")
    for source in sources:
        if source.get("uploaded_to_notebooklm") is not True:
            issues.append(f"Source {source.get('source_id', '?')}: upload is not confirmed.")
        if not source.get("evidence_note", "").strip():
            issues.append(f"Source {source.get('source_id', '?')}: add an ingestion/evidence note.")

    queries = data.get("grounded_queries", [])
    if len(queries) != EXPECTED_GROUNDED_QUERIES:
        issues.append(f"Expected {EXPECTED_GROUNDED_QUERIES} grounded queries; found {len(queries)}.")

    total_citations = 0
    entailed_citations = 0
    badges_verified = 0
    highlighted_spans_verified = 0

    for q in queries:
        qid = q.get("query_id", "?")
        if not q.get("query", "").strip() or not q.get("notebooklm_response", "").strip():
            issues.append(f"{qid}: query and actual NotebookLM response are required.")
        citations = q.get("citations", [])
        if not citations:
            issues.append(f"{qid}: record each citation displayed by NotebookLM.")
        for citation in citations:
            total_citations += 1
            verdict = citation.get("entailment_verified")
            if require_bool(verdict) and verdict:
                entailed_citations += 1
            else:
                issues.append(f"{qid}: citation entailment is unverified or false.")
            if citation.get("citation_badge_verified") is True:
                badges_verified += 1
            else:
                issues.append(f"{qid}: citation badge click/interaction is not verified.")
            if citation.get("highlighted_source_span", "").strip():
                highlighted_spans_verified += 1
            else:
                issues.append(f"{qid}: highlighted source span is missing.")
            if not citation.get("claim_supported", "").strip():
                issues.append(f"{qid}: record the claim supported by this citation.")

    citation_precision = (
        entailed_citations / total_citations if total_citations else None
    )

    probes = data.get("unanswerable_probes", [])
    if len(probes) != EXPECTED_NEGATIVE_PROBES:
        issues.append(f"Expected {EXPECTED_NEGATIVE_PROBES} unanswerable probes; found {len(probes)}.")
    proper_refusals = 0
    for probe in probes:
        pid = probe.get("probe_id", "?")
        if not probe.get("query", "").strip() or not probe.get("notebooklm_response", "").strip():
            issues.append(f"{pid}: record the actual query and NotebookLM response.")
        if probe.get("refused_or_explicitly_qualified") is True:
            proper_refusals += 1
        else:
            issues.append(f"{pid}: refusal/qualification is unverified or not satisfactory.")
        if not probe.get("reason_source_does_not_answer", "").strip():
            issues.append(f"{pid}: explain why the supplied sources cannot answer it.")

    refusal_rate = proper_refusals / len(probes) if probes else None

    audio = data.get("audio_overview", {})
    duration = audio.get("duration_seconds")
    scores = audio.get("scores_1_to_5", {})
    score_names = ("factual_fidelity", "persona_differentiation", "pedagogical_value")
    valid_scores = all(
        isinstance(scores.get(name), (int, float))
        and not isinstance(scores.get(name), bool)
        and 1 <= scores[name] <= 5
        for name in score_names
    )
    audio_average = (
        sum(float(scores[name]) for name in score_names) / len(score_names)
        if valid_scores else None
    )

    if audio.get("generated_in_notebooklm") is not True:
        issues.append("NotebookLM Audio Overview generation is not confirmed.")
    if not audio.get("audio_file_or_link", "").strip():
        issues.append("Provide the actual Audio Overview file/link or evidence reference.")
    if not isinstance(duration, (int, float)) or duration < AUDIO_MIN_SECONDS:
        issues.append("Audio duration is missing or below the required 2 minutes.")
    if not audio.get("dialogue_transcript", "").strip():
        issues.append("The dialogue transcript is missing.")
    if audio.get("transcript_is_verbatim") is not True:
        issues.append("Transcript is not confirmed as verbatim; do not label a paraphrase as verbatim.")
    if not valid_scores:
        issues.append("Enter all three independently justified audio scores from 1 to 5.")
    if not audio.get("scoring_notes", "").strip():
        issues.append("Add evidence-based notes explaining the audio scores.")

    metrics = {
        "source_records": len(sources),
        "grounded_queries_recorded": len(queries),
        "citations_recorded": total_citations,
        "entailed_citations_verified": entailed_citations,
        "citation_precision": citation_precision,
        "citation_badges_verified": badges_verified,
        "highlighted_spans_recorded": highlighted_spans_verified,
        "negative_probes_recorded": len(probes),
        "proper_refusals_verified": proper_refusals,
        "epistemic_refusal_rate": refusal_rate,
        "audio_average_score": audio_average,
        "audio_duration_seconds": duration
    }
    return issues, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate NotebookLM audit evidence for PS05.")
    parser.add_argument("evidence_file", nargs="?", default="audit_evidence_template.json")
    parser.add_argument("--create-template", action="store_true")
    args = parser.parse_args()

    evidence_path = Path(args.evidence_file)
    if args.create_template:
        if evidence_path.exists():
            print(f"Refusing to overwrite existing file: {evidence_path}")
            return 2
        evidence_path.write_text(
            json.dumps(template(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )
        print(f"Created blank evidence template: {evidence_path}")
        print("Complete it with actual NotebookLM responses, citations, spans, and audio evidence.")
        return 0

    if not evidence_path.exists():
        print(f"Evidence file not found: {evidence_path}")
        print("Create a blank template with: python 03-notebooklm-rag-audit.py --create-template")
        return 2

    try:
        data = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read valid JSON: {exc}")
        return 2

    issues, metrics = evaluate(data)
    print("=== PS05 NotebookLM RAG Audit: Evidence Evaluation ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nEvidence checks: {max(0, 1 - len(issues)) if not issues else 'INCOMPLETE'}")
    if issues:
        print(f"\nItems requiring evidence or correction ({len(issues)}):")
        for issue in issues:
            print(f"- {issue}")
        print("\nAudit status: INCOMPLETE — do not report unverified metrics as measured results.")
        return 1

    print("\nAudit status: EVIDENCE RECORDS COMPLETE.")
    print("The evaluator checks recorded evidence; human review is still required for entailment and scoring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
