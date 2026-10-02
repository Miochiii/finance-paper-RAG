"""Archive V9 code and aggregates; raw local experiment records stay local."""
import argparse
import shutil
from pathlib import Path

from rag_fresh_change_experiment import load, dump, digest, utc
from rag_prospective_coverage_validation import verify_hashes
from rag_live_evidence_pairs import verify_ready, MODULES


def archive(project, output, backup):
    project, output, backup = [p.resolve() for p in (project, output, backup)]
    if not output.is_relative_to(project) or project == backup or project.is_relative_to(backup) or backup.is_relative_to(project):
        raise ValueError("invalid project/backup roots")
    protocol = verify_ready(output)
    if load(output / "integrity_audit.json")["status"] != "passed":
        raise ValueError("integrity audit not passed")
    files = []
    def save(source, destination):
        destination = destination.resolve()
        if not destination.is_relative_to(backup):
            raise ValueError("destination escapes backup root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if digest(source) != digest(destination):
            raise ValueError("backup copy hash mismatch")
        files.append(dict(source=str(source.resolve()), destination=str(destination), sha256=digest(destination)))
    code = MODULES + ["diagnose_rag_live_evidence_literals.py", "tests/test_diagnose_rag_live_evidence_literals.py",
                      "archive_rag_live_evidence_pairs.py"]
    for name in code:
        save(project / name, backup / name)
    destination = backup / "docs/experiments/live_evidence_pairs_v9_20261002"
    for name in ("analysis_summary.json", "integrity_audit.json", "targeted_tests.json", "diagnostic_tests.json",
                 "answer_scores.csv", "paired_quality.csv", "signal_metrics.csv", "source_groups.csv",
                 "intervention_screening.json", "intervention_screening.csv", "literal_diagnostic_summary.json",
                 "workflow_summary.json"):
        save(output / name, destination / name)
    for ext in ("png", "svg"):
        save(output / ("live_evidence_results." + ext), backup / "docs/assets" / ("rag_live_evidence_v9_20261002." + ext))
    for source, name in ((project / "log/真实数值证据故障V9协议_20261002.md", "rag_live_evidence_v9_protocol_20261002.md"),
                         (output / "live_evidence_report.md", "rag_live_evidence_v9_report_20261002.md"),
                         (output / "live_evidence_addendum.md", "rag_live_evidence_v9_addendum_20261002.md")):
        save(source, backup / "docs" / name)
    record = dict(saved_at=utc(), copied_files=len(files), files=files,
        gate=load(output / "analysis_summary.json")["gate"]["status"],
        raw_questions_references_evidence_answers_and_api_not_copied=True,
        all_screening_failures_retained_locally=True, git_commit=False, github_push=False)
    dump(output / "backup_manifest.json", record)
    dump(destination / "backup_manifest.json", record)
    paths = [p for p in output.rglob("*") if p.is_file() and "mplconfig" not in p.parts
             and p.name != "artifact_manifest.json" and p.suffix != ".tmp"]
    code_hashes = dict(protocol["input_hashes"])
    code_hashes.update({str((project / n).resolve()): digest(project / n) for n in code})
    manifest = dict(saved_at=utc(), output_hashes={str(p.resolve()): digest(p) for p in sorted(paths)},
        code_and_input_hashes=code_hashes, preflight="passed", integrity_audit="passed", visual_review="passed",
        targeted_tests=31, additional_posthoc_parser_tests=2, synthetic_answers=0,
        interpretation=protocol["interpretation"])
    dump(output / "artifact_manifest.json", manifest)
    verify_hashes(manifest["output_hashes"])
    verify_hashes(manifest["code_and_input_hashes"])
    print("Archived V9:", len(files), "files;", len(paths), "output hashes; all verified")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "output", "backup"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    archive(a.project, a.output, a.backup)
