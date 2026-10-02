"""Save code and aggregate prospective experiment artifacts to the local repo.

Question banks, answers, source snippets, private references and API records stay
in the active experiment directory. This performs no Git commit or network push.
"""
import argparse
import shutil
from pathlib import Path

from rag_fresh_change_experiment import load, dump, digest, utc
from rag_prospective_coverage_validation import verify_hashes


CODE_FILES = [
    "rag_prospective_objective_validation.py", "rag_exact_source_cards.py",
    "rag_exact_source_validation.py", "audit_rag_prospective_objective.py",
    "report_rag_prospective_objective.py", "rag_numeric_unit_mapping.py",
    "rag_objective_admission_guard.py", "diagnose_rag_prospective_validity.py",
    "sync_rag_prospective_artifacts.py", "tests/test_rag_prospective_objective_validation.py",
    "tests/test_rag_exact_source_cards.py", "tests/test_rag_numeric_unit_mapping.py",
    "tests/test_rag_objective_admission_guard.py",
]
AGGREGATES = [
    "analysis_summary.json", "integrity_audit.json", "verification_results.json",
    "candidate_screening.csv", "source_exposure.csv", "answer_scores.csv",
    "signal_metrics.csv", "proxy_basis_availability.csv", "live_source_groups.csv",
    "validity_review/validity_summary.json", "validity_review/validity_diagnostic.csv",
]


def sync(project, output, legacy, backup):
    project, output, legacy, backup = [p.resolve() for p in (project, output, legacy, backup)]
    if not output.is_relative_to(project) or not legacy.is_relative_to(project):
        raise ValueError("experiment paths must stay within active project")
    if backup == project or backup.is_relative_to(project) or project.is_relative_to(backup):
        raise ValueError("backup must be a separate project directory")
    review = load(output / "validity_review/review_inputs_lock.json")
    verify_hashes(review["hashes"])
    verify_hashes(review["code_hashes"])
    copied = []
    asset = "rag_prospective_v7_1_qualified_20261002"
    aggregates_root = backup / "docs/experiments/prospective_exact_source_v7_1_20261002"

    def save(source, destination, content=None):
        destination = destination.resolve()
        if not destination.is_relative_to(backup):
            raise ValueError("destination escapes backup project")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            shutil.copy2(source, destination)
            if digest(source) != digest(destination):raise ValueError("backup differs from source")
        else:
            destination.write_text(content, encoding="utf-8")
        copied.append(dict(source=str(source.resolve()) if source else "generated_index",
                           destination=str(destination), sha256=digest(destination),
                           transformed=content is not None))

    for name in CODE_FILES:save(project / name, backup / name)
    for name in AGGREGATES:save(output / name, aggregates_root / name)
    for name in ("preparation_disposition.json", "source_integrity_screening.csv"):
        save(legacy / name, aggregates_root / ("first_preparation_" + name))
    for ext in ("png", "svg"):
        save(output / "validity_review" / ("qualified_primary_results." + ext),
             backup / "docs/assets" / (asset + "." + ext))
    replacements = {
        (output / "validity_review/qualified_primary_results.png").as_posix(): "assets/" + asset + ".png",
        (output / "prospective_objective_comparison.png").as_posix(): "assets/" + asset + ".png",
        (output / "prospective_objective_report.md").as_posix(): "rag_prospective_v7_1_primary_report_20261002.md",
        (output / "integrity_audit.json").as_posix(): "experiments/prospective_exact_source_v7_1_20261002/integrity_audit.json",
        (output / "validity_review/validity_summary.json").as_posix(): "experiments/prospective_exact_source_v7_1_20261002/validity_review/validity_summary.json",
    }
    reports = [
        (project / "log/前瞻新题验证V7_1协议_20261002.md", "rag_prospective_v7_1_protocol_20261002.md"),
        (output / "prospective_objective_report.md", "rag_prospective_v7_1_primary_report_20261002.md"),
        (output / "validity_review/prospective_validity_report.md", "rag_prospective_v7_1_validity_report_20261002.md"),
    ]
    for source, name in reports:
        content = source.read_text(encoding="utf-8")
        for original, target in replacements.items():content = content.replace(original, target)
        if "primary_report" in name:
            content = "> 后续有效性复查发现题干/参考语义问题；本文件保留原始数值成绩，不能作为全部题目真实正确的结论。请先读[有效性复查](rag_prospective_v7_1_validity_report_20261002.md)。\n\n" + content
        save(source, backup / "docs" / name, content)
    index = """# 前瞻新题验证V7.1归档

原始数值门槛通过；参考有效性仍需修订，不代表16题独立确认全部正确。

- [最终有效性复查与后续顺序](../../rag_prospective_v7_1_validity_report_20261002.md)
- [原始数值报告](../../rag_prospective_v7_1_primary_report_20261002.md)
- [生成回答前的预设协议](../../rag_prospective_v7_1_protocol_20261002.md)

16道新题/8篇论文，32条遗漏或改错合成对照检出28条，4条未定；完整对照误报0/16，含2条代理未定。离线开发修复后依据有效15/16，P0001继续被语义准入拦下；此数字不是新前瞻成绩。相关测试30项通过。

原始题目、证据、标准答案、回答和API记录保存在活动项目；本地仓库保存代码、测试、图和汇总。此次没有提交或推送GitHub。
"""
    save(None, aggregates_root / "README.md", index)
    manifest = dict(saved_at=utc(), backup=str(backup), copied_files=len(copied), files=copied,
                    primary_gate="passed", semantic_validity="needs_revision_before_confirmatory_use",
                    raw_evidence_questions_answers_references_and_api_not_copied=True,
                    git_commit=False, github_push=False)
    dump(output / "backup_manifest.json", manifest)
    dump(aggregates_root / "backup_manifest.json", manifest)
    artifacts = [p for p in output.rglob("*") if p.is_file() and "mplconfig" not in p.parts
                 and p.name not in {"artifact_manifest.json"} and p.suffix != ".tmp"]
    code = [project / name for name in CODE_FILES]
    dump(output / "artifact_manifest.json", dict(saved_at=utc(),
         output_hashes={str(p.resolve()):digest(p) for p in sorted(artifacts)},
         code_hashes={str(p.resolve()):digest(p) for p in code},
         checks=load(output / "verification_results.json"),
         primary_numeric_gate="passed", semantic_validity="needs_revision_before_confirmatory_use",
         all_primary_cases_and_unresolved_preserved=True,
         development_replay_must_not_replace_primary_results=True))
    print("Saved", len(copied), "backup files;", len(artifacts), "output hashes;", len(code), "code hashes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "output", "legacy", "backup"):parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sync(args.project, args.output, args.legacy, args.backup)
