"""Read-only checks for input freezing, request freshness and phase ordering."""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from rag_fresh_change_experiment import (PHASES, digest, dump, load, matrices,
                                         observations, verified_inputs)


def audit(output):
    manifest, _, plan = verified_inputs(output)
    mismatches = [p for p, expected in manifest["input_hashes"].items()
                  if digest(Path(p)) != expected]
    if mismatches:
        raise ValueError("frozen input/code/model changed")
    lock = load(output / "calibration_lock.json")
    records = {phase: observations(output, phase) for phase in PHASES}
    all_records = [r for phase in PHASES for r in records[phase]]
    if len(all_records) != len(plan):
        raise ValueError("incomplete request plan")
    ids = [r["response_id"] for r in all_records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate responses")
    calibration = records["mean"] + records["threshold"]
    tests = records["normal_test"] + records["fault_test"]
    if max(r["finished_at"] for r in calibration) > lock["locked_at"]:
        raise ValueError("calibration not complete at lock")
    if min(r["started_at"] for r in tests) <= lock["locked_at"]:
        raise ValueError("test generated before calibration lock")
    if digest(output / "context_relevance.json") != lock["frozen"]["relevance_sha256"]:
        raise ValueError("auxiliary signal changed after lock")
    for rid, expected in lock["frozen"]["calibration_response_hashes"].items():
        if digest(output / "responses" / (rid + ".json")) != expected:
            raise ValueError("calibration response changed")
    for phase in PHASES[1:]:
        matrices(records[phase])  # Verifies serial chronological requests per stream.
    models = Counter(r["actual_model"] for r in all_records)
    if len(models) != 1:
        raise ValueError("cloud model identifier changed")
    value = dict(status="passed", checked_at=datetime.now(timezone.utc).isoformat(),
                 input_hashes_checked=len(manifest["input_hashes"]),
                 response_files_checked=len(all_records), unique_response_ids=len(set(ids)),
                 calibration_lock_precedes_every_test=True, chronologically_serial_within_stream=True,
                 phase_counts={phase: len(records[phase]) for phase in PHASES},
                 actual_models=dict(models), python_executable=sys.executable,
                 package_versions={name: version(name) for name in
                                   ("numpy", "scipy", "matplotlib", "openai", "torch", "transformers")},
                 limitation="Distinct API requests do not prove independence or natural-traffic validity")
    dump(output / "integrity_audit.json", value)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    audit(ap.parse_args().output)
