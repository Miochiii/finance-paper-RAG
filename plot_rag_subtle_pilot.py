"""Scientific figure from completed pilot summaries, no API calls."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rag_fresh_change_experiment import load
from rag_subtle_fault_pilot import CONDITIONS


def plot(output):
    summary = load(output / "analysis_summary.json")["condition_summary"]
    labels = ["Baseline", "Quote deletion", "Topical donor", "Retained prefix"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    positions = np.arange(len(CONDITIONS))
    bottom = np.zeros(len(CONDITIONS))
    for quality, color, name in (("2", "#3678a6", "Complete (2)"), ("1", "#e7b353", "Partial (1)"),
                                 ("0", "#bf5b5b", "Wrong/unanswered (0)"), ("U", "#aaaaaa", "Uncertain")):
        values = [summary[c]["quality_counts"].get(quality, 0) for c in CONDITIONS]
        axes[0].bar(positions, values, bottom=bottom, color=color, label=name)
        bottom += values
    axes[0].set_ylabel("Answers (same 11 questions per condition)")
    axes[0].set_ylim(0, bottom.max()+1)
    axes[0].set_title("Automatic quality; uncertainty retained")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].bar(positions-.18, [summary[c]["gap512_mean"] for c in CONDITIONS], .36,
                 label="512 tokens (all truncated)", color="#8c719d")
    axes[1].bar(positions+.18, [summary[c]["gap4096_mean"] for c in CONDITIONS], .36,
                 label="4096 tokens (no truncation)", color="#559f8d")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Mean 1-sigmoid(relevance logit)")
    axes[1].set_title("Context relevance gap; not correctness")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(positions, labels, rotation=15)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Frozen paired mechanism pilot; quote deletion may leave semantic evidence")
    fig.tight_layout()
    fig.savefig(output / "subtle_fault_summary.png", dpi=170)
    fig.savefig(output / "subtle_fault_summary.svg")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    plot(ap.parse_args().output)
