"""Generate benchmark charts (PNG) from the CSV/JSON raw data.

Writes reports/benchmarks/charts/*.png used by docs/BENCHMARKS.md.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robotstudio.config import BENCH_ROOT

CHART_DIR = BENCH_ROOT / "charts"
SCENARIO_LABELS = {"empty": "Empty scene", "single_obs": "Single obstacle", "narrow": "Narrow passage"}
PLANNER_LABELS = {"linear": "Linear", "rrt": "RRT", "rrt_connect": "RRT-Connect"}
COLORS = {"linear": "#8a9bab", "rrt": "#e6c07b", "rrt_connect": "#7ed491"}


def _load_rows(scenario: str) -> list[dict]:
    path = BENCH_ROOT / f"planning_{scenario}.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def chart_success_rates() -> None:
    scenarios = ["empty", "single_obs", "narrow"]
    rows_by = {s: _load_rows(s) for s in scenarios}
    planners = ["linear", "rrt", "rrt_connect"]
    x = np.arange(len(scenarios))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    for k, p in enumerate(planners):
        rates = []
        for s in scenarios:
            rs = [r for r in rows_by[s] if r["planner"] == p]
            rates.append(100.0 * sum(1 for r in rs if r["success"] == "True") / len(rs) if rs else 0)
        bars = ax.bar(x + (k - 1) * width, rates, width, label=PLANNER_LABELS[p], color=COLORS[p])
        for b, v in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios])
    ax.set_ylabel("success rate [%]")
    ax.set_ylim(0, 108)
    ax.set_title("Planning success rate (30 seeds per scenario, 8 s budget)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_DIR / "planning_success_rate.png")
    plt.close(fig)


def chart_planning_times() -> None:
    scenarios = ["empty", "single_obs", "narrow"]
    rows_by = {s: _load_rows(s) for s in scenarios}
    planners = ["linear", "rrt", "rrt_connect"]
    data, positions, labels, colors = [], [], [], []
    pos = 1.0
    for s in scenarios:
        for p in planners:
            rs = [float(r["time_s"]) for r in rows_by[s] if r["planner"] == p and r["success"] == "True"]
            if rs:
                data.append(rs)
                positions.append(pos)
                labels.append(PLANNER_LABELS[p])
                colors.append(COLORS[p])
            pos += 1.0
        pos += 1.0
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    bp = ax.boxplot(data, positions=positions, widths=0.7, showfliers=True, patch_artist=True, medianprops=dict(color="#1e2227"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    ax.set_yscale("log")
    ax.set_ylabel("planning time [s] (successful runs)")
    ticks, tick_labels = [], []
    seen = set()
    for pos_, lab in zip(positions, labels):
        if lab not in seen:
            pass
    # group labels per scenario
    for i, s in enumerate(scenarios):
        center = positions[i * 3 + 1] if i * 3 + 1 < len(positions) else positions[-1]
        ticks.append((positions[i * 3] + positions[min(i * 3 + 2, len(positions) - 1)]) / 2)
        tick_labels.append(SCENARIO_LABELS[s])
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[p]) for p in planners]
    ax.legend(handles, [PLANNER_LABELS[p] for p in planners])
    ax.set_title("Planning time distribution (fixed seeds 100-129)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "planning_times.png")
    plt.close(fig)


def chart_ik_times() -> None:
    path = BENCH_ROOT / "ik_benchmark.csv"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["kind"] == "reachable" and r["success"] == "True"]
    times = np.array([float(r["time_ms"]) for r in rows])
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    ax.hist(times, bins=30, color="#7fb4e6", edgecolor="#1e2227")
    ax.axvline(np.mean(times), color="#e6c07b", label=f"mean {np.mean(times):.1f} ms")
    ax.axvline(np.percentile(times, 95), color="#ef7b7b", label=f"p95 {np.percentile(times, 95):.1f} ms")
    ax.set_xlabel("IK solve time [ms] (multi-start DLS)")
    ax.set_ylabel("count")
    ax.set_title("IK runtime over 120 reachable targets")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_DIR / "ik_times.png")
    plt.close(fig)


def chart_grasp_summary() -> None:
    path = BENCH_ROOT / "grasp_summary.json"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
    kinds = [k for k in ("base", "jitter", "hard_place") if k in summary]
    rates = [100 * summary[k]["success_rate"] for k in kinds]
    ns = [summary[k]["n"] for k in kinds]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    bars = ax.bar(kinds, rates, color=["#7ed491", "#7fb4e6", "#e6c07b"])
    for b, v, n in zip(bars, rates, ns):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}% (n={n})", ha="center", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("physical success rate [%]")
    ax.set_title("Pick&place success (physical criteria, 3x realtime)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_DIR / "grasp_success.png")
    plt.close(fig)


def main() -> None:
    chart_success_rates()
    chart_planning_times()
    chart_ik_times()
    chart_grasp_summary()
    print(f"charts written to {CHART_DIR}")


if __name__ == "__main__":
    main()
