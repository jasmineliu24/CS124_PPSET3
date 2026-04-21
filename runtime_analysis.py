#!/usr/bin/env python3
"""
Empirical runtime comparison for partition.py heuristics.

Usage:
  python3 runtime_analysis.py              # line plot vs n → runtime_graph.png
  python3 runtime_analysis.py --quick      # faster line plot
  python3 runtime_analysis.py --bar-chart  # avg time over 50 instances → runtime_bar_chart.png
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import time
from pathlib import Path

# Non-interactive backend; avoids GUI requirements in headless runs.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import partition as P


def random_instance(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(1, 10**12) for _ in range(n)]


def time_call(fn, *, repeats: int) -> float:
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def algorithm_runners():
    """Return (code, line_label, bar_label, runner) list; runner(A, n)."""
    return [
        (0, "KK", "KK", lambda A, n: P.kk(A)),
        (1, "RR (std)", "RR", lambda A, n: P.repeated_random(A, lambda: P.std_random_solution(n), P.std_residue)),
        (2, "HC (std)", "HC", lambda A, n: P.hill_climbing(A, lambda: P.std_random_solution(n), P.std_random_neighbor, P.std_residue)),
        (3, "SA (std)", "SA", lambda A, n: P.simulated_annealing(A, lambda: P.std_random_solution(n), P.std_random_neighbor, P.std_residue)),
        (11, "RR (pp)", "PPRR", lambda A, n: P.repeated_random(A, lambda: P.pp_random_solution(n), P.pp_residue)),
        (12, "HC (pp)", "PPHC", lambda A, n: P.hill_climbing(A, lambda: P.pp_random_solution(n), P.pp_random_neighbor, P.pp_residue)),
        (13, "SA (pp)", "PPSA", lambda A, n: P.simulated_annealing(A, lambda: P.pp_random_solution(n), P.pp_random_neighbor, P.pp_residue)),
    ]


def run_bar_chart(*, instances: int, n: int, quick: bool, out_path: Path) -> None:
    if quick:
        P.ITERATIONS = 3000
        inst = min(instances, 10)
    else:
        P.ITERATIONS = 25000
        inst = instances

    algorithms = algorithm_runners()
    avgs: list[float] = []
    labels: list[str] = []

    print(f"Bar chart: n={n}, instances={inst}, ITERATIONS={P.ITERATIONS}")
    print(f"{'algo':>6} | {'avg_s':>12}")
    print("-" * 22)

    for code, _line, bar, runner in algorithms:
        times_one: list[float] = []
        for k in range(inst):
            A = random_instance(n, seed=10_000 + k * 97 + code)
            t0 = time.perf_counter()
            runner(A, n)
            times_one.append(time.perf_counter() - t0)
        avg = statistics.mean(times_one)
        avgs.append(avg)
        labels.append(bar)
        print(f"{bar:>6} | {avg:12.6f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(labels))
    bars = ax.bar(
        x,
        avgs,
        color="#ff7f0e",
        edgecolor="#888888",
        linewidth=0.8,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average run time (s, log scale)")
    ax.set_title(f"Average algorithm runtime over {inst} instances")
    ax.set_yscale("log")
    if avgs:
        lo = max(1e-7, min(avgs) * 0.25)
        hi = max(avgs) * 1.2
        ax.set_ylim(lo, hi)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="faster, less smooth curves")
    ap.add_argument("--bar-chart", action="store_true", help="bar plot: mean time per algo over many instances")
    ap.add_argument("--instances", type=int, default=50, help="with --bar-chart, number of random A lists")
    ap.add_argument("--n", type=int, default=100, help="with --bar-chart, length of each A")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    if args.bar_chart:
        run_bar_chart(
            instances=args.instances,
            n=args.n,
            quick=args.quick,
            out_path=root / "runtime_bar_chart.png",
        )
        return

    if args.quick:
        ns = [20, 40, 80, 120]
        trials = 2
        P.ITERATIONS = 3000
    else:
        ns = [25, 50, 100, 150, 200, 300]
        trials = 3
        P.ITERATIONS = 8000

    algorithms = [(c, lab, fn) for c, lab, _b, fn in algorithm_runners()]

    # (code, label) -> list of median seconds per n
    series: dict[tuple[int, str], list[float]] = {}
    for code, label, _ in algorithms:
        series[(code, label)] = []

    print(f"ITERATIONS (patched for benchmark) = {P.ITERATIONS}")
    print(f"{'n':>6} | " + " | ".join(f"{lab:>12}" for _, lab, _ in algorithms))
    print("-" * (8 + 14 * len(algorithms)))

    for n in ns:
        row = [f"{n:>6}"]
        for code, label, runner in algorithms:
            A = random_instance(n, seed=1000 + n + code)

            def run_once():
                runner(A, n)

            sec = time_call(run_once, repeats=trials)
            series[(code, label)].append(sec)
            row.append(f"{sec:12.4f}")
        print(" | ".join(row))

    out = Path(__file__).resolve().parent / "runtime_graph.png"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {
        0: dict(color="black", linewidth=2.2, marker="o"),
        1: dict(color="#1f77b4", linestyle="--", marker="s"),
        2: dict(color="#ff7f0e", linestyle="--", marker="^"),
        3: dict(color="#2ca02c", linestyle="--", marker="v"),
        11: dict(color="#9467bd", linestyle="-.", marker="s"),
        12: dict(color="#d62728", linestyle="-.", marker="^"),
        13: dict(color="#8c564b", linestyle="-.", marker="v"),
    }
    for code, label, _ in algorithms:
        ax.plot(ns, series[(code, label)], label=label, **styles.get(code, {}))

    ax.set_xlabel("Input size n (number of integers)")
    ax.set_ylabel(f"Wall time (s), ITERATIONS={P.ITERATIONS}")
    ax.set_title("Empirical runtime: partition heuristics")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
