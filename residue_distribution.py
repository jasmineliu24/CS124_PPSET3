#!/usr/bin/env python3
"""
Figure 2 style: box plots of residues over many random instances.

Runs KK, six heuristics with random starts, and the same six with KK-derived
initial solutions (signs from the KK merge tree; prepartition from two labels).

Usage:
  python3 residue_distribution.py              # 50 instances, 25000 iterations (slow)
  python3 residue_distribution.py --quick      # fewer iterations / instances for a smoke test
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import partition as P


def randomInstance(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(1, 10**12) for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=50, help="number of random A lists")
    ap.add_argument("--n", type=int, default=100, help="length of each A (pset uses 100)")
    ap.add_argument("--iterations", type=int, default=25000, help="heuristic iteration budget")
    ap.add_argument("--quick", action="store_true", help="small n of instances and iterations")
    ap.add_argument("--seed", type=int, default=4242, help="RNG seed for instance generation")
    args = ap.parse_args()

    instances = args.instances
    n = args.n
    iterations = args.iterations
    if args.quick:
        instances = min(instances, 15)
        iterations = min(iterations, 3000)

    P.defaultIterations = iterations

    labels = [
        "KK",
        "Std RR",
        "Std HC",
        "Std SA",
        "PP RR",
        "PP HC",
        "PP SA",
        "Std RR\n(KK)",
        "Std HC\n(KK)",
        "Std SA\n(KK)",
        "PP RR\n(KK)",
        "PP HC\n(KK)",
        "PP SA\n(KK)",
    ]
    data: list[list[float]] = [[] for _ in labels]

    print(f"instances={instances}, n={n}, defaultIterations={P.defaultIterations}")
    for k in range(instances):
        A = randomInstance(n, seed=args.seed + k * 97)
        stdSol = lambda: P.stdRandomSolution(n)
        stdNb = lambda S: P.stdRandomNeighbor(S)
        stdRes = lambda S, arr: P.stdResidue(S, arr)
        ppSol = lambda: P.ppRandomSolution(n)
        ppNb = lambda Pr: P.ppRandomNeighbor(Pr)
        ppRes = lambda Pr, arr: P.ppResidue(Pr, arr)

        kkSigns = P.initialSignsFromKk(A)
        kkPre = P.initialPrepartitionFromSigns(kkSigns)

        data[0].append(float(P.kk(A)))
        data[1].append(float(P.repeatedRandom(A, stdSol, stdRes)))
        data[2].append(float(P.hillClimbing(A, stdSol, stdNb, stdRes)))
        data[3].append(float(P.simulatedAnnealing(A, stdSol, stdNb, stdRes)))
        data[4].append(float(P.repeatedRandom(A, ppSol, ppRes)))
        data[5].append(float(P.hillClimbing(A, ppSol, ppNb, ppRes)))
        data[6].append(float(P.simulatedAnnealing(A, ppSol, ppNb, ppRes)))
        data[7].append(float(P.repeatedRandom(A, stdSol, stdRes, initialSolution=kkSigns)))
        data[8].append(float(P.hillClimbing(A, stdSol, stdNb, stdRes, initialSolution=kkSigns)))
        data[9].append(float(P.simulatedAnnealing(A, stdSol, stdNb, stdRes, initialSolution=kkSigns)))
        data[10].append(float(P.repeatedRandom(A, ppSol, ppRes, initialSolution=kkPre)))
        data[11].append(float(P.hillClimbing(A, ppSol, ppNb, ppRes, initialSolution=kkPre)))
        data[12].append(float(P.simulatedAnnealing(A, ppSol, ppNb, ppRes, initialSolution=kkPre)))

        if (k + 1) % max(1, instances // 10) == 0 or k + 1 == instances:
            print(f"  finished instance {k + 1}/{instances}")

    # Log-scale box plots: clip zeros to 0.9 so log axis is defined
    plotData = [[max(v, 0.9) for v in col] for col in data]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    positions = range(1, len(labels) + 1)
    bp = ax.boxplot(
        plotData,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
    )
    colors = [
        "#4a148c",
        "#f8bbd9",
        "#f8bbd9",
        "#f8bbd9",
        "#ce93d8",
        "#ce93d8",
        "#ce93d8",
        "#7b1fa2",
        "#7b1fa2",
        "#7b1fa2",
        "#ba68c8",
        "#ba68c8",
        "#ba68c8",
    ]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for median in bp["medians"]:
        median.set_color("#212121")
        median.set_linewidth(1.2)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Residue (log scale)")
    ax.set_yscale("log")
    ax.set_title("Distribution of Residues Across Instances")
    ax.grid(True, which="both", axis="y", alpha=0.35)
    fig.text(
        0.5,
        0.01,
        "Figure 2: Distribution of residues across instances (log scale). "
        f"defaultIterations={P.defaultIterations}, n={n}.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    out = Path(__file__).resolve().parent / "residue_boxplot.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
