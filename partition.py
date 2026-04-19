#!/usr/bin/env python3
"""
Algorithms (like the guidelines) :
  0  Karmarkar-Karp
  1  Repeated Random
  2  Hill Climbing
  3  Simulated Annealing
  11 Prepartitioned Repeated Random
  12 Prepartitioned Hill Climbing
  13 Prepartitioned Simulated Annealing
"""

import sys
import heapq
import random
import math

defaultIterations = 25000


#  Karmarkar-Karp (KK)

def kk(A):
    # Karmarkar-Karp: min-heap on negated values acts as max-heap.
    if not A:
        return 0
    heap = [-x for x in A]
    heapq.heapify(heap)

    def earlyExitResidue():
        # If at most one positive value remains (rest are zeros), no further differencing can change the residue
        # if all zeros, residue is 0.
        values = [-x for x in heap]
        positives = [v for v in values if v > 0]
        if len(positives) <= 1:
            return max(values)
        return None

    r = earlyExitResidue()
    if r is not None:
        return r

    while len(heap) > 1:
        larger = -heapq.heappop(heap)
        smaller = -heapq.heappop(heap)
        d = larger - smaller
        heapq.heappush(heap, -d)
        r = earlyExitResidue()
        if r is not None:
            return r

    return abs(-heap[0])



def stdRandomSolution(n):
    return [random.choice([-1, 1]) for _ in range(n)]


def stdRandomNeighbor(S):

    # Flip S[i]; with prob 1/2 also flip S[j], i != j.

    S = S[:]
    n = len(S)
    i, j = random.sample(range(n), 2)
    S[i] = -S[i]
    if random.random() < 0.5:
        S[j] = -S[j]
    return S


def stdResidue(S, A):
    return abs(sum(s * a for s, a in zip(S, A)))


#  Prepartition representation helpers
# - turn A into A' using P
# - run KK on A'
# - return residue of A'

def ppRandomSolution(n):
    return [random.randint(1, n) for _ in range(n)]


def ppRandomNeighbor(P):
    # Choose random index i and new group j != P[i], return new list

    P = P[:]
    n = len(P)
    i = random.randrange(n)
    current = P[i]
    choices = [label for label in range(1, n + 1) if label != current]
    j = random.choice(choices)
    P[i] = j
    return P


def ppResidue(P, A):
    # Build A' by summing elements that share the same partition label, then run KK on A'.

    n = len(A)
    aPrime = [0] * n
    for j in range(n):
        aPrime[P[j] - 1] += A[j]   # P[j] is 1-indexed
    return kk(aPrime)


#  Cooling schedule for simulated annealing
def temperature(iteration):
    return 1e10 * (0.8 ** (iteration // 300))

#  Generic algorithm implementations

def repeatedRandom(A, randSol, residueFn):
    S = randSol()
    bestRes = residueFn(S, A)

    for _ in range(defaultIterations):
        sPrime = randSol()
        rPrime = residueFn(sPrime, A)
        if rPrime < bestRes:
            S = sPrime
            bestRes = rPrime

    return bestRes


def hillClimbing(A, randSol, randNeighbor, residueFn):
    S = randSol()
    bestRes = residueFn(S, A)

    for _ in range(defaultIterations):
        sPrime = randNeighbor(S)
        rPrime = residueFn(sPrime, A)
        if rPrime < bestRes:
            S = sPrime
            bestRes = rPrime

    return bestRes


def simulatedAnnealing(A, randSol, randNeighbor, residueFn):
    S = randSol()
    curRes = residueFn(S, A)
    bestRes = curRes

    for it in range(1, defaultIterations + 1):
        sPrime = randNeighbor(S)
        rPrime = residueFn(sPrime, A)
        delta = rPrime - curRes

        if delta < 0:
            S = sPrime
            curRes = rPrime
        else:
            T = temperature(it)
            if T > 0 and random.random() < math.exp(-delta / T):
                S = sPrime
                curRes = rPrime

        if curRes < bestRes:
            bestRes = curRes

    return bestRes


#  Main to run the experiements
def main():
    if len(sys.argv) != 4:
        print("Usage: python3 partition.py <flag> <algorithm> <inputfile>",
              file=sys.stderr)
        sys.exit(1)

    flag = int(sys.argv[1])   # reserved for your own use
    algorithm = int(sys.argv[2])
    inputFile = sys.argv[3]

    with open(inputFile) as f:
        A = [int(line.strip()) for line in f if line.strip()]

    n = len(A)

    # Helpers for cleaning experiments
    stdSol = lambda: stdRandomSolution(n)
    stdNb = lambda S: stdRandomNeighbor(S)
    stdRes = lambda S, A: stdResidue(S, A)

    ppSol = lambda: ppRandomSolution(n)
    ppNb = lambda P: ppRandomNeighbor(P)
    ppRes = lambda P, A: ppResidue(P, A)

    if algorithm == 0:
        result = kk(A)

    elif algorithm == 1:
        result = repeatedRandom(A, stdSol, stdRes)

    elif algorithm == 2:
        result = hillClimbing(A, stdSol, stdNb, stdRes)

    elif algorithm == 3:
        result = simulatedAnnealing(A, stdSol, stdNb, stdRes)

    elif algorithm == 11:
        result = repeatedRandom(A, ppSol, ppRes)

    elif algorithm == 12:
        result = hillClimbing(A, ppSol, ppNb, ppRes)

    elif algorithm == 13:
        result = simulatedAnnealing(A, ppSol, ppNb, ppRes)

    else:
        print(f"Unknown algorithm code: {algorithm}", file=sys.stderr)
        sys.exit(1)

    print(result)


if __name__ == "__main__":
    main()
