#!/usr/bin/env python3
"""
Algorithms:
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

ITERATIONS = 25000


#  Karmarkar-Karp (KK)

def kk(A):
    """
    Karmarkar-Karp algorithm using max-heap
    Replace larger with difference between 2 elements
    Replace smaller with 0
    O(n log n) time
    """
    # Python's heapq is a min-heap, so negate values for max-heap behavior
    heap = [-x for x in A]
    heapq.heapify(heap)

    while len(heap) > 1:
        larger = -heapq.heappop(heap)
        smaller  = -heapq.heappop(heap)
        d = larger - smaller
        heapq.heappush(heap, -d)

    return abs(-heap[0])


# ─────────────────────────────────────────────
#  Standard representation helpers
#  S[i] in {+1, -1}, residue = |sum(S[i]*A[i])|
# ─────────────────────────────────────────────
def std_random_solution(n):
    return [random.choice([-1, 1]) for _ in range(n)]


def std_random_neighbor(S):
    """
    Flip S[i]; with prob 1/2 also flip S[j], i != j.
    Returns a new list (does not mutate S).
    """
    S = S[:]
    n = len(S)
    i, j = random.sample(range(n), 2)
    S[i] = -S[i]
    if random.random() < 0.5:
        S[j] = -S[j]
    return S


def std_residue(S, A):
    return abs(sum(s * a for s, a in zip(S, A)))


#  Prepartition representation helpers
# - turn A into A' using P
# - run KK on A'
# - return residue of A'

def pp_random_solution(n):
    return [random.randint(1, n) for _ in range(n)]


def pp_random_neighbor(P):
    """
    Choose random index i and new group j != P[i]
    Return new list
    """
    P = P[:]
    n = len(P)
    i = random.randrange(n)
    j = random.randint(1, n - 1)
    if j >= P[i]:
        j += 1
    P[i] = j
    return P


def pp_residue(P, A):
    """
    Build A' by summing elements that share the same partition label,
    then run KK on A'.
    """
    n = len(A)
    A_prime = [0] * n
    for j in range(n):
        A_prime[P[j] - 1] += A[j]   # P[j] is 1-indexed
    return kk(A_prime)


# ─────────────────────────────────────────────
#  Cooling schedule for simulated annealing
# ─────────────────────────────────────────────
def temperature(iteration):
    return 1e10 * (0.8 ** (iteration // 300))


# ─────────────────────────────────────────────
#  Generic algorithm implementations
#  Each takes: A, random_solution(), random_neighbor(), residue()
# ─────────────────────────────────────────────
def repeated_random(A, rand_sol, residue_fn):
    S = rand_sol()
    best_res = residue_fn(S, A)

    for _ in range(ITERATIONS):
        S_prime = rand_sol()
        r_prime = residue_fn(S_prime, A)
        if r_prime < best_res:
            S = S_prime
            best_res = r_prime

    return best_res


def hill_climbing(A, rand_sol, rand_neighbor, residue_fn):
    S = rand_sol()
    best_res = residue_fn(S, A)

    for _ in range(ITERATIONS):
        S_prime = rand_neighbor(S)
        r_prime = residue_fn(S_prime, A)
        if r_prime < best_res:
            S = S_prime
            best_res = r_prime

    return best_res


def simulated_annealing(A, rand_sol, rand_neighbor, residue_fn):
    S = rand_sol()
    cur_res = residue_fn(S, A)
    best_res = cur_res
    S_best = S

    for it in range(1, ITERATIONS + 1):
        S_prime = rand_neighbor(S)
        r_prime = residue_fn(S_prime, A)
        delta = r_prime - cur_res

        if delta < 0:
            S = S_prime
            cur_res = r_prime
        else:
            T = temperature(it)
            if T > 0 and random.random() < math.exp(-delta / T):
                S = S_prime
                cur_res = r_prime

        if cur_res < best_res:
            S_best = S
            best_res = cur_res

    return best_res


# ─────────────────────────────────────────────
#  Main dispatcher
# ─────────────────────────────────────────────
def main():
    if len(sys.argv) != 4:
        print("Usage: python3 partition.py <flag> <algorithm> <inputfile>",
              file=sys.stderr)
        sys.exit(1)

    _flag      = int(sys.argv[1])   # reserved for your own use
    algorithm  = int(sys.argv[2])
    inputfile  = sys.argv[3]

    with open(inputfile) as f:
        A = [int(line.strip()) for line in f if line.strip()]

    n = len(A)

    # Bind representation-specific helpers for cleaner dispatch
    std_sol  = lambda: std_random_solution(n)
    std_nb   = lambda S: std_random_neighbor(S)
    std_res  = lambda S, A: std_residue(S, A)

    pp_sol   = lambda: pp_random_solution(n)
    pp_nb    = lambda P: pp_random_neighbor(P)
    pp_res   = lambda P, A: pp_residue(P, A)

    if algorithm == 0:
        result = kk(A)

    elif algorithm == 1:
        result = repeated_random(A, std_sol, std_res)

    elif algorithm == 2:
        result = hill_climbing(A, std_sol, std_nb, std_res)

    elif algorithm == 3:
        result = simulated_annealing(A, std_sol, std_nb, std_res)

    elif algorithm == 11:
        result = repeated_random(A, pp_sol, pp_res)

    elif algorithm == 12:
        result = hill_climbing(A, pp_sol, pp_nb, pp_res)

    elif algorithm == 13:
        result = simulated_annealing(A, pp_sol, pp_nb, pp_res)

    else:
        print(f"Unknown algorithm code: {algorithm}", file=sys.stderr)
        sys.exit(1)

    print(result)


if __name__ == "__main__":
    main()