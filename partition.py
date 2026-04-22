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
import random
import math

defaultIterations = 25000


# Binary min-heap (list-based, 0-based indexing) for Karmarkar-Karp
# I've implemented minHeap before, so it felt easier to just implement that
# and then negate values to be a max heap

def minHeapSiftDown(heap, index):
    # Restore min-heap property assuming children of index are valid heaps.
    length = len(heap)
    while True:
        smallest = index
        left = 2 * index + 1
        right = 2 * index + 2
        if left < length and heap[left] < heap[smallest]:
            smallest = left
        if right < length and heap[right] < heap[smallest]:
            smallest = right
        if smallest == index:
            break
        heap[index], heap[smallest] = heap[smallest], heap[index]
        index = smallest


def minHeapSiftUp(heap, index):
    # min-heap property holds
    while index > 0:
        parent = (index - 1) // 2
        if heap[index] < heap[parent]:
            heap[index], heap[parent] = heap[parent], heap[index]
            index = parent
        else:
            break


def minHeapify(heap):
    # Turn heap into a min-heap in O(n)
    n = len(heap)
    for i in range(n // 2 - 1, -1, -1):
        minHeapSiftDown(heap, i)


def minHeappush(heap, item):
    heap.append(item)
    minHeapSiftUp(heap, len(heap) - 1)


def minHeappop(heap):
    # Remove and return the smallest element
    last = len(heap) - 1
    if last < 0:
        raise IndexError("pop from empty heap")
    if last == 0:
        return heap.pop()
    root = heap[0]
    heap[0] = heap.pop()
    minHeapSiftDown(heap, 0)
    return root


#  Karmarkar-Karp (KK)

def kk(A):
    # Karmarkar-Karp: min-heap on negated values acts as max-heap.
    if not A:
        return 0
    heap = [-x for x in A]
    minHeapify(heap)

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
        larger = -minHeappop(heap)
        smaller = -minHeappop(heap)
        d = larger - smaller
        minHeappush(heap, -d)
        r = earlyExitResidue()
        if r is not None:
            return r

    return abs(-heap[0])


class KkMergeNode:
    # Merge tree for Karmarkar–Karp
    __slots__ = ("value", "leafIndex", "high", "low")

    def __init__(self, value, leafIndex=None, high=None, low=None):
        self.value = value
        self.leafIndex = leafIndex
        self.high = high
        self.low = low

    def isLeaf(self):
        return self.leafIndex is not None


def initialSignsFromKk(A):
    # get the sign vector S
    n = len(A)
    if n == 0:
        return []
    if n == 1:
        return [1]
    nodes = {}
    heap = []
    nextUid = 0
    for i, val in enumerate(A):
        nodes[nextUid] = KkMergeNode(val, leafIndex=i)
        minHeappush(heap, (-val, nextUid))
        nextUid += 1
    while len(heap) > 1:
        negV1, id1 = minHeappop(heap)
        negV2, id2 = minHeappop(heap)
        v1, v2 = -negV1, -negV2
        if v1 < v2:
            v1, v2, id1, id2 = v2, v1, id2, id1
        highNode = nodes[id1]
        lowNode = nodes[id2]
        parentVal = abs(v1 - v2)
        parent = KkMergeNode(parentVal, high=highNode, low=lowNode)
        nodes[nextUid] = parent
        minHeappush(heap, (-parentVal, nextUid))
        nextUid += 1
    _, rootId = heap[0]
    root = nodes[rootId]
    signs = [0] * n

    def assign(node, sign):
        if node.isLeaf():
            signs[node.leafIndex] = sign
        else:
            assign(node.high, sign)
            assign(node.low, -sign)

    assign(root, 1)
    for i in range(n):
        if signs[i] == 0:
            signs[i] = 1
    return signs


def initialPrepartitionFromSigns(signs):
    #pre partitioning representation
    return [1 if s == 1 else 2 for s in signs]


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

def repeatedRandom(A, randSol, residueFn, initialSolution=None):
    if initialSolution is None:
        S = randSol()
    else:
        S = list(initialSolution)
    bestRes = residueFn(S, A)

    for _ in range(defaultIterations):
        sPrime = randSol()
        rPrime = residueFn(sPrime, A)
        if rPrime < bestRes:
            S = sPrime
            bestRes = rPrime

    return bestRes


def hillClimbing(A, randSol, randNeighbor, residueFn, initialSolution=None):
    if initialSolution is None:
        S = randSol()
    else:
        S = list(initialSolution)
    bestRes = residueFn(S, A)

    for _ in range(defaultIterations):
        sPrime = randNeighbor(S)
        rPrime = residueFn(sPrime, A)
        if rPrime < bestRes:
            S = sPrime
            bestRes = rPrime

    return bestRes


def simulatedAnnealing(A, randSol, randNeighbor, residueFn, initialSolution=None):
    if initialSolution is None:
        S = randSol()
    else:
        S = list(initialSolution)
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
