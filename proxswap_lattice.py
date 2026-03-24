"""
proxswap_lattice.py
-------------------
Python port of the proximity-swap ring lattice construction algorithm.
Ported with Claude Sonnet 4.6 from proxswap_lattice.R / proxswap.c in
the {L0ggm} R package (Alexander P. Christensen, 24 March 2026; CC0-1.0).

License: CC0-1.0
To the extent possible under law, the author(s) have dedicated all
copyright and related and neighboring rights to this software to the
public domain worldwide.  This software is distributed without any
warranty.  See <https://creativecommons.org/publicdomain/zero/1.0/>.

This port is a faithful translation of the C logic (proximity_pass_c /
swapping_pass_c) and the R wrapper (proxswap_lattice).

Dependencies
------------
numpy     -- core array operations throughout.
networkx  -- required for clustering-coefficient and connectivity checks;
             mirrors the igraph calls in the original R implementation.
             nx.average_clustering  replaces igraph::transitivity(type="average")
             nx.is_connected        replaces igraph::is_connected

Functions
---------
proxswap_lattice(network, shuffles)  -- main entry point
_build_pairs(D, distance_seq)
_proximity_pass(nodes, ring, budget, pairs, distance_seq)
_swapping_pass(nodes, ring, budget, total_budget)

Usage
-----
    import numpy as np
    from proxswap_lattice import proxswap_lattice

    ring, cc = proxswap_lattice(network, shuffles=100)
    print("Average clustering coefficient:", cc)
    print("Degree sequences match:", np.array_equal(network.astype(bool).sum(axis=0),
                                                     ring.sum(axis=0)))
"""

from __future__ import annotations

import warnings
import numpy as np
import networkx as nx


# =============================================================================
# Public API
# =============================================================================

def proxswap_lattice(
    network: np.ndarray,
    shuffles: int = 100,
) -> tuple[np.ndarray, float]:
    """Construct a degree-preserving ring lattice via proximity-swap construction.

    Converts a network matrix into a connected ring lattice whose degree
    sequence exactly matches the original, while maximising the average local
    clustering coefficient.  The resulting lattice is intended as a null model
    for small-world network analyses.

    Parameters
    ----------
    network : ndarray, shape (n, n)
        Square, symmetric numeric matrix representing a network (e.g. partial
        correlations).  Non-zero off-diagonal entries are treated as edges; the
        binary adjacency is derived internally as ``network != 0``.
        Isolated nodes (degree zero) are supported.
    shuffles : int, optional
        Number of independent random permutation passes to attempt.  Only
        passes producing a connected graph with zero degree error are retained;
        the one with the highest clustering coefficient is returned.
        Default: ``100``.

    Returns
    -------
    ring : ndarray of bool, shape (n, n)
        Symmetric binary adjacency matrix representing the ring lattice with
        the best average clustering coefficient found.
    cc : float
        Average clustering coefficient of ``ring``.  Equals the empirical CC
        when the empirical-fallback path is taken.

    Notes
    -----
    Algorithm
    ~~~~~~~~~
    1. **Pair precomputation** – unique upper-triangle pairs at each ring
       distance ``d = 1 … floor(n/2)`` are cached once.
    2. **Proximity construction** – the degree sequence is randomly permuted
       onto ring positions; edges are added in increasing distance order,
       sorted by descending ``min(budget_i, budget_j)`` within each band.
    3. **Swap repair** – any residual deficit is resolved by direct connection
       or an edge swap, scanning interleaved clockwise / counter-clockwise
       ring positions.
    4. **Pass selection** – only connected, zero-deficit passes are kept; the
       highest-CC pass is returned.
    5. **Empirical fallback** – if no valid pass beats the empirical CC, the
       original adjacency is returned with a warning.

    Dependencies
    ~~~~~~~~~~~~
    ``nx.average_clustering`` – computes the average local clustering
    coefficient over all nodes; mirrors ``igraph::transitivity(type="average")``.

    ``nx.is_connected`` – confirms that the graph has exactly one connected
    component; mirrors ``igraph::is_connected``.
    """
    # ------------------------------------------------------------------
    # Derive binary adjacency
    # ------------------------------------------------------------------
    A = (network != 0)                              # bool, shape (n, n)

    nodes  = A.shape[1]
    degree = A.sum(axis=0).astype(int)              # 1-D, length n

    # ------------------------------------------------------------------
    # Ring distance matrix  d_ij = min(|i-j|, n-|i-j|)
    # ------------------------------------------------------------------
    idx = np.arange(nodes)
    D   = np.abs(idx[:, None] - idx[None, :])
    D   = np.minimum(D, nodes - D)
    distance_seq = np.arange(1, D.max() + 1)

    # ------------------------------------------------------------------
    # Pre-compute unique upper-triangle pairs at each ring distance
    # ------------------------------------------------------------------
    pairs = _build_pairs(D, distance_seq)

    # ------------------------------------------------------------------
    # Empirical clustering coefficient (fallback reference).
    # nx.average_clustering averages over all nodes, matching
    # igraph::transitivity(type="average").
    # ------------------------------------------------------------------
    empirical_CC = nx.average_clustering(nx.from_numpy_array(A.astype(float)))

    # ------------------------------------------------------------------
    # Initialise best-pass trackers
    # ------------------------------------------------------------------
    best_ring:  np.ndarray | None = None
    best_swap:  np.ndarray | None = None
    best_CC:    float             = -np.inf

    ring_init = np.zeros((nodes, nodes), dtype=np.int32)

    # ------------------------------------------------------------------
    # Shuffle passes
    # ------------------------------------------------------------------
    rng = np.random.default_rng()

    for _ in range(shuffles):

        # Random permutation of the degree sequence onto ring positions
        swap_order = rng.permutation(nodes)

        # Proximity construction (greedy, distance order)
        res = _proximity_pass(nodes, ring_init.copy(),
                              degree[swap_order].copy(),
                              pairs, distance_seq)

        # Swap repair (resolve residual deficit)
        res = _swapping_pass(nodes, res["ring"], res["budget"],
                             res["total_budget"])

        # Reject: residual deficit or disconnected graph.
        # nx.is_connected mirrors igraph::is_connected.
        G_pass = nx.from_numpy_array(res["ring"].astype(float))
        if res["remaining"] > 0 or not nx.is_connected(G_pass):
            continue

        # Clustering coefficient for this valid pass.
        # Reuse the graph already built for the connectivity check.
        pass_CC = nx.average_clustering(G_pass)

        # Keep the best
        if pass_CC > best_CC:
            best_CC   = pass_CC
            best_ring = res["ring"].copy()
            best_swap = swap_order.copy()

    # ------------------------------------------------------------------
    # Empirical fallback check
    # ------------------------------------------------------------------
    empirical_flag = empirical_CC > best_CC

    if empirical_flag:
        warnings.warn(
            f"The lattice solution did not produce a better CC ({best_CC:.3f}) "
            f"than the empirical ({empirical_CC:.3f}).\n"
            "Falling back to empirical solution...",
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Restore original node ordering  (invert the permutation)
    # ------------------------------------------------------------------
    if empirical_flag:
        ring = A.astype(bool)
        cc   = empirical_CC
    else:
        # original_order is the inverse permutation of best_swap:
        #   best_swap[original_order] == arange(nodes)
        original_order = np.empty(nodes, dtype=int)
        original_order[best_swap] = np.arange(nodes)
        ring = best_ring[np.ix_(original_order, original_order)].astype(bool)
        cc   = best_CC

    return ring, cc


# =============================================================================
# _build_pairs
# =============================================================================

def _build_pairs(
    D: np.ndarray,
    distance_seq: np.ndarray,
) -> list[np.ndarray]:
    """Pre-compute upper-triangle (i < j) node pairs at each ring distance.

    For a ring of n nodes the circular distance between positions i and j is
    min(|i - j|, n - |i - j|).  This helper extracts, for every integer
    distance r = 1, ..., floor(n / 2), the set of unordered pairs {i, j}
    (i < j) that sit exactly r steps apart on the ring.  Results are cached
    once and reused across all shuffle passes in proxswap_lattice().

    Parameters
    ----------
    D            : ndarray of int, shape (n, n)
                   Circular ring distance matrix where entry [i, j] holds the
                   ring distance between nodes i and j.
    distance_seq : ndarray of int
                   Array ``1 ... max(D)`` enumerating every distinct ring
                   distance to process.

    Returns
    -------
    pairs : list of ndarray
        List of length ``len(distance_seq)``; each element is an ``M x 2``
        int array of ``[row, col]`` indices (0-based) with ``row < col``,
        listing every pair of nodes at the corresponding ring distance.
    """
    pairs: list[np.ndarray] = []
    for d in distance_seq:
        rows, cols = np.where(D == d)
        mask = rows < cols
        pairs.append(np.column_stack([rows[mask], cols[mask]]))
    return pairs


# =============================================================================
# _proximity_pass
# =============================================================================

def _proximity_pass(
    nodes:        int,
    ring:         np.ndarray,
    budget:       np.ndarray,
    pairs:        list[np.ndarray],
    distance_seq: np.ndarray,
) -> dict:
    """Greedy edge assignment in strictly increasing ring-distance order.

    Translated directly from ``proximity_pass_c`` in ``proxswap.c``.

    Parameters
    ----------
    nodes        : number of nodes.
    ring         : ``(nodes, nodes)`` int array, modified in-place (copy before call).
    budget       : 1-D int array of length ``nodes``, degree deficits.
    pairs        : list from ``_build_pairs`` (0-based indices).
    distance_seq : array ``1 … max_distance``.

    Returns
    -------
    dict with keys ``"ring"``, ``"budget"``, ``"total_budget"``.
    """
    ring   = ring.astype(np.int32)
    budget = budget.astype(np.int32)

    total_budget = int(budget.sum())

    for p in pairs:
        if p.size == 0:
            continue

        row_idx = p[:, 0]
        col_idx = p[:, 1]

        # Eligibility: both endpoints have budget > 0, no existing edge
        elig_mask = (
            (budget[row_idx] > 0) &
            (budget[col_idx] > 0) &
            (ring[row_idx, col_idx] == 0)
        )

        if not np.any(elig_mask):
            continue

        row_e = row_idx[elig_mask]
        col_e = col_idx[elig_mask]

        # Stable descending sort by min(budget[r], budget[c]).
        # NumPy's argsort with kind='stable' matches R's order() tie-breaking.
        min_bud = np.minimum(budget[row_e], budget[col_e])
        srt     = np.argsort(-min_bud, kind="stable")
        row_e   = row_e[srt]
        col_e   = col_e[srt]

        # Assign edges one at a time; re-check budgets before each assignment
        for r, c in zip(row_e, col_e):
            if budget[r] > 0 and budget[c] > 0:
                ring[r, c] = 1
                ring[c, r] = 1
                budget[r] -= 1
                budget[c] -= 1
                total_budget -= 2

        # Early exit once all degree deficits are satisfied
        if total_budget < 1:
            break

    return {"ring": ring, "budget": budget, "total_budget": total_budget}


# =============================================================================
# _swapping_pass
# =============================================================================

def _swapping_pass(
    nodes:        int,
    ring:         np.ndarray,
    budget:       np.ndarray,
    total_budget: int,
) -> dict:
    """Resolve residual degree deficit via direct connection or edge swap.

    Translated directly from ``swapping_pass_c`` in ``proxswap.c``.

    Parameters
    ----------
    nodes        : number of nodes.
    ring         : ``(nodes, nodes)`` int array (copy before call if needed).
    budget       : 1-D int array of degree deficits.
    total_budget : sum of ``budget``.

    Returns
    -------
    dict with keys ``"ring"`` and ``"remaining"``.
    """
    ring   = ring.astype(np.int32)
    budget = budget.astype(np.int32)

    n_dist   = nodes // 2
    max_iter = 2 * nodes * nodes

    for _ in range(max_iter):

        if total_budget == 0:
            break

        # Node with largest remaining deficit
        i = int(np.argmax(budget))
        if budget[i] == 0:
            break

        # -----------------------------------------------------------------
        # Build interleaved clockwise / counter-clockwise position list
        # Mirrors C:  cw = (i + d) % n,  ccw = ((i - d) % n + n) % n
        # Both use 0-based indices throughout this function.
        # -----------------------------------------------------------------
        visited     = np.zeros(nodes, dtype=bool)
        visited[i]  = True
        interleaved = []

        for d in range(1, n_dist + 1):
            cw = (i + d) % nodes
            if not visited[cw]:
                visited[cw] = True
                interleaved.append(cw)

            ccw = (i - d) % nodes        # Python % is always non-negative
            if not visited[ccw]:
                visited[ccw] = True
                interleaved.append(ccw)

        # -----------------------------------------------------------------
        # Attempt 1 – direct connection
        # -----------------------------------------------------------------
        direct = -1
        for j in interleaved:
            if budget[j] > 0 and ring[i, j] == 0:
                direct = j
                break

        if direct >= 0:
            ring[i, direct] = 1
            ring[direct, i] = 1
            budget[i]      -= 1
            budget[direct] -= 1
            total_budget   -= 2
            continue

        # -----------------------------------------------------------------
        # Attempt 2 – edge swap
        # Find nearby unconnected node j; remove one of j's edges (j,k);
        # add edge (i,j).  Node k recovers its budget for later resolution.
        # -----------------------------------------------------------------
        swapped = False

        for j in interleaved:
            if swapped:
                break
            if ring[i, j] != 0:          # already connected
                continue

            for k in range(nodes):
                if k == i:               continue
                if ring[j, k] == 0:      continue   # not a neighbour of j
                if ring[i, k] != 0:      continue   # already connected to i

                # Perform swap: remove (j,k), add (i,j)
                ring[j, k] = 0
                ring[k, j] = 0
                ring[i, j] = 1
                ring[j, i] = 1

                budget[i] -= 1   # i gains an edge
                budget[k] += 1   # k loses an edge (recovers budget)
                # total_budget unchanged: -1 and +1 cancel

                swapped = True
                break

        if not swapped:
            break

    return {"ring": ring.astype(bool), "remaining": int(total_budget)}


