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
proxswap_lattice(network, weighted, shuffles)  -- main entry point
_build_pairs(D, distance_seq)
_proximity_pass(nodes, ring, budget, pairs, distance_seq)
_swapping_pass(nodes, node_seq, ring, budget, total_budget, distance_seq)
_interleave(index, nodes, distance_seq)
_assign_weights(network, A, distance_matrix, rng)

Usage
-----
    import numpy as np
    from proxswap_lattice import proxswap_lattice

    # Binary lattice (default)
    ring, cc = proxswap_lattice(network, shuffles=100)
    print("Average clustering coefficient:", cc)
    print("Degree sequences match:", np.array_equal(network.astype(bool).sum(axis=0),
                                                     ring.sum(axis=0)))

    # Weighted lattice
    ring_w, cc_w = proxswap_lattice(network, weighted=True, shuffles=100)
    print("Weighted average clustering coefficient:", cc_w)
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
    weighted: bool = False,
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
        correlations).  Non-zero off-diagonal entries are treated as edges;
        the binary adjacency is derived internally as ``network != 0``.
        When ``weighted`` is ``False``, edge weights have no effect on the
        result.  Isolated nodes (degree zero) are supported.
    weighted : bool, optional
        Whether to return a weighted ring lattice.  When ``True``, edge weights
        from ``network`` are reassigned to the lattice topology following
        Muldoon, Bridgeford, & Bassett (2016): observed weights are sorted in
        descending order by absolute value and mapped onto lattice edges ranked
        by ascending ring distance, so that shorter (more local) connections
        receive the largest-magnitude weights.  Original signed weights are
        preserved.  The clustering coefficient is computed via
        ``nx.average_clustering(..., weight='weight')``.  When ``False``
        (default), a binary ``bool`` adjacency matrix is returned and the
        unweighted clustering coefficient is used throughout.
    shuffles : int, optional
        Number of independent random permutation passes to attempt.  Only
        passes producing a graph with zero degree error are retained (and,
        when all nodes have degree > 0, also a connected graph); the one with
        the highest clustering coefficient is returned.  Default: ``100``.

    Returns
    -------
    ring : ndarray, shape (n, n)
        Symmetric matrix representing the ring lattice with the best average
        clustering coefficient found.  When ``weighted=False`` (default),
        dtype is ``bool``.  When ``weighted=True``, dtype is ``float64`` and
        entries contain the reassigned edge weights from ``network``.  When
        either fallback path is taken, ``ring`` equals the binarised input
        (``weighted=False``) or the original ``network`` matrix
        (``weighted=True``).
    cc : float
        Average clustering coefficient of ``ring``.  Equals the empirical CC
        when either fallback path is taken.

    Warns
    -----
    UserWarning
        Two distinct warning conditions may be raised:

        *Convergence failure* -- if no shuffle pass produces a valid lattice
        at all (i.e. every pass has a residual degree deficit or is
        disconnected), a warning is issued advising the caller to increase
        ``shuffles``, and the empirical graph is returned.

        *Empirical fallback* -- if at least one valid lattice was found but
        its best clustering coefficient does not exceed the empirical value,
        a warning reporting both coefficients is issued and the original
        adjacency (or weighted matrix) is returned.

    Notes
    -----
    Algorithm
    ~~~~~~~~~
    1. **Pair precomputation** – unique upper-triangle pairs at each ring
       distance ``d = 1 … floor(n/2)`` are cached once in ``_build_pairs``.
    2. **Proximity construction** – the degree sequence is randomly permuted
       onto ring positions; edges are added in increasing distance order,
       sorted by descending ``min(budget_i, budget_j)`` within each band.
       Implemented in ``_proximity_pass``.
    3. **Swap repair** – any residual deficit is resolved by direct connection
       or an edge swap, scanning interleaved clockwise / counter-clockwise
       ring positions produced by ``_interleave``.
       Implemented in ``_swapping_pass``.
    4. **Connectivity check** – skipped entirely when any node has degree zero
       (isolated nodes cannot form a connected graph regardless); otherwise
       ``nx.is_connected`` is called on the candidate lattice.
    5. **Weight assignment** – when ``weighted=True``, observed weights are
       sorted descending by absolute value and mapped onto lattice edges ranked
       by ascending ring distance (``d_ij = min(|i-j|, n-|i-j|)``), so that
       shorter connections receive the largest weights.  Ties in ring distance
       are broken at random.  Implemented in ``_assign_weights``.
    6. **Pass selection** – only zero-deficit (and, when applicable, connected)
       passes are kept; the highest-CC pass is returned.
    7. **Empirical fallback** – see Warns above.

    Dependencies
    ~~~~~~~~~~~~
    ``nx.average_clustering`` – computes the average local clustering
    coefficient over all nodes; mirrors ``igraph::transitivity(type="average")``.
    Accepts a ``weight`` keyword for weighted graphs.

    ``nx.is_connected`` – confirms that the graph has exactly one connected
    component; mirrors ``igraph::is_connected``.
    """

    # ------------------------------------------------------------------
    # Derive binary adjacency
    # ------------------------------------------------------------------
    A = (network != 0)                              # bool, shape (n, n)

    nodes  = A.shape[1]
    degree = A.sum(axis=0).astype(int)              # 1-D, length n

    # Skip the connectivity check when any node has degree zero:
    # isolated nodes cannot form a connected graph regardless.
    check_connectedness = bool(np.all(degree != 0))

    # ------------------------------------------------------------------
    # Ring distance matrix  d_ij = min(|i-j|, n-|i-j|)
    # ------------------------------------------------------------------
    idx          = np.arange(nodes)
    D            = np.abs(idx[:, None] - idx[None, :])
    D            = np.minimum(D, nodes - D)
    distance_seq = np.arange(1, D.max() + 1)

    # ------------------------------------------------------------------
    # Pre-compute unique upper-triangle pairs at each ring distance
    # ------------------------------------------------------------------
    pairs = _build_pairs(D, distance_seq)

    # ------------------------------------------------------------------
    # Empirical clustering coefficient (fallback reference).
    # nx.average_clustering averages over all nodes, matching
    # igraph::transitivity(type="average").
    # When weighted, pass weight="weight" to use edge weights.
    # ------------------------------------------------------------------
    if weighted:
        empirical_CC = nx.average_clustering(
            nx.from_numpy_array(network), weight="weight"
        )
    else:
        empirical_CC = nx.average_clustering(nx.from_numpy_array(A.astype(float)))

    # ------------------------------------------------------------------
    # Initialise best-pass trackers
    # ------------------------------------------------------------------
    best_ring:  np.ndarray | None = None
    best_swap:  np.ndarray | None = None
    best_CC:    float             = -np.inf

    node_seq  = np.arange(nodes)
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
        res = _swapping_pass(nodes, node_seq, res["ring"], res["budget"],
                             res["total_budget"], distance_seq)

        # Reject passes with residual degree deficit
        if res["remaining"] > 0:
            continue

        # Connectivity check -- skipped when isolated nodes are present
        if check_connectedness:
            G_pass = nx.from_numpy_array(res["ring"].astype(float))
            if not nx.is_connected(G_pass):
                continue
        else:
            G_pass = nx.from_numpy_array(res["ring"].astype(float))

        # Assign weights to the valid binary topology when requested.
        # pass_ring is float64 with reassigned weights; otherwise bool.
        if weighted:
            pass_ring = _assign_weights(network, res["ring"], D, rng)
            G_pass    = nx.from_numpy_array(pass_ring)
            pass_CC   = nx.average_clustering(G_pass, weight="weight")
        else:
            pass_ring = res["ring"].copy()
            pass_CC   = nx.average_clustering(G_pass)

        # Keep the best
        if pass_CC > best_CC:
            best_CC   = pass_CC
            best_ring = pass_ring
            best_swap = swap_order.copy()

    # ------------------------------------------------------------------
    # Determine whether any valid solution was found
    # ------------------------------------------------------------------
    if np.isinf(best_CC):

        # Convergence failure: no valid pass produced at all
        warnings.warn(
            "The lattice solution did not converge. "
            "The empirical graph was returned.\n\n"
            "Try increasing `shuffles` to find a suitable solution.",
            stacklevel=2,
        )

        empirical_flag = True

    else:

        # At least one valid pass exists -- check whether empirical beats it
        empirical_flag = empirical_CC > best_CC

        if empirical_flag:
            warnings.warn(
                f"The lattice solution did not produce a better CC ({best_CC:.3f}) "
                f"than the empirical ({empirical_CC:.3f}).\n"
                "Falling back to empirical solution...",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Return the selected result
    # ------------------------------------------------------------------
    if empirical_flag:
        # Return weighted matrix or binary adjacency depending on mode
        ring = network if weighted else A.astype(bool)
        cc   = empirical_CC
    else:
        # original_order is the inverse permutation of best_swap:
        #   best_swap[original_order] == arange(nodes)
        original_order = np.empty(nodes, dtype=int)
        original_order[best_swap] = np.arange(nodes)
        reordered = best_ring[np.ix_(original_order, original_order)]
        ring = reordered if weighted else reordered.astype(bool)
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

    Starting from the supplied (initially empty) ring adjacency, edges are
    added one distance band at a time.  Within each band, eligible pairs --
    those where both endpoints still have remaining degree budget and are not
    yet connected -- are sorted by descending min(budget_i, budget_j) so that
    the most degree-deficient pairs receive the shortest available connections
    first.  Each pair is then considered in that order, with budgets re-checked
    immediately before assignment because earlier placements in the same band
    can exhaust a node's budget.  The loop exits early once all degree budgets
    reach zero.

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
    node_seq:     np.ndarray,
    ring:         np.ndarray,
    budget:       np.ndarray,
    total_budget: int,
    distance_seq: np.ndarray,
) -> dict:
    """Resolve residual degree deficit via direct connection or edge swap.

    Translated directly from ``swapping_pass_c`` in ``proxswap.c``.

    At each iteration the node with the largest remaining deficit (i) is
    targeted.  Ring positions are scanned in interleaved clockwise /
    counter-clockwise order (nearest first), as produced by ``_interleave``,
    to find a connection for i:

    *Direct connection*: if any scanned node j has remaining budget and is not
    yet connected to i, the edge (i, j) is added immediately and the iteration
    continues.

    *Edge swap*: if no direct partner exists, the scan finds the nearest
    unconnected node j that has at least one existing neighbour k not already
    connected to i.  The edge (j, k) is removed and (i, j) is added.  Node k
    recovers its budget and will be re-targeted in a later iteration.  Because
    one deficit is transferred rather than eliminated, total_budget is unchanged
    by a swap.

    Iteration stops when total_budget reaches zero, when no swap can be found,
    or after the hard cap of 2 * nodes ** 2 iterations.

    Parameters
    ----------
    nodes        : number of nodes.
    node_seq     : 1-D int array ``np.arange(nodes)``, passed in to avoid
                   re-creation on every call; used to index neighbours during
                   the swap-candidate search.
    ring         : ``(nodes, nodes)`` int array (copy before call if needed),
                   as returned by ``_proximity_pass``.
    budget       : 1-D int array of degree deficits, as returned by
                   ``_proximity_pass``.
    total_budget : sum of ``budget``, as returned by ``_proximity_pass``.
    distance_seq : int array ``1 … floor(nodes/2)``, passed directly to
                   ``_interleave`` to generate the clockwise / counter-clockwise
                   scan order.

    Returns
    -------
    dict with keys ``"ring"`` and ``"remaining"``.
    """
    ring   = ring.astype(np.int32)
    budget = budget.astype(np.int32)

    max_iter = 2 * nodes * nodes

    for _ in range(max_iter):

        if total_budget == 0:
            break

        # Node with largest remaining deficit (0-based)
        i = int(np.argmax(budget))
        if budget[i] == 0:
            break

        # Interleave clockwise and counter-clockwise ring positions by distance
        interleaved = _interleave(i, nodes, distance_seq)

        # Available connections (diagonal is False; will never appear in interleaved)
        available_ring = ~ring.astype(bool)

        # -----------------------------------------------------------------
        # Attempt 1 -- direct connection to nearest node with budget & no edge
        # -----------------------------------------------------------------
        direct_candidates = [j for j in interleaved
                             if budget[j] > 0 and available_ring[i, j]]

        if direct_candidates:
            target          = direct_candidates[0]
            ring[i, target] = 1
            ring[target, i] = 1
            budget[i]      -= 1
            budget[target] -= 1
            total_budget   -= 2
            continue

        # -----------------------------------------------------------------
        # Attempt 2 -- edge swap
        # Find a nearby unconnected node j; remove one of j's edges (j, k);
        # add edge (i, j).  Node k recovers its budget for later resolution.
        # -----------------------------------------------------------------
        swapped    = False
        candidates = [j for j in interleaved if available_ring[i, j]]

        for j in candidates:
            if swapped:
                break

            # Find first neighbour k of j that is not i and not connected to i
            neighborhood = np.where(
                ring[j].astype(bool) & (node_seq != i) & available_ring[i]
            )[0]

            if neighborhood.size > 0:
                k = int(neighborhood[0])

                # Perform swap: remove (j, k), add (i, j)
                ring[j, k] = 0
                ring[k, j] = 0
                ring[i, j] = 1
                ring[j, i] = 1

                budget[i] -= 1   # i gains an edge
                budget[k] += 1   # k loses an edge (recovers budget)
                # total_budget is unchanged: -1 and +1 cancel

                swapped = True

        if not swapped:
            break

    return {"ring": ring.astype(bool), "remaining": int(total_budget)}


# =============================================================================
# _interleave
# =============================================================================

def _interleave(
    index:        int,
    nodes:        int,
    distance_seq: np.ndarray,
) -> list[int]:
    """Build an interleaved clockwise / counter-clockwise position list.

    Generates the scan order used by ``_swapping_pass`` when searching for a
    direct connection or swap partner for node ``index``.  Starting from
    ``index``, positions at ring distance d = 1, 2, ... are emitted in
    alternating clockwise / counter-clockwise pairs, yielding a sequence that
    visits the nearest ring neighbours first.  Duplicate positions (which can
    occur when nodes is even at the maximum distance) are dropped via a visited
    mask.

    Parameters
    ----------
    index        : scalar int (0-based), the node whose neighbours are to be
                   enumerated.
    nodes        : total number of nodes in the ring.
    distance_seq : int array ``1 … floor(nodes/2)``, the distances at which to
                   emit clockwise / counter-clockwise positions.

    Returns
    -------
    list of int
        0-based node indices in interleaved nearest-first order, with
        duplicates removed.  Length is at most ``2 * floor(nodes / 2)``, and
        may be shorter when nodes is odd or duplicates are dropped at the
        maximum distance.
    """
    visited        = np.zeros(nodes, dtype=bool)
    visited[index] = True
    result: list[int] = []

    for d in distance_seq:
        cw = (index + d) % nodes
        if not visited[cw]:
            visited[cw] = True
            result.append(int(cw))

        # Python's % is always non-negative, so no explicit +nodes guard needed
        ccw = (index - d) % nodes
        if not visited[ccw]:
            visited[ccw] = True
            result.append(int(ccw))

    return result


# =============================================================================
# _assign_weights
# =============================================================================

def _assign_weights(
    network:         np.ndarray,
    A:               np.ndarray,
    distance_matrix: np.ndarray,
    rng:             np.random.Generator,
) -> np.ndarray:
    """Reassign edge weights from ``network`` onto the binary lattice ``A``.

    Translated from ``assign_weights()`` in ``proxswap_lattice.R``.

    Following Muldoon, Bridgeford, & Bassett (2016): observed edge weights are
    sorted in descending order by absolute value and mapped onto lattice edges
    ranked by ascending ring distance (``d_ij = min(|i-j|, n-|i-j|)``), so
    that shorter (more local) connections receive the largest-magnitude weights.
    This uses the ring's structural distances directly rather than any
    network-derived proxy.  Original signed weights are preserved.  Ties in
    ring distance are broken at random.

    Parameters
    ----------
    network : ndarray, shape (n, n)
        The original weighted network as passed into ``proxswap_lattice``
        (with ``weighted=True``).  Non-zero lower-triangle entries supply the
        weight pool; weights are extracted with their original signs and sorted
        by descending absolute value so that larger-magnitude weights are placed
        on shorter lattice edges.
    A : ndarray, shape (n, n)
        Binary adjacency matrix (the lattice topology as returned by
        ``_swapping_pass``).  May be ``bool`` or integer.
    distance_matrix : ndarray of int, shape (n, n)
        Circular ring distance matrix where entry [i, j] holds the ring
        distance between nodes i and j.  Passed in from ``proxswap_lattice``
        so that weight assignment uses the same distance structure as the
        proximity construction.
    rng : numpy.random.Generator
        Random generator passed in from ``proxswap_lattice`` so that weight
        assignment participates in the same reproducible stream.

    Returns
    -------
    result : ndarray of float64, shape (n, n)
        Symmetric weighted matrix with reassigned edge weights.
    """
    n = network.shape[0]

    # Lower-triangle mask (strict: excludes diagonal), matching R's lower.tri()
    lt_rows, lt_cols = np.tril_indices(n, k=-1)

    # Extract non-zero weights from lower triangle of network.
    # Mirrors R: weights <- network[lower_triangle][network_nonzero][
    #               order(abs(network)[lower_triangle][network_nonzero],
    #                     decreasing = TRUE)]
    # Original (possibly signed) values are extracted, then reordered by
    # descending absolute value so that larger-magnitude weights are placed
    # on shorter lattice edges.
    net_lt    = network[lt_rows, lt_cols]
    net_lt_nz = net_lt[net_lt != 0]                              # original signed values
    sort_idx  = np.argsort(-np.abs(net_lt_nz), kind="stable")   # order by |weight|
    weights   = net_lt_nz[sort_idx]                              # 1-D, descending |w|

    # Identify non-zero edges in lower triangle of A
    A_lt      = A[lt_rows, lt_cols].astype(float)
    A_nz_mask = A_lt != 0

    # Ring distances for lattice edges, taken directly from the precomputed
    # distance matrix.  Mirrors R: distance_matrix[lower_triangle][A_nonzero]
    edge_dists = distance_matrix[lt_rows, lt_cols][A_nz_mask].astype(float)

    # Random-tiebreak rank, mirroring R's rank(..., ties.method = "random"):
    # shuffle indices, stable-sort the shuffled distances, then invert.
    n_edges  = edge_dists.size
    rp       = rng.permutation(n_edges)
    sort_idx = np.argsort(edge_dists[rp], kind="stable")
    weight_order                = np.empty(n_edges, dtype=int)
    weight_order[rp[sort_idx]]  = np.arange(n_edges)            # 0-based ranks

    # Write weights into lower triangle, zero upper triangle, symmetrise
    lt_vals            = np.zeros(lt_rows.size)
    lt_vals[A_nz_mask] = weights[weight_order]
    result             = np.zeros((n, n), dtype=float)
    result[lt_rows, lt_cols] = lt_vals
    result             = result + result.T                       # symmetric

    return result
