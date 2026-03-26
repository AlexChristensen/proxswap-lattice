# proxswap_lattice.R
# ------------------
# Proximity-swap ring lattice construction algorithm.
# Part of proxswap.c / proxswap_lattice.R in the {L0ggm} R package
# (Alexander P. Christensen, 24 March 2026; CC0-1.0).
#
# License: CC0-1.0
# To the extent possible under law, the author(s) have dedicated all
# copyright and related and neighboring rights to this software to the
# public domain worldwide.  This software is distributed without any
# warranty.  See <https://creativecommons.org/publicdomain/zero/1.0/>.
#
# Dependencies
# ------------
#   {igraph}
#     igraph::transitivity  -- average local clustering coefficient;
#                              uses edge weights automatically when present,
#                              matching 'type = "average"' throughout.
#     igraph::is_connected  -- single-component connectivity check.
#   {L0ggm} (internal helpers, must be on the search path)
#     L0ggm:::convert2igraph   -- converts a numeric/logical matrix to an
#                              igraph graph object.
#     L0ggm:::swiftelse        -- vectorised ifelse alternative used for
#                              concise conditional assignment.
#     L0ggm:::shuffle          -- wrapper around sample() used to randomly
#                              permute a sequence.
#
# Functions
# ---------
#   proxswap_lattice(network, weighted, shuffles)  -- main entry point
#   build_pairs(distance_matrix, distance_sequence)
#   proximity_pass(nodes, ring, budget, pairs, distance_sequence)
#   swapping_pass(nodes, node_sequence, ring, budget, total_budget, distance_sequence)
#   interleave(index, nodes, distance_sequence)
#   assign_weights(network, A, distance_matrix)
#
# Usage
# -----
#   # Binary lattice (default)
#   ring <- proxswap_lattice(network, shuffles = 100)
#   cat("Average clustering coefficient:", attr(ring, "CC"), "\n")
#   cat("Degree sequences match:",
#       all(colSums(network != 0) == colSums(ring != 0)), "\n")
#
#   # Weighted lattice
#   ring_w <- proxswap_lattice(network, weighted = TRUE, shuffles = 100)
#   cat("Weighted average clustering coefficient:", attr(ring_w, "CC"), "\n")


# Proximity-swap lattice construction ----
# Updated 26.03.2026
proxswap_lattice <- function(network, weighted = FALSE, shuffles = 100)
# PROXSWAP_LATTICE  Construct a degree-preserving ring lattice via
#   proximity-swap construction.
#
#   ring <- proxswap_lattice(network) converts the square symmetric matrix
#   'network' into a connected ring lattice whose degree sequence exactly
#   matches the original, while maximising the average local clustering
#   coefficient.  Non-zero off-diagonal entries are treated as edges; the
#   binary adjacency is derived internally.  Isolated nodes (degree zero)
#   are supported.  Defaults to binary output and 100 shuffle passes.
#
#   The average clustering coefficient of the returned lattice is attached
#   to the result as attr(ring, "CC").
#
#   Arguments
#   ---------
#   network  : n x n numeric matrix (weighted or binary, symmetric).
#              Non-zero off-diagonal entries are treated as edges; the binary
#              adjacency is derived internally as (network != 0).  When
#              'weighted' is FALSE the internal copy of 'network' is replaced
#              by this binary adjacency before processing, so signed or
#              real-valued weights have no effect on the binary path.
#              Isolated nodes (degree zero) are supported.
#   weighted : logical scalar, whether to return a weighted lattice
#              (default FALSE).  When TRUE, observed edge weights from
#              'network' are sorted in descending absolute-value order and
#              mapped onto lattice edges ranked by ascending ring distance,
#              so that shorter (more local) connections receive the
#              largest-magnitude weights, following Muldoon, Bridgeford, &
#              Bassett (2016).  Original signed weights are preserved.  The
#              clustering coefficient is then computed via
#              igraph::transitivity(), which uses edge weights automatically
#              when they are present.  When FALSE (default), a logical
#              adjacency matrix is returned.
#   shuffles : positive integer, number of independent random permutation
#              passes to attempt (default 100).  Only passes producing a
#              graph with zero degree error are retained (and, when all
#              nodes have degree > 0, also a connected graph); the one with
#              the highest clustering coefficient is returned.
#
#   Returns
#   -------
#   ring : n x n symmetric matrix representing the ring lattice with the
#          best average clustering coefficient among all valid passes.
#          When 'weighted' is FALSE (default), entries are logical (TRUE/FALSE).
#          When 'weighted' is TRUE, entries contain the reassigned edge
#          weights from 'network'.  When the empirical fallback is taken,
#          'ring' equals the binarised input ('weighted' FALSE) or the
#          original 'network' matrix ('weighted' TRUE).
#          Row and column names are inherited from 'network' via dimnames().
#          The average clustering coefficient is attached as attr(ring, "CC").
#
#   Warnings
#   --------
#   Two distinct warning conditions may be raised:
#
#   Convergence failure -- if no shuffle pass produces a valid lattice at
#     all (i.e. every pass has a residual degree deficit or is
#     disconnected), a warning is issued advising the caller to increase
#     'shuffles', and the empirical graph is returned.
#
#   Empirical fallback  -- if at least one valid lattice was found but its
#     best clustering coefficient does not exceed the empirical value, a
#     warning reporting both coefficients is issued and the original
#     adjacency (or weighted matrix) is returned.
#
#   Algorithm overview
#   ------------------
#   1. Pair precomputation   -- unique upper-triangle pairs at each ring
#      distance d = 1 ... floor(n/2) are cached once in build_pairs().
#   2. Proximity construction -- the degree sequence is randomly permuted
#      onto ring positions; edges are added in increasing distance order,
#      sorted by descending min(budget_i, budget_j) within each band.
#      Implemented in proximity_pass().
#   3. Swap repair           -- any residual deficit is resolved by direct
#      connection or an edge swap, scanning interleaved clockwise /
#      counter-clockwise positions produced by interleave().
#      Implemented in swapping_pass().
#   4. Connectivity check    -- skipped entirely when any node has degree
#      zero (isolated nodes cannot form a connected graph regardless);
#      otherwise igraph::is_connected() is called on the candidate lattice.
#   5. Weight assignment     -- when 'weighted' is TRUE, observed weights
#      are sorted descending by absolute value and mapped onto lattice edges
#      ranked by ascending ring distance (d_ij = min(|i-j|, n-|i-j|)), so
#      that shorter connections receive the largest weights.  Ties in ring
#      distance are broken at random.  Implemented in assign_weights().
#   6. Pass selection        -- only zero-deficit (and, when applicable,
#      connected) passes are kept; the highest-CC pass is returned.
#   7. Empirical fallback    -- see 'Warnings' above.
#
#   Dependencies
#   ------------
#   igraph::transitivity  -- computes the average local clustering
#                            coefficient over all nodes; uses edge weights
#                            automatically when present (type = "average").
#   igraph::is_connected  -- confirms that the graph has exactly one
#                            connected component.
#   convert2igraph        -- {L0ggm} helper that converts a matrix to an
#                            igraph graph object before each igraph call.
{

  # Automatically construct adjacency
  A <- network != 0

  # Check for weighted
  if(!weighted){
    network <- A
  }

  # Initialize nodes, degree, and connectedness flag
  nodes  <- dim(network)[2]
  degree <- colSums(A)
  check_connectedness <- all(degree != 0)

  # Set up distance matrix
  node_sequence <- seq_len(nodes)
  distance_matrix <- abs(outer(node_sequence, node_sequence, "-"))
  distance_matrix <- pmin(distance_matrix, nodes - distance_matrix)
  distance_sequence <- seq_len(max(distance_matrix))

  # Pre-compute unique node pairs at each ring distance
  pairs <- build_pairs(distance_matrix, distance_sequence)

  # Compute empirical clustering coefficient for fallback check
  empirical_CC <- igraph::transitivity(L0ggm:::convert2igraph(network), type = "average")

  # Initialize best result trackers
  best_swap <- best_ring <- NULL
  best_CC   <- -Inf

  # Initialize to an empty graph
  ring <- matrix(FALSE, nrow = nodes, ncol = nodes)

  # Run each shuffle pass
  for(k in seq_len(shuffles)){

    # Store swap order (returns proper degree sequence)
    swap_order <- L0ggm:::shuffle(node_sequence)

    # Run proximity construction with shuffled degrees
    result <- proximity_pass(
      nodes = nodes, ring = ring, budget = degree[swap_order],
      pairs = pairs, distance_sequence = distance_sequence
    )

    # Perform swapping to
    result <- swapping_pass(
      nodes = nodes, node_sequence = node_sequence, ring = result$ring,
      budget = result$budget, total_budget = result$total_budget,
      distance_sequence = distance_sequence
    )

    # Check for deficit before expensive connectedness check
    if(result$remaining > 0){
      next
    }

    # Convert ring to {igraph}
    iring <- L0ggm:::convert2igraph(result$ring)

    # Determine whether connectedness (and then check it)
    if(check_connectedness && (!igraph::is_connected(iring))){
      next
    }

    # Check for weighted
    if(weighted){

      # Get weights
      result$ring <- assign_weights(network, result$ring, distance_matrix)

      # Update ring to {igraph}
      iring <- L0ggm:::convert2igraph(result$ring)

    }

    # With success, compute clustering coefficient for this pass
    pass_CC <- igraph::transitivity(iring, type = "average")

    # Update best pass
    if(pass_CC > best_CC){
      best_CC <- pass_CC; best_ring <- result$ring; best_swap <- swap_order
    }

  }

  # Determine whether solution was reached
  if(is.infinite(best_CC)){

    # Send warning
    warning(paste0(
      "The lattice solution did not converge. The empirical graph was returned.\n\n",
      "Try increasing `shuffles` to find a suitable solution"
    ))

    # Set empirical flag
    empirical_flag <- TRUE

    # Check for empirical and weighted
    ring <- L0ggm:::swiftelse(weighted, network, A)

  }else{

    # Check whether empirical beats the best lattice
    empirical_flag <- empirical_CC > best_CC

    # Warn on empirical fallback
    if(empirical_flag){
      warning(paste0(
        "The lattice solution did not produce a better CC (", round(best_CC, 3),
        ") than the empirical (", round(empirical_CC, 3), ").\n",
        "Falling back to empirical solution..."
      ))
    }

    # Select between lattice and empirical
    original_order <- order(best_swap)

    # Check for empirical and weighted
    ring <- L0ggm:::swiftelse(
      empirical_flag,
      L0ggm:::swiftelse(weighted, network, A),
      best_ring[original_order, original_order]
    )

  }

  # Ensure named matrix
  dimnames(ring) <- dimnames(network)

  # Attach clustering coefficient
  attr(ring, "CC") <- L0ggm:::swiftelse(empirical_flag, empirical_CC, best_CC)

  # Return ring
  return(ring)

}


# Pre-compute unique node pairs per ring distance ----
# Updated 24.03.2026
build_pairs <- function(distance_matrix, distance_sequence)
# BUILD_PAIRS  Pre-compute upper-triangle (row < col) node pairs at each ring distance.
#
#   For a ring of n nodes the circular distance between positions i and j is
#   min(|i - j|, n - |i - j|).  This helper extracts, for every integer
#   distance r = 1, ..., floor(n / 2), the set of unordered pairs {i, j}
#   (i < j) that sit exactly r steps apart on the ring.  Results are cached
#   once and reused across all shuffle passes in proxswap_lattice().
#
#   Arguments
#   ---------
#   distance_matrix   : n x n integer matrix where entry [i, j] holds the
#                       circular ring distance between nodes i and j.
#   distance_sequence : integer vector seq_len(max(distance_matrix))
#                       enumerating every distinct ring distance to process.
#
#   Returns
#   -------
#   pairs : list of length length(distance_sequence).  Element [[r]] is a
#           two-column integer matrix with columns "row" and "col" (arr.ind
#           layout from which()), with row < col, listing every pair of nodes
#           at ring distance r.
{

  # Loop over distances
  return(
    lapply(distance_sequence, function(i){

      # Obtain pairs
      pairs <- which(distance_matrix == i, arr.ind = TRUE)

      # Return pairs
      return(pairs[pairs[,"row"] < pairs[,"col"],])

    }
    )
  )

}


# Proximity construction ----
# Updated 24.03.2026
proximity_pass <- function(nodes, ring, budget, pairs, distance_sequence)
# PROXIMITY_PASS  Greedy edge assignment in strictly increasing ring-distance order.
#
#   Translated directly from proximity_pass_c in proxswap.c.
#
#   Starting from the supplied (initially empty) ring adjacency, edges are
#   added one distance band at a time.  Within each band, eligible pairs --
#   those where both endpoints still have remaining degree budget and are not
#   yet connected -- are sorted by descending min(budget_i, budget_j) so that
#   the most degree-deficient pairs receive the shortest available connections
#   first.  Each pair is then considered in that order, with budgets re-checked
#   immediately before assignment because earlier placements in the same band
#   can exhaust a node's budget.  The loop exits early once all degree budgets
#   reach zero.
#
#   Arguments
#   ---------
#   nodes             : scalar integer, number of nodes in the ring.
#   ring              : nodes x nodes logical adjacency matrix (typically all
#                       FALSE at the start of a pass; modified in place within
#                       the function via R's copy-on-modify semantics).
#   budget            : named integer vector of length nodes, remaining degree
#                       deficits under the shuffled degree assignment.
#   pairs             : list produced by build_pairs(); element [[r]] contains
#                       all upper-triangle pairs at ring distance r as a
#                       two-column matrix with columns "row" and "col".
#   distance_sequence : integer vector seq_len(max_distance) controlling the
#                       order in which distance bands are processed.
#
#   Returns
#   -------
#   Named list with three elements:
#     $ring         -- updated logical nodes x nodes adjacency matrix.
#     $budget       -- updated integer degree-deficit vector (all zeros if
#                      fully satisfied).
#     $total_budget -- integer sum of remaining deficits (0 on full success).
{

  # Remaining edges to assign per node
  total_budget <- sum(budget)

  # Add edges in increasing ring distance
  for(i in distance_sequence){

    # Retrieve precomputed node-index vectors for this distance band
    rows <- pairs[[i]][,"row"]
    columns <- pairs[[i]][,"col"]

    # Determine eligible
    eligible <- (budget[rows] > 0) & (budget[columns] > 0) & (!ring[cbind(rows, columns)])

    # Skip this distance band if nothing is eligible
    if(!any(eligible)){
      next
    }

    # Eligible candidate indices
    row <- rows[eligible]
    col <- columns[eligible]

    # Get row length
    row_length  <- length(row)

    # Sort: highest min-budget pair first so high-need pairs get short edges
    if(row_length > 1){
      target <- order(pmin(budget[row], budget[col]), decreasing = TRUE)
      row  <- row[target]
      col  <- col[target]
    }

    # Assign edges one at a time; re-check both budgets before each assignment
    for(l in seq_len(row_length)){

      # Set targets
      target_row <- row[l]
      target_col <- col[l]

      # Check budget (can change with loop)
      if((budget[target_row] > 0) && (budget[target_col] > 0)){

        # Add to ring
        ring[target_row, target_col] <- ring[target_col, target_row] <- TRUE

        # Update budget
        budget[target_row] <- budget[target_row] - 1
        budget[target_col] <- budget[target_col] - 1
        total_budget <- total_budget - 2

      }
    }

    # Early exit once all budgets are satisfied
    if(total_budget < 1){
      break
    }

  }

  # Return ring and total absolute degree error
  return(list(ring = ring, budget = budget, total_budget = total_budget))

}


# Swap to amend lattice ----
# Updated 24.03.2026
swapping_pass <- function(nodes, node_sequence, ring, budget, total_budget, distance_sequence)
# SWAPPING_PASS  Resolve residual degree deficit via direct connection or edge swap.
#
#   Translated directly from swapping_pass_c in proxswap.c.
#
#   At each iteration the node with the largest remaining deficit (i) is
#   targeted.  Ring positions are scanned in interleaved clockwise /
#   counter-clockwise order (nearest first), as produced by interleave(),
#   to find a connection for i:
#
#     Direct connection: if any scanned node j has remaining budget and is
#     not yet connected to i, the edge (i, j) is added immediately and the
#     iteration continues.
#
#     Edge swap: if no direct partner exists, the scan finds the nearest
#     unconnected node j that has at least one existing neighbour k not
#     already connected to i.  The edge (j, k) is removed and (i, j) is
#     added.  Node k recovers its budget and will be re-targeted in a later
#     iteration.  Because one deficit is transferred rather than eliminated,
#     total_budget is unchanged by a swap.
#
#   Iteration stops when total_budget reaches zero, when no swap can be
#   found, or after the hard cap of 2 * nodes^2 iterations.
#
#   Arguments
#   ---------
#   nodes             : scalar integer, number of nodes in the ring.
#   node_sequence     : integer vector seq_len(nodes), used to index
#                       neighbours when searching for swap candidates;
#                       passed in to avoid re-creation on every call.
#   ring              : nodes x nodes logical adjacency matrix, as returned
#                       by proximity_pass().
#   budget            : integer vector of remaining degree deficits, as
#                       returned by proximity_pass().
#   total_budget      : scalar integer sum of budget, as returned by
#                       proximity_pass().
#   distance_sequence : integer vector seq_len(floor(nodes / 2)), passed
#                       directly to interleave() to generate the clockwise /
#                       counter-clockwise scan order.
#
#   Returns
#   -------
#   Named list with two elements:
#     $ring      -- updated logical nodes x nodes adjacency matrix.
#     $remaining -- integer total unresolved degree deficit (0 on full success).
{

  # Set maximum iterations
  for(iter in seq_len(nodes * nodes * 2)){

    # Stop when all budgets are satisfied
    if(total_budget == 0){
      break
    }

    # Work on the node with the largest remaining deficit
    i <- which.max(budget)
    if(budget[i] == 0){
      break
    }

    # Interleave clockwise and counter-clockwise ring positions by distance
    distance <- interleave(i, nodes, distance_sequence)

    # Set avaiable ring (diagonal will be TRUE but will never be in 'distance')
    available_ring <- !ring

    # Try direct connection to nearest node that has budget and no existing edge
    direct <- distance[(budget[distance] > 0) & (available_ring[i, distance])]

    # Check connection
    if(length(direct) > 0){

      # Set index
      target <- direct[1]

      # Update ring
      ring[i, target] <- ring[target, i] <- TRUE

      # Update budget
      budget[i] <- budget[i] - 1
      budget[target] <- budget[target] - 1
      total_budget <- total_budget - 2

      # Continue
      next

    }

    # Initialize swap
    swapped <- FALSE

    # Check nearby unconnected nodes
    candidates <- distance[available_ring[i, distance]]

    # Loop over candidates
    for(j in candidates){

      # Find a neighbor of the neighbor that is not connected to current node
      neighborhood <- which(ring[j,] & node_sequence != i & available_ring[i,])

      # Check if any are available
      if(length(neighborhood) > 0){

        # Select first available neighbor
        k <- neighborhood[1]

        # Perform swap
        ring[j, k] <- ring[k, j] <- FALSE
        ring[i, j] <- ring[j, i] <- TRUE

        # Update budgets
        budget[i] <- budget[i] - 1
        budget[k] <- budget[k] + 1

        # Total budget is unchanged with swap
        swapped <- TRUE

        # Break out of loop
        break

      }

    }

    # If completely struck, just break
    if(!swapped){
      break
    }

  }

  # Return ring and remaining
  return(list(ring = ring, remaining = total_budget))

}


# Interleaving function ----
# Updated 24.03.2026
interleave <- function(index, nodes, distance_sequence)
# INTERLEAVE  Build an interleaved clockwise / counter-clockwise position list.
#
#   Generates the scan order used by swapping_pass() when searching for a
#   direct connection or swap partner for node 'index'.  Starting from
#   'index', positions at ring distance d = 1, 2, ... are emitted in
#   alternating clockwise / counter-clockwise pairs, yielding a sequence
#   that visits the nearest ring neighbours first.  Duplicate positions
#   (which can occur when nodes is even at the maximum distance) are
#   removed via unique().
#
#   Arguments
#   ---------
#   index             : scalar integer (1-based), the node whose neighbours
#                       are to be enumerated.
#   nodes             : scalar integer, total number of nodes in the ring.
#   distance_sequence : integer vector seq_len(floor(nodes / 2)),
#                       the distances at which to emit clockwise /
#                       counter-clockwise positions.
#
#   Returns
#   -------
#   Integer vector of 1-based node indices in interleaved nearest-first
#   order, with duplicates removed.  Length is at most 2 * floor(nodes / 2),
#   and may be shorter when nodes is odd or duplicates are dropped.
{

  return(
    unique(
      c(
        rbind(
          (index - 1 + distance_sequence) %% nodes + 1, # clockwise
          (index - 1 - distance_sequence + nodes * distance_sequence) %% nodes + 1
          # counterclockwise
        )
      )
    )
  )

}


# Assign weights based on distance ----
# Updated 26.03.2026
assign_weights <- function(network, A, distance_matrix)
# ASSIGN_WEIGHTS  Reassign edge weights from 'network' onto the binary lattice A.
#
#   Following Muldoon, Bridgeford, & Bassett (2016): observed edge weights are
#   sorted in descending order by absolute value and mapped onto lattice edges
#   ranked by ascending ring distance (d_ij = min(|i-j|, n-|i-j|)), so that
#   shorter (more local) connections receive the largest-magnitude weights.
#   This uses the ring's structural distances directly rather than any
#   network-derived proxy.  Original signed weights are preserved.  Ties in
#   ring distance are broken at random via rank(..., ties.method = "random").
#
#   Arguments
#   ---------
#   network         : n x n numeric matrix, the original weighted network as
#                     passed into proxswap_lattice() (with 'weighted = TRUE').
#                     Non-zero lower-triangle entries supply the weight pool;
#                     weights are extracted in absolute-value descending order
#                     but assigned with their original signs.
#   A               : n x n logical or numeric binary adjacency matrix (the
#                     lattice topology, as returned by swapping_pass()).
#   distance_matrix : n x n integer matrix where entry [i, j] holds the
#                     circular ring distance between nodes i and j, as
#                     computed in proxswap_lattice().
#
#   Returns
#   -------
#   n x n symmetric numeric matrix with reassigned edge weights.  Lower-
#   triangle entries are set to the reordered weights; the upper triangle
#   is filled by transposition.  Off-lattice entries are zero.
{

  # Obtain weights
  lower_triangle <- lower.tri(network)
  network_nonzero <- network[lower_triangle] != 0
  weights <- network[lower_triangle][network_nonzero][
    order(abs(network)[lower_triangle][network_nonzero], decreasing = TRUE)
  ]

  # Set weights
  A_nonzero <- A[lower_triangle] != 0

  # Set weights order
  # Follows: Muldoon, Bridgeford, & Bassett's (2016) implementation
  weight_order <- rank(distance_matrix[lower_triangle][A_nonzero], ties.method = "random")
  A[lower_triangle][A_nonzero] <- weights[weight_order]
  A[!lower_triangle] <- 0
  A <- A + t(A) # make symmetric

  # Return weighted lattice
  return(A)

}
