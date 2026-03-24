# proxswap_lattice.R
# ------------------
# R implementation of the proximity-swap ring lattice construction algorithm.
# Author: Alexander P. Christensen <alexpaulchristensen@gmail.com>
# Updated: 24 March 2026
#
# License: CC0-1.0
# To the extent possible under law, the author(s) have dedicated all
# copyright and related and neighboring rights to this software to the
# public domain worldwide.  This software is distributed without any
# warranty.  See <https://creativecommons.org/publicdomain/zero/1.0/>.
#
# Functions
# ---------
#   proxswap_lattice(network, shuffles)  -- main entry point
#   build_pairs(distance_matrix, distance_sequence)
#   proximity_pass(nodes, ring, budget, pairs, distance_sequence)
#   swapping_pass(nodes, node_sequence, ring, budget, total_budget, distance_sequence)
#   interleave(index, nodes, distance_sequence)


# proxswap_lattice ----
# Updated 24.03.2026
#
# Converts a network matrix into a connected ring lattice whose degree sequence
# exactly matches the original, while maximizing the average local clustering
# coefficient.
#
# An adjacency matrix is derived automatically from 'network' (non-zero entries
# become edges), so the function accepts any weighted or binary network directly.
#
# Each pass randomly permutes the degree sequence onto ring positions, then
# greedily assigns edges in strictly increasing ring-distance order (proximity
# construction). Any residual degree deficit is resolved by a swap-repair phase.
# Passes that yield a disconnected graph or an unsatisfied degree sequence are
# discarded; among valid passes the one with the highest average clustering
# coefficient is returned. 'shuffles' independent passes are attempted in total.
#
# Arguments
# ---------
# network  : Matrix. A square, symmetric numeric matrix representing a network
#            (e.g., partial correlations). Non-zero off-diagonal entries are
#            treated as edges; the binary adjacency is derived internally as
#            (network != 0). Isolated nodes (degree zero) are supported.
#
# shuffles : Numeric (length = 1). Number of independent random permutation
#            passes to attempt. Each pass assigns the observed degree sequence
#            to ring positions in a new random order, runs the proximity
#            construction, and applies swap repair if needed. Only passes
#            producing a connected graph with zero degree error are retained;
#            the one with the highest clustering coefficient is returned.
#            Defaults to 100.
#
# Returns
# -------
# A square symmetric binary adjacency matrix of the same dimension as
# 'network', with row and column names preserved, representing the resulting
# ring lattice. The average clustering coefficient of the returned lattice is
# attached as the attribute "CC" and can be retrieved with attr(result, "CC").
#
# Algorithm
# ---------
# Pair precomputation:
#   A ring distance matrix is computed as d_ij = min(|i - j|, n - |i - j|),
#   giving true circular distances bounded by floor(n / 2). All unique
#   unordered pairs at each distance r = 1, ..., floor(n / 2) are extracted
#   once (retaining only entries where i < j) and cached for reuse across all
#   passes.
#
# Proximity construction:
#   The observed degree sequence is randomly permuted onto ring positions.
#   Starting from an empty graph, edges are added in distance order. At each
#   distance band the eligible pairs (both endpoints have remaining degree
#   budget and are not yet connected) are sorted by descending
#   min(budget_i, budget_j) so that high-need pairs receive short connections
#   first. Pairs are then assigned sequentially with per-pair budget re-checks.
#   The pass exits early once all budgets reach zero.
#
# Swap repair:
#   If any degree deficit remains after the proximity pass, the
#   highest-deficit node i is connected to its nearest available ring
#   neighbour, scanning clockwise and counter-clockwise positions in
#   interleaved distance order. When no direct partner with remaining budget
#   exists, an edge swap is performed: a nearby unconnected node j is found,
#   one of j's existing edges to k is removed, and a new edge (i, j) is added.
#   Node k recovers its budget for resolution in a subsequent iteration. This
#   repeats until all deficits are resolved or the iteration cap (2n^2) is
#   reached.
#
# Pass selection:
#   A pass is valid only if the resulting graph is connected and has zero
#   residual degree error. Among valid passes, the one with the highest average
#   clustering coefficient is returned.
#
# Empirical fallback:
#   If no valid pass is found, or if the best lattice clustering coefficient is
#   lower than that of the original network, the empirical adjacency is
#   returned with a warning.
#
# Examples
# --------
# # Get network
# network <- network_estimation(basic_smallworld)
#
# # Construct ring lattice
# L <- proxswap_lattice(network)
#
# # Retrieve the attached clustering coefficient
# attr(L, "CC")
#
# # Degree sequences should match exactly
# cbind(target = colSums(network != 0), achieved = colSums(L))
#
proxswap_lattice <- function(network, shuffles = 100)
{

  # Automatically construct adjacency
  A <- network != 0

  # Get nodes and degree
  nodes  <- dim(A)[2]
  degree <- colSums(A)

  # Set up distance matrix
  node_sequence <- seq_len(nodes)
  distance_matrix <- abs(outer(node_sequence, node_sequence, "-"))
  distance_matrix <- pmin(distance_matrix, nodes - distance_matrix)
  distance_sequence <- seq_len(max(distance_matrix))

  # Pre-compute unique node pairs at each ring distance
  pairs <- build_pairs(distance_matrix, distance_sequence)

  # Compute empirical clustering coefficient for fallback check
  empirical_CC <- igraph::transitivity(L0ggm:::convert2igraph(A), type = "average")

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

    # Convert ring to {igraph}
    iring <- L0ggm:::convert2igraph(result$ring)

    # Skip passes with residual or fully connected
    if((!igraph::is_connected(iring)) || (result$remaining > 0)){
      next
    }

    # With success, compute clustering coefficient for this pass
    pass_CC <- igraph::transitivity(iring, type = "average")

    # Update best pass
    if(pass_CC > best_CC){
      best_CC <- pass_CC; best_ring <- result$ring; best_swap <- swap_order
    }

  }

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
  ring <- L0ggm:::swiftelse(empirical_flag, A, best_ring[original_order, original_order])

  # Ensure named matrix
  dimnames(ring) <- dimnames(network)

  # Attach clustering coefficient
  attr(ring, "CC") <- L0ggm:::swiftelse(empirical_flag, empirical_CC, best_CC)

  # Return ring
  return(ring)

}

# build_pairs ----
# Updated 24.03.2026
#
# Pre-computes all unique upper-triangle node pairs at each ring distance.
#
# For a ring of n nodes the circular distance between positions i and j is
# min(|i - j|, n - |i - j|). This helper extracts, for every integer distance
# r = 1, ..., floor(n / 2), the set of unordered pairs {i, j} (i < j) that
# sit exactly r steps apart on the ring. Results are cached once and reused
# across all shuffle passes in proxswap_lattice().
#
# Arguments
# ---------
# distance_matrix   : n x n integer matrix where entry [i, j] holds the
#                     circular ring distance between nodes i and j.
# distance_sequence : Integer vector 1:max(distance_matrix) enumerating every
#                     distinct ring distance to process.
#
# Returns
# -------
# A list of length length(distance_sequence). Element [[r]] is a two-column
# integer matrix with columns "row" and "col" (both 1-indexed) listing every
# pair of nodes at ring distance r with row < col. Distances with no pairs
# (possible at r = n/2 for even n) return an empty matrix.
#
build_pairs <- function(distance_matrix, distance_sequence)
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

# proximity_pass ----
# Updated 24.03.2026
#
# Greedily assigns edges in strictly increasing ring-distance order.
#
# Starting from the supplied (initially empty) ring adjacency, edges are added
# one distance band at a time. Within each band, eligible pairs -- those where
# both endpoints still have remaining degree budget and are not yet connected --
# are sorted by descending min(budget_i, budget_j) so that the most
# degree-deficient pairs receive the shortest available connections first. Each
# pair is then considered in that order, with budgets re-checked immediately
# before assignment because earlier placements in the same band can exhaust a
# node's budget. The loop exits early once all degree budgets reach zero.
#
# Arguments
# ---------
# nodes             : Integer. Number of nodes in the ring.
# ring              : Logical n x n adjacency matrix (typically all FALSE at
#                     the start of a pass).
# budget            : Integer vector of length n giving the remaining degree
#                     deficit for each node under the shuffled assignment.
# pairs             : List produced by build_pairs(); element [[r]] contains
#                     all upper-triangle pairs at ring distance r.
# distance_sequence : Integer vector 1:max_distance controlling the order in
#                     which distance bands are processed.
#
# Returns
# -------
# A named list with three elements:
#   ring         -- updated logical n x n adjacency matrix.
#   budget       -- updated integer degree-deficit vector (zeros if fully
#                   satisfied).
#   total_budget -- integer sum of remaining deficits (0 on full success).
#
proximity_pass <- function(nodes, ring, budget, pairs, distance_sequence)
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

# swapping_pass ----
# Updated 24.03.2026
#
# Resolves any residual degree deficit remaining after proximity_pass().
#
# At each iteration the node with the largest remaining deficit (i) is
# targeted. Ring positions are scanned in interleaved clockwise /
# counter-clockwise order (nearest first) to find a connection for i:
#
#   Direct connection: if any scanned node j has remaining budget and is not
#   yet connected to i, the edge (i, j) is added immediately and the
#   iteration continues.
#
#   Edge swap: if no direct partner exists, the scan finds the nearest
#   unconnected node j that has at least one existing neighbour k not already
#   connected to i. The edge (j, k) is removed and (i, j) is added. Node k
#   recovers its budget and will be re-targeted in a later iteration. Because
#   one deficit is transferred rather than eliminated, total_budget is
#   unchanged by a swap.
#
# Iteration stops when total_budget reaches zero, when no swap can be found,
# or after the hard cap of 2 * n^2 iterations.
#
# Arguments
# ---------
# nodes             : Integer. Number of nodes in the ring.
# node_sequence     : Integer vector 1:nodes used for neighbourhood lookup.
# ring              : Logical or integer n x n adjacency matrix, as returned
#                     by proximity_pass().
# budget            : Integer degree-deficit vector, as returned by
#                     proximity_pass().
# total_budget      : Integer sum of budget, as returned by proximity_pass().
# distance_sequence : Integer vector 1:floor(n/2) passed to interleave() to
#                     control the clockwise / counter-clockwise scan order.
#
# Returns
# -------
# A named list with two elements:
#   ring      -- updated logical n x n adjacency matrix.
#   remaining -- integer total unresolved degree deficit (0 on full success).
#
swapping_pass <- function(nodes, node_sequence, ring, budget, total_budget, distance_sequence)
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
        total_budget <- sum(total_budget)
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

# interleave ----
# Updated 24.03.2026
#
# Generates an interleaved sequence of ring positions around a given node,
# alternating between clockwise and counter-clockwise neighbours in strictly
# increasing distance order.
#
# For a node at position 'index' on a ring of 'nodes' positions, the function
# pairs each distance d in 'distance_sequence' with its clockwise neighbour
# at (index - 1 + d) %% nodes + 1 and its counter-clockwise neighbour at
# (index - 1 - d + nodes * d) %% nodes + 1. The two sequences are interleaved
# via rbind() so positions closest to 'index' appear first. Duplicate entries
# (which occur at distance floor(n / 2) when n is even and the two directions
# meet) are removed by unique().
#
# Arguments
# ---------
# index             : Integer (1-indexed). The focal node whose neighbourhood
#                     is being enumerated.
# nodes             : Integer. Total number of nodes on the ring.
# distance_sequence : Integer vector 1:floor(n/2) giving the ring distances
#                     to include.
#
# Returns
# -------
# An integer vector of node indices (1-indexed, no duplicates) ordered by
# ascending ring distance from 'index', with clockwise positions preceding
# counter-clockwise positions within each distance tier.
#
interleave <- function(index, nodes, distance_sequence)
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
