% proxswap_lattice.m
% ------------------
% MATLAB port of the proximity-swap ring lattice construction algorithm.
% Ported with Claude Sonnet 4.6 from proxswap_lattice.R / proxswap.c in
% the {L0ggm} R package (Alexander P. Christensen, 24 March 2026; CC0-1.0).
%
% License: CC0-1.0
% To the extent possible under law, the author(s) have dedicated all
% copyright and related and neighboring rights to this software to the
% public domain worldwide.  This software is distributed without any
% warranty.  See <https://creativecommons.org/publicdomain/zero/1.0/>.
%
% Dependencies
% ------------
%   Brain Connectivity Toolbox (BCT)
%     clustering_coef_bu  -- per-node clustering coefficients (Watts & Strogatz 1998)
%     get_components      -- connected-component labelling (Goni 2009/2011)
%   Both files must be on the MATLAB path before calling proxswap_lattice.
%
% Functions
% ---------
%   proxswap_lattice(network, weighted, shuffles)  -- main entry point
%   build_pairs(D, distance_seq)
%   proximity_pass(nodes, ring, budget, pairs, distance_seq)
%   swapping_pass(nodes, ring, budget, total_budget)
%   assign_weights(network, A)

function [ring, CC] = proxswap_lattice(network, weighted, shuffles)
% PROXSWAP_LATTICE  Construct a degree-preserving ring lattice via
%   proximity-swap construction.
%
%   RING = PROXSWAP_LATTICE(NETWORK) converts the square symmetric matrix
%   NETWORK into a connected ring lattice whose degree sequence exactly
%   matches the original, while maximising the average local clustering
%   coefficient.  Non-zero off-diagonal entries are treated as edges; the
%   binary adjacency is derived internally.  Isolated nodes (degree zero)
%   are supported.  Defaults to binary output and 100 shuffle passes.
%
%   [RING, CC] = PROXSWAP_LATTICE(NETWORK, WEIGHTED, SHUFFLES) also
%   returns the average clustering coefficient of the returned lattice
%   as CC.
%
%   Arguments
%   ---------
%   network  : n x n numeric matrix (weighted or binary, symmetric).
%              Absolute values are taken internally, so signed weights are
%              handled automatically.  Non-zero off-diagonal entries are
%              treated as edges; the binary adjacency is derived internally
%              as (network ~= 0).  Isolated nodes (degree zero) are
%              supported.
%   weighted : logical scalar, whether to return a weighted lattice
%              (default false).  When true, edge weights from NETWORK are
%              reassigned to the lattice topology following Muldoon,
%              Bridgeford, & Bassett (2016): shorter-distance lattice edges
%              receive larger weights, preserving the overall weight
%              distribution.  The clustering coefficient is then computed
%              via clustering_coef_wu (BCT) rather than clustering_coef_bu.
%              When false, a binary logical adjacency matrix is returned.
%   shuffles : positive integer, number of independent random permutation
%              passes to attempt (default 100).  Only passes producing a
%              connected graph with zero degree error are retained; the one
%              with the highest clustering coefficient is returned.
%
%   Returns
%   -------
%   ring : n x n symmetric matrix representing the ring lattice with the
%          best average clustering coefficient among all valid passes.
%          When WEIGHTED is false (default), entries are logical (0/1).
%          When WEIGHTED is true, entries contain the reassigned edge
%          weights from NETWORK.  When the empirical fallback is taken,
%          ring equals the binarised input (WEIGHTED false) or the absolute-
%          value input matrix (WEIGHTED true).
%   CC   : scalar double, average clustering coefficient of ring.  Equals
%          the empirical CC when the empirical-fallback path is taken.
%
%   Algorithm overview
%   ------------------
%   1. Pair precomputation   -- unique upper-triangle pairs at each ring
%      distance d = 1 ... floor(n/2) are cached once.
%   2. Proximity construction -- the degree sequence is randomly permuted
%      onto ring positions; edges are added in increasing distance order,
%      sorted by descending min(budget_i, budget_j) within each band.
%   3. Swap repair           -- any residual deficit is resolved by direct
%      connection or an edge swap, scanning interleaved clockwise /
%      counter-clockwise positions.
%   4. Weight assignment     -- when WEIGHTED is true, edge weights are
%      reassigned after the binary topology is finalised via ASSIGN_WEIGHTS.
%   5. Pass selection        -- only connected, zero-deficit passes are
%      kept; the highest-CC pass is returned.
%   6. Empirical fallback    -- if no valid pass beats the empirical CC,
%      the original adjacency (or weighted matrix) is returned with a
%      warning.
%
%   Dependencies
%   ------------
%   clustering_coef_bu  (BCT) -- per-node clustering coefficients for
%                                binary networks; average taken over nodes.
%   clustering_coef_wu  (BCT) -- per-node clustering coefficients for
%                                weighted networks; used when WEIGHTED true.
%   get_components      (BCT) -- connectivity confirmed when exactly one
%                                component is returned.

    if nargin < 2 || isempty(weighted)
        weighted = false;
    end
    if nargin < 3
        shuffles = 100;
    end

    % ------------------------------------------------------------------
    % Ensure absolute values (handles signed weight matrices)
    % ------------------------------------------------------------------
    network = abs(network);

    % ------------------------------------------------------------------
    % Derive binary adjacency
    % ------------------------------------------------------------------
    A = (network ~= 0);                         % logical, n x n

    nodes  = size(A, 2);
    degree = sum(A, 1);                         % 1 x n row vector

    % ------------------------------------------------------------------
    % Ring distance matrix  d_ij = min(|i-j|, n-|i-j|)
    % ------------------------------------------------------------------
    node_seq = 1:nodes;
    D = abs(bsxfun(@minus, node_seq', node_seq));
    D = min(D, nodes - D);
    distance_seq = 1:max(D(:));

    % ------------------------------------------------------------------
    % Pre-compute unique upper-triangle pairs at each ring distance
    % ------------------------------------------------------------------
    pairs = build_pairs(D, distance_seq);

    % ------------------------------------------------------------------
    % Empirical clustering coefficient (fallback reference).
    % Use clustering_coef_wu for weighted, clustering_coef_bu for binary.
    % ------------------------------------------------------------------
    if weighted
        empirical_CC = mean(clustering_coef_wu(network));
    else
        empirical_CC = mean(clustering_coef_bu(double(A)));
    end

    % ------------------------------------------------------------------
    % Initialise best-pass trackers
    % ------------------------------------------------------------------
    best_ring = [];
    best_swap = [];
    best_CC   = -Inf;

    ring_init = false(nodes);                   % empty starting graph

    % ------------------------------------------------------------------
    % Shuffle passes
    % ------------------------------------------------------------------
    for k = 1:shuffles

        % Random permutation of degree sequence onto ring positions
        swap_order = randperm(nodes);

        % Proximity construction (greedy, distance order)
        result = proximity_pass(nodes, ring_init, degree(swap_order), ...
                                pairs, distance_seq);

        % Swap repair (resolve residual deficit)
        result = swapping_pass(nodes, result.ring, result.budget, ...
                               result.total_budget);

        % Reject: residual deficit or disconnected graph.
        % get_components returns one entry in comp_sizes per component;
        % a connected graph has exactly one component.
        [~, comp_sizes] = get_components(double(result.ring));
        if result.remaining > 0 || numel(comp_sizes) ~= 1
            continue
        end

        % Assign weights to the valid binary topology when requested
        pass_ring = result.ring;
        if weighted
            pass_ring = assign_weights(network, pass_ring);
        end

        % Clustering coefficient for this valid pass.
        % Use clustering_coef_wu for weighted, clustering_coef_bu for binary.
        if weighted
            pass_CC = mean(clustering_coef_wu(pass_ring));
        else
            pass_CC = mean(clustering_coef_bu(double(pass_ring)));
        end

        % Keep the best
        if pass_CC > best_CC
            best_CC   = pass_CC;
            best_ring = pass_ring;
            best_swap = swap_order;
        end

    end % shuffle loop

    % ------------------------------------------------------------------
    % Empirical fallback check
    % ------------------------------------------------------------------
    empirical_flag = (empirical_CC > best_CC);

    if empirical_flag
        warning('proxswap_lattice:empiricalFallback', ...
            ['The lattice solution did not produce a better CC (%.3f) ' ...
             'than the empirical (%.3f).\nFalling back to empirical ' ...
             'solution...'], best_CC, empirical_CC);
    end

    % ------------------------------------------------------------------
    % Restore original node ordering  (invert the permutation)
    % ------------------------------------------------------------------
    if empirical_flag
        % Return weighted matrix or binary adjacency depending on mode
        if weighted
            ring = network;
        else
            ring = A;
        end
        CC   = empirical_CC;
    else
        % original_order is the inverse permutation of best_swap:
        %   best_swap(original_order) == 1:nodes
        original_order = zeros(1, nodes);
        original_order(best_swap) = 1:nodes;
        ring = best_ring(original_order, original_order);
        CC   = best_CC;
    end

end % proxswap_lattice


% ==========================================================================
%  build_pairs
% ==========================================================================
function pairs = build_pairs(D, distance_seq)
% BUILD_PAIRS  Pre-compute upper-triangle (i < j) node pairs at each ring distance.
%
%   For a ring of n nodes the circular distance between positions i and j is
%   min(|i - j|, n - |i - j|).  This helper extracts, for every integer
%   distance r = 1, ..., floor(n / 2), the set of unordered pairs {i, j}
%   (i < j) that sit exactly r steps apart on the ring.  Results are cached
%   once and reused across all shuffle passes in proxswap_lattice().
%
%   Arguments
%   ---------
%   D            : n x n integer matrix where entry (i, j) holds the circular
%                  ring distance between nodes i and j.
%   distance_seq : integer vector 1:max(D(:)) enumerating every distinct ring
%                  distance to process.
%
%   Returns
%   -------
%   pairs : 1 x numel(distance_seq) cell array.  Cell {r} is an M x 2 integer
%           matrix of [row, col] indices (1-based) with row < col, listing
%           every pair of nodes at ring distance r.  Empty cells may occur at
%           r = n/2 for even n.

    n_dist = numel(distance_seq);
    pairs  = cell(1, n_dist);

    for i = 1:n_dist
        [rows, cols] = find(D == i);
        mask      = rows < cols;
        pairs{i}  = [rows(mask), cols(mask)];   % M x 2
    end

end % build_pairs


% ==========================================================================
%  proximity_pass
% ==========================================================================
function result = proximity_pass(nodes, ring, budget, pairs, distance_seq)
% PROXIMITY_PASS  Greedy edge assignment in strictly increasing ring-distance order.
%
%   Translated directly from proximity_pass_c in proxswap.c.
%
%   Starting from the supplied (initially empty) ring adjacency, edges are
%   added one distance band at a time.  Within each band, eligible pairs --
%   those where both endpoints still have remaining degree budget and are not
%   yet connected -- are sorted by descending min(budget_i, budget_j) so that
%   the most degree-deficient pairs receive the shortest available connections
%   first.  Each pair is then considered in that order, with budgets re-checked
%   immediately before assignment because earlier placements in the same band
%   can exhaust a node's budget.  The loop exits early once all degree budgets
%   reach zero.
%
%   Arguments
%   ---------
%   nodes        : scalar integer, number of nodes in the ring.
%   ring         : nodes x nodes logical/numeric adjacency matrix (will be
%                  copied; typically all false at the start of a pass).
%   budget       : 1 x nodes integer vector of remaining degree deficits under
%                  the shuffled assignment.
%   pairs        : cell array produced by build_pairs(); cell {r} contains all
%                  upper-triangle pairs at ring distance r.
%   distance_seq : integer vector 1:max_distance controlling the order in which
%                  distance bands are processed.
%
%   Returns
%   -------
%   result : struct with fields:
%     .ring         -- updated logical nodes x nodes adjacency matrix.
%     .budget       -- updated integer degree-deficit column vector (zeros if
%                      fully satisfied).
%     .total_budget -- integer sum of remaining deficits (0 on full success).

    % Work on mutable copies; use double for arithmetic
    ring   = double(ring ~= 0);
    budget = double(budget(:));          % column vector, length nodes

    total_budget = sum(budget);
    n_dist = numel(distance_seq);

    for di = 1:n_dist

        p = pairs{di};                   % M x 2 matrix [row, col]
        if isempty(p)
            continue
        end

        row_idx = p(:,1);                % M x 1
        col_idx = p(:,2);                % M x 1

        % Column-major linear indices for fast ring lookup
        lin_idx = row_idx + (col_idx - 1) * nodes;

        % Eligibility: both endpoints have budget > 0, no existing edge
        elig_mask = (budget(row_idx) > 0) & ...
                    (budget(col_idx) > 0) & ...
                    (ring(lin_idx)   == 0);

        if ~any(elig_mask)
            continue
        end

        row_e = row_idx(elig_mask);      % eligible rows
        col_e = col_idx(elig_mask);      % eligible cols
        n_elig = numel(row_e);

        % Sort eligible pairs by descending min(budget[r], budget[c]).
        % MATLAB's sort is stable, matching R's order() tie-breaking.
        min_bud = min(budget(row_e), budget(col_e));
        [~, srt] = sort(min_bud, 'descend');
        row_e = row_e(srt);
        col_e = col_e(srt);

        % Assign edges one at a time; re-check budgets before each
        for l = 1:n_elig
            r = row_e(l);
            c = col_e(l);
            if budget(r) > 0 && budget(c) > 0
                ring(r, c) = 1;
                ring(c, r) = 1;
                budget(r) = budget(r) - 1;
                budget(c) = budget(c) - 1;
                total_budget = total_budget - 2;
            end
        end

        % Early exit once all degree deficits are satisfied
        if total_budget < 1
            break
        end

    end % distance band loop

    result.ring         = logical(ring);
    result.budget       = budget;
    result.total_budget = total_budget;

end % proximity_pass


% ==========================================================================
%  swapping_pass
% ==========================================================================
function result = swapping_pass(nodes, ring, budget, total_budget)
% SWAPPING_PASS  Resolve residual degree deficit via direct connection or edge swap.
%
%   Translated directly from swapping_pass_c in proxswap.c.
%
%   At each iteration the node with the largest remaining deficit (i) is
%   targeted.  Ring positions are scanned in interleaved clockwise /
%   counter-clockwise order (nearest first) to find a connection for i:
%
%     Direct connection: if any scanned node j has remaining budget and is not
%     yet connected to i, the edge (i, j) is added immediately and the
%     iteration continues.
%
%     Edge swap: if no direct partner exists, the scan finds the nearest
%     unconnected node j that has at least one existing neighbour k not already
%     connected to i.  The edge (j, k) is removed and (i, j) is added.  Node k
%     recovers its budget and will be re-targeted in a later iteration.  Because
%     one deficit is transferred rather than eliminated, total_budget is
%     unchanged by a swap.
%
%   Iteration stops when total_budget reaches zero, when no swap can be found,
%   or after the hard cap of 2 * nodes^2 iterations.
%
%   Arguments
%   ---------
%   nodes        : scalar integer, number of nodes in the ring.
%   ring         : nodes x nodes logical/numeric adjacency matrix (will be
%                  copied), as returned by proximity_pass().
%   budget       : column vector of remaining degree deficits, as returned by
%                  proximity_pass().
%   total_budget : scalar integer sum of budget, as returned by proximity_pass().
%
%   Returns
%   -------
%   result : struct with fields:
%     .ring      -- updated logical nodes x nodes adjacency matrix.
%     .remaining -- integer total unresolved degree deficit (0 on full success).

    ring   = double(ring ~= 0);
    budget = double(budget(:));          % column vector

    n_dist   = floor(nodes / 2);
    max_iter = 2 * nodes * nodes;

    for iter = 1:max_iter                %#ok<FORFLG>

        if total_budget == 0, break; end

        % Node with largest remaining deficit  (argmax, 1-indexed)
        [max_bud, i] = max(budget);
        if max_bud == 0, break; end

        % -----------------------------------------------------------------
        % Build interleaved clockwise / counter-clockwise position list
        % Mirrors the C code:  cw = (i + d) % n,  ccw = (i - d + n*d) % n
        % translated to 1-based indices.
        % -----------------------------------------------------------------
        visited = false(1, nodes);
        visited(i) = true;
        interleaved = zeros(1, 2 * n_dist);
        n_pos = 0;

        for d = 1:n_dist
            % Clockwise neighbour at ring distance d (1-indexed)
            cw = mod(i - 1 + d, nodes) + 1;
            if ~visited(cw)
                visited(cw)       = true;
                n_pos             = n_pos + 1;
                interleaved(n_pos) = cw;
            end

            % Counter-clockwise neighbour at ring distance d (1-indexed)
            % C: ((i - d) % n + n) % n   with 0-indexed i
            % MATLAB mod() is non-negative, so the +n guard is implicit.
            ccw = mod(i - 1 - d, nodes) + 1;
            if ~visited(ccw)
                visited(ccw)       = true;
                n_pos              = n_pos + 1;
                interleaved(n_pos) = ccw;
            end
        end

        interleaved = interleaved(1:n_pos);

        % -----------------------------------------------------------------
        % Attempt 1 -- direct connection to nearest node with budget & no edge
        % -----------------------------------------------------------------
        direct = 0;
        for pi = 1:n_pos
            j = interleaved(pi);
            if budget(j) > 0 && ring(i, j) == 0
                direct = j;
                break
            end
        end

        if direct > 0
            ring(i, direct) = 1;
            ring(direct, i) = 1;
            budget(i)      = budget(i)      - 1;
            budget(direct) = budget(direct) - 1;
            total_budget   = total_budget   - 2;
            continue
        end

        % -----------------------------------------------------------------
        % Attempt 2 -- edge swap
        % Find a nearby unconnected node j; remove one of j's edges (j,k);
        % add edge (i,j).  Node k recovers its budget for later resolution.
        % -----------------------------------------------------------------
        swapped = false;

        for pi = 1:n_pos
            if swapped, break; end
            j = interleaved(pi);
            if ring(i, j) ~= 0, continue; end  % already connected

            % Find first neighbour k of j that is not i and not connected to i
            for k = 1:nodes
                if k == i,            continue; end
                if ring(j, k) == 0,   continue; end  % not a neighbour of j
                if ring(i, k) ~= 0,   continue; end  % already connected to i

                % Perform swap: remove (j,k), add (i,j)
                ring(j, k) = 0;
                ring(k, j) = 0;
                ring(i, j) = 1;
                ring(j, i) = 1;

                budget(i) = budget(i) - 1;  % i gains an edge
                budget(k) = budget(k) + 1;  % k loses an edge (recovers budget)
                % total_budget is unchanged: budget[i]-1 and budget[k]+1 cancel

                swapped = true;
                break
            end
        end

        % Completely stuck -- exit early
        if ~swapped, break; end

    end % main repair loop

    result.ring      = logical(ring);
    result.remaining = total_budget;

end % swapping_pass


% ==========================================================================
%  assign_weights
% ==========================================================================
function A = assign_weights(network, A)
% ASSIGN_WEIGHTS  Reassign edge weights from NETWORK onto the binary lattice A.
%
%   Translated from assign_weights() in proxswap_lattice.R.
%
%   Following Muldoon, Bridgeford, & Bassett (2016): edge weights are mapped
%   onto the lattice such that shorter-distance lattice edges (quantified by
%   Euclidean distance between adjacency-row profiles) receive larger weights,
%   preserving the overall weight distribution of the original network.
%
%   Arguments
%   ---------
%   network : n x n numeric matrix, the original weighted network (absolute
%             values assumed already taken by the caller).  Non-zero lower-
%             triangle entries supply the weight pool.
%   A       : n x n logical or numeric binary adjacency matrix (the lattice
%             topology, as returned by swapping_pass).
%
%   Returns
%   -------
%   A : n x n symmetric double matrix with reassigned edge weights.

    n = size(network, 1);

    % Lower-triangle mask (strict: excludes diagonal)
    lt_mask = logical(tril(ones(n), -1));

    % Extract non-zero weights from lower triangle of network
    net_lt  = network(lt_mask);
    weights = net_lt(net_lt ~= 0);          % weight pool (column vector)

    % Identify non-zero edges in lower triangle of A
    A_lt      = double(A(lt_mask));
    A_nz_mask = A_lt ~= 0;                  % logical index into lt values

    % Pairwise Euclidean distances between adjacency-row profiles of A.
    % squareform(pdist(A)) mirrors R's as.matrix(dist(A)).
    dist_matrix = squareform(pdist(double(A)));
    dist_lt     = dist_matrix(lt_mask);
    edge_dists  = dist_lt(A_nz_mask);       % distances for lattice edges

    % Random-tiebreak rank, mirroring R's rank(..., ties.method = "random"):
    %   shuffle the distances, sort, then invert to obtain ranks.
    n_edges = numel(edge_dists);
    rp      = randperm(n_edges);
    [~, sort_idx]  = sort(edge_dists(rp));
    weight_order   = zeros(1, n_edges);
    weight_order(rp(sort_idx)) = 1:n_edges;

    % Write weights into lower triangle, zero upper triangle, symmetrise
    result          = zeros(n);
    lt_vals         = zeros(size(A_lt));
    lt_vals(A_nz_mask) = weights(weight_order);
    result(lt_mask) = lt_vals;
    A               = result + result';     % symmetric weighted matrix

end % assign_weights
