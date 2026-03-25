# Degree-Preserving Ring Lattice via Proximity-Swap Construction

**Alexander P. Christensen · 24 March 2026 · CC0-1.0**

---

## Overview

Small-world network analysis requires comparing an observed network against two null models: a **random graph** (low clustering, short path lengths) and a **regular lattice** (high clustering, long path lengths). The small-world index is typically computed as:

$$\sigma = \frac{C / C_L}{L / L_R}$$

where $C$ and $L$ are the clustering coefficient and average path length of the observed network, $C_L$ is the clustering coefficient of the lattice, and $L_R$ is the average path length of the random graph.

The lattice null model is most informative when it **preserves the degree sequence** of the observed network — that is, each node in the lattice has exactly the same number of connections as in the original. Standard ring lattices (Watts–Strogatz) assign a uniform degree to all nodes, which distorts the comparison for real-world networks whose degrees vary substantially across nodes.

`proxswap-lattice` constructs a **degree-preserving ring lattice**: a connected ring graph whose per-node degree sequence matches the observed network exactly, while maximising the average local clustering coefficient. This makes it a principled lattice null model for heterogeneous-degree networks.

---

## Algorithm

### 1. Pair precomputation

A circular distance matrix is computed as $d_{ij} = \min(|i-j|,\, n - |i-j|)$, giving true ring distances bounded by $\lfloor n/2 \rfloor$. All unique unordered pairs $\{i, j\}$ (with $i < j$) at each distance $r = 1, \ldots, \lfloor n/2 \rfloor$ are extracted once and cached for reuse across passes.

### 2. Proximity construction

The observed degree sequence is **randomly permuted** onto ring positions. Starting from an empty graph, edges are added in strictly increasing distance order. Within each distance band, eligible pairs — both endpoints have remaining degree budget and are not yet connected — are sorted by descending $\min(\text{budget}_i, \text{budget}_j)$, so the most degree-deficient nodes receive the shortest connections first. Pairs are assigned sequentially with per-pair budget re-checks. The pass exits early once all budgets reach zero.

### 3. Swap repair

If any degree deficit remains after the proximity pass, a repair phase resolves it. The highest-deficit node $i$ is targeted at each iteration. Ring positions are scanned in interleaved clockwise/counter-clockwise order (nearest first):

- **Direct connection** — if a scanned node $j$ has remaining budget and is not yet connected to $i$, edge $(i, j)$ is added immediately.
- **Edge swap** — if no direct partner exists, a nearby unconnected node $j$ is found. One of $j$'s existing edges to $k$ is removed and replaced by $(i, j)$. Node $k$ recovers its budget and is re-targeted in a later iteration. The total degree deficit is unchanged by a swap; it is redistributed, not eliminated.

Iteration stops when all deficits are resolved, when no swap can be found, or after a hard cap of $2n^2$ iterations.

### 4. Pass selection

A pass is valid only if the resulting graph is **connected** and has **zero residual degree error**. Among all valid passes, the one with the highest average clustering coefficient is returned.

### 5. Empirical fallback

If no valid pass is found, or if the best lattice clustering coefficient is lower than that of the original network, the empirical (binarised) adjacency is returned with a warning.

---

## Available implementations

The algorithm is implemented in four languages. All ports are faithful translations of the same logic and produce identical results up to random-permutation differences across passes.

| File | Language | Dependencies | Notes |
|------|----------|--------------|-------|
| `proxswap_lattice.R` | R | [{igraph}](https://igraph.org/r/) · [{L0ggm}](https://github.com/AlexChristensen/L0ggm) internals (`convert2igraph`, `shuffle`, `swiftelse`) | Reference implementation; used in the {L0ggm} R package. Supports `weighted = TRUE` via `assign_weights()`. |
| `proxswap_lattice.m` | MATLAB | [Brain Connectivity Toolbox](https://sites.google.com/site/bctnet/) (`clustering_coef_bu`, `clustering_coef_wu`, `get_components`) | All three BCT files must be on the MATLAB path. Supports `weighted` argument. Ported with Claude Sonnet 4.6. |
| `proxswap_lattice.py` | Python | [NumPy](https://numpy.org) · [NetworkX](https://networkx.org) | Both required. NetworkX is used for `average_clustering` and `is_connected`, mirroring the igraph calls in the R implementation. Supports `weighted=True`. Ported with Claude Sonnet 4.6. |

---

## Dependencies

### R

`proxswap_lattice.R` is designed to run as part of the [{L0ggm}](https://github.com/AlexChristensen/L0ggm) R package and relies on three internal helpers from that package:

| Helper | Role |
|--------|------|
| `convert2igraph()` | Converts an adjacency matrix to an `igraph` object |
| `shuffle()` | Returns a random permutation of a sequence |
| `swiftelse()` | Vectorised `ifelse` used for the empirical-fallback selection |

The only external package dependency is [{igraph}](https://igraph.org/r/), used for `transitivity()` (average clustering coefficient) and `is_connected()`.

```r
install.packages("igraph")
```

### MATLAB

`proxswap_lattice.m` requires two functions from the [Brain Connectivity Toolbox (BCT)](https://sites.google.com/site/bctnet/):

| BCT function | Role |
|--------------|------|
| `clustering_coef_bu` | Computes per-node clustering coefficients for binary networks; the average is taken across all nodes |
| `clustering_coef_wu` | Computes per-node clustering coefficients for weighted networks; used when `weighted = true` |
| `get_components` | Connected-component labelling; a pass is accepted when exactly one component is returned |

All three files must be on the MATLAB path before calling `proxswap_lattice`.

### Python

`proxswap_lattice.py` requires both of the following:

| Package | Role |
|---------|------|
| [NumPy](https://numpy.org) | Array operations, distance matrix, sorting |
| [NetworkX](https://networkx.org) | `nx.average_clustering` (mirrors `igraph::transitivity(type="average")`) · `nx.is_connected` (mirrors `igraph::is_connected`) |

```bash
pip install numpy networkx
```

---

### R usage

```r
# Install {L0ggm} or source the file directly
source("proxswap_lattice.R")

# Estimate network and construct binary ring lattice
network <- network_estimation(data)
L <- proxswap_lattice(network, shuffles = 100)

# Retrieve attached clustering coefficient
attr(L, "CC")

# Degree sequences should match exactly
cbind(target = colSums(network != 0), achieved = colSums(L))

# Construct weighted ring lattice
L_weighted <- proxswap_lattice(network, weighted = TRUE, shuffles = 100)
attr(L_weighted, "CC")
```

### MATLAB usage

```matlab
% All functions are contained in proxswap_lattice.m
% Binary lattice (default)
[L, CC] = proxswap_lattice(network, false, 100);
fprintf('Average CC: %.4f\n', CC);

% Weighted lattice
[L_w, CC_w] = proxswap_lattice(network, true, 100);
fprintf('Weighted average CC: %.4f\n', CC_w);
```

### Python usage

```python
import numpy as np
from proxswap_lattice import proxswap_lattice

# Binary lattice (default)
ring, cc = proxswap_lattice(network, shuffles=100)
print("Average clustering coefficient:", cc)
print("Degree sequences match:",
      np.array_equal(network.astype(bool).sum(axis=0), ring.sum(axis=0)))

# Weighted lattice
ring_w, cc_w = proxswap_lattice(network, weighted=True, shuffles=100)
print("Weighted average clustering coefficient:", cc_w)
```

---

## License

CC0-1.0 — To the extent possible under law, the author(s) have dedicated all copyright and related and neighboring rights to this software to the public domain worldwide. This software is distributed without any warranty. See <https://creativecommons.org/publicdomain/zero/1.0/>.
