// Browser-side re-clustering: union-find over the mutual-kNN edges shipped by
// /api/corpus/embedding-map. The edge list already passed the mutual-kNN gate
// server-side, so filtering by the slider threshold reproduces the backend
// clustering (generate_categories.js) exactly at any threshold.

class UnionFind {
  constructor(n) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.rank = new Array(n).fill(0);
  }

  find(x) {
    while (this.parent[x] !== x) {
      this.parent[x] = this.parent[this.parent[x]];
      x = this.parent[x];
    }
    return x;
  }

  union(x, y) {
    const px = this.find(x), py = this.find(y);
    if (px === py) return;
    if (this.rank[px] < this.rank[py]) this.parent[px] = py;
    else if (this.rank[px] > this.rank[py]) this.parent[py] = px;
    else { this.parent[py] = px; this.rank[px]++; }
  }
}

/**
 * Cluster `n` points with edges filtered at `threshold`.
 * Returns { assign, clusters }: `assign[i]` is a dense cluster index ordered
 * by cluster size (0 = largest — stable palette slots for the biggest
 * clusters), `clusters` is [{ index, members }] in that same order.
 */
export function clusterize(n, edges, threshold) {
  const uf = new UnionFind(n);
  for (const { i, j, sim } of edges) {
    if (sim >= threshold) uf.union(i, j);
  }

  const byRoot = new Map();
  for (let i = 0; i < n; i++) {
    const root = uf.find(i);
    if (!byRoot.has(root)) byRoot.set(root, []);
    byRoot.get(root).push(i);
  }

  // Size-descending, first-member tiebreak so slot colors are stable while
  // the slider moves (largest topics keep their hue).
  const groups = [...byRoot.values()].sort(
    (a, b) => b.length - a.length || a[0] - b[0]
  );

  const assign = new Int32Array(n);
  groups.forEach((members, idx) => {
    for (const m of members) assign[m] = idx;
  });

  return { assign, clusters: groups.map((members, index) => ({ index, members })) };
}
