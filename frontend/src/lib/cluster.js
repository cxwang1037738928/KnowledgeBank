// Browser-side re-clustering: union-find over the mutual-kNN edges shipped by
// /api/corpus/embedding-map. The edge list already passed the mutual-kNN gate
// server-side, so filtering by the slider threshold reproduces the backend
// clustering (generate_categories.js) exactly at any threshold.

class UnionFind {
  constructor(pointCount) {
    this.parent = Array.from({ length: pointCount }, (_, pointIdx) => pointIdx);
    this.rank = new Array(pointCount).fill(0);
  }

  find(pointIdx) {
    while (this.parent[pointIdx] !== pointIdx) {
      this.parent[pointIdx] = this.parent[this.parent[pointIdx]];
      pointIdx = this.parent[pointIdx];
    }
    return pointIdx;
  }

  union(leftIdx, rightIdx) {
    const leftRoot = this.find(leftIdx), rightRoot = this.find(rightIdx);
    if (leftRoot === rightRoot) return;
    if (this.rank[leftRoot] < this.rank[rightRoot]) this.parent[leftRoot] = rightRoot;
    else if (this.rank[leftRoot] > this.rank[rightRoot]) this.parent[rightRoot] = leftRoot;
    else { this.parent[rightRoot] = leftRoot; this.rank[leftRoot]++; }
  }
}

/**
 * Cluster `pointCount` points with edges filtered at `threshold`.
 * Returns { clusterOfPoint, clusters }: `clusterOfPoint[i]` is a dense cluster
 * index ordered by cluster size (0 = largest — stable palette slots for the
 * biggest clusters), `clusters` is [{ index, members }] in that same order.
 */
export function clusterize(pointCount, edges, threshold) {
  const unionFind = new UnionFind(pointCount);
  for (const { i: sourceIdx, j: targetIdx, sim } of edges) {
    if (sim >= threshold) unionFind.union(sourceIdx, targetIdx);
  }

  const membersByRoot = new Map();
  for (let pointIdx = 0; pointIdx < pointCount; pointIdx++) {
    const root = unionFind.find(pointIdx);
    if (!membersByRoot.has(root)) membersByRoot.set(root, []);
    membersByRoot.get(root).push(pointIdx);
  }

  // Size-descending, first-member tiebreak so slot colors are stable while
  // the slider moves (largest topics keep their hue).
  const groups = [...membersByRoot.values()].sort(
    (groupA, groupB) => groupB.length - groupA.length || groupA[0] - groupB[0]
  );

  const clusterOfPoint = new Int32Array(pointCount);
  groups.forEach((members, clusterIdx) => {
    for (const memberIdx of members) clusterOfPoint[memberIdx] = clusterIdx;
  });

  return { clusterOfPoint, clusters: groups.map((members, index) => ({ index, members })) };
}
