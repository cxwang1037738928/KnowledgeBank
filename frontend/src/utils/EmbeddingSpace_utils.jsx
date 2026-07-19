/**
 * EmbeddingSpace_utils.jsx — non-component helpers for EmbeddingSpace.jsx:
 * point sprites, constellation MSTs, and the padded hull shells that give
 * every cluster a well-defined blob.
 */

import * as THREE from 'three';

export const PAD = 0.16;  // blob padding around each member point (scene units)

// Round-point sprites: PointsMaterial draws squares, so tint a circular alpha
// texture per point instead. `disc` is the star body (used for both the grey
// outline layer and the coloured fill on top of it); `ring` marks the hover.
export function makeSprite(kind) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 64;
  const ctx = canvas.getContext('2d');
  if (kind === 'ring') {
    ctx.beginPath();
    ctx.arc(32, 32, 24, 0, Math.PI * 2);
    ctx.lineWidth = 6;
    ctx.strokeStyle = '#fff';
    ctx.stroke();
  } else {
    // Soft edge only (not a glow) so the grey outline layer beneath stays a
    // crisp rim rather than a smear.
    const edgeGradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 30);
    edgeGradient.addColorStop(0, 'rgba(255,255,255,1)');
    edgeGradient.addColorStop(0.82, 'rgba(255,255,255,1)');
    edgeGradient.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = edgeGradient;
    ctx.fillRect(0, 0, 64, 64);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

/**
 * Minimum spanning tree over a cluster's member points (Prim's). Small clusters
 * are drawn as constellations — each member linked to its nearest already-drawn
 * neighbour — rather than as a blob. An MST is what makes it read as a
 * constellation instead of a mesh: every member is connected, with the fewest,
 * shortest lines and no closed loops.
 */
export function mstEdges(members, positions) {
  const distance = (fromMember, toMember) => Math.hypot(
    positions[fromMember * 3] - positions[toMember * 3],
    positions[fromMember * 3 + 1] - positions[toMember * 3 + 1],
    positions[fromMember * 3 + 2] - positions[toMember * 3 + 2],
  );
  const inTree = [members[0]];
  const remaining = members.slice(1);
  const edges = [];
  while (remaining.length) {
    let nearest = null;
    for (const treeMember of inTree) {
      remaining.forEach((candidate, remainingIdx) => {
        const candidateDistance = distance(treeMember, candidate);
        if (!nearest || candidateDistance < nearest.distance) {
          nearest = { treeMember, candidate, remainingIdx, distance: candidateDistance };
        }
      });
    }
    edges.push([nearest.treeMember, nearest.candidate]);
    inTree.push(nearest.candidate);
    remaining.splice(nearest.remainingIdx, 1);
  }
  return edges;
}

export const centroidOf = (points) => {
  const sum = [0, 0, 0];
  for (const point of points) { sum[0] += point[0]; sum[1] += point[1]; sum[2] += point[2]; }
  return sum.map((axisSum) => axisSum / points.length);
};

/**
 * Padding shells. Instead of hulling the member points directly — which is
 * degenerate below 4 points (3D) or 3 points (2D), and forced an ugly
 * bounding-sphere fallback that ballooned to the cluster's full radius — we
 * hull a small shell of points around EACH member. The result is well-defined
 * at every cluster size and hugs the members: 1 doc gives a small ball, 2 docs
 * a capsule, N docs a padded hull. No special cases, no giant spheres.
 */
export const shell3D = (point) => [
  [point[0] + PAD, point[1], point[2]], [point[0] - PAD, point[1], point[2]],
  [point[0], point[1] + PAD, point[2]], [point[0], point[1] - PAD, point[2]],
  [point[0], point[1], point[2] + PAD], [point[0], point[1], point[2] - PAD],
];

export const shell2D = (point, segments = 12) =>
  Array.from({ length: segments }, (_, segmentIdx) => {
    const angle = (segmentIdx / segments) * Math.PI * 2;
    return [point[0] + Math.cos(angle) * PAD, point[1] + Math.sin(angle) * PAD];
  });
