import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, OrthographicCamera } from '@react-three/drei';
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';
import { getEmbeddingMap } from '../api.js';
import { clusterize } from '../lib/cluster.js';
import { convexHull2D } from '../lib/hull.js';
import { seriesColor } from '../lib/theme.js';

const SCALE     = 2.4;   // UMAP coords arrive in [-1,1]; scene half-extent
const SLOTS     = 8;     // categorical palette slots; larger clusters win them
const PAD       = 0.16;  // blob padding around each member point (scene units)
const MIN_BLOB  = 4;     // clusters smaller than this are drawn as constellations
const OUTLINE   = 1.4;   // outline disc size relative to the coloured fill

// Round-point sprites: PointsMaterial draws squares, so tint a circular alpha
// texture per point instead. `disc` is the star body (used for both the grey
// outline layer and the coloured fill on top of it); `ring` marks the hover.
function makeSprite(kind) {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  if (kind === 'ring') {
    g.beginPath();
    g.arc(32, 32, 24, 0, Math.PI * 2);
    g.lineWidth = 6;
    g.strokeStyle = '#fff';
    g.stroke();
  } else {
    // Soft edge only (not a glow) so the grey outline layer beneath stays a
    // crisp rim rather than a smear.
    const grad = g.createRadialGradient(32, 32, 0, 32, 32, 30);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.82, 'rgba(255,255,255,1)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grad;
    g.fillRect(0, 0, 64, 64);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}

/**
 * Minimum spanning tree over a cluster's member points (Prim's). Small clusters
 * are drawn as constellations — each member linked to its nearest already-drawn
 * neighbour — rather than as a blob. An MST is what makes it read as a
 * constellation instead of a mesh: every member is connected, with the fewest,
 * shortest lines and no closed loops.
 */
function mstEdges(members, positions) {
  const dist = (a, b) => Math.hypot(
    positions[a * 3] - positions[b * 3],
    positions[a * 3 + 1] - positions[b * 3 + 1],
    positions[a * 3 + 2] - positions[b * 3 + 2],
  );
  const inTree = [members[0]];
  const rest = members.slice(1);
  const edges = [];
  while (rest.length) {
    let best = null;
    for (const a of inTree) {
      rest.forEach((b, k) => {
        const d = dist(a, b);
        if (!best || d < best.d) best = { a, b, k, d };
      });
    }
    edges.push([best.a, best.b]);
    inTree.push(best.b);
    rest.splice(best.k, 1);
  }
  return edges;
}

const centroidOf = (pts) => {
  const c = [0, 0, 0];
  for (const p of pts) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }
  return c.map((v) => v / pts.length);
};

/**
 * Padding shells. Instead of hulling the member points directly — which is
 * degenerate below 4 points (3D) or 3 points (2D), and forced an ugly
 * bounding-sphere fallback that ballooned to the cluster's full radius — we
 * hull a small shell of points around EACH member. The result is well-defined
 * at every cluster size and hugs the members: 1 doc gives a small ball, 2 docs
 * a capsule, N docs a padded hull. No special cases, no giant spheres.
 */
const shell3D = (p) => [
  [p[0] + PAD, p[1], p[2]], [p[0] - PAD, p[1], p[2]],
  [p[0], p[1] + PAD, p[2]], [p[0], p[1] - PAD, p[2]],
  [p[0], p[1], p[2] + PAD], [p[0], p[1], p[2] - PAD],
];

const shell2D = (p, segments = 12) =>
  Array.from({ length: segments }, (_, i) => {
    const a = (i / segments) * Math.PI * 2;
    return [p[0] + Math.cos(a) * PAD, p[1] + Math.sin(a) * PAD];
  });

function Scene({ data, mode, assign, showHulls, onHover, hoverIndex }) {
  const n = data.points.length;

  const dotTex  = useMemo(() => makeSprite('dot'), []);
  const ringTex = useMemo(() => makeSprite('ring'), []);

  const positions = useMemo(() => {
    const arr = new Float32Array(n * 3);
    data.points.forEach((p, i) => {
      const c = mode === '3d' ? p.p3 : [...p.p2, 0];
      arr[i * 3]     = c[0] * SCALE;
      arr[i * 3 + 1] = c[1] * SCALE;
      arr[i * 3 + 2] = (c[2] || 0) * SCALE;
    });
    return arr;
  }, [data, mode, n]);

  const colors = useMemo(() => {
    const arr = new Float32Array(n * 3);
    const col = new THREE.Color();
    for (let i = 0; i < n; i++) {
      col.set(seriesColor(assign[i]));
      arr[i * 3] = col.r; arr[i * 3 + 1] = col.g; arr[i * 3 + 2] = col.b;
    }
    return arr;
  }, [assign, n]);

  // Clusters are drawn two ways. A cluster of MIN_BLOB+ members gets a padded
  // convex-hull blob (hulling each member's padding shell, so the blob hugs its
  // members and never degenerates). A cluster below that is drawn as a
  // constellation: its members linked by MST lines in the cluster's own colour —
  // a 2- or 3-doc category reads as a star pattern rather than a bloated
  // capsule, which is both truer to the data and the look we want.
  const { hulls, constellations } = useMemo(() => {
    if (!showHulls) return { hulls: [], constellations: [] };

    const byCluster = new Map();
    for (let i = 0; i < n; i++) {
      if (assign[i] >= SLOTS) continue;              // "other" bucket gets neither
      if (!byCluster.has(assign[i])) byCluster.set(assign[i], []);
      byCluster.get(assign[i]).push(i);
    }

    const hulls = [];
    const constellations = [];

    for (const [slot, members] of byCluster) {
      const color = seriesColor(slot);

      if (members.length < MIN_BLOB) {
        if (members.length < 2) continue;            // a lone star needs no line
        const verts = new Float32Array(mstEdges(members, positions).flatMap(([a, b]) => [
          positions[a * 3], positions[a * 3 + 1], positions[a * 3 + 2],
          positions[b * 3], positions[b * 3 + 1], positions[b * 3 + 2],
        ]));
        constellations.push({ slot, color, verts });
        continue;
      }

      const pts = members.map((i) => [
        positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2],
      ]);
      const center = centroidOf(pts);

      if (mode === '3d') {
        const cloud = pts.flatMap(shell3D)
          .map((p) => new THREE.Vector3(p[0] - center[0], p[1] - center[1], p[2] - center[2]));
        let geometry;
        try {
          geometry = new ConvexGeometry(cloud);
          if (!(geometry.attributes.position?.count >= 4)) throw new Error('degenerate');
        } catch {
          geometry = new THREE.SphereGeometry(PAD, 16, 12);   // never drop a category
        }
        hulls.push({ slot, color, geometry, center });
      } else {
        const cloud = pts.flatMap((p) => shell2D(p));
        const outline = convexHull2D(cloud).map(([x, y]) => [x - center[0], y - center[1]]);
        const shape = new THREE.Shape(outline.map(([x, y]) => new THREE.Vector2(x, y)));
        hulls.push({
          slot, color, center,
          geometry: new THREE.ShapeGeometry(shape),
          outline,
        });
      }
    }
    return { hulls, constellations };
  }, [assign, positions, mode, showHulls, n]);

  // Dispose the previous frame's geometries — the slider rebuilds these on
  // every step, and orphaned BufferGeometry leaks GPU memory.
  const prev = useRef([]);
  useEffect(() => {
    const stale = prev.current;
    prev.current = hulls;
    return () => { for (const h of stale) if (!hulls.includes(h)) h.geometry.dispose(); };
  }, [hulls]);

  return (
    <>
      {mode === '3d'
        ? <PerspectiveCamera makeDefault position={[3.6, 2.8, 3.6]} fov={50} />
        : <OrthographicCamera makeDefault position={[0, 0, 10]} zoom={130} />}
      <OrbitControls
        key={mode}
        makeDefault
        enableRotate={mode === '3d'}
        enableDamping
        dampingFactor={0.12}
      />

      {hulls.map((h) => (
        <group key={`${mode}-${h.slot}`} position={h.center}>
          <mesh geometry={h.geometry} position={mode === '2d' ? [0, 0, -0.02] : undefined}>
            <meshBasicMaterial
              color={h.color}
              transparent
              opacity={mode === '3d' ? 0.1 : 0.14}
              depthWrite={false}
              side={THREE.DoubleSide}
            />
          </mesh>
          {h.outline && (
            <lineLoop position={[0, 0, -0.01]}>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  args={[new Float32Array(h.outline.flatMap(([x, y]) => [x, y, 0])), 3]}
                />
              </bufferGeometry>
              <lineBasicMaterial color={h.color} transparent opacity={0.5} />
            </lineLoop>
          )}
        </group>
      ))}

      {constellations.map((c) => (
        <lineSegments key={`c-${mode}-${c.slot}`} renderOrder={1}>
          <bufferGeometry key={`cg-${mode}-${c.slot}`}>
            <bufferAttribute attach="attributes-position" args={[c.verts, 3]} />
          </bufferGeometry>
          <lineBasicMaterial color={c.color} transparent opacity={0.55} depthWrite={false} />
        </lineSegments>
      ))}

      {/* Grey outline layer: the same disc drawn slightly larger in a neutral
          tone, directly beneath the coloured fill, so every star reads as a
          rimmed disc rather than a bare blob of colour. Two layers (rather than
          one baked sprite) because vertexColors would tint a baked rim too. */}
      <points key={`outline-${mode}`} renderOrder={2}>
        <bufferGeometry key={`og-${mode}`}>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        </bufferGeometry>
        <pointsMaterial
          map={dotTex}
          transparent
          alphaTest={0.02}
          color="#2c3a5e"
          size={(mode === '3d' ? 0.19 : 13) * OUTLINE}
          sizeAttenuation={mode === '3d'}
          depthWrite={false}
        />
      </points>

      <points
        key={mode}
        renderOrder={3}
        onPointerMove={(e) => {
          e.stopPropagation();
          if (e.index !== undefined && e.index !== hoverIndex) {
            onHover(e.index, e.nativeEvent.offsetX, e.nativeEvent.offsetY);
          }
        }}
        onPointerOut={() => onHover(-1)}
      >
        <bufferGeometry key={`g-${mode}`}>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
          <bufferAttribute attach="attributes-color" args={[colors, 3]} />
        </bufferGeometry>
        <pointsMaterial
          vertexColors
          map={dotTex}
          transparent
          alphaTest={0.02}
          size={mode === '3d' ? 0.19 : 13}
          sizeAttenuation={mode === '3d'}
          depthWrite={false}
        />
      </points>

      <HoverRing positions={positions} ringTex={ringTex} mode={mode} idx={hoverIndex} />
    </>
  );
}

// Camera-facing ring on the hovered doc — the "surface ring on overlapping
// marks" treatment, adapted to a point cloud.
function HoverRing({ positions, ringTex, mode, idx }) {
  if (idx < 0 || idx * 3 >= positions.length) return null;
  const pos = new Float32Array([
    positions[idx * 3], positions[idx * 3 + 1], positions[idx * 3 + 2],
  ]);
  return (
    <points renderOrder={4}>
      <bufferGeometry key={idx}>
        <bufferAttribute attach="attributes-position" args={[pos, 3]} />
      </bufferGeometry>
      <pointsMaterial
        map={ringTex}
        alphaTest={0.4}
        transparent
        color="#ffffff"
        size={mode === '3d' ? 0.3 : 22}
        sizeAttenuation={mode === '3d'}
        depthWrite={false}
      />
    </points>
  );
}

export default function EmbeddingSpace({ controlsEl, active }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const [mode, setMode] = useState('3d');
  const [showHulls, setShowHulls] = useState(true);
  const [tooltip, setTooltip] = useState(null);

  useEffect(() => {
    getEmbeddingMap()
      .then((d) => { setData(d); setThreshold(d.defaultThreshold ?? 0.5); })
      .catch((e) => setError(e.message));
  }, []);

  const { assign, clusters } = useMemo(() => {
    if (!data) return { assign: new Int32Array(0), clusters: [] };
    return clusterize(data.points.length, data.edges, threshold);
  }, [data, threshold]);

  const onHover = (index, x, y) =>
    setTooltip(index >= 0 ? { index, x, y } : null);

  if (error) {
    return (
      <div className="viz-empty">
        <h2>No embedding map yet</h2>
        <p>{error}</p>
        <p>Run the pipeline through the categorize stage, then reload.</p>
      </div>
    );
  }
  if (!data) return <div className="viz-empty"><p>Loading embedding space…</p></div>;

  const controls = (
    <>
      <div>
        <label className="control-label" htmlFor="threshold">
          Cluster threshold&ensp;
          <span className="control-value">{threshold.toFixed(3)}</span>
          <span style={{ fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
            {' '}· {clusters.length} categor{clusters.length === 1 ? 'y' : 'ies'}
          </span>
        </label>
        {/* step 0.001 over the full range: one pixel of travel is a small
            similarity change, so the blobs grow/shrink continuously instead of
            snapping between coarse stops. */}
        <input
          id="threshold"
          type="range"
          min="0" max="1" step="0.001"
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
        />
      </div>

      <div>
        <span className="control-label">Projection</span>
        <div className="seg" role="group" aria-label="Projection mode">
          <button className={mode === '3d' ? 'active' : ''} onClick={() => setMode('3d')}>3D</button>
          <button className={mode === '2d' ? 'active' : ''} onClick={() => setMode('2d')}>2D</button>
        </div>
      </div>

      <label className="check-row">
        <input type="checkbox" checked={showHulls} onChange={(e) => setShowHulls(e.target.checked)} />
        Category blobs
      </label>

      <div>
        <span className="control-label">Categories · {data.points.length} docs</span>
        <div className="legend">
          {clusters.slice(0, SLOTS).map((c) => (
            <div className="legend-row" key={c.index}>
              <span
                className="legend-swatch"
                style={{ background: seriesColor(c.index), color: seriesColor(c.index) }}
              />
              <span className="legend-title" title={data.points[c.members[0]].title}>
                {data.points[c.members[0]].title}
              </span>
              <span className="legend-count">{c.members.length}</span>
            </div>
          ))}
          {clusters.length > SLOTS && (
            <div className="legend-row">
              <span
                className="legend-swatch"
                style={{ background: seriesColor(SLOTS), color: seriesColor(SLOTS) }}
              />
              <span className="legend-title">{clusters.length - SLOTS} more categories</span>
              <span className="legend-count">
                {clusters.slice(SLOTS).reduce((s, c) => s + c.members.length, 0)}
              </span>
            </div>
          )}
        </div>
      </div>
    </>
  );

  return (
    <>
      <Canvas
        raycaster={{ params: { Points: { threshold: mode === '3d' ? 0.1 : 0.08 } } }}
        gl={{ antialias: true, alpha: true }}
        dpr={[1, 2]}
      >
        <Scene
          data={data}
          mode={mode}
          assign={assign}
          showHulls={showHulls}
          onHover={onHover}
          hoverIndex={tooltip ? tooltip.index : -1}
        />
      </Canvas>

      {tooltip && (
        <div className="viz-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
          <div>{data.points[tooltip.index].title}</div>
          <div className="sub">{data.points[tooltip.index].filename}</div>
        </div>
      )}

      <div className="viz-hint">
        {mode === '3d' ? 'drag to orbit · scroll to zoom' : 'drag to pan · scroll to zoom'}
        &nbsp;·&nbsp;hover a point for its title
      </div>

      {active && controlsEl && createPortal(controls, controlsEl)}
    </>
  );
}
