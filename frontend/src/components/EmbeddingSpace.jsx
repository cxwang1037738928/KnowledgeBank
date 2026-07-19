import { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, OrthographicCamera } from '@react-three/drei';
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';
import { getEmbeddingMap } from '../api.js';
import { clusterize } from '../lib/cluster.js';
import { convexHull2D } from '../lib/hull.js';
import { seriesColor } from '../lib/theme.js';
import {
  PAD, makeSprite, mstEdges, centroidOf, shell3D, shell2D,
} from '../utils/EmbeddingSpace_utils.jsx';

const SCALE     = 2.4;   // UMAP coords arrive in [-1,1]; scene half-extent
const SLOTS     = 8;     // categorical palette slots; larger clusters win them
const MIN_BLOB  = 4;     // clusters smaller than this are drawn as constellations
const OUTLINE   = 1.4;   // outline disc size relative to the coloured fill

function Scene({ embeddingMap, mode, clusterOfPoint, showHulls, onHover, hoveredIndex }) {
  const pointCount = embeddingMap.points.length;

  const dotTex  = useMemo(() => makeSprite('dot'), []);
  const ringTex = useMemo(() => makeSprite('ring'), []);

  const positions = useMemo(() => {
    const positionArray = new Float32Array(pointCount * 3);
    embeddingMap.points.forEach((docPoint, pointIdx) => {
      const coords = mode === '3d' ? docPoint.p3 : [...docPoint.p2, 0];
      positionArray[pointIdx * 3]     = coords[0] * SCALE;
      positionArray[pointIdx * 3 + 1] = coords[1] * SCALE;
      positionArray[pointIdx * 3 + 2] = (coords[2] || 0) * SCALE;
    });
    return positionArray;
  }, [embeddingMap, mode, pointCount]);

  const colors = useMemo(() => {
    const colorArray = new Float32Array(pointCount * 3);
    const slotColor = new THREE.Color();
    for (let pointIdx = 0; pointIdx < pointCount; pointIdx++) {
      slotColor.set(seriesColor(clusterOfPoint[pointIdx]));
      colorArray[pointIdx * 3] = slotColor.r;
      colorArray[pointIdx * 3 + 1] = slotColor.g;
      colorArray[pointIdx * 3 + 2] = slotColor.b;
    }
    return colorArray;
  }, [clusterOfPoint, pointCount]);

  // Two treatments per cluster: MIN_BLOB+ members get a padded convex-hull blob
  // (see shell3D/shell2D); smaller ones get MST constellation lines, so a 2-doc
  // category reads as a star pattern instead of a bloated capsule.
  const { hulls, constellations } = useMemo(() => {
    if (!showHulls) return { hulls: [], constellations: [] };

    const membersBySlot = new Map();
    for (let pointIdx = 0; pointIdx < pointCount; pointIdx++) {
      const slot = clusterOfPoint[pointIdx];
      if (slot >= SLOTS) continue;                   // "other" bucket gets neither
      if (!membersBySlot.has(slot)) membersBySlot.set(slot, []);
      membersBySlot.get(slot).push(pointIdx);
    }

    const hulls = [];
    const constellations = [];

    for (const [slot, members] of membersBySlot) {
      const color = seriesColor(slot);

      if (members.length < MIN_BLOB) {
        if (members.length < 2) continue;            // a lone star needs no line
        const lineVertices = new Float32Array(
          mstEdges(members, positions).flatMap(([fromMember, toMember]) => [
            positions[fromMember * 3], positions[fromMember * 3 + 1], positions[fromMember * 3 + 2],
            positions[toMember * 3], positions[toMember * 3 + 1], positions[toMember * 3 + 2],
          ]));
        constellations.push({ slot, color, lineVertices });
        continue;
      }

      const memberPoints = members.map((memberIdx) => [
        positions[memberIdx * 3], positions[memberIdx * 3 + 1], positions[memberIdx * 3 + 2],
      ]);
      const center = centroidOf(memberPoints);

      if (mode === '3d') {
        const shellPoints = memberPoints.flatMap(shell3D)
          .map((point) => new THREE.Vector3(
            point[0] - center[0], point[1] - center[1], point[2] - center[2]));
        let geometry;
        try {
          geometry = new ConvexGeometry(shellPoints);
          if (!(geometry.attributes.position?.count >= 4)) throw new Error('degenerate');
        } catch {
          geometry = new THREE.SphereGeometry(PAD, 16, 12);   // never drop a category
        }
        hulls.push({ slot, color, geometry, center });
      } else {
        const shellPoints = memberPoints.flatMap((point) => shell2D(point));
        const outline = convexHull2D(shellPoints).map(([x, y]) => [x - center[0], y - center[1]]);
        const shape = new THREE.Shape(outline.map(([x, y]) => new THREE.Vector2(x, y)));
        hulls.push({
          slot, color, center,
          geometry: new THREE.ShapeGeometry(shape),
          outline,
        });
      }
    }
    return { hulls, constellations };
  }, [clusterOfPoint, positions, mode, showHulls, pointCount]);

  // Dispose replaced geometries — the slider rebuilds hulls on every step, and
  // orphaned BufferGeometry leaks GPU memory. Each rebuild creates all-new
  // geometries, so the outgoing set can be freed whole; the same cleanup also
  // frees the final set on unmount (the old version skipped that).
  useEffect(() => {
    return () => {
      for (const hull of hulls) hull.geometry.dispose();
    };
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

      {hulls.map((hull) => (
        <group key={`${mode}-${hull.slot}`} position={hull.center}>
          <mesh geometry={hull.geometry} position={mode === '2d' ? [0, 0, -0.02] : undefined}>
            <meshBasicMaterial
              color={hull.color}
              transparent
              opacity={mode === '3d' ? 0.1 : 0.14}
              depthWrite={false}
              side={THREE.DoubleSide}
            />
          </mesh>
          {hull.outline && (
            <lineLoop position={[0, 0, -0.01]}>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  args={[new Float32Array(hull.outline.flatMap(([x, y]) => [x, y, 0])), 3]}
                />
              </bufferGeometry>
              <lineBasicMaterial color={hull.color} transparent opacity={0.5} />
            </lineLoop>
          )}
        </group>
      ))}

      {constellations.map((constellation) => (
        <lineSegments key={`c-${mode}-${constellation.slot}`} renderOrder={1}>
          <bufferGeometry key={`cg-${mode}-${constellation.slot}`}>
            <bufferAttribute attach="attributes-position" args={[constellation.lineVertices, 3]} />
          </bufferGeometry>
          <lineBasicMaterial
            color={constellation.color}
            transparent
            opacity={0.55}
            depthWrite={false}
          />
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
        onPointerMove={(event) => {
          event.stopPropagation();
          if (event.index !== undefined && event.index !== hoveredIndex) {
            onHover(event.index, event.nativeEvent.offsetX, event.nativeEvent.offsetY);
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

      <HoverRing positions={positions} ringTex={ringTex} mode={mode} hoveredIndex={hoveredIndex} />
    </>
  );
}

// Camera-facing ring on the hovered doc — the "surface ring on overlapping
// marks" treatment, adapted to a point cloud.
function HoverRing({ positions, ringTex, mode, hoveredIndex }) {
  if (hoveredIndex < 0 || hoveredIndex * 3 >= positions.length) return null;
  const hoveredPosition = new Float32Array([
    positions[hoveredIndex * 3], positions[hoveredIndex * 3 + 1], positions[hoveredIndex * 3 + 2],
  ]);
  return (
    <points renderOrder={4}>
      <bufferGeometry key={hoveredIndex}>
        <bufferAttribute attach="attributes-position" args={[hoveredPosition, 3]} />
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

export default function EmbeddingSpace({ collectionId, corpusVersion, controlsEl, active }) {
  const [embeddingMap, setEmbeddingMap] = useState(null);
  const [error, setError] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const [mode, setMode] = useState('3d');
  const [showHulls, setShowHulls] = useState(true);
  const [tooltip, setTooltip] = useState(null);

  // Refetches when the collection changes AND when a pipeline run finishes
  // (corpusVersion bump) — previously the map only loaded once per mount.
  useEffect(() => {
    if (!collectionId) return;
    setError(null);
    getEmbeddingMap(collectionId)
      .then((loadedMap) => {
        setEmbeddingMap(loadedMap);
        setThreshold(loadedMap.defaultThreshold ?? 0.5);
      })
      .catch((err) => setError(err.message));
  }, [collectionId, corpusVersion]);

  if (!collectionId) {
    return (
      <div className="viz-empty">
        <h2>No collection selected</h2>
        <p>Select a collection in the Documents tab to see its embedding space.</p>
      </div>
    );
  }

  const { clusterOfPoint, clusters } = useMemo(() => {
    if (!embeddingMap) return { clusterOfPoint: new Int32Array(0), clusters: [] };
    return clusterize(embeddingMap.points.length, embeddingMap.edges, threshold);
  }, [embeddingMap, threshold]);

  const onHover = (index, x, y) =>
    setTooltip(index >= 0 ? { index, x, y } : null);

  if (error) {
    return (
      <div className="viz-empty">
        <h2>No embedding map yet</h2>
        <p>{error}</p>
        <p>Run the pipeline through the categorize stage.</p>
      </div>
    );
  }
  if (!embeddingMap) return <div className="viz-empty"><p>Loading embedding space…</p></div>;

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
          onChange={(event) => setThreshold(parseFloat(event.target.value))}
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
        <input
          type="checkbox"
          checked={showHulls}
          onChange={(event) => setShowHulls(event.target.checked)}
        />
        Category blobs
      </label>

      <div>
        <span className="control-label">Categories · {embeddingMap.points.length} docs</span>
        <div className="legend">
          {clusters.slice(0, SLOTS).map((cluster) => (
            <div className="legend-row" key={cluster.index}>
              <span
                className="legend-swatch"
                style={{ background: seriesColor(cluster.index), color: seriesColor(cluster.index) }}
              />
              <span className="legend-title" title={embeddingMap.points[cluster.members[0]].title}>
                {embeddingMap.points[cluster.members[0]].title}
              </span>
              <span className="legend-count">{cluster.members.length}</span>
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
                {clusters.slice(SLOTS).reduce(
                  (total, cluster) => total + cluster.members.length, 0)}
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
          embeddingMap={embeddingMap}
          mode={mode}
          clusterOfPoint={clusterOfPoint}
          showHulls={showHulls}
          onHover={onHover}
          hoveredIndex={tooltip ? tooltip.index : -1}
        />
      </Canvas>

      {tooltip && (
        <div className="viz-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
          <div>{embeddingMap.points[tooltip.index].title}</div>
          <div className="sub">{embeddingMap.points[tooltip.index].filename}</div>
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
