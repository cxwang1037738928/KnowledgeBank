import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import ForceGraph2D from 'react-force-graph-2d';
import { getGraph } from '../api.js';

/**
 * Element size, re-measured whenever the tab becomes active.
 *
 * The tab is mounted inside a display:none container (so camera/layout state
 * survives switches), which means it measures 0x0 at mount — and a
 * ResizeObserver alone does NOT reliably deliver a size when an ancestor flips
 * back to display:block. Without a non-zero size ForceGraph2D renders nothing
 * at all. Re-running the effect on `active` (in useLayoutEffect, i.e. after the
 * DOM shows the container but before paint) measures the real box directly.
 */
function useSize(ref, active) {
  const [size, setSize] = useState({ w: 0, h: 0 });

  useLayoutEffect(() => {
    const wrapEl = ref.current;
    if (!wrapEl) return;
    const measure = () => {
      const width = wrapEl.clientWidth;
      const height = wrapEl.clientHeight;
      setSize((prevSize) =>
        (prevSize.w === width && prevSize.h === height ? prevSize : { w: width, h: height }));
    };
    measure();
    const resizeObserver = new ResizeObserver(measure);
    resizeObserver.observe(wrapEl);
    window.addEventListener('resize', measure);
    return () => {
      resizeObserver.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, [ref, active]);

  return size;
}

export default function KnowledgeGraph({ collectionId, corpusVersion, controlsEl, active }) {
  const [graphJson, setGraphJson] = useState(null);
  const [error, setError] = useState(null);
  const [showSections, setShowSections] = useState(true);
  const [showCites, setShowCites] = useState(true);
  const wrapRef = useRef(null);
  const { w: width, h: height } = useSize(wrapRef, active);

  // Refetches when the collection changes AND when a pipeline run finishes
  // (corpusVersion bump) — previously the graph only loaded once per mount.
  useEffect(() => {
    if (!collectionId) return;
    setError(null);
    getGraph(collectionId).then(setGraphJson).catch((err) => setError(err.message));
  }, [collectionId, corpusVersion]);

  // graph.json → force-graph shape, defensively: edges referencing nodes that
  // were never materialized (known build_graph.js stub gap, pending the
  // LightRAG migration) would crash the layout, so they're dropped + counted.
  const { graphData, stats } = useMemo(() => {
    if (!graphJson) return { graphData: null, stats: null };
    const keepNode = (graphNode) => showSections || graphNode.type !== 'section';
    const nodes = graphJson.nodes.filter(keepNode).map((graphNode) => ({
      id: graphNode.id,
      name: graphNode.type === 'document'
        ? (graphNode.label || graphNode.filename)
        : `${graphNode.label} — ${graphNode.preview?.slice(0, 80) ?? ''}`,
      type: graphNode.type,
      created: graphNode.created,
    }));
    const nodeIds = new Set(nodes.map((node) => node.id));
    let danglingCount = 0;
    const links = graphJson.edges
      .filter((graphEdge) => {
        if (graphEdge.type === 'cites' && !showCites) return false;
        if (graphEdge.type === 'has_section' && !showSections) return false;
        const endpointsPresent = nodeIds.has(graphEdge.source) && nodeIds.has(graphEdge.target);
        if (!endpointsPresent && (showSections || graphEdge.type === 'cites')) danglingCount++;
        return endpointsPresent;
      })
      .map((graphEdge) => ({
        source: graphEdge.source,
        target: graphEdge.target,
        type: graphEdge.type,
      }));
    return {
      graphData: { nodes, links },
      stats: {
        docs: graphJson.nodes.filter((graphNode) => graphNode.type === 'document').length,
        sections: graphJson.nodes.filter((graphNode) => graphNode.type === 'section').length,
        cites: graphJson.edges.filter((graphEdge) => graphEdge.type === 'cites').length,
        dangling: danglingCount,
      },
    };
  }, [graphJson, showSections, showCites]);

  // Night-sky canvas colors (a canvas can't resolve CSS vars itself).
  // Documents are the star-blue accent; sections recede to slate; citation
  // edges are the bright channel, membership edges the faint one.
  const palette = { doc: '#58b0e8', section: '#4a5a75', cite: '#93a4c0', member: '#1f2940' };

  if (!collectionId) {
    return (
      <div className="viz-empty">
        <h2>No collection selected</h2>
        <p>Select a collection in the Documents tab to see its knowledge graph.</p>
      </div>
    );
  }
  if (error) {
    return (
      <div className="viz-empty">
        <h2>No knowledge graph yet</h2>
        <p>{error}</p>
        <p>Run the pipeline through the build-graph stage.</p>
      </div>
    );
  }
  if (!graphData) return <div className="viz-empty"><p>Loading knowledge graph…</p></div>;

  const controls = (
    <>
      <div>
        <span className="control-label">Layers</span>
        <label className="check-row">
          <input
            type="checkbox"
            checked={showSections}
            onChange={(event) => setShowSections(event.target.checked)}
          />
          Section nodes
        </label>
        <label className="check-row" style={{ marginTop: 6 }}>
          <input
            type="checkbox"
            checked={showCites}
            onChange={(event) => setShowCites(event.target.checked)}
          />
          Citation edges
        </label>
      </div>

      <div>
        <span className="control-label">Legend</span>
        <div className="legend">
          <div className="legend-row">
            <span className="legend-swatch" style={{ background: palette.doc }} />
            <span className="legend-title">Document ({stats.docs})</span>
          </div>
          <div className="legend-row">
            <span className="legend-swatch" style={{ background: palette.section }} />
            <span className="legend-title">Section ({stats.sections})</span>
          </div>
          <div className="legend-row">
            <span className="legend-swatch" style={{ background: palette.cite, borderRadius: 2, height: 3 }} />
            <span className="legend-title">cites ({stats.cites})</span>
          </div>
        </div>
        {stats.dangling > 0 && (
          <p style={{ color: 'var(--ink-muted)', fontSize: 12, marginTop: 10 }}>
            {stats.dangling} edge{stats.dangling !== 1 ? 's' : ''} skipped (endpoint not in graph)
          </p>
        )}
      </div>
    </>
  );

  return (
    <div ref={wrapRef} className="viz-fill">
      {width > 0 && (
        <ForceGraph2D
          width={width}
          height={height}
          graphData={graphData}
          backgroundColor="rgba(0,0,0,0)"
          nodeVal={(node) => (node.type === 'document' ? 7 : 1.6)}
          nodeColor={(node) => (node.type === 'document' ? palette.doc : palette.section)}
          nodeLabel={(node) => node.name}
          linkColor={(link) => (link.type === 'cites' ? palette.cite : palette.member)}
          linkWidth={(link) => (link.type === 'cites' ? 1.6 : 0.5)}
          linkDirectionalArrowLength={(link) => (link.type === 'cites' ? 5 : 0)}
          linkDirectionalArrowRelPos={0.95}
          cooldownTicks={200}
        />
      )}
      <div className="viz-hint">drag nodes · scroll to zoom · hover for labels</div>
      {active && controlsEl && createPortal(controls, controlsEl)}
    </div>
  );
}
