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
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      setSize((s) => (s.w === w && s.h === h ? s : { w, h }));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    window.addEventListener('resize', measure);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, [ref, active]);

  return size;
}

export default function KnowledgeGraph({ controlsEl, active }) {
  const [raw, setRaw] = useState(null);
  const [error, setError] = useState(null);
  const [showSections, setShowSections] = useState(true);
  const [showCites, setShowCites] = useState(true);
  const wrapRef = useRef(null);
  const { w, h } = useSize(wrapRef, active);

  useEffect(() => {
    getGraph().then(setRaw).catch((e) => setError(e.message));
  }, []);

  // graph.json → force-graph shape, defensively: edges referencing nodes that
  // were never materialized (known build_graph.js stub gap, pending the
  // LightRAG migration) would crash the layout, so they're dropped + counted.
  const { data, stats } = useMemo(() => {
    if (!raw) return { data: null, stats: null };
    const keepNode = (n) => showSections || n.type !== 'section';
    const nodes = raw.nodes.filter(keepNode).map((n) => ({
      id: n.id,
      name: n.type === 'document' ? (n.label || n.filename) : `${n.label} — ${n.preview?.slice(0, 80) ?? ''}`,
      type: n.type,
      created: n.created,
    }));
    const ids = new Set(nodes.map((n) => n.id));
    let dangling = 0;
    const links = raw.edges
      .filter((e) => {
        if (e.type === 'cites' && !showCites) return false;
        if (e.type === 'has_section' && !showSections) return false;
        const ok = ids.has(e.source) && ids.has(e.target);
        if (!ok && (showSections || e.type === 'cites')) dangling++;
        return ok;
      })
      .map((e) => ({ source: e.source, target: e.target, type: e.type }));
    return {
      data: { nodes, links },
      stats: {
        docs: raw.nodes.filter((n) => n.type === 'document').length,
        sections: raw.nodes.filter((n) => n.type === 'section').length,
        cites: raw.edges.filter((e) => e.type === 'cites').length,
        dangling,
      },
    };
  }, [raw, showSections, showCites]);

  // Night-sky canvas colors (a canvas can't resolve CSS vars itself).
  // Documents are the star-blue accent; sections recede to slate; citation
  // edges are the bright channel, membership edges the faint one.
  const palette = { doc: '#58b0e8', section: '#4a5a75', cite: '#93a4c0', member: '#1f2940' };

  if (error) {
    return (
      <div className="viz-empty">
        <h2>No knowledge graph yet</h2>
        <p>{error}</p>
        <p>Run the pipeline through the build-graph stage, then reload.</p>
      </div>
    );
  }
  if (!data) return <div className="viz-empty"><p>Loading knowledge graph…</p></div>;

  const controls = (
    <>
      <div>
        <span className="control-label">Layers</span>
        <label className="check-row">
          <input
            type="checkbox"
            checked={showSections}
            onChange={(e) => setShowSections(e.target.checked)}
          />
          Section nodes
        </label>
        <label className="check-row" style={{ marginTop: 6 }}>
          <input
            type="checkbox"
            checked={showCites}
            onChange={(e) => setShowCites(e.target.checked)}
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
      {w > 0 && (
        <ForceGraph2D
          width={w}
          height={h}
          graphData={data}
          backgroundColor="rgba(0,0,0,0)"
          nodeVal={(n) => (n.type === 'document' ? 7 : 1.6)}
          nodeColor={(n) => (n.type === 'document' ? palette.doc : palette.section)}
          nodeLabel={(n) => n.name}
          linkColor={(l) => (l.type === 'cites' ? palette.cite : palette.member)}
          linkWidth={(l) => (l.type === 'cites' ? 1.6 : 0.5)}
          linkDirectionalArrowLength={(l) => (l.type === 'cites' ? 5 : 0)}
          linkDirectionalArrowRelPos={0.95}
          cooldownTicks={200}
        />
      )}
      <div className="viz-hint">drag nodes · scroll to zoom · hover for labels</div>
      {active && controlsEl && createPortal(controls, controlsEl)}
    </div>
  );
}
