/**
 * PipelineOverlay — full-app scrim shown while a pipeline run is in flight.
 *
 * The run is one long (minutes) request with no per-stage streaming, so this
 * blocks interaction rather than showing a progress bar: navigating away
 * unmounts the Documents tab and orphans the request, and the graph stage is
 * slow. The scrim intercepts every click, so the user can't tab away mid-run.
 */
export default function PipelineOverlay({ collectionName }) {
  return (
    <div className="pipeline-overlay" role="alertdialog" aria-live="assertive"
         aria-label="Pipeline running">
      <div className="pipeline-overlay-card">
        <div className="pipeline-spinner" aria-hidden="true" />
        <h2>Running pipeline…</h2>
        <p>
          Indexing{collectionName ? <> <strong>{collectionName}</strong></> : ' this collection'}:
          extract → embed → categorize → rank → knowledge graph.
        </p>
        <p className="pipeline-overlay-note">
          This can take several minutes — the knowledge-graph stage runs a local
          model. Keep this tab open; navigation is paused until it finishes.
        </p>
      </div>
    </div>
  );
}
