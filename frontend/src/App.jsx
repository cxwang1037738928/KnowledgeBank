import { useState } from 'react';
import Chat from './components/Chat.jsx';
import DocumentViewer from './components/DocumentViewer.jsx';
import EmbeddingSpace from './components/EmbeddingSpace.jsx';
import KnowledgeGraph from './components/KnowledgeGraph.jsx';
import ModelsPanel from './components/ModelsPanel.jsx';
import { useCrawler, CRAWLERS } from './lib/theme.js';

const GEM = (color, size = 12) => (
  <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true">
    <path d="M8 1.5 14.5 8 8 14.5 1.5 8Z" fill={color} opacity="0.9" />
    <path d="M8 1.5 11.2 8 8 14.5 4.8 8Z" fill="#fff" opacity="0.22" />
  </svg>
);

const ICONS = {
  chat: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" aria-hidden="true">
      <path d="M2.5 3.5h11v7h-6l-3 3v-3h-2z" />
    </svg>
  ),
  docs: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" aria-hidden="true">
      <path d="M4 1.5h6l2.5 2.5v10.5h-8.5z M10 1.5v2.5h2.5" />
      <path d="M6 8h4 M6 10.5h4" strokeLinecap="round" />
    </svg>
  ),
  space: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <circle cx="4" cy="5" r="2" /><circle cx="11.5" cy="3.5" r="1.6" />
      <circle cx="12.5" cy="10" r="2.2" /><circle cx="5" cy="12" r="1.6" />
    </svg>
  ),
  graph: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
      <circle cx="3.5" cy="8" r="2" /><circle cx="12.5" cy="3.5" r="2" /><circle cx="12.5" cy="12.5" r="2" />
      <path d="M5.3 7.2 10.7 4.3 M5.3 8.8 10.7 11.7" />
    </svg>
  ),
  models: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" aria-hidden="true">
      <path d="M2 5h12 M2 11h12" />
      <circle cx="6" cy="5" r="1.8" fill="var(--surface)" />
      <circle cx="10.5" cy="11" r="1.8" fill="var(--surface)" />
    </svg>
  ),
};

const TABS = [
  { id: 'chat',   label: 'Chat' },
  { id: 'docs',   label: 'Documents' },
  { id: 'space',  label: 'Embedding Space' },
  { id: 'graph',  label: 'Knowledge Graph' },
  { id: 'models', label: 'Models' },
];

function ComingSoon({ crawler }) {
  const crawlerInfo = CRAWLERS[crawler];
  return (
    <div className="viz-empty">
      {GEM(crawlerInfo.accent, 44)}
      <h2>{crawlerInfo.name}</h2>
      <p>The {crawlerInfo.tagline} crawler is on the roadmap — its pipeline hasn't landed yet.</p>
      <p>Switch back to Sapphire to explore the academic corpus.</p>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState('chat');
  const [controlsEl, setControlsEl] = useState(null);
  const [docTarget, setDocTarget] = useState(null);   // { docId, chunkId, nonce }
  const { crawler, setCrawler } = useCrawler();
  const live = crawler === 'sapphire';

  // A [n] citation (or source chip) was clicked in Chat: jump to the cited
  // chunk in the Documents tab. query = {text, embedding} of the question
  // that produced it — the viewer highlights only the sentences that score
  // above threshold against it.
  const openCitation = (source, query) => {
    setDocTarget({
      docId: source.docId,
      chunkId: source.chunkId,
      query: query || null,
      nonce: Date.now(),
    });
    setTab('docs');
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1 className="wordmark">OpenCrawl</h1>
          <div className="wordmark-sub">{CRAWLERS[crawler].name.toLowerCase()} · {CRAWLERS[crawler].tagline}</div>
        </div>

        <nav className="tab-rail" aria-label="Views">
          {TABS.map((tabDef) => (
            <button
              key={tabDef.id}
              className={`tab-btn ${tab === tabDef.id ? 'active' : ''}`}
              onClick={() => setTab(tabDef.id)}
            >
              {ICONS[tabDef.id]}
              {tabDef.label}
            </button>
          ))}
        </nav>

        <hr className="sidebar-divider" />

        {/* Active tab portals its contextual controls here. */}
        <div className="sidebar-controls" ref={setControlsEl} />

        <div className="sidebar-footer">
          <span>local corpus</span>
        </div>
      </aside>

      <main className="main">
        {/* Crawler switcher — the top-right jewel box. */}
        <div className="crawler-switch" role="group" aria-label="Crawler">
          {Object.entries(CRAWLERS).map(([crawlerId, crawlerInfo]) => (
            <button
              key={crawlerId}
              className={`crawler-btn ${crawler === crawlerId ? 'active' : ''}`}
              onClick={() => setCrawler(crawlerId)}
              title={`${crawlerInfo.name} — ${crawlerInfo.tagline}${crawlerInfo.ready ? '' : ' (coming soon)'}`}
            >
              {GEM(crawlerInfo.accent)}
              <span>{crawlerInfo.name}</span>
            </button>
          ))}
        </div>

        {!live ? (
          <ComingSoon crawler={crawler} />
        ) : (
          <>
            {/* Tabs stay mounted so camera position / graph layout / chat
                history survive tab switches; hidden ones are display:none. */}
            <div className="viz-fill" style={{ display: tab === 'chat' ? 'block' : 'none' }}>
              <Chat active={tab === 'chat'} onCitation={openCitation} />
            </div>
            <div className="viz-fill" style={{ display: tab === 'docs' ? 'block' : 'none' }}>
              {controlsEl && <DocumentViewer controlsEl={controlsEl} active={tab === 'docs'} target={docTarget} />}
            </div>
            <div className="viz-fill" style={{ display: tab === 'space' ? 'block' : 'none' }}>
              {controlsEl && <EmbeddingSpace controlsEl={controlsEl} active={tab === 'space'} />}
            </div>
            <div className="viz-fill" style={{ display: tab === 'graph' ? 'block' : 'none' }}>
              {controlsEl && <KnowledgeGraph controlsEl={controlsEl} active={tab === 'graph'} />}
            </div>
            <div className="viz-fill" style={{ display: tab === 'models' ? 'block' : 'none' }}>
              <ModelsPanel active={tab === 'models'} />
            </div>
          </>
        )}
      </main>
    </div>
  );
}
