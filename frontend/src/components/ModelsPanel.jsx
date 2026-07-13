import { useEffect, useState } from 'react';
import { getModels, saveSettings } from '../api.js';

const ROLE_LABELS = {
  METADATA_MODEL:         'Metadata extraction',
  EXTRACTION_MODEL:       'Content extraction',
  QUERY_CLASSIFIER_MODEL: 'Query classifier',
  KG_MODEL:               'Knowledge-graph builder',
  REASONING_MODEL:        'Reasoning',
};

function RoleCard({ role, description, value, installed, onSaved }) {
  const [draft, setDraft] = useState(value || '');
  const [state, setState] = useState('idle'); // idle | saving | ok | err
  const [message, setMessage] = useState('');

  useEffect(() => { setDraft(value || ''); }, [value]);

  const dirty = draft.trim() !== (value || '');

  const apply = async () => {
    setState('saving');
    try {
      const resp = await saveSettings({ [role]: draft.trim() });
      onSaved(resp.roles);
      setState('ok');
      setMessage('Saved to .env');
    } catch (e) {
      setState('err');
      setMessage(e.message);
    }
  };

  return (
    <div className="model-card">
      <h3>{ROLE_LABELS[role] || role}</h3>
      <p className="desc">{description}</p>
      <div className="model-row">
        <input
          type="text"
          list="installed-models"
          value={draft}
          placeholder="model name, e.g. phi4"
          onChange={(e) => { setDraft(e.target.value); setState('idle'); }}
          aria-label={`${ROLE_LABELS[role] || role} model`}
        />
        <button className="btn" onClick={apply} disabled={!dirty || !draft.trim() || state === 'saving'}>
          {state === 'saving' ? 'Saving…' : 'Apply'}
        </button>
      </div>
      {state === 'ok' && !dirty && <p className="save-note ok">✓ {message}</p>}
      {state === 'err' && <p className="save-note err">{message}</p>}
      {installed.length > 0 && value && !installed.includes(value) && (
        <p className="save-note" style={{ color: 'var(--ink-muted)' }}>
          Current model isn’t in Ollama’s installed list.
        </p>
      )}
    </div>
  );
}

export default function ModelsPanel() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getModels().then(setInfo).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="viz-empty"><p>{error}</p></div>;
  if (!info) return <div className="viz-empty"><p>Loading models…</p></div>;

  return (
    <div className="models-wrap">
      <div className="models-inner">
        <h2 className="page-title">Models</h2>
        <p className="page-sub">
          Which Ollama model each pipeline role uses. Changes persist to .env and
          apply to the next pipeline run.
        </p>

        {!info.ollamaUp && (
          <div className="banner">
            Ollama is unreachable at {info.ollamaUrl} — the installed-model list is
            unavailable, but you can still type a model name and apply it.
          </div>
        )}

        <datalist id="installed-models">
          {info.installed.map((m) => <option key={m} value={m} />)}
        </datalist>

        {Object.entries(info.descriptions).map(([role, description]) => (
          <RoleCard
            key={role}
            role={role}
            description={description}
            value={info.roles[role]}
            installed={info.installed}
            onSaved={(roles) => setInfo((prev) => ({ ...prev, roles }))}
          />
        ))}
      </div>
    </div>
  );
}
