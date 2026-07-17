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
  const [saveState, setSaveState] = useState('idle'); // idle | saving | ok | err
  const [saveMessage, setSaveMessage] = useState('');

  useEffect(() => { setDraft(value || ''); }, [value]);

  const dirty = draft.trim() !== (value || '');

  const apply = async () => {
    setSaveState('saving');
    try {
      const saved = await saveSettings({ [role]: draft.trim() });
      onSaved(saved.roles);
      setSaveState('ok');
      setSaveMessage('Saved to .env');
    } catch (err) {
      setSaveState('err');
      setSaveMessage(err.message);
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
          onChange={(event) => { setDraft(event.target.value); setSaveState('idle'); }}
          aria-label={`${ROLE_LABELS[role] || role} model`}
        />
        <button className="btn" onClick={apply} disabled={!dirty || !draft.trim() || saveState === 'saving'}>
          {saveState === 'saving' ? 'Saving…' : 'Apply'}
        </button>
      </div>
      {saveState === 'ok' && !dirty && <p className="save-note ok">✓ {saveMessage}</p>}
      {saveState === 'err' && <p className="save-note err">{saveMessage}</p>}
      {installed.length > 0 && value && !installed.includes(value) && (
        <p className="save-note" style={{ color: 'var(--ink-muted)' }}>
          Current model isn’t in Ollama’s installed list.
        </p>
      )}
    </div>
  );
}

export default function ModelsPanel() {
  const [modelsInfo, setModelsInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getModels().then(setModelsInfo).catch((err) => setError(err.message));
  }, []);

  if (error) return <div className="viz-empty"><p>{error}</p></div>;
  if (!modelsInfo) return <div className="viz-empty"><p>Loading models…</p></div>;

  return (
    <div className="models-wrap">
      <div className="models-inner">
        <h2 className="page-title">Models</h2>
        <p className="page-sub">
          Which Ollama model each pipeline role uses. Changes persist to .env and
          apply to the next pipeline run.
        </p>

        {!modelsInfo.ollamaUp && (
          <div className="banner">
            Ollama is unreachable at {modelsInfo.ollamaUrl} — the installed-model list is
            unavailable, but you can still type a model name and apply it.
          </div>
        )}

        <datalist id="installed-models">
          {modelsInfo.installed.map((modelName) => <option key={modelName} value={modelName} />)}
        </datalist>

        {Object.entries(modelsInfo.descriptions).map(([role, description]) => (
          <RoleCard
            key={role}
            role={role}
            description={description}
            value={modelsInfo.roles[role]}
            installed={modelsInfo.installed}
            onSaved={(roles) => setModelsInfo((prev) => ({ ...prev, roles }))}
          />
        ))}
      </div>
    </div>
  );
}
