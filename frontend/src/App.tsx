import { useState, useEffect, useCallback } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import ChatPanel from './components/ChatPanel';

export interface DatasetInfo {
  filename: string;
  row_count: number;
  column_count: number;
  status: 'success' | 'error';
  errorMessage?: string;
}

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [pendingQuery, setPendingQuery] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/sessions', { method: 'POST' })
      .then((res) => {
        if (!res.ok) throw new Error('Server responded with status ' + res.status);
        return res.json() as Promise<{ session_id: string }>;
      })
      .then((data) => {
        setSessionId(data.session_id);
        setLoading(false);
      })
      .catch(() => {
        setError('Failed to start session. Please refresh the page.');
        setLoading(false);
      });
  }, []);

  const handleDatasetsUpdate = useCallback((updated: DatasetInfo[]) => {
    setDatasets((prev) => {
      const map = new Map(prev.map((d) => [d.filename, d]));
      updated.forEach((d) => map.set(d.filename, d));
      return Array.from(map.values());
    });
  }, []);

  const handleQuickAction = useCallback((query: string) => {
    setPendingQuery(query);
  }, []);

  const consumePendingQuery = useCallback((): string | null => {
    const q = pendingQuery;
    setPendingQuery(null);
    return q;
  }, [pendingQuery]);

  if (loading) {
    return (
      <div className="screen-loading">
        <div className="screen-loading__spinner" />
        <p className="screen-loading__text">Starting session...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="screen-error">
        <p className="screen-error__message">{error}</p>
      </div>
    );
  }

  const datasetCount = datasets.filter((d) => d.status === 'success').length;

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__brand">
          <div className="app-header__icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
          </div>
          <div>
            <div className="app-header__title">AI Data Analyst</div>
            <div className="app-header__subtitle">Explore your data with natural language</div>
          </div>
        </div>
        <div className="app-header__meta">
          <span className="app-header__dataset-count">
            {datasetCount === 0 ? 'No datasets' : datasetCount === 1 ? '1 dataset' : datasetCount + ' datasets'}
          </span>
          <div className="app-header__status">
            <span className="app-header__status-dot" />
            Ready
          </div>
        </div>
      </header>

      <aside className="app-sidebar">
        <Sidebar
          sessionId={sessionId!}
          datasets={datasets}
          onDatasetsUpdate={handleDatasetsUpdate}
          onQuickAction={handleQuickAction}
        />
      </aside>

      <main className="app-main">
        <ChatPanel
          sessionId={sessionId!}
          pendingQuery={pendingQuery}
          consumePendingQuery={consumePendingQuery}
        />
      </main>
    </div>
  );
}

export default App;
