import { useRef, useState, DragEvent, ChangeEvent } from 'react';
import type { DatasetInfo } from '../App';

interface ColumnSchema {
  name: string;
  dtype: string;
  null_count: number;
}

interface FileUploadResponse {
  filename: string;
  row_count: number;
  column_schemas: ColumnSchema[];
  errors: string[];
}

interface Props {
  sessionId: string;
  datasets: DatasetInfo[];
  onDatasetsUpdate: (datasets: DatasetInfo[]) => void;
  onQuickAction: (query: string) => void;
}

const QUICK_ACTIONS = [
  {
    label: 'Dataset Profile',
    query: 'Give me a full profile of the dataset.',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
        <rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
      </svg>
    ),
  },
  {
    label: 'Business Insights',
    query: 'What are the key business insights from this data?',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>
        <line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
    ),
  },
  {
    label: 'Trends',
    query: 'What trends do you see in this data?',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
    ),
  },
  {
    label: 'Detect Anomalies',
    query: 'Detect any anomalies in this dataset.',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
    ),
  },
];

function Sidebar({ sessionId, datasets, onDatasetsUpdate, onQuickAction }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [showDropzone, setShowDropzone] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [clientError, setClientError] = useState<string | null>(null);

  function validateFiles(files: File[]): { accepted: File[]; rejected: string[] } {
    const accepted: File[] = [];
    const rejected: string[] = [];
    for (const file of files) {
      if (file.name.toLowerCase().endsWith('.csv')) accepted.push(file);
      else rejected.push(file.name);
    }
    return { accepted, rejected };
  }

  async function uploadFiles(files: File[]) {
    const { accepted, rejected } = validateFiles(files);
    if (rejected.length > 0) {
      setClientError('Rejected (not CSV): ' + rejected.join(', '));
    } else {
      setClientError(null);
    }
    if (accepted.length === 0) return;

    const formData = new FormData();
    for (const file of accepted) formData.append('files', file);

    setUploading(true);
    try {
      const response = await fetch('/api/sessions/' + sessionId + '/files', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error('Upload failed (HTTP ' + response.status + '): ' + text);
      }
      const data: FileUploadResponse[] = await response.json();
      const parsed: DatasetInfo[] = data.map((item) => {
        if (item.errors && item.errors.length > 0) {
          return {
            filename: item.filename,
            row_count: 0,
            column_count: 0,
            status: 'error' as const,
            errorMessage: item.errors.join('; '),
          };
        }
        return {
          filename: item.filename,
          row_count: item.row_count,
          column_count: item.column_schemas.length,
          status: 'success' as const,
        };
      });
      onDatasetsUpdate(parsed);
      setShowDropzone(false);
    } catch (err) {
      setClientError(err instanceof Error ? err.message : 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = '';
    }
  }

  function handleDragOver(e: DragEvent<HTMLDivElement>) { e.preventDefault(); setDragOver(true); }
  function handleDragLeave(e: DragEvent<HTMLDivElement>) { e.preventDefault(); setDragOver(false); }
  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) uploadFiles(files);
  }
  function handleClick() { inputRef.current?.click(); }
  function handleInputChange(e: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (files.length > 0) uploadFiles(files);
  }

  const hasDatasets = datasets.some((d) => d.status === 'success');

  return (
    <>
      <section>
        <p className="sidebar-section__title">Datasets</p>
        {hasDatasets && (
          <div className="dataset-list">
            {datasets.map((d) => (
              <div key={d.filename} className="dataset-card">
                <div className="dataset-card__name" title={d.filename}>{d.filename}</div>
                <div className="dataset-card__meta">
                  {d.status === 'success' ? (
                    <>
                      <span className="dataset-card__badge">{d.row_count.toLocaleString()} rows</span>
                      <span className="dataset-card__badge">{d.column_count} cols</span>
                      <span className="dataset-card__status--success">Loaded</span>
                    </>
                  ) : (
                    <span className="dataset-card__status--error" title={d.errorMessage}>Error</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        <div style={{ marginTop: hasDatasets ? '10px' : '0' }}>
          {!showDropzone ? (
            <button
              className="sidebar-upload-btn"
              onClick={() => setShowDropzone(true)}
              disabled={uploading}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
              </svg>
              {hasDatasets ? 'Upload more CSVs' : 'Upload CSV'}
            </button>
          ) : (
            <div
              className={'upload-dropzone' + (dragOver ? ' upload-dropzone--active' : '')}
              onClick={handleClick}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              role="button"
              tabIndex={0}
              aria-label="Upload CSV files"
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleClick(); }}
            >
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
              <p className="upload-dropzone__text">
                Drop <strong>.csv</strong> files here, or{' '}
                <span className="upload-dropzone__link">browse</span>
              </p>
              <p className="upload-dropzone__hint">Multiple files supported</p>
            </div>
          )}
        </div>

        <input
          ref={inputRef} type="file" accept=".csv" multiple
          style={{ display: 'none' }}
          onChange={handleInputChange} aria-hidden="true"
        />

        {uploading && <p className="upload-status upload-status--loading">Uploading...</p>}
        {clientError && <p className="upload-status upload-status--error" role="alert">{clientError}</p>}
      </section>

      <section>
        <p className="sidebar-section__title">Quick Actions</p>
        <div className="quick-action-list">
          {QUICK_ACTIONS.map((action) => (
            <button
              key={action.label}
              className="quick-action-btn"
              disabled={!hasDatasets}
              onClick={() => onQuickAction(action.query)}
              title={hasDatasets ? action.query : 'Upload a dataset first'}
            >
              <span className="quick-action-btn__icon">{action.icon}</span>
              {action.label}
            </button>
          ))}
        </div>
        {!hasDatasets && (
          <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '10px', padding: '0 4px' }}>
            Upload a dataset to enable quick actions.
          </p>
        )}
      </section>
    </>
  );
}

export default Sidebar;
