import { useRef, useState, DragEvent, ChangeEvent } from 'react';

// Matches the backend FileUploadResponse model
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

interface FileResult {
  filename: string;
  status: 'success' | 'error';
  row_count?: number;
  errorMessage?: string;
}

interface Props {
  sessionId: string;
}

function FileUpload({ sessionId }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState<FileResult[]>([]);
  const [clientError, setClientError] = useState<string | null>(null);

  /** Validate that every file ends with .csv; return rejected names. */
  function validateFiles(files: File[]): { accepted: File[]; rejected: string[] } {
    const accepted: File[] = [];
    const rejected: string[] = [];
    for (const file of files) {
      if (file.name.toLowerCase().endsWith('.csv')) {
        accepted.push(file);
      } else {
        rejected.push(file.name);
      }
    }
    return { accepted, rejected };
  }

  async function uploadFiles(files: File[]) {
    const { accepted, rejected } = validateFiles(files);

    // Show client-side rejection immediately
    if (rejected.length > 0) {
      setClientError(
        `The following file(s) are not CSV and were rejected: ${rejected.join(', ')}`
      );
    } else {
      setClientError(null);
    }

    if (accepted.length === 0) return;

    const formData = new FormData();
    for (const file of accepted) {
      formData.append('files', file);
    }

    setUploading(true);
    setResults([]);

    try {
      const response = await fetch(`/api/sessions/${sessionId}/files`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Upload failed (HTTP ${response.status}): ${text}`);
      }

      const data: FileUploadResponse[] = await response.json();

      const parsed: FileResult[] = data.map((item) => {
        if (item.errors && item.errors.length > 0) {
          return {
            filename: item.filename,
            status: 'error',
            errorMessage: item.errors.join('; '),
          };
        }
        return {
          filename: item.filename,
          status: 'success',
          row_count: item.row_count,
        };
      });

      setResults(parsed);
    } catch (err) {
      setClientError(err instanceof Error ? err.message : 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
      // Reset the file input so the same file can be re-selected
      if (inputRef.current) {
        inputRef.current.value = '';
      }
    }
  }

  function handleDragOver(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(true);
  }

  function handleDragLeave(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      uploadFiles(files);
    }
  }

  function handleClick() {
    inputRef.current?.click();
  }

  function handleInputChange(e: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (files.length > 0) {
      uploadFiles(files);
    }
  }

  return (
    <div className="file-upload">
      {/* Drag-and-drop / click zone */}
      <div
        className={`drop-zone${dragOver ? ' drop-zone--active' : ''}`}
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        role="button"
        tabIndex={0}
        aria-label="Upload CSV files — drag and drop or click to browse"
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') handleClick();
        }}
      >
        <span className="drop-zone__icon">📂</span>
        <p className="drop-zone__text">
          Drag &amp; drop <strong>.csv</strong> files here, or{' '}
          <span className="drop-zone__link">click to browse</span>
        </p>
        <p className="drop-zone__hint">Multiple files supported</p>
      </div>

      {/* Hidden file input */}
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        multiple
        style={{ display: 'none' }}
        onChange={handleInputChange}
        aria-hidden="true"
      />

      {/* Loading indicator */}
      {uploading && (
        <p className="file-upload__status file-upload__status--loading">
          ⏳ Uploading...
        </p>
      )}

      {/* Client-side validation error */}
      {clientError && (
        <p className="file-upload__status file-upload__status--error" role="alert">
          ⚠️ {clientError}
        </p>
      )}

      {/* Per-file results */}
      {results.length > 0 && (
        <ul className="file-upload__results" aria-label="Upload results">
          {results.map((r) => (
            <li
              key={r.filename}
              className={`file-upload__result file-upload__result--${r.status}`}
            >
              {r.status === 'success' ? (
                <>✅ {r.filename} — {r.row_count} rows</>
              ) : (
                <>❌ {r.filename} — {r.errorMessage}</>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default FileUpload;
