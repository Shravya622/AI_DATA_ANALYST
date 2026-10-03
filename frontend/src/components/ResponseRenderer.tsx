import './ResponseRenderer.css';

export interface ReasoningTrace {
  tools_selected: string[];
  columns_used: string[];
  computation_description: string;
}

export interface AnomalyRecord {
  row_index: number;
  column: string;
  value: number;
  z_score: number | null;
  explanation: string;
}

export interface AnomalyReport {
  count: number;
  flagged_rows: AnomalyRecord[];
}

export interface QueryResponseData {
  answer: string;
  chart_base64: string | null;
  code_snippet: string | null;
  code_language: string | null;
  anomaly_report: AnomalyReport | null;
  reasoning_trace: ReasoningTrace;
  sampled: boolean;
}

interface Props {
  response: QueryResponseData;
}

function ResponseRenderer({ response }: Props) {
  const { answer, chart_base64, code_snippet, code_language, anomaly_report, reasoning_trace, sampled } = response;

  return (
    <div className="response-renderer">
      <p className="response-renderer__answer">{answer}</p>

      {chart_base64 && (
        <div className="response-renderer__chart">
          {sampled && (
            <p className="response-renderer__sample-note" role="note">
              Chart shows a representative sample of the data
            </p>
          )}
          <img src={'data:image/png;base64,' + chart_base64} alt="Generated chart" />
        </div>
      )}

      {code_snippet && (
        <div className="response-renderer__code-block">
          <span className="response-renderer__code-label">
            {code_language ? code_language.toUpperCase() : 'CODE'}
          </span>
          <pre className="response-renderer__pre"><code>{code_snippet}</code></pre>
        </div>
      )}

      {anomaly_report && anomaly_report.count > 0 && (
        <div className="response-renderer__anomaly">
          <h4 className="response-renderer__anomaly-title">
            {anomaly_report.count} {anomaly_report.count === 1 ? 'Anomaly' : 'Anomalies'} Detected
          </h4>
          <div className="response-renderer__table-wrapper">
            <table className="response-renderer__table" aria-label="Anomaly report">
              <thead>
                <tr>
                  <th scope="col">Row</th>
                  <th scope="col">Column</th>
                  <th scope="col">Value</th>
                  <th scope="col">Z-Score</th>
                  <th scope="col">Explanation</th>
                </tr>
              </thead>
              <tbody>
                {anomaly_report.flagged_rows.map((rec, idx) => (
                  <tr key={idx}>
                    <td>{rec.row_index}</td>
                    <td>{rec.column}</td>
                    <td>{rec.value}</td>
                    <td>{rec.z_score !== null ? rec.z_score.toFixed(2) : '—'}</td>
                    <td>{rec.explanation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <details className="response-renderer__trace">
        <summary className="response-renderer__trace-summary">
          How this answer was generated
        </summary>
        <div className="response-renderer__trace-body">
          <div className="response-renderer__trace-row">
            <span className="response-renderer__trace-label">Tools used</span>
            <span className="response-renderer__trace-value">
              {reasoning_trace.tools_selected.length > 0
                ? reasoning_trace.tools_selected.map((t) => (
                    <span key={t} className="response-renderer__trace-pill">{t}</span>
                  ))
                : <span>None</span>}
            </span>
          </div>
          {reasoning_trace.columns_used.length > 0 && (
            <div className="response-renderer__trace-row">
              <span className="response-renderer__trace-label">Columns</span>
              <span className="response-renderer__trace-value">{reasoning_trace.columns_used.join(', ')}</span>
            </div>
          )}
          <div className="response-renderer__trace-row">
            <span className="response-renderer__trace-label">Description</span>
            <span className="response-renderer__trace-value">{reasoning_trace.computation_description}</span>
          </div>
        </div>
      </details>
    </div>
  );
}

export default ResponseRenderer;
