import { useEffect, useState } from "react";

const API = "http://127.0.0.1:8000";

type MethodResult = {
  retrieval_metrics: Record<string, number>;
  latency: {
    mean_ms: number;
    median_ms: number;
    std_ms: number;
    min_ms: number;
    max_ms: number;
  };
  queries_per_second: number;
};

type Report = {
  evaluation: string;
  query_count: number;
  k: number;
  candidate_k: number;
  summary: Record<string, MethodResult>;
};

const labels: Record<string, string> = {
  dense: "Dense Retrieval",
  bm25: "BM25 Retrieval",
  hybrid: "Hybrid RRF",
  hybrid_reranked: "Hybrid + Reranking",
};

export default function EvaluationDashboard() {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [runMessage, setRunMessage] = useState("");

  async function loadReport() {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API}/evaluation/report`);

      if (!response.ok) {
        throw new Error("Evaluation report is not available yet.");
      }

      setReport(await response.json());
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not load report."
      );
    } finally {
      setLoading(false);
    }
  }

  async function runEvaluation() {
    setRunning(true);
    setRunMessage("");
    setError("");

    try {
      const response = await fetch(`${API}/evaluation/run`, {
        method: "POST",
      });

      const data = await response.json();

      if (!response.ok) {
        const detail = typeof data.detail === "string"
          ? data.detail
          : data.detail?.message ?? JSON.stringify(data.detail);
        throw new Error(detail || "Evaluation failed.");
      }

      setRunMessage(
        `Evaluation completed for ${data.query_count} benchmark questions.`
      );

      await loadReport();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Evaluation failed."
      );
    } finally {
      setRunning(false);
    }
  }

  useEffect(() => {
    loadReport();
  }, []);

  const methods = report
    ? Object.entries(report.summary)
    : [];

  const metric = (result: MethodResult, name: string) =>
    result.retrieval_metrics[name] ?? 0;

  const bestMethod = (name: string) => {
    if (!methods.length) return null;

    return [...methods].sort(
      (a, b) => metric(b[1], name) - metric(a[1], name)
    )[0];
  };

  return (
    <section className="evaluation-dashboard">
      <div className="evaluation-heading">
        <div>
          <span className="evaluation-eyebrow">RESEARCH ANALYTICS</span>
          <h2>Evaluation Dashboard</h2>
          <p>Measure retrieval quality and system performance.</p>
        </div>

        <div className="evaluation-actions">
          <button onClick={loadReport} disabled={loading || running}>
            Refresh results
          </button>
          <button
            className="evaluation-run-button"
            onClick={runEvaluation}
            disabled={running || loading}
          >
            {running ? "Running experiment..." : "Run evaluation"}
          </button>
          <button
            onClick={() => window.open(`${API}/evaluation/export/csv`, "_blank")}
            disabled={!report}
          >
            Export CSV
          </button>
          <button
            onClick={() => window.open(`${API}/evaluation/export/json`, "_blank")}
            disabled={!report}
          >
            Export JSON
          </button>
        </div>
      </div>

      {loading && <p>Loading evaluation results...</p>}
      {runMessage && <p className="evaluation-success">{runMessage}</p>}

      {!loading && error && (
        <div className="evaluation-notice">
          <strong>No report available</strong>
          <p>{error}</p>
          <p>
            Run the TrustRAG evaluation experiment after annotating
            your benchmark questions.
          </p>
        </div>
      )}

      {report && (
        <>
          <div className="evaluation-stats">
            <article>
              <span>Benchmark questions</span>
              <strong>{report.query_count}</strong>
            </article>
            <article>
              <span>Top K</span>
              <strong>{report.k}</strong>
            </article>
            <article>
              <span>Candidate pool</span>
              <strong>{report.candidate_k}</strong>
            </article>
            <article>
              <span>Methods compared</span>
              <strong>{methods.length}</strong>
            </article>
          </div>

          <div className="evaluation-section">
            <h3>Retrieval performance</h3>

            <div className="evaluation-table-wrap">
              <table className="evaluation-table">
                <thead>
                  <tr>
                    <th>Method</th>
                    <th>Precision@K</th>
                    <th>Recall@K</th>
                    <th>MRR</th>
                    <th>nDCG@K</th>
                    <th>Hit Rate@K</th>
                  </tr>
                </thead>
                <tbody>
                  {methods.map(([name, result]) => (
                    <tr key={name}>
                      <td>{labels[name] ?? name}</td>
                      <td>{metric(result, `precision@${report.k}`).toFixed(3)}</td>
                      <td>{metric(result, `recall@${report.k}`).toFixed(3)}</td>
                      <td>{metric(result, "mrr").toFixed(3)}</td>
                      <td>{metric(result, `ndcg@${report.k}`).toFixed(3)}</td>
                      <td>{metric(result, `hit_rate@${report.k}`).toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="evaluation-section">
            <h3>Retrieval latency</h3>

            {methods.map(([name, result]) => (
              <div className="evaluation-latency" key={name}>
                <div>
                  <strong>{labels[name] ?? name}</strong>
                  <span>{result.latency.mean_ms.toFixed(2)} ms average</span>
                </div>
                <div className="evaluation-bar-track">
                  <div
                    className="evaluation-bar"
                    style={{
                      width: `${
                        Math.max(
                          2,
                          result.latency.mean_ms /
                            Math.max(
                              ...methods.map(
                                ([, item]) => item.latency.mean_ms
                              ),
                              1
                            ) * 100
                        )
                      }%`,
                    }}
                  />
                </div>
                <small>
                  Median: {result.latency.median_ms.toFixed(2)} ms
                  {" · "}
                  QPS: {result.queries_per_second.toFixed(2)}
                </small>
              </div>
            ))}
          </div>

          <div className="evaluation-section">
            <h3>Best performing methods</h3>

            <div className="evaluation-winners">
              {[
                [`Precision@${report.k}`, `precision@${report.k}`],
                [`Recall@${report.k}`, `recall@${report.k}`],
                ["MRR", "mrr"],
                [`nDCG@${report.k}`, `ndcg@${report.k}`],
              ].map(([title, key]) => {
                const winner = bestMethod(key);

                return (
                  <article key={key}>
                    <span>{title}</span>
                    <strong>
                      {winner ? labels[winner[0]] ?? winner[0] : "—"}
                    </strong>
                    <small>
                      {winner
                        ? metric(winner[1], key).toFixed(4)
                        : "No data"}
                    </small>
                  </article>
                );
              })}
            </div>
          </div>

          <p className="evaluation-footnote">
            Results depend on manually verified relevance labels.
            Latency excludes embedding time, answer generation and
            model-loading overhead.
          </p>
        </>
      )}
    </section>
  );
}
