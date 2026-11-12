import { useEffect, useState } from "react";

const API = "http://127.0.0.1:8000";

type Metric = {
  evaluated: number;
  positive?: number;
  correct?: number;
  rate?: number | null;
  accuracy?: number | null;
};

type Report = {
  review_status: string;
  total_questions: number;
  metrics: {
    faithfulness: Metric;
    citation_correctness: Metric;
    unsupported_claim_rate: Metric;
    claim_verification_accuracy: Metric;
  };
};

function percentage(value?: number | null) {
  return value == null ? "Not evaluated" : `${(value * 100).toFixed(1)}%`;
}

export default function ResearchMetrics() {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API}/evaluation/answer-quality`);
      if (!response.ok) throw new Error("Answer-quality report not available yet.");
      setReport(await response.json());
    } catch (e: any) {
      setError(e.message || "Could not load report.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  return (
    <section className="research-metrics-panel">
      <div className="research-metrics-heading">
        <div>
          <h2>Answer Quality Evaluation</h2>
          <p>Human-reviewed answer and citation performance</p>
        </div>
        <button onClick={load} disabled={loading}>Refresh</button>
      </div>

      {loading && <p>Loading answer-quality results...</p>}
      {error && <p>{error}</p>}

      {report && (
        <>
          <p>
            Review status: <strong>{report.review_status}</strong>
            {" · "}
            {report.total_questions} benchmark questions
          </p>
          <div className="research-metric-grid">
            <article>
              <span>Faithfulness</span>
              <strong>{percentage(report.metrics.faithfulness.rate)}</strong>
              <small>{report.metrics.faithfulness.evaluated} reviewed</small>
            </article>
            <article>
              <span>Citation correctness</span>
              <strong>{percentage(report.metrics.citation_correctness.rate)}</strong>
              <small>{report.metrics.citation_correctness.evaluated} reviewed</small>
            </article>
            <article>
              <span>Unsupported claim rate</span>
              <strong>{percentage(report.metrics.unsupported_claim_rate.rate)}</strong>
              <small>{report.metrics.unsupported_claim_rate.evaluated} reviewed</small>
            </article>
            <article>
              <span>Claim verification accuracy</span>
              <strong>{percentage(report.metrics.claim_verification_accuracy.accuracy)}</strong>
              <small>{report.metrics.claim_verification_accuracy.evaluated} labelled</small>
            </article>
          </div>
        </>
      )}
    </section>
  );
}
