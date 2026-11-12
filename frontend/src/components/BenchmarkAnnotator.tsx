import { useEffect, useMemo, useState } from "react";

const API = "http://127.0.0.1:8000";

type Candidate = {
  chunk_id: string;
  text: string;
  document?: string;
  section?: string;
  pages?: number[] | string;
};

type BenchmarkItem = {
  question: string;
  relevant_chunk_ids: string[];
};

function extractCandidates(data: any): Candidate[] {
  const found: Candidate[] = [];
  const seen = new Set<string>();

  function visit(value: any) {
    if (!value) return;

    if (Array.isArray(value)) {
      value.forEach(visit);
      return;
    }

    if (typeof value !== "object") return;

    const id = value.chunk_id ?? value.id;
    const content = value.text ?? value.content ?? value.chunk_text;

    if (id !== undefined && typeof content === "string" && !seen.has(String(id))) {
      seen.add(String(id));
      found.push({
        chunk_id: String(id),
        text: content,
        document: value.document ?? value.document_name ?? value.filename,
        section: value.section,
        pages: value.pages ?? value.page,
      });
      return;
    }

    Object.values(value).forEach(visit);
  }

  visit(data);
  return found;
}

export default function BenchmarkAnnotator() {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [benchmark, setBenchmark] = useState<BenchmarkItem[]>([]);
  const [question, setQuestion] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  async function loadData() {
    setLoading(true);
    setMessage("");

    try {
      const [candidateResponse, benchmarkResponse] = await Promise.all([
        fetch(`${API}/evaluation/candidates`),
        fetch(`${API}/evaluation/benchmark`),
      ]);

      if (!candidateResponse.ok) {
        throw new Error("Candidate file not found. Generate annotation candidates first.");
      }
      if (!benchmarkResponse.ok) {
        throw new Error("Could not load the saved benchmark.");
      }

      const candidateData = await candidateResponse.json();
      const benchmarkData = await benchmarkResponse.json();

      setCandidates(extractCandidates(candidateData));
      setBenchmark(Array.isArray(benchmarkData) ? benchmarkData : []);
    } catch (error: any) {
      setMessage(error.message || "Could not load annotation data.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadData();
  }, []);

  const filtered = useMemo(() => {
    const term = search.toLowerCase().trim();
    if (!term) return candidates;

    return candidates.filter((item) =>
      `${item.chunk_id} ${item.text} ${item.document ?? ""} ${item.section ?? ""}`
        .toLowerCase()
        .includes(term)
    );
  }, [candidates, search]);

  function toggleChunk(id: string) {
    setSelected((current) =>
      current.includes(id)
        ? current.filter((item) => item !== id)
        : [...current, id]
    );
  }

  function addQuestion() {
    const cleanQuestion = question.trim();

    if (!cleanQuestion || selected.length === 0) {
      setMessage("Enter a question and select at least one relevant chunk.");
      return;
    }

    setBenchmark((current) => [
      ...current,
      {
        question: cleanQuestion,
        relevant_chunk_ids: [...selected],
      },
    ]);

    setQuestion("");
    setSelected([]);
    setMessage("Question added. Remember to save your benchmark.");
  }

  async function saveBenchmark() {
    if (benchmark.length === 0) {
      setMessage("Add at least one annotated question before saving.");
      return;
    }

    setSaving(true);
    setMessage("");

    try {
      const response = await fetch(`${API}/evaluation/benchmark`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(benchmark),
      });

      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || "Could not save benchmark.");
      }

      setMessage(`Saved ${result.saved} annotated questions successfully.`);
    } catch (error: any) {
      setMessage(error.message || "Save failed.");
    } finally {
      setSaving(false);
    }
  }

  if (loading) return <section className="benchmark-panel">Loading annotation data...</section>;

  return (
    <section className="benchmark-panel">
      <div className="benchmark-heading">
        <div>
          <h2>Benchmark Annotation</h2>
          <p>Write a research question and select every chunk that genuinely answers it.</p>
        </div>
        <button onClick={loadData}>Reload candidates</button>
      </div>

      <div className="benchmark-stats">
        <span>{candidates.length} candidate chunks</span>
        <span>{benchmark.length} annotated questions</span>
        <span>{selected.length} selected</span>
      </div>

      <label className="benchmark-label" htmlFor="benchmark-question">
        Evaluation question
      </label>
      <textarea
        id="benchmark-question"
        value={question}
        onChange={(event) => setQuestion(event.target.value)}
        placeholder="Example: What method is used to generate sentence embeddings?"
        rows={3}
      />

      <div className="benchmark-candidate-heading">
        <h3>Select relevant evidence chunks</h3>
        <input
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Search chunks..."
          aria-label="Search candidate chunks"
        />
      </div>

      <div className="benchmark-candidate-list">
        {filtered.map((candidate) => (
          <label className="benchmark-candidate" key={candidate.chunk_id}>
            <input
              type="checkbox"
              checked={selected.includes(candidate.chunk_id)}
              onChange={() => toggleChunk(candidate.chunk_id)}
            />
            <div>
              <strong>{candidate.chunk_id}</strong>
              <small>
                {[candidate.document, candidate.section,
                  candidate.pages !== undefined ? `Pages: ${Array.isArray(candidate.pages) ? candidate.pages.join(", ") : candidate.pages}` : ""]
                  .filter(Boolean).join(" · ")}
              </small>
              <p>{candidate.text}</p>
            </div>
          </label>
        ))}
        {filtered.length === 0 && <p>No matching candidate chunks found.</p>}
      </div>

      <div className="benchmark-actions">
        <button onClick={addQuestion}>Add annotated question</button>
        <button onClick={saveBenchmark} disabled={saving || benchmark.length === 0}>
          {saving ? "Saving..." : "Save benchmark"}
        </button>
      </div>

      {message && <p className="benchmark-message">{message}</p>}

      <div className="benchmark-saved-list">
        <h3>Questions in benchmark ({benchmark.length})</h3>
        {benchmark.map((item, index) => (
          <div className="benchmark-saved-item" key={`${index}-${item.question}`}>
            <div>
              <strong>{index + 1}. {item.question}</strong>
              <small>{item.relevant_chunk_ids.length} relevant chunk(s): {item.relevant_chunk_ids.join(", ")}</small>
            </div>
            <button
              onClick={() => setBenchmark((current) => current.filter((_, i) => i !== index))}
              aria-label={`Remove question ${index + 1}`}
            >
              Remove
            </button>
          </div>
        ))}
      </div>
    </section>
  );
}
