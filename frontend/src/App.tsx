import { useEffect, useRef, useState } from "react";
import {
  Activity, BookOpen, Bot, CheckCircle2, ChevronRight, FileText,
  Library, Loader2, Plus, Send, ShieldCheck, Sparkles, UploadCloud,
  X, AlertCircle, ArrowUpRight
} from "lucide-react";
import "./App.css";
import ResearchMetrics from "./components/ResearchMetrics";
import BenchmarkAnnotator from "./components/BenchmarkAnnotator";

const API = "http://127.0.0.1:8000";

type DocumentItem = {
  document_id: string;
  file_name: string;
  page_count: number;
  section_count: number;
  chunk_count: number;
};

type Source = {
  source_id: string;
  document_name: string;
  section: string;
  page_numbers: number[];
  chunk_id?: string;
};

type EvidenceDetail = {
  source_id: string;
  document_name: string;
  section: string;
  page_numbers: number[];
  entailment: number;
  contradiction: number;
  neutral: number;
};

type ClaimResult = {
  claim_id: string;
  text: string;
  cited_sources: string[];
  status: "SUPPORTED" | "CONTRADICTED" | "NOT ENOUGH EVIDENCE";
  entailment_score: number;
  contradiction_score: number;
  supporting_evidence: EvidenceDetail[];
};

type VerificationResult = {
  verification_status: string;
  claims: ClaimResult[];
  verified_claims: number;
  total_claims: number;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  verification?: VerificationResult | null;
  verificationStatus?: string;
};

function App() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");
  const [apiOnline, setApiOnline] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const chatEnd = useRef<HTMLDivElement>(null);

  async function loadDocuments() {
    try {
      const response = await fetch(`${API}/documents`);
      if (!response.ok) throw new Error();
      const data = await response.json();
      setDocuments(data.documents ?? []);
      setApiOnline(true);
      setError("");
    } catch {
      setApiOnline(false);
      setError("Cannot connect to TrustRAG API. Check that FastAPI is running on port 8000.");
    }
  }

  useEffect(() => { void loadDocuments(); }, []);
  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  async function uploadPDF(file?: File) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Please select a PDF file.");
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      setError("PDF must be smaller than 50 MB.");
      return;
    }

    setUploading(true);
    setError("");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API}/documents/upload`, {
        method: "POST",
        body: formData
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "PDF upload failed.");
      await loadDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function askQuestion(event: React.FormEvent) {
    event.preventDefault();
    const currentQuestion = question.trim();
    if (!currentQuestion || asking) return;

    if (!documents.length) {
      setError("Upload a research paper before asking a question.");
      return;
    }

    setError("");
    setQuestion("");
    setMessages(prev => [...prev, { role: "user", content: currentQuestion }]);
    setAsking(true);

    try {
      const response = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: currentQuestion,
          top_k: 5,
          candidate_k: 30
        })
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Question request failed.");

      setMessages(prev => [...prev, {
        role: "assistant",
        content: data.answer,
        sources: data.sources ?? [],
        verification: data.verification ?? null,
        verificationStatus: data.verification_status ?? "not_run"
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: err instanceof Error ? err.message : "Something went wrong."
      }]);
    } finally {
      setAsking(false);
    }
  }

  const totalPages = documents.reduce((sum, doc) => sum + doc.page_count, 0);
  const totalChunks = documents.reduce((sum, doc) => sum + doc.chunk_count, 0);
  const scrollTo = (id: string) =>
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#">
          <span className="brand-name">Trust<span>RAG</span></span>
          <span className="brand-caption">RESEARCH INTELLIGENCE</span>
        </a>

        <nav className="top-navigation">
          <button onClick={() => scrollTo("library")}>Library</button>
          <button onClick={() => scrollTo("assistant")}>AI assistant</button>
          <button onClick={() => scrollTo("overview")}>Overview</button>
        <button onClick={() => scrollTo("benchmark-annotation")}>Benchmark</button>
</nav>

        <div className="header-actions">
          <span className={`connection ${apiOnline ? "connected" : ""}`}>
            <span className="status-dot" />
            {apiOnline ? "System online" : "System offline"}
          </span>
          <button className="header-upload" onClick={() => fileInput.current?.click()}>
            <Plus size={16} /> Add paper
          </button>
        </div>
      </header>

      <main className="page-content">
        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow"><span /> EVIDENCE-LED RESEARCH WORKSPACE</div>
            <h1>Research you can<br /><em>trust.</em></h1>
            <p>
              Bring your scientific papers together. Find the evidence that matters,
              ask better questions, and keep every answer connected to its source.
            </p>
            <div className="hero-actions">
              <button className="button-red" onClick={() => fileInput.current?.click()}>
                <UploadCloud size={17} />
                {uploading ? "Indexing paper..." : "Add a research paper"}
              </button>
              <button className="button-outline" onClick={() => scrollTo("library")}>
                Explore library <ChevronRight size={16} />
              </button>
            </div>
            <div className="hero-proof">
              <ShieldCheck size={15} /> Local-first research · Evidence-linked answers
            </div>
          </div>

          <div className="hero-art">
            <div className="art-grid" />
            <div className="paper paper-back">
              <span className="paper-line short" />
              <span className="paper-line" />
              <span className="paper-line" />
              <span className="paper-line medium" />
            </div>
            <div className="paper paper-front">
              <div className="paper-topline">
                <span>TRUSTED SOURCE</span><span className="paper-dot" />
              </div>
              <div className="paper-title">Evidence<br />in context.</div>
              <div className="paper-rule" />
              <span className="paper-line medium" />
              <span className="paper-line" />
              <span className="paper-line short" />
              <div className="paper-citation"><BookOpen size={14} /> Source · Page 04</div>
            </div>
            <div className="evidence-float">
              <span className="float-check"><CheckCircle2 size={17} /></span>
              <span><strong>Evidence connected</strong><small>Source traceability</small></span>
            </div>
            <div className="hero-orbit orbit-one" />
            <div className="hero-orbit orbit-two" />
            <div className="hero-art-label">READ · RETRIEVE · VERIFY</div>
          </div>
          <div className="hero-bottom">
            <span>BUILT FOR SCIENTIFIC DISCOVERY</span>
            <span>01 / RESEARCH LIBRARY</span>
          </div>
        </section>

        {error && (
          <div className="error-banner">
            <AlertCircle size={17} />
            <span>{error}</span>
            <button onClick={() => setError("")}><X size={16} /></button>
          </div>
        )}

        <section className="metrics" id="overview">
          <div className="metrics-intro">
            <span className="section-kicker">YOUR WORKSPACE</span>
            <h2>A clearer view<br />of your research.</h2>
          </div>
          <div className="metric">
            <span className="metric-icon"><Library size={18} /></span>
            <strong>{documents.length}</strong>
            <span>Research papers</span>
            <small>Indexed in your library</small>
          </div>
          <div className="metric">
            <span className="metric-icon"><FileText size={18} /></span>
            <strong>{totalPages}</strong>
            <span>Pages processed</span>
            <small>Across indexed papers</small>
          </div>
          <div className="metric">
            <span className="metric-icon"><Activity size={18} /></span>
            <strong>{totalChunks}</strong>
            <span>Evidence chunks</span>
            <small>Available for retrieval</small>
          </div>
          <div className="metric verification-metric">
            <span className="metric-icon"><ShieldCheck size={18} /></span>
            <strong>In progress</strong>
            <span>Claim verification</span>
            <small>Independent verification not enabled</small>
          </div>
        </section>

        <section className="section-heading" id="library">
          <div>
            <span className="section-kicker">THE COLLECTION</span>
            <h2>Your research library<span>.</span></h2>
            <p>Scientific documents, organised and ready to explore.</p>
          </div>
          <button className="text-action" onClick={loadDocuments}>
            <Activity size={16} /> Refresh library
          </button>
        </section>

        <section className="library-layout">
          <div className="library-main">
            <div
              className={`upload-card ${dragging ? "dragging" : ""}`}
              onClick={() => fileInput.current?.click()}
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={e => {
                e.preventDefault();
                setDragging(false);
                void uploadPDF(e.dataTransfer.files[0]);
              }}
              role="button"
              tabIndex={0}
              onKeyDown={e => {
                if (e.key === "Enter" || e.key === " ") fileInput.current?.click();
              }}
            >
              <span className="upload-symbol">
                {uploading ? <Loader2 className="spin" size={23} /> : <UploadCloud size={23} />}
              </span>
              <span className="upload-copy">
                <strong>{uploading ? "Indexing your paper..." : "Add to your research collection"}</strong>
                <small>Drop a PDF here or browse files · Maximum 50 MB</small>
              </span>
              <span className="upload-browse">Browse PDFs <ArrowUpRight size={15} /></span>
            </div>

            <div className="document-heading">
              <span>YOUR DOCUMENTS</span>
              <span>{documents.length} PAPERS</span>
            </div>

            {documents.length === 0 ? (
              <div className="empty-library">
                <div className="empty-document"><FileText size={27} /></div>
                <h3>Your library is ready for its first paper.</h3>
                <p>Upload a scientific PDF to start retrieving passages and asking evidence-grounded questions.</p>
              </div>
            ) : (
              <div className="document-grid">
                {documents.map((doc, index) => (
                  <article className="document-card" key={doc.document_id}>
                    <div className="document-card-top">
                      <span className="pdf-label">PDF / {String(index + 1).padStart(2, "0")}</span>
                      <CheckCircle2 size={17} />
                    </div>
                    <div className="document-cover">
                      <FileText size={30} />
                      <span>RESEARCH<br />PAPER</span>
                      <i />
                    </div>
                    <div className="document-info">
                      <strong title={doc.file_name}>{doc.file_name}</strong>
                      <span>{doc.page_count} pages <i /> {doc.section_count} sections</span>
                    </div>
                    <div className="document-card-foot">
                      <span>{doc.chunk_count} evidence chunks</span>
                      <span className="indexed-label">INDEXED</span>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </div>

          <aside className="assistant-card" id="assistant">
            <div className="assistant-card-head">
              <div>
                <span className="section-kicker">YOUR RESEARCH COMPANION</span>
                <h2>Ask TrustRAG<span>.</span></h2>
              </div>
              <span className="assistant-mark"><Bot size={20} /></span>
            </div>

            <p className="assistant-intro">
              Ask questions across your indexed papers. Answers are generated from
              retrieved evidence and include source details.
            </p>

            <div className="assistant-status">
              <span className={`status-dot ${apiOnline ? "online" : ""}`} />
              {apiOnline ? "Local research engine connected" : "Waiting for backend connection"}
            </div>

            <div className="chat-area">
              {messages.length === 0 ? (
                <div className="chat-empty">
                  <div className="sparkle-icon"><Sparkles size={20} /></div>
                  <h3>What are you investigating?</h3>
                  <p>Start with one of these research questions.</p>
                  <div className="suggested-questions">
                    {[
                      "What is the main contribution of these papers?",
                      "Which methods do the authors use?",
                      "What limitations do the papers report?"
                    ].map(item => (
                      <button key={item} onClick={() => setQuestion(item)}>
                        {item}<ChevronRight size={15} />
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="messages">
                  {messages.map((message, index) => (
                    <div className={`message ${message.role}`} key={`${index}-${message.role}`}>
                      <div className="message-avatar">
                        {message.role === "assistant" ? <Bot size={16} /> : "Y"}
                      </div>
                      <div className="message-body">
                        <div className="message-role">
                          {message.role === "assistant" ? "TrustRAG" : "You"}
                        </div>
                        <p>{message.content}</p>
                        {message.role === "assistant" && (
                          <div className="verification-panel">
                            <div className="verification-panel-head">
                              <div>
                                <span className="verification-eyebrow">TRUSTRAG VERIFICATION</span>
                                <strong>Claim assessment</strong>
                              </div>
                              <span className={`verification-overall ${message.verificationStatus || "not_run"}`}>
                                {message.verificationStatus === "verified" ? "Verified" :
                                 message.verificationStatus === "partially_supported" ? "Partially supported" :
                                 message.verificationStatus === "contradicted" ? "Contradiction found" :
                                 message.verificationStatus === "insufficient_evidence" ? "Insufficient evidence" :
                                 message.verificationStatus === "error" ? "Verification error" :
                                 "Not run"}
                              </span>
                            </div>

                            {message.verification?.claims?.length ? (
                              <>
                                <div className="verification-summary">
                                  {message.verification.verified_claims} of {message.verification.total_claims} claims supported
                                </div>
                                <div className="claim-list">
                                  {message.verification.claims.map(claim => (
                                    <details className="claim-card" key={claim.claim_id}>
                                      <summary>
                                        <span className={`claim-status ${claim.status === "SUPPORTED" ? "supported" : claim.status === "CONTRADICTED" ? "contradicted" : "uncertain"}`}>
                                          {claim.status === "SUPPORTED" ? "✓" : claim.status === "CONTRADICTED" ? "!" : "?"}
                                        </span>
                                        <span className="claim-text">{claim.text}</span>
                                        <span className="claim-chevron">⌄</span>
                                      </summary>
                                      <div className="claim-details">
                                        <div className="claim-scores">
                                          <span>Entailment <strong>{(claim.entailment_score * 100).toFixed(1)}%</strong></span>
                                          <span>Contradiction <strong>{(claim.contradiction_score * 100).toFixed(1)}%</strong></span>
                                        </div>
                                        {claim.supporting_evidence?.length ? (
                                          <div className="claim-evidence">
                                            <span className="claim-evidence-title">MATCHED EVIDENCE</span>
                                            {claim.supporting_evidence.map((item, index) => (
                                              <div className="claim-evidence-item" key={`${claim.claim_id}-${item.source_id}-${index}`}>
                                                <strong>{item.source_id} · {item.document_name}</strong>
                                                <span>{item.section} · Pages {item.page_numbers?.join(", ") || "Not available"}</span>
                                              </div>
                                            ))}
                                          </div>
                                        ) : (
                                          <p className="claim-no-evidence">No sufficiently strong supporting or contradicting passage was identified.</p>
                                        )}
                                      </div>
                                    </details>
                                  ))}
                                </div>
                              </>
                            ) : (
                              <p className="verification-empty">
                                {message.verificationStatus === "error"
                                  ? "The verifier could not complete this assessment."
                                  : "Claim-level verification was not run for this response."}
                              </p>
                            )}
                          </div>
                        )}

                        {message.sources && message.sources.length > 0 && (
                          <div className="source-list">
                            <div className="source-heading">RETRIEVED SOURCES</div>
                            {message.sources.map(source => (
                              <div className="source-item" key={source.source_id}>
                                <span className="source-id">{source.source_id}</span>
                                <div>
                                  <strong>{source.document_name}</strong>
                                  <span>{source.section} · Pages {source.page_numbers?.join(", ") || "Not available"}</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  {asking && (
                    <div className="thinking">
                      <Loader2 className="spin" size={16} />
                      Finding evidence and preparing an answer…
                    </div>
                  )}
                  <div ref={chatEnd} />
                </div>
              )}
            </div>

            <form className="chat-composer" onSubmit={askQuestion}>
              <input
                value={question}
                onChange={e => setQuestion(e.target.value)}
                placeholder={documents.length ? "Ask about your research..." : "Upload a paper to begin"}
                disabled={!documents.length || asking}
              />
              <button
                type="submit"
                disabled={!question.trim() || asking || !documents.length}
                aria-label="Send question"
              >
                {asking ? <Loader2 className="spin" size={17} /> : <Send size={17} />}
              </button>
            </form>
            <div className="assistant-disclaimer">
              <ShieldCheck size={13} />
              Evidence-linked responses · Automated claim-level assessment
            </div>
          </aside>
        </section>


              <section id="benchmark-annotation" className="evaluation-page-section">

                <BenchmarkAnnotator />

              </section>


        <section id="research-metrics" className="evaluation-page-section">


          <ResearchMetrics />


        </section>



        <footer className="page-footer">
          <span>TRUSTRAG · RESEARCH INTELLIGENCE</span>
          <span><span className="status-dot online" /> LOCAL-FIRST WORKSPACE</span>
        </footer>
      </main>

      <input
        ref={fileInput}
        type="file"
        accept=".pdf,application/pdf"
        hidden
        onChange={e => void uploadPDF(e.target.files?.[0])}
      />
    </div>
  );
}

export default App;
