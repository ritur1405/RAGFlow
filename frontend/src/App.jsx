import { NavLink, Route, Routes } from "react-router-dom";
import Ask from "./pages/Ask.jsx";
import Documents from "./pages/Documents.jsx";
import EvaluationDashboard from "./pages/EvaluationDashboard.jsx";
import RetrievalExperiments from "./pages/RetrievalExperiments.jsx";

const NAV = [
  { to: "/", label: "Home", end: true },
  { to: "/library", label: "Documents" },
  { to: "/ask", label: "Ask" },
  { to: "/experiments", label: "Experiments" },
  { to: "/evaluation", label: "Evaluation" },
];

const STEPS = [
  {
    to: "/library",
    title: "1. Upload",
    body: "PDFs are extracted, cleaned, chunked and embedded into PostgreSQL + pgvector.",
  },
  {
    to: "/ask",
    title: "2. Ask",
    body: "Dense retrieval plus grounded generation, with every claim cited to its source chunk.",
  },
  {
    to: "/experiments",
    title: "3. Compare retrievers",
    body: "Dense vs BM25 vs Hybrid (RRF) vs cross-encoder reranking on the same query, side by side.",
  },
  {
    to: "/evaluation",
    title: "4. Benchmark",
    body: "Recall@K, Precision@K, MRR, nDCG plus LLM-judged faithfulness and citation accuracy across four configurations.",
  },
];

function HomePage() {
  return (
    <div className="mx-auto max-w-4xl px-6 py-16">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">RAGOps</p>
      <h1 className="mt-3 text-4xl font-semibold tracking-tight">
        Measure retrieval quality, don't just chat with PDFs.
      </h1>
      <p className="mt-4 max-w-2xl text-slate-400">
        Upload documents, ask grounded questions, and compare Baseline, Dense, Hybrid and Hybrid + Cross-Encoder
        configurations on a shared, labelled benchmark with retrieval and generation metrics.
      </p>
      <div className="mt-10 grid gap-4 sm:grid-cols-2">
        {STEPS.map((step) => (
          <NavLink
            key={step.to}
            to={step.to}
            className="rounded-lg border border-slate-800 bg-slate-900/50 p-5 transition-colors hover:border-cyan-500/60"
          >
            <p className="font-semibold text-slate-100">{step.title}</p>
            <p className="mt-1 text-sm text-slate-400">{step.body}</p>
          </NavLink>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur">
        <nav className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-6 py-4">
          <NavLink to="/" className="font-semibold tracking-tight">
            RAGOps
          </NavLink>
          <div className="flex flex-wrap gap-4 text-sm text-slate-400">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) => (isActive ? "text-cyan-400" : "hover:text-slate-200")}
              >
                {item.label}
              </NavLink>
            ))}
          </div>
        </nav>
      </header>
      <Routes>
        <Route path="/" element={<HomePage />} />
        {/* Not /documents: that prefix is proxied to the API by vite and nginx. */}
        <Route path="/library" element={<Documents />} />
        <Route path="/ask" element={<Ask />} />
        <Route path="/experiments" element={<RetrievalExperiments />} />
        <Route path="/evaluation" element={<EvaluationDashboard />} />
      </Routes>
    </div>
  );
}
