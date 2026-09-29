import { NavLink, Route, Routes } from "react-router-dom";
import EvaluationDashboard from "./pages/EvaluationDashboard.jsx";

function HomePage() {
  return (
    <div className="mx-auto max-w-3xl px-6 py-16">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
        RAGOps
      </p>
      <h1 className="mt-3 text-4xl font-semibold tracking-tight">
        Compare RAG configurations with generation evaluation.
      </h1>
      <p className="mt-4 text-slate-400">
        Week 5 focuses on LLM-as-a-judge metrics: faithfulness, answer relevance,
        context relevance, and citation accuracy across Baseline, Dense, Hybrid,
        and Hybrid + Reranker.
      </p>
      <NavLink
        to="/evaluation"
        className="mt-8 inline-flex rounded-lg bg-cyan-500 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-400"
      >
        Open evaluation dashboard
      </NavLink>
    </div>
  );
}

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur">
        <nav className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <NavLink to="/" className="font-semibold tracking-tight">
            RAGOps
          </NavLink>
          <div className="flex gap-4 text-sm text-slate-400">
            <NavLink
              to="/"
              className={({ isActive }) =>
                isActive ? "text-cyan-400" : "hover:text-slate-200"
              }
            >
              Home
            </NavLink>
            <NavLink
              to="/evaluation"
              className={({ isActive }) =>
                isActive ? "text-cyan-400" : "hover:text-slate-200"
              }
            >
              Evaluation
            </NavLink>
          </div>
        </nav>
      </header>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/evaluation" element={<EvaluationDashboard />} />
      </Routes>
    </div>
  );
}
