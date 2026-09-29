import { useState, useEffect, useMemo, useCallback } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

/**
 * EvaluationDashboard.jsx
 * ------------------------
 * Member B — Generation Evaluation dashboard for RAGOps.
 *
 * Consumes:
 *   GET  /api/eval/compare              -> aggregate metrics for all 4 RAG configs
 *   GET  /api/eval/results/:run_id      -> per-question judge scores + reasoning
 *
 * Visual language: a dark "lab console" palette (deep teal-charcoal surfaces,
 * a mint-signal accent for strong scores, amber/coral for mid/weak scores) with
 * monospace numerals for every score and latency figure, since this is a
 * data-auditing tool for engineers, not a marketing surface.
 */

// Relative by default: the vite dev server proxies these paths in development
// and nginx proxies them in the Docker image, so no absolute host is baked in.
// Set VITE_API_BASE_URL to point at a backend on another origin.
const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

const CONFIG_META = {
  baseline: { label: "Baseline RAG", color: "#8FA3C4" },
  dense: { label: "Dense RAG", color: "#6FB1E0" },
  hybrid: { label: "Hybrid (Dense + BM25)", color: "#B18FE0" },
  hybrid_reranker: { label: "Hybrid + Reranker", color: "#5FE0B8" },
};

const METRIC_META = [
  { key: "avg_faithfulness", label: "Faithfulness" },
  { key: "avg_answer_relevance", label: "Answer Relevance" },
  { key: "avg_context_relevance", label: "Context Relevance" },
  { key: "avg_citation_accuracy", label: "Citation Accuracy" },
];

function scoreTone(score) {
  if (score === null || score === undefined) return "muted";
  if (score >= 0.8) return "good";
  if (score >= 0.5) return "warn";
  return "bad";
}

const TONE_CLASSES = {
  good: "text-[#7CE0B8]",
  warn: "text-[#E8B85C]",
  bad: "text-[#E37B6B]",
  muted: "text-[#5F726B]",
};

function ScoreText({ score, digits = 2 }) {
  if (score === null || score === undefined) {
    return <span className={`font-mono ${TONE_CLASSES.muted}`}>&mdash;</span>;
  }
  return (
    <span className={`font-mono font-semibold ${TONE_CLASSES[scoreTone(score)]}`}>
      {score.toFixed(digits)}
    </span>
  );
}

function fmtLatency(ms) {
  if (ms === null || ms === undefined) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;
}

/** Surfaces the backend's `detail` message so a 502/503 is legible in the UI. */
async function readError(res) {
  const body = await res.text().catch(() => "");
  try {
    const parsed = JSON.parse(body);
    if (parsed?.detail) return `${res.status}: ${parsed.detail}`;
  } catch {
    /* not JSON; fall through to the raw body */
  }
  return `${res.status} ${res.statusText}${body ? `: ${body}` : ""}`;
}

async function fetchJSON(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

async function postJSON(path, payload, apiKey) {
  const headers = { "Content-Type": "application/json" };
  // Sent per request from a value the operator types in. Deliberately not read
  // from import.meta.env: a VITE_ variable is compiled into the bundle and
  // would be served to every visitor.
  if (apiKey) headers["X-API-Key"] = apiKey;

  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

/** Remembers the key for this tab only, so it is not persisted to disk. */
const API_KEY_STORAGE = "ragops.evalApiKey";

function readStoredKey() {
  try {
    return sessionStorage.getItem(API_KEY_STORAGE) || "";
  } catch {
    return "";
  }
}

function storeKey(value) {
  try {
    if (value) sessionStorage.setItem(API_KEY_STORAGE, value);
    else sessionStorage.removeItem(API_KEY_STORAGE);
  } catch {
    /* private mode or blocked storage: the key simply is not remembered */
  }
}

/* ------------------------------------------------------------------ */
/* Summary card                                                        */
/* ------------------------------------------------------------------ */

function SummaryCard({ label, value, sublabel, tone }) {
  return (
    <div className="relative rounded-md border border-[#2A3B36] bg-[#16211E] px-5 py-4 overflow-hidden">
      <div
        className="absolute left-0 top-0 h-full w-[3px]"
        style={{
          backgroundColor:
            tone === "good" ? "#7CE0B8" : tone === "warn" ? "#E8B85C" : tone === "bad" ? "#E37B6B" : "#3A4C46",
        }}
      />
      <p className="text-[13px] text-[#8FA39B] mb-2">{label}</p>
      <p className="font-mono text-3xl text-[#E8EDE9] leading-none">{value}</p>
      {sublabel && <p className="text-[12px] text-[#5F726B] mt-2">{sublabel}</p>}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Comparison table                                                   */
/* ------------------------------------------------------------------ */

function ComparisonTable({ configs, onSelectRun, selectedRunId }) {
  return (
    <div className="rounded-md border border-[#2A3B36] bg-[#16211E] overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[#2A3B36] text-left text-[#8FA39B]">
            <th className="px-4 py-3 font-normal">Configuration</th>
            {METRIC_META.map((m) => (
              <th key={m.key} className="px-4 py-3 font-normal text-right">
                {m.label}
              </th>
            ))}
            <th className="px-4 py-3 font-normal text-right">Overall</th>
            <th className="px-4 py-3 font-normal text-right">Avg Latency</th>
            <th className="px-4 py-3 font-normal text-right">Questions</th>
          </tr>
        </thead>
        <tbody>
          {configs.map((row) => {
            const meta = CONFIG_META[row.rag_config] || { label: row.rag_config_label, color: "#8FA3C4" };
            const isSelected = row.run_id && row.run_id === selectedRunId;
            return (
              <tr
                key={row.rag_config}
                onClick={() => row.run_id && onSelectRun(row.run_id, row.rag_config)}
                className={`border-b border-[#20302B] last:border-b-0 transition-colors ${
                  row.run_id ? "cursor-pointer hover:bg-[#1D2C28]" : "opacity-50"
                } ${isSelected ? "bg-[#1D2C28]" : ""}`}
              >
                <td className="px-4 py-3">
                  <span className="inline-flex items-center gap-2 text-[#E8EDE9]">
                    <span
                      className="inline-block h-2 w-2 rounded-full"
                      style={{ backgroundColor: meta.color }}
                    />
                    {meta.label}
                  </span>
                </td>
                {METRIC_META.map((m) => (
                  <td key={m.key} className="px-4 py-3 text-right">
                    <ScoreText score={row[m.key]} />
                  </td>
                ))}
                <td className="px-4 py-3 text-right">
                  <ScoreText score={row.avg_overall_score} />
                </td>
                <td className="px-4 py-3 text-right font-mono text-[#C7D3CE]">
                  {fmtLatency(row.avg_latency_ms)}
                </td>
                <td className="px-4 py-3 text-right font-mono text-[#8FA39B]">
                  {row.num_questions || "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Bar chart                                                          */
/* ------------------------------------------------------------------ */

function MetricBarChart({ configs }) {
  const chartData = METRIC_META.map((m) => {
    const row = { metric: m.label };
    configs.forEach((c) => {
      row[c.rag_config] = c[m.key] ?? 0;
    });
    return row;
  });

  return (
    <div className="rounded-md border border-[#2A3B36] bg-[#16211E] p-4">
      <p className="text-[13px] text-[#8FA39B] mb-4">Metric comparison across configurations</p>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ top: 4, right: 8, left: -12, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#233330" vertical={false} />
          <XAxis dataKey="metric" tick={{ fill: "#8FA39B", fontSize: 12 }} axisLine={{ stroke: "#2A3B36" }} />
          <YAxis
            domain={[0, 1]}
            tick={{ fill: "#8FA39B", fontSize: 12 }}
            axisLine={{ stroke: "#2A3B36" }}
          />
          <Tooltip
            contentStyle={{ background: "#1D2C28", border: "1px solid #2A3B36", borderRadius: 6 }}
            labelStyle={{ color: "#E8EDE9" }}
            formatter={(value) => value.toFixed(3)}
          />
          <Legend
            formatter={(value) => (
              <span style={{ color: "#C7D3CE", fontSize: 12 }}>
                {CONFIG_META[value]?.label || value}
              </span>
            )}
          />
          {Object.keys(CONFIG_META).map((key) => (
            <Bar key={key} dataKey={key} fill={CONFIG_META[key].color} radius={[3, 3, 0, 0]} maxBarSize={28} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Detail accordion — question-by-question judge feedback              */
/* ------------------------------------------------------------------ */

function CitationCheck({ contextIds, citedIds }) {
  const fabricated = citedIds.filter((id) => !contextIds.includes(id));
  const valid = citedIds.filter((id) => contextIds.includes(id));

  return (
    <div className="flex flex-wrap gap-1.5">
      {valid.map((id) => (
        <span
          key={id}
          className="font-mono text-[11px] px-2 py-0.5 rounded border border-[#2F5C4A] bg-[#16281F] text-[#7CE0B8]"
        >
          {id}
        </span>
      ))}
      {fabricated.map((id) => (
        <span
          key={id}
          className="font-mono text-[11px] px-2 py-0.5 rounded border border-[#5C2F2F] bg-[#281616] text-[#E37B6B]"
          title="Cited chunk ID not found in retrieved context"
        >
          {id} ✕
        </span>
      ))}
      {citedIds.length === 0 && (
        <span className="font-mono text-[11px] px-2 py-0.5 rounded border border-[#3A4C46] text-[#5F726B]">
          no citations found
        </span>
      )}
    </div>
  );
}

function MetricRow({ label, detail }) {
  return (
    <div className="py-2.5 border-b border-[#20302B] last:border-b-0">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[13px] text-[#C7D3CE]">{label}</span>
        <ScoreText score={detail?.score} digits={3} />
      </div>
      {detail?.reasoning && (
        <p className="text-[12.5px] leading-relaxed text-[#8FA39B]">{detail.reasoning}</p>
      )}
    </div>
  );
}

function QuestionAccordionItem({ item, isOpen, onToggle }) {
  return (
    <div className="border-b border-[#20302B] last:border-b-0">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between gap-4 px-4 py-3 text-left hover:bg-[#1D2C28] transition-colors"
      >
        <div className="min-w-0 flex-1">
          <p className="text-[14px] text-[#E8EDE9] truncate">{item.question}</p>
        </div>
        <div className="flex items-center gap-4 shrink-0">
          <ScoreText score={item.overall_score} />
          <span
            className={`font-mono text-[#5F726B] transition-transform ${isOpen ? "rotate-90" : ""}`}
          >
            &gt;
          </span>
        </div>
      </button>

      {isOpen && (
        <div className="px-4 pb-4">
          {item.ground_truth && (
            <p className="text-[12.5px] text-[#5F726B] mb-3">
              <span className="text-[#8FA39B]">Reference answer:</span> {item.ground_truth}
            </p>
          )}

          <div className="rounded border border-[#2A3B36] bg-[#121C19] p-3 mb-3">
            <p className="text-[12px] text-[#8FA39B] mb-1">Generated answer</p>
            <p className="text-[13.5px] text-[#C7D3CE] leading-relaxed">{item.generated_answer}</p>
          </div>

          <div className="mb-3">
            <p className="text-[12px] text-[#8FA39B] mb-1.5">Citation check</p>
            <CitationCheck contextIds={item.context_chunk_ids} citedIds={item.cited_chunk_ids} />
          </div>

          <div className="rounded border border-[#2A3B36] bg-[#121C19] px-3">
            <MetricRow label="Faithfulness" detail={item.faithfulness} />
            <MetricRow label="Answer Relevance" detail={item.answer_relevance} />
            <MetricRow label="Context Relevance" detail={item.context_relevance} />
            <MetricRow label="Citation Accuracy" detail={item.citation_accuracy} />
          </div>

          <p className="text-[11.5px] text-[#5F726B] mt-2 font-mono">
            latency: {fmtLatency(item.latency_ms)}
          </p>
        </div>
      )}
    </div>
  );
}

function DetailDrawer({ runSummary, results, loading, error }) {
  const [openId, setOpenId] = useState(null);

  if (!runSummary) {
    return (
      <div className="rounded-md border border-[#2A3B36] bg-[#16211E] p-8 text-center">
        <p className="text-[13px] text-[#5F726B]">
          Select a configuration row above to inspect question-by-question judge feedback.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-md border border-[#2A3B36] bg-[#16211E] overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#2A3B36]">
        <div>
          <p className="text-[14px] text-[#E8EDE9]">
            {CONFIG_META[runSummary.rag_config]?.label || runSummary.rag_config_label}
          </p>
          <p className="text-[12px] text-[#5F726B]">
            {runSummary.dataset_name} · {runSummary.num_questions} questions
          </p>
        </div>
        <ScoreText score={runSummary.avg_overall_score} />
      </div>

      {loading && <p className="px-4 py-6 text-[13px] text-[#8FA39B]">Loading question detail…</p>}
      {error && <p className="px-4 py-6 text-[13px] text-[#E37B6B]">{error}</p>}

      {!loading && !error && (
        <div>
          {results.map((item) => (
            <QuestionAccordionItem
              key={item.id}
              item={item}
              isOpen={openId === item.id}
              onToggle={() => setOpenId(openId === item.id ? null : item.id)}
            />
          ))}
          {results.length === 0 && (
            <p className="px-4 py-6 text-[13px] text-[#5F726B]">No results recorded for this run.</p>
          )}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Main page                                                           */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Run trigger                                                         */
/* ------------------------------------------------------------------ */

const ALL_CONFIG_KEYS = Object.keys(CONFIG_META);

/** Parses the textarea: one question per line, optional "question | reference answer". */
function parseQuestions(raw) {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [question, ...rest] = line.split("|");
      const ground_truth = rest.join("|").trim();
      return { question: question.trim(), ground_truth: ground_truth || null };
    });
}

function RunEvaluationPanel({ onCompleted }) {
  const [open, setOpen] = useState(false);
  const [datasetName, setDatasetName] = useState("benchmark");
  const [rawQuestions, setRawQuestions] = useState("");
  const [selected, setSelected] = useState(ALL_CONFIG_KEYS);
  const [apiKey, setApiKey] = useState(readStoredKey);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [finished, setFinished] = useState(null);

  const questions = useMemo(() => parseQuestions(rawQuestions), [rawQuestions]);
  const canRun =
    !running &&
    questions.length > 0 &&
    selected.length > 0 &&
    datasetName.trim() &&
    apiKey.trim();

  function toggleConfig(key) {
    setSelected((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    );
  }

  async function handleRun() {
    setRunning(true);
    setError(null);
    setFinished(null);
    try {
      storeKey(apiKey.trim());
      const data = await postJSON(
        "/api/eval/run",
        {
          dataset_name: datasetName.trim(),
          questions,
          rag_configs: selected,
        },
        apiKey.trim(),
      );
      setFinished(data.runs || []);
      onCompleted?.();
    } catch (err) {
      // Includes the backend's detail text, so a 503 (judge not configured) or
      // a 502 (pipeline failed) is readable here instead of silently showing
      // stale numbers.
      setError(err.message || "Evaluation failed.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="mb-8 rounded-md border border-[#2A3B36] bg-[#16211E]">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-5 py-3 text-left"
      >
        <span className="text-[15px] text-[#E8EDE9]">Run an evaluation</span>
        <span className="text-[12px] text-[#5F726B]">{open ? "Hide" : "Show"}</span>
      </button>

      {open && (
        <div className="space-y-4 border-t border-[#2A3B36] px-5 py-4">
          <div>
            <label className="mb-1 block text-[12px] text-[#8FA39B]">Dataset name</label>
            <input
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              disabled={running}
              className="w-full rounded border border-[#2A3B36] bg-[#0F1614] px-3 py-2 text-[13px] text-[#E8EDE9] outline-none focus:border-[#6FB1E0] disabled:opacity-50"
            />
          </div>

          <div>
            <label className="mb-1 block text-[12px] text-[#8FA39B]">
              Questions — one per line, optionally{" "}
              <span className="font-mono">question | reference answer</span>
            </label>
            <textarea
              value={rawQuestions}
              onChange={(e) => setRawQuestions(e.target.value)}
              disabled={running}
              rows={5}
              placeholder={"What was Q3 revenue? | It grew 12%\nWho signed the agreement?"}
              className="w-full rounded border border-[#2A3B36] bg-[#0F1614] px-3 py-2 font-mono text-[12px] text-[#E8EDE9] outline-none focus:border-[#6FB1E0] disabled:opacity-50"
            />
            <p className="mt-1 text-[11px] text-[#5F726B]">
              {questions.length} question{questions.length === 1 ? "" : "s"} parsed
            </p>
          </div>

          <div>
            <label className="mb-1 block text-[12px] text-[#8FA39B]">
              API key <span className="text-[#5F726B]">(X-API-Key; server-side EVAL_API_KEY)</span>
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              disabled={running}
              autoComplete="off"
              placeholder="required — running an evaluation spends API credits"
              className="w-full rounded border border-[#2A3B36] bg-[#0F1614] px-3 py-2 font-mono text-[12px] text-[#E8EDE9] outline-none focus:border-[#6FB1E0] disabled:opacity-50"
            />
            <p className="mt-1 text-[11px] text-[#5F726B]">
              Kept for this browser tab only; never written into the build.
            </p>
          </div>

          <div>
            <label className="mb-2 block text-[12px] text-[#8FA39B]">Configurations</label>
            <div className="flex flex-wrap gap-2">
              {ALL_CONFIG_KEYS.map((key) => {
                const active = selected.includes(key);
                return (
                  <button
                    key={key}
                    onClick={() => toggleConfig(key)}
                    disabled={running}
                    className={`rounded border px-3 py-1.5 text-[12px] transition-colors disabled:opacity-50 ${
                      active
                        ? "border-[#6FB1E0] bg-[#16262E] text-[#E8EDE9]"
                        : "border-[#2A3B36] text-[#8FA39B] hover:text-[#E8EDE9]"
                    }`}
                  >
                    {CONFIG_META[key].label}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handleRun}
              disabled={!canRun}
              className="rounded bg-[#6FB1E0] px-4 py-2 text-[13px] font-semibold text-[#0F1614] transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {running ? "Running…" : "Run evaluation"}
            </button>
            {running && (
              <span className="text-[12px] text-[#8FA39B]">
                Executing real pipelines and judge calls — this can take a while.
              </span>
            )}
          </div>

          {error && (
            <div className="rounded border border-[#5C2F2F] bg-[#281616] px-4 py-3 text-[12px] text-[#E37B6B]">
              {error}
            </div>
          )}

          {finished && (
            <div className="rounded border border-[#2F5C48] bg-[#152820] px-4 py-3 text-[12px] text-[#7CE0B8]">
              Completed {finished.length} run{finished.length === 1 ? "" : "s"}:{" "}
              {finished.map((r) => `${r.rag_config_label} (${r.status})`).join(", ")}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function EvaluationDashboard() {
  const [configs, setConfigs] = useState([]);
  const [compareLoading, setCompareLoading] = useState(true);
  const [compareError, setCompareError] = useState(null);

  const [selectedRunId, setSelectedRunId] = useState(null);
  const [selectedConfigKey, setSelectedConfigKey] = useState(null);
  const [runSummary, setRunSummary] = useState(null);
  const [results, setResults] = useState([]);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState(null);

  const loadCompare = useCallback(async () => {
    setCompareLoading(true);
    setCompareError(null);
    try {
      const data = await fetchJSON("/api/eval/compare");
      setConfigs(data.configs || []);
    } catch (err) {
      setCompareError(err.message || "Failed to load comparison data.");
    } finally {
      setCompareLoading(false);
    }
  }, []);

  useEffect(() => {
    loadCompare();
  }, [loadCompare]);

  const handleSelectRun = useCallback(async (runId, configKey) => {
    setSelectedRunId(runId);
    setSelectedConfigKey(configKey);
    setDetailLoading(true);
    setDetailError(null);
    try {
      const data = await fetchJSON(`/api/eval/results/${runId}`);
      setRunSummary(data.run);
      setResults(data.results || []);
    } catch (err) {
      setDetailError(err.message || "Failed to load run detail.");
    } finally {
      setDetailLoading(false);
    }
  }, []);

  // Best-performing configuration (by overall score) drives the top summary cards.
  const bestConfig = useMemo(() => {
    const withScores = configs.filter((c) => c.avg_overall_score !== null && c.avg_overall_score !== undefined);
    if (withScores.length === 0) return null;
    return withScores.reduce((best, c) => (c.avg_overall_score > best.avg_overall_score ? c : best));
  }, [configs]);

  return (
    <div className="min-h-screen bg-[#0F1614] px-6 py-8 md:px-10">
      <div className="max-w-6xl mx-auto">
        <header className="mb-8">
          <p className="text-[13px] text-[#5F726B] mb-1">RAGOps · Generation Evaluation</p>
          <h1 className="text-2xl text-[#E8EDE9]" style={{ fontFamily: "'Sora', 'Inter', sans-serif" }}>
            Evaluation Dashboard
          </h1>
          <p className="text-[13px] text-[#8FA39B] mt-1">
            Faithfulness, relevance, and citation accuracy across Baseline, Dense, Hybrid, and Hybrid + Reranker.
          </p>
        </header>

        {compareError && (
          <div className="mb-6 rounded-md border border-[#5C2F2F] bg-[#281616] px-4 py-3 text-[13px] text-[#E37B6B]">
            {compareError}
          </div>
        )}

        <RunEvaluationPanel onCompleted={loadCompare} />

        {/* Summary cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <SummaryCard
            label="Faithfulness"
            value={bestConfig ? bestConfig.avg_faithfulness?.toFixed(2) ?? "—" : "—"}
            sublabel={bestConfig ? `Best: ${CONFIG_META[bestConfig.rag_config]?.label}` : compareLoading ? "Loading…" : "No runs yet"}
            tone={bestConfig ? scoreTone(bestConfig.avg_faithfulness) : "muted"}
          />
          <SummaryCard
            label="Answer Relevance"
            value={bestConfig ? bestConfig.avg_answer_relevance?.toFixed(2) ?? "—" : "—"}
            sublabel={bestConfig ? `Best: ${CONFIG_META[bestConfig.rag_config]?.label}` : compareLoading ? "Loading…" : "No runs yet"}
            tone={bestConfig ? scoreTone(bestConfig.avg_answer_relevance) : "muted"}
          />
          <SummaryCard
            label="Citation Accuracy"
            value={bestConfig ? bestConfig.avg_citation_accuracy?.toFixed(2) ?? "—" : "—"}
            sublabel={bestConfig ? `Best: ${CONFIG_META[bestConfig.rag_config]?.label}` : compareLoading ? "Loading…" : "No runs yet"}
            tone={bestConfig ? scoreTone(bestConfig.avg_citation_accuracy) : "muted"}
          />
          <SummaryCard
            label="Avg Latency"
            value={bestConfig ? fmtLatency(bestConfig.avg_latency_ms) : "—"}
            sublabel={bestConfig ? `${CONFIG_META[bestConfig.rag_config]?.label}` : compareLoading ? "Loading…" : "No runs yet"}
            tone="muted"
          />
        </div>

        {/* Comparison table + chart */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-[15px] text-[#E8EDE9]">Configuration comparison</h2>
            <button
              onClick={loadCompare}
              className="text-[12px] text-[#8FA39B] hover:text-[#E8EDE9] border border-[#2A3B36] rounded px-3 py-1.5 transition-colors"
            >
              Refresh
            </button>
          </div>

          {compareLoading ? (
            <div className="rounded-md border border-[#2A3B36] bg-[#16211E] p-8 text-center">
              <p className="text-[13px] text-[#8FA39B]">Loading comparison data…</p>
            </div>
          ) : (
            <div className="space-y-4">
              <ComparisonTable configs={configs} onSelectRun={handleSelectRun} selectedRunId={selectedRunId} />
              <MetricBarChart configs={configs} />
            </div>
          )}
        </div>

        {/* Detail drawer */}
        <div>
          <h2 className="text-[15px] text-[#E8EDE9] mb-3">Question-level judge feedback</h2>
          <DetailDrawer
            runSummary={selectedConfigKey ? runSummary : null}
            results={results}
            loading={detailLoading}
            error={detailError}
          />
        </div>
      </div>
    </div>
  );
}