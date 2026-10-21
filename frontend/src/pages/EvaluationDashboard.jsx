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
import { fetchJSON, fmtLatency, postJSON } from "../lib/api.js";

/**
 * EvaluationDashboard.jsx
 * ------------------------
 * Member B — Generation Evaluation dashboard for RAGOps.
 *
 * Consumes:
 *   GET  /api/eval/compare              -> aggregate metrics for all 4 RAG configs
 *   GET  /api/eval/results/:run_id      -> per-question judge scores + reasoning
 *   GET  /api/eval/benchmark            -> the shared page-labelled benchmark
 *
 * Two families of metrics: generation quality from the LLM judge, and
 * retrieval quality (Recall@K, Precision@K, MRR, nDCG@K) against the
 * benchmark's page labels.
 *
 * Visual language: a dark "lab console" palette (deep teal-charcoal surfaces,
 * a mint-signal accent for strong scores, amber/coral for mid/weak scores) with
 * monospace numerals for every score and latency figure, since this is a
 * data-auditing tool for engineers, not a marketing surface.
 */

const CONFIG_META = {
  baseline: { label: "Baseline RAG", color: "#8FA3C4" },
  dense: { label: "Dense RAG", color: "#6FB1E0" },
  hybrid: { label: "Hybrid (Dense + BM25)", color: "#B18FE0" },
  hybrid_reranker: { label: "Hybrid + Cross-Encoder", color: "#5FE0B8" },
};

const METRIC_META = [
  { key: "avg_faithfulness", label: "Faithfulness" },
  { key: "avg_answer_relevance", label: "Answer Relevance" },
  { key: "avg_context_relevance", label: "Context Relevance" },
  { key: "avg_citation_accuracy", label: "Citation Accuracy" },
];

const RETRIEVAL_META = [
  { key: "avg_recall_at_k", label: "Recall@K" },
  { key: "avg_precision_at_k", label: "Precision@K" },
  { key: "avg_mrr", label: "MRR" },
  { key: "avg_ndcg_at_k", label: "nDCG@K" },
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
  const retrievalK = configs.find((c) => c.retrieval_k)?.retrieval_k;
  return (
    <div className="rounded-md border border-[#2A3B36] bg-[#16211E] overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-[11px] uppercase tracking-wider text-[#5F726B]">
            <th />
            <th colSpan={METRIC_META.length + 1} className="px-4 pt-3 font-normal text-right">
              Generation · LLM judge
            </th>
            <th colSpan={RETRIEVAL_META.length} className="border-l border-[#2A3B36] px-4 pt-3 font-normal text-right">
              Retrieval{retrievalK ? ` · K=${retrievalK}` : ""}
            </th>
            <th colSpan={2} />
          </tr>
          <tr className="border-b border-[#2A3B36] text-left text-[#8FA39B]">
            <th className="px-4 py-3 font-normal">Configuration</th>
            {METRIC_META.map((m) => (
              <th key={m.key} className="px-4 py-3 font-normal text-right">
                {m.label}
              </th>
            ))}
            <th className="px-4 py-3 font-normal text-right">Overall</th>
            {RETRIEVAL_META.map((m, i) => (
              <th
                key={m.key}
                className={`px-4 py-3 font-normal text-right ${i === 0 ? "border-l border-[#2A3B36]" : ""}`}
              >
                {m.label}
              </th>
            ))}
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
                {RETRIEVAL_META.map((m, i) => (
                  <td
                    key={m.key}
                    className={`px-4 py-3 text-right ${i === 0 ? "border-l border-[#2A3B36]" : ""}`}
                  >
                    <ScoreText score={row[m.key]} />
                  </td>
                ))}
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

function MetricBarChart({ configs, metrics, title }) {
  const chartData = metrics.map((m) => {
    const row = { metric: m.label };
    configs.forEach((c) => {
      row[c.rag_config] = c[m.key] ?? 0;
    });
    return row;
  });

  return (
    <div className="rounded-md border border-[#2A3B36] bg-[#16211E] p-4">
      <p className="text-[13px] text-[#8FA39B] mb-4">{title}</p>
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

/** Which ranked chunks came from a labelled page, plus the four retrieval scores. */
function RetrievalCheck({ item }) {
  const relevant = new Set(item.relevant_pages);
  const r = item.retrieval;
  return (
    <div className="mt-3 rounded border border-[#2A3B36] bg-[#121C19] p-3">
      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-[12px] text-[#8FA39B]">Retrieved pages</span>
        {item.retrieved_pages.map((page, i) => {
          const hit = relevant.has(page);
          return (
            <span
              key={i}
              title={hit ? "From a labelled page" : "Not from a labelled page"}
              className={`font-mono text-[11px] px-2 py-0.5 rounded border ${
                hit
                  ? "border-[#2F5C4A] bg-[#16281F] text-[#7CE0B8]"
                  : "border-[#3A4C46] text-[#5F726B]"
              }`}
            >
              #{i + 1} p.{page ?? "?"}
            </span>
          );
        })}
        <span className="ml-2 text-[11px] text-[#5F726B]">
          labelled: p.{item.relevant_pages.join(", p.")}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-2 text-[12px] md:grid-cols-4">
        {[
          ["Recall@" + r.k, r.recall_at_k],
          ["Precision@" + r.k, r.precision_at_k],
          ["Reciprocal rank", r.reciprocal_rank],
          ["nDCG@" + r.k, r.ndcg_at_k],
        ].map(([label, value]) => (
          <div key={label} className="flex items-center justify-between gap-2">
            <span className="text-[#8FA39B]">{label}</span>
            <ScoreText score={value} />
          </div>
        ))}
      </div>
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
          {item.retrieval && (
            <span className="hidden text-[11px] text-[#5F726B] md:inline">
              nDCG <ScoreText score={item.retrieval.ndcg_at_k} />
            </span>
          )}
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

          {item.retrieval && <RetrievalCheck item={item} />}

          <p className="text-[11.5px] text-[#5F726B] mt-2 font-mono">
            pipeline: {fmtLatency(item.pipeline_latency_ms)} · judge: {fmtLatency(item.judge_latency_ms)}
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

  // The loaded benchmark keeps its page labels, which the one-line-per-question
  // text format cannot express. Editing the text drops back to typed questions.
  const [benchmark, setBenchmark] = useState(null);
  const [loadingBenchmark, setLoadingBenchmark] = useState(false);

  const typedQuestions = useMemo(() => parseQuestions(rawQuestions), [rawQuestions]);
  const questions = benchmark ? benchmark.questions : typedQuestions;

  async function loadBenchmark() {
    setLoadingBenchmark(true);
    setError(null);
    try {
      const data = await fetchJSON("/api/eval/benchmark");
      setBenchmark(data);
      setDatasetName(data.dataset_name);
      setRawQuestions(
        data.questions.map((q) => `${q.question} | ${q.ground_truth ?? ""}`).join("\n"),
      );
    } catch (err) {
      setError(err.message || "Could not load the benchmark.");
    } finally {
      setLoadingBenchmark(false);
    }
  }
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
              onChange={(e) => {
                setBenchmark(null);
                setRawQuestions(e.target.value);
              }}
              disabled={running}
              rows={5}
              placeholder={"What was Q3 revenue? | It grew 12%\nWho signed the agreement?"}
              className="w-full rounded border border-[#2A3B36] bg-[#0F1614] px-3 py-2 font-mono text-[12px] text-[#E8EDE9] outline-none focus:border-[#6FB1E0] disabled:opacity-50"
            />
            <div className="mt-1 flex flex-wrap items-center justify-between gap-2">
              <p className="text-[11px] text-[#5F726B]">
                {questions.length} question{questions.length === 1 ? "" : "s"}
                {benchmark
                  ? ` from ${benchmark.dataset_name}, page-labelled: retrieval metrics will be computed`
                  : " parsed; typed questions get generation metrics only"}
              </p>
              <button
                type="button"
                onClick={loadBenchmark}
                disabled={running || loadingBenchmark}
                className="rounded border border-[#2A3B36] px-3 py-1 text-[12px] text-[#8FA39B] hover:text-[#E8EDE9] disabled:opacity-50"
              >
                {loadingBenchmark ? "Loading…" : "Load shared benchmark"}
              </button>
            </div>
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

  // Each summary card shows the best configuration for its own metric, so a
  // card never labels one configuration's number as another's best.
  const summaryCards = useMemo(() => {
    const pick = (key, better) => {
      const measured = configs.filter((c) => c[key] !== null && c[key] !== undefined);
      return measured.length ? measured.reduce((best, c) => (better(c[key], best[key]) ? c : best)) : null;
    };
    const higher = (a, b) => a > b;
    return [
      { key: "avg_overall_score", label: "Overall (judge)", better: higher },
      { key: "avg_faithfulness", label: "Faithfulness", better: higher },
      { key: "avg_citation_accuracy", label: "Citation Accuracy", better: higher },
      { key: "avg_ndcg_at_k", label: "nDCG@K", better: higher },
      { key: "avg_mrr", label: "MRR", better: higher },
      { key: "avg_latency_ms", label: "Fastest pipeline", better: (a, b) => a < b, latency: true },
    ].map((card) => ({ ...card, best: pick(card.key, card.better) }));
  }, [configs]);

  return (
    <div className="min-h-screen bg-[#0F1614] px-6 py-8 md:px-10">
      <div className="max-w-6xl mx-auto">
        <header className="mb-8">
          <p className="text-[13px] text-[#5F726B] mb-1">RAGOps · Evaluation</p>
          <h1 className="text-2xl text-[#E8EDE9]" style={{ fontFamily: "'Sora', 'Inter', sans-serif" }}>
            Evaluation Dashboard
          </h1>
          <p className="text-[13px] text-[#8FA39B] mt-1">
            Generation quality (faithfulness, relevance, citation accuracy) and retrieval quality (Recall@K, Precision@K, MRR, nDCG@K) across Baseline, Dense, Hybrid, and Hybrid + Cross-Encoder.
          </p>
        </header>

        {compareError && (
          <div className="mb-6 rounded-md border border-[#5C2F2F] bg-[#281616] px-4 py-3 text-[13px] text-[#E37B6B]">
            {compareError}
          </div>
        )}

        <RunEvaluationPanel onCompleted={loadCompare} />

        {/* Summary cards */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8">
          {summaryCards.map(({ key, label, best, latency }) => (
            <SummaryCard
              key={key}
              label={label}
              value={best ? (latency ? fmtLatency(best[key]) : best[key].toFixed(2)) : "—"}
              sublabel={
                best
                  ? `Best: ${CONFIG_META[best.rag_config]?.label || best.rag_config_label}`
                  : compareLoading
                    ? "Loading…"
                    : "No runs yet"
              }
              tone={best && !latency ? scoreTone(best[key]) : "muted"}
            />
          ))}
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
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <MetricBarChart
                  configs={configs}
                  metrics={METRIC_META}
                  title="Generation quality (LLM judge)"
                />
                <MetricBarChart
                  configs={configs}
                  metrics={RETRIEVAL_META}
                  title="Retrieval quality against page labels"
                />
              </div>
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