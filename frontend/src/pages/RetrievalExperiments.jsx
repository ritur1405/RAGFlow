import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { deleteJSON, fetchJSON, fmtLatency, postJSON } from "../lib/api.js";

/**
 * RetrievalExperiments.jsx
 * ------------------------
 * Week 4 — Dense vs BM25 vs Hybrid retrieval comparison.
 *
 * Consumes:
 *   POST   /api/experiments/compare         -> ranked chunks + latency per strategy
 *   GET    /api/experiments/runs            -> recent runs
 *   GET    /api/experiments/configs         -> saved configurations
 *   POST   /api/experiments/configs         -> save a configuration
 *   DELETE /api/experiments/configs/:id
 *
 * Retrieval only: nothing is generated or judged here, so a comparison is
 * fast and cheap. Chunks retrieved by more than one strategy are marked, and
 * hovering a chunk highlights it in every column, so agreement between the
 * retrievers is visible at a glance. Same palette as the evaluation dashboard.
 */

const STRATEGIES = [
  { key: "dense", label: "Dense", color: "#6FB1E0" },
  { key: "bm25", label: "BM25", color: "#E8B85C" },
  { key: "hybrid", label: "Hybrid (RRF)", color: "#B18FE0" },
  { key: "hybrid_reranker", label: "Hybrid + Cross-Encoder", color: "#5FE0B8" },
  { key: "baseline", label: "Baseline", color: "#8FA3C4" },
];
const STRATEGY_META = Object.fromEntries(STRATEGIES.map((s) => [s.key, s]));
const DEFAULT_STRATEGIES = ["dense", "bm25", "hybrid"];

const panel = "rounded-md border border-[#2A3B36] bg-[#16211E]";
const input =
  "rounded border border-[#2A3B36] bg-[#0F1614] px-3 py-2 text-[13px] text-[#E8EDE9] placeholder-[#5F726B] focus:border-[#5FE0B8] focus:outline-none";

function fmtScore(score) {
  if (score === null || score === undefined) return "—";
  return Math.abs(score) >= 10 ? score.toFixed(1) : score.toFixed(4);
}

function fmtTime(iso) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

/* ------------------------------------------------------------------ */
/* Controls                                                            */
/* ------------------------------------------------------------------ */

function StrategyToggle({ selected, onChange }) {
  const toggle = (key) =>
    onChange(selected.includes(key) ? selected.filter((k) => k !== key) : [...selected, key]);

  return (
    <div className="flex flex-wrap gap-2">
      {STRATEGIES.map((s) => {
        const on = selected.includes(s.key);
        return (
          <button
            key={s.key}
            type="button"
            onClick={() => toggle(s.key)}
            aria-pressed={on}
            className={`rounded-full border px-3 py-1 text-[12px] transition-colors ${
              on ? "text-[#0F1614]" : "border-[#2A3B36] text-[#8FA39B] hover:text-[#E8EDE9]"
            }`}
            style={on ? { backgroundColor: s.color, borderColor: s.color } : undefined}
          >
            {s.label}
          </button>
        );
      })}
    </div>
  );
}

function ConfigBar({ configs, activeConfigId, onApply, onSave, onDelete, canSave }) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      await onSave(name.trim());
      setName("");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-2 border-t border-[#2A3B36] pt-4 md:flex-row md:items-center">
      <select
        className={`${input} md:w-64`}
        value={activeConfigId || ""}
        onChange={(e) => onApply(e.target.value || null)}
        aria-label="Saved configuration"
      >
        <option value="">No saved configuration</option>
        {configs.map((c) => (
          <option key={c.id} value={c.id}>
            {c.name} · {c.strategies.join(", ")} · k={c.top_k}
          </option>
        ))}
      </select>
      {activeConfigId && (
        <button
          type="button"
          onClick={() => onDelete(activeConfigId)}
          className="text-[12px] text-[#E37B6B] hover:underline"
        >
          Delete
        </button>
      )}
      <div className="flex flex-1 gap-2 md:justify-end">
        <input
          className={`${input} flex-1 md:max-w-xs`}
          placeholder="Save current settings as…"
          value={name}
          onChange={(e) => setName(e.target.value)}
          maxLength={100}
        />
        <button
          type="button"
          onClick={save}
          disabled={!name.trim() || !canSave || busy}
          className="rounded border border-[#2A3B36] px-3 py-2 text-[12px] text-[#8FA39B] hover:text-[#E8EDE9] disabled:opacity-40"
        >
          Save
        </button>
      </div>
      {error && <p className="text-[12px] text-[#E37B6B]">{error}</p>}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Results                                                             */
/* ------------------------------------------------------------------ */

function ChunkCard({ chunk, color, overlapCount, totalStrategies, hovered, onHover }) {
  const [expanded, setExpanded] = useState(false);
  const shared = overlapCount > 1;

  return (
    <li
      onMouseEnter={() => onHover(chunk.chunk_id)}
      onMouseLeave={() => onHover(null)}
      className={`rounded border bg-[#0F1614] p-3 transition-colors ${
        hovered ? "border-[#E8EDE9]" : "border-[#2A3B36]"
      }`}
    >
      <div className="mb-1.5 flex items-center justify-between gap-2 text-[11px]">
        <span className="font-mono text-[#8FA39B]">
          <span className="font-semibold" style={{ color }}>
            #{chunk.rank}
          </span>{" "}
          · chunk {chunk.chunk_id} · p.{chunk.page_number ?? "?"}
        </span>
        <span className="font-mono text-[#E8EDE9]">{fmtScore(chunk.score)}</span>
      </div>
      {chunk.file_name && (
        <p className="mb-1 truncate text-[11px] text-[#5F726B]" title={chunk.file_name}>
          {chunk.file_name}
        </p>
      )}
      <p className={`text-[12px] leading-relaxed text-[#C9D4CF] ${expanded ? "" : "line-clamp-4"}`}>
        {chunk.content}
      </p>
      <div className="mt-2 flex items-center justify-between">
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="text-[11px] text-[#8FA39B] hover:text-[#E8EDE9]"
        >
          {expanded ? "Show less" : "Show more"}
        </button>
        {shared && (
          <span className="rounded bg-[#22302C] px-1.5 py-0.5 text-[10px] text-[#7CE0B8]">
            in {overlapCount}/{totalStrategies}
          </span>
        )}
      </div>
    </li>
  );
}

function StrategyColumn({ result, overlap, totalStrategies, hoveredChunk, onHover }) {
  const meta = STRATEGY_META[result.strategy] || { label: result.label, color: "#8FA39B" };
  return (
    <section className={`${panel} flex min-w-0 flex-col p-4`}>
      <header className="mb-3 border-b border-[#2A3B36] pb-3">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-[14px] font-semibold" style={{ color: meta.color }}>
            {result.label}
          </h3>
          <span className="font-mono text-[12px] text-[#E8EDE9]">{fmtLatency(result.latency_ms)}</span>
        </div>
        <p className="mt-0.5 text-[11px] text-[#5F726B]">Score: {result.score_label}</p>
      </header>

      {result.status === "failed" ? (
        <div className="rounded border border-[#5C2F2F] bg-[#281616] p-3 text-[12px] text-[#E37B6B]">
          This strategy failed and returned no results.
          <p className="mt-1 break-words font-mono text-[11px]">{result.error_message}</p>
        </div>
      ) : result.chunks.length === 0 ? (
        <p className="text-[12px] text-[#8FA39B]">
          No chunks retrieved. Dense applies a distance gate and BM25 drops zero-score chunks, so an
          empty list can be a correct answer.
        </p>
      ) : (
        <ol className="space-y-2">
          {result.chunks.map((chunk) => (
            <ChunkCard
              key={chunk.chunk_id}
              chunk={chunk}
              color={meta.color}
              overlapCount={overlap.get(chunk.chunk_id) || 1}
              totalStrategies={totalStrategies}
              hovered={hoveredChunk === chunk.chunk_id}
              onHover={onHover}
            />
          ))}
        </ol>
      )}
    </section>
  );
}

function LatencyChart({ results }) {
  const data = results
    .filter((r) => r.status === "completed")
    .map((r) => ({ name: r.label, key: r.strategy, latency: r.latency_ms }));
  if (data.length === 0) return null;

  return (
    <div className={`${panel} p-4`}>
      <h3 className="mb-3 text-[13px] text-[#E8EDE9]">Retrieval latency</h3>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 24, right: 24 }}>
            <CartesianGrid stroke="#22302C" horizontal={false} />
            <XAxis type="number" stroke="#5F726B" fontSize={11} unit="ms" />
            <YAxis type="category" dataKey="name" stroke="#8FA39B" fontSize={11} width={120} />
            <Tooltip
              cursor={{ fill: "#22302C" }}
              contentStyle={{ background: "#16211E", border: "1px solid #2A3B36", fontSize: 12 }}
              formatter={(v) => [fmtLatency(v), "Latency"]}
            />
            <Bar dataKey="latency" radius={[0, 3, 3, 0]}>
              {data.map((d) => (
                <Cell key={d.key} fill={STRATEGY_META[d.key]?.color || "#8FA39B"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/** Pairwise count of shared chunk IDs: how much the retrievers agree. */
function OverlapMatrix({ results }) {
  const completed = results.filter((r) => r.status === "completed");
  if (completed.length < 2) return null;
  const ids = Object.fromEntries(completed.map((r) => [r.strategy, new Set(r.chunks.map((c) => c.chunk_id))]));

  return (
    <div className={`${panel} overflow-x-auto p-4`}>
      <h3 className="mb-1 text-[13px] text-[#E8EDE9]">Agreement</h3>
      <p className="mb-3 text-[11px] text-[#5F726B]">Chunks shared by each pair of strategies.</p>
      <table className="text-[12px]">
        <thead>
          <tr>
            <th />
            {completed.map((r) => (
              <th key={r.strategy} className="px-3 py-1 text-left font-normal" style={{ color: STRATEGY_META[r.strategy]?.color }}>
                {r.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {completed.map((row) => (
            <tr key={row.strategy}>
              <th className="py-1 pr-3 text-left font-normal" style={{ color: STRATEGY_META[row.strategy]?.color }}>
                {row.label}
              </th>
              {completed.map((col) => {
                const shared = [...ids[row.strategy]].filter((id) => ids[col.strategy].has(id)).length;
                return (
                  <td key={col.strategy} className="px-3 py-1 font-mono text-[#E8EDE9]">
                    {row.strategy === col.strategy ? (
                      <span className="text-[#5F726B]">{shared}</span>
                    ) : (
                      shared
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RunHistory({ runs, loading, onRefresh, onRerun }) {
  return (
    <div className={`${panel} overflow-hidden`}>
      <div className="flex items-center justify-between border-b border-[#2A3B36] px-4 py-3">
        <h3 className="text-[13px] text-[#E8EDE9]">Recent runs</h3>
        <button
          type="button"
          onClick={onRefresh}
          className="rounded border border-[#2A3B36] px-3 py-1 text-[12px] text-[#8FA39B] hover:text-[#E8EDE9]"
        >
          Refresh
        </button>
      </div>
      {loading ? (
        <p className="p-4 text-[12px] text-[#8FA39B]">Loading…</p>
      ) : runs.length === 0 ? (
        <p className="p-4 text-[12px] text-[#8FA39B]">No experiments yet. Run a comparison above.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]">
            <thead className="text-left text-[#5F726B]">
              <tr>
                <th className="px-4 py-2 font-normal">When</th>
                <th className="px-4 py-2 font-normal">Query</th>
                <th className="px-4 py-2 font-normal">Strategy</th>
                <th className="px-4 py-2 font-normal">k</th>
                <th className="px-4 py-2 font-normal">Latency</th>
                <th className="px-4 py-2 font-normal">Top chunks</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.run_id} className="border-t border-[#22302C] text-[#C9D4CF]">
                  <td className="whitespace-nowrap px-4 py-2 text-[#8FA39B]">{fmtTime(run.created_at)}</td>
                  <td className="max-w-xs px-4 py-2">
                    <button
                      type="button"
                      onClick={() => onRerun(run.query)}
                      className="block max-w-full truncate text-left hover:text-[#5FE0B8]"
                      title="Use this query"
                    >
                      {run.query}
                    </button>
                  </td>
                  <td className="whitespace-nowrap px-4 py-2" style={{ color: STRATEGY_META[run.strategy]?.color }}>
                    {run.label}
                  </td>
                  <td className="px-4 py-2 font-mono">{run.top_k}</td>
                  <td className="whitespace-nowrap px-4 py-2 font-mono">
                    {run.status === "failed" ? <span className="text-[#E37B6B]">failed</span> : fmtLatency(run.latency_ms)}
                  </td>
                  <td className="px-4 py-2 font-mono text-[#8FA39B]">{run.top_chunk_ids.join(", ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function RetrievalExperiments() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [strategies, setStrategies] = useState(DEFAULT_STRATEGIES);
  const [activeConfigId, setActiveConfigId] = useState(null);

  const [comparison, setComparison] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [hoveredChunk, setHoveredChunk] = useState(null);

  const [configs, setConfigs] = useState([]);
  const [runs, setRuns] = useState([]);
  const [runsLoading, setRunsLoading] = useState(true);

  const loadConfigs = useCallback(async () => {
    try {
      setConfigs(await fetchJSON("/api/experiments/configs"));
    } catch (err) {
      setError(err.message);
    }
  }, []);

  const loadRuns = useCallback(async () => {
    setRunsLoading(true);
    try {
      setRuns(await fetchJSON("/api/experiments/runs?limit=30"));
    } catch (err) {
      setError(err.message);
    } finally {
      setRunsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadConfigs();
    loadRuns();
  }, [loadConfigs, loadRuns]);

  const applyConfig = (id) => {
    setActiveConfigId(id);
    const config = configs.find((c) => c.id === id);
    if (config) {
      setStrategies(config.strategies);
      setTopK(config.top_k);
    }
  };

  const saveConfig = async (name) => {
    const created = await postJSON("/api/experiments/configs", { name, strategies, top_k: topK });
    await loadConfigs();
    setActiveConfigId(created.id);
  };

  const deleteConfig = async (id) => {
    try {
      await deleteJSON(`/api/experiments/configs/${id}`);
      setActiveConfigId(null);
      await loadConfigs();
    } catch (err) {
      setError(err.message);
    }
  };

  const topKValid = Number.isInteger(topK) && topK >= 1 && topK <= 20;
  const canRun = query.trim() && strategies.length > 0 && topKValid && !running;

  const runComparison = async (e) => {
    e?.preventDefault();
    if (!canRun) return;
    setRunning(true);
    setError(null);
    try {
      // Explicit strategies and top_k always win; config_id only links the runs.
      const result = await postJSON("/api/experiments/compare", {
        query: query.trim(),
        strategies,
        top_k: topK,
        config_id: activeConfigId,
      });
      setComparison(result);
      loadRuns();
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  };

  // chunk_id -> number of strategies that retrieved it.
  const overlap = useMemo(() => {
    const counts = new Map();
    for (const r of comparison?.results || []) {
      for (const c of r.chunks) counts.set(c.chunk_id, (counts.get(c.chunk_id) || 0) + 1);
    }
    return counts;
  }, [comparison]);

  const columns = comparison?.results.length || 0;
  const gridCols =
    columns >= 4 ? "lg:grid-cols-4" : columns === 3 ? "lg:grid-cols-3" : columns === 2 ? "lg:grid-cols-2" : "";

  return (
    <div className="min-h-screen bg-[#0F1614] px-6 py-8 md:px-10">
      <div className="mx-auto max-w-6xl">
        <header className="mb-8">
          <p className="mb-1 text-[13px] text-[#5F726B]">RAGOps · Retrieval Experiments</p>
          <h1 className="text-2xl text-[#E8EDE9]">Dense vs BM25 vs Hybrid</h1>
          <p className="mt-1 text-[13px] text-[#8FA39B]">
            Run one query through several retrievers and compare what each one ranks, how much they agree, and how
            long each takes. Retrieval only: nothing is generated, so comparisons are fast.
          </p>
        </header>

        <form onSubmit={runComparison} className={`${panel} mb-6 space-y-4 p-5`}>
          <div className="flex flex-col gap-3 md:flex-row">
            <input
              className={`${input} flex-1`}
              placeholder="Ask something the uploaded documents can answer…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              maxLength={1000}
              aria-label="Query"
            />
            <label className="flex items-center gap-2 text-[12px] text-[#8FA39B]">
              Top K
              <input
                type="number"
                min={1}
                max={20}
                className={`${input} w-20`}
                value={Number.isNaN(topK) ? "" : topK}
                onChange={(e) => setTopK(parseInt(e.target.value, 10))}
              />
            </label>
            <button
              type="submit"
              disabled={!canRun}
              className="rounded bg-[#5FE0B8] px-5 py-2 text-[13px] font-semibold text-[#0F1614] hover:bg-[#7CE0B8] disabled:opacity-40"
            >
              {running ? "Running…" : "Compare"}
            </button>
          </div>
          <StrategyToggle selected={strategies} onChange={setStrategies} />
          {!topKValid && <p className="text-[12px] text-[#E8B85C]">Top K must be between 1 and 20.</p>}
          <ConfigBar
            configs={configs}
            activeConfigId={activeConfigId}
            onApply={applyConfig}
            onSave={saveConfig}
            onDelete={deleteConfig}
            canSave={strategies.length > 0 && topKValid}
          />
        </form>

        {error && (
          <div className="mb-6 rounded-md border border-[#5C2F2F] bg-[#281616] px-4 py-3 text-[13px] text-[#E37B6B]">
            {error}
          </div>
        )}

        {comparison && (
          <div className="mb-8 space-y-4">
            <div className={`grid grid-cols-1 gap-4 md:grid-cols-2 ${gridCols}`}>
              {comparison.results.map((result) => (
                <StrategyColumn
                  key={result.strategy}
                  result={result}
                  overlap={overlap}
                  totalStrategies={columns}
                  hoveredChunk={hoveredChunk}
                  onHover={setHoveredChunk}
                />
              ))}
            </div>
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <LatencyChart results={comparison.results} />
              <OverlapMatrix results={comparison.results} />
            </div>
          </div>
        )}

        <RunHistory runs={runs} loading={runsLoading} onRefresh={loadRuns} onRerun={setQuery} />
      </div>
    </div>
  );
}
