import { Fragment, useState } from "react";
import { fmtLatency, postJSON } from "../lib/api.js";

/**
 * Ask.jsx
 * -------
 * Week 3 — ask a question and get a grounded answer with its sources.
 *
 * Consumes:
 *   POST /query/   -> { question, answer, sources[] }
 *
 * The answer cites its context as [Source N | file | Page p | Chunk c]. Those
 * markers become links to the matching source card, so every claim can be
 * traced to the chunk it came from.
 */

const panel = "rounded-md border border-[#2A3B36] bg-[#16211E]";

const EXAMPLES = [
  "What k and b parameters were used for Anserini BM25?",
  "Why can a keyword-based search engine miss relevant documents?",
  "Which retrieval architecture achieved the best zero-shot performance?",
];

// Matches the citation markers the generator is prompted to emit.
const CITATION = /\[Source (\d+)[^\]]*\]/g;

function AnswerText({ text, onCite }) {
  const parts = [];
  let last = 0;
  for (const match of text.matchAll(CITATION)) {
    parts.push(text.slice(last, match.index));
    const n = Number(match[1]);
    parts.push(
      <button
        key={`${match.index}-${n}`}
        type="button"
        onClick={() => onCite(n)}
        title={match[0].slice(1, -1)}
        className="mx-0.5 inline-flex rounded bg-[#22302C] px-1.5 align-baseline font-mono text-[11px] text-[#5FE0B8] hover:bg-[#2A3B36]"
      >
        {n}
      </button>,
    );
    last = match.index + match[0].length;
  }
  parts.push(text.slice(last));
  return (
    <p className="whitespace-pre-wrap text-[14px] leading-relaxed text-[#E8EDE9]">
      {parts.map((p, i) => (
        <Fragment key={i}>{p}</Fragment>
      ))}
    </p>
  );
}

function SourceCard({ source, index, highlighted }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <li
      id={`source-${index}`}
      className={`rounded border bg-[#0F1614] p-3 transition-colors ${
        highlighted ? "border-[#5FE0B8]" : "border-[#2A3B36]"
      }`}
    >
      <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2 text-[11px]">
        <span className="font-mono text-[#8FA39B]">
          <span className="font-semibold text-[#5FE0B8]">Source {index}</span>
          {source.file_name ? ` · ${source.file_name}` : ""}
          {source.page_number != null ? ` · p.${source.page_number}` : ""}
          {` · chunk ${source.id}`}
        </span>
        {source.score != null && (
          <span className="font-mono text-[#5F726B]" title="L2 distance; lower is closer">
            distance {source.score.toFixed(3)}
          </span>
        )}
      </div>
      <p className={`text-[12.5px] leading-relaxed text-[#C9D4CF] ${expanded ? "" : "line-clamp-4"}`}>
        {source.content}
      </p>
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="mt-2 text-[11px] text-[#8FA39B] hover:text-[#E8EDE9]"
      >
        {expanded ? "Show less" : "Show more"}
      </button>
    </li>
  );
}

export default function Ask() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [elapsed, setElapsed] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [highlighted, setHighlighted] = useState(null);

  const valid = question.trim().length >= 3;

  const ask = async (e) => {
    e?.preventDefault();
    if (!valid || loading) return;
    setLoading(true);
    setError(null);
    setHighlighted(null);
    const start = performance.now();
    try {
      setResult(await postJSON("/query/", { question: question.trim() }));
      setElapsed(performance.now() - start);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const cite = (n) => {
    setHighlighted(n);
    document.getElementById(`source-${n}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  return (
    <div className="min-h-screen bg-[#0F1614] px-6 py-8 md:px-10">
      <div className="mx-auto max-w-4xl">
        <header className="mb-8">
          <p className="mb-1 text-[13px] text-[#5F726B]">RAGOps · Ask</p>
          <h1 className="text-2xl text-[#E8EDE9]">Ask the knowledge base</h1>
          <p className="mt-1 text-[13px] text-[#8FA39B]">
            Dense retrieval finds the closest chunks, and the answer is generated only from them, with a citation for
            each claim. If nothing relevant is found, it says so instead of guessing.
          </p>
        </header>

        <form onSubmit={ask} className={`${panel} mb-6 p-5`}>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) ask(e);
            }}
            rows={3}
            maxLength={1000}
            placeholder="Ask something your uploaded documents can answer…"
            aria-label="Question"
            className="w-full resize-y rounded border border-[#2A3B36] bg-[#0F1614] px-3 py-2 text-[14px] text-[#E8EDE9] placeholder-[#5F726B] focus:border-[#5FE0B8] focus:outline-none"
          />
          <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  onClick={() => setQuestion(ex)}
                  className="rounded-full border border-[#2A3B36] px-3 py-1 text-[11px] text-[#8FA39B] hover:text-[#E8EDE9]"
                >
                  {ex}
                </button>
              ))}
            </div>
            <button
              type="submit"
              disabled={!valid || loading}
              className="rounded bg-[#5FE0B8] px-5 py-2 text-[13px] font-semibold text-[#0F1614] hover:bg-[#7CE0B8] disabled:opacity-40"
            >
              {loading ? "Thinking…" : "Ask"}
            </button>
          </div>
        </form>

        {error && (
          <div className="mb-6 rounded-md border border-[#5C2F2F] bg-[#281616] px-4 py-3 text-[13px] text-[#E37B6B]">
            {error}
          </div>
        )}

        {result && (
          <div className="space-y-6">
            <section className={`${panel} p-5`}>
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-[13px] text-[#8FA39B]">Answer</h2>
                <span className="font-mono text-[11px] text-[#5F726B]">{fmtLatency(elapsed)}</span>
              </div>
              <AnswerText text={result.answer} onCite={cite} />
            </section>

            <section>
              <h2 className="mb-3 text-[13px] text-[#8FA39B]">
                {result.sources.length} source{result.sources.length === 1 ? "" : "s"} used as context
              </h2>
              {result.sources.length === 0 ? (
                <p className="text-[12px] text-[#5F726B]">
                  No chunk was close enough to the question, so no context was passed to the model.
                </p>
              ) : (
                <ol className="space-y-2">
                  {result.sources.map((source, i) => (
                    <SourceCard key={source.id} source={source} index={i + 1} highlighted={highlighted === i + 1} />
                  ))}
                </ol>
              )}
            </section>
          </div>
        )}
      </div>
    </div>
  );
}
