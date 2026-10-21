"""Runs the shared benchmark through every RAG configuration, within a call budget.

    python scripts/run_benchmark.py --plan              # show the call budget, run nothing
    python scripts/run_benchmark.py                     # all 22 questions, all 4 configs
    python scripts/run_benchmark.py --questions 1-5     # a slice (1-based, inclusive)
    python scripts/run_benchmark.py --report-only       # rebuild the report from saved results

Built for the Gemini free tier, whose limits are per model, per day. Before
anything runs it prints the requests it will make per model and refuses if
either the generation or the judge model would exceed --max-calls-per-model.
Results accumulate in docs/benchmarks/results.json: a question already
scored for a configuration is skipped, so an interrupted or quota-limited run
continues where it stopped, on the same day or the next.

Each question costs, per configuration: 1 generation call (generation
model), 1 judge call (judge model) and 1 query embedding. The benchmark PDF
must already be uploaded (Documents page, or POST /documents/upload/).

Writes to docs/benchmarks/:
    results.json   per-question results from the API, merged across runs
    results.md     tables per configuration and per question type, with coverage
    *.png          charts (needs matplotlib; skipped with a notice otherwise)

EVAL_API_KEY is read from the environment or the repository's .env file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "backend" / "benchmarks" / "ragops_benchmark.json"
DEFAULT_OUT = ROOT / "docs" / "benchmarks"
CONFIGS = ["baseline", "dense", "hybrid", "hybrid_reranker"]
LABELS = {
    "baseline": "Baseline RAG",
    "dense": "Dense RAG",
    "hybrid": "Hybrid RAG (Dense + BM25)",
    "hybrid_reranker": "Hybrid + Cross-Encoder Reranker",
}

GENERATION = [
    ("faithfulness", "Faithfulness"),
    ("answer_relevance", "Answer relevance"),
    ("context_relevance", "Context relevance"),
    ("citation_accuracy", "Citation accuracy"),
    ("overall_score", "Overall (judge)"),
]
RETRIEVAL = [
    ("recall_at_k", "Recall@K"),
    ("precision_at_k", "Precision@K"),
    ("reciprocal_rank", "MRR"),
    ("ndcg_at_k", "nDCG@K"),
]


def read_api_key() -> str:
    key = os.environ.get("EVAL_API_KEY")
    if key:
        return key
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "EVAL_API_KEY":
                return value.strip().strip('"').strip("'")
    sys.exit("EVAL_API_KEY is not set in the environment or .env")


def parse_range(spec: str | None, total: int) -> list[int]:
    """'3-7' -> [2..6] as 0-based indices; None -> all."""
    if not spec:
        return list(range(total))
    start, _, end = spec.partition("-")
    first, last = int(start), int(end or start)
    if not 1 <= first <= last <= total:
        sys.exit(f"--questions must be within 1-{total}")
    return list(range(first - 1, last))


def load_results(path: Path, dataset_name: str) -> dict:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("dataset_name") == dataset_name:
            return data
        print(f"{path.name} holds a different dataset ({data.get('dataset_name')}); starting fresh.")
    return {"dataset_name": dataset_name, "results": {}, "runs": []}


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def plan_calls(todo: dict[str, list[dict]]) -> dict[str, int]:
    n = sum(len(qs) for qs in todo.values())
    # The backend caches query embeddings, so each distinct question is
    # embedded once however many configurations ask it.
    distinct = len({q["question"] for qs in todo.values() for q in qs})
    return {"generation": n, "judge": n, "embedding": distinct}


def run(args, dataset: dict, data: dict) -> None:
    questions = dataset["questions"]
    selected = [questions[i] for i in parse_range(args.questions, len(questions))]
    models = requests.get(f"{args.api}/api/eval/models", timeout=30).json()

    # Skip what earlier runs already scored.
    todo = {
        config: [q for q in selected if q["question"] not in data["results"].get(config, {})]
        for config in args.configs
    }
    calls = plan_calls(todo)
    print(f"{dataset['dataset_name']}: {len(selected)} questions selected x {len(args.configs)} configurations")
    for config in args.configs:
        print(f"  {config:16} {len(todo[config]):3} to run, {len(selected) - len(todo[config])} already scored")
    print("Planned Gemini requests:")
    print(f"  generation  {calls['generation']:4}  on {models['generation']}")
    print(f"  judge       {calls['judge']:4}  on {models['judge']}")
    print(f"  embedding   {calls['embedding']:4}  query embeddings")

    over = [m for m in ("generation", "judge") if calls[m] > args.max_calls_per_model]
    if over:
        sys.exit(
            f"Refusing to run: {', '.join(over)} would exceed --max-calls-per-model="
            f"{args.max_calls_per_model}. Run a smaller --questions slice, or raise the limit "
            "only if the model's free-tier daily quota allows it."
        )
    if args.plan or calls["generation"] == 0:
        print("Nothing to run." if calls["generation"] == 0 else "Plan only; nothing was run.")
        return

    headers = {"X-API-Key": read_api_key()}
    data["models"] = models
    for config in args.configs:
        if not todo[config]:
            continue
        # Small batches: the server saves a run only when all its questions are
        # scored, so one failure late in a 21-question request would discard
        # every request already spent on it. A batch loses at most a few.
        for start in range(0, len(todo[config]), args.batch_size):
            batch = todo[config][start:start + args.batch_size]
            started = time.perf_counter()
            print(f"  {config:16} questions {start + 1}-{start + len(batch)} of {len(todo[config])}...",
                  end="", flush=True)
            try:
                resp = requests.post(
                    f"{args.api}/api/eval/run",
                    json={"dataset_name": dataset["dataset_name"], "questions": batch, "rag_configs": [config]},
                    headers=headers,
                    timeout=args.timeout,
                )
            except requests.Timeout:
                print(f" no response after {args.timeout}s; stopping. Rerun to continue.")
                return
            elapsed = time.perf_counter() - started
            if resp.status_code != 201:
                print(f" FAILED after {elapsed:.0f}s: {resp.status_code} {resp.text[:300]}")
                if resp.status_code == 503:
                    print("  The free-tier quota is likely spent; rerun later to continue from here.")
                    return
                break
            summary = resp.json()["runs"][0]
            detail = requests.get(f"{args.api}/api/eval/results/{summary['run_id']}", timeout=60).json()
            bucket = data["results"].setdefault(config, {})
            for row in detail["results"]:
                bucket[row["question"]] = row
            data["runs"].append({"run_id": summary["run_id"], "rag_config": config,
                                 "questions": len(detail["results"]), "created_at": summary["created_at"]})
            save(data, args)
            print(f" done in {elapsed:.0f}s")


def save(data: dict, args) -> None:
    data["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def summarise(data: dict, types: dict) -> list[dict]:
    """Per-configuration means over every question scored so far."""
    rows = []
    for config in CONFIGS:
        results = list(data["results"].get(config, {}).values())
        if not results:
            continue
        row = {"rag_config": config, "label": LABELS[config], "n": len(results)}
        for key, _ in GENERATION:
            values = [r[key] if key == "overall_score" else r[key]["score"] for r in results]
            row[key] = sum(values) / len(values)
        labelled = [r["retrieval"] for r in results if r.get("retrieval")]
        for key, _ in RETRIEVAL:
            row[key] = sum(x[key] for x in labelled) / len(labelled) if labelled else None
        row["k"] = labelled[0]["k"] if labelled else None
        latencies = [r["pipeline_latency_ms"] for r in results if r.get("pipeline_latency_ms") is not None]
        row["latency_ms"] = sum(latencies) / len(latencies) if latencies else None
        by_type: dict[str, list[dict]] = {}
        for r in results:
            by_type.setdefault(types.get(r["question"]) or "untyped", []).append(r)
        row["by_type"] = {
            t: {"n": len(rs), "overall_score": sum(r["overall_score"] for r in rs) / len(rs)}
            for t, rs in by_type.items()
        }
        rows.append(row)
    return rows


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def marked(rows, key, lower_is_better=False):
    values = [r[key] for r in rows if r.get(key) is not None]
    best = (min if lower_is_better else max)(values) if values else None
    return {r["rag_config"]: f"**{fmt(r[key])}**" if r.get(key) is not None and r[key] == best else fmt(r.get(key))
            for r in rows}


def write_markdown(data: dict, rows: list[dict], total: int, out: Path) -> None:
    k = next((r["k"] for r in rows if r["k"]), "K")
    models = data.get("models", {})
    covered = min((r["n"] for r in rows), default=0)
    lines = [
        f"# Benchmark results: {data['dataset_name']}",
        "",
        f"**Coverage:** {covered} of {total} questions scored for every configuration"
        + ("" if covered == total else " (partial: the run continues as free-tier quota allows)")
        + f". Last updated {data.get('updated_at', '—')}.",
        "",
        f"Generation model `{models.get('generation', '?')}`; judge `{models.get('judge', '?')}` "
        "(one call scores all four metrics, 0-1). Retrieval metrics at "
        f"K={k} against page labels. Latency is the RAG pipeline (retrieval + generation), excluding "
        "judging. Best value per column in bold.",
        "",
        "## Generation quality and latency",
        "",
        "| Configuration | n | " + " | ".join(label for _, label in GENERATION) + " | Avg latency |",
        "|---|---:|" + "---:|" * (len(GENERATION) + 1),
    ]
    cols = {key: marked(rows, key) for key, _ in GENERATION}
    latency = marked(rows, "latency_ms", lower_is_better=True)
    for r in rows:
        cells = [cols[key][r["rag_config"]] for key, _ in GENERATION]
        lines.append(f"| {r['label']} | {r['n']} | " + " | ".join(cells) + f" | {latency[r['rag_config']]} ms |")

    lines += [
        "",
        f"## Retrieval quality (the same questions, K={k})",
        "",
        "| Configuration | " + " | ".join(label.replace("K", str(k)) for _, label in RETRIEVAL) + " |",
        "|---|" + "---:|" * len(RETRIEVAL),
    ]
    cols = {key: marked(rows, key) for key, _ in RETRIEVAL}
    for r in rows:
        lines.append(f"| {r['label']} | " + " | ".join(cols[key][r["rag_config"]] for key, _ in RETRIEVAL) + " |")

    qtypes = [t for t in ("factual", "keyword", "paraphrase", "untyped") if any(t in r["by_type"] for r in rows)]
    if qtypes:
        lines += [
            "",
            "## Overall judge score by question type",
            "",
            "| Question type | " + " | ".join(r["label"] for r in rows) + " |",
            "|---|" + "---:|" * len(rows),
        ]
        for t in qtypes:
            cells = [f"{fmt(r['by_type'].get(t, {}).get('overall_score'))} (n={r['by_type'].get(t, {}).get('n', 0)})"
                     for r in rows]
            lines.append(f"| {t} | " + " | ".join(cells) + " |")

    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_charts(rows: list[dict], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping charts (pip install -r scripts/requirements.txt)")
        return

    colors = {"baseline": "#8FA3C4", "dense": "#6FB1E0", "hybrid": "#B18FE0", "hybrid_reranker": "#3FBF94"}

    def grouped(metrics, title, filename):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        width = 0.8 / len(rows)
        for i, r in enumerate(rows):
            xs = [m + (i - (len(rows) - 1) / 2) * width for m in range(len(metrics))]
            ax.bar(xs, [r.get(key) or 0 for key, _ in metrics], width, label=r["label"],
                   color=colors.get(r["rag_config"], "#999999"))
        ax.set_xticks(range(len(metrics)), [label for _, label in metrics])
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)

    grouped(GENERATION, "Generation quality (LLM judge)", "generation_metrics.png")
    grouped(RETRIEVAL, "Retrieval quality against page labels", "retrieval_metrics.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for r in rows:
        if r.get("latency_ms") is None:
            continue
        ax.scatter(r["latency_ms"], r["overall_score"], s=90, color=colors.get(r["rag_config"]))
        ax.annotate(r["label"], (r["latency_ms"], r["overall_score"]), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.set_xlabel("Avg pipeline latency (ms)")
    ax.set_ylabel("Overall judge score")
    ax.set_title("Quality vs latency")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "quality_vs_latency.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--configs", nargs="+", default=CONFIGS, choices=CONFIGS)
    parser.add_argument("--questions", help="1-based inclusive range, e.g. 1-5. Default: all.")
    parser.add_argument("--max-calls-per-model", type=int, default=100,
                        help="Refuse if the generation or judge model would need more requests than this.")
    parser.add_argument("--plan", action="store_true", help="Print the call budget and exit.")
    parser.add_argument("--fresh", action="store_true", help="Discard saved results and start over.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--batch-size", type=int, default=3,
                        help="Questions per request; results are saved after each batch.")
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds per batch request.")
    parser.add_argument("--report-only", action="store_true", help="Rebuild report and charts from results.json.")
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    out = Path(args.out)
    data = {"dataset_name": dataset["dataset_name"], "results": {}, "runs": []} if args.fresh \
        else load_results(out / "results.json", dataset["dataset_name"])

    if not args.report_only:
        run(args, dataset, data)
        if args.plan:
            return

    if not data["results"]:
        sys.exit("No results yet; nothing to report.")
    types = {q["question"]: q.get("type") for q in dataset["questions"]}
    rows = summarise(data, types)
    out.mkdir(parents=True, exist_ok=True)
    write_markdown(data, rows, len(dataset["questions"]), out)
    write_charts(rows, out)
    print(f"Wrote {out / 'results.md'}")


if __name__ == "__main__":
    main()
