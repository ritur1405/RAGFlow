"""Runs the shared benchmark through every RAG configuration and records the results.

    python scripts/run_benchmark.py                 # all 22 questions, all 4 configs
    python scripts/run_benchmark.py --limit 2       # smoke test
    python scripts/run_benchmark.py --report-only   # rebuild the report from saved results

Each configuration is a separate POST /api/eval/run, so one failing
configuration does not discard the others, and progress is visible. The
benchmark PDF must already be uploaded (Documents page, or
POST /documents/upload/).

Writes to docs/benchmarks/:
    results.json   raw run summaries and per-question results from the API
    results.md     tables for the README and report, overall and per question type
    *.png          charts (needs matplotlib; skipped with a notice otherwise)

EVAL_API_KEY is read from the environment or the repository's .env file.
Every question costs one generation and four judge calls per configuration,
so the full run is roughly 22 x 4 x 5 = 440 Gemini calls.
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

GENERATION = [
    ("avg_faithfulness", "Faithfulness"),
    ("avg_answer_relevance", "Answer relevance"),
    ("avg_context_relevance", "Context relevance"),
    ("avg_citation_accuracy", "Citation accuracy"),
    ("avg_overall_score", "Overall (judge)"),
]
RETRIEVAL = [
    ("avg_recall_at_k", "Recall@K"),
    ("avg_precision_at_k", "Precision@K"),
    ("avg_mrr", "MRR"),
    ("avg_ndcg_at_k", "nDCG@K"),
]
PER_QUESTION = {
    "recall_at_k": "Recall@K",
    "precision_at_k": "Precision@K",
    "reciprocal_rank": "MRR",
    "ndcg_at_k": "nDCG@K",
}


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


def run(args) -> dict:
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    questions = dataset["questions"][: args.limit] if args.limit else dataset["questions"]
    name = dataset["dataset_name"] + (f"-first{args.limit}" if args.limit else "")
    headers = {"X-API-Key": read_api_key()}

    print(f"{name}: {len(questions)} questions x {len(args.configs)} configurations against {args.api}")
    runs, details = [], {}
    for config in args.configs:
        started = time.perf_counter()
        print(f"  {config:16} running...", end="", flush=True)
        resp = requests.post(
            f"{args.api}/api/eval/run",
            json={"dataset_name": name, "questions": questions, "rag_configs": [config]},
            headers=headers,
            timeout=args.timeout,
        )
        elapsed = time.perf_counter() - started
        if resp.status_code != 201:
            print(f" FAILED after {elapsed:.0f}s: {resp.status_code} {resp.text[:300]}")
            continue
        summary = resp.json()["runs"][0]
        runs.append(summary)
        details[config] = requests.get(
            f"{args.api}/api/eval/results/{summary['run_id']}", timeout=60
        ).json()["results"]
        print(f" done in {elapsed:.0f}s (overall {summary['avg_overall_score']:.3f})")

    return {
        "dataset_name": name,
        "num_questions": len(questions),
        "question_types": {q["question"]: q.get("type") for q in questions},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": runs,
        "results": details,
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def best_marked(rows, key, lower_is_better=False):
    values = [r[key] for r in rows if r.get(key) is not None]
    best = (min if lower_is_better else max)(values) if values else None
    return {
        r["rag_config"]: (f"**{fmt(r[key])}**" if r.get(key) is not None and r[key] == best else fmt(r.get(key)))
        for r in rows
    }


def by_type(data) -> dict:
    """Mean per-question retrieval metric for each configuration and question type."""
    types = data["question_types"]
    table = {}
    for config, results in data["results"].items():
        for r in results:
            qtype = types.get(r["question"]) or "untyped"
            if r.get("retrieval") is None:
                continue
            bucket = table.setdefault(qtype, {}).setdefault(config, [])
            bucket.append(r["retrieval"])
    out = {}
    for qtype, configs in table.items():
        out[qtype] = {
            config: {
                metric: sum(item[metric] for item in items) / len(items)
                for metric in PER_QUESTION
            }
            | {"n": len(items)}
            for config, items in configs.items()
        }
    return out


def write_markdown(data, out: Path) -> None:
    rows = data["runs"]
    labels = {r["rag_config"]: r["rag_config_label"] for r in rows}
    k = next((r["retrieval_k"] for r in rows if r.get("retrieval_k")), "K")
    lines = [
        f"# Benchmark results: {data['dataset_name']}",
        "",
        f"{data['num_questions']} questions, run {data['created_at']}. "
        f"Retrieval metrics at K={k} against page labels; generation metrics from the "
        "Gemini LLM judge (0-1). Latency is the RAG pipeline only (retrieval + generation), "
        "excluding judging. Best value per column in bold.",
        "",
        "## Retrieval quality",
        "",
        "| Configuration | " + " | ".join(label.replace("K", str(k)) for _, label in RETRIEVAL) + " |",
        "|---|" + "---:|" * len(RETRIEVAL),
    ]
    cols = {key: best_marked(rows, key) for key, _ in RETRIEVAL}
    for r in rows:
        lines.append(f"| {labels[r['rag_config']]} | " + " | ".join(cols[key][r["rag_config"]] for key, _ in RETRIEVAL) + " |")

    lines += [
        "",
        "## Generation quality and latency",
        "",
        "| Configuration | " + " | ".join(label for _, label in GENERATION) + " | Avg latency |",
        "|---|" + "---:|" * (len(GENERATION) + 1),
    ]
    cols = {key: best_marked(rows, key) for key, _ in GENERATION}
    latency = best_marked(rows, "avg_latency_ms", lower_is_better=True)
    for r in rows:
        cells = [cols[key][r["rag_config"]] for key, _ in GENERATION]
        ms = latency[r["rag_config"]]
        lines.append(f"| {labels[r['rag_config']]} | " + " | ".join(cells) + f" | {ms} ms |")

    typed = by_type(data)
    if typed:
        lines += [
            "",
            f"## nDCG@{k} by question type",
            "",
            "Factual questions restate the paper's wording; keyword questions hinge on exact names and "
            "numbers; paraphrased questions share little vocabulary with the text.",
            "",
            "| Question type | n | " + " | ".join(labels[c] for c in labels) + " |",
            "|---|---:|" + "---:|" * len(labels),
        ]
        for qtype in ("factual", "keyword", "paraphrase", "untyped"):
            if qtype not in typed:
                continue
            n = next(iter(typed[qtype].values()))["n"]
            cells = [fmt(typed[qtype].get(c, {}).get("ndcg_at_k")) for c in labels]
            lines.append(f"| {qtype} | {n} | " + " | ".join(cells) + " |")

    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_charts(data, out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping charts (pip install -r scripts/requirements.txt)")
        return

    rows = data["runs"]
    labels = [r["rag_config_label"] for r in rows]
    colors = {"baseline": "#8FA3C4", "dense": "#6FB1E0", "hybrid": "#B18FE0", "hybrid_reranker": "#3FBF94"}

    def grouped(metrics, title, filename):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        width = 0.8 / len(rows)
        for i, r in enumerate(rows):
            xs = [m + (i - (len(rows) - 1) / 2) * width for m in range(len(metrics))]
            ax.bar(xs, [r.get(key) or 0 for key, _ in metrics], width, label=r["rag_config_label"],
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

    grouped(RETRIEVAL, "Retrieval quality against page labels", "retrieval_metrics.png")
    grouped(GENERATION, "Generation quality (LLM judge)", "generation_metrics.png")

    # Quality vs latency: the trade-off the plan asks to explain.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for r, label in zip(rows, labels):
        if r.get("avg_latency_ms") is None or r.get("avg_ndcg_at_k") is None:
            continue
        ax.scatter(r["avg_latency_ms"], r["avg_ndcg_at_k"], s=90, color=colors.get(r["rag_config"]))
        ax.annotate(label, (r["avg_latency_ms"], r["avg_ndcg_at_k"]), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.set_xlabel("Avg pipeline latency (ms)")
    ax.set_ylabel("nDCG@K")
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
    parser.add_argument("--limit", type=int, help="Only the first N questions (smoke test).")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds per configuration.")
    parser.add_argument("--report-only", action="store_true", help="Rebuild report and charts from results.json.")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    else:
        data = run(args)
        if not data["runs"]:
            sys.exit("No configuration completed; nothing written.")
        (out / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    write_markdown(data, out)
    write_charts(data, out)
    print(f"Wrote {out / 'results.md'}")


if __name__ == "__main__":
    main()
