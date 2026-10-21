# Deployment

RAGOps deploys as three containers from one `docker-compose.yml`:

| Service | Image | Role |
|---|---|---|
| `db` | `pgvector/pgvector:pg16` | PostgreSQL with the `vector` extension (enabled by `db/init/01-enable-pgvector.sql` on first start) |
| `backend` | built from `backend/Dockerfile` | FastAPI on port 8000, with the cross-encoder model baked in |
| `frontend` | built from `frontend/Dockerfile` | Static React build served by nginx, which also proxies the API paths to `backend` |

The same compose file works on a laptop and on any Linux VM with Docker. No
cloud-specific service is required.

## 1. Prepare a server

Any VM with about 2 vCPU and 4 GB RAM is enough: the backend image is roughly
1.4 GB, and the reranker runs on CPU. Install Docker Engine with the compose
plugin, then:

```bash
git clone https://github.com/ritur1405/RAGFlow.git ragops && cd ragops
cp .env.example .env
```

## 2. Configure secrets

Edit `.env`:

```bash
GEMINI_API_KEY=...          # required
EVAL_API_KEY=$(openssl rand -hex 32)   # required to run evaluations
```

- Never commit `.env`; it is in `.gitignore`.
- `EVAL_API_KEY` protects `POST /api/eval/run`, which spends Gemini credits.
  Without it, that endpoint refuses every call with 503.
- Read endpoints stay open so the dashboard can show results without a secret
  in the browser.
- For anything beyond a demo, also change the Postgres password in
  `docker-compose.yml` (`POSTGRES_PASSWORD` and the matching `DATABASE_URL`).

## 3. Expose only the web app

In `docker-compose.yml`, for a public server:

- remove the `ports:` mapping of `db` (5432), so Postgres is reachable only on
  the compose network
- optionally remove `backend`'s `8000:8000`, since nginx in `frontend` already
  proxies `/api`, `/query`, `/upload` and `/documents` to it
- change `frontend`'s mapping to `80:80`, or keep `5173:80` behind a reverse proxy

## 4. Build and start

```bash
docker compose up -d --build
docker compose ps            # db should be "healthy", all three "Up"
docker compose logs -f backend
```

The backend creates its tables at startup. It also adds any newer nullable
columns to an existing database, so pulling a new version does not require a
manual migration for those.

## 5. TLS

Put a reverse proxy with automatic certificates in front of the frontend
container, for example Caddy:

```
ragops.example.com {
    reverse_proxy localhost:5173
}
```

Evaluation requests are long-running (minutes, because each question makes
several judge calls). The bundled nginx allows 900s, so give the outer proxy a
similar read timeout.

## 6. Load the benchmark

1. Open the site and upload `backend/benchmarks/beir-2104.08663v4.pdf` on the
   Documents page.
2. On the Evaluation page, choose **Load shared benchmark**, enter the
   `EVAL_API_KEY`, and run it.

Or run it from any machine that can reach the API:

```bash
pip install -r scripts/requirements.txt
EVAL_API_KEY=... python scripts/run_benchmark.py --api https://ragops.example.com
```

## Operations

| Task | Command |
|---|---|
| Update | `git pull && docker compose up -d --build` |
| Logs | `docker compose logs -f backend` |
| Back up the database | `docker compose exec db pg_dump -U postgres ragflow > ragops-$(date +%F).sql` |
| Restore | `docker compose exec -T db psql -U postgres ragflow < ragops-YYYY-MM-DD.sql` |
| Reset everything (deletes all data) | `docker compose down -v` |

Data lives in the `pgdata` named volume, so it survives `down` and `up` but not
`down -v`.

## Rate limits

The defaults are chosen to run free on the Gemini free tier (see the README's
"Zero cost on the free tier"). Every Gemini call handles limits this way:

- **Transient server errors (5xx):** retried with exponential backoff from 2s
  up to 60s.
- **Per-minute rate limits (429):** retried after the server's suggested
  delay.
- **Both** stop after `GEMINI_RETRY_ATTEMPTS` (default 8) attempts.

Daily quotas cannot be retried away, so they fail at once. When one runs out:

- **Every endpoint** returns **503**, with a message naming the model and
  saying the limit resets daily. The Ask page shows that message instead of an
  answer.
- **Evaluation runs** are marked **failed** with the error recorded.
- **`scripts/run_benchmark.py`** stops and keeps its progress for the next
  run.

None of these reports misleading results.
