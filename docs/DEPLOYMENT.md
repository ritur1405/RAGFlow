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
1.3 GB, and the reranker runs on CPU. Install Docker Engine with the compose
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
- Set `POSTGRES_PASSWORD` in `.env` before the first start (URL-safe
  characters only). Compose passes it to both Postgres and the backend. It
  only applies when the database volume is created; to change it later, run
  `docker compose exec db psql -U postgres -c "ALTER USER postgres PASSWORD 'new'"`,
  then update `.env` and `docker compose up -d`.

## 3. Ports

Only the web app is public by default; no compose edits are needed:

| Service | Published on | Why |
|---|---|---|
| `frontend` | `0.0.0.0:${WEB_PORT:-5173}` | The site. nginx in it proxies `/api`, `/query`, `/upload` and `/documents` to the backend |
| `backend` | `127.0.0.1:8000` | Swagger docs and `scripts/` on the server itself; never reachable from the network |
| `db` | `127.0.0.1:5432` | `psql` and backups on the server itself |

Set `WEB_PORT=80` in `.env` when nothing else serves port 80, or keep the
default behind a TLS reverse proxy (step 5). Every service restarts
automatically after a crash or a reboot (`restart: unless-stopped`).

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
