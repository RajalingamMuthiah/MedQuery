# Deploying MedQuery

MedQuery ships as one Docker image that serves the API and the web app on one port. The [`Dockerfile`](../Dockerfile) builds corpus v2 inside the image: it downloads the 10 public-domain PDFs, checks each against its sha256, builds the corpus and caches the embeddings, so a host only needs:

- **the image:** built from this repository;
- **one secret:** `GROQ_API_KEY`, from the [Groq console](https://console.groq.com);
- **optional variables:** listed below.

Measured locally (Docker 29, 2026-10-26):

| Measure | Value |
|---|---|
| Image size | 818 MB |
| Memory when idle, and after answering | about 296 MiB |
| From container start to `/health` reporting ready | about 6 s |
| Model download | only the 65 MB quantised bge-small embedding model (the live app uses no reranker) |

## Run it yourself

```powershell
Copy-Item .env.example .env      # then set GROQ_API_KEY
docker compose up                # first build: several minutes (downloads, OCR, embeddings)
# open http://localhost:8000
```

The container health check passes only when `/health` reports `ready: true`.

**The corpus inside the image differs slightly from the evaluated one.** It is built with Debian's Tesseract, so the two documents that contain scanned pages (`tb-reported-1964` and `tb-qa-2021`) can OCR slightly differently from the corpus that was evaluated (corpus v2 sha256 `07d7860f…`). The other eight documents are byte-identical. The build prints a warning when the sha differs, and `/health` shows the sha. Reported results refer to the evaluated corpus.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | (required) | Generation and the optional answer check. Set it as a **secret**; it is never baked into the image, logged or returned. |
| `MEDQUERY_DAILY_CAP` | `60` | Groq-calling requests per day, shared by all users. Resets at 00:00 UTC. Cached answers don't count. |
| `MEDQUERY_RATE_PER_MIN` | `6` | Requests per minute per client IP. |
| `MEDQUERY_TRUSTED_PROXIES` | (empty) | The host's proxy IPs or CIDR ranges, comma-separated (see below). |
| `MEDQUERY_CORS_ORIGIN` | `http://localhost:8000` | Only matters if the frontend is served from another origin. |
| `MEDQUERY_PORT` | `8000` | The port the server listens on. |

### Behind a hosting platform's proxy

Every hosted request arrives from the platform's proxy. If `MEDQUERY_TRUSTED_PROXIES` is not set, all visitors share one per-IP limit.

Set it to the proxy's network, for example `10.0.0.0/8,172.16.0.0/12,192.168.0.0/16` for platforms whose proxies use private addresses. The app then reads the visitor's address from `X-Forwarded-For`, but only for requests coming from those addresses. From anyone else the header is ignored, so it can't be spoofed.

To check it works, ask more than 6 questions within a minute from two different networks: only the second network's 7th request should be refused.

### When the daily cap is reached

Once 60 Groq-calling requests have been made in a UTC day, `/ask` and `/verify` return HTTP 429 and the app shows "Daily demo limit reached — try again tomorrow". Cached questions still answer.

Free hosts don't keep the container's disk. A restart or redeploy therefore resets the counter (`data/api_usage.json`) and the answer cache. Groq's own free-tier quota, 200K tokens a day for `openai/gpt-oss-20b`, remains the hard backstop.

## Choosing a free host

The free tiers below were checked on 2026-10-26 and change often. Confirm them on each provider's pricing page before you deploy.

### Recommended: Hugging Face Spaces (Docker)

- **Why:** CPU Basic hardware is 2 vCPU and 16 GB RAM, far more than the 296 MiB MedQuery needs. Spaces build the Dockerfile themselves and give you an HTTPS URL.
- **Check first:** Hugging Face's current docs say a Space running a Docker container needs a paid plan, while older docs and guides describe CPU Basic as free. Check your account before you start.
- **Cold start:** a Space on free hardware sleeps after 48 hours without visits. The next visit starts it again; the image is already built, and the app needs about 6 s to load. Meanwhile the page shows "Starting up…".

Steps (you create the account and the Space; nothing here creates either):

1. Create a new Space with the **Docker** SDK and CPU Basic hardware.
2. Give the Space this repository's files. Either push them to the Space's git remote, or mirror the GitHub repository. The Space's `README.md` must start with this front matter, which a Space requires:
   ```yaml
   ---
   title: MedQuery
   sdk: docker
   app_port: 8000
   ---
   ```
   Add it in the Space's copy only, or on a deploy branch, so the GitHub README stays clean.
3. Under **Settings → Variables and secrets**:
   - add the **secret** `GROQ_API_KEY`;
   - add the variable `MEDQUERY_TRUSTED_PROXIES`, using the private ranges above, then test it as described;
   - optionally, add `MEDQUERY_DAILY_CAP`.
4. Wait for the build (several minutes: PDF downloads, OCR and embeddings), then open the Space URL. `/health` should report `ready: true`.

### Fallback: Render (free web service)

- **Limits:** 512 MB RAM and 0.1 CPU. MedQuery's measured 296 MiB fits, but 0.1 CPU makes model loading and retrieval noticeably slower.
- **Spin-down:** free instances stop after 15 minutes without traffic and take about a minute to wake, plus MedQuery's own load time. There are 750 free instance hours a month.

Steps:

1. Create a new **Web Service** from the GitHub repository, with runtime **Docker**.
2. Add the environment variables:
   - `GROQ_API_KEY`, as a secret;
   - `MEDQUERY_PORT`, set to Render's port (`10000` by default) or set Render's `PORT` to `8000`;
   - `MEDQUERY_TRUSTED_PROXIES`.
3. Set the health check path to `/health`.

### Not recommended: Fly.io

Fly.io has no free allowance for new accounts (usage-based billing), so it doesn't meet the "free" requirement. The image runs there unchanged if paying is acceptable.

## Sources (checked 2026-10-26)

- Hugging Face Spaces overview: <https://huggingface.co/docs/hub/spaces-overview>
- Hugging Face Space runtime (sleep after 48 h on free CPU Basic): <https://huggingface.co/docs/huggingface_hub/package_reference/space_runtime.md>
- Render free tier and spin-down: <https://render.com/articles/platforms-with-a-real-free-tier-for-developers-in-2026.md>
- Render pricing and the 15-minute spin-down: <https://livemy.app/blog/render-pricing>
