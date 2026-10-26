Phase 8: packaging, deployment and the final README. Plan only; no code until I approve. No changes to evaluation code, golden sets, frozen labels or committed reports.

1. Docker
- One image (or a small compose) that serves the backend and the built frontend together (e.g. FastAPI serving frontend/dist, or nginx in front), so the app runs with one command: docker compose up.
- The image builds corpus v2 at build time from the manifest's public URLs (sha256-checked), or documents clearly how to mount data/; embeddings cached in the image so startup is fast. No secrets in the image: GROQ_API_KEY only via environment.
- Small and reproducible: pinned base image, uv lock, npm ci; .dockerignore excludes data/, .env, node_modules.

2. Deployment-ready API
- Trusted proxy (Phase 8 carry-forward): read the client IP from X-Forwarded-For only when the request comes from a configured trusted proxy (MEDQUERY_TRUSTED_PROXIES); otherwise request.client.host. Tests.
- Daily cap and rate limits stay on; CORS set from env; /health used as the container health check.

3. Public demo (recommend a free host and explain trade-offs)
- Propose the best free option for a CPU-only FastAPI + static frontend with ~1 GB of models (e.g. Hugging Face Spaces with Docker, Render, Fly.io), how secrets are set there, cold-start behaviour, and what happens when the daily cap is reached. Write deployment steps in docs/DEPLOY.md. I will do the account/secret steps myself; do not create accounts.

4. Final README
- Top: one-paragraph pitch, demo link placeholder, a short GIF/screenshot placeholder, the headline test results (from the reports), and "Not medical advice".
- Architecture diagram (Mermaid): ingestion -> corpus v2 -> hybrid retrieval -> answer_v2 -> citations -> optional verify; evaluation harness alongside.
- Quick start (docker compose up), local dev, evaluation commands, project structure, limitations, licence and data sources.
- Ask me which licence to use for the code (suggest MIT) before adding a LICENSE file.

5. Rules: tests for the trusted-proxy logic and any new code; CI builds the Docker image (no push to a registry); CLAUDE.md; commits, push, CI green, summary, STOP.

Ask me up to 3 questions if anything conflicts with the repo.
