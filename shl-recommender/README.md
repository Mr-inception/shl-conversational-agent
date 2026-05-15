# SHL Assessment Recommender

Production-style FastAPI service that scrapes the SHL Individual Test Solutions catalog, embeds it with sentence-transformers and FAISS, and uses Google Gemini (`gemini-2.5-flash` by default) for a constrained recommendation assistant.

## Setup

```bash
cd shl-recommender
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env
# set GEMINI_API_KEY in .env (optional: GEMINI_MODEL=gemini-2.5-flash)
```

On first startup, if `catalog.json` is missing the app runs the scraper (slow: many product pages). If `faiss_index.bin` / `faiss_meta.json` are missing, the FAISS index is built from the catalog.

To skip automatic catalog/index creation (for tests): set `SHL_SKIP_STARTUP_ENSURE=1`.

Optional: `SHL_CATALOG_PATH` points to a custom `catalog.json`.

## Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- `GET /health` — liveness
- `POST /chat` — body `{"messages":[{"role":"user","content":"..."}]}`  
  Response: `{"reply":"...","recommendations":[],"end_of_conversation":false}`

## Render

`Procfile` is included for Render (`web: uvicorn main:app --host 0.0.0.0 --port $PORT`). Set `GEMINI_API_KEY` in the dashboard. Cold starts may time out if the catalog must be scraped from scratch; pre-build `catalog.json` and FAISS artifacts or run the scraper offline and commit or upload them.

## Manual scrape

```bash
python scraper.py
```

## Tests

```bash
python test_agent.py
```
