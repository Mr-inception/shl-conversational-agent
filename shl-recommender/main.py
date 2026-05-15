from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from agent import run_agent
from models import ChatRequest, ChatResponse

_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("shl-recommender")


def ensure_data() -> None:
    catalog_path = _ROOT / "catalog.json"
    if not catalog_path.is_file():
        logger.warning("catalog.json missing — scraping SHL catalog (this can take 30+ minutes)")
        from scraper import run_scraper

        run_scraper(catalog_path)
        import agent as agent_mod

        agent_mod._load_catalog.cache_clear()
        logger.info("catalog.json written")

    index_path = _ROOT / "faiss_index.bin"
    meta_path = _ROOT / "faiss_meta.json"
    if not index_path.is_file() or not meta_path.is_file():
        logger.warning(
            "FAISS index missing — building embeddings for catalog "
            "(first run downloads the model and may take several minutes on CPU)"
        )
        import vector_store

        vector_store.build_from_catalog(catalog_path=catalog_path)
        logger.info("FAISS index ready: %s", index_path.name)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(level=logging.INFO)
    if os.environ.get("SHL_SKIP_STARTUP_ENSURE") != "1":
        ensure_data()
    if not os.environ.get("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY is not set — /chat will return a fallback until you add it to .env")
    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest) -> ChatResponse:
    raw = [m.model_dump() for m in req.messages]
    reply, recommendations, end_of_conversation = run_agent(raw)
    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation,
    )
