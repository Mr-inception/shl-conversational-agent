from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()

from models import ChatRequest, ChatResponse

_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("shl-recommender")

_data_ready = False
_data_error: str | None = None


def ensure_data() -> None:
    global _data_ready, _data_error
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
            "(downloads model; may take several minutes on CPU)"
        )
        import vector_store

        vector_store.build_from_catalog(catalog_path=catalog_path)
        logger.info("FAISS index ready: %s", index_path.name)

    _data_ready = True


def _ensure_data_safe() -> None:
    global _data_ready, _data_error
    try:
        if os.environ.get("SHL_SKIP_STARTUP_ENSURE") == "1":
            catalog = _ROOT / "catalog.json"
            index = _ROOT / "faiss_index.bin"
            meta = _ROOT / "faiss_meta.json"
            if catalog.is_file() and index.is_file() and meta.is_file():
                _data_ready = True
                logger.info("Using committed catalog.json and FAISS index")
            else:
                _data_error = (
                    "Missing catalog.json or FAISS files in the repo. "
                    "Commit them or unset SHL_SKIP_STARTUP_ENSURE."
                )
                logger.error(_data_error)
            return
        ensure_data()
    except Exception as exc:
        _data_error = str(exc)
        logger.exception("Startup data initialization failed: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(level=logging.INFO)

    async def _init() -> None:
        await asyncio.to_thread(_ensure_data_safe)

    init_task = asyncio.create_task(_init())

    if not os.environ.get("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY is not set — /chat will return a fallback")

    yield

    if not init_task.done():
        init_task.cancel()


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


@app.get("/health")
def health():
    body: dict = {"status": "ok"}
    if not _data_ready:
        body["initializing"] = True
    if _data_error:
        body["data_error"] = _data_error
    return body


@app.post("/chat")
def chat(req: ChatRequest) -> ChatResponse:
    if _data_error:
        raise HTTPException(
            status_code=503,
            detail="Search index failed to initialize. Check server logs.",
        )
    if not _data_ready:
        raise HTTPException(
            status_code=503,
            detail="Search index is still initializing. Retry in a minute.",
        )
    from agent import run_agent

    raw = [m.model_dump() for m in req.messages]
    reply, recommendations, end_of_conversation = run_agent(raw)
    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation,
    )
