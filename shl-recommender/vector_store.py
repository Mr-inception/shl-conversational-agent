from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_DIR = Path(__file__).resolve().parent
_CATALOG_PATH = Path(os.environ.get("SHL_CATALOG_PATH", str(_DIR / "catalog.json")))
_INDEX_PATH = _DIR / "faiss_index.bin"
_META_PATH = _DIR / "faiss_meta.json"

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_meta: list[dict[str, Any]] | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _chunk_text(item: dict[str, Any]) -> str:
    name = item.get("name") or ""
    ttype = item.get("test_type") or ""
    desc = item.get("description") or ""
    return f"Name: {name}. Type: {ttype}. Description: {desc}"


def build_from_catalog(
    catalog_path: Path | None = None,
    index_path: Path | None = None,
    meta_path: Path | None = None,
) -> None:
    global _index, _meta
    cpath = catalog_path or _CATALOG_PATH
    ipath = index_path or _INDEX_PATH
    mpath = meta_path or _META_PATH

    raw = json.loads(cpath.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("catalog.json must contain a list")

    model = _get_model()
    texts = [_chunk_text(x) for x in raw]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(ipath))
    mpath.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    _index = index
    _meta = raw


def load_index(
    index_path: Path | None = None,
    meta_path: Path | None = None,
) -> None:
    global _index, _meta
    ipath = index_path or _INDEX_PATH
    mpath = meta_path or _META_PATH
    if not ipath.is_file() or not mpath.is_file():
        raise FileNotFoundError("FAISS index or metadata missing; run build_from_catalog first")
    _index = faiss.read_index(str(ipath))
    _meta = json.loads(mpath.read_text(encoding="utf-8"))


def _ensure_loaded() -> None:
    global _index, _meta
    if _index is None or _meta is None:
        load_index()


def search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    _ensure_loaded()
    assert _index is not None and _meta is not None
    model = _get_model()
    q = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    q = np.asarray(q, dtype=np.float32)
    faiss.normalize_L2(q)
    k = min(top_k, len(_meta))
    if k <= 0:
        return []
    scores, idxs = _index.search(q, k)
    out: list[dict[str, Any]] = []
    for score, i in zip(scores[0], idxs[0]):
        if i < 0 or i >= len(_meta):
            continue
        row = dict(_meta[i])
        row["_score"] = float(score)
        out.append(row)
    return out


def get_catalog_list() -> list[dict[str, Any]]:
    _ensure_loaded()
    assert _meta is not None
    return list(_meta)


def allowed_urls() -> set[str]:
    data = json.loads((_CATALOG_PATH).read_text(encoding="utf-8"))
    return {x.get("url", "") for x in data if isinstance(x, dict)}
