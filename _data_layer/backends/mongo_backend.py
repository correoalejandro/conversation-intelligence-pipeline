# _data_layer/backends/mongo_backend.py
from __future__ import annotations

import os
from io import BytesIO
from typing import Any, Tuple
from datetime import datetime, timezone

import joblib
import pandas as pd
from pymongo import MongoClient
from gridfs import GridFS

# ---------- Config ----------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "clasificador_mensajes")
DATA_COLL = os.getenv("MONGO_DATA_COLLECTION", "artifacts_data")   # small payloads
FS_BUCKET = os.getenv("MONGO_FS_BUCKET", "fs")                     # GridFS bucket

# Save small JSON-ish payloads inline in a collection when possible.
# Everything else → GridFS (pickled via joblib).
INLINE_MAX_BYTES = int(os.getenv("MONGO_INLINE_MAX_BYTES", str(2_000_000)))  # ~2MB


def _client_and_fs() -> Tuple[MongoClient, GridFS]:
    client = MongoClient(MONGO_URI, connect=True)
    db = client[MONGO_DB]
    fs = GridFS(db, collection=FS_BUCKET)
    return client, fs


def _try_inline(obj: Any, art_id: str) -> str | None:
    """
    Try storing a small JSON-friendly payload inline.
    Returns data_ref if stored, else None.
    """
    # Only inline if clearly serializable & small (DataFrame is not)
    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        # approximate size via joblib to bytes for safety
        buf = BytesIO()
        joblib.dump(obj, buf)
        n = buf.getbuffer().nbytes
        if n <= INLINE_MAX_BYTES:
            client, _ = _client_and_fs()
            doc = {
                "_id": art_id,
                "kind": "inline",
                "payload": obj,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            client[MONGO_DB][DATA_COLL].replace_one({"_id": art_id}, doc, upsert=True)
            return f"mongo:{DATA_COLL}:{art_id}"
    return None


def save(obj: Any, art_id: str) -> str:
    """
    Persist 'obj' and return a data_ref string that api/registry will keep.
    Contract mirrors other backends used by save_artifact():contentReference[oaicite:0]{index=0}.
    """
    # 1) Inline path for tiny JSON-friendly payloads
    data_ref = _try_inline(obj, art_id)
    if data_ref:
        return data_ref

    # 2) Otherwise serialize to bytes via joblib → GridFS
    _, fs = _client_and_fs()
    buf = BytesIO()
    # Normalize pandas DataFrame to be pickle-stable
    if isinstance(obj, pd.DataFrame):
        # keep full fidelity; downstream loader handles normalization:contentReference[oaicite:1]{index=1}
        joblib.dump(obj, buf)
    else:
        joblib.dump(obj, buf)
    buf.seek(0)
    file_id = fs.put(buf, filename=f"{art_id}.pkl", encoding="binary")
    return f"gridfs:{str(file_id)}"


def load(data_ref: str) -> Any:
    """
    Reverse of save(): return the original Python object
    (often a DataFrame or dict), which api.load_artifact() then normalizes:contentReference[oaicite:2]{index=2}.
    """
    if data_ref.startswith("mongo:"):
        # Format: mongo:<collection>:<doc_id>
        _, coll, doc_id = data_ref.split(":", 2)
        client, _ = _client_and_fs()
        doc = client[MONGO_DB][coll].find_one({"_id": doc_id})
        if not doc:
            raise FileNotFoundError(f"Inline doc not found: {data_ref}")
        return doc.get("payload")

    if data_ref.startswith("gridfs:"):
        _, fs = _client_and_fs()
        _id = data_ref.split(":", 1)[1]
        fh = fs.get(_id)
        buf = BytesIO(fh.read())
        buf.seek(0)
        return joblib.load(buf)

    # Fallback (should not happen)
    raise ValueError(f"Unsupported data_ref for mongo backend: {data_ref}")
