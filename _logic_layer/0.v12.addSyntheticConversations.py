# ingest_messages_duckdb_v12.py
from __future__ import annotations

"""
Ingest Messages with DuckDB v12 — JSONL -> Registry (Interactive)

Flow:
  1) Pick a file from data/0_input/ (or pass --jsonl)
  2) Load & dedup with DuckDB
  3) Create batch via _data_layer.registry.register_batch(...)
  4) Register items via:
       - register_messages(batch_id, items) if available
       - else register_conversations(batch_id, items) as a safe fallback

Each item carries minimal normalized fields plus the full raw record.
"""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import locale
import os
import sys
from typing import Dict, List, Any, Tuple

# Console niceties (Windows, UTF-8)
try:
    locale.setlocale(locale.LC_TIME, "Spanish_Spain.1252")
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- Registry (original helpers) ----
try:
    from _data_layer.registry import register_batch, register_conversations
    try:
        # Optional: if you implemented it
        from _data_layer.registry import register_messages  # type: ignore
    except Exception:
        register_messages = None
except Exception:
    # Fallback to 'registry' module if present in your PYTHONPATH
    from registry import register_batch, register_conversations  # type: ignore
    try:
        from registry import register_messages  # type: ignore
    except Exception:
        register_messages = None

# ---- Dependencies ----
try:
    import duckdb
except Exception as e:
    raise RuntimeError("duckdb is required. Install with: python -m pip install duckdb") from e

INPUT_DIR = Path("data/0_input")
CHUNK_SIZE_DEFAULT = 10000

def human_ts_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

def _menu_choose_jsonl(input_dir: Path) -> Path:
    files = sorted(list(input_dir.glob("*.jsonl")) + list(input_dir.glob("*.ndjson")) + list(input_dir.glob("*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON/JSONL files found in {input_dir}")
    print("📂 Available input files:")
    print("———————————————")
    for i, p in enumerate(files):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"[{i}] {p.name}  ({size_mb:.2f} MB)")
    while True:
        try:
            idx = int(input("Select file #: "))
            if 0 <= idx < len(files):
                return files[idx]
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a number.")

def _build_dedup_relation(con: "duckdb.DuckDBPyConnection", jsonl_path: Path, dedup: bool) -> None:
    # Load as temp view
    con.execute("CREATE OR REPLACE TEMP VIEW src AS SELECT * FROM read_json_auto(?);", [str(jsonl_path)])
    # Normalize main fields for convenience
    con.execute("""
        CREATE OR REPLACE TEMP VIEW norm AS
        SELECT
            *,
            COALESCE(CAST(id AS VARCHAR), CAST(message_id AS VARCHAR), CAST(msg_id AS VARCHAR)) AS _id_raw,
            COALESCE(CAST(sender AS VARCHAR), CAST(author AS VARCHAR), CAST(role AS VARCHAR))   AS _sender_raw,
            COALESCE(CAST(timestamp AS VARCHAR), CAST(ts AS VARCHAR), CAST(time AS VARCHAR), CAST(created_at AS VARCHAR)) AS _ts_raw,
            COALESCE(CAST(text AS VARCHAR), CAST(content AS VARCHAR), CAST(body AS VARCHAR), CAST(message AS VARCHAR))    AS _text_raw
        FROM src;
    """)
    if not dedup:
        con.execute("CREATE OR REPLACE TEMP TABLE dedup AS SELECT * FROM norm;")
        return
    # Dedup key: id/message_id/msg_id OR signature(sender|timestamp|text[0:200])
    con.execute("""
        CREATE OR REPLACE TEMP TABLE dedup AS
        SELECT * EXCLUDE(_rn) FROM (
          SELECT
            *,
            ROW_NUMBER() OVER (
              PARTITION BY COALESCE(_id_raw, sha1(COALESCE(_sender_raw,'') || '|' || COALESCE(_ts_raw,'') || '|' || LEFT(COALESCE(_text_raw,''), 200)))
              ORDER BY 1
            ) AS _rn
          FROM norm
        )
        WHERE _rn = 1;
    """)

def _message_id(row: Dict[str, Any]) -> str:
    for k in ("id", "message_id", "msg_id", "_id_raw"):
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v)
    s = f"{row.get('_sender_raw','')}|{row.get('_ts_raw','')}|{str(row.get('_text_raw',''))[:200]}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _iter_dedup_rows(con: "duckdb.DuckDBPyConnection", chunk_size: int):
    total = con.execute("SELECT COUNT(*) FROM dedup;").fetchone()[0]
    offset = 0
    while offset < total:
        res = con.execute("SELECT * FROM dedup LIMIT ? OFFSET ?;", [chunk_size, offset])
        cols = [d[0] for d in res.description]
        rows = res.fetchall()
        for tup in rows:
            yield dict(zip(cols, tup))
        offset += chunk_size

def ingest_with_duckdb(jsonl_path: Path, tag: str | None, chunk_size: int, dedup: bool) -> Tuple[str, int]:
    con = duckdb.connect()
    _build_dedup_relation(con, jsonl_path, dedup=dedup)

    # Create batch via original helper
    params = {"format": "jsonl", "dedup": bool(dedup)}
    if tag:
        params["tag"] = tag
    batch_id = register_batch(data_ref=str(jsonl_path), parameters=params)
    print(f"✅ Registered batch: {batch_id}")

    total_registered = 0
    buffer: List[Dict[str, Any]] = []

    def flush_buffer(buf: List[Dict[str, Any]]):
        nonlocal total_registered
        if not buf:
            return
        if register_messages:
            # Prefer a dedicated messages API if you have it
            register_messages(batch_id, buf)  # type: ignore
        else:
            # Fallback: use the existing conversations API with one-message conversations
            # This mirrors your original helper usage style.
            conv_like = []
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            for it in buf:
                conv_like.append({
                    "conversation_id": f"conversation_{it['message_id']}",
                    "conversation_start": it.get("timestamp") or it.get("_ts_raw") or now,
                    "batch_id": batch_id,
                    "agent_name": it.get("sender") or it.get("_sender_raw"),
                    "client_name": None,
                    "scenario": "ingested_message",
                    "created_at": now,
                    "data_ref": it.get("data_ref"),
                    "messages": [  # keep an embedded minimal message
                        {
                            "message_id": it["message_id"],
                            "sender": it.get("sender") or it.get("_sender_raw"),
                            "timestamp": it.get("timestamp") or it.get("_ts_raw"),
                            "text": it.get("text") or it.get("_text_raw"),
                            "raw": it.get("raw"),
                        }
                    ],
                })
            register_conversations(batch_id, conv_like)
        total_registered += len(buf)
        buf.clear()

    for row in _iter_dedup_rows(con, chunk_size=chunk_size):
        # Build a normalized item
        msg_id = _message_id(row)
        item = {
            "message_id": msg_id,
            "data_ref": str(jsonl_path),
            "sender": row.get("_sender_raw"),
            "timestamp": row.get("_ts_raw"),
            "text": row.get("_text_raw"),
            "raw": row,  # preserve all original fields
        }
        buffer.append(item)
        if len(buffer) >= chunk_size:
            flush_buffer(buffer)
    # Flush remainder
    flush_buffer(buffer)

    print(f"✅ Registered {total_registered} items under batch {batch_id}")
    return batch_id, total_registered

def main():
    parser = argparse.ArgumentParser(description="Ingest messages JSONL with DuckDB and register in your registry")
    parser.add_argument("--jsonl", help="Path to messages.jsonl (skip menu)")
    parser.add_argument("--tag", help="Optional tag added to batch parameters", default=None)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT, help="Items per registry write (default 10000)")
    parser.add_argument("--no-dedup", action="store_true", help="Disable deduplication")
    args = parser.parse_args()

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
    else:
        print("📋 Message Ingestion Menu")
        print("———————————————")
        print(f"Scanning: {INPUT_DIR.resolve()}")
        jsonl_path = _menu_choose_jsonl(INPUT_DIR)

    ingest_with_duckdb(jsonl_path, tag=args.tag, chunk_size=args.chunk_size, dedup=not args.no_dedup)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Ingestion interrupted by user.")
