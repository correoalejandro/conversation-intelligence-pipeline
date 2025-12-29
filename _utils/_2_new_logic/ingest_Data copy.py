from __future__ import annotations

"""
Ingest Messages with DuckDB v12 — JSONL -> Registry (Interactive)

Flow:
  1) Pick a file from data/0_input/ (or pass --jsonl)
  2) Load & dedup with DuckDB (schema-agnostic)
  3) Create a prompt entry via register_prompt(...) (minimal metadata)
  4) Create batch via register_batch(data_ref, prompt_id, parameters)
  5) Register items via:
       - register_messages(batch_id, items) if available
       - else register_conversations(batch_id, items) as a safe fallback

Each item carries minimal normalized fields plus the full raw record.

Requires:  python -m pip install duckdb
"""

from pathlib import Path
import datetime as dt
from decimal import Decimal
import argparse
import hashlib
import json
import locale
import os
import sys
from typing import Dict, List, Any, Tuple

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")  # good default
except Exception:
    _ENC = None  # we'll fallback to character length if tiktoken not installed
# ---------- Console niceties (Windows, UTF-8) ----------
try:
    locale.setlocale(locale.LC_TIME, "Spanish_Spain.1252")
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------- Registry (keep your original signatures) ----------
try:
    from _data_layer.registry import (
        register_prompt,
        register_batch,
        register_conversations,
    )
    try:
        # Optional convenience, use if present
        from _data_layer.registry import register_messages  # type: ignore
    except Exception:
        register_messages = None
except Exception:
    # Fallback to 'registry' module if you mirror the same API there
    from registry import register_prompt, register_batch, register_conversations  # type: ignore
    try:
        from registry import register_messages  # type: ignore
    except Exception:
        register_messages = None

# ---------- Dependencies ----------
try:
    import duckdb
except Exception as e:
    raise RuntimeError("duckdb is required. Install with: python -m pip install duckdb") from e

INPUT_DIR = Path("data/0_input")
CHUNK_SIZE_DEFAULT = 10000  # change if you want smaller/larger registry writes


# ---------- Helpers ----------

#-------counting tokens (if tiktoken is installed)--------

def _get_text_from_row(row: dict) -> str:
    # same fields you already use for text
    return (
        row.get("text")
        or row.get("content")
        or row.get("body")
        or row.get("message")
        or ""
    )

def _count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is None:               # fallback if tiktoken isn't installed
        return len(text) // 4 + 1  # rough heuristic ~4 chars/token
    return len(_ENC.encode(text))

def _precompute_token_stats(jsonl_path: Path) -> dict:
    """One fast streaming pass to compute batch-level token stats."""
    total_tokens = 0
    max_tokens = 0
    n_rows = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            toks = _count_tokens(_get_text_from_row(row))
            total_tokens += toks
            max_tokens = max(max_tokens, toks)
            n_rows += 1
    avg = (total_tokens / n_rows) if n_rows else 0.0
    return {
        "rows": n_rows,
        "total_tokens": int(total_tokens),
        "avg_tokens": float(avg),
        "max_tokens": int(max_tokens),
        "enc": "cl100k_base" if _ENC else "fallback_chars_heuristic",
    }


#-------end counting tokens--------
def _sql_quote(path_str: str) -> str:
    """Escape a filesystem path so it can be safely used as a SQL string literal."""
    return path_str.replace("\\", "/").replace("'", "''")


def human_ts_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


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
    """Schema-agnostic load + dedup (exact duplicate rows collapse by content hash)."""
    p = _sql_quote(str(jsonl_path))
    # Load JSONL as a temp view
    con.execute(f"CREATE OR REPLACE TEMP VIEW src AS SELECT * FROM read_json_auto('{p}');")

    if not dedup:
        con.execute("CREATE OR REPLACE TEMP TABLE dedup AS SELECT * FROM src;")
        return

    # Dedup by full-row signature; no assumptions about columns
    con.execute("""
        CREATE OR REPLACE TEMP TABLE dedup AS
        SELECT * EXCLUDE(_rn) FROM (
          SELECT
            *,
            ROW_NUMBER() OVER (PARTITION BY sha1(to_json(src)) ORDER BY 1) AS _rn
          FROM src
        )
        WHERE _rn = 1;
    """)


def _iter_dedup_rows(con: "duckdb.DuckDBPyConnection", chunk_size: int):
    total = con.execute("SELECT COUNT(*) FROM dedup;").fetchone()[0]
    offset = 0
    while offset < total:
        res = con.execute("SELECT * FROM dedup LIMIT ? OFFSET ?;", [chunk_size, offset])
        cols = [d[0] for d in res.description]
        for tup in res.fetchall():
            yield dict(zip(cols, tup))
        offset += chunk_size


def _message_id(row: Dict[str, Any]) -> str:
    # Prefer any obvious id-like field; else a stable short hash of the row
    for k in ("id", "message_id", "msg_id", "unit_uid", "dataset_id"):
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v)
    return hashlib.sha1(
        json.dumps(row, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]

def _normalize_sender(row: Dict[str, Any]):
    return row.get("sender") or row.get("author") or row.get("role") or row.get("from") or None

def _normalize_timestamp(row: Dict[str, Any]):
    return row.get("timestamp") or row.get("ts") or row.get("time") or row.get("created_at") or None

def _normalize_text(row: Dict[str, Any]):
    return row.get("text") or row.get("content") or row.get("body") or row.get("message") or None

def _json_safe(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (dt.datetime, dt.date, dt.time)):
        if isinstance(obj, dt.datetime):
            if obj.tzinfo is None:
                obj = obj.replace(tzinfo=dt.timezone.utc)
            return obj.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return obj.isoformat()

    if isinstance(obj, dt.timedelta):
        return obj.total_seconds()

    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]

    return str(obj)

def ingest_with_duckdb(jsonl_path: Path, tag: str | None, chunk_size: int, dedup: bool) -> Tuple[str, int]:
    con = duckdb.connect()
    _build_dedup_relation(con, jsonl_path, dedup=dedup)

    # 1) Minimal prompt entry (keeps register_batch signature intact)
    prompt_desc = f"Ingestion source for {jsonl_path.name}"
    prompt_id = register_prompt(
        prompt_path=str(jsonl_path),
        description=prompt_desc,
        tags=["ingestion", "jsonl", "duckdb"],
        author="ingestor",
    )

    # 2) Create batch with correct signature (data_ref, prompt_id, parameters)
    params = {"format": "jsonl", "dedup": bool(dedup)}
    if tag:
        params["tag"] = tag
    batch_id = register_batch(
        data_ref=str(jsonl_path),
        prompt_id=prompt_id,
        parameters=params,
    )
    print(f"✅ Registered batch: {batch_id}")

    total_registered = 0
    buffer: List[Dict[str, Any]] = []


    def flush_buffer(buf: List[Dict[str, Any]]):
        nonlocal total_registered
        if not buf:
            return

        # Ensure all items are JSON-safe (including nested `raw`)
        safe_buf = [_json_safe(it) for it in buf]

        if register_messages:
            register_messages(batch_id, safe_buf)  # type: ignore
        else:
            conv_like = []
            now_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            for it in safe_buf:
                conv_like.append({
                    "conversation_id": f"conversation_{it['message_id']}",
                    "conversation_start": it.get("timestamp") or now_str,  # already string
                    "batch_id": batch_id,
                    "agent_name": it.get("sender"),
                    "client_name": None,
                    "scenario": "ingested_message",
                    "created_at": now_str,
                    "data_ref": it.get("data_ref"),
                    "messages": [
                        {
                            "message_id": it["message_id"],
                            "sender": it.get("sender"),
                            "timestamp": it.get("timestamp") or now_str,
                            "text": it.get("text"),
                            "raw": it.get("raw"),  # raw is now JSON-safe too
                        }
                    ],
                })
            # Final safety check (catches any stragglers before writing files)
            # import json; json.dumps(conv_like)  # uncomment to self-test
            register_conversations(batch_id, conv_like)

        total_registered += len(buf)
        buf.clear()

    # 3) Stream rows, normalize, and flush in chunks
    for row in _iter_dedup_rows(con, chunk_size=chunk_size):
        text_val = _normalize_text(row)
        item = {
            "message_id": _message_id(row),
            "data_ref": str(jsonl_path),
            "sender": _normalize_sender(row),
            "timestamp": _normalize_timestamp(row),
            "text": text_val,
            "tokens": _count_tokens(text_val),
            "raw": row,
        }
        buffer.append(item)
        if len(buffer) == 1 and total_registered == 0:
            print("🔎 sample item:", {k: item[k] for k in ("message_id","tokens","text") if k in item})

        if len(buffer) >= chunk_size:
            flush_buffer(buffer)


    flush_buffer(buffer)

    print(f"✅ Registered {total_registered} items under batch {batch_id}")
    return batch_id, total_registered


def main():
    parser = argparse.ArgumentParser(description="Ingest messages JSONL with DuckDB and register via _data_layer.registry")
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
