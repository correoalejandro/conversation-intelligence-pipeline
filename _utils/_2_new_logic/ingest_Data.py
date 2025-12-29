#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingest Processor Outputs — interactive, zero-arg CLI (stdlib-only)

Purpose
- Let other software (or you) easily consume the artifacts produced by the
  export processor we built: snapshots and per-run deltas for messages & media.
- Works without arguments: launches an interactive menu. Optional CLI flags for automation.

What it does
- Discovers runs from <OUT_DIR>/runs_index.jsonl
- Lets you choose kind: messages | media
- Lets you choose source: snapshot | latest | run:<RUN_ID>
- Reads JSONL (also supports .jsonl.gz if you compress old runs)
- Optional filters (text contains, time window for messages)
- Writes results to: stdout | file.jsonl | file.csv

No external deps. Python 3.9+
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple, Dict, Any

# =====================
# Utils
# =====================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl(path: Path) -> Iterator[dict]:
    """Stream JSONL (supports plain .jsonl and .jsonl.gz)."""
    if not path.exists():
        raise FileNotFoundError(path)
    opener = gzip.open if path.suffix.endswith(".gz") else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8", newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_csv(path: Path, rows: Iterable[dict], field_order: Optional[List[str]] = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return 0
    # Collect all fields if not specified
    if not field_order:
        keys = set()
        for r in rows:
            keys.update(r.keys())
        field_order = sorted(keys)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in field_order})
    return len(rows)


# =====================
# Data access helpers (contract with the processor)
# =====================

@dataclass
class Context:
    out_dir: Path


def runs_index(ctx: Context) -> List[dict]:
    idx_path = ctx.out_dir / "runs_index.jsonl"
    return list(read_jsonl(idx_path)) if idx_path.exists() else []


def latest_run_id(ctx: Context) -> Optional[str]:
    idx = runs_index(ctx)
    return idx[-1]["run_id"] if idx else None


def resolve_messages_source(ctx: Context, use: str) -> Path:
    if use == "snapshot":
        p = ctx.out_dir / "messages.jsonl"
    elif use == "latest":
        rid = latest_run_id(ctx)
        if not rid:
            raise FileNotFoundError("No runs found in runs_index.jsonl")
        p = ctx.out_dir / "runs" / rid / "messages_delta.jsonl"
    elif use.startswith("run:"):
        rid = use.split(":", 1)[1]
        p = ctx.out_dir / "runs" / rid / "messages_delta.jsonl"
    else:
        raise ValueError("use must be snapshot | latest | run:<RUN_ID>")
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def resolve_media_source(ctx: Context, use: str) -> Path:
    if use == "snapshot":
        p = ctx.out_dir / "media.jsonl"
    elif use == "latest":
        rid = latest_run_id(ctx)
        if not rid:
            raise FileNotFoundError("No runs found in runs_index.jsonl")
        p = ctx.out_dir / "runs" / rid / "media_delta.jsonl"
    elif use.startswith("run:"):
        rid = use.split(":", 1)[1]
        p = ctx.out_dir / "runs" / rid / "media_delta.jsonl"
    else:
        raise ValueError("use must be snapshot | latest | run:<RUN_ID>")
    if not p.exists():
        raise FileNotFoundError(p)
    return p


# =====================
# Filters (optional)
# =====================

def filter_messages(rows: Iterable[dict], text: Optional[str] = None,
                    since: Optional[str] = None, until: Optional[str] = None) -> Iterator[dict]:
    """Filter messages by substring in content (stringified) and time window (ISO in fields create_time/update_time)."""
    def in_window(r: dict) -> bool:
        if not (since or until):
            return True
        # choose an available time field
        ts = r.get("update_time") or r.get("create_time") or ""
        if not ts:
            return False if since else True
        try:
            # tolerate floats / ints
            if isinstance(ts, (int, float)):
                # seconds epoch → ISO
                ts_iso = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                ts_iso = str(ts)
        except Exception:
            ts_iso = str(ts)
        ok = True
        if since and ts_iso < since:
            ok = False
        if until and ts_iso > until:
            ok = False
        return ok

    for r in rows:
        if text is not None:
            blob = json.dumps(r.get("content"), ensure_ascii=False)
            if text.lower() not in blob.lower():
                continue
        if not in_window(r):
            continue
        yield r


def filter_media(rows: Iterable[dict], text: Optional[str] = None) -> Iterator[dict]:
    for r in rows:
        if text is not None:
            blob = (r.get("path") or "") + " " + json.dumps(r, ensure_ascii=False)
            if text.lower() not in blob.lower():
                continue
        yield r

def count_rows(path):
    raise NotImplementedError


# =====================
# Interactive Menu
# =====================

def menu(out_dir: Optional[Path] = None):
    ctx = Context(out_dir=Path(r"C:\Projects\projects\openai-export-processor\data\out").expanduser().resolve())
    ctx.out_dir.mkdir(parents=True, exist_ok=True)

    use = "snapshot"  # default source
    kind = "messages"  # or media
    text = None        # optional substring filter
    since = None       # ISO lower bound
    until = None       # ISO upper bound

    def show_header():
        print("\n=== Ingest Processor Outputs ===")
        print(f"Out dir : {ctx.out_dir}")
        print(f"Kind    : {kind}")
        print(f"Use     : {use}")
        print(f"Filter  : text={text!r} since={since} until={until}")
        print("---------------------------------")

    while True:
        show_header()
        print("1) Change out dir")
        print("2) Choose kind (messages/media)")
        print("3) Choose source (snapshot/latest/run:<RUN_ID>)")
        print("4) Set filters (text, since, until)")
        print("5) Preview count")
        print("6) Export to file (JSONL or CSV)")
        print("7) List recent runs")
        print("0) Exit")
        choice = input("Select: ").strip()
        try:
            if choice == "1":
                p = input("New out dir (default ./out): ").strip() or "./out"
                ctx.out_dir = Path(p).expanduser().resolve()
                ctx.out_dir.mkdir(parents=True, exist_ok=True)
            elif choice == "2":
                k = input("Kind [messages/media]: ").strip().lower()
                if k in ("messages", "media"):
                    kind = k
                else:
                    print("Invalid kind.")
            elif choice == "3":
                print("Source options: snapshot | latest | run:<RUN_ID>")
                if (ctx.out_dir / "runs_index.jsonl").exists():
                    idx = runs_index(ctx)
                    print("\nRecent runs:")
                    for row in idx[-10:]:
                        print(f"  {row['run_id']}  ({row.get('indexed_at')})  +msgs={count_rows(ctx.out_dir / 'runs' / row['run_id'] / 'messages_delta.jsonl')}  +media={count_rows(ctx.out_dir / 'runs' / row['run_id'] / 'media_delta.jsonl')}")
                u = input(f"Use (current {use}): ").strip() or use
                use = u

            elif choice == "4":
                text_in = input("Text contains (empty to clear): ").strip()
                text = text_in if text_in else None
                if kind == "messages":
                    since_in = input("Since (ISO, e.g., 2025-09-01T00:00:00Z) [empty=none]: ").strip()
                    until_in = input("Until (ISO) [empty=none]: ").strip()
                    since = since_in or None
                    until = until_in or None
            elif choice == "5":
                rows = load_rows(ctx, kind, use, text, since, until)
                n = sum(1 for _ in rows)
                print(f"Count: {n}")
            elif choice == "6":
                dest = input("Output path (e.g., ./out/export.jsonl or ./out/export.csv): ").strip()
                if not dest:
                    print("No path given.")
                    continue
                dest_path = Path(dest).expanduser().resolve()
                rows = list(load_rows(ctx, kind, use, text, since, until))
                if dest_path.suffix.lower() == ".csv":
                    n = write_csv(dest_path, rows)
                else:
                    n = write_jsonl(dest_path, rows)
                print(f"Wrote {n} rows → {dest_path}")
            elif choice == "7":
                idx = runs_index(ctx)
                if not idx:
                    print("No runs found.")
                else:
                    print("\nRecent runs:")
                    for row in idx[-20:]:
                        print(f"  {row.get('indexed_at')}  {row.get('run_id')}  +msgs={row.get('new_or_updated_messages',0)}  +media={row.get('new_or_updated_media',0)} renames={row.get('renamed_media',0)}  ({row.get('source_zip')})")
            elif choice == "0":
                print("Bye.")
                return
            else:
                print("Invalid option.")
        except Exception as e:
            print(f"Error: {e}")


def load_rows(ctx: Context, kind: str, use: str, text: Optional[str], since: Optional[str], until: Optional[str]) -> Iterator[dict]:
    if kind == "messages":
        src = resolve_messages_source(ctx, use)
        rows = read_jsonl(src)
        return filter_messages(rows, text=text, since=since, until=until)
    elif kind == "media":
        src = resolve_media_source(ctx, use)
        rows = read_jsonl(src)
        return filter_media(rows, text=text)
    else:
        raise ValueError("kind must be messages or media")


# =====================
# CLI
# =====================

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingest/Export processor outputs (snapshot or per-run deltas)")
    p.add_argument("--interactive", action="store_true", help="Start interactive menu (default if no other args)")
    p.add_argument("--out", type=str, default="./out", help="Processor out directory")
    p.add_argument("--kind", choices=["messages", "media"], help="What to read")
    p.add_argument("--use", default="snapshot", help="snapshot | latest | run:<RUN_ID>")
    p.add_argument("--text", default=None, help="Substring filter")
    p.add_argument("--since", default=None, help="ISO lower bound (messages only)")
    p.add_argument("--until", default=None, help="ISO upper bound (messages only)")
    p.add_argument("--to", default=None, help="Output file path (.jsonl or .csv). If omitted, just prints count")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_cli().parse_args(argv or sys.argv[1:])

    # Default to interactive if not enough info
    if args.interactive or (args.kind is None and args.to is None):
        menu(Path(args.out))
        return 0

    ctx = Context(out_dir=Path(args.out).expanduser().resolve())

    rows = list(load_rows(ctx, args.kind, args.use, args.text, args.since, args.until))
    if args.to:
        dest = Path(args.to).expanduser().resolve()
        if dest.suffix.lower() == ".csv":
            n = write_csv(dest, rows)
        else:
            n = write_jsonl(dest, rows)
        print(f"Wrote {n} rows → {dest}")
    else:
        print(f"Count: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
