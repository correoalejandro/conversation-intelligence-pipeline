#!/usr/bin/env python3
"""
2.v14.batch_selector.py — Interactive selector for **vector batches**
====================================================================

What it does
------------
• Scans the registry for artifacts with stage="vectorizing_text" (source="technical").
• Shows them as an interactive list (newest first) with:
  index, created_at, granularity, model, n_vectors, file name.
• Lets you filter by granularity (message / conversation / all) and by model.
• Lets you inspect a batch (loads its joblib DataFrame and prints a quick summary).
• Optionally export the selected batch to CSV (for quick ad‑hoc checks).

Why
---
Your analysis script was focused on single‑vector artifacts ("vectorizer:*").
This tool lists the **batch artifacts** that aggregate many vectors in one file.

Usage
-----
$ python 2.v14.batch_selector.py
"""

from __future__ import annotations

import sys
import joblib
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project imports
sys.path.append("c:/Projects/clasificador_mensajes")
from _data_layer import registry  # type: ignore

# --------------------------------------
# Fetch & filter registry batch records
# --------------------------------------
def fetch_batches() -> List[Dict[str, Any]]:
    recs = registry.find(stage="vectorizing_text", source="technical")
    # Expect parameters like: {"model": "...", "n_vectors": N, "granularity": "message"|"conversation"}
    recs = [r for r in recs if isinstance(r.get("parameters"), dict)]
    recs.sort(key=lambda r: r.get("created_at", ""), reverse=True)  # newest first
    return recs

def format_row(idx: int, r: Dict[str, Any]) -> str:
    p = r.get("parameters") or {}
    gran = p.get("granularity", "?")
    model = p.get("model", "?")
    nvec = p.get("n_vectors", "?")
    created = r.get("created_at", "?")
    fname = Path(r.get("data_ref", "")).name
    return f"[{idx:>2}] {created} | {gran:<12} | {model:<24} | n={nvec:<7} | {fname}"

def list_batches(batches: List[Dict[str, Any]], granularity: Optional[str], model_filter: Optional[str]) -> List[int]:
    print("\nAvailable vector batches (newest first):")
    print("Idx Created_at           | Granularity  | Model                   | Count    | File")
    print("---- --------------------+--------------+-------------------------+----------+---------------------------")
    shown = []
    for i, r in enumerate(batches):
        p = r.get("parameters") or {}
        gran = (p.get("granularity") or "").lower()
        model = (p.get("model") or "")
        if granularity and gran != granularity.lower():
            continue
        if model_filter and model_filter not in model:
            continue
        print(format_row(len(shown), r))
        shown.append(i)
    if not shown:
        print("(no batches match the current filters)")
    return shown

def load_df_from_batch(rec: Dict[str, Any]) -> pd.DataFrame:
    data_ref = rec.get("data_ref")
    if not data_ref:
        raise SystemExit("❌ Batch has no data_ref.")
    obj = joblib.load(data_ref)
    if isinstance(obj, dict) and "df" in obj:
        df = obj["df"]
    else:
        df = obj
    if not isinstance(df, pd.DataFrame):
        raise SystemExit("❌ Unexpected format (expected DataFrame).")
    return df

def summarize_df(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    n = len(df)
    # typical columns: conversation_id, batch_id, message_id?, text?, embedding (array)
    has_msg = "message_id" in df.columns
    has_conv = "conversation_id" in df.columns
    has_batch = "batch_id" in df.columns
    emb_dim = "embedding"
    dim = None
    if emb_dim in df.columns and n > 0:
        try:
            dim = len(df.iloc[0][emb_dim])
        except Exception:
            dim = None

    parts = [f"rows={n}",
             f"columns={len(cols)}",
             f"embedding_dim={dim if dim is not None else '?'}"]
    if has_conv: parts.append("has conversation_id")
    if has_msg:  parts.append("has message_id")
    if has_batch:parts.append("has batch_id")

    # quick per-batch / per-conversation counts if present
    extras = []
    if has_batch:
        try:
            extras.append(f"unique batches={df['batch_id'].nunique(dropna=True)}")
        except Exception:
            pass
    if has_conv:
        try:
            extras.append(f"unique conversations={df['conversation_id'].nunique(dropna=True)}")
        except Exception:
            pass
    if extras:
        parts.append(" | ".join(extras))

    return " ; ".join(parts)

def main():
    batches = fetch_batches()
    gfilter: Optional[str] = None  # "message" | "conversation" | None
    mfilter: Optional[str] = None  # substring on model, e.g., "3-small"

    while True:
        shown_idx = list_batches(batches, gfilter, mfilter)
        print("\nOptions:")
        print("  [number] Inspect that batch")
        print("  g) Set granularity filter (message / conversation / all)")
        print("  m) Set model filter (substring, empty to clear)")
        print("  r) Refresh list")
        print("  q) Quit")

        choice = input("Select: ").strip().lower()
        if choice == "q":
            break
        elif choice == "r":
            batches = fetch_batches()
            continue
        elif choice == "g":
            val = input("Granularity (message / conversation / all): ").strip().lower()
            if val in ("message", "conversation"):
                gfilter = val
            else:
                gfilter = None
            continue
        elif choice == "m":
            val = input("Model filter substring (blank to clear): ").strip()
            mfilter = val or None
            continue
        else:
            # numeric?
            try:
                pick = int(choice)
            except ValueError:
                print("Invalid option.")
                continue
            if pick < 0 or pick >= len(shown_idx):
                print("Out of range.")
                continue
            rec = batches[shown_idx[pick]]
            print("\n— Batch info —")
            print(f" id        : {rec.get('id')}")
            print(f" created   : {rec.get('created_at')}")
            p = rec.get("parameters") or {}
            print(f" model     : {p.get('model')}")
            print(f" n_vectors : {p.get('n_vectors')}")
            print(f" granularity: {p.get('granularity')}")
            print(f" data_ref  : {rec.get('data_ref')}")

            # load and summarize
            try:
                df = load_df_from_batch(rec)
                print(" summary   :", summarize_df(df))
            except Exception as e:
                print(" summary   : (error loading) →", e)
                df = None

            # drill‑down sub‑menu
            while True:
                print("\n  Actions:")
                print("    v) View first 5 rows (without embeddings)")
                print("    e) Export to CSV (without embeddings)")
                print("    b) Back to list")
                act = input("    Select action: ").strip().lower()
                if act == "b":
                    break
                elif act == "v":
                    if df is None:
                        print("  (no data loaded)")
                        continue
                    # drop heavy column for printing
                    small = df.drop(columns=["embedding"], errors="ignore").head(5)
                    with pd.option_context("display.max_colwidth", 120, "display.width", 160):
                        print("\n", small)
                elif act == "e":
                    if df is None:
                        print("  (no data loaded)")
                        continue
                    out = Path("export_batch.csv")
                    small = df.drop(columns=["embedding"], errors="ignore")
                    small.to_csv(out, index=False)
                    print(f"  CSV exported → {out.resolve()}")
                else:
                    print("  Unknown action.")

if __name__ == "__main__":
    main()
