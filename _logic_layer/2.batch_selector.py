#!/usr/bin/env python3
"""
2.v14.batch_selector.py — Interactive selector & analyzer for **vector batches**
===============================================================================

Now with an **Analyze** action:
• Pick a vectors_batch (message- or conversation-level).
• Run UMAP + HDBSCAN on its embeddings.
• Save models + experiment artifact, and register lineage.

Usage
-----
$ python 2.v14.batch_selector.py
"""

from __future__ import annotations

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

# Models
import umap  # type: ignore
import hdbscan  # type: ignore

# Project imports
sys.path.append("c:/Projects/clasificador_mensajes")
from _data_layer import registry  # type: ignore
from _data_layer.registry import register_embedding_analysis, register_model  # type: ignore

# Storage
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
EXPER_DIR = ROOT / "data/experiments"; EXPER_DIR.mkdir(parents=True, exist_ok=True)


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
    if "embedding" not in df.columns:
        raise SystemExit("❌ Batch DataFrame has no 'embedding' column.")
    return df

def summarize_df(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    n = len(df)
    has_msg = "message_id" in df.columns
    has_conv = "conversation_id" in df.columns
    has_batch = "batch_id" in df.columns
    dim = None
    if n > 0:
        try:
            dim = len(df.iloc[0]["embedding"])
        except Exception:
            dim = None
    parts = [f"rows={n}", f"columns={len(cols)}", f"embedding_dim={dim if dim is not None else '?'}"]
    if has_conv: parts.append("has conversation_id")
    if has_msg:  parts.append("has message_id")
    if has_batch:parts.append("has batch_id")
    if has_batch:
        try: parts.append(f"unique batches={df['batch_id'].nunique(dropna=True)}")
        except Exception: pass
    if has_conv:
        try: parts.append(f"unique conversations={df['conversation_id'].nunique(dropna=True)}")
        except Exception: pass
    return " ; ".join(parts)


# -----------------
# Analysis helpers
# -----------------
def fit_umap(X: np.ndarray, n_neighbors: int = 15, n_components: int = 2, metric: str = "cosine") -> umap.UMAP:
    return umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric=metric, random_state=42).fit(X)

def fit_hdbscan(X_2d: np.ndarray,
                min_cluster_size: int = 10,
                min_samples: Optional[int] = None,
                cluster_selection_epsilon: float = 0.0,
                cluster_selection_method: str = "eom",
                metric: str = "euclidean") -> hdbscan.HDBSCAN:
    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=(min_samples or min_cluster_size),
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        metric=metric,
        algorithm="generic",
        prediction_data=True
    ).fit(X_2d)

def analyze_batch(rec: Dict[str, Any], df: pd.DataFrame) -> Path:
    """Run UMAP + HDBSCAN on batch embeddings, save models and experiment, and register."""
    X = np.vstack(df["embedding"].to_numpy()).astype("float64", copy=False)

    # Fit
    um = fit_umap(X)
    X2 = um.embedding_.astype("float64", copy=False)
    hb = fit_hdbscan(X2)

    # Attach results
    out = df.copy()
    out["umap_x"], out["umap_y"] = X2[:, 0], X2[:, 1]
    out["cluster"] = hb.labels_
    out["strength"] = hb.probabilities_

    # Save models
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    um_path = MODEL_DIR / f"umap_model.{ts}.joblib"
    hb_path = MODEL_DIR / f"hdbscan_model.{ts}.joblib"
    joblib.dump(um, um_path)
    joblib.dump(hb, hb_path)

    # Save experiment
    exp_path = EXPER_DIR / f"exp_batch_{ts}.joblib"
    joblib.dump({"df": out, "source_batch_id": rec.get("id"), "params": rec.get("parameters")}, exp_path, compress=3)

    # Register
    try:
        model_id = register_model(str(um_path), model_type="umap", parameters={"n_neighbors": um.n_neighbors, "metric": um.metric})
        _ = register_model(str(hb_path), model_type="hdbscan",
                           parameters={"min_cluster_size": hb.min_cluster_size,
                                       "min_samples": hb.min_samples_,
                                       "metric": hb.metric,
                                       "algorithm": "generic"})
        register_embedding_analysis(
            data_ref=str(exp_path),
            parameters={"umap_model_id": model_id, "n_rows": len(out), "granularity": rec.get("parameters", {}).get("granularity")},
            parents=[rec.get("id")],
            batches=rec.get("batches") or []
        )
    except Exception as e:
        print("⚠️ Registry registration failed (continuing):", e)

    return exp_path


# -------------
# Main menu UI
# -------------
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
            print(f" id         : {rec.get('id')}")
            print(f" created    : {rec.get('created_at')}")
            p = rec.get("parameters") or {}
            print(f" model      : {p.get('model')}")
            print(f" n_vectors  : {p.get('n_vectors')}")
            print(f" granularity: {p.get('granularity')}")
            print(f" data_ref   : {rec.get('data_ref')}")

            # load and summarize
            try:
                df = load_df_from_batch(rec)
                print(" summary    :", summarize_df(df))
            except Exception as e:
                print(" summary    : (error loading) →", e)
                df = None

            # drill-down sub-menu
            while True:
                print("\n  Actions:")
                print("    a) Analyze this batch (UMAP + HDBSCAN)")
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
                elif act == "a":
                    if df is None:
                        print("  (no data loaded)")
                        continue
                    print("  ▶ Running UMAP + HDBSCAN...")
                    exp_path = analyze_batch(rec, df)
                    print(f"  ✅ Analysis saved → {exp_path.name}")
                else:
                    print("  Unknown action.")

if __name__ == "__main__":
    main()
