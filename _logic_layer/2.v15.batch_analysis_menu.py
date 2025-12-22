#!/usr/bin/env python3
"""
2.v15.batch_analysis_menu.py — Interactive UMAP ⇆ HDBSCAN **on vector batches**
================================================================================

What it does
------------
• Lists **vector batches** from the registry (stage="vectorizing_text", source="technical").
• Lets you filter by granularity (message / conversation) and by model.
• Lets you pick one batch and then runs the **full interactive parameter menu**
  (pipeline order, mode/update, UMAP and HDBSCAN params) — mirroring your
  conversation-level script.
• Saves UMAP/HDBSCAN models and an experiment bundle; registers lineage with the
  selected vectors_batch as the parent.

Usage
-----
$ python 2.v15.batch_analysis_menu.py
"""

from __future__ import annotations

import sys
import json
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


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------
def fetch_batches() -> List[Dict[str, Any]]:
    recs = registry.find(stage="vectorizing_text", source="technical")
    recs = [r for r in recs if isinstance(r.get("parameters"), dict)]
    recs.sort(key=lambda r: r.get("created_at", ""), reverse=True)  # newest first
    return recs

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
        created = r.get("created_at", "?")
        nvec = p.get("n_vectors", "?")
        fname = Path(r.get("data_ref", "")).name
        print(f"[{len(shown):>2}] {created} | {gran:<12} | {model:<24} | n={nvec:<7} | {fname}")
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


# ---------------------------------------------------------------------------
# Interactive parameter menu (mirrors your conversation-level pipeline)
# ---------------------------------------------------------------------------
def _prompt(value, cast, text: str):
    raw = input(f"{text} [{value}]: ").strip()
    return cast(raw) if raw else value

def interactive_menu(num_vectors: int) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "pipeline_order": "umap-first",
        "mode": "new",
        "update_strategy": "retrain",
        # UMAP
        "n_neighbors": 15,
        "n_components": 2,
        "umap_metric": "cosine",
        # HDBSCAN
        "min_cluster_size": 2,
        "min_samples": 2,
        "cluster_selection_epsilon": 0.0,
        "cluster_selection_method": "eom",
        "hdb_metric": "euclidean",
    }

    print("\n⚙️  Pipeline configuration  —", num_vectors, "vectors will be processed")
    print("(Press <Enter> to keep default)")
    print("——————————————————————————————————————————————————————")
    defaults["pipeline_order"] = _prompt(defaults["pipeline_order"], str, "Pipeline order (umap-first|hdbscan-first)")
    defaults["mode"] = _prompt(defaults["mode"], str, "Mode (new|update)")
    defaults["update_strategy"] = _prompt(defaults["update_strategy"], str, "Update strategy (retrain|incremental)")

    print("\n🔹 UMAP parameters")
    defaults["n_neighbors"] = _prompt(defaults["n_neighbors"], int, "n_neighbors")
    defaults["n_components"] = _prompt(defaults["n_components"], int, "n_components (2‑3 typical)")
    defaults["umap_metric"] = _prompt(defaults["umap_metric"], str, "metric")

    print("\n🔹 HDBSCAN parameters")
    defaults["min_cluster_size"] = _prompt(defaults["min_cluster_size"], int, "min_cluster_size")
    defaults["min_samples"] = _prompt(defaults["min_samples"], int, "min_samples")
    defaults["cluster_selection_epsilon"] = _prompt(defaults["cluster_selection_epsilon"], float, "cluster_selection_epsilon")
    defaults["cluster_selection_method"] = _prompt(defaults["cluster_selection_method"], str, "cluster_selection_method (eom|leaf)")
    defaults["hdb_metric"] = _prompt(defaults["hdb_metric"], str, "metric")

    return defaults


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def fit_umap(X: np.ndarray, n_neighbors: int, n_components: int, metric: str) -> umap.UMAP:
    return umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric=metric, random_state=42).fit(X)

def fit_hdbscan(X: np.ndarray,
                min_cluster_size: int,
                min_samples: int,
                cluster_selection_epsilon: float,
                cluster_selection_method: str,
                metric: str) -> hdbscan.HDBSCAN:
    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        metric=metric,
        algorithm='generic',
        prediction_data=True,
    ).fit(X)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Filters
    gfilter: Optional[str] = None  # "message" | "conversation" | None
    mfilter: Optional[str] = None  # substring on model

    while True:
        batches = fetch_batches()
        shown = list_batches(batches, gfilter, mfilter)

        print("\nOptions:")
        print("  [number] Inspect / analyze that batch")
        print("  g) Set granularity filter (message / conversation / all)")
        print("  m) Set model filter (substring, empty to clear)")
        print("  r) Refresh list")
        print("  q) Quit")

        choice = input("Select: ").strip().lower()
        if choice == "q":
            return
        elif choice == "r":
            continue
        elif choice == "g":
            val = input("Granularity (message / conversation / all): ").strip().lower()
            gfilter = val if val in ("message", "conversation") else None
            continue
        elif choice == "m":
            val = input("Model filter substring (blank to clear): ").strip()
            mfilter = val or None
            continue
        else:
            try:
                pick = int(choice)
            except ValueError:
                print("Invalid option."); continue
            if pick < 0 or pick >= len(shown):
                print("Out of range."); continue

            rec = batches[shown[pick]]
            p = rec.get("parameters") or {}
            print("\n— Batch info —")
            print(f" id         : {rec.get('id')}")
            print(f" created    : {rec.get('created_at')}")
            print(f" granularity: {p.get('granularity')}")
            print(f" model      : {p.get('model')}")
            print(f" n_vectors  : {p.get('n_vectors')}")
            print(f" data_ref   : {rec.get('data_ref')}")

            try:
                df = load_df_from_batch(rec)
                print(" summary    :", summarize_df(df))
            except Exception as e:
                print(" summary    : (error loading) →", e)
                df = None

            if df is None:
                input("\n(Press Enter to go back)")
                continue

            # === Full parameter menu (same spirit as your conversation-level menu) ===
            params = interactive_menu(len(df))

            # Prepare matrix
            X = np.vstack(df["embedding"].to_numpy()).astype("float64", copy=False)

            # ----- UMAP -----
            mode = params["mode"]; update_strategy = params["update_strategy"]
            umap_path = MODEL_DIR / "umap_model.joblib"
            if mode == "update" and umap_path.exists():
                umap_model: umap.UMAP = joblib.load(umap_path)
                if update_strategy == "incremental":
                    embedding_2d = umap_model.transform(X).astype("float64", copy=False)
                else:  # retrain
                    X_full = np.vstack([umap_model._raw_data, X])  # type: ignore[attr-defined]
                    umap_model = fit_umap(X_full, params["n_neighbors"], params["n_components"], params["umap_metric"])
                    embedding_2d = umap_model.embedding_[-len(X):].astype("float64", copy=False)
            else:
                umap_model = fit_umap(X, params["n_neighbors"], params["n_components"], params["umap_metric"])
                embedding_2d = umap_model.embedding_.astype("float64", copy=False)

            if mode == "new" or update_strategy == "retrain":
                joblib.dump(umap_model, umap_path)

            # ----- HDBSCAN -----
            hdb_path = MODEL_DIR / "hdbscan_model.joblib"
            if mode == "update" and hdb_path.exists():
                hdb_model: hdbscan.HDBSCAN = joblib.load(hdb_path)
                if update_strategy == "incremental":
                    labels, strengths = hdbscan.approximate_predict(hdb_model, embedding_2d.astype("float64", copy=False))
                    df["cluster"] = labels; df["strength"] = strengths
                else:
                    embedding_2d = embedding_2d.astype("float64", copy=False)
                    hdb_model = fit_hdbscan(
                        embedding_2d,
                        params["min_cluster_size"],
                        params["min_samples"],
                        params["cluster_selection_epsilon"],
                        params["cluster_selection_method"],
                        params["hdb_metric"],
                    )
            else:
                embedding_2d = embedding_2d.astype("float64", copy=False)
                hdb_model = fit_hdbscan(
                    embedding_2d,
                    params["min_cluster_size"],
                    params["min_samples"],
                    params["cluster_selection_epsilon"],
                    params["cluster_selection_method"],
                    params["hdb_metric"],
                )
                df["cluster"] = hdb_model.labels_
                df["strength"] = hdb_model.probabilities_

            if mode == "new" or update_strategy == "retrain":
                joblib.dump(hdb_model, hdb_path)

            # ----- Persist experiment & register -----
            # keep only light columns
            df = df.copy()
            if "embedding" in df.columns:
                del df["embedding"]
            df["umap_x"], df["umap_y"] = embedding_2d[:, 0], embedding_2d[:, 1]

            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_path = EXPER_DIR / f"exp_batch_{ts}.joblib"
            joblib.dump({"df": df, "source_batch_id": rec.get("id"), "params": params}, out_path, compress=3)
            print(f"\n✅ Results saved → {out_path.name}")

            try:
                umap_model_id = register_model(
                    data_ref=str(umap_path),
                    parameters={
                        "n_neighbors": params["n_neighbors"],
                        "n_components": params["n_components"],
                        "metric": params["umap_metric"],
                    },
                    parents=[rec.get("id")],
                    model_type="umap"
                )
                hdbscan_model_id = register_model(
                    data_ref=str(hdb_path),
                    parameters={
                        "min_cluster_size":          params["min_cluster_size"],
                        "min_samples":               params["min_samples"],
                        "cluster_selection_epsilon": params["cluster_selection_epsilon"],
                        "cluster_selection_method":  params["cluster_selection_method"],
                        "metric":                    params["hdb_metric"],
                    },
                    parents=[rec.get("id")],
                    model_type="hdbscan"
                )
                register_embedding_analysis(
                    data_ref=str(out_path),
                    parameters={
                        "pipeline_order": params["pipeline_order"],
                        "mode": mode,
                        "update_strategy": update_strategy,
                        "umap_model": umap_model_id,
                        "hdbscan_model": hdbscan_model_id,
                        "umap": {
                            "n_neighbors": params["n_neighbors"],
                            "n_components": params["n_components"],
                            "metric": params["umap_metric"],
                        },
                        "hdbscan": {
                            "min_cluster_size":          params["min_cluster_size"],
                            "min_samples":               params["min_samples"],
                            "cluster_selection_epsilon": params["cluster_selection_epsilon"],
                            "cluster_selection_method":  params["cluster_selection_method"],
                            "metric":                    params["hdb_metric"],
                        },
                        "granularity": (rec.get("parameters") or {}).get("granularity"),
                        "source_model": (rec.get("parameters") or {}).get("model"),
                        "source_n_vectors": (rec.get("parameters") or {}).get("n_vectors"),
                    },
                    parents=[rec.get("id")],
                    batches=rec.get("batches") or []
                )
                print("🧭 Registered analysis with lineage to the selected vectors_batch.")
            except Exception as e:
                print("⚠️ Registry registration failed (continuing):", e)

            input("\n(Press Enter to go back to the list)")

if __name__ == "__main__":
    main()
