#!/usr/bin/env python3
"""
2.v15.batch_analysis_menu.py — Interactive UMAP ⇆ HDBSCAN on *multiple* vector batches
======================================================================================

• Lists vector *batches* from the registry (stage="vectorizing_text", source="technical").
• You can select **multiple indices separated by spaces** (e.g., `1 3 7` or `all`).
• Loads all selected batches, checks model & dimension compatibility, **concatenates**,
  then runs the full **UMAP/HDBSCAN** parameter menu.
• Saves models + a mixed experiment bundle and registers lineage to **all** selected batches.

No changes are made to the registry API surface (no `batches=` kwarg).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
import numpy as np  # if not already imported


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


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------
def _rec_sort_key(rec: Dict[str, Any]) -> float:
    """Prefer sorting by data_ref mtime; fallback to 0 if missing."""
    try:
        p = Path(rec.get("data_ref", ""))
        return p.stat().st_mtime
    except Exception:
        return 0.0

def fetch_batches() -> List[Dict[str, Any]]:
    recs = registry.find(stage="vectorizing_text", source="technical")
    recs = [r for r in recs if isinstance(r.get("parameters"), dict)]
    recs.sort(key=_rec_sort_key, reverse=True)  # newest first by file mtime
    return recs

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

def _infer_granularity_from_df(df: pd.DataFrame) -> str:
    if "message_id" in df.columns:
        return "message"
    if "conversation_id" in df.columns and "message_id" not in df.columns:
        return "conversation"
    return "unknown"

def _ensure_params_complete(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Return parameters with inferred granularity/n_vectors when missing."""
    p = (rec.get("parameters") or {}).copy()
    need_gran = p.get("granularity") in (None, "", "unknown")
    need_nvec = p.get("n_vectors") in (None, "", 0)
    if need_gran or need_nvec:
        try:
            df = load_df_from_batch(rec)
            if need_gran:
                p["granularity"] = _infer_granularity_from_df(df)
            if need_nvec:
                p["n_vectors"] = int(len(df))
        except Exception:
            pass
    return p

def list_batches(batches: List[Dict[str, Any]], granularity: Optional[str], model_filter: Optional[str]) -> List[int]:
    print("\nAvailable vector batches (newest first):")
    print("Idx Created_at           | Granularity  | Model                   | Count    | File")
    print("---- --------------------+--------------+-------------------------+----------+---------------------------")
    shown = []
    for i, r in enumerate(batches):
        p = _ensure_params_complete(r)
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
# Interactive parameter menu (mirrors conversation-level flow)
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
    defaults["n_components"] = _prompt(defaults["n_components"], int, "n_components (2-3 typical)")
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


## Time helpers
def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@contextmanager
def step(name: str, details: str = ""):
    """Prints start/finish + elapsed time for a named step."""
    hdr = f"[{_now()}] ▶ {name}"
    if details:
        hdr += f" — {details}"
    print(hdr); sys.stdout.flush()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[{_now()}] ✓ {name} — done in {dt:.2f}s\n"); sys.stdout.flush()

def _shape(a):
    try:
        return tuple(a.shape)
    except Exception:
        return "n/a"
##  --------------------------------------------------------------------------

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
# Multi-select helpers
# ---------------------------------------------------------------------------
def _parse_space_indices(s: str, max_idx: int) -> List[int]:
    """Parse space-separated indices ('1 3 7') into a unique, ordered list."""
    toks = [t for t in s.strip().split() if t]
    out: List[int] = []
    for t in toks:
        if not t.isdigit():
            raise ValueError(f"Non-numeric token: {t!r}")
        i = int(t)
        if i < 0 or i >= max_idx:
            raise ValueError(f"Index out of range: {i}")
        if i not in out:
            out.append(i)
    return out

def _load_and_concat_selected(batches: List[Dict[str, Any]], shown: List[int], selection: List[int]) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Load DataFrames for selected UI indices and concatenate them; also return the chosen registry recs."""
    chosen_recs: List[Dict[str, Any]] = []
    dfs: List[pd.DataFrame] = []
    base_dim: Optional[int] = None
    base_model: Optional[str] = None
    for ui_idx in selection:
        rec = batches[shown[ui_idx]]
        params = _ensure_params_complete(rec)
        model = params.get("model")
        df = load_df_from_batch(rec)
        # infer embedding dim
        if len(df) == 0:
            continue
        try:
            dim = len(df.iloc[0]["embedding"])
        except Exception:
            raise SystemExit("❌ Selected batch has no usable 'embedding' array.")
        # model/dim checks
        if base_dim is None: base_dim = dim
        if base_model is None: base_model = model
        if dim != base_dim:
            raise SystemExit(f"❌ Embedding dimension mismatch: {base_dim} vs {dim} in batch {rec.get('id')}")
        if model != base_model:
            raise SystemExit(f"❌ Model mismatch: {base_model} vs {model} in batch {rec.get('id')}")
        # annotate
        # annotate (explicit per-vector provenance)
        df = df.copy()
        df["granularity"] = (params.get("granularity")
                            or _infer_granularity_from_df(df))  # "message" / "conversation"
        df["source_batch_id"] = rec.get("id")
        df["source_model"] = model
        dfs.append(df)
        chosen_recs.append(rec)
    if not dfs:
        raise SystemExit("❌ No rows loaded from the selected batches.")
    mixed = pd.concat(dfs, ignore_index=True)
    return mixed, chosen_recs


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
        print("  [space-separated indices] Analyze those batches (e.g., '1 3 7' or 'all')")
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
            # multi-select by spaces (or 'all')
            if choice == "all":
                sel_ui = list(range(len(shown)))
            else:
                try:
                    sel_ui = _parse_space_indices(choice, len(shown))
                except ValueError as e:
                    print("Selection error:", e)
                    continue
            # Load & concat
            try:
                df, chosen_recs = _load_and_concat_selected(batches, shown, sel_ui)
            except SystemExit as e:
                print(str(e)); input("\n(Press Enter to go back)"); continue

            # quick preview
            gran_mix = sorted(set((_ensure_params_complete(r).get("granularity") or "?") for r in chosen_recs))
            models_mix = sorted(set((_ensure_params_complete(r).get("model") or "?") for r in chosen_recs))
            nrows = len(df)
            print(f"\n— Mix summary —  rows={nrows}  |  granularity={gran_mix}  |  model={models_mix}")
            print("  sources:", [r.get("id") for r in chosen_recs])

            # === Full parameter menu ===
            params = interactive_menu(nrows)

            # Prepare matrix
            X = np.vstack(df["embedding"].to_numpy()).astype("float64", copy=False)

                        # ----- UMAP_kD → HDBSCAN → UMAP_2D (always this order) -----
            mode = params["mode"]; update_strategy = params["update_strategy"]

            # Paths (keep your original path for the *clustering* UMAP; add a new one for projection)
            umap_path      = MODEL_DIR / "umap_model.joblib"        # UMAP used for clustering (k-D)
            umap_proj_path = MODEL_DIR / "umap_proj_model.joblib"   # UMAP used only for 2D projection
            hdb_path       = MODEL_DIR / "hdbscan_model.joblib"

            # 1) UMAP → k-D (clustering space, k = params["n_components"])
            k = max(2, int(params["n_components"]))
            with step("UMAP → k-D (clustering space)",
                    details=f"k={k}, n_neighbors={params['n_neighbors']}, metric={params['umap_metric']}, mode={mode}, strategy={update_strategy}"):
                if mode == "update" and umap_path.exists():
                    with step("Load UMAP model", details=str(umap_path)):
                        umap_model: umap.UMAP = joblib.load(umap_path)
                    if update_strategy == "incremental":
                        with step("UMAP.transform (k-D)", details=f"X={_shape(X)}"):
                            Z_k = umap_model.transform(X).astype("float64", copy=False)
                    else:  # retrain
                        with step("Stack previous raw_data + X", details=f"X={_shape(X)}"):
                            X_full = np.vstack([umap_model._raw_data, X])  # type: ignore[attr-defined]
                        with step("UMAP.fit (retrain on X_full)", details=f"X_full={_shape(X_full)}"):
                            umap_model = fit_umap(X_full, params["n_neighbors"], k, params["umap_metric"])
                        with step("Slice new embedding rows", details=f"take last {len(X)} rows"):
                            Z_k = umap_model.embedding_[-len(X):].astype("float64", copy=False)
                else:
                    with step("UMAP.fit (fresh)", details=f"X={_shape(X)}"):
                        umap_model = fit_umap(X, params["n_neighbors"], k, params["umap_metric"])
                    with step("Extract embedding", details=f"embedding_={_shape(umap_model.embedding_)}"):
                        Z_k = umap_model.embedding_.astype("float64", copy=False)

            with step("Persist UMAP (k-D model)", details=f"path={umap_path}"):
                if mode == "new" or update_strategy == "retrain":
                    joblib.dump(umap_model, umap_path)
                else:
                    print("Skipped (only persisted on 'new' or 'retrain').")

            # 2) HDBSCAN on k-D (cluster in the UMAP_k space, NOT in 2D)
            with step("HDBSCAN on k-D",
                    details=f"Z_k={_shape(Z_k)}, min_cluster_size={params['min_cluster_size']}, min_samples={params['min_samples']}, epsilon={params['cluster_selection_epsilon']}, method={params['cluster_selection_method']}, metric={params['hdb_metric']}, mode={mode}, strategy={update_strategy}"):
                if mode == "update" and hdb_path.exists():
                    with step("Load HDBSCAN model", details=str(hdb_path)):
                        hdb_model: hdbscan.HDBSCAN = joblib.load(hdb_path)
                    if update_strategy == "incremental":
                        with step("HDBSCAN.approximate_predict", details=f"Z_k={_shape(Z_k)}"):
                            labels, strengths = hdbscan.approximate_predict(hdb_model, Z_k)
                        df["cluster"] = labels; df["strength"] = strengths
                    else:
                        with step("HDBSCAN.fit (recluster)", details=f"Z_k={_shape(Z_k)}"):
                            hdb_model = fit_hdbscan(
                                Z_k,
                                params["min_cluster_size"],
                                params["min_samples"],
                                params["cluster_selection_epsilon"],
                                params["cluster_selection_method"],
                                params["hdb_metric"],
                            )
                        df["cluster"]  = hdb_model.labels_
                        df["strength"] = hdb_model.probabilities_
                else:
                    with step("HDBSCAN.fit (fresh)", details=f"Z_k={_shape(Z_k)}"):
                        hdb_model = fit_hdbscan(
                            Z_k,
                            params["min_cluster_size"],
                            params["min_samples"],
                            params["cluster_selection_epsilon"],
                            params["cluster_selection_method"],
                            params["hdb_metric"],
                        )
                    df["cluster"]  = hdb_model.labels_
                    df["strength"] = hdb_model.probabilities_

            with step("Persist HDBSCAN model", details=f"path={hdb_path}"):
                if mode == "new" or update_strategy == "retrain":
                    joblib.dump(hdb_model, hdb_path)
                else:
                    print("Skipped (only persisted on 'new' or 'retrain').")

            # 3) UMAP → 2D (projection only for output/plots; DO NOT use for clustering)
            with step("UMAP → 2D projection",
                    details=f"from k-D Z_k={_shape(Z_k)}, mode={mode}, strategy={update_strategy}"):
                if mode == "update" and umap_proj_path.exists() and update_strategy == "incremental":
                    with step("Load 2D UMAP model", details=str(umap_proj_path)):
                        umap_proj: umap.UMAP = joblib.load(umap_proj_path)
                    with step("UMAP.transform (2D)", details=f"Z_k={_shape(Z_k)}"):
                        embedding_2d = umap_proj.transform(Z_k).astype("float64", copy=False)
                else:
                    nn_2d = max(30, params["n_neighbors"] // 2)
                    with step("UMAP.fit (2D projection)", details=f"n_neighbors={nn_2d}, metric=euclidean, Z_k={_shape(Z_k)}"):
                        umap_proj = fit_umap(Z_k, n_neighbors=nn_2d, n_components=2, metric="euclidean")
                    with step("Extract 2D embedding", details=f"embedding_={_shape(umap_proj.embedding_)}"):
                        embedding_2d = umap_proj.embedding_.astype("float64", copy=False)

            with step("Persist UMAP (2D projection model)", details=f"path={umap_proj_path}"):
                if mode == "new" or update_strategy == "retrain":
                    joblib.dump(umap_proj, umap_proj_path)
                else:
                    print("Skipped (only persisted on 'new' or 'retrain').")

            # ----- Persist experiment & register -----
            df = df.copy()
            if "embedding" in df.columns:
                del df["embedding"]
            df["umap_x"], df["umap_y"] = embedding_2d[:, 0], embedding_2d[:, 1]

            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_path = EXPER_DIR / f"exp_mix_{ts}.joblib"
            joblib.dump({"df": df,
                         "source_batch_ids": [r.get("id") for r in chosen_recs],
                         "params": params},
                        out_path, compress=3)
            print(f"\n✅ Results saved → {out_path.name}")

            try:
                umap_model_id = register_model(
                    data_ref=str(umap_path),
                    parameters={
                        "n_neighbors": params["n_neighbors"],
                        "n_components": params["n_components"],
                        "metric": params["umap_metric"],
                    },
                    parents=[r.get("id") for r in chosen_recs],
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
                    parents=[r.get("id") for r in chosen_recs],
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
                        "granularity_mix": sorted(set((_ensure_params_complete(r).get("granularity") or "?") for r in chosen_recs)),
                        "source_models": sorted(set((_ensure_params_complete(r).get("model") or "?") for r in chosen_recs)),
                        "source_n_vectors_total": int(len(df)),
                        "source_batches": [r.get("batches") for r in chosen_recs],
                        "source_batch_ids": [r.get("id") for r in chosen_recs],
                    },
                    parents=[r.get("id") for r in chosen_recs]
                )
                print("🧭 Registered analysis with lineage to the selected vector batches.")
            except Exception as e:
                print("⚠️ Registry registration failed (continuing):", e)

            input("\n(Press Enter to go back to the list)")

if __name__ == "__main__":
    main()
