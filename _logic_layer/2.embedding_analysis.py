#!/usr/bin/env python3
"""
2.v11.processPipeline_registry.py – registry‑aware, interactive UMAP ⇆ HDBSCAN pipeline
=====================================================================================
* Loads **embedding artifacts directly from the project registry** – **one vector per
  artifact** (`stage = "vectorizer:<conversation_id>"`). All selected vectors are
  stacked into a single matrix for dimensionality‑reduction + clustering.
* Presents an **interactive menu**:
    • Lists how many vectors will be processed.
    • Press **<Enter>** to accept defaults, type custom values, or type **all** to
      select every artifact automatically.
* Retains full v6 functionality (new vs update, retrain vs incremental).
* Registers the experiment via `register_embedding_analysis()` with lineage to
  every source vector.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
sys.path.append("c:/Projects/clasificador_mensajes")

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
import umap  # type: ignore
import hdbscan  # type: ignore

# 🏗  Project helpers ---------------------------------------------------------
from _data_layer import registry  # provides registry.find(), etc.
from _data_layer.registry import register_embedding_analysis, register_model
from _data_layer.api import _backend  # loader helpers

# ---------------------------------------------------------------------------
# 📋 Config & logger
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
EXPER_DIR = ROOT / "data/experiments"; EXPER_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("processPipeline")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# 🔧 Helper functions
# ---------------------------------------------------------------------------

def list_embedding_artifacts() -> List[Dict[str, Any]]:
    """Return registry records that are single‑vector embeddings."""
    recs = registry.find()
    emb_recs: List[Dict[str, Any]] = [
        r for r in recs
        if isinstance(r.get("stage"), str) and r["stage"].startswith("vectorizer")
        and r.get("backend") in {"json", "joblib"}
    ]
    emb_recs.sort(key=lambda r: r.get("created_at", ""), reverse=True)  # newest first
    return emb_recs


def _prompt_select_artifacts(recs: Sequence[Dict[str, Any]]) -> List[str]:
    """Interactive selection – returns list of selected artifact IDs."""
    if not recs:
        raise SystemExit("❌ No embedding artifacts found in registry. Run the embedding stage first.")

    print("\n📦 Found", len(recs), "vector artifacts in registry:")
    for idx, r in enumerate(recs[:20]):  # show first 20 to avoid flooding
        print(f"  [{idx}] {r['id']} | {Path(r['data_ref']).name}")
    if len(recs) > 20:
        print(f"  … and {len(recs) - 20} more")

    msg = "Select artifact numbers (comma‑separated) or type 'all' [all]: "
    sel = input(msg).strip().lower()
    if sel in {"", "all"}:
        return [r["id"] for r in recs]
    idxs = [int(s) for s in sel.split(',') if s.strip().isdigit()]
    invalid = [i for i in idxs if i < 0 or i >= len(recs)]
    if invalid:
        raise SystemExit(f"❌ Invalid selection: {invalid}")
    return [recs[i]["id"] for i in idxs]


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
    defaults["min_samples"] = _prompt(defaults.get("min_samples", defaults["min_cluster_size"]), int, "min_samples")
    defaults["cluster_selection_epsilon"] = _prompt(defaults.get("cluster_selection_epsilon", 0.0), float, "cluster_selection_epsilon")
    defaults["cluster_selection_method"] = _prompt(defaults.get("cluster_selection_method", "eom"), str, "cluster_selection_method (eom|leaf)")
    defaults["hdb_metric"] = _prompt(defaults["hdb_metric"], str, "metric")


    return defaults

# ---------------------------------------------------------------------------
# 🧠 Model fit helpers
# ---------------------------------------------------------------------------

def fit_umap(X: np.ndarray, n_neighbors: int, n_components: int, metric: str) -> umap.UMAP:
    return umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        random_state=42,
    ).fit(X)


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
# 🚀 Main helpers
# ---------------------------------------------------------------------------

def _json_to_df(obj: Any) -> pd.DataFrame:
    """Convert whatever we loaded into a DataFrame with an 'embedding' column."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        if "df" in obj:
            return obj["df"]
        return pd.DataFrame([obj])
    raise ValueError("Unsupported artifact format – expected list/dict/DataFrame")


def load_embeddings_from_artifacts(ids, add_text=False):
    """
    Devuelve un DataFrame con las columnas:
      embedding  (se eliminará luego)
      source_embedding_artifact
      text  (solo si add_text=True)
    """
    import json, pandas as pd, numpy as np
    df_parts = []
    lookup = {r["id"]: r for r in registry.find() if r.get("id")}

    for aid in ids:
        rec = lookup.get(aid);  # salta si no existe
        if not rec:
            continue
        data = _backend(rec["backend"]).load(rec["data_ref"])
        df = _json_to_df(data)
        if "embedding" not in df.columns:
            continue
        df["source_embedding_artifact"] = aid

        if add_text:
            conv_path = rec.get("parameters", {}).get("source")
            if conv_path and Path(conv_path).exists():
                raw = json.load(Path(conv_path).open()) \
                          .get("raw", "")
                df["text"] = raw
        df_parts.append(df)

    if not df_parts:
        raise SystemExit("❌ No embeddings loaded.")
    return pd.concat(df_parts, ignore_index=True)

# ---------------------------------------------------------------------------
# 🏁 Main
# ---------------------------------------------------------------------------

def main(add_text=False):
    emb_recs = list_embedding_artifacts()
    selected_ids = _prompt_select_artifacts(emb_recs)

    logger.info("Selected %d embedding artifacts", len(selected_ids))
    df = load_embeddings_from_artifacts(selected_ids, add_text= True)

    logger.info("Loaded %d vectors for analysis", len(df))

    params = interactive_menu(len(df))

    X = np.vstack(df["embedding"].to_numpy())

    # ----------------------- UMAP ------------------------------------
    mode, update_strategy = params["mode"], params["update_strategy"]
    umap_path = MODEL_DIR / "umap_model.joblib"

    if mode == "update" and umap_path.exists():
        # ── Cargamos modelo existente ───────────────────────────────
        umap_model: umap.UMAP = joblib.load(umap_path)
        logger.info("Loaded existing UMAP model (%s)", umap_path.name)

        if update_strategy == "incremental":
            # Solo transformamos los vectores nuevos
            embedding_2d = umap_model.transform(X).astype("float64", copy=False)
        else:  # retrain
            X_full = np.vstack([umap_model._raw_data, X])       # type: ignore[attr-defined]
            umap_model = fit_umap(
                X_full,
                params["n_neighbors"],
                params["n_components"],
                params["umap_metric"],
            )
            # Coordenadas solo de los ejemplos recién añadidos
            embedding_2d = umap_model.embedding_[-len(X):].astype("float64", copy=False)
    
    else:
        # ── Ejecución fresh: entrenamos desde cero ──────────────────
        umap_model = fit_umap(
            X,
            params["n_neighbors"],
            params["n_components"],
            params["umap_metric"],
        )
        embedding_2d = umap_model.embedding_.astype("float64", copy=False)

    # Guardamos modelo si se creó o se re‑entrenó
    if mode == "new" or update_strategy == "retrain":
        joblib.dump(umap_model, umap_path)
        logger.info("UMAP model saved → %s", umap_path.name)

    # ----------------------- HDBSCAN ---------------------------------


    if mode == "new" or update_strategy == "retrain":
        joblib.dump(umap_model, umap_path)
        logger.info("UMAP model saved → %s", umap_path.name)

   
    hdb_path = MODEL_DIR / "hdbscan_model.joblib"
    if mode == "update" and hdb_path.exists():
        hdb_model: hdbscan.HDBSCAN = joblib.load(hdb_path)
        logger.info("Loaded existing HDBSCAN model (%s)", hdb_path.name)
        if update_strategy == "incremental":
            embedding_2d = embedding_2d.astype("float64", copy=False)  # ← Añadir

            labels, strengths = hdbscan.approximate_predict(hdb_model, embedding_2d)
            df["cluster"] = labels
            df["strength"] = strengths
        else:
            X_full = np.vstack([hdb_model._raw_data, embedding_2d])  # type: ignore[attr-defined]
            embedding_2d = embedding_2d.astype("float64", copy=False)  # ← Añadir

            hdb_model = fit_hdbscan(
                embedding_2d,
                params["min_cluster_size"],
                params["min_samples"],
                params["cluster_selection_epsilon"],
                params["cluster_selection_method"],
                params["hdb_metric"],
            )
    else:
        embedding_2d = embedding_2d.astype("float64", copy=False)  # ← Añadir

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
        print(f"🔹 Clústeres encontrados: {df['cluster'].nunique(dropna=True)} "
      f"(≠ -1) | ruido: {(df['cluster'] == -1).sum()}")


    if mode == "new" or update_strategy == "retrain":
        joblib.dump(hdb_model, hdb_path)
        logger.info("HDBSCAN model saved → %s", hdb_path.name)

    # ----------------------- Persist bundle & register -------------
    del df["embedding"]

    df["umap_x"], df["umap_y"] = embedding_2d[:, 0], embedding_2d[:, 1]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = EXPER_DIR / f"exp_{ts}.joblib"
    joblib.dump({"df": df}, out_path, compress=3)
    logger.info("✅ Results saved → %s", out_path.name)


    # ── We register UMAP and HDBSCAN and get their artifact IDs───────────────────────────────
    umap_model_id = register_model(
        data_ref=str(umap_path),
        parameters={
            "n_neighbors": params["n_neighbors"],
            "n_components": params["n_components"],
            "metric": params["umap_metric"],
        },
        parents=selected_ids,
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
        parents=selected_ids,
        model_type="hdbscan"
    )

    # ── Registramos el análisis de embeddings ───────────────────────
    art_id = register_embedding_analysis(
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
        },
        parents=selected_ids,
    )


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Run embedding‑analysis pipeline")
    parser.add_argument("--add-text", action="store_true",
                        help="Append original raw text to the experiment artifact")
    args = parser.parse_args()

    try:
        main(add_text=args.add_text)
    except KeyboardInterrupt:
        print("\n👋 Process interrupted by user.")
