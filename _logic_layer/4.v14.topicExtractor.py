#!/usr/bin/env python3
"""
Topic Extractor v0.4 – lean edition
===================================

Default outputs (always saved)
------------------------------
1. `topic:merged_projection`
   • One row per message with: text, cluster, top_tokens, cluster_centroid (vector).

2. `topic:tfidf_full_table`
   • Long table (cluster_id, word, tfidf_score).

Optional output
---------------
• `--save-matrix` → `topic:tfidf_matrix`  (scipy .npz sparse TF-IDF matrix)

No other artefacts are written.
"""
from __future__ import annotations

from datetime import datetime, timezone        
import joblib 
import json, re

import sys
sys.path.append("c:/Projects/clasificador_mensajes")
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# ───────── Project helpers ────────────────────────────────────────────────
from _data_layer import registry
from _data_layer.api import _backend, save_artifact


# ---------------------------------------------------------------------------#
# 🔧 Registry helpers                                                        #
# ---------------------------------------------------------------------------#

def _get_record(aid: str) -> dict:
    for rec in registry.find():
        if registry._entry_id(rec) == aid:
            return rec
    raise click.ClickException(f"❌ Artefact not found: {aid}")


def load_df(embedding_analysis_id: str) -> pd.DataFrame:
    rec = _get_record(embedding_analysis_id)
    data = _backend(rec["backend"]).load(rec["data_ref"])
    if isinstance(data, dict) and "df" in data:
        return data["df"]
    if isinstance(data, pd.DataFrame):
        return data
    raise TypeError("Unsupported clustered_projection format.")


def load_vocab(vocab_id: str) -> List[str]:
    rec = _get_record(vocab_id)
    raw = _backend(rec["backend"]).load(rec["data_ref"])

    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            obj: Any = json.loads(raw)
        except json.JSONDecodeError:
            obj = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    else:
        obj = raw

    if isinstance(obj, list) and (not obj or isinstance(obj[0], str)):
        return [str(t) for t in obj]
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        for k in ("term", "token", "keyword"):
            if k in obj[0]:
                return [str(d[k]) for d in obj]
    if isinstance(obj, dict):
        for k in ("vocab", "terms", "tokens", "keywords"):
            if k in obj and isinstance(obj[k], list):
                inner = obj[k]
                if inner and isinstance(inner[0], dict):
                    for kk in ("term", "token", "keyword"):
                        if kk in inner[0]:
                            return [str(d[kk]) for d in inner]
                return [str(t) for t in inner]
    if isinstance(obj, pd.DataFrame) and "term" in obj.columns:
        return obj["term"].astype(str).tolist()
    if hasattr(obj, "vocabulary_"):
        vocab_dict = obj.vocabulary_
        return [w for w, _ in sorted(vocab_dict.items(), key=lambda kv: kv[1])]

    raise ValueError("Unsupported vocabulary artefact format.")


def load_hdbscan(model_id: str):
    rec = _get_record(model_id)
    return _backend(rec["backend"]).load(rec["data_ref"])


def _attach_conv_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Merge Agente, Cliente, Escenario, messages, timestamps."""
    meta_rows = []
    SAS_EPOCH = datetime(1960, 1, 1, tzinfo=timezone.utc)

    def iso_to_sas(ts: str) -> tuple[str, float]:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
            readable = dt.strftime("%d%b%Y:%H:%M:%S").upper()      # 23APR2025:21:29:34
            numeric  = (dt - SAS_EPOCH).total_seconds()            # 2067790174.2
            return readable, numeric
        except Exception:
            return None, None

    for rec in registry.find(stage="conversation_record"):
        conv_id = rec["conversation_id"]
        agente  = rec.get("agent_name")
        cliente = rec.get("client_name")
        escen   = rec.get("scenario")

        # messages list
        msgs = []
        try:
            with Path(rec["data_ref"]).open(encoding="utf-8") as f:
                msgs = json.load(f).get("messages", [])
        except Exception:
            pass

        cstart_read, cstart_num = iso_to_sas(rec.get("conversation_start", ""))
        created_read, created_num = iso_to_sas(rec.get("created_at", ""))

        meta_rows.append({
            "conversation_id": conv_id,
            "Agente": agente,
            "Cliente": cliente,
            "Escenario": escen,
            "messages": msgs,
            "conversation_start_read": cstart_read,
            "conversation_start_num":  cstart_num,
            "created_at_read": created_read,
            "created_at_num":  created_num,
        })

    meta_df = pd.DataFrame(meta_rows)
    return df.merge(meta_df, on="conversation_id", how="left")

# ───────── Interactive “latest artefact” picker ───────────────────────────
def select_latest_artifact(stage_prefix: str, show: int = 10) -> str:
    """
    List the most-recent artefacts whose `stage` starts with *stage_prefix*
    and let the user pick one (default = newest).

    Returns the selected artefact ID.
    """
    recs = [
        r for r in registry.find()
        if str(r.get("stage", "")).startswith(stage_prefix)
    ]
    if not recs:
        raise click.ClickException(f"❌ No artefacts for stage '{stage_prefix}' found.")

    # Sort newest-first by registry timestamp; fallback to file mtime
    def _rec_time(rec):
        if "created_at" in rec:
            return rec["created_at"]
        try:
            return Path(rec["data_ref"]).stat().st_mtime
        except FileNotFoundError:
            return 0

    recs.sort(key=_rec_time, reverse=True)

    click.echo(f"\n📦 Latest {min(show, len(recs))} artefacts of '{stage_prefix}':")
    for idx, r in enumerate(recs[:show]):
        t = _rec_time(r)
        ts = pd.to_datetime(t, unit="s") if isinstance(t, (int, float)) else t
        click.echo(f"  [{idx}] {registry._entry_id(r)} │ {Path(r['data_ref']).name} │ {ts}")
    sel = input("Select number [0]: ").strip()
    sel_idx = int(sel) if sel else 0
    if sel_idx < 0 or sel_idx >= len(recs[:show]):
        raise click.ClickException("Invalid selection.")
    return registry._entry_id(recs[sel_idx])


# ---------------------------------------------------------------------------#
# 🚀 CLI                                                                     #
# ---------------------------------------------------------------------------#

@click.command(context_settings={"help_option_names": ["-h", "--help"]})

@click.option("--model", "--model-id", default=None,
              help="ID of embedding_analysis artefact (or interactive if omitted).")
@click.option("--vocab", "--vocab-id", default=None,
              help="ID of vocabulary artefact (or interactive if omitted).")

@click.option("--cut-distance", default=0.05, show_default=True, type=float,
              help="λ threshold for HDBSCAN condensed-tree cut.")
@click.option("--top-n", default=10, show_default=True,
              help="Number of top tokens stored per cluster in merged table.")
@click.option("--min-cluster-size", default=5, show_default=True,
              help="Ignore clusters smaller than this.")
@click.option("--save-matrix/--no-save-matrix", default=False, show_default=True,
              help="Save full sparse TF-IDF matrix as topic:tfidf_matrix.")


def main(model: str, vocab: str, cut_distance: float,
         top_n: int, min_cluster_size: int, save_matrix: bool):

    # 1️⃣ Load artefacts
    # Interactive pick-latest if values were not passed
    if not model:
        model = select_latest_artifact("embedding_analysis")
    if not vocab:
        vocab  = select_latest_artifact("vocab")

    df = load_df(model)
    text_col = "clean_text" if "clean_text" in df.columns else "text"
    if "cluster" not in df.columns or text_col not in df.columns:
        raise click.ClickException("DataFrame must contain 'cluster' and text columns.")

    if "embedding" not in df.columns:
        click.echo("⚠️  'embedding' column not found – centroids will be omitted.")
    vocab_tokens = load_vocab(vocab)

    hdbscan_id = _get_record(model)["parameters"].get("hdbscan_model")
    if not hdbscan_id:
        raise click.ClickException("embedding_analysis lacks 'hdbscan_model' reference.")
    hdb_model = load_hdbscan(hdbscan_id)

    # 2️⃣ TF-IDF matrix
    vec = TfidfVectorizer(vocabulary=vocab_tokens, ngram_range=(1, 2))
    X = vec.fit_transform(df[text_col].fillna(""))

    vocab_arr = np.array(vec.get_feature_names_out())
    weights = hdb_model.probabilities_.copy()
    if hasattr(hdb_model, "outlier_scores_"):
        weights *= (1.0 - hdb_model.outlier_scores_)

    # 3️⃣ TF-IDF full table (long format)
    full_records = []
    cluster_ids = np.unique(df["cluster"])
    for cid in cluster_ids:
        members = np.where(df["cluster"].to_numpy() == cid)[0]
        if members.size < min_cluster_size:
            continue
        mean_vec = X[members].multiply(weights[members][:, None]).mean(axis=0).A1
        for word, score in zip(vocab_arr, mean_vec):
            full_records.append({"cluster_id": int(cid), "word": word, "tfidf_score": float(score)})
    tfidf_full_df = pd.DataFrame(full_records)

    # 4️⃣ Top tokens & centroids
    top_tokens_map: Dict[int, List[str]] = {}
    for cid in cluster_ids:
        members = np.where(df["cluster"].to_numpy() == cid)[0]
        if members.size < min_cluster_size:
            continue
        mean_vec = X[members].multiply(weights[members][:, None]).mean(axis=0)
        top_idx = np.asarray(mean_vec).ravel().argsort()[-top_n:][::-1]
        top_tokens_map[int(cid)] = vocab_arr[top_idx].tolist()

    if "embedding" in df.columns:
        emb = np.stack(df["embedding"].to_numpy())
        centroids = {
            int(cid): emb[df.cluster == cid].mean(axis=0).tolist()
            for cid in cluster_ids
        }
        df["cluster_centroid"] = df["cluster"].map(centroids)
    else:
        df["cluster_centroid"] = None

    df["top_tokens"] = df["cluster"].map(lambda cid: top_tokens_map.get(cid, []))

    # 5️⃣ Save artefacts
    params = {
        "top_n": top_n,
        "cut_distance": cut_distance,
        "min_cluster_size": min_cluster_size,
    }
    parents = [model, vocab, hdbscan_id]

    df = _attach_conv_metadata(df)

    merged_art = save_artifact(
        df,
        stage="topic:merged_projection",
        backend="json",
        parameters=params,
        parents=parents,
    )

    # ── NEW: dump a .joblib copy for CAS ──────────────────────────────
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    joblib_path = Path("data/experiments") / f"merged_projection_{timestamp}.joblib"
    joblib_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"df": df}, joblib_path)
    click.echo(f"📦 Joblib for CAS saved → {joblib_path}")
    # ------------------------------------------------------------------





    tfidf_table_art = save_artifact(
        tfidf_full_df,
        stage="topic:tfidf_full_table",
        backend="json",
        parameters=params,
        parents=parents,
    )

    msg = f"\n✅ Saved:\n   • merged_projection → {merged_art.id}\n   • tfidf_full_table  → {tfidf_table_art.id}"
    if save_matrix:
        matrix_path = Path(_backend("file")._mktemp("topic_tfidf_matrix", ".npz"))  # type: ignore[attr-defined]
        sparse.save_npz(matrix_path, X)
        matrix_art = save_artifact(
            str(matrix_path),
            stage="topic:tfidf_matrix",
            backend="file",
            parameters=params,
            parents=parents,
        )
        msg += f"\n   • tfidf_matrix      → {matrix_art.id}"
    click.echo(msg)


# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    main()
