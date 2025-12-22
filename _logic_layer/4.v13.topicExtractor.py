#!/usr/bin/env python3
"""
Topic Extractor v0.3.3 – final working version
=============================================

* `--model / --model-id` expects an **embedding_analysis** artefact.
* Auto‑resolves the linked HDBSCAN model via
  `parameters['hdbscan_model']`.
* Only asks for **model** and **vocab**; everything else is inferred.
* Produces `topic:tfidf`, `topic:keywords`, `topic:hierarchy`, and
  `topic:preview` artefacts.
"""
from __future__ import annotations

import json
import sys
sys.path.append("c:/Projects/clasificador_mensajes")
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ───────── Project helpers ────────────────────────────────────────────────
from _data_layer import registry
from _data_layer.api import _backend, save_artifact

# ---------------------------------------------------------------------------
# 🔧 Interactive selection helpers
# ---------------------------------------------------------------------------

def select_artifact(stage_prefix: str, prompt: str) -> str:
    recs = [r for r in registry.find() if str(r.get("stage", "")).startswith(stage_prefix)]
    if not recs:
        raise click.ClickException(f"❌ No artefacts for stage '{stage_prefix}' found in registry.")
    click.echo(f"\n{prompt}")
    for idx, r in enumerate(recs[:20]):
        click.echo(f"  [{idx}] {r['id']} | {Path(r['data_ref']).name}")
    if len(recs) > 20:
        click.echo(f"  … and {len(recs) - 20} more")
    sel = input("Select number [0]: ").strip()
    sel_idx = int(sel) if sel else 0
    if sel_idx < 0 or sel_idx >= len(recs):
        raise click.ClickException("Invalid selection.")
    return recs[sel_idx]["id"]

# ---------------------------------------------------------------------------
# 🔧 Artefact loaders
# ---------------------------------------------------------------------------

def _get_record(aid: str) -> dict:
    """Return registry entry whose primary key matches *aid*."""
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
    raise TypeError("Unsupported clustered_projection format – expected DataFrame or dict with 'df'.")


def load_vocab(vocab_id: str) -> List[str]:
    """
    Return a list of tokens regardless of how the vocabulary artefact
    is stored (JSON, NDJSON, vectorizer, DataFrame…).
    """
    rec = _get_record(vocab_id)
    raw = _backend(rec["backend"]).load(rec["data_ref"])   # bytes | str | obj

    # 1️⃣ bytes  → str
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")

    # 2️⃣ str → try full JSON; fallback to NDJSON -> list[dict]
    if isinstance(raw, str):
        try:
            obj: Any = json.loads(raw)
        except json.JSONDecodeError:
            obj = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    else:
        obj = raw

    # 3️⃣ Recognised formats
    # A) list[str]
    if isinstance(obj, list) and (not obj or isinstance(obj[0], str)):
        return [str(t) for t in obj]

    # B) list[dict] with 'term' / 'token' / 'keyword'
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        for k in ("term", "token", "keyword"):
            if k in obj[0]:
                return [str(d[k]) for d in obj]

    # C) dict with key 'vocab' / 'terms' / 'tokens' / 'keywords'
    if isinstance(obj, dict):
        for k in ("vocab", "terms", "tokens", "keywords"):
            if k in obj and isinstance(obj[k], list):
                inner = obj[k]
                if inner and isinstance(inner[0], dict):
                    for kk in ("term", "token", "keyword"):
                        if kk in inner[0]:
                            return [str(d[kk]) for d in inner]
                return [str(t) for t in inner]

    # D) pandas DataFrame with a 'term' column
    if isinstance(obj, pd.DataFrame) and "term" in obj.columns:
        return obj["term"].astype(str).tolist()

    # E) sklearn vectorizer
    if hasattr(obj, "vocabulary_"):
        vocab_dict = obj.vocabulary_          # type: ignore[attr-defined]
        return [w for w, _ in sorted(vocab_dict.items(), key=lambda kv: kv[1])]

    raise ValueError(
        "Unsupported vocabulary format – expected list[str], list[dict]"
        " with term/token/keyword, dict with vocab|terms|tokens|keywords, "
        "DataFrame with 'term', sklearn vectorizer, or NDJSON of those."
    )



def load_hdbscan(model_id: str):
    rec = _get_record(model_id)
    return _backend(rec["backend"]).load(rec["data_ref"])

# ---------------------------------------------------------------------------
# 🏗  Helper functions
# ---------------------------------------------------------------------------

def build_hierarchy(model, cut_distance: float) -> pd.DataFrame:
    tree = model.condensed_tree_.to_pandas()
    return tree[tree["lambda_val"] >= cut_distance]


def compute_weights(model) -> np.ndarray:
    w = model.probabilities_.copy()
    if hasattr(model, "outlier_scores_"):
        w *= (1.0 - model.outlier_scores_)
    return w


def top_tokens(X, vocab: np.ndarray, idx: np.ndarray, weights: np.ndarray, top_n: int) -> List[str]:
    if idx.size == 0:
        return []
    mean = X[idx].multiply(weights[idx][:, None]).mean(axis=0)
    top_idx = np.asarray(mean).ravel().argsort()[-top_n:][::-1]
    return vocab[top_idx].tolist()

# ---------------------------------------------------------------------------
# 🚀 CLI principal
# ---------------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--model", "--model-id", help="ID of embedding_analysis artefact.")
@click.option("--vocab", "--vocab-id", help="ID of vocabulary artefact.")
@click.option("--cut-distance", default=0.05, show_default=True, type=float,
              help="λ distance threshold for condensed tree cut.")
@click.option("--top-n", default=10, show_default=True, help="Top tokens per cluster.")
@click.option("--min-cluster-size", default=5, show_default=True,
              help="Ignore clusters smaller than this.")
@click.option("--stop-words", default="none", show_default=True,
              help="'none', 'english', or artefact ID / JSON file with list of stop words.")
@click.option("--llm-summary/--no-llm-summary", default=False, show_default=True,
              help="Generate GPT summaries for each topic.")
@click.option("--confirm/--no-confirm", default=True, show_default=True,
              help="Ask confirmation before processing.")

def main(model: str | None, vocab: str | None, cut_distance: float,
         top_n: int, min_cluster_size: int, stop_words: str, llm_summary: bool, confirm: bool):
    # 1️⃣ Interactive selection
    if not model:
        model = select_artifact("embedding_analysis", "Select embedding‑analysis artefact:")
    if not vocab:
        vocab = select_artifact("vocab", "Select vocabulary artefact:")

    # 2️⃣ Resolve HDBSCAN model reference
    exp_rec = _get_record(model)
    hdbscan_id = exp_rec.get("parameters", {}).get("hdbscan_model")
    if not hdbscan_id:
        raise click.ClickException("❌ embedding_analysis lacks 'hdbscan_model' reference. Re‑run pipeline.")

    click.echo("\n🔎 Selections:")
    click.echo(f"   • Embedding analysis → {model}")
    click.echo(f"   • Vocabulary         → {vocab}")
    click.echo(f"   • HDBSCAN model      → {hdbscan_id}")
    if confirm and not click.confirm("Continue?", default=True):
        raise click.Abort()

    # 3️⃣ Load artefacts
    df = load_df(model)
    text_col = "clean_text" if "clean_text" in df.columns else "text"
    if "cluster" not in df.columns or text_col not in df.columns:
        raise click.ClickException("❌ clustered_projection must have 'cluster' and 'text' columns.")

    vocab_tokens = load_vocab(vocab)
    hdb_model = load_hdbscan(hdbscan_id)

    # 4️⃣ Build TF‑IDF matrix
    stopword_list: Optional[List[str]] = None
    if stop_words.lower() == "english":
        stopword_list = "english"
    elif stop_words.lower() not in ("none", ""):
        try:
            stopword_list = load_vocab(stop_words)
        except Exception:
            stopword_list = json.loads(Path(stop_words).read_text())

    vec = TfidfVectorizer(vocabulary=vocab_tokens, ngram_range=(1, 2), stop_words=stopword_list)
    X = vec.fit_transform(df[text_col].fillna(""))
    vocab_arr = np.array(vec.get_feature_names_out())
    weights = compute_weights(hdb_model)

    # 5️⃣ Traverse hierarchy
    hierarchy_df = build_hierarchy(hdb_model, cut_distance)
    topics: Dict[str, Any] = {}
    # ― inside the hierarchy loop ―
    for _, row in hierarchy_df.iterrows():
        cluster_id = int(row["child"])                 # numeric label
        members = np.where(df["cluster"].to_numpy() == cluster_id)[0]
        if members.size < min_cluster_size:
            continue
        tokens = top_tokens(X, vocab_arr, members, weights, top_n)
        topics[row["child"]] = {
            "cluster_id": row["child"],
            "parent": row["parent"],
            "lambda_val": float(row["lambda_val"]),
            "size": int(row["child_size"]),
            "persistence": float(row.get("persistence", 0.0)),
            "top_tokens": tokens,
            "topic_name": "_".join(tokens[:3]) if tokens else f"λ{row['lambda_val']:.2f}",
        }

    # 6️⃣ Save artefacts
    params = {
        "top_n": top_n,
        "cut_distance": cut_distance,
        "min_cluster_size": min_cluster_size,
        "stop_words": stop_words,
    }
    parents = [model, vocab, hdbscan_id]

    vec_art = save_artifact({"vocab": vocab_tokens, "idf": vec.idf_.tolist()},
                            stage="topic:tfidf", backend="json", parameters=params, parents=parents)
    topics_art = save_artifact(topics, stage="topic:keywords", backend="json", parameters=params, parents=parents)
    hier_art = save_artifact(hierarchy_df.to_dict("records"), stage="topic:hierarchy", backend="json", parameters=params, parents=parents)

    preview_txt = "\n".join(
        f"Cluster {t['cluster_id']} (size {t['size']}) → {', '.join(t['top_tokens'])}" for t in topics.values()
    )
    preview_art = save_artifact(preview_txt, stage="topic:preview", backend="json", parameters=params, parents=parents)

    click.echo("\n✅ Artefacts saved:")
    click.echo(f"   • TF‑IDF      → {vec_art.id}")
    click.echo(f"   • Keywords    → {topics_art.id}")
    click.echo(f"   • Hierarchy   → {hier_art.id}")
    click.echo(f"   • Preview     → {preview_art.id}")

    # Optional: placeholder for future GPT summaries
    if llm_summary:
        click.echo("📝 LLM summaries requested – feature not yet implemented.")

# ────────────────────────────────────────────────────────────────────────────
# Main entry-point
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Windows console fix so UTF-8 accents print correctly
    if sys.platform == "win32":
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")

    main()     # ⇐ invokes the @click CLI

