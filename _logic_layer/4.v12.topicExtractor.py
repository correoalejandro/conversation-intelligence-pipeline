#!/usr/bin/env python3
"""
2.v8.topicExtractor_registry.py – registry‑aware topic extraction (patched)
===========================================================================
* Selecciona **embedding_analysis** (clusters + texto) y **vocabulario** desde
  el registry.
* Calcula TF‑IDF por clúster y extrae los *top‑N tokens*.
* Guarda **tres** artefactos vinculados a ambos padres (experimento + vocab):
    • `topic:tfidf`     → vectorizador `TfidfVectorizer` con vocabulario fijo.
    • `topic:keywords`  → lista de topics (tokens por clúster).
    • `topic:hierarchy` → jerarquía plana de topics.
  Además, genera un `topic:preview` (texto legible).
* Jerarquía **siempre** se genera – ya no existe `--html`.

Uso rápido
----------
$ python 2.v8.topicExtractor_registry.py             # totalmente interactivo
$ python 2.v8.topicExtractor_registry.py --exp EXP_ID --vocab VOC_ID \
        --top-n 15 --min-cluster-size 8
"""
from __future__ import annotations
import json
from typing import List
import json
import sys
sys.path.append("c:/Projects/clasificador_mensajes")
from pathlib import Path
from typing import Any, Dict, List

import click
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 🏗  Helpers de proyecto -----------------------------------------------------
from _data_layer import registry
from _data_layer.api import _backend, save_artifact

# ---------------------------------------------------------------------------
# 🔧 Selección de artefactos
# ---------------------------------------------------------------------------

def _select_artifact(stage_prefix: str, prompt: str) -> str:
    recs = [
        r for r in registry.find()
        if str(r.get("stage", "")) == stage_prefix or str(r.get("stage", "")).startswith(stage_prefix + ":")
    ]
    if not recs:
        raise click.ClickException(f"❌ No artifacts for stage '{stage_prefix}' found in registry.")

    print(f"\n{prompt}")
    for idx, r in enumerate(recs[:20]):
        print(f"  [{idx}] {r['id']} | {Path(r['data_ref']).name}")
    if len(recs) > 20:
        print(f"  … and {len(recs)-20} more")

    sel = input("Select number [0]: ").strip()
    sel_idx = int(sel) if sel else 0
    if sel_idx < 0 or sel_idx >= len(recs):
        raise click.ClickException("Invalid selection.")
    return recs[sel_idx]["id"]


# ---------------------------------------------------------------------------
# 🔧 Carga de artefactos
# ---------------------------------------------------------------------------

def _load_df(artifact_id: str) -> pd.DataFrame:
    rec = next(r for r in registry.find() if r.get("id") == artifact_id)
    data = _backend(rec["backend"]).load(rec["data_ref"])
    if isinstance(data, dict) and "df" in data:
        return data["df"]
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, list):
        return pd.DataFrame(data)
    raise ValueError("Unsupported experiment artifact format; expected DataFrame.")



def _load_vocab(artifact_id: str) -> List[str]:
    """Devuelve la lista de tokens, aceptando string JSON, bytes o objetos ya cargados."""
    rec = next(r for r in registry.find() if r.get("id") == artifact_id)
    raw_obj = _backend(rec["backend"]).load(rec["data_ref"])

    # ── 1) Si llega como bytes → decodificar ────────────────────────────────
    if isinstance(raw_obj, (bytes, bytearray)):
        raw_obj = raw_obj.decode("utf-8")

    # ── 2) Si llega como string JSON → parsear a objeto Python ──────────────
    if isinstance(raw_obj, str):
        try:
            obj = json.loads(raw_obj)
        except json.JSONDecodeError:
            raise ValueError("Vocab string is not valid JSON.")
    else:
        obj = raw_obj

    # ── 3) Formatos admitidos ───────────────────────────────────────────────
    # A) lista de strings
    if isinstance(obj, list) and (not obj or isinstance(obj[0], str)):
        return [str(t) for t in obj]

    # B) lista de dicts [{'term': '...', ...}]
    if isinstance(obj, list) and isinstance(obj[0], dict) and "term" in obj[0]:
        return [str(item["term"]) for item in obj]

    # C) dict con clave "vocab": [...]
    if isinstance(obj, dict) and "vocab" in obj and isinstance(obj["vocab"], list):
        return [str(t) for t in obj["vocab"]]

    # D) sklearn vectorizer serializado
    if hasattr(obj, "vocabulary_"):
        vocab_dict = obj.vocabulary_            # type: ignore[attr-defined]
        return [w for w, _ in sorted(vocab_dict.items(), key=lambda kv: kv[1])]
     # E) pandas DataFrame con columna 'term'
    if isinstance(obj, pd.DataFrame) and "term" in obj.columns:
        return obj["term"].astype(str).tolist()

    # F) dict con clave 'terms' (lista de strings o dicts)
    if isinstance(obj, dict) and "terms" in obj:
        terms = obj["terms"]
        if isinstance(terms, list):
            if terms and isinstance(terms[0], dict) and "term" in terms[0]:
                return [str(d["term"]) for d in terms]
            if not terms or isinstance(terms[0], str):
                return [str(t) for t in terms]
    # ── Nada coincidió ──────────────────────────────────────────────────────
    raise ValueError(
        "Unsupported vocabulary artifact format – expected list[str], "
        "list[dict{'term': ...}], dict{'vocab': [...]}, sklearn vectorizer, "
        "or a JSON/bytes representation of alguno de ellos."
    )



# ---------------------------------------------------------------------------
# 🧠 TF‑IDF y Topics
# ---------------------------------------------------------------------------

def top_tokens_by_cluster(df: pd.DataFrame, texts: pd.Series, cluster_col: str,
                          vec: TfidfVectorizer, top_n: int, min_size: int) -> Dict[int, List[str]]:
    X = vec.fit_transform(texts)
    vocab_arr = np.array(vec.get_feature_names_out())

    clusters = np.unique(df[cluster_col])
    token_map: Dict[int, List[str]] = {}
    for cid in clusters:
        if cid == -1:  # ruido
            continue
        mask = df[cluster_col] == cid
        if mask.sum() < min_size:
            continue
        mean_tfidf = X[mask].mean(axis=0)
        top_idx = np.asarray(mean_tfidf).ravel().argsort()[-top_n:][::-1]
        token_map[int(cid)] = vocab_arr[top_idx].tolist()
    return token_map

# ---------------------------------------------------------------------------
# 🚀 CLI principal
# ---------------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--exp", "--experiment-id", help="ID de artifact embedding_analysis.")
@click.option("--vocab", "--vocab-id", help="ID de artifact vocab.")
@click.option("--top-n", default=10, show_default=True, help="Top tokens por clúster.")
@click.option("--min-cluster-size", default=5, show_default=True,
              help="Ignora clústeres más pequeños.")
@click.option("--confirm/--no-confirm", default=True, show_default=True,
              help="Pide confirmación antes de continuar")



def main(exp: str | None,
         vocab: str | None,
         top_n: int,
         min_cluster_size: int,
         confirm: bool):


    # 1️⃣ Selección interactiva si faltan IDs
    if not exp:
        exp = _select_artifact("embedding_analysis", "Select embedding‑analysis experiment:")
    if not vocab:
        vocab = _select_artifact("vocab", "Select vocabulary artifact:")


        # ----- Confirmación de selección -----
    click.echo("\n🔎 Selections:")
    click.echo(f"   • Experiment ID → {exp}")
    click.echo(f"   • Vocabulary ID → {vocab}")
    if confirm and not click.confirm("¿Continuar con estos artefactos?", default=True):
        click.echo("❌ Aborted by user.")
        raise click.Abort()


    # 2️⃣ Carga de datos y vocabulario
    df = _load_df(exp)
    text_col = "clean_text" if "clean_text" in df.columns else "text"
    if "cluster" not in df.columns or text_col not in df.columns:
        raise click.ClickException("❌ Experiment must have 'cluster' and 'text' columns.")
    vocab_tokens = _load_vocab(vocab)

    # Vectorizador con vocabulario fijo
    vectorizer = TfidfVectorizer(vocabulary=vocab_tokens, ngram_range=(1, 2))

    # 3️⃣ Extracción de topics
    token_map = top_tokens_by_cluster(df, df[text_col].fillna(""), "cluster",
                                      vectorizer, top_n, min_cluster_size)
    topics: List[Dict[str, Any]] = []
    for cid, tokens in token_map.items():
        topics.append({
            "cluster_id": cid,
            "size": int((df["cluster"] == cid).sum()),
            "top_tokens": tokens,
            "topic_name": "_".join(tokens[:3]) if tokens else f"cluster_{cid}",
        })

    # 4️⃣ Jerarquía plana por defecto
    hierarchy = [
        {
            "id": f"cluster_{t['cluster_id']}",
            "label": t["topic_name"],
            "size": t["size"],
            "tokens": t["top_tokens"],
            "parent": None,
        }
        for t in topics
    ]





    '''
    # 5️⃣ Guardar artefactos con línea de procedencia
    params = {"top_n": top_n, "min_cluster_size": min_cluster_size}
    parents = [exp, vocab]

    tfidf_art   = save_artifact(vectorizer, stage="topic:tfidf",    parameters=params, parents=parents)

    topics_art  = save_artifact(topics,     stage="topic:keywords", parameters=params, parents=parents)

    hier_art    = save_artifact(hierarchy,  stage="topic:hierarchy",parameters=params, parents=parents)
    preview_txt = "\n".join(
        f"Cluster {t['cluster_id']} (size {t['size']}): {', '.join(t['top_tokens'])}" for t in topics
    )

    preview_art = save_artifact(preview_txt, stage="topic:preview",  parameters=params, parents=parents)

    '''
    # 5️⃣ Guardar artefactos como JSON
    params  = {"top_n": top_n, "min_cluster_size": min_cluster_size}
    parents = [exp, vocab]

    # --- TF‑IDF vectorizer → dict JSON‑friendly ----------------------
    def _jsonfy(x):
        """Asegura que x sea serializable por json."""
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        return str(x)          # clases, funciones, numpy types, etc. → str()

    vec_dict = {
        "vocabulary": vectorizer.get_feature_names_out().tolist(),
        "idf":        [float(v) for v in vectorizer.idf_],        # numpy → float
        "params":     {k: _jsonfy(v) for k, v in vectorizer.get_params().items()},
    }
    tfidf_art  = save_artifact(vec_dict, stage="topic:tfidf",
                            backend="json", parameters=params, parents=parents)

    # --- Topics, jerarquía y preview ya son JSON‑serializables --------
    topics_art = save_artifact(topics,     stage="topic:keywords",
                            backend="json", parameters=params, parents=parents)

    hier_art   = save_artifact(hierarchy,  stage="topic:hierarchy",
                            backend="json", parameters=params, parents=parents)

    preview_txt = "\n".join(
        f"Cluster {t['cluster_id']} (size {t['size']}): {', '.join(t['top_tokens'])}"
        for t in topics
    )
    preview_art = save_artifact(preview_txt, stage="topic:preview",
                                backend="json", parameters=params, parents=parents)


    # 6️⃣ Mensajes de éxito

    click.echo(f"✅ TF‑IDF vectorizer saved  → {tfidf_art.id}")
    click.echo(f"✅ Topics saved            → {topics_art.id}")
    click.echo(f"✅ Hierarchy saved         → {hier_art.id}")
    click.echo(f"✅ Preview saved           → {preview_art.id}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    main()
