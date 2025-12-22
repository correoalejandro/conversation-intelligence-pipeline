#!/usr/bin/env python3
"""
tabla_contingencia_cluster_escenario.py
--------------------------------------
Crea e imprime la tabla de contingencia (frecuencia absoluta)
entre cluster_id y Escenario a partir de la tabla UMAP de conversaciones.
"""

import pandas as pd
import json, re
from pathlib import Path
from _data_layer import registry
from _data_layer.api import _backend
from tabulate import tabulate
# ---------- helper para JSON “concatenado” ---------------------
def cargar_json_lista(ruta: Path) -> list[dict]:
    texto = ruta.read_text(encoding="utf-8").strip()
    if not texto.startswith("["):
        texto = "[" + texto
    if not texto.endswith("]"):
        texto = texto + "]"
    texto = re.sub(r",\s*]", "]", texto)
    return json.loads(texto)

# ---------- rutas a artefactos ---------------------------------
DIR_ART = Path("data/artifacts")
KEYWORDS_JSON  = DIR_ART / "artifact_b8028b5f6d.json"

# ---------- 1) tabla base con cluster_label --------------------
experimento = max(
    (r for r in registry.find() if str(r.get("stage", "")).startswith("embedding_analysis")),
    key=lambda r: r["created_at"]
)
emb_df = _backend(experimento["backend"]).load(experimento["data_ref"])
if isinstance(emb_df, dict) and "df" in emb_df:
    emb_df = emb_df["df"]
emb_df = emb_df.rename(columns={"cluster": "cluster_label"})

# ---------- 2) metadatos de conversaciones ---------------------
convs = []
for r in registry.find(stage="conversation_record"):
    p = Path(r["data_ref"])
    if p.exists():
        convs.append({
            "conversation_id": r["conversation_id"],
            "Escenario": r["scenario"]
        })
meta_df = pd.DataFrame(convs)

# ---------- 3) unir embeddings + escenario --------------------
base_df = emb_df.merge(meta_df, on="conversation_id", how="left")

# ---------- 4) crosstab cluster vs escenario -------------------
tabla = pd.crosstab(base_df["cluster_label"], base_df["Escenario"])
print("\n📊 Contingencia cluster_id × Escenario:\n")


pd.set_option("display.max_rows", None)   # (opcional) muestra todas las filas

print(tabulate(tabla, headers="keys", tablefmt="fancy_grid"))
