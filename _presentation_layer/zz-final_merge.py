import pandas as pd
import json
import re
from pathlib import Path

from datetime import datetime, timezone
import joblib
from pathlib import Path

from _data_layer import registry
from _data_layer.api import _backend

# ------------------------------------------------------------------
# Función robusta: carga lista de objetos JSON incluso si el archivo
# no empieza con '[' ni termina con ']', o si hay comas sobrantes.
# ------------------------------------------------------------------
def cargar_json_lista(ruta_archivo: Path) -> list[dict]:
    texto = ruta_archivo.read_text(encoding="utf-8").strip()

    # Añade corchetes si faltan
    if not texto.startswith("["):
        texto = "[" + texto
    if not texto.endswith("]"):
        texto = texto + "]"

    # Elimina comas finales antes del cierre ']'
    texto = re.sub(r",\s*]", "]", texto)

    return json.loads(texto)

# ------------- 1) Rutas a artefactos ----------------------------------------
DIRECTORIO_ARTEFACTOS = Path("data/artifacts")
RUTA_KEYWORDS  = DIRECTORIO_ARTEFACTOS / "artifact_b8028b5f6d.json"
RUTA_HIERARCHY = DIRECTORIO_ARTEFACTOS / "artifact_a64eafb89f.json"
RUTA_VOCABULARIO = DIRECTORIO_ARTEFACTOS / "artifact_74b7dfa7c6.json"

# ------------- 2) Tabla base: conversaciones + UMAP + clúster ---------------
# 2.1 Experimento embedding_analysis más reciente
experimento = max(
    (reg for reg in registry.find() if str(reg.get("stage", "")).startswith("embedding_analysis")),
    key=lambda reg: reg["created_at"]
)
datos_emb = _backend(experimento["backend"]).load(experimento["data_ref"])
if isinstance(datos_emb, dict) and "df" in datos_emb:
    datos_emb = datos_emb["df"]
tabla_embedding = datos_emb.rename(columns={"cluster": "cluster_label"})

# 2.2 Metadatos de conversaciones
lista_conversaciones = []

for reg in registry.find(stage="conversation_record"):
    ruta_json_conv = Path(reg["data_ref"])
    if ruta_json_conv.exists():
        datos_conv = json.loads(ruta_json_conv.read_text(encoding="utf-8"))
        
        # Convert timestamps to SAS-friendly format
        conv_start = reg.get("conversation_start", "")
        created_at = reg.get("created_at", "")
        
        SAS_EPOCH = datetime(1960, 1, 1, tzinfo=timezone.utc)

        # 🕓 Format 1 – SAS-compatible *readable* timestamp
        # Output: "23APR2025:21:29:34"
        def format_ts_readable(ts):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
                return dt.strftime("%d%b%Y:%H:%M:%S").upper()
            except Exception as e:
                print(f"Error parsing (readable): {ts} — {e}")
                return None

        # 🧮 Format 2 – SAS datetime *numeric* value (seconds since 1960-01-01T00:00:00 UTC)
        # Output: 2067790174.227827
        def format_ts_numeric(ts):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
                delta = dt - SAS_EPOCH
                return delta.total_seconds()
            except Exception as e:
                print(f"Error parsing (numeric): {ts} — {e}")
                return None

    lista_conversaciones.append({
        "conversation_id": reg["conversation_id"],
        "conversation_start": format_ts_readable(conv_start),
        "Agente": reg["agent_name"],
        "Cliente": reg["client_name"],
        "Escenario": reg["scenario"],
        "messages": datos_conv.get("messages", []),
        "raw": datos_conv.get("raw", ""),
        "created_at": format_ts_readable(created_at),
    })
    tabla_conversaciones = pd.DataFrame(lista_conversaciones)

# 2.3 Unir embeddings con metadatos
tabla_base = tabla_embedding.merge(tabla_conversaciones, on="conversation_id", how="left")

# ------------- 3) Cargar tópico:keywords y tópico:hierarchy -----------------
lista_keywords = cargar_json_lista(RUTA_KEYWORDS)
tabla_topics_keywords = pd.DataFrame(lista_keywords)
tabla_topics_keywords["cluster_id"] = tabla_topics_keywords["cluster_id"].astype(int)

lista_hierarchy = cargar_json_lista(RUTA_HIERARCHY)
tabla_topics_hierarchy = pd.DataFrame(lista_hierarchy)
tabla_topics_hierarchy["cluster_id"] = (
    tabla_topics_hierarchy["id"].str.replace("cluster_", "").astype(int)
)

# ------------- 4) Cargar vocabulario TF‑IDF (opcional) ----------------------
datos_vocab = json.loads(RUTA_VOCABULARIO.read_text(encoding="utf-8"))
tabla_vocabulario = pd.DataFrame({
    "term": datos_vocab["vocabulary"],
    "idf": datos_vocab["idf"]
})

# ------------- 5) Crear tabla UMAP (una fila por conversación) --------------
tabla_umap_conversacion = (
    tabla_base
    .merge(tabla_topics_keywords[["cluster_id", "topic_name"]],
           left_on="cluster_label", right_on="cluster_id", how="left")
    [["conversation_id","conversation_start","Agente", "Cliente", "Escenario", "messages", "raw", "created_at",
      "cluster_label", "cluster_id", "topic_name", "umap_x", "umap_y"]]
)

# ------------- 6) Crear tabla Topics (una fila por clúster) -----------------
tabla_topics_cluster = (
    tabla_topics_keywords
    .merge(tabla_topics_hierarchy[["cluster_id", "label", "parent"]],
           on="cluster_id", how="left")
    [["cluster_id", "topic_name", "top_tokens", "size", "label", "parent"]]
)

# ------------- 7) Mostrar cabeceras -----------------------------------------
print("🎯 Tabla UMAP (una fila por conversación):")
print(tabla_umap_conversacion.head(), end="\n\n")

print("🧠 Tabla Topics (una fila por clúster):")
print(tabla_topics_cluster.head())
print("UMAP por conversación:", tabla_umap_conversacion.columns.tolist())
print("Tópicos por clúster:", tabla_topics_cluster.columns.tolist())



# Timestamp único para ambos
timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

# Directorio de salida
output_dir = Path("data/experiments")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Conversaciones UMAP
path_umap = output_dir / f"umap_conversaciones_{timestamp}.joblib"
joblib.dump({"df": tabla_umap_conversacion}, path_umap)
print(f"📁 Conversaciones guardadas en: {path_umap}")


'''
# 2. Tópicos por clúster
path_topics = output_dir / f"topics_cluster_{timestamp}.joblib"
joblib.dump({"df": tabla_topics_cluster}, path_topics)
print(f"📁 Tópicos guardados en: {path_topics}")'''