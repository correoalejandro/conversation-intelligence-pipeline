# z-master_df.py  –  python z-master_df.py
from pathlib import Path
import json, pandas as pd
from _data_layer import registry
from _data_layer.api import _backend

def load_artifact(aid):
    rec = next(r for r in registry.find() if r["id"] == aid)
    return _backend(rec["backend"]).load(rec["data_ref"])

# 1) último embedding_analysis
exp = max(
    (r for r in registry.find()
     if str(r.get("stage", "")).startswith("embedding_analysis")),
    key=lambda r: r["created_at"]
)
emb = load_artifact(exp["id"])
if isinstance(emb, dict) and "df" in emb:
    emb = emb["df"]
emb = emb.rename(columns={"cluster": "cluster_label"})

# 2) meta‑info de conversaciones
rows = []
for rec in registry.find(stage="conversation_record"):
    p = Path(rec["data_ref"])
    if not p.exists():
        continue            # salta faltantes
    data = json.loads(p.read_text(encoding="utf-8"))
    rows.append({
        "conversation_id": rec["conversation_id"],
        "Agente":   rec["agent_name"],
        "Cliente":  rec["client_name"],
        "Escenario":rec["scenario"],
        "messages": data.get("messages", [])
    })
meta = pd.DataFrame(rows)

# 3) merge final
df = (
    emb.merge(meta, on="conversation_id", how="left")
       [["Agente", "Cliente", "Escenario", "messages",
         "cluster_label", "umap_x", "umap_y"]]
)

print(df.head())
# df.to_pickle("master_convo_embeddings.pkl")  # export opcional
