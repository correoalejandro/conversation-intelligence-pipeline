# quick_stats.py
from _data_layer import registry, api
from pathlib import Path
import pandas as pd

def main():
    # 1️⃣  Total de conversaciones registradas
    business = registry.find(stage="conversation_record")
    total_conv = len(business)

    # 2️⃣  Conversaciones vectorizadas  (stage = vectorizer:<conv_id>)
    vect = registry.find()
    vect_ids = {r["stage"].split(":",1)[1]              # conv_id
                for r in vect if str(r.get("stage","")).startswith("vectorizer:")}
    vect_count = len(vect_ids)

    # 3️⃣  Conversaciones usadas en el último embedding_analysis
    exp = registry.latest("embedding_analysis")
    if not exp:
        emb_count = 0
    else:
        bundle = api._backend(exp["backend"]).load(exp["data_ref"])
        df = bundle["df"] if isinstance(bundle, dict) else bundle
        emb_count = df["source_embedding_artifact"].nunique()

    # 👉  Resumen
    print(f"👥  Conversaciones totales           : {total_conv}")
    print(f"🧩  Vectorizadas (vectorizer:*)      : {vect_count}")
    print(f"🔬  En último embedding_analysis    : {emb_count}")

if __name__ == "__main__":
    main()
