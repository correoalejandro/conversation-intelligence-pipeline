# lineage_topics.py  – ejecuta:  python lineage_topics.py
from pathlib import Path
import sys

# Asegúrate de que _data_layer está en PYTHONPATH si usas un módulo externo:
# sys.path.append(str(Path("C:/Projects/clasificador_mensajes/_data_layer").resolve()))

from _data_layer.registry import find, lineage   # tu módulo recién subido

def main():
    # 1️⃣ localiza cualquier artefacto cuyo stage empiece por "topic:"
    topic_recs = [
        r for r in find()
        if isinstance(r.get("stage"), str) and r["stage"].startswith("message")
    ]

    if not topic_recs:
        print("❌ No hay artefactos de tópicos registrados todavía.")
        return

    # 2️⃣ muestra una tabla mínima para elegir
    print("\n👓 Artefactos de tópicos encontrados:")
    for idx, rec in enumerate(topic_recs):
        stamp = rec.get("created_at", "sin‑fecha").split(",")[0]
        print(f"[{idx}] {rec['id']:<35}  {rec['stage']:<15}  {stamp}")

    sel = input("\nSelecciona el número del artefacto [0]: ").strip()
    idx = int(sel) if sel else 0
    aid = topic_recs[idx]["id"]

    # 3️⃣ imprime el árbol de procedencia con hasta 5 niveles
    print("\n📜 Lineage:")
    print(lineage(aid, max_depth=5))

if __name__ == "__main__":
    main()
