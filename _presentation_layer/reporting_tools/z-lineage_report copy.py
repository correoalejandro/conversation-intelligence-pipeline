# z-lineage_vectorizers.py   —  ejecútalo:  python z-lineage_vectorizers.py
from pathlib import Path
import sys, argparse
from _data_layer.registry import find, lineage   # ← usa tu módulo

def main():
    ap = argparse.ArgumentParser(
        description="Imprime el árbol de progenitores de todos los vectorizer:conversation_*")
    ap.add_argument("--contains", "-c", metavar="SUBSTR", help="Filtra IDs que contengan esta cadena")
    ap.add_argument("--max-depth", "-d", type=int, default=4, help="Profundidad máxima del árbol")
    args = ap.parse_args()

    # 1️⃣ Localiza los artefactos vectorizer:conversation_*
    vec_recs = [
        r for r in find()
        if str(r.get("stage", "")).startswith("vectorizer:conversation")
           and (args.contains.lower() in r["id"].lower() if args.contains else True)
    ]

    if not vec_recs:
        print("❌ No se encontraron artefactos vectorizer:conversation.")
        sys.exit(1)

    print(f"🔎 {len(vec_recs)} artefactos encontrados:\n")

    # 2️⃣ Para cada uno, imprime el lineage
    for rec in vec_recs:
        print(f"🗂  {rec['id']}  ({rec['stage']})")
        print(lineage(rec["id"], max_depth=args.max_depth))
        print("-" * 60)

if __name__ == "__main__":
    # Asegúrate de que _data_layer esté en PYTHONPATH si tu proyecto no usa paquetes
    # sys.path.append(str(Path("C:/Projects/clasificador_mensajes/_data_layer").resolve()))
    main()
