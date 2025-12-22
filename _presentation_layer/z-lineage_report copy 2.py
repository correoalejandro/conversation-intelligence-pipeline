# z-convo_diversity.py   –  python z-convo_diversity.py
from pathlib import Path
import json, collections, importlib.util, sys
import pandas as pd   # ✅ útil para un vistazo tabular rápido

# --- 1. Carga registry.py dinámicamente ---
spec = importlib.util.spec_from_file_location(
    "reg", Path("_data_layer/registry.py").resolve()
)
reg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reg)

# --- 2. Obtiene todas las conversaciones registradas ---
convs = reg.find(stage="conversation_record")   # ajusta si tu stage difiere

agents     = collections.Counter()
clients    = collections.Counter()
scenarios  = collections.Counter()
empty_msgs = 0
rows       = []

for rec in convs:
    agents[rec["agent_name"]]     += 1
    clients[rec["client_name"]]   += 1
    scenarios[rec["scenario"]]    += 1

    # lee el JSON asociado para verificar mensajes vacíos
    try:
        data = json.loads(Path(rec["data_ref"]).read_text(encoding="utf-8"))
        if not data.get("messages"):
            empty_msgs += 1
    except Exception as e:
        print(f"⚠️  No pude leer {rec.get('data_ref')}: {e}", file=sys.stderr)

    rows.append({
        "conversation_id": rec["conversation_id"],
        "agent":   rec["agent_name"],
        "client":  rec["client_name"],
        "scenario": rec["scenario"],
        "messages_empty": int(not data.get("messages", []))
    })

# --- 3. Resultados por consola ---
total = len(convs)
print(f"\n🧮 Conversaciones totales: {total}")
print(f"📭 Vacías (messages == []): {empty_msgs}  ({empty_msgs/total:.1%})\n")

def _pretty(counter, label):
    print(f"{label} únicos: {len(counter)}")
    for k, n in counter.most_common():
        print(f"   {k:<30} {n}")
    print()

_pretty(agents,   "👥 Agentes")
_pretty(clients,  "🙋 Clientes")
_pretty(scenarios,"🎯 Escenarios")

# --- 4. DataFrame opcional ---
df = pd.DataFrame(rows)
# • Si corres en Jupyter o VS Code → display(df.head())
# • O exporta: df.to_csv("convo_diversity.csv", index=False)
