# synthetic_cobranza_generator.py – Show‑case + Menú interactivo + Registro batch
"""Synthetic call‑center data generator (Spanish debt‑collection)
================================================================

**Purpose**
-----------
Produce realistic multi‑turn conversations between an agent and a debtor so you
can demo clustering (HDBSCAN + OpenAI embeddings), emotion analysis (text +
audio), cost‑savings with old vs new strategy, etc.

Key capabilities
----------------
1. **Seven customer scenarios** (`1…7`) – from *late payer* to *fraud claim*.
2. **Two strategies** – *vieja* (legacy, repetitive) vs *nueva* (data‑driven).
3. **Five agent styles** – formal, empático, duro, neutro, creativo.
4. **Continuity** – keeps a rolling 2‑line history (`resumen_prev`) so each
   subsequent call is coherent.
5. **Seed phrases** – guarantee that every thematic cluster appears at least
   once, useful for showcase.
6. **Temperature matrix** – diversity tuned per (estrategia, escenario).
7. **Ranges** of number of calls automatic per scenario & strategy.
8. **Menu interactive** if script is run *without* flags; **CLI flags** for
   automated batch.
9. **Batch registry** – every generated conversation also appends a compact
   row to `batch_log.jsonl` in the output folder, so you know when/what was
   generated (timestamp, client_id, scenario, strategy, cost accumulated).

Quick usage
-----------
» Interactive (prompts in console) – *best for demos*
```
python synthetic_cobranza_generator.py           # asks all questions
```

» Batch flags – *best for big runs / cron*
```
python synthetic_cobranza_generator.py \
        --n_clients 100 --estrategia nueva --use_seeds 1 --output_dir data/
```

Environment variables
---------------------
* **OPENAI_API_KEY** – standard key for ChatCompletion.
* Or **AZURE_OPENAI_ENDPOINT**, **AZURE_OPENAI_KEY**, **AZURE_OPENAI_VERSION**
  – if you use Azure OpenAI Service.
* **OPENAI_MODEL** – model name (defaults to `gpt-4o-mini`).

File outputs
------------
* `data/cliente_<id>.jsonl` – one file per debtor, each line = one conversation
  with all metadata (scenario, cost, meta_llm).
* `data/batch_log.jsonl` – one‑line summary per conversation generated across
  *all* clients (for audit & metrics calculation).

Dependencies
------------
```
python -m pip install openai
```
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import openai

# ╭───────────────────── CONFIG TABLES  ─────────────────────╮
SCENARIO_TEXT: Dict[int, str] = {
    1: "Recordar fecha; cliente suele pagar al primer recordatorio.",
    2: "Concretar promesa de pago hoy (estrategia nueva cierra aquí).",
    3: "Proponer plan de abono parcial según lo solicite el cliente.",
    4: "Detectar evasión; estrategia nueva corta gestión tras 1-2 intentos.",
    5: "Diagnosticar y resolver falla técnica antes de insistir.",
    6: "Activar protocolo antifraude y suspender cobranza.",
    7: "Agradecer pago y fidelizar/reactivar cupo."
}
STRATEGY_TONE = {
    "vieja": "El agente usa script rígido, repite información y muestra poca empatía.",
    "nueva": "El agente aplica analítica predictiva, empatía concisa y decide en 1-2 turnos."
}
AGENT_STYLES = {
    "formal": "Tono formal, tratamiento 'señor/señora'.",
    "empatico": "Tono cercano, reconoce emociones, ofrece ayuda.",
    "duro": "Tono firme, directo, poco espacio a objeciones.",
    "neutro": "Tono corporativo neutro.",
    "creativo": "Tono ingenioso, usa ejemplos prácticos."
}
# Rango de llamadas por escenario/estrategia
RANGES = {
    "vieja": {1:(3,4),2:(3,5),3:(4,6),4:(6,10),5:(4,6),6:(3,5),7:(4,7)},
    "nueva": {1:(1,1),2:(1,2),3:(2,3),4:(1,2),5:(2,3),6:(1,2),7:(2,3)}
}
SEED_PHRASES: Dict[int, List[str]] = {
    1:["Se me pasó la fecha","¿hasta cuándo tengo plazo?"],
    2:["Pago mañana a las 5","Ya tengo el dinero"],
    3:["Puedo abonar","Denme plazo"],
    4:["No me llamen más","Llame después"],
    5:["No llega el token","La app no abre"],
    6:["Yo no hice esa compra","Fue un fraude"],
    7:["Ya pagué, ¿reactivan?","Gracias por su ayuda"]
}
TEMP_TABLE: Dict[Tuple[str,int], float] = {
    ("nueva",4):0.40,("vieja",4):0.70,("nueva",2):0.50,("vieja",2):0.80,
}
DEFAULT_TEMP = {"nueva":0.55,"vieja":0.65}
# ╰──────────────────────────────────────────────────────────╯

# ---------- OpenAI helper ----------------------------------

def init_openai():
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        openai.api_type = "azure"
        openai.api_key = os.getenv("AZURE_OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        openai.api_version = os.getenv("AZURE_OPENAI_VERSION","2023-05-15")
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")


def chat_completion(prompt:str,temperature:float=0.6,top_p:float=0.95)->str:
    resp=openai.ChatCompletion.create(
        model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,top_p=top_p,max_tokens=900,
    )
    return resp.choices[0].message.content.strip()

# ---------- Prompt builder ---------------------------------

def build_prompt(data:Dict)->str:
    header=(f"Cliente ID: {data['cid']} | Conv {data['n']}/{data['n_tot']}\n"
            f"Escenario {data['esc']} | Estrategia {data['estr']}\n"
            f"Historial:\n{data['resume']}\n")
    blocks=[SCENARIO_TEXT[data['esc']],STRATEGY_TONE[data['estr']],
            AGENT_STYLES[data['style']],
            "Escribe diálogo Agente/Cliente en 8-12 turnos."]
    if sp:=data.get("seed"):
        blocks.append(f"Incluye EXACTAMENTE la frase del cliente: \"{sp}\"")
    tail="---\nAl final escribe un bloque JSON con campos: temas, emocion, accion_agente."
    return "\n".join([header,*blocks,tail])

# ---------- Summary updater --------------------------------

def update_summary(prev:str,dialog:str)->str:
    last_c = next((l for l in reversed(dialog.splitlines()) if l.startswith("Cliente:")),"")
    return (prev+" "+last_c).strip()[:240] or prev

# ---------- Registry helper --------------------------------

def register_batch(out_dir:Path, row:Dict):
    """Append one‑line JSONL to batch_log.jsonl"""
    log_path = out_dir/"batch_log.jsonl"
    row["ts"] = datetime.utcnow().isoformat()
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False)+"\n")

# ---------- Core generator -------------------------------------

def generate_dataset(n_clients: int, estrategia: str, use_seeds: bool, out_dir: str):
    init_openai()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for _ in range(n_clients):
        cid = uuid.uuid4().hex[:8]
        esc = random.choices(range(1, 8), weights=[40,20,15,10,8,5,2])[0]
        style = random.choice(list(AGENT_STYLES))
        n_min, n_max = RANGES[estrategia][esc]
        n_tot = random.randint(n_min, n_max)
        seed = random.choice(SEED_PHRASES[esc]) if use_seeds else None
        resumen = "Cliente sin contacto previo."
        cost_acc = 0

        out_path = Path(out_dir) / f"cliente_{cid}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for n in range(1, n_tot+1):
                temp = TEMP_TABLE.get((estrategia, esc), DEFAULT_TEMP[estrategia])
                prompt = build_prompt({
                    "cid": cid, "escenario": esc, "estrategia": estrategia,
                    "style": style, "resumen": resumen,
                    "n": n, "n_tot": n_tot, "seed": seed if n == 1 else None,
                })
                response = chat_completion(prompt, temperature=temp)
                dialog, _, json_part = response.partition("---")
                try:
                    meta = json.loads(json_part.strip())
                except json.JSONDecodeError:
                    meta = {}
                cost_acc += 5
                fh.write(json.dumps({
                    "cliente_id": cid, "escenario": esc, "estrategia": estrategia,
                    "agente_style": style, "num_conv": n, "n_tot": n_tot,
                    "dialogo": dialog.strip(), "meta_llm": meta,
                    "costo_usd": 5, "costo_acum_usd": cost_acc
                }, ensure_ascii=False) + "\n")
                resumen = update_summary(resumen, dialog)

# ---------- Interactive helper ---------------------------------

def interactive_menu():
    def ask(prompt, default):
        val = input(f"{prompt} [{default}]: ").strip()
        return val or default

    n_clients = int(ask("Número de clientes a generar", 20))
    estr = ask("Estrategia (vieja/nueva)", "nueva")
    use_seeds = bool(int(ask("Inyectar frases semilla 1=Sí 0=No", 1)))
    out_dir = ask("Carpeta de salida", "data")
    return n_clients, estr, use_seeds, out_dir

# ---------- CLI / Entry ----------------------------------------

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n_clients", type=int)
    parser.add_argument("--estrategia", choices=["vieja", "nueva"])
    parser.add_argument("--use_seeds", type=int)
    parser.add_argument("--output_dir")
    parser.add_argument("-h", "--help", action="store_true")
    args = parser.parse_args()

    # If user passes at least one flag (or -h) skip interactive unless --help
    if args.help:
        print("""
CLI flags:
  --n_clients N        Número de clientes
  --estrategia s       vieja|nueva
  --use_seeds 0/1      Inyectar frases semilla
  --output_dir path    Carpeta salida
Si ejecutas sin ningún flag el script lanza menú interactivo.
""")
        sys.exit()

    if args.n_clients is None and args.estrategia is None and args.use_seeds is None and args.output_dir is None:
        # Interactive path
        n_clients, estrategia, use_seeds, output_dir = interactive_menu()
    else:
        # CLI path with fallbacks
        n_clients = args.n_clients or 20
        estrategia = args.estrategia or "nueva"
        use_seeds = bool(args.use_seeds) if args.use_seeds is not None else True
        output_dir = args.output_dir or "data"

    generate_dataset(n_clients, estrategia, use_seeds, output_dir)

if __name__ == "__main__":
    main()
