from __future__ import annotations  # debe ser la primera línea de código
# ────────────────────────────── System & Locale Setup ─────────────────────────────
import sys, locale, os
sys.path.append("c:/Projects/clasificador_mensajes")  # access _data_layer.registry helpers
try:
    locale.setlocale(locale.LC_TIME, "Spanish_Spain.1252")  # Windows‑style locale
except locale.Error:
    pass  # ignore if locale not installed
os.system("chcp 65001 > nul")  # console UTF‑8 on Windows cmd/PowerShell

# synthetic_cobranza_generator.py – v3 (estrategia dinámica + roster + config)
"""Generador de conversaciones sintéticas para showcase de cobranza
-------------------------------------------------------------------
Incluye:
1. **Cambio de estrategia automático** según agrupación 1‑14.
2. **Carga de configuración** (`showcase_config.json`) con probabilidades de
   escenarios, estilos y rangos de conversaciones.
3. **Roster de agentes reales** con cuotas y distribuciones de estilos.
4. Registro técnico + business registry.
"""

import argparse, json, os, random, sys, uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import openai, pytz
# Registry helpers (new API but backward‑compatible wrappers still work)
from _data_layer.registry import (
    register_prompt,
    register_batch,
    register_conversations,
    audit_batch,
)


# ──────────────────────────── CONFIG USER JSON ───────────────────────────
CONFIG_FILE = "showcase_config.json"
DEFAULT_CONFIG = {
  "escenarios": {"1":0.12,"2":0.13,"3":0.28,"4":0.23,"5":0.10,"6":0.07,"7":0.07},
  "estrategias": {"vieja":1.0,"nueva_probabilidad_activacion":0.7,"reversion_probabilidad":0.1},
  "estilos_agente": {"formal_directo":0.3,"empatico_adaptativo":0.5,"informativo":0.2},
  "agrupaciones": {"1":0.12,"2":0.15,"3":0.18,"4":0.08,"5":0.07,"6":0.10,"7":0.12,
                    "8":0.06,"9":0.04,"10":0.03,"11":0.02,"12":0.03,"13":0.05,"14":0.03},
  "conversaciones_por_cliente":{"sin_analisis":[5,10],"con_analisis":[1,3]},
  "dias_entre_llamadas":{"sin_analisis":[3,7],"con_analisis":[1,3],"escalado":[1,5]}
}
try:
    CONFIG = json.load(open(CONFIG_FILE, "r", encoding="utf-8"))
except FileNotFoundError:
    CONFIG = DEFAULT_CONFIG

# ─────────────────────────── AGENT ROSTER ────────────────────────────────
AGENT_ROSTER = {
    "Maria_Gonzalez":  {"style_probs": {"empatico":0.7, "formal":0.2, "creativo":0.1}, "quota": 400},
    "Carlos_Ramirez":  {"style_probs": {"duro":0.6, "neutro":0.4},                 "quota": 300},
    "Ana_Uribe":       {"style_probs": {"formal":0.5, "empatico":0.3, "neutro":0.2},"quota": 250},
    "Luis_Perez":      {"style_probs": {"creativo":0.4, "neutro":0.3, "empatico":0.3},"quota": 200},
}
agent_usage = Counter()

def pick_agent_and_style() -> tuple[str,str]:
    candidates = [a for a,i in AGENT_ROSTER.items() if agent_usage[a] < i.get("quota",1e12)]
    if not candidates:
        candidates = list(AGENT_ROSTER)  # reset if quotas excedidas
    agent = random.choice(candidates)
    styles, weights = zip(*AGENT_ROSTER[agent]["style_probs"].items())
    style = random.choices(styles, weights)[0]
    agent_usage[agent]+=1
    return agent, style

# ─────────────────────────── CONST TABLES (escenarios) ───────────────────
SCENARIO_TEXT = {
 1:"Recordar fecha; pago inmediato.",2:"Promesa concreta de pago.",3:"Negociar plan/abono.",
 4:"Evasivo/incobrable.",5:"Fricción operativa (token/app).",6:"Ruta especial / disputa/fraude.",
 7:"Fidelizado post‑pago."
}
STRATEGY_TONE = {"vieja":"Agente con guion rígido","nueva":"Agente analítico y empático",
                 "escala":"Agente escalación soporte","cerrada":"Agente cierra con agradecimiento"}
AGENT_STYLE_PROMPT = {
    "empatico":"Tono cercano y empático.",
    "formal":"Tono formal y respetuoso.",
    "creativo":"Tono ingenioso y ejemplos.",
    "duro":"Tono firme y directo.",
    "neutro":"Tono corporativo neutro."
}

BOGOTA = pytz.timezone("America/Bogota")

# ─────────────────────────── OPENAI / AZURE SETUP ───────────────────────────
import os  # needed for env vars
from openai import OpenAI, AzureOpenAI   # v1 SDK

CLIENT: OpenAI | AzureOpenAI | None = None  # type hint for clarity
MODEL_DEPLOY: str | None = None

# ─────────────────────────── OPENAI / AZURE SETUP ───────────────────────────

def init_openai() -> None:
    """Inicializa un cliente OpenAI (stand‑alone) o AzureOpenAI según variables de entorno."""
    global CLIENT, MODEL_DEPLOY

    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        CLIENT = AzureOpenAI(
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key        = os.getenv("AZURE_OPENAI_API_KEY"),
            api_version    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        )
        MODEL_DEPLOY = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")  # nombre de deployment
    else:
        CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        MODEL_DEPLOY = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def chat(prompt: str, temp: float = 0.6) -> str:
    """Envía un prompt al modelo configurado y devuelve la respuesta limpia."""
    if CLIENT is None:
        raise RuntimeError("CLIENT not initialized: llama primero a init_openai()")

    response = CLIENT.chat.completions.create(
        model       = MODEL_DEPLOY,
        messages    = [{"role": "user", "content": prompt}],
        temperature = temp,
        max_tokens  = 900,
        top_p       = 0.95,
    )
    return response.choices[0].message.content.strip()

# ─────────────────────────── DATE HELPERS ───────────────────────────────
import math

def random_business_dt(days_back:int=180):
    base = datetime.now(BOGOTA)-timedelta(days=random.randint(0,days_back))
    dt   = base.replace(hour=random.randint(8,17),minute=random.randint(0,59),second=random.randint(0,59),microsecond=0)
    return BOGOTA.localize(dt).astimezone(timezone.utc)

def next_business_dt(prev):
    local=prev.astimezone(BOGOTA)+timedelta(minutes=random.randint(30,180))
    if local.hour>=18:
        local=BOGOTA.localize(datetime(local.year,local.month,local.day)+timedelta(days=1, hours=9))
    return local.astimezone(timezone.utc)

# ─────────────────────────── STRATEGY DECISION ──────────────────────────
MAPPING = {
    1:"switch",7:"switch",          # vieja→nueva, posible corte
    2:"keep",6:"keep",13:"keep",   # mantener nueva
    4:"escalar",5:"escalar",9:"escalar",10:"escalar", # escalar/derivar
    8:"restruct",3:"restruct",      # mantener/reestructurar
   14:"close"                        # cierre positivo
}

def decide_strategy(curr:str, agrup:int):
    action=MAPPING.get(agrup,"keep")
    if action=="switch" and curr=="vieja":
        return "nueva"
    if action=="escalar":
        return "escala"
    if action=="close":
        return "cerrada"
    return curr

# ─────────────────────────── PROMPT BUILDER ─────────────────────────────

def build_prompt(cid, n, n_tot, esc, strategy, style, hist, seed=None):
    blocks=[f"Cliente ID:{cid} Conv {n}/{n_tot}",f"Escenario {esc}",SCENARIO_TEXT[esc],STRATEGY_TONE[strategy],AGENT_STYLE_PROMPT[style],"Historial:\n"+hist,"Genera diálogo 8‑12 turnos."]
    if seed: blocks.append(f"Incluye EXACTAMENTE la frase del cliente: \"{seed}\"")
    blocks.append("---\nAl final JSON: {\"agrupacion\":#, \"emocion\":texto}")
    return "\n".join(blocks)

# ─────────────────────────── MAIN GENERATOR ─────────────────────────────

def generate_dataset(n_clients: int, out_dir: str):
    """Genera *n_clients* clientes y registra prompt➜batch➜conversaciones."""
    init_openai()
    out = Path(out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)

    # 1️⃣ ─────────── prompt registration (once per folder) ────────────
    prompt_file = out / "prompt_template.md"
    if not prompt_file.exists():
        prompt_file.write_text("""
# Prompt base — Cobranza sintética
Usado por *synthetic_cobranza_generator.py*.

Variables:
- {{cid}}, {{esc}}, {{estr}}, {{style}}, {{hist}}, {{seed}}
""", encoding="utf-8")
    prompt_id = register_prompt(prompt_path=str(prompt_file),
                                description="Prompt base cobranza sintética",
                                author="generator_v3")

    # 2️⃣ ─────────── batch registration ──────────────────────────────
    batch_id = register_batch(data_ref=str(out),
                              prompt_id=prompt_id,
                              parameters={"n_clients": n_clients})

    business_rows: List[Dict] = []
    for idx in range(1, n_clients + 1):
        cid = uuid.uuid4().hex[:8]
        esc = int(random.choices(list(CONFIG["escenarios"].keys()), weights=list(CONFIG["escenarios"].values()))[0])
        agent_name, style = pick_agent_and_style()
        hist = "Inicio de caso."
        convs: List[Dict] = []
        max_calls = random.randint(*CONFIG["conversaciones_por_cliente"]["con_analisis"]) if random.random() < CONFIG["estrategias"]["nueva_probabilidad_activacion"] else random.randint(*CONFIG["conversaciones_por_cliente"]["sin_analisis"])
        strategy = "vieja"
        dt = random_business_dt()

        for n in range(1, max_calls + 1):
            prompt = build_prompt(cid, n, max_calls, esc, strategy, style, hist)
            resp = chat(prompt, 0.6)
            dialog, json_block = resp.split("---", 1) if "---" in resp else (resp, "{}")
            try:
                meta = json.loads(json_block.strip())
            except json.JSONDecodeError:
                meta = {}
            agrup = int(meta.get("agrupacion", esc))
            prev_strategy = strategy
            strategy = decide_strategy(strategy, agrup)
            if strategy == "cerrada":
                max_calls = n  # terminar cadena
            conv_id = f"{cid}_{n}"
            conv_rec = {
                "conversation_id": conv_id,
                "cliente_id": cid,
                "agent_name": agent_name,
                "agent_style": style,
                "scenario": esc,
                "estrategia": prev_strategy,
                "agrupacion_detectada": agrup,
                "dialogo": dialog.strip(),
                "conversation_start": dt.isoformat(),
                "created_at": dt.isoformat(),
                "data_ref": str(out / f"cliente_{cid}.jsonl")
            }
            convs.append(conv_rec)
            hist = dialog.splitlines()[-1][:240]
            dt = next_business_dt(dt)
        # ─── guardar archivo del cliente ───
        with (out / f"cliente_{cid}.jsonl").open("w", encoding="utf-8") as fh:
            for r in convs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        business_rows.extend(convs)

        # progreso simple
        if idx % 10 == 0 or idx == n_clients:
            print(f"  · {idx}/{n_clients} clientes generados…")

    # 3️⃣ ─────────── conversaciones → business registry ───────────
    register_conversations(batch_id=batch_id, conversations=business_rows)

    print(f"✅ Lote completado ➜ carpeta {out} (batch {batch_id})")

def interactive():
    print("=== Generador sintético de Cobranza (v3) ===")
    n = int(input("¿Cuántos clientes desea generar? [20]: ") or 20)
    out_dir = input("Carpeta de salida [data]: ") or "data"
    generate_dataset(n, out_dir)
    n=int(input("¿Cuántos clientes? [20]:") or 20)
    out=input("Carpeta salida [data]:") or "data"
    generate_dataset(n,out)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--n_clients",type=int);ap.add_argument("--out",default="data");args=ap.parse_args()
    if args.n_clients: generate_dataset(args.n_clients,args.out)
    else: interactive()
