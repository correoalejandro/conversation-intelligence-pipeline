from __future__ import annotations
import uuid
"""Synthetic Conversation Generator v11

Adds the extra functionality that existed in v8 but was dropped in v10:
• Scenario catalogue & chooser taken from the markdown prompt.
• Prompt hygiene (system rule + intro‑line stripping).
• Rich message parsing with realistic timestamps and per‑turn delays.
• Batch‑level artefacts: a single JSON (meta + conversations) **and** a human‑readable
  TXT preview for quick inspection.
• File‑locking so concurrent runs cannot corrupt the batch files.

What we intentionally **leave out** (per user request):
• The pure CLI interface (`click`). We keep the interactive menu instead.
• Custom participant labels.
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
import json

import random
import re
import sys, locale, os
import sys
sys.path.append("c:/Projects/clasificador_mensajes")
# 1️⃣ Load the Windows‑style Spanish locale
locale.setlocale(locale.LC_TIME, "Spanish_Spain.1252")   # not es_ES.UTF‑8

# 2️⃣ Switch the current console to UTF‑8 (once per session)
os.system("chcp 65001 > nul")    # cmd / PowerShell

# 3️⃣ Tell Python to emit UTF‑8 to that console
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from datetime import datetime, timezone



from typing import Dict, List, Any, Tuple

from dotenv import load_dotenv
from filelock import FileLock
from openai import AzureOpenAI

# Registry helpers (new API but backward‑compatible wrappers still work)
from _data_layer.registry import (
    register_prompt,
    register_batch,
    register_conversations,
    audit_batch,
)

# ----------------------------
# CONFIGURATION CONSTANTS
# ----------------------------
PROMPTS_DIR = Path("data/prompts/")
DATA_DIR = Path("data/conversations")
BATCH_LOCK = DATA_DIR / "batch.lock"  # for the extra artefact writer

MODEL_BACKEND = "gpt-4o"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_BATCH_SIZE = 50

# Azure OpenAI env vars
load_dotenv()
ENV_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
ENV_KEY = "AZURE_OPENAI_API_KEY"
ENV_DEPLOYMENT = "AZURE_OPENAI_CHAT_DEPLOYMENT"
ENV_API_VERSION = "AZURE_OPENAI_API_VERSION"
DEFAULT_API_VERSION = "2024-02-15-preview"

# Agent/Client name pools (random selection – we keep this)
AGENTS = {"Sofía": 0.4, "Luis": 0.3, "María": 0.2, "Andrés": 0.1}
CLIENTS = {"Carlos": 0.25, "Ana": 0.25, "Javier": 0.25, "Laura": 0.25}

# ----------------------------
# AZURE OPENAI CLIENT HELPER
# ----------------------------

def get_aoai_client() -> AzureOpenAI:
    endpoint = os.getenv(ENV_ENDPOINT)
    key = os.getenv(ENV_KEY)
    deployment = os.getenv(ENV_DEPLOYMENT)
    if not all([endpoint, key, deployment]):
        missing = ", ".join([
            name
            for val, name in [
                (endpoint, ENV_ENDPOINT),
                (key, ENV_KEY),
                (deployment, ENV_DEPLOYMENT),
            ]
            if not val
        ])
        print(f"❌ Missing environment variables: {missing}", file=sys.stderr)
        sys.exit(1)
    return AzureOpenAI(
        api_key=key,
        azure_endpoint=endpoint,
        api_version=os.getenv(ENV_API_VERSION, DEFAULT_API_VERSION),
    )

# ----------------------------
# PROMPT & SCENARIOS
# ----------------------------

def load_prompt_and_scenarios(prompt_path: Path) -> Tuple[str, Dict[int, str]]:
    """Split the markdown prompt into (base_prompt, {n: description}).

    Scenarios are detected after a heading that starts with "###" and contain a
    numbered list like `1. **Label** – description`.
    """
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    base_lines: List[str] = []
    scenarios: Dict[int, str] = {}
    in_scenarios = False
    with prompt_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("###") and "cenario" in line.lower():
                in_scenarios = True
                continue
            if not in_scenarios:
                base_lines.append(line.rstrip("\n"))
                continue
            m = re.match(r"^(\d+)\.\s+\*\*(.*?)\*\*\s+[–-]\s+(.*)$", line.strip())
            if m:
                num = int(m.group(1))
                desc = f"{m.group(2).strip()} – {m.group(3).strip()}"
                scenarios[num] = desc
    if not scenarios:
        # fallback single generic scenario
        scenarios[1] = "Simulación de cobranza"  # default
    base_prompt = "\n".join(base_lines)
    return base_prompt, scenarios

# ----------------------------
# MENU & UTILS
# ----------------------------

def menu_selection(prompt_files: List[str], scenarios: Dict[int, str]) -> Dict[str, Any]:
    print("📋 Conversation Generator Menu")
    print("———————————————")
    for idx, prompt in enumerate(prompt_files):
        print(f"[{idx}] {prompt}")
    prompt_idx = int(input("Select Prompt #: "))
    prompt_file = prompt_files[prompt_idx]

    total_target = int(input("Total Conversations Target [default 1000]: ") or 1000)
    
    
    temperature = DEFAULT_TEMPERATURE
    batch_size = DEFAULT_BATCH_SIZE
    '''
    batch_size = int(input(f"Batch size [{DEFAULT_BATCH_SIZE}]: ") or DEFAULT_BATCH_SIZE)
    temperature = float(input(f"Temperature [{DEFAULT_TEMPERATURE}]: ") or DEFAULT_TEMPERATURE)'''

    # Scenario chooser – 0 means random‑rotate
    print("Available scenarios:")
    for num, desc in scenarios.items():
        print(f"  {num}: {desc}")
    scenario_choice = int(input("Scenario number [0 = random]: ") or 0)

    return {
        "prompt_file": prompt_file,
        "total_conversations": total_target,
        "batch_size": batch_size,
        "temperature": temperature,
        "scenario_choice": scenario_choice,
    }


TIMESTAMP_LINE_RE = re.compile(
    r"""^\[
        (?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)
        \]\s+
        (?P<sender>Agente|Cliente)
        (?:\s*\([^)]+\))?      # alias opcional
        :\s*
        (?P<text>.+)$
    """,
    re.VERBOSE | re.IGNORECASE,
)
# ----------------------------
# PARSING HELPERS (rich messages)
# ----------------------------
def parse_conversation(raw_text: str, debug: bool = False) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for idx, line in enumerate(raw_text.splitlines(), 1):
        line = line.strip()
        if debug:
            print(f"[DEBUG] L{idx}: {line}")
        m = TIMESTAMP_LINE_RE.match(line)
        if not m:
            if debug:
                print(f"[DEBUG] no‑match L{idx}")
            continue
        messages.append(
            {
                "message_id": str(len(messages) + 1),
                "sender": m.group("sender").capitalize(),
                "text": m.group("text").strip(),
                "timestamp": m.group("ts"),
            }
        )
        if debug:
            print(f"[DEBUG] added -> {messages[-1]}")
    return messages

# ----------------------------
# OPENAI CALL & SAVE HELPERS
# ----------------------------

def build_system_messages(base_prompt: str, scenario_desc: str, agent: str, client: str) -> List[Dict[str, str]]:
    system_content = (
        f"{base_prompt}\n\n### Escenario\n{scenario_desc}\n"
        f"Participantes: {agent}, {client}\n"
        "Responde solo con los turnos de la conversación, sin explicaciones."
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "Genera la conversación en formato tipo WhatsApp."},
    ]

def generate_conversation(
    aoai_client: AzureOpenAI,
    agent: str,
    client_name: str,
    base_prompt: str,
    scenario_desc: str,
    temperature: float,
    ) -> Dict[str, Any]:
    
    messages = build_system_messages(base_prompt, scenario_desc, agent, client_name)

    print(f"📞 Generating conversation [{agent} ↔ {client_name}]…")
    response = aoai_client.chat.completions.create(
        model=os.environ[ENV_DEPLOYMENT],
        messages=messages,
        temperature=temperature,
        max_tokens=800,
    )
    raw_text = response.choices[0].message.content.strip()

    start_ts = datetime.now(timezone.utc) - timedelta(days=random.randint(1, 180))

    parsed =  parse_conversation(raw_text, debug=False)

    return {
        "conversation_id": f"conversation_{uuid.uuid4().hex[:10]}",
        "conversation_start": start_ts.isoformat(),
        "agent": agent,
        "client": client_name,
        "scenario": scenario_desc,
        "raw": raw_text,
        "messages": parsed,
    }

# ----------------------------
# BATCH‑LEVEL ARTEFACTS (extra JSON + TXT preview)
# ----------------------------

def write_batch_files(batch_folder: Path, batch_id: str, conversations: List[Dict[str, Any]],
                      scenario_mode: str, model: str, temperature: float) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = batch_folder / f"{ts}_batch.json"
    txt_path = batch_folder / f"{ts}_batch.txt"

    meta = {
        "batch_id": batch_id,
        "scenario_selection": scenario_mode,
        "generation_model": model,
        "generation_temperature": temperature,
        "batch_size": len(conversations),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    with FileLock(str(BATCH_LOCK)):
        # JSON artefact
        with open(json_path, "w", encoding="utf-8") as fj:
            json.dump({"meta": meta, "conversations": conversations}, fj, ensure_ascii=False, indent=2)

        # TXT preview artefact
        with open(txt_path, "w", encoding="utf-8") as ft:
            for conversation in conversations:
                ft.write(f"--- Conversation {conversation['conversation_id']} (Scenario) ---\n")
                for message in conversation["messages"]:
                    ts_human = datetime.strptime(message["timestamp"], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")

                    ft.write(f"[{ts_human}] {message['sender']}: {message['text']}\n")
                ft.write("\n")
        print(f"✅ Batch artefacts saved:\n  • {json_path}\n  • {txt_path}")

# ----------------------------
# MAIN PIPELINE
# ----------------------------

def main():
    # --- prompt selection & registration ---
    prompt_files = [f.name for f in PROMPTS_DIR.glob("*.md")]
    if not prompt_files:
        print("❌ No prompt files found.")
        return

    # Temporarily show first prompt to extract scenarios for the menu
    tmp_prompt_path = PROMPTS_DIR / prompt_files[0]
    base_prompt_tmp, scenarios = load_prompt_and_scenarios(tmp_prompt_path)


    params:Dict = menu_selection(prompt_files, scenarios)
    prompt_path = PROMPTS_DIR / params["prompt_file"]

    # Load prompt & scenarios for real
    base_prompt, scenarios = load_prompt_and_scenarios(prompt_path)

    # Register prompt (returns prompt_id)
    prompt_id = register_prompt(
        prompt_path=str(prompt_path),
        description=f"Synthetic cobranza prompt – {prompt_path.name}",
        tags=["cobranza", "simulación", "español"],
        author="synthetic‑generator",
    )

    aoai_client = get_aoai_client()

    # --- generation loop ---
    total_generated = 0
    batch_counter = 0
    while total_generated < params["total_conversations"]:
        print(f"\n🚀 Generating Batch #{batch_counter}")

        # create a folder for this batch *before* registering it
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        batch_folder = DATA_DIR / f"batch_{timestamp}"
        batch_folder.mkdir(parents=True, exist_ok=True)

        batch_parameters = {
            "batch_size": params["batch_size"],
            "temperature": params["temperature"],
            "backend": MODEL_BACKEND,
            "timestamp": timestamp,           # ensures uniqueness
            "prompt_id": prompt_id,           # optional: adds lineage clarity
        }


        batch_id = register_batch(
            data_ref=str(batch_folder),
            prompt_id=prompt_id,
            parameters=batch_parameters,
        )

        conversations: List[Dict[str, Any]] = []
        scenario_keys = sorted(scenarios.keys())
        scenario_choice = params["scenario_choice"]
        scenario_cycle: List[int]
        if scenario_choice == 0:
            # auto‑rotate through scenarios
           scenario_cycle = random.choices(scenario_keys, k=params["batch_size"])

        else:
            if scenario_choice not in scenarios:
                print(f"❌ Invalid scenario: {scenario_choice}")
                return
            scenario_cycle = [scenario_choice] * params["batch_size"]

        for i in range(params["batch_size"]):
            if total_generated >= params["total_conversations"]:
                print("✅ Conversation target reached in this batch.")
                break

            agent = random.choices(list(AGENTS.keys()), weights=AGENTS.values())[0]
            client_name = random.choices(list(CLIENTS.keys()), weights=CLIENTS.values())[0]
            scn_num = scenario_cycle[i]
            scenario_desc = scenarios[scn_num]

            # --- Verbose interactive pause (kept as requested) ---
            print("\n🚀 Preparing OpenAI call with these parameters:")
            print(f"Prompt ID: {prompt_id}")
            print(f"Batch ID: {batch_id}")
            print(f"Agent: {agent}")
            print(f"Client: {client_name}")
            print(f"Scenario {scn_num}: {scenario_desc}")
            print(f"Temperature: {params['temperature']}")
            print(f"Current Batch Size: {params['batch_size']}")
           

            conversation = generate_conversation(
                aoai_client,
                agent,
                client_name,
                base_prompt,
                scenario_desc,
                params["temperature"],
            )
            # synthetic timestamp at batch level is inside each message already; keep a batch timestamp too
            conversation["created_at"] = datetime.now(timezone.utc).astimezone().strftime("%A, %d de %B de %Y, %H:%M:%S")

            conversations.append(conversation)
            total_generated += 1

        # --- persistence & registry ---
        print("💾 Saving individual conversation files …")
        for c in conversations:
            filename = f"{c['conversation_id']}.json"
            file_path = batch_folder / filename
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(c, f, indent=2, ensure_ascii=False)
            c["data_ref"] = file_path.as_posix()
            c["filename"] = filename

        print("📝 Registering conversations …")
        conversation_metadata_list = [
            {
                "conversation_id": c["conversation_id"],
                "conversation_start": c["conversation_start"],
                "batch_id": batch_id,
                "agent_name": c["agent"],
                "client_name": c["client"],
                "scenario": c["scenario"],
                
                "created_at": c["created_at"],
                "data_ref": c["data_ref"],
            }
            for c in conversations
        ]
        register_conversations(batch_id, conversation_metadata_list)

        # Extra artefacts (JSON + TXT preview)
        write_batch_files(
            batch_folder,
            batch_id,
            conversations,
            "auto" if params["scenario_choice"] == 0 else str(params["scenario_choice"]),
            os.environ[ENV_DEPLOYMENT],
            params["temperature"],
        )

        # Audit – same helper
        audit_report = audit_batch(batch_id)
        print(f"📋 Audit report for {batch_id}: {audit_report['status']}")
        if audit_report.get("issues"):
            for issue in audit_report["issues"]:
                print(f"⚠️  {issue}")
        else:
            print("✅ Batch passed metadata audit!")

        batch_counter += 1

    print(f"🎉 Total conversations generated: {total_generated}/{params['total_conversations']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Generation interrupted by user.")
