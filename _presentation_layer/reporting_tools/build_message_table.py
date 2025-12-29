#!/usr/bin/env python3
"""
build_message_table.py – genera tabla plana de mensajes y la registra en el registry
"""
from __future__ import annotations
from pathlib import Path
import json, joblib
import pandas as pd
from tqdm import tqdm

from _data_layer import registry, api

OUTDIR = Path("data/experiments"); OUTDIR.mkdir(exist_ok=True, parents=True)

def main():
    rows = []
    conv_recs = registry.find(stage="conversation_record")

    for rec in tqdm(conv_recs, desc="conversations"):
        conv_id  = rec.get("conversation_id")
        batch_id = rec.get("batch_id")
        try:
            obj = json.load(Path(rec["data_ref"]).open("r", encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Error reading {rec['data_ref']}: {exc}")
            continue

        for msg in obj.get("messages", []):
            rows.append({
                "conversation_id": conv_id,
                "batch_id": batch_id,
                "message_id": msg.get("message_id"),
                "sender": msg.get("sender"),
                "text": msg.get("text"),
                "timestamp": msg.get("timestamp"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("❌ No messages extracted – abort.")
        return

    # ── guardar como joblib y registrar ─────────────────────────────
    out_path = OUTDIR / "message_table.joblib"
    joblib.dump({"df": df}, out_path, compress=3)

    art = api.save_artifact(
        df,
        stage="message_table",
        parameters={"source_stage": "conversation_record"},
        backend="joblib",
        parents=list(set(r["batch_id"] for r in conv_recs)),
    )
    print(f"🎉 Message table saved → {out_path} and registered as {art.id}")

if __name__ == "__main__":
    main()
