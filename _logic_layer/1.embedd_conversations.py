"""Embed conversations from Business Registry (v11.3)
====================================================

Genera **un artifact de embedding por conversación** (`vectorizer:<conversation_id>`) a
partir de los JSON listados en el *Business Registry* con
`stage == "conversation_record"`, conservando la funcionalidad de deduplicación,
logs, checkpoints y `--force`.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
sys.path.append("c:/Projects/clasificador_mensajes")

import joblib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging




import click
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from _data_layer import api, registry
from _data_layer.paths import LOG_DIR, ARTIFACT_DIR
# ---------------------------------------------------------------------------
# 🔧 Config
# ---------------------------------------------------------------------------

load_dotenv()
CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_BATCH = 512
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger(__name__)
# ─────────────────────────────────────────────────────────────
# Helper: remove conversations already vectorized with the model
# ─────────────────────────────────────────────────────────────
def filter_pending_conversations(
    conversation_records: list[dict],
    model: str,
    *,
    force: bool = False,
) -> list[dict]:
    """
    Return only conversation_records that still need embedding.

    Parameters
    ----------
    conversation_records : list of conversation_record dicts
    model                : embedding model about to be used
    force                : if True, skip deduplication and return unchanged
    """
    if force:
        return conversation_records

    seen: set[tuple[str, str]] = set()
    for art in registry.find(stage="vectorizing_text", source="technical"):
        model_used = art.get("parameters", {}).get("model")
        for ref in art.get("vector_refs", []):
            conv_id = ref.get("conversation_id")
            if conv_id and model_used:
                seen.add((conv_id, model_used))

    return [
        record for record in conversation_records
        if (record["conversation_id"], model) not in seen
    ]


def interactive_embedding_menu(
    default_model: str,
    default_batch_size: int,
    default_limit: int | None,
    default_force: bool,
    default_verbose: bool,
) -> tuple[str, int | None, str, int, bool, bool]:
    """
    Menu UI for selecting batch and embedding params.
    Returns: (batch, limit, model, batch_size, force, verbose)
    """
    # 1️⃣ Get all business records
    all_conversation_records = load_conversations_from_registry(verbose=False)

    # 2️⃣ Get model choice early (so we can filter by it)
    model = click.prompt("Embedding model", default=default_model)

    # 3️⃣ Dedup filter using technical registry
    conversation_records = filter_pending_conversations(
        all_conversation_records, model, force=default_force
    )

    # 4️⃣ Count unembedded per batch
    pending: Dict[str, int] = {}
    for r in conversation_records:
        b = r["batch_id"]
        pending[b] = pending.get(b, 0) + 1

    total_pending = sum(pending.values())
    if total_pending == 0:
        click.secho("🎉 All conversations are already embedded!", fg="green")
        sys.exit(0)

    # 5️⃣ Show choices and counts
    batch_choices = sorted(pending)
    click.echo("\nPending batches:")
    for b in batch_choices:
        click.echo(f"  • {b}   ({pending[b]} convs remain)")
    click.echo(f"\n🧮 Total unembedded conversations: {total_pending}\n")

    if click.confirm("🔍 Preview unembedded conversations before selecting a batch?", default=False):
        click.echo("\nUnembedded conversation IDs by batch:")
        for batch_id in batch_choices:
            ids = [
                r["conversation_id"]
                for r in conversation_records
                if r["batch_id"] == batch_id
            ]
            click.echo(f"• {batch_id} ({len(ids)}): " + ", ".join(ids))
        click.echo("")  # spacing

    # 6️⃣ Ask for batch and other options
    numbered_batches = list(enumerate(batch_choices, start=1))
    click.echo("📦 Available batches:")
    click.echo(f"  [0] All batches ({total_pending} convs)")
    for i, b in numbered_batches:
        click.echo(f"  [{i}] {b} ({pending[b]} conv)")

    index = click.prompt(
        f"\nSelect an option [0–{len(batch_choices)}]",
        type=int,
        default=0,
    )

    if index == 0:
        selected_conversation_records = conversation_records
        batch = None  # handled downstream
    elif 1 <= index <= len(batch_choices):
        batch = batch_choices[index - 1]
        selected_conversation_records = [
            r for r in conversation_records if r["batch_id"] == batch
        ]
    else:
        click.secho("❌ Invalid selection.", fg="red")
        sys.exit(1)

    max_convs = len(selected_conversation_records)
    limit_str = click.prompt(
        f"Limit (# conversations, 0 = no limit) [max {max_convs}]",
        default=str(default_limit or 0),
    )
    limit = int(limit_str) or None

    batch_size = click.prompt("OpenAI batch size", default=default_batch_size, type=int)
    force   = click.confirm("Force re-embed even if already embedded?", default=default_force)
    verbose = click.confirm("Verbose logging?", default=default_verbose)

    # 7️⃣ Confirm
    max_convs = len(selected_conversation_records)
    todo = min(max_convs, limit or max_convs)

    batch_label = batch or "ALL-BATCHES"
    click.echo(
        f"\n📋 About to embed {todo} conversation(s) "
        f"from '{batch_label}' with model '{model}'."
    )
    if not click.confirm("Proceed?", default=True):
        click.echo("Cancelled.")
        sys.exit(0)

    return selected_conversation_records, batch, limit, model, batch_size, force, verbose



# ---------------------------------------------------------------------------
# 🛠 Registry load iterator
# ---------------------------------------------------------------------------

def load_conversations_from_registry(batch_id: str | None = None, *, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Return list of rows with conversation_id, batch_id, data_ref, raw_text,
    reading conversations directly from the registry (stage='conversation_record').
    """
    reg = registry.find(stage="conversation_record")

    rows: List[Dict[str, Any]] = []
    for rec in reg:
        if batch_id and rec.get("batch_id") != batch_id:
            continue
        data_ref = rec.get("data_ref", "")
        if not (data_ref.endswith(".json") and Path(data_ref).exists()):
            if verbose:
                print(f"✖ skipping missing {data_ref}")
            continue
        try:
            with Path(data_ref).open("r", encoding="utf-8") as f:
                obj = json.load(f)
            raw = obj.get("raw")
            if not raw and obj.get("messages"):
                raw = "\n".join(m.get("text", "") for m in obj["messages"])
            if not raw:
                if verbose:
                    print(f"⚠ no text in {data_ref}")
                continue
            rows.append({
                "conversation_id": rec.get("conversation_id"),
                "batch_id": rec.get("batch_id"),
                "data_ref": data_ref,
                "text": raw,
            })
        except Exception as exc:  # pylint: disable=broad-except
            if verbose:
                print(f"💥 error reading {data_ref}: {exc}")
    return rows

# ---------------------------------------------------------------------------
# 🔧 Helpers
# ---------------------------------------------------------------------------

def _stable_conv_id(text: str, conv_id: str | None) -> str:
    return conv_id or hashlib.sha256(text.encode()).hexdigest()


def _embed_texts(texts: List[str], model: str, openai_batch: int) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(texts), openai_batch):
        chunk = texts[i : i + openai_batch]
        resp = CLIENT.embeddings.create(model=model, input=chunk)
        out.extend([d.embedding for d in resp.data])
    return out

def vectorize_given_conversations(
    conversation_records: list[dict],
    model: str,
    openai_batch_size: int,
    checkpoint: int = 100,
    batch_label: str = "multi",
    verbose: bool = False,
) -> str:
    """
    Embed, checkpoint, and log like v6.
    """
    if not conversation_records:
        print("👍 All conversations already embedded.")
        return ""

    # ── 0.  Set up log file ───────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"embedding_{batch_label}_{ts}.log"

    with log_path.open("w", encoding="utf8") as log_f:

        def log(msg: str):
            now = datetime.now().isoformat(timespec="seconds")
            log_f.write(f"{now} {msg}\n")
            log_f.flush()
            if verbose:
                print(msg)

        log(f"▶️  START  | model={model} | records={len(conversation_records):,}")

        # ── 1.  Prepare data ──────────────────────────────────────────────
        texts     = [r["text"] for r in conversation_records]
        conv_ids  = [_stable_conv_id(r["text"], r["conversation_id"]) for r in conversation_records]
        embeddings, new_rows = [], []

        try:
            for i in tqdm(
                range(0, len(texts), openai_batch_size),
                desc="🔄 Embedding",
                unit="batch",
                disable=not verbose,
                ncols=80,
            ):
                chunk   = texts[i : i + openai_batch_size]
                result  = _embed_texts(chunk, model, openai_batch_size)
                embeddings.extend(result)

                for cid, emb in zip(conv_ids[i : i + openai_batch_size], result):
                    new_rows.append({"conversation_id": cid, "embedding": emb})

                # Check-point every *checkpoint* rows
                if checkpoint and len(new_rows) >= checkpoint:
                    df_ck = pd.DataFrame(new_rows)
                    ck_path = ARTIFACT_DIR / f"vec_ckpt_{batch_label}_{ts}_{i}.joblib"
                    joblib.dump(df_ck, ck_path)
                    log(f"💾 Checkpoint: {len(df_ck):,} rows → {ck_path.name}")
                    new_rows.clear()

        except KeyboardInterrupt:
            log("⚠️  INTERRUPTED by user — saving what we have…")
        except Exception as exc:
            log(f"💥 ERROR: {exc} — saving what we have…")
            raise

        # ── 2.  Final flush (always runs before leaving *with*) ───────────
        if new_rows:
            df_ck = pd.DataFrame(new_rows)
            ck_path = ARTIFACT_DIR / f"vec_final_{batch_label}_{ts}.joblib"
            joblib.dump(df_ck, ck_path)
            log(f"💾 Final flush: {len(df_ck):,} rows → {ck_path.name}")

        if not embeddings:
            log("❌ Nothing embedded. Aborting.")
            print(f"📝 Log saved to {log_path}")
            return ""

        # ── 3.  Save complete vector set & register ───────────────────────
        df_vectors = pd.DataFrame(
            {"conversation_id": conv_ids[: len(embeddings)], "embedding": embeddings}
        )
        out_path = ARTIFACT_DIR / f"vectors_{batch_label}_{ts}.joblib"
        joblib.dump(df_vectors, out_path)
        log(f"✅ Saved {len(df_vectors):,} vectors → {out_path.name}")

        vector_refs = [
            {"conversation_id": r["conversation_id"], "source": r["data_ref"]}
            for r in conversation_records[: len(embeddings)]
        ]
        batch_ids = sorted({r["batch_id"] for r in conversation_records if r.get("batch_id")})

        art_id = registry.register_vectors_batch(
            data_ref=str(out_path),
            parameters={"model": model, "n_vectors": len(df_vectors)},
            parents=batch_ids,
            vector_refs=vector_refs,
            batches=batch_ids,
        )
        log(f"🏁 DONE | artifact={art_id}")

    print(f"📝 Log saved to {log_path}")
    return art_id

# ---------------------------------------------------------------------------
# 🚀 CLI
# ---------------------------------------------------------------------------
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--batch", help="Filter business registry by batch_id.")
@click.option("--limit", type=int, help="Process at most N conversations.")
@click.option("--model", "-m", default=DEFAULT_MODEL, show_default=True)
@click.option("--openai-batch-size", default=DEFAULT_OPENAI_BATCH, show_default=True)
@click.option("--checkpoint", default=50, show_default=True, help="Write a log entry every N new embeddings.")
@click.option("--force", is_flag=True, help="Re‑embed even if vectorizer:<conv_id> already exists.")
@click.option("--verbose", "-v", is_flag=True)

def main(batch: str, limit: int | None, model: str, openai_batch_size: int, checkpoint: int, force: bool, verbose: bool):
    """Vectorize a batch of conversations and register as a single artifact."""
    interactive = len(sys.argv) == 1  
    if interactive:
        conversation_records, batch, limit, model, openai_batch_size, force, verbose = interactive_embedding_menu(
            default_model=model,
            default_batch_size=openai_batch_size,
            default_limit=limit,
            default_force=force,
            default_verbose=verbose,
        )
        if limit:
            conversation_records = conversation_records[:limit]

        if not conversation_records:
            click.secho("👍 All conversations already embedded.", fg="green")
            return
        
        artifact_id = vectorize_given_conversations(
            conversation_records,
            model=model,
            openai_batch_size=openai_batch_size,
            checkpoint=checkpoint,
            batch_label=batch or "multi",
            verbose=verbose,
        )
       

    if artifact_id:
        click.secho(f"✅ Done. Registered artifact: {artifact_id}", fg="green")
    else:
        click.secho("⚠️ No artifact created. (Already embedded or skipped)", fg="yellow")



if __name__ == "__main__":
    main()
