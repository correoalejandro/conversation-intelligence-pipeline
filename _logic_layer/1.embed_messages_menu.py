    #!/usr/bin/env python3
"""
1.v12.embed_messages_menu.py — Interactive runner to embed **each message**
===========================================================================

This wraps the non-interactive 1.v12.embed_messages.py logic with a simple
interactive menu (similar to your conversation-level tool). It lets you:

• Inspect pending (not-yet-embedded) messages by batch
• Choose a specific batch or "All batches"
• Set model, API batch size, limit, force, and verbose
• Run the embed job immediately

Requirements
------------
• OPENAI_API_KEY set
• Your _data_layer modules accessible on PYTHONPATH
"""

from __future__ import annotations

import os
import sys
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click
from duckdb import df
import joblib
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# Project imports (same as your other scripts)
sys.path.append("c:/Projects/clasificador_mensajes")
from _data_layer import api, registry  # noqa: E402
from _data_layer.paths import ARTIFACT_DIR, LOG_DIR  # noqa: E402

# -------------------- Defaults --------------------
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_BATCH = 512
CHECKPOINT_DEFAULT = 200
# Tokenizer & limit (embedding-3 models = 8,192 tokens)
MAX_EMBED_TOKENS = 8192
ENC = tiktoken.get_encoding("cl100k_base")  # works for text-embedding-3-small


load_dotenv()
CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _now_utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# -------------------- Data Loading & Pending Computation --------------------
def load_latest_message_table() -> pd.DataFrame:
    df, rec = api.load_artifact("message_table")
    expected = {"conversation_id", "batch_id", "message_id", "sender", "text", "timestamp"}
    missing = expected - set(df.columns)
    if missing:
        raise SystemExit(f"❌ message_table is missing columns: {missing}")
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df.loc[df["text"].str.strip() != ""].copy()
    return df.reset_index(drop=True)


def seen_message_hashes(model: str) -> set[str]:
    """Collect message_hash values already embedded for this model."""
    seen: set[str] = set()
    for art in registry.find(stage="vectorizing_text", source="technical"):
        params = art.get("parameters") or {}
        if params.get("model") != model or params.get("granularity") != "message":
            continue
        for ref in (art.get("vector_refs") or []):
            mh = ref.get("message_hash")
            if mh:
                seen.add(mh)
    return seen


def compute_pending_summary(df: pd.DataFrame, model: str, force: bool, restrict_batch: Optional[str] = None):
    df = df.copy()
    if restrict_batch:
        df = df.loc[df["batch_id"] == restrict_batch]

    df["message_hash"] = [ _sha256_text(t) for t in df["text"].tolist() ]
    if not force:
        seen = seen_message_hashes(model)
        df = df.loc[~df["message_hash"].isin(seen)]

    # Aggregate counts per batch
    per_batch = df.groupby("batch_id", dropna=False).size().sort_values(ascending=False).to_dict()
    total_pending = int(df.shape[0])
    return df, per_batch, total_pending

def _truncate_to_tokens(text: str, max_tokens: int = MAX_EMBED_TOKENS) -> str:
    if not text:
        return text
    toks = ENC.encode(text)
    if len(toks) <= max_tokens:
        return text
    return ENC.decode(toks[:max_tokens])


# -------------------- Embedding Core --------------------
def embed_batch(texts: List[str], model: str) -> List[List[float]]:
    resp = CLIENT.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


@dataclass
class RunCfg:
    model: str
    openai_batch_size: int
    checkpoint: int
    limit: Optional[int]
    force: bool
    verbose: bool
    batch_filter: Optional[str]


def run_embedding(df: pd.DataFrame, cfg: RunCfg) -> None:
    # Apply restrictions
    if cfg.batch_filter:
        df = df.loc[df["batch_id"] == cfg.batch_filter].copy()

    # Prepare dedup (again) based on final model/force
    df = df.copy()
    df["message_hash"] = [ _sha256_text(t) for t in df["text"].tolist() ]
    if not cfg.force:
        seen = seen_message_hashes(cfg.model)
        df = df.loc[~df["message_hash"].isin(seen)].copy()
    if cfg.limit:
        df = df.head(cfg.limit).copy()

    if df.empty:
        click.secho("👍 No messages to embed for the selected options.", fg="green")
        return

    # Embedding
    ts = _now_utc_tag()
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    log_path = LOG_DIR / f"embed_messages_{ts}.log"
    with log_path.open("w", encoding="utf-8") as lf:
        def log(msg: str):
            if cfg.verbose:
                click.echo(msg)
            lf.write(msg + "\n")

        log(f"▶️ START | model={cfg.model} | messages={len(df):,} | batch_filter={cfg.batch_filter or 'ALL'}")
        # Truncate any oversize texts to model window
        texts = [_truncate_to_tokens(t, MAX_EMBED_TOKENS) for t in df["text"].tolist()]


        out_rows: List[Dict[str, Any]] = []
        for i in range(0, len(texts), cfg.openai_batch_size):
            chunk = texts[i:i + cfg.openai_batch_size]
            embs = embed_batch(chunk, cfg.model)

            meta = df.iloc[i:i + len(embs)]
            for (idx, emb) in zip(meta.index, embs):
                row = meta.loc[idx]
                out_rows.append({
                    "conversation_id": row["conversation_id"],
                    "batch_id":        row["batch_id"],
                    "message_id":      row["message_id"],
                    "sender":          row["sender"],
                    "timestamp":       row["timestamp"],
                    "text":            row["text"],
                    "message_hash":    row["message_hash"],
                    "embedding":       emb,
                })

            # checkpoint
            if (i // cfg.openai_batch_size) % max(1, (cfg.checkpoint // max(1, cfg.openai_batch_size))) == 0 and cfg.checkpoint:
                ck_df = pd.DataFrame(out_rows)
                if not ck_df.empty:
                    ck_path = ARTIFACT_DIR / f"msg_ckpt_{ts}_{i}.joblib"
                    joblib.dump(ck_df, ck_path, compress=3)
                    log(f"💾 Checkpoint: {len(ck_df):,} rows → {ck_path.name}")

    # Save final vectors bundle
    df_vectors = pd.DataFrame(out_rows)
    out_path = ARTIFACT_DIR / f"message_vectors_{ts}.joblib"
    joblib.dump(df_vectors, out_path, compress=3)

    click.secho(f"✅ Saved {len(df_vectors):,} message vectors → {out_path.name}", fg="green")
    click.echo(f"📝 Log: {log_path.name}")

    # Register
    vector_refs = [
        {"conversation_id": r["conversation_id"], "message_id": r["message_id"], "message_hash": r["message_hash"]}
        for r in out_rows
    ]
    batch_ids = sorted(set(df_vectors["batch_id"].dropna().astype(str).tolist()))
    art_id = registry.register_vectors_batch(
        data_ref=str(out_path),
        parameters={
            "model": cfg.model,
            "n_vectors": len(df_vectors),
            "granularity": "message",
        },
        parents=batch_ids,
        vector_refs=vector_refs,
        batches=batch_ids,
    )
    click.secho(f"🏁 DONE | registered artifact: {art_id}", fg="cyan")


# -------------------- Interactive Menu --------------------
@click.command()
@click.option("--default-model", default=DEFAULT_MODEL, show_default=True, help="Default embedding model")
@click.option("--default-openai-batch-size", default=DEFAULT_OPENAI_BATCH, show_default=True, help="Default API batch size")
@click.option("--default-checkpoint", default=CHECKPOINT_DEFAULT, show_default=True, help="Rows per checkpoint (approx)")
@click.option("--default-limit", type=int, default=None, help="Default cap on number of messages")
@click.option("--default-force", is_flag=True, help="Default to force re-embedding (ignore dedupe)")
@click.option("--default-verbose", is_flag=True, help="Default to verbose logging")
def main(default_model, default_openai_batch_size, default_checkpoint, default_limit, default_force, default_verbose):
    click.clear()
    click.secho("Message Embedding — Interactive Runner", fg="bright_white", bold=True)
    click.echo("")

    # 1) Load base table
    base_df = load_latest_message_table()

    # 2) Choose model up front (for dedupe preview)
    model = click.prompt("Embedding model", default=default_model)

    # 3) Preview pending counts per batch (respecting model/force=False at this point)
    _, per_batch, total = compute_pending_summary(base_df, model=model, force=default_force)
    if total == 0 and not default_force:
        click.secho("Everything is already embedded for this model (use --default-force or Force=Y to re-run).", fg="yellow")

    click.echo("\nPending messages by batch:")
    for b, n in per_batch.items():
        click.echo(f"  • {b}: {n:,}")

    click.echo(f"\n🧮 Total pending: {total:,}\n")

    # 4) Selection
    batches_sorted = list(per_batch.keys())
    click.echo("Selection: [0] All batches")
    for i, b in enumerate(batches_sorted, start=1):
        click.echo(f"  [{i}] {b} ({per_batch[b]:,} msgs)")

    idx = click.prompt(f"Choose batch [0–{len(batches_sorted)}]", type=int, default=0)
    if idx == 0:
        batch_choice = None
    else:
        batch_choice = batches_sorted[idx-1]

    # 5) Other options
    limit = click.prompt("Limit messages (blank for none)", default=default_limit, type=int, show_default=True) if default_limit is not None else None
    openai_batch_size = click.prompt("OpenAI batch size", default=default_openai_batch_size, type=int)
    checkpoint = click.prompt("Checkpoint every ~N rows (0=off)", default=default_checkpoint, type=int)
    force = click.confirm("Force re-embed (ignore dedupe)?", default=default_force)
    verbose = click.confirm("Verbose logging?", default=default_verbose)

    # 6) Confirm
    click.echo("\nSummary:")
    click.echo(f"  Model: {model}")
    click.echo(f"  Batch: {batch_choice or 'ALL'}")
    click.echo(f"  Limit: {limit or '∞'}")
    click.echo(f"  API batch size: {openai_batch_size}")
    click.echo(f"  Checkpoint: {checkpoint}")
    click.echo(f"  Force: {force}")
    click.echo(f"  Verbose: {verbose}")

    if not click.confirm("\nProceed?", default=True):
        click.secho("Aborted.", fg="yellow")
        return

    # 7) Run
    cfg = RunCfg(
        model=model,
        openai_batch_size=openai_batch_size,
        checkpoint=checkpoint,
        limit=limit,
        force=force,
        verbose=verbose,
        batch_filter=batch_choice,
    )
    run_embedding(base_df, cfg)


if __name__ == "__main__":
    main()
