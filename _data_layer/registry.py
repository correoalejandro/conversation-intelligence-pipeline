from __future__ import annotations

import pandas as pd

"""Registry module — *visible* metadata‑pipeline ledger.

• Centralises artefact metadata in UTF‑8 JSON registries.
• Tracks **stage** and **parents** for lineage reconstruction.
• Keeps legacy helpers so older scripts keep running while you migrate.
"""

import json
import uuid
import hashlib
import sys, locale, os
from typing import Any, Iterable
import itertools, datetime as _dt

# 1️⃣ Load the Windows‑style Spanish locale
locale.setlocale(locale.LC_TIME, "Spanish_Spain.1252")   # not es_ES.UTF‑8

# 2️⃣ Switch the current console to UTF‑8 (once per session)
os.system("chcp 65001 > nul")    # cmd / PowerShell

# 3️⃣ Tell Python to emit UTF‑8 to that console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
from typing import List, Dict, Any   # ya está importado arriba, pero por si acaso

# ---------------------------------------------------------------------------
# Paths & low‑level helpers
# ---------------------------------------------------------------------------

# Root‑level directory that will hold every JSON registry file.
_REG_DIR: Path = (Path.cwd() / "registries").resolve()
# ensure the full chain exists, even on first run
_REG_DIR.mkdir(parents=True, exist_ok=True)

_TECHNICAL_REG_PATH = _REG_DIR / "technical_registry.json"
_BUSINESS_REG_PATH = _REG_DIR / "business_registry.json"
_PROMPT_REG_PATH = _REG_DIR / "prompt_registry.json"

# ---------------------------------------------------------------------------
# JSON helpers (robust to locale & corrupt files)
# ---------------------------------------------------------------------------

def _save(obj: Any, path: Path) -> None:
    """Pretty‑print *obj* as UTF‑8 JSON, creating folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _load(path: Path):
    """Load JSON → Python, returning [] if file is missing/empty/corrupt."""
    if not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            return []
        return json.loads(content)
    except json.JSONDecodeError:
        # leave a backup and reset the file
        backup = path.with_suffix(".corrupt.json")
        path.rename(backup)
        print(f"⚠️ Corrupted registry moved to {backup.name}. Starting fresh.")
        return []

# ---------------------------------------------------------------------------
# Delete helper (remove one artifact from technical registry)
# ---------------------------------------------------------------------------

def delete(artifact_id: str) -> bool:
    """
    Remove an artifact entry (by id) from the technical registry.
    Returns True si se eliminó algo, False si no existía.
    """
    tech = _load(_TECHNICAL_REG_PATH)
    new_tech = [e for e in tech if e.get("id") != artifact_id]
    if len(new_tech) == len(tech):
        return False  # no se encontró
    _save(new_tech, _TECHNICAL_REG_PATH)
    return True


# ---------------------------------------------------------------------------
# Write helper for api.save_artifact
# ---------------------------------------------------------------------------

def add(entry: Dict[str, Any]) -> None:
    """
    Append a single artifact 'entry' to the *technical* registry
    and persist it atomically.
    """
    tech_registry = _load(_TECHNICAL_REG_PATH)
    tech_registry.append(entry)
    _save(tech_registry, _TECHNICAL_REG_PATH)

# ---------------------------------------------------------------------------
# Read helper for api.load_artifact
# ---------------------------------------------------------------------------

def latest(stage: str) -> Dict[str, Any] | None:
    """
    Return the most recent artifact for a given stage,
    or None if no match is found.
    """
    tech_registry = _load(_TECHNICAL_REG_PATH)
    candidates = [e for e in tech_registry if e.get("stage") == stage]
    if not candidates:
        return None
    # sort by created_at DESC (empty dates to the end)
    candidates.sort(key=lambda e: e.get("created_at", ""), reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------


def find(*, source: str = "all", **filters) -> List[Dict[str, Any]]:

    """
    Busca y devuelve artefactos del *technical* y del *business registry*
    que cumplan las condiciones (igualdad exacta en cada clave).

    Ejemplos
    --------
    >>> find(stage="embeddings")
    >>> find(stage="generator", pipeline="pipeline_A")
    >>> find(agent_name="Bot42")

    Devuelve una lista ordenada por `created_at` (reciente primero).
    """
    if source == "technical":
        entries = _load(_TECHNICAL_REG_PATH)
    elif source == "business":
        entries = _load(_BUSINESS_REG_PATH)
    else:
        entries = _load(_TECHNICAL_REG_PATH) + _load(_BUSINESS_REG_PATH)

    def _matches(entry: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if value is None:               # ignora filtros vacíos
                continue
            if entry.get(key) != value:     # igualdad estricta
                return False
        return True

    results = [e for e in entries if _matches(e)]
    # Orden cronológico inverso si existe la clave
    results.sort(key=lambda e: e.get("created_at", ""), reverse=True)
    return results


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%A, %d de %B de %Y, %H:%M:%S")



def new_id(prefix: str = "artifact") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# Content hashing (files *or* directories)
# ---------------------------------------------------------------------------

def _compute_hash(path: Path, chunk_size: int = 131_072) -> str:
    """Deterministic SHA‑256 hash for either a file or an entire directory."""
    hasher = hashlib.sha256()
    if path.is_dir():
        # include structure: rel‑path, size, mtime — deterministic walk
        for file in sorted(p for p in path.rglob("*") if p.is_file()):
            rel = file.relative_to(path).as_posix()
            stats = file.stat()
            hasher.update(rel.encode())
            hasher.update(str(stats.st_size).encode())
            hasher.update(str(int(stats.st_mtime)).encode())
    else:
        with path.open("rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Prompt registration (STAGE 1 — prompt_definition)
# ---------------------------------------------------------------------------

def register_prompt(prompt_path: str,
                    description: str = "",
                    author: str = "",
                    tags: List[str] | None = None) -> str:
    path = Path(prompt_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    prompt_hash = _compute_hash(path)
    prompt_id = f"prompt_{prompt_hash[:6]}"

    entry = {
        #waterfall_id
        "prompt_id": prompt_id,
        "stage": "prompt_definition",
        "hash": prompt_hash,
        "data_ref": path.as_posix(),
        "description": description,
        "author": author,
        "tags": tags or [],
        "created_at": _timestamp(),
    }

    tech_registry = _load(_TECHNICAL_REG_PATH)
    if any(r.get("hash") == prompt_hash for r in tech_registry):
        print(f"⚠️ Prompt already registered: {prompt_id}")
    else:
        tech_registry.append(entry)
        _save(tech_registry, _TECHNICAL_REG_PATH)
        print(f"✅ Registered prompt: {prompt_id}")

    return prompt_id


# ---------------------------------------------------------------------------
# Batch registration (STAGE 2 — text_processing)
# ---------------------------------------------------------------------------

def register_batch(data_ref: str,
                   prompt_id: str,
                   parameters: Dict[str, Any]) -> str:
    path = Path(data_ref).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Batch file/folder not found: {data_ref}")

    # Hash the batch parameters to avoid duplicate hashes from empty folders
    batch_hash = hashlib.sha256(
        json.dumps(parameters, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()

    batch_id = f"batch_{uuid.uuid4().hex[:10]}"

    entry = {
        "batch_id": batch_id,
        "stage": "text_processing",
        "hash": batch_hash,
        "data_ref": path.as_posix(),
        "parents": [prompt_id],
        "parameters": parameters,
        "created_at": _timestamp(),
    }

    tech_registry = _load(_TECHNICAL_REG_PATH)
    if any(r.get("hash") == batch_hash for r in tech_registry):
        print(f"⚠️ Batch already registered (hash collision): {batch_id}")
    else:
        tech_registry.append(entry)
        _save(tech_registry, _TECHNICAL_REG_PATH)
        print(f"✅ Registered batch: {batch_id}")

    return batch_id


# ---------------------------------------------------------------------------
# Conversation, embedding analysis, model registration 
# ---------------------------------------------------------------------------


def register_conversations(batch_id: str,
                           conversations: List[Dict[str, Any]]) -> None:
    """Append conversation metadata to the business registry."""
    business_reg = _load(_BUSINESS_REG_PATH)
    for conv in conversations:
        entry = {
            "conversation_id": conv["conversation_id"],
            "stage": "conversation_record",
            "parents": [batch_id],
            "batch_id": batch_id,
            "scenario": conv.get("scenario") or conv.get("scenario"),
            "agent_name": conv.get("agent_name") or conv.get("agent"),
            "client_name": conv.get("client_name") or conv.get("client"),
             "conversation_start": conv.get("conversation_start") or _timestamp(),
            "created_at": conv.get("created_at") or _timestamp(),
            "data_ref": conv.get("data_ref"),
        }
        business_reg.append(entry)

    _save(business_reg, _BUSINESS_REG_PATH)
    print(f"✅ Registered {len(conversations)} conversations under {batch_id}.")


def register_vectors_batch(data_ref: str,
                           parameters: dict,
                           parents: list[str] | None = None,
                           vector_refs: list[dict] | None = None,
                           batches: list[str] | None = None) -> str:
    """Register a batch of vectors as one artifact under 'vectorizing_text'."""
    path = Path(data_ref).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {data_ref}")

    entry = {
        "id": new_id("vectors_batch"),
        "stage": "vectorizing_text",
        "data_ref": path.as_posix(),
        "backend": "joblib",
        "parameters": {**parameters},
        "parents": parents or [],
        "vector_refs": vector_refs or [],
        "batches": batches or [],
        "created_at": _timestamp()
    }
    add(entry)
    print(f"✅ Registered vectors_batch: {entry['id']}")
    return entry["id"]

# ---------------------------------------------------------------------------
# Embedding‑analysis registration (STAGE 4 — embedding_analysis)
# ---------------------------------------------------------------------------

def register_embedding_analysis(data_ref: str,
                                parameters: Dict[str, Any] | None = None,
                                parents: List[str] | None = None) -> str:
    """
    Register an experiment bundle that contains UMAP coordinates +
    HDBSCAN clusters linked back to one or more *embedding* artifacts.

    Parameters
    ----------
    data_ref : str
        Path to the .joblib file produced by the pipeline.
    parameters : dict, optional
        Dict of hyper‑parameters and options used in the run.
    parents : list[str], optional
        The embedding artifact IDs this experiment depends on.

    Returns
    -------
    str
        The newly created artifact_id (prefix `embedding_analysis_…`).
    """
    path = Path(data_ref).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Experiment file not found: {data_ref}")

    art_hash = _compute_hash(path)
    art_id   = new_id("embedding_analysis")

    entry = {
        "id": art_id,
        "stage": "embedding_analysis",
        "backend": "joblib",
        "hash": art_hash,
        "data_ref": path.as_posix(),
        "parents": parents or [],
        "parameters": parameters or {},
        "created_at": _timestamp(),
    }

    add(entry)          # reuse the low‑level helper
    print(f"✅ Registered embedding analysis: {art_id}")
    return art_id

from _data_layer.registry import add, new_id, _timestamp

def register_model(*,
                   data_ref: str,
                   parameters: dict,
                   parents: list[str],
                   model_type: str = "hdbscan",
                   stage_prefix: str = "model") -> str:
    """
    Register a model artifact (e.g., HDBSCAN or UMAP) to the technical registry.

    Parameters
    ----------
    data_ref : str
        Path to the .joblib file.
    parameters : dict
        Hyperparameters used for training the model.
    parents : list[str]
        IDs of parent artefacts (e.g. embedding_analysis).
    model_type : str
        'hdbscan', 'umap', etc. — appended to stage.
    stage_prefix : str
        Default is 'model'. Will become 'model:hdbscan', etc.

    Returns
    -------
    str
        The registered artifact ID (e.g. hdbscan_model_123abc).
    """
    model_id = new_id(f"{model_type}_model")
    stage = f"{stage_prefix}:{model_type}"

    add({
        "id": model_id,
        "stage": stage,
        "backend": "joblib",
        "data_ref": data_ref,
        "parameters": parameters,
        "parents": parents,
        "created_at": _timestamp(),
    })

    print(f"✅ Registered {model_type.upper()} model: {model_id}")
    return model_id

# ---------------------------------------------------------------------------
# Audit & cleanup helpers
# ---------------------------------------------------------------------------

def audit_batch(batch_id: str) -> Dict[str, Any]:
    business = _load(_BUSINESS_REG_PATH)
    conversations = [c for c in business if c.get("batch_id") == batch_id]
    if not conversations:
        return {"batch_id": batch_id, "status": "❌ No conversations found."}

    issues: List[str] = []
    for c in conversations:
        if not c.get("agent_name") or not c.get("client_name"):
            issues.append(f"Missing agent/client in {c['conversation_id']}")
        if not c.get("scenario"):
            issues.append(f"Missing scenario in {c['conversation_id']}")

    return {
        "batch_id": batch_id,
        "status": "✅ Passed" if not issues else "⚠️ Issues found",
        "issues": issues,
    }


def remove_batch(batch_id: str) -> None:
    tech = _load(_TECHNICAL_REG_PATH)
    tech = [b for b in tech if b.get("batch_id") != batch_id]
    _save(tech, _TECHNICAL_REG_PATH)

    business = _load(_BUSINESS_REG_PATH)
    business = [c for c in business if c.get("batch_id") != batch_id]
    _save(business, _BUSINESS_REG_PATH)

    print(f"🗑 Removed batch {batch_id} and its conversations.")


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def get_conversation_stats() -> Dict[str, Any]:
    business = _load(_BUSINESS_REG_PATH)
    total = len(business)
    agent_counts: Dict[str, int] = {}
    scenario_counts: Dict[str, int] = {}
    for c in business:
        agent_counts[c["agent_name"]] = agent_counts.get(c["agent_name"], 0) + 1
        scenario_counts[c["scenario"]] = scenario_counts.get(c["scenario"], 0) + 1

    return {
        "total_conversations": total,
        "agent_distribution": agent_counts,
        "scenario_distribution": scenario_counts,
    }


def _entry_id(e: dict) -> str | None:
    for k in ("id", "prompt_id", "batch_id", "conversation_id"):
        if k in e:
            return e[k]
    return None

# _data_layer/registry.py
'''
def lineage(artifact_id: str, *, max_depth: int = 5) -> str:
    """Return pretty ASCII tree of parents ↩ children."""
    tech = _load(_TECHNICAL_REG_PATH)
    lookup = { _entry_id(e): e for e in tech if _entry_id(e) }
    lines = []

    def _walk(aid: str, depth: int = 0):
        if depth > max_depth or aid not in lookup:
            return
        art = lookup[aid]
        stamp = art.get("created_at", "no-date")
        label = f"{art['stage']}:{stamp.split(',')[0]}"
        lines.append("  " * depth + f"↳ {label}")
        for parent in art.get("parents", []):
            _walk(parent, depth + 1)

    _walk(artifact_id, 0)
    return "\n".join(lines)

'''

# ---------------------------------------------------------------------------
# Lineage: adaptive ASCII tree with breadth‑limit summarisation
# ---------------------------------------------------------------------------
def lineage(
    artifact_id: str,
    *,
    max_depth: int = 6,
    breadth_limit: int | None = 5,
    summary_fmt: str = "↳ … {n} more parents",
    show_levels: Iterable[int] = (0, 1),
    expand_n_children: int = 1,  # 👈 NEW
) -> str:
    """
    Adaptive ASCII lineage tree.

    Each displayed node now shows:  artifact_id | stage  (created_at)

    Parameters
    ----------
    artifact_id   Root artefact ID.
    max_depth     Recursion limit (default 6).
    breadth_limit Collapse when a node has > breadth_limit parents (None = no
                  collapsing).  Depths listed in `show_levels` are always fully
                  expanded.
    summary_fmt   Format for the summarising line; must include {n}.
    show_levels   Depths that stay fully expanded (default 0,1).

    Returns
    -------
    str
        Multiline lineage tree.
    """
    tech = _load(_TECHNICAL_REG_PATH)
    by_id = { _entry_id(e): e for e in tech if _entry_id(e) }

    lines: list[str] = []
    visited: Set[str] = set()        # cycle guard

    def _walk(aid: str, depth: int = 0) -> None:
        if depth > max_depth or aid in visited:
            return
        visited.add(aid)

        art = by_id.get(aid)
        if art is None:
            lines.append("  " * depth + f"↳ ?? {aid}")   # orphan pointer
            return

        stamp = art.get("created_at", "no-date").split(",")[0]
        stage = art.get("stage", "??")
        label = f"{aid} | {stage}  ({stamp})"
        lines.append("  " * depth + "↳ " + label)

        parents = art.get("parents", [])
        if (
            breadth_limit is not None and
            len(parents) > breadth_limit
        ):
            for p in parents[:breadth_limit]:
                _walk(p, depth + 1)
            lines.append(
                "  " * (depth + 1) +
                summary_fmt.format(n=len(parents) - breadth_limit)
            )
        else:
            for p in parents:
                _walk(p, depth + 1)

    _walk(artifact_id)
    return "\n".join(lines)