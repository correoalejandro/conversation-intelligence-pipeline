"""
CLI Reporter — adaptive lineage, concise listings, controllable expansion.

Commands:
  summary        Overview grouped by stage (default 5 per stage)
  recent         N most recent artefacts (default 5)
  lineage        ASCII tree with adaptive breadth‑limit + --expand-levels flag

Run:  python zz_CLI_reporter.py <command> -h
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

from _data_layer import registry  # provides find(), lineage(), _entry_id


from datetime import datetime
import locale

# make sure Spanish locale is set for strptime
locale.setlocale(locale.LC_TIME, "Spanish_Spain.1252")

def _parse_any_ts(ts: str) -> datetime:
    """Return a comparable datetime; unknown formats => earliest time."""
    for fmt in ("%Y-%m-%dT%H:%M:%S",             # ISO 8601 without micros
                "%Y-%m-%dT%H:%M:%S.%f",          # ISO 8601 with micros
                "%A, %d de %B de %Y, %H:%M:%S"): # Spanish long
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return datetime.min  # push unparseable to the very end




# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────
def _print_artifacts(
    arts: List[dict], *, title: str = "Artifacts", parent_limit: int = 5
) -> None:
    """Pretty‑print artefacts, truncating long parent lists."""
    if not arts:
        print("⚠️  Nothing to show.")
        return

    print(f"\n=== {title} ({len(arts)}) ===")
    for a in arts:
        art_id = registry._entry_id(a) or "??"
        ok = "✅" if os.path.exists(a.get("data_ref", "")) else "❌ MISSING"
        created = a.get("created_at", "no‑date")
        stage = a.get("stage", "??")
        print(f"{art_id}  |  {stage}  |  {created}  [{ok}]")

        parents = a.get("parents", [])
        if parents:
            if parent_limit and len(parents) > parent_limit:
                shown = ", ".join(parents[:parent_limit])
                print(f"   ↳ parents: {shown} … +{len(parents) - parent_limit} more")
            else:
                print("   ↳ parents:", ", ".join(parents))
        print()

# ─────────────────────────────────────────────────────────────────────────────
# Registry helpers
# ─────────────────────────────────────────────────────────────────────────────
def list_recent(limit: int = 5, stage: Optional[str] = None, source: str = "all") -> List[dict]:
    from _data_layer import registry

    results = registry.find(source=source) if stage is None else registry.find(stage=stage, source=source)
    print(f"🔍 Loaded {len(results)} artefacts (stage={stage}, source={source})")
    


    results.sort(key=lambda x: _parse_any_ts(x.get("created_at", "")),
                 reverse=True)

    for i, r in enumerate(results):
        if r.get("id") == "artifact_1e1e0acc48":
            print(f"✅ Found target at index {i}")
            print(f"{r.get('id')} | {r.get('created_at')}")
            break
    else:
        print("❌ Target not found in sorted results")

    return results[:limit]


def print_registry_summary(stage: Optional[str] = None, *, per_stage_limit: int = 5, source: str = "all") -> None:
    from _data_layer import registry

    all_artifacts = registry.find(stage=stage, source=source)

    print("\n=== 📚 Artifact Registry Summary ===")
    stages = {}
    for a in all_artifacts:
        stages.setdefault(a.get("stage", "??"), []).append(a)

    for stg, items in stages.items():
        print(f"\n--- Stage: {stg} ({len(items)} artefacts) ---")
        _print_artifacts(items[:per_stage_limit],
                         title=f"Latest {per_stage_limit} in {stg}")
    print("\n=== Stage Counts Summary ===")
    for stg, items in stages.items():
        print(f"{stg:>25}: {len(items)}")
# ─────────────────────────────────────────────────────────────────────────────
# Lineage wrapper
# ─────────────────────────────────────────────────────────────────────────────
def print_lineage(
    artifact_id: str,
    *,
    max_depth: int = 6,
    breadth_limit: int | None = 5,
    show_levels: Tuple[int, ...] = (0, 1),
    expand_n_children: int = 1,
) -> None:
    try:
        tree = registry.lineage(
            artifact_id,
            max_depth=max_depth,
            breadth_limit=breadth_limit,
            show_levels=show_levels,
        )
        print("\n=== 🪢 Lineage ===")
        print(tree)
    except Exception as e:
        print(f"❌ Could not generate lineage: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI parser
# ─────────────────────────────────────────────────────────────────────────────
def _parse_expand_levels(raw: str) -> Tuple[int, ...]:
    """Allow --expand-levels with or without a value (Windows-safe)."""
    if raw in (None, "", "''", "\"\""):
        return ()
    return tuple(int(x) for x in raw.split(",") if x.strip())

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Artifact Registry Reporter")
    
    sub = p.add_subparsers(dest="cmd", required=True)

    # summary
    s = sub.add_parser("summary", help="Overview grouped by stage")
    s.add_argument("--stage", default=None, help="Filter to one stage")
    s.add_argument("-n", "--per-stage", type=int, default=5,
                   help="How many artefacts per stage (default 5)")
    s.add_argument("--source", choices=["technical", "business"], default="technical",
               help="Which registry to query (default: technical)")
    

    # recent
    r = sub.add_parser("recent", help="List recent artefacts")
    r.add_argument("--stage", default=None, help="Filter to one stage")
    r.add_argument("-n", "--limit", type=int, default=5,
                   help="Number of artefacts to show (default 5)")
    r.add_argument("--source", choices=["technical", "business"], default="technical",
               help="Which registry to query (default: technical)")
    

    # lineage
    l = sub.add_parser("lineage", help="Display lineage tree")
    l.add_argument("artifact_id", help="Artefact ID to inspect")
    l.add_argument("--max-depth", type=int, default=6, help="Depth limit")
    l.add_argument("--breadth-limit", type=int, default=5,
                   help="Parents per level before collapsing (None = all)")
    l.add_argument(
        "--expand-levels",
        nargs="?",            # makes the value optional
        const="",             # if user writes --expand-levels with nothing after it
        type=_parse_expand_levels,
        default=(0, 1),       # still default to depths 0 and 1
        help=(
            "Comma‑separated depths that stay fully expanded "
            "(e.g. 0,1).  Pass nothing (just --expand-levels) to collapse "
            "everything except the root."
        ))
    l.add_argument(
        "--expand-n",
        type=int,
        default=1,
        help="Number of level‑1 children to expand fully (default 1)"
    )


    return p

# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    if args.cmd == "summary":
        print_registry_summary(stage=args.stage, per_stage_limit=args.per_stage, source=args.source)


    elif args.cmd == "recent":
        arts = list_recent(limit=args.limit, stage=args.stage, source=args.source)
        _print_artifacts(arts, title=f"Recent artefacts ({args.source})")

    elif args.cmd == "lineage":
        print_lineage(
            args.artifact_id,
            max_depth=args.max_depth,
            breadth_limit=args.breadth_limit,
            show_levels=args.expand_levels,
            expand_n_children=args.expand_n,
        )

if __name__ == "__main__":   # pragma: no cover
    main()
