#!/usr/bin/env python3
"""
📈 Visualize HDBSCAN hierarchy from a saved topic:hierarchy artefact.
"""

import json
from typing import Any
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

from _data_layer.api import _backend, load_artifact
from _data_layer.registry import _entry_id, find

import click


def load_hierarchy_artifact(art_id: str) -> list[dict]:
    """
    Return hierarchy as list[dict] no matter how the artefact is stored
    (JSON string, NDJSON, or already‑parsed list).
    """
    for rec in find():                     # registry.find()
        if _entry_id(rec) == art_id:
            data = _backend(rec["backend"]).load(rec["data_ref"])
            break
    else:
        raise click.ClickException(f"❌ No artefact {art_id}")

    # bytes → str
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf‑8")

    # already list[dict]
    if isinstance(data, list):
        return data

    # JSON string
    if isinstance(data, str):
        try:
            return json.loads(data)              # full JSON
        except json.JSONDecodeError:
            # NDJSON fallback
            return [json.loads(l) for l in data.splitlines() if l.strip()]

    # joblib‑saved DataFrame
    if isinstance(data, pd.DataFrame):
        return data.to_dict("records")

    raise click.ClickException("Unsupported hierarchy format")



def plot_hierarchy_tree(hierarchy: list[dict[str, Any]]):
    """
    Draw HDBSCAN hierarchy tree from saved topic:hierarchy list-of-dict.
    """
    G = nx.DiGraph()
    for row in hierarchy:
        parent = row["parent"]
        child = row["child"]
        label = f"{child} (λ={row['lambda_val']:.2f})"
        G.add_edge(parent, child, label=label)

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  # Top-down layout
    except:
        pos = nx.spring_layout(G, seed=42)  # Fallback

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [-y0, -y1, None]  # Flip Y to go top-down

    node_x, node_y, labels = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(-y)
        labels.append(str(node))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1, color='gray'),
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=labels, textposition='top center',
        marker=dict(size=8, color='blue'),
        hoverinfo='text'
    ))
    fig.update_layout(
        title="🌲 HDBSCAN Condensed Hierarchy Tree",
        showlegend=False,
        margin=dict(l=10, r=10, b=10, t=40),
        height=600
    )
    fig.show()


def interactive_select() -> str:
    """CLI selection of a topic:hierarchy artefact."""
    candidates = [r for r in find() if r.get("stage") == "topic:hierarchy"]
    if not candidates:
        raise click.ClickException("❌ No topic:hierarchy artefacts found in registry.")

    click.echo("📦 Select hierarchy artefact:")
    for idx, r in enumerate(candidates[:20]):
        click.echo(f"  [{idx}] {_entry_id(r)} | {r['data_ref']}")
    sel = input("Select number [0]: ").strip()
    sel_idx = int(sel) if sel else 0
    return _entry_id(candidates[sel_idx])

from collections import defaultdict, deque


def analyze_hdbscan_tree(hierarchy: list[dict]) -> None:
    from collections import defaultdict, deque

    # Build cluster tree
    children = defaultdict(list)
    parents = {}
    for row in hierarchy:
        p, c = row["parent"], row["child"]
        children[p].append(c)
        parents[c] = p

    all_nodes = set(parents) | set(children)
    root_candidates = [n for n in all_nodes if n not in parents]
    root = root_candidates[0] if root_candidates else None
    if root is None:
        print("❌ No root cluster found.")
        return

    # BFS for max depth
    max_depth = 0
    visited = set()
    queue = deque([(root, 0)])
    while queue:
        node, depth = queue.popleft()
        visited.add(node)
        max_depth = max(max_depth, depth)
        for child in children.get(node, []):
            if child not in visited:
                queue.append((child, depth + 1))

    # Degree statistics
    degree_list = [len(children[n]) for n in children]
    print("🧬 HDBSCAN Condensed Cluster Tree")
    print(f"• Root cluster: {root}")
    print(f"• Total clusters (nodes): {len(all_nodes)}")
    print(f"• Maximum hierarchy depth: {max_depth}")
    print(f"• Branching factor: min={min(degree_list)}, max={max(degree_list)}, avg={sum(degree_list)/len(degree_list):.2f}")
    print(f"• Leaf clusters (no children): {sum(1 for n in all_nodes if n not in children)}")


@click.command()
@click.option("--art", "--artifact-id", help="ID of saved topic:hierarchy artefact.")
def main(art: str | None):
    if not art:
        art = interactive_select()
    hierarchy = load_hierarchy_artifact(art)
    #plot_hierarchy_tree(hierarchy)
    analyze_hdbscan_tree(hierarchy)

if __name__ == "__main__":
    main()
