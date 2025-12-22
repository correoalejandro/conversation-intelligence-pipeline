import streamlit as st
from collections import defaultdict

from _data_layer.api import load_artifact
from _data_layer.registry import find, _entry_id
from _data_layer.api import _backend   # ← already imported

@st.cache_data
def load_hierarchy(artifact_id: str):
    """
    Robust loader: find the registry row whose primary key matches *artifact_id*
    (id / prompt_id / batch_id / …) and return JSON as list[dict].
    """
    # ① locate registry record
    for rec in find():                      # registry.find()
        if _entry_id(rec) == artifact_id:
            data = _backend(rec["backend"]).load(rec["data_ref"])
            break
    else:
        st.error(f"Artefact {artifact_id} not found.")
        st.stop()

    # ② normalise to list[dict]
    import json, pandas as pd
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf‑8")

    if isinstance(data, list):
        return data
    if isinstance(data, str):
        try:
            return json.loads(data)            # full JSON
        except json.JSONDecodeError:
            return [json.loads(l) for l in data.splitlines() if l.strip()]  # NDJSON
    if isinstance(data, pd.DataFrame):
        return data.to_dict("records")

    st.error("Unsupported hierarchy format."); st.stop()


@st.cache_data
def build_tree_map(hierarchy: list[dict]):
    tree_map = defaultdict(list)
    reverse_map = {}
    for row in hierarchy:
        p, c = row["parent"], row["child"]
        tree_map[p].append((c, row))
        reverse_map[c] = p
    return tree_map, reverse_map

def find_roots(tree_map, reverse_map):
    all_parents = set(tree_map.keys())
    all_children = set(reverse_map.keys())
    roots = list(all_parents - all_children)
    return roots

# ─────────────────────────────────────────────────────────────────────────────
# 🌲 Recursive Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
def show_node(tree_map, node_id: int, level=0):
    """Recursive lazy display without nesting expanders."""
    indent = "&nbsp;" * 4 * level
    label  = f"{indent}▶ Cluster {node_id}"
    opened = st.toggle(label, key=f"node_{node_id}", value=False)

    if opened:
        children = tree_map.get(node_id, [])
        if not children:
            st.markdown(f"{indent}&nbsp;&nbsp;• <i>leaf</i>", unsafe_allow_html=True)
            return

        for child_id, meta in sorted(children):
            name   = meta.get("topic_name") or ", ".join(meta.get("top_tokens", [])[:3])
            size   = meta.get("child_size", "?")
            info   = f"{name} ({size})"
            st.markdown(f"{indent}&nbsp;&nbsp;• **{info}**", unsafe_allow_html=True)
            show_node(tree_map, child_id, level + 1)

# ─────────────────────────────────────────────────────────────────────────────
# 🧭 Streamlit App
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("🌳 HDBSCAN Topic Hierarchy Explorer")

    # Select artefact
    hierarchy_arts = [r for r in find() if r.get("stage") == "topic:hierarchy"]
    if not hierarchy_arts:
        st.warning("No topic:hierarchy artefacts found.")
        return

    art_id = st.selectbox("Select topic:hierarchy artefact", options=[_entry_id(r) for r in hierarchy_arts])
    hierarchy = load_hierarchy(art_id)
    tree_map, reverse_map = build_tree_map(hierarchy)
    roots = find_roots(tree_map, reverse_map)

    st.markdown("### 🌱 Root Clusters")
    for rid in sorted(roots):
        show_node(tree_map, rid)

if __name__ == "__main__":
    main()
