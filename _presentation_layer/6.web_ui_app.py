# 📄 4.v6.web_ui_app.py (CLICKABLE TABLE & SYNCHRONIZED VIEWER)
import joblib
from _data_layer.api import load_registry, load_artifact
from _data_layer.registry import find
from _data_layer.api import _backend   # ← loader oficial (json, joblib, etc.)

from _presentation_layer.web_renderer import (
    render_artifact_table,
    render_artifact_viewer,
    render_data_table,
    render_metadata,
    render_umap_plot
)
import pandas as pd
from pathlib import Path
import json
import plotly.express as px
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import streamlit as st
from _data_layer.api import load_artifact
#from _data_layer.registry import find  # ✅ add this
# Sidebar navigation
st.sidebar.title("Navigation")

# ---------------------------------------------------------------------------
# Carga & fusión de todos los technical_registry.json
# ---------------------------------------------------------------------------

def load_all_technical_registries(base_dir: str = "registries") -> pd.DataFrame:
    """
    Devuelve un DataFrame con TODOS los artefactos técnicos encontrados
    en cualquier subcarpeta dentro de *base_dir*, sin importar si cada
    technical_registry.json es:
        • un dict por secciones (generator, embeddings, …)   ó
        • una lista de artefactos ya “planos”
    Siempre garantiza las columnas: stage, created_at, backend, data_ref, pipeline.
    """
    def _norm(art: dict, stage: str, pipeline: str, reg_defaults: dict | None = None) -> dict:
        """Homogeneiza campos para la UI."""
        reg_defaults = reg_defaults or {}
        art = art.copy()
        art["stage"]      = art.get("stage")      or stage or "unknown"
        art["pipeline"]   = pipeline
        art["created_at"] = (art.get("created_at")
                             or art.get("generated_at")
                             or reg_defaults.get("generated_at"))
        art["backend"]    = (art.get("backend")
                             or art.get("model")
                             or art.get("algorithm")
                             or art.get("type"))
        art["data_ref"] = art.get("data_ref") or art.get("id")
        return art

    artifacts: list[dict] = []
    base_path = Path(base_dir)

    if not base_path.exists():
        st.error(f"❌ La carpeta '{base_dir}' no existe.")
        return pd.DataFrame()

    for tech_path in base_path.rglob("technical_registry.json"):
        try:
            reg = json.load(tech_path.open(encoding="utf-8"))
        except Exception as e:
            st.warning(f"⚠️ No se pudo leer {tech_path}: {e}")
            continue

        pipeline = tech_path.parent.name  # ej. “pipeline_A”

        # Caso 1: registry = dict por secciones
        if isinstance(reg, dict):
            for stage in ("generator", "preprocessing", "embeddings",
                          "clusters", "topics", "evaluation"):
                art = reg.get(stage)
                if art:
                    artifacts.append(_norm(art, stage, pipeline, reg))

            # Dict que trae una clave “artifacts”: [ … ]
            if isinstance(reg.get("artifacts"), list):
                for art in reg["artifacts"]:
                    artifacts.append(_norm(art, art.get("stage"), pipeline, reg))

        # Caso 2: registry = lista de artefactos “planos”
        elif isinstance(reg, list):
            for art in reg:
                artifacts.append(_norm(art, art.get("stage"), pipeline))

        else:
            st.warning(f"⚠️ Formato no reconocido en {tech_path}")

    # ---- DataFrame final ----
    df = pd.DataFrame(artifacts)
    for col in ["stage", "created_at", "backend", "data_ref", "pipeline"]:
        if col not in df.columns:
            df[col] = None
    return df

# -----------------------------------------------------------------
# 🔄  Cargar bundle (.joblib) con heurísticas de nombres
# -----------------------------------------------------------------
DATA_FOLDER = Path("./data/experiments")   # ajusta si tu ruta cambia

def load_bundle_joblib(artifact: dict | None):
    """
    Busca un .joblib relacionado con *artifact* e intenta cargarlo.
    Devuelve el bundle o None.
    """
    if not artifact:
        return None

    # Candidatos de búsqueda
    candidates: list[str] = []
    data_ref = str(artifact.get("data_ref", ""))
    digest   = str(artifact.get("hash", ""))
    stage    = str(artifact.get("stage", ""))

    if data_ref:
        candidates += [data_ref, data_ref.split("_")[0]]
    if digest:
        candidates.append(digest)
    if stage:
        candidates.append(stage)

    # Buscar coincidencias
    for pattern in candidates:
        for path in DATA_FOLDER.glob("*.joblib"):
            if pattern and pattern in path.stem:
                try:
                    return joblib.load(path)
                except Exception as e:
                    st.warning(f"⚠️ Error al cargar {path.name}: {e}")
                    return None
    return None


# 📂 Load registry once
artifact_df = load_all_technical_registries()

# 🔄 Session state to track selected artifact
if "selected_artifact_id" not in st.session_state:
    st.session_state.selected_artifact_id = None

# ✅ nueva versión (fila completa → ID)
def on_artifact_select(row):
    st.session_state.selected_artifact_id = row["data_ref"]
    st.session_state.page = "Artifact Viewer"
# Page selector with remembered state
if "page" not in st.session_state:
    st.session_state.page = "Artifact Browser"

page = st.sidebar.radio(
    "Go to", [
        "Artifact Browser",
        "Artifact Viewer",
        "Conversation Viewer",
        "Conversation Timeline"  # ✅ new page
    ],
    index=[
        "Artifact Browser",
        "Artifact Viewer",
        "Conversation Viewer",
        "Conversation Timeline"
    ].index(st.session_state.page)
)



# ────────────────────────────────────────────────────────────────
# 🔍  Artifact Browser 
# ────────────────────────────────────────────────────────────────
# 🔧 columnas visibles en el Artifact Browser
display_cols = [
    c for c in ("stage", "created_at", "backend", "data_ref", "pipeline")
    if c in artifact_df.columns
]

if page == "Artifact Browser":
    st.title("📚 Artifact Browser")
    render_artifact_table(
        artifact_df[display_cols],
        on_select_callback=on_artifact_select   # clic en fila → viewer
    )


# ────────────────────────────────────────────────────────────────
# 🔍  Artifact Viewer  (con vista‑proyector)
# ────────────────────────────────────────────────────────────────
elif page == "Artifact Viewer":
    st.title("🔍 Artifact Viewer")

    # 1️⃣ Recuperar el ID seleccionado
    artifact_id = st.session_state.get("selected_artifact_id")
    if not artifact_id:
        st.info("Selecciona un artefacto desde *Artifact Browser*.")
        st.stop()

    # 2️⃣ Mostrar detalles del registro
    sel = artifact_df[artifact_df["data_ref"] == artifact_id]
    if sel.empty:
        st.error(f"Artefacto '{artifact_id}' no encontrado en los registries.")
        st.stop()
    artifact_dict = sel.iloc[0].dropna().to_dict()
    st.subheader("📄 Registro del artefacto")
    st.json(artifact_dict, expanded=True)

    # 3️⃣ Cargar el .joblib correcto
    st.markdown("## 🧠 Artifact Projection")
    from pathlib import Path
    import joblib

    DATA_FOLDER = Path("data/experiments")
    raw_ref = artifact_dict["data_ref"]
    path = Path(raw_ref)

    # Si data_ref no contiene “.joblib” o el archivo no existe, buscamos en data/experiments
    if not (raw_ref.endswith(".joblib") and path.exists()):
        matches = list(DATA_FOLDER.glob(f"*{artifact_id}*.joblib"))
        if matches:
            path = matches[0]
        else:
            st.warning(f"⚠️ No se encontró ningún .joblib en '{DATA_FOLDER}' que contenga '{artifact_id}'.")
            df = None

    try:
        if path and path.exists():
            bundle = joblib.load(path)
            df = bundle.get("df")
        else:
            df = None
    except Exception as e:
        st.warning(f"⚠️ Error cargando {path.name}: {e}")
        df = None



    ##########################################
    # 4️⃣ Scatter interactivo con Plotly (mejorado)
    
    #---------------------------------------# 
    
    
    # Prepara df_copy con etiquetas legibles
    if df is None:
        st.warning("⚠️ No hay DataFrame proyectable para este artefacto.")
        st.stop()
    df_copy = df.copy()
    df_copy["cluster"] = pd.to_numeric(df_copy["cluster"], errors="coerce").fillna(-1).astype(int)
    df_copy["cluster_label"] = df_copy["cluster"].apply(
        lambda c: "Outlier" if c == -1 else f"Cluster {c:03d}"
    )

    # Orden y lista de clústeres
    unique_clusters = sorted(set(df_copy["cluster"]))
    clusters_no_outlier = [c for c in unique_clusters if c != -1]
    ordered_labels = ["Outlier"] + [f"Cluster {c:03d}" for c in clusters_no_outlier]

    # 1. Generar colores HSV equiespaciados para los no-outliers
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    n = len(clusters_no_outlier)
    hsv_cmap = cm.hsv
    hsv_colors = [mcolors.to_hex(hsv_cmap(i / max(n, 1))) for i in range(n)]

    color_map = {"Outlier": "#CCCCCC"}
    for idx, c in enumerate(clusters_no_outlier):
        color_map[f"Cluster {c:03d}"] = hsv_colors[idx]


    # 2. Configurar símbolos si no hay demasiados clústeres
    symbol_args = {}
    if n <= 12:
        symbol_args = {
            "symbol": "cluster_label",
            "symbol_sequence": ['circle','square','diamond','cross','x','triangle-up','triangle-down']
        }  # :contentReference[oaicite:3]{index=3}

    # 3. Crear scatter
    fig = px.scatter(
        df_copy,
        x="umap_x",
        y="umap_y",
        color="cluster_label",
        color_discrete_map=color_map,
        category_orders={"cluster_label": ordered_labels},
        hover_name="conversation_id",
        render_mode="webgl",
        **symbol_args
    )

    # 4. Añadir botones para seleccionar/deseleccionar
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "direction": "right",
            "x": 1.1,
            "y": 1.15,
            "showactive": False,
            "buttons": [
                {
                    "label": "Deseleccionar todos",
                    "method": "update",
                    "args": [{"visible": ["legendonly"] * len(fig.data)}, {}]
                },
                {
                    "label": "Seleccionar todos",
                    "method": "update",
                    "args": [{"visible": [True] * len(fig.data)}, {}]
                }
            ]
        }],
        legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
        template="plotly_white",
        height=600,
        title="🌌 UMAP Interactive Scatter Plot"
    )

    # 5. Renderizar en Streamlit
    st.plotly_chart(fig, use_container_width=True)



# 🗣 Conversation Viewer (independent of artifacts)


elif page == "Conversation Viewer":
    st.title("🗣 Global Conversation Viewer")

    # Combine all generator batches
    generator_artifacts = find(stage="generator")
    dfs_all = []
    for art in generator_artifacts:
        df, meta = load_artifact(art["id"])
        if "conversation_id" in df.columns:
            dfs_all.append(df)
        else:
            st.warning(f"⚠ Skipping artifact {meta['id']} (no 'conversation_id' column)")
    if not dfs_all:
        st.error("❌ No generator artifacts with 'conversation_id' found.")
    else:
        df_all = pd.concat(dfs_all, ignore_index=True)

        # Also try merging cleaned text if available
        clean_artifacts = find(stage="preprocess:cleaned_json")
        dfs_clean = []
        for art in clean_artifacts:
            df_clean, _ = load_artifact(art["id"])
            if "conversation_id" in df_clean.columns:
                dfs_clean.append(df_clean)
        df_clean_all = pd.concat(dfs_clean, ignore_index=True) if dfs_clean else None

        st.markdown(f"📖 Loaded {len(df_all)} conversations from all batches.")

        # 🔽 Dropdown for conversation_id
        conversation_ids = sorted(df_all["conversation_id"].unique())
        selected_id = st.selectbox(
            "🔍 Select a conversation_id:",
            conversation_ids,
            help="Pick a conversation ID to view its details"
        )

        # Show results from generator data
        matches = df_all[df_all["conversation_id"] == selected_id]
        st.success(f"✅ Found {len(matches)} conversation(s) in generator data.")
        st.dataframe(matches)

        # Show results from cleaned text if available
        if df_clean_all is not None:
            clean_matches = df_clean_all[df_clean_all["conversation_id"] == selected_id]
            if not clean_matches.empty:
                st.info(f"✨ Found {len(clean_matches)} cleaned version(s).")
                st.dataframe(clean_matches)
            else:
                st.warning("⚠ No cleaned version found for this conversation.")

        st.markdown("---")
        st.caption("This viewer works independently of artifacts.")




# 📈 Conversation Timeline Viewer
elif page == "Conversation Timeline":
    st.title("📈 Conversation Timeline")

    # Load all generator batches
    generator_artifacts = find(stage="generator")
    dfs_all = []
    for art in generator_artifacts:
        df, meta = load_artifact(art["id"])
        if "conversation_id" in df.columns:
            dfs_all.append(df)
        else:
            st.warning(f"⚠ Skipping artifact {meta['id']} (no 'conversation_id' column)")

    if not dfs_all:
        st.error("❌ No generator artifacts with 'conversation_id' found.")
    else:
        df_all = pd.concat(dfs_all, ignore_index=True)

        # 🔽 Dropdown for conversation_id
        conversation_ids = sorted(df_all["conversation_id"].unique())
        selected_id = st.selectbox(
            "🔍 Select a conversation_id to explore:",
            conversation_ids,
            help="Pick a conversation to view its timeline"
        )

        # Get the selected conversation
        selected_conv = df_all[df_all["conversation_id"] == selected_id]
        if selected_conv.empty:
            st.error("❌ Could not find conversation.")
        else:
            st.markdown(f"📖 **Conversation ID:** `{selected_id}`")

            # Try extracting message-level timestamps
            timestamps = []
            if "messages" in selected_conv.columns:
                messages = selected_conv.iloc[0]["messages"]
                if isinstance(messages, list) and "timestamp" in messages[0]:
                    timestamps = [msg["timestamp"] for msg in messages if "timestamp" in msg]

            if timestamps:
                st.success(f"✅ Found {len(timestamps)} message timestamps.")
                ts_df = pd.DataFrame({
                    "Timestamp": pd.to_datetime(timestamps),
                    "Message #": range(1, len(timestamps) + 1)
                })

                # 🪟 Side-by-side collapsible panels
                col1, col2 = st.columns(2)

                with col1:
                    with st.expander("📋 Message Timestamps Table", expanded=True):
                        st.dataframe(ts_df, use_container_width=True)

                with col2:
                    with st.expander("📈 Timeline Plot", expanded=True):
                        st.line_chart(ts_df.set_index("Timestamp")["Message #"])
            else:
                st.warning("⚠ No message-level timestamps found in this conversation.")

    st.title("🗣 Global Conversation Viewer")

    # Combine all generator batches
    generator_artifacts = find(stage="generator")
    dfs_all = []
    for art in generator_artifacts:
        df, meta = load_artifact(art["id"])
        if "conversation_id" in df.columns:
            dfs_all.append(df)
        else:
            st.warning(f"⚠ Skipping artifact {meta['id']} (no 'conversation_id' column)")
    if not dfs_all:
        st.error("❌ No generator artifacts with 'conversation_id' found.")
    else:
        df_all = pd.concat(dfs_all, ignore_index=True)

        # Also try merging cleaned text if available
        clean_artifacts = find(stage="preprocess:cleaned_json")
        dfs_clean = []
        for art in clean_artifacts:
            df_clean, _ = load_artifact(art["id"])
            if "conversation_id" in df_clean.columns:
                dfs_clean.append(df_clean)
        df_clean_all = pd.concat(dfs_clean, ignore_index=True) if dfs_clean else None

        st.markdown(f"📖 Loaded {len(df_all)} conversations from all batches.")

        # 🔽 Dropdown for conversation_id
        conversation_ids = sorted(df_all["conversation_id"].unique())
        selected_id = st.selectbox(
            "🔍 Select a conversation_id:",
            conversation_ids,
            help="Pick a conversation ID to view its details"
        )

        # Show results from generator data
        matches = df_all[df_all["conversation_id"] == selected_id]
        st.success(f"✅ Found {len(matches)} conversation(s) in generator data.")
        st.dataframe(matches)

        # Show results from cleaned text if available
        if df_clean_all is not None:
            clean_matches = df_clean_all[df_clean_all["conversation_id"] == selected_id]
            if not clean_matches.empty:
                st.info(f"✨ Found {len(clean_matches)} cleaned version(s).")
                st.dataframe(clean_matches)
            else:
                st.warning("⚠ No cleaned version found for this conversation.")

        st.markdown("---")
        st.caption("This viewer works independently of artifacts.")


    st.title("📈 Conversation Timeline")

    # Load all generator batches
    generator_artifacts = find(stage="generator")
    dfs_all = []
    for art in generator_artifacts:
        df, meta = load_artifact(art["id"])
        if "conversation_id" in df.columns:
            dfs_all.append(df)
        else:
            st.warning(f"⚠ Skipping artifact {meta['id']} (no 'conversation_id' column)")

    if not dfs_all:
        st.error("❌ No generator artifacts with 'conversation_id' found.")
    else:
        df_all = pd.concat(dfs_all, ignore_index=True)

        # 🔽 Dropdown for conversation_id
        conversation_ids = sorted(df_all["conversation_id"].unique())
        selected_id = st.selectbox(
            "🔍 Select a conversation_id to explore:",
            conversation_ids,
            help="Pick a conversation to view its timeline"
        )

        # Get the selected conversation
        selected_conv = df_all[df_all["conversation_id"] == selected_id]
        if selected_conv.empty:
            st.error("❌ Could not find conversation.")
        else:
            st.markdown(f"📖 **Conversation ID:** `{selected_id}`")

            # Try extracting message-level timestamps
            timestamps = []
            if "messages" in selected_conv.columns:
                messages = selected_conv.iloc[0]["messages"]
                if isinstance(messages, list) and "timestamp" in messages[0]:
                    timestamps = [msg["timestamp"] for msg in messages if "timestamp" in msg]

            if timestamps:
                st.success(f"✅ Found {len(timestamps)} message timestamps.")
                ts_df = pd.DataFrame({
                    "Timestamp": pd.to_datetime(timestamps),
                    "Message #": range(1, len(timestamps) + 1)
                })

                # 🪟 Side-by-side collapsible panels
                col1, col2 = st.columns(2)

                with col1:
                    with st.expander("📋 Message Timestamps Table", expanded=True):
                        st.dataframe(ts_df, use_container_width=True)

                with col2:
                    with st.expander("📈 Timeline Plot", expanded=True):
                        st.line_chart(ts_df.set_index("Timestamp")["Message #"])
            else:
                st.warning("⚠ No message-level timestamps found in this conversation.")

    st.title("📈 Conversation Timeline")

    # Load all generator batches
    generator_artifacts = find(stage="generator")
    dfs_all = []
    for art in generator_artifacts:
        df, meta = load_artifact(art["id"])
        if "conversation_id" in df.columns:
            dfs_all.append(df)
        else:
            st.warning(f"⚠ Skipping artifact {meta['id']} (no 'conversation_id' column)")

    if not dfs_all:
        st.error("❌ No generator artifacts with 'conversation_id' found.")
    else:
        df_all = pd.concat(dfs_all, ignore_index=True)

        # 🔽 Dropdown for conversation_id
        conversation_ids = sorted(df_all["conversation_id"].unique())
        selected_id = st.selectbox(
            "🔍 Select a conversation_id to plot:",
            conversation_ids,
            help="Pick a conversation to explore its timeline"
        )

        # Get the selected conversation
        selected_conv = df_all[df_all["conversation_id"] == selected_id]
        if selected_conv.empty:
            st.error("❌ Could not find conversation.")
        else:
            st.markdown(f"📖 **Conversation ID:** `{selected_id}`")

            # Try extracting message-level timestamps
            timestamps = []
            if "messages" in selected_conv.columns:
                messages = selected_conv.iloc[0]["messages"]
                if isinstance(messages, list) and "timestamp" in messages[0]:
                    timestamps = [msg["timestamp"] for msg in messages if "timestamp" in msg]

            # Show timestamps table
            if timestamps:
                st.success(f"✅ Found {len(timestamps)} message timestamps.")
                ts_df = pd.DataFrame({
                    "Timestamp": pd.to_datetime(timestamps),
                    "Message #": range(1, len(timestamps) + 1)
                })
                st.dataframe(ts_df)

                # Plot time series
                st.line_chart(ts_df.set_index("Timestamp")["Message #"])
            else:
                st.warning("⚠ No message-level timestamps found in this conversation.")

            # Optional: check for conversation-level timestamp
            if "timestamp" in selected_conv.columns:
                conv_ts = selected_conv.iloc[0]["timestamp"]
                st.info(f"📅 Conversation-level timestamp: `{conv_ts}`")