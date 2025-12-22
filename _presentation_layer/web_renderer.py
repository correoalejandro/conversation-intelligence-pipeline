# 📄 _presentation_layer/web_renderer.py (CLEANED WITH V1 STYLE)
import streamlit as st
import pandas as pd
import plotly.express as px

def render_umap_plot(df, cluster_col="cluster"):
    """Render UMAP coordinates with cluster coloring."""
    fig = px.scatter(
        df, x="umap_x", y="umap_y", color=cluster_col,
        title="UMAP Projection of Clusters"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_data_table(df):
    """Render a DataFrame in Streamlit."""
    st.dataframe(df)

def render_metadata(meta: dict):
    """Display artifact metadata in Streamlit."""
    st.json(meta)

# ─────────────────────────────────────────────────────────────────────────────
# PRESENTATION‑LAYER  │  web_renderer.py
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd

def render_artifact_table(
        df: pd.DataFrame | None,
        on_select_callback=None,
        use_container_width: bool = True
) -> None:
    """
    Muestra un DataFrame de artefactos en Streamlit y, opcionalmente,
    ejecuta un callback cuando el usuario selecciona una fila.

    Parámetros
    ----------
    df : pd.DataFrame | None
        Tabla de artefactos ya normalizada (una fila por artefacto).
    on_select_callback : callable | None, opcional
        Función a ejecutar cuando el usuario elige un artefacto.
        Se le pasa como argumento la serie (fila) seleccionada.
    use_container_width : bool, opcional
        Se pasa directamente a `st.dataframe`.

    Notas
    -----
    * Si `df` es None o está vacío, se muestra una advertencia y se sale.
    * Para la selección de fila se usa un `st.radio` con los índices
      del DataFrame; esto es simple y evita dependencias externas.
    """
    # ── validaciones ────────────────────────────────────────────────────────
    if df is None:
        st.warning("⚠️ No artifacts found.")
        return
    if not isinstance(df, pd.DataFrame):
        st.error("❌ `render_artifact_table` espera un DataFrame.")
        return
    if df.empty:
        st.warning("⚠️ No artifacts found.")
        return

    # ── tabla ──────────────────────────────────────────────────────────────
    st.dataframe(df, use_container_width=use_container_width)

    # ── selección + callback ───────────────────────────────────────────────
    if callable(on_select_callback):
        st.markdown("### Select an artifact:")
        idx = st.radio(
            label="Choose row index",
            options=df.index.tolist(),
            horizontal=True,
            key="artifact_table_radio"
        )
        if idx is not None:
            on_select_callback(df.loc[idx])





def render_artifact_viewer(df, meta):
    """
    Render metadata and adapt visualization based on artifact type.
    """
    st.subheader("📄 Metadata")
    st.json(meta)

    stage = meta.get("stage", "").lower()

    if "umap" in stage or ("umap_x" in df.columns and "umap_y" in df.columns):
        st.subheader("🌐 UMAP Projection")
        fig = px.scatter(
            df, x="umap_x", y="umap_y",
            color=df["label"] if "label" in df.columns else None,
            title="UMAP Projection"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif "topic" in stage:
        st.subheader("🗝 Topic Overview")
        if "keywords" in df.columns:
            st.dataframe(df[["cluster_id", "keywords", "size"]], use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)

    else:
        st.subheader("📊 Data Preview")
        st.dataframe(df, use_container_width=True)