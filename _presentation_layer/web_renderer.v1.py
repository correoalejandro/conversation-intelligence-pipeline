# _presentation_layer/web_renderer.py

"""
Web Renderer
------------
Components and helpers to render artifacts in Web UIs
(Streamlit, Plotly, etc.).
"""

import plotly.express as px
import streamlit as st
import pandas as pd

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
    """Render artifact metadata as expandable JSON."""
    st.subheader("📄 Metadata")
    if meta:
        st.json(meta)
    else:
        st.warning("⚠️ No metadata found in this artifact.")

def render_artifact_table(registry, on_select_callback=None):
    """
    Render the artifact registry as a single table with a 'View' button per row.
    """
    if not registry:
        st.warning("⚠️ No artifacts found.")
        return

    df = pd.DataFrame(registry)

    # Filter by stage
    stages = df["stage"].unique().tolist()
    selected_stages = st.multiselect("Filter by Stage", stages, default=stages)
    filtered_df = df[df["stage"].isin(selected_stages)].reset_index(drop=True)

    st.subheader("🗂 Artifact Registry")

    # Add a dummy column for buttons
    for idx in filtered_df.index:
        button_label = f"🔍 View {filtered_df.loc[idx, 'stage']}"
        if st.button(button_label, key=f"view_{filtered_df.loc[idx, 'id']}"):
            if on_select_callback:
                on_select_callback(filtered_df.loc[idx, 'id'])

    # Show table
    st.dataframe(
        filtered_df.drop(columns=[]),  # Show original columns only
        use_container_width=True
    )  


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
