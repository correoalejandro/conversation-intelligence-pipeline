from _presentation_layer.common import format_table_summary

from _presentation_layer.common import generate_metadata_summary
from _data_layer.api import load_registry
import os

def print_experiment_preview(df, meta):
    """Prints a quick preview of an experiment DataFrame."""
    print("=== Experiment Preview ===")
    print(f"Artifact ID: {meta.get('id', 'N/A')}")
    print(f"Created at: {meta.get('created_at', 'N/A')}")
    print(f"Stage: {meta.get('stage', 'N/A')}")
    print()

    # Exclude heavy columns like embeddings
    exclude_cols = {"embedding", "metadata", "hash"}
    display_cols = [col for col in df.columns if col not in exclude_cols]

    preview_df = df[display_cols].head(5)
    print(format_table_summary(preview_df))


def print_experiment_report(df, meta):
    """Prints a detailed report of an experiment DataFrame."""
    print("=== Experiment Report ===")
    print(f"Artifact ID: {meta.get('id', 'N/A')}")
    print(f"Created at: {meta.get('created_at', 'N/A')}")
    print(f"Stage: {meta.get('stage', 'N/A')}")
    print()
    print(f"📊 Rows: {len(df):,}")
    print(f"📂 Columns: {len(df.columns)}")
    print()

    # Show unique clusters (if exists)
    if "cluster" in df.columns:
        print(f"🔖 Clusters: {df['cluster'].nunique()} unique")
        print("Top clusters:")
        print(df["cluster"].value_counts().head(5).to_string())
        print()

    # Show text column stats (if exists)
    text_col = "clean_text" if "clean_text" in df.columns else "text" if "text" in df.columns else None
    if text_col:
        avg_len = df[text_col].str.len().mean()
        print(f"📝 Text column: '{text_col}'")
        print(f"Average text length: {avg_len:.1f} characters")
        print()

    # Exclude embedding column
    exclude_cols = {"embedding", "metadata", "hash"}
    display_cols = [col for col in df.columns if col not in exclude_cols]

def print_registry_summary():
    """Prints a summary of all artifacts grouped by stage."""
    registry = load_registry()
    if not registry:
        print("⚠️ No artifacts found in registry.")
        return

    print("\n=== 📚 Artifact Registry Summary ===")
    print(f"Total artifacts: {len(registry)}")

    # Group artifacts by stage
    stages = {}
    for artifact in registry:
        stage = artifact.get("stage", "unknown")
        stages.setdefault(stage, []).append(artifact)

    # Print each group
    for stage, artifacts in stages.items():
        print(f"\n--- Stage: {stage} ({len(artifacts)} artifacts) ---")
        # Sort by creation time descending
        artifacts.sort(key=lambda a: a.get("created_at", ""), reverse=True)
        for i, a in enumerate(artifacts, 1):
            file_exists = os.path.exists(a.get("data_ref", ""))
            file_status = "✅" if file_exists else "❌ MISSING"
            print(f"{i:02d}. {a['id']} ({a['created_at']}) [{file_status}]")
            print(f"    Backend: {a['backend']}")
            print(f"    Data Ref: {a['data_ref']}")
            parents = a.get("parents", [])
            if parents:
                print(f"    Parents: {', '.join(parents)}")
            else:
                print("    Parents: None")

    # 🔎 Prompt user to view details of a specific artifact
    artifact_id = input("\n🔎 Enter Artifact ID to view details (or press Enter to skip): ").strip()
    if artifact_id:
        matching = [a for a in registry if a["id"] == artifact_id]
        if matching:
            print("\n=== 📝 Artifact Details ===")
            print(generate_metadata_summary(matching[0]))
        else:
            print("❌ Artifact not found.")
    """Prints a summary of all artifacts grouped by stage."""
    registry = load_registry()
    if not registry:
        print("⚠️ No artifacts found in registry.")
        return

    print("\n=== 📚 Artifact Registry Summary ===")
    print(f"Total artifacts: {len(registry)}")

    # Group artifacts by stage
    stages = {}
    for artifact in registry:
        stage = artifact.get("stage", "unknown")
        stages.setdefault(stage, []).append(artifact)

    # Print each group
    for stage, artifacts in stages.items():
        print(f"\n--- Stage: {stage} ({len(artifacts)} artifacts) ---")
        # Sort by creation time descending
        artifacts.sort(key=lambda a: a.get("created_at", ""), reverse=True)
        for i, a in enumerate(artifacts, 1):
            file_exists = os.path.exists(a.get("data_ref", ""))
            file_status = "✅" if file_exists else "❌ MISSING"
            print(f"{i:02d}. {a['id']} ({a['created_at']}) [{file_status}]")
            print(f"    Backend: {a['backend']}")
            print(f"    Data Ref: {a['data_ref']}")
            parents = a.get("parents", [])
            if parents:
                print(f"    Parents: {', '.join(parents)}")
            else:
                print("    Parents: None")
    """Prints a summary of all artifacts in the registry."""
    registry = load_registry()
    print("=== Artifact Registry Summary ===")
    stages = {}
    for artifact in registry:
        stage = artifact.get("stage", "unknown")
        stages.setdefault(stage, []).append(artifact)
    
    for stage, artifacts in stages.items():
        print(f"\n📦 Stage: {stage} ({len(artifacts)} artifacts)")
        for a in artifacts:
            print(f"  - ID: {a['id']}")
            print(f"    Created: {a['created_at']}")
            print(f"    Backend: {a['backend']}")