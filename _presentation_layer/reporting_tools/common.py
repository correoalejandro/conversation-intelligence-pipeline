# _presentation_layer/common.py

"""
Common rendering utilities for CLI, Web UI, and SAS.
"""

def format_table_summary(df, max_rows=5):
    """Return a compact preview of a DataFrame as a string (CLI or logs)."""
    return df.head(max_rows).to_string(index=False)

def generate_metadata_summary(meta: dict) -> str:
    """Format metadata dict into a human-readable summary."""
    return "\n".join(f"{k}: {v}" for k, v in meta.items())
