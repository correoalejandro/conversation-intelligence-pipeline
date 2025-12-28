from typing import List, Dict, Any
import os

from pathlib import Path
import importlib
import json
import joblib
import pandas as pd
from _data_layer import registry, models


# Map a backend name to its module path
_BACKENDS = {
    "txt": "_data_layer.backends.txt_backend",
    "json":    "_data_layer.backends.json_backend", 
    "joblib":  "_data_layer.backends.joblib_backend",
    "parquet": "_data_layer.backends.parquet_backend",
    "mongo":   "_data_layer.backends.mongo_backend",
}

def _backend(backend_name):
    return importlib.import_module(_BACKENDS[backend_name])

def save_artifact(df, stage: str, parameters: Dict[str, Any], *,
                  backend="joblib", parents: List[str]=None) -> models.Artifact:
    art_id = registry.new_id()
    data_ref = _backend(backend).save(df, art_id)

    art = models.Artifact(
        id=art_id,
        stage=stage,
        backend=backend,
        data_ref=data_ref,
        parameters=parameters,
        parents=parents or [],
    )
    registry.add(art.__dict__)
    return art

def load_artifact(artifact_id: str):
    """
    Load an artifact by ID or stage and normalize it to a DataFrame if possible.
    """
    # 🗂 Find artifact by ID
    recs = registry.find()
    rec = next((r for r in recs if r.get("id") == artifact_id), None)




    # 📥 If not found by ID, check latest for stage
    if rec is None:
        rec = registry.latest(artifact_id)

    if rec is None:
        raise ValueError(f"❌ No artifact found with ID or stage '{artifact_id}'")

    raw_data = _backend(rec["backend"]).load(rec["data_ref"])

# 🟢 Try to extract full metadata from joblib bundles if available
    full_meta = rec  # start with registry metadata
    if rec["backend"] == "joblib":
        try:
            bundle = joblib.load(rec["data_ref"])
            if isinstance(bundle, dict) and "metadata" in bundle:
                print("📦 [INFO] Using full metadata from joblib bundle")
                full_meta = bundle["metadata"]
        except Exception as e:
            print(f"⚠️ Could not load full metadata from joblib: {e}")


    # 🟢 Normalize to DataFrame

    def _extract_df_from_list(data: list) -> pd.DataFrame:
        """Try to convert list to DataFrame."""
        if all(isinstance(row, dict) for row in data):
            return pd.DataFrame(data)
        elif all(isinstance(row, str) for row in data):
            return pd.DataFrame({"text": data})
        raise TypeError("❌ Unsupported list format in artifact data.")

    def _extract_df_from_dict(data: dict) -> pd.DataFrame:
        """Extract DataFrame from dict with known patterns."""
        if "df" in data and isinstance(data["df"], pd.DataFrame):
            print("📦 [INFO] Extracted DataFrame from dict['df']")
            return data["df"]
        raise TypeError("❌ Dict artifact does not contain a DataFrame under 'df'.")

    def _extract_df_from_tuple(data: tuple) -> pd.DataFrame:
        """Extract DataFrame from tuple (e.g., (df, metadata))."""
        if isinstance(data[0], pd.DataFrame):
            print("📦 [INFO] Extracted DataFrame from tuple[0]")
            return data[0]
        raise TypeError("❌ Tuple artifact does not contain a DataFrame as first element.")

    # 🌟 Dispatch by type
    if isinstance(raw_data, pd.DataFrame):
        df = raw_data
    elif isinstance(raw_data, list):
        df = _extract_df_from_list(raw_data)
    elif isinstance(raw_data, dict):
        df = _extract_df_from_dict(raw_data)
    elif isinstance(raw_data, tuple):
        df = _extract_df_from_tuple(raw_data)
    else:
        raise TypeError(f"❌ Unsupported data type from backend: {type(raw_data)}")


    return df, rec

def load_registry():
    """Load the artifact registry from disk."""
    with open("registries/technical_registry.json", "r", encoding="utf-8") as f:
        return json.load(f)