from pathlib import Path
import pandas as pd

_ROOT = Path("./artifacts/parquet"); _ROOT.mkdir(parents=True, exist_ok=True)

def save(df, artifact_id):
    path = _ROOT / f"{artifact_id}.parquet"
    df.to_parquet(path, index=False)
    return str(path)

def load(path_or_str):
    return pd.read_parquet(Path(path_or_str))
