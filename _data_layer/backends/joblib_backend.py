import joblib
from pathlib import Path

_ROOT = Path("./artifacts/joblib"); _ROOT.mkdir(parents=True, exist_ok=True)

def save(df, artifact_id):
    path = _ROOT / f"{artifact_id}.joblib"
    joblib.dump(df, path, compress=3)
    return str(path)

def load(path_or_str):
    return joblib.load(Path(path_or_str))
