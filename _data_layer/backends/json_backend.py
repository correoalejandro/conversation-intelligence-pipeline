import json
from pathlib import Path
import pandas as pd

def save(data, artifact_id):
    path = Path("data/artifacts") / f"{artifact_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    # ✅ Convert DataFrame to JSON-serializable format
    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient="records")

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(path)

def load(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
        # ✅ Automatically convert list of dicts back to DataFrame
        if isinstance(data, list) and all(isinstance(row, dict) for row in data):
            return pd.DataFrame(data)
        return data
