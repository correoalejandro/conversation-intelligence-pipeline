from pathlib import Path

def save(data: str, artifact_id: str):
    path = Path("data/artifacts") / f"{artifact_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(data)
    return str(path)

def load(path: str) -> str:
    with Path(path).open("r", encoding="utf-8") as f:
        return f.read()
