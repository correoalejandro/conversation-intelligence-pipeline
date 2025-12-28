from _data_layer.registry import add, exists
from pathlib import Path
import joblib
import hashlib
import json
from datetime import datetime, timezone

# 🔄 Path to your legacy artifacts
EXPERIMENTS_DIR = Path("data/experiments/")
BATCHES_DIR = Path("data/batches/")

def file_hash(file: Path, block_size=65536):
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hasher.update(block)
    return hasher.hexdigest()

def infer_stage(file: Path):
    """Guess stage from filename."""
    if "preprocess" in file.name.lower():
        return "preprocess"
    if "embedding" in file.name.lower():
        return "vectorizer"
    if "umap" in file.name.lower():
        return "reducer"
    if "cluster" in file.name.lower():
        return "classifier"
    if "batch" in file.name.lower():
        return "generator"
    return "processPipeline"

def infer_parameters(bundle: dict):
    """Extract params if present in joblib file."""
    return bundle.get("meta", {}).get("parameters", {})

def import_joblib_artifact(file: Path):
    print(f"📦 Importing {file.name}...")
    bundle = joblib.load(file)
    file_digest = file_hash(file)

    if "metadata" in bundle:
        full_metadata = bundle["metadata"]
    else:
        full_metadata = {}

    metadata = {
        "id": file.stem,
        "stage": infer_stage(file),
        "backend": "joblib",
        "data_ref": str(file.resolve()),
        "hash": file_digest,
        "parameters": infer_parameters(bundle),
        "metadata": full_metadata,  # 🟢 Store original metadata here
        "parents": [],
        "created_at": datetime.utcfromtimestamp(file.stat().st_mtime).isoformat(),
    }


def import_json_batch(file: Path):
    print(f"📦 Importing batch {file.name}...")
    file_digest = file_hash(file)

    try:
        with open(file, "r", encoding="utf-8") as f:
            batch_json = json.load(f)

        if isinstance(batch_json, list):
            # 🗂️ Old format: flat list of conversations
            conversations = batch_json
            parameters = {}
        else:
            # 🗂️ New format: meta + conversations
            conversations = batch_json.get("conversations", [])
            parameters = infer_parameters(batch_json)

    except Exception as e:
        print(f"⚠️ Failed to load batch metadata: {e}")
        conversations = []
        parameters = {}

    metadata = {
        "id": file.stem.replace("_batch", ""),
        "stage": "generator",
        "backend": "json",
        "data_ref": str(file.resolve()),
        "hash": file_digest,
        "parameters": parameters,
        "parents": [],
        "created_at": datetime.fromtimestamp(file.stat().st_mtime, timezone.utc).isoformat(),
    }

    if exists(metadata):
        print(f"⚠️ Skipping duplicate: {file.name}")
    else:
        add(metadata)
        print(f"✅ Registered: {file.name}")

        # 🆕 Enrich business registry with client/agent info
        business_registry_path = Path("_data_layer") / "business_registry.json"
        business_lock_file = Path("_data_layer") / "business_registry.lock"

        # Load existing registry or initialize
        if business_registry_path.exists():
            with open(business_registry_path, "r", encoding="utf8") as bf:
                business_registry = json.load(bf)
        else:
            business_registry = {}

        for conv in conversations:
            if not isinstance(conv, dict):
                print(f"⚠️ Skipping invalid conversation in batch {file.name} (expected dict, got {type(conv).__name__})")
                continue

            conv_id = conv.get("conversation_id")
            messages = conv.get("messages", [])

            if not conv_id or not isinstance(messages, list) or not messages:
                print(f"⚠️ Skipping malformed conversation in batch {file.name} (id={conv_id})")
                continue

            # Safe client/agent name extraction
            client_name = next(
                (msg["text"].split()[0].strip(",.:") for msg in messages if msg.get("sender", "").lower() == "cliente"),
                "Desconocido"
            )
            agent_name = next(
                (msg["text"].split()[0].strip(",.:") for msg in messages if msg.get("sender", "").lower() == "agente"),
                "Desconocido"
            )

            start_time = messages[0]["timestamp"]
            end_time = messages[-1]["timestamp"]
            num_turns = len(messages)

            business_registry[conv_id] = {
                "conversation_id": conv_id,
                "client_name": client_name,
                "agent_name": agent_name,
                "start_time": start_time,
                "end_time": end_time,
                "num_turns": num_turns,
                "artifact_id": metadata["id"]
            }

        # Save updated business registry with file lock
        from filelock import FileLock
        with FileLock(str(business_lock_file)):
            with open(business_registry_path, "w", encoding="utf8") as bf:
                json.dump(business_registry, bf, indent=2, ensure_ascii=False)

        print(f"✅ Updated business registry with {len(conversations)} conversations.")



def import_all():
    joblib_files = list(EXPERIMENTS_DIR.glob("*.joblib"))
    json_batches = list(BATCHES_DIR.glob("*.json"))

    if not joblib_files and not json_batches:
        print("⚠️ No .joblib or .json files found.")
        return

    #for f in joblib_files:
    #   import_joblib_artifact(f)
    
    for f in json_batches:
        import_json_batch(f)

    print("✅ Import complete.")

if __name__ == "__main__":
    import_all()
