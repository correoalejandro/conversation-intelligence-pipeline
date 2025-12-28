# _data_layer/paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
REGISTRY_DIR = PROJECT_ROOT / "registries"
DATA_DIR = PROJECT_ROOT / "data"

ARTIFACT_DIR = DATA_DIR / "artifacts"
EXPERIMENT_DIR = DATA_DIR / "experiments"
MODEL_DIR = DATA_DIR / "models"
LOG_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
