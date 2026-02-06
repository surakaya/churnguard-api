from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

from app.config import settings

BASE_DIR = Path(__file__).resolve().parent.parent


def validate_metadata(metadata: Dict[str, Any]) -> None:
    # Metadata şemasının minimum zorunlu alanlarını doğrular.
    required = {
        "model_name",
        "version",
        "roc_auc",
        "trained_on",
        "features",
        "trained_at",
        "notes",
    }
    missing = required - set(metadata.keys())
    if missing:
        raise ValueError(f"Metadata is missing required fields: {sorted(missing)}")


def load_model_and_metadata(model_version: str | None = None) -> Tuple[Any, Dict[str, Any]]:
    version = model_version or settings.MODEL_VERSION
    model_dir = BASE_DIR / "models" / version
    model_path = model_dir / "model.pkl"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    metadata = json.loads(metadata_path.read_text())
    validate_metadata(metadata)

    return model, metadata
