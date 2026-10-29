"""Backend configuration and environment settings."""
import os
from pathlib import Path

# Resolve base paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent.parent

# Bundle directory (can be overridden via env var BUNDLE_DIR)
DEFAULT_BUNDLE = REPO_ROOT / "bundle" / "v1.0.0"
BUNDLE_DIR = Path(os.environ.get("BUNDLE_DIR", str(DEFAULT_BUNDLE)))

# Size and timeout limits
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))  # 20 MB
TIMEOUT_SECONDS = float(os.environ.get("TIMEOUT_SECONDS", 15.0))

# Sample record path for GET /sample
SAMPLE_DIR = BACKEND_DIR / "sample"
