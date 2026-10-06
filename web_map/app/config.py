"""Settings for the web map.

Secrets and asset ids come from the repository's .env file locally, or from
environment variables when deployed (e.g. on Vercel). The class table lives
next to this file in app/data/, built by scripts/build_tables.py, so it ships
with the server.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

WEB_MAP_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = WEB_MAP_DIR.parent
PUBLIC_DIR = WEB_MAP_DIR / "public"
TABLES_DIR = Path(__file__).resolve().parent / "data"

# On Vercel the platform serves public/ from its CDN; the app only answers /api.
ON_VERCEL = bool(os.environ.get("VERCEL"))

load_dotenv(REPO_DIR / ".env")
load_dotenv(WEB_MAP_DIR / ".env", override=True)


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to {REPO_DIR / '.env'} or to the host's environment variables.")
    return value


@dataclass(frozen=True)
class Settings:
    project_id: str
    service_email: str
    private_key: str
    classcode_asset: str
    confidence_asset: str
    forest_asset: str
    conf_threshold: float


@lru_cache
def settings() -> Settings:
    return Settings(
        project_id=_required("PROJECT_ID"),
        service_email=_required("SERVICE_EMAIL"),
        # Keys pasted into hosting dashboards often keep the .env quotes or literal "\n" sequences.
        private_key=_required("PRIVATE_KEY").strip('"\'').replace("\\n", "\n"),
        classcode_asset=_required("PREDICTED_CLASSCODE"),
        confidence_asset=_required("PREDICTED_CONFIDENCE"),
        forest_asset=os.environ.get("FOREST_2020_ASSET") or "JRC/GFC2020/V4",
        conf_threshold=float(os.environ.get("CONF_THRESHOLD") or 0.6),
    )


@lru_cache
def classes() -> dict:
    return json.loads((TABLES_DIR / "classes.json").read_text(encoding="utf-8"))
