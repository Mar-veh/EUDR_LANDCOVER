"""Earth Engine login with the service account from .env."""
from __future__ import annotations

import threading

import ee
from google.oauth2 import service_account

from .config import settings

SCOPES = [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/cloud-platform",
]

_lock = threading.Lock()
_initialized = False


def init() -> None:
    """Initialise Earth Engine once per process."""
    global _initialized
    if _initialized:
        return
    with _lock:
        if _initialized:
            return
        s = settings()
        credentials = service_account.Credentials.from_service_account_info(
            {
                "type": "service_account",
                "client_email": s.service_email,
                "private_key": s.private_key,
                "token_uri": "https://oauth2.googleapis.com/token",
                "project_id": s.project_id,
            },
            scopes=SCOPES,
        )
        ee.Initialize(credentials=credentials, project=s.project_id)
        _initialized = True
