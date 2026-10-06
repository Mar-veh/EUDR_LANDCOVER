"""Helpers shared by the Earth Engine export scripts."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee  # noqa: E402

from app.ee_client import init  # noqa: E402


def asset_exists(asset_id: str) -> bool:
    init()
    try:
        ee.data.getAsset(asset_id)
        return True
    except ee.EEException:
        return False


def find_task(description: str) -> dict | None:
    """The most recent task with this description."""
    init()
    tasks = [t for t in ee.data.getTaskList() if t.get("description") == description]
    return max(tasks, key=lambda t: t.get("creation_timestamp_ms", 0)) if tasks else None


def wait_for(description: str, asset_id: str, poll_s: int = 60) -> None:
    """Block until the export behind asset_id has finished; exit on failure."""
    while True:
        if asset_exists(asset_id):
            print(f"{asset_id} is ready.")
            return
        task = find_task(description)
        if task is None:
            sys.exit(f"No export task named '{description}' and no asset {asset_id}. Start the export first.")
        state = task.get("state")
        if state in ("FAILED", "CANCELLED", "CANCEL_REQUESTED"):
            sys.exit(f"Export '{description}' {state}: {task.get('error_message', '')}")
        if state == "COMPLETED" and not asset_exists(asset_id):
            sys.exit(f"Export '{description}' completed but {asset_id} is missing.")
        print(f"{time.strftime('%H:%M:%S')}  {description}: {state}", flush=True)
        time.sleep(poll_s)
