"""FastAPI app: Earth Engine tile URLs and the pixel inspector.

Run from the web_map folder:

    python -m app            (or: uvicorn app.main:app --port 8000)
"""
from __future__ import annotations

import logging
import mimetypes

import ee
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import analysis, layers
from .config import ON_VERCEL, PUBLIC_DIR, settings

# Windows can map .js to text/plain in the registry; browsers then refuse ES modules.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")

log = logging.getLogger("web_map")

app = FastAPI(title="Ghana EUDR tree-crop map", docs_url="/api/docs", openapi_url="/api/openapi.json")


@app.middleware("http")
async def revalidate_site_files(request, call_next):
    """Without this, browsers may reuse a stale app.css or script after an update."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


@app.exception_handler(ee.EEException)
def earth_engine_error(_request, exc: ee.EEException):
    log.warning("Earth Engine error: %s", exc)
    return JSONResponse(status_code=502, content={"detail": f"Earth Engine: {exc}"})


def _threshold(value: float | None) -> float:
    """Default threshold, or the given one snapped to steps of 0.05, which bounds the tile cache."""
    if value is None:
        return settings().conf_threshold
    return round(round(value * 20) / 20, 2)


@app.get("/api/config")
def get_config():
    s = settings()
    return {
        "threshold": s.conf_threshold,
        "forest_dataset": s.forest_asset,
    }


@app.get("/api/layers/{name}")
def get_layer(
    name: str,
    threshold: float | None = Query(None, ge=0, le=1),
    fade: bool = False,
):
    params = {
        "landcover": {"fade": fade},
        "eudr": {"threshold": _threshold(threshold)},
    }.get(name, {})
    try:
        url = layers.tile_url(name, **params)
    except KeyError:
        raise HTTPException(404, f"Unknown layer '{name}'.")
    return {"name": name, "url": url, "ttl_s": layers.TILE_TTL_S, **params}


@app.get("/api/point")
def get_point(
    lon: float = Query(..., ge=-180, le=180),
    lat: float = Query(..., ge=-90, le=90),
    threshold: float | None = Query(None, ge=0, le=1),
):
    return analysis.point_info(lon, lat, _threshold(threshold))


if ON_VERCEL:
    # Vercel serves public/ from its CDN at the root paths; send a bare "/" to the page.
    @app.get("/", include_in_schema=False)
    def home():
        return RedirectResponse("/index.html")
else:
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="site")
