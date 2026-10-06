"""Earth Engine images behind the map layers and the pixel inspector.

The pixel inspector reads the classification on its own 10 m grid, so every
value it reports comes from the original pixels.

Map tiles are drawn from the same assets. The class asset is pyramided with
MODE, so a zoomed-out tile pixel shows the most common class of the 10 m
pixels it covers; the tree-crop overlays inherit this, so small tree-crop
patches appear as you zoom in.
"""
from __future__ import annotations

import threading
import time
from functools import lru_cache

import ee

from .config import classes, settings
from .ee_client import init

# Overlay colours, validated together for colour-vision deficiency: tree crop
# on 2020 forest below / at or above the confidence threshold, and the 2020
# forest tint.
EUDR_PALETTE = ["fab219", "d03b3b"]
FOREST_COLOR = "00441b"
TILE_TTL_S = 30 * 60

_ee_slots = threading.BoundedSemaphore(8)
_ee_thread = threading.local()


def run(fn, *args, **kwargs):
    """Call Earth Engine with at most eight requests in flight (re-entrant)."""
    init()
    if getattr(_ee_thread, "inside", False):
        return fn(*args, **kwargs)
    with _ee_slots:
        _ee_thread.inside = True
        try:
            return fn(*args, **kwargs)
        finally:
            _ee_thread.inside = False


def tree_crop_code() -> int:
    return classes()["tree_crop_code"]


@lru_cache
def legend_palette() -> list[str]:
    by_code = {e["code"]: e["color"].lstrip("#") for e in classes()["legend"]}
    if sorted(by_code) != list(range(1, len(by_code) + 1)):
        raise RuntimeError("Legend codes must run 1..N without gaps.")
    return [by_code[c] for c in sorted(by_code)]


@lru_cache
def grid() -> dict:
    """CRS, affine transform and size of the classification's native grid."""
    g = run(ee.data.getAsset, settings().classcode_asset)["bands"][0]["grid"]
    t = g["affineTransform"]
    return {
        "crs": g["crsCode"],
        "transform": [t.get("scaleX", 0), t.get("shearX", 0), t.get("translateX", 0),
                      t.get("shearY", 0), t.get("scaleY", 0), t.get("translateY", 0)],
        "width": g["dimensions"]["width"],
        "height": g["dimensions"]["height"],
    }


# --- Source images -------------------------------------------------------------

def class23() -> ee.Image:
    return ee.Image(settings().classcode_asset).select([0], ["class23"])


def confidence() -> ee.Image:
    return ee.Image(settings().confidence_asset).select([0], ["confidence"])


@lru_cache
def _forest_source() -> ee.Image:
    asset = settings().forest_asset
    if run(ee.data.getAsset, asset)["type"] == "IMAGE_COLLECTION":
        return ee.ImageCollection(asset).mosaic().select("Map")
    return ee.Image(asset).select("Map")


def forest2020() -> ee.Image:
    """1 where JRC GFC2020 maps forest on 31 Dec 2020, 0 elsewhere."""
    return _forest_source().eq(1).unmask(0).rename("f2020")


def legend17(c23: ee.Image) -> ee.Image:
    r = classes()["remap"]
    return c23.remap(r["from"], r["to"]).rename("lc17")


def native() -> ee.Image:
    """class23, lc17, confidence and f2020 (0/1), masked to the map."""
    c23 = class23()
    lc17 = legend17(c23)
    return ee.Image.cat([c23, lc17, confidence(), forest2020().updateMask(lc17.mask())])


# --- Map layers ----------------------------------------------------------------

def landcover_layer(fade: bool) -> ee.Image:
    image = legend17(class23()).visualize(min=1, max=len(legend_palette()), palette=legend_palette())
    if fade:
        image = image.updateMask(confidence().unitScale(0.3, 0.9).clamp(0.12, 1))
    return image


def forest_layer() -> ee.Image:
    # Clipped with the confidence asset's mask: same grid and mapped area as the class asset.
    forest = forest2020().updateMask(confidence().mask()).selfMask()
    return forest.visualize(min=0, max=1, palette=[FOREST_COLOR, FOREST_COLOR])


def eudr_layer(threshold: float) -> ee.Image:
    """Tree crop on 2020 forest: 1 below the confidence threshold, 2 at or above it."""
    tree_on_forest = legend17(class23()).eq(tree_crop_code()).And(forest2020())
    status = ee.Image(1).where(confidence().gte(threshold), 2)
    return status.updateMask(tree_on_forest).visualize(min=1, max=2, palette=EUDR_PALETTE)


_tiles: dict[tuple, tuple[float, str]] = {}
_tiles_lock = threading.Lock()


def tile_url(name: str, **params) -> str:
    """XYZ tile URL for a layer, cached for TILE_TTL_S."""
    builders = {
        "landcover": lambda: landcover_layer(params["fade"]),
        "forest2020": forest_layer,
        "eudr": lambda: eudr_layer(params["threshold"]),
    }
    if name not in builders:
        raise KeyError(name)
    key = (name, tuple(sorted(params.items())))
    now = time.time()
    with _tiles_lock:
        hit = _tiles.get(key)
    if hit and now - hit[0] < TILE_TTL_S:
        return hit[1]
    url = run(lambda: builders[name]().getMapId()["tile_fetcher"].url_format)
    with _tiles_lock:
        _tiles[key] = (now, url)
    return url
