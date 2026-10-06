"""Pixel inspector: what the 10 m pixel under a point holds."""
from __future__ import annotations

from functools import lru_cache

import ee

from .config import calibration, classes
from .layers import grid, native, run, tree_crop_code


@lru_cache
def _lookups() -> tuple[dict, dict]:
    c = classes()
    return ({f["code"]: f for f in c["field_classes"]}, {e["code"]: e for e in c["legend"]})


def calibration_for(conf: float) -> dict | None:
    """Observed held-out accuracy of the calibration bin containing conf."""
    for b in calibration()["bins"]:
        if b["low"] <= conf < b["high"] or (conf == 1.0 and b["high"] == 1.0):
            if b["accuracy"] is None:
                return None
            return {"low": b["low"], "high": b["high"], "accuracy": b["accuracy"]}
    return None


def eudr_status(is_tree_crop: bool, on_forest: bool, conf: float, threshold: float) -> str | None:
    if not is_tree_crop:
        return None
    if not on_forest:
        return "outside"
    return "high" if conf >= threshold else "low"


def point_info(lon: float, lat: float, threshold: float) -> dict:
    g = grid()
    values = run(lambda: native().reduceRegion(
        ee.Reducer.first(), ee.Geometry.Point([lon, lat]), crs=g["crs"], crsTransform=g["transform"]
    ).getInfo())
    if values.get("class23") is None:
        return {"covered": False, "lon": lon, "lat": lat}
    field, legend = _lookups()
    c23 = int(values["class23"])
    lc = legend.get(int(values["lc17"])) if values.get("lc17") is not None else None
    conf = float(values["confidence"]) if values.get("confidence") is not None else None
    on_forest = bool(values.get("f2020"))
    is_tree = lc is not None and lc["code"] == tree_crop_code()
    return {
        "covered": True,
        "lon": lon,
        "lat": lat,
        "field_class": {"code": c23, "name": field.get(c23, {}).get("label", str(c23))},
        "legend_class": {"code": lc["code"], "name": lc["name"], "color": lc["color"]} if lc else None,
        "confidence": conf,
        "calibration": calibration_for(conf) if conf is not None else None,
        "forest2020": on_forest,
        "eudr": eudr_status(is_tree, on_forest, conf or 0.0, threshold),
    }
