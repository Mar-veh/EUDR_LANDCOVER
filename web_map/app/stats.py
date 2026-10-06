"""Turn grouped Earth Engine sums into area tables.

Every statistic in the app sums pixel area by a code that packs the legend
class, the 2020-forest flag and a confidence bin (see layers.stats_image), so
one reduction gives class areas, 2020-forest overlap and confidence
histograms, and any confidence threshold on the 0.05 grid can be applied
afterwards without going back to Earth Engine.
"""
from __future__ import annotations

from .layers import CONF_BINS, tree_crop_code


def empty() -> dict:
    return {
        "mapped_ha": 0.0,
        "conf_area_ha": 0.0,                   # sum of confidence x area
        "class_ha": {},                        # legend code -> ha
        "class_conf_area_ha": {},              # legend code -> confidence x ha
        "class_conf_bins": {},                 # legend code -> ha per confidence bin
        "forest2020_ha": 0.0,
        "forest2020_class_ha": {},             # 2025 legend class of 2020-forest land -> ha
        "tc_forest_bins": [0.0] * CONF_BINS,   # tree crop on 2020 forest, ha per confidence bin
        "tc_outside_bins": [0.0] * CONF_BINS,  # tree crop outside 2020 forest
    }


def _add(table: dict, key: int, value: float) -> None:
    table[key] = table.get(key, 0.0) + value


def add_groups(acc: dict, groups: list[dict]) -> dict:
    """Add one grouped reduction ({code, sum: [area_ha, conf_area_ha]} rows) to acc."""
    tree = tree_crop_code()
    for group in groups:
        code = int(group["code"])
        area, conf_area = (float(v) for v in group["sum"])
        lc, forest, conf_bin = code // 1000, (code // 100) % 10, code % 100
        acc["mapped_ha"] += area
        acc["conf_area_ha"] += conf_area
        _add(acc["class_ha"], lc, area)
        _add(acc["class_conf_area_ha"], lc, conf_area)
        acc["class_conf_bins"].setdefault(lc, [0.0] * CONF_BINS)[conf_bin] += area
        if forest:
            acc["forest2020_ha"] += area
            _add(acc["forest2020_class_ha"], lc, area)
        if lc == tree:
            (acc["tc_forest_bins"] if forest else acc["tc_outside_bins"])[conf_bin] += area
    return acc


def merge(acc: dict, other: dict) -> dict:
    """Add another accumulator (e.g. a district) into acc."""
    for key in ("mapped_ha", "conf_area_ha", "forest2020_ha"):
        acc[key] += other[key]
    for key in ("class_ha", "class_conf_area_ha", "forest2020_class_ha"):
        for k, v in other[key].items():
            _add(acc[key], int(k), v)
    for k, bins in other["class_conf_bins"].items():
        target = acc["class_conf_bins"].setdefault(int(k), [0.0] * CONF_BINS)
        for i, v in enumerate(bins):
            target[i] += v
    for key in ("tc_forest_bins", "tc_outside_bins"):
        acc[key] = [a + b for a, b in zip(acc[key], other[key])]
    return acc


def threshold_bin(threshold: float) -> int:
    """Index of the first confidence bin at or above the threshold."""
    return round(threshold * CONF_BINS)


def tree_crop_split(acc: dict, threshold: float) -> dict:
    k = threshold_bin(threshold)
    high, low = sum(acc["tc_forest_bins"][k:]), sum(acc["tc_forest_bins"][:k])
    outside = sum(acc["tc_outside_bins"])
    return {
        "ha": high + low + outside,
        "on_forest_ha": high + low,
        "on_forest_high_ha": high,
        "on_forest_low_ha": low,
        "outside_forest_ha": outside,
    }


def rounded(acc: dict, digits: int = 2) -> dict:
    """JSON-ready copy: string keys, areas rounded."""
    def r(v):
        return round(v, digits)

    out = {}
    for key, value in acc.items():
        if isinstance(value, dict):
            out[key] = {str(k): ([r(x) for x in v] if isinstance(v, list) else r(v))
                        for k, v in sorted(value.items())}
        elif isinstance(value, list):
            out[key] = [r(x) for x in value]
        else:
            out[key] = r(value)
    return out
