"""Area statistics for every district, every region and all of Ghana.

Sums the 10 m pixel areas by legend class, 2020-forest flag and confidence
bin inside each geoBoundaries district (ADM2). Ghana has about 2.4 billion
mapped pixels, too many for an interactive request, so this runs as an Earth
Engine batch export to a table asset (STATS_ASSET, default:
<folder of PREDICTED_CLASSCODE>/web_map_district_stats). The fetch step then
adds districts up into regions and the country, and writes:

    public/data/stats.json          national, regional and district statistics
    public/data/districts.geojson   simplified district boundaries
    public/data/regions.geojson     simplified region boundaries

    python scripts/district_stats.py export    start the Earth Engine export
    python scripts/district_stats.py fetch     wait for it, then write the files
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from ee_tasks import asset_exists, wait_for

import ee  # noqa: E402

from app import layers, stats  # noqa: E402
from app.config import DATA_DIR, settings  # noqa: E402
from app.ee_client import init  # noqa: E402

DESCRIPTION = "web_map_district_stats"
ADM1 = "WM/geoLab/geoBoundaries/600/ADM1"
ADM2 = "WM/geoLab/geoBoundaries/600/ADM2"
COUNTRY = "GHA"
SIMPLIFY_M = 120
COORD_DIGITS = 4  # about 11 m


def boundaries(collection: str) -> ee.FeatureCollection:
    return ee.FeatureCollection(collection).filter(ee.Filter.eq("shapeGroup", COUNTRY))


def export() -> None:
    asset = settings().stats_asset
    if asset_exists(asset):
        sys.exit(f"{asset} already exists. Delete it in the Earth Engine Code Editor to recompute it.")
    regions = boundaries(ADM1)
    g = layers.grid()
    reduced = layers.stats_image().reduceRegions(
        collection=boundaries(ADM2), reducer=layers.stats_reducer(),
        crs=g["crs"], crsTransform=g["transform"], tileScale=4,
    )

    def finish(feature):
        # Table exports refuse null geometries, so each row keeps its district centroid.
        centroid = feature.geometry().centroid(100)
        region = regions.filterBounds(centroid).first()
        return ee.Feature(centroid, {
            "district": feature.get("shapeName"),
            "district_id": feature.get("shapeID"),
            "region": ee.Algorithms.If(region, ee.Feature(region).get("shapeName"), ""),
            "groups": ee.String.encodeJSON(feature.get("groups")),
        })

    task = ee.batch.Export.table.toAsset(collection=reduced.map(finish), description=DESCRIPTION, assetId=asset)
    task.start()
    print(f"Started export {task.id} -> {asset}")


def get_all(collection: ee.FeatureCollection, page: int = 50) -> list[dict]:
    size = collection.size().getInfo()
    items = []
    for offset in range(0, size, page):
        items += collection.toList(page, offset).getInfo()
    return items


def round_coords(value):
    if isinstance(value, float):
        return round(value, COORD_DIGITS)
    return [round_coords(v) for v in value]


def bbox(geometry: dict) -> list[float]:
    def points(value):
        if isinstance(value[0], (int, float)):
            yield value
        else:
            for v in value:
                yield from points(v)

    xs, ys = zip(*((p[0], p[1]) for p in points(geometry["coordinates"])))
    return [round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)]


def boundary_features(collection: str) -> dict[str, dict]:
    simplified = boundaries(collection).map(
        lambda f: f.simplify(SIMPLIFY_M).select(["shapeName", "shapeID"]))
    out = {}
    for f in get_all(simplified):
        geometry = f["geometry"]
        if geometry["type"] == "GeometryCollection":  # simplify can leave stray lines; keep the areas
            polys = [g for g in geometry["geometries"] if g["type"] in ("Polygon", "MultiPolygon")]
            geometry = {"type": "MultiPolygon", "coordinates": [
                c for g in polys for c in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]])]}
        geometry["coordinates"] = round_coords(geometry["coordinates"])
        out[f["properties"]["shapeID"]] = {"name": f["properties"]["shapeName"], "geometry": geometry}
    return out


def fetch() -> None:
    s = settings()
    wait_for(DESCRIPTION, s.stats_asset)
    rows = [f["properties"] for f in get_all(ee.FeatureCollection(s.stats_asset))]
    print(f"Read {len(rows)} districts")
    district_shapes = boundary_features(ADM2)
    region_shapes = boundary_features(ADM1)

    national = stats.empty()
    regions: dict[str, dict] = {}
    districts = []
    for row in sorted(rows, key=lambda r: (r["region"], r["district"])):
        groups = json.loads(row.get("groups") or "[]") or []
        acc = stats.add_groups(stats.empty(), groups)
        stats.merge(national, acc)
        stats.merge(regions.setdefault(row["region"], stats.empty()), acc)
        shape = district_shapes.get(row["district_id"])
        compact = stats.rounded(acc)
        for key in ("class_conf_area_ha", "class_conf_bins"):
            compact.pop(key)
        districts.append({"id": row["district_id"], "name": row["district"], "region": row["region"],
                          "bbox": bbox(shape["geometry"]) if shape else None, **compact})

    region_ids = {v["name"]: k for k, v in region_shapes.items()}
    out = {
        "generated": dt.date.today().isoformat(),
        "conf_bins": layers.CONF_BINS,
        "sources": {
            "classification": s.classcode_asset,
            "confidence": s.confidence_asset,
            "forest_2020": s.forest_asset,
            "boundaries": "geoBoundaries v6.0.0 (Earth Engine: WM/geoLab/geoBoundaries/600)",
        },
        "national": stats.rounded(national),
        "regions": [{"id": region_ids.get(name), "name": name,
                     "bbox": bbox(region_shapes[region_ids[name]]["geometry"]) if name in region_ids else None,
                     **stats.rounded(acc)} for name, acc in sorted(regions.items())],
        "districts": districts,
    }
    (DATA_DIR / "stats.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")

    def geojson(shapes: dict, extra: dict) -> dict:
        return {"type": "FeatureCollection", "features": [
            {"type": "Feature", "id": sid, "properties": {"id": sid, "name": v["name"], **extra.get(sid, {})},
             "geometry": v["geometry"]} for sid, v in shapes.items()]}

    parent = {d["id"]: {"region": d["region"]} for d in districts}
    (DATA_DIR / "districts.geojson").write_text(json.dumps(geojson(district_shapes, parent), separators=(",", ":")),
                                                encoding="utf-8")
    (DATA_DIR / "regions.geojson").write_text(json.dumps(geojson(region_shapes, {}), separators=(",", ":")),
                                              encoding="utf-8")
    tree = stats.tree_crop_split(national, s.conf_threshold)
    print(f"Wrote stats.json, districts.geojson and regions.geojson to {DATA_DIR}")
    print(f"Ghana: {national['mapped_ha'] / 100:,.0f} km² mapped, {national['forest2020_ha'] / 100:,.0f} km² "
          f"forest in 2020, {tree['ha'] / 100:,.0f} km² tree crop, of which {tree['on_forest_ha'] / 100:,.0f} km² "
          f"on 2020 forest ({tree['on_forest_high_ha'] / 100:,.0f} km² at confidence >= {s.conf_threshold}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["export", "fetch"])
    args = parser.parse_args()
    init()
    export() if args.step == "export" else fetch()


if __name__ == "__main__":
    main()
