"""Cartographic helpers for the Ghana Land Cover 2025 maps.

Basemap
    `load_basemap` reads data/natural_earth_west_africa.geojson (Natural Earth
    1:10m Admin-0 countries, public domain); `draw_country` and
    `countries_in_bbox` draw it with matplotlib, without GIS dependencies
    beyond rasterio.

Statistics
    `read_class_preview` reads a class raster at map resolution with mode
    (majority) resampling; `land_area_by_class` measures the area of every
    class inside a country outline from the full-resolution raster.

Figures
    `cartographic_landcover_map` draws the national land-cover map: basemap,
    graticule, place labels, north arrow, scale bar, locator inset, grouped
    legend with area shares, and credits. The frame helpers `style_map_axes`,
    `add_map_titles`, `add_north_arrow`, `add_scale_bar` and `add_attribution`
    draw the single-panel map layout used for the confidence map.

Used by Predict_TabPFN35_PolygonLevel.ipynb.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088

INK = '#0b0b0b'
INK_SECONDARY = '#3d3c39'
MUTED = '#6f6d68'
SEA = '#dce8f1'
LAND_OTHER = '#ebe8e1'
COUNTRY_BORDER = '#ffffff'
OUTLINE = '#2b2a28'
FRAME = '#8a8882'
LOCATOR_HIGHLIGHT = '#2a78d6'
MAP_PAPER = '#eef0eb'
HALO = 'white'


# ---------------------------------------------------------------------------
# Basemap
# ---------------------------------------------------------------------------
def load_basemap(path):
    """Country features of the bundled Natural Earth subset."""
    with open(path, encoding='utf-8') as f:
        return json.load(f)['features']


def country_feature(features, adm0_a3):
    return next(f for f in features if f['properties']['ADM0_A3'] == adm0_a3)


def outer_rings(feature):
    """Exterior ring of every polygon part, as (n, 2) lon/lat arrays."""
    geom = feature['geometry']
    polygons = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
    return [np.asarray(polygon[0]) for polygon in polygons]


def countries_in_bbox(features, x0, x1, y0, y1):
    """Features with at least one polygon part intersecting the lon/lat box."""
    selected = []
    for f in features:
        for ring in outer_rings(f):
            if (ring[:, 0].max() >= x0 and ring[:, 0].min() <= x1
                    and ring[:, 1].max() >= y0 and ring[:, 1].min() <= y1):
                selected.append(f)
                break
    return selected


def draw_country(ax, feature, **kwargs):
    from matplotlib.patches import Polygon

    for ring in outer_rings(feature):
        ax.add_patch(Polygon(ring, closed=True, **kwargs))


# ---------------------------------------------------------------------------
# Raster statistics
# ---------------------------------------------------------------------------
def read_class_preview(path, target_width=3000, resampling='mode'):
    """Class raster read at about `target_width` columns.

    Mode resampling assigns every map pixel the most common class of the
    raster block it covers; NoData does not vote. Returns (array, bounds,
    downsample factor).
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.Env(GDAL_NUM_THREADS='ALL_CPUS'):
        with rasterio.open(path) as src:
            factor = max(1, int(np.ceil(src.width / target_width)))
            data = src.read(
                1,
                out_shape=(max(1, src.height // factor), max(1, src.width // factor)),
                resampling=getattr(Resampling, resampling),
            )
            return data, src.bounds, factor


def land_area_by_class(path, geometry, nodata=0, block_rows=1024, progress_every=20):
    """Pixel count and area (km^2) of every class inside `geometry`.

    Streams the full-resolution EPSG:4326 raster in row blocks; a pixel counts
    when its center lies inside the GeoJSON `geometry` and its value is not
    `nodata`. Pixel areas use a spherical Earth at each row's latitude.
    Returns a DataFrame indexed by class code (columns: pixel_count, area_km2).
    """
    import time

    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import Window

    t0 = time.time()
    counts = np.zeros(256, dtype=np.int64)
    area = np.zeros(256)
    with rasterio.open(path) as src:
        T = src.transform
        dy_km = EARTH_RADIUS_KM * np.radians(abs(T.e))
        n_blocks = int(np.ceil(src.height / block_rows))
        for bi, row0 in enumerate(range(0, src.height, block_rows)):
            rows = min(block_rows, src.height - row0)
            window = Window(0, row0, src.width, rows)
            block = src.read(1, window=window)
            inside = geometry_mask([geometry], out_shape=block.shape,
                                   transform=src.window_transform(window), invert=True)
            valid = inside & (block != nodata)
            rr, cc = np.nonzero(valid)
            if rr.size:
                lat = T.f + (row0 + np.arange(rows) + 0.5) * T.e
                px_km2 = EARTH_RADIUS_KM * np.radians(abs(T.a)) * np.cos(np.radians(lat)) * dy_km
                per_row = np.bincount(rr * 256 + block[rr, cc], minlength=rows * 256).reshape(rows, 256)
                counts += per_row.sum(axis=0)
                area += (per_row * px_km2[:, None]).sum(axis=0)
            if progress_every and ((bi + 1) % progress_every == 0 or (bi + 1) == n_blocks):
                print(f'  land area: block {bi + 1}/{n_blocks} ({(time.time() - t0) / 60:.1f} min elapsed)')
    present = np.nonzero(counts)[0]
    return pd.DataFrame({'pixel_count': counts[present], 'area_km2': area[present]},
                        index=pd.Index(present, name='code'))


# ---------------------------------------------------------------------------
# Single-panel map frame (confidence map)
# ---------------------------------------------------------------------------
def style_map_axes(ax, bounds):
    """True-scale aspect, 1-degree graticule with N/S/E/W labels, and neatline for an
    EPSG:4326 map; returns the center latitude."""
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    center_lat = (bounds.bottom + bounds.top) / 2
    ax.set_facecolor(MAP_PAPER)
    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    ax.set_aspect(1 / np.cos(np.radians(center_lat)))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FuncFormatter(_lon_label))
    ax.yaxis.set_major_formatter(FuncFormatter(_lat_label))
    ax.tick_params(labelsize=10, length=4, width=0.8, top=True, right=True,
                   labeltop=False, labelright=False, direction='out')
    ax.grid(True, linestyle=(0, (1, 3)), linewidth=0.6, color='0.55', zorder=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color('black')
    return center_lat


def add_map_titles(ax, title, subtitle):
    ax.set_title(title, fontsize=19, fontweight='bold', pad=26)
    ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=11, color='0.25')


def add_north_arrow(ax, x=0.955, y=0.955, size=0.045):
    """Split-fill north arrow with an N label, in axes coordinates."""
    from matplotlib.patches import Polygon

    tr = ax.transAxes
    ax.add_patch(Polygon([(x, y - size), (x - 0.42 * size, y - 1.55 * size), (x, y - 1.25 * size)],
                         closed=True, facecolor='black', edgecolor='black', linewidth=0.8,
                         transform=tr, zorder=8))
    ax.add_patch(Polygon([(x, y - size), (x + 0.42 * size, y - 1.55 * size), (x, y - 1.25 * size)],
                         closed=True, facecolor='white', edgecolor='black', linewidth=0.8,
                         transform=tr, zorder=8))
    ax.text(x, y - 0.75 * size, 'N', transform=tr, ha='center', va='center',
            fontsize=13, fontweight='bold', zorder=8)


def add_scale_bar(ax, bounds, center_lat, total_km=100, segments=4):
    """Segmented scale bar on a white backing, bottom-right of the data bounds."""
    from matplotlib.patches import Rectangle

    km_per_deg_lon = 111.320 * np.cos(np.radians(center_lat))
    total_deg = total_km / km_per_deg_lon
    seg_deg = total_deg / segments
    x0 = bounds.right - 0.06 * (bounds.right - bounds.left) - total_deg
    y0 = bounds.bottom + 0.045 * (bounds.top - bounds.bottom)
    bar_h = 0.008 * (bounds.top - bounds.bottom)

    pad_x, pad_y = 0.35 * seg_deg, 3.2 * bar_h
    ax.add_patch(Rectangle((x0 - pad_x, y0 - 1.2 * bar_h), total_deg + 2 * pad_x, pad_y + 2.2 * bar_h,
                           facecolor='white', edgecolor='0.4', linewidth=0.6, alpha=0.85, zorder=5))
    for i in range(segments):
        ax.add_patch(Rectangle((x0 + i * seg_deg, y0), seg_deg, bar_h,
                               facecolor='black' if i % 2 == 0 else 'white',
                               edgecolor='black', linewidth=0.8, zorder=6))
    for km, frac in [(0, 0.0), (total_km // 2, 0.5), (total_km, 1.0)]:
        label = f'{km} km' if km == total_km else f'{km}'
        ax.text(x0 + frac * total_deg, y0 + 1.7 * bar_h, label,
                ha='center', va='bottom', fontsize=8.5, zorder=6)


def add_attribution(ax, text, y=-0.30):
    ax.text(0.5, y, text, transform=ax.transAxes, ha='center', va='top',
            fontsize=8, color='0.35', linespacing=1.6)


def _lon_label(v, _=None):
    return f'{abs(v):g}°W' if v < 0 else ('0°' if v == 0 else f'{v:g}°E')


def _lat_label(v, _=None):
    return f'{abs(v):g}°S' if v < 0 else ('0°' if v == 0 else f'{v:g}°N')


# ---------------------------------------------------------------------------
# National land-cover map
# ---------------------------------------------------------------------------
def _share_label(share):
    return '<0.1%' if 0 < share < 0.001 else f'{share * 100:.1f}%'


def cartographic_landcover_map(preview, bounds, basemap, country, legend_groups, class_names, class_rgba,
                               area_share, title, subtitle, credits, places=(), labels=(),
                               extent=None, locator_extent=(-18.5, 16.5, 3.0, 25.0), scale_km=100,
                               figsize=(13.0, 11.6)):
    """National land-cover map with basemap, labels, locator inset and grouped legend.

    preview        class codes at map resolution (0 = NoData), covering `bounds`
    basemap        country features (load_basemap); `country` is the mapped one
    legend_groups  [(group title, [class codes...]), ...] in display order
    area_share     {class code: share of land area} shown beside each entry
    places         [(name, lon, lat, 'capital' | 'city', 'left' | 'right'), ...]
    labels         [(text, lon, lat, style), ...] with style 'neighbour',
                   'neighbour_vertical', 'sea' or 'lake'
    extent         (lon0, lon1, lat0, lat1) of the main map; defaults to `bounds`
                   with a 0.15-degree margin
    """
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Polygon, Rectangle
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds

    left, bottom, right, top = bounds
    x0, x1, y0, y1 = extent or (left - 0.15, right + 0.15, bottom - 0.15, top + 0.15)
    halo = [pe.withStroke(linewidth=2.6, foreground=HALO)]

    # Class pixels outside the country outline (sea, neighbouring land) are not drawn.
    transform = from_bounds(left, bottom, right, top, preview.shape[1], preview.shape[0])
    inside = geometry_mask([country['geometry']], out_shape=preview.shape, transform=transform, invert=True)
    shown = np.where(inside, preview, 0)

    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_axes([0.035, 0.045, 0.60, 0.86])
    ax.set_facecolor(SEA)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    center_lat = (y0 + y1) / 2
    ax.set_aspect(1 / np.cos(np.radians(center_lat)))

    country_code = country['properties']['ADM0_A3']
    for f in countries_in_bbox(basemap, x0, x1, y0, y1):
        if f['properties']['ADM0_A3'] != country_code:
            draw_country(ax, f, facecolor=LAND_OTHER, edgecolor=COUNTRY_BORDER, linewidth=1.2, zorder=1)

    max_code = max(class_rgba)
    cmap = ListedColormap([(0, 0, 0, 0) if c == 0 else tuple(v / 255 for v in class_rgba.get(c, (0, 0, 0, 255)))
                           for c in range(max_code + 1)])
    ax.imshow(shown, cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, max_code + 1.5), max_code + 1),
              extent=[left, right, bottom, top], interpolation='nearest', zorder=2)
    draw_country(ax, country, facecolor='none', edgecolor=OUTLINE, linewidth=1.1, zorder=3)

    # Graticule: solid white hairlines, degree labels on all four sides.
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FuncFormatter(_lon_label))
    ax.yaxis.set_major_formatter(FuncFormatter(_lat_label))
    ax.grid(True, color='white', linewidth=0.5, alpha=0.9, zorder=1.5)
    ax.tick_params(labelsize=8.5, colors=MUTED, length=3, top=True, right=True, labeltop=True, labelright=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color(FRAME)

    label_styles = {
        'neighbour': dict(fontsize=9, color='#8f8c85', rotation=0),
        'neighbour_vertical': dict(fontsize=9, color='#8f8c85', rotation=90),
        'sea': dict(fontsize=11, color='#5c7f9e', style='italic'),
        'lake': dict(fontsize=9, color='#08519c', style='italic', path_effects=halo),
    }
    for text, lon, lat, style in labels:
        ax.text(lon, lat, text, ha='center' if style != 'lake' else 'left', va='center', zorder=6,
                **label_styles[style])

    for name, lon, lat, kind, side in places:
        is_capital = kind == 'capital'
        ax.plot(lon, lat, marker='*' if is_capital else 'o', markersize=12 if is_capital else 5,
                color=INK, markeredgecolor='white', markeredgewidth=1.0, zorder=7)
        dx = 0.07 if side == 'right' else -0.07
        ax.text(lon + dx, lat + 0.02, name, fontsize=10.5 if is_capital else 9,
                fontweight='bold' if is_capital else 'normal', color=INK,
                ha='left' if side == 'right' else 'right', va='center', zorder=7, path_effects=halo)

    # North arrow (top-right) and scale bar (bottom-right, over the sea).
    add_north_arrow(ax, x=0.94, y=0.955, size=0.04)
    km_per_deg = 111.32 * np.cos(np.radians(y0 + 0.2))
    bar_deg = scale_km / km_per_deg
    sx, sy, bh = x1 - 0.25 - bar_deg, y0 + 0.18, 0.045
    for i in range(4):
        ax.add_patch(Rectangle((sx + i * bar_deg / 4, sy), bar_deg / 4, bh,
                               facecolor=INK if i % 2 == 0 else 'white', edgecolor=INK, linewidth=0.7, zorder=8))
    for km, frac in [(0, 0.0), (scale_km // 2, 0.5), (scale_km, 1.0)]:
        ax.text(sx + frac * bar_deg, sy + bh + 0.04, f'{km} km' if km == scale_km else f'{km}',
                ha='center', va='bottom', fontsize=8, color=INK_SECONDARY, zorder=8)

    # Right column: title, locator inset, grouped legend, credits.
    fig.text(0.665, 0.955, title, fontsize=21, fontweight='bold', color=INK, va='top')
    fig.text(0.665, 0.918, subtitle, fontsize=11, color=INK_SECONDARY, va='top', linespacing=1.4)

    lx0, lx1, ly0, ly1 = locator_extent
    inset = fig.add_axes([0.665, 0.70, 0.30, 0.17])
    inset.set_facecolor(SEA)
    inset.set_xlim(lx0, lx1)
    inset.set_ylim(ly0, ly1)
    inset.set_aspect(1 / np.cos(np.radians((ly0 + ly1) / 2)))
    for f in countries_in_bbox(basemap, lx0, lx1, ly0, ly1):
        highlight = f['properties']['ADM0_A3'] == country_code
        draw_country(inset, f, facecolor=LOCATOR_HIGHLIGHT if highlight else LAND_OTHER,
                     edgecolor='white', linewidth=0.4)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color(FRAME)
    inset.text(0.02, 0.04, 'West Africa', transform=inset.transAxes, fontsize=8, color=MUTED)

    legend = fig.add_axes([0.665, 0.10, 0.31, 0.57])
    legend.set_axis_off()
    legend.set_xlim(0, 1)
    legend.set_ylim(0, 1)
    row = 0.040
    y = 0.985
    legend.text(1.0, 1.02, 'share of land area', fontsize=8, color=MUTED, ha='right', va='bottom', style='italic')
    for group_title, codes in legend_groups:
        legend.text(0.0, y, group_title.upper(), fontsize=8.5, fontweight='bold', color=MUTED, va='top')
        y -= row * 0.95
        for code in codes:
            legend.add_patch(Rectangle((0.0, y - 0.026), 0.075, 0.03,
                                       facecolor=tuple(v / 255 for v in class_rgba[code]),
                                       edgecolor='#5d5b57', linewidth=0.4))
            legend.text(0.10, y - 0.011, class_names[code], fontsize=10, color=INK, va='center')
            legend.text(1.0, y - 0.011, _share_label(area_share.get(code, 0.0)), fontsize=10,
                        color=INK_SECONDARY, va='center', ha='right')
            y -= row
        y -= row * 0.35

    fig.text(0.665, 0.03, credits, fontsize=7.8, color=MUTED, va='bottom', linespacing=1.6)
    return fig
