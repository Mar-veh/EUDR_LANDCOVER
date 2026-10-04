#!/usr/bin/env python3
"""Quick-start: classify a small area of Ghana with TabPFN-3.5 in a few minutes.

The complete training context of the national map is data/polygon_means.csv:
318 field polygons x 64 Google Satellite Embedding features. This script fits
TabPFN-3.5 on it with the configuration of the national run (about a second),
classifies a 64-band embedding GeoTIFF, and writes to --out-dir:

  <name>_classcode.tif     predicted field class (23 classes, 0 = NoData)
  <name>_landcover.tif     final 17-class legend, color table embedded
  <name>_confidence.tif    top-class probability (-1 = NoData)
  <name>_preview.png       land cover and confidence side by side

Examples
--------
Classify the published demo area (download link in README.md):

    python quickstart.py --aoi demo/demo_aoi_dunkwa.tif

Cut a demo area from the full embedding tiles in Images/ and classify it:

    python quickstart.py --images-dir Images --center -1.78 5.96 --size 1024 \\
        --save-aoi demo/demo_aoi_dunkwa.tif

Report per-pixel agreement with the national prediction over the same area:

    python quickstart.py --aoi demo/demo_aoi_dunkwa.tif \\
        --compare-with predictions_tabpfn35_polygon/predicted_classcode.tif

A CUDA GPU classifies a 1024 x 1024 px area (~10 x 10 km) in a few minutes;
on CPU, use a smaller area (e.g. --size 256).
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

import lc_eval
from predict_tile_worker import resolve_band_order

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POLYGON_MEANS = os.path.join(HERE, 'data', 'polygon_means.csv')
RECLASS_XLSX_PATH = os.path.join(HERE, 'new_class.xlsx')
COLOR_CODE_PATH = os.path.join(HERE, 'color_code.txt')

NODATA_CLASS = 0
NODATA_CONF = -1.0
RANDOM_STATE = 42               # same seed as the national model
LOW_CONFIDENCE_THRESHOLD = 0.6  # same threshold as the national confidence map
CONF_MAP_VMIN = 0.4             # same single-hue scale as the national confidence map (0.4 -> 1.0)
LEGEND_MIN_SHARE = 0.01         # preview legend lists classes covering at least 1% of the area
CPU_PIXEL_WARNING = 300_000


def parse_args():
    p = argparse.ArgumentParser(
        description='Classify a small area with TabPFN-3.5 (Ghana Land Cover 2025 quick-start).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples' + __doc__.split('Examples', 1)[1],
    )
    src = p.add_argument_group('input (choose one)')
    src.add_argument('--aoi', help='64-band embedding GeoTIFF (bands A00..A63) to classify.')
    src.add_argument('--images-dir', help='Folder of full embedding tiles to cut an area from.')
    p.add_argument('--center', nargs=2, type=float, metavar=('LON', 'LAT'), default=(-1.78, 5.96),
                   help='Center of the area cut from --images-dir (default: Dunkwa-on-Offin).')
    p.add_argument('--size', type=int, default=1024,
                   help='Width and height in pixels of the area cut from --images-dir (default: 1024).')
    p.add_argument('--save-aoi', help='Write the area cut from --images-dir to this GeoTIFF (lossless).')
    p.add_argument('--polygon-means', default=DEFAULT_POLYGON_MEANS,
                   help='Training context exported by the training notebook.')
    p.add_argument('--out-dir', default=os.path.join(HERE, 'quickstart_output'))
    p.add_argument('--name', help='Output file prefix (default: derived from the input).')
    p.add_argument('--batch-size', type=int, default=50_000,
                   help='Pixels per predict_proba call (default: 50,000).')
    p.add_argument('--compare-with',
                   help='National predicted_classcode.tif; reports per-pixel agreement over this area.')
    args = p.parse_args()
    if bool(args.aoi) == bool(args.images_dir):
        p.error('specify exactly one of --aoi or --images-dir')
    if args.save_aoi and not args.images_dir:
        p.error('--save-aoi requires --images-dir')
    return args


def load_token():
    from dotenv import load_dotenv

    load_dotenv(os.path.join(HERE, '.env'))
    if not os.getenv('TABPFN_TOKEN'):
        raise SystemExit('TABPFN_TOKEN not found. Copy .env.example to .env and set '
                         'TABPFN_TOKEN=<your key> (keys: https://ux.priorlabs.ai/account).')


def fit_tabpfn(polygon_means_path, device):
    """Fit TabPFN-3.5 on the 318 polygon-mean embeddings (national-model configuration)."""
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    # round_trip parsing reproduces the exported float64 values exactly.
    context = pd.read_csv(polygon_means_path, float_precision='round_trip')
    X = context[lc_eval.EMBEDDING_COLS]
    y = context['code'].astype(int)

    clf = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3_5,
        device=device,
        random_state=RANDOM_STATE,
        show_progress_bar=False,
    )
    t0 = time.time()
    clf.fit(X, y)
    print(f'TabPFN-3.5 fit on {len(context)} polygons x {X.shape[1]} features '
          f'in {time.time() - t0:.1f} s ({y.nunique()} classes).')
    return clf


def read_aoi(path):
    with rasterio.open(path) as src:
        return src.read(resolve_band_order(src)), src.transform, src.crs


def cut_aoi_from_tiles(images_dir, lon, lat, size):
    """Read a size x size px window centered on (lon, lat) from the embedding tiles,
    stitching across tile boundaries. All tiles share one pixel grid."""
    tile_paths = sorted(glob.glob(os.path.join(images_dir, '*.tif')))
    if not tile_paths:
        raise SystemExit(f'No .tif tiles found in {images_dir}')
    with rasterio.open(tile_paths[0]) as ref:
        T0, crs, dtype = ref.transform, ref.crs, ref.dtypes[0]

    col0 = int(np.floor((lon - T0.c) / T0.a)) - size // 2
    row0 = int(np.floor((lat - T0.f) / T0.e)) - size // 2
    aoi = np.full((64, size, size), np.nan, dtype=dtype)

    n_tiles = 0
    for path in tile_paths:
        with rasterio.open(path) as src:
            T = src.transform
            if not (np.isclose(T.a, T0.a) and np.isclose(T.e, T0.e)):
                raise SystemExit(f'{path} is not on the reference pixel grid.')
            dc = int(round((T.c - T0.c) / T0.a))
            dr = int(round((T.f - T0.f) / T0.e))
            c_lo, c_hi = max(col0, dc), min(col0 + size, dc + src.width)
            r_lo, r_hi = max(row0, dr), min(row0 + size, dr + src.height)
            if c_lo >= c_hi or r_lo >= r_hi:
                continue
            block = src.read(resolve_band_order(src),
                             window=Window(c_lo - dc, r_lo - dr, c_hi - c_lo, r_hi - r_lo))
            aoi[:, r_lo - row0:r_hi - row0, c_lo - col0:c_hi - col0] = block
            n_tiles += 1
    if n_tiles == 0:
        raise SystemExit(f'({lon}, {lat}) lies outside the embedding tiles in {images_dir}.')
    print(f'Cut {size} x {size} px around ({lon}, {lat}) from {n_tiles} tile(s).')
    return aoi, T0 * Affine.translation(col0, row0), crs


def save_aoi(path, arr, transform, crs):
    """Write the embedding stack losslessly (source dtype, floating-point predictor)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    codec = lc_eval.lossless_float_codec()
    profile = dict(driver='GTiff', width=arr.shape[2], height=arr.shape[1], count=arr.shape[0],
                   dtype=arr.dtype.name, crs=crs, transform=transform, nodata=np.nan,
                   **lc_eval.lossless_float_profile(codec, block_size=256))
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr)
        for i, name in enumerate(lc_eval.EMBEDDING_COLS, start=1):
            dst.set_band_description(i, name)
    print(f'Saved area of interest: {path} ({os.path.getsize(path) / 1e6:.0f} MB, {codec.upper()})')


def classify(clf, arr, batch_size):
    """Top class and its probability for every valid (non-NaN) pixel."""
    bands, h, w = arr.shape
    flat = arr.reshape(bands, h * w).T
    valid = ~np.isnan(flat).any(axis=1)
    X_valid = flat[valid]
    classes = np.asarray(clf.classes_)

    pred = np.empty(len(X_valid), dtype=np.uint8)
    conf = np.empty(len(X_valid), dtype=np.float32)
    t0 = time.time()
    for b0 in range(0, len(X_valid), batch_size):
        b1 = min(b0 + batch_size, len(X_valid))
        proba = clf.predict_proba(pd.DataFrame(X_valid[b0:b1], columns=lc_eval.EMBEDDING_COLS))
        top = np.argmax(proba, axis=1)
        pred[b0:b1] = classes[top].astype(np.uint8)
        conf[b0:b1] = proba[np.arange(len(top)), top]
        rate = b1 / max(time.time() - t0, 1e-9)
        print(f'  classified {b1:,}/{len(X_valid):,} pixels ({rate:,.0f} px/s)')

    class_out = np.full(h * w, NODATA_CLASS, dtype=np.uint8)
    conf_out = np.full(h * w, NODATA_CONF, dtype=np.float32)
    class_out[valid] = pred
    conf_out[valid] = conf
    return class_out.reshape(h, w), conf_out.reshape(h, w), time.time() - t0


def write_band(path, data, transform, crs, nodata, colormap=None):
    is_float = np.issubdtype(data.dtype, np.floating)
    storage = (lc_eval.lossless_float_profile(lc_eval.lossless_float_codec(), block_size=256)
               if is_float else {'compress': 'deflate', 'tiled': True,
                                 'blockxsize': 256, 'blockysize': 256})
    with rasterio.open(path, 'w', driver='GTiff', width=data.shape[1], height=data.shape[0],
                       count=1, dtype=data.dtype.name, crs=crs, transform=transform,
                       nodata=nodata, **storage) as dst:
        dst.write(data, 1)
        if colormap:
            dst.write_colormap(1, colormap)


def degree_formatter(axis):
    from matplotlib.ticker import FuncFormatter

    if axis == 'x':
        return FuncFormatter(lambda v, _: f'{abs(v):.2f}°{"W" if v < 0 else "E"}')
    return FuncFormatter(lambda v, _: f'{abs(v):.2f}°{"S" if v < 0 else "N"}')


def save_preview(path, landcover, conf, transform, legend_names, legend_rgba, title):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Patch

    h, w = landcover.shape
    left, top = transform.c, transform.f
    right, bottom = left + w * transform.a, top + h * transform.e
    extent = [left, right, bottom, top]
    aspect = 1 / np.cos(np.radians((top + bottom) / 2))

    max_code = max(legend_rgba)
    colors = [(0, 0, 0, 0) if c == NODATA_CLASS else
              tuple(v / 255 for v in legend_rgba.get(c, (0, 0, 0, 255))) for c in range(max_code + 1)]
    conf_cmap = LinearSegmentedColormap.from_list('confidence', lc_eval.BLUE_RAMP)
    conf_cmap.set_bad((0, 0, 0, 0))

    fig, (ax_lc, ax_cf) = plt.subplots(1, 2, figsize=(13.5, 7.6))
    ax_lc.imshow(landcover, cmap=ListedColormap(colors),
                 norm=BoundaryNorm(np.arange(-0.5, max_code + 1.5), max_code + 1),
                 extent=extent, interpolation='nearest')
    im = ax_cf.imshow(np.ma.masked_less(conf, 0), cmap=conf_cmap, vmin=CONF_MAP_VMIN, vmax=1.0,
                      extent=extent, interpolation='nearest')
    for ax, label in ((ax_lc, 'Land cover (17-class legend)'), (ax_cf, 'TabPFN-3.5 confidence')):
        ax.set_aspect(aspect)
        ax.set_title(label, loc='left', fontsize=11, fontweight='bold')
        ax.xaxis.set_major_formatter(degree_formatter('x'))
        ax.yaxis.set_major_formatter(degree_formatter('y'))
        ax.tick_params(labelsize=8)
        ax.locator_params(nbins=4)

    # Legend: classes covering at least LEGEND_MIN_SHARE of the area, largest first;
    # the remaining classes are summarised in one line.
    valid = landcover[landcover != NODATA_CLASS]
    codes, counts = np.unique(valid, return_counts=True)
    shares = counts / max(valid.size, 1)
    order = np.argsort(counts)[::-1]
    shown = [i for i in order if shares[i] >= LEGEND_MIN_SHARE]
    handles = [Patch(facecolor=tuple(v / 255 for v in legend_rgba[int(codes[i])]), edgecolor='0.3',
                     linewidth=0.4, label=f'{legend_names.get(int(codes[i]), codes[i])} ({shares[i]:.0%})')
               for i in shown]
    n_rest = len(order) - len(shown)
    if n_rest:
        rest_share = shares[[i for i in order if i not in shown]].sum()
        handles.append(Patch(facecolor='none', edgecolor='none',
                             label=f'{n_rest} further class{"es" if n_rest > 1 else ""}, '
                                   f'{rest_share:.1%} combined'))
    ax_lc.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncols=2,
                 fontsize=8.5, frameon=False)

    # Colorbar in an inset below the confidence panel, so both panels keep the same size.
    cax = ax_cf.inset_axes([0.0, -0.15, 1.0, 0.035])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal', extend='min')
    cbar.set_label('Top-class probability', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax_lc.text(0.0, 1.09, title, transform=ax_lc.transAxes, ha='left', va='bottom',
               fontsize=13, fontweight='bold')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def compare_with_national(national_path, cls, transform):
    """Per-pixel agreement of this run's class codes with the national class mosaic."""
    with rasterio.open(national_path) as src:
        T = src.transform
        if not (np.isclose(T.a, transform.a) and np.isclose(T.e, transform.e)):
            raise SystemExit('The national raster and this area use different pixel sizes.')
        col = int(round((transform.c - T.c) / T.a))
        row = int(round((transform.f - T.f) / T.e))
        national = src.read(1, window=Window(col, row, cls.shape[1], cls.shape[0]),
                            boundless=True, fill_value=NODATA_CLASS)
    both = (national != NODATA_CLASS) & (cls != NODATA_CLASS)
    return float(np.mean(national[both] == cls[both])) if both.any() else float('nan'), int(both.sum())


def main():
    args = parse_args()
    load_token()
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Compute device: {device}')

    t_start = time.time()
    if args.aoi:
        arr, transform, crs = read_aoi(args.aoi)
        name = args.name or os.path.splitext(os.path.basename(args.aoi))[0]
    else:
        lon, lat = args.center
        arr, transform, crs = cut_aoi_from_tiles(args.images_dir, lon, lat, args.size)
        if args.save_aoi:
            save_aoi(args.save_aoi, arr, transform, crs)
        name = args.name or (os.path.splitext(os.path.basename(args.save_aoi))[0] if args.save_aoi else
                             f'aoi_{abs(lat):.3f}{"N" if lat >= 0 else "S"}_{abs(lon):.3f}{"E" if lon >= 0 else "W"}')

    h, w = arr.shape[1:]
    if device == 'cpu' and h * w > CPU_PIXEL_WARNING:
        print(f'[!] {h * w:,} pixels on CPU will take a long time; consider --size 256.')

    clf = fit_tabpfn(args.polygon_means, device)
    cls, conf, predict_seconds = classify(clf, arr, args.batch_size)
    n_valid = int((cls != NODATA_CLASS).sum())

    reclass_lut, _ = lc_eval.load_reclass_lut(RECLASS_XLSX_PATH)
    legend_names, legend_rgba = lc_eval.load_color_table(COLOR_CODE_PATH)
    landcover = reclass_lut[cls]

    os.makedirs(args.out_dir, exist_ok=True)
    out = {k: os.path.join(args.out_dir, f'{name}_{k}') for k in
           ('classcode.tif', 'landcover.tif', 'confidence.tif', 'preview.png')}
    write_band(out['classcode.tif'], cls, transform, crs, NODATA_CLASS)
    write_band(out['landcover.tif'], landcover, transform, crs, NODATA_CLASS, colormap=legend_rgba)
    write_band(out['confidence.tif'], conf, transform, crs, NODATA_CONF)

    center_lon = transform.c + w / 2 * transform.a
    center_lat = transform.f + h / 2 * transform.e
    km_x = w * abs(transform.a) * 111.32 * np.cos(np.radians(center_lat))
    km_y = h * abs(transform.e) * 110.57
    title = (f'TabPFN-3.5 quick-start · {km_x:.1f} × {km_y:.1f} km around '
             f'{abs(center_lat):.3f}°{"N" if center_lat >= 0 else "S"}, '
             f'{abs(center_lon):.3f}°{"E" if center_lon >= 0 else "W"}')
    save_preview(out['preview.png'], landcover, conf, transform, legend_names, legend_rgba, title)

    valid_conf = conf[cls != NODATA_CLASS]
    print('\n' + '=' * 64)
    print(f'  Pixels classified:      {n_valid:,} of {h * w:,}')
    print(f'  Prediction time:        {predict_seconds:.0f} s ({n_valid / max(predict_seconds, 1e-9):,.0f} px/s)')
    if n_valid:
        print(f'  Mean confidence:        {valid_conf.mean():.3f}')
        print(f'  Pixels below {LOW_CONFIDENCE_THRESHOLD:.1f}:       {np.mean(valid_conf < LOW_CONFIDENCE_THRESHOLD):.1%}')
        codes, counts = np.unique(landcover[landcover != NODATA_CLASS], return_counts=True)
        print('  Land cover (17-class legend):')
        for i in np.argsort(counts)[::-1]:
            print(f'    {legend_names.get(int(codes[i]), codes[i]):<32} {counts[i] / n_valid:6.1%}')
    if args.compare_with:
        agreement, n_compared = compare_with_national(args.compare_with, cls, transform)
        print(f'  Agreement with national map: {agreement:.2%} of {n_compared:,} pixels')
    print(f'  Total run time:         {time.time() - t_start:.0f} s')
    print('=' * 64)
    for path in out.values():
        print(f'  wrote {path}')


if __name__ == '__main__':
    main()
