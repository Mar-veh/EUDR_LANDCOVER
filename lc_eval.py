"""Shared evaluation and raster utilities for the Ghana Land Cover 2025 pipeline.

Leave-One-Polygon-Out (LOO) evaluation
    `run_loo` is the single LOO runner used for TabPFN-3.5 and for every baseline
    classifier, so all models are scored under an identical protocol: each field
    polygon is held out once, the model is fit on the remaining polygon-mean
    embeddings, and the held-out polygon is scored on its mean embedding and on
    every one of its raw pixels. `summarize_loo` turns a run into accuracy,
    pixel-level and calibration metrics.

Calibration
    `reliability_table` and `confidence_threshold_summary` measure how closely the
    top-class probability tracks observed accuracy; `plot_reliability_diagrams`
    and `plot_model_comparison` draw the evaluation figures.

Raster utilities
    Legend and reclassification loaders, a lossless float codec probe and
    bit-exact mosaic verification.

Used by Training_TabPFN35_PolygonLevel.ipynb, Predict_TabPFN35_PolygonLevel.ipynb
and quickstart.py.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder

EMBEDDING_COLS = [f'A{i:02d}' for i in range(64)]

# Chart styling: light surface, recessive chrome. TabPFN-3.5 is drawn in the
# accent blue; baselines share one neutral tone and are identified by label.
CHART_COLORS = {
    'surface': '#fcfcfb',
    'ink': '#0b0b0b',
    'ink_secondary': '#52514e',
    'muted': '#898781',
    'grid': '#e1e0d9',
    'axis': '#c3c2b7',
    'accent': '#2a78d6',
    'baseline': '#898781',
}

# Single-hue sequential ramp (light -> dark) used for confidence surfaces.
BLUE_RAMP = [
    '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
    '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b',
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def parse_point_coordinates(geo: pd.Series) -> pd.DataFrame:
    """Longitude/latitude of each pixel from the Earth Engine `.geo` GeoJSON point column."""
    coords = geo.str.extract(
        r'"coordinates"\s*:\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]'
    )
    coords.columns = ['lon', 'lat']
    return coords.astype(float)


def load_reclass_lut(xlsx_path):
    """Old -> new class lookup table from new_class.xlsx.

    Returns a 256-entry uint8 array (unmapped codes -> 0) and the parsed table
    (columns: Old Class Code, Old Class, New Class, New Class Code).
    """
    table = pd.read_excel(xlsx_path)
    table.columns = [str(c).strip() for c in table.columns]
    lut = np.zeros(256, dtype=np.uint8)
    for old_code, new_code in zip(table['Old Class Code'], table['New Class Code']):
        lut[int(old_code)] = int(new_code)
    return lut, table


def load_color_table(path):
    """Parse color_code.txt (`code R G B A name`) into {code: name} and {code: (r, g, b, a)}."""
    names, rgba = {}, {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            code = int(parts[0])
            rgba[code] = tuple(int(v) for v in parts[1:5])
            names[code] = ' '.join(parts[5:])
    return names, rgba


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n (z=1.96 -> ~95% CI)."""
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z ** 2 / n
    center = phat + z ** 2 / (2 * n)
    margin = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return ((center - margin) / denom, (center + margin) / denom)


# ---------------------------------------------------------------------------
# Baseline wrapper
# ---------------------------------------------------------------------------
class LabelEncodedXGBClassifier:
    """XGBoost classifier that accepts arbitrary integer class codes.

    XGBoost requires labels 0..K-1; this wrapper encodes the land-cover codes per
    fit and exposes `classes_` in the original codes, like scikit-learn estimators.
    """

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        from xgboost import XGBClassifier

        self._encoder = LabelEncoder().fit(np.asarray(y))
        self.classes_ = self._encoder.classes_
        self._model = XGBClassifier(**self.params)
        self._model.fit(X, self._encoder.transform(np.asarray(y)))
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# Leave-One-Polygon-Out runner
# ---------------------------------------------------------------------------
@dataclass
class LOOResult:
    """Out-of-sample predictions from one Leave-One-Polygon-Out run.

    records             one row per held-out polygon (mean-embedding prediction
                        and raw-pixel tallies)
    proba               (n_polygons, n_classes) mean-embedding probabilities,
                        columns aligned with `classes`
    pixel_polygon_index (n_pixels,) row in `records` of each held-out raw pixel
    pixel_pred          (n_pixels,) predicted class code of each raw pixel
    pixel_conf          (n_pixels,) top-class probability of each raw pixel
    pixel_correct       (n_pixels,) whether the raw pixel was classified correctly
    """
    model_name: str
    classes: np.ndarray
    records: pd.DataFrame
    proba: np.ndarray
    pixel_polygon_index: np.ndarray
    pixel_pred: np.ndarray
    pixel_conf: np.ndarray
    pixel_correct: np.ndarray
    seconds: float


def run_loo(make_model, X, y, polygon_ids, raw_pixels_by_polygon, classes,
            model_name, code_to_name=None, progress_every=25):
    """Leave-One-Polygon-Out cross-validation with raw-pixel scoring.

    make_model             zero-argument callable returning an estimator with
                           fit / predict_proba / classes_ (a shared instance is
                           valid when fit() fully replaces previous state)
    X, y                   one row per polygon (mean embedding, class code)
    polygon_ids            polygon id of each row of X
    raw_pixels_by_polygon  {polygon_id: DataFrame of that polygon's raw pixels}
    classes                sorted array of all class codes
    """
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y).astype(int))
    polygon_ids = list(polygon_ids)
    classes = np.asarray(classes).astype(int)
    class_pos = {int(c): i for i, c in enumerate(classes)}
    code_to_name = code_to_name or {}
    n = len(X)

    proba = np.zeros((n, len(classes)))
    records = []
    pix_index, pix_pred, pix_conf, pix_correct = [], [], [], []

    print(f'[{model_name}] Leave-One-Polygon-Out across {n} polygons...')
    start = time.time()
    for i, (train_idx, test_idx) in enumerate(LeaveOneOut().split(X)):
        j = int(test_idx[0])
        true_code = int(y.iloc[j])

        model = make_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_classes = np.asarray(model.classes_).astype(int)
        fold_cols = [class_pos[int(c)] for c in fold_classes]

        # Held-out polygon, mean embedding.
        p = np.asarray(model.predict_proba(X.iloc[test_idx]))[0]
        proba[j, fold_cols] = p
        k = int(np.argmax(p))
        pred_code = int(fold_classes[k])
        top3_codes = classes[np.argsort(proba[j])[-3:]]

        # Same fold model, every raw pixel of the held-out polygon.
        raw_X = raw_pixels_by_polygon[polygon_ids[j]]
        raw_p = np.asarray(model.predict_proba(raw_X))
        raw_k = np.argmax(raw_p, axis=1)
        raw_pred = fold_classes[raw_k]
        raw_conf = raw_p[np.arange(len(raw_k)), raw_k]
        raw_ok = raw_pred == true_code

        pix_index.append(np.full(len(raw_k), j, dtype=np.int32))
        pix_pred.append(raw_pred.astype(np.int16))
        pix_conf.append(raw_conf.astype(np.float32))
        pix_correct.append(raw_ok)

        records.append({
            'polygon_id': polygon_ids[j],
            'n_pixels': int(len(raw_X)),
            'true_code': true_code,
            'true_name': code_to_name.get(true_code),
            'pred_code': pred_code,
            'pred_name': code_to_name.get(pred_code),
            'confidence': float(p[k]),
            'correct': pred_code == true_code,
            'top3_correct': bool(true_code in top3_codes),
            'raw_pixel_total': int(len(raw_X)),
            'raw_pixel_correct': int(raw_ok.sum()),
            'raw_pixel_accuracy': float(raw_ok.mean()) if len(raw_X) else np.nan,
            'raw_pixel_mean_confidence': float(raw_conf.mean()) if len(raw_X) else np.nan,
        })

        if progress_every and ((i + 1) % progress_every == 0 or (i + 1) == n):
            elapsed = time.time() - start
            remaining = (n - (i + 1)) * elapsed / (i + 1)
            print(f'  [{model_name}] fold {i + 1}/{n} | {elapsed / 60:.1f} min elapsed | '
                  f'~{remaining / 60:.1f} min remaining')

    seconds = time.time() - start
    print(f'[✓] [{model_name}] LOO completed in {seconds / 60:.1f} min ({seconds / n:.2f} s/fold).')

    return LOOResult(
        model_name=model_name,
        classes=classes,
        records=pd.DataFrame(records),
        proba=proba,
        pixel_polygon_index=np.concatenate(pix_index),
        pixel_pred=np.concatenate(pix_pred),
        pixel_conf=np.concatenate(pix_conf),
        pixel_correct=np.concatenate(pix_correct),
        seconds=seconds,
    )


# ---------------------------------------------------------------------------
# Metrics & calibration
# ---------------------------------------------------------------------------
def reliability_table(conf, correct, weights=None, n_bins=10):
    """Equal-width reliability bins for top-class confidence.

    Returns (table, ece): per bin, the share of total weight, the weighted mean
    confidence and the weighted observed accuracy; `ece` is the expected
    calibration error, sum_b share_b * |accuracy_b - confidence_b|.
    """
    conf = np.clip(np.asarray(conf, dtype=float), 0.0, 1.0)
    correct = np.asarray(correct, dtype=float)
    w = np.ones_like(conf) if weights is None else np.asarray(weights, dtype=float)
    b = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    w_bin = np.bincount(b, weights=w, minlength=n_bins)
    conf_bin = np.bincount(b, weights=w * conf, minlength=n_bins)
    acc_bin = np.bincount(b, weights=w * correct, minlength=n_bins)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_conf = conf_bin / w_bin
        accuracy = acc_bin / w_bin
    table = pd.DataFrame({
        'bin_low': np.arange(n_bins) / n_bins,
        'bin_high': (np.arange(n_bins) + 1) / n_bins,
        'weight_share': w_bin / w_bin.sum(),
        'mean_confidence': mean_conf,
        'accuracy': accuracy,
    })
    nonempty = w_bin > 0
    ece = float(np.sum(w_bin[nonempty] / w_bin.sum()
                       * np.abs(accuracy[nonempty] - mean_conf[nonempty])))
    return table, ece


def confidence_threshold_summary(conf, correct, threshold, weights=None):
    """Weighted share of predictions below `threshold` and accuracy on either side of it."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    w = np.ones_like(conf) if weights is None else np.asarray(weights, dtype=float)
    below = conf < threshold

    def _accuracy(mask):
        return float(np.sum(w[mask] * correct[mask]) / np.sum(w[mask])) if mask.any() else float('nan')

    return {
        'threshold': float(threshold),
        'share_below': float(w[below].sum() / w.sum()),
        'accuracy_below': _accuracy(below),
        'accuracy_at_or_above': _accuracy(~below),
    }


def pixel_polygon_weights(result: LOOResult):
    """Per-pixel weights that give every held-out polygon a total weight of 1."""
    n_pixels = result.records['raw_pixel_total'].to_numpy(dtype=float)
    return 1.0 / n_pixels[result.pixel_polygon_index]


def summarize_loo(result: LOOResult, reclass_lut=None, n_bins=10):
    """Accuracy, pixel-level and calibration metrics of one LOO run (plain Python floats)."""
    rec = result.records
    y_true = rec['true_code'].to_numpy(dtype=int)
    y_pred = rec['pred_code'].to_numpy(dtype=int)
    n = len(y_true)

    true_pos = np.searchsorted(result.classes, y_true)
    onehot = np.zeros_like(result.proba)
    onehot[np.arange(n), true_pos] = 1.0
    p_true = result.proba[np.arange(n), true_pos]

    pixel_true = y_true[result.pixel_polygon_index]
    poly_w = pixel_polygon_weights(result)
    _, ece_polygon = reliability_table(rec['confidence'], rec['correct'], n_bins=n_bins)
    _, ece_pixel_micro = reliability_table(result.pixel_conf, result.pixel_correct, n_bins=n_bins)
    _, ece_pixel_poly = reliability_table(result.pixel_conf, result.pixel_correct,
                                          weights=poly_w, n_bins=n_bins)
    per_class_pixel_acc = pd.Series(result.pixel_correct).groupby(pixel_true).mean()

    out = {
        'model': result.model_name,
        'n_polygons': int(n),
        'n_raw_pixels': int(len(result.pixel_correct)),
        'accuracy_polygon_mean': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy_polygon_mean': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'top3_accuracy': float(rec['top3_correct'].mean()),
        'log_loss': float(-np.mean(np.log(np.clip(p_true, 1e-15, 1.0)))),
        'brier_score': float(np.mean(np.sum((result.proba - onehot) ** 2, axis=1))),
        'ece_polygon_mean': ece_polygon,
        'raw_pixel_accuracy_micro': float(np.mean(result.pixel_correct)),
        'raw_pixel_accuracy_macro': float(rec['raw_pixel_accuracy'].mean()),
        'raw_pixel_accuracy_class_balanced': float(per_class_pixel_acc.mean()),
        'ece_raw_pixel_micro': ece_pixel_micro,
        'ece_raw_pixel_polygon_weighted': ece_pixel_poly,
        'loo_seconds': float(result.seconds),
    }

    if reclass_lut is not None:
        lut = np.asarray(reclass_lut)
        true17, pred17 = lut[y_true], lut[y_pred]
        pixel_ok17 = lut[result.pixel_pred.astype(int)] == lut[pixel_true]
        out.update({
            'accuracy_17class_polygon_mean': float(np.mean(true17 == pred17)),
            'balanced_accuracy_17class_polygon_mean': float(balanced_accuracy_score(true17, pred17)),
            'raw_pixel_accuracy_17class_micro': float(np.mean(pixel_ok17)),
            'raw_pixel_accuracy_17class_macro': float(
                pd.Series(pixel_ok17).groupby(result.pixel_polygon_index).mean().mean()
            ),
        })
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _chart_rc():
    c = CHART_COLORS
    return {
        'figure.facecolor': c['surface'],
        'axes.facecolor': c['surface'],
        'savefig.facecolor': c['surface'],
        'axes.edgecolor': c['axis'],
        'axes.linewidth': 0.8,
        'axes.labelcolor': c['ink_secondary'],
        'axes.titlecolor': c['ink'],
        'text.color': c['ink'],
        'xtick.color': c['axis'],
        'ytick.color': c['axis'],
        'xtick.labelcolor': c['ink_secondary'],
        'ytick.labelcolor': c['ink_secondary'],
        'axes.grid': True,
        'grid.color': c['grid'],
        'grid.linewidth': 0.6,
        'grid.linestyle': '-',
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'font.size': 10,
    }


def plot_reliability_diagrams(results, n_bins=10, highlight='TabPFN-3.5'):
    """Small-multiple reliability diagrams: one column per model, polygon and pixel level.

    results  {model name: LOOResult}, in display order.
    """
    import matplotlib.pyplot as plt

    names = list(results)
    with plt.rc_context(_chart_rc()):
        fig, axes = plt.subplots(2, len(names), figsize=(3.0 * len(names), 6.9),
                                 sharex=True, sharey=True, squeeze=False)
        for col, name in enumerate(names):
            r = results[name]
            color = CHART_COLORS['accent'] if name == highlight else CHART_COLORS['baseline']
            levels = [
                ('Polygon level', r.records['confidence'].to_numpy(),
                 r.records['correct'].to_numpy(), None),
                ('Pixel level', r.pixel_conf, r.pixel_correct, pixel_polygon_weights(r)),
            ]
            for row, (level, conf, correct, weights) in enumerate(levels):
                ax = axes[row, col]
                table, ece = reliability_table(conf, correct, weights=weights, n_bins=n_bins)
                shown = table['weight_share'] > 0
                ax.plot([0, 1], [0, 1], color=CHART_COLORS['axis'], linewidth=1, zorder=1)
                ax.plot(table.loc[shown, 'mean_confidence'], table.loc[shown, 'accuracy'],
                        color=color, linewidth=2, marker='o', markersize=6,
                        markeredgecolor=CHART_COLORS['surface'], markeredgewidth=1.5, zorder=3)
                header = f'{name}\n' if row == 0 else ''
                ax.set_title(f'{header}{level} · ECE {ece:.3f}', loc='left', fontsize=10,
                             fontweight='bold' if (name == highlight and row == 0) else 'normal')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect('equal')
                ax.set_xticks([0, 0.5, 1])
                ax.set_yticks([0, 0.5, 1])
                if col == 0:
                    ax.set_ylabel('Observed accuracy')
                if row == 1:
                    ax.set_xlabel('Predicted confidence')
        fig.suptitle('Reliability under Leave-One-Polygon-Out cross-validation',
                     x=0.01, ha='left', fontsize=13, fontweight='bold')
        fig.text(0.01, -0.01,
                 'Points on the diagonal: predictions made with confidence p are correct with probability p. '
                 'Polygon level: one mean-embedding prediction per held-out polygon. '
                 'Pixel level: every raw pixel of every held-out polygon, each polygon weighted equally.',
                 ha='left', va='top', fontsize=8.5, color=CHART_COLORS['ink_secondary'], wrap=True)
        fig.tight_layout()
    return fig


COMPARISON_METRICS = [
    ('accuracy_polygon_mean', 'Accuracy (polygon)', True),
    ('balanced_accuracy_polygon_mean', 'Balanced accuracy', True),
    ('macro_f1', 'Macro F1', True),
    ('raw_pixel_accuracy_macro', 'Raw-pixel accuracy (per polygon)', True),
    ('accuracy_17class_polygon_mean', '17-class accuracy (polygon)', True),
    ('ece_polygon_mean', 'Calibration error, ECE (lower is better)', False),
]


def plot_model_comparison(comparison, highlight='TabPFN-3.5', metrics=COMPARISON_METRICS):
    """Dot plots of the headline metrics, one panel per metric, one row per model."""
    import matplotlib.pyplot as plt

    models = list(comparison['model'])
    y = np.arange(len(models))
    ncols = 3
    nrows = math.ceil(len(metrics) / ncols)
    with plt.rc_context(_chart_rc()):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, (0.42 * len(models) + 1.3) * nrows),
                                 sharey=True, squeeze=False)
        for ax, (key, label, higher_is_better) in zip(axes.flat, metrics):
            values = comparison[key].to_numpy(dtype=float)
            is_hl = [m == highlight for m in models]
            ax.scatter(values, y, s=64, zorder=3,
                       c=[CHART_COLORS['accent'] if h else CHART_COLORS['baseline'] for h in is_hl],
                       edgecolors=CHART_COLORS['surface'], linewidths=1.5)
            for yi, v, h in zip(y, values, is_hl):
                ax.annotate(f'{v:.3f}', (v, yi), xytext=(8, 0), textcoords='offset points',
                            va='center', fontsize=9,
                            color=CHART_COLORS['ink'] if h else CHART_COLORS['ink_secondary'],
                            fontweight='bold' if h else 'normal')
            lo, hi = np.nanmin(values), np.nanmax(values)
            if higher_is_better:
                x0 = max(0.0, math.floor((lo - 0.03) * 20) / 20)
            else:
                x0 = 0.0
            ax.set_xlim(x0, hi + 0.35 * max(hi - x0, 0.02))
            ax.set_title(label, loc='left', fontsize=10)
            ax.grid(axis='y', visible=False)
            ax.set_yticks(y)
            ax.set_yticklabels(models)
        for ax in list(axes.flat)[len(metrics):]:
            ax.set_visible(False)
        axes[0, 0].invert_yaxis()
        for ax in axes[:, 0]:
            for tick, h in zip(ax.get_yticklabels(), [m == highlight for m in models]):
                if h:
                    tick.set_fontweight('bold')
                    tick.set_color(CHART_COLORS['ink'])
        fig.suptitle('TabPFN-3.5 vs. baselines · identical Leave-One-Polygon-Out protocol',
                     x=0.01, ha='left', fontsize=13, fontweight='bold')
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Raster utilities
# ---------------------------------------------------------------------------
def lossless_float_codec(candidates=('zstd', 'deflate', 'lzw')):
    """First GeoTIFF codec in `candidates` that this GDAL build applies losslessly
    to float32 data with the floating-point predictor (PREDICTOR=3)."""
    import warnings

    from rasterio.io import MemoryFile

    probe = np.random.default_rng(0).random((1, 64, 64)).astype(np.float32)
    for codec in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with MemoryFile() as mem:
                    with mem.open(driver='GTiff', width=64, height=64, count=1, dtype='float32',
                                  compress=codec, predictor=3) as dst:
                        dst.write(probe)
                    with mem.open() as src:
                        used = (src.tags(ns='IMAGE_STRUCTURE').get('COMPRESSION') or '').lower()
                        exact = np.array_equal(src.read(), probe)
            if used == codec and exact:
                return codec
        except Exception:  # noqa: BLE001 - an unsupported codec simply moves to the next candidate
            continue
    raise RuntimeError(f'None of the codecs {candidates} is available in this GDAL build.')


def lossless_float_profile(codec, block_size=512):
    """GeoTIFF creation options for lossless float storage with `codec`."""
    return {
        'compress': codec,
        'predictor': 3,
        'tiled': True,
        'blockxsize': block_size,
        'blockysize': block_size,
        'BIGTIFF': 'YES',
        'NUM_THREADS': 'ALL_CPUS',
    }


def verify_mosaic_against_tiles(mosaic_path, tile_paths, progress_every=50):
    """Bit-exact comparison of every tile raster with its window in the mosaic.

    Raises AssertionError on the first mismatch; returns the number of tiles checked.
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(mosaic_path) as mosaic:
        T = mosaic.transform
        for i, path in enumerate(tile_paths):
            with rasterio.open(path) as tile:
                col_off = int(round((tile.transform.c - T.c) / T.a))
                row_off = int(round((tile.transform.f - T.f) / T.e))
                expected = tile.read(1)
                stored = mosaic.read(1, window=Window(col_off, row_off, tile.width, tile.height))
            if expected.dtype != stored.dtype or expected.tobytes() != stored.tobytes():
                raise AssertionError(f'Mosaic differs from tile {path}')
            if progress_every and ((i + 1) % progress_every == 0 or (i + 1) == len(tile_paths)):
                print(f'  verified {i + 1}/{len(tile_paths)} tiles against {mosaic_path}')
    return len(tile_paths)
