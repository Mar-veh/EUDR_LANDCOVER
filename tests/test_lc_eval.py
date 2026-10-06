"""lc_eval's evaluation code on small made-up inputs with known answers."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import lc_eval

REPO = Path(__file__).resolve().parents[1]


# --- Leave-One-Polygon-Out ------------------------------------------------------------

class RecordingNearestNeighbour:
    """1-nearest-neighbour on polygon means that records the rows of every fit."""

    fits = []

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)
        RecordingNearestNeighbour.fits.append({tuple(row) for row in self.X_})
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        nearest = self.y_[((X[:, None, :] - self.X_[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)]
        return (nearest[:, None] == self.classes_[None, :]).astype(float)


@pytest.fixture
def toy_polygons():
    """Six polygons in two classes; polygon 'e' (class 2) lies among the class-1 polygons."""
    means = pd.DataFrame({'A00': [0.0, 0.2, 0.6, 10.0, 1.3, 11.0],
                          'A01': [0.0, 0.0, 0.0, 10.0, 0.0, 10.0]})
    labels = [1, 1, 1, 2, 2, 2]
    ids = ['a', 'b', 'c', 'd', 'e', 'f']
    offsets = np.array([[0.05, 0.0], [0.0, 0.05], [-0.05, 0.0]])
    pixels = {pid: pd.DataFrame(means.iloc[[i]].to_numpy() + offsets, columns=means.columns)
              for i, pid in enumerate(ids)}
    return means, labels, ids, pixels


def test_run_loo_never_trains_on_the_held_out_polygon(toy_polygons):
    means, labels, ids, pixels = toy_polygons
    RecordingNearestNeighbour.fits = []
    result = lc_eval.run_loo(RecordingNearestNeighbour, means, labels, ids, pixels,
                             classes=[1, 2], model_name='toy', progress_every=0)

    assert len(RecordingNearestNeighbour.fits) == len(ids)
    rows = [tuple(row) for row in means.to_numpy()]
    for held_out, seen in zip(rows, RecordingNearestNeighbour.fits):
        assert held_out not in seen
        assert seen == set(rows) - {held_out}

    assert result.records['polygon_id'].tolist() == ids
    assert len(result.pixel_pred) == 3 * len(ids)
    assert result.pixel_polygon_index.tolist() == [i for i in range(len(ids)) for _ in range(3)]


def test_run_loo_and_summary_scores(toy_polygons):
    means, labels, ids, pixels = toy_polygons
    result = lc_eval.run_loo(RecordingNearestNeighbour, means, labels, ids, pixels,
                             classes=[1, 2], model_name='toy', progress_every=0)

    # Only polygon 'e' is misclassified, on its mean and on all three of its pixels.
    assert result.records['correct'].tolist() == [True, True, True, True, False, True]
    assert result.records['raw_pixel_correct'].tolist() == [3, 3, 3, 3, 0, 3]

    summary = lc_eval.summarize_loo(result)
    assert summary['n_polygons'] == 6 and summary['n_raw_pixels'] == 18
    assert summary['accuracy_polygon_mean'] == pytest.approx(5 / 6)
    assert summary['balanced_accuracy_polygon_mean'] == pytest.approx((1 + 2 / 3) / 2)
    assert summary['raw_pixel_accuracy_micro'] == pytest.approx(15 / 18)
    assert summary['raw_pixel_accuracy_macro'] == pytest.approx(5 / 6)
    assert summary['raw_pixel_accuracy_class_balanced'] == pytest.approx((1 + 2 / 3) / 2)


def test_pixel_weights_give_every_polygon_equal_weight(toy_polygons):
    means, labels, ids, pixels = toy_polygons
    pixels = dict(pixels, a=pd.concat([pixels['a']] * 4, ignore_index=True))  # 12 pixels instead of 3
    result = lc_eval.run_loo(RecordingNearestNeighbour, means, labels, ids, pixels,
                             classes=[1, 2], model_name='toy', progress_every=0)
    weights = lc_eval.pixel_polygon_weights(result)
    per_polygon = pd.Series(weights).groupby(result.pixel_polygon_index).sum()
    assert per_polygon.to_numpy() == pytest.approx(np.ones(len(ids)))


# --- Calibration ------------------------------------------------------------------------

def test_ece_is_zero_when_confidence_matches_accuracy():
    conf = np.full(8, 0.75)
    correct = np.array([1, 1, 1, 0, 1, 1, 1, 0])
    table, ece = lc_eval.reliability_table(conf, correct)
    assert ece == pytest.approx(0.0)
    assert len(table) == 10 and table['weight_share'].sum() == pytest.approx(1.0)


def test_ece_of_an_overconfident_model():
    _, ece = lc_eval.reliability_table(np.full(10, 0.9), np.array([1, 0] * 5))
    assert ece == pytest.approx(0.4)


def test_weighted_ece_by_hand():
    conf = np.array([0.95, 0.95, 0.55, 0.55])
    correct = np.array([1, 0, 1, 1])
    _, ece = lc_eval.reliability_table(conf, correct, weights=np.array([3, 1, 1, 1]))
    # 0.9-1.0 bin: weight 4/6, accuracy 3/4, confidence 0.95; 0.5-0.6 bin: weight 2/6, accuracy 1, confidence 0.55
    assert ece == pytest.approx(4 / 6 * 0.2 + 2 / 6 * 0.45)


def test_confidence_of_one_falls_in_the_top_bin():
    table, _ = lc_eval.reliability_table(np.array([1.0, 0.95]), np.array([1, 1]))
    assert table.loc[9, 'weight_share'] == pytest.approx(1.0)


def test_threshold_summary_counts_the_threshold_as_confident():
    conf = np.array([0.2, 0.5, 0.6, 0.9])
    correct = np.array([0, 1, 1, 1])
    s = lc_eval.confidence_threshold_summary(conf, correct, 0.6)
    assert (s['share_below'], s['accuracy_below'], s['accuracy_at_or_above']) == pytest.approx((0.5, 0.5, 1.0))

    weighted = lc_eval.confidence_threshold_summary(conf, correct, 0.6, weights=np.array([1, 3, 1, 1]))
    assert (weighted['share_below'], weighted['accuracy_below']) == pytest.approx((4 / 6, 3 / 4))


def test_wilson_interval():
    low, high = lc_eval.wilson_ci(47, 49)
    assert (round(low, 3), round(high, 3)) == (0.863, 0.989)
    assert lc_eval.wilson_ci(0, 5)[0] == pytest.approx(0.0, abs=1e-12)
    assert lc_eval.wilson_ci(5, 5)[1] == pytest.approx(1.0, abs=1e-12)
    assert all(np.isnan(lc_eval.wilson_ci(0, 0)))


# --- Legend and reclassification ---------------------------------------------------------

def test_reclassification_maps_23_classes_to_the_17_class_legend():
    lut, table = lc_eval.load_reclass_lut(REPO / 'new_class.xlsx')
    assert len(table) == 23
    assert lut[0] == 0
    assert sorted(set(lut[table['Old Class Code']])) == list(range(1, 18))
    tree_crops = {'cocoa': 22, 'cashew': 23, 'Rubber': 8, 'Palm': 10, 'Mango': 16}
    assert {name: int(lut[code]) for name, code in tree_crops.items()} == dict.fromkeys(tree_crops, 8)


def test_color_table():
    names, rgba = lc_eval.load_color_table(REPO / 'color_code.txt')
    assert sorted(code for code in names if code) == list(range(1, 18))
    assert names[8] == 'Tree Crop Plantation'
    assert all(len(color) == 4 and all(0 <= v <= 255 for v in color) for color in rgba.values())


def test_parse_point_coordinates():
    geo = pd.Series(['{"type":"Point","coordinates":[-1.625,6.6875]}'])
    coords = lc_eval.parse_point_coordinates(geo)
    assert coords.iloc[0].tolist() == [-1.625, 6.6875]
