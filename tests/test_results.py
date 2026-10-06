"""The numbers reported in README.md, recomputed from the result files in the repository.

Only small committed files are read (data/, outputs_tabpfn35_polygon/,
predictions_tabpfn35_polygon/), so these checks run in seconds without a GPU,
TabPFN or the large input data. A failure means README.md and the saved results
disagree.
"""
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import lc_eval

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / 'outputs_tabpfn35_polygon'
PREDICTIONS = REPO / 'predictions_tabpfn35_polygon'


def percent(expected):
    """A percentage as printed in README.md (one decimal)."""
    return pytest.approx(expected / 100, abs=0.0005)


def three_decimals(expected):
    return pytest.approx(expected, abs=0.0005)


# --- Training data -------------------------------------------------------------

def test_training_context(polygon_means):
    assert len(polygon_means) == 318
    assert polygon_means['polygon_id'].is_unique
    assert all(f'A{i:02d}' in polygon_means for i in range(64))
    assert polygon_means['code'].nunique() == 23
    assert polygon_means['n_pixels'].sum() == 440_208

    per_class = polygon_means['code'].value_counts()
    rare = per_class[per_class < 10]
    assert len(rare) == 12 and rare.min() == 3


def test_loo_holds_out_every_polygon_once(loo, polygon_means):
    assert len(loo) == 318
    assert loo['polygon_id'].is_unique
    merged = loo.merge(polygon_means[['polygon_id', 'code']], on='polygon_id', validate='one_to_one')
    assert len(merged) == 318
    assert (merged['true_code'] == merged['code']).all()


# --- TabPFN-3.5, Leave-One-Polygon-Out ------------------------------------------

def test_headline_accuracy(loo, metrics):
    y_true, y_pred = loo['true_code'], loo['pred_code']
    accuracy = accuracy_score(y_true, y_pred)
    balanced = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    assert accuracy == percent(83.3)
    assert balanced == percent(80.7)
    assert macro_f1 == three_decimals(0.809)
    assert loo['top3_correct'].mean() == percent(97.5)

    assert accuracy == pytest.approx(metrics['accuracy_polygon_mean'], rel=1e-12)
    assert balanced == pytest.approx(metrics['balanced_accuracy_polygon_mean'], rel=1e-12)
    assert macro_f1 == pytest.approx(metrics['macro_f1'], rel=1e-12)


def test_raw_pixel_accuracy(loo, metrics):
    assert loo['raw_pixel_total'].sum() == 440_208
    per_pixel = loo['raw_pixel_correct'].sum() / loo['raw_pixel_total'].sum()
    per_polygon = loo['raw_pixel_accuracy'].mean()
    by_class = loo.groupby('true_code')[['raw_pixel_correct', 'raw_pixel_total']].sum()
    per_class = (by_class['raw_pixel_correct'] / by_class['raw_pixel_total']).mean()

    assert per_pixel == percent(87.2)
    assert per_polygon == percent(81.6)
    assert per_class == percent(81.5)

    assert per_pixel == pytest.approx(metrics['raw_pixel_accuracy_micro'], rel=1e-9)
    assert per_polygon == pytest.approx(metrics['raw_pixel_accuracy_macro'], rel=1e-9)
    assert per_class == pytest.approx(metrics['raw_pixel_accuracy_class_balanced'], rel=1e-9)


def test_rare_classes_recall(loo, metrics):
    per_class = loo.groupby('true_code')['correct'].agg(['size', 'mean'])
    rare, common = per_class[per_class['size'] < 10], per_class[per_class['size'] >= 10]

    assert len(rare) == 12 and len(common) == 11
    assert sorted(rare.index) == sorted(metrics['classes_with_fewer_than_10_polygons'])
    assert rare['mean'].mean() == percent(81.0)
    assert common['mean'].mean() == percent(80.4)


@pytest.mark.parametrize('name, correct, total', [
    ('cocoa', 47, 49), ('Palm', 12, 12), ('Rubber', 7, 7), ('mangrove20', 45, 49),
    ('built-up', 14, 15), ('cashew', 18, 20),
    ('Rice', 3, 9), ('Open_Low_Shrubland', 6, 14), ('Shrub_Crop', 8, 15), ('Closed_Forest', 5, 8),
])
def test_strongest_and_weakest_classes(loo, name, correct, total):
    held_out = loo[loo['true_name'] == name]
    assert (int(held_out['correct'].sum()), len(held_out)) == (correct, total)


def test_class_summary_intervals_come_from_wilson_ci():
    summary = pd.read_csv(OUTPUTS / 'polygon_class_summary.csv')
    for row in summary.itertuples():
        low, high = lc_eval.wilson_ci(row.loo_correct, row.loo_total)
        assert (row.loo_accuracy_ci95_low, row.loo_accuracy_ci95_high) == pytest.approx((low, high), abs=1e-12)


def test_fit_and_evaluation_time(metrics):
    assert metrics['final_fit_time_seconds'] == pytest.approx(1.6, abs=0.05)
    assert metrics['loo_seconds'] / 60 == pytest.approx(11.4, abs=0.05)


# --- TabPFN-3.5 vs. baselines ----------------------------------------------------

COMPARISON_COLUMNS = [
    # (column in model_comparison.csv, printed as a percentage in README.md)
    ('accuracy_polygon_mean', True),
    ('balanced_accuracy_polygon_mean', True),
    ('macro_f1', False),
    ('raw_pixel_accuracy_macro', True),
    ('raw_pixel_accuracy_class_balanced', True),
    ('accuracy_17class_polygon_mean', True),
    ('log_loss', False),
    ('brier_score', False),
    ('ece_raw_pixel_polygon_weighted', False),
]
README_COMPARISON = {
    'TabPFN-3.5': (83.3, 80.7, 0.809, 81.6, 81.5, 84.3, 0.545, 0.249, 0.040),
    'Logistic Regression': (80.8, 76.3, 0.772, 78.2, 76.3, 82.1, 0.727, 0.294, 0.042),
    'kNN (k=5)': (80.5, 74.6, 0.749, 75.5, 68.7, 82.4, 2.729, 0.309, 0.053),
    'Random Forest': (80.2, 72.8, 0.747, 75.7, 67.3, 82.4, 0.906, 0.383, 0.251),
    'XGBoost': (78.3, 69.5, 0.698, 73.8, 64.6, 80.5, 0.863, 0.338, 0.038),
}


@pytest.mark.parametrize('model', README_COMPARISON)
def test_comparison_table(comparison, model):
    row = comparison.loc[model]
    for (column, is_percent), expected in zip(COMPARISON_COLUMNS, README_COMPARISON[model]):
        assert row[column] == (percent(expected) if is_percent else three_decimals(expected)), column


def test_tabpfn_ranks_first(comparison):
    higher_is_better = [
        'accuracy_polygon_mean', 'balanced_accuracy_polygon_mean', 'macro_f1', 'macro_precision',
        'macro_recall', 'top3_accuracy', 'raw_pixel_accuracy_micro', 'raw_pixel_accuracy_macro',
        'raw_pixel_accuracy_class_balanced', 'accuracy_17class_polygon_mean',
        'balanced_accuracy_17class_polygon_mean', 'raw_pixel_accuracy_17class_micro',
        'raw_pixel_accuracy_17class_macro',
    ]
    for column in higher_is_better:
        assert comparison[column].idxmax() == 'TabPFN-3.5', column
    for column in ('log_loss', 'brier_score'):
        assert comparison[column].idxmin() == 'TabPFN-3.5', column

    baselines = comparison.drop(index='TabPFN-3.5')
    assert baselines['accuracy_polygon_mean'].max() == percent(80.8)
    assert baselines['balanced_accuracy_polygon_mean'].max() == percent(76.3)
    assert baselines['log_loss'].min() == three_decimals(0.727)
    assert baselines['brier_score'].min() == three_decimals(0.294)


# --- Calibration -----------------------------------------------------------------

def test_calibration_table(calibration, calibration_bins):
    tabpfn = calibration['models']['TabPFN-3.5']['raw_pixel_polygon_weighted']
    assert tabpfn['threshold'] == 0.6
    assert 1 - tabpfn['share_below'] == percent(77.8)
    assert tabpfn['accuracy_at_or_above'] == percent(90.1)
    assert tabpfn['share_below'] == percent(22.2)
    assert tabpfn['accuracy_below'] == percent(51.7)

    top_bin = calibration_bins[(calibration_bins['model'] == 'TabPFN-3.5')
                               & (calibration_bins['level'] == 'raw_pixel_polygon_weighted')
                               & (calibration_bins['bin_low'] == 0.9)].iloc[0]
    assert top_bin['weight_share'] == percent(42.5)
    assert top_bin['accuracy'] == percent(95.6)
    assert top_bin['mean_confidence'] == three_decimals(0.962)

    forest = calibration['models']['Random Forest']['raw_pixel_polygon_weighted']
    assert forest['share_below'] == pytest.approx(0.64, abs=0.005)
    assert forest['accuracy_below'] == pytest.approx(0.64, abs=0.005)


def test_calibration_bins_match_the_summaries(calibration, calibration_bins):
    """ECE and the 0.6-threshold figures follow from the reliability bins of every model and level."""
    for model, levels in calibration['models'].items():
        for level, summary in levels.items():
            bins = calibration_bins[(calibration_bins['model'] == model) & (calibration_bins['level'] == level)]
            bins = bins[bins['weight_share'] > 0]
            assert bins['weight_share'].sum() == pytest.approx(1.0, abs=1e-9), (model, level)

            ece = (bins['weight_share'] * (bins['accuracy'] - bins['mean_confidence']).abs()).sum()
            assert ece == pytest.approx(summary['ece'], abs=1e-6), (model, level)

            above = bins[bins['bin_low'] >= summary['threshold'] - 1e-9]
            share_above = above['weight_share'].sum()
            assert share_above == pytest.approx(1 - summary['share_below'], abs=1e-6), (model, level)
            accuracy_above = (above['weight_share'] * above['accuracy']).sum() / share_above
            assert accuracy_above == pytest.approx(summary['accuracy_at_or_above'], abs=1e-6), (model, level)


# --- The national map --------------------------------------------------------------

README_AREAS_KM2 = {
    'Closed Forest': 6_392, 'Open Forest': 18_880, 'Mangrove': 504,
    'Closed Savannah': 16_578, 'Open Savannah': 25_283, 'Open Low Shrubland': 33_495,
    'Tree Crop Plantation': 29_798, 'Annual Herbaceous Cultivation': 59_591, 'Shrub Crop': 22_263,
    'Swamp': 4_086, 'Seasonally Flooded Woody': 3_431, 'Permanently Flooded Woody': 155, 'Water': 8_706,
    'Built up': 4_536, 'Mining': 2_128, 'Bare Rocks': 2_891, 'Beaches': 121,
}


def test_land_area_table():
    areas = pd.read_csv(PREDICTIONS / 'final_class_land_area.csv').set_index('name')['area_km2']
    assert sorted(areas.index) == sorted(README_AREAS_KM2)
    for name, km2 in README_AREAS_KM2.items():
        assert round(areas[name]) == km2, name
    assert round(areas.sum()) == 238_836


def test_national_pixel_count():
    counts = pd.read_csv(PREDICTIONS / 'predicted_pixel_counts.csv')
    classified = counts.loc[counts['code'] != 0, 'pixel_count'].sum()
    assert classified == 2_461_544_689
    assert round(classified / 1e9, 2) == 2.46
