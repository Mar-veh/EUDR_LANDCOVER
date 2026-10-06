import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / 'outputs_tabpfn35_polygon'
PREDICTIONS = REPO / 'predictions_tabpfn35_polygon'

# lc_eval.py lives in the repository root.
sys.path.insert(0, str(REPO))


@pytest.fixture(scope='session')
def polygon_means():
    return pd.read_csv(REPO / 'data' / 'polygon_means.csv')


@pytest.fixture(scope='session')
def loo():
    return pd.read_csv(OUTPUTS / 'loo_predictions.csv')


@pytest.fixture(scope='session')
def metrics():
    return json.loads((OUTPUTS / 'polygon_metrics_summary.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='session')
def comparison():
    return pd.read_csv(OUTPUTS / 'model_comparison.csv').set_index('model')


@pytest.fixture(scope='session')
def calibration():
    return json.loads((OUTPUTS / 'calibration_summary.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='session')
def calibration_bins():
    return pd.read_csv(OUTPUTS / 'calibration_bins.csv')
