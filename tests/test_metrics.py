import numpy as np

from src.credit_risk.modeling.metrics import calculate_classification_metrics


def test_calculate_classification_metrics() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_probability = np.array([0.1, 0.9, 0.4, 0.2])

    metrics = calculate_classification_metrics(y_true, y_pred, y_probability)

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] > 0
    assert metrics["roc_auc"] == 1.0
