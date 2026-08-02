"""Numerically derivative-free helpers for spent lens-deflection atlases."""

from __future__ import annotations

import numpy as np


def vector_rms(x_values, y_values) -> float:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    return float(np.sqrt(np.mean(x**2 + y**2)))


def vector_alignment(first_x, first_y, second_x, second_y) -> float:
    first_x = np.asarray(first_x, dtype=float)
    first_y = np.asarray(first_y, dtype=float)
    second_x = np.asarray(second_x, dtype=float)
    second_y = np.asarray(second_y, dtype=float)
    numerator = float(np.sum(first_x * second_x + first_y * second_y))
    first_norm = float(np.sum(first_x**2 + first_y**2))
    second_norm = float(np.sum(second_x**2 + second_y**2))
    denominator = np.sqrt(first_norm * second_norm)
    return numerator / max(float(denominator), np.finfo(float).tiny)


def loglog_interpolate(x_values, anchor_x, anchor_y):
    x = np.asarray(x_values, dtype=float)
    anchors = np.asarray(anchor_x, dtype=float)
    values = np.asarray(anchor_y, dtype=float)
    if np.any(x <= 0.0) or np.any(anchors <= 0.0) or np.any(values <= 0.0):
        raise ValueError("log-log interpolation requires positive values")
    return np.exp(np.interp(np.log(x), np.log(anchors), np.log(values)))


def leave_one_out_log_linear(x_values, y_values) -> tuple[np.ndarray, float]:
    """Return leave-one-out predictions and their RMSE for y=a+b*x."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 3:
        raise ValueError("x and y must be equal one-dimensional arrays with >=3 rows")
    predicted = np.empty_like(y)
    for index in range(len(y)):
        use = np.arange(len(y)) != index
        design = np.column_stack([np.ones(np.sum(use)), x[use]])
        coefficients = np.linalg.lstsq(design, y[use], rcond=None)[0]
        predicted[index] = coefficients[0] + coefficients[1] * x[index]
    rmse = float(np.sqrt(np.mean((predicted - y) ** 2)))
    return predicted, rmse


def leave_one_out_constant(y_values) -> tuple[np.ndarray, float]:
    y = np.asarray(y_values, dtype=float)
    if y.ndim != 1 or len(y) < 2:
        raise ValueError("y must be one-dimensional with >=2 rows")
    predicted = np.asarray(
        [np.mean(np.delete(y, index)) for index in range(len(y))], dtype=float
    )
    rmse = float(np.sqrt(np.mean((predicted - y) ** 2)))
    return predicted, rmse

