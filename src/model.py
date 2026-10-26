"""
Machine Learning & Domain Models for Pavement Roughness (IRI) Prediction.
Author: Muhammad Umer Basit
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics (MAE, RMSE, R2)."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "R2": round(float(r2), 4),
    }


def predict_piecewise_domain(sn: float, aadt: float) -> float:
    """
    Piecewise decision tree / multi-linear regression formulation
    developed from the N-5 Highway pavement monitoring dataset.

    Parameters:
        sn: Effective Structural Number (SN) from HWD deflection backcalculation.
        aadt: Annual Average Daily Traffic (vehicles/day).

    Returns:
        Predicted International Roughness Index (IRI in in/mi).
    """
    if aadt < 400:
        return sn * 0.25233439 + aadt * 0.00408931 + 175.062028028023
    elif aadt < 1500:
        return 3.24444 * sn + 144.631
    elif aadt < 2500:
        return -0.1235 * sn + 202.94
    elif aadt < 3500:
        return -5.886 * sn + 220.602
    elif aadt < 5000:
        return 13.75103 * sn + 104.9813
    elif aadt < 8000:
        return 0.25946 * sn + 177.913
    else:
        return sn * 0.25233439 + aadt * 0.00408931 + 175.062028028023


def classify_pavement_condition(iri_in_per_mi: float) -> Tuple[str, str, str]:
    """
    Classify pavement condition and recommend treatment according to
    Federal Highway Administration (FHWA) and World Bank IRI thresholds.

    Returns:
        (Condition Rating, Ride Quality, Recommended Action)
    """
    if iri_in_per_mi < 95.0:
        return (
            "Good",
            "Smooth Ride Quality",
            "Routine maintenance & crack sealing",
        )
    elif iri_in_per_mi <= 170.0:
        return (
            "Fair",
            "Acceptable Ride Quality",
            "Preventative maintenance (slurry seal / micro-surfacing)",
        )
    else:
        return (
            "Poor",
            "Deteriorated / Rough Surface",
            "Structural rehabilitation (mill & overlay or deep reconstruction)",
        )


def train_and_benchmark(csv_path: str) -> Dict[str, Any]:
    """Load dataset, train baseline and ensemble models, and evaluate performance."""
    df = pd.read_csv(csv_path)

    X = df[["Structural_Number_SN", "AADT"]].values
    y = df["IRI_in_per_mi"].values

    models = {
        "Domain Piecewise Model": None,
        "Multi-Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=4, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    }

    results = {}
    predictions = {}

    # 1. Domain model
    y_pred_domain = np.array([predict_piecewise_domain(sn, aadt) for sn, aadt in X])
    results["Domain Piecewise Model"] = evaluate_metrics(y, y_pred_domain)
    predictions["Domain Piecewise Model"] = y_pred_domain

    # 2. Sklearn models
    for name, model in models.items():
        if model is None:
            continue
        model.fit(X, y)
        y_pred = model.predict(X)
        results[name] = evaluate_metrics(y, y_pred)
        predictions[name] = y_pred

    return {
        "df": df,
        "models": models,
        "results": results,
        "predictions": predictions,
    }
