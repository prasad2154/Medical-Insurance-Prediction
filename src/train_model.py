"""
src/train_model.py
End-to-end Machine Learning training pipeline for Medical Insurance Cost Prediction.
Trains, evaluates, compares multiple regression models, and serializes the champion model pipeline.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path regardless of execution directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from src.utils import (
        load_dataset,
        get_model_path,
        get_metrics_path,
        get_base_dir,
        MODEL_DIR,
    )
    from src.preprocessing import (
        create_full_pipeline,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        TARGET_FEATURE,
    )
except ModuleNotFoundError:
    from utils import (
        load_dataset,
        get_model_path,
        get_metrics_path,
        get_base_dir,
        MODEL_DIR,
    )
    from preprocessing import (
        create_full_pipeline,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        TARGET_FEATURE,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes regression performance metrics:
    - MAE: Mean Absolute Error
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - R2: Coefficient of Determination
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MAE": round(float(mae), 2),
        "MSE": round(float(mse), 2),
        "RMSE": round(float(rmse), 2),
        "R2": round(float(r2), 4),
    }


def get_feature_names_from_preprocessor(preprocessor) -> list:
    """
    Extracts output feature names from the fitted ColumnTransformer.
    """
    try:
        # Numeric names
        num_cols = list(NUMERICAL_FEATURES)
        # OneHotEncoded names
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_cols = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        return num_cols + cat_cols
    except Exception as e:
        logger.warning("Could not extract feature names dynamically: %s", e)
        return ["age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest"]


def train_and_save_pipeline() -> dict:
    """
    Executes full training pipeline:
    1. Loads dataset
    2. Splits into train & test (80/20 split, random_state=42)
    3. Fits & compares Linear Regression, Random Forest, Gradient Boosting
    4. Identifies Champion model based on test R2 Score
    5. Serializes champion pipeline to models/insurance_model.pkl
    6. Saves comparative metrics, test predictions, and feature importance to models/model_metrics.json
    """
    logger.info("Starting Medical Insurance ML Model Training Pipeline...")
    
    # Ensure models directory exists
    base_dir = get_base_dir()
    models_dir_path = os.path.join(base_dir, MODEL_DIR)
    os.makedirs(models_dir_path, exist_ok=True)

    # 1. Load Dataset
    df = load_dataset()
    logger.info("Dataset loaded successfully. Shape: %s", df.shape)

    # Drop duplicate records if any
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        logger.info("Found %d duplicate rows. Removing duplicates...", duplicates_count)
        df = df.drop_duplicates().reset_index(drop=True)

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_FEATURE]

    # 2. Train-Test Split (avoiding data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    logger.info("Train set: %d rows | Test set: %d rows", len(X_train), len(X_test))

    # 3. Candidate Models
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=6, min_samples_leaf=3, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.08, max_depth=3, subsample=0.85, random_state=42
        ),
    }

    trained_pipelines = {}
    metrics_summary = {}
    test_predictions = {}

    best_model_name = None
    best_r2_score = -float("inf")

    print("\n" + "="*70)
    print("           MODEL BENCHMARKING & EVALUATION RESULTS")
    print("="*70)
    print(f"{'Model Name':<22} | {'MAE ($)':<10} | {'RMSE ($)':<10} | {'R² Score':<10}")
    print("-"*70)

    for name, estimator in candidate_models.items():
        # Create pipeline with ColumnTransformer + Regressor
        pipeline = create_full_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        # Predictions
        y_test_pred = pipeline.predict(X_test)
        metrics = evaluate_model(y_test.values, y_test_pred)

        trained_pipelines[name] = pipeline
        metrics_summary[name] = metrics
        test_predictions[name] = [round(float(p), 2) for p in y_test_pred]

        print(f"{name:<22} | {metrics['MAE']:<10.2f} | {metrics['RMSE']:<10.2f} | {metrics['R2']:<10.4f}")

        if metrics["R2"] > best_r2_score:
            best_r2_score = metrics["R2"]
            best_model_name = name

    print("="*70)
    print(f"🏆 Champion Model: {best_model_name} (R² = {best_r2_score:.4f})\n")

    champion_pipeline = trained_pipelines[best_model_name]

    # 4. Extract Feature Importances for Champion (or tree-based model)
    feature_importance_dict = {}
    try:
        tree_model = trained_pipelines["Gradient Boosting"].named_steps["regressor"]
        preprocessor = trained_pipelines["Gradient Boosting"].named_steps["preprocessor"]
        feature_names = get_feature_names_from_preprocessor(preprocessor)
        raw_importances = tree_model.feature_importances_

        sorted_indices = np.argsort(raw_importances)[::-1]
        for idx in sorted_indices:
            feature_importance_dict[feature_names[idx]] = round(float(raw_importances[idx]), 4)
    except Exception as e:
        logger.warning("Feature importance extraction warning: %s", e)

    # 5. Serialize Champion Pipeline
    model_save_path = get_model_path()
    joblib.dump(champion_pipeline, model_save_path)
    logger.info("Saved champion pipeline to %s", model_save_path)

    # 6. Save comprehensive evaluation metrics JSON
    metrics_payload = {
        "champion_model": best_model_name,
        "models_comparison": metrics_summary,
        "feature_importance": feature_importance_dict,
        "test_ground_truth": [round(float(val), 2) for val in y_test.values],
        "test_predictions": test_predictions,
        "test_sample_features": X_test.head(15).to_dict(orient="records"),
        "dataset_metadata": {
            "total_records": int(len(df)),
            "features_count": int(X.shape[1]),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "charges_mean": round(float(df["charges"].mean()), 2),
            "charges_median": round(float(df["charges"].median()), 2),
            "charges_std": round(float(df["charges"].std()), 2),
            "bmi_mean": round(float(df["bmi"].mean()), 2),
            "age_mean": round(float(df["age"].mean()), 2),
            "smoker_ratio": round(float((df["smoker"] == "yes").mean() * 100), 2),
        }
    }

    metrics_save_path = get_metrics_path()
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)
    logger.info("Saved model evaluation metrics to %s", metrics_save_path)

    return metrics_payload


if __name__ == "__main__":
    train_and_save_pipeline()
