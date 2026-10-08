"""
src/utils.py
Helper utilities for data management, BMI categorization, currency formatting,
and model serialization for Medical Insurance Cost Predictor.
"""

import os
import sys
import json
import logging
import urllib.request
import pandas as pd
import joblib

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Canonical raw dataset URL
DATASET_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
DATASET_FILENAME = "insurance.csv"
MODEL_DIR = "models"
MODEL_FILENAME = os.path.join(MODEL_DIR, "insurance_model.pkl")
METRICS_FILENAME = os.path.join(MODEL_DIR, "model_metrics.json")


def get_base_dir() -> str:
    """Returns the base project directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_dataset_path() -> str:
    """Returns absolute path to insurance.csv in project root."""
    return os.path.join(get_base_dir(), DATASET_FILENAME)


def get_model_path() -> str:
    """Returns absolute path to saved model pickle."""
    return os.path.join(get_base_dir(), MODEL_FILENAME)


def get_metrics_path() -> str:
    """Returns absolute path to model metrics json."""
    return os.path.join(get_base_dir(), METRICS_FILENAME)


def ensure_dataset_available() -> str:
    """
    Ensures that insurance.csv exists in the project root with all 1,338 records.
    If not present or incomplete, fetches it from canonical public repository.
    """
    csv_path = get_dataset_path()
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 50000:
        return csv_path

    logger.info("Dataset not found locally or incomplete (<50KB). Fetching from %s", DATASET_URL)
    try:
        urllib.request.urlretrieve(DATASET_URL, csv_path)
        logger.info("Successfully downloaded insurance.csv (%d bytes)", os.path.getsize(csv_path))
        return csv_path
    except Exception as e:
        logger.error("Failed to download dataset: %s", e)
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 5000:
            logger.warning("Using existing local insurance.csv (%d bytes) as fallback.", os.path.getsize(csv_path))
            return csv_path
        raise FileNotFoundError(
            f"Could not load insurance.csv locally or from URL ({DATASET_URL}). "
            f"Please ensure insurance.csv is in the project root directory."
        ) from e


def load_dataset() -> pd.DataFrame:
    """
    Loads and validates the insurance dataset.
    Returns a pandas DataFrame with expected 7 columns.
    """
    csv_path = ensure_dataset_available()
    df = pd.read_csv(csv_path)
    
    expected_cols = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"Dataset columns missing. Expected at least {expected_cols}, got {set(df.columns)}")
        
    return df


def calculate_bmi_category(bmi: float) -> dict:
    """
    Returns WHO BMI classification, risk level, and visual badge color class.
    """
    if bmi < 18.5:
        return {
            "category": "Underweight",
            "badge_class": "badge-blue",
            "color": "#3B82F6",
            "description": "Below normal body weight (< 18.5). May indicate nutritional deficit.",
            "risk_multiplier": "Low to Moderate"
        }
    elif 18.5 <= bmi < 25.0:
        return {
            "category": "Normal weight",
            "badge_class": "badge-teal",
            "color": "#10B981",
            "description": "Healthy standard range (18.5 - 24.9). Associated with lowest health risk.",
            "risk_multiplier": "Baseline (Optimal)"
        }
    elif 25.0 <= bmi < 30.0:
        return {
            "category": "Overweight",
            "badge_class": "badge-amber",
            "color": "#F59E0B",
            "description": "Elevated body weight (25.0 - 29.9). Mildly elevated actuarial risk factor.",
            "risk_multiplier": "Moderate"
        }
    else:
        return {
            "category": "Obese",
            "badge_class": "badge-rose",
            "color": "#EF4444",
            "description": "High body mass index (>= 30.0). High risk tier, especially when paired with smoking.",
            "risk_multiplier": "High Risk Surcharge"
        }


def convert_charge(amount: float, currency: str = "USD", pricing_mode: str = "indian_market") -> float:
    """
    Converts model raw dollar prediction into the target currency and market scale.
    - USD: Returns raw USD amount ($2,000 - $40,000).
    - INR with 'indian_market' (default): Calibrated to realistic Indian healthcare insurance
      premium standards (Purchasing Power Parity actuarial scale: ~6.5x factor).
      A 35yo non-smoker with 2 kids costs approx ₹35,000 - ₹42,000/year (₹2,800 - ₹3,500/mo),
      matching real-life middle-class family floater policies (Star Health, HDFC ERGO, Care Insurance).
    - INR with 'direct_forex': Direct currency conversion at nominal exchange rate (83.5x).
    """
    if currency == "INR":
        if pricing_mode == "direct_forex":
            return amount * 83.5
        # Realistic Indian healthcare market parity (~6.5x factor)
        return amount * 6.5
    return amount


def format_currency(amount: float, currency: str = "USD", pricing_mode: str = "indian_market") -> str:
    """
    Formats predicted insurance charge into USD ($) or INR (₹) with appropriate market calibration.
    """
    converted = convert_charge(amount, currency=currency, pricing_mode=pricing_mode)
    if currency == "INR":
        return f"₹{converted:,.2f}"
    return f"${converted:,.2f}"


def load_saved_model():
    """
    Loads serialized machine learning pipeline from disk.
    If missing, trains the model automatically.
    """
    model_path = get_model_path()
    if not os.path.exists(model_path):
        logger.warning("Saved model not found at %s. Triggering training pipeline...", model_path)
        base_dir = get_base_dir()
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)
        try:
            from src.train_model import train_and_save_pipeline
        except ModuleNotFoundError:
            from train_model import train_and_save_pipeline
        train_and_save_pipeline()

    try:
        pipeline = joblib.load(model_path)
        logger.info("Successfully loaded model pipeline from %s", model_path)
        return pipeline
    except Exception as e:
        logger.error("Failed loading model pipeline: %s", e)
        raise RuntimeError(f"Unable to load saved model from {model_path}: {e}") from e


def load_model_metrics() -> dict:
    """
    Loads saved evaluation metrics for Linear Regression, Random Forest, and Gradient Boosting.
    """
    metrics_path = get_metrics_path()
    if not os.path.exists(metrics_path):
        base_dir = get_base_dir()
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)
        try:
            from src.train_model import train_and_save_pipeline
        except ModuleNotFoundError:
            from train_model import train_and_save_pipeline
        train_and_save_pipeline()

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed loading model metrics: %s", e)
        return {}
