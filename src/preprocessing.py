"""
src/preprocessing.py
Preprocessing pipelines and feature transformations for Medical Insurance Cost Predictor.
Ensures zero data leakage by encapsulating transformers inside Scikit-learn Pipelines.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Feature Definitions
NUMERICAL_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
TARGET_FEATURE = "charges"

VALID_SEX = ["male", "female"]
VALID_SMOKER = ["yes", "no"]
VALID_REGIONS = ["northeast", "northwest", "southeast", "southwest"]


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that handles:
    - Numerical features (age, bmi, children): StandardScaler for optimal gradient convergence
    - Categorical features (sex, smoker, region): OneHotEncoder (handle_unknown='ignore')
    """
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    return preprocessor


def create_full_pipeline(regressor_estimator) -> Pipeline:
    """
    Wraps preprocessor and a given regression model estimator into a single Scikit-learn Pipeline.
    Prevents data leakage during cross-validation and test set evaluation.
    """
    preprocessor = build_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor_estimator),
        ]
    )
    return pipeline


def validate_inputs(
    age: int,
    bmi: float,
    children: int,
    sex: str,
    smoker: str,
    region: str
) -> dict:
    """
    Validates user input values against expected domain ranges and allowed categories.
    Returns a dictionary with 'is_valid' boolean and any validation 'error' message.
    """
    if not (18 <= age <= 100):
        return {"is_valid": False, "error": f"Age must be between 18 and 100. Received: {age}"}

    if not (10.0 <= bmi <= 65.0):
        return {"is_valid": False, "error": f"BMI must be between 10.0 and 65.0. Received: {bmi}"}

    if not (0 <= children <= 15):
        return {"is_valid": False, "error": f"Children count must be between 0 and 15. Received: {children}"}

    sex_clean = str(sex).strip().lower()
    if sex_clean not in VALID_SEX:
        return {"is_valid": False, "error": f"Sex must be one of {VALID_SEX}. Received: {sex}"}

    smoker_clean = str(smoker).strip().lower()
    if smoker_clean not in VALID_SMOKER:
        return {"is_valid": False, "error": f"Smoker must be one of {VALID_SMOKER}. Received: {smoker}"}

    region_clean = str(region).strip().lower()
    if region_clean not in VALID_REGIONS:
        return {"is_valid": False, "error": f"Region must be one of {VALID_REGIONS}. Received: {region}"}

    return {"is_valid": True, "error": None}


def format_input_dataframe(
    age: int,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str
) -> pd.DataFrame:
    """
    Normalizes inputs and bundles them into a single-row Pandas DataFrame matching training schema.
    """
    validation = validate_inputs(age, bmi, children, sex, smoker, region)
    if not validation["is_valid"]:
        raise ValueError(validation["error"])

    data = {
        "age": [int(age)],
        "sex": [str(sex).strip().lower()],
        "bmi": [float(bmi)],
        "children": [int(children)],
        "smoker": [str(smoker).strip().lower()],
        "region": [str(region).strip().lower()],
    }

    df = pd.DataFrame(data)
    # Ensure correct column ordering matching dataset
    return df[["age", "sex", "bmi", "children", "smoker", "region"]]
