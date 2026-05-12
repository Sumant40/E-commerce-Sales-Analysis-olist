import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# These are the features fed into ML models
NUMERIC_FEATURES = [
    'log_recency',
    'log_frequency',
    'log_monetary',
    'log_avg_order_value',
    'category_diversity',
    'avg_review_score',
    'review_count',
    'is_reviewer',
    'avg_delivery_days',
    'avg_delivery_delta',
    'late_deliveries',
    'avg_installments',
    'max_installments',
    'high_installment_flag',
    'weekend_purchase_rate',
    'freight_to_revenue_ratio',
    'tenure_days',
]

CATEGORICAL_FEATURES = [
    'preferred_payment',
    'favourite_category',
]

def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value',
                                   unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipe,      NUMERIC_FEATURES),
            ('cat', categorical_pipe,  CATEGORICAL_FEATURES),
        ],
        remainder='drop'
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


if __name__ == "__main__":
    import os

    # Resolve path relative to project root regardless of where script is run from
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    features_path = os.path.join(project_root, "data", "processed", "customer_features.csv")

    assert os.path.exists(features_path), f"Feature store not found at {features_path}"

    features = pd.read_csv(features_path)
    print(f"Loaded feature store: {features.shape}")

    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(features[NUMERIC_FEATURES + CATEGORICAL_FEATURES])

    print(f"Input shape:  {features[NUMERIC_FEATURES + CATEGORICAL_FEATURES].shape}")
    print(f"Output shape: {X.shape}")
    print(f"Mean of scaled numerics (should be ~0): {X[:, :len(NUMERIC_FEATURES)].mean(axis=0).round(3)}")
    print("Pipeline built successfully.")