"""
Valid metrics
__author__: Abhishek Thakur
"""


metrics = {
    "binary_classification": [
        "accuracy",
        "auc",
        "f1",
        "logloss",
        "precision",
        "recall",
    ],
    "multiclass_classification": [
        "accuracy",
        "f1",
        "multiclass_logloss",
        "precision",
        "recall",
        "quadratic_kappa",
    ],
    "multilabel_classification": ["multiclass_logloss"],
    "single_col_regression": ["mae", "msle", "mse", "rmsle", "rmse", "r2"],
    "multi_col_regression": ["mae", "msle", "mse", "rmsle", "rmse", "r2"],
}
