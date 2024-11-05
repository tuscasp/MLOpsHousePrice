import argparse

import numpy as np

import json
import joblib
from pathlib import Path

from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error)


from data_loader import CsvLoader


def build_model():
    categorical_transformer = TargetEncoder()

    categorical_cols = ["type", "sector"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical',
            categorical_transformer,
            categorical_cols)
        ])
    
    model = GradientBoostingRegressor(**{
        "learning_rate":0.01,
        "n_estimators":300,
        "max_depth":5,
        "loss":"absolute_error"
    })

    steps = [
        ('preprocessor', preprocessor),
        ('model', model)
    ]

    pipeline = Pipeline(steps)

    return pipeline


def compute_metrics(predictions, target):
    ret_dir = {
        "RMSE" : np.sqrt(mean_squared_error(predictions, target)),
        "MAPE" : mean_absolute_percentage_error(predictions, target),
        "MAE" : mean_absolute_error(predictions, target)
    }

    return ret_dir


def save_model(model: Pipeline, filepath: Path):
    joblib.dump(model, path_model, compress=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", help="Data for training. Only path to .csv files is currently supported.", type=str)
    parser.add_argument("-t", "--test", help="Data for validating. Only path to .csv files is currently supported.", type=str)
    parser.add_argument("-o", "--output", help="Path to model output directory.")

    args = parser.parse_args()

    target_column = "price"
    data_loader_train = CsvLoader(args.input, target_column)
    X_train, Y_train = data_loader_train.load()

    data_loader_validation = CsvLoader(args.test, target_column)
    X_validation, Y_validation = data_loader_validation.load()

    model = build_model()

    model.fit(X_train, Y_train)

    test_predictions = model.predict(X_validation)

    metrics = compute_metrics(test_predictions, Y_validation.values)

    dir_models = Path(args.output)

    path_results_json = dir_models / "model_metrics.json"
    with open(path_results_json, "w") as outfile:
        json.dump(metrics, outfile, indent=4, sort_keys=False)
    
    path_model = dir_models / "trained_model.pkl"
    save_model(model, path_model)

