"""Training orchestrator: load → split → fit → eval → log → serialize.

Run from repo root with the project venv active:

    python -m rebooking.models.train
    python -m rebooking.models.train --config configs/training.yaml

Reads hyperparameters from a YAML config, parsed into a Pydantic settings
model so typos in keys raise a clear ValidationError at load time. Logs
params, metrics, and the fitted model to MLflow. Serializes the
(transformer, model) pair to artifacts/model.joblib for serving.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Literal

import joblib
import mlflow
import mlflow.sklearn
import yaml
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from rebooking.data.loader import load_bookings
from rebooking.features.transform import FeatureTransformer

ARTIFACT_PATH = Path("artifacts/model.joblib")
DEFAULT_CONFIG = Path("configs/training.yaml")


class DataConfig(BaseModel):
    raw_csv: Path
    test_size: float = Field(gt=0.0, lt=1.0)
    random_state: int


class ModelConfig(BaseModel):
    type: Literal["logistic_regression"]
    C: float = Field(gt=0.0)
    max_iter: int = Field(gt=0)
    class_weight: Literal["balanced"] | None = None


class MLflowConfig(BaseModel):
    experiment_name: str
    tracking_uri: str
    log_model: bool = True


class TrainingConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    mlflow: MLflowConfig


def load_config(path: Path) -> TrainingConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return TrainingConfig.model_validate(raw)


def _git_sha() -> str:
    """Return short git SHA of HEAD; 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def train(cfg: TrainingConfig) -> None:
    df = load_bookings(cfg.data.raw_csv).reset_index(drop=True)
    X = df.drop(columns=["customer_id", "rebooked"])
    y = df["rebooked"]

    transformer = FeatureTransformer()
    X_full = transformer.fit_transform(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
        stratify=y,
    )

    model = LogisticRegression(
        C=cfg.model.C,
        max_iter=cfg.model.max_iter,
        class_weight=cfg.model.class_weight,
        random_state=cfg.data.random_state,
    )
    model.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run():
        mlflow.log_params(cfg.model.model_dump(exclude={"type"}))
        mlflow.log_param("test_size", cfg.data.test_size)
        mlflow.log_metric("train_auc", train_auc)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("pos_rate", float(y.mean()))
        mlflow.set_tag("git_sha", _git_sha())
        mlflow.set_tag("dataset_rows", str(len(df)))
        mlflow.set_tag("model_type", cfg.model.type)
        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(model, name="model")

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"transformer": transformer, "model": model}, ARTIFACT_PATH)

    print(f"Train AUC: {train_auc:.3f}")
    print(f"Test  AUC: {test_auc:.3f}")
    print(f"Artifact: {ARTIFACT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
