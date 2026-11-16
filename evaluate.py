"""Compare exoplanet disposition classifiers on one grouped Kepler holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    make_scorer,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "koi_period", "koi_impact", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr", "koi_steff", "koi_slogg",
    "koi_srad", "koi_kepmag",
]
TAP_ENDPOINT = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
COLUMNS = ["kepid", "koi_disposition", *FEATURES]
QUERY = f"select kepid,kepoi_name,koi_disposition,{','.join(FEATURES)} from cumulative order by kepoi_name"
LABELS = {"FALSE POSITIVE": 0, "CONFIRMED": 1}
SEED = 42


def download_data(path: Path) -> None:
    query = urlencode({"query": QUERY, "format": "csv"})
    with urlopen(f"{TAP_ENDPOINT}?{query}", timeout=60) as response:
        data = response.read()
    header = data.splitlines()[0].decode("utf-8-sig").split(",") if data else []
    if not set(COLUMNS).issubset(header):
        raise ValueError("NASA's archive did not return a KOI CSV with the expected columns")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".csv.tmp")
    try:
        temporary_path.write_bytes(data)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_data(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, comment="#")
    if missing := set(COLUMNS) - set(frame.columns):
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
    frame = frame.loc[frame["koi_disposition"].isin(LABELS), COLUMNS].dropna(subset=["kepid"])
    target = frame["koi_disposition"].map(LABELS).to_numpy()
    groups = frame["kepid"].to_numpy()
    if frame.empty or frame.groupby("koi_disposition")["kepid"].nunique().min() < 6 or len(np.unique(target)) != 2:
        raise ValueError("Each disposition needs at least six distinct Kepler stars")
    features = frame[FEATURES].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    return features, target, groups


def held_out_split(features, labels: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    return next(folds.split(features, labels, groups))


def evaluate(features, labels: np.ndarray, groups: np.ndarray, output_dir: Path) -> dict:
    train, test = held_out_split(features, labels, groups)
    x_train, x_test = features.iloc[train], features.iloc[test]
    y_train, y_test = labels[train], labels[test]
    if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
        raise ValueError("The grouped holdout requires both dispositions in train and test")

    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
        x_train, y_train, groups[train],
    ))
    max_neighbors = min(len(fit) for fit, _ in folds)
    neighbors = [k for k in (3, 7, 15, 25) if k <= max_neighbors] or [1]
    tuning = GridSearchCV(
        estimator=make_pipeline(SimpleImputer(), StandardScaler(), KNeighborsClassifier()),
        param_grid={"kneighborsclassifier__n_neighbors": neighbors, "kneighborsclassifier__p": [1, 2]},
        scoring=make_scorer(f1_score, average="macro", labels=[0, 1], zero_division=0),
        cv=folds,
        error_score="raise",
    )
    tuning.fit(x_train, y_train)
    models = {
        "Most frequent": DummyClassifier(strategy="most_frequent").fit(x_train, y_train),
        "Logistic regression": make_pipeline(
            SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED),
        ).fit(x_train, y_train),
        "K-nearest neighbors": tuning.best_estimator_,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison, axes = plt.subplots(1, len(models), figsize=(11, 3.5), layout="constrained")
    roc, roc_axis = plt.subplots(figsize=(6, 4.5), layout="constrained")
    scores = {}
    for (name, model), axis in zip(models.items(), axes, strict=True):
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        scores[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }
        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions, display_labels=["False positive", "Confirmed"], cmap="Blues",
            colorbar=False, ax=axis,
        )
        axis.set_title(name)
        RocCurveDisplay.from_predictions(y_test, probabilities, name=name, ax=roc_axis)
    roc_axis.plot([0, 1], [0, 1], "--", color="0.55", label="Chance")
    roc_axis.set_title("Kepler disposition on held-out stars")
    roc_axis.legend(loc="lower right")
    comparison.savefig(output_dir / "confusion_matrices.png", dpi=140)
    roc.savefig(output_dir / "roc.png", dpi=140)
    plt.close(comparison)
    plt.close(roc)

    return {
        "confirmed": int((labels == 1).sum()),
        "false_positive": int((labels == 0).sum()),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_stars": len(np.unique(groups[train])),
        "test_stars": len(np.unique(groups[test])),
        "knn_parameters": tuning.best_params_,
        "knn_train_cv_macro_f1": float(tuning.best_score_),
        "holdout": scores,
    }


def explore_clusters(features, labels: np.ndarray, groups: np.ndarray) -> dict[int, float]:
    train, _ = held_out_split(features, labels, groups)
    frame = features.iloc[train]
    # Limit the influence of extreme estimates in this exploratory distance calculation.
    clipped = frame.clip(frame.quantile(0.01), frame.quantile(0.99), axis="columns")
    sample = clipped.sample(min(1500, len(clipped)), random_state=SEED)
    observations = make_pipeline(SimpleImputer(), StandardScaler()).fit_transform(sample)
    scores = {}
    for count in range(2, min(7, len(sample))):
        assignments = KMeans(n_clusters=count, n_init=10, random_state=SEED).fit_predict(observations)
        if 1 < len(np.unique(assignments)) < len(sample):
            scores[count] = float(silhouette_score(observations, assignments))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/kepler-koi.csv"))
    parser.add_argument("--download", action="store_true", help="Download the public KOI table if the CSV is absent")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--cluster", action="store_true", help="Include the optional training-only silhouette analysis")
    args = parser.parse_args()
    try:
        if args.download and not args.data.exists():
            download_data(args.data)
        features, labels, groups = load_data(args.data)
        report = evaluate(features, labels, groups, args.output)
        report["dataset_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
        if args.cluster:
            report["train_silhouette"] = explore_clusters(features, labels, groups)
        summary = json.dumps(report, indent=2)
        (args.output / "metrics.json").write_text(summary + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(summary)


if __name__ == "__main__":
    main()
