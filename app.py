"""Streamlit application for baseline-versus-production drift monitoring.

Run with: streamlit run app.py
"""
from __future__ import annotations

import io
import json
import logging
import os
import smtplib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from ml_utils import infer_task_type

LOGGER = logging.getLogger(__name__)
PREDICTION_COLUMN = "prediction"
MISSING = "<MISSING>"


@dataclass
class DriftResult:
    feature: str
    feature_type: str
    test: str
    statistic: Optional[float]
    p_value: Optional[float]
    drifted: bool
    severity: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return {key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                for key, value in result.items()}


def init_state() -> None:
    """Initialise only stable state; widget values are managed by Streamlit."""
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("analysis_key", None)
    st.session_state.setdefault("email_status", None)


def clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise headers and make duplicated/blank headers unambiguous."""
    frame = frame.copy()
    names: list[str] = []
    seen: dict[str, int] = {}
    for index, name in enumerate(frame.columns):
        base = str(name).strip() or f"unnamed_{index + 1}"
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else f"{base}__{seen[base]}")
    frame.columns = names
    return frame


def read_csv(uploaded: Any) -> pd.DataFrame:
    """Read an uploaded CSV with conservative, useful validation errors."""
    try:
        raw = uploaded.getvalue()
        if not raw:
            raise ValueError("The uploaded file is empty.")
        frame = pd.read_csv(io.BytesIO(raw), low_memory=False)
    except UnicodeDecodeError:
        frame = pd.read_csv(io.BytesIO(uploaded.getvalue()), encoding="latin-1", low_memory=False)
    except Exception as exc:
        raise ValueError(f"Could not read this CSV: {exc}") from exc
    if frame.empty:
        raise ValueError("The CSV has headers but no data rows.")
    return clean_column_names(frame)


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def usable_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)


def categorical_counts(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING).value_counts(dropna=False)


def severity_from_p(p_value: Optional[float], alpha: float) -> str:
    if p_value is None or not np.isfinite(p_value) or p_value >= alpha:
        return "None"
    if p_value < alpha / 100:
        return "Critical"
    if p_value < alpha / 10:
        return "High"
    return "Moderate"


def compare_numeric(feature: str, baseline: pd.Series, production: pd.Series, alpha: float) -> DriftResult:
    left, right = usable_numeric(baseline), usable_numeric(production)
    if len(left) < 2 or len(right) < 2:
        return DriftResult(feature, "numeric", "KS two-sample", None, None, False, "None",
                           "Fewer than two valid numeric values in one dataset.")
    if np.all(left == left[0]) and np.all(right == right[0]):
        changed = not np.isclose(left[0], right[0], equal_nan=True)
        return DriftResult(feature, "numeric", "constant comparison", None, 0.0 if changed else 1.0,
                           changed, "Critical" if changed else "None",
                           "Both datasets contain a constant value.")
    statistic, p_value = ks_2samp(left, right, method="auto")
    return DriftResult(feature, "numeric", "KS two-sample", float(statistic), float(p_value),
                       bool(p_value < alpha), severity_from_p(float(p_value), alpha))


def compare_categorical(feature: str, baseline: pd.Series, production: pd.Series, alpha: float) -> DriftResult:
    left, right = categorical_counts(baseline), categorical_counts(production)
    categories = left.index.union(right.index)
    table = np.array([left.reindex(categories, fill_value=0), right.reindex(categories, fill_value=0)])
    # chi2_contingency rejects a table containing expected zero frequencies; remove
    # categories with no observations in either row (normally impossible, but defensive).
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return DriftResult(feature, "categorical", "chi-square", None, 1.0, False, "None",
                           "Only one observed category; distribution drift is not testable.")
    try:
        statistic, p_value, _, _ = chi2_contingency(table, correction=False)
    except ValueError as exc:
        return DriftResult(feature, "categorical", "chi-square", None, None, False, "None",
                           f"Test not applicable: {exc}")
    return DriftResult(feature, "categorical", "chi-square", float(statistic), float(p_value),
                       bool(p_value < alpha), severity_from_p(float(p_value), alpha))


def detect_drift(baseline: pd.DataFrame, production: pd.DataFrame, alpha: float) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    for feature in sorted(set(baseline.columns) & set(production.columns)):
        result = (compare_numeric(feature, baseline[feature], production[feature], alpha)
                  if is_numeric(baseline[feature]) and is_numeric(production[feature])
                  else compare_categorical(feature, baseline[feature], production[feature], alpha))
        results.append(result.to_dict())
    return pd.DataFrame(results)


def target_candidates(frame: pd.DataFrame) -> list[str]:
    common = ["target", "label", "outcome", "y", "class", "response", "actual"]
    preferred = [column for column in frame.columns if column.lower().strip() in common]
    remaining = [column for column in frame.columns if column not in preferred]
    return preferred + remaining


def valid_target(series: pd.Series, task: str) -> tuple[bool, str]:
    values = series.dropna()
    if values.empty:
        return False, "Target has no non-missing values."
    if task == "Classification" and values.nunique() < 2:
        return False, "Classification needs at least two target classes."
    if task == "Regression":
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any() or len(numeric) < 3:
            return False, "Regression target must contain at least three numeric values."
        if numeric.nunique() < 2:
            return False, "Regression target is constant."
    return True, ""


def build_pipeline(features: pd.DataFrame, task: str) -> Pipeline:
    numeric = [column for column in features.columns if is_numeric(features[column])]
    categorical = [column for column in features.columns if column not in numeric]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    if not transformers:
        raise ValueError("No usable feature columns remain after removing the target and prediction columns.")
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = (RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced_subsample")
             if task == "Classification" else RandomForestRegressor(n_estimators=250, random_state=42))
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def evaluate_predictions(actual: pd.Series, predicted: pd.Series, task: str) -> dict[str, float]:
    if len(actual) != len(predicted) or len(actual) == 0:
        raise ValueError("Actual and predicted values must be non-empty and have the same length.")
    if task == "Classification":
        actual, predicted = actual.astype(str), predicted.astype(str)
        return {"accuracy": float(accuracy_score(actual, predicted)),
                "precision_weighted": float(precision_score(actual, predicted, average="weighted", zero_division=0)),
                "recall_weighted": float(recall_score(actual, predicted, average="weighted", zero_division=0)),
                "f1_weighted": float(f1_score(actual, predicted, average="weighted", zero_division=0))}
    actual = pd.to_numeric(actual, errors="coerce")
    predicted = pd.to_numeric(predicted, errors="coerce")
    valid = actual.notna() & predicted.notna() & np.isfinite(actual) & np.isfinite(predicted)
    if valid.sum() < 2:
        raise ValueError("Regression evaluation needs at least two finite actual/prediction pairs.")
    actual, predicted = actual[valid], predicted[valid]
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted))}


def train_and_evaluate(baseline: pd.DataFrame, production: pd.DataFrame, target: str, task: str) -> tuple[dict[str, float], dict[str, float], Pipeline]:
    if target not in production.columns:
        raise ValueError("Production data does not contain the selected target column.")
    baseline = baseline.dropna(subset=[target]).copy()
    production = production.dropna(subset=[target]).copy()
    exclude = {target, PREDICTION_COLUMN}
    features = [column for column in baseline.columns if column not in exclude and column in production.columns]
    if not features:
        raise ValueError("No common feature columns are available for model monitoring.")
    valid, message = valid_target(baseline[target], task)
    if not valid:
        raise ValueError(message)
    X, y = baseline[features], baseline[target]
    stratify = y if task == "Classification" and y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    if len(X_train) < 2 or len(X_test) < 1:
        raise ValueError("Baseline has too few valid rows for a train/test split.")
    pipeline = build_pipeline(X_train, task)
    pipeline.fit(X_train, y_train)
    validation = evaluate_predictions(y_test.reset_index(drop=True), pd.Series(pipeline.predict(X_test)), task)
    production_metrics = evaluate_predictions(production[target].reset_index(drop=True),
                                             pd.Series(pipeline.predict(production[features])), task)
    return validation, production_metrics, pipeline


def retrain_model(production: pd.DataFrame, target: str, task: str) -> tuple[Pipeline, dict[str, float]]:
    dataset = production.dropna(subset=[target]).copy()
    valid, message = valid_target(dataset[target], task)
    if not valid:
        raise ValueError(f"Cannot retrain: {message}")
    features = [column for column in dataset.columns if column not in {target, PREDICTION_COLUMN}]
    pipeline = build_pipeline(dataset[features], task)
    pipeline.fit(dataset[features], dataset[target])
    # Training-set metrics are explicitly labelled so they are never mistaken for holdout performance.
    metrics = evaluate_predictions(dataset[target].reset_index(drop=True), pd.Series(pipeline.predict(dataset[features])), task)
    return pipeline, metrics


def make_audit(baseline: pd.DataFrame, production: pd.DataFrame, results: pd.DataFrame,
               target: Optional[str], task: str, alpha: float, metrics: Optional[dict[str, float]]) -> dict[str, Any]:
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "alpha": alpha, "task_type": task,
            "target_column": target, "baseline_rows": len(baseline), "production_rows": len(production),
            "baseline_only_columns": sorted(set(baseline.columns) - set(production.columns)),
            "production_only_columns": sorted(set(production.columns) - set(baseline.columns)),
            "drifted_features": int(results["drifted"].sum()) if not results.empty else 0,
            "features_compared": len(results), "production_metrics": metrics,
            "feature_results": results.replace({np.nan: None}).to_dict(orient="records")}


def send_email(recipient: str, audit: dict[str, Any], sender: str, password: str) -> None:
    if not recipient or not sender or not password:
        raise ValueError("Recipient, Gmail address, and Gmail app password are all required.")
    message = EmailMessage()
    message["Subject"] = "Data drift monitoring alert"
    message["From"] = sender
    message["To"] = recipient
    message.set_content("Your data drift monitoring audit is attached.\n\n" + json.dumps(audit, indent=2, default=str))
    payload = json.dumps(audit, indent=2, default=str).encode("utf-8")
    message.add_attachment(payload, maintype="application", subtype="json", filename="drift_audit.json")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
        server.login(sender, password)
        server.send_message(message)


def render() -> None:
    st.set_page_config(page_title="Data Drift Monitor", layout="wide")
    init_state()
    st.title("Data Drift Monitor")
    st.caption("Compare baseline and production CSVs, monitor model performance, retrain safely, and export an audit.")
    with st.sidebar:
        st.header("Settings")
        alpha = st.slider("Drift significance threshold (alpha)", 0.001, 0.20, 0.05, 0.001)
        st.divider()
        st.caption("Gmail alerts use an app password. Credentials are not stored in session state or audit files.")
        recipient = st.text_input("Alert recipient")
        sender = st.text_input("Gmail address", value=os.getenv("DRIFT_ALERT_SENDER", ""))
        password = st.text_input("Gmail app password", value=os.getenv("DRIFT_ALERT_APP_PASSWORD", ""), type="password")

    left, right = st.columns(2)
    baseline_upload = left.file_uploader("Baseline CSV", type="csv", key="baseline_upload")
    production_upload = right.file_uploader("Production CSV", type="csv", key="production_upload")
    if not baseline_upload or not production_upload:
        st.info("Upload both CSV files to begin.")
        return
    try:
        baseline, production = read_csv(baseline_upload), read_csv(production_upload)
    except ValueError as exc:
        st.error(str(exc))
        return
    common = sorted(set(baseline.columns) & set(production.columns))
    if not common:
        st.error("The files have no columns in common after header cleanup.")
        return
    st.caption(f"Baseline: {len(baseline):,} rows · Production: {len(production):,} rows · {len(common)} shared columns")
    target_options = ["No target / drift only"] + target_candidates(baseline[common])
    target_choice = st.selectbox("Target column (optional, required for ML monitoring)", target_options)
    target = None if target_choice == "No target / drift only" else target_choice
    task = infer_task_type(baseline[target]) if target else "Classification"
    analysis_key = (baseline_upload.name, baseline_upload.size, production_upload.name, production_upload.size, alpha, task, target)
    if st.button("Run analysis", type="primary"):
        with st.spinner("Testing feature distributions and validating model performance..."):
            try:
                excluded = [target] if target else []
                drift = detect_drift(baseline.drop(columns=excluded, errors="ignore"), production.drop(columns=excluded, errors="ignore"), alpha)
                validation = production_metrics = None
                model = None
                warning = None
                if target:
                    validation, production_metrics, model = train_and_evaluate(baseline, production, target, task)
                audit = make_audit(baseline, production, drift, target, task, alpha, production_metrics)
                st.session_state.analysis = {"drift": drift, "validation": validation, "production": production_metrics,
                                             "model": model, "audit": audit, "retrain": None, "warning": warning}
                st.session_state.analysis_key = analysis_key
                st.session_state.email_status = None
            except Exception as exc:
                LOGGER.exception("Analysis failed")
                st.error(f"Analysis could not be completed: {exc}")
                return
    analysis = st.session_state.analysis if st.session_state.analysis_key == analysis_key else None
    if not analysis:
        st.info("Choose the settings above and run the analysis.")
        return
    drift = analysis["drift"]
    drifted = int(drift["drifted"].sum()) if not drift.empty else 0
    a, b, c = st.columns(3)
    a.metric("Features compared", len(drift))
    b.metric("Drifted features", drifted)
    c.metric("Drift rate", f"{(100 * drifted / len(drift)) if len(drift) else 0:.1f}%")
    st.subheader("Feature drift")
    display = drift.copy()
    if not display.empty:
        display["p_value"] = display["p_value"].map(lambda value: f"{value:.4g}" if pd.notna(value) else "—")
        st.dataframe(display, use_container_width=True, hide_index=True)
        chart = drift.dropna(subset=["p_value"]).copy()
        if not chart.empty:
            chart["significance"] = -np.log10(chart["p_value"].clip(lower=np.finfo(float).tiny))
            st.plotly_chart(px.bar(chart.sort_values("significance", ascending=False), x="feature", y="significance",
                                  color="severity", title="Distribution-shift significance (−log10 p)"), use_container_width=True)
    else:
        st.warning("No comparable feature columns were available after excluding the target.")
    if analysis["validation"]:
        st.subheader("Model monitoring")
        metrics = pd.DataFrame([analysis["validation"], analysis["production"]], index=["Baseline holdout", "Production"])
        st.dataframe(metrics.style.format("{:.4f}"), use_container_width=True)
        with st.expander("Retrain on production data"):
            st.warning("Retraining uses production labels and overwrites no files. Review the resulting model outside this app before deployment.")
            if st.button("Retrain model", key="retrain"):
                try:
                    _, retrain_metrics = retrain_model(production, target, task)  # model stays in memory only
                    analysis["retrain"] = retrain_metrics
                    st.success("Model retrained successfully. Metrics below are in-sample, not validation metrics.")
                except Exception as exc:
                    LOGGER.exception("Retraining failed")
                    st.error(str(exc))
            if analysis["retrain"]:
                st.dataframe(pd.DataFrame([analysis["retrain"]], index=["Production training set"]).style.format("{:.4f}"))
    st.subheader("Audit and alert")
    audit_bytes = json.dumps(analysis["audit"], indent=2, default=str).encode("utf-8")
    x, y = st.columns(2)
    x.download_button("Download audit JSON", audit_bytes, "drift_audit.json", "application/json")
    y.download_button("Download feature results CSV", drift.to_csv(index=False).encode("utf-8"), "feature_drift_results.csv", "text/csv")
    if st.button("Send Gmail alert"):
        try:
            send_email(recipient, analysis["audit"], sender, password)
            st.session_state.email_status = ("success", f"Audit sent to {recipient}.")
        except Exception as exc:
            LOGGER.exception("Email alert failed")
            st.session_state.email_status = ("error", f"Email was not sent: {exc}")
    if st.session_state.email_status:
        level, text = st.session_state.email_status
        getattr(st, level)(text)


if __name__ == "__main__":
    render()
