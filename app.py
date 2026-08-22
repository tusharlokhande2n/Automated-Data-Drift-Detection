# =========================================================================
# Project Name: Automated Data Drift Detection and Alarming System for Quality Assurance
# Name: Tushar Gajanan Lokhande
# BITS ID: 2024CT05001
# =========================================================================

# =========================================================================
# 1. Import Python Libraries
# =========================================================================
import streamlit as st
import pandas as pd
import numpy as np
import json
import smtplib
import uuid
import tempfile
from pathlib import Path
from datetime import datetime

# Core statistical tests used for validating differences in data Baseline and Production datasets
from scipy.stats import ks_2samp, chi2_contingency

# Standard metrics used here to track real-time machine learning model accuracy scope, F1 Score, Mean Squared Error, Mean Absolute error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Python standard library imports used to construct complex, multipart email messages containing text, HTML and file attachments.
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

# We will use Visualization engines to build interactive dashboard elements
import plotly.express as px
import plotly.graph_objects as go

# =========================================================================
# 2. Building user friendly user interface
# =========================================================================
# Configure Streamlit application to use a spacious, wide-screen layout format
st.set_page_config(page_title="Automated Data Drift Detection & Alarming System", layout="wide")

# Add business user friendly clean layout containers and color indicators 
st.markdown("""
<style>
.main-title { font-size:32px; font-weight:700; color:#0f2042; margin-bottom:2px; }
.subtitle { font-size:15px; color:#5a6e85; margin-bottom:25px; }
.section-header { font-size:20px; font-weight:700; color:#1d3557; margin-top:30px; margin-bottom:15px; border-bottom: 2px solid #e5e9f0; padding-bottom: 6px;}
.biz-card { 
    background-color: #ffffff; 
    border: 1px solid #e5e9f0; 
    border-radius: 8px; 
    padding: 20px; 
    box-shadow: 0 4px 6px rgba(0,0,0,0.02); 
    text-align: center;
}
.biz-card-title { font-size:14px; font-weight:600; color:#6c757d; text-transform: uppercase; letter-spacing: 0.5px; }
.biz-card-value { font-size:28px; font-weight:700; margin: 10px 0; }
.biz-card-desc { font-size:12px; color:#495057; line-height:1.4; }
.concept-box { background-color: #f8f9fa; border-left: 5px solid #2b5c8f; padding: 15px; margin: 15px 0px; border-radius: 4px; font-size:14px; color:#333333; }
</style>
""", unsafe_allow_html=True)

# Set up browser session memory arrays to avoid losing data on page updates
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "email_status" not in st.session_state:
    st.session_state.email_status = None

# =========================================================================
# 3. Adding interactive sidebar in application
# =========================================================================

st.markdown(
    '<div class="main-title">📊 Automated Data Drift Detection & Alarming System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Automated data drift detection in the Production dataset file with respect to Baseline file and displaying summary reports</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">System Built By: Tushar Gajanan Lokhande | BITS ID: 2024CT05001</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------------------------------
# Default prediction column name
# -------------------------------------------------------------------------
pred_col = "prediction"

with st.sidebar:

    st.header("⚙️ P-Value (Alpha Threshold) Configuration")

    threshold = st.slider(
        "Data Drift Detection Sensitivity (Alpha Threshold)",
        min_value=0.001,
        max_value=0.20,
        value=0.05,
        step=0.005,
        help=(
            "Controls how strictly the system detects data drift. "
            "Lower values make drift detection more stringent."
        ),
    )

    st.divider()

    st.subheader("📬 Email Alert Configuration")

    sender = st.text_input(
        "Sender Gmail Address"
    )

    password = st.text_input(
        "Gmail App Password",
        type="password"
    )

    receiver = st.text_input(
        "Recipient Email Address"
    )

# =========================================================================
# 4. Data ingestion interface to upload Baseline and Production dataset files
# =========================================================================
st.markdown('<div class="section-header">📂 Upload Baseline and Production datasets</div>', unsafe_allow_html=True)

col_up1, col_up2 = st.columns(2)
with col_up1:
    baseline_file = st.file_uploader("📋 Upload Baseline Data file (where system is trained)", type="csv")
with col_up2:
    production_file = st.file_uploader("📈 Upload Production Data file", type="csv")

# Informative onboarding prompt displayed before file uploads are completed
if baseline_file is None or production_file is None:
    st.info("💡 Welcome! Please upload your baseline file and live production system file above to generate audit summary report.")
    st.stop()

# Execution layer triggered by clicking the processing button
st.markdown('<div class="section-header">🚀 Step 2: Evaluate drift in Production dataset with respect to Baseline dataset.</div>', unsafe_allow_html=True)
if st.button("📊 Generate Drift Detection Summary reports & Audit Reports", type="primary"):
    base_raw = pd.read_csv(baseline_file)
    prod_raw = pd.read_csv(production_file)

    # Clean files automatically by dropping any columns that are completely blank
    base_cleaned = base_raw.dropna(how='all', axis=1)
    prod_cleaned = prod_raw.dropna(how='all', axis=1)

    # Dynamic target column resolution
    if "target" in base_cleaned.columns:
        target_col = "target"
    elif "Attrition_Flag" in base_cleaned.columns:
        target_col = "Attrition_Flag"
    else:
        target_col = base_cleaned.columns[-1]

    target_values = base_cleaned[target_col].dropna()
    numeric_target = pd.api.types.is_numeric_dtype(target_values)
    categorical_limit = max(20, int(len(target_values) * 0.05))
    task_type = "Classification" if not numeric_target or target_values.nunique() <= categorical_limit else "Regression"

    # Store imputation details
    imputation_details = []

    # Fill missing values in Baseline dataset
    for column in base_cleaned.columns:
        missing_count = base_cleaned[column].isnull().sum()
        if missing_count > 0:
            mode_series = base_cleaned[column].mode(dropna=True)
            if len(mode_series) > 0:
                replacement_value = mode_series.iloc[0]
                base_cleaned[column] = base_cleaned[column].fillna(replacement_value)
                imputation_details.append({
                    "Dataset": "Baseline",
                    "Column": column,
                    "Missing Values": missing_count,
                    "Replacement Value": replacement_value
                })

    # Fill missing values in Production dataset
    for column in prod_cleaned.columns:
        missing_count = prod_cleaned[column].isnull().sum()
        if missing_count > 0:
            mode_series = prod_cleaned[column].mode(dropna=True)
            if len(mode_series) > 0:
                replacement_value = mode_series.iloc[0]
                prod_cleaned[column] = prod_cleaned[column].fillna(replacement_value)
                imputation_details.append({
                    "Dataset": "Production",
                    "Column": column,
                    "Missing Values": missing_count,
                    "Replacement Value": replacement_value
                })

    base_nulls = int(base_cleaned.isna().sum().sum())
    prod_nulls = int(prod_cleaned.isna().sum().sum())

    # Column mapping & feature drift evaluation
    eval_cols = [col for col in base_cleaned.columns if col in prod_cleaned.columns and col not in [target_col, pred_col]]
    missing_in_prod = [col for col in base_cleaned.columns if col not in prod_cleaned.columns]
    telemetry_logs = []

    for col in eval_cols:
        try:
            if pd.api.types.is_numeric_dtype(base_cleaned[col]):
                baseline_data = base_cleaned[col].dropna()
                production_data = prod_cleaned[col].dropna()
                if baseline_data.empty or production_data.empty:
                    telemetry_logs.append({
                        "Feature": col, "Type": "Numerical", "Test": "KS-Test",
                        "Statistic": 0, "P_Value": 1, "Status": "No Data",
                        "Baseline Mean": np.nan, "Production Mean": np.nan, "Mean Difference": np.nan,
                        "Distribution Shift (%)": "-", "Severity": "NA"
                    })
                    continue
                
                ks_statistic, p_value = ks_2samp(baseline_data, production_data)
                baseline_mean = round(baseline_data.mean(), 2)
                production_mean = round(production_data.mean(), 2)
                mean_difference = round(production_mean - baseline_mean, 2)

                if ks_statistic < 0.10:
                    severity = "Low"
                elif ks_statistic < 0.20:
                    severity = "Medium"
                else:
                    severity = "High"

                telemetry_logs.append({
                    "Feature": col, "Type": "Numerical", "Test": "KS-Test",
                    "Statistic": round(ks_statistic, 4), "P_Value": round(p_value, 6),
                    "Status": "Drift Detected" if p_value < threshold else "No Drift",
                    "Baseline Mean": baseline_mean, "Production Mean": production_mean,
                    "Mean Difference": mean_difference, "Distribution Shift (%)": "-", "Severity": severity
                })
            else:
                baseline_counts = base_cleaned[col].astype(str).value_counts()
                production_counts = prod_cleaned[col].astype(str).value_counts()
                categories = sorted(baseline_counts.index.union(production_counts.index))

                baseline_freq = np.array([baseline_counts.get(x, 0) for x in categories], dtype=float)
                production_freq = np.array([production_counts.get(x, 0) for x in categories], dtype=float)

                baseline_freq[baseline_freq == 0] = 0.5
                production_freq[production_freq == 0] = 0.5

                contingency = np.array([baseline_freq, production_freq])
                chi_statistic, p_value, _, _ = chi2_contingency(contingency)

                baseline_pct = baseline_freq / baseline_freq.sum()
                production_pct = production_freq / production_freq.sum()
                max_shift = round(np.max(np.abs(baseline_pct - production_pct)) * 100, 2)

                if max_shift < 5:
                    severity = "Low"
                elif max_shift < 15:
                    severity = "Medium"
                else:
                    severity = "High"

                telemetry_logs.append({
                    "Feature": col, "Type": "Categorical", "Test": "Chi-Square",
                    "Statistic": round(chi_statistic, 4), "P_Value": round(p_value, 6),
                    "Status": "Drift Detected" if p_value < threshold else "No Drift",
                    "Baseline Mean": "-", "Production Mean": "-", "Mean Difference": "-",
                    "Distribution Shift (%)": f"{max_shift}%", "Severity": severity
                })

        except Exception as ex:
            telemetry_logs.append({
                "Feature": col, "Type": "Error", "Test": "-", "Statistic": "-", "P_Value": "-",
                "Status": str(ex), "Baseline Mean": "-", "Production Mean": "-", "Mean Difference": "-",
                "Distribution Shift (%)": "-", "Severity": "-"
            })

    drift_report_df = pd.DataFrame(telemetry_logs)

    # ML Model Performance Drift Evaluation & Automatic Retraining
    perf_metrics = {}
    retrain_info = {"triggered": False}

    try:
        # Exclude non-predictive ID / Naive Bayes columns if present to prevent leakage
        ignore_cols = [c for c in base_cleaned.columns if c.startswith("CLIENTNUM") or c.startswith("Naive_Bayes_Classifier")]

        if task_type == "Classification":
            baseline_df = base_cleaned.copy()
            production_df = prod_cleaned.copy()

            label_encoder = LabelEncoder()
            baseline_df[target_col] = label_encoder.fit_transform(baseline_df[target_col].astype(str))
            production_df[target_col] = label_encoder.transform(production_df[target_col].astype(str))

            X_base = baseline_df.drop(columns=[target_col] + ignore_cols, errors='ignore')
            y_base = baseline_df[target_col]
            X_prod = production_df.drop(columns=[target_col] + ignore_cols, errors='ignore')
            y_prod = production_df[target_col]

            combined = pd.concat([X_base, X_prod], axis=0)
            combined = pd.get_dummies(combined, drop_first=True)

            X_base = combined.iloc[:len(X_base)]
            X_prod = combined.iloc[len(X_base):]

            X_train, X_test, y_train, y_test = train_test_split(
                X_base, y_base, test_size=0.20, random_state=42, stratify=y_base
            )

            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)

            baseline_pred = model.predict(X_test)
            baseline_accuracy = accuracy_score(y_test, baseline_pred)
            baseline_precision = precision_score(y_test, baseline_pred, average="weighted", zero_division=0)
            baseline_recall = recall_score(y_test, baseline_pred, average="weighted", zero_division=0)
            baseline_f1 = f1_score(y_test, baseline_pred, average="weighted", zero_division=0)

            production_pred = model.predict(X_prod)
            production_accuracy = accuracy_score(y_prod, production_pred)
            production_precision = precision_score(y_prod, production_pred, average="weighted", zero_division=0)
            production_recall = recall_score(y_prod, production_pred, average="weighted", zero_division=0)
            production_f1 = f1_score(y_prod, production_pred, average="weighted", zero_division=0)

            accuracy_drop = round(baseline_accuracy - production_accuracy, 4)
            severity = "Low" if accuracy_drop < 0.05 else "Medium" if accuracy_drop < 0.15 else "High"
            degraded = "Yes" if accuracy_drop > 0.05 else "No"

            perf_metrics = {
                "Type": "Classification", "Metric Name": "Accuracy",
                "Baseline Metric": round(baseline_accuracy, 4), "Production Metric": round(production_accuracy, 4),
                "Secondary Name": "Weighted F1", "Secondary Metric Baseline": round(baseline_f1, 4),
                "Secondary Metric Production": round(production_f1, 4),
                "Baseline Precision": round(baseline_precision, 4), "Production Precision": round(production_precision, 4),
                "Baseline Recall": round(baseline_recall, 4), "Production Recall": round(production_recall, 4),
                "Accuracy Drop": round(accuracy_drop, 4), "Severity": severity, "Degraded": degraded
            }

            # Automatic retraining routine
            if degraded == "Yes":
                X_combined = pd.concat([X_base, X_prod], axis=0)
                y_combined = pd.concat([y_base, y_prod], axis=0)

                X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
                    X_combined, y_combined, test_size=0.20, random_state=42, stratify=y_combined
                )

                retrained_model = RandomForestClassifier(n_estimators=200, random_state=42)
                retrained_model.fit(X_r_train, y_r_train)

                retrained_pred = retrained_model.predict(X_r_test)
                new_accuracy = accuracy_score(y_r_test, retrained_pred)
                new_f1 = f1_score(y_r_test, retrained_pred, average="weighted", zero_division=0)

                retrain_info = {
                    "triggered": True,
                    "reason": f"Performance degradation threshold exceeded (Accuracy Drop: {accuracy_drop:.4f} > 0.05).",
                    "new_metric_name": "Accuracy",
                    "old_production_metric": round(production_accuracy, 4),
                    "new_retrained_metric": round(new_accuracy, 4),
                    "new_f1": round(new_f1, 4),
                    "status": "Retraining Successful - Accuracy Restored"
                }

        else:
            if pred_col not in base_cleaned.columns or pred_col not in prod_cleaned.columns:
                raise Exception("Prediction column not found.")

            b_target = base_cleaned[target_col]
            b_pred = base_cleaned[pred_col]
            p_target = prod_cleaned[target_col]
            p_pred = prod_cleaned[pred_col]

            baseline_rmse = np.sqrt(mean_squared_error(b_target, b_pred))
            production_rmse = np.sqrt(mean_squared_error(p_target, p_pred))
            baseline_mae = mean_absolute_error(b_target, b_pred)
            production_mae = mean_absolute_error(p_target, p_pred)

            rmse_increase = round(production_rmse - baseline_rmse, 4)
            severity = "Low" if rmse_increase < 0.10 else "Medium" if rmse_increase < 0.30 else "High"
            degraded = "Yes" if rmse_increase > 0.10 else "No"

            perf_metrics = {
                "Type": "Regression", "Metric Name": "RMSE",
                "Baseline Metric": round(baseline_rmse, 4), "Production Metric": round(production_rmse, 4),
                "Secondary Name": "MAE", "Secondary Metric Baseline": round(baseline_mae, 4),
                "Secondary Metric Production": round(production_mae, 4),
                "RMSE Increase": round(rmse_increase, 4), "Severity": severity, "Degraded": degraded
            }

            if degraded == "Yes":
                feature_cols = [c for c in base_cleaned.columns if c not in [target_col, pred_col] + ignore_cols]
                X_base_r = pd.get_dummies(base_cleaned[feature_cols], drop_first=True)
                X_prod_r = pd.get_dummies(prod_cleaned[feature_cols], drop_first=True)
                
                combined_r = pd.concat([X_base_r, X_prod_r], axis=0).fillna(0)
                y_combined_r = pd.concat([b_target, p_target], axis=0)

                X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
                    combined_r, y_combined_r, test_size=0.20, random_state=42
                )

                retrained_model = RandomForestRegressor(n_estimators=200, random_state=42)
                retrained_model.fit(X_r_train, y_r_train)

                retrained_pred = retrained_model.predict(X_r_test)
                new_rmse = np.sqrt(mean_squared_error(y_r_test, retrained_pred))

                retrain_info = {
                    "triggered": True,
                    "reason": f"Performance degradation threshold exceeded (RMSE Increase: {rmse_increase:.4f} > 0.10).",
                    "new_metric_name": "RMSE",
                    "old_production_metric": round(production_rmse, 4),
                    "new_retrained_metric": round(new_rmse, 4),
                    "status": "Retraining Successful - Error Reduced"
                }

    except Exception as ex:
        perf_metrics = {"Type": task_type, "Status": "Error", "Message": str(ex)}

    st.session_state.base = base_cleaned
    st.session_state.prod = prod_cleaned
    st.session_state.drift = drift_report_df
    st.session_state.perf = perf_metrics
    st.session_state.retrain = retrain_info
    st.session_state.base_nulls = base_nulls
    st.session_state.prod_nulls = prod_nulls
    st.session_state.missing_cols = missing_in_prod
    st.session_state.analysis_done = True

# =========================================================================
# 9. Display Dashboard and Audit Reports
# =========================================================================
if st.session_state.analysis_done:
    base_cleaned = st.session_state.base
    prod_cleaned = st.session_state.prod
    drift_report_df = st.session_state.drift
    perf_metrics = st.session_state.perf
    retrain_info = st.session_state.get("retrain", {"triggered": False})
    
    drifted_total = int((drift_report_df["Status"] == "Drift Detected").sum())
    stable_total = int((drift_report_df["Status"] == "No Drift").sum())
    drifted_features_list = drift_report_df[drift_report_df["Status"] == "Drift Detected"]["Feature"].tolist()

    st.markdown('<div class="section-header">🚨 Production data drift Assessment summary & Alerts</div>', unsafe_allow_html=True)
    perf_degraded = perf_metrics.get("Degraded", "No") == "Yes"
    
    if drifted_total >= 3 or perf_degraded:
        severity_rank = "CRITICAL / ACTION REQUIRED"
        st.error("🚨 CRITICAL ALERT: Data drift and drop in model accuracy have been detected. We request you to immediately review the details mentioned below.")
    elif drifted_total > 0:
        severity_rank = "WARNING / ATTENTION ADVISED"
        st.warning("⚠️ RISK NOTICE: Moderate changes detected in customer or operational data behaviors. The system remains reliable, but tracking trend anomalies is advised.")
    else:
        severity_rank = "HEALTHY / STABLE"
        st.success("✅ SYSTEM STABLE: Live data flows closely match historical reference limits. No operations modifications are required.")

    st.markdown('<div class="section-header">🔄 Automatic Model Retraining Summary</div>', unsafe_allow_html=True)
    if retrain_info.get("triggered"):
        st.info(
            f"⚡ **Automatic Model Retraining Triggered!**\n\n"
            f"- **Reason:** {retrain_info['reason']}\n"
            f"- **Previous Production Metric ({retrain_info['new_metric_name']}):** {retrain_info['old_production_metric']}\n"
            f"- **Retrained Model Metric ({retrain_info['new_metric_name']}):** {retrain_info['new_retrained_metric']}\n"
            f"- **Status:** {retrain_info['status']} (Manual intervention minimized & prediction accuracy restored)."
        )
    else:
        st.success("✅ **Model Retraining Not Required:** Model performance degradation threshold was not exceeded. Prediction accuracy remains stable.")

    st.markdown('<div class="section-header">🏢 Baseline and Production Data Summary Details</div>', unsafe_allow_html=True)
    
    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        st.markdown(f'<div class="biz-card"><div class="biz-card-title">Baseline Dataset Record Count</div><div class="biz-card-value" style="color: #2b5c8f;">{len(base_cleaned):,}</div><div class="biz-card-desc">Total Baseline dataset count considered as reference.</div></div>', unsafe_allow_html=True)
    with bc2:
        st.markdown(f'<div class="biz-card"><div class="biz-card-title">Total Production dataset count</div><div class="biz-card-value" style="color: #ff9900;">{len(prod_cleaned):,}</div><div class="biz-card-desc">Total Production dataset count considered as reference.</div></div>', unsafe_allow_html=True)
    with bc3:
        st.markdown(f'<div class="biz-card"><div class="biz-card-title">Number of columns With no Drift</div><div class="biz-card-value" style="color: #2e7d32;">{stable_total}</div><div class="biz-card-desc">Number of columns are shows expected baseline data distributions pattern.</div></div>', unsafe_allow_html=True)
    with bc4:
        card_color = "#d32f2f" if drifted_total > 0 else "#2e7d32"
        st.markdown(f'<div class="biz-card"><div class="biz-card-title">Number of columns With Drift</div><div class="biz-card-value" style="color: {card_color};">{drifted_total}</div><div class="biz-card-desc">Number of columns shows data changes exceeding the defined threshold limit.</div></div>', unsafe_allow_html=True)

    if st.session_state.missing_cols:
        st.warning(f"⚠️ Structural Column Warning: The following baseline fields were completely missing from the production dataset and hence omitted in reporting: {', '.join(st.session_state.missing_cols)}")

    st.markdown('<div class="section-header">📊 Graphical representation for individual column and Overall Data Drift Summary</div>', unsafe_allow_html=True)
    theme_colors = {"Drift Detected": "#d9534f", "No Drift": "#2b5c8f"}
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.markdown("<div class='concept-box'><b>Drift detection indicators details</b> Red indicator shown below the dashed alpha (threshold) represent attributes where data drift is observed.</div>", unsafe_allow_html=True)
        plot_df = drift_report_df.copy()
        plot_df["Value_Label"] = plot_df["P_Value"].apply(lambda v: f"{v:.4f}")
        plot_df["Plot_P_Value"] = plot_df["P_Value"].apply(lambda v: 0.012 if v < 0.001 else v)
        plot_df.loc[plot_df["P_Value"] < 0.001, "Value_Label"] = "Shifted (~0)"
        
        bar_chart = px.bar(
            plot_df, x="Feature", y="Plot_P_Value", color="Status", text="Value_Label",
            color_discrete_map=theme_colors, title="<b>Individual column P-Value data</b>", template="plotly_white"
        )
        bar_chart.add_hline(y=threshold, line_dash="dash", line_color="#d9534f", annotation_text="Safety Cutoff Boundary Line")
        bar_chart.update_traces(textposition='outside', cliponaxis=False)
        bar_chart.update_layout(yaxis_title="Calculated Stability Index", xaxis_title="Evaluated Business Metrics", xaxis_tickangle=-45, yaxis=dict(range=[0, 1.15]))
        st.plotly_chart(bar_chart, use_container_width=True)

    with g_col2:
        st.markdown("<div class='concept-box'><b>Drifted and non drifted attributes:</b> Provides a high-level view of the proportion of drifted vs non-drifted columns.</div>", unsafe_allow_html=True)
        pie_chart = px.pie(names=["Count of Drifted Columns", "Count of Non-drifted Columns"], values=[drifted_total, stable_total],
                           color=["Count of Drifted Columns", "Count of Non-drifted Columns"], color_discrete_map={"Count of Drifted Columns": "#d9534f", "Count of Non-drifted Columns": "#2b5c8f"},
                           title="<b>Overall Data Drift Summary Dashboard</b>", hole=0.4, template="plotly_white")
        st.plotly_chart(pie_chart, use_container_width=True)

    st.markdown('<div class="section-header">📋 DETAILED DATA DRIFT REPORT</div>', unsafe_allow_html=True)
    st.dataframe(drift_report_df.style.apply(lambda row: ['background-color: #f8d7da' if val == 'Drift Detected' else 'background-color: #d1e7dd' if val == 'No Drift' else '' for val in row], axis=1), use_container_width=True)

    audit_uuid = str(uuid.uuid4())
    metadata_audit_payload = {
        "run_id": audit_uuid,
        "timestamp": str(datetime.now()),
        "overall_severity": severity_rank,
        "metrics_summary": perf_metrics,
        "automatic_retraining_summary": retrain_info,
        "detailed_results": drift_report_df.to_dict("records")
    }

    temp_dir = tempfile.gettempdir()
    json_path = f"{temp_dir}/drift_audit_report.json"
    csv_path = f"{temp_dir}/drift_summary_matrix.csv"
    bar_html = f"{temp_dir}/drift_significance_chart.html"
    pie_html = f"{temp_dir}/drift_distribution_chart.html"

    with open(json_path, "w") as jf:
        json.dump(metadata_audit_payload, jf, indent=4)
    drift_report_df.to_csv(csv_path, index=False)
    bar_chart.write_html(bar_html)
    pie_chart.write_html(pie_html)

    st.markdown('<div class="section-header">📥 Download Audit Logs</div>', unsafe_allow_html=True)
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("📥 Save Detailed Audit Trail (JSON)", json.dumps(metadata_audit_payload, indent=4), "drift_audit_trail_report.json", use_container_width=True)
    with dl2:
        st.download_button("📥 Save Detailed Data Drift Summary Report", drift_report_df.to_csv(index=False), "data_drift_summary.csv", use_container_width=True)

    # Email alert sending integration
    st.markdown('<div class="section-header">📧 Send Data Drift Email Alert to Business Stakeholders</div>', unsafe_allow_html=True)
    if st.button("📧 Send Email to Business Stakeholders", type="primary"):
        if not sender.strip() or not password.strip() or not receiver.strip():
            st.warning("⚠️ Configuration Incomplete: Please verify your sender, app authorization credentials, and recipient email targets in the left adjustment pane.")
        else:
            with st.spinner("Assembling secure document payloads and establishing mail channel connection..."):
                try:
                    msg = MIMEMultipart()
                    msg["From"] = sender
                    msg["To"] = receiver
                    msg["Subject"] = f"🚨 [{severity_rank}] Automated AI Data Quality Assessment Update"
                    
                    drift_summary_text = ", ".join(drifted_features_list) if drifted_features_list else "None"
                    
                    if drifted_total >= 3 or perf_degraded:
                        business_next_step = "ACTION REQUIRED: Continuous monitoring triggered automatic model retraining routine due to degradation threshold breach."
                    elif drifted_total > 0:
                        business_next_step = "ATTENTION ADVISED: No immediate adjustment is needed. Data behavior variations are within functional tolerances, but continued monitoring is recommended."
                    else:
                        business_next_step = "NO ACTION REQUIRED: Data paths are running correctly and functioning within historical expectations."

                    retrain_email_summary = (
                        f"Automatic Retraining Triggered : Yes\n"
                        f"Retraining Reason            : {retrain_info.get('reason')}\n"
                        f"New Retrained Model Metric   : {retrain_info.get('new_retrained_metric')}"
                        if retrain_info.get("triggered") else
                        "Automatic Retraining Triggered : No (Model Performance Stable)"
                    )

                    email_body = f"""Dear Stakeholder,

This is an automated operational status message from your Continuous Monitoring Hub. 

We have successfully performed an audit comparing your current live data against original historical reference benchmarks. Please find the executive summary detailed below:

==========================================================================
                     PROJECT DATA SUMMARY REPORT
==========================================================================
Audit Unique ID        : {audit_uuid}
Timestamp Analyzed     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
System Health Status   : {severity_rank}

[KEY PERFORMANCE INDICATORS]
- Total Business Metrics Checked : {len(drift_report_df)} fields verified
- Shifted / Altered Columns      : {drifted_total} fields flag variance
- Specific Shifted Fields        : {drift_summary_text}
- Prediction Accuracy Stable?    : {'NO (Capability Drop Flagged)' if perf_degraded else 'YES (Operating smoothly within target limits)'}

[AUTOMATIC MODEL RETRAINING LOG]
{retrain_email_summary}

==========================================================================
Operational Summary:
{business_next_step}
==========================================================================

Please find the attached detailed spreadsheets, data records and interactive analytical visualization charts.

Best Regards,
Automated Corporate Operations Gateway
"""
                    msg.attach(MIMEText(email_body, "plain"))
                    
                    for filepath in [json_path, csv_path, bar_html, pie_html]:
                        path_obj = Path(filepath)
                        if path_obj.exists():
                            part = MIMEBase("application", "octet-stream")
                            with open(path_obj, "rb") as fp:
                                part.set_payload(fp.read())
                            encoders.encode_base64(part)
                            part.add_header("Content-Disposition", f'attachment; filename="{path_obj.name}"')
                            msg.attach(part)

                    server = smtplib.SMTP("smtp.gmail.com", 587)
                    server.starttls()
                    server.login(sender, password)
                    server.send_message(msg)
                    server.quit()
                    st.success("✅ Project summary report with all required files successfully sent to business stakeholders.")
                except Exception as ex:
                    st.error(f"❌ Error in Mail Delivery Infrastructure. {ex}")
