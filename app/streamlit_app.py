import sys
import os

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import plotly.express as px
import streamlit as st
import pandas as pd
import joblib
#import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# ── Failure type descriptions ─────────────────────────────────────────────────
FAILURE_DESCRIPTIONS = {
    "No Failure":        ("✅", "No Failure",               "#2ecc71", "Machine is operating normally."),
    "TWF":               ("🔧", "Tool Wear Failure",         "#e67e22", "The cutting tool has exceeded its wear limit and needs replacement."),
    "HDF":               ("🌡️", "Heat Dissipation Failure", "#e74c3c", "Insufficient cooling — temperature difference is too low at high rotational speed."),
    "PWF":               ("⚡", "Power Failure",             "#9b59b6", "Power delivered is outside safe limits for the torque/speed combination."),
    "OSF":               ("💪", "Overstrain Failure",        "#e74c3c", "Product of tool wear and torque exceeds the machine's strain limit."),
    "RNF":               ("🎲", "Random Failure",            "#95a5a6", "Random failure with no specific identifiable cause (0.1% probability)."),
    "Multiple Failure":  ("⚠️", "Multiple Failures",        "#c0392b", "More than one failure type is occurring simultaneously."),
}

# ── Load artifacts ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(os.path.join(BASE_DIR, "artifacts/preprocessing_pipeline.pkl"))
    model_binary = joblib.load(os.path.join(BASE_DIR, "artifacts/best_model.pkl"))
    model_multiclass = joblib.load(os.path.join(BASE_DIR, "artifacts/best_model_multiclass.pkl"))
    return pipeline, model_binary, model_multiclass

pipeline, model_binary, model_multiclass = load_artifacts()

# ── Feature columns ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def predict_binary(input_df: pd.DataFrame):
    X     = pipeline.transform(input_df[FEATURE_COLS])
    pred  = model_binary.predict(X)
    proba = model_binary.predict_proba(X)[:, 1] if hasattr(model_binary, "predict_proba") else None
    return pred, proba


def predict_multiclass(input_df: pd.DataFrame):
    X     = pipeline.transform(input_df[FEATURE_COLS])
    pred    = model_multiclass.predict(X)
    proba   = model_multiclass.predict_proba(X) if hasattr(model_multiclass, "predict_proba") else None
    classes = model_multiclass.classes_ if hasattr(model_multiclass, "classes_") else None
    return pred, proba, classes


def load_default_data():
    return pd.read_csv(os.path.join(BASE_DIR, "data/raw/ai4i2020.csv"))


def color_rows(row):
    color = "background-color: rgba(231,76,60,0.20); color: #ff6b6b;" if row["Binary Prediction"] == 1 else ""
    return [color] * len(row)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Predictive Maintenance AI")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Select Mode",
    ["Dataset Analytics","Single Row Prediction", "Batch Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file (optional)",
    type=["csv"],
    help="Must contain the same columns as the default dataset."
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            st.sidebar.error(f"❌ Missing columns: {missing}")
            df = load_default_data()
        else:
            st.sidebar.success(f"⛁ Loaded: {uploaded_file.name}  ({len(df):,} rows)")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        df = load_default_data()
else:
    df = load_default_data()
    st.sidebar.info("Using default dataset (ai4i2020.csv)")

# ── Failure type legend (sidebar) ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("☰ Failure Type Legend")
for code, (icon, name, color, desc) in FAILURE_DESCRIPTIONS.items():
    st.sidebar.markdown(
        f"<span style='color:{color}'>{icon} **{code}**</span> — {name}",
        unsafe_allow_html=True
    )

# =============================
# DATA ANALYTICS DASHBOARD
# =============================
if mode == "Dataset Analytics":

        st.subheader("Machine Data Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", len(df))
        col2.metric("Average Torque", round(df["Torque [Nm]"].mean(),2))
        col3.metric("Average Tool Wear", round(df["Tool wear [min]"].mean(),2))

        st.divider()

        # Rotational Speed Trend
        fig1 = px.line(
            df,
            y="Rotational speed [rpm]",
            title="Rotational Speed Trend"
        )

        st.plotly_chart(fig1, use_container_width=True)

        # Torque Trend
        fig2 = px.line(
            df,
            y="Torque [Nm]",
            title="Torque Trend"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Tool Wear Distribution
        fig3 = px.histogram(
            df,
            x="Tool wear [min]",
            title="Tool Wear Distribution"
        )

        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Dataset Preview")
        st.dataframe(df)

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 – Single Row Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Single Row Prediction":
    st.title("🔍 Single Row Prediction")
    st.markdown("Type a row number **or** click a row in the table to select it.")

    preview_df = df[FEATURE_COLS].copy().reset_index(drop=False)
    preview_df.rename(columns={"index": "Row #"}, inplace=True)

    # ── Input row + view toggle on same line ──────────────────────────────────
    input_col, toggle_col, _ = st.columns([1, 1, 2])
    with input_col:
        typed_idx = st.number_input(
            "Enter Row Number",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1,
            help=f"Enter any row between 0 and {len(df)-1}"
        )
    with toggle_col:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        show_all = st.toggle("Show full table", value=False)

    # ── Table: full or windowed ────────────────────────────────────────────────
    if show_all:
        table_df = preview_df
        st.subheader(f"📂 Full Dataset ({len(table_df):,} rows)")
    else:
        window = 15
        center = int(typed_idx)
        start  = max(0, center - window)
        end    = min(len(preview_df), center + window + 1)
        table_df = preview_df.iloc[start:end].copy()
        st.subheader(f"📂 Rows {start}–{end-1}  (centered on {center})")

    event = st.dataframe(
        table_df,
        use_container_width=True,
        height=280,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="row_selector",
    )

    # Table click takes priority; otherwise use typed number
    selected_indices = event.selection.rows if event.selection else []
    if selected_indices:
        row_idx = int(table_df.iloc[selected_indices[0]]["Row #"])
    else:
        row_idx = int(typed_idx)

    selected_row = df.iloc[[row_idx]]

    st.subheader(f"✓ Selected Row #{row_idx}")
    st.dataframe(selected_row[FEATURE_COLS], use_container_width=True, hide_index=True)

    st.markdown("---")

    col1, divider, col2 = st.columns([1, 0.05, 1])



    # ── Step 1: Binary ─────────────────────────────────────────────────────────
    pred_b, proba_b = predict_binary(selected_row)

    with col1:
        st.subheader("Step 1 — Will It Fail?")
        st.caption("Binary classifier: Healthy vs Failure")
        if pred_b[0] == 1:
            st.error("🚨 MACHINE FAILURE PREDICTED")
        else:
            st.success("✅ MACHINE IS HEALTHY")
        if proba_b is not None:
            st.markdown("#### Failure Probability")
            #st.metric("", f"{proba_b[0] * 100:.1f}%")

            prob = float(proba_b[0])  # convert to scalar


            # FAILURE GAUGE
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Failure Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bgcolor': "#161B22",
                    'borderwidth': 2,
                    'bordercolor': "#2A2E39",
                    'bar': {'color': "#FF4B4B"},
                    'steps': [
                        {'range': [0, 40], 'color': "#00C853"},
                        {'range': [40, 70], 'color': "#FFB300"},
                        {'range': [70, 100], 'color': "#FF1744"}
                    ]
                }
            ))

            st.plotly_chart(gauge, use_container_width=True)

            st.divider()

            # HEALTH SCORE
            health_score = 100 - (prob * 100)

            health = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health_score,
                title={'text': "Machine Health Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "green"}
                }
            ))

            st.markdown("#### Machine Health Score")
            st.plotly_chart(health, use_container_width=True)

    with divider:
        st.markdown(
        """
        <div style="
            border-left:1px solid rgba(255,255,255,0.2);
            height:1260px;
            margin:auto;
        "></div>
        """,
        unsafe_allow_html=True
    )
        

    # ── Step 2: Multiclass — only runs if binary predicts FAILURE ─────────────
    with col2:
        st.subheader("Step 2 — What Type of Failure?")
        st.caption("Multiclass classifier: identifies the specific failure type")

        if pred_b[0] == 0:
            # Machine is healthy — skip multiclass entirely
            st.markdown(
                """
                <div style="
                    background: #1a3a2a;
                    border: 2px solid #2ecc71;
                    border-radius: 12px;
                    padding: 16px 20px;
                ">
                    <h3 style="color:#2ecc71; margin:0">✅ No Failure Detected</h3>
                    <p style="color:#aaaaaa; margin:8px 0 0 0">
                        The binary model predicts this machine is healthy.<br>
                        Failure type diagnosis is not needed.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Machine predicted to fail — run multiclass to find the type
            pred_m, proba_m, classes_m = predict_multiclass(selected_row)
            failure_type = pred_m[0]

            if failure_type == "No Failure":
                # Convert probabilities to DataFrame
                prob_df = pd.DataFrame({
                    "Failure Type": classes_m,
                    "Probability": proba_m[0]
                })

                # Remove "No Failure"
                prob_df = prob_df[prob_df["Failure Type"] != "No Failure"]

                # Select failure type with highest probability
                failure_type = prob_df.loc[prob_df["Probability"].idxmax(), "Failure Type"]
                st.error(f"Note: First Possible Failure type may be No Failure or Unknow Failure type, 2nd Possible is {failure_type}")

                          
            icon, name, color, desc = FAILURE_DESCRIPTIONS.get(
            failure_type, ("❓", failure_type, "#ffffff", "Unknown failure type.")
            )

            st.markdown(
                f"""
                <div style="
                    background: {color}22;
                    border: 2px solid {color};
                    border-radius: 12px;
                    padding: 16px 20px;
                    margin-bottom: 12px;
                ">
                    <h3 style="color:{color}; margin:0">{icon} {name}</h3>
                    <p style="color:#cccccc; margin:8px 0 0 0">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Probability bar chart per failure type
            if proba_m is not None and classes_m is not None:
                st.markdown("#### **Probability per Failure Type**")
                prob_df = (
                    pd.DataFrame({"Failure Type": classes_m, "Probability": proba_m[0]})
                    .sort_values("Probability", ascending=True)
                )
                bar_colors = [
                    FAILURE_DESCRIPTIONS.get(c, ("", "", "#3498db", ""))[2]
                    for c in prob_df["Failure Type"]
                ]
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(prob_df["Failure Type"], prob_df["Probability"], color=bar_colors)
                ax.set_xlabel("Probability", color="white")
                ax.tick_params(colors="white")
                ax.spines[["top", "right"]].set_visible(False)
                fig.patch.set_facecolor("#0e1117")
                ax.set_facecolor("#0e1117")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)


    # ── Full dataset preview ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Full Dataset Preview")
    st.dataframe(df[FEATURE_COLS].head(100), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 3 – Batch Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Batch Prediction":
    st.title("📊 Batch Prediction")
    st.markdown("Run both models on the entire dataset.")

    pred_b_all, proba_b_all    = predict_binary(df)
    pred_m_all, proba_m_all, _ = predict_multiclass(df)

    results_df = df[FEATURE_COLS].copy()
    results_df["Binary Prediction"]     = pred_b_all
    results_df["Failure Probability %"] = (
        (proba_b_all * 100).round(1) if proba_b_all is not None else "N/A"
    )
    results_df["Health Status"] = results_df["Binary Prediction"].map(
        {0: "✅ Healthy", 1: "🚨 Failure"}
    )
    # Only show multiclass failure type when binary predicts failure
    # Healthy rows get "—" so the table is not misleading
    results_df["Failure Type"] = [
        ft if binary == 1 else "—"
        for ft, binary in zip(pred_m_all, pred_b_all)
    ]

    # ── Summary metrics ────────────────────────────────────────────────────────
    n_total    = len(results_df)
    n_failures = int(pred_b_all.sum())
    n_healthy  = int((pred_b_all == 0).sum())
    failure_rate = pred_b_all.mean() * 100

    if "show_failures_only" not in st.session_state:
        st.session_state.show_failures_only = False

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Machines",     f"{n_total:,}")
    col2.metric("Failures Predicted", f"{n_failures:,}")
    col3.metric("Healthy Machines",   f"{n_healthy:,}")
    col4.metric("Failure Rate",       f"{failure_rate:.1f}%")

    # Toggle filter button below metrics
    active = st.session_state.show_failures_only
    toggle_label = "🔴 Show All Rows" if active else "🔴 Show Failures Only"
    if st.button(toggle_label):
        st.session_state.show_failures_only = not st.session_state.show_failures_only
        st.rerun()
    st.markdown("---")

    # ── Color-coded results table with row drill-down ───────────────────────
    if st.session_state.show_failures_only:
        display_df = results_df[results_df["Binary Prediction"] == 1].copy()
        st.subheader(f"📋 Showing Failures Only — {len(display_df):,} rows")
    else:
        display_df = results_df.copy()
        st.subheader("📋 Prediction Results  *(🔴 row = predicted failure)*")

    st.caption("👆 Click any row to inspect it in detail below.")
    styled = display_df.style.apply(color_rows, axis=1)
    table_event = st.dataframe(
        styled,
        use_container_width=True,
        height=400,
        on_select="rerun",
        selection_mode="single-row",
        key="batch_table",
    )

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv_bytes,
        file_name="predictions_failures_only.csv" if st.session_state.show_failures_only else "predictions.csv",
        mime="text/csv",
    )

    # ── Row drill-down panel ──────────────────────────────────────────────────
    selected_batch_rows = table_event.selection.rows if table_event.selection else []
    if selected_batch_rows:
        sel_display_idx = selected_batch_rows[0]
        # Map back to original df index
        actual_row_idx  = display_df.index[sel_display_idx]
        drilldown_row   = df.iloc[[actual_row_idx]]

        st.markdown("---")
        st.subheader(f"🔬 Drill-Down: Row #{actual_row_idx}")
        st.dataframe(drilldown_row[FEATURE_COLS], use_container_width=True, hide_index=True)

        d_col1, d_col2 = st.columns(2)

        # Step 1: binary result for this row
        pred_d, proba_d = predict_binary(drilldown_row)
        with d_col1:
            st.markdown("**Step 1 — Health Status**")
            if pred_d[0] == 1:
                st.error("🚨 MACHINE FAILURE PREDICTED")
            else:
                st.success("✅ MACHINE IS HEALTHY")
            if proba_d is not None:
                st.markdown("#### Failure Probability")
                #st.metric(f"{proba_d[0]*100:.1f}%")

                prob = float(proba_d[0])  # convert to scalar


                # FAILURE GAUGE
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={'text': "Failure Probability (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bgcolor': "#161B22",
                        'borderwidth': 2,
                        'bordercolor': "#2A2E39",
                        'bar': {'color': "#FF4B4B"},
                        'steps': [
                            {'range': [0, 40], 'color': "#00C853"},
                            {'range': [40, 70], 'color': "#FFB300"},
                            {'range': [70, 100], 'color': "#FF1744"}
                        ]
                    }
                ))

                st.plotly_chart(gauge, use_container_width=True)

                st.divider()

                # HEALTH SCORE
                health_score = 100 - (prob * 100)

                health = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=health_score,
                    title={'text': "Machine Health Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "green"}
                    }
                ))

                st.markdown("#### Machine Health Score")
                st.plotly_chart(health, use_container_width=True)


        # Step 2: multiclass diagnosis only if failure
        with d_col2:
            st.markdown("**Step 2 — Failure Type Diagnosis**")
            if pred_d[0] == 0:
                st.markdown(
                    """
                    <div style="background:#1a3a2a;border:2px solid #2ecc71;
                    border-radius:12px;padding:14px 18px;">
                    <h4 style="color:#2ecc71;margin:0">✅ No Failure Detected</h4>
                    <p style="color:#aaaaaa;margin:6px 0 0 0">Machine is healthy — no diagnosis needed.</p>
                    </div>""",
                    unsafe_allow_html=True
                )
            else:
                pred_dm, proba_dm, classes_dm = predict_multiclass(drilldown_row)
                ft = pred_dm[0]

                if ft == "No Failure":
                    # Convert probabilities to DataFrame
                    prob_df = pd.DataFrame({
                        "Failure Type": classes_dm,
                        "Probability": proba_dm[0]
                    })

                    # Remove "No Failure"
                    prob_df = prob_df[prob_df["Failure Type"] != "No Failure"]

                    # Select failure type with highest probability
                    ft = prob_df.loc[prob_df["Probability"].idxmax(), "Failure Type"]
                    st.error(f"Note: First Possible Failure type may be No Failure or Unknow Failure type, 2nd Possible is {ft}")

                icon, name, color, desc = FAILURE_DESCRIPTIONS.get(
                    ft, ("❓", ft, "#ffffff", "Unknown failure type.")
                )
                st.markdown(
                    f"""<div style="background:{color}22;border:2px solid {color};
                    border-radius:12px;padding:14px 18px;margin-bottom:10px;">
                    <h4 style="color:{color};margin:0">{icon} {name}</h4>
                    <p style="color:#cccccc;margin:6px 0 0 0">{desc}</p>
                    </div>""",
                    unsafe_allow_html=True
                )
                if proba_dm is not None and classes_dm is not None:
                    prob_df = (
                        pd.DataFrame({"Failure Type": classes_dm, "Probability": proba_dm[0]})
                        .sort_values("Probability", ascending=True)
                    )
                    bar_colors = [
                        FAILURE_DESCRIPTIONS.get(c, ("","","#3498db",""))[2]
                        for c in prob_df["Failure Type"]
                    ]
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    ax.barh(prob_df["Failure Type"], prob_df["Probability"], color=bar_colors)
                    ax.set_xlabel("Probability", color="white")
                    ax.tick_params(colors="white")
                    ax.spines[["top","right"]].set_visible(False)
                    fig.patch.set_facecolor("#0e1117")
                    ax.set_facecolor("#0e1117")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

    st.markdown("---")

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Healthy vs Failure")
        sizes = [n_healthy, n_failures]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            sizes,
            labels=["✅ Healthy", "🚨 Failure"],
            autopct="%1.1f%%",
            colors=["#2ecc71", "#e74c3c"],
            startangle=140,
            wedgeprops={"edgecolor": "#0e1117", "linewidth": 2}
        )
        plt.setp(ax.texts, color="white")
        fig.patch.set_facecolor("#0e1117")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Failure Probability Distribution")
        if proba_b_all is not None:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.hist(proba_b_all, bins=40, color="#3498db", edgecolor="#0e1117", alpha=0.85)
            ax.axvline(0.5, color="#e74c3c", linestyle="--", linewidth=1.5, label="Threshold (0.5)")
            ax.set_xlabel("Failure Probability", color="white")
            ax.set_ylabel("Count", color="white")
            ax.legend(facecolor="#1e1e2e", labelcolor="white")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(colors="white")
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            st.pyplot(fig)
            plt.close(fig)

    # ── Failure type breakdown — only among predicted failures ────────────────
    st.markdown("---")
    st.subheader("🔬 Failure Type Breakdown (Multiclass Model)")
    st.caption("Shows failure type distribution among **predicted failure machines only**.")

    failure_types_only = results_df[results_df["Binary Prediction"] == 1]["Failure Type"]
    type_counts = failure_types_only.value_counts().sort_values(ascending=True)
    bar_colors  = [
        FAILURE_DESCRIPTIONS.get(c, ("", "", "#3498db", ""))[2]
        for c in type_counts.index
    ]

    if len(type_counts) == 0:
        st.info("No failures predicted in this dataset.")
    else:
        fig, ax = plt.subplots(figsize=(9, max(3, len(type_counts) * 0.6)))
        ax.barh(type_counts.index, type_counts.values, color=bar_colors)
        ax.set_xlabel("Number of Machines", color="white")
        ax.set_title("Failure Type Distribution (Failures Only)", color="white", fontsize=13, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors="white")
        for i, val in enumerate(type_counts.values):
            ax.text(val + 1, i, str(val), va="center", color="white", fontsize=9)
        fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
