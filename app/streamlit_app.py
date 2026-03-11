import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_machine_failure

st.set_page_config(
    page_title="Predictive Maintenance AI",
    layout="wide"
)

st.title("🔧 AI-Assisted Predictive Maintenance Dashboard")

# Sidebar
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Dataset Analytics", "Row Prediction", "Batch Prediction"]
)

# ==========================================
# DATA SOURCE SELECTION
# ==========================================

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_source = st.sidebar.radio(
    "Select Dataset Source",
    ["Sensor Dataset", "Sensor + Failure Dataset", "Upload Machine Dataset"]
)

sensor_dataset_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "binary_X_test_dataset.csv"
)

sensor_failure_dataset_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "binary_full_test_dataset.csv"
)

df = None

if data_source == "Sensor Dataset":

    df = load_data(sensor_dataset_path)

elif data_source == "Sensor + Failure Dataset":

    df = load_data(sensor_failure_dataset_path)

else:

    uploaded_file = st.sidebar.file_uploader(
        "Upload Machine Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# Stop if dataset still not loaded
if df is None:
    st.warning("Please select or upload a dataset.")
    st.stop()

st.caption(f"Dataset Loaded: {data_source}")
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

    # =============================
    # ROW PREDICTION
    # =============================

elif mode == "Row Prediction":

        st.subheader("Single Machine Prediction")

        st.dataframe(df)

        row_index = st.selectbox(
            "Select Row Index",
            options=list(range(len(df)))
        )

        if st.button("Predict Machine Status"):

            row = df.iloc[[row_index]]

            pred, prob = predict_machine_failure(row)

            col1, col2 = st.columns(2)

            # FAILURE GAUGE
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={'text': "Failure Probability (%)"},
                gauge={
                    'axis': {'range': [0,100]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range':[0,40],'color':"green"},
                        {'range':[40,70],'color':"orange"},
                        {'range':[70,100],'color':"red"}
                    ]
                }
            ))

            col1.plotly_chart(gauge, use_container_width=True)

            # HEALTH SCORE
            health_score = 100 - (prob*100)

            health = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health_score,
                title={'text': "Machine Health Score"},
                gauge={
                    'axis': {'range':[0,100]},
                    'bar': {'color': "green"}
                }
            ))

            col2.plotly_chart(health, use_container_width=True)

            if pred == 1:
                st.error("⚠️ Machine Failure Likely")
            else:
                st.success("✅ Machine Operating Normally")

    # =============================
    # BATCH PREDICTION
    # =============================

elif mode == "Batch Prediction":

        st.subheader("Batch Prediction")

        st.dataframe(df.head())

        if st.button("Predict Entire Dataset"):

            results = []

            for i in range(len(df)):

                row = df.iloc[[i]]

                pred, prob = predict_machine_failure(row)

                results.append({
                    "Prediction": pred,
                    "Failure Probability": prob
                })

            results_df = pd.DataFrame(results)

            output_df = pd.concat([df, results_df], axis=1)

            st.dataframe(output_df)

            # Failure probability chart
            fig = px.histogram(
                output_df,
                x="Failure Probability",
                title="Failure Probability Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

            csv = output_df.to_csv(index=False)

            st.download_button(
                label="Download Predictions.csv",
                data=csv,
                file_name="Predictions.csv",
                mime="text/csv"
            )