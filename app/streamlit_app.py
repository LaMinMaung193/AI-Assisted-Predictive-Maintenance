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

# ==========================================
# INDUSTRIAL DARK THEME
# ==========================================

import plotly.io as pio
pio.templates.default = "plotly_dark"

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}

[data-testid="stSidebar"] {
    background-color: #161B22;
}

h1, h2, h3 {
    color: #4DB6FF;
}

.stMetric {
    background-color: #1E242E;
    padding: 15px;
    border-radius: 10px;
}

.stDataFrame {
    background-color: #1E242E;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
"""
# Predictive Maintenance AI
### Real-Time Machine Health Monitoring System
"""
)
st.markdown("---")


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

st.info(f"Dataset Loaded: {data_source}")
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

        # Run predictions
        predictions = []
        probabilities = []

        for i in range(len(df)):
            row = df.iloc[[i]]
            pred, prob = predict_machine_failure(row)

            predictions.append(pred)
            probabilities.append(prob)

        # Create results dataframe
        results_df = pd.DataFrame({
            "Prediction": predictions,
            "Failure Probability": probabilities
        })

        # Merge with original dataset
        output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        # =============================
        # MACHINE STATUS CLASSIFICATION
        # =============================

        def classify_machine(prob):
            if prob < 0.3:
                return "Healthy"
            elif prob < 0.7:
                return "Warning"
            else:
                return "Critical"

        output_df["Machine Status"] = output_df["Failure Probability"].apply(classify_machine)

        # Show predictions table
        st.dataframe(output_df)

        # =============================
        # FLEET HEALTH SUMMARY
        # =============================

        total = len(output_df)
        healthy = (output_df["Machine Status"] == "Healthy").sum()
        warning = (output_df["Machine Status"] == "Warning").sum()
        critical = (output_df["Machine Status"] == "Critical").sum()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Machines Monitored", total)
        col2.metric("Healthy", healthy)
        col3.metric("Warning", warning)
        col4.metric("Critical", critical)

        # =============================
        # FAILURE PROBABILITY DISTRIBUTION
        # =============================

        fig = px.histogram(
            output_df,
            x="Failure Probability",
            nbins=30,
            title="Failure Probability Distribution"
        )

        # =============================
        # VISUALIZATION
        # =============================

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
            output_df,
            x="Failure Probability",
            nbins=30,
            title="Failure Probability Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            status_fig = px.pie(
            output_df,
            names="Machine Status",
            title="Machine Fleet Health Distribution",
            color="Machine Status",
            color_discrete_map={
                "Healthy": "green",
                "Warning": "orange",
                "Critical": "red"
            }
            )
            st.plotly_chart(status_fig, use_container_width=True)

        # =============================
        # DOWNLOAD RESULTS
        # =============================

        csv = output_df.to_csv(index=False)

        st.download_button(
            label="Download Predictions.csv",
            data=csv,
            file_name="Predictions.csv",
            mime="text/csv"
        )