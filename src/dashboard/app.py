"""
Streamlit Dashboard for Anomaly Detection System

Run with: streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Real-time Anomaly Detection Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")
model_status = st.sidebar.success("✅ Model Loaded")
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.95, 0.01)
refresh_rate = st.sidebar.number_input("Refresh Rate (seconds)", 1, 60, 5)

# Main content
col1, col2, col3, col4 = st.columns(4)

# Metrics
with col1:
    st.metric("Total Samples", "10,234", "+156")
with col2:
    st.metric("Anomalies Detected", "47", "+3")
with col3:
    st.metric("Anomaly Rate", "0.46%", "-0.02%")
with col4:
    st.metric("Uptime", "99.8%", "+0.1%")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Real-time Monitoring", "📈 Historical Analysis", "⚙️ Model Info", "🚨 Alerts"])

with tab1:
    st.header("Real-time Sensor Monitoring")
    
    # Generate sample data
    timestamps = pd.date_range(end=datetime.now(), periods=100, freq='1min')
    sensor_data = {
        'timestamp': timestamps,
        'sensor_1': np.random.randn(100) * 5 + 50,
        'sensor_2': np.random.randn(100) * 10 + 100,
        'sensor_3': np.random.randn(100) * 2 + 25
    }
    df = pd.DataFrame(sensor_data)
    
    # Plot
    fig = go.Figure()
    
    for sensor in ['sensor_1', 'sensor_2', 'sensor_3']:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[sensor],
            name=sensor,
            mode='lines'
        ))
    
    fig.update_layout(
        title="Sensor Readings Over Time",
        xaxis_title="Time",
        yaxis_title="Value",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly score
    st.subheader("Anomaly Scores")
    anomaly_scores = np.random.beta(2, 10, 100)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=timestamps,
        y=anomaly_scores,
        name='Anomaly Score',
        fill='tozeroy'
    ))
    fig2.add_hline(y=threshold, line_dash="dash", line_color="red", 
                   annotation_text="Threshold")
    
    fig2.update_layout(
        title="Anomaly Detection Scores",
        xaxis_title="Time",
        yaxis_title="Score",
        height=300
    )
    
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.header("Historical Analysis")
    
    date_range = st.date_input(
        "Select Date Range",
        value=(datetime.now() - timedelta(days=7), datetime.now())
    )
    
    st.info("Historical anomaly analysis will be displayed here")
    
    # Placeholder metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Anomalies (7 days)", "234")
    with col2:
        st.metric("Avg Daily Anomalies", "33.4")

with tab3:
    st.header("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Architecture")
        st.write("**Model:** LSTM Autoencoder")
        st.write("**Input Dimension:** 10 sensors")
        st.write("**Hidden Dimension:** 64")
        st.write("**Latent Dimension:** 32")
        st.write("**Layers:** 2")
    
    with col2:
        st.subheader("Performance Metrics")
        st.write("**Precision:** 94.2%")
        st.write("**Recall:** 91.8%")
        st.write("**F1-Score:** 93.0%")
        st.write("**Inference Time:** 42ms")
    
    st.subheader("Training History")
    epochs = list(range(1, 51))
    train_loss = [0.5 - i*0.008 for i in epochs]
    val_loss = [0.52 - i*0.0075 for i in epochs]
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=epochs, y=train_loss, name='Train Loss'))
    fig3.add_trace(go.Scatter(x=epochs, y=val_loss, name='Val Loss'))
    
    fig3.update_layout(
        title="Training History",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=300
    )
    
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.header("Active Alerts")
    
    # Sample alerts
    alerts = [
        {"time": "2024-02-26 10:15", "equipment": "PUMP-001", "severity": "High", "status": "Active"},
        {"time": "2024-02-26 09:30", "equipment": "COMPRESSOR-002", "severity": "Medium", "status": "Resolved"},
        {"time": "2024-02-26 08:45", "equipment": "TURBINE-003", "severity": "Low", "status": "Active"}
    ]
    
    st.dataframe(
        pd.DataFrame(alerts),
        use_container_width=True,
        hide_index=True
    )
    
    if st.button("🔄 Refresh Alerts"):
        st.rerun()

# Footer
st.markdown("---")
st.caption("Anomaly Detection System v1.0.0 | Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
