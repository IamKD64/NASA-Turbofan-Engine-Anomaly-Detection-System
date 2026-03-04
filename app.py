# =====================================================
# IMPORTS
# =====================================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Aircraft Engine Health Monitoring System",
    layout="wide",
    page_icon="✈️"
)

# =====================================================
# GLOBAL UI STYLES
# =====================================================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0b1220, #020617 70%);
    font-family: Inter, sans-serif;
    color: #e5e7eb;
}
.hero {
    padding: 40px;
    border-radius: 22px;
    background: linear-gradient(135deg, #020617, #020617cc);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 30px;
}
.hero h1 {
    font-size: 42px;
    font-weight: 800;
}
.hero p {
    font-size: 18px;
    color: #94a3b8;
}
.card {
    background: linear-gradient(180deg, rgba(15,23,42,0.95), rgba(2,6,23,0.95));
    border-radius: 18px;
    padding: 24px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
}
.metric-title {
    font-size: 14px;
    color: #94a3b8;
}
.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #38bdf8;
}
.info {
    color: #cbd5f5;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO SECTION
# =====================================================
st.markdown("""
<div class="hero">
    <h1>✈️ Aircraft Engine Health Monitoring</h1>
    <p>
        An intelligent system that continuously monitors aircraft engines and
        highlights <b>early warning signs</b> before a failure occurs.
        <br><br>
        Designed for <b>engineers, managers, and decision-makers</b> — no technical background required.
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    col_names = (
        ["engine_id", "cycle",
         "op_setting_1", "op_setting_2", "op_setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    train = pd.read_csv("CMAPS/train_FD002.txt", sep=r"\s+", header=None)
    test  = pd.read_csv("CMAPS/test_FD002.txt", sep=r"\s+", header=None)

    train.columns = col_names
    test.columns  = col_names
    return train, test

train_df, test_df = load_data()

# =====================================================
# SIDEBAR (USER FRIENDLY)
# =====================================================
st.sidebar.header("🛠️ Monitoring Controls")

engine_id = st.sidebar.selectbox(
    "Choose an Engine",
    sorted(train_df["engine_id"].unique())
)

st.sidebar.markdown(
    "This dashboard analyzes the selected engine and highlights "
    "any unusual behavior compared to healthy engines."
)

# =====================================================
# FEATURE PREPARATION
# =====================================================
sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]
op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
feature_cols = op_cols + sensor_cols

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df[feature_cols])

engine_data = train_df[train_df["engine_id"] == engine_id].copy()
X_engine = scaler.transform(engine_data[feature_cols])

# =====================================================
# HEALTH SNAPSHOT
# =====================================================
st.markdown("### 📊 Engine Health Snapshot")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Total Operating Cycles</div>
        <div class="metric-value">{engine_data.shape[0]}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Sensors Monitored</div>
        <div class="metric-value">{len(sensor_cols)}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
        <div class="metric-title">Monitoring Methods</div>
        <div class="metric-value">3</div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# MODELS (HIDDEN COMPLEXITY)
# =====================================================
@st.cache_resource
def train_isolation_forest(X):
    model = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
    model.fit(X)
    return model

@st.cache_resource
def train_dense_ae(X):
    inp = Input(shape=(X.shape[1],))
    x = Dense(64, activation="relu")(inp)
    x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(X.shape[1])(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, X, epochs=30, batch_size=256, validation_split=0.1,
              callbacks=[EarlyStopping(patience=5)], verbose=0)
    return model

def create_sequences(data, seq_len=30):
    return np.array([data[i:i+seq_len] for i in range(len(data)-seq_len)])

@st.cache_resource
def train_lstm_ae(train_df):
    seqs = []
    for _, df_e in train_df.groupby("engine_id"):
        X_e = scaler.transform(df_e[feature_cols])
        seqs.append(create_sequences(X_e))
    X_seq = np.vstack(seqs)

    inp = Input(shape=(X_seq.shape[1], X_seq.shape[2]))
    x = LSTM(64)(inp)
    x = RepeatVector(X_seq.shape[1])(x)
    x = LSTM(64, return_sequences=True)(x)
    out = TimeDistributed(Dense(X_seq.shape[2]))(x)

    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_seq, X_seq, epochs=20, batch_size=128,
              validation_split=0.1,
              callbacks=[EarlyStopping(patience=5)], verbose=0)
    return model

# =====================================================
# ANALYSIS TABS (FRIENDLY)
# =====================================================
st.markdown("### 🔍 Engine Behavior Analysis")

tab1, tab2, tab3 = st.tabs(
    ["🚨 Quick Anomaly Scan", "🧠 Behavior Learning Model", "⏱️ Trend-Based Monitoring"]
)

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Purpose:** Quickly highlights unusual engine behavior.")
    model = train_isolation_forest(X_train_scaled)
    scores = model.decision_function(X_engine)
    fig, ax = plt.subplots()
    ax.plot(engine_data["cycle"], scores)
    ax.invert_yaxis()
    ax.set_xlabel("Operating Cycle")
    ax.set_ylabel("Unusual Behavior Score")
    st.pyplot(fig)
    st.info("Lower scores indicate higher chances of abnormal behavior.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Purpose:** Learns normal engine behavior and flags deviations.")
    model = train_dense_ae(X_train_scaled)
    recon = model.predict(X_engine)
    mse = np.mean((X_engine - recon)**2, axis=1)
    fig, ax = plt.subplots()
    ax.plot(engine_data["cycle"], mse)
    ax.set_xlabel("Operating Cycle")
    ax.set_ylabel("Deviation Level")
    st.pyplot(fig)
    st.info("Rising deviation indicates wear or degradation.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Purpose:** Detects gradual degradation over time.")
    model = train_lstm_ae(train_df)
    X_seq = create_sequences(X_engine)
    recon = model.predict(X_seq)
    mse = np.mean((X_seq - recon)**2, axis=(1,2))
    fig, ax = plt.subplots()
    ax.plot(mse)
    ax.set_xlabel("Time Window")
    ax.set_ylabel("Trend Deviation")
    st.pyplot(fig)
    st.info("Helps predict issues before failure occurs.")
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
---
<center>
<b>Aircraft Engine Health Monitoring System</b><br>
Early Fault Detection · Predictive Maintenance<br>
Built using Machine Learning & Deep Learning<br>
NASA C-MAPSS Dataset
</center>
""", unsafe_allow_html=True)
