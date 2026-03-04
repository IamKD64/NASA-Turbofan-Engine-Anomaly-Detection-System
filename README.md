# NASA Turbofan Engine Anomaly Detection System (FD002)

## Overview

The NASA Turbofan Engine Anomaly Detection System is an intelligent machine learning application designed to monitor aircraft engine health and detect early warning signs of failure.

Modern aircraft engines operate under complex and varying environmental conditions. Multiple sensors continuously monitor engine parameters such as temperature, pressure, speed, and fuel flow. These sensors generate high-dimensional multivariate time-series data throughout the engine’s operational lifecycle.

This project builds a predictive monitoring system that analyzes these sensor patterns to identify abnormal behavior and detect gradual degradation before engine failure occurs.

The system provides an interactive dashboard that allows users to explore engine behavior, analyze anomalies, and monitor long-term degradation trends using machine learning and deep learning models.

---

## Problem Statement

Modern aircraft turbofan engines operate under complex and varying environmental conditions and are equipped with numerous sensors that continuously monitor engine health parameters such as temperature, pressure, speed, and fuel flow.

These sensors generate high-dimensional multivariate time-series data throughout the engine’s operational life.

Core Problem:

How can abnormal degradation patterns in turbofan engines be detected early and accurately using only unlabeled multivariate sensor data collected under varying operating conditions?

This problem is addressed as an unsupervised anomaly detection task where models must learn normal engine behavior and identify deviations that indicate impending failure.

---

## Objectives

1. Understand and preprocess the FD002 turbofan dataset, including handling multiple operating conditions and sensor measurements.
2. Analyze sensor behavior across engine lifecycles to distinguish normal operation from degradation trends.
3. Model normal engine behavior using machine learning and deep learning techniques without relying on explicit anomaly labels.
4. Detect anomalies by identifying deviations in sensor patterns that indicate abnormal or degraded engine states.
5. Quantify anomaly severity using anomaly scores to reflect the progression of engine health degradation.
6. Evaluate model performance using indirect validation methods such as degradation trends and Remaining Useful Life alignment.
7. Improve interpretability by analyzing sensor contributions to detected anomalies.
8. Design a deployable framework suitable for real-time aircraft engine health monitoring systems.

---

## System Architecture

                     +------------------------------+
                     |   NASA C-MAPSS Dataset       |
                     |        (FD002)               |
                     +--------------+---------------+
                                    |
                                    v
                     +------------------------------+
                     |       Data Preprocessing     |
                     |  - Cleaning                  |
                     |  - Feature Scaling           |
                     |  - Sensor Selection          |
                     +--------------+---------------+
                                    |
                                    v
                     +------------------------------+
                     |       Feature Engineering    |
                     |  - Operating Settings        |
                     |  - Sensor Signals            |
                     |  - Time Series Windows       |
                     +--------------+---------------+
                                    |
                                    v
                     +------------------------------+
                     |        Model Training        |
                     +------+-----------+-----------+
                            |           |
                            |           |
            +---------------+           +----------------+
            |                                            |
            v                                            v
 +-----------------------+                  +-----------------------+
 |     Isolation Forest  |                  |    Dense Autoencoder  |
 |  (Quick Anomaly Scan) |                  |  (Behavior Learning)  |
 +-----------+-----------+                  +-----------+-----------+
             |                                              |
             |                                              |
             +--------------------+-------------------------+
                                  |
                                  v
                     +------------------------------+
                     |      LSTM Autoencoder        |
                     |   (Trend-Based Monitoring)   |
                     +--------------+---------------+
                                    |
                                    v
                     +------------------------------+
                     |        Anomaly Score         |
                     |   Deviation / Reconstruction |
                     +--------------+---------------+
                                    |
                                    v
                     +------------------------------+
                     |      Engine Health Analysis  |
                     +--------------+---------------+
                                    |
                                    v
                     +------------------------------+
                     |   Streamlit Monitoring App   |
                     +--------------+---------------+
                                    |
                                    v
                   +----------------+----------------+
                   |                |                |
                   v                v                v
       +----------------+ +----------------+ +---------------------+
       | Quick Anomaly  | | Behavior Model | | Trend Monitoring    |
       | Scan Dashboard | | Visualization  | | Degradation Trends  |
       +----------------+ +----------------+ +---------------------+
---

## Dataset

The project uses the **NASA C-MAPSS Turbofan Engine Dataset**, specifically the **FD002 subset**.

Dataset characteristics:

- Multiple engines with complete lifecycle data
- Multiple operating conditions
- 21 sensor measurements per cycle
- Time-series engine degradation patterns

Each engine begins with normal operation and gradually degrades until failure.

---

## Features Used

The dataset contains:

Operating Settings
- op_setting_1
- op_setting_2
- op_setting_3

Sensor Measurements
- sensor_1 to sensor_21

Each record represents one **operating cycle of an engine**.

---

## Monitoring Methods Implemented

### 1. Isolation Forest (Quick Anomaly Scan)

Isolation Forest is used to rapidly detect unusual behavior by isolating abnormal data points in the feature space.

Purpose:
- Quickly identify abnormal engine behavior
- Provide an anomaly score per cycle

---

### 2. Dense Autoencoder (Behavior Learning Model)

A deep autoencoder is trained to reconstruct normal engine behavior.

If reconstruction error increases, it indicates deviation from normal patterns.

Purpose:
- Learn normal sensor relationships
- Detect abnormal sensor behavior

---

### 3. LSTM Autoencoder (Trend-Based Monitoring)

An LSTM-based sequence autoencoder analyzes temporal patterns in engine behavior.

Purpose:
- Capture time-dependent degradation trends
- Detect gradual engine deterioration

---

## Dashboard Features

The Streamlit application provides an interactive monitoring dashboard with the following features:

Engine Health Snapshot
- Total operating cycles
- Number of sensors monitored
- Monitoring methods applied

Engine Behavior Analysis
- Quick anomaly detection
- Deep learning behavior modeling
- Time-series degradation monitoring

Interactive Engine Selection
- Users can select any engine from the dataset
- System analyzes that engine's lifecycle behavior

Visualization
- Anomaly score plots
- Reconstruction error graphs
- Degradation trend curves

---

## Project Structure


NASA-Turbofan-Anomaly-Detection
│
├── app.py
├── NASA Turbofan Anomaly Detection.ipynb
├── CMAPS
│ ├── train_FD002.txt
│ ├── test_FD002.txt
│
├── requirements.txt
└── README.md


File Description

app.py  
Streamlit dashboard for interactive engine health monitoring.

NASA Turbofan Anomaly Detection.ipynb  
Notebook used for data exploration and model development.

CMAPS Dataset  
Contains NASA turbofan engine lifecycle data.

requirements.txt  
List of required Python dependencies.

---

## Technologies Used

Python  
Streamlit  
Scikit-learn  
TensorFlow / Keras  
Pandas  
NumPy  
Matplotlib

---

## Running the Project

cd nasa-turbofan-anomaly-detection


Install dependencies


pip install -r requirements.txt


Run the dashboard


streamlit run app.py


The monitoring dashboard will open in your browser.

---

## Applications

Predictive Maintenance  
Aircraft Engine Health Monitoring  
Industrial Equipment Monitoring  
Fault Detection Systems  
Time-Series Anomaly Detection

---

## Learning Outcomes

This project demonstrates:

Unsupervised anomaly detection techniques  
Time-series modeling using LSTM networks  
Deep learning autoencoder architectures  
Predictive maintenance systems  
Interactive ML dashboards using Streamlit

---

## License

This project is intended for educational and research purposes.
