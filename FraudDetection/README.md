# 🚨 Fraud Detection -- Production-Grade ML System

## Overview

This repository contains a **production-grade fraud detection system**
built using real-world Machine Learning engineering practices.

The project is designed to be: - Reproducible - Stable across runs -
Honest with respect to data limitations - Deployable as a live
application

Rather than focusing on flashy demos, the emphasis is on **correctness,
robustness, and explainability**, which are critical in real fraud
detection systems.

------------------------------------------------------------------------

## Problem Description

The objective is to classify credit card transactions as **fraudulent**
or **legitimate**.

Fraud detection is challenging because: - Fraud cases are extremely rare
(high class imbalance) - False negatives are costly - Models must be
stable and reproducible - Training and inference pipelines must remain
consistent

------------------------------------------------------------------------

## Dataset

**Credit Card Fraud Dataset**

-   Public, anonymized dataset
-   \~284,000 transactions
-   \~0.17% fraud rate
-   Features:
    -   `Time`: Seconds elapsed since the first transaction
    -   `Amount`: Transaction amount
    -   `V1–V28`: PCA-transformed behavioral features
    -   `Class`: Target label (0 = Legitimate, 1 = Fraud)

### Dataset Handling

The dataset is **not committed to GitHub** due to size limits.

Instead, the pipeline **automatically downloads the dataset at runtime**
if it is not present locally.\
This approach keeps the repository clean while ensuring reproducibility.

------------------------------------------------------------------------

## Project Structure

    fraud-detection-ml/
    │
    ├── data/
    │   ├── raw/creditcard.csv
    │   └── processed/processed_data.csv
    │
    ├── models/
    │   └── fraud_model.pkl
    │
    ├── src/
    │   ├── config.py
    │   ├── data_loader.py
    │   ├── data_validation.py
    │   ├── feature_engineering.py
    │   └── train.py
    │
    ├── streamlit_app.py
    ├── requirements.txt
    ├── README.md
    └── system_design.md

------------------------------------------------------------------------

## Data Processing

Raw transaction data is validated and cleaned before training: - Schema
validation - Duplicate removal - Invalid amount filtering

Processed data is stored separately to avoid data leakage.

------------------------------------------------------------------------

## Feature Engineering

Fraud is a behavioral problem rather than a purely monetary one.

In addition to PCA features, the following contextual features are
engineered: - Log-transformed transaction amount - High-amount
indicator - Transaction hour - Amount deviation from average spending

These features improve robustness and interpretability.

------------------------------------------------------------------------

## Model

A **Random Forest classifier** is used because it: - Captures non-linear
relationships - Handles noisy data well - Works effectively with
imbalanced datasets - Produces stable predictions

Class weighting is applied to mitigate imbalance.

------------------------------------------------------------------------

## Evaluation

Accuracy is avoided due to class imbalance.

Primary evaluation metric: - **ROC-AUC**

This metric reflects the model's ability to rank fraudulent transactions
higher than legitimate ones.

------------------------------------------------------------------------

## Stability and Reproducibility

The system enforces: - Fixed random seeds - Centralized configuration -
Controlled model complexity - Strict feature schema consistency

These measures prevent unstable predictions and silent inference bugs.

------------------------------------------------------------------------

## Live Deployment (Streamlit)

The model is deployed using **Streamlit** for interactive inference.

Key points: - Most transactions are predicted as **Legitimate** - This
mirrors real-world fraud systems where fraud is rare - PCA features are
neutral defaults due to lack of upstream behavioral pipelines

This behavior is **intentional and correct**.

------------------------------------------------------------------------

## How to Run Locally

### 1. Clone the Repository

``` bash
git clone <your-repo-url>
cd fraud-detection-ml
```

### 2. Create and Activate Virtual Environment

**macOS / Linux**

``` bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**

``` bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Pipeline

``` bash
python data_validation.py
python train.py
```

### 5. Launch Live App

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## System Design

A high-level system design is provided in `system_design.md`,
covering: - Data ingestion - Training pipeline - Inference flow -
Monitoring concepts - Retraining strategy

------------------------------------------------------------------------

## Known Limitations

-   PCA features are anonymized and not directly interpretable
-   Live demo does not include full behavioral pipelines
-   Production systems would integrate additional user and device
    signals

These limitations reflect dataset constraints rather than modeling
flaws.

------------------------------------------------------------------------

## Conclusion

This project demonstrates: - End-to-end ML pipeline design -
Production-oriented thinking - Honest handling of imbalanced data -
Stable and reproducible inference - Live deployment with clean data
management

The focus is on **engineering quality and realism**, not artificial demo
behavior.

------------------------------------------------------------------------

## Author

Pratyaksh Bhandari
