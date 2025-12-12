import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import streamlit as st


# ======================================
# LOAD DATA
# ======================================
@st.cache_data
def load_data():
    netflix = pd.read_csv("NFLX.csv")

    if "Adj Close" in netflix.columns:
        netflix.drop("Adj Close", axis=1, inplace=True)

    netflix["Date"] = pd.to_datetime(netflix["Date"])
    netflix["Year"] = netflix["Date"].dt.year
    netflix["Month"] = netflix["Date"].dt.month
    netflix["Day"] = netflix["Date"].dt.day

    netflix["HL_diff"] = netflix["High"] - netflix["Low"]
    netflix["Price_range"] = netflix["High"] - netflix["Open"]

    return netflix


# ======================================
# TRAIN MODELS
# ======================================
@st.cache_resource
def train_models(netflix):
    X = netflix.drop(["Close", "Date"], axis=1)
    y = netflix["Close"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=2)
    }

    model_scores = {}
    best_model = None
    best_r2 = -np.inf
    best_name = None

    for name, model in models.items():
        if "Regression" in name:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        model_scores[name] = (model, r2)

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    return model_scores, best_name, best_r2, scaler


# ======================================
# PREDICTION FUNCTION
# ======================================
def predict_price(model_scores, scaler, inputs, model_choice):
    df = pd.DataFrame([inputs])
    model, _ = model_scores[model_choice]

    if "Regression" in model_choice:
        df_scaled = scaler.transform(df)
        return model.predict(df_scaled)[0]
    else:
        return model.predict(df)[0]
