import streamlit as st
import plotly.graph_objects as go
from core_ml import load_data, train_models, predict_price


# ============================
# PAGE CONFIG + THEME
# ============================
st.set_page_config(page_title="Netflix Predictor", layout="wide")

st.markdown("""
<style>
.main { background-color: #0d1117; }
h1, h2, h3, h4, h5, h6, label, p { color: #e6edf3 !important; }
.stNumberInput > div > input {
    background-color: #161b22 !important; color:white !important; border-radius:6px; border:1px solid #30363d;
}
.stButton>button {
    background-color:#238636; color:white; padding:10px 25px;
    border-radius:8px; border:none; font-size:16px;
}
.stButton>button:hover { background-color:#2ea043; }
.card {
    background-color: #161b22; padding:20px; border-radius:12px;
    border:1px solid #30363d; text-align:center;
}
</style>
""", unsafe_allow_html=True)


# ============================
# LOAD DATA + TRAIN MODELS
# ============================
netflix = load_data()
model_scores, best_name, best_r2, scaler = train_models(netflix)


# ============================
# SIDEBAR
# ============================
st.sidebar.header("⚙️ Prediction Settings")
model_choice = st.sidebar.selectbox("Choose Model", list(model_scores.keys()))
st.sidebar.success(f"Best Model: {best_name} (R² = {best_r2:.4f})")


# ============================
# HEADER TITLE
# ============================
st.markdown("<h1 style='text-align:center;'>📈 Netflix Stock Predictor</h1>", unsafe_allow_html=True)


# ============================
# INPUT FORM
# ============================
col1, col2, col3 = st.columns(3)
with col1:
    Open = st.number_input("Open Price")
    High = st.number_input("High Price")
with col2:
    Low = st.number_input("Low Price")
    Volume = st.number_input("Volume")
with col3:
    Year = st.number_input("Year", min_value=2002, max_value=2023)
    Month = st.number_input("Month", min_value=1, max_value=12)
    Day = st.number_input("Day", min_value=1, max_value=31)


# ============================
# PREDICTION ACTION
# ============================
inputs = {
    "Open": Open,
    "High": High,
    "Low": Low,
    "Volume": Volume,
    "Year": Year,
    "Month": Month,
    "Day": Day,
    "HL_diff": High - Low,
    "Price_range": High - Open
}

if st.button("🔮 Predict Close Price"):
    price = predict_price(model_scores, scaler, inputs, model_choice)
    
    st.markdown(
        f"""
        <div class="card">
            <h2>Predicted Close Price</h2>
            <h1 style="color:#2ea043;">${price:.2f}</h1>
            <p>Using {model_choice}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================
# PRICE TREND CHART
# ============================
st.markdown("## 📊 Historical Close Price Trend")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=netflix["Date"],
    y=netflix["Close"],
    mode="lines",
    line=dict(color="#2ea043")
))

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Close Price",
    plot_bgcolor="#0d1117",
    paper_bgcolor="#0d1117",
    font=dict(color="white")
)

st.plotly_chart(fig, use_container_width=True)
