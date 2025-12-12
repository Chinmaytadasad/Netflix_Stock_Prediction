#### **📈 Netflix Stock Prediction System**

A machine learning–powered web application that predicts Netflix (NFLX) stock closing prices based on historical financial data and engineered features.

###🚀 Features

Trains four regression models: Linear Regression, Ridge, Lasso, Random Forest

Performs feature engineering (HL_diff, Price_range, Year–Month–Day extraction)

Includes preprocessing, scaling, and model evaluation (R² score)

Built with a clean separation between ML logic (ml_core.py) and UI (app.py)

Interactive Streamlit dashboard for real-time predictions

Visual trend analysis using Plotly charts

###🛠️ Tech Stack

Machine Learning: Scikit-Learn, Pandas, NumPy

Frontend / UI: Streamlit, Plotly

Backend Logic: Python

Tools: Git, VS Code

📂 Project Structure
📁 Netflix-Stock-Prediction
│── app.py                # Streamlit UI
│── ml_core.py            # ML pipeline, preprocessing, model training, prediction logic
│── NFLX.csv              # Dataset
│── requirements.txt      # Dependencies
└── .streamlit/config.toml (optional theme)

###⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/netflix-stock-prediction.git
cd netflix-stock-prediction

###2️⃣ Install dependencies
pip install -r requirements.txt

###3️⃣ Run the application
streamlit run app.py


The dashboard will open at:

➡️ http://localhost:8501/

###📊 Model Details

The system trains and compares four models:

Model	Description
Linear Regression	Baseline regression model
Ridge Regression	Adds L2 regularization
Lasso Regression	Performs L1 regularization + feature selection
Random Forest	Ensemble method, handles non-linear patterns

The best model is selected based on R² score.

###🧠 How Prediction Works

User enters stock values (Open, High, Low, Volume, Date).

Features are engineered:

HL_diff = High - Low

Price_range = High - Open

Regression models or Random Forest generate a price estimate.

The result appears instantly on the dashboard.

###📉 Stock Trend Visualization

The app includes:

Historical Close Price chart

Interactive Plotly visualization

Date-based filtering (optional extension)

###✨ Future Enhancements

Add LSTM or ARIMA for time-series forecasting

Connect live stock API for real-time predictions

Add technical indicators (SMA, EMA, RSI, MACD)

Enhance UI with advanced charting features

###🤝 Contributing

Pull requests and suggestions are welcome!
If you’d like to contribute, feel free to open an issue.

###📬 Contact

Author: Chinmay Tadasad
📧 chinmaytadasad1@gmail.com

🔗 LinkedIn: linkedin.com/in/chinmay-tadasad
💻 GitHub: github.com/Chinmaytadasad
