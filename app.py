import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Fresh Price Forecast", layout="centered")

# --- Title ---
st.markdown("<h1 style='text-align: center; color: green;'>🌾 Fresh Price</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>A Smart Prediction System for Agricultural Commodities</h4>", unsafe_allow_html=True)
st.markdown("---")

# --- Load the ML Model ---
with st.expander("🔍 Load Machine Learning Model", expanded=True):
    model = None
    try:
        model = joblib.load("model.pkl")
        st.success("✅ ML Model loaded successfully.")
    except FileNotFoundError:
        st.error("❌ ML model file not found! Please upload the model file.")
    except Exception as e:
        st.error(f"❌ Error loading ML model: {e}")

# --- Load CSV Data ---
file_path = "monthly_data.csv"
df = None
try:
    df = pd.read_csv(file_path)
    st.success("📈 CSV data loaded successfully.")
except FileNotFoundError:
    st.error(f"❌ CSV file not found: `{file_path}`")
except Exception as e:
    st.error(f"❌ Error loading CSV data: {e}")

# --- Forecast Section ---
if df is not None:
    try:
        # Preprocessing
        df.set_index("Commodities", inplace=True)
        df = df.T
        df.index = pd.date_range(start="2014-01", periods=len(df), freq="M")
        df = df.ffill()

        st.markdown("---")
        st.subheader("📦 Select a Commodity")

        col1, col2 = st.columns([4, 1])
        with col1:
            commodities = df.columns.tolist()
            selected_commodity = st.selectbox("", commodities)

        with col2:
            forecast_button = st.button("📊 Predict ")

        if forecast_button:
            st.markdown("---")
            st.subheader(f"🔮 {selected_commodity} Price Forecast (2025–2029)")

            data = df[selected_commodity]

            model_sarimax = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
            sarimax_model = model_sarimax.fit(disp=False)

            forecast = sarimax_model.get_forecast(steps=60)
            forecasted_values = forecast.predicted_mean
            forecast_years = pd.date_range(start="2025-01", periods=60, freq="M")

            forecast_df = pd.DataFrame({
                "Year": forecast_years,
                f"{selected_commodity}_Forecast": forecasted_values
            })

            st.dataframe(forecast_df, use_container_width=True)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data, label=f"Actual {selected_commodity} Prices")
            ax.plot(forecast_years, forecasted_values, label="Forecasted Prices", color="orange")
            ax.set_title(f"{selected_commodity} Price Forecast (2025–2029)", fontsize=14)
            ax.set_xlabel("Year")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

            train_rmse = np.sqrt(np.mean((data - sarimax_model.fittedvalues) ** 2))
            st.info(f"📉 Training RMSE: `{train_rmse:.4f}`")

    except Exception as e:
        st.error(f"⚠️ Error during forecasting: {e}")

# --- Optional ML Prediction UI (Template, Still Commented) ---
# if model is not None:
#     st.markdown("---")
#     st.subheader("🤖 Predict with ML Model")

#     col1, col2 = st.columns(2)
#     with col1:
#         feature1 = st.number_input("Feature 1", value=0.0)
#     with col2:
#         feature2 = st.number_input("Feature 2", value=0.0)

#     features = np.array([[feature1, feature2]])

#     if st.button("🔍 Predict"):
#         try:
#             prediction = model.predict(features)
#             st.success(f"🎯 Predicted Price: {prediction[0]}")
#         except Exception as e:
#             st.error(f"Prediction error: {e}")
