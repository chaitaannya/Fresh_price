import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import matplotlib.pyplot as plt
from datetime import date, datetime

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
        st.subheader("📦 Select Forecast Options")

        # Select options
        col1, col2, col3 = st.columns(3)

        with col1:
            commodities = df.columns.tolist()
            selected_commodity = st.selectbox("🧺 Commodity", commodities)

        with col2:
            selected_state = st.selectbox("🌍 State", ["Punjab", "Maharashtra", "Karnataka", "UP", "Tamil Nadu"])  # example

        with col3:
            selected_date = st.date_input("📅 Date", min_value=date.today())

        forecast_button = st.button("📊 Predict")

        if forecast_button:
            st.markdown("---")
            st.subheader(f"🔮 Forecast for: {selected_commodity}")
            st.markdown(f"**📅 Date Chosen:** `{selected_date}`")
            st.markdown(f"**📍 State Selected:** `{selected_state}`")

            # Train SARIMAX
            data = df[selected_commodity]
            model_sarimax = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
            sarimax_model = model_sarimax.fit(disp=False)

            # Forecast for next 60 months
            forecast_steps = 60
            forecast = sarimax_model.get_forecast(steps=forecast_steps)
            forecasted_values = forecast.predicted_mean
            forecast_dates = pd.date_range(start="2025-01-01", periods=forecast_steps, freq="M")

            forecast_df = pd.DataFrame({
                "Date": forecast_dates,
                f"{selected_commodity}_Forecast": forecasted_values
            })

            # Find nearest forecasted date
            selected_datetime = pd.to_datetime(selected_date)
            nearest_date = forecast_df["Date"].iloc[(forecast_df["Date"] - selected_datetime).abs().argsort()[:1]].values[0]
            forecast_value = forecast_df.loc[forecast_df["Date"] == nearest_date, f"{selected_commodity}_Forecast"].values[0]

            st.success(f"📈 Predicted price for **{selected_commodity}** in **{selected_state}** on **{pd.to_datetime(nearest_date).date()}** is: ₹ `{forecast_value:.2f}`")

            # Plot full forecast with selected date marked
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data, label=f"Historical {selected_commodity} Prices")
            ax.plot(forecast_dates, forecasted_values, label="Forecasted Prices", color="orange")
            ax.axvline(x=nearest_date, color="red", linestyle="--", label="Selected Date")
            ax.set_title(f"{selected_commodity} Price Forecast (2025–2029)", fontsize=14)
            ax.set_xlabel("Year")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

            # Training RMSE
            train_rmse = np.sqrt(np.mean((data - sarimax_model.fittedvalues) ** 2))
            st.info(f"📉 Training RMSE: `{train_rmse:.4f}`")

    except Exception as e:
        st.error(f"⚠️ Error during forecasting: {e}")
