import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import matplotlib.pyplot as plt
from datetime import date, datetime
from PIL import Image
import gdown
import joblib
import streamlit as st
import os

st.set_page_config(page_title="Fresh Price Forecast", layout="wide", page_icon="🌾")
@st.cache_resource
def load_model():
    file_id = '1dwfVleT4RwL81sUVN1pX8bjTzs6TL2Uh'  # Your model's Google Drive file ID
    output_path = 'model.pkl'

    # If the model file does not exist locally, download it
    if not os.path.exists(output_path):
        with st.spinner("Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)

    # Load the model from the downloaded file
    model = joblib.load(output_path)
    return model

# Load the model when needed
model = load_model()
st.success("Model loaded successfully!")


# --- Page Config ---

# --- Background Styling with Logo ---
# def add_bg_from_local(image_file):
#     import base64
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
#         background-size: 30%;
#         background-position: right bottom;
#         background-repeat: no-repeat;
#         background-attachment: fixed;
#         background-color: #0e1117;
#         opacity: 0.9;
#     }}
#     .main {{
#         background-color: rgba(14, 17, 23, 0.9);
#     }}
#     h1, h2, h3, h4 {{
#         color: #00ff7f;
#         text-shadow: 1px 1px 2px #000000;
#     }}
#     .css-1d391kg, .css-ffhzg2 {{
#         background-color: rgba(31, 41, 55, 0.9);
#         padding: 1.5em;
#         border-radius: 10px;
#         box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
#     }}
#     .stButton>button {{
#         background-color: #00aa55;
#         color: white;
#         border-radius: 8px;
#         padding: 0.5em 1em;
#         font-weight: bold;
#     }}
#     .stButton>button:hover {{
#         background-color: #008844;
#         color: white;
#     }}
#     .stSelectbox, .stDateInput {{
#         background-color: rgba(255, 255, 255, 0.1);
#     }}
#     .stAlert {{
#         border-radius: 10px;
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
#     )

# # Add your logo.jpg to the same directory as your script
# add_bg_from_local('logo.jpg')

# --- Header ---
col1, col2 = st.columns([1, 3])
with col1:
    st.image("logo.jpg", width=150)  # Larger logo in header
with col2:
    st.markdown("<h1 style='margin-top: 20px;'>Fresh Price</h1>", unsafe_allow_html=True)
    st.markdown("<h4>A Smart Prediction System for Agricultural Commodities</h4>", unsafe_allow_html=True)

st.markdown("---")

# --- About Section ---
with st.expander("📘 About This Project", expanded=False):
    st.markdown("""
    **Fresh Price** is a forecasting web application built to help farmers, traders, and policymakers
    make informed decisions based on the future prices of agricultural commodities. It utilizes the
    **SARIMAX** time-series model to predict prices for the next 5 years using historical monthly data.

    ✨ Key Features:
    - Interactive commodity selection
    - State-wise forecast output
    - Easy-to-read charts
    - RMSE error indicator
    """)

# --- Load the ML Model ---
with st.expander("🔍 Machine Learning Model", expanded=True):
    model = None
    try:
        model = joblib.load("model.pkl")
        st.success("✅ ML Model loaded successfully.")
        st.markdown("""
        <div style="background-color: rgba(0, 170, 85, 0.1); padding: 10px; border-radius: 5px; border-left: 4px solid #00aa55;">
            <p><strong>Model Type:</strong> SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors)</p>
            <p><strong>Training Period:</strong> 2014-2024 monthly data</p>
            <p><strong>Forecast Horizon:</strong> 5 years (60 months)</p>
        </div>
        """, unsafe_allow_html=True)
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
    st.markdown(f"""
    <div style="background-color: rgba(0, 170, 85, 0.1); padding: 10px; border-radius: 5px; border-left: 4px solid #00aa55;">
        <p><strong>Data Period:</strong> {len(df)} months of records</p>
        <p><strong>Commodities Available:</strong> {len(df.columns)-1 if df is not None else 0} items</p>
    </div>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    st.error(f"❌ CSV file not found: `{file_path}`")
except Exception as e:
    st.error(f"❌ Error loading CSV data: {e}")

districts = [
    "Ahmednagar", "Akola", "Amravati", "Aurangabad (Chhatrapati Sambhajinagar)", "Beed", "Bhandara", "Buldhana",
    "Chandrapur", "Dhule", "Gadchiroli", "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur", "Latur", "Mumbai",
    "Mumbai Suburban", "Nagpur", "Nanded", "Nandurbar", "Nashik", "Osmanabad (Dharashiv)", "Palghar", "Parbhani",
    "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara", "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"
]

# Sort the districts alphabetically
districts.sort()

# --- Forecast Section ---
if df is not None:
    try:
        df.set_index("Commodities", inplace=True)
        df = df.T
        df.index = pd.date_range(start="2014-01", periods=len(df), freq="M")
        df = df.ffill()

        st.markdown("---")
        st.subheader("📦 Forecast Parameters")

        # Using columns for better layout
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            st.markdown("<h4 style='color: #00ff7f;'>🧺 Commodity Selection</h4>", unsafe_allow_html=True)
            commodities = df.columns.tolist()
            selected_commodity = st.selectbox("Select agricultural commodity", commodities)
            
        with col2:
            st.markdown("<h4 style='color: #00ff7f;'>🌍 Location</h4>", unsafe_allow_html=True)
            states=df.columns.tolist()
            selected_state = st.selectbox("Select district from maharashtra", districts)
            
        with col3:
            st.markdown("<h4 style='color: #00ff7f;'>📅 Date Selection</h4>", unsafe_allow_html=True)
            selected_date = st.date_input("Choose forecast date", min_value=date.today())

        # Centered button with better styling
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            forecast_button = st.button("🚀 Generate Forecast", use_container_width=True)

        if forecast_button:
            st.markdown("---")
            st.subheader(f"🔮 Forecast Results for {selected_commodity}")
            
            # Info cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div style="background-color: rgba(31, 41, 55, 0.9); padding: 15px; border-radius: 10px; border-left: 4px solid #00aa55;">
                    <h4 style='color: #00ff7f; margin-top: 0;'>Commodity</h4>
                    <p style="font-size: 18px;">{selected_commodity}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style="background-color: rgba(31, 41, 55, 0.9); padding: 15px; border-radius: 10px; border-left: 4px solid #00aa55;">
                    <h4 style='color: #00ff7f; margin-top: 0;'>Location</h4>
                    <p style="font-size: 18px;">{selected_state}</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div style="background-color: rgba(31, 41, 55, 0.9); padding: 15px; border-radius: 10px; border-left: 4px solid #00aa55;">
                    <h4 style='color: #00ff7f; margin-top: 0;'>Forecast Date</h4>
                    <p style="font-size: 18px;">{selected_date}</p>
                </div>
                """, unsafe_allow_html=True)

            # Train SARIMAX
            with st.spinner('🔮 Generating forecast... Please wait...'):
                data = df[selected_commodity]
                model_sarimax = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
                sarimax_model = model_sarimax.fit(disp=False)

                forecast_steps = 60
                forecast = sarimax_model.get_forecast(steps=forecast_steps)
                forecasted_values = forecast.predicted_mean
                forecast_dates = pd.date_range(start="2025-01-01", periods=forecast_steps, freq="M")

                forecast_df = pd.DataFrame({
                    "Date": forecast_dates,
                    f"{selected_commodity}_Forecast": forecasted_values
                })

                # Nearest date
                selected_datetime = pd.to_datetime(selected_date)
                nearest_date = forecast_df["Date"].iloc[(forecast_df["Date"] - selected_datetime).abs().argsort()[:1]].values[0]
                forecast_value = forecast_df.loc[forecast_df["Date"] == nearest_date, f"{selected_commodity}_Forecast"].values[0]

                # Result card
                st.markdown(f"""
                <div style="background-color: rgba(0, 170, 85, 0.2); padding: 20px; border-radius: 10px; border-left: 6px solid #00aa55; margin: 20px 0;">
                    <h3 style='color: #00ff7f; margin-top: 0;'>Predicted Price</h3>
                    <p style="font-size: 24px; font-weight: bold; text-align: center;">₹ {forecast_value:.2f}</p>
                    <p style="text-align: center;">for {selected_commodity} in {selected_state} on {pd.to_datetime(nearest_date).date()}</p>
                </div>
                """, unsafe_allow_html=True)

                # Plot with better styling
                st.markdown("### 📈 Price Trend Forecast")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(data, label="Historical Prices", linewidth=2, color='#00aa55')
                ax.plot(forecast_dates, forecasted_values, color="orange", linestyle='--', label="Forecasted Prices", linewidth=2)
                ax.axvline(x=nearest_date, color="red", linestyle=":", label="Selected Date", linewidth=2)
                
                # Formatting
                ax.set_facecolor('#1f2937')
                fig.patch.set_facecolor('#0e1117')
                ax.set_title(f"{selected_commodity} Price Forecast (2025–2029)", fontsize=16, color='white', pad=20)
                ax.set_xlabel("Year", color='white', fontsize=12)
                ax.set_ylabel("Price (₹)", color='white', fontsize=12)
                ax.tick_params(colors='white')
                ax.grid(color='gray', linestyle=':', alpha=0.3)
                ax.legend(facecolor='#1f2937', edgecolor='none', labelcolor='white')
                
                for spine in ax.spines.values():
                    spine.set_edgecolor('gray')
                
                st.pyplot(fig)

                # Performance metrics
                st.markdown("### 📊 Model Performance")
                train_rmse = np.sqrt(np.mean((data - sarimax_model.fittedvalues) ** 2))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div style="background-color: rgba(31, 41, 55, 0.9); padding: 15px; border-radius: 10px;">
                        <h4 style='color: #00ff7f; margin-top: 0;'>Training RMSE</h4>
                        <p style="font-size: 24px; text-align: center;">{train_rmse:.4f}</p>
                        <p style="font-size: 12px; text-align: center;">Root Mean Square Error</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown("""
                    <div style="background-color: rgba(31, 41, 55, 0.9); padding: 15px; border-radius: 10px;">
                        <h4 style='color: #00ff7f; margin-top: 0;'>Forecast Horizon</h4>
                        <p style="font-size: 24px; text-align: center;">5 Years</p>
                        <p style="font-size: 12px; text-align: center;">60 monthly predictions</p>
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error during forecasting: {e}")
        st.error("Please check your input parameters and try again.")


