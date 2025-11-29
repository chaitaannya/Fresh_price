# Fresh_price: Agricultural Commodity Price Prediction 🌾💰

**Fresh_price** is a Machine Learning project designed to predict the **Modal Price** (the most common market price) of agricultural commodities. By analyzing historical data from various Agricultural Produce Market Committees (APMC), the model helps in estimating crop prices based on location, timing, and market arrival quantities.

## 📝 Project Overview

This project processes agricultural data to train regression models. It compares three different algorithms to find the most accurate predictor for crop prices.

### The Workflow:
1.  **Data Ingestion**: Loads historical commodity data.
2.  **Preprocessing**: Handles date conversion and missing values.
3.  **Feature Engineering**: 
    * **Scaling**: Normalizes numerical values (prices, arrivals) using `MinMaxScaler`.
    * **Encoding**: Converts categorical text (Commodity, APMC, District) into numbers using `LabelEncoder`.
4.  **Modeling**: Trains and evaluates OLS Regression, Decision Trees, and Random Forest.
5.  **Visualization**: Generates insights through heatmaps, scatter plots, and bar charts.
6.  **Export**: Saves the best-performing model for deployment.

## 🛠️ Tech Stack

* **Language**: Python 3.x
* **Data Manipulation**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn (sklearn), Statsmodels
* **Visualization**: Matplotlib, Seaborn
* **Model Serialization**: Joblib

## 📊 Dataset Features

The model is trained on `Agriculture_commodities_dataset.csv` utilizing the following features:

| Feature | Description |
| :--- | :--- |
| `APMC` | Agricultural Produce Market Committee location |
| `Commodity` | Type of crop (e.g., Wheat, Gram, Soybean) |
| `Year` / `Month` | Time of sale |
| `District_name` | District of the market |
| `State_name` | State of the market |
| `Arrivals_in_qtl` | Quantity of produce arrived (in Quintals) |
| `Min_price` | Minimum market price |
| `Max_price` | Maximum market price |
| **Target** | **`Modal_price`** (The price to be predicted) |

## 🤖 Model Performance

The project evaluates three distinct models. Here are the results based on the testing data ($R^2$ Score):

| Model | R² Score | Performance |
| :--- | :--- | :--- |
| **OLS Regression** | ~0.46 | Low accuracy (Linear relationship assumption weak) |
| **Decision Tree** | ~0.85 | Good accuracy, captures non-linear patterns |
| **Random Forest** | **~0.88** | **Best Performance (Selected Model)** |

*The Random Forest Regressor (500 estimators) was selected and saved as `model.pkl`.*

## 📈 Visualizations

The notebook includes Exploratory Data Analysis (EDA) to understand market trends:
* **Correlation Heatmap**: Shows relationships between price points and arrival quantities.
* **Scatter Plots**: Visualizes the spread between Minimum and Maximum prices.
* **Bar Charts**: Analyzes commodity frequency per month.
