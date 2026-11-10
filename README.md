# AeroSense – Cluster-Based Air Quality Index (AQI) Prediction Using Machine Learning with a Health Alert System

## Overview

AeroSense is a machine learning-based air quality prediction system designed to estimate the Air Quality Index (AQI) using pollutant measurements and provide health alerts based on predicted air quality levels.

The project uses historical air quality data from Indian cities to analyze pollution patterns, train machine learning models, and predict AQI values. It also incorporates K-Means clustering to explore different air pollution patterns.

The application is developed using Python and Streamlit, providing an interactive interface for AQI prediction and health-related information.

## Key Features

- **AQI Prediction:** Predicts Air Quality Index values using machine learning.
- **Machine Learning Models:** Implements Linear Regression, Random Forest, and XGBoost.
- **K-Means Clustering:** Groups air quality observations based on pollution characteristics.
- **Health Alert System:** Provides health-related alerts based on predicted AQI categories.
- **Data Analysis:** Examines pollutant distributions, correlations, and city-wise air quality patterns.
- **Interactive Dashboard:** Provides a user-friendly interface using Streamlit.
- **Model Evaluation:** Compares machine learning models using standard regression metrics.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Clustering | K-Means |
| Application | Streamlit |
| Development Environment | Jupyter Notebook, VS Code |

## Dataset

**Dataset:** Air Quality Data in India (2015–2020)

**Source:** [Kaggle – Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

The dataset contains historical air pollution measurements collected across Indian cities.

Important features include:

- City
- Date
- PM2.5
- PM10
- NO
- NO2
- NOx
- NH3
- CO
- SO2
- O3
- Benzene
- Toluene
- Xylene
- AQI
- AQI_Bucket

The AQI value is used as the prediction target.

## Machine Learning Models

The project evaluates the following models:

1. **Linear Regression:** Used as a baseline regression model.
2. **Random Forest Regressor:** Used to predict AQI through an ensemble of decision trees.
3. **XGBoost Regressor:** Used for gradient-boosted regression.
4. **K-Means Clustering:** Used to identify different air quality patterns.
5. **Random Forest with Cluster Feature:** Evaluates whether adding cluster information improves AQI prediction.
6. **Cluster-Specific Random Forest:** Evaluates separate Random Forest models for different clusters.

## Model Performance

The following results were obtained during model evaluation:

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 29.98 | 56.27 | 0.8271 |
| Random Forest | 20.32 | 40.41 | 0.9108 |
| XGBoost | 21.15 | 42.28 | 0.9024 |
| Random Forest + Cluster | 20.30 | 40.39 | 0.9109 |
| Cluster-Specific Random Forest | 20.63 | 41.48 | 0.9061 |

Random Forest achieved strong prediction performance among the evaluated models. Adding the cluster feature produced only a marginal improvement over the standard Random Forest model.

## AQI Categories

The system uses the following AQI categories:

| AQI Range | Category |
|---|---|
| 0–50 | Good |
| 51–100 | Satisfactory |
| 101–200 | Moderate |
| 201–300 | Poor |
| 301–400 | Very Poor |
| Above 400 | Severe |

The health alert system uses the predicted AQI category to communicate air quality conditions and general health precautions.

**Note:** The alerts are informational and are not a substitute for professional medical advice.

## Project Structure

```text
AeroSense-AQI-Prediction/
│
├── app.py
│
├── models/
│   └── Trained model and preprocessing files
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── raw/
│   └── city_day.csv
│
├── processed/
│   ├── model_data_raw.csv
│   └── model_data_engineered.csv
│
├── results/
│   ├── model_comparison.csv
│   ├── final_feature_importance.csv
│   └── aqi_prediction_alert_results.csv
│
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AeroSense-AQI-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd AeroSense-AQI-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## Evaluation Metrics

The machine learning models are evaluated using:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted AQI values.
- **Root Mean Squared Error (RMSE):** Measures prediction error while giving greater weight to larger errors.
- **R² Score:** Measures how well the model explains variation in AQI values.

## Future Enhancements

- Integrate real-time air quality data through APIs.
- Expand the system to cover additional cities.
- Improve prediction accuracy through advanced machine learning techniques.
- Introduce time-series forecasting for future AQI trends.
- Enhance the health alert system with more detailed recommendations.
- Deploy the application for wider public access.

## Conclusion

AeroSense demonstrates the application of machine learning and clustering techniques for air quality analysis and AQI prediction. By combining predictive modelling with a health alert system and an interactive Streamlit application, the project aims to make air quality information more accessible and understandable.

## Author

**Kulsum Fariya**

Department of Computer Science – Artificial Intelligence and Machine Learning

Presidency University

## License

This project is intended for educational and research purposes.
