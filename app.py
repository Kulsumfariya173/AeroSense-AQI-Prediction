import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AeroSense | AQI Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD COMPLETE ML PIPELINE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_pipeline():
    return joblib.load("data/aqi_pipeline.pkl")


pipeline = joblib.load("models/aqi_pipeline.pkl")

# ============================================================
# EXTRACT MODEL COMPONENTS
# ============================================================

model = pipeline["model"]
encoder = pipeline["encoder"]
scaler = pipeline["scaler"]
cluster_medians = pipeline["cluster_medians"]
cluster_scaler = pipeline["cluster_scaler"]
kmeans = pipeline["kmeans"]
train_medians = pipeline["train_medians"]
config = pipeline["config"]


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

pollutant_cols = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene"
]

numeric_cols = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Year",
    "Month"
]

categorical_cols = [
    "City",
    "Season"
]


# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


# ============================================================
# HEALTH ALERT
# ============================================================

def get_health_alert(category):

    alerts = {

        "Good": {
            "message": (
                "Air quality is considered good. "
                "Air pollution poses little or no risk."
            ),
            "advice": (
                "No special precautions are generally needed."
            )
        },

        "Satisfactory": {
            "message": (
                "Air quality is acceptable, but some pollutants "
                "may affect sensitive individuals."
            ),
            "advice": (
                "Sensitive people should consider reducing "
                "prolonged outdoor exposure if they experience discomfort."
            )
        },

        "Moderate": {
            "message": (
                "Air quality may cause discomfort to sensitive individuals."
            ),
            "advice": (
                "People with respiratory or heart conditions should "
                "consider limiting prolonged outdoor exertion."
            )
        },

        "Poor": {
            "message": (
                "Poor air quality may affect health, "
                "especially for sensitive groups."
            ),
            "advice": (
                "Sensitive individuals should reduce prolonged "
                "or strenuous outdoor activity."
            )
        },

        "Very Poor": {
            "message": (
                "Very poor air quality can cause respiratory "
                "health effects."
            ),
            "advice": (
                "Avoid prolonged or strenuous outdoor activity, "
                "especially if you are sensitive to air pollution."
            )
        },

        "Severe": {
            "message": (
                "Severe air pollution can affect health and "
                "may cause serious respiratory effects."
            ),
            "advice": (
                "Avoid unnecessary outdoor exposure and consider "
                "appropriate protective measures."
            )
        }
    }

    return alerts[category]


# ============================================================
# SEASON
# ============================================================

def get_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Summer"

    elif month in [6, 7, 8, 9]:
        return "Monsoon"

    else:
        return "Post-Monsoon"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🌍 AeroSense")

    st.caption(
        "Cluster-Based AQI Prediction & "
        "Health Alert System"
    )

    st.divider()

    st.subheader("AQI Categories")

    st.success("🟢 Good — 0–50")

    st.info("🔵 Satisfactory — 51–100")

    st.warning("🟡 Moderate — 101–200")

    st.warning("🟠 Poor — 201–300")

    st.error("🔴 Very Poor — 301–400")

    st.error("🟥 Severe — 401+")

    st.divider()

    st.subheader("Model")

    st.caption(
        "Random Forest + K-Means Clustering"
    )

    st.caption(
        "Cluster feature is used as an additional "
        "input to the AQI prediction model."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🌍 Air Quality Index Prediction")

st.subheader(
    "Cluster-Based Machine Learning with Health Alerts"
)

st.write(
    "Enter environmental and pollutant information "
    "to predict the Air Quality Index and receive "
    "a corresponding health alert."
)

st.divider()


# ============================================================
# ENVIRONMENTAL INFORMATION
# ============================================================

st.header("📍 Environmental Information")

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# CITY
# ------------------------------------------------------------

with col1:

    cities = list(
        encoder.categories_[0]
    )

    default_city = (
        cities.index("Delhi")
        if "Delhi" in cities
        else 0
    )

    city = st.selectbox(
        "City",
        cities,
        index=default_city
    )


# ------------------------------------------------------------
# YEAR
# ------------------------------------------------------------

with col2:

    year = st.number_input(
        "Year",
        min_value=2015,
        max_value=2030,
        value=2020,
        step=1
    )


# ------------------------------------------------------------
# MONTH
# ------------------------------------------------------------

with col3:

    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]

    month = st.selectbox(
        "Month",
        range(1, 13),
        format_func=lambda x: month_names[x - 1]
    )


# Automatically determine season
season = get_season(month)

st.info(
    f"📅 Season automatically detected: **{season}**"
)

st.divider()


# ============================================================
# POLLUTANT MEASUREMENTS
# ============================================================

st.header("🧪 Pollutant Measurements")

st.write(
    "Enter pollutant measurements using the same "
    "units used by the training dataset."
)

col1, col2, col3 = st.columns(3)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    pm25 = st.number_input(
        "PM2.5",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

    pm10 = st.number_input(
        "PM10",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

    no = st.number_input(
        "NO",
        min_value=0.0,
        value=10.0,
        step=1.0
    )

    no2 = st.number_input(
        "NO2",
        min_value=0.0,
        value=25.0,
        step=1.0
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    nox = st.number_input(
        "NOx",
        min_value=0.0,
        value=30.0,
        step=1.0
    )

    nh3 = st.number_input(
        "NH3",
        min_value=0.0,
        value=15.0,
        step=1.0
    )

    co = st.number_input(
        "CO",
        min_value=0.0,
        value=1.0,
        step=0.1
    )

    so2 = st.number_input(
        "SO2",
        min_value=0.0,
        value=10.0,
        step=1.0
    )


# ============================================================
# COLUMN 3
# ============================================================

with col3:

    o3 = st.number_input(
        "O3",
        min_value=0.0,
        value=30.0,
        step=1.0
    )

    benzene = st.number_input(
        "Benzene",
        min_value=0.0,
        value=1.0,
        step=0.1
    )

    toluene = st.number_input(
        "Toluene",
        min_value=0.0,
        value=3.0,
        step=0.1
    )


st.divider()


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict AQI",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION PIPELINE
# ============================================================

if predict_button:

    try:

        # ====================================================
        # CREATE INPUT DATAFRAME
        # ====================================================

        input_data = pd.DataFrame([{

            "City": city,

            "PM2.5": pm25,
            "PM10": pm10,
            "NO": no,
            "NO2": no2,
            "NOx": nox,
            "NH3": nh3,
            "CO": co,
            "SO2": so2,
            "O3": o3,
            "Benzene": benzene,
            "Toluene": toluene,

            "Year": year,
            "Month": month,
            "Season": season

        }])


        # ====================================================
        # K-MEANS CLUSTER PREDICTION
        # ====================================================

        cluster_input = (
            input_data[pollutant_cols]
            .copy()
        )

        cluster_input = cluster_input.fillna(
            cluster_medians
        )

        cluster_scaled = (
            cluster_scaler.transform(
                cluster_input
            )
        )

        predicted_cluster = int(
            kmeans.predict(
                cluster_scaled
            )[0]
        )


        # ====================================================
        # NUMERIC PREPROCESSING
        # ====================================================

        numeric_input = (
            input_data[numeric_cols]
            .copy()
        )

        numeric_input = numeric_input.fillna(
            train_medians
        )

        numeric_scaled = (
            scaler.transform(
                numeric_input
            )
        )


        # ====================================================
        # CATEGORICAL PREPROCESSING
        # ====================================================

        categorical_input = (
            input_data[categorical_cols]
            .copy()
        )

        categorical_encoded = (
            encoder.transform(
                categorical_input
            )
        )


        # ====================================================
        # FINAL MODEL INPUT
        # ====================================================

        final_features = np.hstack([

            numeric_scaled,

            categorical_encoded,

            np.array([
                [predicted_cluster]
            ])

        ])


        # ====================================================
        # AQI PREDICTION
        # ====================================================

        predicted_aqi = model.predict(
            final_features
        )[0]

        predicted_aqi = max(
            0,
            round(
                float(predicted_aqi),
                2
            )
        )


        # ====================================================
        # AQI CATEGORY
        # ====================================================

        category = get_aqi_category(
            predicted_aqi
        )


        # ====================================================
        # HEALTH ALERT
        # ====================================================

        health_alert = get_health_alert(
            category
        )


        # ====================================================
        # PREDICTION RESULTS
        # ====================================================

        st.divider()

        st.header("📊 Prediction Results")


        # ----------------------------------------------------
        # LARGE AQI RESULT
        # ----------------------------------------------------

        aqi_col, category_col = st.columns([1, 2])


        with aqi_col:

            st.metric(
                label="Predicted AQI",
                value=f"{predicted_aqi:.2f}"
            )


        with category_col:

            st.metric(
                label="Air Quality Category",
                value=category
            )


        # ----------------------------------------------------
        # AQI SCALE
        # ----------------------------------------------------

        st.subheader("📈 AQI Level")

        st.caption(
            "AQI scale: Good → Satisfactory → Moderate → "
            "Poor → Very Poor → Severe"
        )

        # Progress is capped at 500 for visualization.
        # AQI values above 500 remain displayed correctly
        # in the metric above.
        progress_value = min(
            max(predicted_aqi / 500, 0.0),
            1.0
        )

        st.progress(
            progress_value
        )


        # ----------------------------------------------------
        # AQI RANGE
        # ----------------------------------------------------

        scale1, scale2, scale3, scale4, scale5, scale6 = st.columns(6)

        with scale1:
            st.caption("🟢 Good")
            st.caption("0–50")

        with scale2:
            st.caption("🔵 Satisfactory")
            st.caption("51–100")

        with scale3:
            st.caption("🟡 Moderate")
            st.caption("101–200")

        with scale4:
            st.caption("🟠 Poor")
            st.caption("201–300")

        with scale5:
            st.caption("🔴 Very Poor")
            st.caption("301–400")

        with scale6:
            st.caption("🟥 Severe")
            st.caption("401+")


        st.divider()


        # ----------------------------------------------------
        # CLUSTER RESULT
        # ----------------------------------------------------

        st.subheader("🔬 Pollution Pattern")

        cluster_col1, cluster_col2 = st.columns(2)


        with cluster_col1:

            st.metric(
                "Detected Pollution Cluster",
                f"Cluster {predicted_cluster}"
            )


        with cluster_col2:

            st.caption(
                "The K-Means clustering model identifies "
                "the pollution pattern from the entered "
                "pollutant measurements."
            )


        # ----------------------------------------------------
        # AQI STATUS MESSAGE
        # ----------------------------------------------------

        if category == "Good":

            st.success(
                f"🟢 AQI {predicted_aqi:.2f} — {category}"
            )

        elif category == "Satisfactory":

            st.info(
                f"🔵 AQI {predicted_aqi:.2f} — {category}"
            )

        elif category == "Moderate":

            st.warning(
                f"🟡 AQI {predicted_aqi:.2f} — {category}"
            )

        elif category == "Poor":

            st.warning(
                f"🟠 AQI {predicted_aqi:.2f} — {category}"
            )

        elif category == "Very Poor":

            st.error(
                f"🔴 AQI {predicted_aqi:.2f} — {category}"
            )

        else:

            st.error(
                f"🟥 AQI {predicted_aqi:.2f} — {category}"
            )


        # ====================================================
        # HEALTH ALERT
        # ====================================================

        st.divider()

        st.header("❤️ Health Alert")


        st.write(
            health_alert["message"]
        )


        st.info(
            f"💡 Recommended Action: "
            f"{health_alert['advice']}"
        )


        # ====================================================
        # PREDICTION DETAILS
        # ====================================================

        with st.expander(
            "🔎 View Prediction Details"
        ):

            detail1, detail2 = st.columns(2)


            with detail1:

                st.write(
                    f"**Location:** {city}"
                )

                st.write(
                    f"**Year:** {year}"
                )

                st.write(
                    f"**Month:** {month_names[month - 1]}"
                )

                st.write(
                    f"**Season:** {season}"
                )


            with detail2:

                st.write(
                    f"**Detected Cluster:** "
                    f"Cluster {predicted_cluster}"
                )

                st.write(
                    f"**Predicted AQI:** "
                    f"{predicted_aqi:.2f}"
                )

                st.write(
                    f"**AQI Category:** {category}"
                )


        # ====================================================
        # POLLUTANT SUMMARY
        # ====================================================

        with st.expander(
            "🧪 View Pollutant Values"
        ):

            pollutant_values = pd.DataFrame({

                "Pollutant": pollutant_cols,

                "Value": [
                    pm25,
                    pm10,
                    no,
                    no2,
                    nox,
                    nh3,
                    co,
                    so2,
                    o3,
                    benzene,
                    toluene
                ]

            })


            st.dataframe(
                pollutant_values,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AeroSense • Cluster-Based AQI Prediction "
    "Using Machine Learning with a Health Alert System"
)

st.caption(
    "Random Forest + K-Means Clustering"
)