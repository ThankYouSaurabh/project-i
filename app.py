import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")


# ----------------------------
# Data utilities
# ----------------------------
@st.cache_data
def load_data_from_paths(calories_path: str, exercise_path: str) -> pd.DataFrame:
    calories = pd.read_csv(calories_path)
    exercise = pd.read_csv(exercise_path)
    return build_dataset(calories, exercise)

@st.cache_data
def load_data_from_uploads(calories_file, exercise_file) -> pd.DataFrame:
    calories = pd.read_csv(calories_file)
    exercise = pd.read_csv(exercise_file)
    return build_dataset(calories, exercise)

def build_dataset(calories: pd.DataFrame, exercise: pd.DataFrame) -> pd.DataFrame:
    # Merge
    df = exercise.merge(calories, on="User_ID", how="inner").copy()

    # Basic validation
    required_cols = {"Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # BMI
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    df["BMI"] = df["BMI"].round(2)

    # Select relevant columns
    df = df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]].copy()

    # Clean types
    df["Gender"] = df["Gender"].astype(str)
    for c in ["Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    return df


# ----------------------------
# ML utilities
# ----------------------------
def make_pipeline(
    n_estimators: int = 600,
    max_depth: int | None = 10,
    max_features: str | int | float = "sqrt",
    random_state: int = 42,
) -> Pipeline:
    categorical = ["Gender"]
    numeric = ["Age", "BMI", "Duration", "Heart_Rate", "Body_Temp"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
            ]), categorical),
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric),
        ],
        remainder="drop"
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])
    return pipe


@st.cache_resource
def train_and_evaluate(df: pd.DataFrame, n_estimators: int, max_depth: int, max_features: str):
    X = df.drop(columns=["Calories"])
    y = df["Calories"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = make_pipeline(
        n_estimators=n_estimators,
        max_depth=None if max_depth == 0 else max_depth,
        max_features=max_features
    )
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
        "test_size": len(X_test),
        "train_size": len(X_train),
    }

    # Feature importance (for RandomForest)
    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    importances = pipe.named_steps["model"].feature_importances_
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return pipe, metrics, fi, X_train.columns.tolist()


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def heart_rate_zones(age: int) -> dict:
    # Simple common formula (educational)
    max_hr = 220 - age
    return {
        "max_hr": max_hr,
        "moderate_low": int(max_hr * 0.50),
        "moderate_high": int(max_hr * 0.70),
        "vigorous_low": int(max_hr * 0.70),
        "vigorous_high": int(max_hr * 0.85),
    }


# ----------------------------
# Sidebar: Data + training controls
# ----------------------------
st.sidebar.title("Controls")

data_mode = st.sidebar.radio("Data source", ["Use local CSV files", "Upload CSV files"])

df = None
try:
    if data_mode == "Use local CSV files":
        # expects calories.csv and exercise.csv in the same folder
        if Path("calories.csv").exists() and Path("exercise.csv").exists():
            df = load_data_from_paths("calories.csv", "exercise.csv")
        else:
            st.sidebar.warning("Local files not found. Switch to upload mode or place calories.csv and exercise.csv next to app.py.")
    else:
        cal_file = st.sidebar.file_uploader("Upload calories.csv", type=["csv"])
        ex_file = st.sidebar.file_uploader("Upload exercise.csv", type=["csv"])
        if cal_file and ex_file:
            df = load_data_from_uploads(cal_file, ex_file)
except Exception as e:
    st.sidebar.error(f"Data loading error: {e}")

st.sidebar.divider()
st.sidebar.subheader("Model settings")

n_estimators = st.sidebar.slider("n_estimators", 200, 1500, 600, step=100)
max_depth = st.sidebar.slider("max_depth (0 = None)", 0, 30, 10, step=1)
max_features = st.sidebar.selectbox("max_features", ["sqrt", "log2", 0.7, 1.0])

st.sidebar.caption("Tip: Higher n_estimators increases accuracy but also training time.")


# ----------------------------
# Navigation
# ----------------------------
page = st.sidebar.radio("Go to", ["Welcome", "Data & Model", "User Input & Prediction", "Analysis & Recommendations"])

# Initialize history
if "history" not in st.session_state:
    st.session_state["history"] = []


# ----------------------------
# Page: Welcome
# ----------------------------
if page == "Welcome":
    st.title("Personal Fitness Tracker (Calories Burn Prediction)")
    st.write(
        """
        This app predicts **calories burned** based on user exercise information using a supervised ML regression model.
        
        Pages:
        - **Data & Model**: dataset preview, model performance, feature importance
        - **User Input & Prediction**: enter values and predict calories
        - **Analysis & Recommendations**: similar records + personalized guidance
        """
    )
    st.info("Educational tool only. Not medical advice.")


# ----------------------------
# Page: Data & Model
# ----------------------------
elif page == "Data & Model":
    st.title("Data & Model")

    if df is None or df.empty:
        st.warning("Please load the data using the sidebar.")
        st.stop()

    st.subheader("Dataset preview")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(df.head(20), use_container_width=True)
    with c2:
        st.write("Rows:", len(df))
        st.write("Columns:", list(df.columns))
        st.write("Calories range:", float(df["Calories"].min()), "to", float(df["Calories"].max()))

    st.subheader("Train model and view performance")
    with st.spinner("Training model..."):
        pipe, metrics, fi, train_cols = train_and_evaluate(df, n_estimators, max_depth, max_features)

    st.session_state["model"] = pipe
    st.session_state["feature_importance"] = fi
    st.session_state["metrics"] = metrics

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{metrics['MAE']:.2f}")
    m2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    m3.metric("R²", f"{metrics['R2']:.3f}")
    m4.metric("Test samples", f"{metrics['test_size']}")

    st.caption("Lower MAE/RMSE is better; higher R² is better.")

    st.subheader("Feature importance (Random Forest)")
    topn = st.slider("Show top N features", 5, 30, 10)
    fig = px.bar(fi.head(topn), x="importance", y="feature", orientation="h", title="Top feature importances")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Quick exploratory plot")
    fig2 = px.scatter(df, x="Duration", y="Calories", color="Gender", trendline="ols", opacity=0.5)
    st.plotly_chart(fig2, use_container_width=True)


# ----------------------------
# Page: User Input & Prediction
# ----------------------------
elif page == "User Input & Prediction":
    st.title("User Input & Prediction")

    if df is None or df.empty:
        st.warning("Please load the data using the sidebar.")
        st.stop()

    if "model" not in st.session_state:
        with st.spinner("Training model..."):
            pipe, metrics, fi, train_cols = train_and_evaluate(df, n_estimators, max_depth, max_features)
        st.session_state["model"] = pipe
        st.session_state["feature_importance"] = fi
        st.session_state["metrics"] = metrics

    model = st.session_state["model"]

    with st.form("user_inputs"):
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 10, 100, 30)

        with c2:
            weight = st.slider("Weight (kg)", 30.0, 180.0, 70.0, step=0.5)
            height = st.slider("Height (cm)", 120.0, 220.0, 170.0, step=0.5)

        with c3:
            duration = st.slider("Duration (minutes)", 1, 180, 15)
            heart_rate = st.slider("Heart Rate (bpm)", 40, 200, 90)
            body_temp = st.slider("Body Temperature (°C)", 34.0, 42.0, 37.0, step=0.1)

        submitted = st.form_submit_button("Predict calories")

    bmi = round(weight / ((height / 100) ** 2), 2)

    input_df = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
    }])

    st.write("Your inputs")
    st.dataframe(input_df, use_container_width=True)

    if submitted:
        with st.spinner("Predicting..."):
            time.sleep(0.5)
            pred = float(model.predict(input_df)[0])

        st.session_state["last_input"] = input_df
        st.session_state["last_prediction"] = pred

        # Store history
        row = input_df.iloc[0].to_dict()
        row["Predicted_Calories"] = pred
        st.session_state["history"].append(row)

        st.success(f"Estimated calories burned: {pred:.2f} kcal")

        # Simple gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={"text": "Predicted Calories (kcal)"},
            gauge={"axis": {"range": [0, max(500, pred * 1.2)]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        if st.session_state["history"]:
            hist_df = pd.DataFrame(st.session_state["history"])
            st.subheader("Prediction history (this session)")
            st.dataframe(hist_df, use_container_width=True)
            st.download_button(
                "Download history as CSV",
                data=hist_df.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv"
            )


# ----------------------------
# Page: Analysis & Recommendations
# ----------------------------
elif page == "Analysis & Recommendations":
    st.title("Analysis & Recommendations")

    if df is None or df.empty:
        st.warning("Please load the data using the sidebar.")
        st.stop()

    if "last_prediction" not in st.session_state or "last_input" not in st.session_state:
        st.warning("Make a prediction first on the User Input & Prediction page.")
        st.stop()

    pred = float(st.session_state["last_prediction"])
    input_df = st.session_state["last_input"].copy()
    user = input_df.iloc[0]

    st.subheader("Your prediction")
    st.write(f"Predicted calories: **{pred:.2f} kcal**")

    st.subheader("Similar exercise records (by calories)")
    # Robust similar search: widen range if few matches
    window = 10
    similar = pd.DataFrame()
    for window in [10, 25, 50, 100]:
        similar = df[(df["Calories"] >= pred - window) & (df["Calories"] <= pred + window)]
        if len(similar) >= 5:
            break

    if similar.empty:
        st.write("No similar records found.")
    else:
        st.caption(f"Showing samples within ±{window} kcal of your prediction.")
        st.dataframe(similar.sample(min(10, len(similar))), use_container_width=True)

    st.subheader("Where you stand (percentile-style comparisons)")
    def pct_less(col: str, value: float) -> float:
        return float((df[col] < value).mean() * 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Age percentile", f"{pct_less('Age', user['Age']):.1f}% are younger")
    c2.metric("Duration percentile", f"{pct_less('Duration', user['Duration']):.1f}% do shorter")
    c3.metric("Heart-rate percentile", f"{pct_less('Heart_Rate', user['Heart_Rate']):.1f}% have lower HR")
    c4.metric("Body-temp percentile", f"{pct_less('Body_Temp', user['Body_Temp']):.1f}% have lower temp")

    st.subheader("Personalized recommendations (rule-based)")
    recs = []

    # BMI
    cat = bmi_category(float(user["BMI"]))
    recs.append(f"BMI: **{user['BMI']}** ({cat}).")
    if cat == "Underweight":
        recs.append("Nutrition: consider a calorie surplus and strength training focus (if appropriate).")
    elif cat == "Overweight":
        recs.append("Plan: mix steady-state cardio + resistance training and monitor weekly activity consistency.")
    elif cat == "Obese":
        recs.append("Plan: start with low-impact cardio and progressive increase; consider professional guidance.")

    # Heart rate zones
    zones = heart_rate_zones(int(user["Age"]))
    hr = int(user["Heart_Rate"])
    recs.append(
        f"Estimated max HR: **{zones['max_hr']} bpm**. Moderate zone: {zones['moderate_low']}-{zones['moderate_high']} bpm."
    )
    if hr > zones["vigorous_high"]:
        recs.append("Safety: your heart rate is above a typical vigorous zone; reduce intensity and rest if needed.")
    elif hr < zones["moderate_low"] and int(user["Duration"]) >= 20:
        recs.append("Training: if your goal is cardio fitness, you may increase intensity to reach the moderate zone.")
    else:
        recs.append("Heart rate looks within a reasonable exercise range for many people (context matters).")

    # Duration guidance
    dur = int(user["Duration"])
    if dur < 10:
        recs.append("Duration: consider increasing session duration gradually (e.g., +5 minutes per week).")
    elif dur > 90:
        recs.append("Duration: long sessions can increase fatigue; ensure recovery, hydration, and adequate nutrition.")

    # Temperature
    temp = float(user["Body_Temp"])
    if temp >= 39.0:
        recs.append("Body temperature is high; stop exercise, hydrate, and cool down. Seek help if symptoms persist.")
    elif temp >= 38.0:
        recs.append("You are running warm; hydrate and avoid hot environments.")
    else:
        recs.append("Body temperature appears in a typical range.")

    st.write("\n".join([f"- {r}" for r in recs]))
    st.info("These are general educational suggestions, not a medical diagnosis.")

    st.subheader("Visual comparison")
    fig = px.scatter(df, x="Duration", y="Calories", color="Gender", opacity=0.35)
    fig.add_scatter(
        x=[user["Duration"]],
        y=[pred],
        mode="markers",
        marker=dict(size=14, symbol="x", color="black"),
        name="You (predicted)"
    )
    st.plotly_chart(fig, use_container_width=True)
