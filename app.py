# app.py
import io
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


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")


# -----------------------------
# Session defaults
# -----------------------------
DEFAULTS = {
    "data_mode": "Use local CSV files",  # or "Upload CSV files"
    "n_estimators": 600,
    "max_depth": 10,      # 0 means None (unlimited depth)
    "max_features": "sqrt",
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

if "history" not in st.session_state:
    st.session_state["history"] = []


# -----------------------------
# Data building
# -----------------------------
def build_dataset(calories: pd.DataFrame, exercise: pd.DataFrame) -> pd.DataFrame:
    df = exercise.merge(calories, on="User_ID", how="inner").copy()

    required_cols = {"Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # BMI
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    df["BMI"] = df["BMI"].round(2)

    df = df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]].copy()

    # Clean types
    df["Gender"] = df["Gender"].astype(str)
    for c in ["Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    return df


@st.cache_data
def load_data_from_paths(calories_path: str, exercise_path: str) -> pd.DataFrame:
    calories = pd.read_csv(calories_path)
    exercise = pd.read_csv(exercise_path)
    return build_dataset(calories, exercise)


@st.cache_data
def load_data_from_upload_bytes(calories_bytes: bytes, exercise_bytes: bytes) -> pd.DataFrame:
    calories = pd.read_csv(io.BytesIO(calories_bytes))
    exercise = pd.read_csv(io.BytesIO(exercise_bytes))
    return build_dataset(calories, exercise)


def get_df():
    """Get dataset from session; optionally auto-load local files if configured and present."""
    if "df" in st.session_state and isinstance(st.session_state["df"], pd.DataFrame):
        return st.session_state["df"]

    if st.session_state["data_mode"] == "Use local CSV files":
        if Path("calories.csv").exists() and Path("exercise.csv").exists():
            try:
                st.session_state["df"] = load_data_from_paths("calories.csv", "exercise.csv")
                return st.session_state["df"]
            except Exception:
                return None
    return None


# -----------------------------
# ML pipeline + training
# -----------------------------
def make_pipeline(n_estimators: int, max_depth, max_features):
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
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])


@st.cache_resource
def train_and_evaluate(df: pd.DataFrame, n_estimators: int, max_depth_slider: int, max_features):
    X = df.drop(columns=["Calories"])
    y = df["Calories"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    max_depth = None if int(max_depth_slider) == 0 else int(max_depth_slider)

    pipe = make_pipeline(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        max_features=max_features,
    )
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    # Feature importance
    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    importances = pipe.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)

    return pipe, metrics, fi


# -----------------------------
# Helper functions (recommendations)
# -----------------------------
def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def heart_rate_zones(age: int) -> dict:
    max_hr = 220 - age
    return {
        "max_hr": max_hr,
        "moderate_low": int(max_hr * 0.50),
        "moderate_high": int(max_hr * 0.70),
        "vigorous_low": int(max_hr * 0.70),
        "vigorous_high": int(max_hr * 0.85),
    }


def big_stat(col, title, text):
    with col:
        st.markdown(f"#### {title}")
        st.markdown(
            f"""
            <div style="
                padding:14px;
                border-radius:12px;
                border:1px solid #e6e6e6;
                font-size:28px;
                font-weight:700;
                line-height:1.15;">
                {text}
            </div>
            """,
            unsafe_allow_html=True
        )


def inject_kpi_css():
    st.markdown(
        """
        <style>
        .kpi-card{
            background: var(--secondary-background-color);
            border: 1px solid rgba(120,120,120,0.25);
            border-radius: 14px;
            padding: 14px 16px;
            height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .kpi-title{
            font-size: 0.95rem;
            font-weight: 650;
            color: rgba(120,120,120,1);
            margin: 0;
        }
        .kpi-value{
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0;
            color: var(--text-color);
            line-height: 1.0;
        }
        .kpi-sub{
            font-size: 0.95rem;
            color: rgba(120,120,120,1);
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def kpi_card(col, title: str, value: str, subtitle: str):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <p class="kpi-title">{title}</p>
                <p class="kpi-value">{value}</p>
                <p class="kpi-sub">{subtitle}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# Sidebar: Navigator only
# -----------------------------
st.sidebar.title("Navigator")
page = st.sidebar.radio(
    "Go to",
    [
        "Welcome",
        "Data & Model",
        "User Input & Prediction",
        "Analysis & Recommendations",
        "⚙️ Controls & Model Settings",  # last option
    ],
)

st.sidebar.divider()
st.sidebar.caption("Status")
st.sidebar.caption("✅ Data loaded" if "df" in st.session_state else "❌ Data not loaded")
st.sidebar.caption("✅ Model ready" if "model" in st.session_state else "❌ Model not trained")


# -----------------------------
# Pages
# -----------------------------
if page == "Welcome":
    st.title("Personal Fitness Tracker (Calories Burn Prediction)")
    st.write(
        """
        This Streamlit app predicts **calories burned (kcal)** based on:
        **Gender, Age, BMI, Duration, Heart Rate, and Body Temperature**.

        Use the Navigator to:
        - train and evaluate the model,
        - predict calories for a user,
        - view analysis + recommendations.
        """
    )
    st.info("Educational project only. Not medical advice.")


elif page == "⚙️ Controls & Model Settings":
    st.title("⚙️ Controls & Model Settings")

    st.subheader("1) Data source")
    st.session_state["data_mode"] = st.radio(
        "Choose how to load the dataset",
        ["Use local CSV files", "Upload CSV files"],
        index=0 if st.session_state["data_mode"] == "Use local CSV files" else 1
    )

    if st.session_state["data_mode"] == "Use local CSV files":
        st.write("Expected files in the same folder as `app.py`: **calories.csv** and **exercise.csv**")
        c1, c2 = st.columns(2)
        with c1:
            st.write("calories.csv:", "✅ Found" if Path("calories.csv").exists() else "❌ Missing")
        with c2:
            st.write("exercise.csv:", "✅ Found" if Path("exercise.csv").exists() else "❌ Missing")

        if st.button("Load data from local files"):
            try:
                st.session_state["df"] = load_data_from_paths("calories.csv", "exercise.csv")
                # clear old model if any
                st.session_state.pop("model", None)
                st.session_state.pop("metrics", None)
                st.session_state.pop("feature_importance", None)
                st.success("Data loaded successfully.")
            except Exception as e:
                st.error(f"Failed to load local files: {e}")

    else:
        cal_file = st.file_uploader("Upload calories.csv", type=["csv"])
        ex_file = st.file_uploader("Upload exercise.csv", type=["csv"])

        if st.button("Load data from uploads", disabled=not (cal_file and ex_file)):
            try:
                st.session_state["df"] = load_data_from_upload_bytes(cal_file.getvalue(), ex_file.getvalue())
                # clear old model if any
                st.session_state.pop("model", None)
                st.session_state.pop("metrics", None)
                st.session_state.pop("feature_importance", None)
                st.success("Data loaded successfully.")
            except Exception as e:
                st.error(f"Failed to load uploaded files: {e}")

    st.divider()
    st.subheader("2) Model settings (advanced)")

    st.session_state["n_estimators"] = st.slider(
        "n_estimators", 200, 1500, int(st.session_state["n_estimators"]), step=100
    )
    st.session_state["max_depth"] = st.slider(
        "max_depth (0 = None)", 0, 30, int(st.session_state["max_depth"]), step=1
    )

    max_feat_options = ["sqrt", "log2", 0.7, 1.0]
    current_mf = st.session_state["max_features"]
    if current_mf not in max_feat_options:
        current_mf = "sqrt"
    st.session_state["max_features"] = st.selectbox(
        "max_features",
        max_feat_options,
        index=max_feat_options.index(current_mf),
    )

    st.caption("Tip: Changing settings may require retraining (go to Data & Model).")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear trained model (force retrain)"):
            st.session_state.pop("model", None)
            st.session_state.pop("metrics", None)
            st.session_state.pop("feature_importance", None)
            st.success("Cleared model from session. Go to Data & Model to retrain.")
    with c2:
        if st.button("Clear ALL Streamlit caches (if needed)"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.pop("model", None)
            st.session_state.pop("metrics", None)
            st.session_state.pop("feature_importance", None)
            st.success("Cleared cache. Reload data / retrain as needed.")


elif page == "Data & Model":
    st.title("Data & Model")

    df = get_df()
    if df is None or df.empty:
        st.warning("Data not loaded. Go to **⚙️ Controls & Model Settings** to load/upload CSV files.")
        st.stop()

    st.subheader("Dataset preview")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(df.head(20), use_container_width=True)
    with c2:
        st.write("Rows:", len(df))
        st.write("Calories range:", float(df["Calories"].min()), "to", float(df["Calories"].max()))

    st.subheader("Train + Evaluate")

    if st.button("Train / Retrain model"):
        st.session_state.pop("model", None)
        st.session_state.pop("metrics", None)
        st.session_state.pop("feature_importance", None)

    with st.spinner("Training model..."):
        pipe, metrics, fi = train_and_evaluate(
            df,
            st.session_state["n_estimators"],
            st.session_state["max_depth"],
            st.session_state["max_features"],
        )

    st.session_state["model"] = pipe
    st.session_state["metrics"] = metrics
    st.session_state["feature_importance"] = fi

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{metrics['MAE']:.2f}")
    m2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    m3.metric("R²", f"{metrics['R2']:.3f}")
    m4.metric("Test samples", f"{metrics['test_size']}")

    st.subheader("Feature importance")
    topn = st.slider("Show top N features", 5, 30, 10)
    fig_fi = px.bar(fi.head(topn), x="importance", y="feature", orientation="h")
    fig_fi.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_fi, use_container_width=True)

    st.subheader("Quick exploratory plot")
    # If statsmodels is installed, this works; otherwise fallback without trendline.
    try:
        fig2 = px.scatter(df, x="Duration", y="Calories", color="Gender", trendline="ols", opacity=0.5)
        st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        fig2 = px.scatter(df, x="Duration", y="Calories", color="Gender", opacity=0.5)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Trendline disabled (statsmodels not available or trendline error).")


elif page == "User Input & Prediction":
    st.title("User Input & Prediction")

    df = get_df()
    if df is None or df.empty:
        st.warning("Data not loaded. Go to **⚙️ Controls & Model Settings** to load/upload CSV files.")
        st.stop()

    # Ensure model exists
    if "model" not in st.session_state:
        with st.spinner("Training model (first time)..."):
            pipe, metrics, fi = train_and_evaluate(
                df,
                st.session_state["n_estimators"],
                st.session_state["max_depth"],
                st.session_state["max_features"],
            )
        st.session_state["model"] = pipe
        st.session_state["metrics"] = metrics
        st.session_state["feature_importance"] = fi

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

    st.write("### Your inputs")
    st.dataframe(input_df, use_container_width=True)

    if submitted:
        with st.spinner("Predicting..."):
            time.sleep(0.3)
            pred = float(model.predict(input_df)[0])

        st.session_state["last_input"] = input_df
        st.session_state["last_prediction"] = pred

        # Save history
        row = input_df.iloc[0].to_dict()
        row["Predicted_Calories"] = pred
        st.session_state["history"].append(row)

        st.success(f"Estimated calories burned: **{pred:.2f} kcal**")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={"text": "Predicted Calories (kcal)"},
            gauge={"axis": {"range": [0, max(500, pred * 1.2)]}}
        ))
        st.plotly_chart(gauge, use_container_width=True)

        st.subheader("Prediction history (this session)")
        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df, use_container_width=True)
        st.download_button(
            "Download history as CSV",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )


elif page == "Analysis & Recommendations":
    st.title("Analysis & Personalized Recommendations")

    df = get_df()
    if df is None or df.empty:
        st.warning("Data not loaded. Go to **⚙️ Controls & Model Settings** to load/upload CSV files.")
        st.stop()

    if "last_prediction" not in st.session_state or "last_input" not in st.session_state:
        st.warning("Please make a prediction first in **User Input & Prediction**.")
        st.stop()

    pred = float(st.session_state["last_prediction"])
    user_df = st.session_state["last_input"].copy()
    user = user_df.iloc[0]

    st.subheader("Your prediction")
    st.write(f"Predicted calories: **{pred:.2f} kcal**")

    st.subheader("Similar exercise records (by calories)")
    similar = pd.DataFrame()
    used_window = None
    for window in [10, 25, 50, 100]:
        similar = df[(df["Calories"] >= pred - window) & (df["Calories"] <= pred + window)]
        used_window = window
        if len(similar) >= 5:
            break

    if similar.empty:
        st.write("No similar records found.")
    else:
        st.caption(f"Showing samples within ±{used_window} kcal of your predicted value.")
        st.dataframe(similar.sample(min(10, len(similar))), use_container_width=True)

    st.divider()
    st.subheader("Where you stand (percentile-style comparisons)")

    def pct_less(col: str, value: float) -> float:
        return float((df[col] < value).mean() * 100)

    st.divider()
    st.subheader("Where you stand (percentile-style comparisons)")
    
    inject_kpi_css()
    
    def pct_less(col: str, value: float) -> float:
        # % of users below your value (same logic you used earlier, but cleaner)
        return float((df[col] < value).mean() * 100)
    
    age_pct = pct_less("Age", user["Age"])
    dur_pct = pct_less("Duration", user["Duration"])
    hr_pct  = pct_less("Heart_Rate", user["Heart_Rate"])
    tmp_pct = pct_less("Body_Temp", user["Body_Temp"])
    
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    
    kpi_card(c1, "Age", f"{age_pct:.1f}%", "of users are younger than you")
    kpi_card(c2, "Duration", f"{dur_pct:.1f}%", "of users exercise for less time")
    kpi_card(c3, "Heart Rate", f"{hr_pct:.1f}%", "of users have a lower heart rate")
    kpi_card(c4, "Body Temp", f"{tmp_pct:.1f}%", "of users have a lower body temp")

    st.divider()
    st.subheader("Recommendations (rule-based)")

    recs = []

    bmi_val = float(user["BMI"])
    age_val = int(user["Age"])
    hr_val = int(user["Heart_Rate"])
    dur_val = int(user["Duration"])
    temp_val = float(user["Body_Temp"])

    # BMI
    cat = bmi_category(bmi_val)
    recs.append(f"**BMI:** {bmi_val:.2f} ({cat})")
    if cat == "Underweight":
        recs.append("Nutrition: consider a healthy calorie surplus and strength training focus (if appropriate).")
    elif cat == "Overweight":
        recs.append("Plan: add regular cardio + balanced diet; focus on weekly consistency.")
    elif cat == "Obese":
        recs.append("Plan: start with low-impact cardio and gradual progression; consider professional guidance.")

    # Heart-rate zones
    zones = heart_rate_zones(age_val)
    recs.append(
        f"**Heart-rate zones (approx):** Max HR ≈ {zones['max_hr']} bpm; "
        f"Moderate: {zones['moderate_low']}-{zones['moderate_high']} bpm; "
        f"Vigorous: {zones['vigorous_low']}-{zones['vigorous_high']} bpm."
    )
    if hr_val > zones["vigorous_high"]:
        recs.append("Safety: Heart rate is above typical vigorous zone. Reduce intensity and rest if needed.")
    elif hr_val < zones["moderate_low"] and dur_val >= 20:
        recs.append("Training: If cardio improvement is your goal, increase intensity toward the moderate zone.")
    else:
        recs.append("Heart rate appears within a reasonable exercise range for many people (context matters).")

    # Duration
    if dur_val < 10:
        recs.append("Duration: Increase gradually (example: +5 minutes per week) for better fitness adaptation.")
    elif dur_val > 90:
        recs.append("Duration: Long sessions require hydration and recovery; avoid sudden spikes in training volume.")

    # Temperature
    if temp_val >= 39.0:
        recs.append("Body temperature is high: stop exercise, hydrate, cool down, and seek help if symptoms persist.")
    elif temp_val >= 38.0:
        recs.append("You are running warm: hydrate and avoid hot environments.")
    else:
        recs.append("Body temperature appears in a typical range.")

    for r in recs:
        st.write(f"- {r}")

    st.info("Recommendations are educational and not a medical diagnosis.")

    st.subheader("Visual comparison (Duration vs Calories)")
    fig = px.scatter(df, x="Duration", y="Calories", color="Gender", opacity=0.35)
    fig.add_scatter(
        x=[dur_val],
        y=[pred],
        mode="markers",
        marker=dict(size=14, symbol="x", color="black"),
        name="You (predicted)",
    )
    st.plotly_chart(fig, use_container_width=True)
