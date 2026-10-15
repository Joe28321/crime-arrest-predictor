"""Crime Severity / Arrests Predictor — Streamlit Cloud entry point.

Deploy: push this folder to GitHub (keep the two .pkl files),
then https://share.streamlit.io -> New app -> select repo -> Main file: app.py
"""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
SEV_MODEL = BASE / "severity_predictor.pkl"
ARR_MODEL = BASE / "arrest_predictor.pkl"


@st.cache_resource
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


st.set_page_config(page_title="Crime Severity / Arrest Predictor", layout="centered")
st.title("Crime Severity / Arrest Predictor")
st.caption("Pickles trained on crime_incidents_cleaned.csv (RandomForest). Accuracy ~21-26% — weak signal, demo only.")

if not SEV_MODEL.exists() or not ARR_MODEL.exists():
    st.error(
        "Model files not found. Make sure `severity_predictor.pkl` and "
        "`arrest_predictor.pkl` are in the same GitHub repo folder as `app.py`."
    )
    st.stop()

model_choice = st.selectbox("Choose model", ["Severity (Low/Med/High/Critical)", "Arrests (0-5)"])
obj = load_model(SEV_MODEL if model_choice.startswith("Severity") else ARR_MODEL)
pipe = obj["pipeline"]

st.subheader("Enter incident details")
c1, c2 = st.columns(2)
with c1:
    crime_type = st.selectbox("crime_type", ["Theft", "Assault", "Burglary", "Robbery", "Armed Robbery", "Homicide", "Kidnapping", "Fraud", "Cybercrime", "Drug Offense", "DUI", "Domestic Violence", "Sexual Assault", "Arson", "Vandalism", "Trespassing"])
    district = st.selectbox("district", ["North", "South", "East", "West", "Central", "Northeast", "Northwest", "Southeast", "Southwest", "Midtown"])
    city = st.selectbox("city", ["Riverside", "Centerville", "Lakewood", "Springfield", "Oakdale", "Maplewood", "Fairview", "Hillcrest"])
    state = st.selectbox("state", ["NY", "TX", "OH", "PA", "GA", "NC", "CA", "FL", "IL", "AZ"])
    weapon = st.selectbox("weapon_used", ["Unarmed", "Hands/Feet", "Knife", "Blunt Object", "Firearm", "Gun", "Pistol", "Rifle", "Bat", "Unknown"])
    suspect_gender = st.selectbox("suspect_gender", ["Male", "Female", "Unknown", "Other"])
    suspect_race = st.selectbox("suspect_race", ["Black", "White", "Asian", "Hispanic", "Unknown", "Other"])
    victim_gender = st.selectbox("victim_gender", ["Male", "Female", "Unknown", "Other"])
with c2:
    latitude = st.number_input("latitude", 25.0, 48.0, 40.0, 0.1)
    longitude = st.number_input("longitude", -125.0, -60.0, -75.0, 0.1)
    suspect_age = st.number_input("suspect_age", 0, 120, 30, 1)
    victim_age = st.number_input("victim_age", 0, 120, 45, 1)
    property_loss = st.number_input("property_loss_usd", 0.0, 100000.0, 5000.0, 100.0)
    reported = st.selectbox("reported_online", ["Yes (1)", "No (0)", "Unknown (-1)"])
    hour = st.slider("hour", 0, 23, 14)
    month = st.slider("month", 1, 12, 6)
    year = st.selectbox("year", [2018, 2019, 2020, 2021, 2022, 2023, 2024])
    dow = st.selectbox("day of week", ["Mon(0)", "Tue(1)", "Wed(2)", "Thu(3)", "Fri(4)", "Sat(5)", "Sun(6)"])

rep_map = {"Yes (1)": 1, "No (0)": 0, "Unknown (-1)": -1}
row = {
    "crime_type": crime_type, "district": district, "city": city, "state": state,
    "latitude": latitude, "longitude": longitude,
    "suspect_age": suspect_age, "suspect_gender": suspect_gender, "suspect_race": suspect_race,
    "victim_age": victim_age, "victim_gender": victim_gender,
    "weapon_used": weapon, "property_loss_usd": property_loss,
    "reported_online": rep_map[reported],
    "hour": hour, "month": month, "year": year, "dow": int(dow[4]),
}
if model_choice.startswith("Severity"):
    pass
else:  # arrest model also needs severity input
    row["severity"] = st.selectbox("severity (arrest model only)", ["Low", "Medium", "High", "Critical", "Unknown"])

if st.button("Predict"):
    sample = pd.DataFrame([row])
    pred = pipe.predict(sample)[0]
    proba = pipe.predict_proba(sample)[0]
    st.success(f"Prediction: **{pred}**")
    st.write("Probabilities:")
    classes = list(getattr(pipe, "classes_", []))
    st.bar_chart(pd.DataFrame({"prob": proba}, index=[str(c) for c in classes]))
