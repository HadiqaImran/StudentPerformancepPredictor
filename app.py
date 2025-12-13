import streamlit as st
import pandas as pd
import pickle

# Load model, scaler, features
model = pickle.load(open("model_multi.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Predict your average exam score based on your profile</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar Inputs with Tooltips ---
st.sidebar.header("Student Info 📝")
gender = st.sidebar.selectbox("Gender", ["male", "female"], help="Select the gender of the student")
race = st.sidebar.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"], help="Select the student's race/ethnicity group")
parent_edu = st.sidebar.selectbox("Parental Education Level", [
    "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
], help="Select highest parental education level")
lunch = st.sidebar.selectbox("Lunch Type", ["standard", "free/reduced"], help="Select type of lunch provided")
test_prep = st.sidebar.selectbox("Test Preparation Course", ["completed", "none"], help="Did the student complete a test prep course?")

# --- Prepare input dataframe ---
input_df = pd.DataFrame(columns=features)
input_df.loc[0] = 0

# Map user inputs to encoded columns
if "gender_male" in input_df.columns:
    input_df["gender_male"] = 1 if gender=="male" else 0
race_col = f"race/ethnicity_{race}"
if race_col in input_df.columns:
    input_df[race_col] = 1
parent_col = f"parental level of education_{parent_edu}"
if parent_col in input_df.columns:
    input_df[parent_col] = 1
if "lunch_standard" in input_df.columns:
    input_df["lunch_standard"] = 1 if lunch=="standard" else 0
test_col = "test preparation course_completed"
if test_col in input_df.columns:
    input_df[test_col] = 1 if test_prep=="completed" else 0

# --- Prediction ---
if st.button("Predict 🎯"):
    scaled = scaler.transform(input_df)
    preds = model.predict(scaled)[0]
    avg_score = preds.mean()

    st.markdown("<h2 style='text-align: center; color: #008080;'>Your Predicted Scores</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Average Score with fancy highlight
    col1.markdown(f"<div style='background-color:#FFD700; padding:20px; border-radius:10px; text-align:center;'>"
                  f"<h3>Average Score</h3><h2 style='color:#4B0082;'>{avg_score:.2f}</h2></div>", unsafe_allow_html=True)
    
    # Individual scores with progress bars
    col2.markdown(f"<div style='background-color:#ADD8E6; padding:20px; border-radius:10px; text-align:center;'>"
                  f"<h3>Math Score</h3><h2>{preds[0]:.2f}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div style='background-color:#90EE90; padding:20px; border-radius:10px; text-align:center;'>"
                  f"<h3>Reading Score</h3><h2>{preds[1]:.2f}</h2></div>", unsafe_allow_html=True)
    col4.markdown(f"<div style='background-color:#FFB6C1; padding:20px; border-radius:10px; text-align:center;'>"
                  f"<h3>Writing Score</h3><h2>{preds[2]:.2f}</h2></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.progress(int(avg_score))  # show progress bar for average score
    st.info("🎉 Scores are predicted using multi-target Linear Regression. ")
