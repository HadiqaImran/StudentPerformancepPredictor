import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# Load model, scaler, features
# ----------------------------
model = pickle.load(open("model_multi.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("StudentsPerformance.csv")  # make sure CSV is in the same folder

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Data", "Graphs", "Predict"])

# ----------------------------
# HOME PAGE
# ----------------------------
if menu == "Home":
    st.title("🎓 Student Performance Predictor")
    st.write("""
    Welcome! This app predicts the average exam score of students
    based on demographic and educational inputs. You can explore
    the dataset, view graphs, and make predictions.
    """)

# ----------------------------
# DATA PAGE
# ----------------------------
elif menu == "Data":
    st.title("📊 Dataset Preview")
    st.write("First 10 rows of the dataset:")
    st.dataframe(df.head(10))
    
    st.write("Summary Statistics:")
    st.write(df.describe())

# ----------------------------
# GRAPHS PAGE
# ----------------------------
elif menu == "Graphs":
    st.title("📈 Data Visualizations")
    
    st.write("Average Scores Distribution")
    df["average_score"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3
    fig, ax = plt.subplots()
    sns.histplot(df["average_score"], kde=True, bins=20, ax=ax)
    st.pyplot(fig)
    
    st.write("Scores by Gender")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="gender", y="average_score", data=df, ax=ax2)
    st.pyplot(fig2)

    st.write("Scores by Test Preparation Course")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="test preparation course", y="average_score", data=df, ax=ax3)
    st.pyplot(fig3)

# ----------------------------
# PREDICT PAGE
# ----------------------------
elif menu == "Predict":
    st.title("📝 Make a Prediction")
    st.write("Select student info below:")

    # --- User Inputs ---
    gender = st.selectbox("Gender", ["male", "female"])
    race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parent_edu = st.selectbox("Parental Education Level", [
        "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
    ])
    lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
    test_prep = st.selectbox("Test Preparation Course", ["completed", "none"])

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
    if st.button("Predict"):
        scaled = scaler.transform(input_df)
        preds = model.predict(scaled)[0]
        avg_score = preds.mean()
        
        # Fancy cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Score", f"{avg_score:.2f}")
        col2.metric("Math Score", f"{preds[0]:.2f}")
        col3.metric("Reading Score", f"{preds[1]:.2f}")
        col4.metric("Writing Score", f"{preds[2]:.2f}")
