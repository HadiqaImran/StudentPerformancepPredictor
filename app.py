import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt



model = pickle.load(open("model_multi.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))



st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# Load Dataset

df = pd.read_csv("StudentsPerformance.csv")  


st.markdown(
    """
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] > div:first-child {
        background-color: #8acbe3;
        padding: 2rem;
    }
    /* Sidebar menu items */
    .sidebar-button {
        display: block;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #ffffff;
        background-color: #000000;
        border-radius: 0.5rem;
        text-decoration: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .sidebar-button:hover {
        background-color: #1f77b4;
        color: #ffffff;
    }
    .sidebar-button-selected {
        background-color: #ff6600;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Sidebar Navigation

st.sidebar.title("🎓 Navigation")
pages = ["Home", "Data", "Graphs", "Predict"]
icons = ["🏠", "📊", "📈", "📝"]


if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"

for i, page in enumerate(pages):
    if st.sidebar.button(f"{icons[i]} {page}", key=page):
        st.session_state.selected_page = page

menu = st.session_state.selected_page


# HOME PAGE

if menu == "Home":
  
    st.markdown("""
    <div style='background-color:#1f77b4; padding:30px; border-radius:15px; color:white; text-align:center;'>
        <h1>🎓 Student Performance Predictor</h1>
        <p>Predict the <b>average exam score</b> of students based on demographic and educational inputs.</p>
        <p>Explore the dataset, visualize data, and make predictions easily.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    avg_score = (df["math score"] + df["reading score"] + df["writing score"]) / 3
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(df))
    col2.metric("Average Math Score", f"{df['math score'].mean():.2f}")
    col3.metric("Average Reading Score", f"{df['reading score'].mean():.2f}")
    col4.metric("Average Writing Score", f"{df['writing score'].mean():.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 👩‍🎓 Student Demographics")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    df['gender'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1, colors=['#636efa','#ef553b'])
    ax1.set_ylabel('')
    ax1.set_title("Gender Distribution")
    df['race/ethnicity'].value_counts().plot.bar(ax=ax2, color="#00cc96")
    ax2.set_title("Race/Ethnicity Distribution")
    st.pyplot(fig)


    st.info("💡 Tip: Students who complete the test preparation course tend to score higher! 🎯")


# DATA PAGE

elif menu == "Data":
    st.title("📊 Dataset Preview")
    st.write("First 10 rows of the dataset:")
    st.dataframe(df.head(10))
    
    st.write("Summary Statistics:")
    st.write(df.describe())

# GRAPHS PAGE

elif menu == "Graphs":
    st.title("📈 Data Visualizations")
 
    avg_score = (df["math score"] + df["reading score"] + df["writing score"]) / 3
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Score Mean", f"{avg_score.mean():.2f}")
    col2.metric("Average Math Score", f"{df['math score'].mean():.2f}")
    col3.metric("Average Reading Score", f"{df['reading score'].mean():.2f}")
  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    sns.histplot(avg_score, kde=True, bins=20, color="#1f77b4", ax=ax1)
    ax1.set_title("Average Score Distribution")
    sns.boxplot(x="gender", y=avg_score, data=df, palette="Set2", ax=ax2)
    ax2.set_title("Scores by Gender")
    st.pyplot(fig)

    fig2, ax3 = plt.subplots(figsize=(6,5))
    sns.boxplot(x="test preparation course", y=avg_score, data=df, palette="Set3", ax=ax3)
    ax3.set_title("Scores by Test Preparation Course")
    st.pyplot(fig2)


# PREDICT PAGE

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

    input_df = pd.DataFrame(columns=features)
    input_df.loc[0] = 0

    
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

    
    if st.button("Predict"):
        scaled = scaler.transform(input_df)
        preds = model.predict(scaled)[0]
        avg_score = preds.mean()
        
    
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div style='background-color:#00cc96; padding:20px; border-radius:10px; text-align:center; color:white;'><h3>Average Score</h3><h2>{avg_score:.2f}</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div style='background-color:#636efa; padding:20px; border-radius:10px; text-align:center; color:white;'><h3>Math Score</h3><h2>{preds[0]:.2f}</h2></div>", unsafe_allow_html=True)
        col3.markdown(f"<div style='background-color:#ef553b; padding:20px; border-radius:10px; text-align:center; color:white;'><h3>Reading Score</h3><h2>{preds[1]:.2f}</h2></div>", unsafe_allow_html=True)
        col4.markdown(f"<div style='background-color:#ffa15a; padding:20px; border-radius:10px; text-align:center; color:white;'><h3>Writing Score</h3><h2>{preds[2]:.2f}</h2></div>", unsafe_allow_html=True)
