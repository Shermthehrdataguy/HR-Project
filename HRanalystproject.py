import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import plotly.express as px

# App Configuration
st.set_page_config(
    page_title="HR Insights Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Attribution
st.markdown("""
    <style>
    .main-title { font-size:36px; font-weight:bold; color:#2C3E50; }
    .sub-title { font-size:18px; font-weight:normal; color:#7F8C8D; }
    .footer { font-size:14px; color:#95A5A6; text-align:center; margin-top:50px; }
    </style>
    <div class="main-title">HR Insights Dashboard</div>
    <div class="sub-title">Analyze, Predict, and Empower HR Decisions with Data and AI</div>
    <hr style="border-top: 1px solid #BDC3C7;">
    """, unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
menu = st.sidebar.radio(
    "Select a Section",
    ["Employee Overview", "Attrition Prediction", "Sentiment Analysis"]
)

# Generate Synthetic Data
def create_synthetic_data():
    data = {
        "EmployeeID": [1, 2, 3, 4, 5],
        "Department": ["IT", "HR", "Finance", "Marketing", "Sales"],
        "Position": ["Developer", "HR Manager", "Accountant", "Marketer", "Sales Rep"],
        "Gender": ["Male", "Female", "Female", "Male", "Female"],
        "EmploymentStatus": ["Active", "Terminated", "Active", "Active", "Terminated"],
        "Age": [25, 45, 38, 29, 50],
        "PayRate": [50, 70, 60, 55, 65],
        "Tenure": [2, 10, 5, 3, 12],
        "EngagementSurvey": [4.5, 3.2, 4.0, 4.8, 2.5],
        "Salary": [60000, 80000, 70000, 55000, 85000]
    }
    df = pd.DataFrame(data)
    return df

# Load Data
@st.cache_data
def load_data():
    return create_synthetic_data()

data = load_data()

# Employee Overview Section
if menu == "Employee Overview":
    st.subheader("Employee Overview")
    with st.expander("Filters", expanded=True):
        department = st.selectbox("Department", ["All"] + list(data["Department"].unique()))
        job_title = st.selectbox("Job Title", ["All"] + list(data["Position"].unique()))
    
    filtered_data = data.copy()
    if department != "All":
        filtered_data = filtered_data[filtered_data["Department"] == department]
    if job_title != "All":
        filtered_data = filtered_data[filtered_data["Position"] == job_title]
    
    st.write("### Filtered Employee Records")
    st.dataframe(filtered_data)

    # Visualizations
    st.write("### Employee Metrics")
    col1, col2 = st.columns(2)
    with col1:
        gender_dist = filtered_data["Gender"].value_counts()
        st.plotly_chart(px.pie(values=gender_dist, names=gender_dist.index, title="Gender Distribution"), use_container_width=True)
    with col2:
        salary_dist = filtered_data.groupby("Department")["Salary"].mean()
        salary_dist_df = salary_dist.reset_index()
        salary_dist_df.columns = ['Department', 'Average Salary']
        st.plotly_chart(px.bar(salary_dist_df, x='Department', y='Average Salary', title="Average Salary by Department"), use_container_width=True)

# Attrition Prediction Section
elif menu == "Attrition Prediction":
    st.subheader("Attrition Prediction")
    st.write("Use AI to predict employee attrition risk.")

    # Feature Engineering
    data['Attrition'] = np.where(data['EmploymentStatus'] == 'Terminated', 1, 0)
    features = ['Age', 'PayRate', 'Tenure', 'EngagementSurvey']
    target = 'Attrition'

    # Train-Test Split
    X = data[features].dropna()
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model Training
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Performance Metrics
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Model Accuracy: {accuracy:.2f}")

    # Feature Importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values(by="Importance", ascending=False)
    st.bar_chart(feature_importance.set_index("Feature"))

    # Employee Attrition Prediction
    st.write("### Predict Individual Employee's Attrition Risk")
    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        pay_rate = st.number_input("Pay Rate", min_value=10, max_value=200, value=50)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=30, value=5)
        engagement_survey = st.number_input("Engagement Survey Score (1-5)", min_value=1.0, max_value=5.0, value=3.0)
        submitted = st.form_submit_button("Predict")
        if submitted:
            prediction = model.predict([[age, pay_rate, tenure, engagement_survey]])
            result = "High Attrition Risk" if prediction[0] == 1 else "Low Attrition Risk"
            st.success(f"Prediction: {result}")

# Sentiment Analysis Section
elif menu == "Sentiment Analysis":
    st.subheader("Sentiment Analysis on Employee Feedback")
    st.write("Analyze feedback to understand sentiment trends.")

    # Sentiment Analysis
    feedback = st.text_area("Enter Employee Feedback", "This company values its employees.")
    if st.button("Analyze Sentiment"):
        sentiment = TextBlob(feedback).sentiment
        st.metric(label="Polarity", value=f"{sentiment.polarity:.2f}")
        st.metric(label="Subjectivity", value=f"{sentiment.subjectivity:.2f}")

    # Feedback Data
    st.write("### Example Feedback Analysis")
    feedback_data = pd.DataFrame({
        "Feedback": ["Great place to work", "Too much workload", "Amazing opportunities", "Lack of diversity"],
        "Polarity": [0.8, -0.4, 0.9, -0.3]
    })
    st.dataframe(feedback_data)

    # Sentiment Distribution
    st.write("### Sentiment Distribution")
    st.plotly_chart(px.histogram(feedback_data, x="Polarity", title="Polarity Distribution"), use_container_width=True)

# Attribution Footer
st.markdown("""
    <div class="footer">
    This project was created by <strong>Sherman Lee</strong> as part of an HR Analytics showcase.<br>
    Explore more on <a href="https://linkedin.com" target="_blank">LinkedIn</a>.
    </div>
    """, unsafe_allow_html=True)
