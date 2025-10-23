import streamlit as st
import pickle as pkl
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load trained model
with open('attrition.pkl', 'rb') as f:
    Scalar, model, model2 = pkl.load(f)


st.title("HR Employee Attrition Prediction")
st.write("Enter employee details below to predict if they are likely to leave the company.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=65, value=30)
daily_rate = st.number_input("Daily Rate", min_value=100, max_value=2000, value=1000)
distance = st.number_input("Distance From Home", min_value=0, max_value=100, value=10)
education = st.number_input("Education Level (1-5)", min_value=1, max_value=5, value=3)
env_satisfaction = st.number_input("Environment Satisfaction (1-4)", min_value=1, max_value=4, value=3)
hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=200, value=60)
job_involvement = st.number_input("Job Involvement (1-4)", min_value=1, max_value=4, value=3)
job_level = st.number_input("Job Level (1-5)", min_value=1, max_value=5, value=2)
job_satisfaction = st.number_input("Job Satisfaction (1-4)", min_value=1, max_value=4, value=3)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
monthly_rate = st.number_input("Monthly Rate", min_value=1000, max_value=100000, value=20000)
num_companies = st.number_input("Number of Companies Worked", min_value=0, max_value=20, value=2)
overtime = st.selectbox("Overtime", ["Yes", "No"])
percent_salary_hike = st.number_input("Percent Salary Hike", min_value=0, max_value=50, value=10)
performance_rating = st.number_input("Performance Rating (1-5)", min_value=1, max_value=5, value=3)
relationship_satisfaction = st.number_input("Relationship Satisfaction (1-4)", min_value=1, max_value=4, value=3)
stock_option_level = st.number_input("Stock Option Level (0-3)", min_value=0, max_value=3, value=0)
total_working_years = st.number_input("Total Working Years", min_value=0, max_value=50, value=5)
training_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=20, value=2)
work_life_balance = st.number_input("Work Life Balance (1-4)", min_value=1, max_value=4, value=3)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=50, value=3)
years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=2)
years_since_last_promo = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=1)
years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, max_value=20, value=2)

# Encode Overtime
overtime_val = 1 if overtime == "Yes" else 0

# Collect all inputs into numpy array
input_data = np.array([
    age, daily_rate, distance, education, env_satisfaction, hourly_rate,
    job_involvement, job_level, job_satisfaction, monthly_income,
    monthly_rate, num_companies, overtime_val, percent_salary_hike,
    performance_rating, relationship_satisfaction, stock_option_level,
    total_working_years, training_last_year, work_life_balance,
    years_at_company, years_in_current_role, years_since_last_promo,
    years_with_curr_manager
]).reshape(1, -1)

# Scale input using StandardScaler
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_data)  # You can save & load original scaler later

# Predict button
if st.button("Predict Attrition"):
    pred = model.predict(input_scaled)
    if pred[0] == 1:
        st.error("⚠️ This employee is likely to leave (Attrition = Yes).")
    else:
        st.success("✅ This employee is likely to stay (Attrition = No).")
