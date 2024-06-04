import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

st.title('Diabetes Prediction App')

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Columns to clean
columns_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0s with np.nan
data[columns_to_clean] = data[columns_to_clean].replace(0, np.nan)

# Fill NaNs using KNN Imputer
imputer = KNNImputer(n_neighbors=5)
data[columns_to_clean] = imputer.fit_transform(data[columns_to_clean])

# Feature Scaling
X = data.drop('Outcome', axis=1)
y = data['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# User input
st.header('Enter your health metrics:')
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=100)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=140, value=80)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin', min_value=0, max_value=900, value=80)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input('Age', min_value=0, max_value=120, value=30)

user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
user_data_scaled = scaler.transform(user_data)

# Prediction
if st.button('Predict'):
    prediction = model.predict(user_data_scaled)
    prediction_proba = model.predict_proba(user_data_scaled)

    if prediction[0] == 1:
        st.write('Prediction: Diabetic')
    else:
        st.write('Prediction: Not Diabetic')
    
    st.write('Probability of being Diabetic: {:.2f}%'.format(prediction_proba[0][1] * 100))
    st.write('Probability of being Not Diabetic: {:.2f}%'.format(prediction_proba[0][0] * 100))
