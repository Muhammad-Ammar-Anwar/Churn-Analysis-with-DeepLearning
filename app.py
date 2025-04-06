import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

model=tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
  label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
  scaler=pickle.load(file)

with open('one_hot_encoder_geo.pkl','rb') as file:
  one_hot_encoder_geo=pickle.load(file)

## streamlit app

st.markdown("""
    <style>
        .title {
            font-size: 36px;
            color: #0073e6;
            font-weight: bold;
        }
        .section-title {
            font-size: 20px;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
        }
        .stSlider>div {
            font-size: 14px;
        }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="title">Customer Churn Prediction</p>', unsafe_allow_html=True)

# User Input Section
st.markdown("<p class='section-title'>Please enter customer details:</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)  # Split the inputs into two columns

with col1:
    geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92)
    balance = st.number_input('Balance', min_value=0, step=1000)
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850)
    estimated_salary = st.number_input('Estimated Salary', min_value=10000, max_value=200000)
    
with col2:
    tenure = st.slider('Tenure', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data for prediction
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = one_hot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

# Combine the one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display the prediction result
st.subheader("Prediction Results")
st.write(f'Churn Probability: {prediction_proba:.2f}')

# Show result based on probability
if prediction_proba > 0.5:
    st.markdown("<p style='color: red; font-size: 18px; font-weight: bold;'>The customer is likely to churn.</p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='color: green; font-size: 18px; font-weight: bold;'>The customer is not likely to churn.</p>", unsafe_allow_html=True)

# Add a button to refresh the prediction or enter new details
if st.button("Predict Again"):
    st.rerun()