import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
from tensorflow.keras.models import load_model

model = load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder=pickle.load(file)
    
with open('oneHot_encoder_gender.pkl','rb') as file:
    oneHot_encoder=pickle.load(file)
    
with open('StandardScaler.pkl','rb') as file:
    Scalar=pickle.load(file)
    
st.title("customer churn prediction")

geography = st.selectbox('Geography', oneHot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input( 'Balance')
credit_score = st. number_input( 'Credit Score')
estimated_salary = st. number_input("Estimated Salary")
tenure = st. slider ('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data=pd.DataFrame(
    {
    'CreditScore':[credit_score],
    'Gender':[label_encoder.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
    }
)

geo_encoded=oneHot_encoder.transform([[geography]])
geo_encoded_df=pd.DataFrame(geo_encoded,columns=oneHot_encoder.get_feature_names_out())


input_data=pd.concat([input_data,geo_encoded_df],axis=1)
input_data_scaled=Scalar.transform(input_data)
prediction=model.predict(input_data_scaled)
prediction_proba=prediction[0][0]
if(st.button('predict')):
    if prediction_proba>0.5:
        st.text("the customer is likely churn")
    else:
        st.text("the customer is not likely churn")