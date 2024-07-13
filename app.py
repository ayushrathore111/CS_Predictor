import streamlit as st
import pandas as pd
import joblib

# Load the trained models
rf_cs = joblib.load('rf_cs.joblib')
rf_ts = joblib.load('rf_ts.joblib')
lr_cs = joblib.load('lr_cs.joblib')
lr_ts = joblib.load('lr_ts.joblib')
etr_cs = joblib.load('etr_cs.joblib')
etr_ts = joblib.load('etr_ts.joblib')
ar_cs = joblib.load('ar_cs.joblib')
ar_ts = joblib.load('ar_ts.joblib')

# Title of the web app
st.title('Compressive/Tensile Strength Prediction')

# Sidebar with input fields
st.sidebar.title('Input Features')

U = st.sidebar.slider('Cement (kg/m3)', min_value=200.1, max_value=500.1, step=0.1, value=232.5)
H = st.sidebar.slider('Fly Ash (kg/m3)', min_value=200.1, max_value=500.1, step=1.5, value=270.6)
D = st.sidebar.slider('Fine Aggregates (kg/m3)', min_value=800.1, max_value=1200.1, step=1.5, value=845.5)
Fr = st.sidebar.slider('Water (kg/m3)', min_value=150.1, max_value=300.1, step=1.5, value=180.0)
d50 = st.sidebar.slider('Coarse Aggregates (kg/m3)', min_value=500.1, max_value=1000.1, step=1.5, value=542.2)
HD_ratio = st.sidebar.slider('Water to Binder Ratio', min_value=0.01, max_value=1.0, step=0.01, value=0.12)
Dd50_ratio = st.sidebar.slider('Water to Powder Ratio', min_value=0.01, max_value=1.0, step=0.01, value=0.05)
DsD_ratio = st.sidebar.slider('Superplasticizer (kg/m3)', min_value=0.01, max_value=10.0, step=0.01, value=0.18)
Days = st.sidebar.slider('Curing Days', min_value=1, max_value=30, step=1, value=7)

# Function to make predictions using the models
def make_predictions(rf_cs,rf_ts,lr_cs,lr_ts,etr_cs,etr_ts,ar_ts,ar_cs, U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days):
    # Make predictions using the models
    rf_cs_pred = rf_cs.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    rf_ts_pred = rf_ts.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    lr_cs_pred = lr_cs.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    lr_ts_pred = lr_ts.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    etr_cs_pred = etr_cs.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    etr_ts_pred = etr_ts.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    ar_cs_pred = ar_cs.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    ar_ts_pred = ar_ts.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    return rf_cs_pred, rf_ts_pred,lr_cs_pred,lr_ts_pred,etr_cs_pred,etr_ts_pred,ar_cs_pred,ar_ts_pred

# Make predictions using the input values
rf_cs_pred, rf_ts_pred,lr_cs_pred,lr_ts_pred,etr_cs_pred,etr_ts_pred,ar_cs_pred,ar_ts_pred = make_predictions(rf_cs,rf_ts,lr_cs,lr_ts,etr_cs,etr_ts,ar_ts,ar_cs, U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days)

# Display predictions

st.title("Compressive Strength")
st.write('### Random Forest Prediction (cs):', rf_cs_pred)
st.write('### Linear Regression Prediction (cs):', lr_cs_pred)
st.write('### Extra Tree Prediction (cs):', etr_cs_pred)
st.write('### Adaboost Prediction (cs):', ar_cs_pred)


st.title("Tensile Strength")
st.write('### Random Forest Prediction (ts):', rf_ts_pred)
st.write('### Linear Regression Prediction (ts):', lr_ts_pred)
st.write('### Extra Tree Prediction (ts):', etr_ts_pred)
st.write('### Adaboost Prediction (ts):', ar_ts_pred)

