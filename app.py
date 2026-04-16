import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

st.title("Customer Churn Predictor")

@st.cache_data
def load_and_train():
    df = pd.read_csv("Churn_Modelling.csv")
    data = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    le_geo = LabelEncoder()
    le_gen = LabelEncoder()
    data['Geography'] = le_geo.fit_transform(data['Geography'])
    data['Gender']    = le_gen.fit_transform(data['Gender'])
    X = data.drop('Exited', axis=1)
    y = data['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # ANN: 2 hidden layers (64 and 32 neurons), ReLU activation, trained with Adam optimizer
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler, le_geo, le_gen

with st.spinner("Loading model..."):
    model, scaler, le_geo, le_gen = load_and_train()

# ── Input Form ────────────────────────────────────────────────────────────────
geography    = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender       = st.selectbox("Gender", ["Female", "Male"])
credit_score = st.number_input("Credit Score",       min_value=300,  max_value=850,      value=650,      step=1)
age          = st.number_input("Age",                min_value=18,   max_value=100,      value=39,       step=1)
tenure       = st.number_input("Tenure",             min_value=0,    max_value=10,       value=5,        step=1)
balance      = st.number_input("Balance",            min_value=0.0,  max_value=300000.0, value=76486.0,  step=100.0)
num_products = st.number_input("Number of Products", min_value=1,    max_value=4,        value=1,        step=1)
has_cr_card  = st.selectbox("Has Credit Card",  ["Yes", "No"])
is_active    = st.selectbox("Is Active Member", ["Yes", "No"])
est_salary   = st.number_input("Estimated Salary",   min_value=0.0,  max_value=300000.0, value=100090.0, step=100.0)

if st.button("Predict Churn", use_container_width=True):
    input_df = pd.DataFrame([{
        'CreditScore':     credit_score,
        'Geography':       le_geo.transform([geography])[0],
        'Gender':          le_gen.transform([gender])[0],
        'Age':             age,
        'Tenure':          tenure,
        'Balance':         balance,
        'NumOfProducts':   num_products,
        'HasCrCard':       1 if has_cr_card == "Yes" else 0,
        'IsActiveMember':  1 if is_active   == "Yes" else 0,
        'EstimatedSalary': est_salary
    }])
    probs = model.predict_proba(scaler.transform(input_df))[0]
    churn_prob = probs[1]
    stay_prob  = probs[0]
    decision   = "WILL CHURN" if churn_prob >= 0.5 else "WILL STAY"

    st.divider()
    st.markdown("## Prediction")
    st.markdown(f"Churn probability")
    st.markdown(f"# {churn_prob:.2%}")
    st.markdown(f"Stay probability")
    st.markdown(f"# {stay_prob:.2%}")
    st.markdown(f"**Decision:** {decision}")
    st.progress(stay_prob)
