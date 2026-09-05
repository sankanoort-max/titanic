import streamlit as st
import pandas as pd
import pickle


# Load model and encoders
@st.cache_resource
def load_model():

    with open("titanic_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("sex_encoder.pkl", "rb") as f:
        sex_encoder = pickle.load(f)

    with open("embarked_encoder.pkl", "rb") as f:
        embarked_encoder = pickle.load(f)

    return model, sex_encoder, embarked_encoder


model, sex_encoder, embarked_encoder = load_model()


# Page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)


# Title
st.title("🚢 Titanic Survival Prediction")

st.write(
    "Enter the passenger details below to predict "
    "whether the passenger would have survived."
)


# Passenger details
st.subheader("Passenger Details")


pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)


sex = st.selectbox(
    "Sex",
    ["female", "male"]
)


age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=30.0,
    step=1.0
)


sibsp = st.number_input(
    "Number of Siblings / Spouses",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)


parch = st.number_input(
    "Number of Parents / Children",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)


fare = st.number_input(
    "Fare",
    min_value=0.0,
    max_value=1000.0,
    value=32.0,
    step=1.0
)


embarked = st.selectbox(
    "Port of Embarkation",
    ["C", "Q", "S"]
)


# Prediction button
if st.button("Predict Survival"):

    # Encode categorical values
    sex_encoded = sex_encoder.transform([sex])[0]

    embarked_encoded = embarked_encoder.transform([embarked])[0]


    # Create input dataframe
    input_data = pd.DataFrame(
        [[
            pclass,
            sex_encoded,
            age,
            sibsp,
            parch,
            fare,
            embarked_encoded
        ]],
        columns=[
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked"
        ]
    )


    # Make prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]


    # Display result
    st.subheader("Prediction Result")


    if prediction == 1:

        st.success(
            "🎉 The passenger is predicted to SURVIVE."
        )

        st.write(
            f"Survival probability: "
            f"**{probability[1] * 100:.2f}%**"
        )

    else:

        st.error(
            "❌ The passenger is predicted NOT to SURVIVE."
        )

        st.write(
            f"Non-survival probability: "
            f"**{probability[0] * 100:.2f}%**"
        )


    # Show input data
    with st.expander("View Input Data"):

        st.dataframe(input_data)