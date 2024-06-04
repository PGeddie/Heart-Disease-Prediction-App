import streamlit as st
import numpy as np
import pickle

model_filename = 'heart_disease_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

def predict_disease(input_data):
    
    data_as_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(data_as_array)
    return prediction[0]

st.title("Heart Disease Prediction App")
st.write("""Please enter the patient's details """)

age = st.number_input("Age", min_value=0, max_value=120, step=1)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=50, max_value=200, step=1)
chol = st.number_input("Serum Cholesterol (in mg/dl)", min_value=100, max_value=600, step=1)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl ", ["True", "False"])
restecg = st.selectbox("Resting Electrocardiographic Results (0 = normal; 1 = abnormal; 2 = ventricular hypertrophy)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, step=1)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (0 = upsloping; 1 = flat; 2 = downsloping)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
thal = st.selectbox("Status of the heart (1 = normal; 2 = fixed defect; 3 = reversible defect, 0=unknown)", [0, 1, 2, 3])

if st.button("Predict"):
    try:
        input_data = [age, 1 if sex == "Male" else 0, cp, trestbps, chol, 1 if fbs== "True" else 0, restecg, thalach, 1 if exang== "Yes" else 0, oldpeak, slope, ca, thal]
        prediction = predict_disease(input_data)
        if prediction == 1:
            st.warning("The patient is likely to have heart disease. Further tests are recommended.")
        else:
            st.success("The patient is unlikely to have heart disease.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
