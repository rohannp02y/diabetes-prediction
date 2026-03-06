import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

#loading the saved model
loaded_model =pickle.load(open('C:/Users/DELL/Downloads/credit_card_fraud/diabetes_prediction/trained_model.sav', 'rb'))

#creating a function for Prediction
def diabetes_prediction(input_data):
    #changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we predict for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

  

    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0]==0:
        return('The person is not diabetic')
    else:
        return('The person is diabetic')

def main():
    #giving a title
    st.title('Diabetes prediction web app')

    #getting input from user
    Pregnancies=st.text_input('Number of pregenancies')
    Glucose= st.text_input('Glucose Level')
    BloodPressure =st.text_input('Blood_pressure value')
    skin_thickness= st.text_input('skin thickness')
    Insulin = st.text_input("Insulin level")
    BMI= st.text_input("BMI value")
    DiabetsPedigreeFunction=st.text_input("Diabets Pedigree Function")
    Age =st.text_input("age of the person ")

    #code for prediction 
    diagnosis =''

    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([ Pregnancies,Glucose,BloodPressure,skin_thickness,Insulin ,BMI,DiabetsPedigreeFunction,Age])
    st.success(diagnosis)


if __name__ == '__main__':
    main()

