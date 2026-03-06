import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

#loading the saved model
loaded_model =pickle.load(open('C:/Users/DELL/Downloads/credit_card_fraud/diabetes_prediction/trained_model.sav', 'rb'))

#Making a predictive system
input_data = (4,110,92,0,0,37.6,0.191,30)

#changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we predict for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

scaler = StandardScaler()
scaler.fit(input_data_reshaped)

#standarize the input data
std_data = scaler.transform(input_data_reshaped)


prediction = loaded_model.predict(std_data)


if prediction[0]==0:
  print('The person is not diabetic')
else:
  print('The person is diabetic')