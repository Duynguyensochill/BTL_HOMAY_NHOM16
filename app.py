import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request
import os
app = Flask(__name__)
import qrcode

import pickle

def load_model(type):
    with open(f'C:/Users/Lenovo/Desktop/BTL_HOMAY_NHOM16/Test/train_models/{type}_model.pkl', 'rb') as file:
        model = pickle.load(file)  # Phải thụt đầu dòng sau 'with'
    return model
  # Thụt lề đúng cách


  # Tải mô hình từ file
    return model

scaler = StandardScaler()
@app.route("/", methods = ['GET', 'POST'])
def index():
    if (request.method == 'POST'):
        try:
            data = request.get_json()
            fixedAcidity = float(data['fixedAcidity'])
            volatileAcidity = float(data['volatileAcidity'])
            citricAcid = float(data['citricAcid'])
            residualSugar = float(data['residualSugar'])
            chlorides = float(data['chlorides'])
            freeSulfurDioxide = float(data['freeSulfurDioxide'])
            totalSulfurDioxide = float(data['totalSulfurDioxide'])
            density = float(data['density'])
            pH = float(data['pH'])
            sulphates = float(data['sulphates'])
            alcohol = float(data['alcohol'])
            selected_model = data['model']

            input_data = np.array([[fixedAcidity, volatileAcidity, citricAcid, residualSugar, chlorides, freeSulfurDioxide, totalSulfurDioxide, density, pH, sulphates, alcohol]])
            input_data_scaler = scaler.fit_transform(input_data)
            model_predict = load_model(selected_model)
            predict = model_predict.predict(input_data_scaler)[0]
            predict = round(predict, 2)
            return {'prediction': predict}
        except Exception as ex:
            return {'error': str(ex)}
        
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)





