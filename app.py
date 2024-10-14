# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from imblearn.over_sampling import SMOTE
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def ExportFilePickle(model_name, model):
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs("./train_models", exist_ok=True)
    save_path = os.path.join("./train_models", f'{model_name}_model.pkl')  # Đường dẫn lưu mô hình
    with open(save_path, 'wb') as file:
        pickle.dump(model, file)  # Lưu mô hình vào file

def load_model(model_type):
    model_path = f'BTL_HOMAY_NHOM16/Test/train_models/{model_type}_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Mô hình {model_type} không tồn tại tại {model_path}.")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Chuẩn hóa dữ liệu
scaler = StandardScaler()

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
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

            input_data = np.array([[fixedAcidity, volatileAcidity, citricAcid, residualSugar,
                                    chlorides, freeSulfurDioxide, totalSulfurDioxide, density,
                                    pH, sulphates, alcohol]])
            
            # Sử dụng scaler đã được fit trên dữ liệu huấn luyện
            input_data_scaler = scaler.transform(input_data)
            
            model_predict = load_model(selected_model)
            predict = model_predict.predict(input_data_scaler)[0]
            predict = round(predict, 2)
            return jsonify({'prediction': predict})
        except Exception as ex:
            return jsonify({'error': str(ex)})

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
