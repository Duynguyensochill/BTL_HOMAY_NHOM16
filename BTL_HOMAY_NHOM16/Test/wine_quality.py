import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import pickle
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

def ExportFilePickle(model_name, model):
    os.makedirs(".\\train_models", exist_ok=True)
    save_path = os.path.join(".\\train_models", f'{model_name}_model.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(model, file)

# Đọc dữ liệu từ tập tin CSV
red_wine = pd.read_csv('BTL_HOMAY_NHOM16/Test/data_set/winequality-red2.csv', sep=';')

X = red_wine.drop('quality', axis=1)
y = red_wine['quality']

# Xử lý dữ liệu bị thiếu
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Cân bằng dữ liệu bằng SMOTE
smote = SMOTE(sampling_strategy='auto', k_neighbors=min(5, max(1, y.value_counts().min() - 1)), random_state=42)
X, y = smote.fit_resample(X, y)

# Chia dữ liệu
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# --- Xây dựng các mô hình ---
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)
ExportFilePickle('linear_regression', lr_model)

# Lasso Regression
lasso_params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
lasso_grid_search = GridSearchCV(Lasso(), lasso_params, cv=5, scoring='r2')
lasso_grid_search.fit(X_train, y_train)
lasso_model = lasso_grid_search.best_estimator_
y_pred_lasso = lasso_model.predict(X_test)
r2_lasso = r2_score(y_test, y_pred_lasso)
ExportFilePickle('lasso', lasso_model)

# Neural Network
nn_model = MLPRegressor(hidden_layer_sizes=(32, 16), 
                        max_iter=1000, 
                        learning_rate='adaptive', 
                        random_state=42, 
                        alpha=0.1,  
                        early_stopping=True,  
                        validation_fraction=0.15)
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)
r2_nn = r2_score(y_test, y_pred_nn)
ExportFilePickle('neural_network', nn_model)

# Stacking Regressor
estimators = [('lr', lr_model), ('lasso', lasso_model), ('nn', nn_model)]
stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stacking_model.fit(X_train, y_train)
y_pred_stack = stacking_model.predict(X_test)
r2_stack = r2_score(y_test, y_pred_stack)
ExportFilePickle('stacking', stacking_model)

# Tính toán các chỉ số đánh giá
results = {
    'Linear Regression': [np.sqrt(mean_squared_error(y_test, y_pred_lr)), mean_absolute_error(y_test, y_pred_lr), r2_lr],
    'Lasso Regression': [np.sqrt(mean_squared_error(y_test, y_pred_lasso)), mean_absolute_error(y_test, y_pred_lasso), r2_lasso],
    'Neural Network': [np.sqrt(mean_squared_error(y_test, y_pred_nn)), mean_absolute_error(y_test, y_pred_nn), r2_nn],
    'Stacking': [np.sqrt(mean_squared_error(y_test, y_pred_stack)), mean_absolute_error(y_test, y_pred_stack), r2_stack]
}

# Tính toán NSE
def nash_sutcliffe_efficiency(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

# Tính toán NSE cho từng mô hình
nse_lr = nash_sutcliffe_efficiency(y_test, y_pred_lr)
nse_lasso = nash_sutcliffe_efficiency(y_test, y_pred_lasso)
nse_nn = nash_sutcliffe_efficiency(y_test, y_pred_nn)
nse_stack = nash_sutcliffe_efficiency(y_test, y_pred_stack)

# Thêm vào kết quả
results = {
    'Linear Regression': [np.sqrt(mean_squared_error(y_test, y_pred_lr)), 
                         mean_absolute_error(y_test, y_pred_lr), 
                         r2_lr, 
                         nse_lr],
    'Lasso Regression': [np.sqrt(mean_squared_error(y_test, y_pred_lasso)), 
                         mean_absolute_error(y_test, y_pred_lasso), 
                         r2_lasso, 
                         nse_lasso],
    'Neural Network': [np.sqrt(mean_squared_error(y_test, y_pred_nn)), 
                       mean_absolute_error(y_test, y_pred_nn), 
                       r2_nn, 
                       nse_nn],
    'Stacking': [np.sqrt(mean_squared_error(y_test, y_pred_stack)), 
                 mean_absolute_error(y_test, y_pred_stack), 
                 r2_stack, 
                 nse_stack]
}

# Hiển thị kết quả
results_df = pd.DataFrame(results, index=['RMSE', 'MAE', 'R2 Score', 'NSE'])
print(results_df)

# Xuất kết quả ra file CSV
results_df.to_csv('static/results.csv', index=True)
print("Kết quả đã được lưu vào static/results.csv")

# Vẽ biểu đồ so sánh dự đoán
def plot_predictions_multi(y_true_train, y_pred_train, 
                           y_true_test, y_pred_test, 
                           y_true_val=None, y_pred_val=None, 
                           model_name="Model", r2_train=None, r2_test=None, r2_val=None, filename=None):
    
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    random_offsets_train = np.random.normal(0, 0.05, len(y_pred_train))  # Giảm giá trị nhiễu
    plt.scatter(y_true_train + random_offsets_train, y_pred_train + random_offsets_train, alpha=0.6, color='blue', edgecolor='k')
    plt.plot([min(y_true_train), max(y_true_train)], [min(y_true_train), max(y_true_train)], color='red', linestyle='--', linewidth=2)
    plt.xlabel("Giá trị thực tế (Train)")
    plt.ylabel("Giá trị dự đoán (Train)")
    plt.title(f"Train - {model_name} \n R2: {r2_train:.2f}")
    plt.grid()
    
    plt.subplot(1, 3, 2)
    random_offsets_test = np.random.normal(0, 0.05, len(y_pred_test))
    plt.scatter(y_true_test + random_offsets_test, y_pred_test + random_offsets_test, alpha=0.6, color='blue', edgecolor='k')
    plt.plot([min(y_true_test), max(y_true_test)], [min(y_true_test), max(y_true_test)], color='red', linestyle='--', linewidth=2)
    plt.xlabel("Giá trị thực tế (Test)")
    plt.ylabel("Giá trị dự đoán (Test)")
    plt.title(f"Test - {model_name} \n R2: {r2_test:.2f}")
    plt.grid()

    if y_true_val is not None and y_pred_val is not None:
        plt.subplot(1, 3, 3)
        random_offsets_val = np.random.normal(0, 0.05, len(y_pred_val))
        plt.scatter(y_true_val + random_offsets_val, y_pred_val + random_offsets_val, alpha=0.6, color='blue', edgecolor='k')
        plt.plot([min(y_true_val), max(y_true_val)], [min(y_true_val), max(y_true_val)], color='red', linestyle='--', linewidth=2)
        plt.xlabel("Giá trị thực tế (Validation)")
        plt.ylabel("Giá trị dự đoán (Validation)")
        plt.title(f"Validation - {model_name} \n R2: {r2_val:.2f}")
        plt.grid()

    if filename:
        os.makedirs("static/images", exist_ok=True)
        save_path = os.path.join("static/images", filename)
        plt.savefig(save_path)
        print(f"Biểu đồ đã được lưu tại {save_path}")
    
    plt.show()

# Vẽ biểu đồ cho từng mô hình
plot_predictions_multi(y_train, lr_model.predict(X_train), y_test, y_pred_lr, 
                       model_name="Linear Regression", r2_train=r2_score(y_train, lr_model.predict(X_train)), 
                       r2_test=r2_lr, filename="lr_predictions.png")

plot_predictions_multi(y_train, lasso_model.predict(X_train), y_test, y_pred_lasso, 
                       model_name="Lasso Regression", r2_train=r2_score(y_train, lasso_model.predict(X_train)), 
                       r2_test=r2_lasso, filename="lasso_predictions.png")

plot_predictions_multi(y_train, nn_model.predict(X_train), y_test, y_pred_nn, 
                       model_name="Neural Network", r2_train=r2_score(y_train, nn_model.predict(X_train)), 
                       r2_test=r2_nn, filename="nn_predictions.png")

plot_predictions_multi(y_train, stacking_model.predict(X_train), y_test, y_pred_stack, 
                       model_name="Stacking Regressor", r2_train=r2_score(y_train, stacking_model.predict(X_train)), 
                       r2_test=r2_stack, filename="stacking_predictions.png")
