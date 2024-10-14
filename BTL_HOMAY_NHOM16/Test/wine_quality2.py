# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def ExportFilePickle(model_name, model):
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(".\\train_models", exist_ok=True)
    save_path = os.path.join(".\\train_models", f'{model_name}_model.pkl')  # Đường dẫn lưu mô hình
    with open(save_path, 'wb') as file:
        pickle.dump(model, file)  # Lưu mô hình vào file

# Tải dữ liệu Wine Quality
url = ".\\data_set\\winequality-red2.csv"
data = pd.read_csv(url, delimiter=';')
# Kiểm tra thông tin tổng quan về dữ liệu
print("Thông tin về dữ liệu:")
print(data.info())
print("\nCác giá trị bị thiếu trong từng cột:")
print(data.isnull().sum())
print(data.isna().sum())

# Hiển thị thống kê cơ bản của dữ liệu
print("\nThống kê tổng quan của dữ liệu:")
print(data.describe())

# Xử lý dữ liệu bị thiếu (nếu có)
# Nếu có giá trị bị thiếu, ta có thể loại bỏ hoặc thay thế bằng giá trị trung bình
if data.isnull().sum().sum() > 0:
    # Thay thế giá trị bị thiếu bằng giá trị trung bình của cột
    data.fillna(data.mean(), inplace=True)

# Tách dữ liệu thành đầu vào (X) và đầu ra (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Tính Z-score và loại bỏ các outliers
z_scores = np.abs(stats.zscore(X))
filtered_entries = (z_scores < 3).all(axis=1)
X_filtered = X[filtered_entries]
y_filtered = y[filtered_entries]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_filtered, test_size=0.2, random_state=42)

# Huấn luyện mô hình Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
ExportFilePickle("Linear Regression", linear_model)
# Dự đoán và đánh giá
y_pred_linear = linear_model.predict(X_test)

# Huấn luyện mô hình Lasso Regression
lasso_model = Lasso(alpha=0.0001)
lasso_model.fit(X_train, y_train)
ExportFilePickle("Lasso", lasso_model)
# Dự đoán và đánh giá
y_pred_lasso = lasso_model.predict(X_test)

# Huấn luyện mô hình Neural Network
nn_model = MLPRegressor(hidden_layer_sizes=(100, 100),activation='tanh', max_iter=500, alpha=0.1, random_state=42)
nn_model.fit(X_train, y_train)

ExportFilePickle("Neuron Network", nn_model)
# Dự đoán và đánh giá
y_pred_nn = nn_model.predict(X_test)

# Kết hợp Linear Regression, Lasso và Neural Network bằng Stacking
estimators = [('linear', linear_model), ('lasso', lasso_model), ('nn', nn_model)]
stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

ExportFilePickle("Stacking", stacking_model)
# Huấn luyện mô hình Stacking
stacking_model.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred_stacking = stacking_model.predict(X_test)

# Lưu trữ kết quả các mô hình
results = {
    'Linear Regression': [np.sqrt(mean_squared_error(y_test, y_pred_linear)), mean_absolute_error(y_test, y_pred_linear), r2_score(y_test, y_pred_linear)],
    'Lasso Regression': [np.sqrt(mean_squared_error(y_test, y_pred_lasso)), mean_absolute_error(y_test, y_pred_lasso), r2_score(y_test, y_pred_lasso)],
    'Neural Network': [np.sqrt(mean_squared_error(y_test, y_pred_nn)), mean_absolute_error(y_test, y_pred_nn), r2_score(y_test, y_pred_nn)],
    'Stacking': [np.sqrt(mean_squared_error(y_test, y_pred_stacking)), mean_absolute_error(y_test, y_pred_stacking), r2_score(y_test, y_pred_stacking)]
}

# Hiển thị kết quả
results_df = pd.DataFrame(results, index=['RMSE', 'MAE', 'R2 Score'])
print(results_df)

# Vẽ biểu đồ so sánh dự đoán của từng mô hình
def plot_predictions(y_test, y_pred, model_name, r2):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolor='k')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
    plt.xlabel("Giá trị thực tế")
    plt.ylabel("Giá trị dự đoán")
    plt.title(f"Biểu đồ so sánh giá trị thực tế và dự đoán - {model_name} \n R2: {r2}")
    plt.grid()
    plt.show()

# Dự đoán cho từng mô hình và vẽ biểu đồ
plot_predictions(y_test, y_pred_linear, "Linear Regression", round(r2_score(y_test, y_pred_linear), 2))
plot_predictions(y_test, y_pred_lasso, "Lasso Regression", round(r2_score(y_test, y_pred_lasso), 2))
plot_predictions(y_test, y_pred_nn, "Neural Network", round(r2_score(y_test, y_pred_nn), 2))
plot_predictions(y_test, y_pred_stacking, "Stacking Model", round(r2_score(y_test, y_pred_stacking), 2))