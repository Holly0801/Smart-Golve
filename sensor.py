import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import accuracy_score, classification_report

# 1. 加载数据
data = pd.read_csv(r"F:\PythonProject\sensor_data.csv")  # 假设数据保存为CSV文件
X = data.iloc[:, :-1].values  # 特征
y = data.iloc[:, -1].values   # 标签

# 2. 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 特征提取
def extract_features(data):
    features = []
    for sample in data:
        mean = np.mean(sample)
        std = np.std(sample)
        max_val = np.max(sample)
        min_val = np.min(sample)
        features.append([mean, std, max_val, min_val])
    return np.array(features)

X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

# 5. SVM模型
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_features, y_train)
y_pred_svm = svm_model.predict(X_test_features)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# 6. KNN模型
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_features, y_train)
y_pred_knn = knn_model.predict(X_test_features)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))