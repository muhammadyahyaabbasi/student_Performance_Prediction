# main.py

from preprocessing import load_data, create_targets, select_features, scale_features, split_data
from model import train_knn_classifier, train_logistic_regression, train_knn_regressor, train_linear_regression, save_model
from utils import evaluate_classification, evaluate_regression

# -----------------------------
# 1️⃣ Load & Preprocess Data
# -----------------------------
file_paths = ["student-mat.csv", "student-por.csv"]
df = load_data(file_paths)
df = create_targets(df)

# Select features and targets
X = select_features(df)
X_scaled, scaler = scale_features(X)  # Updated to receive scaler

# Classification target
y_class = df['PassFail']

# Regression target
y_reg = df['FinalMarks']

# Split datasets
X_train_c, X_test_c, y_train_c, y_test_c = split_data(X_scaled, y_class)
X_train_r, X_test_r, y_train_r, y_test_r = split_data(X_scaled, y_reg)

# -----------------------------
# 2️⃣ Train Models
# -----------------------------

# Classification Models
knn_clf = train_knn_classifier(X_train_c, y_train_c)
log_reg = train_logistic_regression(X_train_c, y_train_c)

# Regression Models
knn_reg = train_knn_regressor(X_train_r, y_train_r)
lin_reg = train_linear_regression(X_train_r, y_train_r)

# -----------------------------
# 3️⃣ Evaluate Models
# -----------------------------

# Classification Evaluation
knn_acc, knn_cm, knn_report = evaluate_classification(knn_clf, X_test_c, y_test_c)
log_acc, log_cm, log_report = evaluate_classification(log_reg, X_test_c, y_test_c)

print("===== KNN Classification =====")
print(f"Accuracy: {knn_acc}")
print("Confusion Matrix:")
print(knn_cm)
print("Classification Report:")
print(knn_report)

print("\n===== Logistic Regression =====")
print(f"Accuracy: {log_acc}")
print("Confusion Matrix:")
print(log_cm)
print("Classification Report:")
print(log_report)

# Regression Evaluation
knn_mse, knn_r2 = evaluate_regression(knn_reg, X_test_r, y_test_r)
lin_mse, lin_r2 = evaluate_regression(lin_reg, X_test_r, y_test_r)

print("\n===== KNN Regression =====")
print(f"MSE: {knn_mse:.2f}, R2: {knn_r2:.2f}")

print("\n===== Linear Regression =====")
print(f"MSE: {lin_mse:.2f}, R2: {lin_r2:.2f}")

# -----------------------------
# 4️⃣ Save Models (Optional)
# -----------------------------
save_model(knn_clf, "knn_classifier.pkl")
save_model(log_reg, "logistic_regression.pkl")
save_model(knn_reg, "knn_regressor.pkl")
save_model(lin_reg, "linear_regression.pkl")

print("\nModels saved successfully!")
print("Scaler saved as scaler.pkl")