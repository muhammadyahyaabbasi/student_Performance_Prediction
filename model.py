# model.py
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import joblib

# -----------------------------
# 1️⃣ Classification Models
# -----------------------------

# KNN Classifier
def train_knn_classifier(X_train, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

# Logistic Regression Classifier
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

# -----------------------------
# 2️⃣ Regression Models
# -----------------------------

# KNN Regressor
def train_knn_regressor(X_train, y_train, n_neighbors=5):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

# Linear Regression
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# -----------------------------
# 3️⃣ Save / Load Models
# -----------------------------

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)
