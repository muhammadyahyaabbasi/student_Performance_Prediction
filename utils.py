# utils.py
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score

# -----------------------------
# 1️⃣ Classification Evaluation
# -----------------------------
def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, cm, report

# -----------------------------
# 2️⃣ Regression Evaluation
# -----------------------------
def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2
