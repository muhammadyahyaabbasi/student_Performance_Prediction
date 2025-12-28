# gui.py
import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
import os

# -----------------------------
# Create root window first (but keep it hidden initially)
# -----------------------------
root = tk.Tk()
root.withdraw()  # Hide window until models are loaded

# -----------------------------
# Load saved models with error handling
# -----------------------------
try:
    knn_clf = joblib.load("knn_classifier.pkl")
    log_reg = joblib.load("logistic_regression.pkl")
    knn_reg = joblib.load("knn_regressor.pkl")
    lin_reg = joblib.load("linear_regression.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    messagebox.showerror("Error", f"Model files not found. Please run main.py first to train the models.\nMissing: {e.filename}")
    root.destroy()
    exit(1)
except Exception as e:
    messagebox.showerror("Error", f"Error loading models: {str(e)}")
    root.destroy()
    exit(1)

# -----------------------------
# GUI Window Configuration
# -----------------------------
root.deiconify()  # Show the window now that models are loaded
root.title("Student Performance Prediction")
root.geometry("500x600")

# Labels and Entries for features
features = ['studytime', 'absences', 'failures', 'G1', 'G2', 'Medu', 'Fedu', 'traveltime']
entries = {}

for i, feature in enumerate(features):
    label = tk.Label(root, text=feature.capitalize())
    label.grid(row=i, column=0, padx=10, pady=5, sticky="w")
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries[feature] = entry

# -----------------------------
# Prediction Function
# -----------------------------
def predict():
    try:
        # Collect user input
        data = [float(entries[feat].get()) for feat in features]
        data_np = np.array(data).reshape(1, -1)
        
        # Scale the input data (CRITICAL: models were trained on scaled data)
        data_scaled = scaler.transform(data_np)
        
        # Classification Predictions
        knn_result = knn_clf.predict(data_scaled)[0]
        log_result = log_reg.predict(data_scaled)[0]
        
        # Regression Predictions
        knn_marks = knn_reg.predict(data_scaled)[0]
        lin_marks = lin_reg.predict(data_scaled)[0]
        
        # Show results
        result_text = f"""
KNN Classification: {knn_result}
Logistic Regression: {log_result}
KNN Regression Marks: {knn_marks:.2f}
Linear Regression Marks: {lin_marks:.2f}
"""
        messagebox.showinfo("Prediction Result", result_text)
        
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"An error occurred: {str(e)}")

# -----------------------------
# Predict Button
# -----------------------------
predict_btn = tk.Button(root, text="Predict Performance", command=predict, bg="green", fg="white", font=("Arial", 12, "bold"))
predict_btn.grid(row=len(features), column=0, columnspan=2, pady=20)

# Run the GUI
root.mainloop()