# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# 1️⃣ Load dataset(s)
def load_data(file_paths=["student-mat.csv", "student-por.csv"]):
    dfs = [pd.read_csv(path) for path in file_paths]  # Read each file
    df = pd.concat(dfs, ignore_index=True)           # Combine datasets
    df = df.drop_duplicates()                        # Remove duplicates
    return df

# 2️⃣ Create target variables
def create_targets(df):
    # Classification target: Pass/Fail
    df['PassFail'] = df['G3'].apply(lambda x: 'Pass' if x >= 10 else 'Fail')
    # Regression target: Final Marks
    df['FinalMarks'] = df['G3']
    return df

# 3️⃣ Select relevant features
def select_features(df):
    features = ['studytime', 'absences', 'failures', 'G1', 'G2', 'Medu', 'Fedu', 'traveltime']
    X = df[features]
    return X

# 4️⃣ Scale numeric features
def scale_features(X, scaler_path="scaler.pkl"):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, scaler_path)  # Save scaler for later use
    return X_scaled, scaler

# 5️⃣ Split data into train and test sets
def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)