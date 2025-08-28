# ml_forecast_linear.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import os

# -------------------------------
# 1. Load the aligned dataset
# -------------------------------
data_path = r"E:\University\Project\healthcare-icu-forecasting\healthcare-icu-forecasting\data\data_processed\merged.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}. Run the feature preparation step first.")

df = pd.read_csv(data_path, parse_dates=["date"], low_memory=False)

print("Dataset shape before cleaning:", df.shape)
print("Columns:", df.columns.tolist())

non_numeric_cols = df.select_dtypes(include=["object"]).columns.tolist()
print("Dropping non-numeric columns:", non_numeric_cols)

df = df.drop(columns=non_numeric_cols)

# Save features to CSV
features_save_path = r"E:\University\Project\healthcare-icu-forecasting\healthcare-icu-forecasting\data\data_processed\features.csv"
df.to_csv(features_save_path, index=False)
print(f"Features saved to {features_save_path}")
# -------------------------------
# 2. Select features + target
# -------------------------------
if "icu_patients" not in df.columns:
    raise KeyError("Target column 'icu_patients' is missing in the dataset.")

# Example feature set: COVID + Flu cases
feature_cols = [col for col in df.columns if col not in ["date", "icu_patients"]]
target_col = "icu_patients"

X = df[feature_cols]
y = df[target_col]

# Drop rows where target is missing
mask = ~y.isna()
X, y = X[mask], y[mask]

print("Shape after dropping NA in target:", X.shape)
imputer = SimpleImputer(strategy="mean")  # replaces NaN with column mean
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

if X.isna().sum().sum() > 0:
    raise ValueError("Still found NaNs after imputation!")
# -------------------------------
# 3. Train-test split
# -------------------------------
if len(X) < 5:
    raise ValueError(f"Too few samples ({len(X)}) after cleaning. Need at least 5 to train/test split.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# -------------------------------
# 4. Train model
# -------------------------------
model = LinearRegression()

if X_train.shape[0] == 0 or y_train.shape[0] == 0:
    raise ValueError("Training data is empty after split. Check preprocessing step.")

model.fit(X_train, y_train)

# -------------------------------
# 5. Evaluate
# -------------------------------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")

# -------------------------------
# 6. Save model
# -------------------------------
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "linear_forecast.pkl")

joblib.dump(model, model_path)
print(f"Model saved at {model_path}")
print("\n✅ Linear regression model training and evaluation complete.")