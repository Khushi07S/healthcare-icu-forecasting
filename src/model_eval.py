import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load features (from step 27 output)
features = pd.read_csv(r"E:\University\Project\healthcare-icu-forecasting\healthcare-icu-forecasting\data\data_processed\features.csv")

# Drop any rows with NaN (if present)
features = features.dropna()

# 2. Separate features (X) and target (y)
X = features.drop(columns=["total_cases"])
y = features["total_cases"]

# 3. Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # keep time-order intact
)

# 4. Load saved model (from step 27)
model = joblib.load("models/forecasting_model.pkl")

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# 7. Print results
print("📊 Model Evaluation Results")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 8. Save evaluation metrics for reference
metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv("models/model_evaluation_metrics.csv", index=False)

print("\n✅ Evaluation metrics saved to models/model_evaluation_metrics.csv")
