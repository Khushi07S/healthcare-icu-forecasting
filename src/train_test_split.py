import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_test_split_time(df, target_col, split_date):
    """
    Splits dataframe into train/test sets based on a time split.

    Args:
        df (pd.DataFrame): Input dataframe (already aligned features + target).
        target_col (str): Target column name (e.g., 'ICU_admissions').
        split_date (str): Date string (YYYY-MM-DD) for splitting.

    Returns:
        X_train, X_test, y_train, y_test
    """
    train = df[df['date'] <= split_date].copy()
    test = df[df['date'] > split_date].copy()

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]

    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    return X_train, X_test, y_train, y_test


def baseline_forecast(y_train, y_test, method="naive"):
    if method == "naive":
        forecast = np.repeat(y_train.iloc[-1], len(y_test))
    elif method == "seasonal":
        lag = 7
        forecast = []
        for i in range(len(y_test)):
            if i < lag:
                forecast.append(y_train.iloc[-lag + i])
            else:
                forecast.append(y_test.iloc[i - lag])
        forecast = np.array(forecast)
    else:
        raise ValueError("method must be 'naive' or 'seasonal'")

    # Evaluation
    mae = mean_absolute_error(y_test, forecast)
    rmse = mean_squared_error(y_test, forecast) ** 0.5  # <-- fixed
    mape = np.mean(np.abs((y_test - forecast) / y_test)) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    return forecast, metrics


if __name__ == "__main__":
    # === Load aligned dataset ===
    df = pd.read_csv(r"E:\University\Project\healthcare-icu-forecasting\healthcare-icu-forecasting\data\data_processed\merged.csv", parse_dates=["date"])

    # === Train-test split (example: train till Dec 2023) ===
    X_train, X_test, y_train, y_test = train_test_split_time(
        df, target_col="icu_patients", split_date="2023-12-31"
    )
    # Ensure target column has no NaNs
    y_train = y_train.fillna(method="ffill").fillna(method="bfill")
    y_test = y_test.fillna(method="ffill").fillna(method="bfill")


    # === Naive Forecast ===
    naive_preds, naive_metrics = baseline_forecast(y_train, y_test, method="naive")
    print("\nNaive Forecast Metrics:", naive_metrics)

    # === Seasonal Forecast (weekly) ===
    seasonal_preds, seasonal_metrics = baseline_forecast(y_train, y_test, method="seasonal")
    print("\nSeasonal Naive Forecast Metrics:", seasonal_metrics)

    # === Save baseline results ===
    results = pd.DataFrame({
        "date": X_test["date"],
        "actual": y_test.values,
        "naive_pred": naive_preds,
        "seasonal_pred": seasonal_preds
    })
    results.to_csv(r"E:\University\Project\healthcare-icu-forecasting\results\baseline_forecast.csv", index=False)
    print("\n✅ Baseline forecast saved to results/baseline_forecast.csv")
