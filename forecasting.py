import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb


# -----------------------------
# Utility Function
# -----------------------------

def calculate_mape(y_true, y_pred):
    y_true = np.where(y_true < 5, 5, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# -----------------------------
# ARIMA Evaluation
# -----------------------------

def evaluate_arima(df, store_id, product_id):

    data = df[(df['Store ID'] == store_id) &
              (df['Product ID'] == product_id)]

    data = data.set_index('Date')
    series = data['Units Sold']

    train = series[:-30]
    test = series[-30:]

    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=30)

    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = calculate_mape(test, forecast)

    return mae, rmse, mape


# -----------------------------
# XGBoost Evaluation
# -----------------------------

def xgboost_forecast(df, store_id, product_id):

    data = df[(df['Store ID'] == store_id) &
              (df['Product ID'] == product_id)]

    features = [
        'Inventory Level',
        'Price',
        'Discount',
        'Competitor Pricing',
        'day_of_week',
        'month',
        'is_weekend',
        'lag_1',
        'lag_7',
        'rolling_mean_7',
        'rolling_mean_14'
    ]

    X = data[features]
    y = data['Units Sold']

    X_train = X[:-30]
    X_test = X[-30:]
    y_train = y[:-30]
    y_test = y[-30:]

    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = calculate_mape(y_test, predictions)

    return mae, rmse, mape


# -----------------------------
# Model Comparison
# -----------------------------

def compare_models(df, store_id, product_id):

    arima_mae, arima_rmse, arima_mape = evaluate_arima(df, store_id, product_id)
    xgb_mae, xgb_rmse, xgb_mape = xgboost_forecast(df, store_id, product_id)

    if xgb_mae < arima_mae:
        best_model = "XGBoost"
    else:
        best_model = "ARIMA"

    return {
        "ARIMA_MAE": arima_mae,
        "ARIMA_RMSE": arima_rmse,
        "ARIMA_MAPE": arima_mape,
        "XGB_MAE": xgb_mae,
        "XGB_RMSE": xgb_rmse,
        "XGB_MAPE": xgb_mape,
        "Best_Model": best_model
    }


# -----------------------------
# Final Multi-Step Forecast
# -----------------------------

def generate_best_forecast(df, store_id, product_id, steps=7):

    comparison = compare_models(df, store_id, product_id)
    best_model = comparison["Best_Model"]

    data = df[(df['Store ID'] == store_id) &
              (df['Product ID'] == product_id)].copy()

    data = data.sort_values("Date")

    if best_model == "ARIMA":

        data = data.set_index("Date")
        series = data['Units Sold']

        model = ARIMA(series, order=(5, 1, 0))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=steps)

        return {
            "Best_Model": best_model,
            "Forecast": forecast.tolist()
        }

    else:  # XGBoost Multi-Step Forecast

        features = [
            'Inventory Level',
            'Price',
            'Discount',
            'Competitor Pricing',
            'day_of_week',
            'month',
            'is_weekend',
            'lag_1',
            'lag_7',
            'rolling_mean_7',
            'rolling_mean_14'
        ]

        X = data[features]
        y = data['Units Sold']

        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5
        )

        model.fit(X, y)

        forecast = []
        last_known = data.copy()

        for _ in range(steps):

            last_row = last_known.iloc[-1]

            input_features = last_row[features].values.reshape(1, -1)

            next_pred = model.predict(input_features)[0]

            forecast.append(float(next_pred))

            # Create new row
            new_row = last_row.copy()
            new_row['Units Sold'] = next_pred
            new_row['lag_1'] = next_pred

            last_known = pd.concat(
                [last_known, new_row.to_frame().T],
                ignore_index=True
            )

        return {
            "Best_Model": best_model,
            "Forecast": forecast
        }
