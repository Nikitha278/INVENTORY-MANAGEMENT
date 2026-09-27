from preprocessing import load_and_clean_data, create_features
from forecasting import xgboost_forecast

df = load_and_clean_data("retail_store_inventory.csv")
df = create_features(df)

mae, rmse, mape = xgboost_forecast(df, "S001", "P0001")

print("XGBoost MAE:", mae)
print("XGBoost RMSE:", rmse)
print("XGBoost MAPE:", mape)

