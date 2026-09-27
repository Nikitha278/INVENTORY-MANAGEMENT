from preprocessing import load_and_clean_data, create_features
from forecasting import arima_forecast

df = load_and_clean_data("retail_store_inventory.csv")
df = create_features(df)

forecast = arima_forecast(df, "S001", "P0001", steps=7)

print(forecast)
