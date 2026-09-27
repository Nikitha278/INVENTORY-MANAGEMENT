from preprocessing import load_and_clean_data, create_features
from forecasting import evaluate_arima

df = load_and_clean_data("retail_store_inventory.csv")
df = create_features(df)

mae, rmse = evaluate_arima(df, "S001", "P0001")

print("Average Units Sold:",
      df[(df['Store ID']=="S001") &
         (df['Product ID']=="P0001")]['Units Sold'].mean())

print("ARIMA MAE:", mae)
print("ARIMA RMSE:", rmse)
