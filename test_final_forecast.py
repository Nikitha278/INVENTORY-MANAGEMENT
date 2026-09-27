from preprocessing import load_and_clean_data, create_features
from forecasting import generate_best_forecast

df = load_and_clean_data("retail_store_inventory.csv")
df = create_features(df)

result = generate_best_forecast(df, "S002", "P0013", steps=7)

print(result)
