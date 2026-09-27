from preprocessing import load_and_clean_data, create_features
from forecasting import compare_models

df = load_and_clean_data("retail_store_inventory.csv")
df = create_features(df)

result = compare_models(df, "S001", "P0001")

print(result)
