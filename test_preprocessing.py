from preprocessing import load_and_clean_data, create_features

df = load_and_clean_data("retail_store_inventory.csv")
df = create_features(df)

print(df.head())
print(df.columns)
