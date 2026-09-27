from preprocessing import load_and_clean_data, create_features
from inventory_logic import generate_store_level_summary

df = load_and_clean_data("retail_store_inventory.csv")
df = create_features(df)

summary = generate_store_level_summary(df)

for s in summary:
    print(s)
