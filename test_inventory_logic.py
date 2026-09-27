from preprocessing import load_and_clean_data, create_features
from inventory_logic import calculate_inventory_decision

df = load_and_clean_data("retail_store_inventory.csv")
df = create_features(df)

result = calculate_inventory_decision(df, "S002", "P0001")

print(result)
