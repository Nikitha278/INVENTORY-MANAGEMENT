import pandas as pd

df = pd.read_csv("retail_store_inventory.csv")

print("First 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nInfo:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())
