import pandas as pd

df = pd.read_parquet("data/math/train.parquet")
# df = pd.read_parquet("data/math/evaluation/aime24.parquet")

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print()
print(df.head())
print()

for col in df.columns:
    print(f"--- {col} ---")
    print(repr(df[col].iloc[0]))
    print()