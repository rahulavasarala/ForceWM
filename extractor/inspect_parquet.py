import pandas as pd

df = pd.read_parquet("/Users/rahulavasarala/Desktop/ForceWM/data_storage/chicken_extracted/dataset.parquet")
print(df.head())
print(df.columns)
print(df.shape)

df.to_csv("/Users/rahulavasarala/Desktop/ForceWM/extractor/out.csv", index=False)