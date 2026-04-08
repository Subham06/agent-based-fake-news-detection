import pandas as pd

df = pd.read_csv("news_data.csv")

print(df.shape)
print(df.columns)
print(df['publishedAt'].value_counts())