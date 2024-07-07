import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

file = 'https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv'
df_raw = pd.read_csv(file)


df2 = df_raw.drop(['name', 'host_name', 'last_review', 'reviews_per_month', 'id', 'host_id'], axis=1)
print(df2.head)
print("too many zeros in 'availability_365' observed, and no correlation with price seen. Suggesting to drop")
print("same with 'number_of_reviews'")
df = df2.drop(['availability_365', 'number_of_reviews'], axis=1)
print(df.head())

