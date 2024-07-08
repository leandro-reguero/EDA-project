import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# modelado
from sklearn.model_selection import train_test_split

# escalado
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


file = 'https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv'
df_raw = pd.read_csv(file)

# As explored in the notebook EDA, we drop the following columns:
df2 = df_raw.drop(['name', 'host_name', 'last_review', 'reviews_per_month', 'id', 'host_id'], axis=1)

# Moreover, according to the following conclusion and exploration, we reduce the dataset even more
print("too many zeros in 'availability_365' observed, and no correlation with price seen. Suggesting to drop")
print("same with 'number_of_reviews'")
df = df2.drop(['availability_365', 'number_of_reviews', 'latitude', 'longitude'], axis=1)

print(len(df))

# Now we have reduced dataset, we want to scale and model it
# We split the set for train and test 
X = df.drop("price", axis = 1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


print(X_train.head())
# We define the numerical variables in our dataset
num_variables = ["minimum_nights", "calculated_host_listings_count"]

# initiating and training the scaler
scaler = StandardScaler()
scaler.fit(X_train[num_variables])

# apply scaler on both sides of our dataset and convert to dataset (since scaler returns array)
X_train_num_scal = scaler.transform(X_train[num_variables])
X_train_num_scal = pd.DataFrame(X_train_num_scal, index=X_train.index, columns=num_variables)

X_test_num_scal = scaler.transform(X_test[num_variables])
X_test_num_scal = pd.DataFrame(X_test_num_scal, index=X_test.index, columns=num_variables)

print(X_train_num_scal.head(10))

# Now we codify the categorical variables

cat_variables = ['neighbourhood_group', 'neighbourhood', 'room_type']

# The column 'neighbourhood' has a lot of categories, so we will apply label encoding for it
X_train_cat_le = X_train.copy()
X_test_cat_le = X_test.copy()

# starting the encoder
label_encoder_neighbourhood = LabelEncoder()
label_encoder_neighbourhood.fit(X_train['neighbourhood'])

# apply the encoder
X_train_cat_le['neighbourhood_le'] = label_encoder_neighbourhood.transform(X_train['neighbourhood'])
X_test_cat_le['neighbourhood_le'] = label_encoder_neighbourhood.transform(X_test['neighbourhood'])

print(X_train_cat_le.head(10))
