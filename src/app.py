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
print(df_raw.info())
print("")

# First, lets analyze the Null values
contain_na = df_raw.isna().any().any()
print(f"Does the dataset contain NAs? : ", contain_na)
for column in df_raw.columns:
    if df_raw[column].isna().any():
        print(f"Column '{column}' contains NaN values")
print("")

# Now we search for duplicates
repes = df_raw['id'].duplicated().sum()
if repes == 0:
    print(f"There are {repes} duplicates in the 'id' column, which means that all rows are unique entries")
else:
    print(f"There are {repes} duplicated values in the dataset. Mayday!")

print("")

# We drop the irrelevant / problematic columns
df = df_raw.drop(['name', 'host_name', 'last_review', 'reviews_per_month', 'host_id', 'id'], axis=1)
print("")

print("Now this is our dataset without NaNs, Duplicates or irrelevant columns: ")
print(df.head(10))
print("")

# Analyzing the types of data that we have and labeling it
num_variables = ["minimum_nights", "number_of_reviews", "calculated_host_listings_count", "availability_365"]
cat_variables = ["neighbourhood_group", "neighbourhood", "room_type"]

print(f"There are {len(num_variables)} numerical variables")
print(f"There are {len(cat_variables)} categorical variables")
print("")

# Categorical variables analysis:

fig, axis = plt.subplots(2, 2, figsize=(10, 7))
sns.histplot(ax = axis[0,0], data = df, x = "neighbourhood_group").set_xticks([])
sns.histplot(ax = axis[0,1], data = df, x = "neighbourhood", legend=True).set_xticks([])

sns.histplot(ax = axis[1,0], data = df, x = "room_type")
sns.histplot(ax = axis[1,1], data = df, x = "availability_365")


plt.tight_layout()
# plt.show()
print("We can observe that Brooklyn and Manhattan contain the most houses")

shared_room_perc = df['room_type'].value_counts()['Shared room'] / df['room_type'].value_counts().sum()
print(f"Shared room is the least common room type by far ({round(shared_room_perc*100, 3)}% Shared Rooms)")
not_available = df[df['availability_365'] == 0]
print(f"We also observe a lot of zeros in availability_365 (total of: {len(not_available)})")

# Analyzing numerical variables:
fig, axis = plt.subplots(4, 2, figsize = (10, 14), gridspec_kw = {"height_ratios": [6, 1, 6, 1]})

sns.histplot(ax = axis[0, 0], data = df, x = "price")
sns.boxplot(ax = axis[1, 0], data = df, x = "price")

sns.histplot(ax = axis[0, 1], data = df, x = "minimum_nights").set_xlim(0, 200)
sns.boxplot(ax = axis[1, 1], data = df, x = "minimum_nights")

sns.histplot(ax = axis[2, 0], data = df, x = "number_of_reviews")
sns.boxplot(ax = axis[3, 0], data = df, x = "number_of_reviews")

sns.histplot(ax = axis[2,1], data = df, x = "calculated_host_listings_count").set_xlim(0, 70)
sns.boxplot(ax = axis[3, 1], data = df, x = "calculated_host_listings_count")


plt.tight_layout()
# plt.show()

# Combined variables analysis
# numerico-numeric
fig, axis = plt.subplots(4, 2, figsize = (10, 16))

sns.regplot(ax = axis[0, 0], data = df, x = "minimum_nights", y = "price")
sns.heatmap(df[["price", "minimum_nights"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)

sns.regplot(ax = axis[0, 1], data = df, x = "number_of_reviews", y = "price").set(ylabel = None)
sns.heatmap(df[["price", "number_of_reviews"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])

sns.regplot(ax = axis[2, 0], data = df, x = "calculated_host_listings_count", y = "price").set(ylabel = None)
sns.heatmap(df[["price", "calculated_host_listings_count"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 0]).set(ylabel = None)
fig.delaxes(axis[2, 1])
fig.delaxes(axis[3, 1])

plt.tight_layout()
# plt.show()

print("")
print("No relation between price and min_nights or num. of reviews")

# Cat - cat analysis

fig, axis = plt.subplots(figsize = (5, 4))
sns.countplot(data = df, x = "room_type", hue = "neighbourhood_group")

plt.tight_layout()
# plt.show()

print("Manhattan contains the most amount of houses, and the most entire apts to rent")
print("Brooklyn has more private rooms")
print("Staten Island has very few airbnbs")


# num - cat analysis - Factorizing the Room Type and Neighborhood Data
df["room_type"] = pd.factorize(df["room_type"])[0]
df["neighbourhood_group"] = pd.factorize(df["neighbourhood_group"])[0]
df["neighbourhood"] = pd.factorize(df["neighbourhood"])[0]

fig, axes = plt.subplots(figsize=(15, 15))

sns.heatmap(df[["neighbourhood_group", "neighbourhood", "room_type", "price", "minimum_nights", "number_of_reviews", "calculated_host_listings_count", "availability_365"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()
# plt.show()

print("")
print("There is significant relationship between price and neighbourhood")
print("There is some relationship between availability and neighbourhood or host listings")


# removing outliers (based on plotting on the explore file)
df = df[df['price'] > 0]
sns.boxplot(df['minimum_nights'])
df = df[df['minimum_nights'] <= 400]
df = df[df['number_of_reviews'] <= 500]
df = df[df['calculated_host_listings_count'] > 0]

# find missing values
print(df.isnull().sum())




# Now we have reduced dataset, we want to scale and model it
# We split the set for train and test 
X = df.drop("price", axis = 1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

print(X_train.head())

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
# but first, (as you can see in the notebook file) I have dropped neighbourhoods containing less than 30 houses
# so that we dont have neighbourhoods not appearing on one side of the split
x = 28
neighbourhoods_counts = df['neighbourhood'].value_counts()
bigger_nh = neighbourhoods_counts[neighbourhoods_counts >= x].index
nh_filtered_df = df[df['neighbourhood'].isin(bigger_nh)]


X_train_cat_le = X_train.copy()
X_test_cat_le = X_test.copy()

# starting the encoder
label_encoder_neighbourhood = LabelEncoder()
label_encoder_neighbourhood.fit(X_train['neighbourhood'])

# apply the encoder
X_train_cat_le['neighbourhood_le'] = label_encoder_neighbourhood.transform(X_train['neighbourhood'])
X_test_cat_le['neighbourhood_le'] = label_encoder_neighbourhood.transform(X_test['neighbourhood'])

print(X_train_cat_le.head(10))
