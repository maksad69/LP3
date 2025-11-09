import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# 1] Pre-Process the Dataset
data = pd.read_csv('uber.csv')
print(data)

print(data.head())
print(data.shape)
print(data.describe())
data.info()   # get the information of each feature data type

# convert datatype in other data type
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
data.info()

# Handle Missing Values
print(data.isnull().sum())
data.dropna(subset=['dropoff_longitude', 'dropoff_latitude'], inplace=True)
print(data.isnull().sum())

# 2]  Identify outliers.
print(data.describe())  # Look at min/max using describe()
plt.boxplot(data['fare_amount'])  # Draw boxplots
plt.show()
# Use IQR method to find extreme values
Q1 = data['fare_amount'].quantile(0.25)
Q3 = data['fare_amount'].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

outlier = data[(data['fare_amount'] < lower_limit) | (data['fare_amount'] > upper_limit)]
# data = data[(data['fare_amount'] >= lower_limit) & (data['fare_amount'] <= upper_limit)]    Remove Ouliers by this line
print("Number of outliers:", len(outlier))

# 3] Correlation
# Select only numerical columns
numeric_data = data.select_dtypes(include=['int64','float64'])
numeric_data.head()

#Calculate correlation  => The corr() function gives the pairwise correlation between numeric features.
correlation_matrix = numeric_data.corr()
print(correlation_matrix)

# Visualize correlation with heatmap
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# --- Calculate distance ---   This is for to compute the diatance_km feature
def haversine(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

data['distance_km'] = data.apply(lambda row: haversine(
    row['pickup_latitude'], row['pickup_longitude'],
    row['dropoff_latitude'], row['dropoff_longitude']), axis=1)

# Implement linear regression and random forest regression models.
y = data['fare_amount']
# x = data[['distance_km', 'passenger_count']]
x = data[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]   # by this it takes time consumming, predictions are less accurate


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# 1] Linear Regression Model
# Trainning
model1 = LinearRegression()
model1.fit(x_train,y_train)

#Prediction
ypred_LM = model1.predict(x_test)

#Evalution(actual vs predicted)
LR_RMSE = root_mean_squared_error(y_test,ypred_LM)
LR_R2_SCORE = r2_score(y_test,ypred_LM)
print("📊 Linear Regression → RMSE:", LR_RMSE, " R²:", LR_R2_SCORE)

# 2] Random Forest Model
# Training
model2 = RandomForestRegressor()
model2.fit(x_train,y_train)

# Prediction
ypred_RF = model2.predict(x_test)

# Evalution
RF_RMSE = root_mean_squared_error(y_test,ypred_RF)
RF_R2_SCORE = r2_score(y_test,ypred_RF)
print("📊 Random Forest -> RMSE:", RF_RMSE, "R2:", RF_R2_SCORE)