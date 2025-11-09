# 🚗 Uber Fare Prediction – Concise & Complete Version
import pandas as pd, numpy as np
from math import radians, cos, sin, asin, sqrt
import seaborn as sns, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1️⃣ Load & Preprocess Data
df = pd.read_csv('uber.csv')
df.drop(columns=['Unnamed: 0', 'key'], inplace=True, errors='ignore')
df.dropna(subset=['fare_amount', 'pickup_latitude', 'pickup_longitude',
                  'dropoff_latitude', 'dropoff_longitude'], inplace=True)
df = df[(df.fare_amount > 0) & (df.passenger_count.between(1, 6))]
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
df['hour'], df['dayofweek'], df['month'] = df.pickup_datetime.dt.hour, df.pickup_datetime.dt.dayofweek, df.pickup_datetime.dt.month

# Calculate Haversine distance (in km)
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    return 6371 * 2 * asin(sqrt(sin((lat2 - lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2 - lon1)/2)**2))

df['distance_km'] = df.apply(lambda x: haversine(
    x.pickup_latitude, x.pickup_longitude, x.dropoff_latitude, x.dropoff_longitude), axis=1)
df = df[(df.distance_km > 0) & (df.distance_km < 200)]

print(f"✅ Cleaned Data Shape: {df.shape}")

# 2️⃣ Identify Outliers (visual check)
plt.figure(figsize=(5,3))
sns.boxplot(df['fare_amount'])
plt.title("Fare Amount Outliers")
plt.show()

# Optional: Remove extreme outliers
df = df[df['fare_amount'] < 100]

# 3️⃣ Correlation Analysis
plt.figure(figsize=(8,5))
sns.heatmap(df[['fare_amount','distance_km','passenger_count','hour','dayofweek','month']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# 4️⃣ Train Models (Linear Regression & Random Forest)
X = df[['distance_km', 'passenger_count', 'hour', 'dayofweek', 'month']]
y = df['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{name}: R²={r2:.3f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

print("\n🔹 Model Evaluation Results:")
evaluate(LinearRegression(), "Linear Regression")
evaluate(RandomForestRegressor(n_estimators=100, random_state=42), "Random Forest Regression")
