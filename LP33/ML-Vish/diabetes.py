# 🩺 Diabetes Prediction using K-Nearest Neighbors
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# 1️⃣ Load Dataset
df = pd.read_csv("diabetes.csv")

# 2️⃣ Replace medically invalid 0s with mean
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].mean())

# 3️⃣ Feature & Target Split
X, y = df.iloc[:, :-1], df['Outcome']

# 4️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Initialize & Train Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 6️⃣ Evaluate Model
cm = confusion_matrix(y_test, y_pred)
acc, prec, rec = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred)

# 7️⃣ Display Results
print("Confusion Matrix:\n", cm)
print(f"\nAccuracy: {acc:.4f}")
print(f"Error Rate: {1 - acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
