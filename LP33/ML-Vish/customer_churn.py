# customer_churn.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# 1. Read dataset
data = pd.read_csv("Churn_Modelling.csv")

# 2. Feature & target selection
X = data.iloc[:, 3:-1]  # Exclude RowNumber, CustomerId, Surname, and Exited
y = data.iloc[:, -1]

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# 3. Split and normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Initialize and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Results
print("Confusion Matrix:\n", cm)
print(f"\nAccuracy: {acc:.4f}")
