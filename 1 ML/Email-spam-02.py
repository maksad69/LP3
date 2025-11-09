import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Data pre-processing
# -----------------------------
# 1️⃣ Load and inspect dataset
# -----------------------------
data = pd.read_csv("emails.csv")
print("First 5 rows:\n", data.head())
print("\nMissing values:\n", data.isnull().sum())

# Drop missing values if any
data.dropna(inplace=True)

# print(data.shape)

# -----------------------------
# 2️⃣ Select features and target
# -----------------------------
# word frequency columns (col 1 to 3000)    # .values converts it into a NumPy array for model training.
X = data.iloc[:, 1:3001]                    # iloc means “index-based selection”.
# last column: 1 = spam, 0 = not spam       # : means select all rows (all emails).
Y = data.iloc[:, -1].values                 # 1:3001 means select columns from index 1 to 3000 (because Python indexing stops before 3001).
                                            # -1 means the last column in the dataset.

# -----------------------------
# 3️⃣ Split dataset
# -----------------------------
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# -----------------------------
# 4️⃣ Feature scaling (important for KNN)
# -----------------------------
# Scaling makes all features have similar range, so the model treats them fairly.
# Ex :- word1 fequncy = 500  and word2 frequncy = 5. here by scalling we get or convert it to close values
scaler = StandardScaler(with_mean=False)             # Create a scaler tool (Object)
x_train_scaled = scaler.fit_transform(x_train)       # Learn how to scale and apply it to training data
x_test_scaled = scaler.transform(x_test)             # Apply the same scaling to test data  (fit is not used here because standardscalling learns during train and use that as it is during test)

# -----------------------------
# 5️⃣ Support Vector Machine
# -----------------------------
model_svm = SVC(kernel='linear', C=1.0)
model_svm.fit(x_train_scaled,y_train)
svc_pred = model_svm.predict(x_test_scaled)

print("\n----- Support Vector Machine -----")
print("SVM Accuracy:", accuracy_score(y_test, svc_pred))
print("\nSVM Classification Report:\n", classification_report(y_test, svc_pred))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, svc_pred))

# -----------------------------
# 6️⃣ K-Nearest Neighbors
# -----------------------------
model_knn = KNeighborsClassifier()
model_knn.fit(x_train_scaled,y_train)   # fit means learns from tranning data
knn_pred = model_knn.predict(x_test_scaled)

print("\n----- K- Nearest Neighbors -----")
print("SVM Accuracy:", accuracy_score(y_test, knn_pred))
print("\nSVM Classification Report:\n", classification_report(y_test, knn_pred))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))

# # ---------------------------------
# # 7️⃣ Compare Model Accuracies
# # ---------------------------------
# results = {
#     'SVM': accuracy_score(y_test, svc_pred),
#     'KNN': accuracy_score(y_test, knn_pred)
# }
#
# sns.barplot(x=list(results.keys()), y=list(results.values()), palette="coolwarm")
# plt.title("📊 Model Accuracy Comparison")
# plt.ylabel("Accuracy")
# plt.ylim(0.8, 1)
# plt.show()

# # OPTIONAL NOT IMPORTANT
# # Finding outliers using IQR
# Q1 = data.quantile(0.25)
# Q3 = data.quantile(0.75)
# IQR = Q3 - Q1
#
# lower_limit = Q1 - 1.5*IQR
# upper_limit = Q3 + 1.5*IQR
# outlier = data[(data < lower_limit) | (data < upper_limit)]
# print(outlier)

