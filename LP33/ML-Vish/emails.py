# 📧 Email Spam Detection using KNN and SVM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1️⃣ Load Dataset
df = pd.read_csv("emails.csv")
X, y = df.iloc[:, 1:3001], df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2️⃣ K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_knn = knn.predict(X_test)

print("\n🔹 KNN Results:")
print("Accuracy:", accuracy_score(y_test, y_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_knn))

# 3️⃣ Support Vector Machine
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)
y_svm = svm.predict(X_test)

print("\n🔹 SVM Results:")
print("Accuracy:", accuracy_score(y_test, y_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_svm))

# 4️⃣ Model Comparison
print("\n📊 Model Comparison:")
print(pd.DataFrame({
    'Model': ['KNN', 'SVM'],
    'Accuracy': [accuracy_score(y_test, y_knn), accuracy_score(y_test, y_svm)]
}))
