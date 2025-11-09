import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score

# Pre-Processing
data = pd.read_csv('diabetes.csv')
print(data.head())

print(data.isnull().sum())  # all data is clear

print(data.dtypes)  # all features are in correct type

x = data.drop(columns='Outcome')
y = data['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Standard Scalling
print(data.describe())   # by analyzing min and Max values we chhose Standard Scalling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)   # fit() → learns mean & std from training data
x_test_scaled = scaler.transform(x_test)        # transform() → applies scaling

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train_scaled,y_train)
predictions = model.predict(x_test_scaled)

# 1] Confusion Matrix                                         # Confusion Matrix Interpretation:
Confusion_Matrix =  confusion_matrix(y_test,predictions)      # [[TN  FP]
print(Confusion_Matrix)                                       # [FN  TP]]

# 2] accuracy Score
Accuracy_Score = accuracy_score(y_test,predictions)
print(Accuracy_Score)

# 3] Error Rate
error_rate = 1 - Accuracy_Score
print(error_rate)

# 4] Precision Score
Precision_Score = precision_score(y_test,predictions)
print(Precision_Score)

# 5] Recall Score
Recall_Score = recall_score(y_test,predictions)
print(Recall_Score)