import pandas as pd
import seaborn as sns


df = pd.read_csv('bank_customer.csv')

df.shape

df.info()

df.columns

df.head()

#input data
x = df[['CreditScore','Age', 'Tenure', 'Balance' ,'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']]

#output data 
y = df['Exited']


x

sns.countplot(x = y)

y.value_counts()

#Normalize

from sklearn.preprocessing import StandardScaler

scaler =StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled

#cross-validation
from sklearn.model_selection import train_test_split

x_train , x_test, y_train, y_test = train_test_split(x_scaled , y,
                 random_state=0, test_size=0.25)

x.shape
x_test.shape
x_train.shape
from sklearn.neural_network import MLPClassifier

ann = MLPClassifier(hidden_layer_sizes=(100,100,100),
                random_state=0,
                    max_iter=100,
                        activation='relu')

ann.fit(x_train , y_train)
y_pred = ann.predict(x_test)
y_pred


from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score


y_test.value_counts()

ConfusionMatrixDisplay.from_predictions(y_test , y_pred)

accuracy_score(y_test , y_pred)

print(classification_report(y_test,y_pred))