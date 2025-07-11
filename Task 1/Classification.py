import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('C:/Users/USER/.vscode/python/Intern/iris.csv')  

print("First 5 rows of the dataset:")
print(df.head())

print("\nClass distribution:")
print(df['Species'].value_counts())

sns.pairplot(df, hue='Species')
plt.title("Iris Dataset Pairplot")
plt.show()

X = df.drop(['Id', 'Species'], axis=1)
y = df.iloc[:, -1]   

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

sample = np.array([[5.1, 3.5, 1.4, 0.2]]) 
prediction = model.predict(sample)
print("\nPredicted species for sample:", le.inverse_transform(prediction))
