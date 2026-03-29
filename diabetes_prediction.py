import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Column names
columns = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"
]
# Load dataset
data = pd.read_csv("diabetes.csv", names=columns)
pd.set_option('display.max_columns', None)

print("First 5 rows:")
print(data.head())


# Dataset information
print("\nDataset Info:")
print(data.info())

print("\nStatistics:")
print(data.describe())


### DATA CLEANING
# Replace impossible zero values with NaN
cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
for col in cols:
    data[col] = data[col].replace(0, np.nan)
# Fill missing values with column mean
data.fillna(data.mean(), inplace=True)

### DATA VISUALIZATION

plt.figure(figsize=(6,4))
sns.countplot(x="Outcome", data=data)
plt.title("Diabetes Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

### FEATURE PREPARATION

X = data.drop("Outcome", axis=1)
y = data["Outcome"]



### FEATURE SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

### TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
# MODEL 1: Logistic Regression

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

log_acc = accuracy_score(y_test, log_pred)

print("\nLogistic Regression Accuracy:", log_acc)

# MODEL 2: Decision Tree

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)

tree_acc = accuracy_score(y_test, tree_pred)

print("Decision Tree Accuracy:", tree_acc)

# MODEL 3: Random Forest

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_acc)

# CONFUSION MATRIX

print("\nConfusion Matrix (Random Forest):")
cm = confusion_matrix(y_test, rf_pred)
print(cm)

# CLASSIFICATION REPORT

print("\nClassification Report:")

print(classification_report(y_test, rf_pred))

# TEST MODEL WITH NEW PATIENT

new_patient = [[2,120,70,20,79,25,0.5,33]]

# Scale the input 
new_patient_scaled = scaler.transform(new_patient)

prediction = rf_model.predict(new_patient_scaled)

if prediction[0] == 1:
    print("\nPrediction: Patient is likely to have Diabetes")
else:
    print("\nPrediction: Patient is NOT likely to have Diabetes")