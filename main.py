import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("SeoulBikeData.csv", encoding="unicode_escape")

median_rentals = df['Rented Bike Count'].median()
df['High_Demand'] = (df['Rented Bike Count'] > median_rentals).astype(int)

df_encoded = pd.get_dummies(df.drop(columns=['Date', 'Rented Bike Count']), drop_first=True)

X = df_encoded.drop(columns=['High_Demand'])
y = df_encoded['High_Demand']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


print("--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))