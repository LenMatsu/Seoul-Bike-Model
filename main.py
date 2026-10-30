import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Dataset
df = pd.read_csv("SeoulBikeData.csv", encoding="unicode_escape")

# 2. Preprocessing, Feature Engineering & Target Definition
# Create a binary target: 1 = High Demand (> median rentals), 0 = Low/Medium Demand
median_rentals = df['Rented Bike Count'].median()
df['High_Demand'] = (df['Rented Bike Count'] > median_rentals).astype(int)

# Extract Month and Day of Week from Date before dropping it
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Month'] = df['Date'].dt.month.astype(str)
df['DayOfWeek'] = df['Date'].dt.dayofweek.astype(str)

# Treat Hour as a category so rush hours (e.g., 8 AM, 6 PM) get their own weights
df['Hour'] = df['Hour'].astype(str)

# One-Hot Encode categorical features (Hour, Month, DayOfWeek, Seasons, Holiday, Functioning Day)
df_encoded = pd.get_dummies(df.drop(columns=['Date', 'Rented Bike Count']), drop_first=True)

# Separate features (X) and target (y)
X = df_encoded.drop(columns=['High_Demand'])
y = df_encoded['High_Demand']

# 3. Data Splitting (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test_scaled)

print("--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))
