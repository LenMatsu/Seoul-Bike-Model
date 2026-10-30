import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# 1. Load Dataset
df = pd.read_csv("SeoulBikeData.csv", encoding="unicode_escape")

# 2. Basic Cleaning: Before Checks & Summary Statistics
print("--- BEFORE CLEANING CHECKS ---")
print("Initial Shape:", df.shape)
print("Total Missing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())
print("\n--- TARGET SUMMARY STATISTICS ---")
print(df['Rented Bike Count'].describe())

# 3. Exploratory Data Analysis (3 Labelled Plots)
plt.figure(figsize=(8, 5))
plt.hist(df['Rented Bike Count'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Rented Bike Count')
plt.xlabel('Rented Bike Count')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df['Temperature(°C)'], df['Rented Bike Count'], alpha=0.3, color='orange')
plt.title('Temperature vs. Rented Bike Count')
plt.xlabel('Temperature (°C)')
plt.ylabel('Rented Bike Count')
plt.grid(True)
plt.tight_layout()
plt.show()

season_avg = df.groupby('Seasons')['Rented Bike Count'].mean()
plt.figure(figsize=(8, 5))
season_avg.plot(kind='bar', color='teal')
plt.title('Average Bike Rentals per Season')
plt.xlabel('Season')
plt.ylabel('Average Rented Bike Count')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 4. Feature Engineering & Target Encoding
median_rentals = df['Rented Bike Count'].median()
df['High_Demand'] = (df['Rented Bike Count'] > median_rentals).astype(int)

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Month'] = df['Date'].dt.month.astype(str)
df['DayOfWeek'] = df['Date'].dt.dayofweek.astype(str)
df['Hour'] = df['Hour'].astype(str)

df_encoded = pd.get_dummies(df.drop(columns=['Date', 'Rented Bike Count']), drop_first=True)

print("\n--- AFTER CLEANING & ENCODING CHECKS ---")
print("Processed Shape:", df_encoded.shape)
print("Target Class Counts:\n", df_encoded['High_Demand'].value_counts())

X = df_encoded.drop(columns=['High_Demand'])
y = df_encoded['High_Demand']

# 5. Reproducible Data Splitting & Training-Only Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Baseline Model Implementation
baseline = DummyClassifier(strategy="most_frequent", random_state=42)
baseline.fit(X_train_scaled, y_train)
y_base_pred = baseline.predict(X_test_scaled)
print("\n--- BASELINE MODEL (DummyClassifier) ---")
print(f"Baseline Accuracy: {accuracy_score(y_test, y_base_pred):.4f}")

# 7. Logistic Regression Model Training & Evaluation
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("\n--- CONFUSION MATRIX (Logistic Regression) ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- CLASSIFICATION REPORT (Logistic Regression) ---")
print(classification_report(y_test, y_pred))

# 8. Labelled Evaluation Plot (Confusion Matrix Display)
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=['Low/Med Demand (0)', 'High Demand (1)'],
    cmap='Blues',
)
plt.title('Confusion Matrix: Logistic Regression Demand Prediction')
plt.tight_layout()
plt.savefig('Figure_4.png', dpi=150)
plt.show()
