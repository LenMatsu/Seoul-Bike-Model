import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("SeoulBikeData.csv", encoding="unicode_escape")

#EXPLORATORY DATA ANALYSIS (EDA)

# Target Variable Distribution
plt.figure(figsize=(8, 5))
plt.hist(df['Rented Bike Count'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Rented Bike Count')
plt.xlabel('Rented Bike Count')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()

#Temperature vs Rented Bike Count
plt.figure(figsize=(8, 5))
plt.scatter(df['Temperature(°C)'], df['Rented Bike Count'], alpha=0.3, color='orange')
plt.title('Temperature vs. Rented Bike Count')
plt.xlabel('Temperature (°C)')
plt.ylabel('Rented Bike Count')
plt.grid(True)
plt.tight_layout()
plt.show()

#Average Rentals per Season
plt.figure(figsize=(8, 5))
df.groupby('Seasons')['Rented Bike Count'].mean().plot(kind='bar', color='teal')
plt.title('Average Bike Rentals per Season')
plt.xlabel('Season')
plt.ylabel('Average Rented Bike Count')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()