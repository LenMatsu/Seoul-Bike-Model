# Seoul Bike Sharing Demand Classification Model

## Overview
This repository contains a machine learning pipeline for predicting bike rental demand levels using the Seoul Bike Sharing dataset (8,760 records). The project includes Exploratory Data Analysis (EDA), feature preprocessing, and a Logistic Regression classification model.

## Project Structure
* `SeoulBikeData.csv`: Primary dataset containing hourly weather and bike rental data.
* `main.py`: Complete Python script covering data preprocessing, model training, and evaluation.
* `Figure_1.png`: Distribution histogram of bike rental counts.
* `Figure_2.png`: Scatter plot showing Temperature vs. Bike Rentals.
* `Figure_3.png`: Bar chart displaying average bike rentals across seasons.

## Results
* **Model Used:** Logistic Regression
* **Data Split:** 80% Training / 20% Testing (`random_state=42`)
* **Accuracy:** 87%
* **Precision / Recall / F1-Score:** ~0.87
