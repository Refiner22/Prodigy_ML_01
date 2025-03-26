# Prodigy_ML_01
 
  # House Price Prediction Using Machine Learning


# 1️⃣ Situation (S) – Problem Context
What is the problem?
Real estate prices are influenced by multiple factors such as:

Square footage

Number of bedrooms

Number of bathrooms

Location, age of the house, amenities, etc.

Estimating house prices manually can be difficult and prone to error. Hence, we aim to develop a Machine Learning model that predicts house prices based on given features.

## Why is this important?
🏡 For buyers: Helps them understand if a house is priced fairly.

💰 For sellers & agents: Aids in competitive pricing.

📊 For investors: Provides insights into market trends.

# Business Goal
To create a Linear Regression model that can predict house prices with high accuracy and reliability.

# 2️⃣ Task (T) – Defining the Objective
Goal of the Project
Develop a supervised machine learning model that predicts house prices based on historical data.

Key Requirements:
✔ Load, clean, and preprocess data.
✔ Train a Linear Regression model.
✔ Evaluate the model using appropriate metrics.
✔ Provide insights based on results.

Dataset Details
📂 File Used: Merged file.csv
📝 Features: Square footage, number of bedrooms, number of bathrooms, price (target variable).

# 3️⃣ Action (A) – Step-by-Step Implementation
🔹 Step 1: Import Necessary Libraries
First, we load essential Python libraries for data manipulation, visualization, and machine learning.

python
Copy
Edit
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  
📌 Why?

pandas & numpy: Data handling.

matplotlib & seaborn: Data visualization.

sklearn: Model training and evaluation.

## 🔹 Step 2: Load and Explore Data
We read the dataset and check its structure.

python
Copy
Edit
df = pd.read_csv(r"C:\\Users\\kingzuzu\\Desktop\\Merged file.csv")  
print(df.head())  
print(df.info())  
print(df.describe())  
📌 Why?

.head(): Shows the first few rows.

.info(): Checks for missing values and data types.

.describe(): Displays summary statistics.

## 🔹 Step 3: Data Cleaning & Preprocessing
Before training the model, we ensure data quality.

python
Copy
Edit
# Check for missing values
print(df.isnull().sum())

# Remove duplicates if any
df.drop_duplicates(inplace=True)
📌 Why?

Ensures clean and accurate data for model training.

## 🔹 Step 4: Define Features and Target Variable
We define X (independent variables) and y (dependent variable – price).

python
Copy
Edit
X = df[['square_footage', 'bedrooms', 'bathrooms']]  
y = df['price']
📌 Why?

Features are selected based on relevance.

Target variable (y) is what we want to predict.

## 🔹 Step 5: Split Data into Training and Testing Sets
We divide the dataset into 80% training and 20% testing to evaluate model performance.

python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
📌 Why?

The model learns from 80% of the data and is tested on 20% unseen data.

## 🔹 Step 6: Train the Model
We use Linear Regression, which is suitable for predicting continuous values.

python
Copy
Edit
model = LinearRegression()  
model.fit(X_train, y_train)
📌 Why?

### Linear Regression finds the best-fit line between features and target variable.

## 🔹 Step 7: Make Predictions
Now, we use the trained model to predict house prices on the test data.

python
Copy
Edit
y_pred = model.predict(X_test)
📌 Why?

Predictions help assess how well the model generalizes to new data.

## 🔹 Step 8: Model Evaluation
We evaluate the model using three key metrics:

python
Copy
Edit
mae = mean_absolute_error(y_test, y_pred)  
mse = mean_squared_error(y_test, y_pred)  
r2 = r2_score(y_test, y_pred)  

print(f"Mean Absolute Error (MAE): {mae}")  
print(f"Mean Squared Error (MSE): {mse}")  
print(f"R² Score: {r2}")
📌 Why?

MAE: Measures average absolute errors in predictions.

MSE: Penalizes larger errors more than smaller ones.

R² Score: Measures how well our model explains the variance in house prices.

📊 Ideal R² Value:

0.7+ (Good Model) – The model explains 70%+ of the price variance.

0.5-0.7 (Moderate Model) – Acceptable but can be improved.

<0.5 (Poor Model) – Needs more features or better tuning.

4️⃣ Result (R) – Model Performance & Insights
✅ Final Outcomes:

Successfully trained a Linear Regression model for house price prediction.

Evaluated model performance using MAE, MSE, and R² Score.

Provided a baseline model for future improvements.

## 📉 Model Limitations & Possible Enhancements
🚀 To improve accuracy, we can:
✔ Add more features (location, age of the house, crime rate).
✔ Try non-linear models (Decision Trees, Random Forest).
✔ Perform Feature Engineering (creating new variables).
✔ Optimize hyperparameters using GridSearchCV.

## 📊 Visualization Suggestion
Plot Predicted vs. Actual Prices to see how well the model performs:

python
Copy
Edit
plt.figure(figsize=(8,6))  
plt.scatter(y_test, y_pred, color="blue", alpha=0.5)  
plt.xlabel("Actual Prices")  
plt.ylabel("Predicted Prices")  
plt.title("Actual vs Predicted House Prices")  
plt.show()
🔚 Conclusion
🎯 Key Takeaways:
✔ Built a Linear Regression model for house price prediction.
✔ Evaluated performance using MAE, MSE, R² Score.
✔ Identified areas for improvement to enhance model accuracy.
