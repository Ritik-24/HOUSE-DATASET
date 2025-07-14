
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Create a sample dataset
data = {
    'SquareFootage': [1000, 1500, 1800, 2400, 3000, 3500, 4000, 4200, 5000, 6000],
    'Bedrooms':       [2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
    'Bathrooms':      [1, 2, 2, 2, 3, 3, 4, 4, 5, 5],
    'Price':          [150000, 200000, 230000, 280000, 350000, 400000, 450000, 470000, 550000, 650000]
}
df = pd.DataFrame(data)

# Step 2: Define features and target
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict on test data
y_pred = model.predict(X_test)

# Step 6: Print model evaluation
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Step 7: Visualize actual vs predicted prices
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='green', label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Ideal Prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("House Price Prediction: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()
