import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("data/house_data.csv")

# Features (X) and Target (y)
X = data[['Area', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
new_data = pd.DataFrame([[2000, 3, 2]], columns=['Area', 'Bedrooms', 'Bathrooms'])
prediction = model.predict(new_data)
print(f"Predicted House Price: {prediction[0]}")

predictions = model.predict(X_test)

# Evaluate model
error = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", error)

# Test with new input
new_house = [[1600, 3, 2]]
predicted_price = model.predict(new_house)
print("Predicted House Price:", predicted_price[0])
