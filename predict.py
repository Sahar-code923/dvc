# Step 1: Import the required libraries
import pandas as pd
import joblib

# Step 2: Load the trained model from model.pkl
model = joblib.load("model.pkl")

# Step 3: Create a DataFrame for the new house
new_house = pd.DataFrame({
    "rooms": [3],
    "size_sqft": [1350]
})

# Step 4: Predict the house price
predicted_price = model.predict(new_house)[0]

# Step 5: Print the predicted price
print(
    f"3 rooms, 1350 sqft -> predicted price: "
    f"{predicted_price:.1f} lakh"
)