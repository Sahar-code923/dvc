# Step 1: Import the required libraries
import sys
import json
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Step 2: Configure MLflow tracking and experiment
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("house-price-project")

# Step 3: Choose the model type from the command line
# Default model is linear if no argument is provided
model_type = sys.argv[1].lower() if len(sys.argv) > 1 else "linear"

# Step 4: Read the house dataset from the CSV file
data = pd.read_csv("houses.csv")

print("Total houses:", len(data))

# Step 5: Separate features (inputs) and label (output)
X = data[["rooms", "size_sqft"]]
y = data["price_lakh"]

# Step 6: Split data into 80% training and 20% testing
test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42
)

print("Houses used for training:", len(X_train))
print("Houses used for testing:", len(X_test))

# Step 7: Select the model based on the chosen model type
if model_type == "linear":
    model = LinearRegression()
elif model_type == "tree":
    model = DecisionTreeRegressor(random_state=42)
elif model_type == "forest":
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
else:
    raise ValueError(
        "Invalid model type. Choose: linear, tree, or forest."
    )

# Step 8: Train, evaluate, and track the model with MLflow
with mlflow.start_run():
    # Log the experiment parameters
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("rows_in_dataset", len(data))

    # Train the selected model
    model.fit(X_train, y_train)

    # Predict prices for the test houses
    predictions = model.predict(X_test)

    # Calculate the Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, predictions)

    # Log the MAE metric in MLflow
    mlflow.log_metric("mae", mae)

    # Save the score to a file so the CI quality gate can read it
    with open("metrics.json", "w") as f:
        json.dump({"mae": mae}, f)

    # Print the test results
    print("\nModel Type:", model_type)
    print("\nTest Results:")
    print("-" * 75)
    print(
        f"{'Rooms':<8}{'Size (sqft)':<15}"
        f"{'Real Price':<15}{'Guess':<15}{'Off By':<15}"
    )
    print("-" * 75)

    for i in range(len(X_test)):
        rooms = X_test.iloc[i]["rooms"]
        size = X_test.iloc[i]["size_sqft"]
        real_price = y_test.iloc[i]
        guess = predictions[i]

        # Calculate the absolute difference between real and predicted prices
        off_by = abs(real_price - guess)

        print(
            f"{rooms:<8.0f}"
            f"{size:<15.0f}"
            f"{real_price:<15.1f}"
            f"{guess:<15.1f}"
            f"{off_by:<15.1f}"
        )

    print("-" * 75)
    print(f"Model Type: {model_type}")
    print(f"MAE (average prediction error): {mae:.2f} lakh")

    # Step 9: Log and register the trained model in MLflow
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="house-price-model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    # Step 10: Save the trained model using joblib
    joblib.dump(model, "model.pkl")
    print("\nModel saved successfully as model.pkl")