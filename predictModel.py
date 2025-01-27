import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load the saved model and encoders
model = joblib.load("decision_tree_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

# Load the test dataset
test_df = pd.read_csv("test.csv")

# Preprocess the test data
# Extract features and target
X_test = test_df.drop(columns=["Attrition", "Employee ID"])  # Remove target and identifier
y_test = test_df["Attrition"]

# Encode categorical columns in the test dataset
for col, le in label_encoders.items():
    if col in X_test.columns:
        X_test[col] = le.transform(X_test[col])
y_test_encoded = label_encoders["Attrition"].transform(y_test)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Model Accuracy on test data: {accuracy * 100:.2f}%")

# Generate and display a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=label_encoders["Attrition"].classes_))
