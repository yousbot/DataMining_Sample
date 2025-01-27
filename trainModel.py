import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv("train.csv")

# Preprocessing the data
# Encode categorical columns
categorical_columns = [
    "Gender", "Job Role", "Work-Life Balance", "Job Satisfaction",
    "Performance Rating", "Overtime", "Education Level", "Marital Status",
    "Job Level", "Company Size", "Remote Work", "Leadership Opportunities",
    "Innovation Opportunities", "Company Reputation", "Employee Recognition", "Attrition"
]

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for prediction use

# Define features and target
X = df.drop(columns=["Attrition", "Employee ID"])  # Remove target and identifier
y = df["Attrition"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and label encoders
joblib.dump(model, "decision_tree_model.joblib")
joblib.dump(label_encoders, "label_encoders.joblib")
print("Model and encoders saved successfully!")

