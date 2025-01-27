import pandas as pd
import re

# Function to clean special characters
def clean_special_characters(value):
    if isinstance(value, str):
        return re.sub(r"[^a-zA-Z0-9\s]", "", value)  # Keep only alphanumeric and spaces
    return value

# Read the dataset
df = pd.read_csv("test.csv")

# Apply the cleaning function
df_cleaned = df.applymap(clean_special_characters)

# Save the cleaned dataset
df_cleaned.to_csv("test_cleaned.csv", index=False)
print("Special characters cleaned. Cleaned dataset saved as 'test_cleaned.csv'.")

