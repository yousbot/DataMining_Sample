import pandas as pd
import numpy as np
import re

# Load the dataset
df = pd.read_csv("test.csv")

# Detect rows with missing values
missing_values = df[df.isnull().any(axis=1)]

# Detect rows with special characters in string columns
def has_special_characters(value):
    if isinstance(value, str):
        return bool(re.search(r"[^a-zA-Z0-9\s]", value))
    return False

special_char_rows = df.applymap(has_special_characters).any(axis=1)
rows_with_special_chars = df[special_char_rows]

# Detect duplicate rows
duplicates = df[df.duplicated()]

# Output summary
print(f"\nSummary:")
print(f"Total rows with missing values: {len(missing_values)}")
print(f"Total rows with special characters: {len(rows_with_special_chars)}")
print(f"Total duplicate rows: {len(duplicates)}")

