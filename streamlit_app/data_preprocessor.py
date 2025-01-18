import pandas as pd

# File path to the dataset (replace with your actual file path)
file_path = "data/acris_real_property_legals.csv"

# Number of rows to read
num_rows = 100000

# Load the first 100,000 rows into a DataFrame
df = pd.read_csv(file_path, nrows=num_rows)

# Drop rows with critical missing values
df.dropna(subset=['DOCUMENT ID', 'GOOD THROUGH DATE', 'PROPERTY TYPE'], inplace=True)

# Normalize text columns
text_columns = ['DOCUMENT ID', 'PROPERTY TYPE', 'STREET NUMBER', 'STREET NAME', 'UNIT']
for col in text_columns:
    df[col] = df[col].str.strip().str.upper()

# Ensure BOROUGH is numeric before mapping
df['BOROUGH'] = pd.to_numeric(df['BOROUGH'], errors='coerce')

# Map BOROUGH codes to descriptive names
borough_mapping = {
    1: "Manhattan",
    2: "Bronx",
    3: "Brooklyn",
    4: "Queens",
    5: "Staten Island"
}
df['BOROUGH'] = df['BOROUGH'].map(borough_mapping)

# Convert binary columns to boolean
binary_columns = ['EASEMENT', 'AIR RIGHTS', 'SUBTERRANEAN RIGHTS']
for col in binary_columns:
    df[col] = df[col].map({'Y': True, 'N': False})

# Map PARTIAL LOT codes to descriptive names
partial_lot_mapping = {'E': 'Entire', 'N': 'None', 'P': 'Partial'}
df['PARTIAL LOT'] = df['PARTIAL LOT'].map(partial_lot_mapping)

# Convert GOOD THROUGH DATE to datetime
df['GOOD THROUGH DATE'] = pd.to_datetime(df['GOOD THROUGH DATE'], errors='coerce')

# Ensure BLOCK and LOT are integers
df['BLOCK'] = pd.to_numeric(df['BLOCK'], errors='coerce', downcast='integer')
df['LOT'] = pd.to_numeric(df['LOT'], errors='coerce', downcast='integer')

# Save the processed dataset
processed_file_path = "data/processed_acris_data.csv"
df.to_csv(processed_file_path, index=False)

print(f"Processed dataset saved to {processed_file_path}")
