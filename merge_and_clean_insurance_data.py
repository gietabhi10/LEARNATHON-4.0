# filename: merge_and_clean_insurance_data.py

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Step 1: Load and merge CSV files
filenames = [
    "Auto_Insurance_Fraud_Claims_File01.csv",
    "Auto_Insurance_Fraud_Claims_File02.csv",
    "Auto_Insurance_Fraud_Claims_File03.csv"
]

try:
    merged_df = pd.concat([pd.read_csv(file) for file in filenames], ignore_index=True)
    print("✅ Merged shape:", merged_df.shape)
except FileNotFoundError as e:
    print(f"❌ File not found: {e.filename}")
    exit(1)

# Step 2: Drop rows where target 'Fraud_Ind' is missing
merged_df = merged_df[merged_df['Fraud_Ind'].notna()].copy()

# Step 3: Drop columns with more than 30% missing values
missing_pct = merged_df.isnull().mean()
high_missing_cols = missing_pct[missing_pct > 0.3].index
merged_df.drop(columns=high_missing_cols, inplace=True)

# Step 4: Separate numerical and categorical columns
num_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = merged_df.select_dtypes(include='object').columns.tolist()

# Step 5: Impute missing values
# Numeric columns → mean
merged_df[num_cols] = SimpleImputer(strategy='mean').fit_transform(merged_df[num_cols])

# Categorical columns → most frequent
merged_df[cat_cols] = merged_df[cat_cols].astype(str)
merged_df[cat_cols] = pd.DataFrame(
    SimpleImputer(strategy='most_frequent').fit_transform(merged_df[cat_cols]),
    columns=cat_cols,
    index=merged_df.index
)

# Step 6: Label Encode categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    merged_df[col] = le.fit_transform(merged_df[col])
    label_encoders[col] = le  # optional: store for inverse_transform later

# ✅ Final check
print("✅ Cleaned shape:", merged_df.shape)
print(merged_df.head())
