import pandas as pd
import os

# Set the folder path for data files
data_folder = os.path.join(os.getcwd(), "Card Patterns")

# Load the datasets
customer_data_path = os.path.join(data_folder, "Customer_Data.xlsx")
transaction_data_path = os.path.join(data_folder, "Transaction_Data.xlsx")
profitability_metrics_path = os.path.join(data_folder, "Profitability_Metrics.xlsx")

# Read Excel files
customer_data = pd.read_excel(customer_data_path)
transaction_data = pd.read_excel(transaction_data_path)
profitability_metrics = pd.read_excel(profitability_metrics_path)

# Step 1: Merge Data
# Merge transaction data with customer data on CustomerID
merged_data = pd.merge(transaction_data, customer_data, on="CustomerID", how="left")

# Merge profitability metrics with the merged dataset on TransactionID
final_data = pd.merge(merged_data, profitability_metrics, on="TransactionID", how="left")

# Step 2: Data Cleaning
# Drop any rows with missing values (if applicable)
final_data_cleaned = final_data.dropna()

# Convert date columns to datetime format
final_data_cleaned['TransactionDate'] = pd.to_datetime(final_data_cleaned['TransactionDate'])
final_data_cleaned['AccountCreated'] = pd.to_datetime(final_data_cleaned['AccountCreated'])

# Create new columns
# Calculate transaction age in days
final_data_cleaned['TransactionAgeDays'] = (final_data_cleaned['TransactionDate'] - final_data_cleaned['AccountCreated']).dt.days

# Step 3: Feature Engineering
# Aggregate metrics at customer level
customer_summary = final_data_cleaned.groupby('CustomerID').agg({
    'TransactionAmount': ['sum', 'mean', 'count'],  # Total, average, and count of transactions
    'Profit': 'sum',                               # Total profit
    'TransactionAgeDays': 'mean'                   # Average transaction age
}).reset_index()

# Rename columns for clarity
customer_summary.columns = [
    'CustomerID', 
    'TotalTransactionAmount', 
    'AvgTransactionAmount', 
    'TransactionCount', 
    'TotalProfit', 
    'AvgTransactionAgeDays'
]

# Step 4: Save Cleaned Data
# Save cleaned and aggregated data to new Excel files
cleaned_data_path = os.path.join(data_folder, "Cleaned_Data.xlsx")
customer_summary_path = os.path.join(data_folder, "Customer_Summary.xlsx")

final_data_cleaned.to_excel(cleaned_data_path, index=False)
customer_summary.to_excel(customer_summary_path, index=False)

print("ETL operations completed successfully!")
print(f"Cleaned data saved at: {cleaned_data_path}")
print(f"Customer summary saved at: {customer_summary_path}")
