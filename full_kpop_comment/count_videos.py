import pandas as pd

# Read the CSV file
df = pd.read_csv('video.csv')

# Print the number of rows (excluding header)
print(f"Number of videos: {len(df)}")

# Print basic information about the dataset
print("\nDataset Info:")
print(df.info())

# Print first few rows to verify data
print("\nFirst few rows:")
print(df.head()) 