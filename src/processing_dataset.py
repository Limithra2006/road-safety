import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv(r"D:\\Codes\\Projects\\ML\\Limi\\data\\RTA Dataset.csv")
print("Dataset loaded successfully!")

# Handle missing values - forward fill for numerical and categorical columns
df.ffill(inplace=True)

# Convert 'Time' column to datetime and extract hour
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce')
df['Hour'] = df['Time'].dt.hour.fillna(12)  # Default to 12 if time is missing

# Encode 'Accident_severity' column
severity_order = ['Slight Injury', 'Serious Injury', 'Fatal injury']
df['Accident_severity'] = pd.Categorical(df['Accident_severity'], categories=severity_order, ordered=True)

le = LabelEncoder()
df['SeverityEncoded'] = le.fit_transform(df['Accident_severity'])

# Prepare features (X) and target labels (y)
X = df.drop(['Accident_severity', 'SeverityEncoded', 'Time'], axis=1)
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding
y = df['SeverityEncoded']

# Save cleaned data to a new CSV file
cleaned_data = pd.concat([X, y], axis=1)
cleaned_data.to_csv(r"D:\\Codes\\Projects\\ML\\Limi\\data\\cleaned_RTA_dataset.csv", index=False)

print("\nCleaned data saved successfully!")
