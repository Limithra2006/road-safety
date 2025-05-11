import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv(r"D:\\Codes\\Projects\\ML\\Limi\\data\\RTA Dataset.csv")

# Handle missing values
df.ffill(inplace=True)

# Convert Time column to datetime and extract hour
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce')
df['Hour'] = df['Time'].dt.hour.fillna(12)

# Encode target variable
severity_order = ['Slight Injury', 'Serious Injury', 'Fatal injury']
df['Accident_severity'] = pd.Categorical(df['Accident_severity'], categories=severity_order, ordered=True)
le = LabelEncoder()
df['SeverityEncoded'] = le.fit_transform(df['Accident_severity'])

# Prepare features and labels
X = df.drop(['Accident_severity', 'SeverityEncoded', 'Time'], axis=1)
X = pd.get_dummies(X, drop_first=True)
y = df['SeverityEncoded']
