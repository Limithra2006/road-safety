# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv(r"D:\\Codes\\Projects\\ML\\Limi\data\\cleaned_RTA_dataset.csv")
X = df.drop(['SeverityEncoded'], axis=1)
y = df['SeverityEncoded']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "random_forest_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")
