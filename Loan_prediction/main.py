import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\ELCOT\Desktop\Loan_prediction\loan_approval_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

print(df.head())
print("Shape:", df.shape)
print("Missing Values:\n", df.isnull().sum())
print("Duplicate Values:", df.duplicated().sum())

# Encode categorical columns
df["education"] = df["education"].str.strip()
df["self_employed"] = df["self_employed"].str.strip()
df["loan_status"] = df["loan_status"].str.strip()

df["education"] = df["education"].map({
    "Graduate": 1,
    "Not Graduate": 0
})

df["self_employed"] = df["self_employed"].map({
    "Yes": 1,
    "No": 0
})

df["loan_status"] = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})

# Features
features = [
    "self_employed",
    "education",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "income_annum"
]

X = df[features]
y = df["loan_status"]

# Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "loan_approval_model.pkl")
print("Model saved successfully!")

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

top5 = importance.sort_values(
    by="Importance",
    ascending=False
).head(5)

print(top5)

plt.figure(figsize=(8,5))
plt.bar(top5["Feature"], top5["Importance"])
plt.title("Top 5 Important Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()