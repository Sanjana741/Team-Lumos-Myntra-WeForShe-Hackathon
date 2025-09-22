import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

# Load dataset
df = pd.read_csv("heart.csv")
print("Dataset shape:", df.shape)

# Display dataset information
print(df.head())
df.info()

# Target distribution
sns.countplot(x="target", data=df)
plt.title("Distribution of Target (0 = No Disease, 1 = Disease)")
plt.show()

# Feature correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Split dataset
X = df.drop("target", axis=1)   # Independent variables
y = df["target"]                # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Performance metrics
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

# Confusion matrix values
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Results summary table
results = pd.DataFrame({
    "Metric": [
        "Accuracy", "Precision", "Recall", "F1-Score",
        "True Positives", "True Negatives", "False Positives", "False Negatives"
    ],
    "Value": [accuracy, precision, recall, f1, tp, tn, fp, fn]
})

print(results)

# Training and Testing Scores
print("Training Score:", model.score(X_train, y_train))
print("Testing Score:", model.score(X_test, y_test))

# Coefficients of features
coeff = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print(coeff)
