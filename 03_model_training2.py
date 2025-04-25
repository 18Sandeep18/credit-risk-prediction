# 02_model_training_logreg.ipynb

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load preprocessed data
df = pd.read_csv("processed_data.csv")

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
# Split features and target
X = df.drop("Credit_Risk", axis=1)
y = df["Credit_Risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

# Predictions
y_pred = logreg.predict(X_test)
# Combine the actual and predicted values along with the features
bad_predictions_df = X_test.copy()  # Copy the features of the test set
bad_predictions_df['Actual'] = y_test  # Add the actual values
bad_predictions_df['Predicted'] = y_pred  # Add the predicted values

# Filter the rows where the prediction is 'bad' (assuming 'bad' = 1)
bad_predictions = bad_predictions_df[bad_predictions_df['Predicted'] == 1]

# Display the examples where the model predicted 'bad'
print(bad_predictions)

# Evaluation
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix_logreg.png")
plt.show()

# Feature Importance (Coefficients)
coefficients = pd.Series(logreg.coef_[0], index=X.columns)
coefficients.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features - Logistic Regression")
plt.savefig("feature_importance_logreg.png")
plt.show()

# Save model
joblib.dump(logreg, "logreg_model.pkl")

