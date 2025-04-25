import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv('modified_german_credit_dataset.csv')

# Drop unnecessary index column if present
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Show basic info
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

# Class Distribution
bad_count = df[df['Credit_Risk'] == 1].shape[0]
good_count = df[df['Credit_Risk'] == 0].shape[0]
print(f"This class is imbalanced because of:\n"
      f"- Bad credit entries: {bad_count}\n"
      f"- Good credit entries: {good_count}")

sns.countplot(x='Credit_Risk', data=df)
plt.title("Original Credit Risk Distribution")
plt.show()

# Encode Categorical Columns
cat_cols = df.select_dtypes(include='object').columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Save encoders
joblib.dump(le_dict, "label_encoders.pkl")

# Apply SMOTE to balance the data
X = df.drop('Credit_Risk', axis=1)
y = df['Credit_Risk']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine the resampled data
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['Credit_Risk'] = y_resampled

# Show new class distribution
sns.countplot(x='Credit_Risk', data=df_resampled)
plt.title("Balanced Credit Risk Distribution (After SMOTE)")
plt.show()

# Correlation heatmap
sns.heatmap(df_resampled.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Feature Correlation Heatmap (After SMOTE)")
plt.show()

# Save preprocessed and balanced data
df_resampled.to_csv("processed_balanced_data.csv", index=False)
