# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('/Users/sallylim/Documents/Fleximasters/Module 3/Personal-loan-application/DS_Capstone_DataSet -REV.csv')

# Step 1: Preprocess the dataset
data.ffill(inplace=True)
categorical_columns = ['Gender', 'Married', 'Property_Area']
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

features = ['ApplicantIncome', 'Gender', 'Married', 'Credit_History', 'Property_Area']
X = data[features]
y = data['Loan_Status']

print("Distribution of Loan_Status:")
print(y.value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Model Evaluation
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))

# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. feature {features[indices[f]]} ({importances[indices[f]]})")

plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices])
plt.xlim([-1, X.shape[1]])
plt.show()

# Make predictions for a new applicant
new_applicant = {
    'ApplicantIncome': 12000,  # Adjusted high income
    'Gender': 'Female',
    'Married': 'Yes',
    'Credit_History': 1,
    'Property_Area': 'Urban'
}

applicant_df = pd.DataFrame([new_applicant])
applicant_df['Gender'] = label_encoders['Gender'].transform(applicant_df['Gender'])
applicant_df['Married'] = label_encoders['Married'].transform(applicant_df['Married'])
applicant_df['Property_Area'] = label_encoders['Property_Area'].transform(applicant_df['Property_Area'])

prediction = model.predict(applicant_df)
loan_status = 'Approved' if prediction[0] == 1 else 'Rejected'
print(f"Loan status for the applicant is: {loan_status}")
