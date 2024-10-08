import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv('DS_Capstone_DataSet -REV.csv')

# Drop the Loan_ID column since it is not needed for prediction
data = data.drop('Loan_ID', axis=1)

# Print unique values in key categorical columns to check for potential issues
print("Loan_Status unique values:", data['Loan_Status'].unique())
print("Gender unique values:", data['Gender'].unique())
print("Married unique values:", data['Married'].unique())

# Clean categorical variables if needed (for 'Gender' if there are issues)
data['Gender'] = data['Gender'].apply(lambda x: 'Male' if 'Male' in x else 'Female')

# Map the cleaned categorical variables into numeric
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Married'] = data['Married'].map({'No': 0, 'Yes': 1})
data['Education'] = data['Education'].map({'Not Graduate': 0, 'Graduate': 1})
data['Self_Employed'] = data['Self_Employed'].map({'No': 0, 'Yes': 1})
data['Property_Area'] = data['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban': 2})
data['Dependents'] = data['Dependents'].replace('3+', 3).astype(int)

# Ensure all values in Loan_Status are mapped to numeric
data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})

# Print again to verify conversion
print("After mapping, Loan_Status unique values:", data['Loan_Status'].unique())

# Handle missing values (simple imputation)
data = data.fillna(data.mean())

# Convert all data columns to numeric just to ensure no non-numeric values remain
data = data.apply(pd.to_numeric, errors='coerce')

# Check if any column contains NaNs after conversion to numeric
if data.isnull().values.any():
    print("NaN values found, replacing them.")
    data = data.fillna(data.mean())  # Fill NaNs if any exist

# Selecting relevant features
selected_columns = ['ApplicantIncome', 'LoanAmount', 
                    'Loan_Amount_Term', 'Credit_History', 
                    'Gender', 'Married', 'Dependents', 
                    'Education', 'Self_Employed', 'Property_Area', 
                    'Loan_Status']

data = data[selected_columns]

# Splitting the data into input features and target
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save the model to disk
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
