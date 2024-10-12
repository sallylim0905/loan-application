import joblib
import numpy as np

# Load the saved logistic regression model
model = joblib.load('logistic_regression_model.pkl')

# Define a test case for a loan application
# This example corresponds to a valid feature set from the trained model
test_input = {
    "ApplicantIncome": 1000,         # Numeric
    "LoanAmount": 500,               # Numeric
    "Loan_Amount_Term": 360,         # Numeric
    "Credit_History": 1,             # Binary (1 for good credit history)
    "Gender": 0,                     # 0: Male, 1: Female
    "Married": 1,                    # 0: No, 1: Yes
    "Dependents": 2,                 # Numeric (0, 1, 2, or 3+)
    "Education": 1,                  # 0: Not Graduate, 1: Graduate
    "Self_Employed": 0,              # 0: No, 1: Yes
    "Property_Area": 2               # 0: Rural, 1: Semiurban, 2: Urban
}

# Convert the dictionary values to a NumPy array
features = np.array([test_input['ApplicantIncome'], 
                     test_input['LoanAmount'], 
                     test_input['Loan_Amount_Term'], 
                     test_input['Credit_History'], 
                     test_input['Gender'], 
                     test_input['Married'], 
                     test_input['Dependents'], 
                     test_input['Education'], 
                     test_input['Self_Employed'], 
                     test_input['Property_Area']]).reshape(1, -1)

# Make a prediction using the logistic regression model
prediction = model.predict(features)

# Print the prediction result
if prediction[0] == 1:
    print("Test case result: Loan Approved")
else:
    print("Test case result: Loan Rejected")
