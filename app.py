
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
# Load the model
model = joblib.load('logistic_regression_model.pkl')
# Initialize the Flask app
app = Flask(__name__)
# Home route to serve the HTML form
@app.route('/')
def home():
    return render_template('index.html')
# Predict route to process form data and return prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from the POST request
        data = request.form
        # Extract and convert features from the form data
        features = np.array([float(data['ApplicantIncome']), 
                             float(data['LoanAmount']),
                             float(data['Loan_Amount_Term']),
                             float(data['Credit_History']),
                             float(data['Gender']),
                             float(data['Married']),
                             float(data['Dependents']),
                             float(data['Education']),
                             float(data['Self_Employed']),
                             float(data['Property_Area'])]).reshape(1, -1)
        # Make prediction
        prediction = model.predict(features)
        # Return the prediction result to the user
        result = 'Approved' if prediction[0] == 1 else 'Rejected'
        return render_template('index.html', prediction_text=f'Loan Status: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')
if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
# Load the model
model = joblib.load('logistic_regression_model.pkl')
# Initialize the Flask app
app = Flask(__name__)
# Home route to serve the HTML form
@app.route('/')
def home():
    return render_template('index.html')
# Predict route to process form data and return prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from the POST request
        data = request.form
        # Extract and convert features from the form data
        features = np.array([float(data['ApplicantIncome']), 
                             float(data['LoanAmount']),
                             float(data['Loan_Amount_Term']),
                             float(data['Credit_History']),
                             float(data['Gender']),
                             float(data['Married']),
                             float(data['Dependents']),
                             float(data['Education']),
                             float(data['Self_Employed']),
                             float(data['Property_Area'])]).reshape(1, -1)
        # Make prediction
        prediction = model.predict(features)
        # Return the prediction result to the user
        result = 'Approved' if prediction[0] == 1 else 'Rejected'
        return render_template('index.html', prediction_text=f'Loan Status: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')
if __name__ == '__main__':
    app.run(debug=True)
