from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

# Initialize the Flask application
app = Flask(__name__)

# Define the paths to the model and preprocessors
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "naive_bayes_label_encoder.pkl")

# Load the trained model and label encoder
try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
except FileNotFoundError:
    print("Error: Model or Label Encoder file not found. Please run model.py first.")
    exit()

@app.route('/')
def home():
    """Renders the home page with the loan application form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the form.
    
    The user inputs are collected from the web form, converted into a pandas DataFrame,
    and then passed to the trained model for prediction. The result is then encoded
    back to a human-readable format.
    """
    # Get form data from the request
    form_data = request.form.to_dict()

    # Convert numeric values to the correct type
    try:
        form_data['ApplicantIncome'] = int(form_data['ApplicantIncome'])
        form_data['CoapplicantIncome'] = int(form_data['CoapplicantIncome'])
        form_data['LoanAmount'] = int(form_data['LoanAmount'])
        form_data['Loan_Amount_Term'] = int(form_data['Loan_Amount_Term'])
        form_data['Credit_History'] = int(form_data['Credit_History'])
    except ValueError as e:
        return render_template('result.html', prediction=f"Error: Invalid input for numeric fields. {e}")

    # Convert the form data into a pandas DataFrame
    # Note: The order of columns must match the training data
    features = pd.DataFrame({
        'Gender': [form_data['Gender']],
        'Married': [form_data['Married']],
        'Dependents': [form_data['Dependents']],
        'Education': [form_data['Education']],
        'Self_Employed': [form_data['Self_Employed']],
        'ApplicantIncome': [form_data['ApplicantIncome']],
        'CoapplicantIncome': [form_data['CoapplicantIncome']],
        'LoanAmount': [form_data['LoanAmount']],
        'Loan_Amount_Term': [form_data['Loan_Amount_Term']],
        'Credit_History': [form_data['Credit_History']],
        'Property_Area': [form_data['Property_Area']]
    })

    # Make a prediction using the loaded model
    prediction_encoded = model.predict(features)
    
    # Get the original label from the encoded prediction
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

    # Return the prediction result
    return render_template('result.html', prediction=prediction_label)

if __name__ == '__main__':
    # You can run the app with `python app.py` and access it at http://127.0.0.1:5000
    app.run(debug=True)
