# Naive Bayes Loan Approval Predictor

## Project Description

This project demonstrates a machine learning pipeline using the **Gaussian Naive Bayes** algorithm to predict loan approval status. It's built as a Flask web application, allowing users to input various applicant details and receive an instant prediction. The project includes scripts for data generation, model training, and a web interface, all organized in a clear, modular structure.

---

## Project Structure

The repository is organized to separate different components of the project, from data and models to the web application itself.

```
DataScience/
└── NaiveBayes/                       # Project root
    ├── data/                         # All datasets go here
    │   └── loan_approval_dataset.csv
    │
    ├── model/                        # Trained ML models (PKL files)
    │   ├── naive_bayes_model.pkl
    │   └── naive_bayes_label_encoder.pkl
    │
    ├── static/                       # CSS, JS, Images for the frontend
    │   └── style.css
    │
    ├── templates/                    # HTML templates for Flask
    │   ├── index.html
    │   └── result.html
    │
    ├── model.py                      # Script to train & save the Naive Bayes model
    ├── app.py                        # Flask backend app
    └── requirements.txt              # List of dependencies
```

---

## Setup and Installation

1. **Clone the repository (optional):**

If you are using Git, you can clone the project to your local machine.

```bash
git clone <repository-url>
cd DataScience/NaiveBayes
```

2. **Create a virtual environment (recommended):**

It's recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## Why Naive Bayes?

The **Naive Bayes** algorithm is a simple yet powerful probabilistic classifier based on **Bayes' Theorem**. It operates under the "naive" assumption that all features are independent of each other. This makes it computationally efficient, fast to train, and effective for large datasets. It's particularly well-suited for classification problems where you need a quick, baseline model that performs well without a lot of tuning.

---

## Dataset

The dataset used is a synthetic **Loan Approval Dataset** with the following features, which are used to predict the `Loan_Status`:

- **Gender** (categorical)
- **Married** (categorical)
- **Dependents** (categorical)
- **Education** (categorical)
- **Self_Employed** (categorical)
- **ApplicantIncome** (numeric)
- **CoapplicantIncome** (numeric)
- **LoanAmount** (numeric)
- **Loan_Amount_Term** (numeric)
- **Credit_History** (numeric: 0 or 1)
- **Property_Area** (categorical)
- **Loan_Status** (Target: `Y` for Approved, `N` for Denied)

---

## How to Run

1. **Train the Model:**

First, you need to run the `model.py` script. This script will create the `loan_approval_dataset.csv`, train the Naive Bayes model, and save the necessary `.pkl` files in the `model/` directory.

```bash
python model.py
```

2. **Start the Flask app:**

After the model files are created, you can start the Flask application.

```bash
python app.py
```

The application will now be running on `http://127.0.0.1:5000`. Open this URL in your web browser to access the prediction form.

---

## Prediction Goal

The application predicts the loan approval status, which can be either **Approved** or **Denied**.

---

## Tech Stack

* **Python** – Core programming language
* **Pandas & NumPy** – Data manipulation
* **Scikit-learn** – Machine learning model training
* **Flask** – Web framework for deployment
* **HTML/CSS** – Frontend UI design

---

## Future Scope

* **Model Comparison:** Implement and compare the Naive Bayes model with other classification algorithms (e.g., Logistic Regression, Decision Trees) to see which performs best on the dataset.
* **Feature Engineering:** Explore more complex features, such as ratios of income or different ways to handle categorical data, to improve the model's predictive accuracy.
* **Model Deployment:** Deploy the Flask application to a cloud platform like Heroku or Render for public access.
* **User Interface:** Enhance the web application with a more interactive UI, providing visualizations or a confidence score for each prediction.
