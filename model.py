import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Define paths based on the requested structure
DATA_DIR = "dataset"
MODEL_DIR = "model"
DATA_PATH = os.path.join(DATA_DIR, "loan_approval_dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "naive_bayes_label_encoder.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def create_dataset():
    """
    Creates a sample dataset for loan approval and saves it as a CSV file.
    This function ensures the dataset exists for the training script to use.
    """
    data = {
        'Gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Male', 'Male', 'Female', 'Male'] * 5,
        'Married': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No'] * 5,
        'Dependents': ['0', '1', '0', '2', '0', '0', '0', '3+', '0', '0'] * 5,
        'Education': ['Graduate', 'Graduate', 'Graduate', 'Not Graduate', 'Graduate', 'Not Graduate', 'Not Graduate', 'Graduate', 'Graduate', 'Not Graduate'] * 5,
        'Self_Employed': ['No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No'] * 5,
        'ApplicantIncome': [5849, 4583, 3000, 2583, 6000, 5417, 2333, 3036, 4000, 4833] * 5,
        'CoapplicantIncome': [0, 1508, 0, 2358, 0, 4196, 1516, 2504, 2275, 2333] * 5,
        'LoanAmount': [120, 128, 66, 120, 141, 267, 95, 120, 120, 120] * 5,
        'Loan_Amount_Term': [360, 360, 360, 360, 360, 360, 360, 360, 360, 360] * 5,
        'Credit_History': [1, 1, 1, 1, 1, 1, 1, 0, 1, 1] * 5,
        'Property_Area': ['Urban', 'Rural', 'Urban', 'Urban', 'Urban', 'Urban', 'Semiurban', 'Semiurban', 'Urban', 'Semiurban'] * 5,
        'Loan_Status': ['Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'N'] * 5
    }
    df = pd.DataFrame(data)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset saved to {DATA_PATH}")

def train_and_save_model():
    """
    Loads the dataset, preprocesses it, trains a Gaussian Naive Bayes model,
    and saves the trained pipeline and label encoder.
    """
    # Load the dataset
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Dataset not found at {DATA_PATH}. Creating a new one...")
        create_dataset()
        df = pd.read_csv(DATA_PATH)

    # Separate features (X) and target (y)
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Create preprocessing pipelines for numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Create the full pipeline with the preprocessor and the classifier
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())
    ])

    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train the model
    print("Training the Naive Bayes model...")
    model_pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate the model
    accuracy = model_pipeline.score(X_test, y_test)
    print(f"Model accuracy on the test set: {accuracy:.2f}")

    # Save the trained pipeline and the label encoder
    print("Saving the trained model and label encoder...")
    joblib.dump(model_pipeline, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    print("Model and label encoder saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
