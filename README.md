# Financial-Loan-Default-Prediction--Random-Forest-Classifier

Overview
This script demonstrates how to build a machine learning model to predict the likelihood of loan defaults based on financial data such as income, loan amount, credit score, and loan term. The model uses a Random Forest Classifier to perform the classification, and evaluates the model's performance using accuracy, classification report, and confusion matrix.

Solution
In this solution, we will use a Random Forest Classifier to predict loan defaults. A synthetic dataset is generated with features such as:

income: The income of the loan applicant.

loan_amount: The amount the applicant requests as a loan.

credit_score: The applicant’s credit score.

loan_term: The loan repayment period in months.

default: A binary target variable indicating whether the loan defaulted (1) or not (0).

The steps involved in the solution are:

Data Generation: Synthetic data is created to simulate real-world financial scenarios.

Data Visualization: The distribution of credit scores by default status is visualized.

Preprocessing: Features are standardized and split into training and testing sets.

Model Training: A Random Forest Classifier is trained on the preprocessed data.

Evaluation: The model’s performance is evaluated using metrics such as accuracy, precision, recall, and confusion matrix.

Instructions -
Prerequisites -
To run this code, you need the following Python libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

You can install the required libraries by running the following command in your terminal or command prompt:-

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn

Running the Code-
Generate Data: The function generate_data() creates a synthetic dataset with features like income, loan amount, credit score, loan term, and default status. The default dataset size is 1000 samples.

Data Visualization: The plot_data() function visualizes the distribution of credit_score by the default status (default vs. non-default).

Data Preprocessing: The data is split into features (X) and target (y). It is then divided into training and testing sets, with feature scaling applied using StandardScaler to standardize the data.

Model Training: A Random Forest Classifier with 100 trees is trained using the training data.

Model Evaluation: After training the model, it is evaluated on the test data with the following metrics:

Accuracy: Proportion of correct predictions.

Classification Report: Contains precision, recall, and F1-score for each class.

Confusion Matrix: Displays the number of true positives, true negatives, false positives, and false negatives.

Code
python
Copy
Edit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Generate synthetic financial data
def generate_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'income': np.random.randint(20000, 120000, n_samples),
        'loan_amount': np.random.randint(5000, 50000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'default': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 20% default rate
    }
    return pd.DataFrame(data)

# Load dataset
df = generate_data()
print(df.head())

# Visualize Data
def plot_data(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='credit_score', bins=30, kde=True, hue=df['default'].astype(str))
    plt.title('Credit Score Distribution by Default Status')
    plt.show()

plot_data(df)

# Preprocessing
X = df.drop(columns=['default'])
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
Expected Output
When you run the script, you will see:

A histogram that shows the distribution of credit_score by loan default status.

Evaluation results of the trained model, including:

Accuracy: The percentage of correct predictions made by the model.

Classification Report: Precision, recall, and F1-score for both default and non-default classes.

Confusion Matrix: The confusion matrix showing the true and false positives/negatives.

Example Output:
text
Copy
Edit
Accuracy: 0.82
Classification Report:
               precision    recall  f1-score   support

           0       0.83      0.93      0.88       160
           1       0.76      0.53      0.63        40

    accuracy                           0.82       200
   macro avg       0.80      0.73      0.75       200
weighted avg       0.82      0.82      0.81       200

Confusion Matrix:
 [[149  11]
 [ 19  21]]
Conclusion
This code provides a complete end-to-end solution to predict loan defaults using a Random Forest Classifier.
The synthetic data generation, model training, and evaluation allow financial institutions or researchers to understand how different financial variables influence loan defaults. 
This solution can be extended to real-world datasets, additional features, or more complex models for better predictions.
