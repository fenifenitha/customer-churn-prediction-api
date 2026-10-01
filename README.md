# Customer Churn Prediction API

A machine learning project that predicts whether a telecom customer is likely to churn based on customer information and service usage details.

The project includes data preprocessing, exploratory data analysis, handling class imbalance using SMOTE, machine learning model comparison, feature importance analysis, model persistence, and a FastAPI prediction API.

## Project Overview

Customer churn is an important business problem for subscription-based companies. Identifying customers who are likely to leave can help businesses take preventive actions such as improving customer support, offering suitable plans, or providing retention offers.

This project builds a machine learning solution to classify customers into:

* `Yes` — Customer is predicted to churn
* `No` — Customer is predicted to stay

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Category Encoders
* Imbalanced-learn
* Matplotlib
* Seaborn
* Joblib
* FastAPI
* Uvicorn
* Google Colab
* VS Code

## Dataset

The project uses a Telco Customer Churn dataset containing customer demographic information, services, contract details, charges, satisfaction information, and churn labels.

The dataset contains:

* 7,043 customer records
* Multiple numerical and categorical features
* `Churn Label` as the target variable

## Machine Learning Workflow

The project follows the following workflow:

```text
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Duplicate Checking
   ↓
Exploratory Data Analysis
   ↓
Outlier Analysis
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
SMOTE
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Feature Importance
   ↓
Model Saving
   ↓
FastAPI Deployment
```

## 1. Data Understanding and Preprocessing

The dataset was first inspected to understand:

* Dataset shape
* Data types
* Missing values
* Duplicate records
* Numerical features
* Categorical features
* Target class distribution

### Missing Values

Missing values were identified and handled during preprocessing.

For example:

* Missing `Offer` values were replaced with `No Offer`
* Missing `Internet Type` values were replaced with `No Internet`

Columns related directly to the churn outcome explanation were removed from the modeling dataset to reduce data leakage.

## 2. Data Cleaning

The following columns were removed before model training:

* `Customer ID`
* `Churn Score`
* `Customer Status`
* `Zip Code`
* `Latitude`
* `Longitude`
* `Churn Category`
* `Churn Reason`

These columns were removed because some are identifiers, geographic fields, or outcome-related information that could introduce leakage or unnecessary complexity into the prediction task.

## 3. Exploratory Data Analysis

Several visualizations were created to understand relationships between customer characteristics and churn.

The analysis included:

* Churn distribution
* Age vs Churn
* Tenure vs Churn
* Monthly Charge vs Churn
* Contract vs Churn
* Internet Service vs Churn
* Payment Method vs Churn
* Satisfaction Score vs Churn
* Total Revenue vs Churn
* Correlation heatmap
* Outlier analysis

### Contract and Churn

The analysis showed different churn proportions across contract types.

Month-to-month customers showed a substantially higher churn proportion compared with customers on longer-term contracts.

## 4. Class Imbalance

The target distribution was:

```text
No     5174
Yes    1869
```

This represents approximately:

```text
No     73.46%
Yes    26.54%
```

Since the classes were imbalanced, SMOTE was applied only to the training data.

### SMOTE

SMOTE was used to create synthetic samples for the minority class.

Before SMOTE:

```text
No     4139
Yes    1495
```

After SMOTE:

```text
No     4139
Yes    4139
```

The test dataset was kept unchanged for unbiased evaluation.

## 5. Feature Encoding

Categorical features were converted into numerical representations before model training.

### City Encoding

`City` contains many unique values, so Target Encoding was used.

```text
City → Target Encoding
```

### Other Categorical Features

Other categorical columns were converted using One-Hot Encoding.

The final processed dataset contained 68 numerical features.

## 6. Feature Scaling

`StandardScaler` was used to scale the processed numerical feature matrix.

The scaler fitted on the training data was also saved and reused during API prediction.

## 7. Machine Learning Models

Two classification models were trained:

### Logistic Regression

Logistic Regression was used as a baseline classification model.

### Random Forest

Random Forest was used because it can capture non-linear relationships between customer features and churn and also provides feature importance values.

## 8. Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

### Logistic Regression

```text
Accuracy  : 0.9581
Precision : 0.9315
Recall    : 0.9091
F1 Score  : 0.9202
```

Confusion Matrix:

```text
[[1010   25]
 [  34  340]]
```

### Random Forest

```text
Accuracy  : 0.9610
Precision : 0.9761
Recall    : 0.8743
F1 Score  : 0.9224
```

Confusion Matrix:

```text
[[1027    8]
 [  47  327]]
```

The metrics show that the models have different precision and recall characteristics. In a churn-detection application, recall is particularly important because missing an actual churn customer may reduce the opportunity for a retention action.

## 9. Feature Importance

Random Forest feature importance was used to understand which features contributed most to the model's predictions.

Top features included:

```text
Satisfaction Score
City
Contract_Month-to-Month
Contract_Two Year
Tenure in Months
Number of Referrals
Payment Method_Credit Card
Internet Type_Fiber Optic
Dependents_No
Monthly Charge
Total Charges
Total Revenue
```

Feature importance indicates how useful a feature was to the trained model's decision process. It should not be interpreted as proof that the feature directly causes churn.

## 10. Model Saving

The trained Random Forest model and preprocessing objects were saved using Joblib.

Saved files:

```text
random_forest_model.pkl
city_target_encoder.pkl
scaler.pkl
feature_columns.pkl
```

These files allow the trained model and preprocessing pipeline components to be reused without retraining.

## 11. FastAPI Deployment

The trained model was deployed using FastAPI.

The API provides:

### GET `/`

Checks whether the API is running.

Example response:

```json
{
  "message": "Customer Churn Prediction API is running"
}
```

### POST `/predict`

Accepts customer information and returns a churn prediction.

Example response:

```json
{
  "prediction": "Yes"
}
```

The API performs the same preprocessing steps used during model training:

```text
Input JSON
   ↓
City Target Encoding
   ↓
One-Hot Encoding
   ↓
Feature Alignment
   ↓
Scaling
   ↓
Random Forest
   ↓
Prediction
```

## 12. Project Structure

```text
customer-churn-prediction-api/
│
├── notebook/
│   └── customer_churn_prediction.ipynb
│
├── app.py
│
├── requirements.txt
│
├── .gitignore
│
└── content/
    └── model/
        ├── random_forest_model.pkl
        ├── city_target_encoder.pkl
        ├── scaler.pkl
        └── feature_columns.pkl
```

## 13. Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
```

Move into the project directory:

```bash
cd customer-churn-prediction-api
```

Create a virtual environment:

### Windows

```bash
python -m venv env
```

Activate it:

```bash
env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 14. Run the FastAPI Application

Start the API using:

```bash
uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

The `/docs` page can be used to test the prediction API interactively.

## 15. Example API Input

Example customer input:

```json
{
  "Gender": "Male",
  "Age": 30,
  "Under 30": "Yes",
  "Senior Citizen": "No",
  "Married": "Yes",
  "Dependents": "No",
  "Number of Dependents": 0,
  "Country": "United States",
  "State": "California",
  "City": "Los Angeles",
  "Population": 100000,
  "Quarter": "Q3",
  "Referred a Friend": "No",
  "Number of Referrals": 0,
  "Tenure in Months": 5,
  "Offer": "No Offer",
  "Phone Service": "Yes",
  "Avg Monthly Long Distance Charges": 20.5,
  "Multiple Lines": "No",
  "Internet Service": "Yes",
  "Internet Type": "Fiber Optic",
  "Avg Monthly GB Download": 50,
  "Online Security": "No",
  "Online Backup": "No",
  "Device Protection Plan": "No",
  "Premium Tech Support": "No",
  "Streaming TV": "No",
  "Streaming Movies": "No",
  "Streaming Music": "No",
  "Unlimited Data": "Yes",
  "Contract": "Month-to-Month",
  "Paperless Billing": "Yes",
  "Payment Method": "Credit Card",
  "Monthly Charge": 75.5,
  "Total Charges": 377.5,
  "Total Refunds": 0,
  "Total Extra Data Charges": 0,
  "Total Long Distance Charges": 102.5,
  "Total Revenue": 480,
  "Satisfaction Score": 2,
  "CLTV": 4000
}
```

Example response:

```json
{
  "prediction": "Yes"
}
```

## 16. Possible Improvements

Future improvements could include:

* Hyperparameter tuning
* Cross-validation
* XGBoost model implementation
* Better high-cardinality feature encoding
* Model monitoring
* Automated model retraining
* API input validation using Pydantic schemas
* Docker deployment
* Cloud deployment
* Frontend interface for customer predictions

## 17. Conclusion

This project demonstrates an end-to-end machine learning workflow, starting from raw customer data and preprocessing and continuing through model training, evaluation, model persistence, and API deployment.

The final system can receive customer information through a FastAPI endpoint and return a churn prediction in JSON format.
