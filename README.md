#Loan Approval Detection Model

#🚀 Project Overview
This project focuses on detecting fraudulent loan activities with high accuracy using advanced machine learning techniques. The model achieved an impressive 99.59% accuracy and was developed with a robust feature set of 80+ columns.

#📊 Dataset
Size: Over 50,000 rows and 80+ features.
Features: Includes both numerical and categorical data, such as transaction amounts, user profiles, and timestamps.
Target Variable: Multiclass classification (P1: Loan Approved, P2: To be considered, P3: Few loan acceptance, P4: Loan should be denied).

#🛠️ Models Used
Several machine learning algorithms were employed to identify the best-performing model:

- Decision Tree
- Random Forest
- XGBoost
- CatBoost
- LightGBM

#🏆 Performance
- Accuracy: 99.59%
- Evaluation Metrics:
  - Precision
  - Recall
  - F1 Score
  - AUC-ROC Curve
 
#📈 Key Features
- Feature Engineering: Applied techniques like one-hot encoding, label encoding, and scaling.
- Model Optimization: Leveraged hyperparameter tuning (GridSearchCV/Optuna).

#🧰 Tools & Technologies
- Programming Language: Python
- Libraries:
  - scikit-learn
  - XGBoost
  - CatBoost
  - LightGBM
  - polars, numpy for data preprocessing

#🚀 Getting Started
1. Clone the repository:
```
git clone https://github.com/your-username/credit-fraud-detection.git
cd loan-approval-detection
```

2. Install dependencies:
```
pip install -r requirements.txt

```
3. Run the model training script:
```
python model.py
```
