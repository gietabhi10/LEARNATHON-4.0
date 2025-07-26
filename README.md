# AI-Powered Auto Insurance Fraud Detection

This project, developed for the Learnathon 4.0, focuses on building a machine learning model to detect fraudulent claims in the auto insurance industry. By leveraging predictive modeling, the goal is to create a system for smarter, more efficient claims management.

# Team Members
1. Abhishek Kumar Sharma (22CSE1068)
2. Satyabrata Mund       (22CSE045)
3. Raushan Kumar Singh   (22CSE365)

# 📝 Project Description
Insurance fraud is a significant problem that leads to substantial financial losses for insurance companies and results in higher premiums for honest customers. This project aims to automate the detection of suspicious claims by training a binary classification model on historical claims data. The model learns to distinguish between fraudulent and legitimate claims based on various features, allowing investigators to focus their efforts on high-risk cases.

The core task is to predict the **Fraud_Reported** status ('Y' or 'N') for a given insurance claim.

# 📊 Dataset
The dataset for this project is provided in three separate CSV files, which need to be consolidated for analysis. A data dictionary is also provided to explain each feature.

* *Data Source Files*:
    * Auto_Insurance_Fraud_Claims_File01.csv
    * Auto_Insurance_Fraud_Claims_File02.csv
    * Auto_Insurance_Fraud_Claims_File03.csv
* *Data Dictionary*: Data Dictionary for Auto Insurance Fraud Claims Data.txt
* *Target Variable*: Fraud_Reported

# Key Categorical Features
The model relies on several categorical features to identify patterns, including:
* Gender
* Marital_Status
* Accident_Site
* Witness_Present_Ind
* Channel
* Vehicle_Model
* Vehicle_Color

# ⚙ Project Setup and Installation

To run this project, you'll need Python and several data science libraries.

1.  *Clone the repository:*
    bash
    git clone [Your-Repository-Link]
    cd [Your-Repository-Name]
    

2.  *Install the required libraries:*
    It's recommended to use a virtual environment.
    bash
    pip install pandas scikit-learn matplotlib seaborn jupyter
    

# 🚀 Model Building Workflow
The solution was developed following a standard machine learning pipeline:

1.  Data Consolidation: The three source CSV files were loaded and merged into a single DataFrame using the Pandas library.

2.  Exploratory Data Analysis (EDA): We analyzed the dataset to uncover insights and relationships between features and the target variable (Fraud_Reported). This step involved visualizing distributions and identifying initial patterns indicative of fraud.

3.  Data Preprocessing: The data was cleaned and prepared for modeling. This included:
    * Handling any missing values in the dataset.
    * Encoding categorical features (like Marital_Status, Vehicle_Color, etc.) into a numerical format using techniques like One-Hot Encoding so the model could process       them.

4.  Model Training:
    * The dataset was split into training (80%) and testing (20%) sets.
    * Several classification algorithms were trained and evaluated, including Logistic Regression, Random Forest, and XGBoost.

5.  Model Evaluation: Models were evaluated based on their performance on the unseen test data. Key metrics included:
    * Accuracy: Overall correct predictions.
    * Precision: The model's ability to avoid flagging legitimate claims as fraudulent.
    * Recall: The model's ability to identify all actual fraudulent claims.
    * F1-Score: A weighted average of Precision and Recall, providing a balanced measure of performance.

6.  Hyperparameter Tuning: The best-performing model was further optimized by tuning its hyperparameters to achieve maximum performance.

# Usage
To see the model-building process, open and run the Jupyter Notebook provided in this repository.

bash
jupyter notebook [Your-Notebook-Name].ipynb
