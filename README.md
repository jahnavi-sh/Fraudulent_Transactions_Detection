# Fraudulent_Transactions_Detection

## Overview:
This Python project implements a robust fraud detection system using machine learning algorithms on financial transaction data. The goal is to accurately identify fraudulent transactions while minimizing false positives.

## Workflow and Techniques:

### 1. **Data Loading and Exploration:**
   - **Libraries Used:** NumPy, Pandas
   - Reads the dataset from 'Fraud.csv'.
   - Conducts thorough exploratory data analysis (EDA) to understand the data's structure, checking for missing values (none found).
   - Drops unnecessary columns ('nameOrig', 'nameDest') to streamline the dataset.

### 2. **Outlier Detection and Removal:**
   - **Libraries Used:** Seaborn, Matplotlib
   - Identifies outliers using boxplots in relevant columns (e.g., 'amount', 'oldbalanceOrg') and removes them. This step ensures the models aren't skewed by extreme values.

### 3. **Data Visualization:**
   - **Libraries Used:** Seaborn, Matplotlib
   - Visualizes feature correlations using a heatmap to understand relationships between variables.
   - Plots pie charts and bar graphs to analyze transaction types ('type') and their proportions, providing insights into the dataset's composition.

### 4. **Data Preprocessing:**
   - **Libraries Used:** Pandas, Scikit-Learn
   - Implements Min-Max normalization on selected columns to scale features between 0 and 1, ensuring uniformity for accurate modeling.
   - Converts the categorical column 'type' into binary features using one-hot encoding, making it machine-readable.

### 5. **Handling Imbalanced Data:**
   - **Libraries Used:** Imbalanced Learn
   - Utilizes the NearMiss technique for undersampling the majority class ('Not Fraud'). This technique rebalances the dataset, addressing class imbalance and improving model performance.

### 6. **Model Building and Evaluation:**
   - **Libraries Used:** Scikit-Learn
   - Divides the preprocessed data into training and testing sets for model evaluation.
   - Constructs multiple classifiers, including Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Support Vector Machine, Naive Bayes, and XGBoost.
   - Evaluates models using a range of metrics, such as ROC AUC, F1 Score, Confusion Matrix, Classification Report, and Accuracy Score. These metrics provide a comprehensive understanding of each model's performance.

## Conclusion:
This fraud detection system combines data preprocessing, feature engineering, visualization, and machine learning techniques to create a robust and accurate solution. The models are rigorously evaluated to ensure they effectively differentiate between legitimate and fraudulent transactions. The project provides valuable insights into the complexities of fraud detection in financial transactions.
