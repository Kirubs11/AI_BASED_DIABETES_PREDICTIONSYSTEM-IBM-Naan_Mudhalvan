# AI-Based Diabetes Prediction System

## Overview

This project focuses on building an AI-based Diabetes Prediction System that utilizes machine learning algorithms to analyze medical data and predict the likelihood of an individual developing diabetes. The system aims to provide early risk assessment and personalized preventive measures, empowering individuals to take proactive actions to manage their health effectively.

## Table of Contents

1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Machine Learning Algorithm](#machine-learning-algorithm)
5. [Model Training](#model-training)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Innovative Techniques](#innovative-techniques)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)

## Project Description
The AI-Based Diabetes Prediction System is a data-driven healthcare project designed to leverage the power of artificial intelligence and machine learning for early diabetes risk assessment and prevention. Diabetes is a widespread health issue with long-term health consequences. Early detection and personalized interventions are key to improving health outcomes.

**Objectives**:
- Build a robust machine learning model capable of predicting an individual's risk of developing diabetes based on their medical data.
- Enable individuals to receive early risk assessments and personalized recommendations for diabetes prevention.
- Facilitate healthcare professionals in providing more proactive and targeted care to patients at risk.

**Significance**:
- Early diabetes prediction can lead to early interventions, lifestyle adjustments, and better health management.
- Improved healthcare resource allocation by focusing on high-risk individuals.
- Enhanced patient-doctor communication through data-driven insights.
- Potential cost savings in healthcare expenditures by reducing the burden of late-stage diabetes management.

This project aims to harness the potential of AI and data science to make a positive impact on public health by addressing the growing challenge of diabetes. It combines advanced technology with healthcare to promote healthier living and more informed decision-making.

## Dataset

### Source

- Dataset Name: [Diabetes Data Set](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)
- Description: The dataset contains medical features such as glucose levels, blood pressure, BMI, etc., along with information about whether the individual has diabetes or not.

### Data Preprocessing

Data preprocessing is a critical step to ensure that the dataset is well-prepared for model development. This section outlines the steps involved, along with relevant code snippets and visualizations:

#### Handling Missing Values

In some cases, the dataset may contain missing values that need to be addressed. For example, if the 'Glucose' feature has missing values, we can replace them with the mean value of the column using Python and pandas:

```python
# Handle missing values (if any)
data['Glucose'].fillna(data['Glucose'].mean(), inplace=True)
```
#### Feature Selection
Choosing the right features is essential for building an accurate prediction model. Relevant features should be selected based on domain knowledge and data analysis. In this project, we have chosen the following columns for our analysis: 'Glucose', 'BloodPressure', 'BMI', 'Age', and 'Outcome'.

#### Feature Engineering
Feature engineering involves creating new features or transforming existing ones to provide valuable insights. One innovative feature we've added is 'BMI Category,' which categorizes individuals based on their BMI values. This can help capture non-linear relationships with the target variable.
```python
# Example: Creating a new feature for BMI categories
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 24.9:
        return 'Normal'
    elif 25 <= bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obese'

data['BMI Category'] = data['BMI'].apply(categorize_bmi)
```
These preprocessing steps are crucial to ensure that the data is ready for machine learning model development.

## Machine Learning Algorithm

The machine learning algorithm selected for the Diabetes Prediction Model is the Random Forest classifier. Random Forest is a powerful ensemble learning method that combines multiple decision trees to make accurate predictions. It was chosen for the following reasons:

1. **Ensemble Learning**: Random Forest is an ensemble method that combines the predictions of multiple decision trees. This ensemble approach helps improve prediction accuracy and reduces the risk of overfitting.

2. **Non-Linearity Handling**: Random Forest can capture complex non-linear relationships in the data. This is important for predicting diabetes risk, as the relationship between medical features and diabetes can be intricate.

3. **Robustness**: Random Forest is robust to noisy data and outliers, which is valuable when dealing with medical datasets that may contain variations and anomalies.

4. **Feature Importance**: Random Forest provides a measure of feature importance, allowing us to understand which medical features have the most significant impact on diabetes prediction.

5. **Scalability**: Random Forest is suitable for both small and large datasets, making it versatile for different data sizes.

Here is an example of how the Random Forest classifier is initialized and trained using Python's scikit-learn library:

```python
# Importing necessary libraries
from sklearn.ensemble import RandomForestClassifier

# Initializing the Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# Training the Random Forest classifier on the training data
clf.fit(X_train, y_train)
```
Random Forest offers a strong foundation for building a reliable diabetes prediction model, and its ensemble nature makes it well-suited for this project.

## Model Training

In this section, we will discuss how the selected Random Forest classifier model was trained on the diabetes dataset. We'll provide key code snippets and explain important parameters used during the training process.

### Initializing the Random Forest Classifier

First, we import the necessary libraries and initialize the Random Forest classifier. We set the `random_state` parameter to ensure reproducibility.

```python
# Importing necessary libraries
from sklearn.ensemble import RandomForestClassifier

# Initializing the Random Forest classifier
clf = RandomForestClassifier(random_state=42)
```

### Training the Model
Next, we train the Random Forest classifier on the training data. The training data consists of the features (X_train) and the corresponding target variable (y_train).

```python
# Training the Random Forest classifier on the training data
clf.fit(X_train, y_train)
```
During training, the model learns from the training data and builds multiple decision trees to make predictions.

### Model Parameters
The Random Forest classifier allows customization through various parameters, including the number of trees in the forest, maximum depth of the trees, and feature selection criteria. In this example, we used default parameters. However, you can fine-tune these parameters to optimize the model's performance for your specific dataset.

Training the model is a crucial step in building an accurate diabetes prediction system, and Random Forest provides a strong foundation for this task.

## Evaluation Metrics

In this section, we will discuss the evaluation metrics chosen for assessing the performance of our diabetes prediction model. We'll explain the significance of each metric and provide the results obtained from the model evaluation.

### Choice of Evaluation Metrics

1. **Accuracy**: Accuracy measures the overall correctness of predictions and is useful when the class distribution is balanced.

2. **Precision**: Precision assesses the model's ability to correctly classify positive cases (individuals with diabetes) without falsely labeling negative cases as positive.

3. **Recall**: Recall evaluates the model's ability to identify all positive cases (individuals with diabetes) without missing any.

4. **F1-Score**: The F1-Score is the harmonic mean of precision and recall. It balances the trade-off between precision and recall.

5. **ROC-AUC**: The Receiver Operating Characteristic-Area Under the Curve (ROC-AUC) measures the model's ability to distinguish between positive and negative cases across different thresholds.

### Model Evaluation Results

The Random Forest classifier was evaluated using the chosen metrics on the test dataset. Here are the results:

```python
# Importing necessary libraries for evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Predicting outcomes on the test set
y_pred = clf.predict(X_test)

# Calculating evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

# Displaying the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)
```
These evaluation metrics provide a comprehensive understanding of the model's performance in predicting diabetes cases. It is crucial to assess the model using a combination of these metrics to ensure its reliability and effectiveness.

## Innovative Techniques

In this section, we'll highlight the innovative techniques and approaches that have been incorporated into the development of our diabetes prediction system.

### Feature Engineering with 'BMI Category'

One innovative technique we introduced is the creation of a new feature called 'BMI Category.' This feature categorizes individuals based on their BMI (Body Mass Index) values. It allows us to capture non-linear relationships between BMI and the likelihood of developing diabetes. The categorization is as follows:

- 'Underweight' for BMI values less than 18.5
- 'Normal' for BMI values between 18.5 and 24.9
- 'Overweight' for BMI values between 25 and 29.9
- 'Obese' for BMI values of 30 or greater

This approach provides a more granular understanding of how BMI affects diabetes risk, going beyond a simple numerical feature.

### Ensemble Learning with Random Forest

Ensemble learning is an innovative approach that combines multiple decision trees to make predictions. We chose the Random Forest classifier, which is an ensemble method. Random Forest combines the results of multiple decision trees to improve prediction accuracy and reduce the risk of overfitting. This ensemble approach is a powerful and innovative technique in the field of machine learning.

These innovative techniques contribute to the robustness and accuracy of our diabetes prediction system, enabling it to provide valuable insights and early risk assessments for individuals.

## Usage

To run the code and reproduce the results of the AI-based Diabetes Prediction System, follow these steps. You can run the code both locally and online, such as in Google Colab.

### Local Environment

1. **Install Dependencies**:
Make sure you have the necessary Python libraries installed. You can install them using `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn mlxtend
```
2. **Clone the Repository**:
Clone this GitHub repository to your local machine:

```bash
git clone https://github.com/Kirubs11/AI_BASED_DIABETES_PREDICTIONSYSTEM-IBM-Naan_Mudhalvan
```
or else it can be download by clicking `Code` button in the top and clicking the `Download ad zip` option.

3. **Run the Jupyter Notebook**:
Open the Jupyter Notebook provided in the repository and run each code cell step by step. The notebook is located in the notebooks directory.

### Google Colab
1. **Open Google Colab**:
You can directly open the Jupyter Notebook in Google Colab 

2. **Run the Notebook**:
In Google Colab, you can run each cell by clicking the play button or using the keyboard shortcut `Shift+Enter`. Make sure to follow the instructions within the notebook.

Dataset
The dataset used in this project can be found [here](https://www.kaggle.com/datasets/mathchi/diabetes-data-set).

These instructions will allow you to run the code and explore the AI-based Diabetes Prediction System in your preferred environment.

