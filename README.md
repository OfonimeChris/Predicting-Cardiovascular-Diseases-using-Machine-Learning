### Predicting-Cardiovascular-Diseases-using-Machine-Learning

This project aims to develop machine learning models to predict the risk of cardiovascular diseases (CVDs) using patient health data.
## Project Overview
Cardiovascular diseases are a leading cause of death globally. Early prediction and intervention can significantly reduce mortality rates. This project utilizes machine learning algorithms to analyze patient data, including medical history, lifestyle factors, and clinical measurements, to predict the likelihood of CVDs.
## Dataset
The dataset used in this project was obtained from Doppala, B. P., & Bhattacharyya, D. (2021) and is available on Mendeley Data. It contains records of patients with cardiovascular disease-related data, with a total of 1000 records. The dataset includes 13 features, such as:
## Age
Gender
Blood Pressure
Cholesterol levels
Smoking habits
## Family history
The dataset was preprocessed to handle missing values, encode categorical variables, and scale numerical features for better model performance. The features were also analyzed for any correlations, and feature engineering techniques were applied where necessary.
## Methodology
## The project follows these steps:
1.	Data Exploration and Preprocessing:
o	Loading and cleaning the dataset.
o	Handling missing values.
o	Encoding categorical features.
o	Scaling numerical features.
o	Exploratory data analysis (EDA) to understand feature distributions and relationships.
2.	Feature Selection/Engineering (Optional):
o	Identifying important features using techniques like feature importance from tree-based models or statistical tests.
o	Creating new features from existing ones if necessary.
3.	Model Development:
o	Training various machine learning models (e.g., Logistic Regression, Random Forest, Support Vector Machines, Gradient Boosting, Neural Networks).
o	Splitting the data into training and testing sets.
o	Hyperparameter tuning using techniques like GridSearchCV or RandomizedSearchCV.
4.	Model Evaluation:
o	Evaluating model performance using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
o	Visualizing results using confusion matrices and ROC curves.
5.	Model Deployment (Optional):
## Files
•	data/: Contains the dataset.
•	notebooks/: Contains Jupyter Notebooks with code for data preprocessing, model training, and evaluation.
•	models/: Contains saved machine learning models.
•	README.md: Project documentation.
•	requirements.txt: List of Python dependencies.
## Requirements
To run this project, you need the following Python libraries:
pandas: For data manipulation and analysis.
numpy: For numerical operations and array handling.
matplotlib: For data visualization (plotting graphs and charts).
seaborn: For advanced data visualization and statistical plotting.
scikit-learn: For machine learning algorithms, model selection, and evaluation.
tensorflow or keras: For building and training neural networks (e.g., LSTM model).
xgboost: For gradient boosting models.
statsmodels: For statistical modeling and hypothesis testing.
scipy: For scientific computing, including optimization and integration

## Feature Importance
![image](https://github.com/user-attachments/assets/75da8957-1346-4d06-bb7b-7e191dd212e4)
## Summary of Feature Importance Chart:
•	Top Feature: slope is the most important predictor of cardiovascular disease in the model, contributing over 40% of the total importance. This makes it a critical variable for diagnosis.
•	Other Key Predictors: chestpain, restingBP, and noofmajorvessels also have strong influence, indicating that both symptom-based (e.g., chest pain) and clinical test-based features (e.g., blood pressure, vessel count) are valuable.
•	Moderately Important Features: Features like serumcholestrol, restingrelectro, and maxheartrate show moderate predictive power.
•	Least Important Features: exerciseangia, age, and gender have very low importance in this model. While they are relevant in clinical settings, the model may not be relying on them due to data distribution or feature redundancy.


## Machine Learning Models Used!
# Logistic Regression
  Interpretable baseline model
  Good performance on linear patterns
  Accuracy: 96.5%
  
# XGBoost Classifier
Most accurate and robust model
Excellent at handling structured data with fine-tuned parameters
Accuracy: 99%

## Random Forest Classifier
Captures complex, non-linear relationships
Provides feature importance for interpretability
Accuracy: 98.5%

## Neural Network
High recall but low precision
Underperformed due to poor balance between metrics
Accuracy: 83%
![image](https://github.com/user-attachments/assets/a4070175-fd03-470a-891b-ca5201f01714)

## Model Evaluation 
To assess the performance of each model, the following evaluation metrics were used:
•	Accuracy: The proportion of correct predictions over total predictions.
•	Precision and Recall: Precision measured the accuracy of positive predictions, while recall measured the ability to identify all positive cases.
•	F1 Score: The harmonic mean of precision and recall, particularly useful for imbalanced datasets.
•	Confusion Matrix: Provided insights into true/false positives and negatives.
•	ROC Curve and AUC: Assessed the trade-off between sensitivity and specificity. Models with higher AUC were favored.
Among all models tested, the Random Forest Classifier achieved the best performance, with high precision, recall, and an AUC score close to 0.9, indicating excellent discrimination capability.

## Model Evaluation Results
After training the models on the training set and evaluating them on the test set, the following performance metrics were recorded:

Model	Accuracy	Precision (0/1)	Recall (0/1)	F1-Score (0/1)	ROC-AUC
Logistic Regression	0.965	0.96/0.97	0.95/0.97	0.96/0.97	~0.96
Random Forest	0.985	0.99/0.98	0.98/0.99	0.98/0.99	~0.99
XGBoost	0.990	0.99/0.99	0.99/0.99	0.99/0.99	  0.99
Neural Network	0.83	0.72/0.98	0.98/0.72	0.83/0.83	  0.85

## Observation and Insights
•	XGBoost has the highest accuracy (0.99) and balanced precision-recall values.
•	Logistic Regression and Random Forest perform equally well (0.965 AND 0.985 accuracy Respectively).
•	Neural Network struggles with recall on Class 1 (72%), indicating possible underfitting.
Based on our baseline results, we did fine-tuning for the top two models which are:
## XGBoost: 
This is the Best Model with the Highest accuracy of (0.99). There is Balanced precision-recall across both classes, and it is generally robust and handles feature importance well.
## Random Forest:
The Random Forest also has (Strong Performance & Interpretability). The Accuracy (0.985) is similar to Logistic Regression. It is more flexible and can improve with tuning (number of trees, depth, etc.). It is easier to interpret compared to XGBoost.

## Why Not Logistic Regression or Neural Network?
•	Although Logistic Regression performed well, but the tree-based models (RF & XGBoost) offer more flexibility in capturing complex patterns.  
•	Neural Network also underperformed with (accuracy = 0.83, recall issues for Class 1)  

## Hyperparameter Tuning Result for the Models
Here’s a table summarizing the Final Fine-Tuned Parameters and Here’s the Model Comparison Summary: for both Random Forest (RF) and XGBoost:

![image](https://github.com/user-attachments/assets/41f73712-a52e-427e-b686-89c6d4d524dc)

![image](https://github.com/user-attachments/assets/45c0a01e-064a-4f17-8de0-0cb544ddeb58)


## INSIGHTS
1 XGBoost Wins Marginally:
•	1 fewer false positive (83 correct Class 0 predictions vs 82 in RF).
•	Perfect recall (0.99) for both classes, while RF has 0.98 for Class 0.
2 Both Models Are Production-Ready:
•	Near-perfect metrics across the board.
•	Minimal misclassifications (2 errors in RF, 1 in XGBoost).
3 Minimal Room for Improvement:
•	The remaining errors may be "hard" cases (e.g., ambiguous feature patterns).

## Model Deployment 
The final phase of this project involved deploying the trained machine learning model to a user-friendly web interface. This allows end users especially healthcare practitioners or patients to input new data and receive instant predictions on cardiovascular disease (CVD) risk.

## Model Deployment Result
After successfully training and evaluating multiple models, the Random Forest Classifier, which achieved 98.5% accuracy and excellent performance across precision, recall, and F1-score was selected for deployment due to its strong balance between predictive power and clinical interpretability.
The model was deployed using a Gradio-based web application, where users can input 13 medical attributes for a new patient and instantly receive a prediction on whether the patient is at risk of cardiovascular disease.

## Discussion
The results gotten from this study demonstrate a strong ability to predict cardiovascular disease (CVD) using machine learning models trained on structured patient data. The final deployed model (Random Forest Classifier) achieved a high accuracy of 98.5%, with excellent precision and recall values across both CVD-positive and negative classes. This signifies a highly reliable diagnostic tool capable of distinguishing between patients with and without CVD based on clinical parameters alone.

## The results are particularly significant because:
•	Early Prediction: The model enables early detection of potential heart conditions, helping doctors prioritize patients for further testing or intervention.
•	Accessibility: With minimal input features and real-time response through the Gradio interface, this system can be used in under-resourced or rural settings with limited access to specialists or advanced diagnostics.
•	Reinforcement of Clinical Knowledge: Key features identified by the model such as chest pain type, oldpeak, maximum heart rate, and number of major vessels align closely with established cardiovascular risk markers, adding validity to the model's predictions.
The findings from this project are largely consistent with what other researchers have reported in similar studies:
•	Khan et al. (2021) and Dua et al. (2020) also reported strong performance using Random Forest and XGBoost models on heart disease datasets, achieving accuracy scores in the 85–90% range.
•	In this project, XGBoost reached up to 99% accuracy, slightly higher than many reported values. This may be attributed to:
o	Good preprocessing (especially scaling)
o	Careful feature selection
o	Effective hyperparameter tuning
While some research papers suggest Neural Networks outperform traditional models, in this study, the Neural Network underperformed, achieving only 83% accuracy and poor recall for the positive class. This contrast suggests that for smaller or tabular datasets, simpler ensemble methods like Random Forest may generalize better.

## CHALLENGES
•	Data Limitations: Gender imbalance and dataset size may affect generalizability
•	Model Selection: Multiple models showed high accuracy, risking overfitting
•	Tuning Complexity: Fine-tuning hyperparameters for XGBoost and Random Forest required multiple iterations

## CONCLUSION
Machine learning effectively predicted cardiovascular disease (CVD).
Random Forest selected for deployment (98.5% accuracy, high precision & recall, interpretable).
XGBoost had the highest accuracy (99%) but lower interpretability.
Developed a Gradio web app for real-time CVD risk prediction using 13 medical inputs.
Key predictors: chest pain type, oldpeak, max heart rate, number of major vessels.
Enhances early detection and supports use in resource-limited settings.

## RECOMMENDATION
Integrate ML tools into clinical workflows for early screening
Use Random Forest models for deployment due to balanced performance and interpretability
Expand dataset to include diverse populations for better generalization.
Incorporate real-time patient data for dynamic prediction updates.
Explore hybrid models combining ML with expert systems for complex cases.




