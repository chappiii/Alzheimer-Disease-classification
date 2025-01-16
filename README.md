# README: A Comparative Analysis of Machine Learning Models for Alzheimer’s Disease Screening

## Abstract

This project develops a machine learning approach for binary classification of Alzheimer’s disease using a dataset of 2,149 instances and 32 features. A systematic workflow was followed, including feature selection, model evaluation, and interpretability analysis. Models like SVM, Random Forest, XGBoost, and CatBoost, along with a stacked ensemble, were compared. CatBoost slightly outperformed other models; however, Random Forest was selected due to its superior interpretability. The results demonstrate the potential of explainable machine learning in healthcare applications.

---

## Introduction

Alzheimer’s disease (AD) is a progressive neurodegenerative disorder and the leading cause of dementia globally, accounting for 60-70% of cases. Early diagnosis can delay disease progression and improve quality of life. This project aims to develop a machine learning model capable of screening potential patients through a simple questionnaire. The model emphasizes explainability and interpretability to build trust among domain experts and enhance patient confidence.

The project uses a dataset from Kaggle containing structured clinical data for Alzheimer’s diagnosis and prediction. This initial screening helps identify potential cases for further clinical evaluation.

---

## Methodology

### Data Understanding and Preprocessing
- **Dataset**: 2,149 observations with 32 features (15 numerical and 17 categorical).
- **Target Variable**: Binary, where 0 represents non-Alzheimer’s cases, and 1 represents Alzheimer’s cases.
- **Class Imbalance**: Moderate imbalance (35.4% Alzheimer’s, 64.6% non-Alzheimer’s).

#### Preprocessing Steps:
1. **Data Balancing**: Undersampling was used to address class imbalance, ensuring stable and reliable results.
2. **Data Encoding**: One-hot encoding was applied to categorical variables with multiple classes.
3. **Feature Selection**: Techniques included Spearman rank correlation, mutual information, PCA, and Recursive Feature Elimination (RFE). Top five features consistently ranked highest across methods: 
   - FunctionalAssessment
   - ADL
   - MemoryComplaints
   - MMSE
   - BehavioralProblems

### Model Selection
The following machine learning models were evaluated:
- Decision Tree
- Random Forest
- CatBoost
- XGBoost
- AdaBoost
- Support Vector Machine (SVM)

Random Forest was selected as the final model due to its balance of performance and interpretability.

### Performance Metrics
Models were evaluated using Accuracy, Precision, Recall, and F1-Score. Stratified cross-validation was performed to ensure robustness.

---

## Results Analysis and Discussion

### Model Evaluation
Top-performing models:
| Model            | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| CatBoost         | 0.941    | 0.941     | 0.941  | 0.941    |
| Random Forest    | 0.938    | 0.938     | 0.938  | 0.937    |
| XGBoost          | 0.924    | 0.924     | 0.924  | 0.924    |

CatBoost slightly outperformed Random Forest but lacked interpretability essential for medical applications.

### Gender-Based Performance
Random Forest showed better performance in predicting Alzheimer’s disease in females compared to males. Further investigation is needed to address gender-based disparities.

### Feature-Based Evaluation
Using only the top five features yielded comparable performance to using all features, suggesting that a reduced feature set is sufficient and computationally efficient.

### Stacking and Statistical Testing
- A stacking ensemble (Random Forest + CatBoost) was evaluated but did not significantly outperform individual models.
- The Friedman test validated CatBoost as the top performer, but Random Forest was selected due to its interpretability.

---

## Final Model Selection
Random Forest was chosen as the final model for its balance of performance and interpretability, making it more suitable for real-world healthcare applications.

| Metric         | CatBoost | Random Forest |
|----------------|----------|---------------|
| Accuracy       | 0.941    | 0.938         |
| Precision      | 0.941    | 0.938         |
| Recall         | 0.941    | 0.938         |
| F1-Score       | 0.941    | 0.937         |

---

## Conclusion

This project successfully developed a machine learning model for early detection of Alzheimer’s disease. Key findings include:
- Random Forest, while slightly less accurate than CatBoost, was selected due to its interpretability.
- Gender-based disparities in model performance warrant further investigation.
- A reduced feature set achieved similar performance to the full set, suggesting efficiency improvements for future implementations.

This study highlights the importance of explainable machine learning models in healthcare applications, particularly in fostering trust among clinicians and patients.

---

## Requirements

### Python Version
Ensure Python 3.10.12 or higher is installed.

### Required Libraries
Install the following libraries using:
```bash
pip install shap numpy pandas seaborn matplotlib xgboost catboost scipy scikit-learn
```

---

## How to Run

### Steps
1. **Open Jupyter Notebook**:
   - Launch Jupyter Notebook via Anaconda or using the command:
     ```bash
     jupyter notebook
     ```

2. **Navigate to Notebooks**:
   - Open the directory containing the Jupyter notebooks.

3. **Run Notebooks in Order**:
   - `Data analysis.ipynb`
   - `Feature selection.ipynb`
   - `Model Training_Evaluation.ipynb`
   - `Further analysis.ipynb`

4. **Dataset**:
   - Place the Alzheimer’s Disease dataset from Kaggle in the same directory as the notebooks.

5. **Execute Cells**:
   - Run each cell sequentially to avoid errors.

---

## Contributions

The project was a collaborative effort by Kidus Mikael and Bokhtiar Mehedy. Contributions include:
- Model training, evaluation, and selection.
- Feature selection using Spearman correlation, PCA, and RFE.
- Report writing and analysis.

Special thanks to Veselka Boeva and Shahrooz Abghari for their guidance and feedback.

---

For more details, refer to the project report: **"A Comparative Analysis of Machine Learning Models for Alzheimer’s Disease Screening."**
