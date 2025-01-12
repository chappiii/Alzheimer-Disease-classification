README: File structure & how to Run the Code

This document provides instructions to run the Jupyter Notebook file and ensure proper execution of the code.

---

## Project Notebooks

The project is divided into **four Jupyter notebooks**, each focusing on a specific aspect of the machine learning pipeline:

1. **Data analysis.ipynb**  
   - This notebook contains the **data analysis process**, including data exploration, visualization, and preprocessing steps applied to understand the dataset.

2. **Feature selection.ipynb**  
   - This notebook documents the **feature selection process**, detailing the techniques and methods used to identify the most relevant features for Alzheimer's disease prediction.

3. **Model Training_Evaluation.ipynb**  
   - This notebook contains **experiments on multiple machine learning models**, including their training, validation, hyperparameter tuning, and final model selection.

4. **Further analysis.ipynb**  
   - This notebook includes **additional experiments**, such as **gender-based model evaluation** and **feature-based model performance testing**, providing deeper insights into the dataset's behavior.

Each notebook should be executed sequentially to ensure the correct flow of the pipeline, leading from **data preprocessing to model selection and further analysis**.
"""

---

---

## Requirements

### 1. Python Version
Ensure that Python 3.10.12 or higher is installed on your system.

### 2. Required Libraries
The following Python libraries must be installed before running the code:

- shap
- numpy
- pandas
- seaborn
- matplotlib
- xgboost
- catboost
- scipy
- sklearn (scikit-learn)

To install all required libraries, run the following command in the terminal:

```
pip install shap numpy pandas seaborn matplotlib xgboost catboost scipy scikit-learn
```

---

## Dataset
Ensure that the Alzheimer’s Disease dataset from Kaggle is placed in the same directory as the notebook file. If needed, update the dataset file path in the code accordingly.

---

## Steps to Run

### 1. Open Jupyter Notebook
- Open Anaconda Navigator (if installed) and launch Jupyter Notebook, or
- Run the following command in the terminal or command prompt:

```
jupyter notebook
```

### 2. Navigate to the Notebook Directory
- Locate and open the notebook file from the directory where it is saved.

### 3. Ensure the Dataset is in the Correct Location
- Verify that the dataset file is placed in the same directory as the notebook file, or modify the dataset path in the code if necessary.

### 4. Run the Notebook Cells Sequentially
- Execute each cell in the correct order to avoid errors.

---

## Important Notes
- The notebook utilizes machine learning models such as Random Forest, CatBoost, XGBoost, Decision Tree, AdaBoost, and SVM, requiring sufficient computational resources for optimal performance.
- Ensure the dataset is correctly formatted and follows the expected structure.
- The model evaluation includes cross-validation, hyperparameter tuning, stacking models, and statistical testing (Friedman test) to assess performance differences.
- If any library installation issues occur, refer to the official documentation of the respective libraries for troubleshooting.

---