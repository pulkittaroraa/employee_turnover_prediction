
# Employee Turnover Prediction

This project predicts employee turnover (whether employees will leave the company or not) using machine learning models. The primary goal of this project is to analyze employee data, extract insights, and build predictive models that can help a company retain its employees.

## Project Overview
Employee turnover prediction is a critical task for organizations aiming to retain talent. This project leverages Python for exploratory data analysis, data visualization, and machine learning modeling. The dataset used includes key employee attributes such as age, education, experience, and more.

## Dataset
The dataset contains the following features:

- **Age**: Employee age
- **Education**: Level of education
- **ExperienceInCurrentDomain**: Years of experience in the current domain
- **JoiningYear**: Year of joining the company
- **City**: City of the employee
- **PaymentTier**: Salary tier (1 = lowest, 3 = highest)
- **EverBenched**: Whether the employee was ever benched (Yes/No)
- **LeaveOrNot**: Target variable (1 = left the company, 0 = stayed)

## Project Workflow

### 1. Data Exploration
The dataset was analyzed to understand its structure and identify any missing or duplicated values:
- Summary statistics using `describe()` and `info()`
- Count of missing values and duplicates
- Basic visualizations:
  - Histograms (e.g., distribution of age)
  - Count plots (e.g., gender distribution)
  - Boxplots (e.g., experience across genders)
  - Scatter plots (e.g., age vs. experience)

### 2. Data Preprocessing
Steps included:
- Removing duplicate rows
- Label encoding categorical variables: `Education`, `Gender`, and `EverBenched`
- Feature scaling using `StandardScaler`
- Splitting the dataset into training (80%) and testing (20%) sets

### 3. Modeling and Evaluation
The following machine learning models were implemented:

#### Logistic Regression
- A baseline model with no hyperparameter tuning
- Accuracy score: 0.6835443037974683

#### K-Nearest Neighbors (KNN)
- Hyperparameter tuning using GridSearchCV:
  - `n_neighbors`: [3, 5, 7, 9, 11]
  - `weights`: ["uniform", "distance"]
  - `algorithm`: ["auto", "ball_tree", "kd_tree", "brute"]
- Best parameters: {'algorithm': 'brute', 'n_neighbors': 5, 'weights': 'distance'}
- Accuracy score: 0.6781193490054249

#### Support Vector Machine (SVM)
- Hyperparameter tuning using GridSearchCV:
  - `C`: [0.01, 0.1, 0.5, 1]
  - `kernel`: ["linear", "rbf", "poly"]
- Best parameters: {'C': 0.5, 'kernel': 'rbf'}
- Accuracy score: 0.6980108499095841

#### Decision Tree Classifier
- Hyperparameter tuning using GridSearchCV:
  - `criterion`: ["gini", "entropy"]
  - `splitter`: ["best", "random"]
  - `max_depth`: [None, 10, 20, 30, 40, 50]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]
- Best parameters: {'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 4, 'min_samples_split': 10, 'splitter': 'random'}
- Accuracy score: 0.6907775768535263

#### Random Forest Classifier
- Hyperparameter tuning using GridSearchCV:
  - `n_estimators`: [32, 64, 128, 256]
  - `max_features`: [2, 3, 4]
  - `bootstrap`: [True, False]
  - `oob_score`: [True, False]
- Best parameters: {'bootstrap': True, 'max_features': 4, 'n_estimators': 32, 'oob_score': False}
- Accuracy score: 0.6835443037974683

## Tools and Libraries Used
- **Python**: Programming language
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning modeling

## How to Run the Project
1. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Place the dataset (Employee.csv) in the same directory as the script.
3. Run the script (`employee_attrition.ipynb`) to execute data analysis and modeling.
4. Review the outputs to identify the best-performing model.

## Conclusion
This project provides a comprehensive approach to predicting employee turnover, combining effective data analysis and machine learning techniques. It demonstrates the use of data-driven decision-making to address real-world business challenges.
