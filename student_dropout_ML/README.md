# Student Dropout & Academic Success – Classification and Streamlit Deployment

M.Tech (AIML) · Machine Learning · Assignment 2

## 1. Problem Statement
Higher-education institutions lose students to dropout, which hurts both the students and the
institution. The goal is to predict, from information known at enrolment and after the first two
semesters (demographics, socio-economic factors, academic performance), whether a student will
**Dropout**, remain **Enrolled**, or **Graduate**. Early identification lets the institution offer
targeted support. This is a **3-class classification** problem; five classical ML classifiers are
trained on the same data, compared with six metrics, and served through a Streamlit app.

## 2. Dataset Description
- **Name:** Predict Students' Dropout and Academic Success
- **Source:** UCI Machine Learning Repository (dataset id 697) – https://archive.ics.uci.edu/dataset/697
- **Instances:** 4,424  **Features:** 36 (numeric/categorical-coded)  **Target:** `Target`
- **Classes:** Dropout, Enrolled, Graduate (imbalanced – Graduate is the majority)
- **Feature groups:** demographic (age, gender, marital status, nationality), socio-economic
  (parents' qualification/occupation, scholarship, debtor, tuition fees up to date),
  application details (course, application mode/order, previous qualification), and academic
  performance in 1st and 2nd semester (units enrolled/approved, grades, evaluations), plus
  macro-economic indicators (unemployment rate, inflation, GDP).
- **Preprocessing:** column-name cleaning, label-encoding of the target, `StandardScaler`
  (inside each model pipeline), stratified 80/20 train/test split (`random_state=42`).
  `test_data.csv` is the 20 % hold-out set (with the `Target` column) used by the app.

## 3. GitHub Repository Link
https://github.com/Nivi-sara/student_dropout_ML

Live app: https://nivetha-e-studentdropoutml-cgvct4jqdsrvuo8u7qexpt.streamlit.app/



Run locally: `pip install -r requirements.txt` → `python train.py data.csv` → `streamlit run app.py`

## 4. Models Used
Models: Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, SVM (RBF).
Precision, Recall and F1 are **weighted averages**; AUC is **one-vs-rest (weighted)**.

### Comparison table
| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.7684 | 0.8961 |      0.75   |   0.7684 | 0.7531 | 0.615  |
| Decision Tree          |     0.7458 | 0.8687 |      0.7338 |   0.7458 | 0.735  | 0.578  |
| KNN                    |     0.6949 | 0.8327 |      0.6722 |   0.6949 | 0.667  | 0.4874 |
| Naive Bayes            |     0.6588 | 0.8118 |      0.6334 |   0.6588 | 0.6417 | 0.4279 |
| Support Vector Machine |     0.7582 | 0.8865 |      0.7465 |   0.7582 | 0.7454 | 0.5986 |

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <write after seeing results> |
| Decision Tree | <write after seeing results> |
| KNN | <write after seeing results> |
| Naive Bayes | <write after seeing results> |
| Support Vector Machine | <write after seeing results> |
| **Overall winner** | <model name and why (MCC / F1 / AUC)> |

Hints for the observations: look at which class is hardest (see the "Enrolled" row in the
classification report in the app), compare MCC vs Accuracy because of class imbalance, comment on
Decision Tree overfitting vs depth limit, Naive Bayes' independence assumption on correlated
semester features, and KNN/SVM's dependence on feature scaling.

## 5. Streamlit App Features
CSV upload (test data) · model-selection dropdown · six evaluation metrics · confusion matrix ·
classification report · all-model comparison table (highlighting the best value per metric) ·
CSV download of the comparison. Models are loaded from `model/`; nothing is retrained.
