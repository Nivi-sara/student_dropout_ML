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
- **Source:** UCI Machine Learning Repository (dataset id 697)
-  **Dataset URL:** https://archive.ics.uci.edu/dataset/697
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
GitHub Repository: https://github.com/Nivi-sara/student_dropout_ML

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
| Logistic Regression | Best on all six metrics (Accuracy 0.768, AUC 0.896, MCC 0.615). Train and test accuracy are nearly equal (0.773 vs 0.768), so there is no overfitting. Struggles on the Enrolled class (recall 0.33). |
| Decision Tree | Accuracy 0.746, MCC 0.578. Depth limit of 6 keeps overfitting low (train 0.783 vs test 0.746). Second-semester approved units account for about 68% of feature importance. |
| KNN | Accuracy 0.695, MCC 0.487. Overfits heavily: 100% train accuracy vs 69.5% test. Distances are less reliable across 36 features, including category codes. |
| Naive Bayes | Weakest: Accuracy 0.659, AUC 0.812, MCC 0.428. Its feature-independence assumption fails because semester features are strongly correlated. It also underfits (train accuracy 0.675). |
| Support Vector Machine | Close second: Accuracy 0.758, AUC 0.887, MCC 0.599. Mild overfitting (train 0.830 vs test 0.758). Slowest to train. |
| **Overall winner** | **Logistic Regression.** It has the highest Accuracy, AUC, Precision, Recall, F1 and MCC, with no overfitting. All models find the Enrolled class hardest, because it is the smallest class (about 18% of students). |

## 5.Evaluation Metrics
All metrics are computed on the 20% hold-out test set (`test_data.csv`). Because the target has three classes, Precision, Recall and F1 use the weighted average, and AUC uses one-vs-rest (weighted).

| Metric | Meaning |
|---|---|
| Accuracy | Share of all students whose outcome is predicted correctly |
| AUC | How well the model ranks the correct class above the others (1.0 is perfect, 0.5 is random) |
| Precision | Of the students predicted as a class, how many truly belong to it |
| Recall | Of the students who truly belong to a class, how many the model finds |
| F1 Score | Harmonic mean of Precision and Recall |
| MCC | Matthews Correlation Coefficient: a balanced score that stays reliable when classes are imbalanced (1 is perfect, 0 is random) |
```
python train.py data.csv
```.
