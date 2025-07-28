# FINAL-EXAM
# =======================================
# Diabetic Patient Readmission Prediction
# =======================================

# 1. Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load dataset
df = pd.read_csv(r"C:\Users\Pavani\Documents\BOOTCAMP\SCMA\FINAL EXAM\diabetic_data.csv")

# 3. Create binary target variable: readmitted within 30 days (<30)
df['readmitted_flag'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# 4. Select relevant features
selected_vars = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses", "insulin", "diabetesMed", "readmitted_flag"
]
df = df[selected_vars]

# 5. Handle missing/invalid values
df = df[(df["race"] != "?") & (df["gender"] != "Unknown/Invalid")]

# 6. Encode categorical variables
for col in ["race", "gender", "age", "insulin", "diabetesMed"]:
    df[col] = LabelEncoder().fit_transform(df[col])

# 7. Split into train and test sets
X = df.drop("readmitted_flag", axis=1)
y = df["readmitted_flag"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 8. Train Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
y_proba_logreg = logreg.predict_proba(X_test)[:, 1]

# 9. Train Decision Tree model
dtree = DecisionTreeClassifier(max_depth=5, random_state=42)
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)
y_proba_dtree = dtree.predict_proba(X_test)[:, 1]

# 10. Define a function to print evaluation metrics
def print_metrics(y_true, y_pred, y_proba, model_name):
    print(f"\n=== {model_name} Metrics ===")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-score :", f1_score(y_true, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_true, y_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

# 11. Evaluate models
print_metrics(y_test, y_pred_logreg, y_proba_logreg, "Logistic Regression")
print_metrics(y_test, y_pred_dtree, y_proba_dtree, "Decision Tree")

# 12. Plot ROC curves
fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_logreg)
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_proba_dtree)

plt.figure(figsize=(7,5))
plt.plot(fpr_log, tpr_log, label='Logistic Regression (AUC: %.2f)' % roc_auc_score(y_test, y_proba_logreg))
plt.plot(fpr_tree, tpr_tree, label='Decision Tree (AUC: %.2f)' % roc_auc_score(y_test, y_proba_dtree))
plt.plot([0,1], [0,1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 13. Feature Importance (Decision Tree)
importances = pd.Series(dtree.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)
plt.figure(figsize=(7,7))
importances.plot(kind='barh')
plt.title("Feature Importances (Decision Tree)")
plt.show()

