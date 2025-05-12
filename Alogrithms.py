import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("Heart_Prediction_Quantum_Dataset.csv")

# Check for missing values
print("Missing Values:")
print(data.isnull().sum())

# Check for duplicates
print("\nDuplicate Rows:", data.duplicated().sum())

# Describe each variable
print("\nDataset Summary:")
print(data.describe())

# Distribution plots
sns.pairplot(data, hue='HeartDisease')
plt.title("Data Distributions")
plt.show()

# Separate features and target variable
X = data.drop(columns=["HeartDisease"])
y = data["HeartDisease"]

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# K-fold cross-validation (k=5)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg_scores = cross_val_score(log_reg, X_train, y_train, cv=kfold)
log_reg_pred = log_reg.predict(X_test)

# Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
decision_tree_scores = cross_val_score(decision_tree, X_train, y_train, cv=kfold)
decision_tree_pred = decision_tree.predict(X_test)

# Metrics for Logistic Regression
log_reg_acc = accuracy_score(y_test, log_reg_pred)
log_reg_prec = precision_score(y_test, log_reg_pred)
log_reg_rec = recall_score(y_test, log_reg_pred)
log_reg_f1 = f1_score(y_test, log_reg_pred)
log_reg_conf_matrix = confusion_matrix(y_test, log_reg_pred)

# Metrics for Decision Tree
decision_tree_acc = accuracy_score(y_test, decision_tree_pred)
decision_tree_prec = precision_score(y_test, decision_tree_pred)
decision_tree_rec = recall_score(y_test, decision_tree_pred)
decision_tree_f1 = f1_score(y_test, decision_tree_pred)
decision_tree_conf_matrix = confusion_matrix(y_test, decision_tree_pred)

# Feature Importance (Decision Tree Example)
feature_importances = decision_tree.feature_importances_
plt.bar(X.columns, feature_importances)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

# Results
print("\nLogistic Regression Metrics:")
print(f"Accuracy: {log_reg_acc}, Precision: {log_reg_prec}, Recall: {log_reg_rec}, F1: {log_reg_f1}")
print("Confusion Matrix:\n", log_reg_conf_matrix)

print("\nDecision Tree Metrics:")
print(f"Accuracy: {decision_tree_acc}, Precision: {decision_tree_prec}, Recall: {decision_tree_rec}, F1: {decision_tree_f1}")
print("Confusion Matrix:\n", decision_tree_conf_matrix)

# ROC Curve & AUC
# Logistic Regression
log_reg_probs = log_reg.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, log_reg_probs)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Decision Tree
decision_tree_probs = decision_tree.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, decision_tree_probs)
roc_auc_dt = auc(fpr_dt, tpr_dt)

# Display ROC Curve
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={roc_auc_lr:.2f})', linestyle='--')
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC={roc_auc_dt:.2f})', linestyle='-')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="best")
plt.show()
