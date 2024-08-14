pip install -r requirements.txt -q --no-warn-script-location --no-warn-conflicts

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split,
                                     GridSearchCV)
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, 
                             f1_score, matthews_corrcoef, 
                             precision_recall_curve, roc_auc_score, 
                             roc_curve, recall_score, precision_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from lime.lime_tabular import LimeTabularExplainer
from lime import lime_tabular
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import t
from sklearn.model_selection import permutation_test_score
from itertools import combinations
from math import factorial
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
import warnings
import tkinter as tk
from tkinter import messagebox

## **Requirements analysis and dataset upload**  
### Requirements analysis:
#### Columns needed:
def verify_columns(df, col_needed):
    colunas_diff = [col for col in colunas_necessarias if col not in df.columns]
    if colunas_diff:
        mensagem = f"Divergent columns found: {', '.join(colunas_diff)}"
        messagebox.showwarning("Divergent Columns", mensagem)
        return False
    return True

#### Type of columns and values:
  def verificar_valores(df):
    erros = []
    
    # Numerical variables verify:
    if not pd.api.types.is_numeric_dtype(df['age']):
        erros.append("The 'age' column is not just numbers")
    if not pd.api.types.is_numeric_dtype(df['adc']):
        erros.append("The 'adc' column is not just numbers")
    if not pd.api.types.is_numeric_dtype(df['diameter']):
        erros.append("The 'diameter' column is not just numbers")
    
    # 'sex' column:
    if not set(df['sex']).issubset({'M', 'F'}):
        erros.append("Column 'sex' contains values ​​other than 'M' and 'F'")
    
    # 'consistency' column:
    if not set(df['consistency']).issubset({'soft', 'não soft'}):
        erros.append("Column 'consistency' contains values ​​other than 'soft' and 'non-soft'")
    
    if erros:
        mensagem = "\n".join(erros)
        messagebox.showwarning("Divergence in values", mensagem)
        return False
    return True

#### Function to check for missing data and ask user about imputation:
def verificar_missing_values(df):
    missing_values= df.isnull().sum()
    col_with_missing = missing_values[missing_values > 0]
    
    if not col_with_missing.empty:
        resp = messagebox.askyesno("There is missing data in your data", f"The following columns have missing data:n{col_with_missing}\n\nDo you want to impute missing values?")
        if resp:
            df.fillna(df.median(numeric_only=True), inplace=True)
            df.fillna(method='ffill', inplace=True)
            messagebox.showinfo("Imputation Performed", "Missing values ​​were imputed")
        else:
            messagebox.showinfo("Imputation Cancelled", "Imputation of missing values ​​has been canceled")
    else:
        messagebox.showinfo("No missing data", "No missing data found.")

### Dataset upload:
col_needed = ['age', 'adc', 'diameter', 'sex', 'consistency']
df = pd.read_excel('df_dataframe.xlsx')

# Defining the features:
if verificar_colunas(df, colunas_necessarias):
    if verificar_valores(df):
        verificar_dados_ausentes(df)
        
        # Definindo as variáveis X e y
        X = df.drop(columns='consistency', axis=1)
        y = df['consistency']
        messagebox.showinfo("Process Completed", "The dataset was successfully verified and the X and y variables were defined")

## **Functions defined**  

def specificity_sensitivity(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return specificity, sensitivity

def corrected_std(differences, n_train, n_test):
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std

def compute_corrected_ttest(differences, df, n_train, n_test):
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val

def bootstrap_metrics(all_y, all_probs, all_preds, n_iterations=1000, alpha=0.95):
    n_size = len(all_y)
    metrics = {
        'roc_auc': [],
        'pr_auc': [],
        'mcc': [],
        'f1': [],
        'sensitivity': [],
        'specificity': [],
    }
    
    for i in range(n_iterations):
        indices = resample(range(n_size), n_samples=n_size, random_state=i)
        y_sample = all_y[indices]
        probs_sample = all_probs[indices]
        preds_sample = all_preds[indices]

        metrics['roc_auc'].append(roc_auc_score(y_sample, probs_sample))
        metrics['pr_auc'].append(average_precision_score(y_sample, probs_sample))
        metrics['mcc'].append(matthews_corrcoef(y_sample, preds_sample))
        metrics['f1'].append(f1_score(y_sample, preds_sample))
        conf_matrix = confusion_matrix(y_sample, preds_sample)
        specificity, sensitivity = specificity_sensitivity(conf_matrix)
        metrics['specificity'].append(specificity)
        metrics['sensitivity'].append(sensitivity)

    def confidence_interval(data):
        lower = np.percentile(data, ((1.0 - alpha) / 2.0) * 100)
        upper = np.percentile(data, (alpha + ((1.0 - alpha) / 2.0)) * 100)
        return lower, upper

    ci_results = {metric: confidence_interval(metrics[metric]) for metric in metrics}
    
    model_metrics__with_ci = pd.DataFrame({
        'Metric': ['ROC AUC', 'PR AUC', 'Sensitivity', 'Specificity', 'F1 Score', 'MCC'],
        'Value (95% CI)': [
            f"{np.mean(metrics['roc_auc']):.3f} ({ci_results['roc_auc'][0]:.3f}-{ci_results['roc_auc'][1]:.3f})",
            f"{np.mean(metrics['pr_auc']):.3f} ({ci_results['pr_auc'][0]:.3f}-{ci_results['pr_auc'][1]:.3f})",
            f"{np.mean(metrics['sensitivity']):.3f} ({ci_results['sensitivity'][0]:.3f}-{ci_results['sensitivity'][1]:.3f})",
            f"{np.mean(metrics['specificity']):.3f} ({ci_results['specificity'][0]:.3f}-{ci_results['specificity'][1]:.3f})",
            f"{np.mean(metrics['f1']):.2f} ({ci_results['f1'][0]:.2f}-{ci_results['f1'][1]:.2f})",
            f"{np.mean(metrics['mcc']):.2f} ({ci_results['mcc'][0]:.2f}-{ci_results['mcc'][1]:.2f})"
        ]
    })
    return model_metrics__with_ci

## **Model hyperparameterization and testing | Using Leave One Out**  
#### ++ **SVM model  

svm_params = {
    'classifier__C': [1, 10, 100, 1000], 
    'classifier__gamma': [10, 1 ,0.1, 0.001],
    'classifier__kernel': ['linear', 'rbf', 'sigmoid'],
}

svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(probability=True, random_state=42, class_weight='balanced'))
])

kf = LeaveOneOut()
all_y = []
all_probs_svm = []
all_preds_svm = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    grid_search_svm = GridSearchCV(svm_clf, svm_params, cv=10, scoring='roc_auc_ovr', n_jobs=-1)
    grid_search_svm.fit(X_train, y_train)
    best_model_svm = grid_search_svm.best_estimator_
    y_prob = best_model_svm.predict_proba(X_test)[:, 1]
    y_pred = best_model_svm.predict(X_test)
    
    all_y.extend(y_test.tolist())
    all_probs_svm.extend(y_prob.tolist())
    all_preds_svm.extend(y_pred.tolist())

all_y = np.array(all_y)
all_probs_svm = np.array(all_probs_svm)
all_preds_svm = np.array(all_preds_svm)

#### **SVM model | Confision Matrix before threshold adjustment, ROC and PR curves**

roc_auc_svm = roc_auc_score(all_y, all_probs_svm)
fpr_svm, tpr_svm, thresholds_roc_svm = roc_curve(all_y, all_probs_svm)
precision_svm, recall_svm, thresholds_pr_svm = precision_recall_curve(all_y, all_probs_svm)
pr_auc_svm = average_precision_score(all_y, all_probs_svm)
mcc_svm = matthews_corrcoef(all_y, all_preds_svm)
f1_svm = f1_score(all_y, all_preds_svm)
conf_matrix_svm = confusion_matrix(all_y, all_preds_svm)

threshold = 0.19
roc_idx_svm = np.argmin(np.abs(thresholds_roc_svm - threshold))
pr_idx_svm = np.argmin(np.abs(thresholds_pr_svm - threshold))

print(f'ROC AUC: {roc_auc_svm}')
print(f'PR AUC: {pr_auc_svm}')
print(f'MCC: {mcc_svm}')
print(f'F1 Score: {f1_svm}')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

disp_svm = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_svm)
disp_svm.plot(ax=axes[0], cmap=plt.cm.Blues)
axes[0].set_title('Confusion Matrix \nBefore Adjusting Thresholds for SVM\n', fontsize=12)

axes[1].plot(fpr_svm, tpr_svm, lw=2, label='ROC (AUC = %0.2f)' % (roc_auc_svm))
axes[1].plot(fpr_svm[roc_idx_svm], tpr_svm[roc_idx_svm], 'rX', markersize=10, label='Best threshold = 0.19')
axes[1].set_xlim([-0.05, 1.05])
axes[1].set_ylim([-0.05, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve for SVM\n', fontsize=12)
axes[1].legend(loc='best', fontsize=9)

axes[2].plot(recall_svm, precision_svm, lw=2, label='AP (AUC = %0.2f)' % (pr_auc_svm))
axes[2].plot(recall_svm[pr_idx_svm], precision_svm[pr_idx_svm], 'rX', markersize=10, label='Best threshold = 0.19')
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')
axes[2].set_title('Precision-Recall Curve for SVM\n', fontsize=12)
axes[2].legend(loc='best', fontsize=9)

plt.tight_layout()
plt.savefig('svm_cf_matrix_bf_roc_pr_curve.png', dpi=600, bbox_inches='tight')
plt.show()

#### **SVM model | Adjustment threshold**

thresholds = np.arange(0.0, 1.0, 0.01)
precisions_svm_list = []
f1_svm_list = []
recalls_svm_list = []
specificity_svm_list = []

    all_preds_svm = (all_probs_svm >= threshold).astype(int)
    recall = recall_score(all_y, all_preds_svm)
    precision = precision_score(all_y, all_preds_svm, zero_division = 0)
    f1 = f1_score(all_y, all_preds_svm)
    conf_matrix = confusion_matrix(all_y, all_preds_svm)
    specificity, _ = specificity_sensitivity(conf_matrix)

    precisions_svm_list.append(precision)
    f1_svm_list.append(f1)
    recalls_svm_list.append(recall)
    specificity_svm_list.append(specificity)
    
best_threshold_svm = 0.19
all_preds_best_threshold_svm = (all_probs_svm >= best_threshold_svm).astype(int)

roc_auc_svm_tr = roc_auc_score(all_y, all_probs_svm)
fpr_svm_tr, tpr_svm_tr, _ = roc_curve(all_y, all_probs_svm)
precision_svm_tr, recall_svm_tr, _ = precision_recall_curve(all_y, all_probs_svm)
pr_auc_svm_tr = average_precision_score(all_y, all_probs_svm)
mcc_svm_tr = matthews_corrcoef(all_y, all_preds_best_threshold_svm)
f1_svm_tr = f1_score(all_y, all_preds_best_threshold_svm)
conf_matrix_svm_tr = confusion_matrix(all_y, all_preds_best_threshold_svm)
specificity_svm_tr, sensitivity_svm_tr = specificity_sensitivity(conf_matrix_svm_tr)

print(f'ROC AUC: {roc_auc_svm_tr}')
print(f'PR AUC: {pr_auc_svm_tr}')
print(f'Sensitivity: {(sensitivity_svm_tr)}')
print(f'Specificity: {specificity_svm_tr}')
print(f'MCC: {mcc_svm_tr}')
print(f'F1 Score: {f1_svm_tr}')

fig, axes = plt.subplots(1, 2, figsize=(20, 6))

axes[0].plot(thresholds, f1_svm_list, label='F1 Score', color='darkred', linestyle='dotted', lw = 2)
axes[0].plot(thresholds, precisions_svm_list, label='Precision', color='darkblue',lw = 2)
axes[0].plot(thresholds, recalls_svm_list, label='Sensitivity', color='darkgreen', linestyle=(2, (10, 5)), lw = 2)
axes[0].plot(thresholds, specificity_svm_list, label='Specificity', color='darkorange', linestyle=(0, (5, 2, 2)), lw = 2)
axes[0].set_xlabel('Threshold')
axes[0].set_ylabel('Score')
axes[0].set_title('F1 Score, Precision, Sensitivity and Specificity vs Threshold for SVM\n', fontsize=12)
axes[0].set_xticks(np.arange(0, 1.05, 0.05))
axes[0].legend(loc='best', fontsize=10, ncol =2)

conf_matrix_svm_tr = confusion_matrix(all_y, all_preds_best_threshold_svm)
disp_svm_tr = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_svm_tr)
disp_svm_tr.plot(ax=axes[1], cmap=plt.cm.Blues)
axes[1].set_title('Confusion Matrix After Adjusting Thresholds for SVM\n', fontsize=12)
axes[1].text(0.5, 1.02, r'Threshold $= 0.19$', fontsize=10, ha='center', transform=axes[1].transAxes)

plt.subplots_adjust(wspace=-0.18)
plt.savefig('svm_cf_matrix_af_threshold.png', dpi=600, bbox_inches='tight')
plt.show()

#### **SVM model | Bootstrap metrics **

# No change in threshold :
svm_metrics_ci = bootstrap_metrics(all_y, all_probs_svm, all_preds_svm, n_iterations=1000, alpha=0.95)
fpr_svm, tpr_svm, _ = roc_curve(all_y, all_probs_svm)
precision_svm, recall_svm, _ = precision_recall_curve(all_y, all_probs_svm)
display(svm_metrics_ci)

# With new threshold :
threshold_svm = 0.19
all_preds_svm = (all_probs_svm >= threshold_svm).astype(int)

svm_metrics_ci_tr = bootstrap_metrics(all_y, all_probs_svm, all_preds_svm, n_iterations=1000, alpha=0.95)
fpr_svm_tr, tpr_svm_tr, _ = roc_curve(all_y, all_probs_svm)
precision_svm_tr, recall_svm_tr, _ = precision_recall_curve(all_y, all_probs_svm)
display(svm_metrics_ci_tr)
