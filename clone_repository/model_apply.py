import subprocess
import sys

def check_and_install(package):
    try:
        __import__(package)
    except ImportError:
        print(f"{package} não está instalado. Instalando agora...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

check_and_install('pandas')
check_and_install('sklearn')
check_and_install('joblib')
check_and_install('matplotlib')

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (auc, confusion_matrix, f1_score, matthews_corrcoef, 
                             precision_recall_curve, recall_score, roc_curve,average_precision_score,
                             ConfusionMatrixDisplay)
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

df = pd.read_excel('df_pituitary.xlsx')
replace_map_outcome = {
    'soft': 0,
    'non-soft': 1
}

df['consistency'] = df['consistency'].map(replace_map_outcome).fillna(df['consistency'])
encoder = OneHotEncoder(drop = 'first')
encoded_columns = encoder.fit_transform(df[['sex']]).toarray() 
encoded_sex = pd.DataFrame(encoded_columns, 
                                  columns=encoder.get_feature_names_out(['sex']), 
                                  index=df.index)

df_encoded = pd.concat([df.drop(['sex'], 
                                axis=1),
                                encoded_sex], 
                                axis=1)

X_new = df_encoded.drop(columns='consistency', axis = 1)
y_new = df_encoded['consistency']
X_new = X_new[['age', 'sex_M', 'diameter', 'adc']]

best_model_svm = joblib.load('best_model_svm.pkl')
scaler = best_model_svm.named_steps['scaler']
X_new_scaled = scaler.transform(X_new)

y_prob = best_model_svm.predict_proba(X_new)[:, 1]
y_pred = best_model_svm.predict(X_new)

def specificity_sensitivity(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return specificity, sensitivity

fpr, tpr, _ = roc_curve(y_new, y_prob)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_new, y_prob)
pr_auc = average_precision_score(y_new, y_prob)
sensitivity = recall_score(y_new, y_pred)
conf_matrix = confusion_matrix(y_new, y_pred)
specificity, _ = specificity_sensitivity(conf_matrix)
mcc = matthews_corrcoef(y_new, y_pred)
f1 = f1_score(y_new, y_pred)

metrics_data = {
    'ROC AUC': [roc_auc],
    'PR AUC': [pr_auc],
    'Sensitivity': [sensitivity],
    'Specificity': [specificity],
    'F1 Score': [f1],
    'MCC': [mcc]
}

metrics_df = pd.DataFrame(metrics_data)


fig = plt.figure(figsize=(15, 15))

# Intro
fig.suptitle('REPORT - Adeno Predict model in domestic dataset', fontsize=16, fontweight='bold', y=0.98)
plt.figtext(0.5, 0.89, 'Created by: \n\nDavi Ferreira, MD., MSc. E-Mail: davi.ferreira.soares@gmail.com\nFernanda Veloso MD., PhD. candidate. E-Mail: fernandavelosop@gmail.com', fontsize=12, ha='center')

# Title 1
ax_title_table = fig.add_subplot(4, 1, 1)
ax_title_table.axis('off')
ax_title_table.text(0.5, 0.8, 'Table: Metrics Resume', fontsize=14, ha='center', fontweight='bold')

# Table 1
ax_table = fig.add_subplot(4, 1, 1)
ax_table.axis('tight')
ax_table.axis('off')
table = ax_table.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')
table.scale(1, 3)

# Title 2
ax_title_graphs = fig.add_subplot(4, 1, 1)
ax_title_graphs.axis('off')
ax_title_graphs.text(0.5, 0.1, 'Confusion Matrix, ROC Curve and PR-Curve', fontsize=14, ha='center', fontweight='bold')

# Figure 1
ax1 = fig.add_subplot(4, 3, 5)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp_svm.plot(ax=ax1, cmap=plt.cm.Blues)
ax1.set_title('Confusion Matrix\n', fontsize=12)

# Figure 2
ax2 = fig.add_subplot(2, 2, 3)
ax2.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax2.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve', fontsize=12)
ax2.legend(loc='lower right')

# Figure 3
ax3 = fig.add_subplot(2, 2, 4)
ax3.plot(recall, precision, lw=2, label=f'AP (AUC = {pr_auc:.2f})')
ax3.plot([1, 0], [0, 1], color='gray', lw=1, linestyle='--')
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision-Recall Curve', fontsize=12)
ax3.legend(loc='lower left')


with PdfPages('report_metrics_and_plot.pdf') as pdf:
    pdf.savefig(fig)
    plt.close(fig)
