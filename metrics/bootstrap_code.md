## **Imputation process**

## **Libraries**
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, 
                                 precision_recall_curve, roc_auc_score, roc_curve, recall_score, precision_score)
    from sklearn.utils import resample

## **Previously defined variables**
 `specificity_sensitivity` = function to determine specificity
 
    def specificity_sensitivity(conf_matrix):
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        return specificity, sensitivity
  
 ## **Implementation function**

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
