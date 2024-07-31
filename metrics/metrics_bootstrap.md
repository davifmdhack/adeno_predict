## **Confusion matrix**
<div align="center">

|                      | **Predicted Positive** | **Predicted Negative** |
|:--------------------:|:----------------------:|:----------------------:|
| **Actual Positive**  | True Positive (TP)     | False Negative (FN)    |
| **Actual Negative**  | False Positive (FP)    | True Negative (TN)     |

</div>

</br>

## **The mathematical expression**

$$
(1) \ \ \ \ \ \text{AUC} = \displaystyle\int_{0}^{1} TP(FP) \, dFP \ \ \ \  \xrightarrow{\text{or}}  \ \ \ \ \text{AUC} = \displaystyle\int_{0}^{1} TP(FP^{-1}(x))  \, \ dx
$$

<br>


$$
(2) \ \ \ \ \  \text{AP} = \sum_{n=1}^N (R_n - R_{n-1})\cdot P_n
$$

Where $N$ is the number of thresholds at which precision and recall are evaluated. $P_n$ is the precision at the n-th threshold, $R_n$ is the recall at the n-th threshold and $R_{n-1}$ is the recall at the previous threshold. 

<br>

<br>


$$
(3) \ \ \ \ \  \text{Sensitivity} = \dfrac{TP}{(TP + FN)}
$$

<br>

$$
(4) \ \ \ \ \  \text{Specificity} = \dfrac{TN}{(TN + FP)}
$$

<br>

$$
(5) \ \ \ \ \  \text{F1 Score} = \dfrac{TP}{\left[TP + \dfrac{1}{2} \cdot (FP + FN)\right]}
$$

<br>

$$
(6) \ \ \ \ \  \text{MCC Score} = \dfrac{(TP \cdot TN - FP \cdot FN)}{\sqrt{(TP + FP) \cdot (TP + FN) \cdot (TN + FP) \cdot (TN + FN)}}
$$

1 - Area Under Curve (AUC)  
2 - Average precision-recall (AP)  
3 - Sensitivity (Recall)  
4 - Specificity  
5 - F1 Score  
6 - Matthews correlation coefficient (MCC) Score
<p style="text-align: justify;">

## **Bootstrap implementation**

The bootstrap method is a powerful statistical tool used to estimate the distribution of a statistic. Bootstrap method can be particularly useful for estimating the confidence intervals of a model's 
performance metric without needing to make any assumptions about the distribution of the underlying data.
Considering $D$ original dataset containing $d$ data points, i.e, $D = d_1, d_2, ..., d_{N}$, generating $B$ bootstrap samples $(B^\*_1, B^\*_2, ..., B^\*_n)$,  where each $B^\*_i$ is a set of $N$ data points extracted by replacement from $D$. For each $B^\*_i$, its performance metric is $θ\^*_i$. TTherefore, the confidence interval is calculated from sort performance metrics in 1:

1. Sort performance metrics in asceding order { $\theta^\*_1$, $\theta^\*_2$, ..., $\theta^\*_B$ }
2. For a $(1- \alpha)\cdot 100 \%$ confidence, the confidence interval is given by:

$$
\left[\theta^\*_{\left(\frac{\alpha}{2}\right)}\,  \theta^\*\_{\left(1 - \frac{\alpha}{2}\right)}\right]
$$ 
