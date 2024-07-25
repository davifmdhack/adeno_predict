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
(2) \ \ \ \ \  \text{Sensitivity} = \dfrac{TP}{(TP + FN)}
$$

<br>

$$
(3) \ \ \ \ \  \text{Specificity} = \dfrac{TN}{(TN + FP)}
$$

<br>

$$
(4) \ \ \ \ \  \text{F1 Score} = \dfrac{TP}{\left[TP + \dfrac{1}{2} \cdot (FP + FN)\right]}
$$

<br>

$$
(5) \ \ \ \ \  \text{MCC Score} = \dfrac{(TP \cdot TN - FP \cdot FN)}{\sqrt{(TP + FP) \cdot (TP + FN) \cdot (TN + FP) \cdot (TN + FN)}}
$$

1 - Area Under Curve (AUC)  
2 - Sensitivity (Recall)  
3 - Specificity  
4 - F1 Score  
5 - Matthews correlation coefficient (MCC) Score
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
