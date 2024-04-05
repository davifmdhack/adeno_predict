<div align="center">

|                      | **Predicted Positive** | **Predicted Negative** |
|:--------------------:|:----------------------:|:----------------------:|
| **Actual Positive**  | True Positive (TP)     | False Negative (FN)    |
| **Actual Negative**  | False Positive (FP)    | True Negative (TN)     |

</div>

### **The mathematical expression:**

<br>

$$
(1) \ \ \ \ \ \text{AUC} = \displaystyle\int_{0}^{1} TP(FP) \, dFP \ \ \ \  \xrightarrow{\text{or}}  \ \ \ \ \text{AUC} = \displaystyle\int_{0}^{1} TP(FP^{-1}(x))  \, \ dx
$$

<br>

$$
(2) \ \ \ \ \  \text{Accuracy} = \dfrac{(TP + TN)}{(TP + TN + FP + FN)}
$$

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

<p style="text-align: justify;">

<br>

## **Bootstrap implementation**
The bootstrap method is a powerful statistical tool used to estimate the distribution of a statistic. Bootstrap method can be particularly useful for estimating the confidence intervals of a model's 
performance metric without needing to make any assumptions about the distribution of the underlying data. The technique is divided into: 1. Resampling, 2. Statistic or Model Evaltuation 3. Confidence 
Interval Calculation. For each steop, we describe brefly.

</p>
1 Resampling:

Let $(D)$ be the original dataset containing $(N)$ data points, i.e., $D = d_1, d_2, ..., d_{N}$. Generate $(B)$ bootstrap samples $(D^\*_1, D^\*_2, ..., D^\*_B)$, where each $(D^\*_i)$ is a set of $(N)$ 
data points drawn with replacement from $(D)$

2 Statistic or Model Evaltuation

For each bootstrap sample $(D^\*_i)$, train your model and evaluate its performance metric $(\theta^\*_i)$

3 Confidence Interval Calculation

Sort the bootstrap performance metrics $(\theta^\*_1, \theta^\*_2, ..., \theta^\*_B)$ in ascending order. 

For a $(1-\alpha) \cdot 100 \\%$ confidence interval, find the $(\frac{\alpha}{2})$ and $(1-\frac{\alpha}{2})$ percentiles of the bootstrap performance metrics,   

denoted as $\theta^\*_{\alpha /2}$ and  $\theta^\*_{(1-\alpha /2)}$, respectively.







The confidence interval is then given by $\( \left[ \theta^*_{(\frac{\alpha}{2})}, \theta^*_{(1-\frac{\alpha}{2})} \right] \)$.]
