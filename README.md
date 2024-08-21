<div align="center">
  <img src="https://github.com/davifmdhack/adeno_predict/assets/109975635/dec9ac98-0aee-488e-bfbe-b18b8f6d2053" alt="logos" style="width: 300px; border-radius: 20px;"/>
</div>

<div align = "center";> 

## GitHub Page Authors

#### *Davi Ferreira MD., MSc.* 
[![Send e-mail to Davi Ferreira](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:davi.ferreira.soares@gmail.com)

#### *Fernanda Veloso MD., PhD. candidate* 
[![Send e-mail to Fernanda](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:fernandavelosop@gmail.com)

</div>

## **Introduction**
<p style="text-align: justify;">

This repository (__Adeno Predict__) serves the purpose of applying machine learning algorithms to predict the consistency of pituitary macroadenomas from demographic data and brain MRI parameters. 
The objective of this application is to optimize the ability to predict non-soft consistency and consequently improve surgical planning and ultimately reduce post-surgical complications.    

Using a database of 70 patients from Hospital de Clínicas of the State University of Campinas (HC- UNICAMP). Our group opted for the following classification algorithms: Decision Tree (DT), K-nearst Neighbor (KNN), Support Vector Machine (SVM) and Ensemble of two best models (DT and SVM). 

In this repository, we divided the codes according to the following steps: example of `dataset` folder (`dataset__example.csv`), imputation of missing values (`imputation` folder), tunning flow using Leave-One-Out strategy (`workflow_algorithms` folder), metrics and bootstrap (`metrics` folder).  

</p>

## **Dataset format**
<p style="text-align: justify;">
 
Because data collection was carried out in a single research center, it was not necessary to build a server, implement a cluster or distributed processing. The database was built similar to 
the available file `dataset_example.csv` with single acess __PATH__ in domestic domain in `dataset` folder. The features used were described in `features_detail.md`. 

</p>

## **Imputation missing values**
<p style="text-align: justify;">

The imputation process was used according to Van Buuren criteria, six values for *ADC* and eleven for *consistency*. KNN was used for deterministic process and multiple imputation by chained equations (MICE) with linear regression for stochastic methods. More information in `imputation` folder. 

</p>

## **Workflow algorithms**
<p style="text-align: justify;">

We applied a pipeline from scikit-learn of the pre-processed dataset for each algorithm, considering particularities such as standardization of numerical variables. A cross-validation method using Leave-One-Out (`leave_one_out.md`) with
10 folds for cross-validation process until finding the `best_model` for each algorithm considering the parameters (`algorithms_parameters.md`).

</p>

## **Metrics and bootstrap implementation**
<p style="text-align: justify;">

We used the following metrics considering the nature of the problem and its unbalanced data: (1) Area Under Curve (AUC) of Receiver Operating Curve (ROC), (2) Average precision-recall (AP), (3) Sensitivity (or Recall), (4) Specificity, (5) F1 score and (6) Matthew Correlation Coefficient (MCC). The formulas and bootsrap techniques are described in `metrics` folder. Bootstrap was used to find interval confidence (IC) with 95% confidence (n= 1000) after find best threshold (`bootstrap_code.md`). 

</p>

## **Clone repository and application for domestic dataset**
<p style="text-align: justify;">
  
We have developed a step-by-step guide, available in `clone_repository` > `repository_clone.md`, so that researchers can apply our trained model if they have the necessary information.

At the moment, this application is limited to databases that have all the required values. In the future, we will implement a method for imputing missing data.

</p>

## **References**
1. van Buuren S. Flexible Imputation of Missing Data, Second Edition. Second edition. | Boca Raton, Florida : CRC Press, [2019] |: Chapman and Hall/CRC; 2018. doi: 10.1201/9780429492259.
2. Mas̕s, S. (2021). Interpretable Machine Learning with Python : Learn to Build Interpretable High-performance Models with Hands-on Real-world Examples. 1st Edition, Packt Publishing, Birmingham. 
3. Garbin, C., & Marques, O. (2022). Assessing Methods and Tools to Improve Reporting, Increase Transparency, and Reduce Failures in Machine Learning Applications in Health Care. Radiology: Artificial Intelligence, 4(2). https://doi.org/10.1148/ryai.210127.
4. Rouzrokh, P., Khosravi, B., Faghani, S., Moassefi, M., Garcia, D. V. V., Singh, Y., Zhang, K., Conte, G. M., & Erickson, B. J. (2022). Mitigating Bias in Radiology Machine Learning: 1. Data Handling. Radiology: Artificial Intelligence, 4(5). https://doi.org/10.1148/ryai.210290
5. Faghani, S., Khosravi, B., Zhang, K., Moassefi, M., Jagtap, J. M., Nugen, F., Vahdati, S., Kuanar, S. P., Rassoulinejad-Mousavi, S. M., Singh, Y., Vera Garcia, D. v., Rouzrokh, P., & Erickson, B. J. (2022). Mitigating Bias in Radiology Machine Learning: 3. Performance Metrics. Radiology: Artificial Intelligence, 4(5). https://doi.org/10.1148/ryai.220061.
6. Murphy KP. Probabilistic machine learning : advanced topics. Cambridge, Massachusetts: The MIT Press; 2023.

</br>

<hr style="width: 100%;">

### **Institutional** 
</br>
<div style="float: center;">
  <img src="https://github.com/davifmdhack/adeno_predict/assets/109975635/dec66e61-fab1-4091-8655-8c6e0f7b0d17" alt="fcm_unicamp_logo" style="width: 300px;">
</div>

*School of Medical Sciences State University of Campinas - FCM/UNICAMP*


</br>
<div style="float: center;">
  <img src="https://github.com/davifmdhack/adeno_predict/assets/109975635/78c66f70-c8c5-46b8-8f85-d5aaff665d01" alt="logo_hc_unicamp" style="width: 300px;">
</div>

*Unicamp Clinical Hospital - HC/UNICAMP*

<hr style="width: 100%;">

### **Institutional Partnership**
</br>
<div style="float: center;">
  <img src="https://github.com/user-attachments/assets/27f3c1a0-49bc-4edb-81da-1a984a0a76fd" alt="ITA_logo_2" style="width: 300px;">
</div>

*Aeronautics Institute of Technology - ITA*
