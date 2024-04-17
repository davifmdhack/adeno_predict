<div align="center">
  <img src="https://github.com/davifmdhack/adeno_predict/assets/109975635/dec9ac98-0aee-488e-bfbe-b18b8f6d2053" alt="logos" style="width: 256px; border-radius: 20px;"/>
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
The objective of this application is to optimize the ability to predict non-soft tumors and consequently improve surgical planning and ultimately reduce post-surgical complications.    

Using a database of 70 patients from Hospital de Clínicas of the State University of Campinas (HC- UNICAMP). Applying pre-determined inclusion and exclusion criteria, with the aim 
of avoiding methodological biases, two datasets were analyzed, analysis_clear (n = 53) and analysis_imputed (n=59). The latter is the result of imputation of missing data for six values 
of the Apparent diffusion coefficient (ADC) variable from brain MRI, using multiple imputation by chain equations (MICE), more details in `imputation` folder. 
Our group opted for the following classification algorithms: Decision Tree (DT), K-nearst Neighbor (KNN), Naive Bayes (NB), Support Vector Machine (SVM) and Ensemble of all best models. 

In this repository, we divided the codes according to the following steps: example of dataset (`dataset__example.csv`), data pre-processing (`pre_process` folder), imputation of missing values 
for ADC (`imputation` folder), tunning flow for training and obtaining the best model in the test for the algorithms (`workflow_algorithms` folder), metrics and bootstrap (`metrics_boot` folder).  

We are building the session to clone the repository and apply our trained model to local datasets.It will be available in the future at `clone_repository` folder. 

</p>

## **Dataset format**
<p style="text-align: justify;">
 
Because data collection was carried out in a single research center, it was not necessary to build a server, implement a cluster or distributed processing. The database was built similar to 
the available file `dataset_example.csv` with single acess __PATH__ in domestic domain. The features used were described in `features_detail.md`. 

</p>

## **Pre-processing**
<p style="text-align: justify;">
  
We excluded sensitive features, and those associated with surgery process after imputation process. At this point, two datasets were made: **analysis_clear**, after randomized upsampling process 
n = 86, and **analysis_imputed** with n = 96 after upsampling.

</p>

## **Imputation missing values**

<p style="text-align: justify;">

The MICE imputation method was used to reduce the ignorability of missing data, since imputation (using appropriate criteria) increases the variability and reliability of the derived machine learning models.
This step is one of the reasons for collecting post-surgery information (KI 67 and histopathological data) to improve the ability to predict missing values. In the `imputation` folder, a function was developed 
to apply the same criteria to future datasets with the presence of missing data.

</p>

## **Workflow algorithms**


<p style="text-align: justify;">
We apply pipeline from scikit-learn of the pre-processed dataset for each algorithm, considering particularities such as standardization of numerical variables. A cross-validation method using Strafied KFold was 
used to hyperparameterize the models until finding the `best_model` for each algorithm.The standardized code can be found at `workflow_algorithms` folder. 

</p>

## **Metrics and bootstrap implementation**

We used the following metrics considering the nature of the problem and its unbalanced data: (1) Area Under Curve (AUC) of Receiver Operating Curve (ROC), (2) Accuracy, (3) Sensibility (or recall), (4) Specificity, 
(5) F1 score and (6) Matthew Correlation Coefficient (MCC). Accuracy was used only for comparative purposes with other metrics. The formulas and bootsrap techniques are described in `metrics_boot` folder.


## **Repository cloning and application of the model to domestic dataset**
<div align="center"> 
  
  ### **UNDER CONSTRUCTION**
  
</div>

<div align="center">
  <img src="https://github.com/davifmdhack/adeno_predict/assets/109975635/e6d5f07a-c321-4c16-bd76-78817fc2b13b" alt="Under Construction" style="width: 200px;">
</div>

## **References**
1. Van Buuren, S. (2018) Flexible Imputation of Missing Data. 2nd Edition, CRC Press, Boca Raton. https://doi.org/10.1201/9780429492259
2. Mas̕s, S. (2021). Interpretable Machine Learning with Python : Learn to Build Interpretable High-performance Models with Hands-on Real-world Examples. 1st Edition, Packt Publishing, Birmingham. 
3. Garbin, C., & Marques, O. (2022). Assessing Methods and Tools to Improve Reporting, Increase Transparency, and Reduce Failures in Machine Learning Applications in Health Care. Radiology: Artificial Intelligence, 4(2). https://doi.org/10.1148/ryai.210127
4. Rouzrokh, P., Khosravi, B., Faghani, S., Moassefi, M., Garcia, D. V. V., Singh, Y., Zhang, K., Conte, G. M., & Erickson, B. J. (2022). Mitigating Bias in Radiology Machine Learning: 1. Data Handling. Radiology: Artificial Intelligence, 4(5). https://doi.org/10.1148/ryai.210290
5. Faghani, S., Khosravi, B., Zhang, K., Moassefi, M., Jagtap, J. M., Nugen, F., Vahdati, S., Kuanar, S. P., Rassoulinejad-Mousavi, S. M., Singh, Y., Vera Garcia, D. v., Rouzrokh, P., & Erickson, B. J. (2022). Mitigating Bias in Radiology Machine Learning: 3. Performance Metrics. Radiology: Artificial Intelligence, 4(5). https://doi.org/10.1148/ryai.220061


## **Institutional** 
</br>
<div style="float: left;">
  <img src="https://github.com/davifmdhack/adeno_predict/assets/109975635/dec66e61-fab1-4091-8655-8c6e0f7b0d17" alt="fcm_unicamp_logo" style="width: 250px;">
</div>
School of Medical Sciences State University of Campinas - FCM/UNICAMP

<hr style="width: 50%;">
</br>
<div style="float: left;">
  <img src="https://github.com/davifmdhack/adeno_predict/assets/109975635/78c66f70-c8c5-46b8-8f85-d5aaff665d01" alt="logo_hc_unicamp" style="width: 250px;">
</div>
Unicamp Clinical Hospital - HC/UNICAMP


