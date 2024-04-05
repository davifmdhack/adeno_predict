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


</p>

## **Metrics and bootstrap implementation**

We used the following metrics considering the nature of the problem and its unbalanced data: (1) Area Under Curve (AUC) of Receiver Operating Curve (ROC), (2) accuracy, (3) sensibility (or recall), (4) specificity, 
(5) F1 score and (6) Matthew correlation coefficient (MCC). Accuracy was used only for comparative purposes with other metrics. The formulas and bootsrap techniques are described in `metrics_boot` folder.
