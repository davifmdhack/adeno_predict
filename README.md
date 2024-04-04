<div align="center">
  <img src="https://github.com/davifmdhack/adeno_predict/assets/109975635/4d219850-2f80-481e-b1f3-32b4efb85165" alt="Slide12" style="width: 256px; border-radius: 20px;"/>
</div>

<div align = "center";> 

## Authors | Co-Authors

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
of the Apparent diffusion coefficient (ADC) variable from brain MRI, using multiple imputation by chain  equations (MICE) with Bayesian Ridge regression. Our group opted for the following 
classification algorithms: Decision Tree (DT), K-nearst Neighbor (KNN), Naive Bayes (NB), Support Vector Machine (SVM) and Ensemble (considering the sum of the prediction probability for 
non-soft consistency of the best model of each algorithm).

In this repository, we divided the codes according to the following steps: example of dataset (`dataset__example.csv`), data pre-processing (`pre_process` folder), imputation of missing values 
for ADC (`imputation` folder), tunning flow for training and obtaining the best model in the test for the algorithms (`workflow_algorithms` folder), metrics and bootstrap (`metrics_boot` folder).

</p>

## **Dataset format**
<p style="text-align: justify;">
  
Because data collection was carried out in a single research center, it was not necessary to build a server, implement a cluster or distributed processing. The database was built similar to 
the available file `dataset_example.csv` with single acess __PATH__ in domestic domain.

</p>

## **Pre-processing**
<p style="text-align: justify;">
  
Innitialy, we excluded the `ID` feature (sensitive data). We standardized attribute names and content of categorical variables. The dataset was divided into analysis_clear and analysis_imputed, 
with the attributes `ki67` and `pathology` initially excluded from **analysis_clear** and from **analysis_imputed** after the imputation process so as to have the greatest amount of aggregated information 
for predicting missing data. Missing consistency values were removed, considering that there is no simultaneity of missing values for `adc` and `consistency`. In both datasets, One Hot Encoder was 
applied to the variable `sex` and an upsampling process was applied, thus totaling **analysis_clear** to a total of 86 and **analysis_imputed** to a total of 96. Normalization or standardization was 
applied to the pipeline of each algorithm (see in `workflow_algorithms` folder).

</p>

## **Imputation missing values**

<p style="text-align: justify;">

</p>

## **Workflow algorithms**

<p style="text-align: justify;">

</p>

## **Metrics and bootstrap implementation**

<p style="text-align: justify;">

</p>
