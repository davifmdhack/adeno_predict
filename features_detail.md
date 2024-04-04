## **Dataset format**
The features used were:

1. `ID`: Intern identification of patient $\rightarrow$ **sensitive variable excluded**
2. `sex`: Female (F) or Male (M)
3. `age`: in years
4. `diameter`: Size in centimeters of tumor using brain MRI
5. `adc`: ADC of tuymor using brain MRI
6. `pathology`: Divided into basophilic, chromophobe, chromophobe/eosinophilic, corticotroph, and eosinophilic 
7. `ki67`: Use of the interval variable: < 1%, 1 - 3%, 3 - 5%, > 5% 
8. `consistency` $\rightarrow$ **target feature**: soft (class = 0) or non-soft (class = 1)

An example of the dataset is available in the `dataset_example.csv` file
