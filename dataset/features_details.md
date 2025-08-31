# Dataset format

---

## Features used

1. `ID`: Intern identification of patient $\rightarrow$ **sensitive variable excluded**

2. `sex`: Female (F) or Male (M)

3. `age`: Unit: years

4. `diameter`: Size in centimeters of tumor using brain MRI: Unit (cm)

5. `adc`: ADC of tuymor using brain MRI: Unit ($10^{-3} mm^2/s$)

6. `consistency` $\rightarrow$ **target feature**: soft (class = 0) or non-soft (class = 1) consistency

## Features unused\*

1. `pathology`: Divided into basophilic, chromophobe, chromophobe/eosinophilic, corticotroph, and eosinophilic

2. `ki67`: Use of the interval variable: < 1%, 1 - 3%, 3 - 5%, > 5%

### Data set Example

An example of the dataset is available in the `dataset_example.csv` file  

$*$ Variables excluded $\rightarrow$ **postoperatively information**
