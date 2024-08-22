## 1. Prerequisites:

</br>
<div style="text-align: left;">
  
A. [**Install Python Program**](https://www.python.org/downloads/) 

</div>
</br>

## 2. Download the documents of  `clone_repository`: 

A. On the GitHub page, click on the top menu to download the following files (as shown in the figure below):`best_model_svm.pkl` (trained model), `df_check_requirements.py`, and `model_apply.py`. 

</br>
<div style="text-align: left;">
  <img src="https://github.com/user-attachments/assets/79b9522e-a266-4c4c-a623-d817d504ca49" alt="Download Documents " style="width: 100%; max-width: 700px;"/>
</div>
</br>

## 3. Importing the Domestic Dataset Into Your Local Folder:

A. Transfer your dataset, renamed as`df_pituitary.xlsx` into your local `clone_repository` folder, as shown in the figure below.

</br>
<div style="text-align: left;">
  <img src="https://github.com/user-attachments/assets/f76ec799-d42a-4053-a2c3-c37c3ce33c72" alt="Importing the dataset into the repository" style="width: 100%; max-width: 700px;"/>
</div>
</br>

B. Ensure the dataset name is `df_pituitary.xlsx`. This step is crucial for running the machine learning model.

## 4. Running the `df_check_requirements.py` File:

A. This file checks whether the dataset has the appropriate characteristics before applying the algorithm. Specifically, it verifies that the libraries, column names, and values are correct.   
B. The dataset must match the format of the example available for reference in the `dataset ` folder > `dataset_example.csv` file.  
C. Right-click on `df_check_requirements.py` and select **Run File** as shown in the gif below. 

**Attention: The program is not yet compatible with datasets containing missing values. We recommend not using incomplete datasets at this time.**

</br>
<div style="text-align: left;">
  <img src="https://github.com/user-attachments/assets/9f7ca734-a85f-4baf-b614-0cad1fcab5b8" alt="Running files .py" style="width: 100%; max-width: 700px;"/>
</div>
</br>

## 5. Running the `model_apply.py` File:

A. This file will use our trained model `best_model_svm.pkl` to predict tumor consistency from your dataset.  
B. Right-click on `model_apply.py` and  select **Run File** aas shown in the gif above.  
C. For testing purposes, you can use the dataset `df_example.xlsx` in this folder. Rename it to `df_pituitary.xlsx`.   
D. The report below was automatically generated with the name `report_metrics_and_plot.pdf`.

</br>
<div style="text-align: left;">
  <img src="https://github.com/user-attachments/assets/e3f92f4a-ecd9-4cc7-b712-30fb95d9a9fb" alt="Example of the generated report" style="width: 100%; max-width: 700px;"/>
</div>
</br>

## 6. Updates and Contributions:  

*  Upcoming updates will include:
1. Creation of a virtual environment to run the files via Git and VSCode
2. Application of tasks to impute missing values
3. Application of pre-trained algorithms and untrained algorithms for performance comparison

*  Feel free to submit improvements and suggestions via Pull Requests.
*  For any issues or questions, please open an Issue on GitHub.
*  For any issues or questions, open an Issue on GitHub.

