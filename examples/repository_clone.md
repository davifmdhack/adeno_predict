## 1. Prerequisites:

</br>
<div style="text-align: left;">
  
A. [**Install Python Program**](https://www.python.org/downloads/)

</div>
</br>

## 2. Download the documents of  `clone_repository`

A. On the GitHub page, click on the top menu to download the following files (as shown in the figure below): `best_model_svm.pkl` (trained model) and `model_apply.py`.

</br>
<div style="text-align: left;">
  <img src="../images/download_instructions.png" alt="Download Documents " style="width: 100%; max-width: 700px;"/>
</div>
</br>

## 3. Importing the Local Dataset Into Your Folder (CSV only):

A. Transfer your dataset, renamed as `df_pituitary.csv` into your local `clone_repository` folder, as shown in the figure below.

</br>
<div style="text-align: left;">
  <img src="../images/import_dataset.png" alt="Importing the dataset into the repository" style="width: 100%; max-width: 700px;"/>
</div>
</br>

B. Ensure the dataset name is `df_pituitary.csv`. This step is crucial for running the machine learning model.

## 4. Dataset format requirements

A. The dataset must match the format of the example available for reference in the `dataset` folder > `dataset_example.csv` file.

**Attention: The program is not yet compatible with datasets containing missing values. We recommend not using incomplete datasets at this time.**

</br>
<div style="text-align: left;">
  <img src="../images/run_python_file.png" alt="Running files .py" style="width: 100%; max-width: 700px;"/>
</div>
</br>

## 5. Running the `model_apply.py` File

A. This file will use our trained model `best_model_svm.pkl` to predict tumor consistency from your dataset.  
B. Right-click on `model_apply.py` and  select **Run File** as shown in the gif above.  
C. For testing purposes, you can use the dataset example in `dataset/dataset_example.csv`. Copy it and rename to `examples/df_pituitary.csv`.
D. The report below will be automatically generated with the name `report_metrics_and_plot.pdf`.

</br>
<div style="text-align: left;">
  <img src="../images/report_example.png" alt="Example of the generated report" style="width: 100%; max-width: 700px;"/>
</div>
</br>

## 6. Updates and Contributions

1. Application of tasks to impute missing values

2. Application of pre-trained algorithms and untrained algorithms for performance comparison

* Feel free to submit improvements and suggestions via Pull Requests.

* For any issues or questions, please open an Issue on GitHub.
