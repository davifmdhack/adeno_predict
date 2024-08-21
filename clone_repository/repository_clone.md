## 1. Prerequisites:

A. [Git for Windows](https://git-scm.com/download/win)  
B. [Python](https://www.python.org/downloads/)  
C. [Visual Studio Code (VSCode)](https://code.visualstudio.com/)

## 2. Cloning the Repository in VSCode:

A. Open **VSCode**.  
B. Click on **View** in top menu and select **Command Palette** or press``Ctrl + Shift + P``.  
C. In the search bar, type `Git: Clone` and select the option `Clone from GitHub`.  
D. Paste the URL's repository: `https://github.com/davifmdhack/adeno_predict.git`.  

</br>
<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/f5840307-7b9d-4230-93be-822aa515754c" alt="Cloning the Adeno Predict GitHub repository" style="width: 200%; max-width: 600px;"/>
</div>
</br>

E. Choose the directory where you want to install the repository and click **Select Repository Location**.  
F. After cloning, VSCode will prompt you to open your newly created repository. Select **Open**.

</br>
<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/a310fd43-418f-4615-b167-8b84986567a6" alt="Opening the cloned repository in VSCode" style="width: 100%; max-width: 150px;"/>
</div>
</br>

## 3. Importing the Domestic Dataset into the Repository:

A. Transfer the dataset named `df_pituitary.xlsx` into the `clone_repository` folder as shown in the figure below.

</br>
<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/f76ec799-d42a-4053-a2c3-c37c3ce33c72" alt="Importing the dataset into the repository" style="width: 100%; max-width: 150px;"/>
</div>
</br>

B. Ensure the dataset name is `df_pituitary.xlsx`. This step is crucial for running the machine learning model.

## 4.Running the `df_check_requirements.py` File:

A. This file checks whether the dataset has the appropriate characteristics before applying the algorithm. Specifically, it verifies that the libraries, column names, and values are correct. 
B. The dataset must match the format of the example available for reference in the `dataset ` folder > `dataset_example.csv` file. 

**Attention: The program is not yet compatible with datasets containing missing values. We recommend not using incomplete datasets at this time.**

## 5. Running the `model_apply.py` File:

A. This file will use our trained model `best_model_svm.pkl` to predict tumor consistency from your dataset.  
B. Using the dataset `df_example.xlsx`, present in this folder, renaming it to `df_pituitary.xlsx` the report below was automatically generated with the name `report_metrics_and_plot.pdf`

</br>
<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/e3f92f4a-ecd9-4cc7-b712-30fb95d9a9fb" alt="Example of the generated report" style="width: 100%; max-width: 150px;"/>
</div>
</br>

## 6. Updates and Contributions:  

* Feel free to submit improvements and suggestions via Pull Requests.
* For any issues or questions, open an Issue on GitHub.

