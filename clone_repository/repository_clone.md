## 1. Prerequisites:
   
A. [Git for Windows](https://git-scm.com/download/win)  
B. [Python](https://www.python.org/downloads/)  
C. [Visual Studio Code (VSCode)](https://code.visualstudio.com/)

## 2. Cloning repository into VSCode:

A. Open the **VSCode**  
B. Click on **View** in superior menu and select **Command Palette** or ``Ctrl + Shift + P``.  
C. Into search bar, write `Git: Clone` and select the option `Clone from GitHub`.  
D. Paste the URL's repository: `https://github.com/davifmdhack/adeno_predict.git`.  
E. Choose your diretory where you want to install the repository and click on **Select Repository Location**.  
F. After the clone, the VSCode will ask if you want to open your recently created respository. Select **Open**.

## 3. Enviroment configuration:  

A. In the sidebar, click on ``clone_repository.ipynb`` and select the file. 
B. 
B. Create a virtual environment:  
```
bash
python -m venv venv
```
C. Activate the virtual environment:  
```
bash
venv\Scripts\activate
```
D. Install required dependencies:  
```
pip install -r requirements.txt
```
## 4. Configure your dataframe:  

A. Place your CSV dataset in the 'data' folder into the repository.  
B. The file name must be `df_dataframe.csv`

## 5. Run the code:

A. Into the terminal run:
```
python scr/main.py
```

B. The code will load the `df_dataframe.csv` file, and will aplly machine learning methods and show the results. 

## 6. Updates and Contributions  
*  Feel free to send improvements and suggestions via Pull Requests.
*  For any problems or questions, open an Issue on GitHub.


