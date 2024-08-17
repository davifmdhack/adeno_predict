pip install -r requirements.txt -q --no-warn-script-location --no-warn-conflicts
pip install tkinter pandas scikit-learn joblib matplotlib

import tkinter as tk
from tkinter import messagebox
import pandas as pd

df = pd.read_excel('df_pituitary.xlsx')
col_needed = ['age', 'adc', 'diameter', 'sex', 'consistency']

def verify_columns(df, col_needed):
    col_df = all(col in df.columns for col in col_needed)

    if col_df:
        messagebox.showinfo("Column Name Checker", "All name columns are correct.")
    else:
        col_miss = [col for col in col_needed if col not in df.columns]
        messagebox.showerror("Naming Error", f"Some columns have naming error: {', '.join(col_miss)}")

root = tk.Tk()
root.withdraw() 
verify_columns(df, col_needed)

def verify_missing_values(df):
    missing_values = df.isna().any().any() or (df == 0).any().any()
    if missing_values:
        messagebox.showwarning("Warning Missing Values", "The dataframe has missing or null values. It is not recommended in this version to continue.")
    else:
        messagebox.showinfo("Missing Values Verify", "There are no missing data or null values. The program can continue.")

root = tk.Tk()
root.withdraw() 
verify_missing_values(df)

def verify_val_columns(df):
    col_num = ['adc', 'age', 'diameter']
    col_incorrect_num = [col for col in col_num if not pd.api.types.is_numeric_dtype(df[col])]
    
    true_val_consistency = {'non-soft', 'soft'}
    false_val_consistency = df['consistency'][~df['consistency'].isin(true_val_consistency)]

    true_val_sex = {'F', 'M'}
    false_val_sex = df['sex'][~df['sex'].isin(true_val_sex)]

    error_message = []

    if col_incorrect_num:
        error_message.append(f"The columns {', '.join(col_incorrect_num)} must contain only numeric values.\n")
    
    if not false_val_consistency.empty:
        error_message.append(f"The column 'consistency' contain invalid values: {', '.join(false_val_consistency.unique())}. \nMust contain only 'non-soft' or 'soft'.")
    
    if not false_val_sex.empty:
        error_message.append(f"The column 'sex' ontain invalid values: {', '.join(false_val_sex.unique())}. \nMust contain only 'F' ou 'M'.")
    
    if error_message:
        messagebox.showerror("Error in Type of Values", "\n".join(error_message))
    else:
        messagebox.showinfo("Type of Values ", "All columns have correct value types")

root = tk.Tk()
root.withdraw() 
verify_val_columns(df)
