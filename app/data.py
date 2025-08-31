import os
import pandas as pd

def load_example_dataframe(csv_path: str = "examples/df_example.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path)


def save_results(df, path: str = "results/df_prediction-results.csv") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path
