import numpy as np
import pandas as pd

# %% helper function for the output table
def get_results(res_list):
    df = pd.concat(res_list, axis=1)
    df = df[[2, 1, 0, 4, 5]].copy()
    df.columns = ['-0.7', '-0.4', '0', '0.4', '0.7']  # 1.59
    df['Data'] = [-0.007788243018194023, 9.41, 1.42, 6.62, 1.49, 1.60, 6.32,
                  0.787050, 0.570619, 0.610684, -0.002283,
                  0.69, 0.07, 1.12,
                  0.81, 0.59,
                  0.96, 0.98,
                  # 2.1846, 1.748, 1.7, 37.23, 1.21, 3.46, 34.97, 9.28, 0.39,
                  2.1846, 37.23, 1.21, 3.46, 34.97, 9.28, 0.39,
                  -0.21, -2.64, -2.64/0.682, 2.0,
                  3.48, 27.26, 2.01, 12.66, 0.85
                  ]
    df['S.E.'] = [0.0949157995752499, 0.006346, 0.10, np.nan, 0.10, 0.11, np.nan,
                  # 1995Q1 - 2023Q4 excluding 2020
                  0.03645029934467994, 0.06459522584546928, 0.06006193894870951, 0.09578212929453063,
                  0.04447445409023612, np.nan, 0.0726,
                  0.031697627659413016, 0.059500551047643103,
                  0.09128709291752768, 0.09128709291752768,
                  1.250, 18.41, 0.39, 9.28, 18.41, 0.60, 0.025,  # 1994Q1 - 2023Q4
                  0.3, 0.682, np.nan, np.nan,
                  12.74, 18.41, 15.85, 18.41, 0.026182768029732597  # 1994Q1 - 2023Q4
                  ]
    df = df[['Data', 'S.E.']+['-0.7', '-0.4', '0', '0.4', '0.7']]

    return df

# formatting function
def add_brackets(value):
    if pd.isna(value):
        return "---"  # value  # Or return np.nan, or any other logic you want for NaN values
    else:
        return f"({value:.2f})"
def formatting(value):
    if pd.isna(value):
        return "---"  # value  # Or return np.nan, or any other logic you want for NaN values
    else:
        return f"{value:.2f}"