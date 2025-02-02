"""
This script generates simulated moments for forex economy and outputs the results in a formatted table.

Modules:
    time: Provides various time-related functions.
    pandas as pd: A powerful data manipulation and analysis library for Python.
    torch: A deep learning framework for Python.
    econ_forex: Custom module for economic and forex calculations.
    helper_generating_tables: Custom module for generating and formatting tables.
    seaborn as sns: A statistical data visualization library.
    matplotlib.pyplot as plt: A plotting library for creating static, animated, and interactive visualizations.

Functions:
    generating_table(): Generates a table of simulated moments for different parameter configurations.

Global Variables:
    sub_index (list): List of moment names to be included in the output table.
    is_frictional (bool): Indicates whether the economy is frictional or not in the simulation.
    rho_Psi (float): Parameter for the simulation.
    gs_x (float): Parameter for the simulation.
    gk_x (float): Parameter for the simulation.
    gs_carry (float): Parameter for the simulation.
    gs_ERP (float): Parameter for the simulation.
    is_xt_const (bool): Indicates whether xt is constant in the simulation.
    betas (tuple): Tuple of beta parameters for the simulation.
    gs_bar_Psi (float): Parameter for the simulation.
    s_Psi (float): Parameter for the simulation.
    res_list (list): List to store results of the simulations.

Usage:
    Run the script to generate simulated moments and output the results in a formatted table.
    The script can be configured to generate different tables by modifying the global variables.
"""
import time
import pandas as pd
import torch
from econ_forex import *
from helper_generating_tables import get_results, add_brackets, formatting
# plotting
import seaborn as sns
import matplotlib.pyplot as plt
# sns.set_style("whitegrid")
sns.set_palette("Set2")
sns.set_context("notebook",  font_scale=2., rc={"lines.linewidth": 4})
sns.set_style("white")
sns.set_style("ticks")
pd.options.display.float_format = '{:.4f}'.format

# %% Table 2
sub_index = ['ac1($de$)', 'std($de$)(\\%)', 'std($dg_H$)(\\%)', 'std($de$)/std($dg_H$)', 'std($d\\bar c_H$)(\\%)', 'std($d\\bar c_F$)(\\%)', 'std($de$)/std($d\\bar c_H$)', 
            'corr($d\\bar c_H, dg_H$)', 'corr($dg_H, dg_F$)', 'corr($d\\bar c_H, d\\bar c_F$)', 'corr($d\\bar c_H-d\\bar c_F, de$)', 
            'std($r_H-r_F$)(\\%)', 'std($r_H-r_F$)/std($de$)', 'std($r_H$)(\\%)',
            'corr($r_H, r_F$)', 'corr($dr_H, dr_F$)', 
            'ac1($r_H-r_F$)', 'ac1($r_H$)', 
            'Fama-$\\beta$', 'carry SR(\\%)', 'carry i-diff (\%)', 'carry (\%)', 'carry ratio (\%)', 'std(carry) (\%)', 'std(carry i-diff) (\%)',
            'mean(CIP$_H$)(\\%)', 'CIP-$\\beta$', 't CIP',  '$R^2$ CIP(\\%)',
            'ERP$_H$(\%)', 'equity SR$_H$(\%)', 'ERP$_F$(\%)', 'equity SR$_F$(\%)', 'corr(ERP$_H$, ERP$_F$)']


is_frictional, rho_Psi, gs_x, gk_x, gs_carry, gs_ERP, is_xt_const = True, -0.3, 0.045, 0.03, 0.0, 0.0, False
betas = (0.9, 0.9)
gs_bar_Psi, s_Psi = 0.095, -0.4
# Table 13
# betas = (0.501, 0.501) converge to the case with no home-bias
# gs_bar_Psi, s_Psi = 0.0, 1.0 corresponds to no demand shocks

res_list = []
for is_frictional in [False, True]:
    torch.mps.manual_seed(42)
    start = time.time()
    df_dict = torch_get_sim(n_path=int(1e4), T=80, step_size=1/12, init_x=0.0, init_cQ=1.0,
                            is_frictional=is_frictional,
                            paras=(betas, 0.014, 0.7, gs_bar_Psi, s_Psi, rho_Psi, gs_carry, gs_ERP, is_xt_const), gs_x=gs_x, gk_x=gk_x)
    end = time.time()
    print("Seconds to run: %.2f" % (end-start))
    tmp_res = torch_moments(df_dict.copy(), int(1e4))
    res_list.append(pd.Series(tmp_res.moments_dict)[sub_index])

df = pd.concat(res_list, axis=1)
df

# %% helper function for the output table
def generating_table():
    # moments to output
    res_list = []
    for sign_s_Psi in [-1, 1]:
        for s_Psi in [.0, 0.4, 0.7]:
            torch.mps.manual_seed(42)
            start = time.time()
            df_dict = torch_get_sim(n_path=int(1e4), T=80, step_size=1/12, init_x=0.0, init_cQ=1.0,
                                    is_frictional=is_frictional,
                                    paras=((0.9, 0.9), 0.014, 0.7, 0.095, sign_s_Psi*s_Psi, rho_Psi, gs_carry, gs_ERP, is_xt_const), gs_x=gs_x, gk_x=gk_x)
            end = time.time()
            print("Seconds to run: %.2f" % (end-start))
            tmp_res = torch_moments(df_dict.copy(), int(1e4))
            res_list.append(pd.Series(tmp_res.moments_dict)[sub_index])

    df_out = get_results(res_list)
    for item in df_out.columns:
        if item == 'S.E.':
            df_out[item] = df_out[item].apply(add_brackets)
        else:
            df_out[item] = df_out[item].apply(formatting)
    return df_out


# %%
# Table 3
# friction, rho_Psi, gs_x, gk_x, gs_carry, gs_ERP = False, -0.3, 0.045, 0.03, 0.0, 0.0, False
# Table 4
# friction, rho_Psi, gs_x, gk_x, gs_carry, gs_ERP = True, -0.3, 0.045, 0.03, 0.0, 0.0, False
# Table 5
# friction, rho_Psi, gs_x, gk_x, gs_carry, gs_ERP = False, 0.3, 0.045, 0.03, 0.0, 0.0, False
# Table 6
# friction, rho_Psi, gs_x, gk_x, gs_carry, gs_ERP = True, 0.3, 0.045, 0.03, 0.0, 0.0, False
# Table 7
# friction, rho_Psi, gs_x, gk_x, gs_carry, gs_ERP = False, 0.0, 0.045, 0.03, 0.0, 0.0, False
# Table 8
# friction, rho_Psi, gs_x, gk_x, gs_carry, gs_ERP = True, 0.0, 0.045, 0.03, 0.0, 0.0, False
# Table 9
# friction, rho_Psi, gs_x, gk_x, gs_carry, gs_ERP = True, 0.3, 0.03, 0.04, 0.0, 0.0, False
# Table 10
# friction, rho_Psi, gs_x, gk_x, gs_carry, gs_ERP = True, 0.3, 0.03, 0.02, 0.0, 0.0, False
# Table 11
# friction, rho_Psi, gs_x, gk_x, gs_carry, gs_ERP = True, 0.3, 0.06, 0.03, 0.0, 0.0, False
# Table 12
# friction, rho_Psi, gs_x, gk_x, gs_carry, gs_ERP = True, 0.3, 0.06, 0.04, 0.0, 0.0, False
# Table 14
# friction, rho_Psi, gs_x, gk_x, gs_carry, gs_ERP, is_xt_const = False, 0.3, 0.06, 0.03, 0.0, 0.0, True
# Table 15
# friction, rho_Psi, gs_x, gk_x, gs_carry, gs_ERP, is_xt_const = True, 0.3, 0.06, 0.03, 0.0, 0.0, True
# Table 17
# friction, rho_Psi, gs_x, gk_x, gs_carry, gs_ERP = False, 0.3, 0.06, 0.03, -0.5, 0.3, False
# Table 18
# friction, rho_Psi, gs_x, gk_x, gs_carry, gs_ERP = True, 0.3, 0.06, 0.03, -0.5, 0.3, False


df = generating_table()
#df.to_csv("results/moments.csv")
#df.to_latex(buf="moments.tex", column_format="lrrrrrrr")

