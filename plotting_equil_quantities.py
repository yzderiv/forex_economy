"""
This script generates and plots equilibrium quantities for a given set of parameters.

Imports:
    time: Provides various time-related functions.
    pandas as pd: Library for data manipulation and analysis.
    torch: PyTorch library for tensor computations.
    econ_forex: Custom module for economic and forex calculations.
    sns: Seaborn library for statistical data visualization.
    plt: Matplotlib library for plotting.

Global Variables:
    rho_Psi (float): Parameter for the model.
    gs_carry (float): Parameter for the model.
    gs_ERP (float): Parameter for the model.
    is_xt_const (bool): Parameter for the model.
    is_frictional (bool): Indicates if the model includes friction.
    s_Psi (float): Parameter for the model.

Functions:
    make_plots(item_name, is_frictional, params): Generates plots for the given item and parameters.

Usage:
    The script sets various parameters and generates plots for different items such as 'r_H-r_F', 'CRP_H', 'mu_e', 'rc_F-r_F', 'r_H', and 'r_F'.
    The plots are displayed using plt.show().
"""
import time
import pandas as pd
import torch
from econ_forex import *

# sns.set_style("whitegrid")
sns.set_palette("Set2")
sns.set_context("notebook",  font_scale=2., rc={"lines.linewidth": 4})
sns.set_style("white")
sns.set_style("ticks")
pd.options.display.float_format = '{:.4f}'.format


# Parameter choices    
rho_Psi, gs_carry, gs_ERP, is_xt_const = 0.3, 0.0, 0.0, False

is_frictional, s_Psi = True, -0.4 
# Figure 2
# is_frictional, s_Psi = True, -0.4
# Figure 3
# is_frictional, s_Psi = False, -0.4
# Figure 4
# is_frictional, s_Psi = True, 0.4 
# Figure 5
# is_frictional, s_Psi = False, 0.4

for item_name in ['r_H-r_F', 'CRP_H', 'mu_e', 'rc_F-r_F', 'r_H', 'r_F']:
    make_plots(item_name, is_frictional, ((0.9, 0.9), 0.014, 0.7, 0.095, 0.4, rho_Psi, gs_carry, gs_ERP, is_xt_const))
plt.show()