# Project Overview

This project contains scripts for generating and plotting equilibrium quantities and simulated moments for a frictional forex economic model.

## Files

### plotting_equil_quantities.py

This script generates and plots equilibrium quantities for a given set of parameters.

- **Imports**: time, pandas, torch, econ_forex, seaborn, matplotlib.pyplot
- **Global Variables**: rho_Psi, gs_carry, gs_ERP, is_xt_const, is_frictional, s_Psi
- **Functions**: make_plots(item_name, is_frictional, params)
- **Usage**: Sets various parameters and generates plots for different items such as 'r_H-r_F', 'CRP_H', 'mu_e', 'rc_F-r_F', 'r_H', and 'r_F'. The plots are displayed using plt.show().

### generating_simulated_moments.py

This script generates simulated moments for the forex economy and outputs the results in a formatted table.

- **Imports**: time, pandas, torch, econ_forex, helper_generating_tables, seaborn, matplotlib.pyplot
- **Global Variables**: sub_index, is_frictional, rho_Psi, gs_x, gk_x, gs_carry, gs_ERP, is_xt_const, betas, gs_bar_Psi, s_Psi, res_list
- **Functions**: generating_table()
- **Usage**: Run the script to generate simulated moments and output the results in a formatted table. The script can be configured to generate different tables by modifying the global variables.

### econ_forex.py

This module contains classes and functions for the forex economy model.

- **Imports**: torch, numpy, pandas, statsmodels.api, seaborn, matplotlib.pyplot
- **Classes**: baseline, frictional
- **Functions**: get_states(betas), set_inputs(paras), get_econ_wrapper(is_frictional, paras), torch_bilinear, torch_set_approx, torch_get_approx_scalar, torch_get_approx_array, torch_get_sim, get_beta, get_cov, torch_moments, econ_plot_wrapper, make_plots(item_name, is_frictional, paras)
- **Usage**: Provides the core functionality for setting up and simulating the forex economy model, as well as computing and plotting various economic moments and quantities.

### computing_empirical_moments.py

This script computes empirical moments from the provided data files.

- **Imports**: pandas, numpy, matplotlib.pyplot

- **Functions**: get_quantity(df, country1, country2, indicator), get_corr_std(res, n)

- **Usage**: Reads data from CSV files, computes various empirical moments, and plots rolling correlations.
