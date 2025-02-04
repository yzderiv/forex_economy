import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
df1 = pd.read_csv('IFS_07-10-2024 02-33-13-28_timeSeries.csv')
df2 = pd.read_csv('bis_dp_search_export_20240713-154754.csv', skiprows=2)
df2['TIME_PERIOD']    = df2['TIME_PERIOD:Period']
df2['Reference area'] = [item.split(':')[1] + ' (19 countries)' for item in df2['REF_AREA:Reference area']]
df2['de']             = -np.log(df2['OBS_VALUE:Value']).diff()
df2.index             = [f"{item.year}Q{item.quarter}" for item in pd.to_datetime(df2['TIME_PERIOD'])]
# %%
def get_quantity(df, country1, country2, indicator):
    df_c = df.loc[(df['Country Code']==country1) & (df['Indicator Code']==indicator)].T.dropna().T
    tmp1 = df_c[df_c.columns[5:]].T
    tmp1.columns = [df_c['Country Name'].values.item()]
    df_c = df.loc[(df['Country Code']==country2) & (df['Indicator Code']==indicator)].T.dropna().T
    tmp2 = df_c[df_c.columns[5:]].T
    tmp2.columns = [df_c['Country Name'].values.item()]
    df_out = np.log(((tmp2.iloc[:].join(tmp1))).astype(float)).diff()
    (df_out[df_out.columns[0]].rolling(20).corr(df_out[df_out.columns[1]])).plot(title='Rolling 20Q corr')
    return df_out

df_cons = get_quantity(df1, 163, 111, 'NCP_SA_XDC')
df_gdp  = get_quantity(df1, 163, 111, 'NGDP_SA_XDC')

# %%
def get_corr_std(res, n):
    return (1-res**2)/np.sqrt(n-2)

"""
Full Sample (1995Q1 - 2023Q4)
corr(dcH-dcF, de) = 0.005129 (0.09406961211774315)
n = 115
s.e. = (1 - corr(dcH-dcF, de)**2) / np.sqrt(n - 2)

Excluding 2020
corr(dcH-dcF, de) = -0.002283 (0.09578212929453063)
n = 111
s.e. = (1 - corr(dcH-dcF, de)**2) / np.sqrt(n - 2)
"""
df_res            = df_cons.join(df2['de']).dropna()
df_res['dcH-dcF'] = df_res['United States'] - df_res['Euro Area']
df_res['dcH-dcF'].rolling(20).corr(df_res['de']).plot()
df_res.loc[[not item.startswith('2020') for item in df_res.index]].corr()

get_corr_std(-0.002283, 111)
# %%
"""
Full Sample (1995Q1 - 2023Q4)
corr(GDP_H, GDP_F)_GDP = 0.894754 (0.018759411488514618)
n = 115
s.e. = (1 - corr(GDP_H, GDP_F)**2) / np.sqrt(n - 2)

Excluding 2020
corr(GDP_H, GDP_F)_GDP = 0.570619 (0.06459522584546928)
n = 111
s.e. = (1 - corr(GDP_H, GDP_F)**2) / np.sqrt(n - 2)
"""
df_res = df_res.join(df_gdp, rsuffix='_GDP')
df_res.corr()
df_res.loc[[not item.startswith('2020') for item in df_res.index]].corr()

"""
Full Sample
United States    0.029434 (0.001949)
Euro Area        0.039579 (0.002621)
de               0.093530 (0.006194)
dcH-dcF          0.018782 (0.001244)
United States_GDP    0.026964 (0.001786)
Euro Area_GDP        0.031182 (0.002065)
s.e. = sample_std / np.sqrt(2 * (n - 1))

Excluding 2020
United States    0.014863 (0.001002)
Euro Area        0.015975 (0.001077)
de               0.094124 (0.006346)
dcH-dcF          0.013642 (0.000920)
United States_GDP    0.014172 (0.000955)
Euro Area_GDP        0.013712 (0.000924)
ac1(de) -0.007788243018194023 (0.0949157995752499)
"""
df_res.std()*2
df_res.loc[[not item.startswith('2020') for item in df_res.index]].std()*2

"""
Full Sample
corr(c_H, c_F) = 0.892768 (0.01909336915518624)
n = 115
s.e. = (1 - corr(c_H, c_F)**2) / np.sqrt(n - 2) 

Excluding 2020
corr(c_H, c_F) = 0.610684 (0.06006193894870951)
n = 111
s.e. = (1 - corr(c_H, c_F)**2) / np.sqrt(n - 2) 
"""

"""
corr(c_H, GDP_H) = 0.787050 (0.03645029934467994)
n = 111
s.e. = (1 - corr(c_H, GDP_H)**2) / np.sqrt(n - 2)
"""
df_res.corr()
df_res.loc[[not item.startswith('2020') for item in df_res.index]].corr()


# %% Carry Trade
df_data = pd.read_csv("OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0+all.csv")

eu = ['EA19']

def get_short_term_rate(df):
    tmp = df.loc[(df['Measure']=='Short-term interest rates') 
              & (df['REF_AREA'].isin(eu)) 
              & (df['FREQ']=='Q'), ['REF_AREA', 'Reference area', 'TIME_PERIOD', 'OBS_VALUE']].sort_values(by=['REF_AREA', 'TIME_PERIOD'])

    tmp1 = df.loc[(df['Measure']=='Short-term interest rates') 
                & (df['REF_AREA']=='USA') 
                & (df['FREQ']=='Q'), ['REF_AREA', 'TIME_PERIOD', 'OBS_VALUE']].sort_values(by=['REF_AREA', 'TIME_PERIOD'])


    df_r = pd.merge(tmp, tmp1, on='TIME_PERIOD', how='inner', suffixes=('_EU', '_USA'))

    df_r['diff'] = (df_r['OBS_VALUE_USA'] - df_r['OBS_VALUE_EU'])/100/4
    df_r['TIME_PERIOD'] = df_r['TIME_PERIOD'].str.replace('Q1', '03-31').str.replace('Q2', '06-30').str.replace('Q3', '09-30').str.replace('Q4', '12-31')

    return df_r
df_r      = get_short_term_rate(df_data)
df2['de'] = np.log(df2.groupby(['Reference area'])['OBS_VALUE:Value'].shift(-1)) - np.log(df2['OBS_VALUE:Value'])

df_rates          = pd.merge(df_r, df2[['TIME_PERIOD','Reference area', 'OBS_VALUE:Value', 'de', 'Unit']], 
                             on=['TIME_PERIOD', 'Reference area'], how='inner').dropna()
df_rates['date']  = pd.to_datetime(df_rates['TIME_PERIOD'])
df_rates          = df_rates.loc[(df_rates['date'].dt.year>=1980)&(df_rates['date'].dt.year<2024)]
df_rates['crp']   = df_rates['diff'] + df_rates['de']
df_rates['carry'] = df_rates['crp']*np.sign(df_rates['diff'])
df_rates['str']   = df_rates['diff']*np.sign(df_rates['diff']) 

# %% Rates
"""
str: short-term interest rate differential
1994Q1 - 2023Q4,
carry    0.034554 (0.092798)
str      0.012127 (0.003911)
"""
df_rates[['carry','str']].mean()*4

"""
carry    0.092798 (0.006015200913819375)
str      0.003911 (0.00025351247628125153)
"""
df_rates[['carry','str']].std()*2

"""
carry    0.372354 (0.184141)
str      3.100667 (0.185057)
from scipy.special import gamma
n    = 120
sr   = df_rates[['carry','str']].mean()/df_rates[['carry','str']].std()
tmp  = sr/(np.sqrt((n-1)/2)*gamma((n-2)/2)/gamma((n-1)/2))
s.e. = np.sqrt((n-1)/(n-3)*(1+tmp**2) - sr**2)/np.sqrt(n)*2
"""
df_rates[['carry','str']].mean()*4/(df_rates[['carry','str']].std()*2)

"""
corr(r_H, r_F) = 0.809738 (0.031697627659413016)
n    = 120
s.e. = (1 - corr(r_H, r_F)**2) / np.sqrt(n - 2)
"""
df_rates[['OBS_VALUE_EU', 'OBS_VALUE_USA']].corr()

"""
corr(dr_H, dr_F) = 0.594692 (0.059500551047643103)
n    = 120
s.e. = (1 - corr(dr_H, dr_F)**2) / np.sqrt(n - 2)
"""
df_rates[['OBS_VALUE_EU', 'OBS_VALUE_USA']].diff().corr()

"""
std(r_H-r_F) = 0.00686118460512932 (0.0004447445409023612)
"""
(df_rates['diff']).std()*2

(df_rates[['OBS_VALUE_EU', 'OBS_VALUE_USA']]/4/100).std()*2
"""
std(r_EU)     0.011031 (0.000715)
std(r_USA)    0.011207 (0.000726)
"""

"""
ac1(r_H-r_F) = 0.9578499765453311 (0.09128709291752768)
n    = 120
s.e. = 1/np.sqrt(n)
"""
df_rates['diff'].autocorr()

"""
ac1(de) = -0.002060859342143274 (0.09128709291752768)
"""
df_rates['de'].autocorr()

"""
std(de) = 0.09350014229873806 (0.0060607140390699235)
s.e. = std/np.sqrt(2 * (n - 1))
"""
df_rates['de'].std()*2


"""
ac1(r_H)   = 0.9850953635777261 (0.09128709291752768)
ac1(r_USA) = 0.9774329312288402 (0.09128709291752768)
"""
df_rates[['OBS_VALUE_EU', 'OBS_VALUE_USA']].autocorr()

# %% Equity Risk Premium
def get_share_price(df):
    tmp = df.loc[(df['Measure']=='Share prices') 
                & (df['REF_AREA'].isin(eu)) 
                & (df['FREQ']=='Q'), ['REF_AREA', 'Reference area', 'TIME_PERIOD', 'OBS_VALUE']].sort_values(by=['REF_AREA', 'TIME_PERIOD'])

    tmp1 = df.loc[(df['Measure']=='Share prices')
                & (df['REF_AREA']=='USA') 
                & (df['FREQ']=='Q'), ['REF_AREA', 'TIME_PERIOD', 'OBS_VALUE']].sort_values(by=['REF_AREA', 'TIME_PERIOD'])

    df_s = pd.merge(tmp, tmp1, on='TIME_PERIOD', how='inner', suffixes=('_EU', '_USA'))
    df_s['TIME_PERIOD'] = df_s['TIME_PERIOD'].str.replace('Q1', '03-31').str.replace('Q2', '06-30').str.replace('Q3', '09-30').str.replace('Q4', '12-31')

    df_s[['ERP_EU', 'ERP_US']] = np.log(df_s[['OBS_VALUE_EU', 'OBS_VALUE_USA']]).diff().shift(-1)
    return df_s
df_s = get_share_price(df_data)
df_s = df_s.iloc[:-2]

df_s[['ERP_EU', 'ERP_US']].corr()
df_ERP = pd.merge(df_r, df_s[['TIME_PERIOD', 'ERP_EU', 'ERP_US']], on='TIME_PERIOD', how='inner')

"""
1994Q1 - 2023Q4, excluding 2020
Sharpe ratio
ERP_EU    0.109897
ERP_US    0.243808

ERP
ERP_EU    0.017309 (0.157499)
ERP_US    0.030292 (0.124247)

corr
0.839109 (0.02771320759616104)


ERP
ERP_EU    0.020067 (0.158542)
ERP_US    0.034732 (0.127412)

Sharpe ratio
ERP_EU    0.126572 (0.184130)
ERP_US    0.272593 (0.184135)

corr
0.845921 (0.026182768029732597)

"""
df_ERP[['ERP_EU', 'ERP_US']] = df_ERP[['ERP_EU', 'ERP_US']].values - (df_ERP[['OBS_VALUE_EU', 'OBS_VALUE_USA']]/100/4).values
df_ERP[['ERP_EU', 'ERP_US']].corr()

df_ERP[['ERP_EU', 'ERP_US']].mean()*4
df_ERP[['ERP_EU', 'ERP_US']].std()*2
df_ERP[['ERP_EU', 'ERP_US']].mean()*4/(df_ERP[['ERP_EU', 'ERP_US']].std()*2)

"""
Formula for the standard error of the Sharpe ratio
n    = 120
sr   = df_ERP[['ERP_EU', 'ERP_US']].mean()/(df_ERP[['ERP_EU', 'ERP_US']].std())
tmp  = sr/(np.sqrt((n-1)/2)*gamma((n-2)/2)/gamma((n-1)/2))
s.e. = np.sqrt((n-1)/(n-3)*(1+tmp**2) - sr**2)/np.sqrt(n)*2
"""

# %%
import statsmodels.api as sm

X     = sm.add_constant(df_rates['diff'])
y     = df_rates['crp']
model = sm.OLS(y, X).fit(cov_type='HC3')
model.summary()

"""
Fama-beta 2.1846
std err   1.250
t-beta    1.748
R2        0.017
"""