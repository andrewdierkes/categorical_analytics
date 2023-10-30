#!/usr/bin/env python
# coding: utf-8

# <b> Experimental Overview in Effort to Ensure Reproducibility: <br>
# 
# a) 2 seperate solutions were formulated on different days; with 8% HSC & 8 % PNP in a solvent system 8:2 EtOH:H2O<br>
# <pre>    -solutions were mixed at 550RPM, 55C for 2 hours
#     -dispensing occured using the fisnar and I targeted 0.7 mg per dispense
#     -parameters in postgres database accessible on metabolics DB / dispense schemea
#     -drying occured in the dry keeper for 300 seconds</pre>
# 
#         
# b) Meter Script Ran: <br>
# <pre>TEMP 50 SEAL:
# PV 15 0 60
# DELAY 250
# HTR  1 50 50000
# FILL 8
# DELAY 4000
# PV 15 40 1.5
# DELAY 90000
# HTR 0
# DELAY 180000
# PV 15 0 60
# DELAY 4000
# PV 15 60 60
# </pre>
# ture0 60
#     

# In[51]:


import psycopg2
import pandas as pd
import rpy2

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import re
import os
import sys
import pathlib as path

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.cm as cm

from scipy.signal import find_peaks
from scipy import stats

from datetime import date
import pingouin as pg
from itertools import combinations


# In[52]:


sys.path.append('/home/andrewd/jupyter/projects/custom_modules')
import adstat


# In[53]:


get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[54]:

db_params = {'host':'',
             'port':'',
             'dbname':'',
             'password':'',
             'user':'',
             'password':''
            }

conn = psycopg2.connect(**db_params)
cur = conn.cursor()

select_query = '''SELECT '20231023' AS date, repetition, product_1, percent_product_1, product_2, percent_product_2, center_seal, fluid_front_final_location, pf_seconds, temp_in, solvent
                FROM dispense.exp_20231023_center_seal
                WHERE percent_product_1 = 8
                UNION ALL
                SELECT '20231025' AS date,repetition, product_1, percent_product_1, product_2, percent_product_2, center_seal, fluid_front_final_location, pf_seconds, temp_in, solvent
                FROM dispense.exp_20231025_center_seal
                WHERE temp_in = 50 AND percent_product_1 = 8
                '''
cur.execute(select_query)

date_data = cur.fetchall()
date_col = [col[0] for col in cur.description]

cur.close()
conn.close()


# In[55]:


df_date = pd.DataFrame(date_data, columns = date_col)


# In[121]:


df_date.sort_values(by=['date','repetition']).head()


# In[57]:


get_ipython().run_cell_magic('R', '', '\nlibrary(dplyr)\nlibrary(ggplot2)\nlibrary(car)\nlibrary(vcd)\nlibrary(corrplot)\nlibrary(FSA)\nlibrary(tidyr)\nlibrary(tibble)\n')


# In[58]:


with (ro.default_converter + pandas2ri.converter).context():
    df_date_r = ro.conversion.get_conversion().py2rpy(df_date)


# <b> Looking at the sealing effectivity & fluid traveling capabilities of formulations made different days:

# In[143]:


get_ipython().run_cell_magic('R', '', "ggplot(df_date_r) +\n    geom_bar(aes(x=factor(date),fill=factor(center_seal)),position='fill') +\n    labs(title='Center Sealing on Dates 20231023 & 20231025 (n=20)',\n         x='Date Formulated')\n")


# In[144]:


get_ipython().run_cell_magic('R', '-i df_date_r', "ggplot(df_date_r) +\n    geom_bar(aes(x=factor(date), fill=factor(fluid_front_final_location)), position='fill') +\n    labs(title='Fluid Front Final Location By Date (n=20)',\n         x='Date Formulated')\n")


# <b>Wilcoxon Rank Sum with Normal Approximation
# <pre>Ho: There is not a statistically significant difference for fluid_front_final_location distributions between dates 20231023 & 20231025
# Hb: There is a stat-sig difference for fluid_front_final_location distributions between dates
# Alpha = 0.05</pre>

# In[119]:


get_ipython().run_cell_magic('R', '', 'wilcox.test(fluid_front_final_location ~ date, data = df_date_r, exact = FALSE)\n')


# <p> With a p-value of 0.08, we accept our null hypothesis; there is not a statistically significant difference for fluid_front_final_location distributions between dates 20231023 & 20231025. 

# <b> Considering Hydration Distribution for Different Batches

# In[138]:


get_ipython().run_cell_magic('R', '', "\nggplot(df_date_r,aes(x=pf_seconds, y = ..density..)) +\n    geom_histogram(binwidth=3) +\n    facet_grid(date ~ .) +\n    labs(title='Distribution of Hydration Time between Dates',\n         x='Hydration Time')\n")


# <b>Wilcoxon Rank Sum with Normal Approximation
# <pre>Ho: There is not a statistically significant difference for hydration time distributions between different batches/dates
# Hb: There is a stat-sig difference for hydration time distributions between different batches/dates
# Alpha = 0.05</pre>

# In[145]:


get_ipython().run_cell_magic('R', '', 'wilcox.test(pf_seconds ~ date, data = df_date_r, exact = FALSE)\n')


# <p> Accept null hypothesis; there is no difference in distributions of hydration time between batches formulated on different dates

# <b> Is there a difference in hydration times between fluid front final locations 6 & 7?

# In[99]:


get_ipython().run_cell_magic('R', '', "\n#convert fluid_front to a factor & create new column\ndf_date_r$fluid_front_factor <- factor(df_date_r$fluid_front_final_location)\n\n#plot distribution between two \nggplot(df_date_r) +\n    geom_point(aes(x=fluid_front_factor, y= pf_seconds, color=factor(date)))+\n    geom_boxplot(aes(x=fluid_front_factor, y= pf_seconds),\n                 alpha=0.2) +\n    labs(title='Distribution Between Different Fill Positions',\n         x='Fluid Front Final Location',\n         y='Hydration Time')\n")


# In[126]:


get_ipython().run_cell_magic('R', '', "ggplot(df_date_r,aes(x=pf_seconds, y = ..density..)) +\n    geom_histogram(bins=4) +\n    geom_density(color='grey') +\n    facet_grid(factor(fluid_front_final_location) ~ .) +\n    labs(title='Hydration Time Distribution Divided by Fluid Front Factor',\n         x='Hydration Time')\n")


# <b>Wilcoxon Rank Sum with Normal Approximation
# <pre>Ho: There is not a statistically significant difference for hydration time distributions between fluid front final locations 6 & 7
# Hb: There is a stat-sig difference for fluid_front_final_location distributions
# Alpha = 0.05</pre>

# In[132]:


get_ipython().run_cell_magic('R', '', '\nwilcox.test(pf_seconds ~ fluid_front_final_location, data = df_date_r, exact=FALSE)\n')


# <p> Not a statistically significant difference for hydration time distributions between 6 & 7 fluid front final locations

# <b> Concluding Remarks
# <pre>-No statistically significant differences between the distributions of hydration, filling, & sealing between batches made on different days</pre>

# In[ ]:




