#!/usr/bin/env python
# coding: utf-8

# <b> Experimental Overview in Effort to Test Temperature's Effect on Fluid Travel: <br>
# 
# a) Formulation with 8% HSC & 8 % PNP in a solvent system 8:2 EtOH:H2O<br>
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
# 
# TEMP 37 SEAL:
# PV 15 0 60
# DELAY 250
# HTR  1 37
# FILL 8
# DELAY 4000
# PV 15 40 1.5
# DELAY 90000
# HTR 0
# DELAY 40000
# PV 15 0 60
# DELAY 4000
PV 15 60 60
# 
# </pre>
# 
#     

# In[27]:


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


# In[28]:


sys.path.append('/home/andrewd/jupyter/projects/custom_modules')
import adstat


# In[29]:


get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[30]:


db_params = {'host':'',
             'port':'',
             'dbname':'',
             'password':'',
             'user':'',
             'password':''
            }
conn = psycopg2.connect(**db_params)
cur = conn.cursor()

select_query = '''SELECT * FROM dispense.exp_20231025_center_seal
               WHERE percent_product_1 = 8
                '''
cur.execute(select_query)

tempin_data = cur.fetchall()
tempin_col = [col[0] for col in cur.description]

cur.close()
conn.close()


# In[31]:


df_tempin = pd.DataFrame(tempin_data, columns = tempin_col)


# In[32]:


df_tempin.sort_values(by=['temp_in','repetition'])


# In[33]:


get_ipython().run_cell_magic('R', '', '\nlibrary(dplyr)\nlibrary(ggplot2)\nlibrary(car)\nlibrary(vcd)\nlibrary(corrplot)\nlibrary(FSA)\nlibrary(tidyr)\nlibrary(tibble)\n')


# In[34]:


with (ro.default_converter + pandas2ri.converter).context():
    df_tempin_r = ro.conversion.get_conversion().py2rpy(df_tempin)


# <b> Looking at temperature's effect on fluid travel

# In[43]:


get_ipython().run_cell_magic('R', '-i df_tempin_r', "\nggplot(df_tempin_r) +\n    geom_bar(aes(x=factor(temp_in), fill=factor(fluid_front_final_location)),position='fill') +\n    labs(title='Pull-in Temperature\\'s Effect on Fluid Travel',\n         x='Temperature C')\n")


# <b> Wilcoxon Rank Sum Test with Normal Approximation  
# <pre>Ho: No statistically significant difference between distributions for 37c & 50c
# Hb: There is a stat-sig difference between the distributions
# Alpha = 0.05</pre>

# In[89]:


get_ipython().run_cell_magic('R', '', 'wilcox.test(fluid_front_final_location ~ temp_in, data = df_tempin_r, paired=FALSE, exact = FALSE)\n')


# <p> There is a statistically significant difference between medians/distributions in fluid_front_final_location for groups with pull-in temperatures of 37 & 50 c 

# <b> Temperature's Effect on Hydration Time

# In[42]:


get_ipython().run_cell_magic('R', '', "ggplot(df_tempin_r) +\n    geom_boxplot(aes(x=factor(temp_in),pf_seconds,color=factor(temp_in))) +\n    labs(title='Temperatures Effect on Hydration Time',\n         x = 'Temperature (c)',\n         y = 'Hydration Time (s)')\n")


# In[88]:


get_ipython().run_cell_magic('R', '', "ggplot(df_tempin_r,aes(x=pf_seconds, y = ..density..,)) +\n    geom_histogram(binwidth=4,color= 'grey') +\n    facet_grid(factor(temp_in) ~ .) +\n    geom_density(color='grey') +\n    labs(title='Sample Distributions of Hydration Time',\n         x='Hydration Time')\n         \n")


# In[71]:


get_ipython().run_cell_magic('R', '', 'sw_37 <- shapiro.test(df_tempin_r$pf_seconds[df_tempin_r$temp_in == 37])\nsw_50 <- shapiro.test(df_tempin_r$pf_seconds[df_tempin_r$temp_in == 50])\nprint(sw_37)\nprint(sw_50)\n')


# <p> Distribution of each sample is Gaussian, but sample sizes are < 30... so choosing a non parametric hypothesis test for 2 indpendent samples

# <b> Wilcoxon Rank Sum Test
# <pre>Ho: No statistically significant difference in hydration time distributions for 37c & 50c
# Hb: There is a stat-sig difference between the distributions
# Alpha = 0.05</pre>

# In[95]:


get_ipython().run_cell_magic('R', '', 'wilcox.test(pf_seconds ~ temp_in, data = df_tempin_r, paired=FALSE)\n')


# <p> There is not a statistically significant difference in the distribution of hydration times for different temperatures

# <b>Relationship Between Hydration Time & Fluid Filling Faceted by Temperature

# <p> Does the hydration time differ between final locations?

# In[115]:


get_ipython().run_cell_magic('R', '', "ggplot(df_tempin_r) +\n    geom_boxplot(aes(x=factor(fluid_front_final_location),y=pf_seconds, color = factor(temp_in))) +\n    facet_grid(~factor(temp_in)) +\n    labs(title='Distribution of Hydration Times across Ordinal Fluid Front Locations',\n         x='Ordinal Fluid Front Final Location',\n         y='Hydration Time')\n")


# <b> Kendall's Tau; detect the monotonic relations between hydration time & fluid front final location at 50c Temp-in

# In[123]:


get_ipython().run_cell_magic('R', '', "kt_cor <- df_tempin_r %>%\n    filter(temp_in == 50)%>%\n    select(pf_seconds, fluid_front_final_location) %>%\n    cor(method='kendall')\ncorrplot.mixed(kt_cor,\n               upper='shade',\n               order='AOE')\n")


# <b> Concluding Remarks
# <pre>-Temperature has an effect on the distribution of ordinal values in fluid front final location when considering 37 & 50c as independent groups; may allow fluid to travel further
# -Temperature has no statistically significant effect on the distribution of hydration times
# -Slightly positive correlation between hydration time and fluid front final location</pre>
