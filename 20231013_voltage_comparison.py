#!/usr/bin/env python
# coding: utf-8

# In[98]:


import psycopg2
import pandas as pd
import rpy2

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

import matplotlib as mpl
import matplotlib.pyplot as plt


# In[82]:


get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[134]:


db_params = {'host':'',
             'port':'',
             'dbname':'',
             'password':'',
             'user':'',
             'password': ''
            }

conn = psycopg2.connect(**db_params)
cur = conn.cursor()

select_query = '''SELECT product_1, percent_product_1, product_2, percent_product_2, piezo_ctrl, fluid_front_final_location
                FROM dispense.exp_20231011_center_seal
                UNION ALL
                SELECT product_1, percent_product_1, product_2, percent_product_2, piezo_ctrl, fluid_front_final_location
                FROM dispense.exp_20231012_center_seal
                WHERE single_mass_mg = 0.7
                '''
cur.execute(select_query)

voltage_data = cur.fetchall()
voltage_col = [col[0] for col in cur.description]

cur.close()
conn.close()


# In[135]:


df_voltage = pd.DataFrame(voltage_data, columns = voltage_col)


# In[136]:


#clean data
df_voltage['percent_product_2'] = df_voltage['percent_product_2'].apply(
    lambda x: 0 if pd.isna(x) else x)

df_voltage['product_2'] = df_voltage['product_2'].apply(
    lambda x: 'control' if pd.isna(x) else x)

df_voltage['voltage'] = df_voltage['piezo_ctrl'].apply(
    lambda x: 1.5 if x == 'pv_15_40_1.5'
    else 3)

df_voltage['total'] = df_voltage.apply(lambda row: f"{row['product_2']}_{row['percent_product_2']}",axis=1)

df_voltage.head()


# In[137]:


#convert to R
with (ro.default_converter + pandas2ri.converter).context():
    df_voltage_r = ro.conversion.get_conversion().py2rpy(df_voltage)


# <b> Comparing Voltage & Fluid Filling Among Different Formulations

# In[149]:


get_ipython().run_cell_magic('R', '-i df_voltage_r', "library(ggplot2)\nlibrary(dplyr)\n\nglimpse(df_voltage_r)\n\nggplot(data=df_voltage_r, aes(x=total,y= fluid_front_final_location)) +\n    geom_point(aes(x=total,y= fluid_front_final_location, color = voltage)) +\n    geom_boxplot(fill = 'blue', color='black', alpha = 0.2) +\n    scale_color_gradient(low='green',high='red')\n    \n")


# In[151]:


get_ipython().run_cell_magic('R', '-i df_voltage_r', "\nlibrary(dplyr)\nlibrary(corrplot)\nlibrary(vcd)\n\n#select desired variables, in our case we are considering voltage and the fluid front position\n#rename for ease of prez\n\nvolt_cor <- df_voltage_r %>%\n        select(percent_product_2, fluid_front_final_location, voltage) %>%\n        rename(percent_pdt = percent_product_2,\n              fluid_distance = fluid_front_final_location)\n\nvolt_mos <- df_voltage_r %>%\n        select(product_2, percent_product_2, fluid_front_final_location, voltage) %>%\n        rename(percent_pdt = percent_product_2,\n               fluid_distance = fluid_front_final_location)\n\n\n\n#ordinal data use spearman... check for correlation b/t var\nspman_cor <- cor(volt_cor, \n               method= 'spearman')\n\n#visualize correlation\ncorrplot.mixed(spman_cor,\n               upper = 'shade',\n               order ='AOE')\n")


# In[ ]:




