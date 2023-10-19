

import psycopg2
import pandas as pd
import rpy2

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

import matplotlib as mpl
import matplotlib.pyplot as plt


# In[333]:


get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[334]:


db_params = {'host':'',
             'port':'',
             'dbname':'',
             'password':'',
             'user':'',
             'password':''
            }

conn = psycopg2.connect(**db_params)
cur = conn.cursor()

select_query = '''SELECT *
                FROM dispense.exp_20231018_center_seal
                '''
cur.execute(select_query)

temp_data = cur.fetchall()
temp_col = [col[0] for col in cur.description]

cur.close()
conn.close()


# In[335]:


df_temp = pd.DataFrame(temp_data, columns = temp_col)


# In[336]:


df_temp['temperature_c'] = df_temp['temperature_c'].apply(
    lambda x: 30 if x == 28 else x)
df_temp.head()


# <b> Relationships to consider: <br>
# <pre></pre> 
#     % pnp - center seal <br>
#     % pnp - pf_seconds <br>
#     % pnp = fluid front final location <br>
# 
#     temp - center seal <br>
#     temp - pf_seconds <br>
#     temp - fluid front final  <br>
# 
#     pf_seconds - fluid front final location <br>
# 
#     temp / % pnp - fluid front final location

# In[337]:


with (ro.default_converter + pandas2ri.converter).context():
    df_temp_r = ro.conversion.get_conversion().py2rpy(df_temp)


# <b> Mosaic Plot for Categorical Distribution of Groupings with different Percentage's of PNP, Temperatures & their effects on Fluid Front Final Location

# In[338]:


get_ipython().run_cell_magic('R', '-i df_temp_r', 'library(dplyr)\nlibrary(ggplot2)\nlibrary(car)\nlibrary(vcd)\nlibrary(corrplot)\n\n\ntemp <- df_temp_r %>%\n    dplyr::select(percent_product_2, temperature_c, fluid_front_final_location)\n\nscatterplotMatrix(temp,smooth=FALSE)\n\nmosaic_ftable <- ftable(temp)\nprint(mosaic_ftable)\n\nmosaic(~percent_product_2 + fluid_front_final_location + temperature_c,\n       data = mosaic_ftable,\n       shade = FALSE)\n')


# <b> Considering Hydration Time and it's effect on Fluid Front's Final Location; seperated by percentage of pnp

# In[365]:


get_ipython().run_cell_magic('R', '', "\nggplot(data = df_temp_r, aes(x = pf_seconds, y = fluid_front_final_location)) +\n    geom_point(aes(x= pf_seconds, y = fluid_front_final_location, color = temperature_c, shape = center_seal)) +\n    scale_shape_manual(values = c(8,15)) +\n    scale_color_gradient(low = 'blue', high = 'red') +\n    labs(title = 'Hydration Time vs. Fluid Front Final Location',\n         x = 'hydration_time') +\n    facet_wrap(~percent_product_2)\n")


# <b> Grouping of temperature versus fluid final front location

# In[340]:


df_temp.groupby(['temperature_c',
                 'percent_product_2',
                 'fluid_front_final_location'
                ]).size().unstack().fillna(0)


# <b> Box plot for Fluid Front vs. Spot Hydration Time

# In[341]:


get_ipython().run_cell_magic('R', '', "\nggplot(data = df_temp_r, aes(x=factor(fluid_front_final_location), y=pf_seconds)) +\n    geom_boxplot(width = 0.2) +\n    facet_wrap(~temperature_c, nrow=1) +\n    labs(title = 'Exploring Relationships Between Spot Hydration Time & Fluid Filling',\n        x = 'fluid_front_final_location',\n        y= 'hydration_time_seconds')\n")


# <b> Negative correlation seen at 37c

# In[342]:


get_ipython().run_cell_magic('R', '', 'cor.test(df_temp_r$fluid_front_final_location, df_temp_r$pf_seconds, method = 'kendall')\n')


# <p> Not a statisitically significant correlation at 37C

# <b>Distribution of Spot Hydration And Associated Temperatures

# In[343]:


get_ipython().run_cell_magic('R', '', "\nggplot(data = df_temp_r) +\n    geom_histogram(aes(x=pf_seconds, fill =factor(temperature_c)),bins=5) +\n    labs(title='Distribution Plot of Hydration Time & Temperature\\'s Effect')\n")


# <p> We seem quicker hydration at 37C... not statistically proven (working here; partial correlation)
# #install.packages("ppcor")

# <b> Percent PNP and Fluid Front Final Location

# In[195]:


get_ipython().run_cell_magic('R', '', '\nggplot(data = df_temp_r, aes(x = factor(percent_product_2), fill = factor(fluid_front_final_location))) +\n    geom_bar(position= "fill") + \n    labs(title = \'% PNP vs Fluid Front Final Location\',\n         x = \'% PNP\',\n         y = \'Proportion\')\n')


# <b> Percent PNP vs. Spot Hydration <br>
# <p>Looking at the distribution, median, Q1, Q3 & Outliers

# In[357]:


get_ipython().run_cell_magic('R', '', "\nggplot(data = df_temp_r, aes(x = factor(percent_product_2), y = pf_seconds)) +\n    geom_violin(fill = 'gold',\n                width = 1) +\n    geom_boxplot(width = 0.05,\n                 fill = 'lightblue') +\n    labs(title = 'Percent PNP vs. Spot Hydration Time',\n         x = 'percent_pnp',\n         y = 'spot_hydration_time')\n")


# <b> Percent PNP vs. Center Seal

# In[203]:


get_ipython().run_cell_magic('R', '', "\nggplot(data = df_temp_r, aes(x=factor(percent_product_2), fill = center_seal)) + \n    geom_bar(position='fill') +\n    labs(title='Percent PNP vs. Center Seal',\n         x = 'percent_pnp',\n         y = 'Proportion')\n")


# <b> Percent PNP vs. Rear Fluid Leak <br>
# <p> Scale (when piezos acutated to 0; pv 15 0 60): <br>
#     
# Rear fluid front has... <br>
# 3: no movement <br>
# 2: smallest bit of movement <br>
# 1: slight movement in channel <br>
# 0: movement back into the sample port <br>

# In[209]:


get_ipython().run_cell_magic('R', '', "\nggplot(data=df_temp_r, aes(x=factor(percent_product_2), fill = factor(rear_fluid_leak))) +\n    geom_bar(position='fill') +\n    labs(title='Percent PNP vs. Rear Fluid Leak',\n         x='percent_pnp',\n         y='proportion')\n")


# <b> Temperature vs. Center Seal

# In[210]:


get_ipython().run_cell_magic('R', '', "ggplot(data = df_temp_r, aes(x= factor(temperature_c), fill = center_seal)) +\n    geom_bar(position = 'fill')\n")


# <b> Temperature vs. Fluid Front Final Location

# In[211]:


get_ipython().run_cell_magic('R', '', "\nggplot(data = df_temp_r, aes(x=factor(temperature_c), fill = factor(fluid_front_final_location))) +\n    geom_bar(position='fill') +\n    labs(title = 'Temperature vs. Fluid Front Final Location',\n         x = 'temperature_c',\n         y = 'proportion')\n")


# <b> Temperature vs. Hydration Time

# In[212]:


get_ipython().run_cell_magic('R', '', "\nggplot(data = df_temp_r,temp aes(x= factor(temperature_c), y= pf_seconds)) +\n    geom_boxplot(fill='lightblue') +\n    geom_point(aes(color=factor(percent_product_2))) +\n    labs(title = 'Temperature vs. Hydration Time',\n         x = 'temperature_c',\n         y = 'hydration_time')\n")


# <b> Interpret Correlations between Continous (Percent & Temp) and Ordinal Data (Fluid Front Final Location)

# In[315]:


get_ipython().run_cell_magic('R', '', "\n#create df\npnp_temp_fluid_move <- df_temp_r %>%\n    dplyr::select(percent_product_2, temperature_c, fluid_front_final_location) %>%\n    rename(percent_pnp = percent_product_2) %>%\n    glimpse()\n\n#spearman correlation coeff\nsp_cor <- cor(pnp_temp_fluid_move, method = 'spearman')\nprint(sp_cor)\n\ncorrplot.mixed(sp_cor,\n               upper = 'shade',\n               order = 'AOE')\n")


# In[358]:


get_ipython().run_cell_magic('R', '', "kt_matrix <- scatterplotMatrix(pnp_temp_fluid_move,method='kendall', smooth=FALSE)\nkt_matrix\n")


# In[ ]:




