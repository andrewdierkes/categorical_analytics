#!/usr/bin/env python
# coding: utf-8

# In[519]:


import psycopg2
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import fisher_exact
from scipy.stats import boschloo_exact


# In[520]:


db_params = {'dbname':'',
             'user':'',
             'password':'',
             'host':'',
             'port':''
            }

conn = psycopg2.connect(**db_params)
cur = conn.cursor()

#Evaluting Mass & Percent for 1 dispense
select_hiper = '''SELECT * 
                FROM dispense.exp_20231003_center_seal
                WHERE num_dispense = 1 
                '''

cur.execute(select_hiper)

hiper_data = cur.fetchall()
hiper_colname = [col[0] for col in cur.description]

cur.close()
conn.close()    


# In[521]:


df_hiper = pd.DataFrame(hiper_data, columns=hiper_colname)
df_hiper.head()


# In[522]:


#clean data... changing masses to uniform values
df_hiper['single_mass_mg'] = df_hiper[(
    'single_mass_mg')].apply((
    lambda x: 0.7 if x == 0.715 else
    (0.5 if x == 0.480 else x)))


# In[523]:


#create new column with description of fluid_front_final_location
df_hiper['fluid_front_final_description'] = df_hiper['fluid_front_final_location'].apply(
    lambda x: 'filled_into_chamber' if x == 3 else 
    ('filled_past_dispense' if x == 2 else 
    ('filled_dispense' if x == 1 else 'partial_or_no_fill')))


# <b> Grouping Dataframe by percentage of product & mass dispensed to visualize overall trends in vent sealing with piezo's

# In[524]:


pf_piezo_dataset = df_hiper.groupby(['product_1',
                                     'percent_product_1',
                                     'single_mass_mg',
                                     'piezo_seal_test']).size().unstack().fillna(0)
pf_piezo_dataset


# In[525]:


#plotting work for df above
fig, ax = plt.subplots()

pf_piezo_dataset.plot(kind='bar',
                      stacked=True,
                      color=['red','green'],
                      ylabel='count',
                      title='Overview of Dataset Grouped By Mass & %',ax=ax);

ax.set_facecolor('lightblue')


# <b> Grouping Dataframe by percentage of product & mass dispensed to visualize overall trends in the amount of time it took for the spot to fully hydrate

# In[526]:


df_hiper['legend'] = df_hiper.apply(lambda row: f"{row['product_1']}_{str(row['percent_product_1'])}_{row['single_mass_mg']}",axis=1)


# In[527]:


#dataframe for scatter below... not used but efficent to view trends
df_hydration = df_hiper.groupby(['product_1',
                                      'percent_product_1',
                                      'single_mass_mg',
                                      'pf_seconds']).size().unstack().fillna(0)
display(df_hydration)


# In[528]:


#plotting work for hydration trends
fig, ax = plt.subplots(1,figsize=(4,4))
ax.scatter(df_hiper['legend'],df_hiper['pf_seconds'],c='orange')

#set ticks and labels
ticks = ax.get_xticks()
label = ax.get_xticklabels()

ax.set_xticks(ticks)
ax.set_xticklabels(label,rotation=90)

ax.set(ylabel='time_seconds',
       xlabel='product_percent_mass',
       title='Product, Percent & Mass Effect\'s on Spot Hydration Time Visualized')

ax.set_facecolor('lightblue');


# <b> Considering only HD640 Below:

# In[529]:


#new df for hd640 hiper
df_hiper_hd640 = df_hiper[df_hiper['product_1'].isin(['hd640'])]


# In[530]:


#percent product work
df_percent_hd640 = df_hiper_hd640.groupby(['percent_product_1',
                                       'piezo_seal_test'
                                      ]).size().unstack().fillna(0)

#mass work
df_mass_hd640 = df_hiper_hd640.groupby(['single_mass_mg',
                                'piezo_seal_test'
                               ]).size().unstack().fillna(0)

#combination
df_combo_hd640 = df_hiper_hd640.groupby(['percent_product_1',
                                 'single_mass_mg',
                                 'piezo_seal_test'
                                ]).size().unstack().fillna(0)

display(df_combo_hd640)
display(df_percent_hd640)
display(df_mass_hd640)


# In[531]:


#plots for percent & mass effect on channel sealing for hydroslipc
fig, axs = plt.subplots(1,3,figsize=(10,6))

df_percent_hd640.plot(kind='bar',
                    stacked=True,
                    color=['red','green'],
                    ylabel='count',
                    title='Percent HD640',
                    ax=axs[0])

df_mass_hd640.plot(kind='bar',
                 stacked=True,
                 color=['red','green'],
                 ylabel='count',
                 title='Dispense Mass (mg)',
                 ax=axs[1])

df_combo_hd640.plot(kind='bar',
                  stacked=True,
                  color=['red','green'],
                  ylabel='count',
                  title='Percent HD640 & Dispense Mass (mg) Effect',
                  ax=axs[2])

for plot in axs:
    plot.set_facecolor('lightblue')
    

fig.subplots_adjust(wspace=0.5)


# <b> Create Contingency Table for evaluation of relationship between Mass & Vent sealing for 10% HD640

# In[532]:


#looking at mass
ten_p = df_hiper_hd640[(
    df_hiper_hd640['percent_product_1'] == 10)
]

pf_mass_10_p = ten_p.groupby(
    ['piezo_seal_test',
     'single_mass_mg']).size().unstack().fillna(0)

pf_mass_10_p


# <b> Create Contingency Table for evaluation of relationship between Percent HD640 & Vent sealing

# In[533]:


#looking at percent
per_7 = df_hiper[(
    df_hiper['single_mass_mg'].isin([0.7,0.715]))]
pf_per_7 = per_7.groupby([
    'piezo_seal_test',
    'percent_product_1']).size().unstack().fillna(0)
pf_per_7


# <b>Plots for evaluations above

# In[534]:


fig, axs = plt.subplots(1,2)

pf_mass_10_p.plot(kind='bar',
                  stacked=True,
                  color=['red','green'], 
                  ylabel='count',
                  title='Mass Effect at 10% HD640',
                  ax=axs[0])

pf_per_7.plot(kind='bar',
              stacked=True,
              color=['red','green'],
              ylabel='count',
              title='Percent HD640 Effect at 0.7mg',
              ax=axs[1]);

for plot in axs:
    plot.set_facecolor('lightblue')


# <b> Perform Boschloo test to check if there is an association with mass & sealing for HD640

# In[535]:


#use boschloo instead of fishers as we only have a fixed number in our rows but not our columns (pass/fail is variable)
boschloo_outcome_mass_hd640 = boschloo_exact(pf_mass_10_p)
print(boschloo_outcome_mass)
print('There is an association between mass (0.5 v 0.7 mg) and sealing effectivity')


# <b> Perform Boschloo test to check if there is an association with percent & sealing

# In[536]:


boschloo_outcome_percent_hd640 = boschloo_exact(pf_per_7)
print(boschloo_outcome_percent_hd640)
print('There is an association between percent hd640 (8 vs 10 %) and sealing effectivity')


# <b> Visualize Effects of Mass and Percent on Time to Hydrate Dispense Spot

# In[537]:


fig, axs = plt.subplots(1,3,figsize=(8,4))

fig.suptitle('Visualizing the Effects of Mass & Percent HD640 on Spot Hydration')

#scatter plots
axs[0].scatter(df_hiper_hd640['legend'], 
               df_hiper_hd640['pf_seconds'],
               c='orange')

axs[1].scatter(df_hiper_hd640['percent_product_1'],
               df_hiper_hd640['pf_seconds'], 
               c='orange')

axs[2].scatter(df_hiper_hd640['single_mass_mg'],
               df_hiper_hd640['pf_seconds'], 
               c='orange')

#extract ticks and labels
ticks_0 = axs[0].get_xticks()
ticks_2 = axs[2].get_xticks()

label_0 = axs[0].get_xticklabels()
label_2 = axs[2].get_xticklabels()


#apply ticks and labels to axs
axs[0].set_xticks(ticks_0)
axs[0].set_xticklabels(label_0,rotation=90)

axs[2].set_xticks(ticks_2)
axs[2].set_xticklabels(label_2)


#set y labels
for plot in axs:
    plot.set(ylabel='time_seconds')

#set x labels
axs[0].set(xlabel='product_percent_mass')
axs[1].set(xlabel='percentage of product')
axs[2].set(xlabel='mass (mg)')

#set face color
for plot in axs:
    plot.set_facecolor('lightblue')

#adjust spacing
fig.subplots_adjust(wspace=0.5)


# <b> Taking a Look at the Final Location of the Fluid Front for HD640
# <p> This is important as we need enough fluid to pass through the spot to allow for ample sample. Below we have a scale and small description of the numbers
#             

# In[556]:


df_fluid_front_scale = pd.DataFrame({0:'partial or no fill of dispensed spot',
                                     1:'filled only the dispensed spot, but went no further',
                                     2:'filled thru dispensed spot and slightly past',
                                     3:'filled thru the dispensed spot and into the rxn chamber'
                                    },
                                   index=['description of fluid front\'s final location'])

df_fluid_front_scale.columns.set_names('numerical description from data',inplace=True)
df_fluid_front_scale                 


# <b>Effect of Percentage of HD640 on Fluid Front Location 

# In[539]:


df_fill_location_hd640 = df_hiper_hd640.groupby(['percent_product_1',
                                                 'fluid_front_final_description'
                                                 ]).size().unstack().fillna(0)
display(df_fill_location_hd640)


fig, axs = plt.subplots(1, 3, figsize=(25,7))

for i, (desc,data) in enumerate(df_fill_location_hd640.iterrows()):

    ax = axs[i]
    ax.pie(data, labels=data.index,autopct='%1.1f%%')
    ax.set(title = f'Pie Chart for {desc}% Product')
    ax.axis('equal')

fig.legend(df_fill_location_hd640.columns,loc=(0,0))
fig.subplots_adjust(wspace=0.5)


# <b>Effect of Mass of HD640 (mg) on Fluid Front Location 

# In[540]:


df_fill_location_hd640_mass = df_hiper_hd640.groupby(['single_mass_mg',
                                                  'fluid_front_final_description'
                                                 ]).size().unstack().fillna(0)
display(df_fill_location_hd640_mass)


# In[541]:


fig, axs = plt.subplots(1,3, figsize=(30,6))

for i, (desc,data) in enumerate(df_fill_location_hd640_mass.iterrows()):
    ax = axs[i]
    ax.pie(data, labels= data.index, autopct= '%1.1f%%')
    ax.set(title = f'Pie Chart for Dispense Mass of {desc} (mg)')
    ax.axis('equal')


# <b> Looking at Hydroslipc:

# In[542]:


#new df for hd640 hiper
df_hiper_hsc = df_hiper[df_hiper['product_1'].isin(['hydroslipc'])]


# <b> Taking a look at the Effects of Changing Hydroslipc Percentage as well as Dispense Mass (mg) on Channel Sealing

# In[543]:


#percent product work
df_percent_hsc = df_hiper_hsc.groupby(['percent_product_1',
                                       'piezo_seal_test'
                                      ]).size().unstack().fillna(0)

#mass work
df_mass_hsc = df_hiper_hsc.groupby(['single_mass_mg',
                                'piezo_seal_test'
                               ]).size().unstack().fillna(0)

#combination
df_combo_hsc = df_hiper_hsc.groupby(['percent_product_1',
                                 'single_mass_mg',
                                 'piezo_seal_test'
                                ]).size().unstack().fillna(0)
display(df_combo_hsc)
display(df_percent_hsc)
display(df_mass_hsc)


# In[544]:


#plots for percent & mass effect on channel sealing for hydroslipc
fig, axs = plt.subplots(1,3,figsize=(10,6))

df_percent_hsc.plot(kind='bar',
                    stacked=True,
                    color=['red','green'],
                    ylabel='count',
                    title='Percent Hydroslipc',
                    ax=axs[0])

df_mass_hsc.plot(kind='bar',
                 stacked=True,
                 color=['red','green'],
                 ylabel='count',
                 title='Dispense Mass (mg)',
                 ax=axs[1])

df_combo_hsc.plot(kind='bar',
                  stacked=True,
                  color=['red','green'],
                  ylabel='count',
                  title='Percent & Dispense Mass (mg) Effect',
                  ax=axs[2])

for plot in axs:
    plot.set_facecolor('lightblue')

fig.subplots_adjust(wspace=0.5)


# <b> Create Contingency Table for evaluation of relationship between Mass & Vent sealing for 8 Percent Hydroslipc

# In[545]:


#looking at mass
eight_p = df_hiper_hsc[(
    df_hiper_hsc['percent_product_1'] == 8)
]

pf_mass_8_p = eight_p.groupby(
    ['single_mass_mg',
     'piezo_seal_test',
    ]).size().unstack().fillna(0)

pf_mass_8_p


# <b> Create Contingency Table for evaluation of relationship between Percent Hydroslipc & Vent sealing

# In[546]:


#looking at percent
per_7_hsc = df_hiper[(
    df_hiper['single_mass_mg'].isin([0.7]))]

pf_per_7_hsc = per_7_hsc.groupby([
    'percent_product_1',
    'piezo_seal_test'
    ]).size().unstack().fillna(0)
pf_per_7_hsc


# <b>Plots for evaluations above

# In[547]:


fig, axs = plt.subplots(1,2)

pf_mass_8_p.plot(kind='bar',
                 stacked=True,
                 color=['red','green'],
                 ylabel='count',
                 title='Mass Effect At 8% Hydroslipc',
                 ax=axs[0])

pf_per_7.plot(kind='bar',
              stacked=True,
              color=['red','green'],
              ylabel='count',
              title='Percent HSC Effect At 0.7mg',
              ax=axs[1])

for plot in axs:
    plot.set_facecolor('lightblue')

fig.subplots_adjust(wspace=0.3)


# <b> Perform Boschloo test to check if there is an association with mass & sealing for Hydroslipc

# In[548]:


#use boschloo instead of fishers as we only have a fixed number in our rows but not our columns (pass/fail is variable)
boschloo_outcome_mass_hsc = boschloo_exact(pf_mass_8_p)
print(boschloo_outcome_mass_hsc)
print('There is no association between mass (0.5 v 0.7 mg) and sealing effectivity')


# <b> Perform Boschloo test to check if there is an association with percentage & sealing for Hydroslipc

# In[549]:


boschloo_outcome_per_hsc = boschloo_exact(pf_per_7)
print(boschloo_outcome_per_hsc)
print('There is an association between percentage Hydroslipc (8 versus 10 %) and sealing effectively')


# In[550]:


fig, axs = plt.subplots(1,3,figsize=(8,4))

fig.suptitle('Visualizing the Effects of Mass & Percent Hydroslipc on Spot Hydration')

#scatter plots
axs[0].scatter(df_hiper_hsc['legend'], 
               df_hiper_hsc['pf_seconds'],
               c='orange')

axs[1].scatter(df_hiper_hsc['percent_product_1'],
               df_hiper_hsc['pf_seconds'], 
               c='orange')

axs[2].scatter(df_hiper_hsc['single_mass_mg'],
               df_hiper_hsc['pf_seconds'], 
               c='orange')

#extract ticks and labels
ticks_0 = axs[0].get_xticks()
ticks_2 = axs[2].get_xticks()

label_0 = axs[0].get_xticklabels()
label_2 = axs[2].get_xticklabels()


#apply ticks and labels to axs
axs[0].set_xticks(ticks_0)
axs[0].set_xticklabels(label_0,rotation=90)

axs[2].set_xticks(ticks_2)
axs[2].set_xticklabels(label_2)


#set y labels
for plot in axs:
    plot.set(ylabel='time_seconds')

#set x labels
axs[0].set(xlabel='product_percent_mass')
axs[1].set(xlabel='percentage of product')
axs[2].set(xlabel='mass (mg)')

#set face color
for plot in axs:
    plot.set_facecolor('lightblue')

#adjust spacing
fig.subplots_adjust(wspace=0.5)


# <b> Taking a Look at the Final Location of the Fluid Front for Hydroslipc
# <p> This is important as we need enough fluid to pass thru the spot to allow for ample sample. Below we have a scale and small description of the numbers
#          

# In[551]:


df_fluid_front_scale = pd.DataFrame({0:'partial or no fill of dispensed spot',
                                     1:'filled only the dispensed spot, but went no further',
                                     2:'filled thru dispensed spot and slightly past',
                                     3:'filled thru the dispensed spot and into the rxn chamber'
                                    },
                                   index=['description of fluid front\'s final location'])

df_fluid_front_scale.columns.set_names('numerical description from data',inplace=True)
df_fluid_front_scale
                           


# <b>Effect of Percentage of Hydroslipc on Fluid Front Location 

# In[552]:


df_fill_location_hsc_per = df_hiper_hsc.groupby(['percent_product_1',
                                                 'fluid_front_final_description'
                                                 ]).size().unstack().fillna(0)
display(df_fill_location_hsc_per)


# In[553]:


fig, axs = plt.subplots(1,3, figsize=(25,7))

for i, (desc,data) in enumerate(df_fill_location_hsc_per.iterrows()):

    ax = axs[i]
    ax.pie(data, labels= data.index, autopct= '%1.1f%%',
          shadow=True)
    ax.set(title = f'Pie Chart for {desc}% Hydroslipc')
    ax.axis('equal')

fig.legend(df_fill_location_hd640.columns,loc=(0,0))
fig.subplots_adjust(wspace=0.5)


# <b> Effect of Mass of Dispense (Hydroslipc) on Fluid Front Location

# In[554]:


df_fill_location_hsc_mass = df_hiper_hsc.groupby(['single_mass_mg',
                                                  'fluid_front_final_description'
                                                 ]).size().unstack().fillna(0)
display(df_fill_location_hsc_mass)


# In[555]:


fig, axs = plt.subplots(1,2, figsize=(25,7))

for i, (desc,data) in enumerate(df_fill_location_hsc_mass.iterrows()):
    ax = axs[i]
    ax.pie(data, labels= data.index, autopct= '%1.1f%%',
          shadow=True, startangle=90)
    ax.set(title = f'Pie Chart for Dispense Mass of {desc} (mg) for Hydroslipc')
    ax.axis('equal')

