#!/usr/bin/env python
# coding: utf-8

# ### Author : ABDELMONEM ELMONGY
# ### Data     : 09/2023

# # Project : Diabetes Dataset
# This dataset is originally from the N. Inst. of Diabetes & Diges. & Kidney Dis.

# ## About Dataset:
# 
# ### Context:
# This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.
# 
# ### Content:
# Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
# 
#     Pregnancies: Number of times pregnant
#     Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#     BloodPressure: Diastolic blood pressure (mm Hg)
#     SkinThickness: Triceps skin fold thickness (mm)
#     Insulin: 2-Hour serum insulin (mu U/ml)
#     BMI: Body mass index (weight in kg/(height in m)^2)
#     DiabetesPedigreeFunction: Diabetes pedigree function
#     Age: Age (years)
#     Outcome: Class variable (0 or 1)
# 
# ### Sources:
# (a) Original owners: National Institute of Diabetes and Digestive and Kidney Diseases<br>
# 
# (b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu)
#     Research Center, RMI Group Leader<br>
#     Applied Physics Laboratory<br>
#     The Johns Hopkins University<br>
#     Johns Hopkins Road<br>
#     Laurel, MD 20707<br>
#     (301) 953-6231<br>
#     
# (c) Date received: 9 May 1990
# 
# [Source from kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)

# #### Number of Instances: 768
# #### Number of Attributes: 8 plus class
# #### For Each Attribute: (all numeric-valued)
# 
#     1. Number of times pregnant
#     2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#     3. Diastolic blood pressure (mm Hg)
#     4. Triceps skin fold thickness (mm)
#     5. 2-Hour serum insulin (mu U/ml)
#     6. Body mass index (weight in kg/(height in m)^2)
#     7. Diabetes pedigree function
#     8. Age (years)
#     9. Class variable (0 or 1)
# 
# #### Missing Attribute Values: Yes
# #### Class Distribution: (class value 1 is interpreted as "tested positive for diabetes")

# ## Table of Contents
# <ul>
#     <li><a href="#intro">Introduction</a></li>
#     <li><a href="#wrangling">Data Wrangling</a></li>
#     <li><a href="#eda">Exploratory Data Analysis</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction

# #### Your task is to:
# 
# 01 - Gathering data.
# 
# 02 - Assessing data.
# 
# 03 - Cleaning data.
# 
# 04 - Storing data.
# 
# 05 - Analyzing, and visualizing data.
# 
# 06 - Reporting
# 
#         your data wrangling efforts
#         your data analyses and visualizations

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

get_ipython().run_line_magic('matplotlib', 'inline')

import os


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > **Tip**: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you **document your data cleaning steps in mark-down cells precisely and justify your cleaning decisions.**
# 

# ## Data Wrangling which include :
# <li><a href="#Gathering">Gathering Data</a></li>
# <li><a href="#Assessing">Assessing Data</a></li>
# <li><a href="#Cleaning">Cleaning Data</a></li>

# ## Gathering Data
# <a id='Gathering'></a>

# ### 1.Importing Enhanced Twitter Archive

# we aqucistion data from dataset like : csv file in our example

# In[2]:


# Load your data and print out a few lines. Perform operations to inspect data
# types and look for instances of missing or possibly errant data.
diabetes = pd.read_csv('data/diabetes.csv')


# we use head() or tail() function to display a sample of data 

# In[3]:


diabetes.head()


# In[4]:


diabetes.tail()


# ## Assessing Data
# <a id='Assessing'></a>

# ### 1.Assessing Enhanced Twitter Archive

# We assessing our data using some function like : shape , ndim , dtypes , size  , info() , nunique() , isnull()

# In[5]:


# return number of columns and number of row
diabetes.shape


# the shape function get number of rows and number of columns in tuple

# In[6]:


#return number of dimensions of data
diabetes.ndim


# size function show us the result of multiplication of number of rows and number of columns

# In[7]:


# return size of Dataset which is a multiplication of number of rows and number of columns
diabetes.size


# dtypes show us data type of each column (features)

# In[8]:


diabetes.head(1)


# In[9]:


#return types of each column
diabetes.dtypes


# info() function show us number of non_null_value in each column and datatype
# 
# it has two features(no of non_null_value,datatype)

# In[10]:


#return number of non-null-value and datatype of each column
diabetes.info()


# nunique() show us number of unique values in each column

# In[11]:


#return number of unique value
diabetes.nunique()


# In[12]:


Insulin_value_counts = diabetes.Insulin.value_counts()
Insulin_value_counts


# In[13]:


BloodPressure_value_counts = diabetes.BloodPressure.value_counts()
BloodPressure_value_counts


# In[14]:


SkinThickness_value_counts = diabetes.SkinThickness.value_counts()
SkinThickness_value_counts


# In[15]:


Pregnancies_value_counts = diabetes.Pregnancies.value_counts()
Pregnancies_value_counts


# In[16]:


Glucose_value_counts = diabetes.Glucose.value_counts()
Glucose_value_counts


# In[17]:


DiabetesPedigreeFunction_value_counts = diabetes.DiabetesPedigreeFunction .value_counts()
DiabetesPedigreeFunction_value_counts


# In[18]:


BMI_value_counts = diabetes.BMI.value_counts()
BMI_value_counts


# In[19]:


Outcome_value_counts = diabetes.Outcome.value_counts()
Outcome_value_counts


# In[20]:


Age_value_counts = diabetes.Age.value_counts()
Age_value_counts


# isnull() function show us boolean value for each element (each cell) it is null or not
# 
# if it(element) null return True
# 
# else  return False

# In[21]:


# return which value is nul or not for each element in DataSet 
diabetes.isnull()


# isnull().any() function show us boolean value for each column it is null or not
# 
# if column null return True
# 
# else  return False

# In[22]:


# return which value is nul or not for each columns in DataSet 
diabetes.isnull().any()


# isnull().any() function show us boolean value for each column it is null or not
# 
# if column null return 1
# 
# else  return 0

# In[23]:


#return number of columns has a null value
diabetes.isnull().any().sum()


# In[24]:


#return number of null value for each column
diabetes.isnull().sum()


# In[25]:


#return a number of cell has a null value
diabetes.isnull().sum().sum()


# describe() function show us descriptive statistical value for each column
# 
# in 8 value such as: count element in each column
#     
#                     mean : getting average in each column
#                     
#                     std : standarded deviation
#                         
#                     min : minimum value in each column
#                         
#                     max : maximum value in each column
#                         
#                     50% : median of value for each column

# In[26]:


#return statistical descriptive of dataset for each column
diabetes.describe(include = 'all')


# ## Cleaning Data
# <a id='Cleaning'></a>

# ### Data Cleaning 

# #### ISSUSES

# 1.duplicated data
# 
# 2.missing value
# 
# 3.incorrect datatype

# from assessing no NULL value
# 
# we will check for missing value and incorrect datatype

# #### Code

# In[27]:


clean_diabetes = diabetes.copy()


# #### Test

# In[28]:


clean_diabetes.head(1)


# ##### Issues 1#

# ## cleanning tidiness issuses

# ### duplicted value

# #### code

# In[29]:


clean_diabetes.duplicated()


# ##### Test

# In[30]:


clean_diabetes.duplicated().sum()


# ##### Define

# ## cleaning quality issuses 2#

# In[31]:


clean_diabetes.info()


# In[32]:


clean_diabetes.head(1)


# In[33]:


sum(clean_diabetes.isnull().any())


# ### incorrect datatype

# # NON Incorrect datatype

# #### checking

# In[34]:


clean_diabetes.dtypes


# In[35]:


clean_diabetes.info()


# In[36]:


new_diabetes_data = clean_diabetes.copy()


# # storing Data

# In[37]:


import os

directory_path = "./new data/"
os.makedirs(directory_path, exist_ok=True)


# In[38]:


new_diabetes_data.to_csv("./new data/new_diabetes_data.csv",index = False);


# #### DONE
# #### non_dublicated data....non_missing value....non incorrect datatype

# <a id='eda'></a>
# # Analysis and visualization
# 
# ## Exploratory Data Analysis
# 
# > **Tip**: Now that you've trimmed and cleaned your data, you're ready to move on to exploration. **Compute statistics** and **create visualizations** with the goal of addressing the research questions that you posed in the Introduction section. You should compute the relevant statistics throughout the analysis when an inference is made about the data. Note that at least two or more kinds of plots should be created as part of the exploration, and you must  compare and show trends in the varied visualizations. 

# ### load our cleaned data

# In[39]:


# Load your data and print out a few lines. Perform operations to inspect data
# types and look for instances of missing or possibly errant data.
df = pd.read_csv('new data/new_diabetes_data.csv')


# using hist() function to draw Histogram for each column 
# 
# and use figsize(,) parameter to show it obviously

# In[40]:


# Use this, and more code cells, to explore your data. Don't forget to add
# Markdown cells to document your observations and findings.

df.hist(figsize = (30,18));


# using plotting.scatter_matrix() function to draw Histogramsfor each column 
# 
# and scatter plotting between numerical columns 
# 
# and use figsize(,) parameter to show it obviously

# In[41]:


pd.plotting.scatter_matrix(df, figsize = (20,20));


# In[42]:


df.info()


# In[43]:


def linechart(x,y,title):
    plt.plot(x, y)
    plt.xlabel("X-axis")  # add X-axis label
    plt.ylabel("Y-axis")  # add Y-axis label
    plt.title(title)  # add title
    plt.show()


# In[44]:


linechart(df.BMI,df.Insulin,"relationship between Insulin and BMI")


# In[45]:


linechart(df.BMI,df.Age,"relationship between Age and BMI")


# In[46]:


df.Outcome.value_counts().plot(kind='bar');
#df.City.value_counts().plot(kind='barh');


# In[49]:


display(df.corr())


# In[47]:


plt.figure(figsize = (10,8))
sns.heatmap(df.corr(),annot = True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Creating a pie chart
plt.pie (stage_df,
         labels = ['Pupper','Doggo','Puppo','Floofer'],
         autopct = "%1.1f%%",
         shadow = True,
         explode = (0.1,0.2,0.2,0.3)
        )
plt.title ("Percentage of dog stages")
plt.axis ('equal') 


# In[ ]:


plt.scatter(clean_t_archive.retweet_count , clean_t_archive.favorite_count) 
plt.title ("Relationship between retweet count and favorite count") 
plt.xlabel ("Retweet Count") 
plt.ylabel ("Favorite Count")


# In[ ]:


import os

directory_path = "./output/"
os.makedirs(directory_path, exist_ok=True)


# In[ ]:


get_ipython().system('jupyter nbconvert --to html --no-input sales.ipynb  ')


# In[ ]:


get_ipython().system('jupyter nbconvert --to latex sales.ipynb  ')


# In[ ]:


get_ipython().system('jupyter nbconvert --to html  sales.ipynb  ')


# In[ ]:


get_ipython().system('jupyter nbconvert --to python sales.ipynb  ')

