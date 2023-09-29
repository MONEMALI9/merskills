#!/usr/bin/env python
# coding: utf-8

# ### Author : ABDELMONEM ELMONGY
# ### Data     : 09/2023

# # Project : Sales Data Analysis
# Sales of products for 12 months period.

# ## About Dataset:
# 
# The dataset consists of 11 columns, each column representing an attribute of purchase on a product -
#     
#     Order ID - A unique ID for each order placed on a product
#     Product - Item that is purchased
#     Quantity Ordered - Describes how many of that products are ordered
#     Price Each - Price of a unit of that product
#     Order Date - Date on which the order is placed
#     Purchase Address - Address to where the order is shipped
#     Month, Sales, City, Hour - Extra attributes formed from the above.
# 
# ### Sources:
# 
# #### Acknowledgements
# Dataset is downloaded and compiled from KeithGalli's GitHub repository on Pandas Data Science Tasks.
# You find and access the repository [here](https://github.com/KeithGalli/Pandas-Data-Science-Tasks)
# 
# #### Inspiration
# A Dataset to practice basic EDA and Cleaning.
# 
# [Source from kaggle](https://www.kaggle.com/datasets/beekiran/sales-data-analysis)

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
sales = pd.read_csv('data/Sales Data.csv')


# we use head() or tail() function to display a sample of data 

# In[3]:


sales.head()


# In[4]:


sales.tail()


# ## Assessing Data
# <a id='Assessing'></a>

# ### 1.Assessing Enhanced Twitter Archive

# We assessing our data using some function like : shape , ndim , dtypes , size  , info() , nunique() , isnull()

# In[5]:


# return number of columns and number of row
sales.shape


# the shape function get number of rows and number of columns in tuple

# In[6]:


#return number of dimensions of data
sales.ndim


# size function show us the result of multiplication of number of rows and number of columns

# In[7]:


# return size of Dataset which is a multiplication of number of rows and number of columns
sales.size


# dtypes show us data type of each column (features)

# In[8]:


sales.head(1)


# In[9]:


#return types of each column
sales.dtypes


# info() function show us number of non_null_value in each column and datatype
# 
# it has two features(no of non_null_value,datatype)

# In[10]:


#return number of non-null-value and datatype of each column
sales.info()


# nunique() show us number of unique values in each column

# In[11]:


#return number of unique value
sales.nunique()


# In[12]:


City_value_counts = sales.City.value_counts()
City_value_counts


# In[13]:


Product_value_counts = sales.Product.value_counts()
Product_value_counts


# In[14]:


Month_value_counts = sales.Month.value_counts()
Month_value_counts


# In[15]:


Hour_value_counts = sales.Hour.value_counts()
Hour_value_counts


# isnull() function show us boolean value for each element (each cell) it is null or not
# 
# if it(element) null return True
# 
# else  return False

# In[16]:


# return which value is nul or not for each element in DataSet 
sales.isnull()


# isnull().any() function show us boolean value for each column it is null or not
# 
# if column null return True
# 
# else  return False

# In[17]:


# return which value is nul or not for each columns in DataSet 
sales.isnull().any()


# isnull().any() function show us boolean value for each column it is null or not
# 
# if column null return 1
# 
# else  return 0

# In[18]:


#return number of columns has a null value
sales.isnull().any().sum()


# In[19]:


#return number of null value for each column
sales.isnull().sum()


# In[20]:


#return a number of cell has a null value
sales.isnull().sum().sum()


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

# In[21]:


#return statistical descriptive of dataset for each column
sales[['Quantity Ordered','Price Each','Sales']].describe(include = 'all')


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

# ##### Issues 1#

# #### Code

# In[22]:


clean_sales = sales.copy()


# ### Drop unnesscery column 

# In[23]:


# Remove column name 'Unnamed: 0'
clean_sales.drop(['Unnamed: 0'], axis=1 , inplace = True)


# #### Test

# In[24]:


clean_sales.head(1)


# ## cleanning tidiness issuses

# ### duplicted value

# #### code

# In[25]:


clean_sales.duplicated()


# In[26]:


clean_sales.duplicated().any()


# In[27]:


clean_sales.duplicated().any().sum()


# In[28]:


clean_sales[['Order ID']].duplicated().sum()


# In[29]:


clean_sales.duplicated().sum()


# In[30]:


clean_sales.drop_duplicates(inplace = True);


# ##### Test

# In[31]:


clean_sales.duplicated().sum()


# In[32]:


clean_sales[['Order ID']].duplicated().sum()


# ##### Define

# ## cleaning quality issuses 2#

# In[33]:


clean_sales.info()


# In[34]:


clean_sales.head(1)


# In[35]:


sum(clean_sales.isnull().any())


# ### incorrect datatype

# # NON Incorrect datatype

# #### checking

# In[36]:


clean_sales.dtypes


# In[37]:


clean_sales.info()


# In[38]:


new_sales_data = clean_sales.copy()


# # storing Data

# In[39]:


import os

directory_path = "./new data/"
os.makedirs(directory_path, exist_ok=True)


# In[40]:


new_sales_data.to_csv("./new data/new_sales_data.csv",index = False);


# #### DONE
# #### non_dublicated data....non_missing value....non incorrect datatype

# <a id='eda'></a>
# # Analysis and visualization
# 
# ## Exploratory Data Analysis
# 
# > **Tip**: Now that you've trimmed and cleaned your data, you're ready to move on to exploration. **Compute statistics** and **create visualizations** with the goal of addressing the research questions that you posed in the Introduction section. You should compute the relevant statistics throughout the analysis when an inference is made about the data. Note that at least two or more kinds of plots should be created as part of the exploration, and you must  compare and show trends in the varied visualizations. 

# ### load our cleaned data

# In[41]:


# Load your data and print out a few lines. Perform operations to inspect data
# types and look for instances of missing or possibly errant data.
df = pd.read_csv('new data/new_sales_data.csv')


# using hist() function to draw Histogram for each column 
# 
# and use figsize(,) parameter to show it obviously

# In[42]:


# Use this, and more code cells, to explore your data. Don't forget to add
# Markdown cells to document your observations and findings.

df.hist(figsize = (30,18));


# using plotting.scatter_matrix() function to draw Histogramsfor each column 
# 
# and scatter plotting between numerical columns 
# 
# and use figsize(,) parameter to show it obviously

# In[43]:


pd.plotting.scatter_matrix(df, figsize = (20,20));


# In[44]:


df.info()


# In[46]:


City_value_counts = df.City.value_counts()
City_value_counts


# In[51]:


df.City.value_counts().plot(kind='barh');


# In[54]:


# Creating a pie chart for cities
plt.pie (City_value_counts,
         labels = ['San Francisco','Los Angeles','New York City','Boston','Atlanta','Dallas','Seattle','Portland','Austin'],
         autopct = "%1.1f%%",
         shadow = True,
         explode = (0.1,0.2,0.2,0.3,0.3,0.3,0.3,0.3,0.3)
        )
plt.title ("Percentage of City_value_counts")
plt.axis ('equal') ;


# In[55]:


Product_value_counts = df.Product.value_counts()
Product_value_counts


# In[59]:


df.Product.value_counts().plot(kind='bar');
#df.City.value_counts().plot(kind='barh');


# In[65]:


plt.scatter(df.Month , df[['Price Each']]) 
plt.title ("Relationship between month and price") 
plt.xlabel ("month") 
plt.ylabel ("price")


# In[66]:


import os

directory_path = "./output/"
os.makedirs(directory_path, exist_ok=True)


# In[68]:


get_ipython().system('jupyter nbconvert --to html --no-input sales.ipynb  ')


# In[ ]:





# In[ ]:




