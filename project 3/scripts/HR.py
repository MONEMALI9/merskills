#!/usr/bin/env python
# coding: utf-8

# ### Author : ABDELMONEM ELMONGY
# ### Data     : 09/2023

# # Project : HR Employee Dataset
# 

# ## About Dataset:
# 
# ### Context:
# HR Analytics helps us with interpreting organizational data. It finds the people-related trends in the data and allows the HR Department to take the appropriate steps to keep the organization running smoothly and profitably. Attrition in a corporate setup is one of the complex challenges that the people managers and the HRs personnel have to deal with.
# 
# ### Content:
# Interestingly, machine learning models can be deployed to predict potential attrition cases, helping the appropriate HR Personnel take the necessary steps to retain the employee.
# Below are the values each column has. The column names are pretty self-explanatory.
# 
#     AGE Numerical Value
# 
#     ATTRITION Employee leaving the company (0=no, 1=yes)
# 
#     BUSINESS TRAVEL (1=No Travel, 2=Travel Frequently, 3=Travel Rarely)
# 
#     DAILY RATE Numerical Value - Salary Level
# 
#     DEPARTMENT (1=HR, 2=R&D, 3=Sales)
# 
#     DISTANCE FROM HOME Numerical Value - THE DISTANCE FROM WORK TO HOME
# 
#     EDUCATION Numerical Value. (1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor')
# 
#     EDUCATION FIELD (1=HR, 2=LIFE SCIENCES, 3=MARKETING, 4=MEDICAL SCIENCES, 5=OTHERS, 6= TECHNICAL)
# 
#     EMPLOYEE COUNT Numerical Value
# 
#     EMPLOYEE NUMBER Numerical Value - EMPLOYEE ID
# 
#     ENVIRONMENT SATISFACTION Numerical Value - SATISFACTION WITH THE ENVIRONMENT (1 'Low' 2 'Medium' 3 'High' 4 'Very High')
# 
#     GENDER (1=FEMALE, 2=MALE)
# 
#     HOURLY RATE Numerical Value - HOURLY SALARY
# 
#     JOB INVOLVEMENT Numerical Value - JOB INVOLVEMENT (1 'Low' 2 'Medium' 3 'High' 4 'Very High')
# 
#     JOB LEVEL Numerical Value - LEVEL OF JOB
# 
#     JOB ROLE (1=HR REP, 2=HR, 3=LAB TECHNICIAN, 4=MANAGER, 5= MANAGING DIRECTOR, 6= RESEARCH DIRECTOR, 7= RESEARCH SCIENTIST, 8=SALES EXECUTIVE, 9= SALES REPRESENTATIVE)
# 
#     JOB SATISFACTION Numerical Value - SATISFACTION WITH THE JOB (1 'Low' 2 'Medium' 3 'High' 4 'Very High')
# 
#     MARITAL STATUS (1=DIVORCED, 2=MARRIED, 3=SINGLE)
# 
#     MONTHLY INCOME Numerical Value - MONTHLY SALARY
# 
#     MONTHLY RATE Numerical Value - MONTHLY RATE
# 
#     NUMCOMPANIES WORKED Numerical Value - NO. OF COMPANIES WORKED AT
# 
#     OVER 18 (1=YES, 2=NO)
# 
#     OVERTIME (1=NO, 2=YES)
# 
#     PERCENT SALARY HIKE Numerical Value - PERCENTAGE INCREASE IN SALARY
# 
#     PERFORMANCE RATING Numerical Value - PERFORMANCE RATING
# 
#     RELATIONS SATISFACTION Numerical Value - RELATIONS SATISFACTION
# 
#     STANDARD HOURS Numerical Value - STANDARD HOURS
# 
#     STOCK OPTIONS LEVEL Numerical Value - STOCK OPTIONS (Higher the number, the more stock option an employee has)
# 
#     TOTAL WORKING YEARS Numerical Value - TOTAL YEARS WORKED
# 
#     TRAINING TIMES LAST YEAR Numerical Value - HOURS SPENT TRAINING
# 
#     WORK LIFE BALANCE Numerical Value - TIME SPENT BETWEEN WORK AND OUTSIDE
# 
#     YEARS AT COMPANY Numerical Value - TOTAL NUMBER OF YEARS AT THE COMPANY
# 
#     YEARS IN CURRENT ROLE Numerical Value -YEARS IN CURRENT ROLE
# 
#     YEARS SINCE LAST PROMOTION Numerical Value - LAST PROMOTION
# 
#     YEARS WITH CURRENT MANAGER Numerical Value - YEARS SPENT WITH CURRENT MANAGER
# 
# ### Acknowledgements
# 
# [IBM](https://www.ibm.com/communities/analytics/watson-analytics-blog/watson-analytics-use-case-for-hr-retaining-valuable-employees/)
# 
# ### Join Us
# 
# Join and follow the [Co-learning Lounge](https://linktr.ee/colearninglounge) for more.
# 
# 

# ### **Tasks to perform: **
# 
# #### Data Cleaning:
# 
#     Deleting redundant columns.
#     Renaming the columns.
#     Dropping duplicates.
#     Cleaning individual columns.
#     Remove the NaN values from the dataset
#     Check for some more Transformations
# 
# #### Data Visualization:
# 
#     Plot a correlation map for all numeric variables
#     Overtime
#     Marital Status
#     Job Role
#     Gender
#     Education Field
#     Department
#     Business Travel
#     Relation between Overtime and Age
#     Total Working Years
#     Education Level
#     Number of Companies Worked
#     Distance from Home
# 
# ### Sources:
# 
# [Source from kaggle](https://www.kaggle.com/search?q=HR-Employee-Attrition+in%3Adatasets)

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
HR = pd.read_csv('data/HR Employee Attrition.csv')


# we use head() or tail() function to display a sample of data 

# In[3]:


HR.head()


# In[4]:


HR.tail()


# ## Assessing Data
# <a id='Assessing'></a>

# ### 1.Assessing Enhanced Twitter Archive

# We assessing our data using some function like : shape , ndim , dtypes , size  , info() , nunique() , isnull()

# In[5]:


# return number of columns and number of row
HR.shape


# the shape function get number of rows and number of columns in tuple

# In[6]:


#return number of dimensions of data
HR.ndim


# size function show us the result of multiplication of number of rows and number of columns

# In[7]:


# return size of Dataset which is a multiplication of number of rows and number of columns
HR.size


# dtypes show us data type of each column (features)

# In[8]:


HR.head(1)


# In[9]:


#return types of each column
HR.dtypes


# info() function show us number of non_null_value in each column and datatype
# 
# it has two features(no of non_null_value,datatype)

# In[10]:


#return number of non-null-value and datatype of each column
HR.info()


# nunique() show us number of unique values in each column

# In[11]:


#return number of unique value
HR.nunique()


# In[12]:


MaritalStatus_value_counts = HR.MaritalStatus.value_counts()
MaritalStatus_value_counts


# In[13]:


Age_value_counts = HR.Age.value_counts()
Age_value_counts


# isnull() function show us boolean value for each element (each cell) it is null or not
# 
# if it(element) null return True
# 
# else  return False

# In[14]:


# return which value is nul or not for each element in DataSet 
HR.isnull()


# isnull().any() function show us boolean value for each column it is null or not
# 
# if column null return True
# 
# else  return False

# In[15]:


# return which value is nul or not for each columns in DataSet 
HR.isnull().any()


# isnull().any() function show us boolean value for each column it is null or not
# 
# if column null return 1
# 
# else  return 0

# In[16]:


#return number of columns has a null value
HR.isnull().any().sum()


# In[17]:


#return number of null value for each column
HR.isnull().sum()


# In[18]:


#return a number of cell has a null value
HR.isnull().sum().sum()


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

# In[19]:


#return statistical descriptive of dataset for each column
HR.describe(include = 'all')


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

# In[20]:


clean_HR = HR.copy()


# #### Test

# In[21]:


clean_HR.head(1)


# ##### Issues 1#

# ## cleanning tidiness issuses

# ### duplicted value

# #### code

# In[22]:


clean_HR.duplicated()


# ##### Test

# In[23]:


clean_HR.duplicated().sum()


# ##### Define

# ## cleaning quality issuses 2#

# In[24]:


clean_HR.info()


# In[25]:


clean_HR.head(1)


# In[26]:


sum(clean_HR.isnull().any())


# ### incorrect datatype

# # NON Incorrect datatype

# #### checking

# In[27]:


clean_HR.dtypes


# In[28]:


clean_HR.info()


# In[29]:


new_HR_data = clean_HR.copy()


# # storing Data

# In[30]:


import os

directory_path = "./new data/"
os.makedirs(directory_path, exist_ok=True)


# In[31]:


new_HR_data.to_csv("./new data/new_HR_data.csv",index = False);


# #### DONE
# #### non_dublicated data....non_missing value....non incorrect datatype

# <a id='eda'></a>
# # Analysis and visualization
# 
# ## Exploratory Data Analysis
# 
# > **Tip**: Now that you've trimmed and cleaned your data, you're ready to move on to exploration. **Compute statistics** and **create visualizations** with the goal of addressing the research questions that you posed in the Introduction section. You should compute the relevant statistics throughout the analysis when an inference is made about the data. Note that at least two or more kinds of plots should be created as part of the exploration, and you must  compare and show trends in the varied visualizations. 

# ### load our cleaned data

# In[32]:


# Load your data and print out a few lines. Perform operations to inspect data
# types and look for instances of missing or possibly errant data.
df = pd.read_csv('new data/new_HR_data.csv')


# using hist() function to draw Histogram for each column 
# 
# and use figsize(,) parameter to show it obviously

# In[33]:


# Use this, and more code cells, to explore your data. Don't forget to add
# Markdown cells to document your observations and findings.

df.hist(figsize = (30,18));


# using plotting.scatter_matrix() function to draw Histogramsfor each column 
# 
# and scatter plotting between numerical columns 
# 
# and use figsize(,) parameter to show it obviously

# In[34]:


pd.plotting.scatter_matrix(df, figsize = (150,150));


# In[35]:


df.info()


# In[36]:


df.Gender.value_counts().plot(kind='bar');


# In[37]:


df.MaritalStatus.value_counts().plot(kind='bar');


# In[38]:


def linechart(x,y,title):
    plt.plot(x, y)
    plt.xlabel("X-axis")  # add X-axis label
    plt.ylabel("Y-axis")  # add Y-axis label
    plt.title(title)  # add title
    plt.show()


# In[39]:


linechart(df.Age ,df.HourlyRate,"relationship between Age and HourlyRate")


# In[40]:


linechart(df.MaritalStatus,df.OverTime,"relationship between MaritalStatus and OverTime")


# In[41]:


display(df.corr())


# In[42]:


plt.figure(figsize = (10,8))
sns.heatmap(df.corr(),annot = True)


# In[43]:


BusinessTravel_value_counts = df.BusinessTravel.value_counts()
BusinessTravel_value_counts


# In[44]:


# Creating a pie chart for cities
plt.pie (City_value_counts,
         labels = ['Travel_Rarely','Travel_Frequently','Non-Travel'],
         autopct = "%1.1f%%",
         shadow = True,
         explode = (0.5,0.5,0.5)
        )
plt.title ("Percentage of BusinessTravel_value_counts")
plt.axis ('equal') ;


# In[ ]:





# In[ ]:





# In[ ]:


import os

directory_path = "./output/"
os.makedirs(directory_path, exist_ok=True)


# In[ ]:


get_ipython().system('jupyter nbconvert --to html --no-input HR.ipynb')


# In[ ]:


get_ipython().system('jupyter nbconvert --to latex HR.ipynb')


# In[ ]:


get_ipython().system('jupyter nbconvert --to html  HR.ipynb')


# In[ ]:


get_ipython().system('jupyter nbconvert --to python HR.ipynb')

