#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[3]:


get_ipython().system('pip install --upgrade matplotlib')
get_ipython().system('pip install --upgrade seaborn')

get_ipython().system('pip install xgboost')
get_ipython().system('pip install lightgbm')
get_ipython().system('pip install catboost')

get_ipython().system('pip install inflection')
get_ipython().system('pip install dython')
get_ipython().system('pip install shap')


# In[6]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


# model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from xgboost import XGBRFClassifier
# from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer

# model evaluation & tuning hyperparameter
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

# explainable AI
#import shap


# In[5]:


# Load datasets
churn_df = pd.read_csv(r"C:\Users\sirisha.vemula\Desktop\ds_grad\capstone_datasets\churn.csv")
customer_df = pd.read_csv(r"C:\Users\sirisha.vemula\Desktop\ds_grad\capstone_datasets\customer.csv")
products_df = pd.read_csv(r"C:\Users\sirisha.vemula\Desktop\ds_grad\capstone_datasets\products.csv")
orders_df = pd.read_csv(r"C:\Users\sirisha.vemula\Desktop\ds_grad\new_capstone\orders.csv")
promotions_df = pd.read_csv(r"C:\Users\sirisha.vemula\Desktop\ds_grad\capstone_datasets\promotions.csv")


# ### Data Understanding

# In[7]:


print("Churn Data:")
print(churn_df.head())

print("\nCustomer Data:")
print(customer_df.head())

print("\nProducts Data:")
print(products_df.head())

print("\nOrders Data:")
print(orders_df.head())

print("\nPromotions Data:")
print(promotions_df.head())


# In[8]:


# Initial Data Analysis
print("Churn Data:")
print(churn_df.info())

print("\nCustomer Data:")
print(customer_df.info())

print("\nProducts Data:")
print(products_df.info())

print("\nOrders Data:")
print(orders_df.info())

print("\nPromotions Data:")
print(promotions_df.info())


# In[9]:


# Check for duplicates
print("\nDuplicates in Churn Data:", churn_df.duplicated().sum())
print("Duplicates in Customer Data:", customer_df.duplicated().sum())
print("Duplicates in Products Data:", products_df.duplicated().sum())
print("Duplicates in Orders Data:", orders_df.duplicated().sum())
print("Duplicates in Promotions Data:", promotions_df.duplicated().sum())


# In[10]:


# Display descriptive statistics
print("\nDescriptive Statistics for Churn Data:")
print(churn_df.describe())

print("\nDescriptive Statistics for Customer Data:")
print(customer_df.describe())

print("\nDescriptive Statistics for Products Data:")
print(products_df.describe())

print("\nDescriptive Statistics for Orders Data:")
print(orders_df.describe())

print("\nDescriptive Statistics for Promotions Data:")
print(promotions_df.describe())


# In[11]:


# Initial Analysis
print("\nSummary Statistics for Churn Data:")
print(churn_df.describe())
print("\nValue Counts for Gender in Customer Data:")
print(customer_df['Gender'].value_counts())


# ### Initial Data Exploration

# In[13]:


churn_df.info()


# In[12]:


churn_df.head()


# **Recency is number of days since LastPurchaseDate passed as on '20-04-2023'. Similarly, TimeSinceFirstPurchase is no. of days passed since FirstPurchaseDate. So we will drop both the date columns.**

# In[14]:


churn_df.describe()


# In[15]:


numerical_df = churn_df.select_dtypes('number')
numerical_df


# In[16]:


categorical_df = churn_df.select_dtypes('O')
categorical_df


# In[22]:


for i in churn_df.select_dtypes('number').columns:
    if(i!='CustomerID'):
        sns.kdeplot(churn_df[i])
        
    plt.show()


# In[23]:


for i in churn_df.select_dtypes('number').columns:
    if(i!='CustomerID'):
        sns.boxplot(churn_df[i])
        
    plt.show()


# In[17]:


churn_df.Churn.value_counts()


# In[18]:


sns.countplot(x=churn_df.Churn);


# In[19]:


pd.crosstab(index=churn_df.Churn, columns=churn_df.LongTermCustomer).plot.bar(figsize=(8,4));


# **LongTermCustomer does not give us a clear picture of whether the customer will churn or not.**

# In[20]:


pd.crosstab(index=churn_df.Churn, columns=churn_df.Profession).plot.bar(figsize=(8,4));


# In[27]:


pd.DataFrame(churn_df.RFM_Score.value_counts()).plot(kind='bar',figsize=(16,4))


# ### Missing Values Treatment: profession

# In[34]:


churn_df.isnull().sum()


# In[35]:


churn_df.Profession.unique()


# In[36]:


churn_df[churn_df.Profession.isnull()]


# In[37]:


print('Percentage of Missing values in Profession column: ', (churn_df[churn_df.Profession.isnull()].shape[0]/churn_df.shape[0])*100, '%')


# In[38]:


churn_df_v1 = churn_df.fillna({'Profession':churn_df.Profession.mode()[0]})


# In[39]:


churn_df_v1.isnull().sum()


# In[40]:


# Converting Date columns to datetime format

churn_df_v1['LastPurchaseDate'] = pd.to_datetime(churn_df_v1['LastPurchaseDate'], format='%d-%m-%Y')
churn_df_v1['FirstPurchaseDate'] = pd.to_datetime(churn_df_v1['FirstPurchaseDate'], format='%d-%m-%Y')


# In[41]:


churn_df_v1.dtypes


# ### Outlier Treatment

# In[42]:


def plot_histograms(df,column):
    plt.hist(df[column], bins=50)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column}')
    plt.show()


# In[43]:


def detect_outliers(df,col,Q1=20,Q3=80):
    first_quar = np.percentile(df[col],Q1)
    third_quar = np.percentile(df[col],Q3)
    IQR= third_quar - first_quar
    lower_boundary = first_quar - 1.5*IQR
    upper_boundary = third_quar + 1.5*IQR
    df_outlier = df[(df[col]<lower_boundary) | (df[col]>upper_boundary)]
    if len(df_outlier)>0:
        print(f"There are outliers for {col}")
        return True
    else:
        print(f"There are no outliers for {col}")
        return False


# In[44]:


def treat_outliers(df, column, low_quantile=0.20, up_quantile=0.80):
    Q1 = np.nanpercentile(df[column], low_quantile)
    Q3 = np.nanpercentile(df[column], up_quantile)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    
    
    skewness = df[column].skew()

    # Determine skewness type
    skewness_type = None
    if skewness > 0:
        skewness_type = 'Right-skewed'
    elif skewness < 0:
        skewness_type = 'Left-skewed'
    else:
        skewness_type = 'Symmetric'

    if skewness_type=='Right-skewed':
        outlier_fraction = df[df[column]>upper].shape[0]/df.shape[0]
        print(f"Column {column} has {outlier_fraction*100} % outliers.")

        if outlier_fraction<0.05:
            flag=True
            # drop customers with minimum payment greater than 75th percentile
            mask = df[column] > upper
            df = df[~mask]
        else:
            flag=False
            print('keeping outliers since there is significant data in them.')


    elif skewness_type == 'Left-skewed':
        outlier_fraction = df[df[column]<lower].shape[0]/df.shape[0]
        print(f"Column {column} has {outlier_fraction*100} % outliers.")

        if outlier_fraction<0.05:
            flag=True
            # drop customers with minimum payment less than 25th percentile
            mask = df[column] < lower
            df = df[~mask]
        else:
            flag=False
            print('keeping outliers since there is significant data in them.')

    else:
        outlier_fraction = ( df[df[column]>upper].shape[0] + df[df[column]<lower].shape[0] ) / df.shape[0]
        print(f"Column {column} has {outlier_fraction*100} % outliers.")
        
        if outlier_fraction<0.05:
            flag=True
            # drop customers with minimum payment greater than 75th percentile
            mask = df[column] > upper | df[column] < lower
            df = df[~mask]
        else:
            flag=False
            print('keeping outliers since there is significant data in them.')

    return df, flag


# In[45]:


churn_df_v2 = churn_df_v1.copy(True)
for column in ['TotalAmount', 'Annual Income ($)', 'Recency', 'TimeSinceFirstPurchase','Frequency','MonetaryValue']:
    plot_histograms(churn_df_v2, column)
    treat_outlier_flag = detect_outliers(churn_df_v2, column)
    if treat_outlier_flag == True:
        churn_df_v3, flag = treat_outliers(churn_df_v2, column)


# In[46]:


churn_df_v2.shape


# **We can ignore customer dataframe since all columns from customer.csv dataset are present in churn_df.**

# In[47]:


orders_df.describe()


# **Will be dropping PromotionID so, ignoring the missing values from PromotionID.**

# In[48]:


orders_numerical_df = orders_df.select_dtypes('number')
orders_numerical_df


# In[49]:


orders_categorical_df = orders_df.select_dtypes('O')
orders_categorical_df


# In[50]:


orders_df_v1 = orders_df.copy(True)
for column in ['Total_Amount']:
    plot_histograms(orders_df_v1, column)
    treat_outlier_flag = detect_outliers(orders_df_v1, column)
    if treat_outlier_flag == True:
        orders_df_v1, flag = treat_outliers(orders_df_v1, column)


# In[51]:


products_df['Release_Date']=pd.to_datetime(products_df['Release_Date'], format="%Y-%m-%d")


# **There are no missing values in products.csv**

# In[52]:


products_numerical_df = products_df.select_dtypes('number')
products_numerical_df


# In[53]:


products_categorical_df = products_df.select_dtypes('O')
products_categorical_df


# In[54]:


products_df_v1 = products_df.copy(True)
for column in ['Price','Discount']:
    plot_histograms(products_df_v1, column)
    treat_outlier_flag = detect_outliers(products_df_v1, column)
    if treat_outlier_flag == True:
        products_df_v1, flag = treat_outliers(products_df_v1, column)


# **There are no missing values in promotions.csv**

# In[55]:


promotions_df['StartDate']=pd.to_datetime(promotions_df['StartDate'], format="%Y-%m-%d %H:%M:%S")
promotions_df['EndDate']=pd.to_datetime(promotions_df['EndDate'], format="%Y-%m-%d %H:%M:%S")


# In[56]:


promotions_df_v1 = promotions_df.copy(True)
for column in ['DiscountPercentage']:
    plot_histograms(promotions_df_v1, column)
    treat_outlier_flag = detect_outliers(promotions_df_v1, column)
    if treat_outlier_flag == True:
        promotions_df_v1, flag = treat_outliers(promotions_df_v1, column)


# ### Plots

# In[28]:


# Visualize data distributions
sns.histplot(customer_df['Age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()


# In[29]:


sns.boxplot(x='Gender', y='Annual Income ($)', data=customer_df)
plt.xlabel('Gender')
plt.ylabel('Annual Income ($)')
plt.title('Annual Income Distribution by Gender')
plt.show()


# In[30]:


# plotting a scatter plot for relationship between age and annual income in Customer Data
plt.scatter(customer_df['Age'], customer_df['Annual Income ($)'], color='green')
plt.xlabel('Age')
plt.ylabel('Annual Income ($)')
plt.title('Relationship between Age and Annual Income')
plt.show()


# In[31]:


# Visualize relationships between variables
import seaborn as sns

sns.scatterplot(x='TotalQuantity', y='TotalAmount', data=churn_df, hue='Churn')
plt.xlabel('Total Quantity')
plt.ylabel('Total Amount')
plt.title('Relationship between Quantity and Amount')
plt.show()


# In[32]:


# Churn Analysis
churn_by_gender = churn_df.groupby('Gender')['Churn'].mean()
print("\nChurn Rates by Gender:")
print(churn_by_gender)


# In[33]:


# RFM Analysis
sns.boxplot(x='RFM_Score', y='Churn', data=churn_df)
plt.xlabel('RFM Score')
plt.ylabel('Churn')
plt.title('RFM Score vs. Churn')
plt.show()


# In[58]:


# Promotional Analysis
brand_category_promo_count = promotions_df.groupby(['Brand', 'Category'])['PromotionID'].count().unstack()
brand_category_promo_count.plot(kind='bar', stacked=True)
plt.xlabel('Brand')
plt.ylabel('Promotion Count')
plt.title('Brand and Category Promotion Count')
plt.show()


# In[59]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot monthly promotions count
def plot_monthly_promotions(promotions_df):
    # Convert 'StartDate' and 'EndDate' to datetime
    promotions_df['StartDate'] = pd.to_datetime(promotions_df['StartDate'])
    promotions_df['EndDate'] = pd.to_datetime(promotions_df['EndDate'])
    
    # Extract month from 'StartDate' and count promotions per month
    promotions_df['StartMonth'] = promotions_df['StartDate'].dt.month
    monthly_promotions_count = promotions_df['StartMonth'].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=monthly_promotions_count.index, y=monthly_promotions_count.values, color='skyblue')
    plt.title('Monthly Promotions Count')
    plt.xlabel('Month')
    plt.ylabel('Promotions Count')
    plt.xticks(rotation=45)
    plt.show()

# Plot monthly sales change percentage
def plot_monthly_sales_change(orders_df):
    # Convert 'Purchase_Date' to datetime
    orders_df['Purchase_Date'] = pd.to_datetime(orders_df['Purchase_Date'])
    
    # Extract month from 'Purchase_Date' and calculate monthly sales
    orders_df['PurchaseMonth'] = orders_df['Purchase_Date'].dt.month
    monthly_sales = orders_df.groupby('PurchaseMonth')['Discounted_Price'].sum()

    # Calculate monthly sales change percentage
    monthly_sales_change_percentage = monthly_sales.pct_change() * 100

    # Plot
    plt.figure(figsize=(10, 6))
    monthly_sales_change_percentage.plot(marker='o', color='orange')
    plt.title('Monthly Sales Change Percentage')
    plt.xlabel('Month')
    plt.ylabel('Sales Change Percentage')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

# Plot sales percentage change by brand and category
def plot_sales_percentage_change(orders_df):
    # Calculate total sales by brand and category
    brand_category_sales = orders_df.groupby(['Brand', 'Category'])['Discounted_Price'].sum()
    
    # Calculate sales percentage change by brand and category
    sales_percentage_change = brand_category_sales.unstack().pct_change(axis=1) * 100
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(sales_percentage_change, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Sales Percentage Change by Brand and Category')
    plt.xlabel('Category')
    plt.ylabel('Brand')
    plt.show()

# Plot Monthly Promotions Count
plot_monthly_promotions(promotions_df)

# Plot Monthly Sales Change Percentage
plot_monthly_sales_change(orders_df)

# Plot Sales Percentage Change by Brand and Category
plot_sales_percentage_change(orders_df)


# In[60]:


# Generating Plots

# Brand and Category Promotions Count
plt.figure(figsize=(12, 6))
sns.countplot(data=promotions_df, x='Brand', hue='Category')
plt.title('Promotions Count by Brand and Category')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Category')
plt.show()


# In[61]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot monthly promotions count for each year-month combination
def plot_monthly_promotions(promotions_df):
    # Convert 'StartDate' to datetime
    promotions_df['StartDate'] = pd.to_datetime(promotions_df['StartDate'])
    
    # Extract year and month from 'StartDate'
    promotions_df['YearMonth'] = promotions_df['StartDate'].dt.to_period('M')
    
    # Group by year-month and brand-category, count promotions
    promotions_count = promotions_df.groupby(['YearMonth', 'Brand', 'Category']).size().reset_index(name='PromotionsCount')

    # Get unique year-month combinations
    unique_yearmonths = promotions_df['YearMonth'].unique()

    # Plot each year-month combination separately
    for yearmonth in unique_yearmonths:
        # Filter data for the current year-month
        data_yearmonth = promotions_count[promotions_count['YearMonth'] == yearmonth]
        
        # Create a subplot for each year-month combination
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Brand', y='PromotionsCount', hue='Category', data=data_yearmonth)
        plt.title(f'Monthly Promotions Count ({yearmonth})')
        plt.xlabel('Brand')
        plt.ylabel('Promotions Count')
        plt.xticks(rotation=45)
        plt.legend(title='Category')
        plt.tight_layout()
        plt.show()



# Plot Monthly Promotions Count
plot_monthly_promotions(promotions_df)


# - 1.	`Brand and Category Promotions Count`

# In[62]:


#•	How many promotions are run by each brand across different product categories?
temp =  pd.DataFrame(promotions_df.groupby(['Brand','Category'])['PromotionID'].count()).reset_index().sort_values(by='PromotionID',ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(data=temp, x= 'Brand', y= 'PromotionID', hue = 'Category')
plt.xticks(rotation=90)
plt.ylabel('Promotions Counts')
plt.title('Brand wise promotion across different category')
plt.legend(loc='upper right')
plt.show()


# In[63]:


#•	Which brands and categories have the highest number of promotions?

plt.figure(figsize=(10, 8))

# First subplot
plt.subplot(211)
ax1 = pd.DataFrame(promotions_df.groupby('Brand')['PromotionID'].count()) \
        .sort_values(by='PromotionID', ascending=False) \
        .plot(kind='bar', ax=plt.gca())  # Using plt.gca() to get the current axes
plt.legend().remove()
plt.ylabel('Promotions Count')

# Second subplot
plt.subplot(212)
ax2 = pd.DataFrame(promotions_df.groupby('Category')['PromotionID'].count()) \
    .sort_values(by='PromotionID', ascending=False) \
    .plot(kind='bar', ax=plt.gca())  # Using plt.gca() to get the current axes
plt.legend().remove()
plt.ylabel('Promotions Count')

plt.tight_layout()  
plt.show()


# - 2.	`Brand and Category Sales Change Percentage`

# - What is the percentage change in sales for each brand and category before and after promotions?
# - Identify the brands and categories with the most significant positive and negative changes in sales percentage.
# - Generate a plot that compares the percentage change in sales across brands and categories.

# In[64]:


#merging ORDER and PROMOTION data 
temp_order_promotion = pd.merge(orders_df,promotions_df, on=['PromotionID', 'Brand', 'Category'], how = 'left')
temp_order_promotion['Purchase_Year_Month'] = temp_order_promotion.Purchase_Date.dt.to_period('M')
temp_order_promotion['Brand_Category']  = temp_order_promotion.Brand+'-'+temp_order_promotion.Category
temp_no_promo = pd.DataFrame(temp_order_promotion[temp_order_promotion.PromotionID.isna()].groupby(['Brand','Category','Brand_Category','Purchase_Year_Month'])['Total_Amount'].sum()).reset_index()
temp_promo = pd.DataFrame(temp_order_promotion[~temp_order_promotion.PromotionID.isna()].groupby(['Brand','Category','Brand_Category','Purchase_Year_Month'])['Total_Amount'].sum()).reset_index()
temp = pd.merge(temp_no_promo,temp_promo, on = ['Brand','Category','Brand_Category','Purchase_Year_Month'], how  = 'left')
temp.rename(columns={'Total_Amount_x':'Non_Promotion_Sales','Total_Amount_y':'Promotion_Sales'}, inplace=True)
# temp['Sales%'] = (temp.Promotion_Sales/temp.Non_Promotion_Sales -1) *100
# temp.sort_values(by= 'Sales%', ascending=False,inplace=True)
temp


# In[65]:


#Assumption: if there is no promotion ID attached to the order ,it shows non-promotional sales
temp_order_promotion[~((temp_order_promotion.Purchase_Date>=temp_order_promotion.StartDate) & (temp_order_promotion.Purchase_Date<=temp_order_promotion.EndDate))].PromotionID.nunique()


# In[66]:


temp_brand= temp.groupby(['Brand','Purchase_Year_Month']).apply(lambda x: (x['Promotion_Sales'].sum() / x['Non_Promotion_Sales'].sum() - 1)*100).reset_index(name='Sales_Percentage_Change')
temp_category= temp.groupby(['Category','Purchase_Year_Month']).apply(lambda x: (x['Promotion_Sales'].sum() / x['Non_Promotion_Sales'].sum() - 1)*100).reset_index(name='Sales_Percentage_Change')
temp_brand_category= temp.groupby(['Brand_Category','Purchase_Year_Month']).apply(lambda x: (x['Promotion_Sales'].sum() / x['Non_Promotion_Sales'].sum() - 1)*100).reset_index(name='Sales_Percentage_Change')


# In[67]:


def plot_sales(df):
    df = df.sort_values(by='Sales_Percentage_Change', ascending=False)
    year_month = sorted(df.Purchase_Year_Month.unique()) # storing each month-year combination
    for i in year_month:
        plt.figure(figsize=(16,10))
        sns.barplot(data=df[df.Purchase_Year_Month==i],y='Sales_Percentage_Change', x=df.columns[0])
        plt.xticks(rotation = 90)
        plt.title(i)
        plt.tight_layout()
        plt.show()


# In[68]:


plot_sales(temp_category)


# In[69]:


plot_sales(temp_brand_category)


# In[70]:


# Brand and Category Promotion Effectiveness
# Assuming promotion effectiveness can be measured by comparing sales before and after promotions
# and calculating the percentage change in sales
# We'll use promotions_data to filter orders_data and calculate sales change for each brand and category
promotion_effectiveness = orders_df.merge(promotions_df, on='PromotionID')
promotion_effectiveness['Sales_Before_Promotion'] = promotion_effectiveness['Quantity'] * promotion_effectiveness['Original_Price']
promotion_effectiveness['Sales_After_Promotion'] = promotion_effectiveness['Quantity'] * promotion_effectiveness['Discounted_Price']
promotion_effectiveness['Promotion_Effectiveness'] = ((promotion_effectiveness['Sales_After_Promotion'] - promotion_effectiveness['Sales_Before_Promotion']) / promotion_effectiveness['Sales_Before_Promotion']) * 100

plt.figure(figsize=(12, 6))
sns.barplot(data=promotion_effectiveness, x='Brand_x', y='Promotion_Effectiveness', hue='Category_x')
plt.title('Promotion Effectiveness by Brand and Category')
plt.xlabel('Brand')
plt.ylabel('Promotion Effectiveness')
plt.xticks(rotation=45)
plt.legend(title='Category')
plt.show()


# ## Final Unified Dataframe

# **We need to merge using ProductID, Brand, Category, Original_Price(Price)**

# In[71]:


orders_products_df = pd.merge(orders_df, 
                                 products_df, 
                                 left_on=['ProductID', 'Brand', 'Category', 'Original_Price'], 
                                 right_on=['ProductID', 'Brand', 'Category', 'Price'],
                                 how='left')


# In[72]:


orders_products_df.shape[0]


# **We need to merge using ProductID, Brand, Category**

# In[73]:


orders_products_promotions_df = pd.merge(orders_products_df, 
                                         promotions_df, 
                                         left_on=['PromotionID', 'Brand', 'Category'],
                                         right_on=['PromotionID', 'Brand', 'Category'],
                                         how='left')


# In[74]:


orders_df.shape, products_df.shape, orders_products_df.shape, promotions_df.shape, orders_products_promotions_df.shape


# **The discount in 'products' dataset, is products' own discount. Which is applied when no promotions are present.**
# 
# **The discount in 'promotions' dataset, how is it applied? - When a promotion is valid, it's discount is applied in the place of product's discount.**

# In[75]:


master_data_df = pd.merge(orders_products_promotions_df, churn_df_v3, on='CustomerID', how='left')
master_data_df.head()


# In[76]:


orders_products_promotions_df.shape, churn_df_v3.shape, master_data_df.shape


# In[77]:


master_data_df.isnull().sum()


# ### Checking validity of purchase records with a promotion applied that whether they are purchased only within the promotion period :

# In[78]:


mask = (master_data_df['Purchase_Date']>=master_data_df['StartDate'])\
                             & (master_data_df['Purchase_Date']<=master_data_df['EndDate'])

df = master_data_df[(master_data_df['PromotionID'].notnull()) & mask]
df.shape[0]


# In[79]:


master_data_df[(master_data_df['PromotionID'].notnull()) & (~mask)].shape[0]


# **Since all purchases with promotions applied are within promotion period, and the promotion period themselves do not contains any necessary information, we will be dropping them.**

# In[80]:


master_data_df[master_data_df['PromotionID'].notnull()].head()


# **Since DiscountPercentage is Discount in orders dataset, when a promotion is applied, we can drop this column also, create a new column to store this information which denotes whether the discount is from promotion.**

# ## Feature Engineering

# ### Customer Life Time Value

# Customer_Value = 0.33*(R_norm) + 0.34*(F_norm) + 0.33*(M_norm)
# 
# CLTV  = Customer_Value * Customer_Life_Span 
#  
# where, Customer_Life_Span = LastPurchaseDate - FIrstPurchaseDate

# In[81]:


master_data_df['Recency_norm'] = (master_data_df['Recency'] - master_data_df['Recency'].min()) / (master_data_df['Recency'].max() - master_data_df['Recency'].min())
master_data_df['Frequency_norm'] = (master_data_df['Frequency'] - master_data_df['Frequency'].min()) / (master_data_df['Frequency'].max() - master_data_df['Frequency'].min())
master_data_df['MonetaryValue_norm'] = (master_data_df['MonetaryValue'] - master_data_df['MonetaryValue'].min()) / (master_data_df['MonetaryValue'].max() - master_data_df['MonetaryValue'].min())


# In[82]:


master_data_df[['Recency_norm','Frequency_norm','MonetaryValue_norm']]


# In[83]:


master_data_df['Customer_Value'] = 0.33*master_data_df['Recency_norm'] + 0.33* master_data_df['Frequency_norm'] + 0.34* master_data_df['MonetaryValue_norm']


# In[84]:


master_data_df['Customer_Value']


# In[85]:


master_data_df['CLTV'] = master_data_df['Customer_Value'] * ((master_data_df['LastPurchaseDate']-\
                                         master_data_df['FirstPurchaseDate']).dt.days)


# In[86]:


master_data_df['CLTV']


# ## Customer Segments

# In[87]:


bins= np.linspace(master_data_df.Customer_Value.min(), master_data_df.Customer_Value.max(), 4)
bins


# In[88]:


master_data_df['Customer_Segment'] = pd.cut(master_data_df.Customer_Value, bins=bins, 
                                            labels=['Low_Value_Customer', 'Medium_Value_Customer', 'High_Value_Customer'],
                                           include_lowest=True)
master_data_df['Customer_Segment']


# In[89]:


master_data_df['Customer_Segment'].value_counts()


# ### Average discount availed

# In[90]:


master_data_df['Discount_Availed']=master_data_df['Original_Price']-master_data_df['Discounted_Price']


# In[91]:


master_data_df['Average_Discount_Availed'] = master_data_df.groupby(by='CustomerID').Discount_Availed.agg('mean')[master_data_df['CustomerID']].values
master_data_df['Average_Discount_Availed']


# In[92]:


master_data_df['Purchase_Frequency'] = ((master_data_df['LastPurchaseDate']-\
                                         master_data_df['FirstPurchaseDate']).dt.days)/master_data_df['TotalPurchases']


# In[93]:


master_data_df['Promotions_Applied'] = np.where(master_data_df['PromotionID'].notnull(), 1, 0)


# In[94]:


master_data_df.head()


# In[95]:


master_data_df.columns


# In[100]:


# Drop un-necessary columns

columns_to_drop = ['OrderID', 'CustomerID', 'ProductID', 'Purchase_Date',
       'Original_Price', 'Discounted_Price', 'PromotionID', 'Price','Product_Name', 'Discount_y','Release_Date', 
        'DiscountPercentage', 'StartDate', 'EndDate', 'City', 'LastPurchaseDate','FirstPurchaseDate',
        'TotalPurchases', 'R', 'F', 'M','RFM_Score', 'Recency_norm','Frequency_norm', 'MonetaryValue_norm',
        'Customer_Value','Discount_Availed','PurchaseMonth','StartMonth','YearMonth']

master_data_df_v1 = master_data_df.drop(columns_to_drop, axis=1)


# In[101]:


master_data_df_v1.rename({
'Annual Income ($)':'Annual_Income',
'Discount_x':'Final_Discount',
'Work Experience':'Work_Experience',
'Family Size':'Family_Size',
'TotalAmount':'Total_Amount2',
'TotalQuantity':'Total_Quantity',
'TimeSinceFirstPurchase':'Time_Since_First_Purchase',
'MonetaryValue':'Monetary_Value',
'LongTermCustomer':'Long_Term_Customer'
}, axis=1, inplace=True)


# In[102]:


master_data_df_v1.head()


# In[103]:


master_data_df_v1.isna().sum()


# ## Column Correlations

# In[104]:


plt.figure(figsize=(18,12))
sns.heatmap(master_data_df_v1.corr(), cmap='bwr', vmin=-1, vmax=1, annot=True)
plt.show()


# ## Encoding Categorical Variables

# In[105]:


X = master_data_df_v1.drop('Churn', axis=1)
y = master_data_df_v1['Churn']


# In[106]:


X_dummies = pd.get_dummies(X, drop_first=True)
X_dummies


# ## Scaling

# In[107]:


from sklearn.model_selection import train_test_split
# data_X = master_data_df_v1.drop('Churn', axis=1)
# data_y = master_data_df_v1['Churn']

X_dummies = X_dummies.sample(frac = 1)
X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, test_size=0.25,
                                                    random_state=1, stratify=y)


# In[108]:


print('X shape\t\t:', X.shape)
print('y shape\t\t:', y.shape)
print()
print('X_train shape\t:', X_train.shape)
print('y_train shape\t:', y_train.shape)
print()
print('X_test shape\t:', X_test.shape)
print('y_test shape\t:', y_test.shape)


# In[109]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

le.fit(y_train)

y_train_encode = le.transform(y_train)
y_test_encode = le.transform(y_test)


# In[110]:


le.classes_


# #### churn will be no as 0 and yes as 1.

# In[111]:


X_train.head()


# In[112]:


column_numerical = ['Quantity','Final_Discount','Total_Amount','Age','Annual_Income','Work_Experience','Family_Size','Total_Amount2','Total_Quantity','Recency','Time_Since_First_Purchase','Frequency','Monetary_Value','CLTV','Average_Discount_Availed','Purchase_Frequency']


# In[113]:


#min-max scaling since dataset is not normal distribution

from sklearn.preprocessing import MinMaxScaler

X_train_scale = X_train.copy()
X_test_scale = X_test.copy()

for i in column_numerical:
  scaler = MinMaxScaler()
  scaler.fit(X_train_scale[[i]])

  X_train_scale[[i]] = scaler.transform(X_train_scale[[i]])
  X_test_scale[[i]] = scaler.transform(X_test_scale[[i]])


# In[114]:


#Scoring udf : 

def get_score(y_pred_list, y_test, average=None, plot=True, axis=0, cmap='Blues'):
  model_name = []
  accuracy = []
  precision = []
  recall = []
  f1 = []
  roc_auc = []

  for name, y_pred in y_pred_list.items():
    model_name.append(name)
    if average != None:
      accuracy.append(accuracy_score(y_test, y_pred))
      precision.append(precision_score(y_test, y_pred, average=average))
      recall.append(recall_score(y_test, y_pred, average=average))
      f1.append(f1_score(y_test, y_pred, average=average))
      roc_auc.append(roc_auc_score(y_test, y_pred, average=average))

      score_list = {
        'model':model_name,
        'accuracy':accuracy,
        f'{average}_avg_precision':precision,
        f'{average}_avg_recall':recall,
        f'{average}_avg_f1_score':f1,
        'roc_auc':roc_auc
      }
    else:
      accuracy.append(accuracy_score(y_test, y_pred))
      precision.append(precision_score(y_test, y_pred))
      recall.append(recall_score(y_test, y_pred))
      f1.append(f1_score(y_test, y_pred))
      roc_auc.append(roc_auc_score(y_test, y_pred))

      score_list = {
        'model':model_name,
        'accuracy':accuracy,
        'precision':precision,
        'recall':recall,
        'f1_score':f1,
        'roc_auc':roc_auc
      }

  score_df = pd.DataFrame(score_list).set_index('model')

  if plot:
    display(score_df.style.background_gradient(axis=axis, cmap=cmap))

      return score_df


# In[115]:


#Models List : 

model_list = {
    'Logistic Regression':LogisticRegression(max_iter=1000, random_state=1),
    'Ridge Classifier':RidgeClassifier(random_state=1),
    'KNN':KNeighborsClassifier(),
    'SVC':SVC(random_state=1),
    'Neural Network':MLPClassifier(max_iter=1000, random_state=1),
    'Decision Tree':DecisionTreeClassifier(random_state=1),
    'Random Forest':RandomForestClassifier(random_state=1),
    'Gradient Boosting Classifier':GradientBoostingClassifier(random_state=1),
    'AdaBoost Classifier':AdaBoostClassifier(random_state=1),
#     'CatBoost Classifier':CatBoostClassifier(random_state=1, verbose=False),
    'Hist Gradient Boosting':HistGradientBoostingClassifier(random_state=1),
    'XGBoost':XGBClassifier(random_state=1, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM':LGBMClassifier(random_state=1),
}

X_train_model = X_train_scale.copy()
y_train_model = y_train.copy()

X_test_model = X_test_scale.copy()
y_test_model = y_test.copy()


# In[116]:


y_pred_list = dict()

for name, model in model_list.items():
    model.fit(X_train_model, y_train_model)
    # Convert X_test_model to a NumPy array if it is a DataFrame
    if isinstance(X_test_model, pd.DataFrame):
        X_test_model_array = X_test_model.to_numpy()
    else:
        X_test_model_array = X_test_model
    y_pred_list[name] = model.predict(X_test_model_array)


# In[117]:


score_list = get_score(y_pred_list, y_test_model, average='macro')


# ### Model Improvements : 

# In[118]:


model_list_tuned = {
#     'Gradient Boosting Classifier':GradientBoostingClassifier(random_state=1,
#                                                               max_depth=2,
#                                                               n_estimators=500,
#                                                               learning_rate=0.02),
              
#     'AdaBoost Classifier':AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=1, max_depth=2),
#                                              random_state=1,
#                                              n_estimators=80,
#                                              learning_rate=0.04),
              
    'Hist Gradient Boosting':HistGradientBoostingClassifier(random_state=1,
                                                            max_iter=300, 
                                                            learning_rate=0.02,
                                                            max_depth=5),
#     'XGBoost':XGBClassifier(random_state=1, use_label_encoder=False, eval_metric='logloss',
#                             colsample_bytree=0.1,
#                             learning_rate=0.005),
              
#     'LightGBM':LGBMClassifier(random_state=1,
#                               num_leaves=10,
#                               n_estimators=175,
#                               learning_rate=0.01)
}


# In[119]:


#Store results for showcasing the performance of all the models : 
y_pred_list2 = dict()

for name, model in model_list_tuned.items():
  model.fit(X_train_model, y_train_model)
  y_pred_list2[name] = model.predict(X_test_model)

score_list2 = get_score(y_pred_list2, y_test_model, average='macro')


# In[120]:


params = {
          'n_estimators': [120,130,150,170,190,200],
           'max_depth': [8,10,12,14,15],
           
           'min_samples_split': [3,4,5,6],
           
           'min_samples_leaf': [1,2,3],
           'random_state': [13]}


# In[121]:


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

params = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'n_estimators': [50, 100, 200, 300]
}

grid_xgb = GridSearchCV(XGBClassifier(random_state=1), param_grid=params, cv=3, scoring='recall').fit(X_train, y_train)


# In[122]:


print('Best parameters:', grid_xgb.best_params_)
print('Best score:', grid_xgb.best_score_)


# In[123]:


score2 = cross_val_score(grid_xgb, X_train, y_train, cv=5, scoring='recall')


# In[124]:


from statistics import stdev

# Assuming 'score2' contains the cross-validation recall scores
grid_cv_score = score2.mean()
grid_cv_stdev = stdev(score2)

print('Cross Validation Recall scores are: {}'.format(score2))
print('Average Cross Validation Recall score: ', grid_cv_score)
print('Cross Validation Recall standard deviation: ', grid_cv_stdev)


# In[125]:


model_list = {
#     'Logistic Regression':LogisticRegression(max_iter=1000, random_state=1),
#     'Ridge Classifier':RidgeClassifier(random_state=1),
#     'KNN':KNeighborsClassifier(),
#     'SVC':SVC(random_state=1),
#     'Neural Network':MLPClassifier(max_iter=1000, random_state=1),
#     'Decision Tree':DecisionTreeClassifier(random_state=1),
#     'Random Forest':RandomForestClassifier(random_state=1),
    'Gradient Boosting Classifier':GradientBoostingClassifier(random_state=1),
    'AdaBoost Classifier':AdaBoostClassifier(random_state=1),
#     'CatBoost Classifier':CatBoostClassifier(random_state=1, verbose=False),
    'Hist Gradient Boosting':HistGradientBoostingClassifier(random_state=1),
    'XGBoost':XGBClassifier(random_state=1, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM':LGBMClassifier(random_state=1),
}


# In[126]:


model_list_tuned_v1 = {
    'Gradient Boosting Classifier':GradientBoostingClassifier(random_state=42,
                                                              max_depth=2,
                                                              n_estimators=500,
                                                              learning_rate=0.02),
              
    'AdaBoost Classifier':AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=1, max_depth=2),
                                             random_state=42,
                                             n_estimators=80,
                                             learning_rate=0.04),
              
    'Hist Gradient Boosting':HistGradientBoostingClassifier(random_state=42,
                                                            max_iter=200, 
                                                            learning_rate=0.01,
                                                            max_depth=5),
    'XGBoost':XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss',
                            colsample_bytree=0.1,
                            learning_rate=0.005),
              
    'LightGBM':LGBMClassifier(random_state=42,
                              num_leaves=10,
                              n_estimators=175,
                              learning_rate=0.01)
}


# In[127]:


model_list_tuned_v2 = {
#     'Gradient Boosting Classifier':GradientBoostingClassifier(random_state=42,
#                                                               max_depth=2,
#                                                               n_estimators=500,
#                                                               learning_rate=0.02),
              
#     'AdaBoost Classifier':AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=1, max_depth=2),
#                                              random_state=42,
#                                              n_estimators=80,
#                                              learning_rate=0.04),
              
    'Hist Gradient Boosting':HistGradientBoostingClassifier(random_state=100,
                                                            max_iter=300, 
                                                            learning_rate=0.01,
                                                            max_depth=5),
#     'XGBoost':XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss',
#                             colsample_bytree=0.1,
#                             learning_rate=0.005),
              
#     'LightGBM':LGBMClassifier(random_state=42,
#                               num_leaves=10,
#                               n_estimators=175,
#                               learning_rate=0.01)
}

#Store results for showcasing the performance of all the models : 
y_pred_list_v1 = dict()

for name, model in model_list_tuned_v2.items():
          model.fit(X_train_model, y_train_model)
          y_pred_list_v1[name] = model.predict(X_test_model)

score_list_v1 = get_score(y_pred_list_v1, y_test_model, average='macro')


# ### Packaging the ML Model

# ### Deployment

# In[132]:


import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


# In[133]:


from sklearn import set_config

set_config(display='diagram')


# In[134]:


# Assuming merged_data is already defined
data = master_data_df_v1.copy()

# Split the data into features (X) and target variable (y)
X = data.drop(['Churn'], axis=1)
y = data['Churn']


# In[135]:


# Define numerical and categorical attributes
num_attribs = X.select_dtypes('number').columns
cat_attribs = X.select_dtypes('object').columns

# Define preprocessing pipelines for numerical and categorical attributes
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# Define ColumnTransformer to apply different preprocessing steps to different columns
preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])


# In[136]:


preprocessing


# In[137]:


X_transformed = preprocessing.fit_transform(X)


# In[138]:


X_transformed.shape


# In[139]:


X_transformed


# In[140]:


# Combine preprocessing with RandomForestClassifier in a Pipeline
clf_pipe = Pipeline([
    ("preprocessing", preprocessing),
    ("classifier", RandomForestClassifier()),
])


# In[141]:


clf_pipe


# In[142]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the pipeline
clf_pipe.fit(X_train, y_train)


# In[143]:


# Predict using the trained pipeline
clf_pipe.predict(X_test)


# In[144]:


# Save the trained pipeline as a pickle file
with open('ml1_model.pkl', 'wb') as md:
    pickle.dump(clf_pipe, md)


# In[145]:


# Load the saved model
with open('ml1_model.pkl', 'rb') as fl:
    saved_model = pickle.load(fl)


# In[146]:


# Make predictions using the loaded model
saved_model.predict(X_test)

We can combine preprocessing with Gradient Boosting, Logistic regression too in a Pipeline