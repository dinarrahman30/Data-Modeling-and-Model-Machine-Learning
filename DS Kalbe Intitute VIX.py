#!/usr/bin/env python
# coding: utf-8

# ## Informasi Data Diri
# 
# Nama: Dinar Wahyu Rahman
# 
# No. Telp: 083806242160
# 
# Email: dinarrahman30@gmail.com
# 
# LinkedIn: <a href="https://www.linkedin.com/in/dinar-wahyu-rahman-00a405162/">Dinar Wahyu Rahman</a>
# 
# Alamat: Kota Jakarta Barat, DKI Jakarta

# ### membuat model prediktif menggunakan regresi

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[2]:


for dirname, _, filenames in os.walk('D:/Case Study Data Scientist-20230712T153055Z-001/Case Study Data Scientist'): #membuka file
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


customer_df = pd.read_csv('D:/Case Study Data Scientist-20230712T153055Z-001/Case Study Data Scientist/Case Study - Customer.csv', sep=';')
product_df = pd.read_csv('D:/Case Study Data Scientist-20230712T153055Z-001/Case Study Data Scientist/Case Study - Product.csv', sep=';')
store_df = pd.read_csv('D:/Case Study Data Scientist-20230712T153055Z-001/Case Study Data Scientist/Case Study - Store.csv', sep=';')
transaction_df = pd.read_csv('D:/Case Study Data Scientist-20230712T153055Z-001/Case Study Data Scientist/Case Study - Transaction.csv', sep=';')


# In[4]:


customer_df.sample(3)


# In[5]:


product_df.sample()


# In[6]:


store_df.sample()


# In[7]:


transaction_df.sample()


# #### find missing value

# In[8]:


customer_df.isna().sum()


# In[9]:


product_df.isna().sum()


# store_df.isna().sum()

# In[10]:


transaction_df.isna().sum()


# In[11]:


customer_df['Marital Status'] = customer_df['Marital Status'].fillna('Unknown')


# In[12]:


customer_df.isna().sum()


# #### change data type

# In[13]:


transaction_df['Date'] = pd.to_datetime(transaction_df['Date'], format='%d/%m/%Y')


# #### merge data

# In[14]:


#Merging all data frame on transaction
df = pd.merge(transaction_df, customer_df, on='CustomerID', how='inner')
df = pd.merge(df, product_df, on='ProductID', how='inner')
df = pd.merge(df, store_df, on='StoreID')
df.sample(3)


# In[15]:


df.rename(columns={'Price_x' : 'Price'}, inplace=True)


# In[16]:


df.drop(['Latitude','Longitude', 'Price_y'],axis=1,inplace=True)
df.sample()


# In[17]:


df.info()


# In[18]:


df


# ### membuat model regresi
# 
# memprediksi total qty harian dari product yang terjual.

# In[19]:


get_ipython().system('pip install pmdarima')


# In[20]:


from sklearn.impute import KNNImputer
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from itertools import permutations
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")


# #### forecasting data

# In[21]:


#Forecast data
df_fore = df[['Date','Qty']]
df_fore= df_fore.groupby('Date')[['Qty']].sum()
df_fore.head(3)


# In[22]:


#decompose = seasonal_decompose(df_fore)

#fig,ax = plt.subplots(3,1,figsize=(15,12))
#decompose.trend.plot(ax=ax[0])
#ax[0].set_title('Trend')
#decompose.seasonal.plot(ax=ax[1])
#ax[1].set_title('Seasonal')
#decompose.resid.plot(ax=ax[2])
#ax[2].set_title('Residual')

#plt.tight_layout()
#plt.show()


# In[23]:


#df_fore.plot()
#plt.show()


# #### transform data

# In[24]:


#Transform data to log
df_fore = np.log(df_fore)
df_fore.head(3)


# In[25]:


df_fore.plot()
plt.show()


# In[26]:


#Split train and test
df_train = df_fore.iloc[:-31]
df_test = df_fore.iloc[-31:]


# #### cek data stationary

# In[27]:


#ADF test
from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(df_train)
print(f'p-value: {adf_test[1]}')


# In[28]:


#ACF and PACF plot
acf_original = plot_acf(df_train)
pacf_original = plot_pacf(df_train)


# In[29]:


#ADF test
from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(df_train)
print(f'p-value: {adf_test[1]}')


# ### regresion with model ARIMA
# 
# dengan metode autofit ARIMA

# In[30]:


#auto-fit ARIMA
auto_arima = pm.auto_arima(df_train, stepwise=False, seasonal=False)
auto_arima


# In[31]:


#Manual parameter tuning
model = ARIMA(df_train, order=(70, 2, 1))
model_fit = model.fit()


# In[33]:


#plot forecasting
forecast_test = model_fit.forecast(len(df_test))
forecast_auto = auto_arima.predict(len(df_test))

df_plot = df_fore[['Qty']].iloc[-61:]

df_plot['forecast_test'] = [None]*(len(df_plot)-len(forecast_test)) + list(forecast_test)
df_plot['forecast_auto'] = [None]*(len(df_plot)-len(forecast_auto)) + list(forecast_auto)

df_plot.plot()
plt.show()


# In[34]:


#Auto-fit ARIMA metrics

mae = mean_absolute_error(df_test, forecast_auto)
mape = mean_absolute_percentage_error(df_test, forecast_auto)
rmse = np.sqrt(mean_squared_error(df_test, forecast_auto))

print(f'mae - auto: {round(mae,4)}')
print(f'mape - auto: {round(mape,4)}')
print(f'rmse - auto: {round(rmse,4)}')


# #### forecasting overall qty

# In[35]:


#Apply model to forecast data
model = ARIMA(df_fore, order=(70, 2, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)


# In[36]:


#Re-trasnform data
df_fore = np.exp(df_fore)
forecast = np.exp(forecast)


# In[37]:


#Plot forecasting
plt.figure(figsize=(12,5))
plt.plot(df_fore)
plt.plot(forecast,color='orange')
plt.title('Qty Sold Forecasting')
plt.show()


# In[38]:


forecast.mean()


# Dapat disimpulkan bahwa untuk qty penjualan bulan depan adalah sekitar rata-rata 44 pcs per harinya.

# #### forecasting each product

# In[39]:


#Forecast for next 30 days for each product
list_prod = df['Product Name'].unique()

dfp = pd.DataFrame({'Date':pd.date_range(start='2023-01-01',end='2023-01-30')})
dfp = dfp.set_index('Date')
for i in list_prod:
    df_ = df[['Date','Product Name','Qty']]
    df_ = df_[df_['Product Name']==i]
    df_= df_.groupby('Date')[['Qty']].sum()
    df_ = df_.reset_index()

    df_t = pd.DataFrame({'Date':pd.date_range(start='2022-01-01',end='2022-12-31')})
    df_t = df_t.merge(df_,how='left',on='Date')
    df_t = df_t.fillna(0)
    df_t = df_t.set_index('Date')

    model1 = ARIMA(df_t, order=(70, 2, 1))
    model_fit1 = model1.fit()
    forecast1 = model_fit1.forecast(steps=30)
    dfp[i] = forecast1.values
    
dfp.head()


# In[40]:


#Plot forecasting
plt.figure(figsize=(12,5))
plt.plot(dfp)
plt.legend(dfp.columns,loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Products Qty Sold Forecasting')
plt.show()


# In[41]:


#Products Quantity forecast
round(dfp.describe().T['mean'],0)


# ### melakukan clustering
# 
# Tujuan dari pembuatan model machine learning ini adalah untuk dapat membuat cluster customer-customer yang mirip.

# In[43]:


df


# In[44]:


df_clust = df.groupby('CustomerID').agg({'TransactionID':'count','Qty':'sum','TotalAmount':'sum'})
df_clust


# In[45]:


#Check outliers on new dataset
features = df_clust.columns
fig, ax = plt.subplots(1,len(df_clust.columns),figsize=(12,5))
for i in range(0,len(df_clust.columns)):
    sns.boxplot(data=df_clust,y=features[i],ax=ax[i])
plt.tight_layout()
plt.show()


# In[46]:


#Standarisasi dataset baru
X = df_clust.values
X_std = StandardScaler().fit_transform(X)
df_std = pd.DataFrame(data=X_std,columns=df_clust.columns)
df_std.isna().sum()


# #### clustering with Kmeans

# In[47]:


# Kmeans n_cluster = 3
#Clustering Kmeans
kmeans_3 = KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
kmeans_3.fit(X_std)

#Tambah clusters label pada dataset
df_cl3 = pd.DataFrame(data=X_std,columns=df_clust.columns)
df_cl3['cluster'] = kmeans_3.labels_
df_cl3.sample(3)


# In[48]:


#PLot Before PCA
plt.figure(figsize=(4,4))
sns.pairplot(data=df_cl3,hue='cluster',palette='Set1')
plt.show()


# In[49]:


#PCA
pcs_3 = PCA(n_components=2).fit_transform(X_std)
pdf_3 = pd.DataFrame(data=pcs_3,columns=['pc1','pc2'])
pdf_3['cluster'] = df_cl3['cluster']
pdf_3.describe().T


# In[50]:


#PCA plot
fig,ax = plt.subplots(2,1,figsize=(10,8))
# plt.figure(figsize=(10,5))
sns.scatterplot(data=pdf_3,x='pc1',y='pc2',hue='cluster',palette='Set1',ax=ax[0])
ax[0].set_title('PCA scatter')
sns.kdeplot(data=pdf_3,x='pc1',hue='cluster',palette='Set1',fill=True,ax=ax[1])
ax[1].set_title('PCA kde')
plt.tight_layout()
plt.show()


# #### Sillhouette analysis

# In[51]:


n_clust = list(range(2,11))
silhouette_avg = []
for i in n_clust:
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    
    labels = kmeans.fit_predict(X_std)
    silhouette_avg.append(silhouette_score(X_std,labels))
    
plt.plot(n_clust,silhouette_avg)
plt.xlabel('n_clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette analysis for optimal n_clusters')
plt.show()


# #### Cluster Analysis

# In[57]:


df_clust['cluster'] = kmeans_3.labels_

df['recency'] = (pd.to_datetime('2023-01-01') - df_['Date']).dt.days.astype('int')
df_r = df.groupby('CustomerID').agg({'recency':'min'})

df_rfm = df_clust.copy()
df_rfm['recency'] = df_r['recency']


# In[59]:


fig, ax = plt.subplots(3,1,figsize=(10,10))
sns.histplot(data=df_rfm,x='recency',hue='cluster',palette='Set1',ax=ax[0],kde=True)
ax[0].set_title('Recency')
sns.histplot(data=df_rfm,x='TransactionID',hue='cluster',palette='Set1',ax=ax[1],kde=True)
ax[1].set_title('Frequency')
sns.histplot(data=df_rfm,x='TotalAmount',hue='cluster',palette='Set1',ax=ax[2],kde=True)
ax[2].set_title('Monetary')

plt.tight_layout()
plt.show()


# ### Summary
# 
# 
# 0 = New Customer
# 
# strategy -> because it is a new customer, the right company strategy is to provide attractive offers that can increase loyalty in the form of:
# * gift discounts
# * good customer support, and 
# * create new customer satisfaction surveys.
# 
# 
# 1 = Potensial Customer
# 
# strategy -> because they are potential customers, the right company strategy is to need attractive offers that can convert them into loyal customers in the form of: 
# * gift discounts
# * provide proactive communication
# * identify their needs by providing good service, and 
# * carry out regular follow-ups if the potential customer has not decided to buy.
# 
# 
# 2 = Loyal Customer
# 
# strategy -> because they are loyal customers, the right company strategy is to need attractive offers that can maintain and improve good relations with them in the form of offering loyalty programs specifically for loyal customers such as:
# * exclusive product offers
# * gift discounts, and 
# * creating special surveys for loyal customers.
