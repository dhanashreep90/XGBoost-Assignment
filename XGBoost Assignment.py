#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header = None)
test_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', skiprows = 1, header = None)


# In[3]:


col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
                'occupation','relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                 'native_country', 'wage_class']
train_set.columns = col_labels
test_set.columns = col_labels


# In[4]:


train_set.head()


# In[5]:


test_set.head()


# In[6]:


df = pd.concat([train_set,test_set])


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.shape


# In[10]:


df.describe()


# In[11]:


df.size


# In[12]:


df.isnull().sum()


# In[13]:


df.replace(' ?',np.nan,inplace=True)


# In[14]:


df.isnull().sum()


# In[15]:


df.dtypes


# In[16]:


df.wage_class.unique()


# In[17]:


df = df.replace({' <=50K':0,' >50K':1,' <=50K.':0,' >50K.':1})


# In[18]:


df.head()


# In[19]:


plt.figure(figsize=(10,8))
sns.countplot(df['wage_class'])


# In[20]:


df.workclass.unique()


# In[21]:


df= df.replace(' Without-pay', ' Never-worked')


# In[22]:


df['workclass'].unique()


# In[23]:


df['workclass'].value_counts()


# In[24]:


plt.figure(figsize=(10,8))
sns.countplot(df['workclass'])
plt.xticks(rotation=60)


# In[25]:


df['workclass'].fillna('0',inplace=True)


# In[26]:


plt.figure(figsize=(10,8))
sns.countplot(df['workclass'])
plt.xticks(rotation=60)


# In[27]:


df['fnlwgt'].describe()


# In[28]:


df['fnlwgt'] = df['fnlwgt'].apply(lambda x :np.log1p(x))


# In[29]:


df['fnlwgt'].describe()


# In[30]:


df['education'].value_counts()


# In[31]:


df.columns


# In[32]:


sns.catplot(x='education',y='wage_class',data=df,height=10,palette='muted',kind='bar')
plt.xticks(rotation=60)


# In[33]:


def primary(x):
    if x in [' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' 10th', ' 11th', ' 12th']:
        return 'Primary'
    else:
        return x


# In[34]:


df['education'] = df['education'].apply(primary)


# In[35]:


sns.catplot(x='education',y='wage_class',data=df,height=10,palette='muted',kind='bar')
plt.xticks(rotation=60)


# In[36]:


df['marital_status'].value_counts()


# In[37]:


df['marital_status'].replace(' Married-AF-spouse', ' Married-civ-spouse',inplace=True)


# In[38]:


sns.catplot(x='marital_status',y='wage_class',data=df,palette='muted',kind='bar',height=8)
plt.xticks(rotation=60)


# In[39]:


df['occupation'].fillna('0',inplace=True)


# In[40]:


df['occupation'].value_counts()


# In[41]:


df['occupation'].replace(' Armed-Forces','0',inplace=True)


# In[42]:


df['occupation'].value_counts()


# In[43]:


sns.catplot(x='occupation',y='wage_class',data=df,palette='muted',kind='bar',height=8)
plt.xticks(rotation=60)


# In[44]:


df['relationship'].value_counts()


# In[45]:


df['race'].value_counts()


# In[46]:


df.columns


# In[47]:


df['sex'].value_counts()


# In[48]:


df['native_country'].unique()


# In[49]:


def native(country):
    if country in [' United-States',' Canada']:
        return 'North_America'
    elif country in [' Puerto-Rico',' El-Salvador',' Cuba',' Jamaica',' Dominican-Republic',' Guatemala',' Haiti',' Nicaragua',' Trinadad&Tobago',' Honduras']:
        return 'Central_America' 
    elif country in [' Mexico',' Columbia',' Vietnam',' Peru',' Ecuador',' South',' Outlying-US(Guam-USVI-etc)']:
        return 'South_America'
    elif country in [' Germany',' England',' Italy',' Poland',' Portugal',' Greece',' Yugoslavia',' France',' Ireland',' Scotland',' Hungary',' Holand-Netherlands']:
        return 'EU'
    elif country in [' India',' Iran',' China',' Japan',' Thailand',' Hong',' Cambodia',' Laos',' Philippines',' Taiwan']:
        return 'Asian'
    else:
        return country


# In[50]:


df['native_country'] = df['native_country'].apply(native)


# In[51]:


sns.catplot(x='native_country',y='wage_class',data=df,palette='muted',kind='bar',height=8)
plt.xticks(rotation=60)


# In[52]:


corr = df.corr()
plt.figure(figsize=(10,12))
sns.heatmap(corr,annot=True)


# In[53]:


X = df.drop(['wage_class'],axis=1)
y = df['wage_class']


# In[54]:


X.columns


# In[55]:


X_d = pd.get_dummies(X)


# In[56]:


X_d.head()


# In[57]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_d)


# In[58]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.3,random_state=101)


# In[59]:


x_train.shape


# In[60]:


y_train.shape


# In[61]:


params = [{ 'learning_rate':[0.01,0.001],
                        'max_depth': [3,5,10],
                        'n_estimators':[10,50,100,200]
                    }
                   ]


# In[71]:


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
Xbc = XGBClassifier()
Gcv = GridSearchCV(Xbc,params,scoring='accuracy',cv=5,n_jobs=3,verbose=3)
Gcv.fit(x_train,y_train)


# In[70]:


get_ipython().system('pip install xgboost')


# In[72]:


Gcv.best_params_


# In[73]:


XBC = XGBClassifier(learning_rate=0.01,max_depth=10,n_estimators=200)
XBC.fit(x_train,y_train)


# In[74]:


XBC.score(x_test,y_test)


# In[75]:


y_pred = XBC.predict(x_test)


# In[76]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[77]:


print(f'Accuracy Score:{accuracy_score(y_test,y_pred)}')
print('*'*50)
print(f'Confusion Matrix:{confusion_matrix(y_test,y_pred)}')
print('*'*50)
print(f'Classification Report: {classification_report(y_test,y_pred)}')


# In[ ]:




