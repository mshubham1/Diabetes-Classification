#!/usr/bin/env python
# coding: utf-8

# # Diabetes Prediction

# The objective of the project is to classify weather a person is having diabetes or not.
# The datset consist of servral dependent and independent variable.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("C:/Users/hp/Documents/datasets/diabetes.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.isnull().sum()


# In[7]:


data=data.drop_duplicates()


# In[8]:


#checking for 0 values in features
print(data[data['BloodPressure']==0].shape[0])
print(data[data['Glucose']==0].shape[0])
print(data[data['SkinThickness']==0].shape[0])
print(data[data['Insulin']==0].shape[0])
print(data[data['BMI']==0].shape[0])


# In[9]:


# replacing 0 value with the mean of that column
data['BloodPressure']=data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['Glucose']=data['Glucose'].replace(0,data['Glucose'].mean())
data['SkinThickness']=data['SkinThickness'].replace(0,data['SkinThickness'].mean())
data['Insulin']=data['Insulin'].replace(0,data['Insulin'].mean())
data['BMI']=data['BMI'].replace(0,data['BMI'].mean())


# In[10]:


## countplot for checking balanace of data
import warnings #avoid warning flash
warnings.filterwarnings('ignore')
sns.countplot(data['Outcome'])


# In[11]:


# the distribution of data skewed or normal distibution
data.hist(bins=10,figsize=(10,10))
plt.show


# In[12]:


sns.pairplot(data=data,hue='Outcome')
plt.show()


# In[13]:


plt.figure(figsize=(20,15))
sns.heatmap(data.corr(),annot=True)


# In[14]:


for feature in data.columns:
    sns.boxplot(data[feature])
    plt.title(feature)
    plt.figure(figsize=(15,15))


# In[15]:


## handling outliers
y=data['Outcome']
X=data.drop('Outcome',axis=1)


# In[16]:


X.columns


# In[17]:


IQR=X['Pregnancies'].quantile(0.75)-X['Pregnancies'].quantile(0.25)
upper_limit=X['Pregnancies'].quantile(0.75)+(IQR*1.5)
lower_limit=X['Pregnancies'].quantile(0.25)-(IQR*1.5)
print(lower_limit,upper_limit)


# In[18]:


X.loc[X['Pregnancies']>=13.5,'Pregnancies']=13.5
X.loc[X['Pregnancies']<=-6.5,'Pregnancies']=-6.5


# In[19]:


IQR=X['Glucose'].quantile(0.75)-X['Glucose'].quantile(0.25)
upper_limit=X['Glucose'].quantile(0.75)+(IQR*1.5)
lower_limit=X['Glucose'].quantile(0.25)-(IQR*1.5)
print(lower_limit,upper_limit)


# In[20]:


X.loc[X['Glucose']>=201,'Glucose']=201
X.loc[X['Glucose']<=39,'Glucose']=39


# In[22]:


IQR=X['BloodPressure'].quantile(0.75)-X['BloodPressure'].quantile(0.25)
upper_limit=X['BloodPressure'].quantile(0.75)+(IQR*1.5)
lower_limit=X['BloodPressure'].quantile(0.25)-(IQR*1.5)
print(lower_limit,upper_limit)


# In[23]:


X.loc[X['BloodPressure']>=104,'BloodPressure']=104
X.loc[X['BloodPressure']<=40,'BloodPressure']=40


# In[24]:


IQR=X['SkinThickness'].quantile(0.75)-X['SkinThickness'].quantile(0.25)
upper_limit=X['SkinThickness'].quantile(0.75)+(IQR*1.5)
lower_limit=X['SkinThickness'].quantile(0.25)-(IQR*1.5)
print(lower_limit,upper_limit)


# In[25]:


X.loc[X['SkinThickness']>=49.19,'SkinThickness']=49.19
X.loc[X['SkinThickness']<=3.34,'SkinThickness']=3.34


# In[26]:


IQR=X['Insulin'].quantile(0.75)-X['Insulin'].quantile(0.25)
upper_limit=X['Insulin'].quantile(0.75)+(IQR*1.5)
lower_limit=X['Insulin'].quantile(0.25)-(IQR*1.5)
print(lower_limit,upper_limit)


# In[27]:


X.loc[X['Insulin']>=198.42,'Insulin']=198.42
X.loc[X['Insulin']<=8.62,'Insulin']=8.62


# In[28]:


IQR=X['BMI'].quantile(0.75)-X['BMI'].quantile(0.25)
upper_limit=X['BMI'].quantile(0.75)+(IQR*1.5)
lower_limit=X['BMI'].quantile(0.25)-(IQR*1.5)
print(lower_limit,upper_limit)


# In[29]:


X.loc[X['BMI']>=50.25,'BMI']=50.25
X.loc[X['BMI']<=13.84,'BMI']=13.84


# In[30]:


IQR=X['DiabetesPedigreeFunction'].quantile(0.75)-X['DiabetesPedigreeFunction'].quantile(0.25)
upper_limit=X['DiabetesPedigreeFunction'].quantile(0.75)+(IQR*1.5)
lower_limit=X['DiabetesPedigreeFunction'].quantile(0.25)-(IQR*1.5)
print(lower_limit,upper_limit)


# In[31]:


X.loc[X['DiabetesPedigreeFunction']>=1.2,'DiabetesPedigreeFunction']=1.2
X.loc[X['DiabetesPedigreeFunction']<=-0.32,'DiabetesPedigreeFunction']=-0.32


# In[32]:


IQR=X['Age'].quantile(0.75)-X['Age'].quantile(0.25)
upper_limit=X['Age'].quantile(0.75)+(IQR*1.5)
lower_limit=X['Age'].quantile(0.25)-(IQR*1.5)
print(lower_limit,upper_limit)


# In[33]:


X.loc[X['Age']>=66.5,'Age']=66.5
X.loc[X['Age']<=-1.5,'Age']=-1.5


# In[34]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(X)
SSX=scaler.transform(X)


# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(SSX,y,test_size=0.2,random_state=2)


# In[36]:


X_train.shape,y_train.shape


# In[37]:


X_test.shape,y_test.shape


# In[38]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='liblinear',multi_class='ovr') #creating an object
lr.fit(X_train,y_train) #training the model


# In[39]:


lr_pred=lr.predict(X_test)


# In[40]:


lr.score(X_train,y_train)


# In[41]:


lr.score(X_test,y_test)


# In[42]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)


# In[43]:


knn_pred=knn.predict(X_test)


# In[44]:


knn.score(X_test,y_test) 


# In[45]:


knn.score(X_train,y_train)


# In[46]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,y_train)


# In[47]:


nb.predict(X_test)


# In[48]:


nb.score(X_test,y_test) #test score


# In[49]:


nb.score(X_train,y_train) #training score


# In[50]:


from sklearn.svm import SVC
sv=SVC()
sv.fit(X_train,y_train)


# In[51]:


sv.predict(X_test)


# In[52]:


sv.score(X_test,y_test) #test score


# In[53]:


sv.score(X_train,y_train) #training score


# In[54]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[55]:


dt_pred=dt.predict(X_test)


# In[56]:


dt.score(X_test,y_test) #test score


# In[57]:


dt.score(X_train,y_train) #training score


# In[58]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)


# In[59]:


rf.predict(X_test)


# In[60]:


rf.score(X_test,y_test) #test score


# In[61]:


rf.score(X_test,y_test) #test score


# In[63]:


import joblib
joblib.dump(rf,"RandomForestClassifier.pkl")
joblib.dump(lr,"LogisticRegression.pkl")
joblib.dump(knn,"KNeighborsClassifier.pkl")
joblib.dump(nb,"GaussianNB.pkl")
joblib.dump(sv,"SVC_Classifier.pkl")

