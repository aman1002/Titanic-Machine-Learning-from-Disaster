
# coding: utf-8

# # Data Visualization and Machine Learning on Titanic dataset
# 
# For this notebook we will be working with the Titanic Data Set from Kaggle. This is a very famous data set.
# 
# We'll be trying visualize it and trying to predict a classification- survival or deceased.

# In[33]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[34]:


#importing the dataset
titanic = pd.read_csv("train.csv")


# In[35]:


#Let's see preview of the dataset
titanic.head()


# In[36]:


#getting overall info about the dataset
titanic.info()


# We can use seaborn to create a simple heatmap to see where we are missing data!

# In[37]:


sns.heatmap(titanic.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level.

# In[38]:


#people who survived v/s who didn't

sns.set_style('whitegrid')
sns.countplot(x='Survived', data= titanic, palette='RdBu_r')


# In[39]:


sns.countplot(x='Survived', hue='Sex', data= titanic, palette='RdBu_r')


# In[40]:


sns.countplot(x='Survived', hue='Pclass', data= titanic, palette='rainbow')


# In[41]:


#looking at the description of the dataset
titanic.describe()


# In[42]:


sns.distplot(titanic['Age'].dropna(),color='darkred',bins=30)


# The distribution plot for age is slightly right skewed. There is not much problem of outliers as such.

# In[43]:


titanic['Fare'].hist(color='green',bins=40,figsize=(8,4))


# We have a few outliers in the fare section. We can ignore them while training our model

# ## Data Cleaning

# Lets fill in the missing values of the age column. We can do this by taking mean of the age. Smarter way will be to fill in the missing blanks with the mean of the Pclass they belong too.

# In[44]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=titanic,palette='winter')


# We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.

# In[45]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[46]:


titanic['Age'] = titanic[['Age', 'Pclass']].apply(impute_age, axis = 1)


# Let's check the heatmap again

# In[47]:


sns.heatmap(titanic.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# Let's go ahead and give 1 tag to value with valid Cabin no. and 0 tag to value with NaN.

# In[50]:


def impute_cabin(col):
    
    Cabin = col[0]
    
    if type(Cabin) == str:
        return 1
    else:
        return 0


# In[51]:


titanic['Cabin'] = titanic[['Cabin']].apply(impute_cabin, axis = 1)


# Let's check the heatmap again

# In[52]:


sns.heatmap(titanic.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[53]:


titanic.head()


# In[55]:


titanic.dropna(inplace=True)


# We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

# In[57]:


titanic.info()


# In[60]:


#Let's work on a copy of our present dataset for further operations

dataset = titanic


# In[61]:


sex = pd.get_dummies(dataset['Sex'],drop_first=True)
embark = pd.get_dummies(dataset['Embarked'],drop_first=True)

dataset.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

dataset = pd.concat([dataset,sex,embark],axis=1)


# In[62]:


dataset.head()


# ## Building regression models
# 
# Let's start by splitting our data into a training set and test set

# In[63]:


#Train Test Split

from sklearn.model_selection import train_test_split


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(dataset.drop('Survived',axis=1), 
                                                    dataset['Survived'], test_size=0.25, 
                                                    random_state=101)


# ## Training and Predicting

# In[65]:


from sklearn.linear_model import LogisticRegression


# **Using Logistic Regression**

# In[66]:


regressor = LogisticRegression()
regressor.fit(X_train, y_train)


# In[67]:


pred = regressor.predict(X_test)


# ### Let's evaluate

# In[68]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss


# In[70]:


print(classification_report(y_test, pred))
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')
print(accuracy_score(y_test, pred))


# **Using SVM**

# In[71]:


from sklearn.svm import SVC


# In[73]:


regressor2 = SVC()
regressor2.fit(X_train, y_train)


# In[74]:


pred2 = regressor2.predict(X_test)


# In[75]:


print(classification_report(y_test, pred2))
print('\n')
print(confusion_matrix(y_test, pred2))
print('\n')
print(accuracy_score(y_test, pred2))


# **Using K-NN**

# In[76]:


from sklearn.neighbors import KNeighborsClassifier


# In[77]:


regressor3 = KNeighborsClassifier(n_neighbors=5)
regressor3.fit(X_train, y_train)


# In[78]:


pred3 = regressor3.predict(X_test)


# In[79]:


print(classification_report(y_test, pred3))
print('\n')
print(confusion_matrix(y_test, pred3))
print('\n')
print(accuracy_score(y_test, pred3))


# **Using Adaboost Classifier**

# In[80]:


from sklearn.ensemble import AdaBoostClassifier


# In[81]:


regressor4 = AdaBoostClassifier()
regressor4.fit(X_train, y_train)


# In[82]:


pred4 = regressor4.predict(X_test)


# In[83]:


print(classification_report(y_test, pred4))
print('\n')
print(confusion_matrix(y_test, pred4))
print('\n')
print(accuracy_score(y_test, pred4))

