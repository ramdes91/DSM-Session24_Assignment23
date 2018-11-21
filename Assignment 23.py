
# coding: utf-8

# In[134]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.model_selection import train_test_split


# Read the csv file from given URL

# In[135]:


titanic = pd.read_csv('https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv')


# In[136]:


titanic.head()


# In[137]:


titanic.set_index('PassengerId', drop = True, inplace = True)
titanic.head()


# In[138]:


del titanic['Name']


# In[139]:


titanic.head()
titanic.describe()
del titanic['Ticket']


# In[140]:


titanic.head()
titanic['Embarked'].dropna(inplace = True)
titanic.head()


# In[141]:


del titanic['Cabin']
titanic.head()


# In[143]:


del titanic['Embarked']
titanic.head()


# In[144]:


#numerical value for Sex in the column gender
def gender(st):
    if st == 'male':
        return 1
    else:
        return 2
titanic['gender'] = titanic.Sex.apply(gender)
titanic.head()


# In[145]:


del titanic['Sex']


# In[146]:


class_1_surv = titanic[(titanic['Pclass'] == 1)].mean()['Survived']
class_2_surv = titanic[(titanic['Pclass'] == 2)].mean()['Survived']
class_3_surv = titanic[(titanic['Pclass'] == 3)].mean()['Survived']
class_1_surv


# In[147]:


#graph based on class survival
my_xticks = ['Class 1','Class 2','Class 3']
x = [1,2,3]
plt.xticks(x, my_xticks)
plt.plot(x, [class_1_surv, class_2_surv, class_3_surv])
plt.show()


# In[148]:


males = len(titanic[titanic['gender'] == 1])
females = len(titanic[titanic['gender'] == 2])


# In[151]:


# male v/s female pie chart
plt.pie([males,females],
       labels = ['Male', 'Female'],
       explode = [0.10, 0],
       startangle = 0)
plt.show()


# In[153]:


#scatter graph based on age and gender
plt.scatter(titanic['Age'], titanic['gender'])
plt.show()


# In[154]:


survived = titanic[titanic['Survived'] == 1]
surv_avg = survived.mean()['Age']
not_survived = titanic[titanic['Survived'] == 0]
nsurv_avg = not_survived.mean()['Age']


# In[155]:


def fillavg(survv):
    if survv == 1:
        return surv_avg
    else:
        return nsurv_avg


# In[156]:


titanic['avg'] = titanic.Survived.apply(fillavg)


# In[157]:


titanic.Age.fillna(titanic['avg'], inplace = True)
titanic.describe()


# In[158]:


del titanic['avg']


# In[159]:


titanic.head()


# In[160]:


x_train,x_test,y_train,y_test = train_test_split(titanic.drop('Survived',axis = 1),titanic['Survived'],test_size = 0.30,random_state = 101)


# In[161]:


#Decision Tree
clf1 = tree.DecisionTreeClassifier()
clf1.fit(x_train, y_train)
clf1.score(x_test, y_test)

