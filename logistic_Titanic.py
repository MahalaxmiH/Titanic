# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 05:40:07 2018

@author: lenovo
"""
### Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

########Import Libraries
train = pd.read_csv("C:\\Users\\lenovo\\Desktop\\Datascience development\\Logistic\\Logistic\\titanic_train.csv")
train.head()



#### # Exploratory Data Analysis -Missing Data 

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')

sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)

train['Age'].hist(bins=30,color='darkred',alpha=0.7)

sns.countplot(x='SibSp',data=train)

train['Fare'].hist(color='green',bins=40,figsize=(8,4))


### Data cleaning -We want to fill in missing age data instead of just 
#dropping the missing age data rows. One way to do this is by filling 
#in the mean age of all the passengers (imputation)

plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')

### filling the missing information

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
    
    train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
    
    sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
    
`
    
    train.head()
    
    train.dropna(inplace=True)
    
    ## Converting Categorical Features 
    train.info()
    
    sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train = pd.concat([train,sex,embark],axis=1)

train.head()

# Building a Logistic Regression model

## Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)

## Training and Predicting
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

### Model evaluation 
#from sklearn.metrics import classification_report
            
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test, predictions))

###print(classification_report(y_test,predictions))





