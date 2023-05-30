# Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment
# AIM:
To Perform Data Science Process on a complex dataset and save the data to a file.

# ALGORITHM
# STEP 1 :
Read the given Data

# STEP 2 :
Clean the Data Set using Data Cleaning Process

# STEP 3 :
Apply Feature Generation/Feature Selection Techniques on the data set

# STEP 4 :
Apply EDA /Data visualization techniques to all the features of the data set

# CODE:
# Developed by: R BRINDHA
# Register No: 212222230023
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("/content/StudentsPerformance - StudentsPerformance.csv.csv")
print(data)

data.info()

data.isnull().sum()
```
# data cleaning
```
data['test preparation course']=data['test preparation course'].fillna(data['test preparation course'].mode()[0])
data['math score']=data['math score'].fillna(data['math score']).fillna(data['math score'].mean())
data['writing score']=data['writing score'].fillna(data['writing score']).fillna(data['reading score'].median())

data.isnull().sum()

data.describe()

data.head()
```
# removing outliers
```
Q1=data['math score'].quantile(0.25)
Q3=data['math score'].quantile(0.75)
IQR=Q3-Q1
lower=Q1-1.5*IQR
upper=Q3+1.5*IQR
df=data[(data['math score']>=lower) & (data['math score']<=upper)] 
print(df)   #new dataframe.


outliers=data[(data['math score']<lower) | (data['math score']>upper)] 
print(outliers)

df.shape
```
# Feature generation
```
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
df1=df.copy()
r=['group A','group B','group C','group D','group E']
enc=OrdinalEncoder(categories=[r])
enc.fit_transform(df1[['race/ethnicity']])
df1['neword1']=enc.fit_transform(df1[['race/ethnicity']])
df1 


df2=df1.copy()
le=LabelEncoder()
df2['neword2']=le.fit_transform(df2['race/ethnicity'])
df2

from sklearn.preprocessing import OneHotEncoder
df3=df.copy()
ohe=OneHotEncoder(sparse=False)
enc=pd.DataFrame(ohe.fit_transform(df3[['lunch']]))
df3=pd.concat([df3,enc],axis=1)
df3.head()

!pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df4=df.copy()
newdata=be.fit_transform(df4['test preparation course'])
df4=pd.concat([df,newdata],axis=1)
df4.head()
```
# heatmap
```
data.corr()
plt.subplots(figsize=(7,5))
sns.heatmap(data.corr(),annot=True)
```
# Data visualization
# Scatter plot of math score vs. reading score
```
plt.scatter(data['math score'], data['reading score'])
plt.xlabel('Math Score')
plt.ylabel('Reading Score')
plt.title('Math Score vs. Reading Score')
plt.show()

sns.barplot(x='gender',y='reading score',data=df)

sns.boxplot(x="math score",data=df)
```
# OUTPUT:
![1](https://github.com/Brindha77/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118889143/1e07c929-4fc4-4536-bd57-31d10e9cf3d4)
![2](https://github.com/Brindha77/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118889143/7bbf632a-36af-46c4-b353-caa5a0e0651c)
![3](https://github.com/Brindha77/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118889143/ae4cc29f-8e4f-4edc-ba00-6bcea86f504c)
![4](https://github.com/Brindha77/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118889143/5de88adc-f208-4553-9e05-d688d6464baf)
![5](https://github.com/Brindha77/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118889143/eea0d934-a3b5-435b-9561-149b728ecbc7)
![6](https://github.com/Brindha77/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118889143/4317f021-b169-49d5-90bc-62acc32cad5a)
![7](https://github.com/Brindha77/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118889143/9954ccf3-5143-411e-8029-60552ddf2e75)
![8](https://github.com/Brindha77/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118889143/bcb8b928-9758-44fd-a8e6-76497a0cfa44)
![9](https://github.com/Brindha77/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118889143/bc8cf80f-62a4-4d10-9427-3e54ac1c1983)
![10](https://github.com/Brindha77/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118889143/18f9eef0-c1b1-4972-9b21-04ce274b6448)
![11](https://github.com/Brindha77/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118889143/7ddbf2c1-b6a5-4772-a917-735768751661)
![12](https://github.com/Brindha77/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118889143/8c9df4f0-d588-4eb8-a55b-6445cd45fd15)
![13](https://github.com/Brindha77/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118889143/043edf89-eac4-40e8-91de-f7942e09ca3d)
![14](https://github.com/Brindha77/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118889143/0f87261a-07d3-4fd8-af31-c98ff0b832b5)
![15](https://github.com/Brindha77/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118889143/f9c4b65d-f4cc-45f9-972d-b925014f6707)
![16](https://github.com/Brindha77/Ex-10-Data-Science-Process-on-Complex-Dataset-Assignment/assets/118889143/0604f106-71c2-4355-8751-e17b2f7eaafa)
# RESULT:
Hence, Data Science Process is performed on a complex dataset and saved the data to a file.
