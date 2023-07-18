#!/usr/bin/env python
# coding: utf-8

# # Import library for read the dataset

# In[16]:


import pandas as pd
import seaborn as sns


# In[2]:


df = pd.read_csv('Iris.csv')


# In[3]:


df


# # Data Analysis

# In[4]:


df.info()


# In[5]:


df.isnull()


# In[6]:


df.isnull().sum()


# In[9]:


print(df.columns)


# In[11]:


print(df.shape)


# In[12]:


print(df.dtypes)


# In[13]:


print(df.describe)


# In[14]:


df.head(10)


# In[15]:


df.count()


# In[17]:


#check Outliers
sns.heatmap(df.isnull(), yticklabels=False, annot=True)


# # Data Visualization

# In[35]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[19]:


df.hist(bins=10, figsize=(10, 8))
plt.tight_layout()
plt.show()


# In[32]:


# Define the colors based on conditions
colors = ['red' if length > 5.5 else 'yellow' for length in df['SepalLengthCm']]

# Scatter plot with different colors
plt.scatter(df['SepalLengthCm'], df['SepalWidthCm'], c=colors)
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.title('SepalLengthCm vs SepalWidthCm')

# Show the plot
plt.show()


# In[31]:


# Define the colors based on conditions
colors = ['red' if length > 5.5 else 'yellow' for length in df['PetalLengthCm']]

# Scatter plot with different colors
plt.scatter(df['PetalLengthCm'], df['PetalWidthCm'], c=colors)
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.title('PetalLengthCm vs PetalWidthCm')

# Show the plot
plt.show()


# In[34]:


# Select the columns for correlation
columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Calculate the coefficient matrix
correlation_matrix = df[columns].corr()

# Display the coefficient matrix
print(correlation_matrix)

# Plot the correlation matrix as a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[36]:


sns.countplot(x='Species', data=df, )
plt.show()


# In[37]:


sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm',
                hue='Species', data=df, )
 
# Placing Legend outside the Figure
plt.legend(bbox_to_anchor=(1, 1), loc=2)
 
plt.show()


# In[38]:


sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm',
                hue='Species', data=df, )
 
# Placing Legend outside the Figure
plt.legend(bbox_to_anchor=(1, 1), loc=2)
 
plt.show()


# In[39]:


sns.pairplot(df.drop(['Id'], axis = 1),
             hue='Species', height=2)


# In[41]:


plot = sns.FacetGrid(df, hue="Species")
plot.map(sns.distplot, "SepalLengthCm").add_legend()
 
plot = sns.FacetGrid(df, hue="Species")
plot.map(sns.distplot, "SepalWidthCm").add_legend()
 
plot = sns.FacetGrid(df, hue="Species")
plot.map(sns.distplot, "PetalLengthCm").add_legend()
 
plot = sns.FacetGrid(df, hue="Species")
plot.map(sns.distplot, "PetalWidthCm").add_legend()
 
plt.show()


# In[42]:


#create boxplot

def graph(y):
    sns.boxplot(x="Species", y=y, data=df)
 
plt.figure(figsize=(10,10))
     
# Adding the subplot at the specified
# grid position
plt.subplot(221)
graph('SepalLengthCm')
 
plt.subplot(222)
graph('SepalWidthCm')
 
plt.subplot(223)
graph('PetalLengthCm')
 
plt.subplot(224)
graph('PetalWidthCm')
 
plt.show()


# # Creating Machine Learning model

# In[62]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[48]:


# Assuming your features are stored in X and target variable in y
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']


# In[49]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Logistic Regression model
# 

# In[50]:


logistic_reg = LogisticRegression()

# Train the model on the training set
logistic_reg.fit(X_train, y_train)

# Make predictions on the test set
predictions = logistic_reg.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)

# Print the accuracy
print('Accuracy:', accuracy)


# #  Decision Tree Regression model
# 

# In[57]:


decision_tree_cls = DecisionTreeClassifier()

# Train the model on the training set
decision_tree_cls.fit(X_train, y_train)

# Make predictions on the test set
predictions = decision_tree_cls.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)

# Print the accuracy
print('Accuracy:', accuracy)


# In[ ]:





# In[ ]:




