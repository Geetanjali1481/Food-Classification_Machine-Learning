#!/usr/bin/env python
# coding: utf-8

# Project By: 
# Geetanjali Sharma 
# and Michael Quinn

# Problem Statement: 
# 
# For this project we are trying to classify foods into 4 different categories: "Meats", "Fruits", "Vegetables", and "Dairy/Eggs". 
# The input for the classification will be the nutritional facts of the food products. 

# Data Preprocessing
# 1. Scaling the data to reduce the rows to a 1000 and the columns to 17.

# In[125]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[180]:


#import pandas
import pandas as pd
import numpy as np
np.random.seed(100)

#load in data
data = pd.read_csv('nndb_flat.csv')

#check data types and null values
data.info()


# In[181]:


#check unique categories for "FoodGroup"
data['FoodGroup'].unique()


# In[182]:


#drop columns of object types, columns with many NaN values, and columns with minor mineral values
data = data.drop(data.iloc[:,2:4], axis=1)
data = data.drop(data.iloc[:,25:53], axis=1)
data = data.drop(data.iloc[:,18:22], axis=1)
data = data.drop(['ID','CommonName', 'MfgName', 'ScientificName'], axis=1)

#drop some ambiguous categories from "FoodGroup" column
indexes = data[(data['FoodGroup']=='Meals, Entrees, and Side Dishes') | (data['FoodGroup']=='Restaurant Foods') | (data['FoodGroup']=='American Indian/Alaska Native Foods')
  | (data['FoodGroup']=='Fats and Oils') | (data['FoodGroup']=='Soups, Sauces, and Gravies') | (data['FoodGroup']=='Breakfast Cereals') | (data['FoodGroup']=='Snacks')
  | (data['FoodGroup']=='Beverages') | (data['FoodGroup']=='Nut and Seed Products') | (data['FoodGroup']=='Sweets') | (data['FoodGroup']=='Cereal Grains and Pasta')
  | (data['FoodGroup']=='Baked Products') | (data['FoodGroup']=='Baby Foods') | (data['FoodGroup']=='Spices and Herbs') | (data['FoodGroup']=='Legumes and Legume Products')
  | (data['FoodGroup']=='Fast Foods')].index
data.drop(indexes,inplace=True)

#displaying remaining unique categories for "FoodGroup" columns
data['FoodGroup'].unique()


# In[184]:


#keeping 4 unique classes for the "FoodGroup" column for simplicity of classification
data['FoodGroup'] = data['FoodGroup'].replace(['Poultry Products', 'Sausages and Luncheon Meats', 'Pork Products', 'Beef Products', 'Finfish and Shellfish Products', 'Lamb, Veal, and Game Products'], 'Meats')
data['FoodGroup'] = data['FoodGroup'].replace(['Fruits and Fruit Juices'], 'Fruits')
data['FoodGroup'] = data['FoodGroup'].replace(['Vegetables and Vegetable Products'], 'Vegetables')
data['FoodGroup'] = data['FoodGroup'].replace(['Dairy and Egg Products'], 'Dairy/Eggs')

print(data['FoodGroup'].unique())
print(data['FoodGroup'].value_counts())


# In[130]:


#keeping 250 data for each category (total data = 1000)

#creating a function to scale down data to 1000 rows and keeping 250 data for each category
def scale_row (row_name, data):
  f = 250/data[(data['FoodGroup']==row_name)].shape[0]
  data = data.drop(data[(data['FoodGroup']==row_name)].sample(frac=1 - f).index)
  return data

#calling the fuction for each "FoodGroup" category
data = scale_row('Meats', data)
data = scale_row('Fruits', data)
data = scale_row('Vegetables', data)
data = scale_row('Dairy/Eggs', data)

#displaying the "FoodGroup" category 
data['FoodGroup'].value_counts()


# 2. EDA

# In[131]:


#checking the shape of cleaned data
data.shape


# In[132]:


#displaying the first 5 rows
data.head(5)


# In[133]:


#checking data types after cleaning
data.info()


# We can see that there are no null values in the dataset.

# In[134]:


#statistics
data.describe()


# In[135]:


#printing the correlation matrix
corr_matrix = data.corr()
corr_matrix


# 3. Data Visualization 

# In[136]:


#importing matplotlib
import matplotlib.pyplot as plt

#checking data distribution (histgram)
data.hist(bins=50, figsize=(20,15))

plt.show()


# In[137]:


#checking data distribution for "Meats" category in the "FoodGroup" column
data[(data['FoodGroup']=='Meats')].hist(bins=50, figsize=(20,15))

plt.show()


# In[138]:


#checking data distribution for "Fruits" category in the "FoodGroup" column
data[(data['FoodGroup']=='Fruits')].hist(bins=50, figsize=(20,15))

plt.show()


# In[139]:


#checking data distribution for "Vegetables" category in the "FoodGroup" column 
data[(data['FoodGroup']=='Vegetables')].hist(bins=50, figsize=(20,15))

plt.show()


# In[140]:


#checking data distribution for "Dairy/Eggs" category in the "FoodGroup" column
data[(data['FoodGroup']=='Dairy/Eggs')].hist(bins=50, figsize=(20,15))

plt.show()


# Observations:
# 1. Meats have higher amount of Energy_kcal, Protein_g, Fat_g and substantially higher amount of VitB6_mg. They have some amount of Calcium_mg and Iron_mg. They have low amount of Carb_g, Sugar_g and Fiber_g. 
# 2. Fruits have a high amount of Energy_kcal, Protein_g, and Fat_g. They have some amount of Sugar_g, Fiber_g, VitB6_mg, Copper_mg and Iron_mg. They have low amount of Fat_g, Calcium_mg, VitB12_mcg, and VitC_mg.
# 3. Vegetables have a higher amount of Energy_kcal and Carb_b. They have some amount of Fiber_g, Protein_g, and Calcium_mg. They have low amount of VitB12_mcg and VitC_g.
# 4. Diary and Egg Products have a high amount of Energy_kcal,  Calcium_mg, Protein_g, and Carb_g. They have some amount of Fat_g, Sugar_g, and VitB12_mcg. They have low amount of VitC_mg, Fiber_G, and Copper_mcg.   

# In[141]:


#scaling data to draw a box plot for a general visualization 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = data
data_scaled = data_scaled.drop(['FoodGroup'], axis=1)
# data_scaled = scaler.fit_transform(data_scaled)
print(data_scaled.idxmax())

data_scaled = data_scaled.apply(lambda x:(x - x.min())/(x.max() - x.min() ), axis= 0 )
print(data_scaled.describe())
data_scaled.plot(kind='box', figsize=(30, 15))


# Models

# 1. Decision Tree

# In[142]:


#adding labels for classification
data['FoodGroup'] = data['FoodGroup'].replace(['Meats'], 0)
data['FoodGroup'] = data['FoodGroup'].replace(['Fruits'], 1)
data['FoodGroup'] = data['FoodGroup'].replace(['Vegetables'], 2)
data['FoodGroup'] = data['FoodGroup'].replace(['Dairy/Eggs'], 3)


# In[143]:


from sklearn.model_selection import train_test_split

#importing DecisionTreeClassifier from sklearn
from sklearn.tree import DecisionTreeClassifier

y = data['FoodGroup']
X = data.drop("FoodGroup", axis=1)

#splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 100)

#implementing decision tree with max_depth = 2
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X_train, y_train)


# In[144]:


#plotting the tree
from sklearn import tree
from IPython.display import Image 
import matplotlib.pyplot as plt

plt.figure(figsize=(50,50))
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (3,3), dpi=1000)
data1 = data.drop(['FoodGroup'], axis=1)
fn=data1.columns
cn=['Meats', 'Fruits', 'Vegetables', 'Dairy/Eggs']
tree.plot_tree(tree_clf,feature_names = fn, class_names=cn, filled = True);


# In[158]:


#displaying classification report and confusion matrix for decision tree to check model accuracy

from sklearn.metrics import confusion_matrix
import seaborn as sns 
from sklearn.metrics import classification_report, accuracy_score

y_pred = tree_clf.predict(X_test)

print(classification_report(y_test, y_pred))


#confusion matrix

plt.figure(1, figsize=(7, 6))
plt.title("Confusion Matrix")
sns.set(font_scale=1.4)

cm = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cm, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'}) # Yellow + Green + Blue

labels = list(cn)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

ax.set(ylabel="True Label", xlabel="Predicted Label")

plt.show( )


# In[146]:


#displaying rmse for decision tree

from sklearn.metrics import mean_squared_error


tree_mse = mean_squared_error(y_test, y_pred)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# 2. Random Forest

# In[147]:


#importing RandomForestRegressor from sklearn
from sklearn.ensemble import RandomForestRegressor

#implementing Random Forest model
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)


# In[176]:


#checking the accuracy of the model after testing
y_predictions = forest_reg.predict(X_test).round().astype(int)

tree_acc = accuracy_score(y_test,y_pred)
tree_acc


# In[159]:


#displaying classification report and confusion matrix for random forest to check model accuracy

print(classification_report(y_test, y_predictions))

#confusion matrix
plt.figure(1, figsize=(7,6))
plt.title("Confusion Matrix")
sns.set(font_scale=1.4)

cm = confusion_matrix(y_test, y_predictions)
ax = sns.heatmap(cm, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})

labels = list(cn)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

ax.set(ylabel="True Label", xlabel="Predicted Label")

plt.show( )


# In[164]:


forest_acc = accuracy_score(y_test, y_predictions)
#displaying rmse for random forest
forest_mse = mean_squared_error(y_test, y_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# 3. Perceptron

# In[151]:


#importing Perceptron from sklearn
from sklearn.linear_model import Perceptron

#implementing perceptron model
per_clf = Perceptron()
per_clf.fit(X_train, y_train) 


# In[152]:


y_predict = per_clf.predict(X_test)


# In[160]:


#displaying classification report and confusion matrix for percptron to check model accuracy

from sklearn.metrics import confusion_matrix
import seaborn as sns 
from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict))

#confusion matrix
plt.figure(1, figsize=(7,6))
plt.title("Confusion Matrix")


cm = confusion_matrix(y_test, y_predict)
ax = sns.heatmap(cm, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})

labels = list(cn)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

ax.set(ylabel="True Label", xlabel="Predicted Label")

plt.show( )


# In[166]:


perceptron_acc = accuracy_score(y_test, y_predict)
#displaying rmse for perceptron
perceptron_mse = mean_squared_error(y_test, y_predict)
perceptron_rmse = np.sqrt(perceptron_mse)
perceptron_rmse


# In[162]:


# comparison:
rmses = pd.DataFrame({"Decision Tree":[tree_rmse], 
                      "Random Forest":[forest_rmse],
                      "Perceptron":[perceptron_rmse]})

rmses.plot(kind = "bar")
plt.show( )


# In[178]:


accuracies = pd.DataFrame({"Decision Tree":[tree_acc], 
                      "Random Forest":[forest_acc],
                      "Perceptron":[perceptron_acc]})
fig, ax = plt.subplots()
ax.set_ylim(bottom=0.5)
sns.barplot(data=accuracies)


# Models:
# 
# 
# *   Decision Tree has an accuracy of about 83%.
# *   Random Forest has an accuracy of about 90%.
# *   Perceptron has an accuracy of 82%.
# 
# We can see that the best classification model for this problem is Random Forest.
# 
# 
# 
# 
# 

# We were able to classify food into different categories based on its nutritional facts.
