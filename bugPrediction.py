#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd # data processing, CSV file I/O
import matplotlib.pyplot as plt # data visualization
import seaborn as sns  # statistical data visualization

#plotly is a library used to plot graphs in python

import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)
import plotly.graph_objs as go

import os

# Display contents of the current directory.
# print(os.listdir("."))


# In[2]:


data = pd.read_csv('jm1.csv')


# **Attribute Information for the Dataset**
# 
# 1. loc : numeric % McCabe's line count of code
# 2. v(g) : numeric % McCabe "cyclomatic complexity"
# 3. ev(g) : numeric % McCabe "essential complexity"
# 4. iv(g) : numeric % McCabe "design complexity"
# 5. n : numeric % Halstead total operators + operands
# 6. v : numeric % Halstead "volume"
# 7. l : numeric % Halstead "program length"
# 8. d : numeric % Halstead "difficulty"
# 9. i : numeric % Halstead "intelligence"
# 10. e : numeric % Halstead "effort"
# 11. b : numeric % Halstead
# 12. t : numeric % Halstead's time estimator
# 13. lOCode : numeric % Halstead's line count
# 14. lOComment : numeric % Halstead's count of lines of comments
# 15. lOBlank : numeric % Halstead's count of blank lines
# 16. lOCodeAndComment : numeric
# 17. uniq_Op : numeric % unique operators
# 18. uniq_Opnd : numeric % unique operands
# 19. total_Op : numeric % total operators
# 20. total_Opnd : numeric % total operands
# 21. branchCount : numeric % of the flow graph
# 22. defects : {false,true} % module has/has not one or more reported defects

# In[3]:


# data.info() #informs about the data (memory usage, data types etc.)


# In[4]:


# data.head() #shows first 5 rows


# In[5]:


# data.tail() #shows last 5 rows


# In[6]:


# data.sample(10) #shows random rows (sample(number_of_rows))


# In[7]:


# data.shape #shows the number of rows and columns


# In[8]:


'''25% refers to the percentile that is min + (max-min)*0.25. Where min is the value of the data'''
# data.describe() #shows simple statistics (min, max, mean, etc.)


# In[9]:


'''Groupby groups the datapoints having same value of an attribute together. It creates a groupby object.'''
defects_true_false = data.groupby('defects')['b'].apply(lambda x: x.count()) #defect rates (true/false)
print('False : ' , defects_true_false[0])
print('True : ' , defects_true_false[1])


# **Histogram Plot**

# In[10]:


trace = go.Histogram(
    x = data.defects,
    opacity = 0.75,
    name = "Defects",
    marker = dict(color = 'green'))

hist_data = [trace]
hist_layout = go.Layout(barmode='overlay',
                   title = 'Defects',
                   xaxis = dict(title = 'True - False'),
                   yaxis = dict(title = 'Frequency'),
)
fig = go.Figure(data = hist_data, layout = hist_layout)
# iplot(fig)


# **Covariance**
# 
# Covariance is a meausure of the similarity of two given featrues. It's lies between -1 and 1. For features with positive covariance, increasing the value of 1 would result in an increase in the value of the other whereas decreasing the value of one of the feature would lead to decreasing of the value of the other feature. However, the values move in opposite directions for features having negative covariance.

# In[11]:


# data.corr() #shows coveriance matrix


# **Heatmap**
# 
# A heatmap is a more intuitive way of representing the covariance matrix wherein colors are used to indicate the co-variance between two features. The light color in the map indicates that the co-variance is high whereas the dark color indicates that the co-variance is low.

# In[12]:


f,ax = plt.subplots(figsize = (15, 15))
sns.heatmap(data.corr(), annot = True, linewidths = .5, fmt = '.2f')
# # plt.show()


# **Scatter Plot**

# In[13]:


trace = go.Scatter(
    x = data.v,
    y = data.b,
    mode = "markers",
    name = "Volume - Bug",
    marker = dict(color = 'darkblue'),
    text = "Bug (b)")

scatter_data = [trace]
scatter_layout = dict(title = 'Volume - Bug',
              xaxis = dict(title = 'Volume', ticklen = 5),
              yaxis = dict(title = 'Bug' , ticklen = 5),
             )
fig = dict(data = scatter_data, layout = scatter_layout)
# iplot(fig)


# **Data Preprocessing**
# 
# Need to find out whether there are certain empty entries present in the dataset, if yes then need to clean the data if no, we're okay to go.

# In[14]:


data.isnull().sum() #shows how many of the null


# **Outlier Detection**
# 
# A box plot is used for outlier detection. It displays 5 statistics of the dataset namely min, 1st quartile, median, 3rd quartile and the max.

# In[15]:


trace1 = go.Box(
    x = data.uniq_Op,
    name = 'Unique Operators',
    marker = dict(color = 'blue')
    )
box_data = [trace1]
# iplot(box_data)


# **Feature Extractotion**
# 
# We'd separate data points which have unusually high values (high complexity) for the attributes n, v, d, e, and t. We'd label these as "Redesign" whereas we'd label the others as "Successful". We're performing an evaluation of the data and would append the extra column 'complexityEvaluation' to our dataFrame object.

# In[16]:


def evaluation_control(data):    
    evaluation = (data.n < 300) & (data.v < 1000 ) & (data.d < 50) & (data.e < 500000) & (data.t < 5000)
    data['complexityEvaluation'] = pd.DataFrame(evaluation)
    data['complexityEvaluation'] = ['Succesful' if evaluation == True else 'Redesign' for evaluation in data.complexityEvaluation]
    
evaluation_control(data)
# data


# In[17]:


# data.info() # Update, added a new column


# In[18]:


data.groupby("complexityEvaluation").size() #complexityEvalution rates (Succesfull/redesign)


# In[19]:


# Histogram
trace = go.Histogram(
    x = data.complexityEvaluation,
    opacity = 0.75,
    name = 'Complexity Evaluation',
    marker = dict(color = 'darkorange')
)
hist_data = [trace]
hist_layout = go.Layout(barmode='overlay',
                   title = 'Complexity Evaluation',
                   xaxis = dict(title = 'Succesful - Redesign'),
                   yaxis = dict(title = 'Frequency')
)
fig = go.Figure(data = hist_data, layout = hist_layout)
# iplot(fig)


# **Data Normalization (min-max normalization)**
# 
# - Base on the details of the data.describe(), we can identify the features which need normalization. Normalization needed in order to speed up the learning algorithms.

# In[20]:


'''Observe new values of b and v in the data.describe()'''

from sklearn import preprocessing

scale_v = data[['v']]
scale_b = data[['b']]

minmax_scaler = preprocessing.MinMaxScaler()

v_scaled = minmax_scaler.fit_transform(scale_v)
b_scaled = minmax_scaler.fit_transform(scale_b)

data['v_ScaledUp'] = pd.DataFrame(v_scaled)
data['b_ScaledUp'] = pd.DataFrame(b_scaled)

# # data


# In[21]:


scaled_data = pd.concat([data.v, data.b, data.v_ScaledUp, data.b_ScaledUp], axis = 1)
# scaled_data


# **Model Selection**
# 
# - This is a classification problem, thus many learning algorithms can be employed to achieve our purpose such as 
#     1. Naive Bayes Classifier
#     2. Logistic Regression
#     3. Support Vector Machines
#     ...

# **Naive Bayes Classifier**

# In[22]:


# data.info()


# In[28]:


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection

'''We are now selecting the data which we need for our model. For now, the data selected
    is all the columns'''
X = data.iloc[:,:-10].values
Y = data.complexityEvaluation.values # Select classification attribute values



# In[32]:


# Parsing selection and verification datasets
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y,test_size = validation_size, random_state = seed)


# In[33]:


# Creation of Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


# In[35]:


# Calculation of the ACC value by the K-fold cross validation of NB model
scoring = 'accuracy'
k_fold = model_selection.KFold(n_splits = 10, random_state = seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv = k_fold, scoring = scoring)
# cv_results


# In[36]:


msg = "Mean :%f - std : (%f)"%(cv_results.mean(),cv_results.std())
# msg


# In[37]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

#Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#Accuracy score
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(y_pred,y_test))


# In[ ]:




