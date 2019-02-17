# Multiple Linear Regression

# Why any equation is called as linear?
# This is because dependent variable is result of linear relation between independent variables.
# Example: y = b0 + b1x1.. This is simole univariable linear expression.
# Example 2: y = b0 + b1x1 + b2 x2..Here x1 and x2 are linear related to each other due to + summation.
#  Example 3: This is not linear relation  y = b1x1 / b2x2.

# Another examplem of linear regresion is Polynomial linear regression.
#  example:  y = b0 +b1x1 + b2 square (x2) + b3 cube (x3).
 
# DEfinitin Of the Problem:
# Based on the companies data of R and D spend, administrative cost, Marketing spend and the state where it operates, it would determine the profit amount
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('CompaniesData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
# Below 20% data  reserved for testing and 80% for the training. This is not done in random manner but with the sequence.
# idealy ur train and test data must have data from all diverse categories to train and test the model

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Fitting Multiple Linear Regression to the Training set

# Here I am not putting effort on Preprocessing of the data. I also avoided feature scaling


# Applying simple linear regression

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df
from sklearn import metrics 

print('LINEAR REGRESSION') 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  



plot.scatter(y_test, y_pred, color = 'red')
plot.plot(y_test, y_test, color = 'blue')
plot.title('LINEAR REGRESSION :Comparison of Y Test and Y Pred')
plot.xlabel('Y Test')
plot.ylabel('Y pred in RED')
plot.show()  


# from the decision tree regression model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
# y_pred_decision = regressor.predict(X_test)
# Predicting the Test set results
y_pred_decision = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_decision})  
df
from sklearn import metrics  
print('DECISION TREE REGRESSION') 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_decision))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_decision))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_decision)))  



plot.scatter(y_test, y_pred_decision, color = 'red')
plot.plot(y_test, y_test, color = 'blue')
plot.title('DECISION TREE:Comparison of Y Test and Y Pred')
plot.xlabel('Y Test')
plot.ylabel('Y pred in RED')
plot.show()  


# Applying the Ridge and Lasso regression models

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
ridge2 = Ridge(alpha = 0, normalize = True)
ridge2.fit(X_train, y_train)             # Fit a ridge regression on the training data
y_pred = ridge2.predict(X_test)           # Use this model to predict the test data
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df
from sklearn import metrics  

print('RIDGE REGRESSION') 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))




plot.scatter(y_test, y_pred, color = 'red')
plot.plot(y_test, y_test, color = 'blue')
plot.title('RIDGE REGRESSION :Comparison of Y Test and Y Pred')
plot.xlabel('Y Test')
plot.ylabel('Y pred in RED')
plot.show()  



# now the lasso regression:
lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X_train, y_train)
y_pred = lassocv.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df

print('LASSO REGRESSION') 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



plot.scatter(y_test, y_pred, color = 'red')
plot.plot(y_test, y_test, color = 'blue')
plot.title('LASSO: Comparison of Y Test and Y Pred')
plot.xlabel('Y Test')
plot.ylabel('Y pred in RED')
plot.show()  