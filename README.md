# RegressionMultiVariable

This repository explains about MultiVariable regression. 
The explanation has been added into code.

# Multiple Linear Regression

# Why any equation is called as linear?
# This is because dependent variable is result of linear relation between independent variables.
# Example: y = b0 + b1x1.. This is simole univariable linear expression.
# Example 2: y = b0 + b1x1 + b2 x2..Here x1 and x2 are linear related to Y due to + summation.
#  Example 3: This is not linear relation  y = b1x1 / b2x2.

# Another examplem of linear regresion is Polynomial linear regression.
#  example:  y = b0 +b1x1 + b2 square (x2) + b3 cube (x3).

# DEfinitin Of the Problem:
# Based on the companies data of R and D spend, administrative cost, Marketing spend and the state where it operates, it would determine the profit amount.

# Here along with simple linear regression model, other models are also evaluated with the use of  r square and adjusted r square values.
# Detail explanation of other models are added in different repositories.

# I added Simple negression, Decisoin tree regression, Ridge and Lasso regressions.

# see below the result of the differnt models.

LINEAR REGRESSION
Mean Absolute Error: 7514.293659640891
Mean Squared Error: 83502864.03257468
Root Mean Squared Error: 9137.990152794797


DECISION TREE REGRESSION
Mean Absolute Error: 5277.153000000001
Mean Squared Error: 49904185.29361
Root Mean Squared Error: 7064.289440107193


RIDGE REGRESSION
Mean Absolute Error: 7514.293659640597
Mean Squared Error: 83502864.03257713
Root Mean Squared Error: 9137.990152794931

LASSO REGRESSION
Mean Absolute Error: 6742.955069952084
Mean Squared Error: 69903733.50976606
Root Mean Squared Error: 8360.845262876599


From the observation it is found that Decision Tree is clear cut winner. But for Ridgo and Lasso, not all possible pemalties are tried.



