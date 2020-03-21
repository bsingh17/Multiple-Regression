import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('list price vs best price.csv')
x=dataset.iloc[:,0:1].values
y=dataset.iloc[:,1].values


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=8)
x_poly=poly.fit_transform(x)
poly.fit(x_poly,y)
reg2=LinearRegression()
reg2.fit(x_poly,y)

plt.title('Polynomial Regression')
plt.xlabel('List Price')
plt.ylabel('Best Price')
plt.scatter(x,y)
plt.plot(x,reg2.predict(x_poly),color='green')