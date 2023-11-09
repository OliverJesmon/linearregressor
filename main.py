import matplotlib.pyplot as pl #ploting library
import numpy as np #to work with array
from sklearn import linear_model #to import regressor
x_train=np.array([[1],[2],[3],[4]])#training data
y_train=np.array([[26],[25],[25],[24]])
x_test=np.array([[5]])#feature by user

model=linear_model.LinearRegression()
model.fit(x_train,y_train) #getting ready with ml model
y_predict=model.predict(x_test)#label or o/p by machine 
print("Coefficient(mx):",model.coef_)
print("intercept(b)",model.intercept_)#y=mx+b

pl.plot(x_test,y_predict,'o')
pl.show()
#by Oliver Jesmon
