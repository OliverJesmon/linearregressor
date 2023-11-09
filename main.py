import matplotlib.pyplot as pl
import numpy as np 
from sklearn import linear_model
x_train=np.array([[1],[2],[3],[4]])
y_train=np.array([[26],[25],[25],[24]])
x_test=np.array([[5]])

model=linear_model.LinearRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
print("Coefficient(mx):",model.coef_)
print("intercept(b)",model.intercept_)

pl.plot(x_test,y_predict,'o')
pl.show()