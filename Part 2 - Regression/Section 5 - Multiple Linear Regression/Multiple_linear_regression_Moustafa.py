


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt



dataset = pd.read_csv('50_Startups.csv')
x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,4].values
y=y.reshape(-1,1)

from sklearn.preprocessing import  LabelEncoder , OneHotEncoder

labelencoder= LabelEncoder()
x[:,3]=labelencoder.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()        # AGAIN, please take care of this you need to transform it to array

"""from sklearn.preprocessing import StandardScaler
sc=  StandardScaler()
x=sc.fit_transform(x)
y= sc.fit_transform(y)"""

x=x[:,1:]





from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# Predicting the Test set results
predection = regressor.predict(x_test)



y_pred = regressor.predict(x_test)


import statsmodels.formula.api as sm     # we need to feed in the the equation b0+b1*x1+.... but b0 is not found so i need to add it my self
x= np.append(arr= np.ones((50,1)).astype(int), values = x , axis=1)

x_opt= x[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()

x_opt= x[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog = y,exog = x_opt).fit()
regressor_ols.summary()

x_opt=x_opt[:,[0,2,3,4]]
regressor_ols= sm.OLS(endog=y,exog = x_opt).fit()
regressor_ols.summary()

x_opt = x_opt[:,[0,1,3]]
regressor_ols= sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()

x_opt = x_opt[:,[0,1]]
regressor_ols= sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()


x=x_opt

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
regressor.fit(x_train,y_train)
predection = regressor.predict(x_test)
y_pred = regressor.predict(x_test)




# Visualising the data
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,)







# Visualising the Training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()  

# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


"""
import numpy as np
from numpy import sin, cos, pi
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from numpy import sin, cos, pi
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
"""
"""
def fun(x, y, z):
    return cos(x) + cos(y) + cos(z)

x, y, z = pi*np.mgrid[-1:1:31j, -1:1:31j, -1:1:31j]
vol = fun(x, y, z)
verts, faces, _, _ = measure.marching_cubes(vol, 0, spacing=(0.1, 0.1, 0.1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                cmap='Spectral', lw=1)
plt.show()
"""
"""
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x[:,0],x[:,1],y, 'gray')
"""









