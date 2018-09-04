# Data Preprocessing Template

# Importing the libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,3].values

 from sklearn.preprocessing import Imputer
 imputer = Imputer(missing_values="NaN", strategy ="mean", axis =0)
 x=imputer.fit_transform(x[:,1:3])
 
 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_x= LabelEncoder()
x[:,0]=label_x.fit_transform(x[:,0])
hot_x=OneHotEncoder(categorical_features=[0])
x=hot_x.fit_transform(x).toarray() # it is okay to write x= not x[:,0]

label_y= LabelEncoder()
y = label_y.fit_transform(y)
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)
 
# Feature Scaling
# this step must be done after encoding so that it is possible to standardize string like in this case
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x_train= sc_x.fit_transform(x_train)
x_test= sc_x.fit_transform(x_test)
sc_y= StandardScaler() 
y_train=sc_y.fit_transform(y_train.reshape(-1,1))  # we are asking numpy to figure the matrix or the vector if there is only one column -1 in reshape(-1,1) means that the number of row is unknown

