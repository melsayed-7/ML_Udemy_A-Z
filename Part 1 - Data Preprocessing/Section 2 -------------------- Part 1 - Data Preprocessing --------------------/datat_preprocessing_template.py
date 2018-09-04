# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 04:07:10 2018

@author: Moustafa7
"""

# first make object 
 
 
 from sklearn.preprocessing import Imputer
 
 imputer = Imputer(missing_values = 'NaN',strategy = 'mean', axis =0)
 X[:,1:3] = imputer.fit_transform(X[:,1:3])
 
 
 
 
 from sklearn.preprocessing import LabelEncoder, OneHotEncoder
 labelencoder_X = LabelEncoder()
 X[:,0]= labelencoder_X.fit_transform(X[:,0])
 
 onehotencoder =OneHotEncoder (categorical_features = [0])
 X=onehotencoder.fit_transform(X).toarray()

 
 
 #splitting the training set and the test set - very 
 
 from sklearn.cross_validation import train_test_split
 X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size = 0.2, random_state=0)
 
 
 #Feature scaling - most of the libraries do this part but some do not so just comment it
 """
 from sklearn.preprocessing import StandardScaler
 sc_X =StandardScaler()
 X_train=sc_X.fit_transform(X_train)
 X_test=sc_X.transform(X_test)  #note that i did not put fit because i already fitted its similar the "X_train"
 """
 
