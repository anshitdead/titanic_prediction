# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importing the libraries
import pandas as pd
import numpy as np
dataset=pd.read_csv('train.csv')
dataset = dataset.fillna(dataset.median())
dataset.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)
dataset = pd.get_dummies(dataset, columns=['Sex', 'Embarked'], drop_first=True)


dataset2=pd.read_csv('test.csv')
dataset2 = dataset2.fillna(dataset2.median())
dataset2.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)
dataset2 = pd.get_dummies(dataset2, columns=['Sex', 'Embarked'], drop_first=True)


X2 = dataset2.iloc[:, [1,2,3,4,5,6,7,8]].values
X2 = pd.DataFrame(X2)
#Taking care of missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy='mean', axis = 0)
#imputer = imputer.fit(dataset.values[:, 0:889])
#dataset.values[:, 0:889] = imputer.transform(dataset.values[:, 0:889])

#dividing the dependent and independent variables
X = dataset.iloc[:, [2,3,4,5,6,7,8,9]].values
X = pd.DataFrame(X)
y = dataset.iloc[:, 1].values
y = pd.DataFrame(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X2 = sc.transform(X2)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', input_dim = 8))
#classifier.add(Dropout(0.3))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))


# Adding the third hidden layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))


# Adding the fourth hidden layer
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu'))


# Adding the fifth hidden layer
#classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred4 = classifier.predict(X2)
y_pred4 = (y_pred > 0.5).astype(int)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Testset Accuracy:', ((cm[0][0]+cm[1][1])/(cm[0][1]+cm[1][0]+cm[1][1]+cm[0][0]))*100)

from livelossplot import PlotLossesKeras
 
classifier.fit(X_train, y_train,
          epochs=2000,
          validation_data=(X_test, y_test),
          callbacks=[PlotLossesKeras()],
          verbose=0)

