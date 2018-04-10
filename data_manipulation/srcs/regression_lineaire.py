# regression lineaire

# importer les librairies

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# importer le trainset

trainset = pd.read_csv('../data/train/trainset.csv')
X_train = trainset.iloc[:, [5,6,8,9,10,11]].values
y_train = trainset.iloc[:, 4].values 
X_trainFrame = pd.DataFrame(X_train)

# importer le testset

dataset1 = pd.read_csv('../data/valid/test_2017-07-12.csv')
dataset2 = pd.read_csv('../data/valid/test_2017-07-13.csv')

# concatener les 2 fichiers

testset = dataset1.append(dataset2)

testsetFrame = pd.DataFrame(testset)
X_test = testset.iloc[:, [5,6,8,9,10,11]].values
y_test = testset.iloc[:, 4].values
y_test = y_test.reshape(-1,1)
X_testFrame = pd.DataFrame(X_test)

# gerer les donnes manquantes 

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy="mean")
imputer.fit(y_test)
y_test = imputer.transform(y_test)

# gerer la variable categorique 'workday_2'

from sklearn.preprocessing import LabelEncoder

labelencoder_X_train = LabelEncoder()
X_train[:, 4] = labelencoder_X_train.fit_transform(X_train[:, 4])

labelencoder_X_test = LabelEncoder()
X_test[:, 4] = labelencoder_X_test.fit_transform(X_test[:, 4])

# appliquer le feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# construction du modele
 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# prediction

y_pred = regressor.predict(X_test)

# Visualiser les resultats

plt.scatter(X_test[:, 0], y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()