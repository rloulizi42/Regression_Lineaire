# regression lineaire

# importer les librairies

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importer le trainset

trainset = pd.read_csv('../data/train/trainset.csv')
X = trainset.iloc[:, [5,6,8,9,10,11]].values
y = trainset.iloc[:, 4].values 

# importer le testset

dataset1 = pd.read_csv('../data/test/testset_2017-07-12.csv')
dataset2 = pd.read_csv('../data/test/testset_2017-07-13.csv')
X1 = dataset1.iloc[:, [4,5,7,8,9,10]]
X2 = dataset2.iloc[:, [4,5,7,8,9,10]]

# concatener les 2 fichiers

Xtest = X1.append(X2).values

# gerer la variable categorique 'workday_2'

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])

labelencoder_Xtest = LabelEncoder()
Xtest[:, 4] = labelencoder_Xtest.fit_transform(Xtest[:, 4])

# appliquer le feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
y = sc.fit_transform(y)

# construction du modele
 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

# prediction

y_pred = regressor.predict(Xtest)

# Visualiser les resultats

plt.scatter(X,y)