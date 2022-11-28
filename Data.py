from Dataset import *

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


X = dataset.drop(['class','veil-type'],axis=1)
y = dataset['class'] == 'p' # class: Poisonous  = 1 , Eatable = 0 

X_dummy = pd.get_dummies(X,columns=X.columns) #drop_first=True)

# print(X_dummy.head())

X_train, X_test, y_train, y_test = train_test_split(X_dummy, y, test_size=0.3)


mm = make_pipeline(StandardScaler(), MinMaxScaler())

X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)
