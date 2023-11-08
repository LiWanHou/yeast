# Import thư viện
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn import tree

# Import dữ liệu
yeast_data = pd.read_fwf("yeast.data", names = ["Sequence_Name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "localization_site"])
yeast_feature = yeast_data.iloc[:,1:9]
yeast_target = yeast_data.iloc[:,9:10]

# SMOTE
sm = SMOTE(k_neighbors=4, random_state = 0)
X = yeast_feature
y = yeast_target

X_sm, y_sm = sm.fit_resample(X, y)
X_sm.shape, y_sm.shape

# Chia train, test set
X_train,X_test,y_train,y_test=train_test_split(X_sm,y_sm,test_size=0.2,random_state=32)

#KNN
n_knn = round(math.sqrt(len(X_train)))
Model_KNN = KNeighborsClassifier(n_neighbors = n_knn)
Model_KNN.fit(X_train,y_train)

pickle.dump(Model_KNN, open("model_knn.pkl", "wb"))

#DECISION TREE
Model_DT = tree.DecisionTreeClassifier(criterion='gini',random_state = 0)
Model_DT = Model_DT.fit(X_train,y_train)

pickle.dump(Model_DT, open("model_dt.pkl", "wb"))

#RANDOM FOREST
model_RFC = RandomForestClassifier(n_estimators=500, random_state=0)
model_RFC.fit(X_train, y_train)

pickle.dump(model_RFC, open("model_rfc.pkl", "wb"))