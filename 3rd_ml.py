import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()
data=iris.data
datasets=iris.target
features_train,features_test,labels_train,labels_test=train_test_split (features, labels, test_size=0.7, random_state =112)
knn=KNeighborsClassifier(n_neighbors=17)
knn.fit(features_train,labels_train)
predictions=knn.predict(features_test)
accuracy=accuracy_score(labels_test,predictions)
print("Accuracy of the K-NN:", accuracy)