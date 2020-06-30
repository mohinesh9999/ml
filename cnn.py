import pandas as pd
import numpy as np
from sklearn import preprocessing,svm,neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style

df=pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)#df.dropna()
# print(df.head())


df.drop(['id'],1,inplace=True)
x=np.array(df.drop(['class'],1))
y=np.array(df['class'])

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
clf=neighbors.KNeighborsClassifier(n_jobs=1)
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
example_measures=np.array([[4,2,1,1,1,2,3,2,1]])
# example_measures=example_measures.reshape(1,-1)
prediction=clf.predict(example_measures)
print(accuracy,prediction,example_measures)