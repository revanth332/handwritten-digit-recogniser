import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split

data= pd.read_csv(r"C:\Users\USER\Desktop\Python\ML\data_train.csv")

x = data.iloc[:,1:].values
y = data.iloc[:,:1]["label"]

x=x/255

x_train,x_test , y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=20,stratify=y)

bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
bagging.fit(x_train,y_train)

filename = "digit_model.sav"
joblib.dump(bagging,filename)