import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.33, random_state=0)
model_rf = RandomForestClassifier().fit(X_train,y_train)
y_pred_train = model_rf.predict(X_train)
y_pred = model_rf.predict(X_test)
print(accuracy_score(y_train, y_pred_train))
print(accuracy_score(y_test, y_pred))

model_name  = 'model.pkl'
pickle.dump(model_rf, open(model_name, 'wb'))