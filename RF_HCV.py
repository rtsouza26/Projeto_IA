import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def accuracy(y_true, y_pred):
    acertos=0
    for x in range(0,len(y_true)):
        if y_true[x] == y_pred[x]:
            acertos +=1

    porcentagem = acertos/len(y_true)
    return porcentagem


hcv = pd.read_csv('HCV-Egy-Data.csv')

labels = np.array(hcv['Baselinehistological staging'])
hcv = hcv.drop('Baselinehistological staging', axis = 1)
hcv_list = list(hcv.columns)
hcv = np.array(hcv)

train_features, test_features, train_labels, test_labels = train_test_split(hcv, labels, test_size = 0.25, random_state = 42)
rf = RandomForestClassifier(n_estimators=200)

rf.fit(train_features,train_labels)
predictions = rf.predict(test_features)
porcentagem = accuracy(test_labels,predictions)
print(porcentagem)


