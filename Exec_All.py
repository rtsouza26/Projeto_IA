from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import numpy as np
import pandas as pd
from HCV import knn



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
columns = hcv.columns
repcolumns = list()
for index, column in enumerate(columns):
  column = column.replace(" ","")
  repcolumns.append(column)
y = hcv.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(y)
dfnorm = pd.DataFrame(x_scaled, columns = repcolumns)
dfnorm = np.array(dfnorm)
train_features, test_features, train_labels, test_labels = train_test_split(dfnorm, labels, test_size = 0.25, random_state = 42)
n_arvores = [100,200,400]#testando com as profundidades variando entre 100,200 e 400
print('Random Forest:')
for x in n_arvores:
   rf = RandomForestClassifier(n_estimators=x)
   rf.fit(train_features,train_labels)
   predictions = rf.predict(test_features)
   porcentagem = accuracy(test_labels,predictions)
   print(porcentagem,'para',x,'árvores')

print('KNN:')
n_vizinhos = [1,3,5,10]
for x in n_vizinhos:
    kn = knn(k=x)
    kn.train(train_features,train_labels)
    predictions = kn.predict(test_features)
    porcentagem = accuracy(test_labels, predictions)
    print(porcentagem,'para',x,'vizinhos')

