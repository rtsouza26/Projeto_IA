import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


hcv = pd.read_csv('HCV-Egy-Data.csv')

labels = np.array(hcv['Baselinehistological staging'])
hcv = hcv.drop('Baselinehistological staging', axis = 1)
hcv_list = list(hcv.columns)
hcv = np.array(hcv)

train_features, test_features, train_labels, test_labels = train_test_split(hcv, labels, test_size = 0.25, random_state = 42)
rf = RandomForestRegressor(n_estimators=1000,random_state=42)

rf.fit(train_features,train_labels)
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
print('Erro médio absoluto:', round(np.mean(errors), 2), 'degrees.')
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Precisão:', round(accuracy, 2), '%.')

