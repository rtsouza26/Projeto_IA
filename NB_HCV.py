from sklearn.naive_bayes import GaussianNB
import numpy as np
import csv

def converter(array_string):
    retorno = []
    for x in array_string:
        retorno.append(float(x))
    return retorno

def separar_label(matriz):
    vetor_label = []
    for i in matriz:
        num = i[len(i)-1]
        vetor_label.append(num)
        i.pop(len(i)-1)
    return vetor_label
def accuracy(y_true, y_pred):
    acertos=0
    for x in range(0,len(y_true)):
        if y_true[x] == y_pred[x]:
            acertos += 1

    porcentagem = acertos/len(y_true)
    return porcentagem


arquivo = open('HCV-Egy-Data.csv')

linhas = csv.reader(arquivo)
array = []
for linha in linhas:
    array.append(linha)
array.pop(0)
array_convertido = []
for x in array:
    aux = []
    aux = converter(x)
    array_convertido.append(aux)

indices = np.random.permutation(len(array_convertido))
indices_train = indices[:-200]
indices_test = indices[-200:]

labels = separar_label(array_convertido)
labels = np.array(labels)
array_convertido = np.array(array_convertido)


data_train = array_convertido[indices_train]
data_test = array_convertido[indices_test]

label_train = labels[indices_train]
label_test = labels[indices_test]

nb = GaussianNB()
nb.fit(data_train,label_train)
resultado = nb.predict(data_test)
porcentagem = accuracy(label_test,resultado)
print(porcentagem)