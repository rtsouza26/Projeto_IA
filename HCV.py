from sklearn import datasets
import numpy as np
import math
import csv
import random as rd
import operator


def moda(lista):
    cont1 = 0
    cont2 = 0
    cont3 = 0
    cont4 = 0
    for x in lista:
        if x == 1:
            cont1 += 1
        elif x == 2:
            cont2 += 1
        elif x == 3:
            cont3 += 1
        else:
            cont4 += 1

    frequencias = {'1': cont1, '2': cont2, '3': cont3, '4': cont4}
    dici = sorted(frequencias.items(), key=operator.itemgetter(1), reverse=True)
    max = dici[0][1]
    valores_moda = []
    valores_moda.append(dici[0][0])
    for x in range(1, 4):
        if dici[x][1] == max:
            valores_moda.append(dici[x][0])

    tamanho = len(valores_moda) - 1
    retorno = valores_moda[rd.randint(0, tamanho)]
    return int(retorno)

def distance(x, y):
    distance = 0
    for i in range(0,len(x)):
        distance += pow((x[i] - y[i]), 2)
    return math.sqrt(distance)

def accuracy(y_true, y_pred):
    acertos=0
    for x in range(0,len(y_true)):
        if y_true[x] == y_pred[x]:
            acertos += 1

    porcentagem = acertos/len(y_true)
    return porcentagem


class knn():

    def __init__(self, k=3):
        self.k = k

    def train(self,train_x, train_y):
        self.datatrain = train_x
        self.labeltrain = train_y
        #raise NotImplementedError()

    def predict(self,test_x):
        pred = []
        for y in range(0,len(test_x)):
            dicionario = []
            for x in range(0,len(self.datatrain)):
                aux = distance(test_x[y],self.datatrain[x])
                result = self.labeltrain[x]
                dicionario.append({'distance':aux ,'label':result})
            sorted_list = sorted(dicionario, key=lambda k: k['distance'])
            vizinhos=[]
            for x in range(0,self.k):
                vizinhos.append(sorted_list[x])
            labels=[]
            for x in range(0,len(vizinhos)):
                resposta = vizinhos[x]['label']
                labels.append(resposta)
            pred.append(moda(labels))
        return pred
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
    #raise NotImplementedError()

'''
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

k_vals = [1,3,5,10]
acc_vals = []
for val in k_vals:
  obj = knn(k=val)
  obj.train(data_train, label_train)
  pred = obj.predict(data_test)
  acc_vals.append(accuracy(label_test, pred))
print(acc_vals)


'''







