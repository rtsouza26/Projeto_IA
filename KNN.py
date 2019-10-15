from sklearn import datasets
import numpy as np
import math
import random as rd
import operator
import statistics

def moda(lista):
    cont0 = 0
    cont1 = 0
    cont2 = 0
    for x in lista:
        if x == 0:
            cont0 += 1
        elif x == 1:
            cont1 += 1
        else:
            cont2 += 1

    frequencias = {'0': cont0, '1': cont1, '2': cont2}
    dici = sorted(frequencias.items(), key=operator.itemgetter(1), reverse=True)
    max = dici[0][1]
    valores_moda = []
    valores_moda.append(dici[0][0])
    for x in range(1, 3):
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
            acertos +=1

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


        #raise NotImplementedError()






iris = datasets.load_iris()
indices = np.random.permutation(len(iris.data))

indices_train = indices[:-50]
indices_test = indices[-50:]

data_train = iris.data[indices_train]
data_test = iris.data[indices_test]

label_train = iris.target[indices_train]
label_test = iris.target[indices_test]

model = knn(k=5)
model.train(data_train, label_train)

y_pred = model.predict(data_test)
print(accuracy(label_test, y_pred))
