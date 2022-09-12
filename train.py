#leonardo Gracida Munoz A01379812
from random import random
from sklearn.datasets import load_digits
import pickle
import matplotlib.pyplot as plt
import numpy as np
from time import time
#Cargamos la dataset de digitos numericos de sklearn
digits = load_digits()
#Mostramos la forma de la dataset
print("Forma de la data original: ",digits.data.shape)
#Normalizamos los valores de las imagenes para que vayan de cero a uno
digitos = digits.images/np.max(digits.images)
#Aplanamos todas la imagenes de la dataset para poder ingresarlas a la red en este caso de 8,8 a 64,1
X = []
for i in range(len(digitos)):
    X.append(np.reshape(digitos[i],(64,1)))
X = np.array(X)
#Mostramos la nueva forma de la dataseet
print("Forma de la data transformada o aplanada: ",X.shape)
#Hacemos una funcion de generacion de numeros aleatorios usando el reloj de la computadora
def time_random():
    return time() - float(str(time()).split('.')[0])
#Generamos un numero aleatorio dentro de un rango
def gen_random_range(min, max):
    return int(time_random() * (max - min) + min)
#Fncin para poder obtener todos las labels de la dataset en un diccionario
def unique(list1):
    unique_list = {}
    for x in list1:
        if x not in unique_list:
            unique_list[x] = 0
    return unique_list
#Funcion que separa la dataset en un apartado de train y split, escogiendo las muestras de una menra aleatoria
def train_test_split(X,Y,test_size):
    #Obtenemos la proporcion del tamano del test
    test_size = int(X.shape[0]*test_size)
    #Tamano de la parte de train
    train_size = int(X.shape[0] - test_size)
    #listas donde vamos a guardar los numeros random generados para no hagarrar la misma muestra
    #Ademas de crear la listan donde vamos a guardar el test
    randon_numbers = []
    X_test = []
    Y_test = []
    #obtenemos los labels de la dataset
    unique_dic = unique(Y)
    size = int(test_size/len(unique_dic))
    paso = 0
    #Mientras no tengamos todas las muestras
    while paso < test_size:
        #Obtenemos un numero random dentro del tamano de la dataset
        random = gen_random_range(0,X.shape[0]-1)
        #mientras el numero random se repita y ya hayamos hagarrado el suficiente numero de muestras vamos a generar otro numero aleatorio
        while (random in randon_numbers) and (unique_dic[Y[random]] >= size):
            random = gen_random_range(0,X.shape[0]-1)
        #Al encontrar un index lo agregamos a la lista de la parte de test
        if random not in randon_numbers:
            X_test.append(X[random])
            Y_test.append(Y[random])
            #Agregamos el numero random a la lista de numeros aleatorios ya generados
            randon_numbers.append(random)
            #Sumamos al conteo de ese label
            unique_dic[Y[random]] = unique_dic[Y[random]] + 1
            paso += 1
    #Aqui iteramos en todo los index de la dataset completa ignorando a los generados aleatoriamente
    X_train = []
    Y_train = []
    for i in range(X.shape[0]):
        if i not in randon_numbers:
            X_train.append(X[i])
            Y_train.append(Y[i])
    #Lo pasamos todo a ser un array
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    #Devolvemos la dataset separada aleatoriamente
    return (X_train,X_test,Y_train,Y_test)

X_train, X_test, y_train, y_test = train_test_split(X, digits.target, test_size=0.20)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10)

print("Muestras train: ", X_train.shape[0])
print("Muestras test: ", X_test.shape[0])
print("Muestras validation: ", X_val.shape[0])

#Funcion de activacion de las neuronas usadas, en este caso es una sigmoide
def sigmoid(x):
    return(1/(1 + np.exp(-x)))

#Funcion que va a generar todos los pesos de manera aleatoria, para cada una de las capas
def generate_wt(x, y):
    l =[]
    for i in range(x * y):
        l.append(np.random.randn())
    return(np.array(l).reshape(x, y))

#Funcion que va a ingresar la entrada y los pesos para hacer una prediccion
def feed_fwd(x,w0,w1,w2):
    h1 = np.dot(w0,x)
    a1 = sigmoid(h1)
    h2 = np.dot(w1,a1)
    a2 = sigmoid(h2)
    h3 = np.dot(w2,a2)
    a3 = sigmoid(h3)
    return a3

#Funcion que aplica back propagation
def back_prop(x,y,w0,w1,w2,alpha):
    #Input first layer
    h1 = np.dot(w0,x)
    #Output first layer
    a1 = sigmoid(h1)
    #Input second layer
    h2 = np.dot(w1,a1)
    #Output second layer
    a2 = sigmoid(h2)
    #Input third layer
    h3 = np.dot(w2,a2)
    #Output layer (pred)
    a3 = sigmoid(h3)
    #Obtenemos el error de la salida de la red
    op = np.multiply(np.multiply(a3,(1-a3)),(y-a3))
    #En el caso de el error de la capas ocultas vamos a sumar el peso que sale de cada neurona por el eror de la neurona a la que entra
    #Btenemos el error de la segunda capa oculta
    oh2 = np.multiply(np.multiply(a2,(1-a2)),np.dot(w2.T,op))
    #Obtenemos el error de la primera capa oculta
    oh3 = np.multiply(np.multiply(a1,(1-a1)),np.dot(w1.T,oh2))
    #Obtenemos los deltas de todos los pesos

    w2_adj = np.dot(op,a2.T)
    w1_adj = np.dot(oh2,a1.T)
    w0_adj = np.dot(oh3,x.T)
    #Actualziamos todos los pesos
    w0 = w0 + alpha*w0_adj
    w1 = w1 + alpha*w1_adj
    w2 = w2 + alpha*w2_adj
    return (w0,w1,w2)
#Esta funcion lo que hace es iterar un numero de veces determinado o epochs en toda la dataset de train y aplicar back propagation para entrar la red
def train(x, Y, X_val, y_val_cod, w0, w1, w2, alpha = 0.01, epoch = 10):
    acc =[]
    losss =[]
    acc_val =[]
    losss_val =[]
    for j in range(epoch):
        l =[]
        for i in range(len(x)):
            out = feed_fwd(x[i], w0, w1, w2)
            l.append((loss(out, Y[i])))
            w0, w1, w2 = back_prop(x[i], Y[i], w0, w1, w2, alpha)
        l_val =[]
        for i in range(len(X_val)):
            out = feed_fwd(X_val[i], w0, w1, w2)
            l_val.append(loss(out, y_val_cod[i]))
        print("epochs:", j + 1,"===== acc_val:", (1-(sum(l_val)/len(X_val)))*100," ===== loss_val:",sum(l_val)/len(X_val)) 
        print("epochs:", j + 1, "===== acc:", (1-(sum(l)/len(x)))*100," ===== loss:",sum(l)/len(x))  
        acc.append((1-(sum(l)/len(x))))
        losss.append(sum(l)/len(x))
        acc_val.append((1-(sum(l_val)/len(X_val))))
        losss_val.append(sum(l_val)/len(X_val))
    return(acc, losss, acc_val, losss_val, w0, w1, w2)
#Funcion que obtiene la perdida de la red
def loss(out, Y):
    s =(np.square(out-Y))
    #print(s)
    s = np.sum(s)/4
    #print(np.sum(s),len(Y))
    return s
#Generamos todos los pesos de todas las capas
w0 = generate_wt(32,64)
print("Forma de capa w0: ",w0.shape)
w1 = generate_wt(16,32)
print("Forma de capa w1: ",w1.shape)
w2 = generate_wt(10,16)
print("Forma de capa w2: ",w2.shape)

#Funcion que traduce el label de cada muestra a cada una de las salidas de las neuronas
def target_cod(y):
    salida = []
    for i in y:
        if i == 0:
            salida.append(np.array([[1,0,0,0,0,0,0,0,0,0]]).T)
        elif i == 1:
            salida.append(np.array([[0,1,0,0,0,0,0,0,0,0]]).T)
        elif i == 2:
            salida.append(np.array([[0,0,1,0,0,0,0,0,0,0]]).T)
        elif i == 3:
            salida.append(np.array([[0,0,0,1,0,0,0,0,0,0]]).T)
        elif i == 4:
            salida.append(np.array([[0,0,0,0,1,0,0,0,0,0]]).T)
        elif i == 5:
            salida.append(np.array([[0,0,0,0,0,1,0,0,0,0]]).T)
        elif i == 6:
            salida.append(np.array([[0,0,0,0,0,0,1,0,0,0]]).T)
        elif i == 7:
            salida.append(np.array([[0,0,0,0,0,0,0,1,0,0]]).T)
        elif i == 8:
            salida.append(np.array([[0,0,0,0,0,0,0,0,1,0]]).T)
        elif i == 9:
            salida.append(np.array([[0,0,0,0,0,0,0,0,0,1]]).T)
    return np.array(salida)\
#Traducimos los labels
y_test_cod = target_cod(y_test)
y_train_cod = target_cod(y_train)
y_val_cod = target_cod(y_val)

#Declaramos los hiperparametros
alpha = 0.1
epchos = 30

#Entrenamos el modelo
acc,loss,acc_val,loss_val,w0,w1,w2 = train(X_train,y_train_cod,X_val,y_val_cod,w0,w1,w2,alpha,epoch=epchos)

#Guardamos los pesos con pickle
ws = (w0,w1,w2)

with open('model.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(ws, file)

print("Modelo guardado")

#Guardamos el test igual con pickle
test = (X_test,y_test_cod)

with open('test.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(test, file)

print("Test guardado")

fig = plt.figure()
#Guardamos el comporatmiento de la accuracy y la perdida a lo largo de las epochs
plt.plot(np.arange(1,epchos+1,1), acc)
plt.plot(np.arange(1,epchos+1,1), loss)
plt.plot(np.arange(1,epchos+1,1), acc_val)
plt.plot(np.arange(1,epchos+1,1), loss_val)
plt.title("Acc and loss vs epochs")
plt.legend(["Accuracy","Loss","Accuracy_val","Loss_val"])
plt.savefig('acc_loss.png')
plt.show()
