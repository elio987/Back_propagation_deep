#Leonardo Gracida Munoz A01379812
import pickle
import matplotlib.pyplot as plt
import numpy as np


#Abrimos la dataset de test
with open('test.pkl', 'rb') as file:
    myvar = pickle.load(file)
#Abrimos los pesos geenrados
with open('model.pkl', 'rb') as file:
    myvar2 = pickle.load(file)

X_test,y_test_cod = myvar
w0,w1,w2 = myvar2

#Funcion de activacion de la neuronas
def sigmoid(x):
    return(1/(1 + np.exp(-x)))

#Funcion de perdida
def loss(out, Y):
    s =(np.square(out-Y))
    #print(s)
    s = np.sum(s)/4
    #print(np.sum(s),len(Y))
    return s

#Funcion para predecir con la red entranada
def feed_fwd(x,w0,w1,w2):
    h1 = np.dot(w0,x)
    a1 = sigmoid(h1)
    h2 = np.dot(w1,a1)
    a2 = sigmoid(h2)
    h3 = np.dot(w2,a2)
    a3 = sigmoid(h3)
    return a3

#Obtenemos la accuray del modelo con el dataset de test
l =[]
for i in range(len(X_test)):
  out = feed_fwd(X_test[i], w0, w1, w2)
  l.append(loss(out, y_test_cod[i]))
print("acc:", (1-(sum(l)/len(X_test)))*100," ======== loss:",sum(l)/len(X_test)) 

#Ploteamos la unas cuantas imagenes y mostramos la predccion junto al label real_
fig = plt.figure(figsize=(10,10))
lugar = 0
numeros = []
for i in range(1, 11):
    plt.subplot(2, 5, i)
    y = np.argmax(y_test_cod[lugar])
    while y in numeros:
        lugar += 1
        y = np.argmax(y_test_cod[lugar])
    #Predecimos
    #Como salida de una neurona corresponde a cada una de las categorias de la imagen obetnemos el index de la neurona con la salida mas grande
    out = np.argmax(feed_fwd(X_test[lugar],w0,w1,w2).T)
    y = np.argmax(y_test_cod[lugar])
    print("Prediccion: ",out,", Real: ",y)
    plt.imshow(np.reshape(X_test[lugar],(8,8)), cmap='gray')
    plt.title("real: "+str(y)+", pred: "+str(out))
    numeros.append(y)
    lugar += 1
#Guardamos el modelo
plt.savefig('predicciones.png')
plt.show()