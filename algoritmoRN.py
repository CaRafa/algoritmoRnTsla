#imports para leer y guardar los datos https://www.youtube.com/watch?v=UkzhouEk6uY
import numpy as np
import os
#imports para la arquitectura https://www.youtube.com/watch?v=Boo6SmgmHuM
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras import optimizers
from keras.models import load_model

#import para las calculos
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.realpath(__file__))

#FIJAR SEED ALEATORIO PARA REPRODUCIBILIDAD
np.random.seed(50)


#Arquitectura
def model():
	model = Sequential()
	model.add(Dense(27, input_dim=27, activation='softsign' )) #primera capa de entrada, 6 nodos, cada nodo con una entrada
	model.add(Dense(54,  activation='relu'))#segunda capa reducida a 3 nodos
	model.add(Dense(18,  activation='relu'))
	model.add(Dense(1, activation='softsign'))#capa de salida con 1 nodo.
	ad = optimizers.Adam(lr=0.0001)
	# calculando el error por error cuadratico medio, y solicitando que retorne la precision con la que acerta
	#Adam es un optimizador de aprendizaje, lr es learning rate es la proporcion con la que se cambiaran los aleatorios de los pesos
	#adam https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
	model.compile(optimizer=ad ,loss='mean_absolute_error', metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])
	return model



#RED NEURONAL

# Closure Last Week,
# Open This Week,
# Highest Price Last Week,
# Lowest Price Last Week,
# Volume Last Week,
# Variation Percent Last Week,
# Variation Percent This Week (Y)

#CARGA DEL CSV QUE CONTIENE LOS REGISTROS
dataset = np.loadtxt("%s/TSLA.csv" % path, delimiter=",")

#INPUTS Y OUTPUT PARA ENTRENAR
X = dataset[0:50, 0:27]
Y = dataset[0:50, 27]

#INPUNTS Y OUTPUT PARA VALIDAR
A = dataset[0:56, 0:27]
B = dataset[0:56, 27]


#MUESTRA DE GRAFICAS
def graficas(history):
    
    plt.plot(history.history['mean_absolute_error'])
    plt.title('Error absoluto medio (MAE)')
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.show()

    plt.plot(history.history['mean_absolute_percentage_error'])
    plt.title('Error absoluto porcentual medio (MAPE)')
    plt.ylabel('MAPE')
    plt.xlabel('epoch')
    plt.show()

#CALCULO DE PREDICCIONES
def prediccion(model, entrada):
    predictions = model.predict(entrada)
    resultado = [x[0] for x in predictions]
    print (resultado)

#VALIDACION DE LOS REGISTROS
def validacion(model, entrada, salida):
    scores = model.evaluate(entrada, salida, verbose=2)
    print("\n%s: %.2f" % (model.metrics_names[2], scores[2]))


ent = str(input("\n¿ Crear nueva red (new) o usar red guardada (load) ?\n>>> "))
if ent == 'new':
    model = model()
    model.summary()
elif ent == 'load':
    try:
        model = load_model('%s/modelo.h5' % path)
        model.summary()
        print ("\nModelo Importado")
    except:
        print ("\nArchivo inexistente")
        

while (ent != "exit"):
    ent = str(input("\n¿ Entrenar red (train), validar red (valid), guardar red (save) o salir del programa (exit) ?\n>>> "))
    if ent == 'train':
        #ENTRENAMIENTO DE LA RED
        entrenamiento = model.fit(X, Y, epochs=5000, batch_size=4, shuffle=False) #epochs es el numero de repeticiones que realizara el codigo para entrenarse
        print("\nResultados:")
        validacion(model, X, Y)
        print("\n")
        weights = model.layers[0].get_weights()[0]
        bias = model.layers[0].get_weights()[1]
        weights1 = model.layers[1].get_weights()[0]
        bias1 = model.layers[1].get_weights()[1]
        weights2 = model.layers[2].get_weights()[0]
        bias2 = model.layers[2].get_weights()[1]
        print(weights)
        print("\n")
        print(bias)
        print("\n")
        print(weights1)
        print("\n")
        print(bias1)
        print("\n")
        print(weights2)
        print("\n")
        print(bias2)
        graficas(entrenamiento)
    elif ent == "valid":
        #VALIDACION DE LA RED
        print("\nValidacion:")
        validacion(model, A, B)
        val = model.predict(A)
        resultados = [(x[0]*100) for x in val]
        print("\n")
        weights = model.layers[0].get_weights()[0]
        bias = model.layers[0].get_weights()[1]
        weights1 = model.layers[1].get_weights()[0]
        bias1 = model.layers[1].get_weights()[1]
        weights2 = model.layers[2].get_weights()[0]
        bias2 = model.layers[2].get_weights()[1]
              
        print(weights)
        print("\n")
        print(bias)
        print("\n")
        print(weights1)
        print("\n")
        print(bias1)
        print("\n")
        print(weights2)
        print("\n")
        print(bias2)
        print(resultados)
    elif ent == "save":
        #GUARDADO DE LA RED
        model.save('%s/modelo.h5' % path)
        print("\nModelo guardado")
    elif ent == "exit":
        #SALIDA DEL PROGRAMA
        ent = "exit"

