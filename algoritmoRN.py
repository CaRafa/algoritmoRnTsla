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
#import para las calculos y graficar
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.realpath(__file__))

#FIJAR SEED ALEATORIO PARA REPRODUCIBILIDAD
np.random.seed(50)


#Arquitectura
#definicion del modelo utilizando parte de la libreria keras
def model():
	model = Sequential()
	model.add(Dense(27, input_dim=27, activation='softsign' )) #primera capa de entrada, 27 nodos, con funcion de salida softsign
	model.add(Dense(54,  activation='relu'))#segunda capa incrementada a 54 nodos, con funcion relu
	model.add(Dense(18,  activation='relu'))#tercera capa reducida a 18 nodos, con funcion relu
	model.add(Dense(1, activation='softsign'))#capa de salida con 1 nodo, con funcion de salida softsign 
	ad = optimizers.Adam(lr=0.0001)
	# calculando el error por error cuadratico medio, y solicitando que retorne la precision con la que acerta
	#Adam es un optimizador de aprendizaje, lr es learning rate es la proporcion con la que se cambiaran los aleatorios de los pesos
	#adam https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
	model.compile(optimizer=ad ,loss='mean_absolute_error', metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])
	return model



#RED NEURONAL

# Se utilizo para cada registro la data de 4 semanas lo cual resulta en 27 variables de entrada por registro
# Se tomo para las primeras 3 semanas todas las variables abajo mencionadas y para la cuarta semana se tomo
# Solamente las primeras 6 por que el objetivo de la red es precisamente obtener la variacion porcentual de esa semana

# Orden de las variables en el documento .csv

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
X = dataset[0:45, 0:27]
Y = dataset[0:45, 27]

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
    print("Prediccion final")
    print("\n")
    print (resultado)

#VALIDACION DE LOS REGISTROS
def validacion(model, entrada, salida):
    scores = model.evaluate(entrada, salida, verbose=2)
    print("\n%s: %.2f" % (model.metrics_names[2], scores[2]))


ent = str(input("\n¿ Para crear una nueva red escriba (new) para cargarla (load) ?\n>>> "))
if ent == 'new':
    model = model()
    model.summary()
elif ent == 'load':
    try:
        model = load_model('%s/modelo.h5' % path)
        model.summary()
        print ("\nModelo Cargado")
    except:
        print ("\nArchivo inexistente")
        

while (ent != "exit"):
    ent = str(input("\n¿ Entrenar (train), validar (valid), guardar (save) o salir (exit) ?\n>>> "))
    if ent == 'train':
        #ENTRENAMIENTO DE LA RED
        entrenamiento = model.fit(X, Y, epochs=3000, batch_size=30, shuffle=False) 
        #epochs es el numero de repeticiones que realizara el codigo para entrenarse
        #batch_size es el numero de registros que pasan antes de actualizar los pesos
        print("\nResultados:")
        validacion(model, X, Y)
        print("\n")
        #obtencion de los pesos y bias resultantes en el modelo durante el entrenamiento para cada una de las capas
        weights = model.layers[0].get_weights()[0]
        bias = model.layers[0].get_weights()[1]
        weights1 = model.layers[1].get_weights()[0]
        bias1 = model.layers[1].get_weights()[1]
        weights2 = model.layers[2].get_weights()[0]
        bias2 = model.layers[2].get_weights()[1]
        weights3 = model.layers[3].get_weights()[0]
        bias3 = model.layers[3].get_weights()[1]
        # Impresion de todos los resultados ordenados por capas en forma descendente
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
        print("\n")
        print(weights3)
        print("\n")
        print(bias3)
        print("\n")
        graficas(entrenamiento)
    elif ent == "valid":
        #VALIDACION DE LA RED
        print("\nValidacion:")
        validacion(model, A, B)
        val = model.predict(A)
        resultados = [(x[0]*100) for x in val] # se multiplica el resultado por 100 para obtener el porcentaje de aumento o decremento
        print("\n")
        #obtencion de los pesos y bias resultantes en el modelo durante la validacion para cada una de las capas
        weights = model.layers[0].get_weights()[0]
        bias = model.layers[0].get_weights()[1]
        weights1 = model.layers[1].get_weights()[0]
        bias1 = model.layers[1].get_weights()[1]
        weights2 = model.layers[2].get_weights()[0]
        bias2 = model.layers[2].get_weights()[1]
        weights3 = model.layers[3].get_weights()[0]
        bias3 = model.layers[3].get_weights()[1]
        # Impresion de todos los resultados ordenados por capas en forma descendente
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
        print("\n")
        print(weights3)
        print("\n")
        print(bias3)
        print("\n")
        print("Prediccion de la red al validar")
        print("\n")
        print(resultados)
        print("%")
    elif ent == "save":
        #GUARDADO DE LA RED
        model.save('%s/modelo.h5' % path)
        print("\nModelo guardado")
    elif ent == "exit":
        #SALIDA DEL PROGRAMA
        ent = "exit"

