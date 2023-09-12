'''
Autor: 
    Raul Alejandro Olivares A01752057
Descripción del código:
    El siguiente codigo presenta la implementación de la tecnica de arbol de decisión utilizando la libreria de sklearn, 
    prueba distintas configuraciones para posteriomente seleccionar la mejor e indicarsela al usuario.
    
'''

#Importación de las librerias necesarias para la construccion del modelo y su evaluación
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import learning_curve

#Carga del dataset de ejemplo desde sklearn
wine = load_wine()
X = wine.data
y = wine.target

#Division del conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Definicion de los hiperparámetros para búsqueda aleatoria
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 11),
    'min_samples_split': range(2, 11),
    'min_samples_leaf': range(1, 11),
    'random_state':range(1,1000)
}

#Super parametros 
num_models = 5 #Numero de modelos que nos gustaría visualizar, minimo 1
max_models=100 #Maximo de iteraciones para la busqueda 
count=0        #Inicialización de variable contadora, para saber cuantos modelos que cumplan cierto criterio dado por el usuario han sido hallados
thresh_hold=0.90 #Accuracy minimo con el que buscamos nuestros arboles de decisión

#Listas y variables para almacenar resultados de los modelos
model_accuracies = []
model_params = []

best_accuracy = 0  #Inicializa la mejor precisión
best_model = None  #Inicializa el mejor modelo
best_params = None  #Inicializa los mejores hiperparámetros

train_curve = [] #Inicializa la lista de los resultados de la curva de aprendizaje por cada modelo en el conjunto de training
score_curve=[]   #Inicializa la lista de los resultados de la curva de aprendizaje por cada modelo en el conjunto de test
size_curve=[]    #Inicializa la lista de los resultados de la curva de aprendizaje por cada modelo en el conjunto de tamaño de las muestras

#Entrena y evalua modelos con configuraciones aleatorias, hasta un maximo de iteraciones (max models)
for _ in range(max_models):
    #Decisión aleatoria de hiper parametros
    params = {
        'criterion': random.choice(param_grid['criterion']), #Dos tipos diferentes de criterios, gini o entropia, para calcular la impureza de los nodos
        'max_depth': random.choice(param_grid['max_depth']), #Maxima Altura que podría tener el arbol
        'min_samples_split': random.choice(param_grid['min_samples_split']), #Numero minimo de ramas hijo que podría tener un nodo padre
        'min_samples_leaf': random.choice(param_grid['min_samples_leaf']), #Numero de muestras minimo que debe de tener una hoja
        'random_state':random.choice(param_grid['random_state']) #Semilla aleatoria con la que se generara el arbol
    }

    #Se crea y entrena el modelo de árbol de decisión con los parametros aleatorios
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)

    #Realiza las predicciones y calcula la precisión en el conjunto de prueba
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    #Si la precisión fue mejor que un modelo predecesor se actualizan las variables de mejor precisión, el numero del modelo y los hiper parametros utilizados
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = _
        best_params = params

    #Condición de salida del ciclo for, si ya se han encontrado el numero de modelos requeridos que cumplan con el umbral indicado se sale del ciclo
    if count == num_models:
        break
    
    #Si el modelo realizado en la iteración cumple con tener una precisión mas alta o igual que la indicada, guarda sus valores e imprime sus metricas
    if accuracy >= thresh_hold:
        #Se almacenan la precisión y los hiperparámetros utilizados
        model_accuracies.append(accuracy)
        model_params.append(params)

        #METRICAS UTILIZADAS

        # 1. Matriz de Confusión; se genera una matriz de confusión según nuestras variables esperadas y nuestras variables predichas
        #Util para visualizar cuando y donde el modelo tuvo aciertos y fallas
        print(f"Modelo {_} con hiperparámetros: {params}")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Matriz de Confusión:")
        print(conf_matrix)

        # 2. Reporte de Clasificación; genera un reporte con varias metricas relevantes como:
        '''
        Precision: Mide la proporción de predicciones positivas que fueron realmente correctas. Cuanto mayor sea, mejor.

        Recall (Sensibilidad): Mide la proporción de instancias positivas que se predijeron correctamente. Cuanto mayor sea, mejor.

        F1-Score: Es una métrica que combina la precisión y la recuperación en un solo número. Es útil cuando deseas encontrar un equilibrio entre ambas métricas.

        Support: El número de ocurrencias de cada clase en el conjunto de prueba.

        Accuracy (Exactitud): La proporción de predicciones totales que fueron correctas. Es una métrica útil, especialmente cuando las clases están equilibradas.

        Macro Avg: Es el promedio de las métricas en todas las clases, sin tener en cuenta su desequilibrio.

        Weighted Avg: Es similar al promedio macro, pero tiene en cuenta el desequilibrio de clases. Las clases con más muestras tienen un peso mayor en el cálculo.
        '''
        class_report = classification_report(y_test, y_pred)
        print("\nReporte de Clasificación:")
        print(class_report)

        #3. Learning curve; la ultima metrica nos permite ver en un gráfico como fue la curva de aprendizaje de nuestro modelo;
        #Segun la accuracy de las predicciones con los datos de training y los datos de testing
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
        )
        size_curve.append(train_sizes)
        train_curve.append(train_scores)
        score_curve.append(val_scores)
        count +=1

#Si no se logro encontrar ningun modelo que cumpla las condiciones dadas por el usuario, se le notifica
if(len(model_accuracies) == 0):
    print(f"No se logro encontrar ninún modelo que cumpla con el accuracy de: {thresh_hold}")

#Si se logro encontrar un modelo o varios se elabora una gráfica de sus accuracys, y de sus curvas de aprendizaje
else:
    #Gráfico de barras de accuracy de cada modelo
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(model_accuracies)), model_accuracies, tick_label=["Modelo "+str(i+1) for i in range(len(model_accuracies))])
    plt.xlabel("Modelo")
    plt.ylabel("Accuracy")
    plt.title("Accuracy de modelos con hiperparámetros aleatorios")
    plt.ylim(0, 1.0)

    #Etiquetas de los valores de precisión en las barras
    for i, acc in enumerate(model_accuracies):
        plt.text(i, acc, f'{acc:.2f}', ha='center', va='bottom')

    plt.show()

    #Gráficos de curva de aprendizaje de todos los modelos
    plt.figure(figsize=(12, 6))
    for i in range(len(model_accuracies)):
        plt.subplot(2, 3, i+1)
        plt.plot(size_curve[i], np.mean(train_curve[i], axis=1), 'o-', label='Training score')
        plt.plot(size_curve[i], np.mean(score_curve[i], axis=1), 'o-', label='Validation score')
        plt.title(f'Modelo {i+1}')
        plt.xlabel('Número de ejemplos de entrenamiento')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

#Por ultimo;
#Imprime el mejor modelo y sus hiperparámetros
print("Mejor modelo:")
print(best_model)
print("Mejores hiperparámetros:")
print(best_params)
print("Precisión del mejor modelo:", best_accuracy)

