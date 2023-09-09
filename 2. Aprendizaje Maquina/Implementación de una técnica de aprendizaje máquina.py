'''
Autor:
    Raul Alejandro Olivares A01752057

Titulo:
    Implementacion de una tecnica de aprendizaje maquina

Descripción del codigo:
    El siguiente codigo presenta la implementación manual de la técnica de aprendizaje "Decission Tree Classifier".
    La tecnica se implementa realizando el supuesto de "preguntas", en donde el arbol, hace una pregunta por cada feature del set de datos, 
    por cada pregunta se obtiene su ganancia de información, a partir de este valor se determina que pregunta realizara el mejor split, y se 
    continua hasta que todas las distintas clases de target se hayan clasificado.

Instrucciones Generales:
    La ejecución del codigo tarda generalmente un tiempo de 10 minutos, al ejecutar el código despues de 10 minutos se despliega un window dialog de
    matplot lib, es necesario cerrarlo para seguir ejecutando el codigo.
    Despues de esto no es necesario hacer nada extra :)
'''

#Import de librerias necesarias para realizar el arbol de decision
import pandas as pd
from sklearn.datasets import load_wine
import numpy as np

#Tree classificator model, de la linea 28 a la 188 se construye el arbol

#Funcion que cuenta los distintos valores de nuestra variable objetivo, esto es util para el calculo de gini 
def class_counts(df):
    counts = df["target"].value_counts().to_dict()
    return counts

'''
Definición de la clase de pregunta:
    La siguiente clase en la encargada de generar la impresión de la pregunta que se realizo para el split de los datos, 
    Esto es util si el usuario desea imprimir el arbol y analizar cuales fueron las condiciones que generaron las distintas
    particiones de los datos.
'''
class Question:

    #Inicializa las variables de instancia (nombre de la columna y valor)
    def __init__(self, column, value):
        self.column = column
        self.value = value

    #Función que regresa los valores que sean mayor o igual que cierto umbral, el umbral depende del tipo de feature se esta analizando
    def match(self, example):
        val = example[self.column]
        return val >= self.value
    
    #Función que regresa la condición que el arbol elaboro, el formato es "Is (nombre de la columna) mayor o igual que (umbral)?""
    def __repr__(self):
        condition = ">="
        return "Is %s %s %s?" % (self.column, condition, str(self.value))
#Finalización de la clase de pregunta

#La siguiente función realiza una partición sobre nuestro dataset según una pregunta, si el valor satisface las condiciones del umbral
#agrega la fila de nuestro dataset a un sub dataset (true), y si no lo agrega a otro dataset (false), generando así la partición
def partition(df, question):
    true_df = []
    false_df = []
    for ind, row in df.iterrows():
        test = question.match(df.loc[ind])
        if test:
            true_df.append(df.loc[ind])
        else:
            false_df.append(df.loc[ind])
    return true_df,false_df

#Función que regresa la impuresa de un nodo según su gini, esto se calcula sobre un sub set de nuestro dataframe original
#La función de gini cuenta las apariciones de una clase de nuestra variable target sobre la población total del sub dataframe
#La impureza se incializa en 1, puesto que si tenemos una relación del gini de 1/1, el valor de gini se restara a la impureza
#dandonos un valor final de impureza de 0, lo que significa que se ha clasificado una clase de nuestra variable target
def gini(df):
    counts = class_counts(df)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(df.index))
        impurity -= prob_of_lbl**2
    return impurity

#Esta función calcula la ganancia de información
#El calculo de información se realiza a partir de nuestra incertidumbre actual generada por una pregunta del arbol
#Restada por nuestra incertidumbre generada por una pregunta nueva
#Con ello podemos determinar si una pregunta genero una mayor o menor ganancia de información, lo que ayuda al arbol a definir cual sera el
#mejor split de los datos
#Primero se calcula la proporción de la partición (left y right)
#Posteriormente se calcula la incertidumbre generada por la nueva pregunta apoyandose en el calculo de gini
def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

#Esta función es el corazón del arbol, ya que es la encargada de encontar el mejor split de los datos, recibe un dataframe
def find_best_split(df):
    #La mejor ganancia de información y mejor pregunta se inicializan en 0
    best_gain = 0  
    best_question = None  
    current_uncertainty = gini(df)

    #Posteriormente el arbol realiza una pregunta por cada feature del arbol y por cada valor de ese feature, se calcula su
    #ganancia de información y se actualizan las variables de mejor ganancia y mejor pregunta si una pregunta genero una mejor
    #partición que sus antecesores, finalmente regresa la mejor ganancia y la mejor pregunta elaborada
    labels = df.columns.tolist()
    for col in range(0, len(labels)-1): 

        for ind in range(0,len(df.index)):  

            question = Question(labels[col], df.iloc[ind,col])

            true_rows, false_rows = partition(df, question)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            true_rows = pd.DataFrame(true_rows, columns = df.columns.tolist())
            false_rows = pd.DataFrame(false_rows, columns = df.columns.tolist())
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

#Definición de la clase de hoja, esta clase regresa el numero de clasificaciones de una clase especifica de target 
class Leaf:
    def __init__(self, df):
        self.predictions = class_counts(df)
 
#Clase de nodo de decisión, esta clase inicializa la pregunta con la cual fue dividida el arbol junto con los dos sub datasets salientes de la partición
class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

#La siguiente función construye el arbol de forma recursiva con todas las herramientas dadas anteriormente.
def build_tree(df):
    gain, question = find_best_split(df)

    #Si se encuentra una hoja la ganancia de información ya no podra ser mayor, por lo que se regresa el nodo de hoja
    if gain == 0:
        return Leaf(df)

    #Se busca la pregunta que mejor dividira los datos y se divide el dataset en dos
    true_rows, false_rows = partition(df, question)
    true_rows = pd.DataFrame(true_rows, columns = df.columns.tolist())
    false_rows = pd.DataFrame(false_rows, columns = df.columns.tolist())
    
    #De forma recursiva se itera sobre el output true y false, para construir el arbol en las ramas distintas
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    #Finalmente se regresa el nodo de decisión que particiono los datos, junto con los sub sets generados
    return Decision_Node(question, true_branch, false_branch)

#Función valiosa para el usuario, ya que permite visualizar la estructura del arbol en la terminal
#La estructura es la siguiente: Pregunta -> Rama True -> Valor predicho y numero de predicciones -> Rama False -> Valor predicho y numero de predicciones
def print_tree(node, spacing=""):

    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    print (spacing + str(node.question))

    print (spacing + "--> True:")
    print_tree(node.true_branch, spacing + "  ")

    print (spacing + "--> False:")
    print_tree(node.false_branch, spacing + "  ")

#La siguiente función permite probar el arbol de decisión con nuevos datos entrantes según las condiciones generadas por el arbol en los datos de training
#Se busca clasificar cada valor de los datos test de forma recursiva, colocandolos en un patron que el arbol ya genero.
def classify(df, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

#Esta función es util para el usuario, calcula la probabilidad de la clasificación que realizo el arbol, y se la despliega en la 
#terminal al usuario. 
def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

#Inicia la prueba de la construcción del arbol
#Se carga el dataset contenido en la biblioteca de sklearn
wine = load_wine()
features = pd.DataFrame(wine.data, columns=wine.feature_names)
target = pd.Series(wine.target, name='target')
df = pd.concat([features, target], axis=1)

#Funciones utiles para calcular el score del modelo y mostrar que generaliza
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import learning_curve

#Se calcula el score de 5 modelos con distinto random_state en la división de los datos, esto nos permitira
#observar en un grafico si nuestro modelo esta generalizando o no
accuracy=[]
for i in range(0,5):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state= random.randint(1, 1000))
    #Se construye el arbol con nuestros datos de training
    my_tree = build_tree(df_train)
    predictions = []
    #Se obtienen las predicciones de nuestro arbol con los datos de test
    for ind, row in df_test.iterrows():
        predictions.append(list(classify(row, my_tree).keys())[0])  
    
    predictions_df = pd.DataFrame({"Predicted": predictions})

    #Se extraen los datos reales de nuestra variable objetivo con la finalidad de evaluar nuestra predicción
    target_exp = df_test["target"]
    ratios = []
    for ind, row in df_test.iterrows():
        ratios.append(target_exp[ind])

    target_exp = pd.DataFrame(ratios, columns = ["Actual"])

    #Se calcula la precisión de nuestro modelo segun los datos esperados y los datos predichos, y se concatena a una lista que almacena nuestros valores de precisión
    accuracy.append(accuracy_score(target_exp, predictions_df))

#Se calcula la media de nuestro accuracy a lo largo de nuestros 5 modelos diferentes
mean_accuracies = []
mean_accuracies.append(np.mean(accuracy))

#Se genera un grafico el cual permita visualizar el valor de precisión de nuestros 5 modelos diferentes, para mostrar que generaliza
x_indices = list(range(1, len(accuracy) + 1)) 

plt.figure(figsize=(10, 6))  # Tamaño del gráfico
plt.plot(x_indices, accuracy, marker='o', linestyle='-', color='b', label='Accuracy')  # Crear la línea
plt.xlabel('Número de Valor')  # Etiqueta del eje x
plt.ylabel('Accuracy')  # Etiqueta del eje y
plt.title('Gráfico de Accuracy')  # Título del gráfico

# Etiquetar cada punto con su valor de accuracy
for i, acc in enumerate(accuracy):
    plt.annotate(f'{acc:.2f}', (x_indices[i], acc), textcoords="offset points", xytext=(0, 10), ha='center')

plt.grid(True)  # Agregar una cuadrícula al gráfico
plt.legend()  # Mostrar la leyenda
plt.show()  # Mostrar el gráfico

#Se imprime el valor de la media de nuestros valores de precisión
print("Media del accuracy: ",mean_accuracies)

#Metricas utilizadas para la evaluación de nuestro modelo, esto es relevante para visualizar y evaluar el rendimiento de nuestro modelo
#Estas metricas se generan según el ultimo modelo aleatoreo generado

# 1. Matriz de Confusión; se genera una matriz de confusión según nuestras variables esperadas y nuestras variables predichas
#Util para visualizar cuando y donde el modelo tuvo aciertos y fallas
conf_matrix = confusion_matrix(target_exp, predictions_df)
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
class_report = classification_report(target_exp, predictions_df)
print("\nReporte de Clasificación:")
print(class_report)

#Función necesaria para indicarle al learning curve que modelo utilizamos.
from sklearn.tree import DecisionTreeClassifier
#3. Learning curve; la ultima metrica nos permite ver en un gráfico como fue la curva de aprendizaje de nuestro modelo;
#Segun la accuracy de las predicciones con los datos de training y los datos de testing
train_sizes, train_scores, val_scores = learning_curve(
DecisionTreeClassifier(),
predictions_df, target_exp,
cv=5,
scoring='accuracy',
train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, val_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, val_mean + val_std, val_mean - val_std, alpha=0.15, color='green')
#Se imprime el ultimo gráfico
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
