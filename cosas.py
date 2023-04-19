# pandas 
# seaborn 
# matplotlib 
# numpy
# sklearn -> machine learning -> para tratar con variables de varios tipos -> preprocesamiento de datos -> orientado a datos tipo object pasando los datos tipo numerico
# sklearn.preprocessing.LabelEncoder -> para convertir los datos tipo object a numerico -> pasar los valores object con una funcion con dominio natural [0, n-1]
# sklearn.preprocessing.OneHotEncoder -> genera n-1 variables dummy o binaria, es decir, toma valores 0,1 aqui se ecita el que pueda inferirse un orden en la categorias, pues aqui si que la variable 1 == "es norte" y 0 == "no es norte"...
# para tratar los vientos, por ejemplo, se puede usar una variable dummy para cada direccion, y luego se puede usar una variable que sea la suma de las variables dummy, y asi se puede saber si el viento es norte o no, y asi con cada direccion

# TRABAJAR CON COPIAS CREADAS PARA NO MODIFICAR LA ORIGINAL NI LA ORIGINAL MODIFICADA -> df.copy()

# NE
# ENE
# E 
# ESE
# SE
# S 
# SSW
# SW
# W
# WNW
# NW
# N
# NNW
# NW
# W
# WSW
# SW
# S
# SSE
# SE ... 

# Trabajar con csv y excel para manejo de los datos, limpiar na, no nan, nan creo que lo detecta bien.
# mirar manejo de funciones lambda

# https://www.kaggle.com/

# ver que son encoders, algoritmos de deep learning que realizan el trafico de una red neuronal profunda, aunque ya esta implementado en la libreria sklearn

# mirarse los hash maps, hash... -> sklearn para mayor sencillez, pero miratelo bobo

# mirar la funcion .fit de la libreria sklearn.preprocessing.LabelEncoder

# mirar la funcion .transform de la libreria sklearn.preprocessing.LabelEncoder

# mirar la funcion .classe__ 

# con transform obtememos cositas para ya trabajar con numpy

# HASH MAPS -> https://www.youtube.com/watch?v=shs0KM3wKv8

# HASH TABLES -> https://www.youtube.com/watch?v=shs0KM3wKv8

# HASH FUNCTIONS -> https://www.youtube.com/watch?v=shs0KM3wKv8

# HASH COLLISIONS -> https://www.youtube.com/watch?v=shs0KM3wKv8

# Reducir variables para mayor optimizacion

# Filtro por varianza -> eliminar variables que son casi consonantes -> sklearn.feature_selection.VarianceThreshold

# Filtro univariantes basados en una clasificacion de p-valores -> rest estadistico -> chi cuadrado, anova, etc...

# seleccion basada en arbol de decision -> entrenar un arbol de decision muy sobreajustado sobre todo el dataset, despues quedarse con las variables que expliquen un valor determinado de la informacion: 90%, 95%

# Este metodo utiliza un modeo y como veremos mas adelante, todos los modelos de sklearn tienen los siguientes metodo:
# .fit(X = conjunto de variables independientes,  y = variable objetivo del conjunto de train) -> para entrenar el modelo
# .predict(X = conjutno de variables independientes) -> para predecir con el modelo
# .score(y_real, y_predicción) -> para regresion en R2 del modelo, y para clasiicacion en accuracy entendida como el porcentaje de aciertos sobre el total

# Selección basada en métodos recursivos

# Este caso funciona de modo similar a como lo havcen las regresiones backward, es decir, se comienza probando todas las variables para ir sacando todas las variables de una a una

# Importaancia de variables

# Definimos el conjunto de las variables de entrada (variables independientes) y la variable objeetico, y almacenamos esta última en una variable llamada target
# Importamos desde al libreria sklearn la clase para el arbol de regresion. Y procedemos a entrenar uno con todo el dataset y asi obterner las variables mas importantes

# target = df['target']
# features = [x for x in df.columns if x != target]

# Importar el algoritmo de arboles de decisiones de sklearn.tree 

# Asignar el algoritmo e indicar la profundidad maxima del arbol con un numero rotandamente**? grande para sobreajustar

# Con esto entrena un arbol y nos lo ramifica

# de sklearn.metrics sacar mean absolute error mean absolute...

# cuando los coeficientes de determinacion se acercan a 1 ya son buenos para trabajar con ellos

# ver funcion .feature.importances_ de sklearn.tree.DecisionTreeRegressor, esto devuelve la importancia de cada uno de las variables

# añadir la importancia de la columna de la importancia y la importancia acumulada, >=85%

# Definimos la lista de variables no tan importantes. cortando por el porcentaje de 85% de la informacion acumulada

# Filtramos el dataset original para quedarnos solo con las variables importantes

# Planteamiento del ejercicio de clasificación

# Cremos una variabl objetivo de nombre Escenario con 2 clases como 0,1 del modo que:
# El nuvel 0 se corresponda a los valores por debajo del percentil 33 de la variable target
# El nivel 1 se corresponde a los balores por encima del percentil 33

# Eliminamos despues de la columna, del nuevo dataset y procedemos a dividir el dataset en conjuntos de train y test (usualmente un reparto de 80% - 20%)

# Vamos a intentar predecir si la calidad de aire de las distintas zonas está en nivel 1 o 0

# Hacer copias para trabajar sobre el original

# Hacer cuartiles

# count, mean, std (standard deviation), min, 25%, 50%, 75%, max

# violinplot

# ISSUES

# Paso 1: Obtencion y preparacion de datos
    # Preparar el conjunto de datos del modleo 
    # Nomarlizar -> sklearn.preprocessing.StandardScaler
    # .transform
    # Convertir en un datafreme añadiento las etiquetas

# Paso 2: Dividir el dataset en Training y Test
    # Separar los conjuntos de datos de entrenamiento, training, y de prueba, test, para las variables de entrada y salida
    # train_test_split de sklearn.model_selection
    # "test_size" representa la proporcion de conjunto de datos de prueba

# Paso 3: Cargar y elegir el modelo de regresión logística

# Paso 4: Entrenar el modelo de regresion logistica con los datos de entrenamiento
    # Entrenar el modelo y analizamos los resultados para obtener las metricas

# Paso 5: Obtener las predicciones
    #Calcular las predicciones con el conjunto de prueba
    # Imprimir la salida del modelos

# Paso 6: Evaluación del modelo a través de sus métricas
    # accuracy_score de sklearn.metrics
    # esto nos dara los porcentajes
    # Con esto sacamos una matriz de confusion, confusion_matrix de sklearn.metrics
    # Graficar la curva ROC, importando desde sklearn

# 