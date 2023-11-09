import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import metrics

# Ruta a la carpeta principal que contiene las subcarpetas por lenguaje
main_directory = 'data'

data = []
labels = []

# Recorre cada subcarpeta (cada una representa un lenguaje)
for language_folder in os.listdir(main_directory):
    language_path = os.path.join(main_directory, language_folder)
    
    if os.path.isdir(language_path):
        # Recorre los archivos en la subcarpeta del lenguaje
        for filename in os.listdir(language_path):
            file_path = os.path.join(language_path, filename)
            
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    try:
                        code = file.read()
                    except:
                        continue
                    data.append(code)
                    labels.append(language_folder)  # Etiqueta del lenguaje

# Crea un DataFrame con los datos recopilados
dataset = pd.DataFrame({'code': data, 'language': labels})

print("Dataset size:", dataset.shape)
print("Languages:", dataset['language'].unique())

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# Crear un pipeline que incluya un vectorizador de texto y un clasificador Random Forest
model = make_pipeline(CountVectorizer(), RandomForestClassifier(random_state=42))

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
predictions = model.predict(X_test)

# Evaluar el modelo
print(f"Metrics:")
print("Accuracy:", metrics.accuracy_score(y_test, predictions))
print("Precision:", metrics.precision_score(y_test, predictions, average="macro"))
print("Recall:", metrics.recall_score(y_test, predictions, average="macro"))
print("F1 Score:", metrics.f1_score(y_test, predictions, average="macro"))

# Obtener etiquetas únicas de las clases (lenguajes de programación)
unique_labels = list(set(y_test))

# Crear la matriz de confusión
conf_matrix = confusion_matrix(
    y_test,
    predictions,
    labels=unique_labels
)

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 9))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicción')
plt.ylabel('Etiqueta verdadera')
plt.tight_layout()
plt.savefig("out/confusion_matrix.png")
plt.clf()


# Datos para Random Forest y Naive Bayes
rf_data = [0.9315589353612167, 0.946385594333084, 0.9307852965747702, 0.932410244923947]
nb_data = [0.8555133079847909, 0.8972924181885839, 0.8484440267335005, 0.8329007525904902]

# Etiquetas para cada métrica
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Posiciones de las barras para Random Forest y Naive Bayes
rf_positions = np.arange(len(labels))
nb_positions = rf_positions + 0.4  # Espacio de 0.4 entre grupos

# Crear el gráfico de barras
plt.bar(rf_positions, rf_data, width=0.4, color='royalblue', alpha=0.8, label='Random Forest')
plt.bar(nb_positions, nb_data, width=0.4, color='tomato', alpha=0.8, label='Naive Bayes')

# Añadir etiquetas y leyenda
#plt.xlabel('Métricas')
#plt.ylabel('Valor')
#plt.title('Comparación de Métricas entre Random Forest y Naive Bayes')
plt.xticks(rf_positions + 0.2, labels, fontsize=24)
plt.yticks(fontsize=24)
plt.legend(loc='lower right', fontsize=16)
plt.savefig("out/nb_vs_rf.png")