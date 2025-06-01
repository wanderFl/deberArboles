# Importación de bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# 1. Cargando el dataset
dataset = pd.read_csv('Mall_Customers.csv')

# 2. Selección de características (X) y variable objetivo (y)
X = dataset[['Age', 'Annual Income (k$)']].values
le = LabelEncoder()
y = le.fit_transform(dataset['Genre'])  # Codifica "Male"/"Female" a 1/0

# 3. División en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# 4. Escalado de características (Feature Scaling)
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# 5. Entrenamiento del modelo Random Forest
classifier = RandomForestClassifier(
    n_estimators=10,      # Número de árboles
    criterion='entropy',   # Criterio de división
    random_state=0
)
classifier.fit(X_train_scaled, y_train)  # Entrena el modelo

# 6. Predicción de un nuevo ejemplo (Edad=30, Ingreso Anual=75 k$)
nuevo_pred = classifier.predict(sc.transform([[30, 75]]))
nuevo_label = le.inverse_transform(nuevo_pred)  # Vuelve a "Male"/"Female"

# 7. Predicción sobre el conjunto de prueba
y_pred = classifier.predict(X_test_scaled)

# 8. Evaluación del modelo
cm = confusion_matrix(y_test, y_pred)    # Matriz de confusión
accuracy = accuracy_score(y_test, y_pred)  # Precisión (accuracy)

# 9. Resultados por consola
print("Primeras filas del dataset:")
print(dataset.head())

print("\nPredicción para Edad=30 y Ingreso Anual=75 k$:")
print(nuevo_label[0])

print("\nMatriz de Confusión:")
print(cm)

print("\nPrecisión del modelo:")
print(accuracy)

# 10. Visualización de fronteras de decisión

# a) Conjunto de entrenamiento
X_set, y_set = sc.inverse_transform(X_train_scaled), y_train
X1, X2 = np.meshgrid(
    np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 0.25),
    np.arange(start = X_set[:, 1].min() - 5, stop = X_set[:, 1].max() + 5, step = 0.25)
)

plt.figure(figsize=(8, 6))
plt.contourf(
    X1, X2,
    classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0], 
        X_set[y_set == j, 1],
        color='red' if j == 0 else 'green',
        label=le.inverse_transform([j])[0]
    )
plt.title('Random Forest (Conjunto de Entrenamiento)')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.legend()
plt.show()

# b) Conjunto de prueba
X_set, y_set = sc.inverse_transform(X_test_scaled), y_test
X1, X2 = np.meshgrid(
    np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 0.25),
    np.arange(start = X_set[:, 1].min() - 5, stop = X_set[:, 1].max() + 5, step = 0.25)
)

plt.figure(figsize=(8, 6))
plt.contourf(
    X1, X2,
    classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0], 
        X_set[y_set == j, 1],
        color='red' if j == 0 else 'green',
        label=le.inverse_transform([j])[0]
    )
plt.title('Random Forest (Conjunto de Prueba)')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.legend()
plt.show()

# 11. Gráfica de importancias de características

# Extraer la importancia de cada variable
importancias = classifier.feature_importances_
nombres_caracteristicas = ['Age', 'Annual Income (k$)']

plt.figure(figsize=(6, 4))
plt.bar(range(len(importancias)), importancias)
plt.xticks(range(len(importancias)), nombres_caracteristicas)
plt.title('Importancia de Características en Random Forest')
plt.xlabel('Características')
plt.ylabel('Importancia')
plt.tight_layout()
plt.show()