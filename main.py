import numpy as np
import pandas as pd
import mlbench as mlb
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

data = pd.read_csv('letter-recognition.data', sep=',')

X = data.iloc[:, 1:]
y = data['lettr']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

error_rates = []
k_values = range(1, 26)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error_rate = 1 - accuracy_score(y_test, y_pred)
    print('При k равном', k, ', процент ошибок равен', error_rate, '%')

best_k = 1  # Выберите наилучшее значение k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Построение модели kNN с лучшим значением
# Оценка качества модели на тестовой выборке с
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

error_rate = 1 - accuracy_score(y_test, y_pred)
print(f"Процент ошибок на тестовой выборке: {error_rate * 100:.2f}%")
