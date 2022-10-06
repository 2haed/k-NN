import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import math

data = pd.read_csv('Normal_data.csv', encoding='UTF-8', sep=';')


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(['Отметка времени'], axis=1).apply(lambda x: pd.factorize(x)[0]).rename(
        columns={'Отметка времени': 'time', 'Высшая школа': 'high_school', 'Округ': 'region', 'Спорт': 'sport',
                 'Цвет глаз': 'eye_color', 'Во сколько встаешь': 'time', 'Курение': 'smoke',
                 'Чай или кофе': 'tea_or_coffee'})
    df = (df - df.min()) / (df.max() - df.min())
    return df


def e_distance(train, test):
    dist = 0
    for i in range(len(train) - 1):
        dist += (train[i] - test[i]) ** 2
    return math.sqrt(dist)


def get_neighbors(train, test, k=2):
    dists = [(train[i][-1], e_distance(train[i], test)) for i in range(len(train))]
    dists.sort(key=lambda x: x[1])
    neighbors = [dists[i][0] for i in range(k)]
    return neighbors


def prediction(neighbors):
    count = {}
    for i in neighbors:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    target = max(count.items(), key=lambda x: x[1])[0]
    return target


df = normalize_data(data)
X = df.drop(['tea_or_coffee'], axis=1)
y = df.tea_or_coffee
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
correlation_matrix = df.corr()
# print(correlation_matrix['tea_or_coffee'])
print(X_test)
print(X_train)
predictions = []
for i in range (len(X_test)):
    neighbors = get_neighbors(X_train, X_test, k=5)
    result = prediction(neighbors)
    predictions.append(result)
