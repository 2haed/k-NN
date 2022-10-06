import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter





def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(['Отметка времени'], axis=1).apply(lambda x: pd.factorize(x)[0]).rename(
        columns={'Пол': 'sex', 'Высшая школа': 'high_school', 'Округ': 'region', 'Спорт': 'sport',
                 'Цвет глаз': 'eye_color', 'Во сколько встаешь': 'time', 'Курение': 'smoke',
                 'Чай или кофе': 'tea_or_coffee'})
    df = (df - df.min()) / (df.max() - df.min())
    return df


def get_accuracy(predictions: list, y_test) -> str:
    return f' equals: {np.sum(predictions == y_test)/len(y_test)}'


def e_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNearestNeighborsClassifier:
    def __init__(self, k, X, y):
        self.X_train = X
        self.y_train = y
        self.k = k

    def predict(self, X):
        predicted_labels = [self.predicting(x) for x in X]
        return np.array(predicted_labels)

    def predicting(self, x):
        distances = [e_distance(x, x_train) for x_train in self.X_train]
        k_index = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_index]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
