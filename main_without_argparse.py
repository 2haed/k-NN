import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from kNN import KNearestNeighborsClassifier, normalize_data, get_accuracy, coffee_tea_replacer


def main():
    data = pd.read_csv(input("Введите название файла: ").lower() + '.csv', encoding='UTF-8', sep=';')
    k = int(input("Введите значение k: "))
    df = normalize_data(data)
    df_to_show = data.drop(['Отметка времени', 'Чай или кофе'], axis=1)
    X = df.drop(['tea_or_coffee'], axis=1).values
    y = df.tea_or_coffee.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print(df.corr())
    with KNearestNeighborsClassifier(k=k) as self_made_clf:
        self_made_clf.fit(X=X_train, y=y_train)
        scikit_learn_clf = KNeighborsClassifier(n_neighbors=k)
        scikit_learn_clf.fit(X_train, y_train)
        predictions_from_self_made_clf = self_made_clf.predict(X_test).tolist()
        predictions_from_scikit_learn_model = scikit_learn_clf.predict(X_test).tolist()
        self_made_accuracy = get_accuracy(predictions_from_self_made_clf, y_test)
        scikit_learn_accuracy = get_accuracy(predictions_from_scikit_learn_model, y_test)
        while True:
            chooser = int(input("Одиночное предсказание - 1 / Множественное предсказание - 2 / Для выхода - -1: "))
            if chooser == 1:
                print(df_to_show)
                num = int(input("Введите i-ый элемент дата сета: "))
                print(self_made_clf.single_predict(X[num]))
            elif chooser == 2:
                print(
                    f'Точность самописной модели, где k = {k} = {self_made_accuracy * 100}%'
                    f'\n{coffee_tea_replacer(predictions_from_self_made_clf)} '
                    f'Self made model\n{coffee_tea_replacer(y_test)} Test data')
                print(
                    f'Точность компьютерной модели, где k = {k} = {scikit_learn_accuracy * 100}'
                    f'%\n{coffee_tea_replacer(predictions_from_scikit_learn_model)} '
                    f'Scikit learn model\n{coffee_tea_replacer(y_test)} Test data')
            elif chooser == -1:
                print("До свидания!")
                break


if __name__ == '__main__':
    main()
