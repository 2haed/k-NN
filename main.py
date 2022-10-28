import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from kNN import KNearestNeighborsClassifier, normalize_data, get_accuracy, coffee_tea_replacer
import argparse


def main():

    parser = argparse.ArgumentParser(description='Videos to images')
    parser.add_argument('--filename', type=str, help='Input filename')
    parser.add_argument('--k', type=int, help='k number of neighbors')
    parser.add_argument('--i', nargs='?', type=int, help='i element from data to predict')
    args = parser.parse_args()

    data = pd.read_csv(args.filename.lower() + '.csv', encoding='UTF-8', sep=';')
    df = normalize_data(data)
    df_to_show = data.drop(['Отметка времени', 'Чай или кофе'], axis=1)
    X = df.drop(['tea_or_coffee'], axis=1).values
    y = df.tea_or_coffee.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    with KNearestNeighborsClassifier(k=args.k) as self_made_clf:
        self_made_clf.fit(X=X_train, y=y_train)
        if args.i:
            print(df_to_show)
            print(self_made_clf.single_predict(X[args.i]))
        else:
            predictions_from_self_made_clf = self_made_clf.predict(X_test)
            scikit_learn_clf = KNeighborsClassifier(n_neighbors=args.k)
            scikit_learn_clf.fit(X_train, y_train)
            predictions_from_scikit_learn_model = scikit_learn_clf.predict(X_test)
            self_made_accuracy = get_accuracy(predictions_from_self_made_clf, y_test)
            scikit_learn_accuracy = get_accuracy(predictions_from_scikit_learn_model, y_test)
            print(
                f'Точность самописной модели, где k = {args.k} = {self_made_accuracy * 100}%'
                f'\n{coffee_tea_replacer(predictions_from_self_made_clf)} '
                f'Self made model\n{coffee_tea_replacer(y_test)} Test data')
            print(
                f'Точность компьютерной модели, где k = {args.k} = {scikit_learn_accuracy * 100}'
                f'%\n{coffee_tea_replacer(predictions_from_scikit_learn_model)} '
                f'Scikit learn model\n{coffee_tea_replacer(y_test)} Test data')


if __name__ == '__main__':
    main()
