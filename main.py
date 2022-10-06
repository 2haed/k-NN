import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from kNN import KNearestNeighborsClassifier, normalize_data, get_accuracy
import argparse


def main():
    parser = argparse.ArgumentParser(description='Videos to images')
    parser.add_argument('--filename', type=str, help='Input filename')
    parser.add_argument('--k', type=int, help='k number of neighbors')
    args = parser.parse_args()

    data = pd.read_csv(args.filename.lower() + '.csv', encoding='UTF-8', sep=';')
    df = normalize_data(data)
    X = df.drop(['tea_or_coffee'], axis=1).values
    y = df.tea_or_coffee.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    with KNearestNeighborsClassifier(k=args.k) as self_made_clf:
        self_made_clf.fit(X=X_train, y=y_train)
        predictions_from_self_made_clf = self_made_clf.predict(X_test)
        print(f'Accuracy from self-made model{get_accuracy(predictions_from_self_made_clf, y_test)}')

    scikit_learn_clf = KNeighborsClassifier(n_neighbors=args.k)
    scikit_learn_clf.fit(X_train, y_train)
    predictions2 = scikit_learn_clf.predict(X_test)
    print(f'Accuracy from scikit-learn model{get_accuracy(predictions2, y_test)}')


if __name__=='__main__':
    main()