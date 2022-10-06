import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from kNN import KNearestNeighborsClassifier, normalize_data, get_accuracy


def main():
    data = pd.read_csv(input('Input filepath: ').lower()+'.csv', encoding='UTF-8', sep=';')
    df = normalize_data(data)
    X = df.drop(['tea_or_coffee'], axis=1).values
    y = df.tea_or_coffee.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    self_made_clf = KNearestNeighborsClassifier(k=3, X=X_train, y=y_train)
    scikit_learn_clf = KNeighborsClassifier(n_neighbors=3)
    scikit_learn_clf.fit(X_train, y_train)
    predictions1 = self_made_clf.predict(X_test)
    predictions2 = scikit_learn_clf.predict(X_test)
    print(f'Accuracy from self-made model{get_accuracy(predictions1, y_test)}')
    print(f'Accuracy from scikit-learn model{get_accuracy(predictions2, y_test)}')


if __name__=='__main__':
    main()