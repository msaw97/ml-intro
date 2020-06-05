import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.model_selection
import sklearn.neighbors
import sklearn.metrics


print("CEL - klasyfikacja win ze względu na zmienną response\n")
#response określa jakość wina (mediana z opinii trzech ekspertów) - w skali od 0 (bardzo złe) do 10 (doskonałe)


wine = pd.read_csv("data/winequality-all.csv", comment="#")
print(wine.head())


print("\nRozkład liczebności klas zmiennej response:")
ile = wine["response"].value_counts()
print(ile.iloc[np.argsort(ile.index)]) #sortowanie

#rozkład jest nierównomierny, brakuje wartości 8, 9 i 10
#proponujemy zatem dwie klasy response < 5 (wina złe) i response >= 5 (wina dobre)
print("\nDodana zmienna quality - złe jeśli response < 5, dobre jeśli response >= 5:")
wine["quality"] = pd.cut(wine["response"], [0, 5, 10], right=False, labels=["złe", "dobre"])
wine["quality"].value_counts()
print(wine.iloc[5:20,:])

X = wine.iloc[:, 0:11]
#print(X.head())
y = wine["quality"]
#print(y.head())

#skoro mamy do czynienia z klasyfikacją binarną ("złe" - "dobre") - y(i) należy do zbioru {0,1}
#to warto przekodować wartości zmiennej y na zbiór liczb całkowitych
yk = y.cat.codes.values
#print("yk:", yk[1:10])

#wybieram kilka kodów win wybranych losowo
i = np.random.choice(np.arange(len(yk)), 10, replace=False)
#print(yk[i])

#Podział zbioru na próbę uczącą (80%) i testową (20%)

idx_ucz, idx_test = sklearn.model_selection.train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=12345)
X_ucz, X_test = X.iloc[idx_ucz, :], X.iloc[idx_test, :]
y_ucz, y_test = y[idx_ucz], y[idx_test]
yk_ucz, yk_test = yk[idx_ucz], yk[idx_test]


#metoda k-najbliższych sąsiadów
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5) #5-nn
print(knn.fit(X_ucz, yk_ucz))

#dokonujemy predykcji przynależności obserwacji do dwóch klas zbioru yk na 
#podstawie zbioru testowego X_test

yk_pred = knn.predict(X_test)
y_pred = y.cat.categories[yk_pred]
#print(yk_pred.shape)

print("\nPorównanie 0, 250, 500, 600, 750, 800, 1000 obserwacji predykcji i zbioru bazowego")
print("Predykcje:", np.array(y_pred[[0, 250, 500, 600, 750, 800, 1000]]))
print("Wartości bazowe:", np.array(y[[0, 250, 500, 600, 750, 800, 1000]]))

print("\nOcena jakości klasyfikatora")
print("Procent poprawnie zaklasyfikowanych obserwacji:", sklearn.metrics.accuracy_score(yk_test, yk_pred))

print("Macierz pomyłek:") #[[true negative, false positive], [false negative, true positive]]
print(sklearn.metrics.confusion_matrix(yk_test, yk_pred))


#dokładność - accuracy_score (ACC), precyzja - precision_score (P), czułość - recall_score (R),
#miara F1, czyli średnia harmoniczna z czułości i precyzji - f1_score (F1)

def fit_classifier(alg, X_ucz, X_test, y_ucz, y_test):
    alg.fit(X_ucz, y_ucz)
    y_pred = alg.predict(X_test)
    return {
        "ACC:": sklearn.metrics.accuracy_score(y_pred, y_test),
        "P:":   sklearn.metrics.precision_score(y_pred, y_test),
        "R:":   sklearn.metrics.recall_score(y_pred, y_test),
        "F1:":  sklearn.metrics.f1_score(y_pred, y_test)
    }


m = X_ucz.mean(axis=0)
s = X_ucz.std(axis=0, ddof=1)
fc = pd.Series(fit_classifier(sklearn.neighbors.KNeighborsClassifier(n_neighbors=5), (X_ucz-m)/s, (X_test-m)/s, yk_ucz, yk_test))
print(fc)