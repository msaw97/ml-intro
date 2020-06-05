# Introduction to machine learning

## Repozytorium zawiera:
#### Plik ml_part1.py
- wstępną analiza zbioru danych
- porównanie wspóczynnika korelacji liniowej Pearsona
- wizualizacje skupień przy pomocy seaborn pairplot

![](/images/wykres_skupien.png)

#### Plik ml_regression.py
- model regresji liniowej z różnymi podziałami zbioru na próbę uczączą i testową
- model regresji wielomianowej wraz z redukcją zmiennych modelu
- określanie błędów dopasowania i predykcji
- wykres zmiany miary błędów dopasowania MSE i MAE dla zbiorów uczączych i testowych

![](/images/wykres_regresja.png)

#### Plik ml_classifier.py
- model klasyfikatora
- klasyfikacja win ze względu na zmienną response metodą k-najbliższych sąsiadów
- ocena jakości klasyfikatora według miar accuracy_score, precision_score, recall_score, f1_score