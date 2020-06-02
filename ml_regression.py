#Model regresji liniowej

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.linear_model

wine = pd.read_csv("data/winequality-all.csv", comment="#")

white_wine = wine[wine.color == "white"]
white_wine = white_wine.iloc[:, 0:11]
print(white_wine.describe())

y = white_wine.iloc[:, -1]
X = white_wine.iloc[:, :-1]


mnk = sklearn.linear_model.LinearRegression()
mnk.fit(X,y)
#print(mnk.coef_)
#print(mnk.intercept_)


x_nowy = X.mean().values.reshape(1,-1)
#print(x_nowy)

print("\nWartość średnia alkoholu:", white_wine.alcohol.mean())
print("Przewidziana średnia zawartość alkoholu:", mnk.predict(x_nowy))


#dla zbioru wystandaryzowanego, współczynniki regresji nabierają przydatnej interpretacji
#tzn. im większa wartość modułu współczynnika, tym bardziej istotny ma on wpływ na wartość odpowiedzi
print("\nZbiór wystandaryzowany:")
X_std = (X-X.mean(axis=0))/X.std(axis=0)
print(X_std.describe())

mnk_std = sklearn.linear_model.LinearRegression()
mnk_std.fit(X_std, (y-y.mean())/y.std())

y_std = (y-y.mean())/y.std()

print("\nWpływ zmiennej alkoholu na poszególne zmienne:")
print(pd.Series(np.abs(mnk_std.coef_), index=X.columns.to_list()).round(4).sort_values(ascending=False))

#ocena jakości modelu
#porównanie wartości dopasowanych, obliczonych za pomocą modelu z wartościami oryginalnymi
y_pred = mnk.predict(X)

print("\nWartości dopasowane:")
print(pd.Series(y_pred[0:15]))

print("\nWartości oryginalne:")
print(y[0:15])


print("\nWspółczynnik determinacji R2:", sklearn.metrics.r2_score(y, y_pred))

print("MSE:", sklearn.metrics.mean_squared_error(y, y_pred))
print("MAE:", sklearn.metrics.mean_absolute_error(y, y_pred))
print("MedAE:", sklearn.metrics.median_absolute_error(y, y_pred))

#próba ucząca (80%), testową (20%)
X_ucz, X_test, y_ucz, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=12345)
#print(X_ucz.shape)
#print(X_test.shape)
#print(y_ucz.shape)
#print(y_test.shape)


#funkcja, która dopasowuje model regresji liniowej do danej próby
#oraz oblicza miary błędów dopasowania
def fit_regression(X_ucz, X_test, y_ucz, y_test):
    r = sklearn.linear_model.LinearRegression()
    r.fit(X_ucz, y_ucz)
    y_ucz_pred = r.predict(X_ucz)
    y_test_pred = r.predict(X_test)
    r2 = sklearn.metrics.r2_score
    mse = sklearn.metrics.mean_squared_error
    mae = sklearn.metrics.mean_absolute_error
    return {
        "r_score_u": r2(y_ucz, y_ucz_pred),
        "r_score_t": r2(y_test, y_test_pred),
        "MSE_u": mse(y_ucz, y_ucz_pred),
        "MSE_t": mse(y_test, y_test_pred),
        "MAE_u": mae(y_ucz, y_ucz_pred),
        "MAE_t": mae(y_test, y_test_pred)
    }


params = ["Reg. liniowa bazowa"]
res = [fit_regression(X_ucz, X_test, y_ucz, y_test)]

X_ucz70, X_test70, y_ucz70, y_test70 = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=12345)



#próba ucząca (70%), testową (30%)
params.append("Reg. liniowa 70")
res.append(fit_regression(X_ucz70, X_test70, y_ucz70, y_test70))


#próba ucząca (70%), testową (30%) zbiory standaryzowane
params.append("Reg. liniowa 70 std")
X_ucz70std, X_test70std, y_ucz70std, y_test70std = sklearn.model_selection.train_test_split(X_std, y_std, test_size=0.3, random_state=12345)
res.append(fit_regression(X_ucz70std, X_test70std, y_ucz70std, y_test70std))

reg = pd.DataFrame(res, index=params)
print(reg)