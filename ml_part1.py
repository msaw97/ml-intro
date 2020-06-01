import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

#przykładowy zbiór danych - pomiary fizykochemiczne wlasności portugalskich win typu Vinho Verde (białe i czerwone) 
wine = pd.read_csv("data/winequality-all.csv", comment="#")

print(wine.head())
print(wine.info())
print("-----------------------------------------------------------------")
print("shape:", wine.shape)
print("data types:", wine.dtypes)
print("columns:", wine.columns)
print("-----------------------------------------------------------------")


#kolor wina jest typu object, więc musimy zmienić tą zmienną na zmienną kategoryczną
wine.color = wine.color.astype("category")

#print(wine.describe())
print(wine.iloc[:, 0:11].describe().round(1).T.iloc[:, 1:])

#------------------------------------------------------------------------
#CEL - sprawdzimy czy alkohol jest funkcją pozostałych 10 zmiennych i jaka jest ta zależność
#dzięki temu będziemy w stanie wyjaśnić pochodną jakiego zbioru czynników jest dana zawartość alkoholu
#a także przewidzieć zawartość alkoholu w nowo wyprodukowanej partii wina

print("\nIlość win białych i czerwonych w zbiorze:")
print(wine.color.value_counts())

print("\nWina białe:")
white_wine = wine[wine.color == "white"]
white_wine = white_wine.iloc[:, 0:11]
print(white_wine.head())


#tworzymy macierze zmiennych objaśniających (predyktorów) i wektor kolumnowy zmiennej objaśnianej
y = white_wine.iloc[:, -1]
#print(y.head())
X = white_wine.iloc[:, :-1]
#print(X.head())

#obliczmy wspóczynnik korelacji liniowej Pearsona
corr_P = white_wine.corr("pearson")
#print("\nWspóczynnik korelacji liniowej Pearsona:")
#print(corr_P.shape)
#print(corr_P)

#tworzymy macierz trójkątną i wyświetlamy wspóczynnik korelacji większy od 0.5
corr_P_tri = corr_P.where(np.triu(np.ones(corr_P.shape, dtype=np.bool), k=1)).stack().sort_values()
#print(corr_P_tri)

print("\nWspóczynnik korelacji liniowej Pearsona >0.5:")
print(corr_P_tri[abs(corr_P_tri)>0.5])

#wizualizacja skupień przy pomocy seaborn pairplot
sns.pairplot(white_wine)
plt.show()