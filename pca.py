from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
'''
Загрузите данные close_prices.csv. В этом файле приведены цены акций 30 компаний
на закрытии торгов за каждый день периода.

1. На загруженных данных обучите преобразование PCA с числом компоненты равным 10. 
Скольких компонент хватит, чтобы объяснить 90% дисперсии?

Примените построенное преобразование к исходным данным и возьмите значения 
первой компоненты.

2.
Загрузите информацию об индексе Доу-Джонса из файла djia_index.csv. 

 Чему равна 
корреляция Пирсона между первой компонентой и индексом Доу-Джонса?
3. Какая компания имеет наибольший вес в первой компоненте? Укажите ее название с 
большой буквы.
Если ответом является нецелое число, то целую и дробную часть необходимо 
разграничивать точкой, например, 0.42. При необходимости округляйте дробную 
часть до двух знаков.
'''
# 1 - OK
data = pd.read_csv('close_prices.csv', index_col=0)
for comp in range(10, 0, -1):
    pca = PCA(n_components=comp)
    pca.fit(data)
    #print(pca.explained_variance_ratio_.sum())
    #print(pca.n_components)

pca.n_components = 4
data = pca.fit_transform(data)
# 2 - OK
dj = pd.read_csv('djia_index.csv', index_col=0)
corr = np.corrcoef(data[:, 0], dj['^DJI'])
print(corr)
#3
print(pca.components_)
