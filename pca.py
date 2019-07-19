from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# 1 - OK
data_csv = pd.read_csv('close_prices.csv', index_col=0)
for comp in range(10, 0, -1):
    pca = PCA(n_components=comp)
    pca.fit(data_csv)
    #print(pca.explained_variance_ratio_.sum())
    #print(pca.n_components)

pca.n_components = 4
data = pca.fit_transform(data_csv)
# 2 - OK
dj = pd.read_csv('djia_index.csv', index_col=0)
corr = np.corrcoef(data[:, 0], dj['^DJI'])
print(corr)
#3
max_comp = np.argmax(pca.components_[0])
print(data_csv.keys()[max_comp])

print()
corr = np.corrcoef(data_csv, rowvar=False)


# min_correlation

corr_array = np.argwhere(np.abs(corr)<0.002)
for i in corr_array:
    for j in i:
        print(data_csv.keys()[j])




