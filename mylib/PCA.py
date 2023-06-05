import numpy as np
from sklearn.preprocessing import StandardScaler
class PCA:
    def __init__(self, n_classes_after):
        self.n = n_classes_after
    def fit_transform(self, X):
        stsc = StandardScaler()
        X = stsc.fit_transform(X)
        cov_matrix = np.cov(X.T)
        values, vectors = np.linalg.eigh(cov_matrix)
        pairs = list(zip(values, vectors))
        pairs = sorted(pairs, key=lambda x: (x[0],x[1][0],x[1][1]))[:self.n]
        pairs = map(lambda a: a[1], pairs)
        principle_data = np.dot(X, np.array(list(pairs)).T)
        return principle_data