import numpy as np
import pandas as pd
class K_means:
    def __init__(self, num_of_klusters, degree):
        self.degree = degree
        self.k = num_of_klusters
        self.klusters = []
    def __dist(self, x, y, degree):
        distance = 0
        for i in range(x.shape[0]):
            distance+=np.abs(x[i]-y[i])**degree
        return distance**(1/degree)
    def __initialize_random_centroids(self, X):
        centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        for i in range(self.k):
            self.klusters.append(dict(centroid = centroids[i], objects = []))
    def __update_classes(self, X):
        for obj in X:
            distances = []
            for kluster in self.klusters:
                distances.append(self.__dist(obj, kluster["centroid"], self.degree))
            self.klusters[distances.index(min(distances))]["objects"].append(obj)
    def __update_centroids(self, X):
        for i in range(self.k):
            self.klusters[i]["centroid"] = np.mean(self.klusters[i]["objects"], axis=0)
    def fit(self, X, num_of_iterations):
        self.__initialize_random_centroids(X)
        for i in range(num_of_iterations):
            self.__update_classes(X)
            self.__update_centroids(X)
    def predict(self, X):
        predictions = []
        for st in X:
            distances = []
            for kluster in self.klusters:
                distances.append(self.__dist(st, kluster["centroid"], self.degree))
            predictions.append(distances.index(min(distances)))
        return predictions