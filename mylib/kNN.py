import numpy as np
import pandas as pd
import math
def dist(x, y, p):
        distance = 0
        for i in range(x.shape[0]):
            distance+=np.abs(x[i]-y[i])**p
        return distance**(1/p)
class kNN:
    def __init__(self, k, p, classes):
        self.k = k
        self.p = p
        self.classes = classes
    def fit(self,X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
       
    def predict(self,X_test):
        labels = []
        for vector in X_test:
            distance = [[self.y_train[i],dist(vector, self.X_train[i], self.p)]for i in range(self.X_train.shape[0])]
            distance = sorted(distance,key = lambda x: x[1])
            classes_around = np.zeros(self.classes)
            for i in range(self.k):
                classes_around[int(distance[i][0])]+=1
            labels.append(sorted(zip(classes_around, range(self.classes)), key=lambda x:x[0], reverse=True)[0][1])
        return labels