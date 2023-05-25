import numpy as np
import pandas as pd
from collections import defaultdict
import math

class NaiveBayes: 
    def __init__(self):
        pass
    def fit(self, X_train, y_train):
        self.classes_exampls=dict()
        X_train_spec = pd.concat([X_train, y_train], axis=1)
        X_train_spec= np.array(X_train_spec, dtype=np.float64)
        X_train = np.array(X_train, dtype=np.float64)
        y_train = np.array(y_train, dtype=np.float64)
        for j in range(len(set(y_train))):    
            new_list = [i[:-1] for i in X_train_spec if i[-1] == j]
            new_arr = np.array(new_list)
            self.classes_exampls[j]=new_arr
        self.mean_znach, self.stand_otcl = np.zeros((len(set(y_train)), X_train.shape[1])), np.zeros((len(set(y_train)), X_train.shape[1]))
        for i in range(len(self.classes_exampls)):
            buffer = self.classes_exampls[i]
            for j in range(X_train.shape[1]):
                self.mean_znach[int(i)][j] = np.mean(buffer[:,j])
                self.stand_otcl[int(i)][j] = np.std(buffer[:,j])
        
    def predict(self, X_test):
        X_test = np.array(X_test, dtype=np.float64)
        prediction = []
        for line in X_test:
            predict_ver = []
            for clas in self.classes_exampls.keys():
                buffer = 1
                for j in range(len(line)): 
                    exp = math.exp(-((line[j]-self.mean_znach[int(clas)][j])**2)/(2*( self.stand_otcl[int(clas)][j]**2)))
                    buffer*=(1 / (math.sqrt(2*math.pi) * self.stand_otcl[int(clas)][j])) * exp
                predict_ver.append([clas,buffer])
            prediction.append(sorted(predict_ver, key=lambda x: x[1], reverse=True)[0][0])
        return prediction