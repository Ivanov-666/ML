import numpy as np
def confusion_matrix(y, y_pred):
    clas = len(np.unique(y))
    conf_matrix = np.zeros((clas, clas))
    for i in range(len(y)):
        conf_matrix[y.iloc[i]][y_pred[i]] += 1
    return conf_matrix
def acurracy(y, y_pred):
    m = confusion_matrix(y, y_pred)
    return (m[0][0]+m[1][1])/len(y)
def precision(y, y_pred):
    m = confusion_matrix(y, y_pred)
    return m[1][1]/(m[1][1]+m[0][1])
def recall(y, y_pred):
    m = confusion_matrix(y, y_pred)
    return m[1][1]/(m[1][1]+m[1][0])
def f1(y, y_pred):
    return 2/((1/precision(y, y_pred))+(1/recall(y, y_pred)))