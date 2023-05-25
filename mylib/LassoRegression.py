import numpy as np
import pandas as pd
import math
class LassoRegressionModel:
    def __init__(self, education_speed, education_iterations,l1_coef):
        self.education_speed = education_speed
        self.education_iterations = education_iterations
        self.l1_coef = l1_coef
        
    def Train(self, x, y):
        self.x = np.array(x, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64)
        m = self.x.shape[0]
        self.coefs = np.zeros(self.x.shape[1]+1)
        list_errors = []
        for k in range(self.education_iterations):
            for i in range(self.x.shape[1]):
                y_base = self.coefs[0]+np.dot(self.x[i],self.coefs[1:])
                gradient = np.zeros(self.x.shape[1])
                if self.coefs[i+1] > 0 :
                    gradient[i] = ( - ( 2 * ( self.x[:, i] ).dot( self.y - y_base ) ) 
                    + self.l1_coef )/m
                else :
                    gradient[i] = ( - ( 2 * ( self.x[:, i] ).dot( self.y - y_base ) ) 
                    - self.l1_coef )/m
                gradient_b = - 2 * np.sum( self.y - y_base ) / m 
            self.coefs[0]-=self.education_speed * gradient_b
            self.coefs[1:]-=self.education_speed * gradient
        return self
        
    def LassoRegressionPredict(self, x):
        y_base = self.coefs[0]
        y_base+=np.dot(x,self.coefs[1:])
        return y_base

def MAE(y, y_pred):
    m = y.shape[0]
    return (1/m)*np.sum(np.abs(y-y_pred))
def MSE(y, y_pred):
    m = y.shape[0]
    return (1/m)*np.sum((y-y_pred)**2)
def RMSE(y, y_pred):
    return math.sqrt(MSE(y, y_pred))
def MAPE(y, y_pred):
    m = y.shape[0]
    return (1/m)*np.sum(np.abs((y-y_pred)/y))
def R2(y, y_pred):
    m = y.shape[0]
    y_mean = np.mean(y)
    return 1-(MSE(y,y_pred)/((1/m)*np.sum((y-y_mean)**2)*y_mean))