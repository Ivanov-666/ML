import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit, logit
def sigmoid(x):
    #return (1.0/(1.0+np.exp(-x)))
    return expit(x)
def tahn(x):
    return np.tanh(x)    
def relu(x):
    return np.maximum(0,x)
def linear(x):
    return x
def deriv_sigmoid(x):
    #return sigmoid(x)*(1-sigmoid(x))
    return expit(x)*(1-expit(x))
def deriv_tahn(x):
    return -4/(np.exp(x)-np.exp(-x))**2
def deriv_relu(x):
    return (x>0)*1
def deriv_linear(x):
    return 1
def forward_propagation(x_row,layers, activation_functions):
    for layer in layers:
        x_row = np.dot(layer["weights"],x_row.T)+layer["b_coefs"]
        for i in range(len(x_row)):
            x_row[i]=activation_functions[layer["activation"]](x_row[i])
        layer["output"]=x_row
def backward_propagation(layers, y_train, deriv_functions):
    errors = []
    for j in range(len(layers[-1]["output"])):
        errors.append(y_train[j]-layers[-1]["output"][j])
    errors = np.array(errors)
    deriv = []
    for i in range(len(layers[-1]['output'])):
        deriv.append(deriv_functions[layers[-1]["activation"]](layers[-1]['output'][i]))
    adamar = [errors[j]*deriv[j] for j in range(len(errors))]
    layers[-1]["err"]=adamar
    for i in reversed(range(len(layers)-1)):
        error = np.dot(layers[i+1]["err"], layers[i+1]["weights"])
        for j in range(len(layers[i]['output'])):
            deriv.append(deriv_functions[layers[i]["activation"]](layers[i]['output'][j]))
        adamar = [error[j]*deriv[j] for j in range(len(error))]
        layers[i]["err"]=adamar
def update_weights(x_row, layers, learning_rate):
    DWl = learning_rate*np.dot(np.array([x_row]).T,np.array([layers[0]["err"]]))
    layers[0]["weights"]+=DWl.T
    DBl = np.dot(learning_rate,layers[0]["err"])
    layers[0]["b_coefs"]+=DBl.T
    for i in range(1,len(layers)):
        inputs = np.array([layers[i-1]["output"]])
        DWl = learning_rate*np.dot(inputs.T,np.array([layers[i]["err"]]))#поменял минусы на плюсы и заработало
        layers[i]["weights"]+=DWl.T
        DBl = np.dot(learning_rate,layers[i]["err"])
        layers[i]["b_coefs"]+=DBl.T
class Perceptron:
    def __init__(self, input_layer_size, hidden_layers):
        self.activation_functions = dict(sigmoid=sigmoid,tahn=tahn,relu=relu, linear=linear)
        self.deriv_functions = dict(sigmoid=deriv_sigmoid,tahn=deriv_tahn,relu=deriv_relu, linear=deriv_linear)
        self.layers = []
        w = np.random.rand (hidden_layers[0][0], input_layer_size)
        b = np.random.rand (hidden_layers[0][0])
        self.layers.append(dict(weights=w,b_coefs=b, activation=hidden_layers[0][1]))
        for i in range(1, len(hidden_layers)):
            w = np.random.rand (hidden_layers[i][0], hidden_layers[i-1][0])
            b = np.random.rand (hidden_layers[i][0])
            self.layers.append(dict(weights=w,b_coefs=b,activation=hidden_layers[i][1]))
    def fit(self, X_train, y_train, learn_rate, epohs):
        for ep in range(epohs):
            for ind in range((len(X_train))):
                forward_propagation(np.array(X_train[ind]), self.layers, self.activation_functions)
                backward_propagation(self.layers, y_train[ind], self.deriv_functions)
                update_weights(X_train[ind], self.layers, learn_rate)
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            forward_propagation(row, self.layers, self.activation_functions)
            predictions.append(self.layers[-1]["output"])
        return predictions