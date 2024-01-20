import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# import data
df = pd.read_csv("dataset/diabetes2.csv")
X = df.drop(columns="Outcome", axis=1)
y = df[["Outcome"]]

# normalize data
def standard_deviation(Variables):
    features = list(Variables)
    for feature in features:
        standard_deviation = Variables[feature].std()
        mean = Variables[feature].mean()
        Variables[feature] = (Variables[feature]-mean)/standard_deviation
    return(Variables)
X = standard_deviation(X)

#split train/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 64)
X_train = np.asanyarray(X_train)
y_train = np.asanyarray(y_train)
X_test = np.asanyarray(X_test)
y_test = np.asanyarray(y_test)

#sigmoid function
def sigmoid(X, W, b):
    return 1/(1+np.exp(-(np.dot(X, W)+b)))

# cross entropy
def cross_entropy(X, W, b,y):
    y_hat = sigmoid(X, W, b)
    return -y*np.log(y_hat)-(1-y)*np.log(1-y_hat)

# cost function
def cost_function(X, W, b, y):
    m = X.shape[1]
    sum = 0
    for i in range(m):
        sum += cross_entropy(X[i], W, b, y[i])
    return sum/m

# derivative
class derivative:
    def __init__(self, weight, bias, X, y, alpha):
        self.weight = weight
        self.bias = bias
        self.X = X
        self.y = y
        self.y_hat = sigmoid(X, self.weight, self.bias)
        self.alpha = alpha
    def derivative_of_L_respect_to_weight(self):
        n = self.X.shape[1]
        m = self.X.shape[0]
        for i in range(n):
            sum_i_to_m = 0
            for j in range(m):
                sum_i_to_m += (self.y_hat[j][0]-self.y[j][0])*self.X[j][i]
            self.weight[i][0] = self.weight[i][0]-self.alpha*sum_i_to_m
        return self.weight
    def derivative_of_L_respect_to_bias(self):
        m = self.X.shape[0]
        sum = 0
        for i in range(m):
            sum += self.y_hat[i][0]-self.y[i][0]
        self.bias = self.bias - self.alpha*sum
        return self.bias
        
# gradient descent
def gradient_descent(X, y, weight_init, bias_init, alpha, iter):
    W = weight_init
    b = bias_init
    cost = []
    weight = []
    bias = []
    for i in range(iter):
        W = derivative(W, b, X, y, alpha).derivative_of_L_respect_to_weight()
        b = derivative(W, b, X, y, alpha).derivative_of_L_respect_to_bias()
        cost.append(cost_function(X, W, b, y)[0])
        weight.append(np.asanyarray(W))
        bias.append(np.asanyarray(b))
    min_index = cost.index(min(cost))
    return min(cost), weight[min_index], bias[min_index] 

# train/test the logistic model with different learning rate
a = np.linspace(0.001, 0.1, 10)
accuracy = []
cost_arr = []
True_ = []
for j in range(10):
    W =  np.zeros([X_train.shape[1],1])
    b = 0
    cost, weight, bias = gradient_descent(X_train, y_train, W, b, a[j], 100)
    yhat = sigmoid(X_test, weight, bias)
    for i in range(yhat.shape[0]):
        if yhat[i][0] > 0.5:
            yhat[i][0] = 1
        else:
            yhat[i][0] = 0
    true = 0
    for i in range(yhat.shape[0]):
        if yhat[i][0] == y_test[i][0]:
            true += 1
    accuracy.append(true/yhat.shape[0])
    cost_arr.append(cost)
    True_.append(true)
    print("Sweep 1 " + str(j+1) +":")
    print("Cost: "+ str(cost))
    print("Predicted true:" + str(true) + "/" + str(yhat.shape[0]))
    print("Accuracy: " + str(accuracy[j]))

# scale the cost to plot
cost_arr = cost_arr/max(cost_arr)

# visualization
plt.plot(a, accuracy, color = "blue", label = "accuracy")
plt.plot(a, cost_arr, color = "red", label = "cost")
plt.title("Accuracy/Cost according to learning rate")
plt.legend()
plt.xlabel("\u03B1")
plt.ylabel("accuracy")
plt.show()
