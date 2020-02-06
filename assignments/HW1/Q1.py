import numpy as np
import math
import matplotlib.pyplot as plt

############################## Part b ########################################
def sigmoid(X):
    return 1/(1 + np.exp(-X))

def log_likelihood(X, w, y):
    epsilon = 1e-5  
    p = sigmoid(X.dot(w))
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def compute_gradient(X, w, y):
    grad = -X.T.dot(y - sigmoid(X.dot(w)))
    return grad

def hessian(X, w, y):
    m = sigmoid(X.dot(w))*(1-sigmoid(X.dot(w))).reshape(1, X.shape[0])
    h = -X.T @ np.diag(np.diag(m)) @ X
    return h


def newton(X, w, y, num_iter = 100):
    print("***************Newton's method********************")
    l = 0
    l_new = 1
    epsilon = 1e-10
    i = 1
    while abs(l - l_new) > epsilon and i <= num_iter:
        l = l_new
        grad = compute_gradient(X, w, y)
        w = w + np.linalg.inv(hessian(X, w, y)).dot(grad)
        l_new = log_likelihood(X, w, y)
        print("This is Iteration {} and the log likelihood is {}".format(i, l_new))
        i += 1
    print("The final weight is {}".format(w))
    return w
        

# Load the data:
X = np.load('data/q1x.npy')
y = np.reshape(np.load('data/q1y.npy'), (X.shape[0], 1))
N = X.shape[0]
X = np.concatenate((np.ones((N, 1)), X), axis=1)
w = np.zeros((X.shape[1], 1))

# print(X, X.shape)
# print(y, y.shape)
w = newton(X, w, y)



############################## Part c ########################################
x1 = X[:,1]
x2 = X[:,2]

plt.figure(figsize=(10,6))
plt.scatter(x1, x2, c = y[:,0])
x_range = np.arange(min(x1), max(x1))
plt.plot(x_range, -(w[0,] + w[1, ]*x_range)/w[2,], c = "red")
plt.title("A linear decision boundary for Newton's method")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show() 