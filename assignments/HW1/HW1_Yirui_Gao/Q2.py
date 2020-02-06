import numpy as np
import matplotlib.pyplot as plt
import time

# Load the data:
X_train = np.load('data/q2xTrain.npy')
y_train = np.load('data/q2yTrain.npy')
X_test = np.load('data/q2xTest.npy')
y_test = np.load('data/q2yTest.npy')


def construct_polynomial(X_vec, degree):
    X = np.ones((X_vec.shape[0], 1))
    for i in range(1, degree + 1):
        X = np.append(X, np.reshape(X_vec**i, (X_vec.shape[0], 1)), axis=1)
    return X

############################## (a) ############################################
# Define the cost function:
def cost_fuction(X, theta, y, lambda_ = 0):
    regularization = 0.5 * lambda_ * (theta.T.dot(theta))
    return 0.5 * np.sum((np.dot(X, theta) - y) ** 2) + np.asscalar(regularization)

def batchGD(X, theta, y, learning_rate = 0.001, num_iter = 10000):
    print("***************Batch GD********************")
    e = 0
    e_new = 1
    tolerance = 1e-10
    i = 0
    error_list = []
    while abs(e - e_new) > tolerance and i <= num_iter:
        i += 1
        e = e_new
        grad = np.dot(X.T, np.dot(X, theta) - y)
        theta = theta - learning_rate * grad
        e_new = cost_fuction(X, theta, y)
        error_list.append(e_new)
    print("It takes number of iteration # {} with final cost {}".format(i, e_new))
    print("The final result for theta is ", theta)

    return error_list



def SGD(X, theta, y, learning_rate = 0.001, num_iter = 10000):
    print("***************Stochastic GD********************")
    e = 0
    e_new = 1
    tolerance = 1e-10
    j = 0
    error_list = []
    while abs(e - e_new) > tolerance and j <= num_iter:
        j += 1
        e = e_new
        e_new = 0
        for i in range(n):
            X_i = X[i,:].reshape(1, X.shape[1])
            y_i = y[i].reshape(1,1)
            prediction_i = np.dot(X_i, theta)
            grad = np.dot(X_i.T, prediction_i - y_i)
            theta = theta - learning_rate * grad
            e_new += cost_fuction(X_i, theta, y_i)
        error_list.append(e_new)        
    print("It takes number of iteration # {} with final cost {}".format(j, e_new))
    print("The final result for theta is ", theta)
    return error_list



def newton(X, theta, y, num_iter = 15):
    print("***************Newton's method********************")
    H = X.T.dot(X)
    e = 0
    e_new = 1
    tolerance = 1e-10
    i = 0
    error_list = []
    while abs(e - e_new) > tolerance and i <= num_iter:
        i += 1
        e = e_new
        grad = np.dot(X.T, np.dot(X, theta) - y)
        theta = theta -  np.linalg.inv(H).dot(grad)
        e_new = cost_fuction(X, theta, y)
        # print("Iteration # {} with cost {}".format(i, e_new))
        error_list.append(e_new)
    print("It takes number of iteration # {} with final cost {}".format(i, e_new))
    print("The final result for theta is ", theta)
    return error_list, theta


n = X_train.shape[0]
# Create the data matrix of (n,j):
X = construct_polynomial(X_train, 1)
theta = np.zeros((2, 1))
y = np.reshape(y_train, (n, 1))
# print(X, X.shape)
# print(theta, theta.shape)
# print(y, y.shape)

start_time1 = time.time()
error_list1 = batchGD(X, theta, y)
time_GD = time.time() - start_time1

start_time2 = time.time()
error_list2 = SGD(X, theta, y)
time_SGD = time.time() - start_time2

start_time3 = time.time()
error_list3 = newton(X, theta, y)
time_newton = time.time() - start_time3


num_iter = 1000
plt.figure(figsize=(10,6))
plt.plot(range(1, num_iter + 1), error_list1, color='red')
plt.plot(range(1, num_iter + 1), error_list2, color='green')
plt.plot(range(1, num_iter + 1), error_list3, color='blue')
plt.title("Learning curve for Newton's method.")
plt.xlabel("#Iteration")
plt.ylabel("Error")
plt.show() 



# ############################## (b) ############################################

ytest = np.reshape(y_test, (X_test.shape[0], 1))

M = range(0, 10)
ERMS_list = []
ERMS_list_test = []

for i in M:
    X = construct_polynomial(X_train, i)
    testX = construct_polynomial(X_test, i)
    theta = np.zeros((i + 1, 1))
    e, theta = newton(X, theta, y)
    print("The theta for the {}-degree is: {}".format(i, theta))
    e_test = cost_fuction(testX, theta, ytest)
    
    ERMS = np.sqrt(2 * e[-1] / X.shape[0])
    ERMS_list.append(ERMS)

    ERMS_test = np.sqrt(2 * e_test / testX.shape[0])
    ERMS_list_test.append(ERMS_test)

plt.figure(figsize=(10,6))
train, = plt.plot(M, ERMS_list, '--o', color='blue', label='Training')
test, = plt.plot(M, ERMS_list_test, '--o', color='red', label='Test')
plt.legend(handles=[train, test])
plt.xlabel("M")
plt.ylabel("ERMS")
plt.show() 


############################## (c) ############################################
def newton_regularization(X, theta, y, lambda_, num_iter = 15):
    print("***************Newton's method********************")
    H = X.T.dot(X)
    e = 0
    e_new = 1
    tolerance = 1e-10
    i = 0
    error_list = []
    while abs(e - e_new) > tolerance and i <= num_iter:
        i += 1
        e = e_new
        grad = np.dot(X.T, np.dot(X, theta) - y) + lambda_ * theta
        theta = theta - np.linalg.inv(np.add(H, lambda_ * np.identity(X.shape[1]))).dot(grad)
        e_new = cost_fuction(X, theta, y, lambda_)
        # print("Iteration # {} with cost {}".format(i, e_new))
        error_list.append(e_new)
    print("The final result for theta is ", theta)
    return error_list, theta


lambda_list = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
M = 9

ERMS_list = []
ERMS_list_test = []

X = construct_polynomial(X_train, M)
testX = construct_polynomial(X_test, M)

for l in lambda_list:
    theta = np.zeros((M+1, 1))
    e, theta = newton_regularization(X, theta, y, l)
    # print("The theta for the regularization with lambda equal to {} is: {}".format(l, theta))
    e_test = cost_fuction(testX, theta, ytest, lambda_ = l)
    
    ERMS = np.sqrt(2 * e[-1] / X.shape[0])
    ERMS_list.append(ERMS)

    ERMS_test = np.sqrt(2 * e_test / testX.shape[0])
    ERMS_list_test.append(ERMS_test)

lambda_list[0] = 1e-9
plt.figure(figsize=(10,6))
train2, =  plt.plot(np.log10(lambda_list), ERMS_list, '--o', color='blue', label='Training')
test2, = plt.plot(np.log10(lambda_list), ERMS_list_test, '--o', color='red', label='Test')
plt.legend(handles=[train2, test2])
plt.xlabel("log_10 of lambda")
plt.ylabel("ERMS")
plt.show() 