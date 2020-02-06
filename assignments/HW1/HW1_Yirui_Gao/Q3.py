import numpy as np
import math
import matplotlib.pyplot as plt

# Load the data:
X = np.load('data/q3x.npy')
y = np.load('data/q3y.npy')

n = X.shape[0]
X = np.append(np.ones((n, 1)), np.reshape(X, (n, 1)), axis = 1)
print(X, X.shape)

########################## i ###########################
# Use normal equation to get the result for the weight:
theta = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)

print(theta)

plt.figure(figsize=(10,6))
plt.scatter(X[:,1], y, facecolors='none', edgecolors='blue')
plt.plot(X[:,1], X.dot(theta), color='green')
plt.xlabel("X")
plt.ylabel("y")
plt.show() 


########################## ii ###########################
bandwidth = 0.8
def LWLR(X, y, bandwidth):
    # Choose each point between the minimum and the maximum as with step 0.1 as the center points
    x_list = np.arange(min(X[:,1]), max(X[:,1]) + 1, 0.1)
    prediction_list = []
    for k in range(len(x_list)):
        R = np.zeros((n, n))
        for i in range(n):
            center = x_list[k]
            R[i,i] = math.exp(-(center - X[i,1])**2/(2*bandwidth**2))
        # Calculate the theta matrix and then get the predictions:
        theta = np.linalg.inv(X.T.dot(R).dot(X)).dot(X.T).dot(R).dot(y)
        prediction_list.append(theta[0] + theta[1]*x_list[k])
    plt.figure(figsize=(10,6))
    plt.scatter(X[:,1], y, facecolors='none', edgecolors='blue')
    plt.plot(x_list, prediction_list, color='green')
    plt.title("This is the plot with bandwidth equaling to {}".format(bandwidth))
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show() 
LWLR(X, y, 0.8)

########################## iii ###########################
bandwidth_list = [.1,.3,2,10]
for i in bandwidth_list:
    LWLR(X, y, i)