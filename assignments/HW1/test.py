import numpy as np

a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
b = np.array([[1,2,3], [4,5,6], [7,8,9]])

c = np.array([[1], [1], [1]])
d = np.array([[0], [0], [0]])
print(np.sum((c - d) ** 2))


# def f(a,b):
#     c = []
#     for i in range(a.shape[0]):
#         c.append(np.argmin(np.linalg.norm( b - a[i,:], axis=1), axis = 0))
#     return np.array(c)

# print(f(a,b))


# x = np.array([1,2,3])
# tau = .8
# w = np.array([np.exp(- (x - x[i])**2/(2*tau)) for i in range(3)])   
# print(w)
