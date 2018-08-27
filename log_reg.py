import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

#%matplotlib inline

import numpy as np
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# Loading the data (cat/non-cat)
train_set_x_orig, train_y, test_set_x_orig, test_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
#print ('m_train:'+str(m_train) + 'm_test:'+str(m_test)+'num_px:'+str(num_px))

train_set_x= train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x= test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
#print(train_set_x.shape)
#print(test_set_x.shape)

train_x=train_set_x/255
test_x=test_set_x/255

def sigmoid(z):
    s=1+np.exp(-1*z)
    s=1/s
    return s

#print ("sigmoid(0) = " + str(sigmoid(0)))
#print ("sigmoid(9.2) = " + str(sigmoid(9.2)))

def initialize_with_zeros(dim):
    w=np.zeros((dim,1))
    b=0
    return w,b

#dim = 2
#w, b = initialize_with_zeros(dim)
#print ("w = " + str(w))
#print ("b = " + str(b))    

def propagate(w,b,X,Y):
    Z=np.dot(w.T,X)+b
    A=sigmoid(Z)
    m=Y.shape[1]
    L=-1*(Y*np.log(A)+(1+-1*Y)*np.log(1+-1*A))
    J=np.sum(L)/m
    dZ=A-Y
    dw=np.dot(X,dZ.T)/m
    db=np.sum(dZ)/m
    grads={"dw":dw,"db":db}
    return grads,J

#w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
#grads, cost = propagate(w, b, X, Y)
#print ("dw = " + str(grads["dw"]))
#print ("db = " + str(grads["db"]))
#print ("cost = " + str(cost))

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    costs=np.zeros((num_iterations,1))
    for i in range(num_iterations):
        grads,cost=propagate(w,b,X,Y)
        dw=grads["dw"]
        db=grads["db"]
        costs[i]=cost  
        w=w-learning_rate*dw
        b=b-learning_rate*db 
        if print_cost==True:
            if i%100==0:
                print("Cost after iteration "+str(i)+" is "+str(costs[i]))
    params={"w":w,"b":b}
    grads={"dw":dw,"db":db}
    return params,grads,costs

#params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

#print ("w = " + str(params["w"]))
#print ("b = " + str(params["b"]))
#print ("dw = " + str(grads["dw"]))
#print ("db = " + str(grads["db"]))

def predict(w,b,X):
    #Y_prediction=np.zeros(X.shape[1])
    a=sigmoid(np.dot(w.T,X)+b)
    a[a>0.5]=1
    a[a<=0.5]=0   
    return a

#print("predictions = " + str(predict(w, b, X)))    
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w,b=initialize_with_zeros(X_train.shape[0])
    params,grads,costs=optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    Y_train_prediction=predict(params["w"],params["b"],X_train)
    Y_test_prediction=predict(params["w"],params["b"],X_test)
    d = {"costs": costs,
         "Y_prediction_test": Y_test_prediction, 
         "Y_prediction_train" : Y_train_prediction, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d

d = model(train_x, train_y, test_x, test_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
