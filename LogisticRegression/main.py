import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os
import plot


class LogisticRegression():

    ''' implementing logistic regression using a straight line and using sigmoid function as hypothesis function '''

    def __init__(self,learning_rate=0.01,error=1e-10,grad_descent=True):

        self.alpha = learning_rate
        self.error = error
        self.option = grad_descent
        ''' if set True, calculate cost using gradient descent function
                                        else, calculate cost using optimize function from scipy library'''



    def _hypothesisFunction(self,xdata):

        ''' initialise weights corresponding to the xdata '''

        n = xdata.shape[0]

        x  = np.stack([np.ones(n),xdata],axis=1) # add a columns of ones to Xdata to account for theta0

        weight = np.zeros(n+1)

    def _sigmoid(self,z):
        ''' calculate sigmoid function for value z'''

        return 1/(1+np.exp(-z))

    def fit(self,X,Y):


        n = X.ndim  # num of dimensions = num of weights

        x = np.stack([np.ones(np.size(X)//n),X],axis=1) # add a column of ones to X
        w = np.zeros(n+1)
        print(self.option)
        if self.option:
            return self._gradientDescent(w,x,Y)

        else:
            pass

    def _costfunction(self,weights,xdata,ydata):

        ''' compute cost for given weight for logistic regression'''
        m = ydata.shape[0]
        J = -1*np.sum(np.log(self._sigmoid(weights.dot(xdata.T)))*ydata + (1-y)*np.log(1-sigmoid(weights.dot(xdata.T))))/m

        return J

    def _gradientDescent(self,weights,xdata,ydata):

        ''' compute gradient descent algorithm for minimizing the cost function'''

        m = ydata.shape[0]
        weights = weights.copy()

        cost=[0]
        while True:

            # calculate the error in the chosen weights, mx + b - y
            temp = weights.dot(xdata.T)-ydata

            # xdata shape is (n,2), so take the transpose and dot it with temp of shpae (n,)
            temp1 = xdata.T.dot(temp) # temp1 is of shape (2,1)

            weights = weights - self.alpha*temp1/m

            cost.append(self._computeCost(weights,xdata,ydata))


            if abs(cost[-1]-cost[-2]) < self.error:
                return weights

    def predict(self,weights,X):
        ''' predict the decision boundary for classification
        for simple case, decision boundary is a straight line and works well for binary classifications'''

        n = Xvalues.ndim

        Xvalues = Xvalues.copy()

        Xvalues = np.stack([np.ones(np.size(Xvalues)//n),Xvalues],axis=1)

        return weights.dot(Xvalues.T)


if __name__ == '__main__':


    logreg = LogisticRegression()

    data = np.loadtxt(os.path.join("Data",'ex2data1.txt'),delimiter=',')

    X,y = data[:,0:2],data[:,2]
    weights = logreg.fit(X,y)
    plot.plotData(X,y)



