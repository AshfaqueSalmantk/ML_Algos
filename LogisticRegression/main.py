import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os
import plot
import pdb

class LogisticRegression():

    ''' implementing logistic regression using a straight line and using sigmoid function as hypothesis function '''

    def __init__(self,learning_rate=2e-4,error=1e-6,grad_descent=True):

        self.alpha = learning_rate
        self.error = error
        self.option = grad_descent
        ''' if set True, calculate cost using gradient descent function
        else, calculate cost using optimize function from scipy library
        where we can choose which algorithm to use to minimize cost'''



    def _hypothesisFunction(self,xdata):

        ''' initialise weights corresponding to the xdata '''

        n = xdata.shape[0]

        x  = np.hstack([np.ones(n,1),xdata]) # add a columns of ones to Xdata to account for theta0

        weight = np.zeros(n+1)

    def _sigmoid(self,z):
        ''' calculate sigmoid function for value z'''

        return 1/(1+np.exp(-z))

    def fit(self,X,Y):

        n = X.ndim  # num of dimensions = num of weights
        m = X.shape[0] # num of datas


        if n ==1 :
            ''' if n ==1, then xdata is one dimensional and will throw and exception when using np.concatenate.
                So reshape as two dimensional matrix'''
            X = X.reshape((m,1))

        # np.stack does not work when attaching onto existing axis
        # np.concatenate or np.hstack works well, but when passing shape, we must specify 2D shape for np.ones
        x = np.concatenate([np.ones((m,1)),X],axis=1)
        w = np.zeros(n+1)

        print(self.option)
        if self.option:
            return self._gradientDescent(w,x,Y)

        else:
            # set options for optimize.minimize
            options= {'maxiter': 400,'disp':True}

            # see documention for scipy's optimize.minimize  for description about
            # the different parameters
            # The function returns an object `OptimizeResult`
            # We use truncated Newton algorithm for optimization which is
            # equivalent to MATLAB's fminunc
            # See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
            res = optimize.minimize(self._computeCost,
                        w,
                        (x, y),
                        jac=False,
                        method='TNC',
                        options=options)

            # the fun property of `OptimizeResult` object returns
            # the value of costFunction at optimized theta
            cost = res.fun

            # the optimized theta is in the x property
            theta = res.x

            return theta

    def _computeCost(self,weights,xdata,ydata):

        ''' compute cost for given weight for logistic regression'''
        m = ydata.shape[0]
        J = -1*np.sum(np.log(self._sigmoid(weights.dot(xdata.T)))*ydata + (1-y)*np.log(1-self._sigmoid(weights.dot(xdata.T))))/m

        return J

    def _gradientDescent(self,weights,xdata,ydata):

        #pdb.set_trace()

        ''' compute gradient descent algorithm for minimizing the cost function'''

        m = ydata.shape[0]
        weights = weights.copy()

        cost=[0]
        #num_iter = 0
        while True:

            # calculate the error in the chosen weights, mx + b - y
            temp = weights.dot(xdata.T)-ydata

            # xdata shape is (n,2), so take the transpose and dot it with temp of shpae (n,)
            temp1 = xdata.T.dot(temp) # temp1 is of shape (2,1)

            weights = weights - self.alpha*temp1/m

            cost.append(self._computeCost(weights,xdata,ydata))


            if abs(cost[-1]-cost[-2]) < self.error:
                return weights
            #if num_iter == 1500:
                #return weights

            #num_iter +=1

    def predict(self,weights,Xvalues):
        ''' predict the decision boundary for classification
        for simple case, decision boundary is a straight line and works well for binary classifications'''

        n = Xvalues.shape[0]

        Xvalues = Xvalues.copy()

        Xvalues = np.concatenate([np.ones((n,1)),X],axis=1)

        return weights.dot(Xvalues.T)


if __name__ == '__main__':


    logreg = LogisticRegression(grad_descent=True)

    data = np.loadtxt(os.path.join("Data",'ex2data1.txt'),delimiter=',')

    X,y = data[:,0:2],data[:,2]
    weights = logreg.fit(X,y)
    print(weights)
    n = X.shape[0]
    x = np.concatenate([np.ones((n,1)),X],axis=1)
    plot.plotDecisionBoundary(plot.plotData,weights,x,y)
    plot.plt.show()
