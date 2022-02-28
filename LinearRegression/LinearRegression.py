#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import os
import utils
from scipy import optimize

'''LinearRegression using gradient descent algorithm for # of variables n>=1'''


class LinearRegression():

    def __init__(self,learning_rate=0.01,error=1e-10, order=1, grad_descent=False):

        '''
        arguments:
        ---------
        learning_rate = learning rate for gradient descent algorithm
        error         = stoping criteria for gradient descent algorithm
        grad_descent  = True ,  when set True: minimize cost using gradient descent algorithm
                        else, minimize cost using scipy.optimize.minimze function with chosen solver, Truncated newton method is used
                        in this code

        order         = order of the hypothesis function, 1 indicates straight line, linear
                        order >1 non-linear  order =2 quadratic, order =3 : cubic



        regularization = Boolean, True to implement regularized logistic regression to avoid overfitting.
        '''

        self.alpha = learning_rate
        self.error = error
        self.option = grad_descent
        self.order = order



    def _normalize(self,x):
        ''' normalize the input values for better convergence'''

        self.mu = np.mean(x,axis=0)
        self.sigma = np.std(x,axis=0)

        x_norm = (x - self.mu)/self.sigma

        return x_norm


    def _computeCost(self,weights):



        J = np.sum((weights.dot(self.x.T)-self.y)**2)/(2*self.m)

        return J

    def _gradientDescent(self):

        costvals = [0] # keep record of the cost for each weights

        while True:

            temp = self.weights.dot(self.x.T)-self.y  # find the error mx+c-y

            grad = self.x.T.dot(temp)

            #update the weights
            self.weights = self.weights - self.alpha*grad/self.m

            # append the curresponding cost into the costvals list
            costvals.append(self._computeCost(self.weights))

            #check for the stopping condition
            if abs(costvals[-1] - costvals[-2])<self.error:
                return self.weights

    def fit(self,X,Y):

        '''
        arguments:
        ---------

        X: a (m,n) vector input data
        Y: a (m, ) vector input data output

        output:
        ------
        W: a (n+1,) vector
        '''

        self.m = X.shape[0]
        self.n = X.ndim
        self.y = Y
        if self.n > 1 or self.order>1:
            self.xnorm = self._normalize(self.X)
            self.x = utils.mapFeature(self.xnorm[:,0],xnorm[:,1],self.order)
        elif self.n==1 and self.order > 1:
            ''' model according to the data, for example: w0 + w1x1 + w2x1^2 could be a model or x1^0.5 could be another '''
            pass
        else:

            self.X = X.reshape((self.m,1))
            self.xnorm = self._normalize(self.X)
            #self.xnorm = self.X
            self.x = np.concatenate([np.ones((self.m,1)),self.xnorm],axis=1)
            print(self.x[:10,:])
        self.weights = np.zeros(self.x.shape[1])

        if self.option:

            return self._gradientDescent()


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
                        self.weights,
                        jac=False,
                        method='TNC',
                        options=options)

            # the fun property of `OptimizeResult` object returns
            # the value of costFunction at optimized theta
            cost = res.fun

            # the optimized theta is in the x property
            theta = res.x

            return theta


    def predict(self,weights,X):
        '''
        arguments:
        ---------
        X : a (m,n) vector

        output:
        ------
        Y : a (m,) vector

        '''
        xnorm = self._normalize(X)
        #xnorm = X

        return weights[0] + weights[1]*xnorm


def main():

    linreg = LinearRegression(grad_descent=True)

    data =  np.loadtxt(os.path.join('Data','ex1data1.txt'),delimiter=',')

    if data.ndim == 2 :
        X,Y = data[:,0],data[:,1]
    else:
        X,Y = data[:,:data.ndim],data[:,data.ndim]

    weights=linreg.fit(X,Y)

    y = linreg.predict(weights,X)

    #print(y.shape,X.shape)
    utils.plotData(X, Y)
    utils.plt.plot(X,y,'k-',label=f'{weights[0]:.2f}+{weights[1]:.2f}x')
    utils.plt.legend()
    utils.plt.show()

if __name__ == "__main__":
    main()







