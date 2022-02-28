import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os
import utils
import pdb

class LogisticRegression():

    ''' implementing logistic regression using a straight line and using sigmoid function as hypothesis function '''

    def __init__(self,learning_rate=1e-2,error=1e-10, order=6, lambda_=0,grad_descent=False,):

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


        '''

        self.alpha = learning_rate
        self.error = error
        self.option = grad_descent
        self.order = order

        self.lambda_       = lambda_

    def _sigmoid(self,z):

        fun = 1/(1+np.exp(-z))

        return fun

    def _computeCost(self,weights):

        ''' compute the cost for current weights '''

        J = -1*np.sum(np.log(self._sigmoid(weights.dot(self.x.T)))*self.y + (1-self.y)*np.log(1-self._sigmoid(weights.dot(self.x.T))))/self.m

        Jnew = J + self.lambda_*np.sum(weights[1:]**2)/(2*self.m)

        return Jnew

    def fit(self,X,Y):
        ''' inputs:
        X : an array of shape (m,n) , n >=2
        Y : a array of shape (m,)

        output:
        optimized weights for classification
        '''

        self.m = X.shape[0]

        self.x = utils.mapFeature(X[:,0],X[:,1],self.order)
        self.y = Y

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
        ''' predict the output of new input x values'''

        yo = self._sigmoid(weights[0]+weights[1]*x)

        yo[yo>=0.5] = 1
        yo[yo<0.5] = 0

        return yo



def main():

    data = np.loadtxt(os.path.join('Data','ex2data2.txt'),delimiter=',')

    X,Y  = data[:,0:2], data[:,2]

    logreg = LogisticRegression(order=1,lambda_=1)

    print(X.shape)
    w = logreg.fit(X,Y)

    print(w)

    # finding new X for order 6
    n = 1
    lambda_ = 1
    x = utils.mapFeature(X[:,0],X[:,1],n)
    utils.plotDecisionBoundary(utils.plotData,w,x,Y,n)
    utils.plt.xlim(-2,2)
    utils.plt.ylim(-2,2)
    utils.plt.show()

if __name__ == "__main__":
    main()
