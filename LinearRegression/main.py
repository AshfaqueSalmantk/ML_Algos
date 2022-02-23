#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import os

'''LinearRegression using gradient descent algorithm for # of variables n>=1'''


class LinearRegression():

    def __init__(self,learning_rate=0.01,error=1e-10):

        self.alpha= learning_rate
        self.error = error


    def fit(self,X,Y):

        '''
        arguments:

        X: vector containing variables
                eg: if there is only one variable; X = [x1,x2,x3,... ], one dimensional vector

        Y: real output


        output:

        W= vector containing updated weights


        '''
        n = X.ndim  # num of dimensions = num of weights
        m = X.shape[0] # num of datas

        print(n,X.shape,m)
        if n ==1 :
            ''' if n ==1, then xdata is one dimensional and will throw and exception when using np.concatenate.
                So reshape as two dimensional matrix'''
            X = X.reshape((m,1))
        print(X.shape)
        # np.stack does not work when attaching onto existing axis
        # np.concatenate or np.hstack works well, but when passing shape, we must specify 2D shape for np.ones
        x = np.concatenate([np.ones((m,n)),X],axis=1)
        w = np.zeros(n+1)

        return self._gradientDescent(w,x,Y)

    def _computeCost(self,weights,xdata,ydata):
        '''
        '''

        m = ydata.shape[0]
        weights = weights.copy() # for safety

        J = np.sum( ( weights.dot(xdata.T)-ydata)**2)/(2*m)

        return J
    def _gradientDescent(self,weights,xdata,ydata):
        '''

        '''

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


    def predict(self,weights,Xvalues):
        '''
        predicting the new Yvalues using the optimized weights

        weights: new optimized weights getting from fit method
        Xvalues: data to calculate

        ouput:
        yvalues = mx + c  where m,c are in weights
        '''
        n = Xvalues.ndim

        Xvalues = Xvalues.copy()

        Xvalues = np.stack([np.ones(np.size(Xvalues)//n),Xvalues],axis=1)

        return weights.dot(Xvalues.T)

def plotData(x, y):
    """
    Plots the data points x and y into a new figure. Plots the data
    points and gives the figure axes labels of population and profit.

    Parameters
    ----------
    x : array_like
        Data point values for x-axis.

    y : array_like
        Data point values for y-axis. Note x and y should have the same size.

    Instructions
    ------------
    Plot the training data into a figure using the "figure" and "plot"
    functions. Set the axes labels using the "xlabel" and "ylabel" functions.
    Assume the population and revenue data have been passed in as the x
    and y arguments of this function.

    Hint
    ----
    You can use the 'ro' option with plot to have the markers
    appear as red circles. Furthermore, you can make the markers larger by
    using plot(..., 'ro', ms=10), where `ms` refers to marker size. You
    can also set the marker edge color using the `mec` property.
    """
    fig = plt.figure()  # open a new figure

    plt.plot(x, y, 'ro', ms=10, mec='k')
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')



if __name__=="__main__":

   linreg = LinearRegression()

   data = np.loadtxt(os.path.join("Data",'ex1data1.txt'),delimiter=',')

   X,y = data[:,0],data[:,1]
   weights = linreg.fit(data[:,0],data[:,1])
   plotData(X,y)

   plt.plot(X, linreg.predict(weights,X),'-')
   plt.show()


