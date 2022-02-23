import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os
import utils
import pdb

class LogisticRegression():

    ''' implementing logistic regression using a straight line and using sigmoid function as hypothesis function '''

    def __init__(self,learning_rate=1e-2,error=1e-10, order=2,grad_descent=True, regularization= True):

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
        self.regularization = regularization
        self.lambda_       = 10

    def _sigmoid(self,z):

        fun = 1/(1+np.exp(-z))

        return fun


    def _hypothesisFunction(self,weights):

        ''' return a function which is best for the classification:
        for example, for linear classification, function is a straight line,
                     for non-linear classification, function could be any curve containing higher orders'''

        func = weights.dot(self.x.T) # this is simply a straight line

        return func


    def fit(self,X,Y):
        ''' inputs:
        X : an array of shape (m,n)
        Y : a array of shape (m,)

        output:
        optimized weights for classification
        '''

        self.m = X.shape[0] # number of datas or rows
        self.n = X.shape[1] # number of dimensions or # of columns
        self.y       = Y

        # normalize each feature in x for better perfomance

        ''' keep the record of mean and variance to be passed when predicting new x not found in the training set'''
        self.mu =  np.mean(X,axis=0)  #mean
        self.sigma = np.std(X,axis=0) # standard deviation

        x_norm = (X-self.mu)/self.sigma

        self.x = utils.mapFeature(x_norm[:,0],x_norm[:,1],degree=self.order)

        self.weights = np.ones(self.x.shape[1])

        print(self.x)

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

    def _computeCost(self,weights):

        ''' compute cost for the given weight'''

        func =  -1*np.sum(np.log(self._sigmoid(self._hypothesisFunction(weights)))*self.y + (1-self.y)*np.log(1-self._sigmoid(self._hypothesisFunction(weights))))/self.m

        if self.regularization:
            return func + self.lambda_*np.sum(self.weights[1:]**2/(2*self.m))
        else:
            return func


    def _gradientDescent(self):


        costvalues = [0]
        while True:

            temp = self._sigmoid(self._hypothesisFunction(self.weights))-self.y        # calculate the error;  sigmoid(w.dot(x.T)) - y

            grad = self.x.T.dot(temp)/self.m

            ##condition for regularization

            if self.regularization:
                grad[1:] = grad[1:] + self.lambda_*self.weights[1:]/self.m

            ##update the weights

            self.weights = self.weights -self.alpha* grad


            costvalues.append(self._computeCost(self.weights))

            if abs(costvalues[-1]-costvalues[-2]) <= self.error:
                return self.weights

    def predict(self,weights,Xvalues):
        ''' predict new input using optimized weights'''

        m = Xvalues.shape[0]
        # find the new output
        x = (Xvalues - self.mu)/self.sigma


        xval = utils.mapFeature(x[:,0],x[:,1],degree=self.order)
        #print(xval.shape)
        yo = self._sigmoid(weights.dot(xval.T)) # predicted output

        threshold = 0.5

        yo[yo>=0.5] = 1
        yo[yo < 0.5] =0

        return yo



def main():
    logreg = LogisticRegression(2,grad_descent=True)


    data = np.loadtxt(os.path.join("Data",'ex2data2.txt'),delimiter=',')

    X,y = data[:,0:2],data[:,2]
    weights = logreg.fit(X,y)
    print(weights)

    n = X.shape[0]
    mu =  np.mean(X)
    sigma = np.std(X)
    x_norm = (X-mu)/sigma

    x1,x2 = min(x_norm[:,0]) , max(x_norm[:,1])


    x_norm  = utils.mapFeature(x_norm[:,0],x_norm[:,1])
    utils.plotDecisionBoundary(utils.plotData,weights,x_norm,y)

    utils.plt.show()
    ynew = logreg.predict(weights, X)

    correct_classification = 0
    count = 0

    for i in range(ynew.shape[0]):

        if ynew[i] == y[i]:
            correct_classification +=1
        count+=1
    print(f"correctly classified: {(correct_classification/count)*100}")

if __name__ == '__main__':
    main()
