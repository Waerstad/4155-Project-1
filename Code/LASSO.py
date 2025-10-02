import numpy as np
from _LinearRegression import _LinearRegression

class LASSO(_LinearRegression):
    def __init__(self, gd_method = "simple", llambda = 0.01,):
        self.gd_method = gd_method
        self.llambda = llambda
        _LinearRegression.__init__(self, model_type="LASSO", gradient_descent_method = self.gd_method,
                                   _param_getter = self._param_getter)
    
    def _gradient_func(self, theta):
        """
        Computes the gradient at given theta for X and y given by
        self._features and self_targets, which are set by method 
        _LinearRegression.fit()
        """
        X = self._features
        y = self._targets
        return (2.0/self._num_points)*(X.T @ X @ theta - X.T @ y) + self.llambda*np.sign(theta)

    def _param_getter(self, features, targets, **kwargs):
        """
        Get model parameters by the method name stored in the
        string self.gradient_descent_method.
        """
        if self.gradient_descent_method == "analytic":
            raise ValueError("LASSO has no analytic solution")
        else:
            # get the user called for gradient descent function
            gd_func = self._get_gd_method(self.gradient_descent_method)
            return gd_func(features, targets, **kwargs)
            