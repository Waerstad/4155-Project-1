import numpy as np
from _LinearRegression import _LinearRegression

class LASSO(_LinearRegression):
    def __init__(self, gd_method = "simple", llambda = 0.01,):
        self.gd_method = gd_method
        self.llambda = llambda
        _LinearRegression.__init__(self, model_type="LASSO", gradient_descent_method = self.gd_method,
                                   _param_getter = self._param_getter)
    
    def _gradient_func(self, X, y, theta):
        """
        Computes the gradient of the cost function at theta for feature
        matrix X and target vector y.
        """
        return (2.0/self._num_points)*(X.T @ X @ theta - X.T @ y) + self.llambda*np.sign(theta)

    def _gradient_func_precomp(self, theta):
        """
        Computes the gradient of the cost function at theta for feature
        matrix X and target vector y. When X and y do not change we can
        use the precomputed products X.T @ X and X.T @ y
        """
        XTX = self.XTX
        XTy = self.XTy
        return (2.0/self._num_points)*(XTX @ theta - XTy) + self.llambda*np.sign(theta)

    def _param_getter(self, features, targets, **kwargs):
        """
        Get model parameters by the method name stored in the
        string self.gradient_descent_method.
        """
        if self.gradient_descent_method == "analytic":
            raise ValueError("LASSO has no analytic solution")
        else:
            # precompute:
            self.XTX = self._features.T @ self._features
            self.XTy = self._features.T @ self._targets
            # get the user-requested gradient descent function
            gd_func = self._get_gd_method(self.gradient_descent_method)
            return gd_func(features, targets, **kwargs)
            