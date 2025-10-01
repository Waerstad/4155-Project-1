import numpy as np
from LinearRegression import _LinearRegression

class OLS(_LinearRegression):
    def __init__(self, gd_method = "analytic"):
        self.gd_method = gd_method
        _LinearRegression.__init__(self, model_type="OLS", gradient_descent_method = self.gd_method,
                                   _param_getter = self._param_getter)
    
    def _gradient_func(self, theta):
        """
        Computes the gradient at given theta for X and y given by
        self._features and self_targets, which are set by method 
        _LinearRegression.fit()
        """
        X = self._features
        y = self._targets
        return (2.0/self._num_points)*(X.T @ X @ theta - X.T @ y)
    
    def _analytic(self):
        """
        Computed analytic solution for theta for X and y given by
        self._features and self_targets. This method is called by
        _LinearRegression.fit() which sets self._features and
        self._targets
        """
        X = self._features
        y = self._targets
        y.shape = (self._num_points, 1)
        self.model_params = np.linalg.pinv(X.T @ X) @ X.T @ y
        return self.model_params

    def _param_getter(self, features, targets, **kwargs):
        """
        Get model parameters by the method name stored in the
        string self.gradient_descent_method.
        """
        if self.gradient_descent_method == "analytic":
            return self._analytic()
        else:
            gd_func = self._get_gd_method(self.gradient_descent_method)
            return gd_func(features, targets, **kwargs)
            

if __name__ == "__main__":
    def polynomial_features(x, p, intercept=False):
        """
        Take an array of x values, and the desired polynomial degree p.
        Create a feature (design) matrix with first column x**1, second column with x**2, and so on. I.e. the i-th column containing x**(i+1).
        Intercept=True will turn the first column into ones, meaning the i-th column will contain the value x**(j).
        """
        n = len(x)
        if intercept  == True:
            X = np.zeros((int(n), int(p + 1)))
            for i in range(0, int(p+1)):
                X[:, i] = x**i  # Create first column with only ones (since x**0 = 1), the intercept column
        else:
            X = np.zeros((int(n), int(p)))
            for i in range(0, int(p)):
                X[:, i] = x**(i+1)
        return X
    x = np.array([0.2,0.4,0.6,0.8,1])
    X = polynomial_features(x, 2)
    y = x + x**2 

    print("x", x)
    print("y", y)
    print("X", X)

    model = OLS()
    print("analytic")
    print(model.fit(X,y))

    model = OLS("simple")
    print("simple")
    print(model.fit(X,y, initial_theta=np.array([1.1,1.1]), learning_rate=1.01))

    model = OLS("momentum")
    print("momentum")
    print(model.fit(X,y, initial_theta=np.array([1.1,1.1]), learning_rate=1.01))

    model = OLS("adagrad")
    print("adagrad")
    print(model.fit(X,y, initial_theta=np.array([1.1,1.1]), learning_rate=1.01))

    model = OLS("RMSProp")
    print("RMSprop")
    print(model.fit(X,y, initial_theta=np.array([1.1,1.1]), learning_rate=1.01))

    model = OLS("adam")
    print("adam")
    print(model.fit(X,y, initial_theta=np.array([1.1,1.1]), learning_rate=1.01))

    model = OLS("simple_stochastic")
    print("simple_stochastic")
    print(model.fit(X,y, initial_theta=np.array([1.1,1.1]), learning_rate=0.25))
    
    
