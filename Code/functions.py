import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def Polynomial_Features(x, p, intercept=False):
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

def OLS_parameters(X, y):
    """
    The closed form solution of least square optimization.
    Take a feature matrix X, and a column vector y.
    Return the parameters of OLS fit.
    """
    # The inverse of matrix X, using .pinv in case X is not square
    # The beta vector in: X @ beta = y, solved for beta
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y   
    return beta

def Ridge_parameters(X, y, ridge_lambda):
    """
    The closed form solution of Ridge optimization.
    Take a feature matrix X, and a column vector y, and the Ridge lambda parameter.
    Return the parameters of the Ridge fit.
    """
    # Assumes X is scaled and has no intercept column
    I = np.identity(np.shape(X)[1])    # Create identity matrix same shape as X.T @ X, columns of X decide the shape, (nxm)(mxn)=(nxn)
    # Element-wise multiplication with * 
    return np.linalg.pinv(X.T @ X + ridge_lambda*I) @ X.T @ y

def Find_MSE_Ridge_predict_poly_ridgelambda(x_data, y_data, poly_degree, ridge_lambda):
    """
    Calculate the MSE
    """
    x = x_data
    y = y_data
    poly_degree = int(poly_degree)
    
    X = Polynomial_Features(x, poly_degree)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    scaler.fit(X_train)                     # Computes the mean and std to be used for later scaling
    X_train_s = scaler.transform(X_train)   # Using scaler function on the prepared training set from X
    X_test_s = scaler.transform(X_test)     # Using scaler function on the prepared test set from X
    y_offset = np.mean(y_train)             # Mean of y values in the training set for y

    beta_ridge = Ridge_parameters(X_train_s, y_train, ridge_lambda)
    predict_y_test = X_test_s @ beta_ridge + y_offset
    MSE_Ridge_model_lambdas = mean_squared_error(y_test, predict_y_test)

    return MSE_Ridge_model_lambdas