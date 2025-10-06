# 4155-Project-1
## Project 1 for FYS-STK 4155 at UiO.
By John-Magnus Johnsen & Simen Lund Wærstad

This project discusses the linear regression methods ordinary least squares (OLS), ridge, and LASSO. In addition, we discuss various gradient descent methods, including ordinary gradient descent, stochastic gradient descent, momuentum, ADAgrad, RMSprop, and Adam. We also discuss the bias-variance trade-off as well as resampling methods. This is all done in the context of a case study of polynomial fitting on the Runge function.

The required python packages are included in the file `requirements.txt`. To install these packages run the command
`pip install -r requirements.txt`

All the code used to generate data and create figures is included in the folder `Code/`. The python file `Code/functions.py` is a collection of some utility functions like `Polynomial_Features`. The runge function is included in `Code/runge_function.py`. The mini-batch function used in stochastic gradient descent is included in `Code/mini_batch.py`. The linear regression models are implemented as classes. There's more details about this in the section below.

The folder `Code/` also contains all jupyter-notebooks which were used to compute the data and plot all data used for tables and figures contained in the report. The jupyter-notebooks have file-names reflecting the data they compute. All instance of random functions have been seeded using NumPy's seeding functionality. 

### How to use the Classes `OLS`, `Ridge`, and `LASSO`.

The model types OLS, Ridge and LASSO, have all been implemented as separate classes: `OLS`, `Ridge`, and `LASSO`. Each class is implemented in their own file `OLS.py`, `Ridge.py`, and `LASSO.py`. The classes are initialized in the same way by specifying what method to use to compute the parameters (analytic solution or some allowed gradient descent method) and providing any other model-specific parameters. The default values are:
```python
OLS(gd_method="analytic")
Ridge(gd_method="analytic", llambda=0.01)
LASSO(gd_method="simple", llambda=0.01)
```
One then fits the data by using the `fit(features, targets, **kwargs)` method. In the case of `OLS` with gradient descent with momentum, where the momentum parameter is set to 0.01, we write
```python
model = OLS("momentum")
output = model.fit(X, y, momentum=0.01, max_iter = 1000)
```
Here output is the triple `(model_params, number_of_iterations, converged)`. Here `converged` is a boolean variable that is `True` if the gradient descent method converged before the number of iterations equaled or exceeded `max_iter` and `False` otherwise. The model parameters `model_params` are also written to the attribute `model.model_params`.

All gradient descent methods have a few standard parameters, all have standard values. These are:
- `initial_theta = np.zeros((self._num_features, 1))`
- `learning_rate = 0.01`
- `precision = 1e-8`, How close the model parameters $\boldsymbol{\theta}$ has to be to $\pmb{0}$ before the algorithm is considered to have converged.
- `max_iter = 10000`

In addition, some gradient descent methods have their own model specific parameters. The allowed gradient descent methods, their parameters and their standard values are: 

- `"simple"`: uses simple gradient descent. Has no additional parameters.
- `"momentum"`:
  - `momentum = 0.01`
- `"adagrad"`:
  - `num_stab_const = 1e-7`
- `"rmsprop"`:
  - `decay_rate = 0.999`
  - `num_stab_const = 1e-6`
- `"adam"`:
  - `first_moment = 0.9`
  - `second_moment = 0.999`
  - `num_stab_const = 1e-8`

Each gradient descent method also has a stochastic version which uses mini-batches to compute the gradient. These are chosen by appending `"_stochastic"` to the end of their name. All the stochastic versions have the additional parameter `mbatch_size = 20`, which sets the mini-batch size. For example, to use stochastic momentum with mini batch size 25 we write
```python
model = OLS("momentum_stochastic")
output = model.fit(X, y, momentum=0.01, mbatch_size=25)
```

The classes have the user-facing methods:
```python
fit(features, targets, **kwargs)
```
Updates and returns the attribute `model_params`. All `**kwargs` are passed to the gradient descent method.

```python
predict(features, intercept = 0)
```
Returns the resulting predictions from predicting on `features` using model parameters stored in the attribute `model_params` (i.e. `fit()` must be run or `model_params` must be manually updated before predicting.) An intercept can be included using the input variable `intercept`.

### Implementation details

The file `LinearRegression.py`contains the class `_LinearRegression` which is only intended as means to implement the classes OLS, Ridge, and LASSO, which are intended to be user-interfaces. The `_LinearRegression` class contains the user-facing attributes:

```python
model_type 
model_params
gradient_descent_method
```

The attribute `model_type` is merely a string containing the model type, e.g. `"OLS"`. The model parameters are stored in the attribute `model_params` which is initialized as `None` and updated by the method `fit()`. Finally, the attribute `gradient_descent_method` stores the method used for gradient descent as a string, e.g. `"momentum"`. If instead one wants to use the analytical solution (if it exists) one can set `gradient_descent_method` equal to `"analytic"`.

The `_LinearRegression` class also contains some user-facing methods, namely, `fit()` and `predict()`. These are inherited by the model specific classes.

All the three classes `OLS`, `Ridge` and `LASSO` are implemented in the same way. All three classes have the attribute `gd_method` which contains the name of the gradient descent method as a string and is passed onto the `_LinearRegression` attribute `_gradient_descent_method`. In addition, `Ridge` and `LASSO` have the attribute `llambda` which contains the lambda parameter.
They all contain the two methods
`_gradient_func()` and `_param_getter()`, which are not intended to be used by the user. The method `_gradient_func()` computes the gradient of the cost function of the model at a certain set of model parameters. The method `_param_getter()` is passed to `_LinearRegression` via the attribute `_LinearRegression._param_getter`. It is a helper function that is called by the method `_LinearRegression.fit()` and itself calls the appropriate gradient descent method (or analytic solution) and passes on the `**kwargs` received from `_LinearRegression.fit()`. The classes `OLS` and `Ridge`
also contains the method `_analytic` which computes and returns the analytic solution.

In addition, `OLS`, `Ridge`, and `LASSO` inherit all the methods and attributes of `_LinearRegression`. Thus implementing additional linear regression methods should reduce to changing the contents of the three methods unique to each regression method class and adding any additional model specific parameters as attributes.

All gradient descent algorithms are implemented as separate methods of the class `_LinearRegression`. If the gradient descent choice is not set to `"analytic"` the fitting method `_LinearRegression.fit()` calls `_LinearRegression._param_getter`. Which in turn calls the function `_get_gd_method()`. The function `_get_gd_method()` contains a dictionary of all the gradient descent methods were the keys are the names of the methods stored as strings as given in the list presented earlier. One called `_get_gd_method()` retrieves the appropriate gradient descent method which and returns it to `_LinearRegression._param_getter` which then computes the model parameters and returns them to `_LinearRegression.fit()`. 

Adding additional gradient descent methods is then done by implementing a new method `_LinearRegression._new_gradient_descent()` and adding this to the dictionary in `_LinearRegression._get_gd_method()` with the appropriate key string.


