import numpy as np

class LSE(object):
    """
    Least Squares Error Linear Regression.
    # Attributes
    ----------
    coef_ : array, shape (1, )
        Estimated coefficient for the linear regression problem.
    intercept_ : float
        Independent term in the linear model.

    # Examples
    --------
    >>> import numpy as np
    >>> from mlitera.linreg import LSE
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> # y = 2x + 3
    >>> y = (2 * X + 3).flatten()
    >>> m = LSE().fit(X, y)
    >>> m.score(y, m.predict(X))
    1.0
    >>> m.intercept_
    3.0
    >>> m.coef_
    2.0
    """
    
    def fit(self, X, y):
        """
        Fit linear model.
        Parameters
        ----------
        X : array-like, shape (n_samples, 1)
            Training data
        y : array_like, shape (n_samples, )
            Target values.
        Returns
        -------
        self : returns an instance of self.
        """
        X = X.flatten() # unrolled X
        N = len(X)
        theta1 = ( sum(X * y) - (N * X.mean() * y.mean()) ) / sum((X - X.mean()) ** 2)
        theta0 = y.mean() - theta1 * X.mean()
        self.thetas = np.array([theta0, theta1])
        return self
    
    @property
    def intercept_(self):
        """ Independent term in the linear model."""
        return self.thetas[0]
    
    @property
    def coef_(self):
        """ Estimated coefficient for the linear regression problem."""
        return self.thetas[-1]
    
    def score(self, y_true, y_pred):
        """Returns the coefficient of determination R^2 of the prediction.
        The coefficient R^2 is defined as (1 - SSR/SST), where SSR is the residual
        sum of squares sum((y_true - y_pred) ** 2) and SST is the total
        sum of squares sum((y_true - y_true.mean()) ** 2).
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.
        Parameters
        ----------
        y_true : array-like, shape = (n_samples)
            True values for X.
        y_pred : array-like, shape = (n_samples)
            Prediction values for X.
        Returns
        -------
        score : float
            R^2
        """
        SST = sum((y_true - y_true.mean()) ** 2)
        SSR = sum((y_true - y_pred) ** 2)
        return 1 - SSR/ SST
    
    def predict(self, X):
        """Predict using the linear model
        Parameters
        ----------
        X : array_like , shape (n_samples, 1)
            Samples.
        Returns
        -------
        y : array, shape (n_samples,)
            Returns predicted values.
        """
        X = X.flatten() # unrolled X
        y = self.intercept_ + self.coef_ * X
        return y
        

if __name__ == '__main__':
    import sys
    import doctest
    sys.path.append("/home/jose/machine-learning-itera")
    doctest.testmod()