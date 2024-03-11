import torch
from scipy.optimize import minimize

class SCADRegression:
    '''
    A class for fitting and predicting data on a linear model with smoothly clipped absolute deviation (SCAD) penalty.
    '''
    def __init__(self, scad_a=2.0, scad_lambda=1.0):
        '''
        Args:
            scad_a: Controls how quickly the penalty drops off for large coefficient values
            scad_lambda: The level of penalization applied to the coefficients
        '''
        self.scad_a = scad_a
        self.scad_lambda = scad_lambda


    def fit(self, X, y):
        '''
        Fit data on a linear model with SCAD penalty.

        Args:
            X: Training data
            y: Target values
        '''
        device = torch.device('cpu')
        dtype = torch.double

        # Get number of features
        if len(X.shape) == 1: # If the training data only has 1 feature
            n_features = 1
        else: # Training data has more than 1 feature
            n_features = X.shape[1]

        betas = torch.zeros(n_features+1, device=device, dtype=dtype) # Beta coefficients for each feature in X plus the intercept term
        result = minimize(self.__objective, betas, args=(X, y))
        self.coefs_ = torch.tensor(result.x, dtype=dtype, device=device) # The first element is the intercept term
        return

    
    def predict(self, X):
        '''
        Predict using the fitted linear model with SCAD penalty.

        Args:
            X: Sample data
        '''
        if self.is_fitted():
            raise Exception("This SCADRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        if len(X.shape) == 1: # Sample data has 1 feature
            return X*self.coefs_[1] + self.coefs_[0]
        else: # More than 1 feature
            return X.matmul(self.coefs_[1:]) + self.coefs_[0]
        

    def is_fitted(self):
        if hasattr(self, 'coef_'):
            return True
        else:
            return False
        

    def __objective(self, beta, X, y):
        '''Objective function'''
        beta = torch.tensor(beta) # Convert NumPy array to PyTorch tensor

        n = len(y) # Number of observations
        if len(X.shape) == 1: # Training data has 1 feature
            error = y - (X*beta[1] + beta[0])
        else: # More than 1 feature
            # Convert arrays to tensors            
            error = y - (X.matmul(beta[1:]) + beta[0])
        mse = torch.sum(error**2) / n # Mean squared error
        penalty = self.__scad_penalty(beta[1:])
        return mse + penalty
    

    def __scad_penalty(self, beta_hat):
        '''Get total SCAD penalty value'''
        abs_beta = torch.abs(beta_hat[1:])
        
        is_linear = (abs_beta <= self.scad_lambda)
        is_quadratic = torch.logical_and(self.scad_lambda < abs_beta, abs_beta <= self.scad_a * self.scad_lambda)
        is_constant = (self.scad_a * self.scad_lambda) < abs_beta
        
        linear_part = self.scad_lambda * abs_beta * is_linear
        quadratic_part = (2 * self.scad_a * self.scad_lambda * abs_beta - abs_beta**2 - self.scad_lambda**2) / (2 * (self.scad_a - 1)) * is_quadratic
        constant_part = (self.scad_lambda**2 * (self.scad_a + 1)) / 2 * is_constant

        penalties = linear_part + quadratic_part + constant_part
        return torch.sum(penalties, dtype=torch.double)
    