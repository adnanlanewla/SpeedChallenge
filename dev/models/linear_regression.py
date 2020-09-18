import numpy as np
from sklearn.linear_model import LinearRegression


def fit_linear_regression(X,y):
    model = LinearRegression()
    model.fit(X,y)
    return model