import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler
# from feature_eng import PCAperformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def build_mlr(pca=False, n_comp=None):
    if pca == True:
        if n_comp is not None:
            pipeline = Pipeline([
                ('rescale', StandardScaler()),
                ('pca', PCA(n_components=n_comp)),
                ('mlr', LinearRegression(copy_X=True, fit_intercept=True))
            ])
    else:
        pipeline = Pipeline([
            ('rescale', StandardScaler()),
            ('mlr', LinearRegression(copy_X=True, fit_intercept=True))
        ])
        
    return pipeline


def build_ridge(pca=False, n_comp=None):
    parameters = {
        'ridge__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1e0, 2, 5, 10, 20, 50, 100]
        }

    if pca == True:
        if n_comp is not None:
            pipeline = Pipeline([
                ('rescale', StandardScaler()),
                ('pca', PCA(n_components=n_comp)),
                ('ridge', Ridge())
            ])

            cv = GridSearchCV(pipeline, param_grid=parameters, cv=KFold(5, shuffle=True, random_state = 149))
            
    else:
        pipeline = Pipeline([
            ('rescale', StandardScaler()),
            ('ridge', Ridge())
        ])

        cv = GridSearchCV(pipeline, param_grid=parameters, cv=KFold(5, shuffle=True, random_state = 149))
        
    return cv


def build_lasso(pca=False, n_comp=None):
    parameters = {
        'lasso__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1e0, 2, 5, 10, 20, 50, 100]
        }

    if pca == True:
        if n_comp is not None:
            pipeline = Pipeline([
                ('rescale', StandardScaler()),
                ('pca', PCA(n_components=n_comp)),
                ('lasso', Lasso(max_iter=50000))
            ])

            cv = GridSearchCV(pipeline, param_grid=parameters, cv=KFold(5, shuffle=True, random_state = 149))
            
    else:
        pipeline = Pipeline([
            ('rescale', StandardScaler()),
            ('lasso', Lasso(max_iter=50000))
        ])

        cv = GridSearchCV(pipeline, param_grid=parameters, cv=KFold(5, shuffle=True, random_state = 149))
        
    return cv


def build_linear_krr(pca=False, n_comp=None):
    parameters = {
        'krr__kernel':['linear'],
        'krr__alpha':[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1e0, 2, 5, 10, 20, 50, 100],
        }
    
    if pca == True:
        if n_comp is not None:
            pipeline = Pipeline([
                ('rescale', StandardScaler()),
                ('pca', PCA(n_components=n_comp)),
                ('krr', KernelRidge())
            ])

            cv = GridSearchCV(pipeline, param_grid=parameters, cv=KFold(5, shuffle=True, random_state = 149))
            
    else:
        pipeline = Pipeline([
            ('rescale', StandardScaler()),
            ('krr', KernelRidge())
        ])

        cv = GridSearchCV(pipeline, param_grid=parameters, cv=KFold(5, shuffle=True, random_state = 149))
        
    return cv


def build_nonlinear_krr(pca=False, n_comp=None):
    parameters = {
        'krr__kernel':['polynomial','rbf'],
        'krr__alpha':[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1e0, 2, 5, 10, 20, 50, 100],
        'krr__degree':[2,3,4,5],
        'krr__gamma': np.logspace(-2, 2, 5)
        }

    if pca == True:
        if n_comp is not None:
            pipeline = Pipeline([
                ('rescale', StandardScaler()),
                ('pca', PCA(n_components=n_comp)),
                ('krr', KernelRidge())
            ])

            cv = GridSearchCV(pipeline, param_grid=parameters, cv=KFold(5, shuffle=True, random_state = 149))
            
    else:
        pipeline = Pipeline([
            ('rescale', StandardScaler()),
            ('krr', KernelRidge())
        ])

        cv = GridSearchCV(pipeline, param_grid=parameters, cv=KFold(5, shuffle=True, random_state = 149))
        
    return cv


def build_linear_svr(pca=False, n_comp=None):
    parameters = {
        # 'svr__kernel':['linear'],
        'svr__C': [1e-1, 1e0, 1e1, 1e3],
        'svr__epsilon': [0, 1e-3, 1e-2, 1e-1, 1e0]
        # 'svr__dual': ['auto']
        }

    if pca == True:
        if n_comp is not None:
            pipeline = Pipeline([
                ('rescale', StandardScaler()),
                ('pca', PCA(n_components=n_comp)),
                ('svr', LinearSVR())
            ])

            cv = GridSearchCV(pipeline, param_grid=parameters, cv=KFold(5, shuffle=True, random_state = 149))
            
    else:
        pipeline = Pipeline([
            ('rescale', StandardScaler()),
            ('svr', LinearSVR())
        ])

        cv = GridSearchCV(pipeline, param_grid=parameters, cv=KFold(5, shuffle=True, random_state = 149))
        
    return cv


def build_nonlinear_svr(pca=False, n_comp=None):
    parameters = {
        'svr__kernel':['poly','rbf'],
        'svr__degree':[2,3,4,5],
        'svr__C': [1e-1, 1e0, 1e1, 1e3],
        'svr__gamma': ['scale','auto'],
        'svr__epsilon': [0, 1e-3, 1e-2, 1e-1, 1e0]
        }

    if pca == True:
        if n_comp is not None:
            pipeline = Pipeline([
                ('rescale', StandardScaler()),
                ('pca', PCA(n_components=n_comp)),
                ('svr', SVR())
            ])

            cv = GridSearchCV(pipeline, param_grid=parameters, cv=KFold(5, shuffle=True, random_state = 149))
            
    else:
        pipeline = Pipeline([
            ('rescale', StandardScaler()),
            ('svr', SVR())
        ])

        cv = GridSearchCV(pipeline, param_grid=parameters, cv=KFold(5, shuffle=True, random_state = 149))
        
    return cv


def build_rfr(pca=False, n_comp=None):
    parameters = {
        'rfr__max_depth': [10, 12, 15],
        'rfr__min_samples_leaf': [2, 3, 4],
        'rfr__min_samples_split': [4, 6, 8],
        'rfr__n_estimators': [200]
        }
                    
    if pca == True:
        if n_comp is not None:
            pipeline = Pipeline([
                ('rescale', StandardScaler()),
                ('pca', PCA(n_components=n_comp)),
                ('rfr', RandomForestRegressor(random_state=149))
            ])

            cv = GridSearchCV(pipeline, param_grid=parameters, cv=KFold(5, shuffle=True, random_state = 149))
            
    else:
        pipeline = Pipeline([
            ('rescale', StandardScaler()),
            ('rfr', RandomForestRegressor(random_state=149))
        ])

        cv = GridSearchCV(pipeline, param_grid=parameters, cv=KFold(5, shuffle=True, random_state = 149))
        
    return cv


def build_xgb(pca=False, n_comp=None):
    parameters = {
        "xgb__n_estimators": [200],
        "xgb__learning_rate": [0.001, 0.01, 0.1],
        "xgb__max_depth": [3, 5, 7],
        "xgb__min_child_weight": [3, 5, 7],
        "xgb__colsample_bytree": [0.5, 1],
        "xgb__subsample": [0.5, 1]
        }
                    
    if pca == True:
        if n_comp is not None:
            pipeline = Pipeline([
                ('rescale', StandardScaler()),
                ('pca', PCA(n_components=n_comp)),
                ('xgb', XGBRegressor(random_state=149))
            ])

            cv = GridSearchCV(pipeline, param_grid=parameters, cv=KFold(5, shuffle=True, random_state = 149))
            
    else:
        pipeline = Pipeline([
            ('rescale', StandardScaler()),
            ('xgb', XGBRegressor(random_state=149))
        ])

        cv = GridSearchCV(pipeline, param_grid=parameters, cv=KFold(5, shuffle=True, random_state = 149))
        
    return cv


# error calculation
def error_cal(model, X, y):
    pred = model.predict(X)
    error = abs(pred - y)
    return error
